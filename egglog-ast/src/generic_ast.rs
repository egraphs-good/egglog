use std::fmt::Display;
use std::hash::Hash;

use ordered_float::OrderedFloat;

use crate::span::Span;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub enum Literal {
    Int(i64),
    Float(OrderedFloat<f64>),
    String(String),
    Bool(bool),
    Unit,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GenericExpr<Head, Leaf> {
    Var(Span, Leaf),
    Call(Span, Head, Vec<GenericExpr<Head, Leaf>>),
    Lit(Span, Literal),
}

/// Facts are the left-hand side of a [`Command::Rule`].
/// They represent a part of a database query.
/// Facts can be expressions or equality constraints between expressions.
///
/// Note that primitives such as  `!=` are partial.
/// When two things are equal, it returns nothing and the query does not match.
/// For example, the following egglog code runs:
/// ```text
/// (fail (check (!= 1 1)))
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GenericFact<Head, Leaf> {
    Eq(Span, GenericExpr<Head, Leaf>, GenericExpr<Head, Leaf>),
    Fact(GenericExpr<Head, Leaf>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GenericActions<Head: Clone + Display, Leaf: Clone + PartialEq + Eq + Display + Hash>(
    pub Vec<GenericAction<Head, Leaf>>,
);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GenericAction<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    /// Bind a variable to a particular datatype or primitive.
    /// At the top level (in a [`Command::Action`]), this defines a global variable.
    /// In a [`Command::Rule`], this defines a local variable in the actions.
    Let(Span, Leaf, GenericExpr<Head, Leaf>),
    /// `set` a function to a particular result.
    /// `set` should not be used on datatypes-
    /// instead, use `union`.
    Set(
        Span,
        Head,
        Vec<GenericExpr<Head, Leaf>>,
        GenericExpr<Head, Leaf>,
    ),
    /// Delete or subsume (mark as hidden from future rewrites and unextractable) an entry from a function.
    Change(Span, Change, Head, Vec<GenericExpr<Head, Leaf>>),
    /// `union` two datatypes, making them equal
    /// in the implicit, global equality relation
    /// of egglog.
    /// All rules match modulo this equality relation.
    ///
    /// Example:
    /// ```text
    /// (datatype Math (Num i64))
    /// (union (Num 1) (Num 2)); Define that Num 1 and Num 2 are equivalent
    /// (extract (Num 1)); Extracts Num 1
    /// (extract (Num 2)); Extracts Num 1
    /// ```
    Union(Span, GenericExpr<Head, Leaf>, GenericExpr<Head, Leaf>),
    Panic(Span, String),
    Expr(Span, GenericExpr<Head, Leaf>),
}

/// How a rule is evaluated, selected by mutually exclusive rule options.
///
/// The default ([`Seminaive`](RuleEvalMode::Seminaive)) and the two opt-in
/// options ([`:naive`](RuleEvalMode::Naive) and
/// [`:unsafe-seminaive`](RuleEvalMode::UnsafeSeminaive)) are mutually
/// exclusive, so they share a single field on [`GenericRule`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RuleEvalMode {
    /// The default: seminaive (delta) evaluation. The body is matched only
    /// against rows that are new this iteration, and the query/action are
    /// compiled with the restrictive `Pure`/`Write` primitive contexts (no
    /// database reads in the RHS).
    #[default]
    Seminaive,
    /// The `:naive` rule option disables seminaive evaluation. The body is
    /// matched against the entire database every iteration and the
    /// query/action are compiled with the permissive `Read`/`Full` primitive
    /// contexts, allowing primitives that read or write the database inside
    /// queries and actions.
    Naive,
    /// The `:unsafe-seminaive` rule option keeps seminaive (delta)
    /// evaluation but compiles the query/action with the permissive
    /// `Read`/`Full` primitive contexts (like `:naive`), and the
    /// typechecker's "no function lookups in actions" check is skipped. This
    /// lets the RHS perform arbitrary database reads — read-primitives and
    /// function-table lookups — without paying for `:naive`'s whole-database
    /// matching.
    ///
    /// It is **unsafe**: a read on a seminaive rule's RHS observes the
    /// database mid-iteration, so it won't be re-evaluated if the data
    /// changes. The caller takes responsibility.
    UnsafeSeminaive,
}

impl RuleEvalMode {
    /// Whether this rule disables seminaive (delta) evaluation, i.e. it is
    /// `:naive`. Both [`Seminaive`](RuleEvalMode::Seminaive) and
    /// [`UnsafeSeminaive`](RuleEvalMode::UnsafeSeminaive) evaluate seminaively.
    pub fn is_naive(self) -> bool {
        matches!(self, RuleEvalMode::Naive)
    }

    /// Whether the query/action should be compiled with the permissive
    /// `Read`/`Full` primitive contexts. True for both `:naive` and
    /// `:unsafe-seminaive`.
    pub fn uses_read_contexts(self) -> bool {
        !matches!(self, RuleEvalMode::Seminaive)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GenericRule<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub span: Span,
    pub head: GenericActions<Head, Leaf>,
    pub body: Vec<GenericFact<Head, Leaf>>,
    /// A globally unique name for this rule in the EGraph.
    pub name: String,
    /// The ruleset this rule belongs to. Defaults to `""`.
    pub ruleset: String,
    /// How this rule is evaluated. Set by the mutually exclusive `:naive`
    /// and `:unsafe-seminaive` rule options; defaults to
    /// [`RuleEvalMode::Seminaive`].
    pub eval_mode: RuleEvalMode,
    /// If `true`, this rule skips tree-decomposition during query
    /// planning and evaluate rules as a single-bag (without decomposing
    /// it into smaller queries). Set via the `:no-decomp` rule option.
    pub no_decomp: bool,
    /// If `true`, table atoms in this rule match subsumed rows as well as
    /// live rows. This is intended for internal maintenance rules, not
    /// ordinary user rewrites.
    pub include_subsumed: bool,
}

/// Change a function entry.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub enum Change {
    /// `delete` this entry from a function.
    /// Be wary! Only delete entries that are guaranteed to be not useful.
    Delete,
    /// `subsume` this entry so that it cannot be queried or extracted, but still can be checked.
    /// Note that this is currently forbidden for functions with custom merges.
    Subsume,
}
