use std::fmt::Display;

pub use symbol_table::GlobalSymbol as Symbol;

// macro_rules! lalrpop_error {
//     ($($x:tt)*) => { Err(::lalrpop_util::ParseError::User { error: format!($($x)*)}) }
// }

use lalrpop_util::lalrpop_mod;
lalrpop_mod!(
    #[allow(clippy::all)]
    pub parse,
    "/ast/parse.rs"
);

use crate::{
    core::{GenericAtom, GenericAtomTerm, HeadOrEq, Query, ResolvedCall},
    *,
};

mod expr;
pub use expr::*;
pub mod desugar;

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Id(usize);

impl From<usize> for Id {
    fn from(n: usize) -> Self {
        Id(n)
    }
}

impl From<Id> for usize {
    fn from(id: Id) -> Self {
        id.0
    }
}

impl Display for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "id{}", self.0)
    }
}

pub type NCommand = GenericNCommand<Symbol, Symbol, ()>;
/// [`ResolvedNCommand`] is another specialization of [`GenericNCommand`], which
/// adds the type information to heads and leaves of commands.
/// [`TypeInfo::typecheck_command`] turns an [`NCommand`] into a [`ResolvedNCommand`].
pub(crate) type ResolvedNCommand = GenericNCommand<ResolvedCall, ResolvedVar, ()>;

/// A [`NCommand`] is a desugared [`Command`], where syntactic sugars
/// like [`Command::Datatype`], [`Command::Declare`], and [`Command::Rewrite`]
/// are eliminated.
/// Most of the heavy lifting in egglog is done over [`NCommand`]s.
///
/// [`GenericNCommand`] is a generalization of [`NCommand`], like how [`GenericCommand`]
/// is a generalization of [`Command`], allowing annotations over `Head` and `Leaf`.
///
/// TODO: The name "NCommand" used to denote normalized command, but this
/// meaning is obsolete. A future PR should rename this type to something
/// like "DCommand".
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum GenericNCommand<Head, Leaf, Ann> {
    SetOption {
        name: Symbol,
        value: GenericExpr<Head, Leaf, Ann>,
    },
    Sort(
        Symbol,
        Option<(Symbol, Vec<GenericExpr<Symbol, Symbol, Ann>>)>,
    ),
    Function(GenericFunctionDecl<Head, Leaf, Ann>),
    AddRuleset(Symbol),
    NormRule {
        name: Symbol,
        ruleset: Symbol,
        rule: GenericRule<Head, Leaf, Ann>,
    },
    CoreAction(GenericAction<Head, Leaf, Ann>),
    RunSchedule(GenericSchedule<Head, Leaf, Ann>),
    PrintOverallStatistics,
    Check(Vec<GenericFact<Head, Leaf, Ann>>),
    CheckProof,
    PrintTable(Symbol, usize),
    PrintSize(Option<Symbol>),
    Output {
        file: String,
        exprs: Vec<GenericExpr<Head, Leaf, Ann>>,
    },
    Push(usize),
    Pop(usize),
    Fail(Box<GenericNCommand<Head, Leaf, Ann>>),
    Input {
        name: Symbol,
        file: String,
    },
}

impl<Head, Leaf> GenericNCommand<Head, Leaf, ()>
where
    Head: Clone,
    Leaf: Clone,
{
    pub fn to_command(&self) -> GenericCommand<Head, Leaf> {
        match self {
            GenericNCommand::SetOption { name, value } => GenericCommand::SetOption {
                name: *name,
                value: value.clone(),
            },
            GenericNCommand::Sort(name, params) => GenericCommand::Sort(*name, params.clone()),
            GenericNCommand::Function(f) => GenericCommand::Function(f.clone()),
            GenericNCommand::AddRuleset(name) => GenericCommand::AddRuleset(*name),
            GenericNCommand::NormRule {
                name,
                ruleset,
                rule,
            } => GenericCommand::Rule {
                name: *name,
                ruleset: *ruleset,
                rule: rule.clone(),
            },
            GenericNCommand::RunSchedule(schedule) => GenericCommand::RunSchedule(schedule.clone()),
            GenericNCommand::PrintOverallStatistics => GenericCommand::PrintOverallStatistics,
            GenericNCommand::CoreAction(action) => GenericCommand::Action(action.clone()),
            GenericNCommand::Check(facts) => GenericCommand::Check(facts.clone()),
            GenericNCommand::CheckProof => GenericCommand::CheckProof,
            GenericNCommand::PrintTable(name, n) => GenericCommand::PrintFunction(*name, *n),
            GenericNCommand::PrintSize(name) => GenericCommand::PrintSize(*name),
            GenericNCommand::Output { file, exprs } => GenericCommand::Output {
                file: file.to_string(),
                exprs: exprs.clone(),
            },
            GenericNCommand::Push(n) => GenericCommand::Push(*n),
            GenericNCommand::Pop(n) => GenericCommand::Pop(*n),
            GenericNCommand::Fail(cmd) => GenericCommand::Fail(Box::new(cmd.to_command())),
            GenericNCommand::Input { name, file } => GenericCommand::Input {
                name: *name,
                file: file.clone(),
            },
        }
    }
}

pub type Schedule = GenericSchedule<Symbol, Symbol, ()>;
pub(crate) type ResolvedSchedule = GenericSchedule<ResolvedCall, ResolvedVar, ()>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GenericSchedule<Head, Leaf, Ann> {
    Saturate(Box<GenericSchedule<Head, Leaf, Ann>>),
    Repeat(usize, Box<GenericSchedule<Head, Leaf, Ann>>),
    Run(GenericRunConfig<Head, Leaf, Ann>),
    Sequence(Vec<GenericSchedule<Head, Leaf, Ann>>),
}

impl<Head, Leaf, Ann> GenericSchedule<Head, Leaf, Ann> {
    pub fn saturate(self) -> Self {
        GenericSchedule::Saturate(Box::new(self))
    }
}

pub trait ToSexp {
    fn to_sexp(&self) -> Sexp;
}

impl ToSexp for str {
    fn to_sexp(&self) -> Sexp {
        Sexp::Symbol(String::from(self))
    }
}

impl ToSexp for Symbol {
    fn to_sexp(&self) -> Sexp {
        Sexp::Symbol(self.to_string())
    }
}

impl ToSexp for usize {
    fn to_sexp(&self) -> Sexp {
        Sexp::Symbol(self.to_string())
    }
}

impl ToSexp for Sexp {
    fn to_sexp(&self) -> Sexp {
        self.clone()
    }
}

macro_rules! list {
    ($($e:expr,)* ++ $tail:expr) => {{
        let mut list: Vec<Sexp> = vec![$($e.to_sexp(),)*];
        list.extend($tail.iter().map(|e| e.to_sexp()));
        Sexp::List(list)
    }};
    ($($e:expr),*) => {{
        let list: Vec<Sexp> = vec![ $($e.to_sexp(),)* ];
        Sexp::List(list)
    }};
}

impl<Head: Display, Leaf: Display, Ann> ToSexp for GenericSchedule<Head, Leaf, Ann> {
    fn to_sexp(&self) -> Sexp {
        match self {
            GenericSchedule::Saturate(sched) => list!("saturate", sched),
            GenericSchedule::Repeat(size, sched) => list!("repeat", size, sched),
            GenericSchedule::Run(config) => config.to_sexp(),
            GenericSchedule::Sequence(scheds) => list!("seq", ++ scheds),
        }
    }
}

impl<Head: Display, Leaf: Display, Ann> Display for GenericSchedule<Head, Leaf, Ann> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp())
    }
}

pub type Command = GenericCommand<Symbol, Symbol>;

/// A [`Command`] is the top-level construct in egglog.
/// It includes defining rules, declaring functions,
/// adding to tables, and running rules (via a [`Schedule`]).
#[derive(Debug, Clone)]
pub enum GenericCommand<Head, Leaf> {
    /// Egglog supports several *experimental* options
    /// that can be set using the `set-option` command.
    ///
    /// For example, `(set-option node_limit 1000)` sets a hard limit on the number of "nodes" or rows in the database.
    /// Once this limit is reached, no egglog stops running rules.
    ///
    /// Other options supported include:
    /// - "interactive_mode" (default: false): when enabled, egglog prints "(done)" after each command, allowing an external
    /// tool to know when each command has finished running.
    SetOption {
        name: Symbol,
        value: GenericExpr<Head, Leaf, ()>,
    },
    /// Declare a user-defined datatype.
    /// Datatypes can be unioned with [`Action::Union`] either
    /// at the top level or in the actions of a rule.
    /// This makes them equal in the implicit, global equality relation.

    /// Example:
    /// ```text
    /// (datatype Math
    ///   (Num i64)
    ///   (Var String)
    ///   (Add Math Math)
    ///   (Mul Math Math))
    /// ```

    /// defines a simple `Math` datatype with variants for numbers, named variables, addition and multiplication.
    ///
    /// Datatypes desugar directly to a [`Command::Sort`] and a [`Command::Function`] for each constructor.
    /// The code above becomes:
    /// ```text
    /// (sort Math)
    /// (function Num (i64) Math)
    /// (function Var (String) Math)
    /// (function Add (Math Math) Math)
    /// (function Mul (Math Math) Math)

    /// Datatypes are also known as algebraic data types, tagged unions and sum types.
    Datatype {
        name: Symbol,
        variants: Vec<Variant>,
    },
    /// `declare` is syntactic sugar allowing for the declaration of constants.
    /// For example, the following program:
    /// ```text
    /// (sort Bool)
    /// (declare True Bool)
    /// ```
    /// Desugars to:
    /// ```text
    /// (sort Bool)
    /// (function True_table () Bool)
    /// (let True (True_table))
    /// ```

    /// Note that declare inserts the constant into the database,
    /// so rules can use the constant directly as a variable.
    Declare {
        name: Symbol,
        sort: Symbol,
    },
    /// Create a new user-defined sort, which can then
    /// be used in new [`Command::Function`] declarations.
    /// The [`Command::Datatype`] command desugars directly to this command, with one [`Command::Function`]
    /// per constructor.
    /// The main use of this command (as opposed to using [`Command::Datatype`]) is for forward-declaring a sort for mutually-recursive datatypes.
    ///
    /// It can also be used to create
    /// a container sort.
    /// For example, here's how to make a sort for vectors
    /// of some user-defined sort `Math`:
    /// ```text
    /// (sort MathVec (Vec Math))
    /// ```
    ///
    /// Now `MathVec` can be used as an input or output sort.
    Sort(Symbol, Option<(Symbol, Vec<Expr>)>),
    /// Declare an egglog function, which is a database table with a
    /// a functional dependency (also called a primary key) on its inputs to one output.
    ///
    /// ```text
    /// (function <name:Ident> <schema:Schema> <cost:Cost>
    ///        (:on_merge <List<Action>>)?
    ///        (:merge <Expr>)?
    ///        (:default <Expr>)?)
    ///```
    /// A function can have a `cost` for extraction.
    /// It can also have a `default` value, which is used when calling the function.
    ///
    /// Finally, it can have a `merge` and `on_merge`, which are triggered when
    /// the function dependency is violated.
    /// In this case, the merge expression determines which of the two outputs
    /// for the same input is used.
    /// The `on_merge` actions are run after the merge expression is evaluated.
    ///
    /// Note that the `:merge` expression must be monotonic
    /// for the behavior of the egglog program to be consistent and defined.
    /// In other words, the merge function must define a lattice on the output of the function.
    /// If values are merged in different orders, they should still result in the same output.
    /// If the merge expression is not monotonic, the behavior can vary as
    /// actions may be applied more than once with different results.
    ///
    /// The function is a datatype when:
    /// - The output is not a primitive
    /// - No merge function is provided
    /// - No default is provided
    ///
    /// For example, the following is a datatype:
    /// ```text
    /// (function Add (i64 i64) Math)
    /// ```
    ///
    /// However, this function is not:
    /// ```text
    /// (function LowerBound (Math) i64 :merge (max old new))
    /// ```
    ///
    /// A datatype can be unioned with [`Action::Union`]
    /// with another datatype of the same `sort`.
    ///
    /// Functions that are not a datatype can be `set`
    /// with [`Action::Set`].
    Function(GenericFunctionDecl<Head, Leaf, ()>),
    /// The `relation` is syntactic sugar for a named function which returns the `Unit` type.
    /// Example:
    /// ```text
    /// (relation path (i64 i64))
    /// (relation edge (i64 i64))
    /// ```

    /// Desugars to:
    /// ```text
    /// (function path (i64 i64) Unit :default ())
    /// (function edge (i64 i64) Unit :default ())
    /// ```
    Relation {
        constructor: Symbol,
        inputs: Vec<Symbol>,
    },
    /// Using the `ruleset` command, defines a new
    /// ruleset that can be added to in [`Command::Rule`]s.
    /// Rulesets are used to group rules together
    /// so that they can be run together in a [`Schedule`].
    ///
    /// Example:
    /// Ruleset allows users to define a ruleset- a set of rules

    /// ```text
    /// (ruleset myrules)
    /// (rule ((edge x y))
    ///       ((path x y))
    ///       :ruleset myrules)
    /// (run myrules 2)
    /// ```
    AddRuleset(Symbol),
    /// ```text
    /// (rule <body:List<Fact>> <head:List<Action>>)
    /// ```

    /// defines an egglog rule.
    /// The rule matches a list of facts with respect to
    /// the global database, and runs the list of actions
    /// for each match.
    /// The matches are done *modulo equality*, meaning
    /// equal datatypes in the database are considered
    /// equal.

    /// Example:
    /// ```text
    /// (rule ((edge x y))
    ///       ((path x y)))

    /// (rule ((path x y) (edge y z))
    ///       ((path x z)))
    /// ```
    Rule {
        name: Symbol,
        ruleset: Symbol,
        rule: GenericRule<Head, Leaf, ()>,
    },
    /// `rewrite` is syntactic sugar for a specific form of `rule`
    /// which simply unions the left and right hand sides.
    ///
    /// Example:
    /// ```text
    /// (rewrite (Add a b)
    ///          (Add b a))
    /// ```
    ///
    /// Desugars to:
    /// ```text
    /// (rule ((= lhs (Add a b)))
    ///       ((union lhs (Add b a))))
    /// ```
    ///
    /// Additionally, additional facts can be specified
    /// using a `:when` clause.
    /// For example, the same rule can be run only
    /// when `a` is zero:
    ///
    /// ```text
    /// (rewrite (Add a b)
    ///          (Add b a)
    ///          :when ((= a (Num 0)))
    /// ```
    ///
    Rewrite(Symbol, GenericRewrite<Head, Leaf, ()>),
    /// Similar to [`Command::Rewrite`], but
    /// generates two rules, one for each direction.
    ///
    /// Example:
    /// ```text
    /// (bi-rewrite (Mul (Var x) (Num 0))
    ///             (Var x))
    /// ```
    ///
    /// Becomes:
    /// ```text
    /// (rule ((= lhs (Mul (Var x) (Num 0))))
    ///       ((union lhs (Var x))))
    /// (rule ((= lhs (Var x)))
    ///       ((union lhs (Mul (Var x) (Num 0)))))
    /// ```
    BiRewrite(Symbol, GenericRewrite<Head, Leaf, ()>),
    /// Perform an [`Action`] on the global database
    /// (see documentation for [`Action`] for more details).
    /// Example:
    /// ```text
    /// (let xplusone (Add (Var "x") (Num 1)))
    /// ```
    Action(GenericAction<Head, Leaf, ()>),
    /// Runs a [`Schedule`], which specifies
    /// rulesets and the number of times to run them.
    ///
    /// Example:
    /// ```text
    /// (run-schedule
    ///     (saturate my-ruleset-1)
    ///     (run my-ruleset-2 4))
    /// ```
    ///
    /// Runs `my-ruleset-1` until saturation,
    /// then runs `my-ruleset-2` four times.
    ///
    /// See [`Schedule`] for more details.
    RunSchedule(GenericSchedule<Head, Leaf, ()>),
    /// Print runtime statistics about rules
    /// and rulesets so far.
    PrintOverallStatistics,
    // TODO provide simplify docs
    Simplify {
        expr: GenericExpr<Head, Leaf, ()>,
        schedule: GenericSchedule<Head, Leaf, ()>,
    },
    // TODO provide calc docs
    Calc(Vec<IdentSort>, Vec<GenericExpr<Head, Leaf, ()>>),
    /// The `query-extract` command runs a query,
    /// extracting the result for each match that it finds.
    /// For a simpler extraction command, use [`Action::Extract`] instead.
    ///
    /// Example:
    /// ```text
    /// (query-extract (Add a b))
    /// ```
    ///
    /// Extracts every `Add` term in the database, once
    /// for each class of equivalent `a` and `b`.
    ///
    /// The resulting datatype is chosen from the egraph
    /// as the smallest term by size (taking into account
    /// the `:cost` annotations for each constructor).
    /// This cost does *not* take into account common sub-expressions.
    /// For example, the following term has cost 5:
    /// ```text
    /// (Add
    ///     (Num 1)
    ///     (Num 1))
    /// ```
    ///
    /// Under the hood, this command is implemented with the [`EGraph::extract`]
    /// function.
    QueryExtract {
        variants: usize,
        expr: GenericExpr<Head, Leaf, ()>,
    },
    /// The `check` command checks that the given facts
    /// match at least once in the current database.
    /// The list of facts is matched in the same way a [`Command::Rule`] is matched.
    ///
    /// Example:

    /// ```text
    /// (check (= (+ 1 2) 3))
    /// (check (<= 0 3) (>= 3 0))
    /// (fail (check (= 1 2)))
    /// ```

    /// prints

    /// ```text
    /// [INFO ] Checked.
    /// [INFO ] Checked.
    /// [ERROR] Check failed
    /// [INFO ] Command failed as expected.
    /// ```
    Check(Vec<GenericFact<Head, Leaf, ()>>),
    /// Currently unused, this command will check proofs when they are implemented.
    CheckProof,
    /// Print out rows a given function, extracting each of the elements of the function.
    /// Example:
    /// ```text
    /// (print-function Add 20)
    /// ```
    /// prints the first 20 rows of the `Add` function.
    ///
    PrintFunction(Symbol, usize),
    /// Print out the number of rows in a function or all functions.
    PrintSize(Option<Symbol>),
    /// Input a CSV file directly into a function.
    Input {
        name: Symbol,
        file: String,
    },
    /// Extract and output a set of expressions to a file.
    Output {
        file: String,
        exprs: Vec<GenericExpr<Head, Leaf, ()>>,
    },
    /// `push` the current egraph `n` times so that it is saved.
    /// Later, the current database and rules can be restored using `pop`.
    Push(usize),
    /// `pop` the current egraph, restoring the previous one.
    /// The argument specifies how many egraphs to pop.
    Pop(usize),
    /// Assert that a command fails with an error.
    Fail(Box<GenericCommand<Head, Leaf>>),
    /// Include another egglog file directly as text and run it.
    Include(String),
}

impl<Head: Display + ToSexp, Leaf: Display + ToSexp> ToSexp for GenericCommand<Head, Leaf> {
    fn to_sexp(&self) -> Sexp {
        match self {
            GenericCommand::SetOption { name, value } => list!("set-option", name, value),
            GenericCommand::Rewrite(name, rewrite) => rewrite.to_sexp(*name, false),
            GenericCommand::BiRewrite(name, rewrite) => rewrite.to_sexp(*name, true),
            GenericCommand::Datatype { name, variants } => list!("datatype", name, ++ variants),
            GenericCommand::Declare { name, sort } => list!("declare", name, sort),
            GenericCommand::Action(a) => a.to_sexp(),
            GenericCommand::Sort(name, None) => list!("sort", name),
            GenericCommand::Sort(name, Some((name2, args))) => {
                list!("sort", name, list!( name2, ++ args))
            }
            GenericCommand::Function(f) => f.to_sexp(),
            GenericCommand::Relation {
                constructor,
                inputs,
            } => list!("relation", constructor, list!(++ inputs)),
            GenericCommand::AddRuleset(name) => list!("ruleset", name),
            GenericCommand::Rule {
                name,
                ruleset,
                rule,
            } => rule.to_sexp(*ruleset, *name),
            GenericCommand::RunSchedule(sched) => list!("run-schedule", sched),
            GenericCommand::PrintOverallStatistics => list!("print-stats"),
            GenericCommand::Calc(args, exprs) => list!("calc", list!(++ args), ++ exprs),
            GenericCommand::QueryExtract { variants, expr } => {
                list!("query-extract", ":variants", variants, expr)
            }
            GenericCommand::Check(facts) => list!("check", ++ facts),
            GenericCommand::CheckProof => list!("check-proof"),
            GenericCommand::Push(n) => list!("push", n),
            GenericCommand::Pop(n) => list!("pop", n),
            GenericCommand::PrintFunction(name, n) => list!("print-function", name, n),
            GenericCommand::PrintSize(name) => list!("print-size", ++ name),
            GenericCommand::Input { name, file } => list!("input", name, format!("\"{}\"", file)),
            GenericCommand::Output { file, exprs } => {
                list!("output", format!("\"{}\"", file), ++ exprs)
            }
            GenericCommand::Fail(cmd) => list!("fail", cmd),
            GenericCommand::Include(file) => list!("include", format!("\"{}\"", file)),
            GenericCommand::Simplify { expr, schedule } => list!("simplify", schedule, expr),
        }
    }
}

impl<Head: Display + Clone + ToSexp, Leaf: Display + ToSexp + Clone> Display
    for GenericNCommand<Head, Leaf, ()>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_command())
    }
}

impl<Head: Display + ToSexp, Leaf: Display + ToSexp> Display for GenericCommand<Head, Leaf> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            GenericCommand::Rule {
                ruleset,
                name,
                rule,
            } => rule.fmt_with_ruleset(f, *ruleset, *name),
            GenericCommand::Check(facts) => {
                write!(f, "(check {})", ListDisplay(facts, "\n"))
            }
            _ => write!(f, "{}", self.to_sexp()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IdentSort {
    pub ident: Symbol,
    pub sort: Symbol,
}

impl ToSexp for IdentSort {
    fn to_sexp(&self) -> Sexp {
        list!(self.ident, self.sort)
    }
}

impl Display for IdentSort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp())
    }
}

pub type RunConfig = GenericRunConfig<Symbol, Symbol, ()>;
pub(crate) type ResolvedRunConfig = GenericRunConfig<ResolvedCall, ResolvedVar, ()>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GenericRunConfig<Head, Leaf, Ann> {
    pub ruleset: Symbol,
    pub until: Option<Vec<GenericFact<Head, Leaf, Ann>>>,
}

impl<Head: Display, Leaf: Display, Ann> ToSexp for GenericRunConfig<Head, Leaf, Ann>
where
    Head: Display,
    Leaf: Display,
{
    fn to_sexp(&self) -> Sexp {
        let mut res = vec![Sexp::Symbol("run".into())];
        if self.ruleset != "".into() {
            res.push(Sexp::Symbol(self.ruleset.to_string()));
        }
        if let Some(until) = &self.until {
            res.push(Sexp::Symbol(":until".into()));
            res.extend(until.iter().map(|fact| fact.to_sexp()));
        }

        Sexp::List(res)
    }
}

pub type FunctionDecl = GenericFunctionDecl<Symbol, Symbol, ()>;
pub(crate) type ResolvedFunctionDecl = GenericFunctionDecl<ResolvedCall, ResolvedVar, ()>;

/// Represents the declaration of a function
/// directly parsed from source syntax.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GenericFunctionDecl<Head, Leaf, Ann> {
    pub name: Symbol,
    pub schema: Schema,
    pub default: Option<GenericExpr<Head, Leaf, Ann>>,
    pub merge: Option<GenericExpr<Head, Leaf, Ann>>,
    pub merge_action: GenericActions<Head, Leaf, Ann>,
    pub cost: Option<usize>,
    pub unextractable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Variant {
    pub name: Symbol,
    pub types: Vec<Symbol>,
    pub cost: Option<usize>,
}

impl ToSexp for Variant {
    fn to_sexp(&self) -> Sexp {
        let mut res = vec![Sexp::Symbol(self.name.to_string())];
        if !self.types.is_empty() {
            res.extend(self.types.iter().map(|s| Sexp::Symbol(s.to_string())));
        }
        if let Some(cost) = self.cost {
            res.push(Sexp::Symbol(":cost".into()));
            res.push(Sexp::Symbol(cost.to_string()));
        }
        Sexp::List(res)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Schema {
    pub input: Vec<Symbol>,
    pub output: Symbol,
}

impl ToSexp for Schema {
    fn to_sexp(&self) -> Sexp {
        list!(list!(++ self.input), self.output)
    }
}

impl Schema {
    pub fn new(input: Vec<Symbol>, output: Symbol) -> Self {
        Self { input, output }
    }
}

impl FunctionDecl {
    pub fn relation(name: Symbol, input: Vec<Symbol>) -> Self {
        Self {
            name,
            schema: Schema {
                input,
                output: Symbol::from("Unit"),
            },
            merge: None,
            merge_action: Actions::default(),
            default: Some(Expr::Lit((), Literal::Unit)),
            cost: None,
            unextractable: false,
        }
    }
}

impl<Head: Display + ToSexp, Leaf: Display + ToSexp, Ann> ToSexp
    for GenericFunctionDecl<Head, Leaf, Ann>
{
    fn to_sexp(&self) -> Sexp {
        let mut res = vec![
            Sexp::Symbol("function".into()),
            Sexp::Symbol(self.name.to_string()),
        ];

        if let Sexp::List(contents) = self.schema.to_sexp() {
            res.extend(contents);
        } else {
            unreachable!();
        }

        if let Some(cost) = self.cost {
            res.extend(vec![
                Sexp::Symbol(":cost".into()),
                Sexp::Symbol(cost.to_string()),
            ]);
        }

        if self.unextractable {
            res.push(Sexp::Symbol(":unextractable".into()));
        }

        if !self.merge_action.is_empty() {
            res.push(Sexp::Symbol(":on_merge".into()));
            res.push(Sexp::List(
                self.merge_action.iter().map(|a| a.to_sexp()).collect(),
            ));
        }

        if let Some(merge) = &self.merge {
            res.push(Sexp::Symbol(":merge".into()));
            res.push(merge.to_sexp());
        }

        if let Some(default) = &self.default {
            res.push(Sexp::Symbol(":default".into()));
            res.push(default.to_sexp());
        }

        Sexp::List(res)
    }
}

pub type Fact = GenericFact<Symbol, Symbol, ()>;
pub(crate) type ResolvedFact = GenericFact<ResolvedCall, ResolvedVar, ()>;
pub(crate) type MappedFact<Head, Leaf, Ann> = GenericFact<(Head, Leaf), Leaf, Ann>;

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
pub enum GenericFact<Head, Leaf, Ann> {
    /// Must be at least two things in an eq fact
    Eq(Vec<GenericExpr<Head, Leaf, Ann>>),
    Fact(GenericExpr<Head, Leaf, Ann>),
}

pub struct Facts<Head, Leaf, Ann>(pub Vec<GenericFact<Head, Leaf, Ann>>);

impl<Head, Leaf, Ann> Facts<Head, Leaf, Ann>
where
    Head: Clone,
    Leaf: Clone,
    Ann: Clone,
{
    /// Flattens a list of facts into a Query.
    /// For typechecking, we need the correspondence between the original ast
    /// and the flattened one, so that we can annotate the original with types.
    /// That's why this function produces a corresponding list of facts, annotated with
    /// the variable names in the flattened Query.
    /// (Typechecking preserves the original AST this way,
    /// and allows terms and proof instrumentation to do the same).
    pub(crate) fn to_query(
        &self,
        typeinfo: &TypeInfo,
        fresh_gen: &mut impl FreshGen<Head, Leaf>,
    ) -> (
        Query<HeadOrEq<Head>, Leaf>,
        Vec<MappedFact<Head, Leaf, Ann>>,
    )
    where
        Leaf: SymbolLike,
    {
        let mut atoms = vec![];
        let mut new_body = vec![];

        for fact in self.0.iter() {
            match fact {
                GenericFact::Eq(exprs) => {
                    let mut new_exprs = vec![];
                    let mut to_equate = vec![];
                    for expr in exprs {
                        let (child_atoms, expr) = expr.to_query(typeinfo, fresh_gen);
                        atoms.extend(child_atoms);
                        to_equate.push(expr.get_corresponding_var_or_lit(typeinfo));
                        new_exprs.push(expr);
                    }
                    atoms.push(GenericAtom {
                        head: HeadOrEq::Eq,
                        args: to_equate,
                    });
                    new_body.push(GenericFact::Eq(new_exprs));
                }
                GenericFact::Fact(expr) => {
                    let (child_atoms, expr) = expr.to_query(typeinfo, fresh_gen);
                    atoms.extend(child_atoms);
                    new_body.push(GenericFact::Fact(expr));
                }
            }
        }
        (Query { atoms }, new_body)
    }
}

impl<Head: Display, Leaf: Display, Ann> ToSexp for GenericFact<Head, Leaf, Ann>
where
    Head: Display,
    Leaf: Display,
{
    fn to_sexp(&self) -> Sexp {
        match self {
            GenericFact::Eq(exprs) => list!("=", ++ exprs),
            GenericFact::Fact(expr) => expr.to_sexp(),
        }
    }
}

impl<Head, Leaf, Ann> GenericFact<Head, Leaf, Ann>
where
    Ann: Clone,
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub(crate) fn map_exprs<Head2, Leaf2>(
        &self,
        f: &mut impl FnMut(&GenericExpr<Head, Leaf, Ann>) -> GenericExpr<Head2, Leaf2, Ann>,
    ) -> GenericFact<Head2, Leaf2, Ann> {
        match self {
            GenericFact::Eq(exprs) => GenericFact::Eq(exprs.iter().map(f).collect()),
            GenericFact::Fact(expr) => GenericFact::Fact(f(expr)),
        }
    }

    pub(crate) fn subst<Leaf2, Head2>(
        &self,
        subst_leaf: &mut impl FnMut(&Leaf) -> GenericExpr<Head2, Leaf2, Ann>,
        subst_head: &mut impl FnMut(&Head) -> Head2,
    ) -> GenericFact<Head2, Leaf2, Ann> {
        self.map_exprs(&mut |e| e.subst(subst_leaf, subst_head))
    }
}

impl<Head, Leaf> GenericFact<Head, Leaf, ()>
where
    Leaf: Clone + PartialEq + Eq + Display + Hash,
    Head: Clone + Display,
{
    pub(crate) fn to_unresolved(&self) -> Fact
    where
        Leaf: SymbolLike,
        Head: SymbolLike,
    {
        self.subst(&mut |v| GenericExpr::Var((), v.to_symbol()), &mut |h| {
            h.to_symbol()
        })
    }
}

impl<Head, Leaf, Ann> Display for GenericFact<Head, Leaf, Ann>
where
    Head: Display,
    Leaf: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp())
    }
}

pub type Action = GenericAction<Symbol, Symbol, ()>;
pub(crate) type MappedAction = GenericAction<(Symbol, Symbol), Symbol, ()>;
pub(crate) type ResolvedAction = GenericAction<ResolvedCall, ResolvedVar, ()>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GenericAction<Head, Leaf, Ann> {
    /// Bind a variable to a particular datatype or primitive.
    /// At the top level (in a [`Command::Action`]), this defines a global variable.
    /// In a [`Command::Rule`], this defines a local variable in the actions.
    Let(Ann, Leaf, GenericExpr<Head, Leaf, Ann>),
    /// `set` a function to a particular result.
    /// `set` should not be used on datatypes-
    /// instead, use `union`.
    Set(
        Ann,
        Head,
        Vec<GenericExpr<Head, Leaf, Ann>>,
        GenericExpr<Head, Leaf, Ann>,
    ),
    /// `delete` an entry from a function.
    /// Be wary! Only delete entries that are subsumed in some way or
    /// guaranteed to be not useful.
    Delete(Ann, Head, Vec<GenericExpr<Head, Leaf, Ann>>),
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
    Union(
        Ann,
        GenericExpr<Head, Leaf, Ann>,
        GenericExpr<Head, Leaf, Ann>,
    ),
    /// `extract` a datatype from the egraph, choosing
    /// the smallest representative.
    /// By default, each constructor costs 1 to extract
    /// (common subexpressions are not shared in the cost
    /// model).
    /// The second argument is the number of variants to
    /// extract, picking different terms in the
    /// same equivalence class.
    Extract(
        Ann,
        GenericExpr<Head, Leaf, Ann>,
        GenericExpr<Head, Leaf, Ann>,
    ),
    Panic(Ann, String),
    Expr(Ann, GenericExpr<Head, Leaf, Ann>),
    // If(Expr, Action, Action),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GenericActions<Head, Leaf, Ann>(pub Vec<GenericAction<Head, Leaf, Ann>>);
pub type Actions = GenericActions<Symbol, Symbol, ()>;
pub(crate) type ResolvedActions = GenericActions<ResolvedCall, ResolvedVar, ()>;
pub(crate) type MappedActions<Head, Leaf, Ann> = GenericActions<(Head, Leaf), Leaf, Ann>;

impl<Head, Leaf, Ann> Default for GenericActions<Head, Leaf, Ann> {
    fn default() -> Self {
        Self(vec![])
    }
}

impl<Head, Leaf, Ann> GenericActions<Head, Leaf, Ann> {
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &GenericAction<Head, Leaf, Ann>> {
        self.0.iter()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<Head: Display + ToSexp, Leaf: Display + ToSexp, Ann> ToSexp
    for GenericAction<Head, Leaf, Ann>
{
    fn to_sexp(&self) -> Sexp {
        match self {
            GenericAction::Let(_ann, lhs, rhs) => list!("let", lhs, rhs),
            GenericAction::Set(_ann, lhs, args, rhs) => list!("set", list!(lhs, ++ args), rhs),
            GenericAction::Union(_ann, lhs, rhs) => list!("union", lhs, rhs),
            GenericAction::Delete(_ann, lhs, args) => list!("delete", list!(lhs, ++ args)),
            GenericAction::Extract(_ann, expr, variants) => list!("extract", expr, variants),
            GenericAction::Panic(_ann, msg) => list!("panic", format!("\"{}\"", msg.clone())),
            GenericAction::Expr(_ann, e) => e.to_sexp(),
        }
    }
}

impl<Head, Leaf, Ann> GenericAction<Head, Leaf, Ann>
where
    Head: Clone + Display,
    Leaf: Clone + Eq + Display + Hash,
    Ann: Clone + Default,
{
    pub fn map_exprs(
        &self,
        f: &mut impl FnMut(&GenericExpr<Head, Leaf, Ann>) -> GenericExpr<Head, Leaf, Ann>,
    ) -> Self {
        match self {
            GenericAction::Let(ann, lhs, rhs) => {
                GenericAction::Let(ann.clone(), lhs.clone(), f(rhs))
            }
            GenericAction::Set(ann, lhs, args, rhs) => {
                let right = f(rhs);
                GenericAction::Set(
                    ann.clone(),
                    lhs.clone(),
                    args.iter().map(f).collect(),
                    right,
                )
            }
            GenericAction::Delete(ann, lhs, args) => {
                GenericAction::Delete(ann.clone(), lhs.clone(), args.iter().map(f).collect())
            }
            GenericAction::Union(ann, lhs, rhs) => {
                GenericAction::Union(ann.clone(), f(lhs), f(rhs))
            }
            GenericAction::Extract(ann, expr, variants) => {
                GenericAction::Extract(ann.clone(), f(expr), f(variants))
            }
            GenericAction::Panic(ann, msg) => GenericAction::Panic(ann.clone(), msg.clone()),
            GenericAction::Expr(ann, e) => GenericAction::Expr(ann.clone(), f(e)),
        }
    }

    pub fn subst(&self, subst: &mut impl FnMut(&Leaf) -> GenericExpr<Head, Leaf, Ann>) -> Self {
        self.map_exprs(&mut |e| e.subst_leaf(subst))
    }

    pub fn map_def_use(&self, fvar: &mut impl FnMut(&Leaf, bool) -> Leaf) -> Self {
        macro_rules! fvar_expr {
            () => {
                |s: &_| GenericExpr::Var(Ann::default(), fvar(s, false))
            };
        }
        match self {
            GenericAction::Let(ann, lhs, rhs) => {
                let lhs = fvar(lhs, true);
                let rhs = rhs.subst_leaf(&mut fvar_expr!());
                GenericAction::Let(ann.clone(), lhs, rhs)
            }
            GenericAction::Set(ann, lhs, args, rhs) => {
                let args = args
                    .iter()
                    .map(|e| e.subst_leaf(&mut fvar_expr!()))
                    .collect();
                let rhs = rhs.subst_leaf(&mut fvar_expr!());
                GenericAction::Set(ann.clone(), lhs.clone(), args, rhs)
            }
            GenericAction::Delete(ann, lhs, args) => {
                let args = args
                    .iter()
                    .map(|e| e.subst_leaf(&mut fvar_expr!()))
                    .collect();
                GenericAction::Delete(ann.clone(), lhs.clone(), args)
            }
            GenericAction::Union(ann, lhs, rhs) => {
                let lhs = lhs.subst_leaf(&mut fvar_expr!());
                let rhs = rhs.subst_leaf(&mut fvar_expr!());
                GenericAction::Union(ann.clone(), lhs, rhs)
            }
            GenericAction::Extract(ann, expr, variants) => {
                let expr = expr.subst_leaf(&mut fvar_expr!());
                let variants = variants.subst_leaf(&mut fvar_expr!());
                GenericAction::Extract(ann.clone(), expr, variants)
            }
            GenericAction::Panic(ann, msg) => GenericAction::Panic(ann.clone(), msg.clone()),
            GenericAction::Expr(ann, e) => {
                GenericAction::Expr(ann.clone(), e.subst_leaf(&mut fvar_expr!()))
            }
        }
    }
}

impl<Head: Display + ToSexp, Leaf: Display + ToSexp, Ann> Display
    for GenericAction<Head, Leaf, Ann>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp())
    }
}

pub type Rule = GenericRule<Symbol, Symbol, ()>;
pub(crate) type ResolvedRule = GenericRule<ResolvedCall, ResolvedVar, ()>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GenericRule<Head, Leaf, Ann> {
    pub head: GenericActions<Head, Leaf, Ann>,
    pub body: Vec<GenericFact<Head, Leaf, Ann>>,
}

impl<Head: Display + ToSexp, Leaf: Display + ToSexp, Ann> GenericRule<Head, Leaf, Ann> {
    pub(crate) fn fmt_with_ruleset(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        ruleset: Symbol,
        name: Symbol,
    ) -> std::fmt::Result {
        let indent = " ".repeat(7);
        write!(f, "(rule (")?;
        for (i, fact) in self.body.iter().enumerate() {
            if i > 0 {
                write!(f, "{}", indent)?;
            }

            if i != self.body.len() - 1 {
                writeln!(f, "{}", fact)?;
            } else {
                write!(f, "{}", fact)?;
            }
        }
        write!(f, ")\n      (")?;
        for (i, action) in self.head.0.iter().enumerate() {
            if i > 0 {
                write!(f, "{}", indent)?;
            }
            if i != self.head.0.len() - 1 {
                writeln!(f, "{}", action)?;
            } else {
                write!(f, "{}", action)?;
            }
        }
        let ruleset = if ruleset != "".into() {
            format!(":ruleset {}", ruleset)
        } else {
            "".into()
        };
        let name = if name != "".into() {
            format!(":name \"{}\"", name)
        } else {
            "".into()
        };
        write!(f, ")\n{} {} {})", indent, ruleset, name)
    }
}

impl<Head: Display + ToSexp, Leaf: Display + ToSexp, Ann> GenericRule<Head, Leaf, Ann> {
    /// Converts this rule into an s-expression.
    pub fn to_sexp(&self, ruleset: Symbol, name: Symbol) -> Sexp {
        let mut res = vec![
            Sexp::Symbol("rule".into()),
            Sexp::List(self.body.iter().map(|f| f.to_sexp()).collect()),
            Sexp::List(self.head.0.iter().map(|a| a.to_sexp()).collect()),
        ];
        if ruleset != "".into() {
            res.push(Sexp::Symbol(":ruleset".into()));
            res.push(Sexp::Symbol(ruleset.to_string()));
        }
        if name != "".into() {
            res.push(Sexp::Symbol(":name".into()));
            res.push(Sexp::Symbol(format!("\"{}\"", name)));
        }
        Sexp::List(res)
    }
}

impl<Head: Display + ToSexp, Leaf: Display + ToSexp, Ann> Display for GenericRule<Head, Leaf, Ann> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_with_ruleset(f, "".into(), "".into())
    }
}

type Rewrite = GenericRewrite<Symbol, Symbol, ()>;

#[derive(Clone, Debug)]
pub struct GenericRewrite<Head, Leaf, Ann> {
    pub lhs: GenericExpr<Head, Leaf, Ann>,
    pub rhs: GenericExpr<Head, Leaf, Ann>,
    pub conditions: Vec<GenericFact<Head, Leaf, Ann>>,
}

impl<Head: Display, Leaf: Display, Ann> GenericRewrite<Head, Leaf, Ann> {
    /// Converts the rewrite into an s-expression.
    pub fn to_sexp(&self, ruleset: Symbol, is_bidirectional: bool) -> Sexp {
        let mut res = vec![
            Sexp::Symbol(if is_bidirectional {
                "birewrite".into()
            } else {
                "rewrite".into()
            }),
            self.lhs.to_sexp(),
            self.rhs.to_sexp(),
        ];

        if !self.conditions.is_empty() {
            res.push(Sexp::Symbol(":when".into()));
            res.push(Sexp::List(
                self.conditions.iter().map(|f| f.to_sexp()).collect(),
            ));
        }

        if ruleset != "".into() {
            res.push(Sexp::Symbol(":ruleset".into()));
            res.push(Sexp::Symbol(ruleset.to_string()));
        }
        Sexp::List(res)
    }
}

impl<Head: Display, Leaf: Display, Ann> Display for GenericRewrite<Head, Leaf, Ann> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp("".into(), false))
    }
}

impl<Head, Leaf: Clone, Ann> MappedExpr<Head, Leaf, Ann> {
    pub(crate) fn get_corresponding_var_or_lit(&self, typeinfo: &TypeInfo) -> GenericAtomTerm<Leaf>
    where
        Leaf: SymbolLike,
    {
        // Note: need typeinfo to resolve whether a symbol is a global or not
        // This is error-prone and the complexities can be avoided by treating globals
        // as nullary functions.
        match self {
            GenericExpr::Var(_ann, v) => {
                if typeinfo.is_global(v.to_symbol()) {
                    GenericAtomTerm::Global(v.clone())
                } else {
                    GenericAtomTerm::Var(v.clone())
                }
            }
            GenericExpr::Lit(_ann, lit) => GenericAtomTerm::Literal(lit.clone()),
            GenericExpr::Call(_ann, head, _) => GenericAtomTerm::Var(head.1.clone()),
        }
    }
}

impl<Head, Leaf> GenericActions<Head, Leaf, ()>
where
    Head: Clone,
    Leaf: Clone + Hash + Eq + Clone,
{
    pub fn new(actions: Vec<GenericAction<Head, Leaf, ()>>) -> Self {
        Self(actions)
    }

    pub fn singleton(action: GenericAction<Head, Leaf, ()>) -> Self {
        Self(vec![action])
    }
}
