pub mod check_shadowing;
pub mod desugar;
mod expr;
mod parse;
pub mod proof_global_remover;
pub mod remove_globals;

use crate::core::{
    GenericAtom, GenericAtomTerm, GenericExprExt, HeadOrEq, Query, ResolvedCall, ResolvedCoreRule,
};
use crate::*;
pub use egglog_ast::generic_ast::{
    Change, GenericAction, GenericActions, GenericExpr, GenericFact, GenericRule, Literal,
};
pub use egglog_ast::span::{RustSpan, Span};
use egglog_ast::util::ListDisplay;
pub use expr::*;
pub use parse::*;

#[derive(Clone, Debug)]
/// The egglog internal representation of already compiled rules
pub(crate) enum Ruleset {
    /// Represents a ruleset with a set of rules.
    Rules(IndexMap<String, (ResolvedCoreRule, egglog_bridge::RuleId)>),
    /// A combined ruleset may contain other rulesets.
    Combined(Vec<String>),
}

pub type NCommand = GenericNCommand<String, String>;
/// [`ResolvedNCommand`] is another specialization of [`GenericNCommand`], which
/// adds the type information to heads and leaves of commands.
/// [`TypeInfo::typecheck_command`] turns an [`NCommand`] into a [`ResolvedNCommand`].
pub(crate) type ResolvedNCommand = GenericNCommand<ResolvedCall, ResolvedVar>;

/// A [`NCommand`] is a desugared [`Command`], where syntactic sugars
/// like [`Command::Datatype`] and [`Command::Rewrite`]
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
pub enum GenericNCommand<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    Sort {
        span: Span,
        name: String,
        presort_and_args: Option<(String, Vec<GenericExpr<String, String>>)>,
        /// The name of the union-find function for this sort.
        /// Used in term encoding to canonicalize values during extraction.
        uf: Option<String>,
        /// Whether values of this sort can be unioned.
        /// Defaults to true for user-defined sorts.
        /// Set to false for relations and term tables that should not allow union.
        unionable: bool,
    },
    Function(GenericFunctionDecl<Head, Leaf>),
    AddRuleset(Span, String),
    UnstableCombinedRuleset(Span, String, Vec<String>),
    NormRule {
        rule: GenericRule<Head, Leaf>,
    },
    CoreAction(GenericAction<Head, Leaf>),
    Extract(Span, GenericExpr<Head, Leaf>, GenericExpr<Head, Leaf>),
    RunSchedule(GenericSchedule<Head, Leaf>),
    PrintOverallStatistics(Span, Option<String>),
    Check(Span, Vec<GenericFact<Head, Leaf>>),
    PrintFunction(
        Span,
        String,
        Option<usize>,
        Option<String>,
        PrintFunctionMode,
    ),
    ProveExists(Span, Head),
    PrintSize(Span, Option<String>),
    Output {
        span: Span,
        file: String,
        exprs: Vec<GenericExpr<Head, Leaf>>,
    },
    Push(usize),
    Pop(Span, usize),
    Fail(Span, Box<GenericNCommand<Head, Leaf>>),
    Input {
        span: Span,
        name: String,
        file: String,
    },
    UserDefined(Span, String, Vec<Expr>),
}

impl<Head, Leaf> GenericNCommand<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub fn to_command(&self) -> GenericCommand<Head, Leaf> {
        match self {
            GenericNCommand::Sort {
                span,
                name,
                presort_and_args,
                uf,
                unionable,
            } => GenericCommand::Sort {
                span: span.clone(),
                name: name.clone(),
                presort_and_args: presort_and_args.clone(),
                uf: uf.clone(),
                unionable: *unionable,
            },
            GenericNCommand::Function(f) => match f.subtype {
                FunctionSubtype::Constructor => GenericCommand::Constructor {
                    span: f.span.clone(),
                    name: f.name.clone(),
                    schema: f.schema.clone(),
                    cost: f.cost,
                    unextractable: f.unextractable,
                    term_constructor: f.term_constructor.clone(),
                },
                FunctionSubtype::Custom => GenericCommand::Function {
                    span: f.span.clone(),
                    schema: f.schema.clone(),
                    name: f.name.clone(),
                    merge: f.merge.clone(),
                },
            },
            GenericNCommand::AddRuleset(span, name) => {
                GenericCommand::AddRuleset(span.clone(), name.clone())
            }
            GenericNCommand::UnstableCombinedRuleset(span, name, others) => {
                GenericCommand::UnstableCombinedRuleset(span.clone(), name.clone(), others.clone())
            }
            GenericNCommand::NormRule { rule } => GenericCommand::Rule { rule: rule.clone() },
            GenericNCommand::RunSchedule(schedule) => GenericCommand::RunSchedule(schedule.clone()),
            GenericNCommand::PrintOverallStatistics(span, file) => {
                GenericCommand::PrintOverallStatistics(span.clone(), file.clone())
            }
            GenericNCommand::CoreAction(action) => GenericCommand::Action(action.clone()),
            GenericNCommand::Extract(span, expr, variants) => {
                GenericCommand::Extract(span.clone(), expr.clone(), variants.clone())
            }
            GenericNCommand::Check(span, facts) => {
                GenericCommand::Check(span.clone(), facts.clone())
            }
            GenericNCommand::PrintFunction(span, name, n, file, mode) => {
                GenericCommand::PrintFunction(span.clone(), name.clone(), *n, file.clone(), *mode)
            }
            GenericNCommand::ProveExists(span, constructor) => {
                GenericCommand::ProveExists(span.clone(), constructor.clone())
            }
            GenericNCommand::PrintSize(span, name) => {
                GenericCommand::PrintSize(span.clone(), name.clone())
            }
            GenericNCommand::Output { span, file, exprs } => GenericCommand::Output {
                span: span.clone(),
                file: file.to_string(),
                exprs: exprs.clone(),
            },
            GenericNCommand::Push(n) => GenericCommand::Push(*n),
            GenericNCommand::Pop(span, n) => GenericCommand::Pop(span.clone(), *n),
            GenericNCommand::Fail(span, cmd) => {
                GenericCommand::Fail(span.clone(), Box::new(cmd.to_command()))
            }
            GenericNCommand::Input { span, name, file } => GenericCommand::Input {
                span: span.clone(),
                name: name.clone(),
                file: file.clone(),
            },
            GenericNCommand::UserDefined(span, name, exprs) => {
                GenericCommand::UserDefined(span.clone(), name.clone(), exprs.clone())
            }
        }
    }

    /// Applies `f` to
    pub fn visit_queries(
        self,
        f: &mut impl FnMut(Vec<GenericFact<Head, Leaf>>) -> Vec<GenericFact<Head, Leaf>>,
    ) -> Self {
        match self {
            GenericNCommand::Check(span, query) => GenericNCommand::Check(span, f(query)),
            GenericNCommand::NormRule { mut rule } => {
                rule.body = f(rule.body);
                GenericNCommand::NormRule { rule }
            }
            GenericNCommand::RunSchedule(schedule) => {
                GenericNCommand::RunSchedule(schedule.visit_queries(f))
            }
            GenericNCommand::Fail(span, cmd) => {
                GenericNCommand::Fail(span, Box::new(cmd.visit_queries(f)))
            }
            GenericNCommand::Sort { .. }
            | GenericNCommand::Function(..)
            | GenericNCommand::AddRuleset(..)
            | GenericNCommand::UnstableCombinedRuleset(..)
            | GenericNCommand::CoreAction(..)
            | GenericNCommand::Extract(..)
            | GenericNCommand::PrintOverallStatistics(..)
            | GenericNCommand::PrintFunction(..)
            | GenericNCommand::PrintSize(..)
            | GenericNCommand::Output { .. }
            | GenericNCommand::Push(..)
            | GenericNCommand::Pop(..)
            | GenericNCommand::Input { .. }
            | GenericNCommand::UserDefined(..)
            | GenericNCommand::ProveExists(..) => self,
        }
    }

    /// Applies `f` to all expressions in the command, bottom-up.
    pub fn visit_exprs(
        self,
        f: &mut impl FnMut(GenericExpr<Head, Leaf>) -> GenericExpr<Head, Leaf>,
    ) -> Self {
        match self {
            GenericNCommand::Sort {
                span,
                name,
                presort_and_args,
                uf,
                unionable,
            } => GenericNCommand::Sort {
                span,
                name,
                presort_and_args,
                uf,
                unionable,
            },
            GenericNCommand::Function(func) => GenericNCommand::Function(func.visit_exprs(f)),
            GenericNCommand::AddRuleset(span, name) => GenericNCommand::AddRuleset(span, name),
            GenericNCommand::UnstableCombinedRuleset(span, name, rulesets) => {
                GenericNCommand::UnstableCombinedRuleset(span, name, rulesets)
            }
            GenericNCommand::NormRule { rule } => GenericNCommand::NormRule {
                rule: rule.visit_exprs(f),
            },
            GenericNCommand::RunSchedule(schedule) => {
                GenericNCommand::RunSchedule(schedule.visit_exprs(f))
            }
            GenericNCommand::PrintOverallStatistics(span, file) => {
                GenericNCommand::PrintOverallStatistics(span, file)
            }
            GenericNCommand::CoreAction(action) => {
                GenericNCommand::CoreAction(action.visit_exprs(f))
            }
            GenericNCommand::Extract(span, expr, variants) => {
                GenericNCommand::Extract(span, expr.visit_exprs(f), variants.visit_exprs(f))
            }
            GenericNCommand::Check(span, facts) => GenericNCommand::Check(
                span,
                facts.into_iter().map(|fact| fact.visit_exprs(f)).collect(),
            ),
            GenericNCommand::PrintFunction(span, name, n, file, mode) => {
                GenericNCommand::PrintFunction(span, name, n, file, mode)
            }
            GenericNCommand::ProveExists(span, constructor) => {
                GenericNCommand::ProveExists(span, constructor)
            }
            GenericNCommand::PrintSize(span, name) => GenericNCommand::PrintSize(span, name),
            GenericNCommand::Output { span, file, exprs } => GenericNCommand::Output {
                span,
                file,
                exprs: exprs.into_iter().map(f).collect(),
            },
            GenericNCommand::Push(n) => GenericNCommand::Push(n),
            GenericNCommand::Pop(span, n) => GenericNCommand::Pop(span, n),
            GenericNCommand::Fail(span, cmd) => {
                GenericNCommand::Fail(span, Box::new(cmd.visit_exprs(f)))
            }
            GenericNCommand::Input { span, name, file } => {
                GenericNCommand::Input { span, name, file }
            }
            GenericNCommand::UserDefined(span, name, exprs) => {
                // We can't map `f` over UserDefined because UserDefined always assumes plain `Expr`s
                GenericNCommand::UserDefined(span, name, exprs)
            }
        }
    }
}

impl<Head, Leaf> Display for GenericNCommand<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        let command = self.to_command();
        command.fmt(f)
    }
}

pub type Schedule = GenericSchedule<String, String>;
pub(crate) type ResolvedSchedule = GenericSchedule<ResolvedCall, ResolvedVar>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GenericSchedule<Head, Leaf> {
    Saturate(Span, Box<GenericSchedule<Head, Leaf>>),
    Repeat(Span, usize, Box<GenericSchedule<Head, Leaf>>),
    Run(Span, GenericRunConfig<Head, Leaf>),
    Sequence(Span, Vec<GenericSchedule<Head, Leaf>>),
}

impl<Head, Leaf> GenericSchedule<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    /// Applies `f` to all the queries in the schedule.
    pub fn visit_queries(
        self,
        f: &mut impl FnMut(Vec<GenericFact<Head, Leaf>>) -> Vec<GenericFact<Head, Leaf>>,
    ) -> Self {
        match self {
            GenericSchedule::Saturate(span, generic_schedule) => {
                GenericSchedule::Saturate(span, Box::new(generic_schedule.visit_queries(f)))
            }
            GenericSchedule::Repeat(span, iters, generic_schedule) => {
                GenericSchedule::Repeat(span, iters, Box::new(generic_schedule.visit_queries(f)))
            }
            GenericSchedule::Run(span, run_config) => GenericSchedule::Run(
                span,
                GenericRunConfig {
                    ruleset: run_config.ruleset,
                    until: run_config.until.map(f),
                },
            ),
            GenericSchedule::Sequence(span, generic_schedules) => GenericSchedule::Sequence(
                span,
                generic_schedules
                    .into_iter()
                    .map(|schedule| schedule.visit_queries(f))
                    .collect(),
            ),
        }
    }

    /// Recursively flattens nested `Sequence` nodes into a single level.
    /// For example, `(seq (seq a b) c)` becomes `(seq a b c)`.
    /// Also unwraps single-element sequences into their inner schedule.
    fn flatten_sequences(self) -> Self {
        match self {
            GenericSchedule::Saturate(span, sched) => {
                GenericSchedule::Saturate(span, Box::new(sched.flatten_sequences()))
            }
            GenericSchedule::Repeat(span, size, sched) => {
                GenericSchedule::Repeat(span, size, Box::new(sched.flatten_sequences()))
            }
            GenericSchedule::Run(span, config) => GenericSchedule::Run(span, config),
            GenericSchedule::Sequence(span, scheds) => {
                let mut flattened = Vec::new();
                for sched in scheds.into_iter().map(Self::flatten_sequences) {
                    match sched {
                        GenericSchedule::Sequence(_, nested) => flattened.extend(nested),
                        other => flattened.push(other),
                    }
                }

                match flattened.len() {
                    0 => GenericSchedule::Sequence(span, flattened),
                    1 => flattened.into_iter().next().unwrap(),
                    _ => GenericSchedule::Sequence(span, flattened),
                }
            }
        }
    }

    fn visit_exprs(
        self,
        f: &mut impl FnMut(GenericExpr<Head, Leaf>) -> GenericExpr<Head, Leaf>,
    ) -> Self {
        match self {
            GenericSchedule::Saturate(span, sched) => {
                GenericSchedule::Saturate(span, Box::new(sched.visit_exprs(f)))
            }
            GenericSchedule::Repeat(span, size, sched) => {
                GenericSchedule::Repeat(span, size, Box::new(sched.visit_exprs(f)))
            }
            GenericSchedule::Run(span, config) => GenericSchedule::Run(span, config.visit_exprs(f)),
            GenericSchedule::Sequence(span, scheds) => GenericSchedule::Sequence(
                span,
                scheds.into_iter().map(|s| s.visit_exprs(f)).collect(),
            ),
        }
    }

    /// Remaps every head and leaf symbol in the schedule using the supplied closures.
    pub fn map_symbols<Head2, Leaf2>(
        self,
        head: &mut impl FnMut(Head) -> Head2,
        leaf: &mut impl FnMut(Leaf) -> Leaf2,
    ) -> GenericSchedule<Head2, Leaf2>
    where
        Head2: Clone + Display,
        Leaf2: Clone + PartialEq + Eq + Display + Hash,
    {
        match self {
            GenericSchedule::Saturate(span, sched) => {
                GenericSchedule::Saturate(span, Box::new(sched.map_symbols(head, leaf)))
            }
            GenericSchedule::Repeat(span, size, sched) => {
                GenericSchedule::Repeat(span, size, Box::new(sched.map_symbols(head, leaf)))
            }
            GenericSchedule::Run(span, config) => {
                GenericSchedule::Run(span, config.map_symbols(head, leaf))
            }
            GenericSchedule::Sequence(span, scheds) => GenericSchedule::Sequence(
                span,
                scheds
                    .into_iter()
                    .map(|sched| sched.map_symbols(head, leaf))
                    .collect(),
            ),
        }
    }

    /// Applies `fun` to every string-valued symbol contained in the schedule,
    /// normalizes result with `flatten_sequences`.
    pub fn map_string_symbols(
        self,
        fun: &mut impl FnMut(String) -> String,
    ) -> GenericSchedule<Head, Leaf> {
        let mapped = match self {
            GenericSchedule::Saturate(span, sched) => {
                GenericSchedule::Saturate(span, Box::new(sched.map_string_symbols(fun)))
            }
            GenericSchedule::Repeat(span, size, sched) => {
                GenericSchedule::Repeat(span, size, Box::new(sched.map_string_symbols(fun)))
            }
            GenericSchedule::Run(span, config) => {
                GenericSchedule::Run(span, config.map_string_symbols(fun))
            }
            GenericSchedule::Sequence(span, scheds) => GenericSchedule::Sequence(
                span,
                scheds
                    .into_iter()
                    .map(|sched| sched.map_string_symbols(fun))
                    .collect(),
            ),
        };

        mapped.flatten_sequences()
    }

    /// Converts all heads and leaves to strings.
    pub fn make_unresolved(self) -> GenericSchedule<String, String> {
        let mut map_head = |h: Head| h.to_string();
        let mut map_leaf = |l: Leaf| l.to_string();
        self.map_symbols(&mut map_head, &mut map_leaf)
    }
}

impl<Head: Display, Leaf: Display> Display for GenericSchedule<Head, Leaf> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            GenericSchedule::Saturate(_ann, sched) => write!(f, "(saturate {sched})"),
            GenericSchedule::Repeat(_ann, size, sched) => write!(f, "(repeat {size} {sched})"),
            GenericSchedule::Run(_ann, config) => write!(f, "{config}"),
            GenericSchedule::Sequence(_ann, scheds) => {
                write!(f, "(seq {})", ListDisplay(scheds, " "))
            }
        }
    }
}

pub type Command = GenericCommand<String, String>;
pub type ResolvedCommand = GenericCommand<ResolvedCall, ResolvedVar>;

pub type Subsume = bool;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Subdatatypes {
    Variants(Vec<Variant>),
    NewSort(String, Vec<Expr>),
}

/// The mode of printing a function. The default mode prints the function in a user-friendly way and
/// has an unreliable interface.
/// The CSV mode prints the function in the CSV format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrintFunctionMode {
    Default,
    CSV,
}

impl Display for PrintFunctionMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PrintFunctionMode::Default => write!(f, "default"),
            PrintFunctionMode::CSV => write!(f, "csv"),
        }
    }
}

/// A [`Command`] is the top-level construct in egglog.
/// It includes defining rules, declaring functions,
/// adding to tables, and running rules (via a [`Schedule`]).
///
/// # Binding naming convention
/// Bindings introduced by commands fall into two categories:
/// - **Global bindings** must start with [`$`](crate::GLOBAL_NAME_PREFIX).
/// - **Non-global bindings** must *not* start with [`$`](crate::GLOBAL_NAME_PREFIX).
///
/// When `--strict-mode` is enabled, violating these conventions is a type error;
/// otherwise, egglog emits a single warning per program.
#[derive(Debug, Clone)]
pub enum GenericCommand<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
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
    Sort {
        span: Span,
        name: String,
        presort_and_args: Option<(String, Vec<Expr>)>,
        /// The name of the union-find function for this sort.
        /// Used in term encoding to canonicalize values during extraction.
        uf: Option<String>,
        /// Whether values of this sort can be unioned.
        /// Defaults to true for user-defined sorts.
        /// Set to false for relations and term tables that should not allow union.
        unionable: bool,
    },

    /// Egglog supports three types of functions
    ///
    /// A constructor models an egg-style user-defined datatype
    /// It can only be defined through the `datatype`/`datatype*` command
    /// or the `constructor` command
    ///
    /// A relation models a datalog-style mathematical relation
    /// It can only be defined through the `relation` command
    ///
    /// A custom function is a dictionary
    /// It can only be defined through the `function` command
    ///
    /// The `datatype` command declares a user-defined datatype.
    /// Datatypes can be unioned with [`Action::Union`] either
    /// at the top level or in the actions of a rule.
    /// This makes them equal in the implicit, global equality relation.
    ///
    /// Example:
    /// ```text
    /// (datatype Math
    ///   (Num i64)
    ///   (Var String)
    ///   (Add Math Math)
    ///   (Mul Math Math))
    /// ```
    ///
    /// defines a simple `Math` datatype with variants for numbers, named variables, addition and multiplication.
    ///
    /// Datatypes desugar directly to a [`Command::Sort`] and a [`Command::Constructor`] for each constructor.
    /// The code above becomes:
    /// ```text
    /// (sort Math)
    /// (constructor Num (i64) Math)
    /// (constructor Var (String) Math)
    /// (constructor Add (Math Math) Math)
    /// (constructor Mul (Math Math) Math)
    ///
    /// Datatypes are also known as algebraic data types, tagged unions and sum types.
    Datatype {
        span: Span,
        name: String,
        variants: Vec<Variant>,
    },
    Datatypes {
        span: Span,
        datatypes: Vec<(Span, String, Subdatatypes)>,
    },

    /// The `constructor` command defines a new constructor for a user-defined datatype
    /// Example:
    /// ```text
    /// (constructor Add (i64 i64) Math)
    /// ```
    ///
    Constructor {
        span: Span,
        name: String,
        schema: Schema,
        cost: Option<DefaultCost>,
        unextractable: bool,
        /// For view tables in proof encoding: the constructor to use for building
        /// terms from the first n-1 children during extraction.
        term_constructor: Option<String>,
    },

    /// The `relation` command declares a named relation
    /// Example:
    /// ```text
    /// (relation path (i64 i64))
    /// (relation edge (i64 i64))
    /// ```
    Relation {
        span: Span,
        name: String,
        inputs: Vec<String>,
    },

    /// The `function` command declare an egglog custom function, which is a database table with a
    /// a functional dependency (also called a primary key) on its inputs to one output.
    ///
    /// ```text
    /// (function <name:Ident> <schema:Schema> <cost:Cost>
    ///        (:on_merge <List<Action>>)?
    ///        (:merge <Expr>)?)
    ///```
    /// A function can have a `cost` for extraction.
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
    /// ```text
    /// (function LowerBound (Math) i64 :merge (max old new))
    /// ```
    ///
    /// Specifically, a custom function can also have an EqSort output type:
    ///
    /// ```text
    /// (function Add (i64 i64) Math)
    /// ```
    ///
    /// All functions can be `set`
    /// with [`Action::Set`].
    ///
    /// Output of a function, if being the EqSort type, can be unioned with [`Action::Union`]
    /// with another datatype of the same `sort`.
    ///
    Function {
        span: Span,
        name: String,
        schema: Schema,
        merge: Option<GenericExpr<Head, Leaf>>,
    },

    /// Using the `ruleset` command, defines a new
    /// ruleset that can be added to in [`Command::Rule`]s.
    /// Rulesets are used to group rules together
    /// so that they can be run together in a [`Schedule`].
    ///
    /// Example:
    /// Ruleset allows users to define a ruleset- a set of rules
    ///
    /// ```text
    /// (ruleset myrules)
    /// (rule ((edge x y))
    ///       ((path x y))
    ///       :ruleset myrules)
    /// (run myrules 2)
    /// ```
    AddRuleset(Span, String),
    /// Using the `combined-ruleset` command, construct another ruleset
    /// which runs all the rules in the given rulesets.
    /// This is useful for running multiple rulesets together.
    /// The combined ruleset also inherits any rules added to the individual rulesets
    /// after the combined ruleset is declared.
    ///
    /// Example:
    /// ```text
    /// (ruleset myrules1)
    /// (rule ((edge x y))
    ///       ((path x y))
    ///      :ruleset myrules1)
    /// (ruleset myrules2)
    /// (rule ((path x y) (edge y z))
    ///       ((path x z))
    ///       :ruleset myrules2)
    /// (combined-ruleset myrules-combined myrules1 myrules2)
    /// ```
    UnstableCombinedRuleset(Span, String, Vec<String>),
    /// ```text
    /// (rule <body:List<Fact>> <head:List<Action>>)
    /// ```
    ///
    /// defines an egglog rule.
    /// The rule matches a list of facts with respect to
    /// the global database, and runs the list of actions
    /// for each match.
    /// The matches are done *modulo equality*, meaning
    /// equal datatypes in the database are considered
    /// equal.
    ///
    /// Example:
    /// ```text
    /// (rule ((edge x y))
    ///       ((path x y)))
    ///
    /// (rule ((path x y) (edge y z))
    ///       ((path x z)))
    /// ```
    Rule {
        rule: GenericRule<Head, Leaf>,
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
    /// Add the `:subsume` flag to cause the left hand side to be subsumed after matching, which means it can
    /// no longer be matched in a rule, but can still be checked against (See [`Change`] for more details.)
    ///
    /// ```text
    /// (rewrite (Mul a 2) (bitshift-left a 1) :subsume)
    /// ```
    ///
    /// Desugars to:
    /// ```text
    /// (rule ((= lhs (Mul a 2)))
    ///       ((union lhs (bitshift-left a 1))
    ///        (subsume (Mul a 2))))
    /// ```
    Rewrite(String, GenericRewrite<Head, Leaf>, Subsume),
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
    BiRewrite(String, GenericRewrite<Head, Leaf>),
    /// Perform an [`Action`] on the global database
    /// (see documentation for [`Action`] for more details).
    /// Example:
    /// ```text
    /// (let xplusone (Add (Var "x") (Num 1)))
    /// ```
    Action(GenericAction<Head, Leaf>),
    /// `extract` a datatype from the egraph, choosing
    /// the smallest representative.
    /// By default, each constructor costs 1 to extract
    /// (common subexpressions are not shared in the cost
    /// model).
    Extract(Span, GenericExpr<Head, Leaf>, GenericExpr<Head, Leaf>),
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
    RunSchedule(GenericSchedule<Head, Leaf>),
    /// Print runtime statistics about rules
    /// and rulesets so far.
    PrintOverallStatistics(Span, Option<String>),
    /// The `check` command checks that the given facts
    /// match at least once in the current database.
    /// The list of facts is matched in the same way a [`Command::Rule`] is matched.
    ///
    /// Example:
    ///
    /// ```text
    /// (check (= (+ 1 2) 3))
    /// (check (<= 0 3) (>= 3 0))
    /// (fail (check (= 1 2)))
    /// ```
    ///
    /// prints
    ///
    /// ```text
    /// [INFO ] Checked.
    /// [INFO ] Checked.
    /// [ERROR] Check failed
    /// [INFO ] Command failed as expected.
    /// ```
    Check(Span, Vec<GenericFact<Head, Leaf>>),
    Prove(Span, Vec<GenericFact<Head, Leaf>>),
    ProveExists(Span, Head),
    /// Print out rows of a given function, extracting each of the elements of the function.
    /// Example:
    ///
    /// ```text
    /// (print-function Add 20)
    /// ```
    /// prints the first 20 rows of the `Add` function.
    ///
    /// ```text
    /// (print-function Add)
    /// ```
    /// prints all rows of the `Add` function.
    ///
    /// ```text
    /// (print-function Add :file "add.csv")
    /// ```
    /// prints all rows of the `Add` function to a CSV file.
    PrintFunction(
        Span,
        String,
        Option<usize>,
        Option<String>,
        PrintFunctionMode,
    ),
    /// Print out the number of rows in a function or all functions.
    PrintSize(Span, Option<String>),
    /// Input a CSV file directly into a function.
    Input {
        span: Span,
        name: String,
        file: String,
    },
    /// Extract and output a set of expressions to a file.
    Output {
        span: Span,
        file: String,
        exprs: Vec<GenericExpr<Head, Leaf>>,
    },
    /// `push` the current egraph `n` times so that it is saved.
    /// Later, the current database and rules can be restored using `pop`.
    Push(usize),
    /// `pop` the current egraph, restoring the previous one.
    /// The argument specifies how many egraphs to pop.
    Pop(Span, usize),
    /// Assert that a command fails with an error.
    Fail(Span, Box<GenericCommand<Head, Leaf>>),
    /// Include another egglog file directly as text and run it.
    Include(Span, String),
    /// User-defined command.
    UserDefined(Span, String, Vec<Expr>),
}

impl<Head, Leaf> Display for GenericCommand<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            GenericCommand::Rewrite(name, rewrite, subsume) => {
                rewrite.fmt_with_ruleset(f, name, false, *subsume)
            }
            GenericCommand::BiRewrite(name, rewrite) => {
                rewrite.fmt_with_ruleset(f, name, true, false)
            }
            GenericCommand::Datatype {
                span: _,
                name,
                variants,
            } => {
                write!(f, "(datatype {name} {})", ListDisplay(variants, " "))
            }
            GenericCommand::Action(a) => write!(f, "{a}"),
            GenericCommand::Extract(_span, expr, variants) => {
                write!(f, "(extract {expr} {variants})")
            }
            GenericCommand::Sort {
                name,
                presort_and_args: None,
                ..
            } => {
                write!(f, "(sort {name})")
            }
            GenericCommand::Sort {
                name,
                presort_and_args: Some((name2, args)),
                ..
            } => {
                write!(f, "(sort {name} ({name2} {}))", ListDisplay(args, " "))
            }
            GenericCommand::Function {
                span: _,
                name,
                schema,
                merge,
            } => {
                write!(f, "(function {name} {schema}")?;
                if let Some(merge) = &merge {
                    write!(f, " :merge {merge}")?;
                } else {
                    write!(f, " :no-merge")?;
                }
                write!(f, ")")
            }
            GenericCommand::Constructor {
                span: _,
                name,
                schema,
                cost,
                unextractable,
                term_constructor,
            } => {
                write!(f, "(constructor {name} {schema}")?;
                if let Some(cost) = cost {
                    write!(f, " :cost {cost}")?;
                }
                if *unextractable {
                    write!(f, " :unextractable")?;
                }
                if let Some(tc) = term_constructor {
                    write!(f, " :term-constructor {tc}")?;
                }
                write!(f, ")")
            }
            GenericCommand::Relation {
                span: _,
                name,
                inputs,
            } => {
                write!(f, "(relation {name} ({}))", ListDisplay(inputs, " "))
            }
            GenericCommand::AddRuleset(_span, name) => {
                write!(f, "(ruleset {name})")
            }
            GenericCommand::UnstableCombinedRuleset(_span, name, others) => {
                write!(
                    f,
                    "(unstable-combined-ruleset {name} {})",
                    ListDisplay(others, " ")
                )
            }
            GenericCommand::Rule { rule } => rule.fmt(f),
            GenericCommand::RunSchedule(sched) => write!(f, "(run-schedule {sched})"),
            GenericCommand::PrintOverallStatistics(_span, file) => match file {
                Some(file) => write!(f, "(print-stats :file {file})"),
                None => write!(f, "(print-stats)"),
            },
            GenericCommand::Check(_ann, facts) => {
                write!(f, "(check {})", ListDisplay(facts, "\n"))
            }
            GenericCommand::Prove(_span, facts) => {
                if facts.is_empty() {
                    write!(f, "(prove)")
                } else {
                    write!(f, "(prove {})", ListDisplay(facts, " "))
                }
            }
            GenericCommand::ProveExists(_span, constructor) => {
                write!(f, "(prove-exists {constructor})")
            }
            GenericCommand::Push(n) => write!(f, "(push {n})"),
            GenericCommand::Pop(_span, n) => write!(f, "(pop {n})"),
            GenericCommand::PrintFunction(_span, name, n, file, mode) => {
                write!(f, "(print-function {name}")?;
                if let Some(n) = n {
                    write!(f, " {n}")?;
                }
                if let Some(file) = file {
                    write!(f, " :file {file:?}")?;
                }
                match mode {
                    PrintFunctionMode::Default => {}
                    PrintFunctionMode::CSV => write!(f, " :mode csv")?,
                }
                write!(f, ")")
            }
            GenericCommand::PrintSize(_span, name) => {
                write!(f, "(print-size {})", ListDisplay(name, " "))
            }
            GenericCommand::Input {
                span: _,
                name,
                file,
            } => {
                write!(f, "(input {name} {file:?})")
            }
            GenericCommand::Output {
                span: _,
                file,
                exprs,
            } => write!(f, "(output {file:?} {})", ListDisplay(exprs, " ")),
            GenericCommand::Fail(_span, cmd) => write!(f, "(fail {cmd})"),
            GenericCommand::Include(_span, file) => write!(f, "(include {file:?})"),
            GenericCommand::Datatypes { span: _, datatypes } => {
                let datatypes: Vec<_> = datatypes
                    .iter()
                    .map(|(_, name, variants)| match variants {
                        Subdatatypes::Variants(variants) => {
                            format!("({name} {})", ListDisplay(variants, " "))
                        }
                        Subdatatypes::NewSort(head, args) => {
                            format!("(sort {name} ({head} {}))", ListDisplay(args, " "))
                        }
                    })
                    .collect();
                write!(f, "(datatype* {})", ListDisplay(datatypes, " "))
            }
            GenericCommand::UserDefined(_span, name, exprs) => {
                write!(f, "({name} {})", ListDisplay(exprs, " "))
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IdentSort {
    pub ident: String,
    pub sort: String,
}

impl Display for IdentSort {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "({} {})", self.ident, self.sort)
    }
}

pub type RunConfig = GenericRunConfig<String, String>;
pub(crate) type ResolvedRunConfig = GenericRunConfig<ResolvedCall, ResolvedVar>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GenericRunConfig<Head, Leaf> {
    pub ruleset: String,
    pub until: Option<Vec<GenericFact<Head, Leaf>>>,
}

impl<Head, Leaf> GenericRunConfig<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub fn visit_exprs(
        self,
        f: &mut impl FnMut(GenericExpr<Head, Leaf>) -> GenericExpr<Head, Leaf>,
    ) -> Self {
        Self {
            ruleset: self.ruleset,
            until: self
                .until
                .map(|until| until.into_iter().map(|fact| fact.visit_exprs(f)).collect()),
        }
    }

    /// Remaps every head and leaf symbol in the run configuration.
    pub fn map_symbols<Head2, Leaf2>(
        self,
        head: &mut impl FnMut(Head) -> Head2,
        leaf: &mut impl FnMut(Leaf) -> Leaf2,
    ) -> GenericRunConfig<Head2, Leaf2>
    where
        Head2: Clone + Display,
        Leaf2: Clone + PartialEq + Eq + Display + Hash,
    {
        GenericRunConfig {
            ruleset: self.ruleset,
            until: self.until.map(|facts| {
                facts
                    .into_iter()
                    .map(|fact| fact.map_symbols(head, leaf))
                    .collect()
            }),
        }
    }

    /// Applies `fun` to string-valued symbols within the run configuration.
    pub fn map_string_symbols(
        self,
        fun: &mut impl FnMut(String) -> String,
    ) -> GenericRunConfig<Head, Leaf> {
        GenericRunConfig {
            ruleset: fun(self.ruleset),
            until: self.until,
        }
    }

    pub fn make_unresolved(self) -> GenericRunConfig<String, String> {
        let mut map_head = |h: Head| h.to_string();
        let mut map_leaf = |l: Leaf| l.to_string();
        self.map_symbols(&mut map_head, &mut map_leaf)
    }
}

impl<Head: Display, Leaf: Display> Display for GenericRunConfig<Head, Leaf>
where
    Head: Display,
    Leaf: Display,
{
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "(run")?;
        if !self.ruleset.is_empty() {
            write!(f, " {}", self.ruleset)?;
        }
        if let Some(until) = &self.until {
            write!(f, " :until {}", ListDisplay(until, " "))?;
        }
        write!(f, ")")
    }
}

pub type FunctionDecl = GenericFunctionDecl<String, String>;
pub(crate) type ResolvedFunctionDecl = GenericFunctionDecl<ResolvedCall, ResolvedVar>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FunctionSubtype {
    Constructor,
    // TODO rename to [`Function`] to match surface syntax and terminology.
    Custom,
}

impl Display for FunctionSubtype {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            FunctionSubtype::Constructor => write!(f, "constructor"),
            FunctionSubtype::Custom => write!(f, "function"),
        }
    }
}

/// Represents the declaration of a function
/// directly parsed from source syntax.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GenericFunctionDecl<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub name: String,
    pub subtype: FunctionSubtype,
    /// Untyped schema
    pub schema: Schema,
    /// Resolved schema after typechecking is stored here, otherwise "".
    pub resolved_schema: Head,
    pub merge: Option<GenericExpr<Head, Leaf>>,
    pub cost: Option<DefaultCost>,
    pub unextractable: bool,
    /// Globals are desugared to functions, with this flag set to true.
    /// This is used by visualization to handle globals differently.
    pub let_binding: bool,
    pub span: Span,
    /// For view tables in proof encoding: the constructor to use for building
    /// terms from the first n-1 children during extraction.
    pub term_constructor: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Variant {
    pub span: Span,
    pub name: String,
    pub types: Vec<String>,
    pub cost: Option<DefaultCost>,
    pub unextractable: bool,
}

impl Display for Variant {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "({}", self.name)?;
        if !self.types.is_empty() {
            write!(f, " {}", ListDisplay(&self.types, " "))?;
        }
        if let Some(cost) = self.cost {
            write!(f, " :cost {cost}")?;
        }
        write!(f, ")")
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Schema {
    pub input: Vec<String>,
    pub output: String,
}

impl Display for Schema {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "({}) {}", ListDisplay(&self.input, " "), self.output)
    }
}

impl Schema {
    pub fn new(input: Vec<String>, output: String) -> Self {
        Self { input, output }
    }
}

impl FunctionDecl {
    /// Constructs a `function`
    pub fn function(
        span: Span,
        name: String,
        schema: Schema,
        merge: Option<GenericExpr<String, String>>,
    ) -> Self {
        Self {
            name,
            subtype: FunctionSubtype::Custom,
            schema,
            resolved_schema: String::new(),
            merge,
            cost: None,
            unextractable: true,
            let_binding: false,
            span,
            term_constructor: None,
        }
    }

    /// Constructs a `constructor`
    pub fn constructor(
        span: Span,
        name: String,
        schema: Schema,
        cost: Option<DefaultCost>,
        unextractable: bool,
    ) -> Self {
        Self {
            name,
            subtype: FunctionSubtype::Constructor,
            resolved_schema: String::new(),
            schema,
            merge: None,
            cost,
            unextractable,
            let_binding: false,
            span,
            term_constructor: None,
        }
    }
}

impl<Head, Leaf> GenericFunctionDecl<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub fn visit_exprs(
        self,
        f: &mut impl FnMut(GenericExpr<Head, Leaf>) -> GenericExpr<Head, Leaf>,
    ) -> GenericFunctionDecl<Head, Leaf> {
        GenericFunctionDecl {
            name: self.name,
            subtype: self.subtype,
            schema: self.schema,
            resolved_schema: self.resolved_schema,
            merge: self.merge.map(|expr| expr.visit_exprs(f)),
            cost: self.cost,
            unextractable: self.unextractable,
            let_binding: self.let_binding,
            span: self.span,
            term_constructor: self.term_constructor,
        }
    }
}

pub type Fact = GenericFact<String, String>;
pub type ResolvedFact = GenericFact<ResolvedCall, ResolvedVar>;
pub(crate) type MappedFact<Head, Leaf> = GenericFact<CorrespondingVar<Head, Leaf>, Leaf>;

pub struct Facts<Head, Leaf>(pub Vec<GenericFact<Head, Leaf>>);

impl<Head, Leaf> Facts<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
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
    ) -> (Query<HeadOrEq<Head>, Leaf>, Vec<MappedFact<Head, Leaf>>) {
        let mut atoms = vec![];
        let mut new_body = vec![];

        for fact in self.0.iter() {
            match fact {
                GenericFact::Eq(span, e1, e2) => {
                    let mut to_equate = vec![];
                    let mut process = |expr: &GenericExpr<Head, Leaf>| {
                        let (child_atoms, expr) = expr.to_query(typeinfo, fresh_gen);
                        atoms.extend(child_atoms);
                        to_equate.push(expr.get_corresponding_var_or_lit(typeinfo));
                        expr
                    };
                    let e1 = process(e1);
                    let e2 = process(e2);
                    atoms.push(GenericAtom {
                        span: span.clone(),
                        head: HeadOrEq::Eq,
                        args: to_equate,
                    });
                    new_body.push(GenericFact::Eq(span.clone(), e1, e2));
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CorrespondingVar<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub head: Head,
    pub to: Leaf,
}

impl<Head, Leaf> CorrespondingVar<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub fn new(head: Head, leaf: Leaf) -> Self {
        Self { head, to: leaf }
    }
}

impl<Head, Leaf> Display for CorrespondingVar<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{} -> {}", self.head, self.to)
    }
}
pub type Action = GenericAction<String, String>;
pub(crate) type MappedAction = GenericAction<CorrespondingVar<String, String>, String>;
pub(crate) type ResolvedAction = GenericAction<ResolvedCall, ResolvedVar>;

pub type Actions = GenericActions<String, String>;
pub(crate) type ResolvedActions = GenericActions<ResolvedCall, ResolvedVar>;
pub(crate) type MappedActions<Head, Leaf> = GenericActions<CorrespondingVar<Head, Leaf>, Leaf>;

pub type Rule = GenericRule<String, String>;
pub(crate) type ResolvedRule = GenericRule<ResolvedCall, ResolvedVar>;

pub type Rewrite = GenericRewrite<String, String>;

#[derive(Clone, Debug)]
pub struct GenericRewrite<Head, Leaf> {
    pub span: Span,
    pub lhs: GenericExpr<Head, Leaf>,
    pub rhs: GenericExpr<Head, Leaf>,
    pub conditions: Vec<GenericFact<Head, Leaf>>,
}

impl<Head, Leaf> GenericRewrite<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    /// Remaps every head and leaf symbol in the rewrite, including the optional conditions.
    pub fn map_symbols<Head2, Leaf2>(
        self,
        head: &mut impl FnMut(Head) -> Head2,
        leaf: &mut impl FnMut(Leaf) -> Leaf2,
    ) -> GenericRewrite<Head2, Leaf2>
    where
        Head2: Clone + Display,
        Leaf2: Clone + PartialEq + Eq + Display + Hash,
    {
        GenericRewrite {
            span: self.span,
            lhs: self.lhs.map_symbols(head, leaf),
            rhs: self.rhs.map_symbols(head, leaf),
            conditions: self
                .conditions
                .into_iter()
                .map(|fact| fact.map_symbols(head, leaf))
                .collect(),
        }
    }

    pub fn make_unresolved(self) -> GenericRewrite<String, String> {
        let mut map_head = |h: Head| h.to_string();
        let mut map_leaf = |l: Leaf| l.to_string();
        self.map_symbols(&mut map_head, &mut map_leaf)
    }
}

impl<Head: Display, Leaf: Display> GenericRewrite<Head, Leaf> {
    /// Converts the rewrite into an s-expression.
    pub fn fmt_with_ruleset(
        &self,
        f: &mut Formatter,
        ruleset: &str,
        is_bidirectional: bool,
        subsume: bool,
    ) -> std::fmt::Result {
        let direction = if is_bidirectional {
            "birewrite"
        } else {
            "rewrite"
        };
        write!(f, "({direction} {} {}", self.lhs, self.rhs)?;
        if subsume {
            write!(f, " :subsume")?;
        }
        if !self.conditions.is_empty() {
            write!(f, " :when ({})", ListDisplay(&self.conditions, " "))?;
        }
        if !ruleset.is_empty() {
            write!(f, " :ruleset {ruleset}")?;
        }
        write!(f, ")")
    }
}

pub(crate) trait MappedExprExt<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    fn get_corresponding_var_or_lit(&self, typeinfo: &TypeInfo) -> GenericAtomTerm<Leaf>;
}

impl<Head, Leaf> MappedExprExt<Head, Leaf> for MappedExpr<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    fn get_corresponding_var_or_lit(&self, typeinfo: &TypeInfo) -> GenericAtomTerm<Leaf> {
        // Note: need typeinfo to resolve whether a symbol is a global or not
        // This is error-prone and the complexities can be avoided by treating globals
        // as nullary functions.
        match self {
            GenericExpr::Var(span, v) => {
                if typeinfo.is_global(&v.to_string()) {
                    GenericAtomTerm::Global(span.clone(), v.clone())
                } else {
                    GenericAtomTerm::Var(span.clone(), v.clone())
                }
            }
            GenericExpr::Lit(span, lit) => GenericAtomTerm::Literal(span.clone(), lit.clone()),
            GenericExpr::Call(span, head, _) => GenericAtomTerm::Var(span.clone(), head.to.clone()),
        }
    }
}

impl<Head, Leaf> GenericCommand<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    /// The current egglog AST has strings even when resolved.
    /// We map over those strings with this function, used by sanitize_internal_symbols.
    pub fn map_string_symbols(
        self,
        fun: &mut impl FnMut(String) -> String,
    ) -> GenericCommand<Head, Leaf>
    where
        Head: Clone + Display,
        Leaf: Clone + PartialEq + Eq + Display + Hash,
    {
        match self {
            GenericCommand::Sort {
                span,
                name,
                presort_and_args,
                uf,
                unionable,
            } => GenericCommand::Sort {
                span,
                name: fun(name),
                presort_and_args,
                uf,
                unionable,
            },
            GenericCommand::Datatype {
                span,
                name,
                variants,
            } => GenericCommand::Datatype {
                span,
                name: fun(name),
                variants: variants
                    .into_iter()
                    .map(|variant| Variant {
                        span: variant.span,
                        name: fun(variant.name),
                        types: variant.types.into_iter().map(&mut *fun).collect(),
                        cost: variant.cost,
                        unextractable: variant.unextractable,
                    })
                    .collect(),
            },
            GenericCommand::Datatypes { span, datatypes } => GenericCommand::Datatypes {
                span,
                datatypes: datatypes
                    .into_iter()
                    .map(|(span, name, variants)| {
                        let new_name = fun(name);
                        let new_variants = match variants {
                            Subdatatypes::Variants(variants) => Subdatatypes::Variants(
                                variants
                                    .into_iter()
                                    .map(|variant| Variant {
                                        span: variant.span,
                                        name: fun(variant.name),
                                        // Redundant closure helps with type inference here
                                        #[allow(clippy::redundant_closure)]
                                        types: variant
                                            .types
                                            .into_iter()
                                            .map(|ty| fun(ty))
                                            .collect(),
                                        cost: variant.cost,
                                        unextractable: variant.unextractable,
                                    })
                                    .collect(),
                            ),
                            Subdatatypes::NewSort(head, args) => {
                                Subdatatypes::NewSort(fun(head), args)
                            }
                        };
                        (span, new_name, new_variants)
                    })
                    .collect(),
            },
            GenericCommand::Constructor {
                span,
                name,
                schema,
                cost,
                unextractable,
                term_constructor,
            } => GenericCommand::Constructor {
                span,
                name: fun(name),
                schema: Schema {
                    input: schema.input.into_iter().map(&mut *fun).collect(),
                    output: fun(schema.output),
                },
                cost,
                unextractable,
                term_constructor: term_constructor.map(&mut *fun),
            },
            GenericCommand::Relation { span, name, inputs } => GenericCommand::Relation {
                span,
                name: fun(name),
                inputs: inputs.into_iter().map(&mut *fun).collect(),
            },
            GenericCommand::Function {
                span,
                name,
                schema,
                merge,
            } => GenericCommand::Function {
                span,
                name: fun(name),
                schema: Schema {
                    input: schema.input.into_iter().map(&mut *fun).collect(),
                    output: fun(schema.output),
                },
                merge,
            },
            GenericCommand::AddRuleset(span, name) => GenericCommand::AddRuleset(span, fun(name)),
            GenericCommand::UnstableCombinedRuleset(span, name, others) => {
                GenericCommand::UnstableCombinedRuleset(
                    span,
                    fun(name),
                    others.into_iter().map(&mut *fun).collect(),
                )
            }
            GenericCommand::Rule { rule } => {
                let rule = GenericRule {
                    span: rule.span,
                    name: fun(rule.name),
                    ruleset: fun(rule.ruleset),
                    head: rule.head,
                    body: rule.body,
                };
                GenericCommand::Rule { rule }
            }
            GenericCommand::Rewrite(name, rewrite, subsume) => {
                GenericCommand::Rewrite(fun(name), rewrite, subsume)
            }
            GenericCommand::BiRewrite(name, rewrite) => {
                GenericCommand::BiRewrite(fun(name), rewrite)
            }
            GenericCommand::Action(action) => GenericCommand::Action(action),
            GenericCommand::Extract(span, expr, variants) => {
                GenericCommand::Extract(span, expr, variants)
            }
            GenericCommand::RunSchedule(schedule) => {
                GenericCommand::RunSchedule(schedule.map_string_symbols(fun))
            }
            GenericCommand::PrintOverallStatistics(span, file) => {
                GenericCommand::PrintOverallStatistics(span, file)
            }
            GenericCommand::Check(span, facts) => GenericCommand::Check(span, facts),
            GenericCommand::Prove(span, facts) => GenericCommand::Prove(span, facts),
            GenericCommand::ProveExists(span, constructor) => {
                GenericCommand::ProveExists(span, constructor)
            }
            GenericCommand::PrintFunction(span, name, n, file, mode) => {
                GenericCommand::PrintFunction(span, fun(name), n, file, mode)
            }
            GenericCommand::PrintSize(span, name) => GenericCommand::PrintSize(span, name.map(fun)),
            GenericCommand::Input { span, name, file } => GenericCommand::Input {
                span,
                name: fun(name),
                file,
            },
            GenericCommand::Output { span, file, exprs } => {
                GenericCommand::Output { span, file, exprs }
            }
            GenericCommand::Push(n) => GenericCommand::Push(n),
            GenericCommand::Pop(span, n) => GenericCommand::Pop(span, n),
            GenericCommand::Fail(span, cmd) => {
                GenericCommand::Fail(span, Box::new(cmd.map_string_symbols(fun)))
            }
            GenericCommand::Include(span, file) => GenericCommand::Include(span, file),
            GenericCommand::UserDefined(span, name, exprs) => {
                GenericCommand::UserDefined(span, name, exprs)
            }
        }
    }

    /// Applies `f` to all expressions in the command, bottom-up.
    pub fn visit_exprs(
        self,
        f: &mut impl FnMut(GenericExpr<Head, Leaf>) -> GenericExpr<Head, Leaf>,
    ) -> Self {
        match self {
            GenericCommand::Function {
                span,
                name,
                schema,
                merge,
            } => GenericCommand::Function {
                span,
                name,
                schema,
                merge: merge.map(|e| e.visit_exprs(f)),
            },
            GenericCommand::Rule { rule } => GenericCommand::Rule {
                rule: rule.visit_exprs(f),
            },
            GenericCommand::Rewrite(name, rewrite, subsume) => GenericCommand::Rewrite(
                name,
                GenericRewrite {
                    span: rewrite.span,
                    lhs: rewrite.lhs.visit_exprs(f),
                    rhs: rewrite.rhs.visit_exprs(f),
                    conditions: rewrite
                        .conditions
                        .into_iter()
                        .map(|fact| fact.visit_exprs(f))
                        .collect(),
                },
                subsume,
            ),
            GenericCommand::BiRewrite(name, rewrite) => GenericCommand::BiRewrite(
                name,
                GenericRewrite {
                    span: rewrite.span,
                    lhs: rewrite.lhs.visit_exprs(f),
                    rhs: rewrite.rhs.visit_exprs(f),
                    conditions: rewrite
                        .conditions
                        .into_iter()
                        .map(|fact| fact.visit_exprs(f))
                        .collect(),
                },
            ),
            GenericCommand::Action(action) => GenericCommand::Action(action.visit_exprs(f)),
            GenericCommand::Extract(span, expr1, expr2) => {
                GenericCommand::Extract(span, expr1.visit_exprs(f), expr2.visit_exprs(f))
            }
            GenericCommand::Check(span, facts) => GenericCommand::Check(
                span,
                facts.into_iter().map(|fact| fact.visit_exprs(f)).collect(),
            ),
            GenericCommand::Prove(span, facts) => GenericCommand::Prove(
                span,
                facts.into_iter().map(|fact| fact.visit_exprs(f)).collect(),
            ),
            GenericCommand::Output { span, file, exprs } => GenericCommand::Output {
                span,
                file,
                exprs: exprs.into_iter().map(|e| e.visit_exprs(f)).collect(),
            },
            GenericCommand::RunSchedule(schedule) => {
                GenericCommand::RunSchedule(schedule.visit_exprs(f))
            }
            GenericCommand::Fail(span, cmd) => {
                GenericCommand::Fail(span, Box::new(cmd.visit_exprs(f)))
            }
            // All other commands don't contain expressions
            cmd => cmd,
        }
    }

    /// Remaps every head and leaf symbol contained in the command.
    pub fn map_symbols<Head2, Leaf2>(
        self,
        head: &mut impl FnMut(Head) -> Head2,
        leaf: &mut impl FnMut(Leaf) -> Leaf2,
    ) -> GenericCommand<Head2, Leaf2>
    where
        Head2: Clone + Display,
        Leaf2: Clone + PartialEq + Eq + Display + Hash,
    {
        match self {
            GenericCommand::Sort {
                span,
                name,
                presort_and_args,
                uf,
                unionable,
            } => GenericCommand::Sort {
                span,
                name,
                presort_and_args,
                uf,
                unionable,
            },
            GenericCommand::Datatype {
                span,
                name,
                variants,
            } => GenericCommand::Datatype {
                span,
                name,
                variants,
            },
            GenericCommand::Datatypes { span, datatypes } => {
                GenericCommand::Datatypes { span, datatypes }
            }
            GenericCommand::Constructor {
                span,
                name,
                schema,
                cost,
                unextractable,
                term_constructor,
            } => GenericCommand::Constructor {
                span,
                name,
                schema,
                cost,
                unextractable,
                term_constructor,
            },
            GenericCommand::Relation { span, name, inputs } => {
                GenericCommand::Relation { span, name, inputs }
            }
            GenericCommand::Function {
                span,
                name,
                schema,
                merge,
            } => GenericCommand::Function {
                span,
                name,
                schema,
                merge: merge.map(|expr| expr.map_symbols(head, leaf)),
            },
            GenericCommand::AddRuleset(span, name) => GenericCommand::AddRuleset(span, name),
            GenericCommand::UnstableCombinedRuleset(span, name, others) => {
                GenericCommand::UnstableCombinedRuleset(span, name, others)
            }
            GenericCommand::Rule { rule } => GenericCommand::Rule {
                rule: rule.map_symbols(head, leaf),
            },
            GenericCommand::Rewrite(name, rewrite, subsume) => {
                GenericCommand::Rewrite(name, rewrite.map_symbols(head, leaf), subsume)
            }
            GenericCommand::BiRewrite(name, rewrite) => {
                GenericCommand::BiRewrite(name, rewrite.map_symbols(head, leaf))
            }
            GenericCommand::Action(action) => {
                GenericCommand::Action(action.map_symbols(head, leaf))
            }
            GenericCommand::Extract(span, expr, variants) => GenericCommand::Extract(
                span,
                expr.map_symbols(head, leaf),
                variants.map_symbols(head, leaf),
            ),
            GenericCommand::RunSchedule(schedule) => {
                GenericCommand::RunSchedule(schedule.map_symbols(head, leaf))
            }
            GenericCommand::PrintOverallStatistics(span, file) => {
                GenericCommand::PrintOverallStatistics(span, file)
            }
            GenericCommand::Check(span, facts) => GenericCommand::Check(
                span,
                facts
                    .into_iter()
                    .map(|fact| fact.map_symbols(head, leaf))
                    .collect(),
            ),
            GenericCommand::Prove(span, facts) => GenericCommand::Prove(
                span,
                facts
                    .into_iter()
                    .map(|fact| fact.map_symbols(head, leaf))
                    .collect(),
            ),
            GenericCommand::ProveExists(span, constructor) => {
                GenericCommand::ProveExists(span, head(constructor))
            }
            GenericCommand::PrintFunction(span, name, n, file, mode) => {
                GenericCommand::PrintFunction(span, name, n, file, mode)
            }
            GenericCommand::PrintSize(span, name) => GenericCommand::PrintSize(span, name),
            GenericCommand::Input { span, name, file } => {
                GenericCommand::Input { span, name, file }
            }
            GenericCommand::Output { span, file, exprs } => GenericCommand::Output {
                span,
                file,
                exprs: exprs
                    .into_iter()
                    .map(|expr| expr.map_symbols(head, leaf))
                    .collect(),
            },
            GenericCommand::Push(n) => GenericCommand::Push(n),
            GenericCommand::Pop(span, n) => GenericCommand::Pop(span, n),
            GenericCommand::Fail(span, cmd) => {
                GenericCommand::Fail(span, Box::new(cmd.map_symbols(head, leaf)))
            }
            GenericCommand::Include(span, file) => GenericCommand::Include(span, file),
            GenericCommand::UserDefined(span, name, exprs) => {
                GenericCommand::UserDefined(span, name, exprs)
            }
        }
    }

    /// Makes the command unresolved by converting all Head and Leaf types to String.
    pub fn make_unresolved(self) -> GenericCommand<String, String> {
        let mut map_head = |h: Head| h.to_string();
        let mut map_leaf = |l: Leaf| l.to_string();
        self.map_symbols(&mut map_head, &mut map_leaf)
    }

    pub fn visit_actions(
        self,
        f: &mut impl FnMut(GenericAction<Head, Leaf>) -> GenericAction<Head, Leaf>,
    ) -> Self {
        match self {
            GenericCommand::Rule { rule } => GenericCommand::Rule {
                rule: rule.visit_actions(f),
            },
            GenericCommand::Action(action) => GenericCommand::Action(f(action)),
            GenericCommand::Fail(span, cmd) => {
                GenericCommand::Fail(span, Box::new(cmd.visit_actions(f)))
            }
            other => other,
        }
    }
}

/// Sanitizes internal names so they do not contain any internal characters.
/// This enables printing desugared egglog in a way that can be re-parsed.
pub fn sanitize_internal_names<Head, Leaf>(
    program: &[GenericCommand<Head, Leaf>],
) -> Vec<GenericCommand<String, String>>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    // first convert to unresolved
    let unresolved = program
        .iter()
        .map(|cmd| cmd.clone().make_unresolved())
        .collect::<Vec<_>>();
    // now count the max number of underscores in any name
    let mut max_underscores = 0;
    let mut max_underscores2 = 0;
    for cmd in &unresolved {
        cmd.clone().map_symbols(
            &mut |h: String| {
                let count = h.matches(INTERNAL_SYMBOL_PREFIX).count();
                if count > max_underscores {
                    max_underscores = count;
                }
                h
            },
            &mut |l: String| {
                let count = l.matches(INTERNAL_SYMBOL_PREFIX).count();
                if count > max_underscores2 {
                    max_underscores2 = count;
                }
                l
            },
        );
        cmd.clone().map_string_symbols(&mut |s: String| {
            let count = s.matches(INTERNAL_SYMBOL_PREFIX).count();
            if count > max_underscores {
                max_underscores = count;
            }
            s
        });
    }
    let replacement_head = "_".repeat(max_underscores + 1);
    let replacement_leaf = "_".repeat(max_underscores2 + 1);
    // now replace INTERNAL_SYMBOL_PREFIX with replacement
    unresolved
        .into_iter()
        .map(|cmd| {
            let cmd = cmd.map_symbols(
                &mut |h: String| h.replace(INTERNAL_SYMBOL_PREFIX, &replacement_head),
                &mut |l: String| l.replace(INTERNAL_SYMBOL_PREFIX, &replacement_leaf),
            );
            cmd.map_string_symbols(&mut |s: String| {
                s.replace(INTERNAL_SYMBOL_PREFIX, &replacement_head)
            })
        })
        .collect()
}
