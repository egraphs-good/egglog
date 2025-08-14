pub mod check_shadowing;
pub mod desugar;
mod expr;
mod parse;
pub mod remove_globals;

use crate::core::{GenericAtom, GenericAtomTerm, HeadOrEq, Query, ResolvedCall, ResolvedCoreRule};
use crate::*;
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
    SetOption {
        name: String,
        value: GenericExpr<Head, Leaf>,
    },
    Sort(
        Span,
        String,
        Option<(String, Vec<GenericExpr<String, String>>)>,
    ),
    Function(GenericFunctionDecl<Head, Leaf>),
    AddRuleset(Span, String),
    UnstableCombinedRuleset(Span, String, Vec<String>),
    NormRule {
        name: String,
        ruleset: String,
        rule: GenericRule<Head, Leaf>,
    },
    CoreAction(GenericAction<Head, Leaf>),
    Extract(Span, GenericExpr<Head, Leaf>, GenericExpr<Head, Leaf>),
    RunSchedule(GenericSchedule<Head, Leaf>),
    PrintOverallStatistics,
    Check(Span, Vec<GenericFact<Head, Leaf>>),
    PrintFunction(
        Span,
        String,
        Option<usize>,
        Option<String>,
        PrintFunctionMode,
    ),
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
            GenericNCommand::SetOption { name, value } => GenericCommand::SetOption {
                name: name.clone(),
                value: value.clone(),
            },
            GenericNCommand::Sort(span, name, params) => {
                GenericCommand::Sort(span.clone(), name.clone(), params.clone())
            }
            GenericNCommand::Function(f) => match f.subtype {
                FunctionSubtype::Constructor => GenericCommand::Constructor {
                    span: f.span.clone(),
                    name: f.name.clone(),
                    schema: f.schema.clone(),
                    cost: f.cost,
                    unextractable: f.unextractable,
                },
                FunctionSubtype::Relation => GenericCommand::Relation {
                    span: f.span.clone(),
                    name: f.name.clone(),
                    inputs: f.schema.input.clone(),
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
            GenericNCommand::NormRule {
                name,
                ruleset,
                rule,
            } => GenericCommand::Rule {
                name: name.clone(),
                ruleset: ruleset.clone(),
                rule: rule.clone(),
            },
            GenericNCommand::RunSchedule(schedule) => GenericCommand::RunSchedule(schedule.clone()),
            GenericNCommand::PrintOverallStatistics => GenericCommand::PrintOverallStatistics,
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

    pub fn visit_exprs(
        self,
        f: &mut impl FnMut(GenericExpr<Head, Leaf>) -> GenericExpr<Head, Leaf>,
    ) -> Self {
        match self {
            GenericNCommand::SetOption { name, value } => GenericNCommand::SetOption {
                name,
                value: f(value.clone()),
            },
            GenericNCommand::Sort(span, name, params) => GenericNCommand::Sort(span, name, params),
            GenericNCommand::Function(func) => GenericNCommand::Function(func.visit_exprs(f)),
            GenericNCommand::AddRuleset(span, name) => GenericNCommand::AddRuleset(span, name),
            GenericNCommand::UnstableCombinedRuleset(span, name, rulesets) => {
                GenericNCommand::UnstableCombinedRuleset(span, name, rulesets)
            }
            GenericNCommand::NormRule {
                name,
                ruleset,
                rule,
            } => GenericNCommand::NormRule {
                name,
                ruleset,
                rule: rule.visit_exprs(f),
            },
            GenericNCommand::RunSchedule(schedule) => {
                GenericNCommand::RunSchedule(schedule.visit_exprs(f))
            }
            GenericNCommand::PrintOverallStatistics => GenericNCommand::PrintOverallStatistics,
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

/// A [`Command`] is the top-level construct in egglog.
/// It includes defining rules, declaring functions,
/// adding to tables, and running rules (via a [`Schedule`]).
#[derive(Debug, Clone)]
pub enum GenericCommand<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    /// Egglog supports several *experimental* options
    /// that can be set using the `set-option` command.
    ///
    /// Options supported include:
    /// - "interactive_mode" (default: false): when enabled, egglog prints "(done)" after each command, allowing an external
    ///   tool to know when each command has finished running.
    SetOption {
        name: String,
        value: GenericExpr<Head, Leaf>,
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
    Sort(Span, String, Option<(String, Vec<Expr>)>),

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
        name: String,
        ruleset: String,
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
    PrintOverallStatistics,
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
            GenericCommand::SetOption { name, value } => write!(f, "(set-option {name} {value})"),
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
            } => write!(f, "(datatype {name} {})", ListDisplay(variants, " ")),
            GenericCommand::Action(a) => write!(f, "{a}"),
            GenericCommand::Extract(_span, expr, variants) => {
                write!(f, "(extract {expr} {variants})")
            }
            GenericCommand::Sort(_span, name, None) => write!(f, "(sort {name})"),
            GenericCommand::Sort(_span, name, Some((name2, args))) => {
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
            } => {
                write!(f, "(constructor {name} {schema}")?;
                if let Some(cost) = cost {
                    write!(f, " :cost {cost}")?;
                }
                if *unextractable {
                    write!(f, " :unextractable")?;
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
            GenericCommand::AddRuleset(_span, name) => write!(f, "(ruleset {name})"),
            GenericCommand::UnstableCombinedRuleset(_span, name, others) => {
                write!(
                    f,
                    "(unstable-combined-ruleset {name} {})",
                    ListDisplay(others, " ")
                )
            }
            GenericCommand::Rule {
                ruleset,
                name,
                rule,
            } => rule.fmt_with_ruleset(f, ruleset, name),
            GenericCommand::RunSchedule(sched) => write!(f, "(run-schedule {sched})"),
            GenericCommand::PrintOverallStatistics => write!(f, "(print-stats)"),
            GenericCommand::Check(_ann, facts) => {
                write!(f, "(check {})", ListDisplay(facts, "\n"))
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
            } => write!(f, "(input {name} {file:?})"),
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
    Relation,
    Custom,
}

impl Display for FunctionSubtype {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            FunctionSubtype::Constructor => write!(f, "Constructor"),
            FunctionSubtype::Relation => write!(f, "Relation"),
            FunctionSubtype::Custom => write!(f, "CustomFunction"),
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
    pub schema: Schema,
    pub merge: Option<GenericExpr<Head, Leaf>>,
    pub cost: Option<DefaultCost>,
    pub unextractable: bool,
    /// Globals are desugared to functions, with this flag set to true.
    /// This is used by visualization to handle globals differently.
    pub ignore_viz: bool,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Variant {
    pub span: Span,
    pub name: String,
    pub types: Vec<String>,
    pub cost: Option<DefaultCost>,
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
            merge,
            cost: None,
            unextractable: true,
            ignore_viz: false,
            span,
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
            schema,
            merge: None,
            cost,
            unextractable,
            ignore_viz: false,
            span,
        }
    }

    /// Constructs a `relation`
    pub fn relation(span: Span, name: String, input: Vec<String>) -> Self {
        Self {
            name,
            subtype: FunctionSubtype::Relation,
            schema: Schema {
                input,
                output: String::from("Unit"),
            },
            merge: None,
            cost: None,
            unextractable: true,
            ignore_viz: false,
            span,
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
            merge: self.merge.map(|expr| expr.visit_exprs(f)),
            cost: self.cost,
            unextractable: self.unextractable,
            ignore_viz: self.ignore_viz,
            span: self.span,
        }
    }
}

pub type Fact = GenericFact<String, String>;
pub(crate) type ResolvedFact = GenericFact<ResolvedCall, ResolvedVar>;
pub(crate) type MappedFact<Head, Leaf> = GenericFact<CorrespondingVar<Head, Leaf>, Leaf>;

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

impl<Head: Display, Leaf: Display> Display for GenericFact<Head, Leaf> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            GenericFact::Eq(_, e1, e2) => write!(f, "(= {e1} {e2})"),
            GenericFact::Fact(expr) => write!(f, "{expr}"),
        }
    }
}

impl<Head, Leaf> GenericFact<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub(crate) fn visit_exprs(
        self,
        f: &mut impl FnMut(GenericExpr<Head, Leaf>) -> GenericExpr<Head, Leaf>,
    ) -> GenericFact<Head, Leaf> {
        match self {
            GenericFact::Eq(span, e1, e2) => {
                GenericFact::Eq(span, e1.visit_exprs(f), e2.visit_exprs(f))
            }
            GenericFact::Fact(expr) => GenericFact::Fact(expr.visit_exprs(f)),
        }
    }

    pub(crate) fn map_exprs<Head2, Leaf2>(
        &self,
        f: &mut impl FnMut(&GenericExpr<Head, Leaf>) -> GenericExpr<Head2, Leaf2>,
    ) -> GenericFact<Head2, Leaf2> {
        match self {
            GenericFact::Eq(span, e1, e2) => GenericFact::Eq(span.clone(), f(e1), f(e2)),
            GenericFact::Fact(expr) => GenericFact::Fact(f(expr)),
        }
    }

    pub(crate) fn subst<Leaf2, Head2>(
        &self,
        subst_leaf: &mut impl FnMut(&Span, &Leaf) -> GenericExpr<Head2, Leaf2>,
        subst_head: &mut impl FnMut(&Head) -> Head2,
    ) -> GenericFact<Head2, Leaf2> {
        self.map_exprs(&mut |e| e.subst(subst_leaf, subst_head))
    }
}

impl<Head, Leaf> GenericFact<Head, Leaf>
where
    Leaf: Clone + PartialEq + Eq + Display + Hash,
    Head: Clone + Display,
{
    pub(crate) fn make_unresolved(self) -> GenericFact<String, String> {
        self.subst(
            &mut |span, v| GenericExpr::Var(span.clone(), v.to_string()),
            &mut |h| h.to_string(),
        )
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

pub type Action = GenericAction<String, String>;
pub(crate) type MappedAction = GenericAction<CorrespondingVar<String, String>, String>;
pub(crate) type ResolvedAction = GenericAction<ResolvedCall, ResolvedVar>;

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
    // If(Expr, Action, Action),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]

pub struct GenericActions<Head: Clone + Display, Leaf: Clone + PartialEq + Eq + Display + Hash>(
    pub Vec<GenericAction<Head, Leaf>>,
);
pub type Actions = GenericActions<String, String>;
pub(crate) type ResolvedActions = GenericActions<ResolvedCall, ResolvedVar>;
pub(crate) type MappedActions<Head, Leaf> = GenericActions<CorrespondingVar<Head, Leaf>, Leaf>;

impl<Head, Leaf> Default for GenericActions<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    fn default() -> Self {
        Self(vec![])
    }
}

impl<Head, Leaf> GenericActions<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &GenericAction<Head, Leaf>> {
        self.0.iter()
    }

    pub(crate) fn visit_exprs(
        self,
        f: &mut impl FnMut(GenericExpr<Head, Leaf>) -> GenericExpr<Head, Leaf>,
    ) -> Self {
        Self(self.0.into_iter().map(|a| a.visit_exprs(f)).collect())
    }
}

impl<Head, Leaf> Display for GenericAction<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            GenericAction::Let(_ann, lhs, rhs) => write!(f, "(let {lhs} {rhs})"),
            GenericAction::Set(_ann, lhs, args, rhs) => {
                write!(f, "(set ({lhs} {}) {rhs})", ListDisplay(args, " "))
            }
            GenericAction::Union(_ann, lhs, rhs) => write!(f, "(union {lhs} {rhs})"),
            GenericAction::Change(_ann, change, lhs, args) => {
                let change = match change {
                    Change::Delete => "delete",
                    Change::Subsume => "subsume",
                };
                write!(f, "({change} ({lhs} {}))", ListDisplay(args, " "))
            }
            GenericAction::Panic(_ann, msg) => write!(f, "(panic {msg:?})"),
            GenericAction::Expr(_ann, e) => write!(f, "{e}"),
        }
    }
}

impl<Head, Leaf> GenericAction<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + Eq + Display + Hash,
{
    // Applys `f` to all expressions in the action.
    pub fn map_exprs(
        &self,
        f: &mut impl FnMut(&GenericExpr<Head, Leaf>) -> GenericExpr<Head, Leaf>,
    ) -> Self {
        match self {
            GenericAction::Let(span, lhs, rhs) => {
                GenericAction::Let(span.clone(), lhs.clone(), f(rhs))
            }
            GenericAction::Set(span, lhs, args, rhs) => {
                let right = f(rhs);
                GenericAction::Set(
                    span.clone(),
                    lhs.clone(),
                    args.iter().map(f).collect(),
                    right,
                )
            }
            GenericAction::Change(span, change, lhs, args) => GenericAction::Change(
                span.clone(),
                *change,
                lhs.clone(),
                args.iter().map(f).collect(),
            ),
            GenericAction::Union(span, lhs, rhs) => {
                GenericAction::Union(span.clone(), f(lhs), f(rhs))
            }
            GenericAction::Panic(span, msg) => GenericAction::Panic(span.clone(), msg.clone()),
            GenericAction::Expr(span, e) => GenericAction::Expr(span.clone(), f(e)),
        }
    }

    /// Applys `f` to all sub-expressions (including `self`)
    /// bottom-up, collecting the results.
    pub fn visit_exprs(
        self,
        f: &mut impl FnMut(GenericExpr<Head, Leaf>) -> GenericExpr<Head, Leaf>,
    ) -> Self {
        match self {
            GenericAction::Let(span, lhs, rhs) => {
                GenericAction::Let(span, lhs.clone(), rhs.visit_exprs(f))
            }
            // TODO should we refactor `Set` so that we can map over Expr::Call(lhs, args)?
            // This seems more natural to oflatt
            // Currently, visit_exprs does not apply f to the first argument of Set.
            GenericAction::Set(span, lhs, args, rhs) => {
                let args = args.into_iter().map(|e| e.visit_exprs(f)).collect();
                GenericAction::Set(span, lhs.clone(), args, rhs.visit_exprs(f))
            }
            GenericAction::Change(span, change, lhs, args) => {
                let args = args.into_iter().map(|e| e.visit_exprs(f)).collect();
                GenericAction::Change(span, change, lhs.clone(), args)
            }
            GenericAction::Union(span, lhs, rhs) => {
                GenericAction::Union(span, lhs.visit_exprs(f), rhs.visit_exprs(f))
            }
            GenericAction::Panic(span, msg) => GenericAction::Panic(span, msg.clone()),
            GenericAction::Expr(span, e) => GenericAction::Expr(span, e.visit_exprs(f)),
        }
    }

    pub fn subst(&self, subst: &mut impl FnMut(&Span, &Leaf) -> GenericExpr<Head, Leaf>) -> Self {
        self.map_exprs(&mut |e| e.subst_leaf(subst))
    }

    pub fn map_def_use(self, fvar: &mut impl FnMut(Leaf, bool) -> Leaf) -> Self {
        macro_rules! fvar_expr {
            () => {
                |span, s: _| GenericExpr::Var(span.clone(), fvar(s.clone(), false))
            };
        }
        match self {
            GenericAction::Let(span, lhs, rhs) => {
                let lhs = fvar(lhs, true);
                let rhs = rhs.subst_leaf(&mut fvar_expr!());
                GenericAction::Let(span, lhs, rhs)
            }
            GenericAction::Set(span, lhs, args, rhs) => {
                let args = args
                    .into_iter()
                    .map(|e| e.subst_leaf(&mut fvar_expr!()))
                    .collect();
                let rhs = rhs.subst_leaf(&mut fvar_expr!());
                GenericAction::Set(span, lhs.clone(), args, rhs)
            }
            GenericAction::Change(span, change, lhs, args) => {
                let args = args
                    .into_iter()
                    .map(|e| e.subst_leaf(&mut fvar_expr!()))
                    .collect();
                GenericAction::Change(span, change, lhs.clone(), args)
            }
            GenericAction::Union(span, lhs, rhs) => {
                let lhs = lhs.subst_leaf(&mut fvar_expr!());
                let rhs = rhs.subst_leaf(&mut fvar_expr!());
                GenericAction::Union(span, lhs, rhs)
            }
            GenericAction::Panic(span, msg) => GenericAction::Panic(span, msg.clone()),
            GenericAction::Expr(span, e) => {
                GenericAction::Expr(span, e.subst_leaf(&mut fvar_expr!()))
            }
        }
    }
}

pub type Rule = GenericRule<String, String>;
pub(crate) type ResolvedRule = GenericRule<ResolvedCall, ResolvedVar>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GenericRule<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub span: Span,
    pub head: GenericActions<Head, Leaf>,
    pub body: Vec<GenericFact<Head, Leaf>>,
}

impl<Head, Leaf> GenericRule<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub(crate) fn visit_exprs(
        self,
        f: &mut impl FnMut(GenericExpr<Head, Leaf>) -> GenericExpr<Head, Leaf>,
    ) -> Self {
        Self {
            span: self.span,
            head: self.head.visit_exprs(f),
            body: self
                .body
                .into_iter()
                .map(|bexpr| bexpr.visit_exprs(f))
                .collect(),
        }
    }
}

impl<Head, Leaf> GenericRule<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub(crate) fn fmt_with_ruleset(
        &self,
        f: &mut Formatter,
        ruleset: &str,
        name: &str,
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
        let ruleset = if !ruleset.is_empty() {
            format!(":ruleset {}", ruleset)
        } else {
            "".into()
        };
        let name = if !name.is_empty() {
            format!(":name \"{}\"", name)
        } else {
            "".into()
        };
        write!(f, ")\n{} {} {})", indent, ruleset, name)
    }
}

pub type Rewrite = GenericRewrite<String, String>;

#[derive(Clone, Debug)]
pub struct GenericRewrite<Head, Leaf> {
    pub span: Span,
    pub lhs: GenericExpr<Head, Leaf>,
    pub rhs: GenericExpr<Head, Leaf>,
    pub conditions: Vec<GenericFact<Head, Leaf>>,
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

impl<Head, Leaf: Clone> MappedExpr<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub(crate) fn get_corresponding_var_or_lit(
        &self,
        typeinfo: &TypeInfo,
    ) -> GenericAtomTerm<Leaf> {
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

impl<Head, Leaf> GenericActions<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub fn new(actions: Vec<GenericAction<Head, Leaf>>) -> Self {
        Self(actions)
    }

    pub fn singleton(action: GenericAction<Head, Leaf>) -> Self {
        Self(vec![action])
    }
}
