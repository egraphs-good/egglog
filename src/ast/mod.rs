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

use crate::*;

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

pub type CommandId = usize;

// TODO put line numbers in metadata
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Metadata {
    pub id: CommandId,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct NormCommand {
    pub metadata: Metadata,
    pub command: NCommand,
}

impl NormCommand {
    pub fn transforms_to(&self, others: Vec<NCommand>) -> Vec<NormCommand> {
        others
            .into_iter()
            .map(|c| NormCommand {
                metadata: self.metadata.clone(),
                command: c,
            })
            .collect()
    }

    pub fn resugar(&self) -> Command {
        match &self.command {
            NCommand::NormRule {
                name,
                ruleset,
                rule,
            } => Command::Rule {
                name: *name,
                ruleset: *ruleset,
                rule: rule.resugar(),
            },
            NCommand::Check(facts) => {
                Command::Check(NormRule::resugar_facts(facts, &mut Default::default()))
            }
            _ => self.command.to_command(),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum NCommand {
    SetOption {
        name: Symbol,
        value: Expr,
    },
    Sort(Symbol, Option<(Symbol, Vec<Expr>)>),
    Function(NormFunctionDecl),
    AddRuleset(Symbol),
    NormRule {
        name: Symbol,
        ruleset: Symbol,
        rule: NormRule,
    },
    NormAction(NormAction),
    RunSchedule(NormSchedule),
    PrintOverallStatistics,
    Check(Vec<NormFact>),
    CheckProof,
    PrintTable(Symbol, usize),
    PrintSize(Option<Symbol>),
    Output {
        file: String,
        exprs: Vec<Expr>,
    },
    Push(usize),
    Pop(usize),
    Fail(Box<NCommand>),
    // TODO desugar
    Input {
        name: Symbol,
        file: String,
    },
}

impl NormCommand {
    pub fn to_command(&self) -> Command {
        self.command.to_command()
    }
}

impl NCommand {
    pub fn to_command(&self) -> Command {
        match self {
            NCommand::SetOption { name, value } => Command::SetOption {
                name: *name,
                value: value.clone(),
            },
            NCommand::Sort(name, params) => Command::Sort(*name, params.clone()),
            NCommand::Function(f) => Command::Function(f.to_fdecl()),
            NCommand::AddRuleset(name) => Command::AddRuleset(*name),
            NCommand::NormRule {
                name,
                ruleset,
                rule,
            } => Command::Rule {
                name: *name,
                ruleset: *ruleset,
                rule: rule.to_rule(),
            },
            NCommand::RunSchedule(schedule) => Command::RunSchedule(schedule.to_schedule()),
            NCommand::PrintOverallStatistics => Command::PrintOverallStatistics,
            NCommand::NormAction(action) => Command::Action(action.to_action()),
            NCommand::Check(facts) => {
                Command::Check(facts.iter().map(|fact| fact.to_fact()).collect())
            }
            NCommand::CheckProof => Command::CheckProof,
            NCommand::PrintTable(name, n) => Command::PrintFunction(*name, *n),
            NCommand::PrintSize(name) => Command::PrintSize(*name),
            NCommand::Output { file, exprs } => Command::Output {
                file: file.to_string(),
                exprs: exprs.clone(),
            },
            NCommand::Push(n) => Command::Push(*n),
            NCommand::Pop(n) => Command::Pop(*n),
            NCommand::Fail(cmd) => Command::Fail(Box::new(cmd.to_command())),
            NCommand::Input { name, file } => Command::Input {
                name: *name,
                file: file.clone(),
            },
        }
    }

    pub fn map_exprs(&self, f: &mut impl FnMut(&NormExpr) -> NormExpr) -> NCommand {
        match self {
            // Don't map over setoption
            NCommand::SetOption { name, value } => NCommand::SetOption {
                name: *name,
                value: value.clone(),
            },
            NCommand::Sort(name, params) => NCommand::Sort(*name, params.clone()),
            NCommand::Function(f) => NCommand::Function(f.clone()),
            NCommand::AddRuleset(name) => NCommand::AddRuleset(*name),
            NCommand::RunSchedule(schedule) => NCommand::RunSchedule(schedule.clone()),
            NCommand::PrintOverallStatistics => NCommand::PrintOverallStatistics,
            NCommand::NormRule {
                name,
                ruleset,
                rule,
            } => NCommand::NormRule {
                name: *name,
                ruleset: *ruleset,
                rule: rule.map_exprs(f),
            },
            NCommand::NormAction(action) => NCommand::NormAction(action.map_exprs(f)),
            NCommand::Check(facts) => {
                NCommand::Check(facts.iter().map(|fact| fact.map_exprs(f)).collect())
            }
            NCommand::CheckProof => NCommand::CheckProof,
            NCommand::PrintTable(name, n) => NCommand::PrintTable(*name, *n),
            NCommand::PrintSize(name) => NCommand::PrintSize(*name),
            NCommand::Output { file, exprs } => NCommand::Output {
                file: file.to_string(),
                exprs: exprs.clone(),
            },
            NCommand::Push(n) => NCommand::Push(*n),
            NCommand::Pop(n) => NCommand::Pop(*n),
            NCommand::Fail(cmd) => NCommand::Fail(Box::new(cmd.map_exprs(f))),
            NCommand::Input { name, file } => NCommand::Input {
                name: *name,
                file: file.clone(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Schedule {
    Saturate(Box<Schedule>),
    Repeat(usize, Box<Schedule>),
    Run(RunConfig),
    Sequence(Vec<Schedule>),
}

impl Schedule {
    pub fn saturate(self) -> Self {
        Schedule::Saturate(Box::new(self))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NormSchedule {
    Saturate(Box<NormSchedule>),
    Repeat(usize, Box<NormSchedule>),
    Run(NormRunConfig),
    Sequence(Vec<NormSchedule>),
}

impl NormSchedule {
    fn to_schedule(&self) -> Schedule {
        match self {
            NormSchedule::Saturate(sched) => Schedule::Saturate(Box::new(sched.to_schedule())),
            NormSchedule::Repeat(size, sched) => {
                Schedule::Repeat(*size, Box::new(sched.to_schedule()))
            }
            NormSchedule::Run(config) => Schedule::Run(config.to_run_config()),
            NormSchedule::Sequence(scheds) => {
                Schedule::Sequence(scheds.iter().map(|sched| sched.to_schedule()).collect())
            }
        }
    }

    pub fn map_run_commands(&self, f: &mut impl FnMut(&NormRunConfig) -> Schedule) -> Schedule {
        match self {
            NormSchedule::Run(config) => f(config),
            NormSchedule::Saturate(sched) => {
                Schedule::Saturate(Box::new(sched.map_run_commands(f)))
            }
            NormSchedule::Repeat(size, sched) => {
                Schedule::Repeat(*size, Box::new(sched.map_run_commands(f)))
            }
            NormSchedule::Sequence(scheds) => Schedule::Sequence(
                scheds
                    .iter()
                    .map(|sched| sched.map_run_commands(f))
                    .collect(),
            ),
        }
    }
}

trait ToSexp {
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

impl ToSexp for Schedule {
    fn to_sexp(&self) -> Sexp {
        match self {
            Schedule::Saturate(sched) => list!("saturate", sched),
            Schedule::Repeat(size, sched) => list!("repeat", size, sched),
            Schedule::Run(config) => config.to_sexp(),
            Schedule::Sequence(scheds) => list!("seq", ++ scheds),
        }
    }
}

impl Display for Schedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp())
    }
}

impl Display for NormSchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_schedule())
    }
}

// TODO command before and after desugaring should be different
/// A [`Command`] is the top-level construct in egglog.
/// It includes defining rules, declaring functions,
/// adding to tables, and running rules (via a [`Schedule`]).
#[derive(Debug, Clone)]
pub enum Command {
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
        value: Expr,
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
    Function(FunctionDecl),
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
        rule: Rule,
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
    Rewrite(Symbol, Rewrite),
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
    BiRewrite(Symbol, Rewrite),
    /// Perform an [`Action`] on the global database
    /// (see documentation for [`Action`] for more details).
    /// Example:
    /// ```text
    /// (let xplusone (Add (Var "x") (Num 1)))
    /// ```
    Action(Action),
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
    RunSchedule(Schedule),
    /// Print runtime statistics about rules
    /// and rulesets so far.
    PrintOverallStatistics,
    // TODO provide simplify docs
    Simplify {
        expr: Expr,
        schedule: Schedule,
    },
    // TODO provide calc docs
    Calc(Vec<IdentSort>, Vec<Expr>),
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
        expr: Expr,
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
    Check(Vec<Fact>),
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
        exprs: Vec<Expr>,
    },
    /// `push` the current egraph `n` times so that it is saved.
    /// Later, the current database and rules can be restored using `pop`.
    Push(usize),
    /// `pop` the current egraph, restoring the previous one.
    /// The argument specifies how many egraphs to pop.
    Pop(usize),
    /// Assert that a command fails with an error.
    Fail(Box<Command>),
    /// Include another egglog file directly as text and run it.
    Include(String),
}

impl ToSexp for Command {
    fn to_sexp(&self) -> Sexp {
        match self {
            Command::SetOption { name, value } => list!("set-option", name, value),
            Command::Rewrite(name, rewrite) => rewrite.to_sexp(*name, false),
            Command::BiRewrite(name, rewrite) => rewrite.to_sexp(*name, true),
            Command::Datatype { name, variants } => list!("datatype", name, ++ variants),
            Command::Declare { name, sort } => list!("declare", name, sort),
            Command::Action(a) => a.to_sexp(),
            Command::Sort(name, None) => list!("sort", name),
            Command::Sort(name, Some((name2, args))) => list!("sort", name, list!( name2, ++ args)),
            Command::Function(f) => f.to_sexp(),
            Command::Relation {
                constructor,
                inputs,
            } => list!("relation", constructor, list!(++ inputs)),
            Command::AddRuleset(name) => list!("ruleset", name),
            Command::Rule {
                name,
                ruleset,
                rule,
            } => rule.to_sexp(*ruleset, *name),
            Command::RunSchedule(sched) => list!("run-schedule", sched),
            Command::PrintOverallStatistics => list!("print-stats"),
            Command::Calc(args, exprs) => list!("calc", list!(++ args), ++ exprs),
            Command::QueryExtract { variants, expr } => {
                list!("query-extract", ":variants", variants, expr)
            }
            Command::Check(facts) => list!("check", ++ facts),
            Command::CheckProof => list!("check-proof"),
            Command::Push(n) => list!("push", n),
            Command::Pop(n) => list!("pop", n),
            Command::PrintFunction(name, n) => list!("print-function", name, n),
            Command::PrintSize(name) => list!("print-size", ++ name),
            Command::Input { name, file } => list!("input", name, format!("\"{}\"", file)),
            Command::Output { file, exprs } => list!("output", format!("\"{}\"", file), ++ exprs),
            Command::Fail(cmd) => list!("fail", cmd),
            Command::Include(file) => list!("include", format!("\"{}\"", file)),
            Command::Simplify { expr, schedule } => list!("simplify", schedule, expr),
        }
    }
}

impl Display for NormCommand {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_command())
    }
}

impl Display for NCommand {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_command())
    }
}

impl Display for Command {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Command::Rule {
                ruleset,
                name,
                rule,
            } => rule.fmt_with_ruleset(f, *ruleset, *name),
            Command::Check(facts) => {
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RunConfig {
    pub ruleset: Symbol,
    pub until: Option<Vec<Fact>>,
}

impl ToSexp for RunConfig {
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NormRunConfig {
    pub ctx: CommandId,
    pub ruleset: Symbol,
    pub until: Option<Vec<NormFact>>,
}

impl NormRunConfig {
    pub fn to_run_config(&self) -> RunConfig {
        RunConfig {
            ruleset: self.ruleset,
            until: self
                .until
                .as_ref()
                .map(|v| v.iter().map(|f| f.to_fact()).collect()),
        }
    }
}

/// A normalized function declaration- the desugared
/// version of a [`FunctionDecl`].
/// TODO so far only the merge action is normalized,
/// not the default value or merge expression.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NormFunctionDecl {
    pub name: Symbol,
    pub schema: Schema,
    // todo desugar default, merge
    pub default: Option<Expr>,
    pub merge: Option<Expr>,
    pub merge_action: Vec<NormAction>,
    pub cost: Option<usize>,
    pub unextractable: bool,
}

impl NormFunctionDecl {
    pub fn to_fdecl(&self) -> FunctionDecl {
        FunctionDecl {
            name: self.name,
            schema: self.schema.clone(),
            default: self.default.clone(),
            merge: self.merge.clone(),
            merge_action: self.merge_action.iter().map(|a| a.to_action()).collect(),
            cost: self.cost,
            unextractable: self.unextractable,
        }
    }
}

/// Represents the declaration of a function
/// directly parsed from source syntax.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionDecl {
    pub name: Symbol,
    pub schema: Schema,
    pub default: Option<Expr>,
    pub merge: Option<Expr>,
    pub merge_action: Vec<Action>,
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
            merge_action: vec![],
            default: Some(Expr::Lit(Literal::Unit)),
            cost: None,
            unextractable: false,
        }
    }
}

impl ToSexp for FunctionDecl {
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
pub enum Fact {
    /// Must be at least two things in an eq fact
    Eq(Vec<Expr>),
    Fact(Expr),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NormFact {
    Assign(Symbol, NormExpr), // assign symbol to a tuple
    AssignVar(Symbol, Symbol),
    Compute(Symbol, NormExpr), // compute a primative
    AssignLit(Symbol, Literal),
    ConstrainEq(Symbol, Symbol),
}

impl NormFact {
    pub fn to_fact(&self) -> Fact {
        match self {
            NormFact::Assign(symbol, expr) | NormFact::Compute(symbol, expr) => {
                Fact::Eq(vec![Expr::Var(*symbol), expr.to_expr()])
            }
            NormFact::AssignVar(lhs, rhs) => Fact::Eq(vec![Expr::Var(*lhs), Expr::Var(*rhs)]),
            NormFact::ConstrainEq(lhs, rhs) => Fact::Eq(vec![Expr::Var(*lhs), Expr::Var(*rhs)]),
            NormFact::AssignLit(symbol, lit) => {
                Fact::Eq(vec![Expr::Var(*symbol), Expr::Lit(lit.clone())])
            }
        }
    }

    pub fn map_exprs(&self, f: &mut impl FnMut(&NormExpr) -> NormExpr) -> NormFact {
        match self {
            NormFact::Assign(symbol, expr) | NormFact::Compute(symbol, expr) => {
                NormFact::Assign(*symbol, f(expr))
            }
            NormFact::AssignVar(lhs, rhs) => NormFact::AssignVar(*lhs, *rhs),
            NormFact::ConstrainEq(lhs, rhs) => NormFact::ConstrainEq(*lhs, *rhs),
            NormFact::AssignLit(symbol, lit) => NormFact::AssignLit(*symbol, lit.clone()),
        }
    }

    pub(crate) fn map_use(&self, fvar: &mut impl FnMut(Symbol) -> Expr) -> Fact {
        match self {
            NormFact::AssignVar(lhs, rhs) => Fact::Eq(vec![Expr::Var(*lhs), fvar(*rhs)]),
            NormFact::ConstrainEq(lhs, rhs) => Fact::Eq(vec![fvar(*lhs), fvar(*rhs)]),
            NormFact::Compute(lhs, NormExpr::Call(op, children)) => Fact::Eq(vec![
                fvar(*lhs),
                Expr::Call(*op, children.iter().cloned().map(fvar).collect()),
            ]),
            NormFact::AssignLit(..) | NormFact::Assign(..) => self.to_fact(),
        }
    }

    pub(crate) fn map_def_use(&self, fvar: &mut impl FnMut(Symbol, bool) -> Symbol) -> NormFact {
        match self {
            NormFact::Assign(symbol, expr) => {
                NormFact::Assign(fvar(*symbol, true), expr.map_def_use(fvar, true))
            }
            NormFact::AssignVar(lhs, rhs) => {
                NormFact::AssignVar(fvar(*lhs, true), fvar(*rhs, false))
            }
            NormFact::Compute(symbol, expr) => {
                NormFact::Compute(fvar(*symbol, true), expr.map_def_use(fvar, false))
            }
            NormFact::AssignLit(symbol, lit) => {
                NormFact::AssignLit(fvar(*symbol, true), lit.clone())
            }
            NormFact::ConstrainEq(lhs, rhs) => {
                NormFact::ConstrainEq(fvar(*lhs, false), fvar(*rhs, false))
            }
        }
    }
}

impl ToSexp for Fact {
    fn to_sexp(&self) -> Sexp {
        match self {
            Fact::Eq(exprs) => list!("=", ++ exprs),
            Fact::Fact(expr) => expr.to_sexp(),
        }
    }
}

impl Fact {
    pub fn map_exprs(&self, f: &mut impl FnMut(&Expr) -> Expr) -> Fact {
        match self {
            Fact::Eq(exprs) => Fact::Eq(exprs.iter().map(f).collect()),
            Fact::Fact(expr) => Fact::Fact(f(expr)),
        }
    }

    pub fn subst(&self, subst: &HashMap<Symbol, Expr>) -> Fact {
        self.map_exprs(&mut |e| e.subst(subst))
    }
}

impl Display for NormFact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_fact())
    }
}

impl Display for Fact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    /// Bind a variable to a particular datatype or primitive.
    /// At the top level (in a [`Command::Action`]), this defines a global variable.
    /// In a [`Command::Rule`], this defines a local variable in the actions.
    Let(Symbol, Expr),
    /// `set` a function to a particular result.
    /// `set` should not be used on datatypes-
    /// instead, use `union`.
    Set(Symbol, Vec<Expr>, Expr),
    /// `delete` an entry from a function.
    /// Be wary! Only delete entries that are subsumed in some way or
    /// guaranteed to be not useful.
    Delete(Symbol, Vec<Expr>),
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
    Union(Expr, Expr),
    /// `extract` a datatype from the egraph, choosing
    /// the smallest representative.
    /// By default, each constructor costs 1 to extract
    /// (common subexpressions are not shared in the cost
    /// model).
    /// The second argument is the number of variants to
    /// extract, picking different terms in the
    /// same equivalence class.
    Extract(Expr, Expr),
    Panic(String),
    Expr(Expr),
    // If(Expr, Action, Action),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NormAction {
    Let(Symbol, NormExpr),
    LetVar(Symbol, Symbol),
    LetLit(Symbol, Literal),
    Extract(Symbol, Symbol),
    Set(NormExpr, Symbol),
    Delete(NormExpr),
    Union(Symbol, Symbol),
    Panic(String),
}

impl NormAction {
    pub fn to_action(&self) -> Action {
        match self {
            NormAction::Let(symbol, expr) => Action::Let(*symbol, expr.to_expr()),
            NormAction::LetVar(symbol, other) => Action::Let(*symbol, Expr::Var(*other)),
            NormAction::LetLit(symbol, lit) => Action::Let(*symbol, Expr::Lit(lit.clone())),
            NormAction::Set(NormExpr::Call(head, body), other) => Action::Set(
                *head,
                body.iter().map(|s| Expr::Var(*s)).collect(),
                Expr::Var(*other),
            ),
            NormAction::Extract(symbol, variants) => {
                Action::Extract(Expr::Var(*symbol), Expr::Var(*variants))
            }
            NormAction::Delete(NormExpr::Call(symbol, args)) => {
                Action::Delete(*symbol, args.iter().map(|s| Expr::Var(*s)).collect())
            }
            NormAction::Union(lhs, rhs) => Action::Union(Expr::Var(*lhs), Expr::Var(*rhs)),
            NormAction::Panic(msg) => Action::Panic(msg.clone()),
        }
    }

    pub fn map_exprs(&self, f: &mut impl FnMut(&NormExpr) -> NormExpr) -> NormAction {
        match self {
            NormAction::Let(symbol, expr) => NormAction::Let(*symbol, f(expr)),
            NormAction::LetVar(symbol, other) => NormAction::LetVar(*symbol, *other),
            NormAction::LetLit(symbol, lit) => NormAction::LetLit(*symbol, lit.clone()),
            NormAction::Set(expr, other) => NormAction::Set(f(expr), *other),
            NormAction::Extract(var, variants) => NormAction::Extract(*var, *variants),
            NormAction::Delete(expr) => NormAction::Delete(f(expr)),
            NormAction::Union(lhs, rhs) => NormAction::Union(*lhs, *rhs),
            NormAction::Panic(msg) => NormAction::Panic(msg.clone()),
        }
    }

    // fvar accepts a variable and if it is being defined (true) or used (false)
    pub(crate) fn map_def_use(&self, fvar: &mut impl FnMut(Symbol, bool) -> Symbol) -> NormAction {
        match self {
            NormAction::Let(symbol, expr) => {
                NormAction::Let(fvar(*symbol, true), expr.map_def_use(fvar, false))
            }
            NormAction::LetVar(symbol, other) => {
                NormAction::LetVar(fvar(*symbol, true), fvar(*other, false))
            }
            NormAction::LetLit(symbol, lit) => NormAction::LetLit(fvar(*symbol, true), lit.clone()),
            NormAction::Set(expr, other) => {
                NormAction::Set(expr.map_def_use(fvar, false), fvar(*other, false))
            }
            NormAction::Extract(var, variants) => {
                NormAction::Extract(fvar(*var, false), fvar(*variants, false))
            }
            NormAction::Delete(expr) => NormAction::Delete(expr.map_def_use(fvar, false)),
            NormAction::Union(lhs, rhs) => NormAction::Union(fvar(*lhs, false), fvar(*rhs, false)),
            NormAction::Panic(msg) => NormAction::Panic(msg.clone()),
        }
    }
}

impl ToSexp for Action {
    fn to_sexp(&self) -> Sexp {
        match self {
            Action::Let(lhs, rhs) => list!("let", lhs, rhs),
            Action::Set(lhs, args, rhs) => list!("set", list!(lhs, ++ args), rhs),
            Action::Union(lhs, rhs) => list!("union", lhs, rhs),
            Action::Delete(lhs, args) => list!("delete", list!(lhs, ++ args)),
            Action::Extract(expr, variants) => list!("extract", expr, variants),
            Action::Panic(msg) => list!("panic", format!("\"{}\"", msg.clone())),
            Action::Expr(e) => e.to_sexp(),
        }
    }
}

impl Action {
    pub fn map_exprs(&self, f: &mut impl FnMut(&Expr) -> Expr) -> Self {
        match self {
            Action::Let(lhs, rhs) => Action::Let(*lhs, f(rhs)),
            Action::Set(lhs, args, rhs) => {
                let right = f(rhs);
                Action::Set(*lhs, args.iter().map(f).collect(), right)
            }
            Action::Delete(lhs, args) => Action::Delete(*lhs, args.iter().map(f).collect()),
            Action::Union(lhs, rhs) => Action::Union(f(lhs), f(rhs)),
            Action::Extract(expr, variants) => Action::Extract(f(expr), f(variants)),
            Action::Panic(msg) => Action::Panic(msg.clone()),
            Action::Expr(e) => Action::Expr(f(e)),
        }
    }

    pub fn replace_canon(&self, canon: &HashMap<Symbol, Expr>) -> Self {
        match self {
            Action::Let(lhs, rhs) => Action::Let(*lhs, rhs.subst(canon)),
            Action::Set(lhs, args, rhs) => Action::Set(
                *lhs,
                args.iter().map(|e| e.subst(canon)).collect(),
                rhs.subst(canon),
            ),
            Action::Delete(lhs, args) => {
                Action::Delete(*lhs, args.iter().map(|e| e.subst(canon)).collect())
            }
            Action::Union(lhs, rhs) => Action::Union(lhs.subst(canon), rhs.subst(canon)),
            Action::Extract(expr, variants) => {
                Action::Extract(expr.subst(canon), variants.subst(canon))
            }
            Action::Panic(msg) => Action::Panic(msg.clone()),
            Action::Expr(e) => Action::Expr(e.subst(canon)),
        }
    }
}

impl Display for NormAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_action())
    }
}

impl Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp())
    }
}

#[derive(Clone, Debug)]
pub struct Rule {
    // pub query: Query,
    // pub actions: Vec<Action>,
    pub head: Vec<Action>,
    pub body: Vec<Fact>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NormRule {
    pub head: Vec<NormAction>,
    pub body: Vec<NormFact>,
}

impl NormRule {
    pub fn to_rule(&self) -> Rule {
        Rule {
            head: self.head.iter().map(|a| a.to_action()).collect(),
            body: self.body.iter().map(|f| f.to_fact()).collect(),
        }
    }

    pub fn globals_used_in_matcher(facts: &Vec<NormFact>) -> HashSet<Symbol> {
        let mut bound_vars = HashSet::<Symbol>::default();
        for fact in facts {
            fact.map_def_use(&mut |var, def| {
                if def {
                    bound_vars.insert(var);
                }
                var
            });
        }

        let mut unbound_vars = HashSet::<Symbol>::default();
        for fact in facts {
            fact.map_def_use(&mut |var, def| {
                if !def && !bound_vars.contains(&var) {
                    unbound_vars.insert(var);
                }
                var
            });
        }
        unbound_vars
    }

    // just get rid of all the equality constraints for now
    pub fn resugar_facts(facts: &Vec<NormFact>, subst: &mut HashMap<Symbol, Expr>) -> Vec<Fact> {
        let unbound = NormRule::globals_used_in_matcher(facts);
        let mut unionfind = UnionFind::default();
        let mut var_to_id = HashMap::<Symbol, Id>::default();
        let mut id_to_var = HashMap::<Id, Symbol>::default();
        let mut get_id = |var: Symbol, uf: &mut UnionFind| -> Id {
            if let Some(id) = var_to_id.get(&var) {
                *id
            } else {
                let id = uf.make_set();
                var_to_id.insert(var, id);
                id_to_var.insert(id, var);
                id
            }
        };
        for norm_fact in facts {
            if let NormFact::ConstrainEq(v1, v2) = norm_fact {
                let id1 = get_id(*v1, &mut unionfind);
                let id2 = get_id(*v2, &mut unionfind);
                unionfind.union_raw(id1, id2);
            } else if let NormFact::AssignVar(v1, v2) = norm_fact {
                let id1 = get_id(*v1, &mut unionfind);
                let id2 = get_id(*v2, &mut unionfind);
                unionfind.union_raw(id1, id2);
            }
        }

        for (var, id) in &var_to_id {
            let leader = id_to_var.get(&unionfind.find(*id)).unwrap();
            if leader != var {
                subst.insert(*var, Expr::Var(*leader));
            }
        }

        let mut res = vec![];
        for fact in facts {
            match fact {
                NormFact::ConstrainEq(..) => (),
                NormFact::AssignVar(..) => (),
                _ => res.push(fact.to_fact().subst(subst)),
            }
        }

        // add back contraints on unbound variables
        for var in unbound {
            if let Some(id) = var_to_id.get(&var) {
                let leader = id_to_var.get(&unionfind.find(*id)).unwrap();
                if leader != &var {
                    res.push(Fact::Eq(vec![Expr::Var(var), Expr::Var(*leader)]));
                }
            }
        }

        res
    }

    pub fn resugar_actions(&self, subst: &mut HashMap<Symbol, Expr>) -> Vec<Action> {
        let mut used = HashSet::<Symbol>::default();
        let mut head = Vec::<Action>::default();
        for a in &self.head {
            match a {
                NormAction::Let(symbol, expr) => {
                    let new_expr = expr.to_expr();
                    new_expr.map(&mut |subexpr| {
                        if let Expr::Var(v) = subexpr {
                            used.insert(*v);
                        }
                        subexpr.clone()
                    });
                    let substituted = new_expr.subst(subst);

                    // TODO sometimes re-arranging actions is bad
                    // this is because actions can fail
                    // halfway through in the current semantics
                    if substituted.ast_size() > 1 {
                        head.push(Action::Let(*symbol, substituted));
                    } else {
                        subst.insert(*symbol, substituted);
                    }
                }
                NormAction::LetVar(symbol, other) => {
                    let new_expr = subst.get(other).unwrap_or(&Expr::Var(*other)).clone();
                    used.insert(*other);
                    subst.insert(*symbol, new_expr);
                }
                NormAction::Extract(symbol, variants) => {
                    let new_expr = subst.get(symbol).cloned().unwrap_or(Expr::Var(*symbol));
                    used.insert(*symbol);
                    let new_expr2 = subst.get(variants).cloned().unwrap_or(Expr::Var(*variants));
                    used.insert(*variants);
                    head.push(Action::Extract(new_expr, new_expr2));
                }
                NormAction::LetLit(symbol, lit) => {
                    subst.insert(*symbol, Expr::Lit(lit.clone()));
                }
                NormAction::Set(expr, other) => {
                    let new_expr = expr.to_expr();
                    new_expr.map(&mut |subexpr| {
                        if let Expr::Var(v) = subexpr {
                            used.insert(*v);
                        }
                        subexpr.clone()
                    });
                    let other_expr = subst.get(other).unwrap_or(&Expr::Var(*other)).clone();
                    used.insert(*other);
                    let substituted = new_expr.subst(subst);
                    match substituted {
                        Expr::Call(op, children) => {
                            head.push(Action::Set(op, children, other_expr));
                        }
                        _ => panic!("Expected call in set"),
                    }
                }
                NormAction::Delete(expr) => {
                    let new_expr = expr.to_expr();
                    new_expr.map(&mut |subexpr| {
                        if let Expr::Var(v) = subexpr {
                            used.insert(*v);
                        }
                        subexpr.clone()
                    });
                    match new_expr.subst(subst) {
                        Expr::Call(op, children) => {
                            head.push(Action::Delete(op, children));
                        }
                        _ => panic!("Expected call in delete"),
                    }
                }
                NormAction::Union(lhs, rhs) => {
                    let new_lhs = subst.get(lhs).unwrap_or(&Expr::Var(*lhs)).clone();
                    let new_rhs = subst.get(rhs).unwrap_or(&Expr::Var(*rhs)).clone();
                    used.insert(*lhs);
                    used.insert(*rhs);
                    head.push(Action::Union(new_lhs, new_rhs));
                }
                NormAction::Panic(msg) => {
                    head.push(Action::Panic(msg.clone()));
                }
            }
        }

        // unused substitutions need to be added
        // to the action, since they have the side-effect
        // of adding to the database
        for (var, expr) in subst {
            if !used.contains(var) {
                match expr {
                    Expr::Var(..) => (),
                    Expr::Lit(..) => (),
                    Expr::Call(..) => head.push(Action::Expr(expr.clone())),
                };
            }
        }
        head
    }

    pub fn resugar(&self) -> Rule {
        let mut subst = HashMap::<Symbol, Expr>::default();

        let facts_resugared = NormRule::resugar_facts(&self.body, &mut subst);

        Rule {
            head: self.resugar_actions(&mut subst),
            body: facts_resugared,
        }
    }

    pub fn map_exprs(&self, f: &mut impl FnMut(&NormExpr) -> NormExpr) -> Self {
        NormRule {
            head: self.head.iter().map(|a| a.map_exprs(f)).collect(),
            body: self.body.iter().map(|fac| fac.map_exprs(f)).collect(),
        }
    }

    pub fn map_def_use(&self, fvar: &mut impl FnMut(Symbol, bool) -> Symbol) -> Self {
        NormRule {
            head: self.head.iter().map(|a| a.map_def_use(fvar)).collect(),
            body: self.body.iter().map(|fac| fac.map_def_use(fvar)).collect(),
        }
    }
}

impl Display for NormRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_rule())
    }
}

impl Rule {
    pub(crate) fn to_sexp(&self, ruleset: Symbol, name: Symbol) -> Sexp {
        let mut res = vec![
            Sexp::Symbol("rule".into()),
            Sexp::List(self.body.iter().map(|f| f.to_sexp()).collect()),
            Sexp::List(self.head.iter().map(|a| a.to_sexp()).collect()),
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

    pub fn map_exprs(&self, f: &mut impl FnMut(&Expr) -> Expr) -> Self {
        Rule {
            head: self.head.iter().map(|a| a.map_exprs(f)).collect(),
            body: self.body.iter().map(|fact| fact.map_exprs(f)).collect(),
        }
    }

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
        for (i, action) in self.head.iter().enumerate() {
            if i > 0 {
                write!(f, "{}", indent)?;
            }
            if i != self.head.len() - 1 {
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

impl Display for Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_with_ruleset(f, "".into(), "".into())
    }
}

#[derive(Clone, Debug)]
pub struct Rewrite {
    pub lhs: Expr,
    pub rhs: Expr,
    pub conditions: Vec<Fact>,
}

impl Rewrite {
    pub(crate) fn to_sexp(&self, ruleset: Symbol, is_bidirectional: bool) -> Sexp {
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

impl Display for Rewrite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp("".into(), false))
    }
}
