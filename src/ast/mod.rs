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
    typecheck::{GenericAtom, GenericAtomTerm, HeadOrEq, Query, ResolvedCall},
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

pub type UnresolvedNCommand = NCommand<Symbol, Symbol, ()>;
pub(crate) type ResolvedNCommand = NCommand<ResolvedCall, ResolvedVar, ()>;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum NCommand<Head, Leaf, Ann> {
    SetOption {
        name: Symbol,
        value: Expr<Head, Leaf, Ann>,
    },
    Sort(Symbol, Option<(Symbol, Vec<Expr<Symbol, Symbol, Ann>>)>),
    Function(FunctionDecl<Head, Leaf, Ann>),
    AddRuleset(Symbol),
    NormRule {
        name: Symbol,
        ruleset: Symbol,
        rule: Rule<Head, Leaf, Ann>,
    },
    NormAction(Action<Head, Leaf, Ann>),
    RunSchedule(Schedule<Head, Leaf, Ann>),
    PrintOverallStatistics,
    Check(Vec<Fact<Head, Leaf, Ann>>),
    CheckProof,
    PrintTable(Symbol, usize),
    PrintSize(Option<Symbol>),
    Output {
        file: String,
        exprs: Vec<Expr<Head, Leaf, Ann>>,
    },
    Push(usize),
    Pop(usize),
    Fail(Box<NCommand<Head, Leaf, Ann>>),
    // TODO desugar
    Input {
        name: Symbol,
        file: String,
    },
}

impl<Head, Leaf> NCommand<Head, Leaf, ()>
where
    Head: Clone,
    Leaf: Clone,
{
    pub fn to_command(&self) -> Command<Head, Leaf> {
        match self {
            NCommand::SetOption { name, value } => Command::SetOption {
                name: *name,
                value: value.clone(),
            },
            NCommand::Sort(name, params) => Command::Sort(*name, params.clone()),
            NCommand::Function(f) => Command::Function(f.clone()),
            NCommand::AddRuleset(name) => Command::AddRuleset(*name),
            NCommand::NormRule {
                name,
                ruleset,
                rule,
            } => Command::Rule {
                name: *name,
                ruleset: *ruleset,
                rule: rule.clone(),
            },
            NCommand::RunSchedule(schedule) => Command::RunSchedule(schedule.clone()),
            NCommand::PrintOverallStatistics => Command::PrintOverallStatistics,
            NCommand::NormAction(action) => Command::Action(action.clone()),
            NCommand::Check(facts) => Command::Check(facts.clone()),
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

    // pub fn map_exprs(&self, f: &mut impl FnMut(&NormExpr) -> NormExpr) -> NCommand {
    //     match self {
    //         // Don't map over setoption
    //         NCommand::SetOption { name, value } => NCommand::SetOption {
    //             name: *name,
    //             value: value.clone(),
    //         },
    //         NCommand::Sort(name, params) => NCommand::Sort(*name, params.clone()),
    //         NCommand::Function(f) => NCommand::Function(f.clone()),
    //         NCommand::AddRuleset(name) => NCommand::AddRuleset(*name),
    //         NCommand::RunSchedule(schedule) => NCommand::RunSchedule(schedule.clone()),
    //         NCommand::PrintOverallStatistics => NCommand::PrintOverallStatistics,
    //         NCommand::NormRule {
    //             name,
    //             ruleset,
    //             rule,
    //         } => NCommand::NormRule {
    //             name: *name,
    //             ruleset: *ruleset,
    //             rule: rule.clone(),
    //         },
    //         NCommand::NormAction(action) => NCommand::NormAction(action.clone()),
    //         NCommand::Check(facts) => {
    //             NCommand::Check(facts.clone())
    //         }
    //         NCommand::CheckProof => NCommand::CheckProof,
    //         NCommand::PrintTable(name, n) => NCommand::PrintTable(*name, *n),
    //         NCommand::PrintSize(name) => NCommand::PrintSize(*name),
    //         NCommand::Output { file, exprs } => NCommand::Output {
    //             file: file.to_string(),
    //             exprs: exprs.clone(),
    //         },
    //         NCommand::Push(n) => NCommand::Push(*n),
    //         NCommand::Pop(n) => NCommand::Pop(*n),
    //         NCommand::Fail(cmd) => NCommand::Fail(Box::new(cmd.map_exprs(f))),
    //         NCommand::Input { name, file } => NCommand::Input {
    //             name: *name,
    //             file: file.clone(),
    //         },
    //     }
    // }
}

pub type UnresolvedSchedule = Schedule<Symbol, Symbol, ()>;
pub(crate) type ResolvedSchedule = Schedule<ResolvedCall, ResolvedVar, ()>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Schedule<Head, Leaf, Ann> {
    Saturate(Box<Schedule<Head, Leaf, Ann>>),
    Repeat(usize, Box<Schedule<Head, Leaf, Ann>>),
    Run(RunConfig<Head, Leaf, Ann>),
    Sequence(Vec<Schedule<Head, Leaf, Ann>>),
}

impl<Head, Leaf, Ann> Schedule<Head, Leaf, Ann> {
    pub fn saturate(self) -> Self {
        Schedule::Saturate(Box::new(self))
    }
}

impl UnresolvedSchedule {
    pub fn map_run_commands(
        &self,
        f: &mut impl FnMut(&UnresolvedRunConfig) -> UnresolvedSchedule,
    ) -> UnresolvedSchedule {
        match self {
            Schedule::Run(config) => f(config),
            Schedule::Saturate(sched) => Schedule::Saturate(Box::new(sched.map_run_commands(f))),
            Schedule::Repeat(size, sched) => {
                Schedule::Repeat(*size, Box::new(sched.map_run_commands(f)))
            }
            Schedule::Sequence(scheds) => Schedule::Sequence(
                scheds
                    .iter()
                    .map(|sched| sched.map_run_commands(f))
                    .collect(),
            ),
        }
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

impl<Head: Display, Leaf: Display, Ann> ToSexp for Schedule<Head, Leaf, Ann> {
    fn to_sexp(&self) -> Sexp {
        match self {
            Schedule::Saturate(sched) => list!("saturate", sched),
            Schedule::Repeat(size, sched) => list!("repeat", size, sched),
            Schedule::Run(config) => config.to_sexp(),
            Schedule::Sequence(scheds) => list!("seq", ++ scheds),
        }
    }
}

impl<Head: Display, Leaf: Display, Ann> Display for Schedule<Head, Leaf, Ann> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp())
    }
}

pub type UnresolvedCommand = Command<Symbol, Symbol>;

// TODO command before and after desugaring should be different
/// A [`Command`] is the top-level construct in egglog.
/// It includes defining rules, declaring functions,
/// adding to tables, and running rules (via a [`Schedule`]).
#[derive(Debug, Clone)]
pub enum Command<Head, Leaf> {
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
        value: Expr<Head, Leaf, ()>,
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
    Sort(Symbol, Option<(Symbol, Vec<Expr<Symbol, Symbol, ()>>)>),
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
    Function(FunctionDecl<Head, Leaf, ()>),
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
        rule: Rule<Head, Leaf, ()>,
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
    Rewrite(Symbol, Rewrite<Head, Leaf, ()>),
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
    BiRewrite(Symbol, Rewrite<Head, Leaf, ()>),
    /// Perform an [`Action`] on the global database
    /// (see documentation for [`Action`] for more details).
    /// Example:
    /// ```text
    /// (let xplusone (Add (Var "x") (Num 1)))
    /// ```
    Action(Action<Head, Leaf, ()>),
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
    RunSchedule(Schedule<Head, Leaf, ()>),
    /// Print runtime statistics about rules
    /// and rulesets so far.
    PrintOverallStatistics,
    // TODO provide simplify docs
    Simplify {
        expr: Expr<Head, Leaf, ()>,
        schedule: Schedule<Head, Leaf, ()>,
    },
    // TODO provide calc docs
    Calc(Vec<IdentSort>, Vec<Expr<Head, Leaf, ()>>),
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
        expr: Expr<Head, Leaf, ()>,
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
    Check(Vec<Fact<Head, Leaf, ()>>),
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
        exprs: Vec<Expr<Head, Leaf, ()>>,
    },
    /// `push` the current egraph `n` times so that it is saved.
    /// Later, the current database and rules can be restored using `pop`.
    Push(usize),
    /// `pop` the current egraph, restoring the previous one.
    /// The argument specifies how many egraphs to pop.
    Pop(usize),
    /// Assert that a command fails with an error.
    Fail(Box<Command<Head, Leaf>>),
    /// Include another egglog file directly as text and run it.
    Include(String),
}

impl<Head: Display + ToSexp, Leaf: Display + ToSexp> ToSexp for Command<Head, Leaf> {
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

impl<Head: Display + Clone + ToSexp, Leaf: Display + ToSexp + Clone> Display
    for NCommand<Head, Leaf, ()>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_command())
    }
}

impl<Head: Display + ToSexp, Leaf: Display + ToSexp> Display for Command<Head, Leaf> {
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

pub type UnresolvedRunConfig = RunConfig<Symbol, Symbol, ()>;
pub(crate) type ResolvedRunConfig = RunConfig<ResolvedCall, ResolvedVar, ()>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RunConfig<Head, Leaf, Ann> {
    pub ruleset: Symbol,
    pub until: Option<Vec<Fact<Head, Leaf, Ann>>>,
}

impl<Head: Display, Leaf: Display, Ann> ToSexp for RunConfig<Head, Leaf, Ann>
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

pub type UnresolvedFunctionDecl = FunctionDecl<Symbol, Symbol, ()>;
pub(crate) type ResolvedFunctionDecl = FunctionDecl<ResolvedCall, ResolvedVar, ()>;

/// Represents the declaration of a function
/// directly parsed from source syntax.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionDecl<Head, Leaf, Ann> {
    pub name: Symbol,
    pub schema: Schema,
    pub default: Option<Expr<Head, Leaf, Ann>>,
    pub merge: Option<Expr<Head, Leaf, Ann>>,
    pub merge_action: Vec<Action<Head, Leaf, Ann>>,
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

impl UnresolvedFunctionDecl {
    pub fn relation(name: Symbol, input: Vec<Symbol>) -> Self {
        Self {
            name,
            schema: Schema {
                input,
                output: Symbol::from("Unit"),
            },
            merge: None,
            merge_action: vec![],
            default: Some(Expr::Lit((), Literal::Unit)),
            cost: None,
            unextractable: false,
        }
    }
}

impl<Head: Display + ToSexp, Leaf: Display + ToSexp, Ann> ToSexp for FunctionDecl<Head, Leaf, Ann> {
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

pub type UnresolvedFact = Fact<Symbol, Symbol, ()>;
pub(crate) type ResolvedFact = Fact<ResolvedCall, ResolvedVar, ()>;

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
pub enum Fact<Head, Leaf, Ann> {
    /// Must be at least two things in an eq fact
    Eq(Vec<Expr<Head, Leaf, Ann>>),
    Fact(Expr<Head, Leaf, Ann>),
}

pub struct Facts<Head, Leaf, Ann>(pub Vec<Fact<Head, Leaf, Ann>>);

impl<Head, Leaf, Ann> Facts<Head, Leaf, Ann>
where
    Head: Clone,
    Leaf: Clone + Into<Symbol>,
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
        Vec<Fact<(Head, Leaf), Leaf, Ann>>,
    ) {
        let mut atoms = vec![];
        let mut new_body = vec![];

        for fact in self.0.iter() {
            match fact {
                Fact::Eq(exprs) => {
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
                    new_body.push(Fact::Eq(new_exprs));
                }
                Fact::Fact(expr) => {
                    let (child_atoms, expr) = expr.to_query(typeinfo, fresh_gen);
                    atoms.extend(child_atoms);
                    new_body.push(Fact::Fact(expr));
                }
            }
        }
        (Query { atoms }, new_body)
    }
}

impl<Head: Display, Leaf: Display, Ann> ToSexp for Fact<Head, Leaf, Ann>
where
    Head: Display,
    Leaf: Display,
{
    fn to_sexp(&self) -> Sexp {
        match self {
            Fact::Eq(exprs) => list!("=", ++ exprs),
            Fact::Fact(expr) => expr.to_sexp(),
        }
    }
}

impl<Head, Leaf, Ann> Fact<Head, Leaf, Ann>
where
    Ann: Clone,
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub(crate) fn map_exprs(
        &self,
        f: &mut impl FnMut(&Expr<Head, Leaf, Ann>) -> Expr<Head, Leaf, Ann>,
    ) -> Self {
        match self {
            Fact::Eq(exprs) => Fact::Eq(exprs.iter().map(f).collect()),
            Fact::Fact(expr) => Fact::Fact(f(expr)),
        }
    }

    pub(crate) fn subst(&self, subst: &HashMap<Leaf, Expr<Head, Leaf, Ann>>) -> Self {
        self.map_exprs(&mut |e| e.subst(subst))
    }
}

impl<Head, Leaf, Ann> Fact<Head, Leaf, Ann> {
    pub(crate) fn to_unresolved(&self) -> Fact<Symbol, Symbol, ()> {
        todo!()
    }
}

impl<Head, Leaf, Ann> Display for Fact<Head, Leaf, Ann>
where
    Head: Display,
    Leaf: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp())
    }
}

pub type UnresolvedAction = Action<Symbol, Symbol, ()>;
pub(crate) type ResolvedAction = Action<ResolvedCall, ResolvedVar, ()>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Action<Head, Leaf, Ann> {
    /// Bind a variable to a particular datatype or primitive.
    /// At the top level (in a [`Command::Action`]), this defines a global variable.
    /// In a [`Command::Rule`], this defines a local variable in the actions.
    Let(Ann, Leaf, Expr<Head, Leaf, Ann>),
    /// `set` a function to a particular result.
    /// `set` should not be used on datatypes-
    /// instead, use `union`.
    Set(Ann, Head, Vec<Expr<Head, Leaf, Ann>>, Expr<Head, Leaf, Ann>),
    /// `delete` an entry from a function.
    /// Be wary! Only delete entries that are subsumed in some way or
    /// guaranteed to be not useful.
    Delete(Ann, Head, Vec<Expr<Head, Leaf, Ann>>),
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
    Union(Ann, Expr<Head, Leaf, Ann>, Expr<Head, Leaf, Ann>),
    /// `extract` a datatype from the egraph, choosing
    /// the smallest representative.
    /// By default, each constructor costs 1 to extract
    /// (common subexpressions are not shared in the cost
    /// model).
    /// The second argument is the number of variants to
    /// extract, picking different terms in the
    /// same equivalence class.
    Extract(Ann, Expr<Head, Leaf, Ann>, Expr<Head, Leaf, Ann>),
    Panic(Ann, String),
    Expr(Ann, Expr<Head, Leaf, Ann>),
    // If(Expr, Action, Action),
}

pub struct Actions<Head, Leaf>(pub Vec<Action<Head, Leaf, ()>>);

impl<Head, Leaf> Actions<Head, Leaf>
where
    Head: Clone,
    Leaf: Clone + Hash + Eq + Clone + Into<Symbol>,
{
    pub(crate) fn to_norm_actions<FG: FreshGen<Head, Leaf>>(
        &self,
        typeinfo: &TypeInfo,
        binding: &mut HashSet<Leaf>,
        fresh_gen: &mut FG,
    ) -> Result<
        (
            Vec<CoreAction<Head, Leaf>>,
            Vec<Action<(Head, Leaf), Leaf, ()>>,
        ),
        TypeError,
    > {
        let mut norm_actions = vec![];
        let mut mapped_actions = vec![];

        // During the lowering, there are two important guaratees:
        //   Every used variable should be bound.
        //   Every introduced variable should be unbound before.
        for action in self.0.iter() {
            match action {
                Action::Let(_ann, var, expr) => {
                    if binding.contains(var) {
                        return Err(TypeError::AlreadyDefined(var.clone().into()));
                    }
                    let (actions, mapped_expr) =
                        expr.to_norm_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions);
                    norm_actions.push(CoreAction::LetAtomTerm(
                        var.clone(),
                        mapped_expr.get_corresponding_var_or_lit(typeinfo),
                    ));
                    mapped_actions.push(Action::Let(*_ann, var.clone(), mapped_expr));
                    binding.insert(var.clone());
                }
                Action::Set(_ann, head, args, expr) => {
                    let mut mapped_args = vec![];
                    for arg in args {
                        let (actions, mapped_arg) =
                            arg.to_norm_actions(typeinfo, binding, fresh_gen)?;
                        norm_actions.extend(actions);
                        mapped_args.push(mapped_arg);
                    }
                    let (actions, mapped_expr) =
                        expr.to_norm_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions);
                    norm_actions.push(CoreAction::Set(
                        head.clone(),
                        mapped_args
                            .iter()
                            .map(|e| e.get_corresponding_var_or_lit(typeinfo))
                            .collect(),
                        mapped_expr.get_corresponding_var_or_lit(typeinfo),
                    ));
                    let v = fresh_gen.fresh(head);
                    mapped_actions.push(Action::Set(
                        *_ann,
                        (head.clone(), v),
                        mapped_args,
                        mapped_expr,
                    ));
                }
                Action::Delete(_ann, head, args) => {
                    let mut mapped_args = vec![];
                    for arg in args {
                        let (actions, mapped_arg) =
                            arg.to_norm_actions(typeinfo, binding, fresh_gen)?;
                        norm_actions.extend(actions);
                        mapped_args.push(mapped_arg);
                    }
                    norm_actions.push(CoreAction::Delete(
                        head.clone(),
                        mapped_args
                            .iter()
                            .map(|e| e.get_corresponding_var_or_lit(typeinfo))
                            .collect(),
                    ));
                    let v = fresh_gen.fresh(head);
                    mapped_actions.push(Action::Delete(*_ann, (head.clone(), v), mapped_args));
                }
                Action::Union(_ann, e1, e2) => {
                    let (actions1, mapped_e1) = e1.to_norm_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions1);
                    let (actions2, mapped_e2) = e2.to_norm_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions2);
                    norm_actions.push(CoreAction::Union(
                        mapped_e1.get_corresponding_var_or_lit(typeinfo),
                        mapped_e2.get_corresponding_var_or_lit(typeinfo),
                    ));
                    mapped_actions.push(Action::Union(*_ann, mapped_e1, mapped_e2));
                }
                Action::Extract(_ann, e, n) => {
                    let (actions, mapped_e) = e.to_norm_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions);
                    let (actions, mapped_n) = n.to_norm_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions);
                    norm_actions.push(CoreAction::Extract(
                        mapped_e.get_corresponding_var_or_lit(typeinfo),
                        mapped_n.get_corresponding_var_or_lit(typeinfo),
                    ));
                    mapped_actions.push(Action::Extract(*_ann, mapped_e, mapped_n));
                }
                Action::Panic(_ann, string) => {
                    norm_actions.push(CoreAction::Panic(string.clone()));
                    mapped_actions.push(Action::Panic(*_ann, string.clone()));
                }
                Action::Expr(_ann, expr) => {
                    let (actions, mapped_expr) =
                        expr.to_norm_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions);
                    mapped_actions.push(Action::Expr(*_ann, mapped_expr));
                }
            }
        }
        Ok((norm_actions, mapped_actions))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CoreAction<Head, Leaf> {
    Let(Leaf, Head, Vec<GenericAtomTerm<Leaf>>),
    LetAtomTerm(Leaf, GenericAtomTerm<Leaf>),
    Extract(GenericAtomTerm<Leaf>, GenericAtomTerm<Leaf>),
    Set(Head, Vec<GenericAtomTerm<Leaf>>, GenericAtomTerm<Leaf>),
    Delete(Head, Vec<GenericAtomTerm<Leaf>>),
    Union(GenericAtomTerm<Leaf>, GenericAtomTerm<Leaf>),
    Panic(String),
}

pub type NormAction = CoreAction<Symbol, Symbol>;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct CoreActions<Head, Leaf>(pub(crate) Vec<CoreAction<Head, Leaf>>);
impl<Head, Leaf> CoreActions<Head, Leaf>
where
    Leaf: Clone,
{
    pub(crate) fn subst(&mut self, subst: &HashMap<Leaf, GenericAtomTerm<Leaf>>) {
        let actions = subst
            .iter()
            .map(|(symbol, atom_term)| CoreAction::LetAtomTerm(symbol.clone(), atom_term.clone()));
        let existing_actions = std::mem::take(&mut self.0);
        self.0 = actions.chain(existing_actions).collect();
    }
}

impl<Head: Display + ToSexp, Leaf: Display + ToSexp, Ann> ToSexp for Action<Head, Leaf, Ann> {
    fn to_sexp(&self) -> Sexp {
        match self {
            Action::Let(_ann, lhs, rhs) => list!("let", lhs, rhs),
            Action::Set(_ann, lhs, args, rhs) => list!("set", list!(lhs, ++ args), rhs),
            Action::Union(_ann, lhs, rhs) => list!("union", lhs, rhs),
            Action::Delete(_ann, lhs, args) => list!("delete", list!(lhs, ++ args)),
            Action::Extract(_ann, expr, variants) => list!("extract", expr, variants),
            Action::Panic(_ann, msg) => list!("panic", format!("\"{}\"", msg.clone())),
            Action::Expr(_ann, e) => e.to_sexp(),
        }
    }
}

impl<Head, Leaf, Ann> Action<Head, Leaf, Ann>
where
    Head: Clone + Display,
    Leaf: Clone + Eq + Display + Hash,
    Ann: Clone,
{
    pub fn map_exprs(
        &self,
        f: &mut impl FnMut(&Expr<Head, Leaf, Ann>) -> Expr<Head, Leaf, Ann>,
    ) -> Self {
        match self {
            Action::Let(ann, lhs, rhs) => Action::Let(ann.clone(), lhs.clone(), f(rhs)),
            Action::Set(ann, lhs, args, rhs) => {
                let right = f(rhs);
                Action::Set(
                    ann.clone(),
                    lhs.clone(),
                    args.iter().map(f).collect(),
                    right,
                )
            }
            Action::Delete(ann, lhs, args) => {
                Action::Delete(ann.clone(), lhs.clone(), args.iter().map(f).collect())
            }
            Action::Union(ann, lhs, rhs) => Action::Union(ann.clone(), f(lhs), f(rhs)),
            Action::Extract(ann, expr, variants) => {
                Action::Extract(ann.clone(), f(expr), f(variants))
            }
            Action::Panic(ann, msg) => Action::Panic(ann.clone(), msg.clone()),
            Action::Expr(ann, e) => Action::Expr(ann.clone(), f(e)),
        }
    }

    pub fn replace_canon(&self, canon: &HashMap<Leaf, Expr<Head, Leaf, Ann>>) -> Self {
        match self {
            Action::Let(ann, lhs, rhs) => Action::Let(ann.clone(), lhs.clone(), rhs.subst(canon)),
            Action::Set(ann, lhs, args, rhs) => Action::Set(
                ann.clone(),
                lhs.clone(),
                args.iter().map(|e| e.subst(canon)).collect(),
                rhs.subst(canon),
            ),
            Action::Delete(ann, lhs, args) => Action::Delete(
                ann.clone(),
                lhs.clone(),
                args.iter().map(|e| e.subst(canon)).collect(),
            ),
            Action::Union(ann, lhs, rhs) => {
                Action::Union(ann.clone(), lhs.subst(canon), rhs.subst(canon))
            }
            Action::Extract(ann, expr, variants) => {
                Action::Extract(ann.clone(), expr.subst(canon), variants.subst(canon))
            }
            Action::Panic(ann, msg) => Action::Panic(ann.clone(), msg.clone()),
            Action::Expr(ann, e) => Action::Expr(ann.clone(), e.subst(canon)),
        }
    }

    fn map_def_use(&self, mut fvar: impl FnMut(Leaf, bool) -> Leaf) -> Self {
        match self {
            Action::Let(ann, lhs, rhs) => {
                let lhs = fvar(lhs.clone(), true);
                let rhs = rhs.map_def_use(&mut |s| fvar(s, false));
                Action::Let(ann.clone(), lhs, rhs)
            }
            Action::Set(ann, lhs, args, rhs) => {
                let args = args
                    .iter()
                    .map(|e| e.map_def_use(&mut |s| fvar(s, false)))
                    .collect();
                let rhs = rhs.map_def_use(&mut |s| fvar(s, false));
                Action::Set(ann.clone(), lhs.clone(), args, rhs)
            }
            Action::Delete(ann, lhs, args) => {
                let args = args
                    .iter()
                    .map(|e| e.map_def_use(&mut |s| fvar(s, false)))
                    .collect();
                Action::Delete(ann.clone(), lhs.clone(), args)
            }
            Action::Union(ann, lhs, rhs) => {
                let lhs = lhs.map_def_use(&mut |s| fvar(s, false));
                let rhs = rhs.map_def_use(&mut |s| fvar(s, false));
                Action::Union(ann.clone(), lhs, rhs)
            }
            Action::Extract(ann, expr, variants) => {
                let expr = expr.map_def_use(&mut |s| fvar(s, false));
                let variants = variants.map_def_use(&mut |s| fvar(s, false));
                Action::Extract(ann.clone(), expr, variants)
            }
            Action::Panic(ann, msg) => Action::Panic(ann.clone(), msg.clone()),
            Action::Expr(ann, e) => {
                Action::Expr(ann.clone(), e.map_def_use(&mut |s| fvar(s, false)))
            }
        }
    }
}

impl<Head: Display + ToSexp, Leaf: Display + ToSexp, Ann> Display for Action<Head, Leaf, Ann> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp())
    }
}

pub type UnresolvedRule = Rule<Symbol, Symbol, ()>;
pub(crate) type ResolvedRule = Rule<ResolvedCall, ResolvedVar, ()>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Rule<Head, Leaf, Ann> {
    pub head: Vec<Action<Head, Leaf, Ann>>,
    pub body: Vec<Fact<Head, Leaf, Ann>>,
}

impl<Head: Display + ToSexp, Leaf: Display + ToSexp, Ann> Rule<Head, Leaf, Ann> {
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

impl<Head: Display + ToSexp, Leaf: Display + ToSexp, Ann> Rule<Head, Leaf, Ann> {
    /// Converts this rule into an s-expression.
    pub fn to_sexp(&self, ruleset: Symbol, name: Symbol) -> Sexp {
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
}
impl UnresolvedRule {
    pub(crate) fn map_exprs(&self, f: &mut impl FnMut(&UnresolvedExpr) -> UnresolvedExpr) -> Self {
        Rule {
            head: self.head.iter().map(|a| a.map_exprs(f)).collect(),
            body: self.body.iter().map(|fact| fact.map_exprs(f)).collect(),
        }
    }
}

impl<Head: Display + ToSexp, Leaf: Display + ToSexp, Ann> Display for Rule<Head, Leaf, Ann> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_with_ruleset(f, "".into(), "".into())
    }
}

type UnresolvedRewrite = Rewrite<Symbol, Symbol, ()>;

#[derive(Clone, Debug)]
pub struct Rewrite<Head, Leaf, Ann> {
    pub lhs: Expr<Head, Leaf, Ann>,
    pub rhs: Expr<Head, Leaf, Ann>,
    pub conditions: Vec<Fact<Head, Leaf, Ann>>,
}

impl<Head: Display, Leaf: Display, Ann> Rewrite<Head, Leaf, Ann> {
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

impl<Head: Display, Leaf: Display, Ann> Display for Rewrite<Head, Leaf, Ann> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp("".into(), false))
    }
}

impl<Head, Leaf: Clone + Into<Symbol>, Ann> Expr<(Head, Leaf), Leaf, Ann> {
    fn get_corresponding_var_or_lit(&self, typeinfo: &TypeInfo) -> GenericAtomTerm<Leaf> {
        // Note: need typeinfo to resolve whether a symbol is a global or not
        // This is error-prone and the complexities can be avoided by treating globals
        // as nullary functions.
        match self {
            Expr::Var(_ann, v) => {
                if typeinfo.is_global(v.clone().into()) {
                    GenericAtomTerm::Global(v.clone())
                } else {
                    GenericAtomTerm::Var(v.clone())
                }
            }
            Expr::Lit(_ann, lit) => GenericAtomTerm::Literal(lit.clone()),
            Expr::Call(_ann, head, _) => GenericAtomTerm::Var(head.1.clone()),
        }
    }
}

impl<Head: Clone, Leaf: Clone + Into<Symbol>, Ann: Clone> Expr<Head, Leaf, Ann> {
    fn to_query(
        &self,
        typeinfo: &TypeInfo,
        fresh_gen: &mut impl FreshGen<Head, Leaf>,
    ) -> (
        Vec<GenericAtom<HeadOrEq<Head>, Leaf>>,
        Expr<(Head, Leaf), Leaf, Ann>,
    ) {
        match self {
            Expr::Lit(ann, lit) => (vec![], Expr::Lit(ann.clone(), lit.clone())),
            Expr::Var(ann, v) => (vec![], Expr::Var(ann.clone(), v.clone())),
            Expr::Call(ann, f, children) => {
                let fresh = fresh_gen.fresh(f);
                let mut new_children = vec![];
                let mut atoms = vec![];
                let mut child_exprs = vec![];
                for child in children {
                    let (child_atoms, child_expr) = child.to_query(typeinfo, fresh_gen);
                    let child_atomterm = child_expr.get_corresponding_var_or_lit(typeinfo);
                    new_children.push(child_atomterm);
                    atoms.extend(child_atoms);
                    child_exprs.push(child_expr);
                }
                let args = {
                    new_children.push(GenericAtomTerm::Var(fresh.clone()));
                    new_children
                };
                atoms.push(GenericAtom {
                    head: HeadOrEq::Symbol(f.clone()),
                    args,
                });
                (
                    atoms,
                    Expr::Call(ann.clone(), (f.clone(), fresh), child_exprs),
                )
            }
        }
    }

    pub(crate) fn to_norm_actions<FG: FreshGen<Head, Leaf>>(
        &self,
        typeinfo: &TypeInfo,
        binding: &mut HashSet<Leaf>,
        fresh_gen: &mut FG,
    ) -> Result<(Vec<CoreAction<Head, Leaf>>, Expr<(Head, Leaf), Leaf, ()>), TypeError>
    where
        Leaf: Clone + Hash + Eq + Clone + Into<Symbol>,
        Head: Clone,
    {
        match self {
            Expr::Lit(_ann, lit) => Ok((vec![], Expr::Lit((), lit.clone()))),
            Expr::Var(_ann, v) => {
                let sym = v.clone().into();
                if binding.contains(v) || typeinfo.is_global(sym) {
                    Ok((vec![], Expr::Var((), v.clone())))
                } else {
                    Err(TypeError::Unbound(sym))
                }
            }
            Expr::Call(_ann, f, args) => {
                let mut norm_actions = vec![];
                let mut norm_args = vec![];
                let mut mapped_args = vec![];
                for arg in args {
                    let (actions, mapped_arg) =
                        arg.to_norm_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions);
                    norm_args.push(mapped_arg.get_corresponding_var_or_lit(typeinfo));
                    mapped_args.push(mapped_arg);
                }

                let var = fresh_gen.fresh(f);
                binding.insert(var.clone());

                norm_actions.push(CoreAction::Let(var.clone(), f.clone(), norm_args));
                Ok((norm_actions, Expr::Call((), (f.clone(), var), mapped_args)))
            }
        }
    }
}
