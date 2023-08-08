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
    Function(FunctionDecl),
    AddRuleset(Symbol),
    NormRule {
        name: Symbol,
        ruleset: Symbol,
        rule: NormRule,
    },
    NormAction(NormAction),
    RunSchedule(NormSchedule),
    Check(Vec<NormFact>),
    CheckProof,
    PrintTable(Symbol, usize),
    PrintSize(Symbol),
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
            NCommand::Function(f) => Command::Function(f.clone()),
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
            NCommand::NormAction(action) => Command::Action(action.to_action()),
            NCommand::Check(facts) => {
                Command::Check(facts.iter().map(|fact| fact.to_fact()).collect())
            }
            NCommand::CheckProof => Command::CheckProof,
            NCommand::PrintTable(name, n) => Command::PrintTable(*name, *n),
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
        Sexp::String(String::from(self))
    }
}

impl ToSexp for Symbol {
    fn to_sexp(&self) -> Sexp {
        Sexp::String(self.to_string())
    }
}

impl ToSexp for usize {
    fn to_sexp(&self) -> Sexp {
        Sexp::String(self.to_string())
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
#[derive(Debug, Clone)]
pub enum Command {
    SetOption {
        name: Symbol,
        value: Expr,
    },
    Datatype {
        name: Symbol,
        variants: Vec<Variant>,
    },
    Declare {
        name: Symbol,
        sort: Symbol,
    },
    Sort(Symbol, Option<(Symbol, Vec<Expr>)>),
    Function(FunctionDecl),
    AddRuleset(Symbol),
    Rule {
        name: Symbol,
        ruleset: Symbol,
        rule: Rule,
    },
    Rewrite(Symbol, Rewrite),
    BiRewrite(Symbol, Rewrite),
    Action(Action),
    RunSchedule(Schedule),
    Simplify {
        expr: Expr,
        schedule: Schedule,
    },
    Calc(Vec<IdentSort>, Vec<Expr>),
    Extract {
        variants: usize,
        fact: Fact,
    },
    // TODO: this could just become an empty query
    Check(Vec<Fact>),
    CheckProof,
    PrintTable(Symbol, usize),
    PrintSize(Symbol),
    Input {
        name: Symbol,
        file: String,
    },
    Output {
        file: String,
        exprs: Vec<Expr>,
    },
    Push(usize),
    Pop(usize),
    Fail(Box<Command>),
    // TODO desugar include
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
            Command::AddRuleset(name) => list!("ruleset", name),
            Command::Rule {
                name,
                ruleset,
                rule,
            } => rule.to_sexp(*ruleset, *name),
            Command::RunSchedule(sched) => list!("run-schedule", sched),
            Command::Calc(args, exprs) => list!("calc", list!(++ args), ++ exprs),
            Command::Extract { variants, fact } => list!("extract", ":variants", variants, fact),
            Command::Check(facts) => list!("check", ++ facts),
            Command::CheckProof => list!("check-proof"),
            Command::Push(n) => list!("push", n),
            Command::Pop(n) => list!("pop", n),
            Command::PrintTable(name, n) => list!("print-table", name, n),
            Command::PrintSize(name) => list!("print-size", name),
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
        let mut res = vec![Sexp::String("run".into())];
        if self.ruleset != "".into() {
            res.push(Sexp::String(self.ruleset.to_string()));
        }
        if let Some(until) = &self.until {
            res.push(Sexp::String(":until".into()));
            res.extend(until.iter().map(|fact| fact.to_sexp()));
        }

        Sexp::List(res)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NormRunConfig {
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionDecl {
    pub name: Symbol,
    pub schema: Schema,
    pub default: Option<Expr>,
    // TODO we should desugar merge and merge action
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
        let mut res = vec![Sexp::String(self.name.to_string())];
        if !self.types.is_empty() {
            res.extend(self.types.iter().map(|s| Sexp::String(s.to_string())));
        }
        if let Some(cost) = self.cost {
            res.push(Sexp::String(":cost".into()));
            res.push(Sexp::String(cost.to_string()));
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
            default: None,
            cost: None,
            unextractable: false,
        }
    }
}

impl ToSexp for FunctionDecl {
    fn to_sexp(&self) -> Sexp {
        let mut res = vec![
            Sexp::String("function".into()),
            Sexp::String(self.name.to_string()),
        ];

        if let Sexp::List(contents) = self.schema.to_sexp() {
            res.extend(contents);
        } else {
            unreachable!();
        }

        if let Some(cost) = self.cost {
            res.extend(vec![
                Sexp::String(":cost".into()),
                Sexp::String(cost.to_string()),
            ]);
        }

        if self.unextractable {
            res.push(Sexp::String(":unextractable".into()));
        }

        if !self.merge_action.is_empty() {
            res.push(Sexp::String(":on_merge".into()));
            res.push(Sexp::List(
                self.merge_action.iter().map(|a| a.to_sexp()).collect(),
            ));
        }

        if let Some(merge) = &self.merge {
            res.push(Sexp::String(":merge".into()));
            res.push(merge.to_sexp());
        }

        if let Some(default) = &self.default {
            res.push(Sexp::String(":default".into()));
            res.push(default.to_sexp());
        }

        Sexp::List(res)
    }
}

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
    Let(Symbol, Expr),
    Set(Symbol, Vec<Expr>, Expr),
    Delete(Symbol, Vec<Expr>),
    Union(Expr, Expr),
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
            Sexp::String("rule".into()),
            Sexp::List(self.body.iter().map(|f| f.to_sexp()).collect()),
            Sexp::List(self.head.iter().map(|a| a.to_sexp()).collect()),
        ];
        if ruleset != "".into() {
            res.push(Sexp::String(":ruleset".into()));
            res.push(Sexp::String(ruleset.to_string()));
        }
        if name != "".into() {
            res.push(Sexp::String(":name".into()));
            res.push(Sexp::String(format!("\"{}\"", name)));
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
            Sexp::String(if is_bidirectional {
                "birewrite".into()
            } else {
                "rewrite".into()
            }),
            self.lhs.to_sexp(),
            self.rhs.to_sexp(),
        ];

        if !self.conditions.is_empty() {
            res.push(Sexp::String(":when".into()));
            res.push(Sexp::List(
                self.conditions.iter().map(|f| f.to_sexp()).collect(),
            ));
        }

        if ruleset != "".into() {
            res.push(Sexp::String(":ruleset".into()));
            res.push(Sexp::String(ruleset.to_string()));
        }
        Sexp::List(res)
    }
}

impl Display for Rewrite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp("".into(), false))
    }
}
