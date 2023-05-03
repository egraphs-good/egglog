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
    Simplify {
        var: Symbol,
        config: NormRunConfig,
    },
    Extract {
        variants: usize,
        var: Symbol,
    },
    Check(Vec<NormFact>),
    Print(Symbol, usize),
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
            NCommand::Simplify { var, config } => Command::Simplify {
                expr: Expr::Var(*var),
                config: config.to_run_config(),
            },
            NCommand::Extract { variants, var } => Command::Extract {
                variants: *variants,
                e: Expr::Var(*var),
            },
            NCommand::Check(facts) => {
                Command::Check(facts.iter().map(|fact| fact.to_fact()).collect())
            }
            NCommand::Print(name, n) => Command::Print(*name, *n),
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
            NCommand::Simplify { .. } => self.clone(),
            NCommand::Extract { variants, var } => NCommand::Extract {
                variants: *variants,
                var: *var,
            },
            NCommand::Check(facts) => {
                NCommand::Check(facts.iter().map(|fact| fact.map_exprs(f)).collect())
            }
            NCommand::Print(name, n) => NCommand::Print(*name, *n),
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
        cost: Option<usize>,
    },
    Sort(Symbol, Option<(Symbol, Vec<Expr>)>),
    Function(FunctionDecl),
    Define {
        name: Symbol,
        expr: Expr,
    },
    AddRuleset(Symbol),
    Rule {
        name: Symbol,
        ruleset: Symbol,
        rule: Rule,
    },
    Rewrite(Symbol, Rewrite),
    BiRewrite(Symbol, Rewrite),
    Action(Action),
    Run(RunConfig),
    RunSchedule(Schedule),
    Simplify {
        expr: Expr,
        config: RunConfig,
    },
    Calc(Vec<IdentSort>, Vec<Expr>),
    Extract {
        variants: usize,
        e: Expr,
    },
    // TODO: this could just become an empty query
    Check(Vec<Fact>),
    Print(Symbol, usize),
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
            Command::Declare { name, sort, cost } => match cost {
                None => list!("declare", name, sort, ":noextract"),
                Some(cost) => list!("declare", name, sort, ":cost", cost),
            },
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
            Command::Define { name, expr } => list!("define", name, expr),
            Command::Run(config) => config.to_sexp(),
            Command::RunSchedule(sched) => list!("run-schedule", sched),
            Command::Calc(args, exprs) => list!("calc", list!(++ args), ++ exprs),
            Command::Extract { variants, e } => list!("extract", ":variants", variants, e),
            Command::Check(facts) => list!("check", ++ facts),
            Command::Push(n) => list!("push", n),
            Command::Pop(n) => list!("pop", n),
            Command::Print(name, n) => list!("print", name, n),
            Command::PrintSize(name) => list!("print-size", name),
            Command::Input { name, file } => list!("input", name, format!("\"{}\"", file)),
            Command::Output { file, exprs } => list!("output", format!("\"{}\"", file), ++ exprs),
            Command::Fail(cmd) => list!("fail", cmd),
            Command::Include(file) => list!("include", format!("\"{}\"", file)),
            Command::Simplify { expr, config } => match &config.until {
                Some(until) => list!("simplify", config.limit, expr, ":until", ++ until),
                None => list!("simplify", config.limit, expr),
            },
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
    pub limit: usize,
    pub until: Option<Vec<Fact>>,
}

impl ToSexp for RunConfig {
    fn to_sexp(&self) -> Sexp {
        let mut res = vec![Sexp::String("run".into())];
        if self.ruleset != "".into() {
            res.push(Sexp::String(self.ruleset.to_string()));
        }
        res.push(Sexp::String(self.limit.to_string()));
        if let Some(until) = &self.until {
            res.push(Sexp::String(":until".into()));
            res.extend(until.iter().map(|fact| fact.to_sexp()));
        }

        Sexp::List(res)
    }
}

// TODO get rid of limit, just use Repeat
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NormRunConfig {
    pub ruleset: Symbol,
    pub limit: usize,
    pub until: Option<Vec<NormFact>>,
}

impl NormRunConfig {
    pub fn to_run_config(&self) -> RunConfig {
        RunConfig {
            ruleset: self.ruleset,
            limit: self.limit,
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
    pub merge: Option<Expr>,
    pub merge_action: Vec<Action>,
    pub cost: Option<usize>,
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
        } else {
            res.push(Sexp::String(":noextract".into()));
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
        } else {
            res.push(Sexp::String(":noextract".into()));
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
    AssignLit(Symbol, Literal),
    ConstrainEq(Symbol, Symbol),
}

impl NormFact {
    pub fn to_fact(&self) -> Fact {
        match self {
            NormFact::Assign(symbol, expr) => Fact::Eq(vec![Expr::Var(*symbol), expr.to_expr()]),
            NormFact::ConstrainEq(lhs, rhs) => Fact::Eq(vec![Expr::Var(*lhs), Expr::Var(*rhs)]),
            NormFact::AssignLit(symbol, lit) => {
                Fact::Eq(vec![Expr::Var(*symbol), Expr::Lit(lit.clone())])
            }
        }
    }

    pub fn map_exprs(&self, f: &mut impl FnMut(&NormExpr) -> NormExpr) -> NormFact {
        match self {
            NormFact::Assign(symbol, expr) => NormFact::Assign(*symbol, f(expr)),
            NormFact::ConstrainEq(lhs, rhs) => NormFact::ConstrainEq(*lhs, *rhs),
            NormFact::AssignLit(symbol, lit) => NormFact::AssignLit(*symbol, lit.clone()),
        }
    }

    pub(crate) fn map_def_use(&self, fvar: &mut impl FnMut(Symbol, bool) -> Symbol) -> NormFact {
        match self {
            NormFact::Assign(symbol, expr) => {
                NormFact::Assign(fvar(*symbol, true), expr.map_def_use(fvar, true))
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
    Panic(String),
    Expr(Expr),
    // If(Expr, Action, Action),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NormAction {
    Let(Symbol, NormExpr),
    LetVar(Symbol, Symbol),
    LetLit(Symbol, Literal),
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
            Action::Panic(msg) => Action::Panic(msg.clone()),
            Action::Expr(e) => Action::Expr(f(e)),
        }
    }

    pub fn replace_canon(&self, canon: &HashMap<Symbol, Expr>) -> Self {
        match self {
            Action::Let(lhs, rhs) => Action::Let(*lhs, rhs.replace_canon(canon)),
            Action::Set(lhs, args, rhs) => Action::Set(
                *lhs,
                args.iter().map(|e| e.replace_canon(canon)).collect(),
                rhs.replace_canon(canon),
            ),
            Action::Delete(lhs, args) => {
                Action::Delete(*lhs, args.iter().map(|e| e.replace_canon(canon)).collect())
            }
            Action::Union(lhs, rhs) => {
                Action::Union(lhs.replace_canon(canon), rhs.replace_canon(canon))
            }
            Action::Panic(msg) => Action::Panic(msg.clone()),
            Action::Expr(e) => Action::Expr(e.replace_canon(canon)),
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
