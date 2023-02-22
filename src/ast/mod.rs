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

#[derive(Debug, Clone)]
pub enum NormCommand {
    Sort(Symbol, Option<(Symbol, Vec<Expr>)>),
    Function(FunctionDecl),
    AddRuleset(Symbol),
    NormRule(Symbol, NormRule),
    NormAction(NormAction),
    Run(RunConfig),
    RunSchedule(Schedule),
    Simplify { expr: Expr, config: RunConfig },
    // TODO flatten calc, add proof support
    Calc(Vec<IdentSort>, Vec<Expr>),
    Extract { variants: usize, var: Symbol },
    // TODO: this could just become an empty query
    Check(Fact),
    Clear,
    Print(Symbol, usize),
    PrintSize(Symbol),
    Output { file: String, exprs: Vec<Expr> },
    // TODO flatten query
    Query(Vec<Fact>),
    Push(usize),
    Pop(usize),
    Fail(Box<NormCommand>),
    // TODO desugar
    Input { name: Symbol, file: String },
}

impl NormCommand {
    pub fn to_command(&self) -> Command {
        match self {
            NormCommand::Sort(name, params) => Command::Sort(*name, params.clone()),
            NormCommand::Function(f) => Command::Function(f.clone()),
            NormCommand::AddRuleset(name) => Command::AddRuleset(*name),
            NormCommand::NormRule(name, rule) => Command::Rule(*name, rule.to_rule()),
            NormCommand::NormAction(action) => Command::Action(action.to_action()),
            NormCommand::Run(config) => Command::Run(config.clone()),
            NormCommand::RunSchedule(sched) => Command::RunSchedule(sched.clone()),
            NormCommand::Simplify { expr, config } => Command::Simplify {
                expr: expr.clone(),
                config: config.clone(),
            },
            NormCommand::Calc(args, exprs) => Command::Calc(args.clone(), exprs.clone()),
            NormCommand::Extract { variants, var } => Command::Extract {
                variants: *variants,
                e: Expr::Var(*var),
            },
            NormCommand::Check(fact) => Command::Check(fact.clone()),
            NormCommand::Clear => Command::Query(vec![]),
            NormCommand::Print(name, n) => Command::Print(*name, *n),
            NormCommand::PrintSize(name) => Command::PrintSize(*name),
            NormCommand::Output { file, exprs } => Command::Output {
                file: file.to_string(),
                exprs: exprs.clone(),
            },
            NormCommand::Query(facts) => Command::Query(facts.clone()),
            NormCommand::Push(n) => Command::Push(*n),
            NormCommand::Pop(n) => Command::Pop(*n),
            NormCommand::Fail(cmd) => Command::Fail(Box::new(cmd.to_command())),
            NormCommand::Input { name, file } => Command::Input {
                name: *name,
                file: file.clone(),
            },
        }
    }
}

// TODO command before and after desugaring should be different
#[derive(Debug, Clone)]
pub enum Command {
    Datatype {
        name: Symbol,
        variants: Vec<Variant>,
    },
    Sort(Symbol, Option<(Symbol, Vec<Expr>)>),
    Function(FunctionDecl),
    Define {
        name: Symbol,
        expr: Expr,
        cost: Option<usize>,
    },
    AddRuleset(Symbol),
    Rule(Symbol, Rule),
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
    Check(Fact),
    Clear,
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
    Query(Vec<Fact>),
    Push(usize),
    Pop(usize),
    Fail(Box<Command>),
    // TODO desugar include
    Include(String),
}

impl Command {
    fn to_sexp(&self) -> Sexp {
        match self {
            Command::Rewrite(name, rewrite) => rewrite.to_sexp(*name, false),
            Command::BiRewrite(name, rewrite) => rewrite.to_sexp(*name, true),
            Command::Datatype { name, variants } => {
                let mut res = vec![
                    Sexp::String("datatype".into()),
                    Sexp::String(name.to_string()),
                ];
                res.extend(variants.iter().map(|v| v.to_sexp()));
                Sexp::List(res)
            }
            Command::Action(a) => a.to_sexp(),
            Command::Sort(name, None) => Sexp::List(vec![
                Sexp::String("sort".into()),
                Sexp::String(name.to_string()),
            ]),
            Command::Sort(name, Some((name2, args))) => Sexp::List(vec![
                Sexp::String("sort".into()),
                Sexp::String(name.to_string()),
                Sexp::List(
                    vec![Sexp::String(name2.to_string())]
                        .into_iter()
                        .chain(args.iter().map(|e| e.to_sexp()))
                        .collect(),
                ),
            ]),
            Command::Function(f) => f.to_sexp(),
            Command::AddRuleset(name) => Sexp::List(vec![
                Sexp::String("ruleset".into()),
                Sexp::String(name.to_string()),
            ]),
            Command::Rule(ruleset, r) => r.to_sexp(*ruleset),
            Command::Define { name, expr, cost } => {
                let mut res = vec![
                    Sexp::String("define".into()),
                    Sexp::String(name.to_string()),
                    expr.to_sexp(),
                ];
                if let Some(cost) = cost {
                    res.push(Sexp::String(":cost".into()));
                    res.push(Sexp::String(cost.to_string()));
                }

                Sexp::List(res)
            }
            Command::Run(config) => {
                let mut res = vec![Sexp::String("run".into())];
                if config.ruleset != "".into() {
                    res.push(Sexp::String(config.ruleset.to_string()));
                }
                res.push(Sexp::String(config.limit.to_string()));
                if let Some(until) = &config.until {
                    res.push(Sexp::String(":until".into()));
                    res.push(until.to_sexp());
                }

                Sexp::List(res)
            }
            Command::RunSchedule(sched) => {
                Sexp::List(vec![Sexp::String("run-schedule".into()), sched.to_sexp()])
            }
            Command::Calc(args, exprs) => Sexp::List(
                vec![
                    Sexp::String("calc".into()),
                    Sexp::List(args.iter().map(|arg| arg.to_sexp()).collect()),
                ]
                .into_iter()
                .chain(exprs.iter().map(|e| e.to_sexp()))
                .collect(),
            ),
            Command::Extract { variants, e } => Sexp::List(vec![
                Sexp::String("extract".into()),
                Sexp::String(":variants".into()),
                Sexp::String(variants.to_string()),
                e.to_sexp(),
            ]),
            Command::Check(fact) => Sexp::List(vec![Sexp::String("check".into()), fact.to_sexp()]),
            Command::Clear => Sexp::List(vec![Sexp::String("clear".into())]),
            Command::Query(facts) => Sexp::List(
                vec![Sexp::String("query".into())]
                    .into_iter()
                    .chain(facts.iter().map(|f| f.to_sexp()))
                    .collect(),
            ),
            Command::Push(n) => Sexp::List(vec![
                Sexp::String("push".into()),
                Sexp::String(n.to_string()),
            ]),
            Command::Pop(n) => Sexp::List(vec![
                Sexp::String("pop".into()),
                Sexp::String(n.to_string()),
            ]),
            Command::Print(name, n) => Sexp::List(vec![
                Sexp::String("print".into()),
                Sexp::String(name.to_string()),
                Sexp::String(n.to_string()),
            ]),
            Command::PrintSize(name) => Sexp::List(vec![
                Sexp::String("print-size".into()),
                Sexp::String(name.to_string()),
            ]),
            Command::Input { name, file } => Sexp::List(vec![
                Sexp::String("input".into()),
                Sexp::String(name.to_string()),
                Sexp::String(file.to_string()),
            ]),
            Command::Output { file, exprs } => Sexp::List(
                vec![
                    Sexp::String("output".into()),
                    Sexp::String(file.to_string()),
                ]
                .into_iter()
                .chain(exprs.iter().map(|e| e.to_sexp()))
                .collect(),
            ),
            Command::Fail(cmd) => Sexp::List(vec![Sexp::String("fail".into()), cmd.to_sexp()]),
            Command::Include(file) => Sexp::List(vec![
                Sexp::String("include".into()),
                Sexp::String(format!("\"{}\"", file)),
            ]),
            Command::Simplify { expr, config } => {
                let mut res = vec![
                    Sexp::String("simplify".into()),
                    Sexp::String(config.limit.to_string()),
                    expr.to_sexp(),
                ];
                if let Some(until) = &config.until {
                    res.push(Sexp::String(":until".into()));
                    res.push(until.to_sexp());
                }

                Sexp::List(res)
            }
        }
    }
}

impl Display for NormCommand {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_command())
    }
}

impl Display for Command {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Command::Rule(ruleset, r) => r.fmt_with_ruleset(f, *ruleset),
            _ => write!(f, "{}", self.to_sexp()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct IdentSort {
    pub ident: Symbol,
    pub sort: Symbol,
}

impl IdentSort {
    fn to_sexp(&self) -> Sexp {
        Sexp::List(vec![
            Sexp::String(self.ident.to_string()),
            Sexp::String(self.sort.to_string()),
        ])
    }
}

impl Display for IdentSort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp())
    }
}

#[derive(Clone, Debug)]
pub struct RunConfig {
    pub ruleset: Symbol,
    pub limit: usize,
    pub until: Option<Fact>,
}

#[derive(Clone, Debug)]
pub struct FunctionDecl {
    pub name: Symbol,
    pub schema: Schema,
    pub default: Option<Expr>,
    pub merge: Option<Expr>,
    pub merge_action: Vec<Action>,
    pub cost: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct Variant {
    pub name: Symbol,
    pub types: Vec<Symbol>,
    pub cost: Option<usize>,
}

impl Variant {
    pub(crate) fn to_sexp(&self) -> Sexp {
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

#[derive(Clone, Debug)]
pub struct Schema {
    pub input: Vec<Symbol>,
    pub output: Symbol,
}

impl Schema {
    pub(crate) fn to_sexp(&self) -> Sexp {
        Sexp::List(vec![
            Sexp::List(
                self.input
                    .iter()
                    .map(|s| Sexp::String(s.to_string()))
                    .collect(),
            ),
            Sexp::String(self.output.to_string()),
        ])
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

#[derive(Clone, Debug)]
pub enum Fact {
    /// Must be at least two things in an eq fact
    Eq(Vec<Expr>),
    Fact(Expr),
}

#[derive(Clone, Debug)]
pub enum NormFact {
    Assign(Symbol, NormExpr),
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
}

impl Fact {
    pub(crate) fn to_sexp(&self) -> Sexp {
        match self {
            Fact::Eq(exprs) => Sexp::List(
                vec![Sexp::String("=".into())]
                    .into_iter()
                    .chain(exprs.iter().map(|e| e.to_sexp()))
                    .collect(),
            ),
            Fact::Fact(expr) => expr.to_sexp(),
        }
    }

    pub(crate) fn map_exprs(&self, f: &mut impl FnMut(&Expr) -> Expr) -> Fact {
        match self {
            Fact::Eq(exprs) => Fact::Eq(exprs.iter().map(f).collect()),
            Fact::Fact(expr) => Fact::Fact(f(expr)),
        }
    }
}

impl Display for Fact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp())
    }
}

#[derive(Debug, Clone)]
pub enum Schedule {
    Saturate(Box<Schedule>),
    Repeat(usize, Box<Schedule>),
    Ruleset(Symbol),
    Sequence(Vec<Schedule>),
}

impl Schedule {
    fn to_sexp(&self) -> Sexp {
        match self {
            Schedule::Saturate(sched) => {
                Sexp::List(vec![Sexp::String("saturate".into()), sched.to_sexp()])
            }
            Schedule::Repeat(size, sched) => Sexp::List(vec![
                Sexp::String("repeat".into()),
                Sexp::String(size.to_string()),
                sched.to_sexp(),
            ]),
            Schedule::Ruleset(sym) => Sexp::String(sym.to_string()),
            Schedule::Sequence(scheds) => {
                let mut sexps = vec![Sexp::String("seq".into())];
                for sched in scheds {
                    sexps.push(sched.to_sexp());
                }
                Sexp::List(sexps)
            }
        }
    }
}

impl Display for Schedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp())
    }
}

#[derive(Clone, Debug)]
pub enum Action {
    Let(Symbol, Expr),
    Set(Symbol, Vec<Expr>, Expr),
    Delete(Symbol, Vec<Expr>),
    Union(Expr, Expr),
    Panic(String),
    Expr(Expr),
    // If(Expr, Action, Action),
}

#[derive(Clone, Debug)]
pub enum NormAction {
    Let(Symbol, NormExpr),
    LetVar(Symbol, Symbol),
    LetLit(Symbol, Literal),
    Set(Symbol, Vec<Symbol>, Symbol),
    Delete(Symbol, Vec<Symbol>),
    Union(Symbol, Symbol),
    Panic(String),
}

impl NormAction {
    pub fn to_action(&self) -> Action {
        match self {
            NormAction::Let(symbol, expr) => Action::Let(*symbol, expr.to_expr()),
            NormAction::LetVar(symbol, other) => Action::Let(*symbol, Expr::Var(*other)),
            NormAction::LetLit(symbol, lit) => Action::Let(*symbol, Expr::Lit(lit.clone())),
            NormAction::Set(symbol, args, other) => Action::Set(
                *symbol,
                args.iter().map(|s| Expr::Var(*s)).collect(),
                Expr::Var(*other),
            ),
            NormAction::Delete(symbol, args) => {
                Action::Delete(*symbol, args.iter().map(|s| Expr::Var(*s)).collect())
            }
            NormAction::Union(lhs, rhs) => Action::Union(Expr::Var(*lhs), Expr::Var(*rhs)),
            NormAction::Panic(msg) => Action::Panic(msg.clone()),
        }
    }

    // fvar accepts a variable and if it is being defined (true) or used (false)
    pub(crate) fn map_def_use(&self, fvar: &mut impl FnMut(Symbol, bool) -> Symbol) -> NormAction {
        match self {
            NormAction::Let(symbol, expr) => {
                NormAction::Let(fvar(*symbol, true), expr.map_def_use(fvar))
            }
            NormAction::LetVar(symbol, other) => {
                NormAction::LetVar(fvar(*symbol, true), fvar(*other, false))
            }
            NormAction::LetLit(symbol, lit) => NormAction::LetLit(fvar(*symbol, true), lit.clone()),
            NormAction::Set(symbol, args, other) => NormAction::Set(
                *symbol,
                args.iter().map(|s| fvar(*s, false)).collect(),
                fvar(*other, false),
            ),
            NormAction::Delete(symbol, args) => {
                NormAction::Delete(*symbol, args.iter().map(|s| fvar(*s, false)).collect())
            }
            NormAction::Union(lhs, rhs) => NormAction::Union(fvar(*lhs, false), fvar(*rhs, false)),
            NormAction::Panic(msg) => NormAction::Panic(msg.clone()),
        }
    }
}

impl Action {
    pub(crate) fn to_sexp(&self) -> Sexp {
        match self {
            Action::Let(lhs, rhs) => Sexp::List(vec![
                Sexp::String("let".into()),
                Sexp::String(lhs.to_string()),
                rhs.to_sexp(),
            ]),
            Action::Set(lhs, args, rhs) => Sexp::List(vec![
                Sexp::String("set".into()),
                Sexp::List(
                    vec![Sexp::String(lhs.to_string())]
                        .into_iter()
                        .chain(args.iter().map(|e| e.to_sexp()))
                        .collect(),
                ),
                rhs.to_sexp(),
            ]),
            Action::Union(lhs, rhs) => Sexp::List(vec![
                Sexp::String("union".into()),
                lhs.to_sexp(),
                rhs.to_sexp(),
            ]),
            Action::Delete(lhs, args) => Sexp::List(vec![
                Sexp::String("delete".into()),
                Sexp::List(
                    vec![Sexp::String(lhs.to_string())]
                        .into_iter()
                        .chain(args.iter().map(|e| e.to_sexp()))
                        .collect(),
                ),
            ]),
            Action::Panic(msg) => Sexp::List(vec![
                Sexp::String("panic".into()),
                Sexp::String(format!("\"{}\"", msg.clone())),
            ]),
            Action::Expr(e) => e.to_sexp(),
        }
    }

    pub(crate) fn map_exprs(&self, f: &mut impl FnMut(&Expr) -> Expr) -> Self {
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

#[derive(Clone, Debug)]
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
}

impl Display for NormRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_rule())
    }
}

impl Rule {
    pub(crate) fn to_sexp(&self, ruleset: Symbol) -> Sexp {
        let mut res = vec![
            Sexp::String("rule".into()),
            Sexp::List(self.body.iter().map(|f| f.to_sexp()).collect()),
            Sexp::List(self.head.iter().map(|a| a.to_sexp()).collect()),
        ];
        if ruleset != "".into() {
            res.push(Sexp::String(":ruleset".into()));
            res.push(Sexp::String(ruleset.to_string()));
        }
        Sexp::List(res)
    }

    pub(crate) fn map_exprs(&self, f: &mut impl FnMut(&Expr) -> Expr) -> Self {
        Rule {
            head: self.head.iter().map(|a| a.map_exprs(f)).collect(),
            body: self.body.iter().map(|fact| fact.map_exprs(f)).collect(),
        }
    }

    pub(crate) fn fmt_with_ruleset(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        ruleset: Symbol,
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
        if ruleset != "".into() {
            write!(f, ")\n{}:ruleset {})", indent, ruleset)
        } else {
            write!(f, "))")
        }
    }
}

impl Display for Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_with_ruleset(f, "".into())
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
