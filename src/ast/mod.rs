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
    Rule(Rule),
    Rewrite(Rewrite),
    BiRewrite(Rewrite),
    Action(Action),
    Run(RunConfig),
    Calc(Vec<IdentSort>, Vec<Expr>),
    Extract {
        variants: usize,
        e: Expr,
    },
    // TODO: this could just become an empty query
    Check(Fact),
    ClearRules,
    AddRuleset(Symbol),
    LoadRuleset(Symbol),
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
    Include(String),
}

impl Command {
    fn to_sexp(&self) -> Sexp {
        match self {
            Command::Rewrite(_) | Command::BiRewrite(_) => {
                panic!("Rewrites should be desugared before printing");
            }
            Command::Datatype {
                name: _,
                variants: _,
            } => {
                panic!("Datatypes should be desugared before printing");
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
            Command::Rule(r) => r.to_sexp(),
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
            Command::Run(limit) => {
                let mut res = vec![
                    Sexp::String("run".into()),
                    Sexp::String(limit.limit.to_string()),
                ];
                if let Some(until) = &limit.until {
                    res.push(Sexp::String(":until".into()));
                    res.push(until.to_sexp());
                }

                Sexp::List(res)
            }
            Command::AddRuleset(name) => Sexp::List(vec![
                Sexp::String("add-ruleset".into()),
                Sexp::String(name.to_string()),
            ]),
            Command::LoadRuleset(name) => Sexp::List(vec![
                Sexp::String("load-ruleset".into()),
                Sexp::String(name.to_string()),
            ]),
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
            Command::ClearRules => Sexp::List(vec![Sexp::String("clear-rules".into())]),
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
        }
    }
}

impl Display for Command {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp())
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
            res.push(Sexp::String(":merge-action".into()));
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

// TODO make a new type for flattened facts
// after flattening, they always have the form
// var = expr
#[derive(Clone, Debug)]
pub enum Fact {
    /// Must be at least two things in an eq fact
    Eq(Vec<Expr>),
    Fact(Expr),
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
}

impl Display for Fact {
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

impl Rule {
    pub(crate) fn to_sexp(&self) -> Sexp {
        Sexp::List(vec![
            Sexp::String("rule".into()),
            Sexp::List(self.body.iter().map(|f| f.to_sexp()).collect()),
            Sexp::List(self.head.iter().map(|a| a.to_sexp()).collect()),
        ])
    }
}

impl Display for Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp())
    }
}

#[derive(Clone, Debug)]
pub struct Rewrite {
    pub lhs: Expr,
    pub rhs: Expr,
    pub conditions: Vec<Fact>,
}
