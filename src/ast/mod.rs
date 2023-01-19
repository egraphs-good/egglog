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
            Command::Datatype { name, variants } => Sexp::List(
                vec![
                    Sexp::String("datatype".into()),
                    Sexp::String(name.to_string()),
                ]
                .into_iter()
                .chain(variants.iter().map(|v| v.to_sexp()))
                .collect(),
            ),
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
                        .chain(args.iter().map(|e| panic!("TODO"))).collect()
                ),
            ]),

            _ => panic!("TODO"),
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

impl Display for IdentSort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} {})", self.ident, self.sort)
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

impl Variant {
    pub(crate) fn to_sexp(&self) -> Sexp {
        Sexp::List(
            vec![Sexp::String(self.name.to_string())]
                .into_iter()
                .chain(self.types.iter().map(|s| Sexp::String(s.to_string())))
                .chain(if let Some(cost) = self.cost {
                    vec![
                        Sexp::String(":cost".to_string()),
                        Sexp::String(cost.to_string()),
                    ]
                } else {
                    vec![]
                })
                .collect(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct Schema {
    pub input: Vec<Symbol>,
    pub output: Symbol,
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

#[derive(Clone, Debug)]
pub enum Fact {
    /// Must be at least two things in an eq fact
    Eq(Vec<Expr>),
    Fact(Expr),
}

impl Display for Fact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fact::Eq(exprs) => write!(f, "(= {})", ListDisplay(exprs, " ")),
            Fact::Fact(e) => Display::fmt(e, f),
        }
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

impl Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::Let(lhs, rhs) => write!(f, "(define {} {})", lhs, rhs),
            Action::Set(lhs, args, rhs) => {
                write!(f, "(set ({} {}) {})", lhs, ListDisplay(args, " "), rhs)
            }
            Action::Union(lhs, rhs) => write!(f, "(union {} {})", lhs, rhs),
            Action::Panic(msg) => write!(f, "(panic {})", msg),
            Action::Expr(e) => Display::fmt(e, f),
            Action::Delete(sym, args) => write!(f, "(delete ({} {}))", sym, ListDisplay(args, " ")),
            // Action::If(cond, then, else_) => write!(f, "(if {} {} {})", cond, then, else_),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Rule {
    // pub query: Query,
    // pub actions: Vec<Action>,
    pub head: Vec<Action>,
    pub body: Vec<Fact>,
}

impl Display for Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ==> {}",
            ListDisplay(&self.body, " "),
            ListDisplay(&self.head, " ")
        )
    }
}

#[derive(Clone, Debug)]
pub struct Rewrite {
    pub lhs: Expr,
    pub rhs: Expr,
    pub conditions: Vec<Fact>,
}
