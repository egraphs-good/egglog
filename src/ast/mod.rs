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

#[derive(Debug)]
pub enum Command {
    Datatype {
        name: Symbol,
        variants: Vec<Variant>,
    },
    Sort(Symbol, Symbol, Vec<Expr>),
    Function(FunctionDecl),
    Define {
        name: Symbol,
        expr: Expr,
        cost: Option<usize>,
    },
    Rule(Rule),
    Rewrite(Rewrite),
    Action(Action),
    Run(usize),
    Extract {
        variants: usize,
        e: Expr,
    },
    // TODO: this could just become an empty query
    Check(Fact),
    ClearRules,
    Clear,
    Print(Symbol, usize),
    PrintSize(Symbol),
    Input {
        name: Symbol,
        file: String,
    },
    Query(Vec<Fact>),
    Push(usize),
    Pop(usize),
}

#[derive(Clone, Debug)]
pub struct FunctionDecl {
    pub name: Symbol,
    pub schema: Schema,
    pub default: Option<Expr>,
    pub merge: Option<Expr>,
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

impl FunctionDecl {
    pub fn relation(name: Symbol, input: Vec<Symbol>) -> Self {
        Self {
            name,
            schema: Schema {
                input,
                output: Symbol::from("Unit"),
            },
            merge: None,
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
