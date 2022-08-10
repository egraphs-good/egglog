use std::fmt::Display;

pub use symbol_table::GlobalSymbol as Symbol;

macro_rules! lalrpop_error {
    ($($x:tt)*) => { Err(::lalrpop_util::ParseError::User { error: format!($($x)*)}) }
}

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

impl Id {
    pub(crate) fn fake() -> Self {
        Id(0xbadbeef)
    }
}

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
    Function(FunctionDecl),
    Define(Symbol, Expr),
    Rule(Rule),
    Rewrite(Rewrite),
    Action(Action),
    Run(usize),
    Extract(Expr),
    // TODO: this could just become an empty query
    Check(Fact),
    ClearRules,
    Query(Vec<Fact>),
}

#[derive(Clone, Debug)]
pub struct FunctionDecl {
    pub name: Symbol,
    pub schema: Schema,
    pub default: Option<Expr>,
    pub merge: Option<Expr>,
}

#[derive(Clone, Debug)]
pub struct Variant {
    pub name: Symbol,
    pub types: Vec<Type>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Error,
    Unit,
    Sort(Symbol),
    NumType(NumType),
    String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NumType {
    // F64,
    I64,
    Rational,
}

impl Display for NumType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NumType::I64 => write!(f, "i64"),
            NumType::Rational => write!(f, "rational"),
        }
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Sort(s) => Display::fmt(s, f),
            Type::NumType(t) => Display::fmt(t, f),
            Type::String => write!(f, "String"),
            Type::Unit => write!(f, "Unit"),
            Type::Error => write!(f, "Error"),
        }
    }
}

impl Type {
    pub fn is_sort(&self) -> bool {
        matches!(self, Self::Sort(..))
    }
}

#[derive(Clone, Debug)]
pub struct Schema {
    pub input: Vec<Type>,
    pub output: Type,
}

impl FunctionDecl {
    pub fn relation(name: Symbol, input: Vec<Type>) -> Self {
        Self {
            name,
            schema: Schema {
                input,
                output: Type::Unit,
            },
            merge: None,
            default: None,
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
    Define(Symbol, Expr),
    Set(Symbol, Vec<Expr>, Expr),
    Union(Expr, Expr),
    Panic(String),
    Expr(Expr),
    // If(Expr, Action, Action),
}

impl Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::Define(lhs, rhs) => write!(f, "(define {} {})", lhs, rhs),
            Action::Set(lhs, args, rhs) => {
                write!(f, "(set ({} {}) {})", lhs, ListDisplay(args, ""), rhs)
            }
            Action::Union(lhs, rhs) => write!(f, "(union {} {})", lhs, rhs),
            Action::Panic(msg) => write!(f, "(panic {:?})", msg),
            Action::Expr(e) => Display::fmt(e, f),
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
}
