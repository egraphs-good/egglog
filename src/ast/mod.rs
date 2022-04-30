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
    Fact(Fact),
    Run(usize),
    Extract(Expr),
    // TODO: this could just become an empty query
    Check(Fact),
}

#[derive(Clone, Debug)]
pub struct FunctionDecl {
    pub name: Symbol,
    pub schema: Schema,
}

#[derive(Clone, Debug)]
pub struct Variant {
    pub name: Symbol,
    pub types: Vec<InputType>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InputType {
    Sort(Symbol),
    NumType(NumType),
    String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NumType {
    // F64,
    I64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OutputType {
    Unit,
    Type(InputType),
    Max(NumType),
    Min(NumType),
}

impl OutputType {
    pub fn is_sort(&self) -> bool {
        if let OutputType::Type(ty) = self {
            ty.is_sort()
        } else {
            false
        }
    }
}

impl Display for NumType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NumType::I64 => write!(f, "i64"),
        }
    }
}

impl Display for OutputType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputType::Type(t) => Display::fmt(t, f),
            OutputType::Unit => write!(f, "Unit"),
            OutputType::Max(t) => write!(f, "(max {t})"),
            OutputType::Min(t) => write!(f, "(min {t})"),
        }
    }
}

impl Display for InputType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InputType::Sort(s) => Display::fmt(s, f),
            InputType::NumType(t) => Display::fmt(t, f),
            InputType::String => write!(f, "String"),
        }
    }
}

impl InputType {
    pub fn is_sort(&self) -> bool {
        matches!(self, Self::Sort(..))
    }
}

#[derive(Clone, Debug)]
pub struct Schema {
    pub input: Vec<InputType>,
    pub output: OutputType,
}

impl FunctionDecl {
    pub fn relation(name: Symbol, input: Vec<InputType>) -> Self {
        Self {
            name,
            schema: Schema {
                input,
                output: OutputType::Unit,
            },
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
pub struct Rule {
    // pub query: Query,
    // pub actions: Vec<Action>,
    pub head: Vec<Fact>,
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
