mod expr;
mod symbol;

use std::fmt::Display;

use lalrpop_util::lalrpop_mod;
lalrpop_mod!(
    #[allow(clippy::all)]
    pub parse,
    "/ast/parse.rs"
);

use crate::*;

pub use expr::*;
pub use symbol::*;

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
    Function {
        name: Symbol,
        schema: Schema,
        merge: Option<MergeFn>,
    },
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
pub struct Variant {
    pub name: Symbol,
    pub types: Vec<Type>,
}

#[derive(Clone, Debug)]
pub struct MergeFn {
    pub vars: (Symbol, Symbol),
    pub expr: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Sort(Symbol),
    Bool,
    Unit,
    Int,
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Sort(s) => Display::fmt(s, f),
            Type::Bool => write!(f, "Bool"),
            Type::Unit => write!(f, "Unit"),
            Type::Int => write!(f, "Int"),
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

impl Schema {
    pub fn relation(input: Vec<Type>) -> Self {
        Schema {
            input,
            output: Type::Unit,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Fact {
    /// Must be at least two things in an eq fact
    Eq(Vec<Expr>),
    Fact(Expr),
}

#[derive(Clone, Debug)]
pub struct Rule {
    // pub query: Query,
    // pub actions: Vec<Action>,
    pub head: Vec<Fact>,
    pub body: Vec<Fact>,
}

#[derive(Clone, Debug)]
pub struct Rewrite {
    pub lhs: Expr,
    pub rhs: Expr,
}
