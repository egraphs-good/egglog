use std::fmt::Display;
use std::hash::Hash;

use ordered_float::OrderedFloat;

use crate::span::Span;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub enum Literal {
    Int(i64),
    Float(OrderedFloat<f64>),
    String(String),
    Bool(bool),
    Unit,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GenericExpr<Head, Leaf> {
    Var(Span, Leaf),
    Call(Span, Head, Vec<GenericExpr<Head, Leaf>>),
    Lit(Span, Literal),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GenericFact<Head, Leaf> {
    Eq(Span, GenericExpr<Head, Leaf>, GenericExpr<Head, Leaf>),
    Fact(GenericExpr<Head, Leaf>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GenericActions<Head: Clone + Display, Leaf: Clone + PartialEq + Eq + Display + Hash>(
    pub Vec<GenericAction<Head, Leaf>>,
);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GenericAction<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    Let(Span, Leaf, GenericExpr<Head, Leaf>),
    Set(
        Span,
        Head,
        Vec<GenericExpr<Head, Leaf>>,
        GenericExpr<Head, Leaf>,
    ),
    Change(Span, Change, Head, Vec<GenericExpr<Head, Leaf>>),
    Union(Span, GenericExpr<Head, Leaf>, GenericExpr<Head, Leaf>),
    Panic(Span, String),
    Expr(Span, GenericExpr<Head, Leaf>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GenericRule<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub span: Span,
    pub head: GenericActions<Head, Leaf>,
    pub body: Vec<GenericFact<Head, Leaf>>,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub enum Change {
    Delete,
    Subsume,
}
