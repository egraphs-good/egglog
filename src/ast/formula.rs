use crate::*;
use std::fmt::Display;

#[derive(Debug, Clone)]
pub enum Goal {
    Query(Query),
    ForAll(Vec<IdentSort>, Box<Goal>),
    Implies(Prog, Box<Goal>),
}

#[derive(Debug, Clone)]
pub enum Query {
    Atom(Fact),
    And(Vec<Query>),
}

#[derive(Debug, Clone)]
pub enum Prog {
    Atom(Action),
    And(Vec<Prog>),
    ForAll(Vec<IdentSort>, Box<Prog>),
    Implies(Query, Box<Prog>),
}

impl Display for Prog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Prog::Atom(a) => std::fmt::Display::fmt(a, f),
            Prog::And(ps) => write!(f, "(and {})", ListDisplay(ps, " ")),
            Prog::ForAll(xs, p) => write!(f, "(forall ({}) {})", ListDisplay(xs, " "), *p),
            Prog::Implies(gs, p) => write!(f, "(=> {} {})", gs, *p),
        }
    }
}

impl Display for Query {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Query::Atom(fact) => write!(f, "{}", fact),
            Query::And(ps) => write!(f, "(and {})", ListDisplay(ps, " ")),
        }
    }
}

impl Display for Goal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Goal::Query(pg) => write!(f, "{}", pg),
            Goal::ForAll(xs, p) => write!(f, "(forall ({}) {})", ListDisplay(xs, " "), *p),
            Goal::Implies(gs, p) => write!(f, "(=> {} {})", gs, *p),
        }
    }
}
