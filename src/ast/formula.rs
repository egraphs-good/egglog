use crate::*;
use std::fmt::Display;

#[derive(Debug, Clone)]
pub enum Goal {
    // True?
    PrimGoal(PrimGoal),
    ForAll(Vec<IdentSort>, Box<Goal>),
    Implies(Prog, Box<Goal>),
    // And() can be dealt with. Forks the database.
    // It _might_ be possible to put Or in here, painfully
}

#[derive(Debug, Clone)]
pub enum PrimGoal {
    // True?
    Atom(Fact),
    And(Vec<PrimGoal>),
    // Or(Vec<PrimGoal>),
    // Exists(Vec<IdentSort>, Box<PrimGoal>),
}

#[derive(Debug, Clone)]
pub enum Prog {
    // True?
    Atom(Action),
    And(Vec<Prog>),
    ForAll(Vec<IdentSort>, Box<Prog>),
    Implies(PrimGoal, Box<Prog>),
    // Exists via skolemization
    // Exists via
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

impl Display for PrimGoal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrimGoal::Atom(fact) => write!(f, "{}", fact),
            PrimGoal::And(ps) => write!(f, "(and {})", ListDisplay(ps, " ")),
        }
    }
}

impl Display for Goal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Goal::PrimGoal(pg) => write!(f, "{}", pg),
            Goal::ForAll(xs, p) => write!(f, "(forall ({}) {})", ListDisplay(xs, " "), *p),
            Goal::Implies(gs, p) => write!(f, "(=> {} {})", gs, *p),
        }
    }
}
