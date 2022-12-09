use crate::*;

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
}
