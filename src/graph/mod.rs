pub(crate) mod from_egraph;
pub(crate) mod to_graphviz;

use crate::{ast::Expr, util::HashMap};

type EClassID = String;

// Exposed graph structure which can be used to print/visualize the state of the e-graph.
#[derive(Debug)]
pub(crate) struct Graph {
    // All of the primitive values which are outputs of functions
    pub prim_outputs: Vec<PrimOutput>,
    // All of the e-classes which are have non primitive types
    pub eclasses: HashMap<EClassID, Vec<FnCall>>,
}

// A primitive value which is output from a function.
#[derive(Debug)]
pub(crate) struct PrimOutput(pub FnCall, pub PrimValue);

#[derive(Debug)]
pub(crate) struct FnCall(pub Fn, pub Vec<Arg>);

/// An argument is either a primitive value or a reference to a eclass
#[derive(Debug)]
pub(crate) enum Arg {
    Prim(PrimValue),
    Eq(EClassID),
}

#[derive(Debug)]
pub(crate) struct Fn {
    pub name: String,
    // TODO: Add cost
}

/// A primitive value (str, float, int, etc)
#[derive(Debug)]
pub(crate) struct PrimValue(String);

pub(crate) fn from_expr(expr: &Expr) -> PrimValue {
    PrimValue(expr.to_string())
}

impl ToString for PrimValue {
    fn to_string(&self) -> String {
        self.0.clone()
    }
}
