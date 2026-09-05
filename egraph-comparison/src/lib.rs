//! Equality of serialized e-graphs modulo constructor bisimulation.
//!
//! Input IDs are local names, never semantic identities. Comparison refines the
//! disjoint union of both inputs, including cycles, until no block can split.

mod model;
mod refine;

pub use model::{Class, Database, Error, Function, FunctionKind, Row};
pub use refine::{Comparison, compare};
