//! Equality of serialized e-graphs modulo constructor bisimulation.
//!
//! Input IDs are local names, never semantic identities. Comparison refines the
//! disjoint union of both inputs, including cycles, until no block can split.

mod canonical;
mod certificate;
mod model;
mod refine;

pub use canonical::{CanonicalMode, canonicalize};
pub use certificate::{Certificate, Side, Term, certificate, verify};
pub use model::{Class, Database, Error, Function, FunctionKind, Row};
pub use refine::{Comparison, compare};

pub(crate) type HashMap<K, V> =
    hashbrown::HashMap<K, V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;
