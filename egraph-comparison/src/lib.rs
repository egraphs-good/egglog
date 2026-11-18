//! Compare complete serialized databases by conservative partition refinement.
mod hopcroft;
mod ids;
mod model;
mod refine;
mod signatures;
pub use ids::{FormatVersion, RowId};
pub use model::{Class, Database, Error, Function, FunctionKind, Row};
pub use refine::{Comparison, compare};
pub(crate) type HashMap<K, V> =
    hashbrown::HashMap<K, V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;
