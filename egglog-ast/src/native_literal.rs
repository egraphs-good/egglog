//! Type-erased literals supplied by a native engine.

use std::any::Any;
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Native literal behavior. Engines supply the concrete implementation;
/// parsers and portable protocols do not construct this payload themselves.
pub trait NativeLiteral: Any + Debug + Display + Send + Sync {
    /// Recover the engine-owned literal implementation.
    fn as_any(&self) -> &dyn Any;
    /// Compare two literals of the same concrete implementation.
    fn compare(&self, other: &dyn NativeLiteral) -> Ordering;
    /// Hash the owned scalar data and its sort.
    fn value_hash(&self) -> u64;
}

/// An owned scalar literal whose native payload is opaque to the AST crate.
#[derive(Clone, Debug)]
pub struct NativeLiteralValue(pub Arc<dyn NativeLiteral>);

impl PartialEq for NativeLiteralValue {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for NativeLiteralValue {}
impl PartialOrd for NativeLiteralValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for NativeLiteralValue {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .as_any()
            .type_id()
            .cmp(&other.0.as_any().type_id())
            .then_with(|| self.0.compare(other.0.as_ref()))
    }
}
impl Hash for NativeLiteralValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_any().type_id().hash(state);
        self.0.value_hash().hash(state);
    }
}
impl Display for NativeLiteralValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}
