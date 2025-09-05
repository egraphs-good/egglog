//! A concurrent implementation of a union-find datastructure.
//!
//! See the `uf` module for more details on the implementation.

use atomic_int::AtomicInt;
use crate::numeric_id::NumericId;

pub(crate) mod atomic_int;
pub(crate) mod buffer;
pub(crate) mod uf;

#[cfg(test)]
mod tests;

/// A thread-safe implementation of a union-find datastructure.
///
/// This implementation supports concurrent finds and merges, with path
/// compression. Importantly, this implementation supports dynamically resizing
/// the underlying array when new ids appear. This allows callers to generate
/// Ids more flexibly, though huge amounts of resizing will cause contention as
/// callers wait for resizes to complete.
///
/// The `Clone` implementation for this type is shallow: copies of the
/// data-structure see one another's updates.
pub struct UnionFind<T: NumericId> {
    inner: uf::ConcurrentUnionFind<T::Atomic>,
}

impl<T: NumericId> Clone for UnionFind<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: NumericId> Default for UnionFind<T>
where
    T::Atomic: AtomicInt,
{
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

impl<T: NumericId> UnionFind<T>
where
    T::Atomic: AtomicInt<Underlying = T::Rep>,
{
    /// Reset the union-find datastructure, setting each element to point to itself.
    ///
    /// This method blocks until all threads have finished their current operations.
    pub fn reset(&self) {
        self.inner.reset();
    }

    /// Create a deep copy of the union-find datastructure: subsequent unions on
    /// the returned copy will not affect the original.
    pub fn deep_copy(&self) -> Self {
        Self {
            inner: self.inner.deep_copy(),
        }
    }

    /// Initialize a union-find with `capacity` elements pointing to themselves.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: uf::ConcurrentUnionFind::with_capacity(capacity),
        }
    }
    /// Get the canonical value associated with `elt`.
    ///
    /// Note that it may be the case that `find(x) != find(y)` even if `x` and
    /// `y` belong to the same equivalence class, as the canonical value for the
    /// class may have changed between the `find(x)` and `find(y)` calls. To get
    /// a "ground truth" view on this, use the `same_set` method.
    ///
    /// We expose `find` because it _does_ work if you know that there are not
    /// any concurrent `merge` operations on the data-structure: this is common
    /// enough in egglog. It also makes some tests easier to write.
    pub fn find(&self, elt: T) -> T {
        T::new(self.inner.find(elt.rep()))
    }

    /// Check if `l` and `r` belong to the same equivalence class.
    pub fn same_set(&self, l: T, r: T) -> bool {
        self.inner.same_set(l.rep(), r.rep())
    }

    /// Merge the equivalence classes of `l` and `r`, returning the new parent
    /// and new child classes. If `l` and `r` are already in the same class,
    /// then their canonical representative is returned twice.
    pub fn union(&self, l: T, r: T) -> (T, T) {
        let (parent, child) = self.inner.merge(l.rep(), r.rep());
        (T::new(parent), T::new(child))
    }
}
