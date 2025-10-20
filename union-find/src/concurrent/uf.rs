//! The core union-find implementation.
//!
//! This implementation is inspired by "Concurrent Disjoint Set Union" by
//! Siddhartha Jayanti and Robert Tarjan, but it includes changes and extensions
//! to make it work with egglog's requirements. On the whole, it's a bit simpler
//! than the implementation in the paper. A few notes:
//!
//! * We don't bother with union-by-rank: this isn't in use in egg or egglog. We
//!   currently implement "union by min" which is a simple heuristic that folks
//!   _hypothesize_ does well in practice for egraph applications. Depending on
//!   how well this pans out, we may want to use something closer to the
//!   randomized link by rank from the paper.
//!
//! * As in the paper, we use splitting for path compression, using the two-try
//!   variant from the paper.
use std::{cmp, sync::Arc};

use super::{atomic_int::AtomicInt, buffer::Buffer};

pub(crate) struct ConcurrentUnionFind<T> {
    data: Arc<Buffer<T>>,
}

impl<T> Clone for ConcurrentUnionFind<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

impl<T: AtomicInt> Default for ConcurrentUnionFind<T> {
    fn default() -> Self {
        Self::with_capacity(32)
    }
}

impl<T: AtomicInt> ConcurrentUnionFind<T> {
    pub fn deep_copy(&self) -> Self {
        ConcurrentUnionFind {
            data: Arc::new(Buffer::clone(&*self.data)),
        }
    }
}

impl<T: AtomicInt> ConcurrentUnionFind<T> {
    /// Reset the union-find datastructure, setting each element to point to itself.
    ///
    /// This method blocks until all threads have finished their current operations.
    pub fn reset(&self) {
        self.data.re_init(T::from_usize);
    }

    /// Create a new union-find datastructure with the given capacity.
    pub fn with_capacity(capacity: usize) -> ConcurrentUnionFind<T> {
        let data = Arc::new(Buffer::new(capacity, T::from_usize));
        ConcurrentUnionFind { data }
    }

    /// Get the canonical value associated with `elt`.
    ///
    /// Note that it may be the case that `find(x) != find(y)` even if `x` and
    /// `y` belong to the same equivalence class, as the canonical value for the
    /// class may have changed between the `find(x)` and `find(y)` calls. To get
    /// a "ground truth" view on this, use the `same_set` method.
    pub fn find(&self, elt: T::Underlying) -> T::Underlying {
        self.data.with_access(
            T::as_usize(elt) + 1,
            |buf| Self::find_impl(buf, elt),
            T::from_usize,
        )
    }

    /// Check if `l` and `r` belong to the same equivalence class.
    pub fn same_set(&self, l: T::Underlying, r: T::Underlying) -> bool {
        let max_elt = cmp::max(l, r);
        self.data.with_access(
            T::as_usize(max_elt) + 1,
            |buf| {
                let mut l = Self::find_impl(buf, l);
                let mut r = Self::find_impl(buf, r);
                while l != r {
                    let next = buf[T::as_usize(l)].load();
                    if next == l {
                        return false;
                    }
                    l = Self::find_impl(buf, l);
                    r = Self::find_impl(buf, r);
                }
                true
            },
            T::from_usize,
        )
    }

    /// Merge the equivalence classes of `l` and `r`, returning the new parent
    /// and new child classes. If `l` and `r` are already in the same class,
    /// then their canonical representative is returned twice.
    pub fn merge(
        &self,
        l: T::Underlying,
        r: T::Underlying,
    ) -> (
        T::Underlying, /* parent */
        T::Underlying, /* child */
    ) {
        self.data.with_access(
            T::as_usize(cmp::max(l, r)) + 1,
            |buf| {
                let mut l = l;
                let mut r = r;
                l = Self::find_impl(buf, l);
                r = Self::find_impl(buf, r);
                if l != r {
                    // We do "union by min": common in egraphs due to the
                    // hypothesis that smaller ids will be
                    // better-represented in the egraph, and hence
                    // perturbing the maximum id is likely to result in less
                    // work for rebuilding.
                    let parent = cmp::min(l, r);
                    let child = cmp::max(l, r);
                    if buf[T::as_usize(child)].cas(child, parent).is_ok() {
                        return (parent, child);
                    }
                }
                (l, l)
            },
            T::from_usize,
        )
    }

    fn find_impl(buf: &[T], elt: T::Underlying) -> T::Underlying {
        macro_rules! load {
            ($x:expr) => {
                buf[T::as_usize($x)].load()
            };
        }
        let mut cur = elt;
        let mut next = load!(cur);
        let mut grand = load!(next);
        while next != grand {
            let _ = buf[T::as_usize(cur)].cas(next, grand);
            // This is what the paper calls "two-try" splitting.
            // next = load!(cur);
            // grand = load!(next);
            // let _ = buf[T::as_usize(cur)].cas(next, grand);
            cur = next;
            next = load!(cur);
            grand = load!(next);
        }
        next
    }
}
