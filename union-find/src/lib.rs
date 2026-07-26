//! This crate contains two basic union-find implementations:
//!
//! * [`UnionFind`], a basic single-threaded union-find data-structure.
//! * [`concurrent::UnionFind`], a concurrent union-find data-structure.
//!
//! Both structures are fairly rudimentary and are customized to be used in an
//! egraph-related setting. In particular, they do "union by min id", which is a
//! strategy that _does not_ guarantee the same asymptotic complexity as the
//! main techniques in the literature (e.g. union by rank). Union by min is a
//! heuristic introduced to reduce the number of ids perturbed during congruence
//! closure. There's likely more to do in this area but for now it seems to work
//! well enough. It doesn't hurt that it's also simpler to implement.
use egglog_numeric_id as numeric_id;
use numeric_id::{AtomicNumericId, NumericId};
use std::{cmp, marker::PhantomData, mem, slice};

pub mod concurrent;

#[cfg(test)]
mod tests;

/// A basic implementation of a union-find datastructure.
#[derive(Clone)]
pub struct UnionFind<Value> {
    parents: Vec<Value>,
}

/// A concurrent, fixed-capacity view borrowed from a serial [`UnionFind`].
///
/// Creating this view requires exclusive access to the serial union-find. The
/// view may be shared with scoped worker threads, and the serial representation
/// becomes accessible again only after all of those borrows have ended.
///
/// Operations on this fixed view use relaxed atomics. Parent entries are the
/// only shared state: a successful union changes a root from itself to a
/// strictly smaller ID, and path splitting changes a parent to a strictly
/// smaller ancestor. Consequently links never increase, cannot exhibit ABA,
/// and cannot form a cycle. A stale load is still a same-component ancestor,
/// while every mutation uses compare-exchange to verify the observed link.
/// The scope join before this view is dropped provides the happens-before edge
/// needed before ordinary reads resume; parent links do not publish any other
/// payload.
pub struct AtomicUnionFindAccess<'a, Value: AtomicNumericId> {
    parents: &'a [Value::Atomic],
    marker: PhantomData<Value>,
}

impl<Value: AtomicNumericId> AtomicUnionFindAccess<'_, Value>
where
    Value::Atomic: concurrent::atomic_int::AtomicInt<Underlying = Value::Rep>,
{
    /// Merge two equivalence classes.
    ///
    /// Panics if either ID exceeds the fixed capacity selected when this view
    /// was created.
    #[inline]
    pub fn union(&self, l: Value, r: Value) -> (Value, Value) {
        let (parent, child) = concurrent::uf::merge_scoped_impl(self.parents, l.rep(), r.rep());
        (Value::new(parent), Value::new(child))
    }
}

impl<V> Default for UnionFind<V> {
    fn default() -> Self {
        Self {
            parents: Vec::new(),
        }
    }
}

impl<Value: NumericId> UnionFind<Value> {
    /// Reset the union-find data-structure to the point where all Ids are their
    /// own parents.
    pub fn reset(&mut self) {
        for (i, v) in self.parents.iter_mut().enumerate() {
            *v = Value::from_usize(i);
        }
    }

    /// Reserve sufficient space for the given value `v`.
    pub fn reserve(&mut self, v: Value) {
        if v.index() >= self.parents.len() {
            for i in self.parents.len()..=v.index() {
                self.parents.push(Value::from_usize(i));
            }
        }
    }

    /// Merge two equivalence classes.
    pub fn union(&mut self, a: Value, b: Value) -> (Value /* parent */, Value /* child */) {
        self.reserve(a);
        self.reserve(b);
        let a = self.find(a);
        let b = self.find(b);
        if a != b {
            let parent = cmp::min(a, b);
            let child = cmp::max(a, b);
            self.parents[child.index()] = parent;
            (parent, child)
        } else {
            (a, a)
        }
    }

    /// Find the representative of an equivalence class.
    pub fn find(&mut self, id: Value) -> Value {
        self.reserve(id);
        let mut cur = id;
        loop {
            let parent = self.parents[cur.index()];
            if cur == parent {
                break;
            }
            let grand = self.parents[parent.index()];
            self.parents[cur.index()] = grand;
            cur = grand;
        }
        cur
    }

    /// Find the representative of an equivalence class without using path compression.
    ///
    /// The primary advantage of this method is that it allows the ability to answer `find` queries
    /// without holding a mutable reference to the union-find.
    pub fn find_naive(&self, id: Value) -> Value {
        if self.parents.len() <= id.index() {
            return id;
        }
        let mut cur = id;
        loop {
            let parent = self.parents[cur.index()];
            if cur == parent {
                break;
            }
            cur = parent;
        }
        cur
    }
}

impl<Value: AtomicNumericId> UnionFind<Value>
where
    Value::Atomic: concurrent::atomic_int::AtomicInt<Underlying = Value::Rep>,
{
    /// Temporarily borrow this union-find as a fixed-capacity atomic view.
    ///
    /// The backing vector is grown before the atomic view is created and
    /// cannot resize while `f` runs. Atomic and ordinary accesses never
    /// overlap: the mutable receiver excludes ordinary readers until `f` and
    /// every scoped worker borrowing its view have completed. No allocation,
    /// resize, or per-operation access guard occurs while the view is active.
    /// `max_id` is inclusive, and `f` must not request a union containing a
    /// larger ID.
    ///
    /// # Panics
    ///
    /// Panics if the ID and its atomic integer do not have identical size and
    /// alignment on the target platform.
    pub fn with_atomic_access<R>(
        &mut self,
        max_id: Value,
        f: impl FnOnce(&AtomicUnionFindAccess<'_, Value>) -> R,
    ) -> R {
        self.reserve(max_id);
        assert_eq!(
            mem::size_of::<Value>(),
            mem::size_of::<Value::Atomic>(),
            "numeric id and atomic representation must have equal size"
        );
        assert_eq!(
            mem::align_of::<Value>(),
            mem::align_of::<Value::Atomic>(),
            "numeric id and atomic representation must have equal alignment"
        );

        // SAFETY:
        // - `AtomicNumericId` guarantees a transparent integer
        //   representation compatible with the corresponding atomic; the
        //   assertions above defensively verify its size and alignment.
        // - Integer bit patterns are valid atomic integer bit patterns.
        // - `&mut self` excludes every ordinary access to `parents` for the
        //   lifetime of this slice.
        // - `f` cannot let the borrowed view escape, so every atomic access
        //   ends before ordinary access resumes.
        let parents = unsafe {
            slice::from_raw_parts(
                self.parents.as_mut_ptr().cast::<Value::Atomic>(),
                self.parents.len(),
            )
        };
        f(&AtomicUnionFindAccess {
            parents,
            marker: PhantomData,
        })
    }
}
