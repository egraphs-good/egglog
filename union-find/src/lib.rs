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
use numeric_id::NumericId;
use std::cmp;

pub mod concurrent;

#[cfg(test)]
mod tests;

/// A basic implementation of a union-find datastructure.
#[derive(Clone)]
pub struct UnionFind<Value> {
    parents: Vec<Value>,
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
