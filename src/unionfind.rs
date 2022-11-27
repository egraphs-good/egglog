//! Baseline union-find implementation without sizes or ranks, using path
//! halving for compression.
//!
//! This implementation uses interior mutability for `find`.
use crate::{Id, Value};

use std::cell::Cell;
use std::fmt::Debug;

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde-1", derive(serde::Serialize, serde::Deserialize))]
pub struct UnionFind {
    parents: Vec<Cell<Id>>,
    n_unions: usize,
}

impl UnionFind {
    /// The number of unions that have been performed over the lifetime of this
    /// data-structure.
    pub fn n_unions(&self) -> usize {
        self.n_unions
    }

    /// Create a fresh [`Id`].
    pub fn make_set(&mut self) -> Id {
        let res = Id::from(self.parents.len());
        self.parents.push(Cell::new(res));
        res
    }

    /// Canonicalize a [`Value`].
    ///
    /// This method assumes that the given value belongs to an "eq-able" sort.
    /// Its behavior is unspecified on other values.
    pub fn find_value(&self, v: Value) -> Value {
        // NB: this assumes you have an eq-able sort.
        let bits = usize::from(self.find(Id::from(v.bits as usize))) as u64;
        Value { bits, ..v }
    }

    /// Look up the canonical representative for the given [`Id`].
    pub fn find(&self, id: Id) -> Id {
        let mut cur = self.parent(id);
        loop {
            let next = self.parent(cur.get());
            if cur.get() == next.get() {
                return cur.get();
            }
            // Path halving
            let grand = self.parent(next.get());
            cur.set(grand.get());
            cur = grand;
        }
    }

    /// Merge the equivalence classes associated with the two values.
    ///
    /// This method assumes that the given values belong to the same, "eq-able",
    /// sort.  Its behavior is unspecified on other values.
    pub fn union_values(&mut self, val1: Value, val2: Value) -> Value {
        debug_assert_eq!(val1.tag, val2.tag);
        let id1 = Id::from(val1.bits as usize);
        let id2 = Id::from(val2.bits as usize);
        let res = self.union(id1, id2);
        Value {
            bits: usize::from(res) as u64,
            tag: val1.tag,
        }
    }

    pub fn union(&mut self, id1: Id, id2: Id) -> Id {
        let id1 = self.find(id1);
        let id2 = self.find(id2);
        if id1 != id2 {
            self.parent(id2).set(id1);
            self.n_unions += 1;
        }
        id1
    }

    fn parent(&self, id: Id) -> &Cell<Id> {
        &self.parents[usize::from(id)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(us: impl IntoIterator<Item = usize>) -> Vec<Cell<Id>> {
        us.into_iter().map(|u| Cell::new(u.into())).collect()
    }

    #[test]
    fn union_find() {
        let n = 10;
        let id = Id::from;

        let mut uf = UnionFind::default();
        for _ in 0..n {
            uf.make_set();
        }

        // test the initial condition of everyone in their own set
        assert_eq!(uf.parents, ids(0..n));

        // build up one set
        uf.union(id(0), id(1));
        uf.union(id(0), id(2));
        uf.union(id(0), id(3));

        // build up another set
        uf.union(id(6), id(7));
        uf.union(id(6), id(8));
        uf.union(id(6), id(9));

        // this should compress all paths
        for i in 0..n {
            uf.find(id(i));
        }

        // indexes:         0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        let expected = vec![0, 0, 0, 0, 4, 5, 6, 6, 6, 6];
        assert_eq!(uf.parents, ids(expected));
    }
}
