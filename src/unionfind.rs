//! Baseline union-find implementation without sizes or ranks, using path
//! halving for compression.
//!
//! This implementation uses interior mutability for `find`.
use crate::util::HashMap;
use crate::{Id, Symbol, Value};

use std::cell::Cell;
use std::fmt::Debug;
use std::mem;

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde-1", derive(serde::Serialize, serde::Deserialize))]
pub struct UnionFind {
    parents: Vec<Cell<Id>>,
    n_unions: usize,
    recent_ids: HashMap<Symbol, Vec<Id>>,
    staged_ids: HashMap<Symbol, Vec<Id>>,
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

    pub fn num_ids(&self) -> usize {
        self.parents.len()
    }

    /// The number of ids that recently stopped being canonical.
    pub fn new_ids(&self, sort_filter: impl Fn(Symbol) -> bool) -> usize {
        self.recent_ids
            .iter()
            .filter_map(|(sort, ids)| {
                if sort_filter(*sort) {
                    Some(ids.len())
                } else {
                    None
                }
            })
            .sum()
    }

    /// Clear any ids currently marked as dirty and then move any ids marked
    /// non-canonical since the last call to this method (or the
    /// data-structure's creation) into the dirty set.
    pub fn clear_recent_ids(&mut self) {
        mem::swap(&mut self.recent_ids, &mut self.staged_ids);
        self.staged_ids.values_mut().for_each(Vec::clear);
    }

    /// Iterate over the ids of the given sort marked as "dirty", i.e. any
    /// [`Id`]s that ceased to be canonical between the last call to
    /// [`clear_recent_ids`] and the call prior to that.
    ///
    /// [`clear_recent_ids`]: UnionFind::clear_recent_ids
    pub fn dirty_ids(&self, sort: Symbol) -> impl Iterator<Item = Id> + '_ {
        let ids = self
            .recent_ids
            .get(&sort)
            .map(|ids| ids.as_slice())
            .unwrap_or(&[]);
        ids.iter().copied()
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
    /// sort. Its behavior is unspecified on other values.
    pub fn union_values(&mut self, _val1: Value, _val2: Value, _sort: Symbol) -> Value {
        panic!("We should never call union_values due to term encoding");
    }

    /// Like [`union_values`], but operating on raw [`Id`]s.
    ///
    /// [`union_values`]: UnionFind::union_values
    pub fn union(&mut self, id1: Id, id2: Id, sort: Symbol) -> Id {
        panic!("We should never call union due to term encoding");
    }

    /// Merge the underlying equivalence classes for the two ids.
    ///
    /// This method does not update any metadata related to timestamps or sorts;
    /// that metadata will eventually be required for the correctness of
    /// rebuilding. This method should only be used for "out-of-band" use-cases,
    /// such as typechecking.
    pub fn union_raw(&mut self, id1: Id, id2: Id) -> Id {
        self.do_union(id1, id2).0
    }

    fn do_union(&mut self, id1: Id, id2: Id) -> (Id, Option<Id>) {
        let id1 = self.find(id1);
        let id2 = self.find(id2);
        if id1 != id2 {
            self.parent(id2).set(id1);
            self.n_unions += 1;
            (id1, Some(id2))
        } else {
            (id1, None)
        }
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
        uf.union_raw(id(0), id(1));
        uf.union_raw(id(0), id(2));
        uf.union_raw(id(0), id(3));

        // build up another set
        uf.union_raw(id(6), id(7));
        uf.union_raw(id(6), id(8));
        uf.union_raw(id(6), id(9));

        // this should compress all paths
        for i in 0..n {
            uf.find(id(i));
        }

        // indexes:         0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        let expected = vec![0, 0, 0, 0, 4, 5, 6, 6, 6, 6];
        assert_eq!(uf.parents, ids(expected));
    }
}
