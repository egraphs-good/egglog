use crate::{util::IndexMap, Id, Value};

use std::fmt::Debug;
use std::hash::Hash;

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde-1", derive(serde::Serialize, serde::Deserialize))]
pub struct UnionFind {
    parents: Vec<Id>,
    n_unions: usize,
}

impl UnionFind {
    pub fn make_set(&mut self) -> Id {
        let id = Id::from(self.parents.len());
        self.parents.push(id);
        id
    }

    pub fn size(&self) -> usize {
        self.parents.len()
    }

    pub fn n_unions(&self) -> usize {
        self.n_unions
    }

    fn parent(&self, query: Id) -> Id {
        self.parents[usize::from(query)]
    }

    fn parent_mut(&mut self, query: Id) -> &mut Id {
        &mut self.parents[usize::from(query)]
    }

    pub fn find(&self, mut current: Id) -> Id {
        while current != self.parent(current) {
            current = self.parent(current)
        }
        current
    }

    pub fn find_mut(&mut self, mut current: Id) -> Id {
        while current != self.parent(current) {
            let grandparent = self.parent(self.parent(current));
            *self.parent_mut(current) = grandparent;
            current = grandparent;
        }
        current
    }

    pub fn find_mut_value(&mut self, value: Value) -> Value {
        self.find_mut(value.into()).into()
    }

    /// Given two leader ids, unions the two eclasses making root1 the leader.
    pub fn union(&mut self, mut root1: Id, mut root2: Id) -> Id {
        root1 = self.find_mut(root1);
        root2 = self.find_mut(root2);
        if root1 != root2 {
            *self.parent_mut(root2) = root1;
            self.n_unions += 1;
        }
        root1
    }

    pub fn union_values(&mut self, value1: Value, value2: Value) -> Value {
        self.union(value1.into(), value2.into()).into()
    }
}

pub trait UnionFindLike<K> {
    fn len(&self) -> usize;
    fn insert(&mut self, key: K);
    fn index(&self, key: K) -> usize;
    fn key(&self, index: usize) -> &K;
    fn get_parent_index(&self, index: usize) -> usize;
    fn set_parent_index(&mut self, index: usize, new_parent: usize);

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn find_index(&self, query: K) -> usize {
        let mut current = self.index(query);
        while current != self.get_parent_index(current) {
            current = self.get_parent_index(current)
        }
        current
    }

    fn find_index_mut(&mut self, query: K) -> usize {
        let mut current = self.index(query);
        while current != self.get_parent_index(current) {
            let grandparent = self.get_parent_index(self.get_parent_index(current));
            self.set_parent_index(current, grandparent);
            current = grandparent;
        }
        current
    }

    /// Given two leader ids, unions the two eclasses making root1 the leader.
    fn union(&mut self, query1: K, query2: K) -> &K {
        let root1 = self.find_index_mut(query1);
        let root2 = self.find_index_mut(query2);
        let root = if root1 != root2 {
            self.union_roots(root1, root2)
        } else {
            root1
        };
        self.key(root)
    }

    fn union_roots(&mut self, a: usize, b: usize) -> usize {
        self.set_parent_index(b, a);
        a
    }

    fn sets(&self) -> Vec<Vec<K>>
    where
        K: Clone,
    {
        let mut sets = vec![vec![]; self.len()];
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.len() {
            let k = self.key(i);
            let ii = self.find_index(k.clone());
            sets[ii].push(k.clone());
        }
        sets.retain(|set| !set.is_empty());
        sets
    }
}

#[derive(Debug, Clone)]
pub struct SparseUnionFind<K> {
    map: IndexMap<K, usize>,
}

impl<K> Default for SparseUnionFind<K> {
    fn default() -> Self {
        Self {
            map: Default::default(),
        }
    }
}

impl<K: Hash + Eq> UnionFindLike<K> for SparseUnionFind<K> {
    fn len(&self) -> usize {
        self.map.len()
    }

    fn insert(&mut self, key: K) {
        let len = self.map.len();
        self.map.entry(key).or_insert(len);
    }

    fn index(&self, key: K) -> usize {
        self.map.get_index_of(&key).unwrap()
    }

    fn key(&self, index: usize) -> &K {
        self.map.get_index(index).unwrap().0
    }

    fn get_parent_index(&self, index: usize) -> usize {
        *self.map.get_index(index).unwrap().1
    }

    fn set_parent_index(&mut self, index: usize, new_parent: usize) {
        *self.map.get_index_mut(index).unwrap().1 = new_parent;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(us: impl IntoIterator<Item = usize>) -> Vec<Id> {
        us.into_iter().map(|u| u.into()).collect()
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
            uf.find_mut(id(i));
        }

        // indexes:         0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        let expected = vec![0, 0, 0, 0, 4, 5, 6, 6, 6, 6];
        assert_eq!(uf.parents, ids(expected));
    }
}
