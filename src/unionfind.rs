use crate::{util::IndexMap, Id, Value};

use std::fmt::Debug;
use std::hash::Hash;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-1", derive(serde::Serialize, serde::Deserialize))]
pub struct UnionFind<V = ()> {
    parents: Vec<(Id, V)>,
    n_unions: usize,
}

impl<V> Default for UnionFind<V> {
    fn default() -> Self {
        Self {
            parents: Default::default(),
            n_unions: Default::default(),
        }
    }
}

impl<V> UnionFind<V> {
    pub fn n_unions(&self) -> usize {
        self.n_unions
    }

    pub fn make_set_with(&mut self, value: V) -> Id {
        let id = Id::from(self.parents.len());
        self.parents.push((id, value));
        id
    }
}

impl UnionFind<()> {
    pub fn make_set(&mut self) -> Id {
        self.make_set_with(())
    }

    pub fn find_mut_value(&mut self, value: Value) -> Value {
        self.find_mut(value.into()).into()
    }

    pub fn union_values(&mut self, value1: Value, value2: Value) -> Value {
        self.union(value1.into(), value2.into()).into()
    }
}

impl<V: UnifyValue> UnionFindLike<Id, V> for UnionFind<V> {
    fn len(&self) -> usize {
        self.parents.len()
    }

    fn insert_new(&mut self, _: Id, _: V) -> usize {
        panic!("should never insert_new")
    }

    fn get_index(&self, key: Id) -> Option<usize> {
        let i = usize::from(key);
        debug_assert!(i < self.len());
        Some(i)
    }

    fn key(&self, index: usize) -> Id {
        debug_assert!(index < self.len());
        index.into()
    }

    fn get_value_index(&self, i: usize) -> &V {
        &self.parents[i].1
    }

    fn set_value_index(&mut self, i: usize, value: V) {
        self.parents[i].1 = value
    }

    fn get_parent_index(&self, index: usize) -> usize {
        self.parents[index].0.into()
    }

    fn set_parent_index(&mut self, index: usize, new_parent: usize) {
        self.parents[index].0 = new_parent.into();
    }

    fn did_union(&mut self, _: usize) {
        self.n_unions += 1;
    }
}

pub trait UnifyKey: Hash + Eq + Clone + Debug {}
impl<K: Hash + Eq + Clone + Debug> UnifyKey for K {}

pub trait UnifyValue: Sized {
    type Error;
    fn merge(a: &Self, b: &Self) -> Result<Self, Self::Error>;
}

impl UnifyValue for () {
    type Error = std::convert::Infallible;
    fn merge((): &Self, (): &Self) -> Result<Self, Self::Error> {
        Ok(())
    }
}

pub(crate) trait UnionFindLike<K: UnifyKey, V: UnifyValue> {
    fn len(&self) -> usize;
    fn insert_new(&mut self, key: K, value: V) -> usize;
    fn key(&self, index: usize) -> K;
    fn get_index(&self, key: K) -> Option<usize>;
    fn get_value_index(&self, index: usize) -> &V;
    fn set_value_index(&mut self, index: usize, value: V);
    fn get_parent_index(&self, index: usize) -> usize;
    fn set_parent_index(&mut self, index: usize, new_parent: usize);

    fn index(&self, key: K) -> usize {
        self.get_index(key.clone())
            .unwrap_or_else(|| panic!("Couldn't find key {key:?}"))
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn union(&mut self, query1: K, query2: K) -> K
    where
        V: UnifyValue<Error = std::convert::Infallible>,
    {
        self.try_union(query1, query2).unwrap()
    }

    fn insert(&mut self, key: K, value: V) -> &V
    where
        V: UnifyValue<Error = std::convert::Infallible>,
    {
        self.try_insert(key, value).unwrap()
    }

    fn get_value(&self, key: K) -> &V {
        let root = self.find_index(self.index(key));
        self.get_value_index(root)
    }

    fn find(&self, key: K) -> K {
        let index = self.find_index(self.index(key));
        self.key(index)
    }

    fn find_mut(&mut self, key: K) -> K {
        let index = self.index(key);
        let index = self.find_index_mut(index);
        self.key(index)
    }

    fn find_index(&self, mut current: usize) -> usize {
        while current != self.get_parent_index(current) {
            current = self.get_parent_index(current)
        }
        current
    }

    fn find_index_mut(&mut self, mut current: usize) -> usize {
        while current != self.get_parent_index(current) {
            let grandparent = self.get_parent_index(self.get_parent_index(current));
            self.set_parent_index(current, grandparent);
            current = grandparent;
        }
        current
    }

    /// Given two leader ids, unions the two eclasses making root1 the leader.
    fn try_union(&mut self, query1: K, query2: K) -> Result<K, V::Error> {
        let index1 = self.index(query1);
        let index2 = self.index(query2);
        let root1 = self.find_index_mut(index1);
        let root2 = self.find_index_mut(index2);
        let root = if root1 != root2 {
            self.union_roots(root1, root2)?
        } else {
            root1
        };
        debug_assert_eq!(self.find_index(root), root);
        Ok(self.key(root))
    }

    fn try_insert(&mut self, key: K, value: V) -> Result<&V, V::Error> {
        let root = if let Some(index) = self.get_index(key.clone()) {
            let root = self.find_index_mut(index);
            let old_value = self.get_value_index(root);
            let value = V::merge(old_value, &value)?;
            self.set_value_index(root, value);
            root
        } else {
            self.insert_new(key, value)
        };
        Ok(self.get_value_index(root))
    }

    fn union_roots(&mut self, a: usize, b: usize) -> Result<usize, V::Error> {
        debug_assert_ne!(a, b);
        debug_assert_eq!(a, self.get_parent_index(a));
        debug_assert_eq!(b, self.get_parent_index(b));
        let v = V::merge(self.get_value_index(a), self.get_value_index(b))?;
        self.set_value_index(a, v);
        self.set_parent_index(b, a);
        self.did_union(a);
        Ok(a)
    }

    fn did_union(&mut self, _index: usize) {}

    fn sets(&self) -> Vec<Vec<K>>
    where
        K: Clone,
    {
        let mut sets = vec![vec![]; self.len()];
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.len() {
            let k = self.key(i);
            let ii = self.find_index(i);
            sets[ii].push(k.clone());
        }
        sets.retain(|set| !set.is_empty());
        sets
    }
}

#[derive(Debug, Clone)]
pub struct SparseUnionFind<K, V> {
    map: IndexMap<K, (usize, V)>,
}

impl<K, V> Default for SparseUnionFind<K, V> {
    fn default() -> Self {
        Self {
            map: Default::default(),
        }
    }
}

impl<K: UnifyKey, V: UnifyValue> UnionFindLike<K, V> for SparseUnionFind<K, V> {
    fn len(&self) -> usize {
        self.map.len()
    }

    fn insert_new(&mut self, key: K, value: V) -> usize {
        let len = self.map.len();
        let (i, old) = self.map.insert_full(key, (len, value));
        assert!(old.is_none());
        i
    }

    fn get_index(&self, key: K) -> Option<usize> {
        self.map.get_index_of(&key)
    }

    fn key(&self, index: usize) -> K {
        self.map.get_index(index).unwrap().0.clone()
    }

    fn get_value_index(&self, index: usize) -> &V {
        &self.map.get_index(index).unwrap().1 .1
    }

    fn set_value_index(&mut self, index: usize, value: V) {
        self.map.get_index_mut(index).unwrap().1 .1 = value;
    }

    fn get_parent_index(&self, index: usize) -> usize {
        self.map.get_index(index).unwrap().1 .0
    }

    fn set_parent_index(&mut self, index: usize, new_parent: usize) {
        self.map.get_index_mut(index).unwrap().1 .0 = new_parent;
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
        assert_eq!(
            uf.parents.iter().map(|(i, _)| *i).collect::<Vec<Id>>(),
            ids(0..n)
        );

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
        assert_eq!(
            uf.parents.iter().map(|(i, _)| *i).collect::<Vec<Id>>(),
            ids(expected)
        );
    }
}
