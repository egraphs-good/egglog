//! A crate with utilities for working with numeric Ids.
use std::{
    fmt::{self, Debug},
    hash::Hash,
    marker::PhantomData,
    ops,
};

#[cfg(test)]
mod tests;

/// A trait describing "newtypes" that wrap an integer.
pub trait NumericId: Copy + Clone + PartialEq + Eq + PartialOrd + Ord + Hash + Send + Sync {
    type Rep;
    type Atomic;
    fn new(val: Self::Rep) -> Self;
    fn from_usize(index: usize) -> Self;
    fn index(self) -> usize;
    fn rep(self) -> Self::Rep;
    fn inc(self) -> Self {
        Self::from_usize(self.index() + 1)
    }
}

impl NumericId for usize {
    type Rep = usize;
    type Atomic = std::sync::atomic::AtomicUsize;
    fn new(val: usize) -> Self {
        val
    }
    fn from_usize(index: usize) -> Self {
        index
    }

    fn rep(self) -> usize {
        self
    }

    fn index(self) -> usize {
        self
    }
}

/// A mapping from a [`NumericId`] to some value.
///
/// This mapping is _dense_: it stores a flat array indexed by `K::index()`,
/// with no hashing. For sparse mappings, use a HashMap.
#[derive(Clone)]
pub struct DenseIdMap<K, V> {
    data: Vec<Option<V>>,
    _marker: PhantomData<K>,
}

impl<K: NumericId + Debug, V: Debug> Debug for DenseIdMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut map = f.debug_map();
        for (k, v) in self.iter() {
            map.entry(&k, v);
        }
        map.finish()
    }
}

impl<K, V> Default for DenseIdMap<K, V> {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<K: NumericId, V> DenseIdMap<K, V> {
    /// Create an empty map with space for `n` entries pre-allocated.
    pub fn with_capacity(n: usize) -> Self {
        let mut res = Self::new();
        res.reserve_space(K::from_usize(n));
        res
    }

    /// Create an empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear the table's contents.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get the current capacity for the table.
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Get the number of ids currently indexed by the table (including "null"
    /// entries). This is a less useful version of "length" in other containers.
    pub fn n_ids(&self) -> usize {
        self.data.len()
    }

    /// Insert the given mapping into the table.
    pub fn insert(&mut self, key: K, value: V) {
        self.reserve_space(key);
        self.data[key.index()] = Some(value);
    }

    /// Get the key that would be returned by the next call to [`DenseIdMap::push`].
    pub fn next_id(&self) -> K {
        K::from_usize(self.data.len())
    }

    /// Add the given mapping to the table, returning the key corresponding to
    /// [`DenseIdMap::n_ids`].
    pub fn push(&mut self, val: V) -> K {
        let res = self.next_id();
        self.data.push(Some(val));
        res
    }

    /// Get the current mapping for `key` in the table.
    pub fn get(&self, key: K) -> Option<&V> {
        self.data.get(key.index())?.as_ref()
    }

    /// Get a mutable reference to the current mapping for `key` in the table.
    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        self.reserve_space(key);
        self.data.get_mut(key.index())?.as_mut()
    }

    /// Extract the value mapped to by `key` from the table.
    ///
    /// # Panics
    /// This method panics if `key` is not in the table.
    pub fn unwrap_val(&mut self, key: K) -> V {
        self.reserve_space(key);
        self.data.get_mut(key.index()).unwrap().take().unwrap()
    }

    /// Extract the value mapped to by `key` from the table, if it is present.
    pub fn take(&mut self, key: K) -> Option<V> {
        self.reserve_space(key);
        self.data.get_mut(key.index()).unwrap().take()
    }

    /// Get the current mapping for `key` in the table, or insert the value
    /// returned by `f` and return a mutable reference to it.
    pub fn get_or_insert(&mut self, key: K, f: impl FnOnce() -> V) -> &mut V {
        self.reserve_space(key);
        self.data[key.index()].get_or_insert_with(f)
    }

    pub fn iter(&self) -> impl Iterator<Item = (K, &V)> {
        self.data
            .iter()
            .enumerate()
            .filter_map(|(i, v)| Some((K::from_usize(i), v.as_ref()?)))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (K, &mut V)> {
        self.data
            .iter_mut()
            .enumerate()
            .filter_map(|(i, v)| Some((K::from_usize(i), v.as_mut()?)))
    }

    /// Reserve space up to the given key in the table.
    pub fn reserve_space(&mut self, key: K) {
        let index = key.index();
        if index >= self.data.len() {
            self.data.resize_with(index + 1, || None);
        }
    }

    pub fn drain(&mut self) -> impl Iterator<Item = (K, V)> + '_ {
        // To avoid the need to write down the return type.
        self.data
            .drain(..)
            .enumerate()
            .filter_map(|(i, v)| Some((K::from_usize(i), v?)))
    }
}

impl<K: NumericId, V: Send + Sync> DenseIdMap<K, V> {
    /// Get a parallel iterator over the entries in the table.
    pub fn par_iter(&self) -> impl ParallelIterator<Item = (K, &V)> {
        self.data
            .par_iter()
            .enumerate()
            .filter_map(|(i, v)| Some((K::from_usize(i), v.as_ref()?)))
    }

    /// Get a parallel iterator over mutable references to the entries in the table.
    pub fn par_iter_mut(&mut self) -> impl ParallelIterator<Item = (K, &mut V)> {
        self.data
            .par_iter_mut()
            .enumerate()
            .filter_map(|(i, v)| Some((K::from_usize(i), v.as_mut()?)))
    }
}

impl<K: NumericId, V> ops::Index<K> for DenseIdMap<K, V> {
    type Output = V;

    fn index(&self, key: K) -> &Self::Output {
        self.get(key).unwrap()
    }
}

impl<K: NumericId, V> ops::IndexMut<K> for DenseIdMap<K, V> {
    fn index_mut(&mut self, key: K) -> &mut Self::Output {
        self.get_mut(key).unwrap()
    }
}

impl<K: NumericId, V: Default> DenseIdMap<K, V> {
    pub fn get_or_default(&mut self, key: K) -> &mut V {
        self.get_or_insert(key, V::default)
    }
}

pub struct IdVec<K, V> {
    data: Vec<V>,
    _marker: std::marker::PhantomData<K>,
}

impl<K, V> Default for IdVec<K, V> {
    fn default() -> IdVec<K, V> {
        IdVec {
            data: Default::default(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<K, V: Clone> Clone for IdVec<K, V> {
    fn clone(&self) -> Self {
        IdVec {
            data: self.data.clone(),
            _marker: std::marker::PhantomData,
        }
    }
}

/// Like a [`DenseIdMap`], but supports freeing (and reusing) slots.
#[derive(Clone)]
pub struct DenseIdMapWithReuse<K, V> {
    data: DenseIdMap<K, V>,
    free: Vec<K>,
}

impl<K, V> Default for DenseIdMapWithReuse<K, V> {
    fn default() -> Self {
        Self {
            data: Default::default(),
            free: Default::default(),
        }
    }
}

impl<K: NumericId, V> DenseIdMapWithReuse<K, V> {
    /// Reserve a slot in the map for use later with [`DenseIdMapWithReuse::insert`].
    pub fn reserve_slot(&mut self) -> K {
        match self.free.pop() {
            Some(res) => res,
            None => {
                let res = self.data.next_id();
                self.data.reserve_space(res);
                res
            }
        }
    }

    /// Insert the given mapping into the table. You probably
    /// want to use [`DenseIdMapWithReuse::push`] instead, unless you need to use
    /// the key to build the value, in which case you can
    /// use [`DenseIdMapWithReuse::reserve_slot`] to get the key for this method.
    pub fn insert(&mut self, key: K, value: V) {
        self.data.insert(key, value)
    }

    /// Add the given value to the table.
    pub fn push(&mut self, value: V) -> K {
        let res = self.reserve_slot();
        self.insert(res, value);
        res
    }

    /// Remove the given key from the table, if it is present.
    pub fn take(&mut self, id: K) -> Option<V> {
        let res = self.data.take(id);
        if res.is_some() {
            self.free.push(id);
        }
        res
    }
}

impl<K: NumericId, V> std::ops::Index<K> for DenseIdMapWithReuse<K, V> {
    type Output = V;
    fn index(&self, key: K) -> &V {
        &self.data[key]
    }
}

impl<K: NumericId, V> std::ops::IndexMut<K> for DenseIdMapWithReuse<K, V> {
    fn index_mut(&mut self, key: K) -> &mut V {
        &mut self.data[key]
    }
}

impl<K: NumericId, V> IdVec<K, V> {
    pub fn with_capacity(cap: usize) -> IdVec<K, V> {
        IdVec {
            data: Vec::with_capacity(cap),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn push(&mut self, elt: V) -> K {
        let res = K::from_usize(self.data.len());
        self.data.push(elt);
        res
    }

    pub fn resize_with(&mut self, size: usize, init: impl FnMut() -> V) {
        self.data.resize_with(size, init)
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.data.iter()
    }

    pub fn iter(&self) -> impl Iterator<Item = (K, &V)> {
        self.data
            .iter()
            .enumerate()
            .map(|(i, v)| (K::from_usize(i), v))
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (K, &mut V)> {
        self.data
            .iter_mut()
            .enumerate()
            .map(|(i, v)| (K::from_usize(i), v))
    }
    pub fn drain(&mut self) -> impl Iterator<Item = (K, V)> + '_ {
        self.data
            .drain(..)
            .enumerate()
            .map(|(i, v)| (K::from_usize(i), v))
    }
    pub fn get(&self, key: K) -> Option<&V> {
        self.data.get(key.index())
    }
}

impl<K: NumericId, V: Send + Sync> IdVec<K, V> {
    pub fn par_iter_mut(&mut self) -> impl IndexedParallelIterator<Item = (K, &mut V)> {
        self.data
            .par_iter_mut()
            .with_max_len(1)
            .enumerate()
            .map(|(i, v)| (K::from_usize(i), v))
    }
}

impl<K: NumericId, V> ops::Index<K> for IdVec<K, V> {
    type Output = V;

    fn index(&self, key: K) -> &Self::Output {
        &self.data[key.index()]
    }
}

impl<K: NumericId, V> ops::IndexMut<K> for IdVec<K, V> {
    fn index_mut(&mut self, key: K) -> &mut Self::Output {
        &mut self.data[key.index()]
    }
}

use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

#[macro_export]
#[doc(hidden)]
macro_rules! atomic_of {
    (usize) => {
        std::sync::atomic::AtomicUsize
    };
    (u8) => {
        std::sync::atomic::AtomicU8
    };
    (u16) => {
        std::sync::atomic::AtomicU16
    };
    (u32) => {
        std::sync::atomic::AtomicU32
    };
    (u64) => {
        std::sync::atomic::AtomicU64
    };
}

#[macro_export]
macro_rules! define_id {
    ($v:vis $name:ident, $repr:tt) => { define_id!($v, $name, $repr, ""); };
    ($v:vis $name:ident, $repr:tt, $doc:tt) => {
        #[derive(Copy, Clone)]
        #[doc = $doc]
        $v struct $name {
            rep: $repr,
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.rep == other.rep
            }
        }

        impl Eq for $name {}

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for $name {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.rep.cmp(&other.rep)
            }
        }

        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.rep.hash(state);
            }
        }

        impl $name {
            #[allow(unused)]
            $v const fn new_const(id: $repr) -> Self {
                $name {
                    rep: id,
                }
            }

            #[allow(unused)]
            $v fn range(low: Self, high: Self) -> impl Iterator<Item = Self> {
                use $crate::NumericId;
                (low.rep..high.rep).map(|i| $name::new(i))
            }

        }

        impl $crate::NumericId for $name {
            type Rep = $repr;
            type Atomic = $crate::atomic_of!($repr);
            fn new(id: $repr) -> Self {
                Self::new_const(id)
            }
            fn from_usize(index: usize) -> Self {
                assert!(<$repr>::MAX as usize >= index,
                    "overflowing id type {} (represented as {}) with index {}", stringify!($name), stringify!($repr), index);
                $name::new(index as $repr)
            }
            fn index(self) -> usize {
                self.rep as usize
            }
            fn rep(self) -> $repr {
                self.rep
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(fmt, "{}({:?})", stringify!($name), self.rep)
            }
        }
    };
}
