//! A crate with utilities for working with numeric Ids.
use std::{
    fmt::{self, Debug},
    hash::Hash,
    marker::PhantomData,
    mem::MaybeUninit,
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
#[derive(Clone, PartialEq, Eq, Hash)]
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
        if n > 0 {
            res.reserve_space(K::from_usize(n - 1));
        }
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

    /// Test whether `key` is set in this map.
    pub fn contains_key(&self, key: K) -> bool {
        self.data.get(key.index()).is_some_and(Option::is_some)
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

    pub fn raw(&self) -> &[Option<V>] {
        &self.data
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

/// Space-optimized version of [`DenseIdMap`] that uses a bitset to track
/// occupied entries, reducing memory usage for sparse maps.
pub struct DenseIdMapSO<K: NumericId, V> {
    data: Vec<MaybeUninit<V>>,
    bitset: SmallVec<[u64; 2]>,
    _marker: PhantomData<K>,
}

impl<K: NumericId, V: Clone> Clone for DenseIdMapSO<K, V> {
    fn clone(&self) -> Self {
        let mut new_map = DenseIdMapSO::with_capacity(self.n_ids());
        for (k, v) in self.iter() {
            new_map.insert(k, v.clone());
        }
        new_map
    }
}

impl<K: NumericId + PartialEq, V: PartialEq> PartialEq for DenseIdMapSO<K, V> {
    fn eq(&self, other: &Self) -> bool {
        if self.n_ids() != other.n_ids() {
            return false;
        }
        for (k, v) in self.iter() {
            match other.get(k) {
                Some(ov) if ov == v => {}
                _ => return false,
            }
        }
        true
    }
}

impl<K: NumericId + Eq, V: Eq> Eq for DenseIdMapSO<K, V> {}

impl<K: NumericId, V: Hash> Hash for DenseIdMapSO<K, V> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for (k, v) in self.iter() {
            k.hash(state);
            v.hash(state);
        }
    }
}

impl<K: NumericId + Debug, V: Debug> Debug for DenseIdMapSO<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut map = f.debug_map();
        for (k, v) in self.iter() {
            map.entry(&k, v);
        }
        map.finish()
    }
}

impl<K: NumericId, V> Default for DenseIdMapSO<K, V> {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            _marker: PhantomData,
            bitset: SmallVec::new(),
        }
    }
}

/// Check if a bit is set in the bitset at the given index.
fn is_bit_set(bitset: &[u64], index: usize) -> bool {
    bitset
        .get(index / 64)
        .is_some_and(|word| (word & (1 << (index % 64))) != 0)
}

unsafe fn is_bit_set_unchecked(bitset: &[u64], index: usize) -> bool {
    let word = unsafe { bitset.get_unchecked(index / 64) };
    (word & (1 << (index % 64))) != 0
}

impl<K: NumericId, V> DenseIdMapSO<K, V> {
    #[inline(always)]
    unsafe fn set_bit_unchecked(&mut self, index: usize) {
        unsafe {
            let word = self.bitset.get_unchecked_mut(index / 64);
            *word |= 1 << (index % 64);
        }
    }

    #[inline(always)]
    unsafe fn clear_bit_unchecked(&mut self, index: usize) {
        unsafe {
            let word = self.bitset.get_unchecked_mut(index / 64);
            *word &= !(1 << (index % 64));
        }
    }

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
        // Drop all initialized values
        for i in 0..self.data.len() {
            if self.contains_key(K::from_usize(i)) {
                // SAFETY: The bitset indicates this slot is initialized
                unsafe {
                    self.data[i].assume_init_drop();
                }
            }
        }
        self.data.clear();
        self.bitset.clear();
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
        let index = key.index();

        // SAFETY: reserve_space
        unsafe {
            if is_bit_set_unchecked(&self.bitset, index) {
                self.data.get_unchecked_mut(index).assume_init_drop();
            }
            self.data.get_unchecked_mut(index).write(value);
            self.set_bit_unchecked(index);
        }
    }

    /// Get the key that would be returned by the next call to [`DenseIdMapSO::push`].
    pub fn next_id(&self) -> K {
        K::from_usize(self.data.len())
    }

    /// Add the given mapping to the table, returning the key corresponding to
    /// [`DenseIdMapSO::n_ids`].
    pub fn push(&mut self, val: V) -> K {
        let res = self.next_id();
        let index = res.index();

        self.data.push(MaybeUninit::new(val));
        if index / 64 >= self.bitset.len() {
            self.bitset.push(0);
        }
        unsafe {
            self.set_bit_unchecked(index);
        }

        res
    }

    /// Test whether `key` is set in this map.
    pub fn contains_key(&self, key: K) -> bool {
        is_bit_set(&self.bitset, key.index())
    }

    /// Get the current mapping for `key` in the table.
    pub fn get(&self, key: K) -> Option<&V> {
        if !self.contains_key(key) {
            return None;
        }
        // SAFETY: The bitset indicates this slot is initialized
        unsafe { Some(self.data.get_unchecked(key.index()).assume_init_ref()) }
    }

    /// Get a mutable reference to the current mapping for `key` in the table.
    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        if !self.contains_key(key) {
            return None;
        }
        // SAFETY: The bitset indicates this slot is initialized
        unsafe { Some(self.data.get_unchecked_mut(key.index()).assume_init_mut()) }
    }

    /// Extract the value mapped to by `key` from the table.
    ///
    /// # Panics
    /// This method panics if `key` is not in the table.
    pub fn unwrap_val(&mut self, key: K) -> V {
        self.take(key).expect("key not found in DenseIdMapSO")
    }

    /// Extract the value mapped to by `key` from the table, if it is present.
    pub fn take(&mut self, key: K) -> Option<V> {
        if !self.contains_key(key) {
            return None;
        }

        let index = key.index();

        // SAFETY: The bitset indicates this slot is initialized
        let value = unsafe { self.data.get_unchecked(index).assume_init_read() };

        // SAFETY: contains_key succeeded
        unsafe {
            self.clear_bit_unchecked(index);
        }

        Some(value)
    }

    /// Get the current mapping for `key` in the table, or insert the value
    /// returned by `f` and return a mutable reference to it.
    pub fn get_or_insert(&mut self, key: K, f: impl FnOnce() -> V) -> &mut V {
        self.reserve_space(key);
        let index = key.index();

        unsafe {
            // SAFETY: reserve_space
            if !is_bit_set_unchecked(&self.bitset, index) {
                self.data.get_unchecked_mut(index).write(f());
                self.set_bit_unchecked(index);
            }
        }

        // SAFETY: Either the value was already initialized, or we just initialized it
        unsafe { self.data.get_unchecked_mut(index).assume_init_mut() }
    }

    pub fn raw(&self) -> &[MaybeUninit<V>] {
        &self.data
    }

    pub fn iter(&self) -> impl Iterator<Item = (K, &V)> + '_ {
        self.bitset
            .iter()
            .enumerate()
            .flat_map(move |(word_idx, &word)| {
                let base_idx = word_idx * 64;
                if word == 0 {
                    return itertools::Either::Left(std::iter::empty());
                }
                if word == u64::MAX {
                    let slice = &self.data[base_idx..base_idx + 64];
                    return itertools::Either::Right(itertools::Either::Left(
                        slice.iter().enumerate().map(move |(sub_idx, v)| {
                            // SAFETY: word is u64::MAX, so all 64 elements are initialized
                            (K::from_usize(base_idx + sub_idx), unsafe {
                                v.assume_init_ref()
                            })
                        }),
                    ));
                }
                let mut w = word;
                itertools::Either::Right(itertools::Either::Right(std::iter::from_fn(move || {
                    if w == 0 {
                        return None;
                    }
                    let bit_idx = w.trailing_zeros() as usize;
                    w &= !(1 << bit_idx); // Clear the least significant bit
                    let total_idx = base_idx + bit_idx;
                    // SAFETY: Bitset confirms initialization
                    unsafe {
                        Some((
                            K::from_usize(total_idx),
                            self.data.get_unchecked(total_idx).assume_init_ref(),
                        ))
                    }
                })))
            })
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (K, &mut V)> + '_ {
        self.bitset
            .iter()
            .enumerate()
            .zip(self.data.chunks_mut(64))
            .flat_map(|((word_idx, word), chunk)| {
                let base_idx = word_idx * 64;
                if *word == 0 {
                    return itertools::Either::Left(std::iter::empty());
                }
                if *word == u64::MAX {
                    return itertools::Either::Right(itertools::Either::Left(
                        chunk.iter_mut().enumerate().map(move |(sub_idx, v)| {
                            // SAFETY: word is u64::MAX, so all 64 elements are initialized
                            (K::from_usize(base_idx + sub_idx), unsafe {
                                v.assume_init_mut()
                            })
                        }),
                    ));
                }
                let w = *word;
                itertools::Either::Right(itertools::Either::Right(
                    chunk
                        .iter_mut()
                        .enumerate()
                        .filter_map(move |(sub_idx, v)| {
                            if (w & (1 << sub_idx)) == 0 {
                                return None;
                            }
                            let total_idx = base_idx + sub_idx;
                            // SAFETY: Bitset confirms initialization
                            unsafe { Some((K::from_usize(total_idx), v.assume_init_mut())) }
                        }),
                ))
            })
    }

    /// Reserve space up to the given key in the table.
    pub fn reserve_space(&mut self, key: K) {
        let index = key.index();
        if index >= self.data.len() {
            self.data.resize_with(index + 1, MaybeUninit::uninit);
            self.bitset.resize((index / 64) + 1, 0);
        }
    }

    pub fn drain(&mut self) -> impl Iterator<Item = (K, V)> + '_ {
        let bitset = std::mem::take(&mut self.bitset);
        self.data.drain(..).enumerate().filter_map(move |(i, v)| {
            if is_bit_set(&bitset, i) {
                // SAFETY: The bitset indicates this slot is initialized
                Some((K::from_usize(i), unsafe { v.assume_init_read() }))
            } else {
                None
            }
        })
    }
}

impl<K: NumericId, V: Send + Sync> DenseIdMapSO<K, V> {
    /// Get a parallel iterator over the entries in the table.
    pub fn par_iter(&self) -> impl ParallelIterator<Item = (K, &V)> {
        let bitset = &self.bitset;
        self.data.par_iter().enumerate().filter_map(|(i, v)| {
            if is_bit_set(bitset, i) {
                // SAFETY: The bitset indicates this slot is initialized
                Some((K::from_usize(i), unsafe { v.assume_init_ref() }))
            } else {
                None
            }
        })
    }

    /// Get a parallel iterator over mutable references to the entries in the table.
    pub fn par_iter_mut(&mut self) -> impl ParallelIterator<Item = (K, &mut V)> {
        let bitset = &self.bitset;
        self.data.par_iter_mut().enumerate().filter_map(|(i, v)| {
            if is_bit_set(bitset, i) {
                // SAFETY: The bitset indicates this slot is initialized
                Some((K::from_usize(i), unsafe { v.assume_init_mut() }))
            } else {
                None
            }
        })
    }
}

impl<K: NumericId, V> Drop for DenseIdMapSO<K, V> {
    fn drop(&mut self) {
        // Drop all initialized values
        for i in 0..self.data.len() {
            if is_bit_set(&self.bitset, i) {
                // SAFETY: The bitset indicates this slot is initialized
                unsafe {
                    self.data[i].assume_init_drop();
                }
            }
        }
    }
}

impl<K: NumericId, V> ops::Index<K> for DenseIdMapSO<K, V> {
    type Output = V;

    fn index(&self, key: K) -> &Self::Output {
        self.get(key).unwrap()
    }
}

impl<K: NumericId, V> ops::IndexMut<K> for DenseIdMapSO<K, V> {
    fn index_mut(&mut self, key: K) -> &mut Self::Output {
        self.get_mut(key).unwrap()
    }
}

impl<K: NumericId, V: Default> DenseIdMapSO<K, V> {
    pub fn get_or_default(&mut self, key: K) -> &mut V {
        self.get_or_insert(key, V::default)
    }
}

#[derive(Debug)]
pub struct IdVec<K, V> {
    data: Vec<V>,
    _marker: std::marker::PhantomData<K>,
}

impl<K, V> IdVec<K, V> {
    pub fn clear(&mut self) {
        self.data.clear();
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }
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
pub struct DenseIdMapWithReuse<K: NumericId, V> {
    data: DenseIdMap<K, V>,
    free: Vec<K>,
}

impl<K: NumericId, V> Default for DenseIdMapWithReuse<K, V> {
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
use smallvec::SmallVec;

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
            /// return the inner representation of id as usize
            fn index(self) -> usize {
                self.rep as usize
            }
            /// return the inner representation of id.
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
