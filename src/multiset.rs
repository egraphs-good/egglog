use rpds::RedBlackTreeMapSync;
use std::hash::Hash;

/// Immutable multiset implementation, which is threadsafe and hash stable, regardless of insertion order.
///
/// All methods that return a new multiset take ownership of the old multiset.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub(crate) struct MultiSet<T: Hash + Ord + Clone>(
    /// All values should be > 0
    RedBlackTreeMapSync<T, usize>,
);

impl<T: Hash + Ord + Clone> MultiSet<T> {
    /// Create a new empty multiset.
    pub(crate) fn new() -> Self {
        MultiSet(RedBlackTreeMapSync::new_sync())
    }

    /// Check if the multiset contains a key.
    pub(crate) fn contains(&self, value: &T) -> bool {
        self.0.contains_key(value)
    }

    /// Return the total number of elements in the multiset.
    pub(crate) fn len(&self) -> usize {
        self.0.iter().map(|(_, v)| *v).sum()
    }

    /// Return an iterator over all elements in the multiset.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
        self.0
            .iter()
            .flat_map(|(k, v)| std::iter::repeat(k).take(*v))
    }

    /// Return an arbitrary element from the multiset.
    pub(crate) fn pick(&self) -> Option<&T> {
        self.0.first().map(|(k, _)| k)
    }

    /// Map a function over all elements in the multiset, taking ownership of it and returning a new multiset.
    pub(crate) fn map(self, mut f: impl FnMut(&T) -> T) -> MultiSet<T> {
        let mut new = MultiSet::new();
        for (k, v) in self.0.into_iter() {
            new.insert_multiple_mut(f(k), *v);
        }
        new
    }

    /// Insert a value into the multiset, taking ownership of it and returning a new multiset.
    pub(crate) fn insert(mut self, value: T) -> MultiSet<T> {
        self.insert_multiple_mut(value, 1);
        self
    }

    /// Remove a value from the multiset, taking ownership of it and returning a new multiset.
    pub(crate) fn remove(mut self, value: &T) -> Option<MultiSet<T>> {
        if let Some(v) = self.0.get(value) {
            if *v == 1 {
                self.0.remove_mut(value);
            } else {
                self.0.insert_mut(value.clone(), v - 1);
            }
            Some(self)
        } else {
            None
        }
    }

    fn insert_multiple_mut(&mut self, value: T, n: usize) {
        if let Some(v) = self.0.get(&value) {
            self.0.insert_mut(value, v + n);
        } else {
            self.0.insert_mut(value, n);
        }
    }

    /// Create a multiset from an iterator.
    pub(crate) fn from_iter(iter: impl IntoIterator<Item = T>) -> Self {
        let mut multiset = MultiSet::new();
        for value in iter {
            multiset.insert_multiple_mut(value, 1);
        }
        multiset
    }
}
