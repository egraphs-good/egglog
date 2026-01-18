//! A lightweight index that separates one-to-one mappings from one-to-many mappings.
//!
//! The index tracks a key `K` that maps to a value `V` along with the row ids that
//! introduced each mapping. Keys with a single observed value are stored in `unique`,
//! while keys with multiple observed values are stored in `repeat`.

use std::hash::Hash;

use indexmap::IndexMap;

use crate::core_relations::RowId;

#[derive(Debug, Clone)]
pub struct UniqueEntry<V> {
    value: V,
    row_ids: Vec<RowId>,
}

impl<V> UniqueEntry<V> {
    /// Return the unique value associated with this entry.
    pub fn value(&self) -> &V {
        &self.value
    }

    /// Return the row ids that introduced the unique value.
    pub fn row_ids(&self) -> &[RowId] {
        &self.row_ids
    }
}

/// An index that separates unique keys from keys with multiple distinct values.
#[derive(Debug, Clone)]
pub struct UniqueRepeatIndex<K, V> {
    unique: IndexMap<K, UniqueEntry<V>>,
    repeat: IndexMap<K, Vec<(V, RowId)>>,
}

impl<K, V> Default for UniqueRepeatIndex<K, V>
where
    K: Eq + Hash,
    V: Clone + Eq,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> UniqueRepeatIndex<K, V>
where
    K: Eq + Hash,
    V: Clone + Eq,
{
    /// Create an empty `UniqueRepeatIndex`.
    pub fn new() -> Self {
        Self {
            unique: IndexMap::new(),
            repeat: IndexMap::new(),
        }
    }

    /// Remove all entries from the index.
    pub fn clear(&mut self) {
        self.unique.clear();
        self.repeat.clear();
    }

    /// Return the number of keys that currently map to a single value.
    pub fn unique_len(&self) -> usize {
        self.unique.len()
    }

    /// Return the number of keys that currently map to multiple values.
    pub fn repeat_len(&self) -> usize {
        self.repeat.len()
    }

    /// Iterate over keys that currently map to a single value.
    pub fn unique_iter(&self) -> impl Iterator<Item = (&K, &UniqueEntry<V>)> {
        self.unique.iter()
    }

    /// Iterate over keys that currently map to multiple values.
    pub fn repeat_iter(&self) -> impl Iterator<Item = (&K, &[(V, RowId)])> {
        self.repeat
            .iter()
            .map(|(key, entries)| (key, entries.as_slice()))
    }

    /// Get the unique entry for the given key, if it exists.
    pub fn unique_entry(&self, key: &K) -> Option<&UniqueEntry<V>> {
        self.unique.get(key)
    }

    /// Get the repeat entries for the given key, if it exists.
    pub fn repeat_entry(&self, key: &K) -> Option<&[(V, RowId)]> {
        self.repeat.get(key).map(Vec::as_slice)
    }

    /// Add a `(key, value, row_id)` mapping to the index.
    ///
    /// If the key was unique and the value differs from the existing one,
    /// the key is moved to the repeat mapping.
    pub fn add(&mut self, key: K, value: V, row_id: RowId) {
        if let Some(entries) = self.repeat.get_mut(&key) {
            entries.push((value, row_id));
            return;
        }

        if let Some(entry) = self.unique.get_mut(&key) {
            if entry.value == value {
                entry.row_ids.push(row_id);
                return;
            }
        }

        let Some(entry) = self.unique.shift_remove(&key) else {
            self.unique.insert(
                key,
                UniqueEntry {
                    value,
                    row_ids: vec![row_id],
                },
            );
            return;
        };

        let mut entries = Vec::with_capacity(entry.row_ids.len() + 1);
        for existing_row in entry.row_ids {
            entries.push((entry.value.clone(), existing_row));
        }
        entries.push((value, row_id));
        self.repeat.insert(key, entries);
    }
}

#[cfg(test)]
mod tests {
    use super::UniqueRepeatIndex;
    use crate::core_relations::RowId;
    use crate::numeric_id::NumericId;

    #[test]
    fn unique_stays_unique_for_same_value() {
        let mut index = UniqueRepeatIndex::<String, u32>::new();
        let row0 = RowId::from_usize(0);
        let row1 = RowId::from_usize(1);
        index.add("a".to_string(), 10, row0);
        index.add("a".to_string(), 10, row1);

        assert_eq!(index.unique_len(), 1);
        assert_eq!(index.repeat_len(), 0);

        let entry = index.unique_entry(&"a".to_string()).unwrap();
        assert_eq!(*entry.value(), 10);
        assert_eq!(entry.row_ids(), &[row0, row1]);
    }

    #[test]
    fn unique_moves_to_repeat_on_new_value() {
        let mut index = UniqueRepeatIndex::<String, u32>::new();
        let row0 = RowId::from_usize(0);
        let row1 = RowId::from_usize(1);
        index.add("a".to_string(), 10, row0);
        index.add("a".to_string(), 20, row1);

        assert_eq!(index.unique_len(), 0);
        assert_eq!(index.repeat_len(), 1);

        let entries = index.repeat_entry(&"a".to_string()).unwrap();
        assert_eq!(entries, &[(10, row0), (20, row1)]);
    }

    #[test]
    fn repeat_accumulates_rows() {
        let mut index = UniqueRepeatIndex::<String, u32>::new();
        let row0 = RowId::from_usize(0);
        let row1 = RowId::from_usize(1);
        let row2 = RowId::from_usize(2);
        index.add("a".to_string(), 10, row0);
        index.add("a".to_string(), 20, row1);
        index.add("a".to_string(), 30, row2);

        let entries = index.repeat_entry(&"a".to_string()).unwrap();
        assert_eq!(entries, &[(10, row0), (20, row1), (30, row2)]);
    }

    #[test]
    fn unique_and_repeat_can_coexist() {
        let mut index = UniqueRepeatIndex::<String, u32>::new();
        let row0 = RowId::from_usize(0);
        let row1 = RowId::from_usize(1);
        let row2 = RowId::from_usize(2);
        index.add("a".to_string(), 10, row0);
        index.add("b".to_string(), 20, row1);
        index.add("b".to_string(), 30, row2);

        assert_eq!(index.unique_len(), 1);
        assert_eq!(index.repeat_len(), 1);

        let unique = index.unique_entry(&"a".to_string()).unwrap();
        assert_eq!(*unique.value(), 10);
        assert_eq!(unique.row_ids(), &[row0]);

        let repeat = index.repeat_entry(&"b".to_string()).unwrap();
        assert_eq!(repeat, &[(20, row1), (30, row2)]);
    }
}
