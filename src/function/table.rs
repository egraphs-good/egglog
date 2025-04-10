//! A table type used to represent functions.
//!
//! Tables are essentially hash table mapping from vectors of values to values,
//! but they make different trade-offs than standard HashMaps or IndexMaps:
//!
//! * Like indexmap, tables preserve insertion order and support lookups based on
//! vector-like "offsets" in addition to table keys.
//!
//! * Unlike indexmap, these tables support constant-time removals that preserve
//! insertion order. Removals merely mark entries as "stale."
//!
//! These features come at the cost of needing to periodically rehash the table.
//! These rehashes must be done explicitly because they perturb the integer
//! table offsets that are otherwise stable. We prefer explicit rehashes because
//! column-level indexes use raw offsets into the table, and they need to be
//! rebuilt when a rehash happens.
//!
//! The advantage of these features is that tables can be sorted by "timestamp,"
//! making it efficient to iterate over subsets of a table matching a given
//! timestamp range.
//!
//! Note on rehashing: We will eventually want to keep old/stale entries around
//! to facilitate proofs/provenance. Early testing found that removing this in
//! the "obvious" way (keeping 'vals' around, avoiding `mem::take()`s for stale
//! entries, keeping stale entries out of `table` made some workloads very slow.
//! It's likely that we will have to store these "on the side" or use some sort
//! of persistent data-structure for the entire table.
use std::{
    fmt::{Debug, Formatter},
    hash::{BuildHasher, Hash, Hasher},
    mem,
    ops::Range,
};

use hashbrown::HashTable;

use super::binary_search::binary_search_table_by_key;
use crate::{util::BuildHasher as BH, TupleOutput, Value, ValueVec};

type Offset = usize;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct TableOffset {
    // Hashes are stored inline in the table to avoid cache misses during
    // probing, and to avoid `values` lookups entirely during insertions.
    hash: u64,
    off: Offset,
}

#[derive(Default, Clone)]
pub(crate) struct Table {
    max_ts: u32,
    n_stale: usize,
    table: HashTable<TableOffset>,
    pub(crate) vals: Vec<(Input, TupleOutput)>,
}

impl Debug for Table {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Table")
            .field("max_ts", &self.max_ts)
            .field("n_stale", &self.n_stale)
            .field("vals", &self.vals)
            .finish()
    }
}

impl Table {
    /// Clear the contents of the table.
    pub(crate) fn clear(&mut self) {
        self.max_ts = 0;
        self.n_stale = 0;
        self.table.clear();
        self.vals.clear();
    }

    /// Indicates whether or not the table should be rehashed.
    pub(crate) fn too_stale(&self) -> bool {
        self.n_stale > (self.vals.len() / 2)
    }

    /// Rehashes the table, invalidating any offsets stored into the table.
    pub(crate) fn rehash(&mut self) {
        let mut dst = 0usize;
        self.table.clear();
        self.vals.retain(|(inp, _)| {
            if inp.live() {
                let hash = hash_values(inp.data());
                let to = TableOffset { hash, off: dst };
                self.table
                    .entry(hash, |to2| to2 == &to, |to2| to2.hash)
                    .insert(to);
                dst += 1;
                true
            } else {
                false
            }
        });
        self.n_stale = 0;
    }

    /// Get the entry in the table for the given values, if they are in the
    /// table.
    pub(crate) fn get(&self, inputs: &[Value]) -> Option<&TupleOutput> {
        let hash = hash_values(inputs);
        let &TableOffset { off, .. } = self.table.find(hash, self.search_for(hash, inputs))?;
        debug_assert!(self.vals[off].0.live());
        Some(&self.vals[off].1)
    }

    /// Insert the given data into the table at the given timestamp. Return the
    /// previous value, if there was one.
    pub(crate) fn insert(&mut self, inputs: &[Value], out: Value, ts: u32) -> Option<Value> {
        let mut res = None;
        self.insert_and_merge(inputs, ts, false, |prev| {
            res = prev;
            out
        });
        res
    }

    /// Insert the given data into the table at the given timestamp. Thismethod
    /// allows for efficient 'merges', conditional on the previous value mapped
    /// to by the given index.
    ///
    /// * `on_merge(None)` should return the value mapping to the given slot.
    /// * `on_merge(Some(x))` can return a "merged" value (e.g. the union union
    ///   of `x` and `on_merge(None)`).
    pub(crate) fn insert_and_merge(
        &mut self,
        inputs: &[Value],
        ts: u32,
        subsumed: bool,
        on_merge: impl FnOnce(Option<Value>) -> Value,
    ) {
        assert!(ts >= self.max_ts);
        self.max_ts = ts;
        let hash = hash_values(inputs);
        if let Some(TableOffset { off, .. }) = self
            .table
            .find_mut(hash, search_for(&self.vals, hash, inputs))
        {
            let (inp, prev) = &mut self.vals[*off];
            let prev_subsumed = prev.subsumed;
            let next = on_merge(Some(prev.value));
            if next == prev.value && prev_subsumed == subsumed {
                return;
            }
            inp.stale_at = ts;
            self.n_stale += 1;
            let k = mem::take(&mut inp.data);
            let new_offset = self.vals.len();
            self.vals.push((
                Input::new(k),
                TupleOutput {
                    value: next,
                    timestamp: ts,
                    subsumed: subsumed || prev_subsumed,
                },
            ));
            *off = new_offset;
            return;
        }
        let new_offset = self.vals.len();
        self.vals.push((
            Input::new(ValueVec::from_slice(inputs)),
            TupleOutput {
                value: on_merge(None),
                timestamp: ts,
                subsumed,
            },
        ));
        let to = TableOffset {
            hash,
            off: new_offset,
        };
        self.table
            .entry(hash, |to2| to2 == &to, |to2| to2.hash)
            .insert(to);
    }

    /// One more than the maximum (potentially) valid offset into the table.
    pub(crate) fn num_offsets(&self) -> usize {
        self.vals.len()
    }

    /// One more than the actual valid offset (not including stale) into the table.
    pub(crate) fn len(&self) -> usize {
        self.vals.len() - self.n_stale
    }

    /// Whether the table is completely empty, including stale entries.
    pub(crate) fn is_empty(&self) -> bool {
        self.num_offsets() == 0
    }

    /// The minimum timestamp stored by the table, if there is one.
    pub(crate) fn min_ts(&self) -> Option<u32> {
        Some(self.vals.first()?.1.timestamp)
    }

    /// An upper bound for all timestamps stored in the table.
    pub(crate) fn max_ts(&self) -> u32 {
        self.max_ts
    }

    /// Get the timestamp for the entry at index `i`.
    pub(crate) fn get_timestamp(&self, i: usize) -> Option<u32> {
        Some(self.vals.get(i)?.1.timestamp)
    }

    /// Remove the given mapping from the table, returns whether an entry was
    /// removed.
    pub(crate) fn remove(&mut self, inp: &[Value], ts: u32) -> bool {
        let hash = hash_values(inp);
        let Ok(entry) = self
            .table
            .find_entry(hash, search_for(&self.vals, hash, inp))
        else {
            return false;
        };
        let (TableOffset { off, .. }, _) = entry.remove();
        self.vals[off].0.stale_at = ts;
        self.n_stale += 1;
        true
    }

    /// Returns the entries at the given index if the entry is live (and possibly not subsumed) and the index in bounds.
    pub(crate) fn get_index(
        &self,
        i: usize,
        include_subsumed: bool,
    ) -> Option<(&[Value], &TupleOutput)> {
        let (inp, out) = self.vals.get(i)?;
        if !valid_value(inp, out, include_subsumed) {
            return None;
        }
        Some((inp.data(), out))
    }

    /// Iterate over the live entries in the table, in insertion order.
    pub(crate) fn iter(
        &self,
        include_subsumed: bool,
    ) -> impl Iterator<Item = (&[Value], &TupleOutput)> + '_ {
        self.iter_range(0..self.num_offsets(), include_subsumed)
            .map(|(_, y, z)| (y, z))
    }

    /// Iterate over the live entries in the offset range, passing back the
    /// offset corresponding to each entry.
    pub(crate) fn iter_range(
        &self,
        range: Range<usize>,
        include_subsumed: bool,
    ) -> impl Iterator<Item = (usize, &[Value], &TupleOutput)> + '_ {
        self.vals[range.clone()]
            .iter()
            .zip(range)
            .filter_map(move |((inp, out), i)| {
                if valid_value(inp, out, include_subsumed) {
                    Some((i, inp.data(), out))
                } else {
                    None
                }
            })
    }

    #[cfg(debug_assertions)]
    pub(crate) fn assert_sorted(&self) {
        assert!(self
            .vals
            .windows(2)
            .all(|xs| xs[0].1.timestamp <= xs[1].1.timestamp))
    }

    /// Iterate over the live entries in the timestamp range, passing back their
    /// offset into the table.
    pub(crate) fn iter_timestamp_range(
        &self,
        range: &Range<u32>,
        include_subsumed: bool,
    ) -> impl Iterator<Item = (usize, &[Value], &TupleOutput)> + '_ {
        let indexes = self.transform_range(range);
        self.iter_range(indexes, include_subsumed)
    }

    /// Return the approximate number of entries in the table for the given
    /// timestamp range.
    pub(crate) fn approximate_range_size(&self, range: &Range<u32>) -> usize {
        let indexes = self.transform_range(range);
        indexes.end - indexes.start
    }

    /// Transform a range of timestamps to the corresponding range of indexes
    /// into the table.
    pub(crate) fn transform_range(&self, range: &Range<u32>) -> Range<usize> {
        if let Some(start) = binary_search_table_by_key(self, range.start) {
            if let Some(end) = binary_search_table_by_key(self, range.end) {
                start..end
            } else {
                start..self.num_offsets()
            }
        } else {
            0..0
        }
    }

    /// Used for the HashTable probe sequence.
    fn search_for<'a>(
        &'a self,
        hash: u64,
        input: &'a [Value],
    ) -> impl Fn(&TableOffset) -> bool + 'a {
        search_for(&self.vals, hash, input)
    }
}

/// Used for the HashTable probe sequence.
fn search_for<'a>(
    vals: &'a [(Input, TupleOutput)],
    hash: u64,
    input: &'a [Value],
) -> impl Fn(&TableOffset) -> bool + 'a {
    move |to| {
        // Test that hashes match.
        if to.hash != hash {
            return false;
        }
        // If the hash matches, the value should not be stale, and the data
        // should match.
        let (inp, _) = &vals[to.off];
        inp.live() && inp.data() == input
    }
}

/// Returns whether the given value is live and not subsume (if the include_subsumed flag is false).
///
/// For checks, debugging, and serialization, we do want to include subsumed values.
/// but for matching on rules, we do not.
fn valid_value(input: &Input, output: &TupleOutput, include_subsumed: bool) -> bool {
    input.live() && (include_subsumed || !output.subsumed)
}

pub(crate) fn hash_values(vs: &[Value]) -> u64 {
    // Just hash the bits: all inputs to the same function should have matching
    // column types.
    let mut hasher = BH::default().build_hasher();
    for v in vs {
        v.bits.hash(&mut hasher);
    }
    hasher.finish()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Input {
    pub(crate) data: ValueVec,
    /// The timestamp at which the given input became "stale"
    stale_at: u32,
}

impl Input {
    fn new(data: ValueVec) -> Input {
        Input {
            data,
            stale_at: u32::MAX,
        }
    }

    fn data(&self) -> &[Value] {
        self.data.as_slice()
    }

    fn live(&self) -> bool {
        self.stale_at == u32::MAX
    }
}
