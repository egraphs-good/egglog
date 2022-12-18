//! A table type used to represent functions.
use std::{
    hash::{BuildHasher, Hash, Hasher},
    mem,
    ops::Range,
};

use hashbrown::raw::RawTable;

use crate::{
    binary_search::binary_search_table_by_key, util::BuildHasher as BH, Input, TupleOutput, Value,
};

type Offset = u32;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct TableOffset {
    hash: u64,
    off: Offset,
}

#[derive(Default, Clone)]
pub(crate) struct Table {
    max_ts: u32,
    n_stale: usize,
    table: RawTable<TableOffset>,
    vals: Vec<(Input, TupleOutput)>,
}

macro_rules! search_for {
    ($slf:expr, $hash:expr, $inp:expr) => {
        |to| to.hash == $hash && $slf.vals[to.off as usize].0.data() == $inp
    };
}

impl Table {
    pub(crate) fn clear(&mut self) {
        self.max_ts = 0;
        self.n_stale = 0;
        self.table.clear();
        self.vals.clear();
    }
    pub(crate) fn too_stale(&self) -> bool {
        self.n_stale > (self.vals.len() / 2)
    }
    pub(crate) fn rehash(&mut self) {
        let mut src = 0usize;
        let mut dst = 0usize;
        self.vals.retain(|(inp, _)| {
            if inp.live() {
                // Go back and remap the hash
                let hash = hash_values(inp.data());
                let TableOffset { off, .. } = self
                    .table
                    .get_mut(hash, |to| to.off == src as Offset)
                    .unwrap();
                *off = dst as Offset;
                src += 1;
                dst += 1;
                true
            } else {
                src += 1;
                false
            }
        });
        self.n_stale = 0;
    }
    pub(crate) fn get(&self, inputs: &[Value]) -> Option<&TupleOutput> {
        let hash = hash_values(inputs);
        let TableOffset { off, .. } = self.table.get(hash, search_for!(self, hash, inputs))?;
        debug_assert!(self.vals[*off as usize].0.live());
        Some(&self.vals[*off as usize].1)
    }
    pub(crate) fn insert(&mut self, inputs: &[Value], out: Value, ts: u32) -> Option<Value> {
        let mut res = None;
        self.insert_and_merge(inputs, ts, |prev| {
            res = prev;
            out
        });
        res
    }

    pub(crate) fn insert_and_merge(
        &mut self,
        inputs: &[Value],
        ts: u32,
        on_merge: impl FnOnce(Option<Value>) -> Value,
    ) {
        assert!(ts >= self.max_ts);
        self.max_ts = ts;
        let hash = hash_values(inputs);
        if let Some(TableOffset { off, .. }) =
            self.table.get_mut(hash, search_for!(self, hash, inputs))
        {
            let (inp, prev) = &mut self.vals[*off as usize];
            let next = on_merge(Some(prev.value));
            if next == prev.value {
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
                },
            ));
            *off = new_offset as Offset;
            return;
        }
        let new_offset = self.vals.len();
        self.vals.push((
            Input::new(inputs.into()),
            TupleOutput {
                value: on_merge(None),
                timestamp: ts,
            },
        ));
        self.table.insert(
            hash,
            TableOffset {
                hash,
                off: new_offset as Offset,
            },
            |off| off.hash,
        );
    }

    pub(crate) fn len(&self) -> usize {
        self.vals.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn min_ts(&self) -> Option<u32> {
        Some(self.vals.first()?.1.timestamp)
    }

    pub(crate) fn max_ts(&self) -> u32 {
        self.max_ts
    }

    pub(crate) fn get_timestamp(&self, i: usize) -> Option<u32> {
        Some(self.vals.get(i)?.1.timestamp)
    }

    pub(crate) fn remove(&mut self, inp: &[Value], ts: u32) -> bool {
        let hash = hash_values(inp);
        let entry = if let Some(entry) = self.table.remove_entry(hash, search_for!(self, hash, inp))
        {
            entry
        } else {
            return false;
        };
        self.vals[entry.off as usize].0.stale_at = ts;
        self.n_stale += 1;
        true
    }

    pub(crate) fn remove_index(&mut self, i: usize, ts: u32) {
        let (inp, _) = &self.vals[i];
        if !inp.live() {
            return;
        }
        let hash = hash_values(inp.data());
        self.table
            .remove_entry(hash, search_for!(self, hash, inp.data()))
            .expect("non-stale input should have an entry in the map");
        // re-borrow
        let (inp, _) = &mut self.vals[i];
        inp.stale_at = ts;
        self.n_stale += 1;
    }
    /// Returns the entries at the given index if the entry is live and the index in bounds.
    pub(crate) fn get_index(&self, i: usize) -> Option<(&[Value], &TupleOutput)> {
        let (inp, out) = self.vals.get(i)?;
        if !inp.live() {
            return None;
        }
        Some((inp.data(), out))
    }
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&[Value], &TupleOutput)> + '_ {
        self.iter_range(0..self.len()).map(|(x, y, z)| (y, z))
    }

    pub(crate) fn iter_range(
        &self,
        range: Range<usize>,
    ) -> impl Iterator<Item = (usize, &[Value], &TupleOutput)> + '_ {
        self.vals[range.clone()]
            .iter()
            .zip(range)
            .filter_map(|((inp, out), i)| {
                if inp.live() {
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

    // TODO:
    pub(crate) fn iter_timestamp_range(
        &self,
        range: &Range<u32>,
    ) -> impl Iterator<Item = (usize, &[Value], &TupleOutput)> + '_ {
        let indexes = self.transform_range(range);
        self.iter_range(indexes)
    }

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
                start..self.len()
            }
        } else {
            0..0
        }
    }
}

fn hash_values(vs: &[Value]) -> u64 {
    // Just hash the bits: all inputs to the same function should have matching
    // column types.
    let mut hasher = BH::default().build_hasher();
    for v in vs {
        v.bits.hash(&mut hasher);
    }
    hasher.finish()
}
