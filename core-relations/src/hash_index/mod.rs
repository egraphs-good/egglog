//! Hash-based secondary indexes.
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    cmp,
    hash::{Hash, Hasher},
    mem,
    sync::Mutex,
};

use crate::{
    common::IndexMap,
    numeric_id::{IdVec, NumericId, define_id},
};
use egglog_concurrency::{ReadOptimizedLock, ThreadPool};
use hashbrown::HashTable;
use indexmap::map::Entry;
use once_cell::sync::Lazy;
use rustc_hash::FxHasher;
use smallvec::SmallVec;

use crate::{
    OffsetRange, Subset,
    common::{ShardData, ShardId, Value},
    offsets::{RowId, SortedOffsetSlice, SubsetRef},
    parallel,
    parallel_heuristics::parallelize_index_construction,
    pool::{Pooled, with_pool_set},
    row_buffer::{RowBuffer, TaggedRowBuffer},
    table_spec::{ColumnId, Generation, Offset, TableVersion, WrappedTableRef},
};

#[cfg(test)]
mod tests;

#[doc(hidden)]
pub mod bench_support;

#[derive(Clone)]
pub(crate) struct TableEntry<T> {
    hash: u64,
    /// Points into `keys`
    key: RowId,
    vals: T,
}

#[derive(Clone)]
pub(crate) struct Index<TI> {
    key: Vec<ColumnId>,
    updated_to: TableVersion,
    table: TI,
}

/// A stable position for one key within a frozen index.
///
/// Positions are scoped to the exact index that produced them. They may be
/// retained while that index is only read. Existing positions also survive an
/// append-only incremental refresh, but a clear/full rebuild invalidates every
/// position. The two components are deliberately private so callers cannot
/// manufacture positions belonging to another index.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct IndexPosition {
    shard: u32,
    slot: u32,
}

impl IndexPosition {
    fn new(shard: ShardId, slot: usize) -> Self {
        Self {
            shard: u32::try_from(shard.index()).expect("index shard exceeds u32::MAX"),
            slot: u32::try_from(slot).expect("index shard contains more than u32::MAX keys"),
        }
    }

    pub(crate) fn shard(self) -> usize {
        self.shard as usize
    }

    pub(crate) fn slot(self) -> usize {
        self.slot as usize
    }
}

impl<TI: IndexBase> Index<TI> {
    pub(crate) fn new(key: Vec<ColumnId>, table: TI) -> Self {
        Index {
            key,
            updated_to: TableVersion {
                major: Generation::new(0),
                minor: Offset::new(0),
            },
            table,
        }
    }

    /// Get the nonempty subset of rows associated with this key, if there is
    /// one.
    pub(crate) fn get_subset<'a>(&'a self, key: &TI::Key) -> Option<SubsetRef<'a>> {
        self.table.get_subset(key)
    }

    pub(crate) fn needs_refresh(&self, table: WrappedTableRef) -> bool {
        table.version() != self.updated_to
    }

    pub(crate) fn refresh(&mut self, table: WrappedTableRef) {
        let cur_version = table.version();
        if cur_version == self.updated_to {
            return;
        }
        let is_full = cur_version.major != self.updated_to.major;
        let subset = if is_full {
            self.table.clear();
            table.all()
        } else {
            table.updates_since(self.updated_to.minor)
        };
        // Three ways to fold `subset` into the index; all produce the same result (each value's
        // rows sorted ascending and de-duplicated) but trade fixed overhead against throughput:
        //
        // * `merge_parallel` shards the rows across worker threads. It handles both full and
        //   incremental refreshes (it appends to whatever is already indexed), but the thread
        //   coordination only pays off once `subset` clears the size cutoff.
        // * `rebuild_full` is the serial bulk path for a full rebuild: it sorts all (value, row)
        //   pairs and sizes each key's subset in a single allocation, avoiding the repeated
        //   regrowth of the row-at-a-time path. It assumes an empty index, so it is only valid
        //   right after the major-version `clear()` above.
        // * `refresh_serial` scans in batches and inserts one row at a time into the existing
        //   index. Lowest fixed cost, and the only path suited to merging a small incremental
        //   delta into the index's prior contents.
        if parallelize_index_construction(subset.size()) {
            self.table.merge_parallel(&self.key, table, subset.as_ref());
        } else if is_full {
            self.table.rebuild_full(&self.key, table, subset.as_ref());
        } else {
            self.refresh_serial(table, subset);
        }

        self.updated_to = cur_version;
    }

    /// Update the contents of the index to the current version of the table.
    ///
    /// The index is guaranteed to be up to date until `merge` is called on the
    /// table again.
    pub(crate) fn refresh_serial(&mut self, table: WrappedTableRef, subset: Subset) {
        let mut buf = TaggedRowBuffer::new(self.key.len());
        let mut cur = Offset::new(0);
        loop {
            buf.clear();
            if let Some(next) =
                table.scan_project(subset.as_ref(), &self.key, cur, 1024, &[], &mut buf)
            {
                cur = next;
                self.table.merge_rows(&buf);
            } else {
                self.table.merge_rows(&buf);
                break;
            }
        }
    }

    /// Return the number of physical shards in this index.
    pub(crate) fn shard_count(&self) -> usize {
        self.table.shard_count()
    }

    /// Return the number of distinct keys stored in one physical shard.
    pub(crate) fn shard_len(&self, shard: usize) -> usize {
        self.table.shard_len(shard)
    }

    pub(crate) fn len(&self) -> usize {
        self.table.len()
    }
}

impl<TI: PositionedIndexBase> Index<TI> {
    /// Get a key's execution-stable position along with its nonempty subset.
    pub(crate) fn get_subset_positioned<'a>(
        &'a self,
        key: &TI::Key,
    ) -> Option<(IndexPosition, SubsetRef<'a>)> {
        self.table.get_subset_positioned(key)
    }

    /// Visit every key together with its execution-stable position.
    pub(crate) fn for_each_positioned<'a>(
        &'a self,
        f: impl FnMut(IndexPosition, &'a TI::Key, SubsetRef<'a>),
    ) {
        self.table.for_each_positioned(f);
    }

    /// Visit one physical shard together with each key's stable position.
    pub(crate) fn for_each_shard_positioned<'a>(
        &'a self,
        shard: usize,
        f: impl FnMut(IndexPosition, &'a TI::Key, SubsetRef<'a>),
    ) {
        self.table.for_each_shard_positioned(shard, f);
    }
}

pub(crate) struct SubsetTable {
    keys: RowBuffer,
    hash: Pooled<HashTable<TableEntry<BufferedSubset>>>,
}

impl Clone for SubsetTable {
    fn clone(&self) -> Self {
        SubsetTable {
            keys: self.keys.clone(),
            hash: Pooled::cloned(&self.hash),
        }
    }
}

impl SubsetTable {
    fn new(key_arity: usize) -> SubsetTable {
        SubsetTable {
            keys: RowBuffer::new(key_arity),
            hash: with_pool_set(|ps| ps.get()),
        }
    }
}

pub(crate) trait IndexBase {
    /// The type of keys for this index.  Keys can have validity constraints
    /// (e.g. the arity of a slice for `Key = [Value]`). If keys are invalid,
    /// these methods can panic.
    type Key: ?Sized;

    /// The write-side keys for an index. This is generally the same as `Key`, but Column-level
    /// indexes allow for multiple values (e.g. a subset of a row) to be provided, allowing the
    /// index to effectively cover multiple columns. This is useful for rebuilding.
    type WriteKey: ?Sized;

    /// Remove any existing entries in the index.
    fn clear(&mut self);
    /// Get the subset corresponding to this key, if there is one.
    fn get_subset(&self, key: &Self::Key) -> Option<SubsetRef<'_>>;
    /// Add the given key and row id to the table.
    fn add_row(&mut self, key: &Self::WriteKey, row: RowId);
    /// Merge the contents of the [`TaggedRowBuffer`] into the table.
    fn merge_rows(&mut self, buf: &TaggedRowBuffer);
    /// The number of physical shards in the index.
    fn shard_count(&self) -> usize;
    /// The number of distinct keys in one physical shard.
    fn shard_len(&self, shard: usize) -> usize;
    /// The number of keys in the index.
    fn len(&self) -> usize;

    fn merge_parallel(&mut self, cols: &[ColumnId], table: WrappedTableRef, subset: SubsetRef);

    /// Bulk-rebuild this index from scratch (called on major version change after clear()).
    /// The default implementation batches via `scan_project`+`merge_rows`. Implementations
    /// can override this for more efficient bulk construction.
    fn rebuild_full(&mut self, cols: &[ColumnId], table: WrappedTableRef, subset: SubsetRef) {
        let mut buf = TaggedRowBuffer::new(cols.len());
        let mut cur = Offset::new(0);
        loop {
            buf.clear();
            if let Some(next) = table.scan_project(subset, cols, cur, 1024, &[], &mut buf) {
                cur = next;
                self.merge_rows(&buf);
            } else {
                self.merge_rows(&buf);
                break;
            }
        }
    }
}

/// Read-only index operations that also expose a stable per-index key position.
///
/// Implementations may choose different physical slot schemes. Positions only
/// need to remain stable while the index is frozen; callers must treat them as
/// opaque tokens belonging to the exact index that returned them.
pub(crate) trait PositionedIndexBase: IndexBase {
    fn get_subset_positioned<'a>(
        &'a self,
        key: &Self::Key,
    ) -> Option<(IndexPosition, SubsetRef<'a>)>;

    fn for_each_positioned<'a>(
        &'a self,
        f: impl FnMut(IndexPosition, &'a Self::Key, SubsetRef<'a>),
    );

    fn for_each_shard_positioned<'a>(
        &'a self,
        shard: usize,
        f: impl FnMut(IndexPosition, &'a Self::Key, SubsetRef<'a>),
    );
}

struct ColumnIndexShard {
    /// It's important that table is implemented using IndexMap instead of the more efficient
    /// HashMap because we want stable enumeration order.
    table: Pooled<IndexMap<Value, BufferedSubset>>,
    subsets: SubsetBuffer,
}

impl Clone for ColumnIndexShard {
    fn clone(&self) -> Self {
        ColumnIndexShard {
            table: Pooled::cloned(&self.table),
            subsets: self.subsets.clone(),
        }
    }
}

#[derive(Clone)]
pub struct ColumnIndex {
    // A specialized index used when we are indexing on a single column.
    shard_data: ShardData,
    shards: IdVec<ShardId, ColumnIndexShard>,
}

impl IndexBase for ColumnIndex {
    type Key = Value;
    type WriteKey = [Value];
    fn clear(&mut self) {
        for (_, shard) in self.shards.iter_mut() {
            for (_, subset) in shard.table.drain(..) {
                match subset {
                    BufferedSubset::Dense(_) => {}
                    BufferedSubset::Sparse(buffered_vec) => {
                        shard.subsets.return_vec(buffered_vec);
                    }
                }
            }
        }
    }

    fn get_subset<'a>(&'a self, key: &Value) -> Option<SubsetRef<'a>> {
        let shard = self.shard_data.get_shard(key, &self.shards);
        shard.table.get(key).map(|x| x.as_ref(&shard.subsets))
    }
    fn add_row(&mut self, vals: &[Value], row: RowId) {
        for (i, key) in vals.iter().enumerate() {
            // A value repeated across this row's covered columns maps the row in only once.
            if vals[..i].contains(key) {
                continue;
            }
            let shard = self.shard_data.get_shard_mut(key, &mut self.shards);
            // SAFETY: everything in `table` comes from `subsets`.
            unsafe {
                shard
                    .table
                    .entry(*key)
                    .or_insert_with(BufferedSubset::empty)
                    .add_row_sorted(row, &mut shard.subsets);
            }
        }
    }
    fn merge_rows(&mut self, buf: &TaggedRowBuffer) {
        for (src_id, key) in buf.iter() {
            self.add_row(key, src_id);
        }
    }

    fn shard_count(&self) -> usize {
        self.shards.len()
    }

    fn shard_len(&self, shard: usize) -> usize {
        self.shards[ShardId::from_usize(shard)].table.len()
    }

    fn len(&self) -> usize {
        self.shards.iter().map(|(_, shard)| shard.table.len()).sum()
    }

    fn merge_parallel(&mut self, cols: &[ColumnId], table: WrappedTableRef, subset: SubsetRef) {
        const BATCH_SIZE: usize = 1024;
        let shard_data = self.shard_data;
        let mut queues = IdVec::<ShardId, Mutex<Vec<(RowId, TaggedRowBuffer)>>>::with_capacity(
            shard_data.n_shards(),
        );
        queues.resize_with(shard_data.n_shards(), || {
            Mutex::new(Vec::with_capacity((subset.size() / BATCH_SIZE) + 1))
        });
        let split_buf = |buf: TaggedRowBuffer| {
            let mut split = IdVec::<ShardId, TaggedRowBuffer>::default();
            split.resize_with(shard_data.n_shards(), || TaggedRowBuffer::new(1));
            for (row_id, keys) in buf.iter() {
                for (i, key) in keys.iter().enumerate() {
                    // Match `add_row`: a value repeated across this row's covered columns is
                    // recorded once, so a value's subset never holds a duplicate row id.
                    if keys[..i].contains(key) {
                        continue;
                    }
                    shard_data
                        .get_shard_mut(*key, &mut split)
                        .add_row(row_id, &[*key]);
                }
            }
            for (shard_id, buf) in split.drain() {
                if buf.is_empty() {
                    continue;
                }
                let first = buf.get_row(RowId::new(0)).0;
                queues[shard_id].lock().unwrap().push((first, buf));
            }
        };

        run_in_index_thread_pool(|| {
            egglog_concurrency::scope(|inner| {
                let mut cur = Offset::new(0);
                loop {
                    let mut buf = TaggedRowBuffer::new(cols.len());
                    if let Some(next) =
                        table.scan_project(subset, cols, cur, BATCH_SIZE, &[], &mut buf)
                    {
                        cur = next;
                        inner.spawn(move |_| split_buf(buf));
                    } else {
                        inner.spawn(move |_| split_buf(buf));
                        break;
                    }
                }
            });

            parallel::for_each_id_vec_mut(&mut self.shards, |shard_id, shard| {
                // Sort the vector by start row id to ensure we populate subsets in sorted order.
                let mut vec = queues[shard_id].lock().unwrap();
                vec.sort_by_key(|(start, _)| *start);
                for (_, buf) in vec.drain(..) {
                    for (row_id, key) in buf.iter() {
                        debug_assert_eq!(key.len(), 1);
                        match shard.table.entry(key[0]) {
                            Entry::Occupied(mut occ) => {
                                // SAFETY: all of the buffered vectors in this map come from `subsets`.
                                unsafe {
                                    occ.get_mut().add_row_sorted(row_id, &mut shard.subsets);
                                }
                            }
                            Entry::Vacant(v) => {
                                v.insert(BufferedSubset::singleton(row_id));
                            }
                        }
                    }
                }
            });
        });
    }

    /// Sort-based full rebuild: collect all (value, row_id) pairs, sort by (value, row_id),
    /// then build each key's subset with a single pre-sized allocation. Compared to `merge_rows`,
    /// this eliminates the doubling memmoves from `push_vec` that occur in the row-at-a-time `add_row` path.
    ///
    /// Supports multiple columns (e.g. rebuild_index covering all value columns): each value
    /// maps to the union of rows containing it in any of the covered columns.
    fn rebuild_full(&mut self, cols: &[ColumnId], table: WrappedTableRef, subset: SubsetRef) {
        // Collect each column into its own contiguous block, still in RowId-ascending scan
        // order. `bounds[b]..bounds[b + 1]` delimits column `b`'s block; the number of columns
        // is tiny, so it stays inline.
        let rows = subset.size();
        let mut pairs: Vec<(Value, RowId)> = Vec::with_capacity(rows * cols.len());
        let mut bounds: SmallVec<[usize; 8]> = SmallVec::new();
        bounds.push(0);
        for &col in cols {
            table.collect_col_pairs(subset, col, &mut pairs);
            bounds.push(pairs.len());
        }

        // Value-only sort each block. Since each block arrives RowId-ascending and the sort is
        // stable, the block ends up ordered by (Value, RowId) without any RowId pass.
        let mut scratch: Vec<(Value, RowId)> =
            vec![(Value::new_const(0), RowId::new_const(0)); rows];
        for b in 0..cols.len() {
            radix_sort_slice_by_value(&mut pairs[bounds[b]..bounds[b + 1]], &mut scratch);
        }

        if cols.len() == 1 {
            // A single column needs no merge: its block is already (Value, RowId)-sorted, and a
            // row has one value per column so there are no duplicates.
            self.build_subsets_from_sorted(&pairs);
            return;
        }

        // Multiple columns: merge the sorted blocks with a balanced (tournament) two-way merge.
        // Each merge drops duplicate (Value, RowId) pairs -- a value appearing in several of a
        // row's columns -- and dedup composes through the tree, so the result is
        // (Value, RowId)-sorted and unique without ever sorting by RowId. Halving the number of
        // runs each round makes this O(n log k) rather than the O(n*k) of a left fold (whose
        // growing accumulator is re-copied every step), which matters for wide tables.
        let merged = merge_sorted_blocks_dedup(pairs, &bounds);
        self.build_subsets_from_sorted(&merged);
    }
}

impl PositionedIndexBase for ColumnIndex {
    fn get_subset_positioned<'a>(&'a self, key: &Value) -> Option<(IndexPosition, SubsetRef<'a>)> {
        let mut hasher = FxHasher::default();
        key.hash(&mut hasher);
        let shard_id = self.shard_data.shard_id(hasher.finish());
        let shard = &self.shards[shard_id];
        let (slot, _, subset) = shard.table.get_full(key)?;
        Some((
            IndexPosition::new(shard_id, slot),
            subset.as_ref(&shard.subsets),
        ))
    }

    fn for_each_positioned<'a>(
        &'a self,
        mut f: impl FnMut(IndexPosition, &'a Value, SubsetRef<'a>),
    ) {
        for (shard_id, shard) in self.shards.iter() {
            for (slot, (key, subset)) in shard.table.iter().enumerate() {
                f(
                    IndexPosition::new(shard_id, slot),
                    key,
                    subset.as_ref(&shard.subsets),
                );
            }
        }
    }

    fn for_each_shard_positioned<'a>(
        &'a self,
        shard: usize,
        mut f: impl FnMut(IndexPosition, &'a Value, SubsetRef<'a>),
    ) {
        let shard_id = ShardId::from_usize(shard);
        let shard = &self.shards[shard_id];
        for (slot, (key, subset)) in shard.table.iter().enumerate() {
            f(
                IndexPosition::new(shard_id, slot),
                key,
                subset.as_ref(&shard.subsets),
            );
        }
    }
}

/// Number of 8-bit radix passes needed to cover values up to `max`.
fn radix_passes_for(max: u32) -> u32 {
    if max < 256 {
        1
    } else if max < 65_536 {
        2
    } else if max < (1 << 24) {
        3
    } else {
        4
    }
}

/// Adaptive value-only LSB radix sort of a single (Value, RowId) block, in place.
///
/// `scratch` must be at least `data.len()` long; it is used as ping-pong space. Because the
/// sort is stable and `data` arrives in RowId-ascending order, the result is ordered by
/// (Value, RowId). The multi-column rebuild path sorts each column's block this way before
/// merging, so no explicit RowId sort is ever needed.
pub(crate) fn radix_sort_slice_by_value(
    data: &mut [(Value, RowId)],
    scratch: &mut [(Value, RowId)],
) {
    let n = data.len();
    if n < 64 {
        data.sort_unstable();
        return;
    }

    // One scan computes the pass count and detects already-sorted input (common
    // when a column correlates with row order); a sorted block needs no work,
    // since stability makes the result identical.
    let mut max_val = 0u32;
    let mut sorted = true;
    let mut prev = 0u32;
    for &(v, _) in data.iter() {
        let rep = v.rep();
        max_val = max_val.max(rep);
        sorted &= prev <= rep;
        prev = rep;
    }
    if sorted {
        return;
    }
    let n_passes = radix_passes_for(max_val);

    let mut src: &mut [(Value, RowId)] = data;
    let mut dst: &mut [(Value, RowId)] = &mut scratch[..n];

    for pass in 0..n_passes {
        let shift = pass * 8;
        let mut count = [0u32; 256];

        // Count occurrences of the relevant byte of each Value.
        for pair in src.iter() {
            let bucket = (pair.0.rep() >> shift) & 0xFF;
            count[bucket as usize] += 1;
        }

        // Convert counts to exclusive prefix sums (start positions per bucket).
        let mut prefix = 0u32;
        for c in &mut count {
            let prev = *c;
            *c = prefix;
            prefix += prev;
        }

        // Stable scatter: write each element to its bucket's next position.
        for &pair in src.iter() {
            let bucket = ((pair.0.rep() >> shift) & 0xFF) as usize;
            dst[count[bucket] as usize] = pair;
            count[bucket] += 1;
        }

        core::mem::swap(&mut src, &mut dst);
    }

    // After `n_passes` swaps, `src` points to the sorted data. If odd, that is `scratch`;
    // copy it back into `data` (which is now `dst`).
    if n_passes % 2 == 1 {
        dst.copy_from_slice(src);
    }
}

/// Merge two (Value, RowId)-sorted slices, *appending* the result to `out` and dropping pairs
/// equal to the previous one emitted *by this call*.
///
/// Inputs `a` and `b` must each be sorted by (Value, RowId). Duplicates arise when one value
/// appears in several of a row's columns; the dedup check is scoped to this call's output (via
/// `start`) so back-to-back runs packed into the same buffer are not merged into each other.
fn merge2_into(a: &[(Value, RowId)], b: &[(Value, RowId)], out: &mut Vec<(Value, RowId)>) {
    let start = out.len();
    let push = |out: &mut Vec<(Value, RowId)>, next: (Value, RowId)| {
        if out.len() == start || *out.last().unwrap() != next {
            out.push(next);
        }
    };
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] <= b[j] {
            push(out, a[i]);
            i += 1;
        } else {
            push(out, b[j]);
            j += 1;
        }
    }
    for &next in &a[i..] {
        push(out, next);
    }
    for &next in &b[j..] {
        push(out, next);
    }
}

/// Merge the `(Value, RowId)`-sorted column blocks of `src` (block `b` is
/// `src[bounds[b]..bounds[b + 1]]`) into one sorted, de-duplicated vector.
///
/// Uses a balanced (tournament) two-way merge: adjacent runs are merged pairwise, then the
/// results are merged pairwise, halving the run count each round. This is O(n log k) in the
/// number of blocks `k`, versus the O(n*k) of merging a single growing accumulator against
/// each block in turn -- the difference matters when a table has many covered columns.
///
/// Each round packs its merged runs contiguously into a second buffer of the same size and the
/// two buffers ping-pong, so the whole tournament uses just one extra allocation (`src` is
/// reused as the other buffer) rather than a fresh `Vec` per merge. Because merging two
/// de-duplicated sorted runs leaves any shared pair adjacent, dedup composes across rounds.
fn merge_sorted_blocks_dedup(
    mut src: Vec<(Value, RowId)>,
    bounds: &[usize],
) -> Vec<(Value, RowId)> {
    let n = src.len();
    debug_assert!(bounds.len() >= 2);

    // `src` holds the current rounds's runs, delimited by `cur`; `dst` receives the merged runs.
    let mut dst: Vec<(Value, RowId)> = Vec::with_capacity(n);
    let mut cur: SmallVec<[usize; 8]> = bounds.iter().copied().collect();

    // Each round more than halves the run count (`cur.len() - 1`); stop at a single run.
    while cur.len() > 2 {
        dst.clear();
        let mut next: SmallVec<[usize; 8]> = SmallVec::new();
        next.push(0);
        let runs = cur.len() - 1;
        let mut r = 0;
        while r < runs {
            if r + 1 < runs {
                merge2_into(
                    &src[cur[r]..cur[r + 1]],
                    &src[cur[r + 1]..cur[r + 2]],
                    &mut dst,
                );
                r += 2;
            } else {
                // Odd trailing run: already sorted and de-duplicated, so copy it forward.
                dst.extend_from_slice(&src[cur[r]..cur[r + 1]]);
                r += 1;
            }
            next.push(dst.len());
        }
        mem::swap(&mut src, &mut dst);
        cur = next;
    }

    // One run remains, packed at the front of `src`.
    src.truncate(cur[1]);
    src
}

impl ColumnIndex {
    pub(crate) fn new() -> ColumnIndex {
        with_pool_set(|ps| {
            let shard_data = ShardData::new(num_shards());
            let mut shards = IdVec::with_capacity(shard_data.n_shards());
            shards.resize_with(shard_data.n_shards(), || ColumnIndexShard {
                table: ps.get(),
                subsets: SubsetBuffer::default(),
            });
            ColumnIndex { shard_data, shards }
        })
    }

    /// Build each key's subset from `pairs`, which must be sorted by (Value, RowId) and
    /// free of duplicate (Value, RowId) entries. Each contiguous run of equal values
    /// becomes one subset, pre-sized from the run length.
    fn build_subsets_from_sorted(&mut self, pairs: &[(Value, RowId)]) {
        let mut i = 0;
        while i < pairs.len() {
            let key = pairs[i].0;
            let start = i;
            let mut first = pairs[i].1;
            let mut last = pairs[i].1;
            while i < pairs.len() && pairs[i].0 == key {
                last = cmp::max(last, pairs[i].1);
                first = cmp::min(first, pairs[i].1);
                i += 1;
            }
            let shard = self.shard_data.get_shard_mut(key, &mut self.shards);
            let count = i - start;
            let buffered = if last.rep() - first.rep() == (count - 1) as u32 {
                // If the row ids are contiguous, we can represent the subset as a dense range
                // to avoid allocations
                BufferedSubset::Dense(OffsetRange::new(first, last.inc()))
            } else {
                let bv = shard
                    .subsets
                    .new_vec(pairs[start..i].iter().map(|&(_, r)| r));
                BufferedSubset::Sparse(bv)
            };
            shard.table.insert(key, buffered);
        }
    }
}

#[derive(Clone)]
struct TupleIndexShard {
    table: SubsetTable,
    subsets: SubsetBuffer,
}

/// A mapping from keys to subsets of rows.
#[derive(Clone)]
pub struct TupleIndex {
    // NB: we could store RowBuffers inline and then have indexes reference
    // (u32, RowId) instead of RowId. Trades copying off for indirections.
    shard_data: ShardData,
    shards: IdVec<ShardId, TupleIndexShard>,
}

impl TupleIndex {
    pub(crate) fn new(key_arity: usize) -> TupleIndex {
        let shard_data = ShardData::new(num_shards());
        let mut shards = IdVec::with_capacity(shard_data.n_shards());
        shards.resize_with(shard_data.n_shards(), || TupleIndexShard {
            table: SubsetTable::new(key_arity),
            subsets: SubsetBuffer::default(),
        });
        TupleIndex { shard_data, shards }
    }
}

impl IndexBase for TupleIndex {
    type Key = [Value];
    type WriteKey = Self::Key;

    fn clear(&mut self) {
        for (_, shard) in self.shards.iter_mut() {
            shard.table.keys.clear();
            for entry in shard.table.hash.drain() {
                match entry.vals {
                    BufferedSubset::Dense(_) => {}
                    BufferedSubset::Sparse(v) => {
                        shard.subsets.return_vec(v);
                    }
                }
            }
        }
    }

    fn get_subset<'a>(&'a self, key: &[Value]) -> Option<SubsetRef<'a>> {
        let hash = hash_key(key);
        let shard = &self.shards[self.shard_data.shard_id(hash)];
        let entry = shard.table.hash.find(hash, |entry| {
            // SAFETY: entry.key was stored by add_row, which returns a valid RowId.
            entry.hash == hash && unsafe { shard.table.keys.get_row_unchecked(entry.key) } == key
        })?;
        Some(entry.vals.as_ref(&shard.subsets))
    }

    fn add_row(&mut self, key: &[Value], row: RowId) {
        use hashbrown::hash_table::Entry;
        let hash = hash_key(key);
        let shard = &mut self.shards[self.shard_data.shard_id(hash)];
        let table_entry = shard.table.hash.entry(
            hash,
            // SAFETY: entry.key was stored by add_row, which returns a valid RowId.
            |entry| {
                entry.hash == hash
                    && unsafe { shard.table.keys.get_row_unchecked(entry.key) } == key
            },
            |ent| ent.hash,
        );
        match table_entry {
            Entry::Occupied(mut occ) => {
                // SAFETY: everything in `table_entry` comes from `vals`.
                unsafe {
                    occ.get_mut().vals.add_row_sorted(row, &mut shard.subsets);
                }
            }
            Entry::Vacant(v) => {
                let key_id = shard.table.keys.add_row(key);
                let subset = BufferedSubset::singleton(row);
                v.insert(TableEntry {
                    hash,
                    key: key_id,
                    vals: subset,
                });
            }
        }
    }

    fn merge_rows(&mut self, buf: &TaggedRowBuffer) {
        for (src_id, key) in buf.iter() {
            self.add_row(key, src_id);
        }
    }
    fn shard_count(&self) -> usize {
        self.shards.len()
    }

    fn shard_len(&self, shard: usize) -> usize {
        self.shards[ShardId::from_usize(shard)].table.hash.len()
    }

    fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|(_, shard)| shard.table.hash.len())
            .sum()
    }

    fn merge_parallel(&mut self, cols: &[ColumnId], table: WrappedTableRef, subset: SubsetRef) {
        // The structure here is similar to the implementation for ColumnIndex, with
        // slightly more bookkeeping needed to handle arbitrary-arity keys.

        const BATCH_SIZE: usize = 1024;
        let shard_data = self.shard_data;
        let mut queues = IdVec::<ShardId, Mutex<Vec<(RowId, TaggedRowBuffer)>>>::with_capacity(
            shard_data.n_shards(),
        );
        queues.resize_with(shard_data.n_shards(), || {
            Mutex::new(Vec::with_capacity((subset.size() / BATCH_SIZE) + 1))
        });
        let split_buf = |buf: TaggedRowBuffer| {
            let mut split = IdVec::<ShardId, TaggedRowBuffer>::default();
            split.resize_with(shard_data.n_shards(), || TaggedRowBuffer::new(cols.len()));
            for (row_id, key) in buf.iter() {
                shard_data
                    .get_shard_mut(key, &mut split)
                    .add_row(row_id, key);
            }
            for (shard_id, buf) in split.drain() {
                if buf.is_empty() {
                    continue;
                }
                let first = buf.get_row(RowId::new(0)).0;
                queues[shard_id].lock().unwrap().push((first, buf));
            }
        };
        run_in_index_thread_pool(|| {
            egglog_concurrency::scope(|scope| {
                let mut cur = Offset::new(0);
                loop {
                    let mut buf = TaggedRowBuffer::new(cols.len());
                    if let Some(next) =
                        table.scan_project(subset, cols, cur, BATCH_SIZE, &[], &mut buf)
                    {
                        cur = next;
                        scope.spawn(move |_| split_buf(buf));
                    } else {
                        scope.spawn(move |_| split_buf(buf));
                        break;
                    }
                }
            });
            parallel::for_each_id_vec_mut(&mut self.shards, |shard_id, shard| {
                use hashbrown::hash_table::Entry;
                // Sort the vector by start row id to ensure we populate subsets in sorted order.
                let mut vec = queues[shard_id].lock().unwrap();
                vec.sort_by_key(|(start, _)| *start);
                for (_, buf) in vec.drain(..) {
                    for (row_id, key) in buf.iter() {
                        let hash = hash_key(key);
                        let table_entry = shard.table.hash.entry(
                            hash,
                            // SAFETY: entry.key was stored by add_row, which returns a valid RowId.
                            |entry| {
                                entry.hash == hash
                                    && unsafe { shard.table.keys.get_row_unchecked(entry.key) }
                                        == key
                            },
                            |ent| ent.hash,
                        );
                        match table_entry {
                            Entry::Occupied(mut occ) => {
                                // SAFETY: everything in `table_entry` comes from `vals`.
                                unsafe {
                                    occ.get_mut()
                                        .vals
                                        .add_row_sorted(row_id, &mut shard.subsets);
                                }
                            }
                            Entry::Vacant(v) => {
                                let key_id = shard.table.keys.add_row(key);
                                let subset = BufferedSubset::singleton(row_id);
                                v.insert(TableEntry {
                                    hash,
                                    key: key_id,
                                    vals: subset,
                                });
                            }
                        }
                    }
                }
            });
        });
    }
}

impl PositionedIndexBase for TupleIndex {
    fn get_subset_positioned<'a>(
        &'a self,
        key: &[Value],
    ) -> Option<(IndexPosition, SubsetRef<'a>)> {
        let hash = hash_key(key);
        let shard_id = self.shard_data.shard_id(hash);
        let shard = &self.shards[shard_id];
        let entry = shard.table.hash.find(hash, |entry| {
            // SAFETY: entry.key was stored by add_row, which returns a valid RowId.
            entry.hash == hash && unsafe { shard.table.keys.get_row_unchecked(entry.key) } == key
        })?;
        Some((
            IndexPosition::new(shard_id, entry.key.index()),
            entry.vals.as_ref(&shard.subsets),
        ))
    }

    fn for_each_positioned<'a>(
        &'a self,
        mut f: impl FnMut(IndexPosition, &'a [Value], SubsetRef<'a>),
    ) {
        for (shard_id, shard) in self.shards.iter() {
            for entry in shard.table.hash.iter() {
                // SAFETY: entry.key was stored by add_row, so it is always in-bounds.
                let key = unsafe { shard.table.keys.get_row_unchecked(entry.key) };
                f(
                    IndexPosition::new(shard_id, entry.key.index()),
                    key,
                    entry.vals.as_ref(&shard.subsets),
                );
            }
        }
    }

    fn for_each_shard_positioned<'a>(
        &'a self,
        shard: usize,
        mut f: impl FnMut(IndexPosition, &'a [Value], SubsetRef<'a>),
    ) {
        let shard_id = ShardId::from_usize(shard);
        let shard = &self.shards[shard_id];
        for entry in shard.table.hash.iter() {
            // SAFETY: entry.key was stored by add_row, so it is always in-bounds.
            let key = unsafe { shard.table.keys.get_row_unchecked(entry.key) };
            f(
                IndexPosition::new(shard_id, entry.key.index()),
                key,
                entry.vals.as_ref(&shard.subsets),
            );
        }
    }
}

fn hash_key(key: &[Value]) -> u64 {
    let mut hasher = FxHasher::default();
    key.hash(&mut hasher);
    hasher.finish()
}

/// A map from access patterns to indices.
///
/// Implemented as an read-optimized key-value arrays, which should be faster
/// than concurrent hashmaps as long as # indices is smaller than say 64.
///
/// For simplicity we assume the index can be cloned cheaply, e.g., it's behind an [`Arc`].
#[derive(Default)]
pub struct IndexCatalog<K: Clone + std::hash::Hash + Eq, I: Clone> {
    data: ReadOptimizedLock<Vec<(K, I)>>,
    #[cfg(test)]
    get_or_insert_calls: AtomicUsize,
}

impl<K, I: Clone> IndexCatalog<K, I>
where
    K: Clone + std::hash::Hash + Eq,
{
    pub fn new() -> Self {
        IndexCatalog {
            data: ReadOptimizedLock::new(Vec::new()),
            #[cfg(test)]
            get_or_insert_calls: AtomicUsize::new(0),
        }
    }

    pub fn map(&self, f: impl Fn(&(K, I)) -> (K, I)) -> Self {
        let vec = self.data.read().iter().map(f).collect();
        IndexCatalog {
            data: ReadOptimizedLock::new(vec),
            #[cfg(test)]
            get_or_insert_calls: AtomicUsize::new(0),
        }
    }

    pub fn update(&mut self, f: impl Fn(&K, &mut I)) {
        for (k, i) in self.data.as_mut_ref() {
            f(k, i)
        }
    }

    pub fn get_or_insert(&self, k: K, init: impl FnOnce() -> I) -> I {
        #[cfg(test)]
        self.get_or_insert_calls.fetch_add(1, Ordering::Relaxed);
        let data = self.data.read();
        let entry = data.iter().find(|(k1, _)| k1 == &k);
        if let Some(entry) = entry {
            entry.1.clone()
        } else {
            drop(data);
            let mut data = self.data.lock();
            if let Some(entry) = data.iter().find(|(k1, _)| k1 == &k) {
                entry.1.clone()
            } else {
                let index = init();
                data.push((k, index.clone()));
                index
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn get_or_insert_calls(&self) -> usize {
        self.get_or_insert_calls.load(Ordering::Relaxed)
    }
}

define_id!(BufferIndex, u32, "an index into a subset buffer");

/// A shared pool of row ids used to store sorted offset vectors with a common
/// lifetime.
///
/// This is used as the backing store for subsets stored in indexes. While
/// definitely saves some allocations, the primary use for SubsetBuffer is to
/// make deallocation faster: with a standard [`crate::offsets::Subset`]
/// structure stored in the index, dropping requires an O(n) traversal of the
/// index. SubsetBuffer allows deallocation to happen in constant time (given
/// our use of memory pools).
struct SubsetBuffer {
    buf: Pooled<Vec<RowId>>,
    free_list: FreeList,
}

impl Clone for SubsetBuffer {
    fn clone(&self) -> Self {
        SubsetBuffer {
            buf: Pooled::cloned(&self.buf),
            free_list: self.free_list.clone(),
        }
    }
}

impl Default for SubsetBuffer {
    fn default() -> SubsetBuffer {
        with_pool_set(|ps| SubsetBuffer {
            buf: ps.get(),
            free_list: Default::default(),
        })
    }
}

impl SubsetBuffer {
    fn new_vec(&mut self, rows: impl ExactSizeIterator<Item = RowId>) -> BufferedVec {
        let len = rows.len();
        if let Some(v) = self.free_list.get_size_class(len).pop() {
            return self.fill_at(v, rows);
        }
        let start = BufferIndex::from_usize(self.buf.len());
        self.buf.resize(
            start.index() + len.next_power_of_two(),
            RowId::new(u32::MAX),
        );
        self.fill_at(start, rows)
    }

    fn fill_at(
        &mut self,
        start: BufferIndex,
        rows: impl ExactSizeIterator<Item = RowId>,
    ) -> BufferedVec {
        let mut cur = start;
        for i in rows {
            self.buf[cur.index()] = i;
            cur = cur.inc();
        }
        BufferedVec(start, cur)
    }

    fn return_vec(&mut self, vec: BufferedVec) {
        self.free_list.get_size_class(vec.len()).push(vec.0);
    }

    fn push_vec(&mut self, vec: BufferedVec, row: RowId) -> BufferedVec {
        debug_assert!(
            vec.is_empty() || self.buf[vec.1.index() - 1] <= row,
            "vec={vec:?}, row={row:?}, last_elt={:?}",
            self.buf[vec.1.index() - 1]
        );
        if !vec.len().is_power_of_two() {
            self.buf[vec.1.index()] = row;
            return BufferedVec(vec.0, vec.1.inc());
        }

        let res = if let Some(v) = self.free_list.get_size_class(vec.len() + 1).pop() {
            self.buf
                .copy_within(vec.0.index()..vec.1.index(), v.index());
            self.buf[v.index() + vec.len()] = row;
            BufferedVec(v, BufferIndex::from_usize(v.index() + vec.len() + 1))
        } else {
            let start = self.buf.len();
            self.buf.resize(
                start + (vec.len() + 1).next_power_of_two(),
                RowId::new(u32::MAX),
            );
            self.buf.copy_within(vec.0.index()..vec.1.index(), start);
            self.buf[start + vec.len()] = row;
            let end = start + vec.len() + 1;
            BufferedVec(BufferIndex::from_usize(start), BufferIndex::from_usize(end))
        };
        self.return_vec(vec);
        res
    }

    fn make_ref<'a>(&'a self, vec: &BufferedVec) -> SubsetRef<'a> {
        // SAFETY: if `vec` is a valid index into self.buf, it will be sorted.
        //
        // NB: we do not guarantee this in the type signature of BufferedVec,
        // etc. But this is indeed safe given the usage within this module.
        let res = SubsetRef::Sparse(unsafe {
            SortedOffsetSlice::new_unchecked(&self.buf[vec.0.index()..vec.1.index()])
        });
        #[cfg(debug_assertions)]
        {
            use crate::offsets::Offsets;
            res.offsets(|x| assert_ne!(x.rep(), u32::MAX))
        }
        res
    }
}

/// A sorted vector of offsets stored in a [`SubsetBuffer`].
///
/// Note: this implements `Clone` to facilitate cloning entire indexes, but this is a _shallow_
/// clone, making the clone operation work akin to slices in Golang. In particular: code that
/// pushes to a clone of a `BufferedVec` can affect the original, and vice versa.
///
/// Business logic in this module probably shouldn't call clone explicitly. The implicit uses of
/// clone (by other generated `Clone` implementations) are fine because they clone the
/// `SubsetBuffer` that the `BufferedVec` points to at the same time that the vector is cloned.
#[derive(Debug, Clone)]
pub(crate) struct BufferedVec(BufferIndex, BufferIndex);

impl Default for BufferedVec {
    fn default() -> Self {
        BufferedVec(BufferIndex::new(0), BufferIndex::new(0))
    }
}

impl BufferedVec {
    fn is_empty(&self) -> bool {
        self.0 == self.1
    }
    fn len(&self) -> usize {
        self.1.index() - self.0.index()
    }
}

#[derive(Clone)]
pub(crate) enum BufferedSubset {
    Dense(OffsetRange),
    Sparse(BufferedVec),
}

impl BufferedSubset {
    /// *Safety:*  callers must ensure that `self` is either dense, or comes from `buf`.
    unsafe fn add_row_sorted(&mut self, row: RowId, buf: &mut SubsetBuffer) {
        match self {
            BufferedSubset::Dense(range) => {
                if range.end == range.start {
                    range.start = row;
                    range.end = row.inc();
                    return;
                }
                if range.end == row {
                    range.end = row.inc();
                    return;
                }
                let mut v = buf.new_vec((range.start.rep()..range.end.rep()).map(RowId::new));
                v = buf.push_vec(v, row);
                *self = BufferedSubset::Sparse(v);
            }
            BufferedSubset::Sparse(vec) => *vec = buf.push_vec(mem::take(vec), row),
        }
    }

    fn empty() -> Self {
        BufferedSubset::Dense(OffsetRange::new(RowId::new(0), RowId::new(0)))
    }

    fn singleton(row: RowId) -> Self {
        BufferedSubset::Dense(OffsetRange::new(row, row.inc()))
    }

    fn as_ref<'a>(&self, buf: &'a SubsetBuffer) -> SubsetRef<'a> {
        match self {
            BufferedSubset::Dense(range) => SubsetRef::Dense(*range),
            BufferedSubset::Sparse(vec) => buf.make_ref(vec),
        }
    }
}

fn num_shards() -> usize {
    let n_threads = parallel::current_num_threads();
    if n_threads == 1 { 1 } else { n_threads * 2 }
}

/// A thread pool specifically for parallel hash index construction.
///
/// Callers can construct indexes while holding database-level index locks. The
/// separate pool preserves parallelism without tying up the caller's installed
/// pool behind those locks.
static INDEX_THREAD_POOL: Lazy<ThreadPool> =
    Lazy::new(|| ThreadPool::new(parallel::current_num_threads().max(1)));

fn run_in_index_thread_pool<R>(f: impl FnOnce() -> R) -> R {
    INDEX_THREAD_POOL.install(f)
}

/// A simple free list used to reuse slots in a [`SubsetBuffer`].
///
/// This free list works as a map from power-of-two size classes to a vector of offsets that point
/// to the beginning of an unused vector.
///
/// Size classes are indexed by their log2 value (i.e., size_class = 2^idx), so a 32-entry
/// array covers all power-of-two sizes from 1 (idx=0) up to 2^31. This replaces the
/// previous HashMap with an O(1) array index + trailing_zeros().
#[derive(Clone, Default)]
pub(super) struct FreeList {
    data: [Vec<BufferIndex>; 32],
}

impl FreeList {
    fn get_size_class(&mut self, size: usize) -> &mut Vec<BufferIndex> {
        let size_class = size.next_power_of_two();
        let idx = size_class.trailing_zeros() as usize;
        &mut self.data[idx]
    }
}
