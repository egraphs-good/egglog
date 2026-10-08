//! A variable-length-key table with a fixed-width value tail.
//!
//! `SequenceTable` keeps logical rows separate from their packed value
//! storage. A [`RowId`] indexes a dense vector of cumulative end offsets,
//! while hash entries cache both that `RowId` and the full backing-vector slice.
//! This preserves the existing `Subset` and version model without requiring a
//! fixed key width. The final `n_values` entries of every row are non-key
//! values, exactly as the columns after `n_keys` are in `SortedWritesTable`.
//! Hashing and equality use only the variable-length key prefix.
//!
//! Its private layers mirror the fixed-arity table path: `SequenceRows`
//! corresponds to `RowBuffer`, `ParallelSequenceWriter` to
//! `ParallelRowBufWriter`, `HashedSequenceBuffer` to `HashedRowBuffer`, and
//! `SequenceBatch` to `CoalescedInsertBatch`. `SequenceBuffer` and
//! `SequencePendingState` correspond to the fixed table's `Buffer` and
//! `PendingState`, with an extra detachable `SequenceEpoch`. Fixed arity
//! derives row ranges from a constant stride; the sequence variants carry
//! cumulative offsets and reserve row and value ranges together.

use std::{
    cell::{Cell, UnsafeCell},
    cmp, mem,
    sync::{
        Arc, RwLock, Weak,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use crate::numeric_id::{DenseIdMap, NumericId};
use crossbeam_queue::SegQueue;
use hashbrown::HashTable;

use crate::{
    ExecutionState, TableChange,
    common::{ShardData, ShardId, Value},
    hash_index::{ColumnIndex, Index, IndexBase},
    offsets::{OffsetRange, Offsets, RowId, Subset, SubsetRef},
    parallel,
    parallel_heuristics::{
        parallelize_index_construction, parallelize_rebuild, parallelize_table_op,
    },
    pool::{Pooled, with_pool_set},
    table_spec::{Generation, MutationBuffer, Offset, Row, TableVersion, ValueRebuilder},
};

use super::{
    CompactHash, HashedTableEntry, MergeFn, ShardedHashTable, drain_queue, shard_hash_values,
};

const PARALLEL_SEQUENCE_BATCH_ROWS: usize = 1 << 12;
const PARALLEL_SEQUENCE_BATCH_VALUES: usize = 1 << 16;
const REBUILD_CHUNK_ROWS: usize = 1 << 11;
const STALE_OFFSET_BIT: u32 = 1 << 31;
const OFFSET_MASK: u32 = STALE_OFFSET_BIT - 1;

/// Selects the semantic key values recorded in a [`SequenceTable`]'s
/// occurrence index.
///
/// Variable-length encodings commonly contain headers, counts, or bit masks
/// alongside actual e-class ids. The extractor receives only the key prefix
/// and calls `visit` for values that should map back to the containing row.
/// Repeated visits are harmless: the index records each `(value, row)` once.
pub type SequenceIndexFn = dyn Fn(&[Value], &mut dyn FnMut(Value)) + Send + Sync;

#[inline]
fn sequence_key(row: &[Value], n_values: usize) -> &[Value] {
    let key_len = row
        .len()
        .checked_sub(n_values)
        .unwrap_or_else(|| panic!("sequence row has fewer than {n_values} non-key values"));
    &row[..key_len]
}

#[inline]
fn sequence_values(row: &[Value], n_values: usize) -> &[Value] {
    let key_len = row
        .len()
        .checked_sub(n_values)
        .unwrap_or_else(|| panic!("sequence row has fewer than {n_values} non-key values"));
    &row[key_len..]
}

fn assert_valid_merge_output(output: &[Value], key: &[Value], n_values: usize) {
    assert_eq!(
        output.len(),
        key.len() + n_values,
        "sequence merge changed the fixed-width value tail"
    );
    assert_eq!(
        sequence_key(output, n_values),
        key,
        "sequence merge functions must preserve table keys"
    );
}

fn sequence_subset_partition(subset: SubsetRef<'_>, start: usize, end: usize) -> SubsetRef<'_> {
    debug_assert!(start <= end);
    debug_assert!(end <= subset.size());
    match subset {
        SubsetRef::Dense(range) => SubsetRef::Dense(OffsetRange::new(
            RowId::from_usize(range.start.index() + start),
            RowId::from_usize(range.start.index() + end),
        )),
        SubsetRef::Sparse(rows) => SubsetRef::Sparse(rows.subslice(start, end)),
    }
}

/// A half-open slice in the packed values vector.
///
/// Logical-row liveness is stored separately in the high bit of its cumulative
/// end offset, so every bit pattern in a `Value` remains valid row contents.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
struct SequenceSlice {
    start: u32,
    end: u32,
}

impl SequenceSlice {
    fn new(start: usize, end: usize) -> Self {
        assert!(start <= end, "sequence slice starts after it ends");
        assert!(
            end <= OFFSET_MASK as usize,
            "SequenceTable backing storage exceeds its u32 offset space"
        );
        Self {
            start: start as u32,
            end: end as u32,
        }
    }

    fn start(self) -> usize {
        self.start as usize
    }

    fn end(self) -> usize {
        self.end as usize
    }

    fn len(self) -> usize {
        self.end() - self.start()
    }
}

/// The physical id and packed-value range cached by a sequence hash entry.
///
/// A fixed-arity `TableEntry` needs only its `RowId`, because the table's
/// constant stride recovers the row's value range. Variable-width lookup would
/// otherwise require another offset-vector access during every hash probe, so
/// the sequence entry caches that range alongside the generation-scoped id.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct SequenceLocation {
    slice: SequenceSlice,
    row: RowId,
}

type SequenceEntry = HashedTableEntry<SequenceLocation>;

/// Variable-arity analogue of `SortedWritesTable::rebuild_index`.
///
/// Fixed tables describe rebuildable positions with a list of columns. A
/// sequence encoding instead supplies an extractor, which can skip metadata
/// and visit any number of semantic values per row. The underlying
/// [`ColumnIndex`] is shared with fixed tables and therefore maps each value to
/// a compact [`Subset`] of physical row ids.
#[derive(Clone)]
struct SequenceValueIndex {
    values: Arc<SequenceIndexFn>,
    rows: Index<ColumnIndex>,
}

impl SequenceValueIndex {
    fn new(values: Arc<SequenceIndexFn>) -> Self {
        Self {
            values,
            rows: Index::new(Vec::new(), ColumnIndex::new()),
        }
    }

    fn get(&self, value: &Value) -> Option<SubsetRef<'_>> {
        self.rows.get_subset(value)
    }

    fn clear_to(&mut self, version: TableVersion) {
        self.rows.clear_to(version);
    }

    fn refresh(&mut self, rows: &SequenceRows, n_values: usize, version: TableVersion) {
        let values = &self.values;
        self.rows.refresh_with(version, |index, start, full| {
            let start = start.index();
            let end = rows.physical_len();
            if start >= end {
                return;
            }

            let scan = SubsetRef::Dense(OffsetRange::new(
                RowId::from_usize(start),
                RowId::from_usize(end),
            ));
            let workload = scan.size().max(rows.packed_value_span(start, end));

            // Match the fixed rebuild index's low-overhead incremental path:
            // a small append can be folded directly into already sorted
            // subsets without allocating and sorting occurrence pairs.
            if !full && !parallelize_index_construction(workload) {
                let mut selected = with_pool_set(|pools| pools.get::<Vec<Value>>());
                scan.offsets(|row_id| {
                    let Some(row) = rows.get_row(row_id) else {
                        return;
                    };
                    values(sequence_key(row, n_values), &mut |value| {
                        selected.push(value)
                    });
                    index.add_row(&selected, row_id);
                    selected.clear();
                });
                return;
            }

            let chunk_rows = if parallelize_index_construction(workload) {
                scan.size()
                    .div_ceil(parallel::current_num_threads().saturating_mul(2).max(1))
                    .clamp(1, REBUILD_CHUNK_ROWS)
            } else {
                scan.size().max(1)
            };
            let starts = (0..scan.size()).step_by(chunk_rows).collect::<Vec<_>>();
            let chunks = parallel::map(&starts, |_, start| {
                let end = cmp::min(*start + chunk_rows, scan.size());
                let partition = sequence_subset_partition(scan, *start, end);
                let mut pairs = Vec::<(Value, RowId)>::new();
                partition.offsets(|row_id| {
                    let Some(row) = rows.get_row(row_id) else {
                        return;
                    };
                    let key = sequence_key(row, n_values);
                    values(key, &mut |value| pairs.push((value, row_id)));
                });
                pairs
            });
            index.merge_value_row_chunks(chunks);
        });
    }
}

/// Variable-arity counterpart of the fixed-arity `Rows`/`RowBuffer` store.
///
/// The fixed store derives row boundaries from a constant stride and keeps its
/// stale marker in row storage. Here packed values need a parallel cumulative
/// end-offset vector; the high bit of each end records liveness and therefore
/// also supports empty sequences.
///
/// The value allocation deliberately uses ordinary `Value`s rather than
/// `Cell<Value>`. Values never change after publication, so safe row slices
/// cannot alias an interior-mutable value. Only offset metadata uses `Cell`,
/// and shared stale marking is unsafe and restricted to a worker that owns the
/// unique hash shard containing that row.
#[derive(Default)]
struct SequenceRows {
    values: Pooled<Vec<Value>>,
    /// Cumulative end offsets. The high bit marks the corresponding row stale.
    offsets: Pooled<Vec<Cell<u32>>>,
    stale_rows: usize,
    stale_values: usize,
}

// `Cell` makes the type conservatively !Sync. Values are immutable after
// publication, and the only shared metadata mutation is guarded by the same
// disjoint-row requirement as RowBuffer::set_stale_shared.
unsafe impl Sync for SequenceRows {}

impl Clone for SequenceRows {
    fn clone(&self) -> Self {
        Self {
            values: Pooled::cloned(&self.values),
            offsets: Pooled::cloned(&self.offsets),
            stale_rows: self.stale_rows,
            stale_values: self.stale_values,
        }
    }
}

impl SequenceRows {
    fn clear(&mut self) {
        self.values.clear();
        self.offsets.clear();
        self.stale_rows = 0;
        self.stale_values = 0;
    }

    fn physical_len(&self) -> usize {
        self.offsets.len()
    }

    fn live_len(&self) -> usize {
        self.physical_len() - self.stale_rows
    }

    /// Number of packed value slots covered by a dense physical row range.
    /// Stale values are included, making this an O(1) upper bound for index
    /// construction work.
    fn packed_value_span(&self, start: usize, end: usize) -> usize {
        debug_assert!(start <= end);
        debug_assert!(end <= self.physical_len());
        let value_start = if start == 0 {
            0
        } else {
            (self.offsets[start - 1].get() & OFFSET_MASK) as usize
        };
        let value_end = if end == 0 {
            0
        } else {
            (self.offsets[end - 1].get() & OFFSET_MASK) as usize
        };
        value_end - value_start
    }

    fn next_row(&self) -> RowId {
        RowId::from_usize(self.physical_len())
    }

    fn reserve(&mut self, rows: usize, values: usize) {
        let final_rows = self
            .offsets
            .len()
            .checked_add(rows)
            .expect("SequenceTable row count overflow");
        let final_values = self
            .values
            .len()
            .checked_add(values)
            .expect("SequenceTable value count overflow");
        assert!(
            final_rows < u32::MAX as usize,
            "SequenceTable exceeds its RowId space"
        );
        assert!(
            final_values <= OFFSET_MASK as usize,
            "SequenceTable backing storage exceeds its u32 offset space"
        );
        self.offsets.reserve(rows);
        self.values.reserve(values);
    }

    /// Append after the caller has reserved enough row and value capacity.
    fn add_row_reserved(&mut self, values: &[Value]) -> SequenceLocation {
        debug_assert!(self.offsets.len() < self.offsets.capacity());
        debug_assert!(self.values.len() + values.len() <= self.values.capacity());
        let row = self.next_row();
        let start = self.values.len();
        self.values.extend_from_slice(values);
        let slice = SequenceSlice::new(start, self.values.len());
        self.offsets.push(Cell::new(slice.end));
        SequenceLocation { slice, row }
    }

    fn get_row(&self, row: RowId) -> Option<&[Value]> {
        self.get_row_slice(row).map(|slice| self.get_slice(slice))
    }

    unsafe fn get_row_unchecked(&self, row: RowId) -> Option<&[Value]> {
        let index = row.index();
        let marked_end = unsafe { self.offsets.get_unchecked(index) }.get();
        if marked_end & STALE_OFFSET_BIT != 0 {
            return None;
        }
        let start = if index == 0 {
            0
        } else {
            unsafe { self.offsets.get_unchecked(index - 1) }.get() & OFFSET_MASK
        };
        Some(self.get_slice(SequenceSlice {
            start,
            end: marked_end,
        }))
    }

    fn get_row_slice(&self, row: RowId) -> Option<SequenceSlice> {
        let index = row.index();
        let marked_end = self.offsets.get(index)?.get();
        if marked_end & STALE_OFFSET_BIT != 0 {
            return None;
        }
        let start = if index == 0 {
            0
        } else {
            self.offsets[index - 1].get() & OFFSET_MASK
        };
        Some(SequenceSlice {
            start,
            end: marked_end,
        })
    }

    fn get_slice(&self, range: SequenceSlice) -> &[Value] {
        &self.values[range.start()..range.end()]
    }

    fn set_stale(&mut self, location: SequenceLocation) -> bool {
        let offset = &self.offsets[location.row.index()];
        let end = offset.get();
        let was_stale = end & STALE_OFFSET_BIT != 0;
        if !was_stale {
            debug_assert_eq!(end, location.slice.end);
            self.stale_values += location.slice.len();
            offset.set(end | STALE_OFFSET_BIT);
            self.stale_rows += 1;
        }
        was_stale
    }

    /// Mark a logical row stale through shared access.
    ///
    /// # Safety
    /// The caller must own this row exclusively and ensure that no concurrent
    /// read observes its offset metadata.
    unsafe fn set_stale_shared(&self, row: RowId) -> bool {
        let offset = &self.offsets[row.index()];
        let end = offset.get();
        let was_stale = end & STALE_OFFSET_BIT != 0;
        if !was_stale {
            offset.set(end | STALE_OFFSET_BIT);
        }
        was_stale
    }

    fn parallel_writer(&mut self, rows: usize, values: usize) -> ParallelSequenceWriter {
        self.reserve(rows, values);
        let row_limit = self.offsets.len() + rows;
        let value_limit = self.values.len() + values;
        let end = pack_sequence_ends(self.offsets.len(), self.values.len());
        let mut values = Pooled::into_inner(mem::take(&mut self.values));
        let mut offsets = Pooled::into_inner(mem::take(&mut self.offsets));
        ParallelSequenceWriter {
            values_ptr: values.as_mut_ptr(),
            offsets_ptr: offsets.as_mut_ptr(),
            values_capacity: values.capacity(),
            offsets_capacity: offsets.capacity(),
            value_limit,
            row_limit,
            values: UnsafeCell::new(values),
            offsets: UnsafeCell::new(offsets),
            end: AtomicU64::new(end),
            stale_rows: self.stale_rows,
            stale_values: self.stale_values,
        }
    }
}

/// Variable-arity counterpart of `ParallelRowBufWriter`.
///
/// A fixed-width writer reserves one contiguous cell range. This writer must
/// atomically reserve matching ranges in both the packed-value and logical-row
/// offset vectors, then publish their final lengths after all workers join.
struct ParallelSequenceWriter {
    /// Stable element pointers captured before the writer is shared. Parallel
    /// operations never borrow the owning `Vec` metadata through `UnsafeCell`.
    values_ptr: *mut Value,
    offsets_ptr: *mut Cell<u32>,
    values_capacity: usize,
    offsets_capacity: usize,
    value_limit: usize,
    row_limit: usize,
    values: UnsafeCell<Vec<Value>>,
    offsets: UnsafeCell<Vec<Cell<u32>>>,
    /// High 32 bits are the next row, low 32 bits are the next value.
    end: AtomicU64,
    stale_rows: usize,
    stale_values: usize,
}

// The vectors never resize while shared. Appends reserve disjoint ranges with
// `end`; reads touch only the immutable prefix or a completed append, and
// liveness writes target rows exclusively owned by the calling hash shard.
unsafe impl Sync for ParallelSequenceWriter {}

impl ParallelSequenceWriter {
    /// Publish one worker-local batch with a coupled row/value reservation.
    fn append_batch(&self, batch: &SequenceBatch) -> (RowId, usize) {
        let previous = self
            .end
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                let (row_start, value_start) = unpack_sequence_ends(current);
                let row_end = row_start.checked_add(batch.offsets.len())?;
                let value_end = value_start.checked_add(batch.values.len())?;
                if row_end > self.row_limit || value_end > self.value_limit {
                    return None;
                }
                Some(pack_sequence_ends(row_end, value_end))
            })
            .unwrap_or_else(|_| {
                panic!("SequenceTable append exceeds its published producer epoch")
            });
        let (row_start, value_start) = unpack_sequence_ends(previous);
        // SAFETY: the combined atomic gives this call disjoint row and value
        // ranges in matching order. Both vectors were reserved from the
        // completed producer epoch before parallel work began and cannot
        // resize until `finish`.
        unsafe {
            debug_assert!(value_start + batch.values.len() <= self.values_capacity);
            debug_assert!(row_start + batch.offsets.len() <= self.offsets_capacity);
            std::ptr::copy_nonoverlapping(
                batch.values.as_ptr(),
                self.values_ptr.add(value_start),
                batch.values.len(),
            );
            for (index, row) in batch.offsets.iter().enumerate() {
                self.offsets_ptr
                    .add(row_start + index)
                    .write(Cell::new((value_start + row.end()) as u32));
            }
        }
        (RowId::from_usize(row_start), value_start)
    }

    /// Read a published backing range.
    ///
    /// # Safety
    /// `slice` must refer to initialized values in the original prefix or a
    /// completed batch append. Those values must not be mutated for the
    /// lifetime of the returned slice. This writer enforces the latter by
    /// never changing value bytes after an append; stale marking only changes
    /// the separate offset allocation.
    unsafe fn get_slice(&self, slice: SequenceSlice) -> &[Value] {
        unsafe { std::slice::from_raw_parts(self.values_ptr.add(slice.start()), slice.len()) }
    }

    /// Replace a completed row before publishing its hash entry.
    ///
    /// # Safety
    /// The caller must exclusively own `location`, and `values` must have the
    /// same length. No reader may observe the row until this write completes.
    unsafe fn replace_row(&self, location: SequenceLocation, values: &[Value]) {
        assert_eq!(location.slice.len(), values.len());
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                self.values_ptr.add(location.slice.start()),
                values.len(),
            );
        }
    }

    /// Mark an exclusively owned row stale.
    ///
    /// # Safety
    /// The caller must exclusively own `location.row`, and `location.slice`
    /// must be that row's completed append range. Exclusive ownership may come
    /// either from an unpublished append or from the unique destination hash
    /// shard that currently points at the row.
    unsafe fn mark_stale(&self, location: SequenceLocation) {
        let offset = unsafe { &*self.offsets_ptr.add(location.row.index()) };
        debug_assert_eq!(offset.get(), location.slice.end);
        offset.set(location.slice.end | STALE_OFFSET_BIT);
    }

    fn finish(self, newly_stale_rows: usize, newly_stale_values: usize) -> SequenceRows {
        let (row_end, value_end) = unpack_sequence_ends(self.end.load(Ordering::Acquire));
        let mut values = self.values.into_inner();
        let mut offsets = self.offsets.into_inner();
        // SAFETY: every element between the old lengths and these final ends
        // was initialized by exactly one completed append, and all workers
        // have joined before `finish` is called.
        unsafe {
            values.set_len(value_end);
            offsets.set_len(row_end);
        }
        SequenceRows {
            values: Pooled::new(values),
            offsets: Pooled::new(offsets),
            stale_rows: self.stale_rows + newly_stale_rows,
            stale_values: self.stale_values + newly_stale_values,
        }
    }
}

fn pack_sequence_ends(rows: usize, values: usize) -> u64 {
    assert!(rows < u32::MAX as usize);
    assert!(values <= OFFSET_MASK as usize);
    ((rows as u64) << 32) | values as u64
}

fn unpack_sequence_ends(packed: u64) -> (usize, usize) {
    ((packed >> 32) as usize, packed as u32 as usize)
}

/// Staging metadata for one packed sequence.
///
/// This plays the role of `HashedRowBuffer`'s trailing cached-hash value:
/// `end` delimits the row in producer-local packed values, while `hash` routes
/// and probes it without rehashing. Publication rebases the end into the
/// destination offset vector and caches the resulting full slice.
#[derive(Clone, Copy)]
struct HashedSequenceOffset {
    hash: CompactHash,
    end: u32,
}

impl HashedSequenceOffset {
    fn new(hash: CompactHash, end: usize) -> Self {
        assert!(end <= OFFSET_MASK as usize);
        Self {
            hash,
            end: end as u32,
        }
    }

    fn end(self) -> usize {
        self.end as usize
    }

    fn range_at(rows: &[Self], index: usize) -> std::ops::Range<usize> {
        let start = if index == 0 { 0 } else { rows[index - 1].end() };
        start..rows[index].end()
    }
}

/// Variable-arity counterpart of `HashedRowBuffer` for one physical shard.
///
/// Values remain row-major, but cumulative end offsets replace the fixed
/// stride; every offset carries the compact hash computed during staging.
#[derive(Clone, Default)]
struct HashedSequenceBuffer {
    values: Vec<Value>,
    /// Cumulative end offsets paired with hashes cached during staging.
    offsets: Vec<HashedSequenceOffset>,
}

impl HashedSequenceBuffer {
    fn add_row(&mut self, hash: CompactHash, row: &[Value]) {
        let end = self
            .values
            .len()
            .checked_add(row.len())
            .expect("SequenceTable producer value count overflow");
        assert!(end <= OFFSET_MASK as usize);
        self.values.extend_from_slice(row);
        self.offsets.push(HashedSequenceOffset::new(hash, end));
    }

    fn len(&self) -> usize {
        self.offsets.len()
    }

    fn value_len(&self) -> usize {
        self.values.len()
    }

    fn rows_hashed(&self) -> impl Iterator<Item = (CompactHash, &[Value])> {
        let mut start = 0;
        self.offsets.iter().map(move |row| {
            let end = row.end();
            let values = &self.values[start..end];
            start = end;
            (row.hash, values)
        })
    }
}

/// Producer-local mutation buffer corresponding to a fixed table's `Buffer`.
///
/// Rows are hashed and partitioned at staging time. Dropping the buffer moves
/// each shard-local batch into exactly one merge epoch.
struct SequenceBuffer {
    pending_rows: DenseIdMap<ShardId, HashedSequenceBuffer>,
    pending_removals: DenseIdMap<ShardId, HashedSequenceBuffer>,
    state: Weak<SequencePendingState>,
    shard_data: ShardData,
    n_values: usize,
}

impl MutationBuffer for SequenceBuffer {
    fn stage_insert(&mut self, row: &[Value]) {
        let key = sequence_key(row, self.n_values);
        let hash = shard_hash_values(self.shard_data, key);
        self.pending_rows
            .get_or_insert(hash.shard, HashedSequenceBuffer::default)
            .add_row(hash.compact, row);
    }

    fn stage_remove(&mut self, row: &[Value]) {
        // MutationBuffer follows the ordinary table convention: removals stage
        // a key, not a complete row with placeholder values.
        let hash = shard_hash_values(self.shard_data, row);
        self.pending_removals
            .get_or_insert(hash.shard, HashedSequenceBuffer::default)
            .add_row(hash.compact, row);
    }

    fn fresh_handle(&self) -> Box<dyn MutationBuffer> {
        Box::new(Self {
            pending_rows: DenseIdMap::with_capacity(self.shard_data.n_shards()),
            pending_removals: DenseIdMap::with_capacity(self.shard_data.n_shards()),
            state: self.state.clone(),
            shard_data: self.shard_data,
            n_values: self.n_values,
        })
    }
}

impl Drop for SequenceBuffer {
    fn drop(&mut self) {
        let Some(state) = self.state.upgrade() else {
            return;
        };
        let epoch = state
            .current
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let mut pending_rows = 0usize;
        let mut pending_values = 0usize;
        for shard_index in 0..self.pending_rows.n_ids() {
            let shard = ShardId::from_usize(shard_index);
            if let Some(buffer) = self.pending_rows.take(shard) {
                pending_rows = pending_rows.saturating_add(buffer.len());
                pending_values = pending_values.saturating_add(buffer.value_len());
                epoch.pending_rows[shard].push(buffer);
            }
        }
        epoch.add_pending_rows(pending_rows, pending_values);

        let mut pending_removals = 0usize;
        for shard_index in 0..self.pending_removals.n_ids() {
            let shard = ShardId::from_usize(shard_index);
            if let Some(buffer) = self.pending_removals.take(shard) {
                pending_removals = pending_removals.saturating_add(buffer.len());
                epoch.pending_removals[shard].push(buffer);
            }
        }
        epoch.add_pending_removals(pending_removals);
    }
}

/// One detached merge epoch of pre-sharded sequence mutations.
///
/// The coupled row/value count is the reservation bound consumed by
/// `ParallelSequenceWriter`; removals need only a row count.
struct SequenceEpoch {
    pending_rows: DenseIdMap<ShardId, SegQueue<HashedSequenceBuffer>>,
    pending_removals: DenseIdMap<ShardId, SegQueue<HashedSequenceBuffer>>,
    /// High 32 bits count staged rows; low 32 bits count staged values.
    pending_ends: AtomicU64,
    total_removals: AtomicUsize,
}

impl SequenceEpoch {
    fn new(shard_data: ShardData) -> Self {
        let mut pending_rows = DenseIdMap::with_capacity(shard_data.n_shards());
        let mut pending_removals = DenseIdMap::with_capacity(shard_data.n_shards());
        for index in 0..shard_data.n_shards() {
            let shard = ShardId::from_usize(index);
            pending_rows.insert(shard, SegQueue::default());
            pending_removals.insert(shard, SegQueue::default());
        }
        Self {
            pending_rows,
            pending_removals,
            pending_ends: AtomicU64::new(0),
            total_removals: AtomicUsize::new(0),
        }
    }

    fn add_pending_rows(&self, rows: usize, values: usize) {
        if rows == 0 {
            debug_assert_eq!(values, 0);
            return;
        }
        let updated =
            self.pending_ends
                .fetch_update(Ordering::Release, Ordering::Relaxed, |current| {
                    if current == u64::MAX {
                        return Some(u64::MAX);
                    }
                    let (current_rows, current_values) = unpack_sequence_ends(current);
                    let Some(next_rows) = current_rows.checked_add(rows) else {
                        return Some(u64::MAX);
                    };
                    let Some(next_values) = current_values.checked_add(values) else {
                        return Some(u64::MAX);
                    };
                    if next_rows >= u32::MAX as usize || next_values > OFFSET_MASK as usize {
                        Some(u64::MAX)
                    } else {
                        Some(pack_sequence_ends(next_rows, next_values))
                    }
                });
        debug_assert!(updated.is_ok());
    }

    fn add_pending_removals(&self, removals: usize) {
        if removals == 0 {
            return;
        }
        let updated =
            self.total_removals
                .fetch_update(Ordering::Release, Ordering::Relaxed, |current| {
                    Some(current.saturating_add(removals))
                });
        debug_assert!(updated.is_ok());
    }

    fn pending_rows(&self) -> (usize, usize) {
        let current = self.pending_ends.load(Ordering::Acquire);
        assert_ne!(
            current,
            u64::MAX,
            "SequenceTable pending rows exceed its offset space"
        );
        unpack_sequence_ends(current)
    }

    fn pending_removals(&self) -> usize {
        let current = self.total_removals.load(Ordering::Acquire);
        assert_ne!(
            current,
            usize::MAX,
            "SequenceTable pending removal count overflow"
        );
        current
    }

    fn deep_copy(&self) -> Self {
        fn copy_queues(
            source: &DenseIdMap<ShardId, SegQueue<HashedSequenceBuffer>>,
        ) -> DenseIdMap<ShardId, SegQueue<HashedSequenceBuffer>> {
            let mut result = DenseIdMap::with_capacity(source.n_ids());
            for (shard, queue) in source.iter() {
                let contents = drain_queue(queue);
                let copy = SegQueue::default();
                for buffer in contents {
                    copy.push(buffer.clone());
                    queue.push(buffer);
                }
                result.insert(shard, copy);
            }
            result
        }

        Self {
            pending_rows: copy_queues(&self.pending_rows),
            pending_removals: copy_queues(&self.pending_removals),
            pending_ends: AtomicU64::new(self.pending_ends.load(Ordering::Acquire)),
            total_removals: AtomicUsize::new(self.total_removals.load(Ordering::Acquire)),
        }
    }
}

/// Counterpart of fixed-table `PendingState`, with an epoch-detachment gate.
///
/// The additional `RwLock` makes each producer drop atomic with respect to
/// `merge`: a whole buffer lands before the detach or remains for the next
/// epoch.
struct SequencePendingState {
    shard_data: ShardData,
    /// Producer drops hold shared access while publishing a whole buffer.
    /// Merge swaps the epoch under exclusive access, then releases the gate
    /// before processing the detached queues in parallel.
    current: RwLock<SequenceEpoch>,
}

impl SequencePendingState {
    fn new(shard_data: ShardData) -> Self {
        Self {
            shard_data,
            current: RwLock::new(SequenceEpoch::new(shard_data)),
        }
    }

    fn detach(&self) -> SequenceEpoch {
        let mut current = self
            .current
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        mem::replace(&mut *current, SequenceEpoch::new(self.shard_data))
    }

    fn deep_copy(&self) -> Self {
        let current = self
            .current
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Self {
            shard_data: self.shard_data,
            current: RwLock::new(current.deep_copy()),
        }
    }
}

/// A table with a variable-length key and a fixed-width non-key value tail.
///
/// Mutations are staged through [`MutationBuffer`]s and become visible after
/// [`SequenceTable::merge`], matching `SortedWritesTable`'s publication model.
/// Rows of different lengths may coexist, provided each has at least
/// `n_values` entries. The key is `row[..row.len() - n_values]`; the remaining
/// entries are reconciled by the configured merge function when keys collide.
/// With `n_values == 0`, this is the original full-row-keyed set, including an
/// empty row.
///
/// This type is standalone rather than an implementation of [`crate::Table`]:
/// that trait describes fixed-schema relations. It shares the arity-neutral
/// row-id, subset, version, mutation-buffer, hashing, and sharded-table
/// machinery used by fixed-arity tables.
pub struct SequenceTable {
    generation: Generation,
    rows: SequenceRows,
    hash: ShardedHashTable<SequenceEntry>,
    pending_state: Arc<SequencePendingState>,
    n_values: usize,
    merge: Option<Arc<MergeFn>>,
    value_index: Option<SequenceValueIndex>,
}

impl Default for SequenceTable {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SequenceTable {
    fn clone(&self) -> Self {
        Self {
            generation: self.generation,
            rows: self.rows.clone(),
            hash: self.hash.clone(),
            pending_state: Arc::new(self.pending_state.deep_copy()),
            n_values: self.n_values,
            merge: self.merge.clone(),
            value_index: self.value_index.clone(),
        }
    }
}

impl SequenceTable {
    /// Create an empty full-row-keyed sequence table.
    pub fn new() -> Self {
        Self::new_inner(0, None)
    }

    /// Create a sequence table with `n_values` fixed-width non-key values.
    ///
    /// The merge function has the same contract as `SortedWritesTable`: it is
    /// passed the current and incoming complete rows and writes a replacement
    /// complete row when returning `true`. It must preserve the key.
    pub fn new_with_values(n_values: usize, merge: Box<MergeFn>) -> Self {
        assert!(n_values > 0, "use SequenceTable::new for a set table");
        Self::new_inner(n_values, Some(merge.into()))
    }

    /// Create a value-bearing sequence table with a semantic occurrence index.
    ///
    /// The index is maintained by the table's normal publication protocol.
    /// Appends extend it incrementally; compaction or clear rebuilds it for the
    /// new row-id generation. The extractor sees the variable-length key and
    /// selects exactly the values whose rows should be discoverable through
    /// [`SequenceTable::rows_for_indexed_value`].
    pub fn new_with_values_and_index(
        n_values: usize,
        merge: Box<MergeFn>,
        index_values: Box<SequenceIndexFn>,
    ) -> Self {
        assert!(
            n_values > 0,
            "use SequenceTable::new_indexed for a set table"
        );
        let mut table = Self::new_inner(n_values, Some(merge.into()));
        table.value_index = Some(SequenceValueIndex::new(index_values.into()));
        table
    }

    /// Create a full-row-keyed sequence set with a semantic occurrence index.
    pub fn new_indexed(index_values: Box<SequenceIndexFn>) -> Self {
        let mut table = Self::new_inner(0, None);
        table.value_index = Some(SequenceValueIndex::new(index_values.into()));
        table
    }

    fn new_inner(n_values: usize, merge: Option<Arc<MergeFn>>) -> Self {
        let hash = ShardedHashTable::default();
        let shard_data = hash.shard_data();
        Self {
            generation: Generation::new(0),
            rows: SequenceRows::default(),
            hash,
            pending_state: Arc::new(SequencePendingState::new(shard_data)),
            n_values,
            merge,
            value_index: None,
        }
    }

    fn new_sequence_buffer(&self) -> SequenceBuffer {
        let shard_data = self.hash.shard_data();
        SequenceBuffer {
            pending_rows: DenseIdMap::with_capacity(shard_data.n_shards()),
            pending_removals: DenseIdMap::with_capacity(shard_data.n_shards()),
            state: Arc::downgrade(&self.pending_state),
            shard_data,
            n_values: self.n_values,
        }
    }

    /// Create a producer-local mutation buffer.
    ///
    /// Producers may stage and drop buffers concurrently. Buffer publication
    /// is grouped atomically with respect to `merge`: a buffer is included in
    /// the current epoch if its publication wins the epoch gate, and otherwise
    /// remains queued for the next merge.
    pub fn new_buffer(&self) -> Box<dyn MutationBuffer> {
        Box::new(self.new_sequence_buffer())
    }

    pub(crate) fn shard_data(&self) -> ShardData {
        self.hash.shard_data()
    }

    /// Publish staged set mutations for a table with no non-key values.
    ///
    /// Removals are applied before insertions, so removing and inserting the
    /// same sequence in one epoch leaves that sequence present with a fresh
    /// `RowId`.
    pub fn merge(&mut self) -> TableChange {
        assert_eq!(
            self.n_values, 0,
            "tables with non-key values require merge_with_state"
        );
        self.merge_impl(None)
    }

    /// Publish staged removals and insertions, reconciling key collisions.
    pub fn merge_with_state(&mut self, exec_state: &mut ExecutionState) -> TableChange {
        self.merge_impl(Some(exec_state))
    }

    fn merge_impl(&mut self, exec_state: Option<&mut ExecutionState>) -> TableChange {
        let pending = self.pending_state.detach();
        let removed = self.do_delete(&pending);
        let added = self.do_insert(&pending, exec_state);
        self.maybe_compact();
        self.refresh_value_index();
        TableChange { removed, added }
    }

    /// Remap every key value in a set-valued table, then coalesce rows that
    /// canonicalize to the same sequence.
    ///
    /// Large scans are staged in coarse, producer-local chunks; small scans
    /// use one serial producer. The resulting delete/insert batches then use
    /// the normal merge path, which independently selects serial or parallel
    /// publication.
    pub fn rebuild_values(&mut self, rebuilder: &dyn ValueRebuilder) -> TableChange {
        assert_eq!(
            self.n_values, 0,
            "tables with non-key values require rebuild_values_with_state"
        );
        self.rebuild_rows(&|row, rebuilt| {
            rebuilt.extend_from_slice(row);
            rebuilder.rebuild_slice(rebuilt)
        })
    }

    /// Remap every value in the complete row, including the fixed-width tail.
    /// Key collisions are reconciled through the configured merge function.
    pub fn rebuild_values_with_state(
        &mut self,
        rebuilder: &dyn ValueRebuilder,
        exec_state: &mut ExecutionState,
    ) -> TableChange {
        self.rebuild_full_rows_with_state(
            &|row, rebuilt| {
                rebuilt.extend_from_slice(row);
                rebuilder.rebuild_slice(rebuilt)
            },
            exec_state,
        )
    }

    /// Rebuild live rows with a possibly length-changing transformation.
    ///
    /// `rebuild` writes a replacement into the provided empty vector and
    /// returns whether that replacement should be installed. Length-changing
    /// output is important for canonical set and map encodings, whose rebuild
    /// can collapse equal elements or keys.
    pub fn rebuild_rows(
        &mut self,
        rebuild: &(impl Fn(&[Value], &mut Vec<Value>) -> bool + Sync),
    ) -> TableChange {
        assert_eq!(
            self.n_values, 0,
            "tables with non-key values require rebuild_rows_with_state"
        );
        self.rebuild_full_rows_impl(rebuild, None)
    }

    /// Rebuild variable-length keys while preserving each row's value tail.
    pub fn rebuild_rows_with_state(
        &mut self,
        rebuild: &(impl Fn(&[Value], &mut Vec<Value>) -> bool + Sync),
        exec_state: &mut ExecutionState,
    ) -> TableChange {
        let n_values = self.n_values;
        self.rebuild_full_rows_with_state(
            &|row, rebuilt| {
                let key = sequence_key(row, n_values);
                if rebuild(key, rebuilt) {
                    rebuilt.extend_from_slice(sequence_values(row, n_values));
                    true
                } else {
                    false
                }
            },
            exec_state,
        )
    }

    /// Rebuild complete rows, allowing both the variable-length key and the
    /// fixed-width value tail to change.
    ///
    /// `rebuild` receives the complete old row and writes a complete new row.
    /// The new key may have a different length, but the row must still contain
    /// exactly `n_values` trailing non-key values.
    pub fn rebuild_full_rows_with_state(
        &mut self,
        rebuild: &(impl Fn(&[Value], &mut Vec<Value>) -> bool + Sync),
        exec_state: &mut ExecutionState,
    ) -> TableChange {
        self.rebuild_full_rows_impl(rebuild, Some(exec_state))
    }

    /// Rebuild only the live rows named by `subset`.
    ///
    /// This is the sequence-table counterpart of an incremental fixed-table
    /// rebuild driven by `rebuild_index`. Callers typically union one or more
    /// [`SequenceTable::rows_for_indexed_value`] results into an owned subset before
    /// taking the mutable table borrow required here.
    pub fn rebuild_full_rows_subset_with_state(
        &mut self,
        subset: SubsetRef<'_>,
        rebuild: &(impl Fn(&[Value], &mut Vec<Value>) -> bool + Sync),
        exec_state: &mut ExecutionState,
    ) -> TableChange {
        self.rebuild_full_rows_subset_impl(subset, rebuild, Some(exec_state))
    }

    fn rebuild_full_rows_impl(
        &mut self,
        rebuild: &(impl Fn(&[Value], &mut Vec<Value>) -> bool + Sync),
        exec_state: Option<&mut ExecutionState>,
    ) -> TableChange {
        let all = self.all();
        self.rebuild_full_rows_subset_impl(all.as_ref(), rebuild, exec_state)
    }

    fn rebuild_full_rows_subset_impl(
        &mut self,
        subset: SubsetRef<'_>,
        rebuild: &(impl Fn(&[Value], &mut Vec<Value>) -> bool + Sync),
        exec_state: Option<&mut ExecutionState>,
    ) -> TableChange {
        let Some((_low, high)) = subset.bounds() else {
            return TableChange {
                removed: false,
                added: false,
            };
        };
        assert!(high.index() <= self.rows.physical_len());
        let scan_size = subset.size();

        let stage_subset = |partition: SubsetRef<'_>| {
            let mut buffer = self.new_sequence_buffer();
            let mut rebuilt = with_pool_set(|pools| pools.get::<Vec<Value>>());
            let mut changed = 0usize;
            partition.offsets(|row_id| {
                let Some(row) = self.rows.get_row(row_id) else {
                    return;
                };
                let key = sequence_key(row, self.n_values);
                if rebuild(row, &mut rebuilt) {
                    assert!(
                        rebuilt.len() >= self.n_values,
                        "rebuilt sequence row has fewer than {} non-key values",
                        self.n_values
                    );
                    buffer.stage_remove(key);
                    buffer.stage_insert(&rebuilt);
                    changed += 1;
                }
                rebuilt.clear();
            });
            changed
        };

        // Match fixed-arity full rebuilds: avoid parallel dispatch and many
        // small mutation-buffer publications until the scan is large enough
        // to amortize them.
        let changed = if parallelize_rebuild(scan_size) {
            let starts = (0..scan_size)
                .step_by(REBUILD_CHUNK_ROWS)
                .collect::<Vec<_>>();
            parallel::map(&starts, |_, start| {
                stage_subset(sequence_subset_partition(
                    subset,
                    *start,
                    cmp::min(*start + REBUILD_CHUNK_ROWS, scan_size),
                ))
            })
            .into_iter()
            .sum::<usize>()
        } else {
            stage_subset(subset)
        };

        if changed == 0 {
            TableChange {
                removed: false,
                added: false,
            }
        } else {
            self.merge_impl(exec_state)
        }
    }

    /// Remove all rows and pending mutations.
    pub fn clear(&mut self) {
        // Replacing the state invalidates Weak handles held by every existing
        // buffer. A Drop that already upgraded the old state may finish there,
        // but that state is no longer reachable from this table.
        self.pending_state = Arc::new(SequencePendingState::new(self.hash.shard_data()));
        if self.rows.physical_len() == 0 {
            return;
        }
        self.rows.clear();
        self.hash.clear();
        self.generation = self.generation.inc();
        let version = self.version();
        if let Some(index) = &mut self.value_index {
            index.clear_to(version);
        }
    }

    /// Return the generation-scoped physical row id for an exact key.
    pub fn lookup(&self, key: &[Value]) -> Option<RowId> {
        self.find_entry(key).map(|entry| entry.row.row)
    }

    /// Borrow the complete live row for an exact key lookup.
    ///
    /// This uses the slice cached in the hash entry, avoiding a second access
    /// to the cumulative-offset vector.
    pub fn get_row_ref(&self, key: &[Value]) -> Option<(RowId, &[Value])> {
        let entry = self.find_entry(key)?;
        Some((entry.row.row, self.rows.get_slice(entry.row.slice)))
    }

    /// Borrow the fixed-width value tail for an exact key lookup.
    pub fn get_values(&self, key: &[Value]) -> Option<&[Value]> {
        self.get_row_ref(key)
            .map(|(_, row)| sequence_values(row, self.n_values))
    }

    /// Return an owned row for an exact sequence lookup.
    pub fn get_row(&self, key: &[Value]) -> Option<Row> {
        let entry = self.find_entry(key)?;
        let mut vals = with_pool_set(|pools| pools.get::<Vec<Value>>());
        vals.extend_from_slice(self.rows.get_slice(entry.row.slice));
        Some(Row {
            id: entry.row.row,
            vals,
        })
    }

    /// Read a live row by its generation-scoped physical id.
    pub fn row(&self, row: RowId) -> Option<&[Value]> {
        self.rows.get_row(row)
    }

    /// Read only the variable-length key prefix of a live physical row.
    pub fn key(&self, row: RowId) -> Option<&[Value]> {
        self.row(row).map(|row| sequence_key(row, self.n_values))
    }

    /// Read only the fixed-width non-key tail of a live physical row.
    pub fn values(&self, row: RowId) -> Option<&[Value]> {
        self.row(row).map(|row| sequence_values(row, self.n_values))
    }

    /// Number of non-key values stored at the end of every row.
    pub fn n_values(&self) -> usize {
        self.n_values
    }

    /// Return physical rows whose extractor-selected key positions include
    /// `value`.
    ///
    /// Incremental index maintenance intentionally retains row ids made stale
    /// by deletion, just like fixed-table rebuild indexes. Consumers must read
    /// rows through [`SequenceTable::row`] or [`SequenceTable::scan`], both of
    /// which skip stale ids. A compaction changes the major generation and
    /// rebuilds the index without those tombstones.
    pub fn rows_for_indexed_value(&self, value: Value) -> Option<SubsetRef<'_>> {
        self.value_index.as_ref()?.get(&value)
    }

    /// Return an owned snapshot of
    /// [`SequenceTable::rows_for_indexed_value`].
    ///
    /// The borrowed form cannot be retained across a mutable table operation;
    /// this form is intended for selecting candidates before an index-driven
    /// rebuild or another operation that may merge or compact the table.
    pub fn owned_rows_for_indexed_value(&self, value: Value) -> Option<Subset> {
        let rows = self.rows_for_indexed_value(value)?;
        Some(with_pool_set(|pools| rows.to_owned(&pools.get_pool())))
    }

    /// Iterate over live rows in a subset.
    pub fn scan(&self, subset: SubsetRef<'_>, mut f: impl FnMut(RowId, &[Value])) {
        let Some((_low, high)) = subset.bounds() else {
            return;
        };
        assert!(high.index() <= self.rows.physical_len());
        // SAFETY: all subset rows are bounded above by the checked physical
        // length, and offsets remain stable until an exclusive merge/compact.
        subset.offsets(|row| unsafe {
            if let Some(values) = self.rows.get_row_unchecked(row) {
                f(row, values);
            }
        });
    }

    /// Iterate over live rows, splitting each into its key and value tail.
    pub fn scan_key_values(
        &self,
        subset: SubsetRef<'_>,
        mut f: impl FnMut(RowId, &[Value], &[Value]),
    ) {
        self.scan(subset, |row_id, row| {
            let key = sequence_key(row, self.n_values);
            let values = sequence_values(row, self.n_values);
            f(row_id, key, values);
        });
    }

    /// Return every physical row id; scans automatically skip stale rows.
    pub fn all(&self) -> Subset {
        Subset::Dense(OffsetRange::new(RowId::new(0), self.rows.next_row()))
    }

    /// Return physical row ids appended since a minor version offset.
    pub fn updates_since(&self, offset: Offset) -> Subset {
        Subset::Dense(OffsetRange::new(
            RowId::from_usize(offset.index()),
            self.rows.next_row(),
        ))
    }

    /// Return the table generation and next physical row offset.
    pub fn version(&self) -> TableVersion {
        TableVersion {
            major: self.generation,
            minor: Offset::from_usize(self.rows.physical_len()),
        }
    }

    /// Number of live sequences.
    pub fn len(&self) -> usize {
        self.rows.live_len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn has_stale_rows(&self) -> bool {
        self.rows.stale_rows != 0
    }

    fn find_entry(&self, key: &[Value]) -> Option<&SequenceEntry> {
        let hash = shard_hash_values(self.hash.shard_data(), key);
        self.hash
            .get_shard(hash.shard)
            .find(hash.compact.probe().raw(), |entry| {
                entry.hash == hash.compact
                    && sequence_key(self.rows.get_slice(entry.row.slice), self.n_values) == key
            })
    }

    fn do_insert(
        &mut self,
        pending: &SequenceEpoch,
        exec_state: Option<&mut ExecutionState>,
    ) -> bool {
        let (total_rows, total_values) = pending.pending_rows();
        if total_rows == 0 {
            return false;
        }
        if parallelize_table_op(total_rows) {
            self.parallel_insert(pending, total_rows, total_values, exec_state.as_deref())
        } else {
            self.serial_insert(pending, total_rows, total_values, exec_state)
        }
    }

    fn serial_insert(
        &mut self,
        pending: &SequenceEpoch,
        total_rows: usize,
        total_values: usize,
        mut exec_state: Option<&mut ExecutionState>,
    ) -> bool {
        self.rows.reserve(total_rows, total_values);
        let n_values = self.n_values;
        let merge = self.merge.clone();
        let mut scratch = with_pool_set(|pools| pools.get::<Vec<Value>>());
        let mut changed = false;
        for shard_index in 0..pending.pending_rows.n_ids() {
            let shard_id = ShardId::from_usize(shard_index);
            let buffers = drain_queue(&pending.pending_rows[shard_id]);
            let incoming = buffers.iter().map(HashedSequenceBuffer::len).sum();
            let shard = &mut self.hash.mut_shards()[shard_index];
            shard.reserve(incoming, SequenceEntry::raw_probe_hash);
            for buffer in &buffers {
                for (hash, row) in buffer.rows_hashed() {
                    let key = sequence_key(row, n_values);
                    use hashbrown::hash_table::Entry;
                    match shard.entry(
                        hash.probe().raw(),
                        |entry| {
                            entry.hash == hash
                                && sequence_key(self.rows.get_slice(entry.row.slice), n_values)
                                    == key
                        },
                        SequenceEntry::raw_probe_hash,
                    ) {
                        Entry::Occupied(mut occupied) => {
                            let Some(merge) = &merge else {
                                debug_assert_eq!(n_values, 0);
                                continue;
                            };
                            let current_location = occupied.get().row;
                            let current = self.rows.get_slice(current_location.slice);
                            let state = exec_state
                                .as_deref_mut()
                                .expect("value-bearing sequence merges require ExecutionState");
                            if merge(state, current, row, &mut scratch) {
                                assert_valid_merge_output(&scratch, key, n_values);
                                let location = self.rows.add_row_reserved(&scratch);
                                let was_stale = self.rows.set_stale(current_location);
                                debug_assert!(!was_stale);
                                occupied.get_mut().row = location;
                                changed = true;
                            }
                            scratch.clear();
                        }
                        Entry::Vacant(vacant) => {
                            let location = self.rows.add_row_reserved(row);
                            vacant.insert(SequenceEntry {
                                hash,
                                row: location,
                            });
                            changed = true;
                        }
                    }
                }
            }
        }
        changed
    }

    /// Variable-key counterpart of `SortedWritesTable::parallel_insert`.
    ///
    /// Pre-sharding gives each worker exclusive ownership of one destination
    /// shard. A worker coalesces a bounded `SequenceBatch`, appends its rows
    /// with one coupled reservation, then publishes vacant entries or merges
    /// colliding value tails. `finish` publishes the final vector lengths after
    /// every worker joins.
    fn parallel_insert(
        &mut self,
        pending: &SequenceEpoch,
        total_rows: usize,
        total_values: usize,
        exec_state: Option<&ExecutionState>,
    ) -> bool {
        let writer = self.rows.parallel_writer(total_rows, total_values);
        let pending_rows = &pending.pending_rows;
        let n_values = self.n_values;
        let merge = self.merge.clone();
        let results = parallel::map_mut(self.hash.mut_shards(), |shard_index, shard| {
            let mut exec_state = exec_state.cloned();
            let shard_id = ShardId::from_usize(shard_index);
            let buffers = drain_queue(&pending_rows[shard_id]);
            let incoming = buffers.iter().map(HashedSequenceBuffer::len).sum();
            if incoming == 0 {
                return (0, 0, 0);
            }
            let incoming_values = buffers.iter().map(HashedSequenceBuffer::value_len).sum();
            shard.reserve(incoming, SequenceEntry::raw_probe_hash);
            let mut batch = SequenceBatch::new(
                cmp::min(incoming, PARALLEL_SEQUENCE_BATCH_ROWS),
                cmp::min(incoming_values, PARALLEL_SEQUENCE_BATCH_VALUES),
            );
            let mut changed = 0usize;
            let mut stale_rows = 0usize;
            let mut stale_values = 0usize;

            for buffer in &buffers {
                for (hash, row) in buffer.rows_hashed() {
                    batch.insert_with_merge(hash, row, n_values, |current, incoming, output| {
                        let Some(merge) = &merge else {
                            debug_assert_eq!(n_values, 0);
                            return false;
                        };
                        let state = exec_state
                            .as_mut()
                            .expect("value-bearing sequence merges require ExecutionState");
                        merge(state, current, incoming, output)
                    });
                    if batch.offsets.len() >= PARALLEL_SEQUENCE_BATCH_ROWS
                        || batch.values.len() >= PARALLEL_SEQUENCE_BATCH_VALUES
                    {
                        let flushed = flush_sequence_batch_with_merge(
                            &mut batch,
                            &writer,
                            shard,
                            n_values,
                            merge.as_deref(),
                            exec_state.as_mut(),
                        );
                        changed += flushed.0;
                        stale_rows += flushed.1;
                        stale_values += flushed.2;
                    }
                }
            }
            let flushed = flush_sequence_batch_with_merge(
                &mut batch,
                &writer,
                shard,
                n_values,
                merge.as_deref(),
                exec_state.as_mut(),
            );
            (
                changed + flushed.0,
                stale_rows + flushed.1,
                stale_values + flushed.2,
            )
        });
        let (changed, stale_rows, stale_values) =
            results
                .into_iter()
                .fold((0, 0, 0), |(changed, stale_rows, stale_values), new| {
                    (changed + new.0, stale_rows + new.1, stale_values + new.2)
                });

        self.rows = writer.finish(stale_rows, stale_values);
        changed != 0
    }

    fn do_delete(&mut self, pending: &SequenceEpoch) -> bool {
        let total = pending.pending_removals();
        if total == 0 {
            return false;
        }
        if parallelize_table_op(total) {
            self.parallel_delete(pending)
        } else {
            self.serial_delete(pending)
        }
    }

    fn serial_delete(&mut self, pending: &SequenceEpoch) -> bool {
        let mut removed = 0usize;
        for shard_index in 0..pending.pending_removals.n_ids() {
            let shard_id = ShardId::from_usize(shard_index);
            let buffers = drain_queue(&pending.pending_removals[shard_id]);
            let shard = &mut self.hash.mut_shards()[shard_index];
            for buffer in &buffers {
                for (hash, row) in buffer.rows_hashed() {
                    if let Ok(entry) = shard.find_entry(hash.probe().raw(), |entry| {
                        entry.hash == hash
                            && sequence_key(self.rows.get_slice(entry.row.slice), self.n_values)
                                == row
                    }) {
                        let (entry, _) = entry.remove();
                        let was_stale = self.rows.set_stale(entry.row);
                        debug_assert!(!was_stale);
                        removed += 1;
                    }
                }
            }
        }
        removed != 0
    }

    fn parallel_delete(&mut self, pending: &SequenceEpoch) -> bool {
        let rows = &self.rows;
        let n_values = self.n_values;
        let pending_removals = &pending.pending_removals;
        let (removed, stale_values) =
            parallel::map_chunks_mut(self.hash.mut_shards(), |base, shards| {
                let mut removed_rows = with_pool_set(|pools| pools.get::<Vec<RowId>>());
                let mut stale_values = 0usize;
                for (offset, shard) in shards.iter_mut().enumerate() {
                    let shard_id = ShardId::from_usize(base + offset);
                    let buffers = drain_queue(&pending_removals[shard_id]);
                    for buffer in &buffers {
                        for (hash, row) in buffer.rows_hashed() {
                            if let Ok(entry) = shard.find_entry(hash.probe().raw(), |entry| {
                                entry.hash == hash
                                    && sequence_key(rows.get_slice(entry.row.slice), n_values)
                                        == row
                            }) {
                                let (entry, _) = entry.remove();
                                removed_rows.push(entry.row.row);
                                stale_values += entry.row.slice.len();
                            }
                        }
                    }
                }
                // SAFETY: every live sequence has exactly one hash entry in
                // one worker-owned shard, so successful removals across all
                // workers yield disjoint row ids. Probes use cached slices and
                // never inspect offset metadata.
                for &row in &*removed_rows {
                    let was_stale = unsafe { rows.set_stale_shared(row) };
                    debug_assert!(!was_stale);
                }
                (removed_rows.len(), stale_values)
            })
            .into_iter()
            .fold((0, 0), |(rows, values), (new_rows, new_values)| {
                (rows + new_rows, values + new_values)
            });
        self.rows.stale_rows += removed;
        self.rows.stale_values += stale_values;
        removed != 0
    }

    fn maybe_compact(&mut self) {
        let stale_row_pressure = self.rows.stale_rows > cmp::max(16, self.rows.physical_len() / 2);
        let stale_value_pressure =
            self.rows.stale_values > cmp::max(16, self.rows.values.len() / 2);
        if !stale_row_pressure && !stale_value_pressure {
            return;
        }
        self.compact();
    }

    /// Remove stale backing ranges and invalidate existing RowIds/Subsets.
    pub fn compact(&mut self) {
        if self.rows.stale_rows == 0 {
            return;
        }
        self.generation = self.generation.inc();
        if parallelize_table_op(self.rows.physical_len()) {
            self.parallel_compact();
        } else {
            self.serial_compact();
        }
        self.refresh_value_index();
    }

    fn refresh_value_index(&mut self) {
        let version = self.version();
        if let Some(index) = &mut self.value_index {
            index.refresh(&self.rows, self.n_values, version);
        }
    }

    fn serial_compact(&mut self) {
        let mut compacted = SequenceRows::default();
        compacted.reserve(self.rows.live_len(), self.live_value_count());
        for shard in self.hash.mut_shards() {
            for entry in shard.iter_mut() {
                let row = self.rows.get_slice(entry.row.slice);
                entry.row = compacted.add_row_reserved(row);
            }
        }
        self.rows = compacted;
    }

    fn parallel_compact(&mut self) {
        // This mirrors `SortedWritesTable::parallel_rehash_unsorted`. Its
        // single row-count prefix sum becomes two prefix sums here: one assigns
        // disjoint RowId/offset slots, and the other assigns disjoint packed-
        // value ranges.
        let shards = self.hash.mut_shards();
        // Each live row has exactly one hash entry, so shard-local entry and
        // value counts describe the complete compacted output.
        let counts = parallel::map(shards, |_, shard| {
            (
                shard.len(),
                shard
                    .iter()
                    .map(|entry| entry.row.slice.len())
                    .sum::<usize>(),
            )
        });
        // Prefix sums turn those counts into the two disjoint output ranges
        // owned by each shard worker.
        let mut row_starts = Vec::with_capacity(shards.len() + 1);
        let mut value_starts = Vec::with_capacity(shards.len() + 1);
        row_starts.push(0usize);
        value_starts.push(0usize);
        for (rows, values) in counts {
            row_starts.push(row_starts.last().copied().unwrap() + rows);
            value_starts.push(value_starts.last().copied().unwrap() + values);
        }

        let live_rows = *row_starts.last().unwrap();
        let live_values = *value_starts.last().unwrap();
        debug_assert_eq!(live_rows, self.rows.live_len());

        // Reserve both outputs once. Their lengths remain zero while workers
        // initialize disjoint raw ranges, then are published after the join.
        let mut values = with_pool_set(|pools| pools.get::<Vec<Value>>());
        let mut offsets = with_pool_set(|pools| pools.get::<Vec<Cell<u32>>>());
        values.reserve(live_values);
        offsets.reserve(live_rows);
        let values_ptr = values.as_mut_ptr() as usize;
        let offsets_ptr = offsets.as_mut_ptr() as usize;
        let old_rows = &self.rows;

        parallel::for_each_mut(shards, |shard_index, shard| {
            let values_ptr = values_ptr as *mut Value;
            let offsets_ptr = offsets_ptr as *mut Cell<u32>;
            let mut next_row = row_starts[shard_index];
            let mut next_value = value_starts[shard_index];
            for entry in shard.iter_mut() {
                let source = old_rows.get_slice(entry.row.slice);
                let start = next_value;
                // SAFETY: prefix sums assign this shard disjoint initialized
                // output ranges, both vectors retain their reserved allocation
                // until all workers have joined, and Value is Copy.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        source.as_ptr(),
                        values_ptr.add(next_value),
                        source.len(),
                    );
                    next_value += source.len();
                    let range = SequenceSlice::new(start, next_value);
                    offsets_ptr.add(next_row).write(Cell::new(range.end));
                    entry.row = SequenceLocation {
                        slice: range,
                        row: RowId::from_usize(next_row),
                    };
                }
                next_row += 1;
            }
            debug_assert_eq!(next_row, row_starts[shard_index + 1]);
            debug_assert_eq!(next_value, value_starts[shard_index + 1]);
        });

        // SAFETY: every element in both prefix ranges was initialized exactly
        // once, and the parallel scope has joined.
        unsafe {
            values.set_len(live_values);
            offsets.set_len(live_rows);
        }
        self.rows = SequenceRows {
            values,
            offsets,
            stale_rows: 0,
            stale_values: 0,
        };
    }

    fn live_value_count(&self) -> usize {
        self.rows.values.len() - self.rows.stale_values
    }
}

/// Variable-key counterpart of `CoalescedInsertBatch`.
///
/// It hashes only the variable-length key and merges fixed-width value tails
/// within one bounded per-shard batch before shared publication.
struct SequenceBatch {
    values: Pooled<Vec<Value>>,
    offsets: Vec<HashedSequenceOffset>,
    hash: Pooled<HashTable<RowId>>,
    scratch: Pooled<Vec<Value>>,
}

impl SequenceBatch {
    fn new(row_capacity: usize, value_capacity: usize) -> Self {
        let mut result = with_pool_set(|pools| Self {
            values: pools.get(),
            offsets: Vec::with_capacity(row_capacity),
            hash: pools.get(),
            scratch: pools.get(),
        });
        result.values.reserve(value_capacity);
        result.hash.reserve(row_capacity, |row| {
            super::coalesced_hash(result.offsets[row.index()].hash)
        });
        result
    }

    #[cfg(test)]
    fn insert(&mut self, hash: CompactHash, values: &[Value]) {
        self.insert_with_merge(hash, values, 0, |_, _, _| false);
    }

    fn insert_with_merge(
        &mut self,
        hash: CompactHash,
        values: &[Value],
        n_values: usize,
        mut merge: impl FnMut(&[Value], &[Value], &mut Vec<Value>) -> bool,
    ) {
        let key = sequence_key(values, n_values);
        use hashbrown::hash_table::Entry;
        let offsets = &self.offsets;
        let buffered_values = &self.values;
        match self.hash.entry(
            super::coalesced_hash(hash),
            |row| {
                let offset = offsets[row.index()];
                if offset.hash != hash {
                    return false;
                }
                let current =
                    &buffered_values[HashedSequenceOffset::range_at(offsets, row.index())];
                sequence_key(current, n_values) == key
            },
            |row| super::coalesced_hash(offsets[row.index()].hash),
        ) {
            Entry::Occupied(occupied) => {
                let range = HashedSequenceOffset::range_at(&self.offsets, occupied.get().index());
                if merge(&self.values[range.clone()], values, &mut self.scratch) {
                    assert_valid_merge_output(&self.scratch, key, n_values);
                    self.values[range].copy_from_slice(&self.scratch);
                }
                self.scratch.clear();
            }
            Entry::Vacant(vacant) => {
                let start = self.values.len();
                self.values.extend_from_slice(values);
                let row = RowId::from_usize(self.offsets.len());
                debug_assert_eq!(
                    start,
                    self.offsets.last().map_or(0, |previous| previous.end())
                );
                self.offsets
                    .push(HashedSequenceOffset::new(hash, self.values.len()));
                vacant.insert(row);
            }
        }
    }

    fn clear(&mut self) {
        self.hash.clear();
        self.values.clear();
        self.offsets.clear();
        self.scratch.clear();
    }
}

#[cfg(test)]
fn flush_sequence_batch(
    batch: &mut SequenceBatch,
    writer: &ParallelSequenceWriter,
    shard: &mut HashTable<SequenceEntry>,
) -> (usize, usize, usize) {
    flush_sequence_batch_with_merge(batch, writer, shard, 0, None, None)
}

fn flush_sequence_batch_with_merge(
    batch: &mut SequenceBatch,
    writer: &ParallelSequenceWriter,
    shard: &mut HashTable<SequenceEntry>,
    n_values: usize,
    merge: Option<&MergeFn>,
    mut exec_state: Option<&mut ExecutionState>,
) -> (usize, usize, usize) {
    if batch.offsets.is_empty() {
        return (0, 0, 0);
    }
    let (first_row, value_start) = writer.append_batch(batch);
    let mut changed = 0;
    let mut stale_rows = 0;
    let mut stale_values = 0;
    let mut local_start = 0;
    for (index, buffered) in batch.offsets.iter().enumerate() {
        let local_end = buffered.end();
        let location = SequenceLocation {
            slice: SequenceSlice::new(value_start + local_start, value_start + local_end),
            row: RowId::from_usize(first_row.index() + index),
        };
        let incoming = &batch.values[local_start..local_end];
        let key = sequence_key(incoming, n_values);
        use hashbrown::hash_table::Entry;
        match shard.entry(
            buffered.hash.probe().raw(),
            |entry| {
                if entry.hash != buffered.hash {
                    return false;
                }
                // SAFETY: every destination entry points to either the
                // original prefix or a completed batch write. The caller
                // reserved all backing capacity before obtaining this read
                // handle, so concurrent appends cannot invalidate it.
                let stored = unsafe { writer.get_slice(entry.row.slice) };
                sequence_key(stored, n_values) == key
            },
            SequenceEntry::raw_probe_hash,
        ) {
            Entry::Vacant(vacant) => {
                vacant.insert(SequenceEntry {
                    hash: buffered.hash,
                    row: location,
                });
                changed += 1;
            }
            Entry::Occupied(mut occupied) => {
                let mut keep_incoming = false;
                if let Some(merge) = merge {
                    let old_location = occupied.get().row;
                    // SAFETY: the destination shard exclusively owns the old
                    // row, and the incoming row is this worker's unpublished
                    // append. Both backing ranges are initialized and stable.
                    let current = unsafe { writer.get_slice(old_location.slice) };
                    if merge(
                        exec_state
                            .as_deref_mut()
                            .expect("value-bearing sequence merges require ExecutionState"),
                        current,
                        incoming,
                        &mut batch.scratch,
                    ) {
                        assert_valid_merge_output(&batch.scratch, key, n_values);
                        unsafe {
                            writer.replace_row(location, &batch.scratch);
                            writer.mark_stale(old_location);
                        }
                        occupied.get_mut().row = location;
                        stale_rows += 1;
                        stale_values += old_location.slice.len();
                        changed += 1;
                        keep_incoming = true;
                    }
                    batch.scratch.clear();
                } else {
                    debug_assert_eq!(n_values, 0);
                }
                if !keep_incoming {
                    // SAFETY: this row was just reserved by this worker and
                    // has not been published.
                    unsafe { writer.mark_stale(location) };
                    stale_rows += 1;
                    stale_values += location.slice.len();
                }
            }
        }
        local_start = local_end;
    }
    batch.clear();
    (changed, stale_rows, stale_values)
}

#[cfg(test)]
mod tests {
    use std::{
        mem::{align_of, size_of},
        sync::{Arc, Barrier, Mutex},
    };

    use egglog_concurrency::{ThreadPool, scope};

    use super::*;

    fn v(value: usize) -> Value {
        Value::from_usize(value)
    }

    fn row(values: &[usize]) -> Vec<Value> {
        values.iter().copied().map(v).collect()
    }

    fn scan_all(table: &SequenceTable) -> Vec<Vec<Value>> {
        let mut rows = Vec::new();
        table.scan(table.all().as_ref(), |_, row| rows.push(row.to_vec()));
        rows
    }

    fn max_tail_merge() -> Box<MergeFn> {
        Box::new(|_, current, incoming, output| {
            if incoming.last() > current.last() {
                output.extend_from_slice(incoming);
                true
            } else {
                false
            }
        })
    }

    #[test]
    fn sequence_layouts_stay_compact() {
        assert_eq!(size_of::<SequenceSlice>(), 8);
        assert_eq!(align_of::<SequenceSlice>(), 4);
        assert_eq!(size_of::<SequenceLocation>(), 12);
        assert_eq!(size_of::<SequenceEntry>(), 16);
        assert_eq!(align_of::<SequenceEntry>(), 4);
        assert_eq!(size_of::<HashedSequenceOffset>(), 8);
    }

    #[test]
    fn packed_sequence_counter_keeps_both_fields_independent() {
        assert_eq!(unpack_sequence_ends(pack_sequence_ends(0, 0)), (0, 0));
        assert_eq!(
            unpack_sequence_ends(pack_sequence_ends(
                u32::MAX as usize - 1,
                OFFSET_MASK as usize,
            )),
            (u32::MAX as usize - 1, OFFSET_MASK as usize)
        );

        let near_end = pack_sequence_ends(u32::MAX as usize - 11, OFFSET_MASK as usize - 10);
        let delta = pack_sequence_ends(10, 10);
        assert_eq!(
            unpack_sequence_ends(near_end + delta),
            (u32::MAX as usize - 1, OFFSET_MASK as usize)
        );
    }

    #[test]
    fn packed_reservation_keeps_parallel_row_and_value_order_aligned() {
        const BATCHES: usize = 64;
        ThreadPool::new(4).install(|| {
            let expected_values = (0..BATCHES)
                .map(|batch| (batch % 7) + 1 + (batch % 5))
                .sum();
            let mut rows = SequenceRows::default();
            let writer = rows.parallel_writer(BATCHES * 2, expected_values);
            let published = Mutex::new(Vec::new());

            scope(|scope| {
                let writer = &writer;
                let published = &published;
                for batch_index in 0..BATCHES {
                    scope.spawn(move |_| {
                        let first = (0..=batch_index % 7)
                            .map(|offset| v(batch_index * 100 + offset))
                            .collect::<Vec<_>>();
                        let second = (0..batch_index % 5)
                            .map(|offset| v(10_000 + batch_index * 100 + offset))
                            .collect::<Vec<_>>();
                        let mut batch = SequenceBatch::new(2, first.len() + second.len());
                        batch.insert(CompactHash(batch_index as u32), &first);
                        batch.insert(CompactHash((batch_index + BATCHES) as u32), &second);
                        let (start, _) = writer.append_batch(&batch);
                        published.lock().unwrap().push((start, first, second));
                    });
                }
            });

            rows = writer.finish(0, 0);
            assert_eq!(rows.physical_len(), BATCHES * 2);
            assert_eq!(rows.values.len(), expected_values);
            for (start, first, second) in published.into_inner().unwrap() {
                assert_eq!(rows.get_row(start), Some(first.as_slice()));
                assert_eq!(rows.get_row(start.inc()), Some(second.as_slice()));
            }
        });
    }

    #[test]
    fn append_first_duplicate_marks_cumulative_offsets_stale() {
        let empty = row(&[]);
        let nonempty = row(&[1, 2, 3]);
        let shard_data = ShardData::new(1);
        let empty_hash = shard_hash_values(shard_data, &empty).compact;
        let nonempty_hash = shard_hash_values(shard_data, &nonempty).compact;
        let mut rows = SequenceRows::default();
        let writer = rows.parallel_writer(4, nonempty.len() * 2);
        let mut shard = HashTable::new();

        let mut first = SequenceBatch::new(2, nonempty.len());
        first.insert(empty_hash, &empty);
        first.insert(nonempty_hash, &nonempty);
        assert_eq!(
            flush_sequence_batch(&mut first, &writer, &mut shard),
            (2, 0, 0)
        );

        let mut duplicate = SequenceBatch::new(2, nonempty.len());
        duplicate.insert(empty_hash, &empty);
        duplicate.insert(nonempty_hash, &nonempty);
        let result = flush_sequence_batch(&mut duplicate, &writer, &mut shard);
        assert_eq!(result, (0, 2, nonempty.len()));

        rows = writer.finish(result.1, result.2);
        assert_eq!(rows.physical_len(), 4);
        assert_eq!(rows.live_len(), 2);
        assert_eq!(rows.get_row(RowId::new(0)), Some(empty.as_slice()));
        assert_eq!(rows.get_row(RowId::new(1)), Some(nonempty.as_slice()));
        assert_eq!(rows.get_row(RowId::new(2)), None);
        assert_eq!(rows.get_row(RowId::new(3)), None);
        assert_eq!(rows.offsets[2].get() & OFFSET_MASK, 3);
        assert_eq!(rows.offsets[3].get() & OFFSET_MASK, 6);
    }

    #[test]
    fn empty_only_append_uses_zero_capacity_value_backing() {
        let empty = row(&[]);
        let hash = shard_hash_values(ShardData::new(1), &empty).compact;
        let mut rows = SequenceRows {
            values: Pooled::new(Vec::new()),
            offsets: Pooled::new(Vec::new()),
            stale_rows: 0,
            stale_values: 0,
        };
        let writer = rows.parallel_writer(2, 0);
        let mut shard = HashTable::new();

        for expected in [(1, 0, 0), (0, 1, 0)] {
            let mut batch = SequenceBatch::new(1, 0);
            batch.insert(hash, &empty);
            assert_eq!(
                flush_sequence_batch(&mut batch, &writer, &mut shard),
                expected
            );
        }

        rows = writer.finish(1, 0);
        assert_eq!(rows.values.capacity(), 0);
        assert_eq!(rows.get_row(RowId::new(0)), Some(empty.as_slice()));
        assert_eq!(rows.get_row(RowId::new(1)), None);
    }

    #[test]
    fn append_limit_failure_does_not_advance_the_packed_counter() {
        let empty = row(&[]);
        let hash = shard_hash_values(ShardData::new(1), &empty).compact;
        let mut rows = SequenceRows::default();
        let writer = rows.parallel_writer(1, 0);

        let mut first = SequenceBatch::new(1, 0);
        first.insert(hash, &empty);
        assert_eq!(writer.append_batch(&first).0, RowId::new(0));

        let mut extra = SequenceBatch::new(1, 0);
        extra.insert(hash, &empty);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                writer.append_batch(&extra)
            }))
            .is_err()
        );

        rows = writer.finish(0, 0);
        assert_eq!(rows.physical_len(), 1);
        assert_eq!(rows.get_row(RowId::new(0)), Some(empty.as_slice()));
    }

    #[test]
    fn mixed_arity_insert_lookup_and_scan_include_empty_rows() {
        let mut table = SequenceTable::new();
        let initial = table.version();
        let expected = vec![
            row(&[]),
            row(&[1]),
            row(&[1, 2]),
            row(&[1, 2, 3, 4, 5]),
            vec![Value::stale()],
        ];
        let mut buffer = table.new_buffer();
        for sequence in &expected {
            buffer.stage_insert(sequence);
        }
        drop(buffer);

        let change = table.merge();
        assert!(change.added);
        assert!(!change.removed);
        assert_eq!(table.len(), expected.len());
        assert_eq!(table.version().major, initial.major);
        assert_eq!(table.version().minor.index(), expected.len());
        assert_eq!(scan_all(&table), expected);

        for (index, sequence) in expected.iter().enumerate() {
            let id = table.lookup(sequence).unwrap();
            assert_eq!(id, RowId::from_usize(index));
            assert_eq!(table.row(id), Some(sequence.as_slice()));
            let owned = table.get_row(sequence).unwrap();
            assert_eq!(owned.id, id);
            assert_eq!(&*owned.vals, sequence);
        }
        assert!(table.lookup(&row(&[1, 2, 3])).is_none());
        assert!(table.row(RowId::from_usize(expected.len())).is_none());
    }

    #[test]
    fn variable_keys_merge_fixed_width_value_tails() {
        empty_execution_state!(state);
        let mut table = SequenceTable::new_with_values(1, max_tail_merge());
        let mut writes = table.new_buffer();
        writes.stage_insert(&row(&[1, 10]));
        writes.stage_insert(&row(&[1, 30]));
        writes.stage_insert(&row(&[1, 20]));
        writes.stage_insert(&row(&[5])); // empty key, value 5
        writes.stage_insert(&row(&[9])); // same empty key, value 9
        drop(writes);

        let change = table.merge_with_state(&mut state);
        assert!(change.added);
        assert_eq!(table.len(), 2);
        assert_eq!(table.n_values(), 1);

        let keyed = table.lookup(&row(&[1])).unwrap();
        assert_eq!(
            table.get_row_ref(&row(&[1])),
            Some((keyed, row(&[1, 30]).as_slice()))
        );
        assert_eq!(table.get_values(&row(&[1])), Some(row(&[30]).as_slice()));
        assert_eq!(table.key(keyed), Some(row(&[1]).as_slice()));
        assert_eq!(table.values(keyed), Some(row(&[30]).as_slice()));
        assert_eq!(table.row(keyed), Some(row(&[1, 30]).as_slice()));
        let empty = table.lookup(&[]).unwrap();
        assert_eq!(table.key(empty), Some(&[][..]));
        assert_eq!(table.values(empty), Some(row(&[9]).as_slice()));
        assert!(table.lookup(&row(&[1, 30])).is_none());

        let mut split = Vec::new();
        table.scan_key_values(table.all().as_ref(), |_, key, values| {
            split.push((key.to_vec(), values.to_vec()));
        });
        split.sort();
        assert_eq!(split, vec![(row(&[]), row(&[9])), (row(&[1]), row(&[30]))]);

        table.new_buffer().stage_remove(&row(&[1]));
        let change = table.merge_with_state(&mut state);
        assert!(change.removed);
        assert!(!change.added);
        assert!(table.lookup(&row(&[1])).is_none());
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn value_index_tracks_selected_positions_across_append_delete_and_compaction() {
        empty_execution_state!(state);
        let mut table = SequenceTable::new_with_values_and_index(
            1,
            max_tail_merge(),
            Box::new(|key, visit| {
                let (_, payload) = key
                    .split_first()
                    .expect("test encoding always has a metadata header");
                for value in payload.iter().step_by(2) {
                    visit(*value);
                }
            }),
        );
        let first = row(&[99, 1, 2, 1, 10]);
        table.new_buffer().stage_insert(&first);
        assert!(table.merge_with_state(&mut state).added);

        assert!(table.rows_for_indexed_value(v(99)).is_none());
        assert!(table.rows_for_indexed_value(v(2)).is_none());
        let ones = table.rows_for_indexed_value(v(1)).unwrap();
        assert_eq!(ones.size(), 1, "repeated values index a row only once");
        let mut live = Vec::new();
        table.scan(ones, |row, _| live.push(row));
        assert_eq!(live.len(), 1);

        table.new_buffer().stage_remove(&first[..first.len() - 1]);
        assert!(table.merge_with_state(&mut state).removed);
        let ones = table.rows_for_indexed_value(v(1)).unwrap();
        assert_eq!(ones.size(), 1, "incremental indexes retain stale row ids");
        live.clear();
        table.scan(ones, |row, _| live.push(row));
        assert!(live.is_empty());

        let second = row(&[99, 1, 4, 7, 20]);
        table.new_buffer().stage_insert(&second);
        assert!(table.merge_with_state(&mut state).added);
        let ones = table.rows_for_indexed_value(v(1)).unwrap();
        assert_eq!(ones.size(), 2);
        live.clear();
        table.scan(ones, |row, _| live.push(row));
        assert_eq!(live.len(), 1);

        table.compact();
        let ones = table.rows_for_indexed_value(v(1)).unwrap();
        assert_eq!(ones.size(), 1);
        let mut cloned = table.clone();
        assert_eq!(cloned.rows_for_indexed_value(v(1)).unwrap().size(), 1);
        cloned.clear();
        assert!(cloned.rows_for_indexed_value(v(1)).is_none());
    }

    #[test]
    fn parallel_compaction_rebuilds_value_index_for_new_row_ids() {
        const ROWS: usize = 2_048;
        empty_execution_state!(state);
        ThreadPool::new(4).install(|| {
            let mut table = SequenceTable::new_with_values_and_index(
                1,
                max_tail_merge(),
                Box::new(|key, visit| key.iter().copied().for_each(visit)),
            );
            for index in 0..ROWS {
                table.new_buffer().stage_insert(&[
                    v(index % 8),
                    v(100_000 + index),
                    v(200_000 + index),
                ]);
            }
            table.merge_with_state(&mut state);
            for index in 0..1_000 {
                table
                    .new_buffer()
                    .stage_remove(&[v(index % 8), v(100_000 + index)]);
            }
            table.merge_with_state(&mut state);
            assert!(table.has_stale_rows());

            // Exercise the parallel implementation directly without making
            // this unit test allocate enough rows to cross the production
            // compaction heuristic.
            table.generation = table.generation.inc();
            table.parallel_compact();
            table.refresh_value_index();

            let indexed = table.rows_for_indexed_value(v(3)).unwrap();
            let mut found = 0;
            table.scan(indexed, |_, key_and_value| {
                assert_eq!(key_and_value[0], v(3));
                found += 1;
            });
            assert_eq!(found, (1_000..ROWS).filter(|index| index % 8 == 3).count());
        });
    }

    #[test]
    fn indexed_subset_rebuild_touches_only_candidate_rows() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        empty_execution_state!(state);
        let mut table = SequenceTable::new_with_values_and_index(
            1,
            max_tail_merge(),
            Box::new(|key, visit| key.iter().copied().for_each(visit)),
        );
        for complete_row in [row(&[1, 10]), row(&[2, 20]), row(&[3, 30])] {
            table.new_buffer().stage_insert(&complete_row);
        }
        table.merge_with_state(&mut state);

        let mut candidates = Subset::empty();
        table
            .rows_for_indexed_value(v(2))
            .unwrap()
            .offsets(|row| candidates.add_row_sorted(row));
        let visits = AtomicUsize::new(0);
        let change = table.rebuild_full_rows_subset_with_state(
            candidates.as_ref(),
            &|row, rebuilt| {
                visits.fetch_add(1, Ordering::Relaxed);
                rebuilt.extend_from_slice(row);
                rebuilt[0] = v(4);
                true
            },
            &mut state,
        );
        assert!(change.added && change.removed);
        assert_eq!(visits.load(Ordering::Relaxed), 1);
        assert!(table.lookup(&row(&[1])).is_some());
        assert!(table.lookup(&row(&[2])).is_none());
        assert!(table.lookup(&row(&[3])).is_some());
        assert_eq!(table.get_values(&row(&[4])), Some(row(&[20]).as_slice()));

        let mut live_twos = 0;
        table.scan(table.rows_for_indexed_value(v(2)).unwrap(), |_, _| {
            live_twos += 1
        });
        assert_eq!(live_twos, 0);
        let mut live_fours = 0;
        table.scan(table.rows_for_indexed_value(v(4)).unwrap(), |_, _| {
            live_fours += 1
        });
        assert_eq!(live_fours, 1);
    }

    #[test]
    fn parallel_value_tail_merges_coalesce_local_and_published_collisions() {
        const KEYS: usize = 10_000;
        empty_execution_state!(state);
        ThreadPool::new(4).install(|| {
            let mut table = SequenceTable::new_with_values(1, max_tail_merge());
            scope(|scope| {
                for copy in 0..3 {
                    for start in (0..KEYS).step_by(211) {
                        let mut buffer = table.new_buffer();
                        scope.spawn(move |_| {
                            for key in start..cmp::min(start + 211, KEYS) {
                                buffer.stage_insert(&row(&[key, copy * KEYS + key]));
                            }
                        });
                    }
                }
            });

            let pending = table.pending_state.detach();
            let (pending_rows, pending_values) = pending.pending_rows();
            assert!(table.parallel_insert(&pending, pending_rows, pending_values, Some(&state),));
            assert_eq!(table.len(), KEYS);
            for key in 0..KEYS {
                let row_id = table.lookup(&row(&[key])).unwrap();
                assert_eq!(
                    table.values(row_id),
                    Some(row(&[2 * KEYS + key]).as_slice())
                );
            }
        });
    }

    #[test]
    fn empty_merge_is_stable_and_clear_discards_pending_mutations() {
        let mut table = SequenceTable::new();
        let initial = table.version();
        let change = table.merge();
        assert!(!change.added);
        assert!(!change.removed);
        assert_eq!(table.version(), initial);

        let mut pending = table.new_buffer();
        pending.stage_insert(&row(&[]));
        pending.stage_insert(&row(&[1, 2, 3]));
        pending.stage_remove(&row(&[4]));
        drop(pending);
        {
            let current = table.pending_state.current.read().unwrap();
            assert_ne!(current.pending_ends.load(Ordering::Acquire), 0);
            assert_eq!(current.total_removals.load(Ordering::Acquire), 1);
        }

        let mut late = table.new_buffer();
        late.stage_insert(&row(&[9]));
        table.clear();
        drop(late);
        {
            let current = table.pending_state.current.read().unwrap();
            assert_eq!(current.pending_ends.load(Ordering::Acquire), 0);
            assert_eq!(current.total_removals.load(Ordering::Acquire), 0);
        }
        let change = table.merge();
        assert!(!change.added);
        assert!(!change.removed);
        assert!(table.is_empty());
        assert_eq!(table.version(), initial);
    }

    #[test]
    fn detached_epoch_releases_gate_before_parallel_merge() {
        const ROWS: usize = 20_000;
        ThreadPool::new(2).install(|| {
            let mut table = SequenceTable::new();
            let mut initial = table.new_buffer();
            for index in 0..ROWS {
                initial.stage_insert(&row(&[index, index + 1]));
            }
            drop(initial);

            let pending_state = table.pending_state.clone();
            let mut current = pending_state.current.write().unwrap();
            let ready = Arc::new(Barrier::new(3));
            let late = [row(&[ROWS + 1]), row(&[ROWS + 2])];
            scope(|scope| {
                for sequence in &late {
                    let mut buffer = table.new_buffer();
                    buffer.stage_insert(sequence);
                    let ready = ready.clone();
                    let pending_state = pending_state.clone();
                    scope.spawn(move |_| {
                        ready.wait();
                        let gate = pending_state.current.read().unwrap();
                        drop(gate);
                        drop(buffer);
                    });
                }
                ready.wait();
                let pending =
                    mem::replace(&mut *current, SequenceEpoch::new(pending_state.shard_data));
                drop(current);
                assert!(table.do_insert(&pending, None));
            });

            assert_eq!(table.len(), ROWS);
            assert!(late.iter().all(|sequence| table.lookup(sequence).is_none()));
            assert!(table.merge().added);
            assert!(late.iter().all(|sequence| table.lookup(sequence).is_some()));
        });
    }

    #[test]
    fn duplicate_delete_reinsert_and_updates_preserve_row_id_semantics() {
        let mut table = SequenceTable::new();
        let one = row(&[1]);
        let two = row(&[2, 2]);
        let mut inserts = table.new_buffer();
        inserts.stage_insert(&one);
        inserts.stage_insert(&two);
        inserts.stage_insert(&two);
        drop(inserts);
        assert!(table.merge().added);
        assert_eq!(table.len(), 2);

        let before_delete = table.version();
        table.new_buffer().stage_remove(&one);
        let change = table.merge();
        assert!(change.removed);
        assert!(!change.added);
        assert_eq!(table.len(), 1);
        assert!(table.has_stale_rows());
        assert!(table.lookup(&one).is_none());
        assert_eq!(scan_all(&table), vec![two.clone()]);
        let mut no_updates = Vec::new();
        table.scan(
            table.updates_since(before_delete.minor).as_ref(),
            |_, row| no_updates.push(row.to_vec()),
        );
        assert!(no_updates.is_empty());

        table.new_buffer().stage_insert(&one);
        assert!(table.merge().added);
        let new_id = table.lookup(&one).unwrap();
        assert_eq!(new_id.index(), before_delete.minor.index());
        let mut updates = Vec::new();
        table.scan(
            table.updates_since(before_delete.minor).as_ref(),
            |id, row| updates.push((id, row.to_vec())),
        );
        assert_eq!(updates, vec![(new_id, one)]);

        let before_replace = table.lookup(&two).unwrap();
        let mut replacement = table.new_buffer();
        replacement.stage_insert(&two);
        replacement.stage_remove(&two);
        drop(replacement);
        let change = table.merge();
        assert!(change.removed);
        assert!(change.added);
        assert_ne!(table.lookup(&two).unwrap(), before_replace);
    }

    #[test]
    fn pending_mutations_survive_clone() {
        let mut table = SequenceTable::new();
        table.new_buffer().stage_insert(&row(&[1, 2, 3]));
        table.merge();

        let mut pending = table.new_buffer();
        pending.stage_remove(&row(&[1, 2, 3]));
        pending.stage_insert(&row(&[]));
        pending.stage_insert(&row(&[4, 5]));
        drop(pending);

        let mut cloned = table.clone();
        table.merge();
        cloned.merge();
        for table in [&table, &cloned] {
            assert!(table.lookup(&row(&[1, 2, 3])).is_none());
            assert!(table.lookup(&row(&[])).is_some());
            assert!(table.lookup(&row(&[4, 5])).is_some());
            assert_eq!(table.len(), 2);
        }
    }

    struct CollapseValues;

    impl ValueRebuilder for CollapseValues {
        fn rebuild_val(&self, value: Value) -> Value {
            match value.index() {
                1 | 2 => v(0),
                4 => v(5),
                _ => value,
            }
        }
    }

    struct CanonicalizePayload;

    impl ValueRebuilder for CanonicalizePayload {
        fn rebuild_val(&self, value: Value) -> Value {
            match value.index() {
                100 => v(101),
                _ => value,
            }
        }
    }

    #[test]
    fn value_rebuild_coalesces_canonical_rows() {
        let mut table = SequenceTable::new();
        for values in [&[][..], &[1][..], &[2][..], &[3, 4][..]] {
            table.new_buffer().stage_insert(&row(values));
        }
        table.merge();

        let change = table.rebuild_values(&CollapseValues);
        assert!(change.removed);
        assert!(change.added);
        assert_eq!(table.len(), 3);
        assert!(table.lookup(&row(&[])).is_some());
        assert!(table.lookup(&row(&[0])).is_some());
        assert!(table.lookup(&row(&[3, 5])).is_some());
        assert!(table.lookup(&row(&[1])).is_none());
        assert!(table.lookup(&row(&[2])).is_none());
    }

    #[test]
    fn key_rebuild_preserves_and_merges_value_tails() {
        empty_execution_state!(state);
        let mut table = SequenceTable::new_with_values(1, max_tail_merge());
        table.new_buffer().stage_insert(&row(&[1, 100]));
        table.new_buffer().stage_insert(&row(&[2, 200]));
        table.new_buffer().stage_insert(&row(&[3, 50]));
        table.merge_with_state(&mut state);

        let change = table.rebuild_rows_with_state(
            &|key, rebuilt| {
                rebuilt.extend_from_slice(key);
                CollapseValues.rebuild_slice(rebuilt)
            },
            &mut state,
        );
        assert!(change.removed);
        assert!(change.added);
        assert_eq!(table.len(), 2);
        let collapsed = table.lookup(&row(&[0])).unwrap();
        assert_eq!(table.values(collapsed), Some(row(&[200]).as_slice()));
        let unchanged = table.lookup(&row(&[3])).unwrap();
        assert_eq!(table.values(unchanged), Some(row(&[50]).as_slice()));
    }

    #[test]
    fn full_value_rebuild_canonicalizes_the_non_key_tail() {
        empty_execution_state!(state);
        let mut table = SequenceTable::new_with_values(1, max_tail_merge());
        table.new_buffer().stage_insert(&row(&[7, 100]));
        table.merge_with_state(&mut state);

        let change = table.rebuild_values_with_state(&CanonicalizePayload, &mut state);
        assert!(change.removed);
        assert!(change.added);
        let row_id = table.lookup(&row(&[7])).unwrap();
        assert_eq!(table.key(row_id), Some(row(&[7]).as_slice()));
        assert_eq!(table.values(row_id), Some(row(&[101]).as_slice()));
    }

    #[test]
    fn full_row_rebuild_merges_colliding_keys_and_canonical_payloads() {
        empty_execution_state!(state);
        let mut table = SequenceTable::new_with_values(1, max_tail_merge());
        table.new_buffer().stage_insert(&row(&[1, 100]));
        table.new_buffer().stage_insert(&row(&[2, 200]));
        table.merge_with_state(&mut state);

        let change = table.rebuild_full_rows_with_state(
            &|row, rebuilt| {
                rebuilt.push(v(0));
                rebuilt.push(match sequence_values(row, 1)[0].index() {
                    100 => v(10),
                    200 => v(20),
                    unexpected => panic!("unexpected payload {unexpected}"),
                });
                true
            },
            &mut state,
        );
        assert!(change.removed);
        assert!(change.added);
        assert_eq!(table.len(), 1);
        let row_id = table.lookup(&row(&[0])).unwrap();
        assert_eq!(table.row(row_id), Some(row(&[0, 20]).as_slice()));
    }

    #[test]
    fn row_rebuild_can_change_lengths_and_collide_with_empty() {
        let mut table = SequenceTable::new();
        for values in [&[][..], &[7][..], &[8, 8][..], &[9][..]] {
            table.new_buffer().stage_insert(&row(values));
        }
        table.merge();

        let change = table.rebuild_rows(&|old, out| match old {
            [value] if *value == v(7) => {
                out.extend_from_slice(&[v(7), v(70), v(700)]);
                true
            }
            [value] if *value == v(9) => true,
            _ => false,
        });
        assert!(change.removed);
        assert!(change.added);
        assert_eq!(table.len(), 3);
        assert!(table.lookup(&row(&[])).is_some());
        assert!(table.lookup(&row(&[7, 70, 700])).is_some());
        assert!(table.lookup(&row(&[8, 8])).is_some());
        assert!(table.lookup(&row(&[7])).is_none());
        assert!(table.lookup(&row(&[9])).is_none());
    }

    #[test]
    fn compact_repoints_entries_and_invalidates_row_ids() {
        let mut table = SequenceTable::new();
        let rows = (0..64)
            .map(|index| {
                (0..=index % 7)
                    .map(|offset| v(index * 11 + offset))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for row in &rows {
            table.new_buffer().stage_insert(row);
        }
        table.merge();
        let old_generation = table.version().major;
        for row in &rows[..31] {
            table.new_buffer().stage_remove(row);
        }
        // 31/64 stays below automatic compaction's strict majority threshold.
        assert!(table.merge().removed);
        assert!(table.has_stale_rows());

        table.compact();
        assert_eq!(table.version().major, old_generation.inc());
        assert_eq!(table.rows.physical_len(), rows.len() - 31);
        assert_eq!(table.rows.stale_rows, 0);
        for row in &rows[..31] {
            assert!(table.lookup(row).is_none());
        }
        for row in &rows[31..] {
            let id = table.lookup(row).unwrap();
            assert_eq!(table.row(id), Some(row.as_slice()));
        }
    }

    #[test]
    fn deleting_large_sequence_compacts_by_stale_value_pressure() {
        let mut table = SequenceTable::new();
        let large = (0..1_024).map(v).collect::<Vec<_>>();
        table.new_buffer().stage_insert(&large);
        for value in 2_000..2_019 {
            table.new_buffer().stage_insert(&[v(value)]);
        }
        table.merge();
        let old_generation = table.version().major;

        table.new_buffer().stage_remove(&large);
        assert!(table.merge().removed);

        // Only one of twenty rows was removed, but nearly all packed values
        // were stale, so value pressure (not row pressure) compacted storage.
        assert_eq!(table.version().major, old_generation.inc());
        assert_eq!(table.rows.physical_len(), 19);
        assert_eq!(table.rows.values.len(), 19);
        assert_eq!(table.rows.stale_rows, 0);
        assert_eq!(table.rows.stale_values, 0);
    }

    #[test]
    fn compact_hash_collision_survives_sequence_lifecycle() {
        let shard_data = ShardData::new(16);
        let left = vec![Value::new(70_759_468), Value::new(2_418_062_243)];
        let right = vec![Value::new(3_427_223_041), Value::new(1_840_803_043)];
        assert_eq!(
            shard_hash_values(shard_data, &left),
            shard_hash_values(shard_data, &right)
        );

        let mut table = SequenceTable::new();
        table.hash = ShardedHashTable::with_shards(16);
        table.pending_state = Arc::new(SequencePendingState::new(shard_data));
        table.new_buffer().stage_insert(&left);
        table.new_buffer().stage_insert(&right);
        table.merge();
        assert!(table.lookup(&left).is_some());
        assert!(table.lookup(&right).is_some());

        table.new_buffer().stage_remove(&right);
        table.merge();
        assert!(table.lookup(&left).is_some());
        assert!(table.lookup(&right).is_none());
        table.compact();
        table.new_buffer().stage_insert(&right);
        table.merge();
        assert!(table.lookup(&left).is_some());
        assert!(table.lookup(&right).is_some());
    }

    fn parallel_row(index: usize) -> Vec<Value> {
        let len = 1 + index % 17;
        (0..len)
            .map(|offset| v(index.wrapping_mul(37).wrapping_add(offset)))
            .collect()
    }

    #[test]
    fn parallel_insert_delete_and_compact_variable_rows() {
        const ROWS: usize = 20_000;
        ThreadPool::new(4).install(|| {
            let mut table = SequenceTable::new();
            scope(|scope| {
                // Duplicate every row from a distinct producer buffer so the
                // shard-local coalescer sees cross-publication contention.
                for _copy in 0..2 {
                    for start in (0..ROWS).step_by(257) {
                        let mut buffer = table.new_buffer();
                        scope.spawn(move |_| {
                            for index in start..cmp::min(start + 257, ROWS) {
                                buffer.stage_insert(&parallel_row(index));
                            }
                        });
                    }
                }
            });
            let pending = table.pending_state.detach();
            let (pending_rows, pending_values) = pending.pending_rows();
            assert!(table.parallel_insert(&pending, pending_rows, pending_values, None));
            assert_eq!(table.len(), ROWS);
            for index in 0..ROWS {
                assert!(table.lookup(&parallel_row(index)).is_some());
            }

            scope(|scope| {
                for start in (0..ROWS).step_by(257) {
                    let mut buffer = table.new_buffer();
                    scope.spawn(move |_| {
                        for index in (start..cmp::min(start + 257, ROWS)).filter(|x| x % 2 == 0) {
                            buffer.stage_remove(&parallel_row(index));
                        }
                    });
                }
            });
            let pending = table.pending_state.detach();
            assert_eq!(pending.pending_removals(), ROWS / 2);
            assert!(table.parallel_delete(&pending));
            assert_eq!(table.len(), ROWS / 2);
            assert_eq!(table.rows.stale_rows, ROWS / 2);

            table.generation = table.generation.inc();
            table.parallel_compact();
            assert_eq!(table.rows.physical_len(), ROWS / 2);
            assert_eq!(table.rows.stale_rows, 0);
            for index in 0..ROWS {
                assert_eq!(table.lookup(&parallel_row(index)).is_some(), index % 2 == 1);
            }
        });
    }

    #[test]
    fn parallel_delete_adjacent_rows_from_different_shards() {
        const PAIRS: usize = 2_000;
        const SHARDS: usize = 16;
        ThreadPool::new(4).install(|| {
            let mut table = SequenceTable::new();
            table.hash = ShardedHashTable::with_shards(SHARDS);
            let shard_data = table.hash.shard_data();
            table.pending_state = Arc::new(SequencePendingState::new(shard_data));

            let mut next_value = 0usize;
            let mut find_row = |target: usize| loop {
                let candidate = row(&[next_value, next_value.wrapping_mul(37)]);
                next_value += 1;
                if shard_hash_values(shard_data, &candidate).shard.index() == target {
                    break candidate;
                }
            };
            let mut expected = Vec::with_capacity(PAIRS * 2);
            for _ in 0..PAIRS {
                expected.push(find_row(0));
                expected.push(find_row(SHARDS / 2));
            }

            table.rows.reserve(expected.len(), expected.len() * 2);
            for shard in table.hash.mut_shards() {
                shard.reserve(PAIRS, SequenceEntry::raw_probe_hash);
            }
            for values in &expected {
                let hash = shard_hash_values(shard_data, values);
                let location = table.rows.add_row_reserved(values);
                table.hash.mut_shards()[hash.shard.index()].insert_unique(
                    hash.compact.probe().raw(),
                    SequenceEntry {
                        hash: hash.compact,
                        row: location,
                    },
                    SequenceEntry::raw_probe_hash,
                );
            }

            let mut removals = table.new_buffer();
            for values in &expected {
                removals.stage_remove(values);
            }
            drop(removals);
            let pending = table.pending_state.detach();
            assert_eq!(pending.pending_removals(), expected.len());
            assert!(table.parallel_delete(&pending));
            assert_eq!(table.rows.stale_rows, expected.len());
            assert_eq!(table.rows.stale_values, expected.len() * 2);
            assert_eq!(table.len(), 0);
            assert!(expected.iter().all(|values| table.lookup(values).is_none()));
        });
    }
}
