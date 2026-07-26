//! A generic table implementation supporting sorted writes.
//!
//! The primary difference between this table and the `Function` implementation
//! in egglog is that high level concepts like "timestamp" and "merge function"
//! are abstracted away from the core functionality of the table.

use std::{
    any::Any,
    cmp,
    hash::Hasher,
    mem,
    sync::{
        Arc, Weak,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::numeric_id::{DenseIdMap, NumericId};
use crossbeam_queue::SegQueue;
use hashbrown::HashTable;
use rustc_hash::FxHasher;
use sharded_hash_table::ShardedHashTable;

use crate::{
    Pooled, TableChange, TableId,
    action::ExecutionState,
    common::{HashMap, ShardData, ShardId, SubsetTracker, Value},
    hash_index::{ColumnIndex, Index},
    offsets::{OffsetRange, Offsets, RowId, SortedOffsetVector, Subset, SubsetRef},
    parallel,
    parallel_heuristics::parallelize_table_op,
    pool::with_pool_set,
    row_buffer::{HashPartitioning, ParallelRowBufWriter, PartitionedRowBuffer, RowBuffer},
    table_spec::{
        ColumnId, Constraint, Generation, MutationBuffer, Offset, Row, Table, TableSpec,
        TableVersion,
    },
};

mod rebuild;
mod sharded_hash_table;
#[cfg(test)]
mod tests;

// NB: Having this type def lets us switch between 64 and 32 bits of hashcode.
//
// We should consider just using u64 everywhere though. Hashbrown doesn't play nicely with 32-bit
// hashcodes because it uses both the high and low bits of a 64-bit code.

type HashCode = u64;

// The lowest bits select buckets within a cache-sized destination-table
// window. The next bits group writes to those windows before a shard owner
// probes its HashTable.
const MUTATION_CACHE_WINDOW_BITS: u32 = 11;
const MUTATION_PARTITION_BITS: u32 = 6;
const PARALLEL_INSERT_BATCH_SIZE: usize = 1 << 12;

/// A pointer to a row in the table.
#[derive(Clone, Debug)]
pub(crate) struct TableEntry {
    hashcode: HashCode,
    row: RowId,
}

impl TableEntry {
    fn hashcode(&self) -> u64 {
        // We keep the cast here to make it easy to switch to HashCode=u32.
        #[allow(clippy::unnecessary_cast)]
        {
            self.hashcode as u64
        }
    }
}

/// The core data for a table.
///
/// This type is a thin wrapper around `RowBuffer`. The big difference is that
/// it keeps track of how many stale rows are present.
#[derive(Clone)]
struct Rows {
    data: RowBuffer,
    scratch: RowBuffer,
    stale_rows: usize,
}

impl Rows {
    fn new(data: RowBuffer) -> Rows {
        let arity = data.arity();
        Rows {
            data,
            scratch: RowBuffer::new(arity),
            stale_rows: 0,
        }
    }
    fn clear(&mut self) {
        self.data.clear();
        self.stale_rows = 0;
    }
    fn next_row(&self) -> RowId {
        RowId::from_usize(self.data.len())
    }
    fn set_stale(&mut self, row: RowId) {
        if !self.data.set_stale(row) {
            self.stale_rows += 1;
        }
    }

    fn get_row(&self, row: RowId) -> Option<&[Value]> {
        let row = self.data.get_row(row);
        if row[0].is_stale() { None } else { Some(row) }
    }

    /// A variant of `get_row` without bounds-checking on `row`.
    unsafe fn get_row_unchecked(&self, row: RowId) -> Option<&[Value]> {
        let row = unsafe { self.data.get_row_unchecked(row) };
        if row[0].is_stale() { None } else { Some(row) }
    }

    fn add_row(&mut self, row: &[Value]) -> RowId {
        if row[0].is_stale() {
            self.stale_rows += 1;
        }
        self.data.add_row(row)
    }

    fn remove_stale(&mut self, remap: impl FnMut(&[Value], RowId, RowId)) {
        self.data.remove_stale(remap);
        self.stale_rows = 0;
    }
}

/// The type of closures that are used to merge values in a [`SortedWritesTable`].
///
/// The first argument grants access to database using an [`ExecutionState`], the second argument
/// is the current value of the tuple. The third argument is the new, or "incoming" value of the
/// tuple. The fourth argument is a mutable reference to a vector that will be used to store the
/// output of the merge function _if_ it changes the value of the tuple. If it does not, then the
/// merge function should return `false`.
pub type MergeFn =
    dyn Fn(&mut ExecutionState, &[Value], &[Value], &mut Vec<Value>) -> bool + Send + Sync;

pub struct SortedWritesTable {
    generation: Generation,
    data: Rows,
    hash: ShardedHashTable<TableEntry>,

    n_keys: usize,
    n_columns: usize,
    sort_by: Option<ColumnId>,
    offsets: Vec<(Value, RowId)>,

    pending_state: Arc<PendingState>,
    merge: Arc<MergeFn>,
    to_rebuild: Vec<ColumnId>,
    rebuild_index: Index<ColumnIndex>,
    // Used to manage incremental rebuilds.
    subset_tracker: SubsetTracker,
}

impl Clone for SortedWritesTable {
    fn clone(&self) -> SortedWritesTable {
        SortedWritesTable {
            generation: self.generation,
            data: self.data.clone(),
            hash: self.hash.clone(),
            n_keys: self.n_keys,
            n_columns: self.n_columns,
            sort_by: self.sort_by,
            offsets: self.offsets.clone(),
            pending_state: Arc::new(self.pending_state.deep_copy()),
            merge: self.merge.clone(),
            to_rebuild: self.to_rebuild.clone(),
            rebuild_index: Index::new(self.to_rebuild.clone(), ColumnIndex::new()),
            subset_tracker: Default::default(),
        }
    }
}

struct Buffer {
    pending_rows: Option<PartitionedRowBuffer>,
    pending_removals: Option<PartitionedRowBuffer>,
    pending_known_removals: Option<PartitionedRowBuffer>,
    state: Weak<PendingState>,
    n_cols: u32,
    n_keys: u32,
    shard_data: ShardData,
    insert_partitioning: HashPartitioning,
    removal_partitioning: HashPartitioning,
}

impl MutationBuffer for Buffer {
    fn stage_insert(&mut self, row: &[Value]) {
        let (shard, hash) = hash_code(self.shard_data, row, self.n_keys as _);
        self.pending_rows
            .get_or_insert_with(|| {
                PartitionedRowBuffer::new(self.n_cols as _, self.insert_partitioning)
            })
            .add_row(shard.index(), hash, row);
    }
    fn stage_remove(&mut self, key: &[Value]) {
        let (shard, hash) = hash_code(self.shard_data, key, self.n_keys as _);
        self.pending_removals
            .get_or_insert_with(|| {
                PartitionedRowBuffer::new(self.n_keys as _, self.removal_partitioning)
            })
            .add_row(shard.index(), hash, key);
    }
    fn fresh_handle(&self) -> Box<dyn MutationBuffer> {
        Box::new(Buffer {
            pending_rows: None,
            pending_removals: None,
            pending_known_removals: None,
            state: self.state.clone(),
            n_cols: self.n_cols,
            n_keys: self.n_keys,
            shard_data: self.shard_data,
            insert_partitioning: self.insert_partitioning,
            removal_partitioning: self.removal_partitioning,
        })
    }
}

impl Buffer {
    fn stage_remove_row(&mut self, row: RowId, key: &[Value]) {
        let (shard, hashcode) = hash_code(self.shard_data, key, self.n_keys as _);
        self.pending_known_removals
            .get_or_insert_with(|| PartitionedRowBuffer::new(1, self.removal_partitioning))
            .add_row(shard.index(), hashcode, &[Value::new(row.rep())]);
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(state) = self.state.upgrade() {
            let rows = publish_partitioned(
                self.pending_rows.take(),
                &state.pending_rows,
                self.shard_data,
            );
            state.total_rows.fetch_add(rows, Ordering::Relaxed);

            let removals = publish_partitioned(
                self.pending_removals.take(),
                &state.pending_removals,
                self.shard_data,
            ) + publish_partitioned(
                self.pending_known_removals.take(),
                &state.pending_known_removals,
                self.shard_data,
            );
            state.total_removals.fetch_add(removals, Ordering::Relaxed);
        }
    }
}

fn publish_partitioned(
    buffer: Option<PartitionedRowBuffer>,
    queues: &DenseIdMap<ShardId, SegQueue<Arc<PartitionedRowBuffer>>>,
    shard_data: ShardData,
) -> usize {
    let Some(buffer) = buffer else {
        return 0;
    };
    let len = buffer.len();
    let buffer = Arc::new(buffer.seal());
    for shard_index in 0..shard_data.n_shards() {
        if !buffer.group_is_empty(shard_index) {
            queues[ShardId::from_usize(shard_index)].push(buffer.clone());
        }
    }
    len
}

impl Table for SortedWritesTable {
    fn dyn_clone(&self) -> Box<dyn Table> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clear(&mut self) {
        self.pending_state.clear();
        if self.data.data.len() == 0 {
            return;
        }
        self.offsets.clear();
        self.data.clear();
        self.hash.clear();
        self.generation = Generation::from_usize(self.version().major.index() + 1);
    }

    fn spec(&self) -> TableSpec {
        TableSpec {
            n_keys: self.n_keys,
            n_vals: self.n_columns - self.n_keys,
            uncacheable_columns: Default::default(),
            allows_delete: true,
        }
    }

    fn apply_rebuild(
        &mut self,
        table_id: TableId,
        table: &crate::WrappedTable,
        next_ts: Value,
        exec_state: &mut ExecutionState,
    ) -> bool {
        self.do_rebuild(table_id, table, next_ts, exec_state)
    }

    fn refresh_rows_for_values(&mut self, dirty_ids: &[Value], next_ts: Value) -> bool {
        SortedWritesTable::refresh_rows_for_values(self, dirty_ids, next_ts)
    }

    fn version(&self) -> TableVersion {
        TableVersion {
            major: self.generation,
            minor: Offset::from_usize(self.data.next_row().index()),
        }
    }

    fn updates_since(&self, offset: Offset) -> Subset {
        Subset::Dense(OffsetRange::new(
            RowId::from_usize(offset.index()),
            self.data.next_row(),
        ))
    }

    fn all(&self) -> Subset {
        Subset::Dense(OffsetRange::new(RowId::new(0), self.data.next_row()))
    }

    fn has_stale_rows(&self) -> bool {
        self.data.stale_rows > 0
    }

    fn len(&self) -> usize {
        self.data.data.len() - self.data.stale_rows
    }

    fn scan_generic(&self, subset: SubsetRef, mut f: impl FnMut(RowId, &[Value]))
    where
        Self: Sized,
    {
        let Some((_low, hi)) = subset.bounds() else {
            // Empty subset
            return;
        };
        assert!(
            hi.index() <= self.data.data.len(),
            "{} vs. {}",
            hi.index(),
            self.data.data.len()
        );
        if self.data.stale_rows == 0 {
            // Fast path: no stale rows, skip is_stale check per row.
            // SAFETY: subsets are sorted, low must be at most hi, and hi is less
            // than the length of the table.
            // TODO: provide a safe API for this `get_row_unchecked` usage since we have
            // checked the full bounds above.
            subset.offsets(|row| unsafe { f(row, self.data.data.get_row_unchecked(row)) })
        } else {
            // SAFETY: same as above.
            subset.offsets(|row| unsafe {
                if let Some(vals) = self.data.get_row_unchecked(row) {
                    f(row, vals)
                }
            })
        }
    }

    fn scan_generic_bounded(
        &self,
        subset: SubsetRef,
        start: Offset,
        n: usize,
        cs: &[Constraint],
        mut f: impl FnMut(RowId, &[Value]),
    ) -> Option<Offset>
    where
        Self: Sized,
    {
        let Some((_low, hi)) = subset.bounds() else {
            // Empty subset
            return None;
        };
        assert!(
            hi.index() <= self.data.data.len(),
            "{} vs. {}",
            hi.index(),
            self.data.data.len()
        );
        if cs.is_empty() {
            if self.data.stale_rows == 0 {
                // Fast path: no stale rows, skip bounds check and is_stale check.
                // SAFETY: all row IDs are in-bounds.
                subset
                    .iter_bounded(start.index(), start.index() + n, |row| {
                        let entry = unsafe { self.data.data.get_row_unchecked(row) };
                        f(row, entry);
                    })
                    .map(Offset::from_usize)
            } else {
                subset
                    .iter_bounded(start.index(), start.index() + n, |row| {
                        // SAFETY: all row IDs are in-bounds.
                        let Some(entry) = (unsafe { self.data.get_row_unchecked(row) }) else {
                            return;
                        };
                        f(row, entry);
                    })
                    .map(Offset::from_usize)
            }
        } else {
            subset
                .iter_bounded(start.index(), start.index() + n, |row| {
                    // SAFETY: all row IDs are in-bounds.
                    let Some(entry) = (unsafe { self.get_if_unchecked(cs, row) }) else {
                        return;
                    };
                    f(row, entry);
                })
                .map(Offset::from_usize)
        }
    }

    fn fast_subset(&self, constraint: &Constraint) -> Option<Subset> {
        let sort_by = self.sort_by?;
        match constraint {
            Constraint::Eq { .. } => None,
            Constraint::EqConst { col, val } => {
                if col == &sort_by {
                    match self.binary_search_sort_val(*val) {
                        Ok((found, bound)) => Some(Subset::Dense(OffsetRange::new(found, bound))),
                        Err(_) => Some(Subset::empty()),
                    }
                } else {
                    None
                }
            }
            Constraint::LtConst { col, val } => {
                if col == &sort_by {
                    match self.binary_search_sort_val(*val) {
                        Ok((found, _)) => {
                            Some(Subset::Dense(OffsetRange::new(RowId::new(0), found)))
                        }
                        Err(next) => Some(Subset::Dense(OffsetRange::new(RowId::new(0), next))),
                    }
                } else {
                    None
                }
            }
            Constraint::GtConst { col, val } => {
                if col == &sort_by {
                    match self.binary_search_sort_val(*val) {
                        Ok((_, bound)) => {
                            Some(Subset::Dense(OffsetRange::new(bound, self.data.next_row())))
                        }
                        Err(next) => {
                            Some(Subset::Dense(OffsetRange::new(next, self.data.next_row())))
                        }
                    }
                } else {
                    None
                }
            }
            Constraint::LeConst { col, val } => {
                if col == &sort_by {
                    match self.binary_search_sort_val(*val) {
                        Ok((_, bound)) => {
                            Some(Subset::Dense(OffsetRange::new(RowId::new(0), bound)))
                        }
                        Err(next) => Some(Subset::Dense(OffsetRange::new(RowId::new(0), next))),
                    }
                } else {
                    None
                }
            }
            Constraint::GeConst { col, val } => {
                if col == &sort_by {
                    match self.binary_search_sort_val(*val) {
                        Ok((found, _)) => {
                            Some(Subset::Dense(OffsetRange::new(found, self.data.next_row())))
                        }
                        Err(next) => {
                            Some(Subset::Dense(OffsetRange::new(next, self.data.next_row())))
                        }
                    }
                } else {
                    None
                }
            }
        }
    }

    fn refine_one(&self, mut subset: Subset, c: &Constraint) -> Subset {
        // NB: we aren't using any of the `fast_subset` tricks here. We may want
        // to if the higher-level implementations end up using it directly.
        subset.retain(|row| self.eval(std::slice::from_ref(c), row));
        subset
    }

    fn refine_ref(&self, subset: SubsetRef, cs: &[Constraint], _check_live: bool) -> Subset {
        // Single fused pass: `eval` reads each row once, checking liveness
        // (`get_row` skips stale rows) and all constraints together. A dense
        // input whose rows all survive stays dense.
        let n = subset.size();
        let mut vec: Pooled<SortedOffsetVector> = with_pool_set(|ps| ps.get());
        subset.offsets(|row| {
            if self.eval(cs, row) {
                // SAFETY: `offsets` visits rows in ascending order.
                unsafe { vec.push_unchecked(row) }
            }
        });
        if let SubsetRef::Dense(range) = subset
            && vec.slice().inner().len() == n
        {
            return Subset::Dense(range);
        }
        Subset::Sparse(vec)
    }

    fn new_buffer(&self) -> Box<dyn MutationBuffer> {
        Box::new(self.new_table_buffer())
    }

    fn merge(&mut self, exec_state: &mut ExecutionState) -> TableChange {
        let removed = self.do_delete();
        let added = self.do_insert(exec_state);
        self.maybe_rehash();
        TableChange { removed, added }
    }

    fn get_row(&self, key: &[Value]) -> Option<Row> {
        let id = get_entry(key, self.n_keys, &self.hash, |row| {
            &self.data.get_row(row).unwrap()[0..self.n_keys] == key
        })?;
        let mut vals = with_pool_set(|ps| ps.get::<Vec<Value>>());
        vals.extend_from_slice(self.data.get_row(id).unwrap());
        Some(Row { id, vals })
    }

    fn get_row_column(&self, key: &[Value], col: ColumnId) -> Option<Value> {
        let id = get_entry(key, self.n_keys, &self.hash, |row| {
            &self.data.get_row(row).unwrap()[0..self.n_keys] == key
        })?;
        Some(self.data.get_row(id).unwrap()[col.index()])
    }
}

impl SortedWritesTable {
    fn new_table_buffer(&self) -> Buffer {
        Buffer {
            pending_rows: None,
            pending_removals: None,
            pending_known_removals: None,
            state: Arc::downgrade(&self.pending_state),
            n_keys: u32::try_from(self.n_keys).expect("n_keys should fit in u32"),
            n_cols: u32::try_from(self.n_columns).expect("n_columns should fit in u32"),
            shard_data: self.hash.shard_data(),
            insert_partitioning: self.pending_state.insert_partitioning,
            removal_partitioning: self.pending_state.removal_partitioning,
        }
    }

    /// Create a new [`SortedWritesTable`] with the given number of keys,
    /// columns, and an optional sort column.
    ///
    /// The `merge_fn` is used to evaluate conflicts when more than one row is
    /// inserted with the same primary key. The old and new proposed values are
    /// passed as the second and third arguments, respectively, with the
    /// function filling the final argument with the contents of the new row.
    /// The return value indicates whether or not the contents of the vector
    /// should be used.
    ///
    /// Merge functions can access the database via [`ExecutionState`].
    pub fn new(
        n_keys: usize,
        n_columns: usize,
        sort_by: Option<ColumnId>,
        to_rebuild: Vec<ColumnId>,
        merge_fn: Box<MergeFn>,
    ) -> Self {
        let hash = ShardedHashTable::<TableEntry>::default();
        let shard_data = hash.shard_data();
        let rebuild_index = Index::new(to_rebuild.clone(), ColumnIndex::new());
        SortedWritesTable {
            generation: Generation::new(0),
            data: Rows::new(RowBuffer::new(n_columns)),
            hash,
            n_keys,
            n_columns,
            sort_by,
            offsets: Default::default(),
            pending_state: Arc::new(PendingState::new(shard_data, sort_by.is_some())),
            merge: merge_fn.into(),
            to_rebuild,
            rebuild_index,
            subset_tracker: Default::default(),
        }
    }

    /// Flush pending removals in coarse shard partitions. Each worker first
    /// probes and erases all entries in its partition, then marks the removed
    /// rows stale as a separate write-only pass.
    fn parallel_delete(&mut self) -> bool {
        let shard_data = self.hash.shard_data();
        let pending_removals = &self.pending_state.pending_removals;
        let pending_known_removals = &self.pending_state.pending_known_removals;
        let partitioning = self.pending_state.removal_partitioning;
        let partition_count = partitioning.partitions_per_group();
        let data = &self.data.data;
        let n_keys = self.n_keys;
        let stale_delta: usize =
            parallel::map_chunks_mut(self.hash.mut_shards(), |base_shard, shards| {
                let mut removed_rows = with_pool_set(|ps| ps.get::<Vec<RowId>>());
                for (shard_offset, shard) in shards.iter_mut().enumerate() {
                    let shard_id = ShardId::from_usize(base_shard + shard_offset);
                    let removal_buffers = drain_queue(&pending_removals[shard_id]);
                    for partition in 0..partition_count {
                        for buf in &removal_buffers {
                            for hashed in buf.partition(
                                partitioning.partition_index(shard_id.index(), partition),
                            ) {
                                let to_remove = hashed.row;
                                let hc = hashed.hash;
                                debug_assert_eq!(shard_data.shard_id(hc), shard_id);
                                debug_assert_eq!(buf.arity(), n_keys);
                                if let Ok(entry) = shard.find_entry(hc, |entry| {
                                    entry.hashcode == (hc as HashCode)
                                        && &data.get_row(entry.row)[0..n_keys] == to_remove
                                }) {
                                    let (ent, _) = entry.remove();
                                    removed_rows.push(ent.row);
                                }
                            }
                        }
                    }
                    let known_buffers = drain_queue(&pending_known_removals[shard_id]);
                    for partition in 0..partition_count {
                        for buf in &known_buffers {
                            for hashed in buf.partition(
                                partitioning.partition_index(shard_id.index(), partition),
                            ) {
                                let hc = hashed.hash;
                                let row = RowId::new(hashed.row[0].rep());
                                debug_assert_eq!(shard_data.shard_id(hc), shard_id);
                                debug_assert_eq!(buf.arity(), 1);
                                if let Ok(entry) = shard.find_entry(hc, |entry| {
                                    entry.hashcode == hc as HashCode && entry.row == row
                                }) {
                                    let (ent, _) = entry.remove();
                                    removed_rows.push(ent.row);
                                }
                            }
                        }
                    }
                }
                // SAFETY: A worker exclusively owns every shard in this
                // partition. Live rows have exactly one primary-key entry,
                // so successful removals yield disjoint rows across tasks.
                unsafe { mark_rows_stale(data, &removed_rows) };
                removed_rows.len()
            })
            .into_iter()
            .sum();
        self.data.stale_rows += stale_delta;
        stale_delta > 0
    }
    fn serial_delete(&mut self) -> bool {
        let shard_data = self.hash.shard_data();
        let partitioning = self.pending_state.removal_partitioning;
        let partition_count = partitioning.partitions_per_group();
        let mut changed = false;
        self.hash
            .mut_shards()
            .iter_mut()
            .enumerate()
            .for_each(|(shard_id, shard)| {
                let shard_id = ShardId::from_usize(shard_id);
                let removal_buffers = drain_queue(&self.pending_state.pending_removals[shard_id]);
                for partition in 0..partition_count {
                    for buf in &removal_buffers {
                        for hashed in
                            buf.partition(partitioning.partition_index(shard_id.index(), partition))
                        {
                            let to_remove = hashed.row;
                            let hc = hashed.hash;
                            debug_assert_eq!(shard_data.shard_id(hc), shard_id);
                            debug_assert_eq!(buf.arity(), self.n_keys);
                            if let Ok(entry) = shard.find_entry(hc, |entry| {
                                entry.hashcode == (hc as HashCode)
                                    && &self.data.get_row(entry.row).unwrap()[0..self.n_keys]
                                        == to_remove
                            }) {
                                let (ent, _) = entry.remove();
                                self.data.set_stale(ent.row);
                                changed = true;
                            }
                        }
                    }
                }
                let known_buffers =
                    drain_queue(&self.pending_state.pending_known_removals[shard_id]);
                for partition in 0..partition_count {
                    for buf in &known_buffers {
                        for hashed in
                            buf.partition(partitioning.partition_index(shard_id.index(), partition))
                        {
                            let hc = hashed.hash;
                            let row = RowId::new(hashed.row[0].rep());
                            debug_assert_eq!(shard_data.shard_id(hc), shard_id);
                            debug_assert_eq!(buf.arity(), 1);
                            if let Ok(entry) = shard.find_entry(hc, |entry| {
                                entry.hashcode == hc as HashCode && entry.row == row
                            }) {
                                let (ent, _) = entry.remove();
                                self.data.set_stale(ent.row);
                                changed = true;
                            }
                        }
                    }
                }
            });
        changed
    }

    fn do_delete(&mut self) -> bool {
        let total = self.pending_state.total_removals.swap(0, Ordering::Relaxed);

        if parallelize_table_op(total) {
            self.parallel_delete()
        } else {
            self.serial_delete()
        }
    }

    fn do_insert(&mut self, exec_state: &mut ExecutionState) -> bool {
        let total = self.pending_state.total_rows.swap(0, Ordering::Relaxed);
        self.data.data.reserve(total);
        if parallelize_table_op(total) {
            if let Some(col) = self.sort_by {
                self.parallel_insert(
                    exec_state,
                    SortChecker {
                        col,
                        current: None,
                        baseline: self.offsets.last().map(|(v, _)| *v),
                    },
                )
            } else {
                self.parallel_insert(exec_state, ())
            }
        } else {
            self.serial_insert(exec_state)
        }
    }

    fn serial_insert(&mut self, exec_state: &mut ExecutionState) -> bool {
        let mut changed = false;
        let n_keys = self.n_keys;
        let partitioning = self.pending_state.insert_partitioning;
        let partition_count = partitioning.partitions_per_group();
        let mut scratch = with_pool_set(|ps| ps.get::<Vec<Value>>());
        for (outer_shard, queue) in self.pending_state.pending_rows.iter() {
            let buffers = drain_queue(queue);
            let incoming = buffers
                .iter()
                .map(|buffer| buffer.group_len(outer_shard.index()))
                .sum();
            self.hash.mut_shards()[outer_shard.index()].reserve(incoming, TableEntry::hashcode);
            for partition in 0..partition_count {
                for buf in &buffers {
                    for hashed in
                        buf.partition(partitioning.partition_index(outer_shard.index(), partition))
                    {
                        let query = hashed.row;
                        if query.first().is_some_and(Value::is_stale) {
                            continue;
                        }
                        let hc = hashed.hash;
                        debug_assert_eq!(self.hash.shard_data().shard_id(hc), outer_shard);
                        let key = &query[0..n_keys];
                        use hashbrown::hash_table::Entry;
                        match self.hash.mut_shards()[outer_shard.index()].entry(
                            hc,
                            |entry| {
                                entry.hashcode == hc as HashCode
                                    && self
                                        .data
                                        .get_row(entry.row)
                                        .is_some_and(|row| &row[0..n_keys] == key)
                            },
                            TableEntry::hashcode,
                        ) {
                            Entry::Occupied(mut entry) => {
                                let old_row = entry.get().row;
                                let cur = self
                                    .data
                                    .get_row(old_row)
                                    .expect("table should not point to stale entry");
                                if (self.merge)(exec_state, cur, query, &mut scratch) {
                                    debug_assert_eq!(&scratch[0..n_keys], key);
                                    let new = self.data.add_row(&scratch);
                                    if let Some(sort_by) = self.sort_by {
                                        update_sort_offsets(sort_by, query, new, &mut self.offsets);
                                    }
                                    self.data.set_stale(old_row);
                                    entry.get_mut().row = new;
                                    changed = true;
                                }
                                scratch.clear();
                            }
                            Entry::Vacant(entry) => {
                                let new = self.data.add_row(query);
                                if let Some(sort_by) = self.sort_by {
                                    update_sort_offsets(sort_by, query, new, &mut self.offsets);
                                }
                                entry.insert(TableEntry {
                                    hashcode: hc as _,
                                    row: new,
                                });
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
        changed
    }

    fn parallel_insert<C: OrderingChecker>(
        &mut self,
        exec_state: &ExecutionState,
        checker: C,
    ) -> bool {
        // Parallel insert uses one giant parallel foreach. We have updates
        // pre-sharded, and one logical thread can process updates for each
        // shard independently. Updates happen in three phases, which comments
        // describe below.
        let shard_data = self.hash.shard_data();
        let n_keys = self.n_keys;
        let n_cols = self.n_columns;
        let next_offset = RowId::from_usize(self.data.data.len());
        let row_writer = self.data.data.parallel_writer();
        let pending_rows = &self.pending_state.pending_rows;
        let partitioning = self.pending_state.insert_partitioning;
        let partition_count = partitioning.partitions_per_group();
        let merge = self.merge.clone();
        let pending_adds = parallel::map_mut(self.hash.mut_shards(), |shard_id, shard| {
            let shard_id = ShardId::from_usize(shard_id);
            let mut checker = checker.clone();
            let mut exec_state = exec_state.clone();
            let mut scratch = with_pool_set(|ps| ps.get::<Vec<Value>>());
            let queue = &pending_rows[shard_id];
            let buffers = drain_queue(queue);
            let incoming = buffers
                .iter()
                .map(|buffer| buffer.group_len(shard_id.index()))
                .sum();
            shard.reserve(incoming, TableEntry::hashcode);
            let mut marked_stale = 0usize;
            let mut staged = CoalescedInsertBatch::new(n_keys, n_cols, PARALLEL_INSERT_BATCH_SIZE);
            let mut changed = false;
            // Flush after a bounded number of staged physical rows or at a
            // destination-table partition boundary.
            macro_rules! flush_insert_batch {
                    () => {{
                        if !staged.is_empty() {
                        // Phase 2: Write the staged rows to the row writer. This only
                        // works due to the `ParallelRowBufWriter` machinery.
                        let (start_row, stale) = staged.write_output(&row_writer);
                        marked_stale += stale;
                        // Phase 3: With the values buffered in the row buffer, we can
                        // write them back to the shard, pointed to the correct rows.

                        // In the serial implementation, we do phases 2 and 3 inline with
                        // processing the incoming mutation, but separating them out
                        // this way allows us to do a single write to the shared row
                        // buffer, rather than one per row, which would cause
                        // contention.
                        let mut cur_row = start_row;
                        let read_handle = row_writer.read_handle();
                        for (hc, row) in staged.rows_hashed() {
                            if row.first().is_some_and(Value::is_stale) {
                                cur_row = cur_row.inc();
                                continue;
                            }
                            use hashbrown::hash_table::Entry;
                            let key = &row[0..n_keys];
                            #[cfg(any(debug_assertions, test))]
                            {
                                unsafe {
                                    // read the value we wrote at this row and
                                    // check that it matches.
                                    assert_eq!(read_handle.get_row_unchecked(cur_row), row);
                                }
                            }
                            debug_assert_eq!(shard_data.shard_id(hc), shard_id);
                            match shard.entry(
                                hc,
                                // SAFETY: `ent` must point to a valid row
                                |ent| unsafe {
                                    ent.hashcode == hc as HashCode
                                        && &read_handle.get_row_unchecked(ent.row)[0..n_keys] == key
                                },
                                TableEntry::hashcode,
                            ) {
                                Entry::Occupied(mut occ) => {
                                    // SAFETY: `occ` must point to a valid row: we only insert valid rows
                                    // into the map.
                                    let old_row = occ.get().row;
                                    let merged = {
                                        let cur =
                                            unsafe { read_handle.get_row_unchecked(old_row) };
                                        (merge)(&mut exec_state, cur, row, &mut scratch)
                                    };

                                    // SAFETY: The safety requirements of
                                    // `replace_row_shared` and `set_stale_shared`
                                    // are that there are no concurrent accesses
                                    // to either row. `cur_row` is a completed,
                                    // uniquely reserved write that has not yet
                                    // been published, and we have exclusive
                                    // access to every existing row whose hash
                                    // maps to this shard.
                                    if merged {
                                        debug_assert_eq!(
                                            &scratch[0..n_keys],
                                            key,
                                            "merge functions must preserve table keys"
                                        );
                                        checker.check_local(&scratch);
                                        unsafe {
                                            read_handle.replace_row_shared(cur_row, &scratch);
                                        }
                                        occ.get_mut().row = cur_row;
                                        unsafe {
                                            let _was_stale =
                                                read_handle.set_stale_shared(old_row);
                                            debug_assert!(!_was_stale);
                                        }
                                        changed = true;
                                    } else {
                                        // Mark the new row as stale: we didn't end up needing it.
                                        unsafe {
                                            let _was_stale = read_handle.set_stale_shared(cur_row);
                                            debug_assert!(!_was_stale);
                                        }
                                    }
                                    marked_stale += 1;
                                    scratch.clear();
                                }
                                Entry::Vacant(v) => {
                                    checker.check_local(row);
                                    changed = true;
                                    v.insert(TableEntry {
                                        hashcode: hc as HashCode,
                                        row: cur_row,
                                    });
                                }
                            }

                            cur_row = cur_row.inc();
                        }
                        staged.clear();
                        }
                    }};
                }
            // Phase 1: process all incoming updates:
            // * Coalesce rows with the same key in a bounded temporary table.
            // * Copy the staged rows to the shared row buffer in one
            //   reservation.
            // * Apply each live row to the exclusively owned destination
            //   shard in `flush_insert_batch`.
            for partition in 0..partition_count {
                for buf in &buffers {
                    for hashed in
                        buf.partition(partitioning.partition_index(shard_id.index(), partition))
                    {
                        staged.insert_hashed(hashed.hash, hashed.row, |cur, new, out| {
                            (merge)(&mut exec_state, cur, new, out)
                        });
                        if staged.physical_len() >= PARALLEL_INSERT_BATCH_SIZE {
                            flush_insert_batch!();
                        }
                    }
                }
                // Do not carry rows into the next destination-table window.
                flush_insert_batch!();
            }
            (checker, marked_stale, changed)
        });
        self.data.data = row_writer.finish();
        // Now we just need to reset our invariants.

        // Confirm none of the writes violated sort order and update the
        // `offsets` vector.
        let checker = C::check_global(pending_adds.iter().map(|(checker, _, _)| checker));
        checker.update_offsets(next_offset, &mut self.offsets);

        // Update the staleness counters.
        self.data.stale_rows += pending_adds
            .iter()
            .map(|(_, stale, _)| *stale)
            .sum::<usize>();

        // Register any changes.
        pending_adds.iter().any(|(_, _, changed)| *changed)
    }

    fn binary_search_sort_val(&self, val: Value) -> Result<(RowId, RowId), RowId> {
        debug_assert!(
            self.offsets.windows(2).all(|x| x[0].1 < x[1].1),
            "{:?}",
            self.offsets
        );

        debug_assert!(
            self.offsets.windows(2).all(|x| x[0].0 < x[1].0),
            "{:?}",
            self.offsets
        );
        match self.offsets.binary_search_by_key(&val, |(v, _)| *v) {
            Ok(got) => Ok((
                self.offsets[got].1,
                self.offsets
                    .get(got + 1)
                    .map(|(_, r)| *r)
                    .unwrap_or(self.data.next_row()),
            )),
            Err(next) => Err(self
                .offsets
                .get(next)
                .map(|(_, id)| *id)
                .unwrap_or(self.data.next_row())),
        }
    }
    fn eval(&self, cs: &[Constraint], row: RowId) -> bool {
        self.get_if(cs, row).is_some()
    }

    fn eval_constraints(cs: &[Constraint], row: &[Value]) -> bool {
        cs.iter().all(|constraint| match constraint {
            Constraint::Eq { l_col, r_col } => row[l_col.index()] == row[r_col.index()],
            Constraint::EqConst { col, val } => row[col.index()] == *val,
            Constraint::LtConst { col, val } => row[col.index()] < *val,
            Constraint::GtConst { col, val } => row[col.index()] > *val,
            Constraint::LeConst { col, val } => row[col.index()] <= *val,
            Constraint::GeConst { col, val } => row[col.index()] >= *val,
        })
    }

    unsafe fn get_if_unchecked(&self, cs: &[Constraint], row: RowId) -> Option<&[Value]> {
        let row = unsafe { self.data.get_row_unchecked(row) }?;
        if Self::eval_constraints(cs, row) {
            Some(row)
        } else {
            None
        }
    }

    fn get_if(&self, cs: &[Constraint], row: RowId) -> Option<&[Value]> {
        let row = self.data.get_row(row)?;
        if Self::eval_constraints(cs, row) {
            Some(row)
        } else {
            None
        }
    }

    fn maybe_rehash(&mut self) {
        if self.data.stale_rows <= cmp::max(16, self.data.data.len() / 2) {
            return;
        }

        if parallelize_table_op(self.data.data.len()) {
            self.parallel_rehash();
        } else {
            self.rehash();
        }
    }
    fn parallel_rehash(&mut self) {
        // Parallel rehashes go "hash-first" rather than "rows-first".
        //
        // We iterate over each shard and then write out new contents to a fresh row, in parallel.
        self.generation = self.generation.inc();
        let Some(sort_by) = self.sort_by else {
            self.parallel_rehash_unsorted();
            return;
        };
        assert!(!self.offsets.is_empty());
        struct TimestampStats {
            value: Value,
            count: usize,
            histogram: Pooled<DenseIdMap<ShardId, usize>>,
        }
        impl Default for TimestampStats {
            fn default() -> TimestampStats {
                TimestampStats {
                    value: Value::stale(),
                    count: 0,
                    histogram: with_pool_set(|ps| ps.get()),
                }
            }
        }
        let mut results = Vec::<TimestampStats>::with_capacity(self.offsets.len());
        let offset_windows = self
            .offsets
            .windows(2)
            .map(|xs| {
                let [(start_val, start_row), (_, end_row)] = xs else {
                    unreachable!()
                };
                (*start_val, *start_row, *end_row)
            })
            .collect::<Vec<_>>();
        // Use a macro rather than a lambda to avoid borrow issues.
        macro_rules! compute_hist {
            ($start_val: expr, $start_row: expr, $end_row: expr) => {{
                let mut histogram: Pooled<DenseIdMap<ShardId, usize>> =
                    with_pool_set(|ps| ps.get());
                let mut cur_row = $start_row;
                let mut count = 0;
                while cur_row < $end_row {
                    if let Some(row) = self.data.get_row(cur_row) {
                        count += 1;
                        let (shard, _) = hash_code(self.hash.shard_data(), row, self.n_keys);
                        *histogram.get_or_default(shard) += 1;
                    }
                    cur_row = cur_row.inc();
                }
                TimestampStats {
                    value: $start_val,
                    count,
                    histogram,
                }
            }};
        }
        results.extend(parallel::map(
            &offset_windows,
            |_, (start_val, start_row, end_row)| compute_hist!(*start_val, *start_row, *end_row),
        ));
        // And here we handle the final one.
        let (start_val, start_row) = self.offsets.last().unwrap();
        let end_row = self.data.next_row();
        let last = compute_hist!(*start_val, *start_row, end_row);
        results.push(last);
        // Now we need to compute cumulative statistics on the row layouts here.
        // We do this serially a we currently don't have a ton of use for cases with thousands
        // of timestamps or more. There are well-known parallel algorithms for computing these
        // cumulative statistics in parallel, but they are not currently a good fit here.
        let mut prev_count = 0;
        self.offsets.clear();
        for stats in results.iter_mut() {
            if stats.count == 0 {
                continue;
            }
            self.offsets
                .push((stats.value, RowId::from_usize(prev_count)));
            let mut inner = prev_count;
            for (_, count) in stats.histogram.iter_mut() {
                // Each entry in the histogram now points to the start row for that shard's
                // rows for a given timestamp.
                let tmp = *count;
                *count = inner;
                inner += tmp;
            }
            prev_count += stats.count;
            debug_assert_eq!(inner, prev_count)
        }

        // Now the part with some unsafe code.
        // We will iterate over each shard and use the statistics in `results` to guide where
        // each row will go.
        //
        // This involves doing unsynchronized writes to the table (ptr::copy_nonoverlapping)
        // followed by a set_len. The safety of these operations relies on the fact that:
        // * No one grabs a reference to the interior of `scratch` until these operations have
        //   finished.
        // * `scratch` does not overlap `data`.
        // * The sharding function completely partitions the set of objects in the table: one
        //   shard's writes will never stomp on those of another.

        self.data.scratch.clear();
        self.data.scratch.reserve(prev_count);
        let scratch_ptr = self.data.scratch.raw_rows() as usize;
        let data = &self.data.data;
        let n_columns = self.n_columns;
        parallel::for_each_mut(self.hash.mut_shards(), |shard_id, shard| {
            let shard_id = ShardId::from_usize(shard_id);
            let scratch_ptr = scratch_ptr as *const Value;
            let mut progress = HashMap::<Value /* timestamp */, RowId /* next row */>::default();
            progress.reserve(results.len());
            for stats in &results {
                let Some(start) = stats.histogram.get(shard_id) else {
                    continue;
                };
                progress.insert(stats.value, RowId::from_usize(*start));
            }
            for TableEntry { row: row_id, .. } in shard.iter_mut() {
                let row = data.get_row(*row_id);
                debug_assert!(!row[0].is_stale(), "shard should not map to a stale value");
                let val = row[sort_by.index()];
                let next = progress[&val];
                // SAFETY: see above longer comment.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        row.as_ptr(),
                        scratch_ptr.add(next.index() * n_columns) as *mut Value,
                        n_columns,
                    )
                }
                *row_id = next;
                progress.insert(val, next.inc());
            }
        });
        // SAFETY: see above longer comment.
        unsafe { self.data.scratch.set_len(prev_count) };
        mem::swap(&mut self.data.data, &mut self.data.scratch);
        self.data.stale_rows = 0;
    }

    /// Compact an unsorted table by walking its live hash entries rather than
    /// scanning the row store, which may be mostly stale.
    ///
    /// Every live row has exactly one entry in exactly one hash shard. Shard
    /// lengths therefore give both the live-row count and a cheap histogram
    /// for assigning each worker a disjoint output range.
    fn parallel_rehash_unsorted(&mut self) {
        debug_assert!(self.sort_by.is_none());
        debug_assert!(self.offsets.is_empty());

        let shards = self.hash.mut_shards();
        let mut shard_offsets = Vec::with_capacity(shards.len() + 1);
        shard_offsets.push(0usize);
        for shard in shards.iter() {
            let next = shard_offsets
                .last()
                .copied()
                .unwrap()
                .checked_add(shard.len())
                .expect("live row count should fit in usize");
            shard_offsets.push(next);
        }
        let live_rows = *shard_offsets.last().unwrap();
        debug_assert_eq!(
            live_rows,
            self.data.data.len() - self.data.stale_rows,
            "each live row should have exactly one hash entry"
        );

        self.data.scratch.clear();
        self.data.scratch.reserve(live_rows);
        let scratch_ptr = self.data.scratch.raw_rows() as usize;
        let data = &self.data.data;
        let n_columns = self.n_columns;

        // Workers mutate disjoint hash shards and copy into the corresponding
        // disjoint half-open ranges in `scratch`. The raw pointer is passed as
        // an integer because raw pointers are not Sync.
        parallel::for_each_mut(shards, |shard_index, shard| {
            let scratch_ptr = scratch_ptr as *mut Value;
            let mut next = shard_offsets[shard_index];
            let end = shard_offsets[shard_index + 1];
            for TableEntry { row: row_id, .. } in shard.iter_mut() {
                let row = data.get_row(*row_id);
                debug_assert!(
                    !row[0].is_stale(),
                    "a hash entry should never point to a stale row"
                );
                let new_row = RowId::from_usize(next);

                // SAFETY:
                // * `scratch` was reserved once before any worker started and
                //   is not resized until all workers have joined.
                // * The prefix offsets give each shard a disjoint destination
                //   range large enough for all of its entries.
                // * `data` and `scratch` are distinct row buffers, so source
                //   and destination cannot overlap.
                // * No references into the uninitialized part of `scratch`
                //   are created before `set_len` below.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        row.as_ptr(),
                        scratch_ptr.add(next * n_columns),
                        n_columns,
                    );
                }
                *row_id = new_row;
                next += 1;
            }
            debug_assert_eq!(next, end);
        });

        // SAFETY: every row up to `live_rows` was initialized exactly once by
        // the shard workers above, and the parallel scope has joined.
        unsafe { self.data.scratch.set_len(live_rows) };
        mem::swap(&mut self.data.data, &mut self.data.scratch);
        self.data.stale_rows = 0;
    }

    fn rehash_impl(
        sort_by: Option<ColumnId>,
        n_keys: usize,
        rows: &mut Rows,
        offsets: &mut Vec<(Value, RowId)>,
        hash: &mut ShardedHashTable<TableEntry>,
    ) {
        if let Some(sort_by) = sort_by {
            offsets.clear();
            rows.remove_stale(|row, old, new| {
                let stale_entry = get_entry_mut(row, n_keys, hash, |x| x == old)
                    .expect("non-stale entry not mapped in hash");
                *stale_entry = new;
                let sort_col = row[sort_by.index()];
                if let Some((max, _)) = offsets.last() {
                    if sort_col > *max {
                        offsets.push((sort_col, new));
                    }
                } else {
                    offsets.push((sort_col, new));
                }
            })
        } else {
            rows.remove_stale(|row, old, new| {
                let stale_entry = get_entry_mut(row, n_keys, hash, |x| x == old)
                    .expect("non-stale entry not mapped in hash");
                *stale_entry = new;
            })
        }
    }

    fn rehash(&mut self) {
        self.generation = self.generation.inc();
        Self::rehash_impl(
            self.sort_by,
            self.n_keys,
            &mut self.data,
            &mut self.offsets,
            &mut self.hash,
        )
    }
}

fn get_entry(
    row: &[Value],
    n_keys: usize,
    table: &ShardedHashTable<TableEntry>,
    test: impl Fn(RowId) -> bool,
) -> Option<RowId> {
    let (shard, hash) = hash_code(table.shard_data(), row, n_keys);
    table
        .get_shard(shard)
        .find(hash, |ent| {
            ent.hashcode == hash as HashCode && test(ent.row)
        })
        .map(|ent| ent.row)
}

fn get_entry_mut<'a>(
    row: &[Value],
    n_keys: usize,
    table: &'a mut ShardedHashTable<TableEntry>,
    test: impl Fn(RowId) -> bool,
) -> Option<&'a mut RowId> {
    let (shard, hash) = hash_code(table.shard_data(), row, n_keys);
    table.mut_shards()[shard.index()]
        .find_mut(hash, |ent| {
            ent.hashcode == hash as HashCode && test(ent.row)
        })
        .map(|ent| &mut ent.row)
}

fn hash_code(shard_data: ShardData, row: &[Value], n_keys: usize) -> (ShardId, u64) {
    let mut hasher = FxHasher::default();
    for val in &row[0..n_keys] {
        hasher.write_usize(val.index());
    }
    let full_code = hasher.finish();
    // We keep this cast here to allow for experimenting with HashCode=u32.
    #[allow(clippy::unnecessary_cast)]
    (shard_data.shard_id(full_code), full_code as HashCode as u64)
}

fn update_sort_offsets(
    sort_by: ColumnId,
    row: &[Value],
    row_id: RowId,
    offsets: &mut Vec<(Value, RowId)>,
) {
    let sort_val = row[sort_by.index()];
    if let Some(largest) = offsets.last().map(|(value, _)| *value) {
        assert!(
            sort_val >= largest,
            "inserting row that violates sort order ({sort_val:?} vs. {largest:?})"
        );
        if sort_val > largest {
            offsets.push((sort_val, row_id));
        }
    } else {
        offsets.push((sort_val, row_id));
    }
}

fn mutation_partitioning(shard_data: ShardData, ordered: bool) -> HashPartitioning {
    if ordered || shard_data.n_shards() == 1 {
        HashPartitioning::grouped(0, 0, shard_data.n_shards())
    } else {
        HashPartitioning::grouped(
            MUTATION_CACHE_WINDOW_BITS,
            MUTATION_PARTITION_BITS,
            shard_data.n_shards(),
        )
    }
}

fn drain_queue<T>(queue: &SegQueue<T>) -> Vec<T> {
    let mut drained = Vec::new();
    while let Some(item) = queue.pop() {
        drained.push(item);
    }
    drained
}

/// # Safety
/// The caller must ensure exclusive access to every row in `rows`.
unsafe fn mark_rows_stale(data: &RowBuffer, rows: &[RowId]) {
    for row in rows {
        let was_stale = unsafe { data.set_stale_shared(*row) };
        debug_assert!(!was_stale);
    }
}

/// A simple struct for packaging up pending mutations to a `SortedWritesTable`.
struct PendingState {
    pending_rows: DenseIdMap<ShardId, SegQueue<Arc<PartitionedRowBuffer>>>,
    pending_removals: DenseIdMap<ShardId, SegQueue<Arc<PartitionedRowBuffer>>>,
    pending_known_removals: DenseIdMap<ShardId, SegQueue<Arc<PartitionedRowBuffer>>>,
    total_removals: AtomicUsize,
    total_rows: AtomicUsize,
    insert_partitioning: HashPartitioning,
    removal_partitioning: HashPartitioning,
}

impl PendingState {
    fn new(shard_data: ShardData, ordered: bool) -> PendingState {
        let n_shards = shard_data.n_shards();
        let insert_partitioning = mutation_partitioning(shard_data, ordered);
        let removal_partitioning = mutation_partitioning(shard_data, false);
        let mut pending_rows = DenseIdMap::with_capacity(n_shards);
        let mut pending_removals = DenseIdMap::with_capacity(n_shards);
        let mut pending_known_removals = DenseIdMap::with_capacity(n_shards);
        for i in 0..n_shards {
            pending_rows.insert(ShardId::from_usize(i), SegQueue::default());
            pending_removals.insert(ShardId::from_usize(i), SegQueue::default());
            pending_known_removals.insert(ShardId::from_usize(i), SegQueue::default());
        }

        PendingState {
            pending_rows,
            pending_removals,
            pending_known_removals,
            total_removals: AtomicUsize::new(0),
            total_rows: AtomicUsize::new(0),
            insert_partitioning,
            removal_partitioning,
        }
    }
    fn clear(&self) {
        for (_, queue) in self.pending_rows.iter() {
            while queue.pop().is_some() {}
        }

        for (_, queue) in self.pending_removals.iter() {
            while queue.pop().is_some() {}
        }

        for (_, queue) in self.pending_known_removals.iter() {
            while queue.pop().is_some() {}
        }
        self.total_rows.store(0, Ordering::Relaxed);
        self.total_removals.store(0, Ordering::Relaxed);
    }

    /// This is only really used in debugging, but it's annoying enough to write
    /// that it may help to have around.
    ///
    /// We also, however, use it in the clone impl (which should only be called when pending state
    /// is empty).
    fn deep_copy(&self) -> PendingState {
        let mut pending_rows = DenseIdMap::new();
        let mut pending_removals = DenseIdMap::new();
        let mut pending_known_removals = DenseIdMap::new();
        fn drain_queue<T>(queue: &SegQueue<T>) -> Vec<T> {
            let mut res = Vec::new();
            while let Some(x) = queue.pop() {
                res.push(x);
            }
            res
        }
        for (shard, queue) in self.pending_rows.iter() {
            let contents = drain_queue(queue);
            let new_queue = SegQueue::default();
            for x in contents {
                new_queue.push(x.clone());
                queue.push(x);
            }
            pending_rows.insert(shard, new_queue);
        }

        for (shard, queue) in self.pending_removals.iter() {
            let contents = drain_queue(queue);
            let new_queue = SegQueue::default();
            for x in contents {
                new_queue.push(x.clone());
                queue.push(x);
            }
            pending_removals.insert(shard, new_queue);
        }

        for (shard, queue) in self.pending_known_removals.iter() {
            let contents = drain_queue(queue);
            let new_queue = SegQueue::default();
            for x in contents {
                new_queue.push(x.clone());
                queue.push(x);
            }
            pending_known_removals.insert(shard, new_queue);
        }

        PendingState {
            pending_rows,
            pending_removals,
            pending_known_removals,
            total_removals: AtomicUsize::new(self.total_removals.load(Ordering::Acquire)),
            total_rows: AtomicUsize::new(self.total_rows.load(Ordering::Acquire)),
            insert_partitioning: self.insert_partitioning,
            removal_partitioning: self.removal_partitioning,
        }
    }
}

/// A trait that encapsulates the logic of potentially checking that written
/// columns appear in sorted order.
///
/// For rows that are sorted by a column, an OrderingChecker asserts that all
/// new rows have the same value in that column, and that the column is greater
/// than or equal to the column value coming in. For rows not sorted, these
/// checks become no-ops.
trait OrderingChecker: Clone + Send + Sync {
    /// Check any invariants locally, updating the state of the checker when
    /// doing so.
    fn check_local(&mut self, row: &[Value]);
    /// Combine the states of multiple checkers, returning a new checker with
    /// all information assimilated. This is the checker that is suitable for
    /// calling `update_offsets` with.
    fn check_global<'a>(checkers: impl Iterator<Item = &'a Self>) -> Self
    where
        Self: 'a;
    /// Update the sorted offset vector with the current state of the checker.
    fn update_offsets(&self, start: RowId, offsets: &mut Vec<(Value, RowId)>);
}

impl OrderingChecker for () {
    fn check_local(&mut self, _: &[Value]) {}
    fn check_global<'a>(_: impl Iterator<Item = &'a ()>) {}
    fn update_offsets(&self, _: RowId, _: &mut Vec<(Value, RowId)>) {}
}

#[derive(Copy, Clone)]
struct SortChecker {
    col: ColumnId,
    baseline: Option<Value>,
    current: Option<Value>,
}

impl OrderingChecker for SortChecker {
    fn check_local(&mut self, row: &[Value]) {
        let val = row[self.col.index()];
        if let Some(cur) = self.current {
            assert_eq!(
                cur, val,
                "concurrently inserting rows with different sort keys"
            );
        } else {
            self.current = Some(val);
            if let Some(baseline) = self.baseline {
                assert!(val >= baseline, "inserted row violates sort order");
            }
        }
    }

    fn check_global<'a>(mut checkers: impl Iterator<Item = &'a Self>) -> Self {
        let Some(start) = checkers.next() else {
            return SortChecker {
                col: ColumnId::new(!0),
                baseline: None,
                current: None,
            };
        };
        let mut expected = start.current;
        for checker in checkers {
            assert_eq!(checker.baseline, start.baseline);
            match (&mut expected, checker.current) {
                (None, None) => {}
                (cur @ None, Some(x)) => {
                    *cur = Some(x);
                }
                (Some(_), None) => {}
                (Some(x), Some(y)) => {
                    assert_eq!(
                        *x, y,
                        "concurrently inserting rows with different sort keys"
                    );
                }
            }
        }
        SortChecker {
            col: start.col,
            baseline: start.baseline,
            current: expected,
        }
    }

    fn update_offsets(&self, start: RowId, offsets: &mut Vec<(Value, RowId)>) {
        if let Some(cur) = self.current {
            if let Some((max, _)) = offsets.last() {
                if cur > *max {
                    offsets.push((cur, start));
                }
            } else {
                offsets.push((cur, start));
            }
        }
    }
}

/// A bounded, cached-hash coalescer for pending insertion rows.
///
/// Multiple rows with the same key are merged before the surviving value is
/// combined with the destination table's current value. Physical rows and
/// their unchanged full hashes stay aligned, including stale intermediate
/// merge outputs.
struct CoalescedInsertBatch {
    n_keys: usize,
    hash: Pooled<HashTable<TableEntry>>,
    rows: RowBuffer,
    hashes: Vec<HashCode>,
    n_stale: usize,
    scratch: Pooled<Vec<Value>>,
}

impl CoalescedInsertBatch {
    fn rows_hashed(&self) -> impl Iterator<Item = (u64, &[Value])> {
        debug_assert_eq!(self.hashes.len(), self.rows.len());
        self.hashes
            .iter()
            .copied()
            .zip(self.rows.iter())
            .map(|(hash, row)| {
                // Keep this conversion at the boundary so HashCode can still
                // be changed independently.
                #[allow(clippy::unnecessary_cast)]
                let hash = hash as u64;
                (hash, row)
            })
    }

    fn new(n_keys: usize, n_cols: usize, capacity: usize) -> Self {
        let mut res = with_pool_set(|ps| CoalescedInsertBatch {
            n_keys,
            hash: ps.get(),
            rows: RowBuffer::new(n_cols),
            hashes: Vec::with_capacity(capacity),
            n_stale: 0,
            scratch: ps.get(),
        });
        res.hash
            .reserve(capacity, |entry| coalesced_hash(entry.hashcode()));
        res.rows.reserve(capacity);
        res
    }

    fn clear(&mut self) {
        self.hash.clear();
        self.rows.clear();
        self.hashes.clear();
        self.n_stale = 0;
    }

    fn is_empty(&self) -> bool {
        self.rows.len() == 0
    }

    fn physical_len(&self) -> usize {
        self.rows.len()
    }

    fn insert_hashed(
        &mut self,
        hc: u64,
        row: &[Value],
        mut merge_fn: impl FnMut(&[Value], &[Value], &mut Vec<Value>) -> bool,
    ) {
        if row.first().is_some_and(Value::is_stale) {
            return;
        }

        debug_assert_eq!(self.hashes.len(), self.rows.len());
        use hashbrown::hash_table::Entry;
        let entry = self.hash.entry(
            coalesced_hash(hc),
            |entry| {
                entry.hashcode() == hc
                    && self.rows.get_row(entry.row)[0..self.n_keys] == row[0..self.n_keys]
            },
            |entry| coalesced_hash(entry.hashcode()),
        );
        match entry {
            Entry::Occupied(mut occupied) => {
                let cur = self.rows.get_row(occupied.get().row);
                if merge_fn(cur, row, &mut self.scratch) {
                    debug_assert_eq!(
                        &self.scratch[0..self.n_keys],
                        &row[0..self.n_keys],
                        "merge functions must preserve table keys"
                    );
                    let new = self.rows.add_row(&self.scratch);
                    self.hashes.push(hc as HashCode);
                    self.rows.set_stale(occupied.get().row);
                    self.n_stale += 1;
                    occupied.get_mut().row = new;
                }
                self.scratch.clear();
            }
            Entry::Vacant(vacant) => {
                let next = self.rows.add_row(row);
                self.hashes.push(hc as HashCode);
                vacant.insert(TableEntry {
                    hashcode: hc as HashCode,
                    row: next,
                });
            }
        }
        debug_assert_eq!(self.hashes.len(), self.rows.len());
    }

    /// Append all physical rows, returning the first RowId and stale count.
    fn write_output(&self, output: &ParallelRowBufWriter) -> (RowId, usize) {
        debug_assert_eq!(self.hashes.len(), self.rows.len());
        (output.append_contents(&self.rows), self.n_stale)
    }
}

#[inline]
fn coalesced_hash(hash: u64) -> u64 {
    // Cache partitioning fixes a run of middle bits in the full hash. Mix
    // higher bits down before probing the temporary table so those fixed bits
    // do not artificially restrict its usable buckets.
    hash ^ hash.wrapping_shr(17)
}
