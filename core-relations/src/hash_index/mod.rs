//! Hash-based secondary indexes.
use std::{
    hash::{Hash, Hasher},
    mem,
    sync::Mutex,
};

use hashbrown::HashTable;
use numeric_id::{define_id, IdVec, NumericId};
use once_cell::sync::Lazy;
use rayon::iter::ParallelIterator;
use rustc_hash::FxHasher;

use crate::{
    common::{HashMap, IndexMap, ShardData, ShardId, Value},
    offsets::{RowId, SortedOffsetSlice, SubsetRef},
    pool::{with_pool_set, Pooled},
    row_buffer::{RowBuffer, TaggedRowBuffer},
    table_spec::{ColumnId, Generation, Offset, TableVersion, WrappedTableRef},
    OffsetRange, Subset,
};

#[cfg(test)]
mod tests;

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
    pub(crate) fn get_subset<'a>(&'a self, key: &'a TI::Key) -> Option<SubsetRef<'a>> {
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
        let subset = if cur_version.major != self.updated_to.major {
            self.table.clear();
            table.all()
        } else {
            table.updates_since(self.updated_to.minor)
        };
        if do_parallel(subset.size()) {
            self.table.merge_parallel(&self.key, table, subset.as_ref());
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

    pub(crate) fn for_each(&self, f: impl FnMut(&TI::Key, SubsetRef)) {
        self.table.for_each(f);
    }

    pub(crate) fn len(&self) -> usize {
        self.table.len()
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
    fn get_subset(&self, key: &Self::Key) -> Option<SubsetRef>;
    /// Add the given key and row id to the table.
    fn add_row(&mut self, key: &Self::WriteKey, row: RowId);
    /// Merge the contents of the [`TaggedRowBuffer`] into the table.
    fn merge_rows(&mut self, buf: &TaggedRowBuffer);
    /// Call `f` over the elements of the index.
    fn for_each(&self, f: impl FnMut(&Self::Key, SubsetRef));
    /// The number of keys in the index.
    fn len(&self) -> usize;

    fn merge_parallel(&mut self, cols: &[ColumnId], table: WrappedTableRef, subset: SubsetRef);
}

struct ColumnIndexShard {
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
        // SAFETY: everything in `table` comes from `subsets`.
        for key in vals {
            let shard = self.shard_data.get_shard_mut(key, &mut self.shards);
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
    fn for_each(&self, mut f: impl FnMut(&Self::Key, SubsetRef)) {
        for (subsets, (k, v)) in self
            .shards
            .iter()
            .flat_map(|(_, shard)| shard.table.iter().map(|x| (&shard.subsets, x)))
        {
            f(k, v.as_ref(subsets));
        }
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
            for (row_id, keys) in buf.non_stale() {
                for key in keys {
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

        THREAD_POOL.install(|| {
            rayon::scope(|scope| {
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
            self.shards.par_iter_mut().for_each(|(shard_id, shard)| {
                use indexmap::map::Entry;
                // Sort the vector by start row id to ensure we populate subsets in sorted order.
                let mut vec = queues[shard_id].lock().unwrap();
                vec.sort_by_key(|(start, _)| *start);
                for (_, buf) in vec.drain(..) {
                    for (row_id, key) in buf.non_stale() {
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
            entry.hash == hash && shard.table.keys.get_row(entry.key) == key
        })?;
        Some(entry.vals.as_ref(&shard.subsets))
    }

    fn add_row(&mut self, key: &[Value], row: RowId) {
        use hashbrown::hash_table::Entry;
        let hash = hash_key(key);
        let shard = &mut self.shards[self.shard_data.shard_id(hash)];
        let table_entry = shard.table.hash.entry(
            hash,
            |entry| entry.hash == hash && shard.table.keys.get_row(entry.key) == key,
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
    fn for_each(&self, mut f: impl FnMut(&Self::Key, SubsetRef)) {
        for (_, shard) in self.shards.iter() {
            for entry in shard.table.hash.iter() {
                let key = shard.table.keys.get_row(entry.key);
                f(key, entry.vals.as_ref(&shard.subsets));
            }
        }
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
            for (row_id, key) in buf.non_stale() {
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
        THREAD_POOL.install(|| {
            rayon::scope(|scope| {
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
            self.shards.par_iter_mut().for_each(|(shard_id, shard)| {
                use hashbrown::hash_table::Entry;
                // Sort the vector by start row id to ensure we populate subsets in sorted order.
                let mut vec = queues[shard_id].lock().unwrap();
                vec.sort_by_key(|(start, _)| *start);
                for (_, buf) in vec.drain(..) {
                    for (row_id, key) in buf.non_stale() {
                        let hash = hash_key(key);
                        let table_entry = shard.table.hash.entry(
                            hash,
                            |entry| {
                                entry.hash == hash && shard.table.keys.get_row(entry.key) == key
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

fn hash_key(key: &[Value]) -> u64 {
    let mut hasher = FxHasher::default();
    key.hash(&mut hasher);
    hasher.finish()
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
        assert!(
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
/// Business logic in this module probably shouldn't call clone explicitly.
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
    let n_threads = rayon::current_num_threads();
    if n_threads == 1 {
        1
    } else {
        n_threads * 2
    }
}

fn do_parallel(_workload_size: usize) -> bool {
    #[cfg(test)]
    {
        use rand::Rng;
        rand::thread_rng().gen::<bool>()
    }
    #[cfg(not(test))]
    {
        rayon::current_num_threads() > 1 && _workload_size > 20_000
    }
}

/// A thread pool specifically for parallel hash index construction.
///
/// We use a separate thread pool here because callers can construct an index under a lock,
/// and we do not want to take a long-running lock in the global thread pool without another
/// way to get parallelism.
///
/// Earlier solutions using rayon::yield_now() were unreliable.
static THREAD_POOL: Lazy<rayon::ThreadPool> = Lazy::new(|| {
    rayon::ThreadPoolBuilder::new()
        .num_threads(rayon::current_num_threads())
        .build()
        .unwrap()
});

/// A simple free list used to reuse slots in a [`SubsetBuffer`].
///
/// This free list works as a map from power-of-two size classes to a vector of offsets that point
/// to the beginning of an unused vector.
#[derive(Default, Clone)]
pub(super) struct FreeList {
    data: HashMap<usize, Vec<BufferIndex>>,
}
impl FreeList {
    fn get_size_class(&mut self, size: usize) -> &mut Vec<BufferIndex> {
        let size_class = size.next_power_of_two();
        self.data.entry(size_class).or_default()
    }
}
