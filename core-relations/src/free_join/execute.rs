//! Core free join execution.

use std::{
    cmp, iter, mem,
    ops::Range,
    sync::{
        Arc, OnceLock, RwLock,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::{
    common::{HashMap, HashSet, IndexMap},
    free_join::plan::{JoinStages, MatId, MatScanMode, MatSpec},
    numeric_id::{DenseIdMap, IdVec, NumericId},
    query::Atom,
    row_buffer::{RowBuffer, SmallValueVec},
};
use crossbeam::utils::CachePadded;
use dashmap::mapref::entry::Entry;
use dashmap::mapref::one::RefMut;
use egglog_concurrency::Scope;
use egglog_reports::{ReportLevel, RuleReport, RuleSetReport};
use smallvec::SmallVec;
use web_time::Instant;

use crate::{
    Constraint, OffsetRange, Pool, SubsetRef,
    action::{Bindings, ExecutionState},
    common::{DashMap, Value},
    free_join::{
        frame_update::{FrameUpdates, UpdateInstr},
        get_index_from_tableinfo,
    },
    hash_index::{ColumnIndex, Index, IndexBase, TupleIndex},
    offsets::{Offsets, RowId, SortedOffsetSlice, SortedOffsetVector, Subset},
    parallel_heuristics::{
        MIN_TOP_INDEX_KEYS_PER_WORKER, action_batch_size, free_join_fork_depth,
        parallelize_db_level_op,
    },
    pool::Pooled,
    query::RuleSet,
    row_buffer::TaggedRowBuffer,
    table_spec::{ColumnId, Offset, WrappedTableRef},
};

use super::{
    ActionId, AtomId, Database, HashColumnIndex, HashIndex, TableId, TableInfo, Variable,
    get_column_index_from_tableinfo,
    plan::{JoinHeader, JoinStage, Plan},
    with_pool_set,
};

const SMALL_RESIDUAL: usize = 8;

fn top_index_shape_is_eligible(
    workers: usize,
    leader_keys: usize,
    nonempty_shards: usize,
    min_keys_per_worker: usize,
) -> bool {
    workers > 1
        && leader_keys >= min_keys_per_worker.saturating_mul(workers)
        && nonempty_shards >= workers
}

struct SparseColumnIndex {
    n_keys: usize,
    n_subsets: usize,
    keys: [Value; SMALL_RESIDUAL],
    offsets: [usize; SMALL_RESIDUAL],
    subset_ids: [RowId; SMALL_RESIDUAL],
}

/// Return a SubsetRef for the given range of rows in a SparseColumnIndex.
/// Single-row ranges become Dense to skip pool allocation in to_owned.
///
/// # Safety
/// `ids[range]` must be sorted in non-decreasing order. The wider `ids` slice
/// need not be sorted as a whole; only the indicated sub-range. This is the
/// invariant of `SortedOffsetSlice::new_unchecked`.
#[inline]
unsafe fn sparse_subset_ref(ids: &[RowId], range: Range<usize>) -> SubsetRef<'_> {
    if range.len() == 1 {
        let row = ids[range.start];
        SubsetRef::Dense(OffsetRange::new(row, row.inc()))
    } else {
        // SAFETY: caller guarantees `ids[range]` is sorted.
        SubsetRef::Sparse(unsafe { SortedOffsetSlice::new_unchecked(&ids[range]) })
    }
}

impl SparseColumnIndex {
    fn keys(&self) -> &[Value] {
        &self.keys[..self.n_keys]
    }

    fn get_offset_for(&self, i: usize) -> Range<usize> {
        let lo = self.offsets[i];
        let hi = if i + 1 < self.n_keys {
            self.offsets[i + 1]
        } else {
            self.n_subsets
        };
        lo..hi
    }

    fn new(table: WrappedTableRef<'_>, subset: SubsetRef<'_>, col: ColumnId) -> Self {
        let mut rows = [(Value::new_const(0), RowId::new_const(0)); SMALL_RESIDUAL];
        let mut pos = 0;
        table.for_each_col(subset, col, &mut |row_id, val| {
            rows[pos] = (val, row_id);
            pos += 1;
        });
        let n_subsets = pos;

        rows[..pos].sort_unstable();

        let mut n_keys = 0;
        let mut keys = [Value::new_const(0); SMALL_RESIDUAL];
        let mut offsets = [0; SMALL_RESIDUAL];
        let mut subset_ids = [RowId::new_const(0); SMALL_RESIDUAL];
        offsets[0] = 0;

        for (i, &(key, row_id)) in rows[..n_subsets].iter().enumerate() {
            let is_new_key = n_keys == 0 || keys[n_keys - 1] != key;
            if is_new_key {
                offsets[n_keys] = i;
                keys[n_keys] = key;
                n_keys += 1;
            }
            subset_ids[i] = row_id;
        }

        SparseColumnIndex {
            n_keys,
            n_subsets,
            keys,
            offsets,
            subset_ids,
        }
    }

    fn get_subset(&self, key: Value) -> Option<SubsetRef<'_>> {
        if self.n_keys == 0 {
            return None;
        }
        let found = self.keys().binary_search(&key).ok()?;
        let range = self.get_offset_for(found);
        // SAFETY: `subset_ids` was populated from rows sorted by (Value, RowId),
        // so RowIds within any single per-key range (as returned by
        // `get_offset_for`) are in non-decreasing order.
        Some(unsafe { sparse_subset_ref(&self.subset_ids, range) })
    }

    fn for_each(&self, mut f: impl FnMut(&[Value], SubsetRef)) {
        if self.n_keys == 0 {
            return;
        }
        for i in 0..self.n_keys {
            let range = self.get_offset_for(i);
            // SAFETY: see `get_subset` — each per-key range of `subset_ids` is sorted.
            let subset = unsafe { sparse_subset_ref(&self.subset_ids, range) };
            f(&self.keys[i..i + 1], subset);
        }
    }

    fn len(&self) -> usize {
        self.n_keys
    }
}

/// Return a `SubsetRef` for `ids[range]`, which must be nonempty and sorted
/// ascending. A contiguous run is returned as `Dense` to avoid a pool
/// allocation when the subset is later materialized.
///
/// # Safety
/// `ids[range]` must be sorted in non-decreasing order.
#[inline]
unsafe fn dense_or_sparse_ref(ids: &[RowId], range: Range<usize>) -> SubsetRef<'_> {
    let slice = &ids[range];
    let first = slice[0];
    let last = slice[slice.len() - 1];
    if last.index() - first.index() == slice.len() - 1 {
        SubsetRef::Dense(OffsetRange::new(first, last.inc()))
    } else {
        // SAFETY: caller guarantees `slice` is sorted.
        SubsetRef::Sparse(unsafe { SortedOffsetSlice::new_unchecked(slice) })
    }
}

/// A heap-allocated, sort-based single-column index for on-the-fly (per-subset)
/// indexing during joins.
///
/// Unlike a hash-based column index, the (value -> rows) groups live in sorted
/// arrays: `for_each` walks them directly and `get_subset` binary-searches the
/// keys. Building it therefore skips hash-table construction, which is wasteful
/// for the high-cardinality columns joined on in an e-graph, where each value
/// typically maps to only one or two rows.
pub(crate) struct SortedColumnIndex {
    /// Distinct column values (ascending) paired with the start offset of their
    /// rows in `row_ids`. A trailing `(_, row_ids.len())` sentinel delimits the
    /// final group.
    keys: Vec<(Value, u32)>,
    /// Row ids grouped by key; each group is ascending.
    row_ids: Vec<RowId>,
}

impl SortedColumnIndex {
    fn build_for_subset(table: WrappedTableRef, subset: SubsetRef, col: ColumnId) -> Self {
        let mut pairs: Vec<(Value, RowId)> = Vec::new();
        // Rows arrive in RowId-ascending order, so a value-stable sort leaves
        // each value's rows ascending.
        table.collect_col_pairs(subset, col, &mut pairs);
        let mut scratch = vec![(Value::new_const(0), RowId::new_const(0)); pairs.len()];
        crate::hash_index::radix_sort_slice_by_value(&mut pairs, &mut scratch);
        drop(scratch);

        let mut keys: Vec<(Value, u32)> = Vec::new();
        let mut row_ids: Vec<RowId> = Vec::with_capacity(pairs.len());
        for (val, row) in pairs {
            if keys.last().map(|&(v, _)| v) != Some(val) {
                keys.push((val, row_ids.len() as u32));
            }
            row_ids.push(row);
        }
        keys.push((Value::new_const(0), row_ids.len() as u32));
        SortedColumnIndex { keys, row_ids }
    }

    fn get_subset(&self, key: Value) -> Option<SubsetRef<'_>> {
        // The trailing sentinel is never a real match: it is stored as value 0
        // but the search space excludes it via the `len - 1` bound below.
        let n = self.len();
        let i = self.keys[..n]
            .binary_search_by_key(&key, |&(v, _)| v)
            .ok()?;
        let lo = self.keys[i].1 as usize;
        let hi = self.keys[i + 1].1 as usize;
        // SAFETY: rows within a single key's range are ascending (see `build_for_subset`).
        Some(unsafe { dense_or_sparse_ref(&self.row_ids, lo..hi) })
    }

    fn for_each(&self, mut f: impl FnMut(Value, SubsetRef)) {
        let n = self.len();
        for i in 0..n {
            let (val, lo) = self.keys[i];
            let hi = self.keys[i + 1].1 as usize;
            // SAFETY: see `get_subset`.
            let subset = unsafe { dense_or_sparse_ref(&self.row_ids, lo as usize..hi) };
            f(val, subset);
        }
    }

    fn len(&self) -> usize {
        // The last entry is the sentinel offset, not a key.
        self.keys.len().saturating_sub(1)
    }
}

/// A table-index slot retained for one `run_rule_set` call.
///
/// The slot lazily acquires its Arc through the existing fully-refreshing
/// catalog helper on first cached use. Keeping that Arc in an execution-scoped
/// sidecar removes catalog lookups and refcount traffic from recursive join
/// execution without constructing indexes for plan accesses that choose a
/// residual-local strategy at runtime. Initialized slots are dropped before
/// the database resets its indexes during `merge_all`.
enum PreparedIndexSlot {
    Tuple(OnceLock<HashIndex>),
    Column(OnceLock<HashColumnIndex>),
    /// The table specification forbids a global cache for at least one key
    /// column, so execution must use its existing dynamic-index path.
    Uncacheable,
}

fn columns_are_cacheable(info: &TableInfo, cols: &[ColumnId]) -> bool {
    cols.iter().all(|col| {
        !info
            .spec
            .uncacheable_columns
            .get(*col)
            .copied()
            .unwrap_or(false)
    })
}

/// Index handles for one immutable [`JoinStages`] value, positionally aligned
/// with `JoinStages::instrs` and with each stage's scans.
struct PreparedJoinIndexes {
    stages: Box<[SmallVec<[PreparedIndexSlot; 4]>]>,
}

impl PreparedJoinIndexes {
    fn new(db: &Database, atoms: &Arc<DenseIdMap<AtomId, Atom>>, stages: &JoinStages) -> Self {
        let stages = stages
            .instrs
            .iter()
            .map(|stage| {
                let mut handles = SmallVec::new();
                match stage {
                    JoinStage::Intersect { scans, .. } => {
                        handles.extend(scans.iter().map(|scan| {
                            let info = &db.tables[atoms[scan.atom].table];
                            if !columns_are_cacheable(info, &[scan.column]) {
                                PreparedIndexSlot::Uncacheable
                            } else {
                                PreparedIndexSlot::Column(OnceLock::new())
                            }
                        }));
                    }
                    JoinStage::FusedIntersect { to_intersect, .. }
                    | JoinStage::FusedIntersectMat { to_intersect, .. } => {
                        handles.extend(to_intersect.iter().map(|(scan, _)| {
                            let cols = scan.to_index.vars.as_slice();
                            let info = &db.tables[atoms[scan.to_index.atom].table];
                            if !columns_are_cacheable(info, cols) {
                                PreparedIndexSlot::Uncacheable
                            } else if cols.len() == 1 {
                                PreparedIndexSlot::Column(OnceLock::new())
                            } else {
                                PreparedIndexSlot::Tuple(OnceLock::new())
                            }
                        }));
                    }
                }
                handles
            })
            .collect();
        Self { stages }
    }

    fn stage(&self, index: usize) -> &[PreparedIndexSlot] {
        &self.stages[index]
    }
}

/// Execution-scoped index sidecar mirroring the shape of a logical [`Plan`].
enum PreparedPlanIndexes {
    Single(PreparedJoinIndexes),
    Decomposed {
        blocks: Vec<PreparedJoinIndexes>,
        result: PreparedJoinIndexes,
    },
}

impl PreparedPlanIndexes {
    fn new(db: &Database, plan: &Plan) -> Self {
        match plan {
            Plan::SinglePlan(plan) => {
                Self::Single(PreparedJoinIndexes::new(db, &plan.atoms, &plan.stages))
            }
            Plan::DecomposedPlan(plan) => Self::Decomposed {
                blocks: plan
                    .stages
                    .blocks
                    .iter()
                    .map(|(stages, _)| PreparedJoinIndexes::new(db, &plan.atoms, stages))
                    .collect(),
                result: PreparedJoinIndexes::new(db, &plan.atoms, &plan.result_block),
            },
        }
    }
}

enum DynamicIndex<'plan> {
    Cached {
        /// When Some(range), intersect each subset from the index with this dense range.
        /// The range is the Dense outer subset known at Prober construction time.
        intersect_outer: Option<OffsetRange>,
        table: &'plan Index<TupleIndex>,
    },
    CachedColumn {
        /// When Some(range), intersect each subset from the index with this dense range.
        /// The range is the Dense outer subset known at Prober construction time.
        intersect_outer: Option<OffsetRange>,
        table: &'plan Index<ColumnIndex>,
    },
    Dynamic(TupleIndex),
    DynamicColumn(Arc<SortedColumnIndex>),
    SparseColumn(SparseColumnIndex),
}

/// This struct is used to mark subsets that can contain non-stale entries.
/// Whether a subset can be stale depends on the type of index it came from.
/// Indices that come from a table may contain stale entries, while
/// those that are built on the fly will not.
struct PotentiallyStale<T> {
    inner: T,
    can_be_stale: bool,
}

impl<T> PotentiallyStale<T> {
    fn maybe_stale(inner: T) -> Self {
        Self {
            inner,
            can_be_stale: true,
        }
    }

    fn not_stale(inner: T) -> Self {
        Self {
            inner,
            can_be_stale: false,
        }
    }
}

impl PotentiallyStale<SubsetRef<'_>> {
    fn size(&self) -> usize {
        self.inner.size()
    }
}

/// Intersect a `SubsetRef` with a dense `OffsetRange` and return the result as a
/// borrowed `SubsetRef`, or `None` if the intersection is empty.
///
/// This function never allocates — it borrows into
/// the source data via `subslice`. Use this in `for_each` paths where the result
/// may be discarded (e.g., empty after refinement), to avoid pool allocations.
#[inline]
fn intersect_with_dense_ref<'a>(v: SubsetRef<'a>, range: OffsetRange) -> Option<SubsetRef<'a>> {
    match v {
        SubsetRef::Dense(r) => {
            let resl = cmp::max(r.start, range.start);
            let resr = cmp::min(r.end, range.end);
            if resl >= resr {
                None
            } else {
                Some(SubsetRef::Dense(OffsetRange::new(resl, resr)))
            }
        }
        SubsetRef::Sparse(s) => {
            let l = s.binary_search_by_id(range.start);
            let r = s.binary_search_by_id(range.end);
            if l >= r {
                None
            } else {
                Some(SubsetRef::Sparse(s.subslice(l, r)))
            }
        }
    }
}

struct Prober<'plan> {
    node: Arc<TrieNode>,
    ix: DynamicIndex<'plan>,
}

impl Prober<'_> {
    fn get_subset<'a>(&'a self, key: &'a [Value]) -> Option<PotentiallyStale<SubsetRef<'a>>> {
        match &self.ix {
            DynamicIndex::Cached {
                intersect_outer,
                table,
            } => {
                let subset_ref = table.get_subset(key)?;
                let subset = if let Some(range) = intersect_outer {
                    intersect_with_dense_ref(subset_ref, *range)?
                } else {
                    subset_ref
                };
                Some(PotentiallyStale::maybe_stale(subset))
            }
            DynamicIndex::CachedColumn {
                intersect_outer,
                table,
            } => {
                debug_assert_eq!(key.len(), 1);
                let subset_ref = table.get_subset(&key[0])?;
                let subset = if let Some(range) = intersect_outer {
                    intersect_with_dense_ref(subset_ref, *range)?
                } else {
                    subset_ref
                };
                Some(PotentiallyStale::maybe_stale(subset))
            }
            DynamicIndex::Dynamic(tab) => tab.get_subset(key).map(PotentiallyStale::not_stale),
            DynamicIndex::DynamicColumn(tab) => {
                tab.get_subset(key[0]).map(PotentiallyStale::not_stale)
            }
            DynamicIndex::SparseColumn(tab) => {
                debug_assert_eq!(key.len(), 1);
                tab.get_subset(key[0]).map(PotentiallyStale::not_stale)
            }
        }
    }
    fn for_each(&self, mut f: impl FnMut(&[Value], PotentiallyStale<SubsetRef>)) {
        match &self.ix {
            DynamicIndex::Cached {
                intersect_outer: Some(range),
                table,
            } => {
                let range = *range;
                table.for_each(|k, v| {
                    if let Some(res) = intersect_with_dense_ref(v, range) {
                        f(k, PotentiallyStale::maybe_stale(res))
                    }
                });
            }
            DynamicIndex::Cached {
                intersect_outer: None,
                table,
            } => table.for_each(|k, v| f(k, PotentiallyStale::maybe_stale(v))),
            DynamicIndex::CachedColumn {
                intersect_outer: Some(range),
                table,
            } => {
                let range = *range;
                table.for_each(|k, v| {
                    if let Some(res) = intersect_with_dense_ref(v, range) {
                        f(&[*k], PotentiallyStale::maybe_stale(res))
                    }
                });
            }
            DynamicIndex::CachedColumn {
                intersect_outer: None,
                table,
            } => {
                table.for_each(|k, v| f(&[*k], PotentiallyStale::maybe_stale(v)));
            }
            DynamicIndex::Dynamic(tab) => {
                tab.for_each(|k, v| f(k, PotentiallyStale::not_stale(v)));
            }
            DynamicIndex::DynamicColumn(tab) => tab.for_each(|k, v| {
                f(&[k], PotentiallyStale::not_stale(v));
            }),
            DynamicIndex::SparseColumn(tab) => {
                tab.for_each(|k, v| f(k, PotentiallyStale::not_stale(v)));
            }
        }
    }

    /// Enumerate one physical shard of a cached table index.
    ///
    /// Dynamic indexes are deliberately excluded: their storage is private to
    /// this prober and has no coarse, pre-existing partition worth scheduling.
    fn for_each_shard(
        &self,
        shard: usize,
        mut f: impl FnMut(&[Value], PotentiallyStale<SubsetRef>),
    ) {
        match &self.ix {
            DynamicIndex::Cached {
                intersect_outer,
                table,
            } => {
                let intersect_outer = *intersect_outer;
                table.for_each_shard(shard, |key, subset| {
                    let subset = if let Some(range) = intersect_outer {
                        let Some(subset) = intersect_with_dense_ref(subset, range) else {
                            return;
                        };
                        subset
                    } else {
                        subset
                    };
                    f(key, PotentiallyStale::maybe_stale(subset));
                });
            }
            DynamicIndex::CachedColumn {
                intersect_outer,
                table,
            } => {
                let intersect_outer = *intersect_outer;
                table.for_each_shard(shard, |key, subset| {
                    let subset = if let Some(range) = intersect_outer {
                        let Some(subset) = intersect_with_dense_ref(subset, range) else {
                            return;
                        };
                        subset
                    } else {
                        subset
                    };
                    f(&[*key], PotentiallyStale::maybe_stale(subset));
                });
            }
            DynamicIndex::Dynamic(_)
            | DynamicIndex::DynamicColumn(_)
            | DynamicIndex::SparseColumn(_) => {
                unreachable!("only cached indexes expose physical shards")
            }
        }
    }

    /// Return the number of coarse physical partitions when this prober is
    /// backed by a cached hash index.
    fn shard_count(&self) -> Option<usize> {
        match &self.ix {
            DynamicIndex::Cached { table, .. } => Some(table.shard_count()),
            DynamicIndex::CachedColumn { table, .. } => Some(table.shard_count()),
            DynamicIndex::Dynamic(_)
            | DynamicIndex::DynamicColumn(_)
            | DynamicIndex::SparseColumn(_) => None,
        }
    }

    fn shard_len(&self, shard: usize) -> Option<usize> {
        match &self.ix {
            DynamicIndex::Cached {
                intersect_outer: None,
                table,
            } => Some(table.shard_len(shard)),
            DynamicIndex::CachedColumn {
                intersect_outer: None,
                table,
            } => Some(table.shard_len(shard)),
            DynamicIndex::Cached {
                intersect_outer: Some(_),
                ..
            }
            | DynamicIndex::CachedColumn {
                intersect_outer: Some(_),
                ..
            }
            | DynamicIndex::Dynamic(_)
            | DynamicIndex::DynamicColumn(_)
            | DynamicIndex::SparseColumn(_) => None,
        }
    }

    fn len(&self) -> usize {
        match &self.ix {
            DynamicIndex::Cached { table, .. } => table.len(),
            DynamicIndex::CachedColumn { table, .. } => table.len(),
            DynamicIndex::Dynamic(tab) => tab.len(),
            DynamicIndex::DynamicColumn(tab) => tab.len(),
            DynamicIndex::SparseColumn(tab) => tab.len(),
        }
    }
}

impl Database {
    pub fn run_rule_set(&mut self, rule_set: &RuleSet, report_level: ReportLevel) -> RuleSetReport {
        if rule_set.plans.is_empty() {
            return RuleSetReport::default();
        }
        let match_counter = Arc::new(MatchCounter::new(rule_set.actions.n_ids()));
        // Trie roots are shared across all plans in this run. Tables are frozen
        // for the duration, so a given root key always denotes the same subset;
        // the cache is scoped to (and dropped at the end of) this call. Only
        // roots used by more than one plan are shared.
        //
        // The `mark_shared_roots` pre-pass and the per-atom root-signature work
        // are a fixed cost paid every call; on small databases (few/cheap index
        // builds) that cost outweighs the sharing it enables. Gate it on the
        // database size so small rule-set runs keep the zero-overhead per-plan
        // path (an empty `shared` set makes `root_node` skip the signature
        // entirely). The estimate grows over a run, so early/cheap iterations
        // stay ungated while large ones opt in exactly when sharing pays off.
        // Enable cross-plan root sharing only when some root is actually reused
        // across plans. `None` means `root_node` builds fresh per-plan roots with
        // zero added work — no signature machinery and (crucially on many-core
        // hosts) no DashMap allocation. The pre-pass is a cheap scan of the plans'
        // atoms; the shard count is matched to the thread count (see `with_shared`).
        let trie_cache: Option<Arc<TrieCache>> = {
            let shared =
                TrieCache::compute_shared(rule_set.plans.values().map(|(plan, _, _)| plan));
            (!shared.is_empty()).then(|| Arc::new(TrieCache::with_shared(shared)))
        };

        let search_and_apply_timer = Instant::now();
        // Prepare one lazy slot for every table-index access named by the
        // immutable plans before any worker starts. A slot acquires and refreshes
        // its Arc through the regular catalog helper on first cached use. The
        // sidecars are dropped before `merge_all` resets catalog entries.
        let prepared_indexes = rule_set
            .plans
            .values()
            .map(|(plan, _, _)| PreparedPlanIndexes::new(self, plan))
            .collect::<Vec<_>>();
        // let mut rule_reports: HashMap<String, Vec<RuleReport>>;
        let mut rule_reports: HashMap<Arc<str>, Vec<RuleReport>>;
        let exec_state = ExecutionState::new(self.read_only_view(), Default::default());
        if parallelize_db_level_op(self.total_size_estimate) {
            let dash_rule_reports: Arc<DashMap<Arc<str>, Vec<RuleReport>>> =
                Arc::new(DashMap::default());
            let db: &Database = self;
            egglog_concurrency::scope(|scope| {
                for ((plan, desc, symbol_map), prepared_index) in
                    rule_set.plans.values().zip(&prepared_indexes)
                {
                    // TODO: add stats
                    let report_plan = match report_level {
                        ReportLevel::TimeOnly => None,
                        ReportLevel::WithPlan | ReportLevel::StageInfo => {
                            Some(plan.to_report(symbol_map))
                        }
                    };

                    let dash_rule_reports = dash_rule_reports.clone();
                    let desc = desc.clone();
                    let exec_state = exec_state.clone();
                    let match_counter = match_counter.clone();
                    let trie_cache = trie_cache.clone();
                    scope.spawn(move |rule_scope| {
                        let join_state = JoinState::new(db, exec_state.clone(), trie_cache);
                        let mut binding_info = BindingInfo::default();
                        let mut action_buf =
                            ScopedActionBuffer::new(rule_scope, rule_set, match_counter.clone());
                        let search_and_apply_timer = Instant::now();

                        'eval: {
                            for (id, info) in plan.atoms().iter() {
                                let headers: SmallVec<[&JoinHeader; 2]> =
                                    plan.header().iter().filter(|h| h.atom == id).collect();
                                match join_state.root_node(info.table, &headers) {
                                    Some(node) => binding_info.insert_node(id, node),
                                    None => break 'eval,
                                }
                            }

                            match (plan, prepared_index) {
                                (Plan::SinglePlan(plan), PreparedPlanIndexes::Single(prepared)) => {
                                    join_state.run_join_stages(
                                        &plan.stages,
                                        prepared,
                                        &plan.atoms,
                                        plan.actions,
                                        &mut binding_info,
                                        &mut action_buf,
                                    );
                                }
                                (
                                    Plan::DecomposedPlan(plan),
                                    PreparedPlanIndexes::Decomposed {
                                        blocks: prepared_blocks,
                                        result: prepared_result,
                                    },
                                ) => {
                                    let mut materializations: DenseIdMap<
                                        MatId,
                                        Arc<DashMap<Vec<Value>, RowBuffer>>,
                                    > = DenseIdMap::with_capacity(plan.stages.blocks.len());
                                    for i in 0..plan.stages.blocks.len() {
                                        materializations.insert(
                                            MatId::from_usize(i),
                                            Arc::new(Default::default()),
                                        );
                                    }
                                    let specs: Arc<DenseIdMap<MatId, MatSpec>> = Arc::new(
                                        plan.stages
                                            .blocks
                                            .iter()
                                            .enumerate()
                                            .map(|(i, block)| {
                                                (MatId::from_usize(i), block.1.clone())
                                            })
                                            .collect(),
                                    );
                                    let mut materializations = Arc::new(materializations);

                                    for (mat_id, (stage_block, prepared_block)) in
                                        plan.stages.blocks.iter().zip(prepared_blocks).enumerate()
                                    {
                                        let mat_id = MatId::from_usize(mat_id);
                                        egglog_concurrency::scope(|stage_scope| {
                                            let mut materializer = ScopedMaterializer {
                                                scope: stage_scope,
                                                specs: specs.clone(),
                                                materializations: materializations.clone(),
                                                scratch_key: Default::default(),
                                                scratch_val: Default::default(),
                                            };
                                            join_state.run_join_stages(
                                                &stage_block.0,
                                                prepared_block,
                                                &plan.atoms,
                                                mat_id,
                                                &mut binding_info,
                                                &mut materializer,
                                            );
                                        });
                                        if materializations[mat_id].is_empty() {
                                            break 'eval;
                                        }
                                        assert_eq!(Arc::strong_count(&materializations), 1);
                                        let mut materializations_dearc =
                                            Arc::unwrap_or_clone(materializations);
                                        let materialization = mem::take(
                                            Arc::get_mut(&mut materializations_dearc[mat_id])
                                                .unwrap(),
                                        )
                                        .into_iter()
                                        .collect::<IndexMap<_, _>>();
                                        binding_info
                                            .materializations
                                            .insert(mat_id, Arc::new(materialization));
                                        materializations = Arc::new(materializations_dearc);
                                    }
                                    join_state.run_join_stages(
                                        &plan.result_block,
                                        prepared_result,
                                        &plan.atoms,
                                        plan.actions,
                                        &mut binding_info,
                                        &mut action_buf,
                                    );
                                }
                                _ => unreachable!("prepared plan shape must match logical plan"),
                            }
                        }
                        let search_and_apply_time = search_and_apply_timer.elapsed();
                        if action_buf.needs_flush {
                            action_buf.flush(&mut exec_state.clone());
                        }
                        let mut rule_report: RefMut<'_, Arc<str>, Vec<RuleReport>> =
                            dash_rule_reports.entry(desc).or_default();
                        rule_report.value_mut().push(RuleReport {
                            plan: report_plan,
                            search_and_apply_time,
                            num_matches: usize::MAX,
                        });
                    });
                }
            });
            rule_reports = dash_rule_reports
                .iter()
                .map(|entry| (entry.key().clone(), entry.value().clone()))
                .collect();
        } else {
            rule_reports = HashMap::default();
            let join_state = JoinState::new(self, exec_state.clone(), trie_cache.clone());
            // Just run all of the plans in order with a single in-place action
            // buffer.
            let mut action_buf = InPlaceActionBuffer {
                rule_set,
                match_counter: match_counter.as_ref(),
                batches: Default::default(),
            };
            for ((plan, desc, symbol_map), prepared_index) in
                rule_set.plans.values().zip(&prepared_indexes)
            {
                let report_plan = match report_level {
                    ReportLevel::TimeOnly => None,
                    ReportLevel::WithPlan | ReportLevel::StageInfo => {
                        Some(plan.to_report(symbol_map))
                    }
                };
                let mut binding_info = BindingInfo::default();

                let search_and_apply_timer = Instant::now();
                'eval: {
                    for (id, info) in plan.atoms().iter() {
                        let headers: SmallVec<[&JoinHeader; 2]> =
                            plan.header().iter().filter(|h| h.atom == id).collect();
                        match join_state.root_node(info.table, &headers) {
                            Some(node) => binding_info.insert_node(id, node),
                            None => break 'eval,
                        }
                    }
                    match (plan, prepared_index) {
                        (Plan::SinglePlan(plan), PreparedPlanIndexes::Single(prepared)) => {
                            join_state.run_join_stages(
                                &plan.stages,
                                prepared,
                                &plan.atoms,
                                plan.actions,
                                &mut binding_info,
                                &mut action_buf,
                            );
                        }
                        (
                            Plan::DecomposedPlan(plan),
                            PreparedPlanIndexes::Decomposed {
                                blocks: prepared_blocks,
                                result: prepared_result,
                            },
                        ) => {
                            let mut materializations =
                                DenseIdMap::with_capacity(plan.stages.blocks.len());
                            for i in 0..plan.stages.blocks.len() {
                                materializations.insert(MatId::from_usize(i), Default::default());
                            }
                            let mut materializer = InPlaceMaterializer {
                                specs: &plan
                                    .stages
                                    .blocks
                                    .iter()
                                    .enumerate()
                                    .map(|(i, block)| (MatId::from_usize(i), block.1.clone()))
                                    .collect(),
                                materializations,
                                scratch_key: Default::default(),
                                scratch_val: Default::default(),
                            };

                            for (mat_id, (stage_block, prepared_block)) in
                                plan.stages.blocks.iter().zip(prepared_blocks).enumerate()
                            {
                                let mat_id = MatId::from_usize(mat_id);
                                join_state.run_join_stages(
                                    &stage_block.0,
                                    prepared_block,
                                    &plan.atoms,
                                    mat_id,
                                    &mut binding_info,
                                    &mut materializer,
                                );
                                if materializer.materializations[mat_id].is_empty() {
                                    break 'eval;
                                }
                                binding_info.materializations.insert(
                                    mat_id,
                                    Arc::new(materializer.materializations.take(mat_id).unwrap()),
                                );
                            }
                            join_state.run_join_stages(
                                &plan.result_block,
                                prepared_result,
                                &plan.atoms,
                                plan.actions,
                                &mut binding_info,
                                &mut action_buf,
                            );
                        }
                        _ => unreachable!("prepared plan shape must match logical plan"),
                    }
                }
                let search_and_apply_time = search_and_apply_timer.elapsed();

                // TODO: unnecessary cloning in many cases
                let rule_report = rule_reports.entry(desc.clone()).or_default();
                rule_report.push(RuleReport {
                    plan: report_plan,
                    search_and_apply_time,
                    num_matches: usize::MAX,
                });
            }
            action_buf.flush(&mut exec_state.clone());
        }

        for (plan, desc, _symbol_map) in rule_set.plans.values() {
            let reports = rule_reports.get_mut(desc).unwrap();
            let i = reports
                .iter()
                // HACK: Since the order of visiting queries is fixed and # matches need to be obtained
                // seperately from rule execution, we first set all # matches to be usize::MAX and then fill
                // them in one by one.
                .position(|r| r.num_matches == usize::MAX)
                .unwrap();
            // NB: This requires each action ID correspond to only one query.
            // If an action is used by multiple queries, then we can't tell how many matches are
            // caused by individual queries.
            reports[i].num_matches = match_counter.read_matches(plan.actions());
        }
        // `merge_all` requires each catalog Arc to be uniquely owned so it can
        // reset the corresponding ResettableOnceLock. No scoped worker can
        // survive to this point.
        drop(prepared_indexes);
        let search_and_apply_time = search_and_apply_timer.elapsed();

        let merge_timer = Instant::now();
        let changed = self.merge_all();
        let merge_time = merge_timer.elapsed();

        RuleSetReport {
            changed,
            rule_reports,
            search_and_apply_time,
            merge_time,
        }
    }
}

struct ActionState {
    n_runs: usize,
    len: usize,
    bindings: Bindings,
}

impl ActionState {
    fn new(batch_size: usize) -> Self {
        Self {
            n_runs: 0,
            len: 0,
            bindings: Bindings::new(batch_size),
        }
    }
}

struct JoinState<'a> {
    db: &'a Database,
    exec_state: ExecutionState<'a>,
    /// Cached thread-local pool for SortedOffsetVector allocations.
    /// Stored here to avoid a per-call `with_pool_set` TLS access in `get_index`.
    pool: Pool<SortedOffsetVector>,
    /// Cross-plan trie-root cache for the current `run_rule_set`, or `None` when
    /// sharing is disabled (small run, or nothing reused across plans).
    trie_cache: Option<Arc<TrieCache>>,
}

/// Per-column indexes on a trie node's subset, lazily initialized on first access per column.
type ColumnIndexes = IdVec<ColumnId, OnceLock<Arc<SortedColumnIndex>>>;
// The child cache (see [`TrieNode::get_cached_trie_node`]): keyed by the bound
// value, storing the child node and the edge constraints used to build it. The
// stored constraints guard against distinct scans reaching the same
// (node, col, value) with different slow constraints; they are almost always
// empty, in which case the guard is a cheap length check.
type ChildEntry = (Arc<TrieNode>, Box<[Constraint]>);
pub(crate) type ChildLock = RwLock<HashMap<Value, ChildEntry>>;
type ChildrenMaps = IdVec<ColumnId, OnceLock<ChildLock>>;

fn get_or_insert_child(
    map: &ChildLock,
    value: Value,
    edge_cs: &[Constraint],
    sub: impl FnOnce() -> Subset,
) -> Arc<TrieNode> {
    // Optimistic read path: most calls are cache hits, so try a shared lock
    // first. A hit is only valid when the edge constraints match.
    {
        let guard = map.read().unwrap();
        if let Some((node, stored_cs)) = guard.get(&value)
            && &**stored_cs == edge_cs
        {
            return node.clone();
        }
    }
    // Cache miss (or constraint mismatch): acquire the write lock and insert.
    let mut guard = map.write().unwrap();
    if let Some((node, stored_cs)) = guard.get(&value)
        && &**stored_cs == edge_cs
    {
        return node.clone();
    }
    let new_node = Arc::new(TrieNode::new(sub()));
    guard.insert(value, (new_node.clone(), Box::from(edge_cs)));
    new_node
}

/// Canonical signature of a trie root: the table plus its sorted header (fast)
/// constraints. Distinct signatures get distinct base ids from [`TrieCache`].
type BaseSig = (TableId, SmallVec<[Constraint; 2]>);

/// Key for a shared trie root: the table plus an interned id for its fast
/// (header) constraints.
type RootKey = (TableId, u32);

/// A cache of trie *roots* shared across all plans within a single
/// `run_rule_set` call. Two plans that constrain the same table with the same
/// fast constraints share a root; the rest of the trie is then shared implicitly
/// because a shared root's per-node child caches are shared with it. Sharing lets
/// each node's cached sub-indexes and children be built once and reused across
/// plans.
///
/// Only roots that more than one plan actually uses are shared (`shared`), so
/// single-use roots stay per-plan and keep the pool-recycling behavior of the
/// unshared path — sharing a root that is never reused is pure overhead.
///
/// Concurrency: the parallel executor runs plans on multiple threads, so the maps
/// are concurrent. Tables are frozen during a run, so a given key always denotes
/// the same subset.
#[derive(Default)]
struct TrieCache {
    roots: DashMap<RootKey, Arc<TrieNode>>,
    /// Interns base signatures to small ids to keep [`RootKey`] cheap.
    bases: DashMap<BaseSig, u32>,
    next_base: AtomicUsize,
    /// Root signatures used by more than one plan; only these are shared.
    shared: HashSet<BaseSig>,
}

impl TrieCache {
    /// Return the interned base id for a root subset identified by `table` and
    /// its (fast) header constraints.
    ///
    /// Base id 0 is reserved for the (common) unconstrained case, so atoms with
    /// no fast constraints skip the interning map entirely. `RootKey` already
    /// carries `table`, so base ids only need to distinguish constraint sets
    /// within a table.
    fn base_id(&self, table: TableId, fast: &[Constraint]) -> u32 {
        if fast.is_empty() {
            return 0;
        }
        let mut sig: SmallVec<[Constraint; 2]> = SmallVec::from_iter(fast.iter().cloned());
        sig.sort_unstable();
        match self.bases.entry((table, sig)) {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(v) => {
                let id = self.next_base.fetch_add(1, Ordering::Relaxed) as u32 + 1;
                v.insert(id);
                id
            }
        }
    }

    /// The canonical root signature (table + sorted fast constraints) for `atom`
    /// given its headers.
    fn root_sig(plan: &Plan, atom: AtomId, table: TableId) -> BaseSig {
        let mut fast: SmallVec<[Constraint; 2]> = SmallVec::new();
        for h in plan.header().iter().filter(|h| h.atom == atom) {
            fast.extend(h.constraints.iter().cloned());
        }
        fast.sort_unstable();
        (table, fast)
    }

    /// Compute the set of root signatures used by more than one plan atom (across
    /// all plans); only these are worth sharing.
    fn compute_shared<'a>(plans: impl Iterator<Item = &'a Plan>) -> HashSet<BaseSig> {
        let mut counts: HashMap<BaseSig, u32> = HashMap::default();
        for plan in plans {
            for (atom, info) in plan.atoms().iter() {
                *counts
                    .entry(Self::root_sig(plan, atom, info.table))
                    .or_default() += 1;
            }
        }
        counts
            .into_iter()
            .filter_map(|(sig, n)| (n > 1).then_some(sig))
            .collect()
    }

    /// Build a cache for the given shared root signatures. Only called when
    /// `shared` is non-empty, so the DashMap allocations always pay off.
    ///
    /// Shard the maps to the actual thread count rather than DashMap's default
    /// (`4 * num_cpus`): on a many-core host the default allocates hundreds of
    /// shards per `run_rule_set`, which dwarfs the sharing savings on smaller
    /// runs. Serial runs get a single shard.
    fn with_shared(shared: HashSet<BaseSig>) -> TrieCache {
        // DashMap requires at least 2 shards; that is plenty for serial runs and
        // still far below the default (4 * num_cpus).
        let shards = crate::parallel::current_num_threads()
            .next_power_of_two()
            .max(2);
        TrieCache {
            roots: DashMap::with_hasher_and_shard_amount(Default::default(), shards),
            bases: DashMap::with_hasher_and_shard_amount(Default::default(), shards),
            next_base: AtomicUsize::new(0),
            shared,
        }
    }
}

/// Information about the current subset of an atom's relation that is being considered, along with
/// lazily-initialized, cached indexes on that subset.
///
/// This is the standard trie-node used in lazy implementations of GJ as in the original egglog
/// implementation and the FJ paper. It currently does not handle non-column indexes, but that
/// should be a fairly straightforward extension if we start generating plans that need those.
/// (Right now, most plans iterating over more than one column just do a scan anyway).
pub(crate) struct TrieNode {
    /// The actual subset of the corresponding atom.
    subset: Subset,
    /// Any cached indexes on this subset.
    cached_subsets: OnceLock<Pooled<ColumnIndexes>>,
    /// Cached child trie nodes, keyed first by column and then by value. Each
    /// column's lock is allocated lazily. When this node is a shared root (or
    /// reachable from one), this cache is shared across plans too, so children
    /// are shared without any global lookup.
    cached_children: OnceLock<Pooled<ChildrenMaps>>,
}

impl std::fmt::Debug for TrieNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrieNode")
            .field("subset", &self.subset)
            .finish()
    }
}

impl TrieNode {
    fn new(subset: Subset) -> Self {
        Self {
            subset,
            cached_subsets: Default::default(),
            cached_children: Default::default(),
        }
    }

    fn size(&self) -> usize {
        self.subset.size()
    }
    fn get_cached_index(&self, col: ColumnId, info: &TableInfo) -> Arc<SortedColumnIndex> {
        self.cached_subsets.get_or_init(|| {
            // Pre-size the vector so we do not need to borrow it mutably to initialize the index.
            let mut vec: Pooled<ColumnIndexes> = with_pool_set(|ps| ps.get());
            vec.resize_with(info.spec.arity(), OnceLock::new);
            vec
        })[col]
            .get_or_init(|| {
                Arc::new(SortedColumnIndex::build_for_subset(
                    info.table.as_ref(),
                    self.subset.as_ref(),
                    col,
                ))
            })
            .clone()
    }

    fn child_map(&self, col: ColumnId, arity: usize) -> &ChildLock {
        self.cached_children.get_or_init(|| {
            let mut vec: Pooled<ChildrenMaps> = with_pool_set(|ps| ps.get());
            vec.resize_with(arity, OnceLock::new);
            vec
        })[col]
            .get_or_init(|| RwLock::new(HashMap::default()))
    }

    /// Return the child node reached by additionally constraining `col = value`
    /// (and applying `edge_cs`). `sub` computes the child subset and is only
    /// called on a cache miss.
    ///
    /// Children are cached on the node itself, keyed by `value`. When this node
    /// is a shared root (or reachable from one) its child cache is shared across
    /// plans, so a hit yields cross-plan child sharing with a single-value lookup
    /// and no global cache access. The stored constraints guard against distinct
    /// scans reaching the same (node, col, value) with different slow
    /// constraints; they are almost always empty (a cheap length check).
    fn get_cached_trie_node(
        &self,
        col: ColumnId,
        value: Value,
        edge_cs: &[Constraint],
        info: &TableInfo,
        sub: impl FnOnce() -> Subset,
    ) -> Arc<TrieNode> {
        get_or_insert_child(self.child_map(col, info.spec.arity()), value, edge_cs, sub)
    }
}

impl FrameUpdates {
    /// Refine `atom` to `subset`, using the dense fast path to avoid an
    /// `Arc<TrieNode>` allocation when the subset is already a contiguous range.
    fn refine_atom_subset(&mut self, atom: AtomId, subset: Subset) {
        match subset {
            Subset::Dense(range) => self.refine_atom_dense(atom, range),
            sub => self.refine_atom(atom, Arc::new(TrieNode::new(sub))),
        }
    }
}

type BindingSet = Vec<(SmallVec<[Variable; 4]>, Arc<TaggedRowBuffer<SmallValueVec>>)>;

#[derive(Default, Clone)]
struct BindingInfo {
    bindings: DenseIdMap<Variable, Value>,
    binding_sets: BindingSet,
    subsets: DenseIdMap<AtomId, Arc<TrieNode>>,
    materializations: DenseIdMap<MatId, Arc<IndexMap<Vec<Value>, RowBuffer>>>,
}

impl BindingInfo {
    /// Initializes the atom-related metadata in the [`BindingInfo`].    
    fn insert_subset(&mut self, atom: AtomId, subset: Subset) {
        if let Some(slot) = self.subsets.get_mut(atom)
            && let Some(node) = Arc::get_mut(slot)
        {
            node.cached_subsets.take();
            node.cached_children.take();
            node.subset = subset;
            return;
        }
        self.subsets.insert(atom, Arc::new(TrieNode::new(subset)));
    }

    fn insert_node(&mut self, atom: AtomId, node: Arc<TrieNode>) {
        self.subsets.insert(atom, node);
    }

    /// Probers returned from [`JoinState::get_index`] will move atom-related state out of the
    /// [`BindingInfo`]. Once the caller is done using a prober, this method moves it back.
    fn move_back(&mut self, atom: AtomId, prober: Prober<'_>) {
        self.subsets.insert(atom, prober.node);
    }

    fn move_back_node(&mut self, atom: AtomId, node: Arc<TrieNode>) {
        self.subsets.insert(atom, node);
    }

    fn has_empty_subset(&self, atom: AtomId) -> bool {
        self.subsets[atom].subset.is_empty()
    }

    fn unwrap_val(&mut self, atom: AtomId) -> Arc<TrieNode> {
        self.subsets.unwrap_val(atom)
    }
}

impl<'a> JoinState<'a> {
    fn new(
        db: &'a Database,
        exec_state: ExecutionState<'a>,
        trie_cache: Option<Arc<TrieCache>>,
    ) -> Self {
        Self {
            db,
            exec_state,
            pool: with_pool_set(|ps| ps.get_pool()),
            trie_cache,
        }
    }

    /// Look up (or create) the root trie node for `atom` given all of its
    /// headers.
    ///
    /// An atom may carry more than one header (e.g. seminaive adds a timestamp
    /// constraint on top of the plan's original fast constraints); the root
    /// subset is the whole table intersected with every header subset. Returns
    /// `None` when that subset is empty.
    ///
    /// Roots whose signature is used by more than one plan (see
    /// [`TrieCache::shared`]) are shared through the cache; the rest are built
    /// fresh per plan so the pool can recycle them.
    fn root_node(&self, table_id: TableId, headers: &[&JoinHeader]) -> Option<Arc<TrieNode>> {
        // Fast path: when sharing is disabled this run (small database, or no
        // root reused across plans), skip the root-signature machinery entirely
        // and build a fresh per-plan root — matching the pre-sharing behavior at
        // no added cost.
        let Some(trie_cache) = self.trie_cache.as_ref() else {
            return Some(Arc::new(TrieNode::new(
                self.build_root_subset(table_id, headers)?,
            )));
        };
        // The base identity is the union of all fast constraints on this atom.
        let mut fast: SmallVec<[Constraint; 2]> = SmallVec::new();
        for h in headers {
            fast.extend(h.constraints.iter().cloned());
        }
        fast.sort_unstable();
        let sig: BaseSig = (table_id, fast);

        if !trie_cache.shared.contains(&sig) {
            // Not reused across plans: build a fresh, unshared root.
            return Some(Arc::new(TrieNode::new(
                self.build_root_subset(table_id, headers)?,
            )));
        }

        let base = trie_cache.base_id(table_id, &sig.1);
        let key: RootKey = (table_id, base);
        if let Some(node) = trie_cache.roots.get(&key) {
            return (!node.subset.is_empty()).then(|| node.clone());
        }
        let subset = self.build_root_subset(table_id, headers)?;
        let node = match trie_cache.roots.entry(key) {
            Entry::Occupied(o) => o.get().clone(),
            Entry::Vacant(v) => {
                let node = Arc::new(TrieNode::new(subset));
                v.insert(node.clone());
                node
            }
        };
        (!node.subset.is_empty()).then_some(node)
    }

    /// The root subset for `table_id`: the whole table intersected with every
    /// header subset. Returns `None` if the result is empty.
    fn build_root_subset(&self, table_id: TableId, headers: &[&JoinHeader]) -> Option<Subset> {
        let mut subset = self.db.get_table(table_id).all();
        for h in headers {
            if h.subset.is_empty() {
                return None;
            }
            subset.intersect(h.subset.as_ref(), &self.pool);
            if subset.is_empty() {
                return None;
            }
        }
        Some(subset)
    }

    fn get_index<'plan>(
        &self,
        atoms: &Arc<DenseIdMap<AtomId, Atom>>,
        atom: AtomId,
        binding_info: &mut BindingInfo,
        cols: impl Iterator<Item = ColumnId>,
        prepared: &'plan PreparedIndexSlot,
    ) -> Prober<'plan> {
        let cols = SmallVec::<[ColumnId; 4]>::from_iter(cols);
        let trie_node = binding_info.subsets.unwrap_val(atom);
        let subset = &trie_node.subset;

        let table_id = atoms[atom].table;
        let info = &self.db.tables[table_id];
        let dyn_index = if subset.size() <= SMALL_RESIDUAL && cols.len() == 1 {
            DynamicIndex::SparseColumn(SparseColumnIndex::new(
                info.table.as_ref(),
                subset.as_ref(),
                cols[0],
            ))
        } else {
            let all_cacheable = columns_are_cacheable(info, &cols);
            let whole_table = info.table.all();
            if let Subset::Dense(range) = subset
                && all_cacheable
                && whole_table.size() / 2 < subset.size()
            {
                // Skip intersecting with the subset if we are just looking at the
                // whole table.
                let needs_intersect =
                    !(whole_table.is_dense() && subset.bounds() == whole_table.bounds());
                // When intersecting, store the Dense range directly so we can do a
                // combined copy+filter without a runtime match on subset type later.
                let intersect_outer = if needs_intersect { Some(*range) } else { None };
                // heuristic: if the subset we are scanning is somewhat
                // large _or_ it is most of the table, or we already have a cached
                // index for it, then return it.
                if cols.len() != 1 {
                    let PreparedIndexSlot::Tuple(index) = prepared else {
                        unreachable!("multi-column scan must have a prepared tuple index")
                    };
                    let index =
                        index.get_or_init(|| get_index_from_tableinfo(info, cols.as_slice()));
                    DynamicIndex::Cached {
                        intersect_outer,
                        table: index
                            .get()
                            .expect("prepared tuple index must already be refreshed"),
                    }
                } else {
                    let PreparedIndexSlot::Column(index) = prepared else {
                        unreachable!("single-column scan must have a prepared column index")
                    };
                    let index =
                        index.get_or_init(|| get_column_index_from_tableinfo(info, cols[0]));
                    DynamicIndex::CachedColumn {
                        intersect_outer,
                        table: index
                            .get()
                            .expect("prepared column index must already be refreshed"),
                    }
                }
            } else if cols.len() != 1 {
                // NB: we should have a caching strategy for non-column indexes.
                DynamicIndex::Dynamic(info.table.group_by_key(subset.as_ref(), &cols))
            } else {
                DynamicIndex::DynamicColumn(trie_node.get_cached_index(cols[0], info))
            }
        };
        Prober {
            node: trie_node,
            ix: dyn_index,
        }
    }
    fn get_column_index<'plan>(
        &self,
        atoms: &Arc<DenseIdMap<AtomId, Atom>>,
        binding_info: &mut BindingInfo,
        atom: AtomId,
        col: ColumnId,
        prepared: &'plan PreparedIndexSlot,
    ) -> Prober<'plan> {
        self.get_index(atoms, atom, binding_info, iter::once(col), prepared)
    }

    /// Describe an eligible cached index that could drive a top-level
    /// generic-join intersection.
    ///
    /// This mirrors the leader selection in [`Self::run_plan`].  Probers move
    /// trie nodes out of `binding_info`, so restore every node before returning.
    /// Every scan must cover its whole table through an unfiltered cached
    /// hash/column index. Requiring this before constructing any prober avoids
    /// building a dynamic nonleader index during discovery and then rebuilding
    /// it in every shard job. Tiny or badly skewed leaders stay on the existing
    /// path as well.
    fn top_index_shards(
        &self,
        stage: &JoinStage,
        prepared: &[PreparedIndexSlot],
        atoms: &Arc<DenseIdMap<AtomId, Atom>>,
        binding_info: &mut BindingInfo,
        workers: usize,
    ) -> Option<Vec<usize>> {
        let JoinStage::Intersect { scans, .. } = stage else {
            return None;
        };
        if scans.is_empty() || workers <= 1 {
            return None;
        }
        debug_assert_eq!(scans.len(), prepared.len());

        // Mirror the cached/unfiltered branch in `get_index` without actually
        // constructing any dynamic fallback indexes.
        for scan in scans {
            let node = &binding_info.subsets[scan.atom];
            let info = &self.db.tables[atoms[scan.atom].table];
            if node.subset.size() <= SMALL_RESIDUAL || !columns_are_cacheable(info, &[scan.column])
            {
                return None;
            }
            let Subset::Dense(subset) = &node.subset else {
                return None;
            };
            let Subset::Dense(whole_table) = info.table.all() else {
                return None;
            };
            if subset != &whole_table {
                return None;
            }
        }

        let mut leader = 0;
        let mut leader_size = usize::MAX;
        let mut probers = Vec::with_capacity(scans.len());
        for (i, (scan, prepared)) in scans.iter().zip(prepared).enumerate() {
            let prober =
                self.get_column_index(atoms, binding_info, scan.atom, scan.column, prepared);
            let size = prober.len();
            if size < leader_size {
                leader = i;
                leader_size = size;
            }
            probers.push(prober);
        }
        let shards = probers[leader].shard_count().and_then(|shard_count| {
            let shards = (0..shard_count)
                .filter(|shard| probers[leader].shard_len(*shard).unwrap_or(0) != 0)
                .collect::<Vec<_>>();
            top_index_shape_is_eligible(
                workers,
                leader_size,
                shards.len(),
                MIN_TOP_INDEX_KEYS_PER_WORKER,
            )
            .then_some(shards)
        });
        for (scan, prober) in scans.iter().zip(probers) {
            binding_info.move_back(scan.atom, prober);
        }
        shards
    }

    /// Return a coarse partition for the variable already sorted to the top of
    /// the join. Later variables are deliberately not promoted: benchmarking
    /// found that probing and reordering them was not broadly beneficial.
    fn select_top_index_shards(
        &self,
        stages: &JoinStages,
        prepared: &PreparedJoinIndexes,
        atoms: &Arc<DenseIdMap<AtomId, Atom>>,
        order: &InstrOrder,
        binding_info: &mut BindingInfo,
    ) -> Option<Vec<usize>> {
        if order.len() == 0 {
            return None;
        }

        let stage_index = order.get(0);
        self.top_index_shards(
            &stages.instrs[stage_index],
            prepared.stage(stage_index),
            atoms,
            binding_info,
            crate::parallel::current_num_threads(),
        )
    }

    /// Runs the free join plan, starting with the header.
    ///
    /// A bit about the `instr_order` parameter: This defines the order in which the [`JoinStage`]
    /// instructions will run. We want to support cached [`SinglePlan`]s that may be based on stale
    /// ordering information. `instr_order` allows us to specify a new ordering of the instructions
    /// without mutating the plan itself: `run_plan` simply executes
    /// `plan.stages.instrs[instr_order[i]]` at stage `i`.
    ///
    /// This is also a stepping stone towards supporting fully dynamic variable ordering.
    fn run_join_stages<'buf, A: NumericId + 'buf, BUF: ActionBuffer<'buf, A>>(
        &self,
        stages: &'buf JoinStages,
        prepared: &'buf PreparedJoinIndexes,
        atoms: &'buf Arc<DenseIdMap<AtomId, Atom>>,
        action: A,
        binding_info: &mut BindingInfo,
        action_buf: &mut BUF,
    ) where
        'a: 'buf,
    {
        if log::log_enabled!(log::Level::Trace) {
            log::trace!("Starting running query stages:\n{stages:#?}");
        }
        for (_, node) in binding_info.subsets.iter() {
            if node.subset.is_empty() {
                return;
            }
        }
        let mut order = InstrOrder::from_iter(0..stages.instrs.len());
        let mut leaf_scans: LeafScans = smallvec::smallvec![false; stages.instrs.len()];
        sort_plan_by_size(&mut order, &mut leaf_scans, 0, &stages.instrs, binding_info);

        // A cached top index is already partitioned by hash. Schedule one
        // coarse global job per nonempty shard and run each shard's subtree
        // serially. Besides avoiding an intermediate key copy, this keeps
        // related nested probes on one worker. Selecting this path commits the
        // whole subtree to coarse-only parallelism: `recur_global_serial`
        // supplies a serial buffer whose recursive work stays inline. Buffers
        // that cannot construct an independent partition (the in-place
        // executors) decline this path.
        if !stages.instrs.is_empty()
            && action_buf.supports_global_partition()
            && let Some(shards) =
                self.select_top_index_shards(stages, prepared, atoms, &order, binding_info)
        {
            let mut updates = FrameUpdates::with_capacity(0);
            for shard in shards {
                let db = self.db;
                let exec_state_for_factory = self.exec_state.clone();
                let exec_state_for_work = self.exec_state.clone();
                let trie_cache = self.trie_cache.clone();
                action_buf.recur_global_serial(
                    BorrowedLocalState {
                        binding_info,
                        instr_order: &mut order,
                        leaf_scans: &mut leaf_scans,
                        updates: &mut updates,
                    },
                    move || exec_state_for_factory.clone(),
                    move |BorrowedLocalState {
                              binding_info,
                              instr_order,
                              leaf_scans,
                              ..
                          },
                          buf| {
                        JoinState {
                            db,
                            exec_state: exec_state_for_work,
                            pool: with_pool_set(|ps| ps.get_pool()),
                            trie_cache,
                        }
                        .run_plan(
                            stages,
                            prepared,
                            atoms,
                            action,
                            instr_order,
                            leaf_scans,
                            0,
                            Some(shard),
                            binding_info,
                            buf,
                        );
                    },
                );
            }
            return;
        }
        self.run_plan(
            stages,
            prepared,
            atoms,
            action,
            &mut order,
            &mut leaf_scans,
            0,
            None,
            binding_info,
            action_buf,
        );
    }

    /// The core method for executing a free join plan.
    ///
    /// This method takes the plan, mutable data structures for variable binding
    /// and staging actions, and `cur`, the current stage of the plan. A top-level
    /// coarse partition also passes `index_shard`; `Some` is only supplied with
    /// a serial action buffer, so no nested parallelism is available in that
    /// subtree. Recursive calls clear the value so only the first intersection
    /// is restricted to that physical shard, while the serial buffer continues
    /// to keep later work inline.
    #[allow(clippy::too_many_arguments)]
    fn run_plan<'buf, A: NumericId + 'buf, BUF: ActionBuffer<'buf, A>>(
        &self,
        stages: &'buf JoinStages,
        prepared: &'buf PreparedJoinIndexes,
        atoms: &'buf Arc<DenseIdMap<AtomId, Atom>>,
        action: A,
        instr_order: &mut InstrOrder,
        leaf_scans: &mut LeafScans,
        cur: usize,
        index_shard: Option<usize>,
        binding_info: &mut BindingInfo,
        action_buf: &mut BUF,
    ) where
        'a: 'buf,
    {
        if self.exec_state.should_stop() {
            return;
        }

        if cur >= instr_order.len() {
            action_buf.push_bindings_factorized(
                action,
                &mut binding_info.bindings,
                &binding_info.binding_sets,
                &self.exec_state,
            );
            return;
        }
        let chunk_size = action_buf.morsel_size(cur, instr_order.len());
        let mut cur_size = estimate_size(&stages.instrs[instr_order.get(cur)], binding_info);
        if cur_size > 32 && cur % 3 == 1 && cur < instr_order.len() - 1 {
            // If we have a reasonable number of tuples to process, adjust the variable order every
            // 3 rounds, but always make sure to readjust on the second roung.
            sort_plan_by_size(instr_order, leaf_scans, cur, &stages.instrs, binding_info);
            cur_size = estimate_size(&stages.instrs[instr_order.get(cur)], binding_info);
        }

        // Helper macro (not its own method to appease the borrow checker).
        macro_rules! drain_updates {
            ($updates:expr) => {
                if self.exec_state.should_stop() {
                    return;
                }
                // TODO: `supports_parallel_drain`` is a hack because currently
                // `drain_updates_parallel!`` is a bit slower because of the additional ExecutionState clone.
                if index_shard.is_none()
                    && cur < free_join_fork_depth()
                    && action_buf.supports_parallel_drain()
                {
                    drain_updates_parallel!($updates)
                } else {
                    $updates.drain(|update| match update {
                        UpdateInstr::PushBinding(var, val) => {
                            binding_info.bindings.insert(var, val);
                        }
                        UpdateInstr::RefineAtom(atom, subset) => {
                            binding_info.insert_node(atom, subset);
                        }
                        UpdateInstr::RefineAtomDense(atom, range) => {
                            binding_info.insert_subset(atom, Subset::Dense(range));
                        }
                        UpdateInstr::EndFrame => {
                            // Inline leaf-level: if cur+1 is the leaf (no more
                            // join stages), call push_bindings directly without
                            // a recursive run_plan call, avoiding function call
                            // overhead + an extra should_stop() check.
                            if cur + 1 >= instr_order.len() {
                                action_buf.push_bindings_factorized(
                                    action,
                                    &mut binding_info.bindings,
                                    &binding_info.binding_sets,
                                    &self.exec_state,
                                );
                            } else {
                                self.run_plan(
                                    stages,
                                    prepared,
                                    atoms,
                                    action,
                                    instr_order,
                                    leaf_scans,
                                    cur + 1,
                                    None,
                                    binding_info,
                                    action_buf,
                                );
                            }
                        }
                    })
                }
            };
        }
        macro_rules! drain_updates_parallel {
            ($updates:expr) => {{
                if self.exec_state.should_stop() {
                    return;
                }
                let db = self.db;
                let exec_state_for_factory = self.exec_state.clone();
                let exec_state_for_work = self.exec_state.clone();
                let trie_cache = self.trie_cache.clone();
                action_buf.recur(
                    BorrowedLocalState {
                        binding_info,
                        instr_order,
                        leaf_scans,
                        updates: &mut $updates,
                    },
                    move || exec_state_for_factory.clone(),
                    move |BorrowedLocalState {
                              binding_info,
                              instr_order,
                              leaf_scans,
                              updates,
                          },
                          buf| {
                        updates.drain(|update| match update {
                            UpdateInstr::PushBinding(var, val) => {
                                binding_info.bindings.insert(var, val);
                            }
                            UpdateInstr::RefineAtom(atom, subset) => {
                                binding_info.insert_node(atom, subset);
                            }
                            UpdateInstr::RefineAtomDense(atom, range) => {
                                binding_info.insert_subset(atom, Subset::Dense(range));
                            }
                            UpdateInstr::EndFrame => {
                                JoinState {
                                    db,
                                    exec_state: exec_state_for_work.clone(),
                                    // Each scoped task uses its own thread-local pool.
                                    // This makes drain_updates_parallel slightly more expensive
                                    // than drain_updates eevn when both are run in single thread
                                    pool: with_pool_set(|ps| ps.get_pool()),
                                    trie_cache: trie_cache.clone(),
                                }
                                .run_plan(
                                    stages,
                                    prepared,
                                    atoms,
                                    action,
                                    instr_order,
                                    leaf_scans,
                                    cur + 1,
                                    None,
                                    binding_info,
                                    buf,
                                );
                            }
                        })
                    },
                );
                $updates.clear();
            }};
        }

        fn refine_subset(
            sub: PotentiallyStale<SubsetRef<'_>>,
            constraints: &[Constraint],
            table: &WrappedTableRef,
            has_stale: bool,
            pool: &Pool<SortedOffsetVector>,
        ) -> Subset {
            let need_live = sub.can_be_stale && has_stale;
            if constraints.is_empty() && !need_live {
                sub.inner.to_owned(pool)
            } else {
                // Fused copy + liveness + constraint filter (single pass for
                // tables that implement `refine_ref` directly).
                table.refine_ref(sub.inner, constraints, need_live)
            }
        }

        // A sharded top-level job enumerates only its assigned physical index
        // shard.  Every recursive call clears `index_shard`, so this macro is
        // used only by the leading prober of the initial `Intersect` stage.
        macro_rules! for_each_leader {
            ($prober:expr, $callback:expr) => {
                if let Some(shard) = index_shard {
                    $prober.for_each_shard(shard, $callback)
                } else {
                    $prober.for_each($callback)
                }
            };
        }

        let pool = &self.pool;
        let stage_index = instr_order.get(cur);
        let stage = &stages.instrs[stage_index];
        let prepared_indexes = prepared.stage(stage_index);
        debug_assert_eq!(
            prepared_indexes.len(),
            match stage {
                JoinStage::Intersect { scans, .. } => scans.len(),
                JoinStage::FusedIntersect { to_intersect, .. }
                | JoinStage::FusedIntersectMat { to_intersect, .. } => to_intersect.len(),
            }
        );
        match stage {
            JoinStage::Intersect { var, scans } => match scans.as_slice() {
                [] => {}
                [a] => {
                    if binding_info.has_empty_subset(a.atom) {
                        return;
                    }
                    let prober = self.get_column_index(
                        atoms,
                        binding_info,
                        a.atom,
                        a.column,
                        &prepared_indexes[0],
                    );
                    let info = &self.db.tables[atoms[a.atom].table];
                    let table = info.table.as_ref();
                    let has_stale = table.has_stale_rows();
                    let mut updates = FrameUpdates::with_capacity(cmp::min(chunk_size, cur_size));
                    for_each_leader!(prober, |val, x| {
                        updates.push_binding(*var, val[0]);
                        if x.size() <= 16 {
                            let sub = refine_subset(x, &a.cs, &table, has_stale, pool);
                            if sub.is_empty() {
                                updates.rollback();
                                return;
                            }
                            updates.refine_atom_subset(a.atom, sub);
                        } else {
                            let node = prober.node.get_cached_trie_node(
                                a.column,
                                val[0],
                                &a.cs,
                                info,
                                || refine_subset(x, &a.cs, &table, has_stale, pool),
                            );
                            if node.subset.is_empty() {
                                updates.rollback();
                                return;
                            }
                            updates.refine_atom(a.atom, node);
                        }
                        updates.finish_frame();
                        if updates.frames() >= chunk_size {
                            drain_updates!(updates);
                        }
                    });
                    drain_updates!(updates);
                    binding_info.move_back(a.atom, prober);
                }
                [a, b] => {
                    let a_prober = self.get_column_index(
                        atoms,
                        binding_info,
                        a.atom,
                        a.column,
                        &prepared_indexes[0],
                    );
                    let b_prober = self.get_column_index(
                        atoms,
                        binding_info,
                        b.atom,
                        b.column,
                        &prepared_indexes[1],
                    );

                    let ((smaller, smaller_scan), (larger, larger_scan)) =
                        if a_prober.len() <= b_prober.len() {
                            ((&a_prober, a), (&b_prober, b))
                        } else {
                            ((&b_prober, b), (&a_prober, a))
                        };

                    let smaller_atom = smaller_scan.atom;
                    let larger_atom = larger_scan.atom;
                    let large_info = &self.db.tables[atoms[larger_atom].table];
                    let large_table = large_info.table.as_ref();
                    let large_has_stale = large_table.has_stale_rows();
                    let small_info = &self.db.tables[atoms[smaller_atom].table];
                    let small_table = small_info.table.as_ref();
                    let small_has_stale = small_table.has_stale_rows();
                    let mut updates = FrameUpdates::with_capacity(cmp::min(chunk_size, cur_size));
                    for_each_leader!(smaller, |val, small_sub| {
                        if let Some(large_sub) = larger.get_subset(val) {
                            updates.push_binding(*var, val[0]);
                            if small_sub.size() <= 16 {
                                let small_sub = refine_subset(
                                    small_sub,
                                    &smaller_scan.cs,
                                    &small_table,
                                    small_has_stale,
                                    pool,
                                );
                                if small_sub.is_empty() {
                                    updates.rollback();
                                    return;
                                }
                                updates.refine_atom_subset(smaller_atom, small_sub);
                            } else {
                                let smaller_node = smaller.node.get_cached_trie_node(
                                    smaller_scan.column,
                                    val[0],
                                    &smaller_scan.cs,
                                    small_info,
                                    || {
                                        refine_subset(
                                            small_sub,
                                            &smaller_scan.cs,
                                            &small_table,
                                            small_has_stale,
                                            pool,
                                        )
                                    },
                                );
                                if smaller_node.subset.is_empty() {
                                    updates.rollback();
                                    return;
                                }
                                updates.refine_atom(smaller_atom, smaller_node);
                            }
                            if large_sub.size() <= 16 {
                                let large_sub = refine_subset(
                                    large_sub,
                                    &larger_scan.cs,
                                    &large_table,
                                    large_has_stale,
                                    pool,
                                );
                                if large_sub.is_empty() {
                                    updates.rollback();
                                    return;
                                }
                                updates.refine_atom_subset(larger_atom, large_sub);
                            } else {
                                let larger_node = larger.node.get_cached_trie_node(
                                    larger_scan.column,
                                    val[0],
                                    &larger_scan.cs,
                                    large_info,
                                    || {
                                        refine_subset(
                                            large_sub,
                                            &larger_scan.cs,
                                            &large_table,
                                            large_has_stale,
                                            pool,
                                        )
                                    },
                                );
                                if larger_node.subset.is_empty() {
                                    updates.rollback();
                                    return;
                                }
                                updates.refine_atom(larger_atom, larger_node);
                            }
                            updates.finish_frame();
                            if updates.frames() >= chunk_size {
                                drain_updates!(updates);
                            }
                        }
                    });
                    drain_updates!(updates);

                    binding_info.move_back(a.atom, a_prober);
                    binding_info.move_back(b.atom, b_prober);
                }
                rest => {
                    let mut smallest = 0;
                    let mut smallest_size = usize::MAX;
                    let mut probers = Vec::with_capacity(rest.len());
                    for (i, (scan, prepared)) in rest.iter().zip(prepared_indexes).enumerate() {
                        let prober = self.get_column_index(
                            atoms,
                            binding_info,
                            scan.atom,
                            scan.column,
                            prepared,
                        );
                        let size = prober.len();
                        if size < smallest_size {
                            smallest = i;
                            smallest_size = size;
                        }
                        probers.push(prober);
                    }

                    let main_spec = &rest[smallest];
                    let main_spec_info = &self.db.tables[atoms[main_spec.atom].table];
                    let main_spec_table = main_spec_info.table.as_ref();
                    let main_spec_has_stale = main_spec_table.has_stale_rows();
                    // Pre-compute has_stale for each scan to avoid vtable calls in the hot loop.
                    let rest_has_stale: SmallVec<[bool; 3]> = rest
                        .iter()
                        .map(|scan| {
                            self.db.tables[atoms[scan.atom].table]
                                .table
                                .as_ref()
                                .has_stale_rows()
                        })
                        .collect();

                    if smallest_size != 0 {
                        // Smallest leads the scan
                        let mut updates =
                            FrameUpdates::with_capacity(cmp::min(chunk_size, cur_size));
                        for_each_leader!(probers[smallest], |key, sub| {
                            updates.push_binding(*var, key[0]);
                            for (i, scan) in rest.iter().enumerate() {
                                if i == smallest {
                                    continue;
                                }
                                if let Some(sub) = probers[i].get_subset(key) {
                                    let table =
                                        self.db.tables[atoms[rest[i].atom].table].table.as_ref();
                                    if sub.size() <= 16 {
                                        let sub = refine_subset(
                                            sub,
                                            &rest[i].cs,
                                            &table,
                                            rest_has_stale[i],
                                            pool,
                                        );
                                        if sub.is_empty() {
                                            updates.rollback();
                                            return;
                                        }
                                        updates.refine_atom_subset(scan.atom, sub);
                                    } else {
                                        let node = probers[i].node.get_cached_trie_node(
                                            scan.column,
                                            key[0],
                                            &rest[i].cs,
                                            &self.db.tables[atoms[scan.atom].table],
                                            || {
                                                refine_subset(
                                                    sub,
                                                    &rest[i].cs,
                                                    &table,
                                                    rest_has_stale[i],
                                                    pool,
                                                )
                                            },
                                        );
                                        if node.subset.is_empty() {
                                            updates.rollback();
                                            return;
                                        }
                                        updates.refine_atom(scan.atom, node);
                                    }
                                } else {
                                    updates.rollback();
                                    // Empty intersection.
                                    return;
                                }
                            }
                            if sub.size() <= 16 {
                                let main_sub = refine_subset(
                                    sub,
                                    &main_spec.cs,
                                    &main_spec_table,
                                    main_spec_has_stale,
                                    pool,
                                );
                                if main_sub.is_empty() {
                                    updates.rollback();
                                    return;
                                }
                                updates.refine_atom_subset(main_spec.atom, main_sub);
                            } else {
                                let main_node = probers[smallest].node.get_cached_trie_node(
                                    main_spec.column,
                                    key[0],
                                    &main_spec.cs,
                                    main_spec_info,
                                    || {
                                        refine_subset(
                                            sub,
                                            &main_spec.cs,
                                            &main_spec_table,
                                            main_spec_has_stale,
                                            pool,
                                        )
                                    },
                                );
                                if main_node.subset.is_empty() {
                                    updates.rollback();
                                    return;
                                }
                                updates.refine_atom(main_spec.atom, main_node);
                            }
                            updates.finish_frame();
                            if updates.frames() >= chunk_size {
                                drain_updates!(updates);
                            }
                        });
                        drain_updates!(updates);
                    }
                    for (spec, prober) in rest.iter().zip(probers.into_iter()) {
                        binding_info.move_back(spec.atom, prober);
                    }
                }
            },
            JoinStage::FusedIntersect {
                cover,
                bind,
                to_intersect,
            } if to_intersect.is_empty() => {
                let is_leaf_scan = leaf_scans[cur];
                let cover_atom = cover.to_index.atom;
                if binding_info.has_empty_subset(cover_atom) {
                    return;
                }
                if is_leaf_scan {
                    let table = self.db.tables[atoms[cover_atom].table].table.as_ref();
                    let cover_node = binding_info.unwrap_val(cover_atom);
                    let cover_subset = cover_node.subset.as_ref();

                    let proj =
                        SmallVec::<[ColumnId; 4]>::from_iter(bind.iter().map(|(col, _)| *col));
                    let vars = bind.iter().map(|(_, var)| *var).collect();
                    let mut buf = TaggedRowBuffer::new_inline(bind.len());
                    table.scan_project(
                        cover_subset,
                        &proj,
                        Offset::new(0),
                        usize::MAX,
                        &cover.constraints,
                        &mut buf,
                    );

                    if buf.is_empty() {
                        binding_info.move_back_node(cover_atom, cover_node);
                        return;
                    }

                    binding_info.binding_sets.push((vars, Arc::new(buf)));
                    let mut updates = FrameUpdates::with_capacity(1);
                    updates.finish_frame();
                    drain_updates!(updates);
                    binding_info.binding_sets.pop();
                    binding_info.move_back_node(cover_atom, cover_node);
                } else {
                    let proj =
                        SmallVec::<[ColumnId; 4]>::from_iter(bind.iter().map(|(col, _)| *col));
                    let cover_node = binding_info.unwrap_val(cover_atom);
                    let cover_subset = cover_node.subset.as_ref();
                    let mut offset = Offset::new(0);
                    let mut buffer = TaggedRowBuffer::new(bind.len());
                    let mut updates = FrameUpdates::with_capacity(cmp::min(chunk_size, cur_size));
                    loop {
                        buffer.clear();
                        let table = &self.db.tables[atoms[cover_atom].table].table;
                        let next = table.scan_project(
                            cover_subset,
                            &proj,
                            offset,
                            chunk_size,
                            &cover.constraints,
                            &mut buffer,
                        );
                        for (row, key) in buffer.iter() {
                            updates.refine_atom_dense(cover_atom, OffsetRange::new(row, row.inc()));
                            // bind the values
                            for (i, (_, var)) in bind.iter().enumerate() {
                                updates.push_binding(*var, key[i]);
                            }
                            updates.finish_frame();
                            if updates.frames() >= chunk_size {
                                drain_updates!(updates);
                            }
                        }
                        if let Some(next) = next {
                            offset = next;
                            continue;
                        }
                        break;
                    }
                    drain_updates!(updates);
                    // Restore the subsets we swapped out.
                    binding_info.move_back_node(cover_atom, cover_node);
                }
            }
            JoinStage::FusedIntersect {
                cover,
                bind,
                to_intersect,
            } => {
                let cover_atom = cover.to_index.atom;
                if binding_info.has_empty_subset(cover_atom) {
                    return;
                }
                let index_probers = to_intersect
                    .iter()
                    .enumerate()
                    .map(|(i, (spec, _))| {
                        (
                            i,
                            spec.to_index.atom,
                            self.get_index(
                                atoms,
                                spec.to_index.atom,
                                binding_info,
                                spec.to_index.vars.iter().copied(),
                                &prepared_indexes[i],
                            ),
                        )
                    })
                    .collect::<SmallVec<[(usize, AtomId, Prober<'_>); 4]>>();
                // Pre-compute has_stale per prober to avoid vtable calls in the hot loop.
                let index_has_stale: SmallVec<[bool; 4]> = index_probers
                    .iter()
                    .map(|(_, atom, _)| {
                        self.db.tables[atoms[*atom].table]
                            .table
                            .as_ref()
                            .has_stale_rows()
                    })
                    .collect();
                let proj = SmallVec::<[ColumnId; 4]>::from_iter(bind.iter().map(|(col, _)| *col));
                let cover_node = binding_info.unwrap_val(cover_atom);
                let cover_subset = cover_node.subset.as_ref();
                let mut cur = Offset::new(0);
                let mut buffer = TaggedRowBuffer::new(bind.len());
                let mut updates = FrameUpdates::with_capacity(cmp::min(chunk_size, cur_size));
                loop {
                    buffer.clear();
                    let table = &self.db.tables[atoms[cover_atom].table].table;
                    let next = table.scan_project(
                        cover_subset,
                        &proj,
                        cur,
                        chunk_size,
                        &cover.constraints,
                        &mut buffer,
                    );
                    'mid: for (row, key) in buffer.iter() {
                        updates.refine_atom_dense(cover_atom, OffsetRange::new(row, row.inc()));
                        // bind the values
                        for (i, (_, var)) in bind.iter().enumerate() {
                            updates.push_binding(*var, key[i]);
                        }
                        // now probe each remaining indexes
                        for (prober_idx, (i, atom, prober)) in index_probers.iter().enumerate() {
                            // create a key: to_intersect indexes into the key from the cover
                            let index_cols = &to_intersect[*i].1;
                            // Fast path for the common single-column case: avoid SmallVec collect.
                            let index_key_buf: SmallVec<[Value; 4]>;
                            let index_key: &[Value] = if let [col] = index_cols.as_slice() {
                                std::slice::from_ref(&key[col.index()])
                            } else {
                                index_key_buf =
                                    index_cols.iter().map(|col| key[col.index()]).collect();
                                &index_key_buf
                            };
                            let Some(subset) = prober.get_subset(index_key) else {
                                updates.rollback();
                                // There are no possible values for this subset
                                continue 'mid;
                            };
                            // apply any constraints needed in this scan.
                            let table_info = &self.db.tables[atoms[*atom].table];
                            let cs = &to_intersect[*i].0.constraints;
                            let subset = refine_subset(
                                subset,
                                cs,
                                &table_info.table.as_ref(),
                                index_has_stale[prober_idx],
                                pool,
                            );
                            if subset.is_empty() {
                                updates.rollback();
                                // There are no possible values for this subset
                                continue 'mid;
                            }
                            updates.refine_atom_subset(*atom, subset);
                        }
                        updates.finish_frame();
                        if updates.frames() >= chunk_size {
                            drain_updates!(updates);
                        }
                    }
                    if let Some(next) = next {
                        cur = next;
                        continue;
                    }
                    break;
                }
                // TODO: special-case the scenario when the cover doesn't need
                // deduping (and hence we can do a straight scan: e.g. when the
                // cover is binding a superset of the primary key for the
                // table).
                drain_updates!(updates);
                // Restore the subsets we swapped out.
                binding_info.move_back_node(cover_atom, cover_node);
                for (_, atom, prober) in index_probers {
                    binding_info.move_back(atom, prober);
                }
            }
            JoinStage::FusedIntersectMat {
                cover,
                mode,
                bind,
                to_intersect,
            } if leaf_scans[cur]
                && to_intersect.is_empty()
                && matches!(
                    mode,
                    MatScanMode::Full | MatScanMode::KeyOnly | MatScanMode::Value(_)
                ) =>
            {
                // Leaf-scan factorization for FusedIntersectMat: flatten the materialization into
                // one `TaggedRowBuffer`, push it onto `binding_sets`, and recurse to the leaf once.
                let cover_mat = binding_info.materializations[*cover].clone();
                let vars: SmallVec<[Variable; 4]> = bind.iter().map(|(_, v)| *v).collect();
                let mut buf = TaggedRowBuffer::new_inline(bind.len());
                let mut row_scratch: SmallVec<[Value; 8]> = SmallVec::new();
                match mode {
                    MatScanMode::Full => {
                        for group in cover_mat.iter() {
                            let group_key = group.0;
                            let group_key_len = group_key.len();
                            for non_keys in group.1.iter() {
                                row_scratch.clear();
                                for (col, _) in bind.iter() {
                                    let val = if col.index() < group_key_len {
                                        group_key[col.index()]
                                    } else {
                                        non_keys[col.index() - group_key_len]
                                    };
                                    row_scratch.push(val);
                                }
                                buf.add_row(RowId::new(0), &row_scratch);
                            }
                        }
                    }
                    MatScanMode::KeyOnly => {
                        for group in cover_mat.iter() {
                            let group_key = group.0;
                            row_scratch.clear();
                            for (col, _) in bind.iter() {
                                debug_assert!(col.index() < group_key.len());
                                row_scratch.push(group_key[col.index()]);
                            }
                            buf.add_row(RowId::new(0), &row_scratch);
                        }
                    }
                    MatScanMode::Value(index_vars) => {
                        let keys: Vec<Value> = index_vars
                            .iter()
                            .map(|var| binding_info.bindings[*var])
                            .collect();
                        if let Some(group) = cover_mat.get(&keys) {
                            for vals in group.iter() {
                                debug_assert!(vals.len() == bind.len());
                                row_scratch.clear();
                                for (col, _) in bind.iter() {
                                    row_scratch.push(vals[col.index()]);
                                }
                                buf.add_row(RowId::new(0), &row_scratch);
                            }
                        }
                    }
                    MatScanMode::Lookup(_) => unreachable!("guarded above"),
                }
                if buf.is_empty() {
                    return;
                }
                binding_info.binding_sets.push((vars, Arc::new(buf)));
                let mut updates = FrameUpdates::with_capacity(1);
                updates.finish_frame();
                drain_updates!(updates);
                binding_info.binding_sets.pop();
            }
            JoinStage::FusedIntersectMat {
                cover,
                mode,
                bind,
                to_intersect,
            } => {
                let cover_mat = binding_info.materializations[*cover].clone();
                let mut updates = FrameUpdates::with_capacity(cmp::min(chunk_size, cur_size));
                let probers = to_intersect
                    .iter()
                    .zip(prepared_indexes)
                    .map(|((spec, _), prepared)| {
                        self.get_index(
                            atoms,
                            spec.to_index.atom,
                            binding_info,
                            spec.to_index.vars.iter().copied(),
                            prepared,
                        )
                    })
                    .collect::<SmallVec<[Prober<'_>; 4]>>();
                // Pre-compute has_stale per prober to avoid vtable calls in the hot loop.
                let probers_has_stale: SmallVec<[bool; 4]> = to_intersect
                    .iter()
                    .map(|(spec, _)| {
                        self.db.tables[atoms[spec.to_index.atom].table]
                            .table
                            .as_ref()
                            .has_stale_rows()
                    })
                    .collect();

                let mut key = Vec::with_capacity(4);
                let mut prune_probers = |updates: &mut FrameUpdates,
                                         mat_key: Option<&[Value]>,
                                         mat_non_key: Option<&[Value]>|
                 -> bool {
                    for (j, ((spec, cols), prober)) in
                        to_intersect.iter().zip(probers.iter()).enumerate()
                    {
                        key.clear();
                        for col in cols.iter() {
                            let val = match mat_key {
                                Some(mat_key) => {
                                    if col.index() < mat_key.len() {
                                        mat_key[col.index()]
                                    } else {
                                        mat_non_key.unwrap()[col.index() - mat_key.len()]
                                    }
                                }
                                None => mat_non_key.unwrap()[col.index()],
                            };
                            key.push(val);
                        }
                        if let Some(subset) = prober.get_subset(&key) {
                            let subset = refine_subset(
                                subset,
                                &spec.constraints,
                                &self.db.tables[atoms[spec.to_index.atom].table]
                                    .table
                                    .as_ref(),
                                probers_has_stale[j],
                                pool,
                            );
                            if subset.is_empty() {
                                return false;
                            }
                            updates.refine_atom_subset(spec.to_index.atom, subset);
                        } else {
                            return false;
                        }
                    }
                    true
                };

                match mode {
                    MatScanMode::Full | MatScanMode::KeyOnly => {
                        // enumerate keys
                        for group in cover_mat.iter() {
                            let group_key = group.0;
                            let group_val = group.1;
                            let group_key_len = group_key.len();
                            if mode == &MatScanMode::Full {
                                // enumerate non-keys
                                for non_keys in group_val.iter() {
                                    for (col, var) in bind.iter() {
                                        if col.index() < group_key_len {
                                            updates.push_binding(*var, group_key[col.index()]);
                                        }
                                    }

                                    // TODO: optimization that guaratees all keys come before non-keys
                                    for (col, var) in bind.iter() {
                                        if col.index() >= group_key_len {
                                            updates.push_binding(
                                                *var,
                                                non_keys[col.index() - group_key_len],
                                            );
                                        }
                                    }
                                    if prune_probers(&mut updates, Some(group_key), Some(non_keys))
                                    {
                                        updates.finish_frame();
                                    } else {
                                        updates.rollback();
                                    }
                                }
                            } else if mode == &MatScanMode::KeyOnly {
                                for (col, var) in bind.iter() {
                                    debug_assert!(col.index() < group_key_len);
                                    updates.push_binding(*var, group_key[col.index()]);
                                }
                                if prune_probers(&mut updates, Some(group_key), None) {
                                    updates.finish_frame();
                                } else {
                                    updates.rollback();
                                }
                            }
                        }
                    }
                    MatScanMode::Value(index_vars) | MatScanMode::Lookup(index_vars) => {
                        let keys = index_vars
                            .iter()
                            .map(|var| binding_info.bindings[*var])
                            .collect::<Vec<Value>>();
                        // lookup keys
                        if let Some(group) = cover_mat.get(&keys) {
                            if matches!(mode, MatScanMode::Lookup(_)) {
                                debug_assert_eq!(to_intersect.len(), 0);
                                debug_assert_eq!(bind.len(), 0);
                                if group.len() > 0 {
                                    updates.finish_frame();
                                }
                                drain_updates!(updates);
                            } else {
                                // enumerate non-keys
                                // for vals in group.value().iter() {
                                for vals in group.iter() {
                                    debug_assert!(vals.len() == bind.len()); // TODO: not true for non-full query
                                    for (col, var) in bind.iter() {
                                        updates.push_binding(*var, vals[col.index()]);
                                    }
                                    if prune_probers(&mut updates, None, Some(vals)) {
                                        updates.finish_frame();
                                    } else {
                                        updates.rollback();
                                    }
                                    if updates.frames() >= chunk_size {
                                        drain_updates!(updates);
                                    }
                                }
                            }
                        }
                    }
                }

                drain_updates!(updates);
                for (spec, prober) in to_intersect.iter().zip(probers) {
                    binding_info.move_back(spec.0.to_index.atom, prober);
                }
            }
        }
    }
}

const LOCAL_ACTION_BATCH_SIZE: usize = 128;

/// A trait used to abstract over different ways of buffering actions together
/// before running them.
///
/// This trait exists as a fairly ad-hoc wrapper over its two implementations.
/// It allows us to avoid duplicating the (somewhat monstrous) `run_plan` method
/// for serial and parallel modes.
trait ActionBuffer<'state, A: NumericId>: Send {
    type AsLocal<'a>: ActionBuffer<'state, A>
    where
        'state: 'a;
    type AsGlobalSerial<'a>: ActionBuffer<'state, A>
    where
        'state: 'a;

    /// Expand the binding sets to individual bindings and
    /// call push_bindings
    fn push_bindings_factorized(
        &mut self,
        action: A,
        bindings: &mut DenseIdMap<Variable, Value>,
        binding_sets: &BindingSet,
        exec_state: &ExecutionState<'state>,
    ) {
        expand_binding_sets(self, action, bindings, binding_sets, 0, exec_state);
    }

    /// Push the given bindings to be executed for the specified action. If this
    /// buffer has built up a sufficient batch size, it may execute
    /// `to_exec_state` and then execute the action.
    ///
    /// NB: `push_bindings` makes module-specific assumptions on what values are passed to
    /// `bindings` for a common `action`. This is not a general-purpose trait for that reason and
    /// it should not, in general, be used outside of this module.
    fn push_bindings(
        &mut self,
        action: A,
        bindings: &DenseIdMap<Variable, Value>,
        to_exec_state: impl FnMut() -> ExecutionState<'state>,
    );

    /// Execute any remaining actions associated with this buffer.
    fn flush(&mut self, exec_state: &mut ExecutionState);

    /// Execute `work`, potentially asynchronously, with a mutable reference to
    /// an action buffer, potentially handed off to a different thread.
    ///
    /// Callers [`BorrowedLocalState`] values that may be modified by work, or
    /// cloned first and then have a separate copy modified by `work`. Callers
    /// should assume that `local` _is_ modified synchronously.
    // NB: Earlier versions of this method had BorrowedLocalState be a generic instead, but this
    // ran into difficulties when we needed to pass multiple mutable references.
    fn recur<'local>(
        &mut self,
        local: BorrowedLocalState<'local>,
        to_exec_state: impl FnMut() -> ExecutionState<'state> + Send + 'state,
        work: impl for<'a> FnOnce(BorrowedLocalState<'a>, &mut Self::AsLocal<'a>) + Send + 'state,
    );

    /// Run one coarse partition as a global pool job, using a buffer whose
    /// recursive work and action execution are both serial.  Implementations
    /// that return `false` from [`Self::supports_global_partition`] execute the
    /// callback inline; the scoped implementations enqueue exactly one job.
    fn recur_global_serial<'local>(
        &mut self,
        local: BorrowedLocalState<'local>,
        to_exec_state: impl FnMut() -> ExecutionState<'state> + Send + 'state,
        work: impl for<'a> FnOnce(BorrowedLocalState<'a>, &mut Self::AsGlobalSerial<'a>) + Send + 'state,
    );

    /// The unit at which you should batch updates passed to calls to `recur`,
    /// potentially depending on the current level of recursion.
    ///
    /// As of right now this is just a hard-coded value. We may change it in the
    /// future to fan out more at higher levels though.
    fn morsel_size(&mut self, _level: usize, _total: usize) -> usize {
        256
    }

    /// Whether this buffer supports parallel drain operations.
    ///
    /// When `false`, `drain_updates` will use the serial path even at `cur <= 1`,
    /// avoiding the per-frame `ExecutionState::clone()` overhead.
    fn supports_parallel_drain(&self) -> bool {
        true
    }

    /// Whether this buffer can enqueue independent top-level partitions.
    fn supports_global_partition(&self) -> bool {
        false
    }
}

/// The action buffer we use if we are executing in a single-threaded
/// environment. It builds up local batches and then flushes them inline.
struct InPlaceActionBuffer<'a> {
    rule_set: &'a RuleSet,
    match_counter: &'a MatchCounter,
    batches: DenseIdMap<ActionId, ActionState>,
}

impl<'a, 'outer: 'a> ActionBuffer<'a, ActionId> for InPlaceActionBuffer<'outer> {
    type AsLocal<'b>
        = Self
    where
        'a: 'b;
    type AsGlobalSerial<'b>
        = Self
    where
        'a: 'b;

    fn push_bindings(
        &mut self,
        action: ActionId,
        bindings: &DenseIdMap<Variable, Value>,
        mut to_exec_state: impl FnMut() -> ExecutionState<'a>,
    ) {
        let action_state = self
            .batches
            .get_or_insert(action, || ActionState::new(LOCAL_ACTION_BATCH_SIZE));
        action_state.n_runs += 1;
        action_state.len += 1;
        let action_info = &self.rule_set.actions[action];
        // SAFETY: `used_vars` is a constant per-rule. This module only ever calls it with
        // `bindings` produced by the same join.
        unsafe {
            action_state.bindings.push(bindings, &action_info.used_vars);
        }
        if action_state.len >= LOCAL_ACTION_BATCH_SIZE {
            let mut state = to_exec_state();
            let succeeded = state.run_instrs(&action_info.instrs, &mut action_state.bindings);
            action_state.bindings.clear();
            self.match_counter.inc_matches(action, succeeded);
            action_state.len = 0;
        }
    }

    fn flush(&mut self, exec_state: &mut ExecutionState) {
        flush_action_states(
            exec_state,
            &mut self.batches,
            self.rule_set,
            self.match_counter,
        );
    }

    fn recur<'local>(
        &mut self,
        local: BorrowedLocalState<'local>,
        _to_exec_state: impl FnMut() -> ExecutionState<'a> + Send + 'a,
        work: impl for<'b> FnOnce(BorrowedLocalState<'b>, &mut Self) + Send + 'a,
    ) {
        work(local, self)
    }

    fn recur_global_serial<'local>(
        &mut self,
        local: BorrowedLocalState<'local>,
        _to_exec_state: impl FnMut() -> ExecutionState<'a> + Send + 'a,
        work: impl for<'b> FnOnce(BorrowedLocalState<'b>, &mut Self) + Send + 'a,
    ) {
        work(local, self)
    }

    fn supports_parallel_drain(&self) -> bool {
        false
    }
}

/// Strictly serial action buffer used inside one globally scheduled top-index
/// shard. It shares the rule-set match counter with sibling shards, but executes
/// all recursive join and action work inline. It deliberately has no
/// `needs_flush` flag: its owner unconditionally flushes it once after the shard
/// callback returns, and recursive calls reuse the same buffer synchronously.
struct SerialScopedActionBuffer<'scope> {
    rule_set: &'scope RuleSet,
    match_counter: Arc<MatchCounter>,
    batches: DenseIdMap<ActionId, ActionState>,
}

impl<'scope> SerialScopedActionBuffer<'scope> {
    fn new(rule_set: &'scope RuleSet, match_counter: Arc<MatchCounter>) -> Self {
        Self {
            rule_set,
            match_counter,
            batches: Default::default(),
        }
    }
}

impl<'scope> ActionBuffer<'scope, ActionId> for SerialScopedActionBuffer<'scope> {
    type AsLocal<'a>
        = Self
    where
        'scope: 'a;
    type AsGlobalSerial<'a>
        = Self
    where
        'scope: 'a;

    fn push_bindings(
        &mut self,
        action: ActionId,
        bindings: &DenseIdMap<Variable, Value>,
        mut to_exec_state: impl FnMut() -> ExecutionState<'scope>,
    ) {
        let batch_size = action_batch_size();
        let action_state = self
            .batches
            .get_or_insert(action, || ActionState::new(batch_size));
        action_state.n_runs += 1;
        action_state.len += 1;
        let action_info = &self.rule_set.actions[action];
        // SAFETY: `used_vars` is constant for the rule and the bindings come
        // from this rule's join plan.
        unsafe {
            action_state.bindings.push(bindings, &action_info.used_vars);
        }
        if action_state.len >= batch_size {
            let succeeded =
                to_exec_state().run_instrs(&action_info.instrs, &mut action_state.bindings);
            action_state.bindings.clear();
            self.match_counter.inc_matches(action, succeeded);
            action_state.len = 0;
        }
    }

    fn flush(&mut self, exec_state: &mut ExecutionState) {
        flush_action_states(
            exec_state,
            &mut self.batches,
            self.rule_set,
            self.match_counter.as_ref(),
        );
    }

    fn recur<'local>(
        &mut self,
        local: BorrowedLocalState<'local>,
        _to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'a> FnOnce(BorrowedLocalState<'a>, &mut Self) + Send + 'scope,
    ) {
        work(local, self);
    }

    fn recur_global_serial<'local>(
        &mut self,
        local: BorrowedLocalState<'local>,
        _to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'a> FnOnce(BorrowedLocalState<'a>, &mut Self) + Send + 'scope,
    ) {
        work(local, self);
    }

    fn supports_parallel_drain(&self) -> bool {
        false
    }
}

/// An action buffer that hands off batches of actions to scoped worker tasks.
struct ScopedActionBuffer<'inner, 'scope> {
    scope: &'inner Scope<'scope>,
    rule_set: &'scope RuleSet,
    match_counter: Arc<MatchCounter>,
    batches: DenseIdMap<ActionId, ActionState>,
    needs_flush: bool,
}

impl<'inner, 'scope> ScopedActionBuffer<'inner, 'scope> {
    fn new(
        scope: &'inner Scope<'scope>,
        rule_set: &'scope RuleSet,
        match_counter: Arc<MatchCounter>,
    ) -> Self {
        Self {
            scope,
            rule_set,
            batches: Default::default(),
            match_counter,
            needs_flush: false,
        }
    }
}

impl<'scope> ActionBuffer<'scope, ActionId> for ScopedActionBuffer<'_, 'scope> {
    type AsLocal<'a>
        = ScopedActionBuffer<'a, 'scope>
    where
        'scope: 'a;
    type AsGlobalSerial<'a>
        = SerialScopedActionBuffer<'scope>
    where
        'scope: 'a;
    fn push_bindings(
        &mut self,
        action: ActionId,
        bindings: &DenseIdMap<Variable, Value>,
        mut to_exec_state: impl FnMut() -> ExecutionState<'scope>,
    ) {
        self.needs_flush = true;
        let batch_size = action_batch_size();
        let action_state = self
            .batches
            .get_or_insert(action, || ActionState::new(batch_size));
        action_state.n_runs += 1;
        action_state.len += 1;
        let action_info = &self.rule_set.actions[action];
        // SAFETY: `used_vars` is a constant per-rule. This module only ever calls it with
        // `bindings` produced by the same join.
        unsafe {
            action_state.bindings.push(bindings, &action_info.used_vars);
        }
        if action_state.len >= batch_size {
            let mut state = to_exec_state();
            let mut bindings = mem::replace(&mut action_state.bindings, Bindings::new(batch_size));
            action_state.len = 0;
            let match_counter = self.match_counter.clone();
            self.scope.spawn(move |_| {
                let succeeded = state.run_instrs(&action_info.instrs, &mut bindings);
                match_counter.inc_matches(action, succeeded);
            });
        }
    }

    fn flush(&mut self, exec_state: &mut ExecutionState) {
        flush_action_states(
            exec_state,
            &mut self.batches,
            self.rule_set,
            self.match_counter.as_ref(),
        );
        self.needs_flush = false;
    }
    fn recur<'local>(
        &mut self,
        mut local: BorrowedLocalState<'local>,
        mut to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'a> FnOnce(BorrowedLocalState<'a>, &mut ScopedActionBuffer<'a, 'scope>)
        + Send
        + 'scope,
    ) {
        let rule_set = self.rule_set;
        let match_counter = self.match_counter.clone();
        let mut inner = local.clone_state();
        // Keep recursive join work on the current worker when possible. If
        // coarse top-index partitioning was unavailable, stalled workers cause
        // the private deque to donate older siblings to the global queue.
        self.scope.spawn_local(move |scope| {
            let mut buf: ScopedActionBuffer<'_, 'scope> = ScopedActionBuffer {
                scope,
                rule_set,
                match_counter,
                needs_flush: false,
                batches: Default::default(),
            };
            work(inner.borrow_mut(), &mut buf);
            if buf.needs_flush {
                flush_action_states(
                    &mut to_exec_state(),
                    &mut buf.batches,
                    buf.rule_set,
                    buf.match_counter.as_ref(),
                );
            }
        });
    }

    fn recur_global_serial<'local>(
        &mut self,
        mut local: BorrowedLocalState<'local>,
        mut to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'a> FnOnce(BorrowedLocalState<'a>, &mut SerialScopedActionBuffer<'scope>)
        + Send
        + 'scope,
    ) {
        let rule_set = self.rule_set;
        let match_counter = self.match_counter.clone();
        let mut inner = local.clone_state();
        self.scope.spawn_global(move |_| {
            let mut buf = SerialScopedActionBuffer::new(rule_set, match_counter);
            work(inner.borrow_mut(), &mut buf);
            // Unlike a nested `ScopedActionBuffer`, this buffer is owned by the
            // one shard job and is always flushed exactly once before it exits.
            buf.flush(&mut to_exec_state());
        });
    }

    fn morsel_size(&mut self, _level: usize, _total: usize) -> usize {
        // Lower morsel size to increase parallelism.
        match _level {
            0 if _total > 2 => 32,
            _ => 256,
        }
    }

    fn supports_global_partition(&self) -> bool {
        true
    }
}

fn expand_binding_sets<'state, A: NumericId, BUF: ActionBuffer<'state, A> + ?Sized>(
    action_buf: &mut BUF,
    action: A,
    bindings: &mut DenseIdMap<Variable, Value>,
    binding_sets: &BindingSet,
    idx: usize,
    exec_state: &ExecutionState<'state>,
) {
    if exec_state.should_stop() {
        return;
    }
    if idx >= binding_sets.len() {
        action_buf.push_bindings(action, bindings, || exec_state.clone());
        return;
    }
    if idx + 1 == binding_sets.len() {
        let (vars, buf) = &binding_sets[idx];
        for (_, row) in buf.iter() {
            if exec_state.should_stop() {
                return;
            }
            for (var, val) in vars.iter().zip(row.iter()) {
                bindings.insert(*var, *val);
            }
            action_buf.push_bindings(action, bindings, || exec_state.clone());
        }
        return;
    }
    let (vars, buf) = &binding_sets[idx];
    for (_, row) in buf.iter() {
        for (var, val) in vars.iter().zip(row.iter()) {
            bindings.insert(*var, *val);
        }
        expand_binding_sets(
            action_buf,
            action,
            bindings,
            binding_sets,
            idx + 1,
            exec_state,
        );
    }
}

fn flush_action_states(
    exec_state: &mut ExecutionState,
    actions: &mut DenseIdMap<ActionId, ActionState>,
    rule_set: &RuleSet,
    match_counter: &MatchCounter,
) {
    for (action, ActionState { bindings, len, .. }) in actions.iter_mut() {
        if *len > 0 {
            let succeeded = exec_state.run_instrs(&rule_set.actions[action].instrs, bindings);
            bindings.clear();
            match_counter.inc_matches(action, succeeded);
            *len = 0;
        }
    }
}

struct InPlaceMaterializer<'a> {
    specs: &'a DenseIdMap<MatId, MatSpec>,
    materializations: DenseIdMap<MatId, IndexMap<Vec<Value>, RowBuffer>>,
    scratch_key: Vec<Value>,
    scratch_val: Vec<Value>,
}

impl<'a> ActionBuffer<'a, MatId> for InPlaceMaterializer<'a> {
    type AsLocal<'b>
        = Self
    where
        'a: 'b;
    type AsGlobalSerial<'b>
        = Self
    where
        'a: 'b;

    fn push_bindings(
        &mut self,
        mat_id: MatId,
        bindings: &DenseIdMap<Variable, Value>,
        _to_exec_state: impl FnMut() -> ExecutionState<'a>,
    ) {
        let mat = self
            .materializations
            .get_mut(mat_id)
            .expect("invalid mat id");
        let spec = self.specs.get(mat_id).expect("invalid mat id");
        self.scratch_key.clear();
        for key in spec.msg_vars.iter().map(|var| bindings[*var]) {
            self.scratch_key.push(key);
        }
        self.scratch_val.clear();
        for val in spec.val_vars.iter().map(|var| bindings[*var]) {
            self.scratch_val.push(val);
        }
        if self.scratch_val.is_empty() {
            self.scratch_val.push(Value::stale());
        }
        if let Some(buffer) = mat.get_mut(&self.scratch_key) {
            buffer.add_row(&self.scratch_val);
        } else {
            let mut buffer = RowBuffer::new(usize::max(spec.val_vars.len(), 1));
            buffer.add_row(&self.scratch_val);
            mat.insert(self.scratch_key.clone(), buffer);
        }
    }

    fn flush(&mut self, _exec_state: &mut ExecutionState) {
        // No-op for in-place materializer.
    }

    fn recur<'local>(
        &mut self,
        local: BorrowedLocalState<'local>,
        _to_exec_state: impl FnMut() -> ExecutionState<'a> + Send + 'a,
        work: impl for<'b> FnOnce(BorrowedLocalState<'b>, &mut Self) + Send + 'a,
    ) {
        work(local, self)
    }

    fn recur_global_serial<'local>(
        &mut self,
        local: BorrowedLocalState<'local>,
        _to_exec_state: impl FnMut() -> ExecutionState<'a> + Send + 'a,
        work: impl for<'b> FnOnce(BorrowedLocalState<'b>, &mut Self) + Send + 'a,
    ) {
        work(local, self)
    }

    fn supports_parallel_drain(&self) -> bool {
        false
    }
}

/// Serial recursive view of a scoped materializer.  Sibling top-level shards
/// still share the concurrent output maps, but this buffer never schedules
/// deeper work itself.
struct SerialScopedMaterializer {
    specs: Arc<DenseIdMap<MatId, MatSpec>>,
    materializations: Arc<DenseIdMap<MatId, Arc<DashMap<Vec<Value>, RowBuffer>>>>,
    scratch_key: Vec<Value>,
    scratch_val: Vec<Value>,
}

impl SerialScopedMaterializer {
    fn new(
        specs: Arc<DenseIdMap<MatId, MatSpec>>,
        materializations: Arc<DenseIdMap<MatId, Arc<DashMap<Vec<Value>, RowBuffer>>>>,
    ) -> Self {
        Self {
            specs,
            materializations,
            scratch_key: Vec::new(),
            scratch_val: Vec::new(),
        }
    }
}

impl<'scope> ActionBuffer<'scope, MatId> for SerialScopedMaterializer {
    type AsLocal<'a>
        = Self
    where
        'scope: 'a;
    type AsGlobalSerial<'a>
        = Self
    where
        'scope: 'a;

    fn push_bindings(
        &mut self,
        mat_id: MatId,
        bindings: &DenseIdMap<Variable, Value>,
        _to_exec_state: impl FnMut() -> ExecutionState<'scope>,
    ) {
        let mat = self.materializations.get(mat_id).expect("invalid mat id");
        let spec = self.specs.get(mat_id).expect("invalid mat id");
        self.scratch_key.clear();
        self.scratch_key
            .extend(spec.msg_vars.iter().map(|var| bindings[*var]));
        self.scratch_val.clear();
        self.scratch_val
            .extend(spec.val_vars.iter().map(|var| bindings[*var]));
        if self.scratch_val.is_empty() {
            self.scratch_val.push(Value::stale());
        }
        match mat.entry(self.scratch_key.clone()) {
            Entry::Occupied(mut occupied) => {
                occupied.get_mut().add_row(&self.scratch_val);
            }
            Entry::Vacant(vacant) => {
                let mut buffer = RowBuffer::new(usize::max(spec.val_vars.len(), 1));
                buffer.add_row(&self.scratch_val);
                vacant.insert(buffer);
            }
        }
    }

    fn flush(&mut self, _exec_state: &mut ExecutionState) {}

    fn recur<'local>(
        &mut self,
        local: BorrowedLocalState<'local>,
        _to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'a> FnOnce(BorrowedLocalState<'a>, &mut Self) + Send + 'scope,
    ) {
        work(local, self);
    }

    fn recur_global_serial<'local>(
        &mut self,
        local: BorrowedLocalState<'local>,
        _to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'a> FnOnce(BorrowedLocalState<'a>, &mut Self) + Send + 'scope,
    ) {
        work(local, self);
    }

    fn supports_parallel_drain(&self) -> bool {
        false
    }
}

struct ScopedMaterializer<'inner, 'scope> {
    scope: &'inner Scope<'scope>,
    specs: Arc<DenseIdMap<MatId, MatSpec>>,
    materializations: Arc<DenseIdMap<MatId, Arc<DashMap<Vec<Value>, RowBuffer>>>>,
    scratch_key: Vec<Value>,
    scratch_val: Vec<Value>,
}
impl<'scope> ActionBuffer<'scope, MatId> for ScopedMaterializer<'_, 'scope> {
    type AsLocal<'a>
        = ScopedMaterializer<'a, 'scope>
    where
        'scope: 'a;
    type AsGlobalSerial<'a>
        = SerialScopedMaterializer
    where
        'scope: 'a;

    fn push_bindings(
        &mut self,
        mat_id: MatId,
        bindings: &DenseIdMap<Variable, Value>,
        _to_exec_state: impl FnMut() -> ExecutionState<'scope>,
    ) {
        let mat = self.materializations.get(mat_id).expect("invalid mat id");
        let spec = self.specs.get(mat_id).expect("invalid mat id");
        self.scratch_key.clear();
        for key in spec.msg_vars.iter().map(|var| bindings[*var]) {
            self.scratch_key.push(key);
        }
        self.scratch_val.clear();
        for val in spec.val_vars.iter().map(|var| bindings[*var]) {
            self.scratch_val.push(val);
        }
        if self.scratch_val.is_empty() {
            self.scratch_val.push(Value::stale());
        }
        let key = self.scratch_key.clone();
        match mat.entry(key) {
            Entry::Occupied(mut occ) => {
                occ.get_mut().add_row(&self.scratch_val);
            }
            Entry::Vacant(vac) => {
                let mut buffer = RowBuffer::new(usize::max(spec.val_vars.len(), 1));
                buffer.add_row(&self.scratch_val);
                vac.insert(buffer);
            }
        }
    }

    fn flush(&mut self, _exec_state: &mut ExecutionState) {
        // No-op for scoped materializer since we always write to the materialization in-place.
    }

    fn recur<'local>(
        &mut self,
        mut local: BorrowedLocalState<'local>,
        _to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'a> FnOnce(BorrowedLocalState<'a>, &mut ScopedMaterializer<'a, 'scope>)
        + Send
        + 'scope,
    ) {
        let scope = self.scope;
        let specs = self.specs.clone();
        let materializations = self.materializations.clone();
        let mut inner = local.clone_state();
        // Match the action path: preserve locality by default and rely on
        // private-deque donation when other workers need fallback work.
        scope.spawn_local(move |scope| {
            let mut buf: ScopedMaterializer<'_, 'scope> = ScopedMaterializer {
                scope,
                specs,
                materializations: materializations.clone(),
                scratch_key: Vec::new(),
                scratch_val: Vec::new(),
            };
            work(inner.borrow_mut(), &mut buf);
        });
    }

    fn recur_global_serial<'local>(
        &mut self,
        mut local: BorrowedLocalState<'local>,
        _to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'a> FnOnce(BorrowedLocalState<'a>, &mut SerialScopedMaterializer) + Send + 'scope,
    ) {
        let specs = self.specs.clone();
        let materializations = self.materializations.clone();
        let mut inner = local.clone_state();
        self.scope.spawn_global(move |_| {
            let mut buf = SerialScopedMaterializer::new(specs, materializations);
            work(inner.borrow_mut(), &mut buf);
        });
    }

    fn supports_global_partition(&self) -> bool {
        true
    }
}

struct MatchCounter {
    matches: IdVec<ActionId, CachePadded<AtomicUsize>>,
}

impl MatchCounter {
    fn new(n_ids: usize) -> Self {
        let mut matches = IdVec::with_capacity(n_ids);
        matches.resize_with(n_ids, || CachePadded::new(AtomicUsize::new(0)));
        Self { matches }
    }

    fn inc_matches(&self, action: ActionId, by: usize) {
        self.matches[action].fetch_add(by, std::sync::atomic::Ordering::Relaxed);
    }
    fn read_matches(&self, action: ActionId) -> usize {
        self.matches[action].load(std::sync::atomic::Ordering::Acquire)
    }
}

fn estimate_size(join_stage: &JoinStage, binding_info: &BindingInfo) -> usize {
    match join_stage {
        JoinStage::Intersect { scans, .. } => scans
            .iter()
            .map(|scan| binding_info.subsets[scan.atom].size())
            .min()
            .unwrap_or(0),
        JoinStage::FusedIntersect { cover, .. } => binding_info.subsets[cover.to_index.atom].size(),
        JoinStage::FusedIntersectMat { cover, .. } => binding_info.materializations[*cover].len(), // TODO: len() might be expensive.
    }
}

fn num_intersected_rels(join_stage: &JoinStage) -> i32 {
    match join_stage {
        JoinStage::Intersect { scans, .. } => scans.len() as i32,
        JoinStage::FusedIntersect { to_intersect, .. } => to_intersect.len() as i32 + 1,
        JoinStage::FusedIntersectMat { to_intersect, .. } => to_intersect.len() as i32,
    }
}

fn sort_plan_by_size(
    order: &mut InstrOrder,
    leaf_scans: &mut LeafScans,
    start: usize,
    instrs: &[JoinStage],
    binding_info: &mut BindingInfo,
) {
    let mut last_pos = start;
    for i in start..instrs.len() {
        if matches!(
            &instrs[i],
            // These nodes don't commute
            JoinStage::FusedIntersectMat {
                mode: MatScanMode::Lookup(_) | MatScanMode::Value(_) | MatScanMode::Full,
                ..
            }
        ) {
            sort_plan_by_size_inner(order, last_pos..i, instrs, binding_info);
            last_pos = i + 1;
        }
    }
    sort_plan_by_size_inner(order, last_pos..instrs.len(), instrs, binding_info);
    recompute_leaf_scans(order, leaf_scans, instrs, start);
}

/// Recompute `leaf_scans[i]` for every position `i` in `[start, order.len())` against the
/// current order. A position is a leaf scan iff its stage is either a `FusedIntersect` or a
/// `FusedIntersectMat { mode: Full | KeyOnly | Value }`, both with empty `to_intersect`, AND no
/// later stage either (a) for `FusedIntersect`, references the same cover atom, or (b) reads
/// any of the bound variables as a scalar via `FusedIntersectMat { mode: Value | Lookup }`.
/// `FusedIntersectMat::Lookup` itself binds nothing, so it is never marked a leaf scan.
fn recompute_leaf_scans(
    order: &InstrOrder,
    leaf_scans: &mut LeafScans,
    instrs: &[JoinStage],
    start: usize,
) {
    for i in start..order.len() {
        let stage_idx = order.get(i);
        let (cover_atom, bind_vars) = match &instrs[stage_idx] {
            JoinStage::FusedIntersect {
                cover,
                bind,
                to_intersect,
            } if to_intersect.is_empty() => {
                let vars: SmallVec<[Variable; 4]> = bind.iter().map(|(_, v)| *v).collect();
                (Some(cover.to_index.atom), vars)
            }
            JoinStage::FusedIntersectMat {
                mode,
                bind,
                to_intersect,
                ..
            } if to_intersect.is_empty()
                && matches!(
                    mode,
                    MatScanMode::Full | MatScanMode::KeyOnly | MatScanMode::Value(_)
                ) =>
            {
                let vars: SmallVec<[Variable; 4]> = bind.iter().map(|(_, v)| *v).collect();
                (None, vars)
            }
            _ => {
                leaf_scans[i] = false;
                continue;
            }
        };
        let mut blocked = false;
        for j in (i + 1)..order.len() {
            match &instrs[order.get(j)] {
                JoinStage::Intersect { scans, .. } => {
                    if let Some(ca) = cover_atom
                        && scans.iter().any(|scan| scan.atom == ca)
                    {
                        blocked = true;
                        break;
                    }
                }
                JoinStage::FusedIntersect {
                    cover,
                    to_intersect,
                    ..
                } => {
                    if let Some(ca) = cover_atom
                        && (cover.to_index.atom == ca
                            || to_intersect.iter().any(|(s, _)| s.to_index.atom == ca))
                    {
                        blocked = true;
                        break;
                    }
                }
                JoinStage::FusedIntersectMat {
                    mode, to_intersect, ..
                } => {
                    if let Some(ca) = cover_atom
                        && to_intersect.iter().any(|(s, _)| s.to_index.atom == ca)
                    {
                        blocked = true;
                        break;
                    }
                    if let MatScanMode::Value(vars) | MatScanMode::Lookup(vars) = mode
                        && vars.iter().any(|v| bind_vars.contains(v))
                    {
                        blocked = true;
                        break;
                    }
                }
            }
        }
        leaf_scans[i] = !blocked;
    }
}

fn sort_plan_by_size_inner(
    order: &mut InstrOrder,
    range: Range<usize>,
    instrs: &[JoinStage],
    binding_info: &mut BindingInfo,
) {
    // Nothing to sort if there's 0 or 1 element.
    if range.len() <= 1 {
        return;
    }
    // How many times an atom has been intersected/joined
    let mut times_refined = with_pool_set(|ps| ps.get::<DenseIdMap<AtomId, i64>>());

    // Count how many times each atom has been refined so far.
    for ins in &instrs[..range.start] {
        match ins {
            JoinStage::Intersect { scans, .. } => scans.iter().for_each(|scan| {
                *times_refined.get_or_default(scan.atom) += 1;
            }),
            JoinStage::FusedIntersect {
                cover,
                to_intersect,
                ..
            } => {
                *times_refined.get_or_default(cover.to_index.atom) +=
                    cover.to_index.vars.len() as i64;
                to_intersect.iter().for_each(|(spec, _)| {
                    *times_refined.get_or_default(spec.to_index.atom) +=
                        spec.to_index.vars.len() as i64;
                });
            }
            JoinStage::FusedIntersectMat { to_intersect, .. } => {
                to_intersect.iter().for_each(|(spec, _)| {
                    *times_refined.get_or_default(spec.to_index.atom) +=
                        spec.to_index.vars.len() as i64;
                });
            }
        }
    }

    // We prioritize variables by
    //
    //   (1) how many times an atom with this variable has been refined,
    //   (2) then by the cardinality of the variable to be enumerated (smaller → earlier)
    //   (3) then by how many relations join on this variable (more → earlier)
    //
    // Estimate size is second so that stages with very small cardinality (e.g. FunDep
    // consequents with exactly 1 value) are run before multi-relation stages that happen
    // to have a larger current estimate.
    let key_fn = |join_stage: &JoinStage,
                  binding_info: &BindingInfo,
                  times_refined: &DenseIdMap<AtomId, i64>| {
        let refine = match join_stage {
            JoinStage::Intersect { scans, .. } => scans
                .iter()
                .map(|scan| times_refined.get(scan.atom).copied().unwrap_or_default())
                .max()
                .unwrap(),
            JoinStage::FusedIntersect { cover, .. } => times_refined
                .get(cover.to_index.atom)
                .copied()
                .unwrap_or_default(),
            JoinStage::FusedIntersectMat { bind, .. } => bind.len() as _,
        };
        (
            -refine,
            estimate_size(join_stage, binding_info),
            -num_intersected_rels(join_stage),
        )
    };

    for i in range.clone() {
        let mut key_i = key_fn(&instrs[order.get(i)], binding_info, &times_refined);
        for j in (i + 1)..range.end {
            let key_j = key_fn(&instrs[order.get(j)], binding_info, &times_refined);
            if key_j < key_i {
                order.data.swap(i, j);
                key_i = key_j;
            }
        }
        // Update the counts after a new instruction is selected.
        match &instrs[order.get(i)] {
            JoinStage::Intersect { scans, .. } => scans.iter().for_each(|scan| {
                *times_refined.get_or_default(scan.atom) += 1;
            }),
            JoinStage::FusedIntersect {
                cover,
                to_intersect,
                ..
            } => {
                *times_refined.get_or_default(cover.to_index.atom) +=
                    cover.to_index.vars.len() as i64;

                to_intersect.iter().for_each(|(spec, _)| {
                    *times_refined.get_or_default(spec.to_index.atom) +=
                        spec.to_index.vars.len() as i64;
                });
            }
            JoinStage::FusedIntersectMat { to_intersect, .. } => {
                to_intersect.iter().for_each(|(spec, _)| {
                    *times_refined.get_or_default(spec.to_index.atom) +=
                        spec.to_index.vars.len() as i64;
                });
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InstrOrder {
    data: SmallVec<[u16; 8]>,
}

impl InstrOrder {
    fn new() -> Self {
        InstrOrder {
            data: SmallVec::new(),
        }
    }

    fn from_iter(range: impl Iterator<Item = usize>) -> InstrOrder {
        let mut res = InstrOrder::new();
        res.data
            .extend(range.map(|x| u16::try_from(x).expect("too many instructions")));
        res
    }

    fn get(&self, idx: usize) -> usize {
        self.data[idx] as usize
    }
    fn len(&self) -> usize {
        self.data.len()
    }
}

/// Per-position leaf-scan flags. `leaf_scans[i] == true` means the stage currently scheduled at
/// position `i` (i.e. `instrs[instr_order.get(i)]`) can take the factorized-binding fast path.
/// Recomputed by [`sort_plan_by_size`] whenever the order changes.
type LeafScans = SmallVec<[bool; 8]>;

struct BorrowedLocalState<'a> {
    instr_order: &'a mut InstrOrder,
    leaf_scans: &'a mut LeafScans,
    binding_info: &'a mut BindingInfo,
    updates: &'a mut FrameUpdates,
}

impl BorrowedLocalState<'_> {
    fn clone_state(&mut self) -> LocalState {
        LocalState {
            instr_order: self.instr_order.clone(),
            leaf_scans: self.leaf_scans.clone(),
            binding_info: self.binding_info.clone(),
            updates: std::mem::take(self.updates),
        }
    }
}

struct LocalState {
    instr_order: InstrOrder,
    leaf_scans: LeafScans,
    binding_info: BindingInfo,
    updates: FrameUpdates,
}

impl LocalState {
    fn borrow_mut<'a>(&'a mut self) -> BorrowedLocalState<'a> {
        BorrowedLocalState {
            instr_order: &mut self.instr_order,
            leaf_scans: &mut self.leaf_scans,
            binding_info: &mut self.binding_info,
            updates: &mut self.updates,
        }
    }
}

#[cfg(test)]
mod top_index_tests {
    use std::{
        sync::{
            Arc, Barrier,
            atomic::{AtomicUsize, Ordering},
        },
        thread,
    };

    use crate::{common::Value, numeric_id::NumericId, offsets::Subset};

    use super::{ChildLock, get_or_insert_child, top_index_shape_is_eligible};

    #[test]
    fn top_index_partitioning_rejects_serial_tiny_and_skewed_shapes() {
        assert!(!top_index_shape_is_eligible(1, 10_000, 8, 64));
        assert!(!top_index_shape_is_eligible(4, 255, 8, 64));
        assert!(!top_index_shape_is_eligible(4, 10_000, 3, 64));
        assert!(top_index_shape_is_eligible(4, 256, 4, 64));
        assert!(top_index_shape_is_eligible(4, 40, 4, 10));
    }

    #[test]
    fn child_cache_returns_one_arc_for_racing_same_key() {
        const THREADS: usize = 16;
        let map = Arc::new(ChildLock::default());
        let barrier = Arc::new(Barrier::new(THREADS));
        let constructed = Arc::new(AtomicUsize::new(0));
        let value = Value::from_usize(17);

        let handles = (0..THREADS)
            .map(|_| {
                let map = map.clone();
                let barrier = barrier.clone();
                let constructed = constructed.clone();
                thread::spawn(move || {
                    barrier.wait();
                    get_or_insert_child(&map, value, &[], || {
                        constructed.fetch_add(1, Ordering::Relaxed);
                        Subset::empty()
                    })
                })
            })
            .collect::<Vec<_>>();
        let nodes = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(constructed.load(Ordering::Relaxed), 1);
        assert!(nodes[1..].iter().all(|node| Arc::ptr_eq(&nodes[0], node)));
    }
}
