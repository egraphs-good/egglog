//! Core free join execution.

use std::{
    cell::{OnceCell, RefCell},
    cmp, mem,
    ops::Range,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::{
    common::{HashMap, HashSet, IndexMap},
    free_join::plan::{JoinStages, MatId, MatScanMode, MatSpec},
    numeric_id::{DenseIdMap, IdVec, NumericId},
    query::Atom,
    row_buffer::{RowBuffer, RowSink, SmallValueVec},
};
use crossbeam::utils::CachePadded;
use dashmap::mapref::entry::Entry;
use dashmap::mapref::one::RefMut;
use egglog_concurrency::{Handle, Scope, SharedArena};
use egglog_reports::{ReportLevel, RuleReport, RuleSetReport};
use smallvec::SmallVec;
use web_time::Instant;

use crate::{
    Constraint, OffsetRange, Pool, SubsetRef,
    action::{Bindings, ExecutionState, ExecutionStateSeed},
    common::{DashMap, Value},
    free_join::{
        frame_update::{FrameUpdates, UpdateInstr},
        get_index_from_tableinfo,
    },
    hash_index::{ColumnIndex, Index, IndexPosition, TupleIndex},
    offsets::{Offsets, RowId, SortedOffsetSlice, SortedOffsetVector, Subset},
    parallel_heuristics::{
        MIN_TOP_INDEX_KEYS_PER_WORKER, action_batch_size, free_join_fork_depth,
        parallelize_db_level_op,
    },
    query::RuleSet,
    row_buffer::TaggedRowBuffer,
    table_spec::{ColumnId, Offset, WrappedTableRef},
};

use super::{
    ActionId, AtomId, Database, HashColumnIndex, HashIndex, TableId, TableInfo, Variable,
    get_column_index_from_tableinfo,
    packed_trie::{ChildShape, PackedCursor, PackedTrieNode},
    plan::{JoinHeader, JoinStage, Plan, ScanSpec, SingleScanSpec},
    with_pool_set,
};

const SMALL_RESIDUAL: usize = 8;

/// Dense identity of one indexed access to an atom within a single
/// [`JoinStages`] block.  Families are dense per atom, so dynamic packed nodes
/// pay only for accesses that can actually follow that atom's current path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AccessId(u32);

impl AccessId {
    fn new(index: usize) -> Self {
        Self(u32::try_from(index).expect("an atom has more than u32::MAX indexed accesses"))
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// An owned row subset small enough to travel inline with a buffered join
/// frame. Rows are kept sorted so [`Self::subset`] can expose a borrowed
/// `SubsetRef` without allocating or involving the execution arena.
#[derive(Clone, Copy)]
pub(super) struct InlineRows {
    len: u8,
    rows: [RowId; SMALL_RESIDUAL],
}

impl InlineRows {
    fn from_sorted(rows: &[RowId]) -> Self {
        assert!(
            !rows.is_empty() && rows.len() <= SMALL_RESIDUAL,
            "inline row subsets must contain 1..={SMALL_RESIDUAL} rows"
        );
        debug_assert!(rows.windows(2).all(|pair| pair[0] <= pair[1]));
        let mut inline = Self {
            len: rows.len() as u8,
            rows: [RowId::new_const(0); SMALL_RESIDUAL],
        };
        inline.rows[..rows.len()].copy_from_slice(rows);
        inline
    }

    #[inline]
    fn rows(&self) -> &[RowId] {
        &self.rows[..usize::from(self.len)]
    }

    #[inline]
    fn len(&self) -> usize {
        usize::from(self.len)
    }

    #[inline]
    fn subset(&self) -> SubsetRef<'_> {
        // SAFETY: `from_sorted` is the only constructor and records a sorted,
        // nonempty prefix. Copies preserve that invariant.
        SubsetRef::Sparse(unsafe { SortedOffsetSlice::new_unchecked(self.rows()) })
    }
}

struct SmallRowIdSink {
    len: usize,
    rows: [RowId; SMALL_RESIDUAL],
}

impl Default for SmallRowIdSink {
    fn default() -> Self {
        Self {
            len: 0,
            rows: [RowId::new_const(0); SMALL_RESIDUAL],
        }
    }
}

impl SmallRowIdSink {
    fn push(&mut self, row_id: RowId) {
        assert!(
            self.len < SMALL_RESIDUAL,
            "small row scan exceeded its source subset size"
        );
        self.rows[self.len] = row_id;
        self.len += 1;
    }

    fn into_inline(mut self) -> Option<InlineRows> {
        if self.len == 0 {
            return None;
        }
        self.rows[..self.len].sort_unstable();
        Some(InlineRows::from_sorted(&self.rows[..self.len]))
    }
}

impl RowSink for SmallRowIdSink {
    fn add_row(&mut self, row_id: RowId, _row: &[Value]) {
        self.push(row_id);
    }
}

struct SmallExactSink<'key> {
    key: &'key [Value],
    matches: SmallRowIdSink,
}

impl RowSink for SmallExactSink<'_> {
    fn add_row(&mut self, row_id: RowId, row: &[Value]) {
        if row == self.key {
            self.matches.push(row_id);
        }
    }
}

/// Fixed-capacity sink used to project a constrained single column. The source
/// subset has already been proven to contain at most `SMALL_RESIDUAL` rows, so
/// the object-safe table scan cannot overflow this buffer.
struct SmallColumnSink {
    len: usize,
    rows: [(Value, RowId); SMALL_RESIDUAL],
}

impl Default for SmallColumnSink {
    fn default() -> Self {
        Self {
            len: 0,
            rows: [(Value::new_const(0), RowId::new_const(0)); SMALL_RESIDUAL],
        }
    }
}

impl RowSink for SmallColumnSink {
    fn add_row(&mut self, row_id: RowId, row: &[Value]) {
        let [value] = row else {
            unreachable!("a small column scan projects exactly one value")
        };
        assert!(
            self.len < SMALL_RESIDUAL,
            "small column scan exceeded its source subset size"
        );
        self.rows[self.len] = (*value, row_id);
        self.len += 1;
    }
}

/// A stack-owned index for a single column of a residual with at most eight
/// rows. Key groups point into `row_ids`; both arrays are sorted and require no
/// pool, Arc, or arena allocation.
struct SmallColumnIndex {
    n_keys: usize,
    n_rows: usize,
    keys: [Value; SMALL_RESIDUAL],
    offsets: [usize; SMALL_RESIDUAL],
    row_ids: [RowId; SMALL_RESIDUAL],
}

impl SmallColumnIndex {
    fn new(
        table: WrappedTableRef<'_>,
        subset: SubsetRef<'_>,
        constraints: &[Constraint],
        column: ColumnId,
    ) -> Self {
        debug_assert!(subset.size() <= SMALL_RESIDUAL);
        let mut sink = SmallColumnSink::default();
        let next = table.scan_project(
            subset,
            std::slice::from_ref(&column),
            Offset::new(0),
            usize::MAX,
            constraints,
            &mut sink,
        );
        debug_assert!(next.is_none());
        Self::from_projected(sink)
    }

    fn from_projected(mut sink: SmallColumnSink) -> Self {
        sink.rows[..sink.len].sort_unstable();

        let mut index = Self {
            n_keys: 0,
            n_rows: sink.len,
            keys: [Value::new_const(0); SMALL_RESIDUAL],
            offsets: [0; SMALL_RESIDUAL],
            row_ids: [RowId::new_const(0); SMALL_RESIDUAL],
        };
        for (position, &(value, row_id)) in sink.rows[..sink.len].iter().enumerate() {
            if index.n_keys == 0 || index.keys[index.n_keys - 1] != value {
                index.keys[index.n_keys] = value;
                index.offsets[index.n_keys] = position;
                index.n_keys += 1;
            }
            index.row_ids[position] = row_id;
        }
        index
    }

    #[inline]
    fn range(&self, key_index: usize) -> Range<usize> {
        let start = self.offsets[key_index];
        let end = if key_index + 1 < self.n_keys {
            self.offsets[key_index + 1]
        } else {
            self.n_rows
        };
        start..end
    }

    #[inline]
    fn find(&self, value: Value) -> Option<usize> {
        self.keys[..self.n_keys].binary_search(&value).ok()
    }

    #[inline]
    fn rows_at(&self, key_index: usize) -> InlineRows {
        InlineRows::from_sorted(&self.row_ids[self.range(key_index)])
    }

    #[inline]
    fn len(&self) -> usize {
        self.n_keys
    }
}

/// Allocation-free exact probing for an inline residual and a multi-column
/// key. Tuple residuals are only used for exact probes, so scanning at most
/// eight rows is cheaper and simpler than constructing a packed trie node.
struct SmallExactProbe<'ctx> {
    rows: Option<InlineRows>,
    columns: SmallVec<[ColumnId; 4]>,
    table: WrappedTableRef<'ctx>,
}

impl<'ctx> SmallExactProbe<'ctx> {
    fn new(
        table: WrappedTableRef<'ctx>,
        rows: InlineRows,
        columns: SmallVec<[ColumnId; 4]>,
        constraints: &[Constraint],
    ) -> Self {
        let rows = if constraints.is_empty() {
            Some(rows)
        } else {
            let mut sink = SmallRowIdSink::default();
            let next = table.scan_project(
                rows.subset(),
                std::slice::from_ref(&columns[0]),
                Offset::new(0),
                usize::MAX,
                constraints,
                &mut sink,
            );
            debug_assert!(next.is_none());
            sink.into_inline()
        };
        Self {
            rows,
            columns,
            table,
        }
    }

    fn get<'rows, 'exec>(
        &self,
        key: &[Value],
        keep_rows: bool,
    ) -> Option<ProbeMatch<'rows, 'exec>> {
        if key.len() != self.columns.len() {
            return None;
        }
        let rows = self.rows?;
        let mut sink = SmallExactSink {
            key,
            matches: SmallRowIdSink::default(),
        };
        let next = self.table.scan_project(
            rows.subset(),
            &self.columns,
            Offset::new(0),
            usize::MAX,
            &[],
            &mut sink,
        );
        debug_assert!(next.is_none());
        let matched = sink.matches.into_inline()?;
        Some(if keep_rows {
            ProbeMatch::Rows(AtomRows::Inline(matched))
        } else {
            ProbeMatch::Present
        })
    }

    fn len(&self) -> usize {
        self.rows.map_or(0, |rows| rows.len())
    }
}

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

const TOP_COVER_PARTITIONS_PER_WORKER: usize = 4;
const MIN_TOP_COVER_ROWS_PER_PARTITION: usize = 256;
const TOP_INDEX_RANGES_PER_WORKER: usize = 32;

/// Split a cover subset into bounded ordinal ranges without materializing its
/// row IDs. Keeping several ranges per worker gives work stealing room to
/// absorb join-selectivity skew while the minimum size keeps handoff coarse.
fn top_cover_partitions(
    cover_rows: usize,
    workers: usize,
    min_rows_per_partition: usize,
) -> Option<Vec<TopLevelPartition>> {
    if workers <= 1 || cover_rows < min_rows_per_partition.saturating_mul(workers) {
        return None;
    }

    let target_partitions = workers.saturating_mul(TOP_COVER_PARTITIONS_PER_WORKER);
    let rows_per_partition = cover_rows
        .div_ceil(target_partitions)
        .max(min_rows_per_partition);
    let mut partitions = Vec::with_capacity(cover_rows.div_ceil(rows_per_partition));
    for start in (0..cover_rows).step_by(rows_per_partition) {
        partitions.push(TopLevelPartition::CoverRange {
            start,
            scan_size: cmp::min(rows_per_partition, cover_rows - start),
        });
    }
    Some(partitions)
}

/// Split an unsharded scalar index into disjoint key-ordinal ranges.
///
/// Unlike scan-and-project partitioning, this borrows the scalar index that
/// normal join execution has already built for the filtered root. It therefore
/// adds no row-ID copy or second sort for seminaive and constrained subsets.
/// The aggressive overpartitioning leaves enough independent work to absorb
/// skew in the number of join results produced by each leading key.
fn top_index_range_partitions(
    leader_keys: usize,
    workers: usize,
    min_keys_per_worker: usize,
) -> Option<Vec<TopLevelPartition>> {
    if workers <= 1 || leader_keys < min_keys_per_worker.saturating_mul(workers) {
        return None;
    }

    let target_partitions = workers.saturating_mul(TOP_INDEX_RANGES_PER_WORKER);
    let keys_per_partition = leader_keys
        .div_ceil(target_partitions)
        .max(min_keys_per_worker);
    let mut partitions = Vec::with_capacity(leader_keys.div_ceil(keys_per_partition));
    for start in (0..leader_keys).step_by(keys_per_partition) {
        partitions.push(TopLevelPartition::IndexRange {
            start,
            scan_size: cmp::min(keys_per_partition, leader_keys - start),
        });
    }
    Some(partitions)
}

/// One coarse, disjoint partition used to drive the top join level.
///
/// Cached scalar intersections naturally partition by physical index shard.
/// Filtered scalar indexes use borrowed key-ordinal ranges, and fused
/// intersections use ordinal ranges of their cover subset. The range forms
/// work for dense, sparse, whole-table, and seminaive subsets without copying
/// row IDs. `scan_size` is the number of leader keys or cover rows in the
/// partition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TopLevelPartition {
    IndexShard { shard: usize, scan_size: usize },
    IndexRange { start: usize, scan_size: usize },
    CoverRange { start: usize, scan_size: usize },
}

impl TopLevelPartition {
    fn scan_size(self) -> usize {
        match self {
            Self::IndexShard { scan_size, .. }
            | Self::IndexRange { scan_size, .. }
            | Self::CoverRange { scan_size, .. } => scan_size,
        }
    }
}

fn cover_scan_bounds(partition: Option<TopLevelPartition>, cover_rows: usize) -> (Offset, usize) {
    match partition {
        Some(TopLevelPartition::CoverRange { start, scan_size }) => {
            debug_assert!(start <= cover_rows);
            debug_assert!(scan_size <= cover_rows - start);
            (Offset::from_usize(start), scan_size)
        }
        None => (Offset::new(0), cover_rows),
        Some(TopLevelPartition::IndexShard { .. } | TopLevelPartition::IndexRange { .. }) => {
            unreachable!("an index partition can only drive a scalar intersection")
        }
    }
}

/// A key ordinal in a root continuation grid.
///
/// Persistent catalog indexes produce an [`IndexPosition`], while round-local
/// projected roots are unsharded and produce an ordinal directly. Converting
/// both to this execution-local type keeps continuation identity separate from
/// the exact persistent index identity documented by [`IndexPosition`].
#[derive(Clone, Copy)]
struct ContinuationPosition {
    shard: u32,
    slot: u32,
}

impl ContinuationPosition {
    fn unsharded(slot: usize) -> Self {
        Self {
            shard: 0,
            slot: u32::try_from(slot)
                .expect("a root continuation grid cannot contain more than u32::MAX keys"),
        }
    }
}

impl From<IndexPosition> for ContinuationPosition {
    fn from(position: IndexPosition) -> Self {
        Self {
            shard: u32::try_from(position.shard())
                .expect("an index cannot contain more than u32::MAX shards"),
            slot: u32::try_from(position.slot())
                .expect("an index shard cannot contain more than u32::MAX keys"),
        }
    }
}

/// Lazily-created continuation slots for rows read from a persistent or
/// round-local root index. Slots are deliberately per logical scan: a
/// different frozen plan may continue the same root key with a different
/// column or constraint set.
type RootContinuationSlots = Box<[Box<[OnceLock<usize>]>]>;

enum RootContinuationStorage {
    /// The atom has one statically possible indexed successor.  This is the
    /// existing compact path: one continuation slot per physical root key.
    /// Defer allocating that grid until a probe actually asks for a child;
    /// many shallow plans prepare a possible successor but never descend.
    Direct {
        shard_lens: Box<[usize]>,
        slots: OnceLock<RootContinuationSlots>,
    },
    /// More than one access may follow.  Allocate the dense per-position slots
    /// for a family only when DVO actually selects it.
    Dynamic {
        shard_lens: Box<[usize]>,
        families: Box<[OnceLock<RootContinuationSlots>]>,
    },
}

struct RootContinuationCache {
    storage: OnceLock<RootContinuationStorage>,
    #[cfg(debug_assertions)]
    direct_access: OnceLock<AccessId>,
}

impl Default for RootContinuationCache {
    fn default() -> Self {
        Self {
            storage: OnceLock::new(),
            #[cfg(debug_assertions)]
            direct_access: OnceLock::new(),
        }
    }
}

impl RootContinuationCache {
    fn allocate_slots(shard_lens: &[usize]) -> RootContinuationSlots {
        shard_lens
            .iter()
            .map(|&len| std::iter::repeat_with(OnceLock::new).take(len).collect())
            .collect()
    }

    fn prepare(
        &self,
        child_shape: ChildShape,
        shard_count: usize,
        shard_len: impl Fn(usize) -> usize,
    ) {
        assert_ne!(
            child_shape,
            ChildShape::Leaf,
            "a catalog leaf does not need continuation storage"
        );
        let storage = self.storage.get_or_init(|| {
            let shard_lens = (0..shard_count).map(shard_len).collect::<Box<[_]>>();
            match child_shape {
                ChildShape::Leaf => unreachable!(),
                ChildShape::Direct => RootContinuationStorage::Direct {
                    shard_lens,
                    slots: OnceLock::new(),
                },
                ChildShape::Dynamic { families } => RootContinuationStorage::Dynamic {
                    shard_lens,
                    families: std::iter::repeat_with(OnceLock::new)
                        .take(families)
                        .collect(),
                },
            }
        });
        debug_assert!(
            match (child_shape, storage) {
                (ChildShape::Direct, RootContinuationStorage::Direct { .. }) => true,
                (
                    ChildShape::Dynamic { families: expected },
                    RootContinuationStorage::Dynamic { families, .. },
                ) => expected == families.len(),
                _ => false,
            },
            "root continuation shape changed after initialization"
        );
    }

    fn slots(&self, access: AccessId) -> &RootContinuationSlots {
        let storage = self
            .storage
            .get()
            .expect("root continuations must be prepared before probing");
        #[cfg(debug_assertions)]
        if matches!(storage, RootContinuationStorage::Direct { .. }) {
            let expected = self.direct_access.get_or_init(|| access);
            debug_assert_eq!(
                *expected, access,
                "a direct root continuation was used by multiple indexed accesses"
            );
        }
        match storage {
            RootContinuationStorage::Direct { shard_lens, slots } => {
                slots.get_or_init(|| Self::allocate_slots(shard_lens))
            }
            RootContinuationStorage::Dynamic {
                shard_lens,
                families,
            } => families[access.index()].get_or_init(|| Self::allocate_slots(shard_lens)),
        }
    }

    fn slot(&self, position: ContinuationPosition, access: AccessId) -> &OnceLock<usize> {
        let slots = self.slots(access);
        &slots[position.shard as usize][position.slot as usize]
    }
}

/// A table-index slot retained for one logical query execution.
///
/// The slot lazily acquires its Arc through the existing fully-refreshing
/// catalog helper on first cached use. Keeping that Arc in an execution-scoped
/// sidecar removes catalog lookups and refcount traffic from recursive join
/// execution without constructing indexes for plan accesses that choose a
/// residual-local strategy at runtime. Initialized slots are dropped before
/// the database resets its indexes during `merge_all`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreparedIndexKind {
    Tuple,
    Column,
    /// The table specification forbids a global cache for at least one key
    /// column, so execution must use its existing dynamic-index path.
    Uncacheable,
}

#[derive(Clone, Copy, Debug)]
struct PreparedIndexStateId(u32);

impl PreparedIndexStateId {
    fn new(index: usize) -> Self {
        Self(u32::try_from(index).expect("a join block has more than u32::MAX index states"))
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

enum PreparedIndexCache {
    Tuple(OnceLock<HashIndex>),
    Column(OnceLock<HashColumnIndex>),
    Uncacheable,
}

/// Execution-local mutable state for one prepared index access. Keeping these
/// large cache objects out of the stage `SmallVec`s leaves their inline entries
/// as compact, copyable descriptors.
struct PreparedIndexState {
    cache: PreparedIndexCache,
    root_continuations: RootContinuationCache,
    /// Handle to a shared, final-form root index for this logical access.
    /// Keeping the Arc in the prepared sidecar lets probers borrow its arrays
    /// for the whole query without cloning an Arc into every output frame.
    projected_root: OnceLock<RootProjectionSlot>,
    /// Erased arena address of the packed root for this logical scan.
    /// This remains the fallback for roots that are not shared across plans.
    packed_root: OnceLock<usize>,
}

impl PreparedIndexState {
    fn new(kind: PreparedIndexKind) -> Self {
        let cache = match kind {
            PreparedIndexKind::Tuple => PreparedIndexCache::Tuple(OnceLock::new()),
            PreparedIndexKind::Column => PreparedIndexCache::Column(OnceLock::new()),
            PreparedIndexKind::Uncacheable => PreparedIndexCache::Uncacheable,
        };
        Self {
            cache,
            root_continuations: RootContinuationCache::default(),
            projected_root: OnceLock::new(),
            packed_root: OnceLock::new(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PreparedIndexSlot {
    kind: PreparedIndexKind,
    access: AccessId,
    state: PreparedIndexStateId,
}

impl PreparedIndexSlot {
    fn new(kind: PreparedIndexKind, access: AccessId, state: PreparedIndexStateId) -> Self {
        Self {
            kind,
            access,
            state,
        }
    }
}

#[derive(Clone, Copy)]
struct PreparedIndexRef<'a> {
    kind: PreparedIndexKind,
    access: AccessId,
    state: &'a PreparedIndexState,
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

#[derive(Clone, Copy, Default)]
struct PreparedAtomUse {
    touched_stages: u64,
    one_index_access_stages: u64,
    multiple_index_access_stages: u64,
}

/// Order-independent tail metadata for plans small enough to represent their
/// remaining stages in one word. DVO only permutes stages within the fixed
/// barrier phases, so successor shape depends on the remaining set, not its
/// current permutation.
struct PreparedTailMasks {
    atom_uses: DenseIdMap<AtomId, PreparedAtomUse>,
    phase_masks: SmallVec<[u64; 4]>,
    all_stages: u64,
}

impl PreparedTailMasks {
    fn new(
        stages: &[JoinStage],
        prepared_stages: &[SmallVec<[PreparedIndexSlot; 4]>],
        atom_capacity: usize,
    ) -> Option<Self> {
        if stages.len() > u64::BITS as usize {
            return None;
        }
        let mut atom_uses: DenseIdMap<AtomId, PreparedAtomUse> =
            DenseIdMap::with_capacity(atom_capacity);
        for (stage_index, (stage, prepared_stage)) in stages.iter().zip(prepared_stages).enumerate()
        {
            let stage_bit = 1u64 << stage_index;
            for_each_stage_atom(stage, |atom| {
                atom_uses.get_or_default(atom).touched_stages |= stage_bit;
            });

            let mut indexed_counts = SmallVec::<[(AtomId, u8); 4]>::new();
            for_each_stage_indexed_access(stage, prepared_stage, |atom, _| {
                if let Some((_, count)) = indexed_counts
                    .iter_mut()
                    .find(|(candidate, _)| *candidate == atom)
                {
                    *count = count.saturating_add(1);
                } else {
                    indexed_counts.push((atom, 1));
                }
            });
            for (atom, count) in indexed_counts {
                let use_ = atom_uses.get_or_default(atom);
                if count == 1 {
                    use_.one_index_access_stages |= stage_bit;
                } else {
                    use_.multiple_index_access_stages |= stage_bit;
                }
            }
        }

        let mut phase_masks = SmallVec::<[u64; 4]>::new();
        let mut reorderable_phase = 0u64;
        for (stage_index, stage) in stages.iter().enumerate() {
            let stage_bit = 1u64 << stage_index;
            if is_reorder_barrier(stage) {
                if reorderable_phase != 0 {
                    phase_masks.push(reorderable_phase);
                    reorderable_phase = 0;
                }
                phase_masks.push(stage_bit);
            } else {
                reorderable_phase |= stage_bit;
            }
        }
        if reorderable_phase != 0 {
            phase_masks.push(reorderable_phase);
        }

        Some(Self {
            atom_uses,
            phase_masks,
            all_stages: if stages.len() == u64::BITS as usize {
                u64::MAX
            } else {
                (1u64 << stages.len()) - 1
            },
        })
    }

    fn atom_tail_use(&self, atom: AtomId, remaining_stages: u64, families: usize) -> AtomTailUse {
        let use_ = self.atom_uses.get(atom).copied().unwrap_or_default();
        if remaining_stages & use_.touched_stages == 0 {
            return AtomTailUse {
                keep_rows: false,
                child_shape: ChildShape::Leaf,
            };
        }

        for &phase in &self.phase_masks {
            let live = remaining_stages & phase;
            if live & use_.touched_stages == 0 {
                continue;
            }
            let single_accesses = live & use_.one_index_access_stages;
            let multiple_accesses =
                live & use_.multiple_index_access_stages != 0 || single_accesses.count_ones() > 1;
            let child_shape = if multiple_accesses {
                ChildShape::Dynamic { families }
            } else if single_accesses != 0 {
                ChildShape::Direct
            } else {
                ChildShape::Leaf
            };
            return AtomTailUse {
                keep_rows: true,
                child_shape,
            };
        }

        unreachable!("a touched atom must belong to one prepared reorder phase")
    }
}

/// Index handles for one immutable [`JoinStages`] value, positionally aligned
/// with `JoinStages::instrs` and with each stage's scans.
enum PreparedJoinIndexes {
    /// A block made entirely of cover scans cannot build a packed node or use
    /// an index. Avoid constructing any index sidecar for these blocks; unary
    /// rules hit this path especially often.
    NoIndexes,
    Indexed {
        stages: Box<[SmallVec<[PreparedIndexSlot; 4]>]>,
        states: Box<[PreparedIndexState]>,
        access_counts: DenseIdMap<AtomId, usize>,
        tail_masks: Option<PreparedTailMasks>,
    },
}

impl PreparedJoinIndexes {
    fn new(db: &Database, atoms: &Arc<DenseIdMap<AtomId, Atom>>, stages: &JoinStages) -> Self {
        let index_count = stages
            .instrs
            .iter()
            .map(|stage| match stage {
                JoinStage::Intersect { scans, .. } => scans.len(),
                JoinStage::FusedIntersect { to_intersect, .. }
                | JoinStage::FusedIntersectMat { to_intersect, .. } => to_intersect.len(),
            })
            .sum::<usize>();
        if index_count == 0 {
            return Self::NoIndexes;
        }

        fn make_slot(
            db: &Database,
            atoms: &DenseIdMap<AtomId, Atom>,
            access_counts: &mut DenseIdMap<AtomId, usize>,
            states: &mut Vec<PreparedIndexState>,
            atom: AtomId,
            cols: &[ColumnId],
        ) -> PreparedIndexSlot {
            let next = access_counts.get_or_default(atom);
            let access = AccessId::new(*next);
            *next += 1;
            let info = &db.tables[atoms[atom].table];
            let kind = if !columns_are_cacheable(info, cols) {
                PreparedIndexKind::Uncacheable
            } else if cols.len() == 1 {
                PreparedIndexKind::Column
            } else {
                PreparedIndexKind::Tuple
            };
            let state = PreparedIndexStateId::new(states.len());
            states.push(PreparedIndexState::new(kind));
            PreparedIndexSlot::new(kind, access, state)
        }

        let mut access_counts = DenseIdMap::with_capacity(atoms.n_ids());
        let mut prepared_stages = Vec::with_capacity(stages.instrs.len());
        let mut states = Vec::with_capacity(index_count);
        for stage in stages.instrs.iter() {
            let mut handles = SmallVec::new();
            match stage {
                JoinStage::Intersect { scans, .. } => {
                    handles.extend(scans.iter().map(|scan| {
                        make_slot(
                            db,
                            atoms,
                            &mut access_counts,
                            &mut states,
                            scan.atom,
                            std::slice::from_ref(&scan.column),
                        )
                    }));
                }
                JoinStage::FusedIntersect { to_intersect, .. }
                | JoinStage::FusedIntersectMat { to_intersect, .. } => {
                    handles.extend(to_intersect.iter().map(|(scan, _)| {
                        make_slot(
                            db,
                            atoms,
                            &mut access_counts,
                            &mut states,
                            scan.to_index.atom,
                            scan.to_index.vars.as_slice(),
                        )
                    }));
                }
            }
            prepared_stages.push(handles);
        }
        let tail_masks = PreparedTailMasks::new(&stages.instrs, &prepared_stages, atoms.n_ids());
        Self::Indexed {
            stages: prepared_stages.into_boxed_slice(),
            states: states.into_boxed_slice(),
            access_counts,
            tail_masks,
        }
    }

    fn stage(&self, index: usize) -> &[PreparedIndexSlot] {
        match self {
            Self::NoIndexes => &[],
            Self::Indexed { stages, .. } => &stages[index],
        }
    }

    fn access_count(&self, atom: AtomId) -> usize {
        match self {
            Self::NoIndexes => 0,
            Self::Indexed { access_counts, .. } => {
                access_counts.get(atom).copied().unwrap_or_default()
            }
        }
    }

    fn resolve<'a>(&'a self, slot: &PreparedIndexSlot) -> PreparedIndexRef<'a> {
        let Self::Indexed { states, .. } = self else {
            unreachable!("an index slot cannot belong to a block without indexes")
        };
        PreparedIndexRef {
            kind: slot.kind,
            access: slot.access,
            state: &states[slot.state.index()],
        }
    }

    fn all_stage_mask(&self) -> Option<u64> {
        self.tail_masks().map(|masks| masks.all_stages)
    }

    fn tail_masks(&self) -> Option<&PreparedTailMasks> {
        match self {
            Self::NoIndexes => None,
            Self::Indexed { tail_masks, .. } => tail_masks.as_ref(),
        }
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

/// Seek `key` in a sorted scalar index without moving backward.
///
/// Exponential search followed by a bounded binary search avoids walking a
/// large target when the sorted query is sparse.
fn seek_sorted_key(
    key: Value,
    target_len: usize,
    target_cursor: &mut usize,
    target_at: impl Fn(usize) -> Value,
) -> bool {
    if *target_cursor >= target_len {
        return false;
    }

    let current = target_at(*target_cursor);
    match current.cmp(&key) {
        cmp::Ordering::Equal => true,
        cmp::Ordering::Greater => false,
        cmp::Ordering::Less => {
            let base = *target_cursor;
            let mut step = 1usize;
            while let Some(position) = base.checked_add(step).filter(|&pos| pos < target_len) {
                if target_at(position) >= key {
                    break;
                }
                let Some(next) = step.checked_mul(2) else {
                    step = target_len;
                    break;
                };
                step = next;
            }

            let previous = step / 2;
            let mut lo = base
                .saturating_add(previous)
                .saturating_add(1)
                .min(target_len);
            let mut hi = base.saturating_add(step).saturating_add(1).min(target_len);
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if target_at(mid) < key {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            *target_cursor = lo;
            if lo >= target_len {
                return false;
            }
            target_at(lo) == key
        }
    }
}

/// The rows currently associated with an atom during one plan execution.
/// Roots retain ownership of their header-filtered subset. An indexed cursor
/// borrows a first-level group from either a prepared persistent index or a
/// shared round-local root index and carries a plan-local continuation slot.
/// Every lower cursor is just a packed node plus a key ordinal. Dense
/// singletons come from cover scans and are packed lazily if the atom is probed
/// again.
#[derive(Clone, Copy)]
pub(super) struct CatalogContinuation<'rows> {
    cache: &'rows RootContinuationCache,
    position: ContinuationPosition,
}

#[derive(Clone)]
pub(super) enum AtomRows<'rows, 'exec> {
    Root(Arc<TrieNode>),
    Catalog {
        subset: SubsetRef<'rows>,
        continuation: Option<CatalogContinuation<'rows>>,
    },
    Packed(PackedCursor<'rows, 'exec>),
    /// Owned small residual passed directly between buffered or parallel
    /// frames. Unlike `Catalog` and `Packed`, this variant borrows no index or
    /// arena storage.
    Inline(InlineRows),
    Dense(OffsetRange),
}

impl std::fmt::Debug for AtomRows<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomRows")
            .field("kind", &self.kind_name())
            .field("size", &self.size())
            .finish()
    }
}

impl<'rows, 'exec> AtomRows<'rows, 'exec>
where
    'exec: 'rows,
{
    fn kind_name(&self) -> &'static str {
        match self {
            Self::Root(_) => "root",
            Self::Catalog { .. } => "catalog",
            Self::Packed(_) => "packed",
            Self::Inline(_) => "inline",
            Self::Dense(_) => "dense",
        }
    }

    fn subset(&self) -> SubsetRef<'_> {
        match self {
            Self::Root(root) => root.subset.as_ref(),
            Self::Catalog { subset, .. } => *subset,
            Self::Packed(cursor) => cursor.subset(),
            Self::Inline(rows) => rows.subset(),
            Self::Dense(range) => SubsetRef::Dense(*range),
        }
    }

    fn size(&self) -> usize {
        match self {
            Self::Packed(cursor) => cursor.size(),
            _ => self.subset().size(),
        }
    }

    fn is_empty(&self) -> bool {
        self.size() == 0
    }

    #[cfg(test)]
    fn root_arc(&self) -> &Arc<TrieNode> {
        let Self::Root(root) = self else {
            panic!("expected root rows")
        };
        root
    }
}

impl<'rows, 'exec> From<Arc<TrieNode>> for AtomRows<'rows, 'exec> {
    fn from(root: Arc<TrieNode>) -> Self {
        Self::Root(root)
    }
}

enum ProbeIndex<'ctx, 'rows, 'exec> {
    CachedTuple {
        intersect_outer: Option<OffsetRange>,
        table: &'rows Index<TupleIndex>,
        continuations: &'rows RootContinuationCache,
        child_shape: ChildShape,
    },
    CachedColumn {
        intersect_outer: Option<OffsetRange>,
        table: &'rows Index<ColumnIndex>,
        continuations: &'rows RootContinuationCache,
        child_shape: ChildShape,
    },
    ProjectedRoot(RootProjectionProbe<'ctx, 'rows, 'exec>),
    SmallColumn(SmallColumnIndex),
    SmallExact(SmallExactProbe<'ctx>),
    Packed(PackedProbe<'ctx, 'rows, 'exec>),
}

#[derive(Clone, Copy)]
enum SortedScalarProbe<'a, 'exec> {
    Projected(&'a RootProjection),
    Small(&'a SmallColumnIndex),
    Packed(&'a PackedTrieNode<'exec>),
}

impl SortedScalarProbe<'_, '_> {
    fn len(self) -> usize {
        match self {
            Self::Projected(index) => index.len(),
            Self::Small(index) => index.n_keys,
            Self::Packed(index) => index.values().len(),
        }
    }

    fn value_at(self, key_index: usize) -> Value {
        match self {
            Self::Projected(index) => index.value_at(key_index),
            Self::Small(index) => index.keys[key_index],
            Self::Packed(index) => index.values()[key_index],
        }
    }
}

/// A successful probe either carries rows needed by a later stage or only
/// records existence when this atom is dead in the remaining join tail. The
/// latter case avoids copying an inline subset into every buffered frame.
enum ProbeMatch<'rows, 'exec> {
    Present,
    Rows(AtomRows<'rows, 'exec>),
}

/// Worker-local arena handle initialized only by a path that actually builds
/// a packed node. Cover-only queries and scalar projected-root probes never
/// touch the arena allocator.
struct LazyArenaHandle<'exec> {
    arena: &'exec SharedArena,
    handle: OnceCell<Handle<'exec>>,
}

impl<'exec> LazyArenaHandle<'exec> {
    fn new(arena: &'exec SharedArena) -> Self {
        Self {
            arena,
            handle: OnceCell::new(),
        }
    }

    fn get(&self) -> &Handle<'exec> {
        self.handle.get_or_init(|| self.arena.new_handle())
    }
}

impl<'rows, 'exec> ProbeMatch<'rows, 'exec> {
    #[inline]
    fn refine(self, atom: AtomId, updates: &mut FrameUpdates<'rows, 'exec>) {
        if let Self::Rows(rows) = self {
            updates.refine_atom(atom, rows);
        }
    }
}

struct PackedProbe<'ctx, 'rows, 'exec> {
    first: &'rows PackedTrieNode<'exec>,
    columns: SmallVec<[ColumnId; 4]>,
    table: WrappedTableRef<'ctx>,
    handle: &'ctx LazyArenaHandle<'exec>,
    scratch: &'ctx RefCell<Vec<(Value, RowId)>>,
    terminal_child_shape: ChildShape,
}

/// Direct probe over a final-form root projection shared across plans in the
/// current ruleset execution. Immutable keys and rows stay in the shared root;
/// only continuation slots and packed descendants belong to this query.
struct RootProjectionProbe<'ctx, 'rows, 'exec> {
    first: &'rows RootProjection,
    columns: SmallVec<[ColumnId; 4]>,
    table: WrappedTableRef<'ctx>,
    continuations: &'rows RootContinuationCache,
    access: AccessId,
    handle: &'ctx LazyArenaHandle<'exec>,
    scratch: &'ctx RefCell<Vec<(Value, RowId)>>,
    terminal_child_shape: ChildShape,
}

impl<'ctx, 'rows, 'exec> RootProjectionProbe<'ctx, 'rows, 'exec>
where
    'exec: 'rows,
{
    fn scalar_rows(&self, key_index: usize) -> AtomRows<'rows, 'exec> {
        debug_assert_eq!(self.columns.len(), 1);
        AtomRows::Catalog {
            subset: self.first.subset_at(key_index),
            continuation: (self.terminal_child_shape != ChildShape::Leaf).then_some(
                CatalogContinuation {
                    cache: self.continuations,
                    position: ContinuationPosition::unsharded(key_index),
                },
            ),
        }
    }

    fn first_child(&self, key_index: usize) -> &'exec PackedTrieNode<'exec> {
        debug_assert!(self.columns.len() > 1);
        let child_shape = if self.columns.len() > 2 {
            ChildShape::Direct
        } else {
            self.terminal_child_shape
        };
        let slot = self
            .continuations
            .slot(ContinuationPosition::unsharded(key_index), self.access);
        let address = *slot.get_or_init(|| {
            PackedTrieNode::build_from_subset(
                self.handle.get(),
                self.table,
                self.first.subset_at(key_index),
                self.columns[1],
                child_shape,
                &mut self.scratch.borrow_mut(),
            ) as *const PackedTrieNode<'exec> as usize
        });
        // SAFETY: this continuation cache belongs to the prepared query and can
        // only publish nodes allocated by the same query's SharedArena.
        let child = unsafe { &*(address as *const PackedTrieNode<'exec>) };
        assert_eq!(child.child_shape(), child_shape);
        child
    }

    fn get(&self, key: &[Value]) -> Option<AtomRows<'rows, 'exec>> {
        if key.len() != self.columns.len() {
            return None;
        }
        let first_index = self.first.find(key[0])?;
        if self.columns.len() == 1 {
            return Some(self.scalar_rows(first_index));
        }

        let mut node = self.first_child(first_index);
        let mut terminal = None;
        for (depth, &value) in key.iter().enumerate().skip(1) {
            let cursor = PackedCursor::new(node, node.find(value)?);
            terminal = Some(cursor);
            if depth + 1 < self.columns.len() {
                let child_shape = if depth + 2 < self.columns.len() {
                    ChildShape::Direct
                } else {
                    self.terminal_child_shape
                };
                node = cursor.child_index(
                    self.handle.get(),
                    self.table,
                    self.columns[depth + 1],
                    0,
                    child_shape,
                    &mut self.scratch.borrow_mut(),
                );
            }
        }
        terminal.map(AtomRows::Packed)
    }

    fn for_each_packed(
        &self,
        node: &'rows PackedTrieNode<'exec>,
        depth: usize,
        key: &mut SmallVec<[Value; 4]>,
        f: &mut impl FnMut(&[Value], AtomRows<'rows, 'exec>),
    ) {
        for (key_index, &value) in node.values().iter().enumerate() {
            key.push(value);
            let cursor = PackedCursor::new(node, key_index);
            if depth + 1 == self.columns.len() {
                f(key, AtomRows::Packed(cursor));
            } else {
                let child_shape = if depth + 2 < self.columns.len() {
                    ChildShape::Direct
                } else {
                    self.terminal_child_shape
                };
                let child = cursor.child_index(
                    self.handle.get(),
                    self.table,
                    self.columns[depth + 1],
                    0,
                    child_shape,
                    &mut self.scratch.borrow_mut(),
                );
                self.for_each_packed(child, depth + 1, key, f);
            }
            key.pop();
        }
    }

    fn for_each(&self, f: &mut impl FnMut(&[Value], AtomRows<'rows, 'exec>)) {
        let mut key = SmallVec::new();
        for key_index in 0..self.first.len() {
            key.push(self.first.value_at(key_index));
            if self.columns.len() == 1 {
                f(&key, self.scalar_rows(key_index));
            } else {
                let child = self.first_child(key_index);
                self.for_each_packed(child, 1, &mut key, f);
            }
            key.pop();
        }
    }
}

impl<'ctx, 'rows, 'exec> PackedProbe<'ctx, 'rows, 'exec>
where
    'exec: 'rows,
{
    fn get(&self, key: &[Value]) -> Option<AtomRows<'rows, 'exec>> {
        if key.len() != self.columns.len() {
            return None;
        }
        let mut node = self.first;
        let mut terminal = None;
        for (depth, (&_column, &value)) in self.columns.iter().zip(key).enumerate() {
            let cursor = PackedCursor::new(node, node.find(value)?);
            terminal = Some(cursor);
            if depth + 1 < self.columns.len() {
                let child_shape = if depth + 2 < self.columns.len() {
                    ChildShape::Direct
                } else {
                    self.terminal_child_shape
                };
                node = cursor.child_index(
                    self.handle.get(),
                    self.table,
                    self.columns[depth + 1],
                    0,
                    child_shape,
                    &mut self.scratch.borrow_mut(),
                );
            }
        }
        terminal.map(AtomRows::Packed)
    }

    fn for_each_recur(
        &self,
        node: &'rows PackedTrieNode<'exec>,
        depth: usize,
        key: &mut SmallVec<[Value; 4]>,
        f: &mut impl FnMut(&[Value], AtomRows<'rows, 'exec>),
    ) {
        for (key_index, &value) in node.values().iter().enumerate() {
            key.push(value);
            let cursor = PackedCursor::new(node, key_index);
            if depth + 1 == self.columns.len() {
                f(key, AtomRows::Packed(cursor));
            } else {
                let child_shape = if depth + 2 < self.columns.len() {
                    ChildShape::Direct
                } else {
                    self.terminal_child_shape
                };
                let child = cursor.child_index(
                    self.handle.get(),
                    self.table,
                    self.columns[depth + 1],
                    0,
                    child_shape,
                    &mut self.scratch.borrow_mut(),
                );
                self.for_each_recur(child, depth + 1, key, f);
            }
            key.pop();
        }
    }

    fn for_each(&self, f: &mut impl FnMut(&[Value], AtomRows<'rows, 'exec>)) {
        let mut key = SmallVec::new();
        self.for_each_recur(self.first, 0, &mut key, f);
    }
}

struct Prober<'ctx, 'rows, 'exec> {
    source: AtomRows<'rows, 'exec>,
    ix: ProbeIndex<'ctx, 'rows, 'exec>,
    keep_rows: bool,
}

struct ProbeRequest<'scan, 'rows> {
    atom: AtomId,
    columns: SmallVec<[ColumnId; 4]>,
    constraints: &'scan [Constraint],
    keep_rows: bool,
    terminal_child_shape: ChildShape,
    prepared: PreparedIndexRef<'rows>,
}

impl<'scan, 'rows> ProbeRequest<'scan, 'rows> {
    fn column(
        scan: &'scan SingleScanSpec,
        keep_rows: bool,
        terminal_child_shape: ChildShape,
        prepared: PreparedIndexRef<'rows>,
    ) -> Self {
        Self {
            atom: scan.atom,
            columns: SmallVec::from_slice(&[scan.column]),
            constraints: &scan.cs,
            keep_rows,
            terminal_child_shape,
            prepared,
        }
    }

    fn tuple(
        scan: &'scan ScanSpec,
        keep_rows: bool,
        terminal_child_shape: ChildShape,
        prepared: PreparedIndexRef<'rows>,
    ) -> Self {
        Self {
            atom: scan.to_index.atom,
            columns: scan.to_index.vars.iter().copied().collect(),
            constraints: &scan.constraints,
            keep_rows,
            terminal_child_shape,
            prepared,
        }
    }
}

impl<'ctx, 'rows, 'exec> Prober<'ctx, 'rows, 'exec>
where
    'exec: 'rows,
{
    fn keep_or_discard(rows: AtomRows<'rows, 'exec>, keep_rows: bool) -> ProbeMatch<'rows, 'exec> {
        if keep_rows {
            ProbeMatch::Rows(rows)
        } else {
            ProbeMatch::Present
        }
    }

    fn catalog_match(
        position: IndexPosition,
        subset: SubsetRef<'rows>,
        continuations: &'rows RootContinuationCache,
        keep_rows: bool,
        child_shape: ChildShape,
    ) -> ProbeMatch<'rows, 'exec> {
        if keep_rows {
            ProbeMatch::Rows(AtomRows::Catalog {
                subset,
                continuation: (child_shape != ChildShape::Leaf).then_some(CatalogContinuation {
                    cache: continuations,
                    position: position.into(),
                }),
            })
        } else {
            ProbeMatch::Present
        }
    }

    /// Return a terminal catalog result without retaining an index position.
    ///
    /// A leaf can still feed a cover/barrier, so `keep_rows` may require the
    /// subset even though no packed continuation can be built from it.
    fn terminal_catalog_match(
        subset: SubsetRef<'rows>,
        keep_rows: bool,
    ) -> ProbeMatch<'rows, 'exec> {
        if keep_rows {
            ProbeMatch::Rows(AtomRows::Catalog {
                subset,
                continuation: None,
            })
        } else {
            ProbeMatch::Present
        }
    }

    fn get_subset(&self, key: &[Value]) -> Option<ProbeMatch<'rows, 'exec>> {
        match &self.ix {
            ProbeIndex::CachedTuple {
                intersect_outer,
                table,
                continuations,
                child_shape,
            } => {
                let table: &'rows Index<TupleIndex> = table;
                if *child_shape == ChildShape::Leaf || !self.keep_rows {
                    debug_assert!(self.keep_rows || *child_shape == ChildShape::Leaf);
                    let subset = table.get_subset(key)?;
                    let subset = if let Some(range) = intersect_outer {
                        intersect_with_dense_ref(subset, *range)?
                    } else {
                        subset
                    };
                    return Some(Self::terminal_catalog_match(subset, self.keep_rows));
                }
                let (position, subset) = table.get_subset_positioned(key)?;
                let subset = if let Some(range) = intersect_outer {
                    intersect_with_dense_ref(subset, *range)?
                } else {
                    subset
                };
                Some(Self::catalog_match(
                    position,
                    subset,
                    continuations,
                    self.keep_rows,
                    *child_shape,
                ))
            }
            ProbeIndex::CachedColumn {
                intersect_outer,
                table,
                continuations,
                child_shape,
            } => {
                debug_assert_eq!(key.len(), 1);
                let table: &'rows Index<ColumnIndex> = table;
                if *child_shape == ChildShape::Leaf || !self.keep_rows {
                    debug_assert!(self.keep_rows || *child_shape == ChildShape::Leaf);
                    let subset = table.get_subset(&key[0])?;
                    let subset = if let Some(range) = intersect_outer {
                        intersect_with_dense_ref(subset, *range)?
                    } else {
                        subset
                    };
                    return Some(Self::terminal_catalog_match(subset, self.keep_rows));
                }
                let (position, subset) = table.get_subset_positioned(&key[0])?;
                let subset = if let Some(range) = intersect_outer {
                    intersect_with_dense_ref(subset, *range)?
                } else {
                    subset
                };
                Some(Self::catalog_match(
                    position,
                    subset,
                    continuations,
                    self.keep_rows,
                    *child_shape,
                ))
            }
            ProbeIndex::ProjectedRoot(projected) => projected
                .get(key)
                .map(|rows| Self::keep_or_discard(rows, self.keep_rows)),
            ProbeIndex::SmallColumn(index) => {
                let [value] = key else {
                    return None;
                };
                let key_index = index.find(*value)?;
                Some(if self.keep_rows {
                    ProbeMatch::Rows(AtomRows::Inline(index.rows_at(key_index)))
                } else {
                    ProbeMatch::Present
                })
            }
            ProbeIndex::SmallExact(exact) => exact.get(key, self.keep_rows),
            ProbeIndex::Packed(packed) => packed
                .get(key)
                .map(|rows| Self::keep_or_discard(rows, self.keep_rows)),
        }
    }

    fn sorted_scalar_probe(&self) -> Option<SortedScalarProbe<'_, 'exec>> {
        match &self.ix {
            ProbeIndex::ProjectedRoot(projected) if projected.columns.len() == 1 => {
                Some(SortedScalarProbe::Projected(projected.first))
            }
            ProbeIndex::SmallColumn(index) => Some(SortedScalarProbe::Small(index)),
            ProbeIndex::Packed(packed) if packed.columns.len() == 1 => {
                Some(SortedScalarProbe::Packed(packed.first))
            }
            ProbeIndex::CachedTuple { .. }
            | ProbeIndex::CachedColumn { .. }
            | ProbeIndex::ProjectedRoot(..)
            | ProbeIndex::SmallExact(..)
            | ProbeIndex::Packed(..) => None,
        }
    }

    fn sorted_match_at(&self, key_index: usize) -> ProbeMatch<'rows, 'exec> {
        match &self.ix {
            ProbeIndex::ProjectedRoot(projected) if projected.columns.len() == 1 => {
                Self::keep_or_discard(projected.scalar_rows(key_index), self.keep_rows)
            }
            ProbeIndex::SmallColumn(index) => {
                if self.keep_rows {
                    ProbeMatch::Rows(AtomRows::Inline(index.rows_at(key_index)))
                } else {
                    ProbeMatch::Present
                }
            }
            ProbeIndex::Packed(packed) if packed.columns.len() == 1 => {
                let rows = AtomRows::Packed(PackedCursor::new(packed.first, key_index));
                Self::keep_or_discard(rows, self.keep_rows)
            }
            ProbeIndex::CachedTuple { .. }
            | ProbeIndex::CachedColumn { .. }
            | ProbeIndex::ProjectedRoot(..)
            | ProbeIndex::SmallExact(..)
            | ProbeIndex::Packed(..) => {
                unreachable!("only a sorted scalar probe has a match ordinal")
            }
        }
    }

    fn for_each(&self, mut f: impl FnMut(&[Value], ProbeMatch<'rows, 'exec>)) {
        match &self.ix {
            ProbeIndex::CachedTuple {
                intersect_outer,
                table,
                continuations,
                child_shape,
            } => {
                let table: &'rows Index<TupleIndex> = table;
                table.for_each_positioned(|position, key, subset| {
                    let subset = if let Some(range) = intersect_outer {
                        let Some(subset) = intersect_with_dense_ref(subset, *range) else {
                            return;
                        };
                        subset
                    } else {
                        subset
                    };
                    f(
                        key,
                        Self::catalog_match(
                            position,
                            subset,
                            continuations,
                            self.keep_rows,
                            *child_shape,
                        ),
                    );
                });
            }
            ProbeIndex::CachedColumn {
                intersect_outer,
                table,
                continuations,
                child_shape,
            } => {
                let table: &'rows Index<ColumnIndex> = table;
                table.for_each_positioned(|position, value, subset| {
                    let subset = if let Some(range) = intersect_outer {
                        let Some(subset) = intersect_with_dense_ref(subset, *range) else {
                            return;
                        };
                        subset
                    } else {
                        subset
                    };
                    f(
                        &[*value],
                        Self::catalog_match(
                            position,
                            subset,
                            continuations,
                            self.keep_rows,
                            *child_shape,
                        ),
                    );
                });
            }
            ProbeIndex::ProjectedRoot(projected) => projected.for_each(&mut |key, rows| {
                f(key, Self::keep_or_discard(rows, self.keep_rows));
            }),
            ProbeIndex::SmallColumn(index) => {
                for key_index in 0..index.n_keys {
                    let rows = if self.keep_rows {
                        ProbeMatch::Rows(AtomRows::Inline(index.rows_at(key_index)))
                    } else {
                        ProbeMatch::Present
                    };
                    f(&index.keys[key_index..key_index + 1], rows);
                }
            }
            ProbeIndex::SmallExact(..) => {
                unreachable!("small multi-column residuals are exact-probe only")
            }
            ProbeIndex::Packed(packed) => packed.for_each(&mut |key, rows| {
                f(key, Self::keep_or_discard(rows, self.keep_rows));
            }),
        }
    }

    fn for_each_shard(&self, shard: usize, mut f: impl FnMut(&[Value], ProbeMatch<'rows, 'exec>)) {
        match &self.ix {
            ProbeIndex::CachedTuple {
                intersect_outer,
                table,
                continuations,
                child_shape,
            } => {
                let table: &'rows Index<TupleIndex> = table;
                table.for_each_shard_positioned(shard, |position, key, subset| {
                    let subset = if let Some(range) = intersect_outer {
                        let Some(subset) = intersect_with_dense_ref(subset, *range) else {
                            return;
                        };
                        subset
                    } else {
                        subset
                    };
                    f(
                        key,
                        Self::catalog_match(
                            position,
                            subset,
                            continuations,
                            self.keep_rows,
                            *child_shape,
                        ),
                    );
                });
            }
            ProbeIndex::CachedColumn {
                intersect_outer,
                table,
                continuations,
                child_shape,
            } => {
                let table: &'rows Index<ColumnIndex> = table;
                table.for_each_shard_positioned(shard, |position, value, subset| {
                    let subset = if let Some(range) = intersect_outer {
                        let Some(subset) = intersect_with_dense_ref(subset, *range) else {
                            return;
                        };
                        subset
                    } else {
                        subset
                    };
                    f(
                        &[*value],
                        Self::catalog_match(
                            position,
                            subset,
                            continuations,
                            self.keep_rows,
                            *child_shape,
                        ),
                    );
                });
            }
            ProbeIndex::ProjectedRoot(..)
            | ProbeIndex::SmallColumn(..)
            | ProbeIndex::SmallExact(..)
            | ProbeIndex::Packed(..) => {
                unreachable!("only persistent root indexes expose physical shards")
            }
        }
    }

    /// Visit a disjoint ordinal range of an unsharded scalar index.
    ///
    /// `Intersect` stages always probe one column, so partitioning the first
    /// (and only) key level preserves complete key groups. The backing
    /// projected/packed index remains borrowed by every coarse task.
    fn for_each_range(
        &self,
        start: usize,
        scan_size: usize,
        mut f: impl FnMut(&[Value], ProbeMatch<'rows, 'exec>),
    ) {
        let end = start
            .checked_add(scan_size)
            .expect("top index range overflow");
        match &self.ix {
            ProbeIndex::ProjectedRoot(projected) => {
                debug_assert_eq!(projected.columns.len(), 1);
                assert!(end <= projected.first.len());
                for key_index in start..end {
                    let key = [projected.first.value_at(key_index)];
                    f(
                        &key,
                        Self::keep_or_discard(projected.scalar_rows(key_index), self.keep_rows),
                    );
                }
            }
            ProbeIndex::SmallColumn(index) => {
                assert!(end <= index.n_keys);
                for key_index in start..end {
                    let rows = if self.keep_rows {
                        ProbeMatch::Rows(AtomRows::Inline(index.rows_at(key_index)))
                    } else {
                        ProbeMatch::Present
                    };
                    f(&index.keys[key_index..key_index + 1], rows);
                }
            }
            ProbeIndex::Packed(packed) => {
                debug_assert_eq!(packed.columns.len(), 1);
                let values = packed.first.values();
                assert!(end <= values.len());
                for key_index in start..end {
                    let rows = AtomRows::Packed(PackedCursor::new(packed.first, key_index));
                    f(
                        &values[key_index..key_index + 1],
                        Self::keep_or_discard(rows, self.keep_rows),
                    );
                }
            }
            ProbeIndex::CachedTuple { .. } | ProbeIndex::CachedColumn { .. } => {
                unreachable!("persistent indexes use physical shard partitions")
            }
            ProbeIndex::SmallExact(..) => {
                unreachable!("a scalar intersection cannot use an exact tuple probe")
            }
        }
    }

    fn shard_count(&self) -> Option<usize> {
        match &self.ix {
            ProbeIndex::CachedTuple { table, .. } => Some(table.shard_count()),
            ProbeIndex::CachedColumn { table, .. } => Some(table.shard_count()),
            ProbeIndex::ProjectedRoot(..)
            | ProbeIndex::SmallColumn(..)
            | ProbeIndex::SmallExact(..)
            | ProbeIndex::Packed(..) => None,
        }
    }

    fn shard_len(&self, shard: usize) -> Option<usize> {
        match &self.ix {
            ProbeIndex::CachedTuple { table, .. } => Some(table.shard_len(shard)),
            ProbeIndex::CachedColumn { table, .. } => Some(table.shard_len(shard)),
            ProbeIndex::ProjectedRoot(..)
            | ProbeIndex::SmallColumn(..)
            | ProbeIndex::SmallExact(..)
            | ProbeIndex::Packed(..) => None,
        }
    }

    fn len(&self) -> usize {
        match &self.ix {
            ProbeIndex::CachedTuple { table, .. } => table.len(),
            ProbeIndex::CachedColumn { table, .. } => table.len(),
            ProbeIndex::ProjectedRoot(projected) => projected.first.len(),
            ProbeIndex::SmallColumn(index) => index.len(),
            ProbeIndex::SmallExact(exact) => exact.len(),
            // Intersect stages are scalar. Tuple-packed probers are used only
            // for exact probes, so the first-level count is sufficient here.
            ProbeIndex::Packed(packed) => packed.first.values().len(),
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
        // let mut rule_reports: HashMap<String, Vec<RuleReport>>;
        let mut rule_reports: HashMap<Arc<str>, Vec<RuleReport>>;
        let exec_state = ExecutionState::new(self.read_only_view(), Default::default());
        if parallelize_db_level_op(self.total_size_estimate) {
            let dash_rule_reports: Arc<DashMap<Arc<str>, Vec<RuleReport>>> =
                Arc::new(DashMap::default());
            let db: &Database = self;
            egglog_concurrency::scope(|scope| {
                for (plan, desc, symbol_map) in rule_set.plans.values() {
                    let report_plan = match report_level {
                        ReportLevel::TimeOnly => None,
                        ReportLevel::WithPlan | ReportLevel::StageInfo => {
                            Some(plan.to_report(symbol_map))
                        }
                    };

                    let dash_rule_reports = dash_rule_reports.clone();
                    let desc = desc.clone();
                    let exec_state = exec_state.seed();
                    let match_counter = match_counter.clone();
                    let trie_cache = trie_cache.clone();
                    scope.spawn(move |_| {
                        // The arena and every prepared slot belong to exactly one
                        // logical query. A nested scope ensures no descendant task
                        // can retain an arena reference after this job reclaims it.
                        let arena = SharedArena::new();
                        let search_and_apply_timer = Instant::now();
                        let join_state = JoinState::new(db, exec_state, trie_cache, &arena);
                        let mut binding_info = BindingInfo::default();
                        let mut roots_ready = true;
                        for (id, info) in plan.atoms().iter() {
                            let headers: SmallVec<[&JoinHeader; 2]> =
                                plan.header().iter().filter(|h| h.atom == id).collect();
                            match join_state.root_node(info.table, &headers) {
                                Some(node) => binding_info.insert_node(id, node),
                                None => {
                                    roots_ready = false;
                                    break;
                                }
                            }
                        }

                        // Prepared slots are execution-local and can be large.
                        // Empty roots are common in fixpoint confirmation
                        // passes, so do not construct the sidecar until the
                        // query can actually run.
                        let prepared_index =
                            roots_ready.then(|| PreparedPlanIndexes::new(db, plan));
                        let search_and_apply_time = egglog_concurrency::scope(|query_scope| {
                            let mut action_buf = ScopedActionBuffer::new(
                                query_scope,
                                rule_set,
                                match_counter.clone(),
                            );

                            'eval: {
                                let Some(prepared_index) = prepared_index.as_ref() else {
                                    break 'eval;
                                };
                                match (plan, prepared_index) {
                                    (
                                        Plan::SinglePlan(plan),
                                        PreparedPlanIndexes::Single(prepared),
                                    ) => {
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

                                        for (mat_id, (stage_block, prepared_block)) in plan
                                            .stages
                                            .blocks
                                            .iter()
                                            .zip(prepared_blocks)
                                            .enumerate()
                                        {
                                            let mat_id = MatId::from_usize(mat_id);
                                            // Keep task-local join state alive until the stage has
                                            // quiesced. Besides making destruction deterministic,
                                            // this prevents many workers from concurrently decrementing
                                            // the same plan-state Arc counters as their tasks finish.
                                            let retired_states = RetiredLocalStates::default();
                                            egglog_concurrency::scope(|stage_scope| {
                                                let mut materializer = ScopedMaterializer {
                                                    scope: stage_scope,
                                                    retired_states: &retired_states,
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
                                    _ => {
                                        unreachable!("prepared plan shape must match logical plan")
                                    }
                                }
                            }
                            // Preserve the historical per-rule timing boundary: the
                            // ruleset-wide timer includes this final flush, but the
                            // individual rule report does not.
                            let search_and_apply_time = search_and_apply_timer.elapsed();
                            if action_buf.needs_flush {
                                action_buf.flush(&mut exec_state.to_execution_state());
                            }
                            search_and_apply_time
                        });

                        drop(binding_info);
                        drop(prepared_index);
                        drop(join_state);
                        drop(arena);

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
            // Just run all of the plans in order with a single in-place action
            // buffer.
            let mut action_buf = InPlaceActionBuffer {
                rule_set,
                match_counter: match_counter.as_ref(),
                batches: Default::default(),
            };
            for (plan, desc, symbol_map) in rule_set.plans.values() {
                // Serial recursive work is inline, so a lexical block is enough
                // to prove that every arena reference dies before this query's
                // arena is reclaimed.
                let arena = SharedArena::new();
                let report_plan = match report_level {
                    ReportLevel::TimeOnly => None,
                    ReportLevel::WithPlan | ReportLevel::StageInfo => {
                        Some(plan.to_report(symbol_map))
                    }
                };

                let search_and_apply_timer = Instant::now();
                {
                    let join_state =
                        JoinState::new(self, exec_state.seed(), trie_cache.clone(), &arena);
                    let mut binding_info = BindingInfo::default();
                    'eval: {
                        for (id, info) in plan.atoms().iter() {
                            let headers: SmallVec<[&JoinHeader; 2]> =
                                plan.header().iter().filter(|h| h.atom == id).collect();
                            match join_state.root_node(info.table, &headers) {
                                Some(node) => binding_info.insert_node(id, node),
                                None => break 'eval,
                            }
                        }
                        // See the parallel path above. This also ensures any
                        // arena addresses in prepared slots are dropped inside
                        // the arena's lexical lifetime.
                        let prepared_index = PreparedPlanIndexes::new(self, plan);
                        match (plan, &prepared_index) {
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
                                    materializations
                                        .insert(MatId::from_usize(i), Default::default());
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
                                        Arc::new(
                                            materializer.materializations.take(mat_id).unwrap(),
                                        ),
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
                }
                let search_and_apply_time = search_and_apply_timer.elapsed();

                drop(arena);

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
        // No query can use the cross-plan roots after the execution scopes
        // above have joined. Release their cached projections before merging
        // table updates so the two allocation peaks do not overlap.
        drop(trie_cache);
        // Every query-local prepared sidecar and arena has been dropped, and
        // the parallel ruleset scope has joined, before catalog indexes reset.
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

struct JoinState<'db, 'state, 'exec> {
    db: &'db Database,
    exec_state: ExecutionStateSeed<'db, 'state>,
    /// Cached thread-local pool for SortedOffsetVector allocations.
    /// Stored here to avoid a per-call `with_pool_set` TLS access in `get_index`.
    pool: Pool<SortedOffsetVector>,
    /// Cross-plan trie-root cache for the current `run_rule_set`, or `None` when
    /// sharing is disabled (small run, or nothing reused across plans).
    trie_cache: Option<Arc<TrieCache>>,
    arena: &'exec SharedArena,
    handle: LazyArenaHandle<'exec>,
    packed_scratch: RefCell<Vec<(Value, RowId)>>,
}

/// Canonical signature of a trie root: the table plus its sorted header (fast)
/// constraints. Distinct signatures get distinct base ids from [`TrieCache`].
type BaseSig = (TableId, SmallVec<[Constraint; 2]>);

/// Key for a shared trie root: the table plus an interned id for its fast
/// (header) constraints.
type RootKey = (TableId, u32);

/// One round-local root projection can be reused by plans that share the same
/// root subset. Slow constraints are part of the key because they are applied
/// before projection; sorting them makes conjunction order irrelevant.
#[derive(Clone, Eq, Hash, PartialEq)]
struct RootProjectionKey {
    column: ColumnId,
    constraints: SmallVec<[Constraint; 2]>,
}

struct RootProjection {
    /// Final immutable scalar-index representation. Unlike the earlier pair
    /// cache, this is probed directly: queries do not copy it into their arenas.
    /// The trailing entry is an offset-only sentinel.
    keys: Box<[(Value, u32)]>,
    rows: Box<[RowId]>,
}

impl RootProjection {
    fn from_sorted_pairs(pairs: Vec<(Value, RowId)>) -> Self {
        debug_assert!(pairs.windows(2).all(|pair| pair[0] <= pair[1]));
        let distinct = pairs
            .iter()
            .enumerate()
            .filter(|(index, pair)| *index == 0 || pairs[*index - 1].0 != pair.0)
            .count();
        let mut keys = Vec::with_capacity(distinct + 1);
        let mut rows = Vec::with_capacity(pairs.len());
        for (value, row) in pairs {
            if keys.last().map(|&(key, _)| key) != Some(value) {
                keys.push((
                    value,
                    u32::try_from(rows.len())
                        .expect("a projected root index cannot contain more than u32::MAX rows"),
                ));
            }
            rows.push(row);
        }
        keys.push((
            Value::new_const(0),
            u32::try_from(rows.len())
                .expect("a projected root index cannot contain more than u32::MAX rows"),
        ));
        Self {
            keys: keys.into_boxed_slice(),
            rows: rows.into_boxed_slice(),
        }
    }

    fn len(&self) -> usize {
        self.keys.len().saturating_sub(1)
    }

    fn find(&self, value: Value) -> Option<usize> {
        let len = self.len();
        self.keys[..len]
            .binary_search_by_key(&value, |&(key, _)| key)
            .ok()
    }

    fn value_at(&self, key_index: usize) -> Value {
        assert!(key_index < self.len(), "projected root key out of bounds");
        self.keys[key_index].0
    }

    fn subset_at(&self, key_index: usize) -> SubsetRef<'_> {
        assert!(key_index < self.len(), "projected root key out of bounds");
        let start = self.keys[key_index].1 as usize;
        let end = self.keys[key_index + 1].1 as usize;
        let rows = &self.rows[start..end];
        debug_assert!(!rows.is_empty());
        let first = rows[0];
        let last = rows[rows.len() - 1];
        if last.index() - first.index() == rows.len() - 1 {
            SubsetRef::Dense(OffsetRange::new(first, last.inc()))
        } else {
            // SAFETY: construction consumes pairs sorted by `(Value, RowId)`,
            // so every equal-value range is RowId ordered.
            SubsetRef::Sparse(unsafe { SortedOffsetSlice::new_unchecked(rows) })
        }
    }
}

type RootProjectionSlot = Arc<OnceLock<RootProjection>>;
type RootProjectionMap = DashMap<RootProjectionKey, RootProjectionSlot>;

/// A cache of trie roots shared across all plans within a single
/// `run_rule_set` call. Two plans that constrain the same table with the same
/// fast constraints share the owning root subset. Plan-execution packed
/// descendants remain separate.
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

/// Owning root subset for an atom. Lower trie levels are execution-scoped
/// packed nodes rather than persistent `TrieNode`s.
pub(crate) struct TrieNode {
    subset: Subset,
    /// Shared roots lazily cache sorted top-level projections across plans.
    /// Child publication remains query-local in the packed arena.
    root_projections: Option<OnceLock<RootProjectionMap>>,
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
            root_projections: None,
        }
    }

    fn new_shared(subset: Subset) -> Self {
        Self {
            subset,
            root_projections: Some(OnceLock::new()),
        }
    }

    fn projection_slot(
        &self,
        column: ColumnId,
        constraints: &[Constraint],
    ) -> Option<RootProjectionSlot> {
        let projections = self.root_projections.as_ref()?.get_or_init(|| {
            let shards = crate::parallel::current_num_threads()
                .next_power_of_two()
                .max(2);
            DashMap::with_hasher_and_shard_amount(Default::default(), shards)
        });
        let mut canonical: SmallVec<[Constraint; 2]> = constraints.iter().cloned().collect();
        canonical.sort_unstable();
        let key = RootProjectionKey {
            column,
            constraints: canonical,
        };
        Some(match projections.entry(key) {
            Entry::Occupied(entry) => entry.get().clone(),
            Entry::Vacant(entry) => {
                let slot = Arc::new(OnceLock::new());
                entry.insert(slot.clone());
                slot
            }
        })
    }
}

/// Visit every atom whose trie node can be read while executing `stage`.
/// Keep this match exhaustive: a new stage variant must declare its subset
/// dependencies before task-state projection can remain sound.
fn for_each_stage_atom(stage: &JoinStage, mut f: impl FnMut(AtomId)) {
    match stage {
        JoinStage::Intersect { scans, .. } => {
            scans.iter().for_each(|scan| f(scan.atom));
        }
        JoinStage::FusedIntersect {
            cover,
            to_intersect,
            ..
        } => {
            f(cover.to_index.atom);
            to_intersect
                .iter()
                .for_each(|(scan, _)| f(scan.to_index.atom));
        }
        JoinStage::FusedIntersectMat { to_intersect, .. } => {
            to_intersect
                .iter()
                .for_each(|(scan, _)| f(scan.to_index.atom));
        }
    }
}

/// Visit the prepared identity of every residual index probe in `stage`.
/// Cover scans consume a subset directly and therefore do not need a packed
/// child family of their own.
fn for_each_stage_indexed_access(
    stage: &JoinStage,
    prepared: &[PreparedIndexSlot],
    mut f: impl FnMut(AtomId, AccessId),
) {
    match stage {
        JoinStage::Intersect { scans, .. } => {
            debug_assert_eq!(scans.len(), prepared.len());
            scans
                .iter()
                .zip(prepared)
                .for_each(|(scan, slot)| f(scan.atom, slot.access));
        }
        JoinStage::FusedIntersect { to_intersect, .. }
        | JoinStage::FusedIntersectMat { to_intersect, .. } => {
            debug_assert_eq!(to_intersect.len(), prepared.len());
            to_intersect
                .iter()
                .zip(prepared)
                .for_each(|((scan, _), slot)| f(scan.to_index.atom, slot.access));
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AtomTailUse {
    keep_rows: bool,
    child_shape: ChildShape,
}

/// Find the first dynamically reorderable phase that touches `atom`. That
/// phase determines the packed child representation; finding any such phase
/// also proves that the current rows must survive into the tail.
fn scan_atom_tail_use(
    atom: AtomId,
    stages: &[JoinStage],
    prepared: &PreparedJoinIndexes,
    instr_order: &InstrOrder,
    resume_pos: usize,
) -> AtomTailUse {
    let mut phase_start = resume_pos;
    while phase_start < instr_order.len() {
        let first_stage = &stages[instr_order.get(phase_start)];
        let phase_end = if is_reorder_barrier(first_stage) {
            phase_start + 1
        } else {
            (phase_start + 1..instr_order.len())
                .find(|&position| is_reorder_barrier(&stages[instr_order.get(position)]))
                .unwrap_or(instr_order.len())
        };

        let mut touched = false;
        let mut first_access = None;
        let mut multiple_accesses = false;
        for position in phase_start..phase_end {
            let stage_index = instr_order.get(position);
            let stage = &stages[stage_index];
            for_each_stage_atom(stage, |candidate| touched |= candidate == atom);
            for_each_stage_indexed_access(
                stage,
                prepared.stage(stage_index),
                |candidate, access| {
                    if candidate != atom {
                        return;
                    }
                    if let Some(first) = first_access {
                        multiple_accesses |= first != access;
                    } else {
                        first_access = Some(access);
                    }
                },
            );
        }

        if touched {
            let child_shape = if multiple_accesses {
                ChildShape::Dynamic {
                    families: prepared.access_count(atom),
                }
            } else if first_access.is_some() {
                ChildShape::Direct
            } else {
                // A cover consumes this subset directly, so any index in a
                // later phase starts from the cover's residual.
                ChildShape::Leaf
            };
            return AtomTailUse {
                keep_rows: true,
                child_shape,
            };
        }
        phase_start = phase_end;
    }

    AtomTailUse {
        keep_rows: false,
        child_shape: ChildShape::Leaf,
    }
}

fn atom_tail_use(
    atom: AtomId,
    stages: &[JoinStage],
    prepared: &PreparedJoinIndexes,
    remaining_stages: Option<u64>,
    instr_order: &InstrOrder,
    resume_pos: usize,
) -> AtomTailUse {
    let Some((masks, remaining_stages)) = prepared.tail_masks().zip(remaining_stages) else {
        return scan_atom_tail_use(atom, stages, prepared, instr_order, resume_pos);
    };
    let result = masks.atom_tail_use(atom, remaining_stages, prepared.access_count(atom));
    #[cfg(debug_assertions)]
    debug_assert_eq!(
        result,
        scan_atom_tail_use(atom, stages, prepared, instr_order, resume_pos),
        "prepared tail masks diverged from the exact stage scanner"
    );
    result
}

#[cfg(test)]
fn packed_child_shape_in_tail(
    atom: AtomId,
    stages: &[JoinStage],
    prepared: &PreparedJoinIndexes,
    instr_order: &InstrOrder,
    resume_pos: usize,
) -> ChildShape {
    let remaining_stages = prepared.all_stage_mask().map(|_| {
        (resume_pos..instr_order.len()).fold(0u64, |mask, position| {
            mask | (1u64 << instr_order.get(position))
        })
    });
    atom_tail_use(
        atom,
        stages,
        prepared,
        remaining_stages,
        instr_order,
        resume_pos,
    )
    .child_shape
}

fn for_each_stage_materialization(stage: &JoinStage, mut f: impl FnMut(MatId)) {
    match stage {
        JoinStage::Intersect { .. } | JoinStage::FusedIntersect { .. } => {}
        JoinStage::FusedIntersectMat { cover, .. } => f(*cover),
    }
}

fn materialization_is_live_in_tail(
    stages: &[JoinStage],
    instr_order: &InstrOrder,
    resume_pos: usize,
    materialization: MatId,
) -> bool {
    (resume_pos..instr_order.len()).any(|position| {
        let mut found = false;
        for_each_stage_materialization(&stages[instr_order.get(position)], |candidate| {
            found |= candidate == materialization;
        });
        found
    })
}

type BindingSet = Vec<(SmallVec<[Variable; 4]>, Arc<TaggedRowBuffer<SmallValueVec>>)>;

#[derive(Default)]
struct BindingInfo<'rows, 'exec> {
    bindings: DenseIdMap<Variable, Value>,
    binding_sets: BindingSet,
    subsets: DenseIdMap<AtomId, AtomRows<'rows, 'exec>>,
    materializations: DenseIdMap<MatId, Arc<IndexMap<Vec<Value>, RowBuffer>>>,
}

impl<'rows, 'exec> BindingInfo<'rows, 'exec>
where
    'exec: 'rows,
{
    /// Clone the binding state needed to resume a join at `resume_pos`.
    ///
    /// Recursive tasks never execute an already-completed stage, so atom trie
    /// nodes referenced only by that prefix are dead task state. Constructing
    /// the subset map from the remaining stages avoids both the increment and
    /// eventual decrement of their shared `Arc` counts.
    fn clone_for_join_tail<'short>(
        &self,
        stages: &[JoinStage],
        instr_order: &InstrOrder,
        resume_pos: usize,
    ) -> BindingInfo<'short, 'exec>
    where
        'rows: 'short,
    {
        let mut subsets: DenseIdMap<AtomId, AtomRows<'short, 'exec>> = DenseIdMap::new();
        for position in resume_pos..instr_order.len() {
            for_each_stage_atom(&stages[instr_order.get(position)], |atom| {
                if !subsets.contains_key(atom)
                    && let Some(node) = self.subsets.get(atom)
                {
                    subsets.insert(atom, node.clone());
                }
            });
        }
        let mut materializations = DenseIdMap::new();
        for position in resume_pos..instr_order.len() {
            for_each_stage_materialization(&stages[instr_order.get(position)], |mat_id| {
                if !materializations.contains_key(mat_id) {
                    let materialization = self
                        .materializations
                        .get(mat_id)
                        .expect("task state is missing a live materialization");
                    materializations.insert(mat_id, Arc::clone(materialization));
                }
            });
        }
        BindingInfo {
            bindings: self.bindings.clone(),
            binding_sets: self.binding_sets.clone(),
            subsets,
            materializations,
        }
    }

    /// Initializes the atom-related metadata in the [`BindingInfo`].    
    fn insert_subset(&mut self, atom: AtomId, subset: Subset) {
        let rows = match subset {
            Subset::Dense(range) => AtomRows::Dense(range),
            subset => AtomRows::Root(Arc::new(TrieNode::new(subset))),
        };
        self.subsets.insert(atom, rows);
    }

    fn insert_node(&mut self, atom: AtomId, node: impl Into<AtomRows<'rows, 'exec>>) {
        self.subsets.insert(atom, node.into());
    }

    /// Probers returned from [`JoinState::get_index`] will move atom-related state out of the
    /// [`BindingInfo`]. Once the caller is done using a prober, this method moves it back.
    fn move_back(&mut self, atom: AtomId, prober: Prober<'_, 'rows, 'exec>) {
        self.subsets.insert(atom, prober.source);
    }

    fn move_back_node(&mut self, atom: AtomId, node: impl Into<AtomRows<'rows, 'exec>>) {
        self.subsets.insert(atom, node.into());
    }

    fn has_empty_subset(&self, atom: AtomId) -> bool {
        self.subsets[atom].is_empty()
    }

    fn unwrap_val(&mut self, atom: AtomId) -> AtomRows<'rows, 'exec> {
        self.subsets.unwrap_val(atom)
    }
}

impl<'a, 'state, 'exec> JoinState<'a, 'state, 'exec> {
    fn new(
        db: &'a Database,
        exec_state: ExecutionStateSeed<'a, 'state>,
        trie_cache: Option<Arc<TrieCache>>,
        arena: &'exec SharedArena,
    ) -> Self {
        Self {
            db,
            exec_state,
            pool: with_pool_set(|ps| ps.get_pool()),
            trie_cache,
            arena,
            handle: LazyArenaHandle::new(arena),
            packed_scratch: RefCell::new(Vec::new()),
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
                let node = Arc::new(TrieNode::new_shared(subset));
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

    fn build_packed_node(
        &self,
        table: WrappedTableRef<'_>,
        subset: SubsetRef<'_>,
        can_be_stale: bool,
        constraints: &[Constraint],
        column: ColumnId,
        child_shape: ChildShape,
    ) -> &'exec PackedTrieNode<'exec> {
        if constraints.is_empty() {
            // Table scans already omit stale SortedWritesTable rows, so a
            // separate live-subset pass would only scan the same root twice.
            return PackedTrieNode::build_from_subset(
                self.handle.get(),
                table,
                subset,
                column,
                child_shape,
                &mut self.packed_scratch.borrow_mut(),
            );
        }
        let mut filtered = subset.to_owned(&self.pool);
        if can_be_stale && table.has_stale_rows() {
            filtered = table.refine_live(filtered);
        }
        filtered = table.refine(filtered, constraints);
        PackedTrieNode::build_from_subset(
            self.handle.get(),
            table,
            filtered.as_ref(),
            column,
            child_shape,
            &mut self.packed_scratch.borrow_mut(),
        )
    }

    /// Acquire the final-form scalar root index shared by plans using the same
    /// root subset and logical projection. The prepared sidecar retains the
    /// slot Arc, so the returned arrays can be borrowed for `'rows` without an
    /// Arc clone in every output frame.
    fn projected_root_index<'rows>(
        &self,
        root: &TrieNode,
        table: WrappedTableRef<'_>,
        constraints: &[Constraint],
        column: ColumnId,
        prepared: PreparedIndexRef<'rows>,
    ) -> Option<&'rows RootProjection> {
        self.trie_cache.as_ref()?;
        let slot = if let Some(slot) = prepared.state.projected_root.get() {
            slot
        } else {
            let candidate = root.projection_slot(column, constraints)?;
            prepared.state.projected_root.get_or_init(|| candidate)
        };
        let projection = slot.get_or_init(|| {
            let filtered = if constraints.is_empty() {
                None
            } else {
                let mut filtered = root.subset.as_ref().to_owned(&self.pool);
                if table.has_stale_rows() {
                    filtered = table.refine_live(filtered);
                }
                Some(table.refine(filtered, constraints))
            };
            let subset = filtered
                .as_ref()
                .map_or_else(|| root.subset.as_ref(), Subset::as_ref);
            let mut pairs = Vec::with_capacity(subset.size());
            table.for_each_col(subset, column, &mut |row_id, value| {
                pairs.push((value, row_id));
            });
            debug_assert!(
                pairs.windows(2).all(|pair| pair[0].1 <= pair[1].1),
                "root projection must preserve RowId order before sorting"
            );
            if pairs.len() < 64 {
                pairs.sort_unstable();
            } else {
                let mut sort_scratch =
                    vec![(Value::new_const(0), RowId::new_const(0)); pairs.len()];
                crate::hash_index::radix_sort_slice_by_value(&mut pairs, &mut sort_scratch);
            }
            RootProjection::from_sorted_pairs(pairs)
        });
        Some(projection)
    }

    fn packed_index_for_rows<'rows>(
        &self,
        rows: &AtomRows<'rows, 'exec>,
        table: WrappedTableRef<'_>,
        constraints: &[Constraint],
        column: ColumnId,
        child_shape: ChildShape,
        prepared: PreparedIndexRef<'rows>,
    ) -> &'exec PackedTrieNode<'exec>
    where
        'exec: 'rows,
    {
        match rows {
            AtomRows::Root(root) => {
                let address = *prepared.state.packed_root.get_or_init(|| {
                    self.build_packed_node(
                        table,
                        root.subset.as_ref(),
                        true,
                        constraints,
                        column,
                        child_shape,
                    ) as *const PackedTrieNode<'exec> as usize
                });
                // SAFETY: the slot is plan-execution scoped and can only be
                // initialized with a node from this plan's SharedArena.
                let node = unsafe { &*(address as *const PackedTrieNode<'exec>) };
                assert_eq!(node.child_shape(), child_shape);
                node
            }
            AtomRows::Catalog {
                subset,
                continuation,
            } => {
                let continuation = continuation
                    .expect("catalog rows needed a continuation slot for a later atom probe");
                let slot = continuation
                    .cache
                    .slot(continuation.position, prepared.access);
                let address = *slot.get_or_init(|| {
                    self.build_packed_node(table, *subset, false, constraints, column, child_shape)
                        as *const PackedTrieNode<'exec> as usize
                });
                // SAFETY: root continuation slots are allocated in and only
                // publish nodes from the same execution arena.
                let node = unsafe { &*(address as *const PackedTrieNode<'exec>) };
                assert_eq!(node.child_shape(), child_shape);
                node
            }
            AtomRows::Packed(cursor) => cursor.child_index_with(
                self.handle.get(),
                prepared.access.index(),
                child_shape,
                || {
                    self.build_packed_node(
                        table,
                        cursor.subset(),
                        false,
                        constraints,
                        column,
                        child_shape,
                    )
                },
            ),
            AtomRows::Inline(..) => {
                unreachable!("inline residuals must use a stack-owned probe")
            }
            AtomRows::Dense(range) => self.build_packed_node(
                table,
                SubsetRef::Dense(*range),
                true,
                constraints,
                column,
                child_shape,
            ),
        }
    }

    fn get_index<'ctx, 'rows>(
        &'ctx self,
        atoms: &Arc<DenseIdMap<AtomId, Atom>>,
        binding_info: &mut BindingInfo<'rows, 'exec>,
        request: ProbeRequest<'_, 'rows>,
    ) -> Prober<'ctx, 'rows, 'exec>
    where
        'exec: 'rows,
    {
        let ProbeRequest {
            atom,
            columns: cols,
            constraints,
            keep_rows,
            terminal_child_shape,
            prepared,
        } = request;
        assert!(
            !cols.is_empty(),
            "a join index must project at least one column"
        );
        let source = binding_info.subsets.unwrap_val(atom);

        let table_id = atoms[atom].table;
        let info = &self.db.tables[table_id];
        let all_cacheable = columns_are_cacheable(info, &cols);
        let whole_table = info.table.all();
        let root_range = match &source {
            AtomRows::Root(root) => match &root.subset {
                Subset::Dense(range) => Some(*range),
                Subset::Sparse(_) => None,
            },
            _ => None,
        };
        let can_use_catalog = root_range.is_some()
            && all_cacheable
            && constraints.is_empty()
            && !info.table.has_stale_rows()
            && whole_table.size() / 2 < source.size();

        let ix = if cols.len() == 1 && source.size() <= SMALL_RESIDUAL {
            ProbeIndex::SmallColumn(SmallColumnIndex::new(
                info.table.as_ref(),
                source.subset(),
                constraints,
                cols[0],
            ))
        } else if let AtomRows::Inline(rows) = &source {
            ProbeIndex::SmallExact(SmallExactProbe::new(
                info.table.as_ref(),
                *rows,
                cols,
                constraints,
            ))
        } else if can_use_catalog {
            let range = root_range.expect("catalog eligibility requires a dense root");
            let needs_intersect =
                !(whole_table.is_dense() && source.subset().bounds() == whole_table.bounds());
            let intersect_outer = needs_intersect.then_some(range);
            if cols.len() == 1 {
                debug_assert_eq!(prepared.kind, PreparedIndexKind::Column);
                let PreparedIndexCache::Column(index) = &prepared.state.cache else {
                    unreachable!("single-column scan must have a prepared column index")
                };
                let index = index
                    .get_or_init(|| get_column_index_from_tableinfo(info, cols[0]))
                    .get()
                    .expect("prepared column index must already be refreshed");
                if terminal_child_shape != ChildShape::Leaf {
                    prepared.state.root_continuations.prepare(
                        terminal_child_shape,
                        index.shard_count(),
                        |shard| index.shard_len(shard),
                    );
                }
                ProbeIndex::CachedColumn {
                    intersect_outer,
                    table: index,
                    continuations: &prepared.state.root_continuations,
                    child_shape: terminal_child_shape,
                }
            } else {
                debug_assert_eq!(prepared.kind, PreparedIndexKind::Tuple);
                let PreparedIndexCache::Tuple(index) = &prepared.state.cache else {
                    unreachable!("multi-column scan must have a prepared tuple index")
                };
                let index = index
                    .get_or_init(|| get_index_from_tableinfo(info, cols.as_slice()))
                    .get()
                    .expect("prepared tuple index must already be refreshed");
                if terminal_child_shape != ChildShape::Leaf {
                    prepared.state.root_continuations.prepare(
                        terminal_child_shape,
                        index.shard_count(),
                        |shard| index.shard_len(shard),
                    );
                }
                ProbeIndex::CachedTuple {
                    intersect_outer,
                    table: index,
                    continuations: &prepared.state.root_continuations,
                    child_shape: terminal_child_shape,
                }
            }
        } else {
            let first_child_shape = if cols.len() > 1 {
                ChildShape::Direct
            } else {
                terminal_child_shape
            };
            let projected_root = match &source {
                AtomRows::Root(root) => self.projected_root_index(
                    root,
                    info.table.as_ref(),
                    constraints,
                    cols[0],
                    prepared,
                ),
                _ => None,
            };
            if let Some(first) = projected_root {
                if first_child_shape != ChildShape::Leaf {
                    prepared
                        .state
                        .root_continuations
                        .prepare(first_child_shape, 1, |_| first.len());
                }
                ProbeIndex::ProjectedRoot(RootProjectionProbe {
                    first,
                    columns: cols,
                    table: info.table.as_ref(),
                    continuations: &prepared.state.root_continuations,
                    access: prepared.access,
                    handle: &self.handle,
                    scratch: &self.packed_scratch,
                    terminal_child_shape,
                })
            } else {
                let first = self.packed_index_for_rows(
                    &source,
                    info.table.as_ref(),
                    constraints,
                    cols[0],
                    first_child_shape,
                    prepared,
                );
                ProbeIndex::Packed(PackedProbe {
                    first,
                    columns: cols,
                    table: info.table.as_ref(),
                    handle: &self.handle,
                    scratch: &self.packed_scratch,
                    terminal_child_shape,
                })
            }
        };
        Prober {
            source,
            ix,
            keep_rows,
        }
    }

    /// Describe an eligible scalar index that could drive a top-level
    /// generic-join intersection.
    ///
    /// This mirrors leader selection and child-index shape in
    /// [`Self::run_plan`]. Probers move trie nodes out of `binding_info`, so
    /// restore every node before returning. Persistent catalog indexes use
    /// their physical hash shards, including when a dense timestamp range is
    /// intersected at probe time. Projected and packed filtered roots use
    /// disjoint ordinal ranges of the already-built scalar index, avoiding a
    /// scan-and-project handoff copy. Tiny leaders stay on the existing path.
    #[allow(clippy::too_many_arguments)]
    fn top_index_partitions<'rows>(
        &self,
        stages: &JoinStages,
        stage_index: usize,
        prepared: &'rows PreparedJoinIndexes,
        atoms: &Arc<DenseIdMap<AtomId, Atom>>,
        order: &InstrOrder,
        binding_info: &mut BindingInfo<'rows, 'exec>,
        workers: usize,
    ) -> Option<Vec<TopLevelPartition>>
    where
        'exec: 'rows,
    {
        let JoinStage::Intersect { scans, .. } = &stages.instrs[stage_index] else {
            return None;
        };
        if scans.is_empty()
            || workers <= 1
            || estimate_size(&stages.instrs[stage_index], binding_info)
                < MIN_TOP_INDEX_KEYS_PER_WORKER.saturating_mul(workers)
        {
            return None;
        }
        let stage_prepared = prepared.stage(stage_index);
        debug_assert_eq!(scans.len(), stage_prepared.len());
        let remaining_after_current = prepared
            .all_stage_mask()
            .map(|remaining| remaining & !(1u64 << stage_index));

        let mut leader = 0;
        let mut leader_size = usize::MAX;
        let mut probers = Vec::with_capacity(scans.len());
        for (i, (scan, prepared_slot)) in scans.iter().zip(stage_prepared).enumerate() {
            let tail = atom_tail_use(
                scan.atom,
                &stages.instrs,
                prepared,
                remaining_after_current,
                order,
                1,
            );
            let prober = self.get_index(
                atoms,
                binding_info,
                ProbeRequest::column(
                    scan,
                    false,
                    tail.child_shape,
                    prepared.resolve(prepared_slot),
                ),
            );
            let size = prober.len();
            if size < leader_size {
                leader = i;
                leader_size = size;
            }
            probers.push(prober);
        }
        let partitions = if let Some(shard_count) = probers[leader].shard_count() {
            let shards = (0..shard_count)
                .filter_map(|shard| {
                    let scan_size = probers[leader].shard_len(shard).unwrap_or(0);
                    (scan_size != 0).then_some(TopLevelPartition::IndexShard { shard, scan_size })
                })
                .collect::<Vec<_>>();
            top_index_shape_is_eligible(
                workers,
                leader_size,
                shards.len(),
                MIN_TOP_INDEX_KEYS_PER_WORKER,
            )
            .then_some(shards)
        } else {
            top_index_range_partitions(leader_size, workers, MIN_TOP_INDEX_KEYS_PER_WORKER)
        };
        for (scan, prober) in scans.iter().zip(probers) {
            binding_info.move_back(scan.atom, prober);
        }
        partitions
    }

    /// Return coarse partitions for the stage already sorted to the top of the
    /// join. Cached scalar intersections use physical index shards, while
    /// fused intersections use ordinal ranges of their cover subset. Later
    /// stages are deliberately not promoted.
    fn select_top_partitions<'rows>(
        &self,
        stages: &JoinStages,
        prepared: &'rows PreparedJoinIndexes,
        atoms: &Arc<DenseIdMap<AtomId, Atom>>,
        order: &InstrOrder,
        binding_info: &mut BindingInfo<'rows, 'exec>,
    ) -> Option<Vec<TopLevelPartition>>
    where
        'exec: 'rows,
    {
        if order.len() == 0 {
            return None;
        }

        let stage_index = order.get(0);
        let workers = crate::parallel::current_num_threads();
        let stage = &stages.instrs[stage_index];
        match stage {
            JoinStage::Intersect { .. } => self.top_index_partitions(
                stages,
                stage_index,
                prepared,
                atoms,
                order,
                binding_info,
                workers,
            ),
            JoinStage::FusedIntersect { cover, .. } => top_cover_partitions(
                binding_info.subsets[cover.to_index.atom].size(),
                workers,
                MIN_TOP_COVER_ROWS_PER_PARTITION,
            ),
            JoinStage::FusedIntersectMat { .. } => None,
        }
    }

    /// Runs the free join plan, starting with the header.
    ///
    /// A bit about the `instr_order` parameter: This defines the order in which the [`JoinStage`]
    /// instructions will run. We want to support cached [`SinglePlan`]s that may be based on stale
    /// ordering information. `instr_order` allows us to specify a new ordering of the instructions
    /// without mutating the plan itself: `run_plan` simply executes
    /// `plan.stages.instrs[instr_order[i]]` at stage `i`.
    ///
    /// This provides execution-local dynamic variable ordering while leaving
    /// the cached plan immutable.
    fn run_join_stages<
        'plan,
        'rows,
        'scope,
        A: NumericId + 'scope,
        BUF: ActionBuffer<'scope, 'exec, A>,
    >(
        &self,
        stages: &'plan JoinStages,
        prepared: &'rows PreparedJoinIndexes,
        atoms: &'plan Arc<DenseIdMap<AtomId, Atom>>,
        action: A,
        binding_info: &mut BindingInfo<'rows, 'exec>,
        action_buf: &mut BUF,
    ) where
        'a: 'scope,
        'state: 'scope,
        'exec: 'rows,
        'rows: 'scope,
        'plan: 'scope,
    {
        if log::log_enabled!(log::Level::Trace) {
            log::trace!("Starting running query stages:\n{stages:#?}");
        }
        for (_, node) in binding_info.subsets.iter() {
            if node.is_empty() {
                return;
            }
        }
        let mut order = InstrOrder::from_iter(0..stages.instrs.len());
        let mut leaf_scans: LeafScans = smallvec::smallvec![false; stages.instrs.len()];
        sort_plan_by_size(&mut order, &mut leaf_scans, 0, &stages.instrs, binding_info);
        let all_stages = prepared.all_stage_mask();

        // Schedule each coarse top partition as a global job and run its
        // complete subtree inline under the production `Serial` policy. The
        // experimental top-shard policy may instead fan out later recursive
        // work. Cached scalar indexes already provide physical hash shards;
        // fused covers use borrowed ordinal ranges, so seminaive and other
        // filtered roots need no row-ID copy. Buffers that cannot construct an
        // independent partition decline this path.
        if !stages.instrs.is_empty()
            && action_buf.supports_global_partition()
            && let Some(partitions) =
                self.select_top_partitions(stages, prepared, atoms, &order, binding_info)
        {
            let mut updates = FrameUpdates::with_capacity(0);
            for partition in partitions {
                let db = self.db;
                let exec_state_for_factory = self.exec_state;
                let exec_state_for_work = self.exec_state;
                let trie_cache = self.trie_cache.clone();
                let arena = self.arena;
                action_buf.recur_global_serial(
                    BorrowedLocalState {
                        binding_info,
                        instr_order: &mut order,
                        leaf_scans: &mut leaf_scans,
                        updates: &mut updates,
                    },
                    SubsetClonePlan {
                        stages: &stages.instrs,
                        resume_pos: 0,
                    },
                    move || exec_state_for_factory.to_execution_state(),
                    move |BorrowedLocalState {
                              binding_info,
                              instr_order,
                              leaf_scans,
                              ..
                          },
                          buf| {
                        let join_state: JoinState<'a, 'state, 'exec> =
                            JoinState::new(db, exec_state_for_work, trie_cache, arena);
                        join_state.run_plan(
                            stages,
                            prepared,
                            atoms,
                            action,
                            instr_order,
                            leaf_scans,
                            0,
                            Some(partition),
                            all_stages,
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
            all_stages,
            binding_info,
            action_buf,
        );
    }

    /// The core method for executing a free join plan.
    ///
    /// This method takes the plan, mutable data structures for variable binding
    /// and staging actions, and `cur`, the current stage of the plan. A top-level
    /// coarse partition also passes `top_partition`; recursive calls clear it
    /// so only the first intersection or fused cover scan is restricted.
    /// The associated top-shard buffer policy, rather than the presence of the
    /// partition itself, determines whether later recursive work may fan out.
    #[allow(clippy::too_many_arguments)]
    fn run_plan<'plan, 'rows, 'scope, A: NumericId + 'scope, BUF: ActionBuffer<'scope, 'exec, A>>(
        &self,
        stages: &'plan JoinStages,
        prepared: &'rows PreparedJoinIndexes,
        atoms: &'plan Arc<DenseIdMap<AtomId, Atom>>,
        action: A,
        instr_order: &mut InstrOrder,
        leaf_scans: &mut LeafScans,
        cur: usize,
        top_partition: Option<TopLevelPartition>,
        remaining_stages: Option<u64>,
        binding_info: &mut BindingInfo<'rows, 'exec>,
        action_buf: &mut BUF,
    ) where
        'a: 'scope,
        'state: 'scope,
        'exec: 'rows,
        'rows: 'scope,
        'plan: 'scope,
    {
        if self.exec_state.should_stop() {
            return;
        }

        #[cfg(debug_assertions)]
        if let Some(remaining_stages) = remaining_stages {
            let suffix = (cur..instr_order.len()).fold(0u64, |mask, position| {
                mask | (1u64 << instr_order.get(position))
            });
            debug_assert_eq!(
                remaining_stages, suffix,
                "the remaining-stage mask must describe the current physical suffix"
            );
        }

        if cur >= instr_order.len() {
            action_buf.push_bindings_factorized(
                action,
                &mut binding_info.bindings,
                &binding_info.binding_sets,
                self.exec_state,
            );
            return;
        }
        let mut cur_size = top_partition.map_or_else(
            || estimate_size(&stages.instrs[instr_order.get(cur)], binding_info),
            TopLevelPartition::scan_size,
        );
        if cur_size > 32 && cur % 3 == 1 && cur < instr_order.len() - 1 {
            // Re-evaluate the remaining suffix after observing the residuals
            // produced by earlier stages. Packed child families make the
            // resulting atom-local successor choice safe to cache again.
            sort_plan_by_size(instr_order, leaf_scans, cur, &stages.instrs, binding_info);
            cur_size = estimate_size(&stages.instrs[instr_order.get(cur)], binding_info);
        }
        // A rejected intersection candidate consumes a leader key without
        // producing a frame. Subtracting completed frames therefore maintains
        // a conservative upper bound, which is sufficient for choosing a
        // locality-oriented morsel size.
        let mut remaining_scan_upper_bound = cur_size;
        let mut chunk_size =
            action_buf.morsel_size(cur, instr_order.len(), remaining_scan_upper_bound);

        let stage_index = instr_order.get(cur);
        let stage = &stages.instrs[stage_index];
        let prepared_indexes = prepared.stage(stage_index);
        let remaining_after_current = remaining_stages.map(|remaining| {
            let stage_bit = 1u64 << stage_index;
            debug_assert_ne!(
                remaining & stage_bit,
                0,
                "the current stage must still be present in the remaining-stage mask"
            );
            remaining & !stage_bit
        });

        macro_rules! drain_updates_parallel {
            ($updates:expr) => {{
                if self.exec_state.should_stop() {
                    return;
                }
                let db = self.db;
                let exec_state_for_factory = self.exec_state;
                let exec_state_for_work = self.exec_state;
                let trie_cache = self.trie_cache.clone();
                let arena = self.arena;
                action_buf.recur(
                    BorrowedLocalState {
                        binding_info,
                        instr_order,
                        leaf_scans,
                        updates: &mut $updates,
                    },
                    SubsetClonePlan {
                        stages: &stages.instrs,
                        resume_pos: cur + 1,
                    },
                    move || exec_state_for_factory.to_execution_state(),
                    move |BorrowedLocalState {
                              binding_info,
                              instr_order,
                              leaf_scans,
                              updates,
                          },
                          buf| {
                        let join_state: JoinState<'a, 'state, 'exec> =
                            JoinState::new(db, exec_state_for_work, trie_cache, arena);
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
                                join_state.run_plan(
                                    stages,
                                    prepared,
                                    atoms,
                                    action,
                                    instr_order,
                                    leaf_scans,
                                    cur + 1,
                                    None,
                                    remaining_after_current,
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
        // Helper macros (not their own methods to appease the borrow checker).
        macro_rules! drain_updates_body {
            ($updates:expr) => {{
                if self.exec_state.should_stop() {
                    return;
                }
                // TODO: `supports_parallel_drain` is a hack because currently
                // `drain_updates_parallel!` is a bit slower because of the additional
                // `ExecutionState` clone.
                if cur < free_join_fork_depth()
                    && action_buf.supports_parallel_drain(cur, top_partition.is_some())
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
                                    self.exec_state,
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
                                    remaining_after_current,
                                    binding_info,
                                    action_buf,
                                );
                            }
                        }
                    })
                }
            }};
        }
        macro_rules! drain_updates {
            ($updates:expr) => {{
                let completed_frames = $updates.frames();
                drain_updates_body!($updates);
                remaining_scan_upper_bound =
                    remaining_scan_upper_bound.saturating_sub(completed_frames);
                chunk_size =
                    action_buf.morsel_size(cur, instr_order.len(), remaining_scan_upper_bound);
            }};
            ($updates:expr, false) => {{
                drain_updates_body!($updates);
            }};
        }

        // A sharded top-level job enumerates only its assigned physical index
        // shard. Every recursive call clears `top_partition`, so this macro is
        // used only by the leading prober of the initial `Intersect` stage.
        macro_rules! for_each_leader {
            ($prober:expr, $callback:expr) => {
                match top_partition {
                    Some(TopLevelPartition::IndexShard { shard, .. }) => {
                        $prober.for_each_shard(shard, $callback)
                    }
                    Some(TopLevelPartition::IndexRange {
                        start, scan_size, ..
                    }) => $prober.for_each_range(start, scan_size, $callback),
                    None => $prober.for_each($callback),
                    Some(TopLevelPartition::CoverRange { .. }) => {
                        unreachable!("a cover range can only drive a fused intersection")
                    }
                }
            };
        }

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
                    let tail = atom_tail_use(
                        a.atom,
                        &stages.instrs,
                        prepared,
                        remaining_after_current,
                        instr_order,
                        cur + 1,
                    );
                    let prober = self.get_index(
                        atoms,
                        binding_info,
                        ProbeRequest::column(
                            a,
                            tail.keep_rows,
                            tail.child_shape,
                            prepared.resolve(&prepared_indexes[0]),
                        ),
                    );
                    let mut updates = FrameUpdates::with_capacity(cmp::min(chunk_size, cur_size));
                    for_each_leader!(prober, |val, x| {
                        updates.push_binding(*var, val[0]);
                        x.refine(a.atom, &mut updates);
                        updates.finish_frame();
                        if updates.frames() >= chunk_size {
                            drain_updates!(updates);
                        }
                    });
                    drain_updates!(updates, false);
                    binding_info.move_back(a.atom, prober);
                }
                [a, b] => {
                    let a_tail = atom_tail_use(
                        a.atom,
                        &stages.instrs,
                        prepared,
                        remaining_after_current,
                        instr_order,
                        cur + 1,
                    );
                    let a_prober = self.get_index(
                        atoms,
                        binding_info,
                        ProbeRequest::column(
                            a,
                            a_tail.keep_rows,
                            a_tail.child_shape,
                            prepared.resolve(&prepared_indexes[0]),
                        ),
                    );
                    let b_tail = atom_tail_use(
                        b.atom,
                        &stages.instrs,
                        prepared,
                        remaining_after_current,
                        instr_order,
                        cur + 1,
                    );
                    let b_prober = self.get_index(
                        atoms,
                        binding_info,
                        ProbeRequest::column(
                            b,
                            b_tail.keep_rows,
                            b_tail.child_shape,
                            prepared.resolve(&prepared_indexes[1]),
                        ),
                    );

                    let ((smaller, smaller_scan), (larger, larger_scan)) =
                        if a_prober.len() <= b_prober.len() {
                            ((&a_prober, a), (&b_prober, b))
                        } else {
                            ((&b_prober, b), (&a_prober, a))
                        };

                    let mut updates = FrameUpdates::with_capacity(cmp::min(chunk_size, cur_size));
                    if let (Some(smaller_keys), Some(larger_keys)) =
                        (smaller.sorted_scalar_probe(), larger.sorted_scalar_probe())
                    {
                        // Both sides expose sorted scalar arrays. Leapfrog through the larger
                        // side directly, preserving the selected leader and its top-level ordinal
                        // partition. Successful ordinals reconstruct the same ProbeMatch as a
                        // scalar lookup, including row-carrying continuations needed by the tail.
                        let leader_range = match top_partition {
                            Some(TopLevelPartition::IndexRange {
                                start, scan_size, ..
                            }) => start..start + scan_size,
                            None => 0..smaller_keys.len(),
                            Some(TopLevelPartition::IndexShard { .. }) => {
                                unreachable!("a sorted scalar leader is never hash-sharded")
                            }
                            Some(TopLevelPartition::CoverRange { .. }) => {
                                unreachable!("a cover range cannot drive an intersection")
                            }
                        };
                        debug_assert!(leader_range.end <= smaller_keys.len());
                        let both_tiny = matches!(smaller_keys, SortedScalarProbe::Small(..))
                            && matches!(larger_keys, SortedScalarProbe::Small(..));
                        let comparable_sizes =
                            larger_keys.len() <= leader_range.len().saturating_mul(4);
                        if both_tiny || comparable_sizes {
                            // A plain two-finger merge is cheaper for tiny recursive residuals
                            // and similarly sized packed arrays, where a linear pass has better
                            // locality than setting up galloping bounds for every key.
                            let mut leader_index = leader_range.start;
                            let mut target_index = 0;
                            if leader_index != 0 && leader_index < leader_range.end {
                                let first_key = smaller_keys.value_at(leader_index);
                                seek_sorted_key(
                                    first_key,
                                    larger_keys.len(),
                                    &mut target_index,
                                    |key_index| larger_keys.value_at(key_index),
                                );
                            }
                            while leader_index < leader_range.end
                                && target_index < larger_keys.len()
                            {
                                let leader_key = smaller_keys.value_at(leader_index);
                                let target_key = larger_keys.value_at(target_index);
                                match leader_key.cmp(&target_key) {
                                    cmp::Ordering::Less => leader_index += 1,
                                    cmp::Ordering::Greater => target_index += 1,
                                    cmp::Ordering::Equal => {
                                        updates.push_binding(*var, leader_key);
                                        smaller
                                            .sorted_match_at(leader_index)
                                            .refine(smaller_scan.atom, &mut updates);
                                        larger
                                            .sorted_match_at(target_index)
                                            .refine(larger_scan.atom, &mut updates);
                                        updates.finish_frame();
                                        if updates.frames() >= chunk_size {
                                            drain_updates!(updates);
                                        }
                                        leader_index += 1;
                                        target_index += 1;
                                    }
                                }
                            }
                        } else {
                            // Retain one target cursor, but gallop across gaps rather than
                            // scanning the larger side. This protects highly skewed joins where
                            // the smaller sorted side has only a few widely spaced keys.
                            let mut target_cursor = 0;
                            for leader_index in leader_range {
                                let key = smaller_keys.value_at(leader_index);
                                if seek_sorted_key(
                                    key,
                                    larger_keys.len(),
                                    &mut target_cursor,
                                    |key_index| larger_keys.value_at(key_index),
                                ) {
                                    updates.push_binding(*var, key);
                                    smaller
                                        .sorted_match_at(leader_index)
                                        .refine(smaller_scan.atom, &mut updates);
                                    larger
                                        .sorted_match_at(target_cursor)
                                        .refine(larger_scan.atom, &mut updates);
                                    updates.finish_frame();
                                    if updates.frames() >= chunk_size {
                                        drain_updates!(updates);
                                    }
                                    target_cursor += 1;
                                } else if target_cursor == larger_keys.len() {
                                    break;
                                }
                            }
                        }
                    } else {
                        // Catalog hash indexes and unsupported shapes retain the scalar path.
                        for_each_leader!(smaller, |val, small_sub| {
                            if let Some(large_sub) = larger.get_subset(val) {
                                updates.push_binding(*var, val[0]);
                                small_sub.refine(smaller_scan.atom, &mut updates);
                                large_sub.refine(larger_scan.atom, &mut updates);
                                updates.finish_frame();
                                if updates.frames() >= chunk_size {
                                    drain_updates!(updates);
                                }
                            }
                        });
                    }
                    drain_updates!(updates, false);

                    binding_info.move_back(a.atom, a_prober);
                    binding_info.move_back(b.atom, b_prober);
                }
                rest => {
                    let mut smallest = 0;
                    let mut smallest_size = usize::MAX;
                    let mut probers = Vec::with_capacity(rest.len());
                    for (i, (scan, prepared_slot)) in rest.iter().zip(prepared_indexes).enumerate()
                    {
                        let tail = atom_tail_use(
                            scan.atom,
                            &stages.instrs,
                            prepared,
                            remaining_after_current,
                            instr_order,
                            cur + 1,
                        );
                        let prober = self.get_index(
                            atoms,
                            binding_info,
                            ProbeRequest::column(
                                scan,
                                tail.keep_rows,
                                tail.child_shape,
                                prepared.resolve(prepared_slot),
                            ),
                        );
                        let size = prober.len();
                        if size < smallest_size {
                            smallest = i;
                            smallest_size = size;
                        }
                        probers.push(prober);
                    }

                    let main_spec = &rest[smallest];

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
                                    sub.refine(scan.atom, &mut updates);
                                } else {
                                    updates.rollback();
                                    // Empty intersection.
                                    return;
                                }
                            }
                            sub.refine(main_spec.atom, &mut updates);
                            updates.finish_frame();
                            if updates.frames() >= chunk_size {
                                drain_updates!(updates);
                            }
                        });
                        drain_updates!(updates, false);
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
                    let cover_subset = cover_node.subset();

                    let proj =
                        SmallVec::<[ColumnId; 4]>::from_iter(bind.iter().map(|(col, _)| *col));
                    let vars = bind.iter().map(|(_, var)| *var).collect();
                    let mut buf = TaggedRowBuffer::new_inline(bind.len());
                    let (scan_start, scan_size) =
                        cover_scan_bounds(top_partition, cover_subset.size());
                    table.scan_project(
                        cover_subset,
                        &proj,
                        scan_start,
                        scan_size,
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
                    drain_updates!(updates, false);
                    binding_info.binding_sets.pop();
                    binding_info.move_back_node(cover_atom, cover_node);
                } else {
                    let keep_cover = atom_tail_use(
                        cover_atom,
                        &stages.instrs,
                        prepared,
                        remaining_after_current,
                        instr_order,
                        cur + 1,
                    )
                    .keep_rows;
                    let proj =
                        SmallVec::<[ColumnId; 4]>::from_iter(bind.iter().map(|(col, _)| *col));
                    let cover_node = binding_info.unwrap_val(cover_atom);
                    let cover_subset = cover_node.subset();
                    let (mut offset, mut scan_remaining) =
                        cover_scan_bounds(top_partition, cover_subset.size());
                    let mut buffer = TaggedRowBuffer::new(bind.len());
                    let mut updates = FrameUpdates::with_capacity(cmp::min(chunk_size, cur_size));
                    while scan_remaining != 0 {
                        buffer.clear();
                        let table = &self.db.tables[atoms[cover_atom].table].table;
                        let scan_size = cmp::min(chunk_size, scan_remaining);
                        let next = table.scan_project(
                            cover_subset,
                            &proj,
                            offset,
                            scan_size,
                            &cover.constraints,
                            &mut buffer,
                        );
                        scan_remaining -= scan_size;
                        for (row, key) in buffer.iter() {
                            if keep_cover {
                                updates.refine_atom_dense(
                                    cover_atom,
                                    OffsetRange::new(row, row.inc()),
                                );
                            }
                            // bind the values
                            for (i, (_, var)) in bind.iter().enumerate() {
                                updates.push_binding(*var, key[i]);
                            }
                            updates.finish_frame();
                            if updates.frames() >= chunk_size {
                                drain_updates!(updates);
                            }
                        }
                        if scan_remaining != 0 {
                            let Some(next) = next else {
                                break;
                            };
                            offset = next;
                        }
                    }
                    drain_updates!(updates, false);
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
                let keep_cover = atom_tail_use(
                    cover_atom,
                    &stages.instrs,
                    prepared,
                    remaining_after_current,
                    instr_order,
                    cur + 1,
                )
                .keep_rows;
                if binding_info.has_empty_subset(cover_atom) {
                    return;
                }
                let index_probers = to_intersect
                    .iter()
                    .enumerate()
                    .map(|(i, (spec, _))| {
                        let tail = atom_tail_use(
                            spec.to_index.atom,
                            &stages.instrs,
                            prepared,
                            remaining_after_current,
                            instr_order,
                            cur + 1,
                        );
                        (
                            i,
                            spec.to_index.atom,
                            self.get_index(
                                atoms,
                                binding_info,
                                ProbeRequest::tuple(
                                    spec,
                                    tail.keep_rows,
                                    tail.child_shape,
                                    prepared.resolve(&prepared_indexes[i]),
                                ),
                            ),
                        )
                    })
                    .collect::<SmallVec<[(usize, AtomId, Prober<'_, 'rows, 'exec>); 4]>>();
                let proj = SmallVec::<[ColumnId; 4]>::from_iter(bind.iter().map(|(col, _)| *col));
                let cover_node = binding_info.unwrap_val(cover_atom);
                let cover_subset = cover_node.subset();
                let (mut scan_offset, mut scan_remaining) =
                    cover_scan_bounds(top_partition, cover_subset.size());
                let mut buffer = TaggedRowBuffer::new(bind.len());
                let mut updates = FrameUpdates::with_capacity(cmp::min(chunk_size, cur_size));
                while scan_remaining != 0 {
                    buffer.clear();
                    let table = &self.db.tables[atoms[cover_atom].table].table;
                    let scan_size = cmp::min(chunk_size, scan_remaining);
                    let next = table.scan_project(
                        cover_subset,
                        &proj,
                        scan_offset,
                        scan_size,
                        &cover.constraints,
                        &mut buffer,
                    );
                    scan_remaining -= scan_size;
                    'mid: for (row, key) in buffer.iter() {
                        if keep_cover {
                            updates.refine_atom_dense(cover_atom, OffsetRange::new(row, row.inc()));
                        }
                        // bind the values
                        for (i, (_, var)) in bind.iter().enumerate() {
                            updates.push_binding(*var, key[i]);
                        }
                        // now probe each remaining indexes
                        for (i, atom, prober) in &index_probers {
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
                            subset.refine(*atom, &mut updates);
                        }
                        updates.finish_frame();
                        if updates.frames() >= chunk_size {
                            drain_updates!(updates);
                        }
                    }
                    if scan_remaining != 0 {
                        let Some(next) = next else {
                            break;
                        };
                        scan_offset = next;
                    }
                }
                // TODO: special-case the scenario when the cover doesn't need
                // deduping (and hence we can do a straight scan: e.g. when the
                // cover is binding a superset of the primary key for the
                // table).
                drain_updates!(updates, false);
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
                let keep_for_tail =
                    materialization_is_live_in_tail(&stages.instrs, instr_order, cur + 1, *cover);
                let restore_materialization = !keep_for_tail;
                let cover_mat = if keep_for_tail {
                    Arc::clone(&binding_info.materializations[*cover])
                } else {
                    binding_info.materializations.unwrap_val(*cover)
                };
                (|| {
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
                    drain_updates!(updates, false);
                    binding_info.binding_sets.pop();
                })();
                if restore_materialization {
                    let previous = binding_info.materializations.insert(*cover, cover_mat);
                    debug_assert!(previous.is_none());
                }
            }
            JoinStage::FusedIntersectMat {
                cover,
                mode,
                bind,
                to_intersect,
            } => {
                let keep_for_tail =
                    materialization_is_live_in_tail(&stages.instrs, instr_order, cur + 1, *cover);
                let restore_materialization = !keep_for_tail;
                let cover_mat = if keep_for_tail {
                    Arc::clone(&binding_info.materializations[*cover])
                } else {
                    binding_info.materializations.unwrap_val(*cover)
                };
                (|| {
                    let mut updates: FrameUpdates<'rows, 'exec> =
                        FrameUpdates::with_capacity(cmp::min(chunk_size, cur_size));
                    let probers: SmallVec<[Prober<'_, 'rows, 'exec>; 4]> = to_intersect
                        .iter()
                        .zip(prepared_indexes)
                        .map(|((spec, _), prepared_slot)| {
                            let tail = atom_tail_use(
                                spec.to_index.atom,
                                &stages.instrs,
                                prepared,
                                remaining_after_current,
                                instr_order,
                                cur + 1,
                            );
                            self.get_index(
                                atoms,
                                binding_info,
                                ProbeRequest::tuple(
                                    spec,
                                    tail.keep_rows,
                                    tail.child_shape,
                                    prepared.resolve(prepared_slot),
                                ),
                            )
                        })
                        .collect();
                    debug_assert!(to_intersect.iter().enumerate().all(|(i, (spec, _))| {
                        to_intersect[..i]
                            .iter()
                            .all(|(prior, _)| prior.to_index.atom != spec.to_index.atom)
                    }));
                    let mut key = Vec::with_capacity(4);
                    let mut prune_probers = |updates: &mut FrameUpdates<'rows, 'exec>,
                                             mat_key: Option<&[Value]>,
                                             mat_non_key: Option<&[Value]>|
                     -> bool {
                        for ((spec, cols), prober) in to_intersect.iter().zip(probers.iter()) {
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
                                subset.refine(spec.to_index.atom, updates);
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
                                        if prune_probers(
                                            &mut updates,
                                            Some(group_key),
                                            Some(non_keys),
                                        ) {
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

                    drain_updates!(updates, false);
                    for (spec, prober) in to_intersect.iter().zip(probers) {
                        binding_info.move_back(spec.0.to_index.atom, prober);
                    }
                })();
                if restore_materialization {
                    let previous = binding_info.materializations.insert(*cover, cover_mat);
                    debug_assert!(previous.is_none());
                }
            }
        }
    }
}

const LOCAL_ACTION_BATCH_SIZE: usize = 128;
const EXPERIMENTAL_TOP_SHARD_MORSEL_SIZE: usize = 4 * 1024;
const EXPERIMENTAL_COARSE_MIN_MORSEL_SIZE: usize = 256;
const EXPERIMENTAL_TOP_SHARD_POLICY_ENV: &str = "EGGLOG_EXPERIMENTAL_TOP_SHARD_POLICY";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExperimentalTopShardPolicy {
    Serial,
    DemandGlobal,
    DemandLocal,
    CoarseGlobal,
}

impl ExperimentalTopShardPolicy {
    fn parse(value: Option<&str>) -> Result<Self, ()> {
        match value {
            None | Some("serial") => Ok(Self::Serial),
            Some("demand-global") => Ok(Self::DemandGlobal),
            Some("demand-local") => Ok(Self::DemandLocal),
            Some("coarse-global") => Ok(Self::CoarseGlobal),
            Some(_) => Err(()),
        }
    }

    fn split_demand(self, stalled_workers: usize, _local_queue_depth: usize) -> usize {
        match self {
            Self::Serial => 0,
            Self::DemandGlobal | Self::DemandLocal | Self::CoarseGlobal => stalled_workers,
        }
    }

    fn permits_parallel_drain(self, level: usize, partitioned_top_scan: bool) -> bool {
        match self {
            Self::Serial => false,
            Self::DemandGlobal | Self::DemandLocal => !partitioned_top_scan,
            Self::CoarseGlobal => partitioned_top_scan && level == 0,
        }
    }
}

fn coarse_scan_morsel_size(scan_size: usize, workers: usize) -> usize {
    scan_size
        .div_ceil(workers.max(1))
        .max(EXPERIMENTAL_COARSE_MIN_MORSEL_SIZE)
}

fn cached_coarse_scan_morsel_size(
    cached: &mut Option<usize>,
    scan_size: usize,
    workers: usize,
) -> usize {
    *cached.get_or_insert_with(|| coarse_scan_morsel_size(scan_size, workers))
}

fn experimental_top_shard_policy() -> ExperimentalTopShardPolicy {
    static POLICY: OnceLock<ExperimentalTopShardPolicy> = OnceLock::new();
    *POLICY.get_or_init(|| match std::env::var(EXPERIMENTAL_TOP_SHARD_POLICY_ENV) {
        Ok(value) => ExperimentalTopShardPolicy::parse(Some(&value)).unwrap_or_else(|()| {
            panic!(
                "{EXPERIMENTAL_TOP_SHARD_POLICY_ENV} must be one of \
                 serial, demand-global, demand-local, or coarse-global; got {value:?}"
            )
        }),
        Err(std::env::VarError::NotPresent) => ExperimentalTopShardPolicy::Serial,
        Err(std::env::VarError::NotUnicode(value)) => {
            panic!("{EXPERIMENTAL_TOP_SHARD_POLICY_ENV} is not valid UTF-8: {value:?}")
        }
    })
}

/// A trait used to abstract over different ways of buffering actions together
/// before running them.
///
/// This trait exists as a fairly ad-hoc wrapper over its two implementations.
/// It allows us to avoid duplicating the (somewhat monstrous) `run_plan` method
/// for serial and parallel modes.
trait ActionBuffer<'scope, 'exec, A: NumericId>: Send
where
    'exec: 'scope,
{
    type AsLocal<'a>: ActionBuffer<'scope, 'exec, A>
    where
        'scope: 'a;
    type AsGlobalSerial<'a>: ActionBuffer<'scope, 'exec, A>
    where
        'scope: 'a;

    /// Expand the binding sets to individual bindings and
    /// call push_bindings
    fn push_bindings_factorized(
        &mut self,
        action: A,
        bindings: &mut DenseIdMap<Variable, Value>,
        binding_sets: &BindingSet,
        exec_state: ExecutionStateSeed<'scope, '_>,
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
        to_exec_state: impl FnMut() -> ExecutionState<'scope>,
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
    fn recur<'local, 'rows>(
        &mut self,
        local: BorrowedLocalState<'local, 'rows, 'exec>,
        subset_clone_plan: SubsetClonePlan<'_>,
        to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'a> FnOnce(BorrowedLocalState<'a, 'scope, 'exec>, &mut Self::AsLocal<'a>)
        + Send
        + 'scope,
    ) where
        'rows: 'scope;

    /// Run one coarse partition as a global pool job, using a buffer whose
    /// recursive work and action execution are both serial.  Implementations
    /// that return `false` from [`Self::supports_global_partition`] execute the
    /// callback inline; the scoped implementations enqueue exactly one job.
    fn recur_global_serial<'local, 'rows>(
        &mut self,
        local: BorrowedLocalState<'local, 'rows, 'exec>,
        subset_clone_plan: SubsetClonePlan<'_>,
        to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'a> FnOnce(BorrowedLocalState<'a, 'scope, 'exec>, &mut Self::AsGlobalSerial<'a>)
        + Send
        + 'scope,
    ) where
        'rows: 'scope;

    /// The unit at which you should batch updates passed to calls to `recur`,
    /// potentially depending on the current level of recursion.
    ///
    /// As of right now this is just a hard-coded value. We may change it in the
    /// future to fan out more at higher levels though.
    fn morsel_size(&mut self, _level: usize, _total: usize, _estimated_scan_size: usize) -> usize {
        256
    }

    /// Whether this buffer supports parallel drain operations.
    ///
    /// When `false`, `drain_updates` will use the serial path even at `cur <= 1`,
    /// avoiding the per-frame `ExecutionState::clone()` overhead.
    fn supports_parallel_drain(&self, _level: usize, _partitioned_top_scan: bool) -> bool {
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

impl<'scope, 'exec, 'outer: 'scope> ActionBuffer<'scope, 'exec, ActionId>
    for InPlaceActionBuffer<'outer>
where
    'exec: 'scope,
{
    type AsLocal<'b>
        = Self
    where
        'scope: 'b;
    type AsGlobalSerial<'b>
        = Self
    where
        'scope: 'b;

    fn push_bindings(
        &mut self,
        action: ActionId,
        bindings: &DenseIdMap<Variable, Value>,
        mut to_exec_state: impl FnMut() -> ExecutionState<'scope>,
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

    fn recur<'local, 'rows>(
        &mut self,
        mut local: BorrowedLocalState<'local, 'rows, 'exec>,
        subset_clone_plan: SubsetClonePlan<'_>,
        _to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'b> FnOnce(BorrowedLocalState<'b, 'scope, 'exec>, &mut Self) + Send + 'scope,
    ) where
        'rows: 'scope,
    {
        let mut inner: LocalState<'scope, 'exec> = local.clone_state(subset_clone_plan);
        work(inner.borrow_mut(), self)
    }

    fn recur_global_serial<'local, 'rows>(
        &mut self,
        mut local: BorrowedLocalState<'local, 'rows, 'exec>,
        subset_clone_plan: SubsetClonePlan<'_>,
        _to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'b> FnOnce(BorrowedLocalState<'b, 'scope, 'exec>, &mut Self) + Send + 'scope,
    ) where
        'rows: 'scope,
    {
        let mut inner: LocalState<'scope, 'exec> = local.clone_state(subset_clone_plan);
        work(inner.borrow_mut(), self)
    }

    fn supports_parallel_drain(&self, _level: usize, _partitioned_top_scan: bool) -> bool {
        false
    }
}

/// Action buffer used inside one globally scheduled top-index shard.
///
/// `Serial` preserves the production behavior exactly. The fine-grained demand
/// policies may hand recursive drains to the global or worker-local queue.
/// `CoarseGlobal` instead groups many top bindings from one cached-index shard
/// into a bounded global job and runs that job's complete subtree serially.
/// Action batches remain local to each spawned child and are flushed before
/// that child exits. The buffer deliberately has no `needs_flush` flag: every
/// owner unconditionally flushes its local batches before returning.
struct TopShardActionBuffer<'inner, 'scope> {
    scope: &'inner Scope<'scope>,
    rule_set: &'scope RuleSet,
    match_counter: Arc<MatchCounter>,
    batches: DenseIdMap<ActionId, ActionState>,
    policy: ExperimentalTopShardPolicy,
    // Fixed after the first demand signal so one shard produces a bounded
    // number of similarly sized jobs instead of a geometric series.
    coarse_morsel_size: Option<usize>,
}

impl<'inner, 'scope> TopShardActionBuffer<'inner, 'scope> {
    fn new(
        scope: &'inner Scope<'scope>,
        rule_set: &'scope RuleSet,
        match_counter: Arc<MatchCounter>,
        policy: ExperimentalTopShardPolicy,
    ) -> Self {
        Self {
            scope,
            rule_set,
            match_counter,
            batches: Default::default(),
            policy,
            coarse_morsel_size: None,
        }
    }

    fn split_demand(&self) -> usize {
        self.policy
            .split_demand(self.scope.stalled_workers(), self.scope.local_queue_depth())
    }
}

impl<'scope, 'exec> ActionBuffer<'scope, 'exec, ActionId> for TopShardActionBuffer<'_, 'scope>
where
    'exec: 'scope,
{
    type AsLocal<'a>
        = TopShardActionBuffer<'a, 'scope>
    where
        'scope: 'a;
    type AsGlobalSerial<'a>
        = TopShardActionBuffer<'a, 'scope>
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

    fn recur<'local, 'rows>(
        &mut self,
        mut local: BorrowedLocalState<'local, 'rows, 'exec>,
        subset_clone_plan: SubsetClonePlan<'_>,
        mut to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'a> FnOnce(
            BorrowedLocalState<'a, 'scope, 'exec>,
            &mut TopShardActionBuffer<'a, 'scope>,
        ) + Send
        + 'scope,
    ) where
        'rows: 'scope,
    {
        // `supports_parallel_drain` made the first advisory stalled-state
        // observation. Recheck immediately before committing to a queue: a
        // sibling shard may have filled the pool in the meantime.
        let should_split = self.split_demand() != 0;
        let mut inner: LocalState<'scope, 'exec> = local.clone_state(subset_clone_plan);
        if !should_split {
            let mut inline = TopShardActionBuffer {
                scope: self.scope,
                rule_set: self.rule_set,
                match_counter: self.match_counter.clone(),
                batches: mem::take(&mut self.batches),
                policy: self.policy,
                coarse_morsel_size: self.coarse_morsel_size,
            };
            work(inner.borrow_mut(), &mut inline);
            self.batches = inline.batches;
            self.coarse_morsel_size = inline.coarse_morsel_size;
            return;
        }

        let rule_set = self.rule_set;
        let match_counter = self.match_counter.clone();
        let policy = self.policy;
        let child_policy = match policy {
            ExperimentalTopShardPolicy::CoarseGlobal => ExperimentalTopShardPolicy::Serial,
            policy => policy,
        };
        let task = move |scope: &Scope<'scope>| {
            let mut buf = TopShardActionBuffer::new(scope, rule_set, match_counter, child_policy);
            work(inner.borrow_mut(), &mut buf);
            buf.flush(&mut to_exec_state());
        };
        match policy {
            ExperimentalTopShardPolicy::Serial => {
                unreachable!("serial top-shard policy cannot request a demand split")
            }
            ExperimentalTopShardPolicy::DemandGlobal | ExperimentalTopShardPolicy::CoarseGlobal => {
                self.scope.spawn_global(task)
            }
            ExperimentalTopShardPolicy::DemandLocal => self.scope.spawn_local(task),
        }
    }

    fn recur_global_serial<'local, 'rows>(
        &mut self,
        mut local: BorrowedLocalState<'local, 'rows, 'exec>,
        subset_clone_plan: SubsetClonePlan<'_>,
        _to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'a> FnOnce(
            BorrowedLocalState<'a, 'scope, 'exec>,
            &mut TopShardActionBuffer<'a, 'scope>,
        ) + Send
        + 'scope,
    ) where
        'rows: 'scope,
    {
        let mut inner: LocalState<'scope, 'exec> = local.clone_state(subset_clone_plan);
        let mut inline = TopShardActionBuffer {
            scope: self.scope,
            rule_set: self.rule_set,
            match_counter: self.match_counter.clone(),
            batches: mem::take(&mut self.batches),
            policy: self.policy,
            coarse_morsel_size: self.coarse_morsel_size,
        };
        work(inner.borrow_mut(), &mut inline);
        self.batches = inline.batches;
        self.coarse_morsel_size = inline.coarse_morsel_size;
    }

    fn supports_parallel_drain(&self, level: usize, partitioned_top_scan: bool) -> bool {
        // This first check avoids state cloning entirely while all workers are
        // occupied. `recur` performs a second advisory check before spawning.
        let eligible = self
            .policy
            .permits_parallel_drain(level, partitioned_top_scan);
        eligible && self.split_demand() != 0
    }

    fn morsel_size(&mut self, level: usize, _total: usize, estimated_scan_size: usize) -> usize {
        match self.policy {
            ExperimentalTopShardPolicy::DemandGlobal | ExperimentalTopShardPolicy::DemandLocal
                if level == 1 =>
            {
                EXPERIMENTAL_TOP_SHARD_MORSEL_SIZE
            }
            ExperimentalTopShardPolicy::CoarseGlobal if level == 0 && self.split_demand() != 0 => {
                let workers = crate::parallel::current_num_threads();
                cached_coarse_scan_morsel_size(
                    &mut self.coarse_morsel_size,
                    estimated_scan_size,
                    workers,
                )
            }
            _ => 256,
        }
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

impl<'scope, 'exec> ActionBuffer<'scope, 'exec, ActionId> for ScopedActionBuffer<'_, 'scope>
where
    'exec: 'scope,
{
    type AsLocal<'a>
        = ScopedActionBuffer<'a, 'scope>
    where
        'scope: 'a;
    type AsGlobalSerial<'a>
        = TopShardActionBuffer<'a, 'scope>
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
    fn recur<'local, 'rows>(
        &mut self,
        mut local: BorrowedLocalState<'local, 'rows, 'exec>,
        subset_clone_plan: SubsetClonePlan<'_>,
        mut to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'a> FnOnce(
            BorrowedLocalState<'a, 'scope, 'exec>,
            &mut ScopedActionBuffer<'a, 'scope>,
        ) + Send
        + 'scope,
    ) where
        'rows: 'scope,
    {
        let rule_set = self.rule_set;
        let match_counter = self.match_counter.clone();
        let mut inner = local.clone_state(subset_clone_plan);
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

    fn recur_global_serial<'local, 'rows>(
        &mut self,
        mut local: BorrowedLocalState<'local, 'rows, 'exec>,
        subset_clone_plan: SubsetClonePlan<'_>,
        mut to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'a> FnOnce(
            BorrowedLocalState<'a, 'scope, 'exec>,
            &mut TopShardActionBuffer<'a, 'scope>,
        ) + Send
        + 'scope,
    ) where
        'rows: 'scope,
    {
        let rule_set = self.rule_set;
        let match_counter = self.match_counter.clone();
        let mut inner = local.clone_state(subset_clone_plan);
        let policy = experimental_top_shard_policy();
        self.scope.spawn_global(move |scope| {
            let mut buf = TopShardActionBuffer::new(scope, rule_set, match_counter, policy);
            work(inner.borrow_mut(), &mut buf);
            // Unlike a nested `ScopedActionBuffer`, this buffer is owned by the
            // one shard job and is always flushed exactly once before it exits.
            buf.flush(&mut to_exec_state());
        });
    }

    fn morsel_size(&mut self, _level: usize, _total: usize, _estimated_scan_size: usize) -> usize {
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

fn expand_binding_sets<'scope, 'exec, A: NumericId, BUF: ActionBuffer<'scope, 'exec, A> + ?Sized>(
    action_buf: &mut BUF,
    action: A,
    bindings: &mut DenseIdMap<Variable, Value>,
    binding_sets: &BindingSet,
    idx: usize,
    exec_state: ExecutionStateSeed<'scope, '_>,
) where
    'exec: 'scope,
{
    if exec_state.should_stop() {
        return;
    }
    if idx >= binding_sets.len() {
        action_buf.push_bindings(action, bindings, || exec_state.to_execution_state());
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
            action_buf.push_bindings(action, bindings, || exec_state.to_execution_state());
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

impl<'scope, 'exec, 'outer: 'scope> ActionBuffer<'scope, 'exec, MatId>
    for InPlaceMaterializer<'outer>
where
    'exec: 'scope,
{
    type AsLocal<'b>
        = Self
    where
        'scope: 'b;
    type AsGlobalSerial<'b>
        = Self
    where
        'scope: 'b;

    fn push_bindings(
        &mut self,
        mat_id: MatId,
        bindings: &DenseIdMap<Variable, Value>,
        _to_exec_state: impl FnMut() -> ExecutionState<'scope>,
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

    fn recur<'local, 'rows>(
        &mut self,
        mut local: BorrowedLocalState<'local, 'rows, 'exec>,
        subset_clone_plan: SubsetClonePlan<'_>,
        _to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'b> FnOnce(BorrowedLocalState<'b, 'scope, 'exec>, &mut Self) + Send + 'scope,
    ) where
        'rows: 'scope,
    {
        let mut inner: LocalState<'scope, 'exec> = local.clone_state(subset_clone_plan);
        work(inner.borrow_mut(), self)
    }

    fn recur_global_serial<'local, 'rows>(
        &mut self,
        mut local: BorrowedLocalState<'local, 'rows, 'exec>,
        subset_clone_plan: SubsetClonePlan<'_>,
        _to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'b> FnOnce(BorrowedLocalState<'b, 'scope, 'exec>, &mut Self) + Send + 'scope,
    ) where
        'rows: 'scope,
    {
        let mut inner: LocalState<'scope, 'exec> = local.clone_state(subset_clone_plan);
        work(inner.borrow_mut(), self)
    }

    fn supports_parallel_drain(&self, _level: usize, _partitioned_top_scan: bool) -> bool {
        false
    }
}

/// Serial recursive view of a scoped materializer. Sibling top-level shards
/// still share the concurrent output maps, but this buffer never schedules
/// deeper work itself. The experimental demand policy is intentionally limited
/// to direct action execution; materialization remains an unchanged
/// serial-per-shard control.
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

impl<'scope, 'exec> ActionBuffer<'scope, 'exec, MatId> for SerialScopedMaterializer
where
    'exec: 'scope,
{
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

    fn recur<'local, 'rows>(
        &mut self,
        mut local: BorrowedLocalState<'local, 'rows, 'exec>,
        subset_clone_plan: SubsetClonePlan<'_>,
        _to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'a> FnOnce(BorrowedLocalState<'a, 'scope, 'exec>, &mut Self) + Send + 'scope,
    ) where
        'rows: 'scope,
    {
        let mut inner: LocalState<'scope, 'exec> = local.clone_state(subset_clone_plan);
        work(inner.borrow_mut(), self);
    }

    fn recur_global_serial<'local, 'rows>(
        &mut self,
        mut local: BorrowedLocalState<'local, 'rows, 'exec>,
        subset_clone_plan: SubsetClonePlan<'_>,
        _to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'a> FnOnce(BorrowedLocalState<'a, 'scope, 'exec>, &mut Self) + Send + 'scope,
    ) where
        'rows: 'scope,
    {
        let mut inner: LocalState<'scope, 'exec> = local.clone_state(subset_clone_plan);
        work(inner.borrow_mut(), self);
    }

    fn supports_parallel_drain(&self, _level: usize, _partitioned_top_scan: bool) -> bool {
        false
    }
}

struct ScopedMaterializer<'inner, 'scope, 'exec> {
    scope: &'inner Scope<'scope>,
    retired_states: &'scope RetiredLocalStates<'scope, 'exec>,
    specs: Arc<DenseIdMap<MatId, MatSpec>>,
    materializations: Arc<DenseIdMap<MatId, Arc<DashMap<Vec<Value>, RowBuffer>>>>,
    scratch_key: Vec<Value>,
    scratch_val: Vec<Value>,
}
impl<'scope, 'exec> ActionBuffer<'scope, 'exec, MatId> for ScopedMaterializer<'_, 'scope, 'exec>
where
    'exec: 'scope,
{
    type AsLocal<'a>
        = ScopedMaterializer<'a, 'scope, 'exec>
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

    fn recur<'local, 'rows>(
        &mut self,
        mut local: BorrowedLocalState<'local, 'rows, 'exec>,
        subset_clone_plan: SubsetClonePlan<'_>,
        _to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'a> FnOnce(
            BorrowedLocalState<'a, 'scope, 'exec>,
            &mut ScopedMaterializer<'a, 'scope, 'exec>,
        ) + Send
        + 'scope,
    ) where
        'rows: 'scope,
    {
        let scope = self.scope;
        let retired_states = self.retired_states;
        let specs = self.specs.clone();
        let materializations = self.materializations.clone();
        let mut inner = local.clone_state(subset_clone_plan);
        // Match the action path: preserve locality by default and rely on
        // private-deque donation when other workers need fallback work.
        scope.spawn_local(move |scope| {
            let mut buf: ScopedMaterializer<'_, 'scope, 'exec> = ScopedMaterializer {
                scope,
                retired_states,
                specs,
                materializations: materializations.clone(),
                scratch_key: Vec::new(),
                scratch_val: Vec::new(),
            };
            work(inner.borrow_mut(), &mut buf);
            retired_states.retire(inner);
        });
    }

    fn recur_global_serial<'local, 'rows>(
        &mut self,
        mut local: BorrowedLocalState<'local, 'rows, 'exec>,
        subset_clone_plan: SubsetClonePlan<'_>,
        _to_exec_state: impl FnMut() -> ExecutionState<'scope> + Send + 'scope,
        work: impl for<'a> FnOnce(BorrowedLocalState<'a, 'scope, 'exec>, &mut SerialScopedMaterializer)
        + Send
        + 'scope,
    ) where
        'rows: 'scope,
    {
        let specs = self.specs.clone();
        let materializations = self.materializations.clone();
        let retired_states = self.retired_states;
        let mut inner = local.clone_state(subset_clone_plan);
        self.scope.spawn_global(move |_| {
            let mut buf = SerialScopedMaterializer::new(specs, materializations);
            work(inner.borrow_mut(), &mut buf);
            retired_states.retire(inner);
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

fn estimate_size(join_stage: &JoinStage, binding_info: &BindingInfo<'_, '_>) -> usize {
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

fn is_reorder_barrier(stage: &JoinStage) -> bool {
    matches!(
        stage,
        JoinStage::FusedIntersectMat {
            mode: MatScanMode::Lookup(_) | MatScanMode::Value(_) | MatScanMode::Full,
            ..
        }
    )
}

/// Reorder a suffix of the cached logical plan using the original DVO policy.
///
/// Within each phase, the greedy key is decreasing refinement count in the
/// logical plan prefix, increasing live residual size, then decreasing number
/// of intersected relations. Materialization stages that do not commute remain
/// fixed barriers. Reordering can also change factorized leaf-scan eligibility,
/// so [`recompute_leaf_scans`] runs after every change.
fn sort_plan_by_size(
    order: &mut InstrOrder,
    leaf_scans: &mut LeafScans,
    start: usize,
    instrs: &[JoinStage],
    binding_info: &mut BindingInfo<'_, '_>,
) {
    let mut last_pos = start;
    for i in start..instrs.len() {
        if is_reorder_barrier(&instrs[i]) {
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
    binding_info: &mut BindingInfo<'_, '_>,
) {
    // Nothing to sort if there's 0 or 1 element.
    if range.len() <= 1 {
        return;
    }
    // How many times an atom has been intersected/joined
    let mut times_refined = with_pool_set(|ps| ps.get::<DenseIdMap<AtomId, i64>>());
    let update_refinements =
        |stage: &JoinStage, refinements: &mut DenseIdMap<AtomId, i64>| match stage {
            JoinStage::Intersect { scans, .. } => scans.iter().for_each(|scan| {
                *refinements.get_or_default(scan.atom) += 1;
            }),
            JoinStage::FusedIntersect {
                cover,
                to_intersect,
                ..
            } => {
                *refinements.get_or_default(cover.to_index.atom) +=
                    cover.to_index.vars.len() as i64;
                to_intersect.iter().for_each(|(spec, _)| {
                    *refinements.get_or_default(spec.to_index.atom) +=
                        spec.to_index.vars.len() as i64;
                });
            }
            JoinStage::FusedIntersectMat { to_intersect, .. } => {
                to_intersect.iter().for_each(|(spec, _)| {
                    *refinements.get_or_default(spec.to_index.atom) +=
                        spec.to_index.vars.len() as i64;
                });
            }
        };

    // Count how many times each atom has been refined in the logical plan
    // prefix, as the original DVO heuristic does.
    for stage in &instrs[..range.start] {
        update_refinements(stage, &mut times_refined);
    }

    // We prioritize stages by
    //
    //   (1) how many times an atom used by the stage has been refined,
    //   (2) then by the estimated input rows (smaller → earlier),
    //   (3) then by how many relations the stage joins (more → earlier).
    //
    // Estimate size is second so that very small inputs (e.g. FunDep
    // consequents with exactly one value) run before multi-relation stages
    // that happen to have a larger current estimate.
    let key_fn = |join_stage: &JoinStage,
                  binding_info: &BindingInfo<'_, '_>,
                  refinements: &DenseIdMap<AtomId, i64>| {
        let refine = match join_stage {
            JoinStage::Intersect { scans, .. } => scans
                .iter()
                .map(|scan| refinements.get(scan.atom).copied().unwrap_or_default())
                .max()
                .unwrap(),
            JoinStage::FusedIntersect { cover, .. } => refinements
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
        update_refinements(&instrs[order.get(i)], &mut times_refined);
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

struct BorrowedLocalState<'a, 'rows, 'exec> {
    instr_order: &'a mut InstrOrder,
    leaf_scans: &'a mut LeafScans,
    binding_info: &'a mut BindingInfo<'rows, 'exec>,
    updates: &'a mut FrameUpdates<'rows, 'exec>,
}

struct SubsetClonePlan<'a> {
    stages: &'a [JoinStage],
    resume_pos: usize,
}

impl<'rows, 'exec> BorrowedLocalState<'_, 'rows, 'exec>
where
    'exec: 'rows,
{
    fn clone_state<'short>(&mut self, plan: SubsetClonePlan<'_>) -> LocalState<'short, 'exec>
    where
        'rows: 'short,
    {
        let binding_info =
            self.binding_info
                .clone_for_join_tail(plan.stages, self.instr_order, plan.resume_pos);
        let updates: FrameUpdates<'short, 'exec> = std::mem::take(self.updates);
        LocalState {
            instr_order: self.instr_order.clone(),
            leaf_scans: self.leaf_scans.clone(),
            binding_info,
            updates,
        }
    }
}

struct LocalState<'rows, 'exec> {
    instr_order: InstrOrder,
    leaf_scans: LeafScans,
    binding_info: BindingInfo<'rows, 'exec>,
    updates: FrameUpdates<'rows, 'exec>,
}

#[derive(Default)]
struct RetiredLocalStates<'rows, 'exec> {
    states: Mutex<Vec<LocalState<'rows, 'exec>>>,
}

impl<'rows, 'exec> RetiredLocalStates<'rows, 'exec> {
    fn retire(&self, state: LocalState<'rows, 'exec>) {
        self.states
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(state);
    }
}

impl<'rows, 'exec> LocalState<'rows, 'exec> {
    fn borrow_mut<'a>(&'a mut self) -> BorrowedLocalState<'a, 'rows, 'exec> {
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
        cell::Cell,
        mem,
        sync::{
            Arc, Barrier,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use egglog_concurrency::SharedArena;
    use smallvec::{SmallVec, smallvec};

    use crate::{
        common::{IndexMap, Value},
        free_join::{
            AtomId, SubAtom, Variable,
            plan::{JoinStage, JoinStages, MatId, MatScanMode, ScanSpec, SingleScanSpec},
        },
        numeric_id::{DenseIdMap, NumericId},
        offsets::Subset,
        row_buffer::RowBuffer,
        table_spec::ColumnId,
    };

    use super::{
        AccessId, AtomRows, BindingInfo, CatalogContinuation, ContinuationPosition, Database,
        ExperimentalTopShardPolicy, InstrOrder, LazyArenaHandle, PreparedIndexKind,
        PreparedIndexSlot, PreparedIndexState, PreparedIndexStateId, PreparedJoinIndexes,
        PreparedTailMasks, ProbeIndex, ProbeMatch, Prober, RootContinuationCache, RootProjection,
        SmallColumnIndex, SmallColumnSink, TOP_INDEX_RANGES_PER_WORKER, TopLevelPartition,
        TrieNode, cached_coarse_scan_morsel_size, coarse_scan_morsel_size, cover_scan_bounds,
        for_each_stage_atom, materialization_is_live_in_tail, packed_child_shape_in_tail,
        scan_atom_tail_use, seek_sorted_key, sort_plan_by_size_inner, top_cover_partitions,
        top_index_range_partitions, top_index_shape_is_eligible,
    };
    use crate::free_join::packed_trie::ChildShape;

    #[test]
    fn cover_only_stages_skip_prepared_index_state() {
        let stages = JoinStages {
            instrs: Arc::new(vec![JoinStage::Intersect {
                var: Variable::from_usize(0),
                scans: SmallVec::new(),
            }]),
        };
        let atoms = Arc::new(DenseIdMap::new());
        assert!(matches!(
            PreparedJoinIndexes::new(&Database::new(), &atoms, &stages),
            PreparedJoinIndexes::NoIndexes
        ));
    }

    #[test]
    fn arena_handle_is_created_only_on_first_packed_allocation() {
        let arena = SharedArena::new();
        let handle = LazyArenaHandle::new(&arena);
        assert!(handle.handle.get().is_none());
        handle.get();
        assert!(handle.handle.get().is_some());
    }

    #[test]
    fn parses_experimental_top_shard_policy_without_mutating_process_env() {
        assert_eq!(
            ExperimentalTopShardPolicy::parse(None),
            Ok(ExperimentalTopShardPolicy::Serial)
        );
        assert_eq!(
            ExperimentalTopShardPolicy::parse(Some("serial")),
            Ok(ExperimentalTopShardPolicy::Serial)
        );
        assert_eq!(
            ExperimentalTopShardPolicy::parse(Some("demand-global")),
            Ok(ExperimentalTopShardPolicy::DemandGlobal)
        );
        assert_eq!(
            ExperimentalTopShardPolicy::parse(Some("demand-local")),
            Ok(ExperimentalTopShardPolicy::DemandLocal)
        );
        assert_eq!(
            ExperimentalTopShardPolicy::parse(Some("coarse-global")),
            Ok(ExperimentalTopShardPolicy::CoarseGlobal)
        );
        assert_eq!(ExperimentalTopShardPolicy::parse(Some("")), Err(()));
        assert_eq!(ExperimentalTopShardPolicy::parse(Some("demand")), Err(()));
    }

    #[test]
    fn experimental_top_shard_policy_splits_only_on_demand() {
        assert_eq!(ExperimentalTopShardPolicy::Serial.split_demand(3, 0), 0);
        assert_eq!(
            ExperimentalTopShardPolicy::DemandGlobal.split_demand(0, 100),
            0
        );
        assert_eq!(
            ExperimentalTopShardPolicy::DemandGlobal.split_demand(3, 100),
            3
        );
        assert_eq!(
            ExperimentalTopShardPolicy::DemandLocal.split_demand(0, 100),
            0
        );
        assert_eq!(
            ExperimentalTopShardPolicy::DemandLocal.split_demand(3, 0),
            3
        );
        assert_eq!(
            ExperimentalTopShardPolicy::DemandLocal.split_demand(3, 100),
            3
        );
        assert_eq!(
            ExperimentalTopShardPolicy::CoarseGlobal.split_demand(0, 100),
            0
        );
        assert_eq!(
            ExperimentalTopShardPolicy::CoarseGlobal.split_demand(3, 100),
            3
        );
    }

    #[test]
    fn coarse_scan_morsels_are_bounded_and_cover_the_scan() {
        assert_eq!(coarse_scan_morsel_size(0, 0), 256);
        assert_eq!(coarse_scan_morsel_size(100, 8), 256);
        assert_eq!(coarse_scan_morsel_size(2048, 8), 256);
        assert_eq!(coarse_scan_morsel_size(2049, 8), 257);
        assert_eq!(coarse_scan_morsel_size(4096, 3), 1366);

        let mut cached = None;
        assert_eq!(cached_coarse_scan_morsel_size(&mut cached, 4096, 3), 1366);
        assert_eq!(cached_coarse_scan_morsel_size(&mut cached, 512, 3), 1366);
    }

    #[test]
    fn coarse_policy_only_partitions_the_cached_top_scan() {
        assert!(ExperimentalTopShardPolicy::DemandGlobal.permits_parallel_drain(1, false));
        assert!(!ExperimentalTopShardPolicy::DemandGlobal.permits_parallel_drain(0, true));
        assert!(ExperimentalTopShardPolicy::CoarseGlobal.permits_parallel_drain(0, true));
        assert!(!ExperimentalTopShardPolicy::CoarseGlobal.permits_parallel_drain(1, false));
        assert!(!ExperimentalTopShardPolicy::CoarseGlobal.permits_parallel_drain(1, true));
    }

    #[test]
    fn continuation_positions_remain_compact() {
        assert_eq!(
            mem::size_of::<ContinuationPosition>(),
            2 * mem::size_of::<u32>()
        );
        assert_eq!(
            mem::size_of::<CatalogContinuation<'_>>(),
            mem::size_of::<&RootContinuationCache>() + mem::size_of::<ContinuationPosition>()
        );
    }

    #[test]
    fn small_column_index_groups_keys_and_sorts_inline_rows() {
        let mut sink = SmallColumnSink::default();
        for (value, row) in [(2, 7), (1, 5), (2, 3), (1, 4), (3, 6)] {
            sink.rows[sink.len] = (Value::from_usize(value), crate::RowId::from_usize(row));
            sink.len += 1;
        }

        let index = SmallColumnIndex::from_projected(sink);
        assert_eq!(index.len(), 3);
        assert_eq!(index.find(Value::from_usize(0)), None);
        assert_eq!(
            index
                .rows_at(index.find(Value::from_usize(1)).unwrap())
                .rows()
                .iter()
                .map(|row| row.index())
                .collect::<Vec<_>>(),
            vec![4, 5]
        );
        assert_eq!(
            index
                .rows_at(index.find(Value::from_usize(2)).unwrap())
                .rows()
                .iter()
                .map(|row| row.index())
                .collect::<Vec<_>>(),
            vec![3, 7]
        );
        assert_eq!(
            index
                .rows_at(index.find(Value::from_usize(3)).unwrap())
                .rows()[0]
                .index(),
            6
        );
    }

    #[test]
    fn sorted_multi_probe_keeps_a_monotone_cursor_across_batches() {
        let target = [1, 2, 4, 8, 9, 20, 21].map(Value::from_usize);
        let first = [0, 2, 3, 8].map(Value::from_usize);
        let second = [9, 10, 20, 22].map(Value::from_usize);
        let mut cursor = 0;
        let mut matches = Vec::new();

        for (input, key) in first.into_iter().enumerate() {
            if seek_sorted_key(key, target.len(), &mut cursor, |index| target[index]) {
                matches.push((input, cursor));
                cursor += 1;
            }
        }
        assert_eq!(matches, vec![(1, 1), (3, 3)]);
        assert_eq!(cursor, 4);

        matches.clear();
        for (input, key) in second.into_iter().enumerate() {
            if seek_sorted_key(key, target.len(), &mut cursor, |index| target[index]) {
                matches.push((input, cursor));
                cursor += 1;
            }
        }
        assert_eq!(matches, vec![(0, 4), (2, 5)]);
        assert_eq!(cursor, target.len());
    }

    #[test]
    fn sorted_multi_probe_gallops_across_skewed_gaps() {
        let target_len = 100_000;
        let keys = [2, 199_998].map(Value::from_usize);
        let calls = Cell::new(0);
        let mut cursor = 0;
        let mut matches = Vec::new();

        for (input, key) in keys.into_iter().enumerate() {
            if seek_sorted_key(key, target_len, &mut cursor, |index| {
                calls.set(calls.get() + 1);
                Value::from_usize(index * 2)
            }) {
                matches.push((input, cursor));
                cursor += 1;
            }
        }

        assert_eq!(matches, vec![(0, 1), (1, 99_999)]);
        assert!(
            calls.get() < 100,
            "galloping should not linearly scan a large key gap"
        );
    }

    #[test]
    fn sorted_seek_reads_an_immediate_hit_once() {
        let target = [2, 4, 6].map(Value::from_usize);
        let calls = Cell::new(0);
        let mut cursor = 0;
        assert!(seek_sorted_key(
            Value::from_usize(2),
            target.len(),
            &mut cursor,
            |index| {
                calls.set(calls.get() + 1);
                target[index]
            }
        ));
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn sorted_match_ordinal_reconstructs_row_carrying_small_probe() {
        let mut sink = SmallColumnSink::default();
        for (value, row) in [(1, 4), (2, 9), (2, 7)] {
            sink.rows[sink.len] = (Value::from_usize(value), crate::RowId::from_usize(row));
            sink.len += 1;
        }
        let index = SmallColumnIndex::from_projected(sink);
        let source = AtomRows::Inline(index.rows_at(0));
        let mut prober = Prober {
            source,
            ix: ProbeIndex::SmallColumn(index),
            keep_rows: true,
        };

        let ProbeMatch::Rows(AtomRows::Inline(rows)) = prober.sorted_match_at(1) else {
            panic!("a row-carrying small probe must reconstruct inline rows")
        };
        assert_eq!(
            rows.rows()
                .iter()
                .map(|row| row.index())
                .collect::<Vec<_>>(),
            vec![7, 9]
        );

        prober.keep_rows = false;
        assert!(matches!(prober.sorted_match_at(1), ProbeMatch::Present));
    }

    #[test]
    fn root_projection_is_a_final_dense_or_sparse_index() {
        let index = RootProjection::from_sorted_pairs(vec![
            (Value::from_usize(1), crate::RowId::from_usize(4)),
            (Value::from_usize(1), crate::RowId::from_usize(5)),
            (Value::from_usize(2), crate::RowId::from_usize(3)),
            (Value::from_usize(2), crate::RowId::from_usize(7)),
            (Value::from_usize(9), crate::RowId::from_usize(11)),
        ]);

        assert_eq!(index.len(), 3);
        assert_eq!(index.rows.len(), 5);
        assert_eq!(
            index
                .keys
                .iter()
                .map(|&(key, offset)| (key.index(), offset))
                .collect::<Vec<_>>(),
            vec![(1, 0), (2, 2), (9, 4), (0, 5)]
        );
        assert_eq!(index.find(Value::from_usize(0)), None);
        assert_eq!(index.find(Value::from_usize(8)), None);

        let dense = index.subset_at(index.find(Value::from_usize(1)).unwrap());
        let crate::SubsetRef::Dense(dense) = dense else {
            panic!("contiguous root-index rows must use a dense subset")
        };
        assert_eq!((dense.start.index(), dense.end.index()), (4, 6));

        let sparse = index.subset_at(index.find(Value::from_usize(2)).unwrap());
        let crate::SubsetRef::Sparse(sparse) = sparse else {
            panic!("noncontiguous root-index rows must use a sparse subset")
        };
        assert_eq!(
            sparse
                .inner()
                .iter()
                .map(|row| row.index())
                .collect::<Vec<_>>(),
            vec![3, 7]
        );

        let singleton = index.subset_at(index.find(Value::from_usize(9)).unwrap());
        let crate::SubsetRef::Dense(singleton) = singleton else {
            panic!("a singleton root-index group must use a dense subset")
        };
        assert_eq!((singleton.start.index(), singleton.end.index()), (11, 12));

        let empty = RootProjection::from_sorted_pairs(Vec::new());
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.rows.len(), 0);
        assert_eq!(
            empty.keys.len(),
            1,
            "empty indexes retain only the sentinel"
        );
        assert_eq!(empty.find(Value::from_usize(0)), None);
    }

    #[test]
    fn shared_root_projection_keys_are_canonical_and_single_flight() {
        let root = Arc::new(TrieNode::new_shared(Subset::Dense(
            crate::OffsetRange::new(crate::RowId::from_usize(0), crate::RowId::from_usize(1)),
        )));
        let lower = crate::Constraint::GtConst {
            col: ColumnId::from_usize(1),
            val: Value::from_usize(10),
        };
        let upper = crate::Constraint::LtConst {
            col: ColumnId::from_usize(1),
            val: Value::from_usize(20),
        };
        let forward = root
            .projection_slot(ColumnId::from_usize(0), &[lower.clone(), upper.clone()])
            .unwrap();
        let reversed = root
            .projection_slot(ColumnId::from_usize(0), &[upper.clone(), lower.clone()])
            .unwrap();
        assert!(Arc::ptr_eq(&forward, &reversed));
        assert!(!Arc::ptr_eq(
            &forward,
            &root
                .projection_slot(ColumnId::from_usize(1), &[lower.clone(), upper.clone()])
                .unwrap()
        ));

        let builds = AtomicUsize::new(0);
        let barrier = Barrier::new(16);
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..16 {
                let root = root.clone();
                let lower = lower.clone();
                let upper = upper.clone();
                let builds = &builds;
                let barrier = &barrier;
                handles.push(scope.spawn(move || {
                    barrier.wait();
                    // Race both the canonicalized DashMap lookup and the lazy
                    // projection publication, as parallel plans do.
                    let slot = root
                        .projection_slot(ColumnId::from_usize(0), &[upper, lower])
                        .unwrap();
                    slot.get_or_init(|| {
                        builds.fetch_add(1, Ordering::Relaxed);
                        RootProjection::from_sorted_pairs(Vec::new())
                    }) as *const RootProjection as usize
                }));
            }
            let addresses = handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>();
            assert!(addresses.windows(2).all(|pair| pair[0] == pair[1]));
        });
        assert_eq!(builds.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn root_continuation_cache_reuses_direct_and_dynamic_slots() {
        let shard_lens = [2, 0, 3];

        let direct = RootContinuationCache::default();
        direct.prepare(ChildShape::Direct, shard_lens.len(), |shard| {
            shard_lens[shard]
        });
        direct.prepare(ChildShape::Direct, shard_lens.len(), |shard| {
            shard_lens[shard]
        });
        assert_eq!(direct.slots(AccessId::new(0)).len(), 3);

        let dynamic = RootContinuationCache::default();
        dynamic.prepare(
            ChildShape::Dynamic { families: 3 },
            shard_lens.len(),
            |shard| shard_lens[shard],
        );

        let barrier = Barrier::new(16);
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..16 {
                let dynamic = &dynamic;
                let barrier = &barrier;
                handles.push(scope.spawn(move || {
                    barrier.wait();
                    dynamic.slots(AccessId::new(1)).as_ptr() as usize
                }));
            }
            let addresses = handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>();
            assert!(addresses.windows(2).all(|pair| pair[0] == pair[1]));
        });

        assert!(!std::ptr::eq(
            dynamic.slots(AccessId::new(1)),
            dynamic.slots(AccessId::new(2)),
        ));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "root continuation shape changed after initialization")]
    fn root_continuation_prepare_revalidates_initialized_shape() {
        let cache = RootContinuationCache::default();
        cache.prepare(ChildShape::Direct, 1, |_| 1);
        cache.prepare(ChildShape::Dynamic { families: 2 }, 1, |_| 1);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "direct root continuation was used by multiple indexed accesses")]
    fn direct_root_continuation_rejects_different_successors() {
        let cache = RootContinuationCache::default();
        cache.prepare(ChildShape::Direct, 1, |_| 1);
        let _ = cache.slots(AccessId::new(0));
        let _ = cache.slots(AccessId::new(1));
    }

    fn scan(atom: usize) -> ScanSpec {
        ScanSpec {
            to_index: SubAtom {
                atom: AtomId::from_usize(atom),
                vars: smallvec![ColumnId::from_usize(0)],
            },
            constraints: Vec::new(),
        }
    }

    fn mat_stage(mat_id: usize) -> JoinStage {
        JoinStage::FusedIntersectMat {
            cover: MatId::from_usize(mat_id),
            mode: MatScanMode::KeyOnly,
            bind: SmallVec::new(),
            to_intersect: Vec::new(),
        }
    }

    fn intersect_stage(atom: usize, column: usize) -> JoinStage {
        JoinStage::Intersect {
            var: Variable::from_usize(column),
            scans: smallvec![SingleScanSpec {
                atom: AtomId::from_usize(atom),
                column: ColumnId::from_usize(column),
                cs: Vec::new(),
            }],
        }
    }

    #[test]
    fn mixed_recursive_dvo_keeps_the_plan_prefix_as_its_refinement_anchor() {
        let stages = vec![
            intersect_stage(0, 0),
            intersect_stage(1, 0),
            intersect_stage(1, 1),
            mat_stage(0),
        ];
        let mut binding_info = BindingInfo::default();
        for atom in 0..2 {
            binding_info.insert_subset(
                AtomId::from_usize(atom),
                Subset::Dense(crate::OffsetRange::new(
                    crate::RowId::from_usize(0),
                    crate::RowId::from_usize(100),
                )),
            );
        }

        // Stage 1 happened to run first in this branch. The stable plan prefix
        // still anchors the recursive ordering to stage 0 / atom 0. Using the
        // physical prefix here would instead promote stage 2 / atom 1.
        let mut order = InstrOrder::from_iter([1, 0, 2, 3].into_iter());
        sort_plan_by_size_inner(&mut order, 1..3, &stages, &mut binding_info);

        assert_eq!(order.data.as_slice(), &[1, 0, 2, 3]);
    }

    fn prepared_for(stages: &[JoinStage]) -> PreparedJoinIndexes {
        let mut access_counts = crate::numeric_id::DenseIdMap::new();
        let mut states = Vec::new();
        let prepared_stages: Box<[SmallVec<[PreparedIndexSlot; 4]>]> = stages
            .iter()
            .map(|stage| {
                let atoms = match stage {
                    JoinStage::Intersect { scans, .. } => {
                        scans.iter().map(|scan| scan.atom).collect::<Vec<_>>()
                    }
                    JoinStage::FusedIntersect { to_intersect, .. }
                    | JoinStage::FusedIntersectMat { to_intersect, .. } => to_intersect
                        .iter()
                        .map(|(scan, _)| scan.to_index.atom)
                        .collect(),
                };
                atoms
                    .into_iter()
                    .map(|atom| {
                        let next = access_counts.get_or_default(atom);
                        let access = AccessId::new(*next);
                        *next += 1;
                        let kind = PreparedIndexKind::Uncacheable;
                        let state = PreparedIndexStateId::new(states.len());
                        states.push(PreparedIndexState::new(kind));
                        PreparedIndexSlot::new(kind, access, state)
                    })
                    .collect()
            })
            .collect();
        let tail_masks = PreparedTailMasks::new(stages, &prepared_stages, access_counts.n_ids());
        PreparedJoinIndexes::Indexed {
            stages: prepared_stages,
            states: states.into_boxed_slice(),
            access_counts,
            tail_masks,
        }
    }

    fn permutations(values: &mut [usize], start: usize, result: &mut Vec<Vec<usize>>) {
        if start == values.len() {
            result.push(values.to_vec());
            return;
        }
        for index in start..values.len() {
            values.swap(start, index);
            permutations(values, start + 1, result);
            values.swap(start, index);
        }
    }

    #[test]
    fn prepared_tail_masks_match_scanner_for_every_permutation_and_suffix() {
        let stages = vec![
            intersect_stage(0, 0),
            JoinStage::Intersect {
                var: Variable::from_usize(1),
                scans: smallvec![
                    SingleScanSpec {
                        atom: AtomId::from_usize(0),
                        column: ColumnId::from_usize(1),
                        cs: Vec::new(),
                    },
                    SingleScanSpec {
                        atom: AtomId::from_usize(1),
                        column: ColumnId::from_usize(0),
                        cs: Vec::new(),
                    }
                ],
            },
            intersect_stage(1, 1),
            intersect_stage(2, 0),
        ];
        let prepared = prepared_for(&stages);
        let masks = prepared.tail_masks().unwrap();
        let mut orders = Vec::new();
        permutations(&mut [0, 1, 2, 3], 0, &mut orders);
        for order in orders {
            let instr_order = InstrOrder::from_iter(order.iter().copied());
            for resume_pos in 0..=order.len() {
                let remaining = order[resume_pos..]
                    .iter()
                    .fold(0u64, |mask, &stage| mask | (1u64 << stage));
                for atom_index in 0..=3 {
                    let atom = AtomId::from_usize(atom_index);
                    assert_eq!(
                        masks.atom_tail_use(atom, remaining, prepared.access_count(atom)),
                        scan_atom_tail_use(atom, &stages, &prepared, &instr_order, resume_pos,),
                        "tail metadata diverged for order {order:?}, suffix {resume_pos}, atom {atom_index}"
                    );
                }
            }
        }
    }

    #[test]
    fn prepared_tail_masks_use_u64_boundary_and_fallback_after_it() {
        let stages_64 = (0..64)
            .map(|column| intersect_stage(0, column))
            .collect::<Vec<_>>();
        let prepared_64 = prepared_for(&stages_64);
        assert_eq!(prepared_64.tail_masks().unwrap().all_stages, u64::MAX);

        let stages_65 = (0..65)
            .map(|column| intersect_stage(0, column))
            .collect::<Vec<_>>();
        assert!(prepared_for(&stages_65).tail_masks().is_none());
    }

    #[test]
    fn packed_tail_shape_preserves_direct_graph_path() {
        let stages = vec![intersect_stage(0, 0), intersect_stage(0, 1)];
        let prepared = prepared_for(&stages);
        let order = InstrOrder::from_iter(0..stages.len());

        assert_eq!(
            packed_child_shape_in_tail(AtomId::from_usize(0), &stages, &prepared, &order, 1,),
            ChildShape::Direct
        );
        assert_eq!(
            packed_child_shape_in_tail(AtomId::from_usize(0), &stages, &prepared, &order, 2,),
            ChildShape::Leaf
        );
    }

    #[test]
    fn packed_tail_shape_uses_dynamic_families_for_dvo_choice() {
        let stages = vec![
            intersect_stage(0, 0),
            intersect_stage(0, 1),
            intersect_stage(0, 2),
        ];
        let prepared = prepared_for(&stages);
        let order = InstrOrder::from_iter([0, 2, 1].into_iter());

        assert_eq!(
            packed_child_shape_in_tail(AtomId::from_usize(0), &stages, &prepared, &order, 1,),
            ChildShape::Dynamic { families: 3 }
        );
    }

    #[test]
    fn packed_tail_shape_stops_at_cover_and_reorder_barriers() {
        let atom = AtomId::from_usize(0);
        let stages = vec![
            intersect_stage(0, 0),
            JoinStage::FusedIntersect {
                cover: scan(0),
                bind: SmallVec::new(),
                to_intersect: Vec::new(),
            },
            JoinStage::FusedIntersectMat {
                cover: MatId::from_usize(0),
                mode: MatScanMode::Full,
                bind: SmallVec::new(),
                to_intersect: Vec::new(),
            },
            intersect_stage(0, 1),
        ];
        let prepared = prepared_for(&stages);
        let order = InstrOrder::from_iter(0..stages.len());
        assert_eq!(
            packed_child_shape_in_tail(atom, &stages, &prepared, &order, 1),
            ChildShape::Leaf,
            "the cover consumes the packed residual before the later phase"
        );

        let stages = vec![
            intersect_stage(0, 0),
            JoinStage::FusedIntersectMat {
                cover: MatId::from_usize(0),
                mode: MatScanMode::Full,
                bind: SmallVec::new(),
                to_intersect: vec![(scan(0), SmallVec::new())],
            },
            intersect_stage(0, 2),
        ];
        let prepared = prepared_for(&stages);
        let order = InstrOrder::from_iter(0..stages.len());
        assert_eq!(
            packed_child_shape_in_tail(atom, &stages, &prepared, &order, 1),
            ChildShape::Direct,
            "a singleton barrier hides indexed accesses in later phases"
        );
    }

    #[test]
    fn top_index_partitioning_rejects_serial_tiny_and_skewed_shapes() {
        assert!(!top_index_shape_is_eligible(1, 10_000, 8, 64));
        assert!(!top_index_shape_is_eligible(4, 255, 8, 64));
        assert!(!top_index_shape_is_eligible(4, 10_000, 3, 64));
        assert!(top_index_shape_is_eligible(4, 256, 4, 64));
        assert!(top_index_shape_is_eligible(4, 40, 4, 10));
    }

    #[test]
    fn fused_cover_partitions_are_complete_disjoint_and_coarse() {
        assert!(top_cover_partitions(10_000, 1, 16).is_none());
        assert!(top_cover_partitions(63, 4, 16).is_none());

        for (cover_rows, workers) in [(64, 4), (1_003, 4), (585_276, 12)] {
            let partitions = top_cover_partitions(cover_rows, workers, 16).unwrap();
            assert!(partitions.len() <= workers * 4);

            let mut expected_start = 0;
            for partition in &partitions {
                let TopLevelPartition::CoverRange { start, scan_size } = *partition else {
                    panic!("fused covers must produce ordinal ranges")
                };
                assert_eq!(start, expected_start, "ranges must be contiguous");
                assert!(scan_size > 0);
                if start + scan_size != cover_rows {
                    assert!(
                        scan_size >= 16,
                        "all non-tail partitions must remain coarse"
                    );
                }
                expected_start += scan_size;
            }
            assert_eq!(expected_start, cover_rows, "ranges must cover every row");
        }
    }

    #[test]
    fn fused_cover_scan_bounds_preserve_nonzero_subset_origins() {
        let partition = TopLevelPartition::CoverRange {
            start: 17,
            scan_size: 23,
        };
        let (start, len) = cover_scan_bounds(Some(partition), 100);
        assert_eq!(start.index(), 17);
        assert_eq!(len, 23);

        let (start, len) = cover_scan_bounds(None, 100);
        assert_eq!(start.index(), 0);
        assert_eq!(len, 100);
    }

    #[test]
    fn filtered_index_ranges_are_complete_disjoint_and_coarse() {
        assert!(top_index_range_partitions(10_000, 1, 16).is_none());
        assert!(top_index_range_partitions(63, 4, 16).is_none());

        for (leader_keys, workers) in [(64, 4), (1_003, 4), (585_276, 12)] {
            let partitions = top_index_range_partitions(leader_keys, workers, 16).unwrap();
            assert!(partitions.len() <= workers * TOP_INDEX_RANGES_PER_WORKER);

            let mut expected_start = 0;
            for partition in &partitions {
                let TopLevelPartition::IndexRange { start, scan_size } = *partition else {
                    panic!("an unsharded index must produce key-ordinal ranges")
                };
                assert_eq!(start, expected_start, "ranges must be contiguous");
                assert!(scan_size > 0);
                if start + scan_size != leader_keys {
                    assert!(
                        scan_size >= 16,
                        "all non-tail partitions must remain coarse"
                    );
                }
                expected_start += scan_size;
            }
            assert_eq!(expected_start, leader_keys, "ranges must cover every key");
        }
    }

    #[test]
    fn task_clone_keeps_only_atoms_in_the_dynamic_join_tail() {
        let stages = vec![
            JoinStage::Intersect {
                var: Variable::from_usize(0),
                scans: smallvec![SingleScanSpec {
                    atom: AtomId::from_usize(0),
                    column: ColumnId::from_usize(0),
                    cs: Vec::new(),
                }],
            },
            JoinStage::FusedIntersect {
                cover: scan(1),
                bind: SmallVec::new(),
                // Repeat the cover atom to verify that it is cloned once.
                to_intersect: vec![(scan(2), SmallVec::new()), (scan(1), SmallVec::new())],
            },
            JoinStage::FusedIntersectMat {
                cover: MatId::from_usize(0),
                mode: MatScanMode::Full,
                bind: SmallVec::new(),
                to_intersect: vec![(scan(3), SmallVec::new())],
            },
        ];
        // The physical tail is stages 0 and 1, not the lexical suffix 1 and 2.
        let order = InstrOrder::from_iter([2, 0, 1].into_iter());

        let nodes = (0..4)
            .map(|_| Arc::new(TrieNode::new(Subset::empty())))
            .collect::<Vec<_>>();
        let mut source = BindingInfo::default();
        for (atom, node) in nodes.iter().enumerate() {
            source.insert_node(AtomId::from_usize(atom), Arc::clone(node));
        }
        let materializations = (0..2)
            .map(|_| Arc::new(IndexMap::<Vec<Value>, RowBuffer>::default()))
            .collect::<Vec<_>>();
        for (mat_id, materialization) in materializations.iter().enumerate() {
            source
                .materializations
                .insert(MatId::from_usize(mat_id), Arc::clone(materialization));
        }

        let child = source.clone_for_join_tail(&stages, &order, 1);
        for (atom, node) in nodes.iter().enumerate().take(3) {
            let cloned = child.subsets.get(AtomId::from_usize(atom)).unwrap();
            assert!(Arc::ptr_eq(cloned.root_arc(), node));
            assert_eq!(Arc::strong_count(node), 3);
        }
        assert!(!child.subsets.contains_key(AtomId::from_usize(3)));
        assert!(child.materializations.is_empty());
        assert_eq!(Arc::strong_count(&nodes[3]), 2);
        assert!(!materialization_is_live_in_tail(
            &stages,
            &order,
            1,
            MatId::from_usize(0)
        ));
        drop(child);
        assert!(nodes.iter().all(|node| Arc::strong_count(node) == 2));

        // Top-level partition jobs resume at zero and therefore retain the
        // driver stage as well as the rest of the dynamically ordered plan.
        let top = source.clone_for_join_tail(&stages, &order, 0);
        assert!((0..4).all(|atom| top.subsets.contains_key(AtomId::from_usize(atom))));
        assert!(top.materializations.contains_key(MatId::from_usize(0)));
        assert!(!top.materializations.contains_key(MatId::from_usize(1)));
        assert!(materialization_is_live_in_tail(
            &stages,
            &order,
            0,
            MatId::from_usize(0)
        ));
        assert_eq!(Arc::strong_count(&materializations[0]), 3);
        assert_eq!(Arc::strong_count(&materializations[1]), 2);

        // Exercise the exhaustive dependency visitor directly: materialized
        // covers are MatIds, so only their atom probes are reported.
        let mut mat_atoms = Vec::new();
        for_each_stage_atom(&stages[2], |atom| mat_atoms.push(atom));
        assert_eq!(mat_atoms, vec![AtomId::from_usize(3)]);
    }

    #[test]
    fn task_clone_keeps_each_live_materialization_once_in_dynamic_order() {
        let stages = vec![mat_stage(0), mat_stage(1), mat_stage(0)];
        // The dynamic tail after the first stage contains Mat0 twice, while
        // lexical stage 1 (Mat1) has already executed.
        let order = InstrOrder::from_iter([1, 0, 2].into_iter());
        let materializations = (0..2)
            .map(|_| Arc::new(IndexMap::<Vec<Value>, RowBuffer>::default()))
            .collect::<Vec<_>>();
        let mut source = BindingInfo::default();
        for (mat_id, materialization) in materializations.iter().enumerate() {
            source
                .materializations
                .insert(MatId::from_usize(mat_id), Arc::clone(materialization));
        }

        let child = source.clone_for_join_tail(&stages, &order, 1);
        assert!(child.materializations.contains_key(MatId::from_usize(0)));
        assert!(!child.materializations.contains_key(MatId::from_usize(1)));
        assert_eq!(Arc::strong_count(&materializations[0]), 3);
        assert_eq!(Arc::strong_count(&materializations[1]), 2);
        assert!(materialization_is_live_in_tail(
            &stages,
            &order,
            1,
            MatId::from_usize(0)
        ));
        assert!(!materialization_is_live_in_tail(
            &stages,
            &order,
            1,
            MatId::from_usize(1)
        ));
    }
}
