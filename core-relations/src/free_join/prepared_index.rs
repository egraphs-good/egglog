//! Prepared index state for one execution of a logical plan.
//!
//! Start with `PreparedJoinIndexes::new` for the structure of this file: it
//! walks a `JoinStages` block, aligns slots with its indexed scans, assigns
//! per-atom `AccessId`s, and derives the join-tail metadata. Then read
//! `PreparedIndexSlot` for the state associated with one indexed access and
//! `RootContinuationCache` for how a root lookup continues on another column.
//! `PreparedPlanIndexes::new` assembles these per-block structures for a whole
//! plan. Preparation does not build the indexes; `execute.rs` acquires their
//! handles lazily when an access first needs them.

use std::sync::{Arc, OnceLock};

use smallvec::SmallVec;

use crate::{
    Constraint,
    hash_index::{ColumnIndex, Index, IndexPosition, TupleIndex},
    numeric_id::{DenseIdMap, NumericId, define_id},
    query::Atom,
    table_spec::ColumnId,
};

use super::{
    AtomId, Database, HashColumnIndex, HashIndex, TableInfo, get_column_index_from_tableinfo,
    get_index_from_tableinfo,
    join_tail::{
        AtomTailUse, for_each_stage_atom, for_each_stage_indexed_access, is_reorder_barrier,
    },
    packed_cache::{RootProjection, RootProjectionSlot, TrieRoot},
    packed_trie::ChildShape,
    plan::{JoinStage, JoinStages, Plan},
};

define_id!(
    pub(super) AccessId,
    u32,
    r#"Dense identity of one indexed access to an atom within a single
[`JoinStages`] block. Access ids are dense and local to each atom.

For example, the query `T(x, y, z), X(x), Y(y), Z(z)` may produce this
simplified stage sequence:

- intersect `x` through `T[0]` and `X[0]`;
- intersect `y` through `T[1]` and `Y[0]`;
- intersect `z` through `T[2]` and `Z[0]`.

The three accesses to the `T` atom receive ids 0, 1, and 2. Each unary atom
has its own access-id namespace and therefore receives id 0. A dynamic packed
node uses these ids to distinguish the successor families that DVO may choose
after reaching that atom."#
);

/// Identifies where a root-index result stores the packed index used to
/// continue a join.
///
/// A root lookup initially returns all rows having one key. If a later join
/// stage needs to index another column of those same rows, execution builds a
/// packed trie node for that narrower operation. That node is the
/// *continuation* of the root lookup. For example, after looking up `x = 5` in
/// a root index for `R(x, y, z)`, a later access to `R.y` continues from that
/// result by building an index of `y` over only the matching `R` rows.
///
/// Each root key gets a slot that publishes this packed child once and shares
/// it with concurrent probes. Persistent catalog indexes identify the key with
/// an [`IndexPosition`]; unsharded, round-local projections use a key ordinal.
/// This execution-local position represents both forms without confusing it
/// with the exact persistent-index identity documented by [`IndexPosition`].
#[derive(Clone, Copy)]
pub(super) struct ContinuationPosition {
    shard: u32,
    slot: u32,
}

impl ContinuationPosition {
    pub(super) fn unsharded(slot: usize) -> Self {
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

/// Publication slots that map each root-index key to its arena-allocated
/// packed continuation.
///
/// The boxes do not hold trie rows or trie nodes: every initialized
/// [`OnceLock`] contains the erased address of a [`super::packed_trie::TrieNode`] allocated
/// in the query's [`egglog_concurrency::SharedArena`]. This grid is mutable synchronization
/// metadata owned by the prepared-index sidecar. Keeping it heap-owned avoids
/// erasing another arena lifetime merely to store the locks and lets Rust drop
/// their structure normally with the query.
///
/// The nested shape mirrors the persistent index: one outer allocation plus
/// one allocation per physical shard, with no box per key. It therefore maps
/// a [`ContinuationPosition`] to a slot without a prefix-sum lookup. Dynamic
/// ordering adds one lazy grid per successor family, and allocates only a
/// family that is actually selected. Slots are deliberately per logical scan:
/// another plan may continue the same root key using different columns or
/// constraints.
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

pub(super) struct RootContinuationCache {
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

    pub(super) fn prepare(
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

    pub(super) fn slot(
        &self,
        position: ContinuationPosition,
        access: AccessId,
    ) -> &OnceLock<usize> {
        let slots = self.slots(access);
        &slots[position.shard as usize][position.slot as usize]
    }
}

/// The persistent-index strategy available to one logical join access.
///
/// This is part of the compact descriptor stored with a prepared stage. The
/// corresponding [`PreparedIndexState`] owns the large, lazily initialized
/// cache objects used during execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PreparedIndexKind {
    /// A persistent catalog index over two or more columns.
    Tuple,
    /// A persistent catalog index over one column.
    Column,
    /// The table specification forbids a global cache for at least one key
    /// column, so execution must use its existing dynamic-index path.
    Uncacheable,
}

define_id!(
    pub(super) PreparedIndexStateId,
    u32,
    "a dense index into a join block's execution-local prepared-index states"
);

/// Lazily acquired persistent catalog index for one prepared access.
///
/// Keeping the table-owned index handle in execution-local state avoids
/// repeated catalog lookups and reference-count traffic in recursive join
/// execution. `Uncacheable` records that this access must use a round-local
/// packed index instead. All initialized handles are dropped before
/// `merge_all` resets the database's indexes.
enum PreparedIndexCache {
    Tuple(OnceLock<HashIndex>),
    Column(OnceLock<HashColumnIndex>),
    Uncacheable,
}

/// Execution-local mutable state for one prepared index access. Keeping these
/// large cache objects out of the stage `SmallVec`s leaves their inline entries
/// as compact, copyable descriptors.
pub(super) struct PreparedIndexState {
    /// Persistent tuple/column index handle, acquired only if execution chooses
    /// the catalog path for this access.
    cache: PreparedIndexCache,
    /// Per-root-key publication slots for packed indexes that continue this
    /// access on another column of the same atom.
    pub(super) root_continuations: RootContinuationCache,
    /// Shared scalar projection selected for this logical access. Once set, the
    /// retained `Arc` keeps its immutable key and row arrays alive for the
    /// entire query, so output frames can borrow them without cloning the Arc.
    projected_root: OnceLock<RootProjectionSlot>,
    /// Erased arena address of the packed root for this logical scan.
    /// This remains the fallback for roots that are not shared across plans.
    pub(super) packed_root: OnceLock<usize>,
}

impl PreparedIndexState {
    pub(super) fn new(kind: PreparedIndexKind) -> Self {
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

/// Compact descriptor for one indexed access in a prepared logical stage.
///
/// Stage descriptors are copied frequently while the executor walks or
/// reorders the plan, so they contain only ids and an index strategy. The
/// associated locks, cached indexes, and continuation grids live separately in
/// the [`PreparedIndexState`] array and are reached through `state`.
#[derive(Clone, Copy, Debug)]
pub(super) struct PreparedIndexSlot {
    /// Whether this access can use a tuple catalog, a column catalog, or no
    /// persistent catalog at all.
    kind: PreparedIndexKind,
    /// Dense identity of this access among accesses to the same atom. Packed
    /// dynamic children use it to distinguish possible successor families.
    pub(super) access: AccessId,
    /// Position of this access's mutable cache state in `PreparedJoinIndexes`.
    state: PreparedIndexStateId,
}

impl PreparedIndexSlot {
    pub(super) fn new(
        kind: PreparedIndexKind,
        access: AccessId,
        state: PreparedIndexStateId,
    ) -> Self {
        Self {
            kind,
            access,
            state,
        }
    }
}

/// Borrowed execution view obtained by resolving a compact
/// [`PreparedIndexSlot`] against its separately stored mutable state.
///
/// Its methods retain catalog handles and shared projections in
/// [`PreparedIndexState`]. Returned borrows live as long as that state, so the
/// executor can copy or discard this view without shortening those borrows.
#[derive(Clone, Copy)]
pub(super) struct PreparedIndexRef<'a> {
    /// Persistent-index strategy copied from the stage descriptor.
    pub(super) kind: PreparedIndexKind,
    /// Per-atom successor-family identity copied from the stage descriptor.
    pub(super) access: AccessId,
    /// Locks and cache handles retained for this access's query execution.
    pub(super) state: &'a PreparedIndexState,
}

impl<'a> PreparedIndexRef<'a> {
    /// Acquire this access's single-column catalog index on first use.
    ///
    /// The catalog helper refreshes the index before its handle is retained.
    /// Later calls borrow the same index without another catalog lookup or
    /// `Arc` clone. `info` and `column` must identify the same logical access
    /// on every call; the returned borrow is tied to this execution's state.
    /// Panics if this access was not prepared for a column catalog index.
    pub(super) fn column_index(self, info: &TableInfo, column: ColumnId) -> &'a Index<ColumnIndex> {
        debug_assert_eq!(self.kind, PreparedIndexKind::Column);
        let PreparedIndexCache::Column(index) = &self.state.cache else {
            unreachable!("single-column scan must have a prepared column index")
        };
        index
            .get_or_init(|| get_column_index_from_tableinfo(info, column))
            .get()
            .expect("prepared column index must already be refreshed")
    }

    /// Acquire this access's multi-column catalog index on first use.
    ///
    /// The catalog helper refreshes the index before its handle is retained.
    /// Later calls borrow the same index without another catalog lookup or
    /// `Arc` clone. `info` and the ordered `columns` must identify the same
    /// logical access on every call; the borrow is tied to this execution's
    /// state. Panics if this access was not prepared for a tuple catalog index.
    pub(super) fn tuple_index(
        self,
        info: &TableInfo,
        columns: &[ColumnId],
    ) -> &'a Index<TupleIndex> {
        debug_assert_eq!(self.kind, PreparedIndexKind::Tuple);
        let PreparedIndexCache::Tuple(index) = &self.state.cache else {
            unreachable!("multi-column scan must have a prepared tuple index")
        };
        index
            .get_or_init(|| get_index_from_tableinfo(info, columns))
            .get()
            .expect("prepared tuple index must already be refreshed")
    }

    /// Retain this access's shared root projection, initializing it with `build`.
    ///
    /// `root`, `column`, and `slow_constraints` must identify the same logical
    /// projection on every call. Fast constraints already selected the root's
    /// rows. The retained slot avoids repeated map lookups and `Arc` clones;
    /// its shared `OnceLock` coordinates construction across concurrent plans.
    /// `build` runs only if the shared projection needs initialization.
    /// Returns `None` without calling `build` if the root has no shared cache.
    /// The returned projection is borrowed from this execution's retained state.
    pub(super) fn get_or_init_root_projection(
        self,
        root: &TrieRoot,
        column: ColumnId,
        slow_constraints: &[Constraint],
        build: impl FnOnce() -> RootProjection,
    ) -> Option<&'a RootProjection> {
        let slot = if let Some(slot) = self.state.projected_root.get() {
            slot
        } else {
            let candidate = root.projection_slot(column, slow_constraints)?;
            self.state.projected_root.get_or_init(|| candidate)
        };
        Some(slot.get_or_init(build))
    }
}

pub(super) fn columns_are_cacheable(info: &TableInfo, cols: &[ColumnId]) -> bool {
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
    /// Stages that read or refine this atom, including cover-only accesses.
    touched_stages: u64,
    /// Stages containing exactly one indexed access to this atom.
    one_index_access_stages: u64,
    /// Stages containing multiple indexed accesses to this atom.
    multiple_index_access_stages: u64,
}

/// Order-independent tail metadata for plans small enough to represent their
/// remaining stages in one word. DVO only permutes stages within the fixed
/// barrier phases, so successor shape depends on the remaining set, not its
/// current permutation.
pub(super) struct PreparedTailMasks {
    /// Per-atom stage classifications used to decide whether rows must survive
    /// and whether the next packed child has a direct or dynamic shape.
    atom_uses: DenseIdMap<AtomId, PreparedAtomUse>,
    /// Ordered reorder phases. Reorderable stages share a mask; every cover or
    /// materialization barrier occupies a singleton mask so DVO cannot move an
    /// access across it.
    phase_masks: SmallVec<[u64; 4]>,
    /// Initial remaining-stage mask for join-tail execution.
    all_stages: u64,
}

impl PreparedTailMasks {
    pub(super) fn new(
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

    pub(super) fn atom_tail_use(
        &self,
        atom: AtomId,
        remaining_stages: u64,
        families: usize,
    ) -> AtomTailUse {
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
///
/// Start with [`Self::new`] to see how the per-access slots, access identities,
/// and tail metadata fit together.
pub(super) enum PreparedJoinIndexes {
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
    pub(super) fn new(
        db: &Database,
        atoms: &Arc<DenseIdMap<AtomId, Atom>>,
        stages: &JoinStages,
    ) -> Self {
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
            let access = AccessId::from_usize(*next);
            *next += 1;
            let info = &db.tables[atoms[atom].table];
            let kind = if !columns_are_cacheable(info, cols) {
                PreparedIndexKind::Uncacheable
            } else if cols.len() == 1 {
                PreparedIndexKind::Column
            } else {
                PreparedIndexKind::Tuple
            };
            let state = PreparedIndexStateId::from_usize(states.len());
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

    pub(super) fn stage(&self, index: usize) -> &[PreparedIndexSlot] {
        match self {
            Self::NoIndexes => &[],
            Self::Indexed { stages, .. } => &stages[index],
        }
    }

    pub(super) fn access_count(&self, atom: AtomId) -> usize {
        match self {
            Self::NoIndexes => 0,
            Self::Indexed { access_counts, .. } => {
                access_counts.get(atom).copied().unwrap_or_default()
            }
        }
    }

    pub(super) fn resolve<'a>(&'a self, slot: &PreparedIndexSlot) -> PreparedIndexRef<'a> {
        let Self::Indexed { states, .. } = self else {
            unreachable!("an index slot cannot belong to a block without indexes")
        };
        PreparedIndexRef {
            kind: slot.kind,
            access: slot.access,
            state: &states[slot.state.index()],
        }
    }

    pub(super) fn all_stage_mask(&self) -> Option<u64> {
        self.tail_masks().map(|masks| masks.all_stages)
    }

    pub(super) fn tail_masks(&self) -> Option<&PreparedTailMasks> {
        match self {
            Self::NoIndexes => None,
            Self::Indexed { tail_masks, .. } => tail_masks.as_ref(),
        }
    }
}

/// Execution-scoped index sidecar mirroring the shape of a logical [`Plan`].
pub(super) enum PreparedPlanIndexes {
    Single(PreparedJoinIndexes),
    Decomposed {
        blocks: Vec<PreparedJoinIndexes>,
        result: PreparedJoinIndexes,
    },
}

impl PreparedPlanIndexes {
    pub(super) fn new(db: &Database, plan: &Plan) -> Self {
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

#[cfg(test)]
#[path = "prepared_index_tests.rs"]
mod tests;
