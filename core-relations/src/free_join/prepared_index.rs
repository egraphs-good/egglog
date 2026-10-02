//! Cached logical index layout and execution-local index state.
//!
//! `PreparedJoinLayout` walks a `JoinStages` block, aligns slots with its
//! indexed scans, assigns per-atom `AccessId`s, and derives join-tail metadata.
//! This immutable analysis is shared by cached-plan clones. Then read
//! `PreparedIndexSlot` for the state associated with one indexed access and
//! `RootContinuationCache` for how a root lookup continues on another column.
//! `PreparedPlanIndexes::new` assembles these per-block structures for a whole
//! plan, with fresh mutable state on every execution. Preparation does not
//! build the indexes; `execute.rs` acquires their handles lazily on first use.

use std::{
    fmt,
    ops::{BitAnd, BitOr, BitOrAssign, Not},
    sync::{Arc, OnceLock},
};

use smallvec::SmallVec;

use crate::{
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
    packed_cache::{AccessFamilies, FamilyId, TrieRoot},
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
/// an [`IndexPosition`]. Only its shard and slot are needed here: the
/// execution-local catalog handle already fixes the index identity.
#[derive(Clone, Copy)]
pub(super) struct ContinuationPosition {
    shard: u32,
    slot: u32,
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
/// in the run's [`egglog_concurrency::SharedArena`]. This grid is mutable synchronization
/// metadata owned by the prepared-index sidecar or, below a shared root, by
/// the run's trie cache. Keeping it heap-owned avoids erasing another arena
/// lifetime merely to store the locks and lets Rust drop their structure
/// normally with its owner.
///
/// The nested shape mirrors the persistent index: one outer allocation plus
/// one allocation per physical shard, with no box per key. It therefore maps
/// a [`ContinuationPosition`] to a slot without a prefix-sum lookup. Dynamic
/// ordering adds one lazy grid per successor family, and allocates only a
/// family that is actually selected. A plan-local grid keys its families by
/// [`AccessId`]; a grid shared below a shared root keys them by
/// [`FamilyId`], so plans continuing the same root key
/// with different columns or constraints use different families.
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
    /// Whether the published nodes are shared across plans and therefore keyed
    /// by run-global successor families.
    shared: bool,
    #[cfg(debug_assertions)]
    direct_family: OnceLock<usize>,
}

impl Default for RootContinuationCache {
    fn default() -> Self {
        Self {
            storage: OnceLock::new(),
            shared: false,
            #[cfg(debug_assertions)]
            direct_family: OnceLock::new(),
        }
    }
}

impl RootContinuationCache {
    /// A grid shared by every plan of the run, keyed by `FamilyId`.
    pub(super) fn shared() -> Self {
        Self {
            shared: true,
            ..Self::default()
        }
    }

    pub(super) fn is_shared(&self) -> bool {
        self.shared
    }

    fn allocate_slots(shard_lens: &[usize]) -> RootContinuationSlots {
        shard_lens
            .iter()
            .map(|&len| std::iter::repeat_with(OnceLock::new).take(len).collect())
            .collect()
    }

    #[inline]
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

    /// The slot grid of one successor `family`: an [`AccessId`] index for a
    /// plan-local grid, or a `FamilyId` index for a shared one.
    fn slots(&self, family: usize) -> &RootContinuationSlots {
        let storage = self
            .storage
            .get()
            .expect("root continuations must be prepared before probing");
        #[cfg(debug_assertions)]
        if matches!(storage, RootContinuationStorage::Direct { .. }) {
            let expected = self.direct_family.get_or_init(|| family);
            debug_assert_eq!(
                *expected, family,
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
            } => families[family].get_or_init(|| Self::allocate_slots(shard_lens)),
        }
    }

    #[inline]
    pub(super) fn slot(&self, position: ContinuationPosition, family: usize) -> &OnceLock<usize> {
        let slots = self.slots(family);
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
/// execution. A root uses either a persistent catalog index or a round-local
/// packed index. All initialized handles are dropped before `merge_all` resets
/// the database's indexes.
enum PreparedRootIndex {
    Tuple(HashIndex),
    Column(HashColumnIndex),
    Packed(usize),
}

/// An access has one plan root and chooses one index representation for it.
/// Descendant cursors use their own packed slots. Continuations below that
/// root are either shared with other plans or owned by this access.
pub(super) struct PreparedIndexState {
    root: OnceLock<PreparedRootIndex>,
    continuations: OnceLock<Arc<RootContinuationCache>>,
}

impl PreparedIndexState {
    pub(super) fn new() -> Self {
        Self {
            root: OnceLock::new(),
            continuations: OnceLock::new(),
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
/// Its methods retain catalog handles in
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
    /// Successor family of each requested column when this access descends
    /// from a shared node; see [`AccessFamilies`]. Empty if the plan was not
    /// built through `Database::plan_query`.
    pub(super) families: &'a [FamilyId],
}

impl<'a> PreparedIndexRef<'a> {
    pub(super) fn packed_root(self, build: impl FnOnce() -> usize) -> usize {
        let PreparedRootIndex::Packed(address) = self
            .state
            .root
            .get_or_init(|| PreparedRootIndex::Packed(build()))
        else {
            unreachable!("a root cannot change its index representation during execution")
        };
        *address
    }

    pub(super) fn local_continuations(self) -> &'a RootContinuationCache {
        self.state
            .continuations
            .get_or_init(|| Arc::new(RootContinuationCache::default()))
    }

    /// Acquire this access's single-column catalog index on first use.
    ///
    /// The catalog helper refreshes the index before its handle is retained.
    /// Later calls borrow the same index without another catalog lookup or
    /// `Arc` clone. `info` and `column` must identify the same logical access
    /// on every call; the returned borrow is tied to this execution's state.
    /// Panics if this access was not prepared for a column catalog index.
    #[inline]
    pub(super) fn column_index(self, info: &TableInfo, column: ColumnId) -> &'a Index<ColumnIndex> {
        debug_assert_eq!(self.kind, PreparedIndexKind::Column);
        let PreparedRootIndex::Column(index) = self.state.root.get_or_init(|| {
            PreparedRootIndex::Column(get_column_index_from_tableinfo(info, column))
        }) else {
            unreachable!("a root cannot change its index representation during execution")
        };
        index
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
    #[inline]
    pub(super) fn tuple_index(
        self,
        info: &TableInfo,
        columns: &[ColumnId],
    ) -> &'a Index<TupleIndex> {
        debug_assert_eq!(self.kind, PreparedIndexKind::Tuple);
        let PreparedRootIndex::Tuple(index) = self
            .state
            .root
            .get_or_init(|| PreparedRootIndex::Tuple(get_index_from_tableinfo(info, columns)))
        else {
            unreachable!("a root cannot change its index representation during execution")
        };
        index
            .get()
            .expect("prepared tuple index must already be refreshed")
    }

    /// Retain the shared continuation grid of the persistent catalog index
    /// over `columns` below the shared `root`, or `None` if the root is not
    /// shared. `root` and `columns` must identify the same index on every call.
    #[inline]
    pub(super) fn shared_catalog_continuations(
        self,
        root: &TrieRoot,
        columns: &[ColumnId],
    ) -> Option<&'a RootContinuationCache> {
        if !root.is_shared() {
            return None;
        }
        if let Some(cache) = self.state.continuations.get() {
            return Some(cache);
        }
        let candidate = root.catalog_continuations(columns)?;
        Some(self.state.continuations.get_or_init(|| candidate))
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

/// Bit set of logical stage indexes, used to track the unexecuted suffix of a
/// free-join plan. Join execution is monomorphized per width so plans that fit
/// in 64 bits keep single-word masks on the hot path.
pub(super) trait StageMask:
    Copy
    + Eq
    + fmt::Debug
    + Send
    + Sync
    + 'static
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitOrAssign
    + Not<Output = Self>
{
    const BITS: u32;
    const EMPTY: Self;
    fn stage_bit(stage: usize) -> Self;
    fn count_ones(self) -> u32;
    /// The tail masks prepared at this width, if the plan was prepared at it.
    fn tail_masks(width: &PreparedTailMaskWidth) -> Option<&PreparedTailMasks<Self>>;
}

macro_rules! impl_stage_mask {
    ($ty:ty, $variant:ident) => {
        impl StageMask for $ty {
            const BITS: u32 = <$ty>::BITS;
            const EMPTY: Self = 0;
            fn stage_bit(stage: usize) -> Self {
                1 << stage
            }
            fn count_ones(self) -> u32 {
                <$ty>::count_ones(self)
            }
            fn tail_masks(width: &PreparedTailMaskWidth) -> Option<&PreparedTailMasks<Self>> {
                match width {
                    PreparedTailMaskWidth::$variant(masks) => Some(masks),
                    _ => None,
                }
            }
        }
    };
}

impl_stage_mask!(u64, Narrow);
impl_stage_mask!(u128, Wide);

/// Tail masks prepared at the narrowest [`StageMask`] width that fits the plan.
#[derive(Debug)]
pub(super) enum PreparedTailMaskWidth {
    /// The plan has more than 128 stages; callers scan the suffix instead.
    None,
    Narrow(PreparedTailMasks<u64>),
    Wide(PreparedTailMasks<u128>),
}

impl PreparedTailMaskWidth {
    pub(super) fn new(
        stages: &[JoinStage],
        prepared_stages: &[SmallVec<[PreparedIndexSlot; 4]>],
        atom_capacity: usize,
    ) -> Self {
        if stages.len() <= u64::BITS as usize {
            Self::Narrow(PreparedTailMasks::new(
                stages,
                prepared_stages,
                atom_capacity,
            ))
        } else if stages.len() <= u128::BITS as usize {
            Self::Wide(PreparedTailMasks::new(
                stages,
                prepared_stages,
                atom_capacity,
            ))
        } else {
            log::debug!(
                "free-join plan with {} stages exceeds the {}-stage limit of prepared tail \
                 masks; join-tail metadata will be computed by scanning the remaining stages",
                stages.len(),
                u128::BITS
            );
            Self::None
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PreparedAtomUse<M> {
    /// Stages that read or refine this atom, including cover-only accesses.
    touched_stages: M,
    /// Stages containing exactly one indexed access to this atom.
    one_index_access_stages: M,
    /// Stages containing multiple indexed accesses to this atom.
    multiple_index_access_stages: M,
    families: usize,
}

impl<M: StageMask> Default for PreparedAtomUse<M> {
    fn default() -> Self {
        Self {
            touched_stages: M::EMPTY,
            one_index_access_stages: M::EMPTY,
            multiple_index_access_stages: M::EMPTY,
            families: 0,
        }
    }
}

/// Order-independent tail metadata for plans small enough to represent their
/// remaining stages in one `M`. DVO only permutes stages within the fixed
/// barrier phases, so successor shape depends on the remaining set, not its
/// current permutation.
#[derive(Debug)]
pub(super) struct PreparedTailMasks<M> {
    /// Per-atom stage classifications used to decide whether rows must survive
    /// and whether the next packed child has a direct or dynamic shape.
    atom_uses: Vec<PreparedAtomUse<M>>,
    /// Ordered reorder phases. Reorderable stages share a mask; every cover or
    /// materialization barrier occupies a singleton mask so DVO cannot move an
    /// access across it.
    phase_masks: SmallVec<[M; 4]>,
    /// Initial remaining-stage mask for join-tail execution.
    all_stages: M,
}

impl<M: StageMask> PreparedTailMasks<M> {
    pub(super) fn new(
        stages: &[JoinStage],
        prepared_stages: &[SmallVec<[PreparedIndexSlot; 4]>],
        atom_capacity: usize,
    ) -> Self {
        assert!(
            stages.len() <= M::BITS as usize,
            "prepared tail masks cannot represent {} stages in {} bits",
            stages.len(),
            M::BITS
        );
        let mut atom_uses = vec![PreparedAtomUse::<M>::default(); atom_capacity];
        for (stage_index, (stage, prepared_stage)) in stages.iter().zip(prepared_stages).enumerate()
        {
            let stage_bit = M::stage_bit(stage_index);
            for_each_stage_atom(stage, |atom| {
                if atom.index() >= atom_uses.len() {
                    atom_uses.resize(atom.index() + 1, PreparedAtomUse::default());
                }
                atom_uses[atom.index()].touched_stages |= stage_bit;
            });

            let mut indexed_counts = SmallVec::<[(AtomId, u8); 4]>::new();
            for_each_stage_indexed_access(stage, prepared_stage, |atom, _| {
                atom_uses[atom.index()].families += 1;
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
                let use_ = &mut atom_uses[atom.index()];
                if count == 1 {
                    use_.one_index_access_stages |= stage_bit;
                } else {
                    use_.multiple_index_access_stages |= stage_bit;
                }
            }
        }

        let mut phase_masks = SmallVec::<[M; 4]>::new();
        let mut reorderable_phase = M::EMPTY;
        let mut all_stages = M::EMPTY;
        for (stage_index, stage) in stages.iter().enumerate() {
            let stage_bit = M::stage_bit(stage_index);
            all_stages |= stage_bit;
            if is_reorder_barrier(stage) {
                if reorderable_phase != M::EMPTY {
                    phase_masks.push(reorderable_phase);
                    reorderable_phase = M::EMPTY;
                }
                phase_masks.push(stage_bit);
            } else {
                reorderable_phase |= stage_bit;
            }
        }
        if reorderable_phase != M::EMPTY {
            phase_masks.push(reorderable_phase);
        }

        Self {
            atom_uses,
            phase_masks,
            all_stages,
        }
    }

    pub(super) fn atom_tail_use(&self, atom: AtomId, remaining_stages: M) -> AtomTailUse {
        let use_ = self
            .atom_uses
            .get(atom.index())
            .copied()
            .unwrap_or_default();
        if remaining_stages & use_.touched_stages == M::EMPTY {
            return AtomTailUse {
                keep_rows: false,
                child_shape: ChildShape::Leaf,
            };
        }

        for &phase in &self.phase_masks {
            let live = remaining_stages & phase;
            if live & use_.touched_stages == M::EMPTY {
                continue;
            }
            let single_accesses = live & use_.one_index_access_stages;
            let multiple_accesses = live & use_.multiple_index_access_stages != M::EMPTY
                || single_accesses.count_ones() > 1;
            let child_shape = if multiple_accesses {
                ChildShape::Dynamic {
                    families: use_.families,
                }
            } else if single_accesses != M::EMPTY {
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
pub(super) enum PreparedJoinIndexes<'plan> {
    /// A block made entirely of cover scans cannot build a packed node or use
    /// an index. Avoid constructing any index sidecar for these blocks; unary
    /// rules hit this path especially often.
    NoIndexes,
    Indexed {
        layout: &'plan PreparedJoinLayout,
        states: Box<[PreparedIndexState]>,
        families: &'plan [AccessFamilies],
    },
}

/// Immutable access identities and tail analysis belong to the cached logical
/// plan. Only index handles and arena addresses must be rebuilt each run.
#[derive(Debug)]
pub(super) struct PreparedJoinLayout {
    pub(super) stages: Box<[SmallVec<[PreparedIndexSlot; 4]>]>,
    pub(super) state_count: usize,
    pub(super) access_counts: DenseIdMap<AtomId, usize>,
    pub(super) tail_masks: PreparedTailMaskWidth,
}

impl PreparedJoinLayout {
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
            return Self {
                stages: Box::new([]),
                state_count: 0,
                access_counts: DenseIdMap::new(),
                tail_masks: PreparedTailMaskWidth::None,
            };
        }

        let mut access_counts = DenseIdMap::with_capacity(atoms.n_ids());
        let mut state_count = 0;
        let mut prepared_stages = Vec::with_capacity(stages.instrs.len());
        // Slots are assigned in `for_each_indexed_access` order, which is also
        // the order of `stages.families`.
        for stage in stages.instrs.iter() {
            let mut handles = SmallVec::new();
            let mut make_slot = |atom: AtomId, cols: &[ColumnId]| {
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
                let state_id = PreparedIndexStateId::from_usize(state_count);
                state_count += 1;
                PreparedIndexSlot::new(kind, access, state_id)
            };
            match stage {
                JoinStage::Intersect { scans, .. } => {
                    handles.extend(
                        scans
                            .iter()
                            .map(|scan| make_slot(scan.atom, std::slice::from_ref(&scan.column))),
                    );
                }
                JoinStage::FusedIntersect { to_intersect, .. }
                | JoinStage::FusedIntersectMat { to_intersect, .. } => {
                    handles.extend(to_intersect.iter().map(|(scan, _)| {
                        make_slot(scan.to_index.atom, scan.to_index.vars.as_slice())
                    }));
                }
            }
            prepared_stages.push(handles);
        }
        let tail_masks =
            PreparedTailMaskWidth::new(&stages.instrs, &prepared_stages, atoms.n_ids());
        Self {
            stages: prepared_stages.into_boxed_slice(),
            state_count,
            access_counts,
            tail_masks,
        }
    }
}

impl<'plan> PreparedJoinIndexes<'plan> {
    pub(super) fn new(
        db: &Database,
        atoms: &Arc<DenseIdMap<AtomId, Atom>>,
        stages: &'plan JoinStages,
    ) -> Self {
        let layout = stages.prepared_layout.get_or_init(|| {
            let layout = PreparedJoinLayout::new(db, atoms, stages);
            (layout.state_count != 0).then(|| Box::new(layout))
        });
        match layout {
            Some(layout) => Self::from_layout(layout, &stages.families),
            None => Self::NoIndexes,
        }
    }

    pub(super) fn from_layout(
        layout: &'plan PreparedJoinLayout,
        families: &'plan [AccessFamilies],
    ) -> Self {
        if layout.state_count == 0 {
            return Self::NoIndexes;
        }
        Self::Indexed {
            layout,
            states: std::iter::repeat_with(PreparedIndexState::new)
                .take(layout.state_count)
                .collect(),
            families,
        }
    }

    pub(super) fn stage(&self, index: usize) -> &[PreparedIndexSlot] {
        match self {
            Self::NoIndexes => &[],
            Self::Indexed { layout, .. } => &layout.stages[index],
        }
    }

    pub(super) fn access_count(&self, atom: AtomId) -> usize {
        match self {
            Self::NoIndexes => 0,
            Self::Indexed { layout, .. } => {
                layout.access_counts.get(atom).copied().unwrap_or_default()
            }
        }
    }

    // Resolve directly into the probe request: an outlined call introduces a
    // stack temporary for this borrowed view on the hot recursive path.
    #[inline(always)]
    pub(super) fn resolve<'a>(&'a self, slot: &PreparedIndexSlot) -> PreparedIndexRef<'a> {
        let Self::Indexed {
            states, families, ..
        } = self
        else {
            unreachable!("an index slot cannot belong to a block without indexes")
        };
        PreparedIndexRef {
            kind: slot.kind,
            access: slot.access,
            state: &states[slot.state.index()],
            families: families
                .get(slot.state.index())
                .map_or(&[], |families| families.as_slice()),
        }
    }

    /// Whether the plan needs 128-bit stage masks; callers choose the width to
    /// run the join at from this.
    pub(super) fn uses_wide_stage_mask(&self) -> bool {
        matches!(self, Self::Indexed { layout, .. }
            if matches!(layout.tail_masks, PreparedTailMaskWidth::Wide(_)))
    }

    pub(super) fn all_stage_mask<M: StageMask>(&self) -> Option<M> {
        self.tail_masks::<M>().map(|masks| masks.all_stages)
    }

    pub(super) fn tail_masks<M: StageMask>(&self) -> Option<&PreparedTailMasks<M>> {
        match self {
            Self::NoIndexes => None,
            Self::Indexed { layout, .. } => M::tail_masks(&layout.tail_masks),
        }
    }
}

/// Execution-scoped index sidecar mirroring the shape of a logical [`Plan`].
pub(super) enum PreparedPlanIndexes<'plan> {
    Single(PreparedJoinIndexes<'plan>),
    Decomposed {
        blocks: Vec<PreparedJoinIndexes<'plan>>,
        result: PreparedJoinIndexes<'plan>,
    },
}

impl<'plan> PreparedPlanIndexes<'plan> {
    pub(super) fn new(db: &Database, plan: &'plan Plan) -> Self {
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
