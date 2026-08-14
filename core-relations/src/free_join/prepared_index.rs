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

/// Publication slots that map each root-index key to its arena-allocated
/// packed continuation.
///
/// The boxes do not hold trie rows or trie nodes: every initialized
/// [`OnceLock`] contains the erased address of a [`PackedTrieNode`] allocated
/// in the query's [`SharedArena`]. This grid is mutable synchronization
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
    Direct(RootContinuationSlots),
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
                ChildShape::Direct => {
                    RootContinuationStorage::Direct(Self::allocate_slots(&shard_lens))
                }
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
                (ChildShape::Direct, RootContinuationStorage::Direct(_)) => true,
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
        if matches!(storage, RootContinuationStorage::Direct(_)) {
            let expected = self.direct_access.get_or_init(|| access);
            debug_assert_eq!(
                *expected, access,
                "a direct root continuation was used by multiple indexed accesses"
            );
        }
        match storage {
            RootContinuationStorage::Direct(slots) => slots,
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
enum PreparedIndexKind {
    Tuple(OnceLock<HashIndex>),
    Column(OnceLock<HashColumnIndex>),
    /// The table specification forbids a global cache for at least one key
    /// column, so execution must use its existing dynamic-index path.
    Uncacheable,
}

struct PreparedIndexSlot {
    kind: PreparedIndexKind,
    access: AccessId,
    root_continuations: RootContinuationCache,
    /// Handle to a shared, final-form root index for this logical access.
    /// Keeping the Arc in the prepared sidecar lets probers borrow its arrays
    /// for the whole query without cloning an Arc into every output frame.
    projected_root: OnceLock<RootProjectionSlot>,
    /// Erased arena address of the packed root for this logical scan.
    /// This remains the fallback for roots that are not shared across plans.
    packed_root: OnceLock<usize>,
}

impl PreparedIndexSlot {
    fn new(kind: PreparedIndexKind, access: AccessId) -> Self {
        Self {
            kind,
            access,
            root_continuations: RootContinuationCache::default(),
            projected_root: OnceLock::new(),
            packed_root: OnceLock::new(),
        }
    }
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
struct PreparedTailMasks {
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
struct PreparedJoinIndexes {
    stages: Box<[SmallVec<[PreparedIndexSlot; 4]>]>,
    access_counts: DenseIdMap<AtomId, usize>,
    tail_masks: Option<PreparedTailMasks>,
}

impl PreparedJoinIndexes {
    fn new(db: &Database, atoms: &Arc<DenseIdMap<AtomId, Atom>>, stages: &JoinStages) -> Self {
        fn make_slot(
            db: &Database,
            atoms: &DenseIdMap<AtomId, Atom>,
            access_counts: &mut DenseIdMap<AtomId, usize>,
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
                PreparedIndexKind::Column(OnceLock::new())
            } else {
                PreparedIndexKind::Tuple(OnceLock::new())
            };
            PreparedIndexSlot::new(kind, access)
        }

        let mut access_counts = DenseIdMap::with_capacity(atoms.n_ids());
        let mut prepared_stages = Vec::with_capacity(stages.instrs.len());
        for stage in stages.instrs.iter() {
            let mut handles = SmallVec::new();
            match stage {
                JoinStage::Intersect { scans, .. } => {
                    handles.extend(scans.iter().map(|scan| {
                        make_slot(
                            db,
                            atoms,
                            &mut access_counts,
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
                            scan.to_index.atom,
                            scan.to_index.vars.as_slice(),
                        )
                    }));
                }
            }
            prepared_stages.push(handles);
        }
        let tail_masks = PreparedTailMasks::new(&stages.instrs, &prepared_stages, atoms.n_ids());
        Self {
            stages: prepared_stages.into_boxed_slice(),
            access_counts,
            tail_masks,
        }
    }

    fn stage(&self, index: usize) -> &[PreparedIndexSlot] {
        &self.stages[index]
    }

    fn access_count(&self, atom: AtomId) -> usize {
        self.access_counts.get(atom).copied().unwrap_or_default()
    }

    fn all_stage_mask(&self) -> Option<u64> {
        self.tail_masks.as_ref().map(|masks| masks.all_stages)
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
