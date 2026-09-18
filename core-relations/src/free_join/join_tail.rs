//! Analyze remaining join stages and manage recursive task state.
//!
//! Liveness determines which rows and materializations a child task retains.
//! Dynamic variable ordering updates the stage schedule and the leaf-scan
//! flags that allow bindings to remain factorized until action execution.

use std::{
    ops::Range,
    sync::{Arc, Mutex, atomic::AtomicUsize},
};

use crossbeam::utils::CachePadded;
use smallvec::SmallVec;

use crate::{
    common::{IndexMap, Value},
    numeric_id::{DenseIdMap, IdVec},
    offsets::Subset,
    row_buffer::{RowBuffer, SmallValueVec, TaggedRowBuffer},
};

use super::{
    ActionId, AtomId, Variable,
    frame_update::FrameUpdates,
    packed_cache::TrieRoot,
    packed_trie::ChildShape,
    plan::{JoinStage, MatId, MatScanMode},
    prepared_index::{AccessId, PreparedIndexSlot, PreparedJoinIndexes},
    probe::{AtomRows, Prober},
    with_pool_set,
};

/// Visit every atom whose trie node can be read while executing `stage`.
/// Keep this match exhaustive: a new stage variant must declare its subset
/// dependencies before task-state projection can remain sound.
pub(super) fn for_each_stage_atom(stage: &JoinStage, mut f: impl FnMut(AtomId)) {
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
pub(super) fn for_each_stage_indexed_access(
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

/// Whether an atom's rows are needed by the remaining join stages, together
/// with the child-index storage required by its next reorderable phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct AtomTailUse {
    pub(super) keep_rows: bool,
    pub(super) child_shape: ChildShape,
}

/// Find the first dynamically reorderable phase that touches `atom`. That
/// phase determines the packed child representation; finding any such phase
/// also proves that the current rows must survive into the tail.
pub(super) fn scan_atom_tail_use(
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

pub(super) fn atom_tail_use(
    atom: AtomId,
    stages: &[JoinStage],
    prepared: &PreparedJoinIndexes,
    remaining_stages: Option<u64>,
    instr_order: &InstrOrder,
    resume_pos: usize,
) -> AtomTailUse {
    let Some((masks, remaining_stages)) = prepared.tail_masks.as_ref().zip(remaining_stages) else {
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
pub(super) fn packed_child_shape_in_tail(
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

pub(super) fn materialization_is_live_in_tail(
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

/// Deferred, factorized variable bindings produced by leaf scans.
///
/// Each entry is one independent factor: an ordered list of variables paired
/// with a row buffer whose columns contain values for those variables. For
/// example, factors `([x], [[1], [2]])` and `([y], [[3], [4]])` represent the
/// four bindings `(x, y) = (1, 3), (1, 4), (2, 3), (2, 4)` without constructing
/// those combinations during the join. At the action boundary,
/// `expand_binding_sets` enumerates the Cartesian product and merges each row
/// into the scalar bindings accumulated by non-leaf stages. The buffers are
/// shared by `Arc` so recursive task-state clones retain a factor without
/// copying its rows.
pub(super) type BindingSet = Vec<(SmallVec<[Variable; 4]>, Arc<TaggedRowBuffer<SmallValueVec>>)>;

/// Bindings and row sets available in one recursive join branch. This includes
/// scalar bindings, deferred [`BindingSet`] factors, current atom subsets, and
/// materialized subquery results.
#[derive(Default)]
pub(super) struct BindingInfo<'rows, 'exec> {
    pub(super) bindings: DenseIdMap<Variable, Value>,
    pub(super) binding_sets: BindingSet,
    pub(super) subsets: DenseIdMap<AtomId, AtomRows<'rows, 'exec>>,
    pub(super) materializations: DenseIdMap<MatId, Arc<IndexMap<Vec<Value>, RowBuffer>>>,
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
    pub(super) fn clone_for_join_tail<'short>(
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
    pub(super) fn insert_subset(&mut self, atom: AtomId, subset: Subset) {
        let rows = match subset {
            Subset::Dense(range) => AtomRows::Dense(range),
            subset => AtomRows::Root(Arc::new(TrieRoot::new(subset))),
        };
        self.subsets.insert(atom, rows);
    }

    pub(super) fn insert_node(&mut self, atom: AtomId, node: impl Into<AtomRows<'rows, 'exec>>) {
        self.subsets.insert(atom, node.into());
    }

    /// Probers returned from `JoinState::get_index` will move atom-related state out of the
    /// [`BindingInfo`]. Once the caller is done using a prober, this method moves it back.
    pub(super) fn move_back(&mut self, atom: AtomId, prober: Prober<'_, 'rows, 'exec>) {
        self.subsets.insert(atom, prober.source);
    }

    pub(super) fn move_back_node(&mut self, atom: AtomId, node: impl Into<AtomRows<'rows, 'exec>>) {
        self.subsets.insert(atom, node.into());
    }

    pub(super) fn has_empty_subset(&self, atom: AtomId) -> bool {
        self.subsets[atom].is_empty()
    }

    pub(super) fn unwrap_val(&mut self, atom: AtomId) -> AtomRows<'rows, 'exec> {
        self.subsets.unwrap_val(atom)
    }
}

/// Counts complete matches per rule action across concurrent join tasks.
/// Each atomic counter occupies a separate cache line to avoid false sharing
/// between tasks updating different actions.
pub(super) struct MatchCounter {
    matches: IdVec<ActionId, CachePadded<AtomicUsize>>,
}

impl MatchCounter {
    pub(super) fn new(n_ids: usize) -> Self {
        let mut matches = IdVec::with_capacity(n_ids);
        matches.resize_with(n_ids, || CachePadded::new(AtomicUsize::new(0)));
        Self { matches }
    }

    pub(super) fn inc_matches(&self, action: ActionId, by: usize) {
        self.matches[action].fetch_add(by, std::sync::atomic::Ordering::Relaxed);
    }
    pub(super) fn read_matches(&self, action: ActionId) -> usize {
        self.matches[action].load(std::sync::atomic::Ordering::Acquire)
    }
}

pub(super) fn estimate_size(join_stage: &JoinStage, binding_info: &BindingInfo<'_, '_>) -> usize {
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

pub(super) fn is_reorder_barrier(stage: &JoinStage) -> bool {
    matches!(
        stage,
        JoinStage::FusedIntersectMat {
            mode: MatScanMode::Lookup(_) | MatScanMode::Value(_) | MatScanMode::Full,
            ..
        }
    )
}

/// Reorder a suffix of the cached logical plan using the DVO policy.
///
/// Within each phase, the greedy key is decreasing refinement count in the
/// logical plan prefix, increasing live residual size, then decreasing number
/// of intersected relations. Materialization stages that do not commute remain
/// fixed barriers. Reordering can also change factorized leaf-scan eligibility,
/// so [`recompute_leaf_scans`] runs after every change.
pub(super) fn sort_plan_by_size(
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

/// Recompute which scheduled stages can emit a factorized binding set.
///
/// A leaf scan produces bindings that no remaining stage needs to inspect. Its
/// rows can therefore be stored as one [`BindingSet`] factor instead of
/// recursively expanding every row through the rest of the join. Whether a
/// scan is a leaf depends on what comes after it, so the flags are recomputed
/// after the initial DVO sort and whenever runtime DVO reorders a suffix.
///
/// This recomputes `leaf_scans[i]` for every position in
/// `start..order.len()`. A position is eligible when its stage is either a
/// `FusedIntersect` or `FusedIntersectMat { mode: Full | KeyOnly | Value }`,
/// has an empty `to_intersect`, and no later stage either (a) references the
/// same cover atom for `FusedIntersect`, or (b) reads one of its bound variables
/// as a scalar through `FusedIntersectMat { mode: Value | Lookup }`.
/// `FusedIntersectMat::Lookup` binds nothing itself and is never a leaf scan.
pub(super) fn recompute_leaf_scans(
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

pub(super) fn sort_plan_by_size_inner(
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
    // prefix, as the DVO heuristic does.
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

/// Maps execution positions to stage indices in the logical plan. Dynamic
/// variable ordering changes this permutation without moving the plan's stages
/// or their prepared index state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct InstrOrder {
    data: SmallVec<[u16; 8]>,
}

impl InstrOrder {
    pub(super) fn new() -> Self {
        InstrOrder {
            data: SmallVec::new(),
        }
    }

    pub(super) fn from_iter(range: impl Iterator<Item = usize>) -> InstrOrder {
        let mut res = InstrOrder::new();
        res.data
            .extend(range.map(|x| u16::try_from(x).expect("too many instructions")));
        res
    }

    pub(super) fn get(&self, idx: usize) -> usize {
        self.data[idx] as usize
    }
    pub(super) fn len(&self) -> usize {
        self.data.len()
    }
}

/// Per-position factorization flags for the current physical schedule.
///
/// `leaf_scans[i] == true` means the stage at
/// `instrs[instr_order.get(i)]` has no consumers later in the current order and
/// may append a [`BindingSet`] factor instead of expanding its rows immediately.
/// [`sort_plan_by_size`] recomputes the flags whenever DVO changes that order.
pub(super) type LeafScans = SmallVec<[bool; 8]>;

/// Mutable view of the state used by a recursive join call. A call can update
/// these fields directly or use [`Self::clone_state`] to create a separate
/// [`LocalState`] for a child task.
pub(super) struct BorrowedLocalState<'a, 'rows, 'exec> {
    pub(super) instr_order: &'a mut InstrOrder,
    pub(super) leaf_scans: &'a mut LeafScans,
    pub(super) binding_info: &'a mut BindingInfo<'rows, 'exec>,
    pub(super) updates: &'a mut FrameUpdates<'rows, 'exec>,
}

/// Identifies the remaining join stages when cloning state for a child task.
/// `resume_pos` is a position in the current [`InstrOrder`], not an index into
/// `stages`; the suffix determines which atom subsets and materializations
/// the child must retain.
pub(super) struct SubsetClonePlan<'a> {
    pub(super) stages: &'a [JoinStage],
    pub(super) resume_pos: usize,
}

impl<'rows, 'exec> BorrowedLocalState<'_, 'rows, 'exec>
where
    'exec: 'rows,
{
    pub(super) fn clone_state<'short>(
        &mut self,
        plan: SubsetClonePlan<'_>,
    ) -> LocalState<'short, 'exec>
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

/// State owned by one recursive join task: its stage order, factorization
/// flags, bindings, and buffered frame updates. Row references within the
/// state may still borrow shared indexes or execution-arena storage.
pub(super) struct LocalState<'rows, 'exec> {
    pub(super) instr_order: InstrOrder,
    pub(super) leaf_scans: LeafScans,
    pub(super) binding_info: BindingInfo<'rows, 'exec>,
    pub(super) updates: FrameUpdates<'rows, 'exec>,
}

/// Collects completed task states for destruction after their execution scope
/// finishes. Deferring their cleanup reduces concurrent decrements of shared
/// `Arc` counters as workers complete sibling tasks.
#[derive(Default)]
pub(super) struct RetiredLocalStates<'rows, 'exec> {
    states: Mutex<Vec<LocalState<'rows, 'exec>>>,
}

impl<'rows, 'exec> RetiredLocalStates<'rows, 'exec> {
    pub(super) fn retire(&self, state: LocalState<'rows, 'exec>) {
        self.states
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(state);
    }
}

impl<'rows, 'exec> LocalState<'rows, 'exec> {
    pub(super) fn borrow_mut<'a>(&'a mut self) -> BorrowedLocalState<'a, 'rows, 'exec> {
        BorrowedLocalState {
            instr_order: &mut self.instr_order,
            leaf_scans: &mut self.leaf_scans,
            binding_info: &mut self.binding_info,
            updates: &mut self.updates,
        }
    }
}
