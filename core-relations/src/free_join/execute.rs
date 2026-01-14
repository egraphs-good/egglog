//! Core free join execution.

use std::{
    cmp, iter, mem,
    sync::{Arc, OnceLock, atomic::AtomicUsize},
};

use crate::{
    common::HashMap,
    free_join::plan::{JoinStages, MatId, MatScanMode, MatSpec, ScanMatSpec},
    numeric_id::{DenseIdMap, IdVec, NumericId},
    query::Atom,
    row_buffer::RowBuffer,
};
use crossbeam::utils::CachePadded;
use dashmap::mapref::one::RefMut;
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
    hash_index::{ColumnIndex, IndexBase, TupleIndex},
    offsets::{Offsets, SortedOffsetVector, Subset},
    parallel_heuristics::parallelize_db_level_op,
    pool::Pooled,
    query::RuleSet,
    row_buffer::TaggedRowBuffer,
    table_spec::{ColumnId, Offset, WrappedTableRef},
};

use super::{
    ActionId, AtomId, Database, HashColumnIndex, HashIndex, TableInfo, Variable,
    get_column_index_from_tableinfo,
    plan::{JoinHeader, JoinStage, Plan},
    with_pool_set,
};

enum DynamicIndex {
    Cached {
        intersect_outer: bool,
        table: HashIndex,
    },
    CachedColumn {
        intersect_outer: bool,
        table: HashColumnIndex,
    },
    Dynamic(TupleIndex),
    DynamicColumn(Arc<ColumnIndex>),
}

struct Prober {
    node: TrieNode,
    pool: Pool<SortedOffsetVector>,
    ix: DynamicIndex,
}

impl Prober {
    fn get_subset(&self, key: &[Value]) -> Option<Subset> {
        match &self.ix {
            DynamicIndex::Cached {
                intersect_outer,
                table,
            } => {
                let mut sub = table.get().unwrap().get_subset(key)?.to_owned(&self.pool);
                if *intersect_outer {
                    sub.intersect(self.node.subset.as_ref(), &self.pool);
                    if sub.is_empty() {
                        return None;
                    }
                }
                Some(sub)
            }
            DynamicIndex::CachedColumn {
                intersect_outer,
                table,
            } => {
                debug_assert_eq!(key.len(), 1);
                let mut sub = table
                    .get()
                    .unwrap()
                    .get_subset(&key[0])?
                    .to_owned(&self.pool);
                if *intersect_outer {
                    sub.intersect(self.node.subset.as_ref(), &self.pool);
                    if sub.is_empty() {
                        return None;
                    }
                }
                Some(sub)
            }
            DynamicIndex::Dynamic(tab) => tab.get_subset(key).map(|x| x.to_owned(&self.pool)),
            DynamicIndex::DynamicColumn(tab) => {
                tab.get_subset(&key[0]).map(|x| x.to_owned(&self.pool))
            }
        }
    }
    fn for_each(&self, mut f: impl FnMut(&[Value], SubsetRef)) {
        match &self.ix {
            DynamicIndex::Cached {
                intersect_outer: true,
                table,
            } => table.get().unwrap().for_each(|k, v| {
                let mut res = v.to_owned(&self.pool);
                res.intersect(self.node.subset.as_ref(), &self.pool);
                if !res.is_empty() {
                    f(k, res.as_ref())
                }
            }),
            DynamicIndex::Cached {
                intersect_outer: false,
                table,
            } => table.get().unwrap().for_each(|k, v| f(k, v)),
            DynamicIndex::CachedColumn {
                intersect_outer: true,
                table,
            } => {
                table.get().unwrap().for_each(|k, v| {
                    let mut res = v.to_owned(&self.pool);
                    res.intersect(self.node.subset.as_ref(), &self.pool);
                    if !res.is_empty() {
                        f(&[*k], res.as_ref())
                    }
                });
            }
            DynamicIndex::CachedColumn {
                intersect_outer: false,
                table,
            } => {
                table.get().unwrap().for_each(|k, v| f(&[*k], v));
            }
            DynamicIndex::Dynamic(tab) => {
                tab.for_each(f);
            }
            DynamicIndex::DynamicColumn(tab) => tab.for_each(|k, v| {
                f(&[*k], v);
            }),
        }
    }

    fn len(&self) -> usize {
        match &self.ix {
            DynamicIndex::Cached { table, .. } => table.get().unwrap().len(),
            DynamicIndex::CachedColumn { table, .. } => table.get().unwrap().len(),
            DynamicIndex::Dynamic(tab) => tab.len(),
            DynamicIndex::DynamicColumn(tab) => tab.len(),
        }
    }
}

impl Database {
    pub fn run_rule_set(&mut self, rule_set: &RuleSet, report_level: ReportLevel) -> RuleSetReport {
        if rule_set.plans.is_empty() {
            return RuleSetReport::default();
        }
        let match_counter = MatchCounter::new(rule_set.actions.n_ids());

        let search_and_apply_timer = Instant::now();
        // let mut rule_reports: HashMap<String, Vec<RuleReport>>;
        let mut rule_reports: HashMap<Arc<str>, Vec<RuleReport>>;
        let exec_state = ExecutionState::new(self.read_only_view(), Default::default());
        if parallelize_db_level_op(self.total_size_estimate) {
            // let dash_rule_reports: DashMap<String, Vec<RuleReport>> = DashMap::default();
            let dash_rule_reports: DashMap<Arc<str>, Vec<RuleReport>> = DashMap::default();
            rayon::in_place_scope(|scope| {
                for (plan, desc, symbol_map) in rule_set.plans.values() {
                    let Plan::SinglePlan(plan) = plan else {
                        todo!()
                    };
                    // TODO: add stats
                    let report_plan = match report_level {
                        ReportLevel::TimeOnly => None,
                        ReportLevel::WithPlan | ReportLevel::StageInfo => {
                            Some(plan.to_report(symbol_map))
                        }
                    };
                    scope.spawn(|scope| {
                        let join_state = JoinState::new(self, exec_state.clone());
                        let mut action_buf =
                            ScopedActionBuffer::new(scope, rule_set, &match_counter);
                        let mut binding_info = BindingInfo::default();
                        for (id, info) in plan.atoms.iter() {
                            let table = join_state.db.get_table(info.table);
                            binding_info.insert_subset(id, table.all());
                        }

                        let search_and_apply_timer = Instant::now();
                        join_state.run_join_stages(
                            &plan.stages,
                            &plan.atoms,
                            plan.actions,
                            &mut binding_info,
                            &mut action_buf,
                        );
                        let search_and_apply_time = search_and_apply_timer.elapsed();

                        if action_buf.needs_flush {
                            action_buf.flush(&mut exec_state.clone());
                        }
                        let mut rule_report: RefMut<'_, Arc<str>, Vec<RuleReport>> =
                            dash_rule_reports.entry(desc.clone()).or_default();
                        rule_report.value_mut().push(RuleReport {
                            plan: report_plan,
                            search_and_apply_time,
                            num_matches: usize::MAX,
                        });
                    });
                }
            });
            rule_reports = dash_rule_reports.into_iter().collect();
        } else {
            rule_reports = HashMap::default();
            let join_state = JoinState::new(self, exec_state.clone());
            // Just run all of the plans in order with a single in-place action
            // buffer.
            let mut action_buf = InPlaceActionBuffer {
                rule_set,
                match_counter: &match_counter,
                batches: Default::default(),
            };
            for (plan, desc, symbol_map) in rule_set.plans.values() {
                let report_plan = match report_level {
                    ReportLevel::TimeOnly => None,
                    ReportLevel::WithPlan | ReportLevel::StageInfo => {
                        Some(plan.to_report(symbol_map))
                    }
                };
                let mut binding_info = BindingInfo::default();

                for (id, info) in plan.atoms().iter() {
                    let table = join_state.db.get_table(info.table);
                    binding_info.insert_subset(id, table.all());
                }

                let search_and_apply_timer = Instant::now();
                match plan {
                    Plan::SinglePlan(plan) => {
                        join_state.run_join_stages(
                            &plan.stages,
                            &plan.atoms,
                            plan.actions,
                            &mut binding_info,
                            &mut action_buf,
                        );
                    }
                    Plan::DecomposedPlan(plan) => {
                        let mut materializations =
                            DenseIdMap::with_capacity(plan.stages.blocks.len());
                        for i in 0..plan.stages.blocks.len() {
                            materializations.insert(MatId::from_usize(i), DashMap::default());
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
                        for (mat_id, stage_block) in plan.stages.blocks.iter().enumerate() {
                            let mat_id = MatId::from_usize(mat_id);
                            join_state.run_join_stages(
                                &stage_block.0,
                                &plan.atoms,
                                mat_id,
                                &mut binding_info,
                                &mut materializer,
                            );
                            binding_info.materializations.insert(
                                mat_id,
                                Arc::new(materializer.materializations.take(mat_id).unwrap()),
                            );
                        }
                        join_state.run_join_stages(
                            &plan.result_block,
                            &plan.atoms,
                            plan.actions,
                            &mut binding_info,
                            &mut action_buf,
                        );
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

impl Default for ActionState {
    fn default() -> Self {
        Self {
            n_runs: 0,
            len: 0,
            bindings: Bindings::new(VAR_BATCH_SIZE),
        }
    }
}

struct JoinState<'a> {
    db: &'a Database,
    exec_state: ExecutionState<'a>,
}

type ColumnIndexes = IdVec<ColumnId, OnceLock<Arc<ColumnIndex>>>;

/// Information about the current subset of an atom's relation that is being considered, along with
/// lazily-initialized, cached indexes on that subset.
///
/// This is the standard trie-node used in lazy implementations of GJ as in the original egglog
/// implementation and the FJ paper. It currently does not handle non-column indexes, but that
/// should be a fairly straightforward extension if we start generating plans that need those.
/// (Right now, most plans iterating over more than one column just do a scan anyway).
struct TrieNode {
    /// The actual subset of the corresponding atom.
    subset: Subset,
    /// Any cached indexes on this subset.
    cached_subsets: OnceLock<Arc<Pooled<ColumnIndexes>>>,
}

impl TrieNode {
    fn size(&self) -> usize {
        self.subset.size()
    }
    fn get_cached_index(&self, col: ColumnId, info: &TableInfo) -> Arc<ColumnIndex> {
        self.cached_subsets.get_or_init(|| {
            // Pre-size the vector so we do not need to borrow it mutably to initialize the index.
            let mut vec: Pooled<ColumnIndexes> = with_pool_set(|ps| ps.get());
            vec.resize_with(info.spec.arity(), OnceLock::new);
            Arc::new(vec)
        })[col]
            .get_or_init(|| {
                let col_index = info.table.group_by_col(self.subset.as_ref(), col);
                Arc::new(col_index)
            })
            .clone()
    }
}

impl Clone for TrieNode {
    fn clone(&self) -> Self {
        let cached_subsets = OnceLock::new();
        if let Some(cached) = self.cached_subsets.get() {
            cached_subsets.set(cached.clone()).ok().unwrap();
        }
        Self {
            subset: self.subset.clone(),
            cached_subsets,
        }
    }
}

#[derive(Default, Clone)]
struct BindingInfo {
    bindings: DenseIdMap<Variable, Value>,
    subsets: DenseIdMap<AtomId, TrieNode>,
    materializations: DenseIdMap<MatId, Arc<DashMap<Vec<Value>, RowBuffer>>>,
}

impl BindingInfo {
    /// Initializes the atom-related metadata in the [`BindingInfo`].
    fn insert_subset(&mut self, atom: AtomId, subset: Subset) {
        let node = TrieNode {
            subset,
            cached_subsets: Default::default(),
        };
        self.subsets.insert(atom, node);
    }

    /// Probers returned from [`JoinState::get_index`] will move atom-related state out of the
    /// [`BindingInfo`]. Once the caller is done using a prober, this method moves it back.
    fn move_back(&mut self, atom: AtomId, prober: Prober) {
        self.subsets.insert(atom, prober.node);
    }

    fn move_back_node(&mut self, atom: AtomId, node: TrieNode) {
        self.subsets.insert(atom, node);
    }

    fn has_empty_subset(&self, atom: AtomId) -> bool {
        self.subsets[atom].subset.is_empty()
    }

    fn unwrap_val(&mut self, atom: AtomId) -> TrieNode {
        self.subsets.unwrap_val(atom)
    }
}

impl<'a> JoinState<'a> {
    fn new(db: &'a Database, exec_state: ExecutionState<'a>) -> Self {
        Self { db, exec_state }
    }

    fn get_index(
        &self,
        atoms: &Arc<DenseIdMap<AtomId, Atom>>,
        atom: AtomId,
        binding_info: &mut BindingInfo,
        cols: impl Iterator<Item = ColumnId>,
    ) -> Prober {
        let cols = SmallVec::<[ColumnId; 4]>::from_iter(cols);
        let trie_node = binding_info.subsets.unwrap_val(atom);
        let subset = &trie_node.subset;

        let table_id = atoms[atom].table;
        let info = &self.db.tables[table_id];
        let all_cacheable = cols.iter().all(|col| {
            !info
                .spec
                .uncacheable_columns
                .get(*col)
                .copied()
                .unwrap_or(false)
        });
        let whole_table = info.table.all();
        let dyn_index =
            if all_cacheable && subset.is_dense() && whole_table.size() / 2 < subset.size() {
                // Skip intersecting with the subset if we are just looking at the
                // whole table.
                let intersect_outer =
                    !(whole_table.is_dense() && subset.bounds() == whole_table.bounds());
                // heuristic: if the subset we are scanning is somewhat
                // large _or_ it is most of the table, or we already have a cached
                // index for it, then return it.
                if cols.len() != 1 {
                    DynamicIndex::Cached {
                        intersect_outer,
                        table: get_index_from_tableinfo(info, &cols),
                    }
                } else {
                    DynamicIndex::CachedColumn {
                        intersect_outer,
                        table: get_column_index_from_tableinfo(info, cols[0]).clone(),
                    }
                }
            } else if cols.len() != 1 {
                // NB: we should have a caching strategy for non-column indexes.
                DynamicIndex::Dynamic(info.table.group_by_key(subset.as_ref(), &cols))
            } else {
                DynamicIndex::DynamicColumn(trie_node.get_cached_index(cols[0], info))
            };
        Prober {
            node: trie_node,
            pool: with_pool_set(|ps| ps.get_pool().clone()),
            ix: dyn_index,
        }
    }
    fn get_column_index(
        &self,
        atoms: &Arc<DenseIdMap<AtomId, Atom>>,
        binding_info: &mut BindingInfo,
        atom: AtomId,
        col: ColumnId,
    ) -> Prober {
        self.get_index(atoms, atom, binding_info, iter::once(col))
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
        atoms: &'buf Arc<DenseIdMap<AtomId, Atom>>,
        action: A,
        binding_info: &mut BindingInfo,
        action_buf: &mut BUF,
    ) where
        'a: 'buf,
    {
        for JoinHeader { atom, subset, .. } in &stages.header {
            if subset.is_empty() {
                return;
            }
            let mut cur = binding_info.unwrap_val(*atom);
            debug_assert!(cur.cached_subsets.get().is_none());
            cur.subset
                .intersect(subset.as_ref(), &with_pool_set(|ps| ps.get_pool()));
            binding_info.move_back_node(*atom, cur);
        }
        for (_, node) in binding_info.subsets.iter() {
            if node.subset.is_empty() {
                return;
            }
        }
        let mut order = InstrOrder::from_iter(0..stages.instrs.len());
        sort_plan_by_size(&mut order, 0, &stages.instrs, binding_info);
        self.run_plan(
            stages,
            atoms,
            action,
            &mut order,
            0,
            binding_info,
            action_buf,
        );
    }

    /// The core method for executing a free join plan.
    ///
    /// This method takes the plan, mutable data-structures for variable binding and staging
    /// actions, and two indexes: `cur` which is the current stage of the plan to run, and `level`
    /// which is the current "fan-out" node we are in. The latter parameter is an experimental
    /// index used to detect if we are at the "top" of a plan rather than the "bottom", and is
    /// currently used as a heuristic to determine if we should increase parallelism more than the
    /// default.
    fn run_plan<'buf, A: NumericId + 'buf, BUF: ActionBuffer<'buf, A>>(
        &self,
        stages: &'buf JoinStages,
        atoms: &'buf Arc<DenseIdMap<AtomId, Atom>>,
        action: A,
        instr_order: &mut InstrOrder,
        cur: usize,
        binding_info: &mut BindingInfo,
        action_buf: &mut BUF,
    ) where
        'a: 'buf,
    {
        if self.exec_state.should_stop() {
            return;
        }

        if cur >= instr_order.len() {
            action_buf.push_bindings(action, &binding_info.bindings, || {
                ExecutionState::new(self.db.read_only_view(), Default::default())
            });
            return;
        }
        let chunk_size = action_buf.morsel_size(cur, instr_order.len());
        let mut cur_size = estimate_size(&stages.instrs[instr_order.get(cur)], binding_info);
        if cur_size > 32 && cur % 3 == 1 && cur < instr_order.len() - 1 {
            // If we have a reasonable number of tuples to process, adjust the variable order every
            // 3 rounds, but always make sure to readjust on the second roung.
            sort_plan_by_size(instr_order, cur, &stages.instrs, binding_info);
            cur_size = estimate_size(&stages.instrs[instr_order.get(cur)], binding_info);
        }

        // Helper macro (not its own method to appease the borrow checker).
        macro_rules! drain_updates {
            ($updates:expr) => {
                if self.exec_state.should_stop() {
                    return;
                }
                if cur == 0 || cur == 1 {
                    drain_updates_parallel!($updates)
                } else {
                    $updates.drain(|update| match update {
                        UpdateInstr::PushBinding(var, val) => {
                            binding_info.bindings.insert(var, val);
                        }
                        UpdateInstr::RefineAtom(atom, subset) => {
                            binding_info.insert_subset(atom, subset);
                        }
                        UpdateInstr::EndFrame => {
                            self.run_plan(
                                stages,
                                atoms,
                                action,
                                instr_order,
                                cur + 1,
                                binding_info,
                                action_buf,
                            );
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
                action_buf.recur(
                    BorrowedLocalState {
                        binding_info,
                        instr_order,
                        updates: &mut $updates,
                    },
                    move || exec_state_for_factory.clone(),
                    move |BorrowedLocalState {
                              binding_info,
                              instr_order,
                              updates,
                          },
                          buf| {
                        updates.drain(|update| match update {
                            UpdateInstr::PushBinding(var, val) => {
                                binding_info.bindings.insert(var, val);
                            }
                            UpdateInstr::RefineAtom(atom, subset) => {
                                binding_info.insert_subset(atom, subset);
                            }
                            UpdateInstr::EndFrame => {
                                JoinState {
                                    db,
                                    exec_state: exec_state_for_work.clone(),
                                }
                                .run_plan(
                                    stages,
                                    atoms,
                                    action,
                                    instr_order,
                                    cur + 1,
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
            sub: Subset,
            constraints: &[Constraint],
            table: &WrappedTableRef,
        ) -> Subset {
            let sub = table.refine_live(sub);
            table.refine(sub, constraints)
        }

        match &stages.instrs[instr_order.get(cur)] {
            JoinStage::Intersect { var, scans } => match scans.as_slice() {
                [] => {}
                [a] if a.cs.is_empty() => {
                    if binding_info.has_empty_subset(a.atom) {
                        return;
                    }
                    let prober = self.get_column_index(atoms, binding_info, a.atom, a.column);
                    let table = self.db.tables[atoms[a.atom].table].table.as_ref();
                    let mut updates = FrameUpdates::with_capacity(cmp::min(chunk_size, cur_size));
                    with_pool_set(|ps| {
                        prober.for_each(|val, x| {
                            updates.push_binding(*var, val[0]);
                            let sub = refine_subset(x.to_owned(&ps.get_pool()), &[], &table);
                            if sub.is_empty() {
                                updates.rollback();
                                return;
                            }
                            updates.refine_atom(a.atom, sub);
                            updates.finish_frame();
                            if updates.frames() >= chunk_size {
                                drain_updates!(updates);
                            }
                        })
                    });
                    drain_updates!(updates);
                    binding_info.move_back(a.atom, prober);
                }
                [a] => {
                    if binding_info.has_empty_subset(a.atom) {
                        return;
                    }
                    let prober = self.get_column_index(atoms, binding_info, a.atom, a.column);
                    let table = self.db.tables[atoms[a.atom].table].table.as_ref();
                    let mut updates = FrameUpdates::with_capacity(cmp::min(chunk_size, cur_size));
                    with_pool_set(|ps| {
                        prober.for_each(|val, x| {
                            updates.push_binding(*var, val[0]);
                            let sub = refine_subset(x.to_owned(&ps.get_pool()), &a.cs, &table);
                            if sub.is_empty() {
                                updates.rollback();
                                return;
                            }
                            updates.refine_atom(a.atom, sub);
                            updates.finish_frame();
                            if updates.frames() >= chunk_size {
                                drain_updates!(updates);
                            }
                        })
                    });
                    drain_updates!(updates);
                    binding_info.move_back(a.atom, prober);
                }
                [a, b] => {
                    let a_prober = self.get_column_index(atoms, binding_info, a.atom, a.column);
                    let b_prober = self.get_column_index(atoms, binding_info, b.atom, b.column);

                    let ((smaller, smaller_scan), (larger, larger_scan)) =
                        if a_prober.len() < b_prober.len() {
                            ((&a_prober, a), (&b_prober, b))
                        } else {
                            ((&b_prober, b), (&a_prober, a))
                        };

                    let smaller_atom = smaller_scan.atom;
                    let larger_atom = larger_scan.atom;
                    let large_table = self.db.tables[atoms[larger_atom].table].table.as_ref();
                    let small_table = self.db.tables[atoms[smaller_atom].table].table.as_ref();
                    let mut updates = FrameUpdates::with_capacity(cmp::min(chunk_size, cur_size));
                    with_pool_set(|ps| {
                        smaller.for_each(|val, small_sub| {
                            if let Some(mut large_sub) = larger.get_subset(val) {
                                large_sub = refine_subset(large_sub, &larger_scan.cs, &large_table);
                                if large_sub.is_empty() {
                                    updates.rollback();
                                    return;
                                }
                                let small_sub = refine_subset(
                                    small_sub.to_owned(&ps.get_pool()),
                                    &smaller_scan.cs,
                                    &small_table,
                                );
                                if small_sub.is_empty() {
                                    updates.rollback();
                                    return;
                                }
                                updates.push_binding(*var, val[0]);
                                updates.refine_atom(smaller_atom, small_sub);
                                updates.refine_atom(larger_atom, large_sub);
                                updates.finish_frame();
                                if updates.frames() >= chunk_size {
                                    drain_updates_parallel!(updates);
                                }
                            }
                        });
                    });
                    drain_updates!(updates);

                    binding_info.move_back(a.atom, a_prober);
                    binding_info.move_back(b.atom, b_prober);
                }
                rest => {
                    let mut smallest = 0;
                    let mut smallest_size = usize::MAX;
                    let mut probers = Vec::with_capacity(rest.len());
                    for (i, scan) in rest.iter().enumerate() {
                        let prober =
                            self.get_column_index(atoms, binding_info, scan.atom, scan.column);
                        let size = prober.len();
                        if size < smallest_size {
                            smallest = i;
                            smallest_size = size;
                        }
                        probers.push(prober);
                    }

                    let main_spec = &rest[smallest];
                    let main_spec_table =
                        self.db.tables[atoms[main_spec.atom].table].table.as_ref();

                    if smallest_size != 0 {
                        // Smallest leads the scan
                        let mut updates =
                            FrameUpdates::with_capacity(cmp::min(chunk_size, cur_size));
                        probers[smallest].for_each(|key, sub| {
                            with_pool_set(|ps| {
                                updates.push_binding(*var, key[0]);
                                for (i, scan) in rest.iter().enumerate() {
                                    if i == smallest {
                                        continue;
                                    }
                                    if let Some(mut sub) = probers[i].get_subset(key) {
                                        let table = self.db.tables[atoms[rest[i].atom].table]
                                            .table
                                            .as_ref();
                                        sub = refine_subset(sub, &rest[i].cs, &table);
                                        if sub.is_empty() {
                                            updates.rollback();
                                            return;
                                        }
                                        updates.refine_atom(scan.atom, sub)
                                    } else {
                                        updates.rollback();
                                        // Empty intersection.
                                        return;
                                    }
                                }
                                let sub = sub.to_owned(&ps.get_pool());
                                let sub = refine_subset(sub, &main_spec.cs, &main_spec_table);
                                if sub.is_empty() {
                                    updates.rollback();
                                    return;
                                }
                                updates.refine_atom(main_spec.atom, sub);
                                updates.finish_frame();
                                if updates.frames() >= chunk_size {
                                    drain_updates_parallel!(updates);
                                }
                            })
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
                let cover_atom = cover.to_index.atom;
                if binding_info.has_empty_subset(cover_atom) {
                    return;
                }
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
                    for (row, key) in buffer.non_stale() {
                        updates.refine_atom(
                            cover_atom,
                            Subset::Dense(OffsetRange::new(row, row.inc())),
                        );
                        // bind the values
                        for (i, (_, var)) in bind.iter().enumerate() {
                            updates.push_binding(*var, key[i]);
                        }
                        updates.finish_frame();
                        if updates.frames() >= chunk_size {
                            drain_updates_parallel!(updates);
                        }
                    }
                    if let Some(next) = next {
                        cur = next;
                        continue;
                    }
                    break;
                }
                drain_updates!(updates);
                // Restore the subsets we swapped out.
                binding_info.move_back_node(cover_atom, cover_node);
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
                            ),
                        )
                    })
                    .collect::<SmallVec<[(usize, AtomId, Prober); 4]>>();
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
                    'mid: for (row, key) in buffer.non_stale() {
                        updates.refine_atom(
                            cover_atom,
                            Subset::Dense(OffsetRange::new(row, row.inc())),
                        );
                        // bind the values
                        for (i, (_, var)) in bind.iter().enumerate() {
                            updates.push_binding(*var, key[i]);
                        }
                        // now probe each remaining indexes
                        for (i, atom, prober) in &index_probers {
                            // create a key: to_intersect indexes into the key from the cover
                            let index_cols = &to_intersect[*i].1;
                            let index_key = index_cols
                                .iter()
                                .map(|col| key[col.index()])
                                .collect::<SmallVec<[Value; 4]>>();
                            let Some(mut subset) = prober.get_subset(&index_key) else {
                                updates.rollback();
                                // There are no possible values for this subset
                                continue 'mid;
                            };
                            // apply any constraints needed in this scan.
                            let table_info = &self.db.tables[atoms[*atom].table];
                            let cs = &to_intersect[*i].0.constraints;
                            subset = refine_subset(subset, cs, &table_info.table.as_ref());
                            if subset.is_empty() {
                                updates.rollback();
                                // There are no possible values for this subset
                                continue 'mid;
                            }
                            updates.refine_atom(*atom, subset);
                        }
                        updates.finish_frame();
                        if updates.frames() >= chunk_size {
                            drain_updates_parallel!(updates);
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
            } => {
                let cover_mat = binding_info.materializations[*cover].clone();
                let mut updates = FrameUpdates::with_capacity(cmp::min(chunk_size, cur_size));
                let probers = to_intersect
                    .iter()
                    .map(|(spec, _)| {
                        if let ScanMatSpec::Scan(spec) = spec {
                            Some(self.get_index(
                                atoms,
                                spec.to_index.atom,
                                binding_info,
                                spec.to_index.vars.iter().copied(),
                            ))
                        } else {
                            None
                        }
                    })
                    .collect::<SmallVec<[Option<Prober>; 4]>>();

                let mut key = Vec::with_capacity(4);
                let mut prune_probers = |updates: &mut FrameUpdates,
                                         binding_info: &mut BindingInfo,
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
                        match spec {
                            ScanMatSpec::Scan(spec) => {
                                let prober = prober.as_ref().unwrap();
                                if let Some(subset) = prober.get_subset(&key) {
                                    updates.refine_atom(spec.to_index.atom, subset);
                                } else {
                                    return false;
                                }
                            }
                            ScanMatSpec::Materialized(spec) => {
                                let mat = &binding_info.materializations[*spec];
                                if mat.contains_key(&key) {
                                    // We don't refine materializations. Materializations
                                    // are only refined when scanning in ScanMatMode::Refine mode,
                                    // which is done by looking up the relevant variables.
                                } else {
                                    return false;
                                }
                            }
                        }
                    }
                    return true;
                };

                match mode {
                    MatScanMode::Full | MatScanMode::KeyOnly => {
                        // enumerate keys
                        for group in cover_mat.iter() {
                            let group_key_len = group.key().len();
                            if mode == &MatScanMode::Full {
                                // enumerate non-keys
                                for non_keys in group.value().iter() {
                                    for (col, var) in bind.iter() {
                                        if col.index() < group_key_len {
                                            updates.push_binding(*var, group.key()[col.index()]);
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
                                        binding_info,
                                        Some(group.key()),
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
                                    updates.push_binding(*var, group.key()[col.index()]);
                                }
                                if prune_probers(
                                    &mut updates,
                                    binding_info,
                                    Some(group.key()),
                                    None,
                                ) {
                                    updates.finish_frame();
                                } else {
                                    updates.rollback();
                                }
                            }
                        }
                    }
                    MatScanMode::Value(index_vars) => {
                        let keys = index_vars
                            .iter()
                            .map(|var| binding_info.bindings[*var])
                            .collect::<Vec<Value>>();
                        // lookup keys
                        if let Some(group) = cover_mat.get(&keys) {
                            // enumerate non-keys
                            for vals in group.value().iter() {
                                debug_assert!(vals.len() == bind.len()); // TODO: not true for non-full query
                                for (col, var) in bind.iter() {
                                    updates.push_binding(*var, vals[col.index()]);
                                }
                                if prune_probers(&mut updates, binding_info, None, Some(vals)) {
                                    updates.finish_frame();
                                } else {
                                    updates.rollback();
                                }
                                if updates.frames() >= chunk_size {
                                    drain_updates_parallel!(updates);
                                }
                            }
                        }
                    }
                }

                drain_updates!(updates);
                for (spec, prober) in to_intersect.iter().zip(probers) {
                    if let ScanMatSpec::Scan(spec) = &spec.0 {
                        binding_info.move_back(spec.to_index.atom, prober.unwrap());
                    }
                }
            }
        }
    }
}

const VAR_BATCH_SIZE: usize = 128;

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

    /// The unit at which you should batch updates passed to calls to `recur`,
    /// potentially depending on the current level of recursion.
    ///
    /// As of right now this is just a hard-coded value. We may change it in the
    /// future to fan out more at higher levels though.
    fn morsel_size(&mut self, _level: usize, _total: usize) -> usize {
        256
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

    fn push_bindings(
        &mut self,
        action: ActionId,
        bindings: &DenseIdMap<Variable, Value>,
        mut to_exec_state: impl FnMut() -> ExecutionState<'a>,
    ) {
        let action_state = self.batches.get_or_default(action);
        action_state.n_runs += 1;
        action_state.len += 1;
        let action_info = &self.rule_set.actions[action];
        // SAFETY: `used_vars` is a constant per-rule. This module only ever calls it with
        // `bindings` produced by the same join.
        unsafe {
            action_state.bindings.push(bindings, &action_info.used_vars);
        }
        if action_state.len >= VAR_BATCH_SIZE {
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
}

/// An Action buffer that hands off batches to of actions to rayon to execute.
struct ScopedActionBuffer<'inner, 'scope> {
    scope: &'inner rayon::Scope<'scope>,
    rule_set: &'scope RuleSet,
    match_counter: &'scope MatchCounter,
    batches: DenseIdMap<ActionId, ActionState>,
    needs_flush: bool,
}

impl<'inner, 'scope> ScopedActionBuffer<'inner, 'scope> {
    fn new(
        scope: &'inner rayon::Scope<'scope>,
        rule_set: &'scope RuleSet,
        match_counter: &'scope MatchCounter,
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
    fn push_bindings(
        &mut self,
        action: ActionId,
        bindings: &DenseIdMap<Variable, Value>,
        mut to_exec_state: impl FnMut() -> ExecutionState<'scope>,
    ) {
        self.needs_flush = true;
        let action_state = self.batches.get_or_default(action);
        action_state.n_runs += 1;
        action_state.len += 1;
        let action_info = &self.rule_set.actions[action];
        // SAFETY: `used_vars` is a constant per-rule. This module only ever calls it with
        // `bindings` produced by the same join.
        unsafe {
            action_state.bindings.push(bindings, &action_info.used_vars);
        }
        if action_state.len >= VAR_BATCH_SIZE {
            let mut state = to_exec_state();
            let mut bindings =
                mem::replace(&mut action_state.bindings, Bindings::new(VAR_BATCH_SIZE));
            action_state.len = 0;
            let match_counter = self.match_counter;
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
            self.match_counter,
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
        let match_counter = self.match_counter;
        let mut inner = local.clone_state();
        self.scope.spawn(move |scope| {
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
                    buf.match_counter,
                );
            }
        });
    }

    fn morsel_size(&mut self, _level: usize, _total: usize) -> usize {
        // Lower morsel size to increase parallelism.
        match _level {
            0 if _total > 2 => 32,
            _ => 256,
        }
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
    materializations: DenseIdMap<MatId, DashMap<Vec<Value>, RowBuffer>>,
    scratch_key: Vec<Value>,
    scratch_val: Vec<Value>,
}

impl<'a> ActionBuffer<'a, MatId> for InPlaceMaterializer<'a> {
    type AsLocal<'b>
        = Self
    where
        'a: 'b;

    fn push_bindings(
        &mut self,
        mat_id: MatId,
        bindings: &DenseIdMap<Variable, Value>,
        _to_exec_state: impl FnMut() -> ExecutionState<'a>,
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
        if let Some(mut buffer) = mat.get_mut(&self.scratch_key) {
            buffer.add_row(&self.scratch_val);
        } else {
            if spec.val_vars.len() > 0 {
                let mut buffer = RowBuffer::new(spec.val_vars.len());
                buffer.add_row(&self.scratch_val);
                mat.insert(self.scratch_key.clone(), buffer);
            } else {
                let mut buffer = RowBuffer::new(1);
                buffer.add_row(&[Value::stale()]);
                mat.insert(self.scratch_key.clone(), buffer);
            }
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
        JoinStage::FusedIntersectMat { to_intersect, .. } => to_intersect.len() as i32 + 1,
    }
}

fn sort_plan_by_size(
    order: &mut InstrOrder,
    start: usize,
    instrs: &[JoinStage],
    binding_info: &mut BindingInfo,
) {
    // How many times an atom has been intersected/joined
    let mut times_refined = with_pool_set(|ps| ps.get::<DenseIdMap<AtomId, i64>>());

    // Count how many times each atom has been refined so far.
    for ins in instrs[..start].iter() {
        match ins {
            JoinStage::Intersect { scans, .. } => scans.iter().for_each(|scan| {
                *times_refined.get_or_default(scan.atom) += 1;
            }),
            JoinStage::FusedIntersect { cover, .. } => {
                *times_refined.get_or_default(cover.to_index.atom) +=
                    cover.to_index.vars.len() as i64;
            }
            JoinStage::FusedIntersectMat { .. } => {
                continue;
            }
        }
    }

    // We prioritize variables by
    //
    //   (1) how many times an atom with this variable has been refined,
    //   (2) then by how many relations joins on this variable
    //   (3) then by the cardinality of the variable to be enumerated
    let key_fn = |join_stage: &JoinStage,
                  binding_info: &BindingInfo,
                  times_refined: &DenseIdMap<AtomId, i64>| {
        let refine = match join_stage {
            JoinStage::Intersect { scans, .. } => scans
                .iter()
                .map(|scan| times_refined.get(scan.atom).copied().unwrap_or_default())
                .sum::<i64>(),
            JoinStage::FusedIntersect { cover, .. } => times_refined
                .get(cover.to_index.atom)
                .copied()
                .unwrap_or_default(),
            JoinStage::FusedIntersectMat { .. } => i64::MAX - 1, // prioritize materialized scans first
        };
        (
            -refine,
            -num_intersected_rels(join_stage),
            estimate_size(join_stage, binding_info),
        )
    };

    for i in start..order.len() {
        for j in i + 1..order.len() {
            let key_i = key_fn(&instrs[order.get(i)], binding_info, &times_refined);
            let key_j = key_fn(&instrs[order.get(j)], binding_info, &times_refined);
            if key_j < key_i {
                order.data.swap(i, j);
            }
        }
        // Update the counts after a new instruction is selected.
        match &instrs[order.get(i)] {
            JoinStage::Intersect { scans, .. } => scans.iter().for_each(|scan| {
                *times_refined.get_or_default(scan.atom) += 1;
            }),
            JoinStage::FusedIntersect { cover, .. } => {
                *times_refined.get_or_default(cover.to_index.atom) +=
                    cover.to_index.vars.len() as i64;
            }
            JoinStage::FusedIntersectMat { .. } => continue,
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

struct BorrowedLocalState<'a> {
    instr_order: &'a mut InstrOrder,
    binding_info: &'a mut BindingInfo,
    updates: &'a mut FrameUpdates,
}

impl BorrowedLocalState<'_> {
    fn clone_state(&mut self) -> LocalState {
        LocalState {
            instr_order: self.instr_order.clone(),
            binding_info: self.binding_info.clone(),
            updates: std::mem::take(self.updates),
        }
    }
}

struct LocalState {
    instr_order: InstrOrder,
    binding_info: BindingInfo,
    updates: FrameUpdates,
}

impl LocalState {
    fn borrow_mut<'a>(&'a mut self) -> BorrowedLocalState<'a> {
        BorrowedLocalState {
            instr_order: &mut self.instr_order,
            binding_info: &mut self.binding_info,
            updates: &mut self.updates,
        }
    }
}
