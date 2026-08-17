//! Core free join execution.
//!
//! The probe layer in `probe.rs` adapts every physical representation of an
//! atom's current rows to the two operations needed by join execution: look up
//! one key, or enumerate all keys. A probe may read a persistent table index, a
//! projection shared for the current ruleset run, a tiny residual index, or an
//! arena-allocated packed trie. It returns the matching rows in a representation
//! that later stages can refine without copying them. This separation keeps the
//! stage executor independent of the storage strategy selected for each access.
//!
//! The join-tail layer in `join_tail.rs` manages the state needed after
//! execution enters the recursive portion of a plan. Its helpers determine
//! which atom rows and materializations remain live, reorder the remaining
//! stages with dynamic variable ordering (DVO), identify results that can stay
//! factorized until action execution, and clone only the live state when work
//! moves to another task. Index selection and probing remain in the probe
//! layer; the join-tail layer decides their order and lifetime.

use std::{
    cell::RefCell,
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

include!("residual_index.rs");
include!("prepared_index.rs");
include!("probe.rs");
include!("packed_cache.rs");
include!("join_tail.rs");

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
                        let prepared_index = PreparedPlanIndexes::new(db, plan);
                        let search_and_apply_timer = Instant::now();
                        let search_and_apply_time = egglog_concurrency::scope(|query_scope| {
                            let join_state = JoinState::new(db, exec_state, trie_cache, &arena);
                            let mut binding_info = BindingInfo::default();
                            let mut action_buf = ScopedActionBuffer::new(
                                query_scope,
                                rule_set,
                                match_counter.clone(),
                            );

                            'eval: {
                                for (id, info) in plan.atoms().iter() {
                                    let headers: SmallVec<[&JoinHeader; 2]> =
                                        plan.header().iter().filter(|h| h.atom == id).collect();
                                    match join_state.root_node(info.table, &headers) {
                                        Some(node) => binding_info.insert_node(id, node),
                                        None => break 'eval,
                                    }
                                }

                                match (plan, &prepared_index) {
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

                        // Prepared slots can contain arena addresses, so destroy
                        // them before reclaiming the arena.
                        drop(prepared_index);
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
                let prepared_index = PreparedPlanIndexes::new(self, plan);
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

                drop(prepared_index);
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

/// Worker-local context for executing one logical join plan.
///
/// A `JoinState` combines the frozen database view and early-stop state needed
/// by the query with the allocation and caching resources used while traversing
/// that query's trie indexes. Recursive calls to [`JoinState::run_plan`] reuse
/// the same state. Parallel tasks construct their own `JoinState`, sharing the
/// database, cross-plan root cache, and query arena while retaining independent
/// arena handles and scratch storage.
///
/// The current variable bindings and atom row subsets are deliberately not
/// stored here: they describe one recursive branch and live in `BindingInfo`.
/// Likewise, the logical stages and their current physical order are explicit
/// arguments to `run_plan`.
struct JoinState<'db, 'state, 'exec> {
    /// Database owning the frozen tables queried by this plan.
    db: &'db Database,
    /// Copyable query-side database view and shared early-stop flag.
    exec_state: ExecutionStateSeed<'db, 'state>,
    /// Cached thread-local pool for SortedOffsetVector allocations.
    /// Stored here to avoid a per-call `with_pool_set` TLS access in `get_index`.
    pool: Pool<SortedOffsetVector>,
    /// Cross-plan trie-root cache for the current `run_rule_set`, or `None` when
    /// sharing is disabled (small run, or nothing reused across plans).
    trie_cache: Option<Arc<TrieCache>>,
    /// Query-scoped arena shared with any parallel tasks spawned by this plan.
    arena: &'exec SharedArena,
    /// This worker's allocation handle into `arena`.
    handle: Handle<'exec>,
    /// Reused `(value, row)` workspace for constructing packed trie nodes.
    packed_scratch: RefCell<Vec<(Value, RowId)>>,
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
            handle: arena.new_handle(),
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
        let sig: RootSignature = (table_id, fast);

        if !trie_cache.shared.contains(&sig) {
            // Not reused across plans: build a fresh, unshared root.
            return Some(Arc::new(TrieNode::new(
                self.build_root_subset(table_id, headers)?,
            )));
        }

        let header = trie_cache.header_id(&sig.1);
        let key: RootKey = (table_id, header);
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

    /// Build a packed index for one column of `subset` after applying the slow
    /// `constraints`.
    ///
    /// `subset_may_contain_stale_rows` describes the source subset, not the
    /// resulting node. Physical root and dense ranges can contain tombstoned
    /// row ids, whereas catalog indexes and packed child subsets already
    /// contain only live rows. When constraints require materializing a
    /// filtered subset, potentially stale inputs are refined to live rows
    /// first. With no constraints, the table scan used by
    /// `PackedTrieNode::build_from_subset` already skips stale rows.
    fn build_packed_node(
        &self,
        table: WrappedTableRef<'_>,
        subset: SubsetRef<'_>,
        subset_may_contain_stale_rows: bool,
        constraints: &[Constraint],
        column: ColumnId,
        child_shape: ChildShape,
    ) -> &'exec PackedTrieNode<'exec> {
        if constraints.is_empty() {
            // Table scans already omit stale SortedWritesTable rows, so a
            // separate live-subset pass would only scan the same root twice.
            return PackedTrieNode::build_from_subset(
                &self.handle,
                table,
                subset,
                column,
                child_shape,
                &mut self.packed_scratch.borrow_mut(),
            );
        }
        let mut filtered = subset.to_owned(&self.pool);
        if subset_may_contain_stale_rows && table.has_stale_rows() {
            filtered = table.refine_live(filtered);
        }
        filtered = table.refine(filtered, constraints);
        PackedTrieNode::build_from_subset(
            &self.handle,
            table,
            filtered.as_ref(),
            column,
            child_shape,
            &mut self.packed_scratch.borrow_mut(),
        )
    }

    /// Acquire a scalar index shared by plans that project the same root subset.
    ///
    /// A root projection groups the root's rows by `column` after applying
    /// `constraints`. `RootProjection` stores that grouping as immutable sorted
    /// keys plus the `RowId`s for each key. Shared trie roots own a map from
    /// `(column, constraints)` to `Arc<OnceLock<RootProjection>>`, so concurrent
    /// plans build a particular projection at most once.
    ///
    /// `prepared` is query-local state for this particular logical index use.
    /// It retains one clone of the shared `Arc`, keeping the projection alive
    /// and letting recursive output frames borrow it for `'rows` without doing
    /// another cache lookup or `Arc` clone. Returns `None` when root sharing is
    /// disabled or this root has no shared projection map.
    fn projected_root_index<'rows>(
        &self,
        root: &TrieNode,
        table: WrappedTableRef<'_>,
        constraints: &[Constraint],
        column: ColumnId,
        prepared: &'rows PreparedIndexSlot,
    ) -> Option<&'rows RootProjection> {
        self.trie_cache.as_ref()?;
        let slot = if let Some(slot) = prepared.projected_root.get() {
            slot
        } else {
            let candidate = root.projection_slot(column, constraints)?;
            prepared.projected_root.get_or_init(|| candidate)
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
        prepared: &'rows PreparedIndexSlot,
    ) -> &'exec PackedTrieNode<'exec>
    where
        'exec: 'rows,
    {
        match rows {
            AtomRows::Root(root) => {
                let address = *prepared.packed_root.get_or_init(|| {
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
            AtomRows::Packed(cursor) => {
                cursor.child_index_with(&self.handle, prepared.access.index(), child_shape, || {
                    self.build_packed_node(
                        table,
                        cursor.subset(),
                        false,
                        constraints,
                        column,
                        child_shape,
                    )
                })
            }
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
                let PreparedIndexKind::Column(index) = &prepared.kind else {
                    unreachable!("single-column scan must have a prepared column index")
                };
                let index = index
                    .get_or_init(|| get_column_index_from_tableinfo(info, cols[0]))
                    .get()
                    .expect("prepared column index must already be refreshed");
                if terminal_child_shape != ChildShape::Leaf {
                    prepared.root_continuations.prepare(
                        terminal_child_shape,
                        index.shard_count(),
                        |shard| index.shard_len(shard),
                    );
                }
                ProbeIndex::CachedColumn {
                    intersect_outer,
                    table: index,
                    continuations: &prepared.root_continuations,
                    child_shape: terminal_child_shape,
                }
            } else {
                let PreparedIndexKind::Tuple(index) = &prepared.kind else {
                    unreachable!("multi-column scan must have a prepared tuple index")
                };
                let index = index
                    .get_or_init(|| get_index_from_tableinfo(info, cols.as_slice()))
                    .get()
                    .expect("prepared tuple index must already be refreshed");
                if terminal_child_shape != ChildShape::Leaf {
                    prepared.root_continuations.prepare(
                        terminal_child_shape,
                        index.shard_count(),
                        |shard| index.shard_len(shard),
                    );
                }
                ProbeIndex::CachedTuple {
                    intersect_outer,
                    table: index,
                    continuations: &prepared.root_continuations,
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
                        .root_continuations
                        .prepare(first_child_shape, 1, |_| first.len());
                }
                ProbeIndex::ProjectedRoot(RootProjectionProbe {
                    first,
                    columns: cols,
                    table: info.table.as_ref(),
                    continuations: &prepared.root_continuations,
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
    fn top_index_shards<'rows>(
        &self,
        stage: &JoinStage,
        prepared: &'rows [PreparedIndexSlot],
        atoms: &Arc<DenseIdMap<AtomId, Atom>>,
        binding_info: &mut BindingInfo<'rows, 'exec>,
        workers: usize,
    ) -> Option<Vec<usize>>
    where
        'exec: 'rows,
    {
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
            let rows = &binding_info.subsets[scan.atom];
            let info = &self.db.tables[atoms[scan.atom].table];
            if !rows.is_root()
                || !scan.cs.is_empty()
                || info.table.has_stale_rows()
                || !columns_are_cacheable(info, &[scan.column])
            {
                return None;
            }
            let SubsetRef::Dense(subset) = rows.subset() else {
                return None;
            };
            let Subset::Dense(whole_table) = info.table.all() else {
                return None;
            };
            if subset != whole_table {
                return None;
            }
        }

        let mut leader = 0;
        let mut leader_size = usize::MAX;
        let mut probers = Vec::with_capacity(scans.len());
        for (i, (scan, prepared)) in scans.iter().zip(prepared).enumerate() {
            let prober = self.get_index(
                atoms,
                binding_info,
                ProbeRequest::column(scan, false, ChildShape::Leaf, prepared),
            );
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
    fn select_top_index_shards<'rows>(
        &self,
        stages: &JoinStages,
        prepared: &'rows PreparedJoinIndexes,
        atoms: &Arc<DenseIdMap<AtomId, Atom>>,
        order: &InstrOrder,
        binding_info: &mut BindingInfo<'rows, 'exec>,
    ) -> Option<Vec<usize>>
    where
        'exec: 'rows,
    {
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
                            Some(shard),
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

    /// Recursively execute the remaining stages of a free-join plan.
    ///
    /// The `JoinState` supplies the frozen database view, query arena, caches,
    /// and early-stop flag. The parameters describe the plan and the current
    /// recursive branch:
    ///
    /// - `stages` is the immutable logical join program being executed.
    /// - `prepared` contains execution-local cache slots and liveness metadata
    ///   aligned with the logical stages.
    /// - `atoms` maps the atom ids referenced by the stages to their tables and
    ///   column metadata.
    /// - `action` identifies the rule action to run for each complete binding.
    /// - `instr_order` maps each physical execution position to a logical stage;
    ///   dynamic ordering may reorder its unexecuted suffix.
    /// - `leaf_scans` is aligned with physical execution positions and records
    ///   which stages may produce factorized leaf bindings. It is updated when
    ///   `instr_order` changes.
    /// - `cur` is the physical position to execute next. Reaching the end sends
    ///   the completed binding to `action_buf`.
    /// - `index_shard` optionally restricts the first intersection in a
    ///   top-level coarse partition to one physical index shard. Recursive calls
    ///   clear it. It is only present with a serial action buffer, which keeps
    ///   the partition's later work inline rather than introducing nested
    ///   parallelism.
    /// - `remaining_stages` is a bit set of logical stages in the unexecuted
    ///   suffix. It enables constant-time child-shape and value-liveness checks;
    ///   plans with more than 64 stages use `None` and scan the suffix instead.
    /// - `binding_info` owns the partial scalar and factorized bindings, current
    ///   row subsets for each atom, and branch-local materializations.
    /// - `action_buf` receives complete bindings and controls whether recursive
    ///   work is run inline or divided into parallel morsels.
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
        index_shard: Option<usize>,
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
        let chunk_size = action_buf.morsel_size(cur, instr_order.len());
        let mut cur_size = estimate_size(&stages.instrs[instr_order.get(cur)], binding_info);
        if cur_size > 32 && cur % 3 == 1 && cur < instr_order.len() - 1 {
            // Re-evaluate the remaining suffix after observing the residuals
            // produced by earlier stages. Packed child families make the
            // resulting atom-local successor choice safe to cache again.
            sort_plan_by_size(instr_order, leaf_scans, cur, &stages.instrs, binding_info);
            cur_size = estimate_size(&stages.instrs[instr_order.get(cur)], binding_info);
        }

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
            };
        }
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
                            &prepared_indexes[0],
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
                    drain_updates!(updates);
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
                            &prepared_indexes[0],
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
                            &prepared_indexes[1],
                        ),
                    );

                    let ((smaller, smaller_scan), (larger, larger_scan)) =
                        if a_prober.len() <= b_prober.len() {
                            ((&a_prober, a), (&b_prober, b))
                        } else {
                            ((&b_prober, b), (&a_prober, a))
                        };

                    let mut updates = FrameUpdates::with_capacity(cmp::min(chunk_size, cur_size));
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
                    drain_updates!(updates);

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
                                prepared_slot,
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
                    let cover_subset = cover_node.subset();

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
                                    &prepared_indexes[i],
                                ),
                            ),
                        )
                    })
                    .collect::<SmallVec<[(usize, AtomId, Prober<'_, 'rows, 'exec>); 4]>>();
                let proj = SmallVec::<[ColumnId; 4]>::from_iter(bind.iter().map(|(col, _)| *col));
                let cover_node = binding_info.unwrap_val(cover_atom);
                let cover_subset = cover_node.subset();
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
                    drain_updates!(updates);
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
                                    prepared_slot,
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

                    drain_updates!(updates);
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

impl<'scope, 'exec> ActionBuffer<'scope, 'exec, ActionId> for SerialScopedActionBuffer<'scope>
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

impl<'scope, 'exec> ActionBuffer<'scope, 'exec, ActionId> for ScopedActionBuffer<'_, 'scope>
where
    'exec: 'scope,
{
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
            &mut SerialScopedActionBuffer<'scope>,
        ) + Send
        + 'scope,
    ) where
        'rows: 'scope,
    {
        let rule_set = self.rule_set;
        let match_counter = self.match_counter.clone();
        let mut inner = local.clone_state(subset_clone_plan);
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

    fn supports_parallel_drain(&self) -> bool {
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

#[cfg(test)]
#[path = "execute_tests.rs"]
mod top_index_tests;
