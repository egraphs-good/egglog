//! Execute queries against a database using a variant of Free Join.
use std::{
    mem,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::{
    common::IndexSet,
    hash_index::IndexCatalog,
    numeric_id::{DenseIdMap, DenseIdMapWithReuse, NumericId, define_id},
};
use crossbeam_queue::SegQueue;
use egglog_concurrency::{NotificationList, ResettableOnceLock};
use smallvec::SmallVec;

use crate::{
    BaseValues, ContainerValues, PoolSet, QueryEntry, TupleIndex, Value,
    action::{
        Bindings, DbView,
        mask::{Mask, MaskIter, ValueSource},
    },
    containers::{
        ContainerRebuildContext, ContainerRebuildPlan, ContainerRebuildSummary, ContainerValue,
        DynamicContainerEnv, SequenceContainerValue,
    },
    dependency_graph::DependencyGraph,
    hash_index::{ColumnIndex, Index, IndexBase},
    offsets::Subset,
    parallel,
    parallel_heuristics::{parallelize_db_level_op, parallelize_rebuild},
    pool::{Pool, Pooled, with_pool_set},
    query::{Query, RuleSetBuilder},
    table_spec::{
        ColumnId, Constraint, MaintenanceTable, MutationBuffer, Table, TableSpec, WrappedTable,
        WrappedTableRef,
    },
};

use self::plan::Plan;
use crate::action::ExecutionState;

pub(crate) mod execute;
pub(crate) mod frame_update;
// The packed trie is exercised independently before it is wired into execution.
#[allow(dead_code)]
pub(crate) mod packed_trie;
pub(crate) mod plan;

define_id!(
    pub AtomId,
    u32,
    "A component of a query consisting of a function and a list of variables or constants"
);
define_id!(pub Variable, u32, "a variable in a query", pretty "Var");

impl Variable {
    pub fn placeholder() -> Variable {
        Variable::new(!0)
    }
}

define_id!(
    pub TableId,
    u32,
    "a fixed- or variable-arity table in the database"
);

impl TableId {
    pub fn dummy() -> TableId {
        TableId::new(u32::MAX)
    }

    pub fn is_dummy(&self) -> bool {
        self.rep == u32::MAX
    }
}

define_id!(pub(crate) ActionId, u32, "an identifier picking out the RHS of a rule");

#[derive(Debug)]
pub(crate) struct ProcessedConstraints {
    /// The subset of the table matching the fast constraints. If there are no
    /// fast constraints then this is the full table.
    pub(crate) subset: Subset,
    /// The constraints that can be evaluated quickly (O(log(n)) or O(1)).
    pub(crate) fast: Pooled<Vec<Constraint>>,
    /// The constraints that require an O(n) scan to evaluate.
    pub(crate) slow: Pooled<Vec<Constraint>>,
}

impl Clone for ProcessedConstraints {
    fn clone(&self) -> Self {
        ProcessedConstraints {
            subset: self.subset.clone(),
            fast: Pooled::cloned(&self.fast),
            slow: Pooled::cloned(&self.slow),
        }
    }
}

impl ProcessedConstraints {
    /// The size of the subset of the table matching the fast constraints.
    fn approx_size(&self) -> usize {
        self.subset.size()
    }

    pub(crate) fn dummy() -> ProcessedConstraints {
        ProcessedConstraints {
            subset: Subset::empty(),
            fast: Pooled::new(Vec::new()),
            slow: Pooled::new(Vec::new()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SubAtom {
    pub(crate) atom: AtomId,
    pub(crate) vars: SmallVec<[ColumnId; 2]>,
}

impl SubAtom {
    pub(crate) fn new(atom: AtomId) -> SubAtom {
        SubAtom {
            atom,
            vars: Default::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct VarInfo {
    pub(crate) occurrences: Vec<SubAtom>,
    /// Whether or not this variable shows up in the "actions" portion of a
    /// rule.
    pub(crate) used_in_rhs: bool,
    pub(crate) defined_in_rhs: bool,
    pub(crate) name: Option<Arc<str>>,
}

pub(crate) type HashIndex = Arc<ResettableOnceLock<Index<TupleIndex>>>;
pub(crate) type HashColumnIndex = Arc<ResettableOnceLock<Index<ColumnIndex>>>;

pub struct TableInfo {
    pub(crate) name: Option<Arc<str>>,
    pub(crate) spec: TableSpec,
    pub(crate) table: WrappedTable,
    pub(crate) indexes: IndexCatalog<SmallVec<[ColumnId; 4]>, HashIndex>,
    pub(crate) column_indexes: IndexCatalog<ColumnId, HashColumnIndex>,
}

impl TableInfo {
    pub fn table(&self) -> &WrappedTable {
        &self.table
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn spec(&self) -> &TableSpec {
        &self.spec
    }
}

impl Clone for TableInfo {
    fn clone(&self) -> Self {
        fn deep_clone_map<K: Clone + std::hash::Hash + Eq, TI: IndexBase + Clone>(
            map: &IndexCatalog<K, Arc<ResettableOnceLock<Index<TI>>>>,
            table: WrappedTableRef,
        ) -> IndexCatalog<K, Arc<ResettableOnceLock<Index<TI>>>> {
            map.map(|table_ref| {
                let (k, v) = table_ref;
                let v: Index<TI> = v
                    .get_or_update(|index| {
                        index.refresh(table);
                    })
                    .clone();
                (k.clone(), Arc::new(ResettableOnceLock::new(v)))
            })
        }
        TableInfo {
            name: self.name.clone(),
            spec: self.spec.clone(),
            table: self.table.dyn_clone(),
            indexes: deep_clone_map(&self.indexes, self.table.as_ref()),
            column_indexes: deep_clone_map(&self.column_indexes, self.table.as_ref()),
        }
    }
}

define_id!(pub CounterId, u32, "A counter accessible to actions, useful for generating unique Ids.");
define_id!(pub ExternalFunctionId, u32, "A user-defined operation that can be invoked from a query");

/// External functions allow external callers to manipulate database state in
/// near-arbitrary ways.
///
/// This is a useful, if low-level, interface for extending this database with
/// functionality and state not built into the core model.
pub trait ExternalFunction: dyn_clone::DynClone + Send + Sync {
    /// Invoke the function with mutable access to the database. If a value is
    /// not returned, halt the execution of the current rule.
    fn invoke(&self, state: &mut ExecutionState, args: &[Value]) -> Option<Value>;
}

/// Automatically generate an `ExternalFunction` implementation from a function.
pub fn make_external_func<
    F: Fn(&mut ExecutionState, &[Value]) -> Option<Value> + Clone + Send + Sync,
>(
    f: F,
) -> impl ExternalFunction {
    #[derive(Clone)]
    struct Wrapped<F>(F);
    impl<F> ExternalFunction for Wrapped<F>
    where
        F: Fn(&mut ExecutionState, &[Value]) -> Option<Value> + Clone + Send + Sync,
    {
        fn invoke(&self, state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
            (self.0)(state, args)
        }
    }
    Wrapped(f)
}

/// A vectorized variant of [`ExternalFunction::invoke`] to avoid repeated dynamic dispatch.
pub(crate) fn invoke_batch(
    this: &dyn ExternalFunction,
    state: &mut ExecutionState,
    mask: &mut Mask,
    bindings: &mut Bindings,
    args: &[QueryEntry],
    out_var: Variable,
) {
    let pool: Pool<Vec<Value>> = with_pool_set(|ps| ps.get_pool());
    let mut out = pool.get();
    out.reserve(mask.len());
    for_each_binding_with_mask!(mask, args, bindings, |iter| {
        iter.fill_vec(&mut out, Value::stale, |_, args| {
            this.invoke(state, args.as_slice())
        });
    });
    bindings.insert(out_var, &out);
}

/// A variant of [`invoke_batch`] that overwrites the output variable,
/// rather than assigning all new values.
///
/// *Panics* This method will panic if `out_var` doesn't already have an appropriately-sized
/// vector bound in `bindings`.
pub(crate) fn invoke_batch_assign(
    this: &dyn ExternalFunction,
    state: &mut ExecutionState,
    mask: &mut Mask,
    bindings: &mut Bindings,
    args: &[QueryEntry],
    out_var: Variable,
) {
    let mut out = bindings.take(out_var).expect("out_var must be bound");
    for_each_binding_with_mask!(mask, args, bindings, |iter| {
        iter.assign_vec_and_retain(&mut out.vals, |_, args| this.invoke(state, &args))
    });
    bindings.replace(out);
}

// Implements `Clone` for `Box<dyn ExternalFunction>`.
dyn_clone::clone_trait_object!(ExternalFunction);

pub(crate) type ExternalFunctions =
    DenseIdMapWithReuse<ExternalFunctionId, Box<dyn ExternalFunction>>;

// Reservable counters give each execution state a disjoint range of values.
// Values are then handed out by advancing the state's local `range`, avoiding
// a shared atomic operation per value. Dropping the reservation returns its
// unused suffix to `recycled`, and future states reuse those suffixes before
// advancing the atomic high-water mark. A reservation size of one retains the
// exact increment/read behavior needed by observable counters.
struct Counter {
    next: AtomicUsize,
    reservation_size: usize,
    recycled: SegQueue<std::ops::Range<usize>>,
}

impl Counter {
    fn new(reservation_size: usize) -> Self {
        assert!(reservation_size > 0);
        Self {
            next: AtomicUsize::new(0),
            reservation_size,
            recycled: SegQueue::new(),
        }
    }

    fn take_reservation(&self) -> std::ops::Range<usize> {
        if self.reservation_size == 1 {
            let start = self.next.fetch_add(1, Ordering::Release);
            return start..start + 1;
        }
        self.recycled.pop().unwrap_or_else(|| {
            let start = self
                .next
                .fetch_add(self.reservation_size, Ordering::Release);
            start..start + self.reservation_size
        })
    }
}

pub(crate) struct CounterReservation {
    counter: Arc<Counter>,
    range: std::ops::Range<usize>,
}

impl CounterReservation {
    fn new(counter: Arc<Counter>) -> Self {
        let range = counter.take_reservation();
        Self { counter, range }
    }

    pub(crate) fn next(&mut self) -> usize {
        if self.range.start == self.range.end {
            self.range = self.counter.take_reservation();
        }
        let result = self.range.start;
        self.range.start += 1;
        result
    }
}

impl Drop for CounterReservation {
    fn drop(&mut self) {
        if !self.range.is_empty() {
            self.counter.recycled.push(self.range.clone());
        }
    }
}

#[derive(Default)]
pub(crate) struct Counters(DenseIdMap<CounterId, Arc<Counter>>);

impl Clone for Counters {
    fn clone(&self) -> Counters {
        let mut map = DenseIdMap::new();
        for (k, v) in self.0.iter() {
            // NB: we may want to experiment with Ordering::Relaxed here.
            let cloned = Counter {
                next: AtomicUsize::new(v.next.load(Ordering::SeqCst)),
                reservation_size: v.reservation_size,
                // The high-water mark already includes every recycled range,
                // so omitting the free list is safe and avoids sharing
                // reservations between independent database snapshots.
                recycled: SegQueue::new(),
            };
            map.insert(k, Arc::new(cloned));
        }
        Counters(map)
    }
}

impl Counters {
    pub(crate) fn read(&self, ctr: CounterId) -> usize {
        self.0[ctr].next.load(Ordering::Acquire)
    }
    pub(crate) fn inc(&self, ctr: CounterId) -> usize {
        // We synchronize with `read_counter` but not with other increments.
        // NB: we may want to experiment with Ordering::Relaxed here.
        self.0[ctr].next.fetch_add(1, Ordering::Release)
    }
    pub(crate) fn take_reservation(&self, ctr: CounterId) -> CounterReservation {
        CounterReservation::new(Arc::clone(&self.0[ctr]))
    }
}

/// A collection of tables and indexes over them.
///
/// A database also owns the memory pools used by its tables.
#[derive(Default)]
pub struct Database {
    // NB: some fields are pub(crate) to allow some internal modules to avoid
    // borrowing the whole table.
    //
    // TableId is shared with container tables, so this relation-only map may
    // contain holes occupied by `ContainerValues`.
    pub(crate) tables: DenseIdMap<TableId, TableInfo>,
    // Reservable counters amortize shared atomic increments across an
    // ExecutionState. Exact counters retain one atomic increment per value.
    pub(crate) counters: Counters,
    pub(crate) external_functions: ExternalFunctions,
    container_values: ContainerValues,
    /// Next id in the storage namespace shared by relations and both
    /// container backends. Legacy containers deliberately consume an id
    /// without becoming dependency-graph participants.
    next_storage_id: usize,
    /// Participants modified since the last call to [`Database::merge_all`].
    notification_list: NotificationList<TableId>,
    // Tracks relative dependencies between maintenance participants.
    deps: DependencyGraph,
    base_values: BaseValues,
    /// A rough estimate of the total size of the database.
    ///
    /// This is primarily used to determine whether or not to attempt to do some operations in
    /// parallel.
    total_size_estimate: usize,
}

/// A maintenance participant temporarily removed from its typed owner map.
///
/// This is only an ownership adapter for constructing a read-only database
/// view while a mixed relation/container stratum is merged. Persistent
/// identity, notifications, buffers, and dependencies all use [`TableId`].
enum OwnedParticipant {
    Relation {
        id: TableId,
        info: TableInfo,
    },
    Container {
        id: crate::ContainerValueId,
        env: Box<dyn DynamicContainerEnv + Send + Sync>,
    },
}

struct ParticipantWork {
    participant: OwnedParticipant,
    buffers: DenseIdMap<TableId, Box<dyn MutationBuffer>>,
}

#[derive(Default)]
struct ParticipantRebuild {
    changed: bool,
    containers: ContainerRebuildSummary,
}

impl ParticipantRebuild {
    fn extend(&mut self, other: Self) {
        self.changed |= other.changed;
        self.containers.extend(other.containers);
    }
}

impl OwnedParticipant {
    fn maintenance_table_mut(&mut self) -> &mut dyn MaintenanceTable {
        match self {
            OwnedParticipant::Relation { info, .. } => &mut info.table,
            OwnedParticipant::Container { env, .. } => env
                .maintenance_table_mut()
                .expect("scheduled containers must use sequence storage"),
        }
    }

    fn merge(&mut self, exec_state: &mut ExecutionState<'_>) -> bool {
        let change = self.maintenance_table_mut().merge(exec_state);
        change.added || change.removed || exec_state.changed
    }

    fn apply_rebuild(
        &mut self,
        table_id: TableId,
        table: &WrappedTable,
        next_ts: Value,
        container_context: Option<&ContainerRebuildContext<'_>>,
        exec_state: &mut ExecutionState<'_>,
    ) -> ParticipantRebuild {
        match self {
            OwnedParticipant::Relation { info, .. } => ParticipantRebuild {
                changed: info
                    .table
                    .apply_rebuild(table_id, table, next_ts, exec_state),
                containers: ContainerRebuildSummary::default(),
            },
            OwnedParticipant::Container { env, .. } => {
                let containers = container_context
                    .map(|context| context.apply(&mut **env, exec_state))
                    .unwrap_or_default();
                ParticipantRebuild {
                    changed: containers.changed(),
                    containers,
                }
            }
        }
    }
}

impl Clone for Database {
    fn clone(&self) -> Self {
        // Table/container storage owns a deep copy of its pending mutation
        // epochs, so the scheduler queue must be independent too. The normal
        // `NotificationList::clone` shares one queue; either database could
        // then consume the other's only wakeup. Conservatively scheduling all
        // copied participants preserves every pending epoch without mutating
        // the source queue, and harmlessly performs one empty merge for
        // participants that had no work.
        let notification_list = NotificationList::default();
        for participant in self.deps.participants() {
            notification_list.notify(participant);
        }
        Self {
            tables: self.tables.clone(),
            counters: self.counters.clone(),
            external_functions: self.external_functions.clone(),
            container_values: self.container_values.clone(),
            next_storage_id: self.next_storage_id,
            notification_list,
            deps: self.deps.clone(),
            base_values: self.base_values.clone(),
            total_size_estimate: self.total_size_estimate,
        }
    }
}

impl Database {
    /// Create an empty Database.
    ///
    /// Queries use the currently installed egglog thread pool. If no pool is
    /// installed, queries run single-threaded.
    pub fn new() -> Database {
        Database::default()
    }

    /// Initialize a new rulse set to run against this database.
    pub fn new_rule_set(&mut self) -> RuleSetBuilder<'_> {
        RuleSetBuilder::new(self)
    }

    /// Add a new external function to the database.
    pub fn add_external_function(
        &mut self,
        f: Box<dyn ExternalFunction + 'static>,
    ) -> ExternalFunctionId {
        self.external_functions.push(f)
    }

    /// Free an existing external function. Make sure not to use `id` afterwards.
    pub fn free_external_function(&mut self, id: ExternalFunctionId) {
        self.external_functions.take(id);
    }

    pub fn base_values(&self) -> &BaseValues {
        &self.base_values
    }

    pub fn base_values_mut(&mut self) -> &mut BaseValues {
        &mut self.base_values
    }

    pub fn container_values(&self) -> &ContainerValues {
        &self.container_values
    }

    pub fn container_values_mut(&mut self) -> &mut ContainerValues {
        &mut self.container_values
    }

    /// Register a legacy container in the database-wide storage namespace.
    ///
    /// Legacy environments keep their existing compatibility rebuild loop and
    /// are not maintenance-scheduler participants. Allocating their ids here
    /// nevertheless prevents a later relation or sequence container from
    /// reusing the same physical slot while both backends coexist.
    pub fn register_container_type<C: ContainerValue>(
        &mut self,
        id_counter: CounterId,
        merge_fn: impl Fn(&mut ExecutionState, Value, Value) -> Value + Clone + Send + Sync + 'static,
    ) -> crate::ContainerValueId {
        if let Some(container) = self.container_values.registered_type::<C>() {
            return container;
        }

        let participant = self.allocate_storage_id();
        let container = crate::ContainerValueId::from_table_id(participant);
        let registered = self
            .container_values
            .register_type::<C>(container, id_counter, merge_fn);
        assert_eq!(registered, container);
        container
    }

    /// Register a sequence-backed container as a normal database maintenance
    /// participant. Its packed rows remain outside the fixed-schema query
    /// table interface, but merge notifications and dependency ordering share
    /// the same scheduler namespace as relation tables.
    ///
    /// Registration is idempotent by concrete Rust type. The first call fixes
    /// that type's counter, merge callback, and dependency set; later calls
    /// return the existing [`crate::ContainerValueId`].
    ///
    /// Every read and write dependency must already be registered in this
    /// database. Container read dependencies must use the sequence backend.
    pub fn register_sequence_container_type<C: SequenceContainerValue>(
        &mut self,
        id_counter: CounterId,
        merge_fn: impl Fn(&mut ExecutionState, Value, Value) -> Value + Clone + Send + Sync + 'static,
        read_deps: impl IntoIterator<Item = TableId>,
        write_deps: impl IntoIterator<Item = TableId>,
    ) -> crate::ContainerValueId {
        if let Some(container) = self.container_values.registered_sequence_type::<C>() {
            return container;
        }

        let read_deps = read_deps.into_iter().collect::<Vec<_>>();
        let write_deps = write_deps.into_iter().collect::<Vec<_>>();
        self.validate_dependencies(&read_deps, "read");
        self.validate_dependencies(&write_deps, "write");

        let base_values = self.base_values.clone();
        let participant = self.allocate_storage_id();
        let container = crate::ContainerValueId::from_table_id(participant);
        let registered = self.container_values.register_sequence_type::<C>(
            container,
            id_counter,
            merge_fn,
            base_values,
        );
        assert_eq!(registered, container);
        self.deps
            .add_participant(participant, read_deps, write_deps);
        container
    }

    fn validate_dependencies(&self, dependencies: &[TableId], kind: &str) {
        for participant in dependencies {
            assert!(
                self.deps.contains(*participant),
                "maintenance {kind} dependency {participant:?} is not registered"
            );
        }
    }

    fn allocate_storage_id(&mut self) -> TableId {
        let id = TableId::from_usize(self.next_storage_id);
        self.next_storage_id = self
            .next_storage_id
            .checked_add(1)
            .expect("database storage id space exhausted");
        assert!(!id.is_dummy(), "database storage id space exhausted");
        id
    }

    /// Apply one database-maintenance round using the value-level rebuild
    /// encoded by `func_id`.
    ///
    /// Sequence-backed containers use the same notification/dependency
    /// scheduler and rebuild traversal as relation tables. Unmigrated legacy
    /// environments retain their compatibility pass, but consume the same
    /// prepared union-find delta. Both kinds of container rebuild before the
    /// publication barrier because canonicalizing their keys can create new
    /// identities in the rebuild source.
    ///
    /// If publication changes the source, relation rebuilding is deferred to
    /// the caller's next outer round. Stable container identities additionally
    /// require the post-publication dirty-parent refresh below.
    pub fn apply_rebuild(
        &mut self,
        func_id: TableId,
        to_rebuild: &[TableId],
        next_ts: Value,
    ) -> bool {
        let containers = self
            .container_values
            .sequence_env_ids()
            .into_iter()
            .map(TableId::from)
            .collect::<Vec<_>>();
        let relations = to_rebuild.to_vec();
        assert!(
            !containers.contains(&func_id) && !relations.contains(&func_id),
            "the rebuild source cannot also be a mutable rebuild participant"
        );
        let container_plan = self
            .container_values
            .prepare_rebuild_plan(func_id, &self.tables[func_id].table);
        let source_before = self.tables[func_id].table.version();

        let mut rebuild = self.rebuild_batch(containers, func_id, next_ts, container_plan.as_ref());

        // Legacy environments are intentionally outside the common scheduler,
        // but they must observe exactly the same incremental UF subset as the
        // sequence participants. Keep their old pass until the final legacy
        // backend is migrated.
        if let Some(plan) = container_plan.as_ref() {
            let mut container_values = mem::take(&mut self.container_values);
            let (legacy, staged) = {
                let source = &self.tables[func_id].table;
                let context = plan
                    .context(source)
                    .expect("a prepared container rebuild plan must remain applicable");
                self.with_execution_state_tracked(|state| {
                    container_values.rebuild_legacy_pass(&context, state)
                })
            };
            self.container_values = container_values;
            rebuild.changed |= legacy.changed() || staged;
            rebuild.containers.extend(legacy);
        }

        // A container collision can add an identity union to `func_id`.
        // Publish it before constructing relation rebuilders or mutation
        // buffers; otherwise a relation can collapse its key while retaining
        // two still-distinct container values and spuriously invoke :no-merge.
        rebuild.changed |= self.merge_all();
        if self.tables[func_id].table.version() != source_before {
            // A nested container can expose another collision only after this
            // source delta is visible. Preserve stable-id notifications from
            // this round, then let the existing outer rebuild loop run the
            // container batch again before any fixed-table observer proceeds.
            self.refresh_dirty_container_parents(to_rebuild, next_ts, &mut rebuild);
            return true;
        }
        rebuild.extend(self.rebuild_batch(relations, func_id, next_ts, None));

        // Publish every rebuilt participant before dirty-id refresh scans the
        // relation tables; otherwise it can re-stage an obsolete pre-rebuild
        // row.
        rebuild.changed |= self.merge_all();
        self.refresh_dirty_container_parents(to_rebuild, next_ts, &mut rebuild);
        rebuild.changed
    }

    fn refresh_dirty_container_parents(
        &mut self,
        to_rebuild: &[TableId],
        next_ts: Value,
        rebuild: &mut ParticipantRebuild,
    ) {
        self.container_values
            .finish_rebuild(&mut rebuild.containers);
        let dirty_ids = rebuild
            .containers
            .dirty_ids()
            .iter()
            .copied()
            .collect::<Vec<_>>();
        if dirty_ids.is_empty() {
            return;
        }
        self.run_on_tables(to_rebuild, |_, info, _| {
            info.table.refresh_rows_for_values(&dirty_ids, next_ts)
        });
        rebuild.changed |= self.merge_all();
    }

    /// Order one homogeneous rebuild batch and run it through the common
    /// fixed-/variable-arity participant implementation.
    ///
    /// Destination buffers are constructed after any preceding publication
    /// barrier, so none can retain a detached pending epoch across that barrier.
    fn rebuild_batch(
        &mut self,
        mut participants: Vec<TableId>,
        func_id: TableId,
        next_ts: Value,
        container_plan: Option<&ContainerRebuildPlan>,
    ) -> ParticipantRebuild {
        participants.sort_unstable();
        participants.dedup();
        participants.sort_unstable_by_key(|participant| self.deps.level(*participant));

        self.rebuild_participants(&participants, func_id, next_ts, container_plan)
    }

    /// Rebuild a mixed set of fixed- and variable-arity participants through
    /// one dependency-ordered traversal.
    ///
    /// Only one read-dependency stratum is detached at a time. Consequently a
    /// callback can still read every declared predecessor, including the
    /// rebuild source itself, while same-level participants remain eligible
    /// for outer parallelism.
    fn rebuild_participants(
        &mut self,
        scheduled: &[TableId],
        func_id: TableId,
        next_ts: Value,
        container_plan: Option<&ContainerRebuildPlan>,
    ) -> ParticipantRebuild {
        let mut result = ParticipantRebuild::default();
        let mut stratum_start = 0usize;
        while stratum_start < scheduled.len() {
            let level = self.deps.level(scheduled[stratum_start]);
            let stratum_end = scheduled[stratum_start..]
                .iter()
                .position(|participant| self.deps.level(*participant) != level)
                .map_or(scheduled.len(), |offset| stratum_start + offset);
            let stratum = &scheduled[stratum_start..stratum_end];

            let (rows, largest) =
                stratum
                    .iter()
                    .fold((0usize, 0usize), |(rows, largest), participant| {
                        let len = self.maintenance_table(*participant).len();
                        (rows.saturating_add(len), largest.max(len))
                    });
            let do_parallel =
                stratum.len() > 1 && !parallelize_rebuild(largest) && parallelize_db_level_op(rows);

            let rebuild = |db: DbView<'_>, work: &mut ParticipantWork| {
                let source = &db.table_info[func_id].table;
                let container_context = container_plan.and_then(|plan| plan.context(source));
                let mut exec_state = ExecutionState::new(db, mem::take(&mut work.buffers));
                let mut outcome = work.participant.apply_rebuild(
                    func_id,
                    source,
                    next_ts,
                    container_context.as_ref(),
                    &mut exec_state,
                );
                outcome.changed |= exec_state.changed;
                outcome
            };
            let outcomes = if do_parallel {
                self.with_participants(stratum, |db, rebuilding| {
                    parallel::map_mut(rebuilding, |_, work| rebuild(db, work))
                })
            } else {
                // A serial rebuild does not need to detach an entire stratum.
                // Keeping installed destinations in the DbView lets their
                // mutation buffers remain lazy, matching ordinary rule
                // execution and avoiding per-stratum setup proportional to the
                // number of declared write dependencies.
                stratum
                    .iter()
                    .map(|participant| {
                        self.with_participants(std::slice::from_ref(participant), |db, work| {
                            rebuild(db, &mut work[0])
                        })
                    })
                    .collect::<Vec<_>>()
            };

            for (participant, outcome) in stratum.iter().copied().zip(outcomes) {
                if outcome.changed {
                    self.notification_list.notify(participant);
                }
                result.extend(outcome);
            }
            stratum_start = stratum_end;
        }
        result
    }

    fn run_on_tables(
        &mut self,
        table_ids: &[TableId],
        run: impl for<'a> Fn(TableId, &mut TableInfo, &DbView<'a>) -> bool + Sync,
    ) {
        if parallelize_db_level_op(self.total_size_estimate) {
            let mut tables = Vec::with_capacity(table_ids.len());
            for id in table_ids {
                tables.push((*id, self.tables.take(*id).unwrap()));
            }
            let view = self.read_only_view();
            parallel::for_each_mut(&mut tables, |_, (id, info)| {
                if run(*id, info, &view) {
                    self.notification_list.notify(*id);
                }
            });
            for (id, info) in tables {
                self.tables.insert(id, info);
            }
        } else {
            for id in table_ids {
                let mut info = self.tables.take(*id).unwrap();
                let changed = {
                    let view = self.read_only_view();
                    run(*id, &mut info, &view)
                };
                if changed {
                    self.notification_list.notify(*id);
                }
                self.tables.insert(*id, info);
            }
        }
    }

    /// Run `f` with access to an `ExecutionState` mapped to this database.
    pub fn with_execution_state<R>(&self, f: impl FnOnce(&mut ExecutionState) -> R) -> R {
        let mut state = ExecutionState::new(self.read_only_view(), Default::default());
        f(&mut state)
    }

    /// Like [`Database::with_execution_state`], but also reports whether `f`
    /// staged any mutation through the execution state. Callers can use the
    /// flag to skip a subsequent `merge_all` when the closure was read-only.
    pub fn with_execution_state_tracked<R>(
        &self,
        f: impl FnOnce(&mut ExecutionState) -> R,
    ) -> (R, bool) {
        let mut state = ExecutionState::new(self.read_only_view(), Default::default());
        let result = f(&mut state);
        (result, state.changed)
    }

    pub(crate) fn read_only_view(&self) -> DbView<'_> {
        DbView {
            table_info: &self.tables,
            counters: &self.counters,
            external_funcs: &self.external_functions,
            bases: &self.base_values,
            containers: &self.container_values,
            notification_list: &self.notification_list,
        }
    }

    /// Estimate the size of the table. If a constraint is provided, return an
    /// estimate of the size of the subset of the table matching the constraint.
    pub fn estimate_size(&self, table: TableId, c: Option<Constraint>) -> usize {
        let table_info = self
            .tables
            .get(table)
            .expect("table must be declared in the current database");
        let table = &table_info.table;
        if let Some(c) = c {
            if let Some(sub) = table.fast_subset(&c) {
                // In the case where a the constraint can be computed quickly,
                // we do not filter for staleness, which may over-approximate.
                sub.size()
            } else {
                table.refine_one(table.refine_live(table.all()), &c).size()
            }
        } else {
            table.len()
        }
    }

    /// Create a new counter for this database.
    ///
    /// These counters can be used to generate unique ids as part of an action.
    pub fn add_counter(&mut self) -> CounterId {
        self.counters.0.push(Arc::new(Counter::new(1)))
    }

    /// Create a counter whose increments from one [`ExecutionState`] are
    /// allocated in local reservations.
    ///
    /// This is intended for fresh identifiers, where uniqueness matters but a
    /// concurrent read need not equal the number of identifiers already
    /// returned. Ordinary counters created by [`Database::add_counter`] retain
    /// exact increment/read behavior.
    ///
    /// # Panics
    ///
    /// Panics if `reservation_size` is zero.
    pub fn add_reservable_counter(&mut self, reservation_size: usize) -> CounterId {
        self.counters
            .0
            .push(Arc::new(Counter::new(reservation_size)))
    }

    /// Increment the given counter and return its previous value.
    pub fn inc_counter(&self, counter: CounterId) -> usize {
        self.counters.inc(counter)
    }

    /// Get the current value of the given counter.
    pub fn read_counter(&self, counter: CounterId) -> usize {
        self.counters.read(counter)
    }

    /// A helper for merging all pending updates. Used to write to the database after updates have
    /// been staged. Returns true if any participant changed or staged downstream work.
    ///
    /// Exposed for testing purposes.
    ///
    /// Useful for out-of-band insertions into the database.
    pub fn merge_all(&mut self) -> bool {
        let mut ever_changed = false;
        let do_parallel = parallelize_db_level_op(self.total_size_estimate);
        let sequence_size_before = self.container_values.sequence_len();
        // Relation tables modified during this `merge_all` call. Only these need their cached
        // indexes reset at the end so future reads refresh them.
        let mut touched: IndexSet<TableId> = IndexSet::default();
        loop {
            let mut active = self.notification_list.reset();
            touched.extend(
                active
                    .iter()
                    .copied()
                    .filter(|participant| self.tables.contains_key(*participant)),
            );
            if active.len() < 4 {
                ever_changed |= self.merge_simple(active, &mut touched);
                break;
            }
            active.sort_unstable_by_key(|participant| self.deps.level(*participant));

            let mut changed = false;
            let mut stratum_start = 0;
            while stratum_start < active.len() {
                let level = self.deps.level(active[stratum_start]);
                let stratum_end = active[stratum_start..]
                    .iter()
                    .position(|participant| self.deps.level(*participant) != level)
                    .map_or(active.len(), |offset| stratum_start + offset);
                let stratum = &active[stratum_start..stratum_end];

                // Primitive merge callbacks can read containers without those
                // reads appearing in the dependency graph. Preserve that
                // existing API contract by publishing each active container
                // before removing any same-stratum relations. Containers run
                // one at a time so every other container also remains visible.
                // This is only a visibility phase: both kinds still use the
                // same ids, dependency graph, buffers, notifications, and
                // participant merge implementation.
                let containers = stratum
                    .iter()
                    .copied()
                    .filter(|participant| !self.tables.contains_key(*participant))
                    .collect::<Vec<_>>();
                let relations = stratum
                    .iter()
                    .copied()
                    .filter(|participant| self.tables.contains_key(*participant))
                    .collect::<Vec<_>>();
                for participant in containers {
                    changed |= self.merge_participants(std::slice::from_ref(&participant), false);
                }
                changed |= self.merge_participants(&relations, do_parallel);
                stratum_start = stratum_end;
            }
            ever_changed |= changed;
        }
        // Reset the cached indexes of only the tables modified during this call so
        // they refresh on next access; unmodified tables keep their still-valid
        // cached indexes. `touched` must contain *every* table whose version bumped
        // this call: `ResettableOnceLock::get_or_update` runs the index `refresh`
        // only after a `reset()`, so a modified-but-unreset table would keep serving
        // a stale cached index. It does — every merged table comes from
        // `notification_list.reset()`, which is exactly what `touched` accumulates.
        // Relation size is maintained incrementally at each merge (above and in
        // `merge_simple`), so we no longer re-sum every table here.
        for table in touched.iter().copied() {
            if let Some(info) = self.tables.get_mut(table) {
                info.column_indexes.update(|_, ti| {
                    Arc::get_mut(ti).unwrap().reset();
                });
                info.indexes.update(|_, ti| {
                    Arc::get_mut(ti).unwrap().reset();
                });
            }
        }
        self.total_size_estimate = self
            .total_size_estimate
            .wrapping_sub(sequence_size_before)
            .wrapping_add(self.container_values.sequence_len());
        ever_changed
    }

    fn maintenance_table(&self, participant: TableId) -> &dyn MaintenanceTable {
        if let Some(info) = self.tables.get(participant) {
            return &info.table;
        }
        self.container_values
            .maintenance_table(crate::ContainerValueId::from_table_id(participant))
            .unwrap_or_else(|| panic!("maintenance participant {participant:?} has no storage"))
    }

    fn write_buffers(
        &self,
        participant: TableId,
        detached: Option<&DenseIdMap<TableId, ()>>,
    ) -> DenseIdMap<TableId, Box<dyn MutationBuffer>> {
        let mut buffers = DenseIdMap::default();
        for dependency in self.deps.write_deps(participant) {
            let is_detached = detached.map_or(dependency == participant, |participants| {
                participants.contains_key(dependency)
            });
            if is_detached {
                buffers.insert(dependency, self.maintenance_table(dependency).new_buffer());
            }
        }
        buffers
    }

    /// Temporarily own a participant group while retaining a read-only view of
    /// every other table.
    ///
    /// Merge and rebuild both use this ownership protocol. Only destinations
    /// in the detached group need buffers created up front; installed relation
    /// and container destinations are initialized lazily through [`DbView`].
    /// This keeps same-group writes attached to the destination's current
    /// pending epoch without paying for declared writes that never occur.
    fn with_participants<R>(
        &mut self,
        participants: &[TableId],
        run: impl for<'a> FnOnce(DbView<'a>, &mut [ParticipantWork]) -> R,
    ) -> R {
        if let [participant] = participants {
            let buffers = self.write_buffers(*participant, None);
            let mut work = ParticipantWork {
                participant: self.take_participant(*participant),
                buffers,
            };
            let result = run(self.read_only_view(), std::slice::from_mut(&mut work));
            self.put_participant(work.participant);
            return result;
        }

        let mut detached = DenseIdMap::with_capacity(participants.len());
        for participant in participants {
            detached.insert(*participant, ());
        }
        let buffers = participants
            .iter()
            .map(|participant| self.write_buffers(*participant, Some(&detached)))
            .collect::<Vec<_>>();
        let mut work = participants
            .iter()
            .copied()
            .zip(buffers)
            .map(|(participant, buffers)| ParticipantWork {
                participant: self.take_participant(participant),
                buffers,
            })
            .collect::<Vec<_>>();
        let result = run(self.read_only_view(), &mut work);
        for work in work {
            self.put_participant(work.participant);
        }
        result
    }

    /// Merge one visibility-compatible group through the common participant
    /// interface, optionally in parallel.
    ///
    /// Declared destination buffers are created before any participant is
    /// removed. This retains access to a same-group destination's pending
    /// state while its storage is temporarily owned by another worker.
    fn merge_participants(&mut self, participants: &[TableId], do_parallel: bool) -> bool {
        self.with_participants(participants, |db, merging| {
            let merge_one = |work: &mut ParticipantWork| {
                let mut es = ExecutionState::new(db, mem::take(&mut work.buffers));
                work.participant.merge(&mut es)
            };
            if do_parallel {
                parallel::map_mut(merging, |_, work| merge_one(work))
                    .into_iter()
                    .any(|changed| changed)
            } else {
                merging.iter_mut().map(merge_one).max().unwrap_or(false)
            }
        })
    }

    fn take_participant(&mut self, participant: TableId) -> OwnedParticipant {
        if let Some(info) = self.tables.take(participant) {
            self.total_size_estimate = self.total_size_estimate.wrapping_sub(info.table.len());
            return OwnedParticipant::Relation {
                id: participant,
                info,
            };
        }
        let id = crate::ContainerValueId::from_table_id(participant);
        OwnedParticipant::Container {
            id,
            env: self.container_values.take_env(id),
        }
    }

    fn put_participant(&mut self, participant: OwnedParticipant) {
        match participant {
            OwnedParticipant::Relation { id, info } => {
                self.total_size_estimate = self.total_size_estimate.wrapping_add(info.table.len());
                let old = self.tables.insert(id, info);
                assert!(old.is_none(), "relation maintenance storage inserted twice");
            }
            OwnedParticipant::Container { id, env } => {
                self.container_values.put_env(id, env);
            }
        }
    }

    /// A serial fast path for a small number of notified participants.
    ///
    /// It follows read-dependency strata, but avoids taking a whole stratum out
    /// of the database. Declared write buffers are still initialized before
    /// detaching the participant, which also supports self-directed writes.
    fn merge_simple(
        &mut self,
        mut to_merge: SmallVec<[TableId; 4]>,
        touched: &mut IndexSet<TableId>,
    ) -> bool {
        let mut changed = false;
        while !to_merge.is_empty() {
            to_merge.sort_unstable_by_key(|participant| {
                let relation_order = usize::from(self.tables.contains_key(*participant));
                (self.deps.level(*participant), relation_order)
            });
            for participant in &to_merge {
                changed |= self.merge_participants(std::slice::from_ref(participant), false);
            }
            to_merge = self.notification_list.reset();
            touched.extend(
                to_merge
                    .iter()
                    .copied()
                    .filter(|participant| self.tables.contains_key(*participant)),
            );
        }
        changed
    }

    /// A low-level helper for merging pending updates to a particular function.
    ///
    /// Callers should prefer `merge_all`, as the process of merging the data
    /// for a particular table may cause other updates to be buffered
    /// elesewhere. The `merge_all` method runs merges to a fixed point to avoid
    /// surprises here.
    pub fn merge_table(&mut self, table: TableId) -> bool {
        let mut info = self.tables.unwrap_val(table);
        self.total_size_estimate = self.total_size_estimate.wrapping_sub(info.table.len());
        let table_changed = info.table.merge(&mut ExecutionState::new(
            self.read_only_view(),
            Default::default(),
        ));
        self.total_size_estimate = self.total_size_estimate.wrapping_add(info.table.len());
        self.tables.insert(table, info);
        table_changed.added
    }

    /// Get id of the next table to be added to the database.
    ///
    /// This can be useful for "knot tying", when tables need to reference their
    /// own id.
    pub fn next_table_id(&self) -> TableId {
        TableId::from_usize(self.next_storage_id)
    }

    /// Add a table with the given schema to the database.
    ///
    /// The table must have a compatible spec with `types` (e.g. same number of
    /// columns). Every read and write dependency must identify a maintenance
    /// participant already registered in this database.
    pub fn add_table<T: Table + Sized + 'static>(
        &mut self,
        table: T,
        read_deps: impl IntoIterator<Item = TableId>,
        write_deps: impl IntoIterator<Item = TableId>,
    ) -> TableId {
        self.add_table_impl(table, None, read_deps, write_deps)
    }

    /// Named variant of [`Database::add_table`].
    pub fn add_table_named<T: Table + Sized + 'static>(
        &mut self,
        table: T,
        name: Arc<str>,
        read_deps: impl IntoIterator<Item = TableId>,
        write_deps: impl IntoIterator<Item = TableId>,
    ) -> TableId {
        self.add_table_impl(table, Some(name), read_deps, write_deps)
    }

    fn add_table_impl<T: Table + Sized + 'static>(
        &mut self,
        table: T,
        name: Option<Arc<str>>,
        read_deps: impl IntoIterator<Item = TableId>,
        write_deps: impl IntoIterator<Item = TableId>,
    ) -> TableId {
        let read_deps = read_deps.into_iter().collect::<Vec<_>>();
        let write_deps = write_deps.into_iter().collect::<Vec<_>>();
        self.validate_dependencies(&read_deps, "read");
        self.validate_dependencies(&write_deps, "write");

        let spec = table.spec();
        let table = WrappedTable::new(table);
        let res = self.allocate_storage_id();
        let old = self.tables.insert(
            res,
            TableInfo {
                name,
                spec,
                table,
                indexes: IndexCatalog::new(),
                column_indexes: IndexCatalog::new(),
            },
        );
        assert!(
            old.is_none(),
            "global maintenance id already owns relation storage"
        );
        self.deps.add_participant(res, read_deps, write_deps);
        res
    }

    /// Get direct mutable access to the table.
    ///
    /// This method is useful for out-of-band access to databse state.
    ///
    /// **NOTE:** It is legal to call [`Table::new_buffer`] on the returned table handle, and use
    /// that to stage updates to the given table via [`MutationBuffer::stage_insert`] or
    /// [`MutationBuffer::stage_remove`], however this is *likely to be a source of bugs*.
    ///
    /// Updates staged in this way will not cause `table` to be marked as having pending changes in
    /// the next call to [`Database::merge_all`]. Instead, such users should use
    /// [`Database::new_buffer`], which plumbs this signal through correctly, or better yet,
    /// perform all updates through an [`ExecutionState`] or a [`crate::RuleBuilder`]. If these
    /// options do not work, then calling [`Database::merge_table`] directly will force a merge
    /// call on the table.
    pub fn get_table(&self, table: TableId) -> &WrappedTable {
        &self
            .tables
            .get(table)
            .expect("must access a table that has been declared in this database")
            .table
    }

    /// Get a handle on the given table along with metadata about it.
    ///
    ///
    /// **NOTE:** See the note on [`Database::get_table`] around manually staging updates.
    pub fn get_table_info(&self, table: TableId) -> &TableInfo {
        self.tables
            .get(table)
            .expect("must access a table that has been declared in this database")
    }

    /// Create a new mutation buffer for the table with id `id`.
    ///
    /// This will marked the given table as potentially changed for the next round of merging.
    /// Unlike calling [`Table::new_buffer`] on a table returned from a getter, this method also
    /// triggers change notification metadata that is read by [`Database::merge_all`].
    pub fn new_buffer(&self, id: TableId) -> Box<dyn MutationBuffer> {
        self.notification_list.notify(id);
        self.get_table(id).new_buffer()
    }

    pub(crate) fn process_constraints(
        &self,
        table: TableId,
        cs: &[Constraint],
    ) -> ProcessedConstraints {
        let table_info = &self.tables[table];
        let (mut subset, mut fast, mut slow) = table_info.table.split_fast_slow(cs);
        slow.retain(|c| {
            let (col, val) = match c {
                Constraint::EqConst { col, val } => (*col, *val),
                Constraint::Eq { .. }
                | Constraint::LtConst { .. }
                | Constraint::GtConst { .. }
                | Constraint::LeConst { .. }
                | Constraint::GeConst { .. } => return true,
            };
            // We are looking up by a constant: this is something we can build
            // an index for as long as the column is cacheable.
            if *table_info
                .spec
                .uncacheable_columns
                .get(col)
                .unwrap_or(&false)
            {
                return true;
            }
            // We have or will build an index: upgrade this constraint to
            // 'fast'.
            fast.push(c.clone());
            let index = get_column_index_from_tableinfo(table_info, col);
            match index.get().unwrap().get_subset(&val) {
                Some(s) => {
                    with_pool_set(|ps| subset.intersect(s, &ps.get_pool()));
                }
                None => {
                    // There are no rows matching this key! We can constrain this to nothing.
                    subset = Subset::empty();
                }
            }
            // Remove this constraint from the slow list.
            false
        });
        ProcessedConstraints { subset, fast, slow }
    }

    /// Get direct mutable access to the table.
    ///
    /// This method is useful for out-of-band access to databse state.
    ///
    /// **NOTE:** See the warning around staging updates to handles returned through this method in
    /// the documentation for [`Database::get_table`].
    pub fn get_table_mut(&mut self, id: TableId) -> &mut dyn Table {
        &mut *self
            .tables
            .get_mut(id)
            .expect("must access a table that has been declared in this database")
            .table
    }

    /// Remove every row from the given table.
    ///
    /// This is intended as a faster alternative to staging a per-row
    /// `stage_remove` for every key in the table. The underlying [`Table::clear`]
    /// implementation drops the row storage in bulk and bumps the table's major
    /// generation, so any cached indexes/subsets observed by future readers will
    /// be lazily rebuilt against the now-empty table. Any pending staged
    /// inserts or removes for this table are dropped (they pre-dated the clear,
    /// so they no longer make sense once the table is empty).
    ///
    /// This method also resets the cached column- and key-indexes for the
    /// table so subsequent merges can take the `Arc::get_mut`-based reset path,
    /// matching the invariant maintained by [`Database::merge_all`].
    ///
    /// This does **not** flush pending changes for *other* tables; it is the
    /// caller's responsibility to call [`Database::merge_all`] beforehand if
    /// they need staged updates from a previous step to land before the clear.
    pub fn clear_table(&mut self, table: TableId) {
        let info = self
            .tables
            .get_mut(table)
            .expect("must access a table that has been declared in this database");
        let prev_len = info.table.len();
        info.table.clear();
        // The version bump from `clear` is enough on its own to make the
        // indexes self-refresh on next access (see `Index::refresh`). We still
        // reset them eagerly here so that the next `merge_all` sees the same
        // "indexes are resettable" state it expects after a successful merge.
        info.column_indexes.update(|_, ti| {
            if let Some(arc) = Arc::get_mut(ti) {
                arc.reset();
            }
        });
        info.indexes.update(|_, ti| {
            if let Some(arc) = Arc::get_mut(ti) {
                arc.reset();
            }
        });
        self.total_size_estimate = self.total_size_estimate.wrapping_sub(prev_len);
    }

    pub(crate) fn plan_query(&mut self, query: Query) -> Plan {
        plan::plan_query(query, ColumnCardEst::new(self))
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        // Clean up this thread's ambient memory pool.
        with_pool_set(PoolSet::clear);
    }
}

/// The core logic behind getting and updating a hash index.
///
/// This is in a separate function to allow us to reuse it while already
/// borrowing a `TableInfo`.
fn get_index_from_tableinfo(table_info: &TableInfo, cols: &[ColumnId]) -> HashIndex {
    let index: Arc<_> = table_info.indexes.get_or_insert(cols.into(), || {
        Arc::new(ResettableOnceLock::new(Index::new(
            cols.to_vec(),
            TupleIndex::new(cols.len()),
        )))
    });
    index.get_or_update(|index| {
        index.refresh(table_info.table.as_ref());
    });
    debug_assert!(
        !index
            .get()
            .unwrap()
            .needs_refresh(table_info.table.as_ref())
    );
    index
}

/// The core logic behind getting and updating a column index.
///
/// This is the single-column analog to [`get_index_from_tableinfo`].
fn get_column_index_from_tableinfo(table_info: &TableInfo, col: ColumnId) -> HashColumnIndex {
    let index: Arc<_> = table_info.column_indexes.get_or_insert(col, || {
        Arc::new(ResettableOnceLock::new(Index::new(
            vec![col],
            ColumnIndex::new(),
        )))
    });
    index.get_or_update(|index| {
        index.refresh(table_info.table.as_ref());
    });
    debug_assert!(
        !index
            .get()
            .unwrap()
            .needs_refresh(table_info.table.as_ref())
    );
    index
}

#[derive(Clone)]
pub struct ColumnCardEst<'a> {
    db: &'a Database,
}

impl ColumnCardEst<'_> {
    pub fn new(db: &Database) -> ColumnCardEst<'_> {
        ColumnCardEst { db }
    }

    pub fn col_uniqueness(&self, table: TableId, col: ColumnId) -> ColUniqueness {
        let col_idx = get_column_index_from_tableinfo(&self.db.tables[table], col);
        let table = &self.db.tables[table].table;
        ColUniqueness {
            col_size: col_idx.get().unwrap().len(),
            table_size: table.len(),
        }
    }
}

impl std::fmt::Debug for ColumnCardEst<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColumnCardEst").finish_non_exhaustive()
    }
}

/// A coarse cardinality estimate for a column of a table, used by the query
/// planner to decide which variable to eliminate next during tree
/// decomposition.
///
/// `table_size` is the number of rows in the (sub)table and `col_size` is the
/// number of distinct values in the column. Their ratio
/// (`table_size / col_size`) approximates the average number of rows that share
/// a given value of the column: a smaller ratio means the column is closer to
/// being unique and therefore cheaper to join on. [`ColUniqueness`] is ordered
/// by this ratio (see the [`Ord`] impl), so the planner prefers variables with
/// the most selective (most unique) columns.
#[derive(Copy, Clone, Debug)]
pub struct ColUniqueness {
    table_size: usize,
    col_size: usize,
}

impl Default for ColUniqueness {
    fn default() -> ColUniqueness {
        ColUniqueness {
            table_size: 1,
            col_size: 1,
        }
    }
}

impl ColUniqueness {
    #[allow(dead_code)] // not yet wired up into the planner
    fn scale(&self, subset_size: usize) -> ColUniqueness {
        if self.table_size == 0 || subset_size == 0 {
            return ColUniqueness {
                table_size: 0,
                col_size: 0,
            };
        }
        ColUniqueness {
            table_size: subset_size,
            col_size: self.col_size.saturating_mul(subset_size) / self.table_size,
        }
    }
    fn join(&self, other: &ColUniqueness) -> ColUniqueness {
        ColUniqueness {
            table_size: self.table_size.saturating_mul(other.table_size),
            col_size: self.col_size.max(other.col_size),
        }
    }

    #[allow(dead_code)] // not yet wired up into the planner
    fn col_size(&self) -> usize {
        self.col_size
    }
}

impl PartialEq for ColUniqueness {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for ColUniqueness {}

impl PartialOrd for ColUniqueness {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ColUniqueness {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.table_size.saturating_mul(other.col_size))
            .cmp(&(other.table_size.saturating_mul(self.col_size)))
    }
}
