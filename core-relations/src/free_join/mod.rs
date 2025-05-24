//! Execute queries against a database using a variant of Free Join.
use std::{
    mem,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use concurrency::ReadOptimizedLock;
use numeric_id::{define_id, DenseIdMap, DenseIdMapWithReuse, NumericId};
use rayon::prelude::*;
use smallvec::SmallVec;

use crate::{
    action::{
        mask::{Mask, MaskIter, ValueSource},
        Bindings, DbView,
    },
    common::DashMap,
    dependency_graph::DependencyGraph,
    hash_index::{ColumnIndex, Index},
    offsets::Subset,
    pool::{with_pool_set, Pool, Pooled},
    primitives::Primitives,
    query::{Query, RuleSetBuilder},
    table_spec::{ColumnId, Constraint, MutationBuffer, Table, TableSpec, WrappedTable},
    Containers, PoolSet, QueryEntry, TupleIndex, Value,
};

use self::plan::Plan;
use crate::action::{ExecutionState, PredictedVals};

pub(crate) mod execute;
pub(crate) mod plan;

define_id!(
    pub(crate) AtomId,
    u32,
    "A component of a query consisting of a function and a list of variables or constants"
);
define_id!(pub Variable, u32, "a variable in a query");

impl Variable {
    pub fn placeholder() -> Variable {
        Variable::new(!0)
    }
}

define_id!(pub TableId, u32, "a table in the database");
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

#[derive(Debug)]
pub(crate) struct VarInfo {
    pub(crate) occurrences: Vec<SubAtom>,
    /// Whether or not this variable shows up in the "actions" portion of a
    /// rule.
    pub(crate) used_in_rhs: bool,
}

pub(crate) type HashIndex = Arc<ReadOptimizedLock<Index<TupleIndex>>>;
pub(crate) type HashColumnIndex = Arc<ReadOptimizedLock<Index<ColumnIndex>>>;

pub struct TableInfo {
    pub(crate) spec: TableSpec,
    pub(crate) table: WrappedTable,
    pub(crate) indexes: DashMap<SmallVec<[ColumnId; 4]>, HashIndex>,
    pub(crate) column_indexes: DashMap<ColumnId, HashColumnIndex>,
}

impl Clone for TableInfo {
    fn clone(&self) -> Self {
        fn deep_clone_map<K: Clone + std::hash::Hash + Eq, TI: Clone>(
            map: &DashMap<K, Arc<ReadOptimizedLock<TI>>>,
        ) -> DashMap<K, Arc<ReadOptimizedLock<TI>>> {
            map.iter()
                .map(|table_ref| {
                    let (k, v) = table_ref.pair();
                    (
                        k.clone(),
                        Arc::new(ReadOptimizedLock::new(v.read().clone())),
                    )
                })
                .collect()
        }
        TableInfo {
            spec: self.spec.clone(),
            table: self.table.dyn_clone(),
            indexes: deep_clone_map(&self.indexes),
            column_indexes: deep_clone_map(&self.column_indexes),
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

pub(crate) trait ExternalFunctionExt: ExternalFunction {
    /// A vectorized variant of `invoke` to avoid repeated dynamic dispatch.
    ///
    /// Implementors should not override this manually (in fact, they shouldn't
    /// even be able to; some types are private); the default implementation
    /// delegates core logic to `invoke`.
    #[doc(hidden)]
    fn invoke_batch(
        &self,
        state: &mut ExecutionState,
        mask: &mut Mask,
        bindings: &mut Bindings,
        args: &[QueryEntry],
        out_var: Variable,
    ) {
        let pool: Pool<Vec<Value>> = with_pool_set(|ps| ps.get_pool().clone());
        let mut out = pool.get();
        mask.iter_dynamic(
            pool,
            args.iter().map(|v| match v {
                QueryEntry::Var(v) => ValueSource::Slice(&bindings[*v]),
                QueryEntry::Const(c) => ValueSource::Const(*c),
            }),
        )
        .fill_vec(&mut out, Value::stale, |_, args| self.invoke(state, &args));
        bindings.insert(out_var, out);
    }

    /// A variant of [`ExternalFunctionExt::invoke_batch`] that overwrites the output variable,
    /// rather than assigning all new values.
    ///
    /// *Panics* This method will panic if `out_var` doesn't already have an appropriately-sized
    /// vector bound in `bindings`.
    #[doc(hidden)]
    fn invoke_batch_assign(
        &self,
        state: &mut ExecutionState,
        mask: &mut Mask,
        bindings: &mut Bindings,
        args: &[QueryEntry],
        out_var: Variable,
    ) {
        let pool: Pool<Vec<Value>> = with_pool_set(|ps| ps.get_pool().clone());
        let mut out = bindings
            .take(out_var)
            .expect("output variable must be bound");
        mask.iter_dynamic(
            pool,
            args.iter().map(|v| match v {
                QueryEntry::Var(v) => ValueSource::Slice(&bindings[*v]),
                QueryEntry::Const(c) => ValueSource::Const(*c),
            }),
        )
        .assign_vec_and_retain(&mut out, |_, args| self.invoke(state, &args));
        bindings.insert(out_var, out);
    }
}

impl<T: ExternalFunction> ExternalFunctionExt for T {}

// Implements `Clone` for `Box<dyn ExternalFunctionExt>`.
dyn_clone::clone_trait_object!(ExternalFunctionExt);

pub(crate) type ExternalFunctions =
    DenseIdMapWithReuse<ExternalFunctionId, Box<dyn ExternalFunctionExt>>;

#[derive(Default)]
pub(crate) struct Counters(DenseIdMap<CounterId, AtomicUsize>);

impl Clone for Counters {
    fn clone(&self) -> Counters {
        let mut map = DenseIdMap::new();
        for (k, v) in self.0.iter() {
            // NB: we may want to experiment with Ordering::Relaxed here.
            map.insert(k, AtomicUsize::new(v.load(Ordering::SeqCst)))
        }
        Counters(map)
    }
}

impl Counters {
    pub(crate) fn read(&self, ctr: CounterId) -> usize {
        self.0[ctr].load(Ordering::Acquire)
    }
    pub(crate) fn inc(&self, ctr: CounterId) -> usize {
        // We synchronize with `read_counter` but not with other increments.
        // NB: we may want to experiment with Ordering::Relaxed here.
        self.0[ctr].fetch_add(1, Ordering::Release)
    }
}

/// A collection of tables and indexes over them.
///
/// A database also owns the memory pools used by its tables.
#[derive(Clone, Default)]
pub struct Database {
    // NB: some fields are pub(crate) to allow some internal modules to avoid
    // borrowing the whole table.
    pub(crate) tables: DenseIdMap<TableId, TableInfo>,
    // TODO: having a single AtomicUsize per counter can lead to contention. We
    // should look into prefetching counters when creating a new ExecutionState
    // and incrementing locally. Note that the batch size shouldn't be too big
    // because we keep an array per id in the UF.
    pub(crate) counters: Counters,
    pub(crate) external_functions: ExternalFunctions,
    containers: Containers,
    // Tracks the relative dependencies between tables during merge operations.
    deps: DependencyGraph,
    primitives: Primitives,
}

impl Database {
    /// Create an empty Database.
    ///
    /// Queries are executed using the current rayon thread pool, which defaults to the global
    /// thread pool.
    pub fn new() -> Database {
        Database::default()
    }

    /// Initialize a new rulse set to run against this database.
    pub fn new_rule_set(&mut self) -> RuleSetBuilder {
        RuleSetBuilder::new(self)
    }

    /// Add a new external function to the database.
    pub fn add_external_function(
        &mut self,
        f: impl ExternalFunction + 'static,
    ) -> ExternalFunctionId {
        self.external_functions.push(Box::new(f))
    }

    /// Free an existing external function. Make sure not to use `id` afterwards.
    pub fn free_external_function(&mut self, id: ExternalFunctionId) {
        self.external_functions.take(id);
    }

    pub fn primitives(&self) -> &Primitives {
        &self.primitives
    }

    pub fn primitives_mut(&mut self) -> &mut Primitives {
        &mut self.primitives
    }

    pub fn containers(&self) -> &Containers {
        &self.containers
    }

    pub fn containers_mut(&mut self) -> &mut Containers {
        &mut self.containers
    }

    pub fn rebuild_containers(&mut self, table_id: TableId) -> bool {
        let mut containers = mem::take(&mut self.containers);
        let table = &self.tables[table_id].table;
        let res = self.with_execution_state(|state| containers.rebuild_all(table_id, table, state));
        self.containers = containers;
        res
    }

    /// Apply the value-level rebuild encoded by `func_id` to all the tables in `to_rebuild`.
    ///
    /// The native [`Table::apply_rebuild`] method takes a `next_ts` argument for filling in new
    /// values in a table like [`crate::SortedWritesTable`] where values in a certain column need
    /// to be inserted in sorted order; the `next_ts` argument to this method is passed to
    /// `apply_rebuild` for this purpose.
    pub fn apply_rebuild(
        &mut self,
        func_id: TableId,
        to_rebuild: &[TableId],
        next_ts: Value,
    ) -> bool {
        fn do_parallel() -> bool {
            #[cfg(test)]
            {
                use rand::Rng;
                rand::thread_rng().gen_bool(0.5)
            }
            #[cfg(not(test))]
            {
                rayon::current_num_threads() > 1
            }
        }

        let func = self.tables.take(func_id).unwrap();
        let predicted = PredictedVals::default();
        if do_parallel() {
            let mut tables = Vec::with_capacity(to_rebuild.len());
            for id in to_rebuild {
                tables.push((*id, self.tables.take(*id).unwrap()));
            }
            tables.par_iter_mut().for_each(|(_, info)| {
                info.table.apply_rebuild(
                    func_id,
                    &func.table,
                    next_ts,
                    &mut ExecutionState::new(&predicted, self.read_only_view(), Default::default()),
                );
            });
            for (id, info) in tables {
                self.tables.insert(id, info);
            }
        } else {
            for id in to_rebuild {
                let mut info = self.tables.take(*id).unwrap();
                info.table.apply_rebuild(
                    func_id,
                    &func.table,
                    next_ts,
                    &mut ExecutionState::new(&predicted, self.read_only_view(), Default::default()),
                );
                self.tables.insert(*id, info);
            }
        }
        self.tables.insert(func_id, func);
        self.merge_all()
    }

    /// Run `f` with access to an `ExecutionState` mapped to this database.
    pub fn with_execution_state<R>(&self, f: impl FnOnce(&mut ExecutionState) -> R) -> R {
        let predicted = with_pool_set(|ps| ps.get::<PredictedVals>());
        let mut state = ExecutionState::new(&predicted, self.read_only_view(), Default::default());
        f(&mut state)
    }

    pub(crate) fn read_only_view(&self) -> DbView {
        DbView {
            table_info: &self.tables,
            counters: &self.counters,
            external_funcs: &self.external_functions,
            prims: &self.primitives,
            containers: &self.containers,
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
        self.counters.0.push(AtomicUsize::new(0))
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
    /// been staged. Returns true if any tuples were added.
    ///
    /// Exposed for testing purposes.
    ///
    /// Useful for out-of-band insertions into the database.
    pub fn merge_all(&mut self) -> bool {
        let mut ever_changed = false;
        let do_parallel = rayon::current_num_threads() > 1;
        loop {
            let mut changed = false;
            let predicted = with_pool_set(|ps| ps.get::<PredictedVals>());
            let mut tables_merging = DenseIdMap::<
                TableId,
                (
                    // The info needed to merge this table.
                    Option<TableInfo>,
                    // Pre-allocated write buffers, according to the tables declared write
                    // dependencies.
                    DenseIdMap<TableId, Box<dyn MutationBuffer>>,
                ),
            >::with_capacity(self.tables.n_ids());
            for stratum in self.deps.strata() {
                // Initialize the write dependencies first.
                for table in stratum.iter().copied() {
                    let mut bufs = DenseIdMap::default();
                    for dep in self.deps.write_deps(table) {
                        if let Some(info) = self.tables.get(dep) {
                            bufs.insert(dep, info.table.new_buffer());
                        }
                    }
                    tables_merging.insert(table, (None, bufs));
                }
                // Then initialize read dependencies (this two-phase structure is why we have an
                // Option in the tables_merging map).
                for table in stratum.iter().copied() {
                    tables_merging[table].0 = Some(self.tables.unwrap_val(table));
                }
                let db = self.read_only_view();
                changed |= if do_parallel {
                    tables_merging
                        .par_iter_mut()
                        .map(|(_, (info, buffers))| {
                            let mut es = ExecutionState::new(&predicted, db, mem::take(buffers));
                            info.as_mut().unwrap().table.merge(&mut es).added || es.changed
                        })
                        .max()
                        .unwrap_or(false)
                } else {
                    tables_merging
                        .iter_mut()
                        .map(|(_, (info, buffers))| {
                            let mut es = ExecutionState::new(&predicted, db, mem::take(buffers));
                            info.as_mut().unwrap().table.merge(&mut es).added || es.changed
                        })
                        .max()
                        .unwrap_or(false)
                };
                for (id, (table, _)) in tables_merging.drain() {
                    self.tables.insert(id, table.unwrap());
                }
            }
            ever_changed |= changed;
            if !changed {
                break;
            }
        }
        ever_changed
    }

    /// A low-level helper for merging pending updates to a particular function.
    ///
    /// Callers should prefer `merge_all`, as the process of merging the data
    /// for a particular table may cause other updates to be buffered
    /// elesewhere. The `merge_all` method runs merges to a fixed point to avoid
    /// surprises here.
    pub fn merge_table(&mut self, table: TableId) {
        let mut info = self.tables.unwrap_val(table);
        let predicted = with_pool_set(|ps| ps.get::<PredictedVals>());
        let _table_changed = info.table.merge(&mut ExecutionState::new(
            &predicted,
            self.read_only_view(),
            Default::default(),
        ));
        self.tables.insert(table, info);
    }

    /// Get id of the next table to be added to the database.
    ///
    /// This can be useful for "knot tying", when tables need to reference their
    /// own id.
    pub fn next_table_id(&self) -> TableId {
        self.tables.next_id()
    }

    /// Add a table with the given schema to the database.
    ///
    /// The table must have a compatible spec with `types` (e.g. same number of
    /// columns).
    pub fn add_table<T: Table + Sized + 'static>(
        &mut self,
        table: T,
        read_deps: impl IntoIterator<Item = TableId>,
        write_deps: impl IntoIterator<Item = TableId>,
    ) -> TableId {
        let spec = table.spec();
        let table = WrappedTable::new(table);
        let res = self.tables.push(TableInfo {
            spec,
            table,
            indexes: Default::default(),
            column_indexes: Default::default(),
        });
        self.deps.add_table(res, read_deps, write_deps);
        res
    }

    /// Get direct mutable access to the table.
    ///
    /// This method is useful for out-of-band access to databse state.
    pub fn get_table(&self, id: TableId) -> &WrappedTable {
        &self
            .tables
            .get(id)
            .expect("must access a table that has been declared in this database")
            .table
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
            match index.read().get_subset(&val) {
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
    pub fn get_table_mut(&mut self, id: TableId) -> &mut dyn Table {
        &mut *self
            .tables
            .get_mut(id)
            .expect("must access a table that has been declared in this database")
            .table
    }

    pub(crate) fn plan_query(&mut self, query: Query) -> Plan {
        plan::plan_query(query)
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        // Clean up the ambient thread pool.
        //
        // Calling mem::forget on the egraph can result in much faster execution times.
        with_pool_set(PoolSet::clear);
        rayon::broadcast(|_| with_pool_set(PoolSet::clear));
    }
}

/// The core logic behind getting and updating a hash index.
///
/// This is in a separate function to allow us to reuse it while already
/// borrowing a `TableInfo`.
fn get_index_from_tableinfo(table_info: &TableInfo, cols: &[ColumnId]) -> HashIndex {
    let index: Arc<_> = table_info
        .indexes
        .entry(cols.into())
        .or_insert_with(|| {
            Arc::new(ReadOptimizedLock::new(Index::new(
                cols.to_vec(),
                TupleIndex::new(cols.len()),
            )))
        })
        .clone();
    let ix = index.read();
    if ix.needs_refresh(table_info.table.as_ref()) {
        mem::drop(ix);
        let mut ix = index.lock();
        ix.refresh(table_info.table.as_ref());
    }
    index
}

/// The core logic behind getting and updating a column index.
///
/// This is the single-column analog to [`get_index_from_tableinfo`].
fn get_column_index_from_tableinfo(table_info: &TableInfo, col: ColumnId) -> HashColumnIndex {
    let index: Arc<_> = table_info
        .column_indexes
        .entry(col)
        .or_insert_with(|| {
            Arc::new(ReadOptimizedLock::new(Index::new(
                vec![col],
                ColumnIndex::new(),
            )))
        })
        .clone();
    let ix = index.read();
    if ix.needs_refresh(table_info.table.as_ref()) {
        mem::drop(ix);
        let mut ix = index.lock();
        ix.refresh(table_info.table.as_ref());
    }
    index
}
