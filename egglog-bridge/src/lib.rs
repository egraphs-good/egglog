//! An implementation of egglog-style queries on top of core-relations.
//!
//! This module translates a well-typed egglog-esque query into the abstractions
//! from the `core-relations` crate. The main higher-level functionality that it
//! implements are seminaive evaluation, default values, and merge functions.
//!
//! This crate is essentially involved in desugaring: it elaborates the encoding
//! of core egglog functionality, but it does not implement algorithms for
//! joins, union-finds, etc.

use std::{
    iter, mem,
    rc::Rc,
    sync::{Arc, Mutex},
};

use core_relations::{
    ColumnId, Constraint, Container, Containers, CounterId, Database, DisplacedTable,
    DisplacedTableWithProvenance, ExternalFunction, ExternalFunctionId, MergeVal, Offset,
    PlanStrategy, PrimitiveId, Primitives, SortedWritesTable, TableId, TaggedRowBuffer, Value,
    WrappedTable,
};
use indexmap::{map::Entry, IndexMap};
use log::info;
use numeric_id::{define_id, DenseIdMap, DenseIdMapWithReuse, NumericId};
use proof_spec::{ProofReason, ProofReconstructionState, ReasonSpecId};
use smallvec::SmallVec;
use web_time::Instant;

pub mod macros;
pub(crate) mod proof_spec;
pub(crate) mod rule;
pub(crate) mod syntax;
pub(crate) mod term_proof_dag;
#[cfg(test)]
mod tests;

pub use rule::{Function, QueryEntry, RuleBuilder};
use syntax::RuleRepresentation;
use term_proof_dag::TermEnv;
pub use term_proof_dag::{EqProof, TermProof};
use thiserror::Error;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ColumnTy {
    Id,
    Primitive(PrimitiveId),
}

define_id!(pub RuleId, u32, "An egglog-style rule");
define_id!(pub FunctionId, u32, "An id representing an egglog function");
define_id!(pub(crate) Timestamp, u32, "An abstract timestamp used to track execution of egglog rules");
impl Timestamp {
    fn to_value(self) -> Value {
        Value::new(self.rep())
    }
}

/// The state associated with an egglog program.
#[derive(Clone)]
pub struct EGraph {
    db: Database,
    uf_table: TableId,
    id_counter: CounterId,
    reason_counter: CounterId,
    timestamp_counter: CounterId,
    rules: DenseIdMapWithReuse<RuleId, RuleInfo>,
    funcs: DenseIdMap<FunctionId, FunctionInfo>,
    panic_message: SideChannel<String>,
    proof_specs: DenseIdMap<ReasonSpecId, Arc<ProofReason>>,
    /// Side tables used to store proof information. We initialize these lazily
    /// as a proof object with a given number of parameters is added.
    reason_tables: IndexMap<usize /* arity */, TableId>,
    term_tables: IndexMap<usize /* arity */, TableId>,
    tracing: bool,
}

pub type Result<T> = std::result::Result<T, anyhow::Error>;

impl Default for EGraph {
    fn default() -> Self {
        let mut db = Database::new();
        let uf_table = db.add_table(DisplacedTable::default(), iter::empty(), iter::empty());
        EGraph::create_internal(db, uf_table, false)
    }
}

impl EGraph {
    /// Create a new EGraph with tracing (aka 'proofs') enabled.
    ///
    /// Execution of queries against a tracing-enabled EGgraph will be slower,
    /// but will annotate the egraph with annotations that can explain how rows
    /// came to appera.
    pub fn with_tracing() -> EGraph {
        let mut db = Database::new();
        let uf_table = db.add_table(
            DisplacedTableWithProvenance::default(),
            iter::empty(),
            iter::empty(),
        );
        EGraph::create_internal(db, uf_table, true)
    }

    fn create_internal(mut db: Database, uf_table: TableId, tracing: bool) -> EGraph {
        let id_counter = db.add_counter();
        let trace_counter = db.add_counter();
        let ts_counter = db.add_counter();
        // Start the timestamp counter at 1.
        db.inc_counter(ts_counter);

        Self {
            db,
            uf_table,
            id_counter,
            reason_counter: trace_counter,
            timestamp_counter: ts_counter,
            rules: Default::default(),
            funcs: Default::default(),
            panic_message: Default::default(),
            proof_specs: Default::default(),
            reason_tables: Default::default(),
            term_tables: Default::default(),
            tracing,
        }
    }

    fn next_ts(&self) -> Timestamp {
        Timestamp::from_usize(self.db.read_counter(self.timestamp_counter))
    }

    fn inc_ts(&mut self) {
        self.db.inc_counter(self.timestamp_counter);
    }

    /// Get a mutable reference to the underlying table of primitives for this
    /// EGraph.
    pub fn primitives_mut(&mut self) -> &mut Primitives {
        self.db.primitives_mut()
    }

    /// Get a mutable reference to the underlying table of containers for this
    /// EGraph.
    pub fn containers_mut(&mut self) -> &mut Containers {
        self.db.containers_mut()
    }

    /// Get a reference to the underlying table of containers for this EGraph.
    pub fn containers(&self) -> &Containers {
        self.db.containers()
    }

    /// Intern the given container value into the EGraph.
    pub fn get_container_val<C: Container>(&mut self, val: C) -> Value {
        self.register_container_ty::<C>();
        self.db
            .with_execution_state(|state| state.clone().containers().register_val(val, state))
    }

    /// Register the given [`Container`] type with this EGraph.
    ///
    /// The given container will use the EGraph's union-find to manage rebuilding and the merging
    /// of containers with a common id.
    pub fn register_container_ty<C: Container>(&mut self) {
        let uf_table = self.uf_table;
        let ts_counter = self.timestamp_counter;
        self.db
            .containers_mut()
            .register_type::<C>(self.id_counter, move |state, old, new| {
                if old != new {
                    let next_ts = Value::from_usize(state.read_counter(ts_counter));
                    state.stage_insert(uf_table, &[old, new, next_ts]);
                    std::cmp::min(old, new)
                } else {
                    old
                }
            });
    }

    /// Get a reference to the underlying table of primitives for this EGraph.
    pub fn primitives(&self) -> &Primitives {
        self.db.primitives()
    }

    pub fn register_external_func(
        &mut self,
        func: impl ExternalFunction + 'static,
    ) -> ExternalFunctionId {
        self.db.add_external_function(func)
    }

    pub fn free_external_func(&mut self, func: ExternalFunctionId) {
        self.db.free_external_function(func)
    }

    /// Generate a fresh id.
    pub fn fresh_id(&mut self) -> Value {
        Value::from_usize(self.db.inc_counter(self.id_counter))
    }

    /// Look up the canonical value for `val` in the union-find.
    ///
    /// If the value has never been inserted into the union-find, `val` is returned.
    pub fn get_canon(&self, val: Value) -> Value {
        let table = self.db.get_table(self.uf_table);
        let row = table.get_row(&[val]);
        row.map(|row| row.vals[1]).unwrap_or(val)
    }

    fn term_table(&mut self, table: TableId) -> TableId {
        let spec = self.db.get_table(table).spec();
        match self.term_tables.entry(spec.n_keys) {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(v) => {
                let table = SortedWritesTable::new(
                    spec.n_keys + 1,     // added entry for the tableid
                    spec.n_keys + 1 + 2, // one value for the term id, one for the reason,
                    None,
                    vec![], // no rebuilding needed for term table
                    |_, _, _, _| false,
                );
                let table_id = self.db.add_table(table, iter::empty(), iter::empty());
                *v.insert(table_id)
            }
        }
    }

    fn reason_table(&mut self, spec: &ProofReason) -> TableId {
        let arity = spec.arity();
        match self.reason_tables.entry(arity) {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(v) => {
                let table = SortedWritesTable::new(
                    arity,
                    arity + 1, // one value for the reason id
                    None,
                    vec![], // no rebuilding needed for reason tables
                    |_, _, _, _| false,
                );
                let table_id = self.db.add_table(table, iter::empty(), iter::empty());
                *v.insert(table_id)
            }
        }
    }

    /// Load the given values into the database.
    ///
    /// # Panics
    /// This method panics if the values do not match the arity of the function.
    ///
    /// NB: this is not an efficient interface for bulk loading. We should add
    /// one that allows us to pass through a series of RowBuffers before
    /// incrementing the timestamp.
    pub fn add_values(&mut self, values: impl IntoIterator<Item = (FunctionId, Vec<Value>)>) {
        self.add_values_with_desc("", values)
    }

    /// A term-oriented means of adding data to the database: hand back a "term
    /// id" for the given function and keys for the function. Proofs for this
    /// term will include `desc`.
    ///
    /// # Panics
    /// This method panics if the values do not match the arity of the function.
    pub fn add_term(&mut self, func: FunctionId, inputs: &[Value], desc: &str) -> Value {
        let mut extended_row = Vec::new();
        extended_row.extend_from_slice(inputs);
        let res = if self.tracing {
            let reason = self.get_fiat_reason(desc);
            let term = self.get_term(func, inputs, reason);
            extended_row.push(term);
            extended_row.push(self.next_ts().to_value());
            extended_row.push(term);
            term
        } else {
            let id = self.fresh_id();
            extended_row.push(id);
            extended_row.push(self.next_ts().to_value());
            id
        };
        let table_id = self.funcs[func].table;
        self.db
            .get_table(table_id)
            .new_buffer()
            .stage_insert(&extended_row);
        self.db.merge_all();
        self.inc_ts();
        self.rebuild().unwrap();
        self.get_canon(res)
    }

    /// Get an id corresponding to the given term, inserting the value into the
    /// corresponding terms table if it isn't there.
    ///
    /// This method is really only relevant when tracing is enabled.
    fn get_term(&mut self, func: FunctionId, key: &[Value], reason: Value) -> Value {
        let table_id = self.funcs[func].table;
        let term_table_id = self.term_table(table_id);
        let table = self.db.get_table(term_table_id);
        let mut term_key = Vec::with_capacity(key.len() + 1);
        term_key.push(Value::new(func.rep()));
        term_key.extend(key);
        if let Some(row) = table.get_row(&term_key) {
            row.vals[row.vals.len() - 2]
        } else {
            let result = Value::from_usize(self.db.inc_counter(self.id_counter));
            term_key.push(result);
            term_key.push(reason);
            self.db
                .get_table(term_table_id)
                .new_buffer()
                .stage_insert(&term_key);
            self.db.merge_table(term_table_id);
            result
        }
    }

    /// Lookup the id associated with a function `func` and the given arguments
    /// (`key`).
    pub fn lookup_id(&self, func: FunctionId, key: &[Value]) -> Option<Value> {
        let table_id = self.funcs[func].table;
        let table = self.db.get_table(table_id);
        let row = table.get_row(key)?;
        if self.tracing {
            // Return the "term id"
            Some(row.vals[row.vals.len() - 1])
        } else {
            // Return the eclass id
            Some(row.vals[row.vals.len() - 2])
        }
    }

    fn get_fiat_reason(&mut self, desc: &str) -> Value {
        let reason = Arc::new(ProofReason::Fiat { desc: desc.into() });
        let reason_table = self.reason_table(&reason);
        let reason_spec_id = self.proof_specs.push(reason);
        let reason_id = Value::from_usize(self.db.inc_counter(self.reason_counter));
        self.db
            .get_table(reason_table)
            .new_buffer()
            .stage_insert(&[Value::new(reason_spec_id.rep()), reason_id]);
        self.db.merge_table(reason_table);
        reason_id
    }

    /// Load the given values into the database. If tracing is enabled, the
    /// proof rows will be tagged with "desc" as their proof.
    ///
    /// # Panics
    /// This method panics if the values do not match the arity of the function.
    ///
    /// NB: this is not an efficient interface for bulk loading. We should add
    /// one that allows us to pass through a series of RowBuffers before
    /// incrementing the timestamp.
    pub fn add_values_with_desc(
        &mut self,
        desc: &str,
        values: impl IntoIterator<Item = (FunctionId, Vec<Value>)>,
    ) {
        let mut extended_row = SmallVec::<[Value; 8]>::new();
        let reason_id = if self.tracing {
            Some(self.get_fiat_reason(desc))
        } else {
            None
        };
        let mut bufs = DenseIdMap::default();
        for (func, row) in values.into_iter() {
            extended_row.extend_from_slice(&row);
            extended_row.push(self.next_ts().to_value());
            let table_info = &self.funcs[func];
            let table_id = table_info.table;
            if let Some(reason_id) = reason_id {
                // Get the term id itself
                let term_id = self.get_term(func, &row[0..row.len() - 1], reason_id);
                let buf = bufs.get_or_insert(self.uf_table, || {
                    self.db.get_table(self.uf_table).new_buffer()
                });
                // Then union it with the value being set for this term.
                buf.stage_insert(&[
                    *row.last().unwrap(),
                    term_id,
                    self.next_ts().to_value(),
                    reason_id,
                ]);
                extended_row.push(term_id);
            }
            let buf = bufs.get_or_insert(table_id, || self.db.get_table(table_id).new_buffer());
            buf.stage_insert(&extended_row);
            extended_row.clear();
        }
        // Flush the buffers.
        mem::drop(bufs);
        self.db.merge_all();
        self.inc_ts();
        self.rebuild().unwrap();
    }

    pub fn approx_table_size(&self, table: FunctionId) -> usize {
        self.db.estimate_size(self.funcs[table].table, None)
    }

    pub fn table_size(&self, table: FunctionId) -> usize {
        self.db.get_table(self.funcs[table].table).len()
    }

    /// Generate a proof explaining why a given term is in the database.
    ///
    /// # Errors
    /// This method will return an error if tracing is not enabled, or if the row is not in the database.
    ///
    /// # Panics
    /// This method may panic if `key` does not match the arity of the function,
    /// or is otherwise malformed.
    pub fn explain_term(&mut self, id: Value) -> Result<Rc<TermProof>> {
        if !self.tracing {
            return Err(ProofReconstructionError::TracingNotEnabled.into());
        }
        Ok(self.explain_term_inner(id, &mut ProofReconstructionState::default()))
    }

    /// Generate a proof explaining why the term corresponding to `id1`
    /// is equal to that corresponding to `id2`.
    ///
    /// # Errors
    /// This method will return an error if tracing is not enabled, if the row
    /// is not in the database, or if the terms themselves are not equal.
    pub fn explain_terms_equal(&mut self, id1: Value, id2: Value) -> Result<Rc<EqProof>> {
        if !self.tracing {
            return Err(ProofReconstructionError::TracingNotEnabled.into());
        }
        if self.get_canon(id1) != self.get_canon(id2) {
            // These terms aren't equal. Reconstruct the relevant terms so as to
            // get a nicer error message on the way out.
            let p1 = self.explain_term_inner(id1, &mut Default::default());
            let p2 = self.explain_term_inner(id2, &mut Default::default());
            let mut env = TermEnv::default();
            return Err(
                ProofReconstructionError::EqualityExplanationOfUnequalTerms {
                    term1: format!("{}", env.get_term(p1)),
                    term2: format!("{}", env.get_term(p2)),
                }
                .into(),
            );
        }
        Ok(self.explain_terms_equal_inner(id1, id2, &mut Default::default()))
    }

    /// Read the contents of the given function.
    ///
    /// Useful for debugging.
    pub fn dump_table(&self, table: FunctionId, mut f: impl FnMut(&[Value])) {
        let table = self.funcs[table].table;
        let imp = self.db.get_table(table);
        let truncate = if self.tracing { 2 } else { 1 };
        let all = imp.all();
        let mut cur = Offset::new(0);
        let mut buf = TaggedRowBuffer::new(imp.spec().arity());
        while let Some(next) = imp.scan_bounded(all.as_ref(), cur, 500, &mut buf) {
            buf.non_stale()
                .for_each(|(_, row)| f(&row[0..row.len() - truncate]));
            cur = next;
            buf.clear();
        }
        buf.non_stale()
            .for_each(|(_, row)| f(&row[0..row.len() - truncate]));
    }

    /// A basic method for dumping the state of the database to `log::info!`.
    ///
    /// For large tables, this is unlikely to give particularly useful output.
    pub fn dump_debug_info(&self) {
        info!("=== View Tables ===");
        for (id, info) in self.funcs.iter() {
            let table = self.db.get_table(info.table);
            self.scan_table(table, |row| {
                info!(
                    "View Table {name} / {id:?} / {table:?}: {row:?}",
                    name = info.name,
                    table = info.table
                )
            });
        }

        info!("=== Term Tables ===");
        for (_, table_id) in &self.term_tables {
            let table = self.db.get_table(*table_id);
            self.scan_table(table, |row| {
                let name = &self.funcs[FunctionId::new(row[0].rep())].name;
                let row = &row[1..];
                info!("Term Table {table_id:?}: {name}, {row:?}")
            });
        }

        info!("=== Reason Tables ===");
        for (_, table_id) in &self.reason_tables {
            let table = self.db.get_table(*table_id);
            self.scan_table(table, |row| {
                let spec = self.proof_specs[ReasonSpecId::new(row[0].rep())].as_ref();
                let row = &row[1..];
                info!("Reason Table {table_id:?}: {spec:?}, {row:?}")
            });
        }
    }

    /// A helper for scanning the entries in a table.
    fn scan_table(&self, table: &WrappedTable, mut f: impl FnMut(&[Value])) {
        const BATCH_SIZE: usize = 128;
        let all = table.all();
        let mut cur = Offset::new(0);
        let mut out = TaggedRowBuffer::new(table.spec().arity());
        while let Some(next) = table.scan_bounded(all.as_ref(), cur, BATCH_SIZE, &mut out) {
            out.non_stale().for_each(|(_, row)| f(row));
            out.clear();
            cur = next;
        }
        out.non_stale().for_each(|(_, row)| f(row));
    }

    /// Register a function in this EGraph.
    pub fn add_table(
        &mut self,
        schema: Vec<ColumnTy>,
        default: DefaultVal,
        merge: MergeFn,
        name: &str,
    ) -> FunctionId {
        assert!(
            !schema.is_empty(),
            "must have at least one column in schema"
        );
        let to_rebuild: Vec<ColumnId> = schema
            .iter()
            .enumerate()
            .filter(|(_, ty)| matches!(ty, ColumnTy::Id))
            .map(|(i, _)| ColumnId::from_usize(i))
            .collect();
        let n_args = schema.len() - 1;
        let n_cols = if self.tracing {
            schema.len() + 2
        } else {
            schema.len() + 1
        };
        let uf_table = self.uf_table;
        let tracing = self.tracing;
        let next_func_id = self.funcs.next_id();
        let mut read_deps = SmallVec::<[TableId; 2]>::new();
        let mut write_deps = SmallVec::<[TableId; 2]>::new();
        if !tracing {
            write_deps.push(uf_table);
        }
        let table = match merge {
            MergeFn::AssertEq => {
                let panic = self.new_panic(format!("Illegal merge attempted for function {name}"));
                SortedWritesTable::new(
                    n_args,
                    n_cols,
                    Some(ColumnId::from_usize(schema.len())),
                    to_rebuild,
                    move |state, cur, new, _out| {
                        if cur != new {
                            let res = state.call_external_func(panic, &[]);
                            assert_eq!(res, None);
                        }
                        false
                    },
                )
            }
            MergeFn::UnionId => {
                SortedWritesTable::new(
                    n_args,
                    n_cols,
                    Some(ColumnId::from_usize(schema.len())),
                    to_rebuild,
                    move |state, cur, new, out| {
                        let l = cur[n_args];
                        let r = new[n_args];
                        let next_ts = new[n_args + 1];
                        if l != r && !tracing {
                            // When proofs are enabled, these are the same term. They are already
                            // equal and we can just do nothing.
                            state.stage_insert(uf_table, &[l, r, next_ts]);
                            out.extend_from_slice(&new[0..n_args]);
                            // We pick the minimum when unioning. This matches the original egglog
                            // behavior.
                            let res = std::cmp::min(l, r);
                            out.push(std::cmp::min(l, r));
                            out.extend_from_slice(&new[n_args + 1..]);
                            r == res
                        } else {
                            false
                        }
                    },
                )
            }
            MergeFn::Table(merge_table) => {
                read_deps.push(merge_table);
                write_deps.push(merge_table);
                let id_counter = self.id_counter;
                SortedWritesTable::new(
                    n_args,
                    n_cols,
                    Some(ColumnId::from_usize(schema.len())),
                    to_rebuild,
                    move |state, cur, new, out| {
                        // We have F(x0, ..., xn, v1, t1) and F(x0, ..., xn, v2,
                        // t2) in the same table.
                        // F has a merge function given by T. (`merge_table`).
                        let l = cur[n_args];
                        let r = new[n_args];
                        let next_ts = new[n_args + 1];
                        if l == r {
                            // Merge functions should be idempotent: T(x,x)
                            // should eventually equal x. We short-circuit here.
                            return false;
                        }
                        // Look up T(v1, v2), giving us a new value v* (+ a
                        // timestamp that we ignore). We use the "predictions"
                        // API, which is only safe because we do not allow "raw
                        // primitives" in return position here.
                        let res = if tracing {
                            panic!("proofs aren't supported for non-union merge functions")
                        } else {
                            state.predict_val(
                                merge_table,
                                &[l, r],
                                [MergeVal::from(id_counter), MergeVal::from(next_ts)].into_iter(),
                            )
                        };
                        debug_assert_eq!(res.len(), 2);
                        // If v* = v1, return with "no change".
                        if res[0] == l {
                            return false;
                        }
                        // Otherwise, build a new tuple F(x0, ..., xn, v*, t2).
                        out.extend_from_slice(&new[0..n_args]);
                        out.push(res[0]);
                        out.extend_from_slice(&new[n_args + 1..]);
                        true
                    },
                )
            }
        };
        let table_id =
            self.db
                .add_table(table, read_deps.iter().copied(), write_deps.iter().copied());
        let res = self.funcs.push(FunctionInfo {
            table: table_id,
            schema: schema.clone(),
            incremental_rebuild_rules: Default::default(),
            nonincremental_rebuild_rule: RuleId::new(!0),
            default_val: default,
            name: name.into(),
        });
        debug_assert_eq!(res, next_func_id);
        let incremental_rebuild_rules = self.incremental_rebuild_rules(res, &schema);
        let nonincremental_rebuild_rule = self.nonincremental_rebuild(res, &schema);
        let info = &mut self.funcs[res];
        info.incremental_rebuild_rules = incremental_rebuild_rules;
        info.nonincremental_rebuild_rule = nonincremental_rebuild_rule;
        res
    }

    /// Run the given rules, returning whether the database changed.
    ///
    /// If the given rules are malformed, this method can return an error.
    pub fn run_rules(&mut self, rules: &[RuleId]) -> Result<bool> {
        let ts = self.next_ts();
        let changed = run_rules_impl(&mut self.db, &mut self.rules, rules, ts)?;
        if let Some(message) = self.panic_message.lock().unwrap().take() {
            return Err(PanicError(message).into());
        }
        if !changed {
            return Ok(false);
        }
        self.rebuild()?;
        if let Some(message) = self.panic_message.lock().unwrap().take() {
            return Err(PanicError(message).into());
        }
        Ok(true)
    }

    fn rebuild(&mut self) -> Result<()> {
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
        if self.db.get_table(self.uf_table).rebuilder(&[]).is_some() {
            // The UF implementation supports "native"  rebuilding.
            let mut tables = Vec::with_capacity(self.funcs.next_id().index());
            for (_, func) in self.funcs.iter() {
                tables.push(func.table);
            }
            while self
                .db
                .apply_rebuild(self.uf_table, &tables, self.next_ts().to_value())
                || self.db.rebuild_containers(self.uf_table)
            {
                self.inc_ts();
            }
            self.inc_ts();
            return Ok(());
        }
        if do_parallel() {
            return self.rebuild_parallel();
        }
        let start = Instant::now();

        // The database changed. Rebuild. New entries should land after the given rules.
        let mut changed = true;
        while changed {
            changed = false;
            // We need to iterate rebuilding to a fixed point. Future scans
            // should look only at the latest updates.
            self.inc_ts();
            let ts = self.next_ts();
            for (_, info) in self.funcs.iter_mut() {
                let last_rebuilt_at = self.rules[info.nonincremental_rebuild_rule].last_run_at;
                let table_size = self.db.estimate_size(info.table, None);
                let uf_size = self.db.estimate_size(
                    self.uf_table,
                    Some(Constraint::GeConst {
                        col: ColumnId::new(2),
                        val: last_rebuilt_at.to_value(),
                    }),
                );
                if incremental_rebuild(uf_size, table_size, false) {
                    marker_incremental_rebuild(|| -> Result<()> {
                        // Run each of the incremental rules serially.
                        //
                        // This is to avoid recanonicalizing the same row multiple
                        // times.
                        for rule in &info.incremental_rebuild_rules {
                            changed |= run_rules_impl(&mut self.db, &mut self.rules, &[*rule], ts)?;
                        }
                        // Reset the rule we did not run. These two should be equivalent.
                        self.rules[info.nonincremental_rebuild_rule].last_run_at = ts;
                        Ok(())
                    })?;
                } else {
                    marker_nonincremental_rebuild(|| -> Result<()> {
                        changed |= run_rules_impl(
                            &mut self.db,
                            &mut self.rules,
                            &[info.nonincremental_rebuild_rule],
                            ts,
                        )?;
                        for rule in &info.incremental_rebuild_rules {
                            self.rules[*rule].last_run_at = ts;
                        }
                        Ok(())
                    })?;
                }
            }
        }
        log::info!("rebuild took {:?}", start.elapsed());
        Ok(())
    }

    /// A variant of `rebuild` that attempts to combine rebuild rules into
    /// larger rulesets to increase parallelism. This kind of preprocessing can
    /// slow processing down in a single-threaded setting, so it is only used
    /// when the number of active threads is greater than 1.
    fn rebuild_parallel(&mut self) -> Result<()> {
        let start = Instant::now();
        #[derive(Default)]
        struct RebuildState {
            nonincremental: Vec<FunctionId>,
            incremental: DenseIdMap<usize, SmallVec<[FunctionId; 2]>>,
        }

        impl RebuildState {
            fn clear(&mut self) {
                self.nonincremental.clear();
                self.incremental.iter_mut().for_each(|(_, v)| v.clear());
            }
        }

        let mut changed = true;
        let mut state = RebuildState::default();
        let mut scratch = Vec::new();
        while changed {
            changed = false;
            state.clear();
            self.inc_ts();
            // First, figure out which functions will be rebuilt nonincrementally,
            // vs. incrementally. Group them together.
            for (func, info) in self.funcs.iter_mut() {
                let last_rebuilt_at = self.rules[info.nonincremental_rebuild_rule].last_run_at;
                let table_size = self.db.estimate_size(info.table, None);
                let uf_size = self.db.estimate_size(
                    self.uf_table,
                    Some(Constraint::GeConst {
                        col: ColumnId::new(2),
                        val: last_rebuilt_at.to_value(),
                    }),
                );
                if incremental_rebuild(uf_size, table_size, true) {
                    for (i, _) in info.incremental_rebuild_rules.iter().enumerate() {
                        state.incremental.get_or_default(i).push(func);
                    }
                } else {
                    state.nonincremental.push(func);
                }
            }
            let ts = self.next_ts();
            for func in state.nonincremental.iter().copied() {
                scratch.push(self.funcs[func].nonincremental_rebuild_rule);
                for rule in &self.funcs[func].incremental_rebuild_rules {
                    self.rules[*rule].last_run_at = ts;
                }
            }
            changed |= run_rules_impl(&mut self.db, &mut self.rules, &scratch, ts)?;
            scratch.clear();
            let ts = self.next_ts();
            for (i, funcs) in state.incremental.iter() {
                for func in funcs.iter().copied() {
                    let info = &mut self.funcs[func];
                    scratch.push(info.incremental_rebuild_rules[i]);
                    self.rules[info.nonincremental_rebuild_rule].last_run_at = ts;
                }
                changed |= run_rules_impl(&mut self.db, &mut self.rules, &scratch, ts)?;
                scratch.clear();
            }
        }
        log::info!("rebuild took {:?}", start.elapsed());
        Ok(())
    }

    fn incremental_rebuild_rules(&mut self, table: FunctionId, schema: &[ColumnTy]) -> Vec<RuleId> {
        schema
            .iter()
            .enumerate()
            .filter_map(|(i, ty)| match ty {
                ColumnTy::Id => {
                    Some(self.incremental_rebuild_rule(table, schema, ColumnId::from_usize(i)))
                }
                ColumnTy::Primitive(_) => None,
            })
            .collect()
    }

    fn incremental_rebuild_rule(
        &mut self,
        table: FunctionId,
        schema: &[ColumnTy],
        col: ColumnId,
    ) -> RuleId {
        let table_id = self.funcs[table].table;
        let uf_table = self.uf_table;
        // Two atoms, one binding a whole tuple, one binding a displaced column
        let mut rb = self.new_rule();
        rb.set_plan_strategy(PlanStrategy::MinCover);
        let mut vars = Vec::<QueryEntry>::with_capacity(schema.len());
        for ty in schema {
            vars.push(rb.new_var(*ty).into());
        }
        let canon_val = rb.new_var(ColumnTy::Id);
        rb.add_atom_with_timestamp(table_id, &vars);
        rb.add_atom_with_timestamp(uf_table, &[vars[col.index()].clone(), canon_val.into()]);
        rb.set_focus(1); // Set the uf atom as the sole focus.

        // Now canonicalize the entire row.
        let mut canon = Vec::<QueryEntry>::with_capacity(schema.len());
        for (i, (var, ty)) in vars.iter().zip(schema.iter()).enumerate() {
            canon.push(if i == col.index() {
                canon_val.into()
            } else if let ColumnTy::Id = ty {
                rb.lookup_uf(var.clone()).unwrap().into()
            } else {
                var.clone()
            })
        }

        // Remove the old row and insert the new one.
        rb.rebuild_row(table, &vars, &canon);
        rb.build_described(format!("incremental rebuild {table:?}, {col:?}"))
    }

    fn nonincremental_rebuild(&mut self, table: FunctionId, schema: &[ColumnTy]) -> RuleId {
        let mut rb = self.new_nonincremental_rule();
        rb.set_plan_strategy(PlanStrategy::MinCover);
        let mut vars = Vec::<QueryEntry>::with_capacity(schema.len());
        for ty in schema {
            vars.push(rb.new_var(*ty).into());
        }
        rb.add_atom(Function::Table(table), &vars).unwrap();
        let mut lhs = SmallVec::<[QueryEntry; 4]>::new();
        let mut rhs = SmallVec::<[QueryEntry; 4]>::new();
        let mut canon = Vec::<QueryEntry>::with_capacity(schema.len());
        for (var, ty) in vars.iter().zip(schema.iter()) {
            canon.push(if let ColumnTy::Id = ty {
                lhs.push(var.clone());
                let canon_var = QueryEntry::from(rb.lookup_uf(var.clone()).unwrap());
                rhs.push(canon_var.clone());
                canon_var
            } else {
                var.clone()
            })
        }
        rb.check_for_update(&lhs, &rhs).unwrap();
        rb.rebuild_row(table, &vars, &canon);
        rb.build_described(format!("nonincremental rebuild {table:?}"))
    }
}

#[derive(Clone)]
struct RuleInfo {
    last_run_at: Timestamp,
    query: rule::Query,
    syntax: RuleRepresentation,
    desc: String,
}

#[derive(Clone)]
struct FunctionInfo {
    table: TableId,
    schema: Vec<ColumnTy>,
    incremental_rebuild_rules: Vec<RuleId>,
    nonincremental_rebuild_rule: RuleId,
    default_val: DefaultVal,
    name: Arc<str>,
}

impl FunctionInfo {
    fn ret_ty(&self) -> ColumnTy {
        self.schema.last().copied().unwrap()
    }
}

/// How defaults are computed for the given function.
#[derive(Clone)]
pub enum DefaultVal {
    /// Generate a fresh UF id.
    FreshId,
    /// Stop executing the rule. If a lookup occurs in the body of a rule for a
    /// mapping not in a function, execution of that rule will stop. This is
    /// similar to placing the value in the left-hand side of the rule, but this
    /// time the lookup can depend on values bound in the right-hand-side.
    Fail,
    /// Insert a constant of some kind.
    Const(Value),
}

/// How to resolve FD conflicts for a table.
pub enum MergeFn {
    /// Panic if the old and new values don't match.
    AssertEq,
    /// Use congruence to resolve FD conflicts.
    UnionId,
    /// The corresponding output is replaced with the mapping in a table.
    Table(TableId),
}

fn run_rules_impl(
    db: &mut Database,
    rule_info: &mut DenseIdMapWithReuse<RuleId, RuleInfo>,
    rules: &[RuleId],
    next_ts: Timestamp,
) -> Result<bool> {
    let mut rsb = db.new_rule_set();
    for rule in rules {
        let info = &mut rule_info[*rule];
        info.query
            .add_rules(&mut rsb, info.last_run_at, next_ts, &info.desc)?;
        info.last_run_at = next_ts;
    }
    let ruleset = rsb.build();
    Ok(db.run_rule_set(&ruleset))
}

// These markers are just used to make it easy to distinguish time spent in
// incremental vs. nonincremental rebuilds in time-based profiles.

#[inline(never)]
fn marker_incremental_rebuild<R>(f: impl FnOnce() -> R) -> R {
    f()
}

#[inline(never)]
fn marker_nonincremental_rebuild<R>(f: impl FnOnce() -> R) -> R {
    f()
}

/// A useful type definition for external functions that need to pass data
/// to outside code, such as `Panic`.
pub type SideChannel<T> = Arc<Mutex<Option<T>>>;

/// An external function used to grab a value out of the database matching a
/// particular query.
//
// TODO: once we have parallelism wired in, we'll want to replace this with a
// more efficient solution (e.g. one based on crossbeam or arcswap).
#[derive(Clone)]
struct GetFirstMatch(SideChannel<Vec<Value>>);

impl ExternalFunction for GetFirstMatch {
    fn invoke(&self, _: &mut core_relations::ExecutionState, args: &[Value]) -> Option<Value> {
        let mut guard = self.0.lock().unwrap();
        if guard.is_some() {
            return None;
        }
        *guard = Some(args.to_vec());
        Some(Value::new(0))
    }
}

/// An external function used to store a message when a panic occurs.
//
// TODO: once we have parallelism wired in, we'll want to replace this with a
// more efficient solution (e.g. one based on crossbeam or arcswap).
#[derive(Clone)]
struct Panic(String, SideChannel<String>);

impl EGraph {
    /// Create a new `ExternalFunction` that panics with the given message.
    pub fn new_panic(&mut self, message: String) -> ExternalFunctionId {
        let panic = Panic(message, self.panic_message.clone());
        self.db.add_external_function(panic)
    }
}

impl ExternalFunction for Panic {
    fn invoke(&self, _: &mut core_relations::ExecutionState, args: &[Value]) -> Option<Value> {
        // TODO (egglog feature): change this to support interpolating panic messages
        assert!(args.is_empty());

        let mut guard = self.1.lock().unwrap();
        if guard.is_none() {
            *guard = Some(self.0.clone());
        }
        None
    }
}

#[derive(Error, Debug)]
enum ProofReconstructionError {
    #[error("attempting to explain a row without tracing enabled. Try constructing with `EGraph::with_tracing`")]
    TracingNotEnabled,
    #[error("attempting to construct a proof that {term1} = {term2}, but they are not equal")]
    EqualityExplanationOfUnequalTerms { term1: String, term2: String },
}

/// Heuristic for deciding whether to do an incremental or nonincremental
/// rebuild for a given table.
fn incremental_rebuild(uf_size: usize, table_size: usize, parallel: bool) -> bool {
    if parallel {
        uf_size <= (table_size / 16)
    } else {
        uf_size <= (table_size / 8)
    }
}

#[derive(Error, Debug)]
#[error("Panic: {0}")]
struct PanicError(String);
