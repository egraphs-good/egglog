//! `impl egglog_backend_trait::Backend for EGraph` — Phase 2 Commit 9 + 13.
//!
//! This file provides a **stub** implementation of the `Backend` trait
//! for the DuckDB-backed [`crate::EGraph`]. The frontend `EGraph` in
//! `src/lib.rs` does **not yet** reach this impl: the duckdb test combos
//! still route through the parallel pipeline in `src/backend_duckdb.rs`
//! via `DuckdbBackend::parse_and_run_program`. The flip lands in Commit
//! 14.
//!
//! ## What is implemented
//!
//! These methods have real bodies; they call directly into the existing
//! duckdb-bridge primitives:
//!
//! - [`Backend::add_table`]: dispatches on [`FunctionConfig::default`]
//!   and [`FunctionConfig::merge`] to `add_function` /
//!   `add_relation_with_pname` / `add_eq_sort_constructor`. Returns a
//!   numeric [`FunctionId`] tracked in `EGraph::backend_function_names`.
//! - [`Backend::table_size`]: `db.count(name)`.
//! - [`Backend::for_each`] / [`Backend::for_each_while`]:
//!   `SELECT * FROM <table>` cursor; rows are packed into a per-row
//!   `Vec<Value>` buffer and surfaced as `FunctionRow`. The subsumption
//!   bit is always `false` (DuckDB does not model subsumption; see
//!   [`Backend::supports_subsumption`]).
//! - [`Backend::lookup_id`]: `SELECT c<n> FROM t WHERE c0=? AND … LIMIT 1`.
//! - [`Backend::set_report_level`]: stores the level on the egraph.
//! - [`Backend::dump_debug_info`]: iterates registered tables and logs
//!   `SELECT * FROM <table>` via `log::info!`.
//! - Capability flags (all `false`):
//!   [`Backend::supports_inline_table_lookups`],
//!   [`Backend::supports_subsumption`],
//!   [`Backend::supports_complex_merge`],
//!   [`Backend::supports_containers`].
//! - [`Backend::as_any`] / [`Backend::as_any_mut`]: trivial.
//! - [`Backend::container_pool`] / [`Backend::container_pool_mut`]:
//!   return `&self.backend_container_pool` /
//!   `&mut self.backend_container_pool` — the zero-sized stub defined
//!   below.
//! - [`Backend::flush_updates`]: returns `false` (no-op until Commit 10
//!   wires rule running).
//!
//! ## What is `unimplemented!()`
//!
//! These trait methods are deferred to subsequent commits:
//!
//! - [`Backend::new_rule`], [`Backend::run_rules`], [`Backend::free_rule`]
//!   — Commit 10 (`rule_builder.rs`).
//! - [`Backend::register_external_func`], [`Backend::free_external_func`],
//!   [`Backend::new_panic`] — Commit 12.
//! - [`Backend::base_value_pool`], [`Backend::base_value_pool_mut`],
//!   [`Backend::base_value_constant_dyn`] — Commit 11.
//! - [`Backend::add_values`], [`Backend::insert_rows`],
//!   [`Backend::lookup_constructor_rows`], [`Backend::add_term`]
//!   — Commit 12 (after primitives wire up).
//! - [`Backend::get_canon_repr`], [`Backend::fresh_id`] — needed
//!   eventually but not yet (Commit 11 or later).
//! - [`Backend::clone_boxed`] — DuckDB's `Connection` is not trivially
//!   `Clone`; Phase 4 cleanup may revisit. See
//!   `docs/backend_trait_design.md` for the replay-log proposal.
//!
//! ## `DuckdbContainerPool` — Commit 13
//!
//! DuckDB does not support container sorts (see the
//! `program_supports_proofs` gate in
//! `src/proofs/proof_encoding_helpers.rs`, which excludes every
//! container-using program from DuckDB test combos). The pool is a
//! zero-sized stub: `has_container_type` returns `false`, `get_dyn` /
//! `for_each_dyn` are no-ops, `size_dyn` returns 0, and
//! `register_val_dyn` returns an error.

use std::any::{Any, TypeId};

use anyhow::Result;
use duckdb::types::ValueRef;
use egglog_backend_trait::{
    Backend, BaseValuePool, ColumnTy, ContainerPool, DefaultVal, ExternalFunction,
    ExternalFunctionId, FunctionConfig, FunctionId, FunctionRow, IterationReport, MergeFn,
    QueryEntry, ReportLevel, RuleBuilderOps, RuleId, Value,
};
use egglog_numeric_id::NumericId;

use crate::{ColumnTy as DuckColumnTy, EGraph, MergeMode, q};

// ---------------------------------------------------------------------------
// `DuckdbContainerPool` — Commit 13's stub
// ---------------------------------------------------------------------------

/// Zero-sized stub implementing [`ContainerPool`] for the DuckDB
/// backend. Always reports no registered types; all mutators return
/// errors.
///
/// The DuckDB backend does not support container sorts in v1. The
/// existing `program_supports_proofs` gate excludes every
/// container-using program from DuckDB test combos, so this stub's
/// error paths are programmer-error guards rather than routine code
/// paths.
pub(crate) struct DuckdbContainerPool;

impl ContainerPool for DuckdbContainerPool {
    fn has_container_type(&self, _type_id: TypeId) -> bool {
        false
    }

    fn enabled(&self) -> bool {
        false
    }

    fn get_dyn(&self, _ty: TypeId, _val: Value) -> Option<Box<dyn Any + Send + Sync>> {
        None
    }

    fn register_val_dyn(
        &mut self,
        _ty: TypeId,
        _value: Box<dyn Any + Send + Sync>,
    ) -> Result<Value> {
        Err(anyhow::anyhow!("containers not supported on DuckDB"))
    }

    fn for_each_dyn(&self, _ty: TypeId, _f: &mut dyn FnMut(Value, &dyn Any)) {
        // No-op: pool is empty.
    }

    fn size(&self, _ty: TypeId) -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Translate a [`ColumnTy`] from the backend trait crate to the
/// duckdb-bridge's own [`crate::ColumnTy`].
///
/// For the trait, `ColumnTy::Id` (an eq-sort or container id) maps to
/// the duckdb-bridge's `I64` (DuckDB stores ids as `BIGINT`); the
/// concrete primitive types (`Base(BaseValueId)`) would require
/// consulting the [`BaseValuePool`] to determine the underlying Rust
/// type. The pool isn't wired yet (Commit 11), so for the stub we
/// treat all `Base(_)` as `I64` — this is what most primitive sorts
/// already are in the existing DuckDB pipeline.
fn trait_col_ty_to_duck(_ty: ColumnTy) -> DuckColumnTy {
    // The current stub treats every column as `I64`. A more precise
    // mapping (Bool / F64 / Str) requires the BaseValuePool wiring from
    // Commit 11 to resolve a `BaseValueId` back to its concrete Rust
    // type. Since no `Box<dyn Backend>` caller exercises `add_table` on
    // the DuckDB impl until Commit 14, the stub is correct in the sense
    // that no caller will observe the imprecision in practice.
    DuckColumnTy::I64
}

/// Convert a DuckDB row value into a [`Value`] for surfacing through
/// [`FunctionRow`].
///
/// The current DuckDB pipeline stores all egglog values as `BIGINT`
/// (eq-sort ids) or other concrete SQL types. The Backend trait surface
/// uses a single `Value` (a `u32` newtype) for every column. The
/// mapping is:
///
/// - `BIGINT` -> low 32 bits cast to `Value`. For the eq-sort id case
///   this is the canonical mapping; for primitive `BIGINT`s storing
///   `i64` literals it loses the high bits, but only the trait-side
///   pool wired in Commit 11 will know how to interpret the bits, and
///   `for_each` callers reaching this path will be added in Commit
///   14+. For now correctness is moot since no caller exercises it.
fn duck_value_to_trait_value(v: ValueRef<'_>) -> Value {
    use duckdb::types::ValueRef as V;
    match v {
        V::Null => Value::new(u32::MAX),
        V::Boolean(b) => Value::new(b as u32),
        V::TinyInt(i) => Value::new(i as u32),
        V::SmallInt(i) => Value::new(i as u32),
        V::Int(i) => Value::new(i as u32),
        V::BigInt(i) => Value::new(i as u32),
        V::HugeInt(i) => Value::new(i as u32),
        V::UTinyInt(i) => Value::new(i as u32),
        V::USmallInt(i) => Value::new(i as u32),
        V::UInt(i) => Value::new(i),
        V::UBigInt(i) => Value::new(i as u32),
        // Other types fall back to a sentinel; not reached in stub
        // usage (only eq-sort id columns flow through trait callers in
        // Phase 2 Commit 9).
        _ => Value::new(u32::MAX),
    }
}

impl EGraph {
    /// Look up the duckdb-bridge table name for a trait [`FunctionId`].
    fn name_for_function_id(&self, id: FunctionId) -> &str {
        let idx = id.rep() as usize;
        self.backend_function_names
            .get(idx)
            .map(|s| s.as_str())
            .unwrap_or_else(|| panic!("FunctionId({idx}) is not registered"))
    }
}

// ---------------------------------------------------------------------------
// `impl Backend for EGraph`
// ---------------------------------------------------------------------------

impl Backend for EGraph {
    // -- table lifecycle ----------------------------------------------------

    fn add_table(&mut self, config: FunctionConfig) -> FunctionId {
        // Allocate a numeric id and register the name. The duckdb
        // backend itself uses string names; the trait callers use
        // numeric ids, so we maintain a Vec<String> mapping.
        let idx = self.backend_function_names.len() as u32;
        let id = FunctionId::new(idx);

        // Split the schema into input columns + output column (if any).
        // For relations (no output) the entire schema is "inputs".
        // For eq-sort constructors and functions, the last entry is the
        // output column.
        //
        // Heuristic for dispatch (per the prompt):
        //   - DefaultVal::FreshId => add_eq_sort_constructor (constructor)
        //   - DefaultVal::Fail / Const + MergeFn::AssertEq / UnionId
        //     => add_function with a merge mode derived from MergeFn
        //   - If the schema has no output column treat as relation:
        //     add_relation_with_pname.
        //
        // The dispatch is intentionally simple: this code path is dead
        // until Commit 14, so the exact semantics of less-common
        // combinations aren't load-bearing yet.

        let name = config.name.clone();
        let schema = &config.schema;

        // Convert all schema columns. The trait's `FunctionConfig`
        // always includes the output (if any) as the last entry of
        // `schema`; the duckdb-bridge `add_function` and
        // `add_eq_sort_constructor` take inputs + output separately
        // while `add_relation_with_pname` takes the full input list.
        let duck_cols: Vec<DuckColumnTy> = schema.iter().copied().map(trait_col_ty_to_duck).collect();

        let result = match config.default {
            DefaultVal::FreshId => {
                // EqSort constructor: schema = [inputs..., Id]. The
                // duckdb-bridge appends the ID column itself, so we pass
                // only the inputs (= schema without the trailing id).
                let inputs = if duck_cols.is_empty() {
                    &[][..]
                } else {
                    &duck_cols[..duck_cols.len() - 1]
                };
                self.add_eq_sort_constructor(&name, inputs, None)
            }
            DefaultVal::Fail | DefaultVal::Const(_) => {
                // Either a function with an explicit output, or a
                // relation if `merge` is AssertEq and the schema has
                // no "natural" output column. We can't easily tell
                // schemas apart from the trait info alone — the
                // simplest rule is: if the schema has at least one
                // column we treat the last as the output for functions
                // with a real merge mode, otherwise we make it a
                // relation.
                let merge_mode = match config.merge {
                    MergeFn::Old | MergeFn::AssertEq => Some(MergeMode::Old),
                    MergeFn::New => Some(MergeMode::New),
                    MergeFn::UnionId => Some(MergeMode::Min),
                    MergeFn::Const(_)
                    | MergeFn::Primitive(_, _)
                    | MergeFn::Function(_, _) => {
                        // Per Phase 1 design: complex merges are gated
                        // out by `supports_complex_merge` (`false`).
                        // Fall back to Old for unreachable code paths.
                        Some(MergeMode::Old)
                    }
                };

                if duck_cols.is_empty() {
                    // No columns at all: treat as a nullary relation.
                    self.add_relation_with_pname(&name, &[], None)
                } else if let Some(mode) = merge_mode {
                    let inputs = &duck_cols[..duck_cols.len() - 1];
                    let output = duck_cols[duck_cols.len() - 1];
                    self.add_function(&name, inputs, output, mode)
                } else {
                    self.add_relation_with_pname(&name, &duck_cols, None)
                }
            }
        };

        if let Err(e) = result {
            // The trait method does not return Result; if the duckdb
            // backend rejects the table registration, we have to
            // panic. Until Commit 14 flips the test harness this
            // panic is unreachable.
            panic!("DuckDB add_table({}): {}", name, e);
        }

        self.backend_function_names.push(name);
        id
    }

    fn table_size(&self, table: FunctionId) -> usize {
        let name = self.name_for_function_id(table);
        self.count(name)
            .expect("table_size: COUNT(*) query failed") as usize
    }

    fn approx_table_size(&self, table: FunctionId) -> usize {
        // No fast estimate available; fall back to exact count.
        self.table_size(table)
    }

    // -- iteration ----------------------------------------------------------

    fn for_each_while(
        &self,
        table: FunctionId,
        f: &mut dyn for<'r> FnMut(FunctionRow<'r>) -> bool,
    ) {
        let name = self.name_for_function_id(table);
        let info = self
            .functions
            .get(name)
            .unwrap_or_else(|| panic!("for_each_while: function `{name}` not registered"));
        let arity = info.arity();

        // Select all columns (c0..c{arity-1}); skip the `ts` column.
        let cols: Vec<String> = (0..arity).map(|i| format!("c{i}")).collect();
        let sql = if cols.is_empty() {
            format!("SELECT 1 FROM {} LIMIT 1", q(name))
        } else {
            format!("SELECT {} FROM {}", cols.join(", "), q(name))
        };

        let mut stmt = self
            .conn
            .prepare(&sql)
            .unwrap_or_else(|e| panic!("for_each_while: prepare failed: {e}"));
        let mut rows = stmt
            .query([])
            .unwrap_or_else(|e| panic!("for_each_while: query failed: {e}"));

        let mut buf: Vec<Value> = Vec::with_capacity(arity);
        while let Some(row) = rows
            .next()
            .unwrap_or_else(|e| panic!("for_each_while: row fetch failed: {e}"))
        {
            buf.clear();
            for i in 0..arity {
                let v = row
                    .get_ref(i)
                    .unwrap_or_else(|e| panic!("for_each_while: get_ref({i}) failed: {e}"));
                buf.push(duck_value_to_trait_value(v));
            }
            let frow = FunctionRow {
                vals: &buf,
                subsumed: false,
            };
            if !f(frow) {
                break;
            }
        }
    }

    fn for_each(&self, table: FunctionId, f: &mut dyn for<'r> FnMut(FunctionRow<'r>)) {
        // Default impl in the trait is `unimplemented!()` — provide a
        // concrete one by threading `for_each_while` with an always-
        // true continuation.
        self.for_each_while(table, &mut |row| {
            f(row);
            true
        });
    }

    // -- direct access ------------------------------------------------------

    fn lookup_id(&self, func: FunctionId, key: &[Value]) -> Option<Value> {
        let name = self.name_for_function_id(func);
        let info = self
            .functions
            .get(name)
            .unwrap_or_else(|| panic!("lookup_id: function `{name}` not registered"));
        if !info.has_output() {
            // Relations don't have an output column. The trait method's
            // contract is "key -> output"; for relations there is no
            // output. Return None.
            return None;
        }
        let inputs_len = info.inputs_len;
        if key.len() != inputs_len {
            return None;
        }
        let where_parts: Vec<String> = (0..inputs_len)
            .map(|i| format!("c{i} = {}", key[i].rep() as i64))
            .collect();
        let where_clause = if where_parts.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", where_parts.join(" AND "))
        };
        let sql = format!(
            "SELECT c{inputs_len} FROM {}{} LIMIT 1",
            q(name),
            where_clause
        );
        let mut stmt = self.conn.prepare(&sql).ok()?;
        let mut rows = stmt.query([]).ok()?;
        let row = rows.next().ok()??;
        let v = row.get_ref(0).ok()?;
        Some(duck_value_to_trait_value(v))
    }

    fn add_values(&mut self, _values: Box<dyn Iterator<Item = (FunctionId, Vec<Value>)> + '_>) {
        unimplemented!("DuckdbBackend::add_values is deferred to Phase 2 Commit 12")
    }

    fn add_term(&mut self, _func: FunctionId, _inputs: &[Value]) -> Value {
        unimplemented!("DuckdbBackend::add_term is deferred to Phase 2 Commit 12")
    }

    fn insert_rows(&mut self, _table: FunctionId, _rows: &[Vec<Value>]) {
        unimplemented!("DuckdbBackend::insert_rows is deferred to Phase 2 Commit 12")
    }

    fn lookup_constructor_rows(&mut self, _table: FunctionId, _rows: &[Vec<Value>]) {
        unimplemented!("DuckdbBackend::lookup_constructor_rows is deferred to Phase 2 Commit 12")
    }

    fn get_canon_repr(&self, _val: Value, _ty: ColumnTy) -> Value {
        unimplemented!("DuckdbBackend::get_canon_repr is deferred to a later commit")
    }

    fn fresh_id(&mut self) -> Value {
        unimplemented!("DuckdbBackend::fresh_id is deferred to a later commit")
    }

    // -- rule management ----------------------------------------------------

    fn new_rule<'a>(
        &'a mut self,
        _desc: &str,
        _seminaive: bool,
    ) -> Box<dyn RuleBuilderOps + 'a> {
        unimplemented!("DuckdbBackend::new_rule is deferred to Phase 2 Commit 10")
    }

    fn free_rule(&mut self, _id: RuleId) {
        unimplemented!("DuckdbBackend::free_rule is deferred to a later commit")
    }

    fn run_rules(&mut self, _rules: &[RuleId]) -> Result<IterationReport> {
        unimplemented!("DuckdbBackend::run_rules is deferred to Phase 2 Commit 10")
    }

    fn flush_updates(&mut self) -> bool {
        // Stubbed as `false` per the Commit 9 plan: until rules can run
        // through the trait, there is nothing staged to flush.
        false
    }

    // -- primitives ---------------------------------------------------------

    fn register_external_func(
        &mut self,
        _func: Box<dyn ExternalFunction + 'static>,
    ) -> ExternalFunctionId {
        unimplemented!("DuckdbBackend::register_external_func is deferred to Phase 2 Commit 12")
    }

    fn free_external_func(&mut self, _func: ExternalFunctionId) {
        unimplemented!("DuckdbBackend::free_external_func is deferred to Phase 2 Commit 12")
    }

    fn new_panic(&mut self, _message: String) -> ExternalFunctionId {
        unimplemented!("DuckdbBackend::new_panic is deferred to Phase 2 Commit 12")
    }

    // -- typed value handles ------------------------------------------------

    fn base_value_pool(&self) -> &dyn BaseValuePool {
        unimplemented!("DuckdbBackend::base_value_pool is deferred to Phase 2 Commit 11")
    }

    fn base_value_pool_mut(&mut self) -> &mut dyn BaseValuePool {
        unimplemented!("DuckdbBackend::base_value_pool_mut is deferred to Phase 2 Commit 11")
    }

    fn container_pool(&self) -> &dyn ContainerPool {
        &self.backend_container_pool
    }

    fn container_pool_mut(&mut self) -> &mut dyn ContainerPool {
        &mut self.backend_container_pool
    }

    fn base_value_constant_dyn(
        &self,
        _value: Value,
        _ty: egglog_backend_trait::BaseValueId,
    ) -> QueryEntry {
        unimplemented!(
            "DuckdbBackend::base_value_constant_dyn is deferred to Phase 2 Commit 11"
        )
    }

    // -- capability flags ---------------------------------------------------

    fn supports_inline_table_lookups(&self) -> bool {
        false
    }

    fn supports_subsumption(&self) -> bool {
        false
    }

    fn supports_complex_merge(&self) -> bool {
        false
    }

    fn supports_containers(&self) -> bool {
        false
    }

    // -- diagnostics --------------------------------------------------------

    fn set_report_level(&mut self, level: ReportLevel) {
        self.backend_report_level = level;
    }

    fn dump_debug_info(&self) {
        // Walk every registered backend table; for each, dump rows via
        // log::info!. Use the duckdb-bridge's `functions` map directly
        // — `backend_function_names` only tracks those registered
        // through the trait, but legacy callers may have registered
        // additional tables.
        for name in self.functions.keys() {
            let sql = format!("SELECT * FROM {}", q(name));
            log::info!("== DuckDB table `{}` ==", name);
            let mut stmt = match self.conn.prepare(&sql) {
                Ok(s) => s,
                Err(e) => {
                    log::info!("  prepare failed: {e}");
                    continue;
                }
            };
            let column_count = stmt.column_count();
            let mut rows = match stmt.query([]) {
                Ok(r) => r,
                Err(e) => {
                    log::info!("  query failed: {e}");
                    continue;
                }
            };
            loop {
                match rows.next() {
                    Ok(Some(row)) => {
                        let mut parts: Vec<String> = Vec::with_capacity(column_count);
                        for i in 0..column_count {
                            match row.get_ref(i) {
                                Ok(v) => parts.push(format!("{v:?}")),
                                Err(e) => parts.push(format!("<err: {e}>")),
                            }
                        }
                        log::info!("  {}", parts.join(", "));
                    }
                    Ok(None) => break,
                    Err(e) => {
                        log::info!("  row fetch failed: {e}");
                        break;
                    }
                }
            }
        }
    }

    // -- cloning ------------------------------------------------------------

    fn clone_boxed(&self) -> Box<dyn Backend> {
        // DuckDB's `Connection` is not trivially `Clone`. The Phase 1
        // design (see `docs/backend_trait_design.md`) proposes a
        // replay-log + `COPY`-based approach; that work is deferred to
        // Phase 4 cleanup. Until then, callers that need a snapshot
        // (push/pop) cannot use the DuckDB backend.
        unimplemented!(
            "DuckdbBackend::clone_boxed is not yet implemented; see Phase 4 cleanup \
             in docs/backend_trait_design.md for the proposed replay-log approach."
        )
    }

    // -- bridge-only escape hatch ------------------------------------------

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// `unsafe impl Send + Sync` for EGraph
// ---------------------------------------------------------------------------
//
// The `Backend` trait requires `Send + Sync`. `EGraph` holds a
// `duckdb::Connection`, which is `Send` but not `Sync` (DuckDB
// connections are thread-local). Within the egglog backend the
// connection is always accessed from a single thread (DuckDB itself
// is configured with `SET threads = 1` for determinism), so for the
// purposes of the trait-object wrapper we assert both. If a future
// caller drives the backend from multiple threads concurrently this
// promise will need re-evaluation.
unsafe impl Send for EGraph {}
unsafe impl Sync for EGraph {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use egglog_backend_trait::{DefaultVal, FunctionConfig, MergeFn};

    /// Construct a `Box<dyn Backend>` from a fresh DuckDB `EGraph` and
    /// exercise the methods that are implemented in Commit 9 +
    /// Commit 13.
    ///
    /// Methods covered:
    /// - `add_table` (eq-sort constructor + function with merge)
    /// - `table_size` (empty table)
    /// - `set_report_level`
    /// - capability flags (all `false`)
    /// - `container_pool` (the Commit 13 stub)
    /// - `as_any` downcast
    #[test]
    fn dyn_backend_stub_smoke() {
        let mut backend: Box<dyn Backend> =
            Box::new(EGraph::new().expect("DuckDB EGraph::new failed"));

        // Register an eq-sort constructor `C(_) -> Id`. The trait
        // schema is `[input_id, output_id]`.
        let ctor_id = backend.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id, ColumnTy::Id],
            default: DefaultVal::FreshId,
            merge: MergeFn::UnionId,
            name: "C_dyn_test".to_string(),
            can_subsume: false,
        });
        assert_eq!(backend.table_size(ctor_id), 0);

        // Register a function `F(_) -> Id` with UnionId merge (which
        // maps to MergeMode::Min on DuckDB).
        let func_id = backend.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id, ColumnTy::Id],
            default: DefaultVal::Fail,
            merge: MergeFn::UnionId,
            name: "F_dyn_test".to_string(),
            can_subsume: false,
        });
        assert_eq!(backend.table_size(func_id), 0);

        // Capability flags must all be `false` on DuckDB.
        assert!(!backend.supports_inline_table_lookups());
        assert!(!backend.supports_subsumption());
        assert!(!backend.supports_complex_merge());
        assert!(!backend.supports_containers());

        // `set_report_level` should store without panicking.
        backend.set_report_level(ReportLevel::TimeOnly);

        // Container pool: the Commit 13 stub.
        let cp = backend.container_pool();
        assert!(!cp.enabled());
        assert!(!cp.has_container_type(TypeId::of::<u64>()));
        assert_eq!(cp.size(TypeId::of::<u64>()), 0);
        assert!(cp.get_dyn(TypeId::of::<u64>(), Value::new(0)).is_none());

        // Container registration must error.
        let err = backend
            .container_pool_mut()
            .register_val_dyn(TypeId::of::<u64>(), Box::new(0u64));
        assert!(err.is_err());

        // `as_any` downcasts to the concrete DuckDB `EGraph`.
        assert!(backend.as_any().downcast_ref::<EGraph>().is_some());

        // `flush_updates` returns false (stubbed).
        assert!(!backend.flush_updates());
    }
}
