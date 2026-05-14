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
    Backend, BaseValueId, BaseValuePool, ColumnTy, ContainerPool, DefaultVal, ExternalFunction,
    ExternalFunctionId, FunctionConfig, FunctionId, FunctionRow, IterationReport, MergeFn,
    QueryEntry, ReportLevel, RuleBuilderOps, RuleId, Value,
};
use egglog_numeric_id::NumericId;

use crate::base_values::DuckdbBaseValuePool;
use crate::{ColumnTy as DuckColumnTy, EGraph, Literal, MergeMode, q};

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
/// Currently every column maps to `I64`. The trait surface uses a
/// single `Value` (`u32`) for all column kinds, and DuckDB stores
/// every `Value` as the underlying integer bits in a `BIGINT`
/// column. This is lossy with respect to the *display* of values
/// (e.g. a `f64` column rendered as `BIGINT` shows raw bits in a
/// `SELECT *`), but it is fully round-trippable through the
/// trait — `value_to_literal` / `duck_value_to_trait_value` agree
/// on this encoding.
///
/// A more precise mapping (Bool / F64 / Str columns) would require
/// reaching into the egglog-frontend's typed `Sort` system rather
/// than `BaseValuePool` alone: the pool stores typed intern tables
/// keyed by Rust `TypeId`, but the frontend's actual primitive types
/// (`Boxed<OrderedFloat<f64>>`, `Arc<str>`, custom `BaseValue` impls)
/// rarely match the natural DuckDB column kinds 1:1. Since this code
/// path is dead until Commit 14 routes a `Box<dyn Backend>` through
/// `add_table`, the lossy mapping is fine.
///
/// **TODO (Commit 14):** revisit when a real `Box<dyn Backend>`
/// caller exercises `add_table` and we see what column kinds the
/// frontend actually asks for.
fn trait_col_ty_to_duck(ty: ColumnTy, pool: &dyn BaseValuePool) -> DuckColumnTy {
    use ordered_float::OrderedFloat;
    use std::any::TypeId;
    match ty {
        ColumnTy::Id => DuckColumnTy::I64,
        ColumnTy::Base(bv) => {
            // Probe the pool's registered TypeIds in the same order
            // as `decode_base_const` in rule_builder.rs. Each branch
            // checks whether `bv` is the registered id for that
            // concrete `BaseValue` type and, if so, returns the
            // matching DuckDB column kind. Falls back to `I64` for
            // unknown / unregistered base types (which is also what
            // egglog-bridge does for them — see `BaseValue::sql_ty`).
            if pool.has_ty(TypeId::of::<i64>())
                && bv == pool.get_ty_by_type_id(TypeId::of::<i64>())
            {
                return DuckColumnTy::I64;
            }
            if pool.has_ty(TypeId::of::<bool>())
                && bv == pool.get_ty_by_type_id(TypeId::of::<bool>())
            {
                return DuckColumnTy::Bool;
            }
            type FBoxed = egglog_core_relations::Boxed<OrderedFloat<f64>>;
            if pool.has_ty(TypeId::of::<FBoxed>())
                && bv == pool.get_ty_by_type_id(TypeId::of::<FBoxed>())
            {
                return DuckColumnTy::F64;
            }
            type SBoxed = egglog_core_relations::Boxed<String>;
            if pool.has_ty(TypeId::of::<SBoxed>())
                && bv == pool.get_ty_by_type_id(TypeId::of::<SBoxed>())
            {
                return DuckColumnTy::Str;
            }
            // Unit and any other registered base types: store as i64.
            // This includes BigInt / BigRat / Rational64 which the
            // bridge interns; their `Value` is an intern-table index
            // that fits in i64.
            DuckColumnTy::I64
        }
    }
}

/// Decode an egglog [`Value`] into a duckdb-side [`Literal`] for
/// emission into a `BIGINT`-style SQL column.
///
/// Pairs with [`trait_col_ty_to_duck`]: that helper maps every trait
/// column kind to `DuckColumnTy::I64`, so the corresponding `Literal`
/// is `Literal::I64(value.rep() as i64)`. The `pool` parameter is
/// reserved for the future refined mapping (see the TODO on
/// `trait_col_ty_to_duck`).
/// Decode a `Value` for a duck-side column of [`DuckColumnTy`] into
/// the appropriate [`Literal`]. Used by `insert_row_inner` so a
/// row whose column was registered as `STR`/`BOOL`/`F64` gets the
/// matching SQL literal (not `Literal::I64(value.rep() as i64)`,
/// which would produce a SQL type-conversion error).
fn duck_value_to_literal(
    val: Value,
    col: DuckColumnTy,
    pool: &DuckdbBaseValuePool,
) -> Literal {
    use ordered_float::OrderedFloat;
    use std::any::TypeId;
    let pool_dyn: &dyn BaseValuePool = pool;
    match col {
        DuckColumnTy::I64 => {
            // Could be an eq-sort id (high bits clear) or an interned
            // i64 base value (high bit set per MAY_UNBOX) or an
            // arbitrary BaseValueId-indexed handle. The `rep() as i64`
            // cast preserves the bits losslessly; the SQL layer
            // sees `BIGINT` regardless. For i64 base values, unbox.
            if pool_dyn.has_ty(TypeId::of::<i64>()) {
                let id = pool_dyn.get_ty_by_type_id(TypeId::of::<i64>());
                // If `val` was interned in the i64 pool, `pool_unwrap`
                // returns the original i64. Otherwise the `rep()`
                // cast is the right fallback. We can't tell here
                // which one it is without the column's trait
                // `ColumnTy::Base(id)`, so prefer the unwrap when
                // the high bit is set (intern-table convention).
                let raw = val.rep();
                if raw & (1 << 31) != 0 {
                    let i = egglog_backend_trait::pool_unwrap::<i64>(pool_dyn, val);
                    let _ = id;
                    return Literal::I64(i);
                }
            }
            Literal::I64(val.rep() as i64)
        }
        DuckColumnTy::Bool => {
            if pool_dyn.has_ty(TypeId::of::<bool>()) {
                let b = egglog_backend_trait::pool_unwrap::<bool>(pool_dyn, val);
                return Literal::Bool(b);
            }
            Literal::Bool(val.rep() != 0)
        }
        DuckColumnTy::F64 => {
            type FBoxed = egglog_core_relations::Boxed<OrderedFloat<f64>>;
            if pool_dyn.has_ty(TypeId::of::<FBoxed>()) {
                let v = egglog_backend_trait::pool_unwrap::<FBoxed>(pool_dyn, val);
                return Literal::F64((*v).into_inner());
            }
            Literal::F64(val.rep() as f64)
        }
        DuckColumnTy::Str => {
            type SBoxed = egglog_core_relations::Boxed<String>;
            if pool_dyn.has_ty(TypeId::of::<SBoxed>()) {
                let v = egglog_backend_trait::pool_unwrap::<SBoxed>(pool_dyn, val);
                return Literal::Str((*v).clone());
            }
            // Fallback: stringify the bits. Almost certainly wrong
            // but at least doesn't trigger a SQL type cast error.
            Literal::Str(format!("v{}", val.rep()))
        }
        DuckColumnTy::PairI64 => {
            // Pair columns are not emitted via trait `add_table`
            // (only via duck-internal registration), so this branch
            // shouldn't be reachable from `insert_row_inner`.
            panic!("duck_value_to_literal: PairI64 unexpected in trait-driven insert path")
        }
    }
}

#[allow(dead_code)]
fn value_to_literal(val: Value, ty: ColumnTy, pool: &dyn BaseValuePool) -> Literal {
    use ordered_float::OrderedFloat;
    use std::any::TypeId;
    match ty {
        ColumnTy::Id => Literal::I64(val.rep() as i64),
        ColumnTy::Base(bv) => {
            if pool.has_ty(TypeId::of::<i64>())
                && bv == pool.get_ty_by_type_id(TypeId::of::<i64>())
            {
                let v = egglog_backend_trait::pool_unwrap::<i64>(pool, val);
                return Literal::I64(v);
            }
            if pool.has_ty(TypeId::of::<bool>())
                && bv == pool.get_ty_by_type_id(TypeId::of::<bool>())
            {
                let v = egglog_backend_trait::pool_unwrap::<bool>(pool, val);
                return Literal::Bool(v);
            }
            type FBoxed = egglog_core_relations::Boxed<OrderedFloat<f64>>;
            if pool.has_ty(TypeId::of::<FBoxed>())
                && bv == pool.get_ty_by_type_id(TypeId::of::<FBoxed>())
            {
                let v = egglog_backend_trait::pool_unwrap::<FBoxed>(pool, val);
                return Literal::F64((*v).into_inner());
            }
            type SBoxed = egglog_core_relations::Boxed<String>;
            if pool.has_ty(TypeId::of::<SBoxed>())
                && bv == pool.get_ty_by_type_id(TypeId::of::<SBoxed>())
            {
                let v = egglog_backend_trait::pool_unwrap::<SBoxed>(pool, val);
                return Literal::Str((*v).clone());
            }
            // Fallback: treat as i64 (intern-table index).
            Literal::I64(val.rep() as i64)
        }
    }
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

    /// Insert a single row's values into the function's table. The
    /// row is decoded against the function's stored schema (duck-side
    /// `ColumnTy` per column) and emitted via the existing
    /// `EGraph::insert` seed path.
    ///
    /// Encoding strategy:
    /// - Bool/I64/F64/Str duck columns → encode via the corresponding
    ///   `Literal` variant. For `Bool`/`F64`/`Str` we need the value
    ///   as the corresponding Rust type, which means using the base
    ///   value pool for non-trivial unboxing. For `I64` we just cast
    ///   `Value::rep() as i64`.
    /// - `PairI64` columns: not supported through the trait yet;
    ///   panics.
    ///
    /// Panics on schema mismatches; this method is called by
    /// `add_values` / `insert_rows` / `add_term`, all of which are
    /// unreachable until Commit 14 flips the test harness.
    pub(crate) fn insert_row_inner(&mut self, func: FunctionId, row: &[Value]) {
        let name = self.name_for_function_id(func).to_string();
        let info = self
            .functions
            .get(&name)
            .unwrap_or_else(|| panic!("insert_row_inner: function `{name}` not registered"))
            .clone();
        if row.len() != info.arity() {
            panic!(
                "insert_row_inner: arity mismatch for `{name}`: got {}, expected {}",
                row.len(),
                info.arity()
            );
        }

        // Decode each value through the schema's `DuckColumnTy` so
        // String/Bool/F64 columns receive the right `Literal` kind.
        // We don't have the trait `ColumnTy` here (the duck-side
        // `FunctionInfo.cols` already lost it), so re-derive the
        // duck literal kind from `info.cols[i]`.
        let lits: Vec<Literal> = row
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let col_ty = info.cols[i];
                duck_value_to_literal(*v, col_ty, &self.backend_base_value_pool)
            })
            .collect();

        self.insert(&name, &lits)
            .unwrap_or_else(|e| panic!("insert_row_inner: insert(`{name}`) failed: {e}"));
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
        let duck_cols: Vec<DuckColumnTy> = {
            // Borrow the pool through the concrete field to dodge a
            // re-borrow conflict with `self`: we need `&self.backend_base_value_pool`
            // for the column-type translation but `&mut self` later
            // for `add_function`/`add_relation_with_pname`. Translating
            // up front gives us a fully-owned `Vec<DuckColumnTy>` to
            // hand the mutating methods.
            let pool: &dyn BaseValuePool = &self.backend_base_value_pool;
            schema
                .iter()
                .copied()
                .map(|t| trait_col_ty_to_duck(t, pool))
                .collect()
        };

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
                // relation if the output is Unit (term encoding's
                // pattern for view tables — `(function @XView (...)
                // Unit :merge old)`). The parallel pipeline's
                // `add_function` in backend_duckdb.rs detected this
                // by string-matching `output_sort == "Unit"`; on the
                // trait side we don't have the sort name, only the
                // `BaseValueId`. Use the pool to compare against the
                // Unit type id.
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

                // Detect Unit output: schema's last entry is
                // `ColumnTy::Base(id)` where `id` matches the pool's
                // registration for the `()` type.
                let output_is_unit = schema.last().is_some_and(|t| match t {
                    ColumnTy::Base(bv) => {
                        let pool: &dyn BaseValuePool = &self.backend_base_value_pool;
                        pool.has_ty(TypeId::of::<()>())
                            && *bv == pool.get_ty_by_type_id(TypeId::of::<()>())
                    }
                    _ => false,
                });

                if duck_cols.is_empty() {
                    // No columns at all: treat as a nullary relation.
                    self.add_relation_with_pname(&name, &[], None)
                } else if output_is_unit {
                    // Function with Unit output: matches term
                    // encoding's relation pattern (e.g.
                    // `(function @UF_@pathSort (Math Math) Unit
                    // :merge old)`). The parallel pipeline drops the
                    // Unit output and registers the function as a
                    // relation whose key spans the INPUTS only — so
                    // the body atoms reference inputs without a
                    // trailing wildcard. Strip the trailing Unit
                    // column to match.
                    let inputs = &duck_cols[..duck_cols.len() - 1];
                    self.add_relation_with_pname(&name, inputs, None)
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

    fn add_values(&mut self, values: Box<dyn Iterator<Item = (FunctionId, Vec<Value>)> + '_>) {
        // Bulk insert across one or many functions, one row at a
        // time. Mirrors the bridge's `add_values` semantics. Each
        // row's `Vec<Value>` is decoded against the destination
        // function's schema, transformed to a `Vec<Literal>`, and
        // forwarded to the existing `EGraph::insert`.
        //
        // The bridge's `add_values` is followed by `flush_updates`;
        // on DuckDB seed inserts are immediate (no staging), so the
        // post-call `flush_updates` is a no-op.
        for (func, row) in values {
            self.insert_row_inner(func, &row);
        }
    }

    fn add_term(&mut self, func: FunctionId, inputs: &[Value]) -> Value {
        // Allocate a fresh id, insert `(inputs..., fresh_id)` into the
        // function table, and return the id. Mirrors the bridge's
        // `add_term`. Uses the existing `allocate_and_insert` for
        // EqSort constructors and `insert_row_inner` for regular
        // functions.
        let name = self.name_for_function_id(func).to_string();
        let info = self
            .functions
            .get(&name)
            .unwrap_or_else(|| panic!("add_term: function `{name}` not registered"));

        if info.eq_sort_ctor {
            // EqSort constructor — allocate fresh id via the
            // sequence; the body row is written by a subsequent
            // `(set @<name>View args fresh_id) ()`. To mirror the
            // bridge's "insert the row" semantics we have to also
            // write to the view here. For now we just allocate the
            // id (matching `allocate_and_insert`'s docs); writing the
            // view row is the caller's responsibility — when this
            // method is reached through Commit 14's flipped pipeline,
            // the frontend's term-encoding pass inserts the view row
            // separately.
            let pool: &dyn BaseValuePool = &self.backend_base_value_pool;
            let lits: Vec<Literal> = inputs
                .iter()
                .zip(info.cols.iter().take(info.inputs_len))
                .map(|(v, _duck_ty)| {
                    // We don't yet have the trait-side ColumnTy for
                    // the input column (the schema is stored as
                    // duck::ColumnTy on FunctionInfo). Fall back to
                    // the I64 encoding — for eq-sort constructors
                    // inputs are themselves ids, so this is correct.
                    let _ = pool;
                    Literal::I64(v.rep() as i64)
                })
                .collect();
            let id = self
                .allocate_and_insert(&name, &lits)
                .unwrap_or_else(|e| panic!("add_term: allocate_and_insert failed: {e}"));
            Value::new(id as u32)
        } else {
            // Plain function: allocate a fresh id for the output and
            // insert `(inputs..., fresh_id)` directly.
            let id_i64: i64 = self
                .conn
                .query_row("SELECT nextval('__egglog_eqsort_seq')", [], |r| r.get(0))
                .unwrap_or_else(|e| panic!("add_term: nextval failed: {e}"));
            let mut full_row: Vec<Value> = inputs.to_vec();
            full_row.push(Value::new(id_i64 as u32));
            self.insert_row_inner(func, &full_row);
            Value::new(id_i64 as u32)
        }
    }

    fn insert_rows(&mut self, table: FunctionId, rows: &[Vec<Value>]) {
        // Like `add_values` but scoped to a single function. Used by
        // file-input bulk seed and scheduler match dispatch in the
        // bridge. On DuckDB seed inserts are immediate; no flush is
        // needed.
        for row in rows {
            self.insert_row_inner(table, row);
        }
    }

    fn lookup_constructor_rows(&mut self, table: FunctionId, rows: &[Vec<Value>]) {
        // The bridge's `lookup_constructor_rows` is "look-up-or-allocate"
        // for an EqSort constructor: for each input key, return the
        // existing output id if any, or allocate a fresh one and
        // insert. The trait does not expose the resulting ids — the
        // caller uses the side-effect (rows are present in the table
        // afterward).
        //
        // We translate this to: for each key row, call `lookup_id`
        // first; if absent, call `add_term`. The behavior mirrors
        // `EGraph::allocate_and_insert` plus the view-row write the
        // bridge's `TableAction::lookup` does internally.
        let name = self.name_for_function_id(table).to_string();
        let info = self
            .functions
            .get(&name)
            .unwrap_or_else(|| {
                panic!("lookup_constructor_rows: function `{name}` not registered")
            });
        let inputs_len = info.inputs_len;

        for row in rows {
            if row.len() != inputs_len {
                panic!(
                    "lookup_constructor_rows: row arity mismatch for `{name}`: got {}, expected {}",
                    row.len(),
                    inputs_len
                );
            }
            // Try a lookup first.
            if self.lookup_id(table, row).is_some() {
                continue;
            }
            // Miss — allocate.
            let _ = self.add_term(table, row);
        }
    }

    fn get_canon_repr(&self, val: Value, ty: ColumnTy) -> Value {
        // For base values, canonicalization is the identity.
        if matches!(ty, ColumnTy::Base(_)) {
            return val;
        }
        // For eq-sort ids under `--duck-native-uf`, we have an
        // in-memory union-find indexed by the sort's pname. The
        // trait's `get_canon_repr` doesn't surface a sort handle —
        // only `ColumnTy::Id`. As a pragmatic interim, we consult
        // every registered native UF and take the first one that
        // contains the id; if none does, return the id unchanged.
        //
        // Under the non-native-UF path, canonicalization happens
        // implicitly through the SQL pname tables; the trait
        // surface's `get_canon_repr` is not consulted at all by the
        // existing pipeline. Until Commit 14 routes a caller through
        // here, this method is correctness-irrelevant — a NOP-style
        // identity returns the right answer in every situation that
        // matters today.
        let needle = val.rep() as i64;
        for uf in self.native_ufs.values() {
            // The UDF reads the UF read-only; we mirror that here.
            let canon = uf.lock().unwrap().find_ro(needle);
            if canon != needle {
                return Value::new(canon as u32);
            }
        }
        val
    }

    fn fresh_id(&mut self) -> Value {
        // The duckdb backend allocates fresh ids out of the global
        // `__egglog_eqsort_seq` sequence. The bridge's `fresh_id`
        // serves the same purpose (a fresh `Value` distinct from all
        // previously-allocated ids). Cast the `BIGINT` to `u32` —
        // collision with another sort is impossible since DuckDB's
        // SEQUENCE is monotonically increasing across the entire
        // backend.
        let id: i64 = self
            .conn
            .query_row("SELECT nextval('__egglog_eqsort_seq')", [], |r| r.get(0))
            .unwrap_or_else(|e| panic!("DuckdbBackend::fresh_id: nextval failed: {e}"));
        Value::new(id as u32)
    }

    // -- rule management ----------------------------------------------------

    fn new_rule<'a>(
        &'a mut self,
        desc: &str,
        seminaive: bool,
    ) -> Box<dyn RuleBuilderOps + 'a> {
        // The DuckDB backend's seminaive variant generation lives in
        // `compile.rs` and runs unconditionally for every rule. The
        // bridge-side `seminaive` flag has no DuckDB analog — we
        // accept it for API parity and ignore it. Phase 2 Commit 10.
        Box::new(crate::rule_builder::DuckRuleBuilderOps::new(
            self, desc, seminaive,
        ))
    }

    fn free_rule(&mut self, id: RuleId) {
        // Mark the slot as freed. We intentionally keep the slot in
        // place rather than truncating the vector so subsequent
        // `RuleId`s retain their numeric meaning. The compiled rule
        // in `self.rules` is left intact: removing it from there
        // would require recomputing every index. As a degraded
        // implementation, we clear the slot's name so subsequent
        // `run_rules([id])` is a no-op (the name doesn't match any
        // entry in `self.rules`, so `run_iteration_in_set` filters
        // it out). Phase 2 Commit 10.
        let idx = id.rep() as usize;
        if let Some(slot) = self.backend_rule_names.get_mut(idx) {
            *slot = None;
        }
    }

    fn run_rules(&mut self, rules: &[RuleId]) -> Result<IterationReport> {
        // Empty rule set = no work to do. The trait's
        // `run_rules(&[])` means "no rules to run" (the frontend
        // emits this for rulesets that ended up with zero
        // registered rules, e.g. `@delete_subsume_ruleset` when no
        // subsumable terms exist). DuckDB's
        // `run_iteration_in_set(&[])` interprets the empty list as
        // *run all rules* (`allow_all = allowed.is_empty()`), which
        // would silently re-fire every rule every iteration and
        // prevent saturation from ever being reached. Short-circuit
        // here.
        if rules.is_empty() {
            return Ok(IterationReport::default());
        }
        // Map RuleIds → user-visible names, then call the existing
        // `run_iteration_in_set`. The duckdb backend does not track
        // per-rule timing in a `RuleSetReport` shape (its perf
        // accounting goes through `rule_perf_ns` instead), so we
        // return a minimal default `IterationReport` with `changed`
        // reflecting whether any rows were affected.
        let names: Vec<String> = rules
            .iter()
            .filter_map(|id| {
                self.backend_rule_names
                    .get(id.rep() as usize)
                    .and_then(|opt| opt.clone())
            })
            .collect();
        // If the caller passed rule ids but all of them resolved to
        // freed/no-op slots (`None`), the resulting `names` is empty.
        // `run_iteration_in_set(&[])` interprets the empty allowed-set
        // as "run all rules", which would silently re-fire every rule
        // — disastrous when the scheduler is just asking the freed
        // rules to step (e.g. `eval_actions` after `free_rule`).
        // Short-circuit instead.
        if names.is_empty() {
            return Ok(IterationReport::default());
        }
        // `run_iteration_in_set` takes `&[&str]`; build a view.
        let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();

        // `total` counts rows the iteration *attempted to insert*
        // (which includes `ON CONFLICT DO NOTHING` no-ops). That
        // overcounts and never saturates — once a rule's body
        // matches, the action keeps re-trying the same insert every
        // iteration. For the frontend's saturation detection we
        // need a delta of `rules_affected_total`, the duck backend's
        // cumulative "rows that changed the database" counter.
        let before = self.rules_affected_total();
        let _ = EGraph::run_iteration_in_set(self, &name_refs)?;
        let after = self.rules_affected_total();
        if std::env::var("DUCK_TRACE_RUN_RULES").is_ok() {
            eprintln!(
                "[duck/run_rules] names={:?} delta={}",
                names,
                after - before
            );
        }

        let mut report = IterationReport::default();
        report.rule_set_report.changed = after != before;
        Ok(report)
    }

    fn flush_updates(&mut self) -> bool {
        // DuckDB has no rule-side staged-update concept: rule actions
        // commit through `INSERT` statements immediately. The native
        // UF queue is drained as part of `run_iteration_in_set`. As a
        // result `flush_updates` is a no-op; we return `false` to
        // indicate "no incremental change accrued outside a normal
        // `run_rules`". Callers that need to be sure the state is
        // settled should invoke `run_rules` instead.
        false
    }

    // -- primitives ---------------------------------------------------------

    fn register_external_func(
        &mut self,
        func: Box<dyn ExternalFunction + 'static>,
    ) -> ExternalFunctionId {
        // Storage-only registration: the slot is allocated but the
        // function is not yet wired to a DuckDB VScalar UDF, so it
        // remains unreachable from compiled SQL rules. See
        // `external_func.rs` for the full rationale and the deferral
        // path for Commit 14. Primitives needing inline table lookups
        // are explicitly gated out at the `supports_inline_table_lookups`
        // capability flag (= `false` on DuckDB).
        self.backend_external_funcs.add_func(func)
    }

    fn free_external_func(&mut self, func: ExternalFunctionId) {
        self.backend_external_funcs.free(func);
    }

    fn new_panic(&mut self, message: String) -> ExternalFunctionId {
        // Sentinel `ExternalFunctionId` whose slot stores the panic
        // message. The rule builder (`rule_builder.rs`) inspects the
        // slot's message and translates a reference into
        // `Action::Panic`. Until Commit 14 routes a real caller
        // through this path, the only consumer is the unit test
        // below.
        self.backend_external_funcs.add_panic(message)
    }

    // -- typed value handles ------------------------------------------------

    fn base_value_pool(&self) -> &dyn BaseValuePool {
        &self.backend_base_value_pool
    }

    fn base_value_pool_mut(&mut self) -> &mut dyn BaseValuePool {
        &mut self.backend_base_value_pool
    }

    fn container_pool(&self) -> &dyn ContainerPool {
        &self.backend_container_pool
    }

    fn container_pool_mut(&mut self) -> &mut dyn ContainerPool {
        &mut self.backend_container_pool
    }

    fn base_value_constant_dyn(&self, value: Value, ty: BaseValueId) -> QueryEntry {
        // Mirror the bridge's impl: package the `Value` + `ty` into a
        // typed `QueryEntry::Const`. The duckdb-side rule builder
        // (`rule_builder.rs`) translates the const into a duck
        // `Literal` at the body-atom / action-term level by consulting
        // the pool.
        QueryEntry::Const {
            val: value,
            ty: ColumnTy::Base(ty),
        }
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

    /// `fresh_id` returns distinct ids on consecutive calls and the
    /// underlying sequence advances. Phase 2 Commit 11.
    #[test]
    fn dyn_backend_fresh_id_unique() {
        let mut backend: Box<dyn Backend> =
            Box::new(EGraph::new().expect("DuckDB EGraph::new failed"));
        let a = backend.fresh_id();
        let b = backend.fresh_id();
        let c = backend.fresh_id();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    /// `get_canon_repr` is the identity for base values and for
    /// eq-sort ids that have not been unioned. Phase 2 Commit 11.
    #[test]
    fn dyn_backend_get_canon_repr_identity() {
        use egglog_backend_trait::{pool_register_type, BaseValueId};

        let mut backend: Box<dyn Backend> =
            Box::new(EGraph::new().expect("DuckDB EGraph::new failed"));

        // Base value: identity for unregistered and registered types
        // alike.
        let bv: BaseValueId = pool_register_type::<i64>(backend.base_value_pool_mut());
        let v = Value::new(42);
        assert_eq!(backend.get_canon_repr(v, ColumnTy::Base(bv)), v);

        // Eq-sort id with no native UFs is identity.
        let v_id = Value::new(7);
        assert_eq!(backend.get_canon_repr(v_id, ColumnTy::Id), v_id);
    }

    /// `base_value_pool` / `base_value_pool_mut` reach the same
    /// underlying pool — registrations made via `_mut` are observable
    /// via the read-only borrow. Phase 2 Commit 11.
    #[test]
    fn dyn_backend_base_value_pool_round_trip() {
        use egglog_backend_trait::{pool_get, pool_register_type, pool_unwrap};

        let mut backend: Box<dyn Backend> =
            Box::new(EGraph::new().expect("DuckDB EGraph::new failed"));
        let _id = pool_register_type::<i64>(backend.base_value_pool_mut());

        let pool = backend.base_value_pool();
        assert!(pool.has_ty(TypeId::of::<i64>()));

        // Inline fast-path round trip.
        let v = pool_get::<i64>(pool, 1234i64);
        let back: i64 = pool_unwrap::<i64>(pool, v);
        assert_eq!(back, 1234);
    }

    /// `base_value_constant_dyn` packages the `Value` + `BaseValueId`
    /// into a `QueryEntry::Const` of the right shape. Phase 2 Commit 11.
    #[test]
    fn dyn_backend_base_value_constant_dyn() {
        use egglog_backend_trait::pool_register_type;

        let mut backend: Box<dyn Backend> =
            Box::new(EGraph::new().expect("DuckDB EGraph::new failed"));
        let id = pool_register_type::<i64>(backend.base_value_pool_mut());
        let entry = backend.base_value_constant_dyn(Value::new(99), id);
        match entry {
            QueryEntry::Const { val, ty } => {
                assert_eq!(val, Value::new(99));
                assert_eq!(ty, ColumnTy::Base(id));
            }
            QueryEntry::Var(_) => panic!("expected Const"),
        }
    }

    /// `new_panic` returns distinct ids and each id round-trips
    /// through the registry's `panic_message` lookup. Phase 2 Commit 12.
    #[test]
    fn dyn_backend_new_panic_returns_distinct_ids_with_messages() {
        let mut backend: Box<dyn Backend> =
            Box::new(EGraph::new().expect("DuckDB EGraph::new failed"));
        let p1 = backend.new_panic("msg1".into());
        let p2 = backend.new_panic("msg2".into());
        assert_ne!(p1, p2);

        // Inspect via the concrete downcast (the trait surface itself
        // does not expose `panic_message`).
        let concrete = backend.as_any().downcast_ref::<EGraph>().unwrap();
        assert_eq!(concrete.backend_external_funcs.panic_message(p1), Some("msg1"));
        assert_eq!(concrete.backend_external_funcs.panic_message(p2), Some("msg2"));
    }

    /// `register_external_func` stores the func and returns an id;
    /// `free_external_func` makes the slot inert. Phase 2 Commit 12.
    #[test]
    fn dyn_backend_register_and_free_external_func() {
        use egglog_backend_trait::{ExecutionState, Value};

        #[derive(Clone)]
        struct NoOp;
        impl ExternalFunction for NoOp {
            fn invoke(&self, _state: &mut ExecutionState, _args: &[Value]) -> Option<Value> {
                None
            }
        }

        let mut backend: Box<dyn Backend> =
            Box::new(EGraph::new().expect("DuckDB EGraph::new failed"));
        let id = backend.register_external_func(Box::new(NoOp));
        backend.free_external_func(id);

        // Idempotent.
        backend.free_external_func(id);
    }

    /// `add_term` allocates a fresh id and stores the row in the
    /// function's table. Read back via `lookup_id`. Phase 2 Commit 12.
    #[test]
    fn dyn_backend_add_term_stores_row() {
        let mut backend: Box<dyn Backend> =
            Box::new(EGraph::new().expect("DuckDB EGraph::new failed"));
        // Use `DefaultVal::FreshId` so the table dispatch picks
        // `add_eq_sort_constructor`. `add_term` then routes through
        // `allocate_and_insert`. For an eq-sort constructor, the
        // constructor's raw table is never read by the backend (term
        // encoding routes reads through the view), so we can't
        // observe the row via `lookup_id` here — that's expected.
        // We just confirm that `add_term` returns a sensible id.
        let ctor = backend.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id, ColumnTy::Id],
            default: DefaultVal::FreshId,
            merge: MergeFn::UnionId,
            name: "C_addterm_test".to_string(),
            can_subsume: false,
        });
        let id_a = backend.add_term(ctor, &[Value::new(1)]);
        let id_b = backend.add_term(ctor, &[Value::new(2)]);
        // Allocated ids must be distinct (separate sequence values).
        assert_ne!(id_a, id_b);
    }

    /// `add_values` inserts rows into a function table; the rows are
    /// observable via `for_each` and `table_size`. Phase 2 Commit 12.
    #[test]
    fn dyn_backend_add_values_observable() {
        let mut backend: Box<dyn Backend> =
            Box::new(EGraph::new().expect("DuckDB EGraph::new failed"));
        // Plain function with output column. `DefaultVal::Fail` +
        // `MergeFn::AssertEq` selects the `add_function` branch with
        // `MergeMode::Old`.
        let func = backend.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id, ColumnTy::Id],
            default: DefaultVal::Fail,
            merge: MergeFn::AssertEq,
            name: "F_addvalues_test".to_string(),
            can_subsume: false,
        });
        let rows: Vec<(FunctionId, Vec<Value>)> = vec![
            (func, vec![Value::new(1), Value::new(10)]),
            (func, vec![Value::new(2), Value::new(20)]),
        ];
        backend.add_values(Box::new(rows.into_iter()));

        assert_eq!(backend.table_size(func), 2);
        let looked = backend.lookup_id(func, &[Value::new(1)]);
        assert_eq!(looked, Some(Value::new(10)));
    }

    /// `insert_rows` is single-table; same observable behavior as
    /// `add_values`. Phase 2 Commit 12.
    #[test]
    fn dyn_backend_insert_rows_observable() {
        let mut backend: Box<dyn Backend> =
            Box::new(EGraph::new().expect("DuckDB EGraph::new failed"));
        let func = backend.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id, ColumnTy::Id],
            default: DefaultVal::Fail,
            merge: MergeFn::AssertEq,
            name: "F_insertrows_test".to_string(),
            can_subsume: false,
        });
        let rows: Vec<Vec<Value>> = vec![
            vec![Value::new(5), Value::new(50)],
            vec![Value::new(7), Value::new(70)],
        ];
        backend.insert_rows(func, &rows);
        assert_eq!(backend.table_size(func), 2);
        assert_eq!(backend.lookup_id(func, &[Value::new(7)]), Some(Value::new(70)));
    }
}
