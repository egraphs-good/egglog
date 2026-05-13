//! # egglog-backend-trait
//!
//! A backend-agnostic interface to an egglog egraph. This crate exposes the
//! [`Backend`] trait, along with companion traits ([`RuleBuilderOps`],
//! [`BaseValuePool`], [`ContainerPool`]), so that the frontend `EGraph` in
//! the top-level `egglog` crate can drive either the in-memory reference
//! backend (`egglog_bridge::EGraph`) or the DuckDB-backed backend
//! (`egglog_bridge_duckdb::EGraph`) through a single dyn-compatible API.
//!
//! ## Design principles
//!
//! - **Minimal IR changes.** This crate intentionally does NOT introduce a new
//!   neutral rule IR. [`RuleBuilderOps`] mirrors `egglog_bridge::RuleBuilder`
//!   one-for-one. The reference backend's `RuleBuilderOps` impl is a trivial
//!   passthrough; the DuckDB backend's impl accumulates calls into its
//!   internal data IR and submits them to the existing `compile_rule`
//!   pipeline on `build()`. Frontend code (`BackendRule` in
//!   `src/lib.rs::EGraph`) is unchanged in shape.
//! - **Basic id and config types live here.** As of Phase 2 Commit 3,
//!   `FunctionId`, `RuleId`, `ColumnTy`, `FunctionRow`, `FunctionConfig`,
//!   `MergeFn`, `DefaultVal`, `QueryEntry`, `Variable`, and `VariableId` are
//!   defined in this crate. `egglog-bridge` re-exports them so existing
//!   callers continue to work. `Value`, `BaseValueId`, `ContainerValueId`,
//!   `ExecutionState`, `ExternalFunction`, and `ExternalFunctionId` remain in
//!   `egglog-core-relations` (already a neutral crate) and are re-exported
//!   here for caller convenience.
//! - **`Backend` is `Send + Sync` and dyn-compatible.** Methods that need
//!   `T: BaseValue` or `C: ContainerValue` are factored onto
//!   [`BaseValuePool`] and [`ContainerPool`], which expose Any-based dynamic
//!   dispatch. A small set of generic helper functions (see the bottom of
//!   this module) reintroduce the per-`T` sugar on top of those dyn traits.
//! - **Cloning.** Backends must be cloneable via [`Backend::clone_boxed`].
//!   The reference backend already derives `Clone`; DuckDB will need a
//!   bespoke `clone_boxed` (e.g. database snapshot or replay buffer) when
//!   it implements this trait.
//!
//! ## What is intentionally NOT in this trait
//!
//! - `with_execution_state`. The four callers in `src/` are migrated to
//!   dedicated trait methods in a follow-up commit; this keeps `Backend`
//!   dyn-compatible and avoids leaking the lifetime semantics of
//!   `ExecutionState`.
//! - `TableAction` / `UnionAction`. Under the minimal-change posture these
//!   remain inherent methods on the bridge. The frontend keeps using them
//!   directly. Backends that don't support them (DuckDB) error at the
//!   primitive-registration call sites that touch them. The
//!   [`Backend::supports_inline_table_lookups`] capability flag gates this.
//! - A neutral rule IR. [`RuleBuilderOps`] is the seam instead.

use std::any::{Any, TypeId};

use anyhow::Result;

use egglog_numeric_id::define_id;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------
//
// Types that live in neutral lower-level crates and are re-exported here for
// caller convenience.

pub use egglog_core_relations::{
    BaseValue, BaseValueId, ContainerValue, ContainerValueId, ExecutionState, ExternalFunction,
    ExternalFunctionId, Value,
};

pub use egglog_reports::{IterationReport, ReportLevel};

// ---------------------------------------------------------------------------
// Basic id types
// ---------------------------------------------------------------------------
//
// Each backend interprets these handles in its own internal map. The trait
// promises only "we return one to you, you give it back". The bridge's
// internal `TableId` / `core_relations::RuleId` are separate types not
// surfaced through the trait.

define_id!(pub RuleId, u32, "An egglog-style rule");
define_id!(pub FunctionId, u32, "An id representing an egglog function");

// ---------------------------------------------------------------------------
// ColumnTy
// ---------------------------------------------------------------------------

/// The type of a column (or `QueryEntry`): either an eq-sort / container id,
/// or a base value (with the [`BaseValueId`] identifying which base type).
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum ColumnTy {
    Id,
    Base(BaseValueId),
}

// ---------------------------------------------------------------------------
// FunctionConfig / DefaultVal / MergeFn
// ---------------------------------------------------------------------------

/// Properties of a function added to an [`Backend`].
pub struct FunctionConfig {
    /// The function's schema. The last column in the schema is the return type.
    pub schema: Vec<ColumnTy>,
    /// The behavior of the function when lookups are made on keys not currently present.
    pub default: DefaultVal,
    /// How to resolve FD conflicts for the function.
    pub merge: MergeFn,
    /// The function's name
    pub name: String,
    /// Whether or not subsumption is enabled for this function.
    pub can_subsume: bool,
}

/// How defaults are computed for the given function.
#[derive(Copy, Clone)]
pub enum DefaultVal {
    /// Generate a fresh UF id.
    FreshId,
    /// Cause an egglog-level panic if a lookup fails.
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
    /// The output of a merge is determined by applying the given ExternalFunction to the result
    /// of the argument merge functions.
    Primitive(ExternalFunctionId, Vec<MergeFn>),
    /// The output of a merge is determined by looking up the value for the given function and the
    /// given arguments in the egraph.
    Function(FunctionId, Vec<MergeFn>),
    /// Always return the old value for the given function.
    Old,
    /// Always return the new value for the given function.
    New,
    /// Always overwrite the new value for the given function with a constant. This is more useful
    /// as a "base case" in a more complicated merge function (e.g. one that clamps a value between
    /// 1 and 100) than it is as a standalone merge function.
    Const(Value),
}

// ---------------------------------------------------------------------------
// FunctionRow
// ---------------------------------------------------------------------------

/// A struct representing the content of a row in a function table
#[derive(Clone, Debug)]
pub struct FunctionRow<'a> {
    pub vals: &'a [Value],
    pub subsumed: bool,
}

// ---------------------------------------------------------------------------
// Variable / VariableId / QueryEntry
// ---------------------------------------------------------------------------

define_id!(pub VariableId, u32, "A variable in an egglog query");

/// A variable in a rule body / RHS, with an optional display name for
/// debugging.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Variable {
    pub id: VariableId,
    pub name: Option<Box<str>>,
}

impl Variable {
    /// Construct an unnamed variable from a [`VariableId`].
    pub fn from_id(id: VariableId) -> Self {
        Variable { id, name: None }
    }
}

/// A reference in a rule body or RHS: either a variable or a typed constant.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum QueryEntry {
    Var(Variable),
    Const {
        val: Value,
        // Constants can have a type plumbed through, particularly if they
        // correspond to a base value constant in egglog.
        ty: ColumnTy,
    },
}

impl From<Variable> for QueryEntry {
    fn from(var: Variable) -> Self {
        QueryEntry::Var(var)
    }
}

// ---------------------------------------------------------------------------
// The `Backend` trait
// ---------------------------------------------------------------------------

/// A backend that drives an egglog egraph.
///
/// Implementations: `egglog_bridge::EGraph` (reference, in-memory) and
/// `egglog_bridge_duckdb::EGraph` (DuckDB-backed). The frontend `EGraph` in
/// the `egglog` crate holds a `Box<dyn Backend>` and dispatches all state
/// access through this trait.
///
/// ## Method correspondence
///
/// Each method's doc comment names the inherent method on
/// `egglog_bridge::EGraph` that the bridge's `impl Backend` wraps. The
/// reference backend's impl is meant to be one line per method
/// (`fn foo(&self, ...) -> X { self.foo(...) }`). The DuckDB backend's
/// translations are documented in `docs/backend_trait_design.md`.
///
/// ## Dyn-compatibility
///
/// All methods are object-safe. Methods that would otherwise be generic over
/// `T: BaseValue` or `C: ContainerValue` live on the
/// [`BaseValuePool`] / [`ContainerPool`] sub-traits, which use Any-based
/// dynamic dispatch internally.
pub trait Backend: Send + Sync {
    // -- table lifecycle ----------------------------------------------------

    /// Register a function/relation/constructor and return its handle.
    ///
    /// Wraps `egglog_bridge::EGraph::add_table`.
    fn add_table(&mut self, config: FunctionConfig) -> FunctionId;

    /// Number of rows currently in the given function's table.
    ///
    /// Wraps `egglog_bridge::EGraph::table_size`.
    fn table_size(&self, table: FunctionId) -> usize;

    /// Approximate size; backends may return a fast estimate.
    ///
    /// Wraps `egglog_bridge::EGraph::approx_table_size`.
    fn approx_table_size(&self, table: FunctionId) -> usize;

    // -- iteration ----------------------------------------------------------

    /// Iterate over every row in `table`, calling `f` on each.
    ///
    /// Wraps `egglog_bridge::EGraph::for_each`.
    ///
    /// Default impl delegates to [`Backend::for_each_while`].
    fn for_each<'a>(&'a self, table: FunctionId, f: &mut dyn FnMut(FunctionRow<'a>)) {
        // The default implementation cannot be written here without a wrapper
        // because the closure types differ. Implementations should override
        // this to avoid the boolean threading overhead; the no-default
        // alternative would be to require both methods. We mark this as
        // mandatory below to keep impls explicit.
        let _ = (table, f);
        unimplemented!(
            "Backend impls must override for_each; the default exists only \
             to satisfy dyn-compatibility lint chains."
        )
    }

    /// Iterate over rows in `table`, stopping early when `f` returns `false`.
    ///
    /// Wraps `egglog_bridge::EGraph::for_each_while`.
    fn for_each_while<'a>(
        &'a self,
        table: FunctionId,
        f: &mut dyn FnMut(FunctionRow<'a>) -> bool,
    );

    // -- direct access ------------------------------------------------------

    /// Look up the output value associated with `key` in `func`.
    ///
    /// Wraps `egglog_bridge::EGraph::lookup_id`.
    fn lookup_id(&self, func: FunctionId, key: &[Value]) -> Option<Value>;

    /// Bulk-insert one or many rows across one or many functions.
    ///
    /// Wraps `egglog_bridge::EGraph::add_values`. (The boxed iterator is
    /// used to keep this method dyn-compatible.)
    fn add_values(&mut self, values: Box<dyn Iterator<Item = (FunctionId, Vec<Value>)> + '_>);

    /// Add a term-shaped row: stage `(inputs ... fresh_id)` and return the
    /// canonical id of the freshly allocated output.
    ///
    /// Wraps `egglog_bridge::EGraph::add_term`. On DuckDB this maps to an
    /// `INSERT ... RETURNING` against the function's table.
    fn add_term(&mut self, func: FunctionId, inputs: &[Value]) -> Value;

    /// Get the canonical representative of `val` according to `ty`.
    ///
    /// For `ColumnTy::Id` this is the union-find canonicalization. For
    /// `ColumnTy::Base(_)` it returns `val` unchanged.
    ///
    /// Wraps `egglog_bridge::EGraph::get_canon_repr`.
    fn get_canon_repr(&self, val: Value, ty: ColumnTy) -> Value;

    /// Allocate a fresh egraph id (counter increment, returned as a `Value`).
    ///
    /// Wraps `egglog_bridge::EGraph::fresh_id`.
    fn fresh_id(&mut self) -> Value;

    // -- rule management ----------------------------------------------------

    /// Begin building a new rule. Returns a builder whose lifetime is tied to
    /// `&mut self`. Callers populate the builder via [`RuleBuilderOps`] and
    /// finalize with [`RuleBuilderOps::build`].
    ///
    /// Wraps `egglog_bridge::EGraph::new_rule`.
    fn new_rule<'a>(
        &'a mut self,
        desc: &str,
        seminaive: bool,
    ) -> Box<dyn RuleBuilderOps + 'a>;

    /// Drop a registered rule. The handle becomes invalid.
    ///
    /// Wraps `egglog_bridge::EGraph::free_rule`.
    fn free_rule(&mut self, id: RuleId);

    /// Run one iteration of the given rule set. Returns timing and change
    /// counts; the database may have been modified even if `changed` is
    /// false (e.g. timestamps advanced).
    ///
    /// Wraps `egglog_bridge::EGraph::run_rules`.
    fn run_rules(&mut self, rules: &[RuleId]) -> Result<IterationReport>;

    /// Drain staged inserts and run a rebuild pass if the UF changed.
    /// Returns whether the database changed.
    ///
    /// Wraps `egglog_bridge::EGraph::flush_updates`.
    fn flush_updates(&mut self) -> bool;

    // -- primitives ---------------------------------------------------------

    /// Register a user-defined primitive (`ExternalFunction`).
    ///
    /// Wraps `egglog_bridge::EGraph::register_external_func`.
    ///
    /// On DuckDB, primitives that synchronously call back into table state
    /// (e.g. `TableAction::lookup` in the apply body) are not supported in
    /// v1. See [`Backend::supports_inline_table_lookups`].
    fn register_external_func(
        &mut self,
        func: Box<dyn ExternalFunction + 'static>,
    ) -> ExternalFunctionId;

    /// Drop a user-defined primitive.
    ///
    /// Wraps `egglog_bridge::EGraph::free_external_func`.
    fn free_external_func(&mut self, func: ExternalFunctionId);

    /// Register a deferred-panic primitive that stops rule execution when
    /// invoked. The returned id can be used as a fallback target in
    /// `lookup_with_fallback` and similar.
    ///
    /// Wraps `egglog_bridge::EGraph::new_panic`.
    fn new_panic(&mut self, message: String) -> ExternalFunctionId;

    // -- typed value handles (sub-traits) -----------------------------------

    /// Access the backend's [`BaseValuePool`] for typed base-value queries.
    ///
    /// Wraps `egglog_bridge::EGraph::base_values` (returns `&BaseValues`,
    /// which implements this sub-trait either directly or via a thin shim
    /// the bridge provides).
    fn base_value_pool(&self) -> &dyn BaseValuePool;

    /// Mutable access to the [`BaseValuePool`]. Used to register new
    /// `BaseValue` types.
    ///
    /// Wraps `egglog_bridge::EGraph::base_values_mut`.
    fn base_value_pool_mut(&mut self) -> &mut dyn BaseValuePool;

    /// Access the backend's [`ContainerPool`].
    ///
    /// Wraps `egglog_bridge::EGraph::container_values`.
    ///
    /// On DuckDB this returns an empty stub (see [`ContainerPool`]'s docs).
    /// All container-using egglog programs are gated out of DuckDB by the
    /// existing `program_supports_proofs` check, so the stub is never
    /// reached in practice.
    fn container_pool(&self) -> &dyn ContainerPool;

    /// Mutable access to the [`ContainerPool`].
    ///
    /// Wraps `egglog_bridge::EGraph::container_values_mut`.
    fn container_pool_mut(&mut self) -> &mut dyn ContainerPool;

    /// Build a [`QueryEntry`] constant for a base value of dynamic type.
    ///
    /// `value` is the interned `Value`; `ty` is the base-value-type id
    /// returned by `BaseValuePool::register_type` /
    /// `BaseValuePool::get_ty_by_type_id`.
    ///
    /// Wraps `egglog_bridge::EGraph::base_value_constant`. (The bridge's
    /// inherent method is generic over `T: BaseValue`; the dyn-friendly form
    /// takes the already-interned `Value` and the `BaseValueId`. A generic
    /// helper that wraps this is provided below
    /// (`base_value_constant<T>`).)
    fn base_value_constant_dyn(&self, value: Value, ty: BaseValueId) -> QueryEntry;

    // -- capability flags ---------------------------------------------------

    /// Whether this backend's user-defined primitives can synchronously
    /// call back into table state during their `apply()` body.
    ///
    /// Reference backend: `true`. DuckDB: `false` (primitives run inside
    /// DuckDB VScalar UDFs and cannot reenter the database).
    ///
    /// Callers that need `rust_rule` / `query` should gate on this flag.
    fn supports_inline_table_lookups(&self) -> bool;

    /// Whether this backend supports the `subsume` action and the
    /// `is_subsumed` filter on table atoms.
    ///
    /// Reference backend: `true`. DuckDB: `false` in v1 — the trait
    /// surfaces `subsume` on [`RuleBuilderOps`] but the DuckDB impl returns
    /// an error when called.
    fn supports_subsumption(&self) -> bool;

    /// Whether this backend supports `MergeFn::Function` and
    /// `MergeFn::Primitive`.
    ///
    /// Reference backend: `true`. DuckDB: `false` in v1.
    fn supports_complex_merge(&self) -> bool;

    /// Whether this backend supports `Vec` / `Set` / `Map` / `MultiSet`
    /// container sorts.
    ///
    /// Reference backend: `true`. DuckDB: `false` (container sorts are
    /// excluded from DuckDB test combos by the existing `supports_proofs`
    /// gate; see `docs/backend_trait_inventory.md` Section 6.3).
    fn supports_containers(&self) -> bool;

    // -- diagnostics --------------------------------------------------------

    /// Set the verbosity of the per-rule-iteration timing report.
    ///
    /// Wraps `egglog_bridge::EGraph::set_report_level`.
    fn set_report_level(&mut self, level: ReportLevel);

    /// Dump the database state to the `log::info!` channel (debug only).
    ///
    /// Wraps `egglog_bridge::EGraph::dump_debug_info`.
    fn dump_debug_info(&self);

    // -- cloning ------------------------------------------------------------

    /// Produce a deep clone of this backend.
    ///
    /// The frontend uses this for push/pop snapshot support. The reference
    /// backend derives `Clone`, so its impl is a one-liner
    /// (`Box::new(self.clone())`). The DuckDB backend will need a bespoke
    /// implementation (database snapshot / replay buffer); see
    /// `docs/backend_trait_design.md` for the chosen strategy.
    fn clone_boxed(&self) -> Box<dyn Backend>;
}

impl Clone for Box<dyn Backend> {
    fn clone(&self) -> Self {
        self.clone_boxed()
    }
}

// ---------------------------------------------------------------------------
// `RuleBuilderOps` — mirrors `egglog_bridge::RuleBuilder` one-for-one
// ---------------------------------------------------------------------------

/// Operations on an in-progress rule.
///
/// This trait mirrors the public methods on
/// `egglog_bridge::RuleBuilder`. The bridge's impl is a trivial newtype
/// passthrough; the DuckDB impl accumulates calls into its internal
/// `duck::Rule` data IR and submits to `compile_rule` on
/// [`RuleBuilderOps::build`].
///
/// ## Variable creation
///
/// `new_var` and `new_var_named` allocate new variables. The returned
/// `QueryEntry` can be passed back to body atoms / actions. Each `QueryEntry`
/// carries its [`ColumnTy`] for runtime arity / type checking.
///
/// ## Unsupported on DuckDB
///
/// - [`RuleBuilderOps::subsume`]: error.
/// - Complex `MergeFn::Function` / `MergeFn::Primitive` referenced via
///   `query_prim`: error at `build()` time when the rule references such a
///   merge.
pub trait RuleBuilderOps {
    /// Bind a new variable of the given type.
    ///
    /// Wraps `RuleBuilder::new_var`.
    fn new_var(&mut self, ty: ColumnTy) -> QueryEntry;

    /// Bind a new variable of the given type with a display name (eases
    /// debugging).
    ///
    /// Wraps `RuleBuilder::new_var_named`.
    fn new_var_named(&mut self, ty: ColumnTy, name: &str) -> QueryEntry;

    /// Add a table body atom. The final entry is the function's return
    /// value. When `is_subsumed` is `Some`, the atom is constrained to rows
    /// with that subsumption bit.
    ///
    /// Wraps `RuleBuilder::query_table`.
    fn query_table(
        &mut self,
        func: FunctionId,
        entries: &[QueryEntry],
        is_subsumed: Option<bool>,
    ) -> Result<()>;

    /// Add a primitive body atom. The final entry is the return value.
    ///
    /// Wraps `RuleBuilder::query_prim`.
    fn query_prim(
        &mut self,
        func: ExternalFunctionId,
        entries: &[QueryEntry],
        ret_ty: ColumnTy,
    ) -> Result<()>;

    /// Call an external function in the RHS, panicking with `panic_msg` on
    /// failure. Returns the result variable.
    ///
    /// Wraps `RuleBuilder::call_external_func`. The closure is collapsed
    /// into a `String` here to keep the trait dyn-compatible.
    fn call_external_func(
        &mut self,
        func: ExternalFunctionId,
        args: &[QueryEntry],
        ret_ty: ColumnTy,
        panic_msg: String,
    ) -> QueryEntry;

    /// RHS: look up the value of `func(entries)`, with the function's
    /// configured default behavior on miss.
    ///
    /// Wraps `RuleBuilder::lookup`. `panic_msg` is the message used when
    /// the function is configured with `DefaultVal::Fail`.
    fn lookup(
        &mut self,
        func: FunctionId,
        entries: &[QueryEntry],
        panic_msg: String,
    ) -> QueryEntry;

    /// RHS: subsume the row keyed by `entries` in `func`.
    ///
    /// Wraps `RuleBuilder::subsume`.
    ///
    /// **DuckDB**: errors at this call site. Programs that need subsume
    /// must use the reference backend in v1.
    fn subsume(&mut self, func: FunctionId, entries: &[QueryEntry]) -> Result<()>;

    /// RHS: set `func(entries[..n-1])` to `entries[n-1]`.
    ///
    /// Wraps `RuleBuilder::set`.
    fn set(&mut self, func: FunctionId, entries: &[QueryEntry]);

    /// RHS: remove the row keyed by `entries` from `func`.
    ///
    /// Wraps `RuleBuilder::remove`.
    fn remove(&mut self, func: FunctionId, entries: &[QueryEntry]);

    /// RHS: merge two values in the union-find.
    ///
    /// Wraps `RuleBuilder::union`.
    fn union(&mut self, l: QueryEntry, r: QueryEntry);

    /// RHS: panic with the given message.
    ///
    /// Wraps `RuleBuilder::panic`.
    fn panic(&mut self, message: String);

    /// Finalize the rule. Returns the registered [`RuleId`].
    ///
    /// Wraps `RuleBuilder::build`. The DuckDB impl hands the accumulated
    /// `duck::Rule` IR off to `compile_rule` and inserts it into the
    /// backend's rule registry. If any accumulated call referenced a
    /// feature the backend does not support (e.g. subsume on DuckDB), this
    /// is where the error surfaces.
    fn build(self: Box<Self>) -> Result<RuleId>;
}

// ---------------------------------------------------------------------------
// `BaseValuePool` — dyn-compatible base-value registry
// ---------------------------------------------------------------------------

/// A registry for base-value types, exposed through dynamic dispatch so it
/// fits inside `dyn Backend`.
///
/// Generic-over-`T: BaseValue` helpers are provided at the bottom of this
/// module (`pool_get<T>`, `pool_unwrap<T>`, `pool_register_type<T>`,
/// `pool_get_ty<T>`); they wrap the dyn methods below.
///
/// The bridge's impl forwards to `egglog_core_relations::BaseValues`
/// directly. The DuckDB impl maintains its own `BaseValues`-shaped registry
/// in-process; entries for `i64`/`f64`/`bool`/`String`/`()` are encoded
/// inline into SQL columns where possible, and exotic types fall back to
/// an in-memory intern table identical to the bridge's.
pub trait BaseValuePool: Send + Sync {
    /// Register a new base-value type using its `TypeId`. The pool is
    /// responsible for storing a typed intern table for `P` keyed on the
    /// returned [`BaseValueId`]. Implementations may construct the typed
    /// table using a downcast helper provided by the caller; see the
    /// `pool_register_type<T>` free function for the typical use.
    ///
    /// Wraps `BaseValues::register_type` (called via the caller's generic
    /// adapter).
    fn register_type_dyn(&mut self, type_id: TypeId) -> BaseValueId;

    /// Look up the `BaseValueId` for a registered base-value type by its
    /// Rust `TypeId`.
    ///
    /// Wraps `BaseValues::get_ty_by_id`.
    fn get_ty_by_type_id(&self, type_id: TypeId) -> BaseValueId;

    /// Intern an opaque (already-boxed) base value, returning its `Value`
    /// handle. The `Box<dyn Any>` must contain a value of the type
    /// previously registered as `ty`.
    ///
    /// Wraps the dyn dispatch over `BaseValues::get<P>`. Concrete-`T`
    /// callers should prefer `pool_get<T>` below, which special-cases
    /// `T::MAY_UNBOX` and avoids boxing.
    fn intern_dyn(&self, ty: BaseValueId, value: Box<dyn Any + Send + Sync>) -> Value;

    /// Extract a base value of the registered type `ty` from `val`. The
    /// returned `Box<dyn Any>` holds a value of the same type that was
    /// previously interned with `intern_dyn`.
    ///
    /// Wraps the dyn dispatch over `BaseValues::unwrap<P>`. Concrete-`T`
    /// callers should prefer `pool_unwrap<T>` below.
    fn unwrap_dyn(&self, ty: BaseValueId, val: Value) -> Box<dyn Any + Send + Sync>;

    /// True iff a base-value type with the given `TypeId` is registered.
    fn has_ty(&self, type_id: TypeId) -> bool;
}

/// Generic helper: register a `T: BaseValue` with the pool and return its
/// [`BaseValueId`]. Equivalent to `BaseValues::register_type::<T>`.
///
/// This sits outside the trait so the trait remains dyn-compatible.
/// Implementations of [`BaseValuePool`] are free to recognize the `TypeId`
/// of common types and pre-initialize their intern tables.
pub fn pool_register_type<T: BaseValue>(pool: &mut dyn BaseValuePool) -> BaseValueId {
    pool.register_type_dyn(TypeId::of::<T>())
}

/// Generic helper: look up the `BaseValueId` for `T: BaseValue`.
pub fn pool_get_ty<T: BaseValue>(pool: &dyn BaseValuePool) -> BaseValueId {
    pool.get_ty_by_type_id(TypeId::of::<T>())
}

/// Generic helper: intern a typed base value into the pool, returning its
/// `Value`. Mirrors `BaseValues::get<T>`. Honors `T::MAY_UNBOX` for
/// inline-encodable types.
pub fn pool_get<T: BaseValue>(pool: &dyn BaseValuePool, value: T) -> Value {
    if T::MAY_UNBOX
        && let Some(v) = value.try_box()
    {
        return v;
    }
    let ty = pool_get_ty::<T>(pool);
    pool.intern_dyn(ty, Box::new(value))
}

/// Generic helper: extract a typed base value of `T` from a `Value`.
/// Mirrors `BaseValues::unwrap<T>`. Honors `T::MAY_UNBOX`.
///
/// # Panics
///
/// Panics if `val` does not correspond to a value of type `T` previously
/// interned in `pool` (matching the bridge's existing semantics).
pub fn pool_unwrap<T: BaseValue>(pool: &dyn BaseValuePool, val: Value) -> T {
    if T::MAY_UNBOX
        && let Some(p) = T::try_unbox(val)
    {
        return p;
    }
    let ty = pool_get_ty::<T>(pool);
    let boxed = pool.unwrap_dyn(ty, val);
    *boxed
        .downcast::<T>()
        .expect("BaseValuePool::unwrap_dyn returned wrong type")
}

// ---------------------------------------------------------------------------
// `ContainerPool` — stub-friendly container registry
// ---------------------------------------------------------------------------

/// A registry for container values (`Vec` / `Set` / `Map` / `MultiSet` /
/// user-defined `ContainerValue` impls).
///
/// ## Two implementations
///
/// - **Reference backend**: delegates to
///   `egglog_core_relations::ContainerValues`. All methods succeed; the
///   pool participates in the EGraph's rebuild loop.
/// - **DuckDB backend**: an empty stub. All accessor methods return `None`
///   / empty iterators; all mutators return errors. This is safe because
///   the term-encoding gate (`program_supports_proofs` in
///   `src/proofs/proof_encoding_helpers.rs`) excludes every container-using
///   program from DuckDB's test combos. The stub's error path is a
///   defensive measure for programmer error, never a routine code path.
///
/// ## Why not just generic-over-`C`?
///
/// `register_val<C>` / `for_each<C>` are generic over `C: ContainerValue`,
/// which is incompatible with `dyn Backend`. We expose Any-based dynamic
/// dispatch here; concrete-`C` sugar is implemented as free functions in a
/// future commit (the call sites in `src/lib.rs:1924`, `serialize.rs`,
/// `extract.rs` will each adopt the dyn API directly because they already
/// thread a `TypeId` through).
pub trait ContainerPool: Send + Sync {
    /// True iff a container type with the given `TypeId` is registered.
    fn has_container_type(&self, type_id: TypeId) -> bool;

    /// True iff this backend supports containers at all.
    ///
    /// Reference: `true`. DuckDB: `false`. Provided as a convenience so
    /// callers don't need to consult [`Backend::supports_containers`].
    fn enabled(&self) -> bool;

    /// Look up the container value associated with `val`. Returns `None`
    /// when no entry is registered.
    ///
    /// The returned `Box<dyn Any>` holds an instance of the container's
    /// registered Rust type.
    ///
    /// Wraps `ContainerValues::get_val`. On DuckDB always returns `None`.
    fn get_dyn(&self, ty: TypeId, val: Value) -> Option<Box<dyn Any + Send + Sync>>;

    /// Register a Rust container value, returning a fresh `Value` handle.
    ///
    /// On DuckDB this returns an error (containers unsupported).
    fn register_val_dyn(
        &mut self,
        ty: TypeId,
        value: Box<dyn Any + Send + Sync>,
    ) -> Result<Value>;

    /// Iterate (id, container) pairs for all registered values of a
    /// container type. The callback receives the container as a
    /// `&dyn Any` of the registered concrete type.
    ///
    /// Wraps `ContainerValues::for_each<C>`. On DuckDB this is a no-op
    /// (the stub registry is empty).
    fn for_each_dyn(
        &self,
        ty: TypeId,
        f: &mut dyn FnMut(Value, &dyn Any),
    );

    /// Number of registered values of the given container type.
    fn size(&self, ty: TypeId) -> usize;
}

/// Generic helper: register a `C: ContainerValue` with the pool, returning
/// its `Value`. On DuckDB this returns an error.
pub fn container_register_val<C: ContainerValue>(
    pool: &mut dyn ContainerPool,
    value: C,
) -> Result<Value> {
    pool.register_val_dyn(TypeId::of::<C>(), Box::new(value))
}

// ---------------------------------------------------------------------------
// Trait sanity checks
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time check that `Backend` is dyn-compatible.
    #[allow(dead_code)]
    fn assert_dyn_backend(_: &dyn Backend) {}

    /// Compile-time check that `RuleBuilderOps` is dyn-compatible.
    #[allow(dead_code)]
    fn assert_dyn_rule_builder(_: &mut dyn RuleBuilderOps) {}

    /// Compile-time check that `BaseValuePool` is dyn-compatible.
    #[allow(dead_code)]
    fn assert_dyn_base_pool(_: &dyn BaseValuePool) {}

    /// Compile-time check that `ContainerPool` is dyn-compatible.
    #[allow(dead_code)]
    fn assert_dyn_container_pool(_: &dyn ContainerPool) {}

    /// Compile-time check that `Box<dyn Backend>` is `Send + Sync` and
    /// `Clone`.
    #[allow(dead_code)]
    fn assert_box_backend_send_sync_clone(b: Box<dyn Backend>) -> Box<dyn Backend> {
        fn require_send_sync<T: Send + Sync>() {}
        require_send_sync::<Box<dyn Backend>>();
        b.clone()
    }
}
