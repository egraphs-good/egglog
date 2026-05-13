//! `impl egglog_backend_trait::Backend for EGraph`
//!
//! Phase 2 Commit 4 — passthrough implementation. Each trait method
//! delegates to an existing inherent method on `EGraph`. The frontend
//! `EGraph` in `src/lib.rs` does **not yet** use this impl; that flip lands
//! in Commit 8. For now this impl is "dead code" exercised only by the unit
//! tests at the bottom of this file.
//!
//! ## How the dyn sub-pools work
//!
//! `Backend::base_value_pool()` and `Backend::container_pool()` return
//! `&dyn BaseValuePool` / `&dyn ContainerPool`. The implementations of those
//! sub-traits live on wrapper newtypes `BridgeBaseValuePool` and
//! `BridgeContainerPool` (defined below). Each wrapper holds a reference to
//! the underlying `BaseValues` / `ContainerValues` and dispatches to the
//! existing inherent methods.
//!
//! ## Dyn registration paths
//!
//! `BaseValues::register_type<P>` is generic over `P: BaseValue` and cannot
//! be invoked through a pure `TypeId` API alone — constructing the typed
//! intern table requires knowing `P` at compile time. Approach (a) from
//! the Phase 1 design doc resolves this by routing registration through a
//! caller-supplied factory closure (see
//! `BaseValues::register_type_dyn` in core-relations and the
//! `BridgeBaseValuePool::register_type_dyn` impl below). The
//! `pool_register_type<T>` helper in `egglog-backend-trait` builds the
//! factory for the typed table and hands it through; the bridge's wrapper
//! forwards to the new core-relations API.
//!
//! `intern_dyn` and `unwrap_dyn` use the parallel
//! `BaseValues::intern_dyn_by_id` / `unwrap_dyn_by_id` helpers, which in
//! turn dispatch through `DynamicInternTable`'s `intern_dyn` /
//! `unwrap_dyn` methods on the registered typed table.
//!
//! ### Containers: read-only dyn dispatch only
//!
//! `ContainerPool`'s read-only methods (`get_dyn`, `for_each_dyn`, `size`,
//! `has_container_type`) are wired through the new
//! `ContainerValues::{get_dyn, for_each_dyn, size_dyn, has_container_type}`
//! helpers. `register_val_dyn` still returns an error: the bridge's
//! container `register_val<C>` requires a live `ExecutionState` plus
//! access to the egraph's id counter and the UF-aware merge closure that
//! `EGraph::register_container_ty<C>` provides, neither of which is
//! reachable from a `&mut ContainerValues` alone. Container registration
//! continues to go through the concrete `EGraph::register_container_ty<C>`
//! / `EGraph::get_container_value<C>` APIs.

use std::any::{Any, TypeId};

use anyhow::{Result, anyhow};
use egglog_backend_trait::{
    Backend, BaseValuePool, ColumnTy, ContainerPool, ExternalFunction, ExternalFunctionId,
    FunctionConfig, FunctionId, FunctionRow, IterationReport, QueryEntry, ReportLevel, RuleId,
    RuleBuilderOps, Value, Variable,
};
use egglog_core_relations::{BaseValueId, BaseValues, ContainerValues, DynamicInternTable};

use crate::EGraph;

// ---------------------------------------------------------------------------
// `BridgeRuleBuilderOps` — newtype wrapper around the bridge's RuleBuilder.
// ---------------------------------------------------------------------------

/// Trait-object form of an in-progress rule. Wraps the bridge's
/// [`crate::RuleBuilder`] and delegates every operation. The wrapper exists
/// because [`Backend::new_rule`] returns `Box<dyn RuleBuilderOps + '_>`,
/// which requires a sized concrete impl behind it.
struct BridgeRuleBuilderOps<'a> {
    inner: crate::RuleBuilder<'a>,
}

impl<'a> RuleBuilderOps for BridgeRuleBuilderOps<'a> {
    fn new_var(&mut self, ty: ColumnTy) -> QueryEntry {
        // The bridge's inherent `new_var` returns a `Variable`; the trait
        // surface uses `QueryEntry`, so wrap.
        let var: Variable = self.inner.new_var(ty);
        QueryEntry::Var(var)
    }

    fn new_var_named(&mut self, ty: ColumnTy, name: &str) -> QueryEntry {
        self.inner.new_var_named(ty, name)
    }

    fn query_table(
        &mut self,
        func: FunctionId,
        entries: &[QueryEntry],
        is_subsumed: Option<bool>,
    ) -> Result<()> {
        // The bridge returns an `AtomId` we don't use through the trait
        // boundary; map it to `()`.
        self.inner.query_table(func, entries, is_subsumed).map(|_| ())
    }

    fn query_prim(
        &mut self,
        func: ExternalFunctionId,
        entries: &[QueryEntry],
        ret_ty: ColumnTy,
    ) -> Result<()> {
        self.inner.query_prim(func, entries, ret_ty)
    }

    fn call_external_func(
        &mut self,
        func: ExternalFunctionId,
        args: &[QueryEntry],
        ret_ty: ColumnTy,
        panic_msg: String,
    ) -> QueryEntry {
        // The bridge takes a `FnOnce -> String`; we collapse the closure
        // form to the eager `String` form here.
        let var = self
            .inner
            .call_external_func(func, args, ret_ty, move || panic_msg);
        QueryEntry::Var(var)
    }

    fn lookup(
        &mut self,
        func: FunctionId,
        entries: &[QueryEntry],
        panic_msg: String,
    ) -> QueryEntry {
        let var = self.inner.lookup(func, entries, move || panic_msg);
        QueryEntry::Var(var)
    }

    fn subsume(&mut self, func: FunctionId, entries: &[QueryEntry]) -> Result<()> {
        // Bridge's inherent `subsume` is infallible; the trait returns
        // `Result<()>` so DuckDB can error.
        self.inner.subsume(func, entries);
        Ok(())
    }

    fn set(&mut self, func: FunctionId, entries: &[QueryEntry]) {
        self.inner.set(func, entries);
    }

    fn remove(&mut self, func: FunctionId, entries: &[QueryEntry]) {
        self.inner.remove(func, entries);
    }

    fn union(&mut self, l: QueryEntry, r: QueryEntry) {
        self.inner.union(l, r);
    }

    fn panic(&mut self, message: String) {
        self.inner.panic(message);
    }

    fn build(self: Box<Self>) -> Result<RuleId> {
        Ok(self.inner.build())
    }
}

// ---------------------------------------------------------------------------
// `BridgeBaseValuePool` — wrapper around &BaseValues / &mut BaseValues
// ---------------------------------------------------------------------------

/// Wrapper newtype exposing `BaseValues` through the [`BaseValuePool`]
/// trait. Stored inline inside `EGraph::base_value_pool()`'s return value via
/// an unsafe pointer cast trick? No — we return a stored shim instead.
///
/// In practice we cannot return `&dyn BaseValuePool` borrowed from a local
/// wrapper because the wrapper is short-lived. Instead, `BridgeBaseValuePool`
/// is stored on `EGraph` (a tiny zero-sized phantom struct), and its `Deref`
/// to `BaseValues` is implemented via an unsafe-but-safe `BaseValues` field
/// reach. To avoid that complexity, we use a simpler approach: implement
/// `BaseValuePool` directly for `BaseValues` via a helper type.
///
/// Concretely we implement `BaseValuePool` for `BaseValues` via a transparent
/// newtype `BaseValuesAsPool`, and `EGraph` returns the underlying
/// `&BaseValues` cast as `&dyn BaseValuePool` through this newtype.
#[repr(transparent)]
struct BaseValuesAsPool(BaseValues);

impl BaseValuePool for BaseValuesAsPool {
    fn register_type_dyn(
        &mut self,
        type_id: TypeId,
        factory: Box<dyn FnOnce() -> Box<dyn DynamicInternTable>>,
    ) -> BaseValueId {
        self.0.register_type_dyn(type_id, factory)
    }

    fn get_ty_by_type_id(&self, type_id: TypeId) -> BaseValueId {
        self.0.get_ty_by_id(type_id)
    }

    fn intern_dyn(&self, ty: BaseValueId, value: Box<dyn Any + Send + Sync>) -> Value {
        // Deref the box to a `&dyn Any` and feed it through the typed
        // table's dyn-dispatch intern path.
        let any_ref: &dyn Any = &*value;
        self.0.intern_dyn_by_id(ty, any_ref)
    }

    fn unwrap_dyn(&self, ty: BaseValueId, val: Value) -> Box<dyn Any + Send + Sync> {
        self.0.unwrap_dyn_by_id(ty, val)
    }

    fn has_ty(&self, type_id: TypeId) -> bool {
        self.0.has_ty_by_id(type_id)
    }
}

// ---------------------------------------------------------------------------
// `BridgeContainerPool` — wrapper around &ContainerValues
// ---------------------------------------------------------------------------

/// Same shape as [`BaseValuesAsPool`], for `ContainerValues`.
#[repr(transparent)]
struct ContainerValuesAsPool(ContainerValues);

impl ContainerPool for ContainerValuesAsPool {
    fn has_container_type(&self, type_id: TypeId) -> bool {
        self.0.has_container_type(type_id)
    }

    fn enabled(&self) -> bool {
        true
    }

    fn get_dyn(&self, ty: TypeId, val: Value) -> Option<Box<dyn Any + Send + Sync>> {
        self.0.get_dyn(ty, val)
    }

    fn register_val_dyn(
        &mut self,
        _ty: TypeId,
        _value: Box<dyn Any + Send + Sync>,
    ) -> Result<Value> {
        // See file-level docs: registration of container values requires
        // a live `ExecutionState`, the egraph's id counter, and the
        // UF-aware merge closure. None of these are reachable from a bare
        // `&mut ContainerValues`. Callers must continue to use the
        // concrete `egglog_bridge::EGraph::get_container_value::<C>` API
        // (which is what `EGraph::with_execution_state` + the typed
        // `ContainerValues::register_val<C>` does today).
        Err(anyhow!(
            "BridgeContainerPool::register_val_dyn is not supported on the bridge wrapper: \
             container registration requires an ExecutionState and the egraph's id counter; \
             use egglog_bridge::EGraph::get_container_value::<C>(c) instead."
        ))
    }

    fn for_each_dyn(&self, ty: TypeId, f: &mut dyn FnMut(Value, &dyn Any)) {
        self.0.for_each_dyn(ty, f)
    }

    fn size(&self, ty: TypeId) -> usize {
        self.0.size_dyn(ty)
    }
}

// ---------------------------------------------------------------------------
// `impl Backend for EGraph`
// ---------------------------------------------------------------------------

impl Backend for EGraph {
    // -- table lifecycle ----------------------------------------------------

    fn add_table(&mut self, config: FunctionConfig) -> FunctionId {
        EGraph::add_table(self, config)
    }

    fn table_size(&self, table: FunctionId) -> usize {
        EGraph::table_size(self, table)
    }

    fn approx_table_size(&self, table: FunctionId) -> usize {
        EGraph::approx_table_size(self, table)
    }

    // -- iteration ----------------------------------------------------------

    fn for_each(
        &self,
        table: FunctionId,
        f: &mut dyn for<'r> FnMut(FunctionRow<'r>),
    ) {
        EGraph::for_each(self, table, |row| f(row));
    }

    fn for_each_while(
        &self,
        table: FunctionId,
        f: &mut dyn for<'r> FnMut(FunctionRow<'r>) -> bool,
    ) {
        EGraph::for_each_while(self, table, |row| f(row));
    }

    // -- direct access ------------------------------------------------------

    fn lookup_id(&self, func: FunctionId, key: &[Value]) -> Option<Value> {
        EGraph::lookup_id(self, func, key)
    }

    fn add_values(&mut self, values: Box<dyn Iterator<Item = (FunctionId, Vec<Value>)> + '_>) {
        EGraph::add_values(self, values);
    }

    fn add_term(&mut self, func: FunctionId, inputs: &[Value]) -> Value {
        EGraph::add_term(self, func, inputs)
    }

    fn insert_rows(&mut self, table: FunctionId, rows: &[Vec<Value>]) {
        // Wraps `with_execution_state` around a loop of `TableAction::insert`.
        // Mirrors the body that used to live at `src/lib.rs:1611` (input_file
        // bulk insert for non-Constructor functions) and `src/scheduler.rs:225`
        // (scheduler match canonicalization). The frontend calls
        // `flush_updates` afterward.
        let table_action = crate::TableAction::new(self, table);
        self.with_execution_state(|es| {
            for row in rows.iter() {
                table_action.insert(es, row.iter().copied());
            }
        });
    }

    fn lookup_constructor_rows(&mut self, table: FunctionId, rows: &[Vec<Value>]) {
        // Wraps `with_execution_state` around a loop of `TableAction::lookup`.
        // Mirrors the body that used to live at `src/lib.rs:1618` (input_file
        // bulk insert for Constructor functions).
        let table_action = crate::TableAction::new(self, table);
        self.with_execution_state(|es| {
            for row in rows.iter() {
                table_action.lookup(es, row);
            }
        });
    }

    fn get_canon_repr(&self, val: Value, ty: ColumnTy) -> Value {
        EGraph::get_canon_repr(self, val, ty)
    }

    fn fresh_id(&mut self) -> Value {
        EGraph::fresh_id(self)
    }

    // -- rule management ----------------------------------------------------

    fn new_rule<'a>(
        &'a mut self,
        desc: &str,
        seminaive: bool,
    ) -> Box<dyn RuleBuilderOps + 'a> {
        let inner = EGraph::new_rule(self, desc, seminaive);
        Box::new(BridgeRuleBuilderOps { inner })
    }

    fn free_rule(&mut self, id: RuleId) {
        EGraph::free_rule(self, id);
    }

    fn run_rules(&mut self, rules: &[RuleId]) -> Result<IterationReport> {
        EGraph::run_rules(self, rules)
    }

    fn flush_updates(&mut self) -> bool {
        EGraph::flush_updates(self)
    }

    // -- primitives ---------------------------------------------------------

    fn register_external_func(
        &mut self,
        func: Box<dyn ExternalFunction + 'static>,
    ) -> ExternalFunctionId {
        EGraph::register_external_func(self, func)
    }

    fn free_external_func(&mut self, func: ExternalFunctionId) {
        EGraph::free_external_func(self, func);
    }

    fn new_panic(&mut self, message: String) -> ExternalFunctionId {
        EGraph::new_panic(self, message)
    }

    // -- typed value handles ------------------------------------------------

    fn base_value_pool(&self) -> &dyn BaseValuePool {
        // SAFETY: `BaseValuesAsPool` is `#[repr(transparent)]` over
        // `BaseValues`, so transmuting `&BaseValues` to `&BaseValuesAsPool`
        // is sound — they share the same memory layout.
        let bvs: &BaseValues = self.base_values();
        let as_pool: &BaseValuesAsPool = unsafe { &*(bvs as *const BaseValues as *const BaseValuesAsPool) };
        as_pool
    }

    fn base_value_pool_mut(&mut self) -> &mut dyn BaseValuePool {
        let bvs: &mut BaseValues = self.base_values_mut();
        let as_pool: &mut BaseValuesAsPool =
            unsafe { &mut *(bvs as *mut BaseValues as *mut BaseValuesAsPool) };
        as_pool
    }

    fn container_pool(&self) -> &dyn ContainerPool {
        // SAFETY: `ContainerValuesAsPool` is `#[repr(transparent)]` over
        // `ContainerValues`.
        let cvs: &ContainerValues = self.container_values();
        let as_pool: &ContainerValuesAsPool =
            unsafe { &*(cvs as *const ContainerValues as *const ContainerValuesAsPool) };
        as_pool
    }

    fn container_pool_mut(&mut self) -> &mut dyn ContainerPool {
        let cvs: &mut ContainerValues = self.container_values_mut();
        let as_pool: &mut ContainerValuesAsPool =
            unsafe { &mut *(cvs as *mut ContainerValues as *mut ContainerValuesAsPool) };
        as_pool
    }

    fn base_value_constant_dyn(&self, value: Value, ty: BaseValueId) -> QueryEntry {
        QueryEntry::Const {
            val: value,
            ty: ColumnTy::Base(ty),
        }
    }

    // -- capability flags ---------------------------------------------------

    fn supports_inline_table_lookups(&self) -> bool {
        true
    }

    fn supports_subsumption(&self) -> bool {
        true
    }

    fn supports_complex_merge(&self) -> bool {
        true
    }

    fn supports_containers(&self) -> bool {
        true
    }

    // -- diagnostics --------------------------------------------------------

    fn set_report_level(&mut self, level: ReportLevel) {
        EGraph::set_report_level(self, level);
    }

    fn dump_debug_info(&self) {
        EGraph::dump_debug_info(self);
    }

    // -- cloning ------------------------------------------------------------

    fn clone_boxed(&self) -> Box<dyn Backend> {
        Box::new(self.clone())
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use egglog_backend_trait::{ColumnTy, DefaultVal, FunctionConfig, MergeFn};
    use egglog_numeric_id::NumericId;

    /// Exercise the trait surface against a fresh bridge `EGraph`. This
    /// confirms (a) `EGraph: Backend` compiles, (b) `Box<dyn Backend>` is a
    /// usable owned form, and (c) the basic table-lifecycle / iteration
    /// methods round-trip through the trait correctly.
    #[test]
    fn dyn_backend_add_table_for_each() {
        let mut backend: Box<dyn Backend> = Box::new(EGraph::default());

        // Register a 1-arg function: `R(_) -> Id`.
        let func_id = backend.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id, ColumnTy::Id],
            default: DefaultVal::FreshId,
            merge: MergeFn::UnionId,
            name: "R".to_string(),
            can_subsume: false,
        });

        // Empty initially.
        assert_eq!(backend.table_size(func_id), 0);

        // Add two terms; both should produce ids via add_term, and the
        // function table should then contain those rows.
        let v0 = backend.add_term(func_id, &[Value::new(100)]);
        let v1 = backend.add_term(func_id, &[Value::new(200)]);
        assert_ne!(v0, v1);
        assert_eq!(backend.table_size(func_id), 2);

        // Iterate via for_each (through the trait surface).
        let mut count = 0usize;
        let mut found_inputs: Vec<Value> = Vec::new();
        backend.for_each(func_id, &mut |row: FunctionRow<'_>| {
            count += 1;
            found_inputs.push(row.vals[0]);
        });
        assert_eq!(count, 2);
        found_inputs.sort_by_key(|v| v.rep());
        assert_eq!(found_inputs, vec![Value::new(100), Value::new(200)]);

        // lookup_id returns the same value add_term produced.
        let looked = backend.lookup_id(func_id, &[Value::new(100)]);
        assert_eq!(looked, Some(v0));

        // Capability flags for the bridge.
        assert!(backend.supports_inline_table_lookups());
        assert!(backend.supports_subsumption());
        assert!(backend.supports_complex_merge());
        assert!(backend.supports_containers());

        // Clone via the trait's `clone_boxed`; the cloned backend should
        // see the same rows.
        let cloned = backend.clone_boxed();
        assert_eq!(cloned.table_size(func_id), 2);
    }

    /// Sanity-check `Box<dyn Backend>: Clone` works.
    #[test]
    fn dyn_backend_clone() {
        let backend: Box<dyn Backend> = Box::new(EGraph::default());
        let _cloned: Box<dyn Backend> = backend.clone();
    }

    /// Exercise the dyn-aware `BaseValuePool` path end-to-end on `i64`:
    /// register the type through `pool_register_type<T>`, intern a value
    /// through `pool_get<T>`, and unwrap it through `pool_unwrap<T>`.
    ///
    /// `i64` uses `MAY_UNBOX = true` with an intern-table fallback for
    /// values whose top bit is set; we cover both the inline and the
    /// interned path to confirm the dyn dispatch reaches the typed table.
    #[test]
    fn dyn_base_value_pool_register_intern_unwrap_i64() {
        use egglog_backend_trait::{pool_get, pool_get_ty, pool_register_type, pool_unwrap};

        let mut backend: Box<dyn Backend> = Box::new(EGraph::default());

        // Register i64 through the dyn-friendly helper.
        let pool = backend.base_value_pool_mut();
        let id_via_dyn = pool_register_type::<i64>(pool);
        // Re-registering returns the same id (idempotent).
        let id_again = pool_register_type::<i64>(pool);
        assert_eq!(id_via_dyn, id_again);

        // has_ty reports registration.
        assert!(backend.base_value_pool().has_ty(TypeId::of::<i64>()));

        // Inline path: small i64 round-trips via MAY_UNBOX without touching
        // the typed intern table.
        let inline_pool = backend.base_value_pool();
        let small: i64 = 42;
        let small_val = pool_get::<i64>(inline_pool, small);
        let small_back: i64 = pool_unwrap::<i64>(inline_pool, small_val);
        assert_eq!(small_back, small);

        // Interned-path coverage: an i64 whose top bit is set forces the
        // intern-table fallback in `BaseInternTable::intern`. The fallback
        // is exercised by `pool_get<T>` because `try_box` returns `None`
        // for these values.
        let big: i64 = 0x80_00_00_00i64 | 0x12_34_56_78; // top bit set
        let big_val = pool_get::<i64>(inline_pool, big);
        let big_back: i64 = pool_unwrap::<i64>(inline_pool, big_val);
        assert_eq!(big_back, big);

        // Same value should intern to the same id.
        let big_val2 = pool_get::<i64>(inline_pool, big);
        assert_eq!(big_val, big_val2);

        // The trait-level `intern_dyn` / `unwrap_dyn` accept boxed values.
        // Confirm those round-trip through the bridge wrapper as well.
        let id_for_i64 = pool_get_ty::<i64>(inline_pool);
        let boxed: Box<dyn Any + Send + Sync> = Box::new(big);
        let via_intern = inline_pool.intern_dyn(id_for_i64, boxed);
        assert_eq!(via_intern, big_val);
        let unboxed = inline_pool.unwrap_dyn(id_for_i64, via_intern);
        let unboxed_i64 = *unboxed
            .downcast::<i64>()
            .expect("unwrap_dyn returned wrong type");
        assert_eq!(unboxed_i64, big);
    }

    /// Confirm a non-`MAY_UNBOX` base value (`String`) also round-trips
    /// through the dyn registration / intern / unwrap path.
    #[test]
    fn dyn_base_value_pool_register_intern_unwrap_string() {
        use egglog_backend_trait::{pool_get, pool_register_type, pool_unwrap};

        let mut backend: Box<dyn Backend> = Box::new(EGraph::default());
        let _id = pool_register_type::<String>(backend.base_value_pool_mut());

        let pool = backend.base_value_pool();
        let s = String::from("hello");
        let v = pool_get::<String>(pool, s.clone());
        // Re-interning the same string returns the same id.
        let v2 = pool_get::<String>(pool, s.clone());
        assert_eq!(v, v2);

        let back: String = pool_unwrap::<String>(pool, v);
        assert_eq!(back, s);
    }

    /// `Backend::as_any` must downcast successfully to the concrete
    /// backend type — this is the path call sites in `src/prelude.rs`
    /// will use to reach `TableAction::new` / `UnionAction::new` in
    /// Commit 8.
    #[test]
    fn dyn_backend_as_any_downcasts_to_bridge_egraph() {
        let backend: Box<dyn Backend> = Box::new(EGraph::default());
        let downcast = backend.as_any().downcast_ref::<EGraph>();
        assert!(
            downcast.is_some(),
            "as_any must downcast to the concrete bridge EGraph type"
        );
    }

    /// `as_any_mut` must also downcast.
    #[test]
    fn dyn_backend_as_any_mut_downcasts_to_bridge_egraph() {
        let mut backend: Box<dyn Backend> = Box::new(EGraph::default());
        let downcast = backend.as_any_mut().downcast_mut::<EGraph>();
        assert!(
            downcast.is_some(),
            "as_any_mut must downcast to the concrete bridge EGraph type"
        );
    }

    /// Confirm `has_ty` reports `false` for an unregistered type and
    /// `true` after registration.
    ///
    /// Uses `num_rational::Rational64` (re-exported via the bridge's
    /// `num-rational` dep) since `BaseValue` is implemented for it in
    /// core-relations and the bridge does not auto-register it.
    #[test]
    fn dyn_base_value_pool_has_ty() {
        use egglog_backend_trait::pool_register_type;

        let mut backend: Box<dyn Backend> = Box::new(EGraph::default());
        assert!(
            !backend
                .base_value_pool()
                .has_ty(TypeId::of::<num_rational::Rational64>())
        );
        let _ = pool_register_type::<num_rational::Rational64>(backend.base_value_pool_mut());
        assert!(
            backend
                .base_value_pool()
                .has_ty(TypeId::of::<num_rational::Rational64>())
        );
    }
}
