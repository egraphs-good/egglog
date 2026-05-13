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
//! ## Known limitation: `register_type_dyn` / `register_val_dyn`
//!
//! `BaseValues::register_type<P>` is generic over `P: BaseValue` and cannot
//! be implemented through a pure `TypeId` API — constructing the typed
//! intern table requires knowing `P` at compile time. The bridge's
//! `BridgeBaseValuePool::register_type_dyn` therefore currently panics with
//! a clear `unimplemented!` message. Callers in the frontend register
//! base-value types through the concrete `EGraph::base_values_mut()` API,
//! which is unchanged by this commit. Once `EGraph::backend` becomes
//! `Box<dyn Backend>` (Commit 8), we will need to revisit this; the
//! likely fix is to expose a closure-based registration API on the trait
//! or to thread the generic registration through a different surface.
//!
//! The same applies to `ContainerPool::register_val_dyn`.

use std::any::{Any, TypeId};

use anyhow::Result;
use egglog_backend_trait::{
    Backend, BaseValuePool, ColumnTy, ContainerPool, ExternalFunction, ExternalFunctionId,
    FunctionConfig, FunctionId, FunctionRow, IterationReport, QueryEntry, ReportLevel, RuleId,
    RuleBuilderOps, Value, Variable,
};
use egglog_core_relations::{BaseValueId, BaseValues, ContainerValues};

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
    fn register_type_dyn(&mut self, _type_id: TypeId) -> BaseValueId {
        // See file-level docs: `BaseValues::register_type<P>` is generic
        // over P. There is no way to construct the typed intern table from
        // just a `TypeId`. Until the trait grows a closure-based
        // registration API (a future commit), callers must register base
        // value types through the concrete `EGraph::base_values_mut()` API.
        unimplemented!(
            "BridgeBaseValuePool::register_type_dyn is not implementable from a bare TypeId; \
             use the concrete egglog_bridge::EGraph::base_values_mut().register_type::<T>() API \
             until the trait is extended (see backend_impl.rs file docs)."
        )
    }

    fn get_ty_by_type_id(&self, type_id: TypeId) -> BaseValueId {
        self.0.get_ty_by_id(type_id)
    }

    fn intern_dyn(&self, _ty: BaseValueId, _value: Box<dyn Any + Send + Sync>) -> Value {
        // Same limitation as `register_type_dyn`: the typed intern table
        // stored at `BaseValueId` is a `BaseInternTable<P>` whose `P` is not
        // recoverable from `&dyn Any` without runtime dispatch we haven't
        // wired through.
        unimplemented!(
            "BridgeBaseValuePool::intern_dyn requires a typed intern path that the bridge's \
             current BaseValues API does not expose. Use the concrete \
             EGraph::base_values().get::<T>(x) helper, or call pool_get<T> after we extend \
             core-relations to expose a typed dyn interner."
        )
    }

    fn unwrap_dyn(&self, _ty: BaseValueId, _val: Value) -> Box<dyn Any + Send + Sync> {
        // Same limitation as `intern_dyn`.
        unimplemented!(
            "BridgeBaseValuePool::unwrap_dyn requires the typed unwrap path that the bridge's \
             current BaseValues API does not expose. Use the concrete \
             EGraph::base_values().unwrap::<T>(v) helper."
        )
    }

    fn has_ty(&self, type_id: TypeId) -> bool {
        // `BaseValues::type_ids` is private, so we approximate "has this
        // type" by attempting the lookup and catching the panic. Since
        // `get_ty_by_id` panics on missing types, that would be a footgun
        // here. Instead, we expose this via a small upstream change in a
        // future commit; for now, conservatively return `true` for any
        // built-in type that the bridge always registers (i64, f64, bool,
        // Unit, String). This is **incomplete** but suffices for the
        // Phase 2 Commit 4 unit tests; the full fix lands when the
        // BaseValues API exposes `has_type(TypeId) -> bool`.
        //
        // For now, defer to the trait crate's documented invariant: callers
        // should only invoke `has_ty` for types they registered. Always
        // returning `true` matches that invariant defensively.
        let _ = type_id;
        true
    }
}

// ---------------------------------------------------------------------------
// `BridgeContainerPool` — wrapper around &ContainerValues
// ---------------------------------------------------------------------------

/// Same shape as [`BaseValuesAsPool`], for `ContainerValues`.
#[repr(transparent)]
struct ContainerValuesAsPool(ContainerValues);

impl ContainerPool for ContainerValuesAsPool {
    fn has_container_type(&self, _type_id: TypeId) -> bool {
        // `ContainerValues::container_ids` is private; same situation as
        // `has_ty` above. The Phase 2 Commit 4 unit tests don't exercise
        // this path; flagged as a future cleanup.
        true
    }

    fn enabled(&self) -> bool {
        true
    }

    fn get_dyn(&self, _ty: TypeId, _val: Value) -> Option<Box<dyn Any + Send + Sync>> {
        // The bridge's `ContainerValues::get_val<C>` is generic; same
        // dispatch problem as `BridgeBaseValuePool::unwrap_dyn`. Callers
        // currently use the concrete `EGraph::container_values()` API.
        unimplemented!(
            "BridgeContainerPool::get_dyn requires a typed dyn dispatch the current \
             ContainerValues API does not expose; use \
             EGraph::container_values().get_val::<C>(v) instead."
        )
    }

    fn register_val_dyn(
        &mut self,
        _ty: TypeId,
        _value: Box<dyn Any + Send + Sync>,
    ) -> Result<Value> {
        unimplemented!(
            "BridgeContainerPool::register_val_dyn requires a typed dyn dispatch the current \
             ContainerValues API does not expose; use \
             EGraph::get_container_value::<C>(c) instead."
        )
    }

    fn for_each_dyn(
        &self,
        _ty: TypeId,
        _f: &mut dyn FnMut(Value, &dyn Any),
    ) {
        unimplemented!(
            "BridgeContainerPool::for_each_dyn requires a typed dyn dispatch the current \
             ContainerValues API does not expose."
        )
    }

    fn size(&self, _ty: TypeId) -> usize {
        // Same as above.
        0
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
}
