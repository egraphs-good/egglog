//! User-facing execution state wrappers.
//!
//! Four wrappers around `core_relations::ExecutionState` expose different
//! subsets of the database API based on the context in which a primitive runs:
//!
//! | Wrapper              | DB reads | DB writes | Used for                                  |
//! |----------------------|----------|-----------|-------------------------------------------|
//! | `RuleQueryState`     | no       | no        | rule LHS (seminaive on)                   |
//! | `RuleActionState`    | no       | yes       | rule RHS (seminaive on)                   |
//! | `GlobalQueryState`   | yes      | no        | top-level query-shaped commands           |
//! | `GlobalActionState`  | yes      | yes       | top-level action-shaped commands, `eval`  |
//!
//! Capabilities are expressed via three traits; each wrapper implements the
//! ones that match its cell in the table.
//!
//! The wrappers use a single lifetime parameter. `ExecutionState<'db>`
//! carries its own lifetime for internal database borrows, but exposing
//! both an outer borrow lifetime and an inner `'db` would force a two-
//! lifetime GAT, and the implied `'db: 'outer` bound combined with HRTB
//! hits a known Rust limitation (rust-lang/rust#100013). To work around
//! that, the wrappers hold a `&mut dyn ExecStateDyn` trait object that
//! erases `'db` entirely.
//!
//! # Trust boundaries
//!
//! The `#[doc(hidden)]` `UserState::__call_external_func_unchecked` and
//! the `ExecStateWriteDb::with_raw_exec_state` methods are explicit
//! escapes out of the typed system:
//!
//! - `__call_external_func_unchecked` (every wrapper) — the `__` prefix
//!   and `_unchecked` suffix flag this as "caller is responsible for
//!   checking". Intended solely for `FunctionContainer::apply_in` in
//!   the egglog crate, which verifies the callee's `valid_contexts`
//!   covers the caller's state before dispatching.
//!
//! - `with_raw_exec_state` (only on write-capable wrappers) — hands out
//!   a raw `&mut ExecutionState`. Used to invoke `UnionAction::union`,
//!   `TableAction::{insert,remove,subsume,lookup_or_insert}`, and any
//!   other operation that needs simultaneous `&ContainerValues` and
//!   `&mut ExecutionState` access. Available on `RuleActionState` and
//!   `GlobalActionState` only; query-side states cannot reach this
//!   escape.
//!
//! Other raw escapes exist elsewhere in the public API:
//! [`crate::EGraph::with_execution_state`] and
//! [`crate::EGraph::register_external_func`], plus the legacy
//! `egglog::Primitive` trait. Each is documented in place as a trust
//! boundary.

use core_relations::{
    BaseValues, ContainerValue, ContainerValues, CounterId, ExecutionState, ExternalFunctionId,
    TableId, Value, WrappedTable,
};

use crate::core_relations;
use crate::TableAction;

/// The four contexts a primitive may run in.
///
/// Used at registration time to index primitives by the contexts they support.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Context {
    RuleQuery,
    RuleAction,
    GlobalQuery,
    GlobalAction,
}

impl Context {
    pub const ALL: [Context; 4] = [
        Context::RuleQuery,
        Context::RuleAction,
        Context::GlobalQuery,
        Context::GlobalAction,
    ];
}

/// Private dyn-compatible trait that erases `ExecutionState`'s lifetime.
///
/// All the methods the user-facing wrappers need are mirrored here. The
/// returns are tied to `&self` rather than the inner `'db` lifetime, which
/// loses some of `ExecutionState`'s borrow flexibility but gains the
/// ability to hide the lifetime parameter entirely.
pub(crate) trait ExecStateDyn: Send + Sync {
    fn base_values(&self) -> &BaseValues;
    fn container_values(&self) -> &ContainerValues;
    fn read_counter(&self, ctr: CounterId) -> usize;
    fn inc_counter(&self, ctr: CounterId) -> usize;
    fn trigger_early_stop(&self);
    fn should_stop(&self) -> bool;
    fn table_name(&self, table: TableId) -> Option<&str>;
    fn get_table(&self, table: TableId) -> &WrappedTable;
    fn stage_insert(&mut self, table: TableId, row: &[Value]);
    fn stage_remove(&mut self, table: TableId, key: &[Value]);

    /// Escape hatch that runs a closure with raw access to the
    /// `ExecutionState`. Exists because operations like container
    /// registration and union-find union need both `&ContainerValues` and
    /// `&mut ExecutionState` simultaneously, and the dyn-trait cannot
    /// expose generic methods. Users call this through the typed helpers
    /// on write-capable wrappers (`register_container`, `union`, etc.).
    fn with_raw(&mut self, f: &mut dyn FnMut(&mut ExecutionState<'_>));
}

impl<'db> ExecStateDyn for ExecutionState<'db> {
    fn base_values(&self) -> &BaseValues {
        ExecutionState::base_values(self)
    }
    fn container_values(&self) -> &ContainerValues {
        ExecutionState::container_values(self)
    }
    fn read_counter(&self, ctr: CounterId) -> usize {
        ExecutionState::read_counter(self, ctr)
    }
    fn inc_counter(&self, ctr: CounterId) -> usize {
        ExecutionState::inc_counter(self, ctr)
    }
    fn trigger_early_stop(&self) {
        ExecutionState::trigger_early_stop(self)
    }
    fn should_stop(&self) -> bool {
        ExecutionState::should_stop(self)
    }
    fn table_name(&self, table: TableId) -> Option<&str> {
        ExecutionState::table_name(self, table)
    }
    fn get_table(&self, table: TableId) -> &WrappedTable {
        ExecutionState::get_table(self, table)
    }
    fn stage_insert(&mut self, table: TableId, row: &[Value]) {
        ExecutionState::stage_insert(self, table, row)
    }
    fn stage_remove(&mut self, table: TableId, key: &[Value]) {
        ExecutionState::stage_remove(self, table, key)
    }
    fn with_raw(&mut self, f: &mut dyn FnMut(&mut ExecutionState<'_>)) {
        f(self)
    }
}

/// Runs a closure with raw `&mut ExecutionState` access and produces a
/// return value. Used by the typed helpers on write-capable state wrappers.
///
/// The callback must be called exactly once; this helper uses `Option` to
/// shuttle the result out of the non-generic dyn trait method.
fn with_raw_result<R>(
    dyn_state: &mut dyn ExecStateDyn,
    f: impl FnOnce(&mut ExecutionState<'_>) -> R,
) -> R {
    let mut slot: Option<R> = None;
    let mut f_opt = Some(f);
    dyn_state.with_raw(&mut |es| {
        let f = f_opt.take().expect("with_raw callback invoked more than once");
        slot = Some(f(es));
    });
    slot.expect("with_raw callback was never invoked")
}

/// Capabilities available in every context.
///
/// Pure operations (base values, counters, early-stop signal, metadata
/// lookups) never compromise seminaive soundness, so they live on every
/// wrapper. Container registration is also here because container
/// interning is idempotent: registering an existing container returns
/// the same `Value`, and a freshly-registered container cannot be
/// observed by rules until it appears in some table row (a separate
/// write). See issue #772 for the soundness argument.
pub trait ExecStateCore {
    fn base_values(&self) -> &BaseValues;
    fn read_counter(&self, ctr: CounterId) -> usize;
    fn inc_counter(&self, ctr: CounterId) -> usize;
    fn trigger_early_stop(&self);
    fn should_stop(&self) -> bool;
    fn table_name(&self, table: TableId) -> Option<&str>;
}

/// Database-read capabilities.
///
/// Only enabled on wrappers that run outside of a seminaive rule query:
/// `GlobalQueryState` and `GlobalActionState`. See the proposal for issue
/// #772 for the soundness reasoning.
pub trait ExecStateReadDb: ExecStateCore {
    fn get_table(&self, table: TableId) -> &WrappedTable;
}

/// Database-write capabilities.
///
/// Only enabled on wrappers that run outside of a seminaive rule query:
/// `RuleActionState` and `GlobalActionState`. Note that `ExecStateWriteDb`
/// is not object-safe — the generic `with_raw_exec_state` method rules
/// that out — so it should be used as a bound, not a `dyn` type.
pub trait ExecStateWriteDb: ExecStateCore {
    fn stage_insert(&mut self, table: TableId, row: &[Value]);
    fn stage_remove(&mut self, table: TableId, key: &[Value]);

    /// Raw `&mut ExecutionState` access. Used by primitives that need to
    /// invoke operations requiring both `&ContainerValues` and
    /// `&mut ExecutionState` at once — `UnionAction::union`,
    /// `TableAction::{insert, remove, subsume, lookup_or_insert}`, and
    /// nested external-function calls.
    fn with_raw_exec_state<R>(
        &mut self,
        f: impl FnOnce(&mut ExecutionState<'_>) -> R,
    ) -> R;
}

/// Common trait for the user-facing state wrappers.
///
/// Lets registration machinery derive a primitive's valid contexts from its
/// declared [`Primitive::State`] type without caring which of the four
/// concrete wrappers it is.
pub trait UserState<'a>: Sized + ExecStateCore {
    fn wrap(state: &'a mut ExecutionState<'_>) -> Self;
    fn valid_contexts() -> &'static [Context];
    fn container_values(&self) -> &ContainerValues;

    /// Register a container value, returning its interned `Value`.
    ///
    /// Container interning is idempotent: registering an existing
    /// container returns the same `Value`, and a freshly-interned
    /// container is not observable by any rule until it appears in a
    /// table row (a separate write). Safe in every context.
    fn register_container<C: ContainerValue>(&mut self, container: C) -> Value;

    /// Trust-boundary escape used by the core `FunctionContainer::apply_in`
    /// dispatch to invoke an external function from a pure-capability
    /// state. **Not a stable public API.** The caller must verify the
    /// function's `valid_contexts` is compatible with `Self` before calling.
    #[doc(hidden)]
    fn __call_external_func_unchecked(
        &mut self,
        id: ExternalFunctionId,
        args: &[Value],
    ) -> Option<Value>;

    /// Trust-boundary escape used by `FunctionContainer::apply_in` to
    /// dispatch a custom-function table lookup (pure read, no insert)
    /// from a state wrapper that may not otherwise expose
    /// [`ExecStateReadDb`]. **Not a stable public API.** The caller must
    /// verify the lookup is safe in the current context — namely, that
    /// `Self::valid_contexts()` does not include
    /// [`Context::RuleQuery`] (where untracked reads break seminaive).
    #[doc(hidden)]
    fn __table_lookup_unchecked(
        &mut self,
        action: &TableAction,
        key: &[Value],
    ) -> Option<Value>;
}

macro_rules! define_state_wrapper {
    ($name:ident) => {
        pub struct $name<'a> {
            pub(crate) inner: &'a mut (dyn ExecStateDyn + 'a),
        }

        impl<'a> ExecStateCore for $name<'a> {
            fn base_values(&self) -> &BaseValues {
                self.inner.base_values()
            }
            fn read_counter(&self, ctr: CounterId) -> usize {
                self.inner.read_counter(ctr)
            }
            fn inc_counter(&self, ctr: CounterId) -> usize {
                self.inner.inc_counter(ctr)
            }
            fn trigger_early_stop(&self) {
                self.inner.trigger_early_stop()
            }
            fn should_stop(&self) -> bool {
                self.inner.should_stop()
            }
            fn table_name(&self, table: TableId) -> Option<&str> {
                self.inner.table_name(table)
            }
        }
    };
}

define_state_wrapper!(RuleQueryState);
define_state_wrapper!(RuleActionState);
define_state_wrapper!(GlobalQueryState);
define_state_wrapper!(GlobalActionState);

// Reads: only on global contexts.
impl<'a> ExecStateReadDb for GlobalQueryState<'a> {
    fn get_table(&self, table: TableId) -> &WrappedTable {
        self.inner.get_table(table)
    }
}
impl<'a> ExecStateReadDb for GlobalActionState<'a> {
    fn get_table(&self, table: TableId) -> &WrappedTable {
        self.inner.get_table(table)
    }
}

// Writes: only on action contexts.
impl<'a> ExecStateWriteDb for RuleActionState<'a> {
    fn stage_insert(&mut self, table: TableId, row: &[Value]) {
        self.inner.stage_insert(table, row)
    }
    fn stage_remove(&mut self, table: TableId, key: &[Value]) {
        self.inner.stage_remove(table, key)
    }
    fn with_raw_exec_state<R>(
        &mut self,
        f: impl FnOnce(&mut ExecutionState<'_>) -> R,
    ) -> R {
        with_raw_result(self.inner, f)
    }
}
impl<'a> ExecStateWriteDb for GlobalActionState<'a> {
    fn stage_insert(&mut self, table: TableId, row: &[Value]) {
        self.inner.stage_insert(table, row)
    }
    fn stage_remove(&mut self, table: TableId, key: &[Value]) {
        self.inner.stage_remove(table, key)
    }
    fn with_raw_exec_state<R>(
        &mut self,
        f: impl FnOnce(&mut ExecutionState<'_>) -> R,
    ) -> R {
        with_raw_result(self.inner, f)
    }
}

// `register_container` lives on the `UserState` trait (see above) via the
// `impl_register_container!` macro included in each `UserState for X`
// impl below.


// `with_raw_exec_state` lives on the `ExecStateWriteDb` trait (see above),
// so `RuleActionState` and `GlobalActionState` get it automatically.

// UserState impls. `valid_contexts()` encodes that a narrower (more
// restricted) wrapper is usable in every context that a wider one is:
// `RuleQueryState` works everywhere, `GlobalActionState` only as itself.
// The trust-boundary escape `__call_external_func_unchecked` is identical
// across all four wrappers — it reaches through the dyn trait to the raw
// `ExecutionState` and invokes the given function. The safety contract is
// caller-side (the only intended caller is
// `egglog::sort::fn::FunctionContainer::apply_in`).
macro_rules! impl_unchecked_dispatch {
    () => {
        fn __call_external_func_unchecked(
            &mut self,
            id: ExternalFunctionId,
            args: &[Value],
        ) -> Option<Value> {
            with_raw_result(self.inner, |es| es.call_external_func(id, args))
        }
    };
}

// `register_container` is also identical across wrappers. It's generic in
// `C: ContainerValue`, which is fine on the `UserState` trait because the
// trait is used as a bound, not a `dyn` type.
macro_rules! impl_register_container {
    () => {
        fn register_container<C: ContainerValue>(&mut self, container: C) -> Value {
            with_raw_result(self.inner, |es| {
                es.clone().container_values().register_val(container, es)
            })
        }
    };
}

// Pure-read table lookup escape, identical across wrappers. The dyn
// trait already exposes raw `&mut ExecutionState` via `with_raw`; we
// borrow it as `&` for the lookup, which only reads.
macro_rules! impl_table_lookup {
    () => {
        fn __table_lookup_unchecked(
            &mut self,
            action: &TableAction,
            key: &[Value],
        ) -> Option<Value> {
            with_raw_result(self.inner, |es| action.lookup(es, key))
        }
    };
}

impl<'a> UserState<'a> for RuleQueryState<'a> {
    fn wrap(state: &'a mut ExecutionState<'_>) -> Self {
        RuleQueryState { inner: state }
    }
    fn valid_contexts() -> &'static [Context] {
        &Context::ALL
    }
    fn container_values(&self) -> &ContainerValues {
        self.inner.container_values()
    }
    impl_unchecked_dispatch!();
    impl_register_container!();
    impl_table_lookup!();
}

impl<'a> UserState<'a> for RuleActionState<'a> {
    fn wrap(state: &'a mut ExecutionState<'_>) -> Self {
        RuleActionState { inner: state }
    }
    fn valid_contexts() -> &'static [Context] {
        &[Context::RuleAction, Context::GlobalAction]
    }
    fn container_values(&self) -> &ContainerValues {
        self.inner.container_values()
    }
    impl_unchecked_dispatch!();
    impl_register_container!();
    impl_table_lookup!();
}

impl<'a> UserState<'a> for GlobalQueryState<'a> {
    fn wrap(state: &'a mut ExecutionState<'_>) -> Self {
        GlobalQueryState { inner: state }
    }
    fn valid_contexts() -> &'static [Context] {
        &[Context::GlobalQuery, Context::GlobalAction]
    }
    fn container_values(&self) -> &ContainerValues {
        self.inner.container_values()
    }
    impl_unchecked_dispatch!();
    impl_register_container!();
    impl_table_lookup!();
}

impl<'a> UserState<'a> for GlobalActionState<'a> {
    fn wrap(state: &'a mut ExecutionState<'_>) -> Self {
        GlobalActionState { inner: state }
    }
    fn valid_contexts() -> &'static [Context] {
        &[Context::GlobalAction]
    }
    fn container_values(&self) -> &ContainerValues {
        self.inner.container_values()
    }
    impl_unchecked_dispatch!();
    impl_register_container!();
    impl_table_lookup!();
}
