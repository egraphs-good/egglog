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
//! The wrappers use a single lifetime parameter; the inner
//! `&mut dyn ExecStateDyn` trait object hides the database lifetime.
//! (Two lifetimes would force a HRTB-bound GAT, which hits
//! rust-lang/rust#100013.)
//!
//! # Name-indexed convenience methods
//!
//! `RuleActionState` and `GlobalActionState` carry a borrow of the
//! bridge's [`NamedActionRegistry`] and expose inherent
//! `insert(name, …)`, `remove`, `subsume`, `lookup`, `union`, and
//! `panic` methods that resolve the underlying handles from there.
//! The registry is populated and kept up-to-date by
//! `crate::EGraph::add_table`. Read-only wrappers carry the same
//! reference for a uniform `wrap` signature but expose no name-indexed
//! methods of their own.
//!
//! # Trust boundaries
//!
//! A few methods in this module step outside the typed-state contract.
//! They're plain safe Rust, but a careless caller can violate
//! seminaive evaluation; per-method docs spell out the constraint.
//!
//! - [`UserState::call_external_func`] (every wrapper) — invoke an
//!   external function from any state. Caller must ensure the callee's
//!   `valid_contexts` covers the caller's state. Used by
//!   `FunctionContainer::apply_in` in the egglog crate, which performs
//!   the check.
//! - [`UserState::table_lookup`] (every wrapper) — pure-read table
//!   lookup; safe everywhere except [`Context::RuleQuery`], where it's
//!   an untracked seminaive read.
//! - [`ExecStateWriteDb::with_raw_exec_state`] (only on write-capable
//!   wrappers) — hands out a raw `&mut ExecutionState`. Used to invoke
//!   `UnionAction::union`,
//!   `TableAction::{insert,remove,subsume,lookup_or_insert}`, and any
//!   other operation that needs simultaneous `&ContainerValues` and
//!   `&mut ExecutionState` access. Available on `RuleActionState` and
//!   `GlobalActionState` only; query-side states cannot reach this
//!   escape.
//!
//! Other raw escapes exist elsewhere in the public API:
//! [`crate::EGraph::with_execution_state`] and
//! [`crate::EGraph::register_external_func`]. Each is documented in
//! place as a trust boundary.

use std::collections::HashMap;
use std::ops::Deref;

use core_relations::{
    BaseValue, BaseValues, ContainerValue, ContainerValues, CounterId, ExecutionState,
    ExternalFunctionId, TableId, Value,
};
use smallvec::SmallVec;

use crate::core_relations;
use crate::{TableAction, UnionAction};

/// A live registry of action handles for use by typed primitives.
/// Owned by the bridge `EGraph` and shared with state wrappers at
/// invoke time.
///
/// Three independent kinds of handles, each in its own field:
/// - `table_actions` — one [`TableAction`] per user-defined function,
///   keyed by table name. Grows as `add_table` is called. Used by
///   the action wrappers' `insert(name, …)` / `remove(name, …)` /
///   `subsume(name, …)` / `lookup(name, …)` methods.
/// - `union_action` — the single union-find handle for this EGraph,
///   used by the wrappers' `union(x, y)` method.
/// - `default_panic_id` — a shared `ExternalFunctionId` for the
///   wrappers' `panic()` method.
pub struct NamedActionRegistry {
    table_actions: HashMap<String, TableAction>,
    union_action: UnionAction,
    default_panic_id: ExternalFunctionId,
}

impl NamedActionRegistry {
    pub(crate) fn new(
        union_action: UnionAction,
        default_panic_id: ExternalFunctionId,
    ) -> Self {
        Self {
            table_actions: HashMap::new(),
            union_action,
            default_panic_id,
        }
    }

    pub(crate) fn register_table(&mut self, name: String, action: TableAction) {
        self.table_actions.insert(name, action);
    }

    /// Look up the [`TableAction`] for a table by name, or `None` if
    /// no table with that name has been registered.
    pub fn lookup_table(&self, name: &str) -> Option<&TableAction> {
        self.table_actions.get(name)
    }

    /// The shared [`UnionAction`] for this EGraph's union-find.
    pub fn union_action(&self) -> &UnionAction {
        &self.union_action
    }

    /// The default panic external function id, used by
    /// [`RuleActionState::panic`] and [`GlobalActionState::panic`].
    pub fn default_panic_id(&self) -> ExternalFunctionId {
        self.default_panic_id
    }
}

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
/// declared `Primitive::State` type without caring which of the four
/// concrete wrappers it is. (`Primitive` lives in the egglog crate;
/// rustdoc cannot link to it across crates here, but see the egglog
/// crate root for the trait.)
pub trait UserState<'a>: Sized + ExecStateCore {
    fn wrap(state: &'a mut ExecutionState<'_>, registry: &'a NamedActionRegistry) -> Self;
    fn valid_contexts() -> &'static [Context];
    fn container_values(&self) -> &ContainerValues;

    /// Register a container value, returning its interned `Value`.
    ///
    /// Container interning is idempotent: registering an existing
    /// container returns the same `Value`, and a freshly-interned
    /// container is not observable by any rule until it appears in a
    /// table row (a separate write). Safe in every context.
    fn register_container<C: ContainerValue>(&mut self, container: C) -> Value;

    /// Invoke an external function from any state.
    ///
    /// Trust boundary: the caller must verify that the function's
    /// `valid_contexts` covers `Self::valid_contexts()` before calling
    /// — otherwise the called primitive may execute in a context where
    /// its declared `State` would be unsound. See the module-level
    /// "Trust boundaries" section. The intended caller is
    /// `FunctionContainer::apply_in` in the egglog crate, which
    /// performs the check.
    fn call_external_func(
        &mut self,
        id: ExternalFunctionId,
        args: &[Value],
    ) -> Option<Value>;

    /// Look up a row in a function table — pure read, never inserts.
    ///
    /// Trust boundary: safe everywhere except [`Context::RuleQuery`],
    /// where the read would be untracked by seminaive. The caller must
    /// verify that `Self::valid_contexts()` does not include
    /// `Context::RuleQuery` before calling. See the module-level
    /// "Trust boundaries" section.
    fn table_lookup(
        &mut self,
        action: &TableAction,
        key: &[Value],
    ) -> Option<Value>;

    /// Convert an egglog [`Value`] to a Rust base type.
    /// Sugar over `self.base_values().unwrap::<T>(x)`.
    fn value_to_base<T: BaseValue>(&self, x: Value) -> T {
        self.base_values().unwrap::<T>(x)
    }

    /// Convert a Rust base type to an egglog [`Value`].
    /// Sugar over `self.base_values().get::<T>(x)`.
    fn base_to_value<T: BaseValue>(&self, x: T) -> Value {
        self.base_values().get::<T>(x)
    }

    /// Look up the Rust container behind an egglog [`Value`], if any.
    /// Sugar over `self.container_values().get_val::<T>(x)`.
    fn value_to_container<T: ContainerValue>(
        &self,
        x: Value,
    ) -> Option<impl Deref<Target = T> + '_> {
        self.container_values().get_val::<T>(x)
    }

    /// Intern a Rust container into the e-graph and return its
    /// [`Value`]. Sugar over `self.register_container(x)`.
    fn container_to_value<T: ContainerValue>(&mut self, x: T) -> Value {
        self.register_container(x)
    }
}

macro_rules! define_state_wrapper {
    ($name:ident) => {
        pub struct $name<'a> {
            pub(crate) inner: &'a mut (dyn ExecStateDyn + 'a),
            // Used by the action-side wrappers' inherent name-indexed
            // methods. Read-only wrappers carry it for a uniform `wrap`
            // signature but never read from it.
            #[allow(dead_code)]
            pub(crate) registry: &'a NamedActionRegistry,
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
// `call_external_func` is identical across all four wrappers — it
// reaches through the dyn trait to the raw `ExecutionState` and invokes
// the given function. The seminaive-soundness contract is caller-side
// (the only intended caller is
// `egglog::sort::fn::FunctionContainer::apply_in`).
macro_rules! impl_call_external_func {
    () => {
        fn call_external_func(
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
        fn table_lookup(
            &mut self,
            action: &TableAction,
            key: &[Value],
        ) -> Option<Value> {
            with_raw_result(self.inner, |es| action.lookup(es, key))
        }
    };
}

impl<'a> UserState<'a> for RuleQueryState<'a> {
    fn wrap(state: &'a mut ExecutionState<'_>, registry: &'a NamedActionRegistry) -> Self {
        RuleQueryState { inner: state, registry }
    }
    fn valid_contexts() -> &'static [Context] {
        &Context::ALL
    }
    fn container_values(&self) -> &ContainerValues {
        self.inner.container_values()
    }
    impl_call_external_func!();
    impl_register_container!();
    impl_table_lookup!();
}

impl<'a> UserState<'a> for RuleActionState<'a> {
    fn wrap(state: &'a mut ExecutionState<'_>, registry: &'a NamedActionRegistry) -> Self {
        RuleActionState { inner: state, registry }
    }
    fn valid_contexts() -> &'static [Context] {
        &[Context::RuleAction, Context::GlobalAction]
    }
    fn container_values(&self) -> &ContainerValues {
        self.inner.container_values()
    }
    impl_call_external_func!();
    impl_register_container!();
    impl_table_lookup!();
}

impl<'a> UserState<'a> for GlobalQueryState<'a> {
    fn wrap(state: &'a mut ExecutionState<'_>, registry: &'a NamedActionRegistry) -> Self {
        GlobalQueryState { inner: state, registry }
    }
    fn valid_contexts() -> &'static [Context] {
        &[Context::GlobalQuery, Context::GlobalAction]
    }
    fn container_values(&self) -> &ContainerValues {
        self.inner.container_values()
    }
    impl_call_external_func!();
    impl_register_container!();
    impl_table_lookup!();
}

impl<'a> UserState<'a> for GlobalActionState<'a> {
    fn wrap(state: &'a mut ExecutionState<'_>, registry: &'a NamedActionRegistry) -> Self {
        GlobalActionState { inner: state, registry }
    }
    fn valid_contexts() -> &'static [Context] {
        &[Context::GlobalAction]
    }
    fn container_values(&self) -> &ContainerValues {
        self.inner.container_values()
    }
    impl_call_external_func!();
    impl_register_container!();
    impl_table_lookup!();
}

/// Inherent name-indexed convenience methods shared by both write-capable
/// state wrappers. Each method looks the underlying [`TableAction`] /
/// [`UnionAction`] / panic id up in the [`NamedActionRegistry`]; the
/// registry is populated and kept in sync by the bridge as tables are
/// added.
macro_rules! impl_named_action_methods {
    () => {
        fn lookup_table_action(&self, name: &str) -> &TableAction {
            self.registry.lookup_table(name).unwrap_or_else(|| {
                panic!("missing table action for table: {name}")
            })
        }

        /// Insert a row into the named table.
        pub fn insert(&mut self, name: &str, row: impl Iterator<Item = Value>) {
            let action = self.lookup_table_action(name).clone();
            let row: SmallVec<[Value; 8]> = row.collect();
            with_raw_result(self.inner, |es| action.insert(es, row.into_iter()));
        }

        /// Look up the return-value column of a row in the named
        /// table — pure read, never inserts. Returns `None` if the
        /// key is not present.
        pub fn lookup(&mut self, name: &str, key: &[Value]) -> Option<Value> {
            let action = self.lookup_table_action(name).clone();
            with_raw_result(self.inner, |es| action.lookup(es, key))
        }

        /// Remove a row from the named table.
        pub fn remove(&mut self, name: &str, key: &[Value]) {
            let action = self.lookup_table_action(name).clone();
            with_raw_result(self.inner, |es| action.remove(es, key));
        }

        /// Subsume a row in the named table.
        pub fn subsume(&mut self, name: &str, key: &[Value]) {
            let action = self.lookup_table_action(name).clone();
            with_raw_result(self.inner, |es| action.subsume(es, key.iter().copied()));
        }

        /// Union two values in the e-graph's union-find.
        pub fn union(&mut self, x: Value, y: Value) {
            let action = *self.registry.union_action();
            with_raw_result(self.inner, |es| action.union(es, x, y));
        }

        /// Trigger a panic from a primitive. Always returns `None` so
        /// the caller can propagate with `?`.
        pub fn panic(&mut self) -> Option<()> {
            let panic_id = self.registry.default_panic_id();
            with_raw_result(self.inner, |es| es.call_external_func(panic_id, &[]));
            None
        }
    };
}

impl<'a> RuleActionState<'a> {
    impl_named_action_methods!();
}

impl<'a> GlobalActionState<'a> {
    impl_named_action_methods!();
}
