//! User-facing execution state wrappers.
//!
//! Four wrappers around `core_relations::ExecutionState` expose different
//! subsets of the database API based on the context in which a primitive runs:
//!
//! | Wrapper              | DB reads | DB writes | Used for                                  |
//! |----------------------|----------|-----------|-------------------------------------------|
//! | `PureState`     | no       | no        | rule LHS (seminaive on)                   |
//! | `WriteState`    | no       | yes       | rule RHS (seminaive on)                   |
//! | `ReadState`   | yes      | no        | top-level query-shaped commands           |
//! | `FullState`  | yes      | yes       | top-level action-shaped commands, `eval`  |
//!
//! Each wrapper is a typed two-lifetime view over the underlying
//! [`ExecutionState`]: `Wrapper<'a, 'db>` holds `&'a mut ExecutionState<'db>`
//! plus a borrow of the [`ActionRegistry`]. Because the inner reference
//! is typed (no `dyn`), every method call dispatches statically.
//!
//! Each wrapper exposes its capability set as inherent methods — pure
//! operations on every wrapper, DB writes only on the action wrappers,
//! name-indexed `insert`/`remove`/`subsume`/`lookup`/`union`/`panic`
//! only on the action wrappers. The egglog crate's per-capability
//! `Primitive` traits (`PurePrim`, `WritePrim`, ...) name the wrapper
//! they want directly, so the Rust type checker enforces at compile
//! time that a primitive's body only uses methods compatible with its
//! declared state.
//!
//! # Name-indexed convenience methods
//!
//! `WriteState` and `FullState` carry a borrow of the
//! bridge's [`ActionRegistry`] and expose inherent
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
//! - [`call_external_func`](PureState::call_external_func) (every
//!   wrapper) — invoke an external function from any state. Caller must
//!   ensure the callee's `valid_contexts` covers the caller's state.
//!   Used by `FunctionContainer::apply_in` in the egglog crate, which
//!   performs the check.
//! - [`table_lookup`](PureState::table_lookup) (every wrapper) —
//!   pure-read table lookup; safe everywhere except [`Context::RuleQuery`],
//!   where it's an untracked seminaive read.
//! - [`exec_state_mut`](WriteState::exec_state_mut) (only on
//!   write-capable wrappers) — hands out the raw `&mut ExecutionState`.
//!   Available on `WriteState` and `FullState` only.
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
#[derive(Clone)]
pub struct ActionRegistry {
    table_actions: HashMap<String, TableAction>,
    union_action: UnionAction,
    default_panic_id: ExternalFunctionId,
}

impl ActionRegistry {
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
    /// [`WriteState::panic`] and [`FullState::panic`].
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

/// Internal dispatch interface used by `FunctionContainer::apply_in`
/// and similar helpers that need to operate generically over the four
/// state wrappers. Each wrapper auto-impls it via macro by delegating
/// to inherent methods of the same name. **Users do not normally
/// implement or import this trait directly** — call the inherent
/// methods on the concrete wrapper instead.
#[doc(hidden)]
pub trait UserState<'a, 'db> {
    /// The contexts this wrapper is valid in.
    fn valid_contexts() -> &'static [Context];

    /// Container values for this EGraph.
    fn container_values(&self) -> &ContainerValues;

    /// Intern a container into the e-graph.
    fn register_container<C: ContainerValue>(&mut self, container: C) -> Value;

    /// Invoke an external function. Trust boundary: the caller must
    /// verify that the callee's `valid_contexts` covers this state's.
    fn call_external_func(
        &mut self,
        id: ExternalFunctionId,
        args: &[Value],
    ) -> Option<Value>;

    /// Pure table lookup — never inserts. Trust boundary: safe
    /// everywhere except [`Context::RuleQuery`].
    fn table_lookup(&self, action: &TableAction, key: &[Value]) -> Option<Value>;
}

/// Inherent pure-ops methods present on every state wrapper. Each
/// method dispatches statically to the underlying `&mut ExecutionState`.
macro_rules! impl_pure_state_methods {
    () => {
        // ---- pure operations, valid in every context ----

        /// Base-value pool (interned primitives like `i64`, `String`, etc).
        pub fn base_values(&self) -> &BaseValues {
            self.inner.base_values()
        }
        /// Read a counter's current value.
        pub fn read_counter(&self, ctr: CounterId) -> usize {
            self.inner.read_counter(ctr)
        }
        /// Increment and read a counter atomically.
        pub fn inc_counter(&self, ctr: CounterId) -> usize {
            self.inner.inc_counter(ctr)
        }
        /// Signal that rule execution should stop after this firing.
        pub fn trigger_early_stop(&self) {
            self.inner.trigger_early_stop()
        }
        /// Has someone called `trigger_early_stop`?
        pub fn should_stop(&self) -> bool {
            self.inner.should_stop()
        }
        /// Human-readable name for a table id, if registered.
        pub fn table_name(&self, table: TableId) -> Option<&str> {
            self.inner.table_name(table)
        }

        /// Container values for this EGraph.
        pub fn container_values(&self) -> &ContainerValues {
            self.inner.container_values()
        }

        /// Register a container value, returning its interned `Value`.
        ///
        /// Container interning is idempotent: registering an existing
        /// container returns the same `Value`, and a freshly-interned
        /// container is not observable by any rule until it appears in a
        /// table row (a separate write). Safe in every context.
        pub fn register_container<C: ContainerValue>(&mut self, container: C) -> Value {
            let es = &mut *self.inner;
            es.clone().container_values().register_val(container, es)
        }

        /// Invoke an external function from any state.
        ///
        /// Trust boundary: the caller must verify that the function's
        /// `valid_contexts` covers `Self::valid_contexts()` before calling
        /// — otherwise the called primitive may execute in a context where
        /// its declared state would be unsound. See the module-level
        /// "Trust boundaries" section.
        pub fn call_external_func(
            &mut self,
            id: ExternalFunctionId,
            args: &[Value],
        ) -> Option<Value> {
            self.inner.call_external_func(id, args)
        }

        /// Look up a row in a function table — pure read, never inserts.
        ///
        /// Trust boundary: safe everywhere except [`Context::RuleQuery`],
        /// where the read would be untracked by seminaive. See the
        /// module-level "Trust boundaries" section.
        pub fn table_lookup(
            &self,
            action: &TableAction,
            key: &[Value],
        ) -> Option<Value> {
            action.lookup(self.inner, key)
        }

        /// Convert an egglog [`Value`] to a Rust base type.
        /// Sugar over `self.base_values().unwrap::<T>(x)`.
        pub fn value_to_base<T: BaseValue>(&self, x: Value) -> T {
            self.inner.base_values().unwrap::<T>(x)
        }

        /// Convert a Rust base type to an egglog [`Value`].
        /// Sugar over `self.base_values().get::<T>(x)`.
        pub fn base_to_value<T: BaseValue>(&self, x: T) -> Value {
            self.inner.base_values().get::<T>(x)
        }

        /// Look up the Rust container behind an egglog [`Value`], if any.
        /// Sugar over `self.container_values().get_val::<T>(x)`.
        pub fn value_to_container<T: ContainerValue>(
            &self,
            x: Value,
        ) -> Option<impl Deref<Target = T> + '_> {
            self.inner.container_values().get_val::<T>(x)
        }

        /// Intern a Rust container into the e-graph and return its
        /// [`Value`]. Sugar over `self.register_container(x)`.
        pub fn container_to_value<T: ContainerValue>(&mut self, x: T) -> Value {
            self.register_container(x)
        }
    };
}

/// Defines a query-side state wrapper (no registry field — these
/// wrappers don't expose any name-indexed action methods, so there's
/// nothing for the registry to back).
macro_rules! define_pure_state_wrapper {
    ($name:ident) => {
        /// Typed view over an [`ExecutionState`] for primitives running
        /// in a query context. See the module-level docs for the
        /// capability table.
        pub struct $name<'a, 'db> {
            pub(crate) inner: &'a mut ExecutionState<'db>,
        }

        impl<'a, 'db> $name<'a, 'db> {
            /// Wrap an [`ExecutionState`] into the typed state view.
            pub fn wrap(inner: &'a mut ExecutionState<'db>) -> Self {
                $name { inner }
            }

            impl_pure_state_methods!();
        }
    };
}

/// Defines an action-side state wrapper (carries a borrow of
/// [`ActionRegistry`] so the wrapper's name-indexed methods can
/// resolve `TableAction` / `UnionAction` handles by name).
macro_rules! define_action_state_wrapper {
    ($name:ident) => {
        /// Typed view over an [`ExecutionState`] for primitives running
        /// in an action context. See the module-level docs for the
        /// capability table.
        pub struct $name<'a, 'db> {
            pub(crate) inner: &'a mut ExecutionState<'db>,
            pub(crate) registry: &'a ActionRegistry,
        }

        impl<'a, 'db> $name<'a, 'db> {
            /// Wrap an [`ExecutionState`] into the typed state view.
            pub fn wrap(
                inner: &'a mut ExecutionState<'db>,
                registry: &'a ActionRegistry,
            ) -> Self {
                $name { inner, registry }
            }

            impl_pure_state_methods!();
        }
    };
}

define_pure_state_wrapper!(PureState);
define_action_state_wrapper!(WriteState);
define_pure_state_wrapper!(ReadState);
define_action_state_wrapper!(FullState);

// `valid_contexts` constants per state type, used by the registration
// machinery in the egglog crate to populate `PrimitiveWithId.valid_contexts`.
impl<'a, 'db> PureState<'a, 'db> {
    pub const fn valid_contexts() -> &'static [Context] {
        &Context::ALL
    }
}
impl<'a, 'db> WriteState<'a, 'db> {
    pub const fn valid_contexts() -> &'static [Context] {
        &[Context::RuleAction, Context::GlobalAction]
    }
}
impl<'a, 'db> ReadState<'a, 'db> {
    pub const fn valid_contexts() -> &'static [Context] {
        &[Context::GlobalQuery, Context::GlobalAction]
    }
}
impl<'a, 'db> FullState<'a, 'db> {
    pub const fn valid_contexts() -> &'static [Context] {
        &[Context::GlobalAction]
    }
}

macro_rules! impl_user_state {
    ($name:ident) => {
        impl<'a, 'db> UserState<'a, 'db> for $name<'a, 'db> {
            fn valid_contexts() -> &'static [Context] {
                $name::valid_contexts()
            }
            fn container_values(&self) -> &ContainerValues {
                $name::container_values(self)
            }
            fn register_container<C: ContainerValue>(&mut self, container: C) -> Value {
                $name::register_container(self, container)
            }
            fn call_external_func(
                &mut self,
                id: ExternalFunctionId,
                args: &[Value],
            ) -> Option<Value> {
                $name::call_external_func(self, id, args)
            }
            fn table_lookup(&self, action: &TableAction, key: &[Value]) -> Option<Value> {
                $name::table_lookup(self, action, key)
            }
        }
    };
}

impl_user_state!(PureState);
impl_user_state!(WriteState);
impl_user_state!(ReadState);
impl_user_state!(FullState);

// Writes: inherent methods only on action contexts.
macro_rules! impl_write_db_methods {
    () => {
        /// Stage a raw row insert. Most callers should use the
        /// name-indexed [`Self::insert`] instead.
        pub fn stage_insert(&mut self, table: TableId, row: &[Value]) {
            self.inner.stage_insert(table, row)
        }
        /// Stage a raw row removal. Most callers should use the
        /// name-indexed [`Self::remove`] instead.
        pub fn stage_remove(&mut self, table: TableId, key: &[Value]) {
            self.inner.stage_remove(table, key)
        }
    };
}

/// Inherent name-indexed convenience methods + raw escape hatch shared by
/// both write-capable state wrappers. Each method looks the underlying
/// [`TableAction`] / [`UnionAction`] / panic id up in the
/// [`ActionRegistry`]; the registry is populated and kept in sync
/// by the bridge as tables are added.
macro_rules! impl_named_action_methods {
    () => {
        fn lookup_table_action(&self, name: &str) -> &TableAction {
            self.registry.lookup_table(name).unwrap_or_else(|| {
                panic!("missing table action for table: {name}")
            })
        }

        /// Raw `&mut ExecutionState` access — escape hatch for callers
        /// that need to invoke operations not exposed on the typed
        /// wrapper. Trust boundary: callers must respect the
        /// seminaive-safety rules of the wrapper's context.
        pub fn exec_state_mut(&mut self) -> &mut ExecutionState<'db> {
            self.inner
        }

        /// Insert a row into the named table.
        pub fn insert(&mut self, name: &str, row: impl Iterator<Item = Value>) {
            let action = self.lookup_table_action(name).clone();
            action.insert(self.inner, row);
        }

        /// Look up the return-value column of a row in the named
        /// table — pure read, never inserts. Returns `None` if the
        /// key is not present.
        pub fn lookup(&mut self, name: &str, key: &[Value]) -> Option<Value> {
            let action = self.lookup_table_action(name).clone();
            action.lookup(self.inner, key)
        }

        /// Remove a row from the named table.
        pub fn remove(&mut self, name: &str, key: &[Value]) {
            let action = self.lookup_table_action(name).clone();
            action.remove(self.inner, key);
        }

        /// Subsume a row in the named table.
        pub fn subsume(&mut self, name: &str, key: &[Value]) {
            let action = self.lookup_table_action(name).clone();
            action.subsume(self.inner, key.iter().copied());
        }

        /// Union two values in the e-graph's union-find.
        pub fn union(&mut self, x: Value, y: Value) {
            let action = *self.registry.union_action();
            action.union(self.inner, x, y);
        }

        /// Trigger a panic from a primitive. Always returns `None` so
        /// the caller can propagate with `?`.
        pub fn panic(&mut self) -> Option<()> {
            let panic_id = self.registry.default_panic_id();
            self.inner.call_external_func(panic_id, &[]);
            None
        }
    };
}

impl<'a, 'db> WriteState<'a, 'db> {
    impl_named_action_methods!();
    impl_write_db_methods!();
}

impl<'a, 'db> FullState<'a, 'db> {
    impl_named_action_methods!();
    impl_write_db_methods!();
}
