//! User-facing execution state wrappers.
//!
//! Four wrappers around `core_relations::ExecutionState` expose different
//! subsets of the database API based on the context in which a primitive runs:
//!
//! | Wrapper       | DB reads | DB writes | Used for                                 |
//! |---------------|----------|-----------|------------------------------------------|
//! | `PureState`   | no       | no        | rule LHS (seminaive on)                  |
//! | `WriteState`  | no       | yes       | rule RHS (seminaive on)                  |
//! | `ReadState`   | yes      | no        | top-level query-shaped commands          |
//! | `FullState`   | yes      | yes       | top-level action-shaped commands, `eval` |
//!
//! Methods come from sealed capability traits implemented on each
//! wrapper:
//!
//! - [`Core`] — base values, counters, container interning, conversion
//!   sugar. Implemented for all four wrappers.
//! - [`Write`] — name-indexed writes (`insert`/`remove`/`subsume`/
//!   `union`/`panic`). Implemented for [`WriteState`] and [`FullState`].
//!
//! Privileged seams (`call_external_func`, `table_lookup`, raw
//! `&mut ExecutionState`) used by the `FunctionContainer` higher-order
//! dispatch live on the crate-private [`Internal`] trait. User code
//! cannot reach them.
//!
//! [`PurePrim`]: crate::PurePrim
//! [`WritePrim`]: crate::WritePrim
//! [`ReadPrim`]: crate::ReadPrim
//! [`FullPrim`]: crate::FullPrim

use std::ops::Deref;

use crate::core_relations::{
    BaseValue, BaseValues, ContainerValue, ContainerValues, ExecutionState,
    ExternalFunctionId, TableId, Value,
};
use egglog_bridge::{ActionRegistry, TableAction};

/// The four contexts a primitive may run in. Re-exported from
/// `core_relations` so the egglog API surface stays in one place.
pub use crate::core_relations::Context;

// =====================================================================
// Sealed traits.
//
// These traits are `pub(crate)`: external users cannot bring them
// into scope, so they cannot call the methods defined here even on
// values they have. They give the public capability traits (`Core`,
// `Write`) a way to reach the underlying `ExecutionState` from
// default methods, and carry the privileged seams used by the
// `FunctionContainer` higher-order dispatch.
// =====================================================================

/// Crate-private accessor + privileged-dispatch trait. Required
/// methods are the accessors that every wrapper supplies; default
/// methods are the privileged seams used by the `FunctionContainer`
/// higher-order dispatch.
pub(crate) trait Internal<'a, 'db: 'a>: 'a {
    fn es(&self) -> &ExecutionState<'db>;
    fn es_mut(&mut self) -> &mut ExecutionState<'db>;

    fn call_external_func(
        &mut self,
        id: ExternalFunctionId,
        args: &[Value],
    ) -> Option<Value> {
        self.es_mut().call_external_func(id, args)
    }
    fn raw_exec_state(&mut self) -> &mut ExecutionState<'db> {
        self.es_mut()
    }
    /// The current execution context this primitive is being invoked
    /// from. Used by `FunctionContainer::apply` to choose pure-side
    /// vs action-side dispatch.
    fn current_context(&self) -> Context {
        self.es().current_context()
    }
}

/// Sealed accessor for the [`ActionRegistry`]. Implemented by every
/// wrapper that has a registry (`ReadState`, `WriteState`, `FullState`)
/// — the read- and write-side traits both look up `TableAction`s by
/// name through it.
pub(crate) trait RegistrySealed<'a, 'db: 'a>: Internal<'a, 'db> {
    fn registry(&self) -> &ActionRegistry;
}

// =====================================================================
// Public capability traits.
// =====================================================================

/// Core methods available on every state wrapper: base values,
/// counters, container interning, value/base/container conversion sugar.
/// Always seminaive-safe.
#[allow(private_bounds)]
pub trait Core<'a, 'db: 'a>: Internal<'a, 'db> {
    /// Base-value pool (interned primitives like `i64`, `String`, …).
    fn base_values(&self) -> &'a BaseValues {
        self.es().base_values()
    }
    /// Signal that rule execution should stop after this firing.
    fn trigger_early_stop(&self) {
        self.es().trigger_early_stop()
    }
    /// Has someone called `trigger_early_stop`?
    fn should_stop(&self) -> bool {
        self.es().should_stop()
    }
    /// Human-readable name for a table id, if registered.
    fn table_name(&self, table: TableId) -> Option<&'a str> {
        self.es().table_name(table)
    }

    /// Container values for this EGraph.
    fn container_values(&self) -> &'a ContainerValues {
        self.es().container_values()
    }

    /// Register a container value, returning its interned `Value`.
    fn register_container<C: ContainerValue>(&mut self, container: C) -> Value {
        // `container_values()` returns `&'a ContainerValues` — a reference
        // tied to the inner ExecutionState's lifetime, not to `&self` —
        // so it doesn't conflict with the subsequent `&mut` reborrow.
        // Avoiding the clone of `ExecutionState` here matters: this is
        // hot-path code (every container intern goes through it) and
        // the clone copies a non-trivial amount of state.
        let cv = self.container_values();
        let es = self.es_mut();
        cv.register_val(container, es)
    }

    /// Convert an egglog [`Value`] to a Rust base type.
    fn value_to_base<T: BaseValue>(&self, x: Value) -> T {
        self.es().base_values().unwrap::<T>(x)
    }

    /// Convert a Rust base type to an egglog [`Value`].
    fn base_to_value<T: BaseValue>(&self, x: T) -> Value {
        self.es().base_values().get::<T>(x)
    }

    /// Look up the Rust container behind an egglog [`Value`], if any.
    fn value_to_container<T: ContainerValue>(
        &self,
        x: Value,
    ) -> Option<impl Deref<Target = T> + 'a> {
        self.es().container_values().get_val::<T>(x)
    }

    /// Intern a Rust container into the e-graph and return its
    /// [`Value`]. Sugar over `self.register_container(x)`.
    fn container_to_value<T: ContainerValue>(&mut self, x: T) -> Value {
        self.register_container(x)
    }
}

/// Read-side methods — name-indexed table lookup. Implemented for
/// [`ReadState`] and [`FullState`]; *not* for [`PureState`] or
/// [`WriteState`] (a `RuleAction` body must not depend on live DB
/// state). Returns `None` if the row is absent — never inserts.
#[allow(private_bounds)]
pub trait Read<'a, 'db: 'a>: Core<'a, 'db> + RegistrySealed<'a, 'db> {
    /// Look up the return-value column of a row in the named table.
    /// Returns `None` if the key is not present.
    fn lookup(&self, name: &str, key: &[Value]) -> Option<Value> {
        let action = lookup_action(self.registry(), name);
        action.lookup(self.es(), key)
    }
}

/// Action-side write methods — name-indexed inserts/removes/subsumes
/// plus union and panic. Implemented for [`WriteState`] and
/// [`FullState`]; *not* for [`PureState`] or [`ReadState`].
#[allow(private_bounds)]
pub trait Write<'a, 'db: 'a>: Core<'a, 'db> + RegistrySealed<'a, 'db> {
    /// Insert a row into the named table.
    fn insert(&mut self, name: &str, row: impl Iterator<Item = Value>) {
        let action = lookup_action(self.registry(), name).clone();
        action.insert(self.es_mut(), row);
    }

    /// Remove a row from the named table.
    fn remove(&mut self, name: &str, key: &[Value]) {
        let action = lookup_action(self.registry(), name).clone();
        action.remove(self.es_mut(), key);
    }

    /// Subsume a row in the named table.
    fn subsume(&mut self, name: &str, key: &[Value]) {
        let action = lookup_action(self.registry(), name).clone();
        action.subsume(self.es_mut(), key.iter().copied());
    }

    /// Union two values in the e-graph's union-find.
    fn union(&mut self, x: Value, y: Value) {
        let action = *self.registry().union_action();
        action.union(self.es_mut(), x, y);
    }

    /// Trigger a panic from a primitive. Always returns `None` so the
    /// caller can propagate with `?`.
    fn panic(&mut self) -> Option<()> {
        let panic_id = self.registry().default_panic_id();
        self.es_mut().call_external_func(panic_id, &[]);
        None
    }
}

fn lookup_action<'r>(registry: &'r ActionRegistry, name: &str) -> &'r TableAction {
    registry
        .lookup_table(name)
        .unwrap_or_else(|| panic!("missing table action for table: {name}"))
}

// =====================================================================
// The four wrapper types — plain structs, methods come from traits.
// =====================================================================

/// Typed view for primitives running in a pure context. Valid in all
/// four execution contexts. Implements [`Core`] only.
///
/// `PureState` exposes no DB read or write methods to user code: the
/// pure-side privileged seams (`call_external_func`, `table_lookup`)
/// are on the crate-private `Internal` trait so external callers
/// cannot reach them.
///
/// ```compile_fail
/// // Pure context cannot insert: `Write` is not implemented.
/// use egglog::Write;
/// fn _no_writes<'a, 'db>(state: &mut egglog::PureState<'a, 'db>) {
///     state.insert("foo", std::iter::empty());
/// }
/// ```
///
/// ```compile_fail
/// // Pure context cannot reach raw `ExecutionState`.
/// fn _no_raw<'a, 'db>(state: &mut egglog::PureState<'a, 'db>) {
///     state.raw_exec_state();
/// }
/// ```
#[repr(transparent)]
pub struct PureState<'a, 'db> {
    pub(crate) inner: &'a mut ExecutionState<'db>,
}

/// Typed view for read-only primitives. Valid in `GlobalQuery` and
/// `GlobalAction` contexts. Implements [`Core`] + [`Read`] —
/// name-indexed table lookups (`state.lookup("name", &[args])`)
/// return the row's value or `None`.
pub struct ReadState<'a, 'db> {
    pub(crate) inner: &'a mut ExecutionState<'db>,
    pub(crate) registry: &'a ActionRegistry,
}

/// Typed view for primitives running on the RHS of a rule. Valid in
/// `RuleAction` and `GlobalAction` contexts. Implements [`Core`] +
/// [`Write`].
///
/// `WriteState` exposes writes (`insert`/`remove`/`subsume`/`union`/
/// `panic`) but no DB reads — a rule action that depends on live
/// database state would break saturation detection under seminaive.
///
/// ```compile_fail
/// // Write context cannot reach raw `ExecutionState`.
/// fn _no_raw<'a, 'db>(state: &mut egglog::WriteState<'a, 'db>) {
///     state.raw_exec_state();
/// }
/// ```
pub struct WriteState<'a, 'db> {
    pub(crate) inner: &'a mut ExecutionState<'db>,
    pub(crate) registry: &'a ActionRegistry,
}

/// Typed view for top-level action sites with both read and write
/// access. Valid in `GlobalAction` only. Implements [`Core`] +
/// [`Write`].
///
/// ```compile_fail
/// // Even `FullState` cannot reach the raw `ExecutionState` —
/// // privileged seams live on the crate-private `Internal` trait.
/// fn _no_raw<'a, 'db>(state: &mut egglog::FullState<'a, 'db>) {
///     state.raw_exec_state();
/// }
/// ```
pub struct FullState<'a, 'db> {
    pub(crate) inner: &'a mut ExecutionState<'db>,
    pub(crate) registry: &'a ActionRegistry,
}

impl<'a, 'db: 'a> PureState<'a, 'db> {
    pub(crate) fn wrap(es: &'a mut ExecutionState<'db>) -> Self {
        Self { inner: es }
    }
    pub const fn valid_contexts() -> &'static [Context] {
        &Context::ALL
    }
}

impl<'a, 'db: 'a> ReadState<'a, 'db> {
    pub(crate) fn wrap(es: &'a mut ExecutionState<'db>, registry: &'a ActionRegistry) -> Self {
        Self {
            inner: es,
            registry,
        }
    }
    pub const fn valid_contexts() -> &'static [Context] {
        &[Context::GlobalQuery, Context::GlobalAction]
    }
}

impl<'a, 'db: 'a> WriteState<'a, 'db> {
    pub(crate) fn wrap(es: &'a mut ExecutionState<'db>, registry: &'a ActionRegistry) -> Self {
        Self {
            inner: es,
            registry,
        }
    }
    pub const fn valid_contexts() -> &'static [Context] {
        &[Context::RuleAction, Context::GlobalAction]
    }
}

impl<'a, 'db: 'a> FullState<'a, 'db> {
    pub(crate) fn wrap(es: &'a mut ExecutionState<'db>, registry: &'a ActionRegistry) -> Self {
        Self {
            inner: es,
            registry,
        }
    }
    pub const fn valid_contexts() -> &'static [Context] {
        &[Context::GlobalAction]
    }
}

// =====================================================================
// Trait impls. The wrappers implement the sealed accessor traits;
// the public capability traits' default methods do all the rest.
// =====================================================================

impl<'a, 'db: 'a> Internal<'a, 'db> for PureState<'a, 'db> {
    fn es(&self) -> &ExecutionState<'db> {
        self.inner
    }
    fn es_mut(&mut self) -> &mut ExecutionState<'db> {
        self.inner
    }
}
impl<'a, 'db: 'a> Core<'a, 'db> for PureState<'a, 'db> {}

impl<'a, 'db: 'a> Internal<'a, 'db> for ReadState<'a, 'db> {
    fn es(&self) -> &ExecutionState<'db> {
        self.inner
    }
    fn es_mut(&mut self) -> &mut ExecutionState<'db> {
        self.inner
    }
}
impl<'a, 'db: 'a> RegistrySealed<'a, 'db> for ReadState<'a, 'db> {
    fn registry(&self) -> &ActionRegistry {
        self.registry
    }
}
impl<'a, 'db: 'a> Core<'a, 'db> for ReadState<'a, 'db> {}
impl<'a, 'db: 'a> Read<'a, 'db> for ReadState<'a, 'db> {}

impl<'a, 'db: 'a> Internal<'a, 'db> for WriteState<'a, 'db> {
    fn es(&self) -> &ExecutionState<'db> {
        self.inner
    }
    fn es_mut(&mut self) -> &mut ExecutionState<'db> {
        self.inner
    }
}
impl<'a, 'db: 'a> RegistrySealed<'a, 'db> for WriteState<'a, 'db> {
    fn registry(&self) -> &ActionRegistry {
        self.registry
    }
}
impl<'a, 'db: 'a> Core<'a, 'db> for WriteState<'a, 'db> {}
impl<'a, 'db: 'a> Write<'a, 'db> for WriteState<'a, 'db> {}

impl<'a, 'db: 'a> Internal<'a, 'db> for FullState<'a, 'db> {
    fn es(&self) -> &ExecutionState<'db> {
        self.inner
    }
    fn es_mut(&mut self) -> &mut ExecutionState<'db> {
        self.inner
    }
}
impl<'a, 'db: 'a> RegistrySealed<'a, 'db> for FullState<'a, 'db> {
    fn registry(&self) -> &ActionRegistry {
        self.registry
    }
}
impl<'a, 'db: 'a> Core<'a, 'db> for FullState<'a, 'db> {}
impl<'a, 'db: 'a> Read<'a, 'db> for FullState<'a, 'db> {}
impl<'a, 'db: 'a> Write<'a, 'db> for FullState<'a, 'db> {}
