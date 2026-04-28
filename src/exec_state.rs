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
//! dispatch live on the crate-private `__internal::Internal` trait.
//! User code cannot reach them.
//!
//! [`PurePrim`]: crate::PurePrim
//! [`WritePrim`]: crate::WritePrim
//! [`ReadPrim`]: crate::ReadPrim
//! [`FullPrim`]: crate::FullPrim

use std::ops::Deref;

use crate::core_relations::{
    BaseValue, BaseValues, ContainerValue, ContainerValues, CounterId, ExecutionState, TableId,
    Value,
};
use egglog_bridge::{ActionRegistry, TableAction};

/// The four contexts a primitive may run in.
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

mod sealed {
    use crate::core_relations::ExecutionState;
    use egglog_bridge::ActionRegistry;

    /// Crate-private accessor trait. Lives in a private module so
    /// external users cannot bring it into scope to call its methods.
    pub trait CoreSealed<'a, 'db: 'a>: 'a {
        fn es(&self) -> &ExecutionState<'db>;
        fn es_mut(&mut self) -> &mut ExecutionState<'db>;
    }
    pub trait WriteSealed<'a, 'db: 'a>: CoreSealed<'a, 'db> {
        fn registry(&self) -> &ActionRegistry;
    }
}

/// Crate-private surface for the `FunctionContainer` higher-order
/// dispatch and similar helpers — the only call sites that
/// legitimately need raw external-func dispatch, raw table reads, or
/// raw `&mut ExecutionState`. Lives in a `pub(crate)` module, so
/// external users cannot bring it into scope.
pub(crate) mod __internal {
    use super::sealed::CoreSealed;
    use crate::core_relations::{ExecutionState, ExternalFunctionId, Value};
    use egglog_bridge::TableAction;

    pub trait Internal<'a, 'db: 'a>: CoreSealed<'a, 'db> {
        fn call_external_func(
            &mut self,
            id: ExternalFunctionId,
            args: &[Value],
        ) -> Option<Value> {
            self.es_mut().call_external_func(id, args)
        }
        fn table_lookup(&self, action: &TableAction, key: &[Value]) -> Option<Value> {
            action.lookup(self.es(), key)
        }
        fn raw_exec_state(&mut self) -> &mut ExecutionState<'db> {
            self.es_mut()
        }
    }
}

/// Internal dispatch interface used by `FunctionContainer::apply_in`
/// and similar helpers that need to operate generically over the four
/// state wrappers. **Users do not normally implement or import this
/// trait directly** — call methods on the concrete wrapper instead.
#[doc(hidden)]
pub trait UserState<'a, 'db>: Core<'a, 'db> + __internal::Internal<'a, 'db>
where
    'db: 'a,
{
    fn valid_contexts() -> &'static [Context];
}

// =====================================================================
// Public capability traits.
// =====================================================================

/// Core methods available on every state wrapper: base values,
/// counters, container interning, value/base/container conversion sugar.
/// Always seminaive-safe.
pub trait Core<'a, 'db: 'a>: sealed::CoreSealed<'a, 'db> {
    /// Base-value pool (interned primitives like `i64`, `String`, …).
    fn base_values(&self) -> &'a BaseValues {
        self.es().base_values()
    }
    /// Read a counter's current value.
    fn read_counter(&self, ctr: CounterId) -> usize {
        self.es().read_counter(ctr)
    }
    /// Increment and read a counter atomically.
    fn inc_counter(&mut self, ctr: CounterId) -> usize {
        self.es_mut().inc_counter(ctr)
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
        let es = self.es_mut();
        es.clone().container_values().register_val(container, es)
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

/// Action-side write methods — name-indexed inserts/removes/subsumes
/// plus union and panic. Implemented for [`WriteState`] and
/// [`FullState`]; *not* for [`PureState`] or [`ReadState`].
pub trait Write<'a, 'db: 'a>: Core<'a, 'db> + sealed::WriteSealed<'a, 'db> {
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
pub struct PureState<'a, 'db> {
    pub(crate) inner: &'a mut ExecutionState<'db>,
}

/// Typed view for read-only primitives. Valid in `GlobalQuery` and
/// `GlobalAction` contexts. Implements [`Core`] only.
pub struct ReadState<'a, 'db> {
    pub(crate) inner: &'a mut ExecutionState<'db>,
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
    pub(crate) fn wrap(es: &'a mut ExecutionState<'db>) -> Self {
        Self { inner: es }
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

impl<'a, 'db: 'a> sealed::CoreSealed<'a, 'db> for PureState<'a, 'db> {
    fn es(&self) -> &ExecutionState<'db> {
        self.inner
    }
    fn es_mut(&mut self) -> &mut ExecutionState<'db> {
        self.inner
    }
}
impl<'a, 'db: 'a> Core<'a, 'db> for PureState<'a, 'db> {}
impl<'a, 'db: 'a> __internal::Internal<'a, 'db> for PureState<'a, 'db> {}
impl<'a, 'db: 'a> UserState<'a, 'db> for PureState<'a, 'db> {
    fn valid_contexts() -> &'static [Context] {
        Self::valid_contexts()
    }
}

impl<'a, 'db: 'a> sealed::CoreSealed<'a, 'db> for ReadState<'a, 'db> {
    fn es(&self) -> &ExecutionState<'db> {
        self.inner
    }
    fn es_mut(&mut self) -> &mut ExecutionState<'db> {
        self.inner
    }
}
impl<'a, 'db: 'a> Core<'a, 'db> for ReadState<'a, 'db> {}
impl<'a, 'db: 'a> __internal::Internal<'a, 'db> for ReadState<'a, 'db> {}
impl<'a, 'db: 'a> UserState<'a, 'db> for ReadState<'a, 'db> {
    fn valid_contexts() -> &'static [Context] {
        Self::valid_contexts()
    }
}

impl<'a, 'db: 'a> sealed::CoreSealed<'a, 'db> for WriteState<'a, 'db> {
    fn es(&self) -> &ExecutionState<'db> {
        self.inner
    }
    fn es_mut(&mut self) -> &mut ExecutionState<'db> {
        self.inner
    }
}
impl<'a, 'db: 'a> sealed::WriteSealed<'a, 'db> for WriteState<'a, 'db> {
    fn registry(&self) -> &ActionRegistry {
        self.registry
    }
}
impl<'a, 'db: 'a> Core<'a, 'db> for WriteState<'a, 'db> {}
impl<'a, 'db: 'a> Write<'a, 'db> for WriteState<'a, 'db> {}
impl<'a, 'db: 'a> __internal::Internal<'a, 'db> for WriteState<'a, 'db> {}
impl<'a, 'db: 'a> UserState<'a, 'db> for WriteState<'a, 'db> {
    fn valid_contexts() -> &'static [Context] {
        Self::valid_contexts()
    }
}

impl<'a, 'db: 'a> sealed::CoreSealed<'a, 'db> for FullState<'a, 'db> {
    fn es(&self) -> &ExecutionState<'db> {
        self.inner
    }
    fn es_mut(&mut self) -> &mut ExecutionState<'db> {
        self.inner
    }
}
impl<'a, 'db: 'a> sealed::WriteSealed<'a, 'db> for FullState<'a, 'db> {
    fn registry(&self) -> &ActionRegistry {
        self.registry
    }
}
impl<'a, 'db: 'a> Core<'a, 'db> for FullState<'a, 'db> {}
impl<'a, 'db: 'a> Write<'a, 'db> for FullState<'a, 'db> {}
impl<'a, 'db: 'a> __internal::Internal<'a, 'db> for FullState<'a, 'db> {}
impl<'a, 'db: 'a> UserState<'a, 'db> for FullState<'a, 'db> {
    fn valid_contexts() -> &'static [Context] {
        Self::valid_contexts()
    }
}
