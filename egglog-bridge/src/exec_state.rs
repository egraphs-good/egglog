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
//! Each wrapper is a thin newtype around one of two base views:
//!
//! - [`PureView`] — `&mut ExecutionState`. Carries the pure-side methods
//!   (base values, counters, container interning, the
//!   `call_external_func` / `table_lookup` trust-boundary escapes, and
//!   base/container conversion sugar).
//! - [`ActionView`] — `PureView` plus a borrow of [`ActionRegistry`].
//!   Carries the additional action-side methods (DB writes,
//!   name-indexed `insert`/`remove`/`subsume`/`lookup`/`union`/`panic`,
//!   and the `raw_exec_state` escape hatch).
//!
//! Each wrapper `Deref`s to its base view, so methods are defined
//! exactly once on the base and auto-resolve at the call site:
//!
//! ```text
//! PureState   ──Deref──▶ PureView
//! ReadState   ──Deref──▶ PureView
//! WriteState  ──Deref──▶ ActionView ──Deref──▶ PureView
//! FullState   ──Deref──▶ ActionView ──Deref──▶ PureView
//! ```
//!
//! Wrappers that "shouldn't have" an action method simply don't deref
//! to `ActionView`. The egglog crate's per-capability `Primitive`
//! traits (`PurePrim`, `WritePrim`, `ReadPrim`, `FullPrim`) name the
//! wrapper they want directly; the Rust type checker enforces at
//! compile time that a primitive's body only uses methods compatible
//! with its declared state.
//!
//! # Trust boundaries
//!
//! A few methods step outside the typed-state contract — they're plain
//! safe Rust, but a careless caller can violate seminaive evaluation.
//! Per-method docs spell out the constraint.
//!
//! - [`PureView::call_external_func`] — every wrapper. Caller must
//!   ensure the callee's `valid_contexts` covers the caller's state.
//! - [`PureView::table_lookup`] — every wrapper; pure read. Safe
//!   everywhere except [`Context::RuleQuery`], where it's an untracked
//!   seminaive read.
//! - [`ActionView::raw_exec_state`] — only on action wrappers. Hands
//!   out the raw `&mut ExecutionState`.
//!
//! Other raw escapes exist elsewhere:
//! [`crate::EGraph::with_execution_state`] and
//! [`crate::EGraph::register_external_func`].

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use core_relations::{
    BaseValue, BaseValues, ContainerValue, ContainerValues, CounterId, ExecutionState,
    ExternalFunctionId, TableId, Value,
};

use crate::core_relations;
use crate::{TableAction, UnionAction};

/// A live registry of action handles for use by typed primitives.
/// Owned by the bridge `EGraph` and shared with action-side state
/// wrappers at invoke time.
#[derive(Clone)]
pub struct ActionRegistry {
    table_actions: HashMap<String, TableAction>,
    union_action: UnionAction,
    default_panic_id: ExternalFunctionId,
}

impl ActionRegistry {
    pub(crate) fn new(union_action: UnionAction, default_panic_id: ExternalFunctionId) -> Self {
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
    /// [`ActionView::panic`].
    pub fn default_panic_id(&self) -> ExternalFunctionId {
        self.default_panic_id
    }
}

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
    pub trait Sealed {}
}

/// Internal dispatch interface used by `FunctionContainer::apply_in`
/// and similar helpers that need to operate generically over the four
/// state wrappers. Two methods: a static for `valid_contexts` and an
/// accessor for the underlying [`PureView`] (which carries
/// `container_values` / `call_external_func` / `table_lookup`).
///
/// **Users do not normally implement or import this trait directly**
/// — call methods on the concrete wrapper instead.
#[doc(hidden)]
pub trait UserState<'a, 'db>: sealed::Sealed
where
    'db: 'a,
{
    fn valid_contexts() -> &'static [Context];
    fn pure_view(&mut self) -> &mut PureView<'a, 'db>;
}

// =====================================================================
// Pure-side base view: methods every state wrapper exposes.
// =====================================================================

/// Pure-side typed view over an `&mut ExecutionState`. Holds the
/// methods that are safe in any execution context (base values,
/// counters, container interning, `call_external_func` /
/// `table_lookup` escapes, conversion sugar). All four state wrappers
/// `Deref` to this type.
pub struct PureView<'a, 'db> {
    pub(crate) inner: &'a mut ExecutionState<'db>,
}

impl<'a, 'db> PureView<'a, 'db> {
    fn new(inner: &'a mut ExecutionState<'db>) -> Self {
        Self { inner }
    }

    /// Base-value pool (interned primitives like `i64`, `String`, …).
    pub fn base_values(&self) -> &BaseValues {
        self.inner.base_values()
    }
    /// Read a counter's current value.
    pub fn read_counter(&self, ctr: CounterId) -> usize {
        self.inner.read_counter(ctr)
    }
    /// Increment and read a counter atomically.
    pub fn inc_counter(&mut self, ctr: CounterId) -> usize {
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
    /// `valid_contexts` covers the wrapper's `valid_contexts()` before
    /// calling — otherwise the called primitive may execute in a
    /// context where its declared state would be unsound.
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
    /// where the read would be untracked by seminaive.
    pub fn table_lookup(&self, action: &TableAction, key: &[Value]) -> Option<Value> {
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
}

// =====================================================================
// Action-side base view: methods only on write-capable wrappers.
// =====================================================================

/// Action-side typed view. Wraps a [`PureView`] and adds DB writes,
/// name-indexed action methods (`insert` / `remove` / etc.), and the
/// raw `&mut ExecutionState` escape hatch. `WriteState` and
/// `FullState` `Deref` to this type; `ActionView` itself `Deref`s to
/// `PureView` so the pure-side methods are reachable as well.
pub struct ActionView<'a, 'db> {
    pub(crate) inner: PureView<'a, 'db>,
    pub(crate) registry: &'a ActionRegistry,
}

impl<'a, 'db> Deref for ActionView<'a, 'db> {
    type Target = PureView<'a, 'db>;
    fn deref(&self) -> &PureView<'a, 'db> {
        &self.inner
    }
}
impl<'a, 'db> DerefMut for ActionView<'a, 'db> {
    fn deref_mut(&mut self) -> &mut PureView<'a, 'db> {
        &mut self.inner
    }
}

impl<'a, 'db> ActionView<'a, 'db> {
    fn new(es: &'a mut ExecutionState<'db>, registry: &'a ActionRegistry) -> Self {
        Self {
            inner: PureView::new(es),
            registry,
        }
    }

    fn lookup_table_action(&self, name: &str) -> &TableAction {
        self.registry
            .lookup_table(name)
            .unwrap_or_else(|| panic!("missing table action for table: {name}"))
    }

    /// Stage a raw row insert. Most callers should use the
    /// name-indexed [`Self::insert`] instead.
    pub fn stage_insert(&mut self, table: TableId, row: &[Value]) {
        self.inner.inner.stage_insert(table, row)
    }
    /// Stage a raw row removal. Most callers should use the
    /// name-indexed [`Self::remove`] instead.
    pub fn stage_remove(&mut self, table: TableId, key: &[Value]) {
        self.inner.inner.stage_remove(table, key)
    }

    /// Insert a row into the named table.
    pub fn insert(&mut self, name: &str, row: impl Iterator<Item = Value>) {
        let action = self.lookup_table_action(name).clone();
        action.insert(self.inner.inner, row);
    }

    /// Look up the return-value column of a row in the named table —
    /// pure read, never inserts. Returns `None` if the key is not
    /// present.
    pub fn lookup(&mut self, name: &str, key: &[Value]) -> Option<Value> {
        let action = self.lookup_table_action(name).clone();
        action.lookup(self.inner.inner, key)
    }

    /// Remove a row from the named table.
    pub fn remove(&mut self, name: &str, key: &[Value]) {
        let action = self.lookup_table_action(name).clone();
        action.remove(self.inner.inner, key);
    }

    /// Subsume a row in the named table.
    pub fn subsume(&mut self, name: &str, key: &[Value]) {
        let action = self.lookup_table_action(name).clone();
        action.subsume(self.inner.inner, key.iter().copied());
    }

    /// Union two values in the e-graph's union-find.
    pub fn union(&mut self, x: Value, y: Value) {
        let action = *self.registry.union_action();
        action.union(self.inner.inner, x, y);
    }

    /// Trigger a panic from a primitive. Always returns `None` so the
    /// caller can propagate with `?`.
    pub fn panic(&mut self) -> Option<()> {
        let panic_id = self.registry.default_panic_id();
        self.inner.inner.call_external_func(panic_id, &[]);
        None
    }

    /// Raw `&mut ExecutionState` access — escape hatch for callers
    /// that need to invoke operations not exposed on the typed
    /// wrapper. Trust boundary: callers must respect the
    /// seminaive-safety rules of the wrapper's context.
    pub fn raw_exec_state(&mut self) -> &mut ExecutionState<'db> {
        self.inner.inner
    }
}

// =====================================================================
// The four wrapper types — thin newtypes around one of the base views.
// =====================================================================

/// Typed view for primitives running in a pure context. Valid in all
/// four execution contexts. `Deref`s to [`PureView`].
pub struct PureState<'a, 'db>(PureView<'a, 'db>);

impl<'a, 'db> PureState<'a, 'db> {
    pub fn wrap(es: &'a mut ExecutionState<'db>) -> Self {
        Self(PureView::new(es))
    }
    pub const fn valid_contexts() -> &'static [Context] {
        &Context::ALL
    }
}
impl<'a, 'db> Deref for PureState<'a, 'db> {
    type Target = PureView<'a, 'db>;
    fn deref(&self) -> &PureView<'a, 'db> {
        &self.0
    }
}
impl<'a, 'db> DerefMut for PureState<'a, 'db> {
    fn deref_mut(&mut self) -> &mut PureView<'a, 'db> {
        &mut self.0
    }
}

/// Typed view for read-only primitives. Valid in `GlobalQuery` and
/// `GlobalAction` contexts. `Deref`s to [`PureView`].
pub struct ReadState<'a, 'db>(PureView<'a, 'db>);

impl<'a, 'db> ReadState<'a, 'db> {
    pub fn wrap(es: &'a mut ExecutionState<'db>) -> Self {
        Self(PureView::new(es))
    }
    pub const fn valid_contexts() -> &'static [Context] {
        &[Context::GlobalQuery, Context::GlobalAction]
    }
}
impl<'a, 'db> Deref for ReadState<'a, 'db> {
    type Target = PureView<'a, 'db>;
    fn deref(&self) -> &PureView<'a, 'db> {
        &self.0
    }
}
impl<'a, 'db> DerefMut for ReadState<'a, 'db> {
    fn deref_mut(&mut self) -> &mut PureView<'a, 'db> {
        &mut self.0
    }
}

/// Typed view for primitives running on the RHS of a rule. Valid in
/// `RuleAction` and `GlobalAction` contexts. `Deref`s to
/// [`ActionView`] (and via that to [`PureView`]).
pub struct WriteState<'a, 'db>(ActionView<'a, 'db>);

impl<'a, 'db> WriteState<'a, 'db> {
    pub fn wrap(es: &'a mut ExecutionState<'db>, registry: &'a ActionRegistry) -> Self {
        Self(ActionView::new(es, registry))
    }
    pub const fn valid_contexts() -> &'static [Context] {
        &[Context::RuleAction, Context::GlobalAction]
    }
}
impl<'a, 'db> Deref for WriteState<'a, 'db> {
    type Target = ActionView<'a, 'db>;
    fn deref(&self) -> &ActionView<'a, 'db> {
        &self.0
    }
}
impl<'a, 'db> DerefMut for WriteState<'a, 'db> {
    fn deref_mut(&mut self) -> &mut ActionView<'a, 'db> {
        &mut self.0
    }
}

/// Typed view for top-level action sites with both read and write
/// access. Valid in `GlobalAction` only. `Deref`s to [`ActionView`]
/// (and via that to [`PureView`]).
pub struct FullState<'a, 'db>(ActionView<'a, 'db>);

impl<'a, 'db> FullState<'a, 'db> {
    pub fn wrap(es: &'a mut ExecutionState<'db>, registry: &'a ActionRegistry) -> Self {
        Self(ActionView::new(es, registry))
    }
    pub const fn valid_contexts() -> &'static [Context] {
        &[Context::GlobalAction]
    }
}
impl<'a, 'db> Deref for FullState<'a, 'db> {
    type Target = ActionView<'a, 'db>;
    fn deref(&self) -> &ActionView<'a, 'db> {
        &self.0
    }
}
impl<'a, 'db> DerefMut for FullState<'a, 'db> {
    fn deref_mut(&mut self) -> &mut ActionView<'a, 'db> {
        &mut self.0
    }
}

// Sealed marker — only the four state wrappers can implement
// `UserState`.
impl sealed::Sealed for PureState<'_, '_> {}
impl sealed::Sealed for ReadState<'_, '_> {}
impl sealed::Sealed for WriteState<'_, '_> {}
impl sealed::Sealed for FullState<'_, '_> {}

impl<'a, 'db> UserState<'a, 'db> for PureState<'a, 'db> {
    fn valid_contexts() -> &'static [Context] {
        Self::valid_contexts()
    }
    fn pure_view(&mut self) -> &mut PureView<'a, 'db> {
        &mut self.0
    }
}
impl<'a, 'db> UserState<'a, 'db> for ReadState<'a, 'db> {
    fn valid_contexts() -> &'static [Context] {
        Self::valid_contexts()
    }
    fn pure_view(&mut self) -> &mut PureView<'a, 'db> {
        &mut self.0
    }
}
impl<'a, 'db> UserState<'a, 'db> for WriteState<'a, 'db> {
    fn valid_contexts() -> &'static [Context] {
        Self::valid_contexts()
    }
    fn pure_view(&mut self) -> &mut PureView<'a, 'db> {
        &mut self.0.inner
    }
}
impl<'a, 'db> UserState<'a, 'db> for FullState<'a, 'db> {
    fn valid_contexts() -> &'static [Context] {
        Self::valid_contexts()
    }
    fn pure_view(&mut self) -> &mut PureView<'a, 'db> {
        &mut self.0.inner
    }
}
