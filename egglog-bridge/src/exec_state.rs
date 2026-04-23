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

use core_relations::{
    BaseValues, ContainerValue, ContainerValues, CounterId, ExecutionState, TableId, Value,
    WrappedTable,
};

use crate::core_relations;

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
/// `RuleActionState` and `GlobalActionState`.
pub trait ExecStateWriteDb: ExecStateCore {
    fn stage_insert(&mut self, table: TableId, row: &[Value]);
    fn stage_remove(&mut self, table: TableId, key: &[Value]);
}

/// Common trait for the user-facing state wrappers.
///
/// Lets registration machinery derive a primitive's valid contexts from its
/// declared [`TypedPrimitive::State`] type without caring which of the four
/// concrete wrappers it is.
pub trait UserState<'a>: Sized + ExecStateCore {
    fn wrap(state: &'a mut ExecutionState<'_>) -> Self;
    fn valid_contexts() -> &'static [Context];
    fn container_values(&self) -> &ContainerValues;
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
}
impl<'a> ExecStateWriteDb for GlobalActionState<'a> {
    fn stage_insert(&mut self, table: TableId, row: &[Value]) {
        self.inner.stage_insert(table, row)
    }
    fn stage_remove(&mut self, table: TableId, key: &[Value]) {
        self.inner.stage_remove(table, key)
    }
}

// `register_container` is an inherent method on every wrapper because it
// is generic over `C: ContainerValue`, and generic methods cannot live on
// a dyn-compatible trait. Container registration is idempotent interning
// and safe in every context, so all four wrappers expose it. See #772.
macro_rules! define_container_registrar {
    ($name:ident) => {
        impl<'a> $name<'a> {
            /// Register a container value, returning its interned `Value`.
            ///
            /// Container interning is idempotent: registering an existing
            /// container returns the same `Value`, and a freshly-interned
            /// container is not observable by any rule until it appears in
            /// a table row (a separate write). Safe in every context.
            pub fn register_container<C: ContainerValue>(&mut self, container: C) -> Value {
                with_raw_result(self.inner, |es| {
                    es.clone().container_values().register_val(container, es)
                })
            }
        }
    };
}

define_container_registrar!(RuleQueryState);
define_container_registrar!(RuleActionState);
define_container_registrar!(GlobalQueryState);
define_container_registrar!(GlobalActionState);

// `with_raw_exec_state` is a temporary escape hatch restricted to
// write-capable wrappers. It hands out raw `&mut ExecutionState`, so it
// permits both reads and writes; we keep it off the non-action wrappers
// to preserve seminaive soundness guarantees. Each typed method that
// currently uses it is a candidate to become a first-class capability
// method on its respective trait.
macro_rules! define_raw_exec_state_escape {
    ($name:ident) => {
        impl<'a> $name<'a> {
            /// Raw `&mut ExecutionState` access. Used by
            /// `UnionAction::union`, `TableAction::{insert,remove,subsume,
            /// lookup_or_insert}`, and similar operations that need both
            /// `&ContainerValues` and `&mut ExecutionState` at once.
            pub fn with_raw_exec_state<R>(
                &mut self,
                f: impl FnOnce(&mut ExecutionState<'_>) -> R,
            ) -> R {
                with_raw_result(self.inner, f)
            }
        }
    };
}

define_raw_exec_state_escape!(RuleActionState);
define_raw_exec_state_escape!(GlobalActionState);

// UserState impls. `valid_contexts()` encodes that a narrower (more
// restricted) wrapper is usable in every context that a wider one is:
// `RuleQueryState` works everywhere, `GlobalActionState` only as itself.
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
}
