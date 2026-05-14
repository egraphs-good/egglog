//! `DuckdbExternalFuncRegistry` — Phase 2 Commit 12.
//!
//! Storage scaffolding for user-registered primitives (`ExternalFunction`)
//! and deferred-panic sentinels on the DuckDB backend.
//!
//! ## What is implemented
//!
//! - **Storage**: a `Vec<Slot>` indexed by `ExternalFunctionId`. Each
//!   slot is one of:
//!   - `Func { func: Box<dyn ExternalFunction> }` — a user-registered
//!     primitive. Stored opaquely; not yet wired to a DuckDB VScalar
//!     UDF (see "Open issues" below).
//!   - `Panic { message: String }` — a deferred-panic sentinel
//!     registered via `Backend::new_panic`. The rule path (owned by the
//!     Commit 10 agent) translates references to a `Panic` slot into
//!     `Action::Panic(message)` in the duckdb IR.
//!   - `Freed` — a slot whose entry has been dropped via
//!     `Backend::free_external_func`. The slot index is not reused so
//!     ids stay stable.
//!
//! - **`new_panic`** returns a fresh id pointing at a `Panic` slot.
//!   The compiled rule pipeline (Commit 10) inspects the slot to detect
//!   the panic case.
//!
//! - **`register_external_func`** returns a fresh id pointing at a
//!   `Func` slot. `free_external_func` marks it `Freed`.
//!
//! ## What is NOT implemented (deferred)
//!
//! Wiring a registered `ExternalFunction` to a DuckDB VScalar UDF so
//! that the function is callable from rule SQL is **deferred**. The
//! scaffolding here only stores the function; it does not register a
//! UDF with DuckDB. As a result, primitives registered through the
//! trait are unreachable from compiled rules on DuckDB.
//!
//! ### Rationale for the deferral
//!
//! Wiring a general `ExternalFunction` (which expects an
//! `&mut ExecutionState`) into a DuckDB VScalar UDF (which expects
//! a vectorized `DataChunkHandle` of typed SQL values) is **complex**:
//!
//! 1. The UDF's signature has to be derived from the function's egglog
//!    signature, which the trait does not surface; primitives currently
//!    declare their schema through `prelude.rs`'s
//!    `add_primitive_with_validator` machinery.
//! 2. Each call has to decode raw DuckDB inputs (`i64`, `String`, etc.)
//!    back to `Value` via the base value pool, package them, run the
//!    primitive's `invoke`, and encode the result back to a DuckDB
//!    value.
//! 3. The primitive can't reenter the database synchronously (DuckDB
//!    UDFs run inside the executor and `register_scalar_function` would
//!    deadlock on a reentrant SQL call). This is exactly what
//!    [`Backend::supports_inline_table_lookups`] = `false` is meant to
//!    surface — primitives that try to call back into tables must be
//!    gated out at registration time.
//!
//! Without a real `Box<dyn Backend>` caller exercising the path (the
//! frontend doesn't route DuckDB through the trait until Commit 14),
//! the storage-only stub here is sufficient. Commit 14 will surface any
//! reachable callers; we revisit the VScalar wrapper then.
//!
//! ## Open issues for Commit 14
//!
//! - VScalar wrapper for `ExternalFunction` (this file's primary TODO).
//! - Decode / encode primitives for `Value` ↔ DuckDB SQL value via the
//!   `BaseValuePool`. The mapping is mostly trivial (identity for
//!   inline-encodable types; intern-table lookup for the rest), but
//!   needs to thread through the UDF entry point.

use egglog_backend_trait::{ExternalFunction, ExternalFunctionId};
use egglog_numeric_id::NumericId;

/// Per-id storage slot. The `Vec<Slot>` index is the
/// `ExternalFunctionId::rep()` value.
#[allow(dead_code)]
pub(crate) enum ExternalFuncSlot {
    /// A user-registered primitive. Held opaquely; not yet wired to a
    /// VScalar UDF. The function is read at this slot by the future
    /// Commit 14 VScalar wrapper.
    Func(Box<dyn ExternalFunction + 'static>),
    /// A deferred-panic sentinel created by `Backend::new_panic`. The
    /// rule path translates references into `Action::Panic(message)`.
    Panic(String),
    /// A slot whose contents were dropped by
    /// `Backend::free_external_func`. The slot is retained (instead of
    /// reusing the index) so existing ids remain stable.
    Freed,
}

/// Storage for primitives + panic sentinels registered through the
/// `Backend` trait.
#[derive(Default)]
pub(crate) struct DuckdbExternalFuncRegistry {
    slots: Vec<ExternalFuncSlot>,
}

impl DuckdbExternalFuncRegistry {
    /// Add a user-registered primitive, returning its id.
    pub(crate) fn add_func(
        &mut self,
        func: Box<dyn ExternalFunction + 'static>,
    ) -> ExternalFunctionId {
        let idx = self.slots.len();
        self.slots.push(ExternalFuncSlot::Func(func));
        ExternalFunctionId::from_usize(idx)
    }

    /// Add a deferred-panic sentinel, returning its id.
    pub(crate) fn add_panic(&mut self, message: String) -> ExternalFunctionId {
        let idx = self.slots.len();
        self.slots.push(ExternalFuncSlot::Panic(message));
        ExternalFunctionId::from_usize(idx)
    }

    /// Drop the entry at `id`. Future accesses see `Freed`.
    ///
    /// Out-of-range or already-freed ids are silently tolerated (mirrors
    /// the bridge's `free_external_func` semantics).
    pub(crate) fn free(&mut self, id: ExternalFunctionId) {
        let idx = id.rep() as usize;
        if let Some(slot) = self.slots.get_mut(idx) {
            *slot = ExternalFuncSlot::Freed;
        }
    }

    /// Borrow the slot at `id`. Returns `None` for out-of-range ids;
    /// returns `Some(Freed)` for freed entries.
    #[allow(dead_code)]
    pub(crate) fn get(&self, id: ExternalFunctionId) -> Option<&ExternalFuncSlot> {
        self.slots.get(id.rep() as usize)
    }

    /// Look up the panic message stored at `id`. Returns `None` if `id`
    /// does not refer to a `Panic` slot (e.g. was freed, or holds a
    /// regular `Func`).
    ///
    /// Used by the Commit 10 rule-builder to translate a
    /// `call_external_func(panic_id, …)` reference into
    /// `Action::Panic(message)` in the duckdb IR.
    #[allow(dead_code)]
    pub(crate) fn panic_message(&self, id: ExternalFunctionId) -> Option<&str> {
        match self.get(id)? {
            ExternalFuncSlot::Panic(msg) => Some(msg.as_str()),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use egglog_backend_trait::{ExecutionState, Value};

    #[derive(Clone)]
    struct NoOpFunc;

    impl ExternalFunction for NoOpFunc {
        fn invoke(&self, _state: &mut ExecutionState, _args: &[Value]) -> Option<Value> {
            None
        }
    }

    /// Sanity-check the registry's id assignment, panic recall, and
    /// free behavior.
    #[test]
    fn registry_assigns_distinct_ids_and_recalls() {
        let mut reg = DuckdbExternalFuncRegistry::default();
        let f_id = reg.add_func(Box::new(NoOpFunc));
        let p_id = reg.add_panic("kaboom".into());
        assert_ne!(f_id, p_id);

        // Slots are populated.
        assert!(matches!(reg.get(f_id), Some(ExternalFuncSlot::Func(_))));
        assert!(matches!(reg.get(p_id), Some(ExternalFuncSlot::Panic(_))));

        assert_eq!(reg.panic_message(p_id), Some("kaboom"));
        assert_eq!(reg.panic_message(f_id), None);

        // Free the function slot; the panic slot is unaffected.
        reg.free(f_id);
        assert!(matches!(reg.get(f_id), Some(ExternalFuncSlot::Freed)));
        assert_eq!(reg.panic_message(p_id), Some("kaboom"));
    }

    /// Out-of-range and already-freed ids do not panic.
    #[test]
    fn registry_handles_invalid_ids() {
        let mut reg = DuckdbExternalFuncRegistry::default();
        // Free of an empty registry.
        reg.free(ExternalFunctionId::from_usize(0));
        assert!(reg.get(ExternalFunctionId::from_usize(0)).is_none());

        let id = reg.add_panic("oops".into());
        reg.free(id);
        reg.free(id);
        assert!(matches!(reg.get(id), Some(ExternalFuncSlot::Freed)));
    }
}
