//! `DuckdbBaseValuePool` — Phase 2 Commit 11.
//!
//! The DuckDB backend re-uses `egglog_core_relations::BaseValues` directly
//! as its primitive registry. The wrapping `DuckdbBaseValuePool` exists
//! only to bridge the trait's `BaseValuePool` API into the inner
//! `BaseValues` (which has the same shape but lives outside the
//! `egglog-backend-trait` crate).
//!
//! ## Why reuse `BaseValues`?
//!
//! `BaseValues` already handles two encoding strategies for every
//! registered base-value type:
//!
//! 1. **Inline-encodable types** (`i64`, `bool`, `f64`, `Unit`) — the
//!    `Value` itself IS the encoded primitive (with `MAY_UNBOX` and a
//!    sign-extension offset for `i64`). The fast path in
//!    `BaseInternTable::intern` returns immediately via `try_box` without
//!    touching any hash table.
//! 2. **Non-unboxable types** (`String`, `BigInt`, `BigRat`, `Rational64`,
//!    and user-defined `BaseValue` impls) — keyed by their `Hash + Eq`
//!    implementation through an in-memory intern table.
//!
//! Both paths are exactly what DuckDB needs. Concrete DuckDB SQL columns
//! store the `Value`'s `u32` as `BIGINT`; the pool provides the mapping
//! back to typed values for inspection / serialization.
//!
//! ## Wiring
//!
//! `EGraph` holds a `backend_base_value_pool: DuckdbBaseValuePool` field.
//! `Backend::base_value_pool` / `base_value_pool_mut` return `&` /
//! `&mut` borrows of this field, cast to `&dyn BaseValuePool` /
//! `&mut dyn BaseValuePool`.

use std::any::{Any, TypeId};

use egglog_backend_trait::{BaseValueId, BaseValuePool, Value};
use egglog_core_relations::{BaseValues, DynamicInternTable};

/// DuckDB-backend base-value pool. Thin wrapper around
/// `egglog_core_relations::BaseValues`. Public so the egglog
/// frontend can reach `inner()` for `Sort::reconstruct_termdag_base`
/// (which takes `&BaseValues`).
#[derive(Default, Clone)]
pub struct DuckdbBaseValuePool {
    values: BaseValues,
}

impl DuckdbBaseValuePool {
    /// Borrow the inner `BaseValues` for direct typed access via the
    /// concrete `register_type<P>` / `get<P>` / `unwrap<P>` methods.
    /// Public so the egglog frontend's extraction path can call
    /// `Sort::reconstruct_termdag_base` (which takes `&BaseValues`)
    /// without first downcasting to `egglog_bridge::EGraph`.
    pub fn inner(&self) -> &BaseValues {
        &self.values
    }

    /// Mutable borrow of the inner `BaseValues`.
    pub fn inner_mut(&mut self) -> &mut BaseValues {
        &mut self.values
    }
}

impl BaseValuePool for DuckdbBaseValuePool {
    fn register_type_dyn(
        &mut self,
        type_id: TypeId,
        factory: Box<dyn FnOnce() -> Box<dyn DynamicInternTable>>,
    ) -> BaseValueId {
        self.values.register_type_dyn(type_id, factory)
    }

    fn get_ty_by_type_id(&self, type_id: TypeId) -> BaseValueId {
        self.values.get_ty_by_id(type_id)
    }

    fn intern_dyn(&self, ty: BaseValueId, value: Box<dyn Any + Send + Sync>) -> Value {
        let any_ref: &dyn Any = &*value;
        self.values.intern_dyn_by_id(ty, any_ref)
    }

    fn unwrap_dyn(&self, ty: BaseValueId, val: Value) -> Box<dyn Any + Send + Sync> {
        self.values.unwrap_dyn_by_id(ty, val)
    }

    fn has_ty(&self, type_id: TypeId) -> bool {
        self.values.has_ty_by_id(type_id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use egglog_backend_trait::{pool_get, pool_get_ty, pool_register_type, pool_unwrap};

    /// Round-trip `i64` through the dyn pool. Covers both the inline
    /// (`MAY_UNBOX = true`, fits in `u32`) path and the intern-table
    /// fallback (top bit set).
    #[test]
    fn duckdb_base_value_pool_i64_round_trip() {
        let mut pool = DuckdbBaseValuePool::default();
        let id = pool_register_type::<i64>(&mut pool);
        // Idempotent registration.
        assert_eq!(id, pool_register_type::<i64>(&mut pool));

        assert!(pool.has_ty(TypeId::of::<i64>()));

        // Inline path.
        let small: i64 = 42;
        let small_val = pool_get::<i64>(&pool, small);
        let small_back: i64 = pool_unwrap::<i64>(&pool, small_val);
        assert_eq!(small_back, small);

        // Intern-table path — top bit set forces fallback.
        let big: i64 = 0x80_00_00_00i64 | 0x12_34_56_78;
        let big_val = pool_get::<i64>(&pool, big);
        let big_back: i64 = pool_unwrap::<i64>(&pool, big_val);
        assert_eq!(big_back, big);

        // Re-interning the same big value returns the same id.
        let big_val_again = pool_get::<i64>(&pool, big);
        assert_eq!(big_val, big_val_again);

        // Trait-level intern_dyn / unwrap_dyn agree with the typed helpers.
        let id_for_i64 = pool_get_ty::<i64>(&pool);
        let boxed: Box<dyn Any + Send + Sync> = Box::new(big);
        let via_intern = pool.intern_dyn(id_for_i64, boxed);
        assert_eq!(via_intern, big_val);
        let unboxed = pool.unwrap_dyn(id_for_i64, via_intern);
        let unboxed_i64 = *unboxed
            .downcast::<i64>()
            .expect("unwrap_dyn returned wrong type");
        assert_eq!(unboxed_i64, big);
    }

    /// `String` (non-`MAY_UNBOX`) round-trips through the dyn registration
    /// / intern / unwrap path.
    #[test]
    fn duckdb_base_value_pool_string_round_trip() {
        let mut pool = DuckdbBaseValuePool::default();
        let _ = pool_register_type::<String>(&mut pool);

        let s = String::from("hello");
        let v = pool_get::<String>(&pool, s.clone());
        let v2 = pool_get::<String>(&pool, s.clone());
        assert_eq!(v, v2);

        let back: String = pool_unwrap::<String>(&pool, v);
        assert_eq!(back, s);
    }

    /// `has_ty` reports the right answer pre- and post-registration.
    #[test]
    fn duckdb_base_value_pool_has_ty() {
        let mut pool = DuckdbBaseValuePool::default();
        assert!(!pool.has_ty(TypeId::of::<String>()));
        let _ = pool_register_type::<String>(&mut pool);
        assert!(pool.has_ty(TypeId::of::<String>()));
    }
}
