//! Tests for the typed `EClass<M>` wrapper.

use egglog::{EClass, EGraph, EqSortMarker, IntoColumn};

struct Math;
impl EqSortMarker for Math {
    const NAME: &'static str = "Math";
}

struct List;
impl EqSortMarker for List {
    const NAME: &'static str = "List";
}

/// Smoke test: define a sort, look up an eclass, tag it.
#[test]
fn typed_eclass_round_trip() {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(
        None,
        "(datatype Math (Num i64))\n\
         (let $n (Num 7))",
    )
    .unwrap();

    let raw = eg
        .lookup_function("Num", &[eg.base_to_value::<i64>(7)])
        .unwrap();
    let typed: EClass<Math> = eg.typed_eclass::<Math>(raw).unwrap();

    assert_eq!(typed.value(), raw);
    assert_eq!(EClass::<Math>::sort_name(), "Math");
}

/// `typed_eclass` returns `None` if the marker's name doesn't match an
/// eq-sort in this EGraph.
#[test]
fn typed_eclass_unknown_sort_is_none() {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype Math (Num i64))")
        .unwrap();

    // Math exists, List does not.
    let some_value = eg.base_to_value::<i64>(0);
    assert!(eg.typed_eclass::<Math>(some_value).is_some());
    assert!(eg.typed_eclass::<List>(some_value).is_none());
}

/// `EClass<M>` is `Copy`, `Clone`, `Eq`, `Hash`, and prints sensibly
/// with `Debug`.
#[test]
fn typed_eclass_traits() {
    let raw = egglog::Value::new_const(42);
    let a: EClass<Math> = EClass::from_value_unchecked(raw);
    let b = a; // Copy
    #[allow(clippy::clone_on_copy)]
    let _c = a.clone(); // Clone
    assert_eq!(a, b);

    // EClass<M>: Hash — equal values hash equal.
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h1 = DefaultHasher::new();
    let mut h2 = DefaultHasher::new();
    a.hash(&mut h1);
    b.hash(&mut h2);
    assert_eq!(h1.finish(), h2.finish());

    let dbg = format!("{:?}", a);
    assert!(
        dbg.contains("Math"),
        "debug should include sort name: {}",
        dbg
    );
    assert!(
        dbg.contains("EClass"),
        "debug should include EClass: {}",
        dbg
    );
}

/// `EClass<M>` participates in `IntoColumn` — passes through as the
/// underlying [`Value`]. (Verified by passing it to `EGraph::insert`
/// and reading the row back.)
#[test]
fn typed_eclass_is_into_column() {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(
        None,
        "(datatype Math (Num i64))\n\
         (function tag (Math) i64 :no-merge)\n\
         (let $n (Num 7))",
    )
    .unwrap();

    let raw = eg
        .lookup_function("Num", &[eg.base_to_value::<i64>(7)])
        .unwrap();
    let typed: EClass<Math> = eg.typed_eclass::<Math>(raw).unwrap();

    // Use the typed handle as a key column on the way in.
    eg.set("tag", (typed,), 99_i64).unwrap();

    // Look up via the typed handle as the key.
    let v = eg.lookup::<_, i64>("tag", typed).unwrap();
    assert_eq!(v, Some(99));
}

/// Compile-time test: passing an `EClass<List>` where `EClass<Math>`
/// is expected should fail to compile. This is a smoke test that the
/// `IntoColumn` impl is generic over `M`, not a specific marker — so
/// the typing actually happens at the call site (function signatures
/// taking specific markers are where rejections fire). We just verify
/// `EClass<M>` is `IntoColumn` for both markers separately.
#[test]
fn typed_eclass_works_for_multiple_markers() {
    fn assert_into_column<T: IntoColumn>(_x: T) {}
    let raw = egglog::Value::new_const(42);
    let m: EClass<Math> = EClass::from_value_unchecked(raw);
    let l: EClass<List> = EClass::from_value_unchecked(raw);
    assert_into_column(m);
    assert_into_column(l);
}
