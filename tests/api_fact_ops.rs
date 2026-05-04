//! Tests for the typed fact-ops API on `EGraph`:
//! `set`, `add_node`, `lookup`, `contains`, `remove`.

use egglog::prelude::*;
use egglog::{EClass, EqSortMarker, IntoRow, RawValues};

fn make_eg_with_function() -> EGraph {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(function f (i64) i64 :no-merge)")
        .unwrap();
    eg
}

#[test]
fn test_set_then_lookup_function() {
    let mut eg = make_eg_with_function();
    eg.set("f", (1_i64,), 42_i64).unwrap();
    assert_eq!(eg.lookup::<_, i64>("f", 1_i64).unwrap(), Some(42));
}

#[test]
fn test_lookup_missing_returns_none() {
    let eg = make_eg_with_function();
    // Nothing set yet — lookup should be None.
    assert_eq!(eg.lookup::<_, i64>("f", 999_i64).unwrap(), None);
}

#[test]
fn test_contains_function() {
    let mut eg = make_eg_with_function();
    eg.set("f", (1_i64,), 42_i64).unwrap();
    assert!(eg.contains("f", 1_i64));
    assert!(!eg.contains("f", 999_i64));
}

#[test]
fn test_remove_function() {
    let mut eg = make_eg_with_function();
    eg.set("f", (1_i64,), 42_i64).unwrap();
    assert!(eg.contains("f", 1_i64));
    eg.remove("f", 1_i64).unwrap();
    assert!(!eg.contains("f", 1_i64));
    // Removing again is a no-op.
    eg.remove("f", 1_i64).unwrap();
    assert!(!eg.contains("f", 1_i64));
}

#[test]
fn test_relation_add_node_and_contains() {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(relation R (i64 i64))")
        .unwrap();
    eg.add_node("R", (1_i64, 2_i64)).unwrap();
    assert!(eg.contains("R", (1_i64, 2_i64)));
    assert!(!eg.contains("R", (1_i64, 3_i64)));
    assert!(!eg.contains("R", (2_i64, 1_i64)));
}

#[test]
fn test_relation_remove() {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(relation R (i64 i64))")
        .unwrap();
    eg.add_node("R", (1_i64, 2_i64)).unwrap();
    eg.add_node("R", (3_i64, 4_i64)).unwrap();
    assert!(eg.contains("R", (1_i64, 2_i64)));
    assert!(eg.contains("R", (3_i64, 4_i64)));
    eg.remove("R", (1_i64, 2_i64)).unwrap();
    assert!(!eg.contains("R", (1_i64, 2_i64)));
    assert!(eg.contains("R", (3_i64, 4_i64)));
}

#[test]
fn test_constructor_add_node_returns_eclass() {
    // Constructors mint a fresh eclass id when add_node is first called;
    // calling again with the same inputs returns the same eclass.
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype List (Cons i64 List) (Nil))")
        .unwrap();

    // Zero-arg constructor: pass `RawValues(vec![])` because `()` would
    // be interpreted as a single Unit column.
    let nil = eg.add_node("Nil", RawValues(vec![])).unwrap();
    assert!(eg.contains("Nil", RawValues(vec![])));

    let cons = eg.add_node("Cons", (1_i64, nil)).unwrap();
    assert!(eg.contains("Cons", (1_i64, nil)));

    // Calling add_node again with the same inputs returns the same eclass.
    let cons2 = eg.add_node("Cons", (1_i64, nil)).unwrap();
    assert_eq!(cons, cons2);
}

#[test]
fn test_eclass_of_constructor_returns_typed_eclass() {
    struct List;
    impl EqSortMarker for List {
        const NAME: &'static str = "List";
    }

    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype List (Cons i64 List) (Nil))")
        .unwrap();
    let nil = eg.add_node("Nil", RawValues(vec![])).unwrap();
    let cons = eg.add_node("Cons", (1_i64, nil)).unwrap();

    let typed: Option<EClass<List>> = eg.eclass_of::<_, List>("Cons", (1_i64, nil)).unwrap();
    assert!(typed.is_some());
    assert_eq!(typed.unwrap().value(), cons);

    // Missing inputs return None without minting.
    let absent: Option<EClass<List>> = eg.eclass_of::<_, List>("Cons", (99_i64, nil)).unwrap();
    assert!(absent.is_none());
}

#[test]
fn test_lookup_on_constructor_errors() {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype List (Cons i64 List) (Nil))")
        .unwrap();
    // `lookup` is function-only; constructors must use `eclass_of`.
    let err = eg.lookup::<_, i64>("Cons", (1_i64, 0_i64));
    assert!(err.is_err(), "lookup on a constructor should error");
}

#[test]
fn test_eclass_of_on_function_errors() {
    struct AnySort;
    impl EqSortMarker for AnySort {
        const NAME: &'static str = "i64";
    }
    let eg = make_eg_with_function();
    let err = eg.eclass_of::<_, AnySort>("f", 1_i64);
    assert!(err.is_err(), "eclass_of on a function should error");
}

#[test]
fn test_set_constructor_errors() {
    // `set` on a constructor / relation table should error — they
    // don't accept a user-supplied output column.
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype List (Cons i64 List) (Nil))")
        .unwrap();
    let err = eg.set("Nil", RawValues(vec![]), 0_i64);
    assert!(err.is_err(), "set on a constructor should error");
}

#[test]
fn test_add_node_function_errors() {
    // `add_node` on a function table should error — functions need a
    // user-supplied output via `set`.
    let mut eg = make_eg_with_function();
    let err = eg.add_node("f", 1_i64);
    assert!(err.is_err(), "add_node on a function should error");
}

#[test]
fn test_set_replaces_function_value() {
    // For a function with :no-merge, re-setting the same key replaces
    // (or merges, depending on backend semantics).
    let mut eg = make_eg_with_function();
    eg.set("f", (5_i64,), 50_i64).unwrap();
    assert_eq!(eg.lookup::<_, i64>("f", 5_i64).unwrap(), Some(50));
}

#[test]
fn test_set_unknown_table_errors() {
    let mut eg = EGraph::default();
    let err = eg.set("nope", (1_i64,), 2_i64);
    assert!(err.is_err());
}

#[test]
fn test_lookup_unknown_table_errors() {
    let eg = EGraph::default();
    let err = eg.lookup::<_, i64>("nope", 1_i64);
    assert!(err.is_err());
}

#[test]
fn test_contains_unknown_table_returns_false() {
    let eg = EGraph::default();
    assert!(!eg.contains("nope", 1_i64));
}

#[test]
fn test_higher_arity_function() {
    // Function with 3 inputs and 1 output.
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(function g (i64 i64 i64) i64 :no-merge)")
        .unwrap();

    eg.set("g", (1_i64, 2_i64, 3_i64), 7_i64).unwrap();
    assert_eq!(
        eg.lookup::<_, i64>("g", (1_i64, 2_i64, 3_i64)).unwrap(),
        Some(7)
    );
    assert!(eg.contains("g", (1_i64, 2_i64, 3_i64)));
}

#[test]
fn test_string_inputs() {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(function name-length (String) i64 :no-merge)")
        .unwrap();
    eg.set("name-length", ("hello".to_string(),), 5_i64)
        .unwrap();
    assert_eq!(
        eg.lookup::<_, i64>("name-length", "hello".to_string())
            .unwrap(),
        Some(5)
    );
}

// Verifies that IntoRow is exported from the egglog crate and can be used
// generically over caller-defined functions.
#[test]
fn test_into_row_is_public() {
    fn touch<R: IntoRow>(eg: &mut EGraph, table: &str, key: R, value: i64) {
        eg.set(table, key, value).unwrap();
    }
    let mut eg = EGraph::default();
    eg.parse_and_run_program(
        None,
        "(function f1 (i64) i64 :no-merge)\n\
         (function f2 (i64 i64) i64 :no-merge)",
    )
    .unwrap();
    touch(&mut eg, "f1", (1_i64,), 11);
    touch(&mut eg, "f2", (1_i64, 2_i64), 12);
    assert!(eg.contains("f1", 1_i64));
    assert!(eg.contains("f2", (1_i64, 2_i64)));
}
