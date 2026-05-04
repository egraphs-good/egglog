//! Cross-cutting integration tests for the API redesign.
//!
//! Each test below pulls together features from multiple branches:
//! `oflatt-api-typed-egraph` (set/add_node/lookup/eclass_of/EClass<M>),
//! `oflatt-api-rule-ergo` (rust_rule! macro, drop &mut on container
//! interning), `oflatt-api-cleanup` (declare-table builder), and
//! `oflatt-api-block-macro-draft` (egglog! proc macro). The point is
//! to verify these pieces compose — that you can write a realistic
//! program touching all four surfaces.

use egglog::prelude::*;
use egglog::{EClass, EqSortMarker, Error, RawValues, rust_rule};
use egglog_macros::egglog;

/// End-to-end: declare a self-contained block via `egglog!` (the
/// macro validates at compile time), then drive the e-graph from
/// Rust via the typed `set` / `lookup` / `query` methods, and step
/// the rule via the standard ruleset machinery.
///
/// Note: the egglog! block must be self-contained because the
/// proc-macro typechecks against a *fresh* EGraph at expansion time
/// — it can't see Rust-side declarations like `eg.declare(...)`.
/// (Documented as a known limitation; `egglog-block-macro-proposal.md`
/// step 1 would lift this.)
#[test]
fn integration_full_pipeline() -> Result<(), Error> {
    let mut eg = EGraph::default();

    egglog!(
        eg,
        "(function fib (i64) i64 :no-merge)
         (ruleset fib_rs)
         (rule ((= f0 (fib x)) (= f1 (fib (+ x 1))))
               ((set (fib (+ x 2)) (+ f0 f1)))
               :ruleset fib_rs)"
    )?;

    // direct fact ops via set (typed-egraph)
    eg.set("fib", (0_i64,), 0_i64)?;
    eg.set("fib", (1_i64,), 1_i64)?;

    // step the named ruleset
    for _ in 0..10 {
        run_ruleset(&mut eg, "fib_rs")?;
    }

    // typed query (typed-egraph)
    let rows: Vec<(i64, i64)> = eg.query::<(i64, i64)>("fib")?;
    let fib5 = rows.iter().find(|(k, _)| *k == 5).map(|(_, v)| *v);
    assert_eq!(fib5, Some(5));

    // typed lookup (typed-egraph)
    assert_eq!(eg.lookup::<_, i64>("fib", 5_i64)?, Some(5));
    assert!(eg.contains("fib", 5_i64));

    Ok(())
}

/// Declare-table builder (cleanup) + typed runtime API (typed-egraph)
/// without going through `egglog!`. This is the path users on the
/// "I want full Rust control, no DSL strings" track will take.
#[test]
fn integration_declare_builder_with_typed_api() -> Result<(), Error> {
    let mut eg = EGraph::default();

    // declare-table builder
    eg.declare("f").input("i64").output("i64").function(None)?;
    eg.declare("R").input("i64").input("i64").relation()?;

    // typed runtime API
    eg.set("f", (1_i64,), 42_i64)?;
    eg.add_node("R", (1_i64, 2_i64))?;

    assert_eq!(eg.lookup::<_, i64>("f", 1_i64)?, Some(42));
    assert!(eg.contains("R", (1_i64, 2_i64)));

    Ok(())
}

/// `EClass<M>` flows through `set` (as a key column) and back out of
/// `eclass_of` — exercises the row trait surface end-to-end with
/// typed eclass handles.
#[test]
fn integration_eclass_round_trip() -> Result<(), Error> {
    struct List;
    impl EqSortMarker for List {
        const NAME: &'static str = "List";
    }

    let mut eg = EGraph::default();

    egglog!(
        eg,
        "(datatype List (Cons i64 List) (Nil))
         (function list_length (List) i64 :no-merge)"
    )?;

    let nil = eg.add_node("Nil", RawValues(vec![]))?;
    let one_nil = eg.add_node("Cons", (1_i64, nil))?;
    let two_one_nil = eg.add_node("Cons", (2_i64, one_nil))?;

    // Tag the eclass with its sort.
    let typed_two: EClass<List> = eg.typed_eclass(two_one_nil).unwrap();

    // Use the typed handle as a row key — IntoColumn impl on EClass<M>.
    eg.set("list_length", (typed_two,), 2_i64)?;
    assert_eq!(eg.lookup::<_, i64>("list_length", typed_two)?, Some(2));

    // Read constructor row back as a typed eclass.
    let head_eclass: Option<EClass<List>> = eg.eclass_of::<_, List>("Cons", (2_i64, one_nil))?;
    assert_eq!(head_eclass.map(|e| e.value()), Some(two_one_nil));

    Ok(())
}

/// `rust_rule!` (rule-ergo) calling `ctx.set` (typed-egraph) inside
/// the action body — verifies the macro's bindings struct + the typed
/// write API compose.
#[test]
fn integration_rust_rule_with_set() -> Result<(), Error> {
    let mut eg = EGraph::default();
    egglog!(
        eg,
        "(function fib (i64) i64 :no-merge)
         (set (fib 0) 0)
         (set (fib 1) 1)"
    )?;

    let ruleset = "fib_ruleset";
    add_ruleset(&mut eg, ruleset)?;

    rust_rule!(
        &mut eg,
        "step",
        ruleset,
        vars![x: i64, f0: i64, f1: i64],
        facts![ (= f0 (fib x)) (= f1 (fib (+ x 1))) ],
        |ctx, b| {
            // Typed bindings AND typed insert: no value_to_base /
            // base_to_value / iterator construction at the call site.
            ctx.set("fib", (b.x + 2,), b.f0 + b.f1);
            Some(())
        }
    )?;

    for _ in 0..10 {
        run_ruleset(&mut eg, ruleset)?;
    }

    assert_eq!(eg.lookup::<_, i64>("fib", 8_i64)?, Some(21));
    Ok(())
}

/// Subtype-mismatch errors: each direction-specific method errors
/// loudly when called on the wrong subtype.
#[test]
fn integration_subtype_errors() -> Result<(), Error> {
    let mut eg = EGraph::default();
    egglog!(
        eg,
        "(function f (i64) i64 :no-merge)
         (datatype List (Nil))"
    )?;

    // set() on a constructor → WrongTableSubtype
    assert!(eg.set("Nil", RawValues(vec![]), 0_i64).is_err());
    // add_node() on a function → WrongTableSubtype
    assert!(eg.add_node("f", 1_i64).is_err());
    // lookup() on a constructor → WrongTableSubtype
    assert!(eg.lookup::<_, i64>("Nil", RawValues(vec![])).is_err());
    // eclass_of() on a function → WrongTableSubtype
    struct AnyMarker;
    impl EqSortMarker for AnyMarker {
        const NAME: &'static str = "i64";
    }
    assert!(eg.eclass_of::<_, AnyMarker>("f", 1_i64).is_err());

    Ok(())
}
