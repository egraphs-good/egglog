//! Tests for the [`EGraph::declare`] / [`DeclareTable`] builder API.

use egglog::Error;
use egglog::prelude::*;

#[test]
fn declare_function_then_query() -> Result<(), Error> {
    let mut egraph = EGraph::default();

    // Declare `(function f (i64) i64 :no-merge)` via the builder.
    egraph
        .declare("f")
        .input("i64")
        .output("i64")
        .function(None)?;

    // Set a fact via egglog DSL and query it back through the Rust API.
    egraph.parse_and_run_program(None, "(set (f 7) 42)")?;

    let results = query(&mut egraph, vars![y: i64], facts![(= (f 7) y)])?;

    let y = egraph.base_to_value::<i64>(42);
    let rows: Vec<_> = results.iter().collect();
    assert_eq!(rows, [[y]]);

    Ok(())
}

#[test]
fn declare_relation() -> Result<(), Error> {
    let mut egraph = EGraph::default();

    egraph.declare("R").input("i64").input("i64").relation()?;

    // Insert into the relation and query it.
    egraph.parse_and_run_program(None, "(R 1 2)")?;

    let results = query(&mut egraph, vars![a: i64, b: i64], facts![(R a b)])?;

    let a = egraph.base_to_value::<i64>(1);
    let b = egraph.base_to_value::<i64>(2);
    let rows: Vec<_> = results.iter().collect();
    assert_eq!(rows, [[a, b]]);

    Ok(())
}

#[test]
fn declare_constructor_for_datatype() -> Result<(), Error> {
    let mut egraph = EGraph::default();

    // Declare the sort first, then a constructor that takes two `Math`s and
    // returns a `Math`.
    add_sort(&mut egraph, "Math")?;
    egraph
        .declare("Add")
        .input("Math")
        .input("Math")
        .output("Math")
        .constructor(None, false)?;

    // Also declare a leaf constructor with no inputs so we can build a term.
    egraph
        .declare("Lit")
        .input("i64")
        .output("Math")
        .constructor(None, false)?;

    // Build `(Add (Lit 1) (Lit 2))` and confirm we can query the relation.
    egraph.parse_and_run_program(None, "(Add (Lit 1) (Lit 2))")?;

    let results = query(&mut egraph, vars![x: i64], facts![(Add (Lit x) (Lit 2))])?;

    let x = egraph.base_to_value::<i64>(1);
    let rows: Vec<_> = results.iter().collect();
    assert_eq!(rows, [[x]]);

    Ok(())
}

#[test]
#[should_panic(expected = "no output sort set")]
fn declare_function_missing_output_panics() {
    let mut egraph = EGraph::default();

    // Forgot the `.output(...)` call -- we expect a clear panic, not a
    // silently-malformed schema.
    let _ = egraph.declare("g").input("i64").function(None);
}

#[test]
#[should_panic(expected = "no output sort set")]
fn declare_constructor_missing_output_panics() {
    let mut egraph = EGraph::default();
    add_sort(&mut egraph, "Math").unwrap();

    let _ = egraph
        .declare("MissingOut")
        .input("Math")
        .constructor(None, false);
}

#[test]
fn declare_typed_convenience_methods() -> Result<(), Error> {
    let mut egraph = EGraph::default();

    // Use `input_base::<T>()` / `output_base::<T>()` for primitive types.
    egraph
        .declare("h")
        .input_base::<i64>()
        .output_base::<i64>()
        .function(None)?;

    egraph.parse_and_run_program(None, "(set (h 5) 25)")?;

    let results = query(&mut egraph, vars![y: i64], facts![(= (h 5) y)])?;

    let y = egraph.base_to_value::<i64>(25);
    let rows: Vec<_> = results.iter().collect();
    assert_eq!(rows, [[y]]);

    Ok(())
}
