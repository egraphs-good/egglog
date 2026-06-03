//! Tests for `EGraph::function_entries`, `EGraph::constructor_enodes`,
//! and `EGraph::query`.

use egglog::prelude::*;
use egglog::{Error, Value};

/// `function_entries` returns one `(inputs, output)` pair per entry
/// of a function table.
#[test]
fn function_entries_i64_to_i64() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        "
(function f (i64) i64 :no-merge)
(set (f 1) 42)
(set (f 2) 43)
(set (f 7) 100)
",
    )?;

    let mut rows: Vec<(i64, i64)> = egraph
        .update(|fs| fs.function_entries("f"))?
        .into_iter()
        .map(|(inputs, output)| {
            (
                egraph.value_to_base::<i64>(inputs[0]),
                egraph.value_to_base::<i64>(output),
            )
        })
        .collect();
    rows.sort();
    assert_eq!(rows, vec![(1, 42), (2, 43), (7, 100)]);
    Ok(())
}

/// `function_entries` errors when called on a constructor.
#[test]
fn function_entries_on_constructor_errors() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        "(datatype List (Cons i64 List) (Nil))",
    )?;
    let err = egraph
        .update(|fs| fs.function_entries("Cons"))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("Cons") && err.contains("constructor"),
        "got: {err}"
    );
    Ok(())
}

/// `egraph.query(vars![x: i64], facts![(R x)])` matches and binds `x`
/// for every row in the relation. Each match is a
/// `HashMap<String, Value>` keyed by variable name.
#[test]
fn query_pattern_relation_one_var() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        "
(relation R (i64))
(R 1)
(R 2)
(R 7)
",
    )?;

    let mut results: Vec<i64> = egraph
        .query(vars![x: i64], facts![(R x)])?
        .into_iter()
        .map(|m| egraph.value_to_base::<i64>(m["x"]))
        .collect();
    results.sort();
    assert_eq!(results, vec![1, 2, 7]);
    Ok(())
}

/// `egraph.query(vars![], facts![(R 1 2)])` — zero-var case.
///
/// With zero `vars`, every match still produces a `HashMap` (which
/// will be empty since there are no variables to bind), so `.len()`
/// reports the match count. For ground facts that's 0 or 1.
#[test]
fn query_pattern_zero_vars_match() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        "
(relation R (i64 i64))
(R 1 2)
",
    )?;

    let hits = egraph.query(vars![], facts![(R 1 2)])?;
    assert_eq!(hits.len(), 1);

    let misses = egraph.query(vars![], facts![(R 5 5)])?;
    assert_eq!(misses.len(), 0);

    Ok(())
}

/// `constructor_enodes` on a constructor returns one
/// `(inputs, eclass)` per enode.
#[test]
fn constructor_enodes_basic() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        "
(sort Math)
(constructor Add (i64 i64) Math)
(let $a (Add 1 2))
(let $b (Add 3 4))
",
    )?;

    let enodes = egraph.update(|fs| fs.constructor_enodes("Add"))?;
    assert_eq!(enodes.len(), 2);
    for (inputs, _eclass) in &enodes {
        // Two i64 inputs.
        assert_eq!(inputs.len(), 2);
        let _x = egraph.value_to_base::<i64>(inputs[0]);
        let _y = egraph.value_to_base::<i64>(inputs[1]);
    }

    let mut input_pairs: Vec<(i64, i64)> = enodes
        .iter()
        .map(|(inputs, _)| {
            (
                egraph.value_to_base::<i64>(inputs[0]),
                egraph.value_to_base::<i64>(inputs[1]),
            )
        })
        .collect();
    input_pairs.sort();
    assert_eq!(input_pairs, vec![(1, 2), (3, 4)]);
    Ok(())
}

/// A relation desugars to a constructor with a synthetic non-unionable
/// eq-sort output. `constructor_enodes` returns those rows the same
/// way as any other constructor.
#[test]
fn constructor_enodes_relation() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        "
(relation R (i64))
(R 1)
(R 2)
",
    )?;

    let enodes = egraph.update(|fs| fs.constructor_enodes("R"))?;
    assert_eq!(enodes.len(), 2);
    for (inputs, _eclass) in &enodes {
        assert_eq!(inputs.len(), 1, "relation enode: one input");
    }

    // To get just the inputs, use the pattern-query form.
    let mut inputs: Vec<i64> = egraph
        .query(vars![x: i64], facts![(R x)])?
        .into_iter()
        .map(|m| egraph.value_to_base::<i64>(m["x"]))
        .collect();
    inputs.sort();
    assert_eq!(inputs, vec![1, 2]);
    Ok(())
}

/// Querying a pattern with multiple base-sort variables.
#[test]
fn query_pattern_two_vars() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        "
(function f (i64) i64 :no-merge)
(set (f 1) 10)
(set (f 2) 20)
(set (f 3) 30)
",
    )?;

    let mut rows: Vec<(i64, i64)> = egraph
        .query(vars![x: i64, y: i64], facts![(= (f x) y)])?
        .into_iter()
        .map(|m| {
            (
                egraph.value_to_base::<i64>(m["x"]),
                egraph.value_to_base::<i64>(m["y"]),
            )
        })
        .collect();
    rows.sort();
    assert_eq!(rows, vec![(1, 10), (2, 20), (3, 30)]);
    Ok(())
}

/// Iterating an empty function returns an empty Vec.
#[test]
fn function_entries_empty() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(None, "(function h (i64) i64 :no-merge)")?;
    let entries = egraph.update(|fs| fs.function_entries("h"))?;
    assert!(entries.is_empty());
    Ok(())
}

/// `function_entries` on a missing table returns a MissingTable error.
#[test]
fn function_entries_missing_table_errors() {
    let mut egraph = EGraph::default();
    let err = egraph
        .update(|fs| fs.function_entries("nonexistent"))
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("nonexistent") || msg.contains("Unbound"),
        "expected unbound-function-style error, got: {msg}"
    );
}

// Suppress unused-import warning when no test uses Value directly.
#[allow(dead_code)]
fn _value_import_check(_: Value) {}
