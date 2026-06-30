//! Tests for `EGraph::function_entries`, `EGraph::constructor_enodes`,
//! and `EGraph::query`.

use egglog::prelude::*;
use egglog::{Error, Value};

/// `function_entries` calls the callback once per entry of a function
/// table with its `inputs` and `output`.
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

    // `function_entries` takes `&self`, so the callback may call other
    // `&self` methods like `value_to_base` directly on `egraph`.
    let mut rows: Vec<(i64, i64)> = Vec::new();
    egraph.function_entries("f", |entry| {
        rows.push((
            egraph.value_to_base::<i64>(entry.inputs[0]),
            egraph.value_to_base::<i64>(entry.output),
        ));
    })?;
    rows.sort();
    assert_eq!(rows, vec![(1, 42), (2, 43), (7, 100)]);
    Ok(())
}

/// `function_entries` errors when called on a constructor.
#[test]
fn function_entries_on_constructor_errors() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(None, "(datatype List (Cons i64 List) (Nil))")?;
    let err = egraph
        .function_entries("Cons", |_| {})
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

/// `constructor_enodes` calls the callback once per enode with its
/// `children` and `eclass`.
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

    let mut children: Vec<Vec<Value>> = Vec::new();
    egraph.constructor_enodes("Add", |enode| {
        // Two i64 inputs.
        assert_eq!(enode.children.len(), 2);
        children.push(enode.children.to_vec());
    })?;
    assert_eq!(children.len(), 2);

    let mut input_pairs: Vec<(i64, i64)> = children
        .iter()
        .map(|c| {
            (
                egraph.value_to_base::<i64>(c[0]),
                egraph.value_to_base::<i64>(c[1]),
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

    let mut count = 0;
    egraph.constructor_enodes("R", |enode| {
        assert_eq!(enode.children.len(), 1, "relation enode: one input");
        count += 1;
    })?;
    assert_eq!(count, 2);

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

/// Iterating an empty function never calls the callback.
#[test]
fn function_entries_empty() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(None, "(function h (i64) i64 :no-merge)")?;
    let mut count = 0;
    egraph.function_entries("h", |_| count += 1)?;
    assert_eq!(count, 0);
    Ok(())
}

/// `function_entries` on a missing table returns a MissingTable error.
#[test]
fn function_entries_missing_table_errors() {
    let egraph = EGraph::default();
    let err = egraph.function_entries("nonexistent", |_| {}).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("nonexistent") || msg.contains("Unbound"),
        "expected unbound-function-style error, got: {msg}"
    );
}

/// The `subsumed` field reports which enodes have been subsumed;
/// subsumed enodes are still visited.
#[test]
fn constructor_enodes_reports_subsumed() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        "
(datatype Math (Num i64))
(Num 1)
(Num 2)
(subsume (Num 1))
",
    )?;

    let mut rows: Vec<(i64, bool)> = Vec::new();
    egraph.constructor_enodes("Num", |enode| {
        rows.push((
            egraph.value_to_base::<i64>(enode.children[0]),
            enode.subsumed,
        ));
    })?;
    assert_eq!(rows.len(), 2, "subsumed enodes are still visited");
    rows.sort();
    assert_eq!(rows, vec![(1, true), (2, false)]);
    Ok(())
}

// Suppress unused-import warning when no test uses Value directly.
#[allow(dead_code)]
fn _value_import_check(_: Value) {}
