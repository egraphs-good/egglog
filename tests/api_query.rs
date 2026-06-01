//! Tests for the typed `EGraph::table_rows` and `EGraph::query` API.
//! See `src/lib.rs` for `FromRow` and these methods.

use egglog::prelude::*;
use egglog::{Error, Value};

/// Test 1: `eg.table_rows::<(i64, i64)>("f")` returns all rows of a
/// function table as typed tuples. Order is `(input..., output)`.
#[test]
fn query_table_i64_to_i64() -> Result<(), Error> {
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

    let mut rows: Vec<(i64, i64)> = egraph.table_rows::<(i64, i64)>("f")?;
    rows.sort();
    assert_eq!(rows, vec![(1, 42), (2, 43), (7, 100)]);
    Ok(())
}

/// `table_rows::<Vec<Value>>` is the escape hatch — useful when the
/// schema contains eq-sort or container columns and you want raw `Value`s.
#[test]
fn query_table_raw_values() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        "
(function g (i64) i64 :no-merge)
(set (g 5) 50)
",
    )?;

    let rows: Vec<Vec<Value>> = egraph.table_rows::<Vec<Value>>("g")?;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].len(), 2);
    assert_eq!(egraph.value_to_base::<i64>(rows[0][0]), 5);
    assert_eq!(egraph.value_to_base::<i64>(rows[0][1]), 50);
    Ok(())
}

/// Test 2: `query::<(i64,)>(vars![x: i64], facts![(R x)])`
/// matches and binds `x` for every row in the relation.
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

    let mut results: Vec<(i64,)> = egraph.query::<(i64,)>(vars![x: i64], facts![(R x)])?;
    results.sort();
    assert_eq!(results, vec![(1,), (2,), (7,)]);
    Ok(())
}

/// Test 3: `query::<()>(vars![], facts![(R 1 2)])`.
///
/// Decision: with zero `vars`, we return one empty tuple per match.
/// That is, `Vec<()>` has length equal to the number of times the
/// fact pattern is satisfied. This mirrors the behaviour of an
/// egglog rule with no captured variables firing once per unique
/// match (or zero times if there is no match).
///
/// In practice for ground (no variables / no joins) facts, that's
/// either 0 or 1 matches.
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

    // Fact present -> one empty tuple.
    let hits: Vec<()> = egraph.query::<()>(vars![], facts![(R 1 2)])?;
    assert_eq!(hits.len(), 1);

    // Fact absent -> empty Vec.
    let misses: Vec<()> = egraph.query::<()>(vars![], facts![(R 5 5)])?;
    assert_eq!(misses.len(), 0);

    Ok(())
}

/// Test 4: Iterating a constructor table.
///
/// What comes back: each row is `[input_0, ..., input_n, output_eclass]`
/// with raw `Value`s for any eq-sort columns. Eq-sort outputs are
/// e-class ids, NOT extracted terms — extraction is up to the caller
/// (use `EGraph::extract_value` or similar). Base-sort columns can be
/// converted via `value_to_base`.
///
/// For a constructor `(constructor Add (i64 i64) Math)`:
/// - The two `i64` inputs come back as base values.
/// - The `Math` output comes back as an eq-sort `Value` (e-class id).
#[test]
fn query_constructor_table_eclass_values() -> Result<(), Error> {
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

    let rows: Vec<Vec<Value>> = egraph.table_rows::<Vec<Value>>("Add")?;
    assert_eq!(rows.len(), 2);
    for row in &rows {
        // Two i64 inputs and one eq-sort output.
        assert_eq!(row.len(), 3);
        let _x = egraph.value_to_base::<i64>(row[0]);
        let _y = egraph.value_to_base::<i64>(row[1]);
        // row[2] is an eq-sort id; we just verify we get back a Value.
        let _: Value = row[2];
    }

    let mut input_pairs: Vec<(i64, i64)> = rows
        .iter()
        .map(|r| {
            (
                egraph.value_to_base::<i64>(r[0]),
                egraph.value_to_base::<i64>(r[1]),
            )
        })
        .collect();
    input_pairs.sort();
    assert_eq!(input_pairs, vec![(1, 2), (3, 4)]);
    Ok(())
}

/// A relation table is sugar for a constructor with a synthetic
/// non-unionable eq-sort output. `EGraph::table_rows` does not
/// special-case relations — it returns whatever the backend stores,
/// which is `(input..., eclass)`. Use `Vec<Value>` (or `query` which
/// binds only the inputs you name) to inspect such rows.
#[test]
fn query_relation_exposes_synthetic_output() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        "
(relation R (i64))
(R 1)
(R 2)
",
    )?;

    // The backend stores the synthetic eclass column, so the row has
    // 2 columns: the i64 input + the eclass Value.
    let raw: Vec<Vec<Value>> = egraph.table_rows::<Vec<Value>>("R")?;
    for row in &raw {
        assert_eq!(row.len(), 2, "relation row exposes (input, eclass)");
    }

    // To get just the inputs as typed tuples, use the pattern-query form.
    let mut inputs: Vec<(i64,)> =
        egraph.query::<(i64,)>(vars![x: i64], facts![(R x)])?;
    inputs.sort();
    assert_eq!(inputs, vec![(1,), (2,)]);
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

    let mut rows: Vec<(i64, i64)> =
        egraph.query::<(i64, i64)>(vars![x: i64, y: i64], facts![(= (f x) y)])?;
    rows.sort();
    assert_eq!(rows, vec![(1, 10), (2, 20), (3, 30)]);
    Ok(())
}

/// Querying an empty function returns an empty Vec.
#[test]
fn query_empty_table() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(None, "(function h (i64) i64 :no-merge)")?;
    let rows: Vec<(i64, i64)> = egraph.table_rows::<(i64, i64)>("h")?;
    assert!(rows.is_empty());
    Ok(())
}

/// Querying a missing function returns an `UnboundFunction` error.
#[test]
fn query_missing_table_errors() {
    let mut egraph = EGraph::default();
    let err = egraph.table_rows::<(i64, i64)>("nonexistent").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("nonexistent") || msg.contains("Unbound"),
        "expected unbound-function-style error, got: {msg}"
    );
}

/// Test: the legacy free `query()` helper still works (it just goes
/// through `EGraph::query::<Vec<Value>>` internally now).
#[test]
#[allow(deprecated)]
fn legacy_query_still_works() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        "
(function f (i64) i64 :no-merge)
(set (f 1) 42)
",
    )?;

    let results = query(&mut egraph, vars![x: i64, y: i64], facts![(= (f x) y)])?;
    assert!(results.any_matches());
    let rows: Vec<&[Value]> = results.iter().collect();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].len(), 2);
    assert_eq!(egraph.value_to_base::<i64>(rows[0][0]), 1);
    assert_eq!(egraph.value_to_base::<i64>(rows[0][1]), 42);
    Ok(())
}
