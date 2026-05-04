//! Tests for the `rust_rule!` macro.

use egglog::Error;
use egglog::prelude::*;
use egglog::rust_rule;

const FIB_RULESET: &str = "fib_ruleset";

fn fib_setup() -> Result<EGraph, Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        "
(function fib (i64) i64 :no-merge)
(set (fib 0) 0)
(set (fib 1) 1)
        ",
    )?;
    add_ruleset(&mut egraph, FIB_RULESET)?;
    Ok(egraph)
}

/// Basic case: i64 vars, parallel to `benches/rust_api_benchmarking.rs:fib_setup`.
#[test]
fn rust_rule_macro_basic_i64() -> Result<(), Error> {
    let mut egraph = fib_setup()?;
    let ruleset = FIB_RULESET;

    rust_rule!(
        &mut egraph,
        "fib_rule",
        ruleset,
        vars![x: i64, f0: i64, f1: i64],
        facts![
            (= f0 (fib x))
            (= f1 (fib (+ x 1)))
        ],
        |ctx, b| {
            // b.x, b.f0, b.f1 are typed i64 — no per-arg `value_to_base`.
            let y = ctx.base_to_value::<i64>(b.x + 2);
            let f2 = ctx.base_to_value::<i64>(b.f0 + b.f1);
            ctx.insert("fib", [y, f2].into_iter());
            Some(())
        }
    )?;

    for _ in 0..20 {
        run_ruleset(&mut egraph, ruleset)?;
    }

    let big = 20;
    let results = query(
        &mut egraph,
        vars![f: i64],
        facts![(= (fib (unquote exprs::int(big))) f)],
    )?;
    let y = egraph.base_to_value::<i64>(6765);
    let rows: Vec<_> = results.iter().collect();
    assert_eq!(rows, [[y]]);

    Ok(())
}

/// Mixed types: i64 + String.
///
/// Stores `(name -> i64)` rows and a rule that, when matched, inserts a
/// derived row into another i64-keyed table — exercising String binding
/// extraction through the macro.
#[test]
fn rust_rule_macro_mixed_types() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        "
(function age (String) i64 :no-merge)
(function age_squared (String) i64 :no-merge)
(set (age \"alice\") 30)
(set (age \"bob\") 25)
        ",
    )?;
    let ruleset = "age_ruleset";
    add_ruleset(&mut egraph, ruleset)?;

    rust_rule!(
        &mut egraph,
        "age_squared_rule",
        ruleset,
        vars![name: String, a: i64],
        facts![ (= a (age name)) ],
        |ctx, b| {
            // b.name: String, b.a: i64
            assert!(b.name == "alice" || b.name == "bob");
            let key = ctx.base_to_value::<egglog::sort::S>(b.name.clone().into());
            let val = ctx.base_to_value::<i64>(b.a * b.a);
            ctx.insert("age_squared", [key, val].into_iter());
            Some(())
        }
    )?;

    run_ruleset(&mut egraph, ruleset)?;

    // Verify the derived rows exist.
    let results = query(
        &mut egraph,
        vars![n: String, sq: i64],
        facts![ (= sq (age_squared n)) ],
    )?;
    assert_eq!(results.iter().count(), 2);

    Ok(())
}

/// Empty vars: no bindings struct fields, no var slot.
#[test]
fn rust_rule_macro_empty_vars() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        "
(function flag () i64 :no-merge)
(function trigger () i64 :no-merge)
(set (trigger) 1)
        ",
    )?;
    let ruleset = "empty_ruleset";
    add_ruleset(&mut egraph, ruleset)?;

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();

    rust_rule!(
        &mut egraph,
        "empty_rule",
        ruleset,
        vars![],
        facts![(trigger)],
        |ctx, _b| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            let v = ctx.base_to_value::<i64>(42);
            ctx.insert("flag", [v].into_iter());
            Some(())
        }
    )?;

    run_ruleset(&mut egraph, ruleset)?;
    assert!(counter.load(Ordering::SeqCst) >= 1);

    Ok(())
}

/// Verify field types: extract bindings into a known type and assert.
/// The `: i64` annotation on the local would fail to compile if the
/// generated struct's field were the wrong type.
#[test]
fn rust_rule_macro_typed_fields() -> Result<(), Error> {
    let mut egraph = fib_setup()?;
    let ruleset = FIB_RULESET;

    use std::sync::{Arc, Mutex};
    let captured: Arc<Mutex<Vec<(i64, i64, i64)>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = captured.clone();

    rust_rule!(
        &mut egraph,
        "capture_rule",
        ruleset,
        vars![x: i64, f0: i64, f1: i64],
        facts![
            (= f0 (fib x))
            (= f1 (fib (+ x 1)))
        ],
        |_ctx, b| {
            // Type-check the field types explicitly.
            let x: i64 = b.x;
            let f0: i64 = b.f0;
            let f1: i64 = b.f1;
            captured_clone.lock().unwrap().push((x, f0, f1));
            Some(())
        }
    )?;

    run_ruleset(&mut egraph, ruleset)?;

    // The seed rows are (fib 0)=0 and (fib 1)=1, so x=0 should match
    // with f0=0, f1=1.
    let rows = captured.lock().unwrap();
    assert!(rows.iter().any(|&(x, f0, f1)| x == 0 && f0 == 0 && f1 == 1));

    Ok(())
}
