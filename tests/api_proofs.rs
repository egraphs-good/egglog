//! Verify the new Rust API gives clear errors when used with the
//! proof system enabled. `rust_rule` callbacks and direct e-graph
//! writes via `update` both bypass the proof-encoding
//! pipeline and must surface a helpful error rather than silently
//! producing unverifiable proofs.

use egglog::Error;
use egglog::prelude::*;

#[test]
fn rust_rule_with_proofs_enabled_errors() {
    let mut eg = EGraph::new_with_proofs();
    eg.parse_and_run_program(None, "(function f (i64) i64 :merge new)")
        .unwrap();
    add_ruleset(&mut eg, "r").unwrap();

    let result = rust_rule(
        &mut eg,
        "test_rule",
        "r",
        vars![x: i64],
        facts![(= y (f x))],
        |_, _| Some(()),
    );

    let err = result.expect_err("rust_rule should fail under proofs");
    assert!(
        matches!(err, Error::ProofsIncompatibleApi { api, .. } if api == "rust_rule"),
        "expected ProofsIncompatibleApi(rust_rule), got: {err}"
    );
}

#[test]
fn rust_rule_full_with_proofs_enabled_errors() {
    let mut eg = EGraph::new_with_proofs();
    eg.parse_and_run_program(None, "(function f (i64) i64 :merge new)")
        .unwrap();
    add_ruleset(&mut eg, "r").unwrap();

    let result = rust_rule_full(
        &mut eg,
        "test_rule",
        "r",
        vars![x: i64],
        facts![(= y (f x))],
        |_, _| Some(()),
    );

    let err = result.expect_err("rust_rule_full should fail under proofs");
    assert!(
        matches!(err, Error::ProofsIncompatibleApi { api, .. } if api == "rust_rule_full"),
        "expected ProofsIncompatibleApi(rust_rule_full), got: {err}"
    );
}

#[test]
fn update_with_proofs_enabled_errors() {
    let mut eg = EGraph::new_with_proofs();
    eg.parse_and_run_program(None, "(function f (i64) i64 :merge new)")
        .unwrap();

    let result = eg.update(|mut fs| fs.set("f", (1_i64,), 42_i64));

    let err = result.expect_err("update should fail under proofs");
    assert!(
        matches!(err, Error::ProofsIncompatibleApi { api, .. } if api == "EGraph::update"),
        "expected ProofsIncompatibleApi(EGraph::update), got: {err}"
    );
}

#[test]
fn query_with_proofs_enabled_errors_with_query_api_name() {
    // Regression: previously the failure surfaced through the
    // rust_rule check inside query, so the error pointed at
    // "rust_rule" instead of "EGraph::query".
    let mut eg = EGraph::new_with_proofs();
    eg.parse_and_run_program(None, "(function f (i64) i64 :merge new)")
        .unwrap();

    let result = eg.query(vars![x: i64], facts![(= y (f x))]);

    let err = result.expect_err("EGraph::query should fail under proofs");
    assert!(
        matches!(err, Error::ProofsIncompatibleApi { api, .. } if api == "EGraph::query"),
        "expected ProofsIncompatibleApi(EGraph::query), got: {err}"
    );
}
