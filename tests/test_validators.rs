//! Consolidated tests for primitive validators and the add_literal_prim macro

use egglog::prelude::*;
use egglog_add_primitive::add_literal_prim;
use std::sync::Arc;

// ================ Tests for add_literal_prim macro ================

#[test]
#[allow(clippy::let_unit_value)]
#[allow(unused_must_use)]
fn test_add_literal_prim_basic() {
    let mut eg = EGraph::default();

    // Test basic infallible operations
    let _ = add_literal_prim!(&mut eg, "add" = |a: i64, b: i64| -> i64 { a + b });
    let _ = add_literal_prim!(&mut eg, "negate" = |a: i64| -> i64 { -a });
    let _ = add_literal_prim!(&mut eg, "and" = |a: bool, b: bool| -> bool { a && b });
    let _ = add_literal_prim!(&mut eg, "double" = |a: i64| -> i64 { a * 2 });
}

#[test]
#[allow(clippy::let_unit_value)]
#[allow(unused_must_use)]
fn test_add_literal_prim_fallible() {
    let mut eg = EGraph::default();

    // Test fallible operations with -?> arrow
    let _ =
        add_literal_prim!(&mut eg, "checked_add" = |a: i64, b: i64| -?> i64 { a.checked_add(b) });
    let _ =
        add_literal_prim!(&mut eg, "checked_sub" = |a: i64, b: i64| -?> i64 { a.checked_sub(b) });
}

#[test]
#[allow(clippy::let_unit_value)]
#[allow(unused_must_use)]
fn test_add_literal_prim_with_validator() {
    let mut eg = EGraph::with_proofs();

    // Add a primitive with automatic validator
    let _ = add_literal_prim!(&mut eg, "double" = |a: i64| -> i64 { a * 2 });

    // Now use it in a simple program
    let program = r#"
        (datatype Math (Num i64) (Double Math))
        (let x (Num 5))
        (let y (Double x))
    "#;

    eg.parse_and_run_program(None, program).unwrap();
}

// ================ Tests for validator proof checking ================

#[test]
fn test_validators_compute_during_proof_checking() {
    // This test verifies that validators are actually computing results
    // and that proof checking uses those results
    let mut eg = EGraph::with_proof_checking();

    let program = r#"
        ; Test boolean operations - these have validators via add_literal_prim!
        (check (= (and true false) false))  ; Validator computes false
        (check (= (or false false) false))  ; Validator computes false  
        (check (= (not false) true))        ; Validator computes true
        (check (= (xor true false) true))   ; Validator computes true
    "#;

    let result = eg.parse_and_run_program(None, program);
    assert!(
        result.is_ok(),
        "Boolean validators should validate correct results"
    );
}

#[test]
fn test_validators_reject_incorrect_claims() {
    // This test verifies validators reject false claims about primitive results
    let mut eg = EGraph::with_proof_checking();

    // Try to claim (and true true) = false, which is wrong
    let program = r#"
        (check (= (and true true) false))
    "#;

    let result = eg.parse_and_run_program(None, program);
    assert!(
        result.is_err(),
        "Validator should reject: (and true true) != false"
    );
}

#[test]
fn test_validators_work_in_proof_mode() {
    // Test that validators work when proofs are enabled (not just proof checking)
    let mut eg = EGraph::with_proofs();

    let program = r#"
        ; These should work even in proof mode
        (check (= (and true false) false))
        (check (= (or true true) true))
    "#;

    let result = eg.parse_and_run_program(None, program);
    assert!(
        result.is_ok(),
        "Validators should work in proof mode: {:?}",
        result
    );
}

#[test]
fn test_integer_validators_are_installed() {
    // Check that integer arithmetic has validators
    let mut eg = EGraph::with_proof_checking();

    let program = r#"
        ; These should work if validators are installed
        (check (= (+ 1 1) 2))
        (check (= (- 5 2) 3))
        (check (= (* 2 3) 6))
    "#;

    let result = eg.parse_and_run_program(None, program);
    assert!(
        result.is_ok(),
        "Integer arithmetic validators should be installed: {:?}",
        result
    );
}

// ================ Tests for typed validators ================

#[test]
fn test_wrong_type_validator_rejected() {
    // Test that validators correctly reject when types don't match
    let mut eg = EGraph::default();

    // Try to add a validator for a non-existent primitive signature
    let result = eg.add_primitive_validator_typed(
        "nonexistent",
        &["i64", "i64"],
        "i64",
        Arc::new(|_, _| None),
    );

    assert!(
        result.is_err(),
        "Should fail to add validator for non-existent primitive"
    );
}

#[test]
fn test_validator_distinguishes_operations() {
    // Ensure validators correctly distinguish between different operations
    let mut eg = EGraph::with_proof_checking();

    // AND and OR are different operations, should get different results
    let program = r#"
        (check (= (and true false) false))  ; AND gives false
        (check (= (or true false) true))    ; OR gives true
    "#;

    let result = eg.parse_and_run_program(None, program);
    assert!(
        result.is_ok(),
        "Validators should distinguish between AND and OR"
    );

    // Test that we can't swap them
    let mut eg2 = EGraph::with_proof_checking();
    let wrong_program = r#"
        (check (= (and true false) true))  ; Wrong! AND gives false, not true
    "#;

    let wrong_result = eg2.parse_and_run_program(None, wrong_program);
    assert!(
        wrong_result.is_err(),
        "Validator should reject swapped AND/OR results"
    );
}
