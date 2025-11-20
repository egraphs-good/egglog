//! Comprehensive tests for the improved proof checker

use egglog::*;

#[test]
fn test_rule_proof_validation() {
    // Test that rule proofs actually verify the rule produces the claimed result
    let mut egraph = EGraph::with_proof_checking();

    let program = r#"
        (datatype Math
            (Add Math Math)
            (Num i64))
        
        (rule ((= x (Add a b)))
              ((union x (Add b a))))
        
        (let t1 (Add (Num 1) (Num 2)))
        (let t2 (Add (Num 2) (Num 1)))
        (run 1)
        
        ; Use the actual terms, not the let-bound names
        (check (= (Add (Num 1) (Num 2)) (Add (Num 2) (Num 1))))
    "#;

    // This should succeed - the rule correctly produces the equality
    let result = egraph.parse_and_run_program(None, program);
    assert!(
        result.is_ok(),
        "Rule proof validation should succeed: {:?}",
        result
    );
}

#[test]
fn test_congruence_proof_validation() {
    // Test that congruence proofs verify the rewritten term matches
    let mut egraph = EGraph::with_proof_checking();

    let program = r#"
        (datatype Math
            (F Math Math)
            (Num i64))
        
        (let a (Num 1))
        (let b (Num 2))
        (let c (Num 1))
        
        ; a = c, so F(a, b) = F(c, b) by congruence
        (union a c)
        (let fab (F a b))
        (let fcb (F c b))
        
        (check (= fab fcb))
    "#;

    let result = egraph.parse_and_run_program(None, program);
    assert!(
        result.is_ok(),
        "Congruence proof should validate: {:?}",
        result
    );
}

#[test]
fn test_p_fiat_validation() {
    // Test that PFiat proofs are properly validated
    let mut egraph = EGraph::with_proof_checking();

    // PFiat should be used for user-defined globals
    let program = r#"
        (datatype Math (Num i64))
        
        ; This creates a global that uses PFiat
        (let x (Num 42))
        
        ; Check that x exists
        (check x)
    "#;

    let result = egraph.parse_and_run_program(None, program);
    assert!(
        result.is_ok(),
        "PFiat for globals should be accepted: {:?}",
        result
    );
}

#[test]
fn test_primitive_validation_in_proofs() {
    // Test that primitive operations are validated in proofs
    let mut egraph = EGraph::with_proof_checking();

    let program = r#"
        (datatype Math (Num i64))
        
        (let x (Num (+ 2 3)))
        (let y (Num 5))
        
        ; This should work because 2 + 3 = 5
        (check (= x y))
    "#;

    let result = egraph.parse_and_run_program(None, program);
    assert!(
        result.is_ok(),
        "Primitive validation should succeed: {:?}",
        result
    );
}

#[test]
fn test_complex_congruence_chain() {
    // Test complex congruence with multiple rewrites
    let mut egraph = EGraph::with_proof_checking();

    let program = r#"
        (datatype Math
            (G Math Math Math)
            (Num i64))
        
        (let a (Num 1))
        (let b (Num 2))
        (let c (Num 3))
        (let d (Num 1))
        (let e (Num 2))
        
        ; Create equalities
        (union a d)  ; a = d
        (union b e)  ; b = e
        
        ; By congruence: G(a,b,c) = G(d,e,c)
        (let gabc (G a b c))
        (let gdec (G d e c))
        
        (check (= gabc gdec))
    "#;

    let result = egraph.parse_and_run_program(None, program);
    assert!(
        result.is_ok(),
        "Complex congruence should validate: {:?}",
        result
    );
}

#[test]
fn test_rule_with_multiple_premises() {
    // Test that rules with multiple premises are validated correctly
    let mut egraph = EGraph::with_proof_checking();

    let program = r#"
        (datatype Math
            (Add Math Math)
            (Mul Math Math)
            (Zero)
            (One))
        
        ; Distributivity rule
        (rule ((= x (Mul a (Add b c))))
              ((union x (Add (Mul a b) (Mul a c)))))
        
        (let t1 (Mul (One) (Add (Zero) (One))))
        (let t2 (Add (Mul (One) (Zero)) (Mul (One) (One))))
        
        (run 1)
        (check (= t1 t2))
    "#;

    let result = egraph.parse_and_run_program(None, program);
    assert!(
        result.is_ok(),
        "Rule with complex pattern should validate: {:?}",
        result
    );
}

#[test]
fn test_transitivity_chain() {
    // Test that transitivity proofs are validated correctly
    let mut egraph = EGraph::with_proof_checking();

    let program = r#"
        (datatype N (Num i64))
        
        (let a (Num 1))
        (let b (Num 2))
        (let c (Num 3))
        (let d (Num 4))
        
        ; Create a chain: a = b = c = d
        (union a b)
        (union b c)
        (union c d)
        
        ; Check transitivity: a = d
        (check (= a d))
    "#;

    let result = egraph.parse_and_run_program(None, program);
    assert!(
        result.is_ok(),
        "Transitivity chain should validate: {:?}",
        result
    );
}
