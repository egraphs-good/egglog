use egglog::prelude::*;
use egglog::EGraph;

#[test]
fn test_extraction_same_with_proof_mode() {
    // Test that extraction in normal mode produces the same result as extraction in proof mode
    let _ = env_logger::builder().is_test(true).try_init();

    let program = r#"
        (datatype Math
            (Num i64)
            (Add Math Math)
            (Mul Math Math))

        (rewrite (Add (Num a) (Num b)) (Num (+ a b)))
        (rewrite (Mul (Num a) (Num b)) (Num (* a b)))

        ; commutativity
        (rewrite (Add x y) (Add y x))
        (rewrite (Mul x y) (Mul y x))

        ; associativity
        (rewrite (Add (Add x y) z) (Add x (Add y z)))
        (rewrite (Mul (Mul x y) z) (Mul x (Mul y z)))

        ; distributivity
        (rewrite (Mul x (Add y z)) (Add (Mul x y) (Mul x z)))

        (let expr (Mul (Add (Num 1) (Num 2)) (Num 3)))
        (run 10)
    "#;

    // Run in normal mode and extract
    let mut egraph_normal = EGraph::default();
    egraph_normal
        .parse_and_run_program(None, program)
        .unwrap();
    let normal_output = egraph_normal
        .parse_and_run_program(None, "(extract expr)")
        .unwrap();
    let normal_extracted = normal_output[0].to_string();

    // Run in proof mode and extract
    let mut egraph_proofs = EGraph::new_with_proofs();
    egraph_proofs
        .parse_and_run_program(None, program)
        .unwrap();
    let proofs_output = egraph_proofs
        .parse_and_run_program(None, "(extract expr)")
        .unwrap();
    let proofs_extracted = proofs_output[0].to_string();

    // They should produce the same extraction result
    assert_eq!(
        normal_extracted, proofs_extracted,
        "Extraction differs between normal mode and proof mode:\nNormal: {}\nProofs: {}",
        normal_extracted, proofs_extracted
    );

    // The result should be (Num 9) since (1+2)*3 = 9
    assert!(
        normal_extracted.contains("Num") && normal_extracted.contains("9"),
        "Expected (Num 9), got: {}",
        normal_extracted
    );
}

#[test]
fn test_extraction_same_with_proof_mode_using_rule_macro() {
    // Test using the rule! macro from prelude
    let _ = env_logger::builder().is_test(true).try_init();

    // Setup program with datatypes
    let setup = r#"
        (datatype Expr
            (Var String)
            (Lit i64)
            (Add Expr Expr))

        (let x (Add (Lit 1) (Lit 2)))
        (let y (Add (Lit 2) (Lit 1)))
    "#;

    // Run in normal mode
    let mut egraph_normal = EGraph::default();
    egraph_normal.parse_and_run_program(None, setup).unwrap();

    // Add a commutativity rule using the rule! macro
    add_ruleset(&mut egraph_normal, "my_rules").unwrap();
    rule(
        &mut egraph_normal,
        "my_rules",
        facts![(= (Add a b) e)],
        actions![(union e (Add b a))],
    )
    .unwrap();

    // Run the rules
    for _ in 0..5 {
        run_ruleset(&mut egraph_normal, "my_rules").unwrap();
    }

    // Extract
    let normal_output = egraph_normal
        .parse_and_run_program(None, "(extract x)")
        .unwrap();
    let normal_extracted = normal_output[0].to_string();

    // Run in proof mode
    let mut egraph_proofs = EGraph::new_with_proofs();
    egraph_proofs.parse_and_run_program(None, setup).unwrap();

    // Add the same rule using the rule! macro
    add_ruleset(&mut egraph_proofs, "my_rules").unwrap();
    rule(
        &mut egraph_proofs,
        "my_rules",
        facts![(= (Add a b) e)],
        actions![(union e (Add b a))],
    )
    .unwrap();

    // Run the rules
    for _ in 0..5 {
        run_ruleset(&mut egraph_proofs, "my_rules").unwrap();
    }

    // Extract
    let proofs_output = egraph_proofs
        .parse_and_run_program(None, "(extract x)")
        .unwrap();
    let proofs_extracted = proofs_output[0].to_string();

    // They should produce the same extraction result
    assert_eq!(
        normal_extracted, proofs_extracted,
        "Extraction differs between normal mode and proof mode:\nNormal: {}\nProofs: {}",
        normal_extracted, proofs_extracted
    );
}
