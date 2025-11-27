use egglog::prelude::*;
use egglog_bridge::ProofStore;

#[test]
fn test_prove_query_basic() {
    let mut egraph = EGraph::with_proofs();
    let mut store = ProofStore::default();

    // Define a datatype
    let program = r#"
        (datatype Math
            (Num i64)
            (Add Math Math))
        
        ; Add some facts
        (let x (Num 1))
        (let y (Num 2))
        (let z (Add x y))
    "#;

    egraph.parse_and_run_program(None, program).unwrap();

    // Query for something with a variable
    let query_result = egraph.prove_query(facts!((= result (Add (Num 1) (Num 2)))), &mut store);

    // Should succeed and return a proof
    assert!(
        query_result.is_ok(),
        "prove_query should succeed for a fact in the database"
    );
    let proof = query_result.unwrap();
    assert!(proof.is_some(), "Should find a proof for the query");
}

#[test]
fn test_prove_query_with_variables() {
    let mut egraph = EGraph::with_proofs();
    let mut store = ProofStore::default();

    // Define a datatype and rules
    let program = r#"
        (datatype Math
            (Num i64)
            (Add Math Math))
        
        ; Add facts
        (let a (Num 1))
        (let b (Num 2)) 
        (let c (Add a b))
        
        ; Add a rule that fires
        (rule ((= (Add x y) z))
              ((let sum (Add y x))))
        (run 1)
    "#;

    egraph.parse_and_run_program(None, program).unwrap();

    // Query with variables
    let query_result = egraph.prove_query(facts!((Add x (Num 2))), &mut store);

    // Should find matches and generate proof
    assert!(query_result.is_ok(), "prove_query should succeed");
    let proof = query_result.unwrap();
    assert!(
        proof.is_some(),
        "Should find a proof for the variable query"
    );
}

#[test]
fn test_prove_query_no_match() {
    let mut egraph = EGraph::with_proofs();
    let mut store = ProofStore::default();

    // Define a datatype
    let program = r#"
        (datatype Math
            (Num i64)
            (Add Math Math))
        
        ; Add some facts
        (let x (Num 1))
        (let y (Num 2))
    "#;

    egraph.parse_and_run_program(None, program).unwrap();

    // Query for something that doesn't exist
    let query_result = egraph.prove_query(facts!((= notfound (Add (Num 5) (Num 6)))), &mut store);

    // Should succeed but return None (no proof)
    assert!(query_result.is_ok(), "prove_query should not error");
    let proof = query_result.unwrap();
    assert!(
        proof.is_none(),
        "Should not find a proof for a non-existent fact"
    );
}

#[test]
fn test_prove_query_through_command() {
    let mut egraph = EGraph::with_proofs();

    // Set up some facts
    let program = r#"
        (datatype Math
            (Num i64)
            (Add Math Math))
        
        (let x (Num 1))
        (let y (Num 2))
        (let z (Add x y))
    "#;

    egraph.parse_and_run_program(None, program).unwrap();

    // Run prove-query command with a variable query
    let prove_cmd = r#"(prove-query (= result (Add (Num 1) (Num 2))))"#;
    let result = egraph.parse_and_run_program(None, prove_cmd);

    // Should succeed
    assert!(
        result.is_ok(),
        "prove-query command should execute successfully: {:?}",
        result
    );
}

#[test]
fn test_prove_query_ground_term() {
    let mut egraph = EGraph::with_proofs();
    let mut store = ProofStore::default();

    // Define a datatype
    let program = r#"
        (datatype Math
            (Num i64)
            (Add Math Math))
        
        ; Add some facts
        (let x (Num 1))
        (let y (Num 2))
        (let z (Add x y))
    "#;

    egraph.parse_and_run_program(None, program).unwrap();

    // Query for ground term without variables - this returns None
    let query_result = egraph.prove_query(facts!((Add (Num 1) (Num 2))), &mut store);

    // Should succeed but return None (no variables to prove)
    assert!(query_result.is_ok(), "prove_query should not error");
    let proof = query_result.unwrap();
    assert!(
        proof.is_none(),
        "Ground term queries (no variables) return None"
    );
}
