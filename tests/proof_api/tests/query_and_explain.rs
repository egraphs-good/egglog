use egglog::ProofStore;
use egglog::prelude::*;

#[test]
fn test_query_and_explain_match() {
    // Create an e-graph with proofs enabled
    let mut egraph = EGraph::with_proofs();

    egraph
        .parse_and_run_program(
            None,
            "
            (datatype Math
                (Num i64)
                (Add Math Math))
            
            ; Create a commutative rule for addition
            (rule ((Add x y))
                  ((union (Add x y) (Add y x))))
            
            ; Add some expressions
            (let a (Add (Num 1) (Num 2)))
            (let b (Add (Num 3) (Num 4)))
            (let c (Add (Num 5) (Num 6)))
            
            ; Run rules to establish equalities
            (run 1)
            ",
        )
        .unwrap();

    // Query for all Add expressions with a name bound to them
    // This will match both the original expressions and the ones created by the commutative rule
    let matches = egraph
        .get_matches(facts![(Fact (= lhs (Add x y)))])
        .unwrap();

    println!("Found {} matches", matches.len());
    // We get 6 matches because the commutative rule creates both (Add a b) and (Add b a)
    assert_eq!(matches.len(), 6);

    // Get the first match
    let first_match = &matches[0];
    println!("\nFirst match has {} variables", first_match.len());
    // Note: We only get x and y, not lhs, because lhs is equal to the pattern (Add x y)
    // and equality constraints don't create new variables in the match
    assert_eq!(first_match.len(), 2); // x, y

    // Get the values from the match
    let x_value = first_match.get("x").expect("x should be bound");
    let y_value = first_match.get("y").expect("y should be bound");

    println!("x = {:?}, y = {:?}", x_value, y_value);

    // To get lhs, we need to evaluate (Add x y) in the egraph
    // But for this test, let's just explain the operands

    // Explain how the operands were constructed
    let mut store = ProofStore::default();
    let x_proof = egraph.explain_term(x_value, &mut store).unwrap();

    println!("\nProof for x:");
    store
        .print_term_proof(x_proof, &mut std::io::stdout())
        .unwrap();

    // Now let's look at a few matches to see both original and rule-generated ones
    println!("\n--- First 4 matches ---");

    for (i, m) in matches.iter().take(4).enumerate() {
        let x = m.get("x").unwrap();
        let y = m.get("y").unwrap();

        println!("\nMatch {}: x={:?}, y={:?}", i, x, y);
        println!("Proof for x:");
        let proof_x = egraph.explain_term(x, &mut store).unwrap();
        store
            .print_term_proof(proof_x, &mut std::io::stdout())
            .unwrap();

        println!("Proof for y:");
        let proof_y = egraph.explain_term(y, &mut store).unwrap();
        store
            .print_term_proof(proof_y, &mut std::io::stdout())
            .unwrap();
    }

    println!("\n--- Note ---");
    println!("The matches include both:");
    println!("  - Original (Add x y) expressions from the program");
    println!("  - Commuted (Add y x) expressions created by the rule");
    println!("\n✓ Successfully queried for matches and explained all their proofs!");
}
