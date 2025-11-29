use egglog::ProofStore;
use egglog::prelude::*;

#[test]
fn test_prove_match() {
    let mut egraph = EGraph::with_proofs();

    // Build a simple program with addition
    egraph
        .parse_and_run_program(
            Some("test".to_string()),
            r#"
        (datatype Math
            (Num i64)
            (Add Math Math))
        
        (let a (Num 1))
        (let b (Num 2))
        (let c (Add a b))
        
        ;; Commutative rule
        (rule ((= sum (Add x y)))
              ((union sum (Add y x))))
              
        (run 5)
    "#,
        )
        .unwrap();

    // Query for Add expressions
    let matches = egraph.get_matches(facts![(= lhs (Add x y))]).unwrap();

    assert!(!matches.is_empty(), "Should find at least one match");

    // Create a proof store for explanations
    let mut store = ProofStore::default();

    // For each match, we should be able to get a proof for why lhs exists
    for (i, m) in matches.iter().enumerate() {
        let lhs_val = m.get("lhs").expect("lhs should be in match");
        let x_val = m.get("x").expect("x should be in match");
        let y_val = m.get("y").expect("y should be in match");

        // Get proof for why lhs exists (proves the match)
        let lhs_proof = egraph.explain_term(lhs_val.clone(), &mut store).unwrap();
        println!(
            "Match {}: lhs={:?}, x={:?}, y={:?}",
            i, lhs_val, x_val, y_val
        );
        println!("Proof for lhs:");
        store
            .print_term_proof(lhs_proof, &mut std::io::stdout())
            .unwrap();

        // We can also prove why x and y exist
        let _x_proof = egraph.explain_term(x_val.clone(), &mut store).unwrap();
        let _y_proof = egraph.explain_term(y_val.clone(), &mut store).unwrap();
    }
}

#[test]
fn test_match_with_nested_constructors() {
    let mut egraph = EGraph::with_proofs();

    egraph
        .parse_and_run_program(
            Some("test".to_string()),
            r#"
        (datatype Math
            (Num i64)
            (Add Math Math)
            (Mul Math Math))
        
        ;; Create some nested expressions
        (let a (Num 2))
        (let b (Num 3))
        (let c (Add a b))
        (let d (Mul c a))
        
        ;; Rule for distributivity
        (rule ((= result (Mul (Add x y) z)))
              ((union result (Add (Mul x z) (Mul y z)))))
              
        (run 5)
    "#,
        )
        .unwrap();

    // Query for multiplication of addition
    let matches = egraph
        .get_matches(facts![
            (= result (Mul (Add x y) z))
        ])
        .unwrap();

    assert!(!matches.is_empty(), "Should find the nested pattern");

    // Create a proof store for explanations
    let mut store = ProofStore::default();

    for m in &matches {
        let result_val = m.get("result").unwrap();
        let x_val = m.get("x").unwrap();
        let y_val = m.get("y").unwrap();
        let z_val = m.get("z").unwrap();

        println!(
            "Found match: result={:?}, x={:?}, y={:?}, z={:?}",
            result_val, x_val, y_val, z_val
        );

        // Prove why each component exists
        let result_proof = egraph.explain_term(result_val.clone(), &mut store).unwrap();
        println!("Proof for result:");
        store
            .print_term_proof(result_proof, &mut std::io::stdout())
            .unwrap();

        // The distributivity rule should have fired, creating an equivalent form
        // We can check if there's another term equal to result
        // This demonstrates how the match helps us understand rule applications
    }
}

#[test]
fn test_query_match_commutativity() {
    let mut egraph = EGraph::with_proofs();

    egraph
        .parse_and_run_program(
            Some("test".to_string()),
            r#"
        (datatype Math
            (Num i64)
            (Add Math Math))
        
        (let n1 (Num 1))
        (let n2 (Num 2))
        (let sum1 (Add n1 n2))
        (let sum2 (Add n2 n1))
        
        ;; Commutative rule that will union sum1 and sum2
        (rule ((= s (Add a b)))
              ((union s (Add b a))))
              
        (run 5)
    "#,
        )
        .unwrap();

    // Query for all Add expressions
    let matches = egraph
        .get_matches(facts![
            (= s (Add x y))
        ])
        .unwrap();

    // We should find the Add expressions
    assert!(!matches.is_empty(), "Should find matches");
    println!("Found {} matches", matches.len());

    // Create a proof store for explanations
    let mut store = ProofStore::default();

    // Demonstrate that sum1 and sum2 are equal due to the commutative rule
    if matches.len() >= 2 {
        let s1 = matches[0].get("s").unwrap();
        let s2 = matches[1].get("s").unwrap();

        println!("Match 0: s={:?}", s1);
        println!("Match 1: s={:?}", s2);

        // If they're equal, prove it
        if s1 == s2 {
            println!("The two matches refer to the same value (already unified)");
        } else if let Ok(proof) = egraph.explain_terms_equal(s1.clone(), s2.clone(), &mut store) {
            println!("Proved {:?} = {:?}:", s1, s2);
            store.print_eq_proof(proof, &mut std::io::stdout()).unwrap();
        } else {
            println!("The two matches are not provably equal");
        }
    }
}
