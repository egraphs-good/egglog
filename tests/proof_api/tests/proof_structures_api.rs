use egglog::prelude::*;
use egglog::{CongProof, Premise, TermId, TermProof};

#[test]
fn test_proof_structures_accessible() {
    // Create an egraph with proofs enabled
    let mut egraph = EGraph::with_proofs();

    egraph
        .parse_and_run_program(
            None,
            r#"
            (datatype Math
                (Num i64)
                (Add Math Math))
            
            (let x (Num 1))
            (let y (Num 2))
            (let sum (Add x y))
            
            (rule ((= s (Add a b)))
                  ((union s (Add b a))))
            
            (run 1)
            "#,
        )
        .unwrap();

    // Create a ProofStore and query matches
    let mut store = ProofStore::default();
    let matches = egraph.get_matches(facts![(= s (Add a b))]).unwrap();
    assert!(!matches.is_empty());

    // Get proof for the first match
    let m = &matches[0];
    let s_val = m.get("s").unwrap();
    let proof_id = egraph.explain_term(s_val.clone(), &mut store).unwrap();

    // Access the proof structures directly
    let term_proof = store.term_proof(proof_id);
    assert!(term_proof.is_some(), "Should be able to access term proof");

    // Check that we can match on proof types
    if let Some(proof) = term_proof {
        match proof {
            TermProof::PRule {
                rule_name,
                subst,
                body_pfs,
                result,
            } => {
                println!("Rule proof: {}", rule_name);
                println!("Substitution has {} bindings", subst.len());
                println!("Body has {} premises", body_pfs.len());
                println!("Result: {:?}", result);
            }
            TermProof::PProj {
                pf_f_args_ok,
                arg_idx,
            } => {
                println!(
                    "Projection proof: arg {} of proof {:?}",
                    arg_idx, pf_f_args_ok
                );
            }
            TermProof::PCong(CongProof {
                pf_args_eq,
                pf_f_args_ok,
                ..
            }) => {
                println!("Congruence proof with {} arg equalities", pf_args_eq.len());
                println!("Function args proof: {:?}", pf_f_args_ok);
            }
            TermProof::PFiat { desc, term } => {
                println!("Fiat proof: {}", desc);
                println!("Term: {:?}", term);
            }
        }
    }

    // Access the termdag directly
    let _termdag = store.termdag();
    println!("TermDag is accessible");

    // Test that RuleVarBinding fields are accessible
    if let Some(TermProof::PRule { subst, .. }) = term_proof {
        for binding in subst {
            // Access public fields of RuleVarBinding
            let _name: &str = &binding.name;
            let _ty = &binding.ty;
            let _term = binding.term;
            println!("Binding: {} = {:?}", binding.name, binding.term);
        }
    }

    // Test that we can pattern match on Premise
    if let Some(TermProof::PRule { body_pfs, .. }) = term_proof {
        for premise in body_pfs {
            match premise {
                Premise::TermOk(term_proof_id) => {
                    println!("Premise: term ok {:?}", term_proof_id);
                }
                Premise::Eq(eq_proof_id) => {
                    println!("Premise: eq ok {:?}", eq_proof_id);
                }
            }
        }
    }
}

#[test]
fn test_equality_proof_structures() {
    let mut egraph = EGraph::with_proofs();

    egraph
        .parse_and_run_program(
            None,
            r#"
            (datatype Expr
                (Const i64)
                (Plus Expr Expr))
            
            ;; Create some expressions and equalities
            (let a (Const 1))
            (let b (Const 2))
            (let c (Plus a b))
            (let d (Plus b a))
            
            ;; Commutative rule
            (rule ((= sum (Plus x y)))
                  ((union sum (Plus y x))))
            
            (run 3)
            "#,
        )
        .unwrap();

    let mut store = ProofStore::default();

    // Query for Plus expressions
    let matches = egraph.get_matches(facts![(Plus x y)]).unwrap();

    if !matches.is_empty() {
        let m = &matches[0];
        let x_val = m.get("x").unwrap();
        let y_val = m.get("y").unwrap();

        // Get proofs for the operands to demonstrate API usage
        let x_proof_id = egraph.explain_term(x_val, &mut store).unwrap();
        let y_proof_id = egraph.explain_term(y_val, &mut store).unwrap();

        // Access the proofs to test the API
        let x_proof = store.term_proof(x_proof_id);
        let y_proof = store.term_proof(y_proof_id);

        assert!(x_proof.is_some(), "Should be able to access x proof");
        assert!(y_proof.is_some(), "Should be able to access y proof");

        // Test CongProof fields are accessible
        let cong_proof = CongProof {
            pf_args_eq: vec![],
            pf_f_args_ok: x_proof_id,
            old_term: TermId::from(0usize),
            new_term: TermId::from(1usize),
            func: "test".into(),
        };

        // Access public fields
        let _args_eq = &cong_proof.pf_args_eq;
        let _f_args_ok = cong_proof.pf_f_args_ok;
        let _old_term = cong_proof.old_term;
        let _new_term = cong_proof.new_term;
        let _func = &cong_proof.func;

        println!("CongProof fields are accessible");
        println!("Successfully tested proof structure API");
    }
}
