use egglog::ProofStore;
use egglog::prelude::*;

#[test]
fn proof_none_when_no_match() {
    let mut egraph = EGraph::with_proofs();
    egraph
        .parse_and_run_program(
            None,
            "
            (datatype Math
                (Num i64)
                (Add Math Math))
            ",
        )
        .unwrap();

    let mut store = ProofStore::default();
    let proof = egraph.get_proof(facts![(Add x y)], &mut store).unwrap();

    assert!(proof.is_none());
}

#[test]
fn proof_for_single_match() {
    let mut egraph = EGraph::with_proofs();
    egraph
        .parse_and_run_program(
            None,
            "
            (datatype Math
                (Num i64)
                (Add Math Math))
            (let lhs (Add (Num 1) (Num 2)))
            ",
        )
        .unwrap();

    let mut store = ProofStore::default();
    let proof = egraph
        .get_proof(facts![(= lhs (Add x y))], &mut store)
        .unwrap()
        .expect("expected proof");

    store
        .print_term_proof(proof, &mut std::io::stdout())
        .unwrap();
}
