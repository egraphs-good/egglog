use std::io::stdout;

use egglog::ProofStore;
use egglog::prelude::*;

#[test]
fn proofs_from_egg_file() {
    let program = include_str!("simple_math.egg");
    let mut egraph = EGraph::with_proofs();

    egraph.parse_and_run_program(None, program);

    let mut store = ProofStore::default();

    // Evaluate a small term
    let (_, term_value) = egraph
        .eval_expr(&expr!((Add (Num 3) (Num 2))))
        .expect("evaluate expression");
    let term_pf = egraph
        .explain_term(term_value, &mut store)
        .expect("term proof");
    store
        .print_term_proof(term_pf, &mut stdout())
        .expect("print term proof");
}
