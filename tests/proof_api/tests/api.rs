use egglog::prelude::*;
use proof_api::build_program;

#[test]
fn api_program_builds_without_proofs() {
    let mut egraph = build_program().expect("building program succeeds");

    let results = query(&mut egraph, vars![x: i64], facts![ (= x 0) ]).expect("query runs");

    // Ensure we can iterate over the results returned by the API.
    let _count = results.iter().count();
}
