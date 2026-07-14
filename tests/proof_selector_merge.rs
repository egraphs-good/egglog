use egglog::EGraph;

#[test]
fn selector_merge_cleanup_uses_existing_current_proof() {
    let mut candidates = String::new();
    for i in 0..200 {
        candidates.push_str(&format!("(Candidate (K) {i})\n"));
    }

    let program = format!(
        r#"
        (sort Key)
        (constructor K () Key)
        (function Best (Key) i64 :merge old)
        (relation Candidate (Key i64))
        (rule ((Candidate k v))
              ((set (Best k) v)))
        {candidates}
        (run 1)
        (check (= (Best (K)) 0))
        "#
    );

    let mut egraph = EGraph::new_with_proofs().with_proof_testing();
    egraph.parse_and_run_program(None, &program).unwrap();

    let merge_rule_matches: usize = egraph
        .get_overall_run_report()
        .num_matches_per_rule
        .iter()
        .filter(|(rule, _)| rule.as_ref().contains("merge_rule"))
        .map(|(_, matches)| *matches)
        .sum();
    assert_eq!(merge_rule_matches, 0);
}
