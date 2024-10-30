use egglog::EGraph;

#[test]
fn test_looking_up_global() {
    // Test whether looking up undefined value in a global action result in a runtime error
    let mut egraph = EGraph::default();

    let res = egraph
        .parse_and_run_program(
            None, 
            r#"
            (function g () i64)
            (fail (let y (g)))
            "#
        );

    assert!(res.is_err());
}