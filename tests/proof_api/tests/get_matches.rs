use egglog::prelude::*;

#[test]
fn test_get_matches_basic() {
    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(
            None,
            "
            (datatype Math
                (Num i64)
                (Add Math Math))
            
            (Add (Num 1) (Num 2))
            (Add (Num 3) (Num 4))
            (Add (Num 5) (Num 6))
            ",
        )
        .unwrap();

    // Query for all Add expressions
    let matches = egraph
        .get_matches(&[Fact::Fact(expr!((Add x y)))])
        .unwrap();

    // We should have 3 matches
    assert_eq!(matches.len(), 3);
    println!("Found {} matches", matches.len());
    for m in &matches {
        println!("  Match with {} bindings", m.bindings.len());
        for (k, v) in &m.bindings {
            println!("    {}: {:?}", k, v);
        }
        // We expect only x and y (internal variables are filtered out)
        assert_eq!(m.bindings.len(), 2);
        assert!(m.bindings.contains_key("x"));
        assert!(m.bindings.contains_key("y"));
    }
}

#[test]
fn test_get_matches_with_equality() {
    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(
            None,
            "
            (datatype Math
                (Num i64)
                (Add Math Math))
            
            (let a (Num 1))
            (let b (Num 2))
            (let sum (Add a b))
            (union a (Num 10))
            ",
        )
        .unwrap();

    // Query for Add expressions where first arg is (Num 1)
    let matches = egraph
        .get_matches(&[
            Fact::Fact(expr!((Add x y))),
            Fact::Eq(span!(), expr!(x), expr!((Num 1))),
        ])
        .unwrap();

    // Should find the Add expression
    assert!(!matches.is_empty());
    println!("Found {} matches with equality constraint", matches.len());
}

#[test]
fn test_get_matches_empty_result() {
    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(
            None,
            "
            (datatype Math
                (Num i64)
                (Add Math Math))
            
            (Num 1)
            (Num 2)
            ",
        )
        .unwrap();

    // Query for Add expressions (none exist)
    let matches = egraph
        .get_matches(&[Fact::Fact(expr!((Add x y)))])
        .unwrap();

    // Should have no matches
    assert_eq!(matches.len(), 0);
}
