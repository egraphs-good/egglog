use egglog::*;

#[test]
fn test_include_with_fresh() {
    // Test that fresh! works with include statements
    let mut egraph = EGraph::default();

    let result = egraph.parse_and_run_program(None, r#"(include "tests/include_fresh.egg")"#);
    assert!(result.is_ok());
}

#[test]
fn test_fresh_inline_works() {
    // Test that the same fresh! code works when not included
    let mut egraph = EGraph::default();

    let result = egraph.parse_and_run_program(
        None,
        r#"
        (datatype TestSort 
          (Foo i64)
          (Bar TestSort TestSort))
        
        (let init (Foo 1))
        
        (rule ((Foo x))
              ((Foo (+ x 1))
               (Bar (unstable-fresh! TestSort) (Foo x))))
        
        (run 2)
        (check (Foo 1))
        (check (Foo 2))
        "#,
    );

    assert!(result.is_ok());
}

#[test]
fn test_include_fresh_sugar_file() {
    // Test that including the fresh_sugar.egg test file works
    let mut egraph = EGraph::default();

    let result = egraph.parse_and_run_program(None, r#"(include "tests/fresh_sugar.egg")"#);
    assert!(result.is_ok());
}

#[test]
fn test_include_fresh_with_queries() {
    // Test that we can query results after including a file with fresh!
    let mut egraph = EGraph::default();

    let result = egraph.parse_and_run_program(None, r#"(include "tests/include_fresh.egg")"#);
    assert!(result.is_ok());

    // Should be able to query the results
    let check_result = egraph.parse_and_run_program(None, "(check (Foo 1))");
    assert!(check_result.is_ok());
}
