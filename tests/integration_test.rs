use egglog::{ast::Expr, EGraph, ExtractReport, Function, SerializeConfig, Term, Value};
use symbol_table::GlobalSymbol;

#[test]
fn test_subsumed_unextractable_action_extract() {
    // Test when an expression is subsumed, it isn't extracted, even if its the cheapest
    let mut egraph = EGraph::default();

    egraph
        .parse_and_run_program(
            None,
            r#"
            (datatype Math)
            (function exp () Math :cost 100)
            (function cheap () Math :cost 1)
            (union (exp) (cheap))
            (query-extract (exp))
            "#,
        )
        .unwrap();
    // Originally should give back numeric term
    assert!(matches!(
        egraph.get_extract_report(),
        Some(ExtractReport::Best {
            term: Term::App(s, ..),
            ..
        }) if s == &GlobalSymbol::from("cheap")
    ));
    // Then if we make one as subsumed, it should give back the variable term
    egraph
        .parse_and_run_program(
            None,
            r#"
            (subsume (cheap))
            (query-extract (exp))
            "#,
        )
        .unwrap();
    assert!(matches!(
        egraph.get_extract_report(),
        Some(ExtractReport::Best {
            term: Term::App(s, ..),
            ..
        }) if s == &GlobalSymbol::from("exp")
    ));
}

fn get_function(egraph: &EGraph, name: &str) -> Function {
    egraph
        .functions
        .get(&GlobalSymbol::from(name))
        .unwrap()
        .clone()
}
fn get_value(egraph: &EGraph, name: &str) -> Value {
    get_function(egraph, name).get(&[]).unwrap()
}

#[test]
fn test_subsumed_unextractable_rebuild_arg() {
    // Tests that a term stays unextractable even after a rebuild after a union would change the value of one of its args
    let mut egraph = EGraph::default();

    egraph
        .parse_and_run_program(
            None,
            r#"
            (datatype Math)
            (function container (Math) Math)
            (function exp () Math :cost 100)
            (function cheap () Math)
            (function cheap-1 () Math)
            ; we make the container cheap so that it will be extracted if possible, but then we mark it as subsumed
            ; so the (exp) expr should be extracted instead
            (let res (container (cheap)))
            (union res (exp))
            (cheap)
            (cheap-1)
            (subsume (container (cheap)))
            "#,
        ).unwrap();
    // At this point (cheap) and (cheap-1) should have different values, because they aren't unioned
    let orig_cheap_value = get_value(&egraph, "cheap");
    let orig_cheap_1_value = get_value(&egraph, "cheap-1");
    assert_ne!(orig_cheap_value, orig_cheap_1_value);
    // Then we can union them
    egraph
        .parse_and_run_program(
            None,
            r#"
            (union (cheap-1) (cheap))
            "#,
        )
        .unwrap();
    egraph.rebuild_nofail();
    // And verify that their values are now the same and different from the original (cheap) value.
    let new_cheap_value = get_value(&egraph, "cheap");
    let new_cheap_1_value = get_value(&egraph, "cheap-1");
    assert_eq!(new_cheap_value, new_cheap_1_value);
    assert_ne!(new_cheap_value, orig_cheap_value);
    // Now verify that if we extract, it still respects the unextractable, even though it's a different values now
    egraph
        .parse_and_run_program(
            None,
            r#"
            (query-extract res)
            "#,
        )
        .unwrap();
    let report = egraph.get_extract_report().clone().unwrap();
    let ExtractReport::Best { term, termdag, .. } = report else {
        panic!();
    };
    let expr = termdag.term_to_expr(&term);
    assert_eq!(expr, Expr::call_no_span(GlobalSymbol::from("exp"), vec![]));
}

#[test]
fn test_subsumed_unextractable_rebuild_self() {
    // Tests that a term stays unextractable even after a rebuild after a union change its output value.
    let mut egraph = EGraph::default();

    egraph
        .parse_and_run_program(
            None,
            r#"
            (datatype Math)
            (function container (Math) Math)
            (function exp () Math :cost 100)
            (function cheap () Math)
            (let x (cheap))
            (subsume (cheap))
            "#,
        )
        .unwrap();

    let orig_cheap_value = get_value(&egraph, "cheap");
    // Then we can union them
    egraph
        .parse_and_run_program(
            None,
            r#"
            (union (exp) x)
            "#,
        )
        .unwrap();
    egraph.rebuild_nofail();
    // And verify that the cheap value is now different
    let new_cheap_value = get_value(&egraph, "cheap");
    assert_ne!(new_cheap_value, orig_cheap_value);

    // Now verify that if we extract, it still respects the subsumption, even though it's a different values now
    egraph
        .parse_and_run_program(
            None,
            r#"
            (query-extract x)
            "#,
        )
        .unwrap();
    let report = egraph.get_extract_report().clone().unwrap();
    let ExtractReport::Best { term, termdag, .. } = report else {
        panic!();
    };
    let expr = termdag.term_to_expr(&term);
    assert_eq!(expr, Expr::call_no_span(GlobalSymbol::from("exp"), vec![]));
}

#[test]
fn test_subsume_unextractable_insert_and_merge() {
    // Example adapted from https://github.com/egraphs-good/egglog/pull/301#pullrequestreview-1756826062
    let mut egraph = EGraph::default();

    egraph
        .parse_and_run_program(
            None,
            r#"
            (datatype Expr
                (f Expr)
                (Num i64))
            (function exp () Expr :cost 100)

              (f (Num 1))
              (subsume (f (Num 1)))
              (f (Num 2))

              (union (Num 2) (Num 1))
              (union (f (Num 2)) (exp))
              (extract (f (Num 2)))
            "#,
        )
        .unwrap();
    assert!(matches!(
        egraph.get_extract_report(),
        Some(ExtractReport::Best {
            term: Term::App(s, ..),
            ..
        }) if s == &GlobalSymbol::from("exp")
    ));
}

#[test]
fn test_subsume_unextractable_action_extract_multiple() {
    // Test when an expression is set as subsumed, it isn't extracted, like with
    // extract multiple
    let mut egraph = EGraph::default();

    egraph
        .parse_and_run_program(
            None,
            "
            (datatype Math (Num i64))
            (Num 1)
            (union (Num 1) (Num 2))
            (query-extract :variants 2 (Num 1))
            ",
        )
        .unwrap();
    // Originally should give back two terms when extracted
    let report = egraph.get_extract_report();
    assert!(matches!(
        report,
        Some(ExtractReport::Variants { terms, .. }) if terms.len() == 2
    ));
    // Then if we make one unextractable, it should only give back one term
    egraph
        .parse_and_run_program(
            None,
            "
            (subsume (Num 2))
            (query-extract :variants 2 (Num 1))
            ",
        )
        .unwrap();
    let report = egraph.get_extract_report();
    assert!(matches!(
        report,
        Some(ExtractReport::Variants { terms, .. }) if terms.len() == 1
    ));
}

#[test]
fn test_rewrite_subsumed_unextractable() {
    // When a rewrite is marked as a subsumed, the lhs should not be extracted

    let mut egraph = EGraph::default();

    egraph
        .parse_and_run_program(
            None,
            r#"
            (datatype Math)
            (function exp () Math :cost 100)
            (function cheap () Math :cost 1)
            (rewrite (cheap) (exp) :subsume)
            (cheap)
            (run 1)
            (extract (cheap))
            "#,
        )
        .unwrap();
    // Should give back expenive term, because cheap is unextractable
    assert!(matches!(
        egraph.get_extract_report(),
        Some(ExtractReport::Best {
            term: Term::App(s, ..),
            ..
        }) if s == &GlobalSymbol::from("exp")
    ));
}
#[test]
fn test_rewrite_subsumed() {
    // When a rewrite is marked as a subsumed, the lhs should not be extracted

    let mut egraph = EGraph::default();

    // If we rerite most-exp to another term, that rewrite shouldnt trigger since its been subsumed.
    egraph
        .parse_and_run_program(
            None,
            r#"
            (datatype Math)
            (function exp () Math :cost 100)
            (function most-exp () Math :cost 1000)
            (rewrite (most-exp) (exp) :subsume)
            (most-exp)
            (run 1)
            (function cheap () Math :cost 1)
            (rewrite (most-exp) (cheap))
            (run 1)
            (extract (most-exp))
            "#,
        )
        .unwrap();
    assert!(matches!(
        egraph.get_extract_report(),
        Some(ExtractReport::Best {
            term: Term::App(s, ..),
            ..
        }) if s == &GlobalSymbol::from("exp")
    ));
}

#[test]
fn test_subsume() {
    // Test that if we mark a term as subsumed than no rewrites will be applied to it.
    // We can test this by adding a commutative additon property, and verifying it isn't applied on one of the terms
    // but is on the other
    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(
            None,
            r#"
        (datatype Math
          (Add Math Math)
          (Num i64))

        (rewrite (Add a b) (Add b a))
        (let x (Add (Num 1) (Num 2)))
        (let y (Add (Num 3) (Num 4)))
        (subsume (Add (Num 1) (Num 2)))
        (run 1)
        (extract y 10)
        "#,
        )
        .unwrap();
    assert!(matches!(
        egraph.get_extract_report(),
        Some(ExtractReport::Variants { terms, .. }) if terms.len() == 2
    ));

    egraph
        .parse_and_run_program(
            None,
            r#"
        ;; add something equal to x that can be extracted:
        (function otherConst () Math)
        (let other (otherConst))
        (union x other)
        (extract x 10)
        "#,
        )
        .unwrap();
    assert!(matches!(
        egraph.get_extract_report(),
        Some(ExtractReport::Variants { terms, .. }) if terms.len() == 1
    ));
}

#[test]
fn test_subsume_primitive() {
    // Test that we can subsume a primitive

    let mut egraph = EGraph::default();
    let res = egraph.parse_and_run_program(
        None,
        r#"
        (function one () i64)
        (set (one) 1)
        (subsume (one))
        "#,
    );
    assert!(res.is_ok());
}

#[test]
fn test_cant_subsume_merge() {
    // Test that we can't subsume something with a merge function

    let mut egraph = EGraph::default();
    let res = egraph.parse_and_run_program(
        None,
        r#"
        (function one () i64 :merge old)
        (set (one) 1)
        (subsume (one))
        "#,
    );
    assert!(res.is_err());
}

#[test]
fn test_value_to_classid() {
    let mut egraph = EGraph::default();

    egraph
        .parse_and_run_program(
            None,
            r#"
            (datatype Math)
            (function exp () Math )
            (exp)
            (query-extract (exp))
            "#,
        )
        .unwrap();
    let report = egraph.get_extract_report().clone().unwrap();
    let ExtractReport::Best { term, termdag, .. } = report else {
        panic!();
    };
    let expr = termdag.term_to_expr(&term);
    let value = egraph.eval_expr(&expr).unwrap().1;

    let serialized = egraph.serialize(SerializeConfig::default());
    let class_id = egraph.value_to_class_id(&value);
    assert!(serialized.class_data.get(&class_id).is_some());
    assert_eq!(value, egraph.class_id_to_value(&class_id));
}
