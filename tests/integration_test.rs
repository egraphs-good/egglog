use egglog::{ast::Expr, *};
use symbol_table::GlobalSymbol;

#[test]
fn test_simple_extract1() {

    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    let result =
    egraph.parse_and_run_program(
        None,
        r#"
             (datatype Op (Add i64 i64))
             (let expr (Add 1 1))
             (extract expr)"#
        )
        .unwrap();

    log::debug!("{}", result.join("\n"));
    /* 
    let mut termdag = TermDag::default();
    let (sort, value) = egraph.eval_expr(&egglog::var!("expr")).unwrap();
    let (_, extracted) = egraph.extract(value, &mut termdag, &sort).unwrap();
    assert_eq!(termdag.to_string(&extracted), "(Add 1 1)");
    */
}

#[test]
fn test_simple_extract2() {

    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    egraph.parse_and_run_program(
        None,
        r#"
             (datatype Term
               (Origin :cost 0) 
               (BigStep Term :cost 10)
               (SmallStep Term :cost 1)
             )
             (let t (Origin))
             (let tb (BigStep t))
             (let tbs (SmallStep tb))
             (let ts (SmallStep t))
             (let tss (SmallStep ts))
             (let tsss (SmallStep tss))
             (union tbs tsss)
             (let tssss (SmallStep tsss))
             (union tssss tb)
             (extract tb)
             "#
        )
        .unwrap();

    /*
    let mut termdag = TermDag::default();
    let (sort, value) = egraph.eval_expr(&egglog::var!("tb")).unwrap();
    let (cost, extracted) = egraph.extract(value, &mut termdag, &sort).unwrap();
    assert_eq!(cost, 4);
    assert_eq!(termdag.to_string(&extracted), "(SmallStep (SmallStep (SmallStep (SmallStep (Origin)))))");
    */
}

#[test]
fn test_simple_extract3() {

    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    egraph.parse_and_run_program(
        None,
        r#"
             (datatype Fruit
               (Apple i64 :cost 1) 
               (Orange f64 :cost 2)
             )
             (datatype Vegetable
               (Broccoli bool :cost 3)
               (Carrot Fruit :cost 4)
             )
             (let a (Apple 5))
             (let o (Orange 3.14))
             (let b (Broccoli true))
             (let c (Carrot a))
             (extract a)
             "#
        )
        .unwrap();

}

/*
#[test]
fn test_simple_extract4() {

    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    egraph.parse_and_run_program(
        None,
        r#"
             (datatype Foo 
                (Foobar i64)
             ) 
             (let foobar (Foobar 42))
             (datatype Bar
                (Barfoo i64)
             )
             (let barfoo (Barfoo 24))
             (datatype Baz 
                (Bazfoo i64)
             )
             (sort QuaMap (Map Foo Bar))
             (sort QuaVecMap (Vec QuaMap))
             (sort QuaVVM (Vec QuaVecMap))
             (function Quaz () QuaVVM :no-merge)
             (set (Quaz) (vec-empty))
             (extract (Quaz))
             "#
        )
        .unwrap();

}
*/

#[test]
fn test_simple_extract5() {

    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    egraph.parse_and_run_program(
        None,
        r#"
             (datatype Foo 
                (Foobar i64)
             ) 
             (let foobar (Foobar 42))
             (datatype Bar
                (Barfoo i64)
             )
             (let barfoo (Barfoo 24))
             (sort QuaVec (Vec i64))
             (sort QuaMap (Map QuaVec Foo))
             (function Quaz () QuaMap :no-merge)
             (set (Quaz) (map-empty))
             (extract (Quaz))
             "#
        )
        .unwrap();

}

#[test]
fn test_simple_extract6() {

    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    egraph.parse_and_run_program(
        None,
        r#"
             (datatype False)
             (sort QuaVec (Vec i64))
             (sort QuaMap (Map QuaVec False))
             (function Quaz () QuaMap :no-merge)
             (set (Quaz) (map-empty))
             (extract (Quaz))
             "#
        )
        .unwrap();

}

#[test]
fn test_subsumed_unextractable_action_extract() {
    // Test when an expression is subsumed, it isn't extracted, even if its the cheapest
    let mut egraph = EGraph::default();

    egraph
        .parse_and_run_program(
            None,
            r#"
            (datatype Math)
            (constructor exp () Math :cost 100)
            (constructor cheap () Math :cost 1)
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
            (constructor container (Math) Math)
            (constructor exp () Math :cost 100)
            (constructor cheap () Math)
            (constructor cheap-1 () Math)
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
    let span = span!();
    let expr = termdag.term_to_expr(&term, span.clone());
    assert_eq!(expr, Expr::Call(span, GlobalSymbol::from("exp"), vec![]));
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
            (constructor container (Math) Math)
            (constructor exp () Math :cost 100)
            (constructor cheap () Math)
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
    let span = span!();
    let expr = termdag.term_to_expr(&term, span.clone());
    assert_eq!(expr, Expr::Call(span, GlobalSymbol::from("exp"), vec![]));
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
            (constructor exp () Expr :cost 100)

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
            (constructor exp () Math :cost 100)
            (constructor cheap () Math :cost 1)
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
            (constructor exp () Math :cost 100)
            (constructor most-exp () Math :cost 1000)
            (rewrite (most-exp) (exp) :subsume)
            (most-exp)
            (run 1)
            (constructor cheap () Math :cost 1)
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
        (constructor otherConst () Math)
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
fn test_subsume_custom() {
    // Test that we can't subsume  a custom function
    // Only relations and constructors are allowed to be subsumed

    let mut egraph = EGraph::default();
    let res = egraph.parse_and_run_program(
        None,
        r#"
        (function one () i64 :no-merge)
        (set (one) 1)
        (subsume (one))
        "#,
    );
    assert!(res.is_err());
}

#[test]
fn test_subsume_ok() {
    let mut egraph = EGraph::default();
    let res = egraph.parse_and_run_program(
        None,
        r#"
        (sort E)
        (constructor one () E)
        (constructor two () E)
        (one)
        (subsume (one))
        ;; subsuming a non-existent tuple
        (subsume (two))

        (relation R (i64))
        (R 1)
        (subsume (R 1))
        (subsume (R 2))
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
        (constructor one () i64 :merge old)
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
            (constructor exp () Math )
            (exp)
            (query-extract (exp))
            "#,
        )
        .unwrap();
    let report = egraph.get_extract_report().clone().unwrap();
    let ExtractReport::Best { term, termdag, .. } = report else {
        panic!();
    };
    let expr = termdag.term_to_expr(&term, span!());
    let (sort, value) = egraph.eval_expr(&expr).unwrap();

    let serialized = egraph.serialize(SerializeConfig::default());
    let class_id = egraph.value_to_class_id(&sort, &value);
    assert!(serialized.class_data.get(&class_id).is_some());
    assert_eq!(value, egraph.class_id_to_value(&class_id));
}

#[test]
fn test_serialize_subsume_status() {
    let mut egraph = EGraph::default();

    egraph
        .parse_and_run_program(
            None,
            r#"
            (datatype Math)
            (constructor a () Math )
            (constructor b () Math )
            (a)
            (b)
            (subsume (a))
            "#,
        )
        .unwrap();

    let serialized = egraph.serialize(SerializeConfig::default());
    let a_id = egraph.to_node_id(
        None,
        egglog::SerializedNode::Function {
            name: ("a").into(),
            offset: 1,
        },
    );
    let b_id = egraph.to_node_id(
        None,
        egglog::SerializedNode::Function {
            name: "b".into(),
            offset: 0,
        },
    );
    assert!(serialized.nodes[&a_id].subsumed);
    assert!(!serialized.nodes[&b_id].subsumed);
}

#[test]
fn test_shadowing_query() {
    let s = "(function f () i64 :no-merge) (set (f) 2) (check (= (f) f) (= f 2))";
    let e = EGraph::default()
        .parse_and_run_program(None, s)
        .unwrap_err();
    assert!(matches!(e, Error::Shadowing(_, _, _)));
}

#[test]
fn test_shadowing_push() {
    let s = "(push) (let x 1) (pop) (let x 1)";
    EGraph::default().parse_and_run_program(None, s).unwrap();
}
