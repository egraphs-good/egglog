use egglog::*;

#[test]
fn test_simple_extract1() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    let _ = egraph
        .parse_and_run_program(
            None,
            r#"
             (datatype Op (Add i64 i64))
             (let expr (Add 1 1))
             (extract expr)"#,
        )
        .unwrap();

    let report = egraph.get_extract_report().clone().unwrap();
    let ExtractReport::Best { cost, .. } = report else {
        panic!();
    };
    assert_eq!(cost, 3);
}

#[test]
fn test_simple_extract2() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    egraph
        .parse_and_run_program(
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
             "#,
        )
        .unwrap();

    let report = egraph.get_extract_report().clone().unwrap();
    let ExtractReport::Best { cost, .. } = report else {
        panic!();
    };
    assert_eq!(cost, 4);
}

#[test]
fn test_simple_extract3() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    egraph
        .parse_and_run_program(
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
             "#,
        )
        .unwrap();

    let report = egraph.get_extract_report().clone().unwrap();
    let ExtractReport::Best { cost, .. } = report else {
        panic!();
    };
    assert_eq!(cost, 2);
}

#[test]
fn test_simple_extract4() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    egraph
        .parse_and_run_program(
            None,
            r#"
        (datatype Foo
            (Foobar)
        )
        (subsume (Foobar))
        (datatype Bar
            (Barbar Foo)
        )
        (let x (Barbar (Foobar)))
        (extract x)
        "#,
        )
        .unwrap_err();
}

#[test]
fn test_simple_extract5() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    egraph
        .parse_and_run_program(
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
             "#,
        )
        .unwrap();

    let report = egraph.get_extract_report().clone().unwrap();
    let ExtractReport::Best { cost, .. } = report else {
        panic!();
    };
    assert_eq!(cost, 0);
}

#[test]
fn test_simple_extract6() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    egraph
        .parse_and_run_program(
            None,
            r#"
             (datatype False)
             (sort QuaVec (Vec i64))
             (sort QuaMap (Map QuaVec False))
             (function Quaz () QuaMap :no-merge)
             (set (Quaz) (map-empty))
             (extract (Quaz))
             "#,
        )
        .unwrap();

    let report = egraph.get_extract_report().clone().unwrap();
    let ExtractReport::Best { cost, .. } = report else {
        panic!();
    };
    assert_eq!(cost, 0);
}

#[test]
fn test_simple_extract7() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    egraph
        .parse_and_run_program(
            None,
            r#"
            (datatype Foo
                (bar)
                (baz)
            )
            (sort Mapsrt1 (Map i64 Foo))
            (let map1 (map-insert (map-empty) 0 (bar)))
            
            (sort Mapsrt2 (Map bool Foo))
            (let map2 (map-insert (map-empty) false (baz)))
            ;(let map2b (map-insert (map-empty) false (bar)))
            ;(union map2 map2b)

            ;(extract map1)
            ;(extract map2)
            
            ;(function toerr (Mapsrt2) Foo :no-merge)

            ;(set (toerr map2) (bar))

            (union (bar) (baz))
            ; Also unions map1 and map2!?

            (extract map1)
            (extract map2)

            ;(extract (toerr map2))

             "#,
        )
        .unwrap();

    let report = egraph.get_extract_report().clone().unwrap();
    let ExtractReport::Best { cost, .. } = report else {
        panic!();
    };
    assert_eq!(cost, 2);
}

#[test]
fn test_simple_extract8() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    egraph
        .parse_and_run_program(
            None,
            r#"
            (datatype Foo
                (bar :cost 10)
            )
            (function func () Foo :no-merge)
            (set (func) (bar))

            (extract (bar))
            "#,
        )
        .unwrap();

    let report = egraph.get_extract_report().clone().unwrap();
    let ExtractReport::Best { cost, .. } = report else {
        panic!();
    };
    assert_eq!(cost, 10);
}

#[test]
fn test_simple_extract9() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    egraph
        .parse_and_run_program(
            None,
            r#"
            (datatype Foo)
            (datatype NodeA)
            (datatype NodeB)
            (datatype NodeC)
            (constructor ctoa (NodeC) NodeA)
            (constructor atob (NodeA) NodeB)
            (constructor btoc (NodeB) NodeC)

            (constructor bar () Foo :cost 9223372036854775807)
            (constructor barbar (Foo Foo) Foo :cost 2)

            (constructor groundedA (Foo) NodeA)
            (let a (groundedA (barbar (bar) (bar))))
            (let b (atob a))
            (let c (btoc b))
            (let a2 (ctoa c))
            (let b2 (atob a2))
            (let c2 (btoc b2))
            (union a a2)
            (union b b2)
            (union c c2)

            (extract a)
            "#,
        )
        .unwrap();

    let report = egraph.get_extract_report().clone().unwrap();
    let ExtractReport::Best { cost, .. } = report else {
        panic!();
    };
    assert_eq!(cost, usize::MAX);

    egraph
        .parse_and_run_program(
            None,
            r#"
            (extract b)
            "#,
        )
        .unwrap();

    let report = egraph.get_extract_report().clone().unwrap();
    let ExtractReport::Best { cost, .. } = report else {
        panic!();
    };
    assert_eq!(cost, usize::MAX);

    egraph
        .parse_and_run_program(
            None,
            r#"
            (extract c)
            "#,
        )
        .unwrap();

    let report = egraph.get_extract_report().clone().unwrap();
    let ExtractReport::Best { cost, .. } = report else {
        panic!();
    };
    assert_eq!(cost, usize::MAX);
}

#[test]
fn test_extract_variants1() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    egraph
        .parse_and_run_program(
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
             (extract tb 3)
             "#,
        )
        .unwrap();

    let report = egraph.get_extract_report().clone().unwrap();
    let ExtractReport::Variants { terms, .. } = report else {
        panic!();
    };
    assert_eq!(terms.len(), 2);
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
            (extract (exp))
            "#,
        )
        .unwrap();
    // Originally should give back numeric term
    assert!(matches!(
        egraph.get_extract_report(),
        Some(ExtractReport::Best {
            term: Term::App(s, ..),
            ..
        }) if s == "cheap"
    ));
    // Then if we make one as subsumed, it should give back the variable term
    egraph
        .parse_and_run_program(
            None,
            r#"
            (subsume (cheap))
            (extract (exp))
            "#,
        )
        .unwrap();
    assert!(matches!(
        egraph.get_extract_report(),
        Some(ExtractReport::Best {
            term: Term::App(s, ..),
            ..
        }) if s == "exp"
    ));
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
        }) if s == "exp"
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
            (extract (Num 1) 2)
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
            (extract (Num 1) 2)
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
        }) if s == "exp"
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
        }) if s == "exp"
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
            (extract (exp))
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
    let class_id = egraph.value_to_class_id(&sort, value);
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
            name: "a".into(),
            offset: 0,
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
