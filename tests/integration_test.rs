use egglog::{extract::DefaultCost, *};
use egglog_ast::span::{RustSpan, Span};
use hashbrown::HashSet;

#[test]
fn globals_missing_prefix_warns_by_default() {
    testing_logger::setup();

    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(None, "(let value 41)")
        .unwrap();

    testing_logger::validate(|logs| {
        let bodies: Vec<_> = logs.iter().map(|entry| entry.body.clone()).collect();
        assert!(
            bodies
                .iter()
                .any(|body| body.contains("Global `value` should start with `$`")),
            "expected warning about missing global prefix, got logs: {:?}",
            bodies
        );
    });
}

#[test]
fn globals_missing_prefix_warns_for_prefixed_pattern_variable_by_default() {
    testing_logger::setup();

    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(None, "(rule ((= $x 1)) ())")
        .unwrap();

    testing_logger::validate(|logs| {
        let bodies: Vec<_> = logs.iter().map(|entry| entry.body.clone()).collect();
        assert!(
            bodies
                .iter()
                .any(|body| body.contains("Global `x` should start with `$`")),
            "expected warning about missing global prefix, got logs: {:?}",
            bodies
        );
    });
}

#[test]
fn globals_missing_prefix_warns_for_prefixed_rule_let_by_default() {
    testing_logger::setup();

    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(None, "(rule () ((let $y 1)))")
        .unwrap();

    testing_logger::validate(|logs| {
        let bodies: Vec<_> = logs.iter().map(|entry| entry.body.clone()).collect();
        assert!(
            bodies
                .iter()
                .any(|body| body.contains("Global `y` should start with `$`")),
            "expected warning about missing global prefix, got logs: {:?}",
            bodies
        );
    });
}

#[test]
fn globals_missing_prefix_errors_when_opted_in() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();
    egraph.set_strict_mode(true);
    let err = egraph
        .parse_and_run_program(None, "(let value 41)")
        .unwrap_err();
    match err {
        Error::TypeError(TypeError::GlobalMissingPrefix { ref name, .. }) => {
            assert_eq!(name, "value");
        }
        other => panic!("expected missing dollar error, got {other:?}"),
    }
}

#[test]
fn globals_missing_prefix_errors_for_prefixed_pattern_variable_when_opted_in() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();
    egraph.set_strict_mode(true);
    let err = egraph
        .parse_and_run_program(None, "(rule ((= $x 1)) ())")
        .unwrap_err();

    match err {
        Error::TypeError(TypeError::NonGlobalPrefixed { ref name, .. }) => {
            assert_eq!(name, "$x");
        }
        other => panic!("expected non-global prefixed variable error, got {other:?}"),
    }
}

#[test]
fn globals_missing_prefix_errors_for_prefixed_rule_let_when_opted_in() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();
    egraph.set_strict_mode(true);
    let err = egraph
        .parse_and_run_program(None, "(rule () ((let $y 1)))")
        .unwrap_err();

    match err {
        Error::TypeError(TypeError::NonGlobalPrefixed { ref name, .. }) => {
            assert_eq!(name, "$y");
        }
        other => panic!("expected non-global prefixed variable error, got {other:?}"),
    }
}

#[test]
fn globals_cannot_be_shadowed_by_pattern_variables() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();
    let program = r#"
        (let $value 41)
        (rule ((= value $value)) ())
    "#;

    let err = egraph.parse_and_run_program(None, program).unwrap_err();
    match err {
        Error::Shadowing(message, _global_span, _shadow_span) => {
            assert!(message.contains("pattern variable `value`"));
            assert!(message.contains("global `$value`"));
        }
        other => panic!("expected shadowing error, got {other:?}"),
    }
}

#[test]
fn test_simple_extract1() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    let outputs = egraph
        .parse_and_run_program(
            None,
            r#"
             (datatype Op (Add i64 i64))
             (let expr (Add 1 1))
             (extract expr)"#,
        )
        .unwrap();
    let CommandOutput::ExtractBest(_, cost, _) = outputs[0] else {
        panic!();
    };
    assert_eq!(cost, 3);
}

#[test]
fn primitive_error_in_extract_returns_error() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();
    // Evaluating this primitive should surface a user-facing error instead of
    // panicking when the primitive fails.
    let err = egraph
        .parse_and_run_program(None, "(extract (<< 1 10000))")
        .unwrap_err();
    assert!(err.to_string().contains("call of primitive << failed"));
}

#[test]
fn primitive_error_in_run_schedule_returns_error() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();
    let program = r#"
        (ruleset problematic)
        (rule ()
              ((let tmp (<< 1 10000)))
              :ruleset problematic)
        (run-schedule (run problematic))
    "#;

    let err = egraph.parse_and_run_program(None, program).unwrap_err();
    assert!(err.to_string().contains("call of primitive << failed"));
}

#[test]
fn test_simple_extract2() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    let outputs = egraph
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
    assert!(matches!(outputs[0], CommandOutput::ExtractBest(_, 4, _)));
}

#[test]
fn test_simple_extract3() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    let outputs = egraph
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

    let CommandOutput::ExtractBest(_, cost, _) = outputs[0] else {
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

    let outputs = egraph
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

    let CommandOutput::ExtractBest(_, cost, _) = outputs[0] else {
        panic!();
    };
    assert_eq!(cost, 0);
}

#[test]
fn test_simple_extract6() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    let outputs = egraph
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

    let CommandOutput::ExtractBest(_, cost, _) = outputs[0] else {
        panic!();
    };
    assert_eq!(cost, 0);
}

#[test]
fn test_simple_extract7() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    let outputs = egraph
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

    let CommandOutput::ExtractBest(_, cost, _) = outputs[0] else {
        panic!();
    };
    assert_eq!(cost, 2);
}

#[test]
fn test_simple_extract8() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    let outputs = egraph
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

    let CommandOutput::ExtractBest(_, cost, _) = outputs[0] else {
        panic!();
    };
    assert_eq!(cost, 10);
}

// Test copied from https://github.com/egraphs-good/egglog-python/issues/387#issuecomment-3628927075
// to verify that we can implement a cost model that counts DAG nodes and it is extracted properly.
// We simplify the use case assuming all costs are set as one for all constructors
#[derive(Clone, Debug)]
enum DagCostValue {
    // A set of all the nodes in this sub-graph
    Nodes(HashSet<(String, Vec<Value>)>),
    // The "cost" returned by enode_cost is only ever
    // directly passed into fold as the head cost, so we save this as a different value
    EnodeCostResult(String, Vec<Value>),
}

impl DagCostValue {
    fn cost(&self) -> usize {
        let DagCostValue::Nodes(nodes) = self else {
            panic!("never will call")
        };
        nodes.len()
    }
}

impl PartialEq for DagCostValue {
    fn eq(&self, other: &Self) -> bool {
        self.cost() == other.cost()
    }
}
impl Eq for DagCostValue {}

impl PartialOrd for DagCostValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for DagCostValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cost().cmp(&other.cost())
    }
}

impl extract::Cost for DagCostValue {
    fn identity() -> Self {
        panic!("not needed")
    }

    fn unit() -> Self {
        panic!("not needed")
    }

    fn combine(self, _other: &Self) -> Self {
        panic!("not needed")
    }
}

#[derive(Clone, Debug)]
struct DagCost;

impl extract::CostModel<DagCostValue> for DagCost {
    fn fold(
        &self,
        _head: &str,
        children_cost: &[DagCostValue],
        head_cost: DagCostValue,
    ) -> DagCostValue {
        let DagCostValue::EnodeCostResult(name, args) = head_cost else {
            panic!("expected enode cost result")
        };
        let mut combined_costs = HashSet::new();
        for child_cost in children_cost {
            let DagCostValue::Nodes(child_costs) = child_cost else {
                panic!("expected costs")
            };
            combined_costs.extend(child_costs.iter().cloned());
        }
        if combined_costs.contains(&(name.clone(), args.clone())) {
            panic!("cycle detected in DAG cost extraction, should never happen");
        }
        // Add cost for this node
        combined_costs.insert((name, args));
        DagCostValue::Nodes(combined_costs)
    }

    fn enode_cost(
        &self,
        _egraph: &EGraph,
        func: &Function,
        row: &egglog_bridge::FunctionRow,
    ) -> DagCostValue {
        let name = func.name().to_string();
        // Remove last value which is the output
        let args: Vec<Value> = row.vals.iter().cloned().take(row.vals.len() - 1).collect();
        DagCostValue::EnodeCostResult(name, args)
    }

    fn container_cost(
        &self,
        _egraph: &EGraph,
        _sort: &ArcSort,
        _value: Value,
        _element_costs: &[DagCostValue],
    ) -> DagCostValue {
        panic!("not needed");
    }

    fn base_value_cost(&self, _egraph: &EGraph, sort: &ArcSort, value: Value) -> DagCostValue {
        DagCostValue::Nodes(HashSet::from([(sort.name().to_string(), vec![value])]))
    }
}

#[test]
fn test_dag_cost_extractor() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    let program = r#"
(sort S)

(constructor Si (i64)     S)
(constructor Swide (S S S S S S S S) S )
(constructor Ssa (S)       S)
(constructor Ssb (S)      S)
(constructor Ssc (S)      S)
(constructor Sp (S S)     S)


(let w
  (Swide (Si 0) (Si 1) (Si 2) (Si 3) (Si 4) (Si 5) (Si 6) (Si 7)))

(let l (Ssa (Ssb (Ssc (Si 0)))))
(let x (Ssa w))
(let v (Sp w x))

(union x l)
"#;

    egraph.parse_and_run_program(None, program).unwrap();

    let rootsort = egraph.get_sort_by_name("S").unwrap().clone();
    let extractor = extract::Extractor::compute_costs_from_rootsorts(
        Some(vec![rootsort.clone()]),
        &egraph,
        DagCost,
    );
    let mut termdag = TermDag::default();
    let w_value = egraph
        .lookup_function("w", &[])
        .expect("function w missing");

    let (cost, _term) = extractor
        .extract_best_with_sort(&egraph, &mut termdag, w_value, rootsort.clone())
        .expect("failed to extract");

    assert_eq!(cost.cost(), 17);

    let v_value = egraph
        .lookup_function("v", &[])
        .expect("function v missing");

    let (cost, _term) = extractor
        .extract_best_with_sort(&egraph, &mut termdag, v_value, rootsort)
        .expect("failed to extract");
    // Cost might either be cheapest of 19 or higher cost of 21 depending on which one it selects first
    // It's greedy so either is acceptable
    assert!(cost.cost() == 19 || cost.cost() == 21);
}

#[test]
fn test_simple_extract9() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    let outputs = egraph
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

    let CommandOutput::ExtractBest(_, cost, _) = outputs[0] else {
        panic!();
    };
    assert_eq!(cost, DefaultCost::MAX);

    let outputs = egraph
        .parse_and_run_program(
            None,
            r#"
            (extract b)
            "#,
        )
        .unwrap();

    let CommandOutput::ExtractBest(_, cost, _) = outputs[0] else {
        panic!();
    };
    assert_eq!(cost, DefaultCost::MAX);

    let outputs = egraph
        .parse_and_run_program(
            None,
            r#"
            (extract c)
            "#,
        )
        .unwrap();

    let CommandOutput::ExtractBest(_, cost, _) = outputs[0] else {
        panic!();
    };
    assert_eq!(cost, DefaultCost::MAX);
}

#[test]
fn test_extract_variants1() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut egraph = EGraph::default();

    let outputs = egraph
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
    assert_eq!(
        outputs[0].to_string(),
        "(\n   (SmallStep (SmallStep (SmallStep (SmallStep (Origin)))))\n   (BigStep (Origin))\n)\n"
    );
}

#[test]
fn test_subsumed_unextractable_action_extract() {
    // Test when an expression is subsumed, it isn't extracted, even if its the cheapest
    let mut egraph = EGraph::default();

    let outputs = egraph
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
        outputs[0],
        CommandOutput::ExtractBest(_, _, Term::App(ref s, ..)) if s == "cheap"
    ));
    // Then if we make one as subsumed, it should give back the variable term
    let outputs = egraph
        .parse_and_run_program(
            None,
            r#"
            (subsume (cheap))
            (extract (exp))
            "#,
        )
        .unwrap();
    assert!(matches!(
        outputs[0],
        CommandOutput::ExtractBest(_, _, Term::App(ref s, ..)) if s == "exp"
    ));
}

#[test]
fn test_subsume_unextractable_insert_and_merge() {
    // Example adapted from https://github.com/egraphs-good/egglog/pull/301#pullrequestreview-1756826062
    let mut egraph = EGraph::default();

    let outputs = egraph
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
        outputs[0],
        CommandOutput::ExtractBest(_, _, Term::App(ref s, ..)) if s == "exp"
    ));
}

#[test]
fn test_subsume_unextractable_action_extract_multiple() {
    // Test when an expression is set as subsumed, it isn't extracted, like with
    // extract multiple
    let mut egraph = EGraph::default();

    let outputs1 = egraph
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
    assert!(matches!(
        outputs1[0],
        CommandOutput::ExtractVariants(_, ref terms) if terms.len() == 2
    ));
    // Then if we make one unextractable, it should only give back one term
    let outputs2 = egraph
        .parse_and_run_program(
            None,
            "
            (subsume (Num 2))
            (extract (Num 1) 2)
            ",
        )
        .unwrap();
    assert!(matches!(
        outputs2[0],
        CommandOutput::ExtractVariants(_, ref terms) if terms.len() == 1
    ));
}

#[test]
fn test_rewrite_subsumed_unextractable() {
    // When a rewrite is marked as a subsumed, the lhs should not be extracted

    let mut egraph = EGraph::default();

    let outputs = egraph
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
    assert_eq!(outputs[1].to_string(), "(exp)\n");
}

#[test]
fn test_rewrite_subsumed() {
    // When a rewrite is marked as a subsumed, the lhs should not be extracted

    let mut egraph = EGraph::default();

    // If we rerite most-exp to another term, that rewrite shouldnt trigger since its been subsumed.
    let outputs = egraph
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
    assert_eq!(outputs[2].to_string(), "(exp)\n");
}

#[test]
fn test_subsume() {
    // Test that if we mark a term as subsumed than no rewrites will be applied to it.
    // We can test this by adding a commutative additon property, and verifying it isn't applied on one of the terms
    // but is on the other
    let mut egraph = EGraph::default();
    let outputs = egraph
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
        outputs[1],
        CommandOutput::ExtractVariants(_, ref terms) if terms.len() == 2
    ));

    let outputs = egraph
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
        outputs[0],
        CommandOutput::ExtractVariants(_, ref terms) if terms.len() == 1
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

    let outputs = egraph
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
    let CommandOutput::ExtractBest(termdag, _cost, term) = outputs[0].clone() else {
        panic!();
    };
    let expr = termdag.term_to_expr(&term, span!());
    let (sort, value) = egraph.eval_expr(&expr).unwrap();

    let serialize_output = egraph.serialize(SerializeConfig::default());
    assert!(serialize_output.is_complete());
    let class_id = egraph.value_to_class_id(&sort, value);
    assert!(serialize_output.egraph.class_data.get(&class_id).is_some());
    assert_eq!(value, egraph.class_id_to_value(&class_id));
}

#[test]
fn test_serialize_617() {
    let program = "
        (sort Node)
        (constructor mk (i64) Node)
        (constructor mkb (i64) Node)
        (rewrite (mkb x) (mk x))

        (mkb 1) (mkb 3) (mkb 5) (mkb 6)

        (union (mk 1) (mk 3))
        (union (mk 3) (mk 5))

        (run-schedule (saturate (run)))";

    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(None, program).unwrap();

    let serialize_output = egraph.serialize(SerializeConfig::default());
    assert!(serialize_output.is_complete());
    assert_eq!(serialize_output.egraph.class_data.len(), 6);
    assert_eq!(serialize_output.egraph.nodes.len(), 12);
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

    let serialize_output = egraph.serialize(SerializeConfig::default());
    assert!(serialize_output.is_complete());
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
    assert!(serialize_output.egraph.nodes[&a_id].subsumed);
    assert!(!serialize_output.egraph.nodes[&b_id].subsumed);
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

#[test]
fn test_print_function_size() {
    let s = "(function f () i64 :no-merge) (set (f) 2) (print-size f)";
    let outputs = EGraph::default().parse_and_run_program(None, s).unwrap();
    assert_eq!(outputs[0].to_string(), "1\n");
}

#[test]
fn test_print_function() {
    let s = "(function f () i64 :no-merge) (set (f) 2) (print-function f)";
    let outputs = EGraph::default().parse_and_run_program(None, s).unwrap();
    assert_eq!(outputs[0].to_string(), "(\n   (f) -> 2\n)\n");
}

#[test]
fn test_print_function_csv() {
    let s = "(function f () i64 :no-merge) (set (f) 2) (print-function f :mode csv)";
    let outputs = EGraph::default().parse_and_run_program(None, s).unwrap();
    assert_eq!(outputs[0].to_string(), "f,2\n");
}

#[test]
fn test_print_stats() {
    let s = "(run 1) (print-stats)";
    let outputs = EGraph::default().parse_and_run_program(None, s).unwrap();
    assert_eq!(
        outputs[1].to_string(),
        "Overall statistics:\nRuleset : search 0.000s, merge 0.000s, rebuild 0.000s\n"
    );
}

#[test]
fn test_run_report() {
    let s = "(run 1)";
    let outputs = EGraph::default().parse_and_run_program(None, s).unwrap();
    assert_eq!(outputs[0].to_string(), "");
    assert!(matches!(outputs[0], CommandOutput::RunSchedule(..)));
}

#[test]
fn test_serialize_message_max_functions() {
    let mut egraph = EGraph::default();
    // Create three zero-arg constructors
    egraph
        .parse_and_run_program(
            None,
            r#"
            (datatype A)
            (constructor a () A)
            (constructor b () A)
            (constructor c () A)
            (a) (b) (c)
            "#,
        )
        .unwrap();
    let serialize_output = egraph.serialize(SerializeConfig {
        max_functions: Some(2),
        max_calls_per_function: None,
        include_temporary_functions: false,
        root_eclasses: vec![],
    });
    assert!(!serialize_output.is_complete());
    assert_eq!(serialize_output.omitted_description(), "Omitted: c\n");
}

#[test]
fn test_serialize_message_max_calls_per_function() {
    let mut egraph = EGraph::default();
    // Single constructor with many distinct calls (different arguments)
    egraph
        .parse_and_run_program(
            None,
            r#"
            (datatype N)
            (constructor mk (i64) N)
            (mk 0) (mk 1) (mk 2) (mk 3)
            "#,
        )
        .unwrap();
    let serialize_output = egraph.serialize(SerializeConfig {
        max_functions: None,
        max_calls_per_function: Some(2),
        include_temporary_functions: false,
        root_eclasses: vec![],
    });
    assert!(!serialize_output.is_complete());
    assert_eq!(serialize_output.omitted_description(), "Truncated: mk\n");
}
