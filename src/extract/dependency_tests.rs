use super::*;

#[test]
fn scheduler_matches_full_scans_across_cutoff_and_scan_orders() {
    for threads in [1, 4] {
        for depth in [0, 1, 2, 3, 4, 8, 16] {
            let mut program = String::from("(datatype S0 (Seed i64 :cost 1))\n");
            for i in 1..=depth {
                let child = i - 1;
                program += &format!(
                    "(datatype S{i} (Step{i} S{child} :cost 1)
                    (Pair{i} S{child} S{child} :cost 0)
                    (Fallback{i} i64 :cost 30) (Loop{i} S{i} :cost 0))"
                );
            }
            // Cross the 32-row scan buffer at the preparation boundary.
            let lanes = if depth == 4 { 35 } else { 2 };
            for lane in 0..lanes {
                program += &format!("(let n{lane}_0 (Seed {lane}))");
                for i in 1..=depth {
                    let child = i - 1;
                    program += &format!("(let n{lane}_{i} (Step{i} n{lane}_{child}))
                        (union n{lane}_{i} (Pair{i} n{lane}_{child} n{lane}_{child}))
                        (union n{lane}_{i} (Fallback{i} {lane})) (union n{lane}_{i} (Loop{i} n{lane}_{i}))");
                }
            }
            // A delayed improvement reaches a seeded cycle; vectors cover constant/empty/duplicate leaves.
            program += &format!(
                "(sort A) (sort B)
                (constructor Expensive () A :cost 100) (constructor FromB (B) A :cost 0)
                (constructor FromA (A) B :cost 0) (constructor Improve (S{depth}) B :cost 0)
                (let a (Expensive)) (let b (FromA a))
                (union a (FromB b)) (union b (Improve n0_{depth}))
                (sort Values (Vec i64)) (sort Eqs (Vec S0))
                (constructor Constants (Values) S{depth}) (constructor Vector (Eqs) S{depth})
                (Constants (vec-of 1 2)) (Vector (vec-of)) (Vector (vec-of n0_0 n1_0))"
            );
            let mut egraph = EGraph::new(threads);
            egraph.parse_and_run_program(None, &program).unwrap();
            for order in 0..5 {
                let mut roots: Vec<_> = (0..=depth)
                    .map(|i| egraph.get_arcsort_by(|sort| sort.name() == format!("S{i}")))
                    .collect();
                match order {
                    1 => roots.reverse(),
                    2 => roots.rotate_left(depth / 2),
                    3 | 4 => {
                        let name = if order == 3 { "A" } else { "B" };
                        roots = vec![egraph.get_arcsort_by(|sort| sort.name() == name)];
                    }
                    _ => {}
                }
                let extractor = TreeExtractor::compute_costs_from_rootsorts(
                    Some(roots),
                    &egraph,
                    DEFAULT_COST_MODEL,
                );
                assert_matches_full_scans(&egraph, extractor);
            }
        }
    }
}

fn assert_matches_full_scans<C: Cost + std::fmt::Debug>(
    egraph: &EGraph,
    mut extractor: TreeExtractor<'_, C>,
) {
    let state =
        |ex: &TreeExtractor<'_, C>| (ex.costs.clone(), ex.topo_rnk_cnt, ex.parent_edge.clone());
    let scheduled = state(&extractor);
    // Equal costs alone would miss changes to chronological ranks and producers.
    extractor.costs.values_mut().for_each(HashMap::clear);
    extractor.parent_edge.values_mut().for_each(HashMap::clear);
    extractor.topo_rnk_cnt = 0;
    let funcs: Vec<_> = extractor
        .funcs
        .iter()
        .map(|&func| ExtractionFunction {
            func,
            output_sort_name: func.extraction_output_sort().name(),
            output_index: func.extraction_output_index(),
        })
        .collect();
    while extractor.ordered_full_sweep(egraph, &funcs) {}
    extractor.save_best_parent_edges(egraph, &funcs);
    assert_eq!(scheduled, state(&extractor));
}

#[test]
fn scheduler_preserves_convergent_self_improvements() {
    struct HalvingCost;
    impl TreeCostModel<DefaultCost> for HalvingCost {
        type EnodeCost = String;
        type ContainerCost = ();

        fn base_value_cost(&self, _: &EGraph, _: &ArcSort, _: Value) -> DefaultCost {
            0
        }
        fn enode_cost(&self, _: &EGraph, func: &Function, _: &Enode<'_>) -> String {
            func.name().to_owned()
        }
        fn container_cost(&self, _: &EGraph, _: &ArcSort, _: Value) {}
        fn fold_enode_cost(&self, name: String, children: &[DefaultCost]) -> DefaultCost {
            match name.as_str() {
                "Seed" => 64,
                "Half" => children[0] / 2,
                "Finish" => 0,
                _ => 1 + children.iter().sum::<DefaultCost>(),
            }
        }
        fn fold_container_cost(&self, (): (), elements: &[DefaultCost]) -> DefaultCost {
            elements.iter().sum()
        }
    }

    let mut egraph = EGraph::new(1);
    egraph
        .parse_and_run_program(
            None,
            "(datatype G0 (Start))
            (datatype G1 (Step1 G0)) (datatype G2 (Step2 G1))
            (datatype G3 (Step3 G2)) (datatype G4 (Step4 G3))
            (datatype Root (Seed) (Half Root) (Finish G4))
            (let root (Seed)) (union root (Half root))
            (union root (Finish (Step4 (Step3 (Step2 (Step1 (Start)))))))",
        )
        .unwrap();
    let expr = egraph.parser.get_expr_from_string(None, "root").unwrap();
    let (sort, value) = egraph.eval_expr(&expr).unwrap();
    let extractor =
        TreeExtractor::compute_costs_from_rootsorts(Some(vec![sort.clone()]), &egraph, HalvingCost);
    // Finish provides an acyclic producer after scheduled self-improvements.
    let mut dag = TermDag::default();
    let best = extractor
        .extract_best_with_sort(&mut dag, value, sort)
        .unwrap();
    assert_eq!(best.cost, 0);
    assert_eq!(
        dag.to_string(best.term),
        "(Finish (Step4 (Step3 (Step2 (Step1 (Start))))))"
    );
    assert_matches_full_scans(&egraph, extractor);
}
