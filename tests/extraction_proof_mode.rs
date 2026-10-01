use egglog::CommandOutput;
use egglog::EGraph;
use egglog::prelude::*;

fn find_extract_best(outputs: &[CommandOutput]) -> String {
    outputs
        .iter()
        .find(|o| matches!(o, CommandOutput::ExtractBest(..)))
        .expect("No ExtractBest output found")
        .to_string()
}

#[test]
fn test_extraction_same_with_proof_mode() {
    // Test that extraction in normal mode produces the same result as extraction in proof mode
    let _ = env_logger::builder().is_test(true).try_init();

    let program = r#"
        (datatype Math
            (Num i64)
            (Add Math Math)
            (Mul Math Math))

        (rewrite (Add (Num a) (Num b)) (Num (+ a b)))
        (rewrite (Mul (Num a) (Num b)) (Num (* a b)))

        ; commutativity
        (rewrite (Add x y) (Add y x))
        (rewrite (Mul x y) (Mul y x))

        ; associativity
        (rewrite (Add (Add x y) z) (Add x (Add y z)))
        (rewrite (Mul (Mul x y) z) (Mul x (Mul y z)))

        ; distributivity
        (rewrite (Mul x (Add y z)) (Add (Mul x y) (Mul x z)))

        (let expr (Mul (Add (Num 1) (Num 2)) (Num 3)))
        (run 10)
    "#;

    // Run in normal mode and extract
    let mut egraph_normal = EGraph::default();
    egraph_normal.parse_and_run_program(None, program).unwrap();
    let normal_output = egraph_normal
        .parse_and_run_program(None, "(extract expr)")
        .unwrap();
    let normal_extracted = find_extract_best(&normal_output);

    // Run in proof mode and extract
    let mut egraph_proofs = EGraph::new_with_proofs();
    egraph_proofs.parse_and_run_program(None, program).unwrap();
    let proofs_output = egraph_proofs
        .parse_and_run_program(None, "(extract expr)")
        .unwrap();
    let proofs_extracted = find_extract_best(&proofs_output);

    // They should produce the same extraction result
    assert_eq!(
        normal_extracted, proofs_extracted,
        "Extraction differs between normal mode and proof mode:\nNormal: {normal_extracted}\nProofs: {proofs_extracted}"
    );

    // The result should be (Num 9) since (1+2)*3 = 9
    assert!(
        normal_extracted.contains("Num") && normal_extracted.contains("9"),
        "Expected (Num 9), got: {normal_extracted}"
    );
}

#[test]
fn test_extraction_same_with_proof_mode_using_rule_macro() {
    // Test using the rule! macro from prelude
    let _ = env_logger::builder().is_test(true).try_init();

    // Setup program with datatypes - use an expression that simplifies to a unique result
    let setup = r#"
        (datatype Expr
            (Var String)
            (Lit i64)
            (Add Expr Expr))

        ; Simplification rule that gives a unique result
        (rewrite (Add (Lit a) (Lit b)) (Lit (+ a b)))

        (let x (Add (Lit 1) (Lit 2)))
        (run 10)
    "#;

    // Run in normal mode
    let mut egraph_normal = EGraph::default();
    egraph_normal.parse_and_run_program(None, setup).unwrap();

    add_ruleset(&mut egraph_normal, "my_rules").unwrap();
    rule(
        &mut egraph_normal,
        "my_rules",
        facts![(= (Add a b) e)],
        actions![(union e (Add b a))],
    )
    .unwrap();

    for _ in 0..5 {
        run_ruleset(&mut egraph_normal, "my_rules").unwrap();
    }

    let normal_output = egraph_normal
        .parse_and_run_program(None, "(extract x)")
        .unwrap();
    let normal_extracted = find_extract_best(&normal_output);

    // Run in proof mode
    let mut egraph_proofs = EGraph::new_with_proofs();
    egraph_proofs.parse_and_run_program(None, setup).unwrap();

    // Add the same rule
    add_ruleset(&mut egraph_proofs, "my_rules").unwrap();
    rule(
        &mut egraph_proofs,
        "my_rules",
        facts![(= (Add a b) e)],
        actions![(union e (Add b a))],
    )
    .unwrap();

    for _ in 0..5 {
        run_ruleset(&mut egraph_proofs, "my_rules").unwrap();
    }

    // Extract
    let proofs_output = egraph_proofs
        .parse_and_run_program(None, "(extract x)")
        .unwrap();
    let proofs_extracted = find_extract_best(&proofs_output);

    // They should produce the same extraction result
    assert_eq!(
        normal_extracted, proofs_extracted,
        "Extraction differs between normal mode and proof mode:\nNormal: {normal_extracted}\nProofs: {proofs_extracted}"
    );

    // The result should be (Lit 3) since 1+2=3
    assert!(
        normal_extracted.contains("Lit") && normal_extracted.contains("3"),
        "Expected (Lit 3), got: {normal_extracted}"
    );
}

#[test]
fn extraction_preserves_wide_rows_and_aliases_after_delayed_dependencies() {
    // Root-first preparation needs six sweeps to reach Gate5. Many mixed,
    // 18-child rows exercise copied row ranges after the value buffer grows,
    // including the extra output columns used by term/proof views.
    let declarations = r#"
        (datatype Gate0 (Seed :cost 1))
        (datatype Gate1 (Step1 Gate0 :cost 1))
        (datatype Gate2 (Step2 Gate1 :cost 1))
        (datatype Gate3 (Step3 Gate2 :cost 1))
        (datatype Gate4 (Step4 Gate3 :cost 1))
        (datatype Gate5 (Step5 Gate4 :cost 1))
        (let gate (Step5 (Step4 (Step3 (Step2 (Step1 (Seed)))))))
        (datatype Child (ChildLeaf Gate5 i64 :cost 1) (Old :cost 100))
        (datatype Other (OtherLeaf bool :cost 1))
        (sort Children (Vec Child))
        (sort ChildMap (Map Child i64))
        (datatype Root
            (Wide Child i64 bool String Children Child String i64 bool
                ChildMap Other i64 Other Child Children bool String Other :cost 1))
        (let old (Old))
    "#;
    let gate = "(Step5 (Step4 (Step3 (Step2 (Step1 (Seed))))))";
    for mut egraph in [
        EGraph::default(),
        EGraph::new_with_term_encoding(),
        EGraph::new_with_proofs(),
    ] {
        egraph.parse_and_run_program(None, declarations).unwrap();
        let mut rows = String::new();
        let mut expected = Vec::new();
        for i in 0..33 {
            rows.push_str(&format!("(let child{i} (ChildLeaf gate {i}))\n"));
        }
        rows.push_str("(union old child0)\n");
        for i in 0..33 {
            let next = (i + 1) % 33;
            let even = i % 2 == 0;
            let odd = !even;
            let negative = -(i as i64);
            let index = i * 101;
            rows.push_str(&format!(
                "(let wide{i} (Wide child{i} {i} {even} \"row{i}\" \
                 (vec-of child{i} child{next} child{i}) child{next} \"tail{i}\" \
                 {negative} {odd} (map-insert (map-empty) child{i} {i}) \
                 (OtherLeaf {even}) {index} (OtherLeaf {odd}) child{i} \
                 (vec-of child{i} child{next} child{i}) {even} \"end{i}\" (OtherLeaf {even})))\n"
            ));
            let child = format!("(ChildLeaf {gate} {i})");
            let next_child = format!("(ChildLeaf {gate} {next})");
            let children = format!("(vec-of {child} {next_child} {child})");
            expected.push(format!(
                "(Wide {child} {i} {even} \"row{i}\" {children} {next_child} \
                 \"tail{i}\" {negative} {odd} (map-of {child} {i}) \
                 (OtherLeaf {even}) {index} (OtherLeaf {odd}) {child} \
                 {children} {even} \"end{i}\" (OtherLeaf {even}))"
            ));
        }
        egraph.parse_and_run_program(None, &rows).unwrap();
        for (i, expected) in expected.into_iter().enumerate() {
            let mut outputs = egraph
                .parse_and_run_program(None, &format!("(extract wide{i}) (extract wide{i} 2)"))
                .unwrap();
            // Proof encoding also emits reports for its internal schedules.
            outputs.retain(|output| !matches!(output, CommandOutput::RunSchedule(_)));
            let [
                CommandOutput::ExtractBest(dag, 97, term),
                CommandOutput::ExtractVariants(variants_dag, variants),
            ] = outputs.as_slice()
            else {
                panic!("expected best (cost 97) and variant extraction, got {outputs:?}");
            };
            assert_eq!(dag.to_string(*term), expected);
            assert_eq!(variants.len(), 1);
            assert_eq!(variants_dag.to_string(variants[0]), expected);
        }
        let mut outputs = egraph.parse_and_run_program(None, "(extract old)").unwrap();
        outputs.retain(|output| !matches!(output, CommandOutput::RunSchedule(_)));
        let [CommandOutput::ExtractBest(dag, 8, term)] = outputs.as_slice() else {
            panic!("expected alias extraction with cost 8, got {outputs:?}");
        };
        assert_eq!(dag.to_string(*term), format!("(ChildLeaf {gate} 0)"));
    }
}
