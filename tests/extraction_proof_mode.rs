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
    // Six dependency levels, 33 rows and 18 mixed children cross both scan and arena boundaries.
    let gate = "(Step5 (Step4 (Step3 (Step2 (Step1 (Seed))))))";
    let mut program = format!(
        r#"
        (datatype Gate0 (Seed :cost 1)) (datatype Gate1 (Step1 Gate0 :cost 1))
        (datatype Gate2 (Step2 Gate1 :cost 1)) (datatype Gate3 (Step3 Gate2 :cost 1))
        (datatype Gate4 (Step4 Gate3 :cost 1)) (datatype Gate5 (Step5 Gate4 :cost 1))
        (let gate {gate})
        (datatype Child (ChildLeaf Gate5 i64 :cost 1) (Old :cost 100))
        (datatype Other (OtherLeaf bool :cost 1))
        (sort Children (Vec Child)) (sort ChildMap (Map Child i64))
        (datatype Root (Wide Child i64 bool String Children Child String i64 bool
            ChildMap Other i64 Other Child Children bool String Other :cost 1))
        (let old (Old))"#
    );
    for i in 0..33 {
        program += &format!("(let child{i} (ChildLeaf gate {i}))");
    }
    program += "(union old child0)";
    let mut expected = Vec::new();
    let mut extracts = String::new();
    for i in 0..33 {
        let next = (i + 1) % 33;
        let (even, odd, negative, index) = (i % 2 == 0, i % 2 != 0, -(i as i64), i * 101);
        let children = format!("(vec-of child{i} child{next} child{i})");
        let row = format!(
            "(Wide child{i} {i} {even} \"row{i}\" {children} child{next} \"tail{i}\" \
             {negative} {odd} (map-insert (map-empty) child{i} {i}) \
             (OtherLeaf {even}) {index} (OtherLeaf {odd}) child{i} \
             {children} {even} \"end{i}\" (OtherLeaf {even}))"
        );
        program += &format!("(let wide{i} {row})");
        extracts += &format!("(extract wide{i}) (extract wide{i} 2)");
        let term = row
            .replace(
                &format!("child{next}"),
                &format!("(ChildLeaf {gate} {next})"),
            )
            .replace(&format!("child{i}"), &format!("(ChildLeaf {gate} {i})"))
            .replace("map-insert (map-empty)", "map-of");
        expected.extend([(Some(97), vec![term.clone()]), (None, vec![term])]);
    }
    program += &format!("{extracts}(extract old)");
    expected.push((Some(8), vec![format!("(ChildLeaf {gate} 0)")]));
    for mut graph in [
        EGraph::default(),
        EGraph::new_with_term_encoding(),
        EGraph::new_with_proofs(),
    ] {
        let actual: Vec<_> = graph
            .parse_and_run_program(None, &program)
            .unwrap()
            .into_iter()
            .filter_map(|output| match output {
                CommandOutput::ExtractBest(dag, cost, term) => {
                    Some((Some(cost), vec![dag.to_string(term)]))
                }
                CommandOutput::ExtractVariants(dag, terms) => {
                    Some((None, terms.into_iter().map(|t| dag.to_string(t)).collect()))
                }
                CommandOutput::RunSchedule(_) => None,
                other => panic!("unexpected extraction output: {other:?}"),
            })
            .collect();
        assert_eq!(actual, expected);
    }
}
