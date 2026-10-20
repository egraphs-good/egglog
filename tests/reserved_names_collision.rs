use egglog::{
    CommandOutput, EGraph,
    ast::{GenericCommand, Parser},
};

#[test]
fn later_locals_do_not_capture_previous_encoding_temporaries() {
    for mut egraph in [EGraph::new_with_term_encoding(), EGraph::new_with_proofs()] {
        egraph
            .parse_and_run_program(None, "(datatype T (C i64)) (C 1)")
            .unwrap();
        // Encoding the constructor call introduces a top-level temporary of T.
        // Its spelling is not a global binding in the original user program.
        let temporary = egraph
            .functions_iter()
            .find(|(_, function)| {
                function.is_let_binding() && function.func_type().output.name() == "T"
            })
            .expect("term encoding should have introduced a temporary")
            .0
            .clone();
        egraph
            .parse_and_run_program(None, &format!("(check (= {temporary} 7))"))
            .expect("a later local must not refer to a previous encoding temporary");
    }
}

#[test]
fn incremental_resolved_programs_replay_without_capturing_later_locals() {
    for mut egraph in [EGraph::new_with_term_encoding(), EGraph::new_with_proofs()] {
        let mut resolved = egraph
            .resolve_program(None, "(datatype T (C i64)) (C 1)")
            .unwrap();
        let temporary = resolved
            .iter()
            .find_map(|command| match command {
                GenericCommand::Function {
                    name,
                    schema,
                    let_binding: true,
                    ..
                } if schema.output == "T" => Some(name.clone()),
                _ => None,
            })
            .expect("term encoding should have introduced a temporary");
        resolved.extend(
            egraph
                .resolve_program(None, &format!("(check (= {temporary} 7))"))
                .expect("a later local must not refer to a previous encoding temporary"),
        );
        let printed = resolved
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        EGraph::default()
            .parse_and_run_program(None, &printed)
            .expect("concatenated resolved programs must preserve local bindings on replay");
    }
}

#[test]
fn replay_restores_reserved_proof_constructor_names() {
    let source = "
        (datatype User (@Fiat) (@Rule) (@Merge) (@Eval) (@PCons) (@PNil) (Value i64))
        (Value 1)
        (check (Value 1))";
    let printed = EGraph::new_with_proofs()
        .with_proof_testing()
        .resolve_program(None, source)
        .unwrap()
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(printed.contains("@Fiat@1 @Rule@1 @Merge@1 @Eval@1 @PCons@1 @PNil@1"));
    let mut replay = EGraph::default();
    replay
        .set_proof_checking_program(
            Parser::default()
                .get_program_from_string(None, source)
                .unwrap(),
            true,
        )
        .unwrap();
    let outputs = replay.parse_and_run_program(None, &printed).unwrap();
    assert!(
        outputs
            .iter()
            .any(|output| matches!(output, CommandOutput::ProveExists { .. }))
    );
}

#[test]
fn original_container_proof_annotation_still_parses() {
    let source = "(sort P :internal-proof-names C T S N)";
    let parsed = Parser::default()
        .get_program_from_string(None, source)
        .unwrap();
    assert_eq!(parsed[0].to_string(), source);
}
