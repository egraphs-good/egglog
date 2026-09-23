use egglog::{EGraph, Error, ast::*, program::*, span};
use egglog_ast::span::{EgglogSpan, SrcFile};
use std::sync::Arc;

#[test]
fn text_json_execution_and_replay_agree() {
    let source = include_str!("program/portable.egg");
    let program = Program::parse(Some("portable.egg".into()), source).unwrap();
    let json = program.to_json().unwrap();
    let imported = Program::from_json(&json).unwrap();
    assert_eq!(json, imported.to_json().unwrap());
    let mut direct = EGraph::default();
    let mut shared = EGraph::default();
    let direct = direct.parse_and_run_program(None, source).unwrap();
    let shared = shared.run_shared_program(imported.clone()).unwrap();
    assert_eq!(
        direct.last().unwrap().to_string(),
        shared.last().unwrap().to_string()
    );
    let replay = imported.to_replayable_egglog().unwrap();
    EGraph::default()
        .parse_and_run_program(None, &replay)
        .unwrap();
}

#[test]
fn every_command_variant_and_internal_metadata_roundtrip() {
    let source = r#"
        (sort E)
        (datatype T (A i64 :cost 2) (B))
        (datatype* (U (U1 T)) (sort Vs (Vec T)))
        (constructor C (i64) E :cost 5 :unextractable)
        (relation R (E))
        (function F (E) i64 :merge (+ old new) :unextractable :internal-hidden :internal-let :internal-term-constructor C)
        (ruleset r)
        (unstable-combined-ruleset all r)
        (rule ((= x (A y)) (R x)) ((let z y) (set (F x) z) (union x x) (delete (R x)) (subsume (A y)) (panic "stop") (R x)) :ruleset r :name "named" :naive :no-decomp :internal-include-subsumed)
        (rewrite (A x) (A x) :when ((= x 2)) :ruleset r)
        (birewrite (A x) (A x) :ruleset r)
        (let $a (A 1))
        (extract $a 2)
        (run-schedule (seq (repeat 2 (run r :until (= $a (A 1)))) (saturate (run r))))
        (print-stats :file "stats.json")
        (check (= $a (A 1)))
        (prove (= $a (A 1)))
        (prove-exists A)
        (print-function A 2 :file "rows.csv" :mode csv)
        (print-size A)
        (input A "input.csv")
        (output "output.csv" $a)
        (push 2)
        (pop 2)
        (fail (check (= 1 2)))
        (include "other.egg")
    "#;
    let mut program = Program::parse(None, source).unwrap();
    program.commands.push(Command::UserDefined(
        span!(),
        "host hook".into(),
        vec![Expr::Lit(span!(), Literal::Bool(true))],
    ));
    if let Command::Sort {
        uf,
        proof_func,
        container_rebuild,
        proof_constructors,
        unionable,
        ..
    } = &mut program.commands[0]
    {
        *uf = Some(("UF_E".into(), Some("UF_Ef".into())));
        *proof_func = Some("proof_E".into());
        *container_rebuild = Some(ContainerRebuildSpec {
            internal_rebuild_prim: "rebuild".into(),
            internal_rebuild_proof_prim: Some("proof_rebuild".into()),
        });
        *proof_constructors = Some(ProofConstructorNames {
            congr: "c".into(),
            trans: "t".into(),
            sym: "s".into(),
            normalize: "n".into(),
        });
        *unionable = false;
    }
    if let Command::Constructor {
        term_constructor,
        cost,
        ..
    } = &mut program.commands[3]
    {
        *term_constructor = Some("original".into());
        *cost = Some(u64::MAX);
    }
    program.commands.push(Command::RunSchedule(Schedule::Repeat(
        span!(),
        usize::MAX,
        Box::new(Schedule::Sequence(span!(), vec![])),
    )));
    let encoded = program.to_json().unwrap();
    let imported = Program::from_json(&encoded).unwrap();
    assert_eq!(encoded, imported.to_json().unwrap());
    let tags: std::collections::BTreeSet<_> = serde_json::to_value(&imported).unwrap()["commands"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["type"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(
        tags.len(),
        27,
        "all native Command variants must be covered"
    );
    assert!(program.to_replayable_egglog().is_err());
}

#[test]
fn exact_literals_include_large_integers_signed_zero_and_nan_payloads() {
    let bits = [
        0,
        1 << 63,
        0x3ff0_0000_0000_0000,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_1234,
        0x7ff0_0000_0000_0001,
    ];
    let mut literals: Vec<_> = bits
        .into_iter()
        .map(|b| Literal::Float(f64::from_bits(b).into()))
        .collect();
    literals.extend([
        Literal::Int(i64::MIN),
        Literal::Int(i64::MAX),
        Literal::String("\"\\\nλ".into()),
        Literal::Bool(false),
        Literal::Unit,
    ]);
    let commands = literals
        .iter()
        .cloned()
        .map(|lit| Command::Action(Action::Expr(span!(), Expr::Lit(span!(), lit))))
        .collect();
    let program = Program::new(commands).unwrap();
    let json = program.to_json().unwrap();
    let imported = Program::from_json(&json).unwrap();
    for (index, expected) in bits.into_iter().enumerate() {
        let Command::Action(Action::Expr(_, Expr::Lit(_, Literal::Float(actual)))) =
            &imported.commands[index]
        else {
            panic!()
        };
        assert_eq!(expected, actual.0.to_bits());
    }
    assert!(json.contains("\"9223372036854775807\""));
    assert_eq!(json, imported.to_json().unwrap());
    assert!(program.to_replayable_egglog().is_err());
}

#[test]
fn malformed_versions_fields_literals_and_spans_fail_before_execution() {
    let valid = Program::parse(None, "(let $x 1)")
        .unwrap()
        .to_json()
        .unwrap();
    assert!(Program::from_json(&valid.replace("egglog-program-v1", "egglog-program-v2")).is_err());
    let mut value: serde_json::Value = serde_json::from_str(&valid).unwrap();
    value["typo"] = true.into();
    assert!(Program::from_json(&value.to_string()).is_err());
    assert!(Program::from_json(&valid.replace("\"value\":\"1\"", "\"value\":\"01\"")).is_err());
    let source = Arc::new(SrcFile {
        name: None,
        contents: "λ".into(),
    });
    for (i, j) in [(1, 2), (0, 3), (2, 0)] {
        let bad_span = Span::Egglog(Arc::new(EgglogSpan {
            file: source.clone(),
            i,
            j,
        }));
        let commands = vec![Command::Action(Action::Expr(
            bad_span.clone(),
            Expr::Lit(bad_span, Literal::Unit),
        ))];
        assert!(matches!(
            Program::new(commands),
            Err(ProgramError::InvalidSpan)
        ));
        let json = format!(
            r#"{{"format":"egglog-program-v1","commands":[{{"type":"AddRuleset","value":[{{"type":"Egglog","value":{{"file":{{"name":null,"contents":"λ"}},"i":{i},"j":{j}}}}},"r"]}}]}}"#
        );
        assert!(Program::from_json(&json).is_err());
    }
    let invalid = Program::new(vec![Command::Check(
        Span::Panic,
        vec![Fact::Fact(Expr::Call(
            Span::Panic,
            "missing".into(),
            vec![],
        ))],
    )])
    .unwrap();
    let error = EGraph::default().run_shared_program(invalid).unwrap_err();
    assert!(!error.to_string().is_empty());
}

#[test]
fn admitted_typed_depth_roundtrips_and_excessive_ingress_is_rejected() {
    let mut expr = Expr::Lit(span!(), Literal::Int(1));
    for _ in 0..128 {
        expr = Expr::Call(span!(), "f".into(), vec![expr]);
    }
    let program = Program::new(vec![Command::Action(Action::Expr(span!(), expr))]).unwrap();
    let encoded = program.to_json().unwrap();
    assert_eq!(
        encoded,
        Program::from_json(&encoded).unwrap().to_json().unwrap()
    );
    let deeply_nested = format!("{}0{}", "[".repeat(1025), "]".repeat(1025));
    assert!(matches!(
        Program::from_json(&deeply_nested),
        Err(ProgramError::Limit(_))
    ));
    assert!(matches!(
        Program::parse(None, &"(".repeat(257)),
        Err(ProgramError::Limit(_))
    ));
    assert!(matches!(
        Program::from_json(&" ".repeat(MAX_PROGRAM_BYTES + 1)),
        Err(ProgramError::Limit(_))
    ));
}

#[test]
fn native_names_are_lossless_and_diagnostic_text_is_not_replay() {
    for name in ["module::<Type as Trait>::method", "@generated"] {
        let program = Program::new(vec![Command::Action(Action::Let(
            span!(),
            "$value".into(),
            Expr::Var(span!(), name.into()),
        ))])
        .unwrap();
        assert!(program.to_egglog().contains(name));
        assert_eq!(
            program.to_json().unwrap(),
            Program::from_json(&program.to_json().unwrap())
                .unwrap()
                .to_json()
                .unwrap()
        );
        assert!(program.to_replayable_egglog().is_err());
    }
}

#[test]
fn recording_preserves_scopes_outcomes_prefix_and_independent_clones() {
    let mut graph = EGraph::default();
    graph.start_recording();
    graph
        .parse_and_run_program(None, "(datatype E (A)) (push) (let $a (A)) (pop)")
        .unwrap();
    assert!(!graph.try_start_recording());
    assert_eq!(graph.recorded_program().unwrap().unwrap().commands.len(), 4);
    let mut clone = graph.clone();
    clone
        .parse_and_run_program(None, "(ruleset only-clone)")
        .unwrap();
    let error = graph
        .parse_and_run_program(
            None,
            "(ruleset first) (check (= 1 2)) (ruleset unreachable)",
        )
        .unwrap_err();
    assert!(matches!(error, Error::CheckError(..)));
    graph
        .parse_and_run_program(None, "(ruleset after-error)")
        .unwrap();
    let record = graph.stop_recording().unwrap();
    assert_eq!(record.entries.len(), 7);
    assert!(matches!(
        record.entries[5].outcome,
        CommandOutcome::Failure { .. }
    ));
    assert!(matches!(record.entries[6].outcome, CommandOutcome::Success));
    assert_eq!(record.program().unwrap().commands.len(), 7);
    assert_eq!(clone.stop_recording().unwrap().entries.len(), 5);
    assert!(graph.recorded_program().unwrap().is_none());
    assert!(graph.try_start_recording());
    assert!(graph.stop_recording().unwrap().entries.is_empty());
}

#[test]
fn native_programs_keep_frontend_limits_but_wire_limits_are_checked() {
    let program = Program::new(vec![Command::AddRuleset(Span::Panic, "r".into()); 50_001]).unwrap();
    assert!(matches!(program.to_json(), Err(ProgramError::Limit(_))));
}

#[test]
fn schema_is_generated_from_native_types() {
    let schema = Program::schema();
    let checked_in: serde_json::Value =
        serde_json::from_str(include_str!("../schema/program-v1.schema.json")).unwrap();
    assert_eq!(
        schema, checked_in,
        "regenerate with the shared_program schema example"
    );
    assert!(schema["$defs"].as_object().unwrap().contains_key("Literal"));
    let schema = serde_json::to_string(&schema).unwrap();
    assert!(schema.contains("egglog-program-v1"));
    assert!(schema.contains("^[0-9a-f]{16}$"));
    assert!(schema.contains("container_rebuild"));
    assert!(schema.contains("include_subsumed"));
}

#[test]
fn recording_preserves_partial_host_errors_without_duplicate_nested_commands() {
    struct Host;
    impl egglog::UserDefinedCommand for Host {
        fn update(
            &self,
            graph: &mut EGraph,
            _: &[Expr],
        ) -> Result<Vec<egglog::CommandOutput>, Error> {
            graph.parse_and_run_program(None, "(let $committed 7)")?;
            Err(Error::BackendError("original host error".into()))
        }
    }
    let mut graph = EGraph::default();
    graph.add_command("host".into(), Arc::new(Host)).unwrap();
    graph.start_recording();
    let program = Program::new(graph.parse_program(None, "(host)").unwrap()).unwrap();
    let error = graph.run_shared_program(program).unwrap_err();
    assert!(matches!(error, Error::BackendError(ref message) if message == "original host error"));
    graph
        .parse_and_run_program(None, "(check (= $committed 7)) (fail (check (= 1 2)))")
        .unwrap();
    let record = graph.stop_recording().unwrap();
    assert_eq!(record.entries.len(), 3);
    assert!(
        matches!(&record.entries[0].outcome, CommandOutcome::Failure { message } if message == "original host error")
    );
    assert!(matches!(record.entries[1].outcome, CommandOutcome::Success));
    assert!(matches!(record.entries[2].outcome, CommandOutcome::Success));
}

#[test]
fn recording_resumes_after_host_unwind_without_losing_pending_entry() {
    struct Host;
    impl egglog::UserDefinedCommand for Host {
        fn update(&self, _: &mut EGraph, _: &[Expr]) -> Result<Vec<egglog::CommandOutput>, Error> {
            std::panic::panic_any("original host panic");
        }
    }
    let mut graph = EGraph::default();
    graph.add_command("host".into(), Arc::new(Host)).unwrap();
    graph.start_recording();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        graph.parse_and_run_program(None, "(host)").unwrap();
    }))
    .unwrap_err();
    assert_eq!(panic.downcast_ref::<&str>(), Some(&"original host panic"));
    graph
        .parse_and_run_program(None, "(ruleset after-unwind)")
        .unwrap();
    let record = graph.stop_recording().unwrap();
    assert_eq!(record.entries.len(), 2);
    assert!(matches!(record.entries[0].outcome, CommandOutcome::Pending));
    assert!(matches!(record.entries[1].outcome, CommandOutcome::Success));
}

#[test]
fn source_nesting_uses_lexer_quote_boundaries() {
    let input = format!("(let $x\" {}0{}", "(f ".repeat(300), ")".repeat(301));
    assert!(matches!(
        Program::parse(None, &input),
        Err(ProgramError::Limit("256 levels of nesting"))
    ));
    // A quote inside an atom belongs to that atom. A quote at the start of a
    // token begins a string, whose parentheses do not contribute to nesting.
    Program::parse(None, "(let $x\" 0)").unwrap();
    Program::parse(None, &format!("(let $text \"{}\")", "(".repeat(300))).unwrap();
    Program::parse(None, &format!("; {}\n(let $x 0)", "(".repeat(300))).unwrap();
}

#[test]
fn command_error_observation_changes_only_recorded_outcome_not_native_execution() {
    for recording in [false, true] {
        let mut graph = EGraph::default();
        if recording {
            graph.start_recording();
        }
        let program = Program::parse(None, "(let $first 1) (let $suffix 2)").unwrap();
        let mut observations = 0;
        let result = graph.run_shared_program_with_command_error(program, || {
            observations += 1;
            (observations == 1).then(|| "latched host error".into())
        });
        assert!(
            result.is_ok(),
            "host observation must not replace native success"
        );
        assert_eq!(observations, if recording { 2 } else { 0 });
        let record = graph.stop_recording();
        // The suffix executes identically with and without recording.
        graph
            .parse_and_run_program(None, "(check (= $first 1)) (check (= $suffix 2))")
            .unwrap();
        if let Some(record) = record {
            assert!(
                matches!(&record.entries[0].outcome, CommandOutcome::Failure { message } if message == "latched host error")
            );
            assert!(matches!(record.entries[1].outcome, CommandOutcome::Success));
        }

        if recording {
            graph.start_recording();
        }
        let program = Program::parse(None, "(check (= 1 2)) (let $unreached 3)").unwrap();
        let mut observations = 0;
        let error = graph
            .run_shared_program_with_command_error(program, || {
                observations += 1;
                Some("latched host error".into())
            })
            .unwrap_err();
        assert!(matches!(error, Error::CheckError(..)));
        assert_eq!(observations, usize::from(recording));
        if let Some(record) = graph.stop_recording() {
            assert_eq!(record.entries.len(), 1);
            assert!(
                matches!(&record.entries[0].outcome, CommandOutcome::Failure { message } if message.contains("Check failed") && message.contains("latched host error"))
            );
        }
    }
}

#[test]
fn command_record_json_uses_checked_program_codec() {
    let mut expr = Expr::Lit(span!(), Literal::Int(i64::MAX));
    for _ in 0..128 {
        expr = Expr::Call(span!(), "f".into(), vec![expr]);
    }
    let mut record = CommandRecord {
        entries: vec![RecordedCommand {
            command: Command::Action(Action::Expr(span!(), expr)),
            outcome: CommandOutcome::Pending,
        }],
    };
    let encoded = record.to_json().unwrap();
    assert_eq!(
        encoded,
        CommandRecord::from_json(&encoded)
            .unwrap()
            .to_json()
            .unwrap()
    );
    assert!(CommandRecord::from_json(r#"{"entries":[],"typo":true}"#).is_err());
    let too_deep = format!("{}0{}", "[".repeat(1025), "]".repeat(1025));
    assert!(matches!(
        CommandRecord::from_json(&too_deep),
        Err(ProgramError::Limit(_))
    ));
    record.entries[0].outcome = CommandOutcome::Failure {
        message: "x".repeat(MAX_PROGRAM_BYTES + 1),
    };
    assert!(record.to_json().is_err());

    let bad_span = Span::Egglog(Arc::new(EgglogSpan {
        file: Arc::new(SrcFile {
            name: None,
            contents: "λ".into(),
        }),
        i: 1,
        j: 2,
    }));
    record.entries = vec![RecordedCommand {
        command: Command::AddRuleset(bad_span, "r".into()),
        outcome: CommandOutcome::Success,
    }];
    assert!(matches!(record.to_json(), Err(ProgramError::InvalidSpan)));
}
