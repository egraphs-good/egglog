use egglog::ast::{Command, ResolvedNCommand};
use egglog::{
    CommandOutput, EGraph, Error, ExecutionEvent, ExecutionOrigin, proof::ProveExistsError,
};
use std::path::PathBuf;

fn output_kinds(outputs: &[CommandOutput]) -> Vec<&'static str> {
    outputs
        .iter()
        .map(|output| match output {
            CommandOutput::RunSchedule(_) => "run",
            CommandOutput::ProveExists { .. } => "proof",
            CommandOutput::PrintFunctionSize(_) => "size",
            output => panic!("unexpected output: {output:?}"),
        })
        .collect()
}

#[test]
fn output_sink_appends_completed_outputs_and_preserves_runtime_effects() {
    let source = r#"
        (datatype Num (N i64))
        (N 1)
        (print-size N)
        (panic "stop here")
        (N 2)
    "#;
    for preserve_outputs in [false, true] {
        let mut graph = EGraph::default();
        let program = graph.parse_program(None, source).unwrap();
        if preserve_outputs {
            let mut outputs = vec![CommandOutput::PrintFunctionSize(99)];
            let error = graph
                .run_program_with_outputs(program, &mut outputs)
                .unwrap_err();
            assert!(error.to_string().contains("stop here"));
            assert_eq!(outputs.len(), 2);
            assert!(matches!(outputs[0], CommandOutput::PrintFunctionSize(99)));
            assert!(matches!(outputs[1], CommandOutput::PrintFunctionSize(1)));
        } else {
            // The original Result API remains lossy on error, not transactional.
            let error = graph.run_program(program).unwrap_err();
            assert!(error.to_string().contains("stop here"));
        }
        graph.parse_and_run_program(None, "(check (N 1))").unwrap();
        assert!(graph.parse_and_run_program(None, "(check (N 2))").is_err());
    }
}

#[test]
fn resolution_failure_does_not_discard_a_completed_output() {
    let mut graph = EGraph::default();
    let program = graph
        .parse_program(None, "(datatype Num (N i64)) (print-size N) (unknown)")
        .unwrap();
    let mut outputs = vec![];
    let error = graph
        .run_program_with_outputs(program, &mut outputs)
        .unwrap_err();
    assert!(matches!(error, Error::TypeError(_)), "{error}");
    assert!(matches!(
        outputs.as_slice(),
        [CommandOutput::PrintFunctionSize(0)]
    ));
}

#[test]
fn failed_prove_retains_successful_proof_and_completed_helper_runs() {
    let mut graph = EGraph::new_with_proofs();
    graph
        .parse_and_run_program(None, "(datatype Num (N i64)) (N 2)")
        .unwrap();
    let program = graph
        .parse_program(None, "(prove (N 2)) (prove (N 3)) (N 4)")
        .unwrap();
    let mut outputs = vec![];
    let error = graph
        .run_program_with_outputs(program, &mut outputs)
        .unwrap_err();
    assert!(
        matches!(
            error,
            Error::ProofError {
                error: ProveExistsError::QueryDidNotMatch { .. },
                ..
            }
        ),
        "{error}"
    );
    assert_eq!(
        output_kinds(&outputs),
        ["run", "run", "run", "proof", "run", "run", "run", "run"]
    );
    graph.parse_and_run_program(None, "(check (N 2))").unwrap();
    assert!(graph.parse_and_run_program(None, "(check (N 4))").is_err());
    // Returned proofs own their syntax; no live graph or output callback is retained.
    drop(graph);
    let CommandOutput::ProveExists {
        proof_store,
        proof_id,
    } = &outputs[3]
    else {
        unreachable!()
    };
    let proof = proof_store.get(*proof_id);
    assert_eq!(proof.lhs(), proof.rhs());
}

struct IncludeDirectory(PathBuf);

impl Drop for IncludeDirectory {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn nested_includes_preserve_all_completed_outputs_on_error() {
    let directory = IncludeDirectory(std::env::temp_dir().join(format!(
        "egglog-proof-output-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
    )));
    std::fs::create_dir(&directory.0).unwrap();
    let inner = directory.0.join("inner.egg");
    let outer = directory.0.join("outer.egg");
    std::fs::write(
        &inner,
        r#"(N 2) (print-size N) (panic "inside include") (N 3)"#,
    )
    .unwrap();
    std::fs::write(
        &outer,
        format!(
            "(N 1) (print-size N) (include {:?}) (N 4)",
            inner.to_str().unwrap()
        ),
    )
    .unwrap();
    let source = format!(
        "(datatype Num (N i64)) (print-size N) (include {:?}) (N 5)",
        outer.to_str().unwrap()
    );
    let mut graph = EGraph::default();
    let program = graph.parse_program(None, &source).unwrap();
    let mut outputs = vec![];
    let error = graph
        .run_program_with_outputs(program, &mut outputs)
        .unwrap_err();
    assert!(error.to_string().contains("inside include"));
    assert!(matches!(
        outputs.as_slice(),
        [
            CommandOutput::PrintFunctionSize(0),
            CommandOutput::PrintFunctionSize(1),
            CommandOutput::PrintFunctionSize(2),
        ]
    ));
    graph
        .parse_and_run_program(None, "(check (N 1) (N 2))")
        .unwrap();
    for absent in [3, 4, 5] {
        assert!(
            graph
                .parse_and_run_program(None, &format!("(check (N {absent}))"))
                .is_err()
        );
    }
    let mut observed = EGraph::default();
    let program = observed.parse_program(None, &source).unwrap();
    let mut causes = vec![];
    let error = observed
        .run_program_observed(
            program,
            &mut |cause, origin, output| {
                assert_eq!(origin, ExecutionOrigin::Requested);
                assert!(matches!(output, CommandOutput::PrintFunctionSize(_)));
                causes.push(cause);
            },
            &mut |_, _| {},
        )
        .unwrap_err();
    assert!(error.to_string().contains("inside include"));
    assert_eq!(causes, [1, 2, 2]);
}

#[test]
fn successful_sink_and_existing_api_keep_the_same_native_output_order() {
    let source = "(datatype Num (N i64)) (N 2) (prove (N 2)) (print-size N)";
    let mut old = EGraph::new_with_proofs();
    let expected = old.parse_and_run_program(None, source).unwrap();
    let mut graph = EGraph::new_with_proofs();
    let program = graph.parse_program(None, source).unwrap();
    let mut outputs = vec![];
    graph
        .run_program_with_outputs(program, &mut outputs)
        .unwrap();
    assert_eq!(output_kinds(&outputs), output_kinds(&expected));
    assert_eq!(
        CommandOutput::snapshot_stable_under_proof_encoding(&outputs),
        CommandOutput::snapshot_stable_under_proof_encoding(&expected)
    );
}

#[test]
fn observed_leaves_partition_inline_and_trailing_work_in_every_mode() {
    for mode in 0..4 {
        let mut graph = match mode {
            0 => EGraph::default(),
            1 => EGraph::new_with_term_encoding(),
            2 => EGraph::new_with_proofs(),
            _ => EGraph::new_with_proofs().with_proof_testing(),
        };
        graph
            .parse_and_run_program(
                None,
                "
            (datatype Num (N i64) (Goal i64))
            (ruleset rebuilding_ruleset)
            (N 1)
            (rule ((N x)) ((Goal x)) :ruleset rebuilding_ruleset)
        ",
            )
            .unwrap();
        let mut legacy = graph.clone();
        let program = graph
            .parse_program(None, "(run rebuilding_ruleset 1)")
            .unwrap();
        let expected = legacy.run_program(program.clone()).unwrap();
        let mut outputs = vec![];
        let mut leaf_iterations = vec![];
        let mut entered = [0; 2];
        let mut finished = [0; 2];
        graph
            .run_program_observed(
                program,
                &mut |cause, origin, output| {
                    assert_eq!(cause, 0);
                    outputs.push((origin, output));
                },
                &mut |cause, event| {
                    assert_eq!(cause, 0);
                    match event {
                        ExecutionEvent::RunStarted { origin, .. } => {
                            entered[usize::from(origin == ExecutionOrigin::Internal)] += 1
                        }
                        ExecutionEvent::RunFinished { origin, result, .. } => {
                            finished[usize::from(origin == ExecutionOrigin::Internal)] += 1;
                            leaf_iterations.extend(result.unwrap().iterations.iter().cloned());
                        }
                        _ => {}
                    }
                },
            )
            .unwrap();
        assert_eq!(entered, finished);
        assert_eq!(
            entered[0], 1,
            "user spelling must not select internal provenance"
        );
        assert_eq!(entered[1] > 0, mode > 0);
        assert_eq!(
            outputs
                .iter()
                .filter(|(origin, _)| *origin == ExecutionOrigin::Requested)
                .count(),
            1
        );
        let output_iterations: Vec<_> = outputs
            .iter()
            .flat_map(|(_, output)| {
                let CommandOutput::RunSchedule(report) = output else {
                    panic!("run")
                };
                &report.iterations
            })
            .collect();
        assert_eq!(leaf_iterations.len(), output_iterations.len());
        for (leaf, aggregated) in leaf_iterations.iter().zip(output_iterations) {
            assert!(
                std::sync::Arc::ptr_eq(leaf, aggregated),
                "each measured iteration is owned exactly once"
            );
        }
        let raw: Vec<_> = outputs.into_iter().map(|(_, output)| output).collect();
        assert_eq!(output_kinds(&raw), output_kinds(&expected));
        let flags = |outputs: &[CommandOutput]| {
            outputs
                .iter()
                .map(|output| {
                    let CommandOutput::RunSchedule(report) = output else {
                        panic!("run")
                    };
                    (report.updated, report.can_stop)
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(flags(&raw), flags(&expected));
    }
}

#[test]
fn prove_helpers_are_internal_and_fail_keeps_output_suppression() {
    for proof_testing in [false, true] {
        let mut graph = EGraph::new_with_proofs();
        if proof_testing {
            graph = graph.with_proof_testing();
        }
        graph
            .parse_and_run_program(None, "(datatype Num (N i64)) (N 2)")
            .unwrap();
        let program = graph
            .parse_program(
                None,
                if proof_testing {
                    "(check (N 2)) (fail (prove (N 3)))"
                } else {
                    "(prove (N 2)) (fail (prove (N 3)))"
                },
            )
            .unwrap();
        let mut requested = vec![];
        let mut helper_entries = 0;
        graph
            .run_program_observed(
                program,
                &mut |cause, origin, output| {
                    if origin == ExecutionOrigin::Requested {
                        requested.push((cause, output));
                    }
                },
                &mut |_, event| {
                    if let ExecutionEvent::RunStarted { origin, schedule } = event {
                        assert_eq!(origin, ExecutionOrigin::Internal);
                        assert!(
                            schedule.is_none(),
                            "proof/check helper has no submitted schedule: {schedule:?}"
                        );
                        helper_entries += 1;
                    } else if let ExecutionEvent::RunFinished { schedule, .. }
                    | ExecutionEvent::LoopFinished { schedule, .. } = event
                    {
                        assert!(
                            schedule.is_none(),
                            "proof/check helper completion has no submitted schedule"
                        );
                    }
                },
            )
            .unwrap();
        assert!(helper_entries > 0);
        assert_eq!(requested.len(), 1);
        assert_eq!(requested[0].0, 0);
        assert!(matches!(requested[0].1, CommandOutput::ProveExists { .. }));
    }
    let mut graph = EGraph::default();
    let program = graph.parse_program(None, "(fail (run 1))").unwrap();
    let mut outputs = 0;
    let error = graph
        .run_program_observed(program, &mut |_, _, _| outputs += 1, &mut |_, _| {})
        .unwrap_err();
    assert!(matches!(error, Error::ExpectFail(_)));
    assert_eq!(outputs, 0);
}

#[test]
fn failed_entries_have_no_fabricated_measurements_and_checks_are_internal() {
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(None, "(datatype Num (N i64)) (rule () ((panic \"stop\")))")
        .unwrap();
    for (source, expected_origin, measured) in [
        ("(run 1)", ExecutionOrigin::Requested, false),
        ("(check (N 4))", ExecutionOrigin::Internal, true),
    ] {
        let program = graph.parse_program(None, source).unwrap();
        let mut entries = 0;
        let mut completions = 0;
        let error = graph
            .run_program_observed(
                program,
                &mut |_, _, _| panic!("failed command produced an output"),
                &mut |_, event| match event {
                    ExecutionEvent::RunStarted { origin, .. } => {
                        assert_eq!(origin, expected_origin);
                        entries += 1;
                    }
                    ExecutionEvent::RunFinished { origin, result, .. } => {
                        assert_eq!(origin, expected_origin);
                        assert_eq!(result.is_ok(), measured);
                        completions += 1;
                    }
                    _ => {}
                },
            )
            .unwrap_err();
        assert_eq!((entries, completions), (1, 1), "{error}");
    }
}

#[test]
fn until_query_is_measured_separately_from_the_skipped_authored_leaf() {
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(None, "(datatype Num (N i64)) (N 1)")
        .unwrap();
    let program = graph.parse_program(None, "(run 1 :until (N 1))").unwrap();
    let mut measured = vec![];
    graph
        .run_program_observed(program, &mut |_, _, _| {}, &mut |_, event| {
            if let ExecutionEvent::RunFinished { origin, result, .. } = event {
                measured.push((origin, result.unwrap().iterations.len()));
            }
        })
        .unwrap();
    assert_eq!(
        measured,
        [
            (ExecutionOrigin::Internal, 1),
            (ExecutionOrigin::Requested, 0)
        ]
    );
}

#[test]
fn ruleset_identity_survives_clone_and_push_pop_and_completed_prefixes() {
    let mut graph = EGraph::new_with_proofs();
    let setup = graph
        .parse_program(
            None,
            "(datatype Num (N i64) (Goal i64)) (N 1) (rule ((N x)) ((Goal x))) (run 1)",
        )
        .unwrap();
    let mut internal_rulesets = vec![];
    graph
        .run_program_observed(setup, &mut |_, _, _| {}, &mut |_, event| {
            if let ExecutionEvent::CommandCompleted {
                origin: ExecutionOrigin::Internal,
                command: ResolvedNCommand::AddRuleset(_, name),
            } = event
            {
                internal_rulesets.push(name.clone());
            }
        })
        .unwrap();
    assert_eq!(internal_rulesets.len(), 6);
    let maintenance = internal_rulesets.last().unwrap();
    for mut copy in [graph.clone(), graph] {
        copy.parse_and_run_program(None, "(push) (ruleset temporary) (pop)")
            .unwrap();
        copy.parse_and_run_program(None, "(ruleset temporary)")
            .unwrap();
        let mut schedule = copy.parse_program(None, "(run temporary 1)").unwrap();
        let Command::RunSchedule(egglog::ast::Schedule::Repeat(_, _, run)) = &mut schedule[0]
        else {
            panic!("repeat")
        };
        let egglog::ast::Schedule::Run(_, config) = run.as_mut() else {
            panic!("run")
        };
        config.ruleset = maintenance.clone();
        let mut entries = 0;
        copy.run_program_observed(schedule, &mut |_, _, _| {}, &mut |_, event| {
            if let ExecutionEvent::RunStarted { origin, .. } = event {
                assert_eq!(origin, ExecutionOrigin::Internal);
                entries += 1;
            }
        })
        .unwrap();
        assert!(entries > 0);

        // Install a failure in the actual internal ruleset found above. Its
        // name is only an identity reference, never a provenance heuristic.
        let mut poison = copy
            .parse_program(
                None,
                "(rule () ((panic \"maintenance failed\")) :ruleset temporary)",
            )
            .unwrap();
        let Command::Rule { rule } = &mut poison[0] else {
            panic!("rule")
        };
        rule.ruleset = maintenance.clone();
        copy.run_program(poison).unwrap();
        let mut definitions = copy.clone();
        let program = definitions
            .parse_program(None, "(ruleset first) (ruleset second)")
            .unwrap();
        let mut installed = vec![];
        let mut entered = 0;
        let mut completed = 0;
        let mut failed = 0;
        let error = definitions
            .run_program_observed(
                program,
                &mut |_, _, _| {},
                &mut |cause, event| match event {
                    ExecutionEvent::CommandCompleted {
                        origin: ExecutionOrigin::Requested,
                        command: ResolvedNCommand::AddRuleset(_, name),
                    } => installed.push((cause, name.clone())),
                    ExecutionEvent::RunStarted { origin, .. } => {
                        assert_eq!(origin, ExecutionOrigin::Internal);
                        entered += 1;
                    }
                    ExecutionEvent::RunFinished { origin, result, .. } => {
                        assert_eq!(origin, ExecutionOrigin::Internal);
                        if result.is_ok() {
                            completed += 1;
                        } else {
                            failed += 1;
                        }
                    }
                    _ => {}
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("maintenance failed"));
        assert_eq!(installed, [(0, "first".into())]);
        assert_eq!(failed, 1);
        assert_eq!(entered, completed + failed);
        assert!(completed > 0);
        let first = definitions.parse_program(None, "(run first 1)").unwrap();
        definitions.validate_program(first).unwrap();
        let second = definitions.parse_program(None, "(run second 1)").unwrap();
        assert!(
            matches!(definitions.validate_program(second), Err(Error::NoSuchRuleset(name, _)) if name == "second")
        );

        let program = copy
            .parse_program(None, "(prove-exists Goal) (N 2)")
            .unwrap();
        let mut outputs = vec![];
        let error = copy
            .run_program_observed(
                program,
                &mut |cause, origin, output| {
                    if origin == ExecutionOrigin::Requested {
                        assert_eq!(cause, 0);
                        outputs.push(output);
                    }
                },
                &mut |_, _| {},
            )
            .unwrap_err();
        assert!(error.to_string().contains("maintenance failed"));
        drop(copy);
        let [
            CommandOutput::ProveExists {
                proof_store,
                proof_id,
            },
        ] = outputs.as_slice()
        else {
            panic!("retained proof")
        };
        let proof = proof_store.get(*proof_id);
        assert_eq!(proof.lhs(), proof.rhs());
    }
}
