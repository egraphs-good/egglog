use egglog::{CommandOutput, EGraph, Error, proof::ProveExistsError};
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
