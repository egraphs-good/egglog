#![cfg(feature = "comparison")]
use egglog::{CommandOutput, EGraph};
use egraph_comparison::{Database, certificate, compare, verify};
use std::{fs, path::Path, time::Instant};

const GOLDENS: &[&str] = &["eqsat-basic", "path", "fibonacci"];

fn run(program: &str, threads: usize) -> (Database, Vec<CommandOutput>) {
    let mut graph = EGraph::default().with_num_threads(threads);
    let outputs = graph.parse_and_run_program(None, program).unwrap();
    (graph.serialize_for_comparison().unwrap(), outputs)
}

// An explicit update mode, separate from comparison: ordinary test runs never
// rewrite accepted databases. Each threading treatment uses the same baseline.
#[test]
fn golden_databases_match_sequential_and_parallel_runs() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let baseline_dir = root.join("tests/comparison/baselines");
    for name in GOLDENS {
        let program = fs::read_to_string(root.join(format!("tests/web-demo/{name}.egg"))).unwrap();
        let baseline_path = baseline_dir.join(format!("{name}.json"));
        if std::env::var("EGGLOG_UPDATE_COMPARISON_SNAPSHOTS").as_deref() == Ok("1") {
            let (database, _) = run(&program, 1);
            fs::write(
                &baseline_path,
                serde_json::to_string_pretty(&database).unwrap() + "\n",
            )
            .unwrap();
        }
        let expected: Database =
            serde_json::from_slice(&fs::read(&baseline_path).unwrap()).unwrap();
        for threads in [1, 4, 32] {
            let (actual, _) = run(&program, threads);
            let result = compare(&expected, &actual).unwrap();
            if !result.database_equal {
                let witness = certificate(&expected, &actual).unwrap().unwrap();
                assert!(verify(&witness, &expected, &actual).unwrap());
                let output = root.join("target/comparison-snapshots");
                fs::create_dir_all(&output).unwrap();
                let stem = format!("{name}-{threads}threads");
                let actual_path = output.join(format!("{stem}.actual.json"));
                let witness_path = output.join(format!("{stem}.certificate.json"));
                fs::write(&actual_path, serde_json::to_vec_pretty(&actual).unwrap()).unwrap();
                fs::write(&witness_path, serde_json::to_vec_pretty(&witness).unwrap()).unwrap();
                panic!(
                    "{name} ({threads} threads): {result:?}\nbaseline: {}\nactual: {}\ncertificate: {}",
                    baseline_path.display(),
                    actual_path.display(),
                    witness_path.display()
                );
            }
        }
    }
}

#[test]
fn database_snapshots_find_changes_that_keep_output_snapshots_identical() {
    let cases = [
        (
            "(function f (i64) i64 :no-merge)",
            "(set (f 0) 1)",
            "(set (f 0) 2)",
        ),
        ("(datatype E (Num i64))", "(Num 1)", "(Num 2)"),
        (
            "(datatype E (A) (B) (C)) (A) (B) (C)",
            "(union (A) (B))",
            "(union (A) (C))",
        ),
    ];
    for (header, a, b) in cases {
        let (left, a) = run(&format!("{header} {a} (print-size)"), 1);
        let (right, b) = run(&format!("{header} {b} (print-size)"), 4);
        assert_eq!(
            CommandOutput::snapshot_stable_under_proof_encoding(&a),
            CommandOutput::snapshot_stable_under_proof_encoding(&b)
        );
        assert!(!compare(&left, &right).unwrap().database_equal);
        let witness = certificate(&left, &right).unwrap().unwrap();
        assert!(verify(&witness, &left, &right).unwrap());
    }
}

// A bounded, reproducible feasibility survey, not a performance benchmark or a
// silent skip list. Unsupported cases are reported for the migration design.
#[test]
#[ignore = "manual snapshot migration survey; reports supported and blocked cases"]
fn survey_snapshot_candidates() {
    use std::io::Write;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let paths = [
        "web-demo/eqsat-basic",
        "web-demo/path",
        "web-demo/fibonacci",
        "web-demo/points-to",
        "web-demo/bignum",
        "web-demo/subsume",
        "web-demo/set",
        "web-demo/multiset",
        "vec",
        "map",
        "pair",
        "web-demo/unstable-fn",
    ];
    let mut report = Vec::new();
    for path in paths {
        let program = fs::read_to_string(root.join(format!("tests/{path}.egg"))).unwrap();
        let mut baseline = None;
        for mode in ["normal", "32threads", "desugar", "term_encoding", "proofs"] {
            let start = Instant::now();
            let result = (|| -> Result<Database, String> {
                let mut graph = match mode {
                    "term_encoding" => EGraph::new_with_term_encoding(),
                    "proofs" => EGraph::new_with_proofs(),
                    "32threads" => EGraph::default().with_num_threads(32),
                    _ => EGraph::default(),
                };
                let program = if mode == "desugar" {
                    let commands = graph
                        .resolve_program(None, &program)
                        .map_err(|e| e.to_string())?;
                    let result = commands
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("\n");
                    graph = EGraph::default();
                    graph.ensure_no_reserved_symbols(false);
                    result
                } else {
                    program.clone()
                };
                graph
                    .parse_and_run_program(None, &program)
                    .map_err(|e| e.to_string())?;
                graph.serialize_for_comparison().map_err(|e| e.to_string())
            })();
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            let row = match result {
                Ok(database) => {
                    let compare_start = Instant::now();
                    let comparison = baseline
                        .as_ref()
                        .map(|expected| compare(expected, &database).unwrap());
                    let compare_ms = compare_start.elapsed().as_secs_f64() * 1000.0;
                    let row = serde_json::json!({"case":path,"mode":mode,"classes":database.classes.len(),"rows":database.rows.len(),"bytes":serde_json::to_vec_pretty(&database).unwrap().len(),"run_export_ms":elapsed_ms,"compare_ms":compare_ms,"comparison":comparison});
                    if mode == "normal" {
                        baseline = Some(database);
                    }
                    row
                }
                Err(error) => {
                    serde_json::json!({"case":path,"mode":mode,"error":error,"run_export_ms":elapsed_ms})
                }
            };
            report.push(row);
        }
    }
    let json = serde_json::to_string_pretty(&report).unwrap();
    if let Ok(path) = std::env::var("EGGLOG_COMPARISON_SURVEY_OUTPUT") {
        fs::write(path, &json).unwrap();
    }
    writeln!(std::io::stdout().lock(), "{json}").unwrap();
}
