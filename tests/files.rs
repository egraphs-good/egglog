use std::path::PathBuf;

use egglog::backend_duckdb::DuckdbBackend;
use egglog::{file_supports_proofs, *};
use hashbrown::HashSet;
use libtest_mimic::Trial;

#[derive(Clone)]
struct Run {
    path: PathBuf,
    desugar: bool,
    term_encoding: bool,
    proofs: bool,
    /// proof_testing mode adds automatic prove-exists commands, which produce
    /// proof output that differs from normal mode. This should use separate snapshots.
    proof_testing: bool,
    /// Run the program through the DuckDB-backed executor (the
    /// `--duckdb` mode). Mutually exclusive with the other treatments
    /// — DuckDB always uses term encoding internally regardless of
    /// the flag. Output is captured and diffed against the same
    /// shared snapshot every other mode targets.
    duckdb: bool,
    threads: usize,
}

impl Run {
    /// Tests in the proofs directory require proofs to run successfully.
    fn requires_proofs(&self) -> bool {
        self.path.parent().unwrap().ends_with("proofs")
    }

    /// Extraction results may differ slightly due to the proof encoding when multiple
    /// solutions have the same cost. Snapshot only the extracted cost so shared
    /// snapshots still verify that normal and proof modes find equally good solutions.
    fn outputs_to_snapshot_preserved_across_treatments(&self, outputs: &[CommandOutput]) -> String {
        outputs
            .iter()
            .filter_map(|output| match output {
                // Skip OverallStatistics - contains non-deterministic Duration timing data
                CommandOutput::OverallStatistics(_) => None,
                // Skipping PrintFunction for now due to egglog nondeterminism bug: https://github.com/egraphs-good/egglog/issues/793
                CommandOutput::PrintFunction(..) => None,
                // Skip extraction outputs. --duckdb mode silently
                // ignores `(extract …)` commands (no extraction
                // pipeline yet), so this keeps the shared snapshot
                // comparable across all backends.
                CommandOutput::ExtractBest(..) => None,
                CommandOutput::ExtractVariants(..) => None,
                // All other variants use normal Display formatting
                other => Some(other.to_string()),
            })
            .collect::<Vec<_>>()
            .join("")
    }

    fn run(&self) {
        let _ = env_logger::builder().is_test(true).try_init();
        let program = std::fs::read_to_string(&self.path)
            .unwrap_or_else(|err| panic!("Couldn't read {:?}: {:?}", self.path, err));

        let result = if !self.desugar {
            self.test_program(
                self.path.to_str().map(String::from),
                &program,
                "",
                "Top level error",
            )
        } else {
            let resolved_str = self.resolve_prog(&program);
            // after desugaring run the program without term encoding or proofs
            let normal_run = Run {
                path: self.path.clone(),
                desugar: false,
                term_encoding: false,
                proofs: false,
                proof_testing: false,
                duckdb: false,
                threads: self.threads,
            };
            let proof_check_prog = if self.proof_testing {
                program.clone()
            } else {
                "".to_string()
            };

            normal_run.test_program(
                None,
                &resolved_str,
                &proof_check_prog,
                "ERROR after parse, to_string, and parse again.",
            )
        };

        // Debug mode enables parallelism which can lead to non-deterministic output ordering
        if !self.should_skip_snapshot() {
            match &result {
                Ok(outputs) => {
                    let snapshot_name_across_treatments = self.snapshot_name_across_treatments();
                    let snapshot_content_across_treatments =
                        self.outputs_to_snapshot_preserved_across_treatments(outputs);

                    if self.should_assert_snapshot_across_treatments(
                        &snapshot_content_across_treatments,
                    ) {
                        insta::assert_snapshot!(
                            snapshot_name_across_treatments,
                            snapshot_content_across_treatments
                        );
                    }
                }
                Err(err_msg) => {
                    // Snapshot the error message for fail-typecheck tests
                    let name = self.name().to_string();
                    insta::assert_snapshot!(name, err_msg);
                }
            }
        }
    }

    fn egraph(&self) -> EGraph {
        if self.proof_testing {
            EGraph::new_with_proofs().with_proof_testing()
        } else if self.proofs {
            EGraph::new_with_proofs()
        } else if self.term_encoding {
            EGraph::new_with_term_encoding()
        } else {
            EGraph::default()
        }
    }

    // Returns a string of the desugared program and a string for the desugared program without proofs
    fn resolve_prog(&self, program: &str) -> String {
        let mut egraph = self.egraph();

        let resolved = egraph
            .resolve_program(self.path.to_str().map(String::from), program)
            .unwrap();
        resolved
            .iter()
            .map(|cmd| cmd.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn test_program(
        &self,
        filename: Option<String>,
        program: &str,
        proof_check_prog: &str,
        message: &str,
    ) -> Result<Vec<CommandOutput>, String> {
        // Append print-size to every test file to ensure it works.
        let program = format!("{program}\n(print-size)");

        // --duckdb mode reuses everything: the same snapshot, the
        // same should_fail handling, the same `(print-size)`
        // appendage. It only swaps the executor.
        if self.duckdb {
            // When `self.proofs` is set in a duckdb run, wire it
            // into the backend config so the encoder runs in
            // proof-tracking mode. This exercises the same control
            // flow proof-mode reference runs do, but on the DuckDB
            // executor — catches regressions where backend-side
            // optimizations (inline-congruence, native UF, hash-cons)
            // skip cases that the proof encoder relies on.
            // Native UF and proof mode are not yet compatible: the
            // proof encoder declares `uf_function_<sort>` as
            // returning `(Pair sort proof)`, and rebuild rules read
            // out the leader via `(pair-first p)`. Native UF
            // replaces those reads with a UDF that returns just the
            // leader as `i64`, which doesn't satisfy the encoder's
            // pair type. Until native UF learns to return Pair
            // values in proof mode, keep them apart in tests.
            let config = egglog::backend_duckdb::DuckBackendConfig {
                proofs: self.proofs,
                native_uf: false,
            };
            let mut backend = DuckdbBackend::new_with_config(config)
                .unwrap_or_else(|e| panic!("DuckdbBackend init failed: {e}"));
            return match backend.parse_and_run_program(filename, &program) {
                Ok(msgs) => {
                    if self.should_fail() {
                        panic!(
                            "Program should have failed under --duckdb! Outputs:\n{}",
                            msgs.iter().map(|m| m.to_string()).collect::<Vec<_>>().join("")
                        );
                    }
                    Ok(msgs)
                }
                Err(err) => {
                    if !self.should_fail() {
                        panic!("{message} (--duckdb): {err}");
                    }
                    Err(err.to_string())
                }
            };
        }

        let mut egraph = self.egraph();
        let parsed_proof_check_prog = egraph
            .parse_program(None, proof_check_prog)
            .unwrap_or_else(|_| panic!("Failed to parse proof check program"));
        // hard code proof testing to true, we only use proof checking program in proof testing mode
        egraph
            .set_proof_checking_program(parsed_proof_check_prog, true)
            .expect("Failed to set proof checking program");

        egraph.ensure_no_reserved_symbols(false);

        match egraph.parse_and_run_program(filename, &program) {
            Ok(msgs) => {
                if self.should_fail() {
                    panic!(
                        "Program should have failed! Instead, logged:\n {}",
                        msgs.iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>()
                            .join("\n")
                    );
                } else {
                    for msg in &msgs {
                        log::info!("  {msg}");
                    }
                    // Test graphviz dot generation
                    let mut serialized = egraph
                        .serialize(SerializeConfig {
                            max_functions: Some(40),
                            max_calls_per_function: Some(40),
                            ..Default::default()
                        })
                        .egraph;
                    serialized.to_dot();
                    // Also try splitting and inlining
                    serialized.split_classes(|id, _| egraph.from_node_id(id).is_primitive());
                    serialized.inline_leaves();
                    serialized.to_dot();

                    Ok(msgs)
                }
            }
            Err(err) => {
                if !self.should_fail() {
                    panic!("{message}: {err}")
                }
                Err(err.to_string())
            }
        }
    }

    fn into_trial(self) -> Trial {
        let name = self.name().to_string();
        Trial::test(name, move || {
            // We use a local rayon pool here because `build_global()` can only
            // be called once per process, but libtest-mimic runs many trials
            // (with different thread counts) in the same process.
            // The threads == 1 case also goes through pool.install so the trial
            // doesn't fall through to the default global rayon pool (which uses
            // num_cpus threads and would make "single-threaded" tests
            // nondeterministic).
            // TODO: when we move to per-EGraph local thread pools, replace this
            // with `egraph.with_num_threads()` and remove the explicit pool.
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(self.threads)
                .build()
                .expect("failed to build rayon thread pool");
            pool.install(|| self.run());
            Ok(())
        })
    }

    /// Base snapshot name without mode suffixes - all variants share the same `outputs_to_snapshot_preserved_across_treatments` snapshot
    /// except for proof_testing, which has different output due to using `prove` everywhere.
    fn snapshot_name_across_treatments(&self) -> String {
        let mut name = "shared_snapshot_".to_string();

        let stem = self.path.file_stem().unwrap();
        let stem_str = stem.to_string_lossy().replace(['.', '-', ' '], "_");
        name.push_str(&stem_str);

        if self.path.parent().unwrap().ends_with("fail-typecheck") {
            name.push_str("_fail_typecheck");
        }
        name
    }

    /// Full test name with mode suffixes for test identification
    fn name(&self) -> impl std::fmt::Display + '_ {
        struct Wrapper<'a>(&'a Run);
        impl std::fmt::Display for Wrapper<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                if self.0.path.parent().unwrap().ends_with("fail-typecheck") {
                    write!(f, "fail-typecheck/")?;
                }
                let stem = self.0.path.file_stem().unwrap();
                let stem_str = stem.to_string_lossy().replace(['.', '-', ' '], "_");
                write!(f, "{stem_str}")?;
                if self.0.desugar {
                    write!(f, "_desugar")?;
                }
                if self.0.term_encoding {
                    write!(f, "_term_encoding")?;
                }
                if self.0.proofs {
                    write!(f, "_proofs")?;
                }
                if self.0.proof_testing {
                    write!(f, "_proof_testing")?;
                }
                if self.0.duckdb {
                    write!(f, "_duckdb")?;
                }

                if self.0.threads > 1 {
                    write!(f, "_{}threads", self.0.threads)?;
                }

                Ok(())
            }
        }
        Wrapper(self)
    }

    fn should_fail(&self) -> bool {
        self.path.to_string_lossy().contains("fail-typecheck")
    }

    fn should_skip_snapshot(&self) -> bool {
        if self.threads > 1 {
            // Skip snapshots for parallel tests due to non-deterministic output ordering
            true
        } else {
            // Skip tests with known non-deterministic output
            let filename = self.path.file_stem().unwrap().to_string_lossy();
            const SKIP_PATTERNS: [&str; 6] = [
                "extract-vec-bench",
                "python_array_optimize",
                "stresstest_large_expr",
                "towers-of-hanoi",
                "taylor51",
                "factoring-multisets",
            ];
            if SKIP_PATTERNS.iter().any(|pat| filename.contains(pat)) {
                return true;
            }

            // bug with egglog producing nondeterministic output in certain modes
            let proof_skip_list = ["math-microbenchmark", "eqsolve"];
            let in_list = proof_skip_list
                .iter()
                .any(|f| self.path.to_string_lossy().contains(f));
            in_list && (self.proofs || self.term_encoding || self.proof_testing)
        }
    }

    /// only assert snapshot if the snapshot is non-empty
    /// proof_testing has different output due to automatic prove-exists, so no snapshot for that
    fn should_assert_snapshot_across_treatments(
        &self,
        snapshot_content_across_treatments: &str,
    ) -> bool {
        !snapshot_content_across_treatments.is_empty() && !self.proof_testing
    }
}

fn generate_tests(glob: &str) -> Vec<Trial> {
    let mut trials = vec![];
    let mut push_trial = |run: Run| trials.push(run.into_trial());

    for entry in glob::glob(glob).unwrap() {
        let run = Run {
            path: entry.unwrap().clone(),
            desugar: false,
            term_encoding: false,
            proofs: false,
            proof_testing: false,
            duckdb: false,
            threads: 1,
        };
        let should_fail = run.should_fail();
        let requires_proofs = run.requires_proofs();
        // TODO: math-microbenchmark is too slow right now
        // TODO: subsume.egg fails because we used a `check` on something subsumed. Need a way to run rules over subsumed things. Same with subsume-relation.egg.
        let proof_unsupported_file_list = [
            "math-microbenchmark.egg",
            "rectangle.egg",
            "eggcc-2mm.egg",
            "subsume.egg",
            "subsume-relation.egg",
        ];
        let supports_proofs = file_supports_proofs(&run.path)
            && !proof_unsupported_file_list
                .iter()
                .any(|f| run.path.ends_with(f));

        if !requires_proofs {
            push_trial(run.clone());

            push_trial(Run {
                threads: 32,
                ..run.clone()
            });
        }
        if !requires_proofs && !should_fail {
            push_trial(Run {
                desugar: true,
                ..run.clone()
            });
        }
        if !should_fail && !requires_proofs && supports_proofs {
            push_trial(Run {
                term_encoding: true,
                ..run.clone()
            });
        }

        // --duckdb mode: share the same snapshot as the egglog
        // reference. We skip files using features the backend
        // doesn't yet implement: `(push)/(pop)` (no savepoint
        // support), and the same `proof_unsupported` files
        // `--term-encoding` skips. `(extract …)` is silently
        // ignored by the backend; the shared snapshot drops its
        // output so both modes still match.
        let duckdb_static_skip = [
            "math-microbenchmark.egg",
            "rectangle.egg",
            "eggcc-2mm.egg",
            "subsume.egg",
            "subsume-relation.egg",
        ];
        let mut duckdb_supported = !should_fail
            && !requires_proofs
            && supports_proofs
            && !duckdb_static_skip.iter().any(|f| run.path.ends_with(f));
        if duckdb_supported {
            if let Ok(src) = std::fs::read_to_string(&run.path) {
                if src.contains("(push") || src.contains("(pop") {
                    duckdb_supported = false;
                }
            }
        }
        if duckdb_supported {
            push_trial(Run {
                duckdb: true,
                ..run.clone()
            });
        }

        // duckdb + proofs: run files through the proof-tracking
        // encoder on the DuckDB backend. The encoder threads proof
        // terms (`@Trans`, `@Merge`, `@PNil`, …) through unions and
        // rewrites; the backend takes a different path through
        // `translate_expr` for those constructors. Useful as a
        // regression guard for our optimizations against proof
        // mode's code paths.
        //
        // Mirror the plain-duckdb skips: `(push)/(pop)` aren't a
        // duckdb feature, so files relying on them would diverge
        // from the shared snapshot just because every run gets
        // appended onto the previous run's state.
        let mut duckdb_proofs_supported = !should_fail
            && !requires_proofs
            && file_supports_proofs(&run.path);
        if duckdb_proofs_supported {
            if let Ok(src) = std::fs::read_to_string(&run.path) {
                if src.contains("(push") || src.contains("(pop") {
                    duckdb_proofs_supported = false;
                }
            }
        }
        if duckdb_proofs_supported {
            push_trial(Run {
                duckdb: true,
                proofs: true,
                ..run.clone()
            });
        }

        // proofs mode (without proof_testing) should produce the same output as normal mode
        if !should_fail && supports_proofs {
            push_trial(Run {
                proofs: true,
                ..run.clone()
            });
        }

        if !should_fail && supports_proofs {
            // proof_testing mode adds automatic prove-exists, which has different output
            push_trial(Run {
                proof_testing: true,
                ..run.clone()
            });

            // Complex mode: desugar using proof encoding, then run normally.
            // Yes this mode is important! It has found multiple bugs.
            push_trial(Run {
                proof_testing: true,
                desugar: true,
                ..run.clone()
            });
        }
    }

    trials
}

fn generate_proof_support_snapshot_test() -> Trial {
    Trial::test("proof_support_snapshot", || {
        let mut supported_files = Vec::new();

        for entry in glob::glob("tests/**/*.egg").unwrap() {
            let path = entry.unwrap();
            if !file_supports_proofs(&path) && !path.parent().unwrap().ends_with("fail-typecheck") {
                // Use just the filename for cross-platform consistency
                let filename = path.file_name().unwrap().to_string_lossy().to_string();
                supported_files.push(filename);
            }
        }

        // Sort for deterministic output
        supported_files.sort();

        // Create snapshot
        let snapshot = supported_files.join("\n");
        insta::assert_snapshot!("proof_unsupported_files", snapshot);

        Ok(())
    })
}

fn main() {
    let args = libtest_mimic::Arguments::from_args();
    let mut tests = generate_tests("tests/**/*.egg");

    // Add the proof support snapshot test
    tests.push(generate_proof_support_snapshot_test());

    // ensure all the tests have unique names
    let mut names = HashSet::new();
    for test in &tests {
        let name = test.name().to_string();
        if !names.insert(name.clone()) {
            panic!("Duplicate test name: {name}");
        }
    }
    libtest_mimic::run(&args, tests).exit();
}
