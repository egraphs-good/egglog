use std::path::PathBuf;

use egglog::{ast::sanitize_internal_names, file_supports_proofs, *};
use hashbrown::HashSet;
use libtest_mimic::Trial;

#[derive(Clone)]
struct Run {
    path: PathBuf,
    desugar: bool,
    term_encoding: bool,
    proofs: bool,
}

impl Run {
    /// Tests in the proofs directory require proofs to run successfully.
    fn requires_proofs(&self) -> bool {
        self.path.parent().unwrap().ends_with("proofs")
    }

    /// Convert CommandOutput vector to snapshot string, filtering non-deterministic content
    fn outputs_to_snapshot(&self, outputs: &[CommandOutput]) -> String {
        outputs
            .iter()
            .filter_map(|output| match output {
                // Skip OverallStatistics - contains non-deterministic Duration timing data
                CommandOutput::OverallStatistics(_) => None,
                // Skpping PrintFunction for now due to egglog nondeterminism bug: https://github.com/egraphs-good/egglog/issues/793
                CommandOutput::PrintFunction(..) => None,
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

        let _outputs = if !self.desugar {
            self.test_program(
                self.path.to_str().map(String::from),
                &program,
                "Top level error",
            )
        } else {
            let desugared_str = self.desugar_program(&program);
            // after desugaring run the program without term encoding or proofs
            let normal_run = Run {
                path: self.path.clone(),
                desugar: false,
                term_encoding: false,
                proofs: false,
            };

            normal_run.test_program(
                None,
                &desugared_str,
                "ERROR after parse, to_string, and parse again.",
            )
        };

        // Debug mode enables parallelism which can lead to non-deterministic output ordering
        if !self.should_fail()
            && !self.should_skip_snapshot()
            && _outputs
                .iter()
                .any(|o| !matches!(o, CommandOutput::RunSchedule(..)))
        {
            let snapshot_name = self.name().to_string();
            let snapshot_content = self.outputs_to_snapshot(&_outputs);
            insta::assert_snapshot!(snapshot_name, snapshot_content);
        }
    }

    fn egraph(&self) -> EGraph {
        if self.proofs {
            EGraph::new_with_proofs().with_proof_testing()
        } else if self.term_encoding {
            EGraph::new_with_term_encoding()
        } else {
            EGraph::default()
        }
    }

    fn desugar_program(&self, program: &str) -> String {
        let mut egraph = self.egraph();
        sanitize_internal_names(
            &egraph
                .desugar_program(self.path.to_str().map(String::from), program)
                .unwrap(),
        )
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join("\n")
    }

    fn test_program(
        &self,
        filename: Option<String>,
        program: &str,
        message: &str,
    ) -> Vec<CommandOutput> {
        let mut egraph = self.egraph();

        match egraph.parse_and_run_program(filename, program) {
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
                        log::info!("  {}", msg);
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

                    msgs
                }
            }
            Err(err) => {
                if !self.should_fail() {
                    panic!("{}: {err}", message)
                }
                vec![]
            }
        }
    }

    fn into_trial(self) -> Trial {
        let name = self.name().to_string();
        Trial::test(name, move || {
            self.run();
            Ok(())
        })
    }

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
                Ok(())
            }
        }
        Wrapper(self)
    }

    fn should_fail(&self) -> bool {
        self.path.to_string_lossy().contains("fail-typecheck")
    }

    fn should_skip_snapshot(&self) -> bool {
        // in parallel mode always skip
        #[cfg(debug_assertions)]
        {
            true
        }
        // in non-parallel mode, selectively skip
        #[cfg(not(debug_assertions))]
        {
            // Skip proof tests unless they require proofs
            if self.proofs && !self.requires_proofs() {
                return true;
            }
            (filename.contains("math-microbenchmark") && self.term_encoding)
        }
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
        };
        let should_fail = run.should_fail();
        let requires_proofs = run.requires_proofs();
        // TODO: math-microbenchmark is too slow right now
        // TODO: subsume.egg fails because we used a `check` on something subsumed. Need a way to run rules over subsumed things. Same with subsume-relation.egg.
        let proof_unsupported_file_list = [
            "math-microbenchmark.egg",
            "subsume.egg",
            "subsume-relation.egg",
        ];
        let supports_proofs = file_supports_proofs(&run.path)
            && !proof_unsupported_file_list
                .iter()
                .any(|f| run.path.ends_with(f));

        if !requires_proofs {
            push_trial(run.clone());
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

        if !should_fail && supports_proofs {
            push_trial(Run {
                proofs: true,
                ..run.clone()
            });
        }

        // TODO- running desugar + proofs fails on `prove-exists` commands. We can fix this by tying proof tables to constructors in the egglog itself.
        /*if !should_fail
            && supports_proofs
            && !run.path.to_string_lossy().contains("math-microbenchmark")
            && !requires_proofs
        {
            push_trial(Run {
                proofs: true,
                desugar: true,
                ..run.clone()
            });
        }*/
    }

    trials
}

fn generate_proof_support_snapshot_test() -> Trial {
    Trial::test("proof_support_snapshot", || {
        let mut supported_files = Vec::new();

        for entry in glob::glob("tests/**/*.egg").unwrap() {
            let path = entry.unwrap();
            if !file_supports_proofs(&path) {
                // Convert to relative path for consistent snapshots
                let relative = path.strip_prefix("tests/").unwrap_or(&path);
                supported_files.push(relative.to_string_lossy().to_string());
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
            panic!("Duplicate test name: {}", name);
        }
    }
    libtest_mimic::run(&args, tests).exit();
}
