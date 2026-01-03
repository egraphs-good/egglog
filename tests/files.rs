use std::path::PathBuf;

use egglog::*;
use hashbrown::HashSet;
use libtest_mimic::Trial;

#[derive(Clone)]
struct Run {
    path: PathBuf,
    desugar: bool,
    term_encoding: bool,
}

impl Run {
    /// Convert CommandOutput vector to snapshot string, filtering non-deterministic content
    #[cfg(not(debug_assertions))]
    fn outputs_to_snapshot(&self, outputs: &[CommandOutput]) -> String {
        outputs
            .iter()
            .filter_map(|output| match output {
                // Skip OverallStatistics - contains non-deterministic Duration timing data
                CommandOutput::OverallStatistics(_) => None,
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
            let mut egraph = EGraph::default();
            let desugared_str = egraph
                .desugar_program(self.path.to_str().map(String::from), &program)
                .unwrap()
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("\n");

            self.test_program(
                None,
                &desugared_str,
                "ERROR after parse, to_string, and parse again.",
            )
        };

        // Debug mode enables parallelism which can lead to non-deterministic output ordering
        #[cfg(not(debug_assertions))]
        if !self.should_fail() && !self.should_skip_snapshot() && _outputs.iter().any(|o| !matches!(o, CommandOutput::RunSchedule(..))) {
            let snapshot_name = self.snapshot_name();
            let snapshot_content = self.outputs_to_snapshot(&_outputs);
            insta::assert_snapshot!(snapshot_name, snapshot_content);
        }
    }

    fn test_program(
        &self,
        filename: Option<String>,
        program: &str,
        message: &str,
    ) -> Vec<CommandOutput> {
        let mut egraph = EGraph::default();
        if self.term_encoding {
            egraph = egraph.with_term_encoding_enabled();
        }
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
                Ok(())
            }
        }
        Wrapper(self)
    }

    fn should_fail(&self) -> bool {
        self.path.to_string_lossy().contains("fail-typecheck")
    }

    #[cfg(not(debug_assertions))]
    fn should_skip_snapshot(&self) -> bool {
        // Skip tests with known non-deterministic output
        let filename = self.path.file_stem().unwrap().to_string_lossy();
        const SKIP_PATTERNS: [&str; 3] = [
            "extract-vec-bench",
            "python_array_optimize",
            "stresstest_large_expr",
        ];

        SKIP_PATTERNS.iter().any(|pat| filename.contains(pat))
            // Term encoding is currently causing non-deterministic database to be produced
            || (filename.contains("math-microbenchmark") && self.term_encoding)
    }

    #[cfg(not(debug_assertions))]
    fn snapshot_name(&self) -> String {
        let stem = self.path.file_stem().unwrap().to_string_lossy();
        let stem_clean = stem.replace(['.', '-', ' '], "_");

        let mut name = stem_clean.to_string();
        if self.desugar {
            name.push_str("_desugar");
        }
        if self.term_encoding {
            name.push_str("_term_encoding");
        }

        name
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
        };
        let should_fail = run.should_fail();

        push_trial(run.clone());
        if !should_fail {
            push_trial(Run {
                desugar: true,
                ..run.clone()
            });

            if file_supports_proofs(&run.path) {
                push_trial(Run {
                    term_encoding: true,
                    ..run.clone()
                });
            }
        }
    }

    trials
}

fn main() {
    let args = libtest_mimic::Arguments::from_args();
    let tests = generate_tests("tests/**/*.egg");
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
