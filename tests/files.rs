use std::path::PathBuf;

use egglog::{ast::sanitize_internal_names, *};
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
    fn run(&self) {
        let _ = env_logger::builder().is_test(true).try_init();
        let program = std::fs::read_to_string(&self.path)
            .unwrap_or_else(|err| panic!("Couldn't read {:?}: {:?}", self.path, err));

        if !self.desugar {
            self.test_program(
                self.path.to_str().map(String::from),
                &program,
                "Top level error",
            );
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
            );
        }
    }

    fn egraph(&self) -> EGraph {
        if self.proofs {
            EGraph::new_with_proofs()
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

    fn test_program(&self, filename: Option<String>, program: &str, message: &str) {
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
                    for msg in msgs {
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
                }
            }
            Err(err) => {
                if !self.should_fail() {
                    panic!("{}: {err}", message)
                }
            }
        };
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

        push_trial(run.clone());
        if !should_fail {
            push_trial(Run {
                desugar: true,
                ..run.clone()
            });

            // TODO improve performance of proof mode to enable math_microbenchmark tests
            if file_supports_proofs(&run.path) {
                push_trial(Run {
                    term_encoding: true,
                    ..run.clone()
                });

                if !run.path.to_string_lossy().contains("math-microbenchmark") {
                    push_trial(Run {
                        proofs: true,
                        ..run.clone()
                    });

                    // Desugar with proof mode, then run normally. Tests parsing and running proof-instrumented egglog.
                    push_trial(Run {
                        proofs: true,
                        desugar: true,
                        ..run.clone()
                    });
                }
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
