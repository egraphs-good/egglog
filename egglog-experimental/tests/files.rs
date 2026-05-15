use std::path::PathBuf;

use egglog::ast::sanitize_internal_names;
use egglog_experimental::*;
use libtest_mimic::Trial;

#[derive(Clone)]
struct Run {
    path: PathBuf,
    desugar: bool,
}

impl Run {
    fn run(&self) {
        let program = std::fs::read_to_string(&self.path)
            .unwrap_or_else(|err| panic!("Couldn't read {:?}: {:?}", self.path, err));

        if !self.desugar {
            self.test_program(
                self.path.to_str().map(String::from),
                &program,
                "Top level error",
            );
        } else {
            let mut egraph = new_experimental_egraph();
            let resolved = egraph
                .resolve_program(self.path.to_str().map(String::from), &program)
                .unwrap();
            let desugared_str = sanitize_internal_names(&resolved)
                .iter()
                .map(|cmd| cmd.to_string())
                .collect::<Vec<_>>()
                .join("\n");

            self.test_program(
                None,
                &desugared_str,
                "ERROR after parse, to_string, and parse again.",
            );
        }
    }

    fn test_program(&self, filename: Option<String>, program: &str, message: &str) {
        let mut egraph = new_experimental_egraph();
        match egraph.parse_and_run_program(filename, program) {
            Ok(outputs) => {
                if self.should_fail() {
                    panic!(
                        "Program should have failed! Instead, logged:\n {}",
                        outputs
                            .iter()
                            .map(|output| output.to_string())
                            .collect::<Vec<_>>()
                            .join("\n")
                    );
                } else {
                    for output in outputs {
                        print!("  {}", output);
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
                let stem = self.0.path.file_stem().unwrap();
                let stem_str = stem.to_string_lossy().replace(['.', '-', ' '], "_");
                write!(f, "{stem_str}")?;
                if self.0.desugar {
                    write!(f, "_resugar")?;
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
        };
        // let should_fail = run.should_fail();

        push_trial(run.clone());

        // Temporarily removed due to egglog changes. TODO: uncomment once egglog desugar is fixed
        // if !should_fail {
        //     push_trial(Run {
        //         desugar: true,
        //         ..run.clone()
        //     });
        // }
    }

    trials
}

fn main() {
    let args = libtest_mimic::Arguments::from_args();
    let tests = generate_tests("tests/**/*.egg");
    libtest_mimic::run(&args, tests).exit();
}
