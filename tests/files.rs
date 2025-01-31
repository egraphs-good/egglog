use std::path::PathBuf;

use egglog::*;
use libtest_mimic::Trial;

#[derive(Clone)]
struct Run {
    path: PathBuf,
    resugar: bool,
}

impl Run {
    fn run(&self) {
        let _ = env_logger::builder().is_test(true).try_init();
        let program = std::fs::read_to_string(&self.path)
            .unwrap_or_else(|err| panic!("Couldn't read {:?}: {:?}", self.path, err));

        if !self.resugar {
            self.test_program(
                self.path.to_str().map(String::from),
                &program,
                "Top level error",
            );
        } else {
            let mut egraph = EGraph::default();
            egraph.run_mode = RunMode::ShowDesugaredEgglog;
            egraph.set_reserved_symbol("__".into());
            let desugared_str = egraph
                .parse_and_run_program(self.path.to_str().map(String::from), &program)
                .unwrap()
                .join("\n");

            self.test_program(
                None,
                &desugared_str,
                "ERROR after parse, to_string, and parse again.",
            );
        }
    }

    fn test_program(&self, filename: Option<String>, program: &str, message: &str) {
        let mut egraph = EGraph::default();
        egraph.set_reserved_symbol("___".into());
        match egraph.parse_and_run_program(filename, program) {
            Ok(msgs) => {
                if self.should_fail() {
                    panic!(
                        "Program should have failed! Instead, logged:\n {}",
                        msgs.join("\n")
                    );
                } else {
                    for msg in msgs {
                        log::info!("  {}", msg);
                    }
                    // Test graphviz dot generation
                    let mut serialized = egraph.serialize(SerializeConfig {
                        max_functions: Some(40),
                        max_calls_per_function: Some(40),
                        ..Default::default()
                    });
                    serialized.to_dot();
                    // Also try splitting and inlining
                    serialized.split_classes(|id, _| egraph.from_node_id(id).is_primitive());
                    serialized.inline_leaves();
                    serialized.to_dot();

                    // TODO: remove this in phase 2
                    let max = egraph.num_tuples();
                    for (name, func) in &egraph.functions {
                        let old_backend = egraph.function_to_dag(*name, max).unwrap().0.len();
                        let new_backend = egraph.backend.table_size(func.new_backend_id);
                        assert_eq!(old_backend, new_backend);
                    }
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
                if self.0.resugar {
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
            resugar: false,
        };
        let should_fail = run.should_fail();

        push_trial(run.clone());
        if !should_fail {
            push_trial(Run {
                resugar: true,
                ..run.clone()
            });
        }
    }

    trials
}

fn main() {
    let args = libtest_mimic::Arguments::from_args();
    let tests = generate_tests("tests/**/*.egg");
    libtest_mimic::run(&args, tests).exit();
}
