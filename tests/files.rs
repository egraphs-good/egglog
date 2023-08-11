use std::path::PathBuf;

use egglog::*;
use libtest_mimic::Trial;

#[derive(Clone)]
struct Run {
    path: PathBuf,
    test_proofs: bool,
    resugar: bool,
}

impl Run {
    fn run(&self) {
        let _ = env_logger::builder().is_test(true).try_init();
        let program_read = std::fs::read_to_string(&self.path)
            .unwrap_or_else(|err| panic!("Couldn't read {:?}: {:?}", self.path, err));
        let already_enables = program_read.starts_with("(set-option enable_proofs 1)");
        let program = if self.test_proofs && !already_enables {
            format!("(set-option enable_proofs 1)\n{}", program_read)
        } else {
            program_read
        };

        if !self.resugar {
            self.test_program(&program, "Top level error");
        } else if self.resugar {
            let mut egraph = EGraph::default();
            egraph.set_underscores_for_desugaring(3);
            let parsed = egraph.parse_program(&program).unwrap();
            // TODO can we test after term encoding instead?
            // last time I tried it spun out becuase
            // it adds term encoding to term encoding
            let desugared_str = egraph
                .process_commands(parsed, CompilerPassStop::TypecheckDesugared)
                .unwrap()
                .into_iter()
                .map(|x| x.resugar().to_string())
                .collect::<Vec<String>>()
                .join("\n");

            self.test_program(
                &desugared_str,
                "ERROR after parse, to_string, and parse again.",
            );
        }
    }

    fn test_program(&self, program: &str, message: &str) {
        let mut egraph = EGraph::default();
        if self.test_proofs {
            egraph.test_proofs = true;
        }
        egraph.set_underscores_for_desugaring(5);
        match egraph.parse_and_run_program(program) {
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
                    egraph.serialize_for_graphviz().to_dot();
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
            test_proofs: false,
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
