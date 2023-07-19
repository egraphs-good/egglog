use std::path::PathBuf;

use egglog::*;
use libtest_mimic::Trial;

#[derive(Clone)]
struct Run {
    path: PathBuf,
    test_proofs: bool,
}

impl Run {
    fn run(&self) {
        let _ = env_logger::builder().is_test(true).try_init();
        let program = std::fs::read_to_string(&self.path)
            .unwrap_or_else(|err| panic!("Couldn't read {:?}: {:?}", self.path, err));
        self.test_program(&program, "Top level error");
        if !self.should_fail() {
            let mut egraph = EGraph::default();
            egraph.set_underscores_for_desugaring(4);
            let parsed = egraph.parse_program(&program).unwrap();
            let desugared_str = egraph
                .process_commands(parsed)
                .unwrap()
                .into_iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join("\n");

            println!("{}", desugared_str);

            self.test_program(
                &desugared_str,
                &format!(
                    "Program:\n{}\n ERROR after parse, to_string, and parse again.",
                    desugared_str
                ),
            );
        }
    }

    fn test_program(&self, program: &str, message: &str) {
        let mut egraph = EGraph::default();
        egraph.set_underscores_for_desugaring(5);
        if self.test_proofs {
            egraph.enable_proofs();
            egraph.test_proofs = true;
        }
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
                if self.0.test_proofs {
                    write!(f, "_with_proofs")?;
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
        };
        let name = run.name().to_string();

        push_trial(run.clone());

        // make a test with proofs enabled
        // TODO: re-enable herbie, unsound, and eqsolve when proof extraction is faster
        let banned = [
            "herbie",
            "repro_unsound",
            "eqsolve",
            "before_proofs",
            "lambda",
        ];
        if !banned.contains(&name.as_str()) {
            push_trial(Run {
                test_proofs: true,
                ..run
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
