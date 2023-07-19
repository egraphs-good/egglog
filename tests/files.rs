use std::path::PathBuf;

use egglog::*;
use libtest_mimic::Trial;

#[derive(Clone)]
struct Run {
    path: PathBuf,
    test_proofs: bool,
    should_fail: bool,
}

impl Run {
    fn run(&self) {
        let _ = env_logger::builder().is_test(true).try_init();
        let program = std::fs::read_to_string(&self.path).unwrap();
        self.test_program(&program, "Top level error");

        if !self.should_fail {
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
                if self.should_fail {
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
                if !self.should_fail {
                    panic!("{}: {err}", message)
                }
            }
        }
    }
}

fn generate_tests(glob: &str) -> Vec<Trial> {
    let mut trials = vec![];
    let mut mk_trial = |name: String, run: Run| {
        trials.push(Trial::test(name, move || {
            run.run();
            Ok(())
        }))
    };

    for entry in glob::glob(glob).unwrap() {
        let f = entry.unwrap();
        let name = f
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .replace(['.', '-', ' '], "_");

        let should_fail = f.to_string_lossy().contains("fail-typecheck");

        mk_trial(
            name.clone(),
            Run {
                path: f.clone(),
                should_fail,
                test_proofs: false,
            },
        );

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
            mk_trial(
                format!("{name}_with_proofs"),
                Run {
                    path: f.clone(),
                    should_fail,
                    test_proofs: true,
                },
            );
        }
    }

    trials
}

fn main() {
    let args = libtest_mimic::Arguments::from_args();
    let tests = generate_tests("tests/**/*.egg");
    libtest_mimic::run(&args, tests).exit();
}
