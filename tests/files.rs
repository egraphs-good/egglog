use std::path::PathBuf;

use egglog::*;
use libtest_mimic::Trial;

#[derive(Clone)]
struct Run {
    name: String,
    path: PathBuf,
    test_proofs: bool,
    should_fail: bool,
    test_serialize: bool,
}

impl Run {
    fn run(&self) {
        let _ = env_logger::builder().is_test(true).try_init();
        let program = std::fs::read_to_string(&self.path).unwrap();
        self.test_program(
            &program,
            "Top level error",
            self.test_serialize,
            self.name.clone(),
        );
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
                false,
                self.name.clone(),
            );
        }
    }

    fn test_program(&self, program: &str, message: &str, test_serialize: bool, name: String) {
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
        };
        let serialized = egraph.serialize(SerializeConfig {
            max_functions: Some(10),
            max_calls_per_function: Some(10),
            include_temporary_functions: false,
        });
        if test_serialize {
            insta::assert_yaml_snapshot!(name, serialized);
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
    let serialize_tests = vec!["eqsat_basic", "map", "fibonacci_demand", "fibonacci"];
    let serialize_proof_tests = vec!["eqsat_basic"];
    for entry in glob::glob(glob).unwrap() {
        let f = entry.unwrap();
        let name = f
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .replace(['.', '-', ' '], "_");

        let should_fail = f.to_string_lossy().contains("fail-typecheck");
        let test_serialize = serialize_tests.iter().any(|&e| e == name);
        mk_trial(
            name.clone(),
            Run {
                name: name.clone(),
                path: f.clone(),
                should_fail,
                test_proofs: false,
                test_serialize,
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
            let test_serialize = serialize_proof_tests.iter().any(|&e| e == name);
            let name = format!("{}_with_proofs", name);
            mk_trial(
                name.clone(),
                Run {
                    name,
                    path: f.clone(),
                    should_fail,
                    test_proofs: true,
                    test_serialize,
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
