use egg_smol::*;

struct Run {
    path: &'static str,
    test_proofs: bool,
    should_fail: bool,
    resugar: bool,
}

impl Run {
    fn run(&self) {
        let _ = env_logger::builder().is_test(true).try_init();
        let program_read = std::fs::read_to_string(self.path).unwrap();
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
                &format!(
                    "Program:\n{}\n ERROR after parse, to_string, and parse again.",
                    desugared_str
                ),
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

// include the tests generated from the build script
include!(concat!(std::env!("OUT_DIR"), "/files.rs"));

#[test]
#[allow(clippy::assertions_on_constants)]
fn test_number_of_tests() {
    assert!(N_TEST_FILES > 30);
}
