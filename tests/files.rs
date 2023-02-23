use egg_smol::*;

fn test_program(program: &str, message: &str, test_proofs: bool) {
    let mut egraph = EGraph::default();
    match egraph.parse_and_run_program(program, false, test_proofs) {
        Ok(msgs) => {
            for msg in msgs {
                log::info!("  {}", msg);
            }
        }
        Err(err) => panic!("{}: {err}", message),
    }
}

fn run(path: &str, test_proofs: bool) {
    let _ = env_logger::builder().is_test(true).try_init();
    let program = std::fs::read_to_string(path).unwrap();
    test_program(&program, "Top level error", test_proofs);

    let egraph = EGraph::default();
    let program_str = egraph
        .parse_program(&program, false)
        .unwrap()
        .into_iter()
        .map(|x| x.to_string())
        .collect::<Vec<String>>()
        .join("\n");
    test_program(
        &program_str,
        &format!(
            "Program:\n{}\n ERROR after parse, to_string, and parse again.",
            program_str
        ),
        test_proofs,
    );
}

// include the tests generated from the build script
include!(concat!(std::env!("OUT_DIR"), "/files.rs"));
