use egg_smol::*;
use std::path::Path;

fn test_program(program: &str, message: &str) {
    let mut egraph = EGraph::default();
    match egraph.parse_and_run_program(program) {
        Ok(msgs) => {
            for msg in msgs {
                log::info!("  {}", msg);
            }
        }
        Err(err) => panic!("{}: {err}", message),
    }
}

fn run(path: &Path) -> datatest_stable::Result<()> {
    let _ = env_logger::builder().is_test(true).try_init();
    let program = std::fs::read_to_string(path).unwrap();
    test_program(&program, "Top level error");

    let egraph = EGraph::default();
    let program_str = egraph
        .parse_program(&program)
        .unwrap()
        .into_iter()
        .map(|x| x.to_string())
        .collect::<Vec<String>>()
        .join("\n");
    test_program(&program_str, &format!("Program:\n{}\n ERROR after parse, to_string, and parse again.", program_str));

    Ok(())
}

datatest_stable::harness!(run, "tests/", r"\.egg$");
