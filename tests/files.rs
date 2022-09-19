use egg_smol::*;
use std::path::Path;

fn run(path: &Path) -> datatest_stable::Result<()> {
    let _ = env_logger::builder().is_test(true).try_init();
    let program = std::fs::read_to_string(path).unwrap();
    let mut egraph = EGraph::default();
    match egraph.parse_and_run_program(&program) {
        Ok(msgs) => {
            for msg in msgs {
                log::info!("  {}", msg);
            }
        }
        Err(err) => panic!("Top level error: {err}"),
    }
    Ok(())
}

datatest_stable::harness!(run, "tests/", r"\.egg$");
