use egg_smol::*;
use std::path::{Path, PathBuf};

fn init() {
    let _ = env_logger::builder().is_test(true).try_init();
}

fn walk_directory(path: &Path) -> Vec<PathBuf> {
    if path.is_file() {
        vec![path.to_path_buf()]
    } else if path.is_dir() {
        let mut vec = vec![];
        for entry in std::fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            vec.extend(walk_directory(&path));
        }
        vec
    } else {
        panic!("Not a file or directory??")
    }
}

#[test]
fn test_files() {
    init();
    let paths = walk_directory(Path::new("tests/"));

    for path in paths {
        if path.extension().unwrap_or_default() == "egg" {
            println!("Running test {path:?}");
            let program = std::fs::read_to_string(path).unwrap();
            let mut egraph = EGraph::default();
            match egraph.parse_and_run_program(&program) {
                Ok(msgs) => {
                    for msg in msgs {
                        println!("  {}", msg);
                    }
                }
                Err(err) => panic!("Top level error: {err}"),
            }
        }
    }
}
