use egg_smol::*;
use std::path::Path;

fn walk_directory(path: &Path, f: &mut impl FnMut(&Path)) {
    if path.is_file() {
        f(path)
    } else if path.is_dir() {
        for entry in std::fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            walk_directory(&path, f);
        }
    } else {
        panic!("Not a file or directory??")
    }
}

#[test]
fn test_files() {
    walk_directory(Path::new("tests/"), &mut |path| {
        if path.extension().unwrap_or_default() == "egg" {
            println!("Running test {path:?}");
            let program = std::fs::read_to_string(path).unwrap();
            let mut egraph = EGraph::default();
            egraph.run_program(&program).unwrap();
        }
    });
}
