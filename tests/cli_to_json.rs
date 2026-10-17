//! `--to-json` writes the whole e-graph: `--max-functions` and `--max-calls-per-function` only bound what
//! `--to-dot` / `--to-svg` render.

use std::process::Command;

#[test]
fn to_json_is_not_limited_by_the_render_limits() {
    let dir = std::env::temp_dir().join(format!("egglog-cli-to-json-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("many.egg");
    // More rows than the default 40 calls per function, and more functions than the default 40.
    let mut program = String::from("(relation R (i64))\n");
    for i in 0..100 {
        program.push_str(&format!("(R {i})\n"));
    }
    for f in 0..50 {
        program.push_str(&format!("(relation F{f} (i64))\n(F{f} {f})\n"));
    }
    std::fs::write(&input, program).unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_egglog"))
        .arg("--to-json")
        .arg(&input)
        .status()
        .unwrap();
    assert!(status.success());

    let json = egraph_serialize::EGraph::from_json_file(dir.join("many.json")).unwrap();
    let count = |op: &str| json.nodes.values().filter(|n| n.op == op).count();
    assert_eq!(count("R"), 100);
    assert_eq!((0..50).filter(|f| count(&format!("F{f}")) == 1).count(), 50);
    std::fs::remove_dir_all(&dir).ok();
}
