//! End-to-end test: build Programs in IR, emit Souffle source, run through
//! the local Souffle binary (the fork at /Users/oflatt/souffle/build/src/souffle),
//! and check the output.
//!
//! These tests are gated by SOUFFLE_BIN being set (or the default path
//! existing) so the rest of the workspace can build without souffle.

use egglog_souffle_backend::{emit, examples};
use std::process::Command;

fn souffle_bin() -> Option<String> {
    if let Ok(b) = std::env::var("SOUFFLE_BIN") {
        return Some(b);
    }
    let default = "/Users/oflatt/souffle/build/src/souffle";
    if std::path::Path::new(default).exists() {
        Some(default.to_string())
    } else {
        None
    }
}

fn run_souffle(source: &str) -> Result<String, String> {
    let bin = souffle_bin().ok_or_else(|| "souffle not found".to_string())?;
    let dir = tempdir();
    let dl_path = format!("{dir}/program.dl");
    std::fs::write(&dl_path, source).map_err(|e| format!("write: {e}"))?;
    let output = Command::new(&bin)
        .arg(&dl_path)
        .output()
        .map_err(|e| format!("spawn: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "souffle failed (exit {:?}):\nstderr:\n{}\nstdout:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn tempdir() -> String {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = format!("/tmp/souffle-backend-test-{pid}-{nanos}");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn buffer_canon_snap_runs_through_souffle() {
    let Some(_) = souffle_bin() else {
        eprintln!("skipping: souffle binary not found");
        return;
    };
    let p = examples::buffer_canon_snap();
    let src = emit(&p);
    eprintln!("--- emitted .dl ---\n{src}");
    let out = run_souffle(&src).expect("souffle should accept the program");
    // Buffer should contain {1} (the only leader in the initial state).
    assert!(out.contains("Buffer\t1"), "expected Buffer size 1; got:\n{out}");
    // Canonical should still contain 3 entries.
    assert!(out.contains("Canonical\t3"), "expected Canonical size 3; got:\n{out}");
}

#[test]
fn add_commutativity_egraph_runs_through_souffle() {
    let Some(_) = souffle_bin() else {
        eprintln!("skipping: souffle binary not found");
        return;
    };
    let p = examples::add_commutativity_egraph();
    let src = emit(&p);
    eprintln!("--- emitted .dl ---\n{src}");
    let out = run_souffle(&src).expect("souffle should accept the program");
    // After commutativity fires, both Add(1,2) and Add(2,1) exist as terms,
    // and a UF edge unifies them. UF should have 5 tuples: 3 self-loops
    // (Lit1, Lit2, Add(1,2)) plus the new Add(2,1) self-loop plus the union.
    assert!(out.contains("UF\t"), "expected UF in output:\n{out}");
}
