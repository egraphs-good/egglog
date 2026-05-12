//! End-to-end smoke: source .egg → souffle_compat term encoding →
//! souffle_translator IR → emit .dl → run through souffle binary.
//!
//! This is the proving ground for `tests/files.rs` integration. Skipped
//! when no souffle binary is available.

use egglog::EGraph;
use egglog::souffle_translator;
use egglog_souffle_backend::emit;

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

/// Emit Souffle source from a program, run it through the souffle binary
/// with a hard timeout, and return stdout. Panics on souffle failure.
fn run_pipeline(source: &str, with_strata: bool) -> Result<String, String> {
    let bin = souffle_bin().ok_or_else(|| "souffle binary not found".to_string())?;
    let mut egraph = if with_strata {
        EGraph::new_with_term_encoding().with_souffle_compat_strata()
    } else {
        EGraph::new_with_term_encoding().with_souffle_compat()
    };
    let commands = egraph
        .resolve_program(None, source)
        .map_err(|e| format!("egglog resolve failed: {e}"))?;
    let program = souffle_translator::translate(&commands)
        .map_err(|e| format!("translator failed: {e}"))?;
    let dl = emit(&program);
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = format!("/tmp/souffle-pipeline-{pid}-{nanos}.dl");
    std::fs::write(&path, &dl).map_err(|e| format!("write: {e}"))?;
    let output = std::process::Command::new("timeout")
        .arg("10")
        .arg(&bin)
        .arg(&path)
        .output()
        .map_err(|e| format!("spawn souffle: {e}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    if !output.status.success() {
        return Err(format!(
            "souffle failed (exit {:?}):\nstderr:\n{}\n--- emitted .dl ---\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr),
            dl
        ));
    }
    Ok(stdout)
}

/// Phase 6 end-to-end: a (run 1) commutativity program goes through the
/// FULL strata pipeline (encode → translate → emit → souffle) and runs
/// to completion.
///
/// This doesn't yet assert RESULT equality with native egglog — output
/// format normalization is a separate concern. It just confirms the
/// strata-encoded program is accepted by souffle and produces output.
#[test]
fn strata_run_n_commutativity_pipeline() {
    let Some(_) = souffle_bin() else {
        eprintln!("skipping: souffle binary not found");
        return;
    };
    let source = r#"
        (sort Math)
        (constructor Add (i64 i64) Math)
        (Add 1 2)
        (rule ((Add a b))
              ((union (Add a b) (Add b a)))
             :name "commutativity")
        (run 1)
    "#;
    match run_pipeline(source, true) {
        Ok(out) => eprintln!("souffle stdout:\n{out}"),
        Err(e) => panic!("strata pipeline failed:\n{e}"),
    }
}

#[test]
fn full_pipeline_smoke() {
    let Some(bin) = souffle_bin() else {
        eprintln!("skipping: souffle binary not found");
        return;
    };

    // The simplest possible program — a constructor, a top-level term, and
    // a print-size to check the relation populated correctly. Don't use
    // (run) since the translator's schedule handling is rudimentary.
    let source = r#"
        (sort Math)
        (constructor Add (i64 i64) Math)
        (Add 1 2)
        (Add 3 4)
    "#;

    let mut egraph = EGraph::new_with_term_encoding().with_souffle_compat();
    let commands = match egraph.resolve_program(None, source) {
        Ok(c) => c,
        Err(e) => panic!("egglog failed to resolve source: {e}"),
    };
    eprintln!("resolved {} commands", commands.len());

    let program = match souffle_translator::translate(&commands) {
        Ok(p) => p,
        Err(e) => panic!("translator failed: {e}"),
    };
    let dl = emit(&program);
    eprintln!("--- emitted .dl ---\n{dl}\n--- end .dl ---");

    // Write to a temp file and run souffle on it.
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = format!("/tmp/souffle-pipeline-{pid}-{nanos}.dl");
    std::fs::write(&path, &dl).expect("write");

    // Hard 10-second timeout — early translator versions could emit programs
    // that didn't terminate or grew memory quickly; the wrapper prevents a
    // runaway from eating the dev machine while we iterate.
    let output = std::process::Command::new("timeout")
        .arg("10")
        .arg(&bin)
        .arg(&path)
        .output()
        .expect("spawn souffle");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    eprintln!("--- souffle stdout ---\n{stdout}");
    eprintln!("--- souffle stderr ---\n{stderr}");
    assert!(
        output.status.success(),
        "souffle failed (exit {:?})",
        output.status
    );
}
