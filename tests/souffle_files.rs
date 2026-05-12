//! Souffle-backend correctness tests against native egglog.
//!
//! The strong test is **check parity**: each `(check ...)` in a source
//! program must reach the same pass/fail conclusion under souffle as it
//! does under native egglog in default mode. Default mode is the
//! gold-standard semantics — proof mode in `tests/files.rs` validates
//! itself the same way.
//!
//! Print-size parity is *not* tested here. Default egglog tracks every
//! e-node flatly (e.g. (Add 1 2) and (Add 2 1) after commutativity are
//! two e-nodes in one eclass); the souffle backend's strata setup uses
//! subsumption to keep one canonical rep per eclass. Both are valid
//! views of the same egraph — checks of equivalence-class equality
//! agree, but per-relation row counts don't.
//!
//! Skipped when the souffle binary isn't available.
//!
//! TODO: integrate as a treatment in tests/files.rs (alongside proof
//! mode) so the souffle backend gets exercised on the full corpus
//! automatically — same shared-snapshot machinery, just with the
//! check-only comparison.

use egglog::souffle_translator;
use egglog::EGraph;
use egglog_souffle_backend::runner;

/// Files containing `(check ...)`. Souffle's pass/fail conclusion must
/// match native egglog in default mode.
const CHECK_PARITY: &[&str] = &["souffle_smoke_check.egg"];

/// Files we just want to confirm parse + translate + run end-to-end.
/// No semantic comparison — just "doesn't crash."
///
/// Excluded from smoke: math-microbenchmark.egg — the rewrites saturate
/// to ~hundreds-of-thousands of Adds under our cycle-based encoding
/// (default egglog with `(run 11)` reports `Add 641,743`). That's
/// tractable, just not within the test wrapper's 10-second timeout.
/// Putting it back requires either (a) a working `(run N)` bound
/// (blocked on the fork's `.snapshot` ↔ delta-tracking interaction)
/// or (b) raising the timeout for this one file.
const SMOKE: &[&str] = &[
    "souffle_smoke_commutativity.egg",
    "souffle_smoke_or_lor.egg",
    "math-microbenchmark.egg",
];

fn run_native_default(source: &str) -> Result<(), String> {
    let mut egraph = EGraph::default();
    egraph.ensure_no_reserved_symbols(false);
    egraph
        .parse_and_run_program(None, source)
        .map(|_| ())
        .map_err(|e| format!("native (default) egglog failed: {e}"))
}

fn run_souffle(source: &str) -> Result<runner::RunOutput, String> {
    let mut egraph = EGraph::new_with_term_encoding().with_souffle_compat_strata();
    let commands = egraph
        .resolve_program(None, source)
        .map_err(|e| format!("egglog resolve failed: {e}"))?;
    let out = souffle_translator::translate_with_manifest(&commands)
        .map_err(|e| format!("translator failed: {e}"))?;
    runner::run(&out.program, &out.manifest).map_err(|e| format!("souffle run failed: {e}"))
}

fn read_file(file_stem: &str) -> Option<String> {
    let path = format!("/Users/oflatt/egglog/tests/{file_stem}");
    std::fs::read_to_string(&path).ok()
}

#[test]
fn check_parity_against_default_egglog() {
    if runner::find_souffle_binary().is_none() {
        eprintln!("skipping: souffle binary not found");
        return;
    }
    for f in CHECK_PARITY {
        eprintln!("checking check parity: {f}");
        let Some(source) = read_file(f) else {
            eprintln!("  skipping {f}: file not present");
            continue;
        };
        let native_result = run_native_default(&source);
        let souffle = run_souffle(&source).expect("souffle run");
        match native_result {
            Ok(()) => {
                assert!(
                    !souffle.check_results.is_empty(),
                    "{f}: souffle ran but emitted no check relations \
                     (manifest.check_relations empty?)"
                );
                assert!(
                    souffle.check_results.iter().all(|&p| p),
                    "{f}: native (default) accepted all checks but souffle \
                     reported failures: {:?}\nsouffle stdout:\n{}",
                    souffle.check_results,
                    souffle.raw_stdout
                );
            }
            Err(e) => {
                assert!(
                    souffle.check_results.iter().any(|&p| !p),
                    "{f}: native (default) rejected the program ({e}) but \
                     souffle accepted every check: {:?}\nsouffle stdout:\n{}",
                    souffle.check_results,
                    souffle.raw_stdout
                );
            }
        }
    }
}

#[test]
fn smoke_files_run_without_errors() {
    if runner::find_souffle_binary().is_none() {
        eprintln!("skipping: souffle binary not found");
        return;
    }
    for f in SMOKE {
        eprintln!("smoke: {f}");
        let Some(source) = read_file(f) else {
            eprintln!("  skipping {f}: file not present");
            continue;
        };
        // Just confirm the pipeline doesn't error out. Print-size and
        // check parity are tested separately in their own tests.
        run_souffle(&source).expect("souffle run");
    }
}
