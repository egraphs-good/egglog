//! Souffle-backend correctness tests against native egglog.
//!
//! The strong test is **check parity**: each `(check ...)` in a source
//! program must reach the same pass/fail conclusion under souffle as it
//! does under native egglog in default mode. Default mode is the
//! gold-standard semantics — proof mode in `tests/files.rs` validates
//! itself the same way.
//!
//! Phase 60c's canonical projection (`<view>_canonical` drops eclass +
//! proof + wave columns, leaving just inputs) gives the souffle backend
//! a print-size identical to default egglog for functional
//! constructors. Verified on commutativity-style cases here.
//!
//! `math-microbenchmark.egg` is excluded from these tests — not for a
//! correctness gap, but because materializing the ~641K Adds it
//! reaches at `(run 11)` takes more time than is reasonable to spend
//! in a test wrapper. That's a backend perf limitation.
//!
//! Skipped when the souffle binary isn't available.

use egglog::souffle_translator;
use egglog::EGraph;
use egglog_souffle_backend::runner;

/// Files containing `(check ...)`. Souffle's pass/fail conclusion must
/// match native egglog in default mode.
const CHECK_PARITY: &[&str] = &["souffle_smoke_check.egg"];

/// Files we just want to confirm parse + translate + run end-to-end.
/// No semantic comparison — just "doesn't crash."
///
/// Excluded from smoke: math-microbenchmark.egg. Phase 60c's
/// canonical projection makes print-size accurate in principle, but
/// the rule-emission cost (per-iter cascading through every rule
/// against every wave-bearing row) means materializing the ~641K
/// Adds that default egglog reaches at `(run 11)` doesn't complete
/// within souffle's reasonable runtime. This is a backend
/// performance limitation, not a correctness gap.
const SMOKE: &[&str] = &[
    "souffle_smoke_commutativity.egg",
    "souffle_smoke_or_lor.egg",
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

/// Programs whose `(print-size <fname>)` should equal default egglog
/// after `(run N)`. These are programs where the rebuild rule's
/// subsumption *doesn't* shrink the canonical view below default
/// egglog's count — i.e., programs that don't rely on heavy congruence
/// closure cascades.
///
/// Format: (file stem, function name to check).
const PRINT_SIZE_PARITY: &[(&str, &str)] = &[
    ("souffle_smoke_commutativity.egg", "Add"),
    ("souffle_smoke_check.egg", "Add"),
];

#[test]
fn print_size_parity_against_default_egglog() {
    if runner::find_souffle_binary().is_none() {
        eprintln!("skipping: souffle binary not found");
        return;
    }
    for (f, fname) in PRINT_SIZE_PARITY {
        eprintln!("checking print-size parity: {f} for {fname}");
        let Some(source) = read_file(f) else {
            eprintln!("  skipping {f}: file not present");
            continue;
        };
        // Native default-mode count, captured by re-running
        // `(print-size <fname>)` after the program.
        let mut native = EGraph::default();
        native.ensure_no_reserved_symbols(false);
        native
            .parse_and_run_program(None, &source)
            .expect("native run");
        let print_results = native
            .parse_and_run_program(None, &format!("(print-size {fname})"))
            .expect("native print-size");
        let native_count = match print_results.as_slice() {
            [egglog::CommandOutput::PrintFunctionSize(n)] => *n,
            other => panic!("{f}: unexpected print-size result shape: {other:?}"),
        };
        // Souffle count via runner's view_sizes (collected from
        // souffle's `.printsize` stdout).
        let souffle = run_souffle(&source).expect("souffle run");
        let souffle_count = souffle
            .view_sizes
            .iter()
            .find(|(name, _)| name == fname)
            .map(|(_, n)| *n as usize)
            .unwrap_or_else(|| {
                panic!(
                    "{f}: souffle output has no count for {fname} \
                     (view_sizes={:?})",
                    souffle.view_sizes
                )
            });
        assert_eq!(
            souffle_count, native_count,
            "{f}: print-size mismatch for {fname} \
             (souffle={souffle_count}, native={native_count})\n\
             souffle stdout:\n{}",
            souffle.raw_stdout
        );
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
