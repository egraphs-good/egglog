//! Tests for the `egglog!` proc macro from `egglog-macros`.
//!
//! This is a step-2 draft of the proposal in
//! `egglog-block-macro-proposal.md` — the macro validates a source
//! string at compile time and expands to `parse_and_run_program`.
//! Steps 3-5 (typed bindings, eclass typing, AOT skip) are TODO.

use egglog::EGraph;
use egglog_macros::egglog;

/// Smoke test: a well-formed program compiles and runs.
#[test]
fn well_formed_program_runs() {
    let mut eg = EGraph::default();
    egglog!(
        eg,
        "(function fib (i64) i64 :no-merge)
         (set (fib 0) 0)
         (set (fib 1) 1)
         (rule ((= f0 (fib x)) (= f1 (fib (+ x 1))))
               ((set (fib (+ x 2)) (+ f0 f1))))
         (run 5)"
    )
    .unwrap();
    assert_eq!(
        eg.lookup_function("fib", &[eg.base_to_value::<i64>(5)]),
        Some(eg.base_to_value::<i64>(5))
    );
}

/// Datatypes round-trip through the macro the same as through
/// `parse_and_run_program`.
#[test]
fn datatype_program() {
    let mut eg = EGraph::default();
    egglog!(
        eg,
        "(datatype Math
            (Num i64)
            (Add Math Math)
            (Mul Math Math))
         (rewrite (Add x (Num 0)) x)
         (let $e (Add (Num 0) (Num 7)))
         (run 1)"
    )
    .unwrap();
}

// Compile-time-failure cases live as `compile_fail` doctests on the
// proc macro, not as runtime tests, because compile errors are caught
// at `cargo build` time. See egglog-macros/src/lib.rs for those.
