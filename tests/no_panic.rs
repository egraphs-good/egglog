//! Regression tests for previously-reachable panics.
//!
//! Each program below used to abort the process with a `panic!`/`unwrap`/
//! `expect`/`todo!`. They must now fail gracefully instead. Every test calls
//! `parse_and_run_program`, which means a regression (a real panic) would
//! unwind and fail the test binary; returning `Err` is the graceful behavior.

use egglog::EGraph;

/// Assert that `program` returns a recoverable error rather than panicking.
fn assert_errs(program: &str) {
    let mut egraph = EGraph::default();
    let result = egraph.parse_and_run_program(None, program);
    assert!(
        result.is_err(),
        "expected a recoverable error, got Ok for:\n{program}"
    );
}

/// Assert that `program` runs to completion without panicking (the result may
/// be `Ok` or `Err`; the point is that the process is not aborted).
fn assert_no_panic(egraph: &mut EGraph, program: &str) {
    let _ = egraph.parse_and_run_program(None, program);
}

#[test]
fn malformed_sort_declarations_error() {
    assert_errs("(sort S (Vec))");
    assert_errs("(sort S (Vec i64 i64))");
    assert_errs("(sort S (Set))");
    assert_errs("(sort S (Set i64 i64))");
    assert_errs("(sort S (Pair i64))");
    assert_errs("(sort S (Map i64))");
    assert_errs("(sort S (MultiSet))");
    assert_errs("(sort S (UnstableFn))");
    assert_errs("(sort S (UnstableFn x String))");
    assert_errs("(sort S (UnstableFn (i64) (String)))");
}

#[test]
fn partial_primitives_error_instead_of_panicking() {
    // Out-of-range / undefined primitive applications produce no value, so
    // extracting them errors gracefully instead of panicking.
    assert_errs("(sort IV (Vec i64))(let v (vec-of 1 2 3))(extract (vec-set v 10 99))");
    assert_errs("(sort IV (Vec i64))(let v (vec-of 1 2 3))(extract (vec-set v -1 99))");
    assert_errs("(sort IV (Vec i64))(let v (vec-of 1 2 3))(extract (vec-remove v 10))");
    assert_errs("(extract (<< (bigint 5) -3))");
    assert_errs("(extract (>> (bigint 5) -3))");
    assert_errs("(extract (bigrat (bigint 1) (bigint 0)))");
    // NOTE: `log`/`cbrt` of a non-trivial bigrat are intentionally left as
    // `todo!` (unimplemented), not partial primitives: returning "no value"
    // would wrongly claim e.g. `cbrt 8/1` has no rational root.
    assert_errs("(extract (log2 0))");
    assert_errs("(extract (log2 -5))");
    assert_errs("(sort MS (MultiSet i64))(extract (multiset-pick (multiset-of)))");
}

#[test]
fn command_execution_errors() {
    assert_errs("(sort Math)(constructor Num (i64) Math)(let x (Num 5))(extract x -1)");
    assert_errs(
        r#"(datatype Math (Num i64))
           (rule ((= x (Num 1))) ((Num 2)) :name "r")
           (rule ((= x (Num 1))) ((Num 3)) :name "r")"#,
    );
    assert_errs(r#"(input nonexistent_function "/tmp/nofile_xyz.txt")"#);
    assert_errs(
        r#"(sort Math)(constructor Num (i64) Math)
           (function edge (Math) i64 :merge old)
           (input edge "/tmp/nofile_xyz.txt")"#,
    );
    assert_errs(
        r#"(function edge (i64) i64 :merge old)(input edge "/tmp/does_not_exist_xyz.txt")"#,
    );
    assert_errs(
        r#"(sort Math)(constructor Num (i64) Math)
           (let y (Num 7))(delete (Num 7))
           (output "/tmp/egglog_out_xyz.txt" y)"#,
    );
}

#[test]
fn unstable_fn_resolution_errors() {
    // Unknown target function.
    assert_errs(
        r#"(sort BinFn (UnstableFn (i64 i64) i64))
           (function holder () BinFn :merge old)
           (rule ((= s "+")) ((set (holder) (unstable-fn "this_function_does_not_exist"))))
           (run 1)"#,
    );
    // First argument is a (string-typed) variable rather than a string literal.
    assert_errs(
        r#"(sort BinFn (UnstableFn (i64 i64) i64))
           (function holder () BinFn :merge old)
           (rule ((= s "+")) ((set (holder) (unstable-fn s))))
           (run 1)"#,
    );
}

#[test]
fn desugar_and_parse_errors() {
    assert_errs("(fail (datatype*))");
    assert_errs(r#"(fail (include "nonexistent_xyz.egg"))"#);
    assert_errs("(rewrite 1 2 :subsume)");
}

#[test]
fn multiset_index_with_primitive_function_does_not_panic() {
    // Passing a primitive-wrapped function to clear-index used to panic; the
    // operation now simply does not apply instead of aborting.
    let mut egraph = EGraph::default();
    assert_no_panic(
        &mut egraph,
        r#"(sort MS (MultiSet i64))
           (let m (multiset-of 1 2))
           (unstable-multiset-clear-index m (unstable-fn "multiset-count"))"#,
    );
}

#[test]
fn prove_exists_without_proofs_errors() {
    // `prove-exists` requires proofs to be enabled; without them it must error
    // rather than panic.
    let mut egraph = EGraph::default();
    let result = egraph.parse_and_run_program(None, "(datatype M (Foo))(Foo)(prove-exists (Foo))");
    assert!(result.is_err());
}

#[test]
fn term_encoding_with_escaped_string_does_not_panic() {
    // Re-parsing instrumented egglog text that embeds an escaped string literal
    // used to panic; it must now run (or error) without aborting.
    let mut egraph = EGraph::new_with_term_encoding();
    assert_no_panic(
        &mut egraph,
        r#"(sort Math)(constructor MkStr (String) Math)
           (MkStr "x")
           (check (= (MkStr "\"") (MkStr "\"")))"#,
    );
}
