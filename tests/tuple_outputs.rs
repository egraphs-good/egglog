//! Integration tests for tuple-output functions: functions declared with more than one output
//! sort, e.g. `(function f (Math) (i64 i64) :merge (values ...))`. Their value columns are
//! destructured with `(= (values a b) (f x))` and written with `(set (f x) (values a b))`.

use egglog::{EGraph, Error, TypeError};

fn run(prog: &str) -> Result<(), Error> {
    EGraph::default()
        .parse_and_run_program(None, prog)
        .map(|_| ())
}

#[test]
fn tuple_merge_interval_analysis() {
    // The motivating example: an interval analysis whose lower bound merges with `max` and whose
    // upper bound merges with `min`, expressed as a single two-output function.
    run(r#"
        (datatype Math (Num i64) (Add Math Math))
        (function interval (Math) (i64 i64)
          :merge (values (max old0 new0) (min old1 new1)))

        (rule ((= e (Num n)))
          ((set (interval e) (values n n))))
        (rule ((= e (Add a b))
               (= (values alo ahi) (interval a))
               (= (values blo bhi) (interval b)))
          ((set (interval e) (values (+ alo blo) (+ ahi bhi)))))

        (let expr (Add (Num 1) (Num 2)))
        (run 10)
        (check (= (values 3 3) (interval expr)))
    "#)
    .unwrap();
}

#[test]
fn tuple_merge_combines_columns_independently() {
    // Two `set`s on the same key are resolved column-by-column: lo with max, hi with min.
    run(r#"
        (datatype M (V i64))
        (function iv (M) (i64 i64) :merge (values (max old0 new0) (min old1 new1)))
        (set (iv (V 0)) (values 1 10))
        (set (iv (V 0)) (values 3 7))
        (check (= (values 3 7) (iv (V 0))))
    "#)
    .unwrap();
}

#[test]
fn tuple_check_distinguishes_columns() {
    // A `(values ...)` check must match every column, not just the first.
    let bad = run(r#"
        (datatype M (V i64))
        (function iv (M) (i64 i64) :merge (values (max old0 new0) (min old1 new1)))
        (set (iv (V 0)) (values 3 7))
        (check (= (values 3 99) (iv (V 0))))
    "#);
    assert!(bad.is_err(), "check on a wrong second column should fail");
}

#[test]
fn tuple_destructure_binds_each_column_in_rule() {
    run(r#"
        (datatype M (V i64))
        (function iv (M) (i64 i64) :merge (values (max old0 new0) (min old1 new1)))
        (relation lo-is (M i64))
        (relation hi-is (M i64))
        (set (iv (V 0)) (values 3 7))
        (rule ((= (values l h) (iv x))) ((lo-is x l) (hi-is x h)))
        (run 1)
        (check (lo-is (V 0) 3))
        (check (hi-is (V 0) 7))
        (fail (check (lo-is (V 0) 7)))
    "#)
    .unwrap();
}

#[test]
fn tuple_output_eq_sorts() {
    // Output columns may be e-classes, and are kept canonical across unions like any other column.
    run(r#"
        (datatype N (Nn i64))
        (function two (i64) (N N) :merge (values new0 new1))
        (set (two 0) (values (Nn 1) (Nn 2)))
        (relation got (N N))
        (rule ((= (values a b) (two k))) ((got a b)))
        (run 1)
        (check (got (Nn 1) (Nn 2)))

        ; unioning an output e-class is reflected after rebuilding
        (union (Nn 1) (Nn 5))
        (run 1)
        (check (= (values (Nn 5) (Nn 2)) (two 0)))
    "#)
    .unwrap();
}

#[test]
fn tuple_no_merge_asserts_equality() {
    // With no `:merge`, each output column asserts equality; a conflicting write is an error.
    run(r#"
        (datatype M (V i64))
        (function iv (M) (i64 i64))
        (set (iv (V 0)) (values 1 2))
        (set (iv (V 0)) (values 1 3))
    "#)
    .expect_err("conflicting write under default (assert-eq) merge should fail");
}

#[test]
fn constructor_tuple_output_rejected() {
    let err = EGraph::default()
        .parse_and_run_program(None, "(constructor c (i64) (i64 i64))")
        .unwrap_err();
    assert!(
        matches!(err, Error::TypeError(TypeError::TupleOutputNotAllowed(..))),
        "expected TupleOutputNotAllowed, got {err:?}"
    );
}

#[test]
fn tuple_merge_arity_mismatch_rejected() {
    let err = EGraph::default()
        .parse_and_run_program(
            None,
            "(function f (i64) (i64 i64) :merge (values (max old0 new0)))",
        )
        .unwrap_err();
    assert!(
        matches!(err, Error::TypeError(TypeError::TupleMergeArity { .. })),
        "expected TupleMergeArity, got {err:?}"
    );
}

#[test]
fn builtin_keywords_are_reserved() {
    // Built-in keywords (command/action/schedule heads, plus the tuple constructor `values`) may
    // not be used as function/constructor/relation/variant names or as variables.
    for prog in [
        // `values` (the tuple constructor) in each identifier position
        "(function values (i64) i64 :merge new)",
        "(sort S) (constructor values (i64) S)",
        "(relation values (i64))",
        "(datatype D (values i64))",
        "(relation r (i64)) (rule ((r values)) ())",
        // a sampling of command/action/schedule keywords
        "(function set (i64) i64 :merge new)",
        "(relation union (i64))",
        "(sort run)",
        "(datatype D (function i64))",
        "(relation r (i64)) (rule ((r let)) ())",
    ] {
        let err = run(prog).expect_err(&format!("keyword should be rejected in: {prog}"));
        assert!(
            err.to_string().contains("reserved keyword"),
            "expected a reserved-keyword error for {prog:?}, got: {err}"
        );
    }

    // The `:` prefix is reserved for option keywords, so it cannot start an identifier.
    let err = run("(relation :merge (i64))").unwrap_err();
    assert!(
        err.to_string().contains("`:` prefix is reserved"),
        "expected a colon-prefix error, got: {err}"
    );
}

#[test]
fn input_output_are_only_partially_reserved() {
    // `input`/`output` are usable as ordinary variables...
    run("(relation r (i64)) (r 0) (rule ((r input)) ((r (+ input 1)))) (run 1) (check (r 1))")
        .unwrap();

    // ...but not as definition names...
    for prog in [
        "(function input (i64) i64 :merge new)",
        "(relation output (i64))",
        "(sort input)",
        "(datatype D (output i64))",
    ] {
        let err = run(prog).expect_err(&format!("`input`/`output` should be rejected in: {prog}"));
        assert!(
            err.to_string()
                .contains("cannot be used as a definition name"),
            "expected a definition-name error for {prog:?}, got: {err}"
        );
    }

    // ...and not as the head of a call expression.
    let err = run("(relation r (i64)) (rule ((r x)) ((r (input x))))").unwrap_err();
    assert!(
        err.to_string()
            .contains("cannot be used as the head of an expression"),
        "expected a call-head error, got: {err}"
    );
}

#[test]
fn tuple_merge_must_be_values_form() {
    let err = EGraph::default()
        .parse_and_run_program(None, "(function f (i64) (i64 i64) :merge old0)")
        .unwrap_err();
    assert!(
        matches!(err, Error::TypeError(TypeError::TupleMergeNotValues(..))),
        "expected TupleMergeNotValues, got {err:?}"
    );
}
