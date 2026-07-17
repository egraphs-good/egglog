use egglog::EGraph;
use egglog::ast::Command;
use egglog_checked::{egglog_checked, egglog_header, run_egglog_checked};

// Two independent, self-contained schemas, as if declared in (and exported
// from) separate crates. Each is typechecked at its own `egglog_header!` site.
egglog_header!(math_schema
    (datatype Math (Num i64) (Add Math Math))
);
egglog_header!(seen_schema
    (relation seen (i64))
);

#[test]
fn single_header_checks_a_fragment() {
    let fragment: Vec<Command> = egglog_checked!(math_schema;
        (rule ((= e (Add (Num a) (Num b)))) ((union e (Num (+ a b)))))
    )
    .unwrap();
    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(None, "(datatype Math (Num i64) (Add Math Math))")
        .unwrap();
    egraph.run_program(fragment).unwrap();
}

#[test]
fn composed_headers_pool_their_declarations() {
    // The fragment uses `Add`/`Num` (from `math_schema`) *and* `seen` (from
    // `seen_schema`) — it only typechecks with both schemas in scope.
    let fragment: Vec<Command> = egglog_checked!(math_schema, seen_schema;
        (rule ((= e (Add (Num a) (Num b)))) ((seen a)))
    )
    .unwrap();

    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(
            None,
            "(datatype Math (Num i64) (Add Math Math)) (relation seen (i64))",
        )
        .unwrap();
    egraph.run_program(fragment).unwrap();
}

#[test]
fn hands_back_checked_commands_to_run_elsewhere() {
    // Checked at compile time, handed back as unresolved commands, then run
    // into a separately-built e-graph (they re-typecheck against it).
    let program: Vec<Command> = egglog_checked!(
        (datatype Math (Num i64) (Add Math Math))
        (rule ((= e (Add (Num a) (Num b)))) ((union e (Num (+ a b)))))
        (let start (Add (Num 1) (Num 2)))
        (run 1)
        (check (= start (Num 3)))
    )
    .unwrap();

    let mut egraph = EGraph::default();
    egraph.run_program(program).unwrap();
}

#[test]
fn builds_and_runs_a_checked_program() {
    // Compile-time-checked; at run time this builds a fresh e-graph, runs the
    // fold rule once, and the embedded `(check …)` confirms `start` folded.
    let egraph: EGraph = run_egglog_checked!(
        (datatype Math (Num i64) (Add Math Math))
        (rule ((= e (Add (Num a) (Num b)))) ((union e (Num (+ a b)))))
        (let start (Add (Num 1) (Num 2)))
        (run 1)
        (check (= start (Num 3)))
    )
    .unwrap();
    let _ = egraph;
}

#[test]
fn hyphenated_and_keyword_atoms_survive() {
    // `:no-merge`, `my-ruleset`, and negative literals must round-trip through
    // the token renderer as single egglog atoms.
    let egraph: EGraph = run_egglog_checked!(
        (function f (i64) i64 :no-merge)
        (ruleset my-ruleset)
        (set (f 0) -1)
        (rule ((= v (f x))) ((set (f (+ x 1)) v)) :ruleset my-ruleset)
    )
    .unwrap();
    let _ = egraph;
}
