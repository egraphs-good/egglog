use egglog::EGraph;
use egglog::ast::Command;
use egglog_static::{egglog_static, run_egglog_static};

#[test]
fn hands_back_checked_commands_to_run_elsewhere() {
    // Checked at compile time, handed back as unresolved commands, then run
    // into a separately-built e-graph (they re-typecheck against it).
    let program: Vec<Command> = egglog_static!(
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
    let egraph: EGraph = run_egglog_static!(
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
    let egraph: EGraph = run_egglog_static!(
        (function f (i64) i64 :no-merge)
        (ruleset my-ruleset)
        (set (f 0) -1)
        (rule ((= v (f x))) ((set (f (+ x 1)) v)) :ruleset my-ruleset)
    )
    .unwrap();
    let _ = egraph;
}
