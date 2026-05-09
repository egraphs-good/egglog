//! Hand-translation of `tests/web-demo/path.egg` to the duckdb
//! backend. Verifies that running the same program through this
//! backend produces the same final tuple counts as egglog itself.
//!
//! The .egg source:
//!
//! ```text
//! (relation path (i64 i64))
//! (relation edge (i64 i64))
//! (rule ((edge x y))            ((path x y)))
//! (rule ((path x y) (edge y z)) ((path x z)))
//! (edge 1 2)  (edge 2 3)  (edge 3 4)
//! (check (edge 1 2))
//! (fail (check (path 1 2)))
//! (run 3)
//! (check (path 1 4))
//! (fail (check (path 4 1)))
//! (print-size)
//! ```
//!
//! Egglog's `(print-size)` after `(run 3)` reports
//! `((edge 3) (path 6))`. We assert the same counts here.

use anyhow::Result;
use egglog_bridge_duckdb::{Action, Atom, ColumnTy, EGraph, Literal, Rule, Term};

const EXPECTED_EDGE_COUNT: i64 = 3;
const EXPECTED_PATH_COUNT: i64 = 6;

fn main() -> Result<()> {
    let mut eg = EGraph::new()?;
    eg.add_relation("edge", &[ColumnTy::I64, ColumnTy::I64])?;
    eg.add_relation("path", &[ColumnTy::I64, ColumnTy::I64])?;

    // (rule ((edge x y)) ((path x y)))
    eg.add_rule(Rule {
        name: "base".to_string(),
        ruleset: String::new(),
        body: vec![Atom::Func {
            name: "edge".to_string(),
            args: vec![Term::var("x"), Term::var("y")],
        }],
        actions: vec![Action::Insert {
            name: "path".to_string(),
            args: vec![Term::var("x"), Term::var("y")],
        }],
    })?;

    // (rule ((path x y) (edge y z)) ((path x z)))
    eg.add_rule(Rule {
        name: "step".to_string(),
        ruleset: String::new(),
        body: vec![
            Atom::Func {
                name: "path".to_string(),
                args: vec![Term::var("x"), Term::var("y")],
            },
            Atom::Func {
                name: "edge".to_string(),
                args: vec![Term::var("y"), Term::var("z")],
            },
        ],
        actions: vec![Action::Insert {
            name: "path".to_string(),
            args: vec![Term::var("x"), Term::var("z")],
        }],
    })?;

    eg.insert("edge", &[Literal::I64(1), Literal::I64(2)])?;
    eg.insert("edge", &[Literal::I64(2), Literal::I64(3)])?;
    eg.insert("edge", &[Literal::I64(3), Literal::I64(4)])?;

    // Pre-run checks (matching `(check (edge 1 2))` and
    // `(fail (check (path 1 2)))` before `(run 3)`).
    assert!(
        eg.check_exists("edge", &[Literal::I64(1), Literal::I64(2)])?,
        "expected edge(1, 2) before run"
    );
    assert!(
        !eg.check_exists("path", &[Literal::I64(1), Literal::I64(2)])?,
        "did NOT expect path(1, 2) before run"
    );

    // The .egg uses `(run 3)`. Three iterations of the default
    // ruleset is enough for this chain to saturate; we also accept
    // saturating earlier or running a few extra empty iterations.
    for _ in 0..3 {
        eg.run_iteration()?;
    }

    // Post-run checks.
    assert!(
        eg.check_exists("path", &[Literal::I64(1), Literal::I64(4)])?,
        "expected path(1, 4) after run"
    );
    assert!(
        !eg.check_exists("path", &[Literal::I64(4), Literal::I64(1)])?,
        "did NOT expect path(4, 1) after run"
    );

    let edge_count = eg.count("edge")?;
    let path_count = eg.count("path")?;

    println!("edge count: {edge_count} (egglog says {EXPECTED_EDGE_COUNT})");
    println!("path count: {path_count} (egglog says {EXPECTED_PATH_COUNT})");

    assert_eq!(edge_count, EXPECTED_EDGE_COUNT);
    assert_eq!(path_count, EXPECTED_PATH_COUNT);
    println!("counts match egglog's snapshot");
    Ok(())
}
