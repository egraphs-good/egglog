//! Primitives smoke test: rule action computes `(+ x 100)`, body
//! filter is `(< x 3)`. We seed src(0..5) so only src(0..3) should
//! fire the rule, and dest should contain {(0,100), (1,101), (2,102)}.
//!
//! Equivalent egglog program:
//!
//! ```text
//! (relation src (i64))
//! (function dest (i64) i64 :merge old)
//! (rule ((src x) (< x 3))
//!       ((set (dest x) (+ x 100)))
//!      :name "shift")
//! (src 0) (src 1) (src 2) (src 3) (src 4)
//! (run 1)
//! (check (= (dest 0) 100))
//! (check (= (dest 1) 101))
//! (check (= (dest 2) 102))
//! ```
//!
//! Egglog's `(print-size)` after this would report `((src 5) (dest 3))`.

use anyhow::Result;
use egglog_bridge_duckdb::{Action, Atom, ColumnTy, EGraph, Literal, MergeMode, Rule, Term};

fn main() -> Result<()> {
    let mut eg = EGraph::new()?;
    eg.add_relation("src", &[ColumnTy::I64])?;
    eg.add_function("dest", &[ColumnTy::I64], ColumnTy::I64, MergeMode::Old)?;

    // (rule ((src x) (< x 3)) ((set (dest x) (+ x 100))))
    eg.add_rule(Rule {
        name: "shift".to_string(),
        ruleset: String::new(),
        body: vec![
            Atom::Func {
                name: "src".to_string(),
                args: vec![Term::var("x")],
            },
            Atom::Filter(Term::prim("<", vec![Term::var("x"), Term::i64(3)])),
        ],
        actions: vec![Action::Insert {
            name: "dest".to_string(),
            args: vec![
                Term::var("x"),
                Term::prim("+", vec![Term::var("x"), Term::i64(100)]),
            ],
        }],
    })?;

    for i in 0..5 {
        eg.insert("src", &[Literal::I64(i)])?;
    }

    eg.run_iteration()?;

    // dest(0..2) populated, dest(3..4) not.
    for i in 0..3 {
        assert_eq!(
            eg.lookup_i64("dest", &[Literal::I64(i)])?,
            Some(i + 100),
            "dest({i}) should be {}",
            i + 100
        );
    }
    for i in 3..5 {
        assert!(
            !eg.check_exists("dest", &[Literal::I64(i)])?,
            "dest({i}) should NOT exist (filter blocks it)"
        );
    }

    let src_count = eg.count("src")?;
    let dest_count = eg.count("dest")?;
    println!("src count:  {src_count} (want 5)");
    println!("dest count: {dest_count} (want 3)");
    assert_eq!(src_count, 5);
    assert_eq!(dest_count, 3);

    println!("ok");
    Ok(())
}
