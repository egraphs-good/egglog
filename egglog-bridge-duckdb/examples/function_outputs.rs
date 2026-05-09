//! Test for functions with outputs and merge modes (Phase 1.1).
//!
//! Checks that:
//! 1. `:merge old` keeps the first-set value on conflict.
//! 2. `:merge new` overwrites with the latest value on conflict.
//! 3. Rules can read function outputs in body atoms (binding the
//!    output as a variable) and write function outputs in actions.
//!
//! Models the egglog program:
//!
//! ```text
//! (relation src (i64))
//! (function copy_old (i64) i64 :merge old)
//! (function copy_new (i64) i64 :merge new)
//!
//! (rule ((src x))
//!       ((set (copy_old x) x)
//!        (set (copy_new x) x))
//!      :name "copy")
//!
//! ; identity rule reads & writes the same function: should saturate
//! (rule ((copy_old x v))
//!       ((set (copy_old x) v))
//!      :name "id_old")
//!
//! (src 1) (src 2) (src 3)
//! (run-schedule (saturate (run)))
//!
//! (check (= (copy_old 1) 1))
//! (check (= (copy_new 2) 2))
//! ```

use anyhow::Result;
use egglog_bridge_duckdb::{Action, Atom, ColumnTy, EGraph, Literal, MergeMode, Rule, Term};

fn main() -> Result<()> {
    let mut eg = EGraph::new()?;
    eg.add_relation("src", &[ColumnTy::I64])?;
    eg.add_function("copy_old", &[ColumnTy::I64], ColumnTy::I64, MergeMode::Old)?;
    eg.add_function("copy_new", &[ColumnTy::I64], ColumnTy::I64, MergeMode::New)?;

    // (rule ((src x))
    //       ((set (copy_old x) x)
    //        (set (copy_new x) x)))
    eg.add_rule(Rule {
        name: "copy".to_string(),
        ruleset: String::new(),
        body: vec![Atom::Func {
            name: "src".to_string(),
            args: vec![Term::var("x")],
        }],
        actions: vec![
            Action::Insert {
                name: "copy_old".to_string(),
                args: vec![Term::var("x"), Term::var("x")],
            },
            Action::Insert {
                name: "copy_new".to_string(),
                args: vec![Term::var("x"), Term::var("x")],
            },
        ],
    })?;

    // (rule ((copy_old x v)) ((set (copy_old x) v)))
    // Reads output, writes same output. Should saturate after one
    // iteration: each row re-set to itself, no change.
    eg.add_rule(Rule {
        name: "id_old".to_string(),
        ruleset: String::new(),
        body: vec![Atom::Func {
            name: "copy_old".to_string(),
            args: vec![Term::var("x"), Term::var("v")],
        }],
        actions: vec![Action::Insert {
            name: "copy_old".to_string(),
            args: vec![Term::var("x"), Term::var("v")],
        }],
    })?;

    eg.insert("src", &[Literal::I64(1)])?;
    eg.insert("src", &[Literal::I64(2)])?;
    eg.insert("src", &[Literal::I64(3)])?;

    let (iters, _ts) = eg.run_to_saturation()?;
    println!("saturated after {iters} iterations");

    // Direct seeded conflicts to test merge modes:
    eg.insert("copy_old", &[Literal::I64(10), Literal::I64(100)])?;
    eg.insert("copy_old", &[Literal::I64(10), Literal::I64(999)])?; // conflict
    eg.insert("copy_new", &[Literal::I64(20), Literal::I64(200)])?;
    eg.insert("copy_new", &[Literal::I64(20), Literal::I64(999)])?; // conflict

    let copy_old_10 = eg.lookup_i64("copy_old", &[Literal::I64(10)])?;
    let copy_new_20 = eg.lookup_i64("copy_new", &[Literal::I64(20)])?;
    let copy_old_1 = eg.lookup_i64("copy_old", &[Literal::I64(1)])?;
    let copy_new_3 = eg.lookup_i64("copy_new", &[Literal::I64(3)])?;

    println!("copy_old[10]: {copy_old_10:?} (want 100, :merge old keeps first)");
    println!("copy_new[20]: {copy_new_20:?} (want 999, :merge new keeps last)");
    println!("copy_old[1]:  {copy_old_1:?}  (want 1, from rule)");
    println!("copy_new[3]:  {copy_new_3:?}  (want 3, from rule)");

    assert_eq!(copy_old_10, Some(100));
    assert_eq!(copy_new_20, Some(999));
    assert_eq!(copy_old_1, Some(1));
    assert_eq!(copy_new_3, Some(3));
    println!("ok");
    Ok(())
}
