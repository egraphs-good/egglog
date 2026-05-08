//! Transitive closure on (edge, path) — the same workload as
//! `duckdb-spike/src/add_example.rs`, but now driven through the
//! `egglog-bridge-duckdb` public API. If this matches the spike's
//! results, the API is at parity for this workload.

use anyhow::Result;
use egglog_bridge_duckdb::{Action, Atom, ColumnTy, EGraph, Literal, Rule, Term};
use std::time::Instant;

fn main() -> Result<()> {
    let chain_len: i64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let mut eg = EGraph::new()?;
    eg.add_function("edge", &[ColumnTy::I64, ColumnTy::I64])?;
    eg.add_function("path", &[ColumnTy::I64, ColumnTy::I64])?;

    // (rule ((edge a b)) ((path a b)) :name "base")
    eg.add_rule(Rule {
        name: "base".to_string(),
        body: vec![Atom::Func {
            name: "edge".to_string(),
            args: vec![Term::var("a"), Term::var("b")],
        }],
        actions: vec![Action::Insert {
            name: "path".to_string(),
            args: vec![Term::var("a"), Term::var("b")],
        }],
    })?;

    // (rule ((path a b) (edge b c)) ((path a c)) :name "step")
    eg.add_rule(Rule {
        name: "step".to_string(),
        body: vec![
            Atom::Func {
                name: "path".to_string(),
                args: vec![Term::var("a"), Term::var("b")],
            },
            Atom::Func {
                name: "edge".to_string(),
                args: vec![Term::var("b"), Term::var("c")],
            },
        ],
        actions: vec![Action::Insert {
            name: "path".to_string(),
            args: vec![Term::var("a"), Term::var("c")],
        }],
    })?;

    let setup_start = Instant::now();
    for i in 0..chain_len {
        eg.insert("edge", &[Literal::I64(i), Literal::I64(i + 1)])?;
    }
    let setup_ms = setup_start.elapsed().as_secs_f64() * 1000.0;

    let run_start = Instant::now();
    let (iters, ts) = eg.run_to_saturation()?;
    let run_ms = run_start.elapsed().as_secs_f64() * 1000.0;

    let final_path = eg.check_exists(
        "path",
        &[Literal::I64(0), Literal::I64(chain_len)],
    )?;
    let total = eg.count("path")?;
    let expected = chain_len * (chain_len + 1) / 2;

    println!("chain_len: {chain_len}");
    println!("setup:     {setup_ms:.3} ms");
    println!("saturate:  {run_ms:.3} ms ({iters} iterations, ts={ts})");
    println!("0→{chain_len}: {final_path}, total path rows: {total} (want {expected})");

    assert!(final_path);
    assert_eq!(total, expected);
    Ok(())
}
