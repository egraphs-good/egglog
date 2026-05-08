//! Phase 0 spike: transitive closure on (edge, path) using DuckDB.
//!
//! Goal: measure per-iteration overhead of issuing seminaive SQL
//! against an in-process DuckDB connection, so we can compare against
//! egglog's existing backend on the equivalent program.
//!
//! What this is: a hand-translation of the term-encoding-free user
//! program below into DuckDB SQL + a Rust driver loop that mimics
//! the schedule/seminaive bookkeeping.
//!
//! ```text
//! (relation edge (i64 i64))
//! (relation path (i64 i64))
//! (rule ((edge a b))            ((path a b))         :name "base")
//! (rule ((path a b) (edge b c)) ((path a c))         :name "step")
//! ; ... seed edges ...
//! (run-schedule (saturate (run)))
//! (check (path 1 N))
//! ```
//!
//! What this is NOT: the term-encoded version. We're measuring the
//! base "DuckDB can run Datalog seminaive at all" cost. Term encoding
//! adds UF tables, view tables, deletion deferral — orthogonal cost
//! we'll layer in Phase 1.

use duckdb::{Connection, Result, params};
use std::time::Instant;

fn main() -> Result<()> {
    // Configurable problem size: chain of N edges 0→1→2→…→N.
    // Saturated path table has N*(N+1)/2 rows.
    let chain_len: i64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let conn = Connection::open_in_memory()?;
    setup_schema(&conn)?;

    let setup_start = Instant::now();
    seed_edges(&conn, chain_len, 1)?;
    let setup_ms = setup_start.elapsed().as_secs_f64() * 1000.0;

    let run_start = Instant::now();
    let (iterations, ts_final) = run_to_saturation(&conn)?;
    let run_ms = run_start.elapsed().as_secs_f64() * 1000.0;

    // Verify: there should be a path from 0 to chain_len.
    let final_path: i64 = conn.query_row(
        "SELECT COUNT(*) FROM path WHERE a = 0 AND b = ?",
        params![chain_len],
        |row| row.get(0),
    )?;
    let total_paths: i64 = conn.query_row("SELECT COUNT(*) FROM path", [], |row| row.get(0))?;
    let expected_total = chain_len * (chain_len + 1) / 2;

    println!("chain_len: {chain_len}");
    println!("setup (CREATE + seed): {setup_ms:.3} ms");
    println!("saturate:              {run_ms:.3} ms ({iterations} iterations, ts={ts_final})");
    println!("paths from 0 to {chain_len}: {final_path} (want 1)");
    println!("total path rows:       {total_paths} (want {expected_total})");

    assert_eq!(final_path, 1, "expected path 0→{chain_len}");
    assert_eq!(total_paths, expected_total, "wrong path count");

    Ok(())
}

fn setup_schema(conn: &Connection) -> Result<()> {
    // Each table carries a `ts BIGINT` column for seminaive
    // bookkeeping. PRIMARY KEY enforces functional-dependency
    // semantics (egglog's "no duplicate row for same key") and is
    // what we ON CONFLICT DO NOTHING against.
    conn.execute_batch(
        "CREATE TABLE edge (
            a  BIGINT NOT NULL,
            b  BIGINT NOT NULL,
            ts BIGINT NOT NULL,
            PRIMARY KEY (a, b)
         );
         CREATE TABLE path (
            a  BIGINT NOT NULL,
            b  BIGINT NOT NULL,
            ts BIGINT NOT NULL,
            PRIMARY KEY (a, b)
         );",
    )?;
    Ok(())
}

fn seed_edges(conn: &Connection, chain_len: i64, ts: i64) -> Result<()> {
    let mut app = conn.appender("edge")?;
    for i in 0..chain_len {
        app.append_row(params![i, i + 1, ts])?;
    }
    app.flush()?;
    Ok(())
}

/// Run the schedule to saturation. Returns (iterations, final ts).
fn run_to_saturation(conn: &Connection) -> Result<(usize, i64)> {
    // Per-rule "last run at" — the ts at which each rule last ran.
    // For seminaive: variant i of rule R requires the focused atom's
    // ts >= last_run_at_R. After running R in iteration t, set
    // last_run_at_R := t.
    let mut last_run_at_base: i64 = 0;
    let mut last_run_at_step: i64 = 0;
    let mut next_ts: i64 = 1;
    let mut iterations = 0;

    loop {
        iterations += 1;
        next_ts += 1;
        let cur = next_ts;

        // Rule "base": (rule ((edge a b)) ((path a b)))
        // 1 body atom → 1 seminaive variant focused on `edge`.
        let base_inserted: usize = conn
            .prepare_cached(
                "INSERT INTO path (a, b, ts)
                 SELECT a, b, ?2 FROM edge
                 WHERE ts >= ?1
                 ON CONFLICT DO NOTHING",
            )?
            .execute(params![last_run_at_base, cur])?;
        last_run_at_base = cur;

        // Rule "step": (rule ((path a b) (edge b c)) ((path a c)))
        // 2 body atoms → 2 seminaive variants.
        // Variant 1: focus on `path`.
        let step1: usize = conn
            .prepare_cached(
                "INSERT INTO path (a, b, ts)
                 SELECT p.a, e.b, ?2
                 FROM path p JOIN edge e ON p.b = e.a
                 WHERE p.ts >= ?1
                 ON CONFLICT DO NOTHING",
            )?
            .execute(params![last_run_at_step, cur])?;
        // Variant 2: focus on `edge`.
        let step2: usize = conn
            .prepare_cached(
                "INSERT INTO path (a, b, ts)
                 SELECT p.a, e.b, ?2
                 FROM path p JOIN edge e ON p.b = e.a
                 WHERE e.ts >= ?1
                 ON CONFLICT DO NOTHING",
            )?
            .execute(params![last_run_at_step, cur])?;
        last_run_at_step = cur;

        let total = base_inserted + step1 + step2;
        if total == 0 {
            break;
        }
    }

    Ok((iterations, next_ts))
}
