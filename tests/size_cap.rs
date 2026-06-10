//! Tests for the e-graph size cap (`(run-schedule ... :size-limit N)`).
//!
//! The cap bounds *total* e-graph growth over a whole schedule. It is enforced
//! against a running *estimate* of the size that is resynced to the true size
//! after every iteration and bumped by an upper bound on inserted rows in
//! between. Because the estimate is an upper bound, the true size can never run
//! unboundedly past the cap; in the worst case it overshoots only by the rows
//! already in flight across the parallel apply when the budget is spent.

use egglog::*;

/// The largest batch a single action buffer flushes at (`VAR_BATCH_SIZE` in
/// `core-relations`). Batches near the cap are smaller, so this is an upper
/// bound on the rows one in-flight batch can apply.
const MAX_BATCH: usize = 128;

/// An upper bound on how far the true size can overshoot the cap: at most one
/// maximal batch per worker thread can be mid-apply when the cap trips, and each
/// such row inserts at most `enodes_per_match` e-nodes. This is constant in the
/// cap -- doubling the cap does not change it.
fn overshoot_bound(enodes_per_match: usize) -> usize {
    rayon::current_num_threads() * MAX_BATCH * enodes_per_match
}

/// Number of e-graph tuples after running a schedule that grows the relation `R`
/// without bound, under the given size limit. `body` is the rule(s) that drive
/// the growth.
fn run_capped(cap: usize, body: &str) -> usize {
    let mut egraph = EGraph::default();
    let program = format!(
        r#"
        (relation R (i64))
        (R 1)
        (R 2)
        (R 3)
        (R 4)
        (R 5)
        {body}
        (run-schedule (saturate (run)) :size-limit {cap})
        "#
    );
    egraph.parse_and_run_program(None, &program).unwrap();
    egraph.num_tuples()
}

/// Each match inserts two *distinct* new integers (a binary tree of values), so
/// almost nothing the rule produces is a duplicate. Without a cap this relation
/// grows without bound; the cap must stop it right around the limit.
const FEW_DUPLICATES: &str = "(rule ((R x) (< x 1000000000)) ((R (* x 2)) (R (+ (* x 2) 1))))";

/// Every pair of rows re-derives the single row `(R 0)`, so the rule fires a
/// quadratic number of matches that almost all insert an already-present row.
/// The second rule grows the relation slowly so the run never saturates. The
/// estimate (applied rows) explodes while the true size grows slowly.
const MANY_DUPLICATES: &str = r#"
        (rule ((R x) (R y)) ((R 0)))
        (rule ((R x) (< x 1000000000)) ((R (+ x 1))))
        "#;

#[test]
fn few_duplicates_overshoot_is_constant_bounded() {
    let bound = overshoot_bound(2);

    for cap in [20_000usize, 100_000, 400_000] {
        let size = run_capped(cap, FEW_DUPLICATES);
        let overshoot = size - cap;
        assert!(
            overshoot <= bound,
            "few-duplicates overshoot is not constant-bounded: size={size}, \
             cap={cap}, overshoot={overshoot}, bound={bound}"
        );
    }
}

#[test]
fn many_duplicates_stays_within_cap() {
    for cap in [2_000usize, 20_000] {
        let size = run_capped(cap, MANY_DUPLICATES);
        assert!(
            size <= cap,
            "many-duplicates run exceeded the cap: size={size}, cap={cap}"
        );
    }
}
