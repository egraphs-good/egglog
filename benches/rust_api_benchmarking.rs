mod common;
mod math_microbenchmark_support;

#[derive(Clone, Copy)]
struct RustRuleBenchCase {
    n_facts_input: Option<usize>,
    n_rule_run_estimated: Option<usize>,
}

impl std::fmt::Display for RustRuleBenchCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write!(f, "rust_rule_facts{}", self.n_facts_input.unwrap_or(0))
        write!(f, "rule_run_{}", self.n_rule_run_estimated.unwrap_or(0))
    }
}

struct RustRuleBenchInput {
    egraph: egglog::EGraph,
    ruleset: String,
}

// Match-only rust_rule bench setup, to isolate the overhead of matching a rust rule
// without doing any actual work in the rule body.
fn match_only_rust_rule_setup(case: RustRuleBenchCase) -> RustRuleBenchInput {
    use egglog::prelude::*;

    common::configure_rayon_once();

    let mut program = String::new();
    program.push_str("(relation R (i64))\n");

    for i in 0..case
        .n_facts_input
        .expect("n_facts_input must be set for match_only_rust_rule_setup")
    {
        use std::fmt::Write;
        let _ = writeln!(&mut program, "(R {})", i as i64);
    }

    let mut egraph = egglog::EGraph::default();
    egraph.parse_and_run_program(None, &program).unwrap();

    let ruleset = "rust_rule_bench";
    add_ruleset(&mut egraph, ruleset).unwrap();

    rust_rule(
        &mut egraph,
        "rust_rule_bench",
        ruleset,
        vars![x: i64],
        facts![(R x)],
        |_ctx, _values| Some(()),
    )
    .unwrap();

    RustRuleBenchInput {
        egraph,
        ruleset: ruleset.to_owned(),
    }
}

#[derive(Clone, Copy)]
struct RustRuleTableActionBenchCase {
    n_facts_input: usize,
    n_dummy_funcs: usize,
}

impl std::fmt::Display for RustRuleTableActionBenchCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "facts{}_funcs{}", self.n_facts_input, self.n_dummy_funcs)
    }
}

#[derive(Clone, Copy)]
struct RustRuleInsertLoopBenchCase {
    n_ops: usize,
    n_dummy_funcs: usize,
}

impl std::fmt::Display for RustRuleInsertLoopBenchCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ops{}_funcs{}", self.n_ops, self.n_dummy_funcs)
    }
}

#[divan::bench(
    args = [
        RustRuleBenchCase { n_facts_input: Some(50_000), n_rule_run_estimated: Some(1) },
    ],
    sample_count = 10
)]
fn rust_rule_match_overhead(bencher: divan::Bencher, case: RustRuleBenchCase) {
    use egglog::prelude::run_ruleset;

    bencher
        .with_inputs(|| match_only_rust_rule_setup(case))
        .bench_local_refs(|input| {
            run_ruleset(&mut input.egraph, &input.ruleset).unwrap();
        });
}

#[divan::bench(
    args = [
        RustRuleBenchCase { n_facts_input: Some(50_000), n_rule_run_estimated: Some(1) },
    ],
    sample_count = 10
)]
fn rust_rule_match_with_serialize(bencher: divan::Bencher, case: RustRuleBenchCase) {
    use egglog::prelude::run_ruleset;

    bencher
        .with_inputs(|| match_only_rust_rule_setup(case))
        .bench_local_refs(|input| {
            run_ruleset(&mut input.egraph, &input.ruleset).unwrap();
            input.egraph.serialize(egglog::SerializeConfig::default());
        });
}

// stress test for the Rust API "tableaction" hot path:
// a single match triggers a tight loop of `ctx.insert(...)` calls.
fn insert_loop_setup(case: RustRuleInsertLoopBenchCase) -> RustRuleBenchInput {
    use egglog::prelude::*;

    common::configure_rayon_once();

    let mut program = String::new();
    program.push_str("(relation R (i64))\n");
    program.push_str("(function f (i64) i64 :no-merge)\n");
    for i in 0..case.n_dummy_funcs {
        use std::fmt::Write;
        let _ = writeln!(&mut program, "(function dummy_f{} (i64) i64 :no-merge)", i);
    }
    program.push_str("(R 0)\n");

    let mut egraph = egglog::EGraph::default();
    egraph.parse_and_run_program(None, &program).unwrap();

    let ruleset = "rust_rule_insert_loop";
    add_ruleset(&mut egraph, ruleset).unwrap();

    rust_rule(
        &mut egraph,
        "rust_rule_insert_loop",
        ruleset,
        vars![x: i64],
        facts![(R x)],
        // insert f(x) = x + 1, f(x+1) = x + 2, ..., f(x+n_ops-1) = x + n_ops in one rule run
        move |ctx, _values| {
            for i in 0..case.n_ops {
                let k = ctx.base_to_value::<i64>(i as i64);
                let y = ctx.base_to_value::<i64>(i as i64 + 1);
                ctx.insert("f", [k, y].into_iter());
            }
            Some(())
        },
    )
    .unwrap();

    RustRuleBenchInput {
        egraph,
        ruleset: ruleset.to_owned(),
    }
}

fn tableaction_hot_path_setup(case: RustRuleTableActionBenchCase) -> RustRuleBenchInput {
    use egglog::prelude::*;

    common::configure_rayon_once();

    let mut program = String::new();
    program.push_str("(relation R (i64))\n");
    program.push_str("(function f (i64) i64 :no-merge)\n");
    for i in 0..case.n_dummy_funcs {
        use std::fmt::Write;
        let _ = writeln!(&mut program, "(function dummy_f{} (i64) i64 :no-merge)", i);
    }
    for i in 0..case.n_facts_input {
        use std::fmt::Write;
        let _ = writeln!(&mut program, "(R {})", i as i64);
    }

    let mut egraph = egglog::EGraph::default();
    egraph.parse_and_run_program(None, &program).unwrap();

    // We split the workload into two rulesets to avoid the "write then read in the
    // same rust_rule callback" visibility pitfall:
    // - fill: insert f(x) = x+1 for all R(x)
    // - read: lookup f(x) and do a cheap union
    let fill_ruleset = "rust_rule_tableaction_hot_path_fill";
    let read_ruleset = "rust_rule_tableaction_hot_path_read";
    add_ruleset(&mut egraph, fill_ruleset).unwrap();
    add_ruleset(&mut egraph, read_ruleset).unwrap();

    rust_rule(
        &mut egraph,
        "rust_rule_tableaction_hot_path_fill",
        fill_ruleset,
        vars![x: i64],
        facts![(R x)],
        move |ctx, values| {
            let [x] = values else { unreachable!() };
            let x = ctx.value_to_base::<i64>(*x);
            let k = ctx.base_to_value::<i64>(x);
            let y = ctx.base_to_value::<i64>(x + 1);

            // Populate f(x)=x+1. (No lookup here; it may not be visible within the same callback.)
            ctx.insert("f", [k, y].into_iter());
            Some(())
        },
    )
    .unwrap();

    rust_rule(
        &mut egraph,
        "rust_rule_tableaction_hot_path_read",
        read_ruleset,
        vars![x: i64],
        facts![(R x)],
        move |ctx, values| {
            let [x] = values else { unreachable!() };
            let x = ctx.value_to_base::<i64>(*x);
            let k = ctx.base_to_value::<i64>(x);
            let y = ctx.base_to_value::<i64>(x + 1);

            // Stress the Rust API table ops in the action:
            // lookup should succeed because we pre-filled the table.
            let out = ctx.lookup("f", &[k]).expect("f(x) should exist");
            ctx.union(out, y);
            Some(())
        },
    )
    .unwrap();

    run_ruleset(&mut egraph, fill_ruleset).unwrap();

    RustRuleBenchInput {
        egraph,
        ruleset: read_ruleset.to_owned(),
    }
}

// Mimics eggplant's `math-microbenchmark` hotspot pattern:
// many matches, and each match does several `RustRuleContext` table ops
// (insert + lookup + union). Also inflates the number of tables to make any
// per-match table-action cloning/lookup overhead visible.
// which is more representative of real-world Rust rule usage patterns than insert_loop_setup.
#[divan::bench(
    args = [
        RustRuleTableActionBenchCase { n_facts_input: 50_000, n_dummy_funcs: 200 },
    ],
    sample_count = 10
)]
fn rust_rule_tableaction_hot_path(bencher: divan::Bencher, case: RustRuleTableActionBenchCase) {
    use egglog::prelude::run_ruleset;

    bencher
        .with_inputs(|| tableaction_hot_path_setup(case))
        .bench_local_refs(|input| {
            run_ruleset(&mut input.egraph, &input.ruleset).unwrap();
        });
}

#[divan::bench(sample_count = 10)]
fn rust_rule_math_microbenchmark(bencher: divan::Bencher) {
    bencher
        .with_inputs(math_microbenchmark_support::math_microbenchmark_setup)
        .bench_local_refs(|input| {
            math_microbenchmark_support::run_math_microbenchmark(input);
        });
}

#[divan::bench(
    args = [
        RustRuleInsertLoopBenchCase { n_ops: 1_000, n_dummy_funcs: 200 },
    ],
    sample_count = 10
)]
fn rust_rule_insert_loop(bencher: divan::Bencher, case: RustRuleInsertLoopBenchCase) {
    use egglog::prelude::run_ruleset;

    bencher
        .with_inputs(|| insert_loop_setup(case))
        .bench_local_refs(|input| {
            run_ruleset(&mut input.egraph, &input.ruleset).unwrap();
        });
}

fn main() {
    divan::main();
}

fn fib_setup() -> RustRuleBenchInput {
    use egglog::prelude::*;
    common::configure_rayon_once();

    let mut program = String::new();
    program.push_str("(function fib (i64) i64 :no-merge)");
    program.push_str("(set (fib 0) 0)\n");
    program.push_str("(set (fib 1) 1)\n");
    let mut egraph = egglog::EGraph::default();

    egraph.parse_and_run_program(None, &program).unwrap();
    let ruleset = "fib_ruleset";
    add_ruleset(&mut egraph, ruleset).unwrap();
    rust_rule(
        &mut egraph,
        "fib_rule",
        ruleset,
        vars![x: i64, f0: i64, f1: i64],
        facts![
            (= f0 (fib x))
            (= f1 (fib (+ x 1)))
        ],
        move |ctx, values| {
            let [x, f0, f1] = values else { unreachable!() };
            let x = ctx.value_to_base::<i64>(*x);
            let f0 = ctx.value_to_base::<i64>(*f0);
            let f1 = ctx.value_to_base::<i64>(*f1);

            let y = ctx.base_to_value::<i64>(x + 2);
            let f2 = ctx.base_to_value::<i64>(f0 + f1);
            ctx.insert("fib", [y, f2].into_iter());

            Some(())
        },
    )
    .expect("setupt rule failed");

    RustRuleBenchInput {
        egraph,
        ruleset: ruleset.to_owned(),
    }
}

// basic sanity check for the Rust rule API, and a microbenchmark for the
// overhead of running a Rust rule with a non-trivial match condition
// (many matches, but the rule body does almost no work).
#[divan::bench(
    args = [
        RustRuleBenchCase { n_facts_input: None, n_rule_run_estimated: Some(1_000) },
    ],
    sample_count = 10
)]
fn rust_rule_fib(bencher: divan::Bencher, case: RustRuleBenchCase) {
    use egglog::prelude::run_ruleset;

    bencher
        .with_inputs(|| fib_setup())
        .bench_local_refs(|input| {
            for _ in 0..case.n_rule_run_estimated.unwrap() {
                run_ruleset(&mut input.egraph, &input.ruleset).unwrap();
            }
            input.egraph.serialize(egglog::SerializeConfig::default());
            // let mut f = std::fs::File::create("./fib.dot").unwrap();
            // f.write_all(output.egraph.to_dot().as_bytes()).unwrap();
        });
}
