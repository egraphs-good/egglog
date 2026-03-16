mod common;

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

// match only rust rule test, to isolate the overhead of matching a rust rule without any actual work in the rule body
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

// Ultra-minimal stress test for the Rust API "tableaction" hot path:
// a single match triggers a tight loop of `ctx.insert(...)` calls.
fn insert_loop_setup(case: RustRuleInsertLoopBenchCase) -> RustRuleBenchInput {
    use egglog::prelude::*;

    common::configure_rayon_once();

    let mut program = String::new();
    program.push_str("(relation R (i64))\n");
    program.push_str("(function f (i64) i64 :no-merge)\n");
    for i in 0..case.n_dummy_funcs {
        use std::fmt::Write;
        let _ = writeln!(
            &mut program,
            "(function dummy_f{} (i64) i64 :no-merge)",
            i
        );
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

// Mimics eggplant's `math-microbenchmark` hotspot pattern:
// many matches, and each match does several `RustRuleContext` table ops
// (insert + lookup + union). Also inflates the number of tables to make any
// per-match table-action cloning/lookup overhead visible.
fn tableaction_hot_path_setup(case: RustRuleTableActionBenchCase) -> RustRuleBenchInput {
    use egglog::prelude::*;

    common::configure_rayon_once();

    let mut program = String::new();
    program.push_str("(relation R (i64))\n");
    program.push_str("(function f (i64) i64 :no-merge)\n");
    for i in 0..case.n_dummy_funcs {
        use std::fmt::Write;
        let _ = writeln!(
            &mut program,
            "(function dummy_f{} (i64) i64 :no-merge)",
            i
        );
    }
    for i in 0..case.n_facts_input {
        use std::fmt::Write;
        let _ = writeln!(&mut program, "(R {})", i as i64);
    }

    let mut egraph = egglog::EGraph::default();
    egraph.parse_and_run_program(None, &program).unwrap();

    let ruleset = "rust_rule_tableaction_hot_path";
    add_ruleset(&mut egraph, ruleset).unwrap();

    rust_rule(
        &mut egraph,
        "rust_rule_tableaction_hot_path",
        ruleset,
        vars![x: i64],
        facts![(R x)],
        move |ctx, values| {
            let [x] = values else { unreachable!() };
            let x = ctx.value_to_base::<i64>(*x);
            let k = ctx.base_to_value::<i64>(x);
            let y = ctx.base_to_value::<i64>(x + 1);

            // A minimal “hot path” that stresses the Rust API table ops.
            ctx.insert("f", [k, y].into_iter());
            let out = ctx.lookup("f", &[k]).expect("just inserted");
            ctx.union(out, y);

            Some(())
        },
    )
    .unwrap();

    RustRuleBenchInput {
        egraph,
        ruleset: ruleset.to_owned(),
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

#[divan::bench(
    args = [
        RustRuleTableActionBenchCase { n_facts_input: 50_000, n_dummy_funcs: 200 },
    ],
    sample_count = 10
)]
fn rust_rule_tableaction_hot_path_with_serialize(
    bencher: divan::Bencher,
    case: RustRuleTableActionBenchCase,
) {
    use egglog::prelude::run_ruleset;

    bencher
        .with_inputs(|| tableaction_hot_path_setup(case))
        .bench_local_refs(|input| {
            run_ruleset(&mut input.egraph, &input.ruleset).unwrap();
            input.egraph.serialize(egglog::SerializeConfig::default());
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

#[divan::bench(
    args = [
        RustRuleBenchCase { n_facts_input: None, n_rule_run_estimated: Some(1_000)},
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
