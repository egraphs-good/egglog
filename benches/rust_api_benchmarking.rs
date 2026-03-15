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
fn rust_rule_match_overhead_plus_serialize(bencher: divan::Bencher, case: RustRuleBenchCase) {
    use egglog::prelude::run_ruleset;

    bencher
        .with_inputs(|| match_only_rust_rule_setup(case))
        .bench_local_refs(|input| {
            run_ruleset(&mut input.egraph, &input.ruleset).unwrap();
            input.egraph.serialize(egglog::SerializeConfig::default());
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
fn fib(bencher: divan::Bencher, case: RustRuleBenchCase) {
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
