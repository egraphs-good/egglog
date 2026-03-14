#[derive(Clone, Copy)]
struct RustRuleBenchCase {
    n_facts: usize,
    n_funcs: usize,
}

impl std::fmt::Display for RustRuleBenchCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rust_rule_facts{}_funcs{}", self.n_facts, self.n_funcs)
    }
}

struct RustRuleBenchInput {
    egraph: egglog::EGraph,
    ruleset: String,
}

fn rust_rule_setup(case: RustRuleBenchCase) -> RustRuleBenchInput {
    use egglog::prelude::*;

    static CONFIGURE_RAYON: std::sync::Once = std::sync::Once::new();
    CONFIGURE_RAYON.call_once(|| {
        // `build_global` can only be called once; if another benchmark already initialized the
        // global pool we just keep that configuration.
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global();
    });

    let mut program = String::new();
    program.push_str("(relation R (i64))\n");

    for i in 0..case.n_funcs {
        use std::fmt::Write;
        let _ = writeln!(&mut program, "(function dummy_f{} (i64) i64 :no-merge)", i);
    }
    for i in 0..case.n_facts {
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
        RustRuleBenchCase { n_facts: 1_000, n_funcs: 50 },
        RustRuleBenchCase { n_facts: 10_000, n_funcs: 50 },
        RustRuleBenchCase { n_facts: 10_000, n_funcs: 200 },
    ],
    sample_count = 10
)]
fn rust_rule_match_overhead(bencher: divan::Bencher, case: RustRuleBenchCase) {
    use egglog::prelude::run_ruleset;

    bencher
        .with_inputs(|| rust_rule_setup(case))
        .bench_local_refs(|input| {
            run_ruleset(&mut input.egraph, &input.ruleset).unwrap();
        });
}

#[divan::bench(
    args = [
        RustRuleBenchCase { n_facts: 1_000, n_funcs: 50 },
        RustRuleBenchCase { n_facts: 10_000, n_funcs: 50 },
        RustRuleBenchCase { n_facts: 10_000, n_funcs: 200 },
    ],
    sample_count = 10
)]
fn rust_rule_match_overhead_plus_serialize(bencher: divan::Bencher, case: RustRuleBenchCase) {
    use egglog::prelude::run_ruleset;

    bencher
        .with_inputs(|| rust_rule_setup(case))
        .bench_local_refs(|input| {
            run_ruleset(&mut input.egraph, &input.ruleset).unwrap();
            input.egraph.serialize(egglog::SerializeConfig::default());
        });
}

fn main() {
    divan::main();
}
