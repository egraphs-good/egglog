use codspeed_criterion_compat::{criterion_group, criterion_main, Criterion};
use egglog::EGraph;

// Only benchmark longer running examples,
// because many of the short ones have too much variance due to the allocator being non deterministic.
// https://github.com/oxc-project/backlog/issues/89

const BENCHMARKS: &[&str] = &[
    "eggcc-extraction",
    "math-microbenchmark",
    "herbie",
    "typeinfer",
    "lambda",
    "python_array_optimize",
];

fn run_example(filename: &str, program: &str) {
    EGraph::default()
        .parse_and_run_program(Some(filename.to_owned()), program)
        .unwrap();
}

pub fn criterion_benchmark(c: &mut Criterion) {
    for name in BENCHMARKS {
        let filename = format!("tests/{}.egg", name);
        let program = std::fs::read_to_string(&filename).unwrap();
        c.bench_function(name, |b| b.iter(|| run_example(&filename, &program)));
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
