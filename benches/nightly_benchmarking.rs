mod common;

#[divan::bench(args = common::bench_cases("egglog-benchmarks/**/*.egg"), sample_size = 10)]
fn run_nightly_cases(case: &common::BenchCase) {
    common::bench_case(case);
}

fn main() {
    divan::main();
}
