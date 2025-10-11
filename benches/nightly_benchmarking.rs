mod common;

#[divan::bench(args = common::bench_cases("egglog-benchmarks/**/*.egg"), sample_size = 10)]
fn benchmarks(case: &common::BenchCase) {
    common::bench_case(case);
}

fn main() {
    divan::main();
}
