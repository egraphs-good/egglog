mod common;

#[divan::bench(args = common::bench_cases("tests/**/*.egg"), sample_count = 10)]
fn tests(case: &common::BenchCase) {
    common::bench_case(case);
}

#[divan::bench(args = common::bench_cases_duckdb("tests/**/*.egg"), sample_count = 10)]
fn duckdb(case: &common::BenchCase) {
    common::bench_case(case);
}

fn main() {
    divan::main();
}
