mod common;

#[divan::bench(args = common::bench_cases("tests/**/*.egg"), sample_count = 10)]
fn tests(case: &common::BenchCase) {
    common::bench_case(case);
}

#[divan::bench(args = common::bench_cases_proof_testing("tests/**/*.egg"), sample_count = 10)]
fn proof_testing(case: &common::BenchCase) {
    common::bench_case_proof_testing(case);
}

fn main() {
    divan::main();
}
