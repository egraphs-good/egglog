// TODO rename this file to something better after codspeed supports renaming

mod common;

#[divan::bench(args = common::bench_cases("tests/**/*.egg"), sample_count = 10)]
fn run_example_cases(case: &common::BenchCase) {
    common::bench_case(case);
}

fn main() {
    divan::main();
}
