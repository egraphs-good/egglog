mod common;

#[divan::bench(args = common::bench_cases("tests/**/*.egg"), sample_count = 10)]
fn tests(case: &common::BenchCase) {
    common::bench_case(case);
}

fn main() {
    divan::main();
}
