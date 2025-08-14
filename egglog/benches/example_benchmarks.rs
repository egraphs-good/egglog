// TODO rename this file to something better after codspeed supports renaming

mod common;
use codspeed_criterion_compat::{criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    common::benchmark_files_in_glob(c, "tests/**/*.egg");
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = criterion_benchmark
);
criterion_main!(benches);
