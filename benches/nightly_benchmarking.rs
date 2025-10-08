mod common;
use codspeed_criterion_compat::{Criterion, criterion_group, criterion_main};

fn criterion_benchmark(c: &mut Criterion) {
    common::benchmark_files_in_glob(c, "egglog-benchmarks/**/*.egg");
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = criterion_benchmark
);
criterion_main!(benches);
