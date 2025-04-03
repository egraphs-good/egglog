mod common;
use codspeed_criterion_compat::{criterion_group, criterion_main, Criterion};

fn bench(c: &mut Criterion) {
    common::criterion_benchmark(c, "benchmarks/**/raytrace.egg");
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(2);
    targets = bench
);
criterion_main!(benches);
