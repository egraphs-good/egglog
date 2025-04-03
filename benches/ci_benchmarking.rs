mod common;

use codspeed_criterion_compat::{criterion_group, criterion_main, Criterion};

fn bench(c: &mut Criterion) {
    common::criterion_benchmark(c, "tests/**/*.egg");
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench
);
criterion_main!(benches);
