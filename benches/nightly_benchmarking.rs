mod common;
use codspeed_criterion_compat::{criterion_group, criterion_main, Criterion};

fn benchmark(c: &mut Criterion) {
    common::criterion_benchmark(c, "benchmarks/**/raytrace.egg");
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
