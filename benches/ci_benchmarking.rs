mod common;

use codspeed_criterion_compat::{criterion_group, criterion_main, Criterion};

fn bench(c: &mut Criterion) {
    common::criterion_benchmark(c, "tests/**/*.egg");
}

criterion_group!(benches, bench);
criterion_main!(benches);
