// TODO rename this file to something better after codspeed supports renaming

mod common;
use codspeed_criterion_compat::{Criterion, criterion_group, criterion_main};

fn criterion_benchmark(c: &mut Criterion) {
    common::benchmark_files_in_glob(c, "tests/**/*.egg");
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10).measurement_time(std::time::Duration::from_millis(10));
    targets = criterion_benchmark
);
criterion_main!(benches);
