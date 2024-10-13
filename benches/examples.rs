use std::path::Path;

use codspeed_criterion_compat::{criterion_group, criterion_main, BenchmarkId, Criterion};
use egglog::EGraph;

fn run_example(name: &Path) {
    let filename = name.to_string_lossy().to_string();
    let program = std::fs::read_to_string(&filename).unwrap();
    let mut egraph = EGraph::default();
    egraph.set_reserved_symbol("___".into());
    egraph
        .parse_and_run_program(Some(filename), &program)
        .unwrap();
}

pub fn criterion_benchmark(c: &mut Criterion) {
    for entry in glob::glob("tests/**/*.egg").unwrap() {
        let path = entry.unwrap().clone();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        if path
            .to_string_lossy()
            .to_string()
            .contains("fail-typecheck")
        {
            continue;
        }
        c.bench_with_input(BenchmarkId::new("example", &name), &path, |_b, f| {
            run_example(f)
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
