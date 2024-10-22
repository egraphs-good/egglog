use codspeed_criterion_compat::{criterion_group, criterion_main, Criterion};
use egglog::EGraph;

fn run_example(filename: &str, program: &str) {
    EGraph::default()
        .parse_and_run_program(Some(filename.to_owned()), program)
        .unwrap();
}

pub fn criterion_benchmark(c: &mut Criterion) {
    for entry in glob::glob("tests/**/*.egg").unwrap() {
        let path = entry.unwrap().clone();
        let path_string = path.to_string_lossy().to_string();
        if path_string.contains("fail-typecheck") {
            continue;
        }
        // skip python_array_optimize since it is too slow and doesn't even reflect the current python implementation
        if path_string.contains("python_array_optimize") {
            continue;
        }

        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let filename = path.to_string_lossy().to_string();
        let program = std::fs::read_to_string(&filename).unwrap();
        c.bench_function(&name, |b| b.iter(|| run_example(&filename, &program)));
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
