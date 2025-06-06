use std::sync::Once;

use codspeed_criterion_compat::{criterion_group, criterion_main, Criterion};
use egglog::EGraph;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

static CONFIGURE_RAYON: Once = Once::new();

fn run_example(filename: &str, program: &str, no_messages: bool) {
    let mut egraph = EGraph::default();
    if no_messages {
        egraph.disable_messages();
    }
    egraph
        .parse_and_run_program(Some(filename.to_owned()), program)
        .unwrap();
    // test performance of serialization as well
    let _ = egraph.serialize(egglog::SerializeConfig::default());
}

pub fn criterion_benchmark(c: &mut Criterion) {
    CONFIGURE_RAYON.call_once(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .unwrap();
    });
    for entry in glob::glob("tests/**/*.egg").unwrap() {
        let path = entry.unwrap().clone();
        let path_string = path.to_string_lossy().to_string();
        if path_string.contains("fail-typecheck") {
            continue;
        }
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let filename = path.to_string_lossy().to_string();
        let program = std::fs::read_to_string(&filename).unwrap();
        let no_messages = path_string.contains("no-messages");
        c.bench_function(&name, |b| {
            b.iter(|| run_example(&filename, &program, no_messages))
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
