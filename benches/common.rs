use codspeed_criterion_compat::Criterion;
use egglog::EGraph;
use std::sync::Once;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

static CONFIGURE_RAYON: Once = Once::new();

pub fn run_example(filename: &str, program: &str, no_messages: bool) {
    let mut egraph = EGraph::default();
    if no_messages {
        egraph.disable_messages();
    }
    egraph
        .parse_and_run_program(Some(filename.to_owned()), program)
        .unwrap();
    // test performance of serialization as well
    let (_ser, _trimmed) = egraph.serialize(egglog::SerializeConfig::default());
}

pub fn benchmark_files_in_glob(c: &mut Criterion, glob: &str) {
    CONFIGURE_RAYON.call_once(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .unwrap();
    });
    for entry in glob::glob(glob).unwrap() {
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
