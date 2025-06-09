use codspeed_criterion_compat::Criterion;
use egglog::EGraph;

pub fn run_example(filename: &str, program: &str, no_messages: bool) {
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

pub fn benchmark_files_in_glob(c: &mut Criterion, glob: &str) {
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
