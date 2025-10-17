use egglog::EGraph;
use std::{sync::Once, fmt};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

static CONFIGURE_RAYON: Once = Once::new();

pub fn run_example(filename: &str, program: &str) {
    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(Some(filename.to_owned()), program)
        .unwrap();
    // test performance of serialization as well
    egraph.serialize(egglog::SerializeConfig::default());
}

#[derive(Clone)]
pub struct BenchCase {
    pub name: String,
    pub filename: String,
    pub program: String,
}

impl fmt::Display for BenchCase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.name)
    }
}

pub fn bench_cases(glob: &str) -> Vec<BenchCase> {
    configure_rayon_once();

    glob::glob(glob)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|path| !path.to_string_lossy().contains("fail-typecheck"))
        .map(|path| {
            let filename = path.to_string_lossy().to_string();
            let program = std::fs::read_to_string(&filename).unwrap();
            let name = path.file_stem().unwrap().to_string_lossy().to_string();

            BenchCase {
                name,
                filename,
                program,
            }
        })
        .collect()
}

pub fn bench_case(case: &BenchCase) {
    configure_rayon_once();

    run_example(&case.filename, &case.program);
}

fn configure_rayon_once() {
    CONFIGURE_RAYON.call_once(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .unwrap();
    });
}
