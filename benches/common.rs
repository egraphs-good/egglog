use egglog::EGraph;
use std::{fmt, sync::Once};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

static CONFIGURE_RAYON: Once = Once::new();

pub fn run_example(filename: &str, program: &str, proof_testing: bool) {
    let mut egraph = if proof_testing {
        EGraph::new_with_proofs().with_proof_testing()
    } else {
        EGraph::default()
    };
    egraph
        .parse_and_run_program(Some(filename.to_owned()), program)
        .unwrap();
    // test performance of serialization as well
    egraph.serialize(egglog::SerializeConfig::default());
    // We don't destruct the e-graph in CLI mode.
    std::mem::forget(egraph);
}

#[derive(Clone)]
pub struct BenchCase {
    pub name: String,
    pub filename: String,
    pub program: String,
    pub proof_testing: bool,
}

impl fmt::Display for BenchCase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.name)
    }
}

pub fn bench_cases(glob: &str) -> Vec<BenchCase> {
    configure_rayon_once();

    let mut cases = Vec::new();

    // Add regular test cases
    let regular_cases = glob::glob(glob)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|path| !path.to_string_lossy().contains("fail-typecheck"))
        .filter(|path| !path.to_string_lossy().contains("proofs"))
        .map(|path| {
            let filename = path.to_string_lossy().to_string();
            let program = std::fs::read_to_string(&filename).unwrap();
            let name = path.file_stem().unwrap().to_string_lossy().to_string();

            BenchCase {
                name,
                filename,
                program,
                proof_testing: false,
            }
        });
    cases.extend(regular_cases);

    // Add proof testing cases
    cases.extend(bench_cases_proof_testing(glob));

    cases
}

const PROOF_UNSUPPORTED_FILES: &[&str] = &[
    "math-microbenchmark.egg",
    "rectangle.egg",
    "subsume.egg",
    "subsume-relation.egg",
];

pub fn bench_cases_proof_testing(glob: &str) -> Vec<BenchCase> {
    configure_rayon_once();

    glob::glob(glob)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|path| !path.to_string_lossy().contains("fail-typecheck"))
        .filter(|path| egglog::file_supports_proofs(path))
        .filter(|path| !PROOF_UNSUPPORTED_FILES.iter().any(|f| path.ends_with(f)))
        .map(|path| {
            let filename = path.to_string_lossy().to_string();
            let program = std::fs::read_to_string(&filename).unwrap();
            let stem = path.file_stem().unwrap().to_string_lossy().to_string();
            let name = format!("proof_testing_{stem}");

            BenchCase {
                name,
                filename,
                program,
                proof_testing: true,
            }
        })
        .collect()
}

pub fn bench_case(case: &BenchCase) {
    configure_rayon_once();

    run_example(&case.filename, &case.program, case.proof_testing);
}

pub fn configure_rayon_once() {
    CONFIGURE_RAYON.call_once(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .unwrap();
    });
}
