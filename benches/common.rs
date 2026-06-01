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

/// Run a program on the DuckDB backend (term-encoding only, no proofs).
/// Mirrors how `tests/files.rs` builds the `--duckdb` egraph.
pub fn run_example_duckdb(filename: &str, program: &str) {
    let config = egglog::DuckBackendConfig {
        proofs: false,
        native_uf: false,
    };
    let mut egraph =
        EGraph::with_duckdb_backend(config).expect("EGraph::with_duckdb_backend init failed");
    egraph.ensure_no_reserved_symbols(false);
    egraph
        .parse_and_run_program(Some(filename.to_owned()), program)
        .unwrap();
    // The duckdb backend doesn't implement `serialize`, so unlike
    // `run_example` we don't measure serialization here.
    std::mem::forget(egraph);
}

#[derive(Clone)]
pub struct BenchCase {
    pub name: String,
    pub filename: String,
    pub program: String,
    pub proof_testing: bool,
    pub duckdb: bool,
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
                duckdb: false,
            }
        });
    cases.extend(regular_cases);

    // Add proof testing cases
    cases.extend(bench_cases_proof_testing(glob));

    cases
}

/// Bench cases for the DuckDB backend. Mirrors the `--duckdb` test
/// gating: only proof-encodable files (`file_supports_proofs`), minus
/// the static skip list and anything using `(push)`/`(pop)` (no
/// savepoint support).
pub fn bench_cases_duckdb(glob: &str) -> Vec<BenchCase> {
    configure_rayon_once();

    glob::glob(glob)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|path| !path.to_string_lossy().contains("fail-typecheck"))
        .filter(|path| egglog::file_supports_proofs(path))
        .filter(|path| !DUCKDB_STATIC_SKIP.iter().any(|f| path.ends_with(f)))
        .filter_map(|path| {
            let filename = path.to_string_lossy().to_string();
            let program = std::fs::read_to_string(&filename).ok()?;
            if program.contains("(push") || program.contains("(pop") {
                return None;
            }
            let stem = path.file_stem().unwrap().to_string_lossy().to_string();
            Some(BenchCase {
                name: format!("duckdb_{stem}"),
                filename,
                program,
                proof_testing: false,
                duckdb: true,
            })
        })
        .collect()
}

const PROOF_UNSUPPORTED_FILES: &[&str] = &[
    "math-microbenchmark.egg",
    "subsume.egg",
    "subsume-relation.egg",
];

// Files the duckdb backend can't run, mirroring `tests/files.rs`'s
// `duckdb_static_skip`: math-microbenchmark is too slow, eggcc-2mm
// declares a `(Set Expr)` container sort the proof encoding can't
// represent, and the subsume files rely on `check`ing a subsumed term.
// Everything else is gated by `file_supports_proofs` plus a push/pop
// check, exactly like the duckdb test variants.
const DUCKDB_STATIC_SKIP: &[&str] = &[
    "math-microbenchmark.egg",
    "eggcc-2mm.egg",
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
                duckdb: false,
            }
        })
        .collect()
}

pub fn bench_case(case: &BenchCase) {
    configure_rayon_once();

    if case.duckdb {
        run_example_duckdb(&case.filename, &case.program);
    } else {
        run_example(&case.filename, &case.program, case.proof_testing);
    }
}

pub fn configure_rayon_once() {
    CONFIGURE_RAYON.call_once(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .unwrap();
    });
}
