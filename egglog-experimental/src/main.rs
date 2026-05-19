fn main() {
    // Detect `--duckdb` (and `--duck-native-uf`) in argv before
    // `egglog::cli` parses args. The CLI's own duckdb branch rebuilds
    // the egraph from scratch through `EGraph::with_duckdb_backend`,
    // which would drop the experimental commands (`run-schedule`,
    // `multi-extract`, …) and primitives we register here. Building
    // the duckdb-backed egraph ourselves and extending it with the
    // experimental surface up front keeps those alive — the CLI then
    // sees an already-correct egraph and (since `--duckdb` is set on
    // both sides) short-circuits its own rebuild.
    let argv: Vec<String> = std::env::args().collect();
    let want_duckdb = argv.iter().any(|a| a == "--duckdb");
    let want_native_uf = argv.iter().any(|a| a == "--duck-native-uf");
    // `--proof-testing` implies proofs — the desugar pass rewrites
    // `(check ...)` into `(prove-exists ...)` which needs the proof
    // encoding active. Without this, cli.rs's `args.proof_testing`
    // branch would try `with_proofs_enabled()` after construction,
    // and that path clones the egraph for `original_typechecking`,
    // hitting duckdb's unimplemented `clone_boxed`.
    let want_proofs =
        argv.iter().any(|a| a == "--proofs" || a == "--proof-testing");
    let egraph = if want_duckdb {
        egglog_experimental::new_experimental_egraph_duckdb(egglog::DuckBackendConfig {
            native_uf: want_native_uf,
            proofs: want_proofs,
        })
        .expect("failed to start DuckDB-backed experimental egraph")
    } else {
        egglog_experimental::new_experimental_egraph()
    };
    egglog::cli(egraph)
}
