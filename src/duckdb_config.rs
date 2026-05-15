//! Configuration for the DuckDB backend.
//!
//! The two flags here govern how
//! [`EGraph::with_duckdb_backend`](crate::EGraph::with_duckdb_backend)
//! sets up its underlying [`egglog_bridge_duckdb::EGraph`] and how the
//! egglog frontend instruments the program before handing it off:
//!
//! - [`native_uf`](DuckBackendConfig::native_uf) toggles the in-process
//!   union-find replacement for the term-encoded
//!   `@_u_f___<sort>f` table.
//! - [`proofs`](DuckBackendConfig::proofs) runs term encoding in
//!   proof-tracking mode.

/// Knobs for [`EGraph::with_duckdb_backend`](crate::EGraph::with_duckdb_backend).
#[derive(Clone, Default, Debug)]
pub struct DuckBackendConfig {
    /// `--duck-native-uf`: maintain term-encoding's UF
    /// (`@_u_f___<sort>f` function-form) as an in-process union-find
    /// data structure, with reads going through a DuckDB scalar UDF
    /// instead of a JOIN. Skips the singleparent / path_compress /
    /// uf_function_index rulesets at run time. Experimental.
    pub native_uf: bool,
    /// Run the encoder in proof-tracking mode. The term-encoded
    /// program then threads explicit proofs through every union and
    /// rewrite, exercising a different control-flow path through the
    /// encoder. Set by tests to confirm the duckdb-side optimizations
    /// (inline-congruence, native UF, hash-cons) don't regress proof
    /// mode.
    pub proofs: bool,
}
