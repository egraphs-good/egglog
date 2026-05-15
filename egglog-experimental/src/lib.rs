//! # egglog-experimental
//!
//! This crate layers several experimental features on top of the core
//! [`egglog`](https://github.com/egraphs-good/egglog) language and runtime.
//! It can serve as a standard library when building equality
//! saturation workflows in Rust.
//!
//! ## Implemented extensions
//!
//! - [`for`-loops](https://egraphs-good.github.io/egglog-demo/?example=for)
//! - [`with-ruleset`](https://egraphs-good.github.io/egglog-demo/?example=with-ruleset)
//! - [Rationals support](https://egraphs-good.github.io/egglog-demo/?example=rational)
//!   (see [`rational`] for the exposed primitives)
//! - [Dynamic cost models with `set-cost`](https://egraphs-good.github.io/egglog-demo/?example=05-cost-model-and-extraction)
//! - [Custom schedulers via `run-with`](https://egraphs-good.github.io/egglog-demo/?example=math-backoff)
//! - [`(get-size!)` primitive](https://github.com/egraphs-good/egglog-experimental/blob/main/tests/web-demo/node-limit.egg)
//!   for inspecting total tuple counts, optionally restricted to specific tables
//! - [Multi-extraction](https://github.com/egraphs-good/egglog-experimental/blob/main/tests/web-demo/multi-extract.egg)
//!
//! Each bullet links to a runnable demo so you can explore the feature quickly.
//! The rest of this crate exposes the Rust APIs and helpers that back these extensions.
//!
use egglog::ast::Parser;
use egglog::prelude::{RustSpan, Span, add_base_sort};
pub use egglog::*;
use std::sync::Arc;

pub mod rational;
pub use rational::*;
mod scheduling;
pub use scheduling::*;
mod fresh_macro;

mod set_cost;
pub use set_cost::*;
mod multi_extract;
pub use multi_extract::*;
mod size;
pub use size::*;

// Sugar modules using parse-time macros
mod sugar;
pub use sugar::*;

pub fn new_experimental_egraph() -> EGraph {
    let mut egraph = EGraph::default();
    extend_with_experimental(&mut egraph);
    egraph
}

/// Build an experimental egraph backed by DuckDB. Same surface as
/// [`new_experimental_egraph`] but the underlying storage / rule
/// execution go through `egglog_bridge_duckdb`. The frontend's
/// `--duckdb` CLI flag still flows through `egglog::cli`; this
/// constructor is for callers that want a duckdb-backed engine
/// up front (e.g. `egglog-experimental`'s `main`, which has to
/// register `run-schedule` / `multi-extract` and the rational
/// primitives *before* `cli` runs its program).
pub fn new_experimental_egraph_duckdb(
    config: egglog::DuckBackendConfig,
) -> anyhow::Result<EGraph> {
    let mut egraph = EGraph::with_duckdb_backend(config)?;
    extend_with_experimental(&mut egraph);
    Ok(egraph)
}

/// Register all experimental sorts, primitives, commands, and parser
/// macros on the given egraph in place. Factored out so the same
/// extensions can be applied to either a default (bridge-backed) or a
/// DuckDB-backed [`EGraph`] before `egglog::cli` runs.
pub fn extend_with_experimental(egraph: &mut EGraph) {
    // Set up the parser with experimental parse-time macros
    egraph.parser = experimental_parser();

    // Rational support
    add_base_sort(egraph, RationalSort, span!()).unwrap();

    // Support for set cost
    add_set_cost(egraph);
    egraph.add_primitive(GetSizePrimitive);

    // unstable-fresh! macro
    egraph
        .command_macros_mut()
        .register(Arc::new(fresh_macro::FreshMacro::new()));

    // scheduler support
    egraph
        .add_command("run-schedule".into(), Arc::new(RunExtendedSchedule))
        .unwrap();

    egraph
        .add_command(
            "multi-extract".into(),
            Arc::new(MultiExtract::new(DynamicCostModel)),
        )
        .unwrap();
}

// Create a parser with experimental macros
pub fn experimental_parser() -> Parser {
    let mut parser = Parser::default();
    parser.add_command_macro(Arc::new(sugar::For));
    parser.add_command_macro(Arc::new(sugar::WithRuleset));
    parser
}
