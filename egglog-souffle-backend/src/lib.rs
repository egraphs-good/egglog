//! Translator from a small intermediate representation (IR) of the
//! souffle_compat-encoded egglog form to Souffle Datalog (`.dl`) source.
//!
//! The IR lives in [`ir`]; the emitter that prints it as Souffle source is in
//! [`emit`]. A mapping from egglog's resolved AST to this IR will live in
//! `egglog::souffle_translator` once we wire it through the main crate.
//!
//! The Souffle dialect this targets is the `oflatt/souffle:bounded-iteration`
//! fork, which adds:
//!   - `.limititerations R(n=N)` — bound a stratum's fixpoint iterations.
//!   - `.pragma "outer-saturate" "<N>"` — wrap the SCC sequence in an outer
//!     fixpoint loop, with cross-stratum delta tracking.
//!   - `.snapshot R_snap(of = "R")` — declare R_snap as a runtime-refreshed
//!     snapshot of R; lets a rule read R_snap without creating a precedence
//!     edge to R.
//!
//! See `souffle/SCHEDULE-DESIGN.md` in the fork.

pub mod emit;
pub mod examples;
pub mod ir;

pub use emit::emit;
pub use ir::*;
