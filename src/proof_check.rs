//! Standalone proof checking functionality that doesn't require running the program

use crate::proof_checker::{ProofChecker, Proposition};
use crate::*;

/// Check a term proof against a program without running the program.
/// Creates a fresh egraph internally just for type information and desugaring.
pub fn check_term_proof(
    program_str: &str,
    store: &mut ProofStore,
    proof_id: TermProofId,
) -> Result<Proposition, Error> {
    // Create a fresh egraph just for parsing and type info
    let mut egraph = EGraph::default();

    // Use desugar_program to process the program and populate TypeInfo
    // This will resolve and typecheck everything without running it
    let _ = egraph.desugar_program(None, program_str)?;

    // Parse the program to get the original commands for the proof checker
    let commands_for_checker = egraph.parser.get_program_from_string(None, program_str)?;

    // Create ProofChecker with the populated TypeInfo
    let mut checker = ProofChecker::new(store, commands_for_checker, &egraph.type_info);
    checker
        .check_term_proof(proof_id)
        .map_err(|e| Error::ProofError(format!("{:?}", e)))
}

/// Check an equality proof against a program without running the program.
pub fn check_eq_proof(
    program_str: &str,
    store: &mut ProofStore,
    proof_id: EqProofId,
) -> Result<Proposition, Error> {
    // Create a fresh egraph just for parsing and type info
    let mut egraph = EGraph::default();

    // Use desugar_program to process the program and populate TypeInfo
    // This will resolve and typecheck everything without running it
    let _ = egraph.desugar_program(None, program_str)?;

    // Parse the program to get the original commands for the proof checker
    let commands_for_checker = egraph.parser.get_program_from_string(None, program_str)?;

    // Create ProofChecker with the populated TypeInfo
    let mut checker = ProofChecker::new(store, commands_for_checker, &egraph.type_info);
    checker
        .check_eq_proof(proof_id)
        .map_err(|e| Error::ProofError(format!("{:?}", e)))
}
