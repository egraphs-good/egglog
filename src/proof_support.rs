//! Proof support checking for EGraph commands.
//!
//! This module contains helpers to check if egglog commands are compatible
//! with proof mode.

use crate::{
    EGraph, Error,
    ast::{Fact, Facts, ResolvedFact, ResolvedNCommand},
    core,
};
use egglog_ast::generic_ast::GenericExpr;

impl EGraph {
    /// Check if a program is compatible with proof mode.
    /// Returns Ok(()) if the program can be run with proofs,
    /// otherwise returns the first ProofsNotSupported error encountered.
    pub fn check_program_supports_proofs(program_str: &str) -> Result<(), Error> {
        // Create a temporary EGraph with proofs enabled to do the checking
        let mut egraph = EGraph::with_proofs();

        // Try to parse and run the program to see if it's supported
        match egraph.parse_and_run_program(None, program_str) {
            Ok(_) => Ok(()),
            Err(e) => {
                // If it's a ProofsNotSupported error, return it
                if matches!(e, Error::ProofsNotSupported(..)) {
                    return Err(e);
                }
                // Other errors might be expected failures
                Ok(())
            }
        }
    }

    /// Check if a command is compatible with proof mode.
    /// This traverses the entire command AST looking for unsupported features.
    pub(crate) fn check_command_supports_proofs(
        &self,
        command: &ResolvedNCommand,
    ) -> Result<(), Error> {
        match command {
            ResolvedNCommand::Function(fdecl) if fdecl.merge.is_some() => {
                Err(Error::ProofsNotSupported(
                    format!(
                        "Function '{}' has a :merge, which is not supported with proofs",
                        fdecl.name
                    ),
                    fdecl.span.clone(),
                ))
            }
            _ => {
                // Traverse all expressions in the command to check for non-validatable primitives
                let mut has_error = None;
                command.clone().visit_exprs(&mut |expr| {
                    if has_error.is_some() {
                        return expr;
                    }
                    // Check if this expression uses a primitive without a registered validator
                    if let GenericExpr::Call(span, core::ResolvedCall::Primitive(prim), _) = &expr {
                        let prim_name = prim.primitive.primitive.name();

                        // Check if a validator is registered for this specific primitive overload
                        let has_validator = self
                            .type_info
                            .get_prims(prim_name)
                            .and_then(|prims| {
                                // Look for the specific overload matching this primitive's id
                                prims.iter().find(|p| p.id == prim.primitive.id)
                            })
                            .and_then(|prim_with_id| prim_with_id.validator.as_ref())
                            .is_some();

                        if !has_validator {
                            has_error = Some(Error::ProofsNotSupported(
                                format!(
                                    "Primitive '{}' does not have a validator registered. Use EGraph::add_primitive_validator to register a validator for this primitive.",
                                    prim_name
                                ),
                                span.clone(),
                            ));
                        }
                    }
                    expr
                });

                if let Some(err) = has_error {
                    return Err(err);
                }
                Ok(())
            }
        }
    }

    /// Validate the proof for a check command by using prove_query.
    pub(crate) fn validate_check_proof(&mut self, facts: &[ResolvedFact]) -> Result<(), Error> {
        use crate::ProofStore;

        // Use prove_query to get a proof for this check query
        let mut proof_store = ProofStore::default();

        // Convert resolved facts to surface syntax
        let surface_facts: Vec<Fact> = facts.iter().map(|f| f.clone().make_unresolved()).collect();

        // Try to get a proof for this query
        let _proof = self
            .prove_query(Facts(surface_facts), &mut proof_store)
            .map_err(|e| Error::ProofError(format!("Failed to prove check query: {}", e)))?;

        // If prove_query succeeds and returns a proof, the check is valid
        // If it returns None, the query has no matches, which means the check fails
        // (but this should have been caught earlier by check_facts)
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_without_validator_rejected() {
        // Create an egraph with proofs enabled
        let mut egraph = EGraph::with_proofs();

        // Try to use a primitive without a validator registered
        // String primitives don't have validators
        let result = egraph.parse_and_run_program(
            None,
            r#"
            (let x (to-string 42))
            "#,
        );

        // This should fail because to-string doesn't have a validator registered
        assert!(
            result.is_err(),
            "Should reject primitive without validator in proof mode"
        );

        if let Err(Error::ProofsNotSupported(msg, _)) = result {
            assert!(
                msg.contains("to-string") && msg.contains("validator"),
                "Error should mention the primitive and validator, got: {}",
                msg
            );
        } else {
            panic!("Expected ProofsNotSupported error");
        }
    }

    #[test]
    fn test_merge_function_rejected() {
        // Create an egraph with proofs enabled
        let mut egraph = EGraph::with_proofs();

        // Try to define a function with :merge
        let result = egraph.parse_and_run_program(
            None,
            r#"
            (function my-func (i64) i64 :merge (min old new))
            "#,
        );

        // This should fail because :merge is not supported with proofs
        assert!(
            result.is_err(),
            "Should reject :merge functions in proof mode"
        );

        if let Err(Error::ProofsNotSupported(msg, _)) = result {
            assert!(
                msg.contains("merge"),
                "Error should mention merge, got: {}",
                msg
            );
        } else {
            panic!("Expected ProofsNotSupported error");
        }
    }
}
