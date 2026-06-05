use crate::ast::FunctionSubtype;
use crate::extract::{Extractor, TreeAdditiveCostModel};
use crate::proofs::proof_checker::ProofCheckError;
use crate::proofs::proof_encoding::ProofInstrumentor;
use crate::proofs::proof_format::{Justification, ProofId, ProofStore, proof_store_from_term};
use crate::{ResolvedCall, TermDag};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProveExistsError {
    #[error("prove-exists requires a constructor")]
    RequiresConstructor,
    #[error("prove-exists does not support primitives")]
    PrimitivesUnsupported,
    #[error("Could not find a proof due to query not matching (constructor {constructor}).")]
    QueryDidNotMatch { constructor: String },
    #[error("prove/prove-exists requires proofs to be enabled (run with --proofs).")]
    ProofsNotEnabled,
    #[error("constructor {constructor} is not provable by rule for an existence proof.")]
    NotProvableByRule { constructor: String },
    #[error("constructor {constructor} is not declared.")]
    ConstructorNotDeclared { constructor: String },
    #[error("no proof recorded for constructor {constructor}.")]
    NoProofRecorded { constructor: String },
    #[error("failed to extract proof term for constructor {constructor}.")]
    ProofTermExtractionFailed { constructor: String },
    #[error(transparent)]
    ProofCheck(#[from] ProofCheckError),
}

impl ProofInstrumentor<'_> {
    /// Prove the existence of a constructor or fail if a proof cannot be found.
    /// We use a constructor because inserting a value at the top level would give a trivial proof.
    pub(crate) fn prove_exists(
        &mut self,
        call: &ResolvedCall,
    ) -> Result<(ProofStore, ProofId), ProveExistsError> {
        let func = match call {
            ResolvedCall::Func(func) if func.subtype == FunctionSubtype::Constructor => func,
            ResolvedCall::Func(_) => {
                return Err(ProveExistsError::RequiresConstructor);
            }
            ResolvedCall::Primitive(_) => {
                return Err(ProveExistsError::PrimitivesUnsupported);
            }
        };

        let function = self.egraph.functions.get(&func.name).ok_or_else(|| {
            ProveExistsError::ConstructorNotDeclared {
                constructor: func.name.clone(),
            }
        })?;

        let backend_id = function.backend_id;
        let output_sort = function.schema.output.clone();

        // Use the version that ignores unextractable flag since proof extraction
        // needs to extract proofs from all terms including those marked unextractable
        let extractor = Extractor::compute_costs_from_rootsorts_allow_unextractable(
            None,
            self.egraph,
            TreeAdditiveCostModel::default(),
        );

        let mut termdag = TermDag::default();
        let mut witness_value = None;

        self.egraph.backend.for_each_while(backend_id, |row| {
            let value = *row
                .vals
                .last()
                .expect("constructor rows include their output value");
            witness_value = Some(value);
            false
        });

        let witness_value = witness_value.ok_or_else(|| ProveExistsError::QueryDidNotMatch {
            constructor: func.name.clone(),
        })?;

        let proof_function_name = self
            .egraph
            .proof_state
            .proof_func_parent
            .get(output_sort.name())
            .ok_or(ProveExistsError::ProofsNotEnabled)?
            .clone();
        let proof_value = self
            .egraph
            .lookup_function(&proof_function_name, &[witness_value])
            .ok_or_else(|| ProveExistsError::NoProofRecorded {
                constructor: func.name.clone(),
            })?;

        let proof_sort = self
            .egraph
            .functions
            .get(&proof_function_name)
            .ok_or_else(|| ProveExistsError::ConstructorNotDeclared {
                constructor: func.name.clone(),
            })?
            .schema
            .output
            .clone();

        let (_, proof_term_id) = extractor
            .extract_best_with_sort(self.egraph, &mut termdag, proof_value, proof_sort)
            .ok_or_else(|| ProveExistsError::ProofTermExtractionFailed {
                constructor: func.name.clone(),
            })?;

        let (mut proof_store, proof_id) = proof_store_from_term(
            &self.egraph.proof_state.proof_names,
            termdag,
            proof_term_id,
            &self.egraph.proof_check_program,
        );

        // Remove globals from the proof
        proof_store
            .remove_globals(&self.egraph.proof_check_program)
            .map_err(ProveExistsError::ProofCheck)?;

        // if the existence proof has a single premise, extract that premise proof
        let proof = proof_store.get(proof_id);
        let extra_rule_removed = match proof.justification() {
            Justification::Rule { premise_proofs, .. } => match premise_proofs.as_slice() {
                [premise_proof_id] => *premise_proof_id,
                _ => proof_id,
            },
            _ => {
                return Err(ProveExistsError::NotProvableByRule {
                    constructor: func.name.clone(),
                });
            }
        };

        // Check the proof before simplification
        proof_store
            .check_proof(extra_rule_removed, &self.egraph.proof_check_program)
            .map_err(ProveExistsError::ProofCheck)?;

        // simplify the proof
        let simplified_proof = proof_store.simplify(extra_rule_removed);

        // Check the proof after simplification
        proof_store
            .check_proof(simplified_proof, &self.egraph.proof_check_program)
            .map_err(ProveExistsError::ProofCheck)?;

        Ok((proof_store, simplified_proof))
    }
}
