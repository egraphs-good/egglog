use crate::ast::FunctionSubtype;
use crate::extract::{Extractor, TreeAdditiveCostModel};
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
    #[error("constructor {name} is marked :unextractable")]
    ConstructorUnextractable { name: String },
    #[error("prove requires proofs mode")]
    ProofsDisabled,
    #[error("Could not find a proof due to query not matching (constructor {constructor}).")]
    QueryDidNotMatch { constructor: String },
}

impl<'a> ProofInstrumentor<'a> {
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

        let function = self
            .egraph
            .functions
            .get(&func.name)
            .unwrap_or_else(|| panic!("constructor {} is not declared", func.name));

        if function.decl.unextractable {
            return Err(ProveExistsError::ConstructorUnextractable {
                name: func.name.clone(),
            });
        }

        if !self.egraph.proof_state.proofs_enabled {
            return Err(ProveExistsError::ProofsDisabled);
        }

        let backend_id = function.backend_id;
        let output_sort = function.schema.output.clone();

        let extractor = Extractor::compute_costs_from_rootsorts(
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

        let proof_function_name = self.term_proof_name(output_sort.name());
        let proof_value = self
            .egraph
            .lookup_function(&proof_function_name, &[witness_value])
            .unwrap_or_else(|| panic!("no proof recorded for constructor {}", func.name));

        let proof_sort = self
            .egraph
            .functions
            .get(&proof_function_name)
            .unwrap_or_else(|| {
                panic!(
                    "proof table {proof_function_name} for constructor {} was not declared",
                    func.name
                )
            })
            .schema
            .output
            .clone();

        let (_, proof_term_id) = extractor
            .extract_best_with_sort(self.egraph, &mut termdag, proof_value, proof_sort)
            .unwrap_or_else(|| {
                panic!("failed to extract proof term for constructor {}", func.name)
            });

        let (mut proof_store, proof_id) = proof_store_from_term(
            &self.egraph.proof_state.proof_names,
            termdag,
            proof_term_id,
            &self.egraph.desugared_commands,
        );

        // Remove globals from the proof
        proof_store.remove_globals(&self.egraph.desugared_commands);

        // if the existence proof has a single premise, extract that premise proof
        let proof = proof_store.get(proof_id);
        let extra_rule_removed = match proof.justification() {
            Justification::Rule { premise_proofs, .. } => match premise_proofs.as_slice() {
                [premise_proof_id] => *premise_proof_id,
                _ => proof_id,
            },
            _ => panic!("expected rule justification for existence proof"),
        };

        // Check the proof before simplification
        if let Result::Err(e) =
            proof_store.check_proof(extra_rule_removed, &self.egraph.desugared_commands)
        {
            panic!("Existence proof should be valid before simplification: {e}");
        }

        // simplify the proof
        let simplified_proof = proof_store.simplify(extra_rule_removed);

        // Check the proof after simplification
        proof_store
            .check_proof(simplified_proof, &self.egraph.desugared_commands)
            .expect("simplified existence proof should still be valid");

        Ok((proof_store, extra_rule_removed))
    }
}
