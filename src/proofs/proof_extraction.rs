use crate::ast::FunctionSubtype;
use crate::extract::{Extractor, TreeAdditiveCostModel};
use crate::proofs::proof_encoding::ProofInstrumentor;
use crate::proofs::proof_format::{ProofId, ProofStore, proof_store_from_term};
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

        let egraph = &mut *self.egraph;

        let function = egraph
            .functions
            .get(&func.name)
            .unwrap_or_else(|| panic!("constructor {} is not declared", func.name));

        if function.decl.unextractable {
            return Err(ProveExistsError::ConstructorUnextractable {
                name: func.name.clone(),
            });
        }

        if !egraph.proof_state.proofs_enabled {
            return Err(ProveExistsError::ProofsDisabled);
        }

        let backend_id = function.backend_id;
        let output_sort = function.schema.output.clone();

        let extractor =
            Extractor::compute_costs_from_rootsorts(None, egraph, TreeAdditiveCostModel::default());

        let mut termdag = TermDag::default();
        let mut witness_value = None;

        egraph.backend.for_each_while(backend_id, |row| {
            if row.subsumed {
                return true;
            }

            let value = *row
                .vals
                .last()
                .expect("constructor rows include their output value");

            if let Some(_) =
                extractor.extract_best_with_sort(egraph, &mut termdag, value, output_sort.clone())
            {
                witness_value = Some(value);
                return false;
            }

            true
        });

        let witness_value = witness_value.ok_or_else(|| ProveExistsError::QueryDidNotMatch {
            constructor: func.name.clone(),
        })?;

        let proof_function_name = format!("{}Proof", output_sort.name());
        let proof_value = egraph
            .lookup_function(&proof_function_name, &[witness_value])
            .unwrap_or_else(|| panic!("no proof recorded for constructor {}", func.name));

        let proof_sort = egraph
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

        let (_, proof_term) = extractor
            .extract_best_with_sort(egraph, &mut termdag, proof_value, proof_sort)
            .unwrap_or_else(|| {
                panic!("failed to extract proof term for constructor {}", func.name)
            });

        let proof_term_id = termdag.lookup(&proof_term);
        let (proof_store, proof_id) = proof_store_from_term(
            &egraph.proof_state.proof_names,
            termdag,
            proof_term_id,
            &egraph.desugared_commands,
        );

        Ok((proof_store, proof_id))
    }
}
