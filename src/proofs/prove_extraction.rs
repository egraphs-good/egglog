use crate::extract::{DefaultCost, Extractor, TreeAdditiveCostModel};
use crate::proofs::proof_format::{ProofId, ProofStore, proof_store_from_term};
use crate::util::HashMap;
use crate::{ArcSort, EGraph, Term, TermDag, TermId, Value};
use egglog_ast::generic_ast::Literal;
use thiserror::Error;

/// Pre-computed lookup tables for view tables and their associated proof tables.
///
/// For each view table in the e-graph, this struct builds:
/// - A map from the input columns to the output e-class (the second-to-last value in each row).
/// - If a corresponding proof table exists, a map from (inputs, eclass) to the proof value.
///
/// ## View table layout
///
/// The view table for a constructor like `Add(i64, i64) -> Math` is
/// `AddView(i64, i64, Math) -> view_sort`, where the `Math` column stores the
/// e-class representative. In the backend row (`vals`), this is at position
/// `vals.len() - 2` (the second-to-last column), while the final column is the
/// view-sort output value.
///
/// ## Proof table layout
///
/// The proof table `AddViewProof(i64, i64, Math) -> Proof` maps the same key
/// (inputs + eclass) to a proof value that proves the representative equals
/// the concrete term. In the backend row, `vals[..vals.len()-1]` is the key
/// and `vals.last()` is the proof value.
///
/// See `proof_encoding.md` for full details on the term and proof encoding.
pub(crate) struct ViewProofIndex {
    /// For each original function/constructor name, maps the input columns of
    /// its view table to the e-class representative (second-to-last value in
    /// the row).
    ///
    /// Key: original function/constructor name (e.g. `"Add"`).
    /// Value: `HashMap` from input values (`Vec<Value>`) to the e-class `Value`.
    pub(crate) view_eclass: HashMap<String, HashMap<Vec<Value>, Value>>,

    /// For each original function/constructor name that has a proof table, maps
    /// the full proof-table key (inputs ++ \[eclass\]) to the proof `Value`.
    ///
    /// Key: original function/constructor name (e.g. `"Add"`).
    /// Value: `HashMap` from `(inputs, eclass)` concatenated as `Vec<Value>` to
    /// the proof `Value`.
    pub(crate) view_proofs: HashMap<String, HashMap<Vec<Value>, Value>>,
}

impl ViewProofIndex {
    /// Build a [`ViewProofIndex`] by scanning every view table (and its
    /// corresponding proof table, if one exists) that was registered during
    /// proof encoding.
    ///
    /// # Layout assumptions
    ///
    /// * A view-table backend row has the shape
    ///   `[input_0, …, input_n, eclass, view_sort_output]`.
    ///   The e-class representative lives at `vals[vals.len() - 2]`.
    ///
    /// * A proof-table backend row has the shape
    ///   `[input_0, …, input_n, eclass, proof_value]`.
    ///   The key is `vals[..vals.len() - 1]` and the proof is `vals.last()`.
    pub(crate) fn new(egraph: &EGraph) -> Self {
        let proof_names = &egraph.proof_state.proof_names;

        let mut view_eclass: HashMap<String, HashMap<Vec<Value>, Value>> = HashMap::default();
        let mut view_proofs: HashMap<String, HashMap<Vec<Value>, Value>> = HashMap::default();

        // Iterate over every view table that was registered during proof encoding.
        for (original_name, view_table_name) in proof_names.view_name.iter() {
            // Look up the view table's `Function` in the e-graph.
            let view_function = match egraph.get_function(view_table_name) {
                Some(f) => f,
                None => continue,
            };
            let view_backend_id = view_function.backend_id;

            // ----------------------------------------------------------
            // Build the inputs → eclass map for this view table.
            // ----------------------------------------------------------
            let mut eclass_map: HashMap<Vec<Value>, Value> = HashMap::default();
            egraph.backend.for_each(view_backend_id, |row| {
                let vals = row.vals;
                if vals.len() >= 2 {
                    let inputs = vals[..vals.len() - 2].to_vec();
                    let eclass = vals[vals.len() - 2];
                    eclass_map.insert(inputs, eclass);
                }
            });
            view_eclass.insert(original_name.clone(), eclass_map);

            // ----------------------------------------------------------
            // Look for the corresponding proof table and build its map.
            // ----------------------------------------------------------
            let proof_table_name = match proof_names.view_proof_name.get(original_name) {
                Some(name) => name,
                None => continue,
            };
            let proof_function = match egraph.get_function(proof_table_name) {
                Some(f) => f,
                None => continue,
            };
            let proof_backend_id = proof_function.backend_id;

            let mut proof_map: HashMap<Vec<Value>, Value> = HashMap::default();
            egraph.backend.for_each(proof_backend_id, |row| {
                let vals = row.vals;
                if !vals.is_empty() {
                    let key = vals[..vals.len() - 1].to_vec();
                    let proof_value = vals[vals.len() - 1];
                    proof_map.insert(key, proof_value);
                }
            });
            view_proofs.insert(original_name.clone(), proof_map);
        }

        Self {
            view_eclass,
            view_proofs,
        }
    }
}

#[derive(Debug, Error)]
pub enum ProveEqualToRepresentativeError {
    #[error("Term not found in view table for constructor `{constructor}` with inputs {inputs:?}")]
    TermNotInViewTable {
        constructor: String,
        inputs: Vec<Value>,
    },
    #[error("No proof found in view proof table for constructor `{constructor}` with key {key:?}")]
    ProofNotFound {
        constructor: String,
        key: Vec<Value>,
    },
    #[error("Failed to extract proof term for constructor `{constructor}`")]
    ExtractionFailed { constructor: String },
    #[error("Constructor `{constructor}` has no view table registered")]
    NoViewTable { constructor: String },
    #[error("Unsupported term kind: expected App (constructor) or Lit, got Var(`{name}`)")]
    UnsupportedVar { name: String },
}

#[derive(Debug, Error)]
pub enum ExtractWithProofError {
    #[error("No UF table registered for sort `{sort_name}`")]
    NoUfTable { sort_name: String },
    #[error("No UF proof table registered for sort `{sort_name}`")]
    NoUfProofTable { sort_name: String },
    #[error("Extraction failed for value {value:?} with sort `{sort_name}`")]
    ExtractionFailed { value: Value, sort_name: String },
    #[error(
        "UF proof not found for value {value:?} -> representative {representative:?} in sort `{sort_name}`"
    )]
    UfProofNotFound {
        value: Value,
        representative: Value,
        sort_name: String,
    },
    #[error("Failed to extract UF proof term for value {value:?} in sort `{sort_name}`")]
    UfProofExtractionFailed { value: Value, sort_name: String },
    #[error(transparent)]
    ProveError(#[from] ProveEqualToRepresentativeError),
}

/// Constructs low-level proof terms proving that a given [`Term`] is equal to
/// the representative of its e-class.
///
/// This struct holds pre-computed indexes over the e-graph's view tables and
/// proof tables, plus an [`Extractor`] used to turn proof [`Value`]s into
/// [`Term`]s in the [`TermDag`].
///
/// Use [`ProveEqualToRepresentative::new`] to build one from an [`EGraph`],
/// then call [`ProveEqualToRepresentative::prove`] for each term you want a
/// proof for.
pub(crate) struct ProveEqualToRepresentative<'a> {
    egraph: &'a EGraph,
    index: ViewProofIndex,
    extractor: Extractor<DefaultCost>,
    proof_sort: ArcSort,
}

impl<'a> ProveEqualToRepresentative<'a> {
    /// Create a new proof builder for the given e-graph.
    ///
    /// This scans all view/proof tables and pre-computes the extractor costs.
    pub(crate) fn new(egraph: &'a EGraph) -> Self {
        let index = ViewProofIndex::new(egraph);
        let extractor = Extractor::compute_costs_from_rootsorts_allow_unextractable(
            None,
            egraph,
            TreeAdditiveCostModel::default(),
        );

        // Resolve the proof sort by name from the encoding state.
        let proof_sort_name = &egraph.proof_state.proof_names.proof_datatype;
        let proof_sort = egraph
            .type_info
            .get_sort_by_name(proof_sort_name)
            .unwrap_or_else(|| {
                panic!(
                    "proof sort `{proof_sort_name}` not found; \
                     is term-encoding with proofs enabled?"
                )
            })
            .clone();

        Self {
            egraph,
            index,
            extractor,
            proof_sort,
        }
    }

    /// Prove that `term_id` equals the representative of its e-class.
    ///
    /// Returns `(proof_term_id, leader_value)` where:
    /// * `proof_term_id` is a low-level proof [`TermId`] in `termdag` whose
    ///   proposition is `term = representative`.
    /// * `leader_value` is the [`Value`] of the e-class leader.
    ///
    /// The proof is constructed bottom-up:
    /// 1. For each child of the term, recursively obtain its leader and proof.
    /// 2. Look up the view table with the child leaders as inputs to find the
    ///    e-class leader for this term.
    /// 3. Extract the view proof (which proves `leader = f(child_leaders…)`).
    /// 4. Use [`Congr`] steps to replace each child leader with the original
    ///    child term (chaining the child proofs).
    /// 5. Apply [`Sym`] to flip the proof to `f(originals…) = leader`.
    pub(crate) fn prove(
        &self,
        termdag: &mut TermDag,
        term_id: TermId,
    ) -> Result<(TermId, Value), ProveEqualToRepresentativeError> {
        let term = termdag.get(term_id).clone();
        match term {
            // ----- Literal base values -----------------------------------------------
            // Literals are their own representatives. The proof is a Fiat
            // self-equality: Fiat(AstPrim(lit), AstPrim(lit)).
            // The Value is the interned base value.
            Term::Lit(ref lit) => {
                let value = crate::literal_to_value(&self.egraph.backend, lit);
                // Build a fiat proof: lit = lit
                // We need the AST wrapper for the literal. Literals don't have
                // an eq-sort, so there's no sort_to_ast_constructor for them.
                // We just produce a Fiat(term, term) proof term in the TermDag.
                let names = &self.egraph.proof_state.proof_names;
                let fiat = &names.fiat_constructor;
                // Wrap the literal in the AST sort. Find an appropriate ast
                // constructor – for primitive sorts we look up by sort name.
                // Primitives don't necessarily have ast constructors registered.
                // For now, just emit Fiat with the raw literal wrapped.
                // We'll wrap both sides in the same ast constructor.
                let ast_term = self.wrap_in_ast(termdag, term_id);
                let proof_term = termdag.app(fiat.clone(), vec![ast_term, ast_term]);
                Ok((proof_term, value))
            }

            // ----- Constructor / function applications --------------------------------
            Term::App(ref head, ref children) => {
                let children = children.clone();
                let head = head.clone();

                // 1. Recurse on each child to get (child_proof, child_leader).
                let mut child_leaders: Vec<Value> = Vec::with_capacity(children.len());
                let mut child_proofs: Vec<(TermId, TermId)> = Vec::with_capacity(children.len()); // (proof, original_child_id)

                for &child_id in &children {
                    let (child_proof, child_leader) = self.prove(termdag, child_id)?;
                    child_leaders.push(child_leader);
                    child_proofs.push((child_proof, child_id));
                }

                // 2. Look up the view table for `head` with the child leaders.
                let eclass_map = self.index.view_eclass.get(&head).ok_or_else(|| {
                    ProveEqualToRepresentativeError::NoViewTable {
                        constructor: head.clone(),
                    }
                })?;

                let eclass_leader = *eclass_map.get(&child_leaders).ok_or_else(|| {
                    ProveEqualToRepresentativeError::TermNotInViewTable {
                        constructor: head.clone(),
                        inputs: child_leaders.clone(),
                    }
                })?;

                // 3. Look up the view proof: key = (inputs…, eclass)
                let proof_map = self.index.view_proofs.get(&head).ok_or_else(|| {
                    ProveEqualToRepresentativeError::NoViewTable {
                        constructor: head.clone(),
                    }
                })?;

                let mut proof_key = child_leaders.clone();
                proof_key.push(eclass_leader);

                let proof_value = *proof_map.get(&proof_key).ok_or_else(|| {
                    ProveEqualToRepresentativeError::ProofNotFound {
                        constructor: head.clone(),
                        key: proof_key.clone(),
                    }
                })?;

                // 4. Extract the proof Value into a proof TermId.
                let view_proof_term = self
                    .extractor
                    .extract_best_with_sort(
                        self.egraph,
                        termdag,
                        proof_value,
                        self.proof_sort.clone(),
                    )
                    .map(|(_, tid)| tid)
                    .ok_or_else(|| ProveEqualToRepresentativeError::ExtractionFailed {
                        constructor: head.clone(),
                    })?;

                // `view_proof_term` proves: eclass_leader = head(child_leader_0, …)
                //
                // 5. For each child where the original child differs from the
                //    leader, apply Congr to replace the leader with the original.
                //
                //    child_proof proves: original_child = child_leader
                //    Sym(child_proof) proves: child_leader = original_child
                //    Congr(current, i, Sym(child_proof)) proves:
                //        eclass_leader = head(…, original_child, …)
                let names = &self.egraph.proof_state.proof_names;
                let sym_name = &names.eq_sym_constructor;
                let congr_name = &names.congr_constructor;

                let mut current_proof = view_proof_term;
                for (i, (child_proof, child_term_id)) in child_proofs.iter().enumerate() {
                    // Check if the child is already its own leader (Lit children
                    // are always their own representative, but App children may
                    // need congruence).
                    // We always apply Congr for App children since we can't cheaply
                    // compare TermIds to Values here. The proof checker will handle
                    // the trivial case.
                    let child_term = termdag.get(*child_term_id).clone();
                    if matches!(child_term, Term::App(..)) {
                        // Sym(child_proof): child_leader = original_child
                        let sym_child = termdag.app(sym_name.clone(), vec![*child_proof]);
                        // Congr(current_proof, i, sym_child)
                        let index_lit = termdag.lit(Literal::Int(i as i64));
                        current_proof = termdag.app(
                            congr_name.clone(),
                            vec![current_proof, index_lit, sym_child],
                        );
                    }
                }

                // current_proof now proves: eclass_leader = head(originals…)
                // We want: head(originals…) = eclass_leader
                // So apply Sym.
                let final_proof = termdag.app(sym_name.clone(), vec![current_proof]);

                Ok((final_proof, eclass_leader))
            }

            Term::Var(ref name) => {
                Err(ProveEqualToRepresentativeError::UnsupportedVar { name: name.clone() })
            }
        }
    }

    /// Wrap a term in the appropriate `Ast<Sort>` constructor for proof terms.
    ///
    /// For eq-sorts (constructors), we look up the registered
    /// `sort_to_ast_constructor`. For base/primitive sorts (literals), we need
    /// to find an ast constructor too – the encoding registers one per sort
    /// that appears in the program.
    fn wrap_in_ast(&self, termdag: &mut TermDag, inner: TermId) -> TermId {
        // Try to find an ast constructor for the term. For an App, look up
        // via fn_to_term_sort → sort_to_ast_constructor. For a Lit we need the
        // sort name, but we don't have it directly. We'll look through all
        // registered ast constructors and pick the first one that matches the
        // sort. For simplicity in the literal case, we try all constructors.
        let names = &self.egraph.proof_state.proof_names;
        let term = termdag.get(inner).clone();
        match term {
            Term::App(ref head, _) => {
                // Get the sort for this function, then the ast constructor.
                if let Some(sort_name) = names.fn_to_term_sort.get(head) {
                    if let Some(ast_ctor) = names.sort_to_ast_constructor.get(sort_name) {
                        return termdag.app(ast_ctor.clone(), vec![inner]);
                    }
                }
                // Fallback: return unwrapped (shouldn't normally happen).
                inner
            }
            _ => {
                // For literals, we don't have a sort-specific ast wrapper in
                // the general case. This path is only hit for Fiat proofs of
                // base values which are self-equalities; callers may not need
                // the ast wrapper for these.
                inner
            }
        }
    }

    /// Extract the best term for `value` and produce a proof that the
    /// extracted term equals `value`.
    ///
    /// # Algorithm
    ///
    /// 1. **Find the representative.** Look up the UF table for the given
    ///    sort to find the canonical representative of `value`'s e-class.
    ///    If `value` has no UF entry it is already the representative.
    ///
    /// 2. **Extract.** Use the stored [`Extractor`] to extract the best term
    ///    for the representative into a [`TermDag`].
    ///
    /// 3. **Prove extracted = representative.** Call
    ///    [`prove`](Self::prove) on the extracted term to obtain a low-level
    ///    proof term `extracted_term = representative`.
    ///
    /// 4. **Get UF proof (value = representative).** If `value` differs from
    ///    the representative, look up the UF proof table to get a proof that
    ///    `value = representative`.
    ///
    /// 5. **Chain.** Combine the two proofs with `Trans` / `Sym`:
    ///    - `extracted_term = representative` (from step 3)
    ///    - `representative = value` (= `Sym(value = representative)` from step 4)
    ///    - `Trans(step3, Sym(step4))` proves `extracted_term = value`.
    ///
    ///    When `value` is already the representative, the UF proof is skipped
    ///    and the result of step 3 is the final proof.
    ///
    /// 6. **Convert.** Turn the low-level proof [`TermId`] into a
    ///    [`ProofStore`] / [`ProofId`] via [`proof_store_from_term`].
    ///
    /// # Returns
    ///
    /// `(proof_store, proof_id)` – the proof is a ground equality whose
    /// proposition contains the extracted term (LHS) and the input value (RHS).
    pub(crate) fn extract_with_proof(
        &self,
        value: Value,
        sort: ArcSort,
    ) -> Result<(ProofStore, ProofId), ExtractWithProofError> {
        let sort_name = sort.name().to_string();
        let proof_names = &self.egraph.proof_state.proof_names;

        // ── 1. Find the representative via the UF table ─────────────────
        let representative = if sort.is_eq_sort() {
            let uf_table_name = self
                .egraph
                .proof_state
                .uf_parent
                .get(&sort_name)
                .ok_or_else(|| ExtractWithProofError::NoUfTable {
                    sort_name: sort_name.clone(),
                })?;
            // The UF table is a constructor UF_Sort(child, parent) -> uf,
            // so we scan for the row where vals[0] == value and read vals[1].
            // If no entry exists the value IS the representative.
            let uf_func = self.egraph.get_function(uf_table_name).ok_or_else(|| {
                ExtractWithProofError::NoUfTable {
                    sort_name: sort_name.clone(),
                }
            })?;
            let mut canonical = value;
            self.egraph
                .backend
                .for_each(uf_func.backend_id, |row: egglog_bridge::FunctionRow| {
                    if row.vals[0] == value {
                        canonical = row.vals[1];
                    }
                });
            canonical
        } else {
            // Primitive sorts have no UF; the value is its own representative.
            value
        };

        // ── 2. Extract the best term for the representative ─────────────
        let mut termdag = TermDag::default();
        let (_, extracted_term_id) = self
            .extractor
            .extract_best_with_sort(self.egraph, &mut termdag, representative, sort.clone())
            .ok_or_else(|| ExtractWithProofError::ExtractionFailed {
                value: representative,
                sort_name: sort_name.clone(),
            })?;

        // ── 3. Prove extracted_term = representative ────────────────────
        let (prove_proof, _leader) = self.prove(&mut termdag, extracted_term_id)?;
        // `prove_proof` proves: extracted_term = representative

        // ── 4 & 5. Get UF proof and chain ───────────────────────────────
        let final_proof = if value == representative {
            // Value is already the representative; no UF step needed.
            prove_proof
        } else {
            // Look up the UF proof table for this sort.
            let uf_proof_fn_name = proof_names.uf_proof_name.get(&sort_name).ok_or_else(|| {
                ExtractWithProofError::NoUfProofTable {
                    sort_name: sort_name.clone(),
                }
            })?;

            // MathUFProof(value, representative) -> proof of `value = representative`
            let uf_proof_value = self
                .egraph
                .lookup_function(uf_proof_fn_name, &[value, representative])
                .ok_or_else(|| ExtractWithProofError::UfProofNotFound {
                    value,
                    representative,
                    sort_name: sort_name.clone(),
                })?;

            // Extract the UF proof value into a proof TermId.
            let (_, uf_proof_term) = self
                .extractor
                .extract_best_with_sort(
                    self.egraph,
                    &mut termdag,
                    uf_proof_value,
                    self.proof_sort.clone(),
                )
                .ok_or_else(|| ExtractWithProofError::UfProofExtractionFailed {
                    value,
                    sort_name: sort_name.clone(),
                })?;
            // `uf_proof_term` proves: value = representative

            // Sym(uf_proof_term) proves: representative = value
            let sym_name = &proof_names.eq_sym_constructor;
            let sym_uf = termdag.app(sym_name.clone(), vec![uf_proof_term]);

            // Trans(prove_proof, sym_uf):
            //   prove_proof: extracted_term = representative
            //   sym_uf:      representative = value
            //   result:      extracted_term = value
            let trans_name = &proof_names.eq_trans_constructor;
            termdag.app(trans_name.clone(), vec![prove_proof, sym_uf])
        };

        // ── 6. Convert to high-level ProofStore ─────────────────────────
        let (proof_store, proof_id) = proof_store_from_term(
            proof_names,
            termdag.clone(),
            final_proof,
            &self.egraph.proof_check_program,
        );

        Ok((proof_store, proof_id))
    }
}
