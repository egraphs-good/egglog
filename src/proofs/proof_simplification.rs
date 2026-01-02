use crate::{
    Term, TermId,
    ast::ResolvedNCommand,
    proofs::{
        proof_checker::gather_globals,
        proof_format::{Justification, Proof, ProofId, ProofStore},
    },
    util::HashMap,
};

impl ProofStore {
    /// Remove globals from a proof by replacing all global variable references
    /// with their computed values.
    /// This constructs a map of global names to their terms (without globals),
    /// then replaces all occurrences of those globals in the proof's term dag.
    pub fn remove_globals(&mut self, prog: &[ResolvedNCommand]) {
        // Gather all globals and their values as terms
        let globals = gather_globals(prog, &mut self.term_dag);

        // Replace all global function calls (nullary functions) in the term dag
        // with their computed values
        self.replace_global_terms(&globals);
    }

    /// Replace all global function applications in the term dag with their values.
    fn replace_global_terms(&mut self, globals: &HashMap<String, TermId>) {
        // We need to rebuild the term dag, replacing global calls as we go
        // Build a map from old term IDs to new term IDs
        let mut term_mapping: HashMap<TermId, TermId> = HashMap::default();

        // Process all terms in order (since term dag is built bottom-up)
        for term_id in 0..self.term_dag.size() {
            let term = self.term_dag.get(term_id).clone();
            let new_term_id = match term {
                Term::Lit(_) | Term::Var(_) => term_id, // Literals and vars don't change
                Term::App(ref head, ref args) => {
                    // Check if this is a nullary global function call
                    if args.is_empty() && globals.contains_key(head) {
                        // Replace with the global's value, applying the term mapping
                        let global_term_id = globals[head];
                        *term_mapping.get(&global_term_id).unwrap_or(&global_term_id)
                    } else {
                        // Map the children and reconstruct the term if any changed
                        let mapped_args: Vec<TermId> = args
                            .iter()
                            .map(|&child_id| *term_mapping.get(&child_id).unwrap_or(&child_id))
                            .collect();

                        // Check if any children changed
                        let changed = mapped_args
                            .iter()
                            .zip(args.iter())
                            .any(|(new, old)| new != old);

                        if changed {
                            // Reconstruct the term with new children
                            let new_children: Vec<Term> = mapped_args
                                .iter()
                                .map(|&id| self.term_dag.get(id).clone())
                                .collect();
                            let new_term = self.term_dag.app(head.clone(), new_children);
                            self.term_dag.lookup(&new_term)
                        } else {
                            term_id
                        }
                    }
                }
            };
            term_mapping.insert(term_id, new_term_id);
        }

        // Now update all proofs to use the new term IDs
        for proof in &mut self.id_to_proof {
            proof.lhs = *term_mapping.get(&proof.lhs).unwrap_or(&proof.lhs);
            proof.rhs = *term_mapping.get(&proof.rhs).unwrap_or(&proof.rhs);
        }
    }

    /// A simple simplification pass removing unnecessary steps.
    /// For example, congruence steps that do not change the term.
    pub fn simplify(&mut self, proof_id: ProofId) -> ProofId {
        let proof = self.get(proof_id).clone();
        match proof {
            Proof {
                lhs,
                rhs,
                justification:
                    Justification::Congr {
                        child_proof,
                        proof: base_proof,
                        ..
                    },
            } => {
                // if the child proof proves t = t for some t, we can skip the congruence step
                let child_proof = self.get(child_proof);
                if child_proof.lhs == child_proof.rhs {
                    self.simplify(base_proof)
                } else {
                    self.map_child_proofs(proof_id, |store, pid| store.simplify(pid))
                }
            }
            Proof {
                lhs,
                rhs,
                justification: Justification::Trans(p1, p2),
            } => {
                // if either side is a reflexive proof, skip it
                let p1_proof = self.get(p1);
                let p2_proof = self.get(p2);
                if p1_proof.lhs == p1_proof.rhs {
                    return self.simplify(p2);
                } else if p2_proof.lhs == p2_proof.rhs {
                    return self.simplify(p1);
                } else {
                    self.map_child_proofs(proof_id, |store, pid| store.simplify(pid))
                }
            }
            Proof {
                lhs,
                rhs,
                justification: Justification::Sym(inner),
            } => {
                let mut current_id = proof_id;
                // Simplify the inner proof first
                let simplified_inner = self.simplify(inner);

                // Check if the inner proof is also a Sym - if so, we have Sym(Sym(p)) which equals p
                let inner_proof = self.get(simplified_inner);

                if let Justification::Sym(inner_inner) = inner_proof.justification() {
                    current_id = *inner_inner;
                };

                if let Justification::Sym(inner) = self.get(current_id).justification() {
                    if let Proof {
                        lhs,
                        rhs,
                        justification,
                    } = self.get(*inner)
                    {
                        // swap lhs and rhs
                        let simplified = Proof {
                            lhs: *rhs,
                            rhs: *lhs,
                            justification: justification.clone(),
                        };

                        // replace current id with simplified proof
                        self.id_to_proof[current_id] = simplified;
                    }
                }

                current_id
            }

            _ => self.map_child_proofs(proof_id, |store, pid| store.simplify(pid)),
        }
    }

    /// Map over the child proofs of this proof, producing a new proof with the same justification but updated child proofs.
    pub fn map_child_proofs<F>(&mut self, proof_id: ProofId, mut f: F) -> ProofId
    where
        F: FnMut(&mut ProofStore, ProofId) -> ProofId,
    {
        let mut proof = self.id_to_proof[proof_id].clone();

        let mut changed = false;

        match &mut proof.justification {
            Justification::Fiat => return proof_id,
            Justification::Rule { premise_proofs, .. } => {
                for pid in premise_proofs.iter_mut() {
                    let mapped = f(self, *pid);
                    if mapped != *pid {
                        *pid = mapped;
                        changed = true;
                    }
                }
            }
            Justification::MergeFn {
                old_proof,
                new_proof,
                ..
            } => {
                let mapped_old = f(self, *old_proof);
                let mapped_new = f(self, *new_proof);
                if mapped_old != *old_proof || mapped_new != *new_proof {
                    *old_proof = mapped_old;
                    *new_proof = mapped_new;
                    let old = self.get(*old_proof);
                    let new = self.get(*new_proof);
                    proof.lhs = old.lhs;
                    proof.rhs = new.rhs;
                    changed = true;
                }
            }
            Justification::Trans(left, right) => {
                let mapped_left = f(self, *left);
                let mapped_right = f(self, *right);
                if mapped_left != *left || mapped_right != *right {
                    *left = mapped_left;
                    *right = mapped_right;
                    let left_proof = self.get(*left);
                    let right_proof = self.get(*right);
                    debug_assert_eq!(
                        left_proof.rhs, right_proof.lhs,
                        "transitivity requires matching middle terms"
                    );
                    proof.lhs = left_proof.lhs;
                    proof.rhs = right_proof.rhs;
                    changed = true;
                }
            }
            Justification::Sym(inner) => {
                let mapped_inner = f(self, *inner);
                if mapped_inner != *inner {
                    *inner = mapped_inner;
                    let inner_proof = self.get(*inner);
                    proof.lhs = inner_proof.rhs;
                    proof.rhs = inner_proof.lhs;
                    changed = true;
                }
            }
            Justification::Congr {
                proof: base,
                child_index,
                child_proof,
            } => {
                let mapped_base = f(self, *base);
                let mapped_child = f(self, *child_proof);
                if mapped_base != *base || mapped_child != *child_proof {
                    *base = mapped_base;
                    *child_proof = mapped_child;
                    let base_proof = self.get(*base);
                    let child = self.get(*child_proof);
                    proof.lhs = base_proof.lhs;
                    proof.rhs = self.replace_term_child(base_proof.rhs, *child_index, child.rhs);
                    changed = true;
                }
            }
        }

        if !changed {
            return proof_id;
        }

        self.id_to_proof[proof_id] = proof;
        proof_id
    }
}
