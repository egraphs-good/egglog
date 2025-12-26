use crate::{
    Term, TermId,
    ast::ResolvedNCommand,
    proofs::{
        proof_checker::gather_globals,
        proof_format::{Justification, Proof, ProofId, ProofStore, Proposition},
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
                        self.term_dag.app(head.clone(), mapped_args)
                    }
                }
            };
            term_mapping.insert(term_id, new_term_id);
        }

        // Now update all proofs to use the new term IDs
        for proof in &mut self.id_to_proof {
            proof.map_terms_mut(|term_id| *term_mapping.get(&term_id).unwrap_or(&term_id));
        }
    }

    /// Add a new proof to the store and return its ID.
    fn add_proof(&mut self, proof: Proof) -> ProofId {
        let proof_id = self.id_to_proof.len();
        self.id_to_proof.push(proof);
        proof_id
    }

    /// A simple simplification pass removing unnecessary steps.
    /// Applies optimizations until a fixed point is reached.
    ///
    /// Simplifications performed:
    /// - Remove reflexive congruence: Congr(p, refl) -> p
    /// - Remove reflexive transitivity: Trans(refl, p) -> p and Trans(p, refl) -> p
    /// - Remove reflexive symmetry: Sym(p) -> p when p proves t = t
    /// - Collapse double symmetry: Sym(Sym(p)) -> p
    /// - Push symmetry through transitivity: Sym(Trans(p1, p2)) -> Trans(Sym(p2), Sym(p1))
    ///   This enables further simplifications by exposing the inner proofs
    pub fn simplify(&mut self, proof_id: ProofId) -> ProofId {
        // First, recursively simplify all child proofs
        let proof_id = self.map_child_proofs(proof_id, |store, pid| store.simplify(pid));

        // Apply local optimizations until fixed point
        let mut current_id = proof_id;
        loop {
            let new_id = self.apply_local_optimizations(current_id);
            if new_id == current_id {
                break;
            }
            current_id = new_id;
        }
        current_id
    }

    /// Apply local optimizations to a single proof node.
    /// Returns a potentially different proof ID if an optimization was applied.
    fn apply_local_optimizations(&mut self, proof_id: ProofId) -> ProofId {
        // List of optimization functions to try
        let optimizations: &[fn(&mut ProofStore, ProofId) -> Option<ProofId>] = &[
            Self::opt_reflexive_congr,
            Self::opt_reflexive_trans,
            Self::opt_reflexive_sym,
            Self::opt_double_sym,
            Self::opt_sym_trans,
        ];

        for opt in optimizations {
            if let Some(new_id) = opt(self, proof_id) {
                return new_id;
            }
        }
        proof_id
    }

    /// Optimization: Remove reflexive congruence
    /// Congr(p, refl) -> p where refl proves t = t
    fn opt_reflexive_congr(&mut self, proof_id: ProofId) -> Option<ProofId> {
        let proof = self.get(proof_id);
        if let Justification::Congr {
            child_proof,
            proof: base_proof,
            ..
        } = proof.justification()
        {
            let child = self.get(*child_proof);
            if child.lhs() == child.rhs() {
                return Some(*base_proof);
            }
        }
        None
    }

    /// Optimization: Remove reflexive transitivity
    /// Trans(refl, p) -> p and Trans(p, refl) -> p
    fn opt_reflexive_trans(&mut self, proof_id: ProofId) -> Option<ProofId> {
        let proof = self.get(proof_id);
        if let Justification::Trans(p1, p2) = proof.justification() {
            let p1_proof = self.get(*p1);
            let p2_proof = self.get(*p2);
            if p1_proof.lhs() == p1_proof.rhs() {
                return Some(*p2);
            } else if p2_proof.lhs() == p2_proof.rhs() {
                return Some(*p1);
            }
        }
        None
    }

    /// Optimization: Remove reflexive symmetry
    /// Sym(p) -> p when p proves t = t (identity)
    fn opt_reflexive_sym(&mut self, proof_id: ProofId) -> Option<ProofId> {
        let proof = self.get(proof_id);
        if let Justification::Sym(inner) = proof.justification() {
            let inner_proof = self.get(*inner);
            // If inner proof is t = t, then Sym(t = t) is just t = t
            if inner_proof.lhs() == inner_proof.rhs() {
                return Some(*inner);
            }
        }
        None
    }

    /// Optimization: Collapse double symmetry
    /// Sym(Sym(p)) -> p
    fn opt_double_sym(&mut self, proof_id: ProofId) -> Option<ProofId> {
        let proof = self.get(proof_id);
        if let Justification::Sym(inner) = proof.justification() {
            let inner_proof = self.get(*inner);
            if let Justification::Sym(inner_inner) = inner_proof.justification() {
                return Some(*inner_inner);
            }
        }
        None
    }

    /// Optimization: Push symmetry through transitivity
    /// Sym(Trans(p1, p2)) -> Trans(Sym(p2), Sym(p1))
    fn opt_sym_trans(&mut self, proof_id: ProofId) -> Option<ProofId> {
        let proof = self.get(proof_id);
        if let Justification::Sym(inner) = proof.justification() {
            let inner_id = *inner;
            let inner_proof = self.get(inner_id);
            if let Justification::Trans(left, right) = inner_proof.justification() {
                let left_id = *left;
                let right_id = *right;

                // Get the lhs/rhs values we need before any mutations
                let left_proof = self.get(left_id);
                let left_lhs = left_proof.lhs();
                let left_rhs = left_proof.rhs();

                let right_proof = self.get(right_id);
                let right_lhs = right_proof.lhs();
                let right_rhs = right_proof.rhs();

                // Create Sym(p2): c = b
                let sym_right = Proof {
                    proposition: Proposition::new(right_rhs, right_lhs),
                    justification: Justification::Sym(right_id),
                };
                let sym_right_id = self.add_proof(sym_right);

                // Create Sym(p1): b = a
                let sym_left = Proof {
                    proposition: Proposition::new(left_rhs, left_lhs),
                    justification: Justification::Sym(left_id),
                };
                let sym_left_id = self.add_proof(sym_left);

                // Create Trans(Sym(p2), Sym(p1)): c = a
                let new_trans = Proof {
                    proposition: Proposition::new(right_rhs, left_lhs),
                    justification: Justification::Trans(sym_right_id, sym_left_id),
                };

                // Replace current proof
                self.id_to_proof[proof_id] = new_trans;
                return Some(proof_id);
            }
        }
        None
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
                    proof.proposition.lhs = old.lhs();
                    proof.proposition.rhs = new.rhs();
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
                        left_proof.rhs(),
                        right_proof.lhs(),
                        "transitivity requires matching middle terms"
                    );
                    proof.proposition.lhs = left_proof.lhs();
                    proof.proposition.rhs = right_proof.rhs();
                    changed = true;
                }
            }
            Justification::Sym(inner) => {
                let mapped_inner = f(self, *inner);
                if mapped_inner != *inner {
                    *inner = mapped_inner;
                    let inner_proof = self.get(*inner);
                    proof.proposition.lhs = inner_proof.rhs();
                    proof.proposition.rhs = inner_proof.lhs();
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
                    proof.proposition.lhs = base_proof.lhs();
                    proof.proposition.rhs =
                        self.replace_term_child(base_proof.rhs(), *child_index, child.rhs());
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

impl Proof {
    fn map_terms_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(TermId) -> TermId,
    {
        self.proposition.lhs = f(self.proposition.lhs);
        self.proposition.rhs = f(self.proposition.rhs);
        match &mut self.justification {
            Justification::Fiat => {}
            Justification::Rule {
                name: _,
                premise_proofs: _,
                substitution,
            } => {
                for term_id in substitution.values_mut() {
                    *term_id = f(*term_id);
                }
            }
            Justification::MergeFn {
                old_proof: _,
                new_proof: _,
                function: _,
            } => {}
            Justification::Congr {
                proof: _,
                child_index: _,
                child_proof: _,
            } => {}
            Justification::Trans(_, _) => {}
            Justification::Sym(_) => {}
        }
    }
}
