use crate::{
    Term, TermDag, TermId,
    ast::{ResolvedExpr, ResolvedFact, ResolvedNCommand},
    proofs::proof_encoding_helpers::EncodingNames,
    util::{HEntry, HashMap, IndexSet, SymbolGen},
};
use egglog_ast::generic_ast::Literal;

pub type ProofId = usize;

/// A proof straight from the e-graph, not exposed to users.
struct RawProofStore {
    term_dag: TermDag,
    /// Bidirectional map between proof terms and their ids.
    store: IndexSet<RawProof>,
    encoding_names: EncodingNames,
}

pub(crate) fn proof_store_from_term(
    encoding_names: &EncodingNames,
    term_dag: TermDag,
    proof_term: TermId,
    rules: &Vec<ResolvedNCommand>,
) -> (ProofStore, ProofId) {
    let (raw_store, raw_proof_id) =
        RawProofStore::from_extracted(encoding_names, term_dag, proof_term);
    ProofStore::from_raw(rules, raw_store, raw_proof_id)
}

/// Justifies a single grounded equality t1 = t2.
/// Corresponds closely to the proof header in [`proof_encoding_helpers.rs`](crate::proofs::proof_encoding_helpers).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum RawProof {
    Fiat(TermId, TermId),
    /// Given a rule name and proofs for each premise, produces a proof of a grounded equality t1 = t2 from the body of the rule.
    /// The subsitution is implicit- in [`ProofTerm`] they are explicit.
    Rule(String, Vec<ProofId>, TermId, TermId),
    /// Given two proofs f(c1, c2, ..., old) = f(c1, c2, ..., old) and f(c1, c2, ..., new) = f(c1, c2, ..., new), produces a proof
    /// of f(c1, c2, ..., merge_fn)
    MergeFn(String, ProofId, ProofId),
    Trans(ProofId, ProofId),
    Sym(ProofId),
    /// given a proof that t1 = f(..., ci, ...)
    /// and the child index i of ci in the term f(..., ci, ...)
    /// and a proof that ci = c2,
    /// produces a justification that t1 = f(..., c2, ...)
    Congr(ProofId, usize, ProofId),
}

/// The [`ProofStore`] is similar to a [`TermDag`].
/// It's a hash-consed arena so that proofs can share sub-proofs.
#[derive(Clone, Debug)]
pub struct ProofStore {
    term_dag: TermDag,
    proof_id: HashMap<RawProof, ProofId>,
    id_to_proof: Vec<Proof>,
}

/// A proof shows that two grounded terms are equal, justified by a [`Justification`].
#[derive(Clone, Debug)]
struct Proof {
    lhs: TermId,
    rhs: TermId,
    justification: Justification,
}

/// Justifices a single grounded equality t1 = t2.
#[derive(Clone, Debug)]
enum Justification {
    /// Equalities added at the top level are justified by fiat.
    /// Also, primitive reflexive equalities like 2 = 2 are justified by Fiat.
    Fiat,
    /// Given a rule name and proofs for each premise, produces a proof of a grounded equality t1 = t2 from the body of the rule.
    Rule {
        name: String,
        premise_proofs: Vec<ProofId>,
        substitution: HashMap<String, TermId>,
    },
    /// Given two proofs f(c1, c2, ..., old) = f(c1, c2, ..., old) and f(c1, c2, ..., new) = f(c1, c2, ..., new), produces a proof
    /// of f(c1, c2, ..., merge_fn)
    MergeFn {
        function: String,
        old_proof: ProofId,
        new_proof: ProofId,
    },
    Trans(ProofId, ProofId),
    Sym(ProofId),
    /// Extends an equality proof with a congruence step.
    /// given a proof `proof` that t1 = f(..., ci, ...)
    /// and the child_index of ci in the term f(..., ci, ...)
    /// and a child_proof that ci = c2,
    /// proves t1 = f(..., c2, ...)
    Congr {
        proof: ProofId,
        child_index: usize,
        child_proof: ProofId,
    },
}

impl RawProofStore {
    /// After extracting a proof from the e-graph, convert it to a [`RawProof`].
    pub(crate) fn from_extracted(
        encoding_names: &EncodingNames,
        term_dag: TermDag,
        term: TermId,
    ) -> (Self, ProofId) {
        let mut store = RawProofStore {
            term_dag: term_dag.clone(),
            store: IndexSet::default(),
            encoding_names: encoding_names.clone(),
        };
        let parsed = store.parse_proof(term);
        (store, parsed)
    }

    fn parse_proof(&mut self, term_id: TermId) -> ProofId {
        let term = self.term_dag.get(term_id).clone();
        let Term::App(head, args) = term else {
            panic!("expected proof term to be an app, got {:?}", term);
        };

        let proof = if head == self.encoding_names.fiat_constructor {
            assert!(args.len() == 2, "fiat constructor should have 2 args");
            RawProof::Fiat(args[0], args[1])
        } else if head == self.encoding_names.rule_constructor {
            assert!(args.len() == 4, "rule constructor should have 4 args");
            let name = self.parse_string(args[0]);
            let premises = self.parse_proof_list(args[1]);
            RawProof::Rule(name, premises, args[2], args[3])
        } else if head == self.encoding_names.merge_fn_constructor {
            assert!(args.len() == 3, "merge constructor should have 3 args");
            let function = self.parse_string(args[0]);
            let old_proof = self.parse_proof(args[1]);
            let new_proof = self.parse_proof(args[2]);
            RawProof::MergeFn(function, old_proof, new_proof)
        } else if head == self.encoding_names.eq_trans_constructor {
            assert!(args.len() == 2, "trans constructor should have 2 args");
            let left = self.parse_proof(args[0]);
            let right = self.parse_proof(args[1]);
            RawProof::Trans(left, right)
        } else if head == self.encoding_names.eq_sym_constructor {
            assert!(args.len() == 1, "sym constructor should have 1 arg");
            let inner = self.parse_proof(args[0]);
            RawProof::Sym(inner)
        } else if head == self.encoding_names.congr_constructor {
            assert!(args.len() == 3, "congr constructor should have 3 args");
            let proof = self.parse_proof(args[0]);
            let child_index = self.parse_index(args[1]);
            let child_proof = self.parse_proof(args[2]);
            RawProof::Congr(proof, child_index, child_proof)
        } else {
            panic!("Unrecognized proof term head: {}", head);
        };

        self.add_proof(proof)
    }

    fn parse_proof_list(&mut self, list_term: TermId) -> Vec<ProofId> {
        let term = self.term_dag.get(list_term).clone();
        match term {
            Term::App(head, args) => {
                if head == self.encoding_names.pnil {
                    assert!(args.is_empty(), "pnil should not have arguments");
                    Vec::new()
                } else if head == self.encoding_names.pcons {
                    assert!(args.len() == 2, "pcons should have 2 arguments");
                    let head_proof = self.parse_proof(args[0]);
                    let rest = self.parse_proof_list(args[1]);
                    let mut list = Vec::with_capacity(rest.len() + 1);
                    list.push(head_proof);
                    list.extend(rest);
                    list
                } else {
                    panic!("expected proof list constructor, got {}", head);
                }
            }
            other => panic!("expected proof list, got {:?}", other),
        }
    }

    fn parse_string(&self, term_id: TermId) -> String {
        match self.term_dag.get(term_id) {
            Term::Lit(Literal::String(s)) => s.clone(),
            other => panic!("expected string literal in proof term, got {:?}", other),
        }
    }

    fn parse_index(&self, term_id: TermId) -> usize {
        match self.term_dag.get(term_id) {
            Term::Lit(Literal::Int(i)) if *i >= 0 => *i as usize,
            other => panic!(
                "expected non-negative integer literal for congruence index, got {:?}",
                other
            ),
        }
    }

    fn add_proof(&mut self, proof: RawProof) -> ProofId {
        if let Some(id) = self.store.get_index_of(&proof) {
            return id;
        }
        self.store.insert(proof);
        self.store.len() - 1
    }

    fn unwrap_ast(&self, term_id: TermId) -> TermId {
        let term = self.term_dag.get(term_id).clone();
        let Term::App(head, args) = term else {
            panic!("expected ast wrapper application");
        };
        assert!(
            args.len() == 1,
            "ast wrapper should have exactly one child, got {}",
            args.len()
        );
        if !self
            .encoding_names
            .sort_to_ast_constructor
            .values()
            .any(|constructor| constructor == &head)
        {
            panic!("unexpected ast constructor {head}");
        }
        args[0]
    }
}

impl ProofStore {
    fn from_raw(
        rules: &Vec<ResolvedNCommand>,
        raw_store: RawProofStore,
        raw_proof: ProofId,
    ) -> (ProofStore, ProofId) {
        let mut store = ProofStore {
            term_dag: raw_store.term_dag.clone(),
            proof_id: HashMap::default(),
            id_to_proof: Vec::new(),
        };

        let proof_id = store.convert_raw_proof(rules, &raw_store, raw_proof);
        (store, proof_id)
    }

    /// Converts a raw proof into a user-facing proof, recursively converting sub-proofs as needed.
    /// This adds new metadata to the proof, such as the substitution for rules.
    fn convert_raw_proof(
        &mut self,
        rules: &Vec<ResolvedNCommand>,
        raw_store: &RawProofStore,
        raw_proof_id: ProofId,
    ) -> ProofId {
        if let Some(&id) = self.proof_id.get(&raw_store.store[raw_proof_id]) {
            return id;
        }

        let raw_proof = &raw_store.store[raw_proof_id];
        let proof = match raw_proof {
            RawProof::Fiat(lhs, rhs) => Proof {
                lhs: raw_store.unwrap_ast(*lhs),
                rhs: raw_store.unwrap_ast(*rhs),
                justification: Justification::Fiat,
            },
            RawProof::Rule(name, premise_proofs, lhs, rhs) => {
                let converted_premises: Vec<ProofId> = premise_proofs
                    .iter()
                    .map(|pid| self.convert_raw_proof(rules, raw_store, *pid))
                    .collect();

                let substitution = self.compute_rule_substitution(rules, name, &converted_premises);

                Proof {
                    lhs: raw_store.unwrap_ast(*lhs),
                    rhs: raw_store.unwrap_ast(*rhs),
                    justification: Justification::Rule {
                        name: name.clone(),
                        premise_proofs: converted_premises,
                        substitution,
                    },
                }
            }
            RawProof::MergeFn(function, old_raw, new_raw) => {
                let old_proof_id = self.convert_raw_proof(rules, raw_store, *old_raw);
                let new_proof_id = self.convert_raw_proof(rules, raw_store, *new_raw);
                let old_proof = &self.id_to_proof[old_proof_id];
                let new_proof = &self.id_to_proof[new_proof_id];
                debug_assert_eq!(
                    old_proof.lhs, new_proof.lhs,
                    "merge function proofs must agree on the canonical term"
                );
                Proof {
                    lhs: old_proof.lhs,
                    rhs: new_proof.rhs,
                    justification: Justification::MergeFn {
                        function: function.clone(),
                        old_proof: old_proof_id,
                        new_proof: new_proof_id,
                    },
                }
            }
            RawProof::Trans(left_raw, right_raw) => {
                let left_id = self.convert_raw_proof(rules, raw_store, *left_raw);
                let right_id = self.convert_raw_proof(rules, raw_store, *right_raw);
                let left = &self.id_to_proof[left_id];
                let right = &self.id_to_proof[right_id];
                debug_assert_eq!(
                    left.rhs, right.lhs,
                    "transitivity requires matching middle terms"
                );
                Proof {
                    lhs: left.lhs,
                    rhs: right.rhs,
                    justification: Justification::Trans(left_id, right_id),
                }
            }
            RawProof::Sym(inner_raw) => {
                let inner_id = self.convert_raw_proof(rules, raw_store, *inner_raw);
                let inner = &self.id_to_proof[inner_id];
                Proof {
                    lhs: inner.rhs,
                    rhs: inner.lhs,
                    justification: Justification::Sym(inner_id),
                }
            }
            RawProof::Congr(proof_raw, child_index, child_raw) => {
                let base_id = self.convert_raw_proof(rules, raw_store, *proof_raw);
                let child_id = self.convert_raw_proof(rules, raw_store, *child_raw);
                let base_lhs = self.id_to_proof[base_id].lhs;
                let base_rhs = self.id_to_proof[base_id].rhs;
                let child_rhs = self.id_to_proof[child_id].rhs;
                let rhs = self.replace_term_child(base_rhs, *child_index, child_rhs);
                Proof {
                    lhs: base_lhs,
                    rhs,
                    justification: Justification::Congr {
                        proof: base_id,
                        child_index: *child_index,
                        child_proof: child_id,
                    },
                }
            }
        };

        let proof_id = self.id_to_proof.len();
        self.id_to_proof.push(proof);
        self.proof_id.insert(raw_proof.clone(), proof_id);
        proof_id
    }

    fn compute_rule_substitution(
        &self,
        rules: &[ResolvedNCommand],
        rule_name: &str,
        premise_proofs: &[ProofId],
    ) -> HashMap<String, TermId> {
        let substitution = HashMap::default();
        let Some(rule) = rules.iter().find_map(|cmd| match cmd {
            ResolvedNCommand::NormRule { rule } if rule.name == rule_name => Some(rule),
            _ => None,
        }) else {
            return substitution;
        };

        if rule.body.len() != premise_proofs.len() {
            return substitution;
        }

        let mut current_subst = substitution;
        for (fact, proof_id) in rule.body.iter().zip(premise_proofs.iter()) {
            let proof = &self.id_to_proof[*proof_id];
            let Some(next_subst) = self.try_unify_fact(fact, proof, &current_subst) else {
                return HashMap::default();
            };
            current_subst = next_subst;
        }

        current_subst
    }

    fn try_unify_fact(
        &self,
        fact: &ResolvedFact,
        proof: &Proof,
        base_subst: &HashMap<String, TermId>,
    ) -> Option<HashMap<String, TermId>> {
        match fact {
            ResolvedFact::Eq(_, lhs_expr, rhs_expr) => {
                if let Some(subst) = self
                    .try_unify_expr(lhs_expr, proof.lhs, base_subst)
                    .and_then(|candidate| self.try_unify_expr(rhs_expr, proof.rhs, &candidate))
                {
                    return Some(subst);
                }

                self.try_unify_expr(lhs_expr, proof.rhs, base_subst)
                    .and_then(|candidate| self.try_unify_expr(rhs_expr, proof.lhs, &candidate))
            }
            ResolvedFact::Fact(expr) => self
                .try_unify_expr(expr, proof.lhs, base_subst)
                .or_else(|| self.try_unify_expr(expr, proof.rhs, base_subst)),
        }
    }

    fn try_unify_expr(
        &self,
        expr: &ResolvedExpr,
        term_id: TermId,
        base_subst: &HashMap<String, TermId>,
    ) -> Option<HashMap<String, TermId>> {
        let mut candidate = base_subst.clone();
        if self.unify_expr(expr, term_id, &mut candidate).is_ok() {
            Some(candidate)
        } else {
            None
        }
    }

    fn unify_expr(
        &self,
        expr: &ResolvedExpr,
        term_id: TermId,
        substitution: &mut HashMap<String, TermId>,
    ) -> Result<(), ()> {
        match expr {
            ResolvedExpr::Lit(_, lit) => match self.term_dag.get(term_id) {
                Term::Lit(existing) if existing == lit => Ok(()),
                _ => Err(()),
            },
            ResolvedExpr::Var(_, var) => match substitution.entry(var.name.clone()) {
                HEntry::Vacant(entry) => {
                    entry.insert(term_id);
                    Ok(())
                }
                HEntry::Occupied(entry) => {
                    if *entry.get() == term_id {
                        Ok(())
                    } else {
                        Err(())
                    }
                }
            },
            ResolvedExpr::Call(_, call, args) => match self.term_dag.get(term_id) {
                Term::App(head, children) => {
                    if head != call.name() || children.len() != args.len() {
                        return Err(());
                    }
                    for (arg_expr, child_id) in args.iter().zip(children.iter()) {
                        self.unify_expr(arg_expr, *child_id, substitution)?;
                    }
                    Ok(())
                }
                _ => Err(()),
            },
        }
    }

    fn replace_term_child(
        &mut self,
        term_id: TermId,
        child_index: usize,
        new_child: TermId,
    ) -> TermId {
        let term = self.term_dag.get(term_id).clone();
        let Term::App(head, args) = term else {
            panic!("congruence requires an application term");
        };
        assert!(
            child_index < args.len(),
            "congruence child index {child_index} out of bounds for term with {} children",
            args.len()
        );

        let updated_children: Vec<Term> = args
            .iter()
            .enumerate()
            .map(|(idx, child_id)| {
                if idx == child_index {
                    self.term_dag.get(new_child).clone()
                } else {
                    self.term_dag.get(*child_id).clone()
                }
            })
            .collect();

        let new_term = self.term_dag.app(head.clone(), updated_children);
        self.term_dag.lookup(&new_term)
    }

    /// Get a string representation of the proof with the given id.
    /// The string representation is a pretty-printed s-expression block with
    /// let bindings for sub-proofs and sub-terms.
    pub fn proof_to_string(&self, proof_id: ProofId) -> String {
        let symbol_gen = &mut crate::util::SymbolGen::new("".to_string());
        let mut buffer = String::new();
        let res = self.print_to_buffer(symbol_gen, proof_id, &mut buffer);
        buffer.push_str(&res);
        buffer
    }

    /// Print a proof with the given id, with subproofs and terms
    /// added as let bindings in `buffer`.
    /// Returns the printed proof string.
    fn print_to_buffer(
        &self,
        symbol_gen: &mut SymbolGen,
        proof_id: ProofId,
        buffer: &mut String,
    ) -> String {
        let proof = &self.id_to_proof[proof_id];
        match &proof.justification {
            Justification::Fiat => {
                let t1 = self
                    .term_dag
                    .to_string_with_let(symbol_gen, proof.lhs, buffer);
                let t2 = self
                    .term_dag
                    .to_string_with_let(symbol_gen, proof.rhs, buffer);
                format!("(fiat {} {})\n", t1, t2)
            }
            Justification::Rule {
                name,
                premise_proofs,
                substitution,
            } => {
                let premises_strs: Vec<String> = premise_proofs
                    .iter()
                    .map(|pid| self.print_to_buffer(symbol_gen, *pid, buffer))
                    .collect();
                let subs_strs: Vec<String> = substitution
                    .iter()
                    .map(|(var, term_id)| {
                        let term_str = self
                            .term_dag
                            .to_string_with_let(symbol_gen, *term_id, buffer);
                        format!("({} {})", var, term_str)
                    })
                    .collect();
                let t1 = self
                    .term_dag
                    .to_string_with_let(symbol_gen, proof.lhs, buffer);
                let t2 = self
                    .term_dag
                    .to_string_with_let(symbol_gen, proof.rhs, buffer);
                format!(
                    "(rule {} (premises {}) (substitution {}) {} {})\n",
                    name,
                    premises_strs.join(" "),
                    subs_strs.join(" "),
                    t1,
                    t2
                )
            }
            Justification::MergeFn {
                function,
                old_proof,
                new_proof,
            } => {
                let old_str = self.print_to_buffer(symbol_gen, *old_proof, buffer);
                let new_str = self.print_to_buffer(symbol_gen, *new_proof, buffer);
                let t1 = self
                    .term_dag
                    .to_string_with_let(symbol_gen, proof.lhs, buffer);
                let t2 = self
                    .term_dag
                    .to_string_with_let(symbol_gen, proof.rhs, buffer);
                format!(
                    "(merge-fn {} {} {} {} {})\n",
                    function, old_str, new_str, t1, t2
                )
            }
            Justification::Trans(left, right) => {
                let left_str = self.print_to_buffer(symbol_gen, *left, buffer);
                let right_str = self.print_to_buffer(symbol_gen, *right, buffer);
                let t1 = self
                    .term_dag
                    .to_string_with_let(symbol_gen, proof.lhs, buffer);
                let t2 = self
                    .term_dag
                    .to_string_with_let(symbol_gen, proof.rhs, buffer);
                format!("(trans {} {} {} {})\n", left_str, right_str, t1, t2)
            }
            Justification::Sym(inner) => {
                let inner_str = self.print_to_buffer(symbol_gen, *inner, buffer);
                let t1 = self
                    .term_dag
                    .to_string_with_let(symbol_gen, proof.lhs, buffer);
                let t2 = self
                    .term_dag
                    .to_string_with_let(symbol_gen, proof.rhs, buffer);
                format!("(sym {} {} {})\n", inner_str, t1, t2)
            }
            Justification::Congr {
                proof: base,
                child_index,
                child_proof,
            } => {
                let base_str = self.print_to_buffer(symbol_gen, *base, buffer);
                let child_str = self.print_to_buffer(symbol_gen, *child_proof, buffer);
                let t1 = self
                    .term_dag
                    .to_string_with_let(symbol_gen, proof.lhs, buffer);
                let t2 = self
                    .term_dag
                    .to_string_with_let(symbol_gen, proof.rhs, buffer);
                format!(
                    "(congr {} {} {} {} {})\n",
                    base_str, child_str, child_index, t1, t2
                )
            }
        }
    }
}
