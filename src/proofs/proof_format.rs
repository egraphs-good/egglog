use crate::{
    Term, TermDag, TermId,
    ast::ResolvedNCommand,
    match_term_app,
    proofs::proof_encoding_helpers::EncodingNames,
    util::{HashMap, IndexSet},
};

pub type ProofId = usize;

/// A proof straight from the e-graph, not exposed to users.
struct RawProofStore {
    term_dag: TermDag,
    /// Bidirectional map between proof terms and their ids.
    store: IndexSet<RawProof>,
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
struct ProofStore {
    term_dag: TermDag,
    proof_id: HashMap<RawProof, ProofId>,
    id_to_proof: Vec<Proof>,
}

/// A proof shows that two grounded terms are equal, justified by a [`Justification`].
struct Proof {
    lhs: TermId,
    rhs: TermId,
    justification: Justification,
}

/// Justifices a single grounded equality t1 = t2.
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
        };
        let parsed = store.parse_proof(encoding_names, term);
        (store, parsed)
    }

    fn parse_proof(&mut self, encoding_names: &EncodingNames, term: TermId) -> ProofId {
        let term = self.term_dag.get(term);
        let Term::App(head, args) = term else {
            panic!("expected proof term to be an app");
        };
        let prf = if head == &encoding_names.fiat_constructor {
            assert!(args.len() == 2, "fiat constructor should have 2 args");
            RawProof::Fiat(args[0], args[1])
        } else {
            // TODO handle other variants from proof header
            panic!("Unrecognized proof term head: {}", head);
        };

        self.add_proof(prf)
    }

    fn add_proof(&mut self, proof: RawProof) -> ProofId {
        if let Some(id) = self.store.get_index_of(&proof) {
            return id;
        }
        self.store.insert(proof);
        self.store.len() - 1
    }
}

impl ProofStore {
    pub(crate) fn from_raw(
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
                lhs: *lhs,
                rhs: *rhs,
                justification: Justification::Fiat,
            },
            // TODO handle other variants. Rule will be difficult because we need to find the substitution using the original rule
            _ => panic!("Unrecognized raw proof variant"),
        };

        let proof_id = self.id_to_proof.len();
        self.id_to_proof.push(proof);
        self.proof_id.insert(raw_proof.clone(), proof_id);
        proof_id
    }
}
