use crate::{
    ast::{Id, Literal},
    function::ValueVec,
    match_term_app,
    termdag::{Term, TermDag},
    util::{HashMap, HashSet},
    EGraph, Symbol, UnionFind, Value,
};

struct ProofChecker {
    proven_equal: UnionFind,
    proven_provenance: HashSet<usize>,
    termdag: TermDag,
    // a vector of input and output for the EqGraph__ table
    to_check: Vec<(Term, Term)>,
}

const EQ_GRAPH_NAME: &str = "EqGraph__";

pub fn check_proof(egraph: &mut EGraph) {
    let (mut to_check, termdag) = egraph
        .extract_function(EQ_GRAPH_NAME.into(), usize::MAX)
        .unwrap();

    let mut proven_equal = UnionFind::default();
    for _i in 0..termdag.size() {
        proven_equal.make_set();
    }

    // sort by ascending age
    to_check.sort_by_key(|(input, proof)| {
        match_term_app! (proof; {
            ("MakeProofWithAge__", [_proof, age]) => {
                if let Term::Lit(Literal::Int(lit)) = termdag.get(*age) {
                    lit
                } else {
                    panic!("not an int")
                }
            },
        })
    });

    ProofChecker {
        proven_equal,
        proven_provenance: HashSet::default(),
        termdag,
        to_check,
    }
    .check();
}

impl ProofChecker {
    fn check(&mut self) {
        for (input, proof_with_age) in self.to_check.clone() {
            match_term_app! (proof_with_age; {
                ("MakeProofWithAge__", [proof, _age]) => {
                    self.check_proof(self.termdag.get(*proof));
                },
            });

            match_term_app! (input; {
                ("EqGraph__", [_lhs, _rhs]) => ()
            });
        }
    }

    fn check_proof_list(&mut self, prooflist: Term) {
        match_term_app! (prooflist; {
            ("Cons__", [proof, prooflist]) => {
                self.check_proof(self.termdag.get(*proof));
                self.check_proof_list(self.termdag.get(*prooflist));
            },
            ("Nil__", []) => (),
        })
    }

    fn check_proof(&mut self, term: Term) {
        match_term_app! (term; {
            ("Original__", [ast]) => {
                // TODO don't trust calls to "Original__"
                self.proven_provenance.insert(*ast);
            },
            ("OriginalEq__", [term1, term2]) => {
                // TODO don't trust calls to "OriginalEq__"
                self.proven_equal.union(Id::from(*term1), Id::from(*term2), "".into());
            },
            ("Rule__", [prooflist, rule_name]) => {
                // TODO check the rule
                self.check_proof_list(self.termdag.get(*prooflist));
            },
            ("RuleTerm__", [ruleproof, term]) => {
                self.check_proof(self.termdag.get(*ruleproof));
                // TODO check that the term would have been created on the rhs of the rule
                self.proven_provenance.insert(*term);
            },
            ("RuleEquality__", [ruleproof, lhs, rhs]) => {
                self.check_proof(self.termdag.get(*ruleproof));
                // TODO check the lhs and rhs are unioned in the rhs of the rule
                self.proven_equal.union(Id::from(*lhs), Id::from(*rhs), "".into());
            },
            ("ComputePrim__", [prim]) => {
                panic!("Not handled yet");
            },
            ("Transitivity__", [prooflist]) => {
                self.check_proof_list(self.termdag.get(*prooflist));
            },
            ("Flip__", [proof]) => {
                self.check_proof(self.termdag.get(*proof));
            },
            ("Congruence__", [term_proof, prooflist]) => {
                panic!("Not handled yet");
            },
            ("DemandEq__", [term1, term2]) => {
                // should already be proven equal
                // due to edge age ordering
                assert!(self.proven_equal.find(Id::from(*term1)) == self.proven_equal.find(Id::from(*term2)));
            }
        });
    }
}
