use crate::{
    ast::{Id, Literal},
    match_term_app,
    termdag::{Term, TermDag},
    util::HashMap,
    EGraph, Symbol, UnionFind,
};

struct ProofChecker {
    proven_equal: UnionFind,
    termdag: TermDag,
    // a vector of input and output for the EqGraph__ table
    to_check: Vec<(Term, Term)>,
    checked: HashMap<Term, Proof>,
}

const EQ_GRAPH_NAME: &str = "EqGraph__";

pub fn check_proof(egraph: &mut EGraph) {
    let (mut to_check, termdag) = egraph
        .function_to_dag(EQ_GRAPH_NAME.into(), usize::MAX)
        .unwrap();

    let mut proven_equal = UnionFind::default();
    for _i in 0..termdag.size() {
        proven_equal.make_set();
    }

    // sort by ascending age
    to_check.sort_by_key(|(_input, proof)| {
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
        termdag,
        to_check,
        checked: HashMap::default(),
    }
    .check();
}

#[derive(Clone, Debug)]
enum Proof {
    Equality(Term, Term),
    Provenance(Term),
    Rule(Symbol),
}

impl Proof {
    pub fn get_term(&self) -> Option<Term> {
        match self {
            Proof::Equality(_, term) => Some(term.clone()),
            Proof::Provenance(term) => Some(term.clone()),
            Proof::Rule(_sym) => None,
        }
    }
}

impl ProofChecker {
    fn check(&mut self) {
        for (input, proof_with_age) in self.to_check.clone() {
            match_term_app! (proof_with_age; {
                ("MakeProofWithAge__", [proof, age]) => {
                    if let Term::Lit(Literal::Int(lit)) = self.termdag.get(*age) {
                        println!("Checking age {}", lit);
                    } else {
                        panic!("not an int")
                    }
                    let checked = self.check_proof(self.termdag.get(*proof));
                    if let Proof::Equality(_a, _b) = checked {

                    } else {
                        panic!("not an equality")
                    }
                },
            });

            // todo
            match_term_app! (input; {
                ("EqGraph__", [_lhs, _rhs]) => ()
            });
        }
    }

    fn check_proof_list(&mut self, mut prooflist: Term) -> Vec<Proof> {
        let mut terms = vec![];
        loop {
            match_term_app! (prooflist; {
                ("Cons__", [proof, rest]) => {
                    terms.push(self.check_proof(self.termdag.get(*proof)));
                    prooflist = self.termdag.get(*rest);
                },
                ("ProofNull__", []) => break,
            })
        }
        terms
    }

    fn check_proof(&mut self, term: Term) -> Proof {
        if let Some(answer) = self.checked.get(&term) {
            return answer.clone();
        }
        println!("Checking {}", self.termdag.to_string(&term));
        println!();
        let res = match_term_app! (term.clone(); {
            ("Original__", [ast]) => {
                // TODO don't trust calls to "Original__"
                Proof::Provenance(self.termdag.get(*ast))
            },
            ("OriginalEq__", [term1, term2]) => {
                // TODO don't trust calls to "OriginalEq__"
                self.proven_equal.union(Id::from(*term1), Id::from(*term2), "".into());
                Proof::Equality(self.termdag.get(*term1), self.termdag.get(*term2))
            },
            ("Rule__", [prooflist, rule_name]) => {
                // TODO check the rule
                self.check_proof_list(self.termdag.get(*prooflist));
                if let Term::Lit(Literal::String(str)) = self.termdag.get(*rule_name) {
                    Proof::Rule(str)
                } else {
                    panic!("rule name not a string")
                }
            },
            ("RuleTerm__", [ruleproof, term]) => {
                self.check_proof(self.termdag.get(*ruleproof));
                // TODO check that the term would have been created on the rhs of the rule
                Proof::Provenance(self.termdag.get(*term))
            },
            ("RuleEquality__", [ruleproof, lhs, rhs]) => {
                self.check_proof(self.termdag.get(*ruleproof));
                // TODO check the lhs and rhs are unioned in the rhs of the rule
                self.proven_equal.union(Id::from(*lhs), Id::from(*rhs), "".into());
                Proof::Equality(self.termdag.get(*lhs), self.termdag.get(*rhs))
            },
            ("ComputePrim__", [prim, _from]) => {
                // TODO check the prim
                // Return a dummy proof
                Proof::Provenance(self.termdag.get(*prim))
            },
            ("Transitivity__", [prooflist]) => {
                let res = self.check_proof_list(self.termdag.get(*prooflist));
                assert!(!res.is_empty());

                for i in 0..(res.len()-1) {
                    let current = res[i].clone();
                    let next = res[i+1].clone();
                    if let Proof::Equality(_term1, term2) = current {
                        if let Proof::Equality(term3, _term4) = next {
                            if term2 != term3 {
                                panic!("Transitive proof did not match up");
                            }
                        } else {
                            panic!("Not a proof of equality");
                        }
                    } else {
                        panic!("Not a proof of equality");
                    }
                }
                if let Proof::Equality(term1, _term2) = res.first().unwrap() {
                    if let Proof::Equality(_term3, term4) = res.last().unwrap() {
                        self.proven_equal.union(Id::from(self.termdag.lookup(term1)), Id::from(self.termdag.lookup(term4)), "".into());
                        Proof::Equality(term1.clone(), term4.clone())
                    } else {
                        panic!("Not a proof of equality");
                    }
                } else {
                    panic!("Not a proof of equality");
                }
            },
            ("Flip__", [proof]) => {
                if let Proof::Equality(t1, t2) = self.check_proof(self.termdag.get(*proof)) {
                    Proof::Equality(t2, t1)
                } else {
                    panic!("Not a proof of equality");
                }
            },
            ("Congruence__", [term_proof, prooflist]) => {
                let terms = self.check_proof_list(self.termdag.get(*prooflist));
                let prov = self.check_proof(self.termdag.get(*term_proof));
                let term = prov.get_term().unwrap_or_else(|| panic!("Congruence term not a provenance proof"));
                if let Term::App(op, children_ids) = term.clone() {
                    let children = children_ids.iter().map(|id| self.termdag.get(*id)).collect::<Vec<_>>();
                    assert!(terms.len() == children.len());
                    let mut rhs_children = vec![];
                    for (proof, child) in terms.iter().zip(children.iter()) {
                        if let Proof::Equality(term1, term2) = proof {
                            assert!(term1 == child);
                            rhs_children.push(term2.clone());
                        } else {
                            panic!("Not a proof of equality");
                        }
                    }
                    let size_before = self.termdag.size();
                    let rhs = self.termdag.make(op, rhs_children);
                    let size_after = self.termdag.size();
                    assert!(size_before == size_after);
                    self.proven_equal.union(Id::from(self.termdag.lookup(&term)), Id::from(self.termdag.lookup(&rhs)), "".into());
                    Proof::Equality(term, rhs)
                } else {
                    Proof::Equality(term.clone(), term)
                }
            },
            ("DemandEq__", [term1, term2]) => {
                // should already be proven equal
                // due to edge age ordering
                assert_eq!(self.proven_equal.find(Id::from(*term1)), self.proven_equal.find(Id::from(*term2)));
                Proof::Equality(self.termdag.get(*term1), self.termdag.get(*term2))
            }
        });
        self.checked.insert(term, res.clone());
        res
    }
}
