use std::rc::Rc;

use symbol_table::GlobalSymbol as Symbol;

use crate::{
    ast::{Actions, Command, Expr, Fact, GenericAction, GenericFact, Rule},
    HashMap, HashSet, Term, TermDag,
};

type ProofId = usize;

#[derive(Clone)]
pub struct ProofStore {
    store: Vec<ProofTerm>,
    memo: HashMap<ProofTerm, ProofId>,
    termdag: TermDag,
}

type Substitution = Vec<(Symbol, Term)>;

fn subst_get<'a>(subst: &'a Substitution, sym: Symbol) -> Option<&'a Term> {
    for ele in subst {
        if ele.0 == sym {
            return Some(&ele.1);
        }
    }
    None
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Proposition {
    TOk(Term),
    TEq(Term, Term),
}

/// Projects the appropriate expression of an action
/// TODO currently unused- do we need this?
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum ActionProof {
    APExprOK,
    APExprEq,
    APLetOK,
    APLetAct(Rc<ActionProof>),
    APUnionOk1,
    APUnionOk2,
    APUnion,
    APSeq1(Rc<ActionProof>),
    APSeq2(Rc<ActionProof>),
}

// todo how to ignore this warning?
#[warn(clippy::enum_variant_names)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum ProofTerm {
    /// proves a Proposition based on a rule application
    /// the subsitution gives the mapping from variables to terms
    /// the body_pfs gives proofs for each of the conditions in the query of the rule
    /// the act_pf gives a location in the action of the proposition
    PRule {
        rule_name: Symbol,
        subst: Substitution,
        body_pfs: Vec<ProofId>,
        result: Proposition,
    },
    /// A term is equal to itself- proves the proposition t = t
    PRefl {
        t_ok_pf: ProofId,
        t: Term,
    },
    /// The symmetric equality of eq_pf
    PSym {
        eq_pf: ProofId,
    },
    PTrans {
        pfxy: ProofId,
        pfyz: ProofId,
    },
    /// get a proof for the child of a term given a proof of a term
    PProj {
        pf_f_args_ok: ProofId,
        arg_idx: usize,
    },
    /// Proves f(x1, y1, ...) = f(x2, y2, ...) where f is fun_sym
    /// A proof via congruence- one proof for each child of the term
    /// pf_f_args_ok is a proof that the term with the lhs children is valid
    ///
    PCong {
        pf_args_eq: Vec<ProofId>,
        pf_f_args_ok: ProofId,
        fun_sym: Symbol,
    },
}

#[derive(PartialEq, Eq, Debug)]
enum ProofCheckError {
    Todo,
    WrongNumBodyProofs,
    ProofMismatch(Symbol, Fact, Proposition),
}

impl ProofStore {
    pub fn new(termdag: TermDag) -> Self {
        ProofStore {
            memo: HashMap::default(),
            termdag,
            store: vec![],
        }
    }

    fn to_id(&mut self, proof_term: &ProofTerm) -> ProofId {
        match self.memo.get(proof_term) {
            Some(existing) => *existing,
            None => {
                let fresh = self.store.len();
                self.store.push(proof_term.clone());
                self.memo.insert(proof_term.clone(), fresh);
                fresh
            }
        }
    }

    fn get_rule_by_name<'a>(
        self: &Self,
        prog: &'a Vec<Command>,
        name: Symbol,
    ) -> Result<&'a Rule, ProofCheckError> {
        for command in prog {
            if let Command::Rule {
                name: current_name,
                rule,
                ..
            } = command
            {
                if current_name == &name {
                    return Ok(rule);
                }
            }
        }
        Err(ProofCheckError::Todo)
    }

    fn check_rule_fires(
        self: &mut Self,
        prog: &Vec<Command>,
        rule_name: Symbol,
        subst: &Substitution,
        body_pfs: &Vec<ProofId>,
    ) -> Result<(), ProofCheckError> {
        let rule_ast = self.get_rule_by_name(prog, rule_name)?;
        let props = body_pfs
            .iter()
            .map(|pf| self.check_id(*pf, prog))
            .collect::<Result<Vec<_>, _>>()?;
        if props.len() != rule_ast.body.len() {
            return Err(ProofCheckError::WrongNumBodyProofs);
        }

        for (fact, prop) in rule_ast.body.iter().zip(props.iter()) {
            match (fact, prop) {
                (GenericFact::Eq(_span, expr1, expr2), Proposition::TEq(term1, term2)) => {
                    let mut intermediate_terms = HashSet::default();
                    let rule_term1 = self.substitute(expr1, subst, &mut intermediate_terms)?;
                    let rule_term2 = self.substitute(expr2, subst, &mut intermediate_terms)?;
                    if &rule_term1 != term1 || &rule_term2 != term2 {
                        return Err(ProofCheckError::ProofMismatch(
                            rule_name,
                            fact.clone(),
                            prop.clone(),
                        ));
                    }
                }
                (GenericFact::Fact(expr), Proposition::TOk(term)) => {
                    let mut intermediate_terms = HashSet::default();
                    let rule_term = self.substitute(expr, subst, &mut intermediate_terms)?;
                    if &rule_term != term {
                        return Err(ProofCheckError::ProofMismatch(
                            rule_name,
                            fact.clone(),
                            prop.clone(),
                        ));
                    }
                }
                _ => {
                    return Err(ProofCheckError::ProofMismatch(
                        rule_name,
                        fact.clone(),
                        prop.clone(),
                    ));
                }
            }
        }

        // TODO: actually check rule fires lol
        Ok(())
    }

    fn substitute(
        &mut self,
        expr: &Expr,
        substitution: &Substitution,
        intermediate_terms: &mut HashSet<Term>,
    ) -> Result<Term, ProofCheckError> {
        let res = match expr {
            crate::ast::GenericExpr::Lit(span, literal) => self.termdag.lit(literal.clone()),
            crate::ast::GenericExpr::Var(span, v) => {
                let Some(term) = subst_get(substitution, *v) else {
                    return Err(ProofCheckError::Todo);
                };
                term.clone()
            }
            crate::ast::GenericExpr::Call(_span, func, generic_exprs) => {
                let mut children = vec![];
                for expr in generic_exprs {
                    children.push(self.substitute(expr, substitution, intermediate_terms)?);
                }
                self.termdag.app(*func, children)
            }
        };

        intermediate_terms.insert(res.clone());
        Ok(res)
    }

    fn rule_propositions(
        &mut self,
        prog: &Vec<Command>,
        rule_name: Symbol,
        subst: &Substitution,
    ) -> Result<HashSet<Proposition>, ProofCheckError> {
        let rule = self.get_rule_by_name(prog, rule_name)?;

        let mut propositions = HashSet::default();
        let mut current_subst = subst.clone();

        for action in &rule.head.0 {
            match action {
                GenericAction::Let(_span, lhs, generic_expr) => {
                    let mut intermediate = HashSet::default();
                    let rhs = self.substitute(generic_expr, &current_subst, &mut intermediate)?;
                    if (subst_get(&current_subst, *lhs).is_some()) {
                        return Err(ProofCheckError::Todo);
                    }

                    for term in intermediate {
                        propositions.insert(Proposition::TOk(term));
                    }
                    current_subst.push((*lhs, rhs));
                }
                GenericAction::Expr(_span, generic_expr) => {
                    let mut intermediate = HashSet::default();
                    let _rhs = self.substitute(&generic_expr, &current_subst, &mut intermediate)?;

                    for term in intermediate {
                        propositions.insert(Proposition::TOk(term));
                    }
                }
                GenericAction::Set(span, _, generic_exprs, generic_expr) => todo!(),
                GenericAction::Change(span, change, _, generic_exprs) => todo!(),
                GenericAction::Union(span, generic_expr, generic_expr1) => todo!(),
                GenericAction::Extract(span, generic_expr, generic_expr1) => todo!(),
                GenericAction::Panic(span, _) => todo!(),
            }
        }

        Ok(propositions)
    }

    fn check(
        self: &mut Self,
        proof: &ProofTerm,
        prog: &Vec<Command>,
    ) -> Result<Proposition, ProofCheckError> {
        let proofid = self.to_id(proof);
        self.check_id(proofid, prog)
    }

    /// Check a particular proof id, returning the [`Proposition`] it proves.
    /// Borrows proof as mut in order to mutate the [`TermDag`] backing store.
    fn check_id(
        self: &mut Self,
        proof_id: ProofId,
        prog: &Vec<Command>,
    ) -> Result<Proposition, ProofCheckError> {
        eprintln!("checking proof id: {proof_id}");
        match self.store[proof_id].clone() {
            ProofTerm::PRule {
                rule_name,
                subst,
                body_pfs,
                result,
            } => {
                self.check_rule_fires(prog, rule_name, &subst, &body_pfs)?;
                let props = self.rule_propositions(prog, rule_name, &subst)?;
                eprintln!("props: {:?}", props);

                if props.contains(&result) {
                    Ok(result.clone())
                } else {
                    Err(ProofCheckError::Todo)
                }
            }
            ProofTerm::PRefl { t_ok_pf, t } => {
                // check t_ok_pf
                let prop = self.check_id(t_ok_pf, prog)?;
                match prop {
                    Proposition::TOk(term) => Ok(Proposition::TEq(term.clone(), term)),
                    Proposition::TEq(term, term1) => Err(ProofCheckError::Todo),
                }
            }
            ProofTerm::PSym { eq_pf } => {
                let prop = self.check_id(eq_pf, prog)?;
                match prop {
                    Proposition::TOk(_) => Err(ProofCheckError::Todo),
                    Proposition::TEq(a, b) => Ok(Proposition::TEq(b, a)),
                }
            }
            ProofTerm::PTrans { pfxy, pfyz } => {
                let propxy = self.check_id(pfxy, prog)?;
                let propyz = self.check_id(pfyz, prog)?;
                match (propxy, propyz) {
                    (Proposition::TEq(a, b), Proposition::TEq(b2, c)) if b == b2 => {
                        Ok(Proposition::TEq(a, c))
                    }
                    _ => Err(ProofCheckError::Todo),
                }
            }
            ProofTerm::PProj {
                pf_f_args_ok,
                arg_idx,
            } => {
                let prop = self.check_id(pf_f_args_ok, prog)?;
                match prop {
                    Proposition::TOk(Term::App(_symbol, items)) => match items.get(arg_idx) {
                        Some(subterm) => Ok(Proposition::TOk(self.termdag.get(*subterm).clone())),
                        None => Err(ProofCheckError::Todo),
                    },
                    _ => Err(ProofCheckError::Todo),
                }
            }
            ProofTerm::PCong {
                pf_args_eq,
                pf_f_args_ok,
                fun_sym,
            } => {
                let pf_f_args = self.check_id(pf_f_args_ok, prog)?;
                let Proposition::TOk(Term::App(symbol, children)) = pf_f_args else {
                    return Err(ProofCheckError::Todo);
                };

                if pf_args_eq.len() != children.len() {
                    return Err(ProofCheckError::Todo);
                };

                let mut new_children = vec![];
                for (child_term_id, child_proof) in children.iter().zip(pf_args_eq) {
                    let Proposition::TEq(child_term2, next_term) =
                        self.check_id(child_proof, prog)?
                    else {
                        return Err(ProofCheckError::Todo);
                    };
                    let child_term = self.termdag.get(*child_term_id);

                    if &child_term2 != child_term {
                        return Err(ProofCheckError::Todo);
                    }

                    new_children.push(next_term);
                }

                Ok(Proposition::TEq(
                    Term::App(symbol, children.clone()),
                    // use app here, the term might not exist in the termdag yet (TODO: brittle api!)
                    self.termdag.app(symbol, new_children),
                ))
            }
        }
    }
}

mod tests {
    use crate::proofs::proof::{ProofCheckError, ProofStore, ProofTerm, Proposition};
    #[cfg(test)]
    use crate::TermDag;

    #[test]
    fn no_precondition() {
        use crate::EGraph;
        let mut egraph = EGraph::default();
        let prog = "
(datatype Nat
  (S Nat)
  (O))

(rule ()
      ((S (S (O))))
      :name \"sso\")
    ";
        let parsed = egraph.parser.get_program_from_string(None, prog).unwrap();

        let mut termdag = TermDag::default();
        let ss0 = termdag.parse("(S (S (O)))").unwrap();
        let mut proof_store = ProofStore::new(termdag);
        let pt = ProofTerm::PRule {
            rule_name: "sso".into(),
            subst: vec![],
            body_pfs: vec![],
            result: Proposition::TOk(ss0.clone()),
        };
        let res = proof_store.check(&pt, &parsed);

        assert_eq!(res, Ok(Proposition::TOk(ss0.clone())));
    }

    #[test]
    fn handles_binding() {
        use crate::EGraph;
        let mut egraph = EGraph::default();
        let prog = "
(datatype Nat
  (S Nat)
  (O))

(rule ()
      ((let s0 (S (O)))
       (S s0))
      :name \"sso\")
    ";
        let parsed = egraph.parser.get_program_from_string(None, prog).unwrap();

        let mut termdag = TermDag::default();
        let ss0 = termdag.parse("(S (S (O)))").unwrap();
        let mut proof_store = ProofStore::new(termdag);
        let pt = ProofTerm::PRule {
            rule_name: "sso".into(),
            subst: vec![],
            body_pfs: vec![],
            result: Proposition::TOk(ss0.clone()),
        };
        let res = proof_store.check(&pt, &parsed);

        assert_eq!(res, Ok(Proposition::TOk(ss0.clone())));
    }

    #[test]
    fn simple_precondition() {
        use crate::EGraph;
        let mut egraph = EGraph::default();
        let prog = "
(datatype Nat
  (S Nat)
  (O))

(rule ()
      ((S (O)))
      :name \"one\")

(rule ((S a))
      ((S (S a)))
      :name \"succ\")
    ";
        let parsed = egraph.parser.get_program_from_string(None, prog).unwrap();

        let mut termdag = TermDag::default();
        let ss0 = termdag.parse("(S (S (O)))").unwrap();
        let s0 = termdag.parse("(S (O))").unwrap();
        let zero = termdag.parse("(O)").unwrap();
        let mut proof_store = ProofStore::new(termdag);
        let ptbad = ProofTerm::PRule {
            rule_name: "succ".into(),
            subst: vec![("a".into(), zero.clone())],
            body_pfs: vec![],
            result: Proposition::TOk(ss0.clone()),
        };

        let body_proof = ProofTerm::PRule {
            rule_name: "one".into(),
            subst: vec![],
            body_pfs: vec![],
            result: Proposition::TOk(s0.clone()),
        };
        let proof_of_0 = ProofTerm::PRule {
            rule_name: "one".into(),
            subst: vec![],
            body_pfs: vec![],
            result: Proposition::TOk(zero.clone()),
        };

        let ptbad2 = ProofTerm::PRule {
            rule_name: "succ".into(),
            subst: vec![("a".into(), zero.clone())],
            body_pfs: vec![proof_store.to_id(&proof_of_0)],
            result: Proposition::TEq(zero.clone(), zero.clone()),
        };

        let ptbad3 = ProofTerm::PRule {
            rule_name: "succ".into(),
            subst: vec![("a".into(), s0.clone())],
            body_pfs: vec![proof_store.to_id(&body_proof)],
            result: Proposition::TEq(zero.clone(), zero.clone()),
        };

        assert_eq!(
            proof_store.check(&body_proof, &parsed),
            Ok(Proposition::TOk(s0))
        );
        let ptgood = ProofTerm::PRule {
            rule_name: "succ".into(),
            subst: vec![("a".into(), zero)],
            body_pfs: vec![proof_store.to_id(&body_proof)],
            result: Proposition::TOk(ss0.clone()),
        };
        assert_eq!(
            proof_store.check(&ptgood, &parsed),
            Ok(Proposition::TOk(ss0.clone()))
        );
        assert_eq!(
            proof_store.check(&ptbad, &parsed),
            Err(ProofCheckError::WrongNumBodyProofs)
        );
        assert!(matches!(
            proof_store.check(&ptbad2, &parsed),
            Err(ProofCheckError::ProofMismatch(_, _, _))
        ));
        assert!(matches!(
            proof_store.check(&ptbad3, &parsed),
            Err(ProofCheckError::ProofMismatch(_, _, _))
        ));
    }
}
