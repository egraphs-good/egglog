use crate::{
    HashSet,
    ast::{Command, GenericAction, GenericFact, Rule},
};
use core_relations::Variable;
use egglog_bridge::FunctionId;
use egglog_bridge::proof_format::{Premise, ProofStore, TermId, TermProof, TermProofId};
use egglog_bridge::rule::Variable;
use numeric_id::DenseIdMap;

// Variable type must be made public in egglog_bridge or imported from crate::ast if available
// use egglog_bridge::rule::Variable;

#[derive(PartialEq, Eq, Debug)]
pub enum ProofCheckError {
    Todo,
    WrongNumBodyProofs,
    ProofMismatch(String /* rule name */, TermId /* expected term */),
}

/// Check a term proof by its id, returning the resulting TermId or error.
pub fn check(
    proof_store: &mut ProofStore,
    proof_id: TermProofId,
    prog: &Vec<Command>,
) -> Result<TermId, ProofCheckError> {
    check_id(proof_store, proof_id, prog)
}

pub fn check_id(
    proof_store: &mut ProofStore,
    proof_id: TermProofId,
    prog: &Vec<Command>,
) -> Result<TermId, ProofCheckError> {
    let term_proof = proof_store
        .get_term_proof(proof_id)
        .ok_or(ProofCheckError::Todo)?;
    match term_proof {
        TermProof::PRule {
            rule_name,
            subst,
            body_pfs,
            result,
        } => {
            for premise in body_pfs {
                match premise {
                    Premise::TermOk(term_pf_id) => {
                        check_id(proof_store, *term_pf_id, prog)?;
                    }
                    Premise::Eq(_eq_pf_id) => {
                        Ok(())?;
                    }
                }
            }
            let props = rule_propositions(proof_store, prog, rule_name.as_ref(), subst)?;
            if props.contains(result) {
                Ok(*result)
            } else {
                Err(ProofCheckError::Todo)
            }
        }
        TermProof::PProj {
            pf_f_args_ok,
            arg_idx,
        } => {
            // Implement projection logic using TermDag
            let parent_term = check_id(proof_store, *pf_f_args_ok, prog)?;
            // Use TermDag's proj method
            let projected = proof_store.termdag().proj(parent_term, *arg_idx);
            Ok(projected)
        }
        TermProof::PCong(_cong_pf) => Err(ProofCheckError::Todo),
        TermProof::PFiat { desc: _, term } => Ok(*term),
    }
}

pub fn get_rule_by_name<'a>(
    prog: &'a Vec<Command>,
    name: &str,
) -> Result<&'a Rule, ProofCheckError> {
    for command in prog {
        if let Command::Rule {
            name: current_name,
            rule,
            ..
        } = command
        {
            if current_name == name {
                return Ok(rule);
            }
        }
    }
    Err(ProofCheckError::Todo)
}

pub fn check_rule_fires(
    proof_store: &mut ProofStore,
    prog: &Vec<Command>,
    rule_name: &str,
    subst: &DenseIdMap<Variable, TermId>,
    body_pfs: &[TermProofId],
) -> Result<(), ProofCheckError> {
    let rule_ast = get_rule_by_name(prog, rule_name)?;
    let props = body_pfs
        .iter()
        .map(|pf| check_id(proof_store, *pf, prog))
        .collect::<Result<Vec<_>, _>>()?;
    if props.len() != rule_ast.body.len() {
        return Err(ProofCheckError::WrongNumBodyProofs);
    }
    for (fact, prop) in rule_ast.body.iter().zip(props.iter()) {
        match fact {
            GenericFact::Eq(_span, expr1, expr2) => {
                let mut intermediate_terms = HashSet::default();
                let rule_term1 = substitute(proof_store, &expr1, subst, &mut intermediate_terms)?;
                let rule_term2 = substitute(proof_store, &expr2, subst, &mut intermediate_terms)?;
                if &rule_term1 != prop || &rule_term2 != prop {
                    return Err(ProofCheckError::ProofMismatch(rule_name.to_string(), *prop));
                }
            }
            GenericFact::Fact(expr) => {
                let mut intermediate_terms = HashSet::default();
                let rule_term = substitute(proof_store, &expr, subst, &mut intermediate_terms)?;
                if &rule_term != prop {
                    return Err(ProofCheckError::ProofMismatch(rule_name.to_string(), *prop));
                }
            }
            _ => {
                return Err(ProofCheckError::ProofMismatch(rule_name.to_string(), *prop));
            }
        }
    }
    Ok(())
}

// Fix rule_propositions: use get_or_insert for term construction
pub fn rule_propositions(
    proof_store: &mut ProofStore,
    prog: &Vec<Command>,
    rule_name: &str,
    subst: &DenseIdMap<Variable, TermId>,
) -> Result<HashSet<TermId>, ProofCheckError> {
    let rule = get_rule_by_name(prog, rule_name)?;
    let mut propositions = HashSet::default();
    let mut current_subst = subst.clone();
    for action in &rule.head.0 {
        match action {
            GenericAction::Let(_span, lhs, generic_expr) => {
                let mut intermediate = HashSet::default();
                // lhs must be Variable, not String
                let lhs_var = Variable::new_const(lhs)?;
                let rhs = substitute(
                    proof_store,
                    &generic_expr,
                    &current_subst,
                    &mut intermediate,
                )?;
                if current_subst.get(&lhs_var).is_some() {
                    return Err(ProofCheckError::Todo);
                }
                for term in intermediate {
                    propositions.insert(term);
                }
                current_subst.insert(lhs_var, rhs);
            }
            GenericAction::Expr(_span, generic_expr) => {
                let mut intermediate = HashSet::default();
                let _rhs = substitute(
                    proof_store,
                    &generic_expr,
                    &current_subst,
                    &mut intermediate,
                )?;
                for term in intermediate {
                    propositions.insert(term);
                }
            }
            GenericAction::Set(_span, func, generic_exprs, generic_expr) => {
                let mut children = vec![];
                for expr in generic_exprs {
                    let mut intermediate = HashSet::default();
                    let term = substitute(proof_store, &expr, &current_subst, &mut intermediate)?;
                    for term in intermediate {
                        propositions.insert(term);
                    }
                    children.push(term);
                }
                let final_term = substitute(
                    proof_store,
                    &generic_expr,
                    &current_subst,
                    &mut HashSet::default(),
                )?;
                children.push(final_term.clone());
                // Convert func from String to FunctionId
                let func_id = FunctionId::new_const(func.as_str());
                let term = egglog_bridge::proof_format::Term::Func {
                    id: func_id,
                    args: children,
                };
                let res = proof_store.termdag().get_or_insert(&term);
                propositions.insert(res.clone());
            }
            GenericAction::Change(_span, change, _, _generic_exprs) => {
                match change {
                    crate::ast::Change::Delete => {
                        // delete adds an expression to the database,
                    }
                    crate::ast::Change::Subsume => todo!(),
                }
            }
            GenericAction::Union(_span, _generic_expr, _generic_expr1) => todo!(),
            GenericAction::Panic(_span, _) => todo!(),
        }
    }
    Ok(propositions)
}

mod tests {

    #[cfg(test)]
    use crate::TermDag;
    #[cfg(test)]
    use crate::proofs::proof::{ProofCheckError, ProofStore, ProofTerm, Proposition};

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
            body_pfs: vec![proof_store.id_of(&proof_of_0)],
            result: Proposition::TEq(zero.clone(), zero.clone()),
        };

        let ptbad3 = ProofTerm::PRule {
            rule_name: "succ".into(),
            subst: vec![("a".into(), s0.clone())],
            body_pfs: vec![proof_store.id_of(&body_proof)],
            result: Proposition::TEq(zero.clone(), zero.clone()),
        };

        assert_eq!(
            proof_store.check(&body_proof, &parsed),
            Ok(Proposition::TOk(s0))
        );
        let ptgood = ProofTerm::PRule {
            rule_name: "succ".into(),
            subst: vec![("a".into(), zero)],
            body_pfs: vec![proof_store.id_of(&body_proof)],
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
            Err(ProofCheckError::ProofMismatch(_, _))
        ));
        assert!(matches!(
            proof_store.check(&ptbad3, &parsed),
            Err(ProofCheckError::ProofMismatch(_, _))
        ));
    }
}

// Stub for substitute function
fn substitute(
    _proof_store: &mut ProofStore,
    _expr: &crate::ast::Expr,
    _subst: &DenseIdMap<Variable, TermId>,
    _intermediate: &mut HashSet<TermId>,
) -> Result<TermId, ProofCheckError> {
    Err(ProofCheckError::Todo)
}
