use crate::{
    Term, TermDag, TermId,
    ast::{
        FunctionSubtype, GenericAction, GenericNCommand, ResolvedExpr, ResolvedFact,
        ResolvedNCommand,
    },
    core::ResolvedCall,
    proofs::proof_format::{Justification, ProofId, ProofStore},
    typechecking::FuncType,
    util::{HEntry, HashMap, HashSet},
};
use thiserror::Error;

/// Gathers all global variables from a program and computes their values as terms
/// without using globals (i.e., all global references are replaced with their definitions).
///
/// This expects the program to be in proof normalized form where globals appear as Let actions.
pub(crate) fn gather_globals(
    prog: &[ResolvedNCommand],
    term_dag: &mut TermDag,
) -> HashMap<String, TermId> {
    let mut globals = HashMap::default();

    for cmd in prog {
        if let GenericNCommand::CoreAction(GenericAction::Let(_, var, expr)) = cmd {
            // All Let actions in the normalized desugared commands are globals
            let term_id = expr_to_term_without_globals(expr, term_dag, &globals);
            globals.insert(var.name.clone(), term_id);
        }
    }

    globals
}

/// Convert an expression to a term, replacing any global variable references
/// with their pre-computed term values.
fn expr_to_term_without_globals(
    expr: &ResolvedExpr,
    dag: &mut TermDag,
    globals: &HashMap<String, TermId>,
) -> TermId {
    let term = match expr {
        ResolvedExpr::Lit(_, lit) => dag.lit(lit.clone()),
        ResolvedExpr::Var(_, var) => {
            if var.is_global_ref {
                // Replace global reference with its computed value
                if let Some(&term_id) = globals.get(&var.name) {
                    return term_id;
                } else {
                    panic!(
                        "Global variable {} not found in globals map. \
                        This likely means it's being used before it's defined.",
                        var.name
                    );
                }
            } else {
                // Non-global variable - this shouldn't happen in global definitions
                panic!(
                    "Non-global variable {} found while constructing global term",
                    var.name
                );
            }
        }
        ResolvedExpr::Call(_, head, args) => {
            let arg_terms: Vec<Term> = args
                .iter()
                .map(|arg| {
                    let term_id = expr_to_term_without_globals(arg, dag, globals);
                    dag.get(term_id).clone()
                })
                .collect();
            dag.app(head.name().to_string(), arg_terms)
        }
    };
    dag.lookup(&term)
}

/// Errors that can occur during proof checking
#[derive(Debug, Clone, Error)]
pub enum ProofCheckError {
    /// The proof claims terms are equal but they don't match the actual terms
    #[error(
        "Proof {proof_id} claims to prove {expected_lhs:?} = {expected_rhs:?}, but actually proves {actual_lhs:?} = {actual_rhs:?}"
    )]
    TermMismatch {
        proof_id: ProofId,
        expected_lhs: TermId,
        expected_rhs: TermId,
        actual_lhs: TermId,
        actual_rhs: TermId,
    },
    /// Transitivity requires matching middle terms
    #[error(
        "Proof {proof_id}: transitivity requires matching middle terms, but left.rhs = {left_rhs:?} and right.lhs = {right_lhs:?}"
    )]
    TransitivityMismatch {
        proof_id: ProofId,
        left_rhs: TermId,
        right_lhs: TermId,
    },
    /// Congruence proof has invalid structure
    #[error("Proof {proof_id}: congruence error: {reason}")]
    CongruenceMismatch { proof_id: ProofId, reason: String },
    /// Rule application has wrong number of premises
    #[error(
        "Proof {proof_id}: rule '{rule_name}' expects {expected} premises, but proof has {actual}"
    )]
    RulePremiseCountMismatch {
        proof_id: ProofId,
        rule_name: String,
        expected: usize,
        actual: usize,
    },
    /// Rule substitution doesn't match the proof
    #[error("Proof {proof_id}: rule '{rule_name}' substitution error: {reason}")]
    RuleSubstitutionMismatch {
        proof_id: ProofId,
        rule_name: String,
        reason: String,
    },
    /// Primitive operation validator failed
    #[error("Proof {proof_id}: primitive '{function_name}' validation failed: {reason}")]
    PrimitiveValidationFailed {
        proof_id: ProofId,
        function_name: String,
        reason: String,
    },
    /// MergeFn proofs must agree on canonical term
    #[error(
        "Proof {proof_id}: merge function proofs must agree on canonical term, but old.lhs = {old_lhs:?} and new.lhs = {new_lhs:?}"
    )]
    MergeFnMismatch {
        proof_id: ProofId,
        old_lhs: TermId,
        new_lhs: TermId,
    },
    /// Could not find the rule referenced in a proof
    #[error("Proof {proof_id}: could not find rule '{rule_name}'")]
    RuleNotFound {
        proof_id: ProofId,
        rule_name: String,
    },
    /// Could not find the function referenced in a proof
    #[error("Proof {proof_id}: could not find function '{function_name}'")]
    FunctionNotFound {
        proof_id: ProofId,
        function_name: String,
    },
    /// Fiat proof is invalid (not a global or literal)
    #[error("Proof {proof_id}: Fiat proof invalid: {reason}")]
    InvalidFiat { proof_id: ProofId, reason: String },
}

/// Represents a proposition that can be proven
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Proposition {
    /// Two terms are equal
    TermsEq(TermId, TermId),
}

/// Context needed for proof checking
pub(crate) struct ProofCheckContext {
    /// Set of equalities established by global union/set actions
    /// Each entry is a pair (lhs, rhs) that was unified
    /// This includes reflexive equalities (term, term) for all globals
    global_equalities: HashSet<(TermId, TermId)>,
    /// Cache of already-checked proofs
    checked_proofs: HashMap<ProofId, Proposition>,
}

impl ProofCheckContext {
    /// Create a new proof check context by analyzing the program.
    /// This gathers all equalities established by global actions (unions and sets).
    fn new(prog: &[ResolvedNCommand], term_dag: &mut TermDag) -> Self {
        let mut ctx = ProofCheckContext {
            global_equalities: HashSet::default(),
            checked_proofs: HashMap::default(),
        };

        // First pass: gather all global Let bindings
        let globals = gather_globals(prog, term_dag);

        // Store reflexive equality for all global terms
        for &term_id in globals.values() {
            ctx.global_equalities.insert((term_id, term_id));
        }

        // Second pass: process all CoreActions to find global unions and sets
        for cmd in prog {
            if let GenericNCommand::CoreAction(action) = cmd {
                match action {
                    GenericAction::Union(_, lhs_expr, rhs_expr) => {
                        // Top-level unions create global equalities
                        if let (Ok(lhs_term), Ok(rhs_term)) = (
                            Self::eval_expr_with_subst(lhs_expr, term_dag, &globals),
                            Self::eval_expr_with_subst(rhs_expr, term_dag, &globals),
                        ) {
                            // Store both directions of the equality
                            ctx.global_equalities.insert((lhs_term, rhs_term));
                            ctx.global_equalities.insert((rhs_term, lhs_term));
                            // Store reflexive equalities
                            ctx.global_equalities.insert((lhs_term, lhs_term));
                            ctx.global_equalities.insert((rhs_term, rhs_term));
                        }
                    }
                    GenericAction::Set(_, func, args, rhs) => {
                        // Top-level sets create terms, store reflexive equality
                        let mut all_args = args.to_vec();
                        all_args.push(rhs.clone());
                        let call_expr =
                            ResolvedExpr::Call(crate::ast::Span::Panic, func.clone(), all_args);
                        if let Ok(term) = Self::eval_expr_with_subst(&call_expr, term_dag, &globals)
                        {
                            // Store reflexive equality for set-created terms
                            ctx.global_equalities.insert((term, term));
                        }
                    }
                    GenericAction::Let(_, _var, _expr) => {
                        // Let actions define new globals already handled by gather_globals
                    }
                    GenericAction::Expr(_, expr) => {
                        // Expr actions create reflexive equalities for their results
                        if let Ok(term) = Self::eval_expr_with_subst(expr, term_dag, &globals) {
                            ctx.global_equalities.insert((term, term));
                        }
                    }
                    _ => {
                        // Other actions (Panic, Change) don't create global equalities
                    }
                }
            }
        }

        ctx
    }

    /// Evaluate an expression with globals resolved
    fn eval_expr_with_subst(
        expr: &ResolvedExpr,
        dag: &mut TermDag,
        subst: &HashMap<String, TermId>,
    ) -> Result<TermId, ()> {
        match expr {
            ResolvedExpr::Lit(_, lit) => {
                let term = dag.lit(lit.clone());
                Ok(dag.lookup(&term))
            }
            ResolvedExpr::Var(_, var) => {
                if var.is_global_ref {
                    subst.get(&var.name).copied().ok_or(())
                } else {
                    Err(()) // Non-global variables not allowed in global context
                }
            }
            ResolvedExpr::Call(_, head, args) => {
                let mut arg_terms = Vec::new();
                for arg in args {
                    arg_terms.push(Self::eval_expr_with_subst(arg, dag, subst)?);
                }
                let term_refs: Vec<Term> =
                    arg_terms.iter().map(|&tid| dag.get(tid).clone()).collect();
                let term = dag.app(head.name().to_string(), term_refs);
                Ok(dag.lookup(&term))
            }
        }
    }

    fn in_globals(&self, lhs: TermId, rhs: TermId) -> bool {
        self.global_equalities.contains(&(lhs, rhs))
    }
}

impl ProofStore {
    /// Check that a proof is valid with respect to a typechecked program.
    pub(crate) fn check_proof(
        &self,
        proof_id: ProofId,
        program: &[ResolvedNCommand],
    ) -> Result<Proposition, ProofCheckError> {
        let mut term_dag = self.term_dag.clone();
        let mut ctx = ProofCheckContext::new(program, &mut term_dag);
        self.check_proof_with_context(proof_id, program, &mut ctx)
    }

    /// Internal recursive proof checker with context
    fn check_proof_with_context(
        &self,
        proof_id: ProofId,
        program: &[ResolvedNCommand],
        ctx: &mut ProofCheckContext,
    ) -> Result<Proposition, ProofCheckError> {
        // Check cache first
        if let Some(prop) = ctx.checked_proofs.get(&proof_id) {
            return Ok(prop.clone());
        }

        let proof = &self.id_to_proof[proof_id];
        let result = match &proof.justification {
            Justification::Fiat => {
                // if the both terms are primitives and equal, accept
                let term = self.term_dag.get(proof.lhs);
                if matches!(term, Term::Lit(_)) && proof.lhs == proof.rhs {
                    Ok(Proposition::TermsEq(proof.lhs, proof.rhs))
                } else if ctx.in_globals(proof.lhs, proof.rhs) {
                    Ok(Proposition::TermsEq(proof.lhs, proof.rhs))
                } else {
                    Err(ProofCheckError::InvalidFiat {
                        proof_id,
                        reason: format!(
                            "Fiat proof claims {:?} = {:?}, which is not established by globals",
                            self.term_dag.get(proof.lhs),
                            self.term_dag.get(proof.rhs)
                        ),
                    })
                }
            }

            Justification::Rule {
                name,
                premise_proofs,
                substitution,
            } => {
                // Find the rule in the program
                let rule = program
                    .iter()
                    .find_map(|cmd| match cmd {
                        GenericNCommand::NormRule { rule } if &rule.name == name => Some(rule),
                        _ => None,
                    })
                    .ok_or_else(|| ProofCheckError::RuleNotFound {
                        proof_id,
                        rule_name: name.clone(),
                    })?;

                // Check premise count
                if rule.body.len() != premise_proofs.len() {
                    return Err(ProofCheckError::RulePremiseCountMismatch {
                        proof_id,
                        rule_name: name.clone(),
                        expected: rule.body.len(),
                        actual: premise_proofs.len(),
                    });
                }

                // Check each premise proof
                let mut premise_propositions = Vec::new();
                for &premise_id in premise_proofs {
                    let prop = self.check_proof_with_context(premise_id, program, ctx)?;
                    premise_propositions.push(prop);
                }

                // Verify that premises match the rule body under the substitution
                for (fact, prop) in rule.body.iter().zip(premise_propositions.iter()) {
                    self.check_fact_matches_proposition(fact, prop, substitution, proof_id, name)?;
                }

                // Verify that the conclusion matches what the rule produces
                self.check_rule_produces_equality(
                    rule,
                    substitution,
                    proof.lhs,
                    proof.rhs,
                    proof_id,
                    name,
                )?;

                Ok(Proposition::TermsEq(proof.lhs, proof.rhs))
            }

            Justification::MergeFn {
                function,
                old_proof,
                new_proof,
            } => {
                // need to find the merge function from the program, then run the merge function to produce a new term
                todo!();
                Ok(Proposition::TermsEq(proof.lhs, proof.rhs))
            }

            Justification::Trans(left_id, right_id) => {
                // Check both sub-proofs
                let left_prop = self.check_proof_with_context(*left_id, program, ctx)?;
                let right_prop = self.check_proof_with_context(*right_id, program, ctx)?;

                let (left_lhs, left_rhs) = match left_prop {
                    Proposition::TermsEq(l, r) => (l, r),
                };
                let (right_lhs, right_rhs) = match right_prop {
                    Proposition::TermsEq(l, r) => (l, r),
                };

                // Check transitivity: left.rhs must equal right.lhs
                if left_rhs != right_lhs {
                    return Err(ProofCheckError::TransitivityMismatch {
                        proof_id,
                        left_rhs,
                        right_lhs,
                    });
                }

                // Result should be left_lhs = right_rhs
                if proof.lhs != left_lhs || proof.rhs != right_rhs {
                    return Err(ProofCheckError::TermMismatch {
                        proof_id,
                        expected_lhs: proof.lhs,
                        expected_rhs: proof.rhs,
                        actual_lhs: left_lhs,
                        actual_rhs: right_rhs,
                    });
                }

                Ok(Proposition::TermsEq(proof.lhs, proof.rhs))
            }

            Justification::Sym(inner_id) => {
                // Check the inner proof
                let inner_prop = self.check_proof_with_context(*inner_id, program, ctx)?;
                let (inner_lhs, inner_rhs) = match inner_prop {
                    Proposition::TermsEq(l, r) => (l, r),
                };

                // Symmetry swaps lhs and rhs
                if proof.lhs != inner_rhs || proof.rhs != inner_lhs {
                    return Err(ProofCheckError::TermMismatch {
                        proof_id,
                        expected_lhs: proof.lhs,
                        expected_rhs: proof.rhs,
                        actual_lhs: inner_rhs,
                        actual_rhs: inner_lhs,
                    });
                }

                Ok(Proposition::TermsEq(proof.lhs, proof.rhs))
            }

            Justification::Congr {
                proof: base_id,
                child_index,
                child_proof: child_id,
            } => {
                // Check the base proof (proves t1 = f(..., ci, ...))
                let base_prop = self.check_proof_with_context(*base_id, program, ctx)?;
                let (base_lhs, base_rhs) = match base_prop {
                    Proposition::TermsEq(l, r) => (l, r),
                };

                // Check the child proof (proves ci = c2)
                let child_prop = self.check_proof_with_context(*child_id, program, ctx)?;
                let (child_lhs, child_rhs) = match child_prop {
                    Proposition::TermsEq(l, r) => (l, r),
                };

                // base_rhs should be an application f(...)
                let (func_name, children) = match self.term_dag.get(base_rhs) {
                    Term::App(f, cs) => (f.clone(), cs.clone()),
                    _ => {
                        return Err(ProofCheckError::CongruenceMismatch {
                            proof_id,
                            reason: "Base proof rhs is not a function application".to_string(),
                        });
                    }
                };

                // Check child_index is valid
                if *child_index >= children.len() {
                    return Err(ProofCheckError::CongruenceMismatch {
                        proof_id,
                        reason: format!(
                            "Child index {} out of bounds for term with {} children",
                            child_index,
                            children.len()
                        ),
                    });
                }

                // Check that child_lhs matches the child at child_index
                if children[*child_index] != child_lhs {
                    return Err(ProofCheckError::CongruenceMismatch {
                        proof_id,
                        reason: format!(
                            "Child proof lhs {:?} doesn't match base term child {:?} at index {}",
                            self.term_dag.get(child_lhs),
                            self.term_dag.get(children[*child_index]),
                            child_index
                        ),
                    });
                }

                // Construct the expected new term by replacing the child
                let expected_rhs_children: Vec<Term> = children
                    .iter()
                    .enumerate()
                    .map(|(i, &child)| {
                        if i == *child_index {
                            self.term_dag.get(child_rhs).clone()
                        } else {
                            self.term_dag.get(child).clone()
                        }
                    })
                    .collect();

                let mut temp_dag = self.term_dag.clone();
                let expected_rhs_term = temp_dag.app(func_name, expected_rhs_children);
                let expected_rhs_id = temp_dag.lookup(&expected_rhs_term);

                // Verify proof.rhs matches expected
                if proof.rhs != expected_rhs_id {
                    return Err(ProofCheckError::CongruenceMismatch {
                        proof_id,
                        reason: format!(
                            "Proof rhs {:?} doesn't match expected {:?}",
                            self.term_dag.get(proof.rhs),
                            temp_dag.get(expected_rhs_id)
                        ),
                    });
                }

                // Verify proof.lhs matches base_lhs
                if proof.lhs != base_lhs {
                    return Err(ProofCheckError::CongruenceMismatch {
                        proof_id,
                        reason: "Proof lhs doesn't match base proof lhs".to_string(),
                    });
                }

                Ok(Proposition::TermsEq(proof.lhs, proof.rhs))
            }
        };

        // Cache the result
        if let Ok(ref prop) = result {
            ctx.checked_proofs.insert(proof_id, prop.clone());
        }

        result
    }

    /// Check that a fact matches a proposition under a substitution
    fn check_fact_matches_proposition(
        &self,
        fact: &ResolvedFact,
        prop: &Proposition,
        substitution: &HashMap<String, TermId>,
        proof_id: ProofId,
        rule_name: &str,
    ) -> Result<(), ProofCheckError> {
        let Proposition::TermsEq(lhs, rhs) = prop;
        match fact {
            // proof normal form for functions
            ResolvedFact::Eq(
                _,
                ResolvedExpr::Var(_, v),
                ResolvedExpr::Call(_, ResolvedCall::Func(FuncType { subtype: FunctionSubtype::Custom, ..}), args)
            ) => {
                // Evaluate both and put v as the last child of the call
                todo!()
            }
            ResolvedFact::Eq(_, lhs_expr, rhs_expr) => {
                let fact_lhs = self.eval_expr_with_subst(lhs_expr,  substitution)
                    .map_err(|_| ProofCheckError::RuleSubstitutionMismatch {
                        proof_id,
                        rule_name: rule_name.to_string(),
                        reason: "Failed to evaluate LHS expression under substitution".to_string(),
                    })?;
                let fact_rhs = self.eval_expr_with_subst(rhs_expr,  substitution)
                    .map_err(|_| ProofCheckError::RuleSubstitutionMismatch {
                        proof_id,
                        rule_name: rule_name.to_string(),
                        reason: "Failed to evaluate RHS expression under substitution".to_string(),
                    })?;

                if fact_lhs != *lhs || fact_rhs != *rhs {
                    return Err(ProofCheckError::RuleSubstitutionMismatch {
                        proof_id,
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Fact {:?} does not match proposition {:?} under substitution",
                            fact, prop
                        ),
                    });
                }

                Ok(())
            }
            ResolvedFact::Fact(expr) => {
                let fact_term = self.eval_expr_with_subst(expr,  substitution)
                    .map_err(|_| ProofCheckError::RuleSubstitutionMismatch {
                        proof_id,
                        rule_name: rule_name.to_string(),
                        reason: "Failed to evaluate Fact expression under substitution".to_string(),
                    })?;

                // For a Fact, we expect lhs == rhs == fact_term
                if fact_term != *lhs || fact_term != *rhs {
                    return Err(ProofCheckError::RuleSubstitutionMismatch {
                        proof_id,
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Fact {:?} does not match proposition {:?} under substitution",
                            fact, prop
                        ),
                    });
                }

                Ok(())
            }
        }
    }

    /// Evaluate an expression with a variable substitution
    fn eval_expr_with_subst(
        &self,
        expr: &ResolvedExpr,
        substitution: &HashMap<String, TermId>,
    ) -> Result<TermId, ProofCheckError> {
        match expr {
            ResolvedExpr::Lit(_, lit) => {
                let mut temp_dag = self.term_dag.clone();
                let term = temp_dag.lit(lit.clone());
                Ok(temp_dag.lookup(&term))
            }
            ResolvedExpr::Var(_, var) => substitution.get(&var.name).copied().ok_or_else(|| {
                ProofCheckError::RuleSubstitutionMismatch {
                    proof_id: 0, // Will be filled in by caller
                    rule_name: String::new(),
                    reason: format!("Variable {} not in substitution", var.name),
                }
            }),
            ResolvedExpr::Call(_, head, args) => {
                let mut arg_terms = Vec::new();
                for arg in args {
                    arg_terms.push(self.eval_expr_with_subst(arg, substitution)?);
                }
                let term_refs: Vec<Term> = arg_terms
                    .iter()
                    .map(|&tid| self.term_dag.get(tid).clone())
                    .collect();
                let mut temp_dag = self.term_dag.clone();
                let term = temp_dag.app(head.name().to_string(), term_refs);
                Ok(temp_dag.lookup(&term))
            }
        }
    }

    /// Check that a rule produces the claimed equality
    fn check_rule_produces_equality(
        &self,
        rule: &crate::ast::GenericRule<ResolvedCall, crate::ast::ResolvedVar>,
        substitution: &HashMap<String, TermId>,
        claimed_lhs: TermId,
        claimed_rhs: TermId,
        proof_id: ProofId,
        rule_name: &str,
    ) -> Result<(), ProofCheckError> {
        let mut found_match = false;

        for action in &rule.head.0 {
            match action {
                GenericAction::Union(_, lhs_expr, rhs_expr) => {
                    let action_lhs = self.eval_expr_with_subst(&lhs_expr, substitution)?;
                    let action_rhs = self.eval_expr_with_subst(&rhs_expr, substitution)?;

                    // Check if this union matches (in either direction)
                    if (action_lhs == claimed_lhs && action_rhs == claimed_rhs)
                        || (action_lhs == claimed_rhs && action_rhs == claimed_lhs)
                    {
                        found_match = true;
                        break;
                    }
                }
                GenericAction::Set(_, func, args, rhs) => {
                    // Set can produce f(...args, old_rhs) = f(...args, new_rhs)
                    // For simplicity, we'll be permissive here
                    // A more thorough check would verify the exact semantics
                    todo!()
                }
                _ => {}
            }
        }

        if !found_match && !rule.head.is_empty() {
            return Err(ProofCheckError::RuleSubstitutionMismatch {
                proof_id,
                rule_name: rule_name.to_string(),
                reason: format!(
                    "Rule head doesn't produce claimed equality {:?} = {:?}",
                    self.term_dag.get(claimed_lhs),
                    self.term_dag.get(claimed_rhs)
                ),
            });
        }

        Ok(())
    }
}
