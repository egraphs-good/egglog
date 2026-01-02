use crate::{
    Term, TermDag, TermId,
    ast::{
        FunctionSubtype, GenericAction, GenericNCommand, ResolvedExpr, ResolvedFact,
        ResolvedNCommand,
    },
    core::ResolvedCall,
    proofs::proof_format::{Justification, ProofId, ProofStore},
    typechecking::FuncType,
    util::{HashMap, HashSet, SymbolGen},
};
use thiserror::Error;

/// Result of processing actions: terms bound to variables and propositions
#[derive(Debug, Clone)]
pub(crate) struct ActionContext {
    /// Terms bound to variables (from Let actions)
    pub var_bindings: HashMap<String, TermId>,
    /// Propositions (equalities) implied by the actions
    pub propositions: HashSet<(TermId, TermId)>,
}

/// Gathers all global CoreActions from a program.
/// This extracts all actions that occur at the top level, filtering out NormRule and other commands.
pub(crate) fn gather_global_actions(
    prog: &[ResolvedNCommand],
) -> Vec<&GenericAction<ResolvedCall, crate::ast::ResolvedVar>> {
    let mut actions = Vec::new();
    for cmd in prog {
        if let GenericNCommand::CoreAction(action) = cmd {
            actions.push(action);
        }
    }
    actions
}

/// Given a sequence of actions, computes:
/// 1. All the terms bound to variables (from Let actions)
/// 2. All the propositions implied by the actions:
///    - Reflexive equalities for all subterms
///    - Ground equalities from union statements (bidirectional)
///    - Reflexive equalities from set statements
pub(crate) fn process_actions(
    actions: &[&GenericAction<ResolvedCall, crate::ast::ResolvedVar>],
    term_dag: &mut TermDag,
) -> ActionContext {
    let mut var_bindings = HashMap::default();
    let mut propositions = HashSet::default();

    // Single pass: process all actions, accumulating bindings and propositions
    for action in actions {
        match action {
            GenericAction::Let(_, var, expr) => {
                // Evaluate the expression and collect propositions
                if let Ok((term_id, new_props)) =
                    eval_expr_with_globals(expr, term_dag, &var_bindings)
                {
                    var_bindings.insert(var.name.clone(), term_id);
                    propositions.extend(new_props);
                }
            }
            GenericAction::Union(_, lhs_expr, rhs_expr) => {
                // Union creates ground equalities
                if let (Ok((lhs_term, lhs_props)), Ok((rhs_term, rhs_props))) = (
                    eval_expr_with_globals(lhs_expr, term_dag, &var_bindings),
                    eval_expr_with_globals(rhs_expr, term_dag, &var_bindings),
                ) {
                    // Collect propositions from evaluating both sides
                    propositions.extend(lhs_props);
                    propositions.extend(rhs_props);
                    // Store both directions of the equality
                    propositions.insert((lhs_term, rhs_term));
                    propositions.insert((rhs_term, lhs_term));
                }
            }
            GenericAction::Set(_, func, args, rhs) => {
                // Set creates reflexive equality for the resulting term
                let mut all_args = args.to_vec();
                all_args.push(rhs.clone());
                let call_expr = ResolvedExpr::Call(crate::ast::Span::Panic, func.clone(), all_args);
                if let Ok((term, new_props)) =
                    eval_expr_with_globals(&call_expr, term_dag, &var_bindings)
                {
                    propositions.extend(new_props);
                }
            }
            GenericAction::Expr(_, expr) => {
                // Expr creates reflexive equality for its result
                if let Ok((_, new_props)) = eval_expr_with_globals(expr, term_dag, &var_bindings) {
                    propositions.extend(new_props);
                }
            }
            _ => {
                // Other actions (Panic, Change) don't create propositions
            }
        }
    }

    ActionContext {
        var_bindings,
        propositions,
    }
}

/// Evaluate an expression with global variable bindings resolved.
/// Returns Ok((TermId, propositions)) if successful, where propositions include
/// all reflexive equalities for the term and its subterms.
/// Returns Err(()) if evaluation fails.
fn eval_expr_with_globals(
    expr: &ResolvedExpr,
    dag: &mut TermDag,
    globals: &HashMap<String, TermId>,
) -> Result<(TermId, HashSet<(TermId, TermId)>), ()> {
    let mut propositions = HashSet::default();

    let term_id = match expr {
        ResolvedExpr::Lit(_, lit) => {
            let term = dag.lit(lit.clone());
            dag.lookup(&term)
        }
        ResolvedExpr::Var(_, var) => {
            if var.is_global_ref {
                globals.get(&var.name).copied().ok_or(())?
            } else {
                return Err(()); // Non-global variables not allowed in global context
            }
        }
        ResolvedExpr::Call(_, head, args) => {
            let mut arg_terms = Vec::new();
            for arg in args {
                let (arg_term, arg_props) = eval_expr_with_globals(arg, dag, globals)?;
                arg_terms.push(arg_term);
                propositions.extend(arg_props);
            }
            let term_refs: Vec<Term> = arg_terms.iter().map(|&tid| dag.get(tid).clone()).collect();
            let term = dag.app(head.name().to_string(), term_refs);
            dag.lookup(&term)
        }
    };

    // Add reflexive equality for this term and all its subterms
    add_subterm_reflexive_equalities(term_id, dag, &mut propositions);

    Ok((term_id, propositions))
}

/// Add reflexive equalities for all subterms of a term
fn add_subterm_reflexive_equalities(
    term_id: TermId,
    term_dag: &TermDag,
    propositions: &mut HashSet<(TermId, TermId)>,
) {
    // Add reflexive equality for this term
    propositions.insert((term_id, term_id));

    // Recursively add for all children
    if let Term::App(_, children) = term_dag.get(term_id) {
        for &child_id in children {
            add_subterm_reflexive_equalities(child_id, term_dag, propositions);
        }
    }
}

/// Gathers all global variables from a program and computes their values as terms
/// without using globals (i.e., all global references are replaced with their definitions).
///
/// This expects the program to be in proof normalized form where globals appear as Let actions.
pub(crate) fn gather_globals(
    prog: &[ResolvedNCommand],
    term_dag: &mut TermDag,
) -> HashMap<String, TermId> {
    let actions = gather_global_actions(prog);
    let ctx = process_actions(&actions, term_dag);
    ctx.var_bindings
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
        // Use the new refactored functions
        let actions = gather_global_actions(prog);
        let action_ctx = process_actions(&actions, term_dag);

        ProofCheckContext {
            global_equalities: action_ctx.propositions,
            checked_proofs: HashMap::default(),
        }
    }

    fn in_globals(&self, lhs: TermId, rhs: TermId) -> bool {
        self.global_equalities.contains(&(lhs, rhs))
    }
}

/// Helper function to format a term with let bindings
fn format_term(term_dag: &TermDag, term_id: TermId) -> String {
    term_dag.to_string_with_let(&mut SymbolGen::new("".to_string()), term_id)
}

/// Helper function to format a substitution as a string
fn format_substitution(term_dag: &TermDag, substitution: &HashMap<String, TermId>) -> String {
    substitution
        .iter()
        .map(|(k, v)| format!("{} -> {}", k, format_term(term_dag, *v)))
        .collect::<Vec<_>>()
        .join(", ")
}

impl ProofStore {
    /// Check that a proof is valid with respect to a typechecked program.
    pub(crate) fn check_proof(
        &mut self,
        proof_id: ProofId,
        program: &[ResolvedNCommand],
    ) -> Result<Proposition, ProofCheckError> {
        let mut ctx = ProofCheckContext::new(program, &mut self.term_dag);
        self.check_proof_with_context(proof_id, program, &mut ctx)
    }

    /// Internal recursive proof checker with context
    fn check_proof_with_context(
        &mut self,
        proof_id: ProofId,
        program: &[ResolvedNCommand],
        ctx: &mut ProofCheckContext,
    ) -> Result<Proposition, ProofCheckError> {
        // Check cache first
        if let Some(prop) = ctx.checked_proofs.get(&proof_id) {
            return Ok(prop.clone());
        }

        let proof = self.id_to_proof[proof_id].clone();
        let result = match &proof.justification {
            Justification::Fiat => {
                // if the both terms are primitives and equal, accept
                let term = self.term_dag.get(proof.lhs);
                if (matches!(term, Term::Lit(_)) && proof.lhs == proof.rhs)
                    || ctx.in_globals(proof.lhs, proof.rhs)
                {
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
                    self.check_fact_matches_proposition(
                        fact,
                        prop,
                        substitution,
                        proof_id,
                        name,
                        program,
                    )?;
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

                let expected_rhs_term = self.term_dag.app(func_name, expected_rhs_children);
                let expected_rhs_id = self.term_dag.lookup(&expected_rhs_term);

                // Verify proof.rhs matches expected
                if proof.rhs != expected_rhs_id {
                    return Err(ProofCheckError::CongruenceMismatch {
                        proof_id,
                        reason: format!(
                            "Proof rhs {:?} doesn't match expected {:?}",
                            self.term_dag.get(proof.rhs),
                            self.term_dag.get(expected_rhs_id)
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
        &mut self,
        fact: &ResolvedFact,
        prop: &Proposition,
        substitution: &HashMap<String, TermId>,
        proof_id: ProofId,
        rule_name: &str,
        program: &[ResolvedNCommand],
    ) -> Result<(), ProofCheckError> {
        let Proposition::TermsEq(lhs, rhs) = prop;
        match fact {
            // proof normal form for functions: (= v (f args...))
            // In the term representation, custom functions store output as last arg: f(args..., v)
            ResolvedFact::Eq(
                _,
                ResolvedExpr::Var(_, v),
                ResolvedExpr::Call(
                    _,
                    ResolvedCall::Func(FuncType {
                        subtype: FunctionSubtype::Custom,
                        name,
                        ..
                    }),
                    args,
                ),
            ) => {
                // Get the output variable's term
                let var_term = substitution.get(&v.name).copied().ok_or_else(|| {
                    ProofCheckError::RuleSubstitutionMismatch {
                        proof_id,
                        rule_name: rule_name.to_string(),
                        reason: format!("Variable {} not in substitution", v.name),
                    }
                })?;

                // Evaluate all the input arguments
                let mut arg_terms = Vec::new();
                for arg in args {
                    arg_terms.push(self.eval_expr_with_subst(arg, substitution)?);
                }
                // Add the output variable as the last argument
                arg_terms.push(var_term);

                // Build the expected term: func(args..., output)
                let term_refs: Vec<Term> = arg_terms
                    .iter()
                    .map(|&tid| self.term_dag.get(tid).clone())
                    .collect();
                let expected_term = self.term_dag.app(name.clone(), term_refs);
                let expected_term_id = self.term_dag.lookup(&expected_term);

                // The proposition should be a reflexive equality for this term
                if *lhs != expected_term_id || *rhs != expected_term_id {
                    return Err(ProofCheckError::RuleSubstitutionMismatch {
                        proof_id,
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Function fact does not match proposition: expected reflexive equality for {:?}, got {:?} = {:?}",
                            self.term_dag.get(expected_term_id),
                            self.term_dag.get(*lhs),
                            self.term_dag.get(*rhs)
                        ),
                    });
                }

                Ok(())
            }
            ResolvedFact::Eq(_, lhs_expr, rhs_expr) => {
                let fact_lhs = self
                    .eval_expr_with_subst(lhs_expr, substitution)
                    .map_err(|_| ProofCheckError::RuleSubstitutionMismatch {
                        proof_id,
                        rule_name: rule_name.to_string(),
                        reason: "Failed to evaluate LHS expression under substitution".to_string(),
                    })?;
                let fact_lhs_term = self.term_dag.get(fact_lhs);
                let fact_lhs_str = self.term_dag.to_string(fact_lhs_term);
                let fact_rhs = self
                    .eval_expr_with_subst(rhs_expr, substitution)
                    .map_err(|_| ProofCheckError::RuleSubstitutionMismatch {
                        proof_id,
                        rule_name: rule_name.to_string(),
                        reason: "Failed to evaluate RHS expression under substitution".to_string(),
                    })?;
                let fact_rhs_term = self.term_dag.get(fact_rhs);
                let fact_rhs_str = self.term_dag.to_string(fact_rhs_term);

                if fact_lhs != *lhs || fact_rhs != *rhs {
                    return Err(ProofCheckError::RuleSubstitutionMismatch {
                        proof_id,
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Fact {} does not match proposition (= {fact_lhs_str} {fact_rhs_str}) under substitution",
                            fact.to_string(),
                        ),
                    });
                }

                Ok(())
            }
            ResolvedFact::Fact(expr) => {
                let fact_term = self.eval_expr_with_subst(expr, substitution).map_err(|_| {
                    ProofCheckError::RuleSubstitutionMismatch {
                        proof_id,
                        rule_name: rule_name.to_string(),
                        reason: "Failed to evaluate Fact expression under substitution".to_string(),
                    }
                })?;

                // For a Fact, we expect lhs == rhs == fact_term (reflexive equality)
                if fact_term != *lhs || fact_term != *rhs {
                    return Err(ProofCheckError::RuleSubstitutionMismatch {
                        proof_id,
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Fact {} does not match proposition under substitution {}. Got {}, expected {}.",
                            fact,
                            format_substitution(&self.term_dag, substitution),
                            format_term(&self.term_dag, *rhs),
                            format_term(&self.term_dag, fact_term)
                        ),
                    });
                }

                Ok(())
            }
        }
    }

    /// Evaluate an expression with a variable substitution
    fn eval_expr_with_subst(
        &mut self,
        expr: &ResolvedExpr,
        substitution: &HashMap<String, TermId>,
    ) -> Result<TermId, ProofCheckError> {
        match expr {
            ResolvedExpr::Lit(_, lit) => {
                let term = self.term_dag.lit(lit.clone());
                Ok(self.term_dag.lookup(&term))
            }
            ResolvedExpr::Var(_, var) => substitution.get(&var.name).copied().ok_or_else(|| {
                ProofCheckError::RuleSubstitutionMismatch {
                    proof_id: 0, // Will be filled in by caller
                    rule_name: String::new(),
                    reason: format!("Variable {} not in substitution", var.name),
                }
            }),
            ResolvedExpr::Call(_, head, args) => {
                // Evaluate all arguments first
                let mut arg_terms = Vec::new();
                for arg in args {
                    arg_terms.push(self.eval_expr_with_subst(arg, substitution)?);
                }

                match head {
                    ResolvedCall::Primitive(prim) => {
                        // Use the validator to compute the primitive result
                        if let Some(validator) = prim.validator() {
                            let result =
                                validator(&mut self.term_dag, &arg_terms).ok_or_else(|| {
                                    ProofCheckError::PrimitiveValidationFailed {
                                        proof_id: 0, // Will be filled in by caller
                                        function_name: prim.name().to_string(),
                                        reason:
                                            "Validator returned None - primitive operation failed"
                                                .to_string(),
                                    }
                                })?;
                            Ok(result)
                        } else {
                            // No validator available - primitives without validators can't be checked in proofs
                            Err(ProofCheckError::PrimitiveValidationFailed {
                                proof_id: 0,
                                function_name: prim.name().to_string(),
                                reason: "Primitive has no validator - cannot verify in proof"
                                    .to_string(),
                            })
                        }
                    }
                    ResolvedCall::Func(func) => {
                        match func.subtype {
                            FunctionSubtype::Constructor => {
                                // Constructors: build the term normally
                                let term_refs: Vec<Term> = arg_terms
                                    .iter()
                                    .map(|&tid| self.term_dag.get(tid).clone())
                                    .collect();
                                let term = self.term_dag.app(func.name.clone(), term_refs);
                                Ok(self.term_dag.lookup(&term))
                            }
                            FunctionSubtype::Custom => {
                                // Custom functions should not appear in proof normal form!
                                // They should be in the form (= v (f args...)) in the rule body
                                panic!(
                                    "Custom function {} should not appear in expression evaluation during proof checking. \
                                    Functions should be in proof normal form: (= output_var (function args...))",
                                    func.name
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Check that a rule produces the claimed equality
    fn check_rule_produces_equality(
        &mut self,
        rule: &crate::ast::GenericRule<ResolvedCall, crate::ast::ResolvedVar>,
        substitution: &HashMap<String, TermId>,
        claimed_lhs: TermId,
        claimed_rhs: TermId,
        proof_id: ProofId,
        rule_name: &str,
    ) -> Result<(), ProofCheckError> {
        // Use process_actions to get propositions from the rule head
        // Note: process_actions expects global variable bindings, but substitution
        // has the same structure, so we can pass it directly
        let action_refs: Vec<&GenericAction<ResolvedCall, crate::ast::ResolvedVar>> =
            rule.head.0.iter().collect();
        let action_ctx = process_actions_with_substitution(&action_refs, &mut self.term_dag, substitution);

        // Check if the claimed equality is in the propositions
        if action_ctx.propositions.contains(&(claimed_lhs, claimed_rhs))
            || action_ctx.propositions.contains(&(claimed_rhs, claimed_lhs))
        {
            return Ok(());
        }

        // If not found, provide detailed error message using helper functions
        Err(ProofCheckError::RuleSubstitutionMismatch {
            proof_id,
            rule_name: rule_name.to_string(),
            reason: format!(
                "Rule head doesn't produce claimed equality Lhs:\n{}\nRHS:\n{}\nSUBST:\n{}",
                format_term(&self.term_dag, claimed_lhs),
                format_term(&self.term_dag, claimed_rhs),
                format_substitution(&self.term_dag, substitution)
            ),
        })
    }
}

/// Process actions with a substitution map (used for rule checking)
/// This is like process_actions but treats the map as a substitution rather than globals
fn process_actions_with_substitution(
    actions: &[&GenericAction<ResolvedCall, crate::ast::ResolvedVar>],
    term_dag: &mut TermDag,
    substitution: &HashMap<String, TermId>,
) -> ActionContext {
    let mut propositions = HashSet::default();

    for action in actions {
        match action {
            GenericAction::Union(_, lhs_expr, rhs_expr) => {
                // Union creates ground equalities
                if let (Ok((lhs_term, lhs_props)), Ok((rhs_term, rhs_props))) = (
                    eval_expr_with_globals(lhs_expr, term_dag, substitution),
                    eval_expr_with_globals(rhs_expr, term_dag, substitution),
                ) {
                    propositions.extend(lhs_props);
                    propositions.extend(rhs_props);
                    propositions.insert((lhs_term, rhs_term));
                    propositions.insert((rhs_term, lhs_term));
                }
            }
            GenericAction::Expr(_, expr) => {
                // Expr creates reflexive equality
                if let Ok((_, new_props)) = eval_expr_with_globals(expr, term_dag, substitution) {
                    propositions.extend(new_props);
                }
            }
            _ => {
                // Other action types don't produce propositions we need
            }
        }
    }

    ActionContext {
        var_bindings: HashMap::default(), // Not needed for rule checking
        propositions,
    }
}
