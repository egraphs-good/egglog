//! Proof checking for egglog proofs.
//! Given an egglog program and a proof for some proposition, check that the proof is valid.
//! The main work is to check proofs for rules, ensuring that under the substitution
//! the rule would have matched the propositions for each premise, and that the conclusion
//! is contained in the rule's actions.
//! For checking queries and actions, we evaluate the expressions under a substitution
//! and produce a set of valid propositions with `process_actions`.
//! The set of valid propositions includes reflexive equalities for all subterms.

use crate::{
    Term, TermDag, TermId,
    ast::{
        FunctionSubtype, GenericAction, GenericNCommand, ResolvedExpr, ResolvedFact,
        ResolvedNCommand,
    },
    core::ResolvedCall,
    proofs::proof_format::{Justification, ProofId, ProofStore, Proposition},
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
    pub propositions: HashSet<Proposition>,
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

/// Run a merge function and return the resulting term, as well as a set of propositions learned.
pub(crate) fn run_merge(
    term_dag: &mut TermDag,
    func_name: &str,
    prog: &[ResolvedNCommand],
    old_term: TermId,
    new_term: TermId,
) -> Result<(TermId, HashSet<Proposition>), ProofCheckError> {
    let mut subst = HashMap::default();
    subst.insert("old".to_string(), old_term);
    subst.insert("new".to_string(), new_term);
    for cmd in prog {
        if let GenericNCommand::Function(func_decl) = cmd {
            if func_decl.name == func_name {
                // run the merge function for this function using eval_expr
                let expr =
                    func_decl
                        .merge
                        .as_ref()
                        .ok_or_else(|| ProofCheckError::FunctionNotFound {
                            function_name: func_name.to_string(),
                        })?;
                return eval_expr_with_subst("merge_function", expr, term_dag, &subst);
            }
        }
    }
    Err(ProofCheckError::FunctionNotFound {
        function_name: func_name.to_string(),
    })
}

/// Given a sequence of actions, computes:
/// 1. All the terms bound to variables (from Let actions)
/// 2. All the propositions implied by the actions:
///    - Reflexive equalities for all subterms
///    - Ground equalities from union statements (bidirectional)
///    - Reflexive equalities from set statements
pub(crate) fn process_actions(
    rule_name: &str,
    mut bindings: HashMap<String, TermId>,
    actions: &[&GenericAction<ResolvedCall, crate::ast::ResolvedVar>],
    term_dag: &mut TermDag,
) -> Result<ActionContext, ProofCheckError> {
    let mut propositions = HashSet::default();

    // Single pass: process all actions, accumulating bindings and propositions
    for action in actions {
        match action {
            GenericAction::Let(_, var, expr) => {
                // Evaluate the expression and collect propositions
                let (term_id, new_props) =
                    eval_expr_with_subst(rule_name, expr, term_dag, &bindings)?;
                bindings.insert(var.name.clone(), term_id);
                propositions.extend(new_props);
            }
            GenericAction::Union(_, lhs_expr, rhs_expr) => {
                // Union creates ground equalities
                let (lhs_term, lhs_props) =
                    eval_expr_with_subst(rule_name, lhs_expr, term_dag, &bindings)?;
                let (rhs_term, rhs_props) =
                    eval_expr_with_subst(rule_name, rhs_expr, term_dag, &bindings)?;

                // Collect propositions from evaluating both sides
                propositions.extend(lhs_props);
                propositions.extend(rhs_props);
                // Store both directions of the equality
                propositions.insert(Proposition::new(lhs_term, rhs_term));
                propositions.insert(Proposition::new(rhs_term, lhs_term));
            }
            GenericAction::Set(_, func, args, rhs) => {
                // Set creates reflexive equality for the resulting term
                let mut all_args = args.to_vec();
                all_args.push(rhs.clone());
                let call_expr = ResolvedExpr::Call(crate::ast::Span::Panic, func.clone(), all_args);
                let (_term, new_props) =
                    eval_expr_with_subst(rule_name, &call_expr, term_dag, &bindings)?;
                propositions.extend(new_props);
            }
            GenericAction::Expr(_, expr) => {
                // Expr creates reflexive equality for its result
                let (_, new_props) = eval_expr_with_subst(rule_name, expr, term_dag, &bindings)?;
                propositions.extend(new_props);
            }
            GenericAction::Panic(_, _) => {
                // Panics do not create propositions
            }
            GenericAction::Change(_, _, _, _) => {
                // Changes do not create propositions
            }
        }
    }

    Ok(ActionContext {
        var_bindings: bindings,
        propositions,
    })
}

/// Evaluate an expression under a substitution.
/// Returns Ok((TermId, propositions)) if successful, where propositions include
/// all reflexive equalities for the term and its subterms.
/// Returns Err(()) if evaluation fails.
fn eval_expr_with_subst(
    rule_name: &str,
    expr: &ResolvedExpr,
    dag: &mut TermDag,
    subst: &HashMap<String, TermId>,
) -> Result<(TermId, HashSet<Proposition>), ProofCheckError> {
    let mut propositions = HashSet::default();

    let term_id = match expr {
        ResolvedExpr::Lit(_, lit) => dag.lit(lit.clone()),
        ResolvedExpr::Var(_, var) => {
            subst
                .get(&var.name)
                .copied()
                .ok_or(ProofCheckError::RuleSubstitutionMismatch {
                    rule_name: rule_name.to_string(),
                    reason: format!("Variable {} not found in substitution", var.name),
                })?
        }
        ResolvedExpr::Call(_, head, args) => match head {
            ResolvedCall::Func(_func_type) => {
                let mut arg_terms = Vec::new();
                for arg in args {
                    let (arg_term, arg_props) = eval_expr_with_subst(rule_name, arg, dag, subst)?;
                    arg_terms.push(arg_term);
                    propositions.extend(arg_props);
                }
                dag.app(head.name().to_string(), arg_terms)
            }
            ResolvedCall::Primitive(specialized_primitive) => {
                // run validator, throwing error if it fails
                let mut arg_terms = Vec::new();
                for arg in args {
                    let (arg_term, arg_props) = eval_expr_with_subst(rule_name, arg, dag, subst)?;
                    arg_terms.push(arg_term);
                    propositions.extend(arg_props);
                }
                // checked by earlier code showing this program supports proofs
                let validator = specialized_primitive
                    .validator()
                    .expect("Expected primitive to have validator since proof mode is enabled");
                validator(dag, &arg_terms).ok_or(ProofCheckError::PrimitiveValidationFailed {
                    function_name: specialized_primitive.name().to_string(),
                    reason: "Primitive validator failed".to_string(),
                })?
            }
        },
    };

    // Add reflexive equality for this term and all its subterms
    add_subterm_reflexive_equalities(term_id, dag, &mut propositions);

    Ok((term_id, propositions))
}

/// Add reflexive equalities for all subterms of a term
fn add_subterm_reflexive_equalities(
    term_id: TermId,
    term_dag: &TermDag,
    propositions: &mut HashSet<Proposition>,
) {
    // Add reflexive equality for this term
    propositions.insert(Proposition::new(term_id, term_id));

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
    let ctx = process_actions("global_action", HashMap::default(), &actions, term_dag)
        .expect("Failed to process global actions");
    ctx.var_bindings
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
    #[error("Rule '{rule_name}' substitution error: {reason}")]
    RuleSubstitutionMismatch { rule_name: String, reason: String },
    /// Primitive operation validator failed
    #[error("Primitive '{function_name}' validation failed: {reason}")]
    PrimitiveValidationFailed {
        function_name: String,
        reason: String,
    },
    /// Could not find the rule referenced in a proof
    #[error("Could not find rule '{rule_name}'")]
    RuleNotFound { rule_name: String },
    /// Could not find the function referenced in a proof
    #[error("Could not find function '{function_name}'")]
    FunctionNotFound { function_name: String },
    /// Fiat proof is invalid (not a global or literal)
    #[error("Proof {proof_id}: Fiat proof invalid: {reason}")]
    InvalidFiat { proof_id: ProofId, reason: String },
    /// MergeFn proof check failed
    #[error("Proof {proof_id}: MergeFn error: {reason}")]
    MergeFnError { proof_id: ProofId, reason: String },
}

/// Context needed for proof checking
pub(crate) struct ProofCheckContext {
    /// Set of equalities established by global union/set actions
    /// Each entry is a pair (lhs, rhs) that was unified
    /// This includes reflexive equalities (term, term) for all globals
    global_equalities: HashSet<Proposition>,
    /// Map of global variable names to their TermIds
    global_bindings: HashMap<String, TermId>,
    /// Cache of already-checked proofs
    checked_proofs: HashMap<ProofId, Proposition>,
}

impl ProofCheckContext {
    /// Create a new proof check context by analyzing the program.
    /// This gathers all equalities established by global actions (unions and sets).
    fn new(prog: &[ResolvedNCommand], term_dag: &mut TermDag) -> Self {
        // Use the new refactored functions
        let actions = gather_global_actions(prog);
        let action_ctx = process_actions("global_actions", HashMap::default(), &actions, term_dag)
            .expect("Failed to process global actions for proof checking");

        ProofCheckContext {
            global_equalities: action_ctx.propositions,
            checked_proofs: HashMap::default(),
            global_bindings: action_ctx.var_bindings,
        }
    }

    fn in_globals(&self, lhs: TermId, rhs: TermId) -> bool {
        self.global_equalities.contains(&Proposition::new(lhs, rhs))
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
                let term = self.term_dag.get(proof.lhs());
                if (matches!(term, Term::Lit(_)) && proof.lhs() == proof.rhs())
                    || ctx.in_globals(proof.lhs(), proof.rhs())
                {
                    Ok(Proposition::new(proof.lhs(), proof.rhs()))
                } else {
                    Err(ProofCheckError::InvalidFiat {
                        proof_id,
                        reason: format!(
                            "Fiat proof claims {:?} = {:?}, which is not established by globals",
                            self.term_dag.get(proof.lhs()),
                            self.term_dag.get(proof.rhs())
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
                    ctx,
                    rule,
                    substitution,
                    proof.proposition(),
                    name,
                )?;

                Ok(Proposition::new(proof.lhs(), proof.rhs()))
            }

            Justification::MergeFn {
                function,
                old_proof,
                new_proof,
            } => {
                // Check both sub-proofs - they should be reflexive proofs
                let old_prop = self.check_proof_with_context(*old_proof, program, ctx)?;
                let new_prop = self.check_proof_with_context(*new_proof, program, ctx)?;

                let (_old_lhs, old_rhs) = (old_prop.lhs, old_prop.rhs);
                let (_new_lhs, new_rhs) = (new_prop.lhs, new_prop.rhs);
                let old_view_term = self.term_dag.get(old_rhs);
                let new_view_term = self.term_dag.get(new_rhs);

                let (old_term, new_term, view_head, input_args) = match (
                    old_view_term,
                    new_view_term,
                ) {
                    (Term::App(old_head, old_args), Term::App(new_head, new_args)) => {
                        // Verify both are views of the same function
                        if old_head != new_head {
                            return Err(ProofCheckError::MergeFnError {
                                proof_id,
                                reason: format!(
                                    "old and new proofs should be for the same function, but got {} and {}",
                                    old_head, new_head
                                ),
                            });
                        }
                        // The last argument is the output
                        let old_output =
                            *old_args
                                .last()
                                .ok_or_else(|| ProofCheckError::MergeFnError {
                                    proof_id,
                                    reason: "old view term has no arguments".to_string(),
                                })?;
                        let new_output =
                            *new_args
                                .last()
                                .ok_or_else(|| ProofCheckError::MergeFnError {
                                    proof_id,
                                    reason: "new view term has no arguments".to_string(),
                                })?;
                        // Get the input arguments (all but the last)
                        let inputs: Vec<TermId> = old_args[..old_args.len() - 1].to_vec();
                        // inputs should match for old and new
                        if inputs.len() != new_args.len() - 1
                            || inputs
                                .iter()
                                .zip(new_args[..new_args.len() - 1].iter())
                                .any(|(a, b)| a != b)
                        {
                            return Err(ProofCheckError::MergeFnError {
                                proof_id,
                                reason: "old and new view terms have different input arguments"
                                    .to_string(),
                            });
                        }

                        (old_output, new_output, old_head.clone(), inputs)
                    }
                    _ => {
                        return Err(ProofCheckError::MergeFnError {
                            proof_id,
                            reason: format!(
                                "expected function application terms, got {:?} and {:?}",
                                old_view_term, new_view_term
                            ),
                        });
                    }
                };

                // Run the merge function to get the expected result
                let (merged_term_child, mut merged_props) =
                    run_merge(&mut self.term_dag, function, program, old_term, new_term)?;
                // Add f(inputs..., merged_term) to merged_props
                let mut merged_view_args = input_args.clone();
                merged_view_args.push(merged_term_child);
                let merged_term = self.term_dag.app(view_head, merged_view_args);
                merged_props.insert(Proposition::new(merged_term, merged_term));
                // Verify the proof's claimed equality is in the merged propositions
                if !merged_props.contains(&Proposition::new(proof.lhs(), proof.rhs())) {
                    return Err(ProofCheckError::MergeFnError {
                        proof_id,
                        reason: format!(
                            "proof claims {} = {}, which is not established by merge function",
                            self.term_dag.to_string_with_let(
                                &mut SymbolGen::new("".to_string()),
                                proof.lhs()
                            ),
                            self.term_dag.to_string_with_let(
                                &mut SymbolGen::new("".to_string()),
                                proof.rhs()
                            )
                        ),
                    });
                }

                Ok(Proposition::new(proof.lhs(), proof.rhs()))
            }

            Justification::Trans(left_id, right_id) => {
                // Check both sub-proofs
                let left_prop = self.check_proof_with_context(*left_id, program, ctx)?;
                let right_prop = self.check_proof_with_context(*right_id, program, ctx)?;

                let (left_lhs, left_rhs) = (left_prop.lhs, left_prop.rhs);
                let (right_lhs, right_rhs) = (right_prop.lhs, right_prop.rhs);

                // Check transitivity: left.rhs must equal right.lhs
                if left_rhs != right_lhs {
                    return Err(ProofCheckError::TransitivityMismatch {
                        proof_id,
                        left_rhs,
                        right_lhs,
                    });
                }

                // Result should be left_lhs = right_rhs
                if proof.lhs() != left_lhs || proof.rhs() != right_rhs {
                    return Err(ProofCheckError::TermMismatch {
                        proof_id,
                        expected_lhs: proof.lhs(),
                        expected_rhs: proof.rhs(),
                        actual_lhs: left_lhs,
                        actual_rhs: right_rhs,
                    });
                }

                Ok(Proposition::new(proof.lhs(), proof.rhs()))
            }

            Justification::Sym(inner_id) => {
                // Check the inner proof
                let inner_prop = self.check_proof_with_context(*inner_id, program, ctx)?;
                let (inner_lhs, inner_rhs) = (inner_prop.lhs, inner_prop.rhs);

                // Symmetry swaps lhs and rhs
                if proof.lhs() != inner_rhs || proof.rhs() != inner_lhs {
                    return Err(ProofCheckError::TermMismatch {
                        proof_id,
                        expected_lhs: proof.lhs(),
                        expected_rhs: proof.rhs(),
                        actual_lhs: inner_rhs,
                        actual_rhs: inner_lhs,
                    });
                }

                Ok(Proposition::new(proof.lhs(), proof.rhs()))
            }

            Justification::Congr {
                proof: base_id,
                child_index,
                child_proof: child_id,
            } => {
                // Check the base proof (proves t1 = f(..., ci, ...))
                let base_prop = self.check_proof_with_context(*base_id, program, ctx)?;
                let (base_lhs, base_rhs) = (base_prop.lhs, base_prop.rhs);

                // Check the child proof (proves ci = c2)
                let child_prop = self.check_proof_with_context(*child_id, program, ctx)?;
                let (child_lhs, child_rhs) = (child_prop.lhs, child_prop.rhs);

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
                let expected_rhs_children: Vec<TermId> = children
                    .iter()
                    .enumerate()
                    .map(
                        |(i, &child)| {
                            if i == *child_index { child_rhs } else { child }
                        },
                    )
                    .collect();

                let expected_rhs_id = self.term_dag.app(func_name, expected_rhs_children);

                // Verify proof.rhs() matches expected
                if proof.rhs() != expected_rhs_id {
                    return Err(ProofCheckError::CongruenceMismatch {
                        proof_id,
                        reason: format!(
                            "Proof rhs {:?} doesn't match expected {:?}",
                            self.term_dag.get(proof.rhs()),
                            self.term_dag.get(expected_rhs_id)
                        ),
                    });
                }

                // Verify proof.lhs() matches base_lhs
                if proof.lhs() != base_lhs {
                    return Err(ProofCheckError::CongruenceMismatch {
                        proof_id,
                        reason: "Proof lhs doesn't match base proof lhs".to_string(),
                    });
                }

                Ok(Proposition::new(proof.lhs(), proof.rhs()))
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
    ) -> Result<(), ProofCheckError> {
        let (lhs, rhs) = (prop.lhs, prop.rhs);
        match fact {
            // proof normal form for functions: (= (f args...) v)
            // In the term representation, custom functions store output as last arg: f(args..., v)
            ResolvedFact::Eq(
                _,
                ResolvedExpr::Call(
                    _,
                    ResolvedCall::Func(FuncType {
                        subtype: FunctionSubtype::Custom,
                        name,
                        ..
                    }),
                    args,
                ),
                ResolvedExpr::Var(_, v),
            ) => {
                // Get the output variable's term
                let var_term = substitution.get(&v.name).copied().ok_or_else(|| {
                    ProofCheckError::RuleSubstitutionMismatch {
                        rule_name: rule_name.to_string(),
                        reason: format!("Variable {} not in substitution", v.name),
                    }
                })?;

                // Evaluate all the input arguments
                let mut arg_terms = Vec::new();
                for arg in args {
                    arg_terms.push(self.eval_expr_with_subst(rule_name, arg, substitution)?);
                }
                // Add the output variable as the last argument
                arg_terms.push(var_term);

                let expected_term_id = self.term_dag.app(name.clone(), arg_terms);

                // The proposition should be a reflexive equality for this term
                if lhs != expected_term_id || rhs != expected_term_id {
                    return Err(ProofCheckError::RuleSubstitutionMismatch {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Function fact does not match proposition: expected reflexive equality for {:?}, got {:?} = {:?}",
                            self.term_dag.get(expected_term_id),
                            self.term_dag.get(lhs),
                            self.term_dag.get(rhs)
                        ),
                    });
                }

                Ok(())
            }
            ResolvedFact::Eq(_, lhs_expr, rhs_expr) => {
                let fact_lhs = self.eval_expr_with_subst(rule_name, lhs_expr, substitution)?;
                let fact_rhs = self.eval_expr_with_subst(rule_name, rhs_expr, substitution)?;
                if fact_lhs != lhs || fact_rhs != rhs {
                    let fact_lhs_str = self.term_dag.to_string(fact_lhs);
                    let fact_rhs_str = self.term_dag.to_string(fact_rhs);
                    let subst_str = format_substitution(&self.term_dag, substitution);
                    let proof_str = self.proof_to_string(proof_id);
                    return Err(ProofCheckError::RuleSubstitutionMismatch {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Fact {fact} does not match proven proposition under substitution {subst_str}.\nSubstituted: (= {fact_lhs_str} {fact_rhs_str})\nPremise proves: (= {} {})\nProof: {proof_str}",
                            format_term(&self.term_dag, lhs),
                            format_term(&self.term_dag, rhs),
                        ),
                    });
                }

                Ok(())
            }
            // For a plain expr, the proof should have the form t1 = t2 where t2 matches the expr under substitution
            ResolvedFact::Fact(expr) => {
                let fact_term = self.eval_expr_with_subst(rule_name, expr, substitution)?;

                if fact_term != rhs {
                    return Err(ProofCheckError::RuleSubstitutionMismatch {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Fact {} does not match proposition under substitution {}. Got {}, expected {}.",
                            fact,
                            format_substitution(&self.term_dag, substitution),
                            format_term(&self.term_dag, rhs),
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
        rule_name: &str,
        expr: &ResolvedExpr,
        substitution: &HashMap<String, TermId>,
    ) -> Result<TermId, ProofCheckError> {
        match expr {
            ResolvedExpr::Lit(_, lit) => Ok(self.term_dag.lit(lit.clone())),
            ResolvedExpr::Var(_, var) => substitution.get(&var.name).copied().ok_or_else(|| {
                ProofCheckError::RuleSubstitutionMismatch {
                    rule_name: rule_name.to_string(),
                    reason: format!(
                        "Could not find variable '{}' in substitution. Available variables: {}",
                        var.name,
                        substitution
                            .keys()
                            .map(|k| k.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                }
            }),
            ResolvedExpr::Call(_, head, args) => {
                // Evaluate all arguments first
                let mut arg_terms = Vec::new();
                for arg in args {
                    arg_terms.push(self.eval_expr_with_subst(rule_name, arg, substitution)?);
                }

                match head {
                    ResolvedCall::Primitive(prim) => {
                        // Use the validator to compute the primitive result
                        if let Some(validator) = prim.validator() {
                            let result =
                                validator(&mut self.term_dag, &arg_terms).ok_or_else(|| {
                                    ProofCheckError::PrimitiveValidationFailed {
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
                                function_name: prim.name().to_string(),
                                reason: "Primitive has no validator - cannot verify in proof"
                                    .to_string(),
                            })
                        }
                    }
                    ResolvedCall::Func(func) => {
                        match func.subtype {
                            FunctionSubtype::Constructor => {
                                Ok(self.term_dag.app(func.name.clone(), arg_terms))
                            }
                            FunctionSubtype::Custom => {
                                // Custom functions should not appear in proof normal form!
                                // They should be in the form (= (f args...) v) in the rule body
                                panic!(
                                    "Custom function {} should not appear in expression evaluation during proof checking. \
                                    Functions should be in proof normal form: (= (function args...) output_var)",
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
        ctx: &ProofCheckContext,
        rule: &crate::ast::GenericRule<ResolvedCall, crate::ast::ResolvedVar>,
        substitution: &HashMap<String, TermId>,
        claimed: &Proposition,
        rule_name: &str,
    ) -> Result<(), ProofCheckError> {
        // Use process_actions to get propositions from the rule head
        // Note: process_actions expects global variable bindings, but substitution
        // has the same structure, so we can pass it directly
        let action_refs: Vec<&GenericAction<ResolvedCall, crate::ast::ResolvedVar>> =
            rule.head.0.iter().collect();
        let mut bindings = ctx.global_bindings.clone();
        bindings.extend(substitution.clone());
        let action_ctx = process_actions(rule_name, bindings, &action_refs, &mut self.term_dag)?;

        // Check if the claimed equality is in the propositions
        if action_ctx.propositions.contains(claimed) {
            return Ok(());
        }

        Err(ProofCheckError::RuleSubstitutionMismatch {
            rule_name: rule_name.to_string(),
            reason: format!(
                "Rule head doesn't produce claimed equality Lhs:\n{}\nRHS:\n{}\nSUBST:\n{}",
                format_term(&self.term_dag, claimed.lhs()),
                format_term(&self.term_dag, claimed.rhs()),
                format_substitution(&self.term_dag, substitution)
            ),
        })
    }
}
