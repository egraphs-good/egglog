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
) -> impl Iterator<Item = &GenericAction<ResolvedCall, crate::ast::ResolvedVar>> {
    prog.iter().filter_map(|cmd| {
        if let GenericNCommand::CoreAction(action) = cmd {
            Some(action)
        } else {
            None
        }
    })
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
                let expr = func_decl.merge.as_ref().ok_or_else(|| {
                    ProofCheckError::from(ProofCheckErrorKind::FunctionNotFound {
                        function_name: func_name.to_string(),
                    })
                })?;
                return eval_expr_with_subst("merge_function", expr, term_dag, &subst);
            }
        }
    }
    Err(ProofCheckErrorKind::FunctionNotFound {
        function_name: func_name.to_string(),
    }
    .into())
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
        ResolvedExpr::Var(_, var) => subst.get(&var.name).copied().ok_or_else(|| {
            ProofCheckError::from(ProofCheckErrorKind::UnboundVariable {
                rule_name: rule_name.to_string(),
                variable: var.name.clone(),
                available: subst.keys().cloned().collect::<Vec<_>>().join(", "),
            })
        })?,
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
                validator(dag, &arg_terms).ok_or_else(|| {
                    ProofCheckError::from(ProofCheckErrorKind::PrimitiveValidatorFailed {
                        function_name: specialized_primitive.name().to_string(),
                    })
                })?
            }
        },
    };

    // Add reflexive equality for this term and all its subterms
    add_subterm_reflexive_equalities(term_id, dag, &mut propositions);

    Ok((term_id, propositions))
}

/// Add reflexive equalities for all subterms of a term.
/// For example, if we have proved `(f (g 1)) = (f (g 1))`, then we have also proved
// that `(g 1) = (g 1)` and `1 = 1`.
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
) -> Result<HashMap<String, TermId>, ProofCheckError> {
    let actions: Vec<_> = gather_global_actions(prog).collect();
    let ctx = process_actions("global_action", HashMap::default(), &actions, term_dag)?;
    Ok(ctx.var_bindings)
}

/// Errors that can occur during proof checking.
/// This is a boxed wrapper to keep the error type small.
#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub struct ProofCheckError(Box<ProofCheckErrorKind>);

impl ProofCheckError {
    /// Create a new proof check error
    fn new(kind: ProofCheckErrorKind) -> Self {
        ProofCheckError(Box::new(kind))
    }
}

impl From<ProofCheckErrorKind> for ProofCheckError {
    fn from(kind: ProofCheckErrorKind) -> Self {
        ProofCheckError::new(kind)
    }
}

/// The kinds of errors that can occur during proof checking
#[derive(Debug, Clone, Error)]
pub enum ProofCheckErrorKind {
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
    /// Congruence proof: base rhs is not a function application
    #[error("Proof {proof_id}: congruence error - base proof rhs is not a function application")]
    CongruenceBaseNotApp { proof_id: ProofId },
    /// Congruence proof: child index out of bounds
    #[error(
        "Proof {proof_id}: congruence error - child index {child_index} out of bounds for term with {num_children} children"
    )]
    CongruenceChildIndexOutOfBounds {
        proof_id: ProofId,
        child_index: usize,
        num_children: usize,
    },
    /// Congruence proof: child proof lhs doesn't match base term child
    #[error(
        "Proof {proof_id}: congruence error - child proof lhs {child_lhs:?} doesn't match base term child {base_child:?} at index {child_index}"
    )]
    CongruenceChildMismatch {
        proof_id: ProofId,
        child_lhs: TermId,
        base_child: TermId,
        child_index: usize,
    },
    /// Congruence proof: result doesn't match expected
    #[error(
        "Proof {proof_id}: congruence error - proof rhs {proof_rhs:?} doesn't match expected {expected_rhs:?}"
    )]
    CongruenceResultMismatch {
        proof_id: ProofId,
        proof_rhs: TermId,
        expected_rhs: TermId,
    },
    /// Congruence proof: lhs doesn't match base proof lhs
    #[error("Proof {proof_id}: congruence error - proof lhs doesn't match base proof lhs")]
    CongruenceLhsMismatch { proof_id: ProofId },
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
    /// Variable not found in substitution during proof checking
    #[error(
        "Rule '{rule_name}': variable '{variable}' not found in substitution. Available: {available}"
    )]
    UnboundVariable {
        rule_name: String,
        variable: String,
        available: String,
    },
    /// Function fact doesn't match the expected reflexive equality proposition
    #[error(
        "Rule '{rule_name}': function fact mismatch - expected reflexive equality for {expected}, got {actual_lhs} = {actual_rhs}"
    )]
    FunctionFactMismatch {
        rule_name: String,
        expected: String,
        actual_lhs: String,
        actual_rhs: String,
    },
    /// Equality fact doesn't match the proven proposition under substitution
    #[error(
        "Rule '{rule_name}': equality fact mismatch under substitution.\nFact: {fact}\nSubstituted: (= {substituted_lhs} {substituted_rhs})\nPremise proves: (= {proven_lhs} {proven_rhs})"
    )]
    EqualityFactMismatch {
        rule_name: String,
        fact: String,
        substituted_lhs: String,
        substituted_rhs: String,
        proven_lhs: String,
        proven_rhs: String,
    },
    /// Plain fact expression doesn't match proposition under substitution
    #[error(
        "Rule '{rule_name}': fact mismatch - {fact} under substitution {substitution} gives {actual}, expected {expected}"
    )]
    FactMismatch {
        rule_name: String,
        fact: String,
        substitution: String,
        actual: String,
        expected: String,
    },
    /// Rule head actions don't produce the claimed equality
    #[error(
        "Rule '{rule_name}': rule head doesn't produce claimed equality.\nLHS: {claimed_lhs}\nRHS: {claimed_rhs}\nSubstitution: {substitution}"
    )]
    RuleHeadMismatch {
        rule_name: String,
        claimed_lhs: String,
        claimed_rhs: String,
        substitution: String,
    },
    /// Primitive operation validator failed
    #[error("Primitive '{function_name}' validation failed")]
    PrimitiveValidatorFailed { function_name: String },
    /// Primitive has no validator for proof checking
    #[error("Primitive '{function_name}' has no validator - cannot verify in proof")]
    PrimitiveNoValidator { function_name: String },
    /// Could not find the rule referenced in a proof
    #[error("Could not find rule '{rule_name}'")]
    RuleNotFound { rule_name: String },
    /// Could not find the function referenced in a proof
    #[error("Could not find function '{function_name}'")]
    FunctionNotFound { function_name: String },
    /// Fiat proof claims equality not established by globals
    #[error(
        "Proof {proof_id}: Fiat proof claims {lhs:?} = {rhs:?}, which is not established by globals"
    )]
    InvalidFiat {
        proof_id: ProofId,
        lhs: TermId,
        rhs: TermId,
    },
    /// MergeFn proof: old and new proofs are for different functions
    #[error(
        "Proof {proof_id}: MergeFn error - old and new proofs should be for the same function, but got {old_func} and {new_func}"
    )]
    MergeFnFunctionMismatch {
        proof_id: ProofId,
        old_func: String,
        new_func: String,
    },
    /// MergeFn proof: view term has no arguments
    #[error("Proof {proof_id}: MergeFn error - {which} view term has no arguments")]
    MergeFnEmptyArgs { proof_id: ProofId, which: String },
    /// MergeFn proof: old and new view terms have different input arguments
    #[error(
        "Proof {proof_id}: MergeFn error - old and new view terms have different input arguments"
    )]
    MergeFnInputMismatch { proof_id: ProofId },
    /// MergeFn proof: expected function application terms
    #[error(
        "Proof {proof_id}: MergeFn error - expected function application terms, got {old_term:?} and {new_term:?}"
    )]
    MergeFnNotApp {
        proof_id: ProofId,
        old_term: TermId,
        new_term: TermId,
    },
    /// MergeFn proof: claimed equality not established by merge function
    #[error(
        "Proof {proof_id}: MergeFn error - proof claims {claimed_lhs} = {claimed_rhs}, which is not established by merge function"
    )]
    MergeFnResultMismatch {
        proof_id: ProofId,
        claimed_lhs: String,
        claimed_rhs: String,
    },
    /// MergeFn proof: sub-proof is not reflexive
    #[error(
        "Proof {proof_id}: MergeFn error - {which} proof is not reflexive, lhs {lhs:?} != rhs {rhs:?}"
    )]
    MergeFnNotReflexive {
        proof_id: ProofId,
        which: String,
        lhs: TermId,
        rhs: TermId,
    },
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
    fn new(prog: &[ResolvedNCommand], term_dag: &mut TermDag) -> Result<Self, ProofCheckError> {
        // Use the new refactored functions
        let actions: Vec<_> = gather_global_actions(prog).collect();
        let action_ctx = process_actions("global_actions", HashMap::default(), &actions, term_dag)?;

        Ok(ProofCheckContext {
            global_equalities: action_ctx.propositions,
            checked_proofs: HashMap::default(),
            global_bindings: action_ctx.var_bindings,
        })
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
        let mut ctx = ProofCheckContext::new(program, &mut self.term_dag)?;
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
                    Err(ProofCheckErrorKind::InvalidFiat {
                        proof_id,
                        lhs: proof.lhs(),
                        rhs: proof.rhs(),
                    }
                    .into())
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
                    .ok_or_else(|| {
                        ProofCheckError::from(ProofCheckErrorKind::RuleNotFound {
                            rule_name: name.clone(),
                        })
                    })?;

                // Check premise count
                if rule.body.len() != premise_proofs.len() {
                    return Err(ProofCheckErrorKind::RulePremiseCountMismatch {
                        proof_id,
                        rule_name: name.clone(),
                        expected: rule.body.len(),
                        actual: premise_proofs.len(),
                    }
                    .into());
                }

                // Check each premise proof
                let mut premise_propositions = Vec::new();
                for &premise_id in premise_proofs {
                    let prop = self.check_proof_with_context(premise_id, program, ctx)?;
                    premise_propositions.push(prop);
                }

                // Verify that premises match the rule body under the substitution
                for (fact, prop) in rule.body.iter().zip(premise_propositions.iter()) {
                    self.check_fact_matches_proposition(fact, prop, substitution, name)?;
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

                let (old_lhs, old_rhs) = (old_prop.lhs, old_prop.rhs);
                let (new_lhs, new_rhs) = (new_prop.lhs, new_prop.rhs);

                // MergeFn proofs expect reflexive equality proofs
                if old_lhs != old_rhs {
                    return Err(ProofCheckErrorKind::MergeFnNotReflexive {
                        proof_id,
                        which: "old".to_string(),
                        lhs: old_lhs,
                        rhs: old_rhs,
                    }
                    .into());
                }
                if new_lhs != new_rhs {
                    return Err(ProofCheckErrorKind::MergeFnNotReflexive {
                        proof_id,
                        which: "new".to_string(),
                        lhs: new_lhs,
                        rhs: new_rhs,
                    }
                    .into());
                }

                let old_view_term = self.term_dag.get(old_rhs);
                let new_view_term = self.term_dag.get(new_rhs);

                let (old_term, new_term, view_head, input_args) =
                    match (old_view_term.clone(), new_view_term.clone()) {
                        (Term::App(old_head, old_args), Term::App(new_head, new_args)) => {
                            // Verify both are views of the same function
                            if old_head != new_head {
                                return Err(ProofCheckErrorKind::MergeFnFunctionMismatch {
                                    proof_id,
                                    old_func: old_head.clone(),
                                    new_func: new_head.clone(),
                                }
                                .into());
                            }
                            // The last argument is the output
                            let old_output = *old_args.last().ok_or_else(|| {
                                ProofCheckError::from(ProofCheckErrorKind::MergeFnEmptyArgs {
                                    proof_id,
                                    which: "old".to_string(),
                                })
                            })?;
                            let new_output = *new_args.last().ok_or_else(|| {
                                ProofCheckError::from(ProofCheckErrorKind::MergeFnEmptyArgs {
                                    proof_id,
                                    which: "new".to_string(),
                                })
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
                                return Err(
                                    ProofCheckErrorKind::MergeFnInputMismatch { proof_id }.into()
                                );
                            }

                            (old_output, new_output, old_head.clone(), inputs)
                        }
                        _ => {
                            return Err(ProofCheckErrorKind::MergeFnNotApp {
                                proof_id,
                                old_term: old_rhs,
                                new_term: new_rhs,
                            }
                            .into());
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
                    return Err(ProofCheckErrorKind::MergeFnResultMismatch {
                        proof_id,
                        claimed_lhs: self
                            .term_dag
                            .to_string_with_let(&mut SymbolGen::new("".to_string()), proof.lhs()),
                        claimed_rhs: self
                            .term_dag
                            .to_string_with_let(&mut SymbolGen::new("".to_string()), proof.rhs()),
                    }
                    .into());
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
                    return Err(ProofCheckErrorKind::TransitivityMismatch {
                        proof_id,
                        left_rhs,
                        right_lhs,
                    }
                    .into());
                }

                // Result should be left_lhs = right_rhs
                if proof.lhs() != left_lhs || proof.rhs() != right_rhs {
                    return Err(ProofCheckErrorKind::TermMismatch {
                        proof_id,
                        expected_lhs: proof.lhs(),
                        expected_rhs: proof.rhs(),
                        actual_lhs: left_lhs,
                        actual_rhs: right_rhs,
                    }
                    .into());
                }

                Ok(Proposition::new(proof.lhs(), proof.rhs()))
            }

            Justification::Sym(inner_id) => {
                // Check the inner proof
                let inner_prop = self.check_proof_with_context(*inner_id, program, ctx)?;
                let (inner_lhs, inner_rhs) = (inner_prop.lhs, inner_prop.rhs);

                // Symmetry swaps lhs and rhs
                if proof.lhs() != inner_rhs || proof.rhs() != inner_lhs {
                    return Err(ProofCheckErrorKind::TermMismatch {
                        proof_id,
                        expected_lhs: proof.lhs(),
                        expected_rhs: proof.rhs(),
                        actual_lhs: inner_rhs,
                        actual_rhs: inner_lhs,
                    }
                    .into());
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
                        return Err(ProofCheckErrorKind::CongruenceBaseNotApp { proof_id }.into());
                    }
                };

                // Check child_index is valid
                if *child_index >= children.len() {
                    return Err(ProofCheckErrorKind::CongruenceChildIndexOutOfBounds {
                        proof_id,
                        child_index: *child_index,
                        num_children: children.len(),
                    }
                    .into());
                }

                // Check that child_lhs matches the child at child_index
                if children[*child_index] != child_lhs {
                    return Err(ProofCheckErrorKind::CongruenceChildMismatch {
                        proof_id,
                        child_lhs,
                        base_child: children[*child_index],
                        child_index: *child_index,
                    }
                    .into());
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
                    return Err(ProofCheckErrorKind::CongruenceResultMismatch {
                        proof_id,
                        proof_rhs: proof.rhs(),
                        expected_rhs: expected_rhs_id,
                    }
                    .into());
                }

                // Verify proof.lhs() matches base_lhs
                if proof.lhs() != base_lhs {
                    return Err(ProofCheckErrorKind::CongruenceLhsMismatch { proof_id }.into());
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
                    ProofCheckErrorKind::UnboundVariable {
                        rule_name: rule_name.to_string(),
                        variable: v.name.clone(),
                        available: substitution.keys().cloned().collect::<Vec<_>>().join(", "),
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
                    return Err(ProofCheckErrorKind::FunctionFactMismatch {
                        rule_name: rule_name.to_string(),
                        expected: format_term(&self.term_dag, expected_term_id),
                        actual_lhs: format_term(&self.term_dag, lhs),
                        actual_rhs: format_term(&self.term_dag, rhs),
                    }
                    .into());
                }

                Ok(())
            }
            ResolvedFact::Eq(_, lhs_expr, rhs_expr) => {
                let fact_lhs = self.eval_expr_with_subst(rule_name, lhs_expr, substitution)?;
                let fact_rhs = self.eval_expr_with_subst(rule_name, rhs_expr, substitution)?;
                if fact_lhs != lhs || fact_rhs != rhs {
                    return Err(ProofCheckErrorKind::EqualityFactMismatch {
                        rule_name: rule_name.to_string(),
                        fact: format!("{fact}"),
                        substituted_lhs: self.term_dag.to_string(fact_lhs),
                        substituted_rhs: self.term_dag.to_string(fact_rhs),
                        proven_lhs: format_term(&self.term_dag, lhs),
                        proven_rhs: format_term(&self.term_dag, rhs),
                    }
                    .into());
                }

                Ok(())
            }
            // For a plain expr, the proof should have the form t1 = t2 where t2 matches the expr under substitution
            ResolvedFact::Fact(expr) => {
                let fact_term = self.eval_expr_with_subst(rule_name, expr, substitution)?;

                if fact_term != rhs {
                    return Err(ProofCheckErrorKind::FactMismatch {
                        rule_name: rule_name.to_string(),
                        fact: format!("{fact}"),
                        substitution: format_substitution(&self.term_dag, substitution),
                        actual: format_term(&self.term_dag, rhs),
                        expected: format_term(&self.term_dag, fact_term),
                    }
                    .into());
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
                ProofCheckError::from(ProofCheckErrorKind::UnboundVariable {
                    rule_name: rule_name.to_string(),
                    variable: var.name.clone(),
                    available: substitution.keys().cloned().collect::<Vec<_>>().join(", "),
                })
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
                                    ProofCheckErrorKind::PrimitiveValidatorFailed {
                                        function_name: prim.name().to_string(),
                                    }
                                })?;
                            Ok(result)
                        } else {
                            // No validator available - primitives without validators can't be checked in proofs
                            Err(ProofCheckErrorKind::PrimitiveNoValidator {
                                function_name: prim.name().to_string(),
                            }
                            .into())
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

        Err(ProofCheckErrorKind::RuleHeadMismatch {
            rule_name: rule_name.to_string(),
            claimed_lhs: format_term(&self.term_dag, claimed.lhs()),
            claimed_rhs: format_term(&self.term_dag, claimed.rhs()),
            substitution: format_substitution(&self.term_dag, substitution),
        }
        .into())
    }
}
