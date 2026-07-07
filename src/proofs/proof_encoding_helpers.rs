//! Proof encoding helper functions that handle
//! naming, headers, and checking whether a program supports proof encoding.

use std::path::Path;

use crate::{
    EGraph, TypeInfo,
    ast::{
        Command, Expr, Fact, GenericCommand, ResolvedAction, ResolvedCommand, ResolvedExpr,
        ResolvedExprExt, ResolvedFact, Schedule,
    },
    core::ResolvedCall,
    proofs::proof_encoding::ProofInstrumentor,
    util::{FreshGen, HashMap, SymbolGen},
};

/// Holds all the names used in proof encoding.
/// We need fresh names that don't collide with user-defined names.
/// All of these names should be generated with the single global [`SymbolGen`].
#[derive(Clone)]
pub(crate) struct EncodingNames {
    pub(crate) proof_list_sort: String,
    pub(crate) ast_sort: String,
    pub(crate) proof_datatype: String,
    pub(crate) fiat_constructor: String,
    pub(crate) rule_constructor: String,
    pub(crate) merge_fn_constructor: String,
    pub(crate) eq_trans_constructor: String,
    pub(crate) eq_sym_constructor: String,
    pub(crate) congr_constructor: String,
    pub(crate) container_normalize_constructor: String,
    pub(crate) eval_constructor: String,
    /// For a given function symbol, the name of the function that converts to the AST type.
    pub(crate) sort_to_ast_constructor: HashMap<String, String>,
    pub(crate) fn_to_term_sort: HashMap<String, String>,
    pub(crate) single_parent_ruleset_name: String,
    pub(crate) uf_function_index_ruleset_name: String,
    pub(crate) pcons: String,
    pub(crate) pnil: String,
    // Ruleset names
    pub(crate) path_compress_ruleset_name: String,
    pub(crate) rebuilding_ruleset_name: String,
    pub(crate) rebuilding_cleanup_ruleset_name: String,
    pub(crate) delete_subsume_ruleset_name: String,
    // Per-function fresh names
    pub(crate) view_name: HashMap<String, String>,
    pub(crate) to_delete_name: HashMap<String, String>,
    pub(crate) subsumed_name: HashMap<String, String>,
    pub(crate) term_proof_name: HashMap<String, String>,
}

/// Packages proof information for instrumenting actions.
/// We may not know yet what terms we are instrumenting, so all but Proof leave that information to be filled in later.
/// This is only used internally in this file, it's not part of the proof format.
pub(crate) enum Justification {
    Rule(String, String), // rule name and proof list
    Fiat,
    Proof(String),                 // existing proof
    Merge(String, String, String), // function name, proof1, proof2
}

impl EncodingNames {
    pub(crate) fn new(symbol_gen: &mut SymbolGen) -> Self {
        Self {
            proof_list_sort: symbol_gen.fresh("ProofList"),
            ast_sort: symbol_gen.fresh("Ast"),
            proof_datatype: symbol_gen.fresh("Proof"),
            fiat_constructor: symbol_gen.fresh("Fiat"),
            rule_constructor: symbol_gen.fresh("Rule"),
            merge_fn_constructor: symbol_gen.fresh("Merge"),
            eq_trans_constructor: symbol_gen.fresh("Trans"),
            eq_sym_constructor: symbol_gen.fresh("Sym"),
            congr_constructor: symbol_gen.fresh("Congr"),
            container_normalize_constructor: symbol_gen.fresh("ContainerNormalize"),
            eval_constructor: symbol_gen.fresh("Eval"),
            sort_to_ast_constructor: HashMap::default(),
            fn_to_term_sort: HashMap::default(),
            single_parent_ruleset_name: symbol_gen.fresh("single_parent"),
            uf_function_index_ruleset_name: symbol_gen.fresh("uf_function_index"),
            pcons: symbol_gen.fresh("PCons"),
            pnil: symbol_gen.fresh("PNil"),
            path_compress_ruleset_name: symbol_gen.fresh("parent"),
            rebuilding_ruleset_name: symbol_gen.fresh("rebuilding"),
            rebuilding_cleanup_ruleset_name: symbol_gen.fresh("rebuilding_cleanup"),
            delete_subsume_ruleset_name: symbol_gen.fresh("delete_subsume_ruleset"),
            view_name: HashMap::default(),
            to_delete_name: HashMap::default(),
            subsumed_name: HashMap::default(),
            term_proof_name: HashMap::default(),
        }
    }
}

impl ProofInstrumentor<'_> {
    pub(crate) fn uf_name(&mut self, sort: &str) -> String {
        if let Some(name) = self.egraph.proof_state.uf_parent.get(sort) {
            name.clone()
        } else {
            let fresh_name = self.egraph.parser.symbol_gen.fresh(&format!("UF_{sort}"));
            self.egraph
                .proof_state
                .uf_parent
                .insert(sort.to_string(), fresh_name.clone());
            fresh_name
        }
    }

    pub(crate) fn uf_function_name(&mut self, sort: &str) -> String {
        if let Some(name) = self.egraph.proof_state.uf_function.get(sort) {
            name.clone()
        } else {
            let fresh_name = self.egraph.parser.symbol_gen.fresh(&format!("UF_{sort}f"));
            self.egraph
                .proof_state
                .uf_function
                .insert(sort.to_string(), fresh_name.clone());
            fresh_name
        }
    }

    /// Returns the name of the Pair sort used to bundle (leader, proof) in the UF function index.
    /// Only used in proof mode.
    pub(crate) fn uf_pair_sort_name(&mut self, sort: &str) -> String {
        self.egraph
            .parser
            .symbol_gen
            .fresh(&format!("UFPair_{sort}"))
    }

    pub(crate) fn parse_program(&mut self, input: &str) -> Vec<Command> {
        self.egraph.parser.ensure_no_reserved_symbols = false;
        let res = self.egraph.parser.get_program_from_string(None, input);
        self.egraph.parser.ensure_no_reserved_symbols = true;

        res.unwrap()
    }

    pub(crate) fn format_prooflist(&self, proofs: &[String]) -> String {
        let pcons = &self.proof_names().pcons;
        let pnil = &self.proof_names().pnil;

        let mut prooflist = format!("({pnil})");
        for proof in proofs.iter().rev() {
            prooflist = format!("({pcons} {proof} {prooflist})");
        }
        prooflist
    }

    /// Header commands for term encoding, setting up rulesets.
    pub(crate) fn term_header(&mut self) -> Vec<Command> {
        let str = format!(
            "(ruleset {})
             (ruleset {})
             (ruleset {})
             (ruleset {})
             (ruleset {})
             (ruleset {})",
            self.proof_names().path_compress_ruleset_name,
            self.proof_names().single_parent_ruleset_name,
            self.proof_names().uf_function_index_ruleset_name,
            self.proof_names().rebuilding_ruleset_name,
            self.proof_names().rebuilding_cleanup_ruleset_name,
            self.proof_names().delete_subsume_ruleset_name
        );
        self.parse_program(&str)
    }

    /// Internal parse helper for term encoding- parse and crash on failure.
    pub(crate) fn parse_schedule(&mut self, input: String) -> Schedule {
        self.egraph.parser.ensure_no_reserved_symbols = false;
        let res = self.egraph.parser.get_schedule_from_string(None, &input);
        self.egraph.parser.ensure_no_reserved_symbols = true;
        res.unwrap()
    }

    /// Internal parse helper for term encoding- parse and crash on failure.
    pub(crate) fn parse_facts(&mut self, input: &[String]) -> Vec<Fact> {
        self.egraph.parser.ensure_no_reserved_symbols = false;
        let res = input
            .iter()
            .map(|f| self.egraph.parser.get_fact_from_string(None, f).unwrap())
            .collect();
        self.egraph.parser.ensure_no_reserved_symbols = true;
        res
    }

    /// Internal parse helper for term encoding- parse an expression and crash on failure.
    pub(crate) fn parse_expr(&mut self, input: &str) -> Expr {
        self.egraph.parser.ensure_no_reserved_symbols = false;
        let res = self.egraph.parser.get_expr_from_string(None, input);
        self.egraph.parser.ensure_no_reserved_symbols = true;
        res.unwrap()
    }

    // Each function/constructor gets a view table, the canonicalized e-nodes to accelerate e-matching.
    pub(crate) fn view_name(&mut self, name: &str) -> String {
        if let Some(n) = self.egraph.proof_state.proof_names.view_name.get(name) {
            n.clone()
        } else {
            let fresh_name = self.egraph.parser.symbol_gen.fresh(&format!("{name}View"));
            self.egraph
                .proof_state
                .proof_names
                .view_name
                .insert(name.to_string(), fresh_name.clone());
            fresh_name
        }
    }

    pub(crate) fn delete_name(&mut self, name: &str) -> String {
        if let Some(n) = self.egraph.proof_state.proof_names.to_delete_name.get(name) {
            n.clone()
        } else {
            let fresh_name = self
                .egraph
                .parser
                .symbol_gen
                .fresh(&format!("to_delete_{name}"));
            self.egraph
                .proof_state
                .proof_names
                .to_delete_name
                .insert(name.to_string(), fresh_name.clone());
            fresh_name
        }
    }

    pub(crate) fn subsumed_name(&mut self, name: &str) -> String {
        if let Some(n) = self.egraph.proof_state.proof_names.subsumed_name.get(name) {
            n.clone()
        } else {
            let fresh_name = self
                .egraph
                .parser
                .symbol_gen
                .fresh(&format!("to_subsume_{name}"));
            self.egraph
                .proof_state
                .proof_names
                .subsumed_name
                .insert(name.to_string(), fresh_name.clone());
            fresh_name
        }
    }

    pub(crate) fn proof_names(&self) -> &EncodingNames {
        &self.egraph.proof_state.proof_names
    }

    pub(crate) fn proofs_enabled(&self) -> bool {
        self.egraph.proof_state.proofs_enabled
    }

    /// Returns the proof output type: `Proof` when proofs are enabled, `Unit` otherwise.
    pub(crate) fn proof_type_str(&self) -> &str {
        if self.proofs_enabled() {
            &self.proof_names().proof_datatype
        } else {
            "Unit"
        }
    }

    /// Returns code for a constructor that converts from sort to AST.
    /// Adds to the sort to AST constructor map.
    pub(crate) fn add_to_ast(&mut self, sort: &str) -> String {
        if self.proofs_enabled() {
            // Check if we've already created an AST constructor for this sort
            if self
                .egraph
                .proof_state
                .proof_names
                .sort_to_ast_constructor
                .contains_key(sort)
            {
                // Return empty string since the constructor already exists
                return "".to_string();
            }

            let to_ast_constructor = self.egraph.parser.symbol_gen.fresh(&format!("Ast{sort}"));
            self.egraph
                .proof_state
                .proof_names
                .sort_to_ast_constructor
                .insert(sort.to_string(), to_ast_constructor.clone());
            let ast_sort = &self.proof_names().ast_sort;
            format!("(constructor {to_ast_constructor} ({sort}) {ast_sort} :internal-hidden)")
        } else {
            "".to_string()
        }
    }

    /// Given a function name, returns the name of the AST constructor for that function's sort.
    pub(crate) fn fname_to_ast_name(&self, fname: &str) -> &str {
        let fn_sort = self
            .proof_names()
            .fn_to_term_sort
            .get(fname)
            .unwrap_or_else(|| panic!("Function {fname} has no recorded sort"))
            .clone();
        self.proof_names()
            .sort_to_ast_constructor
            .get(&fn_sort)
            .unwrap_or_else(|| {
                panic!("Function {fname}'s sort {fn_sort} has no recorded AST constructor")
            })
    }

    pub(crate) fn term_proof_name(&mut self, name: &str) -> String {
        if let Some(n) = self
            .egraph
            .proof_state
            .proof_names
            .term_proof_name
            .get(name)
        {
            n.clone()
        } else {
            let fresh_name = self.egraph.parser.symbol_gen.fresh(&format!("{name}Proof"));
            self.egraph
                .proof_state
                .proof_names
                .term_proof_name
                .insert(name.to_string(), fresh_name.clone());
            fresh_name
        }
    }

    pub(crate) fn fresh_var(&mut self) -> String {
        self.egraph.parser.symbol_gen.fresh("v")
    }

    /// Header string for proof encoding, defining sorts and constructors.
    /// Correspondings to [`RawProof`] in the Rust code.
    pub(crate) fn proof_header(&mut self) -> String {
        let mut to_ast_constructors = Vec::new();
        // need to build a Ast{lit} for each lit sort in self
        for sort_name in self.egraph.type_info.sorts.keys().clone() {
            if !self
                .proof_names()
                .sort_to_ast_constructor
                .contains_key(sort_name)
            {
                let ast_constructor = self
                    .egraph
                    .parser
                    .symbol_gen
                    .fresh(&format!("Ast{sort_name}"));
                self.egraph
                    .proof_state
                    .proof_names
                    .sort_to_ast_constructor
                    .insert(sort_name.clone(), ast_constructor.clone());
                to_ast_constructors.push(format!(
                    "(constructor {ast_constructor} ({sort_name} ) {} :internal-hidden)",
                    self.proof_names().ast_sort
                ));
            }
        }
        let to_ast_str = to_ast_constructors.join("\n");

        let EncodingNames {
            ref proof_list_sort,
            ref ast_sort,
            ref proof_datatype,
            ref fiat_constructor,
            ref rule_constructor,
            ref merge_fn_constructor,
            ref eq_trans_constructor,
            ref eq_sym_constructor,
            ref congr_constructor,
            ref container_normalize_constructor,
            ref eval_constructor,
            ref pcons,
            ref pnil,
            ..
        } = *self.proof_names();

        format!(
            "
(sort {proof_list_sort})
(sort {ast_sort}) ;; wrap sorts in this for proofs
;; The proof datatype records the global proof constructor names so container
;; rebuild can recover them on re-parse (see ContainerRebuildSpec).
(sort {proof_datatype} :internal-proof-names {congr_constructor} {eq_trans_constructor} {eq_sym_constructor} {container_normalize_constructor})

(constructor {pcons} ({proof_datatype} {proof_list_sort}) {proof_list_sort} :internal-hidden)
(constructor {pnil} () {proof_list_sort} :internal-hidden)

{to_ast_str}

;; Fiat justification for globals and primitives, gives two terms t1 = t2 for the proposition being justified
(constructor {fiat_constructor} ({ast_sort} {ast_sort}) {proof_datatype} :internal-hidden)
;; name of rule, one proof per fact in the query, proposition being proven t1 = t2
(constructor {rule_constructor} (String {proof_list_sort} {ast_sort} {ast_sort}) {proof_datatype} :internal-hidden)

;; merge function justification- name of function and two proofs for the two terms being merged,
;; and the proposition being justified t = t
(constructor {merge_fn_constructor} (String {proof_datatype} {proof_datatype} {ast_sort}) {proof_datatype} :internal-hidden)

;; transitivity of equality proofs
(constructor {eq_trans_constructor} ({proof_datatype} {proof_datatype}) {proof_datatype} :internal-hidden)

;; symmetry of equality proofs
(constructor  {eq_sym_constructor} ({proof_datatype}) {proof_datatype} :internal-hidden)
;; given a proof that t1 = f(..., ci, ...)
;; and the child index i of ci in the term f(..., ci, ...)
;; and a proof that ci = c2,
;; produces a justification that t1 = f(..., c2, ...)
(constructor  {congr_constructor} ({proof_datatype} i64 {proof_datatype}) {proof_datatype} :internal-hidden)

;; given a proof that t1 = c, where c is a container term, produces a proof that
;; t1 = normalize(c) (the container's canonicalization: sort/dedup for sets,
;; last-write-wins for maps, sort for multisets)
(constructor  {container_normalize_constructor} ({proof_datatype}) {proof_datatype} :internal-hidden)

;; marks the proof of a container side condition. Carries nothing: the side
;; condition is re-evaluated against the rule body when checked.
(constructor  {eval_constructor} () {proof_datatype} :internal-hidden)
                "
        )
    }
}

/// Reads a file and checks that its commands support the proof encoding.
pub fn file_supports_proofs(path: &Path) -> bool {
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(_) => return false,
    };

    let canonical = match std::fs::canonicalize(path) {
        Ok(canonical) => canonical,
        Err(_) => return false,
    };

    let mut egraph = EGraph::default();
    let filename = canonical.to_string_lossy().into_owned();
    let desugared = match egraph.resolve_program(Some(filename.clone()), &contents) {
        Ok(commands) => commands,
        Err(_) => return false,
    };

    program_supports_proofs(&desugared, &egraph.type_info)
}

/// Reasons why a command doesn't support proof encoding
#[derive(Debug, Clone, thiserror::Error)]
pub enum ProofEncodingUnsupportedReason {
    #[error("primitive operation lacks a validator function")]
    PrimitiveWithoutValidator,
    #[error(
        "action contains a function lookup. Finding the output of a function is only supported in queries."
    )]
    FunctionLookupInAction,
    #[error(
        "a container constructed in the query (a container-producing primitive result) is used in the actions. A query-built container is a side condition with no carryable proof, so it cannot be carried into an action."
    )]
    ContainerCreatedInQueryUsedInAction,
    #[error(
        "sort has a presort (custom sort container implementation). Custom sorts are not supported by proof encoding."
    )]
    SortWithPresort,
    #[error(
        "sort has a :internal-uf annotation. The :internal-uf annotation is used internally by term encoding and cannot be specified manually in proof mode."
    )]
    SortWithUfAnnotation,
    #[error(
        "sort has a :internal-proof-func annotation. The :internal-proof-func annotation is used internally by proof encoding and cannot be specified manually in proof mode."
    )]
    SortWithProofFuncAnnotation,
    #[error("user-defined commands are not supported.")]
    UserDefinedCommand,
    #[error("input commands are not supported.")]
    InputCommand,
    #[error("missing merge function. All functions need to specify a :merge function.")]
    NoMergeOnNonGlobalFunction,
    #[error(
        "let binding with a primitive in the body. For silly internal reasons, we don't support primitive bindings for proofs at the moment, sorry."
    )]
    LetBindingWithNonEqSort,
    #[error(
        "rule uses `:unsafe-seminaive`. Arbitrary RHS database reads are not representable by the term/proof encoding."
    )]
    UnsafeSeminaive,
    #[error(
        "rule uses `:naive` with an eq-sort primitive in the body. Proof encoding can only look up proofs for primitive eq-sort fact results under seminaive-safe query evaluation."
    )]
    NaiveEqSortPrimitiveFact,
}

/// Checks whether a desugared program supports proof encoding.
pub fn program_supports_proofs(commands: &[ResolvedCommand], type_info: &TypeInfo) -> bool {
    for command in commands {
        if command_supports_proof_encoding(command, type_info).is_err() {
            return false;
        }
    }
    true
}

/// Recursively check if all primitives in an expression have validators
fn expr_primitives_have_validators(expr: &ResolvedExpr) -> bool {
    use crate::ast::GenericExpr;
    use crate::core::ResolvedCall;

    let mut all_valid = true;
    expr.walk(
        &mut |e| {
            if let GenericExpr::Call(_, ResolvedCall::Primitive(prim), _) = e
                && prim.validator().is_none()
            {
                all_valid = false;
            }
        },
        &mut |_| {},
    );
    all_valid
}

/// Check if an action contains non-global function lookups in any of its expressions
fn action_has_function_lookup(action: &ResolvedAction, type_info: &TypeInfo) -> bool {
    let mut has_lookup = false;
    action.clone().visit_exprs(&mut |expr| {
        if type_info.expr_has_function_lookup(&expr).is_some() {
            has_lookup = true;
        }
        expr
    });
    has_lookup
}

/// Check if a fact contains a primitive expression whose result needs a stored term proof.
fn fact_has_eq_sort_primitive_result(fact: &ResolvedFact) -> bool {
    let mut has_eq_sort_primitive = false;
    fact.clone().visit_exprs(&mut |expr| {
        if let ResolvedExpr::Call(_, ResolvedCall::Primitive(prim), _) = &expr
            && (prim.output().is_eq_sort() || prim.output().is_eq_container_sort())
        {
            has_eq_sort_primitive = true;
        }
        expr
    });
    has_eq_sort_primitive
}

/// Checks whether a resolved command supports proof encoding.
/// Returns Ok(()) if supported, or Err with the reason if not.
pub(crate) fn command_supports_proof_encoding(
    command: &ResolvedCommand,
    type_info: &TypeInfo,
) -> Result<(), ProofEncodingUnsupportedReason> {
    // `:unsafe-seminaive` rules perform arbitrary reads against the live
    // database; the term/proof encoding can't represent that.
    if let crate::ast::GenericCommand::Rule { rule } = command
        && rule.eval_mode == crate::ast::RuleEvalMode::UnsafeSeminaive
    {
        return Err(ProofEncodingUnsupportedReason::UnsafeSeminaive);
    }
    if let crate::ast::GenericCommand::Rule { rule } = command
        && rule.eval_mode == crate::ast::RuleEvalMode::Naive
        && rule.body.iter().any(fact_has_eq_sort_primitive_result)
    {
        return Err(ProofEncodingUnsupportedReason::NaiveEqSortPrimitiveFact);
    }
    // Check all expressions for primitives without validators
    let mut all_primitives_have_validators = true;
    command.clone().visit_exprs(&mut |expr| {
        if !expr_primitives_have_validators(&expr) {
            all_primitives_have_validators = false;
        }
        expr
    });

    if !all_primitives_have_validators {
        return Err(ProofEncodingUnsupportedReason::PrimitiveWithoutValidator);
    }

    // Check actions (not queries) for function lookups
    // Egglog supports lookups in actions at the global level, but not in proofs mode
    // (global function calls are allowed - they get desugared to constructors)
    let mut has_function_lookup_in_action = false;
    command.clone().visit_actions(&mut |action| {
        has_function_lookup_in_action |= action_has_function_lookup(&action, type_info);
        action
    });

    if has_function_lookup_in_action {
        return Err(ProofEncodingUnsupportedReason::FunctionLookupInAction);
    }

    // A container built by a primitive in the query is a side condition with no
    // carryable proof, so it can't be used in an action. Reject a rule that binds
    // such a container to a variable used in its actions.
    if let GenericCommand::Rule { rule } = command {
        let mut constructed: Vec<String> = Vec::new();
        for fact in &rule.body {
            if let ResolvedFact::Eq(_, lhs, rhs) = fact {
                for (var_side, call_side) in [(lhs, rhs), (rhs, lhs)] {
                    if let ResolvedExpr::Var(_, v) = var_side
                        && let ResolvedExpr::Call(_, ResolvedCall::Primitive(prim), _) = call_side
                        && prim.output().is_eq_container_sort()
                    {
                        constructed.push(v.name.clone());
                    }
                }
            }
        }
        if !constructed.is_empty() {
            let mut used_in_action = false;
            for action in &rule.head.0 {
                action.clone().visit_exprs(&mut |expr| {
                    expr.walk(
                        &mut |e| {
                            if let ResolvedExpr::Var(_, v) = e
                                && constructed.contains(&v.name)
                            {
                                used_in_action = true;
                            }
                        },
                        &mut |_| {},
                    );
                    expr
                });
            }
            if used_in_action {
                return Err(ProofEncodingUnsupportedReason::ContainerCreatedInQueryUsedInAction);
            }
        }
    }

    // Now check command-specific constraints
    match command {
        GenericCommand::Sort {
            name,
            presort_and_args: Some(_),
            ..
        } => type_info
            .get_sort_by_name(name)
            .filter(|sort| sort.is_container_sort())
            .map(|_| ())
            .ok_or(ProofEncodingUnsupportedReason::SortWithPresort),
        GenericCommand::Sort { uf: Some(_), .. } => {
            Err(ProofEncodingUnsupportedReason::SortWithUfAnnotation)
        }
        GenericCommand::Sort {
            proof_func: Some(_),
            ..
        } => Err(ProofEncodingUnsupportedReason::SortWithProofFuncAnnotation),
        GenericCommand::UserDefined(..) => Err(ProofEncodingUnsupportedReason::UserDefinedCommand),
        GenericCommand::Input { .. } => Err(ProofEncodingUnsupportedReason::InputCommand),
        // Extract commands can't have non-global function lookups
        // because instrument_action_expr doesn't support them
        // (global function calls are fine - they get desugared to constructors)
        GenericCommand::Extract(_, expr, variants) => {
            if type_info.expr_has_function_lookup(expr).is_some()
                || type_info.expr_has_function_lookup(variants).is_some()
            {
                Err(ProofEncodingUnsupportedReason::FunctionLookupInAction)
            } else {
                Ok(())
            }
        }
        // no-merge on a non-global function
        // To add support: https://github.com/egraphs-good/egglog/issues/774
        GenericCommand::Function {
            merge: None, name, ..
        } => {
            if type_info.is_global(name) {
                Ok(())
            } else {
                Err(ProofEncodingUnsupportedReason::NoMergeOnNonGlobalFunction)
            }
        }
        // let binding with non-eq sort not supported by proof_global_desugar
        ResolvedCommand::Action(ResolvedAction::Let(_, _, expr)) => {
            // let binding with non-eq sort not supported by proof_global_desugar
            // we detect as setting something that is no-merge to a primitive not supported (global primitive binding)
            if expr.output_type().is_eq_sort() {
                Ok(())
            } else {
                Err(ProofEncodingUnsupportedReason::LetBindingWithNonEqSort)
            }
        }
        // After global desugar it may look like this
        ResolvedCommand::Action(ResolvedAction::Set(_span, head, _children, expr)) => {
            if !type_info.is_global(head.name()) || expr.output_type().is_eq_sort() {
                Ok(())
            } else {
                Err(ProofEncodingUnsupportedReason::LetBindingWithNonEqSort)
            }
        }
        _ => Ok(()),
    }
}
