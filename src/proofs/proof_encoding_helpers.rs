//! Proof encoding helper functions that handle
//! naming, headers, and checking whether a program supports proof encoding.

use std::path::Path;

use crate::{
    EGraph, TypeInfo,
    ast::{
        Command, Fact, GenericCommand, ResolvedAction, ResolvedCommand, ResolvedExpr,
        ResolvedExprExt, Schedule,
    },
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
    /// For a given function symbol, the name of the function that converts to the AST type.
    pub(crate) sort_to_ast_constructor: HashMap<String, String>,
    pub(crate) fn_to_term_sort: HashMap<String, String>,
    pub(crate) uf_proof_name: HashMap<String, String>,
    pub(crate) single_parent_ruleset_name: String,
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
    pub(crate) view_proof_name: HashMap<String, String>,
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
            sort_to_ast_constructor: HashMap::default(),
            fn_to_term_sort: HashMap::default(),
            uf_proof_name: HashMap::default(),
            single_parent_ruleset_name: symbol_gen.fresh("single_parent"),
            pcons: symbol_gen.fresh("PCons"),
            pnil: symbol_gen.fresh("PNil"),
            path_compress_ruleset_name: symbol_gen.fresh("parent"),
            rebuilding_ruleset_name: symbol_gen.fresh("rebuilding"),
            rebuilding_cleanup_ruleset_name: symbol_gen.fresh("rebuilding_cleanup"),
            delete_subsume_ruleset_name: symbol_gen.fresh("delete_subsume_ruleset"),
            view_name: HashMap::default(),
            to_delete_name: HashMap::default(),
            subsumed_name: HashMap::default(),
            view_proof_name: HashMap::default(),
            term_proof_name: HashMap::default(),
        }
    }
}

impl<'a> ProofInstrumentor<'a> {
    pub(crate) fn uf_name(&mut self, sort: &str) -> String {
        if let Some(name) = self.egraph.proof_state.uf_parent.get(sort) {
            name.clone()
        } else {
            let fresh_name = self.egraph.parser.symbol_gen.fresh(&format!("UF_{}", sort));
            self.egraph
                .proof_state
                .uf_parent
                .insert(sort.to_string(), fresh_name.clone());
            fresh_name
        }
    }

    /// A UF proof gives a proof of equality between two terms.
    /// Given two terms, gives a proof they are equal.
    pub(crate) fn uf_proof_name(&mut self, sort: &str) -> String {
        if let Some(name) = self.egraph.proof_state.proof_names.uf_proof_name.get(sort) {
            name.clone()
        } else {
            let fresh_name = self
                .egraph
                .parser
                .symbol_gen
                .fresh(&format!("{}UFProof", sort));
            self.egraph
                .proof_state
                .proof_names
                .uf_proof_name
                .insert(sort.to_string(), fresh_name.clone());
            fresh_name
        }
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
             (ruleset {})",
            self.proof_names().path_compress_ruleset_name,
            self.proof_names().single_parent_ruleset_name,
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

    // Each function/constructor gets a view table, the canonicalized e-nodes to accelerate e-matching.
    pub(crate) fn view_name(&mut self, name: &str) -> String {
        if let Some(n) = self.egraph.proof_state.proof_names.view_name.get(name) {
            n.clone()
        } else {
            let fresh_name = self
                .egraph
                .parser
                .symbol_gen
                .fresh(&format!("{}View", name));
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
                .fresh(&format!("to_delete_{}", name));
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
                .fresh(&format!("to_subsume_{}", name));
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

            let to_ast_constructor = self.egraph.parser.symbol_gen.fresh(&format!("Ast{}", sort));
            self.egraph
                .proof_state
                .proof_names
                .sort_to_ast_constructor
                .insert(sort.to_string(), to_ast_constructor.clone());
            let ast_sort = &self.proof_names().ast_sort;
            format!("(constructor {to_ast_constructor} ({sort}) {ast_sort})")
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
            .unwrap_or_else(|| panic!("Function {} has no recorded sort", fname))
            .clone();
        self.proof_names()
            .sort_to_ast_constructor
            .get(&fn_sort)
            .unwrap_or_else(|| {
                panic!(
                    "Function {}'s sort {} has no recorded AST constructor",
                    fname, fn_sort
                )
            })
    }

    /// A view proof for functions proves ... = t by some justification, where t is the term of the view row.
    /// A constructor view is more complex, representing f(c1, c2, c3, t_r) where t_r is a representative.
    /// A proof for a view proves that t_r = f(c1, c2, c3).
    pub(crate) fn view_proof_name(&mut self, name: &str) -> String {
        if let Some(n) = self
            .egraph
            .proof_state
            .proof_names
            .view_proof_name
            .get(name)
        {
            n.clone()
        } else {
            let fresh_name = self
                .egraph
                .parser
                .symbol_gen
                .fresh(&format!("{}ViewProof", name));
            self.egraph
                .proof_state
                .proof_names
                .view_proof_name
                .insert(name.to_string(), fresh_name.clone());
            fresh_name
        }
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
            let fresh_name = self
                .egraph
                .parser
                .symbol_gen
                .fresh(&format!("{}Proof", name));
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
                    .fresh(&format!("Ast{}", sort_name));
                self.egraph
                    .proof_state
                    .proof_names
                    .sort_to_ast_constructor
                    .insert(sort_name.clone(), ast_constructor.clone());
                to_ast_constructors.push(format!(
                    "(constructor {ast_constructor} ({sort_name} ) {})",
                    self.proof_names().ast_sort
                ));
            }
        }
        let to_ast_str = to_ast_constructors.join("\n");

        let (
            proof_list_sort,
            ast_sort,
            proof_datatype,
            fiat_constructor,
            rule_constructor,
            merge_fn_constructor,
            eq_trans_constructor,
            eq_sym_constructor,
            congr_constructor,
            pcons,
            pnil,
        ) = {
            let names = self.proof_names();
            (
                &names.proof_list_sort,
                &names.ast_sort,
                &names.proof_datatype,
                &names.fiat_constructor,
                &names.rule_constructor,
                &names.merge_fn_constructor,
                &names.eq_trans_constructor,
                &names.eq_sym_constructor,
                &names.congr_constructor,
                &names.pcons,
                &names.pnil,
            )
        };

        format!(
            "
(sort {proof_list_sort})
(sort {ast_sort}) ;; wrap sorts in this for proofs
(sort {proof_datatype})

(constructor {pcons} ({proof_datatype} {proof_list_sort}) {proof_list_sort})
(constructor {pnil} () {proof_list_sort})

{to_ast_str}

;; Fiat justification for globals and primitives, gives two terms t1 = t2 for the proposition being justified
(constructor {fiat_constructor} ({ast_sort} {ast_sort}) {proof_datatype})
;; name of rule, one proof per fact in the query, proposition being proven t1 = t2
(constructor {rule_constructor} (String {proof_list_sort} {ast_sort} {ast_sort}) {proof_datatype})

;; merge function justification- name of function and two proofs for the two terms being merged,
;; and the proposition being justified t = t
(constructor {merge_fn_constructor} (String {proof_datatype} {proof_datatype} {ast_sort}) {proof_datatype})

;; transitivity of equality proofs
(constructor {eq_trans_constructor} ({proof_datatype} {proof_datatype}) {proof_datatype})

;; symmetry of equality proofs
(constructor  {eq_sym_constructor} ({proof_datatype}) {proof_datatype})
;; given a proof that t1 = f(..., ci, ...)
;; and the child index i of ci in the term f(..., ci, ...)
;; and a proof that ci = c2,
;; produces a justification that t1 = f(..., c2, ...)
(constructor  {congr_constructor} ({proof_datatype} i64 {proof_datatype}) {proof_datatype})
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
    let desugared = match egraph.desugar_program(Some(filename.clone()), &contents) {
        Ok(commands) => commands,
        Err(_) => return false,
    };

    program_supports_proofs(&desugared, &egraph.type_info)
}

/// Checks whether a desugared program supports proof encoding.
pub fn program_supports_proofs(commands: &[ResolvedCommand], type_info: &TypeInfo) -> bool {
    for command in commands {
        if !command_supports_proof_encoding(command, type_info) {
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
            if let GenericExpr::Call(_, ResolvedCall::Primitive(prim), _) = e {
                if prim.validator().is_none() {
                    all_valid = false;
                }
            }
        },
        &mut |_| {},
    );
    all_valid
}

/// Checks whether a resolved command supports proof encoding.
pub(crate) fn command_supports_proof_encoding(
    command: &ResolvedCommand,
    type_info: &TypeInfo,
) -> bool {
    // First, use visit_exprs to check all expressions in the command
    let mut all_primitives_have_validators = true;
    command.clone().visit_exprs(&mut |expr| {
        if !expr_primitives_have_validators(&expr) {
            all_primitives_have_validators = false;
        }
        expr
    });

    if !all_primitives_have_validators {
        return false;
    }

    // Now check command-specific constraints
    match command {
        GenericCommand::Sort(_, _, Some(_))
        | GenericCommand::UserDefined(..)
        | GenericCommand::Input { .. } => false,
        // no-merge on a non-global function
        // To add support: https://github.com/egraphs-good/egglog/issues/774
        GenericCommand::Function {
            merge: None, name, ..
        } => {
            if type_info.is_global(name) {
                return true;
            }
            false
        }
        // let binding with non-eq sort not supported by proof_global_desugar
        ResolvedCommand::Action(ResolvedAction::Let(_, _, expr)) => {
            // let binding with non-eq sort not supported by proof_global_desugar
            // we detect as setting something that is no-merge to a primitive not supported (global primitive binding)
            expr.output_type().is_eq_sort()
        }
        // After global desugar it may look like this
        ResolvedCommand::Action(ResolvedAction::Set(_span, head, _children, expr)) => {
            !type_info.is_global(head.name()) || expr.output_type().is_eq_sort()
        }
        _ => true,
    }
}
