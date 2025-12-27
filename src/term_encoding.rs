//! Utilities for rewriting an [`EGraph`](crate::EGraph) into its "term encoding" form.
//!
//! The term encoding instruments an egglog program so that every constructor and
//! function call is backed by explicit term tables, view tables, and per-sort
//! union-find structures.  This makes canonical representatives and their
//! equality proofs first-class data, paving the way for proof production while
//! keeping the operational semantics equivalent to the standard encoding (for the
//! subset of commands that are currently supported).
//!
//! The transformation is triggered when an `EGraph` is created with
//! [`EGraph::new_with_term_encoding`](crate::EGraph::new_with_term_encoding) or
//! converted via [`EGraph::with_term_encoding_enabled`](crate::EGraph::with_term_encoding_enabled).
//!
//! # Example term encoding for a function
//! Consider a tiny program that defines a pure arithmetic helper and checks a fact about it:
//!
//! ```text
//! (function add (i64 i64) i64 :merge old)
//! (check (= (add 0 0) 0))
//! ```
//!
//! Lowering the program with term encoding expands it to the following commands:
//!
//! ```text
//! (ruleset parent)
//! (ruleset single_parent)
//! (ruleset rebuilding)
//! (ruleset rebuilding_cleanup)
//! (ruleset delete_subsume_ruleset)
//! (sort _view)
//! (constructor add (i64 i64 i64) _view)
//! (constructor addView (i64 i64 i64) _view)
//! (constructor to_delete_add (i64 i64) _view)
//! (constructor to_subsume_add (i64 i64) _view)
//! (sort _mergecleanupsort)
//! (constructor _mergecleanup (i64 i64) _mergecleanupsort)
//! (rule ((addView c0_ c1_ old)
//!        (addView c0_ c1_ new)
//!        (!= old new)
//!        (= (ordering-max old new) new))
//!       ((addView c0_ c1_ old)
//!        (@mergecleanup old old)
//!        (@mergecleanup old new))
//!         :ruleset rebuilding :name "_merge_rule")
//! (rule ((@mergecleanup merged old)
//!        (addView c0_ c1_ merged)
//!        (addView c0_ c1_ old)
//!        (!= merged old))
//!       ((delete (addView c0_ c1_ old)))
//!         :ruleset rebuilding_cleanup :name "_merge_cleanup")
//! (rule ((to_delete_add c0_ c1_)
//!        (addView c0_ c1_ out))
//!       ((delete (addView c0_ c1_ out))
//!        (delete (to_delete_add c0_ c1_)))
//!         :ruleset delete_subsume_ruleset :name "_delete_rule")
//! (rule ((to_subsume_add c0_ c1_)
//!        (addView c0_ c1_ out))
//!       ((delete (addView c0_ c1_ out)))
//!         :ruleset delete_subsume_ruleset :name "_delete_rule_subsume")
//! (check (addView 0 0 @v1)
//! (= @v1 0))
//! (run-schedule (seq (saturate (seq (run rebuilding_cleanup) (saturate (seq (run single_parent))) (saturate (seq (run parent))) (run rebuilding))) (run delete_subsume_ruleset)))
//!```
//!
//! *The new rulesets* orchestrate maintenance for per-sort union-find tables (`parent` and `single_parent`),
//! rebuild-time congruence (`rebuilding` + `rebuilding_cleanup`), and deferred deletions (`delete_subsume_ruleset`).
//!
//! *The constructors* materialise term tables (`add`) alongside their view tables (`addView`) and
//! housekeeping helpers (`to_delete_add`, `to_subsume_add`, and the `_mergecleanup*` pair).
//!
//! *The extra rules* keep the view tables and union-find tables coherent by pruning stale rows and
//! ensuring that congruent applications collapse to a single canonical representative.
//!
//! *Original commands* are rewritten to reference these structures. The `check` now reasons about
//! the canonical representative produced by `addView`, and the trailing `run-schedule` executes the
//! maintenance rules so that the encoded program stays behaviourally equivalent to the uninstrumented version.
use crate::ast::GenericCommand;
use crate::term_encoding_helpers::{EncodingNames, JustificationKind};
use crate::typechecking::FuncType;
use crate::*;
use std::path::Path;

// TODO refactor so that encoding state is optional on the e-graph, ProofNames not optional on EncodingState. Then we don't have to clone proof names everywhere.
#[derive(Clone)]
pub(crate) struct EncodingState {
    pub uf_parent: HashMap<String, String>,
    pub term_header_added: bool,
    // TODO this is very ugly- we should separate out a typechecking struct
    // since we didn't need an entire e-graph
    // When Some term encoding is enabled.
    pub original_typechecking: Option<Box<EGraph>>,
    pub proofs_enabled: bool,
    pub proof_names: EncodingNames,
}

impl EncodingState {
    pub(crate) fn new(symbol_gen: &mut SymbolGen) -> Self {
        Self {
            uf_parent: HashMap::default(),
            term_header_added: false,
            original_typechecking: None,
            proofs_enabled: false,
            proof_names: EncodingNames::new(symbol_gen),
        }
    }
}

/// Thin wrapper around an [`EGraph`] for the term encoding
pub(crate) struct TermState<'a> {
    pub(crate) egraph: &'a mut EGraph,
}

impl<'a> TermState<'a> {
    /// Make a term state and use it to instrument the code.
    pub(crate) fn add_term_encoding(
        egraph: &'a mut EGraph,
        program: Vec<ResolvedNCommand>,
    ) -> Vec<Command> {
        Self { egraph }.add_term_encoding_helper(program)
    }

    /// Mark two things as equal, adding proof if proofs are enabled.
    pub(crate) fn union(&mut self, type_name: &str, lhs: &str, rhs: &str, proof: &str) -> String {
        let uf_name = self.uf_name(type_name);
        let uf_proof_name = self.uf_proof_name(type_name);
        let set_proof = if self.egraph.proof_state.proofs_enabled {
            format!(
                "(set ({uf_proof_name} (ordering-max {lhs} {rhs}) (ordering-min {lhs} {rhs})) {proof})"
            )
        } else {
            "".to_string()
        };

        format!(
            "
        ({uf_name} (ordering-max {lhs} {rhs}) (ordering-min {lhs} {rhs}))
        {set_proof}",
        )
    }

    /// The parent table is the database representation of a union-find datastructure.
    /// When one term has two parents, those parents are unioned in the merge action.
    /// Also, we have a rule that maintains the invariant that each term points to its
    /// canonical representative.
    fn make_uf_table(&mut self, sort_name: &str) -> Vec<Command> {
        let pname = self.uf_name(sort_name);
        let fresh_sort = self.egraph.parser.symbol_gen.fresh("uf");
        let fresh_name = self.egraph.parser.symbol_gen.fresh("uf_update");
        let proof_tables = if self.egraph.proof_state.proofs_enabled {
            let proof_type = self.proof_names().proof_datatype.clone();
            let uf_proof_name = self.uf_proof_name(sort_name);
            format!(
                "(function {uf_proof_name} ({sort_name} {sort_name}) {proof_type} :merge old)
                 "
            )
        } else {
            "".to_string()
        };

        let (proof_query1, proof_action1, to_ast_constructor_code, proof_query2, proof_action2) =
            if self.egraph.proof_state.proofs_enabled {
                let uf_proof_name = self.uf_proof_name(sort_name);
                let p1_fresh = self.egraph.parser.symbol_gen.fresh("p1");
                let p2_fresh = self.egraph.parser.symbol_gen.fresh("p2");
                assert!(
                    self.proof_names()
                        .sort_to_ast_constructor
                        .get(sort_name)
                        .is_none()
                );
                let to_ast_constructor = self
                    .egraph
                    .parser
                    .symbol_gen
                    .fresh(&format!("Ast{}", sort_name));
                self.egraph
                    .proof_state
                    .proof_names
                    .sort_to_ast_constructor
                    .insert(sort_name.to_string(), to_ast_constructor.clone());
                let uf_proof_name = self.uf_proof_name(sort_name);
                let p_fresh = self.egraph.parser.symbol_gen.fresh("p");
                let p2_fresh = self.egraph.parser.symbol_gen.fresh("p2");
                let trans_constructor = &self.proof_names().eq_trans_constructor;
                let symm_constructor = &self.proof_names().eq_sym_constructor;
                let ast_sort = &self.proof_names().ast_sort;

                (
                    format!(
                        "(= {p1_fresh} ({uf_proof_name} a b))
                     (= {p2_fresh} ({uf_proof_name} b c))"
                    ),
                    format!(
                        "(set ({uf_proof_name} a c)
                              ({trans_constructor} {p1_fresh} {p2_fresh}))"
                    ),
                    format!("(constructor {to_ast_constructor} ({sort_name}) {ast_sort})"),
                    format!(
                        "(= {p_fresh} ({uf_proof_name} a b))
                         (= {p2_fresh} ({uf_proof_name} a c))"
                    ),
                    format!(
                        "(set ({uf_proof_name} b c)
                          ({trans_constructor}
                             ({symm_constructor} {p_fresh})
                             {p2_fresh}))"
                    ),
                )
            } else {
                (
                    "".to_string(),
                    "".to_string(),
                    "".to_string(),
                    "".to_string(),
                    "".to_string(),
                )
            };

        let parent_direct_ruleset_name = self.parent_direct_ruleset_name();
        let single_parent_ruleset_name = self.proof_names().single_parent_ruleset_name.clone();

        self.parse_program(&format!(
            "(sort {fresh_sort})
             (constructor {pname} ({sort_name} {sort_name}) {fresh_sort})
             {to_ast_constructor_code}
             {proof_tables}
             (rule (({pname} a b)
                    ({pname} b c)
                    (!= b c)
                    {proof_query1})
                  ((delete ({pname} a b))
                   ({pname} a c)
                   {proof_action1})
                   :ruleset {parent_direct_ruleset_name}
                   :name \"{fresh_name}\")
             (rule (({pname} a b)
                    ({pname} a c)
                    (!= b c)
                    (= (ordering-max b c) b)
                    {proof_query2})
                  ((delete ({pname} a b))
                   ({pname} b c)
                    {proof_action2})
                   :ruleset {single_parent_ruleset_name}
                   :name \"singleparent{fresh_name}\")
                   ",
        ))
    }

    fn delete_and_subsume(&mut self, fdecl: &ResolvedFunctionDecl) -> String {
        let child_names = fdecl
            .schema
            .input
            .iter()
            .enumerate()
            .map(|(i, _)| format!("c{i}_"))
            .collect::<Vec<_>>()
            .join(" ");
        let to_delete_name = self.to_delete_name(&fdecl.name);
        let subsumed_name = self.subsumed_name(&fdecl.name);
        let view_name = self.view_name(&fdecl.name);
        let delete_subsume_ruleset = self.delete_subsume_ruleset_name();
        let fresh_name = self.egraph.parser.symbol_gen.fresh("delete_rule");

        // Delete just goes once while subsume continues to delete in the future
        format!(
            "(rule (({to_delete_name} {child_names})
                    ({view_name} {child_names} out))
                   ((delete ({view_name} {child_names} out))
                    (delete ({to_delete_name} {child_names})))
                    :ruleset {delete_subsume_ruleset}
                    :name \"{fresh_name}\")
             (rule (({subsumed_name} {child_names})
                    ({view_name} {child_names} out))
                   ((delete ({view_name} {child_names} out)))
                    :ruleset {delete_subsume_ruleset}
                    :name \"{fresh_name}_subsume\")"
        )
    }

    // Generate a rule that runs the merge function for custom functions.
    fn handle_merge_fn(&mut self, fdecl: &ResolvedFunctionDecl) -> String {
        let child_names = fdecl
            .schema
            .input
            .iter()
            .enumerate()
            .map(|(i, _)| format!("c{i}_"))
            .collect::<Vec<_>>();
        let child_names_str = child_names.join(" ");
        let rebuilding_ruleset = self.rebuilding_ruleset_name();
        let view_name = self.view_name(&fdecl.name);
        if fdecl.subtype == FunctionSubtype::Custom {
            let name = &fdecl.name;

            let merge_fn = &fdecl
                .merge
                .as_ref()
                .unwrap_or_else(|| panic!("Proofs don't support :no-merge"));

            let rebuilding_cleanup_ruleset = self.rebuilding_cleanup_ruleset_name();
            let fresh_name = self.egraph.parser.symbol_gen.fresh("merge_rule");
            let cleanup_name = self.egraph.parser.symbol_gen.fresh("merge_cleanup");
            let p1_fresh = self.egraph.parser.symbol_gen.fresh("p1");
            let p2_fresh = self.egraph.parser.symbol_gen.fresh("p2");
            let view_proof_name = self.view_proof_name(&fdecl.name);
            let proof_query = if self.egraph.proof_state.proofs_enabled {
                format!(
                    "(= {p1_fresh} ({view_proof_name} {child_names_str} old))
                     (= {p2_fresh} ({view_proof_name} {child_names_str} new))
                    "
                )
            } else {
                "".to_string()
            };
            let proof_var = self.fresh_var();
            let rule_proof = if self.egraph.proof_state.proofs_enabled {
                let merge_fn_constructor = self.proof_names().merge_fn_constructor.clone();
                format!(
                    "(let {proof_var}
                            ({merge_fn_constructor} \"{name}\"
                                  {p1_fresh}
                                  {p2_fresh}))"
                )
            } else {
                "".to_string()
            };

            let mut merge_fn_code = vec![];
            let merge_fn_var =
                self.instrument_action_expr(merge_fn, &mut merge_fn_code, &proof_var);
            let merge_fn_code_str = merge_fn_code.join("\n");
            let mut updated = child_names.clone();
            updated.push(merge_fn_var.clone());
            let term_and_proof = self.update_view(name, &updated, &proof_var);
            let fresh_constructor = self.egraph.parser.symbol_gen.fresh("mergecleanup");
            let fresh_sort = self.egraph.parser.symbol_gen.fresh("mergecleanupsort");
            let output_sort = fdecl.schema.output.clone();

            // The first runs the merge function adding a new row.
            // The second deletes rows with old values for the old variable, while the third deletes rows with new values for the new variable.
            format!(
                "(sort {fresh_sort})
                 (constructor {fresh_constructor} ({output_sort} {output_sort}) {fresh_sort})
                 (rule (({view_name} {child_names_str} old)
                        ({view_name} {child_names_str} new)
                        (!= old new)
                        (= (ordering-max old new) new)
                        {proof_query})
                       ({merge_fn_code_str}
                        {rule_proof}
                        {term_and_proof}
                        ({fresh_constructor} {merge_fn_var} old)
                        ({fresh_constructor} {merge_fn_var} new)
                       )
                        :ruleset {rebuilding_ruleset}
                        :name \"{fresh_name}\")
                
                 (rule (({fresh_constructor} merged old)
                        ({view_name} {child_names_str} merged)
                        ({view_name} {child_names_str} old)
                        (!= merged old))
                       ((delete ({view_name} {child_names_str} old)))
                        :ruleset {rebuilding_cleanup_ruleset}
                        :name \"{cleanup_name}\")
                ",
            )
        } else {
            // Congruence rule
            let fresh_name = self.egraph.parser.symbol_gen.fresh("congruence_rule");
            let uf_name = self.uf_name(&fdecl.schema.output);
            format!(
                "(rule (({view_name} {child_names_str} old)
                        ({view_name} {child_names_str} new)
                        (!= old new)
                        (= (ordering-max old new) new))
                       (({uf_name} new old))
                        :ruleset {rebuilding_ruleset}
                        :name \"{fresh_name}\")"
            )
        }
    }

    /// Each function/constructor gets a term table and a view table.
    /// The term table stores underlying representative terms.
    /// The view table stores child terms and their eclass.
    /// The view table is mutated using delete, but we never delete from term tables.
    /// We re-use the original name of the function for the term table.
    fn term_and_view(&mut self, fdecl: &ResolvedFunctionDecl) -> Vec<Command> {
        let schema = &fdecl.schema;
        let out_type = schema.output.clone();

        let name = &fdecl.name;
        let view_name = self.view_name(&fdecl.name);
        let in_sorts = ListDisplay(schema.input.clone(), " ");
        let fresh_sort = self.egraph.parser.symbol_gen.fresh("view");
        let delete_rule = self.delete_and_subsume(fdecl);
        let to_delete_name = self.to_delete_name(&fdecl.name);
        let subsumed_name = self.subsumed_name(&fdecl.name);
        let term_sorts = format!(
            "{in_sorts} {}",
            if fdecl.subtype == FunctionSubtype::Constructor {
                "".to_string()
            } else {
                schema.output.to_string()
            }
        );
        let view_sorts = format!("{in_sorts} {out_type}");
        let proof_constructors = self.proof_functions(fdecl, &view_sorts);

        let view_sort = if fdecl.subtype == FunctionSubtype::Constructor {
            schema.output.clone()
        } else {
            fresh_sort.clone()
        };

        if self.egraph.proof_state.proofs_enabled {
            self.egraph
                .proof_state
                .proof_names
                .fn_to_term_sort
                .insert(name.clone(), view_sort.clone());
        }
        let merge_rule = self.handle_merge_fn(fdecl);
        // the term table has child_sorts as inputs
        // the view table has child_sorts + the leader term for the eclass
        self.parse_program(&format!(
            "
            (sort {fresh_sort})
            (constructor {name} ({term_sorts}) {view_sort})
            (constructor {view_name} ({view_sorts}) {fresh_sort})
            (constructor {to_delete_name} ({in_sorts}) {fresh_sort})
            (constructor {subsumed_name} ({in_sorts}) {fresh_sort})
            {proof_constructors}
            {merge_rule}
            {delete_rule}",
        ))
    }

    fn proof_functions(&mut self, fdecl: &ResolvedFunctionDecl, view_sorts: &str) -> String {
        if !self.egraph.proof_state.proofs_enabled {
            return "".to_string();
        }

        let view_proof_name = self.view_proof_name(&fdecl.name);
        let proof_type = self.proof_names().proof_datatype.clone();

        // A view proof gives a proof that the representative term t_r equals the term in the view.
        // Example: (AddView 2 3 t_r) proves the proposition that t_r = Add(2, 3) in that direction.
        // This direction makes adding more proofs easier.
        format!(
            "
            (function {view_proof_name} ({view_sorts}) {proof_type} :merge old)
            "
        )
    }

    /// Rules that update the views when children change.
    fn rebuilding_rules(&mut self, fdecl: &ResolvedFunctionDecl) -> Vec<Command> {
        let types = fdecl.resolved_schema.view_types();
        let mut res = vec![];
        // a rule updating index i
        for i in 0..types.len() {
            // if the type at index i is not an eq sort, skip
            if !types[i].is_eq_sort() {
                continue;
            }

            let types = fdecl.resolved_schema.view_types();

            let view_name = self.view_name(&fdecl.name);
            let child = |i| format!("c{i}_");
            let children_vec = (0..types.len()).map(child).collect::<Vec<_>>();
            let children = format!("{}", ListDisplay(&children_vec, " "));
            let mut children_updated = vec![];
            let old_child = child(i);
            let Some((updated_child_query, updated_child_var, updated_child_prf)) =
                self.wrap_parent(child(i), types[i].clone())
            else {
                panic!("Failed to get canonical expr for rebuilding rule");
            };

            for j in 0..types.len() {
                if j == i {
                    children_updated.push(updated_child_var.clone());
                } else {
                    children_updated.push(child(j).to_string());
                }
            }

            let fresh_name = self.egraph.parser.symbol_gen.fresh("rebuild_rule");
            let view_prf = self.query_view_and_get_proof(&fdecl.name, &children_vec);

            let (pf_code, pf_var) = if self.egraph.proof_state.proofs_enabled {
                let proof = self.fresh_var();
                let eq_trans_constructor = self.proof_names().eq_trans_constructor.clone();
                let congr_constructor = self.proof_names().congr_constructor.clone();

                // if we are updating the last element of a constructor
                // it's updating the representative term, use transitivity
                (
                    if fdecl.subtype == FunctionSubtype::Constructor && i == types.len() - 1 {
                        format!(
                            "(let {proof}
                               ({eq_trans_constructor}
                                  {updated_child_prf}
                                  {view_prf}))",
                        )
                    } else {
                        // otherwise we are updating a child via congruence
                        format!(
                            "(let {proof}
                                  ({congr_constructor} {view_prf} {i}
                                                       {updated_child_prf}))
                    ",
                        )
                    },
                    proof,
                )
            } else {
                ("".to_string(), "".to_string())
            };
            let updated_view = self.update_view(
                &fdecl.name,
                &children_updated,
                &pf_var,
                fdecl.subtype == FunctionSubtype::Constructor,
            );

            // Make a rule that updates the view
            let rule = format!(
                "(rule (({view_name} {children})
                        {updated_child_query}
                        (!= {updated_child_var} {old_child})
                        )
                     (
                      {pf_code}
                      {updated_view}
                      (delete ({view_name} {children}))
                     )
                      :ruleset {} :name \"{fresh_name}\")",
                self.rebuilding_ruleset_name(),
            );
            res.extend(self.parse_program(&rule));
        }
        res
    }

    /// Instrument fact replaces terms with looking up
    /// canonical versions in the view.
    /// It also needs to look up references to globals.
    fn instrument_fact(&mut self, fact: &ResolvedFact, res: &mut Vec<String>) {
        match fact {
            GenericFact::Eq(_span, generic_expr, generic_expr1) => {
                let v1 = self.instrument_fact_expr(generic_expr, res);
                let v2 = self.instrument_fact_expr(generic_expr1, res);
                res.push(format!("(= {} {})", v1, v2));
            }
            GenericFact::Fact(generic_expr) => {
                let _ = self.instrument_fact_expr(generic_expr, res);
            }
        }
    }

    /// Instruments a fact expression to use the view tables.
    /// Returns a variable representing the expression.
    fn instrument_fact_expr(&mut self, expr: &ResolvedExpr, res: &mut Vec<String>) -> String {
        match expr {
            ResolvedExpr::Lit(_, lit) => format!("{}", lit),
            ResolvedExpr::Var(_, resolved_var) => resolved_var.name.clone(),
            ResolvedExpr::Call(_, resolved_call, args) => {
                let args = args
                    .iter()
                    .map(|arg| self.instrument_fact_expr(arg, res))
                    .collect::<Vec<_>>();
                match resolved_call {
                    ResolvedCall::Func(func_type) => {
                        let fv = self.fresh_var();
                        res.push(format!(
                            "({} {} {fv})",
                            self.view_name(&func_type.name),
                            ListDisplay(args, " ")
                        ));
                        fv
                    }
                    ResolvedCall::Primitive(specialized_primitive) => {
                        if specialized_primitive.output().is_eq_sort() {
                            panic!(
                                "Term encoding does not support eq-sort primitive expressions in facts"
                            );
                        }
                        let fv = self.fresh_var();
                        res.push(format!(
                            "(= {fv} ({} {}))",
                            specialized_primitive.name(),
                            ListDisplay(args, " ")
                        ));
                        fv
                    }
                }
            }
        }
    }

    // Returns a query, variable for the updated child, and proof
    fn wrap_parent(&mut self, var: String, sort: ArcSort) -> Option<(String, String, String)> {
        if sort.is_eq_sort() {
            let fresh = self.fresh_var();
            let parent = self.uf_name(sort.name());
            let fresh_proof = self.fresh_var();
            let proof = if self.egraph.proof_state.proofs_enabled {
                let uf_proof = self.uf_proof_name(sort.name());
                format!("(= {fresh_proof} ({uf_proof} {var}))")
            } else {
                "".to_string()
            };
            Some((
                format!(
                    "({parent} {var} {fresh})
                     {proof}"
                ),
                fresh,
                fresh_proof,
            ))
        } else {
            None
        }
    }

    fn instrument_facts(&mut self, facts: &[ResolvedFact]) -> Vec<String> {
        let mut res = vec![];
        for fact in facts {
            self.instrument_fact(fact, &mut res);
        }
        res
    }

    // Actions need to be instrumented to add to the view
    // as well as to the terms tables.
    fn instrument_action(
        &mut self,
        action: &ResolvedAction,
        justification: &JustificationKind,
    ) -> Vec<String> {
        let mut res = vec![];

        match action {
            ResolvedAction::Let(_span, v, generic_expr) => {
                let v2 = self.instrument_action_expr(generic_expr, &mut res, justification);
                res.push(format!("(let {} {})", v.name, v2));
            }
            ResolvedAction::Set(_span, h, generic_exprs, generic_expr) => {
                let mut exprs = vec![];
                for e in generic_exprs.iter().chain(std::iter::once(generic_expr)) {
                    exprs.push(self.instrument_action_expr(e, &mut res, justification));
                }

                let ResolvedCall::Func(func_type) = h else {
                    panic!(
                        "Set action on non-function, should have been prevented by typechecking"
                    );
                };

                let (add_code, _fv) = self.add_term_and_view(func_type, &exprs, justification);
                res.extend(add_code);
            }
            ResolvedAction::Change(_span, change, h, generic_exprs) => {
                if let ResolvedCall::Func(func_type) = h {
                    let symbol = match change {
                        Change::Delete => self.to_delete_name(&func_type.name),
                        Change::Subsume => self.subsumed_name(&func_type.name),
                    };
                    let children = generic_exprs
                        .iter()
                        .map(|e| self.instrument_action_expr(e, &mut res, justification))
                        .collect::<Vec<_>>();

                    res.push(format!("({symbol} {})", ListDisplay(children, " ")));
                } else {
                    panic!(
                        "Delete action on non-function, should have been prevented by typechecking"
                    );
                }
            }
            ResolvedAction::Union(_span, generic_expr, generic_expr1) => {
                let v1 = self.instrument_action_expr(generic_expr, &mut res, justification);
                let v2 = self.instrument_action_expr(generic_expr1, &mut res, justification);
                let ot = generic_expr.output_type();
                let type_name = ot.name();
                let unioned = self.union(type_name, &v1, &v2, justification);
                res.push(unioned);
            }
            ResolvedAction::Panic(..) => {
                res.push(format!("{}", action));
            }
            ResolvedAction::Expr(_span, generic_expr) => {
                self.instrument_action_expr(generic_expr, &mut res, justification);
            }
        }

        res
    }

    /// Update the view with  the given arguments.
    /// The arguments include the eclass for constructors.
    fn update_view(&mut self, fname: &str, args: &[String], proof: &str) -> String {
        let mut res = vec![];
        res.push(format!(
            "({} {})",
            self.view_name(fname),
            ListDisplay(args, " "),
        ));

        if self.egraph.proof_state.proofs_enabled {
            let proof_name = self.view_proof_name(fname);
            res.push(format!(
                "(set ({proof_name} {}) {proof})",
                ListDisplay(args, " ")
            ));
        }
        res.join("\n")
    }

    /// Return some code adding to the view and term tables.
    /// For constructors, the arguments do not include the eclass of the resulting term (since it may not exist yet).
    fn add_term_and_view(
        &mut self,
        func_type: &FuncType,
        args: &[String],
        justification: &JustificationKind,
    ) -> (Vec<String>, String) {
        let fv = self.fresh_var();
        let mut res = vec![];
        res.push(format!(
            "(let {fv} ({} {}))",
            func_type.name,
            ListDisplay(args, " ")
        ));

        let args_with_fv = if func_type.subtype == FunctionSubtype::Constructor {
            let mut a = args.to_vec();
            a.push(fv.clone());
            a
        } else {
            args.to_vec()
        };

        let (proof_str, proof_var) = if self.egraph.proof_state.proofs_enabled {
            let (term1, term2) = match func_type.subtype {
                FunctionSubtype::Constructor => {
                    
                }
                FunctionSubtype::Custom => (
                    fv.clone(),
                    format!("({} {})", func_type.name, ListDisplay(args, " ")),
                ),
                FunctionSubtype::Relation => {
                    panic!("Relations not supported by proofs, should have been caught earlier.")
                }
            };
        } else {
            ("".to_string(), "".to_string())
        };

        res.push(self.update_view(
            &func_type.name,
            &args_with_fv,
            proof,
            func_type.subtype == FunctionSubtype::Constructor,
        ));

        // add to uf table to initialize eclass for constructors
        if func_type.subtype == FunctionSubtype::Constructor {
            self.union(func_type.output.name(), &fv, &fv, proof);
        }

        (res, fv)
    }

    /// Returns a query for (fname args) and in proof mode returns a variable for the proof.
    fn query_view_and_get_proof(&mut self, fname: &str, args: &[String]) -> String {
        let mut res = vec![];
        res.push(format!(
            "({} {})",
            self.view_name(fname),
            ListDisplay(args, " "),
        ));

        if self.egraph.proof_state.proofs_enabled {
            let proof_name = self.view_proof_name(fname);
            let pf_var = self.fresh_var();
            res.push(format!(
                "(= {pf_var} ({proof_name} {}))",
                ListDisplay(args, " ")
            ));
        }

        res.join("\n")
    }

    // Add to view and term tables, returning a variable for the created term.
    fn instrument_action_expr(
        &mut self,
        expr: &ResolvedExpr,
        res: &mut Vec<String>,
        proof: &JustificationKind,
    ) -> String {
        match expr {
            ResolvedExpr::Lit(_, lit) => format!("{}", lit),
            ResolvedExpr::Var(_, resolved_var) => resolved_var.name.clone(),
            ResolvedExpr::Call(_, resolved_call, args) => {
                let args = args
                    .iter()
                    .map(|arg| self.instrument_action_expr(arg, res, proof))
                    .collect::<Vec<_>>();
                match resolved_call {
                    ResolvedCall::Func(func_type) => {
                        if func_type.subtype == FunctionSubtype::Custom {
                            panic!(
                                "Found a function lookup in actions, should have been prevented by typechecking"
                            );
                        }
                        let (add_code, fv) = self.add_term_and_view(func_type, &args, proof);
                        res.extend(add_code);

                        fv
                    }
                    ResolvedCall::Primitive(specialized_primitive) => {
                        let fv = self.fresh_var();
                        res.push(format!(
                            "(let {} ({} {}))",
                            fv,
                            specialized_primitive.name(),
                            ListDisplay(args, " ")
                        ));
                        fv
                    }
                }
            }
        }
    }

    /// In proof mode, rule_proof justifies the actions taken.
    fn instrument_actions(
        &mut self,
        actions: &[ResolvedAction],
        justification: &JustificationKind,
    ) -> Vec<String> {
        let mut res = vec![];
        for action in actions {
            res.extend(self.instrument_action(action, proof));
        }
        res
    }

    /// Instrument a rule to use term encoding. This involves using the view tables in facts,
    /// adding to term and view tables in actions.
    /// When proofs are enabled we query proof tables, then build a proof for the rule in the actions.
    /// Finally, each view update also updates the proof tables.
    fn instrument_rule(&mut self, rule: &ResolvedRule) -> Vec<Command> {
        let facts = self.instrument_facts(&rule.body);
        // TODO get rule proof
        let todoproof = JustificationKind::Rule(rule.name.clone(), format!("(TODOList)"));
        let actions = self.instrument_actions(&rule.head.0, &todoproof);
        let name = &rule.name;
        let instrumented = format!(
            "(rule ({})
                   ({})
                    {}
                    :name \"{name}\")",
            ListDisplay(facts, " "),
            ListDisplay(actions, " "),
            if rule.ruleset.is_empty() {
                "".to_string()
            } else {
                format!(":ruleset {}", rule.ruleset)
            }
        );
        self.parse_program(&instrumented)
    }

    /// TODO experiment with schedule- unclear what is fastest.
    /// Any schedule should be sound as long as we saturate.
    fn rebuild(&mut self) -> Schedule {
        let parent_direct_ruleset = self.parent_direct_ruleset_name();
        let single_parent = self.proof_names().single_parent_ruleset_name.clone();
        let rebuilding_cleanup_ruleset = self.rebuilding_cleanup_ruleset_name();
        let rebuilding_ruleset = self.rebuilding_ruleset_name();
        let delete_ruleset = self.delete_subsume_ruleset_name();
        self.parse_schedule(format!(
            "(seq
              (saturate
                  {rebuilding_cleanup_ruleset}
                  (saturate {single_parent})
                  (saturate {parent_direct_ruleset})
                  {rebuilding_ruleset})
              {delete_ruleset})"
        ))
    }

    fn instrument_schedule(&mut self, schedule: &ResolvedSchedule) -> Schedule {
        match schedule {
            ResolvedSchedule::Run(span, config) => {
                let new_run = match config.until {
                    Some(ref facts) => {
                        let instrumented = self.instrument_facts(facts);
                        let instrumented_facts = self.parse_facts(&instrumented);
                        Schedule::Run(
                            span.clone(),
                            RunConfig {
                                ruleset: config.ruleset.clone(),
                                until: Some(instrumented_facts),
                            },
                        )
                    }
                    None => Schedule::Run(
                        span.clone(),
                        RunConfig {
                            ruleset: config.ruleset.clone(),
                            until: None,
                        },
                    ),
                };
                Schedule::Sequence(span.clone(), vec![new_run, self.rebuild()])
            }
            ResolvedSchedule::Sequence(span, schedules) => Schedule::Sequence(
                span.clone(),
                schedules
                    .iter()
                    .map(|s| self.instrument_schedule(s))
                    .collect(),
            ),
            ResolvedSchedule::Saturate(span, schedule) => {
                Schedule::Saturate(span.clone(), Box::new(self.instrument_schedule(schedule)))
            }
            GenericSchedule::Repeat(span, n, schedule) => Schedule::Repeat(
                span.clone(),
                *n,
                Box::new(self.instrument_schedule(schedule)),
            ),
        }
    }

    fn term_encode_command(&mut self, command: &ResolvedNCommand, res: &mut Vec<Command>) {
        match &command {
            ResolvedNCommand::Sort(_span, name, _presort_and_args) => {
                res.push(command.to_command().make_unresolved());
                res.extend(self.make_uf_table(name));
            }
            ResolvedNCommand::Function(fdecl) => {
                res.extend(self.term_and_view(fdecl));
                res.extend(self.rebuilding_rules(fdecl));
            }
            ResolvedNCommand::NormRule { rule } => {
                res.extend(self.instrument_rule(rule));
            }
            ResolvedNCommand::CoreAction(action) => {
                let fiat = self.fiat_proof();
                let instrumented = self.instrument_action(action, &fiat).join("\n");
                res.extend(self.parse_program(&instrumented));
            }
            ResolvedNCommand::Check(span, facts) => {
                let instrumented = self.instrument_facts(facts);
                res.push(Command::Check(
                    span.clone(),
                    self.parse_facts(&instrumented),
                ));
            }
            ResolvedNCommand::RunSchedule(schedule) => {
                res.push(Command::RunSchedule(self.instrument_schedule(schedule)));
            }
            ResolvedNCommand::Fail(span, cmd) => {
                self.term_encode_command(cmd, res);
                let last = res.pop().unwrap();
                res.push(Command::Fail(span.clone(), Box::new(last)));
            }
            ResolvedNCommand::Pop(..)
            | ResolvedNCommand::Push(..)
            | ResolvedNCommand::AddRuleset(..)
            | ResolvedNCommand::PrintSize(..)
            | ResolvedNCommand::Output { .. }
            | ResolvedNCommand::Input { .. }
            | ResolvedNCommand::UnstableCombinedRuleset(..)
            | ResolvedNCommand::PrintOverallStatistics(..)
            | ResolvedNCommand::PrintFunction(..) => {
                res.push(command.to_command().make_unresolved());
            }
            ResolvedNCommand::Extract(..) => {
                // TODO we just omit extract for now, support in future
            }
            ResolvedNCommand::UserDefined(..) => {
                panic!("User defined commands unsupported in term encoding");
            }
        }
    }

    pub(crate) fn add_term_encoding_helper(
        &mut self,
        program: Vec<ResolvedNCommand>,
    ) -> Vec<Command> {
        let mut res = vec![];

        if !self.egraph.proof_state.term_header_added {
            res.extend(self.term_header());
            if self.egraph.proof_state.proofs_enabled {
                let proof_header = self.proof_header();
                res.extend(self.parse_program(&proof_header));
            }
            self.egraph.proof_state.term_header_added = true;
        }

        for command in program {
            self.term_encode_command(&command, &mut res);

            // run rebuilding after every command except a few
            if let ResolvedNCommand::Function(..)
            | ResolvedNCommand::NormRule { .. }
            | ResolvedNCommand::Sort(..) = &command
            {
            } else {
                res.push(Command::RunSchedule(self.rebuild()));
            }
        }

        res
    }
}

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
    let desugared = match egraph.desugar_program(Some(filename), &contents) {
        Ok(commands) => commands,
        Err(_) => return false,
    };

    commands_support_proof_encoding(&desugared)
}

pub fn commands_support_proof_encoding(commands: &[ResolvedCommand]) -> bool {
    for command in commands {
        if !command_supports_proof_encoding(command) {
            return false;
        }
    }
    true
}

pub fn command_supports_proof_encoding(command: &ResolvedCommand) -> bool {
    match command {
        GenericCommand::Sort(_, _, Some(_))
        | GenericCommand::UserDefined(..)
        | GenericCommand::Relation { .. }
        | GenericCommand::Input { .. } => false,
        ResolvedCommand::Action(ResolvedAction::Let(_, _, expr)) => expr.output_type().is_eq_sort(),
        // no-merge isn't supported right now
        ResolvedCommand::Function { merge: None, .. } => false,
        // delete or subsume on custom functions isn't supported
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Parser;
    use crate::ast::desugar::desugar_command;
    use crate::ast::proof_global_remover;

    fn term_encode(source: &str) -> Vec<Command> {
        let mut egraph = crate::EGraph::new_with_term_encoding();
        let mut parser = Parser::default();
        let program = parser
            .get_program_from_string(None, source)
            .expect("failed to parse program");
        let mut ncommands = Vec::new();
        for command in program {
            let desugared =
                desugar_command(command, &mut parser).expect("failed to desugar command");
            ncommands.extend(desugared);
        }

        let mut resolved = egraph
            .typecheck_program(&ncommands)
            .expect("failed to typecheck program");
        resolved = proof_global_remover::remove_globals(resolved, &mut parser.symbol_gen);
        TermState::add_term_encoding(&mut egraph, resolved)
    }

    #[test]
    fn doc_example_add_function2() {
        let commands = term_encode(
            r#"
            (function add (i64 i64) i64 :merge old)
            (check (= (add 0 0) 0))
            "#,
        );

        let snapshot = sanitize_internal_names(&commands)
            .iter()
            .map(|cmd| cmd.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        insta::assert_snapshot!("doc_example_add_function2", snapshot);
    }

    #[test]
    fn doc_example_add_function1() {
        let commands = term_encode(
            r#"
            (sort Math)
            (constructor Add (i64 i64) Math)
            (union (Add 1 2) (Add 2 1))
            (check (= (Add 1 2) (Add 2 1)))
            "#,
        );

        let snapshot = sanitize_internal_names(&commands)
            .iter()
            .map(|cmd| cmd.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        insta::assert_snapshot!("doc_example_add_function1", snapshot);
    }
}
