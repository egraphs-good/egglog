use crate::ast::GenericCommand;
use crate::typechecking::FuncType;
use crate::*;
use std::path::Path;

#[derive(Clone)]
pub(crate) struct ProofNames {
    pub proof_list_sort: String,
    pub ast_sort: String,
    pub justification_datatype: String,
    pub proof_datatype: String,
    pub fiat_constructor: String,
    pub rule_constructor: String,
    pub merge_fn_constructor: String,
    pub eq_constructor: String,
    pub prim_refl_constructor: String,
    pub eq_trans_constructor: String,
    pub eq_sym_constructor: String,
    pub congr_constructor: String,
}

impl ProofNames {
    fn new(symbol_gen: &mut SymbolGen) -> Self {
        Self {
            proof_list_sort: symbol_gen.fresh("proof_list"),
            ast_sort: symbol_gen.fresh("proof_ast"),
            justification_datatype: symbol_gen.fresh("justification"),
            proof_datatype: symbol_gen.fresh("proof"),
            fiat_constructor: symbol_gen.fresh("fiat"),
            rule_constructor: symbol_gen.fresh("rule"),
            merge_fn_constructor: symbol_gen.fresh("merge_fn"),
            eq_constructor: symbol_gen.fresh("eq"),
            prim_refl_constructor: symbol_gen.fresh("prim_refl"),
            eq_trans_constructor: symbol_gen.fresh("eq_trans"),
            eq_sym_constructor: symbol_gen.fresh("eq_sym"),
            congr_constructor: symbol_gen.fresh("congr"),
        }
    }
}

#[derive(Default, Clone)]
pub(crate) struct EncodingState {
    pub uf_parent: HashMap<String, String>,
    pub term_header_added: bool,
    // TODO this is very ugly- we should separate out a typechecking struct
    // since we didn't need an entire e-graph
    // When Some term encoding is enabled.
    pub original_typechecking: Option<Box<EGraph>>,
    pub proofs_enabled: bool,
    pub proof_names: Option<ProofNames>,
}

/// Thin wrapper around an [`EGraph`] for the term encoding
pub(crate) struct TermState<'a> {
    egraph: &'a mut EGraph,
}

impl<'a> TermState<'a> {
    /// Make a term state and use it to instrument the code.
    pub(crate) fn add_term_encoding(
        egraph: &'a mut EGraph,
        program: Vec<ResolvedNCommand>,
    ) -> Vec<Command> {
        Self { egraph }.add_term_encoding_helper(program)
    }

    pub(crate) fn uf_name(&mut self, sort: &str) -> String {
        if let Some(name) = self.egraph.proof_state.uf_parent.get(sort) {
            name.clone()
        } else {
            self.egraph.proof_state.uf_parent.insert(
                sort.to_string(),
                format!("{INTERNAL_SYMBOL_PREFIX}ufparent_{}", sort,),
            );
            self.egraph.proof_state.uf_parent[sort].clone()
        }
    }

    pub(crate) fn uf_proof_name(&mut self, sort: &str) -> String {
        format!("{}Proof", self.uf_name(sort),)
    }

    fn single_parent_ruleset_name(&self) -> String {
        "single_parent".to_string()
    }

    fn to_ast_name(&self, sort: &str) -> String {
        format!("AST{}", sort,)
    }

    fn proof_names(&mut self) -> &ProofNames {
        if self.egraph.proof_state.proof_names.is_none() {
            let names = ProofNames::new(&mut self.egraph.parser.symbol_gen);
            self.egraph.proof_state.proof_names = Some(names);
        }
        self.egraph
            .proof_state
            .proof_names
            .as_ref()
            .expect("proof names initialized")
    }

    fn fiat_justification(&mut self) -> String {
        if !self.egraph.proof_state.proofs_enabled {
            return String::new();
        }
        let names = self.proof_names();
        format!("({})", names.fiat_constructor)
    }

    /// Mark two things as equal, adding justification if proofs are enabled.
    pub(crate) fn union(
        &mut self,
        type_name: &str,
        lhs: &str,
        rhs: &str,
        justification: &str,
    ) -> String {
        let uf_name = self.uf_name(type_name);
        let uf_proof_name = self.uf_proof_name(type_name);
        let set_justification = if self.egraph.proof_state.proofs_enabled {
            format!(
                "(set ({uf_proof_name} (ordering-max {lhs} {rhs}) (ordering-min {lhs} {rhs})) {justification})"
            )
        } else {
            "".to_string()
        };

        format!(
            "
        ({uf_name} (ordering-max {lhs} {rhs}) (ordering-min {lhs} {rhs}))
        {set_justification}",
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
        let proof_code = if self.egraph.proof_state.proofs_enabled {
            let proof_type = self.proof_names().proof_datatype.clone();
            let ast_sort = self.proof_names().ast_sort.clone();
            let uf_proof_name = self.uf_proof_name(sort_name);
            let to_ast_name = self.to_ast_name(sort_name);
            format!(
                "(function {uf_proof_name} ({sort_name} {sort_name}) {proof_type} :merge old)
                 (constructor {to_ast_name} ({sort_name}) {ast_sort})"
            )
        } else {
            "".to_string()
        };

        self.parse_program(&format!(
            "(sort {fresh_sort})
             (constructor {pname} ({sort_name} {sort_name}) {fresh_sort})
             {proof_code}
             (rule (({pname} a b)
                    ({pname} b c)
                    (!= b c))
                  ((delete ({pname} a b))
                   ({pname} a c))
                   :ruleset {}
                   :name \"{fresh_name}\")
             (rule (({pname} a b)
                    ({pname} a c)
                    (!= b c))
                  ((delete ({pname} a (ordering-max b c)))
                   ({pname} (ordering-max b c) (ordering-min b c)))
                   :ruleset {}
                   :name \"singleparent{fresh_name}\")
                   ",
            self.parent_direct_ruleset_name(),
            self.single_parent_ruleset_name(),
        ))
    }

    // Each function/constructor gets a view table, the canonicalized e-nodes to accelerate e-matching.
    fn view_name(&self, name: &str) -> String {
        format!("{}View", name,)
    }

    fn to_delete_name(&self, name: &str) -> String {
        format!("to_delete_{}", name,)
    }

    fn subsumed_name(&self, name: &str) -> String {
        format!("to_subsume_{}", name,)
    }

    fn delete_subsume_ruleset_name(&self) -> String {
        "delete_subsume_ruleset".to_string()
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
            let rule_proof_var = self.fresh_var();
            let rule_proof = if self.egraph.proof_state.proofs_enabled {
                let merge_fn_constructor = self.proof_names().merge_fn_constructor.clone();
                format!(
                    "(let {rule_proof_var}
                            ({merge_fn_constructor} \"{name}\"
                                  {p1_fresh}
                                  {p2_fresh}))"
                )
            } else {
                "".to_string()
            };

            let mut merge_fn_code = vec![];
            let merge_fn_var =
                self.instrument_action_expr(merge_fn, &mut merge_fn_code, &rule_proof_var);
            let merge_fn_code_str = merge_fn_code.join("\n");
            let mut updated = child_names.clone();
            updated.push(merge_fn_var.clone());
            let term_and_proof = self.update_view(name, &updated, &rule_proof_var);
            let fresh_constructor = self.egraph.parser.symbol_gen.fresh("mergecleanup");
            let fresh_sort = self.egraph.parser.symbol_gen.fresh("mergecleanup");
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
        let merge_rule = self.handle_merge_fn(fdecl);
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
        let view_sorts = format!("{in_sorts} {out_type}",);
        let proof_constructors = self.proof_functions(fdecl, &view_sorts);

        // the term table has child_sorts as inputs
        // the view table has child_sorts + the leader term for the eclass
        self.parse_program(&format!(
            "
            (sort {fresh_sort})
            (constructor {name} ({term_sorts}) {})
            (constructor {view_name} ({view_sorts}) {fresh_sort})
            (constructor {to_delete_name} ({in_sorts}) {fresh_sort})
            (constructor {subsumed_name} ({in_sorts}) {fresh_sort})
            {proof_constructors}
            {merge_rule}
            {delete_rule}",
            if fdecl.subtype == FunctionSubtype::Constructor {
                schema.output.clone()
            } else {
                fresh_sort.clone()
            },
        ))
    }

    /// A view proof for functions proves ... = t by some justification, where t is the term of the view row.
    /// A constructor view is more complex, representing f(c1, c2, c3, t_r) where t_r is a representative.
    /// A proof for a view proves that t_r = f(c1, c2, c3).
    fn view_proof_name(&self, name: &str) -> String {
        format!("{}ViewProof", name,)
    }

    fn proof_header(&mut self) -> String {
        let (
            proof_list_sort,
            ast_sort,
            justification_datatype,
            proof_datatype,
            fiat_constructor,
            rule_constructor,
            merge_fn_constructor,
            eq_constructor,
            prim_refl_constructor,
            eq_trans_constructor,
            eq_sym_constructor,
            congr_constructor,
        ) = {
            let names = self.proof_names();
            (
                names.proof_list_sort.clone(),
                names.ast_sort.clone(),
                names.justification_datatype.clone(),
                names.proof_datatype.clone(),
                names.fiat_constructor.clone(),
                names.rule_constructor.clone(),
                names.merge_fn_constructor.clone(),
                names.eq_constructor.clone(),
                names.prim_refl_constructor.clone(),
                names.eq_trans_constructor.clone(),
                names.eq_sym_constructor.clone(),
                names.congr_constructor.clone(),
            )
        };

        format!(
            "
(sort {proof_list_sort})
(sort {ast_sort}) ;; wrap sorts in this for proofs

(datatype {justification_datatype}
    ({fiat_constructor})
    ({rule_constructor}
        String ;; rule name
        {proof_list_sort} ;; proofs for body)
    ({merge_fn_constructor}
        String ;; function name
        {proof_datatype} ;; proof for old term
        {proof_datatype} ;; proof for new term)
        )


;; prove a grounded equality between two terms
;; proof by refl is not allowed- must be justified by fiat or rule
(datatype {proof_datatype}
    ;; proves a term is equal to another
    ({eq_constructor} {justification_datatype} {ast_sort} {ast_sort})

    ({prim_refl_constructor} {ast_sort}) ;; primitives require no proof for reflexivity

    ({eq_trans_constructor} {proof_datatype} {proof_datatype})
    ({eq_sym_constructor} {proof_datatype})
    ;; given a proof that t1 = f(..., c1, ...) and a term f(..., c2, ...)
    ;; and the child index of c1 in the term f(..., c1, ...)
    ;; and a proof that c1 = c2,
    ;; produces a proof that t1 = f(..., c2, ...)
    ({congr_constructor} {proof_datatype} i64 {proof_datatype})
    )
                "
        )
    }

    fn proof_functions(&mut self, fdecl: &ResolvedFunctionDecl, view_sorts: &str) -> String {
        if !self.egraph.proof_state.proofs_enabled {
            return "".to_string();
        }

        let view_proof_name = self.view_proof_name(&fdecl.name);
        let proof_type = self.proof_names().proof_datatype.clone();

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
                let pf_var = self.fresh_var();
                let eq_trans_constructor = self.proof_names().eq_trans_constructor.clone();
                let congr_constructor = self.proof_names().congr_constructor.clone();

                // if we are updating the last element of a constructor
                // it's updating the representative term, use transitivity
                (
                    if fdecl.subtype == FunctionSubtype::Constructor && i == types.len() - 1 {
                        format!(
                            "(let {pf_var}
                               ({eq_trans_constructor}
                                  {updated_child_prf}
                                  {view_prf}))",
                        )
                    } else {
                        // otherwise we are updating a child via congruence
                        format!(
                            "(let {pf_var}
                           ({congr_constructor} {view_prf} {i}
                                  {updated_child_prf}))
                    ",
                        )
                    },
                    pf_var,
                )
            } else {
                ("".to_string(), "".to_string())
            };
            let updated_view = self.update_view(&fdecl.name, &children_updated, &pf_var);

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

    fn fresh_var(&mut self) -> String {
        self.egraph.parser.symbol_gen.fresh("v")
    }

    // Actions need to be instrumented to add to the view
    // as well as to the terms tables.
    fn instrument_action(&mut self, action: &ResolvedAction, justification: &str) -> Vec<String> {
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

    /// Return some code adding to the view and term tables,
    /// returning a let-bound variable for the created term.
    /// Expects the term arguments as `args`.
    fn add_term_and_view(
        &mut self,
        func_type: &FuncType,
        args: &[String],
        justification: &str,
    ) -> (Vec<String>, String) {
        let fv = self.fresh_var();
        let mut res = vec![];
        res.push(format!(
            "(let {fv} ({} {}))",
            func_type.name,
            ListDisplay(args, " ")
        ));

        if func_type.subtype == FunctionSubtype::Constructor {
            res.push(format!(
                "({} {} {fv})",
                self.view_name(&func_type.name),
                ListDisplay(args, " ")
            ));
        } else {
            res.push(format!(
                "({} {})",
                self.view_name(&func_type.name),
                ListDisplay(args, " ")
            ));
        }

        // in proof mode, give a justification for the view and term addition
        if self.egraph.proof_state.proofs_enabled {
            let proof_name = self.view_proof_name(&func_type.name);
            res.push(format!(
                "(set ({proof_name} {}) {justification})",
                ListDisplay(args, " ")
            ));
        }

        (res, fv)
    }

    /// Update the view, with arguments including the eclass for constructors.
    fn update_view(&mut self, fname: &str, args: &[String], pf: &str) -> String {
        let mut res = vec![];
        res.push(format!(
            "({} {})",
            self.view_name(fname),
            ListDisplay(args, " "),
        ));

        if self.egraph.proof_state.proofs_enabled {
            let proof_name = self.view_proof_name(fname);
            res.push(format!(
                "(set ({proof_name} {}) {pf})",
                ListDisplay(args, " ")
            ));
        }
        res.join("\n")
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
        justification: &str,
    ) -> String {
        match expr {
            ResolvedExpr::Lit(_, lit) => format!("{}", lit),
            ResolvedExpr::Var(_, resolved_var) => resolved_var.name.clone(),
            ResolvedExpr::Call(_, resolved_call, args) => {
                let args = args
                    .iter()
                    .map(|arg| self.instrument_action_expr(arg, res, justification))
                    .collect::<Vec<_>>();
                match resolved_call {
                    ResolvedCall::Func(func_type) => {
                        if func_type.subtype == FunctionSubtype::Custom {
                            panic!(
                                "Found a function lookup in actions, should have been prevented by typechecking"
                            );
                        }
                        let (add_code, fv) =
                            self.add_term_and_view(func_type, &args, justification);
                        res.extend(add_code);

                        // add to uf table to initialize eclass for eq sorts
                        if func_type.output.is_eq_sort() {
                            self.union(func_type.output.name(), &fv, &fv, justification);
                        }

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
    fn instrument_actions(&mut self, actions: &[ResolvedAction], rule_proof: &str) -> Vec<String> {
        let mut res = vec![];
        for action in actions {
            res.extend(self.instrument_action(action, rule_proof));
        }
        res
    }

    fn instrument_rule(&mut self, rule: &ResolvedRule) -> Vec<Command> {
        let facts = self.instrument_facts(&rule.body);
        // TODO get rule proof
        let fiat = self.fiat_justification();
        let actions = self.instrument_actions(&rule.head.0, &fiat);
        let name = &rule.name;
        let instrumented = format!(
            "(rule ({} )
                   ({} )
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
        let single_parent = self.single_parent_ruleset_name();
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
                let fiat = self.fiat_justification();
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

    fn parent_direct_ruleset_name(&self) -> String {
        "parent".to_string()
    }

    fn rebuilding_ruleset_name(&self) -> String {
        "rebuilding".to_string()
    }

    fn rebuilding_cleanup_ruleset_name(&self) -> String {
        "rebuilding_cleanup".to_string()
    }

    pub(crate) fn term_header(&mut self) -> Vec<Command> {
        let str = format!(
            "(ruleset {})
             (ruleset {})
             (ruleset {})
             (ruleset {})
             (ruleset {})",
            self.parent_direct_ruleset_name(),
            self.single_parent_ruleset_name(),
            self.rebuilding_ruleset_name(),
            self.rebuilding_cleanup_ruleset_name(),
            self.delete_subsume_ruleset_name(),
        );
        self.parse_program(&str)
    }

    fn parse_program(&mut self, input: &str) -> Vec<Command> {
        self.egraph.parser.ensure_no_reserved_symbols = false;
        let res = self.egraph.parser.get_program_from_string(None, input);
        self.egraph.parser.ensure_no_reserved_symbols = true;

        res.unwrap()
    }

    fn parse_schedule(&mut self, input: String) -> Schedule {
        self.egraph.parser.ensure_no_reserved_symbols = false;
        let res = self.egraph.parser.get_schedule_from_string(None, &input);
        self.egraph.parser.ensure_no_reserved_symbols = true;
        res.unwrap()
    }

    fn parse_facts(&mut self, input: &[String]) -> Vec<Fact> {
        self.egraph.parser.ensure_no_reserved_symbols = false;
        let res = input
            .iter()
            .map(|f| self.egraph.parser.get_fact_from_string(None, f).unwrap())
            .collect();
        self.egraph.parser.ensure_no_reserved_symbols = true;
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
