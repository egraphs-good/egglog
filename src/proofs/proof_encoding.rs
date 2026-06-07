#[doc = include_str!("proof_encoding.md")]
use crate::proofs::proof_encoding_helpers::{
    EncodingNames, Justification, proof_container_constructor,
};
use crate::typechecking::FuncType;
use crate::*;

// TODO refactor so that encoding state is optional on the e-graph, ProofNames not optional on EncodingState. Then we don't have to clone proof names everywhere.
#[derive(Clone)]
pub(crate) struct EncodingState {
    pub uf_parent: HashMap<String, String>,
    pub uf_function: HashMap<String, String>,
    /// Maps sort name -> proof function name (set from :internal-proof-func annotation).
    pub proof_func_parent: HashMap<String, String>,
    pub term_header_added: bool,
    // TODO this is very ugly- we should separate out a typechecking struct
    // since we didn't need an entire e-graph
    // When Some term encoding is enabled.
    pub original_typechecking: Option<Box<EGraph>>,
    pub proofs_enabled: bool,
    pub proof_testing: bool,
    pub proof_names: EncodingNames,
}

impl EncodingState {
    pub(crate) fn new(symbol_gen: &mut SymbolGen) -> Self {
        Self {
            uf_parent: HashMap::default(),
            uf_function: HashMap::default(),
            proof_func_parent: HashMap::default(),
            term_header_added: false,
            original_typechecking: None,
            proofs_enabled: false,
            proof_names: EncodingNames::new(symbol_gen),
            proof_testing: false,
        }
    }
}

/// Thin wrapper around an [`EGraph`] for the term encoding
pub(crate) struct ProofInstrumentor<'a> {
    pub(crate) egraph: &'a mut EGraph,
}

impl<'a> ProofInstrumentor<'a> {
    /// Make a term state and use it to instrument the code.
    pub(crate) fn add_term_encoding(
        egraph: &'a mut EGraph,
        program: Vec<ResolvedNCommand>,
    ) -> Vec<Command> {
        Self { egraph }.add_term_encoding_helper(program)
    }

    /// Mark two things as equal, adding proof if proofs are enabled.
    pub(crate) fn union(
        &mut self,
        type_name: &str,
        lhs: &str,
        rhs: &str,
        justification: &Justification,
    ) -> String {
        let uf_name = self.uf_name(type_name);
        let smaller = format!("(ordering-min {lhs} {rhs})");
        let larger = format!("(ordering-max {lhs} {rhs})");
        let proof = if self.egraph.proof_state.proofs_enabled {
            let to_ast_constructor = self
                .proof_names()
                .sort_to_ast_constructor
                .get(type_name)
                .unwrap();
            let rule_constructor = &self.proof_names().rule_constructor;
            let fiat_constructor = &self.proof_names().fiat_constructor;
            match justification {
                Justification::Rule(rule_name, proof_list) => format!(
                    "({rule_constructor} \"{rule_name}\" {proof_list} ({to_ast_constructor} {larger}) ({to_ast_constructor} {smaller}))"
                ),
                Justification::Fiat => format!(
                    "({fiat_constructor} ({to_ast_constructor} {larger}) ({to_ast_constructor} {smaller}))"
                ),
                Justification::Merge(_func_name, _proof1, _proof2) => panic!(
                    "Merge functions do not include union actions, so proof should not be by merge"
                ),
                Justification::Proof(existing_proof) => existing_proof.clone(),
            }
        } else {
            "()".to_string()
        };
        format!("(set ({uf_name} {larger} {smaller}) {proof})")
    }

    /// The parent table is the database representation of a union-find datastructure.
    /// When one term has two parents, those parents are unioned in the merge action.
    /// Also, we have a rule that maintains the invariant that each term points to its
    /// canonical representative.
    fn declare_sort(&mut self, sort_name: &str) -> Vec<Command> {
        let pname = self.uf_name(sort_name);
        let uf_function_name = self.uf_function_name(sort_name);
        let fresh_name = self.egraph.parser.symbol_gen.fresh("uf_update");
        let uf_function_index_name = self.egraph.parser.symbol_gen.fresh("uf_function_index");

        let path_compress_ruleset_name = self.proof_names().path_compress_ruleset_name.clone();
        let single_parent_ruleset_name = self.proof_names().single_parent_ruleset_name.clone();
        let uf_function_index_ruleset_name =
            self.proof_names().uf_function_index_ruleset_name.clone();

        let proof_type = self.proof_type_str().to_string();

        // In proof mode, path compression composes proofs via Trans/Sym.
        // In term mode, the proof output is Unit and we just write ().
        let (path_compress_query, path_compress_action, single_parent_query, single_parent_action) =
            if self.egraph.proof_state.proofs_enabled {
                let p1_fresh = self.egraph.parser.symbol_gen.fresh("p1");
                let p2_fresh = self.egraph.parser.symbol_gen.fresh("p2");
                let trans = self.proof_names().eq_trans_constructor.clone();
                let sym = self.proof_names().eq_sym_constructor.clone();
                (
                    format!(
                        "(= {p1_fresh} ({pname} a b))
                        (= {p2_fresh} ({pname} b c))"
                    ),
                    format!(
                        "(delete ({pname} a b))
                       (set ({pname} a c) ({trans} {p1_fresh} {p2_fresh}))"
                    ),
                    format!(
                        "(= {p1_fresh} ({pname} a b))
                        (= {p2_fresh} ({pname} a c))"
                    ),
                    format!(
                        "(delete ({pname} a b))
                       (set ({pname} b c) ({trans} ({sym} {p1_fresh}) {p2_fresh}))"
                    ),
                )
            } else {
                (
                    format!("({pname} a b)\n                        ({pname} b c)"),
                    format!(
                        "(delete ({pname} a b))\n                       (set ({pname} a c) ())"
                    ),
                    format!("({pname} a b)\n                        ({pname} a c)"),
                    format!(
                        "(delete ({pname} a b))\n                       (set ({pname} b c) ())"
                    ),
                )
            };

        // In proof mode, UF function index stores (leader, proof) pairs.
        // In term mode, it just stores the leader.
        let (uf_function_output_type, uf_pair_sort_decl, uf_index_query, uf_index_action) =
            if self.egraph.proof_state.proofs_enabled {
                let pair_sort = self.uf_pair_sort_name(sort_name);
                let proof_fresh = self.egraph.parser.symbol_gen.fresh("uf_idx_proof");
                (
                    pair_sort.clone(),
                    format!("(sort {pair_sort} (Pair {sort_name} {proof_type}))"),
                    format!("(= {proof_fresh} ({pname} a b))"),
                    format!("(set ({uf_function_name} a) (pair b {proof_fresh}))"),
                )
            } else {
                (
                    sort_name.to_string(),
                    "".to_string(),
                    format!("({pname} a b)"),
                    format!("(set ({uf_function_name} a) b)"),
                )
            };

        let mut code = format!(
            "{uf_pair_sort_decl}
             (function {pname} ({sort_name} {sort_name}) {proof_type} :merge old :internal-hidden)
             (function {uf_function_name} ({sort_name}) {uf_function_output_type} :merge new :unextractable :internal-hidden)
             ;; performs path compression, ensuring each term points to the representative
             (rule ({path_compress_query}
                    (!= b c))
                  ({path_compress_action})
                   :ruleset {path_compress_ruleset_name}
                   :name \"{fresh_name}\")
             ;; ensures each term has only one parent
             (rule ({single_parent_query}
                    (!= b c)
                    (= (ordering-max b c) b))
                  ({single_parent_action})
                   :ruleset {single_parent_ruleset_name}
                   :name \"singleparent{fresh_name}\")
             ;; mirrors UF rows into a function-backed UF index for faster rebuild lookups
             (rule ({uf_index_query})
                   ({uf_index_action})
                   :ruleset {uf_function_index_ruleset_name}
                   :name \"{uf_function_index_name}\")
                   "
        );

        if self.egraph.proof_state.proofs_enabled {
            let term_proof_name = self.term_proof_name(sort_name);
            let add_to_ast_code = self.add_to_ast(sort_name);
            code = format!(
                "{add_to_ast_code}
                 (function {term_proof_name} ({sort_name}) {proof_type} :merge old :internal-hidden)
                 {code}"
            );
        }

        self.parse_program(&code)
    }

    /// Rules that execute deletion and subsumption based on the tables requesting the deletion/subsumption.
    fn delete_and_subsume(&mut self, fdecl: &ResolvedFunctionDecl) -> String {
        let child_names = fdecl
            .schema
            .input
            .iter()
            .enumerate()
            .map(|(i, _)| format!("c{i}_"))
            .collect::<Vec<_>>()
            .join(" ");
        let to_delete_name = self.delete_name(&fdecl.name);
        let subsumed_name = self.subsumed_name(&fdecl.name);
        let view_name = self.view_name(&fdecl.name);
        let delete_subsume_ruleset = self.proof_names().delete_subsume_ruleset_name.clone();
        let fresh_name = self.egraph.parser.symbol_gen.fresh("delete_rule");

        format!(
            "(rule (({to_delete_name} {child_names})
                    ({view_name} {child_names} out))
                   ((delete ({view_name} {child_names} out))
                    (delete ({to_delete_name} {child_names})))
                    :ruleset {delete_subsume_ruleset}
                    :name \"{fresh_name}\")
             (rule (({subsumed_name} {child_names})
                    ({view_name} {child_names} out))
                   ((subsume ({view_name} {child_names} out)))
                    :ruleset {delete_subsume_ruleset}
                    :name \"{fresh_name}_subsume\")"
        )
    }

    /// Generate rules that run a merge function for a custom function.
    /// One rule runs the merge function when two different values are present for the same children.
    /// Another rule cleans up old values, necessary because the newly merged value may be equal to one of the old values.
    fn handle_merge_fn(
        &mut self,
        fdecl: &ResolvedFunctionDecl,
        child_names: &[String],
        child_names_str: &str,
        _view_name: &str,
        rebuilding_ruleset: &str,
    ) -> String {
        let name = &fdecl.name;

        let merge_fn = &fdecl
            .merge
            .as_ref()
            .unwrap_or_else(|| panic!("Proofs don't support :no-merge"));

        let fresh_name = self.egraph.parser.symbol_gen.fresh("merge_rule");
        let cleanup_name = self.egraph.parser.symbol_gen.fresh("merge_cleanup");

        let p1_fresh = self.egraph.parser.symbol_gen.fresh("p1");
        let p2_fresh = self.egraph.parser.symbol_gen.fresh("p2");
        let view_name = self.view_name(&fdecl.name);
        let rebuilding_cleanup_ruleset = self.proof_names().rebuilding_cleanup_ruleset_name.clone();
        let proof_query = if self.egraph.proof_state.proofs_enabled {
            // View is a function with proof output; bind proof variables
            format!(
                "(= {p1_fresh} ({view_name} {child_names_str} old))
                     (= {p2_fresh} ({view_name} {child_names_str} new))
                    "
            )
        } else {
            // View is a function with Unit output; no need to bind the output
            "".to_string()
        };
        let proof_var = if self.egraph.proof_state.proofs_enabled {
            self.fresh_var()
        } else {
            "()".to_string()
        };
        let mut merge_fn_code = vec![];
        let merge_fn_var = self.instrument_action_expr(
            merge_fn,
            &mut merge_fn_code,
            &Justification::Merge(name.clone(), p1_fresh.clone(), p2_fresh.clone()),
        );
        let merge_fn_code_str = merge_fn_code.join("\n");
        let mut updated = child_names.to_vec();
        updated.push(merge_fn_var.clone());
        let term = format!("({name} {child_names_str} {merge_fn_var})");

        let rule_proof = if self.egraph.proof_state.proofs_enabled {
            let to_ast = self.fname_to_ast_name(name);
            let merge_fn_constructor = self.proof_names().merge_fn_constructor.clone();
            format!(
                "(let {proof_var}
                            ({merge_fn_constructor} \"{name}\"
                                  {p1_fresh}
                                  {p2_fresh}
                                  ({to_ast} {term})))"
            )
        } else {
            "".to_string()
        };
        let term_and_proof = self.update_view(name, &updated, &proof_var);
        let cleanup_constructor = self.egraph.parser.symbol_gen.fresh("mergecleanup");
        let fresh_sort = self.egraph.parser.symbol_gen.fresh("mergecleanupsort");
        let output_sort = fdecl.schema.output.clone();

        // The first runs the merge function adding a new row.
        // The second deletes rows with old values for the old variable, while the third deletes rows with new values for the new variable.
        format!(
            "(sort {fresh_sort})
                 (constructor {cleanup_constructor} ({output_sort} {output_sort}) {fresh_sort} :internal-hidden)
                 (rule (({view_name} {child_names_str} old)
                        ({view_name} {child_names_str} new)
                        (!= old new)
                        (= (ordering-max old new) new)
                        {proof_query})
                       (
                        {merge_fn_code_str}
                        {rule_proof}
                        {term_and_proof}
                        ({cleanup_constructor} {merge_fn_var} old)
                        ({cleanup_constructor} {merge_fn_var} new)
                       )
                        :ruleset {rebuilding_ruleset}
                        :name \"{fresh_name}\")
                
                 (rule (({cleanup_constructor} merged old)
                        ({view_name} {child_names_str} merged)
                        ({view_name} {child_names_str} old)
                        (!= merged old))
                       ((delete ({view_name} {child_names_str} old)))
                        :ruleset {rebuilding_cleanup_ruleset}
                        :name \"{cleanup_name}\")
                ",
        )
    }

    /// Generate a rule that handles congruence for constructor-like proof views.
    /// When two different values are present for the same children,
    /// we union those two values together.
    fn handle_congruence(
        &mut self,
        view_key: &str,
        output_sort: &str,
        child_names: &[String],
        rebuilding_ruleset: &str,
        fresh_prefix: &str,
    ) -> String {
        let fresh_name = self.egraph.parser.symbol_gen.fresh(fresh_prefix);
        let mut child_names_new = child_names.to_vec();
        child_names_new.push("new".to_string());
        let mut child_names_old = child_names.to_vec();
        child_names_old.push("old".to_string());
        let (query1, prf1) = self.query_view_and_get_proof(view_key, &child_names_new);
        let (query2, prf2) = self.query_view_and_get_proof(view_key, &child_names_old);
        let sym = &self.proof_names().eq_sym_constructor;
        let trans = &self.proof_names().eq_trans_constructor;

        // Proof is by transitivity. A view proof gives a proof that
        // representative r_1 = f(c_1,...,c_n).
        // We also have a proof that other eclass representative r_2 = f(c_1,...,c_n), the same term.
        // We want a proof that r1 = r2, which we get by transitivity.
        let union_code = self.union(
            output_sort,
            "new",
            "old",
            &Justification::Proof(format!("({trans} {prf1} ({sym} {prf2}))")),
        );
        format!(
            "(rule ({query1}
                        {query2}
                        (!= old new)
                        (= (ordering-max old new) new))
                       ({union_code})
                        :ruleset {rebuilding_ruleset}
                        :name \"{fresh_name}\")"
        )
    }

    /// Generate rules that handle merge functions or congruence.
    /// For custom functions, we generate rules that run the merge function.
    /// For constructors, we generate congruence rules.
    fn handle_merge_or_congruence(&mut self, fdecl: &ResolvedFunctionDecl) -> String {
        let child_names = fdecl
            .schema
            .input
            .iter()
            .enumerate()
            .map(|(i, _)| format!("c{i}_"))
            .collect::<Vec<_>>();
        let child_names_str = child_names.join(" ");
        let rebuilding_ruleset = self.proof_names().rebuilding_ruleset_name.clone();
        let view_name = self.view_name(&fdecl.name);
        if fdecl.subtype == FunctionSubtype::Custom {
            self.handle_merge_fn(
                fdecl,
                &child_names,
                &child_names_str,
                &view_name,
                &rebuilding_ruleset,
            )
        } else {
            self.handle_congruence(
                &fdecl.name,
                &fdecl.schema.output,
                &child_names,
                &rebuilding_ruleset,
                "congruence_rule",
            )
        }
    }

    fn is_proof_canonicalizable_sort(&self, sort: &ArcSort) -> bool {
        sort.is_eq_sort() || (sort.is_eq_container_sort() && sort.container_proof_spec().is_some())
    }

    fn rebuilding_rules_for_view(
        &mut self,
        types: &[ArcSort],
        view_key: &str,
        fresh_prefix: &str,
        representative_column: Option<usize>,
    ) -> Vec<Command> {
        if !types
            .iter()
            .any(|sort| self.is_proof_canonicalizable_sort(sort))
        {
            return vec![];
        }

        let children_vec = (0..types.len())
            .map(|i| format!("c{i}_"))
            .collect::<Vec<_>>();
        let children = format!("{}", ListDisplay(&children_vec, " "));
        let mut uf_queries = vec![];
        let mut children_updated = vec![];
        let mut bool_neq_exprs = vec![];
        let mut uf_proof_vars = vec![];

        for (i, ty) in types.iter().enumerate() {
            let ci = &children_vec[i];
            if self.is_proof_canonicalizable_sort(ty) {
                let leader_var = format!("c{i}_leader_");
                let uf_function_name = self.uf_function_name(ty.name());

                if self.egraph.proof_state.proofs_enabled {
                    // UF function index returns a Pair(leader, proof); one lookup gives both
                    let pair_var = self.fresh_var();
                    let proof_var = format!("(pair-second {pair_var})");
                    uf_queries.push(format!(
                        "(= {pair_var} ({uf_function_name} {ci}))
                         (= {leader_var} (pair-first {pair_var}))"
                    ));
                    uf_proof_vars.push(Some(proof_var));
                } else {
                    uf_queries.push(format!("(= {leader_var} ({uf_function_name} {ci}))"));
                    uf_proof_vars.push(None);
                }

                bool_neq_exprs.push(format!("(bool-!= {ci} {leader_var})"));
                children_updated.push(leader_var);
            } else {
                children_updated.push(ci.clone());
                uf_proof_vars.push(None);
            }
        }

        let uf_query_str = uf_queries.join("\n       ");
        let filter_query = format!("(guard (or {}))", bool_neq_exprs.join("\n             "));

        let view_name = self.view_name(view_key);
        let fresh_name = self.egraph.parser.symbol_gen.fresh(fresh_prefix);
        let (query_view, view_prf) = self.query_view_and_get_proof(view_key, &children_vec);
        let (pf_code, pf_var) = if self.egraph.proof_state.proofs_enabled {
            let eq_trans_constructor = self.proof_names().eq_trans_constructor.clone();
            let congr_constructor = self.proof_names().congr_constructor.clone();
            let sym_constructor = self.proof_names().eq_sym_constructor.clone();
            let mut current_proof = view_prf;
            let mut proof_code_parts = vec![];

            for (i, ty) in types.iter().enumerate() {
                if !self.is_proof_canonicalizable_sort(ty) {
                    continue;
                }

                let uf_prf = uf_proof_vars[i].as_ref().unwrap();
                let new_proof = self.fresh_var();
                if Some(i) == representative_column {
                    proof_code_parts.push(format!(
                        "(let {new_proof}
                       ({eq_trans_constructor}
                          ({sym_constructor} {uf_prf})
                          {current_proof}))",
                    ));
                } else {
                    proof_code_parts.push(format!(
                        "(let {new_proof}
                          ({congr_constructor} {current_proof} {i}
                                               {uf_prf}))",
                    ));
                }
                current_proof = new_proof;
            }

            (proof_code_parts.join("\n"), current_proof)
        } else {
            ("".to_string(), "()".to_string())
        };
        let updated_view = self.update_view(view_key, &children_updated, &pf_var);

        let rule = format!(
            "(rule ({query_view}
                    {uf_query_str}
                    {filter_query}
                    )
                 (
                  {pf_code}
                  {updated_view}
                  (delete ({view_name} {children}))
                 )
                  :ruleset {} :name \"{fresh_name}\")",
            self.proof_names().rebuilding_ruleset_name
        );
        self.parse_program(&rule)
    }

    fn proof_for_value(&mut self, sort: &ArcSort, value: &str, res: &mut Vec<String>) -> String {
        if !self.egraph.proof_state.proofs_enabled {
            "()".to_string()
        } else if sort.is_eq_sort() || sort.container_proof_spec().is_some() {
            let term_proof_name = self.term_proof_name(sort.name());
            let fresh_proof = self.fresh_var();
            res.push(format!("(= {fresh_proof} ({term_proof_name} {value}))"));
            fresh_proof
        } else {
            let fiat_constructor = &self.proof_names().fiat_constructor;
            let to_ast = self
                .proof_names()
                .sort_to_ast_constructor
                .get(sort.name())
                .unwrap();
            format!("({fiat_constructor} ({to_ast} {value}) ({to_ast} {value}))")
        }
    }

    fn query_fact_view(
        &mut self,
        view_key: &str,
        mut view_args: Vec<String>,
        arg_proofs: Vec<Option<String>>,
        res: &mut Vec<String>,
    ) -> (String, String) {
        let fv = self.fresh_var();
        view_args.push(fv.clone());
        let (query, view_proof_var) = self.query_view_and_get_proof(view_key, &view_args);
        res.push(query);
        if self.proofs_enabled() {
            let mut proof = view_proof_var;
            for (i, arg_proof) in arg_proofs.into_iter().enumerate() {
                if let Some(arg_proof) = arg_proof {
                    let congr = &self.proof_names().congr_constructor;
                    proof = format!(
                        "
                            ({congr} {proof} {i} {arg_proof})
                            "
                    );
                }
            }
            (fv, proof)
        } else {
            (fv, "()".to_string())
        }
    }

    fn container_view_key(constructor_name: &str, output_sort: &str) -> String {
        format!("{constructor_name}_{output_sort}")
    }

    fn proof_container_encoding(&mut self, sort: &ArcSort) -> Vec<Command> {
        let Some(spec) = sort.container_proof_spec() else {
            return vec![];
        };

        spec.constructors
            .iter()
            .flat_map(|constructor| {
                let view_key = Self::container_view_key(constructor.name, sort.name());
                let view_name = self.view_name(&view_key);
                let mut view_sorts = constructor
                    .input_sorts
                    .iter()
                    .map(|sort| sort.name().to_string())
                    .collect::<Vec<_>>();
                view_sorts.push(sort.name().to_string());

                let child_names = constructor
                    .input_sorts
                    .iter()
                    .enumerate()
                    .map(|(i, _)| format!("c{i}_"))
                    .collect::<Vec<_>>();
                let rebuilding_ruleset = self.proof_names().rebuilding_ruleset_name.clone();
                let congruence_rule = self.handle_congruence(
                    &view_key,
                    sort.name(),
                    &child_names,
                    &rebuilding_ruleset,
                    "container_congruence_rule",
                );
                let to_ast_view_sort = self.add_to_ast(sort.name());
                let proof_type = self.proof_type_str().to_string();

                let mut commands = self.parse_program(&format!(
                    "{to_ast_view_sort}
                     (function {view_name} ({}) {proof_type} :merge old :internal-term-constructor {} :internal-hidden :unextractable)
                     {congruence_rule}",
                    ListDisplay(view_sorts, " "),
                    constructor.name,
                ));
                let mut types = constructor.input_sorts.clone();
                types.push(sort.clone());
                commands.extend(self.rebuilding_rules_for_view(
                    &types,
                    &view_key,
                    "container_rebuild_rule",
                    Some(types.len() - 1),
                ));
                commands
            })
            .collect()
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
        let to_delete_name = self.delete_name(&fdecl.name);
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
        let to_ast_view_sort = self.add_to_ast(&view_sort);

        if self.egraph.proof_state.proofs_enabled {
            self.egraph
                .proof_state
                .proof_names
                .fn_to_term_sort
                .insert(name.clone(), view_sort.clone());
        }
        let merge_rule = self.handle_merge_or_congruence(fdecl);
        // the term table has child_sorts as inputs
        // the view table has child_sorts + the leader term for the eclass
        // Propagate cost, unextractable, hidden, and internal_let flags from the original function
        let mut term_flags = String::new();
        if let Some(cost) = fdecl.cost {
            term_flags.push_str(&format!(" :cost {cost}"));
        }
        // View is always a function (returning Proof or Unit), with :merge old
        let proof_type = self.proof_type_str().to_string();
        let mut view_flags = String::new();
        if fdecl.unextractable {
            view_flags.push_str(" :unextractable");
        }
        if fdecl.internal_hidden {
            view_flags.push_str(" :internal-hidden");
        }
        if fdecl.internal_let {
            view_flags.push_str(" :internal-let");
        }
        let view_decl = format!(
            "(function {view_name} ({view_sorts}) {proof_type} :merge old :internal-term-constructor {name}{view_flags})"
        );
        self.parse_program(&format!(
            "
            (sort {fresh_sort})
            {to_ast_view_sort}
            (constructor {name} ({term_sorts}) {view_sort}{term_flags} :internal-hidden :unextractable)
            {view_decl}
            (constructor {to_delete_name} ({in_sorts}) {fresh_sort} :internal-hidden)
            (constructor {subsumed_name} ({in_sorts}) {fresh_sort} :internal-hidden)
            {proof_constructors}
            {merge_rule}
            {delete_rule}",
        ))
    }

    fn proof_functions(&mut self, _fdecl: &ResolvedFunctionDecl, _view_sorts: &str) -> String {
        // ViewProof is now merged into the view table as its output column
        "".to_string()
    }

    /// Rules that update the views when children change.
    fn rebuilding_rules(&mut self, fdecl: &ResolvedFunctionDecl) -> Vec<Command> {
        let types = fdecl.resolved_schema.view_types();
        let representative_column =
            (fdecl.subtype == FunctionSubtype::Constructor).then_some(types.len() - 1);
        self.rebuilding_rules_for_view(&types, &fdecl.name, "rebuild_rule", representative_column)
    }

    /// Instrument fact replaces terms with looking up
    /// canonical versions in the view.
    /// It also needs to look up references to globals.
    /// Adds the instrumented fact to `res` and returns a proof that the fact matched.
    fn instrument_fact(&mut self, fact: &ResolvedFact, res: &mut Vec<String>) -> String {
        match fact {
            // In proof normal form, this is the only way that function calls apppear.
            ResolvedFact::Eq(
                _span,
                ResolvedExpr::Call(
                    _span2,
                    head @ ResolvedCall::Func(FuncType {
                        subtype: FunctionSubtype::Custom,
                        ..
                    }),
                    args,
                ),
                // TODO this could actually be arbitrary pretty easily, it's just nested functions that are hard.
                ResolvedExpr::Var(_span3, v),
            ) => {
                let mut new_args = vec![];
                let mut arg_proofs = vec![];
                for arg in args {
                    let (var, proof) = self.instrument_fact_expr(arg, res);
                    new_args.push(var);
                    arg_proofs.push(proof);
                }
                new_args.push(v.to_string());

                let view_name = self.view_name(head.name());
                let args_str = ListDisplay(new_args, " ");

                // View is always a function; query it and bind the output
                let proof_var = self.fresh_var();
                res.push(format!("(= {proof_var} ({view_name} {args_str}))"));

                if self.egraph.proof_state.proofs_enabled {
                    let mut proof = proof_var;
                    for (i, arg_proof) in arg_proofs.into_iter().enumerate() {
                        let congr = &self.proof_names().congr_constructor;
                        proof = format!(
                            "
                            ({congr} {proof} {i} {arg_proof})
                            "
                        );
                    }
                    proof
                } else {
                    "()".to_string()
                }
            }
            ResolvedFact::Eq(_span, left_expr, right_expr) => {
                let (v1, p1) = self.instrument_fact_expr(left_expr, res);
                let (v2, p2) = self.instrument_fact_expr(right_expr, res);
                res.push(format!("(= {v1} {v2})"));
                let sym = &self.proof_names().eq_sym_constructor;
                let trans = &self.proof_names().eq_trans_constructor;

                format!("({trans} ({sym} {p1}) {p2})",)
            }
            ResolvedFact::Fact(generic_expr) => {
                let (_, proof) = self.instrument_fact_expr(generic_expr, res);
                proof
            }
        }
    }

    /// Instruments a fact expression to use the view tables.
    /// Assumes there are no function lookups in the term.
    /// Returns a variable representing the expression and a proof that the expression was matched.
    /// Proves a ground equality t1 = t2 where t1 is the eclass representative and t2 matches `expr` syntactically.
    fn instrument_fact_expr(
        &mut self,
        expr: &ResolvedExpr,
        res: &mut Vec<String>,
    ) -> (String, String) {
        match expr {
            ResolvedExpr::Lit(_, lit) => {
                let proof_code = if self.egraph.proof_state.proofs_enabled {
                    let fiat_constructor = &self.proof_names().fiat_constructor;
                    let lit_sort = literal_sort(lit);
                    let to_ast = self
                        .proof_names()
                        .sort_to_ast_constructor
                        .get(lit_sort.name())
                        .unwrap();
                    format!("({fiat_constructor} ({to_ast} {lit}) ({to_ast} {lit}))")
                } else {
                    "()".to_string()
                };

                (format!("{lit}"), proof_code)
            }
            ResolvedExpr::Var(_, resolved_var) => {
                let var = &resolved_var.name;
                (
                    resolved_var.name.clone(),
                    self.proof_for_value(&resolved_var.sort, var, res),
                )
            }
            ResolvedExpr::Call(_, resolved_call, args) => {
                let mut new_args = vec![];
                // Variables and constants don't need subproofs, but constructor calls do.
                let mut arg_proofs: Vec<Option<String>> = vec![];
                for arg in args {
                    if matches!(arg, ResolvedExpr::Var(_, _) | ResolvedExpr::Lit(_, _)) {
                        new_args.push(arg.to_string());
                        arg_proofs.push(None);
                    } else {
                        let (arg_str, proof) = self.instrument_fact_expr(arg, res);
                        new_args.push(arg_str);
                        arg_proofs.push(Some(proof));
                    }
                }
                match resolved_call {
                    ResolvedCall::Func(func_type) => {
                        if func_type.subtype == FunctionSubtype::Custom {
                            return self.query_fact_view(
                                &func_type.name,
                                new_args,
                                arg_proofs,
                                res,
                            );
                        }

                        assert!(
                            func_type.subtype == FunctionSubtype::Constructor,
                            "Only constructor function calls are allowed in fact expressions due to proof normal form. Got {func_type:?}",
                        );

                        self.query_fact_view(&func_type.name, new_args, arg_proofs, res)
                    }
                    ResolvedCall::Primitive(specialized_primitive) => {
                        if let Some(constructor) =
                            proof_container_constructor(specialized_primitive)
                        {
                            let output_sort = specialized_primitive.output().name();
                            let view_key = Self::container_view_key(constructor.name, output_sort);
                            return self.query_fact_view(&view_key, new_args, arg_proofs, res);
                        }

                        let projection = (|| {
                            if specialized_primitive.input().len() != 1 {
                                return None;
                            }
                            let container_sort = specialized_primitive.input()[0].clone();
                            let spec = container_sort.container_proof_spec()?;
                            let (constructor, projection) =
                                spec.projection(specialized_primitive.name())?;
                            let field_index = projection.field;
                            let field_sort = constructor.input_sorts.get(field_index)?.clone();
                            (field_sort.name() == specialized_primitive.output().name())
                                .then_some((constructor.clone(), container_sort, field_index))
                        })();
                        if let Some((constructor, container_sort, field_index)) = projection {
                            let Some(container_value) = new_args.first() else {
                                panic!("container projection primitive missing container argument");
                            };
                            let mut field_vars = constructor
                                .input_sorts
                                .iter()
                                .map(|_| self.fresh_var())
                                .collect::<Vec<_>>();
                            let selected = field_vars[field_index].clone();
                            field_vars.push(container_value.clone());
                            let view_key =
                                Self::container_view_key(constructor.name, container_sort.name());
                            let (query, _view_proof_var) =
                                self.query_view_and_get_proof(&view_key, &field_vars);
                            res.push(query);
                            let proof = self.proof_for_value(
                                specialized_primitive.output(),
                                &selected,
                                res,
                            );
                            return (selected, proof);
                        }

                        if specialized_primitive.output().is_eq_sort() {
                            panic!(
                                "Term encoding does not support eq-sort primitive expressions in facts"
                            );
                        }
                        let fv = self.fresh_var();
                        res.push(format!(
                            "(= {fv} ({} {}))",
                            specialized_primitive.name(),
                            ListDisplay(new_args, " ")
                        ));

                        let proof = if self.proofs_enabled() {
                            let fiat_constructor = &self.proof_names().fiat_constructor;
                            let to_ast = self
                                .proof_names()
                                .sort_to_ast_constructor
                                .get(specialized_primitive.output().name())
                                .unwrap();
                            format!("({fiat_constructor} ({to_ast} {fv}) ({to_ast} {fv}))")
                        } else {
                            "()".to_string()
                        };

                        (fv.clone(), proof)
                    }
                }
            }
        }
    }

    /// Return a new query and a proof that the query matched.
    fn instrument_facts(&mut self, facts: &[ResolvedFact]) -> (Vec<String>, String) {
        let mut res = vec![];
        let mut proof = vec![];

        for fact in facts.iter() {
            let f_proof = self.instrument_fact(fact, &mut res);
            proof.push(f_proof);
        }

        (res, self.format_prooflist(&proof))
    }

    // Actions need to be instrumented to add to the view
    // as well as to the terms tables.
    fn instrument_action(
        &mut self,
        action: &ResolvedAction,
        justification: &Justification,
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
                        Change::Delete => self.delete_name(&func_type.name),
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
                res.push(format!("{action}"));
            }
            ResolvedAction::Expr(_span, generic_expr) => {
                self.instrument_action_expr(generic_expr, &mut res, justification);
            }
        }

        res
    }

    /// Update the view with the given arguments.
    /// The arguments include the eclass for constructors.
    /// View is always a function (returning Proof or Unit).
    fn update_view(&mut self, fname: &str, args: &[String], proof: &str) -> String {
        let view_name = self.view_name(fname);
        format!("(set ({view_name} {}) {proof})", ListDisplay(args, " "))
    }

    /// Return some code adding to the view and term tables.
    /// For constructors, `args` should not include the eclass of the resulting term (since it may not exist yet).
    /// For custom functions, `args` should include all arguments (including the output for the function).
    ///
    /// Returns a vector of strings representing code to add and a variable for the created term.
    /// We could return the term itself, but this might make the encoding blow up the code.
    fn add_term_and_view(
        &mut self,
        func_type: &FuncType,
        args: &[String],
        justification: &Justification,
    ) -> (Vec<String>, String) {
        let to_ast = self
            .egraph
            .proof_state
            .proofs_enabled
            .then(|| self.fname_to_ast_name(&func_type.name).to_string());
        let canonical_sort =
            (func_type.subtype == FunctionSubtype::Constructor).then(|| func_type.output.name());
        self.add_created_term_and_view(
            &func_type.name,
            args,
            &func_type.name,
            to_ast.as_deref(),
            canonical_sort,
            justification,
        )
    }

    fn add_created_term_and_view(
        &mut self,
        term_name: &str,
        args: &[String],
        view_key: &str,
        to_ast: Option<&str>,
        canonical_sort: Option<&str>,
        justification: &Justification,
    ) -> (Vec<String>, String) {
        let fv = self.fresh_var();
        let mut res = vec![];
        // TODO might be able to get rid of this intermediate variable in encoding
        res.push(format!(
            "(let {fv} ({term_name} {}))",
            ListDisplay(args, " ")
        ));

        let view_args = if canonical_sort.is_some() {
            let mut a = args.to_vec();
            a.push(fv.clone());
            a
        } else {
            args.to_vec()
        };

        let (proof_str, view_proof_var) = if self.egraph.proof_state.proofs_enabled {
            let to_ast = to_ast.expect("proof AST constructor missing for created term");
            let rule_constructor = &self.proof_names().rule_constructor;
            let fiat_constructor = &self.proof_names().fiat_constructor;
            let proof = match justification {
                Justification::Rule(rule_name, rule_proof) => {
                    format!(
                        "({rule_constructor} \"{rule_name}\" {rule_proof} ({to_ast} {fv}) ({to_ast} {fv}))",
                    )
                }
                Justification::Fiat => {
                    format!("({fiat_constructor} ({to_ast} {fv}) ({to_ast} {fv}))",)
                }
                Justification::Merge(fn_name, p1, p2) => {
                    let merge_constructor = &self.proof_names().merge_fn_constructor;
                    format!("({merge_constructor} \"{fn_name}\" {p1} {p2} ({to_ast} {fv}))",)
                }
                Justification::Proof(existing_proof) => existing_proof.clone(),
            };

            let proof_var = self.fresh_var();
            let term_proof = if let Some(sort) = canonical_sort {
                let term_proof_constructor = self.term_proof_name(sort);
                format!("(set ({term_proof_constructor} {fv}) {proof_var})")
            } else {
                "".to_string()
            };

            (
                format!(
                    "(let {proof_var} {proof})
                     {term_proof}"
                ),
                proof_var,
            )
        } else {
            ("".to_string(), "()".to_string())
        };

        res.push(proof_str);
        res.push(self.update_view(view_key, &view_args, &view_proof_var));

        // add to uf table to initialize eclass for constructors
        if let Some(sort) = canonical_sort {
            res.push(self.union(sort, &fv, &fv, &Justification::Proof(view_proof_var)));
        }

        (res, fv)
    }

    /// Returns a query for (fname args) and a variable for the proof (or Unit) output.
    /// View is always a function, so we always use `(= var (view ...))` form.
    fn query_view_and_get_proof(&mut self, fname: &str, args: &[String]) -> (String, String) {
        let view_name = self.view_name(fname);
        let pf_var = self.fresh_var();
        let query = format!("(= {pf_var} ({view_name} {}))", ListDisplay(args, " "));
        (query, pf_var)
    }

    // Add to view and term tables, returning a variable for the created term.
    fn instrument_action_expr(
        &mut self,
        expr: &ResolvedExpr,
        res: &mut Vec<String>,
        proof: &Justification,
    ) -> String {
        match expr {
            ResolvedExpr::Lit(_, lit) => format!("{lit}"),
            ResolvedExpr::Var(_, resolved_var) => resolved_var.name.clone(),
            ResolvedExpr::Call(_, resolved_call, args) => {
                let args = args
                    .iter()
                    .map(|arg| self.instrument_action_expr(arg, res, proof))
                    .collect::<Vec<_>>();
                match resolved_call {
                    ResolvedCall::Func(func_type) => {
                        if func_type.subtype == FunctionSubtype::Custom {
                            // Globals are desugared to no-arg functions (in non-proof mode)
                            // They're allowed, in proof mode they are constructors.
                            if self.egraph.type_info.is_global(&func_type.name) {
                                return format!("({} {})", func_type.name, ListDisplay(&args, " "));
                            }
                            panic!(
                                "Found a function lookup in actions, should have been prevented by typechecking"
                            );
                        }
                        let (add_code, fv) = self.add_term_and_view(func_type, &args, proof);
                        res.extend(add_code);

                        fv
                    }
                    ResolvedCall::Primitive(specialized_primitive) => {
                        if let Some(constructor) =
                            proof_container_constructor(specialized_primitive)
                        {
                            let output_sort = specialized_primitive.output().name();
                            let to_ast = self.egraph.proof_state.proofs_enabled.then(|| {
                                self.proof_names()
                                    .sort_to_ast_constructor
                                    .get(output_sort)
                                    .unwrap()
                                    .clone()
                            });
                            let view_key = Self::container_view_key(constructor.name, output_sort);
                            let (add_code, fv) = self.add_created_term_and_view(
                                specialized_primitive.name(),
                                &args,
                                &view_key,
                                to_ast.as_deref(),
                                Some(output_sort),
                                proof,
                            );
                            res.extend(add_code);
                            return fv;
                        }

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
        justification: &Justification,
    ) -> Vec<String> {
        let mut res = vec![];
        for action in actions {
            res.extend(self.instrument_action(action, justification));
        }
        res
    }

    /// Instrument a rule to use term encoding. This involves using the view tables in facts,
    /// adding to term and view tables in actions.
    /// When proofs are enabled we query proof tables, then build a proof for the rule in the actions.
    /// Finally, each view update also updates the proof tables.
    fn instrument_rule(&mut self, rule: &ResolvedRule) -> Vec<Command> {
        let (facts, proof_str) = self.instrument_facts(&rule.body);
        let proof_var = self.fresh_var();
        let proof = Justification::Rule(rule.name.clone(), proof_var.clone());
        let proof_var_binding = if self.egraph.proof_state.proofs_enabled {
            format!(
                "(let {proof_var}
                          {proof_str})"
            )
        } else {
            "".to_string()
        };

        let actions = self.instrument_actions(&rule.head.0, &proof);
        let name = &rule.name;
        let instrumented = format!(
            "(rule ({})
                   ({proof_var_binding}
                    {})
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

    /// Any schedule should be sound as long as we saturate.
    fn rebuild(&mut self) -> Schedule {
        let path_compress_ruleset = self.proof_names().path_compress_ruleset_name.clone();
        let single_parent = self.proof_names().single_parent_ruleset_name.clone();
        let uf_function_index = self.proof_names().uf_function_index_ruleset_name.clone();
        let rebuilding_cleanup_ruleset = self.proof_names().rebuilding_cleanup_ruleset_name.clone();
        let rebuilding_ruleset = self.proof_names().rebuilding_ruleset_name.clone();
        let delete_ruleset = self.proof_names().delete_subsume_ruleset_name.clone();
        self.parse_schedule(format!(
            "(seq
              (saturate
                  {rebuilding_cleanup_ruleset}
                  (saturate {single_parent})
                  (saturate {path_compress_ruleset})
                  (saturate {uf_function_index})
                  {rebuilding_ruleset})
              {delete_ruleset})"
        ))
    }

    fn instrument_schedule(&mut self, schedule: &ResolvedSchedule) -> Schedule {
        match schedule {
            ResolvedSchedule::Run(span, config) => {
                let new_run = match config.until {
                    Some(ref facts) => {
                        let (instrumented, _proof) = self.instrument_facts(facts);
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
        log::debug!("Term encoding for {command}");
        match &command {
            ResolvedNCommand::Sort {
                span,
                name,
                presort_and_args,
                unionable,
                ..
            } => {
                let uf_name = self.uf_name(name);
                let proof_func = if self.egraph.proof_state.proofs_enabled {
                    Some(self.term_proof_name(name))
                } else {
                    None
                };
                res.push(Command::Sort {
                    span: span.clone(),
                    name: name.clone(),
                    presort_and_args: presort_and_args.clone(),
                    uf: Some(uf_name),
                    proof_func,
                    unionable: *unionable,
                });
                res.extend(self.declare_sort(name));
                let container_sort = self
                    .egraph
                    .proof_state
                    .original_typechecking
                    .as_ref()
                    .and_then(|egraph| egraph.type_info.get_sort_by_name(name))
                    .or_else(|| self.egraph.type_info.get_sort_by_name(name))
                    .cloned();
                if let Some(sort) = container_sort {
                    res.extend(self.proof_container_encoding(&sort));
                }
            }
            ResolvedNCommand::Function(fdecl) => {
                res.extend(self.term_and_view(fdecl));
                res.extend(self.rebuilding_rules(fdecl));
            }
            ResolvedNCommand::NormRule { rule } => {
                res.extend(self.instrument_rule(rule));
            }
            ResolvedNCommand::CoreAction(action) => {
                let instrumented = self
                    .instrument_action(action, &Justification::Fiat)
                    .join("\n");
                res.extend(self.parse_program(&instrumented));
            }
            ResolvedNCommand::Check(span, facts) => {
                let (instrumented, _proof) = self.instrument_facts(facts);
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
            ResolvedNCommand::Extract(span, expr, variants) => {
                // Instrument the expressions to use view tables (like actions, not facts)
                let mut action_stmts = vec![];
                let instrumented_expr =
                    self.instrument_action_expr(expr, &mut action_stmts, &Justification::Fiat);
                let instrumented_variants =
                    self.instrument_action_expr(variants, &mut action_stmts, &Justification::Fiat);

                // Add any action statements needed to set up the expressions
                for stmt in action_stmts {
                    res.extend(self.parse_program(&stmt));
                }
                // Rebuild before extract; we may have added new view rows that need canonicalization
                res.push(Command::RunSchedule(self.rebuild()));
                res.push(Command::Extract(
                    span.clone(),
                    self.parse_expr(&instrumented_expr),
                    self.parse_expr(&instrumented_variants),
                ));
            }
            ResolvedNCommand::PrintSize(span, name) => {
                // In proof mode, print the size of the view table for constructors
                let new_name = name.as_ref().map(|n| {
                    if self
                        .egraph
                        .type_info
                        .get_func_type(n)
                        .is_some_and(|f| f.subtype == FunctionSubtype::Constructor)
                    {
                        self.view_name(n)
                    } else {
                        n.clone()
                    }
                });
                res.push(Command::PrintSize(span.clone(), new_name));
            }
            ResolvedNCommand::Pop(..)
            | ResolvedNCommand::Push(..)
            | ResolvedNCommand::AddRuleset(..)
            | ResolvedNCommand::Output { .. }
            | ResolvedNCommand::Input { .. }
            | ResolvedNCommand::UnstableCombinedRuleset(..)
            | ResolvedNCommand::PrintOverallStatistics(..)
            | ResolvedNCommand::PrintFunction(..)
            | ResolvedNCommand::ProveExists(..) => {
                res.push(command.to_command().make_unresolved());
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
            | ResolvedNCommand::Sort { .. } = &command
            {
            } else {
                res.push(Command::RunSchedule(self.rebuild()));
            }
        }

        res
    }
}
