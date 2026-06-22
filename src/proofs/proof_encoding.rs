#[doc = include_str!("proof_encoding.md")]
use crate::exec_state::{Internal, RegistrySealed};
use crate::proofs::proof_encoding_helpers::{EncodingNames, Justification};
use crate::typechecking::FuncType;
use crate::*;

// TODO refactor so that encoding state is optional on the e-graph, ProofNames not optional on EncodingState. Then we don't have to clone proof names everywhere.
#[derive(Clone)]
pub(crate) struct EncodingState {
    pub uf_parent: HashMap<String, String>,
    pub uf_function: HashMap<String, String>,
    /// Maps sort name -> proof function name (set from :internal-proof-func annotation).
    pub proof_func_parent: HashMap<String, String>,
    /// Maps container sort name -> the name of its registered container-rebuild
    /// primitive (see [`ContainerRebuild`]). Cached so each container sort gets
    /// a single rebuild primitive shared across all functions using it.
    pub container_rebuild_name: HashMap<String, String>,
    /// Maps container sort name -> the name of its registered proof-producing
    /// container-rebuild primitive (see [`ContainerRebuildProof`]). Proof mode only.
    pub container_rebuild_proof_name: HashMap<String, String>,
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
            container_rebuild_name: HashMap::default(),
            container_rebuild_proof_name: HashMap::default(),
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
    fn declare_sort(&mut self, sort_name: &str, is_container: bool) -> Vec<Command> {
        // Container sorts are never unioned directly; their values are
        // canonicalized structurally by the container rebuild path (see
        // `rebuilding_rules`). So they get no per-sort union-find tables or
        // maintenance rules. In proof mode they still get a `<Sort>Proof`
        // term-proof table (and an `Ast<Sort>` wrapper) used as the reflexive
        // anchor for container rebuild proofs.
        if is_container {
            if self.egraph.proof_state.proofs_enabled {
                let term_proof_name = self.term_proof_name(sort_name);
                let add_to_ast_code = self.add_to_ast(sort_name);
                let proof_type = self.proof_type_str().to_string();
                return self.parse_program(&format!(
                    "{add_to_ast_code}
                     (function {term_proof_name} ({sort_name}) {proof_type} :merge old :internal-hidden)"
                ));
            }
            return vec![];
        }
        let pname = self.uf_name(sort_name);
        let uf_function_name = self.uf_function_name(sort_name);
        let fresh_name = self.egraph.parser.symbol_gen.fresh("uf_update");
        let uf_function_index_name = self.egraph.parser.symbol_gen.fresh("uf_function_index");
        // Fresh query-variable names for the UF maintenance rules. These must be
        // gensym'd (not literal `a`/`b`/`c`) so they cannot shadow a user-defined
        // global constructor of the same name when the encoded program is
        // re-parsed (e.g. a program with a `(constructor b () ...)`).
        let a = self.egraph.parser.symbol_gen.fresh("uf_a");
        let b = self.egraph.parser.symbol_gen.fresh("uf_b");
        let c = self.egraph.parser.symbol_gen.fresh("uf_c");

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
                        "(= {p1_fresh} ({pname} {a} {b}))
                        (= {p2_fresh} ({pname} {b} {c}))"
                    ),
                    format!(
                        "(delete ({pname} {a} {b}))
                       (set ({pname} {a} {c}) ({trans} {p1_fresh} {p2_fresh}))"
                    ),
                    format!(
                        "(= {p1_fresh} ({pname} {a} {b}))
                        (= {p2_fresh} ({pname} {a} {c}))"
                    ),
                    format!(
                        "(delete ({pname} {a} {b}))
                       (set ({pname} {b} {c}) ({trans} ({sym} {p1_fresh}) {p2_fresh}))"
                    ),
                )
            } else {
                (
                    format!("({pname} {a} {b})\n                        ({pname} {b} {c})"),
                    format!(
                        "(delete ({pname} {a} {b}))\n                       (set ({pname} {a} {c}) ())"
                    ),
                    format!("({pname} {a} {b})\n                        ({pname} {a} {c})"),
                    format!(
                        "(delete ({pname} {a} {b}))\n                       (set ({pname} {b} {c}) ())"
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
                    format!("(= {proof_fresh} ({pname} {a} {b}))"),
                    format!("(set ({uf_function_name} {a}) (pair {b} {proof_fresh}))"),
                )
            } else {
                (
                    sort_name.to_string(),
                    "".to_string(),
                    format!("({pname} {a} {b})"),
                    format!("(set ({uf_function_name} {a}) {b})"),
                )
            };

        let mut code = format!(
            "{uf_pair_sort_decl}
             (function {pname} ({sort_name} {sort_name}) {proof_type} :merge old :internal-hidden)
             (function {uf_function_name} ({sort_name}) {uf_function_output_type} :merge new :unextractable :internal-hidden)
             ;; performs path compression, ensuring each term points to the representative
             (rule ({path_compress_query}
                    (!= {b} {c}))
                  ({path_compress_action})
                   :ruleset {path_compress_ruleset_name}
                   :name \"{fresh_name}\")
             ;; ensures each term has only one parent
             (rule ({single_parent_query}
                    (!= {b} {c})
                    (= (ordering-max {b} {c}) {b}))
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

    /// Generate a rule that handles congruence for constructors.
    /// When two different values are present for the same children,
    /// we union those two values together.
    fn handle_congruence(
        &mut self,
        fdecl: &ResolvedFunctionDecl,
        child_names: &[String],
        rebuilding_ruleset: &str,
    ) -> String {
        // Congruence rule
        let fresh_name = self.egraph.parser.symbol_gen.fresh("congruence_rule");
        let mut child_names_new = child_names.to_vec();
        child_names_new.push("new".to_string());
        let mut child_names_old = child_names.to_vec();
        child_names_old.push("old".to_string());
        let (query1, prf1) = self.query_view_and_get_proof(&fdecl.name, &child_names_new);
        let (query2, prf2) = self.query_view_and_get_proof(&fdecl.name, &child_names_old);
        let sym = &self.proof_names().eq_sym_constructor;
        let trans = &self.proof_names().eq_trans_constructor;

        // Proof is by transitivity. A view proof gives a proof that
        // representative r_1 = f(c_1,...,c_n).
        // We also have a proof that other eclass representative r_2 = f(c_1,...,c_n), the same term.
        // We want a proof that r1 = r2, which we get by transitivity.
        let union_code = self.union(
            &fdecl.schema.output,
            "new",
            "old",
            &Justification::Proof(format!("({trans} {prf1} ({sym} {prf2}))",)),
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
            self.handle_congruence(fdecl, &child_names, &rebuilding_ruleset)
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
        let proofs_enabled = self.egraph.proof_state.proofs_enabled;

        // Check if there are any rebuildable columns at all; if not, no rule needed.
        // Container columns (eq-container sorts) are rebuilt by calling a
        // per-container rebuild primitive; eq-sort columns by a UF lookup.
        if !types
            .iter()
            .any(|t| t.is_eq_sort() || t.is_eq_container_sort())
        {
            return vec![];
        }

        let view_name = self.view_name(&fdecl.name);
        let child = |i: usize| format!("c{i}_");
        let children_vec: Vec<String> = (0..types.len()).map(child).collect();
        let children = format!("{}", ListDisplay(&children_vec, " "));

        // For each rebuildable column, compute its canonical value (and, in
        // proof mode, a proof that the original equals the canonical value).
        // Non-rebuildable columns keep their original value and have no proof.
        let mut uf_queries = vec![];
        let mut leader_vars: Vec<String> = vec![];
        let mut bool_neq_exprs = vec![];
        let mut uf_proof_vars: Vec<Option<String>> = vec![];
        // Action prologue for container columns (proof mode): bind each
        // container rebuild proof and set a reflexive anchor for the rebuilt
        // container value so it can itself be rebuilt later.
        let mut container_proof_bindings: Vec<String> = vec![];
        let mut container_reflexive_sets: Vec<String> = vec![];
        // Whether any container column is present (forces a `:naive` rule, since
        // the rebuild primitives read the UF index / mint proofs).
        let mut has_container = false;

        for (i, ty) in types.iter().enumerate() {
            if ty.is_eq_sort() {
                let leader_var = format!("c{i}_leader_");
                let uf_function_name = self.uf_function_name(ty.name());
                let ci = child(i);

                if proofs_enabled {
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
                leader_vars.push(leader_var);
            } else if ty.is_eq_container_sort() {
                // Container column: canonicalize its elements via the per-container
                // rebuild primitive in the rule body. The rule is `:naive` so this
                // read primitive is permitted there.
                has_container = true;
                let rebuilt_var = format!("c{i}_rebuilt_");
                let ci = child(i);
                let value_prim = self.ensure_container_rebuild(ty);
                uf_queries.push(format!("(= {rebuilt_var} ({value_prim} {ci}))"));
                bool_neq_exprs.push(format!("(bool-!= {ci} {rebuilt_var})"));
                leader_vars.push(rebuilt_var.clone());

                if proofs_enabled {
                    // The proof primitive (run in the action) mints a congruence
                    // proof that `ci = rebuilt`. Anchor a reflexive proof on the
                    // rebuilt value via `Trans(Sym p, p)` for future rebuilds.
                    let proof_prim = self.ensure_container_rebuild_proof(ty);
                    let proof_var = self.fresh_var();
                    let cproof = self.term_proof_name(ty.name());
                    let sym = self.proof_names().eq_sym_constructor.clone();
                    let trans = self.proof_names().eq_trans_constructor.clone();
                    container_proof_bindings.push(format!("(let {proof_var} ({proof_prim} {ci}))"));
                    container_reflexive_sets.push(format!(
                        "(set ({cproof} {rebuilt_var}) ({trans} ({sym} {proof_var}) {proof_var}))"
                    ));
                    uf_proof_vars.push(Some(proof_var));
                } else {
                    uf_proof_vars.push(None);
                }
            } else {
                leader_vars.push(child(i));
                uf_proof_vars.push(None);
            }
        }

        let uf_query_str = uf_queries.join("\n       ");
        let or_expr = format!("(or {})", bool_neq_exprs.join("\n             "));
        let filter_query = format!("(guard {or_expr})");

        // Build the updated children: use leader_var for rebuildable columns, original for others.
        let children_updated: Vec<String> = leader_vars.clone();

        let fresh_name = self.egraph.parser.symbol_gen.fresh("rebuild_rule");
        let (query_view, view_prf) = self.query_view_and_get_proof(&fdecl.name, &children_vec);

        // Build proof code if proofs are enabled.
        // We chain congruence proofs for each updated child and a transitivity proof
        // for the representative (last column) update.
        let (pf_code, pf_var) = if proofs_enabled {
            let eq_trans_constructor = self.proof_names().eq_trans_constructor.clone();
            let congr_constructor = self.proof_names().congr_constructor.clone();
            let sym_constructor = self.proof_names().eq_sym_constructor.clone();

            // Start with the view proof and apply congruence for each rebuilt child
            // (using transitivity instead for the representative column of a constructor).
            let mut current_proof = view_prf.clone();
            let mut proof_code_parts = vec![];

            for i in 0..types.len() {
                let Some(uf_prf) = uf_proof_vars[i].clone() else {
                    continue;
                };

                if types[i].is_eq_sort()
                    && fdecl.subtype == FunctionSubtype::Constructor
                    && i == types.len() - 1
                {
                    // Updating the representative term (last column of constructor):
                    // use transitivity with sym of the UF proof
                    let new_proof = self.fresh_var();
                    proof_code_parts.push(format!(
                        "(let {new_proof}
                           ({eq_trans_constructor}
                              ({sym_constructor} {uf_prf})
                              {current_proof}))",
                    ));
                    current_proof = new_proof;
                } else {
                    // Updating a child (eq-sort or container) via congruence
                    let new_proof = self.fresh_var();
                    proof_code_parts.push(format!(
                        "(let {new_proof}
                              ({congr_constructor} {current_proof} {i}
                                                   {uf_prf}))",
                    ));
                    current_proof = new_proof;
                }
            }

            (proof_code_parts.join("\n"), current_proof)
        } else {
            ("".to_string(), "()".to_string())
        };

        let updated_view = self.update_view(&fdecl.name, &children_updated, &pf_var);
        let container_proof_bindings_str = container_proof_bindings.join("\n");
        let container_reflexive_sets_str = container_reflexive_sets.join("\n");

        // Make a single rule that updates the view when any child's leader differs.
        let naive = if has_container { " :naive" } else { "" };
        let rule = format!(
            "(rule ({query_view}
                    {uf_query_str}
                    {filter_query}
                    )
                 (
                  {container_proof_bindings_str}
                  {pf_code}
                  {updated_view}
                  {container_reflexive_sets_str}
                  (delete ({view_name} {children}))
                 ){naive}
                  :ruleset {} :name \"{fresh_name}\")",
            self.proof_names().rebuilding_ruleset_name
        );
        self.parse_program(&rule)
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
                    if !self.egraph.proof_state.proofs_enabled {
                        "()".to_string()
                    } else if resolved_var.sort.is_eq_sort() {
                        let term_proof_name = self.term_proof_name(resolved_var.sort.name());
                        let fresh_proof = self.fresh_var();
                        res.push(format!("(= {fresh_proof} ({term_proof_name} {var}))"));
                        fresh_proof
                    } else {
                        let fiat_constructor = &self.proof_names().fiat_constructor;
                        let lit_sort = resolved_var.sort.name();
                        let to_ast = self
                            .proof_names()
                            .sort_to_ast_constructor
                            .get(lit_sort)
                            .unwrap();
                        format!("({fiat_constructor} ({to_ast} {var}) ({to_ast} {var}))")
                    },
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
                        assert!(
                            func_type.subtype == FunctionSubtype::Constructor,
                            "Only constructor function calls are allowed in fact expressions due to proof normal form. Got {func_type:?}",
                        );

                        let fv = self.fresh_var();
                        let view_name = self.view_name(&func_type.name);
                        let args_str = ListDisplay(new_args, " ");

                        let proof = {
                            // View is always a function; query it and bind the output
                            let view_proof_var = self.fresh_var();
                            res.push(format!(
                                "(= {view_proof_var} ({view_name} {args_str} {fv}))"
                            ));
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
                                proof
                            } else {
                                "()".to_string()
                            }
                        };
                        (fv, proof)
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
                            ListDisplay(new_args, " ")
                        ));

                        let proof = if !self.proofs_enabled() {
                            "()".to_string()
                        } else if specialized_primitive.output().is_eq_container_sort() {
                            // A container term is an `App`, not a literal, so a
                            // `Fiat` reflexive proof would be rejected by the
                            // checker (not a global). Instead look up the
                            // container's `<CSort>Proof` reflexive anchor, the
                            // same way eq-sort terms use their term-proof table.
                            let cproof =
                                self.term_proof_name(specialized_primitive.output().name());
                            let fresh_proof = self.fresh_var();
                            res.push(format!("(= {fresh_proof} ({cproof} {fv}))"));
                            fresh_proof
                        } else {
                            // Base primitives produce a literal result; a
                            // reflexive `Fiat` over a literal is checker-valid.
                            let fiat_constructor = &self.proof_names().fiat_constructor;
                            let to_ast = self
                                .proof_names()
                                .sort_to_ast_constructor
                                .get(specialized_primitive.output().name())
                                .unwrap();
                            format!("({fiat_constructor} ({to_ast} {fv}) ({to_ast} {fv}))")
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

    /// Build the proof term that justifies a freshly-created term `fv`
    /// (wrapped by the AST constructor `to_ast`) proving `fv = fv`, from the
    /// surrounding [`Justification`]. Shared by constructor creation
    /// (`add_term_and_view`) and container creation.
    fn term_proof_for_justification(
        &self,
        fv: &str,
        to_ast: &str,
        justification: &Justification,
    ) -> String {
        let rule_constructor = &self.proof_names().rule_constructor;
        let fiat_constructor = &self.proof_names().fiat_constructor;
        match justification {
            Justification::Rule(rule_name, rule_proof) => format!(
                "({rule_constructor} \"{rule_name}\" {rule_proof} ({to_ast} {fv}) ({to_ast} {fv}))"
            ),
            Justification::Fiat => {
                format!("({fiat_constructor} ({to_ast} {fv}) ({to_ast} {fv}))")
            }
            Justification::Merge(fn_name, p1, p2) => {
                let merge_constructor = &self.proof_names().merge_fn_constructor;
                format!("({merge_constructor} \"{fn_name}\" {p1} {p2} ({to_ast} {fv}))")
            }
            Justification::Proof(existing_proof) => existing_proof.clone(),
        }
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
        // A fresh variable for the new term.
        let fv = self.fresh_var();
        let mut res = vec![];
        // TODO might be able to get rid of this intermediate variable in encoding
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

        let (proof_str, view_proof_var) = if self.egraph.proof_state.proofs_enabled {
            let to_ast = self.fname_to_ast_name(&func_type.name);
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
            // add a proof for the constructor if needed
            let term_proof = if func_type.subtype == FunctionSubtype::Constructor {
                let term_proof_constructor = self.term_proof_name(func_type.output.name());
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
        res.push(self.update_view(&func_type.name, &args_with_fv, &view_proof_var));

        // add to uf table to initialize eclass for constructors
        if func_type.subtype == FunctionSubtype::Constructor {
            res.push(self.union(
                func_type.output.name(),
                &fv,
                &fv,
                &Justification::Proof(view_proof_var),
            ));
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
                        let fv = self.fresh_var();
                        res.push(format!(
                            "(let {} ({} {}))",
                            fv,
                            specialized_primitive.name(),
                            ListDisplay(&args, " ")
                        ));
                        // In proof mode, a primitive that builds a container
                        // value records a reflexive term-proof in `<CSort>Proof`.
                        // This is the anchor for the container's rebuild
                        // congruence proofs (see `rebuilding_rules`).
                        if self.egraph.proof_state.proofs_enabled {
                            let out = specialized_primitive.output();
                            if out.is_eq_container_sort() {
                                let csort = out.name().to_string();
                                let to_ast = self
                                    .proof_names()
                                    .sort_to_ast_constructor
                                    .get(&csort)
                                    .unwrap()
                                    .clone();
                                let proof_str =
                                    self.term_proof_for_justification(&fv, &to_ast, proof);
                                let cproof = self.term_proof_name(&csort);
                                res.push(format!("(set ({cproof} {fv}) {proof_str})"));
                            }
                        }
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
                // After the proof-encoding gate, any sort carrying a presort
                // is one of the supported container sorts. Containers have no
                // per-sort union-find (they are canonicalized structurally),
                // so they get `uf: None` and `find_canonical` leaves their
                // value unchanged during extraction.
                let is_container = presort_and_args.is_some();
                let uf_name = if is_container {
                    None
                } else {
                    Some(self.uf_name(name))
                };
                let proof_func = if self.egraph.proof_state.proofs_enabled && !is_container {
                    Some(self.term_proof_name(name))
                } else {
                    None
                };
                // For container sorts, build the rebuild-primitive spec now (it
                // generates and caches the fresh primitive names used by the
                // rebuild rules below) and attach it as an annotation so the
                // primitives can be re-registered when this desugared Sort
                // command is typechecked / re-parsed.
                let container_rebuild = if is_container {
                    let container_sort = self
                        .egraph
                        .proof_state
                        .original_typechecking
                        .as_ref()
                        .and_then(|tc| tc.get_sort_by_name(name).cloned())
                        .unwrap_or_else(|| {
                            panic!("container sort {name} not found while term-encoding")
                        });
                    Some(self.build_container_rebuild_spec(&container_sort))
                } else {
                    None
                };
                res.push(Command::Sort {
                    span: span.clone(),
                    name: name.clone(),
                    presort_and_args: presort_and_args.clone(),
                    uf: uf_name,
                    proof_func,
                    unionable: *unionable,
                    container_rebuild,
                });
                res.extend(self.declare_sort(name, is_container));
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

    /// Collect, transitively, the `UF_<E>f` index table name for every eq-sort
    /// reachable from `sort`'s elements, recursing through nested containers.
    fn collect_container_uf_names(
        &mut self,
        sort: &ArcSort,
        uf_names: &mut HashMap<String, String>,
    ) {
        for elem in sort.inner_sorts() {
            if elem.is_eq_sort() {
                if !uf_names.contains_key(elem.name()) {
                    let uf = self.uf_function_name(elem.name());
                    uf_names.insert(elem.name().to_string(), uf);
                }
            } else if elem.is_eq_container_sort() {
                self.collect_container_uf_names(&elem, uf_names);
            }
        }
    }

    /// Collect, transitively, the `<CSort>Proof` table name for `sort` and every
    /// nested container sort reachable from its elements.
    fn collect_container_proof_names(
        &mut self,
        sort: &ArcSort,
        cproof_names: &mut HashMap<String, String>,
    ) {
        if !cproof_names.contains_key(sort.name()) {
            let cp = self.term_proof_name(sort.name());
            cproof_names.insert(sort.name().to_string(), cp);
        }
        for elem in sort.inner_sorts() {
            if elem.is_eq_container_sort() {
                self.collect_container_proof_names(&elem, cproof_names);
            }
        }
    }

    /// Build the rebuild-primitive spec for a container sort, returning the
    /// serialized spec (to attach as the `:internal-container-rebuild`
    /// annotation). Generates and caches the fresh primitive names but does
    /// *not* register the primitives — registration happens from the spec when
    /// the Sort command is typechecked (see [`register_container_rebuild_from_spec`]),
    /// so it works both during encoding and on re-parse of the desugared output.
    ///
    /// The spec is a space-separated token list:
    /// `value_prim proof_flag uf_count (elemsort ufname)* [proof_prim congr trans sym axiom proof_datatype cp_count (csort cproof)*]`
    fn build_container_rebuild_spec(&mut self, container_sort: &ArcSort) -> String {
        let sort_name = container_sort.name().to_string();
        let proof_mode = self.egraph.proof_state.proofs_enabled;

        let value_prim = self.egraph.parser.symbol_gen.fresh("container_rebuild");
        self.egraph
            .proof_state
            .container_rebuild_name
            .insert(sort_name.clone(), value_prim.clone());

        let mut uf_names = HashMap::default();
        self.collect_container_uf_names(container_sort, &mut uf_names);
        let uf_pairs: Vec<(String, String)> = uf_names.into_iter().collect();

        let mut tokens: Vec<String> = vec![
            value_prim,
            if proof_mode { "1" } else { "0" }.to_string(),
            uf_pairs.len().to_string(),
        ];
        for (es, uf) in &uf_pairs {
            tokens.push(es.clone());
            tokens.push(uf.clone());
        }

        if proof_mode {
            let proof_prim = self
                .egraph
                .parser
                .symbol_gen
                .fresh("container_rebuild_proof");
            self.egraph
                .proof_state
                .container_rebuild_proof_name
                .insert(sort_name, proof_prim.clone());

            let mut cproof_names = HashMap::default();
            self.collect_container_proof_names(container_sort, &mut cproof_names);
            let cp_pairs: Vec<(String, String)> = cproof_names.into_iter().collect();

            let congr = self.proof_names().congr_constructor.clone();
            let trans = self.proof_names().eq_trans_constructor.clone();
            let sym = self.proof_names().eq_sym_constructor.clone();
            let axiom = self.proof_names().container_normalize_constructor.clone();
            let proof_datatype = self.proof_names().proof_datatype.clone();

            tokens.extend([proof_prim, congr, trans, sym, axiom, proof_datatype]);
            tokens.push(cp_pairs.len().to_string());
            for (cs, cp) in &cp_pairs {
                tokens.push(cs.clone());
                tokens.push(cp.clone());
            }
        }

        tokens.join(" ")
    }

    /// The (already-built) container value-rebuild primitive name for a sort.
    fn ensure_container_rebuild(&mut self, container_sort: &ArcSort) -> String {
        self.egraph
            .proof_state
            .container_rebuild_name
            .get(container_sort.name())
            .cloned()
            .unwrap_or_else(|| {
                panic!(
                    "container rebuild primitive not built for sort {}",
                    container_sort.name()
                )
            })
    }

    /// The (already-built) container proof-rebuild primitive name for a sort.
    fn ensure_container_rebuild_proof(&mut self, container_sort: &ArcSort) -> String {
        self.egraph
            .proof_state
            .container_rebuild_proof_name
            .get(container_sort.name())
            .cloned()
            .unwrap_or_else(|| {
                panic!(
                    "container rebuild proof primitive not built for sort {}",
                    container_sort.name()
                )
            })
    }
}

/// Register a container sort's rebuild primitives from a spec built by
/// [`ProofInstrumentor::build_container_rebuild_spec`]. Called when a container
/// Sort command carrying an `:internal-container-rebuild` annotation is
/// typechecked, so the primitives exist before the rebuild rules are
/// typechecked — both during encoding and when the desugared program is
/// re-parsed.
pub(crate) fn register_container_rebuild_from_spec(eg: &mut EGraph, sort_name: &str, spec: &str) {
    let Some(container_sort) = eg.get_sort_by_name(sort_name).cloned() else {
        return;
    };
    let tokens: Vec<&str> = spec.split_whitespace().collect();
    let mut i = 0;
    let next = |i: &mut usize| -> String {
        let t = tokens[*i].to_string();
        *i += 1;
        t
    };
    let value_prim = next(&mut i);
    let proof_mode = next(&mut i) == "1";
    let uf_count: usize = next(&mut i).parse().unwrap();
    let mut uf_names = HashMap::default();
    for _ in 0..uf_count {
        let es = next(&mut i);
        let uf = next(&mut i);
        uf_names.insert(es, uf);
    }
    eg.add_read_primitive(
        ContainerRebuild {
            name: value_prim,
            container_sort: container_sort.clone(),
            uf_names: uf_names.clone(),
            proof_mode,
        },
        None,
    );
    if proof_mode {
        let proof_prim = next(&mut i);
        let congr_name = next(&mut i);
        let trans_name = next(&mut i);
        let sym_name = next(&mut i);
        let container_normalize_name = next(&mut i);
        let proof_datatype = next(&mut i);
        let cp_count: usize = next(&mut i).parse().unwrap();
        let mut cproof_names = HashMap::default();
        for _ in 0..cp_count {
            let cs = next(&mut i);
            let cp = next(&mut i);
            cproof_names.insert(cs, cp);
        }
        let proof_sort: ArcSort = std::sync::Arc::new(EqSort {
            name: proof_datatype,
        });
        eg.add_full_primitive(
            ContainerRebuildProof {
                name: proof_prim,
                container_sort,
                proof_sort,
                uf_names,
                cproof_names,
                congr_name,
                trans_name,
                sym_name,
                container_normalize_name,
            },
            None,
        );
    }
}

/// Recursively canonicalize a container `value` of sort `sort` for the term
/// encoding, returning the rebuilt interned value. Each element is resolved by
/// a uniform per-child rule: an eq-sort element maps to its union-find leader
/// (via `UF_<E>f`; in proof mode the index stores `(pair leader proof)`), a
/// container element is recursively rebuilt, and anything else is unchanged.
fn rebuild_container_value_rec(
    state: &mut ReadState,
    sort: &ArcSort,
    value: Value,
    uf_names: &HashMap<String, String>,
    proof_mode: bool,
) -> Option<Value> {
    let elements = {
        let cvs = state.container_values();
        sort.inner_values(cvs, value)
    };
    let mut leaders: HashMap<Value, Value> = HashMap::default();
    for (esort, eval) in &elements {
        let new = if esort.is_eq_sort() {
            match self_lookup_leader(state, uf_names, esort, *eval, proof_mode)? {
                Some(leader) => leader,
                None => *eval,
            }
        } else if esort.is_eq_container_sort() {
            rebuild_container_value_rec(state, esort, *eval, uf_names, proof_mode)?
        } else {
            *eval
        };
        if new != *eval {
            leaders.insert(*eval, new);
        }
    }
    let cvs = state.container_values();
    let es = state.raw_exec_state();
    Some(sort.rebuild_container_with_leaders(cvs, es, value, &leaders))
}

/// Look up an eq-sort element's union-find leader, if a `UF_<E>f` row exists.
/// In proof mode the index stores `(pair leader proof)`, so take `pair-first`.
/// Returns `Ok(None)` when there is no UF row (element is its own leader).
fn self_lookup_leader(
    state: &mut ReadState,
    uf_names: &HashMap<String, String>,
    esort: &ArcSort,
    eval: Value,
    proof_mode: bool,
) -> Option<Option<Value>> {
    let Some(uf_name) = uf_names.get(esort.name()) else {
        return Some(None);
    };
    let Some(looked_up) = state.lookup(uf_name, &[eval]) else {
        return Some(None);
    };
    let leader = if proof_mode {
        state
            .container_values()
            .get_val::<crate::sort::PairContainer>(looked_up)?
            .first
    } else {
        looked_up
    };
    Some(Some(leader))
}

/// A term-encoding primitive that canonicalizes a container value's elements to
/// their union-find leaders (recursing through nested containers). Registered
/// per container sort by [`ProofInstrumentor::ensure_container_rebuild`] and
/// invoked from the container-column arm of the rebuild rules. It reads the
/// `UF_<E>f` tables, so it is only valid in a `:naive` rule (read-context body).
#[derive(Clone)]
struct ContainerRebuild {
    name: String,
    container_sort: ArcSort,
    /// element-sort name -> `UF_<E>f` table name (all reachable eq-sorts)
    uf_names: HashMap<String, String>,
    /// In proof mode the UF index returns `(pair leader proof)`; take pair-first.
    proof_mode: bool,
}

impl Primitive for ContainerRebuild {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            &self.name,
            vec![self.container_sort.clone(), self.container_sort.clone()],
            span.clone(),
        )
        .into_box()
    }
}

impl ReadPrim for ContainerRebuild {
    fn apply<'a, 'db>(&self, mut state: ReadState<'a, 'db>, args: &[Value]) -> Option<Value> {
        rebuild_container_value_rec(
            &mut state,
            &self.container_sort,
            args[0],
            &self.uf_names,
            self.proof_mode,
        )
    }
}

/// Proof-mode counterpart of [`ContainerRebuild`]: mints a `Congr` chain
/// proving `old_container = rebuilt_container` (recursing through nested
/// containers). Reads `UF_<E>f` (element equality proofs) and `<CSort>Proof`
/// (reflexive bases), mints `Congr`/`Trans`/`Sym` terms, and anchors a
/// reflexive proof on each rebuilt container so it can be rebuilt again later.
/// It is a [`FullPrim`], valid only in a `:naive` rule's action.
#[derive(Clone)]
struct ContainerRebuildProof {
    name: String,
    container_sort: ArcSort,
    proof_sort: ArcSort,
    /// element-sort name -> `UF_<E>f` table name (all reachable eq-sorts)
    uf_names: HashMap<String, String>,
    /// container-sort name -> `<CSort>Proof` table name (all reachable containers)
    cproof_names: HashMap<String, String>,
    /// `Congr` / `Trans` / `Sym` / `ContainerNormalize` proof constructor names
    congr_name: String,
    trans_name: String,
    sym_name: String,
    container_normalize_name: String,
}

impl Primitive for ContainerRebuildProof {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            &self.name,
            vec![self.container_sort.clone(), self.proof_sort.clone()],
            span.clone(),
        )
        .into_box()
    }
}

impl FullPrim for ContainerRebuildProof {
    fn apply<'a, 'db>(&self, mut state: FullState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let (_rebuilt, proof) =
            rebuild_container_proof_rec(&mut state, self, &self.container_sort, args[0])?;
        Some(proof)
    }
}

/// Recursively rebuild `value` (of container sort `sort`) and produce a proof
/// that `value = rebuilt`. Returns `(rebuilt_value, proof)`. Uses the same
/// per-child resolution as [`rebuild_container_value_rec`], additionally
/// folding a `Congr` step for every changed child and recording a reflexive
/// anchor `<CSort>Proof(rebuilt) = Trans(Sym proof, proof)` so the rebuilt
/// value can itself be rebuilt in a later iteration.
fn rebuild_container_proof_rec(
    state: &mut FullState,
    prim: &ContainerRebuildProof,
    sort: &ArcSort,
    value: Value,
) -> Option<(Value, Value)> {
    // Reflexive base proof `value = value`.
    let base = state.lookup(prim.cproof_names.get(sort.name())?, &[value])?;
    let elements = {
        let cvs = state.container_values();
        sort.inner_values(cvs, value)
    };

    let mut leaders: HashMap<Value, Value> = HashMap::default();
    let mut child_proofs: Vec<(usize, Value)> = vec![];
    for (j, (esort, eval)) in elements.iter().enumerate() {
        if esort.is_eq_sort() {
            if let Some(uf_name) = prim.uf_names.get(esort.name())
                && let Some(pair_val) = state.lookup(uf_name, &[*eval])
            {
                let (leader, proof) = {
                    let pc = state
                        .container_values()
                        .get_val::<crate::sort::PairContainer>(pair_val)?;
                    (pc.first, pc.second)
                };
                if leader != *eval {
                    leaders.insert(*eval, leader);
                    child_proofs.push((j, proof));
                }
            }
        } else if esort.is_eq_container_sort() {
            let (rebuilt_child, child_proof) =
                rebuild_container_proof_rec(state, prim, esort, *eval)?;
            if rebuilt_child != *eval {
                leaders.insert(*eval, rebuilt_child);
                child_proofs.push((j, child_proof));
            }
        }
    }

    // Rebuild the value against the collected leaders.
    let rebuilt = {
        let cvs = state.container_values();
        let es = state.raw_exec_state();
        sort.rebuild_container_with_leaders(cvs, es, value, &leaders)
    };

    // Fold a `Congr` step per changed child onto the reflexive base. This
    // proves `value = raw`, where `raw` is the term with children replaced in
    // place (it may be in non-canonical order, or have duplicate/clobbering
    // entries for collapsing containers).
    let congr_action = state.registry().lookup_table(&prim.congr_name)?.clone();
    let mut current = base;
    for (j, proof) in child_proofs {
        let j_val = state.base_values().get::<i64>(j as i64);
        current =
            congr_action.lookup_or_insert(state.raw_exec_state(), &[current, j_val, proof])?;
    }

    // Bridge the (possibly non-canonical) `raw` term to the canonical `rebuilt`
    // term with the container axiom: `ContainerNormalize(current)` proves
    // `value = normalize(raw)`, which the checker recomputes to match
    // `reconstruct_termdag(rebuilt)`. We mint it unconditionally; for
    // order/arity-preserving containers (Vec/Pair) the normalization is the
    // identity, so the axiom is a no-op the proof simplifier removes.
    let axiom_action = state
        .registry()
        .lookup_table(&prim.container_normalize_name)?
        .clone();
    current = axiom_action.lookup_or_insert(state.raw_exec_state(), &[current])?;

    // Anchor a reflexive proof on the rebuilt value for future rebuilds.
    if rebuilt != value {
        let sym_action = state.registry().lookup_table(&prim.sym_name)?.clone();
        let trans_action = state.registry().lookup_table(&prim.trans_name)?.clone();
        let cproof_action = state
            .registry()
            .lookup_table(prim.cproof_names.get(sort.name())?)?
            .clone();
        // Sym(current): rebuilt = value;  Trans(Sym(current), current): rebuilt = rebuilt.
        let sym_p = sym_action.lookup_or_insert(state.raw_exec_state(), &[current])?;
        let refl = trans_action.lookup_or_insert(state.raw_exec_state(), &[sym_p, current])?;
        cproof_action.insert(state.raw_exec_state(), [rebuilt, refl].into_iter());
    }

    Some((rebuilt, current))
}
