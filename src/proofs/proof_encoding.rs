#[doc = include_str!("proof_encoding.md")]
use crate::proofs::proof_encoding_helpers::{EncodingNames, Justification};
use crate::typechecking::FuncType;
use crate::*;

// TODO refactor so that encoding state is optional on the e-graph, ProofNames not optional on EncodingState. Then we don't have to clone proof names everywhere.
#[derive(Clone)]
pub(crate) struct EncodingState {
    pub uf_parent: HashMap<String, String>,
    pub uf_function: HashMap<String, String>,
    /// Per-sort name of the `(Pair sort proof)` sort, shared by the UF function
    /// index and the FD view value column. Memoized so all references agree.
    pub uf_pair_sort: HashMap<String, String>,
    /// Names of functions declared with the FD pair-valued view (constructors and
    /// primitive-bodied custom functions). Used so action/rebuild sites route the
    /// same way the view declaration did.
    pub fd_view_funcs: HashSet<String>,
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
            uf_pair_sort: HashMap::default(),
            fd_view_funcs: HashSet::default(),
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

    /// Whether `fdecl` uses the FD pair-valued view (`(children) -> (pair output proof)`,
    /// keyed on children only) rather than the legacy custom shape (output-in-key,
    /// proof-as-value).
    ///
    /// Constructors always use the FD view. A custom (non-constructor) function uses
    /// the FD view iff its `:merge` body is a PRIMITIVE-bodied expression (every call
    /// in the body is a primitive — e.g. `min`/`max`/`or`/`set-union`) AND its output
    /// sort is NOT an eq-sort. Function/constructor-bodied merges (those whose merge
    /// body calls a user function) and eq-sort-output customs stay on the legacy
    /// `handle_merge_fn` shape. Custom functions with no `:merge` also stay legacy.
    pub(crate) fn is_fd_pair_view(&self, fdecl: &ResolvedFunctionDecl) -> bool {
        match fdecl.subtype {
            FunctionSubtype::Constructor => true,
            FunctionSubtype::Custom => {
                self.is_primitive_bodied_fd_custom(fdecl)
                    || self.is_constructor_bodied_fd_custom(fdecl)
            }
        }
    }

    /// True iff `fdecl` is a custom function whose `:merge` body mints CONSTRUCTORS
    /// (function-bodied) and whose INPUTS are non-eq-sort, making it eligible for the
    /// FD pair-valued view (Phase B). The merge body is allowed to call constructors
    /// (minted via the `fd-mint` Construct surface form) and primitives; calls to
    /// non-constructor user functions are NOT supported here (those stay legacy).
    ///
    /// Unlike the primitive-bodied case, the OUTPUT may be an eq-sort (the merge
    /// builds eq-sort terms via constructor FD views). Inputs must still be non-eq
    /// sorts: an eq-sort input is canonicalized during rebuild, which re-keys the row
    /// and rewrites its proof to a non-reflexive congruence proof, breaking the
    /// `MergeRow`/`MergeIdx` reflexive-premise requirement.
    fn is_constructor_bodied_fd_custom(&self, fdecl: &ResolvedFunctionDecl) -> bool {
        if fdecl.subtype != FunctionSubtype::Custom {
            return false;
        }
        let has_eq_sort_input = fdecl.schema.input.iter().any(|s| {
            self.egraph
                .type_info
                .get_sort_by_name(s)
                .map(|sort| sort.is_eq_sort())
                .unwrap_or(false)
        });
        if has_eq_sort_input {
            return false;
        }
        match &fdecl.merge {
            None => false,
            Some(merge) => {
                // Must actually call a user constructor (otherwise it is primitive-
                // bodied, handled by the other predicate), and EVERY user-function
                // call must be a constructor (so `fd-mint`/Construct can mint it).
                Self::merge_body_calls_constructor(merge)
                    && Self::merge_body_funcs_all_constructors(merge)
            }
        }
    }

    /// Whether the merge body contains at least one user-function call.
    fn merge_body_calls_constructor(expr: &ResolvedExpr) -> bool {
        match expr {
            ResolvedExpr::Lit(..) | ResolvedExpr::Var(..) => false,
            ResolvedExpr::Call(_, ResolvedCall::Func(_), _) => true,
            ResolvedExpr::Call(_, ResolvedCall::Primitive(_), args) => {
                args.iter().any(Self::merge_body_calls_constructor)
            }
        }
    }

    /// Whether every user-function call in the merge body targets a CONSTRUCTOR.
    fn merge_body_funcs_all_constructors(expr: &ResolvedExpr) -> bool {
        match expr {
            ResolvedExpr::Lit(..) | ResolvedExpr::Var(..) => true,
            ResolvedExpr::Call(_, ResolvedCall::Func(f), args) => {
                f.subtype == FunctionSubtype::Constructor
                    && args.iter().all(Self::merge_body_funcs_all_constructors)
            }
            ResolvedExpr::Call(_, ResolvedCall::Primitive(_), args) => {
                args.iter().all(Self::merge_body_funcs_all_constructors)
            }
        }
    }

    /// True iff `fdecl` is a custom function with a primitive-bodied `:merge` and a
    /// non-eq-sort output, making it eligible for the FD pair-valued view.
    fn is_primitive_bodied_fd_custom(&self, fdecl: &ResolvedFunctionDecl) -> bool {
        if fdecl.subtype != FunctionSubtype::Custom {
            return false;
        }
        // Eq-sort INPUTS are allowed (Phase C): rebuild canonicalizes an eq-sort
        // input, re-keys the view row, and rewrites its per-row proof into a
        // non-reflexive CONGRUENCE proof. The FD merge's `MergeRow`/`MergeIdx`
        // justification requires REFLEXIVE premise proofs, so at resugaring time
        // (`convert_raw_proof`) each such congruence premise `p : orig = canon` is
        // replaced with `Trans(Sym(p), p) : canon = canon` (reflexive, landing on
        // the canonical view row). See `reflexivize_premise` in proof_format.rs.
        // Output must not be an eq-sort: the FD view's output column is carried as a
        // plain value (no union-find canonicalization). Primitive merges produce
        // value/lattice sorts (i64, Set, Vec, ...), not eq-sort terms.
        let output_is_eq_sort = self
            .egraph
            .type_info
            .get_sort_by_name(&fdecl.schema.output)
            .map(|s| s.is_eq_sort())
            .unwrap_or(true);
        if output_is_eq_sort {
            return false;
        }
        match &fdecl.merge {
            None => false,
            Some(merge) => Self::merge_body_is_all_primitive(merge),
        }
    }

    /// Whether every call in a (resolved) merge body is a primitive call (no calls
    /// to user functions/constructors). Vars and literals are fine.
    fn merge_body_is_all_primitive(expr: &ResolvedExpr) -> bool {
        match expr {
            ResolvedExpr::Lit(..) | ResolvedExpr::Var(..) => true,
            ResolvedExpr::Call(_, ResolvedCall::Func(_), _) => false,
            ResolvedExpr::Call(_, ResolvedCall::Primitive(_), args) => {
                args.iter().all(Self::merge_body_is_all_primitive)
            }
        }
    }

    /// Whether the function with the given name was declared with the FD pair-valued
    /// view. Populated by [`Self::term_and_view`] when it declares the view table, so
    /// later action/rebuild sites (which only have the function name / [`FuncType`])
    /// can route consistently. Constructors are always FD even before being recorded.
    pub(crate) fn name_is_fd_pair_view(&self, name: &str) -> bool {
        // Rely solely on the recorded set: `term_and_view` records EVERY FD function
        // (constructors and primitive-bodied customs) when it declares the view. We
        // must NOT fall back to `get_func_type(name)` here, because by the time this
        // runs the encoder has redeclared the original function name as the hidden
        // inner term constructor, so `get_func_type` would report `Constructor` for a
        // legacy custom function.
        self.egraph.proof_state.fd_view_funcs.contains(name)
    }

    /// Like [`Self::name_is_fd_pair_view`], for a resolved [`FuncType`] at an action
    /// site (the view declaration has already recorded FD custom functions by name).
    pub(crate) fn func_type_is_fd_pair_view(&self, func_type: &FuncType) -> bool {
        if func_type.subtype == FunctionSubtype::Constructor {
            return true;
        }
        self.name_is_fd_pair_view(&func_type.name)
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
        let uf_function_index_ruleset_name =
            self.proof_names().uf_function_index_ruleset_name.clone();

        let proof_type = self.proof_type_str().to_string();

        // In proof mode, path compression composes proofs via Trans/Sym.
        // In term mode, the proof output is Unit and we just write ().
        let (path_compress_query, path_compress_action) = if self.egraph.proof_state.proofs_enabled
        {
            let p1_fresh = self.egraph.parser.symbol_gen.fresh("p1");
            let p2_fresh = self.egraph.parser.symbol_gen.fresh("p2");
            let trans = self.proof_names().eq_trans_constructor.clone();
            (
                format!(
                    "(= {p1_fresh} ({pname} a b))
                        (= {p2_fresh} ({pname} b c))"
                ),
                format!(
                    "(delete ({pname} a b))
                       (set ({pname} a c) ({trans} {p1_fresh} {p2_fresh}))"
                ),
            )
        } else {
            (
                format!("({pname} a b)\n                        ({pname} b c)"),
                format!("(delete ({pname} a b))\n                       (set ({pname} a c) ())"),
            )
        };

        // The UF function index stores, per source term, the current leader (and,
        // in proof mode, a proof that the source equals the leader). It is the only
        // writer of `(UF_<sort>f a)`, so a key collision means a single source `a`
        // has two parents `b` and `c` (the SINGLE-PARENT situation). The index's
        // `:merge` resolves that collision: it picks the smaller leader (closer to
        // the root) as the surviving value AND writes the union edge between the two
        // parents back into the constructor UF table (the source of truth). Path
        // compression then chases `a -> larger -> smaller` and deletes the redundant
        // `(UF a larger)` row, so the UF table converges to a single parent per
        // source — exactly the work the old `single_parent` rule did, now folded into
        // the merge.
        //
        // In proof mode, UF function index stores (leader, proof) pairs.
        // In term mode, it just stores the leader.
        let (
            uf_function_output_type,
            uf_pair_sort_decl,
            uf_index_query,
            uf_index_action,
            uf_index_merge,
        ) = if self.egraph.proof_state.proofs_enabled {
            let pair_sort = self.uf_pair_sort_name(sort_name);
            let proof_fresh = self.egraph.parser.symbol_gen.fresh("uf_idx_proof");
            let trans = self.proof_names().eq_trans_constructor.clone();
            let sym = self.proof_names().eq_sym_constructor.clone();
            // old = (pair leader_o p_o), p_o proves a = leader_o.
            // new = (pair leader_n p_n), p_n proves a = leader_n.
            let lo = "(pair-first old)";
            let po = "(pair-second old)";
            let ln = "(pair-first new)";
            let pn = "(pair-second new)";
            let larger = format!("(ordering-max {lo} {ln})");
            let smaller = format!("(ordering-min {lo} {ln})");
            // Proof that larger = smaller, oriented by which leader is larger.
            // If leader_o is the larger one: leader_o = leader_n via Trans(Sym p_o, p_n).
            // Otherwise leader_n is larger: leader_n = leader_o via Trans(Sym p_n, p_o).
            let union_proof = format!(
                "(select-eq {larger} {lo} ({trans} ({sym} {po}) {pn}) ({trans} ({sym} {pn}) {po}))"
            );
            // The surviving index proof must prove a = smaller; keep the existing
            // premise proof for whichever leader is the smaller one (value-stable,
            // so the row saturates instead of minting a fresh proof each iteration).
            let surviving_proof = format!("(select-eq {smaller} {lo} {po} {pn})");
            // Block-form merge: write the parent-union edge into the UF table,
            // then return the surviving (leader, proof) pair.
            let merge = format!(
                "((set ({pname} {larger} {smaller}) {union_proof}) (pair {smaller} {surviving_proof}))"
            );
            (
                pair_sort.clone(),
                format!("(sort {pair_sort} (Pair {sort_name} {proof_type}))"),
                format!("(= {proof_fresh} ({pname} a b))"),
                format!("(set ({uf_function_name} a) (pair b {proof_fresh}))"),
                merge,
            )
        } else {
            // Term mode: index value is just the leader. The merge writes the
            // parent-union edge and keeps the smaller leader.
            let merge = format!(
                "((set ({pname} (ordering-max old new) (ordering-min old new)) ()) (ordering-min old new))"
            );
            (
                sort_name.to_string(),
                "".to_string(),
                format!("({pname} a b)"),
                format!("(set ({uf_function_name} a) b)"),
                merge,
            )
        };

        let mut code = format!(
            "{uf_pair_sort_decl}
             (function {pname} ({sort_name} {sort_name}) {proof_type} :merge old :internal-hidden)
             ;; The index's :merge folds in the single-parent invariant: a key
             ;; collision (one source with two parents) unions the parents back into
             ;; the UF table and keeps the smaller leader.
             (function {uf_function_name} ({sort_name}) {uf_function_output_type} :merge {uf_index_merge} :unextractable :internal-hidden)
             ;; performs path compression, ensuring each term points to the representative
             (rule ({path_compress_query}
                    (!= b c))
                  ({path_compress_action})
                   :ruleset {path_compress_ruleset_name}
                   :name \"{fresh_name}\")
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

        if self.is_fd_pair_view(fdecl) {
            // FD view: the key is the children only (the output is the value),
            // so we delete/subsume by the children key. Bind the value with
            // `out` only to guard that the row exists.
            format!(
                "(rule (({to_delete_name} {child_names})
                        (= out ({view_name} {child_names})))
                       ((delete ({view_name} {child_names}))
                        (delete ({to_delete_name} {child_names})))
                        :ruleset {delete_subsume_ruleset}
                        :name \"{fresh_name}\")
                 (rule (({subsumed_name} {child_names})
                        (= out ({view_name} {child_names})))
                       ((subsume ({view_name} {child_names})))
                        :ruleset {delete_subsume_ruleset}
                        :name \"{fresh_name}_subsume\")"
            )
        } else {
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
        let term_and_proof = self.update_view(name, &updated, &proof_var, false, "");
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
        // Legacy custom functions (function/constructor-bodied or no-merge) get a
        // separate merge rule. FD functions (constructors and primitive-bodied
        // customs) handle the merge in the FD view's `:merge` instead, so no rule.
        if fdecl.subtype == FunctionSubtype::Custom && !self.is_fd_pair_view(fdecl) {
            self.handle_merge_fn(
                fdecl,
                &child_names,
                &child_names_str,
                &view_name,
                &rebuilding_ruleset,
            )
        } else {
            let _ = (&child_names, &rebuilding_ruleset);
            String::new()
        }
    }

    /// The `:merge` expression for a constructor's FD view table.
    ///
    /// The FD view maps `(children) -> output`, so when two terms with the
    /// same children have different outputs (i.e. they are congruent) the
    /// key collides and this merge runs. It unions the two outputs by writing
    /// the union-find parent edge on the side with a block-merge `set`, then
    /// keeps the smaller representative as the surviving output.
    fn constructor_view_merge(&mut self, out_type: &str) -> String {
        let uf = self.uf_name(out_type);
        // Block-form merge: as an effect, record the congruence union edge in
        // the UF table on the side (a `set` action — it names `uf` statically
        // so the backend declares the write-dependency); then the value is the
        // smaller representative.
        if self.egraph.proof_state.proofs_enabled {
            // The view value is a `(pair output proof)` (two value columns);
            // `old`/`new` are the two colliding pairs (both reps of the same
            // f(children), since they collided on the view key). The proof in
            // each pair proves `output = f(children)`. We orient them by the
            // ordering of the outputs and build the congruence union proof
            // `Trans(p_large, Sym(p_small))` proving `larger = smaller`.
            let trans = self.proof_names().eq_trans_constructor.clone();
            let sym = self.proof_names().eq_sym_constructor.clone();
            let oo = "(pair-first old)";
            let on = "(pair-first new)";
            let po = "(pair-second old)";
            let pn = "(pair-second new)";
            let larger = format!("(ordering-max {oo} {on})");
            let smaller = format!("(ordering-min {oo} {on})");
            // p_large proves `larger = f(children)`, p_small proves
            // `smaller = f(children)`. select-eq picks the proof of whichever
            // output equals larger/smaller. On a tie (oo == on) both pick the
            // OLD proof `po`, so the surviving value is unchanged and the merge
            // does not re-fire forever.
            let p_large = format!("(select-eq {larger} {oo} {po} {pn})");
            let p_small = format!("(select-eq {smaller} {oo} {po} {pn})");
            let union_proof = format!("({trans} {p_large} ({sym} {p_small}))");
            format!("((set ({uf} {larger} {smaller}) {union_proof}) (pair {smaller} {p_small}))")
        } else {
            // Term-encoding mode (no proofs): the view value is the output (an
            // eclass id) directly, and the UF proof column is Unit.
            format!(
                "((set ({uf} (ordering-max old new) (ordering-min old new)) ()) (ordering-min old new))"
            )
        }
    }

    /// The `:merge` expression for a primitive-bodied custom function's FD view.
    ///
    /// The FD view maps `(children) -> (pair output proof)`. When two rows collide on
    /// the children key (the same `f(children)` derived with two different output
    /// values) the merge runs the user's primitive merge body on the two OUTPUT
    /// columns and produces a `MergeIdx` justification for the merged row. Unlike a
    /// constructor, there is NO union: the output is a merged value, not an eclass
    /// collision.
    ///
    /// In term mode (no proofs) the view value is the output directly and the merge is
    /// simply the (rewritten) primitive merge body.
    fn custom_fd_view_merge(&mut self, fname: &str, merge: &ResolvedExpr) -> String {
        if self.egraph.proof_state.proofs_enabled {
            // `old`/`new` in the user merge body refer to the OUTPUT value, which in
            // the pair-view lives in `(pair-first old)`/`(pair-first new)`.
            let body = Self::render_merge_on_pair_first(merge);
            let merge_row = self.proof_names().merge_fn_row_constructor.clone();
            // The proof column. `MergeRow` reconstructs the top view row
            // `f(inputs..., merged)` by evaluating the WHOLE merge body on the premise
            // outputs.
            //
            // STABILITY: when the merged output equals one of the premise outputs
            // (the common case for `min`/`max`/`or`/lattice ops) we keep that
            // premise's existing proof verbatim, so the resulting pair is identical to
            // the surviving row and the merge does not re-fire forever. Using
            // `select-eq` keeps the OLD pair when `merged == (pair-first old)`, the NEW
            // pair's proof when `merged == (pair-first new)`, and otherwise (a value
            // distinct from both premises) mints a fresh `MergeRow` justification.
            let fresh = format!("({merge_row} \"{fname}\" (pair-second old) (pair-second new))");
            let proof = format!(
                "(select-eq {body} (pair-first old) (pair-second old) \
                  (select-eq {body} (pair-first new) (pair-second new) {fresh}))"
            );
            format!("(pair {body} {proof})")
        } else {
            Self::render_merge_on_old_new(merge)
        }
    }

    /// The `:merge` expression for a CONSTRUCTOR-bodied custom function's FD view
    /// (Phase B). Like [`Self::custom_fd_view_merge`], but the merge body mints
    /// constructors. Each constructor node `(C a b)` is rendered as a `fd-mint` over
    /// `C`'s pair-valued FD view, carrying a `MergeIdx(f, p_old, p_new, idx)` proof of
    /// that node's existence (idx = the node's pre-order position over ALL body nodes).
    /// The output column is the resulting nested tree of `fd-mint`s; the f-view proof
    /// column is `MergeRow(f, p_old, p_new)` (with the saturation select-eq guard).
    fn constructor_bodied_fd_view_merge(&mut self, fname: &str, merge: &ResolvedExpr) -> String {
        if self.egraph.proof_state.proofs_enabled {
            let merge_idx = self.proof_names().merge_fn_idx_constructor.clone();
            let merge_row = self.proof_names().merge_fn_row_constructor.clone();
            // Output column: the merge body with constructor nodes minted via `fd-mint`
            // (each carrying its own pre-order-indexed existence proof), `old`/`new`
            // projected to the output column.
            let mut idx = 0usize;
            let body = self.render_construct_body(merge, true, fname, &merge_idx, &mut idx);
            // Proof column for f's view row. `MergeRow` reconstructs `f(inputs...,
            // merged)` by running the WHOLE merge body on the premise outputs.
            //
            // STABILITY: if the merged value equals a premise output, keep that
            // premise's existing proof so the pair is identical to the surviving row
            // (the merge then saturates). For a value distinct from both premises mint
            // a fresh `MergeRow`. (The constructor-bodied output is built from
            // `pair-first old`/`new`, so the comparison is over the output column.)
            let out_value = Self::render_merge_on_pair_first(merge);
            let fresh = format!("({merge_row} \"{fname}\" (pair-second old) (pair-second new))");
            let proof = format!(
                "(select-eq {out_value} (pair-first old) (pair-second old) \
                  (select-eq {out_value} (pair-first new) (pair-second new) {fresh}))"
            );
            format!("(pair {body} {proof})")
        } else {
            // Term mode (no proofs): the view value is the output directly, but the
            // constructor nodes must STILL be minted into their FD views (a plain
            // `(C a b)` call would create the term but not its view row, which all
            // lookups consult). So wrap each constructor in `fd-mint` with no proof.
            let mut idx = 0usize;
            self.render_construct_body(merge, false, fname, "", &mut idx)
        }
    }

    /// Render a constructor-bodied merge body for the OUTPUT column of a Phase B FD
    /// view's `:merge`. Constructor calls are wrapped in `(fd-mint (C ...) [proof])`,
    /// which mints `C` into its FD view (so the e-class is queryable) and returns the
    /// output e-class.
    ///
    /// In proof mode (`proofs`), each node carries `MergeIdx(fname, (pair-second old),
    /// (pair-second new), idx)` (idx = the node's pre-order position over ALL body
    /// nodes, matching `subexpr_at_index` in the checker) and `old`/`new` project to
    /// the output column via `pair-first`. In term mode there is no proof arg and
    /// `old`/`new` stay bare. `idx` is threaded so every node gets a distinct index.
    fn render_construct_body(
        &mut self,
        expr: &ResolvedExpr,
        proofs: bool,
        fname: &str,
        merge_idx: &str,
        idx: &mut usize,
    ) -> String {
        let my_idx = *idx;
        *idx += 1;
        match expr {
            ResolvedExpr::Lit(_, lit) => format!("{lit}"),
            ResolvedExpr::Var(_, var) => {
                let n = var.name.as_str();
                if proofs && (n == "old" || n == "new") {
                    format!("(pair-first {n})")
                } else {
                    n.to_string()
                }
            }
            ResolvedExpr::Call(_, ResolvedCall::Func(c), args) => {
                // A constructor node: mint it via `fd-mint`, carrying its own
                // existence proof (indexed by its pre-order position) in proof mode.
                // The FD view name is emitted as a string literal so the desugared
                // program is self-contained (it does not depend on the encoder's
                // runtime view-name map when re-parsed).
                let view_name = self.view_name(&c.name);
                let rendered: Vec<String> = args
                    .iter()
                    .map(|a| self.render_construct_body(a, proofs, fname, merge_idx, idx))
                    .collect();
                let ctor_call = format!("({} {})", c.name, ListDisplay(rendered, " "));
                if proofs {
                    let node_proof = format!(
                        "({merge_idx} \"{fname}\" (pair-second old) (pair-second new) {my_idx})"
                    );
                    format!("(fd-mint \"{view_name}\" {ctor_call} {node_proof})")
                } else {
                    format!("(fd-mint \"{view_name}\" {ctor_call})")
                }
            }
            ResolvedExpr::Call(_, ResolvedCall::Primitive(p), args) => {
                let rendered: Vec<String> = args
                    .iter()
                    .map(|a| self.render_construct_body(a, proofs, fname, merge_idx, idx))
                    .collect();
                format!("({} {})", p.name(), ListDisplay(rendered, " "))
            }
        }
    }

    /// Render a resolved merge body to egglog source, replacing the bare leaf vars
    /// `old`/`new` with `(pair-first old)`/`(pair-first new)` (the output column of the
    /// pair-view value).
    fn render_merge_on_pair_first(expr: &ResolvedExpr) -> String {
        Self::render_merge_expr(expr, true)
    }

    /// Render a resolved merge body to egglog source as-is (term mode: the view value
    /// is the output directly, so `old`/`new` stay bare).
    fn render_merge_on_old_new(expr: &ResolvedExpr) -> String {
        Self::render_merge_expr(expr, false)
    }

    fn render_merge_expr(expr: &ResolvedExpr, project_first: bool) -> String {
        match expr {
            ResolvedExpr::Lit(_, lit) => format!("{lit}"),
            ResolvedExpr::Var(_, var) => {
                let n = var.name.as_str();
                if project_first && (n == "old" || n == "new") {
                    format!("(pair-first {n})")
                } else {
                    n.to_string()
                }
            }
            ResolvedExpr::Call(_, call, args) => {
                let rendered: Vec<String> = args
                    .iter()
                    .map(|a| Self::render_merge_expr(a, project_first))
                    .collect();
                format!("({} {})", call.name(), ListDisplay(rendered, " "))
            }
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

        let is_fd = self.is_fd_pair_view(fdecl);
        let is_fd_custom = is_fd && fdecl.subtype == FunctionSubtype::Custom;
        if is_fd {
            self.egraph
                .proof_state
                .fd_view_funcs
                .insert(fdecl.name.clone());
        }

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
        // For an FD custom view the output column's pair sort `(Pair output proof)`
        // is NOT declared by `declare_sort` (the output is typically a non-eq-sort
        // like i64/Set, which has no UF setup), so declare it here.
        let mut extra_pair_sort_decl = String::new();
        let view_decl = if fdecl.subtype == FunctionSubtype::Constructor {
            // FD view: key is the children only. In term mode the value is the
            // output (the eclass representative), a plain eq-sort kept indexable
            // for cheap rebuild canonicalization. In proof mode the value is a
            // `(pair output proof)` (two value columns): the output stays a real
            // eq-sort column (fast rebuild) while the per-row existence proof
            // rides alongside it. Congruence is handled by `:merge`, which reads
            // both rows' proofs from the pair.
            let ctor_merge = self.constructor_view_merge(&out_type);
            let view_value_sort = if self.egraph.proof_state.proofs_enabled {
                self.uf_pair_sort_name(&out_type)
            } else {
                out_type.clone()
            };
            format!(
                "(function {view_name} ({in_sorts}) {view_value_sort} :merge {ctor_merge} :internal-term-constructor {name}{view_flags})"
            )
        } else if is_fd_custom {
            // Custom function on the FD pair-valued view: key is the children only,
            // value is `(pair output proof)` in proof mode (output only in term mode).
            // The user's `:merge` runs on the output column. No union: the output is a
            // merged value, not an eclass collision.
            //
            // Primitive-bodied merges (`min`/`or`/...) render directly; constructor-
            // bodied merges (Phase B, e.g. `(C2 (C1 old new) ...)`) mint each
            // constructor node via the `fd-mint` Construct surface form. Both carry a
            // `MergeRow` justification on the f-view proof column.
            let merge = fdecl
                .merge
                .as_ref()
                .expect("FD custom view requires a :merge");
            let custom_merge = if self.is_constructor_bodied_fd_custom(fdecl) {
                self.constructor_bodied_fd_view_merge(&fdecl.name.clone(), merge)
            } else {
                self.custom_fd_view_merge(&fdecl.name.clone(), merge)
            };
            let view_value_sort = if self.egraph.proof_state.proofs_enabled {
                // Only emit the `(sort ...)` declaration the first time we mint the
                // pair sort for this output sort (another FD custom function with the
                // same output sort would otherwise re-declare it).
                let already_declared = self
                    .egraph
                    .proof_state
                    .uf_pair_sort
                    .contains_key(out_type.as_str());
                let pair_sort = self.uf_pair_sort_name(&out_type);
                if !already_declared {
                    let proof_type = self.proof_type_str().to_string();
                    extra_pair_sort_decl =
                        format!("(sort {pair_sort} (Pair {out_type} {proof_type}))");
                }
                pair_sort
            } else {
                out_type.clone()
            };
            format!(
                "(function {view_name} ({in_sorts}) {view_value_sort} :merge {custom_merge} :internal-term-constructor {name}{view_flags})"
            )
        } else {
            // Function/constructor-bodied (or no-merge) custom functions keep the old
            // shape (output in key, proof as the value) and use `handle_merge_fn`.
            format!(
                "(function {view_name} ({view_sorts}) {proof_type} :merge old :internal-term-constructor {name}{view_flags})"
            )
        };
        self.parse_program(&format!(
            "
            (sort {fresh_sort})
            {extra_pair_sort_decl}
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

        // Check if there are any eq-sort columns at all; if not, no rebuild rule needed.
        if !types.iter().any(|t| t.is_eq_sort()) {
            return vec![];
        }

        let view_name = self.view_name(&fdecl.name);
        // FD views (constructors + primitive-bodied customs) key on the children only
        // (the output lives in the value column); legacy custom views key on all
        // columns (the output is part of the key).
        let is_fd = self.is_fd_pair_view(fdecl);
        let child = |i: usize| format!("c{i}_");
        let children_vec: Vec<String> = (0..types.len()).map(child).collect();
        let delete_key = if is_fd {
            ListDisplay(&children_vec[..children_vec.len() - 1], " ").to_string()
        } else {
            ListDisplay(&children_vec, " ").to_string()
        };

        // For each eq-sort column, look up its leader via the UF table.
        // For non-eq-sort columns, the leader is the same as the original.
        let mut uf_queries = vec![];
        let mut leader_vars: Vec<String> = vec![];
        let mut bool_neq_exprs = vec![];
        let mut uf_proof_vars: Vec<Option<String>> = vec![];

        for (i, ty) in types.iter().enumerate() {
            if ty.is_eq_sort() {
                let leader_var = format!("c{i}_leader_");
                let uf_function_name = self.uf_function_name(ty.name());
                let ci = child(i);

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
                leader_vars.push(leader_var);
            } else {
                leader_vars.push(child(i));
                uf_proof_vars.push(None);
            }
        }

        let uf_query_str = uf_queries.join("\n       ");
        let or_expr = format!("(or {})", bool_neq_exprs.join("\n             "));
        let filter_query = format!("(guard {or_expr})");

        // Build the updated children: use leader_var for eq-sort columns, original for others.
        let children_updated: Vec<String> = leader_vars.clone();

        let fresh_name = self.egraph.parser.symbol_gen.fresh("rebuild_rule");
        let (query_view, view_prf) =
            self.query_view_and_get_proof(&fdecl.name, &children_vec, is_fd, &fdecl.schema.output);

        // Build proof code if proofs are enabled.
        // We chain congruence proofs for each updated child and a transitivity proof
        // for the representative (last column) update.
        let (pf_code, pf_var) = if self.egraph.proof_state.proofs_enabled {
            let eq_trans_constructor = self.proof_names().eq_trans_constructor.clone();
            let congr_constructor = self.proof_names().congr_constructor.clone();
            let sym_constructor = self.proof_names().eq_sym_constructor.clone();

            // Start with the view proof and apply congruence for each eq-sort child
            // (excluding the last column if this is a constructor, since that's the representative).
            let mut current_proof = view_prf.clone();
            let mut proof_code_parts = vec![];

            for (i, ty) in types.iter().enumerate() {
                if !ty.is_eq_sort() {
                    continue;
                }

                let uf_prf = uf_proof_vars[i].as_ref().unwrap();

                if fdecl.subtype == FunctionSubtype::Constructor && i == types.len() - 1 {
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
                    // Updating a child via congruence
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

        let updated_view = self.update_view(
            &fdecl.name,
            &children_updated,
            &pf_var,
            is_fd,
            &fdecl.schema.output,
        );

        // Make a single rule that updates the view when any child's leader differs.
        let rule = format!(
            "(rule ({query_view}
                    {uf_query_str}
                    {filter_query}
                    )
                 (
                  {pf_code}
                  {updated_view}
                  (delete ({view_name} {delete_key}))
                 )
                  :ruleset {} :name \"{fresh_name}\" :internal-include-subsumed)",
            self.proof_names().rebuilding_ruleset_name
        );
        self.parse_program(&rule)
    }

    /// Rules that update the to_subsume tables when children change.
    /// copied from above and changed to remove last param since we dont deal with output value in to subsumed rows, removed proof flags since we dont need proofs for this, and changed from function returning unit to constructor for to subsume
    fn rebuilding_subsumed_rules(&mut self, fdecl: &ResolvedFunctionDecl) -> Vec<Command> {
        let ResolvedCall::Func(FuncType { input, .. }) = &fdecl.resolved_schema else {
            panic!("cannot create subsumed rules for primitives")
        };

        // Check if there are any eq-sort columns at all; if not, no rebuild rule needed.
        if !input.iter().any(|t| t.is_eq_sort()) {
            return vec![];
        }

        let subsumed_name = self.subsumed_name(&fdecl.name);
        let child = |i: usize| format!("c{i}_");
        let children_vec: Vec<String> = (0..input.len()).map(child).collect();
        let children = format!("{}", ListDisplay(&children_vec, " "));

        // For each eq-sort column, look up its leader via the UF table.
        // For non-eq-sort columns, the leader is the same as the original.
        let mut uf_queries = vec![];
        let mut leader_vars: Vec<String> = vec![];
        let mut bool_neq_exprs = vec![];

        for (i, ty) in input.iter().enumerate() {
            if ty.is_eq_sort() {
                let leader_var = format!("c{i}_leader_");
                let uf_function_name = self.uf_function_name(ty.name());
                let ci = child(i);

                if self.egraph.proof_state.proofs_enabled {
                    // UF function index returns a Pair(leader, proof); one lookup gives both
                    let pair_var = self.fresh_var();
                    uf_queries.push(format!(
                        "(= {pair_var} ({uf_function_name} {ci}))
                         (= {leader_var} (pair-first {pair_var}))"
                    ));
                } else {
                    uf_queries.push(format!("(= {leader_var} ({uf_function_name} {ci}))"));
                }

                bool_neq_exprs.push(format!("(bool-!= {ci} {leader_var})"));
                leader_vars.push(leader_var);
            } else {
                leader_vars.push(child(i));
            }
        }

        let uf_query_str = uf_queries.join("\n       ");
        let or_expr = format!("(or {})", bool_neq_exprs.join("\n             "));
        let filter_query = format!("(guard {or_expr})");

        // Build the updated children: use leader_var for eq-sort columns, original for others.
        let children_updated: Vec<String> = leader_vars.clone();

        let fresh_name = self
            .egraph
            .parser
            .symbol_gen
            .fresh("rebuild_to_subsume_rule");

        let updated_children_view = ListDisplay(children_updated, " ");

        // Make a single rule that updates the view when any child's leader differs.
        let rule = format!(
            "(rule (({subsumed_name} {children})
                    {uf_query_str}
                    {filter_query}
                    )
                 (
                  ({subsumed_name} {updated_children_view})
                  (delete ({subsumed_name} {children}))
                 )
                  :ruleset {} :name \"{fresh_name}\" :internal-include-subsumed)",
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
            // In proof normal form, this is the only way that function calls appear.
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

                let view_name = self.view_name(head.name());
                let is_fd = self.name_is_fd_pair_view(head.name());

                // Query the view and obtain the base existence proof. For an FD custom
                // view the key is the children only and the value is `(pair output
                // proof)`: bind the output via `pair-first` and source the proof from
                // `pair-second`. For the legacy view the output `v` is part of the key
                // and the value IS the proof.
                let base_proof = if is_fd {
                    let key_str = ListDisplay(&new_args, " ");
                    if self.egraph.proof_state.proofs_enabled {
                        let pair_var = self.fresh_var();
                        res.push(format!("(= {pair_var} ({view_name} {key_str}))"));
                        res.push(format!("(= {v} (pair-first {pair_var}))"));
                        format!("(pair-second {pair_var})")
                    } else {
                        res.push(format!("(= {v} ({view_name} {key_str}))"));
                        "()".to_string()
                    }
                } else {
                    new_args.push(v.to_string());
                    let args_str = ListDisplay(&new_args, " ");
                    let proof_var = self.fresh_var();
                    res.push(format!("(= {proof_var} ({view_name} {args_str}))"));
                    if self.egraph.proof_state.proofs_enabled {
                        proof_var
                    } else {
                        "()".to_string()
                    }
                };

                if self.egraph.proof_state.proofs_enabled {
                    let mut proof = base_proof;
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

                        // FD view: key is the children, value is a
                        // `(pair output proof)` in proof mode (output only in
                        // term mode). Bind the output (pair-first) and source
                        // the existence proof from pair-second.
                        let proof = if self.proofs_enabled() {
                            let pair_var = self.fresh_var();
                            res.push(format!("(= {pair_var} ({view_name} {args_str}))"));
                            res.push(format!("(= {fv} (pair-first {pair_var}))"));
                            let tp_var = format!("(pair-second {pair_var})");
                            let mut proof = tp_var;
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
                            res.push(format!("(= {fv} ({view_name} {args_str}))"));
                            "()".to_string()
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
    ///
    /// For legacy custom functions (old shape) the view key is `args` (children +
    /// output) and the value is the `proof`.
    ///
    /// For FD views (constructors + primitive-bodied customs) the view key is the
    /// children (all but the last arg) and the value is `(pair output proof)`. For an
    /// eq-sort output (constructors) we ALSO record `term_proof(output) = proof` (an
    /// existence proof for the eclass, used for bare eq-sort vars in facts). For a
    /// non-eq-sort output (FD customs) there is no `term_proof` table, so we skip it.
    fn update_view(
        &mut self,
        fname: &str,
        args: &[String],
        proof: &str,
        is_fd: bool,
        out_sort: &str,
    ) -> String {
        let view_name = self.view_name(fname);
        if is_fd {
            let (output, key) = args.split_last().expect("FD view needs an output");
            let key = ListDisplay(key, " ");
            if self.egraph.proof_state.proofs_enabled {
                // The FD view value is a `(pair output proof)` (two value columns):
                // the output and its existence proof.
                //
                // For an eq-sort output (constructors) we ALSO record
                // `term_proof(output) = proof`, the existence proof for the eclass,
                // used for bare eq-sort variables in facts and by prove-exists. The
                // `term_proof` table only exists for eq-sorts (it is created in
                // `declare_sort`). For an FD custom function with a non-eq-sort output
                // (i64/Set/...) there is no such table, so skip it.
                if self.sort_has_term_proof(out_sort) {
                    let tp = self.term_proof_name(out_sort);
                    format!(
                        "(set ({view_name} {key}) (pair {output} {proof}))\n(set ({tp} {output}) {proof})"
                    )
                } else {
                    format!("(set ({view_name} {key}) (pair {output} {proof}))")
                }
            } else {
                format!("(set ({view_name} {key}) {output})")
            }
        } else {
            format!("(set ({view_name} {}) {proof})", ListDisplay(args, " "))
        }
    }

    /// Whether a `term_proof` table exists for `sort_name`. These tables are created
    /// by `declare_sort` for every user-declared eq-sort, so a `term_proof` table
    /// exists iff we have minted a `term_proof_name` for the sort.
    fn sort_has_term_proof(&self, sort_name: &str) -> bool {
        self.egraph
            .proof_state
            .proof_names
            .term_proof_name
            .contains_key(sort_name)
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
            // For constructors, `term_proof(output) = proof` is written by
            // `update_view` (single source of truth for the FD view + its
            // proof). For custom functions there is no term_proof entry.
            (format!("(let {proof_var} {proof})"), proof_var)
        } else {
            ("".to_string(), "()".to_string())
        };

        res.push(proof_str);
        // FD functions (constructors + primitive-bodied customs) use the pair-valued
        // view keyed on children only. For constructors `args_with_fv` appended the
        // freshly-minted eclass `fv` as the output; for FD customs `args` already
        // ends with the output value (from the `(set (f c..) v)` action).
        let is_fd = self.func_type_is_fd_pair_view(func_type);
        res.push(self.update_view(
            &func_type.name,
            &args_with_fv,
            &view_proof_var,
            is_fd,
            func_type.output.name(),
        ));

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

    /// Returns a query binding the view's value and a variable for the proof
    /// (or Unit) output.
    ///
    /// For custom functions (old shape) `args` is the full key (children +
    /// output) and the value is the proof.
    ///
    /// For constructors (FD shape) the last element of `args` is the output
    /// variable to bind from the view value; the rest is the key. In proof
    /// mode the value is a `(pair output proof)`, so we bind the output via
    /// `pair-first` and return `(pair-second ...)` as the proof.
    fn query_view_and_get_proof(
        &mut self,
        fname: &str,
        args: &[String],
        is_fd: bool,
        out_sort: &str,
    ) -> (String, String) {
        let _ = out_sort;
        let view_name = self.view_name(fname);
        if is_fd {
            let (output, key) = args.split_last().expect("FD view needs an output");
            let key = ListDisplay(key, " ");
            if self.egraph.proof_state.proofs_enabled {
                // The FD view value is a `(pair output proof)`; bind the pair,
                // then project the output and proof out of it.
                let pair_var = self.fresh_var();
                let pf_var = format!("(pair-second {pair_var})");
                let query = format!(
                    "(= {pair_var} ({view_name} {key}))\n(= {output} (pair-first {pair_var}))"
                );
                (query, pf_var)
            } else {
                let query = format!("(= {output} ({view_name} {key}))");
                (query, "()".to_string())
            }
        } else {
            let pf_var = self.fresh_var();
            let query = format!("(= {pf_var} ({view_name} {}))", ListDisplay(args, " "));
            (query, pf_var)
        }
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
        let uf_function_index = self.proof_names().uf_function_index_ruleset_name.clone();
        let rebuilding_cleanup_ruleset = self.proof_names().rebuilding_cleanup_ruleset_name.clone();
        let rebuilding_ruleset = self.proof_names().rebuilding_ruleset_name.clone();
        let delete_ruleset = self.proof_names().delete_subsume_ruleset_name.clone();
        // The single-parent invariant is now maintained by the UF function index's
        // `:merge` (a source with two parents collides on the index key), so there is
        // no separate `single_parent` ruleset to saturate here.
        self.parse_schedule(format!(
            "(seq
              (saturate
                  {rebuilding_cleanup_ruleset}
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
        log::trace!("Term encoding for {command}");
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
            }
            ResolvedNCommand::Function(fdecl) => {
                res.extend(self.term_and_view(fdecl));
                res.extend(self.rebuilding_rules(fdecl));
                res.extend(self.rebuilding_subsumed_rules(fdecl));
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
                    if self.name_is_fd_pair_view(n) {
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
