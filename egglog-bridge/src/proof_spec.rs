use std::rc::Rc;
use std::{iter, sync::Arc};

use crate::core_relations::{
    ColumnId, DisplacedTableWithProvenance, ProofReason as UfProofReason, ProofStep, RuleBuilder,
    Value,
};
use crate::numeric_id::{DenseIdMap, NumericId, define_id};
use crate::rule::Variable;
use crate::termdag::TermId;
use egglog_reports::ReportLevel;
use hashbrown::{HashMap, HashSet};

use crate::{
    ColumnTy, EGraph, FunctionId, GetFirstMatch, QueryEntry, Result, RuleId, SideChannel,
    SourceExpr, TopLevelLhsExpr,
    proof_format::{
        CongProof, EqProof, EqProofId, Premise, ProofStore, RuleVarBinding, TermProof, TermProofId,
    },
    rule::{AtomId, Bindings, DstVar, VariableId},
    syntax::{RuleData, SourceSyntax, SourceVar, SyntaxId},
    TermSchema,
};

define_id!(pub(crate) ReasonSpecId, u32, "A unique identifier for the step in a proof.");

/// Reasons provide extra provenance information accompanying a term being
/// instantiated, or marked as equal to another term. All reasons are pointed
/// to by a row in a terms table.
#[derive(Debug)]
pub(crate) enum ProofReason {
    Rule(RuleData),
    /// Congruence reasons contain the "old" term id that the new term is equal
    /// to. Pairwise equality proofs are rebuilt at proof reconstruction time.
    CongRow,
    /// A row that was created with no added justification (e.g. base values).
    Fiat {
        desc: Arc<str>,
    },
    /// A proof that a term equals itself.
    ///
    /// This is generally only used when two identical terms are created, but with a different term
    /// id (due to concurrency / "term consistency" reasons).
    Refl,
}

impl ProofReason {
    pub(crate) fn arity(&self) -> usize {
        // All start with a proof spec id.
        1 + match self {
            ProofReason::CongRow => 1,
            ProofReason::Rule(data) => data.n_vars(),
            ProofReason::Refl | ProofReason::Fiat { .. } => 0,
        }
    }
}

pub(crate) struct ProofBuilder {
    pub(crate) rule_description: Arc<str>,
    pub(crate) rule_id: RuleId,
    pub(crate) term_vars: DenseIdMap<AtomId, QueryEntry>,
}

pub(crate) struct RebuildVars {
    pub(crate) new_term: Variable,
    pub(crate) reason: Variable,
    pub(crate) before_term: Variable,
}

impl ProofBuilder {
    pub(crate) fn new(description: &str, rule_id: RuleId) -> ProofBuilder {
        ProofBuilder {
            rule_id,
            rule_description: description.into(),
            term_vars: Default::default(),
        }
    }

    /// Generate a proof for a newly rebuilt row.
    pub(crate) fn rebuild_proof(
        &mut self,
        func: FunctionId,
        after: &[QueryEntry],
        vars: RebuildVars,
        db: &mut EGraph,
    ) -> impl Fn(&mut Bindings, &mut RuleBuilder) -> Result<()> + Clone + use<> {
        let reason_spec = ProofReason::CongRow;
        let reason_table = db.reason_table(&reason_spec);
        let reason_spec_id = db.cong_spec;
        let reason_counter = db.reason_counter;
        let func_info = &db.funcs[func];
        let func_table = func_info.table;
        let term_schema = func_info.term_schema();
        let term_table = db.term_table(func_table, func_info.term_has_output);
        let term_counter = db.id_counter;
        let after = after.to_vec();
        move |inner, rb| {
            let old_term = inner.mapping[vars.before_term.id];
            let reason_id = rb.lookup_or_insert(
                reason_table,
                &[Value::new(reason_spec_id.rep()).into(), old_term],
                &[reason_counter.into()],
                ColumnId::new(2),
            )?;
            let mut entries = Vec::new();
            entries.push(Value::new(func.rep()).into());
            for entry in &after[..after.len() - 1] {
                entries.push(inner.convert(entry));
            }
            let ret_val = term_schema
                .has_output
                .then(|| inner.convert(after.last().unwrap()));
            // Now get the new term value, inserting it if the term is new.
            let mut default_vals = Vec::with_capacity(2 + if term_schema.has_output { 1 } else { 0 });
            if let Some(ret_val) = &ret_val {
                default_vals.push(ret_val.clone().into());
            }
            default_vals.push(term_counter.into());
            default_vals.push(reason_id.into());
            let term_id = rb.lookup_or_insert(
                term_table,
                &entries,
                &default_vals,
                ColumnId::from_usize(term_schema.term_id_col()),
            )?;
            inner.mapping.insert(vars.new_term.id, term_id.into());
            inner.mapping.insert(vars.reason.id, reason_id.into());
            Ok(())
        }
    }

    /// Generate a callback that will add a row to the term database, as well as
    /// a reason for that term existing.
    pub(crate) fn new_row(
        &mut self,
        func: FunctionId,
        entries: Vec<QueryEntry>,
        res_id: Option<VariableId>,
        term_var: VariableId,
        db: &mut EGraph,
    ) -> impl Fn(&mut Bindings, &mut RuleBuilder) -> Result<()> + Clone + use<> {
        let func_info = &db.funcs[func];
        let func_table = func_info.table;
        let fiat_reason = func_info.fiat_reason;
        let term_schema = func_info.term_schema();
        let term_table = db.term_table(func_table, func_info.term_has_output);
        let func_val = Value::new(func.rep());
        move |inner, rb| {
            let reason_var: DstVar = if let Some(fiat_reason) = fiat_reason {
                // This table has been marked as "fiat only", meaning that all proofs for this
                // table should have a single hard-coded fiat reason, rather than the ambient used
                // for the rule.
                fiat_reason.into()
            } else {
                inner
                    .lhs_reason
                    .expect("must have a reason variable for new rows")
            };
            let mut translated = Vec::with_capacity(term_schema.total_cols());
            translated.push(func_val.into());
            for entry in &entries[0..entries.len() - 1] {
                translated.push(inner.convert(entry));
            }
            if term_schema.has_output {
                let ret_val = inner.convert(entries.last().unwrap());
                translated.push(ret_val);
            }
            translated.push(inner.mapping[term_var]);
            translated.push(reason_var);
            if let Some(res_id) = res_id {
                rb.insert_if_eq(
                    term_table,
                    inner.mapping[res_id],
                    inner.mapping[term_var],
                    &translated,
                )?;
            } else {
                rb.insert(term_table, &translated)?;
            }
            Ok(())
        }
    }
}

pub(crate) struct ProofReconstructionState<'a> {
    in_progress: HashSet<Value>,
    store: &'a mut ProofStore,
    term_memo: HashMap<(Value, ColumnTy), TermId>,
    term_prf_memo: HashMap<(Value, ColumnTy), TermProofId>,
    eq_memo: HashMap<(Value, Value), EqProofId>,
}

impl<'a> ProofReconstructionState<'a> {
    pub(crate) fn new(store: &'a mut ProofStore) -> ProofReconstructionState<'a> {
        ProofReconstructionState {
            in_progress: HashSet::new(),
            store,
            term_memo: HashMap::new(),
            term_prf_memo: HashMap::new(),
            eq_memo: HashMap::new(),
        }
    }
}

// Proof reconstruction code. A lot of this code assumes that it is running
// outside of the "hot path" for an application; it allocates a lot of small
// vectors, does a good amount of not-exactly-stack-safe recursion, etc.

impl EGraph {
    /// Given the base `subst`, reconstruct the hierarchy of term ids for a given piece of
    /// SourceSyntax.
    fn get_syntax_val(
        &mut self,
        node: SyntaxId,
        syntax: &SourceSyntax,
        subst: &DenseIdMap<VariableId, Value>,
        memo: &mut DenseIdMap<SyntaxId, Value>,
    ) -> Value {
        if let Some(prev) = memo.get(node).copied() {
            return prev;
        }
        let res = match &syntax.backing[node] {
            SourceExpr::Const { val, .. } => *val,
            SourceExpr::Var { id, .. } => subst[*id],
            SourceExpr::ExternalCall { var, .. } => subst[*var],
            SourceExpr::FunctionCall { func, args, .. } => {
                // This is the interesting part.
                //
                // We want to find the term id that corresponds to
                // (func args...).
                let mut row_key = Vec::with_capacity(args.len() + 1);
                row_key.push(Value::new(func.rep()));
                for arg in args {
                    row_key.push(self.get_syntax_val(*arg, syntax, subst, memo));
                }
                let func_info = &self.funcs[*func];
                let term_schema = func_info.term_schema();
                let term_table = self.term_table(func_info.table, func_info.term_has_output);
                let Some(val) = self
                    .db
                    .get_table(term_table)
                    .get_row_column(
                        &row_key,
                        ColumnId::from_usize(term_schema.term_id_col()),
                    )
                else {
                    panic!(
                        "failed to find term for function call ({func:?} {:?}), memo={:?}, arg={node:?}",
                        &row_key[1..],
                        memo
                    )
                };
                val
            }
        };
        memo.insert(node, res);
        res
    }

    fn rule_proof(
        &mut self,
        RuleData { syntax, .. }: &RuleData,
        vars: &[Value],
        state: &mut ProofReconstructionState,
    ) -> (Vec<RuleVarBinding>, Vec<Premise>) {
        // First, reconstruct terms for all the relevant variables.
        let mut subst_term = Vec::with_capacity(syntax.vars.len());
        let mut subst_val = DenseIdMap::<VariableId, Value>::new();
        for (SourceVar { id, ty, name }, term_id) in syntax.vars.iter().zip(vars) {
            subst_val.insert(*id, *term_id);
            let term = self.reconstruct_term(*term_id, *ty, state);
            subst_term.push(RuleVarBinding {
                name: Arc::clone(name),
                ty: *ty,
                term,
            });
        }
        let mut terms = DenseIdMap::<SyntaxId, Value>::new();
        let mut premises = Vec::new();
        for toplevel in &syntax.roots {
            match toplevel {
                TopLevelLhsExpr::Exists(id) => {
                    let val = self.get_syntax_val(*id, syntax, &subst_val, &mut terms);
                    premises.push(Premise::TermOk(self.explain_term_inner(val, state)));
                }
                TopLevelLhsExpr::Eq(id1, id2) => {
                    let lhs = self.get_syntax_val(*id1, syntax, &subst_val, &mut terms);
                    let rhs = self.get_syntax_val(*id2, syntax, &subst_val, &mut terms);
                    premises.push(Premise::Eq(self.explain_terms_equal_inner(lhs, rhs, state)));
                }
            }
        }
        (subst_term, premises)
    }

    pub(crate) fn explain_term_inner(
        &mut self,
        term_id: Value,
        state: &mut ProofReconstructionState,
    ) -> TermProofId {
        let term_id = self.canonicalize_term_id(term_id);
        if let Some(prev) = state.term_prf_memo.get(&(term_id, ColumnTy::Id)) {
            return *prev;
        }
        assert!(
            state.in_progress.insert(term_id),
            "term id {term_id:?} has a cycle in its explanation!"
        );
        let (term_row, term_schema) = self.get_term_row(term_id);
        debug_assert_eq!(term_row[term_schema.term_id_col()], term_id);
        let reason = term_row[term_schema.reason_col()];
        let reason_row = self.get_reason(reason);
        let spec = self.proof_specs[ReasonSpecId::new(reason_row[0].rep())].clone();
        let res = match &*spec {
            ProofReason::Rule(data) => {
                debug_assert_eq!(
                    self.rules[data.rule_id].desc.as_ref(),
                    data.rule_name.as_ref()
                );
                let (subst, body_pfs) = self.rule_proof(data, &reason_row[1..], state);
                let result = self.reconstruct_term(term_id, ColumnTy::Id, state);
                state.store.intern_term(&TermProof::PRule {
                    rule_name: Rc::<str>::from(data.rule_name.as_ref()),
                    subst,
                    body_pfs,
                    result,
                })
            }
            ProofReason::CongRow => {
                let cong = self.create_cong_proof(reason_row[1], term_id, state);
                state.store.intern_term(&TermProof::PCong(cong))
            }
            ProofReason::Fiat { desc } => {
                let term = self.reconstruct_term(term_id, ColumnTy::Id, state);
                state.store.intern_term(&TermProof::PFiat {
                    desc: String::from(&**desc).into(),
                    term,
                })
            }
            ProofReason::Refl => {
                panic!(
                    "Refl cannot be a reason for a term's existence. This is an internal proofs error"
                );
            }
        };

        state.in_progress.remove(&term_id);
        state.term_prf_memo.insert((term_id, ColumnTy::Id), res);
        res
    }

    pub(crate) fn reconstruct_term(
        &mut self,
        term_id: Value,
        ty: ColumnTy,
        state: &mut ProofReconstructionState,
    ) -> TermId {
        let key_id = match ty {
            ColumnTy::Id => self.canonicalize_term_id(term_id),
            ColumnTy::Base(_) => term_id,
        };
        if let Some(cached) = state.term_memo.get(&(key_id, ty)) {
            return *cached;
        }
        let res = match ty {
            ColumnTy::Id => {
                let (term_row, term_schema) = self.get_term_row(key_id);
                let func = FunctionId::new(term_row[0].rep());
                let func_name = self.funcs[func].name.to_string();
                let info = &self.funcs[func];
                // NB: this clone is needed because `get_term_row` borrows the whole egraph, though it
                // really only needs mutable access to `db`. This is of course fixable if we wanted to get
                // rid of the clone.
                let schema = info.schema.clone();
                let input_len = schema.len() - 1;
                let input_slice = &term_row[1..1 + input_len];
                let output_val = term_schema
                    .output_col()
                    .map(|idx| term_row[idx])
                    .unwrap_or_else(|| term_row[term_schema.term_id_col()]);
                let mut args = Vec::with_capacity(schema.len());
                for (ty, entry) in schema[0..schema.len() - 1].iter().zip(input_slice.iter()) {
                    let term = self.reconstruct_term(*entry, *ty, state);
                    args.push(state.store.termdag.get(term).clone());
                }
                let ret_ty = *schema.last().unwrap();
                let ret_term = self.reconstruct_term(output_val, ret_ty, state);
                args.push(state.store.termdag.get(ret_term).clone());
                let app = state.store.termdag.app(func_name, args);
                state.store.termdag.lookup(&app)
            }
            ColumnTy::Base(ty) => {
                let term = if let Some(literal) = self.value_to_literal(&term_id, ty) {
                    state.store.termdag.lit(literal)
                } else {
                    state.store.termdag.unknown_lit()
                };
                state.store.termdag.lookup(&term)
            }
        };

        state.term_memo.insert((key_id, ty), res);
        res
    }

    pub(crate) fn explain_terms_equal_inner(
        &mut self,
        l: Value,
        r: Value,
        state: &mut ProofReconstructionState,
    ) -> EqProofId {
        let l = self.canonicalize_term_id(l);
        let r = self.canonicalize_term_id(r);
        if let Some(prev) = state.eq_memo.get(&(l, r)) {
            return *prev;
        }
        let res = if l == r {
            let term = self.reconstruct_term(l, ColumnTy::Id, state);
            let term_proof = self.explain_term_inner(l, state);
            state.store.refl(term_proof, term)
        } else {
            let uf_table = self
                .db
                .get_table(self.uf_table)
                .as_any()
                .downcast_ref::<DisplacedTableWithProvenance>()
                .unwrap();

            let Some(steps) = uf_table.get_proof(l, r) else {
                panic!(
                    "attempting to explain why two terms ({l:?} and {r:?}) are equal, but they aren't equal"
                );
            };

            assert!(!steps.is_empty(), "empty proof for equality");
            debug_assert!(steps.windows(2).all(|w| {
                let [x, y] = w else { unreachable!() };
                x.rhs == y.lhs
            }));
            debug_assert!(steps.first().is_some_and(|s| s.lhs == l));
            debug_assert!(steps.last().is_some_and(|s| s.rhs == r));

            let subproofs: Vec<_> = steps
                .into_iter()
                .map(|ProofStep { lhs, rhs, reason }| match reason {
                    UfProofReason::Forward(reason) => {
                        self.create_eq_proof_step(reason, lhs, rhs, state)
                    }
                    UfProofReason::Backward(reason) => {
                        let base = self.create_eq_proof_step(reason, rhs, lhs, state);
                        state.store.sym(base)
                    }
                })
                .collect();
            state.store.sequence_proofs(&subproofs)
        };
        state.eq_memo.insert((l, r), res);
        res
    }

    fn create_cong_proof(
        &mut self,
        old_term_id: Value,
        new_term_id: Value,
        state: &mut ProofReconstructionState,
    ) -> CongProof {
        let (old_term_row, term_schema) = self.get_term_row(old_term_id);
        let (new_term_row, new_term_schema) = self.get_term_row(new_term_id);
        debug_assert_eq!(
            term_schema, new_term_schema,
            "term schemas should match for congruent terms"
        );
        let old_term_proof = self.explain_term_inner(old_term_id, state);
        let func_id = FunctionId::new(old_term_row[0].rep());
        debug_assert_eq!(
            old_term_row[0], new_term_row[0],
            "non-matching function ids in a congruence proof"
        );
        let info = &self.funcs[func_id];
        let schema = info.schema.clone();
        let mut args_eq_proofs = Vec::with_capacity(schema.len() - 1);
        let old_term = self.reconstruct_term(old_term_id, ColumnTy::Id, state);
        let new_term = self.reconstruct_term(new_term_id, ColumnTy::Id, state);
        let input_len = schema.len() - 1;
        let old_inputs = &old_term_row[1..1 + input_len];
        let new_inputs = &new_term_row[1..1 + input_len];
        for (i, (ty, (lhs, rhs))) in schema[0..schema.len() - 1]
            .iter()
            .zip(old_inputs.iter().zip(new_inputs.iter()))
            .enumerate()
        {
            let eq_proof = match ty {
                ColumnTy::Id => self.explain_terms_equal_inner(*lhs, *rhs, state),
                ColumnTy::Base(_) => {
                    assert_eq!(lhs, rhs, "congruence proof must have equal base values");
                    let arg_exists = state.store.intern_term(&TermProof::PProj {
                        pf_f_args_ok: old_term_proof,
                        arg_idx: i,
                    });
                    let arg_term = state.store.termdag.proj_id(old_term, i).unwrap();
                    state.store.refl(arg_exists, arg_term)
                }
            };
            args_eq_proofs.push(eq_proof);
        }
        let func_name = Arc::from(self.funcs[func_id].name.as_ref());
        CongProof {
            pf_args_eq: args_eq_proofs,
            pf_f_args_ok: old_term_proof,
            old_term,
            new_term,
            func: func_name,
        }
    }

    fn create_eq_proof_step(
        &mut self,
        reason_id: Value,
        l: Value,
        r: Value,
        state: &mut ProofReconstructionState,
    ) -> EqProofId {
        let reason_row = self.get_reason(reason_id);
        let spec = self.proof_specs[ReasonSpecId::new(reason_row[0].rep())].clone();
        match &*spec {
            ProofReason::Rule(data) => {
                debug_assert_eq!(
                    self.rules[data.rule_id].desc.as_ref(),
                    data.rule_name.as_ref()
                );
                let (subst, body_pfs) = self.rule_proof(data, &reason_row[1..], state);
                let l_term = self.reconstruct_term(l, ColumnTy::Id, state);
                let r_term = self.reconstruct_term(r, ColumnTy::Id, state);
                let rule_name = Rc::<str>::from(data.rule_name.as_ref());
                state.store.intern_eq(&EqProof::PRule {
                    rule_name,
                    subst,
                    body_pfs,
                    result_lhs: l_term,
                    result_rhs: r_term,
                })
            }
            ProofReason::CongRow => {
                let cong = self.create_cong_proof(l, r, state);
                state.store.intern_eq(&EqProof::PCong(cong))
            }
            ProofReason::Fiat { .. } => {
                // NB: we could add this if we wanted to.
                panic!("fiat reason being used to explain equality, rather than a row's existence")
            }
            ProofReason::Refl => {
                let l = self.canonicalize_term_id(l);
                let r = self.canonicalize_term_id(r);
                assert_eq!(l, r, "refl justification for two non-equal terms");
                let t_ok_pf = self.explain_term_inner(l, state);
                let t = self.reconstruct_term(l, ColumnTy::Id, state);
                state.store.intern_eq(&EqProof::PRefl { t_ok_pf, t })
            }
        }
    }

    fn get_term_row(&mut self, term_id: Value) -> (Vec<Value>, TermSchema) {
        let term_id = self.canonicalize_term_id(term_id);
        let mut atom = Vec::<DstVar>::new();
        let mut cur = 0;
        loop {
            // Iterate over the table by index to avoid borrowing issues with the
            // call to `get_proof`.
            let Some(((arity, has_output), table)) = self.term_tables.get_index(cur) else {
                panic!("failed to find term with id {term_id:?}")
            };
            let term_schema = TermSchema {
                key_len: arity + 1,
                has_output: *has_output,
            };

            let gfm_sc = SideChannel::default();
            let gfm_id = self.db.add_external_function(GetFirstMatch(gfm_sc.clone()));
            {
                let mut rsb = self.db.new_rule_set();
                let mut qb = rsb.new_rule();
                for idx in 0..term_schema.total_cols() {
                    if idx == term_schema.term_id_col() {
                        atom.push(term_id.into());
                    } else {
                        atom.push(qb.new_var().into());
                    }
                }
                qb.add_atom(*table, &atom, iter::empty()).unwrap();
                let mut rb = qb.build();
                rb.call_external(gfm_id, &atom).unwrap();
                rb.build();
                let rs = rsb.build();
                atom.clear();
                self.db.run_rule_set(&rs, ReportLevel::TimeOnly);
            }
            self.db.free_external_function(gfm_id);

            if let Some(vals) = gfm_sc.lock().unwrap().take() {
                return (vals, term_schema);
            }
            cur += 1;
        }
    }

    fn get_reason(&mut self, reason_id: Value) -> Vec<Value> {
        let reason_id = self.canonicalize_reason_id(reason_id);
        let mut atom = Vec::<DstVar>::new();
        let mut cur = 0;
        loop {
            // Iterate over the table by index to avoid borrowing issues with the
            // call to `get_proof`.
            let (arity, table) = self
                .reason_tables
                .get_index(cur)
                .unwrap_or_else(|| panic!("failed to find reason with id {reason_id:?}"));

            let gfm_sc = SideChannel::default();
            let gfm_id = self.db.add_external_function(GetFirstMatch(gfm_sc.clone()));
            {
                let mut rsb = self.db.new_rule_set();
                let mut qb = rsb.new_rule();
                for _ in 0..*arity {
                    atom.push(qb.new_var().into());
                }
                atom.push(reason_id.into());
                qb.add_atom(*table, &atom, iter::empty()).unwrap();
                let mut rb = qb.build();
                rb.call_external(gfm_id, &atom).unwrap();
                rb.build();
                let rs = rsb.build();
                atom.clear();
                self.db.run_rule_set(&rs, ReportLevel::TimeOnly);
            }
            self.db.free_external_function(gfm_id);

            if let Some(vals) = gfm_sc.lock().unwrap().take() {
                return vals;
            }
            cur += 1;
        }
    }
}
