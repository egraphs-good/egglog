use std::{iter, rc::Rc, sync::Arc};

use core_relations::{
    BaseValuePrinter, ColumnId, DisplacedTableWithProvenance, ExternalFunctionId,
    ProofReason as UfProofReason, ProofStep, RuleBuilder, Value,
};
use hashbrown::{HashMap, HashSet};
use numeric_id::{define_id, NumericId};

use crate::{
    rule::{Bindings, DstVar, Variable},
    syntax::{Binding, RuleRepresentation, TermFragment},
    term_proof_dag::{BaseValueConstant, EqProof, EqReason, RuleTarget, TermProof, TermValue},
    ColumnTy, EGraph, FunctionId, GetFirstMatch, QueryEntry, Result, RuleId, SideChannel,
};

define_id!(pub(crate) ReasonSpecId, u32, "A unique identifier for the step in a proof.");

/// Reasons provide extra provenance information accompanying a term being
/// instantiated, or marked as equal to another term. All reasons are pointed
/// to by a row in a terms table.
///
#[derive(Debug)]
pub(crate) enum ProofReason {
    Rule(RuleData),
    /// Congrence reasons contain the "old" term id that the new term is equal
    /// to. Pairwise equalty proofs are rebuilt at proof reconstruction time.
    CongRow,
    /// A row that was created with no added justification (e.g. base values).
    Fiat {
        desc: Arc<str>,
    },
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum CaonicalIdRef {
    Mapped(usize),
    WithinAtom { atom: AtomIndex, proj: usize },
}

#[derive(Debug)]
pub(crate) struct RuleData {
    rule_id: RuleId,
    desc: Arc<str>,
    insert_to: Insertable,
    /// The atoms on the LHS of the rule.
    lhs_atoms: Vec<Vec<QueryEntry>>,
    /// Any atoms implicitly introduced in the RHS of the rule, before `dst_atom`.
    rhs_atoms: Vec<Vec<QueryEntry>>,
    /// The Atom this is a proof of
    dst_atom: Vec<QueryEntry>,

    /// Rule-based proofs include a set of "canonical" ids that are used to
    /// ground other terms in the rest of the match. For example, if a rule
    /// looks like:
    ///
    /// > (= x (add a b))
    /// > (= x (mul a 2))
    /// > =>
    /// > (f x)
    ///
    /// It's possible that the underlying term reconstructed for `(f x)` will
    /// look like either `(add a b)`, or `(mul a 2)`, or something else entirely
    /// (`(add b a)`, say).
    ///
    /// These canonical ids are recorded in the RuleData to guide generation of
    /// equality proofs.
    n_canonical: usize,
    canonical_mapping: HashMap<AtomIndex, CaonicalIdRef>,
    /// The number of LHS bindings in scope for this insertion.
    lhs_bindings: usize,
    /// The number of RHS bindings in scope for this insertion.
    rhs_bindings: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum AtomIndex {
    Lhs(usize),
    Rhs(usize),
}

impl ProofReason {
    pub(crate) fn arity(&self) -> usize {
        // All start with a proof spec id.
        1 + match self {
            ProofReason::Rule(RuleData {
                lhs_atoms,
                rhs_atoms,
                n_canonical,
                ..
            }) => *n_canonical + lhs_atoms.len() + rhs_atoms.len(),
            ProofReason::CongRow => 1,
            ProofReason::Fiat { .. } => 0,
        }
    }
}

/// The sort of insertion that an existence proof may correspond to.
#[derive(Debug, Copy, Clone)]
pub(crate) enum Insertable {
    /// An insertion into a function / table.
    Func(FunctionId),
    /// An egglog `union`.
    UnionFind,
}

pub(crate) type SyntaxEnv = HashMap<Variable, Arc<TermFragment<Variable>>>;

pub(crate) struct ProofBuilder {
    pub(crate) rule_description: Arc<str>,
    rule_id: RuleId,
    lhs_atoms: Vec<Vec<QueryEntry>>,
    /// The atom against which to compare during proofs. Serves as a guide for
    /// generating equality proofs.
    to_compare: Vec<usize>,
    rhs_atoms: Vec<Vec<QueryEntry>>,
    lhs_term_vars: Vec<Variable>,
    rhs_term_vars: Vec<Variable>,
    representatives: HashMap<Variable, usize>,
    // The "syntax" fields are used to build a checker for the proofs in this
    // rule.
    pub(crate) syntax_env: SyntaxEnv,
    pub(crate) syntax: RuleRepresentation,
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
            lhs_atoms: Default::default(),
            rhs_atoms: Default::default(),
            to_compare: Default::default(),
            lhs_term_vars: Default::default(),
            rhs_term_vars: Default::default(),
            representatives: Default::default(),
            syntax_env: Default::default(),
            syntax: Default::default(),
        }
    }

    pub(crate) fn add_lhs(&mut self, entries: &[QueryEntry], term_var: Variable) {
        let id_var = entries.last().expect("entries should be nonempty").var();
        let to_compare = *self
            .representatives
            .entry(id_var)
            .or_insert(self.lhs_atoms.len());
        self.lhs_atoms.push(entries.to_vec());
        self.to_compare.push(to_compare);
        self.lhs_term_vars.push(term_var);
    }
    pub(crate) fn add_rhs(&mut self, entries: &[QueryEntry], term_var: Variable) {
        self.rhs_atoms.push(entries.to_vec());
        self.rhs_term_vars.push(term_var);
    }

    fn canonical_mappings(&self) -> (Vec<AtomIndex>, HashMap<AtomIndex, CaonicalIdRef>) {
        let mut result = HashMap::new();
        let mut atoms_to_materialize = Vec::new();
        #[derive(Default)]
        struct VariableMention {
            as_term: Vec<AtomIndex>,
            as_subterm: Vec<(AtomIndex, usize)>,
        }
        impl VariableMention {
            fn update_term(&mut self, ix: AtomIndex) {
                self.as_term.push(ix);
            }
            fn update_subterm(&mut self, ix: AtomIndex, proj: usize) {
                self.as_subterm.push((ix, proj));
            }
        }
        let mut mapping = HashMap::<Variable, VariableMention>::new();
        for (atom_ix, atom) in self
            .lhs_atoms
            .iter()
            .enumerate()
            .map(|(ix, atom)| (AtomIndex::Lhs(ix), atom))
            .chain(
                self.rhs_atoms
                    .iter()
                    .enumerate()
                    .map(|(ix, atom)| (AtomIndex::Rhs(ix), atom)),
            )
        {
            mapping
                .entry(atom.last().unwrap().var())
                .or_default()
                .update_term(atom_ix);
            for (col, entry) in atom[..atom.len() - 1].iter().enumerate() {
                let QueryEntry::Var { id, .. } = entry else {
                    continue;
                };
                mapping.entry(*id).or_default().update_subterm(atom_ix, col);
            }
        }
        for (_, mentions) in mapping {
            let Some(first_term_atom) = mentions.as_term.first().copied() else {
                // If the only mentions for a variable are as a subterm, then
                // there's nothing to do.
                continue;
            };
            let id_ref = if let Some((atom, proj)) = mentions.as_subterm.first().copied() {
                // If we have a subterm-level reference, we can reuse that to
                // find the id. No need to write it in the canonical id mapping.
                CaonicalIdRef::WithinAtom { atom, proj }
            } else {
                // We need to instantiate a new canonical id for this one, it is
                // only referenced by "term heads".
                let next_id = atoms_to_materialize.len();
                atoms_to_materialize.push(first_term_atom);
                CaonicalIdRef::Mapped(next_id)
            };
            for term_ref in &mentions.as_term {
                result.insert(*term_ref, id_ref);
            }
        }
        (atoms_to_materialize, result)
    }

    /// Generate a proof for a newly rebuilt row.
    pub(crate) fn rebuild_proof(
        &mut self,
        func: FunctionId,
        after: &[QueryEntry],
        vars: RebuildVars,
        db: &mut EGraph,
    ) -> impl Fn(&mut Bindings, &mut RuleBuilder) -> Result<()> + Clone {
        // TODO/optimization: we only ever need one CongRow reason.
        let reason_spec = Arc::new(ProofReason::CongRow);
        let reason_table = db.reason_table(&reason_spec);
        let reason_spec_id = db.proof_specs.push(reason_spec);
        let reason_counter = db.reason_counter;
        let func_table = db.funcs[func].table;
        let term_table = db.term_table(func_table);
        let term_counter = db.id_counter;
        let after = after.to_vec();
        move |inner, rb| {
            let old_term = inner.mapping[vars.before_term];
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
            // Now get the new term value, inserting it if the term is new.
            let term_result = rb.lookup_or_insert(
                term_table,
                &entries,
                &[term_counter.into(), reason_id.into()],
                ColumnId::from_usize(entries.len()),
            )?;
            inner.mapping.insert(vars.new_term, term_result.into());
            inner.mapping.insert(vars.reason, reason_id.into());
            Ok(())
        }
    }

    fn make_reason(
        &mut self,
        insert_to: Insertable,
        dst_atom: &[QueryEntry],
        reason_var: Variable,
        db: &mut EGraph,
    ) -> impl Fn(&mut Bindings, &mut RuleBuilder) -> Result<()> + Clone {
        // NB: we could cache these.
        let (to_materialize, canonical_mapping) = self.canonical_mappings();
        let spec = Arc::new(ProofReason::Rule(RuleData {
            desc: self.rule_description.clone(),
            insert_to,
            rule_id: self.rule_id,
            lhs_atoms: self.lhs_atoms.clone(),
            rhs_atoms: self.rhs_atoms.clone(),
            n_canonical: to_materialize.len(),
            canonical_mapping,
            dst_atom: dst_atom.to_vec(),
            lhs_bindings: self.syntax.lhs_bindings.len(),
            rhs_bindings: self.syntax.rhs_bindings.len(),
        }));
        let spec_table = db.reason_table(&spec);
        let spec_id = db.proof_specs.push(spec);
        let proof_vars: Vec<Variable> = to_materialize
            .iter()
            .map(|atom_ix| match atom_ix {
                AtomIndex::Lhs(ix) => self.lhs_atoms[*ix].last().unwrap().var(),
                AtomIndex::Rhs(ix) => self.rhs_atoms[*ix].last().unwrap().var(),
            })
            .chain(
                self.lhs_term_vars
                    .iter()
                    .copied()
                    .chain(self.rhs_term_vars.iter().copied()),
            )
            .collect();
        let reason_counter = db.reason_counter;
        move |inner, rb| {
            let mut args = Vec::with_capacity(proof_vars.len() + 1);
            args.push(Value::new(spec_id.rep()).into());
            for var in &proof_vars {
                args.push(inner.mapping[*var]);
            }
            let x = rb.lookup_or_insert(
                spec_table,
                &args,
                &[reason_counter.into()],
                ColumnId::from_usize(args.len()),
            )?;
            inner.mapping.insert(reason_var, x.into());
            Ok(())
        }
    }

    /// Generate a callback that will add a proof reason to the database and
    /// bind a pointer to that reason to `reason_var`.
    pub(crate) fn union(
        &mut self,
        l: QueryEntry,
        r: QueryEntry,
        reason_var: Variable,
        db: &mut EGraph,
    ) -> impl Fn(&mut Bindings, &mut RuleBuilder) -> Result<()> + Clone {
        self.make_reason(Insertable::UnionFind, &[l, r], reason_var, db)
    }

    /// Generate a callback that will add a row to the term database, as well as
    /// a reason for that term existing.
    pub(crate) fn new_row(
        &mut self,
        func: FunctionId,
        entries: Vec<QueryEntry>,
        term_var: Variable,
        reason_var: Variable,
        db: &mut EGraph,
    ) -> impl Fn(&mut Bindings, &mut RuleBuilder) -> Result<()> + Clone {
        let make_reason = self.make_reason(Insertable::Func(func), &entries, reason_var, db);
        self.add_rhs(&entries, term_var);
        let func_table = db.funcs[func].table;
        let term_table = db.term_table(func_table);
        let func_val = Value::new(func.rep());
        let res_var = entries.last().unwrap().var();
        let rhs_term = Arc::new(TermFragment::App(
            func,
            entries[..entries.len() - 1]
                .iter()
                .map(|e| e.to_syntax(db).unwrap())
                .collect(),
        ));
        self.syntax_env
            .entry(res_var)
            .or_insert_with(|| rhs_term.clone());
        self.syntax.rhs_bindings.push(Binding {
            var: res_var,
            syntax: rhs_term,
        });
        move |inner, rb| {
            make_reason(inner, rb)?;
            let mut translated = Vec::new();
            translated.push(func_val.into());
            for entry in &entries[0..entries.len() - 1] {
                translated.push(inner.convert(entry));
            }
            translated.push(inner.mapping[term_var]);
            translated.push(inner.mapping[reason_var]);
            rb.insert(term_table, &translated)?;
            Ok(())
        }
    }
    pub(crate) fn register_prim(
        &mut self,
        func: ExternalFunctionId,
        args: &[QueryEntry],
        res: Variable,
        ty: ColumnTy,
        db: &EGraph,
    ) {
        let app = Arc::new(TermFragment::Prim(
            func,
            args.iter().map(|v| v.to_syntax(db).unwrap()).collect(),
            ty,
        ));
        assert!(self.syntax_env.insert(res, app.clone()).is_none());
        self.syntax.rhs_bindings.push(Binding {
            var: res,
            syntax: app,
        });
    }
}

#[derive(Default)]
pub(crate) struct ProofReconstructionState {
    in_progress: HashSet<Value>,
    term_memo: HashMap<Value, Rc<TermProof>>,
    eq_memo: HashMap<(Value, Value), Rc<EqProof>>,

    // maps for "hash-consing" proofs
    id: HashMap<*const TermProof, Rc<EqProof>>,
    backwards: HashMap<*const EqProof, Rc<EqProof>>,
    base: HashMap<(*const TermProof, *const TermProof, *const TermProof), Rc<EqProof>>,
}

impl ProofReconstructionState {
    fn id(&mut self, term: Rc<TermProof>) -> Rc<EqProof> {
        self.id
            .entry(term.as_ref() as *const _)
            .or_insert_with(move || {
                Rc::new(EqProof {
                    lhs: term.clone(),
                    rhs: term.clone(),
                    reason: EqReason::Id(term),
                })
            })
            .clone()
    }

    fn backwards(&mut self, eq: Rc<EqProof>) -> Rc<EqProof> {
        self.backwards
            .entry(eq.as_ref() as *const _)
            .or_insert_with(move || {
                Rc::new(EqProof {
                    lhs: eq.rhs.clone(),
                    rhs: eq.lhs.clone(),
                    reason: EqReason::Backwards(eq),
                })
            })
            .clone()
    }

    fn base(&mut self, term: Rc<TermProof>, lhs: Rc<TermProof>, rhs: Rc<TermProof>) -> Rc<EqProof> {
        self.base
            .entry((
                term.as_ref() as *const _,
                lhs.as_ref() as *const _,
                rhs.as_ref() as *const _,
            ))
            .or_insert_with(move || {
                Rc::new(EqProof {
                    lhs,
                    rhs,
                    reason: EqReason::Base(term),
                })
            })
            .clone()
    }
}

// Proof reconstruction code. A lot of this code assumes that it is running
// outside of the "hot path" for an application; it allocates a lot of small
// vectors, does a good amount of not-exactly-stack-safe recursion, etc.

impl EGraph {
    pub(crate) fn explain_term_inner(
        &mut self,
        term_id: Value,
        state: &mut ProofReconstructionState,
    ) -> Rc<TermProof> {
        if let Some(prev) = state.term_memo.get(&term_id) {
            return prev.clone();
        }
        assert!(
            state.in_progress.insert(term_id),
            "term id {term_id:?} has a cycle in its explanation!"
        );
        let term_row = self.get_term_row(term_id);
        debug_assert_eq!(term_row[term_row.len() - 2], term_id);
        // We have something like (F x y) => `term_id` + `reason`
        // There are two things to do at this juncture:
        // 1. Explain the children (namely `x` and `y`)
        // 2. Explain anything about `reason`.

        let reason = *term_row.last().unwrap();
        let reason_row = self.get_reason(reason);
        let spec = self.proof_specs[ReasonSpecId::new(reason_row[0].rep())].clone();
        let res = match &*spec {
            ProofReason::Rule(data) => {
                self.create_rule_proof(data, &term_row[1..term_row.len() - 2], &reason_row, state)
            }
            ProofReason::CongRow => self.create_cong_proof(reason_row[1], term_id, state),
            ProofReason::Fiat { desc } => {
                let func_id = FunctionId::new(term_row[0].rep());
                let info = &self.funcs[func_id];
                let schema = info.schema.clone();
                let func = info.name.clone();
                let desc = desc.clone();
                let row = self.get_term_values(
                    term_row[1..].iter().zip(schema[0..schema.len() - 1].iter()),
                    state,
                );
                Rc::new(TermProof::Fiat {
                    desc,
                    func,
                    row,
                    func_id,
                })
            }
        };

        state.in_progress.remove(&term_id);
        state.term_memo.insert(term_id, res.clone());
        res
    }

    pub(crate) fn explain_terms_equal_inner(
        &mut self,
        l: Value,
        r: Value,
        state: &mut ProofReconstructionState,
    ) -> Rc<EqProof> {
        if let Some(prev) = state.eq_memo.get(&(l, r)) {
            return prev.clone();
        }
        #[allow(clippy::never_loop)]
        let res = loop {
            // We are using a loop as a block that we can break out of.
            if l == r {
                let term_proof = self.explain_term_inner(l, state);
                break state.id(term_proof);
            }
            let uf_table = self
                .db
                .get_table(self.uf_table)
                .as_any()
                .downcast_ref::<DisplacedTableWithProvenance>()
                .unwrap();

            let Some(steps) = uf_table.get_proof(l, r) else {
                panic!("attempting to explain why two terms ({l:?} and {r:?}) are equal, but they aren't equal");
            };

            assert!(!steps.is_empty(), "empty proof for equality");

            break if steps.len() == 1 {
                let ProofStep { lhs, rhs, reason } = &steps[0];
                assert_eq!(*lhs, l);
                assert_eq!(*rhs, r);
                match reason {
                    UfProofReason::Forward(reason) => {
                        self.create_eq_proof_step(*reason, *lhs, *rhs, state)
                    }
                    UfProofReason::Backward(reason) => {
                        let base = self.create_eq_proof_step(*reason, *rhs, *lhs, state);
                        state.backwards(base)
                    }
                }
            } else {
                assert!(
                    steps
                        .iter()
                        .zip(steps.iter().skip(1))
                        .all(|(x, y)| x.rhs == y.lhs),
                    "malformed proofs out of UF: {steps:?}"
                );
                let subproofs: Vec<_> = steps
                    .into_iter()
                    .map(|ProofStep { lhs, rhs, reason }| match reason {
                        UfProofReason::Forward(reason) => {
                            self.create_eq_proof_step(reason, lhs, rhs, state)
                        }
                        UfProofReason::Backward(reason) => {
                            let base = self.create_eq_proof_step(reason, rhs, lhs, state);
                            state.backwards(base)
                        }
                    })
                    .collect();
                let lhs = subproofs[0].lhs.clone();
                let rhs = subproofs.last().unwrap().rhs.clone();
                Rc::new(EqProof {
                    lhs,
                    rhs,
                    reason: EqReason::Trans(subproofs),
                })
            };
        };
        state.eq_memo.insert((l, r), res.clone());
        res
    }

    fn create_cong_proof(
        &mut self,
        old_term_id: Value,
        new_term_id: Value,
        state: &mut ProofReconstructionState,
    ) -> Rc<TermProof> {
        let old_term = self.get_term_row(old_term_id);
        let old_term_proof = self.explain_term_inner(old_term_id, state);
        let new_term = self.get_term_row(new_term_id);
        let func_id = FunctionId::new(old_term[0].rep());
        let info = &self.funcs[func_id];
        let func: Arc<str> = info.name.clone();
        let schema = info.schema.clone();
        let pairwise_eq = self.lift_to_values(
            old_term[1..]
                .iter()
                .zip(new_term[1..].iter())
                .zip(schema[0..schema.len() - 1].iter()),
            |slf, state, (old, new)| slf.explain_terms_equal_inner(*old, *new, state),
            |(old, new)| {
                assert_eq!(*old, *new, "base values must be equal");
                *old
            },
            state,
        );
        Rc::new(TermProof::Cong {
            func_id,
            func,
            old_term: old_term_proof,
            pairwise_eq,
        })
    }

    fn project_value(&mut self, val: Value, col: usize) -> Value {
        let row = self.get_term_row(val);
        row[col + 1]
    }

    fn create_rule_proof(
        &mut self,
        data: &RuleData,
        trunc_term_row: &[Value],
        reason_row: &[Value],
        state: &mut ProofReconstructionState,
    ) -> Rc<TermProof> {
        let RuleData {
            desc,
            insert_to,
            lhs_atoms,
            rhs_atoms,
            rule_id,
            dst_atom,
            n_canonical,
            canonical_mapping,
            lhs_bindings,
            rhs_bindings,
        } = data;
        let rule_desc = desc.clone();
        let atom_desc: Rc<str> = format!("{dst_atom:?}").into();
        let (func_id, func, schema) = match insert_to {
            Insertable::Func(f) => (
                RuleTarget::Row(
                    *f,
                    Rc::new(TermFragment::App(
                        *f,
                        dst_atom[0..dst_atom.len() - 1]
                            .iter()
                            .map(|entry| entry.to_syntax(self).unwrap())
                            .collect(),
                    )),
                ),
                self.funcs[*f].name.clone(),
                self.funcs[*f].schema.clone(),
            ),
            Insertable::UnionFind => (
                RuleTarget::Union,
                "union".into(),
                vec![ColumnTy::Id, ColumnTy::Id],
            ),
        };
        let row: Vec<_> = self.get_term_values(trunc_term_row.iter().zip(schema.iter()), state);
        let prem_term_ids = &reason_row[1 + *n_canonical..reason_row.len() - 1];
        let premises: Vec<_> = reason_row[1 + *n_canonical..reason_row.len() - 1]
            .iter()
            .map(|prem| self.explain_term_inner(*prem, state))
            .collect();
        let get_premise = |ix: AtomIndex| match ix {
            AtomIndex::Lhs(ix) => prem_term_ids[ix],
            AtomIndex::Rhs(ix) => prem_term_ids[lhs_atoms.len() + ix],
        };
        let premises_eq: Vec<Rc<EqProof>> = (0..lhs_atoms.len())
            .map(AtomIndex::Lhs)
            .chain((0..rhs_atoms.len()).map(AtomIndex::Rhs))
            .map(|atom_ix| match canonical_mapping[&atom_ix] {
                CaonicalIdRef::Mapped(i) => {
                    let l = reason_row[1 + i];
                    let r = get_premise(atom_ix);
                    self.explain_terms_equal_inner(l, r, state)
                }
                CaonicalIdRef::WithinAtom { atom, proj } => {
                    let atom_id = get_premise(atom);
                    let l = self.project_value(atom_id, proj);
                    let r = get_premise(atom_ix);
                    self.explain_terms_equal_inner(l, r, state)
                }
            })
            .collect();
        Rc::new(TermProof::FromRule {
            rule_id: *rule_id,
            func_id,
            rule_desc,
            atom_desc,
            func,
            row,
            premises,
            premises_eq,
            lhs_atoms: *lhs_bindings,
            rhs_atoms: *rhs_bindings,
        })
    }

    fn create_eq_proof_step(
        &mut self,
        reason_id: Value,
        l: Value,
        r: Value,
        state: &mut ProofReconstructionState,
    ) -> Rc<EqProof> {
        let reason_row = self.get_reason(reason_id);
        let spec = self.proof_specs[ReasonSpecId::new(reason_row[0].rep())].clone();
        let l_term = self.explain_term_inner(l, state);
        match &*spec {
            ProofReason::Rule(data) => {
                assert!(
                    matches!(data.insert_to, Insertable::UnionFind),
                    "non-UF insertion being used to explain equality"
                );
                let r_term = self.explain_term_inner(r, state);
                let term_proof = self.create_rule_proof(data, &[l, r], &reason_row, state);
                state.base(term_proof, l_term, r_term)
            }
            ProofReason::CongRow => {
                let l_term = self.explain_term_inner(l, state);
                let proof = self.create_cong_proof(l, r, state);
                state.base(proof.clone(), l_term, proof)
            }
            ProofReason::Fiat { .. } => {
                // NB: we could add this if we wanted to.
                panic!("fiat reason being used to explain equality, rather than a row's existence")
            }
        }
    }

    fn get_term_values<'a, 'b>(
        &mut self,
        term_with_schema: impl Iterator<Item = (&'a Value, &'b ColumnTy)>,
        state: &mut ProofReconstructionState,
    ) -> Vec<TermValue<TermProof>> {
        self.lift_to_values(
            term_with_schema,
            |slf, state, child| slf.explain_term_inner(*child, state),
            |v| *v,
            state,
        )
    }

    fn lift_to_values<'a, 'b, T, R>(
        &mut self,
        term_with_schema: impl Iterator<Item = (T, &'b ColumnTy)>,
        mut f: impl FnMut(&mut EGraph, &mut ProofReconstructionState, T) -> Rc<R>,
        mut to_value: impl FnMut(T) -> Value,
        state: &mut ProofReconstructionState,
    ) -> Vec<TermValue<R>> {
        term_with_schema
            .map(|(child, ty)| match ty {
                ColumnTy::Id => TermValue::SubTerm(f(self, state, child)),
                ColumnTy::Base(p) => {
                    let interned = to_value(child);
                    TermValue::Base(BaseValueConstant {
                        ty: *p,
                        interned,
                        rendered: format!(
                            "{:?}",
                            BaseValuePrinter {
                                base: self.db.base_values(),
                                ty: *p,
                                val: interned,
                            }
                        )
                        .into(),
                    })
                }
            })
            .collect()
    }

    fn get_term_row(&mut self, term_id: Value) -> Vec<Value> {
        let mut atom = Vec::<DstVar>::new();
        let mut cur = 0;
        loop {
            // Iterate over the table by index to avoid borrowing issues with the
            // call to `get_proof`.
            let Some((keys, table)) = self.term_tables.get_index(cur) else {
                panic!("failed to find term with id {term_id:?}")
            };

            let gfm_sc = SideChannel::default();
            let gfm_id = self.db.add_external_function(GetFirstMatch(gfm_sc.clone()));
            {
                let mut rsb = self.db.new_rule_set();
                let mut qb = rsb.new_rule();
                for _ in 0..*keys + 1 {
                    atom.push(qb.new_var().into());
                }
                atom.push(term_id.into());
                atom.push(qb.new_var().into()); // reason
                qb.add_atom(*table, &atom, iter::empty()).unwrap();
                let mut rb = qb.build();
                rb.call_external(gfm_id, &atom).unwrap();
                rb.build();
                let rs = rsb.build();
                atom.clear();
                self.db.run_rule_set(&rs);
            }
            self.db.free_external_function(gfm_id);

            if let Some(vals) = gfm_sc.lock().unwrap().take() {
                return vals;
            }
            cur += 1;
        }
    }

    fn get_reason(&mut self, reason_id: Value) -> Vec<Value> {
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
                self.db.run_rule_set(&rs);
            }
            self.db.free_external_function(gfm_id);

            if let Some(vals) = gfm_sc.lock().unwrap().take() {
                return vals;
            }
            cur += 1;
        }
    }
}
