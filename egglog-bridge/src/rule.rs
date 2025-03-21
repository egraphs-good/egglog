//! APIs for building egglog rules.
//!
//! Egglog rules are ultimately just (sets of) `core-relations` rules
//! parameterized by a range of timestamps used as constraints during seminaive
//! evaluation.

use std::{cmp::Ordering, sync::Arc};

use anyhow::Context;
use core_relations::{
    ColumnId, Constraint, CounterId, ExternalFunctionId, PlanStrategy, PrimitivePrinter,
    QueryBuilder, RuleBuilder as CoreRuleBuilder, RuleSetBuilder, TableId, Value, WriteVal,
};
use hashbrown::HashSet;
use log::debug;
use numeric_id::{define_id, DenseIdMap, NumericId};
use smallvec::SmallVec;
use thiserror::Error;

use crate::syntax::{Binding, Entry, Statement, TermFragment};
use crate::term_proof_dag::PrimitiveConstant;
use crate::{
    proof_spec::{ProofBuilder, RebuildVars},
    ColumnTy, DefaultVal, EGraph, FunctionId, Result, RuleId, RuleInfo, Timestamp,
};

define_id!(pub Variable, u32, "A variable in an egglog query");
pub(crate) type DstVar = core_relations::QueryEntry;

#[derive(Debug, Error)]
enum RuleBuilderError {
    #[error("type mismatch: expected {expected:?}, got {got:?}")]
    TypeMismatch { expected: ColumnTy, got: ColumnTy },
    #[error("arity mismatch: expected {expected:?}, got {got:?}")]
    ArityMismatch { expected: usize, got: usize },
}

#[derive(Clone)]
struct VarInfo {
    ty: ColumnTy,
    /// If there is a "term-level" variant of this variable bound elsewhere, it
    /// is stored here. Otherwise, this points back to the variable itself.
    term_var: Variable,
}

#[derive(Clone, Debug)]
pub enum QueryEntry {
    Var {
        id: Variable,
        name: Option<Box<str>>,
    },
    Const {
        val: Value,
        // Constants can have a type plumbed through, particularly if they
        // correspond to a primitive constant in egglog.
        ty: ColumnTy,
    },
}

impl QueryEntry {
    /// Get the variable associated with this entry, panicking if it isn't a
    /// variable.
    pub(crate) fn var(&self) -> Variable {
        match self {
            QueryEntry::Var { id, .. } => *id,
            QueryEntry::Const { .. } => panic!("expected variable, found constant"),
        }
    }

    pub(crate) fn to_syntax(&self, eg: &EGraph) -> Option<Entry<Variable>> {
        Some(match self {
            QueryEntry::Var { id, .. } => Entry::Placeholder(*id),
            QueryEntry::Const { val, ty } => {
                let ColumnTy::Primitive(ty) = *ty else {
                    panic!("expected primitive type, found {:?}", ty);
                };
                let interned = *val;

                Entry::Const(PrimitiveConstant {
                    interned,
                    ty,
                    rendered: format!(
                        "{:?}",
                        PrimitivePrinter {
                            val: interned,
                            ty,
                            prim: eg.db.primitives(),
                        }
                    )
                    .into(),
                })
            }
        })
    }
}

impl From<Variable> for QueryEntry {
    fn from(id: Variable) -> Self {
        QueryEntry::Var { id, name: None }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Function {
    Table(FunctionId),
    Prim(ExternalFunctionId),
}

impl From<FunctionId> for Function {
    fn from(f: FunctionId) -> Self {
        Function::Table(f)
    }
}

impl From<ExternalFunctionId> for Function {
    fn from(f: ExternalFunctionId) -> Self {
        Function::Prim(f)
    }
}

trait Brc: Fn(&mut Bindings, &mut CoreRuleBuilder) -> Result<()> + dyn_clone::DynClone + Send {}
impl<T: Fn(&mut Bindings, &mut CoreRuleBuilder) -> Result<()> + Clone + Send> Brc for T {}
dyn_clone::clone_trait_object!(Brc);
type BuildRuleCallback = Box<dyn Brc>;

#[derive(Clone)]
pub(crate) struct Query {
    uf_table: TableId,
    id_counter: CounterId,
    tracing: bool,
    rule_id: RuleId,
    vars: DenseIdMap<Variable, VarInfo>,
    /// The current proofs that are in scope.
    atom_proofs: Vec<Variable>,
    atoms: Vec<(TableId, Vec<QueryEntry>)>,
    /// The builders for queries in this module essentially wrap the lower-level
    /// builders from the `core_relations` crate. A single egglog rule can turn
    /// into N core-relations rules. The code is structured by constructing a
    /// series of callbacks that will iteratively build up a low-level rule that
    /// looks like the high-level rule, passing along an environment that keeps
    /// track of the mappings between low and high-level variables.
    add_rule: Vec<BuildRuleCallback>,
    /// If set, execute a single rule (rather than O(atoms.len()) rules) during
    /// seminaive, with the given atom as the focus.
    sole_focus: Option<usize>,
    seminaive: bool,
    plan_strategy: PlanStrategy,
}

pub struct RuleBuilder<'a> {
    egraph: &'a mut EGraph,
    proof_builder: ProofBuilder,
    query: Query,
}

impl EGraph {
    /// Add a rewrite rule for this [`EGraph`] using a [`RuleBuilder`].
    /// If you aren't sure, use `egraph.new_rule("", true)`.
    pub fn new_rule(&mut self, desc: &str, seminaive: bool) -> RuleBuilder {
        let uf_table = self.uf_table;
        let id_counter = self.id_counter;
        let tracing = self.tracing;
        let rule_id = self.rules.reserve_slot();
        RuleBuilder {
            egraph: self,
            proof_builder: ProofBuilder::new(desc, rule_id),
            query: Query {
                uf_table,
                id_counter,
                tracing,
                rule_id,
                seminaive,
                sole_focus: None,
                atom_proofs: Default::default(),
                vars: Default::default(),
                atoms: Default::default(),
                add_rule: Default::default(),
                plan_strategy: Default::default(),
            },
        }
    }

    /// Remove a rewrite rule from this [`EGraph`].
    pub fn free_rule(&mut self, id: RuleId) {
        self.rules.take(id);
    }
}

impl RuleBuilder<'_> {
    fn add_callback(&mut self, cb: impl Brc + 'static) {
        self.query.add_rule.push(Box::new(cb));
    }

    /// Access the underlying egraph within the builder.
    pub fn egraph(&self) -> &EGraph {
        self.egraph
    }

    pub(crate) fn set_plan_strategy(&mut self, strategy: PlanStrategy) {
        self.query.plan_strategy = strategy;
    }

    /// Get the canonical value of an id in the union-find. An internal-only
    /// routine used to implement rebuilding.
    ///
    /// Note, calling this with a non-Id entry can cause errors at rule runtime
    /// (The derived rules will not compile).
    pub(crate) fn lookup_uf(&mut self, entry: QueryEntry) -> Result<Variable> {
        let res = self.new_var(ColumnTy::Id);
        let uf_table = self.query.uf_table;
        self.assert_has_ty(&entry, ColumnTy::Id)
            .context("lookup_uf: ")?;
        self.add_callback(move |inner, rb| {
            let entry = inner.convert(&entry);
            let res_inner = rb.lookup_with_default(uf_table, &[entry], entry, ColumnId::new(1))?;
            inner.mapping.insert(res, res_inner.into());
            Ok(())
        });
        Ok(res)
    }

    /// A low-level routine used in rebuilding. Halts execution if `lhs` and
    /// `rhs` are equal (pointwise).
    ///
    /// Note, calling this with invalid arguments (e.g. different lengths for
    /// `lhs` and `rhs`) can cause errors at rule runtime.
    pub(crate) fn check_for_update(
        &mut self,
        lhs: &[QueryEntry],
        rhs: &[QueryEntry],
    ) -> Result<()> {
        let lhs = SmallVec::<[QueryEntry; 4]>::from_iter(lhs.iter().cloned());
        let rhs = SmallVec::<[QueryEntry; 4]>::from_iter(rhs.iter().cloned());
        if lhs.len() != rhs.len() {
            return Err(RuleBuilderError::ArityMismatch {
                expected: lhs.len(),
                got: rhs.len(),
            }
            .into());
        }
        lhs.iter().zip(rhs.iter()).try_for_each(|(l, r)| {
            self.assert_same_ty(l, r).with_context(|| {
                format!("check_for_update: {lhs:?} and {rhs:?}, mismatch between {l:?} and {r:?}")
            })
        })?;

        self.add_callback(move |inner, rb| {
            let lhs = inner.convert_all(&lhs);
            let rhs = inner.convert_all(&rhs);
            rb.assert_any_ne(&lhs, &rhs).context("check_for_update")
        });
        Ok(())
    }

    fn assert_same_ty(
        &self,
        l: &QueryEntry,
        r: &QueryEntry,
    ) -> std::result::Result<(), RuleBuilderError> {
        match (l, r) {
            (QueryEntry::Var { id: v1, .. }, QueryEntry::Var { id: v2, .. }) => {
                let ty1 = self.query.vars[*v1].ty;
                let ty2 = self.query.vars[*v2].ty;
                if ty1 != ty2 {
                    return Err(RuleBuilderError::TypeMismatch {
                        expected: ty1,
                        got: ty2,
                    });
                }
            }
            // constants can be untyped
            (QueryEntry::Const { .. }, QueryEntry::Const { .. })
            | (QueryEntry::Var { .. }, QueryEntry::Const { .. })
            | (QueryEntry::Const { .. }, QueryEntry::Var { .. }) => {}
        }
        Ok(())
    }

    fn assert_has_ty(
        &self,
        entry: &QueryEntry,
        ty: ColumnTy,
    ) -> std::result::Result<(), RuleBuilderError> {
        if let QueryEntry::Var { id: v, .. } = entry {
            let var_ty = self.query.vars[*v].ty;
            if var_ty != ty {
                return Err(RuleBuilderError::TypeMismatch {
                    expected: var_ty,
                    got: ty,
                });
            }
        }
        Ok(())
    }

    /// Register the given rule with the egraph.
    pub fn build(mut self) -> RuleId {
        if self.query.atoms.len() == 1 {
            self.query.plan_strategy = PlanStrategy::MinCover;
        }
        let res = self.query.rule_id;
        let info = RuleInfo {
            last_run_at: Timestamp::new(0),
            query: self.query,
            syntax: self.proof_builder.syntax,
            desc: self.proof_builder.rule_description,
        };
        debug!("created rule {res:?} / {}:\n{:?}", info.desc, info.syntax);
        self.egraph.rules.insert(res, info);
        res
    }

    pub(crate) fn set_focus(&mut self, focus: usize) {
        self.query.sole_focus = Some(focus);
    }

    /// Bind a new variable of the given type in the query.
    pub fn new_var(&mut self, ty: ColumnTy) -> Variable {
        let res = self.query.vars.next_id();
        self.query.vars.push(VarInfo { ty, term_var: res })
    }

    /// Bind a new variable of the given type in the query.
    ///
    /// This method attaches the given name to the [`QueryEntry`], which can
    /// make debugging easier.
    pub fn new_var_named(&mut self, ty: ColumnTy, name: &str) -> QueryEntry {
        let id = self.new_var(ty);
        QueryEntry::Var {
            id,
            name: Some(name.into()),
        }
    }

    pub(crate) fn add_atom_func(&mut self, func: FunctionId, entries: &[QueryEntry]) {
        let table_id = self.egraph.funcs[func].table;
        self.add_atom_with_timestamp_and_func(table_id, Some(func), entries);
    }

    pub(crate) fn add_atom_with_timestamp(&mut self, table: TableId, entries: &[QueryEntry]) {
        self.add_atom_with_timestamp_and_func(table, None, entries);
    }

    fn add_atom_with_timestamp_and_func(
        &mut self,
        table: TableId,
        func: Option<FunctionId>,
        entries: &[QueryEntry],
    ) {
        let mut atom = entries.to_vec();
        atom.push(self.new_var(ColumnTy::Id).into());
        if self.egraph.tracing {
            let proof_var = self.new_var(ColumnTy::Id);
            self.proof_builder.add_lhs(entries, proof_var);
            self.query.atom_proofs.push(proof_var);
            if let Some(func) = func {
                // If we have a function, record its syntax as a LHS term.
                let term = Arc::new(TermFragment::App(
                    func,
                    entries[0..entries.len() - 1]
                        .iter()
                        .map(|x| {
                            x.to_syntax(self.egraph)
                                .expect("should have all needed type information")
                        })
                        .collect(),
                ));
                let bound = entries.last().unwrap().var();
                self.proof_builder.syntax.lhs_bindings.push(Binding {
                    var: bound,
                    syntax: term,
                });
            }
            if let Some(QueryEntry::Var { id, .. }) = entries.last() {
                if table != self.egraph.uf_table {
                    // Don't overwrite "term_var" for uf_table; it stores
                    // reasons inline / doesn't have terms.
                    self.query.vars[*id].term_var = proof_var;
                }
            }
            atom.push(proof_var.into());
        }
        self.query.atoms.push((table, atom));
    }

    pub fn call_external_func(
        &mut self,
        func: ExternalFunctionId,
        args: &[QueryEntry],
        ret_ty: ColumnTy,
    ) -> Variable {
        let args = args.to_vec();
        let res = self.new_var(ret_ty);
        if self.egraph.tracing {
            self.proof_builder
                .register_prim(func, &args, res, ret_ty, self.egraph);
        }
        self.query.add_rule.push(Box::new(move |inner, rb| {
            let args = inner.convert_all(&args);
            let var = rb.call_external(func, &args)?;
            inner.mapping.insert(res, var.into());
            Ok(())
        }));
        res
    }

    /// Add the given table atom to query. As elsewhere in the crate, the last
    /// argument is the "return value" of the function.
    pub fn query_table(&mut self, func: FunctionId, entries: &[QueryEntry]) -> Result<()> {
        let schema = &self.egraph.funcs[func].schema;
        if schema.len() != entries.len() {
            return Err(anyhow::Error::from(RuleBuilderError::ArityMismatch {
                expected: schema.len(),
                got: entries.len(),
            }))
            .with_context(|| format!("query_table: mismatch between {entries:?} and {schema:?}"));
        }
        entries
            .iter()
            .zip(schema.iter())
            .try_for_each(|(entry, ty)| {
                self.assert_has_ty(entry, *ty)
                    .with_context(|| format!("query_table: mismatch between {entry:?} and {ty:?}"))
            })?;
        self.add_atom_func(func, entries);
        Ok(())
    }

    /// Add the given primitive atom to query. As elsewhere in the crate, the last
    /// argument is the "return value" of the function.
    pub fn query_prim(
        &mut self,
        func: ExternalFunctionId,
        entries: &[QueryEntry],
        ret_ty: ColumnTy,
    ) -> Result<()> {
        // Primitives on the LHS side of a rule turn into a call to a
        // primitive, along with an assertion that the result equals the
        // return value.
        let entries = entries.to_vec();
        let lhs_term = Arc::new(TermFragment::Prim(
            func,
            entries[..entries.len() - 1]
                .iter()
                .map(|x| {
                    x.to_syntax(self.egraph)
                        .expect("all variables should have type information")
                })
                .collect(),
            ret_ty,
        ));
        let rhs_term = entries.last().unwrap().to_syntax(self.egraph).unwrap();
        match rhs_term {
            Entry::Placeholder(var) => {
                self.proof_builder.syntax.rhs_bindings.push(Binding {
                    var,
                    syntax: lhs_term,
                });
            }
            Entry::Const(primitive_constant) => {
                let anon = self.new_var(ret_ty);
                self.proof_builder.syntax.rhs_bindings.push(Binding {
                    var: anon,
                    syntax: lhs_term,
                });
                self.proof_builder
                    .syntax
                    .statements
                    .push(Statement::AssertEq(
                        Entry::Placeholder(anon),
                        Entry::Const(primitive_constant),
                    ));
            }
        }
        self.query.add_rule.push(Box::new(move |inner, rb| {
            let mut dst_vars = inner.convert_all(&entries);
            let expected = dst_vars.pop().expect("must specify a return value");
            let var = rb.call_external(func, &dst_vars)?;
            match entries.last().unwrap() {
                QueryEntry::Var { id, .. } if !inner.grounded.contains(id) => {
                    inner.mapping.insert(*id, var.into());
                    inner.grounded.insert(*id);
                }
                _ => rb.assert_eq(var.into(), expected),
            }
            Ok(())
        }));
        Ok(())
    }

    /// Look up the value of a function in the database. If the value is not
    /// present, the configured default for the function is used.
    pub fn lookup(&mut self, func: FunctionId, entries: &[QueryEntry]) -> Variable {
        let entries = entries.to_vec();
        let val_col = ColumnId::from_usize(entries.len());
        let info = &self.egraph.funcs[func];
        let res = self.query.vars.push(VarInfo {
            ty: info.ret_ty(),
            term_var: self.query.vars.next_id(),
        });
        let table = info.table;
        let id_counter = self.query.id_counter;
        let cb: BuildRuleCallback = match info.default_val {
            DefaultVal::Const(_) | DefaultVal::FreshId => {
                let (wv, wv_ref): (WriteVal, WriteVal) = match &info.default_val {
                    DefaultVal::Const(c) => ((*c).into(), (*c).into()),
                    DefaultVal::FreshId => (
                        WriteVal::IncCounter(id_counter),
                        // When we create a new term, we should
                        // simply "reuse" the value we just minted
                        // for the value.
                        WriteVal::CurrentVal(val_col.index()),
                    ),
                    _ => unreachable!(),
                };
                if self.egraph.tracing {
                    let term_var = self.new_var(ColumnTy::Id);
                    self.query.vars[res].term_var = term_var;
                    let ts_var = self.new_var(ColumnTy::Id);
                    let reason_var = self.new_var(ColumnTy::Id);
                    let mut insert_entries = entries.to_vec();
                    insert_entries.push(res.into());
                    let add_proof = self.proof_builder.new_row(
                        func,
                        insert_entries,
                        term_var,
                        reason_var,
                        self.egraph,
                    );
                    Box::new(move |inner, rb| {
                        let dst_vars = inner.convert_all(&entries);
                        // NB: having one `lookup_or_insert` call
                        // per projection is pretty inefficient
                        // here, but merging these into a custom
                        // instruction didn't move the needle on a
                        // write-heavy benchmark when I tried it
                        // early on. May be worth revisiting after
                        // more low-hanging fruit has been
                        // optimized.
                        let var = rb.lookup_or_insert(
                            table,
                            &dst_vars,
                            &[wv, inner.next_ts.to_value().into(), wv_ref],
                            val_col,
                        )?;
                        let ts = rb.lookup_or_insert(
                            table,
                            &dst_vars,
                            &[wv, inner.next_ts.to_value().into(), wv_ref],
                            val_col.inc(),
                        )?;
                        let term = rb.lookup_or_insert(
                            table,
                            &dst_vars,
                            &[wv, inner.next_ts.to_value().into(), wv_ref],
                            val_col.inc().inc(),
                        )?;
                        inner.mapping.insert(term_var, term.into());
                        inner.mapping.insert(res, var.into());
                        inner.mapping.insert(ts_var, ts.into());
                        rb.assert_eq(var.into(), term.into());
                        // The following bookeeping is only needed
                        // if the value is new. That only happens if
                        // the main id equals the term id.
                        add_proof(inner, rb)?;
                        Ok(())
                    })
                } else {
                    Box::new(move |inner, rb| {
                        let dst_vars = inner.convert_all(&entries);
                        let var = rb.lookup_or_insert(
                            table,
                            &dst_vars,
                            &[wv, inner.next_ts.to_value().into()],
                            val_col,
                        )?;
                        inner.mapping.insert(res, var.into());
                        Ok(())
                    })
                }
            }
            DefaultVal::Fail => {
                if self.egraph.tracing {
                    let term_var = self.new_var(ColumnTy::Id);
                    self.proof_builder.add_lhs(&entries, term_var);
                    Box::new(move |inner, rb| {
                        let dst_vars = inner.convert_all(&entries);
                        let var = rb.lookup(table, &dst_vars, val_col)?;
                        let term = rb.lookup(table, &dst_vars, ColumnId::new(val_col.rep() + 1))?;
                        inner.mapping.insert(res, var.into());
                        inner.mapping.insert(term_var, term.into());
                        Ok(())
                    })
                } else {
                    Box::new(move |inner, rb| {
                        let dst_vars = inner.convert_all(&entries);
                        let var = rb.lookup(table, &dst_vars, val_col)?;
                        inner.mapping.insert(res, var.into());
                        Ok(())
                    })
                }
            }
        };
        self.query.add_rule.push(cb);
        res
    }

    /// Merge the two values in the union-find.
    pub fn union(&mut self, mut l: QueryEntry, mut r: QueryEntry) {
        let cb: BuildRuleCallback = if self.query.tracing {
            // We should really have the proof builder module handle this, but
            // we rewrite `l` and `r` below, but `syntax_env` reflects regular
            // vars not term-level vars, sadly.
            self.proof_builder.syntax.statements.push(Statement::Union(
                l.to_syntax(self.egraph).unwrap(),
                r.to_syntax(self.egraph).unwrap(),
            ));

            // Union proofs should reflect term-level variables rather than the
            // current leader of the e-class.
            for entry in [&mut l, &mut r] {
                if let QueryEntry::Var { id, .. } = entry {
                    *id = self.query.vars[*id].term_var;
                }
            }
            let reason_var = self.new_var(ColumnTy::Id);
            let add_proof = self
                .proof_builder
                .union(l.clone(), r.clone(), reason_var, self.egraph);
            Box::new(move |inner, rb| {
                let l = inner.convert(&l);
                let r = inner.convert(&r);
                add_proof(inner, rb)?;
                let proof = inner.mapping[reason_var];
                rb.insert(
                    inner.uf_table,
                    &[l, r, inner.next_ts.to_value().into(), proof],
                )
                .context("union")
            })
        } else {
            Box::new(move |inner, rb| {
                let l = inner.convert(&l);
                let r = inner.convert(&r);
                rb.insert(inner.uf_table, &[l, r, inner.next_ts.to_value().into()])
                    .context("union")
            })
        };
        self.query.add_rule.push(cb);
    }

    /// This method is equivalent to `remove(table, before); set(table, after)`
    /// when tracing/proofs aren't enabled. When proofs are enabled, it
    /// creates a proof term specialized for equality.
    ///
    /// This allows us to reconstruct proofs lazily from the UF, rather than
    /// running the proof generation algorithm eagerly as we query the table.
    /// Proof generation is a relatively expensive operation, and we'd prefer to
    /// avoid doing it on every union-find lookup.
    pub(crate) fn rebuild_row(
        &mut self,
        func: FunctionId,
        before: &[QueryEntry],
        after: &[QueryEntry],
    ) {
        assert_eq!(before.len(), after.len());
        self.remove(func, &before[..before.len() - 1]);
        if !self.egraph.tracing {
            self.set(func, after);
            return;
        }
        let table = self.egraph.funcs[func].table;
        let term_var = self.new_var(ColumnTy::Id);
        let reason_var = self.new_var(ColumnTy::Id);
        let before_id = before.last().unwrap().var();
        let before_term = self.query.vars[before_id].term_var;
        debug_assert_ne!(before_term, before_id);
        let add_proof = self.proof_builder.rebuild_proof(
            func,
            after,
            RebuildVars {
                before_term,
                new_term: term_var,
                reason: reason_var,
            },
            self.egraph,
        );
        let after = SmallVec::<[_; 4]>::from_iter(after.iter().cloned());
        let uf_table = self.query.uf_table;
        self.query.add_rule.push(Box::new(move |inner, rb| {
            add_proof(inner, rb)?;
            let mut dst_vars = inner.convert_all(&after);
            dst_vars.push(inner.next_ts.to_value().into());
            dst_vars.push(inner.mapping[term_var]);
            // This congruence rule will also serve as a proof that the old and
            // new terms are equal.
            rb.insert(
                uf_table,
                &[
                    inner.mapping[before_term],
                    inner.mapping[term_var],
                    inner.next_ts.to_value().into(),
                    inner.mapping[reason_var],
                ],
            )
            .context("rebuild_row_uf")?;
            rb.insert(table, &dst_vars).context("rebuild_row_table")
        }));
    }

    /// Set the value of a function in the database.
    pub fn set(&mut self, func: FunctionId, entries: &[QueryEntry]) {
        let table = self.egraph.funcs[func].table;
        let entries = entries.to_vec();
        if self.egraph.tracing {
            let res = self.lookup(func, &entries[0..entries.len() - 1]);
            self.union(res.into(), entries.last().unwrap().clone());
        } else {
            self.query.add_rule.push(Box::new(move |inner, rb| {
                let mut dst_vars = inner.convert_all(&entries);
                dst_vars.push(inner.next_ts.to_value().into());
                rb.insert(table, &dst_vars).context("set")
            }));
        };
    }

    /// Remove the value of a function from the database.
    pub fn remove(&mut self, table: FunctionId, entries: &[QueryEntry]) {
        let table = self.egraph.funcs[table].table;
        let entries = entries.to_vec();
        let cb: BuildRuleCallback = Box::new(move |inner, rb| {
            let dst_vars = inner.convert_all(&entries);
            rb.remove(table, &dst_vars).context("remove")
        });
        self.query.add_rule.push(cb);
    }

    /// Panic with a given message.
    pub fn panic(&mut self, message: String) {
        let panic = self.egraph.new_panic(message.clone());
        self.call_external_func(panic, &[], ColumnTy::Id);
    }
}

impl Query {
    fn query_state<'a, 'outer>(
        &self,
        rsb: &'a mut RuleSetBuilder<'outer>,
        next_ts: Timestamp,
    ) -> (QueryBuilder<'outer, 'a>, Bindings) {
        let mut qb = rsb.new_rule();
        qb.set_plan_strategy(self.plan_strategy);
        let mut inner = Bindings {
            uf_table: self.uf_table,
            next_ts,
            mapping: Default::default(),
            grounded: Default::default(),
        };
        for (var, _) in self.vars.iter() {
            inner.mapping.insert(var, DstVar::Var(qb.new_var()));
        }
        (qb, inner)
    }

    fn run_rules_and_build(&self, qb: QueryBuilder, mut inner: Bindings, desc: &str) -> Result<()> {
        let mut rb = qb.build();
        self.add_rule
            .iter()
            .try_for_each(|f| f(&mut inner, &mut rb))?;
        rb.build_with_description(desc);
        Ok(())
    }

    /// Translate the egglog query into a (set of) queries against the database.
    ///
    /// The timestamp values are used to guide seminaive evaluation. The query
    /// is taken to have run against the database values between start_ts and
    /// mid_ts (half-open), but the database (now) contains values up to
    /// next_ts.
    pub(crate) fn add_rules(
        &self,
        rsb: &mut RuleSetBuilder,
        mid_ts: Timestamp,
        next_ts: Timestamp,
        desc: &str,
    ) -> Result<()> {
        fn get_ts_col(atom: &[QueryEntry], tracing: bool) -> ColumnId {
            if tracing {
                ColumnId::from_usize(atom.len() - 2)
            } else {
                ColumnId::from_usize(atom.len() - 1)
            }
        }
        fn add_atom(
            qb: &mut QueryBuilder,
            table: TableId,
            entries: &[QueryEntry],
            constraints: &[Constraint],
            inner: &mut Bindings,
        ) -> Result<()> {
            for entry in entries {
                if let QueryEntry::Var { id, .. } = entry {
                    inner.grounded.insert(*id);
                }
            }
            let vars = inner.convert_all(entries);
            qb.add_atom(table, &vars, constraints)?;
            Ok(())
        }
        // For N atoms, we create N queries for seminaive evaluation.
        if !self.seminaive || (self.atoms.is_empty() && mid_ts == Timestamp::new(0)) {
            // If a rule has an empty LHS, we still want to run it once. This will cause the right
            // hand side of the rule to run once, globally across all runs.
            let (mut qb, mut inner) = self.query_state(rsb, next_ts);
            for (table, entries) in &self.atoms {
                add_atom(&mut qb, *table, entries, &[], &mut inner)?;
            }
            return self.run_rules_and_build(qb, inner, desc);
        }
        if let Some(focus_atom) = self.sole_focus {
            let (mut qb, mut inner) = self.query_state(rsb, next_ts);
            for (i, (table, entries)) in self.atoms.iter().enumerate() {
                let ts_col = get_ts_col(entries, self.tracing);
                let constraint = if i == focus_atom {
                    Some(Constraint::GeConst {
                        col: ts_col,
                        val: mid_ts.to_value(),
                    })
                } else {
                    None
                };
                if let Some(c) = constraint {
                    add_atom(&mut qb, *table, entries, &[c], &mut inner)
                } else {
                    add_atom(&mut qb, *table, entries, &[], &mut inner)
                }?;
            }
            return self.run_rules_and_build(
                qb,
                inner,
                &format!("{desc}-atom({focus_atom})[{mid_ts:?},{next_ts:?}]"),
            );
        }
        'outer: for focus_atom in 0..self.atoms.len() {
            let (mut qb, mut inner) = self.query_state(rsb, next_ts);
            for (i, (table, entries)) in self.atoms.iter().enumerate() {
                let ts_col = get_ts_col(entries, self.tracing);
                let constraint = match i.cmp(&focus_atom) {
                    Ordering::Less => {
                        if mid_ts == Timestamp::new(0) {
                            continue 'outer;
                        }
                        Some(Constraint::LtConst {
                            col: ts_col,
                            val: mid_ts.to_value(),
                        })
                    }
                    Ordering::Equal => Some(Constraint::GeConst {
                        col: ts_col,
                        val: mid_ts.to_value(),
                    }),
                    Ordering::Greater => None,
                };
                if let Some(c) = constraint {
                    add_atom(&mut qb, *table, entries, &[c], &mut inner)
                } else {
                    add_atom(&mut qb, *table, entries, &[], &mut inner)
                }?;
            }
            self.run_rules_and_build(
                qb,
                inner,
                &format!("{desc}-atom({focus_atom})[{mid_ts:?},{next_ts:?}]"),
            )?;
        }
        Ok(())
    }
}

/// State that is used during query execution to translate variabes in egglog
/// rules into variables for core-relations rules.
pub(crate) struct Bindings {
    uf_table: TableId,
    pub(crate) next_ts: Timestamp,
    pub(crate) mapping: DenseIdMap<Variable, DstVar>,
    grounded: HashSet<Variable>,
}

impl Bindings {
    pub(crate) fn convert(&self, entry: &QueryEntry) -> DstVar {
        match entry {
            QueryEntry::Var { id: v, .. } => self.mapping[*v],
            QueryEntry::Const { val, .. } => DstVar::Const(*val),
        }
    }
    pub(crate) fn convert_all(&self, entries: &[QueryEntry]) -> SmallVec<[DstVar; 4]> {
        entries.iter().map(|e| self.convert(e)).collect()
    }
}
