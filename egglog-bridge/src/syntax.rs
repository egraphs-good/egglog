//! Egglog proofs reference the source syntax of the query, but the syntax in `egglog-bridge` is a
//! lower-level, "desugared" representation of that syntax.
//!
//! This module defines the [`SourceSyntax`] and [`SourceExpr`] types, which allow callers to
//! reflect the syntax of the original egglog query, along with how it maps to the desugared query
//! language in this crate. The proofs machinery then reconstructs proofs according to this syntax.
use std::{iter, sync::Arc};

use crate::{EGraph, ProofReason, QueryEntry, ReasonSpecId, Result, SchemaMath, NOT_SUBSUMED};
use core_relations::{
    make_external_func, ColumnId, CounterId, ExecutionState, ExternalFunctionId, MergeVal,
    RuleBuilder, TableId, Value, WriteVal,
};
use numeric_id::{define_id, DenseIdMap, IdVec, NumericId};
use smallvec::SmallVec;

use crate::{
    proof_spec::ProofBuilder,
    rule::{AtomId, Bindings, Variable},
    ColumnTy, FunctionId, RuleId,
};

define_id!(pub SyntaxId, u32, "an offset into a Syntax DAG.");

#[derive(Debug, Clone)]
pub enum TopLevelLhsExpr {
    /// Simply requires the presence of a term matching the given [`SourceExpr`].
    Exists(SyntaxId),
    /// Asserts the equality of two expressions matching the given [`SourceExpr`]s.
    Eq(SyntaxId, SyntaxId),
}

/// Representative source syntax for _one line_ of an egglog query, namely, the left-hand-side of
/// an egglog rule.
#[derive(Debug, Clone)]
pub enum SourceExpr {
    /// A constant.
    Const { ty: ColumnTy, val: Value },
    /// A single variable.
    Var {
        id: Variable,
        ty: ColumnTy,
        name: String,
    },
    /// A call to an external (aka primitive) function.
    ExternalCall {
        /// This external function call must be present in the destination query, and bound to this
        /// variable
        var: Variable,
        ty: ColumnTy,
        func: ExternalFunctionId,
        args: Vec<SyntaxId>,
    },
    /// A query of an egglog-level function (i.e. a table).
    FunctionCall {
        /// The egglog function being bound.
        func: FunctionId,
        /// The atom in the _destination_ query (i.e. at the egglog-bridge level) to which this
        /// call corresponds.
        atom: AtomId,
        /// Arguments to the function.
        args: Vec<SyntaxId>,
    },
}

/// A data-structure representing an egglog query. Essentially, multiple [`SourceExpr`]s, one per
/// line, along with a backing store accounting for subterms indexed by [`SyntaxId].
#[derive(Debug, Clone, Default)]
pub struct SourceSyntax {
    pub(crate) backing: IdVec<SyntaxId, SourceExpr>,
    pub(crate) vars: Vec<(Variable, ColumnTy)>,
    pub(crate) roots: Vec<TopLevelLhsExpr>,
}

impl SourceSyntax {
    /// Add `expr` to the known syntax of the [`SourceSyntax`].
    ///
    /// The returned [`SyntaxId`] can be used to construct another [`SourceExpr`] or a
    /// [`TopLevelLhsExpr`].
    pub fn add_expr(&mut self, expr: SourceExpr) -> SyntaxId {
        match &expr {
            SourceExpr::Const { .. } | SourceExpr::FunctionCall { .. } => {}
            SourceExpr::Var { id, ty, .. } => self.vars.push((*id, *ty)),
            SourceExpr::ExternalCall { var, ty, .. } => self.vars.push((*var, *ty)),
        };
        self.backing.push(expr)
    }

    /// Add `expr` to the toplevel representation of the syntax.
    pub fn add_toplevel_expr(&mut self, expr: TopLevelLhsExpr) {
        self.roots.push(expr);
    }

    fn funcs(&self) -> impl Iterator<Item = FunctionId> + '_ {
        self.backing.iter().filter_map(|(_, v)| {
            if let SourceExpr::FunctionCall { func, .. } = v {
                Some(*func)
            } else {
                None
            }
        })
    }
}

/// The data associated with a proof of a given term whose premises are given by a
/// [`SourceSyntax`].
#[derive(Debug)]
pub(crate) struct RuleData {
    pub(crate) rule_id: RuleId,
    pub(crate) syntax: SourceSyntax,
}

impl RuleData {
    pub(crate) fn n_vars(&self) -> usize {
        self.syntax.vars.len()
    }
}

impl ProofBuilder {
    /// Given a [`SourceSyntax`] build a callback that returns a variable corresponding to the id
    /// of the "reason" for a given rule. This callback does two things, both based on the context
    /// of the syntax being passed in:
    ///
    /// 1. It reconstructs any terms specified by the syntax. This is done by applying congruence
    ///    rules to the `AtomId`s mapped in the syntax.
    ///
    /// 2. It writes a reason holding the concrete substitution corersponding to the current match
    ///    for this syntax.
    ///
    /// Like most of the rest of this crate, the return value is a callback that consumes state
    /// associated with instantiating a rule in the `core-relations` sense.
    pub(crate) fn create_reason(
        &mut self,
        syntax: SourceSyntax,
        egraph: &mut EGraph,
    ) -> impl Fn(&mut Bindings, &mut RuleBuilder) -> Result<core_relations::Variable> + Clone {
        // first, create all the relevant cong metadata
        let mut metadata = DenseIdMap::default();
        for func in syntax.funcs() {
            metadata.insert(func, self.build_cong_metadata(func, egraph));
        }

        let reason_spec = Arc::new(ProofReason::Rule(RuleData {
            rule_id: self.rule_id,
            syntax: syntax.clone(),
        }));
        let reason_table = egraph.reason_table(&reason_spec);
        let reason_id = egraph.proof_specs.push(reason_spec);
        let reason_counter = egraph.reason_counter;
        let atom_mapping = self.term_vars.clone();
        move |bndgs, rb| {
            // Now, insert all needed reconstructed terms.
            let mut state = TermReconstructionState {
                syntax: &syntax,
                syntax_mapping: Default::default(),
                metadata: metadata.clone(),
                atom_mapping: atom_mapping.clone(),
            };
            for toplevel_expr in &syntax.roots {
                match toplevel_expr {
                    TopLevelLhsExpr::Exists(id) => {
                        state.justify_query(*id, bndgs, rb)?;
                    }
                    TopLevelLhsExpr::Eq(id1, id2) => {
                        state.justify_query(*id1, bndgs, rb)?;
                        state.justify_query(*id2, bndgs, rb)?;
                    }
                }
            }
            // Once those terms are all guaranteed to be in the e-graph, we only need to write down
            // the base substitution of variables into a reason table.
            let mut row = SmallVec::<[core_relations::QueryEntry; 8]>::new();
            row.push(Value::new(reason_id.rep()).into());
            for (var, _) in &syntax.vars {
                row.push(bndgs.mapping[*var]);
            }
            Ok(rb.lookup_or_insert(
                reason_table,
                &row,
                &[WriteVal::IncCounter(reason_counter)],
                ColumnId::from_usize(row.len()),
            )?)
        }
    }

    fn build_cong_metadata(&self, func: FunctionId, egraph: &mut EGraph) -> FunctionCongMetadata {
        let func_info = &egraph.funcs[func];
        let func_underlying = func_info.table;
        let schema_math = SchemaMath {
            subsume: func_info.can_subsume,
            tracing: true,
            func_cols: func_info.schema.len(),
        };
        let cong_args = CongArgs {
            func_table: func,
            func_underlying,
            schema_math,
            reason_table: egraph.reason_table(&ProofReason::CongRow),
            term_table: egraph.term_table(func_underlying),
            reason_counter: egraph.reason_counter,
            term_counter: egraph.id_counter,
            ts_counter: egraph.timestamp_counter,
            reason_spec_id: egraph.cong_spec,
        };
        let build_term = egraph.register_external_func(make_external_func(move |es, vals| {
            cong_term(&cong_args, es, vals)
        }));
        FunctionCongMetadata {
            table: func_underlying,
            build_term,
            schema_math,
        }
    }
}

/// Metadata needed to reconstruct a term whose head corresponds to a particular function.
#[derive(Copy, Clone)]
struct FunctionCongMetadata {
    table: TableId,
    build_term: ExternalFunctionId,
    schema_math: SchemaMath,
}

struct TermReconstructionState<'a> {
    /// The syntax of the LHS of a rule that we are reconstructing.
    ///
    /// This is an immutable reference to make it easy to borrow across recursive calls.
    syntax: &'a SourceSyntax,
    /// A memo cache from syntax node to the [`core_relations::QueryEntry`] that it corresponds to
    /// in the reconstructed term.
    syntax_mapping: DenseIdMap<SyntaxId, core_relations::QueryEntry>,
    /// The [`QueryEntry`] (in `egglog-bridge`, not `core-relations`) to which the given atom
    /// corresponds.
    atom_mapping: DenseIdMap<AtomId, QueryEntry>,
    metadata: DenseIdMap<FunctionId, FunctionCongMetadata>,
}

impl TermReconstructionState<'_> {
    fn justify_query(
        &mut self,
        node: SyntaxId,
        bndgs: &mut Bindings,
        rb: &mut RuleBuilder,
    ) -> Result<core_relations::QueryEntry> {
        if let Some(entry) = self.syntax_mapping.get(node) {
            return Ok(*entry);
        }
        let syntax = self.syntax;
        let res = match &syntax.backing[node] {
            SourceExpr::Const { val, .. } => return Ok(core_relations::QueryEntry::Const(*val)),
            SourceExpr::Var { id, .. } => bndgs.mapping[*id],
            SourceExpr::ExternalCall { var, args, .. } => {
                for arg in args {
                    self.justify_query(*arg, bndgs, rb)?;
                }
                bndgs.mapping[*var]
            }
            SourceExpr::FunctionCall { func, atom, args } => {
                let old_term = bndgs.convert(&self.atom_mapping[*atom]);
                let mut buf: Vec<core_relations::QueryEntry> = vec![old_term];

                for arg in args.iter().map(|s| self.justify_query(*s, bndgs, rb)) {
                    buf.push(arg?);
                }
                let FunctionCongMetadata {
                    table,
                    build_term,
                    schema_math,
                } = &self.metadata[*func];
                let term_col = ColumnId::from_usize(schema_math.proof_id_col());
                rb.lookup_with_fallback(*table, &buf[1..], term_col, *build_term, &buf)?
                    .into()
            }
        };
        self.syntax_mapping.insert(node, res);
        Ok(res)
    }
}

/// Metadata from the EGraph that we copy into an [`core_relations::ExternalFunction`] closure that
/// recreates terms justified by congruence.
#[derive(Clone)]
struct CongArgs {
    /// The function that we are applying congruence to.
    func_table: FunctionId,
    /// The undcerlying `core_relations` table that this function corresponds to.
    func_underlying: TableId,
    /// Schema-related offset information needed for writing to the table.
    schema_math: SchemaMath,
    /// The table that will hold the reason justifying the new term, if we need to insert one.
    reason_table: TableId,
    /// The table that will hold the new term, if we need to insert one.
    term_table: TableId,
    /// The counter that will be incremented when we insert a new reason.
    reason_counter: CounterId,
    /// The counter that will be incremented when we insert a new term.
    term_counter: CounterId,
    /// The counter that will be used to read the current timestamp for the new row.
    ts_counter: CounterId,
    /// The specification (or schema) for the reason we are writing (congruence, in this case).
    reason_spec_id: ReasonSpecId,
}

fn cong_term(args: &CongArgs, es: &mut ExecutionState, vals: &[Value]) -> Option<Value> {
    let old_term = vals[0];
    let new_term = &vals[1..];
    let reason = es.predict_col(
        args.reason_table,
        &[Value::new(args.reason_spec_id.rep()), old_term],
        iter::once(MergeVal::Counter(args.reason_counter)),
        ColumnId::new(2),
    );
    let mut term_row = SmallVec::<[Value; 8]>::default();
    term_row.push(Value::new(args.func_table.rep()));
    term_row.extend_from_slice(new_term);
    let term_val = es.predict_col(
        args.term_table,
        &term_row,
        [
            MergeVal::Counter(args.term_counter),
            MergeVal::Constant(reason),
        ]
        .into_iter(),
        ColumnId::from_usize(term_row.len()),
    );

    // We should be able to do a raw insert at this point. All conflicting inserts will have the
    // same term value, and this function only gets called when a lookup fails.

    let ts = Value::from_usize(es.read_counter(args.ts_counter));
    term_row.resize(args.schema_math.table_columns(), NOT_SUBSUMED);
    term_row[args.schema_math.ret_val_col()] = term_val;
    term_row[args.schema_math.proof_id_col()] = term_val;
    term_row[args.schema_math.ts_col()] = ts;
    es.stage_insert(args.func_underlying, &term_row);
    Some(term_val)
}
