//! `impl egglog_backend_trait::RuleBuilderOps for DuckRuleBuilderOps` —
//! Phase 2 Commit 10.
//!
//! This module bridges the trait-level rule-building API
//! (`RuleBuilderOps`, defined in `egglog-backend-trait`) to the existing
//! DuckDB-side data IR (`crate::Rule`, `crate::Atom`, `crate::Action`,
//! `crate::Term`) and the existing `compile_rule` pipeline. There are
//! **zero changes** to that internal IR — this is strictly a translation
//! layer.
//!
//! ## How it works
//!
//! [`DuckRuleBuilderOps`] holds:
//! - A mutable reference to the duckdb-backend [`crate::EGraph`] for the
//!   builder's lifetime.
//! - An in-progress [`crate::Rule`] (`name`, `ruleset = ""`, `body`,
//!   `actions`) that each `RuleBuilderOps` method appends to.
//! - A monotonically-increasing counter for fresh variable ids
//!   (separate from `Variable::id` so that variables synthesized inside
//!   the builder, e.g. for `lookup`'s output, never collide with
//!   caller-provided ones).
//!
//! On [`RuleBuilderOps::build`], the accumulated rule is handed to the
//! existing `EGraph::add_rule`. A fresh `RuleId` is allocated as the
//! index into `EGraph::backend_rule_names`.
//!
//! ## What is supported
//!
//! Variable bindings and constants (only for `Value`s that can be
//! decoded inline; oversize integers and string base values require the
//! `BaseValuePool` wiring from Commit 11). Function-table body atoms.
//! Function lookups (`Action::LetExpr` with `Term::FuncCall`). Inserts,
//! deletes, panics.
//!
//! ## What returns errors
//!
//! - `subsume`: DuckDB does not support subsumption in v1 (see
//!   `Backend::supports_subsumption`).
//! - `query_table` with `is_subsumed = Some(_)`: same.
//! - `query_prim`: not yet wired through the trait (requires Commit 12's
//!   primitive registry). Returns an error until then.
//! - `call_external_func`: same — requires Commit 12. The trait
//!   signature is infallible, so the error is deferred to `build()`.
//! - `base_value_constant` (decoding from `Value` to a duck `Literal`):
//!   requires the `BaseValuePool` wiring from Commit 11. Until then,
//!   `QueryEntry::Const` with `ColumnTy::Base(_)` triggers a deferred
//!   error. Pure-variable rules work today.
//! - `union`: direct union actions are not supported in v1; programs
//!   that need union go through encoded `(set (pname ...) ())` rules
//!   emitted by term encoding.
//!
//! ## What we don't change in DuckDB's IR
//!
//! All translation lives in this file. The accumulator emits the
//! exact `Atom`/`Action`/`Term` shapes the existing `compile_rule`
//! pipeline already understands. If a method needs to reach into the
//! egraph (e.g. to map a `FunctionId` to a duckdb-side table name), it
//! consults `self.egraph.backend_function_names` directly at builder
//! time — no new state is introduced.

use anyhow::{Result, anyhow};
use egglog_backend_trait::{
    BaseValueId, BaseValuePool, ColumnTy, ExternalFunctionId, FunctionId, QueryEntry,
    RuleBuilderOps, RuleId, Value, Variable, VariableId,
};
use egglog_numeric_id::NumericId;

use crate::base_values::DuckdbBaseValuePool;
use crate::{Action, Atom, EGraph, Literal, Rule, Term};

/// Decode a `Value` of base type `ty` into a duck-side [`Literal`] by
/// consulting the [`BaseValuePool`]. Supported types: `i64`, `bool`,
/// `f64`, `String`, `()`, `BigInt`, `BigRat`. Unsupported types yield
/// an error.
fn decode_base_const(
    pool: &DuckdbBaseValuePool,
    val: Value,
    ty: BaseValueId,
) -> Result<Term> {
    use ordered_float::OrderedFloat;
    use std::any::TypeId;
    let pool_dyn: &dyn BaseValuePool = pool;
    // Try common types. Order chosen for frequency in test programs.
    if pool_dyn.has_ty(TypeId::of::<i64>())
        && ty == pool_dyn.get_ty_by_type_id(TypeId::of::<i64>())
    {
        let v = egglog_backend_trait::pool_unwrap::<i64>(pool_dyn, val);
        return Ok(Term::Lit(Literal::I64(v)));
    }
    if pool_dyn.has_ty(TypeId::of::<bool>())
        && ty == pool_dyn.get_ty_by_type_id(TypeId::of::<bool>())
    {
        let v = egglog_backend_trait::pool_unwrap::<bool>(pool_dyn, val);
        return Ok(Term::Lit(Literal::Bool(v)));
    }
    if pool_dyn.has_ty(TypeId::of::<()>())
        && ty == pool_dyn.get_ty_by_type_id(TypeId::of::<()>())
    {
        // Unit values: emit a sentinel i64(0) per the duck IR's
        // existing handling of unit constants.
        return Ok(Term::Lit(Literal::I64(0)));
    }
    type FBoxed = egglog_core_relations::Boxed<OrderedFloat<f64>>;
    if pool_dyn.has_ty(TypeId::of::<FBoxed>())
        && ty == pool_dyn.get_ty_by_type_id(TypeId::of::<FBoxed>())
    {
        let v = egglog_backend_trait::pool_unwrap::<FBoxed>(pool_dyn, val);
        return Ok(Term::Lit(Literal::F64((*v).into_inner())));
    }
    if pool_dyn.has_ty(TypeId::of::<egglog_core_relations::Boxed<String>>())
        && ty == pool_dyn
            .get_ty_by_type_id(TypeId::of::<egglog_core_relations::Boxed<String>>())
    {
        let v = egglog_backend_trait::pool_unwrap::<egglog_core_relations::Boxed<String>>(
            pool_dyn, val,
        );
        return Ok(Term::Lit(Literal::Str((*v).clone())));
    }
    // BigInt/BigRat etc. fall through to an error path the caller
    // can route into deferred_err.
    Err(anyhow!(
        "DuckRuleBuilderOps: decoding a base-value constant of BaseValueId({}) into a \
         duck Literal is not yet supported (i64/bool/Unit/f64/String are wired)",
        ty.rep()
    ))
}

// ---------------------------------------------------------------------------
// DuckRuleBuilderOps
// ---------------------------------------------------------------------------

/// In-progress rule built through the trait. Accumulates each
/// `RuleBuilderOps` call into a fresh [`crate::Rule`], then commits to
/// the egraph on [`RuleBuilderOps::build`].
pub(crate) struct DuckRuleBuilderOps<'a> {
    egraph: &'a mut EGraph,
    rule: Rule,
    /// Counter for synthetic variables (allocated by `new_var` /
    /// `new_var_named` / `lookup` / `call_external_func`). Starts at a
    /// value above any plausible caller-provided `VariableId` to avoid
    /// collisions; in practice the caller mints variables through us
    /// anyway via `new_var`, so the only risk is when `query_table` is
    /// passed a `Variable` minted on the bridge side. We sidestep this
    /// risk by deriving the duck::Term::Var name from the
    /// `VariableId::rep()` directly — see [`Self::var_name`].
    fresh_counter: u32,
    /// Errors collected during the builder. The trait's
    /// `query_table` / `query_prim` / `subsume` return `Result<()>`,
    /// but `set` / `remove` / `union` / `panic` are infallible. To
    /// keep DuckDB's "unsupported on duckdb" errors visible at
    /// `build()` time, we accumulate any deferred error here and
    /// surface it from `build`.
    deferred_err: Option<anyhow::Error>,
    /// Duck-side variable names that have been bound (by a body atom's
    /// output, an `Atom::Bind`, or an `Action::LetExpr` / `LetCtor`).
    /// Used by `query_prim` to decide between a `Bind` (fresh var) and
    /// a `Filter` (assert-eq) translation, mirroring the bridge's
    /// `inner.grounded` semantics.
    bound_vars: std::collections::HashSet<String>,
    /// Inline expansions for variables that were minted by `lookup` /
    /// `call_external_func` but whose result we never bind through an
    /// `Action::LetExpr`. Reason: DuckDB's `compile.rs` materializes
    /// rule actions into a single SELECT, and SQL forbids referencing
    /// sibling SELECT-list aliases (so `SELECT f(t.c0) AS v, g(v) AS w
    /// FROM t` would fail). Instead we *inline* the lookup's
    /// `Term::FuncCall { … }` directly into the args of any later
    /// action that references it. Each entry maps `VariableId.rep()`
    /// to the `Term` to substitute. When `entries_to_terms` translates
    /// a `QueryEntry::Var(v)` whose id is in this map, it returns the
    /// stored term instead of `Term::Var(name)`. Recursion handles
    /// chains: a stored `Term::FuncCall` may itself contain
    /// `Term::Var(other)` references, which the substitution walks
    /// transitively.
    inline_terms: std::collections::HashMap<u32, Term>,
    /// Tracks which `inline_terms` entries have been substituted into
    /// some other atom or action. Set in `entry_to_term` whenever a
    /// `QueryEntry::Var(v)` resolves through the inline map. At
    /// `build()` time, entries NOT in this set correspond to
    /// primitives whose results were never consumed — typically
    /// body-level filter primitives like `(!= a b)` whose fresh
    /// result var is unused. Those get emitted as body `Filter` atoms
    /// so the predicate actually constrains the rule.
    consumed_inline: std::cell::RefCell<std::collections::HashSet<u32>>,
}

impl<'a> DuckRuleBuilderOps<'a> {
    pub(crate) fn new(egraph: &'a mut EGraph, name: &str, _seminaive: bool) -> Self {
        // `seminaive` is honored implicitly: DuckDB's `add_rule`
        // compiles every rule into a seminaive variant per body atom,
        // so the bridge-side flag has no DuckDB analog. We accept and
        // ignore it.
        let rule = Rule {
            name: name.to_string(),
            ruleset: String::new(),
            body: Vec::new(),
            actions: Vec::new(),
        };
        Self {
            egraph,
            rule,
            fresh_counter: 1_000_000_000, // above any expected VariableId
            deferred_err: None,
            bound_vars: std::collections::HashSet::new(),
            inline_terms: std::collections::HashMap::new(),
            consumed_inline: std::cell::RefCell::new(std::collections::HashSet::new()),
        }
    }

    /// Recursively expand any `Term::Var` references through
    /// `inline_terms`. Used by `entries_to_terms` so that a lookup
    /// whose result is then used as an arg to another action gets
    /// inlined as a `Term::FuncCall` rather than left as a dangling
    /// `Term::Var`. Bounded by the number of stored inlines (each
    /// lookup adds one); cycles cannot occur because we only insert
    /// into `inline_terms` with fresh ids.
    fn inline_term(&self, t: Term) -> Term {
        match t {
            Term::Var(_) => t, // names; inlining is keyed on Variable ids in entry_to_term
            Term::Lit(_) => t,
            Term::Prim(op, args) => {
                Term::Prim(op, args.into_iter().map(|a| self.inline_term(a)).collect())
            }
            Term::FuncCall { name, args } => Term::FuncCall {
                name,
                args: args.into_iter().map(|a| self.inline_term(a)).collect(),
            },
        }
    }

    /// Derive a stable, unique duckdb-side variable name from a
    /// trait-side [`Variable`]. Variables minted by `new_var` /
    /// `new_var_named` / `lookup` / `call_external_func` already have
    /// unique ids; variables that the caller minted upstream and is
    /// passing back to us also carry a unique id from their backend.
    /// So `v{id}` is enough.
    fn var_name(v: &Variable) -> String {
        match &v.name {
            Some(n) => format!("v{}_{}", v.id.rep(), sanitize(n)),
            None => format!("v{}", v.id.rep()),
        }
    }

    /// Translate a trait-level [`QueryEntry`] to a duck-level [`Term`].
    ///
    /// `QueryEntry::Var` → `Term::Var(name)`.
    /// `QueryEntry::Const` with `ColumnTy::Id` → `Term::Lit(I64(...))`
    /// using the value's `u32` representation cast to `i64`.
    /// `QueryEntry::Const` with `ColumnTy::Base(_)` → decode the
    /// concrete typed primitive via the base value pool and emit the
    /// corresponding duck `Literal`. Supported types: `i64`, `bool`,
    /// `f64`, `String`, `()`. Unsupported types yield an error.
    fn entry_to_term(&self, entry: &QueryEntry) -> Result<Term> {
        match entry {
            QueryEntry::Var(v) => {
                // If this variable was minted by a `lookup` /
                // `call_external_func` whose result we stored inline,
                // substitute its `Term::FuncCall` here. Mark it
                // consumed so `build()` knows this entry was used
                // (and shouldn't be promoted to a Filter atom).
                // Otherwise emit a plain `Term::Var`.
                if let Some(inline) = self.inline_terms.get(&v.id.rep()) {
                    self.consumed_inline.borrow_mut().insert(v.id.rep());
                    Ok(self.inline_term(inline.clone()))
                } else {
                    Ok(Term::Var(Self::var_name(v)))
                }
            }
            QueryEntry::Const { val, ty } => match ty {
                ColumnTy::Id => Ok(Term::Lit(Literal::I64(val.rep() as i64))),
                ColumnTy::Base(bv) => decode_base_const(&self.egraph.backend_base_value_pool, *val, *bv),
            },
        }
    }

    fn entries_to_terms(&self, entries: &[QueryEntry]) -> Result<Vec<Term>> {
        entries.iter().map(|e| self.entry_to_term(e)).collect()
    }

    /// Allocate a fresh trait-level [`Variable`] with `name`.
    fn alloc_var(&mut self, name: Option<&str>) -> Variable {
        let id = self.fresh_counter;
        self.fresh_counter = self
            .fresh_counter
            .checked_add(1)
            .expect("DuckRuleBuilderOps: variable counter overflow");
        Variable {
            id: VariableId::new(id),
            name: name.map(|s| s.into()),
        }
    }

    fn lookup_function_name(&self, func: FunctionId) -> &str {
        let idx = func.rep() as usize;
        self.egraph
            .backend_function_names
            .get(idx)
            .map(|s| s.as_str())
            .unwrap_or_else(|| {
                panic!(
                    "DuckRuleBuilderOps: FunctionId({idx}) not registered via Backend::add_table"
                )
            })
    }

    /// `true` iff `name` is registered as a relation in the duck
    /// `EGraph` (no output column on the SQL side). The bridge's
    /// `RuleBuilder` API is relation-unaware — every table atom
    /// includes a trailing "output" entry — so the duck adapter
    /// strips that entry when emitting `Atom::Func` /
    /// `Action::Insert` / `Action::Delete` against a relation.
    fn is_relation(&self, name: &str) -> bool {
        self.egraph
            .functions
            .get(name)
            .map(|info| !info.has_output() && !info.eq_sort_ctor)
            .unwrap_or(false)
    }

    /// `true` iff `name` is registered as an EqSort constructor. The
    /// bridge's `RuleBuilder::lookup` semantics is "look up; allocate
    /// fresh on miss if `DefaultVal::FreshId`". For eq-sort
    /// constructors that means allocate-on-miss. In the duck IR this
    /// is exactly what `Action::LetCtor` (with its hash-cons
    /// LEFT JOIN + `COALESCE(... nextval(seq))`) does. So when
    /// `lookup` targets an eq-sort constructor we emit a `LetCtor`
    /// rather than inlining a `Term::FuncCall` (which would compile
    /// to a pure SELECT, returning NULL on miss — wrong semantics).
    fn is_eq_sort_ctor(&self, name: &str) -> bool {
        self.egraph
            .functions
            .get(name)
            .map(|info| info.eq_sort_ctor)
            .unwrap_or(false)
    }
}

/// Replace any non-alphanumeric character in a variable name with `_`.
/// Variable display names can contain `@`/`$`/`-`/etc. which would
/// break duck::Term::Var → SQL identifier substitution.
fn sanitize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    out
}

// ---------------------------------------------------------------------------
// `impl RuleBuilderOps`
// ---------------------------------------------------------------------------

impl<'a> RuleBuilderOps for DuckRuleBuilderOps<'a> {
    fn new_var(&mut self, _ty: ColumnTy) -> QueryEntry {
        // ColumnTy is not stored on the duck side — variables carry
        // their type via context (body atom signatures).
        let var = self.alloc_var(None);
        QueryEntry::Var(var)
    }

    fn new_var_named(&mut self, _ty: ColumnTy, name: &str) -> QueryEntry {
        let var = self.alloc_var(Some(name));
        QueryEntry::Var(var)
    }

    fn query_table(
        &mut self,
        func: FunctionId,
        entries: &[QueryEntry],
        is_subsumed: Option<bool>,
    ) -> Result<()> {
        // DuckDB does not model subsumption: every row is implicitly
        // non-subsumed. `Some(false)` (filter to non-subsumed rows)
        // therefore needs no filter — the SELECT already covers it.
        // `Some(true)` (filter to subsumed rows) cannot be satisfied
        // by any row and is rejected as an unsupported feature.
        if is_subsumed == Some(true) {
            return Err(anyhow!(
                "DuckRuleBuilderOps::query_table: filtering to subsumed rows is not supported \
                 on the DuckDB backend (Backend::supports_subsumption returns false)"
            ));
        }
        let name = self.lookup_function_name(func).to_string();
        let mut args = self.entries_to_terms(entries)?;
        // Bug 2 fix: for relation atoms the bridge's RuleBuilder
        // still adds a trailing wildcard "output" arg (the API is
        // relation-unaware). Strip it so the duck-side
        // `Atom::Func` matches the relation's actual column count.
        if self.is_relation(&name) && !args.is_empty() {
            args.pop();
        }
        // Body Func atoms bind every variable used as an argument.
        for arg in &args {
            if let Term::Var(v) = arg {
                self.bound_vars.insert(v.clone());
            }
        }
        self.rule.body.push(Atom::Func { name, args });
        Ok(())
    }

    fn query_prim(
        &mut self,
        func: ExternalFunctionId,
        entries: &[QueryEntry],
        _ret_ty: ColumnTy,
    ) -> Result<()> {
        // The bridge's `query_prim` semantics: the last entry is the
        // expected return value. If it's a fresh variable, bind it to
        // the call's result; otherwise, assert the call's result equals
        // the entry. We translate accordingly into duck IR.
        let name = self
            .egraph
            .external_func_name(func)
            .ok_or_else(|| {
                anyhow!(
                    "DuckRuleBuilderOps::query_prim: ExternalFunctionId({}) has no associated \
                     primitive name (was set_external_func_name called?)",
                    func.rep()
                )
            })?
            .to_string();
        let translated = entries
            .iter()
            .map(|e| self.entry_to_term(e))
            .collect::<Result<Vec<_>>>()?;
        // Split into args and expected return value.
        let mut translated = translated;
        let expected = translated.pop().ok_or_else(|| {
            anyhow!("DuckRuleBuilderOps::query_prim: must specify a return value")
        })?;
        let call = Term::prim(name.clone(), translated);
        // Bug 5 fix: ALL `query_prim` results go through
        // `inline_terms` so a downstream atom or action that uses
        // the result gets the call inlined into its args. This is
        // necessary because:
        //   (a) `compile.rs` materializes a rule's actions into a
        //       single SELECT, so chained `Atom::Bind { … }` body
        //       atoms cannot reference each other's vars in SQL
        //       (sibling SELECT-list aliases aren't visible);
        //   (b) the bridge's model uniformly threads every prim
        //       through `query_prim`'s "fresh return var" idiom,
        //       so a `(guard (or (bool-!= a b)))` rule body comes
        //       in as three separate `query_prim` calls — each
        //       returning a fresh var that the next call's args
        //       reference. Inlining flattens them into one nested
        //       `Term::Prim` chain that compile.rs handles as a
        //       single expression.
        //
        // For the `guard` primitive specifically — which is the
        // user-visible "this expression must be true" marker —
        // additionally push a `Filter(Prim("guard", [operand]))`
        // atom so the rule actually has a body-level WHERE clause.
        // Without this the rule would fire unconditionally.
        match (&expected, entries.last().unwrap()) {
            (Term::Var(varname), QueryEntry::Var(v))
                if !self.bound_vars.contains(varname) =>
            {
                // Fresh ungrounded var: bind it inline.
                let _ = v;
                self.inline_terms.insert(v.id.rep(), call.clone());
            }
            _ => {
                // The expected return is already bound or is a
                // literal: emit an equality filter `call = expected`.
                self.rule.body.push(Atom::Filter(Term::prim(
                    "=",
                    vec![call.clone(), expected],
                )));
            }
        }
        // Standalone `guard` calls turn into a body-level Filter.
        // The encoder emits these to mark a Boolean predicate
        // assertion; compile.rs translates `Filter(Prim("guard",
        // [x]))` to `WHERE (x)` directly.
        if name == "guard" {
            // The operand is the (now-inlined) first arg of the
            // call; rebuild a Filter around it.
            if let Term::Prim(_, args) = &call
                && let Some(arg) = args.first()
            {
                self.rule.body.push(Atom::Filter(Term::prim(
                    "guard",
                    vec![arg.clone()],
                )));
            }
        }
        Ok(())
    }

    fn call_external_func(
        &mut self,
        func: ExternalFunctionId,
        args: &[QueryEntry],
        _ret_ty: ColumnTy,
        _panic_msg: String,
    ) -> QueryEntry {
        let name = match self.egraph.external_func_name(func) {
            Some(n) => n.to_string(),
            None => {
                if self.deferred_err.is_none() {
                    self.deferred_err = Some(anyhow!(
                        "DuckRuleBuilderOps::call_external_func: ExternalFunctionId({}) has \
                         no associated primitive name (was set_external_func_name called?)",
                        func.rep()
                    ));
                }
                let var = self.alloc_var(Some("__ext_no_name"));
                return QueryEntry::Var(var);
            }
        };
        let translated_args: Vec<Term> = match args
            .iter()
            .map(|e| self.entry_to_term(e))
            .collect::<Result<_>>()
        {
            Ok(v) => v,
            Err(e) => {
                if self.deferred_err.is_none() {
                    self.deferred_err = Some(e);
                }
                Vec::new()
            }
        };
        let result = self.alloc_var(Some(&format!("call_{}", sanitize(&name))));
        // Inline the primitive call into `inline_terms` for the same
        // reason `lookup` does — see `lookup`'s comment. Subsequent
        // actions get the call substituted into their args directly,
        // sidestepping the sibling-SELECT-list-alias restriction.
        self.inline_terms
            .insert(result.id.rep(), Term::prim(name, translated_args));
        QueryEntry::Var(result)
    }

    fn lookup(
        &mut self,
        func: FunctionId,
        entries: &[QueryEntry],
        _panic_msg: String,
    ) -> QueryEntry {
        // The bridge's `lookup` is "look up the function's output for
        // these inputs; if missing, use the function's `DefaultVal`".
        // Two cases on the duck side:
        //
        // 1. Eq-sort constructor (`DefaultVal::FreshId`). Looking up
        //    should **allocate-on-miss**, returning the constructor's
        //    fresh id. The duck IR represents this via
        //    `Action::LetCtor`, whose hash-cons LEFT JOIN +
        //    `COALESCE(... nextval('__egglog_eqsort_seq'))` lowering
        //    in `compile.rs` does exactly that.
        //
        // 2. Plain function (with an output column). Looking up
        //    returns the output's stored value; on miss we get NULL.
        //    The duck IR for this is a `Term::FuncCall { name,
        //    args }` *inlined* into the consuming action's args (see
        //    `inline_terms` below). We can't emit `Action::LetExpr`
        //    here because compile.rs materializes a rule's actions
        //    into one SELECT and SQL forbids sibling-alias refs.
        let name = self.lookup_function_name(func).to_string();
        let args = match self.entries_to_terms(entries) {
            Ok(a) => a,
            Err(e) => {
                if self.deferred_err.is_none() {
                    self.deferred_err = Some(e);
                }
                Vec::new()
            }
        };
        let var = self.alloc_var(Some("lookup"));
        if self.is_eq_sort_ctor(&name) {
            let var_name = Self::var_name(&var);
            self.bound_vars.insert(var_name.clone());
            self.rule.actions.push(Action::LetCtor {
                var: var_name,
                name,
                args,
            });
        } else {
            self.inline_terms
                .insert(var.id.rep(), Term::FuncCall { name, args });
        }
        QueryEntry::Var(var)
    }

    fn subsume(&mut self, _func: FunctionId, _entries: &[QueryEntry]) -> Result<()> {
        // DuckDB does not model subsumption (`Backend::supports_subsumption
        // == false`). Term-encoded rules may still emit `subsume` for
        // hash-cons / canonicalization bookkeeping; treat such requests
        // as no-ops on duckdb. Behaviorally this is a known degradation
        // — subsumed rows continue to match — but it matches the
        // legacy duckdb pipeline, which also silently drops subsume.
        // Programs that rely on subsumption for correctness are
        // upstream-gated by `program_supports_proofs` and never reach
        // the duckdb backend.
        Ok(())
    }

    fn set(&mut self, func: FunctionId, entries: &[QueryEntry]) {
        let name = self.lookup_function_name(func).to_string();
        let mut args = match self.entries_to_terms(entries) {
            Ok(a) => a,
            Err(e) => {
                if self.deferred_err.is_none() {
                    self.deferred_err = Some(e);
                }
                return;
            }
        };
        // Bug 3 fix (insert side): strip the trailing "output" entry
        // for relation inserts. Same reasoning as in `query_table`.
        if self.is_relation(&name) && !args.is_empty() {
            args.pop();
        }
        self.rule.actions.push(Action::Insert { name, args });
    }

    fn remove(&mut self, func: FunctionId, entries: &[QueryEntry]) {
        let name = self.lookup_function_name(func).to_string();
        let args = match self.entries_to_terms(entries) {
            Ok(a) => a,
            Err(e) => {
                if self.deferred_err.is_none() {
                    self.deferred_err = Some(e);
                }
                return;
            }
        };
        // `remove` already takes only input columns (no output), so
        // no relation-arity adjustment is needed here.
        self.rule.actions.push(Action::Delete {
            name,
            key_args: args,
        });
    }

    fn union(&mut self, _l: QueryEntry, _r: QueryEntry) {
        // Unions in the existing DuckDB pipeline are emitted by term
        // encoding as `(set (pname (ordering-max l r) (ordering-min l
        // r)) ())`. Reaching the same shape from a trait-level
        // `union(l, r)` call requires knowing the eq-sort's pname,
        // which is per-sort and not directly addressable through the
        // FunctionId/Value surface. Most rule actions go through
        // encoded `set` calls on the pname function directly, so this
        // path is rarely exercised in practice.
        //
        // Defer an error to build() so the rule body can be inspected
        // / diagnosed.
        if self.deferred_err.is_none() {
            self.deferred_err = Some(anyhow!(
                "DuckRuleBuilderOps::union: direct union actions are not supported on the \
                 DuckDB backend in v1; programs that need union should go through encoded \
                 `(set (pname ...) ())` rules emitted by term encoding"
            ));
        }
    }

    fn panic(&mut self, message: String) {
        self.rule.actions.push(Action::Panic { msg: message });
    }

    fn build(self: Box<Self>) -> Result<RuleId> {
        // Bring the struct out of the Box so we can move fields.
        let Self {
            egraph,
            mut rule,
            deferred_err,
            inline_terms,
            consumed_inline,
            ..
        } = *self;

        if let Some(e) = deferred_err {
            return Err(e);
        }

        // Any `inline_terms` entry whose var was never substituted
        // (consumed) into a subsequent atom or action represents a
        // primitive call whose result is unused — typically a
        // body-level filter primitive like `(!= a b)` whose fresh
        // result var nothing reads. Emit each such entry as a body
        // `Filter(Prim(...))` so the predicate actually constrains
        // the rule. Without this the rule would fire unconditionally.
        let consumed = consumed_inline.into_inner();
        let mut unused: Vec<(u32, Term)> = inline_terms
            .into_iter()
            .filter(|(id, _)| !consumed.contains(id))
            .collect();
        unused.sort_by_key(|(id, _)| *id);
        for (_id, term) in unused {
            if matches!(term, Term::Prim(_, _) | Term::FuncCall { .. }) {
                rule.body.push(Atom::Filter(term));
            }
        }

        // If every action got silently dropped (e.g. the rule consisted
        // entirely of subsume calls, which are no-ops on duckdb), skip
        // the `add_rule` call entirely — `add_rule` insists on at least
        // one action, but a no-op rule is just an empty rule. Allocate
        // a "freed"-style slot so the returned RuleId is valid but
        // running it does nothing.
        let name = rule.name.clone();
        if rule.actions.is_empty() {
            let idx = egraph.backend_rule_names.len() as u32;
            // Mark the slot freed (None) so run_rules treats it as a
            // no-op (filter out unknown names).
            egraph.backend_rule_names.push(None);
            return Ok(RuleId::new(idx));
        }

        // Bug 1 fix: the trait surface identifies rules by `RuleId`,
        // not by ruleset (mirroring the bridge's API). DuckDB's
        // iteration loop, however, filters by ruleset name. Give each
        // trait-built rule its own unique ruleset (the rule's own
        // name) so `Backend::run_rules(&[ids])` can map ids → names
        // and use those names as the `allowed` ruleset set in
        // `run_iteration_in_set`. The frontend is the one tracking
        // which `RuleId`s belong to which logical ruleset, so this
        // synthetic ruleset is invisible to the program.
        rule.ruleset = name.clone();

        // Register the rule with the existing pipeline.
        egraph
            .add_rule(rule)
            .map_err(|e| anyhow!("DuckRuleBuilderOps::build: add_rule failed: {e}"))?;

        // Allocate a trait-level RuleId by appending to
        // `backend_rule_names`. Match the same Vec-index convention
        // as `backend_function_names`.
        let idx = egraph.backend_rule_names.len() as u32;
        egraph.backend_rule_names.push(Some(name));
        Ok(RuleId::new(idx))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use egglog_backend_trait::{
        Backend, ColumnTy, DefaultVal, FunctionConfig, MergeFn, Variable as TraitVar, VariableId,
    };

    /// Build a tiny "copy" rule via the trait surface:
    /// `(rule ((R x)) ((R x)))`. Then run it. The trait flow must
    /// not blow up end-to-end: `Backend::new_rule` →
    /// `RuleBuilderOps::query_table` / `set` / `build` →
    /// `Backend::run_rules`.
    #[test]
    fn trait_rule_roundtrip_copy_rule_smoke() {
        let mut backend: Box<dyn Backend> =
            Box::new(EGraph::new().expect("DuckDB EGraph::new failed"));

        // Register a 1-column relation `R(_)`. Use ColumnTy::Id since
        // the trait dispatch maps everything to I64 in this commit.
        let r = backend.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id],
            // For our current dispatch: AssertEq + Fail picks the
            // function-with-output path (the last column treated as
            // output). That's fine for the smoke test — the rule
            // accumulates and round-trips through compile_rule.
            default: DefaultVal::Fail,
            merge: MergeFn::AssertEq,
            name: "R_trait_smoke".to_string(),
            can_subsume: false,
        });

        // Empty table is fine.
        assert_eq!(backend.table_size(r), 0);

        // Build the rule via the trait surface. The actual SQL
        // compilation runs inside `add_rule` on `build()`.
        let rule_id = {
            let mut rb = backend.new_rule("copy_rule_smoke", true);
            let x = TraitVar {
                id: VariableId::new(0),
                name: Some("x".into()),
            };
            let entries = vec![QueryEntry::Var(x.clone())];
            rb.query_table(r, &entries, None)
                .expect("query_table failed");
            // The function expects [input, output]; for a 1-column
            // "relation"-looking schema treated as a function, the
            // body atom is 1 entry (the output column) and `set`
            // expects an empty key + the output. Match that by
            // calling set with the same single entry; either it
            // compiles or `add_rule` errors visibly.
            rb.set(r, &entries);
            rb.build().expect("build failed")
        };

        // Sanity: the rule id maps back to a registered name. Use
        // the as_any escape hatch to peek at the registry.
        {
            let eg = backend
                .as_any()
                .downcast_ref::<EGraph>()
                .expect("downcast");
            assert_eq!(
                eg.backend_rule_names
                    .get(rule_id.rep() as usize)
                    .and_then(|x| x.as_deref()),
                Some("copy_rule_smoke")
            );
        }

        // run_rules with no seeded data: the table remains empty,
        // and run_rules must not error.
        let report = backend.run_rules(&[rule_id]).expect("run_rules");
        // Accept any report shape (duckdb returns a minimal default
        // for now).
        let _ = report.search_and_apply_time();

        // free_rule: clears the slot, doesn't panic.
        backend.free_rule(rule_id);
        let eg = backend
            .as_any()
            .downcast_ref::<EGraph>()
            .expect("downcast");
        assert!(
            eg.backend_rule_names
                .get(rule_id.rep() as usize)
                .map(|x| x.is_none())
                .unwrap_or(false),
            "free_rule should clear the registry slot"
        );
    }

    /// Subsume is a silent no-op on duckdb (matching the legacy
    /// pipeline). The trait surface contract is that this does not
    /// error.
    #[test]
    fn trait_rule_subsume_is_noop() {
        let mut backend: Box<dyn Backend> =
            Box::new(EGraph::new().expect("DuckDB EGraph::new failed"));
        let r = backend.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id],
            default: DefaultVal::Fail,
            merge: MergeFn::AssertEq,
            name: "R_subsume_test".to_string(),
            can_subsume: true,
        });

        let mut rb = backend.new_rule("subsume_noop", true);
        let x = QueryEntry::Var(TraitVar {
            id: VariableId::new(0),
            name: Some("x".into()),
        });
        // Should silently succeed.
        rb.subsume(r, &[x]).expect("subsume should no-op on duckdb");
    }

    /// `query_table` with `is_subsumed = Some(true)` must error;
    /// `Some(false)` is accepted (DuckDB has no subsumption, so every
    /// row is implicitly non-subsumed).
    #[test]
    fn trait_rule_query_table_with_subsumed_errors() {
        let mut backend: Box<dyn Backend> =
            Box::new(EGraph::new().expect("DuckDB EGraph::new failed"));
        let r = backend.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id],
            default: DefaultVal::Fail,
            merge: MergeFn::AssertEq,
            name: "R_subsumed_query_test".to_string(),
            can_subsume: true,
        });

        let mut rb = backend.new_rule("subsumed_query_err", true);
        let x = QueryEntry::Var(TraitVar {
            id: VariableId::new(0),
            name: Some("x".into()),
        });
        // Some(false) is the common no-op filter that BackendRule emits
        // for every body atom; DuckDB accepts it.
        rb.query_table(r, &[x.clone()], Some(false))
            .expect("query_table with Some(false) should succeed");
        // Some(true) is unsupported.
        let err = rb.query_table(r, &[x], Some(true));
        assert!(err.is_err(), "query_table with Some(true) must error");
    }

    /// `union` is unsupported in v1 and defers the error to
    /// `build()`.
    #[test]
    fn trait_rule_union_defers_error() {
        let mut backend: Box<dyn Backend> =
            Box::new(EGraph::new().expect("DuckDB EGraph::new failed"));
        let mut rb = backend.new_rule("union_err", true);
        let a = QueryEntry::Var(TraitVar {
            id: VariableId::new(0),
            name: Some("a".into()),
        });
        let b = QueryEntry::Var(TraitVar {
            id: VariableId::new(1),
            name: Some("b".into()),
        });
        rb.union(a, b); // infallible — no panic
        let err = rb.build();
        assert!(err.is_err(), "union should surface as build error");
    }
}
