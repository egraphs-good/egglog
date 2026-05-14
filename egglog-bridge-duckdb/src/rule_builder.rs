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
    ColumnTy, ExternalFunctionId, FunctionId, QueryEntry, RuleBuilderOps, RuleId, Variable,
    VariableId,
};
use egglog_numeric_id::NumericId;

use crate::{Action, Atom, EGraph, Literal, Rule, Term};

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
    /// `QueryEntry::Const` with `ColumnTy::Base(_)` → error (decoding
    /// requires the `BaseValuePool` wiring from Commit 11).
    fn entry_to_term(&self, entry: &QueryEntry) -> Result<Term> {
        match entry {
            QueryEntry::Var(v) => Ok(Term::Var(Self::var_name(v))),
            QueryEntry::Const { val, ty } => match ty {
                ColumnTy::Id => Ok(Term::Lit(Literal::I64(val.rep() as i64))),
                ColumnTy::Base(_) => Err(anyhow!(
                    "DuckRuleBuilderOps: decoding a base-value constant \
                     requires the BaseValuePool wiring scheduled for Phase 2 \
                     Commit 11; got a QueryEntry::Const of type \
                     ColumnTy::Base(_)"
                )),
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
        if is_subsumed.is_some() {
            return Err(anyhow!(
                "DuckRuleBuilderOps::query_table: subsumption filter is not supported on the \
                 DuckDB backend (Backend::supports_subsumption returns false)"
            ));
        }
        let name = self.lookup_function_name(func).to_string();
        let args = self.entries_to_terms(entries)?;
        self.rule.body.push(Atom::Func { name, args });
        Ok(())
    }

    fn query_prim(
        &mut self,
        _func: ExternalFunctionId,
        _entries: &[QueryEntry],
        _ret_ty: ColumnTy,
    ) -> Result<()> {
        // External primitives go through a name registry that is
        // wired up in Phase 2 Commit 12. Until then, the trait
        // surface has no way to map an `ExternalFunctionId` to a
        // duckdb-side primitive name; report an error.
        Err(anyhow!(
            "DuckRuleBuilderOps::query_prim: external primitives are not yet wired through the \
             DuckDB trait surface (deferred to Phase 2 Commit 12)"
        ))
    }

    fn call_external_func(
        &mut self,
        _func: ExternalFunctionId,
        _args: &[QueryEntry],
        _ret_ty: ColumnTy,
        _panic_msg: String,
    ) -> QueryEntry {
        // Infallible signature; defer the error to build() so callers
        // who never finish the rule don't trip over it.
        if self.deferred_err.is_none() {
            self.deferred_err = Some(anyhow!(
                "DuckRuleBuilderOps::call_external_func: external primitives are not yet wired \
                 through the DuckDB trait surface (deferred to Phase 2 Commit 12)"
            ));
        }
        // Return a dummy variable so the caller can keep building.
        let var = self.alloc_var(Some("__ext_unimpl"));
        QueryEntry::Var(var)
    }

    fn lookup(
        &mut self,
        func: FunctionId,
        entries: &[QueryEntry],
        _panic_msg: String,
    ) -> QueryEntry {
        // `lookup` reads a function's output column for the given
        // input args. On DuckDB this is a `Term::FuncCall { name,
        // args }` bound to a fresh variable via an `Action::LetExpr`.
        //
        // Subsequent actions can then reference `var`.
        //
        // Note: this is an *action*, not a body atom. The bridge's
        // `lookup` is an RHS construct; on the bridge side it appears
        // in actions only. The duck-side compile pipeline handles
        // `Action::LetExpr` accordingly.
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
        let var_name = Self::var_name(&var);
        self.rule.actions.push(Action::LetExpr {
            var: var_name,
            expr: Term::FuncCall { name, args },
        });
        QueryEntry::Var(var)
    }

    fn subsume(&mut self, _func: FunctionId, _entries: &[QueryEntry]) -> Result<()> {
        Err(anyhow!(
            "DuckRuleBuilderOps::subsume: subsumption is not supported on the DuckDB backend \
             (Backend::supports_subsumption returns false)"
        ))
    }

    fn set(&mut self, func: FunctionId, entries: &[QueryEntry]) {
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
            rule,
            deferred_err,
            ..
        } = *self;

        if let Some(e) = deferred_err {
            return Err(e);
        }

        // Register the rule with the existing pipeline.
        let name = rule.name.clone();
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

    /// Subsume must error at the call site.
    #[test]
    fn trait_rule_subsume_errors() {
        let mut backend: Box<dyn Backend> =
            Box::new(EGraph::new().expect("DuckDB EGraph::new failed"));
        let r = backend.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id],
            default: DefaultVal::Fail,
            merge: MergeFn::AssertEq,
            name: "R_subsume_test".to_string(),
            can_subsume: true,
        });

        let mut rb = backend.new_rule("subsume_err", true);
        let x = QueryEntry::Var(TraitVar {
            id: VariableId::new(0),
            name: Some("x".into()),
        });
        let err = rb.subsume(r, &[x]);
        assert!(err.is_err(), "subsume should error on duckdb");
    }

    /// `query_table` with `is_subsumed = Some(_)` must error.
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
        let err = rb.query_table(r, &[x], Some(false));
        assert!(err.is_err(), "query_table with subsumed filter must error");
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
