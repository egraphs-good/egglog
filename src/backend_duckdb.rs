//! Translator that runs a parsed egglog program on the
//! `egglog-bridge-duckdb` backend instead of the default
//! `egglog-bridge`.
//!
//! Phase 1.3 scope: relations + simple `:merge old`/`:merge new`
//! functions, conjunctive bodies with primitive filters, top-level
//! action inserts, `(run N)` schedules, `check`. See
//! `../duckdb-backend-plan.md` for the full plan and what's still
//! out of scope (term encoding, custom merges, containers, etc.).
//!
//! The translator owns a private "typechecker" `EGraph` that is used
//! only for parse + desugar + typecheck. Execution is dispatched to
//! the embedded `egglog_bridge_duckdb::EGraph`.

use crate::ast::*;
use crate::core::ResolvedCall;
use crate::{EGraph, Error, ResolvedNCommand};

use egglog_ast::generic_ast::{GenericAction, GenericExpr, GenericFact, Literal as EgLit};
use egglog_bridge_duckdb as duck;

/// Programs run via this backend report tuple counts for these
/// function names back to the caller.
pub struct DuckdbBackend {
    typechecker: EGraph,
    db: duck::EGraph,
    /// Function/relation names registered in the duckdb backend so
    /// far. Used to skip re-registration if we encounter the same
    /// declaration twice (e.g. via `Push`/`Pop` — currently a no-op).
    registered: hashbrown::HashSet<String>,
    /// Sort name → `unionable` flag. Used to recognize relations:
    /// `(relation foo ...)` desugars to a `(sort fooSort)` (with
    /// `unionable: false`) plus a `(constructor foo (...) fooSort)`.
    /// We detect that pattern here and treat such constructors as
    /// relations instead of needing term encoding.
    sorts: hashbrown::HashMap<String, bool>,
    /// Function name → `true` iff registered as a relation (no
    /// output column on the duckdb side). Used by the rule
    /// translator to decide whether a body atom needs a wildcard
    /// output variable.
    is_relation: hashbrown::HashMap<String, bool>,
}

impl DuckdbBackend {
    pub fn new() -> anyhow::Result<Self> {
        // Term encoding is mandatory for the DuckDB backend: any
        // program with `(datatype ...)`, `(union ...)`, or custom
        // merges only makes sense after the term-encoding pass turns
        // those into ordinary relational rules. We turn it on
        // unconditionally — the cost on programs that don't need it
        // (pure Datalog like path.egg) is small extra setup.
        let typechecker = EGraph::default().with_term_encoding_enabled();
        Ok(Self {
            typechecker,
            db: duck::EGraph::new()?,
            registered: hashbrown::HashSet::default(),
            sorts: hashbrown::HashMap::default(),
            is_relation: hashbrown::HashMap::default(),
        })
    }

    /// Parse, resolve, and dispatch each command to the duckdb backend.
    pub fn run_program(
        &mut self,
        filename: Option<String>,
        input: &str,
    ) -> Result<(), DuckdbBackendError> {
        let resolved = self
            .typechecker
            .resolve_program_to_ncommands(filename, input)
            .map_err(DuckdbBackendError::Frontend)?;
        for ncmd in resolved {
            self.dispatch(&ncmd)?;
        }
        Ok(())
    }

    /// Total tuples in a registered table.
    pub fn count(&self, name: &str) -> anyhow::Result<i64> {
        self.db.count(name)
    }

    /// Whether a row with the given args exists.
    pub fn check_exists(&self, name: &str, args: &[duck::Literal]) -> anyhow::Result<bool> {
        self.db.check_exists(name, args)
    }

    fn dispatch(&mut self, ncmd: &ResolvedNCommand) -> Result<(), DuckdbBackendError> {
        match ncmd {
            // Sorts: track the `unionable` flag so we can recognize
            // relation-desugar patterns. Sorts themselves don't need
            // any DDL — we don't model EqSort UF on this backend yet.
            GenericNCommand::Sort {
                name, unionable, ..
            } => {
                self.sorts.insert(name.clone(), *unionable);
                Ok(())
            }
            // A function declaration in the resolved IR.
            GenericNCommand::Function(decl) => self.add_function(decl),
            // Rulesets: ignore, we run all rules together.
            GenericNCommand::AddRuleset(_, _) | GenericNCommand::UnstableCombinedRuleset(..) => {
                Ok(())
            }
            GenericNCommand::NormRule { rule } => self.add_rule(rule),
            GenericNCommand::CoreAction(action) => self.run_top_action(action),
            GenericNCommand::RunSchedule(sched) => self.run_schedule(sched),
            GenericNCommand::Check(_, facts) => self.run_check(facts),
            GenericNCommand::PrintSize(_, name) => self.run_print_size(name.as_deref()),
            // Things we silently skip for now — they don't affect
            // the rule semantics on the supported subset.
            GenericNCommand::PrintFunction(..)
            | GenericNCommand::PrintOverallStatistics(..)
            | GenericNCommand::Output { .. }
            | GenericNCommand::Push(_)
            | GenericNCommand::Pop(..)
            | GenericNCommand::Extract(..)
            | GenericNCommand::ProveExists(..)
            | GenericNCommand::Input { .. } => Ok(()),
            // (fail X): X must fail. If it succeeds, surface a
            // backend-level error matching egglog's semantics.
            GenericNCommand::Fail(_, inner) => match self.dispatch(inner) {
                Ok(()) => Err(DuckdbBackendError::CheckFailed(format!(
                    "command should have failed: {}",
                    inner.to_command()
                ))),
                Err(_) => Ok(()),
            },
            // Things we deliberately error on: our reaching them
            // would mean we missed a feature, not a no-op.
            GenericNCommand::UserDefined(..) => Err(DuckdbBackendError::Unsupported(
                "user-defined commands".to_string(),
            )),
        }
    }

    fn add_function(
        &mut self,
        decl: &GenericFunctionDecl<ResolvedCall, ResolvedVar>,
    ) -> Result<(), DuckdbBackendError> {
        if self.registered.contains(&decl.name) {
            return Ok(());
        }
        let inputs = decl
            .schema
            .input
            .iter()
            .map(|s| sort_to_column_ty(s))
            .collect::<Result<Vec<_>, _>>()?;
        let output_sort = &decl.schema.output;

        let mut as_relation = false;

        // A constructor is treated as a relation here when its
        // output sort is presence-only:
        //   - non-unionable sorts (the `(relation foo)` desugar),
        //   - `:internal-hidden` constructors (term encoding's
        //     `to_delete_X`, `to_subsume_X`, cleanup helpers, etc.,
        //     whose output IDs are never read — only their rows'
        //     existence matters).
        // A constructor that's neither — e.g. `(constructor Add
        // (i64 i64) Math)` returning a real, unionable EqSort —
        // requires UF tables to model. Term encoding has emitted
        // `Add`'s UF and view tables alongside, so for our backend
        // the term table itself is also presence-only: it stores
        // (i64, i64, Math) tuples without any `set` semantics on
        // the output column.
        if decl.subtype == FunctionSubtype::Constructor {
            let unionable = self.sorts.get(output_sort).copied().unwrap_or(true);
            if !unionable || decl.internal_hidden {
                as_relation = true;
            } else {
                // Real EqSort constructor (e.g. `Add`). Term encoding
                // has lowered all "interesting" semantics (UF, view,
                // congruence) into companion tables and rules. The
                // term table itself stores (inputs..., id) tuples
                // with no merge semantics — model as a relation
                // whose key columns are inputs + output, all stored,
                // with no `set` operations.
                let mut full_inputs = inputs.clone();
                full_inputs.push(duck::ColumnTy::I64); // the EqSort ID
                self.db
                    .add_relation(&decl.name, &full_inputs)
                    .map_err(DuckdbBackendError::Backend)?;
                self.is_relation.insert(decl.name.clone(), false);
                self.registered.insert(decl.name.clone());
                return Ok(());
            }
        }

        let merge = parse_merge(decl.merge.as_ref())?;
        if !as_relation && output_sort == "Unit" {
            if merge == Some(duck::MergeMode::New) {
                return Err(DuckdbBackendError::Unsupported(format!(
                    "relation-shaped function `{}` with :merge new",
                    decl.name
                )));
            }
            as_relation = true;
        }

        if as_relation {
            self.db
                .add_relation(&decl.name, &inputs)
                .map_err(DuckdbBackendError::Backend)?;
            self.is_relation.insert(decl.name.clone(), true);
        } else {
            let output_ty = sort_to_column_ty(output_sort)?;
            let merge_mode = merge.unwrap_or(duck::MergeMode::Old);
            self.db
                .add_function(&decl.name, &inputs, output_ty, merge_mode)
                .map_err(DuckdbBackendError::Backend)?;
            self.is_relation.insert(decl.name.clone(), false);
        }
        self.registered.insert(decl.name.clone());
        Ok(())
    }

    fn add_rule(
        &mut self,
        rule: &GenericRule<ResolvedCall, ResolvedVar>,
    ) -> Result<(), DuckdbBackendError> {
        let body = rule
            .body
            .iter()
            .map(|f| self.translate_fact(f))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect();
        let actions: Vec<duck::Action> = rule
            .head
            .0
            .iter()
            .map(|a| self.translate_action(a))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect();
        self.db
            .add_rule(duck::Rule {
                name: rule.name.clone(),
                body,
                actions,
            })
            .map_err(DuckdbBackendError::Backend)
    }

    /// Whether `name` is registered as a relation (no output column
    /// on the duckdb side). Returns false if the function isn't
    /// registered yet, which is fine since we'd error on the body
    /// atom in compile.rs anyway.
    fn name_is_relation(&self, name: &str) -> bool {
        *self.is_relation.get(name).unwrap_or(&false)
    }

    fn run_top_action(
        &mut self,
        action: &GenericAction<ResolvedCall, ResolvedVar>,
    ) -> Result<(), DuckdbBackendError> {
        // Top-level actions in the resolved IR are typically:
        // - `Set(span, Func, args, val)` for explicit function sets.
        // - `Expr(span, Call(Func, args))` for relation insertions.
        // Other forms (Let, Union, Change, Panic) aren't yet handled.
        match action {
            GenericAction::Set(_, head, args, val) => {
                let ResolvedCall::Func(f) = head else {
                    return Err(DuckdbBackendError::Unsupported(format!(
                        "top-level (set ({}) ...) on non-function head",
                        head.name()
                    )));
                };
                let mut tr_args: Vec<duck::Term> = args
                    .iter()
                    .map(|e| self.translate_expr(e))
                    .collect::<Result<_, _>>()?;
                if !self.name_is_relation(&f.name) {
                    tr_args.push(self.translate_expr(val)?);
                }
                self.db
                    .insert_terms(&f.name, &tr_args)
                    .map_err(DuckdbBackendError::Backend)
            }
            GenericAction::Expr(_, GenericExpr::Call(_, ResolvedCall::Func(f), args)) => {
                let tr_args: Vec<duck::Term> = args
                    .iter()
                    .map(|e| self.translate_expr(e))
                    .collect::<Result<_, _>>()?;
                self.db
                    .insert_terms(&f.name, &tr_args)
                    .map_err(DuckdbBackendError::Backend)
            }
            // No-op: a primitive expression at top level (e.g. a
            // standalone `(check-fact)` desugaring artifact) doesn't
            // change tables.
            GenericAction::Expr(_, GenericExpr::Call(_, ResolvedCall::Primitive(_), _)) => Ok(()),
            other => Err(DuckdbBackendError::Unsupported(format!(
                "top-level action: {other}"
            ))),
        }
    }

    fn run_schedule(
        &mut self,
        sched: &GenericSchedule<ResolvedCall, ResolvedVar>,
    ) -> Result<(), DuckdbBackendError> {
        match sched {
            GenericSchedule::Run(_, cfg) => {
                // Phase 1.3: ignore `:until` and the ruleset name —
                // run all rules to saturation. Behaves like `(run N)`
                // with saturation when N is large enough.
                let _ = cfg;
                self.db
                    .run_to_saturation()
                    .map_err(DuckdbBackendError::Backend)?;
                Ok(())
            }
            GenericSchedule::Saturate(_, inner) => self.run_schedule(inner),
            GenericSchedule::Repeat(_, _, inner) => self.run_schedule(inner),
            GenericSchedule::Sequence(_, scheds) => {
                for s in scheds {
                    self.run_schedule(s)?;
                }
                Ok(())
            }
        }
    }

    fn run_check(
        &mut self,
        facts: &[GenericFact<ResolvedCall, ResolvedVar>],
    ) -> Result<(), DuckdbBackendError> {
        for fact in facts {
            self.run_check_fact(fact)?;
        }
        Ok(())
    }

    fn run_check_fact(
        &mut self,
        fact: &GenericFact<ResolvedCall, ResolvedVar>,
    ) -> Result<(), DuckdbBackendError> {
        match fact {
            // `(check (f a b))` — a row matching the args must exist.
            GenericFact::Fact(GenericExpr::Call(_, ResolvedCall::Func(f), args)) => {
                let lits: Vec<duck::Literal> = args
                    .iter()
                    .map(expr_to_literal)
                    .collect::<Result<Vec<_>, _>>()?;
                let exists = self
                    .db
                    .check_exists(&f.name, &lits)
                    .map_err(DuckdbBackendError::Backend)?;
                if !exists {
                    return Err(DuckdbBackendError::CheckFailed(format!("{fact}")));
                }
                Ok(())
            }
            // `(check (= (f args) value))` — the function's output
            // for these args must equal `value`. Same with sides
            // swapped. We currently only support i64 outputs.
            GenericFact::Eq(_, lhs, rhs) => {
                let (call_side, val_side) = match (lhs, rhs) {
                    (GenericExpr::Call(_, ResolvedCall::Func(_), _), other) => (lhs, other),
                    (other, GenericExpr::Call(_, ResolvedCall::Func(_), _)) => (rhs, other),
                    _ => {
                        return Err(DuckdbBackendError::Unsupported(format!(
                            "check fact form (only `(= (f args) val)` supported): {fact}"
                        )));
                    }
                };
                let GenericExpr::Call(_, ResolvedCall::Func(f), args) = call_side else {
                    unreachable!()
                };
                if self.name_is_relation(&f.name) {
                    return Err(DuckdbBackendError::Unsupported(format!(
                        "check on relation output (`{}` has no value): {fact}",
                        f.name
                    )));
                }
                let arg_lits: Vec<duck::Literal> = args
                    .iter()
                    .map(expr_to_literal)
                    .collect::<Result<Vec<_>, _>>()?;
                let want = expr_to_literal(val_side)?;
                let actual = self
                    .db
                    .lookup_i64(&f.name, &arg_lits)
                    .map_err(DuckdbBackendError::Backend)?;
                let want_i64 = match want {
                    duck::Literal::I64(i) => i,
                    duck::Literal::Bool(_) => {
                        return Err(DuckdbBackendError::Unsupported(format!(
                            "check (= ...) on non-i64 expected value: {fact}"
                        )));
                    }
                };
                if actual != Some(want_i64) {
                    return Err(DuckdbBackendError::CheckFailed(format!(
                        "{fact}: got {actual:?}, want Some({want_i64})"
                    )));
                }
                Ok(())
            }
            _ => Err(DuckdbBackendError::Unsupported(format!(
                "check fact form: {fact}"
            ))),
        }
    }

    fn run_print_size(&mut self, name: Option<&str>) -> Result<(), DuckdbBackendError> {
        match name {
            Some(n) => {
                let count = self.db.count(n).map_err(DuckdbBackendError::Backend)?;
                println!("({n}: {count})");
            }
            None => {
                // Print all registered tables, sorted by name for
                // determinism. Match egglog's bracket placement:
                // `((name1 N1)\n (name2 N2))` — open paren on the
                // first entry's line, close paren on the last.
                let mut names: Vec<&str> = self.registered.iter().map(|s| s.as_str()).collect();
                names.sort();
                for (i, n) in names.iter().enumerate() {
                    let count = self.db.count(n).map_err(DuckdbBackendError::Backend)?;
                    let prefix = if i == 0 { "(" } else { " " };
                    let suffix = if i + 1 == names.len() { ")" } else { "" };
                    println!("{prefix}({n} {count}){suffix}");
                }
            }
        }
        Ok(())
    }
}

/// Convert an egglog sort name to our duckdb-backend column type.
/// Primitives map to their natural types; user sorts (EqSorts, like
/// `Math` or `@pathSort`) are stored as BIGINT IDs. Term encoding
/// runs upstream of this backend, so we don't see ad-hoc EqSort
/// values in queries — only IDs.
fn sort_to_column_ty(sort: &str) -> Result<duck::ColumnTy, DuckdbBackendError> {
    match sort {
        "i64" => Ok(duck::ColumnTy::I64),
        "bool" => Ok(duck::ColumnTy::Bool),
        // Unit shouldn't reach this function (it's filtered upstream
        // when deciding relation-vs-function), but if it does we'd
        // be storing a constant — treat as i64 placeholder.
        "Unit" => Ok(duck::ColumnTy::I64),
        // All other sort names are user EqSorts. Term encoding has
        // already lowered any ad-hoc structure into rule operations
        // over IDs, so we can safely store them as BIGINT.
        // Primitive types we don't support yet (f64, String, etc.)
        // would also fall here and produce wrong results — we'll
        // need to add them as we hit them.
        _ => Ok(duck::ColumnTy::I64),
    }
}

/// Decode a `:merge old` / `:merge new` declaration. Custom-merge
/// expressions are rejected.
fn parse_merge(
    merge: Option<&GenericExpr<ResolvedCall, ResolvedVar>>,
) -> Result<Option<duck::MergeMode>, DuckdbBackendError> {
    match merge {
        None => Ok(None),
        Some(GenericExpr::Var(_, v)) if v.name == "old" => Ok(Some(duck::MergeMode::Old)),
        Some(GenericExpr::Var(_, v)) if v.name == "new" => Ok(Some(duck::MergeMode::New)),
        Some(other) => Err(DuckdbBackendError::Unsupported(format!(
            "custom merge expression `{other}`"
        ))),
    }
}

impl DuckdbBackend {
    /// Translate a body fact into 0 or 1 duck::Atoms.
    fn translate_fact(
        &self,
        fact: &GenericFact<ResolvedCall, ResolvedVar>,
    ) -> Result<Vec<duck::Atom>, DuckdbBackendError> {
        match fact {
            GenericFact::Fact(GenericExpr::Call(_, ResolvedCall::Func(f), args)) => {
                let mut tr_args: Vec<duck::Term> = args
                    .iter()
                    .map(|e| self.translate_expr(e))
                    .collect::<Result<_, _>>()?;
                // Functions on the duckdb side have arity = inputs + 1
                // (including output column). For a bare call atom we
                // don't bind the output; add a fresh wildcard. Skip
                // for relations (no output column on the duckdb side).
                if !self.name_is_relation(&f.name) {
                    tr_args.push(duck::Term::var(format!("__unused_{}", fresh_id())));
                }
                Ok(vec![duck::Atom::Func {
                    name: f.name.clone(),
                    args: tr_args,
                }])
            }
            GenericFact::Fact(GenericExpr::Call(_, ResolvedCall::Primitive(p), args)) => {
                let tr_args: Vec<duck::Term> = args
                    .iter()
                    .map(|e| self.translate_expr(e))
                    .collect::<Result<_, _>>()?;
                Ok(vec![duck::Atom::Filter(duck::Term::prim(
                    p.name(),
                    tr_args,
                ))])
            }
            GenericFact::Eq(_, lhs, rhs) => self.translate_eq(lhs, rhs),
            GenericFact::Fact(other) => Err(DuckdbBackendError::Unsupported(format!(
                "body fact form: {other}"
            ))),
        }
    }

    /// Translate `(= lhs rhs)` to one or more atoms.
    fn translate_eq(
        &self,
        lhs: &GenericExpr<ResolvedCall, ResolvedVar>,
        rhs: &GenericExpr<ResolvedCall, ResolvedVar>,
    ) -> Result<Vec<duck::Atom>, DuckdbBackendError> {
        // `(= var (f args))` where f is a function: bind var to f's
        // output via Atom::Func with var as the last arg. Same on
        // the other side. For relations we'd be binding to Unit,
        // which doesn't work — fall through to a Filter `=`.
        if let (GenericExpr::Var(_, v), GenericExpr::Call(_, ResolvedCall::Func(f), args)) =
            (lhs, rhs)
            && !self.name_is_relation(&f.name)
        {
            let mut tr_args: Vec<duck::Term> = args
                .iter()
                .map(|e| self.translate_expr(e))
                .collect::<Result<_, _>>()?;
            tr_args.push(duck::Term::var(v.name.clone()));
            return Ok(vec![duck::Atom::Func {
                name: f.name.clone(),
                args: tr_args,
            }]);
        }
        if let (GenericExpr::Call(_, ResolvedCall::Func(f), args), GenericExpr::Var(_, v)) =
            (lhs, rhs)
            && !self.name_is_relation(&f.name)
        {
            let mut tr_args: Vec<duck::Term> = args
                .iter()
                .map(|e| self.translate_expr(e))
                .collect::<Result<_, _>>()?;
            tr_args.push(duck::Term::var(v.name.clone()));
            return Ok(vec![duck::Atom::Func {
                name: f.name.clone(),
                args: tr_args,
            }]);
        }
        let lhs_t = self.translate_expr(lhs)?;
        let rhs_t = self.translate_expr(rhs)?;
        Ok(vec![duck::Atom::Filter(duck::Term::prim(
            "=",
            vec![lhs_t, rhs_t],
        ))])
    }

    /// Translate a value expression.
    fn translate_expr(
        &self,
        e: &GenericExpr<ResolvedCall, ResolvedVar>,
    ) -> Result<duck::Term, DuckdbBackendError> {
        match e {
            GenericExpr::Var(_, v) => Ok(duck::Term::var(v.name.clone())),
            GenericExpr::Lit(_, l) => Ok(duck::Term::Lit(literal_to_duck(l)?)),
            GenericExpr::Call(_, ResolvedCall::Primitive(p), args) => {
                let tr: Vec<duck::Term> = args
                    .iter()
                    .map(|e| self.translate_expr(e))
                    .collect::<Result<_, _>>()?;
                Ok(duck::Term::prim(p.name(), tr))
            }
            // Function call in expression position: a read.
            // Compiles to a SQL subquery that fetches the function's
            // output for the given arg values. Term encoding emits
            // these for global accesses like `(__v9 )`.
            GenericExpr::Call(_, ResolvedCall::Func(f), args) => {
                if self.name_is_relation(&f.name) {
                    return Err(DuckdbBackendError::Unsupported(format!(
                        "relation `{}` read in expression position (relations have no value)",
                        f.name
                    )));
                }
                let tr: Vec<duck::Term> = args
                    .iter()
                    .map(|e| self.translate_expr(e))
                    .collect::<Result<_, _>>()?;
                Ok(duck::Term::FuncCall {
                    name: f.name.clone(),
                    args: tr,
                })
            }
        }
    }

    fn translate_action(
        &self,
        action: &GenericAction<ResolvedCall, ResolvedVar>,
    ) -> Result<Vec<duck::Action>, DuckdbBackendError> {
        match action {
            GenericAction::Set(_, head, args, val) => {
                let ResolvedCall::Func(f) = head else {
                    return Err(DuckdbBackendError::Unsupported(format!(
                        "(set ({}) ...) on non-function head",
                        head.name()
                    )));
                };
                let mut tr_args: Vec<duck::Term> = args
                    .iter()
                    .map(|e| self.translate_expr(e))
                    .collect::<Result<_, _>>()?;
                // For relations the duckdb table has no output column,
                // and the val is `()` (Unit) which we can't represent
                // anyway. Skip the val push in that case.
                if !self.name_is_relation(&f.name) {
                    tr_args.push(self.translate_expr(val)?);
                }
                Ok(vec![duck::Action::Insert {
                    name: f.name.clone(),
                    args: tr_args,
                }])
            }
            GenericAction::Expr(_, GenericExpr::Call(_, ResolvedCall::Func(f), args)) => {
                let tr_args: Vec<duck::Term> = args
                    .iter()
                    .map(|e| self.translate_expr(e))
                    .collect::<Result<_, _>>()?;
                Ok(vec![duck::Action::Insert {
                    name: f.name.clone(),
                    args: tr_args,
                }])
            }
            GenericAction::Expr(_, GenericExpr::Call(_, ResolvedCall::Primitive(_), _)) => {
                Ok(Vec::new())
            }
            // (delete (f a b)) — match key columns only. The body
            // resolution is handled in compile.rs at SQL emit time.
            GenericAction::Change(_, egglog_ast::generic_ast::Change::Delete, head, args) => {
                let ResolvedCall::Func(f) = head else {
                    return Err(DuckdbBackendError::Unsupported(format!(
                        "(delete ({}) ...) on non-function head",
                        head.name()
                    )));
                };
                let tr_args: Vec<duck::Term> = args
                    .iter()
                    .map(|e| self.translate_expr(e))
                    .collect::<Result<_, _>>()?;
                Ok(vec![duck::Action::Delete {
                    name: f.name.clone(),
                    key_args: tr_args,
                }])
            }
            // Subsume: not modeled yet — the underlying duck::EGraph
            // doesn't have a "subsumed" flag. Skip silently for now;
            // subsumed-aware queries aren't generated by anything we
            // currently translate.
            GenericAction::Change(_, egglog_ast::generic_ast::Change::Subsume, _, _) => {
                Ok(Vec::new())
            }
            // Let bindings in actions: bind a fresh value-typed
            // variable for use later in the same action sequence.
            // We don't currently support this — the term-encoded
            // examples mostly use `let v (Add a b)` which, after our
            // term-encoding-friendly translation, becomes a function
            // call we can already insert. For now, error.
            GenericAction::Let(_, _, _) => Err(DuckdbBackendError::Unsupported(format!(
                "(let ...) in rule actions: {action}"
            ))),
            other => Err(DuckdbBackendError::Unsupported(format!(
                "rule action: {other}"
            ))),
        }
    }
}

fn literal_to_duck(l: &EgLit) -> Result<duck::Literal, DuckdbBackendError> {
    match l {
        EgLit::Int(i) => Ok(duck::Literal::I64(*i)),
        EgLit::Bool(b) => Ok(duck::Literal::Bool(*b)),
        // Unit is the "value" for relations. We shouldn't reach here
        // for inserts (those skip the value column for relations) but
        // if some context forces a Unit-typed value, encode it as 0.
        EgLit::Unit => Ok(duck::Literal::I64(0)),
        other => Err(DuckdbBackendError::Unsupported(format!(
            "literal type: {other:?}"
        ))),
    }
}

fn expr_to_literal(
    e: &GenericExpr<ResolvedCall, ResolvedVar>,
) -> Result<duck::Literal, DuckdbBackendError> {
    match e {
        GenericExpr::Lit(_, l) => literal_to_duck(l),
        other => Err(DuckdbBackendError::Unsupported(format!(
            "expected literal, got: {other}"
        ))),
    }
}

/// A monotonically increasing counter so that fresh wildcard
/// variables introduced by the translator don't collide. Lives in
/// a thread-local because we don't currently have a translator
/// state object to thread through.
fn fresh_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, thiserror::Error)]
pub enum DuckdbBackendError {
    #[error("egglog frontend error: {0}")]
    Frontend(Error),
    #[error("duckdb backend error: {0}")]
    Backend(anyhow::Error),
    #[error("not yet supported by duckdb backend: {0}")]
    Unsupported(String),
    #[error("check failed: {0}")]
    CheckFailed(String),
}
