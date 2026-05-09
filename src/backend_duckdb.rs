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
use egglog_ast::util::ListDisplay;
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
    /// Names of EqSort constructors — those whose calls allocate
    /// fresh IDs via the duckdb sequence. Distinct from regular
    /// functions: those are read-back via subqueries.
    eq_sort_ctor: hashbrown::HashSet<String>,
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
            eq_sort_ctor: hashbrown::HashSet::default(),
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
        if std::env::var("DUCK_TRACE_CMDS").is_ok() {
            eprintln!("[duck/cmd] {}", ncmd.to_command());
        }
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

        // Distinguish two kinds of `:internal-hidden` constructors
        // emitted by term encoding:
        //   - helper constructors (to_delete_X, to_subsume_X,
        //     cleanup helpers) — output IDs are never read; only
        //     row existence matters. Detected by
        //     `internal_hidden && !unextractable`.
        //   - user-level constructors (path, edge, Add, etc.) —
        //     marked `:internal-hidden :unextractable` by term
        //     encoding so the user-facing IDs aren't reused. These
        //     still need real EqSort allocation because the emitted
        //     view/UF tables key on the IDs.
        if decl.subtype == FunctionSubtype::Constructor {
            let is_helper = decl.internal_hidden && !decl.unextractable;
            if is_helper {
                as_relation = true;
            } else {
                // Real EqSort constructor. The bridge crate emits a
                // table with (input..., id) cols and a PK over all
                // of them, plus an `allocate_and_insert` path that
                // appends a fresh ID per call. Multiple rows per
                // input with distinct IDs are allowed; the
                // congruence + rebuild rules emitted by term
                // encoding unify them later.
                self.db
                    .add_eq_sort_constructor(&decl.name, &inputs)
                    .map_err(DuckdbBackendError::Backend)?;
                // is_relation = false so body atoms `(Add a b)` add
                // a wildcard for the ID column to match the table
                // schema.
                self.is_relation.insert(decl.name.clone(), false);
                self.eq_sort_ctor.insert(decl.name.clone());
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
        if std::env::var("DUCK_TRACE_RULES").is_ok() {
            eprintln!("[duck] rule {}", rule.name);
            eprintln!("       body:");
            for f in &rule.body {
                eprintln!("         {f}");
            }
            eprintln!("       actions:");
            for a in &rule.head.0 {
                eprintln!("         {a}");
            }
        }
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
        // If every action translated to a no-op (e.g. subsume-only
        // rules that we silently skip), there's nothing to emit. The
        // bridge crate would error on the empty list; skip the rule
        // entirely.
        if actions.is_empty() {
            return Ok(());
        }
        self.db
            .add_rule(duck::Rule {
                name: rule.name.clone(),
                ruleset: rule.ruleset.clone(),
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

    /// Whether `name` is a constructor whose call should allocate a
    /// fresh EqSort ID via the duckdb backend's sequence. True for
    /// constructors we registered with an output column (real
    /// EqSort), false for constructors we registered as relations
    /// (deferred-action helpers, the (relation foo) desugar) or for
    /// regular functions.
    fn eq_sort_constructor(&self, name: &str) -> bool {
        self.eq_sort_ctor.contains(name)
    }

    /// Translate a top-level expression to a `duck::Term`, eagerly
    /// running side-effects: for an EqSort constructor call, this
    /// allocates a fresh id and returns it as a literal Term. For
    /// other function calls, it returns a `Term::FuncCall` (compiled
    /// to a SQL subquery). Literals and primitive expressions stay
    /// as ordinary Terms.
    fn top_arg_term(
        &mut self,
        e: &GenericExpr<ResolvedCall, ResolvedVar>,
    ) -> Result<duck::Term, DuckdbBackendError> {
        match e {
            GenericExpr::Call(_, ResolvedCall::Func(f), args)
                if self.eq_sort_constructor(&f.name) =>
            {
                let lits = args
                    .iter()
                    .map(|a| self.eval_top_expr(a))
                    .collect::<Result<Vec<_>, _>>()?;
                let id = self
                    .db
                    .allocate_and_insert(&f.name, &lits)
                    .map_err(DuckdbBackendError::Backend)?;
                Ok(duck::Term::Lit(duck::Literal::I64(id)))
            }
            _ => self.translate_expr(e),
        }
    }

    /// Evaluate a top-level expression to a literal value, eagerly
    /// allocating EqSort ids for any nested constructor calls and
    /// reading global function values via SQL subquery.  Used to
    /// reduce things like `(L 0)` or `(Prog (L 0))` to a concrete
    /// `Literal::I64(<id>)` so they can flow into a top-level Set.
    fn eval_top_expr(
        &mut self,
        e: &GenericExpr<ResolvedCall, ResolvedVar>,
    ) -> Result<duck::Literal, DuckdbBackendError> {
        match e {
            GenericExpr::Lit(_, l) => literal_to_duck(l),
            GenericExpr::Var(_, v) => Err(DuckdbBackendError::Unsupported(format!(
                "unbound variable in top-level expression: {}",
                v.name
            ))),
            GenericExpr::Call(_, ResolvedCall::Func(f), args) => {
                if self.eq_sort_constructor(&f.name) {
                    let lits = args
                        .iter()
                        .map(|a| self.eval_top_expr(a))
                        .collect::<Result<Vec<_>, _>>()?;
                    let id = self
                        .db
                        .allocate_and_insert(&f.name, &lits)
                        .map_err(DuckdbBackendError::Backend)?;
                    Ok(duck::Literal::I64(id))
                } else if self.name_is_relation(&f.name) {
                    Err(DuckdbBackendError::Unsupported(format!(
                        "relation `{}` read in top-level expression position",
                        f.name
                    )))
                } else {
                    // A regular function (with a real output value).
                    // Read its current output via lookup. We only
                    // support i64 output here.
                    let lits = args
                        .iter()
                        .map(|a| self.eval_top_expr(a))
                        .collect::<Result<Vec<_>, _>>()?;
                    let v = self
                        .db
                        .lookup_i64(&f.name, &lits)
                        .map_err(DuckdbBackendError::Backend)?
                        .ok_or_else(|| {
                            DuckdbBackendError::Unsupported(format!(
                                "no row found for {}({:?})",
                                f.name, lits
                            ))
                        })?;
                    Ok(duck::Literal::I64(v))
                }
            }
            GenericExpr::Call(_, ResolvedCall::Primitive(_), _) => {
                Err(DuckdbBackendError::Unsupported(format!(
                    "primitive call in top-level expression: {e}"
                )))
            }
        }
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
                // Args: each becomes a Term. For constructor calls
                // we recurse via eval_top_expr to get a Lit id, then
                // wrap as Term::Lit; for everything else, use
                // translate_expr (subquery for global reads).
                let mut tr_args: Vec<duck::Term> = Vec::with_capacity(args.len() + 1);
                for a in args {
                    tr_args.push(self.top_arg_term(a)?);
                }
                if !self.name_is_relation(&f.name) {
                    tr_args.push(self.top_arg_term(val)?);
                }
                self.db
                    .insert_terms(&f.name, &tr_args)
                    .map_err(DuckdbBackendError::Backend)
            }
            GenericAction::Expr(_, GenericExpr::Call(_, ResolvedCall::Func(f), args)) => {
                if self.eq_sort_constructor(&f.name) {
                    let mut lits = Vec::with_capacity(args.len());
                    for a in args {
                        lits.push(self.eval_top_expr(a)?);
                    }
                    self.db
                        .allocate_and_insert(&f.name, &lits)
                        .map(|_| ())
                        .map_err(DuckdbBackendError::Backend)
                } else {
                    let mut tr_args: Vec<duck::Term> = Vec::with_capacity(args.len());
                    for a in args {
                        tr_args.push(self.top_arg_term(a)?);
                    }
                    self.db
                        .insert_terms(&f.name, &tr_args)
                        .map_err(DuckdbBackendError::Backend)
                }
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
            // `(run rs)` is a *single* iteration. The wrapping
            // schedule (Repeat/Saturate) decides how many times.
            GenericSchedule::Run(_, cfg) => {
                let rs = if cfg.ruleset.is_empty() {
                    None
                } else {
                    Some(cfg.ruleset.as_str())
                };
                self.db
                    .run_iteration_in(rs)
                    .map_err(DuckdbBackendError::Backend)?;
                Ok(())
            }
            // `(saturate inner)` — run inner repeatedly until none
            // of its rules' SQL statements affect any rows. We
            // detect that by snapshotting `rules_affected_total`
            // (the sum of rows affected by any DML executed inside
            // run_iteration_in) before/after one execution; equal
            // counters → no rule fired → stop. This is more reliable
            // than comparing total tuple counts, which can balance
            // out (deletes equal inserts within a rule).
            GenericSchedule::Saturate(_, inner) => {
                loop {
                    let before = self.db.rules_affected_total();
                    self.run_schedule(inner)?;
                    let after = self.db.rules_affected_total();
                    if before == after {
                        break;
                    }
                }
                Ok(())
            }
            // `(repeat N inner)` — run inner up to N times, stopping
            // early on a fixpoint (matches egglog: `(repeat 100
            // (run))` saturates earlier if it can).
            GenericSchedule::Repeat(_, limit, inner) => {
                for _ in 0..*limit {
                    let before = self.db.rules_affected_total();
                    self.run_schedule(inner)?;
                    let after = self.db.rules_affected_total();
                    if before == after {
                        break;
                    }
                }
                Ok(())
            }
            GenericSchedule::Sequence(_, scheds) => {
                for s in scheds {
                    self.run_schedule(s)?;
                }
                Ok(())
            }
        }
    }

    /// Total number of rows across every registered table. Used as
    /// a coarse fixpoint detector for Saturate / Repeat.
    fn total_tuples(&self) -> Result<i64, DuckdbBackendError> {
        let mut total: i64 = 0;
        for name in &self.registered {
            total += self.db.count(name).map_err(DuckdbBackendError::Backend)?;
        }
        Ok(total)
    }

    fn run_check(
        &mut self,
        facts: &[GenericFact<ResolvedCall, ResolvedVar>],
    ) -> Result<(), DuckdbBackendError> {
        // A `(check fact ...)` is the same query as a rule body —
        // a conjunctive query that passes iff there's any matching
        // assignment. Reuse the body compiler instead of pattern-
        // matching on individual fact shapes.
        let atoms: Vec<duck::Atom> = facts
            .iter()
            .map(|f| self.translate_fact(f))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect();
        let exists = self.db.body_exists(&atoms).map_err(DuckdbBackendError::Backend)?;
        if !exists {
            return Err(DuckdbBackendError::CheckFailed(format!(
                "{}",
                ListDisplay(facts, " ")
            )));
        }
        Ok(())
    }

    fn run_print_size(&mut self, name: Option<&str>) -> Result<(), DuckdbBackendError> {
        match name {
            Some(n) => {
                let count = self.db.count(n).map_err(DuckdbBackendError::Backend)?;
                println!("({n}: {count})");
            }
            None => {
                // Match egglog's behavior: report sizes of
                // user-visible functions only. After term encoding,
                // user names map to their view tables; internal
                // tables (UF, term, deferred-action helpers, fresh
                // sym names, etc.) carry the `@` prefix and are
                // hidden. For each user-level name `foo`, report the
                // size of `@fooView` instead of `foo` itself, since
                // the view has the canonicalized count.
                // Hide internal tables: `@`-prefixed (term encoding's
                // helpers, view tables, etc.) and `$`-prefixed
                // (user globals, lifted to nullary constructors —
                // egglog hides these from print-size too).
                let user_names: Vec<&str> = self
                    .registered
                    .iter()
                    .map(|s| s.as_str())
                    .filter(|n| !n.starts_with('@') && !n.starts_with('$'))
                    .collect();
                let mut sorted: Vec<&str> = user_names.clone();
                sorted.sort();
                for (i, n) in sorted.iter().enumerate() {
                    // Prefer the view-table count if it exists.
                    let view_name = format!("@{n}View");
                    let count_name: &str = if self.registered.contains(&view_name) {
                        // Owned String; we'll need an owned variant.
                        // (small allocation, fine for print-size.)
                        Box::leak(view_name.into_boxed_str())
                    } else {
                        n
                    };
                    let count = self
                        .db
                        .count(count_name)
                        .map_err(DuckdbBackendError::Backend)?;
                    let prefix = if i == 0 { "(" } else { " " };
                    let suffix = if i + 1 == sorted.len() { ")" } else { "" };
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
        "f64" => Ok(duck::ColumnTy::F64),
        "String" => Ok(duck::ColumnTy::Str),
        // Unit shouldn't reach this function (it's filtered upstream
        // when deciding relation-vs-function), but if it does we'd
        // be storing a constant — treat as i64 placeholder.
        "Unit" => Ok(duck::ColumnTy::I64),
        // Other sort names are user EqSorts. Term encoding has
        // lowered any ad-hoc structure into rule operations over
        // IDs, so we can safely store them as BIGINT. Sorts we don't
        // know fall here too; if a program turns out to need them
        // typed differently, we'll see test failures.
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
        // `(= var1 var2)` — egglog rule semantics is unification:
        // both vars name the same value. Emit a Bind so the second
        // var resolves to whatever the first one is already bound
        // to. (If neither is bound by a prior atom, walk_body will
        // error when it tries to resolve the expr — fine, since
        // such a rule is ill-formed.)
        if let (GenericExpr::Var(_, v1), GenericExpr::Var(_, v2)) = (lhs, rhs) {
            return Ok(vec![duck::Atom::Bind {
                var: v1.name.clone(),
                expr: duck::Term::var(v2.name.clone()),
            }]);
        }
        // `(= var (primitive_call args))` — bind var to the
        // primitive's result AND require the result to be truthy
        // (since term encoding rewrites the original `(prim args)`
        // filter into this form to satisfy proof normal form).
        // Comparison/boolean primitives in egglog return `Unit` on
        // success and fail on no-result; in SQL we model that as a
        // WHERE constraint on the same expression. Arithmetic
        // primitives return values, but for those `(= var (+ x y))`
        // is a pure binding — and our `Atom::Filter` over an
        // arithmetic SQL expression evaluates non-zero as truthy,
        // which is conservatively safe (rules in practice don't use
        // `(= var (+ x y))` as a filter).
        for (var_side, expr_side) in [(lhs, rhs), (rhs, lhs)] {
            if let GenericExpr::Var(_, v) = var_side
                && let GenericExpr::Call(_, ResolvedCall::Primitive(p), _) = expr_side
            {
                let expr = self.translate_expr(expr_side)?;
                let mut out = vec![duck::Atom::Bind {
                    var: v.name.clone(),
                    expr: expr.clone(),
                }];
                if is_filter_primitive(p.name()) {
                    out.push(duck::Atom::Filter(expr));
                }
                return Ok(out);
            }
        }
        // `(= var (f args))` where f is a function-with-output:
        // bind var to f's output via Atom::Func with var as the
        // last arg.
        // `(= var (relation args))`: relations don't have a value,
        // so the binding is to Unit. We model this as a bare body
        // atom on the relation; the surrounding rule won't actually
        // use `var` for anything meaningful (and if it does, the
        // resulting SQL will fail loudly rather than silently — fine).
        for (var_side, call_side) in [(lhs, rhs), (rhs, lhs)] {
            if let GenericExpr::Var(_, _) = var_side
                && let GenericExpr::Call(_, ResolvedCall::Func(f), args) = call_side
            {
                let tr_args: Vec<duck::Term> = args
                    .iter()
                    .map(|e| self.translate_expr(e))
                    .collect::<Result<_, _>>()?;
                if self.name_is_relation(&f.name) {
                    return Ok(vec![duck::Atom::Func {
                        name: f.name.clone(),
                        args: tr_args,
                    }]);
                }
                let GenericExpr::Var(_, v) = var_side else {
                    unreachable!()
                };
                let mut tr_args = tr_args;
                tr_args.push(duck::Term::var(v.name.clone()));
                return Ok(vec![duck::Atom::Func {
                    name: f.name.clone(),
                    args: tr_args,
                }]);
            }
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
            // (let v (C a b)) where C is an EqSort constructor:
            // allocate a fresh ID via C, bind v.
            GenericAction::Let(
                _,
                v,
                GenericExpr::Call(_, ResolvedCall::Func(f), ctor_args),
            ) if self.eq_sort_constructor(&f.name) => {
                let tr_args: Vec<duck::Term> = ctor_args
                    .iter()
                    .map(|e| self.translate_expr(e))
                    .collect::<Result<_, _>>()?;
                Ok(vec![duck::Action::LetCtor {
                    var: v.name.clone(),
                    name: f.name.clone(),
                    args: tr_args,
                }])
            }
            // (let v <pure expr>) — bind v to a primitive/literal/
            // function-output expression. No allocation, just a
            // computed column in the materialized match table.
            GenericAction::Let(_, v, rhs) => {
                let expr = self.translate_expr(rhs)?;
                Ok(vec![duck::Action::LetExpr {
                    var: v.name.clone(),
                    expr,
                }])
            }
            other => Err(DuckdbBackendError::Unsupported(format!(
                "rule action: {other}"
            ))),
        }
    }
}

/// Whether a primitive's "success" value should be interpreted as a
/// filter constraint (returns Unit-on-success in egglog). Term
/// encoding wraps these in `(= var (prim ...))` to satisfy proof
/// normal form, and the rule semantics is "filter on the
/// primitive holding".
fn is_filter_primitive(name: &str) -> bool {
    matches!(
        name,
        "<" | "<=" | ">" | ">=" | "=" | "!=" | "bool-!=" | "guard" | "and" | "or" | "not"
    )
}

fn literal_to_duck(l: &EgLit) -> Result<duck::Literal, DuckdbBackendError> {
    match l {
        EgLit::Int(i) => Ok(duck::Literal::I64(*i)),
        EgLit::Bool(b) => Ok(duck::Literal::Bool(*b)),
        EgLit::Float(f) => Ok(duck::Literal::F64(f.into_inner())),
        EgLit::String(s) => Ok(duck::Literal::Str(s.clone())),
        // Unit is the "value" for relations. We shouldn't reach here
        // for inserts (those skip the value column for relations) but
        // if some context forces a Unit-typed value, encode it as 0.
        EgLit::Unit => Ok(duck::Literal::I64(0)),
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
