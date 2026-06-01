//! Compile a high-level `Rule` into a `CompiledVariant` per
//! seminaive variant.
//!
//! A rule with N function-table atoms produces N variants. Variant i
//! adds `WHERE alias_i.ts >= ?1` on the focused atom and reads the
//! current epoch as `?2`. Both come in as bind parameters at run
//! time. Each variant runs all its actions in order on every match.
//!
//! When a rule has any `Action::LetCtor` (allocate a fresh EqSort
//! ID and bind it for use in later actions), the compiler switches
//! to the *materialized* path:
//!   1. CREATE OR REPLACE TEMP TABLE __m_<rule>_<variant> AS
//!      SELECT <body cols>, nextval('seq') AS v1, … FROM <body
//!      atoms> WHERE focus.ts >= ?1.
//!   2. Each subsequent action runs as INSERT/DELETE FROM __m,
//!      referencing both body bindings and let-allocated ids by
//!      column.
//! The materialize SQL has `?1` baked in (the focus ts threshold);
//! the action SQLs only need `?2` (the current epoch).

use anyhow::{Result, anyhow};
use std::collections::HashMap;

use crate::{
    Action, Atom, CompiledAction, CompiledRule, CompiledVariant, FunctionInfo, Rule, Term,
    conflict_clause, prefix_with_comma, q,
};

/// Glue a list of WHERE atoms into a single SQL fragment that
/// starts with a leading space + `WHERE`, or "" if empty.
fn format_where(parts: &[String]) -> String {
    if parts.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", parts.join(" AND "))
    }
}

/// Target table written by an action, if any. Used by the runner
/// to bump the table's insert watermark after a successful execute.
fn action_target(a: &Action, functions: &HashMap<String, FunctionInfo>) -> Option<String> {
    match a {
        Action::Insert { name, .. } | Action::Delete { name, .. } => Some(name.clone()),
        // LetCtor that has no associated `@<name>View` table actually
        // INSERTs into the constructor itself (term-encoding helpers
        // `@to_delete_X`, `@to_subsume_X`, `mergecleanup-N`). For
        // those, treat the constructor as the action's target so the
        // body-table watermark gate sees the write and downstream
        // rules don't get skipped.
        Action::LetCtor { name, .. } => {
            let view = format!("@{name}View");
            if functions.contains_key(&view) {
                None
            } else {
                Some(name.clone())
            }
        }
        Action::LetExpr { .. } | Action::Panic { .. } => None,
    }
}

/// Mark each body atom as demotable iff it is a function-table
/// lookup whose function has a registered native union-find UDF
/// (`info.native_uf_udf == Some(_)`), the inputs are bound by an
/// earlier body atom, and the output is a fresh `Var`. The compiler
/// will emit a scalar UDF call binding the output var rather than
/// adding a join atom.
///
/// Walks the body greedily in source order. A demoted atom only
/// contributes its output binding, so later atoms that depend on
/// it can still resolve.
fn compute_native_uf_demote(
    body: &[Atom],
    functions: &HashMap<String, FunctionInfo>,
) -> Vec<bool> {
    let mut demote = vec![false; body.len()];
    let mut bound: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (i, atom) in body.iter().enumerate() {
        match atom {
            Atom::Func { name, args } => {
                let info = match functions.get(name) {
                    Some(i) => i,
                    None => continue,
                };
                let inputs_bound = args[..info.inputs_len].iter().all(|t| match t {
                    Term::Var(v) => bound.contains(v),
                    _ => true,
                });
                let output_fresh = info.inputs_len + 1 == args.len()
                    && matches!(&args[info.inputs_len], Term::Var(v) if !bound.contains(v));
                if info.native_uf_udf.is_some() && inputs_bound && output_fresh {
                    demote[i] = true;
                    if let Term::Var(v) = &args[info.inputs_len] {
                        bound.insert(v.clone());
                    }
                } else {
                    for arg in args {
                        if let Term::Var(v) = arg {
                            bound.insert(v.clone());
                        }
                    }
                }
            }
            Atom::Bind { var, .. } => {
                bound.insert(var.clone());
            }
            Atom::Filter(_) => {}
        }
    }
    demote
}

/// Lift inline `Term::FuncCall`s that target an EqSort constructor
/// (or its term-encoded view) into explicit `LetCtor` actions, so
/// the existing hash-cons machinery in the materialized path can
/// allocate-or-reuse an ID for them instead of doing a bare lookup.
///
/// Term encoding's proof mode is the only thing that currently emits
/// nested calls here — e.g. the congruence rule's
/// `(set (pname l s) (Trans p1 (Sym p2)))`. Without this lift the
/// nested `(Trans …)` and `(Sym …)` compile to scalar subqueries
/// against `@_trans` / `@_sym`, which return NULL when the proof
/// term has never been constructed; the NULL then hits the
/// `NOT NULL` constraint on `pname.c2`. After lifting, each nested
/// call becomes a fresh `LetCtor` that the materialized path knows
/// how to hash-cons-or-allocate.
///
/// Only Calls whose target is `eq_sort_ctor` get lifted: regular
/// functions still resolve via a lookup, which is what the encoder
/// expects for them.
fn lift_nested_eq_sort_calls(
    rule: &Rule,
    functions: &HashMap<String, FunctionInfo>,
) -> Rule {
    let mut counter: usize = 0;
    let mut new_actions: Vec<Action> = Vec::with_capacity(rule.actions.len());
    for action in &rule.actions {
        let mut lifted = action.clone();
        match &mut lifted {
            Action::Insert { args, .. } | Action::Delete { key_args: args, .. } => {
                let new_args = args
                    .iter()
                    .cloned()
                    .map(|t| lift_term(t, &rule.name, &mut counter, &mut new_actions, functions))
                    .collect();
                *args = new_args;
            }
            Action::LetCtor { args, .. } => {
                let new_args = args
                    .iter()
                    .cloned()
                    .map(|t| lift_term(t, &rule.name, &mut counter, &mut new_actions, functions))
                    .collect();
                *args = new_args;
            }
            Action::LetExpr { expr, .. } => {
                *expr = lift_term(
                    expr.clone(),
                    &rule.name,
                    &mut counter,
                    &mut new_actions,
                    functions,
                );
            }
            Action::Panic { .. } => {}
        }
        new_actions.push(lifted);
    }
    Rule {
        name: rule.name.clone(),
        ruleset: rule.ruleset.clone(),
        body: rule.body.clone(),
        actions: new_actions,
    }
}

/// Recursively lift FuncCalls in `term`. Lifted calls become
/// `LetCtor` actions pushed to `out` in dependency order (innermost
/// first, outermost last), and the call site is replaced with a
/// `Var` reference to the lifted name. Calls whose target is *not*
/// an `eq_sort_ctor` are left in place — they read existing
/// function rows.
fn lift_term(
    term: Term,
    rule_name: &str,
    counter: &mut usize,
    out: &mut Vec<Action>,
    functions: &HashMap<String, FunctionInfo>,
) -> Term {
    match term {
        Term::FuncCall { name, args } => {
            let lifted_args: Vec<Term> = args
                .into_iter()
                .map(|a| lift_term(a, rule_name, counter, out, functions))
                .collect();
            let is_eq_sort_ctor = functions
                .get(&name)
                .map(|i| i.eq_sort_ctor)
                .unwrap_or(false);
            if is_eq_sort_ctor {
                let var = format!("__lifted_{rule_name}_{}", *counter);
                *counter += 1;
                out.push(Action::LetCtor {
                    var: var.clone(),
                    name,
                    args: lifted_args,
                });
                Term::Var(var)
            } else {
                Term::FuncCall {
                    name,
                    args: lifted_args,
                }
            }
        }
        Term::Prim(op, args) => {
            let lifted_args: Vec<Term> = args
                .into_iter()
                .map(|a| lift_term(a, rule_name, counter, out, functions))
                .collect();
            Term::Prim(op, lifted_args)
        }
        other => other,
    }
}


pub(crate) fn compile_rule(
    rule: &Rule,
    functions: &HashMap<String, FunctionInfo>,
    proofs_enabled: bool,
) -> Result<CompiledRule> {
    // Rewrite the rule before validation/compilation so any nested
    // eq-sort constructor calls in action positions become explicit
    // `LetCtor`s (hash-cons-or-allocate). See
    // `lift_nested_eq_sort_calls` for why.
    //
    // NB: there used to be a `dedupe_body_atoms` pass here that merged
    // body atoms with identical inputs. It existed only to make the
    // inlined Herbie dumps' deeply-repeated patterns tractable, but it
    // was unsound for self-joins (it collapsed the nullary-`:merge`
    // reconciliation and per-sort congruence rules into always-false
    // `x <> x` filters) and caused multiple bugs. It's been removed
    // along with the Herbie `inline-bigrat-lets` workaround that
    // created the duplicate atoms; non-inlined dumps don't blow up.
    let rule_owned = lift_nested_eq_sort_calls(rule, functions);
    let rule = &rule_owned;
    // Validate function-table atoms.
    for atom in &rule.body {
        if let Atom::Func { name, args } = atom {
            let info = functions
                .get(name)
                .ok_or_else(|| anyhow!("rule {}: unknown function {name}", rule.name))?;
            if args.len() != info.arity() {
                return Err(anyhow!(
                    "rule {}: atom {name} has {} args, expected {}",
                    rule.name,
                    args.len(),
                    info.arity()
                ));
            }
        }
    }
    // Validate actions.
    for action in &rule.actions {
        match action {
            Action::Insert { name, args } => {
                let info = functions
                    .get(name)
                    .ok_or_else(|| anyhow!("rule {}: unknown action target {name}", rule.name))?;
                if args.len() != info.arity() {
                    return Err(anyhow!(
                        "rule {}: action target {name} has {} args, expected {}",
                        rule.name,
                        args.len(),
                        info.arity()
                    ));
                }
            }
            Action::Delete { name, key_args } => {
                let info = functions
                    .get(name)
                    .ok_or_else(|| anyhow!("rule {}: unknown action target {name}", rule.name))?;
                if key_args.len() != info.inputs_len {
                    return Err(anyhow!(
                        "rule {}: delete on {name} has {} args, expected {} (input cols)",
                        rule.name,
                        key_args.len(),
                        info.inputs_len
                    ));
                }
            }
            Action::LetCtor { name, args, .. } => {
                let info = functions
                    .get(name)
                    .ok_or_else(|| anyhow!("rule {}: unknown LetCtor target {name}", rule.name))?;
                if !info.eq_sort_ctor {
                    return Err(anyhow!(
                        "rule {}: LetCtor target {name} is not an EqSort constructor",
                        rule.name
                    ));
                }
                if args.len() != info.inputs_len {
                    return Err(anyhow!(
                        "rule {}: LetCtor {name} has {} args, expected {}",
                        rule.name,
                        args.len(),
                        info.inputs_len
                    ));
                }
            }
            Action::LetExpr { .. } => { /* no validation needed */ }
            Action::Panic { .. } => { /* no validation needed */ }
        }
    }

    // Native-UF demote plan: body atoms against `:merge ordering-min`
    // functions get rewritten as scalar UDF calls
    // (`duck_uf_<sort>_find(input)`) rather than join atoms. They
    // don't contribute to the seminaive triangulation either —
    // dropping them from `func_atom_indices` removes the variants
    // that would have focused on them. Empty `Vec<bool>` (all
    // false) when `--duck-native-uf` is off.
    let demote = compute_native_uf_demote(&rule.body, functions);

    let func_atom_indices: Vec<usize> = rule
        .body
        .iter()
        .enumerate()
        .filter_map(|(i, a)| (matches!(a, Atom::Func { .. }) && !demote[i]).then_some(i))
        .collect();

    if rule.actions.is_empty() {
        return Err(anyhow!("rule {}: at least one action required", rule.name));
    }

    // Fuse all K seminaive variants for this rule into a single
    // `CompiledVariant`. The materialize SELECT becomes a UNION ALL
    // of K branches differing only in their ts predicates; the
    // partitioning is disjoint by construction (each match has
    // exactly one "first new" atom), so no dedup is needed. Cuts
    // the per-iter statement count by ~K× without changing the
    // total join work.
    let mut variants = vec![compile_fused_variant(
        rule,
        &func_atom_indices,
        functions,
        &demote,
        None,
        proofs_enabled,
    )?];
    // For each distinct table referenced by a demoted atom, emit
    // one gated "UF-changed" recovery variant: scan the body with
    // no `>= ?1` lower bound, look up canonical roots via UDF,
    // canonicalize stale rows. The runner runs this only when the
    // gated table's watermark has advanced since this rule last
    // ran — saves work when nothing has unified.
    let mut demote_tables: Vec<String> = Vec::new();
    for (i, atom) in rule.body.iter().enumerate() {
        if !demote[i] {
            continue;
        }
        if let Atom::Func { name, .. } = atom {
            if !demote_tables.iter().any(|n| n == name) {
                demote_tables.push(name.clone());
            }
        }
    }
    for gated in &demote_tables {
        // Native-UF gating: writes that affect the demoted atom's
        // canonicalization land in the corresponding pname (raw UF)
        // table, not the function-form table itself. Derive pname
        // by the term-encoding naming convention so the runner can
        // skip the gated variant when nothing new has unified this
        // iteration.
        let gate_pname = gated.strip_suffix('f').unwrap_or(gated.as_str()).to_string();
        variants.push(compile_fused_variant(
            rule,
            &func_atom_indices,
            functions,
            &demote,
            Some(&gate_pname),
            proofs_enabled,
        )?);
    }
    Ok(CompiledRule {
        name: rule.name.clone(),
        ruleset: rule.ruleset.clone(),
        variants,
        // `add_rule` re-derives this from `rule.body` so it stays in
        // sync with the IR even if compilation rewrites the variants.
        body_tables: Vec::new(),
    })
}

/// Compile all seminaive variants of `rule` into a single
/// `CompiledVariant` by UNION-ALL-ing the per-focus branches.
///
/// Each branch shares the same FROM clause, hash-cons LEFT JOINs,
/// and SELECT projection — the branches differ only in their `ts`
/// predicates. Seminaive triangulation guarantees the partitions
/// are disjoint, so UNION ALL preserves exactly the union-of-
/// per-focus matches with no duplicates.
fn compile_fused_variant(
    rule: &Rule,
    func_atom_indices: &[usize],
    functions: &HashMap<String, FunctionInfo>,
    demote: &[bool],
    gate_table: Option<&str>,
    proofs_enabled: bool,
) -> Result<CompiledVariant> {
    // 1) Build the body's FROM/WHERE and a binding for body vars.
    let (from, base_where_parts, mut binding) =
        walk_body(&rule.body, &rule.name, functions, demote)?;
    // No Func atoms → use a 1-row dummy as FROM source.
    let from = if from.is_empty() {
        "(SELECT 1) __empty__".to_string()
    } else {
        from
    };
    let where_clause = format_where(&base_where_parts);

    // 2) Decide whether we need the materialized path.
    //
    // We materialize the body when:
    //  (a) Any action allocates a fresh ID (LetCtor) or computes a
    //      derived value (LetExpr) that subsequent actions reference.
    //  (b) The rule has multiple actions — even pure INSERT/DELETE
    //      pairs need to see the SAME body snapshot. Without
    //      materialization, an action like `(delete X)` would remove
    //      rows from the body before a subsequent `(set X')` action
    //      could read them, dropping the corresponding insertions.
    //      This is the @uf_update/@singleparent pattern: delete an
    //      old row, insert the canonicalized successor.
    let has_let = rule
        .actions
        .iter()
        .any(|a| matches!(a, Action::LetCtor { .. } | Action::LetExpr { .. }));
    let needs_snapshot = has_let || rule.actions.len() > 1;

    // The K seminaive partitions, expressed as per-focus WHERE
    // clauses (built from the shared body WHERE + per-focus ts
    // predicates). For gated variants, no `>= ?1` lower bound: the
    // variant is triggered by an external watermark and wants a
    // full scan. Empty body → one branch with no ts predicates.
    let branch_wheres: Vec<String> = if gate_table.is_some() {
        // Gated: full scan of non-demoted Func atoms with only the
        // upper-bound `< ?2` filter (avoid seeing rows inserted by
        // this same iteration).
        let mut parts = base_where_parts.clone();
        for (i, atom) in rule.body.iter().enumerate() {
            if !matches!(atom, Atom::Func { .. }) {
                continue;
            }
            if demote.get(i).copied().unwrap_or(false) {
                continue;
            }
            parts.push(format!("t{i}.ts < ?2"));
        }
        vec![format_where(&parts)]
    } else if func_atom_indices.is_empty() {
        vec![where_clause.clone()]
    } else {
        func_atom_indices
            .iter()
            .map(|&focus| {
                let mut parts = base_where_parts.clone();
                parts.extend(ts_predicates_for_focus(&rule.body, focus, demote));
                format_where(&parts)
            })
            .collect()
    };
    let _ = where_clause;

    if !needs_snapshot {
        // Single-action rule with no Let: simple INSERT/DELETE FROM
        // <body> is correct, and avoids the temp-table overhead.
        // For K-way fusion we UNION ALL the per-focus branches as
        // the source of the action; the partitioning is disjoint so
        // no rows are duplicated.
        let actions = rule
            .actions
            .iter()
            .map(|a| {
                let sql = compile_simple_action_fused(
                    a,
                    &binding,
                    &from,
                    &branch_wheres,
                    &rule.name,
                    functions,
                )?;
                Ok(CompiledAction { sql, target: action_target(a, functions) })
            })
            .collect::<Result<_>>()?;
        return Ok(CompiledVariant {
            materialize: None,
            temp_table: None,
            actions,
            gate_table: gate_table.map(str::to_string),
        });
    }

    // 3) Materialized path. Determine the columns of the temp table:
    // - one column per body-bound variable (so subsequent actions can
    //   reference body vars).
    // - one nextval column per LetCtor.
    let temp_table = format!("__m_{}", sanitize(&rule.name));

    let mut select_cols: Vec<String> = Vec::new();
    // body vars: take their stable binding location and assign an
    // alias matching the var name. Sort to make codegen
    // deterministic (binding is a HashMap).
    let mut body_vars: Vec<(String, String)> = binding
        .iter()
        .map(|(v, expr)| (v.clone(), expr.clone()))
        .collect();
    body_vars.sort_by(|a, b| a.0.cmp(&b.0));
    for (v, expr) in &body_vars {
        select_cols.push(format!("{expr} AS {}", q(v)));
    }
    // After materialization, body vars are accessed via their alias.
    let mut mat_binding: HashMap<String, String> = HashMap::new();
    for (v, _) in &body_vars {
        mat_binding.insert(v.clone(), q(v));
    }
    // Let allocations: each adds a column to the temp table.
    // LetCtor: when all args resolve to body-bound vars (i.e., they
    // don't reference an earlier LetCtor's var), hash-cons against
    // the existing view via a scalar subquery — only burn a fresh
    // sequence value when no row exists for these args. Avoids the
    // duplicate-ID-then-congruence-collapse churn that drives
    // @UF_<sort> growth. When args reference an earlier LetCtor's
    // value, we can't read sibling SELECT-list aliases in the same
    // SELECT, so fall back to plain nextval.
    // LetExpr: just an alias for an expression evaluated against
    // the body bindings.
    // For LetCtors whose args are body-resolvable, we add a LEFT
    // JOIN against a derived table that aggregates the constructor
    // view by (input cols), giving at most one existing-id row per
    // (args). The temp table's column for that LetCtor is then
    // `COALESCE(hc<i>.id, nextval(seq))` — reusing an existing id
    // when the term has been seen before, else allocating fresh.
    // This is the in-band hash-cons that avoids the duplicate-id-
    // then-congruence-collapse pattern.
    let mut hc_joins: Vec<String> = Vec::new();
    for action in &rule.actions {
        match action {
            Action::LetCtor { var, name, args } => {
                let body_resolvable = args.iter().all(|t| match t {
                    Term::Var(v) => binding.contains_key(v),
                    _ => true,
                });
                // Which table to hash-cons against. In non-proof mode
                // we use the canonicalized `@<name>View`: its id column
                // is rewritten to the e-class leader by rebuild, so
                // congruent terms fold onto one id (faster, fewer
                // rows). In proof mode that is incorrect — reusing the
                // leader gives the constructed term a *different*
                // constructor's id, so proof extraction reconstructs
                // the wrong (equal) term and the checker rejects it.
                // The bare term table (`name`) keeps each application's
                // own id, like the bridge backend, so proofs stay
                // faithful. See `EGraph::proofs_enabled`.
                let view = format!("@{name}View");
                // Only redirect to the bare term table for genuine
                // eq-sort *user* constructors — those that have a
                // `@<name>View` (whose id column the rebuild rewrites
                // to the leader). Proof-machinery constructors (AstC,
                // Rule, Trans, …) have no view and keep their existing
                // hash-cons/nextval behavior.
                let view_exists = functions.contains_key(&view);
                let hc_table = if proofs_enabled && view_exists {
                    name.clone()
                } else {
                    view.clone()
                };
                // Hash-cons is only meaningful when the hash-cons table
                // actually stores the output ID (i.e., it has
                // args.len() + 1 columns: inputs + id). For
                // constructors whose original return type was a
                // primitive (e.g., a `(function foo () i64 :merge …)`
                // gets rewritten by term encoding into a 1-arg
                // constructor backed by a 1-column @fooView relation
                // that only tracks existence, no id), there's no id
                // to look up — fall back to plain nextval.
                let view_has_id = functions
                    .get(&hc_table)
                    .map(|i| i.cols.len() > args.len())
                    .unwrap_or(false);
                // Hash-cons: look up an existing canonical id for
                // (args) before allocating, falling back to a fresh
                // sequence value when nothing matches. Saves
                // substantial work on rule-heavy workloads
                // (math-microbenchmark run-10: 34.6s/10.27 GB →
                // 17.4s/668 MB) without affecting tuple counts —
                // verified through run-11 against the reference
                // backend, and the full snapshot suite.
                let col_expr = if body_resolvable && view_has_id {
                    let arg_sqls: Vec<String> = args
                        .iter()
                        .map(|t| term_sql(t, &binding, &rule.name))
                        .collect::<Result<_>>()?;
                    let hc_alias = format!("hc{}", hc_joins.len());
                    let id_col_idx = arg_sqls.len();
                    let in_cols: Vec<String> =
                        (0..id_col_idx).map(|i| format!("c{i}")).collect();
                    if arg_sqls.is_empty() {
                        // 0-input: there's only ever one such row.
                        // Cross join with a 1-row aggregate.
                        hc_joins.push(format!(
                            "LEFT JOIN (SELECT MIN(c{id_col_idx}) AS id FROM {}) {hc_alias} ON TRUE",
                            q(&hc_table)
                        ));
                    } else {
                        let on_conds: Vec<String> = arg_sqls
                            .iter()
                            .enumerate()
                            .map(|(i, s)| format!("{hc_alias}.c{i} = {s}"))
                            .collect();
                        hc_joins.push(format!(
                            "LEFT JOIN (SELECT {}, MIN(c{id_col_idx}) AS id FROM {} GROUP BY {}) {hc_alias} ON {}",
                            in_cols.join(", "),
                            q(&hc_table),
                            in_cols.join(", "),
                            on_conds.join(" AND ")
                        ));
                    }
                    format!("COALESCE({hc_alias}.id, nextval('__egglog_eqsort_seq'))")
                } else {
                    "nextval('__egglog_eqsort_seq')".to_string()
                };
                select_cols.push(format!("{col_expr} AS {}", q(var)));
                mat_binding.insert(var.clone(), q(var));
            }
            Action::LetExpr { var, expr } => {
                // Pure aliases `(let $a @v6)` where the RHS is a
                // var already bound by an earlier LetCtor live in
                // `mat_binding`, not `binding`. SELECT-list aliases
                // can't reference each other within the same SELECT,
                // so just re-point the alias instead of emitting a
                // column.
                if let Term::Var(other) = expr {
                    if let Some(src) = mat_binding.get(other) {
                        let src = src.clone();
                        mat_binding.insert(var.clone(), src);
                        continue;
                    }
                }
                let expr_sql = term_sql(expr, &binding, &rule.name)?;
                select_cols.push(format!("{expr_sql} AS {}", q(var)));
                mat_binding.insert(var.clone(), q(var));
            }
            _ => {}
        }
    }
    let from_with_hc = if hc_joins.is_empty() {
        from.clone()
    } else {
        // SQL grammar: `t0, t1 LEFT JOIN d ON …` parses the LEFT
        // JOIN as binding to `t1` alone, so the ON clause can't see
        // `t0`. Convert comma-joins to explicit CROSS JOIN so the
        // LEFT JOIN binds to the whole left side.
        let xj = from.replace(", ", " CROSS JOIN ");
        format!("{xj} {}", hc_joins.join(" "))
    };
    // Build the materialize SQL as a UNION ALL of one branch per
    // focus. Each branch has the same SELECT projection and FROM
    // clause (including hash-cons LEFT JOINs); only the WHERE
    // changes. Seminaive partitions are disjoint by construction,
    // so UNION ALL is exact — no need for UNION/DISTINCT.
    let select_cols_str = select_cols.join(", ");
    let branches: Vec<String> = branch_wheres
        .iter()
        .map(|w| format!("SELECT {select_cols_str} FROM {from_with_hc}{w}"))
        .collect();
    let materialize = format!(
        "CREATE OR REPLACE TEMP TABLE {} AS {}",
        q(&temp_table),
        branches.join(" UNION ALL "),
    );

    // 4) Build per-action SQLs that select FROM the temp table.
    let mut action_sqls: Vec<CompiledAction> = Vec::new();
    for action in &rule.actions {
        let sql = compile_materialized_action(
            action,
            &mat_binding,
            &temp_table,
            &rule.name,
            functions,
        )?;
        action_sqls.push(CompiledAction {
            sql,
            target: action_target(action, functions),
        });
    }

    // Suppress `binding` unused warnings; it's only relevant on the
    // simple path above.
    let _ = &mut binding;

    // No explicit DROP — the next iteration's CREATE OR REPLACE
    // recycles the temp table in place. Issuing a redundant DROP
    // cost ~25% of our SQL-statement count on math-microbenchmark.
    Ok(CompiledVariant {
        materialize: Some(materialize),
        temp_table: Some(temp_table),
        actions: action_sqls,
        gate_table: gate_table.map(str::to_string),
    })
}

/// Compile a list of body atoms (as in a rule body or a `(check ...)`)
/// into a SQL `SELECT 1 FROM <body> [WHERE …] LIMIT 1` returning a
/// row iff the conjunction has any match. No seminaive focus
/// predicate; that's only for rules.
pub(crate) fn compile_body_select(
    atoms: &[Atom],
    functions: &std::collections::HashMap<String, crate::FunctionInfo>,
) -> Result<String> {
    // Validate.
    for atom in atoms {
        if let Atom::Func { name, args } = atom {
            let info = functions
                .get(name)
                .ok_or_else(|| anyhow!("body atom: unknown function {name}"))?;
            if args.len() != info.arity() {
                return Err(anyhow!(
                    "body atom {name} has {} args, expected {}",
                    args.len(),
                    info.arity()
                ));
            }
        }
    }
    // `(check ...)` runs against the current state; no native-UF
    // demote applies (we want the literal join, not a UDF lookup
    // bypass) so all-false `demote`.
    let no_demote = vec![false; atoms.len()];
    let (from, where_parts, _binding) =
        walk_body(atoms, "<check>", functions, &no_demote)?;
    let where_clause = format_where(&where_parts);
    if from.is_empty() {
        // No function atoms: a pure-primitive `(check (= 1 1))`-style
        // assertion. Evaluate the WHERE clause as a constant.
        return Ok(format!(
            "SELECT COUNT(*) FROM (SELECT 1{}) LIMIT 1",
            where_clause
        ));
    }
    Ok(format!(
        "SELECT COUNT(*) FROM (SELECT 1 FROM {from}{where_clause} LIMIT 1)"
    ))
}

/// Shared body walker used by both rule compilation (via
/// `build_query`) and check compilation (via `compile_body_select`).
/// Try to process a single body atom. Returns `Ok(true)` if the
/// atom was fully processed and contributed to `binding` /
/// `from_parts` / `where_parts`; `Ok(false)` if it depends on a var
/// that hasn't been bound yet and should be retried later. The Err
/// path is reserved for genuine compilation errors (e.g., unbound
/// var inside a Prim subterm).
fn try_walk_atom(
    atom: &Atom,
    i: usize,
    binding: &mut HashMap<String, String>,
    from_parts: &mut Vec<String>,
    where_parts: &mut Vec<String>,
    rule_name: &str,
    functions: &HashMap<String, FunctionInfo>,
    demoted: bool,
) -> Result<bool> {
    match atom {
        Atom::Func { name, args } => {
            if demoted {
                // Native-UF demote: emit a scalar UDF call binding
                // the output var instead of adding a join atom. The
                // compute_native_uf_demote pre-pass already verified
                // shape (inputs bound, single fresh output Var), but
                // the worklist may not have processed the dependency
                // yet on this pass — defer if any input var is still
                // unbound.
                let info = &functions[name];
                if !args[..info.inputs_len].iter().all(|t| match t {
                    Term::Var(v) => binding.contains_key(v),
                    _ => true,
                }) {
                    return Ok(false);
                }
                let udf = info
                    .native_uf_udf
                    .as_ref()
                    .expect("demoted atom must have native_uf_udf set");
                let arg_sqls: Vec<String> = args[..info.inputs_len]
                    .iter()
                    .map(|t| term_sql(t, binding, rule_name))
                    .collect::<Result<_>>()?;
                let call = format!("{}({})", udf, arg_sqls.join(", "));
                if let Term::Var(v) = &args[info.inputs_len] {
                    binding.insert(v.clone(), call);
                }
                return Ok(true);
            }
            let alias = format!("t{i}");
            from_parts.push(format!("{} {alias}", q(name)));
            for (col, term) in args.iter().enumerate() {
                let lhs = format!("{alias}.c{col}");
                match term {
                    Term::Var(v) => match binding.get(v) {
                        None => {
                            binding.insert(v.clone(), lhs);
                        }
                        Some(prev) => {
                            where_parts.push(format!("{prev} = {lhs}"));
                        }
                    },
                    Term::Lit(_) | Term::Prim(_, _) | Term::FuncCall { .. } => {
                        let rhs = term_sql(term, binding, rule_name)?;
                        where_parts.push(format!("{lhs} = {rhs}"));
                    }
                }
            }
            Ok(true)
        }
        Atom::Filter(t) => {
            if !term_vars_all_bound(t, binding) {
                return Ok(false);
            }
            where_parts.push(term_sql(t, binding, rule_name)?);
            Ok(true)
        }
        Atom::Bind { var, expr } => {
            // Symmetric handling for Var-Var binds: if one side is
            // already bound and the other isn't, propagate the
            // binding to the unbound side. If neither is bound,
            // defer (a later Func atom probably binds one of them).
            if let Term::Var(other) = expr {
                let var_bound = binding.contains_key(var);
                let other_bound = binding.contains_key(other);
                match (var_bound, other_bound) {
                    (true, true) => {
                        let l = binding[var].clone();
                        let r = binding[other].clone();
                        where_parts.push(format!("{l} = {r}"));
                        Ok(true)
                    }
                    (true, false) => {
                        let l = binding[var].clone();
                        binding.insert(other.clone(), l);
                        Ok(true)
                    }
                    (false, true) => {
                        let r = binding[other].clone();
                        binding.insert(var.clone(), r);
                        Ok(true)
                    }
                    (false, false) => Ok(false),
                }
            } else {
                if !term_vars_all_bound(expr, binding) {
                    return Ok(false);
                }
                let s = term_sql(expr, binding, rule_name)?;
                match binding.get(var) {
                    None => {
                        binding.insert(var.clone(), s);
                    }
                    Some(prev) => {
                        where_parts.push(format!("{prev} = {s}"));
                    }
                }
                Ok(true)
            }
        }
    }
}

/// Check whether every `Term::Var` referenced inside `t` (directly
/// or transitively through Prim/FuncCall children) is already
/// bound. Used to decide whether a Filter/Bind atom is ready to
/// process or should be deferred.
fn term_vars_all_bound(t: &Term, binding: &HashMap<String, String>) -> bool {
    match t {
        Term::Var(v) => binding.contains_key(v),
        Term::Lit(_) => true,
        Term::Prim(_, args) | Term::FuncCall { args, .. } => {
            args.iter().all(|a| term_vars_all_bound(a, binding))
        }
    }
}

/// Compute the seminaive ts predicates for a single focus choice.
/// Returned strings reference table aliases `t{i}` for each Func
/// atom; they're meant to be AND-combined with the body's own WHERE
/// parts (which `walk_body` builds and returns unbounded).
/// Demoted atoms (those that become scalar UDF calls) don't have a
/// `tN` alias and are skipped — caller passes `demote` matching the
/// per-atom plan used by `walk_body`.
fn ts_predicates_for_focus(atoms: &[Atom], focus_idx: usize, demote: &[bool]) -> Vec<String> {
    let mut parts = Vec::new();
    for (i, atom) in atoms.iter().enumerate() {
        if !matches!(atom, Atom::Func { .. }) {
            continue;
        }
        if demote.get(i).copied().unwrap_or(false) {
            continue;
        }
        match i.cmp(&focus_idx) {
            std::cmp::Ordering::Equal => {
                parts.push(format!("t{i}.ts >= ?1"));
                parts.push(format!("t{i}.ts < ?2"));
            }
            std::cmp::Ordering::Less => {
                parts.push(format!("t{i}.ts < ?1"));
            }
            std::cmp::Ordering::Greater => {
                parts.push(format!("t{i}.ts < ?2"));
            }
        }
    }
    parts
}

fn walk_body(
    atoms: &[Atom],
    rule_name: &str,
    functions: &HashMap<String, FunctionInfo>,
    demote: &[bool],
) -> Result<(String, Vec<String>, HashMap<String, String>)> {
    let mut binding: HashMap<String, String> = HashMap::new();
    let mut from_parts: Vec<String> = Vec::new();
    let mut where_parts: Vec<String> = Vec::new();

    // Term encoding can emit Bind atoms whose vars are introduced
    // by *later* Func atoms (e.g. `(rewrite p $True :when …)` →
    // body atom 0 is `(= @rewrite_var__ p)` where `p` is first
    // bound by atom 3). Defer any atom we can't process yet and
    // retry until we converge. Func atoms always process first
    // pass since they introduce their own bindings unconditionally.
    let mut pending: Vec<usize> = (0..atoms.len()).collect();
    loop {
        let mut made_progress = false;
        let mut still_pending: Vec<usize> = Vec::new();
        for i in pending {
            let atom = &atoms[i];
            let ok = try_walk_atom(
                atom,
                i,
                &mut binding,
                &mut from_parts,
                &mut where_parts,
                rule_name,
                functions,
                demote.get(i).copied().unwrap_or(false),
            )?;
            if ok {
                made_progress = true;
            } else {
                still_pending.push(i);
            }
        }
        if still_pending.is_empty() {
            break;
        }
        if !made_progress {
            // Pick the first unsatisfied atom and error with a
            // descriptive message rather than looping forever.
            let i = still_pending[0];
            match &atoms[i] {
                Atom::Bind { var, expr } => {
                    return Err(anyhow!(
                        "rule {rule_name}: cannot resolve Bind {var} = {expr:?}: dependent vars never get bound"
                    ));
                }
                Atom::Filter(t) => {
                    return Err(anyhow!(
                        "rule {rule_name}: cannot resolve Filter {t:?}: dependent vars never get bound"
                    ));
                }
                Atom::Func { name, .. } => {
                    return Err(anyhow!(
                        "rule {rule_name}: cannot resolve Func {name}: internal error"
                    ));
                }
            }
        }
        pending = still_pending;
    }
    let from = from_parts.join(", ");
    Ok((from, where_parts, binding))
}

/// Compile one rule action under the *simple* path, fused across
/// all K seminaive partitions. Each `branch_wheres` entry is the
/// fully-formed WHERE clause (including the per-focus ts
/// predicates) for one partition; the action's source SELECT is
/// the UNION ALL of one branch per entry.
fn compile_simple_action_fused(
    action: &Action,
    binding: &HashMap<String, String>,
    from: &str,
    branch_wheres: &[String],
    rule_name: &str,
    functions: &HashMap<String, FunctionInfo>,
) -> Result<String> {
    let union_branches = |body: &str| -> String {
        branch_wheres
            .iter()
            .map(|w| format!("{body}{w}"))
            .collect::<Vec<_>>()
            .join(" UNION ALL ")
    };
    match action {
        Action::Insert {
            name: target,
            args: targs,
        } => {
            let info = &functions[target];
            let select_cols: Vec<String> = targs
                .iter()
                .map(|t| term_sql(t, binding, rule_name))
                .collect::<Result<_>>()?;
            let target_cols: Vec<String> =
                (0..targs.len()).map(|i| format!("c{i}")).collect();
            let select_list = format!("{}?2", prefix_with_comma(&select_cols));
            let insert_cols = format!("{}ts", prefix_with_comma(&target_cols));
            let conflict = conflict_clause(info);
            let body = format!("SELECT {select_list} FROM {from}");
            Ok(format!(
                "INSERT INTO {} ({insert_cols}) {} {conflict}",
                q(target),
                union_branches(&body),
            ))
        }
        Action::Delete {
            name: target,
            key_args,
        } => {
            // 0-key deletes: SQL `()` and an empty SELECT are
            // invalid. Use `EXISTS (SELECT 1 FROM body)` instead —
            // delete every row of the target iff the body matches.
            if key_args.is_empty() {
                let body = format!("SELECT 1 FROM {from}");
                return Ok(format!(
                    "DELETE FROM {} WHERE EXISTS ({})",
                    q(target),
                    union_branches(&body),
                ));
            }
            let key_cols: Vec<String> =
                (0..key_args.len()).map(|i| format!("c{i}")).collect();
            let key_select: Vec<String> = key_args
                .iter()
                .map(|t| term_sql(t, binding, rule_name))
                .collect::<Result<_>>()?;
            let body = format!("SELECT {} FROM {from}", key_select.join(", "));
            Ok(format!(
                "DELETE FROM {} WHERE ({}) IN ({})",
                q(target),
                key_cols.join(", "),
                union_branches(&body),
            ))
        }
        Action::LetCtor { .. } | Action::LetExpr { .. } => {
            // Unreachable: presence of any Let switches us to the
            // materialized path before we reach this point.
            Err(anyhow!(
                "rule {rule_name}: internal: Let reached simple-action codegen"
            ))
        }
        Action::Panic { msg } => {
            // `error()` is only evaluated on returned rows, so 0 rows
            // = no error, ≥1 row = SQL exception.
            let safe = msg.replace('\'', "''");
            let body = format!("SELECT 1 FROM {from}");
            Ok(format!(
                "SELECT error('panic: {safe}') FROM ({}) LIMIT 1",
                union_branches(&body),
            ))
        }
    }
}

/// Compile a single action under the *materialized* path: INSERT/
/// DELETE FROM the temp table.
fn compile_materialized_action(
    action: &Action,
    mat_binding: &HashMap<String, String>,
    temp_table: &str,
    rule_name: &str,
    functions: &HashMap<String, FunctionInfo>,
) -> Result<String> {
    let from = q(temp_table);
    match action {
        Action::Insert {
            name: target,
            args: targs,
        } => {
            let info = &functions[target];
            let select_cols: Vec<String> = targs
                .iter()
                .map(|t| term_sql(t, mat_binding, rule_name))
                .collect::<Result<_>>()?;
            let target_cols: Vec<String> =
                (0..targs.len()).map(|i| format!("c{i}")).collect();
            let select_list = format!("{}?2", prefix_with_comma(&select_cols));
            let insert_cols = format!("{}ts", prefix_with_comma(&target_cols));
            let conflict = conflict_clause(info);
            Ok(format!(
                "INSERT INTO {} ({insert_cols}) SELECT {select_list} FROM {from} {conflict}",
                q(target),
            ))
        }
        Action::Delete {
            name: target,
            key_args,
        } => {
            if key_args.is_empty() {
                return Ok(format!(
                    "DELETE FROM {} WHERE EXISTS (SELECT 1 FROM {from})",
                    q(target)
                ));
            }
            let key_select: Vec<String> = key_args
                .iter()
                .map(|t| term_sql(t, mat_binding, rule_name))
                .collect::<Result<_>>()?;
            // Reshape `DELETE WHERE (cols) IN (SELECT … FROM temp)`
            // into `DELETE FROM target USING (SELECT …) __d WHERE
            // target.cN = __d.dN`. DuckDB's tuple-IN subquery
            // sometimes falls back to a full scan; the USING-join
            // form gives the optimizer a clear hash-join shape and
            // lets it pick the PK index on `target` for lookup.
            let proj_cols: Vec<String> = key_select
                .iter()
                .enumerate()
                .map(|(i, expr)| format!("{expr} AS d{i}"))
                .collect();
            let join_preds: Vec<String> = (0..key_select.len())
                .map(|i| format!("{}.c{i} = __d.d{i}", q(target)))
                .collect();
            Ok(format!(
                "DELETE FROM {} USING (SELECT {} FROM {from}) __d WHERE {}",
                q(target),
                proj_cols.join(", "),
                join_preds.join(" AND "),
            ))
        }
        Action::LetExpr { .. } => {
            // The expression's value is already in the temp table
            // column corresponding to `var`; nothing to insert.
            // Return a no-op SELECT so we have *something* in the
            // action list (the runner expects a SQL string).
            Ok("SELECT 1 WHERE FALSE".to_string())
        }
        Action::LetCtor { var, name, args } => {
            // Insert the (args…, fresh_id, ts) row into the
            // constructor's own table so it mirrors the bridge
            // backend's `TableAction::lookup` with `FreshId`
            // default. The fresh ID is already in the
            // materialized temp table's `var` column (via
            // `nextval`/hash-cons COALESCE); just project it
            // through.
            //
            // Previously this branch skipped the INSERT whenever
            // `@<name>View` existed (relying on the accompanying
            // `Insert @<name>View` to emit the canonical row).
            // That made the raw `<name>` table go unwritten, which
            // broke bridge-compatible callers that walk the
            // constructor table by name (`prove_exists`, the
            // extractor, etc.). Now that strings are stored as
            // interned `Value` handles (BIGINT) — matching the
            // bridge — `for_each_while` round-trips correctly, so
            // it's safe to populate both tables.
            let info = functions
                .get(name)
                .ok_or_else(|| anyhow!("rule {rule_name}: unknown LetCtor target {name}"))?;
            let arg_sqls: Vec<String> = args
                .iter()
                .map(|t| term_sql(t, mat_binding, rule_name))
                .collect::<Result<_>>()?;
            // Defensive: caller arity must equal `inputs_len`.
            if arg_sqls.len() != info.inputs_len {
                return Err(anyhow!(
                    "rule {rule_name}: LetCtor {name} arity {} doesn't match inputs_len {}",
                    arg_sqls.len(),
                    info.inputs_len
                ));
            }
            let target_cols: Vec<String> =
                (0..info.cols.len()).map(|i| format!("c{i}")).collect();
            let select_list = format!("{}{}, ?2", prefix_with_comma(&arg_sqls), q(var));
            let conflict = conflict_clause(info);
            Ok(format!(
                "INSERT INTO {} ({}, ts) SELECT {select_list} FROM {from} {conflict}",
                q(name),
                target_cols.join(", "),
            ))
        }
        Action::Panic { msg } => {
            let safe = msg.replace('\'', "''");
            Ok(format!(
                "SELECT error('panic: {safe}') FROM {from} LIMIT 1"
            ))
        }
    }
}

/// Compile a `Term` to SQL with an empty variable binding.
pub(crate) fn term_sql_no_binding(t: &Term, ctx: &str) -> Result<String> {
    let empty = HashMap::new();
    term_sql(t, &empty, ctx)
}

fn term_sql(t: &Term, binding: &HashMap<String, String>, rule_name: &str) -> Result<String> {
    match t {
        Term::Var(v) => binding
            .get(v)
            .cloned()
            .ok_or_else(|| anyhow!("rule {rule_name}: unbound variable {v}")),
        Term::Lit(l) => Ok(lit_sql(l)),
        Term::Prim(op, args) => {
            let arg_sqls: Vec<String> = args
                .iter()
                .map(|a| term_sql(a, binding, rule_name))
                .collect::<Result<_>>()?;
            prim_sql(op, &arg_sqls, rule_name)
        }
        Term::FuncCall { name, args } => {
            let arg_sqls: Vec<String> = args
                .iter()
                .map(|a| term_sql(a, binding, rule_name))
                .collect::<Result<_>>()?;
            let out_col = args.len();
            let where_clause = if arg_sqls.is_empty() {
                String::new()
            } else {
                let conjs: Vec<String> = arg_sqls
                    .iter()
                    .enumerate()
                    .map(|(i, s)| format!("c{i} = {s}"))
                    .collect();
                format!(" WHERE {}", conjs.join(" AND "))
            };
            Ok(format!(
                "(SELECT c{out_col} FROM {}{where_clause} LIMIT 1)",
                q(name)
            ))
        }
    }
}

fn prim_sql(op: &str, args: &[String], rule_name: &str) -> Result<String> {
    let binop = |sql_op: &str| -> Result<String> {
        if args.len() != 2 {
            return Err(anyhow!(
                "rule {rule_name}: primitive `{op}` expects 2 args, got {}",
                args.len()
            ));
        }
        Ok(format!("({} {sql_op} {})", args[0], args[1]))
    };
    let unop = |sql_op: &str| -> Result<String> {
        if args.len() != 1 {
            return Err(anyhow!(
                "rule {rule_name}: primitive `{op}` expects 1 arg, got {}",
                args.len()
            ));
        }
        Ok(format!("({sql_op} {})", args[0]))
    };
    let func = |sql_func: &str| -> Result<String> {
        Ok(format!("{sql_func}({})", args.join(", ")))
    };
    let variadic_join = |sql_op: &str, identity: &str| -> Result<String> {
        if args.is_empty() {
            return Ok(format!("({identity})"));
        }
        Ok(format!("({})", args.join(&format!(" {sql_op} "))))
    };
    match op {
        "+" | "-" | "*" | "/" => binop(op),
        "int-div" => binop("//"),
        // Bitwise integer ops. DuckDB uses & | for AND/OR but `^` is
        // exponentiation — for f64 `^` (power) we let it through; for
        // i64 `^` (XOR) the frontend rewrites to `i64-xor`.
        "&" | "|" => binop(op),
        "^" => binop(op),
        "i64-xor" => {
            if args.len() != 2 {
                return Err(anyhow!(
                    "rule {rule_name}: primitive `i64-xor` expects 2 args, got {}",
                    args.len()
                ));
            }
            Ok(format!("xor({}, {})", args[0], args[1]))
        }
        "<<" | ">>" => binop(op),
        "<" | "<=" | ">" | ">=" => binop(op),
        "=" => binop("="),
        "!=" | "<>" | "bool-!=" => binop("<>"),
        "and" => binop("AND"),
        "or" => variadic_join("OR", "FALSE"),
        "not" => unop("NOT"),
        "ordering-max" | "max" => func("GREATEST"),
        "ordering-min" | "min" => func("LEAST"),
        "abs" => func("ABS"),
        "%" => binop("%"),
        "neg" => unop("-"),
        "not-i64" => unop("~"),
        "to-f64" => {
            if args.len() != 1 {
                return Err(anyhow!(
                    "rule {rule_name}: primitive `to-f64` expects 1 arg, got {}",
                    args.len()
                ));
            }
            Ok(format!("CAST({} AS DOUBLE)", args[0]))
        }
        "to-i64" => {
            if args.len() != 1 {
                return Err(anyhow!(
                    "rule {rule_name}: primitive `to-i64` expects 1 arg, got {}",
                    args.len()
                ));
            }
            Ok(format!("CAST({} AS BIGINT)", args[0]))
        }
        // The frontend (`rename_prim` in `lib.rs`) routes overloads
        // of `to-string` to sort-specific names so the SQL emitter
        // can pick the right UDF:
        //   i64-to-string    → __egglog_i64_to_string
        //   f64-to-string    → __egglog_f64_to_string
        //   bigint-to-string → __egglog_bigint_to_string
        // Strings now live as interned `Value` handles in BIGINT
        // columns, so we route to UDFs that take the input, format
        // it, and intern the result string. The original
        // `CAST(... AS VARCHAR)` path no longer works because
        // downstream consumers expect a BIGINT handle, not a raw
        // VARCHAR.
        "i64-to-string" => {
            if args.len() != 1 {
                return Err(anyhow!(
                    "rule {rule_name}: primitive `i64-to-string` expects 1 arg, got {}",
                    args.len()
                ));
            }
            Ok(format!("__egglog_i64_to_string({})", args[0]))
        }
        "f64-to-string" => {
            if args.len() != 1 {
                return Err(anyhow!(
                    "rule {rule_name}: primitive `f64-to-string` expects 1 arg, got {}",
                    args.len()
                ));
            }
            Ok(format!("__egglog_f64_to_string({})", args[0]))
        }
        "bigint-to-string" => {
            if args.len() != 1 {
                return Err(anyhow!(
                    "rule {rule_name}: primitive `bigint-to-string` expects 1 arg, got {}",
                    args.len()
                ));
            }
            Ok(format!("__egglog_bigint_to_string({})", args[0]))
        }
        // String-concat is variadic on the egglog side; the duck
        // UDF mirrors the signature with a variadic BIGINT arg list.
        // Empty concat is well-defined: emit a literal for the
        // empty string (interned at compile time would be cleaner,
        // but the 0-arg form is rare enough that we leave the UDF
        // to handle it).
        "string-concat" => {
            if args.is_empty() {
                return Err(anyhow!(
                    "rule {rule_name}: empty `string-concat` not supported on duckdb"
                ));
            }
            Ok(format!("__egglog_string_concat({})", args.join(", ")))
        }
        "replace" => {
            if args.len() != 3 {
                return Err(anyhow!(
                    "rule {rule_name}: primitive `replace` expects 3 args, got {}",
                    args.len()
                ));
            }
            Ok(format!(
                "__egglog_replace({}, {}, {})",
                args[0], args[1], args[2]
            ))
        }
        "count-matches" => {
            if args.len() != 2 {
                return Err(anyhow!(
                    "rule {rule_name}: primitive `count-matches` expects 2 args, got {}",
                    args.len()
                ));
            }
            Ok(format!("__egglog_count_matches({}, {})", args[0], args[1]))
        }
        "guard" => unop(""),
        // Pair container primitives. The term encoder uses these in
        // proof mode: `uf_function_<sort>` returns a
        // `(Pair sort proof)`, and rebuild rules read out the
        // leader with `(pair-first p)`. The bridge represents Pair
        // as a DuckDB STRUCT with `first` / `second` BIGINT fields.
        "pair" => {
            if args.len() != 2 {
                return Err(anyhow!(
                    "rule {rule_name}: primitive `pair` expects 2 args, got {}",
                    args.len()
                ));
            }
            // `struct_pack` keeps the field names attached so later
            // `.first` / `.second` access on the result is just a
            // typed field reference and DuckDB's planner can fold it.
            Ok(format!(
                "struct_pack(first := {}, second := {})",
                args[0], args[1]
            ))
        }
        "pair-first" => {
            if args.len() != 1 {
                return Err(anyhow!(
                    "rule {rule_name}: primitive `pair-first` expects 1 arg, got {}",
                    args.len()
                ));
            }
            Ok(format!("({}).first", args[0]))
        }
        "pair-second" => {
            if args.len() != 1 {
                return Err(anyhow!(
                    "rule {rule_name}: primitive `pair-second` expects 1 arg, got {}",
                    args.len()
                ));
            }
            Ok(format!("({}).second", args[0]))
        }
        // BigInt / BigRat constructors are routed to DuckDB scalar
        // UDFs registered by `EGraph::register_builtin_prim_udf` (see
        // `lib.rs`). The UDFs call into the shared base-value pool to
        // intern the parsed/constructed value and return its `Value`
        // rep as a BIGINT handle.
        "from-string" => {
            if args.len() != 1 {
                return Err(anyhow!(
                    "rule {rule_name}: primitive `from-string` expects 1 arg, got {}",
                    args.len()
                ));
            }
            Ok(format!("__egglog_from_string({})", args[0]))
        }
        "bigrat" => {
            if args.len() != 2 {
                return Err(anyhow!(
                    "rule {rule_name}: primitive `bigrat` expects 2 args, got {}",
                    args.len()
                ));
            }
            Ok(format!("__egglog_bigrat({}, {})", args[0], args[1]))
        }
        // egglog-experimental's `get-size!` zero-arg form. The
        // variadic table-filter form is not yet supported on duckdb;
        // `EGraph::refresh_table_sizes` must be called before each
        // query that may reference this UDF so it sees a fresh
        // snapshot.
        "get-size!" => {
            if !args.is_empty() {
                return Err(anyhow!(
                    "rule {rule_name}: variadic `get-size!` arg form is unsupported on duckdb \
                     (only the zero-arg total form works)"
                ));
            }
            Ok("__egglog_get_size()".to_string())
        }
        // BigRat-specific aliases — the frontend renames the
        // BigRat-overloaded `+`, `<`, `round`, ... call sites to
        // `bigrat-add`, `bigrat-lt`, `bigrat-round`, ... so we route
        // them to the per-op UDFs registered by
        // `register_builtin_prim_udf` (see `lib.rs`). The duck UDF
        // name uses `_` instead of `-` to keep it as a valid SQL
        // identifier — same convention as `sanitize_for_udf`.
        s if s.starts_with("bigrat-") => {
            let udf = format!("__egglog_{}", s.replace('-', "_"));
            Ok(format!("{udf}({})", args.join(", ")))
        }
        _ => Err(anyhow!("rule {rule_name}: unknown primitive `{op}`")),
    }
}

/// Public version of lit_sql for use by the bridge's seed inserts.
pub(crate) fn lit_sql_pub(l: &crate::Literal) -> String {
    lit_sql(l)
}

fn lit_sql(l: &crate::Literal) -> String {
    match l {
        crate::Literal::I64(i) => i.to_string(),
        crate::Literal::Bool(b) => if *b { "TRUE" } else { "FALSE" }.to_string(),
        crate::Literal::F64(f) => format!("CAST({} AS DOUBLE)", f),
        crate::Literal::Str(s) => format!("'{}'", s.replace('\'', "''")),
    }
}

/// Make a string safe to embed in a SQL identifier (temp table name).
fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}
