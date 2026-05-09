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
    Action, Atom, CompiledRule, CompiledVariant, FunctionInfo, Rule, Term, conflict_clause,
    prefix_with_comma, q,
};

pub(crate) fn compile_rule(
    rule: &Rule,
    functions: &HashMap<String, FunctionInfo>,
) -> Result<CompiledRule> {
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
        }
    }

    let func_atom_indices: Vec<usize> = rule
        .body
        .iter()
        .enumerate()
        .filter_map(|(i, a)| matches!(a, Atom::Func { .. }).then_some(i))
        .collect();

    if func_atom_indices.is_empty() {
        return Err(anyhow!(
            "rule {}: at least one function-table atom required",
            rule.name
        ));
    }
    if rule.actions.is_empty() {
        return Err(anyhow!("rule {}: at least one action required", rule.name));
    }

    let mut variants = Vec::with_capacity(func_atom_indices.len());
    for (variant_idx, &focus) in func_atom_indices.iter().enumerate() {
        variants.push(compile_variant(rule, variant_idx, focus, functions)?);
    }
    Ok(CompiledRule {
        name: rule.name.clone(),
        ruleset: rule.ruleset.clone(),
        variants,
    })
}

fn compile_variant(
    rule: &Rule,
    variant_idx: usize,
    focus: usize,
    functions: &HashMap<String, FunctionInfo>,
) -> Result<CompiledVariant> {
    // 1) Build the body's FROM/WHERE and a binding for body vars.
    let (from, where_clause, mut binding) = build_query(rule, focus)?;

    // 2) Decide whether we need the materialized path.
    let has_let = rule
        .actions
        .iter()
        .any(|a| matches!(a, Action::LetCtor { .. } | Action::LetExpr { .. }));

    if !has_let {
        // Simple path: each action becomes its own INSERT/DELETE
        // FROM <body>.
        let actions = rule
            .actions
            .iter()
            .map(|a| compile_simple_action(a, &binding, &from, &where_clause, &rule.name, functions))
            .collect::<Result<_>>()?;
        return Ok(CompiledVariant {
            materialize: None,
            actions,
        });
    }

    // 3) Materialized path. Determine the columns of the temp table:
    // - one column per body-bound variable (so subsequent actions can
    //   reference body vars).
    // - one nextval column per LetCtor.
    let temp_table = format!(
        "__m_{}_{}",
        sanitize(&rule.name),
        variant_idx + 1
    );

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
    // LetCtor uses nextval() (and the action issues an INSERT to
    // the constructor table at run-time); LetExpr is just an alias
    // for an expression evaluated against the body bindings.
    for action in &rule.actions {
        match action {
            Action::LetCtor { var, .. } => {
                select_cols.push(format!(
                    "nextval('__egglog_eqsort_seq') AS {}",
                    q(var)
                ));
                mat_binding.insert(var.clone(), q(var));
            }
            Action::LetExpr { var, expr } => {
                let expr_sql = term_sql(expr, &binding, &rule.name)?;
                select_cols.push(format!("{expr_sql} AS {}", q(var)));
                mat_binding.insert(var.clone(), q(var));
            }
            _ => {}
        }
    }
    // Build the materialize SQL.
    let materialize = format!(
        "CREATE OR REPLACE TEMP TABLE {} AS SELECT {} FROM {}{}",
        q(&temp_table),
        select_cols.join(", "),
        from,
        where_clause,
    );

    // 4) Build per-action SQLs that select FROM the temp table.
    let mut action_sqls: Vec<String> = Vec::new();
    for action in &rule.actions {
        action_sqls.push(compile_materialized_action(
            action,
            &mat_binding,
            &temp_table,
            &rule.name,
            functions,
        )?);
    }

    // Suppress `binding` unused warnings; it's only relevant on the
    // simple path above.
    let _ = &mut binding;

    Ok(CompiledVariant {
        materialize: Some(materialize),
        actions: action_sqls,
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
    let (from, where_clause, _binding) = walk_body(atoms, None, "<check>")?;
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
fn walk_body(
    atoms: &[Atom],
    focus: Option<usize>,
    rule_name: &str,
) -> Result<(String, String, HashMap<String, String>)> {
    let mut binding: HashMap<String, String> = HashMap::new();
    let mut from_parts: Vec<String> = Vec::new();
    let mut where_parts: Vec<String> = Vec::new();

    for (i, atom) in atoms.iter().enumerate() {
        match atom {
            Atom::Func { name, args } => {
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
                            let rhs = term_sql(term, &binding, rule_name)?;
                            where_parts.push(format!("{lhs} = {rhs}"));
                        }
                    }
                }
            }
            Atom::Filter(t) => {
                where_parts.push(term_sql(t, &binding, rule_name)?);
            }
            Atom::Bind { var, expr } => {
                let s = term_sql(expr, &binding, rule_name)?;
                binding.insert(var.clone(), s);
            }
        }
    }
    if let Some(focus_idx) = focus {
        where_parts.push(format!("t{focus_idx}.ts >= ?1"));
    }
    let from = from_parts.join(", ");
    let where_clause = if where_parts.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", where_parts.join(" AND "))
    };
    Ok((from, where_clause, binding))
}

fn build_query(
    rule: &Rule,
    focus: usize,
) -> Result<(String, String, HashMap<String, String>)> {
    walk_body(&rule.body, Some(focus), &rule.name)
}

/// Compile a single action under the *simple* path: INSERT/DELETE
/// FROM <body>.
fn compile_simple_action(
    action: &Action,
    binding: &HashMap<String, String>,
    from: &str,
    where_clause: &str,
    rule_name: &str,
    functions: &HashMap<String, FunctionInfo>,
) -> Result<String> {
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
            Ok(format!(
                "INSERT INTO {} ({insert_cols}) SELECT {select_list} FROM {from}{where_clause} {conflict}",
                q(target),
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
                return Ok(format!(
                    "DELETE FROM {} WHERE EXISTS (SELECT 1 FROM {from}{where_clause})",
                    q(target)
                ));
            }
            let key_cols: Vec<String> =
                (0..key_args.len()).map(|i| format!("c{i}")).collect();
            let key_select: Vec<String> = key_args
                .iter()
                .map(|t| term_sql(t, binding, rule_name))
                .collect::<Result<_>>()?;
            Ok(format!(
                "DELETE FROM {} WHERE ({}) IN (SELECT {} FROM {from}{where_clause})",
                q(target),
                key_cols.join(", "),
                key_select.join(", "),
            ))
        }
        Action::LetCtor { .. } | Action::LetExpr { .. } => {
            // Unreachable: presence of any Let switches us to the
            // materialized path before we reach this point.
            Err(anyhow!(
                "rule {rule_name}: internal: Let reached simple-action codegen"
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
            let select_list = format!("{}?1", prefix_with_comma(&select_cols));
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
            let key_cols: Vec<String> =
                (0..key_args.len()).map(|i| format!("c{i}")).collect();
            let key_select: Vec<String> = key_args
                .iter()
                .map(|t| term_sql(t, mat_binding, rule_name))
                .collect::<Result<_>>()?;
            Ok(format!(
                "DELETE FROM {} WHERE ({}) IN (SELECT {} FROM {from})",
                q(target),
                key_cols.join(", "),
                key_select.join(", "),
            ))
        }
        Action::LetExpr { .. } => {
            // The expression's value is already in the temp table
            // column corresponding to `var`; nothing to insert.
            // Return a no-op SELECT so we have *something* in the
            // action list (the runner expects a SQL string).
            Ok("SELECT 1 WHERE FALSE".to_string())
        }
        Action::LetCtor {
            var,
            name: target,
            args,
        } => {
            // Insert into the constructor table using the
            // pre-allocated id from the temp table column.
            let info = &functions[target];
            let arg_sqls: Vec<String> = args
                .iter()
                .map(|t| term_sql(t, mat_binding, rule_name))
                .collect::<Result<_>>()?;
            // The let var was given a column in the temp table.
            let id_col = mat_binding
                .get(var)
                .cloned()
                .ok_or_else(|| anyhow!("rule {rule_name}: internal: LetCtor var {var} not in mat binding"))?;
            let target_cols: Vec<String> =
                (0..info.cols.len()).map(|i| format!("c{i}")).collect();
            let select_list = format!(
                "{}{id_col}, ?1",
                prefix_with_comma(&arg_sqls)
            );
            let insert_cols = format!("{}ts", prefix_with_comma(&target_cols));
            let conflict = conflict_clause(info);
            Ok(format!(
                "INSERT INTO {} ({insert_cols}) SELECT {select_list} FROM {from} {conflict}",
                q(target),
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
        // Bitwise integer ops (DuckDB uses C-style operators).
        "&" | "|" | "^" => binop(op),
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
        "guard" => unop(""),
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
