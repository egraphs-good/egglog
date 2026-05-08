//! Compile a high-level `Rule` into one SQL string per (variant ×
//! action).
//!
//! A rule with N function-table atoms produces N variants. Variant i
//! adds `WHERE alias_i.ts >= ?1` on the focused atom and reads the
//! current epoch as `?2`. Both come in as bind parameters at run
//! time. Each variant runs all its actions in order on every match.

use anyhow::{Result, anyhow};
use std::collections::HashMap;

use crate::{Action, Atom, CompiledRule, FunctionInfo, Rule, Term, conflict_clause};

pub(crate) fn compile_rule(
    rule: &Rule,
    functions: &HashMap<String, FunctionInfo>,
) -> Result<CompiledRule> {
    // Validate function-table atoms (filters are typed implicitly
    // via SQL).
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
    for action in &rule.actions {
        let Action::Insert { name, args } = action;
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

    // Count function-table atoms — each gets a seminaive variant.
    // Filter atoms don't get variants (they're not "fresh source"
    // candidates for new matches).
    let func_atom_indices: Vec<usize> = rule
        .body
        .iter()
        .enumerate()
        .filter_map(|(i, a)| matches!(a, Atom::Func { .. }).then_some(i))
        .collect();

    if func_atom_indices.is_empty() {
        return Err(anyhow!(
            "rule {}: at least one function-table atom required in body",
            rule.name
        ));
    }
    if rule.actions.is_empty() {
        return Err(anyhow!("rule {}: at least one action required", rule.name));
    }

    let mut variants = Vec::with_capacity(func_atom_indices.len());
    for &focus in &func_atom_indices {
        variants.push(compile_variant(rule, focus, functions)?);
    }
    Ok(CompiledRule {
        name: rule.name.clone(),
        variants,
    })
}

/// Build one SQL string per action in the rule, all sharing the
/// FROM/WHERE derived from the body atoms with focus on `focus_idx`.
fn compile_variant(
    rule: &Rule,
    focus: usize,
    functions: &HashMap<String, FunctionInfo>,
) -> Result<Vec<String>> {
    let (from, where_clause, binding) = build_query(rule, focus)?;

    let mut sqls = Vec::with_capacity(rule.actions.len());
    for action in &rule.actions {
        let Action::Insert {
            name: target,
            args: targs,
        } = action;
        let info = &functions[target];
        let select_cols: Vec<String> = targs
            .iter()
            .map(|t| term_sql(t, &binding, &rule.name))
            .collect::<Result<_>>()?;
        let target_cols: Vec<String> = (0..targs.len()).map(|i| format!("c{i}")).collect();
        // ?2 is :next_ts.
        let select_list = format!("{}, ?2", select_cols.join(", "));
        let insert_cols = format!("{}, ts", target_cols.join(", "));
        let conflict = conflict_clause(info);
        sqls.push(format!(
            "INSERT INTO {target} ({insert_cols}) SELECT {select_list} FROM {from}{where_clause} {conflict}"
        ));
    }
    Ok(sqls)
}

/// Walk the body atoms, returning `(from_clause, where_clause,
/// var_binding)`.  Function-table atoms become FROM aliases and
/// either bind variables or contribute equality constraints; filter
/// atoms become WHERE constraints. The focused atom contributes a
/// `tFOCUS.ts >= ?1` predicate.
fn build_query(
    rule: &Rule,
    focus: usize,
) -> Result<(String, String, HashMap<String, String>)> {
    let mut binding: HashMap<String, String> = HashMap::new();
    let mut from_parts: Vec<String> = Vec::new();
    let mut where_parts: Vec<String> = Vec::new();

    for (i, atom) in rule.body.iter().enumerate() {
        match atom {
            Atom::Func { name, args } => {
                let alias = format!("t{i}");
                from_parts.push(format!("{name} {alias}"));
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
                        Term::Lit(_) | Term::Prim(_, _) => {
                            let rhs = term_sql(term, &binding, &rule.name)?;
                            where_parts.push(format!("{lhs} = {rhs}"));
                        }
                    }
                }
            }
            Atom::Filter(t) => {
                let s = term_sql(t, &binding, &rule.name)?;
                where_parts.push(s);
            }
        }
    }
    where_parts.push(format!("t{focus}.ts >= ?1"));

    let from = from_parts.join(", ");
    let where_clause = format!(" WHERE {}", where_parts.join(" AND "));
    Ok((from, where_clause, binding))
}

/// Compile a `Term` to its SQL text given the current variable
/// binding. Variables must already be bound (introduced by an
/// earlier function-table atom).
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
    }
}

/// Map a primitive op name + its already-compiled arg SQLs into a
/// SQL expression. Wraps in parens to preserve precedence at the
/// surrounding context.
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
    match op {
        "+" | "-" | "*" | "/" => binop(op),
        "<" | "<=" | ">" | ">=" => binop(op),
        "=" => binop("="),
        "!=" | "<>" => binop("<>"),
        "and" => binop("AND"),
        "or" => binop("OR"),
        "not" => unop("NOT"),
        _ => Err(anyhow!(
            "rule {rule_name}: unknown primitive `{op}` (supported: +, -, *, /, <, <=, >, >=, =, !=, and, or, not)"
        )),
    }
}

fn lit_sql(l: &crate::Literal) -> String {
    match l {
        crate::Literal::I64(i) => i.to_string(),
        crate::Literal::Bool(b) => if *b { "TRUE" } else { "FALSE" }.to_string(),
    }
}
