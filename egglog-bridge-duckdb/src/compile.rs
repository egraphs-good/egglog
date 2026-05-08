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
    // Validate atoms.
    for atom in &rule.body {
        let Atom::Func { name, args } = atom;
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
    // Validate actions.
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

    let function_atom_count = rule.body.len();
    if function_atom_count == 0 {
        return Err(anyhow!(
            "rule {} has no body atoms; not yet supported",
            rule.name
        ));
    }
    if rule.actions.is_empty() {
        return Err(anyhow!(
            "rule {} has no actions; not yet supported",
            rule.name
        ));
    }

    let mut variants = Vec::with_capacity(function_atom_count);
    for focus in 0..function_atom_count {
        variants.push(compile_variant(rule, focus, functions)?);
    }
    Ok(CompiledRule {
        name: rule.name.clone(),
        variants,
    })
}

/// Build one SQL string per action in the rule, all sharing the
/// same FROM/WHERE clauses derived from the body atoms with the
/// focus on `focus_idx`.
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
            .map(|t| match t {
                Term::Var(v) => binding
                    .get(v)
                    .cloned()
                    .ok_or_else(|| anyhow!("rule {}: unbound var {v} in action", rule.name)),
                Term::Lit(l) => Ok(lit_sql(l)),
            })
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

/// Walk the body atoms and produce `(from_clause, where_clause,
/// var_binding)`. The first occurrence of each variable defines its
/// binding location (`tN.cM`); later occurrences become `=`
/// constraints. Literal arguments become `=` constraints. The
/// focused atom contributes a `tFOCUS.ts >= ?1` predicate.
fn build_query(
    rule: &Rule,
    focus: usize,
) -> Result<(String, String, HashMap<String, String>)> {
    let mut binding: HashMap<String, String> = HashMap::new();
    let mut from_parts: Vec<String> = Vec::new();
    let mut where_parts: Vec<String> = Vec::new();

    for (i, atom) in rule.body.iter().enumerate() {
        let Atom::Func { name, args } = atom;
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
                Term::Lit(l) => {
                    where_parts.push(format!("{lhs} = {}", lit_sql(l)));
                }
            }
        }
    }
    where_parts.push(format!("t{focus}.ts >= ?1"));

    let from = from_parts.join(", ");
    let where_clause = format!(" WHERE {}", where_parts.join(" AND "));
    Ok((from, where_clause, binding))
}

fn lit_sql(l: &crate::Literal) -> String {
    match l {
        crate::Literal::I64(i) => i.to_string(),
        crate::Literal::Bool(b) => b.to_string(),
    }
}
