//! Compile a high-level `Rule` into one SQL string per seminaive
//! variant.
//!
//! A rule with N function-table atoms produces N variants. Variant i
//! adds a `WHERE alias_i.ts >= ?1` predicate on the focused atom and
//! reads the current epoch as `?2`. Both come in as bind parameters
//! at execution time.

use anyhow::{Result, anyhow};
use std::collections::HashMap;

use crate::{Action, Atom, CompiledRule, FunctionInfo, Rule, Term};

pub(crate) fn compile_rule(
    rule: &Rule,
    functions: &HashMap<String, FunctionInfo>,
) -> Result<CompiledRule> {
    // Validate that every atom references a registered function.
    for atom in &rule.body {
        let Atom::Func { name, args } = atom;
        let info = functions
            .get(name)
            .ok_or_else(|| anyhow!("rule {}: unknown function {name}", rule.name))?;
        if args.len() != info.schema.len() {
            return Err(anyhow!(
                "rule {}: atom {name} has {} args, expected {}",
                rule.name,
                args.len(),
                info.schema.len()
            ));
        }
    }

    // No function atoms → one variant with no focus restriction.
    let function_atom_count = rule.body.len();
    if function_atom_count == 0 {
        return Err(anyhow!(
            "rule {} has no body atoms; not yet supported",
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

fn compile_variant(
    rule: &Rule,
    focus: usize,
    _functions: &HashMap<String, FunctionInfo>,
) -> Result<String> {
    // Strategy:
    // 1. Each body atom becomes a table reference in the FROM clause
    //    with alias `t<idx>`.
    // 2. Variables are resolved to `t<idx>.c<col>` using the FIRST
    //    atom that mentions them; later occurrences become equality
    //    constraints in the WHERE clause.
    // 3. Literal arguments in body atoms become equality constraints.
    // 4. The focused atom adds `t<focus>.ts >= ?1`.
    // 5. Each action becomes one `INSERT INTO ... SELECT ... ON
    //    CONFLICT DO NOTHING`. We emit one SQL string per variant
    //    that runs all actions; if there are multiple actions, we
    //    chain them with `;`.

    let mut binding: HashMap<String, String> = HashMap::new(); // var -> "tN.cM"
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
                        binding.insert(v.clone(), lhs.clone());
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

    // Focus predicate.
    where_parts.push(format!("t{focus}.ts >= ?1"));

    let from = from_parts.join(", ");
    let where_clause = if where_parts.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", where_parts.join(" AND "))
    };

    // Build the inserts. With multiple actions we join them with `;`
    // — DuckDB executes them in order.
    let mut inserts: Vec<String> = Vec::new();
    for action in &rule.actions {
        let Action::Insert {
            name: target,
            args: targs,
        } = action;
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
        inserts.push(format!(
            "INSERT INTO {target} ({insert_cols}) SELECT {select_list} FROM {from}{where_clause} ON CONFLICT DO NOTHING"
        ));
    }
    if inserts.is_empty() {
        return Err(anyhow!(
            "rule {} has no actions; not yet supported",
            rule.name
        ));
    }
    // Multiple actions in the same statement: chain with `;`. DuckDB
    // accepts batch statements via `execute_batch`, but `prepare` is
    // single-statement, so for now restrict to one action per rule.
    if inserts.len() > 1 {
        return Err(anyhow!(
            "rule {} has {} actions; multi-action rules not yet supported (todo: split into multiple prepared statements)",
            rule.name,
            inserts.len()
        ));
    }
    Ok(inserts.into_iter().next().unwrap())
}

fn lit_sql(l: &crate::Literal) -> String {
    match l {
        crate::Literal::I64(i) => i.to_string(),
        crate::Literal::Bool(b) => b.to_string(),
    }
}
