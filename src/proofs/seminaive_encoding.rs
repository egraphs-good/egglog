//! Seminaive encoding: a source-to-source pass over term-encoded
//! programs that replaces backend-level seminaive evaluation with
//! per-rule timestamp predicates.
//!
//! For each function `f` in the program, a parallel `f_ts` table is
//! created with the same key columns and an `i64` output. Every
//! insertion to `f` is mirrored by an insertion to `f_ts` carrying
//! the current epoch.
//!
//! Each rule with N function-table atoms is expanded into N variants.
//! Variant `i` adds `(>= ts_i (last_run_at_<src>))` to the focused
//! atom and binds `(= now (next_ts))` once per body, used in actions.
//!
//! The bookkeeping globals `next_ts` and `last_run_at_<src>` are
//! managed by the schedule executor in `src/lib.rs::step_rules`,
//! which detects them by name and updates them between iterations.
//!
//! Scope: term/proof mode only. See `seminaive-encoding-experiment.md`
//! at the repo root for the experiment context.

use crate::ast::{Command, GenericAction, GenericCommand, GenericExpr, GenericFact, GenericRule};
use crate::util::{FreshGen, HashSet};
use egglog_ast::generic_ast::Change;

/// Walk the (term-encoded) commands and produce a seminaive-encoded
/// version. The schedule executor's hook in `step_rules` is what
/// makes the encoded predicates actually filter at runtime.
pub(crate) fn add_seminaive_encoding(
    commands: Vec<Command>,
    parser: &mut crate::ast::Parser,
    tracked: &mut HashSet<String>,
    emit_session_header: bool,
) -> Vec<Command> {
    let mut rule_sources: HashSet<String> = HashSet::default();
    // Map original rule.name → clean source name we'll use in
    // `last_run_at_<src>` globals and variant suffixes. Some rules
    // arrive with their textual s-expression as `name` (when the
    // user didn't supply `:name`); those have characters that can't
    // appear in a function identifier, so we substitute a fresh
    // symbol.
    let mut rule_name_remap: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    // First pass: extend the cross-call tracked set with anything new
    // declared in this batch, and collect the rule source names so we
    // can emit the right `last_run_at_<src>` globals.
    for cmd in &commands {
        match cmd {
            GenericCommand::Function { name, .. } => {
                tracked.insert(name.clone());
            }
            GenericCommand::Constructor { name, .. } => {
                tracked.insert(name.clone());
            }
            GenericCommand::Relation { name, .. } => {
                tracked.insert(name.clone());
            }
            GenericCommand::Rule { rule } => {
                let src = clean_rule_source(parser, &rule.name, &mut rule_name_remap);
                rule_sources.insert(src);
            }
            _ => {}
        }
    }

    let mut out: Vec<Command> = Vec::new();

    // Bookkeeping globals. `next_ts` is emitted exactly once per
    // session (controlled by `emit_session_header`). `last_run_at_*`
    // is emitted per source rule we see in this batch — each call
    // sees only the rules added in that command, so we declare them
    // here lazily.
    let mut header = String::new();
    if emit_session_header {
        header.push_str("(function next_ts () i64 :merge new)\n");
        header.push_str("(set (next_ts) 0)\n");
    }
    let mut rule_sources_sorted: Vec<_> = rule_sources.iter().cloned().collect();
    rule_sources_sorted.sort();
    for src in &rule_sources_sorted {
        let g = last_run_at_name(src);
        header.push_str(&format!("(function {g} () i64 :merge new)\n"));
        header.push_str(&format!("(set ({g}) 0)\n"));
    }
    if !header.is_empty() {
        out.extend(parse(parser, &header));
    }

    for cmd in commands {
        match cmd {
            GenericCommand::Function {
                ref name, ref schema, ref merge, ..
            } => {
                let ts_decl = ts_decl_for_function(name, schema, merge);
                out.push(cmd.clone());
                out.extend(parse(parser, &ts_decl));
            }
            GenericCommand::Constructor {
                ref name, ref schema, ..
            } => {
                let ts_decl = ts_decl_for_inputs(name, &schema.input);
                out.push(cmd.clone());
                out.extend(parse(parser, &ts_decl));
            }
            GenericCommand::Relation {
                ref name, ref inputs, ..
            } => {
                let ts_decl = ts_decl_for_inputs(name, inputs);
                out.push(cmd.clone());
                out.extend(parse(parser, &ts_decl));
            }
            GenericCommand::Rule { rule } => {
                let src = clean_rule_source(parser, &rule.name, &mut rule_name_remap);
                let variants = expand_rule(parser, &rule, &src, tracked);
                out.extend(variants);
            }
            GenericCommand::Action(action) => {
                out.extend(mirror_top_level_action(parser, action, tracked));
            }
            other => out.push(other),
        }
    }

    out
}

fn parse(parser: &mut crate::ast::Parser, src: &str) -> Vec<Command> {
    let saved = parser.ensure_no_reserved_symbols;
    parser.ensure_no_reserved_symbols = false;
    let res = parser
        .get_program_from_string(None, src)
        .unwrap_or_else(|e| {
            panic!(
                "seminaive encoding produced unparseable text:\n\
                 --- begin ---\n{src}\n--- end ---\nerror: {e}"
            );
        });
    parser.ensure_no_reserved_symbols = saved;
    res
}

fn last_run_at_name(src: &str) -> String {
    format!("last_run_at_{src}")
}

fn ts_name(name: &str) -> String {
    format!("{name}_ts")
}

fn ts_decl_for_function(
    name: &str,
    schema: &crate::ast::Schema,
    _merge: &Option<GenericExpr<String, String>>,
) -> String {
    let inputs = schema.input.join(" ");
    // `_ts` is always `:merge old` (first-epoch-wins). Reasons:
    // - For `:merge old` base functions: the value never changes
    //   after first set, so first-epoch ts is correct.
    // - For `:merge new` and custom-merge base functions: making
    //   `_ts` `:merge new` would let the ts update on every set
    //   even when the underlying value didn't change. That breaks
    //   egglog's saturation detection — the schedule sees a "change"
    //   to the ts table on every iteration and never terminates.
    //   With `:merge old` ts, we accept some under-firing for
    //   downstream rules that depend on a `:merge new` base
    //   function's value updating.
    let mode = "old";
    let ts = ts_name(name);
    format!("(function {ts} ({inputs}) i64 :merge {mode})\n")
}

fn ts_decl_for_inputs(name: &str, inputs: &[String]) -> String {
    let inputs_s = inputs.join(" ");
    let ts = ts_name(name);
    format!("(function {ts} ({inputs_s}) i64 :merge old)\n")
}

/// Return a list of `Command`s representing the seminaive variants
/// of a single source rule. Each variant focuses on one body atom
/// whose head is in `tracked`. Rules with no tracked atoms in their
/// body get a single variant with no focus predicate (still gets the
/// ts mirroring on actions and the `now` binding for use in actions).
fn expand_rule(
    parser: &mut crate::ast::Parser,
    rule: &GenericRule<String, String>,
    source_name: &str,
    tracked: &HashSet<String>,
) -> Vec<Command> {
    let function_atoms = collect_function_atom_indices(&rule.body, tracked);

    if function_atoms.is_empty() {
        return vec![build_variant(
            parser,
            rule,
            source_name,
            &format!("{source_name}@1"),
            tracked,
            None,
        )];
    }

    function_atoms
        .iter()
        .enumerate()
        .map(|(variant_idx, &focus_idx)| {
            build_variant(
                parser,
                rule,
                source_name,
                &format!("{source_name}@{}", variant_idx + 1),
                tracked,
                Some(focus_idx),
            )
        })
        .collect()
}

/// Convert a rule's stored `name` into a clean identifier we can
/// embed in a function/global name. Rules without an explicit
/// `:name` clause are auto-named with their textual s-expression in
/// `desugar.rs::rule_name`, which is fine for debugging but contains
/// parens/spaces/quotes — we substitute a fresh symbol for those.
/// The remap table makes the substitution stable across the per-call
/// passes so a single source rule maps to a single source name.
fn clean_rule_source(
    parser: &mut crate::ast::Parser,
    raw: &str,
    remap: &mut std::collections::HashMap<String, String>,
) -> String {
    if let Some(existing) = remap.get(raw) {
        return existing.clone();
    }
    let clean = if is_clean_identifier(raw) {
        raw.to_string()
    } else {
        parser.symbol_gen.fresh("seminaive_rule")
    };
    remap.insert(raw.to_string(), clean.clone());
    clean
}

fn is_clean_identifier(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| {
            c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '@' || c == '$'
        })
}

/// Find indices into `body` of facts whose top-level head is a
/// tracked function/constructor.
fn collect_function_atom_indices(
    body: &[GenericFact<String, String>],
    tracked: &HashSet<String>,
) -> Vec<usize> {
    let mut out = Vec::new();
    for (i, fact) in body.iter().enumerate() {
        if function_atom_head(fact)
            .map(|h| tracked.contains(h))
            .unwrap_or(false)
        {
            out.push(i);
        }
    }
    out
}

/// If a fact is a function-call atom (either bare `(f a b)` or an
/// equality binding `(= v (f a b))`), return the function name.
fn function_atom_head(fact: &GenericFact<String, String>) -> Option<&str> {
    match fact {
        GenericFact::Fact(GenericExpr::Call(_, head, _)) => Some(head.as_str()),
        GenericFact::Eq(_, _, GenericExpr::Call(_, head, _)) => Some(head.as_str()),
        _ => None,
    }
}

/// Extract the args of a function-call atom (assumes
/// `function_atom_head` would return Some).
fn function_atom_args(
    fact: &GenericFact<String, String>,
) -> Vec<GenericExpr<String, String>> {
    match fact {
        GenericFact::Fact(GenericExpr::Call(_, _, args)) => args.clone(),
        GenericFact::Eq(_, _, GenericExpr::Call(_, _, args)) => args.clone(),
        _ => unreachable!(),
    }
}

fn build_variant(
    parser: &mut crate::ast::Parser,
    rule: &GenericRule<String, String>,
    source_name: &str,
    variant_name: &str,
    tracked: &HashSet<String>,
    focus_idx: Option<usize>,
) -> Command {
    let ruleset = &rule.ruleset;
    let now_var = parser.symbol_gen.fresh("now");
    let last_var = parser.symbol_gen.fresh("last");

    // Build the rule's text from scratch using s-expression
    // formatting. We do this rather than mutating the AST because
    // the s-exp reparse is the simplest way to get types and atoms
    // re-resolved consistently.
    let mut body_lines = Vec::<String>::new();

    // Add a `_ts` query and focus predicate for the focused atom
    // only. Unfocused atoms don't need the ts — they just need to
    // match against current state. This keeps join arity O(N+1)
    // instead of O(2N) and avoids burdening the planner with N-1
    // useless `_ts` lookups per rule.
    for (i, fact) in rule.body.iter().enumerate() {
        body_lines.push(format!("{fact}"));
        if Some(i) == focus_idx {
            // Safe: focus_idx is only set when the indexed fact is a
            // tracked function atom (see collect_function_atom_indices).
            let head = function_atom_head(fact).unwrap();
            let args = function_atom_args(fact);
            let args_s = exprs_text(&args);
            let ts_var = parser.symbol_gen.fresh(&format!("ts{i}"));
            body_lines.push(format!(
                "(= {} ({} {}))",
                ts_var,
                ts_name(head),
                args_s
            ));
            let lan = last_run_at_name(source_name);
            body_lines.push(format!("(= {last_var} ({lan}))"));
            body_lines.push(format!("(>= {ts_var} {last_var})"));
        }
    }
    // Bind `now` once per variant. It's referenced from action
    // mirrors below.
    body_lines.push(format!("(= {now_var} (next_ts))"));

    // Walk actions; mirror calls on tracked names.
    let mut action_lines = Vec::<String>::new();
    for act in &rule.head.0 {
        action_lines.push(format!("{act}"));
        action_lines.extend(mirror_action_text(act, tracked, &now_var));
    }

    let body_s = body_lines.join("\n      ");
    let actions_s = action_lines.join("\n      ");
    let ruleset_clause = if ruleset.is_empty() {
        String::new()
    } else {
        format!(":ruleset {ruleset} ")
    };
    let text = format!(
        "(rule ({body_s})\n      ({actions_s})\n      {ruleset_clause}:name \"{variant_name}\")"
    );
    let mut parsed = parse(parser, &text);
    assert_eq!(parsed.len(), 1, "rule reparse should produce one command");
    parsed.pop().unwrap()
}

/// If `act` is a `set`/`delete`/`let`/expr-call on a tracked name,
/// return the matching `_ts`-mirror action(s) as text. Returns
/// multiple lines for nested calls (e.g. `(C (D x) y)` yields a
/// mirror for both C and D). Otherwise empty.
fn mirror_action_text(
    act: &GenericAction<String, String>,
    tracked: &HashSet<String>,
    now_var: &str,
) -> Vec<String> {
    let mut out = Vec::new();
    match act {
        GenericAction::Set(_, head, args, val) => {
            for arg in args {
                collect_call_mirrors(arg, tracked, now_var, &mut out);
            }
            collect_call_mirrors(val, tracked, now_var, &mut out);
            if tracked.contains(head) {
                let args_s = exprs_text(args);
                out.push(format!("(set ({} {}) {})", ts_name(head), args_s, now_var));
            }
        }
        GenericAction::Change(_, Change::Delete, head, args) => {
            for arg in args {
                collect_call_mirrors(arg, tracked, now_var, &mut out);
            }
            if tracked.contains(head) {
                let args_s = exprs_text(args);
                out.push(format!("(delete ({} {}))", ts_name(head), args_s));
            }
        }
        // Subsume doesn't need a paired _ts action — the base row is
        // just marked, the _ts row's correctness is unaffected.
        GenericAction::Change(_, Change::Subsume, _, args) => {
            for arg in args {
                collect_call_mirrors(arg, tracked, now_var, &mut out);
            }
        }
        GenericAction::Let(_, _, expr) | GenericAction::Expr(_, expr) => {
            collect_call_mirrors(expr, tracked, now_var, &mut out);
        }
        GenericAction::Union(_, lhs, rhs) => {
            collect_call_mirrors(lhs, tracked, now_var, &mut out);
            collect_call_mirrors(rhs, tracked, now_var, &mut out);
        }
        GenericAction::Panic(_, _) => {}
    }
    out
}

/// Walk an expression, and for every nested call to a tracked
/// function/constructor, append a `(set (<name>_ts <args>) <now>)`
/// mirror. With `:merge old` on the `_ts` table, re-setting an
/// existing key is a no-op so this is safe to emit unconditionally.
fn collect_call_mirrors(
    expr: &GenericExpr<String, String>,
    tracked: &HashSet<String>,
    now_var: &str,
    out: &mut Vec<String>,
) {
    if let GenericExpr::Call(_, head, args) = expr {
        for arg in args {
            collect_call_mirrors(arg, tracked, now_var, out);
        }
        if tracked.contains(head) {
            let args_s = exprs_text(args);
            out.push(format!("(set ({} {}) {})", ts_name(head), args_s, now_var));
        }
    }
}

fn exprs_text(args: &[GenericExpr<String, String>]) -> String {
    args.iter()
        .map(|a| format!("{a}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Mirror a top-level action (e.g. `(set (AddView 1 2 v) ())`) by
/// emitting the action plus, if applicable, a paired `_ts` set.
/// At the top level we use `(next_ts)` directly since lookup of
/// non-constructor functions IS allowed in top-level actions.
fn mirror_top_level_action(
    parser: &mut crate::ast::Parser,
    action: GenericAction<String, String>,
    tracked: &HashSet<String>,
) -> Vec<Command> {
    let mut out = Vec::new();
    // At the top level, lookup of non-constructor functions is
    // allowed (the LookupInRuleDisallowed check fires only inside
    // rules), so we can use `(next_ts)` directly in the mirror text.
    let lines = mirror_action_text(&action, tracked, "(next_ts)");
    out.push(GenericCommand::Action(action));
    for line in lines {
        out.extend(parse(parser, &line));
    }
    out
}

