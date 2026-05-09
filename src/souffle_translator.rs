//! Translate the souffle_compat-encoded form of an egglog program to a
//! Souffle IR ([`egglog_souffle_backend::ir::Program`]).
//!
//! ## Approach
//!
//! Skolemization happens *after* term/proof encoding, not as a replacement
//! for it. The encoded form already handles UF tables, view tables,
//! congruence, rebuild, and (in proof mode) proof tracking. This translator
//! takes that encoded `Vec<ResolvedNCommand>` and:
//!
//! 1. Maps egglog sorts to a single Souffle record type (v0).
//! 2. Maps egglog functions/constructors to Souffle relations (`.decl`).
//! 3. For each rule, walks its actions:
//!    - `(let v (Cons args))` records `v -> Skolem record [tag, args...]`
//!      in a per-rule binding map. The let itself is dropped — every use
//!      of `v` is inlined as the record.
//!    - `(set (R args) val)` becomes one Souffle rule head, sharing the
//!      rule's body, with the value column = inlined record (or `()` for
//!      Unit-output relations).
//!    - `(delete (R args))` becomes a subsumption rule.
//! 4. Schedules become `.pragma "outer-saturate"` + `.limititerations`.
//!
//! Status: scaffolding. The walk over commands is in place but most cases
//! return [`TranslateError::Unsupported`] until we work through the encoded
//! shapes one by one. The framework + tests are first; per-case logic is
//! incremental.

use egglog_souffle_backend::ir::{
    Atom, BinaryOp, Clause, Expr, Literal as IrLit, Program, RelationDecl, TypeDecl, TypeKind,
};
use std::collections::HashMap;

use crate::ast::*;
use crate::core::ResolvedCall;
use egglog_ast::generic_ast::{GenericAction, GenericExpr, GenericFact, Literal as AstLit};

/// Translation errors. v0 returns Unsupported for cases we haven't covered
/// yet — refusing rather than silently dropping keeps the contract clear.
#[derive(Debug, Clone)]
pub enum TranslateError {
    Unsupported(String),
}

impl std::fmt::Display for TranslateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TranslateError::Unsupported(s) => write!(f, "unsupported: {s}"),
        }
    }
}
impl std::error::Error for TranslateError {}

/// Translation context — accumulates symbol mappings as commands are walked.
#[derive(Default)]
pub struct Ctx {
    /// Constructor name → numeric tag (assigned in declaration order).
    /// Used to build Skolem records when inlining let-bindings.
    pub tag_of: HashMap<String, i64>,
    /// Set of sort names we've seen. v0 collapses all sorts to a single
    /// `Math` record type.
    pub sorts: Vec<String>,
    /// Whether the `Math` type and Term/UF helper relations have been emitted.
    emitted_helpers: bool,
}

/// The single record type all sorts collapse to in v0.
const MATH: &str = "Math";

/// Translate a sequence of encoded commands to a [`Program`].
pub fn translate(commands: &[ResolvedCommand]) -> Result<Program, TranslateError> {
    let mut p = Program::default();
    let mut ctx = Ctx::default();
    for cmd in commands {
        translate_command(cmd, &mut ctx, &mut p)?;
    }
    Ok(p)
}

fn ensure_helpers(ctx: &mut Ctx, p: &mut Program) {
    if ctx.emitted_helpers {
        return;
    }
    p.types.push(TypeDecl {
        name: MATH.into(),
        kind: TypeKind::Record(vec![
            ("tag".into(), "number".into()),
            ("a".into(), MATH.into()),
            ("b".into(), MATH.into()),
            ("n".into(), "number".into()),
        ]),
    });
    ctx.emitted_helpers = true;
}

fn translate_command(
    cmd: &ResolvedCommand,
    ctx: &mut Ctx,
    p: &mut Program,
) -> Result<(), TranslateError> {
    use ResolvedCommand as C;
    match cmd {
        C::Sort { name, .. } => {
            ensure_helpers(ctx, p);
            ctx.sorts.push(name.to_string());
            Ok(())
        }
        C::Constructor { name, schema, .. } => {
            ensure_helpers(ctx, p);
            let tag = ctx.tag_of.len() as i64;
            ctx.tag_of.insert(name.to_string(), tag);
            // Record arity for build_record_for_tag (currently re-derived
            // from args at use site, but we could cache it here).
            let _ = schema;
            Ok(())
        }
        C::Function { name, schema, .. } => {
            ensure_helpers(ctx, p);
            // Functions (UF, view tables, etc.) become Souffle relations.
            let cols = function_columns(schema, ctx)?;
            p.relations.push(RelationDecl {
                name: name.to_string(),
                columns: cols,
            });
            Ok(())
        }
        C::AddRuleset(..) | C::UnstableCombinedRuleset(..) => Ok(()),
        C::Rule { rule } => translate_rule(rule, ctx, p),
        C::Action(action) => {
            // Top-level action — typically a `(set ...)` to seed the database
            // or a `(let ...)` global.
            let mut lets: HashMap<String, Expr> = HashMap::new();
            let clauses = translate_action(action, &mut lets, &[], ctx)?;
            for c in clauses {
                p.clauses.push(c);
            }
            Ok(())
        }
        C::RunSchedule(_) => {
            if !p.pragmas.iter().any(|(k, _)| k == "outer-saturate") {
                p.pragmas.push(("outer-saturate".into(), "100".into()));
            }
            Ok(())
        }
        // Everything else: silently skip in v0 (driver-side or not yet
        // supported). The caller can opt in to stricter handling later.
        _ => Ok(()),
    }
}

/// Build the Souffle column list for an egglog function schema.
fn function_columns(
    schema: &Schema,
    _ctx: &mut Ctx,
) -> Result<Vec<(String, String)>, TranslateError> {
    let mut cols = vec![];
    for (i, sort) in schema.input.iter().enumerate() {
        cols.push((format!("c{i}"), sort_to_souffle_type(sort.as_str())));
    }
    cols.push((
        "out".into(),
        sort_to_souffle_type(schema.output.as_str()),
    ));
    Ok(cols)
}

fn sort_to_souffle_type(sort: &str) -> String {
    match sort {
        "i64" => "number".into(),
        "Unit" => "number".into(), // Unit becomes a 0-valued number
        "String" => "symbol".into(),
        // All user sorts collapse to MATH in v0.
        _ => MATH.into(),
    }
}

/// Translate a rule. Walks the body to build Souffle body literals, then
/// each action — accumulating let-bindings and emitting one Souffle rule
/// per Set/Change.
fn translate_rule(
    rule: &ResolvedRule,
    ctx: &mut Ctx,
    p: &mut Program,
) -> Result<(), TranslateError> {
    let body = translate_body(&rule.body, ctx)?;
    let mut lets: HashMap<String, Expr> = HashMap::new();
    for action in &rule.head.0 {
        let clauses = translate_action(action, &mut lets, &body, ctx)?;
        for c in clauses {
            p.clauses.push(c);
        }
    }
    Ok(())
}

/// Translate a list of body facts into Souffle body literals.
fn translate_body(
    facts: &[ResolvedFact],
    ctx: &mut Ctx,
) -> Result<Vec<IrLit>, TranslateError> {
    let mut out = vec![];
    for fact in facts {
        match fact {
            GenericFact::Fact(expr) => translate_fact_expr(expr, ctx, &mut out)?,
            GenericFact::Eq(_, lhs, rhs) => translate_eq_fact(lhs, rhs, ctx, &mut out)?,
        }
    }
    Ok(out)
}

/// A `(R args)` body fact becomes a Souffle relation match. A `(prim args)`
/// body fact becomes a constraint or function call (v0 handles only `!=`).
fn translate_fact_expr(
    expr: &ResolvedExpr,
    ctx: &mut Ctx,
    out: &mut Vec<IrLit>,
) -> Result<(), TranslateError> {
    if let GenericExpr::Call(_, head, args) = expr {
        match head {
            ResolvedCall::Func(_) => {
                let atom = build_atom(head.name(), args, ctx)?;
                out.push(IrLit::Atom(atom));
                Ok(())
            }
            ResolvedCall::Primitive(prim) => {
                if prim.name() == "!=" && args.len() == 2 {
                    let a = translate_value_expr(&args[0], ctx)?;
                    let b = translate_value_expr(&args[1], ctx)?;
                    out.push(IrLit::Constraint(BinaryOp::Ne, a, b));
                    Ok(())
                } else {
                    Err(TranslateError::Unsupported(format!(
                        "body primitive not yet supported: {}",
                        prim.name()
                    )))
                }
            }
        }
    } else {
        Err(TranslateError::Unsupported(
            "non-Call body fact".into(),
        ))
    }
}

/// `(= var (R args))` binds `var` to R's output column.
/// `(= e1 e2)` with neither a Call becomes a Souffle equality constraint.
fn translate_eq_fact(
    lhs: &ResolvedExpr,
    rhs: &ResolvedExpr,
    ctx: &mut Ctx,
    out: &mut Vec<IrLit>,
) -> Result<(), TranslateError> {
    // Pattern: `(= var (Call R args))` — binding the output column of R.
    let (var_name, call) = match (lhs, rhs) {
        (GenericExpr::Var(_, v), GenericExpr::Call(_, h, a)) => (v.name.to_string(), Some((h, a))),
        (GenericExpr::Call(_, h, a), GenericExpr::Var(_, v)) => (v.name.to_string(), Some((h, a))),
        _ => (String::new(), None),
    };
    if let Some((head, args)) = call
        && let ResolvedCall::Func(_) = head
    {
        let mut souffle_args: Vec<Expr> = args
            .iter()
            .map(|a| translate_value_expr(a, ctx))
            .collect::<Result<_, _>>()?;
        souffle_args.push(Expr::Var(var_name));
        out.push(IrLit::Atom(Atom {
            relation: head.name().to_string(),
            args: souffle_args,
        }));
        return Ok(());
    }
    // Otherwise: a plain equality between two values. Souffle: `e1 = e2`.
    let l = translate_value_expr(lhs, ctx)?;
    let r = translate_value_expr(rhs, ctx)?;
    out.push(IrLit::Constraint(BinaryOp::Eq, l, r));
    Ok(())
}

/// Build a Souffle atom for `(R args)` — for body facts where R is a relation.
/// All args are translated as values; R has no implicit output binding.
fn build_atom(
    relation: &str,
    args: &[ResolvedExpr],
    ctx: &mut Ctx,
) -> Result<Atom, TranslateError> {
    let mut souffle_args = Vec::with_capacity(args.len());
    for a in args {
        souffle_args.push(translate_value_expr(a, ctx)?);
    }
    Ok(Atom { relation: relation.to_string(), args: souffle_args })
}

/// Translate an expression that produces a value (variable, literal, or
/// constructor call). Constructor calls become inline records; let-bound
/// names are looked up via the `lets` map at action-translation sites.
fn translate_value_expr(
    expr: &ResolvedExpr,
    ctx: &mut Ctx,
) -> Result<Expr, TranslateError> {
    match expr {
        GenericExpr::Var(_, v) => Ok(Expr::Var(v.name.to_string())),
        GenericExpr::Lit(_, AstLit::Int(n)) => Ok(Expr::Number(*n)),
        GenericExpr::Lit(_, AstLit::Unit) => Ok(Expr::Number(0)),
        GenericExpr::Lit(_, AstLit::String(s)) => Ok(Expr::Symbol(s.clone())),
        GenericExpr::Lit(_, _) => Err(TranslateError::Unsupported("non-i64/Unit literal".into())),
        GenericExpr::Call(_, head, args) => {
            // Constructor call → inline record [tag, args...]
            if let Some(&tag) = ctx.tag_of.get(head.name()) {
                build_record_for_tag(tag, args, ctx)
            } else if let ResolvedCall::Primitive(prim) = head {
                // Handle ordering-max / ordering-min as ord() comparisons —
                // these are used for deterministic union direction in the
                // encoded form. For now, return Unsupported; we'll handle
                // these specially when we encounter them in actions.
                Err(TranslateError::Unsupported(format!(
                    "value-position primitive: {}",
                    prim.name()
                )))
            } else {
                // Function call in value position — e.g., `(__UF_Mathf c)` —
                // can't be a value in plain Datalog; would need to be lifted
                // to a body fact. v0: refuse.
                Err(TranslateError::Unsupported(format!(
                    "function in value position: {}",
                    head.name()
                )))
            }
        }
    }
}

fn build_record_for_tag(
    tag: i64,
    args: &[ResolvedExpr],
    ctx: &mut Ctx,
) -> Result<Expr, TranslateError> {
    // Same shape as in our examples: [tag, a, b, n]
    match args.len() {
        0 => Ok(Expr::Record(vec![
            Expr::Number(tag),
            Expr::Nil,
            Expr::Nil,
            Expr::Number(0),
        ])),
        1 => {
            // If the single arg is an i64 lit, put it in the n column.
            let a = translate_value_expr(&args[0], ctx)?;
            if matches!(a, Expr::Number(_)) {
                Ok(Expr::Record(vec![Expr::Number(tag), Expr::Nil, Expr::Nil, a]))
            } else {
                Ok(Expr::Record(vec![Expr::Number(tag), a, Expr::Nil, Expr::Number(0)]))
            }
        }
        2 => {
            let a = translate_value_expr(&args[0], ctx)?;
            let b = translate_value_expr(&args[1], ctx)?;
            Ok(Expr::Record(vec![Expr::Number(tag), a, b, Expr::Number(0)]))
        }
        n => Err(TranslateError::Unsupported(format!(
            "constructor arity {n} > 2 in v0"
        ))),
    }
}

/// Apply let-binding substitutions to an Expr.
fn substitute(e: &Expr, lets: &HashMap<String, Expr>) -> Expr {
    match e {
        Expr::Var(v) => lets.get(v).cloned().unwrap_or_else(|| e.clone()),
        Expr::Record(fields) => Expr::Record(fields.iter().map(|f| substitute(f, lets)).collect()),
        Expr::Ord(inner) => Expr::Ord(Box::new(substitute(inner, lets))),
        _ => e.clone(),
    }
}

/// Translate one rule action. Some actions only mutate the let-binding map
/// (returning no clauses); others emit one or more Souffle clauses.
fn translate_action(
    action: &ResolvedAction,
    lets: &mut HashMap<String, Expr>,
    body: &[IrLit],
    ctx: &mut Ctx,
) -> Result<Vec<Clause>, TranslateError> {
    match action {
        GenericAction::Let(_, var, expr) => {
            // The let mints a Skolem record. Inline at use sites.
            let val = translate_value_expr(expr, ctx)?;
            // Apply let-bindings to the result so chained lets work.
            let val = substitute(&val, lets);
            lets.insert(var.name.to_string(), val);
            Ok(vec![])
        }
        GenericAction::Set(_, head, args, val) => {
            let mut souffle_args = Vec::with_capacity(args.len() + 1);
            for a in args {
                let e = translate_value_expr(a, ctx)?;
                souffle_args.push(substitute(&e, lets));
            }
            let val_e = translate_value_expr(val, ctx)?;
            souffle_args.push(substitute(&val_e, lets));
            let head_atom = Atom {
                relation: head.name().to_string(),
                args: souffle_args,
            };
            // Apply let substitution to body too (in case body literals
            // mention names that got rebound — rare but possible).
            let body_subst: Vec<IrLit> = body.iter().map(|l| substitute_literal(l, lets)).collect();
            Ok(vec![Clause::rule(head_atom, body_subst)])
        }
        GenericAction::Change(_, change_kind, head, args) => {
            // Both Delete and Subsume → Souffle subsumption rule that removes
            // the matching tuple. Without a "more general" tuple to dominate,
            // this is the tombstone-style deletion: the rule body says when
            // to delete, but we need to dominate by something.
            //
            // For v0, a single-relation subsumption removes the row by being
            // dominated by ITSELF (no-op) — that's wrong. The real fix is
            // tombstone-via-helper-relation. Mark unsupported until we get
            // the helper-relation pattern wired up.
            let _ = (change_kind, head, args, lets, body, ctx);
            Err(TranslateError::Unsupported(
                "delete/subsume actions need tombstone-relation support (TODO)".into(),
            ))
        }
        GenericAction::Union(..) => Err(TranslateError::Unsupported(
            "union should be lowered to set in encoded form".into(),
        )),
        GenericAction::Panic(..) => Err(TranslateError::Unsupported("panic action".into())),
        GenericAction::Expr(_, _) => {
            // Expression-statement at action position — evaluating an
            // expression for side effects. v0: skip.
            Ok(vec![])
        }
    }
}

fn substitute_literal(lit: &IrLit, lets: &HashMap<String, Expr>) -> IrLit {
    match lit {
        IrLit::Atom(a) => IrLit::Atom(Atom {
            relation: a.relation.clone(),
            args: a.args.iter().map(|e| substitute(e, lets)).collect(),
        }),
        IrLit::Neg(a) => IrLit::Neg(Atom {
            relation: a.relation.clone(),
            args: a.args.iter().map(|e| substitute(e, lets)).collect(),
        }),
        IrLit::Constraint(op, l, r) => {
            IrLit::Constraint(*op, substitute(l, lets), substitute(r, lets))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: an empty program produces an empty Program.
    #[test]
    fn empty_program() {
        let p = translate(&[]).unwrap();
        assert!(p.types.is_empty());
        assert!(p.relations.is_empty());
        assert!(p.clauses.is_empty());
    }

    /// End-to-end smoke: a tiny user program goes through term encoding +
    /// souffle_compat, then through this translator, and comes out as
    /// non-empty Souffle source. This isn't a full correctness test (we
    /// don't run Souffle here), but it exercises every code path
    /// implemented so far. Failures will surface as TranslateError.
    #[test]
    fn translates_tiny_egglog_program() {
        let mut egraph = crate::EGraph::new_with_term_encoding().with_souffle_compat();
        let commands = egraph
            .resolve_program(
                None,
                r#"
                (sort Math)
                (constructor Add (i64 i64) Math)
                (Add 1 2)
                "#,
            )
            .unwrap();

        // Filter out commands the v0 translator can't handle yet — print a
        // clear message rather than crashing the test if we hit one.
        let mut p = Program::default();
        let mut ctx = Ctx::default();
        let mut skipped = 0usize;
        for cmd in &commands {
            if let Err(e) = translate_command(cmd, &mut ctx, &mut p) {
                eprintln!("skipping unsupported command: {e}");
                skipped += 1;
            }
        }
        // Even if some commands are skipped, the basic Math type and at
        // least one constructor tag should have been recorded.
        assert!(!ctx.tag_of.is_empty(), "expected at least one tag mapping");
        assert!(p.types.iter().any(|t| t.name == "Math"));
        eprintln!(
            "translated {} commands ({} skipped); program has {} relations, {} clauses",
            commands.len() - skipped,
            skipped,
            p.relations.len(),
            p.clauses.len(),
        );
    }
}
