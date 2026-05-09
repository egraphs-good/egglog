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

use egglog_souffle_backend::ir::*;
use std::collections::HashMap;

use crate::ast::*;

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
pub fn translate(commands: &[ResolvedNCommand]) -> Result<Program, TranslateError> {
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
    cmd: &ResolvedNCommand,
    ctx: &mut Ctx,
    p: &mut Program,
) -> Result<(), TranslateError> {
    use ResolvedNCommand as C;
    match cmd {
        C::Sort { name, .. } => {
            ensure_helpers(ctx, p);
            ctx.sorts.push(name.to_string());
            Ok(())
        }
        C::Function(fdecl) => {
            ensure_helpers(ctx, p);
            // Record tag for constructors so inlined records know which tag
            // to use.
            if fdecl.subtype == FunctionSubtype::Constructor {
                let tag = ctx.tag_of.len() as i64;
                ctx.tag_of.insert(fdecl.name.to_string(), tag);
                // Constructors don't need a separate relation — the record
                // value IS the e-class identity. Skip emitting a .decl.
                return Ok(());
            }
            // Functions (UF, view tables, etc.) become Souffle relations.
            let cols = fdecl_columns(fdecl, ctx)?;
            p.relations.push(RelationDecl {
                name: fdecl.name.to_string(),
                columns: cols,
            });
            Ok(())
        }
        C::AddRuleset(..) | C::UnstableCombinedRuleset(..) => Ok(()),
        C::NormRule { .. }
        | C::CoreAction(..)
        | C::RunSchedule(..)
        | C::Check(..)
        | C::Extract(..) => {
            // Per-rule / per-action translation lives in follow-up commits.
            // For now, refuse so callers get a clear error rather than a
            // silently-incomplete program.
            Err(TranslateError::Unsupported(format!(
                "command kind not yet handled: {:?}",
                std::mem::discriminant(cmd)
            )))
        }
        // Top-level imperative commands have no Souffle analog; we just
        // skip them (they need to be handled by the driver outside Souffle).
        C::Push(..) | C::Pop(..) | C::Fail(..) | C::PrintOverallStatistics(..)
        | C::PrintFunction(..) | C::PrintSize(..) | C::Output { .. } | C::Input { .. }
        | C::ProveExists(..) | C::UserDefined(..) => Ok(()),
    }
}

/// Build the Souffle column list for an egglog function declaration.
/// Each input column becomes the same-named Souffle column with type `Math`
/// (for sort-typed columns) or the matching primitive type for i64/etc.
fn fdecl_columns(
    fdecl: &ResolvedFunctionDecl,
    _ctx: &mut Ctx,
) -> Result<Vec<(String, String)>, TranslateError> {
    let mut cols = vec![];
    for (i, sort) in fdecl.schema.input.iter().enumerate() {
        cols.push((format!("c{i}"), sort_to_souffle_type(sort.as_str())));
    }
    cols.push((
        "out".into(),
        sort_to_souffle_type(fdecl.schema.output.as_str()),
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
}
