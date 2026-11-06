//! Remove global variables from the program by translating
//! them into constructors, making proof generation easier.
//! Does not support primitive-valued globals.

use crate::ast::remove_globals::{fail_last, global_call, global_decl, replace_global_vars};
use crate::ast::{FunctionSubtype, GenericNCommand, ResolvedExprExt, ResolvedNCommand};
use crate::*;
use egglog_ast::generic_ast::{GenericAction, GenericRule};

/// Removes all globals from a program.
/// No top level lets are allowed after this pass,
/// nor any variable that references a global.
/// Adds new functions for global variables
/// and replaces references to globals with
/// references to the new functions.
/// e.g.
/// ```ignore
/// (let x (Add 1 2))
/// ```
/// becomes
/// ```ignore
/// (function x () Math)
/// (union (x) (Add 1 2))
/// ```
pub(crate) fn remove_globals(
    prog: Vec<ResolvedNCommand>,
    _fresh: &mut SymbolGen,
) -> Vec<ResolvedNCommand> {
    prog.into_iter().flat_map(remove_globals_cmd).collect()
}

fn remove_globals_cmd(cmd: ResolvedNCommand) -> Vec<ResolvedNCommand> {
    let mut replace = replace_global_vars(FunctionSubtype::Constructor);
    match cmd {
        GenericNCommand::CoreAction(GenericAction::Let(span, name, expr)) => {
            let ty = expr.output_type();
            if !ty.is_eq_sort() {
                panic!("Global variable {} has non-eq sort {}", name, ty.name());
            }
            vec![
                GenericNCommand::Function(global_decl(
                    span.clone(),
                    name.name.clone(),
                    FunctionSubtype::Constructor,
                    &ty,
                )),
                GenericNCommand::CoreAction(GenericAction::Union(
                    span.clone(),
                    ResolvedExpr::Call(
                        span,
                        global_call(name.name, FunctionSubtype::Constructor, ty),
                        vec![],
                    ),
                    expr.visit_exprs(&mut replace),
                )),
            ]
        }
        GenericNCommand::NormRule { rule } => vec![GenericNCommand::NormRule {
            rule: GenericRule {
                body: rule
                    .body
                    .into_iter()
                    .map(|fact| fact.visit_exprs(&mut replace))
                    .collect(),
                head: rule.head.visit_exprs(&mut replace),
                ..rule
            },
        }],
        GenericNCommand::Fail(span, cmd) => fail_last(span, remove_globals_cmd(*cmd)),
        cmd => vec![cmd.visit_exprs(&mut replace)],
    }
}
