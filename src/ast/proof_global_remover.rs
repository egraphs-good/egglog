//! Remove global variables from the program by translating
//! them constructors, making proof generation easier.
//! Does not support primitive-valued globals.

use crate::ast::{
    FunctionSubtype, GenericNCommand, ResolvedAction, ResolvedActions, ResolvedExprExt,
    ResolvedFunctionDecl, ResolvedNCommand, Schema,
};
use crate::*;
use crate::{core::ResolvedCall, typechecking::FuncType};
use egglog_ast::generic_ast::{GenericAction, GenericExpr, GenericFact, GenericRule};

struct ProofGlobalRemover<'a> {
    fresh: &'a mut SymbolGen,
}

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
    fresh: &mut SymbolGen,
) -> Vec<ResolvedNCommand> {
    let mut remover = ProofGlobalRemover { fresh };
    prog.into_iter()
        .flat_map(|cmd| remover.remove_globals_cmd(cmd))
        .collect()
}

fn resolved_var_to_call(var: &ResolvedVar) -> ResolvedCall {
    assert!(
        var.is_global_ref,
        "resolved_var_to_call called on non-global var"
    );
    ResolvedCall::Func(FuncType {
        name: var.name.clone(),
        subtype: FunctionSubtype::Constructor,
        input: vec![],
        output: var.sort.clone(),
    })
}

/// TODO (yz) it would be better to implement replace_global_var
/// as a function from ResolvedVar to ResolvedExpr
/// and use it as an argument to `subst` instead of `visit_expr`,
/// but we have not implemented `subst` for command.
fn replace_global_vars(expr: ResolvedExpr) -> ResolvedExpr {
    match expr.get_global_var() {
        Some(resolved_var) => {
            GenericExpr::Call(expr.span(), resolved_var_to_call(&resolved_var), vec![])
        }
        None => expr,
    }
}

fn remove_globals_expr(expr: ResolvedExpr) -> ResolvedExpr {
    expr.visit_exprs(&mut replace_global_vars)
}

fn remove_globals_action(action: ResolvedAction) -> ResolvedAction {
    action.visit_exprs(&mut replace_global_vars)
}

impl ProofGlobalRemover<'_> {
    fn remove_globals_cmd(&mut self, cmd: ResolvedNCommand) -> Vec<ResolvedNCommand> {
        match cmd {
            GenericNCommand::CoreAction(action) => match action {
                GenericAction::Let(span, name, expr) => {
                    let ty = expr.output_type();
                    if !ty.is_eq_sort() {
                        panic!("Global variable {} has non-eq sort {}", name, ty.name());
                    }

                    let resolved_call = ResolvedCall::Func(FuncType {
                        name: name.name.clone(),
                        subtype: FunctionSubtype::Constructor,
                        input: vec![],
                        output: ty.clone(),
                    });
                    let func_decl = ResolvedFunctionDecl {
                        name: name.name,
                        subtype: FunctionSubtype::Constructor,
                        schema: Schema {
                            input: vec![],
                            output: ty.name().to_owned(),
                        },
                        resolved_schema: resolved_call.clone(),
                        merge: None,
                        cost: None,
                        unextractable: true,
                        let_binding: true,
                        span: span.clone(),
                    };
                    vec![
                        GenericNCommand::Function(func_decl),
                        GenericNCommand::CoreAction(GenericAction::Union(
                            span.clone(),
                            ResolvedExpr::Call(span.clone(), resolved_call, vec![]),
                            remove_globals_expr(expr),
                        )),
                    ]
                }
                _ => vec![GenericNCommand::CoreAction(remove_globals_action(action))],
            },
            GenericNCommand::NormRule { rule } => {
                let new_rule = GenericRule {
                    span: rule.span,
                    body: rule
                        .body
                        .iter()
                        .map(|fact| fact.clone().visit_exprs(&mut replace_global_vars))
                        .collect(),
                    head: ResolvedActions::new(
                        rule.head
                            .iter()
                            .map(|action| action.clone().visit_exprs(&mut replace_global_vars))
                            .collect(),
                    ),
                    name: rule.name.clone(),
                    ruleset: rule.ruleset.clone(),
                };
                vec![GenericNCommand::NormRule { rule: new_rule }]
            }
            // Handle the corner case where a global command is wrap in (fail )
            GenericNCommand::Fail(span, cmd) => {
                let mut removed = self.remove_globals_cmd(*cmd);
                let last = removed.pop().unwrap();
                let boxed_last = Box::new(last);
                let new_command = GenericNCommand::Fail(span, boxed_last);
                removed.push(new_command);
                removed
            }
            _ => vec![cmd.visit_exprs(&mut replace_global_vars)],
        }
    }
}
