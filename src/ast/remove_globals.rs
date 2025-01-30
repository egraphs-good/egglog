//! Remove global variables from the program by translating
//! them into functions with no arguments.
//! This requires type information, so it is done after type checking.
//! Primitives are translated into functions with a primitive output.
//! When a globally-bound primitive value is used in the actions of a rule,
//! we add a new variable to the query bound to the primitive value.

use crate::*;
use crate::{core::ResolvedCall, typechecking::FuncType};

struct GlobalRemover<'a> {
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
/// (let x 3)
/// (Add x x)
/// ```
/// becomes
/// ```ignore
/// (function x () i64)
/// (set (x) 3)
/// (Add (x) (x))
/// ```
///
/// If later, this global is referenced in a rule:
/// ```ignore
/// (rule ((Neg y))
///       ((Add x x)))
/// ```
/// We instrument the query to make the value available:
/// ```ignore
/// (rule ((Neg y)
///        (= fresh_var_for_x (x)))
///       ((Add fresh_var_for_x fresh_var_for_x)))
/// ```
pub(crate) fn remove_globals(
    prog: Vec<ResolvedNCommand>,
    fresh: &mut SymbolGen,
) -> Vec<ResolvedNCommand> {
    let mut remover = GlobalRemover { fresh };
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
        name: var.name,
        subtype: FunctionSubtype::Custom,
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

impl GlobalRemover<'_> {
    fn remove_globals_cmd(&mut self, cmd: ResolvedNCommand) -> Vec<ResolvedNCommand> {
        match cmd {
            GenericNCommand::CoreAction(action) => match action {
                GenericAction::Let(span, name, expr) => {
                    let ty = expr.output_type();

                    let func_decl = ResolvedFunctionDecl {
                        name: name.name,
                        subtype: FunctionSubtype::Custom,
                        schema: Schema {
                            input: vec![],
                            output: ty.name(),
                        },
                        merge: None,
                        cost: None,
                        unextractable: true,
                        ignore_viz: true,
                        span: span.clone(),
                    };
                    let resolved_call = ResolvedCall::Func(FuncType {
                        name: name.name,
                        subtype: FunctionSubtype::Custom,
                        input: vec![],
                        output: ty.clone(),
                    });
                    vec![
                        GenericNCommand::Function(func_decl),
                        GenericNCommand::CoreAction(GenericAction::Set(
                            span,
                            resolved_call,
                            vec![],
                            remove_globals_expr(expr),
                        )),
                    ]
                }
                _ => vec![GenericNCommand::CoreAction(remove_globals_action(action))],
            },
            GenericNCommand::NormRule {
                name,
                ruleset,
                rule,
            } => {
                // A map from the global variables in actions to their new names
                // in the query.
                let mut globals = HashMap::default();
                rule.head.clone().visit_exprs(&mut |expr| {
                    if let Some(resolved_var) = expr.get_global_var() {
                        let new_name = self.fresh.fresh(&resolved_var.name);
                        globals.insert(
                            resolved_var.clone(),
                            GenericExpr::Var(
                                expr.span(),
                                ResolvedVar {
                                    name: new_name,
                                    sort: resolved_var.sort.clone(),
                                    is_global_ref: false,
                                },
                            ),
                        );
                    }
                    expr
                });
                let new_facts: Vec<ResolvedFact> = globals
                    .iter()
                    .map(|(old, new)| {
                        GenericFact::Eq(
                            new.span(),
                            GenericExpr::Call(new.span(), resolved_var_to_call(old), vec![]),
                            new.clone(),
                        )
                    })
                    .collect();

                let new_rule = GenericRule {
                    span: rule.span,
                    // instrument the old facts and add the new facts to the end
                    body: rule
                        .body
                        .iter()
                        .map(|fact| fact.clone().visit_exprs(&mut replace_global_vars))
                        .chain(new_facts)
                        .collect(),
                    // replace references to globals with the newly bound names
                    head: rule.head.clone().visit_exprs(&mut |expr| {
                        if let Some(resolved_var) = expr.get_global_var() {
                            globals.get(&resolved_var).unwrap().clone()
                        } else {
                            expr
                        }
                    }),
                };
                vec![GenericNCommand::NormRule {
                    name,
                    ruleset,
                    rule: new_rule,
                }]
            }
            //Handle the corner case where a global command is wrap in (fail )
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
