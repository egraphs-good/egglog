//! Remove global variables from the program by translating
//! them into functions with no arguments.
//! This requires type information, so it is done after type checking.

use crate::{
    core::ResolvedCall, typechecking::FuncType, GenericAction, GenericActions, GenericExpr,
    GenericNCommand, HashMap, ResolvedAction, ResolvedExpr, ResolvedFunctionDecl, ResolvedNCommand,
    ResolvedVar, Schema, Symbol, SymbolGen, TypeInfo,
};

struct GlobalRemover {
    fresh: SymbolGen,
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
/// Primitives and containers need to be added to the query
/// so they can be referenced in actions.
/// Datatypes can be referenced directly from actions.
pub(crate) fn remove_globals(
    type_info: &TypeInfo,
    prog: Vec<ResolvedNCommand>,
) -> Vec<ResolvedNCommand> {
    let mut remover = GlobalRemover {
        fresh: SymbolGen::new(),
    };
    prog.into_iter()
        .flat_map(|cmd| remover.remove_globals_cmd(type_info, cmd))
        .collect()
}

/// TODO (yz) it would be better to implement replace_global_var
/// as a function from ResolvedVar to ResolvedExpr
/// and use it as an argument to `subst` instead of `visit_expr`,
/// but we have not implemented `subst` for command.
fn replace_global_var(expr: ResolvedExpr) -> ResolvedExpr {
    match expr {
        GenericExpr::Var(
            ann,
            ResolvedVar {
                name,
                sort,
                is_global_ref: true,
            },
        ) => GenericExpr::Call(
            ann,
            ResolvedCall::Func(FuncType {
                name,
                input: vec![],
                output: sort,
                is_datatype: false,
                has_default: false,
            }),
            vec![],
        ),
        _ => expr,
    }
}

fn remove_globals_expr(expr: ResolvedExpr) -> ResolvedExpr {
    expr.visit_exprs(&mut replace_global_var)
}

fn remove_globals_action(action: ResolvedAction) -> ResolvedAction {
    action.visit_exprs(&mut replace_global_var)
}

impl GlobalRemover {
    fn remove_globals_cmd(
        &mut self,
        type_info: &TypeInfo,
        cmd: ResolvedNCommand,
    ) -> Vec<ResolvedNCommand> {
        match cmd {
            GenericNCommand::CoreAction(action) => match action {
                GenericAction::Let(ann, name, expr) => {
                    let ty = expr.output_type(type_info);
                    if ty.is_eq_sort() {
                        // output is eq-able, so generate a union
                        let func_decl = ResolvedFunctionDecl {
                            name: name.name,
                            schema: Schema {
                                input: vec![],
                                output: ty.name(),
                            },
                            default: None,
                            merge: None,
                            merge_action: GenericActions(vec![]),
                            cost: None,
                            unextractable: true,
                            ignore_viz: true,
                        };
                        vec![
                            GenericNCommand::Function(func_decl),
                            GenericNCommand::CoreAction(GenericAction::Union(
                                ann,
                                GenericExpr::Call(
                                    ann,
                                    ResolvedCall::Func(FuncType {
                                        name: name.name,
                                        input: vec![],
                                        output: ty,
                                        is_datatype: true,
                                        has_default: false,
                                    }),
                                    vec![],
                                ),
                                remove_globals_expr(expr),
                            )),
                        ]
                    } else {
                        let func_decl = ResolvedFunctionDecl {
                            name: name.name,
                            schema: Schema {
                                input: vec![],
                                output: ty.name(),
                            },
                            default: None,
                            merge: None,
                            merge_action: GenericActions(vec![]),
                            cost: None,
                            unextractable: true,
                            ignore_viz: true,
                        };

                        vec![
                            GenericNCommand::Function(func_decl),
                            GenericNCommand::CoreAction(GenericAction::Set(
                                ann,
                                ResolvedCall::Func(FuncType {
                                    name: name.name,
                                    input: vec![],
                                    output: ty,
                                    is_datatype: false,
                                    has_default: false,
                                }),
                                vec![],
                                remove_globals_expr(expr),
                            )),
                        ]
                    }
                }
                _ => vec![GenericNCommand::CoreAction(remove_globals_action(action))],
            },
            /*GenericNCommand::NormRule {
                name,
                ruleset,
                rule,
            } => {
                // rules handled specially because actions can't refer to
                // global functions with primitive outputs
                GenericNCommand::NormRule {
                    name,
                    ruleset,
                    rule: self.remove_globals_rule(rule),
                }
            }*/
            _ => vec![cmd.visit_exprs(&mut replace_global_var)],
        }
    }
}
