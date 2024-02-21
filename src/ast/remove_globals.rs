//! Remove global variables from the program by translating
//! them into functions with no arguments.
//! This requires type information, so it is done after type checking.

use crate::{
    core::ResolvedCall, typechecking::FuncType, GenericAction, GenericActions, GenericExpr,
    GenericNCommand, ResolvedAction, ResolvedExpr, ResolvedFunctionDecl, ResolvedNCommand,
    ResolvedVar, Schema, TypeInfo,
};

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
pub(crate) fn remove_globals(
    type_info: &TypeInfo,
    prog: Vec<ResolvedNCommand>,
) -> Vec<ResolvedNCommand> {
    prog.into_iter()
        .flat_map(|cmd| remove_globals_cmd(type_info, cmd))
        .collect()
}

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

fn remove_globals_cmd(type_info: &TypeInfo, cmd: ResolvedNCommand) -> Vec<ResolvedNCommand> {
    match cmd {
        GenericNCommand::CoreAction(action) => match action {
            GenericAction::Let(ann, name, expr) => {
                let ty = expr.output_type(type_info);
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
                    is_global: true,
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
            _ => vec![GenericNCommand::CoreAction(remove_globals_action(action))],
        },
        _ => vec![cmd.visit_exprs(&mut remove_globals_expr)],
    }
}
