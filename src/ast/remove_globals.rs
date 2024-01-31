//! Remove global variables from the program by translating
//! them into functions with no arguments.
//! This requires type information, so it is done after type checking.

use crate::{
    desugar::Desugar, Action, GenericAction, GenericActions, GenericExpr, GenericFunctionDecl,
    GenericNCommand, NCommand, ResolvedActions, ResolvedFunctionDecl, ResolvedNCommand, Schema,
    Symbol, TypeInfo,
};
use hashbrown::HashSet;

pub(crate) fn remove_globals(
    type_info: &TypeInfo,
    prog: &Vec<ResolvedNCommand>,
) -> Vec<ResolvedNCommand> {
    let mut res = Vec::new();
    for cmd in prog {
        res.extend(remove_globals_cmd(type_info, cmd));
    }
    res
}

/// Removes all globals from a command.
/// Adds new functions for new globals
/// and replaces references to globals with
/// references to the new functions.
/// Also adds the types for these functions to
/// the type info struct.
fn remove_globals_cmd(type_info: &TypeInfo, cmd: &ResolvedNCommand) -> Vec<ResolvedNCommand> {
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
                };
                let mut res = vec![
                    GenericNCommand::Function(func_decl),
                    GenericNCommand::CoreAction(GenericAction::Union(
                        (),
                        GenericExpr::Call((), ResolvedCall {}, vec![]),
                        expr,
                    )),
                ];

                res
            }
        },
    }
}
