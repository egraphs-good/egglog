//! Remove global variables from the program by translating
//! them into functions with no arguments.
//! This requires type information, so it is done after type checking.
//! Primitives are translated into functions with a primitive output.
//! When a globally-bound primitive value is used in the actions of a rule,
//! we add a new variable to the query bound to the primitive value.

use crate::*;
use crate::{core::ResolvedCall, typechecking::FuncType};
use egglog_ast::generic_ast::{GenericAction, GenericExpr, GenericFact, GenericRule};

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

/// The nullary function of `subtype` that the global `name` of sort `output` becomes.
pub(crate) fn global_call(name: String, subtype: FunctionSubtype, output: ArcSort) -> ResolvedCall {
    ResolvedCall::Func(Arc::new(FuncType {
        name,
        subtype,
        input: vec![],
        output,
    }))
}

/// The declaration of the function [`global_call`] calls.
pub(crate) fn global_decl(
    span: Span,
    name: String,
    subtype: FunctionSubtype,
    output: &ArcSort,
) -> ResolvedFunctionDecl {
    ResolvedFunctionDecl {
        name,
        subtype,
        schema: Schema {
            input: vec![],
            output: output.name().to_owned(),
        },
        merge: None,
        cost: None,
        unextractable: true,
        internal_hidden: false,
        internal_let: true,
        span,
        term_constructor: None,
    }
}

/// Replaces each reference to a global with a call to the function of `subtype` it became.
///
/// TODO (yz) it would be better to implement replace_global_var
/// as a function from ResolvedVar to ResolvedExpr
/// and use it as an argument to `subst` instead of `visit_expr`,
/// but we have not implemented `subst` for command.
pub(crate) fn replace_global_vars(
    subtype: FunctionSubtype,
) -> impl FnMut(ResolvedExpr) -> ResolvedExpr {
    move |expr| match expr.get_global_var() {
        Some(var) => GenericExpr::Call(
            expr.span(),
            global_call(var.name, subtype, var.sort),
            vec![],
        ),
        None => expr,
    }
}

pub(crate) fn remove_globals_expr(expr: ResolvedExpr) -> ResolvedExpr {
    expr.visit_exprs(&mut replace_global_vars(FunctionSubtype::Custom))
}

/// Wraps the last command of a `fail`'s expansion back in the `fail`.
pub(crate) fn fail_last(span: Span, mut removed: Vec<ResolvedNCommand>) -> Vec<ResolvedNCommand> {
    let last = removed
        .pop()
        .expect("removing globals expands every command to at least one command");
    removed.push(GenericNCommand::Fail(span, Box::new(last)));
    removed
}

impl GlobalRemover<'_> {
    fn remove_globals_cmd(&mut self, cmd: ResolvedNCommand) -> Vec<ResolvedNCommand> {
        let mut replace = replace_global_vars(FunctionSubtype::Custom);
        match cmd {
            GenericNCommand::CoreAction(GenericAction::Let(span, name, expr)) => {
                let ty = expr.output_type();
                vec![
                    GenericNCommand::Function(global_decl(
                        span.clone(),
                        name.name.clone(),
                        FunctionSubtype::Custom,
                        &ty,
                    )),
                    GenericNCommand::CoreAction(GenericAction::Set(
                        span,
                        global_call(name.name, FunctionSubtype::Custom, ty),
                        vec![],
                        expr.visit_exprs(&mut replace),
                    )),
                ]
            }
            GenericNCommand::NormRule { rule } => {
                // A map from the global variables in actions to their new names
                // in the query.
                let mut globals = HashMap::default();
                rule.head.visit_vars(&mut |span, resolved_var| {
                    // Type checking marks let binders as local, so only references
                    // to globals contribute query facts.
                    if resolved_var.is_global_ref {
                        let new_name = self.fresh.fresh(&resolved_var.name);
                        globals.insert(
                            resolved_var.clone(),
                            GenericExpr::Var(
                                span.clone(),
                                ResolvedVar {
                                    name: new_name,
                                    sort: resolved_var.sort.clone(),
                                    is_global_ref: false,
                                },
                            ),
                        );
                    }
                });
                let new_facts: Vec<ResolvedFact> = globals
                    .iter()
                    .map(|(old, new)| {
                        GenericFact::Eq(
                            new.span(),
                            GenericExpr::Call(
                                new.span(),
                                global_call(
                                    old.name.clone(),
                                    FunctionSubtype::Custom,
                                    old.sort.clone(),
                                ),
                                vec![],
                            ),
                            new.clone(),
                        )
                    })
                    .collect();

                let new_rule = GenericRule {
                    // instrument the old facts and add the new facts to the end
                    body: rule
                        .body
                        .into_iter()
                        .map(|fact| fact.visit_exprs(&mut replace))
                        .chain(new_facts)
                        .collect(),
                    // replace references to globals with the newly bound names
                    head: rule.head.visit_exprs(&mut |expr| {
                        if let Some(resolved_var) = expr.get_global_var() {
                            globals.get(&resolved_var).unwrap().clone()
                        } else {
                            expr
                        }
                    }),
                    ..rule
                };
                vec![GenericNCommand::NormRule { rule: new_rule }]
            }
            // Handle the corner case where a global command is wrap in (fail )
            GenericNCommand::Fail(span, cmd) => fail_last(span, self.remove_globals_cmd(*cmd)),
            cmd => vec![cmd.visit_exprs(&mut replace)],
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::EGraph;

    #[test]
    fn globals_in_rule_actions_preserve_local_bindings() {
        EGraph::default()
            .parse_and_run_program(
                None,
                r#"
                (let $offset 3)
                (function result (i64) i64 :no-merge)
                (relation seed (i64))
                (seed 1)
                (rule ((seed n) (= $offset 3))
                      ((let local (+ n $offset))
                       (set (result n) (+ local $offset))))
                (run 1)
                (check (= (result 1) 7))
                "#,
            )
            .unwrap();
    }
}
