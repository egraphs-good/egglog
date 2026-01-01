use crate::ast::{FunctionSubtype, ResolvedNCommand};
use crate::*;
use crate::{core::ResolvedCall, typechecking::FuncType};
use egglog_ast::generic_ast::GenericExpr;

/// Puts queries in "proof form". In proof form, function calls like (lower-bound a b)
/// are always top level and look like this:
/// (= (lower-bound a b) c)
///
/// Nested function calls like this are not allowed:
/// (Add c (lower-bound a b))
pub(crate) fn proof_form(
    prog: Vec<ResolvedNCommand>,
    fresh: &mut SymbolGen,
) -> Vec<ResolvedNCommand> {
    prog.into_iter()
        .map(|cmd| proof_form_cmd(cmd, fresh))
        .collect()
}

fn proof_form_cmd(cmd: ResolvedNCommand, fresh: &mut SymbolGen) -> ResolvedNCommand {
    cmd.visit_queries(&mut |query| {
        let mut new_query = vec![];
        for fact in query {
            let rewritten = proof_form_fact(fact, &mut new_query, fresh);
            new_query.push(rewritten);
        }
        new_query
    })
}

fn proof_form_fact(
    fact: ResolvedFact,
    res: &mut Vec<ResolvedFact>,
    fresh: &mut SymbolGen,
) -> ResolvedFact {
    match fact {
        ResolvedFact::Eq(
            span,
            ResolvedExpr::Call(
                span2,
                head @ ResolvedCall::Func(FuncType {
                    subtype: FunctionSubtype::Custom,
                    ..
                }),
                args,
            ),
            ResolvedExpr::Var(span3, v),
        ) => {
            let mut new_args = vec![];
            for arg in args {
                new_args.push(proof_form_expr(arg, res, fresh));
            }
            ResolvedFact::Eq(
                span,
                ResolvedExpr::Call(span2, head, new_args),
                ResolvedExpr::Var(span3, v),
            )
        }
        GenericFact::Eq(span, generic_expr, generic_expr2) => GenericFact::Eq(
            span,
            proof_form_expr(generic_expr, res, fresh),
            proof_form_expr(generic_expr2, res, fresh),
        ),
        GenericFact::Fact(generic_expr) => {
            GenericFact::Fact(proof_form_expr(generic_expr, res, fresh))
        }
    }
}

fn proof_form_expr(
    fact: ResolvedExpr,
    res: &mut Vec<ResolvedFact>,
    fresh: &mut SymbolGen,
) -> ResolvedExpr {
    match fact {
        ref fact @ ResolvedExpr::Call(
            ref span,
            ref head @ ResolvedCall::Func(FuncType {
                subtype: FunctionSubtype::Custom,
                ref output,
                ..
            }),
            ref args,
        ) => {
            // bind this to a new variable
            let new_args = args
                .iter()
                .map(|expr| proof_form_expr(expr.clone(), res, fresh))
                .collect();
            let resolved = GenericExpr::Var(
                span.clone(),
                ResolvedVar {
                    name: fresh.fresh("n"),
                    sort: output.clone(),
                    is_global_ref: false,
                },
            );
            res.push(ResolvedFact::Eq(
                span.clone(),
                ResolvedExpr::Call(span.clone(), head.clone(), new_args),
                resolved.clone(),
            ));
            log::warn!(
                "Input program not in proof normal form! All function calls must be top-level in query.
            Original fact: {fact}
            New top level fact: {}
            Replace with new variable {}
                ",
                res.last().unwrap(),
                resolved
            );

            resolved
        }
        ResolvedExpr::Call(span, head, args) => {
            let new_args = args
                .into_iter()
                .map(|expr| proof_form_expr(expr, res, fresh))
                .collect();
            ResolvedExpr::Call(span, head, new_args)
        }
        ResolvedExpr::Lit(..) | ResolvedExpr::Var(..) => fact,
    }
}
