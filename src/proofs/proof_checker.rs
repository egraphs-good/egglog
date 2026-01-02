use crate::{
    Term, TermDag, TermId,
    ast::{GenericAction, GenericNCommand, ResolvedExpr, ResolvedNCommand},
    util::HashMap,
};

/// Gathers all global variables from a program and computes their values as terms
/// without using globals (i.e., all global references are replaced with their definitions).
/// 
/// This expects the program to be in proof normalized form where globals appear as Let actions.
pub(crate) fn gather_globals(
    prog: &[ResolvedNCommand],
    term_dag: &mut TermDag,
) -> HashMap<String, TermId> {
    let mut globals = HashMap::default();

    for cmd in prog {
        if let GenericNCommand::CoreAction(GenericAction::Let(_, var, expr)) = cmd {
            // All Let actions in the normalized desugared commands are globals
            let term_id = expr_to_term_without_globals(expr, term_dag, &globals);
            globals.insert(var.name.clone(), term_id);
        }
    }

    globals
}

/// Convert an expression to a term, replacing any global variable references
/// with their pre-computed term values.
fn expr_to_term_without_globals(
    expr: &ResolvedExpr,
    dag: &mut TermDag,
    globals: &HashMap<String, TermId>,
) -> TermId {
    let term = match expr {
        ResolvedExpr::Lit(_, lit) => dag.lit(lit.clone()),
        ResolvedExpr::Var(_, var) => {
            if var.is_global_ref {
                // Replace global reference with its computed value
                if let Some(&term_id) = globals.get(&var.name) {
                    return term_id;
                } else {
                    panic!(
                        "Global variable {} not found in globals map. \
                        This likely means it's being used before it's defined.",
                        var.name
                    );
                }
            } else {
                // Non-global variable - this shouldn't happen in global definitions
                panic!(
                    "Non-global variable {} found while constructing global term",
                    var.name
                );
            }
        }
        ResolvedExpr::Call(_, head, args) => {
            let arg_terms: Vec<Term> = args
                .iter()
                .map(|arg| {
                    let term_id = expr_to_term_without_globals(arg, dag, globals);
                    dag.get(term_id).clone()
                })
                .collect();
            dag.app(head.name().to_string(), arg_terms)
        }
    };
    dag.lookup(&term)
}
