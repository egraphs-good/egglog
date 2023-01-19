use crate::*;

fn make_ast_version(egraph: &EGraph, name: &Symbol) -> Symbol {
    if egraph.sorts.get(name).is_some() {
        name.clone()
    } else {
        Symbol::from(format!("_Ast{}", name))
    }
}

fn make_ast_func(egraph: &EGraph, fdecl: &FunctionDecl) -> FunctionDecl {
    FunctionDecl {
        name: make_ast_version(egraph, &fdecl.name),
        schema: Schema {
            input: fdecl
                .schema
                .input
                .iter()
                .map(|sort| make_ast_version(egraph, sort))
                .collect(),
            output: make_ast_version(egraph, &fdecl.schema.output),
        },
        merge: None,
        merge_action: vec![],
        default: None,
        cost: None,
    }
}

// the egraph is the initial egraph with only default sorts
pub(crate) fn add_proofs(egraph: &EGraph, program: Vec<Command>) -> Vec<Command> {
    let mut res = vec![];
    for command in program {
        match &command {
            Command::Datatype {
                name: _,
                variants: _,
            } => {
                panic!("Datatype should have been desugared");
            }
            Command::Sort(name, presort_and_args) => {
                res.push(Command::Sort(
                    make_ast_version(egraph, name),
                    presort_and_args.clone(),
                ));
                res.push(command);
            }
            Command::Function(fdecl) => {
                res.push(Command::Function(make_ast_func(egraph, fdecl)));
                res.push(command);
            }
            _ => res.push(command),
        }
    }

    res
}
