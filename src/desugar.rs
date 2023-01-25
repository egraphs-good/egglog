use crate::*;

fn desugar_datatype(name: Symbol, variants: Vec<Variant>) -> Vec<Command> {
    vec![Command::Sort(name, None)]
        .into_iter()
        .chain(variants.into_iter().map(|variant| {
            Command::Function(FunctionDecl {
                name: variant.name,
                schema: Schema {
                    input: variant.types,
                    output: name,
                },
                merge: None,
                merge_action: vec![],
                default: None,
                cost: variant.cost,
            })
        }))
        .collect()
}

fn desugar_rewrite(rewrite: &Rewrite) -> Vec<Command> {
    let var = Symbol::from("rewrite_var__");
    vec![Command::Rule(Rule {
        body: [Fact::Eq(vec![Expr::Var(var), rewrite.lhs.clone()])]
            .into_iter()
            .chain(rewrite.conditions.clone())
            .collect(),
        head: vec![Action::Union(Expr::Var(var), rewrite.rhs.clone())],
    })]
}

fn desugar_birewrite(rewrite: &Rewrite) -> Vec<Command> {
    let rw2 = Rewrite {
        lhs: rewrite.rhs.clone(),
        rhs: rewrite.lhs.clone(),
        conditions: rewrite.conditions.clone(),
    };
    desugar_rewrite(rewrite)
        .into_iter()
        .chain(desugar_rewrite(&rw2))
        .collect()
}

// TODO: write an IR that these commands desugar to
pub(crate) fn desugar_command(command: Command) -> Vec<Command> {
    match command {
        Command::Datatype { name, variants } => desugar_datatype(name, variants),
        Command::Rewrite(rewrite) => desugar_rewrite(&rewrite),
        Command::BiRewrite(rewrite) => desugar_birewrite(&rewrite),
        _ => vec![command],
    }
}

pub(crate) fn desugar_program(program: Vec<Command>) -> Vec<Command> {
    program.into_iter().flat_map(desugar_command).collect()
}
