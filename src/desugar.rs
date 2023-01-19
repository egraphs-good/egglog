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

pub(crate) fn desugar_command(command: Command) -> Vec<Command> {
    match command {
        Command::Datatype { name, variants } => desugar_datatype(name, variants),
        _ => vec![command],
    }
}

pub(crate) fn desugar_program(program: Vec<Command>) -> Vec<Command> {
    program.into_iter().flat_map(desugar_command).collect()
}
