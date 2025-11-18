use super::{Rewrite, Rule};
use crate::*;

/// Desugars a program, removing syntactic sugar.
/// If part of the program has already been desugared,
/// the `type_info` contains type infromation for the already desugared parts.
pub(crate) fn desugar_program(
    program: Vec<Command>,
    parser: &mut Parser,
    type_info: &TypeInfo,
    seminaive_transform: bool,
) -> Result<Vec<NCommand>, Error> {
    let mut res = vec![];
    for command in program.into_iter() {
        let mut desugared = desugar_command(command, parser, type_info, seminaive_transform)?;
        res.append(&mut desugared);
    }
    Ok(res)
}

/// Desugars a single command, removing syntactic sugar.
/// The `type_info` contains type infromation for already desugared parts of the program.
pub(crate) fn desugar_command(
    command: Command,
    parser: &mut Parser,
    type_info: &TypeInfo,
    seminaive_transform: bool,
) -> Result<Vec<NCommand>, Error> {
    let rule_name = rule_name(&command);
    let res = match command {
        Command::Function {
            span,
            name,
            schema,
            merge,
        } => vec![NCommand::Function(FunctionDecl::function(
            span, name, schema, merge,
        ))],
        Command::Constructor {
            span,
            name,
            schema,
            cost,
            unextractable,
        } => vec![NCommand::Function(FunctionDecl::constructor(
            span,
            name,
            schema,
            cost,
            unextractable,
        ))],
        Command::Relation { span, name, inputs } => vec![NCommand::Function(
            FunctionDecl::relation(span, name, inputs),
        )],
        Command::Datatype {
            span,
            name,
            variants,
        } => desugar_datatype(span, name, variants),
        Command::Datatypes { span: _, datatypes } => {
            // first declare all the datatypes as sorts, then add all explicit sorts which could refer to the datatypes, and finally add all the variants as functions
            let mut res = vec![];
            for datatype in datatypes.iter() {
                let span = datatype.0.clone();
                let name = datatype.1.clone();
                if let Subdatatypes::Variants(..) = datatype.2 {
                    res.push(NCommand::Sort(span, name, None));
                }
            }
            let (variants_vec, sorts): (Vec<_>, Vec<_>) = datatypes
                .into_iter()
                .partition(|datatype| matches!(datatype.2, Subdatatypes::Variants(..)));

            for sort in sorts {
                let span = sort.0.clone();
                let name = sort.1;
                let Subdatatypes::NewSort(sort, args) = sort.2 else {
                    unreachable!()
                };
                res.push(NCommand::Sort(span, name, Some((sort, args))));
            }

            for variants in variants_vec {
                let datatype = variants.1;
                let Subdatatypes::Variants(variants) = variants.2 else {
                    unreachable!();
                };
                for variant in variants {
                    res.push(NCommand::Function(FunctionDecl::constructor(
                        variant.span,
                        variant.name,
                        Schema {
                            input: variant.types,
                            output: datatype.clone(),
                        },
                        variant.cost,
                        false,
                    )));
                }
            }

            res
        }
        Command::Rewrite(ruleset, rewrite, subsume) => {
            desugar_rewrite(ruleset, rule_name, rewrite, subsume, parser)
        }
        Command::BiRewrite(ruleset, rewrite) => {
            desugar_birewrite(ruleset, rule_name, rewrite, parser)
        }
        Command::Include(span, file) => {
            let s = std::fs::read_to_string(&file)
                .unwrap_or_else(|_| panic!("{span} Failed to read file {file}"));
            return desugar_program(
                parser.get_program_from_string(Some(file), &s)?,
                parser,
                type_info,
                seminaive_transform,
            );
        }
        Command::Rule { mut rule } => {
            if rule.name.is_empty() {
                // format rule and use it as the name
                rule.name = rule_name;
            }
            vec![NCommand::NormRule { rule }]
        }
        Command::Sort(span, sort, option) => vec![NCommand::Sort(span, sort, option)],
        Command::AddRuleset(span, name) => vec![NCommand::AddRuleset(span, name)],
        Command::UnstableCombinedRuleset(span, name, subrulesets) => {
            vec![NCommand::UnstableCombinedRuleset(span, name, subrulesets)]
        }
        Command::Action(action) => vec![NCommand::CoreAction(action)],
        Command::RunSchedule(sched) => {
            vec![NCommand::RunSchedule(sched.clone())]
        }
        Command::PrintOverallStatistics(span, file) => {
            vec![NCommand::PrintOverallStatistics(span, file.clone())]
        }
        Command::Extract(span, expr, variants) => vec![NCommand::Extract(span, expr, variants)],
        Command::Check(span, facts) => vec![NCommand::Check(span, facts)],
        Command::PrintFunction(span, symbol, size, file, mode) => {
            vec![NCommand::PrintFunction(span, symbol, size, file, mode)]
        }
        Command::PrintSize(span, symbol) => vec![NCommand::PrintSize(span, symbol)],
        Command::Output { span, file, exprs } => {
            vec![NCommand::Output { span, file, exprs }]
        }
        Command::Push(num) => {
            vec![NCommand::Push(num)]
        }
        Command::Pop(span, num) => {
            vec![NCommand::Pop(span, num)]
        }
        Command::Fail(span, cmd) => {
            let mut desugared = desugar_command(*cmd, parser, type_info, seminaive_transform)?;

            let last = desugared.pop().unwrap();
            desugared.push(NCommand::Fail(span, Box::new(last)));
            return Ok(desugared);
        }
        Command::Input { span, name, file } => {
            vec![NCommand::Input { span, name, file }]
        }
        Command::UserDefined(span, name, args) => {
            vec![NCommand::UserDefined(span, name, args)]
        }
    };

    Ok(res)
}

fn desugar_datatype(span: Span, name: String, variants: Vec<Variant>) -> Vec<NCommand> {
    vec![NCommand::Sort(span.clone(), name.clone(), None)]
        .into_iter()
        .chain(variants.into_iter().map(|variant| {
            NCommand::Function(FunctionDecl::constructor(
                variant.span,
                variant.name,
                Schema {
                    input: variant.types,
                    output: name.clone(),
                },
                variant.cost,
                variant.unextractable,
            ))
        }))
        .collect()
}

fn desugar_rewrite(
    ruleset: String,
    name: String,
    rewrite: Rewrite,
    subsume: bool,
    parser: &mut Parser,
) -> Vec<NCommand> {
    let span = rewrite.span.clone();
    let var = parser.symbol_gen.fresh("rewrite_var__");
    let mut head = Actions::singleton(Action::Union(
        span.clone(),
        Expr::Var(span.clone(), var.clone()),
        rewrite.rhs.clone(),
    ));
    if subsume {
        match &rewrite.lhs {
            Expr::Call(_, f, args) => {
                head.0.push(Action::Change(
                    span.clone(),
                    Change::Subsume,
                    f.clone(),
                    args.to_vec(),
                ));
            }
            _ => {
                panic!("Subsumed rewrite must have a function call on the lhs");
            }
        }
    }
    // make two rules- one to insert the rhs, and one to union
    // this way, the union rule can only be fired once,
    // which helps proofs not add too much info
    vec![NCommand::NormRule {
        rule: Rule {
            span: span.clone(),
            body: [Fact::Eq(
                span.clone(),
                Expr::Var(span, var),
                rewrite.lhs.clone(),
            )]
            .into_iter()
            .chain(rewrite.conditions.clone())
            .collect(),
            head,
            ruleset,
            name,
        },
    }]
}

fn desugar_birewrite(
    ruleset: String,
    name: String,
    rewrite: Rewrite,
    parser: &mut Parser,
) -> Vec<NCommand> {
    let span = rewrite.span.clone();
    let rw2 = Rewrite {
        span,
        lhs: rewrite.rhs.clone(),
        rhs: rewrite.lhs.clone(),
        conditions: rewrite.conditions.clone(),
    };
    desugar_rewrite(ruleset.clone(), format!("{name}=>"), rewrite, false, parser)
        .into_iter()
        .chain(desugar_rewrite(
            ruleset,
            format!("{name}<="),
            rw2,
            false,
            parser,
        ))
        .collect()
}

pub fn rule_name<Head, Leaf>(command: &GenericCommand<Head, Leaf>) -> String
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Hash + Display,
{
    command.to_string().replace('\"', "'")
}
