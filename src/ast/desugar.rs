use super::{Rewrite, Rule};
use crate::ast::{Action, Actions, Expr, Fact};
use crate::*;
use egglog_ast::span::Span;

/// Desugars a single command, removing syntactic sugar.
pub(crate) fn desugar_command(
    command: Command,
    parser: &mut Parser,
    proof_testing: bool,
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
            term_constructor,
        } => {
            let mut fdecl =
                FunctionDecl::constructor(span, name, schema, cost, unextractable, true);
            fdecl.term_constructor = term_constructor;
            vec![NCommand::Function(fdecl)]
        }
        Command::Relation { span, name, inputs } => desugar_relation(parser, span, name, inputs),
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
                        true,
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
        Command::Include(_span, _file) => {
            unreachable!("Include commands should be expanded before desugaring")
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
        Command::Check(span, facts) => {
            if proof_testing {
                desugar_prove(parser, span.clone(), facts.clone())
            } else {
                vec![NCommand::Check(span, facts)]
            }
        }
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
            let mut desugared = desugar_command(*cmd, parser, proof_testing)?;

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
        Command::Prove(span, query) => desugar_prove(parser, span, query),
        Command::ProveExists(span, constructor) => {
            vec![NCommand::ProveExists(span, constructor)]
        }
    };

    Ok(res)
}

/// Desugars a `prove` command into egglog commands.
/// For example, `(prove (= a b))` becomes:
/// ```text
/// (sort ExistsSort)
/// (function ExistsConstructor () ExistsSort)
/// (ruleset exists)
/// (rule ((= a b))
///       ((ExistsConstructor))
///       :ruleset exists
///       :name "prove_exists_rule")
/// (run exists)
/// (prove-exists ExistsConstructor)
/// ```
/// This creates a fresh constructor that can only be created if the query holds.
/// Then `prove-exists` extracts a proof that the constructor exists.
fn desugar_prove(parser: &mut Parser, span: Span, query: Vec<Fact>) -> Vec<NCommand> {
    let fresh_sort = parser.symbol_gen.fresh("ExistsSort");
    let constructor_name = parser.symbol_gen.fresh("ExistsConstructor");
    let ruleset = parser.symbol_gen.fresh("exists");
    let name = parser.symbol_gen.fresh("prove_exists_rule");
    vec![
        NCommand::Sort(span.clone(), fresh_sort.clone(), None),
        NCommand::Function(FunctionDecl::constructor(
            span.clone(),
            constructor_name.clone(),
            Schema {
                input: vec![],
                output: fresh_sort.clone(),
            },
            None,
            false,
            false,
        )),
        NCommand::AddRuleset(span.clone(), ruleset.clone()),
        // rule that constructs the new constructor
        NCommand::NormRule {
            rule: Rule {
                span: span.clone(),
                body: query,
                head: Actions::singleton(Action::Expr(
                    span.clone(),
                    Expr::Call(span.clone(), constructor_name.clone(), vec![]),
                )),
                ruleset: ruleset.clone(),
                name,
            },
        },
        // run the rule
        NCommand::RunSchedule(GenericSchedule::Run(
            span.clone(),
            GenericRunConfig {
                ruleset,
                until: None,
            },
        )),
        // get a proof for the constructor
        NCommand::ProveExists(span, constructor_name),
    ]
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
                true,
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

/// Desugar relation by making a new sort and a constructor for it.
fn desugar_relation(
    parser: &mut Parser,
    span: Span,
    name: String,
    inputs: Vec<String>,
) -> Vec<NCommand> {
    let dashes_removed = name.replace('-', "");
    let fresh_sort = parser.symbol_gen.fresh(&format!("{dashes_removed}Sort"));
    vec![
        NCommand::Sort(span.clone(), fresh_sort.clone(), None),
        NCommand::Function(FunctionDecl::constructor(
            span,
            name,
            Schema {
                input: inputs,
                output: fresh_sort,
            },
            None,
            false,
            false,
        )),
    ]
}

pub fn rule_name<Head, Leaf>(command: &GenericCommand<Head, Leaf>) -> String
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Hash + Display,
{
    command.to_string().replace('\"', "'")
}
