use super::{Rewrite, Rule};
use crate::*;

/// Desugars a list of commands into the normalized form.
/// Gets rid of a bunch of syntactic sugar, but also
/// makes rules into a SSA-like format (see [`NormFact`]).
pub(crate) fn desugar_program(
    program: Vec<Command>,
    symbol_gen: &mut SymbolGen,
    seminaive_transform: bool,
) -> Result<Vec<NCommand>, Error> {
    let mut res = vec![];
    for command in program {
        let desugared = desugar_command(command, symbol_gen, seminaive_transform)?;
        res.extend(desugared);
    }
    Ok(res)
}

/// Desugars a single command into the normalized form.
/// Gets rid of a bunch of syntactic sugar, but also
/// makes rules into a SSA-like format (see [`NormFact`]).
pub(crate) fn desugar_command(
    command: Command,
    symbol_gen: &mut SymbolGen,
    seminaive_transform: bool,
) -> Result<Vec<NCommand>, Error> {
    let res = match command {
        Command::SetOption { name, value } => {
            vec![NCommand::SetOption { name, value }]
        }
        Command::Function(fdecl) => vec![NCommand::Function(fdecl)],
        Command::Relation {
            span,
            constructor,
            inputs,
        } => vec![NCommand::Function(FunctionDecl::relation(
            span,
            constructor,
            inputs,
        ))],
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
                let name = datatype.1;
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
                    res.push(NCommand::Function(FunctionDecl {
                        name: variant.name,
                        schema: Schema {
                            input: variant.types,
                            output: datatype,
                        },
                        merge: None,
                        merge_action: Actions::default(),
                        default: None,
                        cost: variant.cost,
                        unextractable: false,
                        ignore_viz: false,
                        span: variant.span,
                    }));
                }
            }

            res
        }
        Command::Rewrite(ruleset, rewrite, subsume) => {
            desugar_rewrite(ruleset, rewrite_name(&rewrite).into(), &rewrite, subsume)
        }
        Command::BiRewrite(ruleset, rewrite) => {
            desugar_birewrite(ruleset, rewrite_name(&rewrite).into(), &rewrite)
        }
        Command::Include(span, file) => {
            let s = std::fs::read_to_string(&file)
                .unwrap_or_else(|_| panic!("{span} Failed to read file {file}"));
            return desugar_program(
                parse_program(Some(file), &s)?,
                symbol_gen,
                seminaive_transform,
            );
        }
        Command::Rule {
            ruleset,
            mut name,
            rule,
        } => {
            if name == "".into() {
                name = rule.to_string().replace('\"', "'").into();
            }

            let mut result = vec![NCommand::NormRule {
                ruleset,
                name,
                rule: rule.clone(),
            }];

            if seminaive_transform {
                if let Some(new_rule) = add_semi_naive_rule(symbol_gen, rule) {
                    result.push(NCommand::NormRule {
                        ruleset,
                        name,
                        rule: new_rule,
                    });
                }
            }

            result
        }
        Command::Sort(span, sort, option) => vec![NCommand::Sort(span, sort, option)],
        Command::AddRuleset(name) => vec![NCommand::AddRuleset(name)],
        Command::UnstableCombinedRuleset(name, subrulesets) => {
            vec![NCommand::UnstableCombinedRuleset(name, subrulesets)]
        }
        Command::Action(action) => vec![NCommand::CoreAction(action)],
        Command::Simplify {
            span,
            expr,
            schedule,
        } => desugar_simplify(&expr, &schedule, span, symbol_gen),
        Command::RunSchedule(sched) => {
            vec![NCommand::RunSchedule(sched.clone())]
        }
        Command::PrintOverallStatistics => {
            vec![NCommand::PrintOverallStatistics]
        }
        Command::QueryExtract {
            span,
            variants,
            expr,
        } => {
            let variants = Expr::Lit(span.clone(), Literal::Int(variants.try_into().unwrap()));
            if let Expr::Var(..) = expr {
                // (extract {v} {variants})
                vec![NCommand::CoreAction(Action::Extract(
                    span.clone(),
                    expr,
                    variants,
                ))]
            } else {
                // (check {expr})
                // (ruleset {fresh_ruleset})
                // (rule ((= {fresh} {expr}))
                //       ((extract {fresh} {variants}))
                //       :ruleset {fresh_ruleset})
                // (run {fresh_ruleset} 1)
                let fresh = symbol_gen.fresh(&"desugar_qextract_var".into());
                let fresh_ruleset = symbol_gen.fresh(&"desugar_qextract_ruleset".into());
                let fresh_rulename = symbol_gen.fresh(&"desugar_qextract_rulename".into());
                let rule = Rule {
                    span: span.clone(),
                    body: vec![Fact::Eq(
                        span.clone(),
                        vec![Expr::Var(span.clone(), fresh), expr.clone()],
                    )],
                    head: Actions::singleton(Action::Extract(
                        span.clone(),
                        Expr::Var(span.clone(), fresh),
                        variants,
                    )),
                };
                vec![
                    NCommand::Check(span.clone(), vec![Fact::Fact(expr.clone())]),
                    NCommand::AddRuleset(fresh_ruleset),
                    NCommand::NormRule {
                        name: fresh_rulename,
                        ruleset: fresh_ruleset,
                        rule,
                    },
                    NCommand::RunSchedule(Schedule::Run(
                        span.clone(),
                        RunConfig {
                            ruleset: fresh_ruleset,
                            until: None,
                        },
                    )),
                ]
            }
        }
        Command::Check(span, facts) => vec![NCommand::Check(span, facts)],
        Command::PrintFunction(span, symbol, size) => {
            vec![NCommand::PrintTable(span, symbol, size)]
        }
        Command::PrintSize(span, symbol) => vec![NCommand::PrintSize(span, symbol)],
        Command::Output { span, file, exprs } => vec![NCommand::Output { span, file, exprs }],
        Command::Push(num) => {
            vec![NCommand::Push(num)]
        }
        Command::Pop(span, num) => {
            vec![NCommand::Pop(span, num)]
        }
        Command::Fail(span, cmd) => {
            let mut desugared = desugar_command(*cmd, symbol_gen, seminaive_transform)?;

            let last = desugared.pop().unwrap();
            desugared.push(NCommand::Fail(span, Box::new(last)));
            return Ok(desugared);
        }
        Command::Input { span, name, file } => {
            vec![NCommand::Input { span, name, file }]
        }
    };

    Ok(res)
}

fn desugar_datatype(span: Span, name: Symbol, variants: Vec<Variant>) -> Vec<NCommand> {
    vec![NCommand::Sort(span.clone(), name, None)]
        .into_iter()
        .chain(variants.into_iter().map(|variant| {
            NCommand::Function(FunctionDecl {
                name: variant.name,
                schema: Schema {
                    input: variant.types,
                    output: name,
                },
                merge: None,
                merge_action: Actions::default(),
                default: None,
                cost: variant.cost,
                unextractable: false,
                ignore_viz: false,
                span: variant.span,
            })
        }))
        .collect()
}

fn desugar_rewrite(
    ruleset: Symbol,
    name: Symbol,
    rewrite: &Rewrite,
    subsume: bool,
) -> Vec<NCommand> {
    let span = rewrite.span.clone();
    let var = Symbol::from("rewrite_var__");
    let mut head = Actions::singleton(Action::Union(
        span.clone(),
        Expr::Var(span.clone(), var),
        rewrite.rhs.clone(),
    ));
    if subsume {
        match &rewrite.lhs {
            Expr::Call(_, f, args) => {
                head.0.push(Action::Change(
                    span.clone(),
                    Change::Subsume,
                    *f,
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
        ruleset,
        name,
        rule: Rule {
            span: span.clone(),
            body: [Fact::Eq(
                span.clone(),
                vec![Expr::Var(span, var), rewrite.lhs.clone()],
            )]
            .into_iter()
            .chain(rewrite.conditions.clone())
            .collect(),
            head,
        },
    }]
}

fn desugar_birewrite(ruleset: Symbol, name: Symbol, rewrite: &Rewrite) -> Vec<NCommand> {
    let span = rewrite.span.clone();
    let rw2 = Rewrite {
        span,
        lhs: rewrite.rhs.clone(),
        rhs: rewrite.lhs.clone(),
        conditions: rewrite.conditions.clone(),
    };
    desugar_rewrite(ruleset, format!("{}=>", name).into(), rewrite, false)
        .into_iter()
        .chain(desugar_rewrite(
            ruleset,
            format!("{}<=", name).into(),
            &rw2,
            false,
        ))
        .collect()
}

// TODO(yz): we can delete this code once we enforce that all rule bodies cannot read the database (except EqSort).
fn add_semi_naive_rule(symbol_gen: &mut SymbolGen, rule: Rule) -> Option<Rule> {
    let mut new_rule = rule;
    // Whenever an Let(_, expr@Call(...)) or Set(_, expr@Call(...)) is present in action,
    // an additional seminaive rule should be created.
    // Moreover, for each such expr, expr and all variable definitions that it relies on should be moved to trigger.
    let mut new_head_atoms = vec![];
    let mut add_new_rule = false;

    let mut var_set = HashSet::default();
    for head_slice in new_rule.head.0.iter_mut().rev() {
        match head_slice {
            Action::Set(span, _, _, expr) => {
                var_set.extend(expr.vars());
                if let Expr::Call(..) = expr {
                    add_new_rule = true;

                    let fresh_symbol = symbol_gen.fresh(&"desugar_snrule".into());
                    let fresh_var = Expr::Var(span.clone(), fresh_symbol);
                    let expr = std::mem::replace(expr, fresh_var.clone());
                    new_head_atoms.push(Fact::Eq(span.clone(), vec![fresh_var, expr]));
                };
            }
            Action::Let(span, symbol, expr) if var_set.contains(symbol) => {
                var_set.extend(expr.vars());
                if let Expr::Call(..) = expr {
                    add_new_rule = true;

                    let var = Expr::Var(span.clone(), *symbol);
                    new_head_atoms.push(Fact::Eq(span.clone(), vec![var, expr.clone()]));
                }
            }
            _ => (),
        }
    }

    if add_new_rule {
        new_rule.body.extend(new_head_atoms.into_iter().rev());
        // remove all let action
        new_rule.head.0.retain_mut(
            |action| !matches!(action, Action::Let(_ann, var, _) if var_set.contains(var)),
        );
        log::debug!("Added a semi-naive desugared rule:\n{}", new_rule);
        Some(new_rule)
    } else {
        None
    }
}

fn desugar_simplify(
    expr: &Expr,
    schedule: &Schedule,
    span: Span,
    symbol_gen: &mut SymbolGen,
) -> Vec<NCommand> {
    let mut res = vec![NCommand::Push(1)];
    let lhs = symbol_gen.fresh(&"desugar_simplify".into());
    res.push(NCommand::CoreAction(Action::Let(
        span.clone(),
        lhs,
        expr.clone(),
    )));
    res.push(NCommand::RunSchedule(schedule.clone()));
    res.extend(
        desugar_command(
            Command::QueryExtract {
                span: span.clone(),
                variants: 0,
                expr: Expr::Var(span.clone(), lhs),
            },
            symbol_gen,
            false,
        )
        .unwrap(),
    );

    res.push(NCommand::Pop(span, 1));
    res
}

pub(crate) fn rewrite_name(rewrite: &Rewrite) -> String {
    rewrite.to_string().replace('\"', "'")
}
