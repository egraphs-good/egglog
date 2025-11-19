use super::{Rewrite, Rule};
use crate::ast::{Action, Actions, Expr, Fact, ResolvedExpr, ResolvedExprExt, ResolvedFact};
use crate::typechecking::TypeError;
use crate::util::{IndexMap, IndexSet};
use crate::*;
use egglog_ast::generic_ast::Literal;
use egglog_ast::span::Span;

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
            desugar_rule(rule, parser, type_info)?
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

fn desugar_rule(
    rule: Rule,
    parser: &mut Parser,
    type_info: &TypeInfo,
) -> Result<Vec<NCommand>, Error> {
    let fresh_sorts = collect_fresh_sorts(&rule.head, type_info)?;
    // If there are no fresh! uses in the actions, return the rule unchanged.
    if fresh_sorts.is_empty() {
        return Ok(vec![NCommand::NormRule { rule }]);
    }

    let Rule {
        span,
        head,
        body,
        name,
        ruleset,
    } = rule;

    // Typecheck the query to collect column schema
    let resolved_facts = type_info
        .typecheck_facts(&mut parser.symbol_gen, &body)
        .map_err(Error::TypeError)?;
    let column_info = collect_query_columns(&resolved_facts);

    let column_exprs: Vec<Expr> = column_info.iter().map(|(expr, _)| expr.clone()).collect();
    let mut constructor_inputs: Vec<String> = column_info
        .iter()
        .map(|(_, sort)| sort.name().to_string())
        .collect();
    constructor_inputs.push("i64".to_string());

    let mut table_names: IndexMap<String, String> = IndexMap::default();
    let mut constructor_decls: Vec<NCommand> = Vec::new();

    for (sort_name, _sort) in fresh_sorts.iter() {
        if !table_names.contains_key(sort_name) {
            let fresh_table_name = parser.symbol_gen.fresh("GeneratedFreshTable");
            table_names.insert(sort_name.clone(), fresh_table_name.clone());
            let schema = Schema {
                input: constructor_inputs.clone(),
                output: sort_name.clone(),
            };
            constructor_decls.push(NCommand::Function(FunctionDecl::constructor(
                span.clone(),
                fresh_table_name,
                schema,
                None,
                true,
            )));
        }
    }

    let mut counters: IndexMap<String, i64> = IndexMap::default();
    let rewritten_actions: Vec<_> = head
        .0
        .into_iter()
        .map(|action| {
            action.visit_exprs(&mut |expr| {
                rewrite_fresh_expr(expr, &table_names, &column_exprs, &mut counters)
            })
        })
        .collect();

    let rewritten_rule = Rule {
        span,
        head: Actions::new(rewritten_actions),
        body,
        name,
        ruleset,
    };

    let mut out = constructor_decls;
    out.push(NCommand::NormRule {
        rule: rewritten_rule,
    });
    Ok(out)
}

fn collect_query_columns(resolved_facts: &[ResolvedFact]) -> Vec<(Expr, ArcSort)> {
    // Gather all resolved subexpressions in pre-order
    let mut resolved_nodes: Vec<ResolvedExpr> = Vec::new();
    for resolved_fact in resolved_facts.iter() {
        match resolved_fact {
            ResolvedFact::Fact(e) => e.collect_subexprs_preorder(&mut resolved_nodes),
            ResolvedFact::Eq(_, e1, e2) => {
                e1.collect_subexprs_preorder(&mut resolved_nodes);
                e2.collect_subexprs_preorder(&mut resolved_nodes);
            }
        }
    }

    // Convert to surface Expr and deduplicate using an IndexSet to preserve order
    let mut seen_surface: IndexSet<Expr> = IndexSet::default();
    let mut out: Vec<(Expr, ArcSort)> = Vec::new();
    for re in resolved_nodes.into_iter() {
        let e = re.to_surface();
        if seen_surface.insert(e.clone()) {
            out.push((e, re.output_type()));
        }
    }
    out
}

fn collect_fresh_sorts(
    actions: &Actions,
    type_info: &TypeInfo,
) -> Result<IndexMap<String, ArcSort>, Error> {
    let mut sorts = IndexMap::default();
    let mut error: Option<Error> = None;

    for action in actions.iter().cloned() {
        action.visit_exprs(&mut |expr: Expr| {
            if error.is_some() {
                return expr;
            }
            if let Expr::Call(span, head, args) = &expr {
                if head == "fresh!" {
                    if args.len() != 1 {
                        error = Some(Error::TypeError(TypeError::Arity {
                            expr: expr.clone(),
                            expected: 1,
                        }));
                        return expr;
                    }
                    match &args[0] {
                        Expr::Var(_, sort_name) => match type_info.get_sort_by_name(sort_name) {
                            Some(sort) => {
                                sorts
                                    .entry(sort.name().to_string())
                                    .or_insert_with(|| sort.clone());
                            }
                            None => {
                                error = Some(Error::TypeError(TypeError::UndefinedSort(
                                    sort_name.clone(),
                                    span.clone(),
                                )));
                            }
                        },
                        other => {
                            let sort_display = other.to_string();
                            error = Some(Error::TypeError(TypeError::UndefinedSort(
                                sort_display,
                                span.clone(),
                            )));
                        }
                    }
                }
            }
            expr
        });
        if error.is_some() {
            break;
        }
    }

    error.map_or(Ok(sorts), Err)
}

fn rewrite_fresh_expr(
    expr: Expr,
    table_names: &IndexMap<String, String>,
    column_exprs: &[Expr],
    counters: &mut IndexMap<String, i64>,
) -> Expr {
    match expr {
        Expr::Call(span, head, args) => {
            if head == "fresh!" {
                let sort_name = match args.as_slice() {
                    [Expr::Var(_, sort)] => sort.clone(),
                    _ => return Expr::Call(span, head, args),
                };
                if let Some(table_name) = table_names.get(&sort_name) {
                    let counter = counters.entry(sort_name.clone()).or_insert(0);
                    let mut call_args = column_exprs.to_vec();
                    call_args.push(Expr::Lit(span.clone(), Literal::Int(*counter)));
                    *counter += 1;
                    Expr::Call(span, table_name.clone(), call_args)
                } else {
                    Expr::Call(span, head, args)
                }
            } else {
                let new_args = args
                    .into_iter()
                    .map(|arg| rewrite_fresh_expr(arg, table_names, column_exprs, counters))
                    .collect();
                Expr::Call(span, head, new_args)
            }
        }
        _ => expr,
    }
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
