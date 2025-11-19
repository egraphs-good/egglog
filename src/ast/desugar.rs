use super::{Rewrite, Rule};
use crate::ast::{Action, Actions, Expr, Fact, Facts, GenericAction, ResolvedExpr, ResolvedExprExt, ResolvedFact};
use crate::constraint::Problem;
use crate::typechecking::TypeError;
use crate::util::IndexSet;
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
    // If there are no fresh! uses in the actions, return the rule unchanged.
    if collect_fresh_sites(&rule.head, type_info)?.is_empty() {
        return Ok(vec![NCommand::NormRule { rule }]);
    }

    let Rule {
        span,
        head,
        body,
        name,
        ruleset,
    } = rule;

    // Generate a unique constructor name using the parser's symbol generator
    let fresh_table_name = parser.symbol_gen.fresh("GeneratedFreshTable");

    let resolved_facts = typecheck_query_facts(parser, type_info, &body)?;
    let column_info = collect_query_columns(&body, &resolved_facts);

    let column_exprs: Vec<Expr> = column_info.iter().map(|(expr, _)| expr.clone()).collect();
    let mut constructor_inputs: Vec<String> = column_info
        .iter()
        .map(|(_, sort)| sort.name().to_string())
        .collect();
    constructor_inputs.push("i64".to_string());

    let fresh_sites = collect_fresh_sites(&head, type_info)?;

    let output_sort = fresh_sites[0].return_sort.clone();
    for site in &fresh_sites[1..] {
        if site.return_sort.name() != output_sort.name() {
            return Err(Error::TypeError(TypeError::Mismatch {
                expr: site.original_expr.clone(),
                expected: output_sort.clone(),
                actual: site.return_sort.clone(),
            }));
        }
    }

    let schema = Schema {
        input: constructor_inputs,
        output: output_sort.name().to_string(),
    };

    let constructor_command = NCommand::Function(FunctionDecl::constructor(
        span.clone(),
        fresh_table_name.clone(),
        schema,
        None,
        true,
    ));

    let mut rewrite_counter: i64 = 0;
    let rewritten_actions: Vec<_> = head
        .0
        .into_iter()
        .map(|action| {
            action.visit_exprs(&mut |expr| {
                rewrite_fresh_expr(expr, &fresh_table_name, &column_exprs, &mut rewrite_counter)
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

    Ok(vec![
        constructor_command,
        NCommand::NormRule {
            rule: rewritten_rule,
        },
    ])
}

fn typecheck_query_facts(
    parser: &mut Parser,
    type_info: &TypeInfo,
    body: &[Fact],
) -> Result<Vec<ResolvedFact>, Error> {
    let mut symbol_gen = parser.symbol_gen.clone();
    let (query, mapped_facts) = Facts(body.to_vec()).to_query(type_info, &mut symbol_gen);
    let mut problem = Problem::default();
    problem.add_query(&query, type_info)?;
    let assignment = problem
        .solve(|sort: &ArcSort| sort.name())
        .map_err(|err| Error::TypeError(err.to_type_error()))?;
    Ok(assignment.annotate_facts(&mapped_facts, type_info))
}

fn collect_query_columns(facts: &[Fact], resolved_facts: &[ResolvedFact]) -> Vec<(Expr, ArcSort)> {
    assert_eq!(facts.len(), resolved_facts.len());
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

#[derive(Clone)]
struct FreshSite {
    original_expr: Expr,
    return_sort: ArcSort,
}

fn collect_fresh_sites(actions: &Actions, type_info: &TypeInfo) -> Result<Vec<FreshSite>, Error> {
    let mut sites = Vec::new();
    for action in actions.iter() {
        collect_fresh_sites_action(action, type_info, &mut sites)?;
    }
    Ok(sites)
}

fn collect_fresh_sites_action(
    action: &GenericAction<String, String>,
    type_info: &TypeInfo,
    sites: &mut Vec<FreshSite>,
) -> Result<(), Error> {
    match action {
        GenericAction::Let(_, _, expr) => collect_fresh_sites_expr(expr, type_info, sites),
        GenericAction::Set(_, _, args, expr) => {
            for arg in args.iter() {
                collect_fresh_sites_expr(arg, type_info, sites)?;
            }
            collect_fresh_sites_expr(expr, type_info, sites)
        }
        GenericAction::Change(_, _, _, args) => {
            for arg in args.iter() {
                collect_fresh_sites_expr(arg, type_info, sites)?;
            }
            Ok(())
        }
        GenericAction::Union(_, lhs, rhs) => {
            collect_fresh_sites_expr(lhs, type_info, sites)?;
            collect_fresh_sites_expr(rhs, type_info, sites)
        }
        GenericAction::Expr(_, expr) => collect_fresh_sites_expr(expr, type_info, sites),
        GenericAction::Panic(_, _) => Ok(()),
    }
}

fn collect_fresh_sites_expr(
    expr: &Expr,
    type_info: &TypeInfo,
    sites: &mut Vec<FreshSite>,
) -> Result<(), Error> {
    match expr {
        Expr::Call(span, head, args) if head == "fresh!" => {
            if args.len() != 1 {
                return Err(Error::TypeError(TypeError::Arity {
                    expr: expr.clone(),
                    expected: 1,
                }));
            }
            let sort_expr = &args[0];
            let sort_name = match sort_expr {
                Expr::Var(_, name) => name,
                _ => {
                    let sort_display = sort_expr.to_string();
                    return Err(Error::TypeError(TypeError::UndefinedSort(
                        sort_display,
                        span.clone(),
                    )));
                }
            };
            let sort = type_info
                .get_sort_by_name(sort_name)
                .cloned()
                .ok_or_else(|| {
                    Error::TypeError(TypeError::UndefinedSort(sort_name.clone(), span.clone()))
                })?;
            sites.push(FreshSite {
                original_expr: expr.clone(),
                return_sort: sort,
            });
            Ok(())
        }
        Expr::Call(_, _, args) => {
            for arg in args.iter() {
                collect_fresh_sites_expr(arg, type_info, sites)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn rewrite_fresh_expr(
    expr: Expr,
    table_name: &str,
    column_exprs: &[Expr],
    counter: &mut i64,
) -> Expr {
    match expr {
        Expr::Call(span, head, args) => {
            if head == "fresh!" {
                let mut call_args = column_exprs.to_vec();
                call_args.push(Expr::Lit(span.clone(), Literal::Int(*counter)));
                *counter += 1;
                Expr::Call(span, table_name.to_string(), call_args)
            } else {
                let new_args = args
                    .into_iter()
                    .map(|arg| rewrite_fresh_expr(arg, table_name, column_exprs, counter))
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
