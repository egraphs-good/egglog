use std::cmp::max;

use crate::*;

pub(crate) type Fresh = dyn FnMut() -> Symbol;
pub(crate) type NewId = dyn FnMut() -> CommandId;

pub(crate) fn literal_name(desugar: &Desugar, literal: &Literal) -> Symbol {
    desugar.egraph.type_info.infer_literal(literal).name()
}

// Makes a function that gets fresh names by counting
// the max number of underscores in the program
pub(crate) fn make_get_fresh(program: &Vec<Command>) -> impl FnMut() -> Symbol {
    make_get_fresh_from_str(&ListDisplay(program, "\n").to_string())
}

fn make_get_fresh_from_str(program_str: &str) -> impl FnMut() -> Symbol {
    let mut max_underscores: usize = 0;
    let mut counter: i64 = -1;
    for char in program_str.chars() {
        if char == '_' {
            counter = max(counter, 0);
            counter += 1;
            max_underscores = max(max_underscores, counter as usize);
        } else {
            counter = -1;
        }
    }

    let underscores = "_".repeat(max_underscores + 1);
    let mut fcounter = 0;
    move || {
        fcounter += 1;
        format!("v{}{}", fcounter, underscores).into()
    }
}

fn desugar_datatype(name: Symbol, variants: Vec<Variant>) -> Vec<NCommand> {
    vec![NCommand::Sort(name, None)]
        .into_iter()
        .chain(variants.into_iter().map(|variant| {
            if variant.types.is_empty() {
                NCommand::Declare(variant.name, name)
            } else {
                NCommand::Function(FunctionDecl {
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
            }
        }))
        .collect()
}

fn desugar_rewrite(ruleset: Symbol, rewrite: &Rewrite, desugar: &mut Desugar) -> Vec<NCommand> {
    let var = Symbol::from("rewrite_var__");
    vec![NCommand::NormRule(
        ruleset,
        flatten_rule(
            Rule {
                body: [Fact::Eq(vec![Expr::Var(var), rewrite.lhs.clone()])]
                    .into_iter()
                    .chain(rewrite.conditions.clone())
                    .collect(),
                head: vec![Action::Union(Expr::Var(var), rewrite.rhs.clone())],
            },
            desugar,
        ),
    )]
}

fn desugar_birewrite(ruleset: Symbol, rewrite: &Rewrite, desugar: &mut Desugar) -> Vec<NCommand> {
    let rw2 = Rewrite {
        lhs: rewrite.rhs.clone(),
        rhs: rewrite.lhs.clone(),
        conditions: rewrite.conditions.clone(),
    };
    desugar_rewrite(ruleset, rewrite, desugar)
        .into_iter()
        .chain(desugar_rewrite(ruleset, &rw2, desugar))
        .collect()
}

fn expr_to_ssa(
    lhs: Symbol,
    expr: &Expr,
    desugar: &mut Desugar,
    res: &mut Vec<NormFact>,
) {
    match expr {
        Expr::Lit(l) => {
            res.push(NormFact::AssignLit(lhs, l.clone()));
        }
        Expr::Var(v) => {
            res.push(NormFact::ConstrainEq(lhs, *v));
        }
        Expr::Call(f, children) => {
            let mut new_children = vec![];
            for child in children {
                match child {
                    Expr::Var(v) => {
                        new_children.push(*v);
                    }
                    _ => {
                        let fresh = (desugar.get_fresh)();
                        expr_to_ssa(fresh, child, desugar, res);
                        new_children.push(fresh);
                    }
                }
            }
            res.push(NormFact::Assign(lhs, NormExpr::Call(*f, new_children)));
        }
    }
}


fn flatten_equalities(equalities: Vec<(Symbol, Expr)>, desugar: &mut Desugar) -> Vec<NormFact> {
    let mut res = vec![];

    for (lhs, rhs) in equalities {
        expr_to_ssa(
            lhs,
            &rhs,
            desugar,
            &mut res,
        );
    }

    res
}

fn flatten_facts(facts: &Vec<Fact>, desugar: &mut Desugar) -> Vec<NormFact> {
    let mut equalities = vec![];
    for fact in facts {
        match fact {
            Fact::Eq(args) => {
                assert!(args.len() == 2);
                let lhs = &args[0];
                let rhs = &args[1];
                if let Expr::Var(v) = lhs {
                    equalities.push((*v, rhs.clone()));
                } else if let Expr::Var(v) = rhs {
                    equalities.push((*v, lhs.clone()));
                } else {
                    let fresh = (desugar.get_fresh)();
                    equalities.push((fresh, lhs.clone()));
                    equalities.push((fresh, rhs.clone()));
                }
            }
            Fact::Fact(expr) => {
                equalities.push(((desugar.get_fresh)(), expr.clone()));
            }
        }
    }

    flatten_equalities(equalities, desugar)
}

fn expr_to_flat_actions(
    expr: &Expr,
    desugar: &mut Desugar,
    res: &mut Vec<NormAction>,
    memo: &mut HashMap<Expr, Symbol>,
) -> Symbol {
    if let Some(existing) = memo.get(expr) {
        return *existing;
    }
    let assign = (desugar.get_fresh)();
    match expr {
        Expr::Lit(l) => {
            res.push(NormAction::LetLit(assign, l.clone()));
        }
        Expr::Var(v) => {
            res.push(NormAction::LetVar(assign, *v));
        }
        Expr::Call(f, children) => {
            let mut new_children = vec![];
            for child in children {
                match child {
                    Expr::Var(v) => {
                        new_children.push(*v);
                    }
                    _ => {
                        let child = expr_to_flat_actions(child, desugar, res, memo);
                        new_children.push(child);
                    }
                }
            }
            res.push(NormAction::Let(assign, NormExpr::Call(*f, new_children)));
        }
    }
    memo.insert(expr.clone(), assign);
    assign
}

fn flatten_actions(actions: &Vec<Action>, desugar: &mut Desugar) -> Vec<NormAction> {
    let mut memo = Default::default();
    let mut add_expr = |expr: Expr, res: &mut Vec<NormAction>| -> Symbol {
        expr_to_flat_actions(&expr, desugar, res, &mut memo)
    };

    let mut res = vec![];

    for action in actions {
        match action {
            Action::Let(symbol, expr) => {
                let added = add_expr(expr.clone(), &mut res);
                res.push(NormAction::LetVar(*symbol, added));
            }
            Action::Set(symbol, exprs, rhs) => {
                let set = NormAction::Set(
                    NormExpr::Call(
                        *symbol,
                        exprs
                            .clone()
                            .into_iter()
                            .map(|ex| add_expr(ex, &mut res))
                            .collect(),
                    ),
                    add_expr(rhs.clone(), &mut res),
                );
                res.push(set);
            }
            Action::Delete(symbol, exprs) => {
                let del = NormAction::Delete(NormExpr::Call(
                    *symbol,
                    exprs
                        .clone()
                        .into_iter()
                        .map(|ex| add_expr(ex, &mut res))
                        .collect(),
                ));
                res.push(del);
            }
            Action::Union(lhs, rhs) => {
                let un = NormAction::Union(
                    add_expr(lhs.clone(), &mut res),
                    add_expr(rhs.clone(), &mut res),
                );
                res.push(un);
            }
            Action::Panic(msg) => {
                res.push(NormAction::Panic(msg.clone()));
            }
            Action::Expr(expr) => {
                add_expr(expr.clone(), &mut res);
            }
        };
    }

    res
}

fn give_unique_names(desugar: &mut Desugar, facts: Vec<NormFact>) -> Vec<NormFact> {
    let mut name_used: HashSet<Symbol> = Default::default();
    let mut constraints: Vec<NormFact> = Default::default();
    let mut res = vec![];
    for fact in facts {
        let mut name_used_immediately: HashSet<Symbol> = Default::default();
        let mut constraints_before = vec![];
        let new_fact = fact.map_def_use(&mut |var, is_def| {
            if is_def {
                if name_used.insert(var) {
                    name_used_immediately.insert(var);
                    var
                } else {
                    let fresh = (desugar.get_fresh)();
                    // typechecking BS- for primitives
                    // we need to define variables before they are used
                    if name_used_immediately.contains(&var) {
                        constraints.push(NormFact::ConstrainEq(fresh, var));
                    } else {
                        constraints_before.push(NormFact::ConstrainEq(fresh, var));
                    }
                    fresh
                }
            } else {
                var
            }
            });
        res.extend(constraints_before);
        res.push(new_fact);
    }

    res.extend(constraints);
    res
}

fn flatten_rule(rule: Rule, desugar: &mut Desugar) -> NormRule {
    let flat_facts = flatten_facts(&rule.body, desugar);
    let with_unique_names = give_unique_names(desugar, flat_facts);

    NormRule {
        head: flatten_actions(&rule.head, desugar),
        body: with_unique_names,
    }
}

pub struct Desugar<'a> {
    pub get_fresh: Box<Fresh>,
    pub get_new_id: Box<NewId>,
    pub egraph: &'a mut EGraph,
}

pub(crate) fn desugar_command(
    command: Command,
    desugar: &mut Desugar,
) -> Result<Vec<NormCommand>, Error> {
    let res = match command {
        Command::Function(fdecl) => {
            vec![NCommand::Function(fdecl)]
        }
        Command::Datatype { name, variants } => desugar_datatype(name, variants),
        Command::Declare(name, parent) => vec![NCommand::Declare(name, parent)],
        Command::Rewrite(ruleset, rewrite) => desugar_rewrite(ruleset, &rewrite, desugar),
        Command::BiRewrite(ruleset, rewrite) => desugar_birewrite(ruleset, &rewrite, desugar),
        Command::Include(file) => {
            let s = std::fs::read_to_string(&file)
                .unwrap_or_else(|_| panic!("Failed to read file {file}"));
            return desugar_commands(desugar.egraph.parse_program(&s)?, desugar);
        }
        Command::Rule(ruleset, rule) => {
            vec![NCommand::NormRule(ruleset, flatten_rule(rule, desugar))]
        }
        Command::Sort(sort, option) => vec![NCommand::Sort(sort, option)],
        // TODO ignoring cost for now
        Command::Define {
            name,
            expr,
            cost: _cost,
        } => {
            let mut commands = vec![];

            let mut actions = vec![];
            let fresh = expr_to_flat_actions(&expr, desugar, &mut actions, &mut Default::default());
            actions.push(NormAction::LetVar(name, fresh));
            for action in actions {
                commands.push(NCommand::NormAction(action));
            }
            commands
        }
        Command::AddRuleset(name) => vec![NCommand::AddRuleset(name)],
        Command::Action(action) => flatten_actions(&vec![action], desugar)
            .into_iter()
            .map(NCommand::NormAction)
            .collect(),
        Command::Run(run) => vec![NCommand::Run(run)],
        Command::Simplify { expr, config } => vec![NCommand::Simplify { expr, config }],
        Command::Calc(idents, exprs) => vec![NCommand::Calc(idents, exprs)],
        Command::Extract { variants, e } => {
            let fresh = (desugar.get_fresh)();
            flatten_actions(&vec![Action::Let(fresh, e)], desugar)
                .into_iter()
                .map(NCommand::NormAction)
                .chain(
                    vec![NCommand::Extract {
                        variants,
                        var: fresh,
                    }]
                    .into_iter(),
                )
                .collect()
        }
        Command::Check(check) => vec![NCommand::Check(check)],
        Command::Clear => vec![NCommand::Clear],
        Command::Print(symbol, size) => vec![NCommand::Print(symbol, size)],
        Command::PrintSize(symbol) => vec![NCommand::PrintSize(symbol)],
        Command::Output { file, exprs } => vec![NCommand::Output { file, exprs }],
        Command::Query(facts) => {
            vec![NCommand::Query(facts)]
        }
        Command::Push(num) => vec![NCommand::Push(num)],
        Command::Pop(num) => vec![NCommand::Pop(num)],
        Command::Fail(cmd) => {
            let mut desugared = desugar_command(*cmd, desugar)?;

            let last = desugared.pop().unwrap();
            desugared.push(NormCommand {
                metadata: last.metadata,
                command: NCommand::Fail(Box::new(last.command)),
            });
            return Ok(desugared);
        }
        Command::Input { .. } => {
            todo!("desugar input");
        }
    };

    Ok(res
        .into_iter()
        .map(|c| NormCommand {
            metadata: Metadata {
                id: (desugar.get_new_id)(),
            },
            command: c,
        })
        .collect())
}

fn make_get_new_id() -> impl FnMut() -> usize {
    let mut id = 0;
    move || {
        let res = id;
        id += 1;
        res
    }
}

pub(crate) fn desugar_program(
    egraph: &mut EGraph,
    program: Vec<Command>,
) -> Result<(Vec<NormCommand>, Desugar), Error> {
    let get_fresh = Box::new(make_get_fresh(&program));
    let mut desugar = Desugar {
        get_fresh,
        get_new_id: Box::new(make_get_new_id()),
        egraph,
    };
    let res = desugar_commands(program, &mut desugar)?;
    Ok((res, desugar))
}

pub(crate) fn desugar_commands(
    program: Vec<Command>,
    desugar: &mut Desugar,
) -> Result<Vec<NormCommand>, Error> {
    let mut res = vec![];

    for command in program {
        let desugared = desugar_command(command, desugar)?;
        res.extend(desugared);
    }
    Ok(res)
}
