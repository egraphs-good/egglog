use crate::*;

pub(crate) type Fresh = dyn FnMut() -> Symbol;

fn desugar_datatype(name: Symbol, variants: Vec<Variant>) -> Vec<NormCommand> {
    vec![NormCommand::Sort(name, None)]
        .into_iter()
        .chain(variants.into_iter().map(|variant| {
            NormCommand::Function(FunctionDecl {
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

fn desugar_rewrite(
    ruleset: Symbol,
    rewrite: &Rewrite,
    globals: &HashSet<Symbol>,
) -> Vec<NormCommand> {
    let var = Symbol::from("rewrite_var__");
    vec![NormCommand::NormRule(
        ruleset,
        flatten_rule(
            Rule {
                body: [Fact::Eq(vec![Expr::Var(var), rewrite.lhs.clone()])]
                    .into_iter()
                    .chain(rewrite.conditions.clone())
                    .collect(),
                head: vec![Action::Union(Expr::Var(var), rewrite.rhs.clone())],
            },
            globals,
        ),
    )]
}

fn desugar_birewrite(
    ruleset: Symbol,
    rewrite: &Rewrite,
    globals: &HashSet<Symbol>,
) -> Vec<NormCommand> {
    let rw2 = Rewrite {
        lhs: rewrite.rhs.clone(),
        rhs: rewrite.lhs.clone(),
        conditions: rewrite.conditions.clone(),
    };
    desugar_rewrite(ruleset, rewrite, globals)
        .into_iter()
        .chain(desugar_rewrite(ruleset, &rw2, globals))
        .collect()
}

// TODO use an egraph to perform the Norm translation without introducing
// so many fresh variables
fn expr_to_ssa(
    expr: &Expr,
    get_fresh: &mut Fresh,
    var_used: &mut HashSet<Symbol>,
    var_just_used: &mut HashSet<Symbol>,
    res: &mut Vec<NormFact>,
    constraints: &mut Vec<NormFact>,
) -> Symbol {
    match expr {
        Expr::Lit(l) => {
            let fresh = get_fresh();
            res.push(NormFact::AssignLit(fresh, l.clone()));
            let fresh2 = get_fresh();
            res.push(NormFact::ConstrainEq(fresh2, fresh));
            fresh2
        }
        Expr::Var(v) => {
            if var_used.insert(*v) {
                var_just_used.insert(*v);
                *v
            } else {
                let fresh = get_fresh();
                // logic to satisfy typechecker
                // if we used the variable in this recurrence, add the constraint afterwards
                if var_just_used.contains(v) {
                    constraints.push(NormFact::ConstrainEq(fresh, *v));
                // otherwise add the constrain immediately so we have the type
                } else {
                    res.push(NormFact::ConstrainEq(fresh, *v));
                }
                fresh
            }
        }
        Expr::Call(f, children) => {
            let mut new_children = vec![];
            for child in children {
                new_children.push(expr_to_ssa(
                    child,
                    get_fresh,
                    var_used,
                    var_just_used,
                    res,
                    constraints,
                ));
            }
            let fresh = get_fresh();
            res.push(NormFact::Assign(fresh, NormExpr::Call(*f, new_children)));
            let fresh2 = get_fresh();
            res.push(NormFact::ConstrainEq(fresh2, fresh));
            fresh2
        }
    }
}

fn ssa_valid_expr(expr: &NormExpr, var_used: &mut HashSet<Symbol>) -> bool {
    match expr {
        NormExpr::Call(_, children) => {
            for child in children {
                if !var_used.insert(*child) {
                    return false;
                }
            }
        }
    }
    true
}

pub(crate) fn assert_ssa_valid(facts: &Vec<NormFact>, actions: &Vec<NormAction>) -> bool {
    //println!("assert_ssa_valid: {:?}", facts);
    let mut var_used: HashSet<Symbol> = Default::default();
    let mut var_used_constraints: HashSet<Symbol> = Default::default();
    for fact in facts {
        match fact {
            NormFact::Assign(v, expr) => {
                if !var_used.insert(*v) {
                    panic!("invalid Norm variable: {:?}", v);
                }

                if !ssa_valid_expr(expr, &mut var_used) {
                    panic!("invalid Norm fact: {:?}", expr);
                }
            }
            NormFact::ConstrainEq(v, v2) => {
                let b1 = var_used_constraints.insert(*v);
                let b2 = var_used_constraints.insert(*v2);
                // any constraints on variables are valid, but one needs to be defined
                if !var_used.contains(v) && !var_used.contains(v2) && b1 && b2 {
                    panic!("invalid Norm constraint: {:?} = {:?}", v, v2);
                }
            }
            NormFact::AssignLit(v, _) => {
                if !var_used.insert(*v) {
                    panic!("invalid Norm variable: {:?}", v);
                }
            }
        }
    }

    var_used.extend(var_used_constraints);

    let mut fdefuse = |var, isdef| {
        if isdef {
            if !var_used.insert(var) {
                panic!("invalid Norm variable: {:?}", var);
            }
        } else if !var_used.contains(&var) {
            panic!("invalid Norm variable: {:?}", var);
        }
        var
    };
    for action in actions {
        action.map_def_use(&mut fdefuse);
    }

    true
}

fn flatten_equalities(equalities: Vec<(Symbol, Expr)>, get_fresh: &mut Fresh) -> Vec<NormFact> {
    let mut res = vec![];

    let mut var_used = Default::default();
    for (lhs, rhs) in equalities {
        let mut constraints = vec![];
        let result = expr_to_ssa(
            &rhs,
            get_fresh,
            &mut var_used,
            &mut Default::default(),
            &mut res,
            &mut constraints,
        );
        res.extend(constraints);

        var_used.insert(lhs);
        res.push(NormFact::ConstrainEq(lhs, result));
    }

    res
}

fn flatten_facts(facts: &Vec<Fact>, get_fresh: &mut Fresh) -> Vec<NormFact> {
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
                    let fresh = get_fresh();
                    equalities.push((fresh, lhs.clone()));
                    equalities.push((fresh, rhs.clone()));
                }
            }
            Fact::Fact(expr) => {
                equalities.push((get_fresh(), expr.clone()));
            }
        }
    }

    flatten_equalities(equalities, get_fresh)
}

fn expr_to_flat_actions(
    assign: Symbol,
    expr: &Expr,
    get_fresh: &mut Fresh,
    res: &mut Vec<NormAction>,
) {
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
                let fresh = get_fresh();
                expr_to_flat_actions(fresh, child, get_fresh, res);
                new_children.push(fresh);
            }
            res.push(NormAction::Let(assign, NormExpr::Call(*f, new_children)));
        }
    }
}

fn flatten_actions(actions: &Vec<Action>, get_fresh: &mut Fresh) -> Vec<NormAction> {
    let mut add_expr = |expr: Expr, res: &mut Vec<NormAction>| {
        let fresh = get_fresh();
        expr_to_flat_actions(fresh, &expr, get_fresh, res);
        fresh
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
                    *symbol,
                    exprs
                        .clone()
                        .into_iter()
                        .map(|ex| add_expr(ex, &mut res))
                        .collect(),
                    add_expr(rhs.clone(), &mut res),
                );
                res.push(set);
            }
            Action::Delete(symbol, exprs) => {
                let del = NormAction::Delete(
                    *symbol,
                    exprs
                        .clone()
                        .into_iter()
                        .map(|ex| add_expr(ex, &mut res))
                        .collect(),
                );
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

// In egglog, you are allowed to refer to variables
// (which desugar to functions with no args)
// without parenthesis.
// This fixes that so normal translation is easier.
fn parenthesize_globals(rule: Rule, globals: &HashSet<Symbol>) -> Rule {
    rule.map_exprs(&mut |e| {
        e.map(&mut |e| match e {
            Expr::Var(v) if globals.contains(v) => Expr::Call(*v, vec![]),
            _ => e.clone(),
        })
    })
}

fn flatten_rule(rule_in: Rule, globals: &HashSet<Symbol>) -> NormRule {
    let rule = parenthesize_globals(rule_in, globals);
    let mut varcount = 0;
    let mut get_fresh = move || {
        varcount += 1;
        Symbol::from(format!("fvar{}__", varcount))
    };

    let res = NormRule {
        head: flatten_actions(&rule.head, &mut get_fresh),
        body: flatten_facts(&rule.body, &mut get_fresh),
    };
    assert_ssa_valid(&res.body, &res.head);
    res
}

pub(crate) struct Desugar {
    pub(crate) globals: HashSet<Symbol>,
    pub(crate) get_fresh: Box<Fresh>,
}

pub(crate) fn desugar_command(
    egraph: &EGraph,
    command: Command,
    desugar: &mut Desugar,
) -> Result<Vec<NormCommand>, Error> {
    Ok(match command {
        Command::Function(fdecl) => {
            vec![NormCommand::Function(fdecl)]
        }
        Command::Datatype { name, variants } => desugar_datatype(name, variants),
        Command::Rewrite(ruleset, rewrite) => desugar_rewrite(ruleset, &rewrite, &desugar.globals),
        Command::BiRewrite(ruleset, rewrite) => {
            desugar_birewrite(ruleset, &rewrite, &desugar.globals)
        }
        Command::Include(file) => {
            let s = std::fs::read_to_string(&file)
                .unwrap_or_else(|_| panic!("Failed to read file {file}"));
            desugar_commands(egraph, egraph.parse_program(&s)?, desugar)?
        }
        Command::Rule(ruleset, rule) => vec![NormCommand::NormRule(
            ruleset,
            flatten_rule(rule, &desugar.globals),
        )],
        Command::Sort(sort, option) => vec![NormCommand::Sort(sort, option)],
        // TODO ignoring cost for now
        Command::Define {
            name,
            expr,
            cost: _cost,
        } => {
            let mut commands = vec![];

            let mut actions = vec![];
            expr_to_flat_actions(name, &expr, &mut desugar.get_fresh, &mut actions);
            for action in actions {
                commands.push(NormCommand::NormAction(action));
            }
            commands
        }
        Command::AddRuleset(name) => vec![NormCommand::AddRuleset(name)],
        Command::Action(action) => flatten_actions(&vec![action], &mut desugar.get_fresh)
            .into_iter()
            .map(NormCommand::NormAction)
            .collect(),
        Command::Run(run) => vec![NormCommand::Run(run)],
        Command::Simplify { expr, config } => vec![NormCommand::Simplify { expr, config }],
        Command::Calc(idents, exprs) => vec![NormCommand::Calc(idents, exprs)],
        Command::Extract { variants, e } => {
            let fresh = (desugar.get_fresh)();
            flatten_actions(&vec![Action::Let(fresh, e)], &mut desugar.get_fresh)
                .into_iter()
                .map(NormCommand::NormAction)
                .chain(
                    vec![NormCommand::Extract {
                        variants,
                        var: fresh,
                    }]
                    .into_iter(),
                )
                .collect()
        }
        Command::Check(check) => vec![NormCommand::Check(check)],
        Command::Clear => vec![NormCommand::Clear],
        Command::Print(symbol, size) => vec![NormCommand::Print(symbol, size)],
        Command::PrintSize(symbol) => vec![NormCommand::PrintSize(symbol)],
        Command::Output { file, exprs } => vec![NormCommand::Output { file, exprs }],
        Command::Query(facts) => {
            vec![NormCommand::Query(facts)]
        }
        Command::Push(num) => vec![NormCommand::Push(num)],
        Command::Pop(num) => vec![NormCommand::Pop(num)],
        Command::Fail(cmd) => {
            let mut desugared = desugar_command(egraph, *cmd, desugar)?;

            let last = desugared.pop();
            desugared.push(NormCommand::Fail(Box::new(last.unwrap())));
            desugared
        }
        Command::Input { .. } => {
            todo!("desugar input");
        }
    })
}

pub(crate) fn desugar_program(
    egraph: &EGraph,
    program: Vec<Command>,
) -> Result<Vec<NormCommand>, Error> {
    let mut counter = 0;
    desugar_commands(
        egraph,
        program,
        &mut Desugar {
            globals: Default::default(),
            get_fresh: Box::new(move || {
                counter += 1;
                Symbol::from(format!("var{}__", counter))
            }),
        },
    )
}

pub(crate) fn desugar_commands(
    egraph: &EGraph,
    program: Vec<Command>,
    desugar: &mut Desugar,
) -> Result<Vec<NormCommand>, Error> {
    let mut res = vec![];

    for command in program {
        let desugared = desugar_command(egraph, command, desugar)?;

        for newcommand in &desugared {
            match newcommand {
                NormCommand::NormAction(NormAction::Let(name, _))
                | NormCommand::NormAction(NormAction::LetLit(name, _))
                | NormCommand::NormAction(NormAction::LetVar(name, _)) => {
                    desugar.globals.insert(*name);
                }
                NormCommand::Function(fdecl) => {
                    // add to globals if it has no arguments
                    if fdecl.schema.input.is_empty() {
                        desugar.globals.insert(fdecl.name);
                    }
                }
                _ => (),
            }
        }

        res.extend(desugared);
    }
    Ok(res)
}
