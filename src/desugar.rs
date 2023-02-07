use crate::*;

pub(crate) type Fresh = dyn FnMut() -> Symbol;

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

fn desugar_rewrite(ruleset: Symbol, rewrite: &Rewrite, globals: &HashSet<Symbol>) -> Vec<Command> {
    let var = Symbol::from("rewrite_var__");
    vec![Command::FlatRule(
        ruleset,
        flatten_rule(parenthesize_globals(Rule {
            body: [Fact::Eq(vec![Expr::Var(var), rewrite.lhs.clone()])]
                .into_iter()
                .chain(rewrite.conditions.clone())
                .collect(),
            head: vec![Action::Union(Expr::Var(var), rewrite.rhs.clone())],
        }, globals)),
    )]
}

fn desugar_birewrite(ruleset: Symbol, rewrite: &Rewrite, globals: &HashSet<Symbol>) -> Vec<Command> {
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

// TODO use an egraph to perform the SSA translation without introducing
// so many fresh variables
fn expr_to_ssa(
    expr: &Expr,
    get_fresh: &mut Fresh,
    var_used: &mut HashSet<Symbol>,
    varJustUsed: &mut HashSet<Symbol>,
    res: &mut Vec<SSAFact>,
    constraints: &mut Vec<SSAFact>,
) -> Symbol {
    match expr {
        Expr::Lit(l) => {
            let fresh = get_fresh();
            res.push(SSAFact::AssignLit(fresh, l.clone()));
            let fresh2 = get_fresh();
            res.push(SSAFact::ConstrainEq(fresh2, fresh));
            fresh2
        }
        Expr::Var(v) => {
            if var_used.insert(*v) {
                varJustUsed.insert(*v);
                *v
            } else {
                let fresh = get_fresh();
                // logic to satisfy typechecker
                // if we used the variable in this recurrence, add the constraint afterwards
                if varJustUsed.contains(v) {
                    constraints.push(SSAFact::ConstrainEq(fresh, *v));
                // otherwise add the constrain immediately so we have the type
                } else {
                    res.push(SSAFact::ConstrainEq(fresh, *v));
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
                    varJustUsed,
                    res,
                    constraints,
                ));
            }
            let fresh = get_fresh();
            res.push(SSAFact::Assign(
                fresh,
                SSAExpr::Call(f.clone(), new_children),
            ));
            let fresh2 = get_fresh();
            res.push(SSAFact::ConstrainEq(fresh2, fresh));
            fresh2
        }
    }
}

fn ssa_valid_expr(expr: &SSAExpr, var_used: &mut HashSet<Symbol>) -> bool {
    match expr {
        SSAExpr::Call(_, children) => {
            for child in children {
                if !var_used.insert(*child) {
                    return false;
                }
            }
        }
    }
    true
}

// Given facts where variables are referenced multiple times,
// refactor it into SSA form again
// TODO all of the get_fresh functions should be unified
pub(crate) fn make_ssa_again(facts: Vec<SSAFact>) -> Vec<SSAFact> {
    let mut vars_used: HashSet<Symbol> = Default::default();
    let mut res = vec![];
    let mut var_counter = 0;
    let mut get_fresh = || {
        let fresh = format!("ssa{}__", var_counter);
        var_counter += 1;
        fresh.into()
    };
    for fact in facts {
        match fact {
            SSAFact::Assign(v, expr) => {
                if !vars_used.insert(v) {
                    panic!("invalid assignment to SSA variable: {:?}", v);
                }
                match expr {
                    SSAExpr::Call(f, children) => {
                        let mut new_children = vec![];
                        let mut constraints = vec![];
                        for child in children {
                            if vars_used.insert(child) {
                                new_children.push(child);
                            } else {
                                let fresh = get_fresh();
                                constraints.push(SSAFact::ConstrainEq(fresh, child));
                                new_children.push(fresh);
                            }
                        }
                        res.push(SSAFact::Assign(v, SSAExpr::Call(f, new_children)));
                        res.extend(constraints);
                    }
                }
            }
            SSAFact::ConstrainEq(v, v2) => {
                res.push(SSAFact::ConstrainEq(v, v2));
            }
            SSAFact::AssignLit(v, l) => {
                if vars_used.insert(v) {
                    res.push(SSAFact::AssignLit(v, l));
                } else {
                    let fresh = get_fresh();
                    res.push(SSAFact::AssignLit(fresh, l));
                    res.push(SSAFact::ConstrainEq(fresh, v));
                }
            }
        }
    }
    res
}

pub(crate) fn assert_ssa_valid(facts: &Vec<SSAFact>, actions: &Vec<SSAAction>) -> bool {
    //println!("assert_ssa_valid: {:?}", facts);
    let mut var_used: HashSet<Symbol> = Default::default();
    let mut var_used_constraints: HashSet<Symbol> = Default::default();
    for fact in facts {
        match fact {
            SSAFact::Assign(v, expr) => {
                if !var_used.insert(*v) {
                    panic!("invalid SSA variable: {:?}", v);
                }

                if !ssa_valid_expr(expr, &mut var_used) {
                    panic!("invalid SSA fact: {:?}", expr);
                }
            }
            SSAFact::ConstrainEq(v, v2) => {
                let b1 = var_used_constraints.insert(*v);
                let b2 = var_used_constraints.insert(*v2);
                // any constraints on variables are valid, but one needs to be defined
                if !var_used.contains(v) && !var_used.contains(v2) && b1 && b2 {
                    panic!("invalid SSA constraint: {:?} = {:?}", v, v2);
                }
            }
            SSAFact::AssignLit(v, _) => {
                if !var_used.insert(*v) {
                    panic!("invalid SSA variable: {:?}", v);
                }
            }
        }
    }

    var_used.extend(var_used_constraints);

    let mut fdefuse = |var, isdef| {
        if isdef {
        if !var_used.insert(var) {
            panic!("invalid SSA variable: {:?}", var);
        }
    } else if !var_used.contains(&var) {
            panic!("invalid SSA variable: {:?}", var);
        }
        var
    };
    for action in actions {
        action.map_def_use(&mut fdefuse);
    }

    true
}

fn flatten_equalities(equalities: Vec<(Symbol, Expr)>, get_fresh: &mut Fresh) -> Vec<SSAFact> {
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
        res.push(SSAFact::ConstrainEq(lhs, result));
    }
    
    res
}

fn flatten_facts(facts: &Vec<Fact>, get_fresh: &mut Fresh) -> Vec<SSAFact> {
    let mut equalities = vec![];
    for fact in facts {
        match fact {
            Fact::Eq(args) => {
                assert!(args.len() == 2);
                let lhs = &args[0];
                let rhs = &args[1];
                if let Expr::Var(v) = lhs {
                    equalities.push((v.clone(), rhs.clone()));
                } else if let Expr::Var(v) = rhs {
                    equalities.push((v.clone(), lhs.clone()));
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
    res: &mut Vec<SSAAction>,
) {
    match expr {
        Expr::Lit(l) => {
            res.push(SSAAction::LetLit(assign, l.clone()));
        }
        Expr::Var(v) => {
            res.push(SSAAction::LetVar(assign, v.clone()));
        }
        Expr::Call(f, children) => {
            let mut new_children = vec![];
            for child in children {
                let fresh = get_fresh();
                expr_to_flat_actions(fresh, child, get_fresh, res);
                new_children.push(fresh);
            }
            res.push(SSAAction::Let(
                assign,
                SSAExpr::Call(f.clone(), new_children),
            ));
        }
    }
}

fn flatten_actions(actions: &Vec<Action>, get_fresh: &mut Fresh) -> Vec<SSAAction> {
    let mut add_expr = |expr: Expr, res: &mut Vec<SSAAction>| {
        let fresh = get_fresh();
        expr_to_flat_actions(fresh, &expr, get_fresh, res);
        fresh
    };

    let mut res = vec![];

    for action in actions {
        match action {
            Action::Let(symbol, expr) => {
                let added = add_expr(expr.clone(), &mut res);
                res.push(SSAAction::LetVar(*symbol, added));
            }
            Action::Set(symbol, exprs, rhs) => {
                let set = SSAAction::Set(
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
                let del = SSAAction::Delete(
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
                let un = SSAAction::Union(
                    add_expr(lhs.clone(), &mut res),
                    add_expr(rhs.clone(), &mut res),
                );
                res.push(un);
            }
            Action::Panic(msg) => {
                res.push(SSAAction::Panic(msg.clone()));
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
    rule.map_exprs(&mut |e| match e {
        Expr::Var(v) if globals.contains(v) => Expr::Call(*v, vec![]),
        _ => e.clone(),
    })
}

fn flatten_rule(rule: Rule) -> FlatRule {
    let mut varcount = 0;
    let mut get_fresh = move || {
        varcount += 1;
        Symbol::from(format!("fvar{}__", varcount))
    };

    let res = FlatRule {
        head: flatten_actions(&rule.head, &mut get_fresh),
        body: flatten_facts(&rule.body, &mut get_fresh),
    };
    println!("Before rule: {}", rule);
    println!("Flat rule: {}", res);
    assert_ssa_valid(&res.body, &res.head);
    res
}

pub(crate) fn desugar_command(
    egraph: &EGraph,
    command: Command,
    globals: &mut HashSet<Symbol>,
) -> Result<Vec<Command>, Error> {
    Ok(match command {
        Command::Datatype { name, variants } => desugar_datatype(name, variants),
        Command::Rewrite(ruleset, rewrite) => desugar_rewrite(ruleset, &rewrite, &globals),
        Command::BiRewrite(ruleset, rewrite) => desugar_birewrite(ruleset, &rewrite, &globals),
        Command::Include(file) => {
            let s = std::fs::read_to_string(&file)
                .unwrap_or_else(|_| panic!("Failed to read file {file}"));
            egraph.parse_program(&s)?
        }
        Command::Rule(ruleset, rule) => vec![Command::FlatRule(
            ruleset,
            flatten_rule(parenthesize_globals(rule, globals)),
        )],
        _ => vec![command],
    })
}

// TODO desugar define to function tables (it requires type inference)
pub(crate) fn desugar_program(
    egraph: &EGraph,
    program: Vec<Command>,
) -> Result<Vec<Command>, Error> {
    let mut globals: HashSet<Symbol> = Default::default();
    let mut res = vec![];

    for command in program {
        let desugared = desugar_command(egraph, command, &mut globals)?;

        for newcommand in &desugared {
            match newcommand {
                Command::Define {
                    name,
                    expr: _,
                    cost: _,
                } => {
                    globals.insert(*name);
                }
                Command::Function(fdecl) => {
                    // add to globals if it has no arguments
                    if fdecl.schema.input.is_empty() {
                        globals.insert(fdecl.name);
                    }
                }
                _ => (),
            }
        }

        res.extend(desugared);
    }
    Ok(res)
}

pub fn to_rules(program: Vec<Command>) -> Vec<Command> {
    program
        .into_iter()
        .map(|command| match command {
            Command::FlatRule(ruleset, rule) => Command::Rule(ruleset, rule.to_rule()),
            _ => command,
        })
        .collect()
}
