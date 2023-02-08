use std::cmp::max;

use crate::*;

pub(crate) type Fresh = dyn FnMut() -> Symbol;

fn make_get_fresh(program: &Vec<Command>) -> impl FnMut() -> Symbol
{
    let mut max_underscores: usize = 0;
    let program_str = ListDisplay(program, "\n").to_string();
    let mut counter: i64 = -1;
    for char in program_str.chars() {
        if char == '_' {
            counter = max(counter, 0);
            counter += 1;
            max_underscores = max(max_underscores,  counter as usize);
        } else {
            counter = -1;
        }
    }

    let underscores = "_".repeat(max_underscores+1);
    let mut fcounter = 0;
    move || {
        fcounter += 1;
        format!("v{}{}", fcounter, underscores).into()
    }
}

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
    desugar: &mut Desugar,
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
            desugar
        ),
    )]
}

fn desugar_birewrite(
    ruleset: Symbol,
    rewrite: &Rewrite,
    desugar: &mut Desugar,
) -> Vec<NormCommand> {
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

// TODO use an egraph to perform the Norm translation without introducing
// so many fresh variables
fn expr_to_ssa(
    expr: &Expr,
    desugar: &mut Desugar,
    var_used: &mut HashSet<Symbol>,
    var_just_used: &mut HashSet<Symbol>,
    res: &mut Vec<NormFact>,
    constraints: &mut Vec<NormFact>,
) -> Symbol {
    match expr {
        Expr::Lit(l) => {
            let fresh = (desugar.get_fresh)();
            res.push(NormFact::AssignLit(fresh, l.clone()));
            let fresh2 = (desugar.get_fresh)();
            res.push(NormFact::ConstrainEq(fresh2, fresh));
            fresh2
        }
        Expr::Var(v) => {
            if var_used.insert(*v) {
                var_just_used.insert(*v);
                *v
            } else {
                let fresh = (desugar.get_fresh)();
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
                    desugar,
                    var_used,
                    var_just_used,
                    res,
                    constraints,
                ));
            }
            let fresh = (desugar.get_fresh)();
            res.push(NormFact::Assign(fresh, NormExpr::Call(*f, new_children)));
            let fresh2 = (desugar.get_fresh)();
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

fn flatten_equalities(equalities: Vec<(Symbol, Expr)>, desugar: &mut Desugar) -> Vec<NormFact> {
    let mut res = vec![];

    let mut var_used = Default::default();
    for (lhs, rhs) in equalities {
        let mut constraints = vec![];
        let result = expr_to_ssa(
            &rhs,
            desugar,
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
    assign: Symbol,
    expr: &Expr,
    desugar: &mut Desugar,
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
                let fresh = (desugar.get_fresh)();
                expr_to_flat_actions(fresh, child, desugar, res);
                new_children.push(fresh);
            }
            res.push(NormAction::Let(assign, NormExpr::Call(*f, new_children)));
        }
    }
}

fn flatten_actions(actions: &Vec<Action>, desugar: &mut Desugar) -> Vec<NormAction> {
    let mut add_expr = |expr: Expr, res: &mut Vec<NormAction>| {
        let fresh = (desugar.get_fresh)();
        expr_to_flat_actions(fresh, &expr, desugar, res);
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

fn flatten_rule(rule_in: Rule, desugar: &mut Desugar) -> NormRule {
    let rule = parenthesize_globals(rule_in, &desugar.globals);

    let res = NormRule {
        head: flatten_actions(&rule.head, desugar),
        body: flatten_facts(&rule.body, desugar),
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
        Command::Rewrite(ruleset, rewrite) => desugar_rewrite(ruleset, &rewrite, desugar),
        Command::BiRewrite(ruleset, rewrite) => {
            desugar_birewrite(ruleset, &rewrite, desugar)
        }
        Command::Include(file) => {
            let s = std::fs::read_to_string(&file)
                .unwrap_or_else(|_| panic!("Failed to read file {file}"));
            desugar_commands(egraph, egraph.parse_program(&s)?, desugar)?
        }
        Command::Rule(ruleset, rule) => vec![NormCommand::NormRule(
            ruleset,
            flatten_rule(rule, desugar),
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
            expr_to_flat_actions(name, &expr, desugar, &mut actions);
            for action in actions {
                commands.push(NormCommand::NormAction(action));
            }
            commands
        }
        Command::AddRuleset(name) => vec![NormCommand::AddRuleset(name)],
        Command::Action(action) => flatten_actions(&vec![action], desugar)
            .into_iter()
            .map(NormCommand::NormAction)
            .collect(),
        Command::Run(run) => vec![NormCommand::Run(run)],
        Command::Simplify { expr, config } => vec![NormCommand::Simplify { expr, config }],
        Command::Calc(idents, exprs) => vec![NormCommand::Calc(idents, exprs)],
        Command::Extract { variants, e } => {
            let fresh = (desugar.get_fresh)();
            flatten_actions(&vec![Action::Let(fresh, e)], desugar)
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
    let get_fresh = Box::new(make_get_fresh(&program));
    desugar_commands(
        egraph,
        program,
        &mut Desugar {
            globals: Default::default(),
            get_fresh,
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
