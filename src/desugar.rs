use crate::*;

type Fresh = dyn FnMut() -> Symbol;

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

fn desugar_rewrite(rewrite: &Rewrite) -> Vec<Command> {
    let var = Symbol::from("rewrite_var__");
    vec![Command::FlatRule(flatten_rule(Rule {
        body: [Fact::Eq(vec![Expr::Var(var), rewrite.lhs.clone()])]
            .into_iter()
            .chain(rewrite.conditions.clone())
            .collect(),
        head: vec![Action::Union(Expr::Var(var), rewrite.rhs.clone())],
    }))]
}

fn desugar_birewrite(rewrite: &Rewrite) -> Vec<Command> {
    let rw2 = Rewrite {
        lhs: rewrite.rhs.clone(),
        rhs: rewrite.lhs.clone(),
        conditions: rewrite.conditions.clone(),
    };
    desugar_rewrite(rewrite)
        .into_iter()
        .chain(desugar_rewrite(&rw2))
        .collect()
}

// TODO make new flat expr type like AtomTerm
fn flatten_expr(expr: &Expr, res: &mut Vec<(Symbol, FlatExpr)>, get_fresh: &mut Fresh) -> Symbol {
    match expr {
        Expr::Lit(l) => {
            let rvar = get_fresh();
            res.push((rvar, FlatExpr::Lit(l.clone())));
            rvar
        }
        Expr::Var(v) => *v,
        Expr::Call(f, children) => {
            let mut new_children = vec![];
            for child in children {
                new_children.push(flatten_expr(child, res, get_fresh));
            }
            let rvar = get_fresh();
            res.push((rvar, FlatExpr::Call(f.clone(), new_children)));
            rvar
        }
    }
}


// Makes sure the expression does not have nested calls
fn flatten_equality(equality: (Symbol, Expr), get_fresh: &mut Fresh) -> Vec<FlatFact> {
    let mut flattened = vec![];
    let fvar = flatten_expr(&equality.1, &mut flattened, get_fresh);
    let mut res = vec![];
    for (var, expr) in flattened {
        res.push(FlatFact::new(var, expr));
    }
    res.push(FlatFact::new(equality.0, FlatExpr::Var(fvar)));
    res
}

fn flatten_equalities(equalities: Vec<(Symbol, Expr)>, get_fresh: &mut Fresh) -> Vec<FlatFact> {
    equalities.into_iter().flat_map(|e| flatten_equality(e, get_fresh)).collect()
}

fn flatten_facts(facts: &Vec<Fact>, get_fresh: &mut Fresh) -> Vec<FlatFact> {
    let mut equalities = vec![];
    for fact in facts {
        match fact {
            Fact::Eq(args) => {
                assert!(args.len() == 2);
                let lhs = &args[0];
                let rhs = &args[1];
                if let Expr::Var(v) = lhs {
                    equalities.push((v.clone(), rhs.clone()));
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

fn flatten_actions(actions: &Vec<Action>, get_fresh: &mut Fresh) -> Vec<FlatAction> {
    let mut setup = vec![];
    let mut add_expr = |expr: Expr| {
        let mut flattened = vec![];
        let fvar = flatten_expr(&expr, &mut flattened, get_fresh);
        for (var, expr) in flattened {
            setup.push(FlatAction::Let(var, expr));
        }
        FlatExpr::Var(fvar)
    };

    let mut res = vec![];

    for action in actions {
        res.push(match action {
            Action::Let(symbol, expr) => {
                FlatAction::Let(*symbol, add_expr(expr.clone()))
            }
            Action::Set(symbol, exprs, rhs) => {
                FlatAction::Set(*symbol, exprs.clone().into_iter().map(&mut add_expr).collect(), add_expr(rhs.clone()))
            }
            Action::Delete(symbol, exprs) => {
                FlatAction::Delete(*symbol, exprs.clone().into_iter().map(&mut add_expr).collect())
            }
            Action::Union(lhs, rhs) => {
                FlatAction::Union(add_expr(lhs.clone()), add_expr(rhs.clone()))
            }
            Action::Panic(msg) => FlatAction::Panic(msg.clone()),
            Action::Expr(expr) => FlatAction::Expr(add_expr(expr.clone())),
        });
    }

    setup.into_iter().chain(res.into_iter()).collect()
}

fn flatten_rule(rule: Rule) -> FlatRule {
    let mut varcount = 0;
    let mut get_fresh = move || {
        varcount += 1;
        Symbol::from(format!("fvar{}__", varcount))
    };

    FlatRule {
        head: flatten_actions(&rule.head, &mut get_fresh),
        body: flatten_facts(&rule.body, &mut get_fresh),
    }
}

pub(crate) fn desugar_command(egraph: &EGraph, command: Command) -> Result<Vec<Command>, Error> {
    Ok(match command {
        Command::Datatype { name, variants } => desugar_datatype(name, variants),
        Command::Rewrite(rewrite) => desugar_rewrite(&rewrite),
        Command::BiRewrite(rewrite) => desugar_birewrite(&rewrite),
        Command::Include(file) => {
            let s = std::fs::read_to_string(&file)
                .unwrap_or_else(|_| panic!("Failed to read file {file}"));
            egraph.parse_program(&s)?
        }
        Command::Rule(rule) => vec![Command::FlatRule(flatten_rule(rule))],
        _ => vec![command],
    })
}

pub(crate) fn desugar_program(
    egraph: &EGraph,
    program: Vec<Command>,
) -> Result<Vec<Command>, Error> {
    let intermediate: Result<Vec<Vec<Command>>, Error> = program
        .into_iter()
        .map(|command| desugar_command(egraph, command))
        .collect();
    intermediate.map(|v| v.into_iter().flatten().collect())
}
