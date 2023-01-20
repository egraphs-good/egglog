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
    vec![Command::Rule(flatten_rule(Rule {
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

// Makes sure the expression does not have nested calls
fn flatten_equality(equality: (Symbol, Expr), get_fresh: &mut Fresh) -> Vec<Fact> {
    match &equality.1 {
        Expr::Lit(_l) => vec![Fact::Eq(vec![Expr::Var(equality.0), equality.1])],
        Expr::Var(_v) => vec![Fact::Eq(vec![Expr::Var(equality.0), equality.1])],
        Expr::Call(f, children) => {
            let mut res = vec![];
            let mut new_children = vec![];
            for child in children {
                if let Expr::Call(_f, _c) = child {
                    let fresh = get_fresh();
                    res.extend(flatten_equality((fresh.clone(), child.clone()), get_fresh));
                    new_children.push(Expr::Var(fresh));
                } else {
                    new_children.push(child.clone());
                }
            }
            res.push(Fact::Eq(vec![Expr::Var(equality.0), Expr::Call(f.clone(), new_children)]));
            res
        }
    }
}

fn flatten_equalities(equalities: Vec<(Symbol, Expr)>, get_fresh: &mut Fresh) -> Vec<Fact> {
    equalities.into_iter().flat_map(|e| flatten_equality(e, get_fresh)).collect()
}

fn flatten_facts(facts: &Vec<Fact>, get_fresh: &mut Fresh) -> Vec<Fact> {
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

fn flatten_rule(rule: Rule) -> Rule {
    let mut varcount = 0;
    let mut get_fresh = move || {
        varcount += 1;
        Symbol::from(format!("fvar{}__", varcount))
    };

    
    Rule {
        head: rule.head,
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
        Command::Rule(rule) => vec![Command::Rule(flatten_rule(rule))],
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
