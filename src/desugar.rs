use std::cmp::max;

use crate::*;

pub(crate) type Fresh = dyn FnMut() -> Symbol;
pub(crate) type NewId = dyn FnMut() -> CommandId;

pub(crate) fn literal_name(egraph: &EGraph, literal: &Literal) -> Symbol {
    egraph.type_info.infer_literal(literal).name()
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

fn desugar_birewrite(
    ruleset: Symbol,
    rewrite: &Rewrite,
    desugar: &mut Desugar,
) -> Vec<NCommand> {
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
            // fresh variable for call
            let fresh = (desugar.get_fresh)();
            if desugar.egraph.type_info.primitives.contains_key(f) {
                res.push(NormFact::Compute(fresh, NormExpr::Call(*f, new_children)));
            } else {
                res.push(NormFact::Assign(fresh, NormExpr::Call(*f, new_children)));
            }

            // fresh variable for any use
            let fresh2 = (desugar.get_fresh)();
            res.push(NormFact::ConstrainEq(fresh2, fresh));
            fresh2
        }
    }
}

fn ssa_valid_expr(expr: &NormExpr, var_used: &mut HashSet<Symbol>, desugar: &Desugar) -> bool {
    match expr {
        NormExpr::Call(_, children) => {
            for child in children {
                if !desugar.let_types.contains_key(child) && !var_used.insert(*child) {
                    return false;
                }
            }
        }
    }
    true
}

pub(crate) fn assert_ssa_valid(
    facts: &Vec<NormFact>,
    actions: &Vec<NormAction>,
    desugar: &Desugar,
) -> bool {
    let mut var_used: HashSet<Symbol> = Default::default();
    let mut var_used_constraints: HashSet<Symbol> = Default::default();
    for fact in facts {
        match fact {
            NormFact::Assign(v, expr) | NormFact::Compute(v, expr) => {
                if desugar.let_types.contains_key(v) {
                    panic!("invalid Norm variable: {:?}", v);
                }
                if !var_used.insert(*v) {
                    panic!("invalid Norm variable: {:?}", v);
                }

                if !ssa_valid_expr(expr, &mut var_used, desugar) {
                    panic!("invalid Norm fact: {:?}", expr);
                }
            }
            NormFact::ConstrainEq(v, v2) => {
                let b1 = var_used_constraints.insert(*v);
                let b2 = var_used_constraints.insert(*v2);
                // any constraints on variables are valid, but one needs to be defined
                if !desugar.let_types.contains_key(v)
                    && !desugar.let_types.contains_key(v2)
                    && !var_used.contains(v)
                    && !var_used.contains(v2)
                    && b1
                    && b2
                {
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
            if desugar.let_types.contains_key(&var) {
                panic!("invalid Norm variable: {:?}", var);
            }
            if !var_used.insert(var) {
                panic!("invalid Norm variable: {:?}", var);
            }
        } else if !var_used.contains(&var) && !desugar.let_types.contains_key(&var) {
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
                match child {
                    Expr::Var(v) => {
                        new_children.push(*v);
                    }
                    _ => {
                        let fresh = (desugar.get_fresh)();
                        expr_to_flat_actions(fresh, child, desugar, res);
                        new_children.push(fresh);
                    }
                }
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

fn flatten_rule(rule: Rule, desugar: &mut Desugar) -> NormRule {
    let res = NormRule {
        head: flatten_actions(&rule.head, desugar),
        body: flatten_facts(&rule.body, desugar),
    };
    // TODO re-add with type info
    //assert_ssa_valid(&res.body, &res.head, desugar);
    res
}

pub struct Desugar<'a> {
    pub func_types: HashMap<Symbol, Schema>,
    pub let_types: HashMap<Symbol, Schema>,
    pub get_fresh: Box<Fresh>,
    pub get_new_id: Box<NewId>,
    pub egraph: &'a EGraph,
}

impl<'a> Desugar<'a> {
    pub fn get_type(&self, symbol: Symbol) -> Option<&Schema> {
        self.func_types
            .get(&symbol)
            .or_else(|| self.let_types.get(&symbol))
    }
}

pub(crate) fn desugar_command(
    command: Command,
    desugar: &mut Desugar,
) -> Result<Vec<NormCommand>, Error> {
    let res = match command.clone() {
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
            return desugar_commands(desugar.egraph.parse_program(&s)?, desugar)
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
            expr_to_flat_actions(name, &expr, desugar, &mut actions);
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
                command: NCommand::Fail(Box::new(last.command))
        });
            return Ok(desugared)
        }
        Command::Input { .. } => {
            todo!("desugar input");
        }
    };

    Ok(res.into_iter().map(|c| {
        NormCommand {
            metadata: Metadata { id: (desugar.get_new_id)() },
            command: c,
        }
    }).collect())
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
    egraph: &EGraph,
    program: Vec<Command>,
) -> Result<(Vec<NormCommand>, Desugar), Error> {
    let get_fresh = Box::new(make_get_fresh(&program));
    let mut desugar = Desugar {
        func_types: Default::default(),
        let_types: Default::default(),
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
