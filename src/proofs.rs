use crate::*;

use symbolic_expressions::Sexp;

fn proof_header(egraph: &EGraph) -> Vec<Command> {
    let str = include_str!("proofheader.egg");
    egraph.parse_program(str).unwrap()
}

// make ast versions even for primitives
fn make_ast_version(_egraph: &EGraph, name: &Symbol) -> Symbol {
    /*if egraph.sorts.get(name).is_some() {
        name.clone()
    } else {*/
    Symbol::from(format!("Ast{}__", name))
}

fn make_rep_version(name: &Symbol) -> Symbol {
    Symbol::from(format!("{}Rep__", name))
}

fn make_ast_primitives(egraph: &EGraph) -> Vec<Command> {
    egraph.sorts.iter().map(|(name, _)| {
        Command::Function(
            FunctionDecl {
                name: make_ast_version(egraph, name),
                schema: Schema {
                    input: vec![*name],
                    output: "Ast__".into(),
                },
                merge: None,
                merge_action: vec![],
                default: None,
                cost: None,
            }
        )
    }).collect()
}

fn make_ast_func(egraph: &EGraph, fdecl: &FunctionDecl) -> FunctionDecl {
    FunctionDecl {
        name: make_ast_version(egraph, &fdecl.name),
        schema: Schema {
            input: fdecl
                .schema
                .input
                .iter()
                .map(|sort| "Ast__".into())
                .collect(),
            output: "Ast__".into(),
        },
        merge: None,
        merge_action: vec![],
        default: None,
        cost: None,
    }
}

fn merge_action(egraph: &EGraph, fdecl: &FunctionDecl) -> Vec<Action> {
    let child1 = |i| Symbol::from(format!("c1_{}__", i));
    let child2 = |i| Symbol::from(format!("c2_{}__", i));

    let mut congr_prf = Sexp::String("Null__".to_string());
    for i in 0..fdecl.schema.input.len() {
        let current = fdecl.schema.input.len() - i - 1;
        congr_prf = Sexp::List(vec![
            Sexp::String("Cons__".to_string()),
            Sexp::List(vec![
                Sexp::String("DemandEq__".to_string()),
                Sexp::String(child1(current).to_string()),
                Sexp::String(child2(current).to_string()),
            ]),
            congr_prf,
        ]);
    }

    vec![
        "(let t1 (TrmOf__ old))".to_string(),
        "(let t2 (TrmOf__ new))".to_string(),
        "(let p1 (PrfOf__ old))".to_string(),
    ]
    .into_iter()
    .chain(
        fdecl
            .schema
            .input
            .iter()
            .enumerate()
            .flat_map(|(i, _sort)| {
                vec![
                    format!("(let {} (GetChild__ t1 {}))", child1(i), i),
                    format!("(let {} (GetChild__ t2 {}))", child2(i), i),
                ]
            }),
    )
    .chain(vec![
        format!("(let congr_prf__ (Congruence__ p1 {}))", congr_prf),
        "(set (EqGraph__ t1 t2) congr_prf__)".to_string(),
        "(set (EqGraph__ t2 t1) (Flip__ congr_prf__))".to_string(),
    ])
    .map(|s| egraph.action_parser.parse(&s).unwrap())
    .collect()
}

fn instrument_rule(_egraph: &EGraph, rule: &FlatRule) -> FlatRule {
    let mut varcount = 0;
    let mut get_fresh = move || {
        varcount += 1;
        Symbol::from(format!("pvar{}__", varcount))
    };

    let mut facts = rule.body.clone();
    // TODO make flat expessions only be calls
    // contrain two variables to be equal
    let mut var_eq_constraints = vec![];
    // contrain a variable to be equal to a literal
    let mut lit_constraints = vec![];
    // all of the variables for the proofs of all of the representatives
    let mut term_proofs = vec![];
    // maps each variable to the variable representing a term
    // these terms need to be proven to be equal for the proof to go through
    let mut var_terms: HashMap<Symbol, Vec<Symbol>> = Default::default();

    for fact in &rule.body {
        match &fact.expr {
            FlatExpr::Var(v) => var_eq_constraints.push((fact.symbol, v)),
            FlatExpr::Lit(l) => lit_constraints.push((fact.symbol, l)),
            FlatExpr::Call(head, body) => {
                let rep = get_fresh();
                let rep_trm = get_fresh();
                let rep_prf = get_fresh();
                facts.push(FlatFact::new(
                    rep,
                    FlatExpr::Call(make_rep_version(head), body.clone()),
                ));
                facts.push(FlatFact::new(
                    rep_trm,
                    FlatExpr::Call("TrmOf__".into(), vec![rep]),
                ));
                facts.push(FlatFact::new(
                    rep_prf,
                    FlatExpr::Call("PrfOf__".into(), vec![rep]),
                ));
                term_proofs.push(rep_prf);

                for (i, child) in body.iter().enumerate() {
                    let child_trm = get_fresh();
                    let const_var = get_fresh();
                    facts.push(FlatFact::new(
                        const_var,
                        FlatExpr::Lit(Literal::Int(i as i64)),
                    ));
                    facts.push(FlatFact::new(
                        child_trm,
                        FlatExpr::Call("GetChild__".into(), vec![rep_trm, const_var]),
                    ));
                    if let Some(vars) = var_terms.get_mut(child) {
                        vars.push(child_trm);
                    } else {
                        var_terms.insert(child.clone(), vec![child_trm]);
                    }
                }
            }
        }
    }

    // res.head.extend();
    FlatRule {
        head: rule.head.clone(),
        body: facts,
    }
}

fn make_rep_func(egraph: &EGraph, fdecl: &FunctionDecl) -> FunctionDecl {
    FunctionDecl {
        name: make_rep_version(&fdecl.name),
        schema: Schema {
            input: fdecl.schema.input.clone(),
            output: "TrmPrf__".into(),
        },
        merge: Some(Expr::Var("old".into())),
        merge_action: merge_action(egraph, fdecl),
        default: None,
        cost: None,
    }
}

fn make_proof_rule(egraph: &EGraph, fdecl: &FunctionDecl) -> Command {
    let getchild = |i| Symbol::from(format!("c{}__", i));
    Command::Rule(
        "proofrules__".into(),
        Rule {
            body: vec![Fact::Eq(vec![
                Expr::Var("ast__".into()),
                Expr::Call(
                    make_ast_version(egraph, &fdecl.name),
                    fdecl.schema.input.iter().enumerate().map(|(i, _)| {
                        Expr::Var(getchild(i))
                    }).collect()
                ),
            ])],
            head: fdecl
                .schema
                .input
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    Action::Set(
                        "GetChild__".into(),
                        vec![Expr::Var("ast__".into()), Expr::Lit(Literal::Int(i as i64))],
                        Expr::Var(getchild(i)),
                    )
                })
                .collect(),
        },
    )
}

fn make_runner(config: &RunConfig) -> Vec<Command> {
    let mut res = vec![];
    let run_proof_rules = Command::Run(RunConfig {
        ruleset: "proofrules__".into(),
        limit: 100,
        until: None
    });
    for i in 0..config.limit {
        res.push(run_proof_rules.clone());
        res.push(Command::Run(RunConfig {
            ruleset: config.ruleset,
            limit: 1,
            until: config.until.clone(),
        }));
    }
    res.push(run_proof_rules);
    res
}

// the egraph is the initial egraph with only default sorts
pub(crate) fn add_proofs(egraph: &EGraph, program: Vec<Command>) -> Vec<Command> {
    let mut res = proof_header(egraph);
    res.extend(make_ast_primitives(egraph));

    for command in program {
        match &command {
            Command::Datatype {
                name: _,
                variants: _,
            } => {
                panic!("Datatype should have been desugared");
            }
            Command::Sort(name, presort_and_args) => {
                res.push(command.clone());
                res.push(Command::Sort(
                    make_ast_version(egraph, name),
                    presort_and_args.clone(),
                ));
            }
            Command::Function(fdecl) => {
                res.push(command.clone());
                res.push(Command::Function(make_ast_func(egraph, fdecl)));
                res.push(Command::Function(make_rep_func(egraph, fdecl)));
                res.push(make_proof_rule(egraph, fdecl));
            }
            Command::Rule(_ruleset, _rule) => {
                panic!("Rule should have been desugared");
            }
            Command::FlatRule(ruleset, rule) => {
                //res.push(Command::Rule(*ruleset, rule.to_rule()));
                res.push(Command::Rule(
                    *ruleset,
                    instrument_rule(egraph, &rule).to_rule(),
                ));
            }
            Command::Run(config) => {
                res.extend(make_runner(config));
            }
            _ => res.push(command),
        }
    }

    res
}
