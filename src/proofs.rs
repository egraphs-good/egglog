use crate::*;

use crate::desugar::Fresh;
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
    egraph
        .sorts
        .iter()
        .map(|(name, _)| {
            Command::Function(FunctionDecl {
                name: make_ast_version(egraph, name),
                schema: Schema {
                    input: vec![*name],
                    output: "Ast__".into(),
                },
                merge: None,
                merge_action: vec![],
                default: None,
                cost: None,
            })
        })
        .collect()
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

#[derive(Default, Clone)]
struct ProofInfo {
    // TODO make flat expessions only be calls
    // contrain two variables to be equal
    pub var_eq_constraints: Vec<(Symbol, Symbol)>,
    // contrain a variable to be equal to a literal
    pub lit_constraints: Vec<(Symbol, Literal)>,
    // proofs for each variable
    pub term_proofs: HashMap<Symbol, Symbol>,
    // maps each variable to variables each representing a term
    // these terms need to be proven to be equal for the proof to go through
    pub var_terms: HashMap<Symbol, Vec<Symbol>>,
}

fn instrument_facts(body: &Vec<SSAFact>, get_fresh: &mut Fresh) -> (ProofInfo, Vec<SSAFact>) {
    let mut info: ProofInfo = Default::default();
    let mut facts = body.clone();

    /*for fact in body {
        match &fact.expr {
            SSAExpr::Var(v) => info.var_eq_constraints.push((fact.symbol, *v)),
            SSAExpr::Lit(l) => info.lit_constraints.push((fact.symbol, l.clone())),
            SSAExpr::Call(head, body) => {
                let rep = get_fresh();
                let rep_trm = get_fresh();
                let rep_prf = get_fresh();
                facts.push(SSAFact::new(
                    rep,
                    SSAExpr::Call(make_rep_version(head), body.clone()),
                ));
                facts.push(SSAFact::new(
                    rep_trm,
                    SSAExpr::Call("TrmOf__".into(), vec![rep]),
                ));
                facts.push(SSAFact::new(
                    rep_prf,
                    SSAExpr::Call("PrfOf__".into(), vec![rep]),
                ));
                info.term_proofs.insert(rep, rep_prf);

                for (i, child) in body.iter().enumerate() {
                    let child_trm = get_fresh();
                    let const_var = get_fresh();
                    facts.push(SSAFact::new(
                        const_var,
                        SSAExpr::Lit(Literal::Int(i as i64)),
                    ));
                    facts.push(SSAFact::new(
                        child_trm,
                        SSAExpr::Call("GetChild__".into(), vec![rep_trm, const_var]),
                    ));
                    if let Some(vars) = info.var_terms.get_mut(child) {
                        vars.push(child_trm);
                    } else {
                        info.var_terms.insert(child.clone(), vec![child_trm]);
                    }
                }
            }
        }
    }*/

    (info, facts)
}

// Adds the proof for an expr
fn add_expr_proof(info: &ProofInfo, expr: &Expr, res: &mut Vec<SSAAction>) {
    match expr {
        Expr::Var(v) => (),
        Expr::Lit(l) => (),
        Expr::Call(head, body) => {
            todo!()
        }
    }
}

fn add_rule_proof(info: &ProofInfo, res: &mut Vec<SSAAction>) {}

fn instrument_rule(_egraph: &EGraph, rule: &FlatRule) -> FlatRule {
    let mut varcount = 0;
    let mut get_fresh = move || {
        varcount += 1;
        Symbol::from(format!("pvar{}__", varcount))
    };

    let (mut info, facts) = instrument_facts(&rule.body, &mut get_fresh);

    let mut actions = vec![];
    for action in &rule.head {
        actions.push(action.clone());
    }

    // res.head.extend();
    FlatRule {
        head: actions,
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

fn make_getchild_rule(egraph: &EGraph, fdecl: &FunctionDecl) -> Command {
    let getchild = |i| Symbol::from(format!("c{}__", i));
    Command::Rule(
        "proofrules__".into(),
        Rule {
            body: vec![Fact::Eq(vec![
                Expr::Var("ast__".into()),
                Expr::Call(
                    make_ast_version(egraph, &fdecl.name),
                    fdecl
                        .schema
                        .input
                        .iter()
                        .enumerate()
                        .map(|(i, _)| Expr::Var(getchild(i)))
                        .collect(),
                ),
            ])],
            head: fdecl
                .schema
                .input
                .iter()
                .enumerate()
                .map(|(i, _s)| {
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
        until: None,
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
    let mut res = vec![];

    for command in proof_header(egraph) {
        match command {
            Command::FlatRule(ruleset, rule) => {
                res.push(Command::Rule(ruleset, rule.to_rule()));
            }
            _ => {
                res.push(command);
            }
        }
    }

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
                res.push(make_getchild_rule(egraph, fdecl));
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
