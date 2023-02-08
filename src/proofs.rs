use crate::*;

use crate::desugar::{Fresh};
use symbolic_expressions::Sexp;

fn proof_header(egraph: &EGraph) -> Vec<Command> {
    let str = include_str!("proofheader.egg");
    egraph.parse_program(str).unwrap()
}

// make ast versions even for primitives
fn make_ast_version(_egraph: &EGraph, name: &Symbol) -> Symbol {
    Symbol::from(format!("Ast{}__", name))
}

fn make_rep_version(name: &Symbol) -> Symbol {
    Symbol::from(format!("{}Rep__", name))
}

fn literal_name(egraph: &EGraph, literal: &Literal) -> Symbol {
    egraph.infer_literal(literal).name()
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
                .map(|_sort| "Ast__".into())
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

#[derive(Default, Clone, Debug)]
struct ProofInfo {
    // proof for each variable bound in an assignment (lhs or rhs)
    pub var_term: HashMap<Symbol, Symbol>,
    // proofs for each variable
    pub var_proof: HashMap<Symbol, Symbol>,
}

// This function makes use of the property that the body is Norm
// variables appear at most once (including the rhs of assignments)
// besides when they appear in constraints
fn instrument_facts(
    egraph: &EGraph,
    body: &Vec<NormFact>,
    get_fresh: &mut Fresh,
) -> (ProofInfo, Vec<Fact>) {
    let mut info: ProofInfo = Default::default();
    let mut facts: Vec<Fact> = body.iter().map(|f| f.to_fact()).collect();

    for fact in body {
        match fact {
            NormFact::AssignLit(lhs, rhs) => {
                let literal_name = literal_name(&egraph, rhs);
                let rep = get_fresh();
                let rep_trm = get_fresh();
                let rep_prf = get_fresh();
                facts.push(Fact::Eq(
                    vec![Expr::Var(rep),
                    Expr::Call(make_rep_version(&literal_name), vec![Expr::Var(*lhs)])]
                ));
                facts.push(Fact::Eq(
                    vec![Expr::Var(rep_trm),
                    Expr::Call("TrmOf__".into(), vec![Expr::Var(rep)])]
                ));
                facts.push(Fact::Eq(vec![
                    Expr::Var(rep_prf),
                    Expr::Call("PrfOf__".into(), vec![Expr::Var(rep)])]
                ));

                assert!(info.var_term.insert(*lhs, rep_trm).is_none());
                assert!(info.var_proof.insert(*lhs, rep_prf).is_none());
            }
            NormFact::Assign(lhs, rhs) => {
                let head = match rhs {
                    NormExpr::Call(head, _) => head,
                };
                let body = match rhs {
                    NormExpr::Call(_, body) => body,
                };
                let rep = get_fresh();
                let rep_trm = get_fresh();
                let rep_prf = get_fresh();
                facts.push(Fact::Eq(vec![
                    Expr::Var(rep),
                    Expr::Call(make_rep_version(head), body.iter().map(|x| Expr::Var(*x)).collect())]
                ));
                facts.push(Fact::Eq(vec![
                    Expr::Var(rep_trm),
                    Expr::Call("TrmOf__".into(), vec![Expr::Var(rep)])]
                ));
                facts.push(Fact::Eq(vec![
                    Expr::Var(rep_prf),
                    Expr::Call("PrfOf__".into(), vec![Expr::Var(rep)])]
                ));

                assert!(info.var_term.insert(*lhs, rep_trm).is_none());
                assert!(info.var_proof.insert(*lhs, rep_prf).is_none());

                for (i, child) in body.iter().enumerate() {
                    //println!("child: {:?}", child);
                    let child_trm = get_fresh();
                    let const_var = get_fresh();
                    facts.push(Fact::Eq(vec![Expr::Var(const_var),
                    Expr::Lit(Literal::Int(i as i64))]));

                    facts.push(Fact::Eq(vec![
                        Expr::Var(child_trm),
                        Expr::Call("GetChild__".into(), vec![Expr::Var(rep_trm), Expr::Var(const_var)])]
                    ));
                    assert!(info.var_term.insert(*child, child_trm).is_none());
                }
            }
            NormFact::ConstrainEq(lhs, rhs) => (),
        }
    }

    // now fill in representitive terms for any aliases
    for fact in body {
        if let NormFact::ConstrainEq(lhs, rhs) = fact {
            if let Some(rep_term) = info.var_term.get(lhs) {
                if info.var_term.get(rhs).is_none() {
                    info.var_term.insert(*rhs, *rep_term);
                }
            } else if let Some(rep_term) = info.var_term.get(rhs) {
                if info.var_term.get(lhs).is_none() {
                    info.var_term.insert(*lhs, *rep_term);
                }
            } else {
                panic!(
                    "Contraint without representative term for at least one side {} = {}",
                    lhs, rhs
                );
            }
        }
    }

    (info, facts)
}

fn add_action_proof(
    rule_proof: Symbol,
    info: &mut ProofInfo,
    action: &NormAction,
    res: &mut Vec<NormAction>,
    get_fresh: &mut Fresh,
    egraph: &EGraph,
) {
    match action {
        NormAction::LetVar(var1, var2) => {
            info.var_term
                .insert(*var1, *info.var_term.get(var2).unwrap());
        }
        NormAction::Delete(..) | NormAction::Panic(..) => (),
        NormAction::Union(var1, var2) => {
            res.push(NormAction::Set(
                "EqGraph__".into(),
                vec![
                    *info.var_term.get(var1).unwrap(),
                    *info.var_term.get(var2).unwrap(),
                ],
                rule_proof,
            ));
            res.push(NormAction::Set(
                "EqGraph__".into(),
                vec![
                    *info.var_term.get(var2).unwrap(),
                    *info.var_term.get(var1).unwrap(),
                ],
                rule_proof,
            ));
        }
        NormAction::Set(head, children, rhs) => {
            // add to the equality graph when we set things equal to each other
            let newterm = get_fresh();
            res.push(NormAction::Let(
                newterm,
                NormExpr::Call(
                    make_ast_version(egraph, head),
                    children
                        .iter()
                        .map(|v| *info.var_term.get(v).unwrap())
                        .collect(),
                ),
            ));
            res.push(NormAction::Set(
                "EqGraph__".into(),
                vec![newterm, *info.var_term.get(rhs).unwrap()],
                rule_proof,
            ));
            res.push(NormAction::Set(
                "EqGraph__".into(),
                vec![*info.var_term.get(rhs).unwrap(), newterm],
                rule_proof,
            ));
        }
        NormAction::Let(lhs, NormExpr::Call(rhsname, rhsvars)) => {
            let newterm = get_fresh();
            // make the term for this variable
            res.push(NormAction::Let(
                newterm,
                NormExpr::Call(
                    make_ast_version(egraph, rhsname),
                    rhsvars
                        .iter()
                        .map(|v| *info.var_term.get(v).unwrap())
                        .collect(),
                ),
            ));
            info.var_term.insert(*lhs, newterm);

            let ruletrm = get_fresh();
            res.push(NormAction::Let(
                ruletrm,
                NormExpr::Call("RuleTerm__".into(), vec![rule_proof, newterm]),
            ));

            let trmprf = get_fresh();
            res.push(NormAction::Let(
                trmprf,
                NormExpr::Call("MakeTrmPrf__".into(), vec![newterm, ruletrm]),
            ));

            res.push(NormAction::Set(
                make_rep_version(rhsname),
                rhsvars.clone(),
                trmprf,
            ));
        }
        // very similar to let case
        NormAction::LetLit(lhs, lit) => {
            let newterm = get_fresh();
            // make the term for this variable
            res.push(NormAction::Let(
                newterm,
                NormExpr::Call(
                    make_ast_version(egraph, &literal_name(egraph, lit)),
                    vec![*lhs],
                ),
            ));
            info.var_term.insert(*lhs, newterm);

            let ruletrm = get_fresh();
            res.push(NormAction::Let(
                ruletrm,
                NormExpr::Call("RuleTerm__".into(), vec![rule_proof, newterm]),
            ));

            let trmprf = get_fresh();
            res.push(NormAction::Let(
                trmprf,
                NormExpr::Call("MakeTrmPrf__".into(), vec![newterm, ruletrm]),
            ));

            res.push(NormAction::Set(
                make_rep_version(&literal_name(egraph, lit)),
                vec![*lhs],
                trmprf,
            ));
        }
    }
}

fn add_rule_proof(
    rule_name: Symbol,
    info: &ProofInfo,
    facts: &Vec<NormFact>,
    res: &mut Vec<NormAction>,
    get_fresh: &mut Fresh,
) -> Symbol {
    let mut current_proof = get_fresh();
    res.push(NormAction::Let(
        current_proof,
        NormExpr::Call("Null__".into(), vec![]),
    ));

    for fact in facts {
        match fact {
            NormFact::Assign(lhs, _rhs) => {
                let fresh = get_fresh();
                res.push(NormAction::Let(
                    fresh,
                    NormExpr::Call("Cons__".into(), vec![info.var_proof[lhs], current_proof]),
                ));
                current_proof = fresh;
            }
            // same as Assign case
            NormFact::AssignLit(lhs, _rhs) => {
                let fresh = get_fresh();
                res.push(NormAction::Let(
                    fresh,
                    NormExpr::Call("Cons__".into(), vec![info.var_proof[lhs], current_proof]),
                ));
                current_proof = fresh;
            }
            NormFact::ConstrainEq(lhs, rhs) => {
                let pfresh = get_fresh();
                res.push(NormAction::Let(
                    pfresh,
                    NormExpr::Call(
                        "DemandEq__".into(),
                        vec![info.var_term[lhs], info.var_term[rhs]],
                    ),
                ));
            }
        }
    }

    let name_const = get_fresh();
    res.push(NormAction::LetLit(name_const, Literal::String(rule_name)));
    let rule_proof = get_fresh();
    res.push(NormAction::Let(
        rule_proof,
        NormExpr::Call("Rule__".into(), vec![current_proof, name_const]),
    ));
    rule_proof
}

fn instrument_rule(egraph: &EGraph, rule: &NormRule, rule_name: Symbol) -> Rule {
    let mut varcount = 0;
    let mut get_fresh = move || {
        varcount += 1;
        Symbol::from(format!("pvar{}__", varcount))
    };

    let (mut info, facts) = instrument_facts(egraph, &rule.body, &mut get_fresh);

    let mut actions = rule.head.clone();
    let rule_proof = add_rule_proof(rule_name, &info, &rule.body, &mut actions, &mut get_fresh);

    for action in &rule.head {
        add_action_proof(
            rule_proof,
            &mut info,
            action,
            &mut actions,
            &mut get_fresh,
            egraph,
        );
    }

    // res.head.extend();
    let res = Rule {
        head: actions.iter().map(|a| a.to_action()).collect(),
        body: facts,
    };
    res
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
    for _i in 0..config.limit {
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
pub(crate) fn add_proofs(egraph: &EGraph, program: Vec<NormCommand>) -> Vec<NormCommand> {
    let mut res = proof_header(egraph);

    res.extend(make_ast_primitives(egraph));

    for command in program {
        match &command {
            NormCommand::Sort(name, presort_and_args) => {
                res.push(command.to_command());
                res.push(Command::Sort(
                    make_ast_version(egraph, name),
                    presort_and_args.clone(),
                ));
            }
            NormCommand::Function(fdecl) => {
                res.push(command.to_command());
                res.push(Command::Function(make_ast_func(egraph, fdecl)));
                res.push(Command::Function(make_rep_func(egraph, fdecl)));
                res.push(make_getchild_rule(egraph, fdecl));
            }
            NormCommand::NormRule(ruleset, rule) => {
                res.push(Command::Rule(
                    *ruleset,
                    instrument_rule(egraph, rule, "TODOrulename".into()),
                ));
            }
            NormCommand::Run(config) => {
                res.extend(make_runner(config));
            }
            _ => res.push(command.to_command()),
        }
    }

    desugar_program(egraph, res).unwrap()
}

pub(crate) fn should_add_proofs(_program: &[NormCommand]) -> bool {
    true
}