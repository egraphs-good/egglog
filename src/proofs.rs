use crate::*;

use crate::desugar::{desugar_commands, literal_name, Desugar};
use symbolic_expressions::Sexp;

fn proof_header(egraph: &EGraph) -> Vec<Command> {
    let str = include_str!("proofheader.egg");
    egraph.parse_program(str).unwrap()
}

// primitives don't need type info
fn make_ast_version_prim(proof_state: &ProofState, name: Symbol) -> Symbol {
    make_ast_version(proof_state, name, vec![])
}

fn make_ast_version(_proof_state: &ProofState, name: Symbol, input_types: Vec<Symbol>) -> Symbol {
    Symbol::from(format!("Ast{}_{}__", name, ListDisplay(input_types, "_"),))
}

fn make_rep_version(name: &Symbol, proof_state: &ProofState) -> Symbol {
    Symbol::from(format!(
        "{}Rep_{}_{}__",
        name,
        ListDisplay(
            proof_state
                .desugar
                .func_types
                .get(name)
                .unwrap()
                .input
                .clone(),
            "_"
        ),
        proof_state.desugar.func_types.get(name).unwrap().output,
    ))
}

// representatives for primitive values
fn make_rep_version_prim(name: &Symbol) -> Symbol {
    Symbol::from(format!("{}Rep__", name))
}

fn setup_primitives(proof_state: &ProofState) -> Vec<Command> {
    let mut commands = vec![];
    commands.extend(make_ast_primitives_funcs(proof_state));
    commands.extend(make_ast_primitives_sorts(proof_state));
    commands.extend(make_rep_primitive_sorts(proof_state));
    commands.extend(make_rep_primitive_funcs(proof_state));
    commands
}

fn prim_input_types(prim: &Primitive) -> Vec<Symbol> {
    prim.get_type()
        .0
        .iter()
        .map(|x| x.name())
        .collect::<Vec<Symbol>>()
}

fn make_rep_primitive_funcs(proof_state: &ProofState) -> Vec<Command> {
    let mut res = vec![];
    for (name, primitives) in &proof_state.desugar.egraph.type_info.primitives {
        for prim in primitives {
            res.push(Command::Function(FunctionDecl {
                name: make_rep_version(name, proof_state),
                schema: Schema {
                    input: vec![make_rep_version_prim(name); prim.get_type().0.len()],
                    output: "TrmPrf__".into(),
                },
                // Right now we just union every proof of some primitive.
                merge: None,
                merge_action: vec![],
                default: None,
                cost: None,
            }))
        }
    }

    res
}

fn make_rep_primitive_sorts(proof_state: &ProofState) -> Vec<Command> {
    proof_state.desugar.egraph
        .type_info
        .sorts
        .iter()
        .map(|(name, _)| {
            Command::Function(FunctionDecl {
                name: make_rep_version_prim(name),
                schema: Schema {
                    input: vec![*name],
                    output: "TrmPrf__".into(),
                },
                // Right now we just union every proof of some primitive.
                merge: None,
                merge_action: vec![],
                default: None,
                cost: None,
            })
        })
        .collect()
}

fn make_ast_primitives_funcs(proof_state: &ProofState) -> Vec<Command> {
    let mut res = vec![];
    for (name, primitives) in &proof_state.desugar.egraph.type_info.primitives {
        for prim in primitives {
            res.push(Command::Function(FunctionDecl {
                name: make_ast_version(proof_state, *name, prim_input_types(prim)),
                schema: Schema {
                    input: vec!["Ast__".into(); prim.get_type().0.len()],
                    output: "Ast__".into(),
                },
                merge: None,
                merge_action: vec![],
                default: None,
                cost: None,
            }));
        }
    }
    res
}

fn make_ast_primitives_sorts(proof_state: &ProofState) -> Vec<Command> {
    proof_state.desugar.egraph
        .type_info
        .sorts
        .iter()
        .map(|(name, _)| {
            Command::Function(FunctionDecl {
                name: make_ast_version_prim(proof_state, *name),
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

fn make_ast_func(proof_state: &ProofState, fdecl: &FunctionDecl) -> FunctionDecl {
    FunctionDecl {
        name: make_ast_version(proof_state, fdecl.name, fdecl.schema.input.clone()),
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
    body: &Vec<NormFact>,
    proof_state: &mut ProofState,
) -> (ProofInfo, Vec<Fact>) {
    let mut info: ProofInfo = Default::default();
    let mut facts: Vec<Fact> = body.iter().map(|f| f.to_fact()).collect();

    for fact in body {
        println!("fact: {}", fact.to_fact());
        match fact {
            NormFact::AssignLit(lhs, rhs) => {
                let literal_name = literal_name(&proof_state.desugar, rhs);
                let rep = proof_state.get_fresh();
                let rep_trm = proof_state.get_fresh();
                let rep_prf = proof_state.get_fresh();
                facts.push(Fact::Eq(vec![
                    Expr::Var(rep),
                    Expr::Call(make_rep_version_prim(&literal_name), vec![Expr::Var(*lhs)]),
                ]));
                facts.push(Fact::Eq(vec![
                    Expr::Var(rep_trm),
                    Expr::Call("TrmOf__".into(), vec![Expr::Var(rep)]),
                ]));
                facts.push(Fact::Eq(vec![
                    Expr::Var(rep_prf),
                    Expr::Call("PrfOf__".into(), vec![Expr::Var(rep)]),
                ]));

                assert!(info.var_term.insert(*lhs, rep_trm).is_none());
                assert!(info.var_proof.insert(*lhs, rep_prf).is_none());
            }
            NormFact::Compute(lhs, NormExpr::Call(head, body)) => {
                let rep = proof_state.get_fresh();
                let rep_trm = proof_state.get_fresh();
                let rep_prf = proof_state.get_fresh();

                facts.push(Fact::Eq(vec![
                    Expr::Var(rep),
                    Expr::Call(
                        make_rep_version_prim(head),
                        body.iter().map(|x| Expr::Var(*x)).collect(),
                    ),
                ]));
                facts.push(Fact::Eq(vec![
                    Expr::Var(rep_trm),
                    Expr::Call("TrmOf__".into(), vec![Expr::Var(rep)]),
                ]));
                facts.push(Fact::Eq(vec![
                    Expr::Var(rep_prf),
                    Expr::Call("PrfOf__".into(), vec![Expr::Var(rep)]),
                ]));

                assert!(info.var_term.insert(*lhs, rep_trm).is_none());
                assert!(info.var_proof.insert(*lhs, rep_prf).is_none());

                for (i, child) in body.iter().enumerate() {
                    //println!("child: {:?}", child);
                    let child_trm = proof_state.get_fresh();
                    let const_var = proof_state.get_fresh();
                    facts.push(Fact::Eq(vec![
                        Expr::Var(const_var),
                        Expr::Lit(Literal::Int(i as i64)),
                    ]));

                    facts.push(Fact::Eq(vec![
                        Expr::Var(child_trm),
                        Expr::Call(
                            "GetChild__".into(),
                            vec![Expr::Var(rep_trm), Expr::Var(const_var)],
                        ),
                    ]));
                    assert!(info.var_term.insert(*child, child_trm).is_none());
                }
            }
            NormFact::Assign(lhs, NormExpr::Call(head, body)) => {
                let rep = proof_state.get_fresh();
                let rep_trm = proof_state.get_fresh();
                let rep_prf = proof_state.get_fresh();

                facts.push(Fact::Eq(vec![
                    Expr::Var(rep),
                    Expr::Call(
                        make_rep_version(head, proof_state),
                        body.iter().map(|x| Expr::Var(*x)).collect(),
                    ),
                ]));
                facts.push(Fact::Eq(vec![
                    Expr::Var(rep_trm),
                    Expr::Call("TrmOf__".into(), vec![Expr::Var(rep)]),
                ]));
                facts.push(Fact::Eq(vec![
                    Expr::Var(rep_prf),
                    Expr::Call("PrfOf__".into(), vec![Expr::Var(rep)]),
                ]));

                assert!(info.var_term.insert(*lhs, rep_trm).is_none());
                assert!(info.var_proof.insert(*lhs, rep_prf).is_none());

                for (i, child) in body.iter().enumerate() {
                    //println!("child: {:?}", child);
                    let child_trm = proof_state.get_fresh();
                    let const_var = proof_state.get_fresh();
                    facts.push(Fact::Eq(vec![
                        Expr::Var(const_var),
                        Expr::Lit(Literal::Int(i as i64)),
                    ]));

                    facts.push(Fact::Eq(vec![
                        Expr::Var(child_trm),
                        Expr::Call(
                            "GetChild__".into(),
                            vec![Expr::Var(rep_trm), Expr::Var(const_var)],
                        ),
                    ]));
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
    proof_state: &mut ProofState,
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
            let newterm = proof_state.get_fresh();
            res.push(NormAction::Let(
                newterm,
                NormExpr::Call(
                    make_ast_version(
                        proof_state,
                        *head,
                        proof_state
                            .desugar
                            .func_types
                            .get(head)
                            .unwrap()
                            .input
                            .clone(),
                    ),
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
            let newterm = proof_state.get_fresh();
            // make the term for this variable
            res.push(NormAction::Let(
                newterm,
                NormExpr::Call(
                    make_ast_version(
                        proof_state,
                        *rhsname,
                        proof_state
                            .desugar
                            .func_types
                            .get(rhsname)
                            .unwrap()
                            .input
                            .clone(),
                    ),
                    rhsvars
                        .iter()
                        .map(|v| *info.var_term.get(v).unwrap())
                        .collect(),
                ),
            ));
            info.var_term.insert(*lhs, newterm);

            let ruletrm = proof_state.get_fresh();
            res.push(NormAction::Let(
                ruletrm,
                NormExpr::Call("RuleTerm__".into(), vec![rule_proof, newterm]),
            ));

            let trmprf = proof_state.get_fresh();
            res.push(NormAction::Let(
                trmprf,
                NormExpr::Call("MakeTrmPrf__".into(), vec![newterm, ruletrm]),
            ));

            res.push(NormAction::Set(
                make_rep_version(rhsname, proof_state),
                rhsvars.clone(),
                trmprf,
            ));
        }
        // very similar to let case
        NormAction::LetLit(lhs, lit) => {
            let newterm = proof_state.get_fresh();
            // make the term for this variable
            res.push(NormAction::Let(
                newterm,
                NormExpr::Call(
                    make_ast_version_prim(proof_state, literal_name(&proof_state.desugar, lit)),
                    vec![*lhs],
                ),
            ));
            info.var_term.insert(*lhs, newterm);

            let ruletrm = proof_state.get_fresh();
            res.push(NormAction::Let(
                ruletrm,
                NormExpr::Call("RuleTerm__".into(), vec![rule_proof, newterm]),
            ));

            let trmprf = proof_state.get_fresh();
            res.push(NormAction::Let(
                trmprf,
                NormExpr::Call("MakeTrmPrf__".into(), vec![newterm, ruletrm]),
            ));

            res.push(NormAction::Set(
                make_rep_version_prim(&literal_name(&proof_state.desugar, lit)),
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
    proof_state: &mut ProofState,
) -> Symbol {
    let mut current_proof = proof_state.get_fresh();
    res.push(NormAction::Let(
        current_proof,
        NormExpr::Call("Null__".into(), vec![]),
    ));

    for fact in facts {
        match fact {
            NormFact::Assign(lhs, _rhs) | NormFact::Compute(lhs, _rhs) => {
                let fresh = proof_state.get_fresh();
                res.push(NormAction::Let(
                    fresh,
                    NormExpr::Call("Cons__".into(), vec![info.var_proof[lhs], current_proof]),
                ));
                current_proof = fresh;
            }
            // same as Assign case
            NormFact::AssignLit(lhs, _rhs) => {
                let fresh = proof_state.get_fresh();
                res.push(NormAction::Let(
                    fresh,
                    NormExpr::Call("Cons__".into(), vec![info.var_proof[lhs], current_proof]),
                ));
                current_proof = fresh;
            }
            NormFact::ConstrainEq(lhs, rhs) => {
                let pfresh = proof_state.get_fresh();
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

    let name_const = proof_state.get_fresh();
    res.push(NormAction::LetLit(name_const, Literal::String(rule_name)));
    let rule_proof = proof_state.get_fresh();
    res.push(NormAction::Let(
        rule_proof,
        NormExpr::Call("Rule__".into(), vec![current_proof, name_const]),
    ));
    rule_proof
}

fn instrument_rule(
    rule: &NormRule,
    rule_name: Symbol,
    proof_state: &mut ProofState,
) -> Rule {
    let (mut info, facts) = instrument_facts(&rule.body, proof_state);

    let mut actions = rule.head.clone();
    let rule_proof = add_rule_proof(rule_name, &info, &rule.body, &mut actions, proof_state);

    for action in &rule.head {
        add_action_proof(
            rule_proof,
            &mut info,
            action,
            &mut actions,
            proof_state,
        );
    }

    // res.head.extend();
    let res = Rule {
        head: actions.iter().map(|a| a.to_action()).collect(),
        body: facts,
    };
    res
}

fn make_rep_func(proof_state: &ProofState, fdecl: &FunctionDecl) -> FunctionDecl {
    FunctionDecl {
        name: make_rep_version(&fdecl.name, proof_state),
        schema: Schema {
            input: fdecl.schema.input.clone(),
            output: "TrmPrf__".into(),
        },
        merge: Some(Expr::Var("old".into())),
        merge_action: merge_action(proof_state.desugar.egraph, fdecl),
        default: None,
        cost: None,
    }
}

fn make_getchild_rule(proof_state: &ProofState, fdecl: &FunctionDecl) -> Command {
    let getchild = |i| Symbol::from(format!("c{}__", i));
    Command::Rule(
        "proofrules__".into(),
        Rule {
            body: vec![Fact::Eq(vec![
                Expr::Var("ast__".into()),
                Expr::Call(
                    make_ast_version(proof_state, fdecl.name, fdecl.schema.input.clone()),
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

pub(crate) struct ProofState<'a> {
    pub(crate) global_var_ast: HashMap<Symbol, Symbol>,
    pub(crate) desugar: Desugar<'a>,
}

impl<'a> ProofState<'a> {
    pub(crate) fn get_fresh(&mut self) -> Symbol {
        (self.desugar.get_fresh)()
    }
}

fn proof_original_action(action: &NormAction, proof_state: &mut ProofState) -> Vec<Command> {
    match action {
        NormAction::Let(lhs, NormExpr::Call(head, body)) => {
            let ast_var = proof_state.get_fresh();
            proof_state.global_var_ast.insert(*lhs, ast_var);
            let ast_action = format!(
                "(let {} ({} {}))",
                ast_var,
                make_ast_version(
                    &proof_state,
                    *head,
                    proof_state.desugar.func_types[head].input.clone()
                ),
                ListDisplay(body.iter().map(|e| proof_state.global_var_ast[e]), " ")
            );

            vec![
                Command::Action(
                    proof_state
                        .desugar
                        .egraph
                        .action_parser
                        .parse(&ast_action)
                        .unwrap(),
                ),
                Command::Action(
                    proof_state
                        .desugar
                        .egraph
                        .action_parser
                        .parse(&format!(
                            "(set ({} {})
                         (MakeTrmPrf__ {} (Original__ {})))",
                            make_rep_version(head, proof_state),
                            ListDisplay(body, " "),
                            ast_var,
                            ast_var
                        ))
                        .unwrap(),
                ),
            ]
        }
        NormAction::LetVar(var1, var2) => {
            proof_state
                .global_var_ast
                .insert(*var1, proof_state.global_var_ast[var2]);
            vec![]
        }
        NormAction::LetLit(lhs, literal) => {
            let ast_var = proof_state.get_fresh();
            proof_state.global_var_ast.insert(*lhs, ast_var);
            vec![
                Command::Action(
                    proof_state
                        .desugar
                        .egraph
                        .action_parser
                        .parse(&format!(
                            "(let {} ({} {}))",
                            ast_var,
                            make_ast_version_prim(
                                &proof_state,
                                literal_name(&proof_state.desugar, literal)
                            ),
                            literal
                        ))
                        .unwrap(),
                ),
                Command::Action(
                    proof_state
                        .desugar
                        .egraph
                        .action_parser
                        .parse(&format!(
                            "(set ({} {})
                         (MakeTrmPrf__ {} (Original__ {})))",
                            make_rep_version_prim(&literal_name(
                                &proof_state.desugar,
                                literal
                            )),
                            literal,
                            ast_var,
                            ast_var
                        ))
                        .unwrap(),
                ),
            ]
        }
        NormAction::Set(head, body, var) => {
            let left_ast = Expr::Call(
                make_ast_version(
                    &proof_state,
                    *head,
                    proof_state.desugar.func_types[head].input.clone(),
                ),
                body.iter()
                    .map(|e| Expr::Var(proof_state.global_var_ast[e]))
                    .collect(),
            );
            vec![Command::Action(
                proof_state
                    .desugar
                    .egraph
                    .action_parser
                    .parse(&format!(
                        "(set (EqGraph__ {} {}) (OriginalEq__ {} {}))",
                        left_ast,
                        proof_state.global_var_ast[var],
                        left_ast,
                        proof_state.global_var_ast[var]
                    ))
                    .unwrap(),
            )]
        }
        NormAction::Union(var1, var2) => {
            vec![Command::Action(
                proof_state
                    .desugar
                    .egraph
                    .action_parser
                    .parse(&format!(
                        "(set (EqGraph__ {} {}) (OriginalEq__ {} {}))",
                        proof_state.global_var_ast[var1],
                        proof_state.global_var_ast[var2],
                        proof_state.global_var_ast[var1],
                        proof_state.global_var_ast[var2]
                    ))
                    .unwrap(),
            )]
        }
        NormAction::Delete(..) | NormAction::Panic(..) => vec![],
    }
}

// the egraph is the initial egraph with only default sorts
pub(crate) fn add_proofs(program: Vec<NormCommand>, desugar: Desugar) -> Vec<NormCommand> {
    let mut res = proof_header(&desugar.egraph);
    let mut proof_state = ProofState {
        global_var_ast: Default::default(),
        desugar,
    };

    res.extend(setup_primitives(&proof_state));

    for command in program {
        match &command.command {
            NCommand::Sort(_name, _presort_and_args) => {
                res.push(command.to_command());
            }
            NCommand::Function(fdecl) => {
                res.push(command.to_command());
                res.push(Command::Function(make_ast_func(
                    &proof_state,
                    fdecl,
                )));
                res.push(Command::Function(make_rep_func(&proof_state, fdecl)));
                res.push(make_getchild_rule(&proof_state, fdecl));
            }
            NCommand::NormRule(ruleset, rule) => {
                res.push(Command::Rule(
                    *ruleset,
                    instrument_rule(
                        rule,
                        "TODOrulename".into(),
                        &mut proof_state,
                    ),
                ));
            }
            NCommand::Run(config) => {
                res.extend(make_runner(config));
            }
            NCommand::NormAction(action) => {
                res.push(Command::Action(action.to_action()));
                res.extend(proof_original_action(action, &mut proof_state));
            }
            _ => res.push(command.to_command()),
        }
    }

    let mut desugar = Desugar {
        func_types: Default::default(),
        let_types: Default::default(),
        get_fresh: proof_state.desugar.get_fresh,
        get_new_id: proof_state.desugar.get_new_id,
        egraph: proof_state.desugar.egraph,
    };

    desugar_commands(res, &mut desugar).unwrap()
}

pub(crate) fn should_add_proofs(_program: &[NormCommand]) -> bool {
    false
}
