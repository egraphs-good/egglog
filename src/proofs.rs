use crate::*;

use crate::desugar::Desugar;
use crate::typechecking::FuncType;
use symbolic_expressions::Sexp;

pub const RULE_PROOF_KEYWORD: &str = "rule-proof";

// primitives don't need type info
fn make_ast_version_prim(name: Symbol) -> Symbol {
    Symbol::from(format!("Ast{}__", name))
}

fn make_ast_version(proof_state: &mut ProofState, expr: &NormExpr) -> Symbol {
    let NormExpr::Call(name, _) = expr;
    let types = proof_state
        .type_info
        .typecheck_expr(proof_state.current_ctx, expr, true)
        .unwrap();
    Symbol::from(format!(
        "Ast{}_{}__",
        name,
        ListDisplay(types.input.iter().map(|sort| sort.name()), "_"),
    ))
}

fn make_rep_version(proof_state: &mut ProofState, expr: &NormExpr) -> Symbol {
    let NormExpr::Call(name, _) = expr;
    let types = proof_state
        .type_info
        .typecheck_expr(proof_state.current_ctx, expr, true)
        .unwrap();
    Symbol::from(format!(
        "Rep{}_{}__",
        name,
        ListDisplay(types.input.iter().map(|sort| sort.name()), "_"),
    ))
}

// representatives for primitive values
fn make_rep_version_prim(name: &Symbol) -> Symbol {
    Symbol::from(format!("Rep{}__", name))
}

fn setup_primitives() -> Vec<Command> {
    let mut commands = vec![];
    let fresh_types = TypeInfo::new();
    commands.extend(make_ast_primitives_sorts(&fresh_types));
    commands.extend(make_rep_primitive_sorts(&fresh_types));
    commands
}

fn make_rep_primitive_sorts(type_info: &TypeInfo) -> Vec<Command> {
    type_info
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

fn make_ast_primitives_sorts(type_info: &TypeInfo) -> Vec<Command> {
    type_info
        .sorts
        .iter()
        .map(|(name, _)| {
            Command::Function(FunctionDecl {
                name: make_ast_version_prim(*name),
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
fn make_ast_function(proof_state: &mut ProofState, expr: &NormExpr) -> FunctionDecl {
    let NormExpr::Call(_head, body) = expr;
    FunctionDecl {
        name: make_ast_version(proof_state, expr),
        schema: Schema {
            input: body.iter().map(|_sort| "Ast__".into()).collect(),
            output: "Ast__".into(),
        },
        merge: None,
        merge_action: vec![],
        default: None,
        cost: None,
    }
}

fn merge_action(proof_state: &mut ProofState, types: FuncType) -> Vec<Action> {
    let child1 = |i| Symbol::from(format!("c1_{}__", i));
    let child2 = |i| Symbol::from(format!("c2_{}__", i));

    let mut congr_prf = Sexp::String("Null__".to_string());
    for i in 0..types.input.len() {
        let current = types.input.len() - i - 1;
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
    let t1 = proof_state.get_fresh();
    let t2 = proof_state.get_fresh();
    let p1 = proof_state.get_fresh();

    vec![
        format!("(let {t1} (TrmOf__ old))"),
        format!("(let {t2} (TrmOf__ new))"),
        format!("(let {p1} (PrfOf__ old))"),
    ]
    .into_iter()
    .chain(types.input.iter().enumerate().flat_map(|(i, _sort)| {
        vec![
            format!("(let {} (GetChild__ {t1} {}))", child1(i), i),
            format!("(let {} (GetChild__ {t2} {}))", child2(i), i),
        ]
    }))
    .chain(vec![
        format!("(let congr_prf__ (Congruence__ {p1} {}))", congr_prf),
        format!("(set (EqGraph__ {t1} {t2}) congr_prf__)"),
        format!("(set (EqGraph__ {t2} {t1}) (Flip__ congr_prf__))"),
    ])
    .map(|s| proof_state.desugar.action_parser.parse(&s).unwrap())
    .collect()
}

#[derive(Clone, Debug)]
pub(crate) struct ProofInfo {
    // proof for each variable bound in an assignment (lhs or rhs)
    pub var_term: HashMap<Symbol, Symbol>,
    // proofs for each variable
    pub var_proof: HashMap<Symbol, Symbol>,
    pub rule_proof: Option<Symbol>,
    pub rule_proof_ast: Option<Symbol>,
}

// This function makes use of the property that the body is Norm
// variables appear at most once (including the rhs of assignments)
// besides when they appear in constraints
fn instrument_facts(
    body: &Vec<NormFact>,
    proof_state: &mut ProofState,
    actions: &mut Vec<NormAction>,
) -> ProofInfo {
    let mut info: ProofInfo = ProofInfo {
        var_term: Default::default(),
        var_proof: Default::default(),
        rule_proof: None,
        rule_proof_ast: None,
    };

    for fact in body {
        match fact {
            NormFact::AssignLit(lhs, rhs) => {
                let literal_name = proof_state.literal_name(rhs);
                let rep_trm = proof_state.get_fresh();
                let rep_prf = proof_state.get_fresh();
                actions.push(NormAction::Let(
                    rep_trm,
                    NormExpr::Call(make_ast_version_prim(literal_name), vec![*lhs]),
                ));
                actions.push(NormAction::Let(
                    rep_prf,
                    NormExpr::Call("ComputePrim__".into(), vec![rep_trm]),
                ));

                info.var_term.insert(*lhs, rep_trm);
                assert!(info.var_proof.insert(*lhs, rep_prf).is_none());
            }
            NormFact::Assign(lhs, NormExpr::Call(head, body))
                if proof_state.type_info.is_primitive(*head) =>
            {
                // child terms should already exist if we are computing something
                let rep_trm = proof_state.get_fresh();
                actions.push(NormAction::Let(
                    rep_trm,
                    NormExpr::Call(
                        make_ast_version(proof_state, &NormExpr::Call(*head, body.clone())),
                        body.iter()
                            .map(|v| get_var_term(*v, proof_state, &info))
                            .collect(),
                    ),
                ));

                let rep_prf = proof_state.get_fresh();

                actions.push(NormAction::Let(
                    rep_prf,
                    NormExpr::Call("ComputePrim__".into(), vec![rep_trm]),
                ));
                info.var_term.insert(*lhs, rep_trm);
                info.var_proof.insert(*lhs, rep_prf);
            }
            NormFact::Assign(lhs, NormExpr::Call(head, body)) => {
                let rep = proof_state.get_fresh();
                let rep_trm = proof_state.get_fresh();
                let rep_prf = proof_state.get_fresh();

                actions.push(NormAction::Let(
                    rep,
                    NormExpr::Call(
                        make_rep_version(proof_state, &NormExpr::Call(*head, body.clone())),
                        body.clone(),
                    ),
                ));
                actions.push(NormAction::Let(
                    rep_trm,
                    NormExpr::Call("TrmOf__".into(), vec![rep]),
                ));
                actions.push(NormAction::Let(
                    rep_prf,
                    NormExpr::Call("PrfOf__".into(), vec![rep]),
                ));

                info.var_term.insert(*lhs, rep_trm);
                assert!(info.var_proof.insert(*lhs, rep_prf).is_none());

                for (i, child) in body.iter().enumerate() {
                    let child_trm = proof_state.get_fresh();
                    let const_var = proof_state.get_fresh();
                    actions.push(NormAction::LetLit(const_var, Literal::Int(i as i64)));
                    actions.push(NormAction::Let(
                        child_trm,
                        NormExpr::Call("GetChild__".into(), vec![rep_trm, const_var]),
                    ));
                    info.var_term.insert(*child, child_trm);
                }
            }
            NormFact::ConstrainEq(lhs, rhs) => {
                // variables need to have an ast so they can be computed on
                // but if they are used in a binding instead of a computation, this proof and term is overridden in the hashmap
                if let Some(term) = get_var_term_option(*rhs, proof_state, &info) {
                    if get_var_term_option(*lhs, proof_state, &info).is_none() {
                        assert!(info.var_term.insert(*lhs, term).is_none());
                    }
                } else if let Some(term) = get_var_term_option(*lhs, proof_state, &info) {
                    if get_var_term_option(*rhs, proof_state, &info).is_none() {
                        assert!(info.var_term.insert(*rhs, term).is_none());
                    }
                } else {
                    panic!(
                        "Contraint without representative term for at least one side {} = {}",
                        lhs, rhs
                    );
                }
            }
        }
    }

    // now fill in representitive terms for any aliases
    for fact in body {
        if let NormFact::ConstrainEq(lhs, rhs) = fact {
            let lhsterm = get_var_term_option(*lhs, proof_state, &info);
            let rhsterm = get_var_term_option(*rhs, proof_state, &info);
            if let Some(rep_term) = lhsterm {
                if rhsterm.is_none() {
                    info.var_term.insert(*rhs, rep_term);
                }
            } else if let Some(rep_term) = rhsterm {
                info.var_term.insert(*lhs, rep_term);
            } else {
                panic!(
                    "Contraint without representative term for at least one side {} = {}",
                    lhs, rhs
                );
            }
        }
    }

    info
}

fn get_var_term_option(
    var: Symbol,
    proof_state: &ProofState,
    proof_info: &ProofInfo,
) -> Option<Symbol> {
    if var == RULE_PROOF_KEYWORD.into() {
        return Some(proof_info.rule_proof_ast.unwrap());
    }
    proof_info
        .var_term
        .get(&var)
        .or_else(|| proof_state.global_var_ast.get(&var))
        .cloned()
}

fn get_var_term(var: Symbol, proof_state: &ProofState, proof_info: &ProofInfo) -> Symbol {
    get_var_term_option(var, proof_state, proof_info).unwrap()
}

fn add_eqgraph_equality(
    astvar1: Symbol,
    astvar2: Symbol,
    rule_proof: Symbol,
    res: &mut Vec<NormAction>,
) {
    res.push(NormAction::Set(
        NormExpr::Call("EqGraph__".into(), vec![astvar1, astvar2]),
        rule_proof,
    ));
    res.push(NormAction::Set(
        NormExpr::Call("EqGraph__".into(), vec![astvar2, astvar1]),
        rule_proof,
    ));
}

fn make_expr_ast(
    proof_state: &mut ProofState,
    proof_info: &ProofInfo,
    expr: &NormExpr,
    res: &mut Vec<NormAction>,
) -> Symbol {
    let NormExpr::Call(head, body) = expr;
    let newterm = proof_state.get_fresh();
    // make the term for this variable
    res.push(NormAction::Let(
        newterm,
        NormExpr::Call(
            make_ast_version(proof_state, &NormExpr::Call(*head, body.clone())),
            body.iter()
                .map(|v| get_var_term(*v, proof_state, proof_info))
                .collect(),
        ),
    ));

    newterm
}

fn make_expr_rep(
    proof_state: &mut ProofState,
    proof_info: &ProofInfo,
    expr: &NormExpr,
    res: &mut Vec<NormAction>,
) -> Symbol {
    let NormExpr::Call(head, body) = expr;
    let newterm = make_expr_ast(proof_state, proof_info, expr, res);

    let ruletrm = proof_state.get_fresh();
    res.push(NormAction::Let(
        ruletrm,
        NormExpr::Call(
            "RuleTerm__".into(),
            vec![proof_info.rule_proof.unwrap(), newterm],
        ),
    ));

    let trmprf = proof_state.get_fresh();
    res.push(NormAction::Let(
        trmprf,
        NormExpr::Call("MakeTrmPrf__".into(), vec![newterm, ruletrm]),
    ));

    res.push(NormAction::Set(
        NormExpr::Call(
            make_rep_version(proof_state, &NormExpr::Call(*head, body.clone())),
            body.clone(),
        ),
        trmprf,
    ));
    newterm
}

fn add_action_proof(
    proof_info: &mut ProofInfo,
    action: &NormAction,
    res: &mut Vec<NormAction>,
    proof_state: &mut ProofState,
) {
    match action {
        NormAction::LetVar(var1, var2) => {
            // update var1's term
            proof_info
                .var_term
                .insert(*var1, get_var_term(*var2, proof_state, proof_info));
        }
        NormAction::Delete(..) | NormAction::Panic(..) => (),
        NormAction::Union(var1, var2) => {
            add_eqgraph_equality(
                get_var_term(*var1, proof_state, proof_info),
                get_var_term(*var2, proof_state, proof_info),
                proof_info.rule_proof.unwrap(),
                res,
            );
        }
        NormAction::Set(expr, rhs) => {
            let new_term = make_expr_rep(proof_state, proof_info, expr, res);
            // add to the equality graph when we set things equal to each other
            add_eqgraph_equality(
                new_term,
                get_var_term(*rhs, proof_state, proof_info),
                proof_info.rule_proof.unwrap(),
                res,
            )
        }
        NormAction::Let(lhs, expr) => {
            let ast = make_expr_rep(proof_state, proof_info, expr, res);
            proof_info.var_term.insert(*lhs, ast);
        }
        // very similar to let case
        NormAction::LetLit(lhs, lit) => {
            let newterm = proof_state.get_fresh();
            // make the term for this variable
            res.push(NormAction::Let(
                newterm,
                NormExpr::Call(
                    make_ast_version_prim(proof_state.literal_name(lit)),
                    vec![*lhs],
                ),
            ));
            proof_info.var_term.insert(*lhs, newterm);

            let ruletrm = proof_state.get_fresh();
            res.push(NormAction::Let(
                ruletrm,
                NormExpr::Call(
                    "RuleTerm__".into(),
                    vec![proof_info.rule_proof.unwrap(), newterm],
                ),
            ));

            let trmprf = proof_state.get_fresh();
            res.push(NormAction::Let(
                trmprf,
                NormExpr::Call("MakeTrmPrf__".into(), vec![newterm, ruletrm]),
            ));

            res.push(NormAction::Set(
                NormExpr::Call(
                    make_rep_version_prim(&proof_state.literal_name(lit)),
                    vec![*lhs],
                ),
                trmprf,
            ));
        }
    }
}

fn add_rule_proof(
    rule_name: Symbol,
    proof_info: &ProofInfo,
    facts: &Vec<NormFact>,
    res: &mut Vec<NormAction>,
    proof_state: &mut ProofState,
) -> Symbol {
    let mut current_proof = proof_state.get_fresh();
    res.push(NormAction::LetVar(current_proof, "Null__".into()));

    for fact in facts {
        match fact {
            NormFact::Assign(lhs, _rhs) => {
                let fresh = proof_state.get_fresh();
                res.push(NormAction::Let(
                    fresh,
                    NormExpr::Call(
                        "Cons__".into(),
                        vec![proof_info.var_proof[lhs], current_proof],
                    ),
                ));
                current_proof = fresh;
            }
            // same as Assign case
            NormFact::AssignLit(lhs, _rhs) => {
                let fresh = proof_state.get_fresh();
                res.push(NormAction::Let(
                    fresh,
                    NormExpr::Call(
                        "Cons__".into(),
                        vec![proof_info.var_proof[lhs], current_proof],
                    ),
                ));
                current_proof = fresh;
            }
            NormFact::ConstrainEq(lhs, rhs) => {
                let pfresh = proof_state.get_fresh();
                res.push(NormAction::Let(
                    pfresh,
                    NormExpr::Call(
                        "DemandEq__".into(),
                        vec![
                            get_var_term(*lhs, proof_state, proof_info),
                            get_var_term(*rhs, proof_state, proof_info),
                        ],
                    ),
                ));

                let fresh = proof_state.get_fresh();
                res.push(NormAction::Let(
                    fresh,
                    NormExpr::Call("Cons__".into(), vec![pfresh, current_proof]),
                ));
                current_proof = fresh;
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

// replace the rule-proof keyword with the proof of the rule
fn replace_rule_proof(actions: &[NormAction], rule_proof: Symbol) -> Vec<NormAction> {
    actions
        .iter()
        .map(|action| {
            action.map_def_use(&mut |var, _isdef| {
                if var == RULE_PROOF_KEYWORD.into() {
                    rule_proof
                } else {
                    var
                }
            })
        })
        .collect()
}

fn instrument_rule(rule: &NormRule, rule_name: Symbol, proof_state: &mut ProofState) -> Rule {
    let mut actions = vec![];
    let info = instrument_facts(&rule.body, proof_state, &mut actions);
    let rule_proof = add_rule_proof(rule_name, &info, &rule.body, &mut actions, proof_state);

    let rule_proof_ast = proof_state.get_fresh();
    actions.push(NormAction::Let(
        rule_proof_ast,
        NormExpr::Call("AstProof__".into(), vec![rule_proof]),
    ));

    actions.extend(replace_rule_proof(&rule.head, rule_proof));

    // make a new proofinfo with the rule_proof symbol added
    let mut proof_info = ProofInfo {
        var_term: info.var_term,
        var_proof: info.var_proof,
        rule_proof: Some(rule_proof),
        rule_proof_ast: Some(rule_proof_ast),
    };

    for action in &rule.head {
        add_action_proof(&mut proof_info, action, &mut actions, proof_state);
    }

    NormRule {
        head: actions,
        body: rule.body.clone(),
    }
    .to_rule()
}

fn make_rep_function(proof_state: &mut ProofState, expr: &NormExpr) -> FunctionDecl {
    let types = proof_state
        .type_info
        .typecheck_expr(proof_state.current_ctx, expr, true)
        .unwrap();
    FunctionDecl {
        name: make_rep_version(proof_state, expr),
        schema: Schema {
            input: types.input.iter().map(|sort| sort.name()).collect(),
            output: "TrmPrf__".into(),
        },
        merge: Some(Expr::Var("old".into())),
        merge_action: merge_action(proof_state, types),
        default: None,
        cost: None,
    }
}

fn make_getchild_rule(proof_state: &mut ProofState, expr: &NormExpr) -> Command {
    let NormExpr::Call(_name, body) = expr;
    let getchild = |i| Symbol::from(format!("c{}__", i));
    Command::Rule {
        ruleset: "proofrules__".into(),
        name: "".into(),
        rule: Rule {
            body: vec![Fact::Eq(vec![
                Expr::Var("ast__".into()),
                Expr::Call(
                    make_ast_version(proof_state, expr),
                    body.iter()
                        .enumerate()
                        .map(|(i, _)| Expr::Var(getchild(i)))
                        .collect(),
                ),
            ])],
            head: body
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
    }
}

#[derive(Clone)]
pub(crate) struct ProofState {
    pub(crate) global_var_ast: HashMap<Symbol, Symbol>,
    pub(crate) ast_funcs_created: HashSet<Symbol>,
    pub(crate) current_ctx: CommandId,
    pub(crate) desugar: Desugar,
    pub(crate) type_info: TypeInfo,
}

fn make_rep_command(proof_state: &mut ProofState, lhs: Symbol, expr: &NormExpr) -> Vec<Command> {
    let NormExpr::Call(head, body) = expr;
    let ast_var = proof_state.get_fresh();
    let ast_action = format!(
        "(let {} ({} {}))",
        ast_var,
        make_ast_version(proof_state, &NormExpr::Call(*head, body.clone())),
        ListDisplay(body.iter().map(|e| { proof_state.global_var_ast[e] }), " ")
    );
    proof_state.global_var_ast.insert(lhs, ast_var);
    let rep = make_rep_version(proof_state, expr);
    vec![
        Command::Action(
            proof_state
                .desugar
                .action_parser
                .parse(&ast_action)
                .unwrap(),
        ),
        Command::Action(
            proof_state
                .desugar
                .action_parser
                .parse(&format!(
                    "(set ({} {})
                         (MakeTrmPrf__ {} (Original__ {})))",
                    rep,
                    ListDisplay(body, " "),
                    ast_var,
                    ast_var
                ))
                .unwrap(),
        ),
    ]
}

fn proof_original_action(action: &NormAction, proof_state: &mut ProofState) -> Vec<Command> {
    match action {
        NormAction::Let(lhs, expr) => make_rep_command(proof_state, *lhs, expr),
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
                        .action_parser
                        .parse(&format!(
                            "(let {} ({} {}))",
                            ast_var,
                            make_ast_version_prim(proof_state.literal_name(literal)),
                            literal
                        ))
                        .unwrap(),
                ),
                Command::Action(
                    proof_state
                        .desugar
                        .action_parser
                        .parse(&format!(
                            "(set ({} {})
                         (MakeTrmPrf__ {} (Original__ {})))",
                            make_rep_version_prim(&proof_state.literal_name(literal)),
                            literal,
                            ast_var,
                            ast_var
                        ))
                        .unwrap(),
                ),
            ]
        }
        NormAction::Set(expr, var) => {
            let fresh = proof_state.get_fresh();
            let mut rep_commands = make_rep_command(proof_state, fresh, expr);

            rep_commands.push(Command::Action(
                proof_state
                    .desugar
                    .action_parser
                    .parse(&format!(
                        "(set (EqGraph__ {} {}) (OriginalEq__ {} {}))",
                        proof_state.global_var_ast[&fresh],
                        proof_state.global_var_ast[var],
                        proof_state.global_var_ast[&fresh],
                        proof_state.global_var_ast[var]
                    ))
                    .unwrap(),
            ));
            rep_commands
        }
        NormAction::Union(var1, var2) => {
            vec![Command::Action(
                proof_state
                    .desugar
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

fn instrument_schedule(schedule: &NormSchedule) -> Schedule {
    match schedule {
        NormSchedule::Saturate(schedule) => {
            Schedule::Saturate(Box::new(instrument_schedule(schedule)))
        }
        NormSchedule::Repeat(times, schedule) => {
            Schedule::Repeat(*times, Box::new(instrument_schedule(schedule)))
        }
        // We only do anything in the run case
        NormSchedule::Run(run_config) => Schedule::Sequence(vec![
            Schedule::Saturate(Box::new(Schedule::Run(RunConfig {
                ruleset: "proofrules__".into(),
                until: None,
                limit: 1,
            }))),
            Schedule::Run(run_config.to_run_config()),
        ]),
        NormSchedule::Sequence(schedules) => {
            Schedule::Sequence(schedules.iter().map(instrument_schedule).collect())
        }
    }
}

impl ProofState {
    pub fn parse_program(&self, input: &str) -> Result<Vec<Command>, Error> {
        self.desugar.parse_program(input)
    }

    pub fn new() -> Self {
        ProofState {
            type_info: TypeInfo::new(),
            ast_funcs_created: Default::default(),
            current_ctx: 0,
            global_var_ast: Default::default(),
            desugar: Desugar::new(),
        }
    }

    // TODO we need to also instrument merge actions and merge because they can add new terms that need representatives
    // the egraph is the initial egraph with only default sorts
    pub(crate) fn add_proofs(&mut self, program: Vec<NormCommand>) -> Vec<Command> {
        let mut res = vec![];

        for command in program {
            self.current_ctx = command.metadata.id;

            // first, set up any rep functions that we need
            command.command.map_exprs(&mut |expr| {
                let ast_name = make_ast_version(self, expr);
                if self.ast_funcs_created.insert(ast_name) {
                    let commands = vec![
                        Command::Function(make_ast_function(self, expr)),
                        Command::Function(make_rep_function(self, expr)),
                        make_getchild_rule(self, expr),
                    ];
                    res.extend(commands);
                }
                expr.clone()
            });

            match &command.command {
                NCommand::Push(_num) => {
                    res.push(command.to_command());
                }
                NCommand::Sort(_name, _presort_and_args) => {
                    res.push(command.to_command());
                }
                NCommand::Function(_fdecl) => {
                    res.push(command.to_command());
                }
                NCommand::NormRule {
                    ruleset,
                    name,
                    rule,
                } => {
                    res.push(Command::Rule {
                        ruleset: *ruleset,
                        name: *name,
                        rule: instrument_rule(rule, *name, self),
                    });
                }
                NCommand::NormAction(action) => {
                    res.push(Command::Action(action.to_action()));
                    res.extend(proof_original_action(action, self));
                }
                NCommand::Check(_facts) => {
                    res.push(command.to_command());
                }
                NCommand::RunSchedule(schedule) => {
                    res.push(Command::RunSchedule(instrument_schedule(schedule)));
                }
                _ => res.push(command.to_command()),
            }
        }

        res
    }

    pub(crate) fn get_fresh(&mut self) -> Symbol {
        self.desugar.get_fresh()
    }

    pub(crate) fn proof_header(&self) -> Vec<Command> {
        let str = include_str!("proofheader.egg");
        let rest_of_header = setup_primitives();
        self.parse_program(str)
            .unwrap()
            .into_iter()
            .chain(rest_of_header)
            .collect()
    }

    pub(crate) fn literal_name(&self, lit: &Literal) -> Symbol {
        self.type_info.infer_literal(lit).name()
    }
}
