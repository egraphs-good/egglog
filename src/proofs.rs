use crate::*;

use symbolic_expressions::Sexp;

fn proof_header(egraph: &EGraph) -> Vec<Command> {
    let str = include_str!("proofheader.egg");
    egraph.parse_program(str).unwrap()
}

fn make_ast_version(egraph: &EGraph, name: &Symbol) -> Symbol {
    if egraph.sorts.get(name).is_some() {
        name.clone()
    } else {
        Symbol::from(format!("Ast{}__", name))
    }
}

fn make_rep_version(name: &Symbol) -> Symbol {
    Symbol::from(format!("{}Rep__", name))
}

fn make_ast_func(egraph: &EGraph, fdecl: &FunctionDecl) -> FunctionDecl {
    FunctionDecl {
        name: make_ast_version(egraph, &fdecl.name),
        schema: Schema {
            input: fdecl
                .schema
                .input
                .iter()
                .map(|sort| make_ast_version(egraph, sort))
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
    .chain(fdecl.schema.input.iter().enumerate().flat_map(|(i, sort)| {
        vec![
            format!("(let {} (GetChild__ t1 {}))", child1(i), i),
            format!("(let {} (GetChild__ t2 {}))", child2(i), i),
        ]
    }))
    .chain(vec![
        format!("(let congr_prf__ (Congruence__ p1 {}))", congr_prf),
        "(set (EqGraph__ t1 t2) congr_prf__)".to_string(),
        "(set (EqGraph__ t2 t1) (Flip__ congr_prf__))".to_string(),
    ])
    .map(|s| egraph.action_parser.parse(&s).unwrap())
    .collect()
}

fn instrument_rule(_egraph: &EGraph, rule: &Rule) -> Rule {
    let res = rule.clone();
    // res.head.extend();
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

// the egraph is the initial egraph with only default sorts
pub(crate) fn add_proofs(egraph: &EGraph, program: Vec<Command>) -> Vec<Command> {
    let mut res = proof_header(egraph);
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
            }
            Command::Rule(rule) => {
                res.push(Command::Rule(instrument_rule(egraph, &rule)));
            }
            _ => res.push(command),
        }
    }

    res
}
