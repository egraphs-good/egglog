use std::io::Read;

use egglog_experimental::ast::Command;
use egglog_experimental::scheduler::Matches;
use egglog_experimental::util::IndexMap;
use egglog_experimental::{self as egglog, add_scheduler_builder, CustomCostModel};
use egglog_experimental::{
    ast::Literal,
    extract::{CostModel},
    new_experimental_egraph,
    prelude::{exprs::*, *},
    scheduler::Scheduler,
    Term, TermDag,
};

use crate::ast::{Type};

mod ast;

fn program() -> &'static str {
    include_str!("defn.egg")
}

fn to_egglog_expr(expr: &ast::CoreExpr) -> egglog::ast::Expr {
    match expr {
        ast::CoreExpr::SVar(name) | ast::CoreExpr::MVar(name) => var(name),
        // ast::CoreExpr::SVar(name) => call(
        //     "SVar",
        //     // TODO: string in Rust API
        //     vec![egglog::ast::Expr::Lit(
        //         span!(),
        //         egglog::ast::Literal::String(name.clone()),
        //     )],
        // ),
        // ast::CoreExpr::MVar(name) => call(
        //     "MVar",
        //     vec![egglog::ast::Expr::Lit(
        //         span!(),
        //         egglog::ast::Literal::String(name.clone()),
        //     )],
        // ),
        ast::CoreExpr::Num(value) => call("Num", vec![exprs::int(*value)]),
        ast::CoreExpr::SAdd(left, right)
        | ast::CoreExpr::SMul(left, right)
        | ast::CoreExpr::MAdd(left, right)
        | ast::CoreExpr::MMul(left, right)
        | ast::CoreExpr::Scale(left, right)
        | ast::CoreExpr::SSub(left, right)
        | ast::CoreExpr::SDiv(left, right) => {
            let left_expr = to_egglog_expr(left);
            let right_expr = to_egglog_expr(right);
            let constructor = constructor_to_string(expr);
            call(constructor, vec![left_expr, right_expr])
        }
    }
}

fn constructor_to_string(constructor: &ast::CoreExpr) -> &str {
    match constructor {
        ast::CoreExpr::SAdd(..) => "SAdd",
        ast::CoreExpr::SMul(..) => "SMul",
        ast::CoreExpr::MAdd(..) => "MAdd",
        ast::CoreExpr::MMul(..) => "MMul",
        ast::CoreExpr::Scale(..) => "Scale",
        ast::CoreExpr::SSub(..) => "SSub",
        ast::CoreExpr::SDiv(..) => "SDiv",
        _ => unreachable!("binary constructor expected"),
    }
}

fn string_to_binary_constructor(
    constructor: &str,
    l: ast::CoreExpr,
    r: ast::CoreExpr,
) -> ast::CoreExpr {
    match constructor {
        "SAdd" => ast::CoreExpr::SAdd(Box::new(l), Box::new(r)),
        "SMul" => ast::CoreExpr::SMul(Box::new(l), Box::new(r)),
        "MAdd" => ast::CoreExpr::MAdd(Box::new(l), Box::new(r)),
        "MMul" => ast::CoreExpr::MMul(Box::new(l), Box::new(r)),
        "Scale" => ast::CoreExpr::Scale(Box::new(l), Box::new(r)),
        "SSub" => ast::CoreExpr::SSub(Box::new(l), Box::new(r)),
        "SDiv" => ast::CoreExpr::SDiv(Box::new(l), Box::new(r)),
        _ => panic!("Invalid constructor: {}", constructor),
    }
}

fn main() {
    let mut egraph = new_experimental_egraph();
    egraph
        .parse_and_run_program(Some("defn.egg".to_string()), program())
        .unwrap();

    let mut program = String::new();
    std::io::stdin().read_to_string(&mut program).unwrap();

    let bindings = ast::grammar::BindingsParser::new()
        .parse(&program)
        .expect("parsing bindings");
    let core_bindings = match bindings.lower() {
        Ok(bindings) => bindings,
        Err(e) => {
            println!("Error: {:?}", e);
            return;
        }
    };

    for decl in core_bindings.declares.iter() {
        let m = &decl.var;
        if let Type::Matrix { nrows, ncols } = decl.ty {
            let program = format!(
                "(MatrixDim \"{m}\" {nrows} {ncols})
                (let {m} (MVar \"{m}\"))"
            );
            egraph.parse_and_run_program(None, &program).unwrap();
        } else {
            let program = format!("(let {m} (SVar \"{m}\"))");
            egraph.parse_and_run_program(None, &program).unwrap();
        }
    }

    for bind in core_bindings.bindings.iter() {
        let var = &bind.var;
        let expr = &bind.expr;
        let expr = to_egglog_expr(expr);
        let action = Action::Let(span!(), var.to_string(), expr);

        egraph.run_program(vec![Command::Action(action)]).unwrap();
    }

    add_scheduler_builder("first-n".into(), Box::new(new_first_n_scheduler));
    let schedule = "
    (run-schedule 
      (let-scheduler first-100 (first-n 100))
      (repeat 100 (run-with first-100 optimization))
      (saturate (run cost-analysis))
    )
    ";
    // egraph.parse_and_run_program(None, "(run 20)").unwrap();
    egraph.parse_and_run_program(None, schedule.into()).unwrap();

    let output = core_bindings.bindings.last().unwrap();
    let (sort, value) = egraph.eval_expr(&var(&output.var)).unwrap();
    let (termdag, term, cost) = egraph
        .extract_value_with_cost_model(&sort, value, CustomCostModel)
        .unwrap();
    eprintln!("Cost after optimization: {cost}");
    let bindings = termdag_to_bindings(core_bindings.declares, &termdag, &term);
    println!("{}", bindings.to_string());
}

pub fn termdag_to_bindings(
    declares: Vec<ast::Declare>,
    termdag: &TermDag,
    term: &Term,
) -> ast::CoreBindings {
    fn process_term(
        termdag: &TermDag,
        term: &Term,
        bindings: &mut IndexMap<String, ast::CoreExpr>,
        name: String,
    ) -> String {
        if bindings.contains_key(&name) {
            return name;
        }
        match term {
            Term::App(op, args) => match op.as_str() {
                "SAdd" | "SMul" | "SSub" | "SDiv" | "Scale" | "MAdd" | "MMul" => {
                    let left = termdag.get(args[0]);
                    let lvar = process_term(termdag, left, bindings, format!("v{}", args[0]));
                    let right = termdag.get(args[1]);
                    let rvar = process_term(termdag, right, bindings, format!("v{}", args[1]));
                    let (lchild, rchild) = if op.as_str() == "Scale" {
                        (ast::CoreExpr::SVar(lvar), ast::CoreExpr::MVar(rvar))
                    } else if op.as_str().starts_with("S") {
                        (ast::CoreExpr::SVar(lvar), ast::CoreExpr::SVar(rvar))
                    } else {
                        (ast::CoreExpr::MVar(lvar), ast::CoreExpr::MVar(rvar))
                    };
                    let expr = string_to_binary_constructor(&op, lchild, rchild);

                    bindings.insert(name.to_string(), expr);
                    return name;
                }
                "MVar" | "SVar" => {
                    let arg = termdag.get(args[0]);
                    let Term::Lit(Literal::String(name)) = arg else {
                        unreachable!()
                    };
                    return name.to_string();
                }
                "Num" => {
                    let arg = termdag.get(args[0]);
                    let Term::Lit(Literal::Int(value)) = arg else {
                        unreachable!()
                    };
                    let expr = ast::CoreExpr::Num(*value);
                    bindings.insert(name.to_string(), expr);
                    return name;
                }
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };
    }

    let mut bindings: IndexMap<String, ast::CoreExpr> = Default::default();
    process_term(termdag, term, &mut bindings, "e".into());
    let bindings = bindings
        .into_iter()
        .map(|(name, expr)| ast::CoreBinding { var: name, expr })
        .collect();
    ast::CoreBindings { bindings, declares }
}

pub struct AstDepthCostModel;

pub type C = usize;
impl CostModel<C> for AstDepthCostModel {
    fn fold(&self, _head: &str, children_cost: &[C], head_cost: C) -> C {
        children_cost.iter().max().unwrap_or(&0) + head_cost
    }

    fn enode_cost(
        &self,
        _egraph: &EGraph,
        _func: &egglog::Function,
        _row: &egglog::FunctionRow,
    ) -> C {
        1
    }

    fn container_primitive(
        &self,
        _egraph: &EGraph,
        _sort: &egglog::ArcSort,
        _value: egglog::Value,
        element_costs: &[C],
    ) -> C {
        *element_costs.iter().max().unwrap_or(&0)
    }

    fn leaf_primitive(
        &self,
        _egraph: &EGraph,
        _sort: &egglog::ArcSort,
        _value: egglog::Value,
    ) -> C {
        1
    }
}

#[derive(Clone)]
struct FirstNScheduler {
    n: usize,
}

impl Scheduler for FirstNScheduler {
    fn filter_matches(&mut self, _rule: &str, _ruleset: &str, matches: &mut Matches) -> bool {
        if matches.match_size() <= self.n {
            matches.choose_all();
        } else {
            for i in 0..self.n {
                matches.choose(i);
            }
        }
        matches.match_size() < self.n * 2
    }
}

pub fn new_first_n_scheduler(_egraph: &EGraph, exprs: &[egglog::ast::Expr]) -> Box<dyn Scheduler> {
    assert!(exprs.len() == 1);
    let egglog::ast::Expr::Lit(_, Literal::Int(n))  = exprs[0] else {
        panic!("wrong arguments to first n scheduler");
    };
    Box::new(FirstNScheduler {
        n: n as usize
    })
}