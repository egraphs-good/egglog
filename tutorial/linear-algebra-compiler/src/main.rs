use std::collections::HashMap;

use egglog_experimental as egglog;
use egglog_experimental::{
    ast::Literal,
    extract::{CostModel, TreeAdditiveCostModel},
    new_experimental_egraph,
    prelude::{exprs::call, *},
    Term, TermDag,
};

mod ast;

fn program() -> &'static str {
    include_str!("defn.egg")
}

fn to_egglog_expr(expr: &ast::CoreExpr) -> egglog::ast::Expr {
    match expr {
        ast::CoreExpr::SVar(name) => call(
            "SVar",
            // TODO: string in Rust API
            vec![egglog::ast::Expr::Lit(
                span!(),
                egglog::ast::Literal::String(name.clone()),
            )],
        ),
        ast::CoreExpr::MVar { name } => call(
            "MVar",
            vec![egglog::ast::Expr::Lit(
                span!(),
                egglog::ast::Literal::String(name.clone()),
            )],
        ),
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

// TODO: transpose
fn constructor_to_string(constructor: &ast::CoreExpr) -> &str {
    match constructor {
        ast::CoreExpr::SVar(_) => "SVar",
        ast::CoreExpr::MVar { .. } => "MVar",
        ast::CoreExpr::Num(_) => "Num",
        ast::CoreExpr::SAdd(..) => "SAdd",
        ast::CoreExpr::SMul(..) => "SMul",
        ast::CoreExpr::MAdd(..) => "MAdd",
        ast::CoreExpr::MMul(..) => "MMul",
        ast::CoreExpr::Scale(..) => "Scale",
        ast::CoreExpr::SSub(..) => "SSub",
        ast::CoreExpr::SDiv(..) => "SDiv",
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
    let bindings = ast::grammar::BindingsParser::new()
        .parse("x: R; y: R; A: [R; 2x2]; B = (x + y) * A;")
        .expect("parsing bindings");
    let core_bindings = match bindings.lower() {
        Ok(bindings) => bindings,
        Err(e) => {
            println!("Error: {:?}", e);
            return;
        }
    };

    let mut egraph = new_experimental_egraph();
    egraph
        .parse_and_run_program(Some("defn.egg".to_string()), program())
        .unwrap();
    let expr = &core_bindings.bindings[0].expr;
    dbg!(to_egglog_expr(expr).to_string());
    let (sort, value) = egraph.eval_expr(&to_egglog_expr(expr)).unwrap();
    let (termdag, term, cost) = egraph.extract_value(&sort, value).unwrap();
    let (termdag, term, cost) = egraph
        .extract_value_with_cost_model(&sort, value, TreeAdditiveCostModel {})
        .unwrap();
    let bindings = termdag_to_bindings(core_bindings.declares, &termdag, &term);
    dbg!(bindings);
    eprintln!("{}", termdag.to_string(&term));
}

pub fn termdag_to_bindings(
    declares: Vec<ast::Declare>,
    termdag: &TermDag,
    term: &Term,
) -> ast::CoreBindings {
    fn process_term(
        termdag: &TermDag,
        term: &Term,
        bindings: &mut HashMap<String, ast::CoreExpr>,
        name: &str,
    ) {
        if bindings.contains_key(name) {
            return;
        }
        let expr = match term {
            Term::App(op, args) => match op.as_str() {
                "SAdd" | "SMul" | "SSub" | "SDiv" | "Scale" | "MAdd" | "MMul" => {
                    let lvar = format!("v{}", args[0]);
                    let left = termdag.get(args[0]);
                    process_term(termdag, left, bindings, &lvar);
                    let right = termdag.get(args[1]);
                    let rvar = format!("v{}", args[1]);
                    process_term(termdag, right, bindings, &rvar);
                    let (lchild, rchild) = if op.as_str() == "Scale" {
                        (ast::CoreExpr::SVar(lvar), ast::CoreExpr::SVar(rvar))
                    } else if op.as_str().starts_with("S") {
                        (ast::CoreExpr::SVar(lvar), ast::CoreExpr::SVar(rvar))
                    } else {
                        (
                            ast::CoreExpr::MVar { name: lvar },
                            ast::CoreExpr::MVar { name: rvar },
                        )
                    };
                    string_to_binary_constructor(&op, lchild, rchild)
                }
                "MVar" | "SVar" => {
                    let arg = termdag.get(args[0]);
                    let Term::Lit(Literal::String(name)) = arg else {
                        unreachable!()
                    };
                    if op.as_str() == "MVar" {
                        ast::CoreExpr::MVar { name: name.clone() }
                    } else {
                        ast::CoreExpr::SVar(name.clone())
                    }
                }
                "Num" => {
                    let arg = termdag.get(args[0]);
                    let Term::Lit(Literal::Int(value)) = arg else {
                        unreachable!()
                    };
                    ast::CoreExpr::Num(*value)
                }
                _ => todo!(),
            },
            _ => todo!(),
        };
        bindings.insert(name.to_string(), expr);
    }

    let mut bindings: HashMap<String, ast::CoreExpr> = HashMap::new();
    process_term(termdag, term, &mut bindings, "e");
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
