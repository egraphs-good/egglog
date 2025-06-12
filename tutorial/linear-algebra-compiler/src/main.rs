use egglog::{ast::Command, extract::{CostModel, TreeAdditiveCostModel}, prelude::{exprs::call, *}};
mod ast;

fn program() -> &'static str {
    include_str!("defn.egg")
}

fn to_egglog_expr(expr: &ast::CoreExpr) -> egglog::ast::Expr {
    match expr {
        ast::CoreExpr::SVar(name) => expr!(
            (SVar (unquote egglog::ast::Expr::Lit(span!(), egglog::ast::Literal::String(
                name.clone(),
            ))))
        ),
        ast::CoreExpr::MVar { name, nrows, ncols } => {
            call("MVar", vec![
                egglog::ast::Expr::Lit(span!(), egglog::ast::Literal::String(name.clone())),
                egglog::ast::Expr::Lit(span!(), egglog::ast::Literal::Int(*nrows as i64)),
                egglog::ast::Expr::Lit(span!(), egglog::ast::Literal::Int(*ncols as i64)),
            ])
        },
        ast::CoreExpr::Num(value) => expr!( (Num (unquote exprs::int(*value) )) ),
        ast::CoreExpr::SAdd(left, right) => {
            let left_expr = to_egglog_expr(left);
            let right_expr = to_egglog_expr(right);
            expr!( (SAdd (unquote left_expr) (unquote right_expr)) )
        }
        ast::CoreExpr::SMul(left, right) => {
            let left_expr = to_egglog_expr(left);
            let right_expr = to_egglog_expr(right);
            expr!( (SMul (unquote left_expr) (unquote right_expr)) )
        }
        ast::CoreExpr::MAdd(left, right) => {
            let left_expr = to_egglog_expr(left);
            let right_expr = to_egglog_expr(right);
            expr!( (MAdd (unquote left_expr) (unquote right_expr)) )
        }
        ast::CoreExpr::MMul(left, right) => {
            let left_expr = to_egglog_expr(left);
            let right_expr = to_egglog_expr(right);
            expr!( (MMul (unquote left_expr) (unquote right_expr)) )
        }
        ast::CoreExpr::Scale(base, factor) => {
            let base_expr = to_egglog_expr(base);
            let factor_expr = to_egglog_expr(factor);
            expr!( (Scale (unquote base_expr) (unquote factor_expr)) )
        }
        ast::CoreExpr::SSub(left, right) => {
            let left_expr = to_egglog_expr(left);
            let right_expr = to_egglog_expr(right);
            expr!( (SSub (unquote left_expr) (unquote right_expr)) )
        }
        ast::CoreExpr::SDiv(left, right) => {
            let left_expr = to_egglog_expr(left);
            let right_expr = to_egglog_expr(right);
            expr!( (SDiv (unquote left_expr) (unquote right_expr)) )
        }
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

    let mut egraph = EGraph::default();

    egraph
        .parse_and_run_program(Some("defn.egg".to_string()), program())
        .unwrap();
    let expr = match &core_bindings[3] {
        ast::CoreBinding::Bind { var, expr } => expr,
        ast::CoreBinding::Declare { var, ty } => unreachable!(),
    };
    dbg!(to_egglog_expr(expr).to_string());
    let (sort,value) = egraph.eval_expr(&to_egglog_expr(expr))
        .unwrap();
    let (termdag, term, cost) = egraph.extract_value(&sort, value).unwrap();
    let (termdag, term, cost) = egraph.extract_value_with_cost_model(&sort, value, TreeAdditiveCostModel {}).unwrap();
    eprintln!("{}", termdag.to_string(&term));
}


struct AstDepthCostModel;

type C = usize;
impl CostModel<C> for AstDepthCostModel {
    fn fold(&self, _head: &str, children_cost: &[C], head_cost: C) -> C {
        children_cost.iter().max().unwrap_or(&0) + head_cost
    }

    fn enode_cost(&self, _egraph: &EGraph, _func: &egglog::Function, _row: &egglog::FunctionRow) -> C {
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
    
    fn leaf_primitive(&self, _egraph: &EGraph, _sort: &egglog::ArcSort, _value: egglog::Value) -> C {
        1
    }
}