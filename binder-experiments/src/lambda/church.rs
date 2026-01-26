use super::{LambdaEnv, run_to_fixpoint};
use crate::lambda::sexp::{ParseError, add_expr_from_sexp};
use egglog_bridge::EGraph;
use egglog_core_relations::Value;
fn church_zero_sexp() -> String {
    "(lam f (lam x x))".to_string()
}

fn church_suc_sexp() -> String {
    "(lam n (lam f (lam x (x n))))".to_string()
}

fn church_y_sexp() -> String {
    "(lam f ((lam x (f (x x))) (lam x (f (x x)))))".to_string()
}

fn church_add_impl_sexp() -> String {
    let suc = church_suc_sexp();
    format!("(lam add (lam x (lam y (x y (lam z (add z ({suc} y)))))))")
}

fn church_add_sexp() -> String {
    let y = church_y_sexp();
    let add_impl = church_add_impl_sexp();
    format!("({y} {add_impl})")
}

pub(crate) fn church_num_sexp(n: usize) -> String {
    let suc = church_suc_sexp();
    let mut out = church_zero_sexp();
    for _ in 0..n {
        out = format!("({suc} {out})");
    }
    out
}

pub(crate) fn church_add_application_sexp(n: usize) -> String {
    let add = church_add_sexp();
    let num = church_num_sexp(n);
    log::info!("({add} {num} {num})");
    format!("({add} {num} {num})")
}

pub(crate) fn add_church_add_application(
    egraph: &mut EGraph,
    env: &LambdaEnv,
    n: usize,
) -> Result<Value, ParseError> {
    let expr = church_add_application_sexp(n);
    add_expr_from_sexp(egraph, env, &expr)
}

#[allow(dead_code)]
pub(crate) fn run_church_demo(n: usize, run_partition_refinement: bool) {
    let mut egraph = if run_partition_refinement {
        EGraph::with_partition_refinement()
    } else {
        EGraph::default()
    };
    let env = super::setup_lambda(&mut egraph);
    let _expr = add_church_add_application(&mut egraph, &env, n).expect("church demo parse failed");
    let mut rules = Vec::new();
    rules.extend(env.rules.free_vars.iter().copied());
    rules.extend(env.rules.subst.iter().copied());
    rules.extend(env.rules.beta.iter().copied());
    let _ = run_to_fixpoint(&mut egraph, &rules, run_partition_refinement);
    let _ = &env.rules.arith;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn church_num_zero() {
        assert_eq!(church_num_sexp(0), "(lam f (lam x x))");
    }

    #[test]
    fn church_num_one() {
        let expected = "((lam n (lam f (lam x (x n)))) (lam f (lam x x)))";
        assert_eq!(church_num_sexp(1), expected);
    }

    #[test]
    fn church_num_two() {
        let expected = concat!(
            "((lam n (lam f (lam x (x n)))) ",
            "((lam n (lam f (lam x (x n)))) (lam f (lam x x))))"
        );
        assert_eq!(church_num_sexp(2), expected);
    }

    #[test]
    fn church_add_application_uses_num_twice() {
        let expr = church_add_application_sexp(1);
        let num = church_num_sexp(1);
        assert_eq!(expr.matches(&num).count(), 2);
    }
}
