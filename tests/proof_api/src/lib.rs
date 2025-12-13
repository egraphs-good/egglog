use egglog::ast::Schema;
use egglog::prelude::*;

pub fn build_program() -> Result<EGraph, egglog::Error> {
    let mut egraph = EGraph::default();

    add_sort(&mut egraph, "Expr")?;

    add_constructor(
        &mut egraph,
        "Num",
        Schema {
            input: vec!["i64".into()],
            output: "Expr".into(),
        },
        None,
        false,
    )?;

    add_constructor(
        &mut egraph,
        "Add",
        Schema {
            input: vec!["Expr".into(), "Expr".into()],
            output: "Expr".into(),
        },
        None,
        false,
    )?;

    let ruleset = "rules";
    add_ruleset(&mut egraph, ruleset)?;
    rule(
        &mut egraph,
        ruleset,
        facts![
            (= (Add (Num x) (Num y)) (Num n))
            (= (+ x y) n)
        ],
        actions![],
    )?;

    Ok(egraph)
}
