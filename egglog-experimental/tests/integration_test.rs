use std::sync::Arc;

use egglog::{
    ast::{Expr, Literal},
    prelude::{RustSpan, Span},
};

#[test]
fn test_extract() {
    let mut egraph = egglog_experimental::new_experimental_egraph();

    let result = egraph
        .parse_and_run_program(
            None,
            "
        (with-dynamic-cost
            (datatype E (Add E E) (Sub E E :cost 200) (Num i64))
        )

        (union (Num 2) (Add (Num 1) (Num 1)))
        (set-cost (Num 2) 1000)
        (set-cost (Num 1) 100)
        (extract (Num 2))

        (push)
        (set-cost (Add (Num 1) (Num 1)) 800)
        (extract (Num 2))
        (pop)

        (push)
        (set-cost (Add (Num 1) (Num 1)) 798)
        (extract (Num 2))
        (pop)

        ;; 200 + 1 + 1 > 1 + 100 + 100
        (union (Num 2) (Sub (Num 5) (Num 3)))
        (extract (Num 2))
        (set-cost (Sub (Num 5) (Num 3)) 198)
        ;; 198 + 1 + 1 < 1 + 100 + 100
        (extract (Num 2))",
        )
        .unwrap();

    assert_eq!(result.len(), 5);
    assert_eq!(result[0].to_string(), "(Add (Num 1) (Num 1))\n");
    assert_eq!(result[1].to_string(), "(Num 2)\n");
    assert_eq!(result[2].to_string(), "(Add (Num 1) (Num 1))\n");
    assert_eq!(result[3].to_string(), "(Add (Num 1) (Num 1))\n");
    assert_eq!(result[4].to_string(), "(Sub (Num 5) (Num 3))\n");
}

#[test]
fn test_get_size_primitive() {
    let mut egraph = egglog_experimental::new_experimental_egraph();

    let span = Span::Rust(Arc::new(RustSpan {
        file: "integration_test",
        line: 0,
        column: 0,
    }));

    let make_expr = |names: &[&str]| {
        Expr::Call(
            span.clone(),
            "get-size!".into(),
            names
                .iter()
                .map(|name| Expr::Lit(span.clone(), Literal::String((*name).into())))
                .collect(),
        )
    };

    let eval_size = |egraph: &mut egglog::EGraph, names: &[&str]| -> i64 {
        let expr = make_expr(names);
        let (_, value) = egraph.eval_expr(&expr).unwrap();
        egraph.value_to_base::<i64>(value)
    };

    assert_eq!(eval_size(&mut egraph, &[]), 0);
    assert_eq!(eval_size(&mut egraph, &["MkFoo"]), 0);
    assert_eq!(eval_size(&mut egraph, &["MkBar"]), 0);
    assert_eq!(eval_size(&mut egraph, &["MkFoo", "MkBar"]), 0);

    egraph
        .parse_and_run_program(
            None,
            "
            (datatype Foo (MkFoo i64))
            (datatype Bar (MkBar i64))
            (MkFoo 1)
            (MkFoo 2)
            (MkBar 10)
        ",
        )
        .unwrap();

    assert_eq!(eval_size(&mut egraph, &[]), 3);
    assert_eq!(eval_size(&mut egraph, &["MkFoo"]), 2);
    assert_eq!(eval_size(&mut egraph, &["MkBar"]), 1);
    assert_eq!(eval_size(&mut egraph, &["MkFoo", "MkBar"]), 3);
    assert_eq!(eval_size(&mut egraph, &["Unknown"]), 0);
}

#[test]
fn test_extract_set_cost_multiple_times_should_fail() {
    let mut egraph = egglog_experimental::new_experimental_egraph();

    egraph
        .parse_and_run_program(
            None,
            "(with-dynamic-cost
                (datatype E (Add E E) (Sub E E :cost 200) (Num i64))
            )
            (set-cost (Num 2) 1000)",
        )
        .unwrap();

    egraph
        .parse_and_run_program(None, "(set-cost (Num 2) 1000)")
        .unwrap();

    let result = egraph.parse_and_run_program(None, "(set-cost (Num 2) 1)");
    assert!(result.is_err());
}

#[test]
fn test_extract_set_cost_decls() {
    let mut egraph = egglog_experimental::new_experimental_egraph();

    egraph
        .parse_and_run_program(
            None,
            "(with-dynamic-cost
                (datatype E (Add E E) (Sub E E :cost 200) (Num i64))
                (constructor Mul (E E) E :cost 100)
                (datatype*
                  (E2 (Add2 E2 E2) (Sub2 E2 E2 :cost 200) (List VecE2) (Num2 i64))
                  (sort VecE2 (Vec E2))
                )
                (constructor Mul2 (E2 E2) E2)
            )
            (set-cost (Num 2) 1000)
            (set-cost (Num2 2) 1000)
            (set-cost (Mul (Num 2) (Num 2)) 1000)
            (set-cost (Sub2 (Num2 2) (Num2 2)) 1000)",
        )
        .unwrap();
}

#[test]
fn test_multi_extract_two_variants_two_terms() {
    let mut egraph = egglog_experimental::new_experimental_egraph();

    let result = egraph
        .parse_and_run_program(
            None,
            "
        (with-dynamic-cost
            (datatype E (Add E E) (Mul E E) (Num i64))
        )

        (union (Num 2) (Add (Num 1) (Num 1)))
        (union (Num 2) (Mul (Num 1) (Num 2)))

        (union (Num 4) (Add (Num 2) (Num 2)))
        (union (Num 4) (Mul (Num 2) (Num 2)))

        (multi-extract 2 (Num 2) (Num 4))",
        )
        .unwrap();

    assert_eq!(result.len(), 1);
    let output = result[0].to_string();
    assert!(output.contains("(Num 2)"));
    assert!(output.contains("(Add (Num 1) (Num 1))") || output.contains("(Mul (Num 1) (Num 2))"));
    assert!(output.contains("(Num 4)"));
    assert!(output.contains("(Add (Num 2) (Num 2))") || output.contains("(Mul (Num 2) (Num 2))"));
}

#[test]
fn test_multi_extract_single_variant_minimal_cost() {
    let mut egraph = egglog_experimental::new_experimental_egraph();

    let result = egraph
        .parse_and_run_program(
            None,
            "
        (with-dynamic-cost
            (datatype E (Add E E :cost 3) (Mul E E :cost 10) (Num i64 :cost 1))
        )

        (union (Num 5) (Add (Num 2) (Num 3)))
        (union (Num 5) (Mul (Num 1) (Num 5)))
        (union (Add (Num 5) (Num 5)) (Mul (Num 2) (Num 5)))

        (multi-extract 1 (Mul (Num 2) (Num 5)))",
        )
        .unwrap();

    assert_eq!(result.len(), 1);
    let output = result[0].to_string();
    assert!(output.contains("(Add (Num 5) (Num 5))"));
    assert!(!output.contains("Mul"));
}

#[test]
fn test_multi_extract_with_set_cost() {
    let mut egraph = egglog_experimental::new_experimental_egraph();

    let result = egraph
        .parse_and_run_program(
            None,
            "
        (with-dynamic-cost
            (datatype E (Add E E) (Mul E E) (Num i64))
        )

        (union (Num 10) (Add (Num 5) (Num 5)))
        (union (Num 10) (Mul (Num 2) (Num 5)))

        (union (Num 6) (Add (Num 3) (Num 3)))
        (union (Num 6) (Mul (Num 2) (Num 3)))

        (set-cost (Add (Num 5) (Num 5)) 1)
        (set-cost (Add (Num 3) (Num 3)) 1)

        (set-cost (Mul (Num 2) (Num 5)) 1000)
        (set-cost (Mul (Num 2) (Num 3)) 1000)

        (multi-extract 2 (Num 10) (Num 6))",
        )
        .unwrap();

    assert_eq!(result.len(), 1);
    let output = result[0].to_string();
    assert!(output.contains("(Add (Num 5) (Num 5))"));
    assert!(output.contains("(Add (Num 3) (Num 3))"));
    assert!(!output.contains("Mul"));
}
