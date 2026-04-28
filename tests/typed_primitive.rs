//! Tests for the typed [`egglog::Primitive`] API and the seminaive-safety
//! enforcement added in issue #772.

use egglog::ast::Span;
use egglog::constraint::{SimpleTypeConstraint, TypeConstraint};
use egglog::sort::I64Sort;
use egglog::{ExecStateCore, RuleActionState};
use egglog::{EGraph, PrimitiveCommon, RuleActionPrim, Value, prelude::*};

/// A primitive implementing `RuleActionPrim` is rejected when used in
/// a rule-query (seminaive LHS) context. The check fires at typecheck
/// time via `PrimitiveWithId::valid_contexts`.
#[test]
fn typed_primitive_rejected_in_rule_query() {
    // A primitive that writes to the UF table — `RuleActionPrim` is
    // valid only in action contexts, never in a rule query.
    #[derive(Clone)]
    struct FakeWriter;
    impl PrimitiveCommon for FakeWriter {
        fn name(&self) -> &str {
            "fake-writer"
        }
        fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
            SimpleTypeConstraint::new(
                self.name(),
                vec![I64Sort.to_arcsort(), I64Sort.to_arcsort()],
                span.clone(),
            )
            .into_box()
        }
    }
    impl RuleActionPrim for FakeWriter {
        fn apply<'a, 'db>(
            &self,
            state: &mut RuleActionState<'a, 'db>,
            args: &[Value],
        ) -> Option<Value> {
            // Just exercise ExecStateCore — proving the body type-checks.
            let _ = state.base_values();
            Some(args[0])
        }
    }

    let mut egraph = EGraph::default();
    egraph.add_rule_action_primitive(FakeWriter);

    // Using a writing primitive in an RHS is fine.
    egraph
        .parse_and_run_program(
            None,
            "(function f (i64) i64 :no-merge)\n(rule () ((set (f 0) (fake-writer 42))))",
        )
        .unwrap();

    // Using it as a filter inside a rule LHS (RuleQuery) must be rejected
    // by the typechecker, which filters primitives to those whose
    // `valid_contexts` include the call site's context before building
    // the XOR.
    let result = egraph.parse_and_run_program(
        None,
        "(function g (i64) i64 :no-merge)\n\
         (rule ((= x (fake-writer 1))) ((set (g 0) x)))",
    );
    assert!(
        result.is_err(),
        "expected a typechecker error when using writing primitive in a rule query"
    );
}

/// `unstable-app` is registered as two context-specialized variants
/// (ApplyPure + ApplyFull). In a query context the pure variant
/// dispatches through `FunctionContainer::apply_in`, which succeeds only
/// when the inner function is a primitive valid in that context — e.g.
/// `+` over i64.
#[test]
fn unstable_app_query_with_pure_primitive() {
    let mut egraph = EGraph::default();
    // `(unstable-fn "+")` wraps the i64 primitive `+`; `+` is pure, valid
    // in every context, so `unstable-app` inside a `check` works.
    egraph
        .parse_and_run_program(
            None,
            r#"
            (sort IntFn (UnstableFn (i64 i64) i64))
            (let plus (unstable-fn "+"))
            (check (= (unstable-app plus 2 3) 5))
            "#,
        )
        .unwrap();
}

/// `unstable-app` over a constructor in a query context does NOT
/// silently mint a fresh eclass — `apply_in` returns None for that case,
/// so the match filters out and the check fails.
#[test]
fn unstable_app_query_with_constructor_is_filtered() {
    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(
            None,
            r#"
            (datatype Math (Num i64))
            (sort MathFn (UnstableFn (i64) Math))
            (let make (unstable-fn "Num"))
            ; (Num 99) has not been created, so `unstable-app make 99`
            ; inside a check must not mint one — the check must fail.
            "#,
        )
        .unwrap();
    let result = egraph.parse_and_run_program(
        None,
        r#"(check (= (unstable-app make 99) (Num 99)))"#,
    );
    assert!(
        result.is_err(),
        "check should fail: `unstable-app` on a constructor in a query \
         must return None rather than mint an eclass"
    );
}
