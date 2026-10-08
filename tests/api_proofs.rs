//! Verify the new Rust API gives clear errors when used with the
//! proof system enabled. `rust_rule` callbacks and direct e-graph
//! writes via `update` both bypass the proof-encoding
//! pipeline and must surface a helpful error rather than silently
//! producing unverifiable proofs.

use egglog::Error;
use egglog::ast::Span;
use egglog::constraint::{SimpleTypeConstraint, TypeConstraint};
use egglog::prelude::*;
use egglog::proof::ProveExistsError;
use egglog::sort::I64Sort;
use egglog::{CommandOutput, Primitive, PurePrim, PureState, Value};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Clone)]
struct ProofIdentity(Arc<AtomicUsize>);

impl Primitive for ProofIdentity {
    fn name(&self) -> &str {
        "proof-identity"
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

impl PurePrim for ProofIdentity {
    fn apply<'a, 'db>(&self, _state: PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Some(args[0])
    }
}

#[test]
fn canonical_pair_instances_keep_nested_projection_proofs() {
    for mode in ["terms", "proofs", "proof-testing"] {
        let graph = EGraph::default();
        let mut graph = match mode {
            "terms" => graph.with_term_encoding_enabled(),
            "proofs" => graph.with_proofs_enabled(),
            _ => graph.with_proofs_enabled().with_proof_testing(),
        };
        graph
            .parse_and_run_program(
                None,
                r#"
            (datatype N (Z) (S N))
            (sort P (Pair N N)) (sort Nested (Pair P P))
            (relation Row (Nested)) (relation Result (N N))
            (Row (pair (pair (Z) (S (Z))) (pair (S (Z)) (Z))))
            (rule ((Row p) (= a (pair-second (pair-first p)))
                          (= b (pair-first (pair-second p))))
                  ((Result a b)))
            (run 1)
            (check (Result (S (Z)) (S (Z))))
        "#,
            )
            .unwrap();
        if mode != "terms" {
            let outputs = graph
                .parse_and_run_program(None, "(prove (Result (S (Z)) (S (Z))))")
                .unwrap();
            assert!(
                outputs
                    .iter()
                    .any(|o| matches!(o, CommandOutput::ProveExists { .. }))
            );
        }
    }
}

#[test]
fn registered_providers_are_captured_before_enabling_encoding() {
    for mode in ["terms", "proofs", "proof-testing"] {
        let calls = Arc::new(AtomicUsize::new(0));
        let validations = Arc::new(AtomicUsize::new(0));
        let counted_validations = validations.clone();
        let mut graph = EGraph::default();
        graph.add_pure_primitive(
            ProofIdentity(calls.clone()),
            Some(Arc::new(move |_, args| {
                counted_validations.fetch_add(1, Ordering::SeqCst);
                args.first().copied()
            })),
        );
        let mut graph = match mode {
            "terms" => graph.with_term_encoding_enabled(),
            "proofs" => graph.with_proofs_enabled(),
            _ => graph.with_proofs_enabled().with_proof_testing(),
        };
        let mut checker = graph.clone();
        let command = checker
            .parse_program(None, "(proof-identity 2)")
            .unwrap()
            .remove(0);
        checker.resolve_command_before_proofs(command).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(validations.load(Ordering::SeqCst), 0);

        let outputs = graph
            .parse_and_run_program(
                None,
                r#"
                (datatype Num (N i64) (Goal i64))
                (N 2)
                (rule ((N a) (= b (proof-identity a))) ((Goal b)) :name "provider-rule")
                (run 1)
                (check (Goal 2))
                "#,
            )
            .unwrap();
        assert_eq!(
            outputs
                .iter()
                .any(|out| matches!(out, CommandOutput::ProveExists { .. })),
            mode == "proof-testing"
        );
        if mode != "terms" {
            let outputs = graph
                .parse_and_run_program(None, "(prove (Goal 2))")
                .unwrap();
            assert!(
                outputs
                    .iter()
                    .any(|out| matches!(out, CommandOutput::ProveExists { .. }))
            );
            assert!(validations.load(Ordering::SeqCst) > 0);
        }
        assert!(calls.load(Ordering::SeqCst) > 0);
    }
}

#[test]
fn providers_added_after_capture_are_not_in_the_authoring_checker() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut graph = EGraph::default().with_proofs_enabled();
    graph.add_pure_primitive(
        ProofIdentity(calls.clone()),
        Some(Arc::new(|_, args| args.first().copied())),
    );
    let error = graph
        .parse_and_run_program(None, "(proof-identity 2)")
        .unwrap_err();
    assert!(matches!(error, Error::TypeError(_)), "{error}");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn provider_without_a_validator_still_rejects_proof_encoding() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut graph = EGraph::default();
    graph.add_pure_primitive(ProofIdentity(calls.clone()), None);
    let mut graph = graph.with_proofs_enabled();
    let error = graph
        .parse_and_run_program(None, "(proof-identity 2)")
        .unwrap_err();
    assert!(
        matches!(error, Error::UnsupportedProofCommand { .. }),
        "{error}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn public_proof_errors_keep_native_witness_before_mode_order() {
    for has_witness in [false, true] {
        let mut graph = EGraph::default();
        graph
            .parse_and_run_program(None, "(datatype Num (N i64))")
            .unwrap();
        if has_witness {
            graph.parse_and_run_program(None, "(N 2)").unwrap();
        }
        let error = graph
            .parse_and_run_program(None, "(prove-exists N)")
            .unwrap_err();
        match error {
            Error::ProofError {
                error: ProveExistsError::ProofsNotEnabled,
                ..
            } => assert!(has_witness),
            Error::ProofError {
                error: ProveExistsError::QueryDidNotMatch { constructor },
                ..
            } => {
                assert!(!has_witness);
                assert_eq!(constructor, "N");
            }
            error => panic!("unexpected error: {error}"),
        }
    }
}

#[test]
fn rust_rule_with_proofs_enabled_errors() {
    let mut eg = EGraph::new_with_proofs();
    eg.parse_and_run_program(None, "(function f (i64) i64 :merge new)")
        .unwrap();
    add_ruleset(&mut eg, "r").unwrap();

    let result = rust_rule(
        &mut eg,
        "test_rule",
        "r",
        vars![x: i64],
        facts![(= y (f x))],
        |_, _| Some(()),
    );

    let err = result.expect_err("rust_rule should fail under proofs");
    assert!(
        matches!(err, Error::ProofsIncompatibleApi { api, .. } if api == "rust_rule"),
        "expected ProofsIncompatibleApi(rust_rule), got: {err}"
    );
}

#[test]
fn rust_rule_full_with_proofs_enabled_errors() {
    let mut eg = EGraph::new_with_proofs();
    eg.parse_and_run_program(None, "(function f (i64) i64 :merge new)")
        .unwrap();
    add_ruleset(&mut eg, "r").unwrap();

    let result = rust_rule_full(
        &mut eg,
        "test_rule",
        "r",
        vars![x: i64],
        facts![(= y (f x))],
        |_, _| Some(()),
    );

    let err = result.expect_err("rust_rule_full should fail under proofs");
    assert!(
        matches!(err, Error::ProofsIncompatibleApi { api, .. } if api == "rust_rule_full"),
        "expected ProofsIncompatibleApi(rust_rule_full), got: {err}"
    );
}

#[test]
fn update_with_proofs_enabled_errors() {
    let mut eg = EGraph::new_with_proofs();
    eg.parse_and_run_program(None, "(function f (i64) i64 :merge new)")
        .unwrap();

    let result = eg.update(|mut fs| fs.set("f", (1_i64,), 42_i64));

    let err = result.expect_err("update should fail under proofs");
    assert!(
        matches!(err, Error::ProofsIncompatibleApi { api, .. } if api == "EGraph::update"),
        "expected ProofsIncompatibleApi(EGraph::update), got: {err}"
    );
}

#[test]
fn query_with_proofs_enabled_errors_with_query_api_name() {
    // Regression: previously the failure surfaced through the
    // rust_rule check inside query, so the error pointed at
    // "rust_rule" instead of "EGraph::query".
    let mut eg = EGraph::new_with_proofs();
    eg.parse_and_run_program(None, "(function f (i64) i64 :merge new)")
        .unwrap();

    let result = eg.query(vars![x: i64], facts![(= y (f x))]);

    let err = result.expect_err("EGraph::query should fail under proofs");
    assert!(
        matches!(err, Error::ProofsIncompatibleApi { api, .. } if api == "EGraph::query"),
        "expected ProofsIncompatibleApi(EGraph::query), got: {err}"
    );
}
