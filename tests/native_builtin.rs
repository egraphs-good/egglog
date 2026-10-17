use std::sync::Arc;

use egglog::builtin::{BuiltinInstance, BuiltinTyping, NativeFunctionTarget, SortPattern};
use egglog::constraint::TypeConstraint;
use egglog::prelude::*;
use egglog::{ArcSort, EGraph, Primitive, PurePrim, PureState, Value};

#[test]
fn native_scalar_description_drives_complete_signature_checks() {
    let graph = EGraph::default();
    let types = graph.authoring_type_info();
    let integer = types.get_sort_by_name("i64").unwrap().clone();
    let string = types.get_sort_by_name("String").unwrap().clone();
    let description = types.builtin_description("egglog.core.i64.add").unwrap();
    let BuiltinTyping::Signature(signature) = &description.typing else {
        panic!("expected scalar signature")
    };
    assert_eq!(signature.fixed[0].name, "a");
    assert_eq!(signature.fixed[1].name, "b");
    assert_eq!(
        signature.output,
        SortPattern::Family {
            name: "i64".into(),
            args: vec![]
        }
    );
    assert!(
        types
            .check_native_signature(
                description,
                &[integer.clone(), integer.clone(), integer.clone()]
            )
            .is_ok()
    );
    assert!(
        types
            .check_native_signature(description, &[integer.clone(), integer, string])
            .is_err()
    );
}

#[derive(Clone)]
struct ExactNative {
    instance: BuiltinInstance,
}

impl Primitive for ExactNative {
    fn name(&self) -> &str {
        "test-exact-native"
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        self.instance.type_constraints(span)
    }

    fn builtin_instance(&self) -> Option<&BuiltinInstance> {
        Some(&self.instance)
    }
}

impl PurePrim for ExactNative {
    fn apply<'a, 'db>(&self, _state: PureState<'a, 'db>, _args: &[Value]) -> Option<Value> {
        panic!("native selection must not execute the primitive")
    }
}

#[test]
fn exact_occurrences_keep_nominal_aliases_and_clone_identity() {
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(None, "(sort A (Vec i64)) (sort B (Vec i64))")
        .unwrap();
    let types = graph.authoring_type_info();
    let a = types.get_sort_by_name("A").unwrap().clone();
    let b = types.get_sort_by_name("B").unwrap().clone();
    let integer = types.get_sort_by_name("i64").unwrap().clone();
    assert_eq!(
        types.native_sort_pattern(&a).unwrap(),
        types.native_sort_pattern(&b).unwrap()
    );
    let instance = types
        .instantiate_builtin_signature(
            "egglog.core.vec.set",
            &[a.clone(), integer.clone(), integer.clone()],
            &[],
            &b,
        )
        .unwrap();
    graph.add_pure_primitive(ExactNative { instance }, None);
    let signature: Vec<ArcSort> = vec![a.clone(), integer.clone(), integer, b];
    let selected = graph
        .authoring_type_info()
        .resolve_builtin("egglog.core.vec.set", &signature)
        .unwrap();
    assert_eq!(selected.primitive.name(), "test-exact-native");
    let mut other = signature.clone();
    *other.last_mut().unwrap() = a;
    assert!(!selected.accept(&other, graph.authoring_type_info()));
    let cloned = graph.clone();
    let same = cloned
        .authoring_type_info()
        .resolve_builtin("egglog.core.vec.set", &signature)
        .unwrap();
    assert!(Arc::ptr_eq(&selected.primitive, &same.primitive));
    assert_eq!(selected.context_ids, same.context_ids);
}

#[test]
fn generic_native_matcher_checks_equality_bounds() {
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(
            None,
            "(datatype Math (Num i64)) (sort Maths (Vec Math)) (sort Ints (Vec i64))",
        )
        .unwrap();
    let types = graph.authoring_type_info();
    let description = types.builtin_description("egglog.core.vec.union").unwrap();
    let maths = types.get_sort_by_name("Maths").unwrap().clone();
    let ints = types.get_sort_by_name("Ints").unwrap().clone();
    assert!(
        types
            .check_native_signature(description, &[maths.clone(), maths.clone(), maths])
            .is_ok()
    );
    assert!(
        types
            .check_native_signature(description, &[ints.clone(), ints.clone(), ints])
            .is_err()
    );
}

#[test]
fn function_target_selection_preserves_runtime_contexts() {
    let graph = EGraph::default();
    let types = graph.authoring_type_info();
    let integer = types.get_sort_by_name("i64").unwrap().clone();
    let native = types
        .resolve_builtin(
            "egglog.core.i64.add",
            &[integer.clone(), integer.clone(), integer.clone()],
        )
        .unwrap();
    let NativeFunctionTarget::Primitive(selected) = types
        .resolve_function_target("+", &[integer.clone(), integer.clone()], &integer, None)
        .unwrap()
    else {
        panic!("expected primitive target")
    };
    assert_eq!(selected.len(), 1);
    assert!(Arc::ptr_eq(&native.primitive, &selected[0].primitive));
    assert_eq!(native.context_ids, selected[0].context_ids);
    let mut context_ids = native.context_ids;
    context_ids[egglog::Context::Pure] = None;
    let stale = egglog::sort::ResolvedFunctionId::Primitive { context_ids };
    assert!(
        types
            .resolve_function_target(
                "+",
                &[integer.clone(), integer.clone()],
                &integer,
                Some(&stale)
            )
            .is_err()
    );
}

#[test]
fn inventory_keeps_opaque_native_registration_gaps() {
    let mut graph = EGraph::default();
    egglog::add_primitive!(
        &mut graph,
        "test-opaque-native" = |value: i64| -> i64 { value }
    );
    let catalog = graph.authoring_type_info().builtin_catalog().unwrap();
    assert!(
        catalog
            .undescribed_primitives
            .iter()
            .any(|name| name == "test-opaque-native")
    );
    assert!(
        catalog
            .descriptions
            .iter()
            .any(|description| description.key == "egglog.core.i64.add")
    );
}
