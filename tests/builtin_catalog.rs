use egglog::extract::DEFAULT_COST_MODEL;
use egglog::sort::{PairContainer, Presort, VecContainer, VecSort};
use egglog::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[test]
fn builtin_default_context_baseline_gap_isolated_and_legacy_multiroot() {
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    CALLS.store(0, Ordering::SeqCst);
    let mut graph = EGraph::default();
    add_literal_prim!(
        &mut graph,
        "default-helper"[id = "test.default_helper"] = || -> i64 {
            {
                CALLS.fetch_add(1, Ordering::SeqCst);
                17
            }
        }
    );
    let integer = graph.get_sort_by_name("i64").unwrap().clone();
    let mut isolated = builtin::closed_signature(
        "test.default_consumer",
        "default-consumer",
        &[("value", integer.clone())],
        &[],
        integer,
    )
    .unwrap();
    let root = isolated.declarations.last_mut().unwrap();
    let Some(proto::declaration::Kind::HostPrimitive(primitive)) = &root.kind else {
        unreachable!()
    };
    let Some(proto::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
        unreachable!()
    };
    isolated.nodes.push(proto::Node {
        sort_id: signature.output.unwrap(),
        kind: Some(proto::node::Kind::Call(proto::Call {
            func: "test.default_helper".into(),
            args: vec![],
        })),
        ..Default::default()
    });
    root.bindings.as_mut().unwrap().python = Some(proto::PythonBindings {
        views: vec![proto::PythonCallable {
            kind: proto::PythonCallKind::Function.into(),
            path: vec!["test".into(), "default_consumer".into()],
            params: vec![proto::PythonParameter {
                core_input: Some(0),
                name: "value".into(),
                default_expr: Some(0),
            }],
            ..Default::default()
        }],
    });
    let mut destination = proto::Program::default();
    let before = destination.clone();
    let error =
        builtin::definitions::reconcile_declarations(&isolated, &mut destination).unwrap_err();
    assert!(error.contains("unknown default callee"), "{error}");
    assert_eq!(destination, before);

    let mut complete = proto::Program::default();
    graph
        .type_info()
        .export_builtin_definition("test.default_helper", &mut complete)
        .unwrap();
    builtin::definitions::reconcile_declarations(&isolated, &mut complete).unwrap();
    assert_eq!(
        complete
            .declarations
            .iter()
            .filter(|declaration| {
                matches!(
                    declaration.kind,
                    Some(proto::declaration::Kind::HostPrimitive(_))
                )
            })
            .count(),
        2
    );
    let error = builtin::import_definition(&complete, &mut destination).unwrap_err();
    assert!(error.contains("exactly one builtin"), "{error}");
    assert_eq!(destination, before);
    assert_eq!(CALLS.load(Ordering::SeqCst), 0);
}

mod rooted_default_context {
    use super::*;

    static CALLS: AtomicUsize = AtomicUsize::new(0);

    // Compose a genuine helper export with a generated root, including all
    // index relocation. Reordering happens only after the closed import.
    fn supply(
        types: &TypeInfo,
        key: &str,
        draft: &mut proto::Program,
        helper: &str,
        helper_first: bool,
        relocated: bool,
        argument: bool,
    ) -> Result<(), String> {
        let mut consumer = draft.clone();
        let root = consumer.declarations.iter().position(|declaration| {
            matches!(&declaration.kind, Some(proto::declaration::Kind::HostPrimitive(primitive)) if primitive.name == key)
        }).unwrap();
        let Some(proto::declaration::Kind::HostPrimitive(primitive)) =
            &consumer.declarations[root].kind
        else {
            unreachable!()
        };
        let Some(proto::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
            unreachable!()
        };
        let output = signature.output.unwrap();
        let args = if argument {
            let index = consumer.nodes.len() as u32;
            consumer.nodes.push(proto::Node {
                sort_id: output,
                kind: Some(proto::node::Kind::PrimitiveValue(proto::PrimitiveValue {
                    value: Some(proto::primitive_value::Value::I64(3)),
                })),
                ..Default::default()
            });
            vec![index]
        } else {
            vec![]
        };
        let default = consumer.nodes.len() as u32;
        consumer.files.push(proto::SourceFile {
            name: "default.egg".into(),
            contents: Some("helper".into()),
        });
        consumer.nodes.push(proto::Node {
            sort_id: output,
            kind: Some(proto::node::Kind::Call(proto::Call {
                func: helper.into(),
                args,
            })),
            span: Some(proto::Span {
                file: 0,
                start: 0,
                end: 6,
            }),
        });
        consumer.declarations[root]
            .bindings
            .as_mut()
            .unwrap()
            .python = Some(proto::PythonBindings {
            views: vec![proto::PythonCallable {
                kind: proto::PythonCallKind::Function.into(),
                path: vec!["test".into(), key.replace('.', "_")],
                params: vec![proto::PythonParameter {
                    core_input: Some(0),
                    name: "value".into(),
                    default_expr: Some(default),
                }],
                ..Default::default()
            }],
        });
        let mut complete = proto::Program {
            ir_version: 1,
            ..Default::default()
        };
        if relocated {
            complete.sorts.push(proto::Sort {
                kind: Some(proto::sort::Kind::Family(proto::HostSort {
                    name: "String".into(),
                    args: vec![],
                })),
                ..Default::default()
            });
            complete.nodes.push(proto::Node {
                sort_id: 0,
                kind: Some(proto::node::Kind::PrimitiveValue(proto::PrimitiveValue {
                    value: Some(proto::primitive_value::Value::String("unused".into())),
                })),
                ..Default::default()
            });
            complete.files.push(proto::SourceFile {
                name: "padding.egg".into(),
                contents: None,
            });
        }
        types.export_builtin_definition(helper, &mut complete)?;
        builtin::definitions::reconcile_declarations(&consumer, &mut complete)?;
        if !helper_first {
            let root = complete.declarations.iter().position(|declaration| {
                matches!(&declaration.kind, Some(proto::declaration::Kind::HostPrimitive(primitive)) if primitive.name == key)
            }).unwrap();
            let helper = complete.declarations.iter().position(|declaration| {
                matches!(&declaration.kind, Some(proto::declaration::Kind::HostPrimitive(primitive)) if primitive.name == helper)
            }).unwrap();
            complete.declarations.swap(root, helper);
        }
        *draft = complete;
        Ok(())
    }

    #[test]
    fn macro_context_retains_root_closure_and_native_helper_identity() {
        for (helper_first, relocated) in [(true, false), (false, true)] {
            let mut graph = EGraph::default();
            add_literal_prim!(
                &mut graph,
                "context-helper"[id = "test.context_helper"] = || -> i64 {
                    {
                        CALLS.fetch_add(1, Ordering::SeqCst);
                        17
                    }
                }
            );
            let integer = graph.get_sort_by_name("i64").unwrap().clone();
            let before = ResolvedCall::from_resolution(
                "context-helper",
                std::slice::from_ref(&integer),
                graph.type_info(),
                Context::Pure,
                &ast::Span::Panic,
            )
            .unwrap();
            let count = CALLS.load(Ordering::SeqCst);
            add_literal_prim!(&mut graph, "context-root"[id = "test.context_root", context = |types, key, draft| {
                supply(types, key, draft, "test.context_helper", helper_first, relocated, false)
            }] = |value: i64| -> i64 { value });
            for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
                let alias = ResolvedCall::from_resolution(
                    "context-root",
                    &[integer.clone(), integer.clone()],
                    graph.type_info(),
                    context,
                    &ast::Span::Panic,
                )
                .unwrap();
                let keyed = ResolvedCall::from_resolution(
                    "test.context_root",
                    &[integer.clone(), integer.clone()],
                    graph.type_info(),
                    context,
                    &ast::Span::Panic,
                )
                .unwrap();
                assert_eq!(alias, keyed);
            }
            let after = ResolvedCall::from_resolution(
                "context-helper",
                std::slice::from_ref(&integer),
                graph.type_info(),
                Context::Pure,
                &ast::Span::Panic,
            )
            .unwrap();
            assert_eq!(before, after);
            let (ResolvedCall::Primitive(before), ResolvedCall::Primitive(after)) = (before, after)
            else {
                unreachable!()
            };
            assert!(Arc::ptr_eq(
                before.validator().unwrap(),
                after.validator().unwrap()
            ));
            assert_eq!(
                graph
                    .type_info()
                    .get_prims("test.context_helper")
                    .unwrap()
                    .len(),
                1
            );
            let mut exported = proto::Program::default();
            graph
                .type_info()
                .export_builtin_definition("test.context_root", &mut exported)
                .unwrap();
            let root = exported.declarations.iter().find(|declaration| {
                matches!(&declaration.kind, Some(proto::declaration::Kind::HostPrimitive(primitive)) if primitive.name == "test.context_root")
            }).unwrap();
            let default = root
                .bindings
                .as_ref()
                .unwrap()
                .python
                .as_ref()
                .unwrap()
                .views[0]
                .params[0]
                .default_expr
                .unwrap();
            if !relocated {
                assert_eq!(default, 0);
            }
            assert!(
                matches!(&exported.nodes[default as usize].kind, Some(proto::node::Kind::Call(call)) if call.func == "test.context_helper" && call.args.is_empty())
            );
            assert_eq!(
                CALLS.load(Ordering::SeqCst),
                count,
                "metadata must invoke neither body nor validator"
            );
            graph
                .parse_and_run_program(
                    None,
                    "(check (= (context-root 5) 5) (= (context-helper) 17))",
                )
                .unwrap();
        }
    }

    #[test]
    fn context_rejects_mutation_unknown_helpers_and_atomic_duplicates() {
        let mut graph = EGraph::default();
        add_literal_prim!(
            &mut graph,
            "context-helper"[id = "test.context_helper"] = || -> i64 { 17 }
        );
        let integer = graph.get_sort_by_name("i64").unwrap().clone();
        let generated = builtin::closed_signature(
            "test.context_root",
            "context-root",
            &[("value", integer.clone())],
            &[],
            integer.clone(),
        )
        .unwrap();
        for mutation in 0..13 {
            let before = graph.type_info().builtin_catalog().unwrap().definitions;
            let error = graph.type_info().prepare_builtin_context("test.context_root", "context-root", generated.clone(), |types, key, draft| {
                supply(types, key, draft, "test.context_helper", true, true, false)?;
                let helper = draft.declarations.iter().position(|declaration| {
                    matches!(&declaration.kind, Some(proto::declaration::Kind::HostPrimitive(primitive)) if primitive.name == "test.context_helper")
                }).unwrap();
                let root = draft.declarations.iter().position(|declaration| {
                    matches!(&declaration.kind, Some(proto::declaration::Kind::HostPrimitive(primitive)) if primitive.name == key)
                }).unwrap();
                let default = draft.declarations[root].bindings.as_ref().unwrap().python.as_ref().unwrap().views[0].params[0].default_expr.unwrap();
                match mutation {
                    0 => draft.declarations[helper].bindings.as_mut().unwrap().python = Some(proto::PythonBindings::default()),
                    1 => draft.declarations[helper].bindings.as_mut().unwrap().rust = Some(proto::RustBindings::default()),
                    2 => draft.declarations[helper].bindings.as_mut().unwrap().egglog = None,
                    3 => draft.declarations[helper].doc = Some("forged helper".into()),
                    4 => if let Some(proto::node::Kind::Call(call)) = &mut draft.nodes[default as usize].kind { call.func = "test.unknown".into(); },
                    5 => if let Some(proto::node::Kind::Call(call)) = &mut draft.nodes[default as usize].kind { call.args.push(999); },
                    6 => draft.nodes[default as usize].kind = Some(proto::node::Kind::Var("free".into())),
                    7 => draft.nodes[default as usize].sort_id = 0,
                    8 => {
                        let Some(proto::declaration::Kind::HostPrimitive(primitive)) = &draft.declarations[root].kind else { unreachable!() };
                        let Some(proto::host_primitive::Typing::Signature(signature)) = &primitive.typing else { unreachable!() };
                        draft.sorts[signature.output.unwrap() as usize].kind = Some(proto::sort::Kind::Family(proto::HostSort { name: "String".into(), args: vec![] }));
                    }
                    9 => draft.declarations.push(draft.declarations[root].clone()),
                    10 => types.export_builtin_definition("egglog.core.i64.add", draft)?,
                    11 => draft.ir_version = 0,
                    12 => draft.declarations.push(draft.declarations[helper].clone()),
                    _ => unreachable!(),
                }
                Ok(())
            }).unwrap_err();
            assert!(!error.is_empty());
            assert_eq!(
                graph.type_info().builtin_catalog().unwrap().definitions,
                before
            );
            assert!(graph.type_info().get_prims("context-root").is_none());
        }
        add_literal_prim!(&mut graph, "context-root"[id = "test.context_root", context = |types, key, draft| {
            supply(types, key, draft, "test.context_helper", false, false, false)
        }] = |value: i64| -> i64 { value });
        let before = ResolvedCall::from_resolution(
            "context-root",
            &[integer.clone(), integer.clone()],
            graph.type_info(),
            Context::Pure,
            &ast::Span::Panic,
        )
        .unwrap();
        let catalog = graph.type_info().builtin_catalog().unwrap().definitions;
        assert!(
            graph
                .type_info()
                .prepare_builtin_context(
                    "test.context_root",
                    "second-root",
                    generated,
                    |_, _, _| Ok(())
                )
                .is_err()
        );
        assert_eq!(
            graph.type_info().builtin_catalog().unwrap().definitions,
            catalog
        );
        assert!(graph.type_info().get_prims("second-root").is_none());
        let after = ResolvedCall::from_resolution(
            "context-root",
            &[integer.clone(), integer],
            graph.type_info(),
            Context::Pure,
            &ast::Span::Panic,
        )
        .unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn context_follows_helper_defaults_even_with_explicit_call_arguments() {
        let mut graph = EGraph::default();
        add_literal_prim!(
            &mut graph,
            "context-leaf"[id = "test.context_leaf"] = || -> i64 { 9 }
        );
        add_literal_prim!(&mut graph, "context-middle"[id = "test.context_middle", context = |types, key, draft| {
            supply(types, key, draft, "test.context_leaf", true, false, false)
        }] = |value: i64| -> i64 { value });
        let integer = graph.get_sort_by_name("i64").unwrap().clone();
        let generated = builtin::closed_signature(
            "test.context_root",
            "context-root",
            &[("value", integer.clone())],
            &[],
            integer,
        )
        .unwrap();
        let prepared = graph
            .type_info()
            .prepare_builtin_context(
                "test.context_root",
                "context-root",
                generated.clone(),
                |types, key, draft| {
                    supply(types, key, draft, "test.context_middle", true, true, true)
                },
            )
            .unwrap();
        for key in [
            "test.context_leaf",
            "test.context_middle",
            "test.context_root",
        ] {
            assert!(prepared.declarations.iter().any(|declaration| {
                matches!(&declaration.kind, Some(proto::declaration::Kind::HostPrimitive(primitive)) if primitive.name == key)
            }));
        }
        let mut relocated = proto::Program {
            ir_version: 1,
            sorts: vec![proto::Sort {
                kind: Some(proto::sort::Kind::Family(proto::HostSort {
                    name: "String".into(),
                    args: vec![],
                })),
                ..Default::default()
            }],
            ..Default::default()
        };
        builtin::import_rooted_definition("test.context_root", &prepared, &mut relocated).unwrap();
        let once = relocated.clone();
        builtin::import_rooted_definition("test.context_root", &prepared, &mut relocated).unwrap();
        assert_eq!(
            relocated, once,
            "equivalent defaults survive arena relocation"
        );
        let error = graph.type_info().prepare_builtin_context("test.context_root", "context-root", generated.clone(), |types, key, draft| {
            supply(types, key, draft, "test.context_middle", true, true, true)?;
            let helper = draft.declarations.iter().find(|declaration| {
                matches!(&declaration.kind, Some(proto::declaration::Kind::HostPrimitive(primitive)) if primitive.name == "test.context_middle")
            }).unwrap();
            let default = helper.bindings.as_ref().unwrap().python.as_ref().unwrap().views[0].params[0].default_expr.unwrap();
            draft.nodes[default as usize].kind = Some(proto::node::Kind::PrimitiveValue(proto::PrimitiveValue {
                value: Some(proto::primitive_value::Value::I64(99)),
            }));
            Ok(())
        }).unwrap_err();
        assert!(error.contains("helper defaults"), "{error}");
        let error = graph.type_info().prepare_builtin_context("test.context_root", "context-root", generated, |types, key, draft| {
            supply(types, key, draft, "test.context_middle", false, false, true)?;
            draft.declarations.retain(|declaration| {
                !matches!(&declaration.kind, Some(proto::declaration::Kind::HostPrimitive(primitive)) if primitive.name == "test.context_leaf")
            });
            Ok(())
        }).unwrap_err();
        assert!(error.contains("callee"), "{error}");
    }

    #[test]
    fn root_context_preserves_absent_empty_and_first_language_supplies() {
        let mut graph = EGraph::default();
        let integer = graph.get_sort_by_name("i64").unwrap().clone();
        let generated = builtin::closed_signature(
            "test.language_root",
            "language-root",
            &[("value", integer.clone())],
            &[],
            integer,
        )
        .unwrap();
        for python in [false, true] {
            for empty_first in [false, true] {
                let mut first = generated.clone();
                let root = first.declarations.last_mut().unwrap();
                let Some(proto::declaration::Kind::HostPrimitive(primitive)) = &root.kind else {
                    unreachable!()
                };
                let Some(proto::host_primitive::Typing::Signature(signature)) = &primitive.typing
                else {
                    unreachable!()
                };
                let presentation =
                    builtin::scalar_member_bindings(signature, "method", "method", &[], None, None);
                if python {
                    root.bindings.as_mut().unwrap().python = if empty_first {
                        Some(proto::PythonBindings::default())
                    } else {
                        presentation.python
                    };
                } else {
                    root.bindings.as_mut().unwrap().rust = if empty_first {
                        Some(proto::RustBindings::default())
                    } else {
                        presentation.rust
                    };
                }
                let prepared = graph
                    .type_info()
                    .prepare_builtin_context(
                        "test.language_root",
                        "language-root",
                        generated.clone(),
                        |_, _, draft| {
                            draft.declarations.last_mut().unwrap().bindings =
                                first.declarations.last().unwrap().bindings.clone();
                            Ok(())
                        },
                    )
                    .unwrap();
                let frozen = prepared.clone();
                let mut absent = generated.clone();
                builtin::definitions::reconcile_declarations(&prepared, &mut absent).unwrap();
                assert_eq!(absent, prepared);
                let mut changed = first;
                let bindings = changed
                    .declarations
                    .last_mut()
                    .unwrap()
                    .bindings
                    .as_mut()
                    .unwrap();
                if python {
                    let views = &mut bindings.python.as_mut().unwrap().views;
                    if empty_first {
                        views.push(proto::PythonCallable {
                            kind: proto::PythonCallKind::Function.into(),
                            path: vec!["different".into()],
                            params: vec![proto::PythonParameter {
                                core_input: Some(0),
                                name: "value".into(),
                                default_expr: None,
                            }],
                            ..Default::default()
                        });
                    } else {
                        views.clear();
                    }
                } else {
                    let views = &mut bindings.rust.as_mut().unwrap().views;
                    if empty_first {
                        views.push(proto::RustCallable {
                            path: vec!["different".into()],
                            params: vec![proto::RustParameter {
                                core_input: Some(0),
                                name: "value".into(),
                                borrowed: false,
                            }],
                            ..Default::default()
                        });
                    } else {
                        views.clear();
                    }
                }
                let mut destination = prepared;
                assert!(
                    builtin::definitions::reconcile_declarations(&changed, &mut destination)
                        .is_err()
                );
                assert_eq!(destination, frozen);
            }
        }
    }

    #[test]
    fn context_family_evidence_cannot_first_supply_absent_languages() {
        #[derive(Debug)]
        struct ContextScalar;
        impl prelude::BaseSort for ContextScalar {
            type Base = i64;
            fn name(&self) -> &str {
                "ContextScalar"
            }
            fn reconstruct_termdag(
                &self,
                _: &sort::BaseValues,
                _: Value,
                _: &mut TermDag,
            ) -> TermId {
                panic!("family metadata must not reconstruct a native value")
            }
        }
        for outer_present in [false, true] {
            let mut graph = EGraph::default();
            prelude::add_base_sort(&mut graph, ContextScalar, ast::Span::Panic).unwrap();
            graph
                .type_info()
                .register_builtin_family(proto::HostSortFamily {
                    name: "ContextScalar".into(),
                    arity: 0,
                    bindings: outer_present.then(proto::SortBindings::default),
                })
                .unwrap();
            let scalar = graph.get_sort_by_name("ContextScalar").unwrap().clone();
            let generated = builtin::closed_signature(
                "test.family_root",
                "family-root",
                &[("value", scalar.clone())],
                &[],
                scalar,
            )
            .unwrap();
            let before = graph.type_info().builtin_catalog().unwrap().definitions;
            let prepared = graph
                .type_info()
                .prepare_builtin_context(
                    "test.family_root",
                    "family-root",
                    generated.clone(),
                    |_, _, _| Ok(()),
                )
                .unwrap();
            let family = prepared
                .declarations
                .iter()
                .find_map(|declaration| match &declaration.kind {
                    Some(proto::declaration::Kind::HostSortFamily(family))
                        if family.name == "ContextScalar" =>
                    {
                        Some(family)
                    }
                    _ => None,
                })
                .unwrap();
            assert_eq!(
                family.bindings,
                outer_present.then(proto::SortBindings::default)
            );
            for mutation in 0..3 {
                if mutation == 0 && outer_present {
                    continue;
                }
                let error = graph
                    .type_info()
                    .prepare_builtin_context(
                        "test.family_root",
                        "family-root",
                        generated.clone(),
                        |_, _, draft| {
                            let family = draft
                                .declarations
                                .iter_mut()
                                .find_map(|declaration| match &mut declaration.kind {
                                    Some(proto::declaration::Kind::HostSortFamily(family))
                                        if family.name == "ContextScalar" =>
                                    {
                                        Some(family)
                                    }
                                    _ => None,
                                })
                                .unwrap();
                            if mutation == 0 {
                                family.bindings = Some(proto::SortBindings::default());
                            } else {
                                let forged = proto::TypeBinding {
                                    path: vec!["forged".into(), "ContextScalar".into()],
                                    ..Default::default()
                                };
                                let bindings = family.bindings.get_or_insert_default();
                                if mutation == 1 {
                                    bindings.python = Some(forged);
                                } else {
                                    bindings.rust = Some(forged);
                                }
                            }
                            Ok(())
                        },
                    )
                    .unwrap_err();
                assert!(error.contains("context family"), "{error}");
                assert_eq!(
                    graph.type_info().builtin_catalog().unwrap().definitions,
                    before
                );
                assert!(!graph.type_info().is_primitive("family-root"));
            }
            for duplicate in [false, true] {
                let error = graph.type_info().prepare_builtin_context("test.family_root", "family-root", generated.clone(), |_, _, draft| {
                    let family = draft.declarations.iter().position(|declaration| {
                        matches!(&declaration.kind, Some(proto::declaration::Kind::HostSortFamily(family)) if family.name == "ContextScalar")
                    }).unwrap();
                    if duplicate { draft.declarations.push(draft.declarations[family].clone()); }
                    else {
                        draft.files.push(proto::SourceFile { name: "forged-family.egg".into(), contents: Some("family".into()) });
                        draft.declarations[family].span = Some(proto::Span { file: 0, start: 0, end: 6 });
                    }
                    Ok(())
                }).unwrap_err();
                assert!(error.contains("context family"), "{error}");
                assert_eq!(
                    graph.type_info().builtin_catalog().unwrap().definitions,
                    before
                );
            }
        }
    }
}

#[test]
fn native_signature_all_selection_routes_intersect_native_predicate() {
    // This regression uses only baseline APIs, so the same test can demonstrate
    // behavioral red-before/fix-after independently of the new binding APIs.
    #[derive(Clone)]
    struct Narrowed {
        definition: proto::Program,
        native: ArcSort,
    }
    impl Primitive for Narrowed {
        fn name(&self) -> &str {
            "signature-intersection"
        }
        fn get_type_constraints(&self, span: &ast::Span) -> Box<dyn constraint::TypeConstraint> {
            constraint::SimpleTypeConstraint::new(
                self.name(),
                vec![self.native.clone(), self.native.clone()],
                span.clone(),
            )
            .into_box()
        }
        fn builtin_definition(&self) -> Option<(&str, &proto::Program)> {
            Some(("test.signature_intersection", &self.definition))
        }
    }
    impl PurePrim for Narrowed {
        fn apply<'a, 'db>(&self, _: PureState<'a, 'db>, _: &[Value]) -> Option<Value> {
            panic!("signature validation must not execute this fixture")
        }
    }
    let mut graph = EGraph::default();
    let integer = graph.get_sort_by_name("i64").unwrap().clone();
    let string = graph.get_sort_by_name("String").unwrap().clone();
    let definition = builtin::closed_signature(
        "test.signature_intersection",
        "signature-intersection",
        &[("input", integer.clone())],
        &[],
        string,
    )
    .unwrap();
    graph.add_pure_primitive(
        Narrowed {
            definition,
            native: integer.clone(),
        },
        Some(Arc::new(|_: &mut TermDag, _: &[TermId]| {
            panic!("signature validation must not execute a validator")
        })),
    );
    let actual = [integer.clone(), integer];
    let types = graph.type_info();
    for key in ["signature-intersection", "test.signature_intersection"] {
        assert!(
            !types.get_prims(key).unwrap()[0].accept(&actual, types),
            "native identity constraints must also satisfy the String-result descriptor"
        );
        for ctx in [Context::Pure, Context::Read, Context::Write, Context::Full] {
            assert!(
                ResolvedCall::from_resolution(key, &actual, types, ctx, &ast::Span::Panic).is_err()
            );
        }
        assert!(
            types
                .export_function_target(
                    key,
                    &actual[..1],
                    &actual[1],
                    None,
                    &mut proto::Program::default()
                )
                .is_err()
        );
    }
    graph
        .parse_and_run_program(
            None,
            "(sort F (UnstableFn (i64) i64)) (function saved () F :merge new)",
        )
        .unwrap();
    for key in ["signature-intersection", "test.signature_intersection"] {
        assert!(
            graph
                .parse_and_run_program(None, &format!("(set (saved) (unstable-fn {key:?}))"))
                .is_err()
        );
    }
}

#[test]
fn function_value_observer_keeps_typed_occurrences() {
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(
            None,
            r#"
        (sort X)
        (sort SI (Set i64))
        (sort SX (Set X))
        (function si () SI :merge new)
        (function sx () SX :merge new)
        (set (si) (set-empty))
        (set (sx) (set-empty))
        (sort F (UnstableFn () bool))
        (datatype W (Wrap F) (Other F))
        (let $fi (unstable-fn "egglog.core.bool_ne" (si) (si)))
        (let $fx (unstable-fn "egglog.core.bool_ne" (sx) (sx)))
        (check (!= $fi $fx))
        (let $wi (Wrap $fi))
        (let $wx (Wrap $fx))
        (let $other (Other $fi))
        (union $wi $wx)
    "#,
        )
        .unwrap();
    let mut roots = vec![];
    for name in ["$fi", "$fx", "$wi", "$other"] {
        roots.push(
            graph
                .eval_expr(&ast::Expr::Var(ast::Span::Panic, name.into()))
                .unwrap(),
        );
    }
    let mut nodes: Vec<(TermId, String, Vec<usize>)> = vec![];
    let best_roots = vec![roots[0].clone(), roots[1].clone(), roots[0].clone()];
    let legacy = graph.extract_best(best_roots.clone()).unwrap();
    let (best, ids) = graph
        .extract_best_with_cost_model_and_observer(
            best_roots,
            DEFAULT_COST_MODEL,
            |_, term, sort, _, children| {
                let id = nodes.len();
                nodes.push((term, sort.name().into(), children.to_vec()));
                Ok(id)
            },
        )
        .unwrap();
    assert_eq!(best.termdag, legacy.termdag);
    assert_eq!(
        best.terms
            .iter()
            .map(|t| t.as_ref().map(|t| (t.term, t.cost)))
            .collect::<Vec<_>>(),
        legacy
            .terms
            .iter()
            .map(|t| t.as_ref().map(|t| (t.term, t.cost)))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        best.terms[0].as_ref().unwrap().term,
        best.terms[1].as_ref().unwrap().term
    );
    assert_ne!(ids[0], ids[1]);
    assert_eq!(ids[0], ids[2], "duplicate roots retain consumer sharing");
    for (id, expected) in ids.iter().take(2).zip(["SI", "SX"]) {
        let children = &nodes[id.unwrap()].2;
        assert_eq!(
            children.len(),
            2,
            "target-name literal is not a semantic child"
        );
        assert_eq!(nodes[children[0]].1, expected);
        assert_eq!(children[0], children[1]);
    }
    let legacy = graph
        .extract_variants(vec![roots[2].clone(), roots[3].clone()], 2)
        .unwrap();
    nodes.clear();
    let (variants, ids) = graph
        .extract_variants_with_cost_model_and_observer(
            vec![roots[2].clone(), roots[3].clone()],
            2,
            DEFAULT_COST_MODEL,
            |_, term, sort, _, children| {
                let id = nodes.len();
                nodes.push((term, sort.name().into(), children.to_vec()));
                Ok(id)
            },
        )
        .unwrap();
    assert_eq!(variants.termdag, legacy.termdag);
    let terms: Vec<_> = variants.variants[0]
        .iter()
        .map(|t| (t.term, t.cost))
        .collect();
    assert_eq!(
        terms,
        legacy.variants[0]
            .iter()
            .map(|t| (t.term, t.cost))
            .collect::<Vec<_>>()
    );
    assert_eq!(terms.len(), 2);
    assert_eq!(terms[0], terms[1]);
    assert_eq!(terms[0].1, 1);
    assert_ne!(
        ids[0][0], ids[0][1],
        "selected variants must not be deduplicated by Eq root or TermId"
    );
    let mut prefixes: Vec<_> = ids[0]
        .iter()
        .map(|id| {
            let function = nodes[*id].2[0];
            nodes[nodes[function].2[0]].1.as_str()
        })
        .collect();
    prefixes.sort();
    assert_eq!(prefixes, ["SI", "SX"]);
    let shared = nodes[ids[1][0]].2[0];
    assert!(
        ids[0].iter().any(|id| nodes[*id].2[0] == shared),
        "distinct roots retain their common selected function child"
    );
    let error = graph
        .extract_best_with_cost_model_and_observer(
            vec![roots[0].clone()],
            DEFAULT_COST_MODEL,
            |_, _, _, _, _: &[()]| Err::<(), _>(Error::ExtractError("observer failure".into())),
        )
        .unwrap_err();
    assert!(error.to_string().contains("observer failure"));
    let error = graph
        .extract_variants_with_cost_model_and_observer(
            vec![roots[2].clone()],
            2,
            DEFAULT_COST_MODEL,
            |_, _, _, _, _: &[()]| Err::<(), _>(Error::ExtractError("observer failure".into())),
        )
        .unwrap_err();
    assert!(error.to_string().contains("observer failure"));
}

#[test]
fn function_value_observer_uses_semantic_view_children() {
    let mut graph = EGraph::new_with_proofs();
    graph
        .parse_and_run_program(None, "(datatype M (Num i64)) (let $m (Num 5))")
        .unwrap();
    let mut values = vec![];
    graph
        .constructor_enodes("Num", |row| values.push(row.eclass))
        .unwrap();
    assert_eq!(values.len(), 1);
    let root = (graph.get_sort_by_name("M").unwrap().clone(), values[0]);
    let mut nodes: Vec<(String, Vec<usize>)> = vec![];
    let (_, roots) = graph
        .extract_best_with_cost_model_and_observer(
            vec![root],
            DEFAULT_COST_MODEL,
            |_, _, sort, _, children| {
                let id = nodes.len();
                nodes.push((sort.name().into(), children.to_vec()));
                Ok(id)
            },
        )
        .unwrap();
    let root = &nodes[roots[0].unwrap()];
    assert_eq!(root.0, "M");
    assert_eq!(root.1.len(), 1);
    assert_eq!(nodes[root.1[0]].0, "i64");
}

#[test]
fn function_application_python_view_is_provider_owned() {
    let mut graph = EGraph::default();
    let catalog = graph.type_info().builtin_catalog().unwrap().definitions;
    let declaration = catalog
        .declarations
        .iter()
        .find(|d| {
            matches!(&d.kind,
                Some(proto::declaration::Kind::HostPrimitive(p)) if p.name == "egglog.core.fn.apply"
            )
        })
        .unwrap();
    let bindings = declaration.bindings.as_ref().unwrap();
    assert_eq!(
        bindings.egglog.as_ref().unwrap().views[0].symbol,
        "unstable-app"
    );
    assert_eq!(bindings.rust.as_ref().unwrap().views[0].path, ["call"]);
    assert!(matches!(
        bindings.rust.as_ref().unwrap().views[0]
            .owner
            .as_ref()
            .and_then(|owner| owner.kind.as_ref()),
        Some(proto::binding_owner::Kind::Function(_))
    ));
    assert_eq!(
        bindings
            .python
            .as_ref()
            .expect("provider function-owner view")
            .views,
        [proto::PythonCallable {
            kind: proto::PythonCallKind::Method.into(),
            path: vec!["__call__".into()],
            owner: Some(proto::BindingOwner {
                kind: Some(proto::binding_owner::Kind::Function(
                    proto::FunctionTypeOwner {}
                )),
            }),
            receiver: Some(0),
            params: vec![proto::PythonParameter {
                core_input: Some(1),
                name: "args".into(),
                default_expr: None,
            }],
            ..Default::default()
        }]
    );
}

#[test]
fn function_application_python_metadata_reconciles_and_rejects_malformed_views() {
    let mut definition = <sort::FunctionSort as Presort>::builtin_definitions().remove(0);
    definition.declarations[0].bindings.as_mut().unwrap().rust = None;
    let view = proto::PythonCallable {
        kind: proto::PythonCallKind::Method.into(),
        path: vec!["__call__".into()],
        owner: Some(proto::BindingOwner {
            kind: Some(proto::binding_owner::Kind::Function(
                proto::FunctionTypeOwner {},
            )),
        }),
        receiver: Some(0),
        params: vec![proto::PythonParameter {
            core_input: Some(1),
            name: "args".into(),
            default_expr: None,
        }],
        ..Default::default()
    };
    definition.declarations[0].bindings.as_mut().unwrap().python =
        Some(proto::PythonBindings { views: vec![view] });
    let mut absent = definition.clone();
    absent.declarations[0].bindings.as_mut().unwrap().python = None;
    for supplies in [[&absent, &definition], [&definition, &absent]] {
        let mut destination = proto::Program::default();
        for supply in supplies {
            builtin::definitions::reconcile_declarations(supply, &mut destination).unwrap();
        }
        builtin::definitions::reconcile_declarations(&definition, &mut destination).unwrap();
        assert_eq!(
            destination, definition,
            "absence cannot erase the first supplied block"
        );
        let mut batch = definition.clone();
        batch.declarations = supplies.iter().map(|p| p.declarations[0].clone()).collect();
        let mut destination = proto::Program::default();
        builtin::definitions::reconcile_declarations(&batch, &mut destination).unwrap();
        assert_eq!(
            destination, definition,
            "same-fragment reconciliation agrees"
        );
    }
    let mut empty = absent.clone();
    empty.declarations[0].bindings.as_mut().unwrap().python =
        Some(proto::PythonBindings::default());
    for supplies in [[&empty, &definition], [&definition, &empty]] {
        let mut destination = proto::Program::default();
        builtin::definitions::reconcile_declarations(supplies[0], &mut destination).unwrap();
        let before = destination.clone();
        assert!(
            builtin::definitions::reconcile_declarations(supplies[1], &mut destination).is_err()
        );
        assert_eq!(
            destination, before,
            "a supplied empty block is still immutable"
        );
        let mut batch = definition.clone();
        batch.declarations = supplies.iter().map(|p| p.declarations[0].clone()).collect();
        let mut destination = proto::Program::default();
        assert!(builtin::definitions::reconcile_declarations(&batch, &mut destination).is_err());
        assert_eq!(destination, proto::Program::default());
    }
    let mut no_bindings = absent.clone();
    no_bindings.declarations[0].bindings = None;
    let mut destination = no_bindings;
    builtin::definitions::reconcile_declarations(&definition, &mut destination).unwrap();
    assert_eq!(destination, definition);
    let mut empty_rust = definition.clone();
    empty_rust.declarations[0].bindings.as_mut().unwrap().rust =
        Some(proto::RustBindings::default());
    builtin::definitions::reconcile_declarations(&empty_rust, &mut destination).unwrap();
    assert_eq!(destination, empty_rust);
    for case in [
        "owner_missing",
        "owner_unset",
        "owner_sort",
        "kind",
        "unknown_kind",
        "path_empty",
        "path_component_empty",
        "path_qualified",
        "path_other",
        "receiver_missing",
        "receiver_tail",
        "params_missing",
        "params_extra",
        "slot_missing",
        "slot_receiver",
        "slot_other",
        "param_empty",
        "param_other",
        "default",
        "mutates_zero",
        "mutates_tail",
        "tail_zero",
        "tail_one",
        "views_extra",
        "views_duplicate",
        "non_application",
        "rust",
        "egglog_empty",
        "egglog_member",
    ] {
        let mut invalid = definition.clone();
        // A valid referenced scalar/default makes the negative discriminate
        // presentation validation from an out-of-bounds arena reference.
        invalid.sorts.push(proto::Sort {
            kind: Some(proto::sort::Kind::Family(proto::HostSort {
                name: "i64".into(),
                args: vec![],
            })),
            ..Default::default()
        });
        invalid.nodes.push(proto::Node {
            sort_id: 0,
            kind: Some(proto::node::Kind::PrimitiveValue(proto::PrimitiveValue {
                value: Some(proto::primitive_value::Value::I64(1)),
            })),
            ..Default::default()
        });
        let bindings = invalid.declarations[0].bindings.as_mut().unwrap();
        let view = &mut bindings.python.as_mut().unwrap().views[0];
        match case {
            "owner_missing" => view.owner = None,
            "owner_unset" => view.owner = Some(proto::BindingOwner::default()),
            "owner_sort" => {
                view.owner.as_mut().unwrap().kind = Some(proto::binding_owner::Kind::Sort(0))
            }
            "kind" => view.kind = proto::PythonCallKind::Function.into(),
            "unknown_kind" => view.kind = 99,
            "path_empty" => view.path.clear(),
            "path_component_empty" => view.path[0].clear(),
            "path_qualified" => view.path.insert(0, "module".into()),
            "path_other" => view.path[0] = "apply".into(),
            "receiver_missing" => view.receiver = None,
            "receiver_tail" => view.receiver = Some(1),
            "params_missing" => view.params.clear(),
            "params_extra" => view.params.push(view.params[0].clone()),
            "slot_missing" => view.params[0].core_input = None,
            "slot_receiver" => view.params[0].core_input = Some(0),
            "slot_other" => view.params[0].core_input = Some(2),
            "param_empty" => view.params[0].name.clear(),
            "param_other" => view.params[0].name = "values".into(),
            "default" => view.params[0].default_expr = Some(0),
            "mutates_zero" => view.mutates = Some(0),
            "mutates_tail" => view.mutates = Some(1),
            "tail_zero" => view.tail_repetitions = Some(0),
            "tail_one" => view.tail_repetitions = Some(1),
            "views_extra" => {
                let mut other = view.clone();
                other.path[0] = "apply".into();
                bindings.python.as_mut().unwrap().views.push(other);
            }
            "views_duplicate" => {
                let duplicate = view.clone();
                bindings.python.as_mut().unwrap().views.push(duplicate);
            }
            "non_application" => {
                let Some(proto::declaration::Kind::HostPrimitive(primitive)) =
                    &mut invalid.declarations[0].kind
                else {
                    unreachable!()
                };
                primitive.typing = Some(proto::host_primitive::Typing::Signature(
                    proto::GenericSignature {
                        inputs: vec![proto::Arg {
                            name: "receiver".into(),
                            sort: 0,
                        }],
                        varargs: vec![proto::Arg {
                            name: "args".into(),
                            sort: 0,
                        }],
                        output: Some(0),
                        ..Default::default()
                    },
                ));
            }
            "rust" => {
                bindings.rust = Some(proto::RustBindings {
                    views: vec![proto::RustCallable::default()],
                })
            }
            "egglog_empty" => bindings.egglog.as_mut().unwrap().views[0].symbol.clear(),
            "egglog_member" => bindings.egglog.as_mut().unwrap().views[0].datatype_member = true,
            _ => unreachable!(),
        }
        let mut destination = proto::Program::default();
        let error = builtin::definitions::reconcile_declarations(&invalid, &mut destination)
            .expect_err(case);
        if case == "rust" {
            assert!(error.contains("invalid function application Rust view"));
        }
        assert_eq!(
            destination,
            proto::Program::default(),
            "{case} must be transactional"
        );
    }
}

#[test]
fn function_application_rust_metadata_rejects_malformed_views() {
    let definition = <sort::FunctionSort as Presort>::builtin_definitions().remove(0);
    builtin::definitions::reconcile_declarations(&definition, &mut proto::Program::default())
        .unwrap();
    for case in 0..13 {
        let mut invalid = definition.clone();
        let bindings = invalid.declarations[0]
            .bindings
            .as_mut()
            .unwrap()
            .rust
            .as_mut()
            .unwrap();
        let view = &mut bindings.views[0];
        match case {
            0 => view.path.clear(),
            1 => view.path[0].clear(),
            2 => view.owner = None,
            3 => view.owner = Some(proto::BindingOwner::default()),
            4 => view.receiver = None,
            5 => view.receiver.as_mut().unwrap().core_input = Some(1),
            6 => view.receiver.as_mut().unwrap().borrowed = false,
            7 => view.params.clear(),
            8 => view.params[0].core_input = Some(0),
            9 => view.params[0].borrowed = true,
            10 => view.borrowed_self = true,
            11 => view.tail_repetitions = Some(1),
            _ => {
                let duplicate = view.clone();
                bindings.views.push(duplicate);
            }
        }
        assert!(
            builtin::definitions::reconcile_declarations(&invalid, &mut proto::Program::default())
                .is_err(),
            "malformed Rust application view {case}"
        );
    }
}

#[test]
fn multiset_index_operations_preserve_function_capture_prefixes() {
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(
            None,
            r#"
        (sort PrefixMS (MultiSet i64))
        (sort PrefixFn (UnstableFn (PrefixMS i64) i64))
        (function prefix-index (i64 PrefixMS i64) i64 :merge new)
        (let xs (multiset-of 1 1 2))
        (let f (unstable-fn "prefix-index" 41))
        (set (prefix-index 42 xs 1) 99)
        (unstable-multiset-fill-index xs f)
        (check (= (prefix-index 41 xs 1) 2))
        (check (= (prefix-index 41 xs 2) 1))
        (check (= (prefix-index 42 xs 1) 99))
        (unstable-multiset-clear-index xs f)
        (fail (check (= (prefix-index 41 xs 1) 2)))
        (fail (check (= (prefix-index 41 xs 2) 1)))
        (check (= (prefix-index 42 xs 1) 99))
    "#,
        )
        .unwrap();
}

#[test]
fn function_value_structural_provenance_and_application_catalog() {
    let mut graph = EGraph::default();
    let catalog = graph.type_info().builtin_catalog().unwrap().definitions;
    let declaration = catalog
        .declarations
        .iter()
        .find(|d| {
            matches!(&d.kind,
        Some(proto::declaration::Kind::HostPrimitive(p)) if p.name == "egglog.core.fn.apply")
        })
        .expect("uninstantiated function application descriptor");
    let Some(proto::declaration::Kind::HostPrimitive(primitive)) = &declaration.kind else {
        unreachable!()
    };
    assert!(matches!(
        primitive.typing,
        Some(proto::host_primitive::Typing::Application(_))
    ));
    graph
        .type_info()
        .check_builtin_signature(&catalog, primitive)
        .unwrap();
    graph
        .parse_and_run_program(
            None,
            "(sort F (UnstableFn (i64) i64)) (sort G (UnstableFn (i64) i64))",
        )
        .unwrap();
    let mut sorts = vec![];
    let f = graph.get_sort_by_name("F").unwrap().clone();
    let g = graph.get_sort_by_name("G").unwrap().clone();
    let index = graph.export_sort(&f, &mut sorts).unwrap();
    assert_eq!(graph.export_sort(&g, &mut sorts).unwrap(), index);
    let Some(proto::sort::Kind::Func(function)) = &sorts[index as usize].kind else {
        panic!("expected structural Func")
    };
    assert_eq!(function.params, [function.result]);
    let mut imported = vec![];
    assert_eq!(
        builtin::import_sort(&sorts, index, &mut imported).unwrap(),
        index
    );
    assert_eq!(imported, sorts);
    for sort in [f, g] {
        let i64_sort = graph.get_sort_by_name("i64").unwrap().clone();
        let native = [sort.clone(), i64_sort.clone(), i64_sort.clone()];
        let instance = graph
            .type_info()
            .instantiate_builtin("egglog.core.fn.apply", &sort)
            .unwrap();
        instance.check_abi(&[None, None], None, false).unwrap();
        assert!(instance.check_abi(&[None], None, false).is_err());
        assert!(instance.check_abi(&[None, None], None, true).is_err());
        assert!(
            instance
                .check_abi(
                    &[Some((std::any::TypeId::of::<i64>(), false)), None],
                    None,
                    false
                )
                .is_err()
        );
        for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
            let source = ResolvedCall::from_resolution(
                "unstable-app",
                &native,
                graph.type_info(),
                context,
                &ast::Span::Panic,
            )
            .unwrap();
            let keyed = ResolvedCall::from_resolution(
                "egglog.core.fn.apply",
                &native,
                graph.type_info(),
                context,
                &ast::Span::Panic,
            )
            .unwrap();
            assert_eq!(source, keyed);
            let ResolvedCall::Primitive(primitive) = keyed else {
                unreachable!()
            };
            assert!(primitive.validator().is_none());
        }
    }
}

#[test]
fn remaining_scalar_catalog_preserves_closed_providers() {
    let mut graph = EGraph::default();
    let catalog = graph.type_info().builtin_catalog().unwrap().definitions;
    let cases: &[(&str, &str, &[&str], Option<&str>)] = &[
        ("i64.bitor", "|", &["i64", "i64", "i64"], Some("__or__")),
        ("i64.bitxor", "^", &["i64", "i64", "i64"], Some("__xor__")),
        ("i64.shl", "<<", &["i64", "i64", "i64"], Some("__lshift__")),
        ("i64.shr", ">>", &["i64", "i64", "i64"], Some("__rshift__")),
        ("i64.not", "not-i64", &["i64", "i64"], Some("__invert__")),
        ("i64.log2", "log2", &["i64", "i64"], Some("log2")),
        ("i64.abs", "abs", &["i64", "i64"], Some("__abs__")),
        ("i64.gt", ">", &["i64", "i64", "Unit"], Some("__gt__")),
        ("i64.le", "<=", &["i64", "i64", "Unit"], Some("__le__")),
        ("i64.bool_eq", "bool-=", &["i64", "i64", "bool"], None),
        (
            "i64.bool_lt",
            "bool-<",
            &["i64", "i64", "bool"],
            Some("bool_lt"),
        ),
        (
            "i64.bool_gt",
            "bool->",
            &["i64", "i64", "bool"],
            Some("bool_gt"),
        ),
        (
            "i64.bool_le",
            "bool-<=",
            &["i64", "i64", "bool"],
            Some("bool_le"),
        ),
        (
            "i64.bool_ge",
            "bool->=",
            &["i64", "i64", "bool"],
            Some("bool_ge"),
        ),
        (
            "string.count_matches",
            "count-matches",
            &["String", "String", "i64"],
            Some("count_matches"),
        ),
        ("f64.sub", "-", &["f64", "f64", "f64"], Some("__sub__")),
        ("f64.mul", "*", &["f64", "f64", "f64"], Some("__mul__")),
        ("f64.div", "/", &["f64", "f64", "f64"], Some("__truediv__")),
        ("f64.rem", "%", &["f64", "f64", "f64"], Some("__mod__")),
        ("f64.pow", "^", &["f64", "f64", "f64"], Some("__pow__")),
        ("f64.neg", "neg", &["f64", "f64"], Some("__neg__")),
        ("f64.lt", "<", &["f64", "f64", "Unit"], Some("__lt__")),
        ("f64.gt", ">", &["f64", "f64", "Unit"], Some("__gt__")),
        ("f64.le", "<=", &["f64", "f64", "Unit"], Some("__le__")),
        ("f64.ge", ">=", &["f64", "f64", "Unit"], Some("__ge__")),
        ("f64.min", "min", &["f64", "f64", "f64"], Some("min")),
        ("f64.max", "max", &["f64", "f64", "f64"], Some("max")),
        ("f64.abs", "abs", &["f64", "f64"], Some("__abs__")),
        ("f64.exp", "exp", &["f64", "f64"], Some("exp")),
        ("f64.log", "log", &["f64", "f64"], Some("log")),
        ("f64.sqrt", "sqrt", &["f64", "f64"], Some("sqrt")),
        ("f64.from_i64", "to-f64", &["i64", "f64"], Some("from_i64")),
        ("f64.to_i64", "to-i64", &["f64", "i64"], Some("to_i64")),
        (
            "f64.to_string",
            "to-string",
            &["f64", "String"],
            Some("to_string"),
        ),
        ("bool.guard", "guard", &["bool", "Unit"], None),
    ];
    assert_eq!(cases.len(), 35);
    let missing: Vec<_> = cases
        .iter()
        .filter_map(|(suffix, ..)| {
            let key = format!("egglog.core.{suffix}");
            (!catalog.declarations.iter().any(|d| {
                matches!(&d.kind,
            Some(proto::declaration::Kind::HostPrimitive(p)) if p.name == key)
            }))
            .then_some(key)
        })
        .collect();
    assert!(
        missing.is_empty(),
        "missing scalar catalog entries: {missing:?}"
    );
    for &(suffix, alias, names, python) in cases {
        let key = format!("egglog.core.{suffix}");
        let declarations: Vec<_> = catalog
            .declarations
            .iter()
            .filter(|d| {
                matches!(&d.kind,
            Some(proto::declaration::Kind::HostPrimitive(p)) if p.name == key)
            })
            .collect();
        assert_eq!(declarations.len(), 1, "{key}");
        let declaration = declarations[0];
        let Some(proto::declaration::Kind::HostPrimitive(primitive)) = &declaration.kind else {
            unreachable!()
        };
        assert_eq!(
            builtin::signature_sorts(&catalog, primitive).unwrap(),
            names,
            "{key}"
        );
        let Some(proto::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
            unreachable!()
        };
        assert!(signature.type_params.is_empty() && signature.varargs.is_empty());
        let bindings = declaration.bindings.as_ref().unwrap();
        assert_eq!(bindings.egglog.as_ref().unwrap().views[0].symbol, alias);
        if let Some(python) = python {
            let py = &bindings.python.as_ref().unwrap().views[0];
            let rs = &bindings.rust.as_ref().unwrap().views[0];
            assert_eq!(py.path.last().unwrap(), python);
            assert_eq!(py.owner, rs.owner);
            match suffix {
                "f64.from_i64" => {
                    assert_eq!(py.kind, i32::from(proto::PythonCallKind::ClassMethod));
                    assert_eq!(
                        py.owner.as_ref().unwrap().kind,
                        Some(proto::binding_owner::Kind::Sort(signature.output.unwrap()))
                    );
                    assert_eq!(py.params[0].name, "i");
                    assert_eq!(rs.params[0].name, "i");
                    assert!(py.receiver.is_none() && rs.receiver.is_none());
                }
                "string.count_matches" => {
                    assert_eq!(py.kind, i32::from(proto::PythonCallKind::Function));
                    assert!(py.owner.is_none() && py.receiver.is_none() && rs.receiver.is_none());
                    assert_eq!(
                        py.params
                            .iter()
                            .map(|p| p.name.as_str())
                            .collect::<Vec<_>>(),
                        ["s", "pattern"]
                    );
                    assert_eq!(
                        rs.params
                            .iter()
                            .map(|p| p.name.as_str())
                            .collect::<Vec<_>>(),
                        ["s", "pattern"]
                    );
                }
                _ => {
                    assert_eq!(py.kind, i32::from(proto::PythonCallKind::Method));
                    assert_eq!(py.receiver, Some(0));
                    assert_eq!(rs.receiver.as_ref().unwrap().core_input, Some(0));
                    assert_eq!(
                        py.owner.as_ref().unwrap().kind,
                        Some(proto::binding_owner::Kind::Sort(signature.inputs[0].sort))
                    );
                    if matches!(suffix, "i64.gt" | "i64.le") {
                        assert!(rs.receiver.as_ref().unwrap().borrowed);
                    }
                    if suffix == "f64.pow" {
                        assert!(rs.trait_impl.is_none());
                    }
                }
            }
        } else {
            assert!(
                bindings.python.is_none() && bindings.rust.is_none(),
                "{key}"
            );
        }
        let sorts: Vec<_> = names
            .iter()
            .map(|name| graph.get_sort_by_name(name).unwrap().clone())
            .collect();
        for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
            let alias = ResolvedCall::from_resolution(
                alias,
                &sorts,
                graph.type_info(),
                context,
                &ast::Span::Panic,
            )
            .unwrap();
            let keyed = ResolvedCall::from_resolution(
                &key,
                &sorts,
                graph.type_info(),
                context,
                &ast::Span::Panic,
            )
            .unwrap();
            assert_eq!(alias, keyed, "{key}");
            let ResolvedCall::Primitive(primitive) = keyed else {
                unreachable!()
            };
            assert!(primitive.validator().is_some(), "{key}");
        }
        for mutation in ["input", "output", "arity"] {
            let mut wrong = primitive.clone();
            let Some(proto::host_primitive::Typing::Signature(s)) = &mut wrong.typing else {
                unreachable!()
            };
            let index = if mutation == "output" {
                s.output.unwrap()
            } else {
                s.inputs[0].sort
            };
            let other = catalog.sorts.iter().position(|sort| matches!(&sort.kind,
                Some(proto::sort::Kind::Family(f)) if f.args.is_empty() && f.name != names[if mutation == "output" {names.len()-1} else {0}])).unwrap() as u32;
            assert_ne!(other, index);
            match mutation {
                "input" => s.inputs[0].sort = other,
                "output" => s.output = Some(other),
                _ => {
                    s.inputs.pop();
                }
            }
            assert!(
                graph
                    .type_info()
                    .check_builtin_signature(&catalog, &wrong)
                    .is_err(),
                "{key}: {mutation}"
            );
        }
    }
}

#[test]
fn remaining_scalar_generic_catalog_preserves_homogeneous_providers() {
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(
            None,
            "(datatype Item (A) (B)) (sort Items (Vec Item)) (sort Ints (Vec i64))",
        )
        .unwrap();
    let catalog = graph.type_info().builtin_catalog().unwrap().definitions;
    for (key, alias, output, validator) in [
        ("egglog.core.bool_ne", "bool-!=", "bool", true),
        ("egglog.core.value_eq", "value-eq", "Unit", false),
    ] {
        let declaration = catalog
            .declarations
            .iter()
            .find(|d| {
                matches!(&d.kind,
            Some(proto::declaration::Kind::HostPrimitive(p)) if p.name == key)
            })
            .unwrap_or_else(|| panic!("missing {key}"));
        let Some(proto::declaration::Kind::HostPrimitive(primitive)) = &declaration.kind else {
            unreachable!()
        };
        let Some(proto::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
            unreachable!()
        };
        assert_eq!(signature.type_params.len(), 1);
        assert_eq!(signature.inputs.len(), 2);
        assert!(signature.varargs.is_empty());
        assert_eq!(signature.inputs[0].sort, signature.inputs[1].sort);
        assert_eq!(
            catalog.sorts[signature.inputs[0].sort as usize].kind,
            Some(proto::sort::Kind::Var(0))
        );
        assert!(
            matches!(&catalog.sorts[signature.output.unwrap() as usize].kind, Some(proto::sort::Kind::Family(f)) if f.name == output && f.args.is_empty())
        );
        let bindings = declaration.bindings.as_ref().unwrap();
        assert_eq!(bindings.egglog.as_ref().unwrap().views[0].symbol, alias);
        assert!(bindings.python.is_none() && bindings.rust.is_none());
        for name in [
            "i64", "bool", "f64", "String", "BigInt", "BigRat", "Unit", "Item", "Items", "Ints",
        ] {
            let sorts =
                [name, name, output].map(|name| graph.get_sort_by_name(name).unwrap().clone());
            for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
                let source = ResolvedCall::from_resolution(
                    alias,
                    &sorts,
                    graph.type_info(),
                    context,
                    &ast::Span::Panic,
                )
                .unwrap();
                let canonical = ResolvedCall::from_resolution(
                    key,
                    &sorts,
                    graph.type_info(),
                    context,
                    &ast::Span::Panic,
                )
                .unwrap();
                assert_eq!(source, canonical);
                let ResolvedCall::Primitive(primitive) = canonical else {
                    unreachable!()
                };
                assert_eq!(primitive.validator().is_some(), validator);
            }
        }
        for expression in [format!("({alias} 1)"), format!("({alias} 1 2 3)")] {
            let error = graph
                .parse_and_run_program(None, &expression)
                .unwrap_err()
                .to_string();
            assert!(error.contains(alias), "{error}");
            assert!(!error.contains(key), "{error}");
        }
        let error = graph
            .parse_and_run_program(None, &format!("({alias} 1 true)"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("bool") && error.contains("i64"), "{error}");
        assert!(!error.contains(key), "{error}");
    }
}

#[test]
fn remaining_scalar_native_boundaries() {
    let mut graph = EGraph::default();
    graph.parse_and_run_program(None, r#"
        (check (= (| 2 4) 6) (= (^ 6 2) 4) (= (<< 1 63) -9223372036854775808) (= (>> -8 2) -2))
        (check (= (not-i64 0) -1) (= (log2 8) 3) (= (abs -7) 7) (> 3 2) (<= 2 2))
        (check (= (bool-= 1 1) true) (= (bool-= 1 2) false))
        (check (= (bool-< 1 2) true) (= (bool-< 2 2) false) (= (bool-< 3 2) false))
        (check (= (bool-> 3 2) true) (= (bool-> 2 2) false) (= (bool-> 1 2) false))
        (check (= (bool-<= 1 2) true) (= (bool-<= 2 2) true) (= (bool-<= 3 2) false))
        (check (= (bool->= 3 2) true) (= (bool->= 2 2) true) (= (bool->= 1 2) false))
        (check (= (bool-< -9223372036854775808 9223372036854775807) true))
        (check (= (count-matches "ééé" "éé") 1) (= (count-matches "é" "") 2) (= (count-matches "" "") 1))
        (check (= (- 3.0 2.0) 1.0) (= (* 3.0 2.0) 6.0) (= (/ 3.0 2.0) 1.5) (= (% 3.0 2.0) 1.0))
        (check (= (^ 4.0 0.5) 2.0) (= (neg 2.0) -2.0) (< 1.0 2.0) (> 2.0 1.0) (<= 1.0 1.0) (>= 1.0 1.0))
        (check (= (min 1.0 2.0) 1.0) (= (max 1.0 2.0) 2.0) (= (abs -2.0) 2.0) (= (exp 0.0) 1.0) (= (log 1.0) 0.0) (= (sqrt 4.0) 2.0))
        (check (= (to-f64 2) 2.0) (= (to-i64 -2.9) -2) (= (to-string 2.0) "2.0") (guard true))
        (check (= (bool-!= 1 2) true) (= (bool-!= true true) false) (value-eq "a" "a"))
        (datatype Item (A) (B)) (sort Items (Vec Item)) (A) (B)
        (check (= (bool-!= (A) (B)) true) (= (bool-!= (vec-of (A)) (vec-of (B))) true))
        (fail (check (value-eq (A) (B))))
        (union (A) (B))
        (check (= (bool-!= (A) (B)) false) (= (bool-!= (vec-of (A)) (vec-of (B))) false) (value-eq (A) (B)) (value-eq (vec-of (A)) (vec-of (B))))
    "#).unwrap();
    for expression in [
        "(<< 1 -1)",
        "(<< 1 64)",
        "(>> 1 -1)",
        "(>> 1 64)",
        "(log2 0)",
        "(log2 -1)",
        "(abs -9223372036854775808)",
        "(/ 1.0 0.0)",
        "(/ 1.0 -0.0)",
        "(% 1.0 0.0)",
        "(% 1.0 -0.0)",
        "(log 0.0)",
        "(sqrt -1.0)",
        "(guard false)",
    ] {
        assert!(
            EGraph::default()
                .parse_and_run_program(None, &format!("(let $invalid {expression})"))
                .is_err(),
            "{expression}"
        );
    }
    let error = EGraph::new_with_proofs()
        .parse_and_run_program(None, "(check (value-eq 1 1))")
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("primitive operation lacks a validator function"),
        "{error}"
    );
}

#[test]
fn inequality_native_baseline() {
    let text = r#"
        (check (!= 1 2) (!= true false))
        (fail (check (!= 1 1)))
        (fail (check (!= false false)))
        (datatype Item (A) (B))
        (sort Items (Vec Item))
        (sort Ints (Vec i64))
        (A) (B)
        (check (!= (A) (B)) (!= (vec-of (A)) (vec-of (B))) (!= (vec-of 1) (vec-of 2)))
        (fail (check (!= (vec-of (A)) (vec-of (A)))))
        (union (A) (B))
        (check (= (A) (B)) (= (vec-of (A)) (vec-of (B))))
        (fail (check (!= (A) (B))))
        (fail (check (!= (vec-of (A)) (vec-of (B)))))
    "#;
    for mode in 0..3 {
        let mut graph = match mode {
            0 => EGraph::default(),
            1 => EGraph::new_with_proofs(),
            _ => EGraph::new_with_proofs().with_proof_testing(),
        };
        graph
            .parse_and_run_program(None, text)
            .unwrap_or_else(|error| panic!("mode {mode}: {error}"));
        if mode > 0 {
            graph
                .parse_and_run_program(None, "(datatype Witness (Wrap)) (rule ((!= 1 2)) ((Wrap))) (run 1) (prove-exists Wrap)")
                .unwrap();
        }
    }
}

#[test]
fn inequality_catalog_uses_one_generic_registration() {
    let mut graph = EGraph::default();
    let catalog = graph.type_info().builtin_catalog().unwrap();
    let declaration = catalog.definitions.declarations.iter().find(|d| {
        matches!(&d.kind, Some(proto::declaration::Kind::HostPrimitive(p)) if p.name == "egglog.core.ne")
    }).expect("native inequality is missing its canonical descriptor");
    let Some(proto::declaration::Kind::HostPrimitive(primitive)) = &declaration.kind else {
        unreachable!()
    };
    let Some(proto::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
        unreachable!()
    };
    assert_eq!(signature.type_params.len(), 1);
    assert_eq!(signature.inputs.len(), 2);
    assert!(signature.varargs.is_empty());
    assert_eq!(signature.inputs[0].sort, signature.inputs[1].sort);
    assert_eq!(
        catalog.definitions.sorts[signature.inputs[0].sort as usize].kind,
        Some(proto::sort::Kind::Var(0))
    );
    assert!(
        matches!(&catalog.definitions.sorts[signature.output.unwrap() as usize].kind,
        Some(proto::sort::Kind::Family(f)) if f.name == "Unit" && f.args.is_empty())
    );
    let bindings = declaration.bindings.as_ref().unwrap();
    assert_eq!(bindings.egglog.as_ref().unwrap().views[0].symbol, "!=");
    assert!(bindings.python.is_none());
    assert_eq!(
        bindings
            .rust
            .as_ref()
            .expect("inequality needs a Rust view")
            .views,
        vec![proto::RustCallable {
            path: ["egglog_experimental", "typed", "builtins", "ne"]
                .map(str::to_owned)
                .into(),
            params: signature
                .inputs
                .iter()
                .enumerate()
                .map(|(index, input)| {
                    proto::RustParameter {
                        name: input.name.clone(),
                        core_input: Some(index as u32),
                        borrowed: false,
                    }
                })
                .collect(),
            ..Default::default()
        }]
    );
    assert!(
        !catalog
            .undescribed_primitives
            .iter()
            .any(|name| name == "!=")
    );
    graph.parse_and_run_program(None, "(datatype Item (A) (B)) (sort Items (Vec Item)) (sort Ints (Vec i64)) (sort Pairs (Pair Item i64)) (sort Sets (Set Item)) (sort Maps (Map i64 Item))").unwrap();
    for name in [
        "i64", "bool", "f64", "String", "BigInt", "BigRat", "Unit", "Item", "Items", "Ints",
        "Pairs", "Sets", "Maps",
    ] {
        let sort = graph.get_sort_by_name(name).unwrap().clone();
        let sorts = [
            sort.clone(),
            sort,
            graph.get_sort_by_name("Unit").unwrap().clone(),
        ];
        for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
            let alias = ResolvedCall::from_resolution(
                "!=",
                &sorts,
                graph.type_info(),
                context,
                &ast::Span::Panic,
            )
            .unwrap();
            let canonical = ResolvedCall::from_resolution(
                "egglog.core.ne",
                &sorts,
                graph.type_info(),
                context,
                &ast::Span::Panic,
            )
            .unwrap();
            assert_eq!(alias, canonical);
            let ResolvedCall::Primitive(primitive) = canonical else {
                unreachable!()
            };
            assert!(primitive.validator().is_some());
        }
        let mut program = proto::Program {
            ir_version: 1,
            ..Default::default()
        };
        let input = graph.export_sort(&sorts[0], &mut program.sorts).unwrap();
        let output = graph.export_sort(&sorts[2], &mut program.sorts).unwrap();
        assert_eq!(
            graph
                .type_info()
                .resolve_builtin(&program, "egglog.core.ne", &[input, input], output, &sorts)
                .unwrap(),
            "egglog.core.ne"
        );
    }
    for text in ["(!= 1)", "(!= 1 2 3)"] {
        let error = graph
            .parse_and_run_program(None, text)
            .unwrap_err()
            .to_string();
        assert!(error.contains("!="), "{error}");
        assert!(!error.contains("egglog.core.ne"), "{error}");
    }
    let error = graph
        .parse_and_run_program(None, "(!= 1 true)")
        .unwrap_err()
        .to_string();
    assert!(error.contains("bool") && error.contains("i64"), "{error}");
}

#[test]
fn inequality_catalog_export_does_not_execute_generic_provider() {
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let mut graph = EGraph::default();
    add_primitive_with_validator!(&mut graph, "counted-generic" [id = "test.counted-generic"] = |a: #, b: #| -?> () {
        { CALLS.fetch_add(1, Ordering::SeqCst); (a != b).then_some(()) }
    }, |_: &mut TermDag, _: &[TermId]| { CALLS.fetch_add(1, Ordering::SeqCst); None });
    let before = graph.type_info().builtin_catalog().unwrap().definitions;
    let after = graph.type_info().builtin_catalog().unwrap().definitions;
    assert_eq!(before, after);
    assert!(before.declarations.iter().any(|d| matches!(&d.kind,
        Some(proto::declaration::Kind::HostPrimitive(p)) if p.name == "test.counted-generic")));
    assert_eq!(CALLS.load(Ordering::SeqCst), 0);
}

#[test]
fn bigrat_properties_export_actual_signatures_and_views() {
    let mut graph = EGraph::default();
    let program = graph.type_info().builtin_catalog().unwrap().definitions;
    for name in ["numer", "denom"] {
        let key = format!("egglog.core.bigrat.{name}");
        let declaration = program
            .declarations
            .iter()
            .find(|d| matches!(&d.kind, Some(proto::declaration::Kind::HostPrimitive(p)) if p.name == key))
            .unwrap_or_else(|| panic!("missing native catalog definition {key}"));
        let Some(proto::declaration::Kind::HostPrimitive(primitive)) = &declaration.kind else {
            unreachable!()
        };
        assert_eq!(
            builtin::signature_sorts(&program, primitive).unwrap(),
            ["BigRat", "BigInt"]
        );
        let Some(proto::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
            unreachable!()
        };
        let bindings = declaration.bindings.as_ref().unwrap();
        let python = &bindings.python.as_ref().unwrap().views[0];
        let rust = &bindings.rust.as_ref().unwrap().views[0];
        assert_eq!(python.path, [name]);
        assert_eq!(python.kind, i32::from(proto::PythonCallKind::Property));
        assert_eq!(python.receiver, Some(0));
        assert!(python.params.is_empty());
        assert_eq!(
            python.owner.as_ref().unwrap().kind,
            Some(proto::binding_owner::Kind::Sort(signature.inputs[0].sort))
        );
        assert_eq!(rust.path, [name]);
        assert_eq!(rust.owner, python.owner);
        assert_eq!(rust.receiver.as_ref().unwrap().core_input, Some(0));
        assert!(rust.params.is_empty());
        let sorts = ["BigRat", "BigInt"].map(|s| graph.get_sort_by_name(s).unwrap().clone());
        for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
            let ResolvedCall::Primitive(alias) = ResolvedCall::from_resolution(
                name,
                &sorts,
                graph.type_info(),
                context,
                &ast::Span::Panic,
            )
            .unwrap() else {
                unreachable!()
            };
            let ResolvedCall::Primitive(canonical) = ResolvedCall::from_resolution(
                &key,
                &sorts,
                graph.type_info(),
                context,
                &ast::Span::Panic,
            )
            .unwrap() else {
                unreachable!()
            };
            assert_eq!(alias, canonical);
            assert!(canonical.validator().is_none());
        }
    }
}

#[test]
fn numeric_catalog_members_preserve_native_providers_and_typed_views() {
    let mut graph = EGraph::default();
    let program = graph.type_info().builtin_catalog().unwrap().definitions;
    let mut count = 0;
    for declaration in &program.declarations {
        let Some(proto::declaration::Kind::HostPrimitive(primitive)) = &declaration.kind else {
            continue;
        };
        let owner = if primitive.name.starts_with("egglog.core.bigint.") {
            "BigInt"
        } else if primitive.name.starts_with("egglog.core.bigrat.") {
            "BigRat"
        } else {
            continue;
        };
        let member = primitive.name.rsplit('.').next().unwrap();
        if matches!(member, "from_i64" | "from_string" | "make") {
            continue; // Constructor contracts are checked separately.
        }
        count += 1;
        let Some(proto::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
            unreachable!()
        };
        assert!(signature.type_params.is_empty() && signature.varargs.is_empty());
        let sort_names = builtin::signature_sorts(&program, primitive).unwrap();
        assert_eq!(sort_names[0], owner);
        match member {
            "numer" | "denom" => assert_eq!(sort_names, ["BigRat", "BigInt"]),
            "shl" | "shr" => assert_eq!(sort_names, ["BigInt", "i64", "BigInt"]),
            "lt" | "gt" | "le" | "ge" => assert_eq!(sort_names, [owner, owner, "Unit"]),
            "bool_eq" | "bool_lt" | "bool_gt" | "bool_le" | "bool_ge" => {
                assert_eq!(sort_names, ["BigInt", "BigInt", "bool"]);
            }
            "to_string" => assert_eq!(sort_names, ["BigInt", "String"]),
            "to_i64" => assert_eq!(sort_names, ["BigRat", "i64"]),
            "to_f64" => assert_eq!(sort_names, ["BigRat", "f64"]),
            _ => assert!(sort_names.iter().all(|name| name == owner)),
        }
        let bindings = declaration.bindings.as_ref().unwrap();
        let alias = &bindings.egglog.as_ref().unwrap().views[0].symbol;
        let python = &bindings.python.as_ref().unwrap().views[0];
        let rust = &bindings.rust.as_ref().unwrap().views[0];
        assert_eq!(python.receiver, Some(0));
        assert_eq!(
            python.owner.as_ref().unwrap().kind,
            Some(proto::binding_owner::Kind::Sort(signature.inputs[0].sort))
        );
        assert_eq!(rust.owner, python.owner);
        assert_eq!(rust.path, [member]);
        assert_eq!(rust.receiver.as_ref().unwrap().core_input, Some(0));
        let expected_slots = (1..signature.inputs.len() as u32).collect::<Vec<_>>();
        assert_eq!(
            python
                .params
                .iter()
                .map(|p| p.core_input.unwrap())
                .collect::<Vec<_>>(),
            expected_slots
        );
        assert_eq!(
            rust.params
                .iter()
                .map(|p| p.core_input.unwrap())
                .collect::<Vec<_>>(),
            expected_slots
        );
        let expected_trait = match member {
            "add" => Some("Add"),
            "sub" => Some("Sub"),
            "mul" => Some("Mul"),
            "div" => Some("Div"),
            "rem" => Some("Rem"),
            "bitand" => Some("BitAnd"),
            "bitor" => Some("BitOr"),
            "bitxor" => Some("BitXor"),
            "shl" => Some("Shl"),
            "shr" => Some("Shr"),
            "not" => Some("Not"),
            "neg" => Some("Neg"),
            _ => None,
        };
        assert_eq!(
            rust.trait_impl.as_ref().map(|t| t.path.join("::")),
            expected_trait.map(|name| format!("core::ops::{name}"))
        );
        if let Some(trait_impl) = &rust.trait_impl {
            assert_eq!(
                trait_impl
                    .args
                    .iter()
                    .map(|a| a.sort.unwrap())
                    .collect::<Vec<_>>(),
                signature.inputs[1..]
                    .iter()
                    .map(|p| p.sort)
                    .collect::<Vec<_>>()
            );
        }
        let native_sorts = sort_names
            .iter()
            .map(|name| graph.get_sort_by_name(name).unwrap().clone())
            .collect::<Vec<_>>();
        for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
            let ResolvedCall::Primitive(source) = ResolvedCall::from_resolution(
                alias,
                &native_sorts,
                graph.type_info(),
                context,
                &ast::Span::Panic,
            )
            .unwrap() else {
                unreachable!()
            };
            let ResolvedCall::Primitive(canonical) = ResolvedCall::from_resolution(
                &primitive.name,
                &native_sorts,
                graph.type_info(),
                context,
                &ast::Span::Panic,
            )
            .unwrap() else {
                unreachable!()
            };
            assert_eq!(source, canonical);
            assert!(source.validator().is_none() && canonical.validator().is_none());
        }
    }
    assert_eq!(count, 47);
}

#[test]
fn closed_i64_catalog_preserves_signatures_aliases_and_borrowed_views() {
    let mut graph = EGraph::default();
    let program = graph.type_info().builtin_catalog().unwrap().definitions;
    let float = program
        .sorts
        .iter()
        .position(|s| matches!(&s.kind, Some(proto::sort::Kind::Family(f)) if f.name == "f64"))
        .unwrap() as u32;
    for (key, alias, python, method, trait_name, output) in [
        ("sub", "-", "__sub__", "sub", Some("Sub"), "i64"),
        ("mul", "*", "__mul__", "mul", Some("Mul"), "i64"),
        ("div", "/", "__truediv__", "div", Some("Div"), "i64"),
        ("rem", "%", "__mod__", "rem", Some("Rem"), "i64"),
        ("bitand", "&", "__and__", "bitand", Some("BitAnd"), "i64"),
        ("lt", "<", "__lt__", "lt", None, "Unit"),
        ("ge", ">=", "__ge__", "ge", None, "Unit"),
        ("min", "min", "min", "min", None, "i64"),
        ("max", "max", "max", "max", None, "i64"),
    ] {
        let key = format!("egglog.core.i64.{key}");
        let definitions: Vec<_> = program
            .declarations
            .iter()
            .filter(|d| {
                matches!(
                    &d.kind, Some(proto::declaration::Kind::HostPrimitive(p)) if p.name == key
                )
            })
            .collect();
        assert_eq!(definitions.len(), 1, "{key}");
        let declaration = definitions[0];
        let Some(proto::declaration::Kind::HostPrimitive(primitive)) = &declaration.kind else {
            unreachable!()
        };
        assert_eq!(
            builtin::signature_sorts(&program, primitive).unwrap(),
            ["i64", "i64", output]
        );
        let Some(proto::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
            unreachable!()
        };
        assert!(signature.type_params.is_empty());
        assert!(signature.varargs.is_empty());
        let bindings = declaration.bindings.as_ref().unwrap();
        assert_eq!(bindings.egglog.as_ref().unwrap().views[0].symbol, alias);
        assert_eq!(bindings.python.as_ref().unwrap().views[0].path, [python]);
        let rust = &bindings.rust.as_ref().unwrap().views[0];
        assert_eq!(rust.path, [method]);
        assert_eq!(rust.receiver.as_ref().unwrap().core_input, Some(0));
        assert_eq!(
            rust.receiver.as_ref().unwrap().borrowed,
            trait_name.is_none()
        );
        assert_eq!(rust.params.len(), 1);
        assert_eq!(rust.params[0].core_input, Some(1));
        assert!(!rust.params[0].borrowed);
        assert_eq!(
            rust.trait_impl.as_ref().map(|t| t.path.join("::")),
            trait_name.map(|t| format!("core::ops::{t}"))
        );
        let sorts = ["i64", "i64", output].map(|s| graph.get_sort_by_name(s).unwrap().clone());
        for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
            let ResolvedCall::Primitive(source) = ResolvedCall::from_resolution(
                alias,
                &sorts,
                graph.type_info(),
                context,
                &ast::Span::Panic,
            )
            .unwrap() else {
                unreachable!()
            };
            let ResolvedCall::Primitive(canonical) = ResolvedCall::from_resolution(
                &key,
                &sorts,
                graph.type_info(),
                context,
                &ast::Span::Panic,
            )
            .unwrap() else {
                unreachable!()
            };
            assert_eq!(source, canonical);
            assert!(canonical.validator().is_some());
        }
        for corrupt_output in [false, true] {
            let mut wrong = primitive.clone();
            let Some(proto::host_primitive::Typing::Signature(signature)) = &mut wrong.typing
            else {
                unreachable!()
            };
            if corrupt_output {
                signature.output = Some(float);
            } else {
                signature.inputs[0].sort = float;
            }
            assert!(
                graph
                    .type_info()
                    .check_builtin_signature(&program, &wrong)
                    .is_err()
            );
        }
        let mut unknown = primitive.clone();
        unknown.name = format!("test.unregistered.{key}");
        assert!(
            graph
                .type_info()
                .check_builtin_signature(&program, &unknown)
                .is_err()
        );
    }
}

#[test]
fn closed_i64_native_partiality_and_signed_extrema() {
    let mut graph = EGraph::default();
    for (expression, expected) in [
        ("(- -13 5)", -18),
        ("(* -13 5)", -65),
        ("(/ -13 5)", -2),
        ("(% -13 5)", -3),
        ("(& -13 5)", 1),
        ("(min -9223372036854775808 9223372036854775807)", i64::MIN),
        ("(max -9223372036854775808 9223372036854775807)", i64::MAX),
    ] {
        graph
            .parse_and_run_program(None, &format!("(check (= {expression} {expected}))"))
            .unwrap();
    }
    for (fact, matches) in [
        ("(< -13 5)", true),
        ("(< 5 -13)", false),
        ("(>= -13 5)", false),
        ("(>= 5 -13)", true),
        ("(< 5 5)", false),
        ("(>= 5 5)", true),
    ] {
        assert_eq!(
            graph
                .parse_and_run_program(None, &format!("(check {fact})"))
                .is_ok(),
            matches,
            "{fact}"
        );
    }
    for expression in [
        "(/ 1 0)",
        "(% 1 0)",
        "(/ -9223372036854775808 -1)",
        "(% -9223372036854775808 -1)",
        "(- -9223372036854775808 1)",
        "(* 9223372036854775807 2)",
    ] {
        assert!(
            EGraph::default()
                .parse_and_run_program(None, &format!("(let invalid {expression})"))
                .is_err(),
            "{expression}"
        );
    }
}

#[test]
fn set_catalog_precedes_instances_and_preserves_native_aliases() {
    let mut graph = EGraph::default();
    let before = graph.type_info().builtin_catalog().unwrap();
    assert!(!before.undescribed_families.iter().any(|f| f == "Set"));
    assert!(
        !before
            .undescribed_family_primitives
            .iter()
            .any(|f| f.starts_with("Set::"))
    );
    let family = before
        .definitions
        .declarations
        .iter()
        .find_map(|d| match &d.kind {
            Some(proto::declaration::Kind::HostSortFamily(f)) if f.name == "Set" => Some(f),
            _ => None,
        })
        .unwrap();
    assert_eq!(family.arity, 1);
    let bindings = family.bindings.as_ref().unwrap();
    assert_eq!(
        bindings.python.as_ref().unwrap().path,
        ["egglog", "builtins", "Set"]
    );
    assert_eq!(
        bindings.rust.as_ref().unwrap().path,
        ["egglog_experimental", "typed", "builtins", "Set"]
    );
    assert_eq!(before.definitions.declarations.iter().filter(|d| matches!(&d.kind, Some(proto::declaration::Kind::HostPrimitive(p)) if p.name.starts_with("egglog.core.set."))).count(), 11);
    graph.parse_and_run_program(None, "(sort SI (Set i64)) (sort Other (Set i64)) (sort Math) (constructor A () Math) (sort SM (Set Math))").unwrap();
    for (element, set) in [("i64", "SI"), ("Math", "SM")] {
        for (key, alias, inputs, output) in [
            ("empty", "set-empty", &[][..], set),
            ("of", "set-of", &[element, element][..], set),
            ("get", "set-get", &[set, "i64"][..], element),
            ("insert", "set-insert", &[set, element][..], set),
            ("remove", "set-remove", &[set, element][..], set),
            ("length", "set-length", &[set][..], "i64"),
            ("contains", "set-contains", &[set, element][..], "Unit"),
            (
                "not_contains",
                "set-not-contains",
                &[set, element][..],
                "Unit",
            ),
            ("union", "set-union", &[set, set][..], set),
            ("diff", "set-diff", &[set, set][..], set),
            ("intersect", "set-intersect", &[set, set][..], set),
        ] {
            let key = format!("egglog.core.set.{key}");
            let sorts = inputs
                .iter()
                .copied()
                .chain([output])
                .map(|name| graph.get_sort_by_name(name).unwrap().clone())
                .collect::<Vec<_>>();
            let d = before.definitions.declarations.iter().find(|d| matches!(&d.kind, Some(proto::declaration::Kind::HostPrimitive(p)) if p.name == key)).unwrap();
            let Some(proto::declaration::Kind::HostPrimitive(p)) = &d.kind else {
                unreachable!()
            };
            let Some(proto::host_primitive::Typing::Signature(s)) = &p.typing else {
                unreachable!()
            };
            assert_eq!(s.type_params, ["T"]);
            assert_eq!(s.varargs.len(), usize::from(alias == "set-of"));
            assert_eq!(
                s.inputs.len(),
                if alias == "set-of" { 0 } else { inputs.len() }
            );
            let bindings = d.bindings.as_ref().unwrap();
            assert_eq!(bindings.egglog.as_ref().unwrap().views[0].symbol, alias);
            let rs = &bindings.rust.as_ref().unwrap().views[0];
            assert_eq!(rs.path, [key.strip_prefix("egglog.core.set.").unwrap()]);
            assert_eq!(bindings.python.is_some(), alias != "set-get");
            if let Some(python) = &bindings.python {
                let py = &python.views[0];
                assert_eq!(py.owner, rs.owner);
                assert_eq!(
                    py.receiver,
                    (!matches!(alias, "set-of" | "set-empty")).then_some(0)
                );
            }
            for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
                let ResolvedCall::Primitive(a) = ResolvedCall::from_resolution(
                    alias,
                    &sorts,
                    graph.type_info(),
                    context,
                    &ast::Span::Panic,
                )
                .unwrap() else {
                    unreachable!()
                };
                let ResolvedCall::Primitive(b) = ResolvedCall::from_resolution(
                    &key,
                    &sorts,
                    graph.type_info(),
                    context,
                    &ast::Span::Panic,
                )
                .unwrap() else {
                    unreachable!()
                };
                assert_eq!(a, b);
                assert_eq!(a.validator().is_some(), alias != "set-get");
                let mut exported = proto::Program::default();
                assert_eq!(a.export_builtin(&mut exported).unwrap(), key);
            }
        }
    }
    assert_eq!(
        graph.type_info().builtin_catalog().unwrap().definitions,
        before.definitions
    );
    graph.parse_and_run_program(None, "(function first () SI :merge new) (function second () Other :merge new) (set (first) (set-empty)) (set (second) (set-of)) (check (= (set-length (first)) 0)) (check (= (set-length (second)) 0))").unwrap();
    for invalid in [
        "(set-empty)",
        "(set-union (first) (second))",
        "(set (first) (set-of 1.0))",
        "(set (first) (set-empty 1))",
        "(set-get (first))",
    ] {
        assert!(
            graph.parse_and_run_program(None, invalid).is_err(),
            "{invalid}"
        );
    }
}

#[test]
fn exact_integer_families_and_constructors_use_actual_native_providers() {
    let mut graph = EGraph::default();
    assert!(graph.get_sort_by_name("Rational").is_none());
    let catalog = graph.type_info().builtin_catalog().unwrap();
    assert!(!catalog.definitions.declarations.iter().any(|d| matches!(&d.kind, Some(proto::declaration::Kind::HostPrimitive(p)) if p.name.starts_with("egglog.experimental.rational."))));
    for name in ["BigInt", "BigRat"] {
        let family = catalog
            .definitions
            .declarations
            .iter()
            .find_map(|d| match &d.kind {
                Some(proto::declaration::Kind::HostSortFamily(f)) if f.name == name => Some(f),
                _ => None,
            })
            .unwrap();
        assert_eq!(family.arity, 0);
        let bindings = family.bindings.as_ref().unwrap();
        assert_eq!(
            bindings.python.as_ref().unwrap().path,
            ["egglog", "builtins", name]
        );
        assert_eq!(
            bindings.rust.as_ref().unwrap().path,
            ["egglog_experimental", "typed", "builtins", name]
        );
    }
    for (key, alias, inputs, output, python_kind, python_path, rust_path) in [
        (
            "egglog.core.bigint.from_i64",
            "bigint",
            &["i64"][..],
            "BigInt",
            proto::PythonCallKind::Initializer,
            &[][..],
            "new",
        ),
        (
            "egglog.core.bigint.from_string",
            "from-string",
            &["String"][..],
            "BigInt",
            proto::PythonCallKind::ClassMethod,
            &["from_string"][..],
            "from_string",
        ),
        (
            "egglog.core.bigrat.make",
            "bigrat",
            &["BigInt", "BigInt"][..],
            "BigRat",
            proto::PythonCallKind::Initializer,
            &[][..],
            "new",
        ),
    ] {
        let d = catalog.definitions.declarations.iter().find(|d| matches!(&d.kind, Some(proto::declaration::Kind::HostPrimitive(p)) if p.name == key)).unwrap();
        let Some(proto::declaration::Kind::HostPrimitive(p)) = &d.kind else {
            unreachable!()
        };
        let Some(proto::host_primitive::Typing::Signature(s)) = &p.typing else {
            unreachable!()
        };
        assert!(s.type_params.is_empty() && s.varargs.is_empty());
        assert_eq!(s.inputs.len(), inputs.len());
        for (sort, name) in s
            .inputs
            .iter()
            .map(|arg| arg.sort)
            .chain(s.output)
            .zip(inputs.iter().copied().chain([output]))
        {
            assert!(
                matches!(&catalog.definitions.sorts[sort as usize].kind, Some(proto::sort::Kind::Family(f)) if f.name == name && f.args.is_empty())
            );
        }
        let b = d.bindings.as_ref().unwrap();
        assert_eq!(b.egglog.as_ref().unwrap().views[0].symbol, alias);
        let py = &b.python.as_ref().unwrap().views[0];
        let rs = &b.rust.as_ref().unwrap().views[0];
        assert_eq!(py.kind, i32::from(python_kind));
        assert_eq!(py.path, python_path);
        assert_eq!(rs.path, [rust_path]);
        assert_eq!(
            py.owner.as_ref().unwrap().kind,
            Some(proto::binding_owner::Kind::Sort(s.output.unwrap()))
        );
        assert_eq!(rs.owner, py.owner);
        assert_eq!(py.receiver, None);
        assert_eq!(rs.receiver, None);
        assert_eq!(
            py.params
                .iter()
                .map(|p| p.core_input.unwrap())
                .collect::<Vec<_>>(),
            (0..inputs.len() as u32).collect::<Vec<_>>()
        );
        assert_eq!(
            rs.params
                .iter()
                .map(|p| p.core_input.unwrap())
                .collect::<Vec<_>>(),
            (0..inputs.len() as u32).collect::<Vec<_>>()
        );
        let sorts = inputs
            .iter()
            .chain([&output])
            .map(|name| graph.get_sort_by_name(name).unwrap().clone())
            .collect::<Vec<_>>();
        for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
            let ResolvedCall::Primitive(alias) = ResolvedCall::from_resolution(
                alias,
                &sorts,
                graph.type_info(),
                context,
                &ast::Span::Panic,
            )
            .unwrap() else {
                unreachable!()
            };
            let ResolvedCall::Primitive(canonical) = ResolvedCall::from_resolution(
                key,
                &sorts,
                graph.type_info(),
                context,
                &ast::Span::Panic,
            )
            .unwrap() else {
                unreachable!()
            };
            assert_eq!(alias, canonical); // same concrete sorts and native context IDs
            assert!(alias.validator().is_none() && canonical.validator().is_none());
        }
    }
    graph.parse_and_run_program(None, "(check (= (bigint 7) (from-string \"7\"))) (check (= (bigrat (bigint 2) (bigint -4)) (bigrat (bigint -1) (bigint 2)))) (fail (from-string \"not-an-integer\")) (fail (bigrat (bigint 1) (bigint 0)))").unwrap();
}

#[test]
fn annotated_builtins_resolve_in_the_nonexecuting_authoring_context() {
    use ast::{Action, Command, Expr, Literal};
    for mode in 0..4 {
        let mut graph = match mode {
            0 => EGraph::default(),
            1 => EGraph::new_with_term_encoding(),
            2 => EGraph::new_with_proofs(),
            _ => EGraph::new_with_proofs().with_proof_testing(),
        };
        let declarations = graph
            .parse_program(None, "(sort V (Vec i64)) (relation Values (V))")
            .unwrap();
        for declaration in declarations {
            graph.resolve_command_before_proofs(declaration).unwrap();
        }
        // The logical sort exists, while the encoded runtime has not installed
        // this family instance. Selecting only from runtime TypeInfo cannot work.
        assert_eq!(graph.get_sort_by_name("V").is_none(), mode > 0);
        let program = proto::Program {
            ir_version: 1,
            sorts: vec![
                proto::Sort {
                    kind: Some(proto::sort::Kind::Family(proto::HostSort {
                        name: "i64".into(),
                        args: vec![],
                    })),
                    ..Default::default()
                },
                proto::Sort {
                    kind: Some(proto::sort::Kind::Family(proto::HostSort {
                        name: "Vec".into(),
                        args: vec![0],
                    })),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let add = graph
            .resolve_builtin_call(
                &program,
                "egglog.core.i64.add",
                &[0, 0],
                0,
                &["i64", "i64", "i64"],
            )
            .unwrap();
        let empty = graph
            .resolve_builtin_call(&program, "egglog.core.vec.empty", &[], 1, &["V"])
            .unwrap();
        let span = ast::Span::Panic;
        let plus = Expr::Call(
            span.clone(),
            add,
            vec![
                Expr::Lit(span.clone(), Literal::Int(1)),
                Expr::Lit(span.clone(), Literal::Int(2)),
            ],
        );
        graph
            .resolve_command_before_proofs(Command::Action(Action::Expr(span.clone(), plus)))
            .unwrap();
        let empty = Expr::Call(
            span.clone(),
            "Values".into(),
            vec![Expr::Call(span.clone(), empty, vec![])],
        );
        graph
            .resolve_command_before_proofs(Command::Action(Action::Expr(span, empty)))
            .unwrap();
        assert!(
            graph
                .resolve_builtin_call(&program, "egglog.core.i64.add", &[0], 0, &["i64", "i64"])
                .is_err()
        );
        assert!(
            graph
                .resolve_builtin_call(&program, "egglog.core.vec.empty", &[], 1, &["i64"])
                .is_err()
        );
        assert!(
            graph
                .resolve_builtin_call(&program, "egglog.core.vec.empty", &[], 1, &["missing"])
                .is_err()
        );
    }
}

#[test]
fn flat_map_export_packs_existing_refs_without_native_pair_vec_aliases() {
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(
            None,
            "(datatype Pair (Z)) (sort Vec) (sort Map) (sort M (Map Pair Pair))",
        )
        .unwrap();
    let n = graph.get_sort_by_name("Pair").unwrap().clone();
    let m = graph.get_sort_by_name("M").unwrap().clone();
    for count in [0, 1, 3] {
        let mut sorts = vec![n.clone(); count * 2];
        sorts.push(m.clone());
        let ResolvedCall::Primitive(primitive) = ResolvedCall::from_resolution(
            "map-of",
            &sorts,
            graph.type_info(),
            Context::Pure,
            &ast::Span::Panic,
        )
        .unwrap() else {
            panic!("primitive")
        };
        let mut program = proto::Program {
            ir_version: 1,
            files: vec![proto::SourceFile {
                name: "map.egg".into(),
                contents: None,
            }],
            ..Default::default()
        };
        assert!(primitive.export_builtin(&mut program).is_err());
        let sort_id = graph.export_sort(&n, &mut program.sorts).unwrap();
        for offset in [0, 2] {
            program.nodes.push(proto::Node {
                sort_id,
                kind: Some(proto::node::Kind::Union(proto::Union::default())),
                span: Some(proto::Span {
                    file: 0,
                    start: offset,
                    end: offset + 1,
                }),
            });
        }
        let original = program.nodes.clone();
        let args = [0, 1, 0, 0, 1, 0][..count * 2].to_vec();
        let span = Some(proto::Span {
            file: 0,
            start: 0,
            end: 10,
        });
        let call = primitive
            .export_builtin_call(&graph, &args, span, &mut program)
            .unwrap();
        assert_eq!(call.func, "egglog.core.map.of");
        assert_eq!(call.args.len(), 1);
        assert_eq!(&program.nodes[..2], original);
        assert_eq!(program.nodes.len(), count + 3);
        for (pair, fields) in program.nodes[2..2 + count].iter().zip(args.chunks_exact(2)) {
            let Some(proto::node::Kind::Call(call)) = &pair.kind else {
                panic!("pair call")
            };
            assert_eq!(call.func, "egglog.core.pair.make");
            assert_eq!(call.args, fields);
            assert_eq!(pair.span, span);
        }
        let Some(proto::node::Kind::Call(vector)) =
            &program.nodes[*call.args.first().unwrap() as usize].kind
        else {
            panic!("vector call")
        };
        assert_eq!(
            vector.func,
            if count == 0 {
                "egglog.core.vec.empty"
            } else {
                "egglog.core.vec.of"
            }
        );
        assert_eq!(vector.args, (2..2 + count as u32).collect::<Vec<_>>());
        assert_eq!(program.nodes.last().unwrap().span, span);
    }
}

#[test]
fn map_catalog_precedes_instances_and_uses_one_vec_of_pairs_input() {
    let mut graph = EGraph::default();
    let before = graph.type_info().builtin_catalog().unwrap().definitions;
    let family = before
        .declarations
        .iter()
        .find_map(|d| match &d.kind {
            Some(proto::declaration::Kind::HostSortFamily(f)) if f.name == "Map" => Some(f),
            _ => None,
        })
        .unwrap();
    assert_eq!(family.arity, 2);
    let bindings = family.bindings.as_ref().unwrap();
    assert_eq!(
        bindings.python.as_ref().unwrap(),
        &proto::TypeBinding {
            path: vec!["egglog".into(), "builtins".into(), "Map".into()],
            type_params: vec!["K".into(), "V".into()]
        }
    );
    assert_eq!(
        bindings.rust.as_ref().unwrap(),
        &proto::TypeBinding {
            path: vec![
                "egglog_experimental".into(),
                "typed".into(),
                "builtins".into(),
                "Map".into()
            ],
            type_params: vec!["K".into(), "V".into()]
        }
    );
    for method in ["empty", "of", "get", "insert"] {
        let declaration = before.declarations.iter().find(|d| matches!(&d.kind,
            Some(proto::declaration::Kind::HostPrimitive(p)) if p.name == format!("egglog.core.map.{method}")
        )).expect("canonical Map provider must exist before instances");
        let views = declaration.bindings.as_ref().unwrap();
        let python = &views.python.as_ref().unwrap().views[0];
        let rust = &views.rust.as_ref().unwrap().views[0];
        assert_eq!(python.owner, rust.owner);
        let owner = match python.owner.as_ref().unwrap().kind {
            Some(proto::binding_owner::Kind::Sort(i)) => i,
            _ => panic!("owner"),
        };
        let Some(proto::sort::Kind::Family(owner)) = &before.sorts[owner as usize].kind else {
            panic!("family")
        };
        assert_eq!(owner.name, "Map");
        assert_eq!(
            owner
                .args
                .iter()
                .map(|i| before.sorts[*i as usize].kind.clone())
                .collect::<Vec<_>>(),
            [
                Some(proto::sort::Kind::Var(0)),
                Some(proto::sort::Kind::Var(1))
            ]
        );
        let receiver = matches!(method, "get" | "insert");
        assert_eq!(
            python.kind,
            if receiver {
                proto::PythonCallKind::Method
            } else {
                proto::PythonCallKind::ClassMethod
            } as i32
        );
        assert_eq!(
            python.path,
            [match method {
                "of" => "from_pairs",
                "get" => "__getitem__",
                _ => method,
            }]
        );
        assert_eq!(python.receiver, receiver.then_some(0));
        assert!(python.mutates.is_none());
        assert_eq!(
            rust.path,
            [if method == "of" { "from_pairs" } else { method }]
        );
        assert_eq!(
            rust.receiver,
            receiver.then_some(proto::RustReceiver {
                core_input: Some(0),
                borrowed: true
            })
        );
        assert!(rust.trait_impl.is_none() && !rust.borrowed_self);
        let expected: &[(&str, u32)] = match method {
            "empty" => &[],
            "of" => &[("entries", 0)],
            "get" => &[("key", 1)],
            _ => &[("key", 1), ("value", 2)],
        };
        assert_eq!(
            python
                .params
                .iter()
                .map(|p| (p.name.as_str(), p.core_input.unwrap()))
                .collect::<Vec<_>>(),
            expected
        );
        assert!(python.params.iter().all(|p| p.default_expr.is_none()));
        assert_eq!(
            rust.params
                .iter()
                .map(|p| (p.name.as_str(), p.core_input.unwrap()))
                .collect::<Vec<_>>(),
            expected
        );
        assert!(rust.params.iter().all(|p| !p.borrowed));
        if method == "of" {
            let Some(proto::declaration::Kind::HostPrimitive(p)) = &declaration.kind else {
                unreachable!()
            };
            let Some(proto::host_primitive::Typing::Signature(signature)) = &p.typing else {
                unreachable!()
            };
            assert_eq!(signature.inputs.len(), 1);
            assert!(signature.varargs.is_empty());
            let Some(proto::sort::Kind::Family(vec)) =
                &before.sorts[signature.inputs[0].sort as usize].kind
            else {
                panic!("Vec")
            };
            assert_eq!(vec.name, "Vec");
            let Some(proto::sort::Kind::Family(pair)) = &before.sorts[vec.args[0] as usize].kind
            else {
                panic!("Pair")
            };
            assert_eq!(pair.name, "Pair");
            assert_eq!(
                declaration
                    .bindings
                    .as_ref()
                    .unwrap()
                    .egglog
                    .as_ref()
                    .unwrap()
                    .views[0]
                    .symbol,
                "map-from-pairs"
            );
        }
    }
    graph
        .parse_and_run_program(None, "(sort M (Map i64 f64))")
        .unwrap();
    assert_eq!(
        graph.type_info().builtin_catalog().unwrap().definitions,
        before
    );
}

#[test]
fn map_from_computed_pairs_preserves_flat_construction_and_duplicates() {
    EGraph::default()
        .parse_and_run_program(
            None,
            "
        (sort M (Map i64 f64))
        (sort P (Pair i64 f64)) (sort V (Vec P))
        (function entries () V :merge new)
        (set (entries) (vec-of (pair 1 2.0) (pair 3 4.0) (pair 1 5.0)))
        (check (= (map-from-pairs (entries)) (map-of 1 2.0 3 4.0 1 5.0)))
        (check (= (map-get (map-from-pairs (entries)) 1) 5.0))
        (check (= (map-get (map-insert (map-empty) 1 6.0) 1) 6.0))
        (set (entries) (vec-empty))
        (check (= (map-from-pairs (entries)) (map-of)))
        (check (= (map-of) (map-empty)))
    ",
        )
        .unwrap();
}

#[test]
fn real_map_accepts_both_foreign_aliases_and_rejects_wrong_nested_sorts() {
    for owner_first in [false, true] {
        let mut graph = EGraph::default();
        if owner_first {
            graph
                .parse_and_run_program(None, "(sort M (Map i64 f64))")
                .unwrap();
        }
        graph
            .parse_and_run_program(
                None,
                "
            (sort P (Pair i64 f64)) (sort Q (Pair i64 f64))
            (sort VP (Vec P)) (sort VQ (Vec Q))
            (sort BadP (Pair f64 i64)) (sort BadV (Vec BadP))
            (sort ScalarV (Vec i64))
            (function p () VP :merge new) (function q () VQ :merge new)
            (function bad () BadV :merge new) (function scalar () ScalarV :merge new)
            (set (p) (vec-of (pair 1 2.0))) (set (q) (vec-of (pair 3 4.0)))
        ",
            )
            .unwrap();
        if !owner_first {
            graph
                .parse_and_run_program(None, "(sort M (Map i64 f64))")
                .unwrap();
        }
        graph
            .parse_and_run_program(
                None,
                "
            (check (= (map-get (map-from-pairs (p)) 1) 2.0))
            (check (= (map-get (map-from-pairs (q)) 3) 4.0))
        ",
            )
            .unwrap();
        for invalid in [
            "(map-from-pairs (bad))",
            "(map-from-pairs (scalar))",
            "(map-from-pairs)",
            "(map-from-pairs (p) (q))",
            "(map-from-pairs (vec-empty))",
        ] {
            assert!(
                graph.parse_and_run_program(None, invalid).is_err(),
                "{invalid}"
            );
        }
        let error = graph
            .parse_and_run_program(None, "(map-of 1 2.0 3)")
            .unwrap_err()
            .to_string();
        assert!(error.contains("map-of"), "{error}");
        assert!(!error.contains("egglog.core.map"), "{error}");
    }
}

#[test]
fn resolved_call_export_checks_direct_annotations_and_preserves_ordinary_calls() {
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(
            None,
            "(sort P (Pair i64 i64)) (sort V (Vec i64)) (sort M (Map i64 i64))",
        )
        .unwrap();
    let i = graph.get_sort_by_name("i64").unwrap().clone();
    for (name, output, count, key) in [
        ("+", "i64", 2, "egglog.core.i64.add"),
        ("|", "i64", 2, "egglog.core.i64.bitor"),
        ("pair", "P", 2, "egglog.core.pair.make"),
        ("vec-of", "V", 2, "egglog.core.vec.of"),
        ("map-of", "M", 2, "egglog.core.map.of"),
    ] {
        let mut native = vec![i.clone(); count];
        native.push(graph.get_sort_by_name(output).unwrap().clone());
        let ResolvedCall::Primitive(primitive) = ResolvedCall::from_resolution(
            name,
            &native,
            graph.type_info(),
            Context::Pure,
            &ast::Span::Panic,
        )
        .unwrap() else {
            unreachable!()
        };
        let mut source = proto::Program::default();
        let sort_id = graph.export_sort(&i, &mut source.sorts).unwrap();
        source.nodes.push(proto::Node {
            sort_id,
            kind: Some(proto::node::Kind::PrimitiveValue(proto::PrimitiveValue {
                value: Some(proto::primitive_value::Value::I64(2)),
            })),
            ..Default::default()
        });
        let call = primitive
            .export_builtin_call(&graph, &[0, 0], None, &mut source.clone())
            .unwrap();
        assert_eq!(call.func, key);
        if name != "map-of" {
            assert_eq!(call.args, [0, 0]);
        }
        for args in [&[0][..], &[0, 9][..], &[0, 0, 0][..]] {
            assert!(
                primitive
                    .export_builtin_call(&graph, args, None, &mut source.clone())
                    .is_err()
            );
        }
        for wrong in [
            proto::sort::Kind::Eq("i64".into()),
            proto::sort::Kind::Var(0),
            proto::sort::Kind::Family(proto::HostSort {
                name: "f64".into(),
                args: vec![],
            }),
            proto::sort::Kind::Family(proto::HostSort {
                name: "Vec".into(),
                args: vec![0],
            }),
        ] {
            let mut bad = source.clone();
            bad.sorts[0].kind = Some(wrong);
            assert!(
                primitive
                    .export_builtin_call(&graph, &[0, 0], None, &mut bad)
                    .is_err()
            );
        }
        let mut relocated = source.clone();
        relocated.sorts.insert(
            0,
            proto::Sort {
                kind: Some(proto::sort::Kind::Eq("Other".into())),
                ..Default::default()
            },
        );
        relocated.nodes[0].sort_id += 1;
        primitive
            .export_builtin_call(&graph, &[0, 0], None, &mut relocated)
            .unwrap();
    }
    let ResolvedCall::Primitive(unsupported) = ResolvedCall::from_resolution(
        "ordering-min",
        &[i.clone(), i.clone(), i],
        graph.type_info(),
        Context::Pure,
        &ast::Span::Panic,
    )
    .unwrap() else {
        unreachable!()
    };
    let mut source = proto::Program {
        sorts: vec![proto::Sort {
            kind: Some(proto::sort::Kind::Family(proto::HostSort {
                name: "i64".into(),
                args: vec![],
            })),
            ..Default::default()
        }],
        nodes: vec![proto::Node {
            sort_id: 0,
            kind: Some(proto::node::Kind::Var("x".into())),
            ..Default::default()
        }],
        ..Default::default()
    };
    assert!(
        unsupported
            .export_builtin_call(&graph, &[0, 0], None, &mut source)
            .is_err()
    );
}

#[test]
fn vec_catalog_has_generic_and_closed_language_views() {
    let catalog = EGraph::default()
        .type_info()
        .builtin_catalog()
        .unwrap()
        .definitions;
    let family = catalog
        .declarations
        .iter()
        .find_map(|d| match &d.kind {
            Some(proto::declaration::Kind::HostSortFamily(f)) if f.name == "Vec" => Some(f),
            _ => None,
        })
        .unwrap();
    let binding = family.bindings.as_ref().expect("Vec family presentation");
    let python = binding.python.as_ref().expect("Vec Python binding");
    assert_eq!(python.path, ["egglog", "builtins", "Vec"]);
    assert_eq!(python.type_params, ["T"]);
    let rust = binding.rust.as_ref().unwrap();
    assert_eq!(
        rust.path,
        ["egglog_experimental", "typed", "builtins", "Vec"]
    );
    assert_eq!(rust.type_params, ["T"]);
    for method in ["empty", "of", "get", "range"] {
        let declaration = catalog.declarations.iter().find(|d| matches!(&d.kind, Some(proto::declaration::Kind::HostPrimitive(p)) if p.name == format!("egglog.core.vec.{method}"))).unwrap();
        let bindings = declaration.bindings.as_ref().unwrap();
        let python = &bindings.python.as_ref().unwrap().views[0];
        assert_eq!(
            python.kind,
            match method {
                "of" => proto::PythonCallKind::Initializer,
                "empty" => proto::PythonCallKind::ClassMethod,
                _ => proto::PythonCallKind::Method,
            } as i32
        );
        assert_eq!(
            python.path,
            match method {
                "of" => vec![],
                "empty" => vec!["empty"],
                "range" => vec!["range"],
                _ => vec!["__getitem__"],
            }
        );
        assert_eq!(
            python.receiver,
            matches!(method, "get" | "range").then_some(0)
        );
        assert_eq!(
            python.params,
            match method {
                "of" => vec![proto::PythonParameter {
                    core_input: Some(0),
                    name: "args".into(),
                    default_expr: None
                }],
                "get" => vec![proto::PythonParameter {
                    core_input: Some(1),
                    name: "index".into(),
                    default_expr: None
                }],
                _ => vec![],
            }
        );
        assert!(python.mutates.is_none());
        assert_eq!(
            bindings.egglog.as_ref().unwrap().views[0].symbol,
            format!("vec-{method}")
        );
        let rust = &bindings
            .rust
            .as_ref()
            .expect("Vec callable presentation")
            .views[0];
        assert_eq!(python.owner, rust.owner);
        assert_eq!(rust.path, [method]);
        let Some(proto::binding_owner::Kind::Sort(owner)) = rust.owner.unwrap().kind else {
            panic!("sort owner")
        };
        let Some(proto::sort::Kind::Family(owner)) = &catalog.sorts[owner as usize].kind else {
            panic!("Vec owner")
        };
        if method == "range" {
            assert_eq!(owner.name, "i64");
            assert!(owner.args.is_empty());
            let Some(proto::declaration::Kind::HostPrimitive(primitive)) = &declaration.kind else {
                unreachable!()
            };
            let Some(proto::host_primitive::Typing::Signature(signature)) = &primitive.typing
            else {
                unreachable!()
            };
            assert!(signature.type_params.is_empty());
            assert!(signature.varargs.is_empty());
            assert_eq!(signature.inputs.len(), 1);
            assert_eq!(
                catalog.sorts[signature.inputs[0].sort as usize].kind,
                Some(proto::sort::Kind::Family(owner.clone()))
            );
            let Some(proto::sort::Kind::Family(output)) =
                &catalog.sorts[signature.output.unwrap() as usize].kind
            else {
                unreachable!()
            };
            assert_eq!(output.name, "Vec");
            assert_eq!(output.args.len(), 1);
            assert_eq!(
                catalog.sorts[output.args[0] as usize].kind,
                Some(proto::sort::Kind::Family(owner.clone()))
            );
        } else {
            assert_eq!(owner.name, "Vec");
            assert!(matches!(
                catalog.sorts[owner.args[0] as usize].kind,
                Some(proto::sort::Kind::Var(0))
            ));
        }
        let expected = match method {
            "empty" => (None, vec![]),
            "of" => (
                None,
                vec![proto::RustParameter {
                    core_input: Some(0),
                    name: "values".into(),
                    borrowed: false,
                }],
            ),
            "range" => (
                Some(proto::RustReceiver {
                    core_input: Some(0),
                    borrowed: false,
                }),
                vec![],
            ),
            _ => (
                Some(proto::RustReceiver {
                    core_input: Some(0),
                    borrowed: true,
                }),
                vec![proto::RustParameter {
                    core_input: Some(1),
                    name: "index".into(),
                    borrowed: false,
                }],
            ),
        };
        assert_eq!((rust.receiver, rust.params.clone()), expected);
    }
}

#[test]
fn constant_catalog_export_does_not_execute_the_body_or_validator() {
    static BODY: AtomicUsize = AtomicUsize::new(0);
    static VALIDATOR: AtomicUsize = AtomicUsize::new(0);
    let bindings = |_: &proto::GenericSignature| proto::CallableBindings {
        python: Some(proto::PythonBindings {
            views: vec![proto::PythonCallable {
                kind: proto::PythonCallKind::Constant.into(),
                path: vec!["test".into(), "C".into()],
                ..Default::default()
            }],
        }),
        egglog: Some(proto::EgglogBindings {
            views: vec![proto::EgglogCallable {
                symbol: "constant".into(),
                datatype_member: false,
            }],
        }),
        ..Default::default()
    };
    let mut graph = EGraph::default();
    add_primitive_with_validator!(&mut graph, "constant" [id = "test.constant", bindings = bindings] = || -> i64 {
        { BODY.fetch_add(1, Ordering::SeqCst); 7 }
    }, |_: &mut TermDag, _: &[TermId]| { VALIDATOR.fetch_add(1, Ordering::SeqCst); None });
    let catalog = graph.type_info().builtin_catalog().unwrap().definitions;
    let declaration = catalog.declarations.iter().find(|d| matches!(&d.kind, Some(proto::declaration::Kind::HostPrimitive(p)) if p.name == "test.constant")).unwrap();
    assert_eq!(
        declaration
            .bindings
            .as_ref()
            .unwrap()
            .python
            .as_ref()
            .unwrap()
            .views[0]
            .kind,
        proto::PythonCallKind::Constant as i32
    );
    assert_eq!(BODY.load(Ordering::SeqCst), 0);
    assert_eq!(VALIDATOR.load(Ordering::SeqCst), 0);
    let i = graph.get_sort_by_name("i64").unwrap().clone();
    let ResolvedCall::Primitive(primitive) = ResolvedCall::from_resolution(
        "constant",
        &[i],
        graph.type_info(),
        Context::Pure,
        &ast::Span::Panic,
    )
    .unwrap() else {
        unreachable!()
    };
    let mut exported = proto::Program::default();
    assert_eq!(
        primitive
            .export_builtin_call(&graph, &[], None, &mut exported)
            .unwrap()
            .func,
        "test.constant"
    );
    assert_eq!(BODY.load(Ordering::SeqCst), 0);
    assert_eq!(VALIDATOR.load(Ordering::SeqCst), 0);
    graph
        .parse_and_run_program(None, "(check (= (constant) 7))")
        .unwrap();
    assert!(BODY.load(Ordering::SeqCst) > 0);
    assert_eq!(
        graph.type_info().builtin_catalog().unwrap().definitions,
        catalog
    );
}

#[test]
fn map_constructor_validator_consumes_canonical_values_and_keeps_flat_normal_form() {
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(
            None,
            "(sort M (Map i64 i64)) (sort P (Pair i64 i64)) (sort V (Vec P))",
        )
        .unwrap();
    let m = graph.get_sort_by_name("M").unwrap().clone();
    let v = graph.get_sort_by_name("V").unwrap().clone();
    let ResolvedCall::Primitive(primitive) = ResolvedCall::from_resolution(
        "map-from-pairs",
        &[v, m],
        graph.type_info(),
        Context::Pure,
        &ast::Span::Panic,
    )
    .unwrap() else {
        unreachable!()
    };
    let validate = primitive.validator().unwrap();
    let mut terms = TermDag::default();
    let one = terms.lit(1.into());
    let two = terms.lit(2.into());
    let three = terms.lit(3.into());
    let a = terms.app("pair".into(), vec![one, two]);
    let b = terms.app("pair".into(), vec![one, three]);
    let entries = terms.app("vec-of".into(), vec![a, b]);
    let expected = terms.app("map-of".into(), vec![one, three]);
    assert_eq!(validate(&mut terms, &[entries]), Some(expected));
    let empty = terms.app("map-of".into(), vec![]);
    for name in ["vec-of", "vec-empty"] {
        let entries = terms.app(name.into(), vec![]);
        assert_eq!(validate(&mut terms, &[entries]), Some(empty));
    }
    let short = terms.app("pair".into(), vec![one]);
    let malformed = terms.app("vec-of".into(), vec![short]);
    let wrong = terms.app("vec-empty".into(), vec![a]);
    let raw = terms.app("vec-of".into(), vec![one]);
    for bad in [one, a, malformed, wrong, raw] {
        assert_eq!(validate(&mut terms, &[bad]), None);
    }
    assert_eq!(validate(&mut terms, &[]), None);
    assert_eq!(validate(&mut terms, &[entries, entries]), None);
}

#[test]
fn map_packing_preserves_ordered_failing_child_evaluation() {
    for packed in [false, true] {
        let calls = Arc::new(std::sync::Mutex::new(vec![]));
        let observed = calls.clone();
        let mut graph = EGraph::default();
        add_primitive!(&mut graph, "observe" = {observed: Arc<std::sync::Mutex<Vec<i64>>>} |x: i64| -?> i64 {{
            self.ctx.lock().unwrap().push(x);
            (x != 3).then_some(x)
        }});
        graph
            .parse_and_run_program(
                None,
                "(sort M (Map i64 i64)) (sort P (Pair i64 i64)) (sort V (Vec P))",
            )
            .unwrap();
        let expr = if packed {
            "(map-from-pairs (vec-of (pair (observe 1) (observe 2)) (pair (observe 3) (observe 4))))"
        } else {
            "(map-of (observe 1) (observe 2) (observe 3) (observe 4))"
        };
        assert!(graph.parse_and_run_program(None, expr).is_err());
        assert_eq!(*calls.lock().unwrap(), [1, 2, 3]);
    }
}

#[test]
fn scalar_catalog_has_authoritative_language_views() {
    let catalog = EGraph::default()
        .type_info()
        .builtin_catalog()
        .unwrap()
        .definitions;
    for (scalar, rust_name) in [("i64", "I64"), ("f64", "F64")] {
        let family = catalog
            .declarations
            .iter()
            .find_map(|d| match &d.kind {
                Some(proto::declaration::Kind::HostSortFamily(f)) if f.name == scalar => Some(f),
                _ => None,
            })
            .unwrap();
        let bindings = family
            .bindings
            .as_ref()
            .expect("native family must supply presentation");
        assert_eq!(
            bindings.python.as_ref().unwrap().path,
            ["egglog", "builtins", scalar]
        );
        assert_eq!(
            bindings.rust.as_ref().unwrap().path,
            ["egglog_experimental", "typed", "builtins", rust_name]
        );
        let declaration = catalog.declarations.iter().find(|d| matches!(&d.kind, Some(proto::declaration::Kind::HostPrimitive(p)) if p.name == format!("egglog.core.{scalar}.add"))).unwrap();
        let bindings = declaration.bindings.as_ref().unwrap();
        let Some(proto::declaration::Kind::HostPrimitive(primitive)) = &declaration.kind else {
            unreachable!()
        };
        let Some(proto::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
            unreachable!()
        };
        let python = &bindings.python.as_ref().unwrap().views[0];
        assert_eq!(python.kind, proto::PythonCallKind::Method as i32);
        assert_eq!(python.path, ["__add__"]);
        assert_eq!(python.receiver, Some(0));
        assert_eq!(python.params[0].core_input, Some(1));
        let rust = &bindings.rust.as_ref().unwrap().views[0];
        assert_eq!(rust.path, ["add"]);
        assert_eq!(rust.receiver.unwrap().core_input, Some(0));
        assert_eq!(
            rust.trait_impl.as_ref().unwrap().path,
            ["core", "ops", "Add"]
        );
        assert_eq!(
            rust.trait_impl
                .as_ref()
                .unwrap()
                .output_associated_type
                .as_deref(),
            Some("Output")
        );
        assert_eq!(python.owner, rust.owner);
        // Preserve the complete previously supplied Add blocks, including
        // parameter labels and absent specialization fields.
        assert_eq!(
            bindings.python,
            Some(proto::PythonBindings {
                views: vec![proto::PythonCallable {
                    kind: proto::PythonCallKind::Method.into(),
                    path: vec!["__add__".into()],
                    owner: python.owner,
                    receiver: Some(0),
                    params: vec![proto::PythonParameter {
                        core_input: Some(1),
                        name: "other".into(),
                        default_expr: None,
                    }],
                    ..Default::default()
                }],
            })
        );
        assert_eq!(
            bindings.rust,
            Some(proto::RustBindings {
                views: vec![proto::RustCallable {
                    path: vec!["add".into()],
                    owner: rust.owner,
                    receiver: Some(proto::RustReceiver {
                        core_input: Some(0),
                        borrowed: false
                    }),
                    params: vec![proto::RustParameter {
                        core_input: Some(1),
                        name: "rhs".into(),
                        borrowed: false,
                    }],
                    trait_impl: Some(proto::RustTrait {
                        path: vec!["core".into(), "ops".into(), "Add".into()],
                        args: vec![proto::RustType {
                            sort: Some(signature.inputs[1].sort),
                            borrowed: false,
                        }],
                        output_associated_type: Some("Output".into()),
                    }),
                    ..Default::default()
                }],
            })
        );
    }
}

#[test]
fn scalar_families_and_actual_operation_views_are_complete() {
    let catalog = EGraph::default().type_info().builtin_catalog().unwrap();
    for (family, name) in [("String", "String"), ("bool", "Bool"), ("Unit", "Unit")] {
        let declaration = catalog
            .definitions
            .declarations
            .iter()
            .find_map(|d| match &d.kind {
                Some(proto::declaration::Kind::HostSortFamily(f)) if f.name == family => Some(f),
                _ => None,
            })
            .unwrap();
        assert_eq!(declaration.arity, 0);
        let bindings = declaration.bindings.as_ref().unwrap();
        assert_eq!(
            bindings.python.as_ref().unwrap().path,
            ["egglog", "builtins", name]
        );
        assert_eq!(
            bindings.rust.as_ref().unwrap().path,
            ["egglog_experimental", "typed", "builtins", name]
        );
        assert!(!catalog.undescribed_sorts.iter().any(|s| s == family));
    }
    for (key, alias, fixed, python_name, rust_name, trait_name) in [
        ("string.concat", "+", 0, "__add__", "add", Some("Add")),
        ("string.replace", "replace", 3, "replace", "replace", None),
        ("bool.not", "not", 1, "__invert__", "not", Some("Not")),
        ("bool.and", "and", 2, "__and__", "bitand", Some("BitAnd")),
        ("bool.or", "or", 0, "__or__", "bitor", Some("BitOr")),
        ("bool.xor", "xor", 2, "__xor__", "bitxor", Some("BitXor")),
        ("bool.implies", "=>", 2, "implies", "implies", None),
        (
            "i64.to_string",
            "to-string",
            1,
            "to_string",
            "to_string",
            None,
        ),
    ] {
        let d = catalog.definitions.declarations.iter().find(|d| matches!(&d.kind,
            Some(proto::declaration::Kind::HostPrimitive(p)) if p.name == format!("egglog.core.{key}"))).unwrap();
        let Some(proto::declaration::Kind::HostPrimitive(p)) = &d.kind else {
            unreachable!()
        };
        let Some(proto::host_primitive::Typing::Signature(s)) = &p.typing else {
            unreachable!()
        };
        assert!(s.type_params.is_empty());
        assert_eq!(s.inputs.len(), fixed);
        assert_eq!(s.varargs.len(), usize::from(fixed == 0));
        let b = d.bindings.as_ref().unwrap();
        assert_eq!(b.egglog.as_ref().unwrap().views[0].symbol, alias);
        let py_views = &b.python.as_ref().unwrap().views;
        let rs_views = &b.rust.as_ref().unwrap().views;
        if key == "string.concat" {
            assert_eq!(py_views.len(), 2);
            assert_eq!(rs_views.len(), 2);
            assert_eq!(py_views[0].kind, proto::PythonCallKind::Function as i32);
            assert_eq!(py_views[0].path, ["egglog", "builtins", "join"]);
            assert_eq!(
                rs_views[0].path,
                ["egglog_experimental", "typed", "builtins", "join"]
            );
            assert_eq!(py_views[0].tail_repetitions, None);
            assert_eq!(rs_views[0].tail_repetitions, None);
            assert_eq!(py_views[0].params[0].core_input, Some(0));
            assert_eq!(rs_views[0].params[0].core_input, Some(0));
        } else {
            assert_eq!(py_views.len(), 1);
            assert_eq!(rs_views.len(), 1);
        }
        let py = py_views.last().unwrap();
        let rs = rs_views.last().unwrap();
        assert_eq!(py.path, [python_name]);
        assert_eq!(rs.path, [rust_name]);
        assert_eq!(py.owner, rs.owner);
        assert_eq!(py.receiver, Some(0));
        assert_eq!(rs.receiver.unwrap().core_input, Some(0));
        assert_eq!(py.tail_repetitions, (fixed == 0).then_some(2));
        assert_eq!(rs.tail_repetitions, (fixed == 0).then_some(2));
        assert_eq!(
            rs.trait_impl.as_ref().map(|t| t.path[2].as_str()),
            trait_name
        );
        if key == "bool.not" {
            assert!(py.params.is_empty());
            assert!(rs.trait_impl.as_ref().unwrap().args.is_empty());
        }
    }
    for migrated in ["guard", "count-matches", "bool-<"] {
        assert!(
            !catalog
                .undescribed_primitives
                .iter()
                .any(|name| name == migrated)
        );
    }
}

#[test]
fn scalar_variadic_aliases_share_resolved_provider_contexts_and_validators() {
    let mut graph = EGraph::default();
    for (family, alias, key) in [
        ("String", "+", "egglog.core.string.concat"),
        ("bool", "or", "egglog.core.bool.or"),
        ("i64", "+", "egglog.core.i64.add"),
        ("f64", "+", "egglog.core.f64.add"),
    ] {
        let sort = graph.get_sort_by_name(family).unwrap().clone();
        for count in [0, 1, 2, 5] {
            let sorts = vec![sort.clone(); count + 1];
            for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
                let alias_result = ResolvedCall::from_resolution(
                    alias,
                    &sorts,
                    graph.type_info(),
                    context,
                    &ast::Span::Panic,
                );
                let key_result = ResolvedCall::from_resolution(
                    key,
                    &sorts,
                    graph.type_info(),
                    context,
                    &ast::Span::Panic,
                );
                if matches!(family, "i64" | "f64") && count != 2 {
                    assert!(alias_result.is_err());
                    assert!(key_result.is_err());
                    continue;
                }
                let ResolvedCall::Primitive(alias_prim) = alias_result.unwrap() else {
                    unreachable!()
                };
                let ResolvedCall::Primitive(key_prim) = key_result.unwrap() else {
                    unreachable!()
                };
                assert_eq!(
                    alias_prim, key_prim,
                    "same context-ID set and concrete sorts"
                );
                assert!(Arc::ptr_eq(
                    alias_prim.validator().unwrap(),
                    key_prim.validator().unwrap()
                ));
                let mut exported = proto::Program {
                    ir_version: 1,
                    ..Default::default()
                };
                let sort_id = graph.export_sort(&sort, &mut exported.sorts).unwrap();
                for index in 0..count {
                    use proto::primitive_value::Value as V;
                    let value = match family {
                        "String" => V::String(format!("s{index}\0\n☃")),
                        "bool" => V::Bool(index % 2 == 0),
                        "i64" => V::I64(index as i64),
                        "f64" => V::F64Bits((index as f64).to_bits()),
                        _ => unreachable!(),
                    };
                    exported.nodes.push(proto::Node {
                        sort_id,
                        kind: Some(proto::node::Kind::PrimitiveValue(proto::PrimitiveValue {
                            value: Some(value),
                        })),
                        ..Default::default()
                    });
                }
                let args = (0..count as u32).rev().collect::<Vec<_>>();
                let call = alias_prim
                    .export_builtin_call(&graph, &args, None, &mut exported)
                    .unwrap();
                assert_eq!(call.func, key);
                assert_eq!(call.args, args);
                assert_eq!(key_prim.export_builtin(&mut exported).unwrap(), key);
                let native_key = graph
                    .type_info()
                    .resolve_builtin(&exported, key, &vec![sort_id; count], sort_id, &sorts)
                    .unwrap();
                assert_eq!(native_key, key);
            }
        }
    }
}

#[test]
fn scalar_variadic_validators_preserve_empty_many_utf8_and_truth_tables() {
    let mut graph = EGraph::default();
    for (name, family) in [("+", "String"), ("or", "bool")] {
        let sort = graph.get_sort_by_name(family).unwrap().clone();
        for count in [0, 1, 2, 5] {
            let ResolvedCall::Primitive(p) = ResolvedCall::from_resolution(
                name,
                &vec![sort.clone(); count + 1],
                graph.type_info(),
                Context::Pure,
                &ast::Span::Panic,
            )
            .unwrap() else {
                unreachable!()
            };
            let mut terms = TermDag::default();
            let mut args = vec![];
            for index in 0..count {
                args.push(terms.lit(if family == "String" {
                    ast::Literal::String(format!("{index}\0\n☃"))
                } else {
                    ast::Literal::Bool(index == 1)
                }));
            }
            let expected = terms.lit(if family == "String" {
                ast::Literal::String((0..count).map(|index| format!("{index}\0\n☃")).collect())
            } else {
                ast::Literal::Bool(count > 1)
            });
            assert_eq!(p.validator().unwrap()(&mut terms, &args), Some(expected));
            let invalid = terms.app("not-a-literal".into(), vec![]);
            assert_eq!(p.validator().unwrap()(&mut terms, &[invalid]), None);
        }
    }
    for a in [false, true] {
        graph
            .parse_and_run_program(None, &format!("(check (= (not {a}) {}))", !a))
            .unwrap();
        for b in [false, true] {
            for (op, expected) in [
                ("and", a && b),
                ("or", a || b),
                ("xor", a ^ b),
                ("=>", !a || b),
            ] {
                graph
                    .parse_and_run_program(None, &format!("(check (= ({op} {a} {b}) {expected}))"))
                    .unwrap();
            }
        }
    }
    graph
        .parse_and_run_program(
            None,
            r#"
        (check (= (+) ""))
        (check (= (+ "☃") "☃"))
        (check (= (+ "a" "b" "☃" "c") "ab☃c"))
        (check (= (replace "ababa" "ba" "☃") "a☃☃"))
        (check (= (to-string -42) "-42"))
        (check (= (or) false))
        (check (= (or false) false))
        (check (= (or false true false false) true))
        (check (= () ()))
    "#,
        )
        .unwrap();
}

#[test]
fn scalar_catalog_operations_preserve_native_proofs() {
    EGraph::new_with_proofs()
        .with_proof_testing()
        .parse_and_run_program(
            None,
            r#"
        (datatype E (Text String) (Truth bool) (Number i64))
        (Text "") (Text "a") (Text "ab") (Text "abc")
        (Truth false) (Truth true) (Number 3)
        (prove (= (Text (+)) (Text "")))
        (prove (= (Text (+ "a")) (Text "a")))
        (prove (= (Text (+ "a" "b")) (Text "ab")))
        (prove (= (Text (+ "a" "b" "c")) (Text "abc")))
        (prove (= (Truth (or)) (Truth false)))
        (prove (= (Truth (or true)) (Truth true)))
        (prove (= (Truth (or false true)) (Truth true)))
        (prove (= (Truth (or false true false)) (Truth true)))
        (prove (= (Truth (not false)) (Truth true)))
        (prove (= (Number (+ 1 2)) (Number 3)))
    "#,
        )
        .unwrap();
}

#[test]
fn family_metadata_is_optional_and_frozen_per_language() {
    let graph = EGraph::default();
    let scalar = graph.get_sort_by_name("i64").unwrap().clone();
    let mut definition = builtin::closed_signature(
        "test.identity",
        "identity",
        &[("x", scalar.clone())],
        &[],
        scalar,
    )
    .unwrap();
    let mut destination = proto::Program {
        ir_version: 1,
        ..Default::default()
    };
    builtin::import_definition(&definition, &mut destination).unwrap();
    let Some(proto::declaration::Kind::HostSortFamily(family)) =
        &mut definition.declarations[0].kind
    else {
        unreachable!()
    };
    family.bindings = Some(proto::SortBindings {
        python: Some(proto::TypeBinding {
            path: vec!["test".into(), "Int".into()],
            ..Default::default()
        }),
        ..Default::default()
    });
    builtin::import_definition(&definition, &mut destination).unwrap();
    let frozen = destination.clone();
    let Some(proto::declaration::Kind::HostSortFamily(family)) =
        &mut definition.declarations[0].kind
    else {
        unreachable!()
    };
    family.bindings = None;
    builtin::import_definition(&definition, &mut destination).unwrap();
    assert_eq!(destination, frozen);
    let Some(proto::declaration::Kind::HostSortFamily(family)) =
        &mut definition.declarations[0].kind
    else {
        unreachable!()
    };
    family.bindings = Some(proto::SortBindings {
        python: Some(proto::TypeBinding::default()),
        ..Default::default()
    });
    assert!(builtin::import_definition(&definition, &mut destination).is_err());
    assert_eq!(
        destination, frozen,
        "conflict must not modify the destination"
    );
}

#[test]
fn authoritative_family_exports_without_a_described_callable() {
    struct CatalogOnly;
    impl Presort for CatalogOnly {
        fn presort_name() -> &'static str {
            "CatalogOnly"
        }
        fn reserved_primitives() -> Vec<&'static str> {
            vec![]
        }
        fn make_sort(
            _: &mut TypeInfo,
            _: String,
            _: &[ast::Expr],
            _: ast::Span,
        ) -> Result<ArcSort, TypeError> {
            panic!("catalog export must not instantiate a native factory")
        }
    }
    let mut graph = EGraph::default();
    graph
        .type_info()
        .add_presort::<CatalogOnly>(ast::Span::Panic)
        .unwrap();
    graph
        .type_info()
        .register_builtin_family(proto::HostSortFamily {
            name: "CatalogOnly".into(),
            arity: 1,
            bindings: Some(proto::SortBindings {
                python: Some(proto::TypeBinding {
                    path: vec!["test".into(), "CatalogOnly".into()],
                    type_params: vec!["T".into()],
                }),
                ..Default::default()
            }),
        })
        .unwrap();
    let catalog = graph.type_info().builtin_catalog().unwrap();
    assert!(catalog.definitions.declarations.iter().any(|d| matches!(&d.kind, Some(proto::declaration::Kind::HostSortFamily(f)) if f.name == "CatalogOnly" && f.bindings.is_some())));
    assert!(
        !catalog
            .undescribed_families
            .iter()
            .any(|name| name == "CatalogOnly")
    );
}

#[test]
fn family_registration_rejects_native_kind_and_arity_mismatches_transactionally() {
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(None, "(sort NativeEq) (sort Alias (Vec i64))")
        .unwrap();
    let before = graph.type_info().builtin_catalog().unwrap();
    for (name, arity) in [("bool", 1), ("NativeEq", 0), ("Alias", 0), ("Absent", 0)] {
        assert!(
            graph
                .type_info()
                .register_builtin_family(proto::HostSortFamily {
                    name: name.into(),
                    arity,
                    ..Default::default()
                })
                .is_err(),
            "invalid family {name}/{arity}"
        );
        let after = graph.type_info().builtin_catalog().unwrap();
        assert_eq!(after.definitions, before.definitions);
        assert_eq!(after.undescribed_primitives, before.undescribed_primitives);
        assert_eq!(after.undescribed_sorts, before.undescribed_sorts);
        assert_eq!(after.undescribed_families, before.undescribed_families);
        assert_eq!(
            after.undescribed_family_primitives,
            before.undescribed_family_primitives
        );
    }
}

#[test]
fn family_registration_uses_presort_despite_same_named_nominal_sort() {
    let mut graph = EGraph::default();
    let before = graph.type_info().builtin_catalog().unwrap().definitions;
    graph.parse_and_run_program(None, "(sort Pair)").unwrap();
    graph
        .type_info()
        .register_builtin_family(proto::HostSortFamily {
            name: "Pair".into(),
            arity: 2,
            bindings: None,
        })
        .unwrap();
    assert_eq!(
        graph.type_info().builtin_catalog().unwrap().definitions,
        before
    );
    let error = graph
        .type_info()
        .register_builtin_family(proto::HostSortFamily {
            name: "Pair".into(),
            arity: 1,
            bindings: None,
        })
        .unwrap_err();
    assert!(error.contains("conflicting declaration"), "{error}");
    assert_eq!(
        graph.type_info().builtin_catalog().unwrap().definitions,
        before
    );
    graph
        .parse_and_run_program(
            None,
            "(constructor P () Pair) (sort Real (Pair Pair i64)) (pair (P) 7)",
        )
        .unwrap();
    let mut sorts = vec![];
    let nominal = graph.get_sort_by_name("Pair").unwrap().clone();
    let scalar = graph.get_sort_by_name("i64").unwrap().clone();
    let actual = graph.get_sort_by_name("Real").unwrap().clone();
    let first = graph.export_sort(&nominal, &mut sorts).unwrap();
    let second = graph.export_sort(&scalar, &mut sorts).unwrap();
    let output = graph.export_sort(&actual, &mut sorts).unwrap();
    let program = proto::Program {
        ir_version: 1,
        sorts,
        ..Default::default()
    };
    let native = [nominal, scalar, actual];
    graph
        .type_info()
        .resolve_builtin(
            &program,
            "egglog.core.pair.make",
            &[first, second],
            output,
            &native,
        )
        .unwrap();
    for index in [first, second, output] {
        let mut wrong = program.clone();
        wrong.sorts[index as usize].kind =
            Some(match wrong.sorts[index as usize].kind.as_ref().unwrap() {
                proto::sort::Kind::Eq(name) => proto::sort::Kind::Family(proto::HostSort {
                    name: name.clone(),
                    args: vec![],
                }),
                proto::sort::Kind::Family(f) => proto::sort::Kind::Eq(f.name.clone()),
                _ => unreachable!(),
            });
        assert!(
            graph
                .type_info()
                .resolve_builtin(
                    &wrong,
                    "egglog.core.pair.make",
                    &[first, second],
                    output,
                    &native
                )
                .is_err(),
            "other-kind native availability must not satisfy sort {index}"
        );
    }
}

#[test]
fn export_is_inert_and_reports_unmigrated_registrations() {
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let mut graph = EGraph::default();
    add_literal_prim!(
        &mut graph,
        "counted"[id = "test.counted"] = |a: i64| -> i64 {
            {
                CALLS.fetch_add(1, Ordering::SeqCst);
                a
            }
        }
    );
    add_primitive!(
        &mut graph,
        "unannotated-catalog-operation" = |a: i64| -> i64 { a }
    );
    let catalog = graph.type_info().builtin_catalog().unwrap();
    assert_eq!(CALLS.load(Ordering::SeqCst), 0);
    assert!(
        catalog
            .undescribed_primitives
            .iter()
            .any(|name| name == "unannotated-catalog-operation")
    );
    assert!(
        catalog
            .undescribed_primitives
            .iter()
            .all(|name| name != "ordering-min" && name != "ordering-max")
    );
    assert!(catalog.undescribed_families.is_empty());
    assert!(catalog.undescribed_family_primitives.is_empty());
    assert!(catalog.undescribed_sorts.is_empty());
    let keys = catalog
        .definitions
        .declarations
        .iter()
        .filter_map(|declaration| match &declaration.kind {
            Some(proto::declaration::Kind::HostPrimitive(primitive)) => {
                Some(primitive.name.as_str())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        keys,
        [
            "egglog.core.bigint.add",
            "egglog.core.bigint.bitand",
            "egglog.core.bigint.bitor",
            "egglog.core.bigint.bits",
            "egglog.core.bigint.bitxor",
            "egglog.core.bigint.bool_eq",
            "egglog.core.bigint.bool_ge",
            "egglog.core.bigint.bool_gt",
            "egglog.core.bigint.bool_le",
            "egglog.core.bigint.bool_lt",
            "egglog.core.bigint.div",
            "egglog.core.bigint.from_i64",
            "egglog.core.bigint.from_string",
            "egglog.core.bigint.ge",
            "egglog.core.bigint.gt",
            "egglog.core.bigint.le",
            "egglog.core.bigint.lt",
            "egglog.core.bigint.max",
            "egglog.core.bigint.min",
            "egglog.core.bigint.mul",
            "egglog.core.bigint.not",
            "egglog.core.bigint.rem",
            "egglog.core.bigint.shl",
            "egglog.core.bigint.shr",
            "egglog.core.bigint.sub",
            "egglog.core.bigint.to_string",
            "egglog.core.bigrat.abs",
            "egglog.core.bigrat.add",
            "egglog.core.bigrat.cbrt",
            "egglog.core.bigrat.ceil",
            "egglog.core.bigrat.denom",
            "egglog.core.bigrat.div",
            "egglog.core.bigrat.floor",
            "egglog.core.bigrat.ge",
            "egglog.core.bigrat.gt",
            "egglog.core.bigrat.le",
            "egglog.core.bigrat.log",
            "egglog.core.bigrat.lt",
            "egglog.core.bigrat.make",
            "egglog.core.bigrat.max",
            "egglog.core.bigrat.min",
            "egglog.core.bigrat.mul",
            "egglog.core.bigrat.neg",
            "egglog.core.bigrat.numer",
            "egglog.core.bigrat.pow",
            "egglog.core.bigrat.round",
            "egglog.core.bigrat.sqrt",
            "egglog.core.bigrat.sub",
            "egglog.core.bigrat.to_f64",
            "egglog.core.bigrat.to_i64",
            "egglog.core.bool.and",
            "egglog.core.bool.guard",
            "egglog.core.bool.implies",
            "egglog.core.bool.not",
            "egglog.core.bool.or",
            "egglog.core.bool.xor",
            "egglog.core.bool_ne",
            "egglog.core.f64.abs",
            "egglog.core.f64.add",
            "egglog.core.f64.div",
            "egglog.core.f64.exp",
            "egglog.core.f64.from_i64",
            "egglog.core.f64.ge",
            "egglog.core.f64.gt",
            "egglog.core.f64.le",
            "egglog.core.f64.log",
            "egglog.core.f64.lt",
            "egglog.core.f64.max",
            "egglog.core.f64.min",
            "egglog.core.f64.mul",
            "egglog.core.f64.neg",
            "egglog.core.f64.pow",
            "egglog.core.f64.rem",
            "egglog.core.f64.sqrt",
            "egglog.core.f64.sub",
            "egglog.core.f64.to_i64",
            "egglog.core.f64.to_string",
            "egglog.core.fn.apply",
            "egglog.core.i64.abs",
            "egglog.core.i64.add",
            "egglog.core.i64.bitand",
            "egglog.core.i64.bitor",
            "egglog.core.i64.bitxor",
            "egglog.core.i64.bool_eq",
            "egglog.core.i64.bool_ge",
            "egglog.core.i64.bool_gt",
            "egglog.core.i64.bool_le",
            "egglog.core.i64.bool_lt",
            "egglog.core.i64.div",
            "egglog.core.i64.ge",
            "egglog.core.i64.gt",
            "egglog.core.i64.le",
            "egglog.core.i64.log2",
            "egglog.core.i64.lt",
            "egglog.core.i64.max",
            "egglog.core.i64.min",
            "egglog.core.i64.mul",
            "egglog.core.i64.not",
            "egglog.core.i64.rem",
            "egglog.core.i64.shl",
            "egglog.core.i64.shr",
            "egglog.core.i64.sub",
            "egglog.core.i64.to_string",
            "egglog.core.map.contains",
            "egglog.core.map.empty",
            "egglog.core.map.get",
            "egglog.core.map.insert",
            "egglog.core.map.length",
            "egglog.core.map.not_contains",
            "egglog.core.map.of",
            "egglog.core.map.remove",
            "egglog.core.multiset.clear_index",
            "egglog.core.multiset.contains",
            "egglog.core.multiset.contains_swapped",
            "egglog.core.multiset.count",
            "egglog.core.multiset.fill_index",
            "egglog.core.multiset.filter",
            "egglog.core.multiset.filter_not",
            "egglog.core.multiset.flat_map",
            "egglog.core.multiset.insert",
            "egglog.core.multiset.intersection",
            "egglog.core.multiset.length",
            "egglog.core.multiset.map",
            "egglog.core.multiset.not_contains",
            "egglog.core.multiset.not_contains_swapped",
            "egglog.core.multiset.of",
            "egglog.core.multiset.pick",
            "egglog.core.multiset.pick_max",
            "egglog.core.multiset.reduce",
            "egglog.core.multiset.remove",
            "egglog.core.multiset.remove_swapped",
            "egglog.core.multiset.reset_counts",
            "egglog.core.multiset.single",
            "egglog.core.multiset.subtract",
            "egglog.core.multiset.subtract_swapped",
            "egglog.core.multiset.sum",
            "egglog.core.multiset.sum_multisets",
            "egglog.core.multiset.union_values",
            "egglog.core.ne",
            "egglog.core.pair.first",
            "egglog.core.pair.make",
            "egglog.core.pair.second",
            "egglog.core.set.contains",
            "egglog.core.set.diff",
            "egglog.core.set.empty",
            "egglog.core.set.get",
            "egglog.core.set.insert",
            "egglog.core.set.intersect",
            "egglog.core.set.length",
            "egglog.core.set.not_contains",
            "egglog.core.set.of",
            "egglog.core.set.remove",
            "egglog.core.set.union",
            "egglog.core.string.concat",
            "egglog.core.string.count_matches",
            "egglog.core.string.replace",
            "egglog.core.value_eq",
            "egglog.core.vec.append",
            "egglog.core.vec.contains",
            "egglog.core.vec.empty",
            "egglog.core.vec.get",
            "egglog.core.vec.length",
            "egglog.core.vec.map",
            "egglog.core.vec.not_contains",
            "egglog.core.vec.of",
            "egglog.core.vec.pop",
            "egglog.core.vec.push",
            "egglog.core.vec.range",
            "egglog.core.vec.remove",
            "egglog.core.vec.set",
            "egglog.core.vec.union",
            "test.counted"
        ]
    );
    graph
        .parse_and_run_program(None, "(check (= (counted 7) 7))")
        .unwrap();
    assert!(CALLS.load(Ordering::SeqCst) > 0);
}

#[test]
fn pair_definitions_precede_instances_and_preserve_nominal_overloads() {
    let mut graph = EGraph::default();
    let before = graph.type_info().builtin_catalog().unwrap();
    assert!(
        !before
            .undescribed_families
            .iter()
            .any(|name| name == "Pair")
    );
    assert!(
        !before
            .undescribed_family_primitives
            .iter()
            .any(|name| name.starts_with("Pair::"))
    );
    let family = before
        .definitions
        .declarations
        .iter()
        .find_map(|d| match &d.kind {
            Some(proto::declaration::Kind::HostSortFamily(f)) if f.name == "Pair" => Some(f),
            _ => None,
        })
        .unwrap();
    assert_eq!(family.arity, 2);
    assert_eq!(
        family.bindings,
        Some(proto::SortBindings {
            python: Some(proto::TypeBinding {
                path: vec!["egglog".into(), "builtins".into(), "Pair".into()],
                type_params: vec!["L".into(), "R".into()],
            }),
            rust: Some(proto::TypeBinding {
                path: vec![
                    "egglog_experimental".into(),
                    "typed".into(),
                    "builtins".into(),
                    "Pair".into()
                ],
                type_params: vec!["L".into(), "R".into()],
            }),
            ..Default::default()
        })
    );
    for (key, alias, inputs, output) in [
        ("make", "pair", vec![0, 1], 2),
        ("first", "pair-first", vec![2], 0),
        ("second", "pair-second", vec![2], 1),
    ] {
        let mut definition = proto::Program::default();
        graph
            .type_info()
            .export_builtin_definition(&format!("egglog.core.pair.{key}"), &mut definition)
            .unwrap();
        let d = definition
            .declarations
            .iter()
            .find(|d| matches!(d.kind, Some(proto::declaration::Kind::HostPrimitive(_))))
            .unwrap();
        let Some(proto::declaration::Kind::HostPrimitive(p)) = &d.kind else {
            unreachable!()
        };
        let Some(proto::host_primitive::Typing::Signature(s)) = &p.typing else {
            panic!("signature")
        };
        assert_eq!(s.type_params, ["A", "B"]);
        assert!(s.varargs.is_empty());
        let patterns = [
            definition
                .sorts
                .iter()
                .position(|s| s.kind == Some(proto::sort::Kind::Var(0)))
                .unwrap() as u32,
            definition
                .sorts
                .iter()
                .position(|s| s.kind == Some(proto::sort::Kind::Var(1)))
                .unwrap() as u32,
            definition
                .sorts
                .iter()
                .position(
                    |s| matches!(&s.kind, Some(proto::sort::Kind::Family(f)) if f.name == "Pair"),
                )
                .unwrap() as u32,
        ];
        assert_eq!(
            s.inputs.iter().map(|a| a.sort).collect::<Vec<_>>(),
            inputs.into_iter().map(|i| patterns[i]).collect::<Vec<_>>()
        );
        assert_eq!(s.output, Some(patterns[output]));
        let Some(proto::sort::Kind::Family(f)) = &definition.sorts[patterns[2] as usize].kind
        else {
            unreachable!()
        };
        assert_eq!(f.args, patterns[..2]);
        let bindings = d.bindings.as_ref().unwrap();
        assert_eq!(
            s.inputs.iter().map(|a| a.name.as_str()).collect::<Vec<_>>(),
            if key == "make" {
                vec!["first", "second"]
            } else {
                vec!["pair"]
            }
        );
        let owner = Some(proto::BindingOwner {
            kind: Some(proto::binding_owner::Kind::Sort(patterns[2])),
        });
        assert_eq!(
            bindings.python.as_ref().unwrap().views,
            [proto::PythonCallable {
                kind: if key == "make" {
                    proto::PythonCallKind::Initializer
                } else {
                    proto::PythonCallKind::Property
                }
                .into(),
                path: match key {
                    "make" => vec![],
                    "first" => vec!["left".into()],
                    _ => vec!["right".into()],
                },
                owner,
                receiver: (key != "make").then_some(0),
                params: if key == "make" {
                    ["left", "right"]
                        .into_iter()
                        .enumerate()
                        .map(|(i, name)| proto::PythonParameter {
                            core_input: Some(i as u32),
                            name: name.into(),
                            default_expr: None,
                        })
                        .collect()
                } else {
                    vec![]
                },
                mutates: None,
                tail_repetitions: None,
            }]
        );
        assert_eq!(
            bindings.rust.as_ref().unwrap().views,
            [proto::RustCallable {
                path: vec![if key == "make" { "new" } else { key }.into()],
                owner,
                receiver: (key != "make").then_some(proto::RustReceiver {
                    core_input: Some(0),
                    borrowed: true
                }),
                params: if key == "make" {
                    ["left", "right"]
                        .into_iter()
                        .enumerate()
                        .map(|(i, name)| proto::RustParameter {
                            core_input: Some(i as u32),
                            name: name.into(),
                            borrowed: false,
                        })
                        .collect()
                } else {
                    vec![]
                },
                ..Default::default()
            }]
        );
        assert_eq!(bindings.egglog.as_ref().unwrap().views[0].symbol, alias);
    }
    graph
        .parse_and_run_program(
            None,
            r#"
        (sort P (Pair i64 String)) (sort Q (Pair i64 String))
        (sort R (Pair String i64)) (sort V (Vec P)) (sort Nested (Pair P V))
        (function p () P :merge new) (function q () Q :merge new)
        (function r () R :merge new) (function n () Nested :merge new)
        (set (p) (pair 1 "a")) (set (q) (pair 2 "b"))
        (set (r) (pair "c" 3)) (set (n) (pair (p) (vec-of (p))))
        (check (= (pair-first (p)) 1) (= (pair-second (q)) "b"))
        (check (= (pair-first (r)) "c") (= (pair-second (r)) 3))
        (check (= (pair-first (n)) (p)) (= (vec-get (pair-second (n)) 0) (p)))
    "#,
        )
        .unwrap();
    assert_eq!(
        graph.type_info().builtin_catalog().unwrap().definitions,
        before.definitions
    );
    for bad in [
        "(set (p) (q))",
        "(set (p) (pair \"a\" 1))",
        "(set (p) (pair 1))",
        "(pair-first 1)",
    ] {
        assert!(
            graph.clone().parse_and_run_program(None, bad).is_err(),
            "{bad}"
        );
    }
    let mut sorts = vec![];
    let p = graph.get_sort_by_name("P").unwrap();
    let q = graph.get_sort_by_name("Q").unwrap();
    assert_eq!(
        graph.export_sort(p, &mut sorts).unwrap(),
        graph.export_sort(q, &mut sorts).unwrap()
    );
    let nested = graph
        .export_sort(graph.get_sort_by_name("Nested").unwrap(), &mut sorts)
        .unwrap();
    assert!(
        matches!(&sorts[nested as usize].kind, Some(proto::sort::Kind::Family(f)) if f.name == "Pair" && f.args.len() == 2)
    );
    let i = graph.get_sort_by_name("i64").unwrap();
    let s = graph.get_sort_by_name("String").unwrap();
    for (alias, key, types) in [
        (
            "pair",
            "egglog.core.pair.make",
            vec![i.clone(), s.clone(), p.clone()],
        ),
        (
            "pair-first",
            "egglog.core.pair.first",
            vec![p.clone(), i.clone()],
        ),
        (
            "pair-second",
            "egglog.core.pair.second",
            vec![p.clone(), s.clone()],
        ),
    ] {
        for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
            assert_eq!(
                ResolvedCall::from_resolution(
                    alias,
                    &types,
                    graph.type_info(),
                    context,
                    &ast::Span::Panic
                )
                .unwrap(),
                ResolvedCall::from_resolution(
                    key,
                    &types,
                    graph.type_info(),
                    context,
                    &ast::Span::Panic
                )
                .unwrap()
            );
        }
    }
}

#[test]
fn pair_instances_reject_wrong_ownership_annotations_and_converters() {
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(None, "(sort P (Pair i64 String)) (sort V (Vec i64))")
        .unwrap();
    let p = graph.get_sort_by_name("P").unwrap().clone();
    let v = graph.get_sort_by_name("V").unwrap().clone();
    let i = graph.get_sort_by_name("i64").unwrap().clone();
    let s = graph.get_sort_by_name("String").unwrap().clone();
    let before = graph.type_info().builtin_catalog().unwrap().definitions;
    assert!(
        graph
            .type_info()
            .instantiate_builtin("egglog.core.pair.make", &v)
            .is_err()
    );
    assert!(
        graph
            .type_info()
            .instantiate_builtin("egglog.core.vec.of", &p)
            .is_err()
    );
    for (key, types) in [
        (
            "egglog.core.pair.make",
            vec![s.clone(), i.clone(), p.clone()],
        ),
        (
            "egglog.core.pair.make",
            vec![i.clone(), s.clone(), v.clone()],
        ),
        ("egglog.core.pair.first", vec![p.clone(), s.clone()]),
        ("egglog.core.pair.second", vec![p.clone(), i.clone()]),
    ] {
        assert!(
            ResolvedCall::from_resolution(
                key,
                &types,
                graph.type_info(),
                Context::Pure,
                &ast::Span::Panic
            )
            .is_err()
        );
    }
    for wrong in 0..4 {
        let instance = graph
            .type_info()
            .instantiate_builtin("egglog.core.pair.first", &p)
            .unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match wrong {
            0 => {
                add_primitive_with_validator!(&mut graph, "bad-pair" [instance = instance] = |xs: @VecContainer| -> # {
                    { CALLS.fetch_add(1, Ordering::SeqCst); xs.data[0] }
                }, |_: &mut TermDag, _: &[TermId]| { CALLS.fetch_add(1, Ordering::SeqCst); None });
            }
            1 => {
                add_primitive!(&mut graph, "bad-pair" [instance = instance] = |xs: @PairContainer| -> String { { let _ = xs; String::new() } });
            }
            2 => {
                add_primitive!(&mut graph, "bad-pair" [instance = instance] = |xs: @PairContainer, extra: #| -> # { { let _ = extra; xs.first } });
            }
            _ => {
                add_primitive!(&mut graph, "bad-pair" [instance = instance] = [mut xs: @PairContainer] -> # { xs.next().unwrap().first });
            }
        }));
        assert!(result.is_err(), "bad converter {wrong}");
        assert!(!graph.type_info().is_primitive("bad-pair"));
        assert_eq!(
            graph.type_info().builtin_catalog().unwrap().definitions,
            before
        );
        assert_eq!(CALLS.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn vec_definitions_precede_instances_and_preserve_nominal_overloads() {
    let mut graph = EGraph::default();
    let before = graph.type_info().builtin_catalog().unwrap();
    assert!(!before.undescribed_families.iter().any(|name| name == "Vec"));
    assert!(
        !before
            .undescribed_family_primitives
            .iter()
            .any(|name| name == "Vec::unstable-vec-map")
    );
    assert!(
        !before
            .undescribed_family_primitives
            .iter()
            .any(|name| matches!(name.as_str(), "Vec::vec-of" | "Vec::vec-range"))
    );
    for key in [
        "egglog.core.vec.empty",
        "egglog.core.vec.of",
        "egglog.core.vec.get",
        "egglog.core.vec.range",
        "egglog.core.vec.map",
    ] {
        assert!(before.definitions.declarations.iter().any(|declaration| matches!(&declaration.kind, Some(proto::declaration::Kind::HostPrimitive(primitive)) if primitive.name == key)));
    }
    graph
        .parse_and_run_program(
            None,
            r#"
        (sort V (Vec i64)) (sort W (Vec i64)) (sort Nested (Vec V)) (sort F (Vec f64))
        (function v () V :merge new) (function w () W :merge new)
        (function nested () Nested :merge new)
        (set (v) (vec-empty)) (set (w) (vec-of 3 4))
        (set (nested) (vec-of (v)))
        (check (= (vec-get (w) 1) 4) (= (vec-get (nested) 0) (v)))
        (set (v) (vec-range 1)) (set (w) (vec-range 5))
        (check (= (vec-get (v) 0) 0) (= (vec-get (w) 4) 4))
    "#,
        )
        .unwrap();
    assert_eq!(
        graph.type_info().builtin_catalog().unwrap().definitions,
        before.definitions
    );
    assert!(graph.parse_and_run_program(None, "(set (v) (w))").is_err());
    assert!(
        graph
            .parse_and_run_program(None, "(set (v) (vec-of 1.0))")
            .is_err()
    );
    let v = graph.get_sort_by_name("V").unwrap().clone();
    let i = graph.get_sort_by_name("i64").unwrap().clone();
    let w = graph.get_sort_by_name("W").unwrap().clone();
    let f = graph.get_sort_by_name("F").unwrap().clone();
    let float = graph.get_sort_by_name("f64").unwrap().clone();
    assert!(
        graph
            .type_info()
            .instantiate_builtin("egglog.core.vec.range", &f)
            .is_err()
    );
    assert!(
        graph
            .type_info()
            .instantiate_builtin("egglog.core.vec.range", &i)
            .is_err()
    );
    for (alias, key, types) in [
        ("vec-empty", "egglog.core.vec.empty", vec![v.clone()]),
        ("vec-of", "egglog.core.vec.of", vec![i.clone(), v.clone()]),
        (
            "vec-range",
            "egglog.core.vec.range",
            vec![i.clone(), v.clone()],
        ),
        (
            "vec-range",
            "egglog.core.vec.range",
            vec![i.clone(), w.clone()],
        ),
        (
            "vec-get",
            "egglog.core.vec.get",
            vec![v.clone(), i.clone(), i.clone()],
        ),
    ] {
        for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
            let canonical = ResolvedCall::from_resolution(
                key,
                &types,
                graph.type_info(),
                context,
                &ast::Span::Panic,
            )
            .unwrap();
            if key == "egglog.core.vec.range" {
                let ResolvedCall::Primitive(primitive) = &canonical else {
                    unreachable!()
                };
                assert!(primitive.validator().is_none());
            }
            assert_eq!(
                ResolvedCall::from_resolution(
                    alias,
                    &types,
                    graph.type_info(),
                    context,
                    &ast::Span::Panic
                )
                .unwrap(),
                canonical,
            );
        }
    }
    for types in [
        vec![v.clone()],
        vec![i.clone(), i.clone(), v.clone()],
        vec![f.clone(), v.clone()],
        vec![float, v.clone()],
        vec![i.clone(), f.clone()],
    ] {
        for alias in ["vec-range", "egglog.core.vec.range"] {
            assert!(
                ResolvedCall::from_resolution(
                    alias,
                    &types,
                    graph.type_info(),
                    Context::Full,
                    &ast::Span::Panic
                )
                .is_err()
            );
        }
    }
}

#[test]
fn ordinary_vec_map_definitions_are_complete() {
    let catalog = EGraph::default().type_info().builtin_catalog().unwrap();
    for key in [
        "egglog.core.vec.append",
        "egglog.core.vec.union",
        "egglog.core.vec.push",
        "egglog.core.vec.pop",
        "egglog.core.vec.length",
        "egglog.core.vec.contains",
        "egglog.core.vec.not_contains",
        "egglog.core.vec.set",
        "egglog.core.vec.remove",
        "egglog.core.map.remove",
        "egglog.core.map.length",
        "egglog.core.map.contains",
        "egglog.core.map.not_contains",
    ] {
        assert!(catalog.definitions.declarations.iter().any(|declaration| matches!(
            &declaration.kind,
            Some(proto::declaration::Kind::HostPrimitive(primitive)) if primitive.name == key
        )), "missing definition {key}");
    }
}

#[test]
fn ordinary_vec_map_signatures_aliases_and_contexts_are_exact() {
    let mut graph = EGraph::default();
    let before = graph.type_info().builtin_catalog().unwrap().definitions;
    graph
        .parse_and_run_program(None, "(sort VI (Vec i64)) (sort MI (Map String i64))")
        .unwrap();
    assert_eq!(
        graph.type_info().builtin_catalog().unwrap().definitions,
        before
    );
    for (family, key, fixed, output, native) in [
        ("vec", "push", vec![1, 0], 1, vec!["VI", "i64", "VI"]),
        ("vec", "pop", vec![1], 1, vec!["VI", "VI"]),
        ("vec", "length", vec![1], 2, vec!["VI", "i64"]),
        ("vec", "contains", vec![1, 0], 3, vec!["VI", "i64", "Unit"]),
        (
            "vec",
            "not_contains",
            vec![1, 0],
            3,
            vec!["VI", "i64", "Unit"],
        ),
        (
            "vec",
            "set",
            vec![1, 2, 0],
            1,
            vec!["VI", "i64", "i64", "VI"],
        ),
        ("vec", "remove", vec![1, 2], 1, vec!["VI", "i64", "VI"]),
        ("map", "remove", vec![2, 0], 2, vec!["MI", "String", "MI"]),
        ("map", "length", vec![2], 5, vec!["MI", "i64"]),
        (
            "map",
            "contains",
            vec![2, 0],
            6,
            vec!["MI", "String", "Unit"],
        ),
        (
            "map",
            "not_contains",
            vec![2, 0],
            6,
            vec!["MI", "String", "Unit"],
        ),
    ] {
        let canonical = format!("egglog.core.{family}.{key}");
        let alias = format!("{family}-{}", key.replace('_', "-"));
        let definitions = if family == "vec" {
            VecSort::builtin_definitions()
        } else {
            egglog::sort::MapSort::builtin_definitions()
        };
        let (definition, declaration) = definitions.iter().find_map(|p| p.declarations.iter().find(|d| matches!(&d.kind, Some(proto::declaration::Kind::HostPrimitive(h)) if h.name == canonical)).map(|d| (p, d))).unwrap();
        let Some(proto::declaration::Kind::HostPrimitive(primitive)) = &declaration.kind else {
            unreachable!()
        };
        let Some(proto::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
            unreachable!()
        };
        assert_eq!(
            signature.type_params,
            if family == "vec" {
                vec!["T"]
            } else {
                vec!["K", "V"]
            }
        );
        assert_eq!(
            signature.inputs.iter().map(|a| a.sort).collect::<Vec<_>>(),
            fixed,
            "{canonical}"
        );
        assert_eq!(signature.output, Some(output));
        assert!(signature.varargs.is_empty());
        assert!(signature.equality_type_params.is_empty());
        let bindings = declaration.bindings.as_ref().unwrap();
        assert_eq!(bindings.egglog.as_ref().unwrap().views[0].symbol, alias);
        let python = &bindings.python.as_ref().unwrap().views[0];
        let rust = &bindings.rust.as_ref().unwrap().views[0];
        assert_eq!(python.path, [key]);
        assert_eq!(rust.path, [key]);
        assert_eq!(python.receiver, Some(0));
        assert_eq!(rust.receiver.as_ref().unwrap().core_input, Some(0));
        assert!(rust.receiver.as_ref().unwrap().borrowed);
        assert_eq!(
            python
                .params
                .iter()
                .map(|p| p.core_input.unwrap())
                .collect::<Vec<_>>(),
            (1..fixed.len() as u32).collect::<Vec<_>>()
        );
        assert_eq!(
            rust.params
                .iter()
                .map(|p| p.core_input.unwrap())
                .collect::<Vec<_>>(),
            (1..fixed.len() as u32).collect::<Vec<_>>()
        );
        builtin::definitions::reconcile_declarations(definition, &mut proto::Program::default())
            .unwrap();
        let native = native
            .iter()
            .map(|name| graph.get_sort_by_name(name).unwrap().clone())
            .collect::<Vec<_>>();
        for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
            assert_eq!(
                ResolvedCall::from_resolution(
                    &canonical,
                    &native,
                    graph.type_info(),
                    context,
                    &ast::Span::Panic
                )
                .unwrap(),
                ResolvedCall::from_resolution(
                    &alias,
                    &native,
                    graph.type_info(),
                    context,
                    &ast::Span::Panic
                )
                .unwrap()
            );
        }
    }
}

#[test]
fn vec_tail_receiver_and_equality_restriction_are_exact() {
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(
            None,
            "(sort A) (sort VI (Vec i64)) (sort VA (Vec A)) (sort VVA (Vec VA))",
        )
        .unwrap();
    for (name, equality) in [("VI", false), ("VA", true), ("VVA", false)] {
        let vector = graph.get_sort_by_name(name).unwrap().clone();
        assert_eq!(
            graph
                .type_info()
                .instantiate_builtin("egglog.core.vec.union", &vector)
                .is_ok(),
            equality
        );
        for arity in [0, 1, 3] {
            let sorts = vec![vector.clone(); arity + 1];
            for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
                assert_eq!(
                    ResolvedCall::from_resolution(
                        "vec-append",
                        &sorts,
                        graph.type_info(),
                        context,
                        &ast::Span::Panic
                    )
                    .unwrap(),
                    ResolvedCall::from_resolution(
                        "egglog.core.vec.append",
                        &sorts,
                        graph.type_info(),
                        context,
                        &ast::Span::Panic
                    )
                    .unwrap(),
                );
                for key in ["vec-union", "egglog.core.vec.union"] {
                    assert_eq!(
                        ResolvedCall::from_resolution(
                            key,
                            &vec![vector.clone(); 3],
                            graph.type_info(),
                            context,
                            &ast::Span::Panic
                        )
                        .is_ok(),
                        equality && matches!(context, Context::Write | Context::Full)
                    );
                }
            }
        }
    }
    for definition in VecSort::builtin_definitions() {
        let index = definition.declarations.iter().position(|d| matches!(&d.kind, Some(proto::declaration::Kind::HostPrimitive(p)) if p.name == "egglog.core.vec.append" || p.name == "egglog.core.vec.union"));
        let Some(index) = index else { continue };
        let Some(proto::declaration::Kind::HostPrimitive(primitive)) =
            &definition.declarations[index].kind
        else {
            unreachable!()
        };
        let Some(proto::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
            unreachable!()
        };
        let union = primitive.name.ends_with("union");
        assert_eq!(signature.type_params, ["T"]);
        assert_eq!(signature.output, Some(1));
        assert_eq!(
            signature.inputs.iter().map(|a| a.sort).collect::<Vec<_>>(),
            if union { vec![1, 1] } else { vec![] }
        );
        assert_eq!(
            signature.varargs.iter().map(|a| a.sort).collect::<Vec<_>>(),
            if union { vec![] } else { vec![1] }
        );
        assert_eq!(
            signature.equality_type_params,
            if union { vec![0] } else { vec![] }
        );
        let mut installed = proto::Program::default();
        builtin::definitions::reconcile_declarations(&definition, &mut installed).unwrap();
        for mutation in 0..if union { 4 } else { 6 } {
            let mut invalid = definition.clone();
            let declaration = &mut invalid.declarations[index];
            if union {
                let Some(proto::declaration::Kind::HostPrimitive(p)) = &mut declaration.kind else {
                    unreachable!()
                };
                let Some(proto::host_primitive::Typing::Signature(s)) = &mut p.typing else {
                    unreachable!()
                };
                match mutation {
                    0 => s.equality_type_params.clear(), // distinct semantic signature, not a merge
                    1 => s.equality_type_params = vec![0, 0],
                    2 => s.equality_type_params = vec![1],
                    3 => {
                        s.type_params.push("U".into());
                        s.equality_type_params = vec![1, 0];
                    }
                    _ => unreachable!(),
                }
            } else {
                let view = &mut declaration
                    .bindings
                    .as_mut()
                    .unwrap()
                    .python
                    .as_mut()
                    .unwrap()
                    .views[0];
                match mutation {
                    0 => view.receiver = Some(1),
                    1 => view.params[0].core_input = Some(0),
                    2 => view.params.clear(),
                    3 => {
                        view.owner.as_mut().unwrap().kind =
                            Some(proto::binding_owner::Kind::Sort(2))
                    }
                    4 => view.kind = proto::PythonCallKind::Property as i32,
                    5 => {
                        let Some(proto::declaration::Kind::HostPrimitive(p)) =
                            &mut declaration.kind
                        else {
                            unreachable!()
                        };
                        let Some(proto::host_primitive::Typing::Signature(s)) = &mut p.typing
                        else {
                            unreachable!()
                        };
                        s.varargs.push(s.varargs[0].clone());
                    }
                    _ => unreachable!(),
                }
            }
            assert!(
                builtin::definitions::reconcile_declarations(&invalid, &mut installed.clone())
                    .is_err(),
                "{union} mutation {mutation}"
            );
            if !union || mutation != 0 {
                assert!(
                    builtin::definitions::reconcile_declarations(
                        &invalid,
                        &mut proto::Program::default()
                    )
                    .is_err(),
                    "fresh {union} mutation {mutation}"
                );
            }
        }
    }
}

#[test]
fn closed_maybe_defaults_preserve_optional_zero_and_reject_wrong_children() {
    // This exercises declaration validation only, not a substitute provider.
    let base = proto::Program {
        ir_version: 1,
        sorts: vec![
            proto::Sort {
                kind: Some(proto::sort::Kind::Family(proto::HostSort {
                    name: "i64".into(),
                    args: vec![],
                })),
                ..Default::default()
            },
            proto::Sort {
                kind: Some(proto::sort::Kind::Family(proto::HostSort {
                    name: "Maybe".into(),
                    args: vec![0],
                })),
                ..Default::default()
            },
            proto::Sort {
                kind: Some(proto::sort::Kind::Family(proto::HostSort {
                    name: "Maybe".into(),
                    args: vec![1],
                })),
                ..Default::default()
            },
        ],
        nodes: vec![
            proto::Node {
                sort_id: 0,
                kind: Some(proto::node::Kind::PrimitiveValue(proto::PrimitiveValue {
                    value: Some(proto::primitive_value::Value::I64(0)),
                })),
                ..Default::default()
            },
            proto::Node {
                sort_id: 1,
                kind: Some(proto::node::Kind::PrimitiveValue(proto::PrimitiveValue {
                    value: Some(proto::primitive_value::Value::Maybe(proto::MaybeValue {
                        value: None,
                    })),
                })),
                ..Default::default()
            },
            proto::Node {
                sort_id: 1,
                kind: Some(proto::node::Kind::PrimitiveValue(proto::PrimitiveValue {
                    value: Some(proto::primitive_value::Value::Maybe(proto::MaybeValue {
                        value: Some(0),
                    })),
                })),
                ..Default::default()
            },
            proto::Node {
                sort_id: 2,
                kind: Some(proto::node::Kind::PrimitiveValue(proto::PrimitiveValue {
                    value: Some(proto::primitive_value::Value::Maybe(proto::MaybeValue {
                        value: Some(1),
                    })),
                })),
                ..Default::default()
            },
        ],
        declarations: vec![proto::Declaration {
            kind: Some(proto::declaration::Kind::Function(proto::Function {
                name: "take_maybe".into(),
                inputs: vec![proto::Arg {
                    name: "value".into(),
                    sort: 1,
                }],
                output: 0,
                merge: None,
            })),
            bindings: Some(proto::CallableBindings {
                python: Some(proto::PythonBindings {
                    views: vec![proto::PythonCallable {
                        kind: proto::PythonCallKind::Function.into(),
                        path: vec!["test".into(), "take_maybe".into()],
                        params: vec![proto::PythonParameter {
                            core_input: Some(0),
                            name: "value".into(),
                            default_expr: Some(1),
                        }],
                        ..Default::default()
                    }],
                }),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    for root in 1..=3 {
        let mut program = base.clone();
        let Some(proto::declaration::Kind::Function(function)) = &mut program.declarations[0].kind
        else {
            panic!()
        };
        function.inputs[0].sort = program.nodes[root].sort_id;
        program.declarations[0]
            .bindings
            .as_mut()
            .unwrap()
            .python
            .as_mut()
            .unwrap()
            .views[0]
            .params[0]
            .default_expr = Some(root as u32);
        let mut installed = proto::Program::default();
        builtin::definitions::reconcile_declarations(&program, &mut installed).unwrap();
        let frozen = installed.clone();
        builtin::definitions::reconcile_declarations(&program, &mut installed).unwrap();
        assert_eq!(installed, frozen);
        for mutation in 0..5 {
            let mut malformed = program.clone();
            match mutation {
                0 => {
                    malformed.nodes[root].kind =
                        Some(proto::node::Kind::PrimitiveValue(proto::PrimitiveValue {
                            value: Some(proto::primitive_value::Value::Maybe(proto::MaybeValue {
                                value: Some(99),
                            })),
                        }))
                }
                1 => {
                    malformed.nodes[root].kind =
                        Some(proto::node::Kind::PrimitiveValue(proto::PrimitiveValue {
                            value: Some(proto::primitive_value::Value::Maybe(proto::MaybeValue {
                                value: Some(root as u32),
                            })),
                        }))
                }
                2 => {
                    malformed.sorts[program.nodes[root].sort_id as usize].kind =
                        Some(proto::sort::Kind::Family(proto::HostSort {
                            name: "Vec".into(),
                            args: vec![0],
                        }))
                }
                3 => {
                    malformed.sorts[program.nodes[root].sort_id as usize].kind =
                        Some(proto::sort::Kind::Family(proto::HostSort {
                            name: "Maybe".into(),
                            args: vec![],
                        }))
                }
                _ => malformed.nodes[root].kind = Some(proto::node::Kind::Var("free".into())),
            }
            assert!(
                builtin::definitions::reconcile_declarations(
                    &malformed,
                    &mut proto::Program::default()
                )
                .is_err(),
                "root {root}, mutation {mutation}"
            );
            assert!(
                builtin::definitions::reconcile_declarations(&malformed, &mut installed).is_err()
            );
            assert_eq!(installed, frozen);
        }
    }
}

#[test]
fn closed_defaults_enforce_exact_equality_binders_after_substitution() {
    let definition = VecSort::builtin_definitions().into_iter().find(|p| p.declarations.iter().any(|d| matches!(&d.kind, Some(proto::declaration::Kind::HostPrimitive(h)) if h.name == "egglog.core.vec.union"))).unwrap();
    for element in [
        proto::sort::Kind::Eq("Atom".into()),
        proto::sort::Kind::Family(proto::HostSort {
            name: "i64".into(),
            args: vec![],
        }),
        proto::sort::Kind::Family(proto::HostSort {
            name: "Vec".into(),
            args: vec![4],
        }),
    ] {
        let valid = matches!(element, proto::sort::Kind::Eq(_));
        let mut candidate = definition.clone();
        let concrete = candidate.sorts.len() as u32;
        candidate.sorts.push(proto::Sort {
            kind: Some(element),
            ..Default::default()
        });
        candidate.sorts.push(proto::Sort {
            kind: Some(proto::sort::Kind::Family(proto::HostSort {
                name: "Vec".into(),
                args: vec![concrete],
            })),
            ..Default::default()
        });
        // In the container-negative case, use a distinct closed Eq leaf.
        if !valid
            && matches!(&candidate.sorts[concrete as usize].kind, Some(proto::sort::Kind::Family(f)) if f.name == "Vec")
        {
            let leaf = candidate.sorts.len() as u32;
            candidate.sorts.push(proto::Sort {
                kind: Some(proto::sort::Kind::Eq("Atom".into())),
                ..Default::default()
            });
            let Some(proto::sort::Kind::Family(f)) = &mut candidate.sorts[concrete as usize].kind
            else {
                unreachable!()
            };
            f.args = vec![leaf];
        }
        candidate.declarations.push(proto::Declaration {
            kind: Some(proto::declaration::Kind::EqSort(proto::EqSort {
                name: "Atom".into(),
                ..Default::default()
            })),
            ..Default::default()
        });
        candidate.nodes = vec![
            proto::Node {
                sort_id: concrete + 1,
                kind: Some(proto::node::Kind::PrimitiveValue(proto::PrimitiveValue {
                    value: Some(proto::primitive_value::Value::Vec(Default::default())),
                })),
                ..Default::default()
            },
            proto::Node {
                sort_id: concrete + 1,
                kind: Some(proto::node::Kind::Call(proto::Call {
                    func: "egglog.core.vec.union".into(),
                    args: vec![0, 0],
                })),
                ..Default::default()
            },
        ];
        let declaration = candidate.declarations.iter_mut().find(|d| matches!(&d.kind, Some(proto::declaration::Kind::HostPrimitive(h)) if h.name == "egglog.core.vec.union")).unwrap();
        declaration
            .bindings
            .as_mut()
            .unwrap()
            .python
            .as_mut()
            .unwrap()
            .views[0]
            .params[0]
            .default_expr = Some(1);
        let result = builtin::definitions::reconcile_declarations(
            &candidate,
            &mut proto::Program::default(),
        );
        if valid {
            result.unwrap();
        } else {
            assert!(result.unwrap_err().contains("requires an equality sort"));
        }
    }
}

#[test]
fn vec_instance_macro_rejects_conversion_and_arity_mismatches_before_registration() {
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(None, "(sort V (Vec i64))")
        .unwrap();
    let before = graph.type_info().builtin_catalog().unwrap().definitions;
    for wrong in 0..6 {
        let native = graph.get_sort_by_name("V").unwrap().clone();
        let instance = graph
            .type_info()
            .instantiate_builtin(
                if wrong < 3 {
                    "egglog.core.vec.get"
                } else {
                    "egglog.core.vec.range"
                },
                &native,
            )
            .unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if wrong == 0 {
                add_primitive_with_validator!(&mut graph, "bad" [instance = instance] = |xs: @VecContainer, index: String| -?> # {
                    { CALLS.fetch_add(1, Ordering::SeqCst); xs.data.get(index.len()).copied() }
                }, |_: &mut TermDag, _: &[TermId]| { CALLS.fetch_add(1, Ordering::SeqCst); None });
            } else if wrong == 1 {
                add_primitive!(&mut graph, "bad" [instance = instance] = |xs: @VecContainer, index: i64| -> bool { { let _ = (xs, index); false } });
            } else if wrong == 2 {
                add_primitive!(&mut graph, "bad" [instance = instance] = [mut xs: #] -> # { xs.next().unwrap() });
            } else if wrong == 3 {
                add_primitive_with_validator!(&mut graph, "bad" [instance = instance] = |end: String| -> @VecContainer {
                    { CALLS.fetch_add(1, Ordering::SeqCst); let _ = end; VecContainer { do_rebuild: false, data: vec![] } }
                }, |_: &mut TermDag, _: &[TermId]| { CALLS.fetch_add(1, Ordering::SeqCst); None });
            } else if wrong == 4 {
                add_primitive!(
                    &mut graph,
                    "bad"[instance = instance] = |end: i64| -> i64 { end }
                );
            } else {
                add_primitive!(&mut graph, "bad" [instance = instance] = [end: i64] -> @VecContainer { { let _ = end; VecContainer { do_rebuild: false, data: vec![] } } });
            }
        }));
        assert!(result.is_err());
        assert_eq!(
            graph.type_info().builtin_catalog().unwrap().definitions,
            before
        );
        assert_eq!(CALLS.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn generic_signature_import_remaps_children_and_ignores_binder_labels() {
    let mut definitions = VecSort::builtin_definitions();
    let mut definition = definitions.remove(1);
    let mut destination = proto::Program {
        ir_version: 1,
        sorts: vec![proto::Sort {
            kind: Some(proto::sort::Kind::Eq("Prior".into())),
            ..Default::default()
        }],
        ..Default::default()
    };
    builtin::import_definition(&definition, &mut destination).unwrap();
    let before = destination.clone();
    let Some(proto::declaration::Kind::HostPrimitive(primitive)) =
        &mut definition.declarations.last_mut().unwrap().kind
    else {
        unreachable!()
    };
    let Some(proto::host_primitive::Typing::Signature(signature)) = &mut primitive.typing else {
        unreachable!()
    };
    signature.type_params[0] = "Renamed".into();
    signature.varargs[0].name = "renamed".into();
    builtin::import_definition(&definition, &mut destination).unwrap();
    assert_eq!(destination, before);
    definition.sorts[0].kind = Some(proto::sort::Kind::Var(1));
    assert!(builtin::import_definition(&definition, &mut destination).is_err());
}

#[test]
fn program_primitive_reconciliation_checks_body_scope_and_identity() {
    let source = proto::Program {
        ir_version: 1,
        sorts: vec![proto::Sort {
            kind: Some(proto::sort::Kind::Family(proto::HostSort {
                name: "i64".into(),
                args: vec![],
            })),
            ..Default::default()
        }],
        nodes: vec![proto::Node {
            sort_id: 0,
            kind: Some(proto::node::Kind::Var("_0".into())),
            ..Default::default()
        }],
        declarations: vec![proto::Declaration {
            kind: Some(proto::declaration::Kind::Primitive(proto::Primitive {
                name: "identity".into(),
                inputs: vec![proto::Arg {
                    name: "value".into(),
                    sort: 0,
                }],
                output: 0,
                body: Some(0),
            })),
            ..Default::default()
        }],
        ..Default::default()
    };
    let mut destination = proto::Program {
        ir_version: 1,
        sorts: vec![proto::Sort {
            kind: Some(proto::sort::Kind::Eq("Prior".into())),
            ..Default::default()
        }],
        ..Default::default()
    };
    assert_eq!(
        builtin::definitions::reconcile_declarations(&source, &mut destination).unwrap(),
        [true]
    );
    let before = destination.clone();
    for mutation in [
        "label",
        "missing",
        "bounds",
        "free",
        "out-of-scope",
        "noncanonical",
        "sort",
        "changed-body",
        "shared-scope",
        "cycle",
        "lambda",
    ] {
        let mut candidate = source.clone();
        let Some(proto::declaration::Kind::Primitive(primitive)) =
            &mut candidate.declarations[0].kind
        else {
            unreachable!()
        };
        match mutation {
            "label" => primitive.inputs[0].name = "renamed".into(),
            "missing" => primitive.body = None,
            "bounds" => primitive.body = Some(9),
            "free" => candidate.nodes[0].kind = Some(proto::node::Kind::Var("outer".into())),
            "out-of-scope" => candidate.nodes[0].kind = Some(proto::node::Kind::Var("_1".into())),
            "noncanonical" => candidate.nodes[0].kind = Some(proto::node::Kind::Var("_00".into())),
            "sort" => {
                candidate.sorts.push(proto::Sort {
                    kind: Some(proto::sort::Kind::Family(proto::HostSort {
                        name: "bool".into(),
                        args: vec![],
                    })),
                    ..Default::default()
                });
                candidate.nodes[0].sort_id = 1;
            }
            "changed-body" => {
                candidate.nodes[0].kind =
                    Some(proto::node::Kind::PrimitiveValue(proto::PrimitiveValue {
                        value: Some(proto::primitive_value::Value::I64(1)),
                    }))
            }
            "shared-scope" => candidate.declarations.push(proto::Declaration {
                kind: Some(proto::declaration::Kind::Primitive(proto::Primitive {
                    name: "nullary".into(),
                    inputs: vec![],
                    output: 0,
                    body: Some(0),
                })),
                ..Default::default()
            }),
            "cycle" => {
                candidate.nodes[0].kind = Some(proto::node::Kind::Call(proto::Call {
                    func: "identity".into(),
                    args: vec![0],
                }))
            }
            "lambda" => {
                candidate.nodes[0].kind =
                    Some(proto::node::Kind::PrimitiveValue(proto::PrimitiveValue {
                        value: Some(proto::primitive_value::Value::Lambda(proto::Lambda {
                            captures: vec![],
                            body: 0,
                        })),
                    }))
            }
            _ => unreachable!(),
        }
        let result = builtin::definitions::reconcile_declarations(&candidate, &mut destination);
        assert_eq!(
            result.is_ok(),
            mutation == "label",
            "{mutation}: {result:?}"
        );
        assert_eq!(destination, before, "{mutation}");
        let mut fresh = proto::Program::default();
        let fresh_result = builtin::definitions::reconcile_declarations(&candidate, &mut fresh);
        assert_eq!(
            fresh_result.is_ok(),
            matches!(mutation, "label" | "changed-body"),
            "fresh {mutation}: {fresh_result:?}"
        );
    }
    let mut distinct = source;
    let Some(proto::declaration::Kind::Primitive(primitive)) = &mut distinct.declarations[0].kind
    else {
        unreachable!()
    };
    primitive.name = "another_identity".into();
    assert_eq!(
        builtin::definitions::reconcile_declarations(&distinct, &mut destination).unwrap(),
        [true]
    );
    assert_eq!(destination.declarations.len(), 2);
}

#[test]
fn migrated_vec_operations_preserve_native_proofs() {
    EGraph::new_with_proofs()
        .parse_and_run_program(
            None,
            r#"
        (sort V (Vec i64))
        (datatype E (Wrap V) (Num i64))
        (Wrap (vec-empty)) (Wrap (vec-of 1 2)) (Num (vec-get (vec-of 1 2) 1))
        (prove (= (Num 2) (Num (vec-get (vec-of 1 2) 1))))
        (prove (= (Wrap (vec-of)) (Wrap (vec-empty))))
    "#,
        )
        .unwrap();
}

#[test]
fn builtin_dispatch_rejects_native_bindings_that_disagree_with_wire_sorts() {
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(None, "(sort V (Vec i64)) (sort W (Vec f64))")
        .unwrap();
    let v = graph.get_sort_by_name("V").unwrap().clone();
    let w = graph.get_sort_by_name("W").unwrap().clone();
    let mut program = proto::Program {
        ir_version: 1,
        ..Default::default()
    };
    let output = graph.export_sort(&v, &mut program.sorts).unwrap();
    assert!(
        graph
            .type_info()
            .resolve_builtin(&program, "egglog.core.vec.empty", &[], output, &[w])
            .is_err()
    );
}

#[test]
fn empty_definition_keys_are_rejected_by_import_and_registration() {
    #[derive(Clone)]
    struct Invalid(proto::Program);
    impl Primitive for Invalid {
        fn name(&self) -> &str {
            "invalid-definition"
        }
        fn get_type_constraints(&self, span: &ast::Span) -> Box<dyn constraint::TypeConstraint> {
            builtin::type_constraints(&self.0, span)
        }
        fn builtin_definition(&self) -> Option<(&str, &proto::Program)> {
            Some(("", &self.0))
        }
    }
    impl PurePrim for Invalid {
        fn apply<'a, 'db>(&self, _: PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
            args.first().copied()
        }
    }
    let mut graph = EGraph::default();
    let sort = graph.get_sort_by_name("i64").unwrap().clone();
    let mut definition = builtin::closed_signature(
        "test.valid",
        "invalid-definition",
        &[("value", sort.clone())],
        &[],
        sort,
    )
    .unwrap();
    let Some(proto::declaration::Kind::HostPrimitive(primitive)) =
        &mut definition.declarations.last_mut().unwrap().kind
    else {
        unreachable!()
    };
    primitive.name.clear();
    let imported = builtin::import_definition(
        &definition,
        &mut proto::Program {
            ir_version: 1,
            ..Default::default()
        },
    );
    let primitive = Invalid(definition);
    let mut destination = proto::Program::default();
    let sort = graph.get_sort_by_name("i64").unwrap().clone();
    assert_eq!(
        primitive
            .lower_builtin_call(
                graph.type_info(),
                (&[], &sort),
                &[],
                0,
                None,
                &mut destination
            )
            .unwrap(),
        None
    );
    assert_eq!(destination, proto::Program::default());
    graph.add_pure_primitive(primitive, None);
    assert!(imported.is_err(), "empty key must not import");
    assert!(
        graph.type_info().builtin_catalog().is_err(),
        "empty key must not register"
    );
}

#[test]
fn canonical_signature_drives_native_checking_and_export() {
    #[derive(Clone)]
    struct Identity(Arc<proto::Program>);
    impl Primitive for Identity {
        fn name(&self) -> &str {
            "identity"
        }
        fn get_type_constraints(&self, span: &ast::Span) -> Box<dyn constraint::TypeConstraint> {
            builtin::type_constraints(&self.0, span)
        }
        fn builtin_definition(&self) -> Option<(&str, &proto::Program)> {
            Some(("test.identity", &self.0))
        }
    }
    impl PurePrim for Identity {
        fn apply<'a, 'db>(&self, _: PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
            args.first().copied()
        }
    }
    for (sort_name, good, bad) in [("i64", "1", "1.0"), ("f64", "1.0", "1")] {
        let mut graph = EGraph::default();
        let sort = graph.get_sort_by_name(sort_name).unwrap();
        let definition = builtin::closed_signature(
            "test.identity",
            "identity",
            &[("value", sort.clone())],
            &[],
            sort.clone(),
        )
        .unwrap();
        graph.add_pure_primitive(Identity(Arc::new(definition.clone())), None);
        graph
            .parse_and_run_program(None, &format!("(check (= (identity {good}) {good}))"))
            .unwrap();
        assert!(
            graph
                .parse_and_run_program(None, &format!("(identity {bad})"))
                .is_err()
        );
        let mut exported = proto::Program {
            ir_version: 1,
            ..Default::default()
        };
        let catalog = graph.type_info().builtin_catalog().unwrap();
        builtin::import_definition(&definition, &mut exported).unwrap();
        assert!(catalog.definitions.declarations.iter().any(|declaration| matches!(&declaration.kind, Some(proto::declaration::Kind::HostPrimitive(primitive)) if primitive.name == "test.identity")));
        assert_eq!(exported, definition);
    }
}

#[test]
fn duplicate_catalog_identity_does_not_replace_native_aliases() {
    let mut graph = EGraph::default();
    add_literal_prim!(
        &mut graph,
        "one"[id = "test.duplicate"] = |a: i64| -> i64 { a }
    );
    add_literal_prim!(
        &mut graph,
        "two"[id = "test.duplicate"] = |a: i64| -> i64 { a + 1 }
    );
    assert!(graph.type_info().builtin_catalog().is_err());
    graph
        .parse_and_run_program(None, "(check (= (one 1) 1) (= (two 1) 2))")
        .unwrap();
}

#[test]
fn migrated_addition_keeps_native_proof_and_context_behavior() {
    let mut proofs = EGraph::new_with_proofs();
    proofs
        .parse_and_run_program(
            None,
            "(datatype E (Num i64)) (Num (+ 1 2)) (prove (= (Num 3) (Num (+ 1 2))))",
        )
        .unwrap();
    let mut graph = EGraph::default();
    for name in ["egglog.core.i64.add", "egglog.core.f64.add"] {
        let sort = graph
            .get_sort_by_name(if name.contains("i64") { "i64" } else { "f64" })
            .unwrap()
            .clone();
        let types = [sort.clone(), sort.clone(), sort];
        let alias = ResolvedCall::from_resolution(
            "+",
            &types,
            graph.type_info(),
            Context::Pure,
            &ast::Span::Panic,
        )
        .unwrap();
        let keyed = ResolvedCall::from_resolution(
            name,
            &types,
            graph.type_info(),
            Context::Pure,
            &ast::Span::Panic,
        )
        .unwrap();
        assert_eq!(
            alias, keyed,
            "wire key must retain the same native registration identity"
        );
        let primitives = graph.type_info().get_prims(name).unwrap();
        assert_eq!(primitives.len(), 1);
        for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
            assert!(primitives[0].is_valid_in_context(context));
        }
    }
}

#[test]
fn native_builtin_arity_errors_keep_source_aliases() {
    let error = EGraph::default()
        .parse_and_run_program(None, "(+ 1)")
        .unwrap_err()
        .to_string();
    assert!(error.contains('+'));
    assert!(
        !error.contains("egglog.core."),
        "native source diagnostic changed: {error}"
    );
}

#[test]
fn imported_definitions_keep_namespaces_and_diagnostic_labels_separate() {
    let graph = EGraph::default();
    let sort = graph.get_sort_by_name("i64").unwrap().clone();
    let definition = builtin::closed_signature(
        "i64",
        "identity",
        &[("value", sort.clone())],
        &[],
        sort.clone(),
    )
    .unwrap();
    let mut destination = proto::Program {
        ir_version: 1,
        ..Default::default()
    };
    builtin::import_definition(&definition, &mut destination).unwrap();
    assert_eq!(
        destination.declarations.len(),
        2,
        "sort and callable i64 are distinct"
    );

    let definition = builtin::closed_signature(
        "test.identity",
        "identity",
        &[("value", sort.clone())],
        &[],
        sort,
    )
    .unwrap();
    let mut destination = proto::Program {
        ir_version: 1,
        declarations: vec![proto::Declaration {
            kind: Some(proto::declaration::Kind::Function(proto::Function {
                name: "i64".into(),
                output: 0,
                ..Default::default()
            })),
            ..Default::default()
        }],
        ..Default::default()
    };
    builtin::import_definition(&definition, &mut destination).unwrap();
    assert_eq!(destination.declarations.len(), 3);
    let mut relabeled = definition.clone();
    let declaration = relabeled.declarations.last_mut().unwrap();
    let Some(proto::declaration::Kind::HostPrimitive(primitive)) = &mut declaration.kind else {
        unreachable!()
    };
    let Some(proto::host_primitive::Typing::Signature(signature)) = &mut primitive.typing else {
        unreachable!()
    };
    signature.inputs[0].name = "renamed_diagnostic_label".into();
    builtin::import_definition(&relabeled, &mut destination).unwrap();
    assert_eq!(destination.declarations.len(), 3);

    // Existing presentation information remains immutable.
    relabeled
        .declarations
        .last_mut()
        .unwrap()
        .bindings
        .as_mut()
        .unwrap()
        .egglog
        .as_mut()
        .unwrap()
        .views[0]
        .symbol = "different".into();
    assert!(builtin::import_definition(&relabeled, &mut destination).is_err());
    let Some(proto::declaration::Kind::Function(function)) = &mut destination.declarations[0].kind
    else {
        unreachable!()
    };
    function.name = "test.identity".into();
    assert!(builtin::import_definition(&definition, &mut destination).is_err());
}
