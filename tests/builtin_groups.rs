use egglog::{
    proto as pb,
    sort::{F, MapContainer, MapSort, Presort},
    *,
};
use std::{
    any::TypeId,
    sync::atomic::{AtomicUsize, Ordering},
};

// Test-only native family: reuse Map storage, not its handwritten constraints.
// Production Map registration and catalog definitions remain untouched.
struct Grouped;
impl Presort for Grouped {
    fn presort_name() -> &'static str {
        "TestGrouped"
    }
    fn reserved_primitives() -> Vec<&'static str> {
        vec!["grouped"]
    }
    fn builtin_definitions() -> Vec<pb::Program> {
        vec![pb::Program {
            ir_version: 1,
            sorts: vec![
                pb::Sort {
                    kind: Some(pb::sort::Kind::Var(0)),
                    ..Default::default()
                },
                pb::Sort {
                    kind: Some(pb::sort::Kind::Var(1)),
                    ..Default::default()
                },
                pb::Sort {
                    kind: Some(pb::sort::Kind::Family(pb::HostSort {
                        name: "TestGrouped".into(),
                        args: vec![0, 1],
                    })),
                    ..Default::default()
                },
                pb::Sort {
                    kind: Some(pb::sort::Kind::Family(pb::HostSort {
                        name: "i64".into(),
                        args: vec![],
                    })),
                    ..Default::default()
                },
            ],
            declarations: vec![
                pb::Declaration {
                    kind: Some(pb::declaration::Kind::HostSortFamily(pb::HostSortFamily {
                        name: "TestGrouped".into(),
                        arity: 2,
                        ..Default::default()
                    })),
                    ..Default::default()
                },
                pb::Declaration {
                    kind: Some(pb::declaration::Kind::HostSortFamily(pb::HostSortFamily {
                        name: "i64".into(),
                        ..Default::default()
                    })),
                    ..Default::default()
                },
                pb::Declaration {
                    kind: Some(pb::declaration::Kind::HostPrimitive(pb::HostPrimitive {
                        name: "test.grouped".into(),
                        typing: Some(pb::host_primitive::Typing::Signature(
                            pb::GenericSignature {
                                type_params: vec!["K".into(), "V".into()],
                                inputs: vec![pb::Arg {
                                    name: "prefix".into(),
                                    sort: 3,
                                }],
                                varargs: vec![
                                    pb::Arg {
                                        name: "key".into(),
                                        sort: 0,
                                    },
                                    pb::Arg {
                                        name: "value".into(),
                                        sort: 1,
                                    },
                                ],
                                output: Some(2),
                            },
                        )),
                    })),
                    bindings: Some(pb::CallableBindings {
                        egglog: Some(pb::EgglogBindings {
                            views: vec![pb::EgglogCallable {
                                symbol: "grouped".into(),
                                datatype_member: false,
                            }],
                        }),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }]
    }
    fn make_sort(
        types: &mut TypeInfo,
        name: String,
        args: &[ast::Expr],
        span: ast::Span,
    ) -> Result<ArcSort, TypeError> {
        let native = MapSort::make_sort(types, name, args, span)?;
        types
            .register_builtin_sort(Self::presort_name(), native.clone(), native.inner_sorts())
            .unwrap();
        Ok(native)
    }
}

static BODY: AtomicUsize = AtomicUsize::new(0);
static VALIDATOR: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone)]
struct GroupedPrimitive(builtin::BuiltinInstance);
impl Primitive for GroupedPrimitive {
    fn name(&self) -> &str {
        "grouped"
    }
    fn get_type_constraints(&self, span: &ast::Span) -> Box<dyn constraint::TypeConstraint> {
        self.0.type_constraints(span)
    }
    fn builtin_instance(&self) -> Option<&builtin::BuiltinInstance> {
        Some(&self.0)
    }
}
impl PurePrim for GroupedPrimitive {
    fn apply<'a, 'db>(&self, _: PureState<'a, 'db>, _: &[Value]) -> Option<Value> {
        BODY.fetch_add(1, Ordering::SeqCst);
        panic!("typing/export must not execute this fixture")
    }
}

fn grouped_graph() -> EGraph {
    let mut graph = EGraph::default();
    graph
        .type_info()
        .add_presort::<Grouped>(ast::Span::Panic)
        .unwrap();
    let before = graph.type_info().builtin_catalog().unwrap().definitions;
    graph.parse_and_run_program(None, "(sort G (TestGrouped i64 f64)) (sort H (TestGrouped f64 i64)) (sort Same (TestGrouped i64 i64)) (function g () G :merge new)").unwrap();
    for name in ["G", "H", "Same"] {
        let native = graph.get_sort_by_name(name).unwrap().clone();
        let instance = graph
            .type_info()
            .instantiate_builtin("test.grouped", &native)
            .unwrap();
        instance
            .check_abi(
                &[None, None, None],
                Some((TypeId::of::<MapContainer>(), true)),
                true,
            )
            .unwrap();
        graph.add_pure_primitive(
            GroupedPrimitive(instance),
            Some(std::sync::Arc::new(|_: &mut TermDag, _: &[TermId]| {
                VALIDATOR.fetch_add(1, Ordering::SeqCst);
                None
            })),
        );
    }
    assert_eq!(
        graph.type_info().builtin_catalog().unwrap().definitions,
        before
    );
    graph
}

#[test]
fn grouped_native_and_exact_key_typing_share_the_canonical_pattern() {
    let mut graph = grouped_graph();
    for text in [
        "(check (= (grouped 7) (g)))",
        "(check (= (grouped 7 1 2.0) (g)))",
        "(check (= (grouped 7 1 2.0 3 4.0) (g)))",
    ] {
        let command = graph.parse_program(None, text).unwrap().remove(0);
        graph.resolve_command_before_proofs(command).unwrap();
    }
    for text in [
        "(grouped)",
        "(grouped 7 1)",
        "(grouped 7 1 2.0 3)",
        "(check (= (grouped 7 1.0 2) (g)))",
        "(check (= (grouped 7 1 2.0 3 4) (g)))",
        "(grouped 7.0)",
    ] {
        let command = graph.parse_program(None, text).unwrap().remove(0);
        assert!(
            graph.resolve_command_before_proofs(command).is_err(),
            "{text}"
        );
    }
    let i = graph.get_sort_by_name("i64").unwrap().clone();
    let f = graph.get_sort_by_name("f64").unwrap().clone();
    let g = graph.get_sort_by_name("G").unwrap().clone();
    let h = graph.get_sort_by_name("H").unwrap().clone();
    let same = graph.get_sort_by_name("Same").unwrap().clone();
    for (args, output, valid) in [
        (vec![i.clone()], g.clone(), true),
        (vec![i.clone()], h.clone(), true),
        (
            vec![i.clone(), i.clone(), f.clone(), i.clone(), f.clone()],
            g.clone(),
            true,
        ),
        (vec![i.clone(), i.clone(), i.clone()], same.clone(), true),
        (vec![i.clone(), i.clone()], same, false),
        (vec![], g.clone(), false),
        (vec![i.clone(), i.clone(), f.clone()], h, false),
        (
            vec![i.clone(), i.clone(), f.clone(), f.clone(), i.clone()],
            g.clone(),
            false,
        ),
        (vec![i.clone(), i.clone(), f], i, false),
    ] {
        let mut program = pb::Program {
            ir_version: 1,
            ..Default::default()
        };
        let arguments = args
            .iter()
            .map(|s| graph.export_sort(s, &mut program.sorts).unwrap())
            .collect::<Vec<_>>();
        let result = graph.export_sort(&output, &mut program.sorts).unwrap();
        let native = args.into_iter().chain([output]).collect::<Vec<_>>();
        let resolution = graph.type_info().resolve_builtin(
            &program,
            "test.grouped",
            &arguments,
            result,
            &native,
        );
        assert_eq!(resolution.is_ok(), valid, "{resolution:?}");
        if let Ok(key) = resolution {
            for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
                assert_eq!(
                    ResolvedCall::from_resolution(
                        "grouped",
                        &native,
                        graph.type_info(),
                        context,
                        &ast::Span::Panic
                    )
                    .unwrap(),
                    ResolvedCall::from_resolution(
                        &key,
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
    assert_eq!(BODY.load(Ordering::SeqCst), 0);
    assert_eq!(VALIDATOR.load(Ordering::SeqCst), 0);
}

#[test]
fn grouped_abi_checks_every_prefix_pattern_and_result_conversion() {
    let mut graph = grouped_graph();
    let g = graph.get_sort_by_name("G").unwrap().clone();
    let instance = graph
        .type_info()
        .instantiate_builtin("test.grouped", &g)
        .unwrap();
    let i = Some((TypeId::of::<i64>(), false));
    let f = Some((TypeId::of::<F>(), false));
    let output = Some((TypeId::of::<MapContainer>(), true));
    instance.check_abi(&[i, i, f], output, true).unwrap();
    for (inputs, result, variadic) in [
        (vec![i, i, f], output, false),
        (vec![i, f], output, true),
        (vec![i, f, i], output, true),
        (vec![i, i, f], i, true),
        (
            vec![i, i, f],
            Some((TypeId::of::<MapContainer>(), false)),
            true,
        ),
    ] {
        assert!(instance.check_abi(&inputs, result, variadic).is_err());
    }
    let before = graph.type_info().builtin_catalog().unwrap().definitions;
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        add_primitive!(&mut graph, "bad-group" [instance = instance] = [mut values: #] -> # { values.next().unwrap() });
    }));
    assert!(
        panic.is_err(),
        "homogeneous iterator cannot implement a two-wide tail"
    );
    assert_eq!(
        graph.type_info().builtin_catalog().unwrap().definitions,
        before
    );
    assert_eq!(BODY.load(Ordering::SeqCst), 0);
    assert_eq!(VALIDATOR.load(Ordering::SeqCst), 0);
}

#[test]
fn grouped_signature_identity_preserves_width_and_order_after_relocation() {
    let mut graph = grouped_graph();
    let mut definition = Grouped::builtin_definitions().remove(0);
    let mut destination = pb::Program::default();
    builtin::import_definition(&definition, &mut destination).unwrap();
    let before = destination.clone();
    // Rotate the arena and alpha-rename every diagnostic label, not the binder.
    definition.sorts.rotate_left(1);
    for sort in &mut definition.sorts {
        if let Some(pb::sort::Kind::Family(family)) = &mut sort.kind {
            for index in &mut family.args {
                *index = (*index + 3) % 4;
            }
        }
    }
    let Some(pb::declaration::Kind::HostPrimitive(primitive)) =
        &mut definition.declarations[2].kind
    else {
        unreachable!()
    };
    let Some(pb::host_primitive::Typing::Signature(signature)) = &mut primitive.typing else {
        unreachable!()
    };
    signature.type_params = vec!["Renamed".into(), "Renamed".into()];
    signature.output = signature.output.map(|i| (i + 3) % 4);
    for arg in signature.inputs.iter_mut().chain(&mut signature.varargs) {
        arg.sort = (arg.sort + 3) % 4;
        arg.name = "renamed".into();
    }
    builtin::import_definition(&definition, &mut destination).unwrap();
    assert_eq!(destination, before);
    let Some(pb::declaration::Kind::HostPrimitive(primitive)) = &definition.declarations[2].kind
    else {
        unreachable!()
    };
    graph
        .type_info()
        .check_builtin_signature(&definition, primitive)
        .unwrap();
    for mutation in 0..4 {
        let mut invalid = definition.clone();
        let Some(pb::declaration::Kind::HostPrimitive(primitive)) =
            &mut invalid.declarations[2].kind
        else {
            unreachable!()
        };
        let Some(pb::host_primitive::Typing::Signature(signature)) = &mut primitive.typing else {
            unreachable!()
        };
        match mutation {
            0 => {
                signature.varargs.pop();
            }
            1 => signature.varargs.reverse(),
            2 => signature.varargs.push(signature.varargs[0].clone()),
            _ => signature.varargs[1] = signature.varargs[0].clone(),
        }
        let Some(pb::declaration::Kind::HostPrimitive(primitive)) = &invalid.declarations[2].kind
        else {
            unreachable!()
        };
        assert!(
            graph
                .type_info()
                .check_builtin_signature(&invalid, primitive)
                .is_err()
        );
        assert!(builtin::import_definition(&invalid, &mut destination).is_err());
        assert_eq!(destination, before);
    }
    // Repeating the same type twice is not a homogeneous one-position tail.
    let Some(pb::declaration::Kind::HostPrimitive(primitive)) =
        &mut definition.declarations[2].kind
    else {
        unreachable!()
    };
    let Some(pb::host_primitive::Typing::Signature(signature)) = &mut primitive.typing else {
        unreachable!()
    };
    signature.varargs[1] = signature.varargs[0].clone();
    let mut equal_positions = pb::Program::default();
    builtin::import_definition(&definition, &mut equal_positions).unwrap();
    let frozen = equal_positions.clone();
    let Some(pb::declaration::Kind::HostPrimitive(primitive)) =
        &mut definition.declarations[2].kind
    else {
        unreachable!()
    };
    let Some(pb::host_primitive::Typing::Signature(signature)) = &mut primitive.typing else {
        unreachable!()
    };
    signature.varargs.pop();
    assert!(builtin::import_definition(&definition, &mut equal_positions).is_err());
    assert_eq!(equal_positions, frozen);
}

#[test]
fn grouped_defaults_and_surface_slots_validate_without_evaluation() {
    let mut source = Grouped::builtin_definitions().remove(0);
    source.sorts.extend([
        pb::Sort {
            kind: Some(pb::sort::Kind::Family(pb::HostSort {
                name: "f64".into(),
                args: vec![],
            })),
            ..Default::default()
        },
        pb::Sort {
            kind: Some(pb::sort::Kind::Family(pb::HostSort {
                name: "TestGrouped".into(),
                args: vec![3, 4],
            })),
            ..Default::default()
        },
    ]);
    source.nodes = vec![
        pb::Node {
            sort_id: 3,
            kind: Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::I64(7)),
            })),
            ..Default::default()
        },
        pb::Node {
            sort_id: 4,
            kind: Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::F64Bits(2.0f64.to_bits())),
            })),
            ..Default::default()
        },
        pb::Node {
            sort_id: 5,
            kind: Some(pb::node::Kind::Call(pb::Call {
                func: "test.grouped".into(),
                args: vec![0, 0, 1, 0, 1],
            })),
            ..Default::default()
        },
    ];
    source.declarations[2].bindings.as_mut().unwrap().python = Some(pb::PythonBindings {
        views: vec![pb::PythonCallable {
            kind: pb::PythonCallKind::Function.into(),
            path: vec!["test".into(), "pairs".into()],
            params: vec![
                pb::PythonParameter {
                    core_input: Some(0),
                    name: "prefix".into(),
                    ..Default::default()
                },
                pb::PythonParameter {
                    core_input: Some(1),
                    name: "pairs".into(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }],
    });
    source.declarations[2].bindings.as_mut().unwrap().rust = Some(pb::RustBindings {
        views: vec![pb::RustCallable {
            path: vec!["test".into(), "pairs".into()],
            params: vec![
                pb::RustParameter {
                    core_input: Some(0),
                    name: "prefix".into(),
                    borrowed: false,
                },
                pb::RustParameter {
                    core_input: Some(1),
                    name: "pairs".into(),
                    borrowed: false,
                },
            ],
            ..Default::default()
        }],
    });
    source.declarations.push(pb::Declaration {
        kind: Some(pb::declaration::Kind::Function(pb::Function {
            name: "consume".into(),
            inputs: vec![pb::Arg {
                sort: 5,
                name: "value".into(),
            }],
            output: 3,
            ..Default::default()
        })),
        bindings: Some(pb::CallableBindings {
            python: Some(pb::PythonBindings {
                views: vec![pb::PythonCallable {
                    kind: pb::PythonCallKind::Function.into(),
                    path: vec!["test".into(), "consume".into()],
                    params: vec![pb::PythonParameter {
                        core_input: Some(0),
                        name: "value".into(),
                        default_expr: Some(2),
                    }],
                    ..Default::default()
                }],
            }),
            ..Default::default()
        }),
        ..Default::default()
    });
    for args in [vec![0], vec![0, 0, 1], vec![0, 0, 1, 0, 1]] {
        let Some(pb::node::Kind::Call(call)) = &mut source.nodes[2].kind else {
            unreachable!()
        };
        call.args = args;
        builtin::definitions::reconcile_declarations(&source, &mut pb::Program::default()).unwrap();
    }
    for mutation in 0..12 {
        let mut invalid = source.clone();
        match mutation {
            0..=3 => {
                let Some(pb::node::Kind::Call(call)) = &mut invalid.nodes[2].kind else {
                    unreachable!()
                };
                call.args =
                    [vec![], vec![0, 0], vec![0, 1, 0], vec![0, 0, 1, 0, 0]][mutation].clone();
            }
            4 => invalid.nodes[2].sort_id = 3,
            5 => {
                let Some(pb::declaration::Kind::HostPrimitive(primitive)) =
                    &mut invalid.declarations[2].kind
                else {
                    unreachable!()
                };
                let Some(pb::host_primitive::Typing::Signature(signature)) = &mut primitive.typing
                else {
                    unreachable!()
                };
                signature.type_params.push("Undetermined".into());
            }
            _ => {
                let binding = invalid.declarations[2].bindings.as_mut().unwrap();
                match mutation {
                    6 => binding.python.as_mut().unwrap().views[0].params[1].default_expr = Some(0),
                    7 => binding.python.as_mut().unwrap().views[0].params[1].core_input = Some(2),
                    8 => binding.python.as_mut().unwrap().views[0].params.reverse(),
                    9 => binding.rust.as_mut().unwrap().views[0].params[1].core_input = Some(2),
                    10 => binding.rust.as_mut().unwrap().views[0].params.reverse(),
                    _ => binding.python.as_mut().unwrap().views[0].receiver = Some(1),
                }
            }
        }
        let mut destination = pb::Program::default();
        assert!(
            builtin::definitions::reconcile_declarations(&invalid, &mut destination).is_err(),
            "mutation {mutation}"
        );
        assert_eq!(destination, pb::Program::default());
    }
    assert_eq!(BODY.load(Ordering::SeqCst), 0);
    assert_eq!(VALIDATOR.load(Ordering::SeqCst), 0);
}
