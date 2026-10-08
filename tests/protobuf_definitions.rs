use egglog::{builtin::definitions::reconcile_declarations, proto as pb};

fn constant_definition() -> pb::Program {
    let mut p = box_definition();
    p.nodes.clear();
    let Some(pb::declaration::Kind::Constructor(c)) = &mut p.declarations[1].kind else {
        unreachable!()
    };
    c.inputs.clear();
    let Some(pb::declaration::Kind::Function(f)) = &mut p.declarations[2].kind else {
        unreachable!()
    };
    f.merge = None;
    p.declarations.push(pb::Declaration {
        kind: Some(pb::declaration::Kind::HostPrimitive(pb::HostPrimitive {
            name: "test.constant".into(),
            typing: Some(pb::host_primitive::Typing::Signature(
                pb::GenericSignature {
                    output: Some(0),
                    ..Default::default()
                },
            )),
        })),
        ..Default::default()
    });
    for (d, name) in p.declarations[1..]
        .iter_mut()
        .zip(["BOX", "CURRENT", "HOST"])
    {
        d.bindings = Some(pb::CallableBindings {
            python: Some(pb::PythonBindings {
                views: vec![pb::PythonCallable {
                    kind: pb::PythonCallKind::Constant.into(),
                    path: vec!["test".into(), name.into()],
                    ..Default::default()
                }],
            }),
            ..Default::default()
        });
    }
    p
}

#[test]
fn constant_views_install_resupply_and_freeze_without_evaluation() {
    let source = constant_definition();
    let mut absent = source.clone();
    for d in &mut absent.declarations {
        d.bindings = None;
    }
    let mut destination = pb::Program::default();
    reconcile_declarations(&absent, &mut destination).unwrap();
    assert_eq!(
        reconcile_declarations(&source, &mut destination).unwrap(),
        [false; 4]
    );
    let frozen = destination.clone();
    for _ in 0..3 {
        reconcile_declarations(&source, &mut destination).unwrap();
        reconcile_declarations(&absent, &mut destination).unwrap();
        assert_eq!(destination, frozen);
    }
    let mut function = source;
    function.declarations[2]
        .bindings
        .as_mut()
        .unwrap()
        .python
        .as_mut()
        .unwrap()
        .views[0]
        .kind = pb::PythonCallKind::Function.into();
    let mut independent = pb::Program::default();
    reconcile_declarations(&function, &mut independent).unwrap();
    assert_ne!(independent, frozen, "nullary FUNCTION is not CONSTANT");
    assert!(reconcile_declarations(&function, &mut destination).is_err());
    assert_eq!(
        destination, frozen,
        "first supplied Python metadata is frozen"
    );
}

#[test]
fn empty_constant_names_preserve_exact_metadata_and_freeze_transactionally() {
    for path in [vec!["".to_string()], vec!["example".into(), "".into()]] {
        let mut source = constant_definition();
        source.declarations[2]
            .bindings
            .as_mut()
            .unwrap()
            .python
            .as_mut()
            .unwrap()
            .views[0]
            .path = path;
        let mut absent = source.clone();
        absent.declarations[2].bindings = None;
        let mut destination = pb::Program::default();
        reconcile_declarations(&absent, &mut destination).unwrap();
        let mut independent = destination.clone();
        reconcile_declarations(&source, &mut destination).unwrap();
        assert_eq!(
            destination.declarations[2].bindings,
            source.declarations[2].bindings
        );
        let frozen = destination.clone();
        reconcile_declarations(&source, &mut destination).unwrap();
        reconcile_declarations(&absent, &mut destination).unwrap();
        assert_eq!(destination, frozen);

        let mut renamed = source.clone();
        renamed.declarations[2]
            .bindings
            .as_mut()
            .unwrap()
            .python
            .as_mut()
            .unwrap()
            .views[0]
            .path
            .last_mut()
            .unwrap()
            .push_str("named");
        assert!(reconcile_declarations(&renamed, &mut destination).is_err());
        assert_eq!(
            destination, frozen,
            "empty and nonempty names must not coalesce"
        );
        reconcile_declarations(&renamed, &mut independent).unwrap();
        let independent_frozen = independent.clone();
        assert!(reconcile_declarations(&source, &mut independent).is_err());
        assert_eq!(
            independent, independent_frozen,
            "clone freezes its own first supply"
        );

        let mut function = source.clone();
        function.declarations[2]
            .bindings
            .as_mut()
            .unwrap()
            .python
            .as_mut()
            .unwrap()
            .views[0]
            .kind = pb::PythonCallKind::Function.into();
        let mut fresh = pb::Program::default();
        assert!(reconcile_declarations(&function, &mut fresh).is_err());
        assert_eq!(
            fresh,
            pb::Program::default(),
            "FUNCTION does not admit an empty final name"
        );
        for bad in [
            vec![],
            vec!["", ""],
            vec!["", "value"],
            vec!["example", "", "value"],
            vec!["example", "", ""],
        ] {
            let mut invalid = source.clone();
            invalid.declarations[2]
                .bindings
                .as_mut()
                .unwrap()
                .python
                .as_mut()
                .unwrap()
                .views[0]
                .path = bad.into_iter().map(str::to_owned).collect();
            assert!(reconcile_declarations(&invalid, &mut fresh).is_err());
            assert_eq!(fresh, pb::Program::default());
            assert!(reconcile_declarations(&invalid, &mut destination).is_err());
            assert_eq!(destination, frozen);
        }
        let Some(pb::declaration::Kind::Function(f)) = &mut source.declarations[2].kind else {
            unreachable!()
        };
        f.name.clear();
        assert!(reconcile_declarations(&source, &mut fresh).is_err());
        assert_eq!(
            fresh,
            pb::Program::default(),
            "semantic names remain nonempty"
        );
    }
}

#[test]
fn constant_views_reject_malformed_forms_and_signatures_transactionally() {
    let source = constant_definition();
    let mut destination = pb::Program::default();
    reconcile_declarations(&source, &mut destination).unwrap();
    for bad in 0..15 {
        let mut candidate = source.clone();
        let view = &mut candidate.declarations[3]
            .bindings
            .as_mut()
            .unwrap()
            .python
            .as_mut()
            .unwrap()
            .views[0];
        match bad {
            0 => view.path.clear(),
            1 => view.path.insert(0, String::new()),
            2 => {
                view.owner = Some(pb::BindingOwner {
                    kind: Some(pb::binding_owner::Kind::Sort(0)),
                })
            }
            3 => view.receiver = Some(0),
            4 => view.params.push(pb::PythonParameter {
                core_input: Some(0),
                name: "x".into(),
                ..Default::default()
            }),
            5 => view.mutates = Some(0),
            6 => view.kind = 99,
            7..=10 => {
                let Some(pb::declaration::Kind::HostPrimitive(h)) =
                    &mut candidate.declarations[3].kind
                else {
                    unreachable!()
                };
                let Some(pb::host_primitive::Typing::Signature(s)) = &mut h.typing else {
                    unreachable!()
                };
                match bad {
                    7 => s.inputs.push(pb::Arg {
                        name: "x".into(),
                        sort: 0,
                    }),
                    8 => {
                        s.varargs = Some(pb::Arg {
                            name: "xs".into(),
                            sort: 0,
                        })
                    }
                    9 => s.type_params.push("Undetermined".into()),
                    _ => s.output = Some(99),
                }
            }
            11 => {
                let Some(pb::declaration::Kind::Constructor(c)) =
                    &mut candidate.declarations[1].kind
                else {
                    unreachable!()
                };
                c.inputs.push(pb::Arg {
                    name: "x".into(),
                    sort: 0,
                });
            }
            12 => {
                let Some(pb::declaration::Kind::Function(f)) = &mut candidate.declarations[2].kind
                else {
                    unreachable!()
                };
                f.inputs.push(pb::Arg {
                    name: "x".into(),
                    sort: 0,
                });
            }
            13 => candidate.sorts[0].kind = Some(pb::sort::Kind::Var(0)),
            _ => {
                let Some(pb::declaration::Kind::HostPrimitive(h)) =
                    &mut candidate.declarations[3].kind
                else {
                    unreachable!()
                };
                h.typing = Some(pb::host_primitive::Typing::Application(
                    pb::FunctionApplication {},
                ));
            }
        }
        let before = destination.clone();
        assert!(
            reconcile_declarations(&candidate, &mut destination).is_err(),
            "bad case {bad}"
        );
        assert_eq!(destination, before, "bad case {bad}");
        let mut fresh = pb::Program::default();
        assert!(
            reconcile_declarations(&candidate, &mut fresh).is_err(),
            "fresh bad case {bad}"
        );
        assert_eq!(fresh, pb::Program::default());
    }
}

fn box_definition() -> pb::Program {
    pb::Program {
        ir_version: 1,
        sorts: vec![
            pb::Sort {
                kind: Some(pb::sort::Kind::Family(pb::HostSort {
                    name: "i64".into(),
                    args: vec![],
                })),
                ..Default::default()
            },
            pb::Sort {
                kind: Some(pb::sort::Kind::Eq("Box".into())),
                ..Default::default()
            },
        ],
        nodes: vec![
            pb::Node {
                sort_id: 0,
                kind: Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                    value: Some(pb::primitive_value::Value::I64(4)),
                })),
                ..Default::default()
            },
            pb::Node {
                sort_id: 0,
                kind: Some(pb::node::Kind::Call(pb::Call {
                    func: "current".into(),
                    args: vec![],
                })),
                ..Default::default()
            },
            pb::Node {
                sort_id: 0,
                kind: Some(pb::node::Kind::Var("new".into())),
                ..Default::default()
            },
        ],
        declarations: vec![
            pb::Declaration {
                kind: Some(pb::declaration::Kind::EqSort(pb::EqSort {
                    name: "Box".into(),
                    bindings: Some(pb::SortBindings {
                        python: Some(pb::TypeBinding {
                            path: vec!["test".into(), "Box".into()],
                            ..Default::default()
                        }),
                        ..Default::default()
                    }),
                })),
                ..Default::default()
            },
            pb::Declaration {
                kind: Some(pb::declaration::Kind::Constructor(pb::Constructor {
                    name: "Box".into(),
                    inputs: vec![pb::Arg {
                        sort: 0,
                        name: "value".into(),
                    }],
                    output: 1,
                    ..Default::default()
                })),
                bindings: Some(pb::CallableBindings {
                    python: Some(pb::PythonBindings {
                        views: vec![pb::PythonCallable {
                            kind: pb::PythonCallKind::Initializer.into(),
                            owner: Some(pb::BindingOwner {
                                kind: Some(pb::binding_owner::Kind::Sort(1)),
                            }),
                            params: vec![pb::PythonParameter {
                                core_input: Some(0),
                                name: "value".into(),
                                default_expr: Some(0),
                            }],
                            ..Default::default()
                        }],
                    }),
                    rust: Some(pb::RustBindings {
                        views: vec![pb::RustCallable {
                            path: vec!["new".into()],
                            owner: Some(pb::BindingOwner {
                                kind: Some(pb::binding_owner::Kind::Sort(1)),
                            }),
                            params: vec![pb::RustParameter {
                                core_input: Some(0),
                                name: "value".into(),
                                borrowed: false,
                            }],
                            ..Default::default()
                        }],
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },
            pb::Declaration {
                kind: Some(pb::declaration::Kind::Function(pb::Function {
                    name: "current".into(),
                    inputs: vec![],
                    output: 0,
                    merge: Some(2),
                })),
                ..Default::default()
            },
        ],
        ..Default::default()
    }
}

#[test]
fn declaration_closures_relocate_and_resupply_structurally() {
    let source = box_definition();
    let mut destination = pb::Program::default();
    assert_eq!(
        reconcile_declarations(&source, &mut destination).unwrap(),
        [true, true, true]
    );
    let installed = destination.clone();
    let mut permuted = source.clone();
    permuted.sorts.reverse();
    permuted.nodes.reverse();
    for n in &mut permuted.nodes {
        n.sort_id = 1 - n.sort_id;
    }
    let Some(pb::declaration::Kind::Constructor(c)) = &mut permuted.declarations[1].kind else {
        unreachable!()
    };
    c.inputs[0].sort = 1;
    c.inputs[0].name = "diagnostic alpha rename".into();
    c.output = 0;
    let b = permuted.declarations[1].bindings.as_mut().unwrap();
    b.python.as_mut().unwrap().views[0]
        .owner
        .as_mut()
        .unwrap()
        .kind = Some(pb::binding_owner::Kind::Sort(0));
    b.python.as_mut().unwrap().views[0].params[0].default_expr = Some(2);
    b.rust.as_mut().unwrap().views[0]
        .owner
        .as_mut()
        .unwrap()
        .kind = Some(pb::binding_owner::Kind::Sort(0));
    let Some(pb::declaration::Kind::Function(f)) = &mut permuted.declarations[2].kind else {
        unreachable!()
    };
    f.output = 1;
    f.merge = Some(0);
    permuted.declarations.reverse();
    assert_eq!(
        reconcile_declarations(&permuted, &mut destination).unwrap(),
        [false, false, false]
    );
    assert_eq!(
        destination, installed,
        "compatible resupply must not retain unreachable template copies"
    );
    let before = destination.clone();
    let Some(pb::declaration::Kind::Function(f)) = &mut permuted.declarations[0].kind else {
        unreachable!()
    };
    f.merge = None;
    assert!(reconcile_declarations(&permuted, &mut destination).is_err());
    assert_eq!(destination, before);
}

#[test]
fn default_validation_is_contextual_typed_and_inert() {
    for bad in 0..6 {
        let mut source = box_definition();
        let root = match bad {
            0 => 2, // new is bound in the merge only.
            1 => {
                source.nodes[0].sort_id = 1;
                0
            }
            2 => {
                source.nodes[0].kind = Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                    value: Some(pb::primitive_value::Value::Bool(true)),
                }));
                0
            }
            3 => {
                let Some(pb::node::Kind::Call(c)) = &mut source.nodes[1].kind else {
                    unreachable!()
                };
                c.args.push(0);
                1
            }
            4 => {
                source.nodes[1].sort_id = 1;
                1
            }
            _ => {
                let Some(pb::node::Kind::Call(c)) = &mut source.nodes[1].kind else {
                    unreachable!()
                };
                c.func = "unknown".into();
                1
            }
        };
        source.declarations[1]
            .bindings
            .as_mut()
            .unwrap()
            .python
            .as_mut()
            .unwrap()
            .views[0]
            .params[0]
            .default_expr = Some(root);
        let mut destination = pb::Program::default();
        assert!(
            reconcile_declarations(&source, &mut destination).is_err(),
            "bad default {bad}"
        );
        assert_eq!(destination, pb::Program::default());
    }
    let mut source = box_definition();
    source.declarations[1]
        .bindings
        .as_mut()
        .unwrap()
        .python
        .as_mut()
        .unwrap()
        .views[0]
        .params[0]
        .default_expr = Some(1);
    // No evaluator is available here: a missing current() row is still valid code.
    reconcile_declarations(&source, &mut pb::Program::default()).unwrap();
}

#[test]
fn all_default_roots_preserve_union_identity_but_not_ordinary_sharing() {
    let mut source = box_definition();
    source.nodes.push(pb::Node {
        sort_id: 0,
        kind: Some(pb::node::Kind::Union(pb::Union {
            members: vec![0, 0],
        })),
        ..Default::default()
    });
    let Some(pb::declaration::Kind::Constructor(c)) = &mut source.declarations[1].kind else {
        unreachable!()
    };
    c.inputs.push(c.inputs[0].clone());
    let b = source.declarations[1].bindings.as_mut().unwrap();
    b.rust = None;
    let view = &mut b.python.as_mut().unwrap().views[0];
    view.params[0].default_expr = Some(3);
    view.params.push(pb::PythonParameter {
        core_input: Some(1),
        name: "other".into(),
        default_expr: Some(3),
    });
    let mut destination = pb::Program::default();
    reconcile_declarations(&source, &mut destination).unwrap();
    source.nodes.push(source.nodes[0].clone());
    let Some(pb::node::Kind::Union(u)) = &mut source.nodes[3].kind else {
        unreachable!()
    };
    u.members[1] = 4;
    reconcile_declarations(&source, &mut destination).unwrap();
    source.nodes.push(source.nodes[3].clone());
    source.declarations[1]
        .bindings
        .as_mut()
        .unwrap()
        .python
        .as_mut()
        .unwrap()
        .views[0]
        .params[1]
        .default_expr = Some(5);
    let before = destination.clone();
    assert!(reconcile_declarations(&source, &mut destination).is_err());
    assert_eq!(destination, before);
}

#[test]
fn languages_freeze_independently_and_compatible_imports_do_not_grow() {
    let source = box_definition();
    let mut absent = source.clone();
    absent.declarations[1].bindings = None;
    let mut destination = pb::Program::default();
    reconcile_declarations(&absent, &mut destination).unwrap();
    let mut python = source.clone();
    python.declarations[1].bindings.as_mut().unwrap().rust = None;
    reconcile_declarations(&python, &mut destination).unwrap();
    let mut rust = source.clone();
    rust.declarations[1].bindings.as_mut().unwrap().python = None;
    reconcile_declarations(&rust, &mut destination).unwrap();
    let installed = destination.clone();
    absent.files.push(pb::SourceFile {
        name: "unused".into(),
        contents: None,
    });
    absent.sorts.push(pb::Sort {
        kind: Some(pb::sort::Kind::Eq("Unused".into())),
        ..Default::default()
    });
    for _ in 0..20 {
        reconcile_declarations(&source, &mut destination).unwrap();
        reconcile_declarations(&absent, &mut destination).unwrap();
        assert_eq!(destination, installed);
    }
    python.declarations[1].bindings.as_mut().unwrap().python = Some(pb::PythonBindings::default());
    assert!(reconcile_declarations(&python, &mut destination).is_err());
    assert_eq!(destination, installed);
    let mut first_empty = pb::Program::default();
    reconcile_declarations(&python, &mut first_empty).unwrap();
    assert!(reconcile_declarations(&source, &mut first_empty).is_err());
}

#[test]
fn malformed_surface_mappings_reject_transactionally() {
    for invalid in 0..8 {
        let mut source = box_definition();
        let b = source.declarations[1].bindings.as_mut().unwrap();
        let v = &mut b.python.as_mut().unwrap().views[0];
        match invalid {
            0 => v.params[0].core_input = None,
            1 => v.params.push(v.params[0].clone()),
            2 => v.owner.as_mut().unwrap().kind = Some(pb::binding_owner::Kind::Sort(0)),
            3 => v.receiver = Some(0),
            4 => v.params[0].default_expr = Some(999),
            5 => {
                b.rust.as_mut().unwrap().views[0].receiver = Some(pb::RustReceiver {
                    core_input: Some(0),
                    borrowed: false,
                })
            }
            6 => b.rust.as_mut().unwrap().views[0].borrowed_self = true,
            _ => {
                source.nodes[1].kind = Some(pb::node::Kind::Call(pb::Call {
                    func: "current".into(),
                    args: vec![1],
                }));
                v.params[0].default_expr = Some(1);
            }
        }
        let mut destination = pb::Program::default();
        assert!(
            reconcile_declarations(&source, &mut destination).is_err(),
            "invalid binding {invalid}"
        );
        assert_eq!(destination, pb::Program::default());
    }
}

#[test]
fn default_calls_determine_all_generic_parameters_including_empty_results() {
    for variadic in [false, true] {
        let mut source = box_definition();
        source.sorts.push(pb::Sort {
            kind: Some(pb::sort::Kind::Var(0)),
            ..Default::default()
        });
        source.declarations.push(pb::Declaration {
            kind: Some(pb::declaration::Kind::HostPrimitive(pb::HostPrimitive {
                name: "generic".into(),
                typing: Some(pb::host_primitive::Typing::Signature(
                    pb::GenericSignature {
                        type_params: vec!["T".into()],
                        inputs: vec![],
                        output: Some(0),
                        varargs: variadic.then(|| pb::Arg {
                            sort: 2,
                            name: "tail".into(),
                        }),
                    },
                )),
            })),
            ..Default::default()
        });
        source.nodes[1].kind = Some(pb::node::Kind::Call(pb::Call {
            func: "generic".into(),
            args: vec![],
        }));
        source.declarations[1]
            .bindings
            .as_mut()
            .unwrap()
            .python
            .as_mut()
            .unwrap()
            .views[0]
            .params[0]
            .default_expr = Some(1);
        assert!(
            reconcile_declarations(&source, &mut pb::Program::default()).is_err(),
            "undetermined T with variadic={variadic}"
        );
        if variadic {
            let Some(pb::node::Kind::Call(c)) = &mut source.nodes[1].kind else {
                unreachable!()
            };
            c.args.push(0);
            reconcile_declarations(&source, &mut pb::Program::default()).unwrap();
        }
    }
    use egglog::sort::Presort;
    let mut source = box_definition();
    egglog::builtin::import_definition(
        &egglog::sort::VecSort::builtin_definitions()[0],
        &mut source,
    )
    .unwrap();
    let concrete = source.sorts.len() as u32;
    source.sorts.push(pb::Sort {
        kind: Some(pb::sort::Kind::Family(pb::HostSort {
            name: "Vec".into(),
            args: vec![0],
        })),
        ..Default::default()
    });
    let root = source.nodes.len() as u32;
    source.nodes.push(pb::Node {
        sort_id: concrete,
        kind: Some(pb::node::Kind::Call(pb::Call {
            func: "egglog.core.vec.empty".into(),
            args: vec![],
        })),
        ..Default::default()
    });
    let Some(pb::declaration::Kind::Constructor(c)) = &mut source.declarations[1].kind else {
        unreachable!()
    };
    c.inputs[0].sort = concrete;
    source.declarations[1]
        .bindings
        .as_mut()
        .unwrap()
        .python
        .as_mut()
        .unwrap()
        .views[0]
        .params[0]
        .default_expr = Some(root);
    reconcile_declarations(&source, &mut pb::Program::default()).unwrap();
    let Some(pb::sort::Kind::Family(f)) = &mut source.sorts[concrete as usize].kind else {
        unreachable!()
    };
    f.args.clear();
    assert!(reconcile_declarations(&source, &mut pb::Program::default()).is_err());
}
