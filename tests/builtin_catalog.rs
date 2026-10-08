use egglog::sort::{PairContainer, Presort, VecContainer, VecSort};
use egglog::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[test]
fn vec_catalog_has_generic_rust_views_without_python_views() {
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
    assert!(binding.python.is_none());
    let rust = binding.rust.as_ref().unwrap();
    assert_eq!(
        rust.path,
        ["egglog_experimental", "typed", "builtins", "Vec"]
    );
    assert_eq!(rust.type_params, ["T"]);
    for method in ["empty", "of", "get"] {
        let declaration = catalog.declarations.iter().find(|d| matches!(&d.kind, Some(proto::declaration::Kind::HostPrimitive(p)) if p.name == format!("egglog.core.vec.{method}"))).unwrap();
        let bindings = declaration.bindings.as_ref().unwrap();
        assert!(bindings.python.is_none());
        assert_eq!(
            bindings.egglog.as_ref().unwrap().views[0].symbol,
            format!("vec-{method}")
        );
        let rust = &bindings
            .rust
            .as_ref()
            .expect("Vec callable presentation")
            .views[0];
        assert_eq!(rust.path, [method]);
        let Some(proto::binding_owner::Kind::Sort(owner)) = rust.owner.unwrap().kind else {
            panic!("sort owner")
        };
        let Some(proto::sort::Kind::Family(owner)) = &catalog.sorts[owner as usize].kind else {
            panic!("Vec owner")
        };
        assert_eq!(owner.name, "Vec");
        assert!(matches!(
            catalog.sorts[owner.args[0] as usize].kind,
            Some(proto::sort::Kind::Var(0))
        ));
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
    }
}

#[test]
fn family_metadata_is_optional_and_frozen_per_language() {
    let graph = EGraph::default();
    let scalar = graph.get_sort_by_name("i64").unwrap().clone();
    let mut definition = builtin::closed_signature(
        "test.identity",
        "identity",
        &[("x", scalar.clone())],
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
    let mut graph = EGraph::default();
    graph
        .type_info()
        .register_builtin_family(proto::HostSortFamily {
            name: "bool".into(),
            arity: 0,
            bindings: Some(proto::SortBindings {
                python: Some(proto::TypeBinding {
                    path: vec!["test".into(), "Bool".into()],
                    ..Default::default()
                }),
                ..Default::default()
            }),
        })
        .unwrap();
    let catalog = graph.type_info().builtin_catalog().unwrap();
    assert!(catalog.definitions.declarations.iter().any(|d| matches!(&d.kind, Some(proto::declaration::Kind::HostSortFamily(f)) if f.name == "bool" && f.bindings.is_some())));
    assert!(!catalog.undescribed_sorts.iter().any(|name| name == "bool"));
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
    let catalog = graph.type_info().builtin_catalog().unwrap();
    assert_eq!(CALLS.load(Ordering::SeqCst), 0);
    assert!(
        catalog
            .undescribed_primitives
            .iter()
            .any(|name| name == "-")
    );
    assert!(
        catalog
            .undescribed_families
            .iter()
            .any(|name| name == "Map")
    );
    assert!(
        catalog
            .undescribed_sorts
            .iter()
            .any(|name| name == "String")
    );
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
            "egglog.core.f64.add",
            "egglog.core.i64.add",
            "egglog.core.pair.first",
            "egglog.core.pair.make",
            "egglog.core.pair.second",
            "egglog.core.vec.empty",
            "egglog.core.vec.get",
            "egglog.core.vec.of",
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
    assert!(before.definitions.declarations.iter().any(|d| matches!(&d.kind,
        Some(proto::declaration::Kind::HostSortFamily(f)) if f.name == "Pair" && f.arity == 2 && f.bindings.is_none())));
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
        assert!(bindings.python.is_none() && bindings.rust.is_none());
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
        before
            .undescribed_family_primitives
            .iter()
            .any(|name| name == "Vec::vec-append")
    );
    assert!(
        !before
            .undescribed_family_primitives
            .iter()
            .any(|name| name == "Vec::vec-of")
    );
    for key in [
        "egglog.core.vec.empty",
        "egglog.core.vec.of",
        "egglog.core.vec.get",
    ] {
        assert!(before.definitions.declarations.iter().any(|declaration| matches!(&declaration.kind, Some(proto::declaration::Kind::HostPrimitive(primitive)) if primitive.name == key)));
    }
    graph
        .parse_and_run_program(
            None,
            r#"
        (sort V (Vec i64)) (sort W (Vec i64)) (sort Nested (Vec V))
        (function v () V :merge new) (function w () W :merge new)
        (function nested () Nested :merge new)
        (set (v) (vec-empty)) (set (w) (vec-of 3 4))
        (set (nested) (vec-of (v)))
        (check (= (vec-get (w) 1) 4) (= (vec-get (nested) 0) (v)))
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
    let v = graph.get_sort_by_name("V").unwrap();
    let i = graph.get_sort_by_name("i64").unwrap();
    for (alias, key, types) in [
        ("vec-empty", "egglog.core.vec.empty", vec![v.clone()]),
        ("vec-of", "egglog.core.vec.of", vec![i.clone(), v.clone()]),
        (
            "vec-get",
            "egglog.core.vec.get",
            vec![v.clone(), i.clone(), i.clone()],
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
                .unwrap(),
            );
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
    for wrong in 0..3 {
        let native = graph.get_sort_by_name("V").unwrap().clone();
        let instance = graph
            .type_info()
            .instantiate_builtin("egglog.core.vec.get", &native)
            .unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if wrong == 0 {
                add_primitive_with_validator!(&mut graph, "bad" [instance = instance] = |xs: @VecContainer, index: String| -?> # {
                    { CALLS.fetch_add(1, Ordering::SeqCst); xs.data.get(index.len()).copied() }
                }, |_: &mut TermDag, _: &[TermId]| { CALLS.fetch_add(1, Ordering::SeqCst); None });
            } else if wrong == 1 {
                add_primitive!(&mut graph, "bad" [instance = instance] = |xs: @VecContainer, index: i64| -> bool { { let _ = (xs, index); false } });
            } else {
                add_primitive!(&mut graph, "bad" [instance = instance] = [mut xs: #] -> # { xs.next().unwrap() });
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
        fn builtin_definition(&self) -> Option<&proto::Program> {
            Some(&self.0)
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
    graph.add_pure_primitive(Invalid(definition), None);
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
        fn builtin_definition(&self) -> Option<&proto::Program> {
            Some(&self.0)
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
    let definition =
        builtin::closed_signature("i64", "identity", &[("value", sort.clone())], sort.clone())
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
