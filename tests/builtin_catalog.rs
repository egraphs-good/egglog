use egglog::sort::{PairContainer, Presort, VecContainer, VecSort};
use egglog::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

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
        "-",
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
fn vec_catalog_has_generic_language_views() {
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
    for method in ["empty", "of", "get"] {
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
                _ => vec!["__getitem__"],
            }
        );
        assert_eq!(python.receiver, (method == "get").then_some(0));
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
    for deferred in ["guard", "count-matches", "bool-<"] {
        assert!(
            catalog
                .undescribed_primitives
                .iter()
                .any(|name| name == deferred)
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
    let mut graph = EGraph::default();
    graph
        .type_info()
        .register_builtin_family(proto::HostSortFamily {
            name: "Set".into(),
            arity: 1,
            bindings: Some(proto::SortBindings {
                python: Some(proto::TypeBinding {
                    path: vec!["test".into(), "Set".into()],
                    type_params: vec!["T".into()],
                }),
                ..Default::default()
            }),
        })
        .unwrap();
    let catalog = graph.type_info().builtin_catalog().unwrap();
    assert!(catalog.definitions.declarations.iter().any(|d| matches!(&d.kind, Some(proto::declaration::Kind::HostSortFamily(f)) if f.name == "Set" && f.bindings.is_some())));
    assert!(
        !catalog
            .undescribed_families
            .iter()
            .any(|name| name == "Set")
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
            .any(|name| name == "Set")
    );
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
            "egglog.core.bigint.from_i64",
            "egglog.core.bigint.from_string",
            "egglog.core.bigrat.make",
            "egglog.core.bool.and",
            "egglog.core.bool.implies",
            "egglog.core.bool.not",
            "egglog.core.bool.or",
            "egglog.core.bool.xor",
            "egglog.core.f64.add",
            "egglog.core.i64.add",
            "egglog.core.i64.to_string",
            "egglog.core.map.empty",
            "egglog.core.map.get",
            "egglog.core.map.insert",
            "egglog.core.map.of",
            "egglog.core.pair.first",
            "egglog.core.pair.make",
            "egglog.core.pair.second",
            "egglog.core.string.concat",
            "egglog.core.string.replace",
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
    assert_eq!(
        primitive
            .lower_builtin_call(graph.type_info(), &[], 0, None, &mut destination)
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
