use egglog::{
    proto as pb,
    sort::{F, MapContainer, MapSort, PairContainer, Presort, VecContainer},
    *,
};
use std::{
    any::TypeId,
    sync::atomic::{AtomicUsize, Ordering},
};

// Reuse actual Map storage without installing Map's own provenance/primitives
// for these distinct test providers. Their canonical signatures stay independent.
#[derive(Debug)]
struct FixtureMap(ArcSort);
impl sort::Sort for FixtureMap {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn column_ty(&self, backend: &egglog_bridge::EGraph) -> sort::ColumnTy {
        self.0.column_ty(backend)
    }
    fn inner_sorts(&self) -> Vec<ArcSort> {
        self.0.inner_sorts()
    }
    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        self.0.register_type(backend)
    }
    fn as_arc_any(self: std::sync::Arc<Self>) -> std::sync::Arc<dyn std::any::Any + Send + Sync> {
        self
    }
    fn is_container_sort(&self) -> bool {
        true
    }
    fn value_type(&self) -> Option<TypeId> {
        self.0.value_type()
    }
}

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
        let native: ArcSort =
            std::sync::Arc::new(FixtureMap(MapSort::make_sort(types, name, args, span)?));
        types
            .register_builtin_sort(Self::presort_name(), native.clone(), native.inner_sorts())
            .unwrap();
        Ok(native)
    }
}

static BODY: AtomicUsize = AtomicUsize::new(0);
static VALIDATOR: AtomicUsize = AtomicUsize::new(0);

// This fixture exercises binding only. It does not implement Map construction.
struct Foreign<const INPUT: u32 = 5, const SWAP: bool = false>;
impl<const INPUT: u32, const SWAP: bool> Presort for Foreign<INPUT, SWAP> {
    fn presort_name() -> &'static str {
        "TestForeign"
    }
    fn reserved_primitives() -> Vec<&'static str> {
        vec!["foreign"]
    }
    fn builtin_definitions() -> Vec<pb::Program> {
        let mut definition = Grouped::builtin_definitions().remove(0);
        let Some(pb::sort::Kind::Family(owner)) = &mut definition.sorts[2].kind else {
            unreachable!()
        };
        owner.name = "TestForeign".into();
        let Some(pb::declaration::Kind::HostSortFamily(owner)) =
            &mut definition.declarations[0].kind
        else {
            unreachable!()
        };
        owner.name = "TestForeign".into();
        for (name, args) in [
            ("Pair", if SWAP { vec![1, 0] } else { vec![0, 1] }),
            ("Vec", vec![4]),
        ] {
            definition.declarations.push(pb::Declaration {
                kind: Some(pb::declaration::Kind::HostSortFamily(pb::HostSortFamily {
                    name: name.into(),
                    arity: args.len() as u32,
                    ..Default::default()
                })),
                ..Default::default()
            });
            definition.sorts.push(pb::Sort {
                kind: Some(pb::sort::Kind::Family(pb::HostSort {
                    name: name.into(),
                    args,
                })),
                ..Default::default()
            });
        }
        let declaration = &mut definition.declarations[2];
        let Some(pb::declaration::Kind::HostPrimitive(primitive)) = &mut declaration.kind else {
            unreachable!()
        };
        primitive.name = "test.foreign".into();
        let Some(pb::host_primitive::Typing::Signature(signature)) = &mut primitive.typing else {
            unreachable!()
        };
        signature.inputs = vec![pb::Arg {
            name: "entries".into(),
            sort: INPUT,
        }];
        signature.varargs.clear();
        declaration
            .bindings
            .as_mut()
            .unwrap()
            .egglog
            .as_mut()
            .unwrap()
            .views[0]
            .symbol = "foreign".into();
        vec![definition]
    }
    fn make_sort(
        types: &mut TypeInfo,
        name: String,
        args: &[ast::Expr],
        span: ast::Span,
    ) -> Result<ArcSort, TypeError> {
        let native: ArcSort =
            std::sync::Arc::new(FixtureMap(MapSort::make_sort(types, name, args, span)?));
        types
            .register_builtin_sort(Self::presort_name(), native.clone(), native.inner_sorts())
            .unwrap();
        Ok(native)
    }
}

#[test]
fn foreign_instance_before_nested_family_aliases_checks_abi() {
    let mut graph = EGraph::default();
    graph
        .type_info()
        .add_presort::<Foreign>(ast::Span::Panic)
        .unwrap();
    let before = graph.type_info().builtin_catalog().unwrap().definitions;
    graph
        .parse_and_run_program(None, "(sort G (TestForeign i64 f64))")
        .unwrap();
    let native = graph.get_sort_by_name("G").unwrap().clone();
    let instance = graph
        .type_info()
        .instantiate_builtin("test.foreign", &native)
        .unwrap();
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    for wrong in 0..4 {
        let instance = instance.clone();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match wrong {
                0 => add_primitive_with_validator!(&mut graph, "bad-foreign" [instance = instance] = |xs: @PairContainer| -> # {
                    { CALLS.fetch_add(1, Ordering::SeqCst); xs.first }
                }, |_: &mut TermDag, _: &[TermId]| { CALLS.fetch_add(1, Ordering::SeqCst); None }),
                1 => add_primitive!(&mut graph, "bad-foreign" [instance = instance] = |xs: @VecContainer| -> i64 {
                    { CALLS.fetch_add(1, Ordering::SeqCst); xs.data.len() as i64 }
                }),
                2 => add_primitive!(&mut graph, "bad-foreign" [instance = instance] = || -> i64 {
                    { CALLS.fetch_add(1, Ordering::SeqCst); 0 }
                }),
                _ => add_primitive!(&mut graph, "bad-foreign" [instance = instance] = [mut xs: #] -> # {
                    { CALLS.fetch_add(1, Ordering::SeqCst); xs.next().unwrap() }
                }),
            }
        })).is_err());
    }
    assert!(
        instance
            .check_abi(&[Some((TypeId::of::<VecContainer>(), false))], None, false)
            .is_err()
    );
    assert_eq!(CALLS.load(Ordering::SeqCst), 0);
    instance
        .check_abi(
            &[Some((TypeId::of::<sort::VecContainer>(), true))],
            Some((TypeId::of::<MapContainer>(), true)),
            false,
        )
        .unwrap();
    assert_eq!(
        graph.type_info().builtin_catalog().unwrap().definitions,
        before
    );
}

#[test]
fn foreign_computed_aliases_resolve_in_both_declaration_orders() {
    for owner_first in [false, true] {
        for reverse in [false, true] {
            let mut graph = EGraph::default();
            graph
                .type_info()
                .add_presort::<Foreign>(ast::Span::Panic)
                .unwrap();
            let before = graph.type_info().builtin_catalog().unwrap().definitions;
            let aliases = if reverse {
                "(sort Q (Pair i64 f64)) (sort P (Pair i64 f64)) (sort B (Vec Q)) (sort A (Vec P))"
            } else {
                "(sort P (Pair i64 f64)) (sort Q (Pair i64 f64)) (sort A (Vec P)) (sort B (Vec Q))"
            };
            if !owner_first {
                graph.parse_and_run_program(None, aliases).unwrap();
            }
            graph.parse_and_run_program(None, "(sort G (TestForeign i64 f64)) (function seed () G :merge new) (sort SeedMap (Map i64 f64)) (function raw-seed () SeedMap :merge new) (set (raw-seed) (map-empty))").unwrap();
            let native = graph.get_sort_by_name("G").unwrap().clone();
            let instance = graph
                .type_info()
                .instantiate_builtin("test.foreign", &native)
                .unwrap();
            let (_, seed) = graph
                .eval_expr(&ast::Expr::Call(
                    ast::Span::Panic,
                    "raw-seed".into(),
                    vec![],
                ))
                .unwrap();
            let seed = graph
                .value_to_container::<MapContainer>(seed)
                .unwrap()
                .clone();
            let calls = std::sync::Arc::new(AtomicUsize::new(0));
            // Only read the actual computed Vec and return existing owner storage.
            // The fixture deliberately does not implement Map's entry semantics.
            add_primitive!(&mut graph, "foreign" [instance = instance] = {(seed, calls.clone()): (MapContainer, std::sync::Arc<AtomicUsize>)} |xs: @VecContainer| -> @MapContainer {
                { self.ctx.1.fetch_add(xs.data.len() + 1, Ordering::SeqCst); self.ctx.0.clone() }
            });
            if owner_first {
                graph.parse_and_run_program(None, aliases).unwrap();
            }
            graph
                .parse_and_run_program(
                    None,
                    "
                (function a () A :merge new)
                (function b () B :merge new)
                (set (a) (vec-of (pair 1 2.0)))
                (set (b) (vec-of (pair 3 4.0) (pair 5 6.0)))
                (set (seed) (foreign (a)))
                (check (= (foreign (a)) (seed)))
                (check (= (foreign (b)) (seed)))
                (set (a) (vec-empty))
                (check (= (foreign (a)) (seed)))
            ",
                )
                .unwrap();
            assert!(calls.load(Ordering::SeqCst) >= 6);
            assert_eq!(
                graph.type_info().builtin_catalog().unwrap().definitions,
                before
            );

            graph
                .parse_and_run_program(
                    None,
                    "
                (sort WrongK (Pair f64 f64)) (sort WrongV (Pair i64 i64))
                (sort WrongOrder (Pair f64 i64)) (sort Nested (Pair i64 P))
                (sort WK (Vec WrongK)) (sort WV (Vec WrongV))
                (sort WO (Vec WrongOrder)) (sort WN (Vec Nested))
                (sort NotPair (Vec i64)) (sort Nominal) (sort VN (Vec Nominal))
                (function wk () WK :merge new) (function wv () WV :merge new)
                (function wo () WO :merge new) (function wn () WN :merge new)
                (function scalar () NotPair :merge new) (function nominal () VN :merge new)
                (function pair-only () P :merge new)
            ",
                )
                .unwrap();
            let count = calls.load(Ordering::SeqCst);
            for expression in [
                "(foreign (wk))",
                "(foreign (wv))",
                "(foreign (wo))",
                "(foreign (wn))",
                "(foreign (scalar))",
                "(foreign (nominal))",
                "(foreign (pair-only))",
                "(foreign)",
                "(foreign (a) (b))",
            ] {
                let command = graph.parse_program(None, expression).unwrap().remove(0);
                assert!(
                    graph.resolve_command_before_proofs(command).is_err(),
                    "{expression}"
                );
            }
            // A result constraint fixes K/V, but cannot choose either nominal Vec.
            let command = graph
                .parse_program(None, "(check (= (foreign (vec-empty)) (seed)))")
                .unwrap()
                .remove(0);
            assert!(graph.resolve_command_before_proofs(command).is_err());
            assert_eq!(calls.load(Ordering::SeqCst), count);

            for (name, valid) in [
                ("A", true),
                ("B", true),
                ("WK", false),
                ("WV", false),
                ("WO", false),
                ("WN", false),
                ("NotPair", false),
                ("VN", false),
                ("P", false),
            ] {
                let input = graph.get_sort_by_name(name).unwrap().clone();
                let mut program = pb::Program::default();
                let arg = graph.export_sort(&input, &mut program.sorts).unwrap();
                let out = graph.export_sort(&native, &mut program.sorts).unwrap();
                let actuals = [input, native.clone()];
                let resolution = graph.type_info().resolve_builtin(
                    &program,
                    "test.foreign",
                    &[arg],
                    out,
                    &actuals,
                );
                assert_eq!(resolution.is_ok(), valid, "{name}: {resolution:?}");
                if let Ok(key) = resolution {
                    for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
                        assert_eq!(
                            ResolvedCall::from_resolution(
                                "foreign",
                                &actuals,
                                graph.type_info(),
                                context,
                                &ast::Span::Panic
                            )
                            .unwrap(),
                            ResolvedCall::from_resolution(
                                &key,
                                &actuals,
                                graph.type_info(),
                                context,
                                &ast::Span::Panic
                            )
                            .unwrap(),
                        );
                    }
                }
            }
            assert_eq!(calls.load(Ordering::SeqCst), count);
        }
    }
}

#[test]
fn foreign_descriptor_mutations_control_nested_matching_and_converter_abi() {
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let mut graph = EGraph::default();
    // Change the actual registered input from Vec<Pair<K,V>> to Pair<K,V>.
    graph
        .type_info()
        .add_presort::<Foreign<4>>(ast::Span::Panic)
        .unwrap();
    graph
        .parse_and_run_program(None, "(sort G (TestForeign i64 f64))")
        .unwrap();
    let owner = graph.get_sort_by_name("G").unwrap().clone();
    let instance = graph
        .type_info()
        .instantiate_builtin("test.foreign", &owner)
        .unwrap();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            add_primitive!(&mut graph, "foreign" [instance = instance] = |xs: @VecContainer| -> # {
                { CALLS.fetch_add(1, Ordering::SeqCst); xs.data[0] }
            });
        }))
        .is_err()
    );

    let mut graph = EGraph::default();
    // Change only nested binder order. The outer Vec carrier stays identical.
    graph
        .type_info()
        .add_presort::<Foreign<5, true>>(ast::Span::Panic)
        .unwrap();
    graph
        .parse_and_run_program(
            None,
            "
        (sort G (TestForeign i64 f64)) (function g () G :merge new)
        (sort P (Pair i64 f64)) (sort Q (Pair f64 i64))
        (sort A (Vec P)) (sort B (Vec Q))
        (function a () A :merge new) (function b () B :merge new)
    ",
        )
        .unwrap();
    let owner = graph.get_sort_by_name("G").unwrap().clone();
    let instance = graph
        .type_info()
        .instantiate_builtin("test.foreign", &owner)
        .unwrap();
    add_primitive!(&mut graph, "foreign" [instance = instance] = |xs: @VecContainer| -> # {
        { CALLS.fetch_add(1, Ordering::SeqCst); xs.data[0] }
    });
    for (text, valid) in [
        ("(check (= (foreign (a)) (g)))", false),
        ("(check (= (foreign (b)) (g)))", true),
        // Result-directed empty succeeds when exactly one nominal Vec matches.
        ("(check (= (foreign (vec-empty)) (g)))", true),
    ] {
        let command = graph.parse_program(None, text).unwrap().remove(0);
        assert_eq!(
            graph.resolve_command_before_proofs(command).is_ok(),
            valid,
            "{text}"
        );
    }
    // The same instance observes aliases in the current snapshot, not a captured
    // registration-time list; pop and independent clones restore that boundary.
    let mut cloned = graph.clone();
    for (extra, valid) in [("(push) (sort Later (Vec Q))", false), ("(pop)", true)] {
        cloned.parse_and_run_program(None, extra).unwrap();
        let command = cloned
            .parse_program(None, "(check (= (foreign (vec-empty)) (g)))")
            .unwrap()
            .remove(0);
        assert_eq!(cloned.resolve_command_before_proofs(command).is_ok(), valid);
        let command = graph
            .parse_program(None, "(check (= (foreign (vec-empty)) (g)))")
            .unwrap()
            .remove(0);
        graph.resolve_command_before_proofs(command).unwrap();
    }
    let mut stale = Foreign::<5, false>::builtin_definitions().remove(0);
    let Some(pb::declaration::Kind::HostPrimitive(primitive)) = &stale.declarations[2].kind else {
        unreachable!()
    };
    assert!(
        graph
            .type_info()
            .check_builtin_signature(&stale, primitive)
            .is_err()
    );
    let input = graph.get_sort_by_name("B").unwrap().clone();
    let mut wire = pb::Program::default();
    let arg = graph.export_sort(&input, &mut wire.sorts).unwrap();
    let out = graph.export_sort(&owner, &mut wire.sorts).unwrap();
    assert!(
        graph
            .type_info()
            .resolve_builtin(
                &wire,
                "test.foreign",
                &[arg],
                out,
                &[input.clone(), owner.clone()]
            )
            .is_ok()
    );
    let Some(pb::sort::Kind::Family(vec)) = &wire.sorts[arg as usize].kind else {
        unreachable!()
    };
    let pair = vec.args[0];
    let Some(pb::sort::Kind::Family(pair)) = &mut wire.sorts[pair as usize].kind else {
        unreachable!()
    };
    pair.args.reverse();
    assert!(
        graph
            .type_info()
            .resolve_builtin(&wire, "test.foreign", &[arg], out, &[input, owner])
            .is_err()
    );
    // A supplied descriptor must agree even when every runtime cast is unchanged.
    let Some(pb::sort::Kind::Family(pair)) = &mut stale.sorts[4].kind else {
        unreachable!()
    };
    pair.args.reverse();
    let Some(pb::declaration::Kind::HostPrimitive(primitive)) = &stale.declarations[2].kind else {
        unreachable!()
    };
    graph
        .type_info()
        .check_builtin_signature(&stale, primitive)
        .unwrap();
    assert_eq!(CALLS.load(Ordering::SeqCst), 0);
}

#[test]
fn foreign_provider_rejects_spoofed_nested_carrier_provenance() {
    let mut graph = EGraph::default();
    let args = [
        ast::Expr::Var(ast::Span::Panic, "i64".into()),
        ast::Expr::Var(ast::Span::Panic, "f64".into()),
    ];
    let fake_pair = MapSort::make_sort(
        graph.type_info(),
        "FakePair".into(),
        &args,
        ast::Span::Panic,
    )
    .unwrap();
    let parameters = fake_pair.inner_sorts();
    let error = graph
        .type_info()
        .register_builtin_sort("Pair", fake_pair, parameters)
        .unwrap_err();
    assert!(error.contains("provider representation"), "{error}");
    let fake_vec =
        MapSort::make_sort(graph.type_info(), "FakeVec".into(), &args, ast::Span::Panic).unwrap();
    let element = graph.get_sort_by_name("i64").unwrap().clone();
    let error = graph
        .type_info()
        .register_builtin_sort("Vec", fake_vec, vec![element])
        .unwrap_err();
    assert!(error.contains("provider representation"), "{error}");
}

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
