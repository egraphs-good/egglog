use egglog::{
    proto as pb,
    sort::{F, MapContainer, MapSort, PairContainer, Presort, VecContainer},
    *,
};
use std::{
    any::TypeId,
    sync::atomic::{AtomicUsize, Ordering},
};

mod native_signature {
    use super::*;
    use egglog::constraint::{Constraint, TypeConstraint};

    // An owner for test descriptors only; no production family/primitive is added.
    struct Provider<const CASE: u8>;
    impl<const CASE: u8> Presort for Provider<CASE> {
        fn presort_name() -> &'static str {
            "NativeSignatureFixture"
        }
        fn reserved_primitives() -> Vec<&'static str> {
            vec![]
        }
        fn builtin_definitions() -> Vec<pb::Program> {
            use pb::sort::Kind;
            let mut sorts = vec![
                Kind::Var(0),
                Kind::Var(1),
                Kind::Family(pb::HostSort {
                    name: "Vec".into(),
                    args: vec![0],
                }),
                Kind::Family(pb::HostSort {
                    name: "Vec".into(),
                    args: vec![1],
                }),
                Kind::Func(pb::FuncSort {
                    params: vec![0],
                    result: 1,
                }),
                Kind::Func(pb::FuncSort {
                    params: vec![2],
                    result: 3,
                }),
                Kind::Family(pb::HostSort {
                    name: "Vec".into(),
                    args: vec![2],
                }),
                Kind::Family(pb::HostSort {
                    name: "Vec".into(),
                    args: vec![3],
                }),
                Kind::Family(pb::HostSort {
                    name: "i64".into(),
                    args: vec![],
                }),
                Kind::Family(pb::HostSort {
                    name: "String".into(),
                    args: vec![],
                }),
                Kind::Eq("E".into()),
            ];
            let (inputs, output, binders) = match CASE {
                0 => (vec![4, 2], 3, 2),
                1 | 2 | 6 | 7 => (vec![0, 0], 0, if CASE == 7 { 2 } else { 1 }),
                3 => (vec![8], 9, 0),
                4 => (vec![5, 6], 7, 2),
                5 => (vec![], 2, 1),
                8 => (vec![10], 10, 0),
                _ => unreachable!(),
            };
            if CASE == 2 {
                sorts[0] = Kind::Var(2); // Outside the descriptor's binder scope.
            }
            vec![pb::Program {
                ir_version: 1,
                sorts: sorts
                    .into_iter()
                    .map(|kind| pb::Sort {
                        kind: Some(kind),
                        ..Default::default()
                    })
                    .collect(),
                declarations: vec![pb::Declaration {
                    kind: Some(pb::declaration::Kind::HostPrimitive(pb::HostPrimitive {
                        name: "test.native_signature".into(),
                        typing: Some(pb::host_primitive::Typing::Signature(
                            pb::GenericSignature {
                                type_params: (0..binders).map(|i| format!("T{i}")).collect(),
                                equality_type_params: if CASE == 6 { vec![0] } else { vec![] },
                                inputs: inputs
                                    .into_iter()
                                    .enumerate()
                                    .map(|(i, sort)| pb::Arg {
                                        name: format!("arg{i}"),
                                        sort,
                                    })
                                    .collect(),
                                output: Some(output),
                                ..Default::default()
                            },
                        )),
                    })),
                    ..Default::default()
                }],
                ..Default::default()
            }]
        }
        fn make_sort(
            _: &mut TypeInfo,
            _: String,
            _: &[ast::Expr],
            span: ast::Span,
        ) -> Result<ArcSort, TypeError> {
            Err(TypeError::BadPresortArguments(
                Self::presort_name().into(),
                span,
            ))
        }
    }

    #[derive(Clone)]
    struct Exact(builtin::BuiltinInstance);
    impl Primitive for Exact {
        fn name(&self) -> &str {
            "signature-exact"
        }
        fn get_type_constraints(&self, span: &ast::Span) -> Box<dyn TypeConstraint> {
            self.0.type_constraints(span)
        }
        fn builtin_instance(&self) -> Option<&builtin::BuiltinInstance> {
            Some(&self.0)
        }
    }
    impl PurePrim for Exact {
        fn apply<'a, 'db>(&self, _: PureState<'a, 'db>, _: &[Value]) -> Option<Value> {
            panic!("signature checks must not execute the fixture")
        }
    }

    #[derive(Clone)]
    struct Open {
        instance: builtin::BuiltinInstance,
        equality: Option<bool>,
    }
    impl Primitive for Open {
        fn name(&self) -> &str {
            "signature-open"
        }
        fn get_type_constraints(&self, span: &ast::Span) -> Box<dyn TypeConstraint> {
            Box::new(OpenConstraint {
                equality: self.equality,
                span: span.clone(),
            })
        }
        fn builtin_instance(&self) -> Option<&builtin::BuiltinInstance> {
            Some(&self.instance)
        }
    }
    impl PurePrim for Open {
        fn apply<'a, 'db>(&self, _: PureState<'a, 'db>, _: &[Value]) -> Option<Value> {
            panic!("open-domain selection must not execute the fixture")
        }
    }
    impl WritePrim for Open {
        fn apply<'a, 'db>(&self, _: WriteState<'a, 'db>, _: &[Value]) -> Option<Value> {
            panic!("open-domain selection must not execute the write fixture")
        }
    }
    struct OpenConstraint {
        equality: Option<bool>,
        span: ast::Span,
    }
    impl TypeConstraint for OpenConstraint {
        fn get(
            &self,
            arguments: &[AtomTerm],
            types: &TypeInfo,
        ) -> Vec<Box<dyn Constraint<AtomTerm, ArcSort>>> {
            let signatures = types
                .get_arcsorts_by(|sort| self.equality.is_none_or(|eq| sort.is_eq_sort() == eq));
            vec![constraint::xor(
                signatures
                    .into_iter()
                    .map(|sort| {
                        constraint::SimpleTypeConstraint::new(
                            "signature-open",
                            vec![sort.clone(); arguments.len()],
                            self.span.clone(),
                        )
                        .get(arguments, types)
                        .into_iter()
                        .collect::<Vec<_>>()
                    })
                    .map(constraint::and)
                    .collect(),
            )]
        }
    }

    #[derive(Clone)]
    struct NativeApplication {
        instance: builtin::BuiltinInstance,
        signature: Vec<ArcSort>,
    }
    impl Primitive for NativeApplication {
        fn name(&self) -> &str {
            "signature-application"
        }
        fn get_type_constraints(&self, span: &ast::Span) -> Box<dyn TypeConstraint> {
            constraint::SimpleTypeConstraint::new(self.name(), self.signature.clone(), span.clone())
                .into_box()
        }
        fn builtin_instance(&self) -> Option<&builtin::BuiltinInstance> {
            Some(&self.instance)
        }
    }
    impl PurePrim for NativeApplication {
        fn apply<'a, 'db>(&self, _: PureState<'a, 'db>, _: &[Value]) -> Option<Value> {
            panic!("Application selection must not execute the native fixture")
        }
    }

    // Counts actual metadata reads, giving the DAG regression a deterministic
    // work bound instead of a timing assertion or an exponentially slow oracle.
    #[derive(Debug)]
    struct CountedPair {
        native: ArcSort,
        reads: std::sync::Arc<AtomicUsize>,
    }
    impl sort::Sort for CountedPair {
        fn name(&self) -> &str {
            assert!(
                self.reads.fetch_add(1, Ordering::Relaxed) < 200_000,
                "shared Pair comparison repeated its sub-DAG"
            );
            self.native.name()
        }
        fn column_ty(&self, backend: &egglog_bridge::EGraph) -> sort::ColumnTy {
            self.native.column_ty(backend)
        }
        fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
            self.native.register_type(backend)
        }
        fn as_arc_any(
            self: std::sync::Arc<Self>,
        ) -> std::sync::Arc<dyn std::any::Any + Send + Sync> {
            self
        }
        fn inner_sorts(&self) -> Vec<ArcSort> {
            self.native.inner_sorts()
        }
        fn is_container_sort(&self) -> bool {
            true
        }
        fn value_type(&self) -> Option<TypeId> {
            self.native.value_type()
        }
    }

    fn fixture_graph<const CASE: u8>() -> EGraph {
        let mut graph = EGraph::default();
        graph
            .type_info()
            .add_presort::<Provider<CASE>>(ast::Span::Panic)
            .unwrap();
        graph
    }

    // Exercises both wire selection and direct/canonical/private native selection,
    // comparing complete original registrations rather than adding diagnostic APIs.
    fn selected(graph: &mut EGraph, roots: &[ArcSort], native_name: &str) -> String {
        let mut program = pb::Program::default();
        let mut ids = roots
            .iter()
            .map(|s| graph.export_sort(s, &mut program.sorts).unwrap())
            .collect::<Vec<_>>();
        let output = ids.pop().unwrap();
        let types = graph.type_info();
        let key = types
            .resolve_builtin(&program, "test.native_signature", &ids, output, roots)
            .unwrap();
        for ctx in [Context::Pure, Context::Read, Context::Write, Context::Full] {
            let native =
                ResolvedCall::from_resolution(native_name, roots, types, ctx, &ast::Span::Panic);
            for alias in ["test.native_signature", &key] {
                let selected =
                    ResolvedCall::from_resolution(alias, roots, types, ctx, &ast::Span::Panic);
                match (&native, selected) {
                    (Ok(native), Ok(selected)) => {
                        assert_eq!(native, &selected);
                        let (ResolvedCall::Primitive(native), ResolvedCall::Primitive(selected)) =
                            (native, &selected)
                        else {
                            panic!("expected native primitives")
                        };
                        match (native.validator(), selected.validator()) {
                            (Some(native), Some(selected)) => {
                                assert!(std::sync::Arc::ptr_eq(native, selected))
                            }
                            (None, None) => (),
                            _ => panic!("alias changed the native validator"),
                        }
                    }
                    (Err(_), Err(_)) => (),
                    _ => panic!("alias changed native context availability"),
                }
            }
        }
        assert_eq!(
            types
                .export_function_target(
                    native_name,
                    &roots[..roots.len() - 1],
                    roots.last().unwrap(),
                    None,
                    &mut pb::Program::default()
                )
                .unwrap(),
            "test.native_signature"
        );
        key
    }

    // A complete negative must agree across native/canonical/private selection,
    // exported targets, wire selection, and named-function construction.
    fn rejected(
        graph: &mut EGraph,
        roots: &[ArcSort],
        private: &str,
        context: Context,
        table: &str,
    ) {
        let aliases = ["signature-open", "test.native_signature", private];
        let mut wire = pb::Program::default();
        let mut ids = roots
            .iter()
            .map(|sort| graph.export_sort(sort, &mut wire.sorts).unwrap())
            .collect::<Vec<_>>();
        let output = ids.pop().unwrap();
        let types = graph.type_info();
        for alias in aliases {
            assert!(
                ResolvedCall::from_resolution(alias, roots, types, context, &ast::Span::Panic)
                    .is_err(),
                "{alias}"
            );
            assert!(
                types
                    .export_function_target(
                        alias,
                        &roots[..roots.len() - 1],
                        roots.last().unwrap(),
                        None,
                        &mut pb::Program::default()
                    )
                    .is_err(),
                "{alias}"
            );
        }
        assert!(
            types
                .resolve_builtin(&wire, "test.native_signature", &ids, output, roots)
                .is_err()
        );
        let inputs = roots[..roots.len() - 1]
            .iter()
            .map(|sort| sort.name())
            .collect::<Vec<_>>()
            .join(" ");
        graph.parse_and_run_program(None, &format!("(sort {table}Fn (UnstableFn ({inputs}) {})) (function {table} () {table}Fn :merge new)", roots.last().unwrap().name())).unwrap();
        for alias in aliases {
            assert!(
                graph
                    .parse_and_run_program(
                        None,
                        &format!("(set ({table}) (unstable-fn {alias:?}))")
                    )
                    .is_err(),
                "{alias}"
            );
        }
    }

    #[test]
    fn native_signature_keeps_alias_roots_per_occurrence_and_function() {
        let mut graph = fixture_graph::<1>();
        graph.parse_and_run_program(None, "(sort V (Vec i64)) (sort W (Vec i64)) (sort F (UnstableFn (i64) i64)) (sort G (UnstableFn (i64) i64))").unwrap();
        let v = graph.get_sort_by_name("V").unwrap().clone();
        let w = graph.get_sort_by_name("W").unwrap().clone();
        let f = graph.get_sort_by_name("F").unwrap().clone();
        let g = graph.get_sort_by_name("G").unwrap().clone();
        let signatures = [
            vec![v.clone(), w.clone(), v],
            vec![w.clone(), graph.get_sort_by_name("V").unwrap().clone(), w],
            vec![f.clone(), g.clone(), f],
            vec![g.clone(), graph.get_sort_by_name("F").unwrap().clone(), g],
        ];
        let before = graph.type_info().builtin_catalog().unwrap().definitions;
        for roots in &signatures {
            let instance = graph
                .type_info()
                .instantiate_builtin_signature("test.native_signature", &roots[..2], &[], &roots[2])
                .unwrap();
            instance.check_abi(&[None, None], None, false).unwrap();
            graph.add_pure_primitive(Exact(instance), None);
        }
        let mut keys = hashbrown::HashSet::new();
        for roots in signatures {
            assert!(keys.insert(selected(&mut graph, &roots, "signature-exact")));
        }
        assert_eq!(
            graph.type_info().builtin_catalog().unwrap().definitions,
            before
        );
        assert!(
            ResolvedCall::from_resolution(
                "signature-exact",
                &vec![graph.get_sort_by_name("V").unwrap().clone(); 3],
                graph.type_info(),
                Context::Pure,
                &ast::Span::Panic
            )
            .is_err()
        );
    }

    #[test]
    fn native_signature_nested_functions_result_binders_and_eq_bounds() {
        let mut graph = fixture_graph::<4>();
        graph.parse_and_run_program(None, "(sort X (Vec i64)) (sort Y (Vec String)) (sort XX (Vec X)) (sort YY (Vec Y)) (sort F (UnstableFn (X) Y)) (sort Bad (UnstableFn (X X) Y))").unwrap();
        let roots = ["F", "XX", "YY"].map(|name| graph.get_sort_by_name(name).unwrap().clone());
        let instance = graph
            .type_info()
            .instantiate_builtin_signature("test.native_signature", &roots[..2], &[], &roots[2])
            .unwrap();
        graph.add_pure_primitive(Exact(instance), None);
        selected(&mut graph, &roots, "signature-exact");
        assert!(
            graph
                .type_info()
                .instantiate_builtin_signature("test.native_signature", &roots[..2], &[], &roots[1])
                .is_err()
        );
        let bad = graph.get_sort_by_name("Bad").unwrap().clone();
        assert!(
            graph
                .type_info()
                .instantiate_builtin_signature(
                    "test.native_signature",
                    &[bad, roots[1].clone()],
                    &[],
                    &roots[2]
                )
                .is_err()
        );

        let mut zero = fixture_graph::<5>();
        zero.parse_and_run_program(None, "(sort V (Vec i64))")
            .unwrap();
        let v = zero.get_sort_by_name("V").unwrap().clone();
        let instance = zero
            .type_info()
            .instantiate_builtin_signature("test.native_signature", &[], &[], &v)
            .unwrap();
        zero.add_pure_primitive(Exact(instance), None);
        selected(&mut zero, &[v], "signature-exact");
        let mut missing = fixture_graph::<7>();
        let i = missing.get_sort_by_name("i64").unwrap().clone();
        assert!(
            missing
                .type_info()
                .instantiate_builtin_signature(
                    "test.native_signature",
                    &[i.clone(), i.clone()],
                    &[],
                    &i
                )
                .is_err()
        );
        let mut malformed = fixture_graph::<2>();
        assert!(
            malformed
                .type_info()
                .builtin_catalog()
                .err()
                .unwrap()
                .contains("unbound builtin type parameter")
        );
        assert_eq!(
            malformed
                .type_info()
                .instantiate_builtin_signature(
                    "test.native_signature",
                    &[i.clone(), i.clone()],
                    &[],
                    &i
                )
                .err()
                .as_deref(),
            Some("unknown builtin definition")
        );
        let mut eq = fixture_graph::<6>();
        eq.parse_and_run_program(None, "(sort E) (sort V (Vec E))")
            .unwrap();
        for (name, valid) in [("E", true), ("V", false)] {
            let sort = eq.get_sort_by_name(name).unwrap().clone();
            assert_eq!(
                eq.type_info()
                    .instantiate_builtin_signature(
                        "test.native_signature",
                        &[sort.clone(), sort.clone()],
                        &[],
                        &sort
                    )
                    .is_ok(),
                valid
            );
        }
        let mut nominal = fixture_graph::<8>();
        nominal
            .parse_and_run_program(None, "(sort E) (sort Other)")
            .unwrap();
        let e = nominal.get_sort_by_name("E").unwrap().clone();
        let other = nominal.get_sort_by_name("Other").unwrap().clone();
        assert!(
            nominal
                .type_info()
                .instantiate_builtin_signature(
                    "test.native_signature",
                    std::slice::from_ref(&e),
                    &[],
                    &e
                )
                .is_ok()
        );
        assert!(
            nominal
                .type_info()
                .instantiate_builtin_signature(
                    "test.native_signature",
                    std::slice::from_ref(&other),
                    &[],
                    &other
                )
                .is_err()
        );
    }

    #[test]
    fn native_signature_open_domains_keep_late_sorts_and_reject_overlap() {
        let mut graph = fixture_graph::<1>();
        let instance = graph
            .type_info()
            .instantiate_builtin_open("test.native_signature")
            .unwrap();
        assert!(instance.check_abi(&[None, None], None, false).is_err());
        let validator: PrimitiveValidator = std::sync::Arc::new(|_: &mut TermDag, _: &[TermId]| {
            panic!("selection must not run the original validator")
        });
        graph.add_write_primitive(
            Open {
                instance: instance.clone(),
                equality: Some(false),
            },
            Some(validator.clone()),
        );
        graph.add_pure_primitive(
            Open {
                instance: instance.clone(),
                equality: Some(true),
            },
            None,
        );
        let i = graph.get_sort_by_name("i64").unwrap().clone();
        let roots = vec![i; 3];
        let private = selected(&mut graph, &roots, "signature-open");
        let before = ResolvedCall::from_resolution(
            "signature-open",
            &roots,
            graph.type_info(),
            Context::Write,
            &ast::Span::Panic,
        )
        .unwrap();
        let ResolvedCall::Primitive(original) = &before else {
            panic!("expected native primitive")
        };
        assert!(std::sync::Arc::ptr_eq(
            original.validator().unwrap(),
            &validator
        ));
        assert!(
            ResolvedCall::from_resolution(
                "signature-open",
                &roots,
                graph.type_info(),
                Context::Pure,
                &ast::Span::Panic
            )
            .is_err()
        );
        graph.parse_and_run_program(None, "(sort Later) (sort V (Vec i64)) (sort W (Vec i64)) (sort F (UnstableFn (i64) i64)) (function saved () F :merge new)").unwrap();
        for name in ["i64", "Later"] {
            let root = graph.get_sort_by_name(name).unwrap().clone();
            assert_eq!(
                selected(
                    &mut graph,
                    &[root.clone(), root.clone(), root],
                    "signature-open"
                ),
                private
            );
        }
        let after = ResolvedCall::from_resolution(
            "signature-open",
            &roots,
            graph.type_info(),
            Context::Write,
            &ast::Span::Panic,
        )
        .unwrap();
        assert_eq!(before, after);
        let ResolvedCall::Primitive(after) = after else {
            panic!("expected native primitive")
        };
        assert!(std::sync::Arc::ptr_eq(
            after.validator().unwrap(),
            &validator
        ));
        let v = graph.get_sort_by_name("V").unwrap().clone();
        let w = graph.get_sort_by_name("W").unwrap().clone();
        // The descriptor agrees structurally; the original nominal predicate
        // still rejects differently named argument occurrences.
        assert!(
            graph
                .type_info()
                .instantiate_builtin_signature(
                    "test.native_signature",
                    &[v.clone(), w.clone()],
                    &[],
                    &v
                )
                .is_ok()
        );
        rejected(
            &mut graph,
            &[v.clone(), w, v],
            &private,
            Context::Write,
            "mixed",
        );
        // Named-function construction captures one input; no fixture body runs.
        for alias in ["signature-open", "test.native_signature", private.as_str()] {
            graph
                .parse_and_run_program(None, &format!("(set (saved) (unstable-fn {alias:?} 1))"))
                .unwrap();
        }
        graph.add_pure_primitive(
            Open {
                instance,
                equality: None,
            },
            None,
        );
        rejected(&mut graph, &roots, &private, Context::Write, "overlap");
    }

    #[test]
    fn native_signature_native_acceptance_cannot_bypass_descriptor() {
        let mut control = fixture_graph::<1>();
        let instance = control
            .type_info()
            .instantiate_builtin_open("test.native_signature")
            .unwrap();
        control.add_pure_primitive(
            Open {
                instance,
                equality: None,
            },
            None,
        );
        let i = control.get_sort_by_name("i64").unwrap().clone();
        let private = selected(&mut control, &[i.clone(), i.clone(), i], "signature-open");
        let mut graph = fixture_graph::<3>();
        let instance = graph
            .type_info()
            .instantiate_builtin_open("test.native_signature")
            .unwrap();
        graph.add_pure_primitive(
            Open {
                instance,
                equality: None,
            },
            None,
        );
        let i = graph.get_sort_by_name("i64").unwrap().clone();
        let roots = [i.clone(), i.clone()]; // Native identity predicate accepts; descriptor requires String.
        let types = graph.type_info();
        for key in ["signature-open", "test.native_signature", private.as_str()] {
            assert!(!types.get_prims(key).unwrap()[0].accept(&roots, types));
            assert!(
                ResolvedCall::from_resolution(key, &roots, types, Context::Pure, &ast::Span::Panic)
                    .is_err()
            );
            assert!(
                types
                    .export_function_target(
                        key,
                        &roots[..1],
                        &roots[1],
                        None,
                        &mut pb::Program::default()
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
        for key in ["signature-open", "test.native_signature", private.as_str()] {
            assert!(
                graph
                    .parse_and_run_program(None, &format!("(set (saved) (unstable-fn {key:?}))"))
                    .is_err()
            );
        }
        let mut wire = pb::Program::default();
        let id = graph.export_sort(&i, &mut wire.sorts).unwrap();
        assert!(
            graph
                .type_info()
                .resolve_builtin(&wire, "test.native_signature", &[id], id, &roots)
                .is_err()
        );
    }

    #[test]
    fn native_signature_instances_cannot_cross_registry_owners() {
        let mut first = fixture_graph::<1>();
        let instance = first
            .type_info()
            .instantiate_builtin_open("test.native_signature")
            .unwrap();
        let mut other = fixture_graph::<1>();
        other.add_pure_primitive(
            Open {
                instance,
                equality: None,
            },
            None,
        );
        assert!(other.type_info().builtin_catalog().is_err());
    }

    #[test]
    fn native_signature_exact_tail_groups_and_converter_abi_are_checked() {
        let mut graph = grouped_graph();
        let i = graph.get_sort_by_name("i64").unwrap().clone();
        let f = graph.get_sort_by_name("f64").unwrap().clone();
        let g = graph.get_sort_by_name("G").unwrap().clone();
        let instance = graph
            .type_info()
            .instantiate_builtin_signature(
                "test.grouped",
                std::slice::from_ref(&i),
                &[i.clone(), f.clone()],
                &g,
            )
            .unwrap();
        assert!(
            instance
                .check_abi(
                    &[None, None, None],
                    Some((TypeId::of::<VecContainer>(), true)),
                    true
                )
                .is_err()
        );
        assert!(
            instance
                .check_abi(&[None, None, None], None, false)
                .is_err()
        );
        assert!(
            graph
                .type_info()
                .instantiate_builtin_signature(
                    "test.grouped",
                    std::slice::from_ref(&i),
                    std::slice::from_ref(&i),
                    &g
                )
                .is_err()
        );
        // The direct alias selects this implementation separately from the
        // existing grouped owner aliases; the canonical group now overlaps.
        graph.add_pure_primitive(Exact(instance), None);
        let types = graph.type_info();
        for roots in [
            vec![i.clone(), g.clone()],
            vec![i.clone(), i.clone(), f.clone(), g.clone()],
            vec![i.clone(), i.clone(), g.clone()],
        ] {
            assert_eq!(
                types.get_prims("signature-exact").unwrap()[0].accept(&roots, types),
                roots.len() != 3
            );
        }
    }

    #[test]
    fn native_signature_shared_pair_aliases_use_bounded_native_acceptance() {
        let mut graph = fixture_graph::<1>();
        let instance = graph
            .type_info()
            .instantiate_builtin_open("test.native_signature")
            .unwrap();
        graph.add_pure_primitive(
            Open {
                instance,
                equality: None,
            },
            None,
        );
        let reads = std::sync::Arc::new(AtomicUsize::new(0));
        let mut left = graph.get_sort_by_name("i64").unwrap().clone();
        let mut right = left.clone();
        for level in 0..24 {
            for (prefix, child) in [("A", left.clone()), ("B", right.clone())] {
                let name = format!("Shared{prefix}{level}");
                let args = [
                    ast::Expr::Var(ast::Span::Panic, child.name().into()),
                    ast::Expr::Var(ast::Span::Panic, child.name().into()),
                ];
                let native =
                    sort::PairSort::make_sort(graph.type_info(), name, &args, ast::Span::Panic)
                        .unwrap();
                let root: ArcSort = std::sync::Arc::new(CountedPair {
                    native,
                    reads: reads.clone(),
                });
                graph
                    .type_info()
                    .register_builtin_sort("Pair", root.clone(), vec![child.clone(), child])
                    .unwrap();
                graph.add_arcsort(root.clone(), ast::Span::Panic).unwrap();
                if prefix == "A" {
                    left = root;
                } else {
                    right = root;
                }
            }
        }
        let roots = [left.clone(), right, left];
        // No export_sort call: that is a separate existing recursive path.
        reads.store(0, Ordering::Relaxed);
        let types = graph.type_info();
        assert!(!types.get_prims("signature-open").unwrap()[0].accept(&roots, types));
        reads.store(0, Ordering::Relaxed);
        let instance = graph
            .type_info()
            .instantiate_builtin_signature("test.native_signature", &roots[..2], &[], &roots[2])
            .unwrap();
        graph.add_pure_primitive(Exact(instance), None);
        reads.store(0, Ordering::Relaxed);
        let types = graph.type_info();
        assert!(types.get_prims("signature-exact").unwrap()[0].accept(&roots, types));
        assert!(reads.load(Ordering::Relaxed) < 200_000);
    }

    #[test]
    fn native_signature_application_rejects_detached_function_roots() {
        let mut graph = EGraph::default();
        let integer = graph.get_sort_by_name("i64").unwrap().clone();
        let args = [
            ast::Expr::Call(ast::Span::Panic, "i64".into(), vec![]),
            ast::Expr::Var(ast::Span::Panic, "i64".into()),
        ];
        let function = sort::FunctionSort::make_sort(
            graph.type_info(),
            "Detached".into(),
            &args,
            ast::Span::Panic,
        )
        .unwrap();
        assert!(graph.get_sort_by_name("Detached").is_none());
        let roots = vec![function.clone(), integer.clone(), integer];
        let instance = graph
            .type_info()
            .instantiate_builtin_open("egglog.core.fn.apply")
            .unwrap();
        graph.add_pure_primitive(
            NativeApplication {
                instance,
                signature: roots.clone(),
            },
            None,
        );
        let types = graph.type_info();
        assert!(!types.get_prims("signature-application").unwrap()[0].accept(&roots, types));
        assert!(
            ResolvedCall::from_resolution(
                "signature-application",
                &roots,
                types,
                Context::Pure,
                &ast::Span::Panic
            )
            .is_err()
        );
        graph.add_arcsort(function, ast::Span::Panic).unwrap();
        // Installation also adds the standard Apply implementation, so use the
        // fixture's native alias to isolate this same-root membership control.
        let types = graph.type_info();
        assert!(types.get_prims("signature-application").unwrap()[0].accept(&roots, types));
        assert!(
            ResolvedCall::from_resolution(
                "signature-application",
                &roots,
                types,
                Context::Pure,
                &ast::Span::Panic
            )
            .is_ok()
        );
    }
}

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
                                ..Default::default()
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

mod manual_root_context {
    use super::*;

    const ROOT: &str = "test.manual_context_root";
    const HELPER: &str = "test.manual_context_helper";
    static HELPER_CALLS: AtomicUsize = AtomicUsize::new(0);

    #[derive(Clone)]
    struct Manual(std::sync::Arc<pb::Program>);

    impl Primitive for Manual {
        fn name(&self) -> &str {
            "manual-context-root"
        }
        fn get_type_constraints(&self, span: &ast::Span) -> Box<dyn constraint::TypeConstraint> {
            builtin::rooted_type_constraints(ROOT, &self.0, span)
        }
        fn builtin_definition(&self) -> Option<(&str, &pb::Program)> {
            Some((ROOT, &self.0))
        }
    }

    impl PurePrim for Manual {
        fn apply<'a, 'db>(&self, _: PureState<'a, 'db>, _: &[Value]) -> Option<Value> {
            panic!("manual context validation must not execute the root")
        }
    }

    struct Template;
    impl Presort for Template {
        fn presort_name() -> &'static str {
            "ManualContextTemplate"
        }
        fn reserved_primitives() -> Vec<&'static str> {
            vec![]
        }
        fn builtin_definitions() -> Vec<pb::Program> {
            vec![pb::Program {
                ir_version: 1,
                sorts: vec![pb::Sort {
                    kind: Some(pb::sort::Kind::Family(pb::HostSort {
                        name: "i64".into(),
                        args: vec![],
                    })),
                    ..Default::default()
                }],
                declarations: vec![pb::Declaration {
                    kind: Some(pb::declaration::Kind::HostPrimitive(pb::HostPrimitive {
                        name: HELPER.into(),
                        typing: Some(pb::host_primitive::Typing::Signature(
                            pb::GenericSignature {
                                output: Some(0),
                                ..Default::default()
                            },
                        )),
                    })),
                    ..Default::default()
                }],
                ..Default::default()
            }]
        }
        fn make_sort(
            _: &mut TypeInfo,
            _: String,
            _: &[ast::Expr],
            _: ast::Span,
        ) -> Result<ArcSort, TypeError> {
            panic!("default context validation must not instantiate the template")
        }
    }

    fn prepare(types: &TypeInfo) -> Result<pb::Program, String> {
        let integer = types.get_sort_by_name("i64").unwrap().clone();
        let generated = builtin::closed_signature(
            ROOT,
            "manual-context-root",
            &[("value", integer.clone())],
            &[],
            integer,
        )?;
        types.prepare_builtin_context(ROOT, "manual-context-root", generated, |types, key, draft| {
            let mut context = pb::Program::default();
            types.export_builtin_definition(HELPER, &mut context)?;
            let root = draft.declarations.iter_mut().find(|declaration| {
                matches!(&declaration.kind, Some(pb::declaration::Kind::HostPrimitive(primitive)) if primitive.name == key)
            }).unwrap();
            let Some(pb::declaration::Kind::HostPrimitive(primitive)) = &root.kind else {
                unreachable!()
            };
            let Some(pb::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
                unreachable!()
            };
            let default = draft.nodes.len() as u32;
            draft.nodes.push(pb::Node {
                sort_id: signature.output.unwrap(),
                kind: Some(pb::node::Kind::Call(pb::Call {
                    func: HELPER.into(),
                    args: vec![],
                })),
                ..Default::default()
            });
            root.bindings.as_mut().unwrap().python = Some(pb::PythonBindings {
                views: vec![pb::PythonCallable {
                    kind: pb::PythonCallKind::Function.into(),
                    path: vec!["test".into(), "manual_context_root".into()],
                    params: vec![pb::PythonParameter {
                        core_input: Some(0),
                        name: "value".into(),
                        default_expr: Some(default),
                    }],
                    ..Default::default()
                }],
            });
            builtin::definitions::reconcile_declarations(draft, &mut context)?;
            *draft = context;
            Ok(())
        })
    }

    #[test]
    fn explicit_manual_root_follows_helper_first_evidence() {
        let mut graph = EGraph::default();
        add_literal_prim!(
            &mut graph,
            "manual-context-helper"[id = "test.manual_context_helper"] = || -> i64 {
                {
                    HELPER_CALLS.fetch_add(1, Ordering::SeqCst);
                    17
                }
            }
        );
        let integer = graph.get_sort_by_name("i64").unwrap().clone();
        let helper = ResolvedCall::from_resolution(
            "manual-context-helper",
            std::slice::from_ref(&integer),
            graph.type_info(),
            Context::Pure,
            &ast::Span::Panic,
        )
        .unwrap();
        let definition = prepare(graph.type_info()).unwrap();
        let order = definition
            .declarations
            .iter()
            .filter_map(|declaration| match &declaration.kind {
                Some(pb::declaration::Kind::HostPrimitive(primitive)) => {
                    Some(primitive.name.as_str())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(order, [HELPER, ROOT]);
        let mut destination = pb::Program::default();
        assert!(builtin::import_definition(&definition, &mut destination).is_err());
        assert_eq!(destination, pb::Program::default());

        graph.add_pure_primitive(Manual(std::sync::Arc::new(definition)), None);
        let signature = [integer.clone(), integer.clone()];
        for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
            let root = ResolvedCall::from_resolution(
                ROOT,
                &signature,
                graph.type_info(),
                context,
                &ast::Span::Panic,
            );
            let alias = ResolvedCall::from_resolution(
                "manual-context-root",
                &signature,
                graph.type_info(),
                context,
                &ast::Span::Panic,
            );
            match (root, alias) {
                (Ok(root), Ok(alias)) => assert_eq!(root, alias),
                (Err(_), Err(_)) => (),
                _ => panic!("explicit root changed native context availability"),
            }
        }
        assert!(
            ResolvedCall::from_resolution(
                ROOT,
                &signature,
                graph.type_info(),
                Context::Pure,
                &ast::Span::Panic,
            )
            .is_ok()
        );
        assert!(
            ResolvedCall::from_resolution(
                ROOT,
                std::slice::from_ref(&integer),
                graph.type_info(),
                Context::Pure,
                &ast::Span::Panic,
            )
            .is_err()
        );
        assert_eq!(
            helper,
            ResolvedCall::from_resolution(
                "manual-context-helper",
                std::slice::from_ref(&integer),
                graph.type_info(),
                Context::Pure,
                &ast::Span::Panic,
            )
            .unwrap()
        );
        assert_eq!(
            graph
                .type_info()
                .get_prims("manual-context-helper")
                .unwrap()
                .len(),
            1
        );
        graph
            .type_info()
            .export_builtin_definition(ROOT, &mut destination)
            .unwrap();
        assert!(destination.nodes.iter().any(|node| {
            matches!(&node.kind, Some(pb::node::Kind::Call(call)) if call.func == HELPER)
        }));
        assert!(graph.type_info().builtin_catalog().is_ok());
        assert_eq!(HELPER_CALLS.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn malformed_manual_contexts_fail_before_alias_publication() {
        for mutation in [
            "unknown",
            "forged",
            "duplicate",
            "root",
            "family",
            "node",
            "unreachable",
        ] {
            let mut graph = EGraph::default();
            add_literal_prim!(
                &mut graph,
                "manual-context-helper"[id = "test.manual_context_helper"] = || -> i64 {
                    {
                        HELPER_CALLS.fetch_add(1, Ordering::SeqCst);
                        17
                    }
                }
            );
            let mut definition = prepare(graph.type_info()).unwrap();
            let helper = definition.declarations.iter().position(|declaration| {
                matches!(&declaration.kind, Some(pb::declaration::Kind::HostPrimitive(primitive)) if primitive.name == HELPER)
            }).unwrap();
            let root = definition.declarations.iter().position(|declaration| {
                matches!(&declaration.kind, Some(pb::declaration::Kind::HostPrimitive(primitive)) if primitive.name == ROOT)
            }).unwrap();
            match mutation {
                "unknown" => {
                    let Some(pb::declaration::Kind::HostPrimitive(primitive)) =
                        &mut definition.declarations[helper].kind
                    else {
                        unreachable!()
                    };
                    primitive.name = "test.unregistered_manual_helper".into();
                    for node in &mut definition.nodes {
                        if let Some(pb::node::Kind::Call(call)) = &mut node.kind {
                            call.func = "test.unregistered_manual_helper".into();
                        }
                    }
                }
                "forged" => definition.declarations[helper].doc = Some("forged helper".into()),
                "duplicate" => definition
                    .declarations
                    .push(definition.declarations[helper].clone()),
                "root" => {
                    let Some(pb::declaration::Kind::HostPrimitive(primitive)) =
                        &mut definition.declarations[root].kind
                    else {
                        unreachable!()
                    };
                    primitive.name = "test.wrong_manual_root".into();
                }
                "family" => {
                    let family = definition
                        .declarations
                        .iter_mut()
                        .find_map(|declaration| match &mut declaration.kind {
                            Some(pb::declaration::Kind::HostSortFamily(family)) => Some(family),
                            _ => None,
                        })
                        .unwrap();
                    family.arity += 1;
                }
                "node" => definition.nodes.clear(),
                "unreachable" => {
                    definition.declarations[root]
                        .bindings
                        .as_mut()
                        .unwrap()
                        .python = None
                }
                _ => unreachable!(),
            }
            graph.add_pure_primitive(Manual(std::sync::Arc::new(definition)), None);
            assert!(
                !graph.type_info().is_primitive("manual-context-root"),
                "{mutation}"
            );
            assert!(!graph.type_info().is_primitive(ROOT), "{mutation}");
            assert!(graph.type_info().builtin_catalog().is_err(), "{mutation}");
            assert_eq!(
                graph
                    .type_info()
                    .get_prims("manual-context-helper")
                    .unwrap()
                    .len(),
                1
            );
            assert_eq!(HELPER_CALLS.load(Ordering::SeqCst), 0);
        }
    }

    #[test]
    fn template_only_helper_cannot_supply_native_default_authority() {
        let mut graph = EGraph::default();
        graph
            .type_info()
            .add_presort::<Template>(ast::Span::Panic)
            .unwrap();
        let mut exported = pb::Program::default();
        graph
            .type_info()
            .export_builtin_definition(HELPER, &mut exported)
            .unwrap();
        assert!(exported.declarations.iter().any(|declaration| {
            matches!(&declaration.kind, Some(pb::declaration::Kind::HostPrimitive(primitive)) if primitive.name == HELPER)
        }));
        assert!(graph.type_info().get_prims(HELPER).is_none());
        let error = prepare(graph.type_info()).unwrap_err();
        assert!(error.contains("no registered implementation"), "{error}");
        assert!(!graph.type_info().is_primitive("manual-context-root"));
        assert!(!graph.type_info().is_primitive(ROOT));
        assert!(graph.type_info().builtin_catalog().is_ok());
    }
}
