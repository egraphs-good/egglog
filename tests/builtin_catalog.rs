use egglog::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

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
            .any(|name| name == "Vec")
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
        ["egglog.core.f64.add", "egglog.core.i64.add", "test.counted"]
    );
    graph
        .parse_and_run_program(None, "(check (= (counted 7) 7))")
        .unwrap();
    assert!(CALLS.load(Ordering::SeqCst) > 0);
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
