use egglog::{
    EGraph, Error, RawValues, Read,
    ast::{Command, FunctionSubtype, Schema},
    span,
};

#[test]
fn relation_origin_survives_native_ast_execution() -> Result<(), Error> {
    let mut graph = EGraph::default();
    graph.run_program(vec![Command::Relation {
        span: span!(),
        name: "LooksLikeConstructor".into(),
        inputs: vec!["i64".into()],
    }])?;

    let relation = graph.get_function("LooksLikeConstructor").unwrap();
    assert!(relation.func_type().is_relation);
    assert_eq!(relation.func_type().subtype, FunctionSubtype::Constructor);
    assert!(relation.func_type().output.is_eq_sort());
    assert!(relation.can_subsume());
    let generated_sort = relation.func_type().output.name().to_owned();

    // Even the exact output sort used by a relation does not make another
    // constructor a relation. Neither its spelling nor its physical shape does.
    graph.run_program(vec![Command::Constructor {
        span: span!(),
        name: "looks-like-relation".into(),
        schema: Schema::new(vec!["i64".into()], generated_sort),
        cost: None,
        unextractable: false,
        hidden: false,
        let_binding: false,
        term_constructor: None,
    }])?;
    assert!(
        !graph
            .get_function("looks-like-relation")
            .unwrap()
            .func_type()
            .is_relation
    );
    assert!(
        graph
            .type_info()
            .get_func_type("LooksLikeConstructor")
            .unwrap()
            .is_relation
    );

    graph.parse_and_run_program(
        None,
        "(LooksLikeConstructor 1)
         (LooksLikeConstructor 1)
         (check (LooksLikeConstructor 1))
         (fail (check (LooksLikeConstructor 2)))
         (delete (LooksLikeConstructor 1))
         (fail (check (LooksLikeConstructor 1)))",
    )?;
    Ok(())
}

#[test]
fn parsed_relation_metadata_is_scope_and_clone_local() -> Result<(), Error> {
    let mut graph = EGraph::default();
    graph.parse_and_run_program(
        None,
        "(datatype Term (Leaf i64))
         (relation Keep (Term))
         (function NotARelation (Term) i64 :merge old)
         (let $saved (Leaf 1))
         (Keep $saved)",
    )?;
    for (name, function) in graph.functions_iter() {
        assert_eq!(function.func_type().is_relation, name == "Keep", "{name}");
    }

    graph.push();
    graph.parse_and_run_program(None, "(relation Temporary (i64)) (Temporary 2)")?;
    let mut clone = graph.clone();
    assert!(
        clone
            .get_function("Temporary")
            .unwrap()
            .func_type()
            .is_relation
    );

    graph.pop()?;
    assert!(graph.get_function("Temporary").is_none());
    assert!(graph.type_info().get_func_type("Temporary").is_none());
    graph.parse_and_run_program(None, "(constructor Temporary (i64) Term)")?;
    assert!(
        !graph
            .get_function("Temporary")
            .unwrap()
            .func_type()
            .is_relation
    );
    assert!(
        clone
            .get_function("Temporary")
            .unwrap()
            .func_type()
            .is_relation
    );

    clone.pop()?;
    assert!(clone.get_function("Temporary").is_none());
    assert!(clone.get_function("Keep").unwrap().func_type().is_relation);
    assert!(graph.get_function("Keep").unwrap().func_type().is_relation);
    Ok(())
}

#[test]
fn relation_execution_in_term_and_proof_modes_is_unchanged() -> Result<(), Error> {
    for mut graph in [EGraph::new_with_term_encoding(), EGraph::new_with_proofs()] {
        graph.parse_and_run_program(
            None,
            "(datatype Term (Leaf i64))
             (relation Seen (Term))
             (relation Copied (Term))
             (let $leaf (Leaf 1))
             (Seen $leaf)
             (rule ((Seen x)) ((Copied x)))
             (run 1)
             (check (Copied $leaf))",
        )?;
    }
    Ok(())
}

#[test]
fn term_encoding_mode_is_available_on_empty_graphs_and_scopes() -> Result<(), Error> {
    for (mut graph, term_encoding, proofs) in [
        (EGraph::default(), false, false),
        (EGraph::new_with_term_encoding(), true, false),
        (EGraph::new_with_proofs(), true, true),
    ] {
        assert_eq!(graph.is_term_encoding_enabled(), term_encoding);
        assert_eq!(graph.are_proofs_enabled(), proofs);
        graph.push();
        let mut clone = graph.clone();
        assert_eq!(clone.is_term_encoding_enabled(), term_encoding);
        graph.pop()?;
        clone.pop()?;
        assert_eq!(graph.is_term_encoding_enabled(), term_encoding);
        assert_eq!(clone.is_term_encoding_enabled(), term_encoding);
        assert_eq!(graph.are_proofs_enabled(), proofs);
        assert_eq!(clone.are_proofs_enabled(), proofs);
    }
    Ok(())
}

#[test]
fn canonical_values_follow_unions_without_changing_base_or_container_ids() -> Result<(), Error> {
    let mut graph = EGraph::default();
    graph.parse_and_run_program(
        None,
        "(datatype Term (Leaf i64))
         (sort Terms (Vec Term))
         (function Saved () Terms :no-merge)
         (set (Saved) (vec-of (Leaf 1) (Leaf 2)))",
    )?;
    let (left, right, container) = graph.read(|reader| -> Result<_, Error> {
        Ok((
            reader.eclass_of("Leaf", (1_i64,))?.unwrap(),
            reader.eclass_of("Leaf", (2_i64,))?.unwrap(),
            reader.lookup("Saved", RawValues(vec![]))?.unwrap(),
        ))
    })?;
    let term_sort = graph.get_sort_by_name("Term").unwrap().clone();
    let container_sort = graph.get_sort_by_name("Terms").unwrap().clone();
    let integer_sort = graph.get_sort_by_name("i64").unwrap().clone();
    let integer = graph.base_to_value(1_i64);
    let before = graph.num_tuples();
    assert_ne!(
        graph.canonical_value(&term_sort, left),
        graph.canonical_value(&term_sort, right)
    );
    assert_eq!(graph.canonical_value(&integer_sort, integer), integer);
    assert_eq!(graph.canonical_value(&container_sort, container), container);
    assert_eq!(graph.num_tuples(), before);

    graph.parse_and_run_program(None, "(union (Leaf 1) (Leaf 2))")?;
    let before = graph.num_tuples();
    assert_eq!(
        graph.canonical_value(&term_sort, left),
        graph.canonical_value(&term_sort, right)
    );
    assert_eq!(graph.canonical_value(&integer_sort, integer), integer);
    assert_eq!(graph.canonical_value(&container_sort, container), container);
    assert_eq!(graph.num_tuples(), before);
    Ok(())
}
