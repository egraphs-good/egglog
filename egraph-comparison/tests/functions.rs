use egraph_comparison::{Certificate, Database, certificate, compare, verify};

fn db(rows: serde_json::Value) -> Database {
    serde_json::from_value(serde_json::json!({
        "version":1,
        "classes":{"x":{"sort":"E"},"y":{"sort":"E"}},
        "functions":{
            "f":{"kind":"function","inputs":["E"],"output":"E"},
            "g":{"kind":"function","inputs":["E"],"output":"E"}
        },
        "rows": rows,
    }))
    .unwrap()
}

#[test]
fn function_structure_survives_without_constructor_terms() {
    let left = db(serde_json::json!([
        {"function":"f","inputs":["x"],"output":"y"},
        {"function":"g","inputs":["x"],"output":"x"}
    ]));
    let right = db(serde_json::json!([
        {"function":"f","inputs":["x"],"output":"y"},
        {"function":"g","inputs":["y"],"output":"y"}
    ]));
    let result = compare(&left, &right).unwrap();
    assert!(result.terms_equal);
    assert!(!result.database_equal);
    for (a, b) in [(&left, &right), (&right, &left)] {
        let cert = certificate(a, b).unwrap().unwrap();
        assert!(matches!(cert, Certificate::Row { .. }));
        assert!(verify(&cert, a, b).unwrap());
        assert!(!verify(&cert, a, a).unwrap());
    }
}

#[test]
fn bisimilar_function_cycles_ignore_ids_and_multiplicity() {
    let left = db(serde_json::json!([
        {"function":"f","inputs":["x"],"output":"x"}
    ]));
    let right = db(serde_json::json!([
        {"function":"f","inputs":["x"],"output":"y"},
        {"function":"f","inputs":["y"],"output":"x"}
    ]));
    assert!(compare(&left, &right).unwrap().database_equal);
    assert!(certificate(&left, &right).unwrap().is_none());
}

#[test]
fn function_members_change_database_but_never_term_witnesses() {
    let mut left = db(serde_json::json!([
        {"function":"f","inputs":["x"],"output":"y"}
    ]));
    left.functions.get_mut("g").unwrap().kind = egraph_comparison::FunctionKind::Constructor;
    left.rows.push(egraph_comparison::Row {
        function: "g".into(),
        inputs: vec!["x".into()],
        output: "x".into(),
        subsumed: false,
    });
    let mut right = left.clone();
    right.rows[0].output = "x".into();
    let result = compare(&left, &right).unwrap();
    assert!(result.terms_equal);
    assert!(!result.database_equal);
    assert!(matches!(
        certificate(&left, &right).unwrap().unwrap(),
        Certificate::Row { .. }
    ));
}
