use egraph_comparison::{
    Class, Database, Function, FunctionKind, Row, certificate, compare, verify,
};

fn wide() -> Database {
    let mut db = Database::default();
    for i in 0..8 {
        db.classes.insert(
            i.to_string(),
            Class {
                sort: "i64".into(),
                literal: Some(i.to_string()),
            },
        );
    }
    db.classes.insert(
        "out".into(),
        Class {
            sort: "E".into(),
            literal: None,
        },
    );
    db.functions.insert(
        "wide".into(),
        Function {
            kind: FunctionKind::Constructor,
            inputs: vec!["i64".into(); 8],
            output: "E".into(),
        },
    );
    db.rows.push(Row {
        function: "wide".into(),
        inputs: (0..8).map(|i| i.to_string()).collect(),
        output: "out".into(),
        subsumed: false,
    });
    db
}

#[test]
fn spilled_signatures_preserve_argument_order_and_certificates() {
    let left = wide();
    let mut right = left.clone();
    right.rows[0].inputs.swap(6, 7);
    assert!(!compare(&left, &right).unwrap().terms_equal);
    let witness = certificate(&left, &right).unwrap().unwrap();
    assert!(verify(&witness, &left, &right).unwrap());
    assert!(!verify(&witness, &left, &left).unwrap());
}

#[test]
fn interned_labels_include_arity_and_kind() {
    let left = wide();
    let mut right = left.clone();
    right.rows[0].inputs.pop();
    right.functions.get_mut("wide").unwrap().inputs.pop();
    assert!(!compare(&left, &right).unwrap().terms_equal);
    let mut right = left.clone();
    right.functions.get_mut("wide").unwrap().kind = FunctionKind::Function;
    assert!(!compare(&left, &right).unwrap().terms_equal);
    assert!(!compare(&left, &right).unwrap().database_equal);
}

#[test]
fn full_database_root_coverage_detects_subsumption_and_missing_function_rows() {
    let mut left = wide();
    left.functions.insert(
        "analysis".into(),
        Function {
            kind: FunctionKind::Function,
            inputs: vec!["E".into()],
            output: "i64".into(),
        },
    );
    left.rows.push(Row {
        function: "analysis".into(),
        inputs: vec!["out".into()],
        output: "0".into(),
        subsumed: false,
    });
    let mut right = left.clone();
    right.rows.pop();
    let result = compare(&left, &right).unwrap();
    assert!(result.terms_equal);
    assert!(!result.database_equal);
    let mut right = left.clone();
    right.rows[0].subsumed = true;
    let result = compare(&left, &right).unwrap();
    assert!(result.terms_equal);
    assert!(!result.database_equal);
    right = left.clone();
    right.rows.reverse();
    right.rows.push(right.rows[0].clone());
    assert!(compare(&left, &right).unwrap().database_equal);
}
