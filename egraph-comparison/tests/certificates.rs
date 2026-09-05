use egraph_comparison::{
    Certificate, Class, Database, Function, FunctionKind, Row, Side, Term, certificate, verify,
};

fn graph(rows: &[(&str, &[&str], &str)]) -> Database {
    let mut db = Database::default();
    for &(op, inputs, output) in rows {
        for id in inputs.iter().copied().chain([output]) {
            db.classes.insert(
                id.into(),
                Class {
                    sort: "E".into(),
                    literal: None,
                },
            );
        }
        db.functions.insert(
            op.into(),
            Function {
                kind: FunctionKind::Constructor,
                inputs: vec!["E".into(); inputs.len()],
                output: "E".into(),
            },
        );
        db.rows.push(Row {
            function: op.into(),
            inputs: inputs.iter().map(|s| (*s).into()).collect(),
            output: output.into(),
            subsumed: false,
        });
    }
    db
}

fn witness(left: &Database, right: &Database) -> Certificate {
    let cert = certificate(left, right).unwrap().unwrap();
    assert!(verify(&cert, left, right).unwrap());
    let json = serde_json::to_string(&cert).unwrap();
    let cert: Certificate = serde_json::from_str(&json).unwrap();
    assert!(verify(&cert, left, right).unwrap());
    assert!(!verify(&cert, left, left).unwrap());
    cert
}

#[test]
fn missing_terms_in_either_direction() {
    let left = graph(&[("a", &[], "x"), ("f", &["x"], "y")]);
    let right = graph(&[("a", &[], "renamed")]);
    assert!(matches!(
        witness(&left, &right),
        Certificate::MissingTerm {
            side: Side::Left,
            ..
        }
    ));
    assert!(matches!(
        witness(&right, &left),
        Certificate::MissingTerm {
            side: Side::Right,
            ..
        }
    ));
    assert!(certificate(&left, &left).unwrap().is_none());
}

#[test]
fn equality_witnesses_in_both_directions() {
    let left = graph(&[("a", &[], "x"), ("b", &[], "x"), ("f", &["x"], "y")]);
    let right = graph(&[
        ("a", &[], "x"),
        ("b", &[], "z"),
        ("f", &["x"], "y"),
        ("f", &["z"], "y"),
    ]);
    assert!(matches!(
        witness(&left, &right),
        Certificate::UnequalTerms { .. }
    ));
    assert!(matches!(
        witness(&right, &left),
        Certificate::UnequalTerms { .. }
    ));
}

#[test]
fn productive_cycles_have_finite_witnesses() {
    let left = graph(&[("a", &[], "x"), ("f", &["x"], "x")]);
    let right = graph(&[("a", &[], "x"), ("f", &["x"], "y")]);
    assert!(matches!(
        witness(&left, &right),
        Certificate::UnequalTerms { .. }
    ));
}

#[test]
fn pure_cycles_and_holes_have_structural_certificates() {
    let left = graph(&[("f", &["x"], "x")]);
    let right = graph(&[("g", &["x"], "x")]);
    assert!(matches!(
        witness(&left, &right),
        Certificate::Structure { .. }
    ));
    let right = graph(&[("f", &["hole"], "x")]);
    assert!(matches!(
        witness(&left, &right),
        Certificate::Structure { .. }
    ));
}

#[test]
fn declarations_rows_and_subsumption_have_certificates() {
    let left = graph(&[("a", &[], "x")]);
    let mut right = left.clone();
    right.functions.insert(
        "empty".into(),
        Function {
            kind: FunctionKind::Function,
            inputs: vec![],
            output: "E".into(),
        },
    );
    assert!(matches!(
        witness(&left, &right),
        Certificate::Declaration { .. }
    ));
    let mut left = right.clone();
    left.rows.push(Row {
        function: "empty".into(),
        inputs: vec![],
        output: "x".into(),
        subsumed: false,
    });
    assert!(matches!(witness(&left, &right), Certificate::Row { .. }));
    let mut right = left.clone();
    right.rows[0].subsumed = true;
    assert!(matches!(witness(&left, &right), Certificate::Row { .. }));
}

#[test]
fn rejects_invalid_dags_roots_and_structural_claims() {
    let left = graph(&[("a", &[], "x")]);
    let right = Database::default();
    for terms in [
        vec![Term::Apply {
            function: "f".into(),
            inputs: vec![0],
        }],
        vec![Term::Apply {
            function: "a".into(),
            inputs: vec![],
        }],
    ] {
        let cert = Certificate::MissingTerm {
            side: Side::Left,
            terms,
            term: 99,
        };
        assert!(!verify(&cert, &left, &right).unwrap());
    }
    for class in ["x", "unknown"] {
        let cert = Certificate::Structure {
            side: Side::Left,
            class: class.into(),
            rounds: usize::MAX,
        };
        assert!(!verify(&cert, &left, &right).unwrap());
    }
    let mut db = graph(&[("a", &[], "x")]);
    db.functions.get_mut("a").unwrap().kind = FunctionKind::Function;
    let cert = Certificate::MissingTerm {
        side: Side::Left,
        terms: vec![Term::Apply {
            function: "a".into(),
            inputs: vec![],
        }],
        term: 0,
    };
    assert!(!verify(&cert, &db, &right).unwrap());
}

#[test]
fn shared_subterms_stay_linear_in_certificate_size() {
    let mut left = graph(&[("a", &[], "0")]);
    left.functions.insert(
        "pair".into(),
        Function {
            kind: FunctionKind::Constructor,
            inputs: vec!["E".into(); 2],
            output: "E".into(),
        },
    );
    for i in 1..60 {
        left.classes.insert(
            i.to_string(),
            Class {
                sort: "E".into(),
                literal: None,
            },
        );
        left.rows.push(Row {
            function: "pair".into(),
            inputs: vec![(i - 1).to_string(); 2],
            output: i.to_string(),
            subsumed: false,
        });
    }
    let mut right = left.clone();
    right.rows.pop();
    let cert = witness(&left, &right);
    let Certificate::MissingTerm { terms, .. } = cert else {
        panic!("expected a finite term");
    };
    assert_eq!(terms.len(), 60); // Expanded tree would contain 2^60 - 1 nodes.
}
