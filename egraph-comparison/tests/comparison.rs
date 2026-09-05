use egraph_comparison::{Class, Database, Function, FunctionKind, Row, compare};
use std::collections::BTreeSet;

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

#[test]
fn renaming_order_duplicates_and_cycles() {
    let left = graph(&[("f", &["x"], "x")]);
    let mut right = graph(&[("f", &["b"], "a"), ("f", &["a"], "b"), ("f", &["c"], "c")]);
    right.rows.reverse();
    right.rows.push(right.rows[0].clone());
    assert!(compare(&left, &right).unwrap().database_equal);
}

#[test]
fn missing_terms_and_different_equalities() {
    let left = graph(&[("a", &[], "x"), ("b", &[], "x")]);
    let split = graph(&[("a", &[], "x"), ("b", &[], "y")]);
    let absent = graph(&[("a", &[], "x")]);
    assert!(!compare(&left, &split).unwrap().terms_equal);
    assert!(!compare(&left, &absent).unwrap().terms_equal);
    assert!(
        compare(&Database::default(), &Database::default())
            .unwrap()
            .database_equal
    );
}

#[test]
fn ordered_children_sorts_and_empty_classes() {
    let left = graph(&[("a", &[], "x"), ("b", &[], "y"), ("p", &["x", "y"], "z")]);
    let mut right = left.clone();
    right.rows[2].inputs.reverse();
    assert!(!compare(&left, &right).unwrap().terms_equal);
    let left = graph(&[("f", &["hole"], "root")]);
    let right = graph(&[("f", &["root"], "root")]);
    assert!(!compare(&left, &right).unwrap().terms_equal);
    let mut right = left.clone();
    right.classes.get_mut("hole").unwrap().sort = "Other".into();
    right.functions.get_mut("f").unwrap().inputs = vec!["Other".into()];
    assert!(!compare(&left, &right).unwrap().terms_equal);
}

#[test]
fn function_values_and_subsumption_do_not_create_terms() {
    let mut left = graph(&[("a", &[], "x"), ("cost", &["x"], "one")]);
    left.functions.get_mut("cost").unwrap().kind = FunctionKind::Function;
    left.functions.get_mut("cost").unwrap().output = "i64".into();
    left.classes.insert(
        "one".into(),
        Class {
            sort: "i64".into(),
            literal: Some("1".into()),
        },
    );
    let mut right = left.clone();
    right.classes.get_mut("one").unwrap().literal = Some("2".into());
    let result = compare(&left, &right).unwrap();
    assert!(result.terms_equal);
    assert!(!result.database_equal);
    right = left.clone();
    right.rows[0].subsumed = true;
    let result = compare(&left, &right).unwrap();
    assert!(result.terms_equal);
    assert!(!result.database_equal);
    right = left.clone();
    right.functions.insert(
        "empty".into(),
        Function {
            kind: FunctionKind::Function,
            inputs: vec![],
            output: "E".into(),
        },
    );
    assert!(compare(&left, &right).unwrap().terms_equal);
    assert!(!compare(&left, &right).unwrap().database_equal);
}

#[test]
fn validation_errors() {
    let db = graph(&[("a", &[], "x"), ("f", &["x"], "y")]);
    let mut bad = db.clone();
    bad.version = 9;
    assert!(bad.validate().unwrap_err().to_string().contains("version"));
    bad = db.clone();
    bad.rows[1].inputs[0] = "missing".into();
    assert!(
        bad.validate()
            .unwrap_err()
            .to_string()
            .contains("unknown class")
    );
    bad = db.clone();
    bad.rows[1].inputs.clear();
    assert!(bad.validate().unwrap_err().to_string().contains("arity"));
    bad = db.clone();
    bad.rows.push(Row {
        output: "y".into(),
        ..bad.rows[0].clone()
    });
    assert!(
        bad.validate()
            .unwrap_err()
            .to_string()
            .contains("conflicting")
    );
    bad = db.clone();
    bad.classes.get_mut("x").unwrap().sort = "other".into();
    assert!(bad.validate().unwrap_err().to_string().contains("sort"));
    bad = db.clone();
    bad.functions.clear();
    assert!(
        bad.validate()
            .unwrap_err()
            .to_string()
            .contains("undeclared")
    );
    bad = db.clone();
    bad.classes.get_mut("x").unwrap().literal = Some("0".into());
    assert!(bad.validate().unwrap_err().to_string().contains("literal"));
    bad.classes.get_mut("y").unwrap().literal = Some("0".into());
    assert!(
        bad.validate()
            .unwrap_err()
            .to_string()
            .contains("duplicate literal")
    );
    let mut json = serde_json::to_value(&db).unwrap();
    json["typo"] = true.into();
    assert!(serde_json::from_value::<Database>(json).is_err());
    let roundtrip: Database = serde_json::from_str(&serde_json::to_string(&db).unwrap()).unwrap();
    assert!(compare(&db, &roundtrip).unwrap().database_equal);
}

// Independent greatest-fixed-point relation elimination, without partition IDs.
fn oracle(left: &Database, right: &Database) -> bool {
    let mut relation: BTreeSet<_> = left
        .classes
        .keys()
        .flat_map(|l| right.classes.keys().map(move |r| (l.clone(), r.clone())))
        .collect();
    loop {
        let previous = relation.clone();
        relation.retain(|(l, r)| {
            let lc = &left.classes[l];
            let rc = &right.classes[r];
            if lc.sort != rc.sort || lc.literal != rc.literal {
                return false;
            }
            let lr: Vec<_> = left.rows.iter().filter(|row| &row.output == l).collect();
            let rr: Vec<_> = right.rows.iter().filter(|row| &row.output == r).collect();
            let matches = |a: &&Row, b: &&Row| {
                a.function == b.function
                    && a.inputs.len() == b.inputs.len()
                    && a.inputs
                        .iter()
                        .zip(&b.inputs)
                        .all(|(x, y)| previous.contains(&(x.clone(), y.clone())))
            };
            lr.iter().all(|a| rr.iter().any(|b| matches(a, b)))
                && rr.iter().all(|b| lr.iter().any(|a| matches(a, b)))
        });
        if previous == relation {
            break;
        }
    }
    left.rows.iter().all(|a| {
        right
            .rows
            .iter()
            .any(|b| relation.contains(&(a.output.clone(), b.output.clone())))
    }) && right.rows.iter().all(|b| {
        left.rows
            .iter()
            .any(|a| relation.contains(&(a.output.clone(), b.output.clone())))
    })
}

#[test]
fn agrees_with_relation_oracle_on_small_cyclic_graphs() {
    fn generated(mut seed: usize) -> Database {
        let mut db = Database::default();
        for i in 0..4 {
            db.classes.insert(
                i.to_string(),
                Class {
                    sort: "E".into(),
                    literal: None,
                },
            );
        }
        for op in ["f", "g"] {
            db.functions.insert(
                op.into(),
                Function {
                    kind: FunctionKind::Constructor,
                    inputs: vec!["E".into()],
                    output: "E".into(),
                },
            );
            for input in 0..4 {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                if seed.is_multiple_of(7) {
                    continue;
                }
                db.rows.push(Row {
                    function: op.into(),
                    inputs: vec![input.to_string()],
                    output: ((seed >> 8) % 4).to_string(),
                    subsumed: false,
                });
            }
        }
        db
    }
    for seed in 0..100 {
        let left = generated(seed);
        let right = generated(seed / 2);
        assert_eq!(
            compare(&left, &right).unwrap().terms_equal,
            oracle(&left, &right),
            "seed {seed}"
        );
        assert!(compare(&left, &left).unwrap().database_equal);
    }
}

#[test]
fn propagates_a_difference_through_a_long_chain() {
    let mut left = graph(&[("a", &[], "0")]);
    left.functions.insert(
        "f".into(),
        Function {
            kind: FunctionKind::Constructor,
            inputs: vec!["E".into()],
            output: "E".into(),
        },
    );
    for i in 1..80 {
        left.classes.insert(
            i.to_string(),
            Class {
                sort: "E".into(),
                literal: None,
            },
        );
        left.rows.push(Row {
            function: "f".into(),
            inputs: vec![(i - 1).to_string()],
            output: i.to_string(),
            subsumed: false,
        });
    }
    let mut right = left.clone();
    right.rows.last_mut().unwrap().output = "78".into();
    assert!(!compare(&left, &right).unwrap().terms_equal);
    assert!(compare(&left, &left).unwrap().refinement_rounds >= 79);
}
