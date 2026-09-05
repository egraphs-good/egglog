use egraph_comparison::{
    CanonicalMode, Class, Database, Function, FunctionKind, Row, canonicalize, compare,
};
use std::collections::BTreeMap;

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

fn agrees(left: &Database, right: &Database) {
    let result = compare(left, right).unwrap();
    for (mode, expected) in [
        (CanonicalMode::Terms, result.terms_equal),
        (CanonicalMode::Database, result.database_equal),
    ] {
        assert_eq!(
            canonicalize(left, mode).unwrap() == canonicalize(right, mode).unwrap(),
            expected,
            "{mode:?}: {left:?} vs {right:?}"
        );
    }
}

fn renamed(db: &Database) -> Database {
    let names: BTreeMap<_, _> = db
        .classes
        .keys()
        .enumerate()
        .map(|(i, id)| (id.clone(), format!("renamed-{:06}", db.classes.len() - i)))
        .collect();
    let mut result = db.clone();
    result.classes = db
        .classes
        .iter()
        .map(|(id, class)| (names[id].clone(), class.clone()))
        .collect();
    for row in &mut result.rows {
        row.output = names[&row.output].clone();
        for input in &mut row.inputs {
            *input = names[input].clone();
        }
    }
    result.rows.reverse();
    result.rows.extend(result.rows.clone());
    result
}

#[test]
fn canonical_format_is_versioned_and_byte_stable() {
    let db = graph(&[("f", &["x"], "x")]);
    assert_eq!(
        String::from_utf8(canonicalize(&db, CanonicalMode::Terms).unwrap()).unwrap(),
        "{\"format\":\"egraph-comparison-canonical\",\"version\":1,\"mode\":\"terms\",\"sorts\":[\"E\"],\"labels\":[{\"Call\":[\"f\",{\"kind\":\"constructor\",\"inputs\":[\"E\"],\"output\":\"E\"},false]}],\"roots\":[0],\"classes\":[{\"sort\":0,\"nodes\":[[0,0]]}]}\n"
    );
    let empty = canonicalize(&Database::default(), CanonicalMode::Database).unwrap();
    assert_eq!(
        String::from_utf8(empty).unwrap(),
        "{\"format\":\"egraph-comparison-canonical\",\"version\":1,\"mode\":\"database\",\"declarations\":{},\"sorts\":[],\"labels\":[],\"roots\":[],\"classes\":[]}\n"
    );
}

#[test]
fn quotients_cycles_and_ignores_unobserved_classes() {
    let left = graph(&[("f", &["x"], "x")]);
    let mut right = graph(&[("f", &["b"], "a"), ("f", &["a"], "b"), ("f", &["c"], "c")]);
    right.classes.insert(
        "unused".into(),
        Class {
            sort: "AAA".into(),
            literal: Some("zzz".into()),
        },
    );
    agrees(&left, &right);
    assert_eq!(
        canonicalize(&left, CanonicalMode::Database).unwrap(),
        canonicalize(&right, CanonicalMode::Database).unwrap()
    );
    agrees(&right, &renamed(&right));
    right.functions.insert(
        "AAA".into(),
        Function {
            kind: FunctionKind::Constructor,
            inputs: vec![],
            output: "Unused".into(),
        },
    );
    agrees(&left, &right);
    assert_eq!(
        canonicalize(&left, CanonicalMode::Terms).unwrap(),
        canonicalize(&right, CanonicalMode::Terms).unwrap()
    );
    let empty = Database {
        classes: right.classes,
        ..Database::default()
    };
    agrees(&Database::default(), &empty);
}

#[test]
fn terms_functions_subsumption_literals_and_schemas() {
    let mut left = graph(&[("a", &[], "x"), ("cost", &["x"], "n")]);
    left.functions.get_mut("cost").unwrap().kind = FunctionKind::Function;
    left.functions.get_mut("cost").unwrap().output = "i64".into();
    left.classes.insert(
        "n".into(),
        Class {
            sort: "i64".into(),
            literal: Some("1".into()),
        },
    );
    agrees(&left, &renamed(&left));
    let mut right = left.clone();
    right.classes.get_mut("n").unwrap().literal = Some("2".into());
    agrees(&left, &right);
    right = left.clone();
    right.rows[0].subsumed = true;
    agrees(&left, &right);
    right = left.clone();
    right.rows[1].subsumed = true;
    agrees(&left, &right);
    right = left.clone();
    right.functions.insert(
        "empty".into(),
        Function {
            kind: FunctionKind::Function,
            inputs: vec![],
            output: "E".into(),
        },
    );
    agrees(&left, &right);
    right = left.clone();
    right.functions.get_mut("a").unwrap().kind = FunctionKind::Function;
    agrees(&left, &right);
    assert_ne!(
        canonicalize(&left, CanonicalMode::Terms).unwrap(),
        canonicalize(&left, CanonicalMode::Database).unwrap()
    );
}

#[test]
fn preserves_equalities_ordered_children_and_holes() {
    let left = graph(&[("a", &[], "x"), ("b", &[], "x")]);
    agrees(&left, &graph(&[("a", &[], "x"), ("b", &[], "y")]));
    let left = graph(&[
        ("a", &[], "x"),
        ("b", &[], "y"),
        ("p", &["x", "y", "x", "y"], "z"),
    ]);
    let mut right = left.clone();
    right.rows[2].inputs.reverse();
    agrees(&left, &right);
    agrees(&left, &renamed(&left));
    let left = graph(&[("f", &["hole"], "root")]);
    agrees(&left, &graph(&[("f", &["root"], "root")]));
    let mut right = left.clone();
    right.classes.get_mut("hole").unwrap().sort = "Other".into();
    right.functions.get_mut("f").unwrap().inputs = vec!["Other".into()];
    agrees(&left, &right);
}

#[test]
fn quotient_can_have_multiple_outputs_for_the_same_quotient_inputs() {
    // x and y collapse, but g(x) and g(y) do not: only g(x) also contains a.
    let left = graph(&[
        ("f", &["x"], "x"),
        ("f", &["y"], "y"),
        ("g", &["x"], "u"),
        ("g", &["y"], "v"),
        ("a", &[], "u"),
    ]);
    agrees(&left, &renamed(&left));
    let bytes = canonicalize(&left, CanonicalMode::Database).unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["classes"].as_array().unwrap().len(), 3);
    assert!(serde_json::from_slice::<Database>(&bytes).is_err());
}

#[test]
fn rejects_invalid_inputs() {
    let mut db = graph(&[("a", &[], "x")]);
    db.version = 99;
    for mode in [CanonicalMode::Terms, CanonicalMode::Database] {
        assert!(canonicalize(&db, mode).is_err());
    }
    db.version = 1;
    db.rows[0].output = "missing".into();
    assert!(canonicalize(&db, CanonicalMode::Database).is_err());
}

#[test]
fn independent_canonicalizations_agree_with_joint_refinement() {
    fn generated(mut seed: usize) -> Database {
        let mut db = graph(&[("a", &[], "0")]);
        for i in 1..6 {
            db.classes.insert(
                i.to_string(),
                Class {
                    sort: "E".into(),
                    literal: None,
                },
            );
        }
        for (op, kind) in [
            ("f", FunctionKind::Constructor),
            ("g", FunctionKind::Function),
        ] {
            db.functions.insert(
                op.into(),
                Function {
                    kind,
                    inputs: vec!["E".into()],
                    output: "E".into(),
                },
            );
            for input in 0..6 {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                if seed.is_multiple_of(7) {
                    continue;
                }
                db.rows.push(Row {
                    function: op.into(),
                    inputs: vec![input.to_string()],
                    output: ((seed >> 8) % 6).to_string(),
                    subsumed: seed.is_multiple_of(11),
                });
            }
        }
        db
    }
    for seed in 0..200 {
        let left = generated(seed);
        agrees(&left, &renamed(&left));
        agrees(&left, &generated(seed / 2));
    }
    let mut chain = graph(&[("a", &[], "0")]);
    chain.functions.insert(
        "f".into(),
        Function {
            kind: FunctionKind::Constructor,
            inputs: vec!["E".into()],
            output: "E".into(),
        },
    );
    for i in 1..100 {
        chain.classes.insert(
            i.to_string(),
            Class {
                sort: "E".into(),
                literal: None,
            },
        );
        chain.rows.push(Row {
            function: "f".into(),
            inputs: vec![(i - 1).to_string()],
            output: i.to_string(),
            subsumed: false,
        });
    }
    agrees(&chain, &renamed(&chain));
    let mut other = chain.clone();
    other.rows.last_mut().unwrap().output = "98".into();
    agrees(&chain, &other);
}

#[test]
fn cli_canonical_modes_and_argument_errors() {
    use std::{fs, process::Command};
    let dir = std::env::temp_dir().join(format!("canonical-cli-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("graph.json");
    let db = graph(&[("a", &[], "x")]);
    fs::write(&path, serde_json::to_vec(&db).unwrap()).unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_egraph-comparison"))
            .arg(&path)
            .args(args)
            .output()
            .unwrap()
    };
    for (args, mode) in [
        (vec!["--canonical"], CanonicalMode::Database),
        (vec!["--canonical", "--terms-only"], CanonicalMode::Terms),
    ] {
        let output = run(&args);
        assert!(output.status.success());
        assert_eq!(output.stdout, canonicalize(&db, mode).unwrap());
    }
    for args in [
        vec![],
        vec!["--canonical", "--certificate"],
        vec!["--canonical", "--verify-certificate", "witness.json"],
        vec!["--canonical", path.to_str().unwrap()],
    ] {
        assert_eq!(run(&args).status.code(), Some(2));
    }
    fs::remove_dir_all(dir).unwrap();
}
