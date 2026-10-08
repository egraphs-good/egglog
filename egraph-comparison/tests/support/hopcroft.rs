use super::*;
use crate::{Class, Database, Function, FunctionKind, Row};
use std::collections::BTreeMap;

fn classes(n: usize) -> Database {
    let mut db = Database::default();
    for i in 0..n {
        db.classes.insert(
            i.to_string(),
            Class {
                sort: "E".into(),
                literal: None,
            },
        );
    }
    db
}

fn declare(db: &mut Database, name: &str, arity: usize, kind: FunctionKind) {
    db.functions.insert(
        name.into(),
        Function {
            kind,
            inputs: vec!["E".into(); arity],
            output: "E".into(),
        },
    );
}

fn row(db: &mut Database, name: &str, inputs: &[usize], output: usize) {
    db.rows.push(Row {
        function: name.into(),
        inputs: inputs.iter().map(usize::to_string).collect(),
        output: output.to_string(),
        subsumed: false,
    });
}

fn check(left: &Database, right: &Database, functions: bool) {
    left.validate().unwrap();
    right.validate().unwrap();
    let make = || {
        if functions {
            Partition::database(left, right)
        } else {
            Partition::new(left, right)
        }
    };
    let mut reference = make();
    reference.finish();
    let mut fast = make();
    let mut work = Worklist::new(&fast);
    work.finish(&mut fast);
    // Compare entire equivalence relations, not arbitrary block numbers or just
    // root coverage. Maps in both directions detect both over- and under-splits.
    let (mut forward, mut backward) = (BTreeMap::new(), BTreeMap::new());
    for (&a, &b) in reference.blocks.iter().zip(&fast.blocks) {
        assert_eq!(*forward.entry(a).or_insert(b), b);
        assert_eq!(*backward.entry(b).or_insert(a), a);
    }
    assert_eq!(reference.terms_equal(), fast.terms_equal());
    let bound = fast.blocks.len().max(1).ilog2() as usize;
    assert!(work.moves.iter().all(|&moves| moves <= bound));
    // Every member occurs exactly once, in the right block, with a valid inverse
    // position. At termination all members are clean and every block is nonempty.
    for (i, block) in work.blocks.iter().enumerate() {
        assert!(block.start < block.end);
        assert_eq!(block.clean, block.end);
        for pos in block.start..block.end {
            let id = work.members[pos];
            assert_eq!(work.position[id.index()], pos);
            assert_eq!(fast.blocks[id.index()].index(), i);
        }
    }
}

#[test]
fn matches_synchronous_refinement_on_generated_set_valued_graphs() {
    fn generated(mut seed: u64) -> Database {
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as usize
        };
        let n = 1 + next() % 32;
        let mut db = classes(n);
        for (name, arity, kind) in [
            ("f", 1, FunctionKind::Constructor),
            ("g", 2, FunctionKind::Constructor),
            ("h", 4, FunctionKind::Function),
        ] {
            declare(&mut db, name, arity, kind);
            let mut table = BTreeMap::new();
            for _ in 0..n * 3 {
                let inputs = (0..arity).map(|_| next() % n).collect::<Vec<_>>();
                table.insert(inputs, next() % n);
            }
            for (inputs, output) in table {
                row(&mut db, name, &inputs, output);
                db.rows.last_mut().unwrap().subsumed = next().is_multiple_of(3);
                if next().is_multiple_of(5) {
                    db.rows.push(db.rows.last().unwrap().clone());
                }
            }
        }
        db
    }
    for seed in 1..401 {
        let left = generated(seed);
        let right = generated(seed * 193);
        for functions in [false, true] {
            check(&left, &right, functions);
            let mut right = left.clone();
            right.rows.reverse();
            check(&left, &right, functions);
        }
    }
}

#[test]
fn preserves_sorts_literals_empty_classes_and_cycles() {
    let mut left = classes(8);
    declare(&mut left, "f", 1, FunctionKind::Constructor);
    declare(&mut left, "pair", 2, FunctionKind::Constructor);
    row(&mut left, "f", &[0], 0);
    row(&mut left, "f", &[1], 2);
    row(&mut left, "f", &[2], 1);
    row(&mut left, "pair", &[0, 1], 3);
    row(&mut left, "pair", &[1, 0], 3);
    row(&mut left, "pair", &[4, 4], 5);
    left.classes.get_mut("6").unwrap().sort = "Other".into();
    left.classes.get_mut("7").unwrap().literal = Some("literal".into());
    let mut right = left.clone();
    right.rows.pop();
    check(&left, &right, false);
    check(&Database::default(), &Database::default(), false);
    check(&left, &Database::default(), false);
}

#[test]
fn peels_a_long_chain_without_rescanning_the_clean_remainder() {
    let n = 16_384;
    let mut db = classes(n);
    declare(&mut db, "end", 0, FunctionKind::Constructor);
    declare(&mut db, "f", 1, FunctionKind::Constructor);
    row(&mut db, "end", &[], 0);
    for i in 1..n {
        row(&mut db, "f", &[i - 1], i);
    }
    let mut partition = Partition::new(&db, &db);
    let mut work = Worklist::new(&partition);
    work.finish(&mut partition);
    assert!(partition.terms_equal());
    assert_eq!(work.blocks.len(), n);
    assert!(work.evaluations < 8 * n, "{} evaluations", work.evaluations);
    assert!(work.moves.iter().all(|&moves| moves <= 1));
}

#[test]
fn duplicate_edges_and_high_fanout_do_not_duplicate_pending_blocks() {
    let mut db = classes(128);
    declare(&mut db, "base", 0, FunctionKind::Constructor);
    declare(&mut db, "f", 2, FunctionKind::Constructor);
    row(&mut db, "base", &[], 0);
    for i in 1..128 {
        row(&mut db, "f", &[0, i], i);
        for _ in 0..4 {
            db.rows.push(db.rows.last().unwrap().clone());
        }
    }
    check(&db, &db, false);
    let mut partition = Partition::new(&db, &db);
    let mut work = Worklist::new(&partition);
    work.finish(&mut partition);
    assert!(work.steps <= 3);
}
