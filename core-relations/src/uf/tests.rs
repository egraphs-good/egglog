use crate::numeric_id::NumericId;

use crate::{
    common::Value,
    table_spec::{ColumnId, Constraint, Table},
    uf::ProofReason,
    DisplacedTableWithProvenance, ProofStep,
};

use super::DisplacedTable;

fn v(x: usize) -> Value {
    Value::from_usize(x)
}

#[test]
fn displaced() {
    empty_execution_state!(e);
    let mut d = DisplacedTable::default();
    {
        let mut buf = d.new_buffer();
        buf.stage_insert(&[v(0), v(1), v(0)]);
        buf.stage_insert(&[v(2), v(3), v(0)]);
    }
    d.merge(&mut e);
    let all = d.all();
    let mut updates = Vec::new();
    d.scan_generic(all.as_ref(), |_, row| {
        assert_eq!(row[2], v(0));
        updates.push((row[0], row[1]))
    });
    assert_eq!(updates.len(), 2);
    assert_ne!(updates[0], updates[1]);
    let eq_fst = d.refine(
        all,
        &[Constraint::EqConst {
            col: ColumnId::new(0),
            val: updates[0].0,
        }],
    );
    let mut rows = Vec::new();
    d.scan_generic(eq_fst.as_ref(), |_, row| {
        assert_eq!(row.len(), 3);
        rows.push((row[0], row[1], row[2]))
    });
    assert_eq!(rows, vec![(updates[0].0, updates[0].1, v(0))]);

    d.new_buffer().stage_insert(&[v(1), v(3), v(1)]);
    d.merge(&mut e);

    let all = d.all();
    let mut updates_2 = Vec::new();
    d.scan_generic(all.as_ref(), |_, row| updates_2.push((row[0], row[1])));
    assert!(updates_2.windows(2).all(|x| x[0].1 == x[1].1));
}

#[test]
fn displaced_proof() {
    empty_execution_state!(e);
    let p1 = Value::new(1000);
    let p2 = p1.inc();
    let p3 = p2.inc();
    let mut d = DisplacedTableWithProvenance::default();
    d.new_buffer().stage_insert(&[v(0), v(1), v(0), p1]);
    d.new_buffer().stage_insert(&[v(2), v(3), v(0), p2]);
    d.merge(&mut e);
    let all = d.all();
    let mut updates = Vec::new();
    d.scan_generic(all.as_ref(), |_, row| {
        assert_eq!(row[2], v(0));
        updates.push((row[0], row[1]))
    });
    assert_eq!(updates.len(), 2);
    assert_ne!(updates[0], updates[1]);

    {
        let mut buf = d.new_buffer();
        buf.stage_insert(&[v(0), v(1), v(0), p1]);
        buf.stage_insert(&[v(2), v(0), v(0), p3]);
    }

    d.merge(&mut e);

    let proof = d.get_proof(v(0), v(3)).unwrap();
    assert_eq!(
        proof,
        vec![
            ProofStep {
                lhs: v(0),
                rhs: v(2),
                reason: ProofReason::Backward(p3),
            },
            ProofStep {
                lhs: v(2),
                rhs: v(3),
                reason: ProofReason::Forward(p2),
            },
        ]
    )
}
