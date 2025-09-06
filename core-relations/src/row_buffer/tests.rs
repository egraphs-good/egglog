use crate::numeric_id::NumericId;

use crate::{common::Value, offsets::RowId};

use super::{RowBuffer, TaggedRowBuffer};

fn v(n: usize) -> Value {
    Value::from_usize(n)
}

#[test]
fn remap_basic() {
    let mut rows = RowBuffer::new(2);
    let r1 = rows.add_row(&[v(0), v(1)]);
    let r2 = rows.add_row(&[v(2), v(3)]);
    let r3 = rows.add_row(&[v(4), v(5)]);
    let r4 = rows.add_row(&[v(6), v(7)]);

    assert!(!rows.set_stale(r2));
    assert!(!rows.set_stale(r3));

    let mut got = Vec::new();
    rows.remove_stale(|row, old, new| got.push((row.to_vec(), old, new)));
    assert_eq!(
        got,
        vec![(vec![v(0), v(1)], r1, r1), (vec![v(6), v(7)], r4, r2)]
    );

    assert_eq!(rows.get_row(r2), [v(6), v(7)].as_slice());
}

#[test]
fn remap_narrow() {
    let mut rows = RowBuffer::new(1);
    let r1 = rows.add_row(&[v(0)]);
    let r2 = rows.add_row(&[v(1)]);
    let r3 = rows.add_row(&[v(2)]);
    let r4 = rows.add_row(&[v(3)]);

    rows.set_stale(r2);
    rows.set_stale(r3);
    let mut got = Vec::new();
    rows.remove_stale(|row, old, new| got.push((row.to_vec(), old, new)));
    assert_eq!(got, vec![(vec![v(0)], r1, r1), (vec![v(3)], r4, r2)]);
    assert_eq!(rows.get_row(r2), [v(3)]);
}

#[test]
fn basic_tagged_row() {
    let mut rows = TaggedRowBuffer::new(2);
    let r1 = rows.add_row(RowId::new(4), &[v(0), v(1)]);
    let r2 = rows.add_row(RowId::new(7), &[v(2), v(3)]);
    assert_eq!(rows.get_row(r1), (RowId::new(4), [v(0), v(1)].as_slice()));
    assert_eq!(rows.get_row(r2), (RowId::new(7), [v(2), v(3)].as_slice()));
    assert_eq!(
        rows.iter().collect::<Vec<_>>(),
        vec![
            (RowId::new(4), [v(0), v(1)].as_slice()),
            (RowId::new(7), [v(2), v(3)].as_slice())
        ]
    );
}
