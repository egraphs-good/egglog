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
fn remap_preserves_complete_rows_for_empty_and_mixed_layouts() {
    for width in [1, 2, 9, 17] {
        for count in [0, 1, 7, 129] {
            for pattern in 0..4 {
                let mut rows = RowBuffer::new(width);
                let mut expected = Vec::new();
                let mut expected_remaps = Vec::new();
                for old in 0..count {
                    let row: Vec<_> = (0..width).map(|col| v(old * width + col)).collect();
                    let id = rows.add_row(&row);
                    let keep = match pattern {
                        0 => true,
                        1 => false,
                        2 => old % 3 == 1,
                        _ => old >= count / 3 && old < count * 2 / 3,
                    };
                    if keep {
                        expected_remaps.push((row.clone(), id, RowId::from_usize(expected.len())));
                        expected.push(row);
                    } else {
                        rows.set_stale(id);
                    }
                }
                let mut got_remaps = Vec::new();
                rows.remove_stale(|row, old, new| {
                    got_remaps.push((row.to_vec(), old, new));
                });
                assert_eq!(got_remaps, expected_remaps);
                assert_eq!(rows.len(), expected.len());
                assert_eq!(rows.iter().map(<[_]>::to_vec).collect::<Vec<_>>(), expected);
            }
        }
    }
}

#[test]
fn remap_panic_preserves_unprocessed_rows_and_row_count() {
    let mut rows = RowBuffer::new(2);
    for n in 0..5 {
        rows.add_row(&[v(n), v(n + 10)]);
    }
    rows.set_stale(RowId::new(1));
    rows.set_stale(RowId::new(3));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rows.remove_stale(|_, old, _| {
            assert_ne!(old, RowId::new(2), "intentional callback panic");
        });
    }));
    assert!(result.is_err());
    assert_eq!(rows.len(), 4);
    assert_eq!(rows.get_row(RowId::new(0)), &[v(0), v(10)]);
    assert_eq!(rows.get_row(RowId::new(1)), &[v(2), v(12)]);
    assert!(rows.get_row(RowId::new(2))[0].is_stale());
    assert_eq!(rows.get_row(RowId::new(3)), &[v(4), v(14)]);
    rows.remove_stale(|_, _, _| {});
    assert_eq!(rows.len(), 3);
    assert_eq!(rows.get_row(RowId::new(2)), &[v(4), v(14)]);
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

#[test]
fn parallel_row_buf_append_contents() {
    let mut rows = RowBuffer::new(2);
    rows.add_row(&[v(0), v(1)]);
    let writer = rows.parallel_writer();

    let mut extra = RowBuffer::new(2);
    extra.add_row(&[v(2), v(3)]);
    extra.add_row(&[v(4), v(5)]);
    let start = writer.append_contents(&extra);
    assert_eq!(start, RowId::new(1));

    let rows = writer.finish();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows.get_row(RowId::new(0)), [v(0), v(1)].as_slice());
    assert_eq!(rows.get_row(RowId::new(1)), [v(2), v(3)].as_slice());
    assert_eq!(rows.get_row(RowId::new(2)), [v(4), v(5)].as_slice());
}
