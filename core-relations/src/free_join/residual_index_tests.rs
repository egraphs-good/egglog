use super::*;
use std::mem;

#[test]
fn small_column_index_groups_keys_and_sorts_inline_rows() {
    assert_eq!(
        mem::size_of::<InlineRows>(),
        mem::size_of::<u32>() + SMALL_RESIDUAL * mem::size_of::<crate::RowId>()
    );

    let mut sink = SmallColumnSink::default();
    for (value, row) in [(2, 7), (1, 5), (2, 3), (1, 4), (3, 6)] {
        sink.rows[sink.len] = (Value::from_usize(value), crate::RowId::from_usize(row));
        sink.len += 1;
    }

    let index = SmallColumnIndex::from_projected(sink);
    assert_eq!(index.len(), 3);
    assert_eq!(index.find(Value::from_usize(0)), None);
    assert_eq!(
        index
            .rows_at(index.find(Value::from_usize(1)).unwrap())
            .rows()
            .iter()
            .map(|row| row.index())
            .collect::<Vec<_>>(),
        vec![4, 5]
    );
    assert_eq!(
        index
            .rows_at(index.find(Value::from_usize(2)).unwrap())
            .rows()
            .iter()
            .map(|row| row.index())
            .collect::<Vec<_>>(),
        vec![3, 7]
    );
    assert_eq!(
        index
            .rows_at(index.find(Value::from_usize(3)).unwrap())
            .rows()[0]
            .index(),
        6
    );
}

use crate::free_join::probe::{ProbeIndex, Prober};

#[test]
fn sorted_match_ordinal_reconstructs_row_carrying_small_probe() {
    let mut sink = SmallColumnSink::default();
    for (value, row) in [(1, 4), (2, 9), (2, 7)] {
        sink.rows[sink.len] = (Value::from_usize(value), crate::RowId::from_usize(row));
        sink.len += 1;
    }
    let index = SmallColumnIndex::from_projected(sink);
    let source = AtomRows::Inline(index.rows_at(0));
    let mut prober = Prober {
        source,
        ix: ProbeIndex::SmallColumn(index),
        keep_rows: true,
    };

    let ProbeMatch::Rows(AtomRows::Inline(rows)) = prober.sorted_match_at(1) else {
        panic!("a row-carrying small probe must reconstruct inline rows")
    };
    assert_eq!(
        rows.rows()
            .iter()
            .map(|row| row.index())
            .collect::<Vec<_>>(),
        vec![7, 9]
    );

    prober.keep_rows = false;
    assert!(matches!(prober.sorted_match_at(1), ProbeMatch::Present));
}
