use crate::numeric_id::NumericId;

use crate::{
    common::Value,
    table_spec::{ColumnId, Constraint, Table},
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
fn parallel_merge_preserves_timestamp_order_and_lookup() {
    empty_execution_state!(e);
    let mut d = DisplacedTable::default();
    let n = 40_000;
    {
        let mut buf = d.new_buffer();
        for i in (0..n).step_by(2) {
            buf.stage_insert(&[v(i), v(i + 1), v(0)]);
        }
        for i in (2..n).step_by(2) {
            buf.stage_insert(&[v(0), v(i), v(1)]);
        }
    }

    let pool = egglog_concurrency::ThreadPool::new(4);
    let change = pool.install(|| d.merge(&mut e));
    assert!(change.added);
    assert_eq!(d.len(), n - 1);

    let mut timestamps = Vec::new();
    d.scan_generic(d.all().as_ref(), |_, row| timestamps.push(row[2]));
    assert_eq!(
        timestamps
            .iter()
            .filter(|&&timestamp| timestamp == v(0))
            .count(),
        n / 2
    );
    assert_eq!(
        timestamps
            .iter()
            .filter(|&&timestamp| timestamp == v(1))
            .count(),
        n / 2 - 1
    );
    assert!(timestamps.windows(2).all(|pair| pair[0] <= pair[1]));

    for i in 1..n {
        let row = d.get_row(&[v(i)]).expect("every displaced id is indexed");
        assert_eq!(row.vals[0], v(i));
        assert_eq!(row.vals[1], v(0));
    }
}

#[test]
fn parallel_merge_records_each_displaced_root_once_under_contention() {
    empty_execution_state!(e);
    let mut d = DisplacedTable::default();
    let n = 20_000;
    {
        let mut buf = d.new_buffer();
        for repeat in 0..4 {
            for i in 1..n {
                let (left, right) = if (i + repeat) % 2 == 0 {
                    (i - 1, i)
                } else {
                    (i, i - 1)
                };
                buf.stage_insert(&[v(left), v(right), v(0)]);
            }
        }
    }

    let pool = egglog_concurrency::ThreadPool::new(4);
    pool.install(|| d.merge(&mut e));
    assert_eq!(d.len(), n - 1);
    for i in 1..n {
        assert_eq!(d.get_row_column(&[v(i)], ColumnId::new(1)), Some(v(0)));
    }

    let mut clone = d.clone();
    clone.new_buffer().stage_insert(&[v(0), v(n), v(1)]);
    clone.merge(&mut e);
    assert_eq!(clone.underlying_uf().find_naive(v(n)), v(0));
    assert_eq!(d.underlying_uf().find_naive(v(n)), v(n));
}

#[test]
fn parallel_merge_matches_serial_timestamp_oracle() {
    empty_execution_state!(e);
    let mut d = DisplacedTable::default();
    let mut oracle = crate::union_find::UnionFind::default();
    let mut expected = crate::common::HashMap::default();
    let mut random = 0x1234_5678_u64;
    {
        let mut buf = d.new_buffer();
        for timestamp in 0..4 {
            for _ in 0..20_000 {
                random = random
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let left = v((random as usize) % 50_000);
                random = random
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let right = v((random as usize) % 50_000);
                buf.stage_insert(&[left, right, v(timestamp)]);

                let (parent, child) = oracle.union(left, right);
                if parent != child {
                    assert_eq!(expected.insert(child, v(timestamp)), None);
                }
            }
        }
    }

    let pool = egglog_concurrency::ThreadPool::new(8);
    pool.install(|| d.merge(&mut e));
    assert_eq!(d.len(), expected.len());
    d.scan_generic(d.all().as_ref(), |row_id, row| {
        assert_eq!(expected.remove(&row[0]), Some(row[2]));
        assert_eq!(row[1], oracle.find_naive(row[0]));
        assert_eq!(d.get_row(&[row[0]]).map(|found| found.id), Some(row_id));
    });
    assert!(expected.is_empty());
}
