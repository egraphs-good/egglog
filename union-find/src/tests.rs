use numeric_id::{NumericId, define_id};

use crate::UnionFind;

define_id!(pub(crate) Value, u32, "a value for testing the UF");

fn v(u: usize) -> Value {
    Value::from_usize(u)
}

#[test]
fn basic_uf() {
    let mut uf = UnionFind::<Value>::default();
    let ids1 = (0..100).map(v).collect::<Vec<_>>();
    let ids2 = (100..200).map(v).collect::<Vec<_>>();

    for ids in [&ids1, &ids2] {
        ids.windows(2).for_each(|w| {
            let (parent, child) = uf.union(w[0], w[1]);
            assert_eq!(uf.find_naive(w[0]), parent);
            assert_eq!(uf.find_naive(w[1]), parent);
            assert_eq!(uf.find(w[0]), parent);
            assert_eq!(uf.find(w[1]), parent);
            assert!(child == w[0] || child == w[1]);
        });
    }

    assert!(ids1.windows(2).all(|w| uf.find(w[0]) == uf.find(w[1])));
    assert!(ids2.windows(2).all(|w| uf.find(w[0]) == uf.find(w[1])));
    assert_ne!(uf.find(ids1[0]), uf.find(ids2[0]));

    uf.union(ids1[5], ids2[20]);

    let target = uf.find(ids1[0]);
    assert!(
        ids1.iter()
            .chain(ids2.iter())
            .all(|&id| uf.find(id) == target)
    );
}
