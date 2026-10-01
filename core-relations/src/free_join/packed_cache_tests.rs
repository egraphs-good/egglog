use super::*;

#[test]
fn root_projection_is_a_final_dense_or_sparse_index() {
    let index = RootProjection::from_sorted_pairs(vec![
        (Value::from_usize(1), crate::RowId::from_usize(4)),
        (Value::from_usize(1), crate::RowId::from_usize(5)),
        (Value::from_usize(2), crate::RowId::from_usize(3)),
        (Value::from_usize(2), crate::RowId::from_usize(7)),
        (Value::from_usize(9), crate::RowId::from_usize(11)),
    ]);

    assert_eq!(index.len(), 3);
    assert_eq!(index.rows.len(), 5);
    assert_eq!(
        index
            .keys
            .iter()
            .map(|&(key, offset)| (key.index(), offset))
            .collect::<Vec<_>>(),
        vec![(1, 0), (2, 2), (9, 4), (0, 5)]
    );
    assert_eq!(index.find(Value::from_usize(0)), None);
    assert_eq!(index.find(Value::from_usize(8)), None);

    let dense = index.subset_at(index.find(Value::from_usize(1)).unwrap());
    let crate::SubsetRef::Dense(dense) = dense else {
        panic!("contiguous root-index rows must use a dense subset")
    };
    assert_eq!((dense.start.index(), dense.end.index()), (4, 6));

    let sparse = index.subset_at(index.find(Value::from_usize(2)).unwrap());
    let crate::SubsetRef::Sparse(sparse) = sparse else {
        panic!("noncontiguous root-index rows must use a sparse subset")
    };
    assert_eq!(
        sparse
            .inner()
            .iter()
            .map(|row| row.index())
            .collect::<Vec<_>>(),
        vec![3, 7]
    );

    let singleton = index.subset_at(index.find(Value::from_usize(9)).unwrap());
    let crate::SubsetRef::Dense(singleton) = singleton else {
        panic!("a singleton root-index group must use a dense subset")
    };
    assert_eq!((singleton.start.index(), singleton.end.index()), (11, 12));

    let empty = RootProjection::from_sorted_pairs(Vec::new());
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.rows.len(), 0);
    assert_eq!(
        empty.keys.len(),
        1,
        "empty indexes retain only the sentinel"
    );
    assert_eq!(empty.find(Value::from_usize(0)), None);
}
