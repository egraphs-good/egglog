use super::*;

#[test]
fn packed_root_slots_belong_to_one_run_and_one_root() {
    let rows = Subset::Dense(crate::OffsetRange::new(
        crate::RowId::new(0),
        crate::RowId::new(8),
    ));
    let first = TrieRoot::new_shared(rows.clone());
    let next_run = TrieRoot::new_shared(rows);
    let family = FamilyId::new(0);
    first.packed_root_slot(family, 2).unwrap().set(123).unwrap();
    assert!(
        first
            .packed_root_slot(FamilyId::new(1), 2)
            .unwrap()
            .get()
            .is_none()
    );
    assert!(
        next_run
            .packed_root_slot(family, 2)
            .unwrap()
            .get()
            .is_none()
    );
}
