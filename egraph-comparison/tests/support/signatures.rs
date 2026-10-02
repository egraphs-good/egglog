use super::*;

#[test]
fn colliding_hashes_and_table_growth_preserve_full_key_equality() {
    let mut signatures = Signatures::default();
    for n in 0..1000 {
        signatures.scratch = vec![n, 1, 7];
        assert_eq!(signatures.intern_hashed(0), BlockId::from_usize(n));
    }
    let allocated = signatures.words.len();
    for n in (0..1000).rev() {
        signatures.scratch = vec![n, 1, 7];
        assert_eq!(signatures.intern_hashed(0), BlockId::from_usize(n));
    }
    assert_eq!(signatures.words.len(), allocated);
}

#[test]
fn node_boundaries_and_previous_blocks_are_part_of_the_key() {
    let mut signatures = Signatures::default();
    let a = [SmallVec::from_slice(&[1, 2]), SmallVec::from_slice(&[3])];
    let b = [SmallVec::from_slice(&[1]), SmallVec::from_slice(&[2, 3])];
    let old = BlockId::from_usize(0);
    let first = signatures.intern(old, &a);
    assert_ne!(first, signatures.intern(old, &b));
    assert_ne!(first, signatures.intern(BlockId::from_usize(1), &a));
    assert_eq!(first, signatures.intern(old, &a));
}
