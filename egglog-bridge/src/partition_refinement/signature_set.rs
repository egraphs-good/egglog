//! Interning utilities for exact signatures represented as sequences and sets.
//!
//! Each signature is stored as a slice into a single backing vector to avoid
//! allocating a `Vec` per signature. The returned `SignatureId` can be used for
//! fast equality checks.

use std::hash::{Hash, Hasher};

use hashbrown::HashTable;
use rustc_hash::FxHasher;

use crate::numeric_id::{NumericId, define_id};

define_id!(pub SignatureId, u32, "An interned exact signature set.");
define_id!(pub EnodeSig, u32, "An interned enode signature id.");
define_id!(pub EClassSig, u32, "An interned e-class signature id.");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SignatureRange {
    start: usize,
    len: usize,
    hash: u64,
}

/// An interner for ordered signature sequences that avoids per-sequence allocations.
#[derive(Debug, Clone)]
pub struct SignatureSet<T> {
    table: HashTable<SignatureId>,
    ranges: Vec<SignatureRange>,
    elements: Vec<T>,
}

impl<T> Default for SignatureSet<T> {
    fn default() -> Self {
        Self {
            table: HashTable::new(),
            ranges: Vec::new(),
            elements: Vec::new(),
        }
    }
}

/// An interner for e-class signatures built from enode signature ids.
#[derive(Debug, Default, Clone)]
pub struct EClassSignatureTable {
    signatures: SignatureSet<EnodeSig>,
}

impl EClassSignatureTable {
    /// Create an empty `EClassSignatureTable`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Remove all interned signatures while retaining allocated capacity.
    pub fn clear(&mut self) {
        self.signatures.clear();
    }

    /// Return the number of distinct e-class signatures stored.
    pub fn len(&self) -> usize {
        self.signatures.len()
    }

    /// Return `true` if no signatures are stored.
    pub fn is_empty(&self) -> bool {
        self.signatures.is_empty()
    }

    /// Intern an e-class signature from enode signature ids, sorting and de-duplicating them.
    pub fn intern<I>(&mut self, enodes: I) -> EClassSig
    where
        I: IntoIterator<Item = EnodeSig>,
    {
        let mut scratch = enodes.into_iter().collect::<Vec<_>>();
        scratch.sort_unstable();
        scratch.dedup();
        EClassSig::from_usize(self.signatures.intern(&scratch).index())
    }

    /// Return the canonical enode signature ids for the given e-class signature.
    pub fn signature(&self, sig: EClassSig) -> &[EnodeSig] {
        let id = SignatureId::from_usize(sig.index());
        self.signatures.signature(id)
    }
}

impl<T> SignatureSet<T> {
    /// Create an empty `SignatureSet`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Remove all interned signatures while retaining allocated capacity.
    pub fn clear(&mut self) {
        self.table.clear();
        self.ranges.clear();
        self.elements.clear();
    }

    /// Return the number of distinct signatures stored.
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Return `true` if no signatures are stored.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Return the canonical elements for the given signature id.
    pub fn signature(&self, id: SignatureId) -> &[T] {
        let range = self
            .ranges
            .get(id.index())
            .expect("signature id out of bounds");
        &self.elements[range.start..range.start + range.len]
    }
}

impl<T> SignatureSet<T>
where
    T: Copy + Ord + Hash,
{
    /// Intern an ordered slice of elements.
    pub fn intern(&mut self, elements: &[T]) -> SignatureId {
        let hash = hash_elements(elements);
        if let Some(id) = self.table.find(hash, |id| self.signature(*id) == elements) {
            return *id;
        }
        let start = self.elements.len();
        self.elements.extend_from_slice(elements);
        let id = SignatureId::from_usize(self.ranges.len());
        self.ranges.push(SignatureRange {
            start,
            len: elements.len(),
            hash,
        });
        let ranges = &self.ranges;
        *self
            .table
            .insert_unique(hash, id, |id| ranges[id.index()].hash)
            .get()
    }
}

fn hash_elements<T: Hash>(elements: &[T]) -> u64 {
    let mut hasher = FxHasher::default();
    elements.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::EClassSignatureTable;
    use super::EnodeSig;
    use super::SignatureId;
    use super::SignatureSet;
    use crate::numeric_id::NumericId;

    #[test]
    fn intern_preserves_order() {
        let mut set = SignatureSet::<u32>::new();
        let id = set.intern(&[1, 2, 3]);
        assert_eq!(id, SignatureId::from_usize(0));
        assert_eq!(set.signature(id), &[1, 2, 3]);

        let id2 = set.intern(&[1, 2, 3]);
        let id3 = set.intern(&[1, 3, 2]);
        assert_eq!(id, id2);
        assert_ne!(id, id3);
    }

    #[test]
    fn intern_distinguishes_sequences() {
        let mut set = SignatureSet::<u32>::new();
        let id1 = set.intern(&[1, 2, 3, 4]);
        let id2 = set.intern(&[1, 2, 3]);
        assert_ne!(id1, id2);
    }

    #[test]
    fn clear_resets_storage() {
        let mut set = SignatureSet::<u32>::new();
        let id1 = set.intern(&[1, 2]);
        set.clear();
        assert!(set.is_empty());
        let id2 = set.intern(&[1, 2]);
        assert_eq!(id1, SignatureId::from_usize(0));
        assert_eq!(id2, SignatureId::from_usize(0));
    }

    #[test]
    fn empty_set_is_stable() {
        let mut set = SignatureSet::<u32>::new();
        let empty: [u32; 0] = [];
        let id1 = set.intern(&empty);
        let id2 = set.intern(&[]);
        assert_eq!(id1, id2);
        assert!(set.signature(id1).is_empty());
    }

    #[test]
    fn eclass_signature_interns_enode_ids() {
        let mut table = EClassSignatureTable::new();
        let sig = table.intern([
            EnodeSig::from_usize(3),
            EnodeSig::from_usize(1),
            EnodeSig::from_usize(1),
        ]);
        assert_eq!(
            table.signature(sig),
            &[EnodeSig::from_usize(1), EnodeSig::from_usize(3)]
        );

        let sig2 = table.intern([EnodeSig::from_usize(1), EnodeSig::from_usize(3)]);
        assert_eq!(sig, sig2);
    }
}
