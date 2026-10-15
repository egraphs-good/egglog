use std::{ops::Deref, ops::Range};

use fixedbitset::FixedBitSet;

use super::Clear;

/// A bitset whose retained allocation remains measurable after its length shrinks.
#[derive(Debug, Default)]
pub(crate) struct ReusableBitSet {
    bits: FixedBitSet,
    capacity_bits: usize,
}

impl ReusableBitSet {
    fn reserve(&mut self, bits: usize, preserve: bool) {
        if bits <= self.capacity_bits {
            return;
        }
        // The supported FixedBitSet SIMD blocks divide 256 bits. Allocating
        // whole blocks explicitly avoids its hidden Vec growing geometrically.
        let capacity_bits = bits
            .checked_next_power_of_two()
            .expect("bitset capacity overflow")
            .max(256);
        let mut replacement = FixedBitSet::with_capacity(capacity_bits);
        if preserve {
            replacement.clone_from(&self.bits);
        }
        self.bits = replacement;
        self.capacity_bits = capacity_bits;
    }

    pub(crate) fn grow(&mut self, bits: usize) {
        self.reserve(bits, true);
        self.bits.grow(bits);
    }

    pub(crate) fn set(&mut self, index: usize, value: bool) {
        self.bits.set(index, value);
    }

    pub(crate) fn set_range(&mut self, range: Range<usize>, value: bool) {
        self.bits.set_range(range, value);
    }

    pub(crate) fn clear(&mut self) {
        self.bits.clear();
    }

    pub(crate) fn union_with(&mut self, other: &Self) {
        self.reserve(other.len(), true);
        self.bits.union_with(&other.bits);
    }

    pub(crate) fn symmetric_difference_with(&mut self, other: &Self) {
        self.reserve(other.len(), true);
        self.bits.symmetric_difference_with(&other.bits);
    }
}

impl Deref for ReusableBitSet {
    type Target = FixedBitSet;

    fn deref(&self) -> &Self::Target {
        &self.bits
    }
}

impl Clone for ReusableBitSet {
    fn clone(&self) -> Self {
        let mut result = Self::default();
        result.clone_from(self);
        result
    }

    fn clone_from(&mut self, source: &Self) {
        self.reserve(source.len(), false);
        self.bits.clone_from(&source.bits);
    }
}

impl Clear for ReusableBitSet {
    fn clear(&mut self) {
        self.bits.clone_from(&FixedBitSet::default());
    }

    fn reuse(&self) -> bool {
        self.capacity_bits != 0
    }

    fn bytes(&self) -> usize {
        self.capacity_bits / 8
    }
}
