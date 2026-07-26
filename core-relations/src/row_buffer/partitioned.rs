//! Hash-partitioned batches of rows.
//!
//! Producers append rows, unchanged full hashes, and their destination
//! partition densely. Before publication the buffer is sealed with a stable
//! counting partition, giving consumers one contiguous sequence per partition.

use std::iter::FusedIterator;

use crate::{common::Value, numeric_id::NumericId};

/// Selects a power-of-two hash partition within one of several outer groups.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HashPartitioning {
    shift: u8,
    bits: u8,
    groups: u32,
}

impl HashPartitioning {
    #[cfg(test)]
    pub(crate) fn new(shift: u32, bits: u32) -> Self {
        Self::grouped(shift, bits, 1)
    }

    pub(crate) fn grouped(shift: u32, bits: u32, groups: usize) -> Self {
        assert!(
            shift.checked_add(bits).is_some_and(|sum| sum <= 64),
            "hash partition bits must fit in a u64"
        );
        assert!(
            bits < usize::BITS,
            "hash partition count must fit in a usize"
        );
        assert_ne!(groups, 0, "hash partitioning must have at least one group");
        let groups = u32::try_from(groups).expect("hash partition group count must fit in u32");
        (1usize << bits)
            .checked_mul(groups as usize)
            .expect("total hash partition count overflow");
        Self {
            shift: shift as u8,
            bits: bits as u8,
            groups,
        }
    }

    pub(crate) fn partitions_per_group(self) -> usize {
        1usize << self.bits
    }

    pub(crate) fn partition_count(self) -> usize {
        self.partitions_per_group() * self.groups as usize
    }

    #[inline]
    pub(crate) fn partition(self, group: usize, hash: u64) -> usize {
        assert!(
            group < self.groups as usize,
            "hash partition group out of bounds"
        );
        let within_group = if self.bits == 0 {
            0
        } else {
            (hash.wrapping_shr(self.shift as u32) & (self.partitions_per_group() as u64 - 1))
                as usize
        };
        self.partition_index(group, within_group)
    }

    pub(crate) fn partition_index(self, group: usize, within_group: usize) -> usize {
        assert!(
            group < self.groups as usize,
            "hash partition group out of bounds"
        );
        assert!(
            within_group < self.partitions_per_group(),
            "hash subpartition out of bounds"
        );
        group * self.partitions_per_group() + within_group
    }
}

/// One row yielded from a [`PartitionedRowBuffer`].
pub(crate) struct HashedRow<'a> {
    pub(crate) hash: u64,
    pub(crate) row: &'a [Value],
}

/// A stable hash-partitioned row batch.
///
/// Before sealing, all vectors are dense and insertion ordered.
/// `partition_ids[i]` describes `hashes[i]` and row `i`. Sealing performs a
/// stable counting partition and replaces `partition_ids` with prefix offsets.
#[derive(Clone)]
pub(crate) struct PartitionedRowBuffer {
    arity: usize,
    partitioning: HashPartitioning,
    hashes: Vec<u64>,
    rows: Vec<Value>,
    partition_ids: Vec<u32>,
    offsets: Option<Box<[u32]>>,
}

impl PartitionedRowBuffer {
    pub(crate) fn new(arity: usize, partitioning: HashPartitioning) -> Self {
        Self {
            arity,
            partitioning,
            hashes: Vec::new(),
            rows: Vec::new(),
            partition_ids: Vec::new(),
            offsets: None,
        }
    }

    pub(crate) fn arity(&self) -> usize {
        self.arity
    }

    pub(crate) fn len(&self) -> usize {
        self.hashes.len()
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    pub(crate) fn partition_count(&self) -> usize {
        self.partitioning.partition_count()
    }

    /// Append a row, its outer group, and its unchanged full hash.
    pub(crate) fn add_row(&mut self, group: usize, hash: u64, row: &[Value]) {
        assert!(
            self.offsets.is_none(),
            "cannot append to a sealed partitioned row buffer"
        );
        assert_eq!(
            row.len(),
            self.arity,
            "attempting to add a row with mismatched arity"
        );
        let partition = self.partitioning.partition(group, hash);
        let partition =
            u32::try_from(partition).expect("partitioned row buffer partition id overflow");
        self.hashes.push(hash);
        self.rows.extend_from_slice(row);
        self.partition_ids.push(partition);
    }

    /// Convert insertion-ordered staging vectors into contiguous stable
    /// partition runs.
    pub(crate) fn seal(mut self) -> Self {
        if self.offsets.is_some() {
            return self;
        }
        debug_assert_eq!(self.partition_ids.len(), self.hashes.len());
        debug_assert_eq!(self.rows.len(), self.hashes.len() * self.arity);

        if self.partition_count() == 1 {
            let len = u32::try_from(self.len()).expect("partitioned row buffer row count overflow");
            self.partition_ids.clear();
            self.offsets = Some(Box::new([0, len]));
            return self;
        }

        let mut offsets = vec![0u32; self.partition_count() + 1];
        for &partition in &self.partition_ids {
            offsets[partition as usize + 1] = offsets[partition as usize + 1]
                .checked_add(1)
                .expect("partitioned row count overflow");
        }
        for partition in 0..self.partition_count() {
            offsets[partition + 1] = offsets[partition + 1]
                .checked_add(offsets[partition])
                .expect("partitioned row offset overflow");
        }
        debug_assert_eq!(offsets[self.partition_count()] as usize, self.len());

        let mut cursors = offsets[..self.partition_count()].to_vec();
        let mut hashes = vec![0u64; self.len()];
        let row_values = self
            .len()
            .checked_mul(self.arity)
            .expect("partitioned row value count overflow");
        let mut rows = vec![Value::new(0); row_values];
        for source in 0..self.len() {
            let partition = self.partition_ids[source] as usize;
            let destination = cursors[partition] as usize;
            cursors[partition] += 1;
            hashes[destination] = self.hashes[source];
            if self.arity != 0 {
                rows[destination * self.arity..(destination + 1) * self.arity]
                    .copy_from_slice(&self.rows[source * self.arity..(source + 1) * self.arity]);
            }
        }

        self.hashes = hashes;
        self.rows = rows;
        self.partition_ids = Vec::new();
        self.offsets = Some(offsets.into_boxed_slice());
        self
    }

    #[cfg(test)]
    pub(crate) fn partition_len(&self, partition: usize) -> usize {
        let offsets = self.offsets();
        (offsets[partition + 1] - offsets[partition]) as usize
    }

    pub(crate) fn group_is_empty(&self, group: usize) -> bool {
        self.group_len(group) == 0
    }

    pub(crate) fn group_len(&self, group: usize) -> usize {
        let first = self.partitioning.partition_index(group, 0);
        let after_last = first + self.partitioning.partitions_per_group();
        let offsets = self.offsets();
        (offsets[after_last] - offsets[first]) as usize
    }

    pub(crate) fn partition(&self, partition: usize) -> PartitionIter<'_> {
        let offsets = self.offsets();
        PartitionIter {
            buffer: self,
            next: offsets[partition] as usize,
            end: offsets[partition + 1] as usize,
        }
    }

    fn offsets(&self) -> &[u32] {
        self.offsets
            .as_deref()
            .expect("partitioned row buffer must be sealed before reading")
    }
}

pub(crate) struct PartitionIter<'a> {
    buffer: &'a PartitionedRowBuffer,
    next: usize,
    end: usize,
}

impl<'a> Iterator for PartitionIter<'a> {
    type Item = HashedRow<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.end {
            return None;
        }
        let index = self.next;
        self.next += 1;
        let row = if self.buffer.arity == 0 {
            &[]
        } else {
            &self.buffer.rows[index * self.buffer.arity..(index + 1) * self.buffer.arity]
        };
        Some(HashedRow {
            hash: self.buffer.hashes[index],
            row,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.next;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for PartitionIter<'_> {}
impl FusedIterator for PartitionIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(value: usize) -> Value {
        Value::from_usize(value)
    }

    #[test]
    fn partitions_rows_stably_and_keeps_hashes_aligned() {
        let partitioning = HashPartitioning::new(4, 2);
        let mut buffer = PartitionedRowBuffer::new(2, partitioning);
        let mut expected = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for i in 0..80usize {
            let partition = (i * 3) & 3;
            let hash = ((partition as u64) << 4) | (i as u64 & 0xf);
            let row = [v(i), v(i + 1)];
            buffer.add_row(0, hash, &row);
            expected[partition].push((hash, row));
        }
        let buffer = buffer.seal();

        assert_eq!(buffer.len(), 80);
        assert_eq!(buffer.arity(), 2);
        for (partition, expected) in expected.iter().enumerate() {
            assert_eq!(buffer.partition_len(partition), expected.len());
            let mut iter = buffer.partition(partition);
            assert_eq!(iter.len(), expected.len());
            let actual = iter
                .by_ref()
                .map(|hashed| (hashed.hash, hashed.row.to_vec()))
                .collect::<Vec<_>>();
            assert_eq!(iter.len(), 0);
            let expected = expected
                .iter()
                .map(|(hash, row)| (*hash, row.to_vec()))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn supports_groups_zero_arity_and_logical_clone() {
        let partitioning = HashPartitioning::grouped(1, 3, 2);
        let mut buffer = PartitionedRowBuffer::new(0, partitioning);
        for (group, hash) in [(0, 0), (1, 2), (0, 4), (1, 2), (1, 14), (0, 0)] {
            buffer.add_row(group, hash, &[]);
        }
        let buffer = buffer.seal();
        let cloned = buffer.clone();

        assert_eq!(cloned.len(), 6);
        assert!(!cloned.is_empty());
        assert_eq!(cloned.group_len(0), 3);
        assert_eq!(cloned.group_len(1), 3);
        for partition in 0..partitioning.partition_count() {
            let original = buffer
                .partition(partition)
                .map(|hashed| (hashed.hash, hashed.row.len()))
                .collect::<Vec<_>>();
            let copied = cloned
                .partition(partition)
                .map(|hashed| (hashed.hash, hashed.row.len()))
                .collect::<Vec<_>>();
            assert_eq!(copied, original);
        }
    }

    #[test]
    fn is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PartitionedRowBuffer>();
    }

    #[test]
    fn one_partition_seals_without_copying_rows_or_hashes() {
        let partitioning = HashPartitioning::new(0, 0);
        let mut buffer = PartitionedRowBuffer::new(2, partitioning);
        for i in 0..32usize {
            buffer.add_row(0, i as u64, &[v(i), v(i + 1)]);
        }
        let hashes = buffer.hashes.as_ptr();
        let rows = buffer.rows.as_ptr();

        let buffer = buffer.seal();

        assert_eq!(buffer.hashes.as_ptr(), hashes);
        assert_eq!(buffer.rows.as_ptr(), rows);
        assert!(buffer.partition_ids.is_empty());
        assert_eq!(buffer.partition_len(0), 32);
        assert_eq!(
            buffer
                .partition(0)
                .map(|hashed| (hashed.hash, hashed.row.to_vec()))
                .collect::<Vec<_>>(),
            (0..32usize)
                .map(|i| (i as u64, vec![v(i), v(i + 1)]))
                .collect::<Vec<_>>()
        );
    }
}
