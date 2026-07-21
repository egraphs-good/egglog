//! Arena-backed scalar trie levels for one join execution.
//!
//! A node is one allocation containing an immutable sorted column index and,
//! when another level follows it, one child publication slot per distinct key.
//! The allocation is tied to a [`Handle`] and deliberately has no destructor:
//! its only non-`Copy` trailing values are `OnceLock`s containing references
//! into the same execution-scoped arena.

use std::{alloc::Layout, marker::PhantomData, ptr, sync::OnceLock};

use egglog_concurrency::Handle;

use crate::{
    OffsetRange, SubsetRef, Value,
    numeric_id::NumericId,
    offsets::{RowId, SortedOffsetSlice},
    table_spec::ColumnId,
};

const NO_CHILDREN: u32 = u32::MAX;

/// One immutable, scalar trie level allocated for the lifetime of an execution.
///
/// The dynamically sized portions immediately following this header are laid
/// out as keys, boundaries, grouped row ids, and (optionally) child locks. Use
/// the accessors rather than relying on those offsets outside this module.
#[repr(C)]
pub(crate) struct PackedTrieNode<'exec> {
    key_len: u32,
    row_len: u32,
    column: ColumnId,
    keys_offset: u32,
    boundaries_offset: u32,
    rows_offset: u32,
    children_offset: u32,
    allocation_bytes: u32,
    _execution: PhantomData<&'exec ()>,
}

struct PackedLayout {
    allocation: Layout,
    keys_offset: usize,
    boundaries_offset: usize,
    rows_offset: usize,
    children_offset: Option<usize>,
}

impl PackedLayout {
    fn new(key_len: usize, row_len: usize, with_children: bool) -> Self {
        // Lifetimes do not affect layout; use one concrete instantiation so
        // callers do not need to manufacture a lifetime solely for arithmetic.
        let (layout, keys_offset) = Layout::new::<PackedTrieNode<'static>>()
            .extend(Layout::array::<Value>(key_len).expect("packed trie key layout overflow"))
            .expect("packed trie key offset overflow");
        let (layout, boundaries_offset) = layout
            .extend(
                Layout::array::<u32>(
                    key_len
                        .checked_add(1)
                        .expect("packed trie boundary count overflow"),
                )
                .expect("packed trie boundary layout overflow"),
            )
            .expect("packed trie boundary offset overflow");
        let (layout, rows_offset) = layout
            .extend(Layout::array::<RowId>(row_len).expect("packed trie row layout overflow"))
            .expect("packed trie row offset overflow");
        let (layout, children_offset) = if with_children {
            let (layout, offset) = layout
                .extend(
                    Layout::array::<OnceLock<&'static PackedTrieNode<'static>>>(key_len)
                        .expect("packed trie child layout overflow"),
                )
                .expect("packed trie child offset overflow");
            (layout, Some(offset))
        } else {
            (layout, None)
        };

        Self {
            allocation: layout.pad_to_align(),
            keys_offset,
            boundaries_offset,
            rows_offset,
            children_offset,
        }
    }

    fn checked_u32(value: usize, section: &str) -> u32 {
        u32::try_from(value)
            .unwrap_or_else(|_| panic!("packed trie {section} exceeds u32::MAX bytes"))
    }
}

impl<'exec> PackedTrieNode<'exec> {
    /// Build a node from pairs sorted lexicographically by `(Value, RowId)`.
    ///
    /// Sorting by value groups equal keys. Sorting each group by row id is what
    /// makes every range returned by [`Self::subset_at`] a valid `SubsetRef`.
    pub(crate) fn build_from_sorted_pairs(
        arena: &Handle<'exec>,
        column: ColumnId,
        pairs: &[(Value, RowId)],
        with_children: bool,
    ) -> &'exec Self {
        assert!(
            pairs.windows(2).all(|pair| pair[0] <= pair[1]),
            "packed trie input must be sorted by (Value, RowId)"
        );

        let row_len = u32::try_from(pairs.len())
            .expect("a packed trie node cannot contain more than u32::MAX rows");
        let key_len_usize = pairs
            .iter()
            .enumerate()
            .filter(|(index, pair)| *index == 0 || pairs[*index - 1].0 != pair.0)
            .count();
        let key_len = u32::try_from(key_len_usize)
            .expect("a packed trie node cannot contain more than u32::MAX keys");
        let layout = PackedLayout::new(key_len_usize, pairs.len(), with_children);
        let keys_offset = PackedLayout::checked_u32(layout.keys_offset, "key offset");
        let boundaries_offset =
            PackedLayout::checked_u32(layout.boundaries_offset, "boundary offset");
        let rows_offset = PackedLayout::checked_u32(layout.rows_offset, "row offset");
        let children_offset = layout
            .children_offset
            .map(|offset| PackedLayout::checked_u32(offset, "child offset"))
            .unwrap_or(NO_CHILDREN);
        let allocation_bytes =
            PackedLayout::checked_u32(layout.allocation.size(), "allocation size");
        let mut allocation = arena.alloc_layout(layout.allocation);
        let base = allocation.as_mut_ptr();

        // SAFETY: every pointer below is derived from `Layout::extend` offsets
        // within this allocation. Each element is written exactly once, no
        // shared reference exists yet, and all lengths were checked above.
        unsafe {
            let keys = base.add(layout.keys_offset).cast::<Value>();
            let boundaries = base.add(layout.boundaries_offset).cast::<u32>();
            let rows = base.add(layout.rows_offset).cast::<RowId>();
            let children = layout
                .children_offset
                .map(|offset| base.add(offset).cast::<OnceLock<&'exec Self>>());

            let mut next_key = 0usize;
            for (row_index, &(value, row_id)) in pairs.iter().enumerate() {
                if row_index == 0 || pairs[row_index - 1].0 != value {
                    ptr::write(keys.add(next_key), value);
                    ptr::write(boundaries.add(next_key), row_index as u32);
                    if let Some(children) = children {
                        ptr::write(children.add(next_key), OnceLock::new());
                    }
                    next_key += 1;
                }
                ptr::write(rows.add(row_index), row_id);
            }
            debug_assert_eq!(next_key, key_len_usize);
            ptr::write(boundaries.add(key_len_usize), row_len);
            ptr::write(
                base.cast::<Self>(),
                Self {
                    key_len,
                    row_len,
                    column,
                    keys_offset,
                    boundaries_offset,
                    rows_offset,
                    children_offset,
                    allocation_bytes,
                    _execution: PhantomData,
                },
            );

            // The complete header and every trailing element are initialized
            // before this conversion publishes an immutable arena reference.
            allocation.assume_init_no_drop::<Self>().into_ref()
        }
    }

    pub(crate) fn values(&self) -> &[Value] {
        // SAFETY: construction initializes exactly `key_len` values at this
        // layout-derived offset, and the arena outlives `self`.
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self)
                    .cast::<u8>()
                    .add(self.keys_offset as usize)
                    .cast(),
                self.key_len(),
            )
        }
    }

    pub(crate) fn boundaries(&self) -> &[u32] {
        // SAFETY: construction initializes one boundary per key plus a final
        // sentinel at this layout-derived offset.
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self)
                    .cast::<u8>()
                    .add(self.boundaries_offset as usize)
                    .cast(),
                self.key_len() + 1,
            )
        }
    }

    pub(crate) fn rows(&self) -> &[RowId] {
        // SAFETY: construction initializes exactly `row_len` row ids at this
        // layout-derived offset.
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self)
                    .cast::<u8>()
                    .add(self.rows_offset as usize)
                    .cast(),
                self.row_len(),
            )
        }
    }

    pub(crate) fn find(&self, value: Value) -> Option<usize> {
        self.values().binary_search(&value).ok()
    }

    pub(crate) fn column(&self) -> ColumnId {
        self.column
    }

    pub(crate) fn subset_at(&self, key_index: usize) -> SubsetRef<'_> {
        assert!(key_index < self.key_len(), "packed trie key out of bounds");
        let boundaries = self.boundaries();
        let rows = &self.rows()[boundaries[key_index] as usize..boundaries[key_index + 1] as usize];
        debug_assert!(!rows.is_empty());
        let first = rows[0];
        let last = rows[rows.len() - 1];
        if last.index() - first.index() == rows.len() - 1 {
            SubsetRef::Dense(OffsetRange::new(first, last.inc()))
        } else {
            // SAFETY: the constructor requires `(Value, RowId)` order, so the
            // rows within each equal-value range are non-decreasing.
            SubsetRef::Sparse(unsafe { SortedOffsetSlice::new_unchecked(rows) })
        }
    }

    pub(crate) fn child_slot(
        &self,
        key_index: usize,
    ) -> Option<&OnceLock<&'exec PackedTrieNode<'exec>>> {
        assert!(key_index < self.key_len(), "packed trie key out of bounds");
        if self.children_offset == NO_CHILDREN {
            return None;
        }
        // SAFETY: a non-leaf construction initializes exactly `key_len` child
        // locks at this layout-derived offset.
        Some(unsafe {
            &*(self as *const Self)
                .cast::<u8>()
                .add(self.children_offset as usize)
                .cast::<OnceLock<&'exec Self>>()
                .add(key_index)
        })
    }

    pub(crate) fn allocation_bytes(&self) -> usize {
        self.allocation_bytes as usize
    }

    fn key_len(&self) -> usize {
        self.key_len as usize
    }

    fn row_len(&self) -> usize {
        self.row_len as usize
    }
}

#[cfg(test)]
mod tests {
    use std::{
        mem::{align_of, size_of},
        sync::atomic::{AtomicUsize, Ordering},
    };

    use egglog_concurrency::SharedArena;

    use super::*;

    fn value(value: usize) -> Value {
        Value::from_usize(value)
    }

    fn row(row: usize) -> RowId {
        RowId::from_usize(row)
    }

    fn column(column: usize) -> ColumnId {
        ColumnId::from_usize(column)
    }

    #[test]
    fn packed_trie_builds_sorted_ranges() {
        let arena = SharedArena::new();
        let handle = arena.new_handle();
        let pairs = [
            (value(1), row(1)),
            (value(1), row(3)),
            (value(2), row(2)),
            (value(3), row(4)),
            (value(3), row(5)),
        ];
        let node = PackedTrieNode::build_from_sorted_pairs(&handle, column(2), &pairs, true);

        assert_eq!(node.column(), column(2));
        assert_eq!(node.values(), &[value(1), value(2), value(3)]);
        assert_eq!(node.boundaries(), &[0, 2, 3, 5]);
        assert_eq!(node.rows(), &[row(1), row(3), row(2), row(4), row(5)]);
        assert_eq!(node.find(value(2)), Some(1));
        assert_eq!(node.find(value(9)), None);

        let SubsetRef::Sparse(one) = node.subset_at(0) else {
            panic!("noncontiguous range should be sparse")
        };
        assert_eq!(one.inner(), &[row(1), row(3)]);
        assert!(matches!(
            node.subset_at(1),
            SubsetRef::Dense(range) if range == OffsetRange::new(row(2), row(3))
        ));
        assert!(matches!(
            node.subset_at(2),
            SubsetRef::Dense(range) if range == OffsetRange::new(row(4), row(6))
        ));
    }

    #[test]
    fn packed_trie_leaf_omits_child_locks() {
        let arena = SharedArena::new();
        let handle = arena.new_handle();
        let pairs = [(value(1), row(0)), (value(2), row(1))];
        let leaf = PackedTrieNode::build_from_sorted_pairs(&handle, column(0), &pairs, false);
        let branch = PackedTrieNode::build_from_sorted_pairs(&handle, column(0), &pairs, true);

        assert!(leaf.child_slot(0).is_none());
        assert!(leaf.child_slot(1).is_none());
        assert!(branch.child_slot(0).is_some());
        assert!(branch.child_slot(1).is_some());
        assert!(leaf.allocation_bytes() < branch.allocation_bytes());
    }

    #[test]
    fn packed_trie_sections_are_aligned() {
        let arena = SharedArena::new();
        let handle = arena.new_handle();
        let pairs = [(value(1), row(0)), (value(2), row(2))];
        let node = PackedTrieNode::build_from_sorted_pairs(&handle, column(1), &pairs, true);

        assert_eq!(
            node as *const _ as usize % align_of::<PackedTrieNode<'_>>(),
            0
        );
        assert_eq!(node.values().as_ptr() as usize % align_of::<Value>(), 0);
        assert_eq!(node.boundaries().as_ptr() as usize % align_of::<u32>(), 0);
        assert_eq!(node.rows().as_ptr() as usize % align_of::<RowId>(), 0);
        assert_eq!(
            node.child_slot(0).unwrap() as *const _ as usize
                % align_of::<OnceLock<&PackedTrieNode<'_>>>(),
            0
        );
        assert!(node.allocation_bytes() >= size_of::<PackedTrieNode<'_>>());
    }

    #[test]
    fn packed_trie_child_is_published_once_under_race() {
        let arena = SharedArena::new();
        let handle = arena.new_handle();
        let parent = PackedTrieNode::build_from_sorted_pairs(
            &handle,
            column(0),
            &[(value(1), row(0))],
            true,
        );
        let slot = parent.child_slot(0).unwrap();
        let initializations = AtomicUsize::new(0);

        std::thread::scope(|scope| {
            for _ in 0..8 {
                let arena = &arena;
                let initializations = &initializations;
                scope.spawn(move || {
                    let child = slot.get_or_init(|| {
                        initializations.fetch_add(1, Ordering::Relaxed);
                        let handle = arena.new_handle();
                        PackedTrieNode::build_from_sorted_pairs(
                            &handle,
                            column(1),
                            &[(value(7), row(9))],
                            false,
                        )
                    });
                    assert_eq!(child.values(), &[value(7)]);
                });
            }
        });

        assert_eq!(initializations.load(Ordering::Relaxed), 1);
        assert_eq!(slot.get().unwrap().column(), column(1));
        assert_eq!(slot.get().unwrap().values(), &[value(7)]);
    }
}
