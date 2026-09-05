//! Arena-backed scalar trie levels for one join execution.
//!
//! A node is one allocation containing an immutable sorted column index and,
//! when another level follows it, one child publication slot per distinct key.
//! The raw allocation is tied to a [`Handle`] and never runs destructors.

use std::{
    alloc::Layout,
    marker::PhantomData,
    mem::{align_of, size_of},
    ptr,
    sync::{
        OnceLock,
        atomic::{AtomicPtr, Ordering},
    },
};

use egglog_concurrency::Handle;

use crate::{
    SubsetRef, Value,
    offsets::{RowId, SortedOffsetSlice},
    table_spec::{ColumnId, WrappedTableRef},
};

/// A packed distinct-key count and description of the trailing child storage.
///
/// The low 30 bits store the key count. Bit 30 selects dynamic child families,
/// while bit 31 records that the node has children at all.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PackedKeyLen(u32);

impl PackedKeyLen {
    const HAS_CHILDREN: u32 = 1 << 31;
    const DYNAMIC_CHILDREN: u32 = 1 << 30;
    const LEN_MASK: u32 = Self::DYNAMIC_CHILDREN - 1;

    fn new(key_len: usize, child_shape: ChildShape) -> Self {
        assert!(
            key_len <= Self::LEN_MASK as usize,
            "a packed trie node cannot contain 2^30 or more keys"
        );
        let flags = match child_shape {
            ChildShape::Leaf => 0,
            ChildShape::Direct => Self::HAS_CHILDREN,
            ChildShape::Dynamic { families } => {
                assert!(
                    families > 0,
                    "a dynamic packed trie node needs at least one child family"
                );
                assert!(
                    u32::try_from(families).is_ok(),
                    "a packed trie node cannot contain more than u32::MAX child families"
                );
                Self::HAS_CHILDREN | Self::DYNAMIC_CHILDREN
            }
        };
        Self(key_len as u32 | flags)
    }

    fn len(self) -> usize {
        (self.0 & Self::LEN_MASK) as usize
    }

    fn has_children(self) -> bool {
        self.0 & Self::HAS_CHILDREN != 0
    }

    fn has_dynamic_children(self) -> bool {
        self.0 & Self::DYNAMIC_CHILDREN != 0
    }
}

type ChildSlot<'exec> = OnceLock<&'exec PackedTrieNode<'exec>>;

/// The child-publication storage trailing a packed trie node.
///
/// Returning slices from one accessor centralizes the unsafe offset arithmetic
/// for both inline direct slots and lazily allocated dynamic families.
enum PackedChildren<'node, 'exec> {
    Leaf,
    Direct(&'node [ChildSlot<'exec>]),
    Families(&'node [AtomicPtr<ChildSlot<'exec>>]),
}

/// How one packed trie level reaches its children.
///
/// Tuple-index interior levels and fixed-order join levels use [`Self::Direct`]
/// and retain one inline child slot per key. A dynamically ordered join level
/// uses one lazily allocated child-slot family per eligible successor. The
/// family arrays themselves are raw arena allocations and therefore add no
/// registered destructors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ChildShape {
    Leaf,
    Direct,
    Dynamic { families: usize },
}

/// One immutable, scalar trie level allocated for the lifetime of an execution.
///
/// Each node starts with the same packed index. `boundaries[i]..boundaries[i + 1]`
/// selects the rows belonging to `keys[i]`:
///
/// ```text
/// +----------------------------+
/// | PackedTrieNode header      |
/// +----------------------------+
/// | Value keys[K]              |
/// +----------------------------+
/// | u32 boundaries[K + 1]      |
/// +----------------------------+
/// | RowId rows[R]              |
/// +----------------------------+
/// ```
///
/// A leaf ends there. A direct node appends one inline child-publication slot
/// per key:
///
/// ```text
/// | ChildSlot direct[K]        |
/// +----------------------------+
/// ```
///
/// A dynamically ordered node instead appends a family count and one atomic
/// pointer per eligible successor. Each pointer lazily publishes a separate
/// arena allocation containing one `ChildSlot[K]` array:
///
/// ```text
/// | u32 family_count           |       family pointer
/// +----------------------------+             |
/// | AtomicPtr families[F]      |-------------+--> ChildSlot family[K]
/// +----------------------------+
/// ```
///
/// Alignment may insert padding between regions. Use the accessors rather
/// than relying on these offsets outside this module.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct PackedTrieNode<'exec> {
    key_len: PackedKeyLen,
    row_len: u32,
    // The trailing child slots contain `OnceLock<&'exec PackedTrieNode<'exec>>`,
    // which is invariant in `'exec` because `OnceLock` permits publication.
    // Model that invariance in the sized header even though the slots live in
    // trailing storage. A function argument plus return keeps the marker
    // invariant without making the immutable node `!Sync`.
    _execution: PhantomData<fn(&'exec ()) -> &'exec ()>,
}

/// A checked position within one packed scalar trie level.
///
/// Keeping the ordinal beside the arena reference makes descendant binding
/// state `Copy`: the represented subset is the corresponding range in the
/// node's row array, rather than another owned trie node.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedCursor<'node, 'exec> {
    node: &'node PackedTrieNode<'exec>,
    key_index: u32,
}

impl<'node, 'exec> PackedCursor<'node, 'exec>
where
    'exec: 'node,
{
    pub(crate) fn new(node: &'node PackedTrieNode<'exec>, key_index: usize) -> Self {
        assert!(
            key_index < node.key_len(),
            "packed trie cursor key out of bounds"
        );
        Self {
            node,
            key_index: u32::try_from(key_index)
                .expect("packed trie cursor key ordinal exceeds u32::MAX"),
        }
    }

    pub(crate) fn subset(self) -> SubsetRef<'node> {
        self.node.subset_at(self.key_index())
    }

    pub(crate) fn size(self) -> usize {
        let boundaries = self.node.boundaries();
        boundaries[self.key_index() + 1] as usize - boundaries[self.key_index()] as usize
    }

    /// Return the next scalar index below this cursor, building and publishing
    /// it once for the current key when necessary.
    ///
    /// A child slot belongs to a frozen execution plan: every use must request
    /// the same next column and leaf shape. The compact node header records the
    /// shape but deliberately does not repeat the plan's column, so requesting
    /// the same column is a caller invariant.
    pub(crate) fn child_index(
        self,
        arena: &Handle<'exec>,
        table: WrappedTableRef<'_>,
        column: ColumnId,
        family: usize,
        child_shape: ChildShape,
        scratch: &mut Vec<(Value, RowId)>,
    ) -> &'exec PackedTrieNode<'exec> {
        self.child_index_with(arena, family, child_shape, || {
            PackedTrieNode::build_from_subset(
                arena,
                table,
                self.subset(),
                column,
                child_shape,
                scratch,
            )
        })
    }

    /// Like [`Self::child_index`], but publish an index over a subset that the
    /// caller has already refined by the next scan's row constraints.
    ///
    /// The prefiltered subset is copied into the arena allocation during this
    /// call, so it may borrow short-lived executor scratch. As with the column
    /// and leaf shape, a frozen plan must supply equivalent filtering on every
    /// use of this cursor's child slot.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn child_index_from_subset(
        self,
        arena: &Handle<'exec>,
        table: WrappedTableRef<'_>,
        subset: SubsetRef<'_>,
        column: ColumnId,
        family: usize,
        child_shape: ChildShape,
        scratch: &mut Vec<(Value, RowId)>,
    ) -> &'exec PackedTrieNode<'exec> {
        self.child_index_with(arena, family, child_shape, || {
            PackedTrieNode::build_from_subset(arena, table, subset, column, child_shape, scratch)
        })
    }

    /// Return this cursor's child, invoking `build` only when its slot is
    /// empty. Callers that must copy or refine the cursor subset should put
    /// that work in `build`, so a cache hit remains allocation-free.
    ///
    /// The cursor's node must have children. Valid executor plans establish
    /// that invariant from the requested [`ChildShape`]; calling this on a leaf
    /// is a plan-shape bug.
    pub(crate) fn child_index_with(
        self,
        arena: &Handle<'exec>,
        family: usize,
        child_shape: ChildShape,
        build: impl FnOnce() -> &'exec PackedTrieNode<'exec>,
    ) -> &'exec PackedTrieNode<'exec> {
        let slot = self.node.child_slot(arena, self.key_index(), family);
        let child = *slot.get_or_init(build);
        assert_eq!(
            child.child_shape(),
            child_shape,
            "packed trie child cache shape mismatch"
        );
        child
    }

    fn key_index(self) -> usize {
        self.key_index as usize
    }
}

struct PackedLayout {
    allocation: Layout,
    keys_offset: usize,
    boundaries_offset: usize,
    rows_offset: usize,
    children_offset: Option<usize>,
    dynamic_family_count_offset: Option<usize>,
}

impl PackedLayout {
    fn new(key_len: usize, row_len: usize, child_shape: ChildShape) -> Self {
        let _ = PackedKeyLen::new(key_len, child_shape);
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
        let (layout, children_offset, dynamic_family_count_offset) = match child_shape {
            ChildShape::Leaf => (layout, None, None),
            ChildShape::Direct => {
                let (layout, offset) = layout
                    .extend(
                        Layout::array::<ChildSlot<'static>>(key_len)
                            .expect("packed trie direct-child layout overflow"),
                    )
                    .expect("packed trie direct-child offset overflow");
                (layout, Some(offset), None)
            }
            ChildShape::Dynamic { families } => {
                let (layout, family_count_offset) = layout
                    .extend(Layout::new::<u32>())
                    .expect("packed trie dynamic-family count offset overflow");
                let (layout, children_offset) = layout
                    .extend(
                        Layout::array::<AtomicPtr<ChildSlot<'static>>>(families)
                            .expect("packed trie dynamic-family layout overflow"),
                    )
                    .expect("packed trie dynamic-family offset overflow");
                (layout, Some(children_offset), Some(family_count_offset))
            }
        };

        Self {
            allocation: layout.pad_to_align(),
            keys_offset,
            boundaries_offset,
            rows_offset,
            children_offset,
            dynamic_family_count_offset,
        }
    }
}

impl<'exec> PackedTrieNode<'exec> {
    /// Project one column of an already-filtered subset and build its packed
    /// scalar trie level.
    ///
    /// `subset` must already reflect every row-local constraint that applies
    /// at this point in the join. The caller owns `scratch` so recursive join
    /// execution can reuse the projection and radix-sort allocation. On
    /// return, `scratch` contains the sorted `(Value, RowId)` pairs used to
    /// build the node; its capacity also retains the radix sort's ping-pong
    /// storage for the next build.
    pub(crate) fn build_from_subset(
        arena: &Handle<'exec>,
        table: WrappedTableRef<'_>,
        subset: SubsetRef<'_>,
        column: ColumnId,
        child_shape: ChildShape,
        scratch: &mut Vec<(Value, RowId)>,
    ) -> &'exec Self {
        scratch.clear();
        scratch.reserve(subset.size());
        table.for_each_col(subset, column, &mut |row_id, value| {
            scratch.push((value, row_id));
        });
        // A SubsetRef is RowId-ordered, and for_each_col preserves that scan
        // order. The repository's value-stable radix sort therefore produces
        // full (Value, RowId) order without a separate RowId pass. A scalar
        // projection has exactly one pair per input row, so retaining every
        // pair (rather than deduplicating) preserves the subset exactly. Keep
        // the sort's ping-pong half in the same caller-owned Vec so repeated
        // descendant construction does not allocate another temporary buffer.
        debug_assert!(
            scratch.windows(2).all(|pair| pair[0].1 <= pair[1].1),
            "packed trie subset projection must be RowId-ordered"
        );
        let pair_len = scratch.len();
        if pair_len < 64 {
            // This mirrors radix_sort_slice_by_value's small-input path, but
            // avoids growing and initializing a second half that it would not
            // use. Most lower-level trie subsets fall into this case.
            scratch.sort_unstable();
        } else {
            let storage_len = pair_len
                .checked_mul(2)
                .expect("packed trie sort scratch size overflow");
            scratch.resize(storage_len, (Value::new_const(0), RowId::new_const(0)));
            {
                let (pairs, sort_scratch) = scratch.split_at_mut(pair_len);
                crate::hash_index::radix_sort_slice_by_value(pairs, sort_scratch);
            }
            scratch.truncate(pair_len);
        }

        Self::build_from_sorted_pairs(arena, scratch, child_shape)
    }

    /// Build a node from pairs sorted lexicographically by `(Value, RowId)`.
    ///
    /// Sorting by value groups equal keys. Sorting each group by row id is what
    /// makes every range returned by [`Self::subset_at`] a valid `SubsetRef`.
    pub(crate) fn build_from_sorted_pairs(
        arena: &Handle<'exec>,
        pairs: &[(Value, RowId)],
        child_shape: ChildShape,
    ) -> &'exec Self {
        debug_assert!(
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
        let key_len = PackedKeyLen::new(key_len_usize, child_shape);
        let layout = PackedLayout::new(key_len_usize, pairs.len(), child_shape);
        debug_assert_eq!(layout.keys_offset, Self::keys_offset());
        debug_assert_eq!(
            layout.boundaries_offset,
            Self::boundaries_offset_for(key_len_usize)
        );
        debug_assert_eq!(layout.rows_offset, Self::rows_offset_for(key_len_usize));
        match child_shape {
            ChildShape::Leaf => {
                debug_assert_eq!(layout.children_offset, None);
                debug_assert_eq!(layout.dynamic_family_count_offset, None);
            }
            ChildShape::Direct => {
                debug_assert_eq!(
                    layout.children_offset,
                    Some(Self::direct_children_offset_for(key_len_usize, pairs.len()))
                );
                debug_assert_eq!(layout.dynamic_family_count_offset, None);
            }
            ChildShape::Dynamic { .. } => {
                debug_assert_eq!(
                    layout.dynamic_family_count_offset,
                    Some(Self::dynamic_family_count_offset_for(
                        key_len_usize,
                        pairs.len()
                    ))
                );
                debug_assert_eq!(
                    layout.children_offset,
                    Some(Self::dynamic_children_offset_for(
                        key_len_usize,
                        pairs.len()
                    ))
                );
            }
        }
        let mut allocation = arena.alloc_layout(layout.allocation);
        let base = allocation.as_mut_ptr();

        // SAFETY: every pointer below is derived from `Layout::extend` offsets
        // within this allocation. Each element is written exactly once, no
        // shared reference exists yet, and all lengths were checked above.
        unsafe {
            let keys = base.add(layout.keys_offset).cast::<Value>();
            let boundaries = base.add(layout.boundaries_offset).cast::<u32>();
            let rows = base.add(layout.rows_offset).cast::<RowId>();
            let direct_children = matches!(child_shape, ChildShape::Direct).then(|| {
                base.add(
                    layout
                        .children_offset
                        .expect("direct children need an offset"),
                )
                .cast::<ChildSlot<'exec>>()
            });

            let mut next_key = 0usize;
            for (row_index, &(value, row_id)) in pairs.iter().enumerate() {
                if row_index == 0 || pairs[row_index - 1].0 != value {
                    ptr::write(keys.add(next_key), value);
                    ptr::write(boundaries.add(next_key), row_index as u32);
                    if let Some(children) = direct_children {
                        ptr::write(children.add(next_key), OnceLock::new());
                    }
                    next_key += 1;
                }
                ptr::write(rows.add(row_index), row_id);
            }
            debug_assert_eq!(next_key, key_len_usize);
            ptr::write(boundaries.add(key_len_usize), row_len);
            if let ChildShape::Dynamic { families } = child_shape {
                let family_count = base
                    .add(
                        layout
                            .dynamic_family_count_offset
                            .expect("dynamic children need a family-count offset"),
                    )
                    .cast::<u32>();
                ptr::write(
                    family_count,
                    u32::try_from(families)
                        .expect("a packed trie node cannot contain more than u32::MAX families"),
                );
                let family_table = base
                    .add(
                        layout
                            .children_offset
                            .expect("dynamic children need a table offset"),
                    )
                    .cast::<AtomicPtr<ChildSlot<'exec>>>();
                for family in 0..families {
                    ptr::write(family_table.add(family), AtomicPtr::new(ptr::null_mut()));
                }
            }
            ptr::write(
                base.cast::<Self>(),
                Self {
                    key_len,
                    row_len,
                    _execution: PhantomData,
                },
            );

            // The complete header and every trailing element are initialized
            // before this conversion publishes an immutable arena reference.
            allocation.assume_init_no_drop::<Self>()
        }
    }

    pub(crate) fn values(&self) -> &[Value] {
        // SAFETY: construction initializes exactly `key_len` values at this
        // layout-derived offset, and the arena outlives `self`.
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self)
                    .cast::<u8>()
                    .add(Self::keys_offset())
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
                    .add(Self::boundaries_offset_for(self.key_len()))
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
                    .add(Self::rows_offset_for(self.key_len()))
                    .cast(),
                self.row_len(),
            )
        }
    }

    pub(crate) fn find(&self, value: Value) -> Option<usize> {
        self.values().binary_search(&value).ok()
    }

    pub(crate) fn subset_at(&self, key_index: usize) -> SubsetRef<'_> {
        assert!(key_index < self.key_len(), "packed trie key out of bounds");
        let boundaries = self.boundaries();
        let rows = &self.rows()[boundaries[key_index] as usize..boundaries[key_index + 1] as usize];
        debug_assert!(!rows.is_empty());
        // SAFETY: the constructor requires `(Value, RowId)` order, so the rows
        // within each equal-value range are non-decreasing.
        SubsetRef::Sparse(unsafe { SortedOffsetSlice::new_unchecked(rows) })
    }

    /// Return the child-publication storage trailing this node.
    ///
    /// Direct slots are indexed by key ordinal. Dynamic entries are indexed by
    /// successor family and point to lazily allocated `ChildSlot[key_len]`
    /// arrays.
    fn children(&self) -> PackedChildren<'_, 'exec> {
        if !self.has_children() {
            return PackedChildren::Leaf;
        }

        if self.key_len.has_dynamic_children() {
            let families = self.dynamic_family_count();
            // SAFETY: dynamic construction initialized exactly `families`
            // atomic pointers at this layout-derived offset before publishing
            // the immutable node.
            PackedChildren::Families(unsafe {
                std::slice::from_raw_parts(
                    (self as *const Self)
                        .cast::<u8>()
                        .add(Self::dynamic_children_offset_for(
                            self.key_len(),
                            self.row_len(),
                        ))
                        .cast(),
                    families,
                )
            })
        } else {
            // SAFETY: direct construction initialized exactly `key_len` child
            // locks at this layout-derived offset before publishing the node.
            PackedChildren::Direct(unsafe {
                std::slice::from_raw_parts(
                    (self as *const Self)
                        .cast::<u8>()
                        .add(Self::direct_children_offset_for(
                            self.key_len(),
                            self.row_len(),
                        ))
                        .cast(),
                    self.key_len(),
                )
            })
        }
    }

    /// Get a key's child slot in `family`, lazily publishing the whole dynamic
    /// family array when this is its first use.
    ///
    fn child_slot(
        &self,
        arena: &Handle<'exec>,
        key_index: usize,
        family: usize,
    ) -> &ChildSlot<'exec> {
        assert!(key_index < self.key_len(), "packed trie key out of bounds");
        let family_entry = match self.children() {
            PackedChildren::Leaf => {
                unreachable!("a packed trie leaf cannot publish a child index")
            }
            PackedChildren::Direct(slots) => {
                // Direct nodes have exactly one deterministic successor. The
                // executor passes its plan-local AccessId uniformly as
                // `family`; it is intentionally ignored in this shape.
                return &slots[key_index];
            }
            PackedChildren::Families(families) => {
                assert!(
                    family < families.len(),
                    "packed trie dynamic child family out of bounds"
                );
                &families[family]
            }
        };
        let mut slots = family_entry.load(Ordering::Acquire);
        if slots.is_null() {
            let candidate = Self::allocate_child_slot_family(arena, self.key_len());
            slots = match family_entry.compare_exchange(
                ptr::null_mut(),
                candidate,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => candidate,
                Err(winner) => {
                    debug_assert!(
                        !winner.is_null(),
                        "a failed null-to-candidate CAS must observe its winner"
                    );
                    winner
                }
            };
        }

        // SAFETY: either the Acquire load observed a fully initialized family
        // array, or the CAS published/observed one with a Release/Acquire
        // synchronization edge. Every family array has exactly `key_len`
        // slots, and `key_index` was checked above. Arena lifetime is `'exec`.
        unsafe { &*slots.add(key_index) }
    }

    fn allocate_child_slot_family(arena: &Handle<'exec>, key_len: usize) -> *mut ChildSlot<'exec> {
        debug_assert!(key_len > 0, "a cursor cannot exist for an empty node");
        let layout = Layout::array::<ChildSlot<'exec>>(key_len)
            .expect("packed trie dynamic child-slot family layout overflow");
        let mut allocation = arena.alloc_layout(layout);
        let slots = allocation.as_mut_ptr().cast::<ChildSlot<'exec>>();
        // SAFETY: `slots` is aligned storage for exactly `key_len` ChildSlots.
        // Each OnceLock is initialized once before the pointer can be
        // published. Publishing the first slot as the raw allocation's header
        // ties the returned reference (and therefore this copied pointer) to
        // the arena lifetime; all trailing slots have the same lifetime.
        unsafe {
            for key_index in 0..key_len {
                ptr::write(slots.add(key_index), OnceLock::new());
            }
            allocation.assume_init_no_drop::<ChildSlot<'exec>>() as *const ChildSlot<'exec>
                as *mut ChildSlot<'exec>
        }
    }

    fn dynamic_family_count(&self) -> usize {
        assert!(
            self.key_len.has_dynamic_children(),
            "only dynamic packed trie nodes store a family count"
        );
        // SAFETY: dynamic construction writes one u32 at this derived offset
        // before publishing the immutable node.
        unsafe {
            *(self as *const Self)
                .cast::<u8>()
                .add(Self::dynamic_family_count_offset_for(
                    self.key_len(),
                    self.row_len(),
                ))
                .cast::<u32>() as usize
        }
    }

    pub(crate) fn child_shape(&self) -> ChildShape {
        match self.children() {
            PackedChildren::Leaf => ChildShape::Leaf,
            PackedChildren::Direct(_) => ChildShape::Direct,
            PackedChildren::Families(families) => ChildShape::Dynamic {
                families: families.len(),
            },
        }
    }

    #[cfg(test)]
    fn allocation_bytes(&self) -> usize {
        PackedLayout::new(self.key_len(), self.row_len(), self.child_shape())
            .allocation
            .size()
    }

    fn has_children(&self) -> bool {
        self.key_len.has_children()
    }

    fn key_len(&self) -> usize {
        self.key_len.len()
    }

    fn row_len(&self) -> usize {
        self.row_len as usize
    }

    #[inline]
    fn keys_offset() -> usize {
        align_up(size_of::<Self>(), align_of::<Value>())
    }

    #[inline]
    fn boundaries_offset_for(key_len: usize) -> usize {
        align_up(
            Self::keys_offset() + key_len * size_of::<Value>(),
            align_of::<u32>(),
        )
    }

    #[inline]
    fn rows_offset_for(key_len: usize) -> usize {
        align_up(
            Self::boundaries_offset_for(key_len) + (key_len + 1) * size_of::<u32>(),
            align_of::<RowId>(),
        )
    }

    #[inline]
    fn direct_children_offset_for(key_len: usize, row_len: usize) -> usize {
        align_up(
            Self::rows_offset_for(key_len) + row_len * size_of::<RowId>(),
            align_of::<ChildSlot<'exec>>(),
        )
    }

    #[inline]
    fn dynamic_family_count_offset_for(key_len: usize, row_len: usize) -> usize {
        align_up(
            Self::rows_offset_for(key_len) + row_len * size_of::<RowId>(),
            align_of::<u32>(),
        )
    }

    #[inline]
    fn dynamic_children_offset_for(key_len: usize, row_len: usize) -> usize {
        align_up(
            Self::dynamic_family_count_offset_for(key_len, row_len) + size_of::<u32>(),
            align_of::<AtomicPtr<ChildSlot<'exec>>>(),
        )
    }
}

#[inline]
fn align_up(offset: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    (offset + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use std::{
        mem::{align_of, size_of},
        sync::atomic::{AtomicUsize, Ordering},
    };

    use egglog_concurrency::SharedArena;

    use super::*;
    use crate::{
        numeric_id::NumericId,
        table_shortcuts::fill_table,
        table_spec::{Constraint, Table, WrappedTableRef},
    };

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
    fn packed_trie_header_is_exactly_two_u32s() {
        assert_eq!(size_of::<PackedKeyLen>(), size_of::<u32>());
        assert_eq!(size_of::<PackedTrieNode<'static>>(), 8);
        assert_eq!(align_of::<PackedTrieNode<'static>>(), align_of::<u32>());
        assert_eq!(std::mem::offset_of!(PackedTrieNode<'static>, row_len), 4);
        assert_eq!(
            std::mem::offset_of!(PackedTrieNode<'static>, _execution),
            8,
            "the invariant lifetime marker must not grow the header"
        );

        let leaf = PackedKeyLen::new(PackedKeyLen::LEN_MASK as usize, ChildShape::Leaf);
        assert_eq!(leaf.len(), PackedKeyLen::LEN_MASK as usize);
        assert!(!leaf.has_children());
        assert!(!leaf.has_dynamic_children());

        let direct = PackedKeyLen::new(PackedKeyLen::LEN_MASK as usize, ChildShape::Direct);
        assert_eq!(direct.len(), PackedKeyLen::LEN_MASK as usize);
        assert!(direct.has_children());
        assert!(!direct.has_dynamic_children());

        let dynamic = PackedKeyLen::new(
            PackedKeyLen::LEN_MASK as usize,
            ChildShape::Dynamic { families: 2 },
        );
        assert_eq!(dynamic.0, u32::MAX);
        assert_eq!(dynamic.len(), PackedKeyLen::LEN_MASK as usize);
        assert!(dynamic.has_children());
        assert!(dynamic.has_dynamic_children());
    }

    #[test]
    #[should_panic(expected = "cannot contain 2^30 or more keys")]
    fn packed_trie_rejects_key_count_that_uses_shape_bit() {
        PackedKeyLen::new(PackedKeyLen::LEN_MASK as usize + 1, ChildShape::Leaf);
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
        let node = PackedTrieNode::build_from_sorted_pairs(&handle, &pairs, ChildShape::Direct);

        assert_eq!(node.values(), &[value(1), value(2), value(3)]);
        assert_eq!(node.boundaries(), &[0, 2, 3, 5]);
        assert_eq!(node.rows(), &[row(1), row(3), row(2), row(4), row(5)]);
        assert_eq!(node.find(value(2)), Some(1));
        assert_eq!(node.find(value(9)), None);

        let cursor = PackedCursor::new(node, 0);
        assert_eq!(cursor.size(), 2);
        let SubsetRef::Sparse(one) = cursor.subset() else {
            panic!("noncontiguous range should be sparse")
        };
        assert_eq!(one.inner(), &[row(1), row(3)]);
        let SubsetRef::Sparse(two) = node.subset_at(1) else {
            panic!("packed trie ranges remain sparse even when contiguous")
        };
        assert_eq!(two.inner(), &[row(2)]);
        let SubsetRef::Sparse(three) = node.subset_at(2) else {
            panic!("packed trie ranges remain sparse even when contiguous")
        };
        assert_eq!(three.inner(), &[row(4), row(5)]);
    }

    #[test]
    fn packed_trie_duplicate_rows_are_not_misclassified_as_dense() {
        let arena = SharedArena::new();
        let handle = arena.new_handle();
        // The endpoints span exactly three row ids, but row 1 is repeated and
        // row 2 is absent. An endpoint-only density check would incorrectly
        // turn this into the range 1..4 and admit row 2.
        let pairs = [(value(1), row(1)), (value(1), row(1)), (value(1), row(3))];
        let node = PackedTrieNode::build_from_sorted_pairs(&handle, &pairs, ChildShape::Leaf);

        let SubsetRef::Sparse(rows) = node.subset_at(0) else {
            panic!("a non-contiguous row sequence with duplicates must stay sparse")
        };
        assert_eq!(rows.inner(), &[row(1), row(1), row(3)]);
    }

    #[test]
    #[should_panic(expected = "packed trie cursor key out of bounds")]
    fn packed_cursor_checks_key_ordinal() {
        let arena = SharedArena::new();
        let handle = arena.new_handle();
        let node = PackedTrieNode::build_from_sorted_pairs(
            &handle,
            &[(value(1), row(0))],
            ChildShape::Leaf,
        );
        let _ = PackedCursor::new(node, 1);
    }

    #[test]
    fn packed_trie_builds_from_prefiltered_subset_with_reusable_scratch() {
        // 65 selected rows exercises the radix path. The parity constraint
        // also makes the input subset sparse, while its RowIds remain sorted.
        let table = fill_table(
            (0..130).map(|row| vec![value(row), value((row * 37) % 11), value(row % 2)]),
            1,
            None,
            |_, new| Some(new.to_vec()),
        );
        let filtered = table.refine_one(
            table.all(),
            &Constraint::EqConst {
                col: column(2),
                val: value(0),
            },
        );
        assert_eq!(filtered.as_ref().size(), 65);
        assert!(matches!(filtered.as_ref(), SubsetRef::Sparse(_)));

        let arena = SharedArena::new();
        let handle = arena.new_handle();
        let mut scratch = vec![(value(999), row(999))];
        WrappedTableRef::with_wrapper(&table, |table| {
            let node = PackedTrieNode::build_from_subset(
                &handle,
                table,
                filtered.as_ref(),
                column(1),
                ChildShape::Direct,
                &mut scratch,
            );
            let mut expected: Vec<_> = (0..130)
                .step_by(2)
                .map(|row_id| (value((row_id * 37) % 11), row(row_id)))
                .collect();
            expected.sort_unstable();

            assert_eq!(scratch, expected);
            assert_eq!(node.rows().len(), expected.len());
            let expected_values: Vec<_> = (0..11).map(value).collect();
            assert_eq!(node.values(), expected_values.as_slice());
            for key_index in 0..node.values().len() {
                let lo = node.boundaries()[key_index] as usize;
                let hi = node.boundaries()[key_index + 1] as usize;
                let expected_rows: Vec<_> =
                    expected[lo..hi].iter().map(|&(_, row_id)| row_id).collect();
                assert_eq!(node.rows()[lo..hi], expected_rows);
            }
        });

        // Reusing the same Vec for a much smaller subset must clear both old
        // projected pairs and the retained radix ping-pong region.
        let small = table.refine_one(
            table.refine_one(
                table.all(),
                &Constraint::EqConst {
                    col: column(2),
                    val: value(1),
                },
            ),
            &Constraint::LtConst {
                col: column(0),
                val: value(20),
            },
        );
        WrappedTableRef::with_wrapper(&table, |table| {
            let node = PackedTrieNode::build_from_subset(
                &handle,
                table,
                small.as_ref(),
                column(1),
                ChildShape::Leaf,
                &mut scratch,
            );
            let mut expected: Vec<_> = (1..20)
                .step_by(2)
                .map(|row_id| (value((row_id * 37) % 11), row(row_id)))
                .collect();
            expected.sort_unstable();
            assert_eq!(scratch, expected);
            assert_eq!(node.rows().len(), 10);
        });
    }

    #[test]
    fn packed_trie_leaf_omits_child_locks() {
        let arena = SharedArena::new();
        let handle = arena.new_handle();
        let pairs = [(value(1), row(0)), (value(2), row(1))];
        let leaf = PackedTrieNode::build_from_sorted_pairs(&handle, &pairs, ChildShape::Leaf);
        let branch = PackedTrieNode::build_from_sorted_pairs(&handle, &pairs, ChildShape::Direct);

        assert_eq!(leaf.key_len, PackedKeyLen::new(2, ChildShape::Leaf));
        assert_eq!(branch.key_len, PackedKeyLen::new(2, ChildShape::Direct));
        assert_eq!(leaf.values(), branch.values());
        assert!(matches!(leaf.children(), PackedChildren::Leaf));
        let PackedChildren::Direct(slots) = branch.children() else {
            panic!("a direct branch must expose inline child slots")
        };
        assert_eq!(slots.len(), 2);
        assert!(leaf.allocation_bytes() < branch.allocation_bytes());
    }

    #[test]
    #[should_panic(expected = "a packed trie leaf cannot publish a child index")]
    fn packed_trie_leaf_rejects_child_publication() {
        let arena = SharedArena::new();
        let handle = arena.new_handle();
        let leaf = PackedTrieNode::build_from_sorted_pairs(
            &handle,
            &[(value(1), row(0))],
            ChildShape::Leaf,
        );
        let cursor = PackedCursor::new(leaf, 0);
        cursor.child_index_with(&handle, 0, ChildShape::Leaf, || {
            unreachable!("a leaf must reject publication before invoking the builder")
        });
    }

    #[test]
    fn packed_trie_sections_are_aligned() {
        let arena = SharedArena::new();
        let handle = arena.new_handle();
        let pairs = [(value(1), row(0)), (value(2), row(2))];
        let node = PackedTrieNode::build_from_sorted_pairs(&handle, &pairs, ChildShape::Direct);

        assert_eq!(
            node as *const _ as usize % align_of::<PackedTrieNode<'_>>(),
            0
        );
        assert_eq!(node.values().as_ptr() as usize % align_of::<Value>(), 0);
        assert_eq!(node.boundaries().as_ptr() as usize % align_of::<u32>(), 0);
        assert_eq!(node.rows().as_ptr() as usize % align_of::<RowId>(), 0);
        let PackedChildren::Direct(slots) = node.children() else {
            panic!("a direct node must expose inline child slots")
        };
        assert_eq!(
            slots.as_ptr() as usize % align_of::<OnceLock<&PackedTrieNode<'_>>>(),
            0
        );
        assert_eq!(
            node.values().as_ptr() as usize - node as *const _ as usize,
            size_of::<PackedTrieNode<'_>>()
        );
        assert_eq!(
            node.allocation_bytes(),
            PackedLayout::new(2, 2, ChildShape::Direct)
                .allocation
                .size()
        );
    }

    #[test]
    fn packed_trie_child_is_published_once_under_race() {
        let arena = SharedArena::new();
        let handle = arena.new_handle();
        let parent = PackedTrieNode::build_from_sorted_pairs(
            &handle,
            &[(value(1), row(0))],
            ChildShape::Direct,
        );
        let PackedChildren::Direct(slots) = parent.children() else {
            panic!("a direct node must expose inline child slots")
        };
        let slot = &slots[0];
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
                            &[(value(7), row(9))],
                            ChildShape::Leaf,
                        )
                    });
                    assert_eq!(child.values(), &[value(7)]);
                });
            }
        });

        assert_eq!(initializations.load(Ordering::Relaxed), 1);
        assert_eq!(slot.get().unwrap().values(), &[value(7)]);
    }

    #[test]
    fn packed_cursor_child_index_reuses_cached_node() {
        let table = fill_table(
            (0..8).map(|row_id| vec![value(row_id), value(row_id % 3), value(7 - row_id)]),
            1,
            None,
            |_, new| Some(new.to_vec()),
        );
        let arena = SharedArena::new();
        let handle = arena.new_handle();
        let parent_pairs: Vec<_> = (0..8).map(|row_id| (value(0), row(row_id))).collect();
        let parent =
            PackedTrieNode::build_from_sorted_pairs(&handle, &parent_pairs, ChildShape::Direct);
        let cursor = PackedCursor::new(parent, 0);

        WrappedTableRef::with_wrapper(&table, |table| {
            let mut scratch = Vec::new();
            let first =
                cursor.child_index(&handle, table, column(1), 7, ChildShape::Leaf, &mut scratch);
            assert_eq!(first.values(), &[value(0), value(1), value(2)]);

            let sentinel = (value(999), row(999));
            scratch.clear();
            scratch.push(sentinel);
            let second =
                cursor.child_index(&handle, table, column(1), 7, ChildShape::Leaf, &mut scratch);
            assert!(ptr::eq(first, second));
            assert_eq!(scratch, &[sentinel], "a cache hit must not use scratch");
        });
    }

    #[test]
    fn packed_dynamic_children_keep_two_successor_families_distinct() {
        let table = fill_table(
            (0..12).map(|row_id| vec![value(row_id), value(row_id % 2), value((row_id * 5) % 3)]),
            1,
            None,
            |_, new| Some(new.to_vec()),
        );
        let arena = SharedArena::new();
        {
            let handle = arena.new_handle();
            let parent_pairs: Vec<_> = (0..12).map(|row_id| (value(0), row(row_id))).collect();
            let parent = PackedTrieNode::build_from_sorted_pairs(
                &handle,
                &parent_pairs,
                ChildShape::Dynamic { families: 2 },
            );
            assert_eq!(parent.child_shape(), ChildShape::Dynamic { families: 2 });
            assert_eq!(
                parent.key_len,
                PackedKeyLen::new(1, ChildShape::Dynamic { families: 2 })
            );
            let cursor = PackedCursor::new(parent, 0);

            WrappedTableRef::with_wrapper(&table, |table| {
                let mut scratch = Vec::new();
                let first_family = cursor.child_index(
                    &handle,
                    table,
                    column(1),
                    0,
                    ChildShape::Leaf,
                    &mut scratch,
                );
                assert_eq!(first_family.values(), &[value(0), value(1)]);

                let second_family = cursor.child_index(
                    &handle,
                    table,
                    column(2),
                    1,
                    ChildShape::Leaf,
                    &mut scratch,
                );
                assert_eq!(second_family.values(), &[value(0), value(1), value(2)]);
                assert!(!ptr::eq(first_family, second_family));

                let cached_first = cursor.child_index(
                    &handle,
                    table,
                    column(1),
                    0,
                    ChildShape::Leaf,
                    &mut scratch,
                );
                assert!(ptr::eq(first_family, cached_first));
            });
        }
    }

    #[test]
    fn packed_dynamic_same_family_converges_under_race() {
        let table = fill_table(
            (0..256).map(|row_id| vec![value(row_id), value((row_id * 17) % 31)]),
            1,
            None,
            |_, new| Some(new.to_vec()),
        );
        let arena = SharedArena::new();
        let handle = arena.new_handle();
        let parent_pairs: Vec<_> = (0..256).map(|row_id| (value(0), row(row_id))).collect();
        let parent = PackedTrieNode::build_from_sorted_pairs(
            &handle,
            &parent_pairs,
            ChildShape::Dynamic { families: 2 },
        );
        let cursor = PackedCursor::new(parent, 0);
        let published = AtomicUsize::new(0);
        let start = std::sync::Barrier::new(16);

        WrappedTableRef::with_wrapper(&table, |table| {
            std::thread::scope(|scope| {
                for _ in 0..16 {
                    let arena = &arena;
                    let published = &published;
                    let start = &start;
                    scope.spawn(move || {
                        let handle = arena.new_handle();
                        let mut scratch = Vec::new();
                        start.wait();
                        let child = cursor.child_index(
                            &handle,
                            table,
                            column(1),
                            1,
                            ChildShape::Leaf,
                            &mut scratch,
                        );
                        assert_eq!(child.values().len(), 31);
                        let address = child as *const _ as usize;
                        let observed = published
                            .compare_exchange(0, address, Ordering::AcqRel, Ordering::Acquire)
                            .unwrap_or_else(|already_published| already_published);
                        assert!(observed == 0 || observed == address);
                    });
                }
            });
        });

        assert_ne!(published.load(Ordering::Acquire), 0);
    }

    #[test]
    fn packed_cursor_child_index_uses_prefiltered_subset() {
        let table = fill_table(
            (0..8).map(|row_id| vec![value(row_id), value(row_id % 3), value(7 - row_id)]),
            1,
            None,
            |_, new| Some(new.to_vec()),
        );
        let filtered = table.refine_one(
            table.all(),
            &Constraint::LtConst {
                col: column(0),
                val: value(4),
            },
        );
        let arena = SharedArena::new();
        let handle = arena.new_handle();
        let parent_pairs: Vec<_> = (0..8).map(|row_id| (value(0), row(row_id))).collect();
        let parent =
            PackedTrieNode::build_from_sorted_pairs(&handle, &parent_pairs, ChildShape::Direct);
        let cursor = PackedCursor::new(parent, 0);

        WrappedTableRef::with_wrapper(&table, |table| {
            let mut scratch = Vec::new();
            let child = cursor.child_index_from_subset(
                &handle,
                table,
                filtered.as_ref(),
                column(1),
                0,
                ChildShape::Leaf,
                &mut scratch,
            );
            assert_eq!(child.rows().len(), 4);
            assert!(child.rows().iter().all(|row_id| row_id.index() < 4));
        });
    }

    #[test]
    fn packed_cursor_child_index_is_published_once_under_race() {
        let table = fill_table(
            (0..96).map(|row_id| vec![value(row_id), value((row_id * 17) % 13), value(row_id % 5)]),
            1,
            None,
            |_, new| Some(new.to_vec()),
        );
        let arena = SharedArena::new();
        let handle = arena.new_handle();
        let parent_pairs: Vec<_> = (0..96).map(|row_id| (value(0), row(row_id))).collect();
        let parent =
            PackedTrieNode::build_from_sorted_pairs(&handle, &parent_pairs, ChildShape::Direct);
        let cursor = PackedCursor::new(parent, 0);
        let published = AtomicUsize::new(0);

        WrappedTableRef::with_wrapper(&table, |table| {
            std::thread::scope(|scope| {
                for _ in 0..8 {
                    let arena = &arena;
                    let published = &published;
                    scope.spawn(move || {
                        let handle = arena.new_handle();
                        let mut scratch = Vec::new();
                        let child = cursor.child_index(
                            &handle,
                            table,
                            column(1),
                            0,
                            ChildShape::Leaf,
                            &mut scratch,
                        );
                        assert_eq!(child.values().len(), 13);
                        let address = child as *const _ as usize;
                        let observed = published
                            .compare_exchange(0, address, Ordering::AcqRel, Ordering::Acquire)
                            .unwrap_or_else(|already_published| already_published);
                        assert!(observed == 0 || observed == address);
                    });
                }
            });
        });
        assert_ne!(published.load(Ordering::Acquire), 0);
    }

    #[test]
    #[should_panic(expected = "packed trie child cache shape mismatch")]
    fn packed_cursor_rejects_cached_child_with_another_shape() {
        let table = fill_table(
            (0..4).map(|row_id| vec![value(row_id), value(row_id % 2), value(row_id + 10)]),
            1,
            None,
            |_, new| Some(new.to_vec()),
        );
        let arena = SharedArena::new();
        let handle = arena.new_handle();
        let parent_pairs: Vec<_> = (0..4).map(|row_id| (value(0), row(row_id))).collect();
        let parent =
            PackedTrieNode::build_from_sorted_pairs(&handle, &parent_pairs, ChildShape::Direct);
        let cursor = PackedCursor::new(parent, 0);

        WrappedTableRef::with_wrapper(&table, |table| {
            let mut scratch = Vec::new();
            cursor.child_index(&handle, table, column(1), 0, ChildShape::Leaf, &mut scratch);
            cursor.child_index(
                &handle,
                table,
                column(1),
                0,
                ChildShape::Direct,
                &mut scratch,
            );
        });
    }
}
