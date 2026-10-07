//! Adapt catalog, shared-root, residual, and packed indexes to join probes.

use std::{
    cell::{OnceCell, RefCell},
    cmp,
    sync::Arc,
};

use egglog_concurrency::{Handle, SharedArena};
use smallvec::SmallVec;

use crate::{
    common::Value,
    hash_index::{ColumnIndex, Index, IndexPosition, TupleIndex},
    numeric_id::NumericId,
    offsets::{OffsetRange, RowId, Subset, SubsetRef},
    table_spec::{Constraint, WrappedTableRef},
};

use super::{
    AtomId, ColumnIds,
    frame_update::FrameUpdates,
    packed_cache::{FamilyId, OwnedAtomRows},
    packed_trie::{ChildShape, PackedCursor, TrieNode},
    plan::{ScanSpec, SingleScanSpec},
    prepared_index::{ContinuationPosition, PreparedIndexRef, RootContinuationCache},
    residual_index::{InlineRows, SMALL_RESIDUAL, SmallColumnIndex, SmallExactProbe},
};

/// Intersect a `SubsetRef` with a dense `OffsetRange` and return the result as a
/// borrowed `SubsetRef`, or `None` if the intersection is empty.
///
/// This function never allocates — it borrows into
/// the source data via `subslice`. Use this in `for_each` paths where the result
/// may be discarded (e.g., empty after refinement), to avoid pool allocations.
#[inline]
fn intersect_with_dense_ref<'a>(v: SubsetRef<'a>, range: OffsetRange) -> Option<SubsetRef<'a>> {
    match v {
        SubsetRef::Dense(r) => {
            let resl = cmp::max(r.start, range.start);
            let resr = cmp::min(r.end, range.end);
            if resl >= resr {
                None
            } else {
                Some(SubsetRef::Dense(OffsetRange::new(resl, resr)))
            }
        }
        SubsetRef::Sparse(s) => {
            let l = s.binary_search_by_id(range.start);
            let r = s.binary_search_from(l, range.end);
            if l >= r {
                None
            } else {
                Some(SubsetRef::Sparse(s.subslice(l, r)))
            }
        }
    }
}

/// Seek `key` in a sorted scalar index without moving backward.
///
/// Exponential search followed by a bounded binary search avoids walking a
/// large target when the sorted query is sparse.
pub(super) fn seek_sorted_key(
    key: Value,
    target_len: usize,
    target_cursor: &mut usize,
    target_at: impl Fn(usize) -> Value,
) -> bool {
    if *target_cursor >= target_len {
        return false;
    }

    let current = target_at(*target_cursor);
    match current.cmp(&key) {
        cmp::Ordering::Equal => true,
        cmp::Ordering::Greater => false,
        cmp::Ordering::Less => {
            let base = *target_cursor;
            let mut step = 1usize;
            while let Some(position) = base.checked_add(step).filter(|&pos| pos < target_len) {
                if target_at(position) >= key {
                    break;
                }
                let Some(next) = step.checked_mul(2) else {
                    step = target_len;
                    break;
                };
                step = next;
            }

            let previous = step / 2;
            let mut lo = base
                .saturating_add(previous)
                .saturating_add(1)
                .min(target_len);
            let mut hi = base.saturating_add(step).saturating_add(1).min(target_len);
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if target_at(mid) < key {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            *target_cursor = lo;
            if lo >= target_len {
                return false;
            }
            target_at(lo) == key
        }
    }
}

/// A continuation for rows borrowed from a catalog or root index. Its
/// [`RootContinuationCache`] is plan-local below an unshared root and run-wide
/// below a shared one.
#[derive(Clone, Copy)]
pub(super) struct CatalogContinuation<'rows> {
    pub(super) cache: &'rows RootContinuationCache,
    pub(super) position: ContinuationPosition,
}

/// The rows currently associated with an atom during one plan execution.
/// Owned rows retain a plan's header-filtered subset or a frame's residual
/// subset. An indexed cursor borrows a first-level group from a prepared
/// persistent index or a shared round-local root index and carries its
/// continuation slot. Every lower cursor is just a packed node plus a key
/// ordinal. Dense singletons
/// come from cover scans and are packed lazily if the atom is probed again, as
/// are owned residuals left by a constrained catalog match.
#[derive(Clone)]
pub(super) enum AtomRowsKind<'rows, 'exec> {
    /// An owned subset serving as the atom's plan root or a frame-local residual.
    Owned(Arc<OwnedAtomRows>),
    /// A persistent index group. It may still contain stale rows; every
    /// consumer of retained rows skips them.
    Catalog {
        subset: SubsetRef<'rows>,
        continuation: Option<CatalogContinuation<'rows>>,
    },
    Packed(PackedCursor<'rows, 'exec>),
    /// Owned small residual passed directly between buffered or parallel
    /// frames. Unlike `Catalog` and `Packed`, this variant borrows no index or
    /// arena storage.
    Inline(InlineRows),
    Dense(OffsetRange),
}

/// Keep the row count beside the cursor, as the old owning subsets did.
/// DVO compares cardinalities repeatedly; resolving packed boundaries or
/// redispatching the storage representation on every comparison adds work.
/// Rows are immutable during a ruleset run, and constructors establish this
/// count before a handle enters the bindings or an update buffer.
#[derive(Clone)]
pub(super) struct AtomRows<'rows, 'exec> {
    kind: AtomRowsKind<'rows, 'exec>,
    cardinality: usize,
}

impl std::fmt::Debug for AtomRows<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomRows")
            .field("kind", &self.kind_name())
            .field("size", &self.size())
            .finish()
    }
}

impl<'rows, 'exec> AtomRows<'rows, 'exec>
where
    'exec: 'rows,
{
    /// A short representation label used by the custom [`std::fmt::Debug`]
    /// output. The underlying row sets may be large, so diagnostics report only
    /// this label and their cardinality rather than formatting their contents.
    fn kind_name(&self) -> &'static str {
        match &self.kind {
            AtomRowsKind::Owned(_) => "owned",
            AtomRowsKind::Catalog { .. } => "catalog",
            AtomRowsKind::Packed(_) => "packed",
            AtomRowsKind::Inline(_) => "inline",
            AtomRowsKind::Dense(_) => "dense",
        }
    }

    /// Wrap the live rows left after filtering a borrowed group.
    pub(super) fn from_owned(subset: Subset) -> Self {
        match subset {
            Subset::Dense(range) => Self::dense(range),
            Subset::Sparse(rows) if rows.slice().inner().len() <= SMALL_RESIDUAL => {
                Self::inline(InlineRows::from_sorted(rows.slice().inner()))
            }
            subset => Self::owned(Arc::new(OwnedAtomRows::new_residual(subset))),
        }
    }

    pub(super) fn subset(&self) -> SubsetRef<'_> {
        match &self.kind {
            AtomRowsKind::Owned(root) => root.subset.as_ref(),
            AtomRowsKind::Catalog { subset, .. } => *subset,
            AtomRowsKind::Packed(cursor) => cursor.subset(),
            AtomRowsKind::Inline(rows) => rows.subset(),
            AtomRowsKind::Dense(range) => SubsetRef::Dense(*range),
        }
    }

    #[inline]
    pub(super) fn size(&self) -> usize {
        self.cardinality
    }

    pub(super) fn kind(&self) -> &AtomRowsKind<'rows, 'exec> {
        &self.kind
    }

    pub(super) fn owned(root: Arc<OwnedAtomRows>) -> Self {
        Self {
            cardinality: root.subset.size(),
            kind: AtomRowsKind::Owned(root),
        }
    }
    pub(super) fn packed(cursor: PackedCursor<'rows, 'exec>) -> Self {
        Self {
            cardinality: cursor.size(),
            kind: AtomRowsKind::Packed(cursor),
        }
    }
    pub(super) fn inline(rows: InlineRows) -> Self {
        Self {
            cardinality: rows.len(),
            kind: AtomRowsKind::Inline(rows),
        }
    }
    pub(super) fn dense(range: OffsetRange) -> Self {
        Self {
            cardinality: range.size(),
            kind: AtomRowsKind::Dense(range),
        }
    }
    pub(super) fn catalog(
        subset: SubsetRef<'rows>,
        continuation: Option<CatalogContinuation<'rows>>,
    ) -> Self {
        Self {
            cardinality: subset.size(),
            kind: AtomRowsKind::Catalog {
                subset,
                continuation,
            },
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.size() == 0
    }

    #[cfg(test)]
    pub(super) fn owned_arc(&self) -> &Arc<OwnedAtomRows> {
        let AtomRowsKind::Owned(root) = &self.kind else {
            panic!("expected owned rows")
        };
        root
    }
}

impl<'rows, 'exec> From<Arc<OwnedAtomRows>> for AtomRows<'rows, 'exec> {
    fn from(root: Arc<OwnedAtomRows>) -> Self {
        Self::owned(root)
    }
}

/// The row-level checks a persistent index group must still pass.
///
/// Persistent indexes group physical rows, so a group can hold tombstoned rows
/// and ignores the scan's slow constraints. A plan applies slow constraints
/// exactly once, at this scan, so a constrained match materializes the rows
/// that survive. Stale rows only need excluding from existence-only matches:
/// every consumer of retained rows skips them.
#[derive(Clone, Copy)]
pub(super) struct CatalogFilter<'ctx> {
    pub(super) table: WrappedTableRef<'ctx>,
    pub(super) constraints: &'ctx [Constraint],
    /// Whether the table currently holds stale rows.
    pub(super) check_live: bool,
}

impl CatalogFilter<'_> {
    /// Turn a borrowed index group into a probe match, or `None` when no live
    /// row satisfies the constraints.
    #[inline]
    fn resolve<'rows, 'exec>(
        &self,
        subset: SubsetRef<'rows>,
        keep_rows: bool,
        continuation: Option<CatalogContinuation<'rows>>,
    ) -> Option<ProbeMatch<'rows, 'exec>> {
        if self.constraints.is_empty() {
            if keep_rows {
                return Some(ProbeMatch::Rows(AtomRows::catalog(subset, continuation)));
            }
            if !self.check_live {
                return Some(ProbeMatch::Present);
            }
        }
        self.resolve_slow(subset, keep_rows)
    }

    /// Check the group row by row.
    // Out of line: inlining this into `get_subset` measurably slowed the
    // common unfiltered probe.
    #[inline(never)]
    fn resolve_slow<'rows, 'exec>(
        &self,
        subset: SubsetRef<'rows>,
        keep_rows: bool,
    ) -> Option<ProbeMatch<'rows, 'exec>> {
        if !keep_rows {
            return self
                .table
                .contains_match(subset, self.constraints)
                .then_some(ProbeMatch::Present);
        }
        let filtered = self
            .table
            .refine_ref(subset, self.constraints, self.check_live);
        (filtered.size() != 0).then(|| ProbeMatch::Rows(AtomRows::from_owned(filtered)))
    }
}

/// Physical strategies available for looking up or enumerating the current
/// rows of one atom. `JoinState::get_index` chooses one variant for each
/// logical scan based on the source subset, requested columns, and whether a
/// reusable table index is valid.
pub(super) enum ProbeIndex<'ctx, 'rows, 'exec> {
    /// A persistent, fully refreshed multi-column table index. This is used for
    /// a large dense root with cacheable columns. `intersect_outer` clips
    /// results when the root is a dense subrange rather than the whole table,
    /// and `filter` applies the checks the stored groups cannot encode.
    CachedTuple {
        intersect_outer: Option<OffsetRange>,
        table: &'rows Index<TupleIndex>,
        continuations: Option<&'rows RootContinuationCache>,
        child_shape: ChildShape,
        filter: CatalogFilter<'ctx>,
    },
    /// The single-column counterpart of [`Self::CachedTuple`], selected under
    /// the same catalog-index conditions.
    CachedColumn {
        intersect_outer: Option<OffsetRange>,
        table: &'rows Index<ColumnIndex>,
        continuations: Option<&'rows RootContinuationCache>,
        child_shape: ChildShape,
        filter: CatalogFilter<'ctx>,
    },
    /// An inline scalar index for a source containing at most
    /// [`super::residual_index::SMALL_RESIDUAL`] rows and no publication slot
    /// (a root, dense singleton, inline residual, or terminal catalog match).
    /// It supports both exact lookup and enumeration without constructing a
    /// general packed trie. Tiny sources with a slot use a packed node
    /// instead, since the slot lets later probes reuse it.
    SmallColumn(SmallColumnIndex),
    /// An exact-only multi-column probe over an inline residual. Join stages
    /// select it when the source is already [`AtomRows::inline`]; it scans those
    /// few rows directly and is never used as an enumeration leader.
    SmallExact(SmallExactProbe<'ctx>),
    /// The general fallback: an arena-allocated packed trie over an arbitrary
    /// source subset, with lower column indexes constructed lazily.
    Packed(PackedProbe<'ctx, 'rows, 'exec>),
    /// A scalar packed index needs only its immutable key/row arrays.
    /// Tuple descent state belongs only to accesses projecting several columns.
    PackedColumn(&'rows TrieNode<'exec>),
}

/// Borrowed, ordered scalar keys used by merge and galloping intersections.
#[derive(Clone, Copy)]
pub(super) enum SortedScalarProbe<'a, 'exec> {
    Small(&'a SmallColumnIndex),
    Packed(&'a TrieNode<'exec>),
}

impl SortedScalarProbe<'_, '_> {
    pub(super) fn len(self) -> usize {
        match self {
            Self::Small(index) => index.n_keys,
            Self::Packed(index) => index.values().len(),
        }
    }

    pub(super) fn value_at(self, key_index: usize) -> Value {
        match self {
            Self::Small(index) => index.keys[key_index],
            Self::Packed(index) => index.values()[key_index],
        }
    }
}

/// A successful probe either carries rows needed by a later stage or only
/// records existence when this atom is dead in the remaining join tail. The
/// latter case avoids copying an inline subset into every buffered frame.
pub(super) enum ProbeMatch<'rows, 'exec> {
    Present,
    Rows(AtomRows<'rows, 'exec>),
}

/// Worker-local arena handle initialized only by a path that actually builds
/// a packed node. Cover-only queries and catalog probes never touch the
/// arena allocator.
pub(super) struct LazyArenaHandle<'exec> {
    arena: &'exec SharedArena,
    handle: OnceCell<Handle<'exec>>,
}

impl<'exec> LazyArenaHandle<'exec> {
    pub(super) fn new(arena: &'exec SharedArena) -> Self {
        Self {
            arena,
            handle: OnceCell::new(),
        }
    }

    pub(super) fn get(&self) -> &Handle<'exec> {
        self.handle.get_or_init(|| self.arena.new_handle())
    }
}

impl<'rows, 'exec> ProbeMatch<'rows, 'exec> {
    #[inline]
    pub(super) fn refine(self, atom: AtomId, updates: &mut FrameUpdates<'rows, 'exec>) {
        if let Self::Rows(rows) = self {
            updates.refine_atom(atom, rows);
        }
    }
}

/// How a multi-column probe publishes the packed node below each key.
///
/// Below an unshared root, children are plan-local: tuple interiors use the
/// single direct slot and the final level uses the plan's tail shape. Below a
/// shared root, every level uses the table's shared shape and the run-global
/// family of the next column, so other plans find the same nodes.
#[derive(Clone, Copy)]
pub(super) enum Descent<'rows> {
    Local {
        terminal_child_shape: ChildShape,
    },
    Shared {
        child_shape: ChildShape,
        families: &'rows [FamilyId],
    },
}

impl Descent<'_> {
    /// The `(family, shape)` of the node indexing column `depth + 1` of a
    /// `column_count`-column probe.
    fn child(self, depth: usize, column_count: usize) -> (usize, ChildShape) {
        match self {
            Self::Local {
                terminal_child_shape,
            } => {
                let child_shape = if depth + 2 < column_count {
                    ChildShape::Direct
                } else {
                    terminal_child_shape
                };
                (0, child_shape)
            }
            Self::Shared {
                child_shape,
                families,
            } => (families[depth + 1].index(), child_shape),
        }
    }
}

pub(super) struct PackedProbe<'ctx, 'rows, 'exec> {
    pub(super) first: &'rows TrieNode<'exec>,
    pub(super) columns: ColumnIds,
    pub(super) table: WrappedTableRef<'ctx>,
    pub(super) handle: &'ctx LazyArenaHandle<'exec>,
    pub(super) scratch: &'ctx RefCell<Vec<(Value, RowId)>>,
    pub(super) descent: Descent<'rows>,
}

impl<'ctx, 'rows, 'exec> PackedProbe<'ctx, 'rows, 'exec>
where
    'exec: 'rows,
{
    fn get(&self, key: &[Value]) -> Option<AtomRows<'rows, 'exec>> {
        debug_assert_eq!(key.len(), self.columns.len());
        let mut node = self.first;
        let mut terminal = None;
        for (depth, (&_column, &value)) in self.columns.iter().zip(key).enumerate() {
            let cursor = PackedCursor::new(node, node.find(value)?);
            terminal = Some(cursor);
            if depth + 1 < self.columns.len() {
                let (family, child_shape) = self.descent.child(depth, self.columns.len());
                node = cursor.child_index(
                    self.handle.get(),
                    self.table,
                    self.columns[depth + 1],
                    family,
                    child_shape,
                    &mut self.scratch.borrow_mut(),
                );
            }
        }
        terminal.map(AtomRows::packed)
    }

    fn for_each_recur(
        &self,
        node: &'rows TrieNode<'exec>,
        depth: usize,
        key: &mut SmallVec<[Value; 4]>,
        f: &mut impl FnMut(&[Value], AtomRows<'rows, 'exec>),
    ) {
        for (key_index, &value) in node.values().iter().enumerate() {
            key.push(value);
            let cursor = PackedCursor::new(node, key_index);
            if depth + 1 == self.columns.len() {
                f(key, AtomRows::packed(cursor));
            } else {
                let (family, child_shape) = self.descent.child(depth, self.columns.len());
                let child = cursor.child_index(
                    self.handle.get(),
                    self.table,
                    self.columns[depth + 1],
                    family,
                    child_shape,
                    &mut self.scratch.borrow_mut(),
                );
                self.for_each_recur(child, depth + 1, key, f);
            }
            key.pop();
        }
    }

    fn for_each(&self, f: &mut impl FnMut(&[Value], AtomRows<'rows, 'exec>)) {
        let mut key = SmallVec::new();
        self.for_each_recur(self.first, 0, &mut key, f);
    }
}

pub(super) struct Prober<'ctx, 'rows, 'exec> {
    pub(super) source: AtomRows<'rows, 'exec>,
    pub(super) ix: ProbeIndex<'ctx, 'rows, 'exec>,
    /// Whether a match returns [`ProbeMatch::Rows`] so a later stage can probe,
    /// scan, or materialize this atom's refined rows. If the atom is never read
    /// again, return only [`ProbeMatch::Present`] and skip refining its frame.
    /// This can be true even with [`ChildShape::Leaf`]: a later cover scan needs
    /// the rows without needing another column index.
    pub(super) keep_rows: bool,
}

/// Normalized input to `JoinState::get_index` for one indexed scan.
///
/// Both scalar [`SingleScanSpec`]s and tuple [`ScanSpec`]s are converted to
/// this form so physical-strategy selection has one code path. It identifies
/// the atom and projected columns, carries constraints that must be applied to
/// the source rows, records whether later stages need the matching rows or only
/// existence, and supplies the prepared sidecar slot used for cached indexes
/// and continuations. `terminal_child_shape` describes how execution may
/// continue after the final requested column.
pub(super) struct ProbeRequest<'scan, 'rows> {
    pub(super) atom: AtomId,
    pub(super) columns: ColumnIds,
    pub(super) constraints: &'scan [Constraint],
    pub(super) keep_rows: bool,
    pub(super) terminal_child_shape: ChildShape,
    pub(super) prepared: PreparedIndexRef<'rows>,
}

impl<'scan, 'rows> ProbeRequest<'scan, 'rows> {
    pub(super) fn column(
        scan: &'scan SingleScanSpec,
        keep_rows: bool,
        terminal_child_shape: ChildShape,
        prepared: PreparedIndexRef<'rows>,
    ) -> Self {
        Self {
            atom: scan.atom,
            columns: smallvec::smallvec![scan.column],
            constraints: &scan.cs,
            keep_rows,
            terminal_child_shape,
            prepared,
        }
    }

    pub(super) fn tuple(
        scan: &'scan ScanSpec,
        keep_rows: bool,
        terminal_child_shape: ChildShape,
        prepared: PreparedIndexRef<'rows>,
    ) -> Self {
        Self {
            atom: scan.to_index.atom,
            columns: scan.to_index.vars.clone(),
            constraints: &scan.constraints,
            keep_rows,
            terminal_child_shape,
            prepared,
        }
    }
}

impl<'ctx, 'rows, 'exec> Prober<'ctx, 'rows, 'exec>
where
    'exec: 'rows,
{
    fn keep_or_discard(rows: AtomRows<'rows, 'exec>, keep_rows: bool) -> ProbeMatch<'rows, 'exec> {
        if keep_rows {
            ProbeMatch::Rows(rows)
        } else {
            ProbeMatch::Present
        }
    }

    /// A nonterminal catalog result retains its [`IndexPosition`] so a later
    /// access can use the corresponding continuation slot to publish a packed
    /// child index. A leaf never needs that slot: its rows only feed a later
    /// cover or materialization barrier.
    fn catalog_continuation(
        continuations: Option<&'rows RootContinuationCache>,
        position: IndexPosition,
        child_shape: ChildShape,
    ) -> Option<CatalogContinuation<'rows>> {
        (child_shape != ChildShape::Leaf).then(|| CatalogContinuation {
            cache: continuations.expect("nonterminal catalog access needs continuation storage"),
            position: position.into(),
        })
    }

    pub(super) fn get_subset(&self, key: &[Value]) -> Option<ProbeMatch<'rows, 'exec>> {
        match &self.ix {
            ProbeIndex::CachedTuple {
                intersect_outer,
                table,
                continuations,
                child_shape,
                filter,
            } => {
                let table: &'rows Index<TupleIndex> = table;
                if *child_shape == ChildShape::Leaf || !self.keep_rows {
                    debug_assert!(self.keep_rows || *child_shape == ChildShape::Leaf);
                    let subset = table.get_subset(key)?;
                    let subset = if let Some(range) = intersect_outer {
                        intersect_with_dense_ref(subset, *range)?
                    } else {
                        subset
                    };
                    return filter.resolve(subset, self.keep_rows, None);
                }
                let (position, subset) = table.get_subset_positioned(key)?;
                let subset = if let Some(range) = intersect_outer {
                    intersect_with_dense_ref(subset, *range)?
                } else {
                    subset
                };
                filter.resolve(
                    subset,
                    self.keep_rows,
                    Self::catalog_continuation(*continuations, position, *child_shape),
                )
            }
            ProbeIndex::CachedColumn {
                intersect_outer,
                table,
                continuations,
                child_shape,
                filter,
            } => {
                debug_assert_eq!(key.len(), 1);
                let table: &'rows Index<ColumnIndex> = table;
                if *child_shape == ChildShape::Leaf || !self.keep_rows {
                    debug_assert!(self.keep_rows || *child_shape == ChildShape::Leaf);
                    let subset = table.get_subset(&key[0])?;
                    let subset = if let Some(range) = intersect_outer {
                        intersect_with_dense_ref(subset, *range)?
                    } else {
                        subset
                    };
                    return filter.resolve(subset, self.keep_rows, None);
                }
                let (position, subset) = table.get_subset_positioned(&key[0])?;
                let subset = if let Some(range) = intersect_outer {
                    intersect_with_dense_ref(subset, *range)?
                } else {
                    subset
                };
                filter.resolve(
                    subset,
                    self.keep_rows,
                    Self::catalog_continuation(*continuations, position, *child_shape),
                )
            }
            ProbeIndex::SmallColumn(index) => {
                let [value] = key else {
                    return None;
                };
                let key_index = index.find(*value)?;
                Some(if self.keep_rows {
                    ProbeMatch::Rows(AtomRows::inline(index.rows_at(key_index)))
                } else {
                    ProbeMatch::Present
                })
            }
            ProbeIndex::SmallExact(exact) => exact.get(key, self.keep_rows),
            ProbeIndex::PackedColumn(node) => {
                let [value] = key else {
                    return None;
                };
                let ordinal = node.find(*value)?;
                Some(Self::keep_or_discard(
                    AtomRows::packed(PackedCursor::new(node, ordinal)),
                    self.keep_rows,
                ))
            }
            ProbeIndex::Packed(packed) => packed
                .get(key)
                .map(|rows| Self::keep_or_discard(rows, self.keep_rows)),
        }
    }

    /// Borrow the ordered keys when this probe supports scalar intersection.
    pub(super) fn sorted_scalar_probe(&self) -> Option<SortedScalarProbe<'_, 'exec>> {
        match &self.ix {
            ProbeIndex::SmallColumn(index) => Some(SortedScalarProbe::Small(index)),
            ProbeIndex::PackedColumn(node) => Some(SortedScalarProbe::Packed(node)),
            ProbeIndex::CachedTuple { .. }
            | ProbeIndex::CachedColumn { .. }
            | ProbeIndex::SmallExact(..)
            | ProbeIndex::Packed(..) => None,
        }
    }

    /// Reconstruct the match at an ordinal returned by a sorted scalar probe.
    /// Preserves the row-retention policy used by ordinary key lookup.
    pub(super) fn sorted_match_at(&self, key_index: usize) -> ProbeMatch<'rows, 'exec> {
        match &self.ix {
            ProbeIndex::SmallColumn(index) => {
                if self.keep_rows {
                    ProbeMatch::Rows(AtomRows::inline(index.rows_at(key_index)))
                } else {
                    ProbeMatch::Present
                }
            }
            ProbeIndex::PackedColumn(node) => {
                let rows = AtomRows::packed(PackedCursor::new(node, key_index));
                Self::keep_or_discard(rows, self.keep_rows)
            }
            ProbeIndex::CachedTuple { .. }
            | ProbeIndex::CachedColumn { .. }
            | ProbeIndex::SmallExact(..)
            | ProbeIndex::Packed(..) => {
                unreachable!("only a sorted scalar probe has a match ordinal")
            }
        }
    }

    pub(super) fn for_each(&self, mut f: impl FnMut(&[Value], ProbeMatch<'rows, 'exec>)) {
        match &self.ix {
            ProbeIndex::CachedTuple {
                intersect_outer,
                table,
                continuations,
                child_shape,
                filter,
            } => {
                let table: &'rows Index<TupleIndex> = table;
                table.for_each_positioned(|position, key, subset| {
                    let subset = if let Some(range) = intersect_outer {
                        let Some(subset) = intersect_with_dense_ref(subset, *range) else {
                            return;
                        };
                        subset
                    } else {
                        subset
                    };
                    if let Some(found) = filter.resolve(
                        subset,
                        self.keep_rows,
                        Self::catalog_continuation(*continuations, position, *child_shape),
                    ) {
                        f(key, found);
                    }
                });
            }
            ProbeIndex::CachedColumn {
                intersect_outer,
                table,
                continuations,
                child_shape,
                filter,
            } => {
                let table: &'rows Index<ColumnIndex> = table;
                table.for_each_positioned(|position, value, subset| {
                    let subset = if let Some(range) = intersect_outer {
                        let Some(subset) = intersect_with_dense_ref(subset, *range) else {
                            return;
                        };
                        subset
                    } else {
                        subset
                    };
                    if let Some(found) = filter.resolve(
                        subset,
                        self.keep_rows,
                        Self::catalog_continuation(*continuations, position, *child_shape),
                    ) {
                        f(&[*value], found);
                    }
                });
            }
            ProbeIndex::SmallColumn(index) => {
                for key_index in 0..index.n_keys {
                    let rows = if self.keep_rows {
                        ProbeMatch::Rows(AtomRows::inline(index.rows_at(key_index)))
                    } else {
                        ProbeMatch::Present
                    };
                    f(&index.keys[key_index..key_index + 1], rows);
                }
            }
            ProbeIndex::SmallExact(..) => {
                unreachable!("small multi-column residuals are exact-probe only")
            }
            ProbeIndex::PackedColumn(node) => {
                for (ordinal, value) in node.values().iter().enumerate() {
                    let rows = AtomRows::packed(PackedCursor::new(node, ordinal));
                    f(
                        std::slice::from_ref(value),
                        Self::keep_or_discard(rows, self.keep_rows),
                    );
                }
            }
            ProbeIndex::Packed(packed) => packed.for_each(&mut |key, rows| {
                f(key, Self::keep_or_discard(rows, self.keep_rows));
            }),
        }
    }

    pub(super) fn for_each_shard(
        &self,
        shard: usize,
        mut f: impl FnMut(&[Value], ProbeMatch<'rows, 'exec>),
    ) {
        match &self.ix {
            ProbeIndex::CachedTuple {
                intersect_outer,
                table,
                continuations,
                child_shape,
                filter,
            } => {
                let table: &'rows Index<TupleIndex> = table;
                table.for_each_shard_positioned(shard, |position, key, subset| {
                    let subset = if let Some(range) = intersect_outer {
                        let Some(subset) = intersect_with_dense_ref(subset, *range) else {
                            return;
                        };
                        subset
                    } else {
                        subset
                    };
                    if let Some(found) = filter.resolve(
                        subset,
                        self.keep_rows,
                        Self::catalog_continuation(*continuations, position, *child_shape),
                    ) {
                        f(key, found);
                    }
                });
            }
            ProbeIndex::CachedColumn {
                intersect_outer,
                table,
                continuations,
                child_shape,
                filter,
            } => {
                let table: &'rows Index<ColumnIndex> = table;
                table.for_each_shard_positioned(shard, |position, value, subset| {
                    let subset = if let Some(range) = intersect_outer {
                        let Some(subset) = intersect_with_dense_ref(subset, *range) else {
                            return;
                        };
                        subset
                    } else {
                        subset
                    };
                    if let Some(found) = filter.resolve(
                        subset,
                        self.keep_rows,
                        Self::catalog_continuation(*continuations, position, *child_shape),
                    ) {
                        f(&[*value], found);
                    }
                });
            }
            ProbeIndex::SmallColumn(..)
            | ProbeIndex::SmallExact(..)
            | ProbeIndex::Packed(..)
            | ProbeIndex::PackedColumn(..) => {
                unreachable!("only persistent root indexes expose physical shards")
            }
        }
    }

    /// Visit a disjoint ordinal range of an unsharded scalar index.
    ///
    /// `Intersect` stages always probe one column, so partitioning the first
    /// (and only) key level preserves complete key groups. The backing
    /// packed index remains borrowed by every coarse task.
    pub(super) fn for_each_range(
        &self,
        start: usize,
        scan_size: usize,
        mut f: impl FnMut(&[Value], ProbeMatch<'rows, 'exec>),
    ) {
        let end = start
            .checked_add(scan_size)
            .expect("top index range overflow");
        match &self.ix {
            ProbeIndex::SmallColumn(index) => {
                assert!(end <= index.n_keys);
                for key_index in start..end {
                    let rows = if self.keep_rows {
                        ProbeMatch::Rows(AtomRows::inline(index.rows_at(key_index)))
                    } else {
                        ProbeMatch::Present
                    };
                    f(&index.keys[key_index..key_index + 1], rows);
                }
            }
            ProbeIndex::PackedColumn(node) => {
                let values = node.values();
                assert!(end <= values.len());
                for key_index in start..end {
                    let rows = AtomRows::packed(PackedCursor::new(node, key_index));
                    f(
                        &values[key_index..key_index + 1],
                        Self::keep_or_discard(rows, self.keep_rows),
                    );
                }
            }
            ProbeIndex::CachedTuple { .. } | ProbeIndex::CachedColumn { .. } => {
                unreachable!("persistent indexes use physical shard partitions")
            }
            ProbeIndex::SmallExact(..) | ProbeIndex::Packed(..) => {
                unreachable!("a scalar intersection cannot use an exact tuple probe")
            }
        }
    }

    pub(super) fn shard_count(&self) -> Option<usize> {
        match &self.ix {
            ProbeIndex::CachedTuple { table, .. } => Some(table.shard_count()),
            ProbeIndex::CachedColumn { table, .. } => Some(table.shard_count()),
            ProbeIndex::SmallColumn(..)
            | ProbeIndex::SmallExact(..)
            | ProbeIndex::Packed(..)
            | ProbeIndex::PackedColumn(..) => None,
        }
    }

    pub(super) fn shard_len(&self, shard: usize) -> Option<usize> {
        match &self.ix {
            ProbeIndex::CachedTuple { table, .. } => Some(table.shard_len(shard)),
            ProbeIndex::CachedColumn { table, .. } => Some(table.shard_len(shard)),
            ProbeIndex::SmallColumn(..)
            | ProbeIndex::SmallExact(..)
            | ProbeIndex::Packed(..)
            | ProbeIndex::PackedColumn(..) => None,
        }
    }

    pub(super) fn len(&self) -> usize {
        match &self.ix {
            ProbeIndex::CachedTuple { table, .. } => table.len(),
            ProbeIndex::CachedColumn { table, .. } => table.len(),
            ProbeIndex::SmallColumn(index) => index.len(),
            ProbeIndex::SmallExact(exact) => exact.len(),
            // Intersect stages are scalar. Tuple-packed probers are used only
            // for exact probes, so the first-level count is sufficient here.
            ProbeIndex::Packed(packed) => packed.first.values().len(),
            ProbeIndex::PackedColumn(node) => node.values().len(),
        }
    }
}

#[cfg(test)]
#[path = "probe_tests.rs"]
mod tests;
