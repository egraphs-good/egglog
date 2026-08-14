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
            let r = s.binary_search_by_id(range.end);
            if l >= r {
                None
            } else {
                Some(SubsetRef::Sparse(s.subslice(l, r)))
            }
        }
    }
}

/// The rows currently associated with an atom during one plan execution.
/// Roots retain ownership of their header-filtered subset. An indexed cursor
/// borrows a first-level group from either a prepared persistent index or a
/// shared round-local root index and carries a plan-local continuation slot.
/// Every lower cursor is just a packed node plus a key ordinal. Dense
/// singletons come from cover scans and are packed lazily if the atom is probed
/// again.
#[derive(Clone, Copy)]
pub(super) struct CatalogContinuation<'rows> {
    cache: &'rows RootContinuationCache,
    position: ContinuationPosition,
}

#[derive(Clone)]
pub(super) enum AtomRows<'rows, 'exec> {
    Root(Arc<TrieNode>),
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
        match self {
            Self::Root(_) => "root",
            Self::Catalog { .. } => "catalog",
            Self::Packed(_) => "packed",
            Self::Inline(_) => "inline",
            Self::Dense(_) => "dense",
        }
    }

    fn subset(&self) -> SubsetRef<'_> {
        match self {
            Self::Root(root) => root.subset.as_ref(),
            Self::Catalog { subset, .. } => *subset,
            Self::Packed(cursor) => cursor.subset(),
            Self::Inline(rows) => rows.subset(),
            Self::Dense(range) => SubsetRef::Dense(*range),
        }
    }

    fn size(&self) -> usize {
        match self {
            Self::Packed(cursor) => cursor.size(),
            _ => self.subset().size(),
        }
    }

    fn is_empty(&self) -> bool {
        self.size() == 0
    }

    fn is_root(&self) -> bool {
        matches!(self, Self::Root(_))
    }

    #[cfg(test)]
    fn root_arc(&self) -> &Arc<TrieNode> {
        let Self::Root(root) = self else {
            panic!("expected root rows")
        };
        root
    }
}

impl<'rows, 'exec> From<Arc<TrieNode>> for AtomRows<'rows, 'exec> {
    fn from(root: Arc<TrieNode>) -> Self {
        Self::Root(root)
    }
}

/// Physical strategies available for looking up or enumerating the current
/// rows of one atom. [`JoinState::get_index`] chooses one variant for each
/// logical scan based on the source subset, requested columns, and whether a
/// reusable table index is valid.
enum ProbeIndex<'ctx, 'rows, 'exec> {
    /// A persistent, fully refreshed multi-column table index. This is used for
    /// a large dense root with cacheable columns, no additional constraints,
    /// and no stale rows. `intersect_outer` clips results when the root is a
    /// dense subrange rather than the whole table.
    CachedTuple {
        intersect_outer: Option<OffsetRange>,
        table: &'rows Index<TupleIndex>,
        continuations: &'rows RootContinuationCache,
        child_shape: ChildShape,
    },
    /// The single-column counterpart of [`Self::CachedTuple`], selected under
    /// the same catalog-index conditions.
    CachedColumn {
        intersect_outer: Option<OffsetRange>,
        table: &'rows Index<ColumnIndex>,
        continuations: &'rows RootContinuationCache,
        child_shape: ChildShape,
    },
    /// A first-column index shared by plans that start from the same filtered
    /// root subset. This is used when a shared root exists but the persistent
    /// catalog fast path is invalid, for example because the root is sparse or
    /// the scan has additional constraints.
    ProjectedRoot(RootProjectionProbe<'ctx, 'rows, 'exec>),
    /// An inline scalar index for a source containing at most
    /// [`SMALL_RESIDUAL`] rows. It supports both exact lookup and enumeration
    /// without constructing a general packed trie.
    SmallColumn(SmallColumnIndex),
    /// An exact-only multi-column probe over an inline residual. Join stages
    /// select it when the source is already [`AtomRows::Inline`]; it scans those
    /// few rows directly and is never used as an enumeration leader.
    SmallExact(SmallExactProbe<'ctx>),
    /// The general fallback: an arena-allocated packed trie over an arbitrary
    /// source subset, with lower column indexes constructed lazily.
    Packed(PackedProbe<'ctx, 'rows, 'exec>),
}

/// A successful probe either carries rows needed by a later stage or only
/// records existence when this atom is dead in the remaining join tail. The
/// latter case avoids copying an inline subset into every buffered frame.
enum ProbeMatch<'rows, 'exec> {
    Present,
    Rows(AtomRows<'rows, 'exec>),
}

impl<'rows, 'exec> ProbeMatch<'rows, 'exec> {
    #[inline]
    fn refine(self, atom: AtomId, updates: &mut FrameUpdates<'rows, 'exec>) {
        if let Self::Rows(rows) = self {
            updates.refine_atom(atom, rows);
        }
    }
}

struct PackedProbe<'ctx, 'rows, 'exec> {
    first: &'rows PackedTrieNode<'exec>,
    columns: SmallVec<[ColumnId; 4]>,
    table: WrappedTableRef<'ctx>,
    handle: &'ctx Handle<'exec>,
    scratch: &'ctx RefCell<Vec<(Value, RowId)>>,
    terminal_child_shape: ChildShape,
}

/// Probes a first-column grouping shared by plans with the same root subset.
///
/// A root projection is an execution-local index that groups a root's rows by
/// one requested column and stores each distinct value beside its matching row
/// subset. [`JoinState::get_index`] selects this representation when the atom
/// has a cross-plan shared root but cannot use a persistent table index, such
/// as when header or scan constraints have filtered the root. The grouping is
/// built once and reused directly by every qualifying plan.
///
/// For a one-column scan, this object returns the shared row subset. For a
/// multi-column scan, only the first grouping is shared; subsequent packed
/// trie nodes and their continuation slots remain local to this query.
struct RootProjectionProbe<'ctx, 'rows, 'exec> {
    first: &'rows RootProjection,
    columns: SmallVec<[ColumnId; 4]>,
    table: WrappedTableRef<'ctx>,
    continuations: &'rows RootContinuationCache,
    access: AccessId,
    handle: &'ctx Handle<'exec>,
    scratch: &'ctx RefCell<Vec<(Value, RowId)>>,
    terminal_child_shape: ChildShape,
}

impl<'ctx, 'rows, 'exec> RootProjectionProbe<'ctx, 'rows, 'exec>
where
    'exec: 'rows,
{
    fn scalar_rows(&self, key_index: usize) -> AtomRows<'rows, 'exec> {
        debug_assert_eq!(self.columns.len(), 1);
        AtomRows::Catalog {
            subset: self.first.subset_at(key_index),
            continuation: (self.terminal_child_shape != ChildShape::Leaf).then_some(
                CatalogContinuation {
                    cache: self.continuations,
                    position: ContinuationPosition::unsharded(key_index),
                },
            ),
        }
    }

    fn first_child(&self, key_index: usize) -> &'exec PackedTrieNode<'exec> {
        debug_assert!(self.columns.len() > 1);
        let child_shape = if self.columns.len() > 2 {
            ChildShape::Direct
        } else {
            self.terminal_child_shape
        };
        let slot = self
            .continuations
            .slot(ContinuationPosition::unsharded(key_index), self.access);
        let address = *slot.get_or_init(|| {
            PackedTrieNode::build_from_subset(
                self.handle,
                self.table,
                self.first.subset_at(key_index),
                self.columns[1],
                child_shape,
                &mut self.scratch.borrow_mut(),
            ) as *const PackedTrieNode<'exec> as usize
        });
        // SAFETY: this continuation cache belongs to the prepared query and can
        // only publish nodes allocated by the same query's SharedArena.
        let child = unsafe { &*(address as *const PackedTrieNode<'exec>) };
        assert_eq!(child.child_shape(), child_shape);
        child
    }

    fn get(&self, key: &[Value]) -> Option<AtomRows<'rows, 'exec>> {
        if key.len() != self.columns.len() {
            return None;
        }
        let first_index = self.first.find(key[0])?;
        if self.columns.len() == 1 {
            return Some(self.scalar_rows(first_index));
        }

        let mut node = self.first_child(first_index);
        let mut terminal = None;
        for (depth, &value) in key.iter().enumerate().skip(1) {
            let cursor = PackedCursor::new(node, node.find(value)?);
            terminal = Some(cursor);
            if depth + 1 < self.columns.len() {
                let child_shape = if depth + 2 < self.columns.len() {
                    ChildShape::Direct
                } else {
                    self.terminal_child_shape
                };
                node = cursor.child_index(
                    self.handle,
                    self.table,
                    self.columns[depth + 1],
                    0,
                    child_shape,
                    &mut self.scratch.borrow_mut(),
                );
            }
        }
        terminal.map(AtomRows::Packed)
    }

    fn for_each_packed(
        &self,
        node: &'rows PackedTrieNode<'exec>,
        depth: usize,
        key: &mut SmallVec<[Value; 4]>,
        f: &mut impl FnMut(&[Value], AtomRows<'rows, 'exec>),
    ) {
        for (key_index, &value) in node.values().iter().enumerate() {
            key.push(value);
            let cursor = PackedCursor::new(node, key_index);
            if depth + 1 == self.columns.len() {
                f(key, AtomRows::Packed(cursor));
            } else {
                let child_shape = if depth + 2 < self.columns.len() {
                    ChildShape::Direct
                } else {
                    self.terminal_child_shape
                };
                let child = cursor.child_index(
                    self.handle,
                    self.table,
                    self.columns[depth + 1],
                    0,
                    child_shape,
                    &mut self.scratch.borrow_mut(),
                );
                self.for_each_packed(child, depth + 1, key, f);
            }
            key.pop();
        }
    }

    fn for_each(&self, f: &mut impl FnMut(&[Value], AtomRows<'rows, 'exec>)) {
        let mut key = SmallVec::new();
        for key_index in 0..self.first.len() {
            key.push(self.first.value_at(key_index));
            if self.columns.len() == 1 {
                f(&key, self.scalar_rows(key_index));
            } else {
                let child = self.first_child(key_index);
                self.for_each_packed(child, 1, &mut key, f);
            }
            key.pop();
        }
    }
}

impl<'ctx, 'rows, 'exec> PackedProbe<'ctx, 'rows, 'exec>
where
    'exec: 'rows,
{
    fn get(&self, key: &[Value]) -> Option<AtomRows<'rows, 'exec>> {
        if key.len() != self.columns.len() {
            return None;
        }
        let mut node = self.first;
        let mut terminal = None;
        for (depth, (&_column, &value)) in self.columns.iter().zip(key).enumerate() {
            let cursor = PackedCursor::new(node, node.find(value)?);
            terminal = Some(cursor);
            if depth + 1 < self.columns.len() {
                let child_shape = if depth + 2 < self.columns.len() {
                    ChildShape::Direct
                } else {
                    self.terminal_child_shape
                };
                node = cursor.child_index(
                    self.handle,
                    self.table,
                    self.columns[depth + 1],
                    0,
                    child_shape,
                    &mut self.scratch.borrow_mut(),
                );
            }
        }
        terminal.map(AtomRows::Packed)
    }

    fn for_each_recur(
        &self,
        node: &'rows PackedTrieNode<'exec>,
        depth: usize,
        key: &mut SmallVec<[Value; 4]>,
        f: &mut impl FnMut(&[Value], AtomRows<'rows, 'exec>),
    ) {
        for (key_index, &value) in node.values().iter().enumerate() {
            key.push(value);
            let cursor = PackedCursor::new(node, key_index);
            if depth + 1 == self.columns.len() {
                f(key, AtomRows::Packed(cursor));
            } else {
                let child_shape = if depth + 2 < self.columns.len() {
                    ChildShape::Direct
                } else {
                    self.terminal_child_shape
                };
                let child = cursor.child_index(
                    self.handle,
                    self.table,
                    self.columns[depth + 1],
                    0,
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

struct Prober<'ctx, 'rows, 'exec> {
    source: AtomRows<'rows, 'exec>,
    ix: ProbeIndex<'ctx, 'rows, 'exec>,
    keep_rows: bool,
}

/// Normalized input to [`JoinState::get_index`] for one indexed scan.
///
/// Both scalar [`SingleScanSpec`]s and tuple [`ScanSpec`]s are converted to
/// this form so physical-strategy selection has one code path. It identifies
/// the atom and projected columns, carries constraints that must be applied to
/// the source rows, records whether later stages need the matching rows or only
/// existence, and supplies the prepared sidecar slot used for cached indexes
/// and continuations. `terminal_child_shape` describes how execution may
/// continue after the final requested column.
struct ProbeRequest<'scan, 'rows> {
    atom: AtomId,
    columns: SmallVec<[ColumnId; 4]>,
    constraints: &'scan [Constraint],
    keep_rows: bool,
    terminal_child_shape: ChildShape,
    prepared: &'rows PreparedIndexSlot,
}

impl<'scan, 'rows> ProbeRequest<'scan, 'rows> {
    fn column(
        scan: &'scan SingleScanSpec,
        keep_rows: bool,
        terminal_child_shape: ChildShape,
        prepared: &'rows PreparedIndexSlot,
    ) -> Self {
        Self {
            atom: scan.atom,
            columns: SmallVec::from_slice(&[scan.column]),
            constraints: &scan.cs,
            keep_rows,
            terminal_child_shape,
            prepared,
        }
    }

    fn tuple(
        scan: &'scan ScanSpec,
        keep_rows: bool,
        terminal_child_shape: ChildShape,
        prepared: &'rows PreparedIndexSlot,
    ) -> Self {
        Self {
            atom: scan.to_index.atom,
            columns: scan.to_index.vars.iter().copied().collect(),
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

    fn catalog_match(
        position: IndexPosition,
        subset: SubsetRef<'rows>,
        continuations: &'rows RootContinuationCache,
        keep_rows: bool,
        child_shape: ChildShape,
    ) -> ProbeMatch<'rows, 'exec> {
        if keep_rows {
            ProbeMatch::Rows(AtomRows::Catalog {
                subset,
                continuation: (child_shape != ChildShape::Leaf).then_some(CatalogContinuation {
                    cache: continuations,
                    position: position.into(),
                }),
            })
        } else {
            ProbeMatch::Present
        }
    }

    fn get_subset(&self, key: &[Value]) -> Option<ProbeMatch<'rows, 'exec>> {
        match &self.ix {
            ProbeIndex::CachedTuple {
                intersect_outer,
                table,
                continuations,
                child_shape,
            } => {
                let table: &'rows Index<TupleIndex> = table;
                let (position, subset) = table.get_subset_positioned(key)?;
                let subset = if let Some(range) = intersect_outer {
                    intersect_with_dense_ref(subset, *range)?
                } else {
                    subset
                };
                Some(Self::catalog_match(
                    position,
                    subset,
                    continuations,
                    self.keep_rows,
                    *child_shape,
                ))
            }
            ProbeIndex::CachedColumn {
                intersect_outer,
                table,
                continuations,
                child_shape,
            } => {
                debug_assert_eq!(key.len(), 1);
                let table: &'rows Index<ColumnIndex> = table;
                let (position, subset) = table.get_subset_positioned(&key[0])?;
                let subset = if let Some(range) = intersect_outer {
                    intersect_with_dense_ref(subset, *range)?
                } else {
                    subset
                };
                Some(Self::catalog_match(
                    position,
                    subset,
                    continuations,
                    self.keep_rows,
                    *child_shape,
                ))
            }
            ProbeIndex::ProjectedRoot(projected) => projected
                .get(key)
                .map(|rows| Self::keep_or_discard(rows, self.keep_rows)),
            ProbeIndex::SmallColumn(index) => {
                let [value] = key else {
                    return None;
                };
                let key_index = index.find(*value)?;
                Some(if self.keep_rows {
                    ProbeMatch::Rows(AtomRows::Inline(index.rows_at(key_index)))
                } else {
                    ProbeMatch::Present
                })
            }
            ProbeIndex::SmallExact(exact) => exact.get(key, self.keep_rows),
            ProbeIndex::Packed(packed) => packed
                .get(key)
                .map(|rows| Self::keep_or_discard(rows, self.keep_rows)),
        }
    }

    fn for_each(&self, mut f: impl FnMut(&[Value], ProbeMatch<'rows, 'exec>)) {
        match &self.ix {
            ProbeIndex::CachedTuple {
                intersect_outer,
                table,
                continuations,
                child_shape,
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
                    f(
                        key,
                        Self::catalog_match(
                            position,
                            subset,
                            continuations,
                            self.keep_rows,
                            *child_shape,
                        ),
                    );
                });
            }
            ProbeIndex::CachedColumn {
                intersect_outer,
                table,
                continuations,
                child_shape,
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
                    f(
                        &[*value],
                        Self::catalog_match(
                            position,
                            subset,
                            continuations,
                            self.keep_rows,
                            *child_shape,
                        ),
                    );
                });
            }
            ProbeIndex::ProjectedRoot(projected) => projected.for_each(&mut |key, rows| {
                f(key, Self::keep_or_discard(rows, self.keep_rows));
            }),
            ProbeIndex::SmallColumn(index) => {
                for key_index in 0..index.n_keys {
                    let rows = if self.keep_rows {
                        ProbeMatch::Rows(AtomRows::Inline(index.rows_at(key_index)))
                    } else {
                        ProbeMatch::Present
                    };
                    f(&index.keys[key_index..key_index + 1], rows);
                }
            }
            ProbeIndex::SmallExact(..) => {
                unreachable!("small multi-column residuals are exact-probe only")
            }
            ProbeIndex::Packed(packed) => packed.for_each(&mut |key, rows| {
                f(key, Self::keep_or_discard(rows, self.keep_rows));
            }),
        }
    }

    fn for_each_shard(&self, shard: usize, mut f: impl FnMut(&[Value], ProbeMatch<'rows, 'exec>)) {
        match &self.ix {
            ProbeIndex::CachedTuple {
                intersect_outer,
                table,
                continuations,
                child_shape,
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
                    f(
                        key,
                        Self::catalog_match(
                            position,
                            subset,
                            continuations,
                            self.keep_rows,
                            *child_shape,
                        ),
                    );
                });
            }
            ProbeIndex::CachedColumn {
                intersect_outer,
                table,
                continuations,
                child_shape,
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
                    f(
                        &[*value],
                        Self::catalog_match(
                            position,
                            subset,
                            continuations,
                            self.keep_rows,
                            *child_shape,
                        ),
                    );
                });
            }
            ProbeIndex::ProjectedRoot(..)
            | ProbeIndex::SmallColumn(..)
            | ProbeIndex::SmallExact(..)
            | ProbeIndex::Packed(..) => {
                unreachable!("only persistent root indexes expose physical shards")
            }
        }
    }

    fn shard_count(&self) -> Option<usize> {
        match &self.ix {
            ProbeIndex::CachedTuple { table, .. } => Some(table.shard_count()),
            ProbeIndex::CachedColumn { table, .. } => Some(table.shard_count()),
            ProbeIndex::ProjectedRoot(..)
            | ProbeIndex::SmallColumn(..)
            | ProbeIndex::SmallExact(..)
            | ProbeIndex::Packed(..) => None,
        }
    }

    fn shard_len(&self, shard: usize) -> Option<usize> {
        match &self.ix {
            ProbeIndex::CachedTuple {
                intersect_outer: None,
                table,
                ..
            } => Some(table.shard_len(shard)),
            ProbeIndex::CachedColumn {
                intersect_outer: None,
                table,
                ..
            } => Some(table.shard_len(shard)),
            ProbeIndex::CachedTuple {
                intersect_outer: Some(_),
                ..
            }
            | ProbeIndex::CachedColumn {
                intersect_outer: Some(_),
                ..
            }
            | ProbeIndex::ProjectedRoot(..)
            | ProbeIndex::SmallColumn(..)
            | ProbeIndex::SmallExact(..)
            | ProbeIndex::Packed(..) => None,
        }
    }

    fn len(&self) -> usize {
        match &self.ix {
            ProbeIndex::CachedTuple { table, .. } => table.len(),
            ProbeIndex::CachedColumn { table, .. } => table.len(),
            ProbeIndex::ProjectedRoot(projected) => projected.first.len(),
            ProbeIndex::SmallColumn(index) => index.len(),
            ProbeIndex::SmallExact(exact) => exact.len(),
            // Intersect stages are scalar. Tuple-packed probers are used only
            // for exact probes, so the first-level count is sufficient here.
            ProbeIndex::Packed(packed) => packed.first.values().len(),
        }
    }
}
