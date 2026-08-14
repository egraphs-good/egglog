use crate::numeric_id::define_id;

const SMALL_RESIDUAL: usize = 8;

define_id!(
    AccessId,
    u32,
    r#"Dense identity of one indexed access to an atom within a single
[`JoinStages`] block. Access ids are dense and local to each atom.

For example, the query `T(x, y, z), X(x), Y(y), Z(z)` may produce this
simplified stage sequence:

- intersect `x` through `T[0]` and `X[0]`;
- intersect `y` through `T[1]` and `Y[0]`;
- intersect `z` through `T[2]` and `Z[0]`.

The three accesses to the `T` atom receive ids 0, 1, and 2. Each unary atom
has its own access-id namespace and therefore receives id 0. A dynamic packed
node uses these ids to distinguish the successor families that DVO may choose
after reaching that atom."#
);

/// An owned row subset small enough to travel inline with a buffered join
/// frame. Rows are kept sorted so [`Self::subset`] can expose a borrowed
/// `SubsetRef` without allocating or involving the execution arena.
///
/// The direct array representation is intentional: `tinyvec::ArrayVec`
/// requires its elements to implement `Default`, while numeric ids such as
/// [`RowId`] deliberately have no distinguished default value.
#[derive(Clone, Copy)]
pub(super) struct InlineRows {
    len: u32,
    rows: [RowId; SMALL_RESIDUAL],
}

impl InlineRows {
    fn from_sorted(rows: &[RowId]) -> Self {
        assert!(
            !rows.is_empty() && rows.len() <= SMALL_RESIDUAL,
            "inline row subsets must contain 1..={SMALL_RESIDUAL} rows"
        );
        debug_assert!(rows.windows(2).all(|pair| pair[0] <= pair[1]));
        let mut inline = Self {
            len: rows.len() as u32,
            rows: [RowId::new_const(0); SMALL_RESIDUAL],
        };
        inline.rows[..rows.len()].copy_from_slice(rows);
        inline
    }

    #[inline]
    fn rows(&self) -> &[RowId] {
        &self.rows[..self.len as usize]
    }

    #[inline]
    fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    fn subset(&self) -> SubsetRef<'_> {
        // SAFETY: `from_sorted` is the only constructor and records a sorted,
        // nonempty prefix. Copies preserve that invariant.
        SubsetRef::Sparse(unsafe { SortedOffsetSlice::new_unchecked(self.rows()) })
    }
}

struct SmallRowIdSink {
    len: usize,
    rows: [RowId; SMALL_RESIDUAL],
}

impl Default for SmallRowIdSink {
    fn default() -> Self {
        Self {
            len: 0,
            rows: [RowId::new_const(0); SMALL_RESIDUAL],
        }
    }
}

impl SmallRowIdSink {
    fn push(&mut self, row_id: RowId) {
        assert!(
            self.len < SMALL_RESIDUAL,
            "small row scan exceeded its source subset size"
        );
        self.rows[self.len] = row_id;
        self.len += 1;
    }

    fn into_inline(mut self) -> Option<InlineRows> {
        if self.len == 0 {
            return None;
        }
        self.rows[..self.len].sort_unstable();
        Some(InlineRows::from_sorted(&self.rows[..self.len]))
    }
}

impl RowSink for SmallRowIdSink {
    fn add_row(&mut self, row_id: RowId, _row: &[Value]) {
        self.push(row_id);
    }
}

struct SmallExactSink<'key> {
    key: &'key [Value],
    matches: SmallRowIdSink,
}

impl RowSink for SmallExactSink<'_> {
    fn add_row(&mut self, row_id: RowId, row: &[Value]) {
        if row == self.key {
            self.matches.push(row_id);
        }
    }
}

/// Fixed-capacity sink used to project a constrained single column. The source
/// subset has already been proven to contain at most `SMALL_RESIDUAL` rows, so
/// the object-safe table scan cannot overflow this buffer.
struct SmallColumnSink {
    len: usize,
    rows: [(Value, RowId); SMALL_RESIDUAL],
}

impl Default for SmallColumnSink {
    fn default() -> Self {
        Self {
            len: 0,
            rows: [(Value::new_const(0), RowId::new_const(0)); SMALL_RESIDUAL],
        }
    }
}

impl RowSink for SmallColumnSink {
    fn add_row(&mut self, row_id: RowId, row: &[Value]) {
        let [value] = row else {
            unreachable!("a small column scan projects exactly one value")
        };
        assert!(
            self.len < SMALL_RESIDUAL,
            "small column scan exceeded its source subset size"
        );
        self.rows[self.len] = (*value, row_id);
        self.len += 1;
    }
}

/// A stack-owned index for a single column of a residual with at most eight
/// rows. Key groups point into `row_ids`; both arrays are sorted and require no
/// pool, Arc, or arena allocation.
struct SmallColumnIndex {
    n_keys: usize,
    n_rows: usize,
    keys: [Value; SMALL_RESIDUAL],
    offsets: [usize; SMALL_RESIDUAL],
    row_ids: [RowId; SMALL_RESIDUAL],
}

impl SmallColumnIndex {
    fn new(
        table: WrappedTableRef<'_>,
        subset: SubsetRef<'_>,
        constraints: &[Constraint],
        column: ColumnId,
    ) -> Self {
        debug_assert!(subset.size() <= SMALL_RESIDUAL);
        let mut sink = SmallColumnSink::default();
        let next = table.scan_project(
            subset,
            std::slice::from_ref(&column),
            Offset::new(0),
            usize::MAX,
            constraints,
            &mut sink,
        );
        debug_assert!(next.is_none());
        Self::from_projected(sink)
    }

    fn from_projected(mut sink: SmallColumnSink) -> Self {
        sink.rows[..sink.len].sort_unstable();

        let mut index = Self {
            n_keys: 0,
            n_rows: sink.len,
            keys: [Value::new_const(0); SMALL_RESIDUAL],
            offsets: [0; SMALL_RESIDUAL],
            row_ids: [RowId::new_const(0); SMALL_RESIDUAL],
        };
        for (position, &(value, row_id)) in sink.rows[..sink.len].iter().enumerate() {
            if index.n_keys == 0 || index.keys[index.n_keys - 1] != value {
                index.keys[index.n_keys] = value;
                index.offsets[index.n_keys] = position;
                index.n_keys += 1;
            }
            index.row_ids[position] = row_id;
        }
        index
    }

    #[inline]
    fn range(&self, key_index: usize) -> Range<usize> {
        let start = self.offsets[key_index];
        let end = if key_index + 1 < self.n_keys {
            self.offsets[key_index + 1]
        } else {
            self.n_rows
        };
        start..end
    }

    #[inline]
    fn find(&self, value: Value) -> Option<usize> {
        self.keys[..self.n_keys].binary_search(&value).ok()
    }

    #[inline]
    fn rows_at(&self, key_index: usize) -> InlineRows {
        InlineRows::from_sorted(&self.row_ids[self.range(key_index)])
    }

    #[inline]
    fn len(&self) -> usize {
        self.n_keys
    }
}

/// Allocation-free exact probing for an inline residual and a multi-column
/// key. Tuple residuals are only used for exact probes, so scanning at most
/// eight rows is cheaper and simpler than constructing a packed trie node.
struct SmallExactProbe<'ctx> {
    rows: Option<InlineRows>,
    columns: SmallVec<[ColumnId; 4]>,
    table: WrappedTableRef<'ctx>,
}

impl<'ctx> SmallExactProbe<'ctx> {
    fn new(
        table: WrappedTableRef<'ctx>,
        rows: InlineRows,
        columns: SmallVec<[ColumnId; 4]>,
        constraints: &[Constraint],
    ) -> Self {
        let rows = if constraints.is_empty() {
            Some(rows)
        } else {
            let mut sink = SmallRowIdSink::default();
            let next = table.scan_project(
                rows.subset(),
                std::slice::from_ref(&columns[0]),
                Offset::new(0),
                usize::MAX,
                constraints,
                &mut sink,
            );
            debug_assert!(next.is_none());
            sink.into_inline()
        };
        Self {
            rows,
            columns,
            table,
        }
    }

    fn get<'rows, 'exec>(
        &self,
        key: &[Value],
        keep_rows: bool,
    ) -> Option<ProbeMatch<'rows, 'exec>> {
        if key.len() != self.columns.len() {
            return None;
        }
        let rows = self.rows?;
        let mut sink = SmallExactSink {
            key,
            matches: SmallRowIdSink::default(),
        };
        let next = self.table.scan_project(
            rows.subset(),
            &self.columns,
            Offset::new(0),
            usize::MAX,
            &[],
            &mut sink,
        );
        debug_assert!(next.is_none());
        let matched = sink.matches.into_inline()?;
        Some(if keep_rows {
            ProbeMatch::Rows(AtomRows::Inline(matched))
        } else {
            ProbeMatch::Present
        })
    }

    fn len(&self) -> usize {
        self.rows.map_or(0, |rows| rows.len())
    }
}

fn top_index_shape_is_eligible(
    workers: usize,
    leader_keys: usize,
    nonempty_shards: usize,
    min_keys_per_worker: usize,
) -> bool {
    workers > 1
        && leader_keys >= min_keys_per_worker.saturating_mul(workers)
        && nonempty_shards >= workers
}
