//! Apply value-level rebuilds to a table.

use std::{cmp, mem};

use crate::numeric_id::NumericId;

use crate::{
    ColumnId, ExecutionState, Offset, OffsetRange, RowId, Subset, SubsetRef, Table, TableId,
    TaggedRowBuffer, Value, WrappedTable,
    common::HashSet,
    hash_index::{ColumnIndex, Index},
    offsets::Offsets,
    parallel,
    parallel_heuristics::{parallelize_incremental_rebuild, parallelize_rebuild},
    table_spec::{Rebuilder, WrappedTableRef},
};

use super::SortedWritesTable;

// Helper macro used for adjusting sort before inserting to a mutation buffer.
macro_rules! insert_row {
    ($this: expr, $mutation_buf: expr, $row:expr, $next_ts:expr) => {{
        let row = $row;
        let this = &*$this;
        let next_ts = $next_ts;
        if let Some(sort_by) = this.sort_by {
            row[sort_by.index()] = next_ts;
        }
        $mutation_buf.stage_insert(row);
    }};
}

impl SortedWritesTable {
    fn refresh_rebuild_index(&mut self) {
        let mut index = mem::replace(
            &mut self.rebuild_index,
            Index::new(vec![], ColumnIndex::new()),
        );
        WrappedTableRef::with_wrapper(self, |wrapped| {
            index.refresh(wrapped);
        });
        self.rebuild_index = index;
    }

    pub(super) fn do_rebuild(
        &mut self,
        table_id: TableId,
        table: &WrappedTable,
        next_ts: Value,
        exec_state: &mut ExecutionState,
    ) -> bool {
        if self.to_rebuild.is_empty() {
            return false;
        }
        let Some(rebuilder) = table.rebuilder(&self.to_rebuild) else {
            return false;
        };
        // First, decide whether to do an incremental or full rebuild.
        if let Some(hint_col) = rebuilder.hint_col() {
            // Incremental rebuilds are possible if we can scan the subset of the columns that are
            // relevant.
            let to_scan = self.subset_tracker.recent_updates(table_id, table);
            if incremental_rebuild(
                to_scan.size(),
                self.data.next_row().index(),
                parallelize_rebuild(to_scan.size()),
            ) {
                self.rebuild_incremental(table, &*rebuilder, hint_col, to_scan, next_ts, exec_state)
            } else {
                self.rebuild_nonincremental(&*rebuilder, next_ts, exec_state)
            }
        } else {
            self.rebuild_nonincremental(&*rebuilder, next_ts, exec_state)
        }
    }

    pub(super) fn refresh_rows_for_values(&mut self, dirty_ids: &[Value], next_ts: Value) -> bool {
        if dirty_ids.is_empty() || self.to_rebuild.is_empty() {
            return false;
        }
        // Reuse the rebuild index to find rows whose rebuildable columns mention
        // one of the same-id dirty container ids.
        self.refresh_rebuild_index();

        let mut candidate_rows = HashSet::<RowId>::default();
        for value in dirty_ids {
            let Some(subset) = self.rebuild_index.get_subset(value) else {
                continue;
            };
            subset.offsets(|row_id| {
                candidate_rows.insert(row_id);
            });
        }

        if candidate_rows.is_empty() {
            return false;
        }

        let mut changed = false;
        let mut mutation_buf = self.new_buffer();
        let mut refreshed_row = Vec::<Value>::with_capacity(self.n_columns);
        for row_id in candidate_rows {
            let Some(current_row) = self.data.get_row(row_id) else {
                continue;
            };
            // Preserve the logical row and only advance its sort/timestamp
            // column, so seminaive treats this as a fresh parent-row delta.
            mutation_buf.stage_remove(&current_row[0..self.n_keys]);
            refreshed_row.clear();
            refreshed_row.extend_from_slice(current_row);
            if let Some(sort_by) = self.sort_by {
                refreshed_row[sort_by.index()] = next_ts;
            }
            mutation_buf.stage_insert(&refreshed_row);
            changed = true;
        }
        changed
    }

    fn rebuild_incremental(
        &mut self,
        table: &WrappedTable,
        rebuilder: &dyn Rebuilder,
        search_col: ColumnId,
        to_scan: Subset,
        next_ts: Value,
        exec_state: &mut ExecutionState,
    ) -> bool {
        self.refresh_rebuild_index();

        if parallel::current_num_threads() >= MIN_COARSE_REBUILD_THREADS
            && parallelize_incremental_rebuild(to_scan.size())
        {
            WrappedTableRef::with_wrapper(self, |wrapped| {
                let source = table.as_ref();
                let subset = to_scan.as_ref();
                let partition_size = rebuild_partition_size(
                    subset.size(),
                    parallel::current_num_threads(),
                    MIN_REBUILD_PARTITION_ROWS,
                );
                let starts = (0..subset.size())
                    .step_by(partition_size)
                    .collect::<Vec<_>>();
                parallel::map(&starts, |_, start| {
                    let partition = subset_partition(
                        subset,
                        *start,
                        cmp::min(*start + partition_size, subset.size()),
                    );
                    let mut mutation_buf = self.new_buffer();
                    let mut exec_state = exec_state.clone();
                    let mut changed = false;
                    let mut scanned = TaggedRowBuffer::new(self.n_columns);
                    source.for_each_col(partition, search_col, &mut |_, id| {
                        let Some(rows) = self.rebuild_index.get_subset(&id) else {
                            return;
                        };
                        rebuilder.rebuild_subset(wrapped, rows, &mut scanned, &mut exec_state);
                        for (row_id, row) in scanned.non_stale_mut() {
                            let to_remove = self.data.get_row(row_id).map(|x| &x[0..self.n_keys]);
                            if let Some(key) = to_remove {
                                mutation_buf.stage_remove(key);
                            }
                            changed = true;
                            insert_row!(self, mutation_buf, row, next_ts);
                        }
                        scanned.clear();
                    });
                    changed
                })
                .into_iter()
                .any(|changed| changed)
            })
        } else {
            let mut ids = TaggedRowBuffer::new(1);
            table.scan_project(
                to_scan.as_ref(),
                &[search_col],
                Offset::new(0),
                usize::MAX,
                &[],
                &mut ids,
            );
            let mut scratch = TaggedRowBuffer::new(self.n_columns);
            let mut changed = false;
            for (_, id) in ids.iter() {
                let Some(subset) = self.rebuild_index.get_subset(&id[0]) else {
                    continue;
                };
                WrappedTableRef::with_wrapper(self, |wrapped| {
                    rebuilder.rebuild_subset(wrapped, subset, &mut scratch, exec_state);
                });
                changed |= subset.size() > 0;
            }
            if !scratch.is_empty() {
                let mut write_buf = self.new_buffer();
                for (row_id, row) in scratch.non_stale_mut() {
                    if let Some(to_remove) = self.data.get_row(row_id).map(|x| &x[0..self.n_keys]) {
                        write_buf.stage_remove(to_remove);
                    }
                    insert_row!(self, write_buf, row, next_ts);
                }
            }
            changed
        }
    }

    fn rebuild_nonincremental(
        &mut self,
        rebuilder: &dyn Rebuilder,
        next_ts: Value,
        exec_state: &mut ExecutionState,
    ) -> bool {
        const STEP_SIZE: usize = 2048;
        let max_row = self.data.next_row().index();
        if parallelize_rebuild(max_row) {
            let workers = parallel::current_num_threads();
            if workers < MIN_COARSE_REBUILD_THREADS {
                // Keep the original fine-grained loop at low thread counts. Reusing
                // state across a coarse partition creates larger mutation-buffer
                // publications, whose merge cost outweighs the saved setup here.
                let starts = (0..max_row).step_by(STEP_SIZE).collect::<Vec<_>>();
                return parallel::map(&starts, |_, start| {
                    let mut mutation_buf = self.new_buffer();
                    let mut buf = TaggedRowBuffer::new(self.n_columns);
                    let mut exec_state = exec_state.clone();
                    let mut changed = false;
                    rebuilder.rebuild_buf(
                        &self.data.data,
                        RowId::from_usize(*start),
                        RowId::from_usize(cmp::min(*start + STEP_SIZE, max_row)),
                        &mut buf,
                        &mut exec_state,
                    );
                    for (row_id, row) in buf.non_stale_mut() {
                        let to_remove = self.data.get_row(row_id).map(|x| &x[0..self.n_keys]);
                        changed = true;
                        if let Some(key) = to_remove {
                            mutation_buf.stage_remove(key);
                        }
                        insert_row!(self, mutation_buf, row, next_ts);
                    }
                    buf.clear();
                    changed
                })
                .into_iter()
                .any(|changed| changed);
            }

            let partition_size = rebuild_partition_size(max_row, workers, STEP_SIZE);
            let starts = (0..max_row).step_by(partition_size).collect::<Vec<_>>();
            parallel::map(&starts, |_, start| {
                let partition_end = cmp::min(*start + partition_size, max_row);
                let mut mutation_buf = self.new_buffer();
                let mut buf = TaggedRowBuffer::new(self.n_columns);
                let mut exec_state = exec_state.clone();
                let mut changed = false;
                for chunk_start in (*start..partition_end).step_by(STEP_SIZE) {
                    rebuilder.rebuild_buf(
                        &self.data.data,
                        RowId::from_usize(chunk_start),
                        RowId::from_usize(cmp::min(chunk_start + STEP_SIZE, partition_end)),
                        &mut buf,
                        &mut exec_state,
                    );
                    for (row_id, row) in buf.non_stale_mut() {
                        let to_remove = self.data.get_row(row_id).map(|x| &x[0..self.n_keys]);
                        changed = true;
                        if let Some(key) = to_remove {
                            mutation_buf.stage_remove(key);
                        }
                        insert_row!(self, mutation_buf, row, next_ts);
                    }
                    buf.clear();
                }
                changed
            })
            .into_iter()
            .any(|changed| changed)
        } else {
            let mut buf = TaggedRowBuffer::new(self.n_columns);
            let mut changed = false;

            for start in (0..max_row).step_by(STEP_SIZE) {
                rebuilder.rebuild_buf(
                    &self.data.data,
                    RowId::from_usize(start),
                    RowId::from_usize(cmp::min(start + STEP_SIZE, max_row)),
                    &mut buf,
                    exec_state,
                );
            }
            if !buf.is_empty() {
                let mut write_buf = self.new_buffer();
                for (row_id, row) in buf.non_stale_mut() {
                    if let Some(to_remove) = self.data.get_row(row_id).map(|x| &x[0..self.n_keys]) {
                        write_buf.stage_remove(to_remove);
                    }
                    insert_row!(self, write_buf, row, next_ts);
                    changed = true;
                }
            }
            changed
        }
    }
}

const MIN_REBUILD_PARTITION_ROWS: usize = 256;
const MIN_COARSE_REBUILD_THREADS: usize = 4;

fn rebuild_partition_size(scan_size: usize, workers: usize, minimum: usize) -> usize {
    scan_size.div_ceil(workers.max(1)).max(minimum)
}

fn subset_partition<'a>(subset: SubsetRef<'a>, start: usize, end: usize) -> SubsetRef<'a> {
    debug_assert!(start <= end);
    debug_assert!(end <= subset.size());
    match subset {
        SubsetRef::Dense(range) => SubsetRef::Dense(OffsetRange::new(
            RowId::from_usize(range.start.index() + start),
            RowId::from_usize(range.start.index() + end),
        )),
        SubsetRef::Sparse(rows) => SubsetRef::Sparse(rows.subslice(start, end)),
    }
}

fn incremental_rebuild(uf_size: usize, table_size: usize, parallel: bool) -> bool {
    if parallel {
        table_size > 10_000 && uf_size * 8192 <= table_size
    } else {
        table_size > 10000 && uf_size * 8 <= table_size
    }
}

#[cfg(test)]
mod tests {
    use super::rebuild_partition_size;

    #[test]
    fn rebuild_partitions_are_coarse_and_nonzero() {
        assert_eq!(rebuild_partition_size(0, 0, 256), 256);
        assert_eq!(rebuild_partition_size(1_000, 4, 256), 256);
        assert_eq!(rebuild_partition_size(4_096, 3, 256), 1_366);
    }
}
