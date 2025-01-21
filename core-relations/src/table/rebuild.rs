//! Apply value-level rewrites to a table.

use std::{cmp, mem};

use crossbeam_queue::SegQueue;
use numeric_id::{IdVec, NumericId};
use rayon::prelude::*;

use crate::{
    common::ShardId,
    hash_index::{ColumnIndex, Index},
    table_spec::{Rewriter, WrappedTableRef},
    ColumnId, ExecutionState, Offset, RowId, Subset, Table, TableId, TaggedRowBuffer, Value,
    WrappedTable,
};

use super::{hash_code, SortedWritesTable};

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
    pub(super) fn do_rewrite(
        &mut self,
        table_id: TableId,
        table: &WrappedTable,
        next_ts: Value,
        exec_state: &mut ExecutionState,
    ) {
        if self.to_rebuild.is_empty() {
            return;
        }
        let Some(rewrite) = table.rewriter(&self.to_rebuild) else {
            return;
        };
        // First, decide whether to do an incremental or full rewrite.
        if let Some(hint_col) = rewrite.hint_col() {
            // Incremental rewrites are possible if we can scan the subset of the columns that are
            // relevant.
            let to_scan = self.subset_tracker.recent_updates(table_id, table);
            if incremental_rebuild(
                to_scan.size(),
                self.data.next_row().index(),
                do_parallel(to_scan.size()),
            ) {
                self.rewrite_incremental(table, &*rewrite, hint_col, to_scan, next_ts, exec_state);
            } else {
                self.rewrite_nonincremental(&*rewrite, next_ts, exec_state);
            }
        } else {
            self.rewrite_nonincremental(&*rewrite, next_ts, exec_state);
        }
    }

    fn rewrite_incremental(
        &mut self,
        table: &WrappedTable,
        rewriter: &dyn Rewriter,
        search_col: ColumnId,
        to_scan: Subset,
        next_ts: Value,
        exec_state: &mut ExecutionState,
    ) {
        let mut index = mem::replace(
            &mut self.rebuild_index,
            Index::new(vec![], ColumnIndex::new()),
        );
        // Update the index.
        WrappedTableRef::with_wrapper(self, |wrapped| {
            index.refresh(wrapped);
        });
        self.rebuild_index = index;
        let mut buf = TaggedRowBuffer::new(1);
        table.scan_project(
            to_scan.as_ref(),
            &[search_col],
            Offset::new(0),
            usize::MAX,
            &[],
            &mut buf,
        );

        if do_parallel(to_scan.size()) {
            // Iterate over `buf` in parallel and then fan out to a per-shard set of rows then
            // process each shard in parallel.
            let mut queues = IdVec::<ShardId, SegQueue<TaggedRowBuffer>>::default();
            let shard_data = self.hash.shard_data();
            queues.resize_with(shard_data.n_shards(), SegQueue::new);

            WrappedTableRef::with_wrapper(self, |wrapped| {
                buf.par_iter()
                    .fold(
                        || (self.new_buffer(), exec_state.new_handle()),
                        |(mut mutation_buf, mut exec_state), (_, row)| {
                            let Some(subset) = self.rebuild_index.get_subset(&row[0]) else {
                                return (mutation_buf, exec_state);
                            };
                            let mut scanned = TaggedRowBuffer::new(self.n_columns);
                            rewriter.rewrite_subset(wrapped, subset, &mut scanned, &mut exec_state);
                            for (row_id, row) in scanned.non_stale_mut() {
                                let to_remove =
                                    self.data.get_row(row_id).map(|x| &x[0..self.n_keys]);
                                if let Some(key) = to_remove {
                                    mutation_buf.stage_remove(key);
                                }
                                insert_row!(self, mutation_buf, row, next_ts);
                            }
                            (mutation_buf, exec_state)
                        },
                    )
                    .for_each(|_| {});
            });
        } else {
            let mut scratch = TaggedRowBuffer::new(self.n_columns);
            let mut write_buf = self.new_buffer();
            for (_, id) in buf.iter() {
                let Some(subset) = self.rebuild_index.get_subset(&id[0]) else {
                    continue;
                };
                WrappedTableRef::with_wrapper(self, |wrapped| {
                    rewriter.rewrite_subset(wrapped, subset, &mut scratch, exec_state);
                });
                for (row_id, row) in scratch.non_stale_mut() {
                    let Some(to_remove) = self.data.get_row(row_id) else {
                        continue;
                    };
                    let (shard, hc) = hash_code(self.hash.shard_data(), to_remove, self.n_keys);
                    if let Ok(entry) = self.hash.mut_shards()[shard.index()]
                        .find_entry(hc, |entry| entry.row == row_id)
                    {
                        entry.remove();
                    }
                    self.data.set_stale(row_id);
                    insert_row!(self, write_buf, row, next_ts);
                }
                scratch.clear();
            }
        }
    }

    fn rewrite_nonincremental(
        &mut self,
        rewriter: &dyn Rewriter,
        next_ts: Value,
        exec_state: &mut ExecutionState,
    ) {
        const STEP_SIZE: usize = 2048;
        if do_parallel(self.data.next_row().index()) {
            (0..self.data.next_row().index())
                .into_par_iter()
                .step_by(STEP_SIZE)
                .fold(
                    || {
                        (
                            self.new_buffer(),
                            TaggedRowBuffer::new(self.n_columns),
                            exec_state.new_handle(),
                        )
                    },
                    |(mut mutation_buf, mut buf, mut exec_state), start| {
                        rewriter.rewrite_buf(
                            &self.data.data,
                            RowId::from_usize(start),
                            RowId::from_usize(cmp::min(
                                start + STEP_SIZE,
                                self.data.next_row().index(),
                            )),
                            &mut buf,
                            &mut exec_state,
                        );
                        for (row_id, row) in buf.non_stale_mut() {
                            let to_remove = self.data.get_row(row_id).map(|x| &x[0..self.n_keys]);
                            if let Some(key) = to_remove {
                                mutation_buf.stage_remove(key);
                            }
                            insert_row!(self, mutation_buf, row, next_ts);
                        }
                        buf.clear();
                        (mutation_buf, buf, exec_state)
                    },
                )
                .for_each(|_| {});
        } else {
            let mut buf = TaggedRowBuffer::new(self.n_columns);
            let mut write_buf = self.new_buffer();
            for start in (0..self.data.next_row().index()).step_by(STEP_SIZE) {
                rewriter.rewrite_buf(
                    &self.data.data,
                    RowId::from_usize(start),
                    RowId::from_usize(cmp::min(start + STEP_SIZE, self.data.next_row().index())),
                    &mut buf,
                    exec_state,
                );
                for (row_id, row) in buf.non_stale_mut() {
                    if let Some(to_remove) = self.data.get_row(row_id).map(|x| &x[0..self.n_keys]) {
                        let (shard, hc) = hash_code(self.hash.shard_data(), to_remove, self.n_keys);
                        if let Ok(entry) = self.hash.mut_shards()[shard.index()]
                            .find_entry(hc, |entry| entry.row == row_id)
                        {
                            entry.remove();
                        }
                        self.data.set_stale(row_id);
                    }
                    insert_row!(self, write_buf, row, next_ts);
                }
                buf.clear();
            }
        }
    }
}

fn do_parallel(_workload_size: usize) -> bool {
    #[cfg(test)]
    {
        use rand::{thread_rng, Rng};
        thread_rng().gen::<bool>()
    }
    #[cfg(not(test))]
    {
        _workload_size > 1000 && rayon::current_num_threads() > 1
    }
}

fn incremental_rebuild(_uf_size: usize, _table_size: usize, _parallel: bool) -> bool {
    #[cfg(test)]
    {
        use rand::{thread_rng, Rng};
        thread_rng().gen::<bool>()
    }
    #[cfg(not(test))]
    {
        if _parallel {
            _table_size > 10_000 && _uf_size * 8192 <= _table_size
        } else {
            _table_size > 10000 && _uf_size * 8 <= _table_size
        }
    }
}
