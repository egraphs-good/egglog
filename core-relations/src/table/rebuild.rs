//! Apply value-level rebuilds to a table.

use std::{cmp, mem};

use crate::numeric_id::NumericId;

use crate::{
    ColumnId, ExecutionState, Offset, OffsetRange, RowId, Subset, SubsetRef, TableId,
    TaggedRowBuffer, Value, WrappedTable,
    common::HashSet,
    hash_index::{ColumnIndex, Index, index_pool_threads},
    offsets::Offsets,
    parallel,
    parallel_heuristics::{
        parallelize_incremental_rebuild, parallelize_index_construction, parallelize_rebuild,
    },
    table_spec::{MutationBuffer, Rebuilder, TableVersion, WrappedTableRef},
};

use super::SortedWritesTable;

// Helper macro used for adjusting sort before inserting to a mutation buffer.
macro_rules! insert_row {
    ($this: expr, $mutation_buf: expr, $row:expr, $next_ts:expr) => {{
        let row = $row;
        let this = &*$this;
        let next_ts = $next_ts;
        row[this.sort_by.index()] = next_ts;
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
        let Some(hint_col) = rebuilder.hint_col() else {
            return self.rebuild_nonincremental(&*rebuilder, next_ts, exec_state);
        };
        // Advance the cursor even for small tables, so crossing the threshold
        // does not replay dirty IDs already handled by earlier full scans.
        let to_scan = self.subset_tracker.recent_updates(table_id, table);
        let rows = self.data.next_row().index();
        if rows <= 10_000 {
            self.rebuild_scan_credit = 0.0;
            return self.rebuild_nonincremental(&*rebuilder, next_ts, exec_state);
        }
        if to_scan.size() == 0 {
            // The incremental path has no candidates; defer refreshing its
            // index until there are dirty IDs that can use it.
            return false;
        }
        if self.prefer_incremental_rebuild(table, &*rebuilder, hint_col, to_scan.as_ref()) {
            self.rebuild_incremental(table, &*rebuilder, hint_col, to_scan, next_ts, exec_state)
        } else {
            self.rebuild_nonincremental(&*rebuilder, next_ts, exec_state)
        }
    }

    /// Estimate scan, index-maintenance, and candidate-lookup work separately.
    /// Estimates only choose between complete rebuild implementations; sampled
    /// rows never determine which rows will actually be canonicalized.
    fn prefer_incremental_rebuild(
        &mut self,
        table: &WrappedTable,
        rebuilder: &dyn Rebuilder,
        hint_col: ColumnId,
        to_scan: SubsetRef<'_>,
    ) -> bool {
        let rows = self.data.next_row().index();
        let dirty = to_scan.size();
        let columns = self.to_rebuild.len();
        let (pending, same_generation, indexed_rows) =
            self.rebuild_index.refresh_work(TableVersion {
                major: self.generation,
                minor: Offset::from_usize(rows),
            });
        let threads = parallel::current_num_threads();
        // Use the execution paths' actual parallel gates and partition limits.
        // Square-root scaling is a sublinear prior for shared memory bandwidth
        // and concurrent table rebuilds, not a guaranteed or measured speedup.
        let full_workers = if parallelize_rebuild(rows) {
            threads.min(rows.div_ceil(2048)).max(1)
        } else {
            1
        };
        let index_workers = if parallelize_index_construction(pending) {
            index_pool_threads()
                .min(self.rebuild_index.shard_count())
                .max(1)
        } else {
            1
        };
        let incremental_workers =
            if threads >= MIN_COARSE_REBUILD_THREADS && parallelize_incremental_rebuild(dirty) {
                threads
                    .min(dirty.div_ceil(MIN_REBUILD_PARTITION_ROWS))
                    .max(1)
            } else {
                1
            };
        let speed = |workers: usize| (workers as f64).sqrt();
        // These relative costs and amortization thresholds are heuristic priors,
        // validated across workloads, rather than fitted timing coefficients.
        const INDEX_SAMPLES: usize = 64;
        const ROW_SAMPLES: usize = 128;
        const INDEX_VALUE_COST: f64 = 2.0;
        const INDEXED_VISIT_COST: f64 = 2.0;
        const DIRTY_LOOKUP_COST: f64 = 2.0;
        const SPARSE_WORK_FRACTION: f64 = 0.25;
        const INDEX_SETUP_FRACTION: f64 = 0.5;
        // Normalize scans to three rebuildable columns. Index maintenance and
        // indexed visits have less locality than a contiguous scan.
        let scan_width = columns as f64 / 3.0;
        let full_cost = rows as f64 * scan_width / speed(full_workers);
        let refresh_cost =
            pending as f64 * columns as f64 * INDEX_VALUE_COST / speed(index_workers);

        // A current-generation index can provide fanout statistics without
        // refreshing it. Count repeated visits through distinct dirty values;
        // stale row references also cost work in the incremental implementation.
        let mut sample_count = 0;
        let mut sample_visits = 0usize;
        let mut max_fanout = 0usize;
        if same_generation && indexed_rows != 0 {
            let samples = dirty.min(INDEX_SAMPLES);
            for i in 0..samples {
                let offset = sample_offset(i, dirty, samples);
                table.as_ref().for_each_col(
                    subset_partition(to_scan, offset, offset + 1),
                    hint_col,
                    &mut |_, id| {
                        let count = self.rebuild_index.get_subset(&id).map_or(0, |s| s.size());
                        sample_visits = sample_visits.saturating_add(count);
                        max_fanout = max_fanout.max(count);
                        sample_count += 1;
                    },
                );
            }
        }
        let existing_visits = if sample_count != 0 {
            sample_visits as f64 * dirty as f64 / sample_count as f64
        } else {
            0.0
        };
        let pending_visits = if pending != 0 {
            self.sample_rebuild_visits(rebuilder, indexed_rows, rows, ROW_SAMPLES)
                .2
        } else {
            0.0
        };
        // One high-fanout dirty ID is handled by one worker, even when other
        // dirty IDs can be partitioned across workers.
        let visit_cost = ((existing_visits + pending_visits) / speed(incremental_workers))
            .max(max_fanout as f64)
            * INDEXED_VISIT_COST
            * scan_width;
        let lookup_cost = dirty as f64 * DIRTY_LOOKUP_COST / speed(incremental_workers);
        if refresh_cost + visit_cost + lookup_cost < full_cost {
            self.rebuild_scan_credit = 0.0;
            return true;
        }
        if visit_cost + lookup_cost < full_cost * SPARSE_WORK_FRACTION {
            // Repeated sparse scans amortize the setup cost of a cold or stale
            // index, avoiding a policy that keeps choosing full scans forever.
            // The discount anticipates further reuse after repeated sparse work.
            let saving = full_cost - visit_cost - lookup_cost;
            let budget = refresh_cost * INDEX_SETUP_FRACTION;
            if self.rebuild_scan_credit + saving >= budget {
                self.rebuild_scan_credit = 0.0;
                return true;
            }
            self.rebuild_scan_credit = (self.rebuild_scan_credit + saving).min(budget);
        } else {
            self.rebuild_scan_credit = 0.0;
        }
        false
    }

    /// Estimate visits in unindexed physical rows without constructing an
    /// index. Rebuilding a value only reads the current canonical map; sampled
    /// rows are not changed. Stale rows contribute zero to the physical range.
    fn sample_rebuild_visits(
        &self,
        rebuilder: &dyn Rebuilder,
        start: usize,
        end: usize,
        limit: usize,
    ) -> (usize, usize, f64) {
        let span = end.saturating_sub(start);
        let samples = span.min(limit);
        if samples == 0 {
            return (0, 0, 0.0);
        }
        let mut changed_values = 0usize;
        // Narrow rows avoid allocating scratch space. Reuse one set for wide
        // rows so estimating visits does not take quadratic work in the width.
        let mut distinct_values = (self.to_rebuild.len() > 16).then(HashSet::<Value>::default);
        for i in 0..samples {
            let row_id = RowId::from_usize(start + sample_offset(i, span, samples));
            let Some(row) = self.data.get_row(row_id) else {
                continue;
            };
            if let Some(seen) = &mut distinct_values {
                seen.clear();
            }
            for (position, column) in self.to_rebuild.iter().enumerate() {
                let value = row[column.index()];
                let repeated = if let Some(seen) = &mut distinct_values {
                    !seen.insert(value)
                } else {
                    self.to_rebuild[..position]
                        .iter()
                        .any(|prior| row[prior.index()] == value)
                };
                if repeated {
                    continue;
                }
                if rebuilder.rebuild_val(value) != value {
                    changed_values += 1;
                }
            }
        }
        // Exact for short deltas; otherwise one extra observation avoids
        // treating a sample with no changes as proof that the delta is clean.
        // The estimate chooses a path and never skips canonicalization.
        let estimate = if samples == span {
            changed_values as f64
        } else {
            (changed_values + 1) as f64 * span as f64 / (samples + 1) as f64
        };
        (samples, changed_values, estimate)
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
        let mut mutation_buf = self.new_table_buffer();
        let mut refreshed_row = Vec::<Value>::with_capacity(self.n_columns);
        for row_id in candidate_rows {
            let Some(current_row) = self.data.get_row(row_id) else {
                continue;
            };
            // Preserve the logical row and only advance its sort/timestamp
            // column, so seminaive treats this as a fresh parent-row delta.
            mutation_buf.stage_remove_row(row_id, &current_row[0..self.n_keys]);
            refreshed_row.clear();
            refreshed_row.extend_from_slice(current_row);
            refreshed_row[self.sort_by.index()] = next_ts;
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
                    let mut mutation_buf = self.new_table_buffer();
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
                                mutation_buf.stage_remove_row(row_id, key);
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
                let mut write_buf = self.new_table_buffer();
                for (row_id, row) in scratch.non_stale_mut() {
                    if let Some(to_remove) = self.data.get_row(row_id).map(|x| &x[0..self.n_keys]) {
                        write_buf.stage_remove_row(row_id, to_remove);
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
                    let mut mutation_buf = self.new_table_buffer();
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
                            mutation_buf.stage_remove_row(row_id, key);
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
                let mut mutation_buf = self.new_table_buffer();
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
                            mutation_buf.stage_remove_row(row_id, key);
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
                let mut write_buf = self.new_table_buffer();
                for (row_id, row) in buf.non_stale_mut() {
                    if let Some(to_remove) = self.data.get_row(row_id).map(|x| &x[0..self.n_keys]) {
                        write_buf.stage_remove_row(row_id, to_remove);
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

#[cfg(test)]
#[path = "rebuild_tests.rs"]
mod end_to_end_tests;

// Compute floor(sample * span / samples) without overflowing on 32-bit
// targets. The remainder product is bounded by the small sample count.
fn sample_offset(sample: usize, span: usize, samples: usize) -> usize {
    debug_assert!(sample < samples);
    (span / samples) * sample + (span % samples) * sample / samples
}

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

#[cfg(test)]
mod tests {
    use super::rebuild_partition_size;

    #[test]
    fn sampling_large_ranges_does_not_overflow() {
        // In particular, the previous multiplication overflowed on wasm32
        // before reaching the largest representable physical row count.
        for samples in [1, 64, 128] {
            assert_eq!(super::sample_offset(0, usize::MAX, samples), 0);
            assert_eq!(
                super::sample_offset(samples - 1, usize::MAX, samples),
                usize::MAX - usize::MAX.div_ceil(samples),
            );
        }
        assert_eq!(super::sample_offset(2, 10, 3), 6);
    }

    #[test]
    fn rebuild_partitions_are_coarse_and_nonzero() {
        assert_eq!(rebuild_partition_size(0, 0, 256), 256);
        assert_eq!(rebuild_partition_size(1_000, 4, 256), 256);
        assert_eq!(rebuild_partition_size(4_096, 3, 256), 1_366);
    }
    use crate::{
        ExecutionState, SubsetRef, Value,
        numeric_id::NumericId,
        offsets::{Offsets, RowId},
        row_buffer::{RowBuffer, TaggedRowBuffer},
        table_spec::{ColumnId, Rebuilder, Table, ValueRebuilder, WrappedTableRef},
    };

    use super::SortedWritesTable;

    struct SampleRebuilder;

    impl ValueRebuilder for SampleRebuilder {
        fn rebuild_val(&self, value: Value) -> Value {
            match value {
                value if value == Value::new(2) => Value::new(1),
                value if value == Value::new(4) => Value::new(3),
                value => value,
            }
        }
    }

    impl Rebuilder for SampleRebuilder {
        fn hint_col(&self) -> Option<ColumnId> {
            Some(ColumnId::new(0))
        }

        fn rebuild_buf(
            &self,
            _: &RowBuffer,
            _: RowId,
            _: RowId,
            _: &mut TaggedRowBuffer,
            _: &mut ExecutionState,
        ) {
            unreachable!("sampling only reads individual canonical values")
        }

        fn rebuild_subset(
            &self,
            _: WrappedTableRef,
            _: SubsetRef,
            _: &mut TaggedRowBuffer,
            _: &mut ExecutionState,
        ) {
            unreachable!("sampling only reads individual canonical values")
        }
    }

    fn sample_table(rows: &[[u32; 3]]) -> SortedWritesTable {
        let mut table = SortedWritesTable::new(
            1,
            4,
            ColumnId::new(3),
            vec![ColumnId::new(0), ColumnId::new(1), ColumnId::new(2)],
            Box::new(|_, _, _, _| unreachable!("sampling never merges rows")),
        );
        for &[a, b, c] in rows {
            table.data.add_row(&[a, b, c, 0].map(Value::new));
        }
        table
    }

    #[test]
    fn rebuild_visit_sample_counts_distinct_values_and_skips_tombstones() {
        let mut table = sample_table(&[[2, 2, 4], [2, 2, 2], [1, 3, 5], [2, 4, 4]]);
        table.data.set_stale(RowId::new(3));

        // Each (changed value, row) produces one reverse-index visit. Repeated
        // values within a row do not, and a stale row contributes no visits.
        assert_eq!(
            table.sample_rebuild_visits(&SampleRebuilder, 0, 4, 128),
            (4, 3, 3.0)
        );
        assert_eq!(
            table.sample_rebuild_visits(&SampleRebuilder, 1, 4, 128),
            (3, 1, 1.0)
        );
        assert_eq!(
            table.sample_rebuild_visits(&SampleRebuilder, 2, 4, 128),
            (2, 0, 0.0)
        );
        assert_eq!(
            table.sample_rebuild_visits(&SampleRebuilder, 4, 4, 128),
            (0, 0, 0.0)
        );
        assert_eq!(
            table.sample_rebuild_visits(&SampleRebuilder, 0, 4, 0),
            (0, 0, 0.0)
        );
        // Reading canonical values for the estimate must not alter the table.
        assert_eq!(
            table.data.get_row(RowId::new(0)).unwrap(),
            &[Value::new(2), Value::new(2), Value::new(4), Value::new(0)]
        );
    }

    #[test]
    fn rebuild_visit_sample_uses_physical_rows_for_its_denominator() {
        let mut table = sample_table(&[[2, 2, 2]; 8]);
        table.data.set_stale(RowId::new(0));

        // The two sampled physical positions are 0 and 4. The stale position
        // stays in the denominator instead of inflating the live-row estimate.
        let (samples, changed, estimate) = table.sample_rebuild_visits(&SampleRebuilder, 0, 8, 2);
        assert_eq!((samples, changed), (2, 1));
        assert!((estimate - 16.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn rebuild_visit_sample_deduplicates_wide_rows_independently() {
        let mut table = SortedWritesTable::new(
            1,
            65,
            ColumnId::new(64),
            (0..64).map(ColumnId::new).collect(),
            Box::new(|_, _, _, _| unreachable!("sampling never merges rows")),
        );
        let mut row: Vec<Value> = (1..=64).map(Value::new).collect();
        row[63] = Value::new(2);
        row.push(Value::new(0));
        table.data.add_row(&row);
        table.data.add_row(&row);
        table.data.add_row(&row);
        table.data.set_stale(RowId::new(2));
        // Each live row has two distinct changed values (2 and 4). Duplicate
        // values within a row count once, and the scratch set resets per row.
        assert_eq!(
            table.sample_rebuild_visits(&SampleRebuilder, 0, 3, 128),
            (3, 4, 4.0),
        );
        assert_eq!(table.data.get_row(RowId::new(0)).unwrap(), row);
    }

    #[test]
    fn rebuild_visit_sample_does_not_treat_a_missed_change_as_a_clean_delta() {
        let mut rows = [[1, 3, 5]; 8];
        rows[7] = [2, 3, 5];
        let table = sample_table(&rows);

        // Positions 0, 2, 4, 6 miss the changed final row. Smoothing keeps the
        // estimate positive; this advisory estimate cannot prove zero work.
        let (samples, changed, estimate) = table.sample_rebuild_visits(&SampleRebuilder, 0, 8, 4);
        assert_eq!((samples, changed), (4, 0));
        assert!((estimate - 8.0 / 5.0).abs() < 1e-12);
        assert_eq!(
            table.sample_rebuild_visits(&SampleRebuilder, 0, 8, 8),
            (8, 1, 1.0)
        );
    }

    #[test]
    fn rebuild_scan_credit_is_reset_when_cloning_or_clearing() {
        let mut table = sample_table(&[[1, 3, 5]]);
        table.rebuild_scan_credit = 123.0;
        let cloned = table.clone();
        assert_eq!(cloned.rebuild_scan_credit, 0.0);
        assert_eq!(table.rebuild_scan_credit, 123.0);
        assert_eq!(cloned.data.data.len(), 1);

        table.clear();
        assert_eq!(table.rebuild_scan_credit, 0.0);
        assert_eq!(table.data.data.len(), 0);

        // clear() must also reset credit before its empty-table early return.
        table.rebuild_scan_credit = 456.0;
        table.clear();
        assert_eq!(table.rebuild_scan_credit, 0.0);
    }

    #[test]
    fn rebuild_selection_amortizes_sparse_work_and_reuses_a_warm_index() {
        let source = crate::WrappedTable::new(sample_table(&[[2, 3, 5]]));
        let dirty = source.all();
        let mut rows = vec![[1, 3, 5]; 20_000];
        // This changed row is beyond the sampled positions. The policy remains
        // advisory: refreshing the index must still find the affected final row.
        rows[19_999] = [2, 3, 5];
        let mut table = sample_table(&rows);
        assert!(!table.prefer_incremental_rebuild(
            &source,
            &SampleRebuilder,
            ColumnId::new(0),
            dirty.as_ref(),
        ));
        let eventually_incremental = (0..16).any(|_| {
            table.prefer_incremental_rebuild(
                &source,
                &SampleRebuilder,
                ColumnId::new(0),
                dirty.as_ref(),
            )
        });
        assert!(
            eventually_incremental,
            "repeated sparse scans must warm the index"
        );

        table.refresh_rebuild_index();
        let mut candidates = Vec::new();
        table
            .rebuild_index
            .get_subset(&Value::new(2))
            .unwrap()
            .offsets(|row| candidates.push(row.index()));
        assert_eq!(candidates, vec![19_999]);
        assert!(table.prefer_incremental_rebuild(
            &source,
            &SampleRebuilder,
            ColumnId::new(0),
            dirty.as_ref(),
        ));
        assert_eq!(table.rebuild_scan_credit, 0.0);
    }

    #[test]
    fn rebuild_selection_rejects_high_fanout_and_dense_unindexed_rows() {
        let source = crate::WrappedTable::new(sample_table(&[[2, 3, 5]]));
        let dirty = source.all();
        let mut high_fanout = sample_table(&vec![[2, 2, 2]; 20_000]);
        high_fanout.refresh_rebuild_index();
        // A single dirty ID is not automatically cheap: its reverse-index list
        // covers every row, even though repeated values within each row deduplicate.
        assert!(!high_fanout.prefer_incremental_rebuild(
            &source,
            &SampleRebuilder,
            ColumnId::new(0),
            dirty.as_ref(),
        ));

        let mut pending = sample_table(&vec![[1, 3, 5]; 20_000]);
        pending.refresh_rebuild_index();
        assert!(pending.prefer_incremental_rebuild(
            &source,
            &SampleRebuilder,
            ColumnId::new(0),
            dirty.as_ref(),
        ));
        for _ in 0..20_000 {
            pending.data.add_row(&[2, 2, 2, 0].map(Value::new));
        }
        // The old index contains no affected rows. Sampling its unindexed suffix
        // must prevent that stale statistic from making the pending work look free.
        assert!(!pending.prefer_incremental_rebuild(
            &source,
            &SampleRebuilder,
            ColumnId::new(0),
            dirty.as_ref(),
        ));
        assert_eq!(pending.rebuild_scan_credit, 0.0);
    }
}
