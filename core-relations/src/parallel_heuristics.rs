//! Basic heuristics for whether or not to use a parallel or serial verion of an algorithm.
//!
//! The parallel implementations in this crate generally have a noticeable overhead when compared
//! to the serial versions on small problem sizes.

/// These are operations that work on a per-table or per-rule level where the size of the workload
/// is hard to gauge ahead of time. In this case, we gate parallel execution based on the number of
/// threads available and whether the total size of the database exceeds a certain threshold.
pub(crate) fn parallelize_db_level_op(db_size: usize) -> bool {
    db_size > 10_000 && rayon::current_num_threads() > 1
}

/// Whether or not to use a parallel algorithm to construct a hash index.
pub(crate) fn parallelize_index_construction(items_to_insert: usize, num_threads: usize) -> bool {
    items_to_insert > 20_000 && num_threads > 1
}

/// Whether or not to use a parallel algorithm to rebuild a [`crate::table::SortedWritesTable`].
pub(crate) fn parallelize_rebuild(table_size: usize) -> bool {
    table_size > 10_000 && rayon::current_num_threads() > 1
}

/// Whether or not to perform an operation for a given container memo table.
pub(crate) fn parallelize_intra_container_op(num_containers: usize) -> bool {
    num_containers > 1_000 && rayon::current_num_threads() > 1
}

/// Whether or not to perform an operation in parallel across a set of different container memo
/// tables.
pub(crate) fn parallelize_inter_container_op(num_containers: usize) -> bool {
    num_containers > 1 && rayon::current_num_threads() > 1
}

#[track_caller]
pub(crate) fn parallelize_table_op(table_size: usize) -> bool {
    table_size > 20_000 && rayon::current_num_threads() > 1
}
