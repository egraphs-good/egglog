//! Basic heuristics for whether or not to use a parallel or serial version of an algorithm.
//!
//! The parallel implementations in this crate generally have a noticeable overhead when compared
//! to the serial versions on small problem sizes.

use std::sync::OnceLock;

const DEFAULT_DB_LEVEL_OP_CUTOFF: usize = 10_000;
const DEFAULT_INDEX_CONSTRUCTION_CUTOFF: usize = 400_000;
const DEFAULT_REBUILD_CUTOFF: usize = 400_000;
const DEFAULT_INTRA_CONTAINER_CUTOFF: usize = 10_000;
const DEFAULT_INTER_CONTAINER_CUTOFF: usize = 8;
const DEFAULT_TABLE_OP_CUTOFF: usize = 400_000;
const DEFAULT_FREE_JOIN_FORK_DEPTH: usize = 2;
const DEFAULT_ACTION_BATCH_SIZE: usize = 8 * 1024;

static CUTOFFS: OnceLock<Cutoffs> = OnceLock::new();

struct Cutoffs {
    db_level_op: usize,
    index_construction: usize,
    rebuild: usize,
    intra_container: usize,
    inter_container: usize,
    table_op: usize,
    free_join_fork_depth: usize,
    action_batch_size: usize,
}

/// These are operations that work on a per-table or per-rule level where the size of the workload
/// is hard to gauge ahead of time. In this case, we gate parallel execution based on the number of
/// threads available and whether the total size of the database exceeds a certain threshold.
pub(crate) fn parallelize_db_level_op(db_size: usize) -> bool {
    should_parallelize(db_size, cutoffs().db_level_op)
}

/// Whether or not to use a parallel algorithm to construct a hash index.
pub(crate) fn parallelize_index_construction(items_to_insert: usize) -> bool {
    should_parallelize(items_to_insert, cutoffs().index_construction)
}

/// Whether or not to use a parallel algorithm to rebuild a [`crate::table::SortedWritesTable`].
pub(crate) fn parallelize_rebuild(table_size: usize) -> bool {
    should_parallelize(table_size, cutoffs().rebuild)
}

/// Whether or not to perform an operation for a given container memo table.
pub(crate) fn parallelize_intra_container_op(num_containers: usize) -> bool {
    should_parallelize(num_containers, cutoffs().intra_container)
}

/// Whether or not to perform an operation in parallel across a set of different container memo
/// tables.
pub(crate) fn parallelize_inter_container_op(num_containers: usize) -> bool {
    should_parallelize(num_containers, cutoffs().inter_container)
}

#[track_caller]
pub(crate) fn parallelize_table_op(table_size: usize) -> bool {
    should_parallelize(table_size, cutoffs().table_op)
}

/// Number of top free-join frames that may fork recursive drain work.
pub(crate) fn free_join_fork_depth() -> usize {
    cutoffs().free_join_fork_depth
}

/// Number of action bindings to batch before dispatching a scoped worker task.
pub(crate) fn action_batch_size() -> usize {
    cutoffs().action_batch_size
}

fn should_parallelize(len: usize, cutoff: usize) -> bool {
    len > cutoff && crate::parallel::current_num_threads() > 1
}

fn cutoffs() -> &'static Cutoffs {
    CUTOFFS.get_or_init(|| Cutoffs {
        db_level_op: cutoff(
            "EGGLOG_PARALLEL_DB_LEVEL_OP_CUTOFF",
            DEFAULT_DB_LEVEL_OP_CUTOFF,
        ),
        index_construction: cutoff(
            "EGGLOG_PARALLEL_INDEX_CONSTRUCTION_CUTOFF",
            DEFAULT_INDEX_CONSTRUCTION_CUTOFF,
        ),
        rebuild: cutoff("EGGLOG_PARALLEL_REBUILD_CUTOFF", DEFAULT_REBUILD_CUTOFF),
        intra_container: cutoff(
            "EGGLOG_PARALLEL_INTRA_CONTAINER_CUTOFF",
            DEFAULT_INTRA_CONTAINER_CUTOFF,
        ),
        inter_container: cutoff(
            "EGGLOG_PARALLEL_INTER_CONTAINER_CUTOFF",
            DEFAULT_INTER_CONTAINER_CUTOFF,
        ),
        table_op: cutoff("EGGLOG_PARALLEL_TABLE_OP_CUTOFF", DEFAULT_TABLE_OP_CUTOFF),
        free_join_fork_depth: cutoff(
            "EGGLOG_PARALLEL_FREE_JOIN_FORK_DEPTH",
            DEFAULT_FREE_JOIN_FORK_DEPTH,
        ),
        action_batch_size: cutoff(
            "EGGLOG_PARALLEL_ACTION_BATCH_SIZE",
            DEFAULT_ACTION_BATCH_SIZE,
        )
        .max(1),
    })
}

fn cutoff(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}
