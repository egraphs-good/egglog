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

// Fixed generic-join scheduling choices selected by cross-workload
// benchmarking. Coarse index partitioning is enabled. Recursive fallback work
// uses worker-local queues, but already-sharded partitions do not create an
// additional packet level. Aligned child caches and top-variable promotion
// remain disabled.
const GJ_TOP_INDEX_SHARDING: bool = true;
const GJ_LOCAL_DEPTH: usize = 0;
const GJ_LOCAL_MORSEL: usize = 64;
const GJ_LOCAL_QUEUE_LIMIT: usize = 2;
const GJ_CHILD_CACHE: GjChildCache = GjChildCache::Single;
const GJ_TOP_PROMOTION: GjTopPromotion = GjTopPromotion::Current;
const GJ_MIN_KEYS_PER_WORKER: usize = 16;

/// Layout for root trie-node child caches during coarse generic-join
/// partitioning.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GjChildCache {
    /// One lock per cached column.
    Single,
    /// One lock per top-index shard, using the same value hash partition.
    Aligned,
}

/// Policy for replacing the initially sorted top generic-join variable with a
/// coarse-partitionable later variable.
// The non-selected policies remain available to the executor's deterministic
// policy tests even though production uses the fixed `Current` policy.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GjTopPromotion {
    /// Consider only the variable already sorted into position zero.
    Current,
    /// Keep an eligible current variable, otherwise choose the first guarded
    /// eligible variable in the leading Intersect prefix.
    Eligible,
    /// Choose the eligible variable with the largest actual leader cardinality.
    Largest,
}

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

/// Whether a parallel generic join may schedule one coarse global job per
/// physical shard of its top cached index.
pub(crate) fn gj_top_index_sharding() -> bool {
    GJ_TOP_INDEX_SHARDING
}

/// Number of recursive levels below a coarse top-index shard that may enqueue
/// worker-local packets. The implementation currently accepts only zero or one.
pub(crate) fn gj_local_depth() -> usize {
    GJ_LOCAL_DEPTH
}

/// Number of top-level generic-join frames coalesced into one local packet.
pub(crate) fn gj_local_morsel() -> usize {
    GJ_LOCAL_MORSEL
}

/// Maximum number of outstanding local packets per coarse index shard.
pub(crate) fn gj_local_queue_limit() -> usize {
    GJ_LOCAL_QUEUE_LIMIT
}

/// Child-cache layout used by eligible coarse top-index partitions.
pub(crate) fn gj_child_cache() -> GjChildCache {
    GJ_CHILD_CACHE
}

/// Policy for promoting an eligible top generic-join variable.
pub(crate) fn gj_top_promotion() -> GjTopPromotion {
    GJ_TOP_PROMOTION
}

/// Minimum number of leader keys required per worker for coarse top-index
/// partitioning.
pub(crate) fn gj_min_keys_per_worker() -> usize {
    GJ_MIN_KEYS_PER_WORKER
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
