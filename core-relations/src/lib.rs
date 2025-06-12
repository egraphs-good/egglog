#[macro_use]
#[cfg(test)]
pub(crate) mod table_shortcuts;
pub(crate) mod action;
pub(crate) mod base_values;
pub(crate) mod common;
pub(crate) mod containers;
pub(crate) mod dependency_graph;
pub(crate) mod free_join;
pub(crate) mod hash_index;
pub(crate) mod offsets;
pub(crate) mod parallel_heuristics;
pub(crate) mod pool;
pub(crate) mod query;
pub(crate) mod row_buffer;
pub(crate) mod table;

pub(crate) mod table_spec;
pub(crate) mod uf;

#[cfg(test)]
mod tests;

pub use action::{ExecutionState, MergeVal, QueryEntry, WriteVal};
pub use base_values::{BaseValue, BaseValueId, BaseValuePrinter, BaseValues, Boxed};
pub use common::Value;
pub use containers::{Container, ContainerId, Containers};
pub use free_join::{
    make_external_func, plan::PlanStrategy, AtomId, CounterId, Database, ExternalFunction,
    ExternalFunctionId, RuleReport, RuleSetReport, TableId, Variable,
};
pub use hash_index::TupleIndex;
pub use offsets::{OffsetRange, RowId, Subset, SubsetRef};
pub use pool::{Pool, PoolSet, Pooled};
pub use query::{
    CachedPlan, QueryBuilder, QueryError, RuleBuilder, RuleId, RuleSet, RuleSetBuilder,
};
pub use row_buffer::TaggedRowBuffer;
pub use table::{MergeFn, SortedWritesTable};
pub use table_spec::{
    ColumnId, Constraint, Offset, Rebuilder, Row, Table, TableChange, TableSpec, TableVersion,
    WrappedTable,
};
pub use uf::{DisplacedTable, DisplacedTableWithProvenance, ProofReason, ProofStep};
