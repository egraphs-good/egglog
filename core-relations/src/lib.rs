#[macro_use]
#[cfg(test)]
pub(crate) mod table_shortcuts;
pub(crate) mod action;
pub(crate) mod common;
pub(crate) mod containers;
pub(crate) mod dependency_graph;
pub(crate) mod free_join;
pub(crate) mod hash_index;
pub(crate) mod offsets;
pub(crate) mod pool;
pub(crate) mod primitives;
pub(crate) mod query;
pub(crate) mod row_buffer;
pub(crate) mod table;

pub(crate) mod table_spec;
pub(crate) mod uf;

#[cfg(test)]
mod tests;

pub use action::{ExecutionState, MergeVal, QueryEntry, WriteVal};
pub use common::Value;
pub use containers::{Container, ContainerId, Containers};
pub use free_join::{
    make_external_func, plan::PlanStrategy, CounterId, Database, ExternalFunction,
    ExternalFunctionId, RuleReport, RuleSetReport, TableId, Variable,
};
pub use hash_index::TupleIndex;
pub use offsets::{OffsetRange, RowId, Subset, SubsetRef};
pub use pool::{Pool, PoolSet, Pooled};
pub use primitives::{Primitive, PrimitiveId, PrimitivePrinter, Primitives};
pub use query::{QueryBuilder, QueryError, RuleBuilder, RuleSet, RuleSetBuilder};
pub use row_buffer::TaggedRowBuffer;
pub use table::{MergeFn, SortedWritesTable};
pub use table_spec::{
    ColumnId, Constraint, Offset, Rebuilder, Row, Table, TableChange, TableSpec, TableVersion,
    WrappedTable,
};
pub use uf::{DisplacedTable, DisplacedTableWithProvenance, ProofReason, ProofStep};
