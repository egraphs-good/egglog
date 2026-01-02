//! A simple data-structure for tracking the dependencies of the merge functions
//! from different tables on one another.

use egglog_numeric_id::DenseIdMapSO;

use crate::numeric_id::{DenseIdMap, NumericId, define_id};

use crate::{TableId, common::IndexSet};

define_id!(
    LevelId,
    u32,
    "an identifier for a level in the dependency graph"
);

#[derive(Clone, Default)]
pub(crate) struct DependencyGraph {
    levels: DenseIdMap<LevelId, IndexSet<TableId>>,
    to_level: DenseIdMapSO<TableId, LevelId>,
    write_deps: DenseIdMap<TableId, IndexSet<TableId>>,
}

impl DependencyGraph {
    /// Register a table's dependencies with the dependency graph.
    ///
    /// Tables can have two kinds of dependences:
    /// 1. Read dependencies are tables that must be readable during the table's merge function.
    /// 2. Write dependencies are tables that must merely be writable during the table's merge
    ///    function.
    ///
    /// Write dependencies are generally weaker than read dependencies. Two tables with write
    /// dependencies on one another can run their merge operations in parallel. The same is not
    /// true for read dependencies.
    pub(crate) fn add_table(
        &mut self,
        table: TableId,
        read_deps: impl IntoIterator<Item = TableId>,
        write_deps: impl IntoIterator<Item = TableId>,
    ) {
        self.write_deps.get_or_default(table).extend(write_deps);
        assert!(
            self.to_level.get(table).is_none(),
            "table {table:?} already added to graph"
        );
        let level = match read_deps
            .into_iter()
            .map(|dep| *self.to_level.get(dep).unwrap())
            .max()
        {
            Some(level) => level.inc(),
            None => LevelId::new(0),
        };
        self.to_level.insert(table, level);
        self.levels.get_or_default(level).insert(table);
    }

    pub(crate) fn strata(&self) -> impl Iterator<Item = &IndexSet<TableId>> {
        self.levels.iter().map(|(_, tables)| tables)
    }
    pub(crate) fn write_deps(&self, table: TableId) -> impl Iterator<Item = TableId> + '_ {
        self.write_deps
            .get(table)
            .into_iter()
            .flat_map(|deps| deps.iter().copied())
    }
}
