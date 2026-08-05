//! A simple data-structure for tracking the dependencies of the merge functions
//! from different tables on one another.

use crate::numeric_id::DenseIdMap;

use crate::{TableId, common::IndexSet};

#[derive(Clone, Default)]
pub(crate) struct DependencyGraph {
    to_level: DenseIdMap<TableId, usize>,
    write_deps: DenseIdMap<TableId, IndexSet<TableId>>,
}

impl DependencyGraph {
    /// Register one maintenance participant with the dependency graph.
    ///
    /// Participants can have two kinds of dependencies:
    /// 1. Read dependencies are participants whose storage must remain readable
    ///    during this participant's merge function.
    /// 2. Write dependencies are participants that must merely be writable
    ///    during this participant's merge function.
    ///
    /// Write dependencies are generally weaker than read dependencies. Two tables with write
    /// dependencies on one another can run their merge operations in parallel. The same is not
    /// true for read dependencies.
    pub(crate) fn add_participant(
        &mut self,
        participant: TableId,
        read_deps: impl IntoIterator<Item = TableId>,
        write_deps: impl IntoIterator<Item = TableId>,
    ) {
        self.write_deps
            .get_or_default(participant)
            .extend(write_deps);
        assert!(
            self.to_level.get(participant).is_none(),
            "maintenance participant {participant:?} already added to graph"
        );
        let level = match read_deps.into_iter().map(|dep| self.to_level[dep]).max() {
            Some(level) => level + 1,
            None => 0,
        };
        self.to_level.insert(participant, level);
    }

    pub(crate) fn contains(&self, participant: TableId) -> bool {
        self.to_level.contains_key(participant)
    }

    pub(crate) fn participants(&self) -> impl Iterator<Item = TableId> + '_ {
        self.to_level.iter().map(|(participant, _)| participant)
    }

    pub(crate) fn level(&self, participant: TableId) -> usize {
        self.to_level[participant]
    }

    pub(crate) fn write_deps(&self, participant: TableId) -> impl Iterator<Item = TableId> + '_ {
        self.write_deps
            .get(participant)
            .into_iter()
            .flat_map(|deps| deps.iter().copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric_id::NumericId;

    #[test]
    fn maintenance_levels_keep_table_write_dependencies() {
        let first = TableId::new(0);
        let second = TableId::new(1);
        let sink = TableId::new(7);
        let mut graph = DependencyGraph::default();

        graph.add_participant(first, [], [sink]);
        graph.add_participant(second, [first], []);

        assert_eq!(graph.level(first), 0);
        assert_eq!(graph.level(second), 1);
        assert_eq!(graph.write_deps(first).collect::<Vec<_>>(), vec![sink]);
        assert!(graph.write_deps(second).next().is_none());
    }

    #[test]
    fn participant_ids_may_have_legacy_storage_holes() {
        let mut graph = DependencyGraph::default();
        graph.add_participant(TableId::new(3), [], []);
        graph.add_participant(TableId::new(7), [TableId::new(3)], []);

        assert!(!graph.contains(TableId::new(0)));
        assert!(graph.contains(TableId::new(3)));
        assert_eq!(
            graph.participants().collect::<Vec<_>>(),
            [TableId::new(3), TableId::new(7)]
        );
    }
}
