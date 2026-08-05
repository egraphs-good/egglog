//! Sequence-table storage for container values.
//!
//! A row is `[serialized container key..., identity]`. The identity is the
//! table's sole non-key value, so equal serialized containers collide and run
//! the ordinary container merge function. Producers predict identities in
//! their execution-local cache; no shared eager interning map is involved.

use std::{
    hash::{Hash, Hasher},
    marker::PhantomData,
};

use crossbeam_queue::SegQueue;
use rustc_hash::FxHasher;

use crate::{
    BaseValues, ExecutionState, Offset, RowId, SequenceTable, Subset, TableChange, TableVersion,
    TaggedRowBuffer, Value,
    common::{HashMap, IndexSet, ShardData, ShardId},
    numeric_id::{DenseIdMap, NumericId},
    offsets::Offsets,
    parallel_heuristics::parallelize_intra_container_op,
    table_spec::{MaintenanceTable, Rebuilder, ValueRebuilder},
};

use super::{
    ClosureRebuilder, ContainerBackend, ContainerRebuildSummary, ContainerValue, ContainerValueId,
    DynamicContainerEnv, MergeFn, incremental_rebuild,
};

/// A container with a canonical flat sequence representation.
///
/// The default rebuild path is intentionally the slow compatibility path: it
/// deserializes the sequence, invokes [`ContainerValue::rebuild_contents`],
/// and serializes the result. Implementations may override
/// [`SequenceContainerValue::rebuild_sequence`] to transform the flat values
/// directly.
pub trait SequenceContainerValue: ContainerValue {
    /// Append the canonical serialized key to `out`.
    fn encode_sequence(&self, base_values: &BaseValues, out: &mut Vec<Value>);

    /// Reconstruct the Rust container used by slow primitives and external
    /// APIs.
    fn decode_sequence(sequence: &[Value], base_values: &BaseValues) -> Self;

    /// Return the fast primitive view of the serialized key.
    ///
    /// Metadata used only to distinguish container modes belongs outside this
    /// slice. Most containers return their contained values directly; compact
    /// encodings may return an encoded payload and override
    /// [`SequenceContainerValue::visit_sequence_values`] for dependency
    /// discovery.
    fn sequence_values(sequence: &[Value]) -> &[Value];

    /// Visit semantic child values used for transitive dirty-container
    /// discovery. The default visits the fast primitive view directly.
    fn visit_sequence_values(sequence: &[Value], visitor: &mut dyn FnMut(Value)) {
        for value in Self::sequence_values(sequence) {
            visitor(*value);
        }
    }

    /// Rebuild a serialized key into the initially empty `out` buffer.
    ///
    /// Returning `false` requires leaving `out` empty. The default implements
    /// the slow deserialize/rebuild/serialize round trip.
    fn rebuild_sequence(
        sequence: &[Value],
        base_values: &BaseValues,
        rebuilder: &dyn ValueRebuilder,
        out: &mut Vec<Value>,
    ) -> bool {
        let mut container = Self::decode_sequence(sequence, base_values);
        if container.rebuild_contents(rebuilder) {
            container.encode_sequence(base_values, out);
            true
        } else {
            false
        }
    }
}

#[derive(Clone)]
struct SequenceCodec<C> {
    base_values: BaseValues,
    encode: fn(&C, &BaseValues, &mut Vec<Value>),
    decode: fn(&[Value], &BaseValues) -> C,
    values: for<'a> fn(&'a [Value]) -> &'a [Value],
    rebuild: fn(&[Value], &BaseValues, &dyn ValueRebuilder, &mut Vec<Value>) -> bool,
    marker: PhantomData<fn() -> C>,
}

impl<C: SequenceContainerValue> SequenceCodec<C> {
    fn new(base_values: BaseValues) -> Self {
        Self {
            base_values,
            encode: C::encode_sequence,
            decode: C::decode_sequence,
            values: C::sequence_values,
            rebuild: C::rebuild_sequence,
            marker: PhantomData,
        }
    }
}

/// A lock-free read-side reverse index, physically split into hash shards.
///
/// Merges update it under exclusive access to the environment. Rule execution
/// only performs immutable shard lookups, so no `DashMap` or per-read locking
/// is needed.
#[derive(Clone)]
struct SequenceReverse {
    shard_data: ShardData,
    shards: DenseIdMap<ShardId, HashMap<Value, RowId>>,
}

impl SequenceReverse {
    fn new(shard_data: ShardData) -> Self {
        let mut shards = DenseIdMap::with_capacity(shard_data.n_shards());
        for index in 0..shard_data.n_shards() {
            shards.insert(ShardId::from_usize(index), HashMap::default());
        }
        Self { shard_data, shards }
    }

    fn shard(&self, value: Value) -> ShardId {
        let mut hasher = FxHasher::default();
        value.hash(&mut hasher);
        self.shard_data.shard_id(hasher.finish())
    }

    fn get(&self, value: Value) -> Option<RowId> {
        self.shards[self.shard(value)].get(&value).copied()
    }

    fn insert(&mut self, value: Value, row: RowId) {
        let shard = self.shard(value);
        self.shards[shard].insert(value, row);
    }

    fn clear(&mut self) {
        for (_, shard) in self.shards.iter_mut() {
            shard.clear();
        }
    }
}

#[derive(Clone)]
pub(super) struct SequenceContainerEnv<C: ContainerValue> {
    id: ContainerValueId,
    counter: crate::CounterId,
    table: SequenceTable,
    reverse: SequenceReverse,
    codec: SequenceCodec<C>,
}

impl<C: SequenceContainerValue> SequenceContainerEnv<C> {
    pub(super) fn new(
        id: ContainerValueId,
        merge: Box<dyn MergeFn>,
        counter: crate::CounterId,
        base_values: BaseValues,
    ) -> Self {
        let table_merge = move |state: &mut ExecutionState,
                                current: &[Value],
                                incoming: &[Value],
                                out: &mut Vec<Value>| {
            let current_id = *current.last().expect("container row must have an identity");
            let incoming_id = *incoming
                .last()
                .expect("container row must have an identity");
            let merged = merge(state, current_id, incoming_id);
            if merged == current_id {
                false
            } else {
                out.extend_from_slice(&current[..current.len() - 1]);
                out.push(merged);
                true
            }
        };
        let table = SequenceTable::new_with_values_and_index(
            1,
            Box::new(table_merge),
            Box::new(C::visit_sequence_values),
        );
        let reverse = SequenceReverse::new(table.shard_data());
        Self {
            id,
            counter,
            table,
            reverse,
            codec: SequenceCodec::new(base_values),
        }
    }
}

impl<C: ContainerValue> SequenceContainerEnv<C> {
    fn get_key(&self, value: Value) -> Option<&[Value]> {
        let row = self.reverse.get(value)?;
        let values = self.table.values(row)?;
        if values != [value] {
            return None;
        }
        self.table.key(row)
    }

    pub(super) fn get_values(&self, value: Value) -> Option<&[Value]> {
        let key = self.get_key(value)?;
        Some((self.codec.values)(key))
    }

    pub(super) fn get_container(&self, value: Value) -> Option<C> {
        Some((self.codec.decode)(
            self.get_key(value)?,
            &self.codec.base_values,
        ))
    }

    pub(super) fn get_or_insert(&self, container: &C, exec_state: &mut ExecutionState) -> Value {
        let mut key = Vec::new();
        (self.codec.encode)(container, &self.codec.base_values, &mut key);
        self.get_or_insert_key(&key, exec_state)
    }

    pub(super) fn get_or_insert_key(
        &self,
        key: &[Value],
        exec_state: &mut ExecutionState,
    ) -> Value {
        if let Some(values) = self.table.get_values(key) {
            return values[0];
        }
        exec_state.predict_container_value(self.id, key, self.counter, || self.table.new_buffer())
    }

    pub(super) fn get_key_with_predictions<'a>(
        &'a self,
        exec_state: &'a ExecutionState<'_>,
        value: Value,
    ) -> Option<&'a [Value]> {
        if let Some(row) = exec_state.predicted_container_row(self.id, value) {
            return row.get(..row.len().checked_sub(1)?);
        }
        self.get_key(value)
    }

    pub(super) fn get_values_with_predictions<'a>(
        &'a self,
        exec_state: &'a ExecutionState<'_>,
        value: Value,
    ) -> Option<&'a [Value]> {
        Some((self.codec.values)(
            self.get_key_with_predictions(exec_state, value)?,
        ))
    }

    pub(super) fn get_container_with_predictions(
        &self,
        exec_state: &ExecutionState<'_>,
        value: Value,
    ) -> Option<C> {
        Some((self.codec.decode)(
            self.get_key_with_predictions(exec_state, value)?,
            &self.codec.base_values,
        ))
    }

    pub(super) fn for_each(&self, f: &mut impl FnMut(&C, Value)) {
        self.table
            .scan_key_values(self.table.all().as_ref(), |_, key, values| {
                let container = (self.codec.decode)(key, &self.codec.base_values);
                f(&container, values[0]);
            });
    }

    fn refresh_reverse(&mut self, previous: &TableVersion) {
        let current = self.table.version();
        let to_scan = if current.major == previous.major {
            self.table.updates_since(previous.minor)
        } else {
            self.reverse.clear();
            self.table.all()
        };
        let reverse = &mut self.reverse;
        self.table
            .scan_key_values(to_scan.as_ref(), |row, _, values| {
                reverse.insert(values[0], row);
            });
    }

    fn merge_table(&mut self, exec_state: &mut ExecutionState) -> TableChange {
        let previous = self.table.version();
        let changed = self.table.merge_with_state(exec_state);
        self.refresh_reverse(&previous);
        changed
    }

    /// Resolve the union-find delta through the sequence occurrence index.
    ///
    /// As with the fixed-table rebuild index, the UF supplies changed values
    /// and the local index maps those values back to candidate rows. The
    /// identity reverse map contributes a row when the container's own id is
    /// one of the changed values. Stale index entries are harmless because the
    /// subset rebuild reads rows through `SequenceTable::row`.
    fn incremental_rebuild_subset(
        &self,
        table: &crate::WrappedTable,
        rebuilder: &dyn Rebuilder,
        to_scan: crate::SubsetRef<'_>,
    ) -> Option<Subset> {
        if !incremental_rebuild(
            to_scan.size(),
            self.table.len(),
            parallelize_intra_container_op(self.table.len()),
        ) {
            return None;
        }
        let search_col = rebuilder
            .hint_col()
            .expect("incremental container rebuild requires a hint column");
        let mut dirty = TaggedRowBuffer::new(1);
        table.scan_project(
            to_scan,
            &[search_col],
            Offset::new(0),
            usize::MAX,
            &[],
            &mut dirty,
        );

        let mut candidate_rows = Vec::<RowId>::new();
        for (_, value) in dirty.iter() {
            let value = value[0];
            if let Some(row) = self.reverse.get(value) {
                candidate_rows.push(row);
            }
            if let Some(rows) = self.table.rows_for_indexed_value(value) {
                rows.offsets(|row| candidate_rows.push(row));
            }
        }
        candidate_rows.sort_unstable();
        candidate_rows.dedup();
        let mut subset = Subset::empty();
        for row in candidate_rows {
            subset.add_row_sorted(row);
        }
        Some(subset)
    }
}

impl<C: ContainerValue> MaintenanceTable for SequenceContainerEnv<C> {
    fn len(&self) -> usize {
        self.table.len()
    }

    fn new_buffer(&self) -> Box<dyn crate::MutationBuffer> {
        self.table.new_buffer()
    }

    fn merge(&mut self, exec_state: &mut ExecutionState) -> TableChange {
        self.merge_table(exec_state)
    }
}

impl<C: ContainerValue> DynamicContainerEnv for SequenceContainerEnv<C> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn backend(&self) -> ContainerBackend {
        ContainerBackend::Sequence
    }

    fn len(&self) -> usize {
        self.table.len()
    }

    fn maintenance_table(&self) -> Option<&dyn MaintenanceTable> {
        Some(self)
    }

    fn maintenance_table_mut(&mut self) -> Option<&mut dyn MaintenanceTable> {
        Some(self)
    }

    fn apply_rebuild(
        &mut self,
        table: &crate::WrappedTable,
        rebuilder: &dyn Rebuilder,
        subset: Option<crate::SubsetRef>,
        exec_state: &mut ExecutionState,
    ) -> ContainerRebuildSummary {
        let rebuild_sequence = self.codec.rebuild;
        let base_values = &self.codec.base_values;
        let stable_changed_ids = SegQueue::new();
        let previous = self.table.version();
        let rebuild_row = |row: &[Value], rebuilt: &mut Vec<Value>| {
            let key_end = row
                .len()
                .checked_sub(1)
                .expect("container row must have an identity");
            let key = &row[..key_end];
            let old_id = row[key_end];
            let new_id = rebuilder.rebuild_val(old_id);
            let key_changed = rebuild_sequence(key, base_values, rebuilder, rebuilt);
            if !key_changed && new_id == old_id {
                debug_assert!(rebuilt.is_empty());
                return false;
            }
            if !key_changed {
                rebuilt.extend_from_slice(key);
            }
            rebuilt.push(new_id);
            if key_changed && new_id == old_id {
                stable_changed_ids.push(old_id);
            }
            true
        };
        let incremental =
            subset.and_then(|subset| self.incremental_rebuild_subset(table, rebuilder, subset));
        let change = if let Some(subset) = &incremental {
            self.table.rebuild_full_rows_subset_with_state(
                subset.as_ref(),
                &rebuild_row,
                exec_state,
            )
        } else {
            self.table
                .rebuild_full_rows_with_state(&rebuild_row, exec_state)
        };
        self.refresh_reverse(&previous);

        let mut summary = ContainerRebuildSummary::default();
        if change.added || change.removed {
            summary.note_change();
        }
        while let Some(value) = stable_changed_ids.pop() {
            if self.get_key(value).is_some() {
                summary.note_dirty_id(value);
            }
        }
        summary
    }

    fn extend_containers_containing(&self, values: &IndexSet<Value>, out: &mut IndexSet<Value>) {
        for value in values {
            let Some(rows) = self.table.rows_for_indexed_value(*value) else {
                continue;
            };
            rows.offsets(|row| {
                if let Some(ids) = self.table.values(row) {
                    out.insert(ids[0]);
                }
            });
        }
    }

    fn rebuild_val_with(
        &self,
        value: Value,
        exec_state: &mut ExecutionState,
        remap: &(dyn Fn(Value) -> Value + Send + Sync),
    ) -> Option<Value> {
        let key = self.get_key_with_predictions(exec_state, value)?.to_vec();
        let mut rebuilt = Vec::new();
        if !(self.codec.rebuild)(
            &key,
            &self.codec.base_values,
            &ClosureRebuilder { remap },
            &mut rebuilt,
        ) {
            return Some(value);
        }
        Some(self.get_or_insert_key(&rebuilt, exec_state))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::numeric_id::NumericId;
    use crate::{
        ColumnId, ContainerValue, Database, ExecutionState, Rebuilder, RowId, SortedWritesTable,
        Table, Value, ValueRebuilder, WrappedTable, row_buffer::RowBuffer,
        table_spec::WrappedTableRef,
    };

    use super::{DynamicContainerEnv, SequenceContainerEnv, SequenceContainerValue};

    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    struct TestSequence(Vec<Value>);

    impl ContainerValue for TestSequence {
        fn rebuild_contents(&mut self, rebuilder: &dyn ValueRebuilder) -> bool {
            rebuilder.rebuild_slice(&mut self.0)
        }

        fn iter(&self) -> impl Iterator<Item = Value> + '_ {
            self.0.iter().copied()
        }
    }

    impl SequenceContainerValue for TestSequence {
        fn encode_sequence(&self, _base_values: &crate::BaseValues, out: &mut Vec<Value>) {
            out.extend_from_slice(&self.0);
        }

        fn decode_sequence(sequence: &[Value], _base_values: &crate::BaseValues) -> Self {
            Self(sequence.to_vec())
        }

        fn sequence_values(sequence: &[Value]) -> &[Value] {
            sequence
        }

        fn rebuild_sequence(
            sequence: &[Value],
            _base_values: &crate::BaseValues,
            rebuilder: &dyn ValueRebuilder,
            out: &mut Vec<Value>,
        ) -> bool {
            out.extend_from_slice(sequence);
            if rebuilder.rebuild_slice(out) {
                true
            } else {
                out.clear();
                false
            }
        }
    }

    fn value(value: usize) -> Value {
        Value::from_usize(value)
    }

    fn test_env(db: &mut Database) -> SequenceContainerEnv<TestSequence> {
        let counter = db.add_reservable_counter(8);
        SequenceContainerEnv::new(
            crate::ContainerValueId::new(0),
            Box::new(|_state: &mut ExecutionState, left, right| left.min(right)),
            counter,
            db.base_values().clone(),
        )
    }

    #[test]
    fn local_prediction_is_immediately_readable_in_both_forms() {
        let mut db = Database::new();
        let env = test_env(&mut db);
        db.with_execution_state(|state| {
            let expected = TestSequence(vec![value(10), value(20)]);
            let id = env.get_or_insert(&expected, state);
            assert_eq!(
                env.get_container_with_predictions(state, id),
                Some(expected)
            );
            assert_eq!(
                env.get_values_with_predictions(state, id),
                Some([value(10), value(20)].as_slice())
            );
            assert_eq!(env.get_or_insert_key(&[value(10), value(20)], state), id);
        });
    }

    #[test]
    fn independent_predictions_coalesce_during_merge() {
        let mut db = Database::new();
        let mut env = test_env(&mut db);
        let key = TestSequence(vec![value(7), value(8)]);
        let first = db.with_execution_state(|state| env.get_or_insert(&key, state));
        let second = db.with_execution_state(|state| env.get_or_insert(&key, state));
        assert_ne!(first, second);

        db.with_execution_state(|state| {
            let change = env.merge_table(state);
            assert!(change.added || change.removed);
        });
        let winner = first.min(second);
        assert_eq!(env.get_container(winner), Some(key));
        assert_eq!(env.get_container(first.max(second)), None);
    }

    #[test]
    fn reverse_index_is_rebuilt_after_sequence_compaction() {
        let mut db = Database::new();
        let mut env = test_env(&mut db);
        let ids = db.with_execution_state(|state| {
            (0..40)
                .map(|index| env.get_or_insert_key(&[value(100 + index)], state))
                .collect::<Vec<_>>()
        });
        db.with_execution_state(|state| {
            let change = env.merge_table(state);
            assert!(change.added || change.removed);
        });

        let mut removals = env.table.new_buffer();
        for index in 0..25 {
            removals.stage_remove(&[value(100 + index)]);
        }
        drop(removals);
        db.with_execution_state(|state| {
            let change = env.merge_table(state);
            assert!(change.added || change.removed);
        });

        for (index, id) in ids.into_iter().enumerate() {
            if index < 25 {
                assert_eq!(env.get_key(id), None);
            } else {
                assert_eq!(env.get_key(id), Some([value(100 + index)].as_slice()));
            }
        }
    }

    struct HintRebuilder {
        from: Value,
        to: Value,
        hint: Option<ColumnId>,
        visits: AtomicUsize,
    }

    impl ValueRebuilder for HintRebuilder {
        fn rebuild_val(&self, value: Value) -> Value {
            self.visits.fetch_add(1, Ordering::Relaxed);
            if value == self.from { self.to } else { value }
        }
    }

    impl Rebuilder for HintRebuilder {
        fn hint_col(&self) -> Option<ColumnId> {
            self.hint
        }

        fn rebuild_buf(
            &self,
            _buf: &RowBuffer,
            _start: RowId,
            _end: RowId,
            _out: &mut crate::TaggedRowBuffer,
            _exec_state: &mut ExecutionState,
        ) {
            unreachable!("sequence incremental test does not bulk-rebuild tables")
        }

        fn rebuild_subset(
            &self,
            _other: WrappedTableRef<'_>,
            _subset: crate::SubsetRef<'_>,
            _out: &mut crate::TaggedRowBuffer,
            _exec_state: &mut ExecutionState,
        ) {
            unreachable!("sequence incremental test does not bulk-rebuild tables")
        }
    }

    fn empty_rebuild_table() -> WrappedTable {
        WrappedTable::new(SortedWritesTable::new(
            1,
            1,
            None,
            vec![],
            Box::new(|_, _, _, _| false),
        ))
    }

    #[test]
    fn full_rebuild_marks_a_stable_identity_dirty() {
        let from = value(10);
        let to = value(20);
        let mut db = Database::new();
        let mut env = test_env(&mut db);
        let id = db.with_execution_state(|state| env.get_or_insert_key(&[from], state));
        db.with_execution_state(|state| {
            let change = env.merge_table(state);
            assert!(change.added || change.removed);
        });

        let rebuilder = HintRebuilder {
            from,
            to,
            hint: None,
            visits: AtomicUsize::new(0),
        };
        let summary = db.with_execution_state(|state| {
            env.apply_rebuild(&empty_rebuild_table(), &rebuilder, None, state)
        });

        assert!(summary.changed());
        assert_eq!(
            summary.dirty_ids().iter().copied().collect::<Vec<_>>(),
            [id]
        );
        assert_eq!(env.get_key(id), Some([to].as_slice()));
    }

    #[test]
    fn full_rebuild_remaps_an_outer_identity_without_marking_it_dirty() {
        let child = value(30);
        let mut db = Database::new();
        let mut env = test_env(&mut db);
        let old_id = db.with_execution_state(|state| env.get_or_insert_key(&[child], state));
        db.with_execution_state(|state| {
            let change = env.merge_table(state);
            assert!(change.added || change.removed);
        });
        let new_id = value(old_id.index() + 100_000);

        let rebuilder = HintRebuilder {
            from: old_id,
            to: new_id,
            hint: None,
            visits: AtomicUsize::new(0),
        };
        let summary = db.with_execution_state(|state| {
            env.apply_rebuild(&empty_rebuild_table(), &rebuilder, None, state)
        });

        assert!(summary.changed());
        assert!(summary.dirty_ids().is_empty());
        assert_eq!(env.get_key(old_id), None);
        assert_eq!(env.get_key(new_id), Some([child].as_slice()));
    }

    #[test]
    fn incremental_rebuild_uses_value_index_candidates() {
        const CONTAINERS: usize = 1_002;
        let from = value(50_000);
        let to = value(60_000);
        let mut db = Database::new();
        let mut env = test_env(&mut db);
        let ids = db.with_execution_state(|state| {
            (0..CONTAINERS)
                .map(|index| {
                    let child = if index == 777 {
                        from
                    } else {
                        value(100_000 + index)
                    };
                    env.get_or_insert_key(&[child], state)
                })
                .collect::<Vec<_>>()
        });
        db.with_execution_state(|state| {
            let change = env.merge_table(state);
            assert!(change.added || change.removed);
        });

        let mut dirty = SortedWritesTable::new(1, 1, None, vec![], Box::new(|_, _, _, _| false));
        dirty.new_buffer().stage_insert(&[from]);
        db.with_execution_state(|state| {
            dirty.merge(state);
        });
        let dirty = WrappedTable::new(dirty);
        let dirty_rows = dirty.all();
        let rebuilder = HintRebuilder {
            from,
            to,
            hint: Some(ColumnId::new(0)),
            visits: AtomicUsize::new(0),
        };
        let summary = db.with_execution_state(|state| {
            env.apply_rebuild(&dirty, &rebuilder, Some(dirty_rows.as_ref()), state)
        });

        assert!(summary.changed());
        assert_eq!(
            rebuilder.visits.load(Ordering::Relaxed),
            2,
            "only the selected row's identity and one child should rebuild"
        );
        assert_eq!(env.get_key(ids[777]), Some([to].as_slice()));
        assert_eq!(env.get_key(ids[776]), Some([value(100_776)].as_slice()));
    }
}
