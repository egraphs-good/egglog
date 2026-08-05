//! Support for containers
//!
//! Each container type owns a variable-arity table keyed by its canonical flat
//! sequence. The table's one non-key column is the container identity, which
//! shares the ordinary e-graph id space and is therefore sparse. A type-erased
//! registry coordinates publication, rebuilding, and dependency scheduling
//! across those tables.

use std::{
    any::{Any, TypeId},
    hash::Hash,
    ops::Deref,
};

use crate::numeric_id::{DenseIdMap, NumericId, define_id};

use crate::{
    BaseValues, CounterId, ExecutionState, Subset, SubsetRef, TableId, Value, WrappedTable,
    common::{HashMap, IndexSet, SubsetTracker},
    table_spec::{MaintenanceTable, Rebuilder, ValueRebuilder},
};

mod sequence;

#[cfg(test)]
mod tests;

use sequence::SequenceContainerEnv;

define_id!(
    pub ContainerValueId,
    u32,
    "a typed identifier for a container's variable-arity table"
);

// Container tables participate in the same dense storage namespace as
// fixed-arity relation tables. `ContainerValueId` remains a distinct public
// type so APIs cannot accidentally use a relation as a container, but the
// representation is the backing table's `TableId`.
impl From<ContainerValueId> for TableId {
    fn from(id: ContainerValueId) -> Self {
        TableId::from_usize(id.index())
    }
}

impl ContainerValueId {
    pub(crate) fn from_table_id(id: TableId) -> Self {
        ContainerValueId::from_usize(id.index())
    }
}

pub trait MergeFn:
    Fn(&mut ExecutionState, Value, Value) -> Value + dyn_clone::DynClone + Send + Sync
{
}
impl<T: Fn(&mut ExecutionState, Value, Value) -> Value + Clone + Send + Sync> MergeFn for T {}

// Implements `Clone` for `Box<dyn MergeFn>`.
dyn_clone::clone_trait_object!(MergeFn);

#[derive(Clone, Default)]
struct ContainerIds {
    ids: HashMap<TypeId, ContainerValueId>,
}

impl ContainerIds {
    fn insert(&mut self, ty: TypeId, id: ContainerValueId) -> ContainerValueId {
        *self.ids.entry(ty).or_insert(id)
    }

    fn get(&self, ty: &TypeId) -> Option<ContainerValueId> {
        self.ids.get(ty).copied()
    }
}

#[derive(Clone, Default)]
pub struct ContainerValues {
    subset_tracker: SubsetTracker,
    container_ids: ContainerIds,
    // ContainerValueId shares the global TableId representation, so relation
    // registrations appear as holes in this container-only map.
    data: DenseIdMap<ContainerValueId, Box<dyn DynamicContainerEnv + Send + Sync>>,
}

/// An owned container decoded from its canonical sequence.
///
/// This private `Deref` facade preserves the existing public lookup API even
/// though lookups no longer borrow from shared container storage.
/// Performance-sensitive primitives should
/// use [`ExecutionState::container_sequence`] and operate on the serialized
/// values directly.
struct ContainerValueRef<C>(C);

impl<C> Deref for ContainerValueRef<C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Summary returned by container rebuild.
///
/// `changed` means some container entry changed during rebuild, either because
/// its contents changed or because its outer id canonicalized.
///
/// `dirty_ids` is narrower: it records container ids whose semantics changed
/// while their stored outer id stayed stable. Ordinary table rebuild already
/// handles changed-id cases; these ids need a follow-up parent-row refresh.
/// This includes containers that changed directly and containers whose
/// contained containers changed in place.
///
/// For example, `l(vec-of(w(k(b))))` can rebuild to `l(vec-of(k(b)))` without
/// changing the `Vec` id. The row is now newly matchable, but seminaive will
/// miss it unless the parent row is retimestamped.
#[derive(Clone, Default)]
pub(crate) struct ContainerRebuildSummary {
    changed: bool,
    // Container ids whose semantics changed in a way that may not produce a
    // fresh parent-row delta during ordinary table rebuild.
    dirty_ids: IndexSet<Value>,
}

impl ContainerRebuildSummary {
    pub(crate) fn changed(&self) -> bool {
        self.changed
    }

    pub(crate) fn dirty_ids(&self) -> &IndexSet<Value> {
        &self.dirty_ids
    }

    fn note_change(&mut self) {
        self.changed = true;
    }

    fn note_dirty_id(&mut self, value: Value) {
        self.changed = true;
        self.dirty_ids.insert(value);
    }

    pub(crate) fn extend(&mut self, other: Self) {
        self.changed |= other.changed;
        self.dirty_ids.extend(other.dirty_ids);
    }
}

/// The shared union-find rebuild inputs for one container rebuild cycle.
///
/// Preparing this once lets every container type share the same incremental
/// union-find delta without advancing the subset tracker more than once.
pub(crate) struct ContainerRebuildContext<'a> {
    table: &'a WrappedTable,
    rebuilder: Box<dyn Rebuilder + 'a>,
    to_scan: Option<SubsetRef<'a>>,
}

/// Owned portion of a container rebuild context. Keeping the union-find subset
/// separate lets the database detach and restore one dependency stratum at a
/// time while recreating the short-lived rebuilder borrow from the still-
/// installed source table.
pub(crate) struct ContainerRebuildPlan {
    to_scan: Option<Subset>,
}

impl ContainerRebuildPlan {
    pub(crate) fn context<'a>(
        &'a self,
        table: &'a WrappedTable,
    ) -> Option<ContainerRebuildContext<'a>> {
        Some(ContainerRebuildContext {
            table,
            rebuilder: table.rebuilder(&[])?,
            to_scan: self.to_scan.as_ref().map(Subset::as_ref),
        })
    }
}

impl ContainerRebuildContext<'_> {
    pub(crate) fn apply(
        &self,
        environment: &mut (dyn DynamicContainerEnv + Send + Sync),
        exec_state: &mut ExecutionState,
    ) -> ContainerRebuildSummary {
        environment.apply_rebuild(self.table, &*self.rebuilder, self.to_scan, exec_state)
    }
}

impl ContainerValues {
    pub fn new() -> Self {
        Default::default()
    }

    fn get_env<C: ContainerValue>(
        &self,
    ) -> Option<(ContainerValueId, &(dyn DynamicContainerEnv + Send + Sync))> {
        let id = self.container_ids.get(&TypeId::of::<C>())?;
        Some((id, &**self.data.get(id)?))
    }

    fn get_typed_env<C: ContainerValue>(&self) -> Option<&SequenceContainerEnv<C>> {
        self.get_env::<C>()?
            .1
            .as_any()
            .downcast_ref::<SequenceContainerEnv<C>>()
    }

    /// Return the id of an already registered container environment.
    ///
    /// This check does not mutate the registry, so the database can distinguish
    /// an idempotent registration from a first registration before installing
    /// the corresponding maintenance-scheduler participant.
    pub(crate) fn registered_type<C: ContainerValue>(&self) -> Option<ContainerValueId> {
        let (id, env) = self.get_env::<C>()?;
        assert!(
            env.as_any()
                .downcast_ref::<SequenceContainerEnv<C>>()
                .is_some(),
            "container environment has the wrong concrete value type"
        );
        Some(id)
    }

    /// Iterate over the containers of the given type.
    pub fn for_each<C: ContainerValue>(&self, mut f: impl FnMut(&C, Value)) {
        let Some((_, env)) = self.get_env::<C>() else {
            return;
        };
        env.as_any()
            .downcast_ref::<SequenceContainerEnv<C>>()
            .expect("container environment has the wrong concrete value type")
            .for_each(&mut f);
    }

    /// Get the container associated with the value `val` in the database. The caller must know the
    /// type of the container.
    ///
    /// This is the slow compatibility path: the returned facade owns a Rust
    /// container reconstructed from its canonical sequence.
    pub fn get_val<C: ContainerValue>(&self, val: Value) -> Option<impl Deref<Target = C> + '_> {
        Some(ContainerValueRef(
            self.get_typed_env::<C>()?.get_container(val)?,
        ))
    }

    pub fn register_val<C: ContainerValue>(
        &self,
        container: C,
        exec_state: &mut ExecutionState,
    ) -> Value {
        self.get_typed_env::<C>()
            .expect("must register container type before registering a value")
            .get_or_insert(&container, exec_state)
    }

    /// Return the committed flat value sequence for a container. This is the
    /// fast read path and performs no Rust-container reconstruction.
    pub fn get_sequence<C: ContainerValue>(&self, value: Value) -> Option<&[Value]> {
        self.get_typed_env::<C>()?.get_values(value)
    }

    /// Rebuild a single container value by remapping each contained value
    /// through `remap`, returning the (possibly new) interned value, or `value`
    /// unchanged if it is not a registered container of the type behind
    /// `type_id`.
    ///
    /// The caller supplies the remapping explicitly and identifies the
    /// container type dynamically by its [`TypeId`].
    pub fn rebuild_val_with(
        &self,
        type_id: TypeId,
        value: Value,
        exec_state: &mut ExecutionState,
        remap: &(dyn Fn(Value) -> Value + Send + Sync),
    ) -> Value {
        let Some(id) = self.container_ids.get(&type_id) else {
            return value;
        };
        let Some(env) = self.data.get(id) else {
            return value;
        };
        env.rebuild_val_with(value, exec_state, remap)
            .unwrap_or(value)
    }

    /// Prepare the source-table delta once without retaining a borrow of the
    /// source table. A short-lived [`ContainerRebuildContext`] can then be
    /// reconstructed for each mixed participant stratum.
    pub(crate) fn prepare_rebuild_plan(
        &mut self,
        table_id: TableId,
        table: &WrappedTable,
    ) -> Option<ContainerRebuildPlan> {
        let rebuilder = table.rebuilder(&[])?;
        let to_scan = rebuilder
            .hint_col()
            .map(|_| self.subset_tracker.recent_updates(table_id, table));
        Some(ContainerRebuildPlan { to_scan })
    }

    /// Add ancestor containers to the dirty-id set until it is transitively closed.
    ///
    /// A rebuild can change a container's semantics in place without changing
    /// its id. If that container is itself stored inside another container,
    /// the parent container has also changed semantically even though no direct
    /// rebuild touched its contents. For example, with
    /// `(p (vec-of (vec-of (w (b)))))` and `(rewrite (w x) x)`, the inner
    /// `Vec` rebuilds in place to `vec-of (b)`. Without this closure, only the
    /// inner `Vec` id is dirty; the outer `Vec` row is not retimestamped, so a
    /// later rule like `(rewrite (p (vec-of (vec-of (b)))) (b))` can miss the
    /// newly matchable parent row.
    pub(crate) fn expand_dirty_id_closure(&self, summary: &mut ContainerRebuildSummary) {
        let mut frontier = summary.dirty_ids.clone();
        let mut seen = frontier.iter().copied().collect::<IndexSet<_>>();

        while !frontier.is_empty() {
            let mut next = IndexSet::default();
            for (_, env) in self.data.iter() {
                env.extend_containers_containing(&frontier, &mut next);
            }
            frontier.clear();
            for value in next {
                if seen.insert(value) {
                    summary.note_dirty_id(value);
                    frontier.insert(value);
                }
            }
        }
    }

    /// Register a container type whose canonical representation is a
    /// variable-length sequence followed by one non-key identity value.
    ///
    /// Database-level registration wraps this method so the environment also
    /// becomes a maintenance-scheduler participant.
    pub(crate) fn register_type<C: ContainerValue>(
        &mut self,
        id: ContainerValueId,
        id_counter: CounterId,
        merge_fn: impl MergeFn + 'static,
        base_values: BaseValues,
    ) -> ContainerValueId {
        let id = self.container_ids.insert(TypeId::of::<C>(), id);
        self.data.get_or_insert(id, || {
            Box::new(SequenceContainerEnv::<C>::new(
                id,
                Box::new(merge_fn),
                id_counter,
                base_values,
            ))
        });
        assert!(
            self.data[id]
                .as_any()
                .downcast_ref::<SequenceContainerEnv<C>>()
                .is_some(),
            "container environment has the wrong concrete value type"
        );
        id
    }

    /// Return the current number of values in a registered environment.
    #[cfg(test)]
    pub(crate) fn env_len(&self, id: ContainerValueId) -> usize {
        self.data[id].len()
    }

    /// Return the combined number of committed container rows.
    pub(crate) fn container_len(&self) -> usize {
        self.data.iter().map(|(_, env)| env.len()).sum()
    }

    /// Return the common maintenance surface for a container's backing
    /// sequence table.
    pub(crate) fn maintenance_table(&self, id: ContainerValueId) -> Option<&dyn MaintenanceTable> {
        self.data
            .get(id)
            .map(|environment| &**environment as &dyn MaintenanceTable)
    }

    /// Snapshot the ids of all registered container environments.
    pub(crate) fn env_ids(&self) -> Vec<ContainerValueId> {
        self.data.iter().map(|(id, _)| id).collect()
    }

    pub(crate) fn take_env(
        &mut self,
        id: ContainerValueId,
    ) -> Box<dyn DynamicContainerEnv + Send + Sync> {
        self.data
            .take(id)
            .expect("pending container environment must still be registered")
    }

    pub(crate) fn put_env(
        &mut self,
        id: ContainerValueId,
        env: Box<dyn DynamicContainerEnv + Send + Sync>,
    ) {
        let old = self.data.insert(id, env);
        assert!(old.is_none(), "container environment inserted twice");
    }
}

impl ExecutionState<'_> {
    /// Decode a container visible to this execution, including an identity
    /// predicted earlier by the same execution.
    ///
    /// This is the slow compatibility path for primitive implementations.
    /// Containers are reconstructed from their flat key.
    pub fn get_container<C: ContainerValue>(&self, value: Value) -> Option<C> {
        self.db
            .containers
            .get_typed_env::<C>()?
            .get_container_with_predictions(self, value)
    }

    /// Borrow the fast serialized payload for a container visible to this
    /// execution. No Rust container is constructed. The payload may include
    /// type-specific metadata, such as compact counts.
    pub fn container_sequence<C: ContainerValue>(&self, value: Value) -> Option<&[Value]> {
        self.db
            .containers
            .get_typed_env::<C>()?
            .get_values_with_predictions(self, value)
    }

    /// Intern an already serialized container key.
    ///
    /// This is the fast write path. Callers avoid constructing a Rust
    /// container but remain responsible for producing that type's canonical
    /// key format.
    pub fn register_container_sequence<C: ContainerValue>(&mut self, key: &[Value]) -> Value {
        let containers = self.db.containers;
        let env = containers
            .get_typed_env::<C>()
            .expect("container type is not registered");
        env.get_or_insert_key(key, self)
    }
}

/// A container with a canonical flat sequence representation.
///
/// The encoded sequence is the container's table key: two containers that are
/// semantically equal must encode identically, and decoding an encoded key
/// must recover the same container. The registry stores one non-key identity
/// value alongside that key.
///
/// [`BaseValues`] is available to codecs that keep type-erased descriptors in
/// the base-value pool. Most containers do not need it.
pub trait ContainerValue: Hash + Eq + Clone + Send + Sync + 'static {
    /// Append this container's canonical table key to `out`.
    fn encode_sequence(&self, base_values: &BaseValues, out: &mut Vec<Value>);

    /// Reconstruct the Rust value used by slow primitives and external APIs.
    fn decode_sequence(sequence: &[Value], base_values: &BaseValues) -> Self;

    /// Return the fast primitive view of a serialized key.
    ///
    /// The view may include compact metadata needed by fast primitives. Use
    /// [`ContainerValue::visit_sequence_values`] to identify the child values
    /// that actually participate in congruence closure.
    fn sequence_values(sequence: &[Value]) -> &[Value];

    /// Visit semantic child values used by the occurrence index and transitive
    /// dirty-container discovery.
    ///
    /// Implementations with metadata or non-rebuildable lanes must override
    /// this method so only values eligible for rebuilding are visited.
    fn visit_sequence_values(sequence: &[Value], visitor: &mut dyn FnMut(Value)) {
        for value in Self::sequence_values(sequence) {
            visitor(*value);
        }
    }

    /// Rebuild a serialized key into the initially empty `out` buffer.
    ///
    /// Returning `false` requires leaving `out` empty. Returning `true`
    /// requires writing a canonical key, including any unchanged metadata.
    fn rebuild_sequence(
        sequence: &[Value],
        base_values: &BaseValues,
        rebuilder: &dyn ValueRebuilder,
        out: &mut Vec<Value>,
    ) -> bool;
}

pub(crate) trait DynamicContainerEnv:
    Any + dyn_clone::DynClone + MaintenanceTable + Send + Sync
{
    fn as_any(&self) -> &dyn Any;
    fn apply_rebuild(
        &mut self,
        table: &WrappedTable,
        rebuilder: &dyn Rebuilder,
        subset: Option<SubsetRef>,
        exec_state: &mut ExecutionState,
    ) -> ContainerRebuildSummary;
    /// Add ids for containers in this environment that contain any `values`.
    ///
    /// This uses the container table's occurrence index and lets callers climb
    /// from dirty child ids to all directly containing parent container ids.
    fn extend_containers_containing(&self, values: &IndexSet<Value>, out: &mut IndexSet<Value>);
    /// Rebuild the single container `value` by remapping each contained value
    /// through `remap`, returning the (possibly new) interned value, or `None`
    /// if `value` is not registered in this environment.
    fn rebuild_val_with(
        &self,
        value: Value,
        exec_state: &mut ExecutionState,
        remap: &(dyn Fn(Value) -> Value + Send + Sync),
    ) -> Option<Value>;
}

// Implements `Clone` for `Box<dyn DynamicContainerEnv>`.
dyn_clone::clone_trait_object!(DynamicContainerEnv);

fn incremental_rebuild(uf_size: usize, table_size: usize, parallel: bool) -> bool {
    if parallel {
        table_size > 1000 && uf_size * 512 <= table_size
    } else {
        table_size > 1000 && uf_size * 8 <= table_size
    }
}

/// A [`ValueRebuilder`] that remaps individual values through a caller-supplied
/// closure. Used by [`ContainerValues::rebuild_val_with`] to rebuild a single
/// container against an explicit value mapping rather than the database union-find.
struct ClosureRebuilder<'a> {
    remap: &'a (dyn Fn(Value) -> Value + Send + Sync),
}

impl ValueRebuilder for ClosureRebuilder<'_> {
    fn rebuild_val(&self, val: Value) -> Value {
        (self.remap)(val)
    }
}
