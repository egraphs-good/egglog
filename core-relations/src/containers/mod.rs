//! Support for containers
//!
//! Containers behave a lot like base values. They are implemented differently because
//! their ids share a space with other Ids in the egraph and as a result, their ids need to be
//! sparse.
//!
//! This is a relatively "eagler" implementation of containers, reflecting egglog's current
//! semantics. One could imagine a variant of containers in which they behave more like egglog
//! functions than base values.

use std::{
    any::{Any, TypeId},
    hash::{Hash, Hasher},
    ops::Deref,
};

use crate::numeric_id::{DenseIdMap, IdVec, NumericId, define_id};
use crossbeam_queue::SegQueue;
use dashmap::SharedValue;
use rustc_hash::FxHasher;

use crate::{
    ColumnId, CounterId, ExecutionState, Offset, Subset, SubsetRef, TableId, TaggedRowBuffer,
    Value, WrappedTable,
    common::{DashMap, IndexSet, SubsetTracker},
    parallel,
    parallel_heuristics::{
        parallelize_db_level_op, parallelize_inter_container_op, parallelize_intra_container_op,
        parallelize_rebuild,
    },
    table_spec::{Rebuilder, ValueRebuilder},
};

mod sequence;

#[cfg(test)]
mod tests;

use sequence::SequenceContainerEnv;
pub use sequence::SequenceContainerValue;

define_id!(pub ContainerValueId, u32, "an identifier for containers");

pub trait MergeFn:
    Fn(&mut ExecutionState, Value, Value) -> Value + dyn_clone::DynClone + Send + Sync
{
}
impl<T: Fn(&mut ExecutionState, Value, Value) -> Value + Clone + Send + Sync> MergeFn for T {}

// Implements `Clone` for `Box<dyn MergeFn>`.
dyn_clone::clone_trait_object!(MergeFn);

#[derive(Clone, Default)]
struct ContainerIds {
    ids: IndexSet<TypeId>,
}

impl ContainerIds {
    fn insert(&mut self, ty: TypeId) -> ContainerValueId {
        if let Some(idx) = self.ids.get_index_of(&ty) {
            ContainerValueId::from_usize(idx)
        } else {
            let idx = self.ids.len();
            self.ids.insert(ty);
            ContainerValueId::from_usize(idx)
        }
    }

    fn get(&self, ty: &TypeId) -> Option<ContainerValueId> {
        self.ids.get_index_of(ty).map(ContainerValueId::from_usize)
    }
}

#[derive(Clone, Default)]
pub struct ContainerValues {
    subset_tracker: SubsetTracker,
    container_ids: ContainerIds,
    data: DenseIdMap<ContainerValueId, Box<dyn DynamicContainerEnv + Send + Sync>>,
}

enum ContainerValueRefInner<'a, C> {
    Legacy(Box<dyn Deref<Target = C> + 'a>),
    Sequence(C),
}

/// A borrowed legacy container or an owned value decoded from a sequence.
///
/// Sequence-backed lookups deliberately reconstruct the Rust container: this
/// is the slow compatibility path. Performance-sensitive primitives should
/// use [`ExecutionState::container_sequence`] and operate on the serialized
/// values directly.
struct ContainerValueRef<'a, C> {
    inner: ContainerValueRefInner<'a, C>,
}

impl<C> Deref for ContainerValueRef<'_, C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        match &self.inner {
            ContainerValueRefInner::Legacy(value) => value,
            ContainerValueRefInner::Sequence(value) => value,
        }
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
pub struct ContainerRebuildSummary {
    changed: bool,
    // Container ids whose semantics changed in a way that may not produce a
    // fresh parent-row delta during ordinary table rebuild.
    dirty_ids: IndexSet<Value>,
}

impl ContainerRebuildSummary {
    /// Returns whether any container entry changed during rebuild.
    pub fn changed(&self) -> bool {
        self.changed
    }

    /// Returns the container ids whose parent rows may need retimestamping.
    pub fn dirty_ids(&self) -> &IndexSet<Value> {
        &self.dirty_ids
    }

    fn note_change(&mut self) {
        self.changed = true;
    }

    fn note_dirty_id(&mut self, value: Value) {
        self.changed = true;
        self.dirty_ids.insert(value);
    }

    fn extend(&mut self, other: Self) {
        self.changed |= other.changed;
        self.dirty_ids.extend(other.dirty_ids);
    }
}

/// The storage implementation behind a type-erased container environment.
///
/// Sequence environments are eligible to become independently scheduled
/// database maintenance targets. Legacy environments remain in the registry's
/// compatibility rebuild pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ContainerBackend {
    Legacy,
    Sequence,
}

/// The shared union-find rebuild inputs for one container rebuild cycle.
///
/// Preparing this once is important for incremental rebuilds: consulting the
/// subset tracker separately for the sequence and legacy passes would advance
/// its watermark twice and make the second pass miss the relevant UF delta.
struct ContainerRebuildContext<'a> {
    table: &'a WrappedTable,
    rebuilder: Box<dyn Rebuilder + 'a>,
    to_scan: Option<Subset>,
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

    fn get_sequence_env<C: ContainerValue>(&self) -> Option<&SequenceContainerEnv<C>> {
        self.get_env::<C>()?
            .1
            .as_any()
            .downcast_ref::<SequenceContainerEnv<C>>()
    }

    /// Return the id of an already registered sequence environment.
    ///
    /// This check does not mutate the registry, so the database can distinguish
    /// an idempotent registration from a first registration before installing
    /// the corresponding maintenance-scheduler participant.
    pub(crate) fn registered_sequence_type<C: SequenceContainerValue>(
        &self,
    ) -> Option<ContainerValueId> {
        let (id, env) = self.get_env::<C>()?;
        assert!(
            env.as_any()
                .downcast_ref::<SequenceContainerEnv<C>>()
                .is_some(),
            "container type was already registered with a different backend"
        );
        Some(id)
    }

    /// Iterate over the containers of the given type.
    pub fn for_each<C: ContainerValue>(&self, mut f: impl FnMut(&C, Value)) {
        let Some((_, env)) = self.get_env::<C>() else {
            return;
        };
        if let Some(env) = env.as_any().downcast_ref::<ContainerEnv<C>>() {
            for ent in env.to_id.iter() {
                f(ent.key(), *ent.value());
            }
        } else if let Some(env) = env.as_any().downcast_ref::<SequenceContainerEnv<C>>() {
            env.for_each(&mut f);
        } else {
            unreachable!("container environment has the wrong concrete value type");
        }
    }

    /// Get the container associated with the value `val` in the database. The caller must know the
    /// type of the container.
    ///
    /// The return type of this function may contain lock guards. Attempts to modify the contents
    /// of the containers database may deadlock if the given guard has not been dropped.
    pub fn get_val<C: ContainerValue>(&self, val: Value) -> Option<impl Deref<Target = C> + '_> {
        let (_, env) = self.get_env::<C>()?;
        if let Some(env) = env.as_any().downcast_ref::<ContainerEnv<C>>() {
            return Some(ContainerValueRef {
                inner: ContainerValueRefInner::Legacy(Box::new(env.get_container(val)?)),
            });
        }
        let env = env
            .as_any()
            .downcast_ref::<SequenceContainerEnv<C>>()
            .expect("container environment has the wrong concrete value type");
        Some(ContainerValueRef {
            inner: ContainerValueRefInner::Sequence(env.get_container(val)?),
        })
    }

    fn get_owned<C: ContainerValue>(&self, value: Value) -> Option<C> {
        let (_, env) = self.get_env::<C>()?;
        if let Some(env) = env.as_any().downcast_ref::<ContainerEnv<C>>() {
            return Some(env.get_container(value)?.deref().clone());
        }
        env.as_any()
            .downcast_ref::<SequenceContainerEnv<C>>()
            .expect("container environment has the wrong concrete value type")
            .get_container(value)
    }

    pub fn register_val<C: ContainerValue>(
        &self,
        container: C,
        exec_state: &mut ExecutionState,
    ) -> Value {
        let (_, env) = self
            .get_env::<C>()
            .expect("must register container type before registering a value");
        if let Some(env) = env.as_any().downcast_ref::<ContainerEnv<C>>() {
            env.get_or_insert(&container, exec_state)
        } else {
            env.as_any()
                .downcast_ref::<SequenceContainerEnv<C>>()
                .expect("container environment has the wrong concrete value type")
                .get_or_insert(&container, exec_state)
        }
    }

    /// Return the committed flat value sequence for a sequence-backed
    /// container. This is the fast read path and performs no Rust-container
    /// reconstruction.
    pub fn get_sequence<C: ContainerValue>(&self, value: Value) -> Option<&[Value]> {
        self.get_sequence_env::<C>()?.get_values(value)
    }

    /// Rebuild a single container value by remapping each contained value
    /// through `remap`, returning the (possibly new) interned value, or `value`
    /// unchanged if it is not a registered container of the type behind
    /// `type_id`.
    ///
    /// Unlike [`ContainerValues::rebuild_all`], which drives rebuilds off the
    /// backend union-find, the caller supplies the remapping explicitly and
    /// identifies the container type dynamically by its [`TypeId`].
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

    /// Apply the given rebuild to the contents of each container.
    pub fn rebuild_all(
        &mut self,
        table_id: TableId,
        table: &WrappedTable,
        exec_state: &mut ExecutionState,
    ) -> ContainerRebuildSummary {
        let Some(context) = self.prepare_rebuild(table_id, table) else {
            return Default::default();
        };
        let mut summary = self.rebuild_sequence_pass(&context, exec_state);
        summary.extend(self.rebuild_legacy_pass(&context, exec_state));
        self.finish_rebuild(&mut summary);
        summary
    }

    /// Prepare the immutable UF inputs shared by the backend-specific passes
    /// in one container rebuild cycle.
    fn prepare_rebuild<'a>(
        &mut self,
        table_id: TableId,
        table: &'a WrappedTable,
    ) -> Option<ContainerRebuildContext<'a>> {
        let rebuilder = table.rebuilder(&[])?;
        let to_scan = rebuilder.hint_col().map(|_| {
            // We may attempt an incremental rebuild. This must be computed
            // exactly once and shared by both backend passes.
            self.subset_tracker.recent_updates(table_id, table)
        });
        Some(ContainerRebuildContext {
            table,
            rebuilder,
            to_scan,
        })
    }

    /// Rebuild only sequence-backed container environments.
    fn rebuild_sequence_pass(
        &mut self,
        context: &ContainerRebuildContext<'_>,
        exec_state: &mut ExecutionState,
    ) -> ContainerRebuildSummary {
        self.rebuild_backend_pass(ContainerBackend::Sequence, context, exec_state)
    }

    /// Rebuild only legacy container environments.
    fn rebuild_legacy_pass(
        &mut self,
        context: &ContainerRebuildContext<'_>,
        exec_state: &mut ExecutionState,
    ) -> ContainerRebuildSummary {
        self.rebuild_backend_pass(ContainerBackend::Legacy, context, exec_state)
    }

    fn rebuild_backend_pass(
        &mut self,
        backend: ContainerBackend,
        context: &ContainerRebuildContext<'_>,
        exec_state: &mut ExecutionState,
    ) -> ContainerRebuildSummary {
        let (selected, selected_rows, largest_env) = self
            .data
            .iter()
            .filter(|(_, env)| env.backend() == backend)
            .fold(
                (0usize, 0usize, 0usize),
                |(count, rows, largest), (_, env)| {
                    let len = env.len();
                    (count + 1, rows.saturating_add(len), largest.max(len))
                },
            );
        if selected == 0 {
            return ContainerRebuildSummary::default();
        }

        // Sequence environments are themselves tables, so a few large types
        // are enough useful work to parallelize even when the legacy
        // environment-count heuristic does not fire. Keep the aggregate-size
        // alternative sequence-only, and use it only while every individual
        // sequence scan remains serial. This avoids nesting outer environment
        // tasks around tables that already use all workers internally; legacy
        // environments likewise retain their existing nested rebuild policy.
        let parallelize_large_sequence_batch = backend == ContainerBackend::Sequence
            && selected > 1
            && !parallelize_rebuild(largest_env)
            && parallelize_db_level_op(selected_rows);
        if parallelize_inter_container_op(selected) || parallelize_large_sequence_batch {
            parallel::map_dense_id_map_mut(&mut self.data, |_, env| {
                if env.backend() != backend {
                    return ContainerRebuildSummary::default();
                }
                let mut exec_state = exec_state.clone();
                env.apply_rebuild(
                    context.table,
                    &*context.rebuilder,
                    context.to_scan.as_ref().map(|subset| subset.as_ref()),
                    &mut exec_state,
                )
            })
            .into_iter()
            .fold(ContainerRebuildSummary::default(), |mut acc, summary| {
                acc.extend(summary);
                acc
            })
        } else {
            let mut summary = ContainerRebuildSummary::default();
            for (_, env) in self.data.iter_mut() {
                if env.backend() != backend {
                    continue;
                }
                summary.extend(env.apply_rebuild(
                    context.table,
                    &*context.rebuilder,
                    context.to_scan.as_ref().map(|subset| subset.as_ref()),
                    exec_state,
                ));
            }
            summary
        }
    }

    /// Finish a split container rebuild by propagating stable semantic changes
    /// through all containing containers.
    ///
    /// Call this once, after every backend-specific pass has completed and all
    /// temporarily removed environments have been restored.
    fn finish_rebuild(&self, summary: &mut ContainerRebuildSummary) {
        self.expand_dirty_id_closure(summary);
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
    fn expand_dirty_id_closure(&self, summary: &mut ContainerRebuildSummary) {
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

    /// Add a new container type to the given [`ContainerValue`] instance.
    ///
    /// Container types need a meaans of generating fresh ids (`id_counter`) along with a means of
    /// merging conflicting ids (`merge_fn`).
    pub fn register_type<C: ContainerValue>(
        &mut self,
        id_counter: CounterId,
        merge_fn: impl MergeFn + 'static,
    ) -> ContainerValueId {
        let id = self.container_ids.insert(TypeId::of::<C>());
        self.data.get_or_insert(id, || {
            Box::new(ContainerEnv::<C>::new(Box::new(merge_fn), id_counter))
        });
        id
    }

    /// Register a container type whose canonical representation is a
    /// variable-length sequence followed by one non-key identity value.
    pub(crate) fn register_sequence_type<C: SequenceContainerValue>(
        &mut self,
        id_counter: CounterId,
        merge_fn: impl MergeFn + 'static,
    ) -> ContainerValueId {
        let id = self.container_ids.insert(TypeId::of::<C>());
        self.data.get_or_insert(id, || {
            Box::new(SequenceContainerEnv::<C>::new(
                id,
                Box::new(merge_fn),
                id_counter,
            ))
        });
        assert!(
            self.data[id]
                .as_any()
                .downcast_ref::<SequenceContainerEnv<C>>()
                .is_some(),
            "container type was already registered with a different backend"
        );
        id
    }

    /// Return the storage backend for a registered type-erased environment.
    #[cfg(test)]
    pub(crate) fn env_backend(&self, id: ContainerValueId) -> ContainerBackend {
        self.data[id].backend()
    }

    /// Return the current number of values in a registered environment.
    #[cfg(test)]
    pub(crate) fn env_len(&self, id: ContainerValueId) -> usize {
        self.data[id].len()
    }

    /// Return the combined number of committed sequence-backed rows.
    pub(crate) fn sequence_len(&self) -> usize {
        self.data
            .iter()
            .filter(|(_, env)| env.backend() == ContainerBackend::Sequence)
            .map(|(_, env)| env.len())
            .sum()
    }

    /// Snapshot the ids of all sequence-backed environments.
    #[cfg(test)]
    pub(crate) fn sequence_env_ids(&self) -> Vec<ContainerValueId> {
        self.data
            .iter()
            .filter_map(|(id, env)| (env.backend() == ContainerBackend::Sequence).then_some(id))
            .collect()
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
    /// Sequence-backed containers are reconstructed from their flat key.
    pub fn get_container<C: ContainerValue>(&self, value: Value) -> Option<C> {
        if let Some(env) = self.db.containers.get_sequence_env::<C>() {
            env.get_container_with_predictions(self, value)
        } else {
            self.db.containers.get_owned::<C>(value)
        }
    }

    /// Borrow the fast serialized payload for a sequence-backed container
    /// visible to this execution. No Rust container is constructed. The
    /// payload may include type-specific metadata, such as compact counts.
    pub fn container_sequence<C: ContainerValue>(&self, value: Value) -> Option<&[Value]> {
        self.db
            .containers
            .get_sequence_env::<C>()?
            .get_values_with_predictions(self, value)
    }

    /// Intern an already serialized key for a sequence-backed container.
    ///
    /// This is the fast write path. Callers avoid constructing a Rust
    /// container but remain responsible for producing that type's canonical
    /// key format.
    pub fn register_container_sequence<C: ContainerValue>(&mut self, key: &[Value]) -> Value {
        let containers = self.db.containers;
        let env = containers
            .get_sequence_env::<C>()
            .expect("container type is not registered with the sequence backend");
        env.get_or_insert_key(key, self)
    }
}

/// A trait implemented by container types.
///
/// Containers behave a lot like base values, but they include extra trait methods to support
/// rebuilding of container contents and merging containers that become equal after a rebuild pass
/// has taken place.
pub trait ContainerValue: Hash + Eq + Clone + Send + Sync + 'static {
    /// Rebuild an additional container in place according the the given [`ValueRebuilder`].
    ///
    /// If this method returns `false` then the container must not have been modified (i.e. it must
    /// hash to the same value, and compare equal to a copy of itself before the call).
    fn rebuild_contents(&mut self, rebuilder: &dyn ValueRebuilder) -> bool;

    /// Iterate over the contents of the container.
    ///
    /// Note that containers can be more structured than just a sequence of values. This iterator
    /// is used to populate an index that in turn is used to speed up rebuilds. If a value in the
    /// container is eligible for a rebuild and it is not mentioned by this iterator, the outer
    /// container registry may skip rebuilding this container.
    fn iter(&self) -> impl Iterator<Item = Value> + '_;
}

pub(crate) trait DynamicContainerEnv: Any + dyn_clone::DynClone + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn backend(&self) -> ContainerBackend;
    fn len(&self) -> usize;
    /// Publish one epoch of staged sequence-table mutations.
    fn merge_pending(&mut self, _exec_state: &mut ExecutionState) -> bool {
        false
    }
    fn apply_rebuild(
        &mut self,
        table: &WrappedTable,
        rebuilder: &dyn Rebuilder,
        subset: Option<SubsetRef>,
        exec_state: &mut ExecutionState,
    ) -> ContainerRebuildSummary;
    /// Add ids for containers in this environment that contain any `values`.
    ///
    /// This uses the container content index populated from
    /// [`ContainerValue::iter`] and lets callers climb from dirty child ids to
    /// all directly containing parent container ids.
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

fn hash_container(container: &impl ContainerValue) -> u64 {
    let mut hasher = FxHasher::default();
    container.hash(&mut hasher);
    hasher.finish()
}

#[derive(Clone)]
struct ContainerEnv<C: Eq + Hash> {
    merge_fn: Box<dyn MergeFn>,
    counter: CounterId,
    to_id: DashMap<C, Value>,
    to_container: DashMap<Value, (usize /* hash code */, usize /* map */)>,
    /// Map from a Value to the set of ids of containers that contain that value.
    val_index: DashMap<Value, IndexSet<Value>>,
}

impl<C: ContainerValue> DynamicContainerEnv for ContainerEnv<C> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn backend(&self) -> ContainerBackend {
        ContainerBackend::Legacy
    }

    fn len(&self) -> usize {
        self.to_id.len()
    }

    fn apply_rebuild(
        &mut self,
        table: &WrappedTable,
        rebuilder: &dyn Rebuilder,
        subset: Option<SubsetRef>,
        exec_state: &mut ExecutionState,
    ) -> ContainerRebuildSummary {
        if let Some(subset) = subset
            && incremental_rebuild(
                subset.size(),
                self.to_id.len(),
                parallelize_intra_container_op(self.to_id.len()),
            )
        {
            return self.apply_rebuild_incremental(
                table,
                rebuilder,
                exec_state,
                subset,
                rebuilder.hint_col().unwrap(),
            );
        }
        self.apply_rebuild_nonincremental(rebuilder, exec_state)
    }

    fn extend_containers_containing(&self, values: &IndexSet<Value>, out: &mut IndexSet<Value>) {
        for value in values {
            if let Some(containers) = self.val_index.get(value) {
                out.extend(containers.iter().copied());
            }
        }
    }

    fn rebuild_val_with(
        &self,
        value: Value,
        exec_state: &mut ExecutionState,
        remap: &(dyn Fn(Value) -> Value + Send + Sync),
    ) -> Option<Value> {
        // Clone out of the guard before re-interning to avoid deadlocking on
        // the underlying map.
        let mut container = self.get_container(value)?.clone();
        container.rebuild_contents(&ClosureRebuilder { remap });
        Some(self.get_or_insert(&container, exec_state))
    }
}

impl<C: ContainerValue> ContainerEnv<C> {
    pub fn new(merge_fn: Box<dyn MergeFn>, counter: CounterId) -> Self {
        Self {
            merge_fn,
            counter,
            to_id: DashMap::default(),
            to_container: DashMap::default(),
            val_index: DashMap::default(),
        }
    }

    fn get_or_insert(&self, container: &C, exec_state: &mut ExecutionState) -> Value {
        if let Some(value) = self.to_id.get(container) {
            return *value;
        }

        // Time to insert a new mapping. First, insert into `to_container`: the moment that we
        // insert a new value into `to_id`, someone else can return it from another call to
        // `get_or_insert` and then feed that value to `get_container`.

        let value = Value::from_usize(exec_state.inc_counter(self.counter));
        let target_map = self.to_id.determine_map(container);
        // This assertion is here because in parallel rebuilding we use `to_container` to
        // compute the intended shard for to_id, because we have a mutable borrow of
        // `to_container` that means we cannot call `determine_map` on `to_id`.
        debug_assert_eq!(
            target_map,
            self.to_container
                .determine_shard(hash_container(container) as usize)
        );
        self.to_container
            .insert(value, (hash_container(container) as usize, target_map));

        // Now insert into `to_id`, handling the case where a different thread is doing the same
        // thing.
        match self.to_id.entry(container.clone()) {
            dashmap::Entry::Vacant(vac) => {
                // Common case: insert the mapping in to_id and update the index.
                vac.insert(value);
                for val in container.iter() {
                    self.val_index.entry(val).or_default().insert(value);
                }
                value
            }
            dashmap::Entry::Occupied(occ) => {
                // Someone inserted `container` into the mapping since we looked it up. Remove the
                // mapping that we inserted into `to_container` (we won't use it), and instead
                // return the "winning" value.
                let res = *occ.get();
                std::mem::drop(occ); // drop the lock.
                self.to_container.remove(&value);
                res
            }
        }
    }

    fn insert_owned(&self, container: C, value: Value, exec_state: &mut ExecutionState) -> Value {
        let hc = hash_container(&container);
        let target_map = self.to_id.determine_map(&container);
        match self.to_id.entry(container) {
            dashmap::Entry::Occupied(mut occ) => {
                let result = (self.merge_fn)(exec_state, *occ.get(), value);
                let old_val = *occ.get();
                if result != old_val {
                    self.to_container.remove(&old_val);
                    self.to_container.insert(result, (hc as usize, target_map));
                    *occ.get_mut() = result;
                    for val in occ.key().iter() {
                        let mut index = self.val_index.entry(val).or_default();
                        index.swap_remove(&old_val);
                        index.insert(result);
                    }
                }
                result
            }
            dashmap::Entry::Vacant(vacant_entry) => {
                self.to_container.insert(value, (hc as usize, target_map));
                for val in vacant_entry.key().iter() {
                    self.val_index.entry(val).or_default().insert(value);
                }
                vacant_entry.insert(value);
                value
            }
        }
    }

    fn reinsert_incremental(
        &self,
        container: C,
        old_id: Value,
        rebuilt_id: Value,
        container_changed: bool,
        exec_state: &mut ExecutionState,
        summary: &mut ContainerRebuildSummary,
    ) {
        if container_changed || rebuilt_id != old_id {
            summary.note_change();
        }
        if rebuilt_id != old_id {
            // Parent rows will get a real delta from ordinary table rebuild, so
            // we only need an explicit refresh when the outer id stayed stable.
            self.to_container.remove(&old_id);
        }
        let actual = self.insert_owned(container, rebuilt_id, exec_state);
        if container_changed && rebuilt_id == old_id && actual == old_id {
            summary.note_dirty_id(old_id);
        }
    }

    fn apply_rebuild_incremental(
        &mut self,
        table: &WrappedTable,
        rebuilder: &dyn Rebuilder,
        exec_state: &mut ExecutionState,
        to_scan: SubsetRef,
        search_col: ColumnId,
    ) -> ContainerRebuildSummary {
        // NB: there is no parallel implementation as of now.
        //
        // Implementing one should be straightforward, but we should wait for a real benchmark that
        // requires it. It's possible that incremental rebuilding will only be profitable when the
        // total number of ids to rebuild is small, in which case the overhead of parallelism may
        // not be worth it in the first place.
        let mut summary = ContainerRebuildSummary::default();
        let mut buf = TaggedRowBuffer::new(1);
        table.scan_project(
            to_scan,
            &[search_col],
            Offset::new(0),
            usize::MAX,
            &[],
            &mut buf,
        );
        // For each value in the buffer, rebuild all containers that mention it.
        let mut to_rebuild = IndexSet::<Value>::default();
        for (_, row) in buf.iter() {
            to_rebuild.insert(row[0]);
            let Some(ids) = self.val_index.get(&row[0]) else {
                continue;
            };
            to_rebuild.extend(&*ids);
        }
        for id in to_rebuild {
            let Some((hc, target_map)) = self.to_container.get(&id).map(|x| *x) else {
                continue;
            };
            let shard_mut = self.to_id.shards_mut()[target_map].get_mut();
            let Some((mut container, _)) =
                shard_mut.remove_entry(hc as u64, |(_, v)| *v.get() == id)
            else {
                continue;
            };
            let rebuilt_id = rebuilder.rebuild_val(id);
            let container_changed = container.rebuild_contents(rebuilder);
            self.reinsert_incremental(
                container,
                id,
                rebuilt_id,
                container_changed,
                exec_state,
                &mut summary,
            );
        }
        summary
    }

    fn apply_rebuild_nonincremental(
        &mut self,
        rebuilder: &dyn Rebuilder,
        exec_state: &mut ExecutionState,
    ) -> ContainerRebuildSummary {
        if parallelize_inter_container_op(self.to_id.len()) {
            return self.apply_rebuild_nonincremental_parallel(rebuilder, exec_state);
        }
        let mut summary = ContainerRebuildSummary::default();
        let mut to_reinsert = Vec::new();
        let shards = self.to_id.shards_mut();
        for shard in shards.iter_mut() {
            let shard = shard.get_mut();
            // SAFETY: the iterator does not outlive `shard`.
            for bucket in unsafe { shard.iter() } {
                // SAFETY: the bucket is valid; we just got it from the iterator.
                let (container, val) = unsafe { bucket.as_mut() };
                let old_val = *val.get();
                let new_val = rebuilder.rebuild_val(old_val);
                let container_changed = container.rebuild_contents(rebuilder);
                if !container_changed && new_val == old_val {
                    // Nothing changed about this entry. Leave it in place.
                    continue;
                }
                summary.note_change();
                if container_changed {
                    // The container changed. Remove both map entries then reinsert.
                    // SAFETY: This is a valid bucket. Furthermore, iterators remain valid if
                    // buckets they have already yielded have been removed.
                    let ((container, _), _) = unsafe { shard.remove(bucket) };
                    self.to_container.remove(&old_val);
                    to_reinsert.push((container, new_val, new_val == old_val));
                } else {
                    // Just the value changed. Leave the container in place.
                    *val.get_mut() = new_val;
                    let prev = self.to_container.remove(&old_val).unwrap().1;
                    self.to_container.insert(new_val, prev);
                }
            }
        }
        for (container, val, stable_id) in to_reinsert {
            let actual = self.insert_owned(container, val, exec_state);
            // Refresh only when rebuild changed container semantics in place.
            // If the outer id changed, ordinary table rebuild already creates a
            // fresh parent-row delta for seminaive to follow.
            if stable_id && actual == val {
                summary.note_dirty_id(val);
            }
        }
        summary
    }

    fn apply_rebuild_nonincremental_parallel(
        &mut self,
        rebuilder: &dyn Rebuilder,
        exec_state: &mut ExecutionState,
    ) -> ContainerRebuildSummary {
        // This is very similar to the serial variant. The main difference is that
        // `to_reinsert` isn't a flat vector. It's instead a vector of queues - one per
        // destination map shard. This lets us do a bulk insertion in parallel without having
        // to grab a lock per container.
        let mut to_reinsert =
            IdVec::<usize /* to_id shard */, SegQueue<(C, Value, bool)>>::default();
        to_reinsert.resize_with(self.to_id.shards().len(), Default::default);

        let shards = self.to_id.shards_mut();
        let changed = parallel::map_mut(shards, |_, shard| {
            let mut changed = false;
            let shard = shard.get_mut();
            // SAFETY: the iterator does not outlive `shard`.
            for bucket in unsafe { shard.iter() } {
                // SAFETY: the bucket is valid; we just got it from the iterator.
                let (container, val) = unsafe { bucket.as_mut() };
                let old_val = *val.get();
                let new_val = rebuilder.rebuild_val(old_val);
                let container_changed = container.rebuild_contents(rebuilder);
                if !container_changed && new_val == old_val {
                    // Nothing changed about this entry. Leave it in place.
                    continue;
                }
                changed = true;
                if container_changed {
                    // The container changed. Remove both map entries then reinsert.
                    // SAFETY: This is a valid bucket. Furthermore, iterators remain valid if
                    // buckets they have already yielded have been removed.
                    let ((container, _), _) = unsafe { shard.remove(bucket) };
                    self.to_container.remove(&old_val);
                    // Spooky: we're using `to_container` to determine the shard for
                    // `to_id`. We are assuming that the # shards determination is
                    // deterministic here. There is a debug assertion in `get_or_insert`
                    // that attempts to verify this.
                    let shard = self
                        .to_container
                        .determine_shard(hash_container(&container) as usize);
                    to_reinsert[shard].push((container, new_val, new_val == old_val));
                } else {
                    // Just the value changed. Leave the container in place.
                    *val.get_mut() = new_val;
                    let prev = self.to_container.remove(&old_val).unwrap().1;
                    self.to_container.insert(new_val, prev);
                }
            }
            changed
        })
        .into_iter()
        .any(|changed| changed);

        let dirty_ids = SegQueue::new();
        parallel::for_each_mut(shards, |shard_id, shard| {
            let mut exec_state = exec_state.clone();
            // This bit is a real slog. Once Dashmap updates from RawTable to HashTable for
            // the underlying shard, this will get a little better.
            //
            // NB: We are probably leaving some paralellism on the floor with these calls
            // to `to_container` and `val_index`.
            let shard = shard.get_mut();
            let queue = &to_reinsert[shard_id];
            while let Some((container, val, stable_id)) = queue.pop() {
                let hc = hash_container(&container);
                let target_map = self.to_container.determine_shard(hc as usize);
                match shard.find_or_find_insert_slot(
                    hc,
                    |(c, _)| c == &container,
                    |(c, _)| hash_container(c),
                ) {
                    Ok(bucket) => {
                        // SAFETY: the bucket is valid; we just got it from the shard and
                        // we have not done any operations that can invalidate the bucket.
                        let (container, val_slot) = unsafe { bucket.as_mut() };
                        let old_val = *val_slot.get();
                        let result = (self.merge_fn)(&mut exec_state, old_val, val);
                        if result != old_val {
                            self.to_container.remove(&old_val);
                            self.to_container.insert(result, (hc as usize, target_map));
                            *val_slot.get_mut() = result;
                            for val in container.iter() {
                                let mut index = self.val_index.entry(val).or_default();
                                index.swap_remove(&old_val);
                                index.insert(result);
                            }
                        }
                        // As in the serial path, only same-id semantic
                        // changes need an explicit parent-row refresh.
                        if stable_id && result == val {
                            dirty_ids.push(val);
                        }
                    }
                    Err(slot) => {
                        self.to_container.insert(val, (hc as usize, target_map));
                        for v in container.iter() {
                            self.val_index.entry(v).or_default().insert(val);
                        }
                        // SAFETY: We just got this slot from `find_or_find_insert_slot`
                        // and we have not mutated the map at all since then.
                        unsafe {
                            shard.insert_in_slot(hc, slot, (container, SharedValue::new(val)));
                        }
                        if stable_id {
                            dirty_ids.push(val);
                        }
                    }
                }
            }
        });
        let mut summary = ContainerRebuildSummary::default();
        if changed {
            summary.note_change();
        }
        while let Some(value) = dirty_ids.pop() {
            summary.note_dirty_id(value);
        }
        summary
    }

    fn get_container(&self, value: Value) -> Option<impl Deref<Target = C> + '_> {
        let (hc, target_map) = *self.to_container.get(&value)?;
        let shard = &self.to_id.shards()[target_map];
        let read_guard = shard.read();
        let val_ptr: *const (C, _) = shard
            .read()
            .find(hc as u64, |(_, v)| *v.get() == value)?
            .as_ptr();
        struct ValueDeref<'a, T, Guard> {
            _guard: Guard,
            data: &'a T,
        }

        impl<T, Guard> Deref for ValueDeref<'_, T, Guard> {
            type Target = T;

            fn deref(&self) -> &T {
                self.data
            }
        }

        Some(ValueDeref {
            _guard: read_guard,
            // SAFETY: the value will remain valid for as long as `read_guard` is in scope.
            data: unsafe {
                let unwrapped: &(C, _) = &*val_ptr;
                &unwrapped.0
            },
        })
    }
}

fn incremental_rebuild(uf_size: usize, table_size: usize, parallel: bool) -> bool {
    if parallel {
        table_size > 1000 && uf_size * 512 <= table_size
    } else {
        table_size > 1000 && uf_size * 8 <= table_size
    }
}

/// A [`ValueRebuilder`] that remaps individual values through a caller-supplied
/// closure. Used by [`ContainerValues::rebuild_val_with`] to rebuild a single
/// container against an explicit value mapping rather than a backend union-find.
struct ClosureRebuilder<'a> {
    remap: &'a (dyn Fn(Value) -> Value + Send + Sync),
}

impl ValueRebuilder for ClosureRebuilder<'_> {
    fn rebuild_val(&self, val: Value) -> Value {
        (self.remap)(val)
    }
}
