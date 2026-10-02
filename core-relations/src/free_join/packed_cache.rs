//! Trie state shared across the plans of one rule-set execution.
//!
//! Plans that constrain the same table with the same fast constraints share
//! one [`TrieRoot`]. Everything built below a shared root is shared as well:
//! its scalar projections, the continuation grids of its persistent catalog
//! indexes, and every packed descendant node. Descendants are published under
//! table-wide [`FamilyId`]s, so two plans that reach the same rows and index
//! the same next column with the same constraints build that index once.

use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicUsize, Ordering},
};

use dashmap::mapref::entry::Entry;
use smallvec::SmallVec;

use crate::{
    common::{DashMap, HashMap, HashSet, Value},
    numeric_id::{NumericId, define_id},
    offsets::{OffsetRange, RowId, SortedOffsetSlice, Subset, SubsetRef},
    table_spec::{ColumnId, Constraint},
};

use super::{AtomId, ColumnIds, TableId, plan::Plan, prepared_index::RootContinuationCache};

/// Canonical identity of the rows available at one atom's trie root: the
/// atom's table together with the sorted conjunction of its fast (header)
/// constraints.
///
/// For example, if `edge` has columns `(src, dst)`, the atom `edge(x, y)` has
/// signature `(edge, [])`, while `edge(x, 3)` has signature
/// `(edge, [dst == 3])`. Two plans with the latter atom share the same owning
/// root subset even if their later join stages differ. Sorting makes
/// `src > 0 && dst == 3` identical to `dst == 3 && src > 0`; including the
/// table prevents the same constraints on another relation from sharing a
/// root.
type RootSignature = (TableId, SmallVec<[Constraint; 2]>);

define_id!(
    pub(super) HeaderConstraintId,
    u32,
    "an execution-local id for a canonical set of fast trie-root constraints"
);

define_id!(
    pub(crate) FamilyId,
    u32,
    r#"Identity of one way to index the rows below a shared trie node of a
table: the next column together with the sorted slow constraints applied
before projecting it.

Two plans that descend from the same shared node with the same family build
the same child index, so the child is published once per family. Family ids
are dense per table and are interned by the table when a plan is built, so a
run sees a fixed family count (see `TableInfo::shared_child_shape`)."#
);

/// The families of every column of one indexed access: `[0]` indexes the
/// first column after applying the scan's constraints, and `[i]` indexes
/// column `i` of the same rows.
pub(crate) type AccessFamilies = SmallVec<[FamilyId; 4]>;

/// Key for a shared trie root: the table plus an interned id for its fast
/// (header) constraints.
///
/// Only fast constraints are interned here; they have already been applied to
/// the root subset. Plans with the same fast constraints can share this root
/// even when their remaining slow constraints differ.
type RootKey = (TableId, HeaderConstraintId);

/// The canonical key of a [`FamilyId`]: a column and sorted constraints.
pub(crate) type SuccessorSig = (ColumnId, SmallVec<[Constraint; 2]>);

pub(crate) fn canonical_constraints(constraints: &[Constraint]) -> SmallVec<[Constraint; 2]> {
    let mut canonical: SmallVec<[Constraint; 2]> = constraints.iter().cloned().collect();
    canonical.sort_unstable();
    canonical
}

/// One round-local root projection can be reused by plans that share the same
/// root subset. Slow constraints are part of the key because they are applied
/// before projection; sorting them makes conjunction order irrelevant.
#[derive(Clone, Eq, Hash, PartialEq)]
struct RootProjectionKey {
    column: ColumnId,
    /// Only the scan's remaining slow constraints, applied before projection.
    /// Fast constraints are represented by the owning root's [`RootKey`].
    constraints: SmallVec<[Constraint; 2]>,
}

pub(super) struct RootProjection {
    /// Final immutable scalar-index representation. Unlike the earlier pair
    /// cache, this is probed directly: queries do not copy it into their arenas.
    /// The trailing entry is an offset-only sentinel.
    keys: Box<[(Value, u32)]>,
    rows: Box<[RowId]>,
}

impl RootProjection {
    pub(super) fn from_sorted_pairs(pairs: Vec<(Value, RowId)>) -> Self {
        debug_assert!(pairs.windows(2).all(|pair| pair[0] <= pair[1]));
        let distinct = pairs
            .iter()
            .enumerate()
            .filter(|(index, pair)| *index == 0 || pairs[*index - 1].0 != pair.0)
            .count();
        let mut keys = Vec::with_capacity(distinct + 1);
        let mut rows = Vec::with_capacity(pairs.len());
        for (value, row) in pairs {
            if keys.last().map(|&(key, _)| key) != Some(value) {
                keys.push((
                    value,
                    u32::try_from(rows.len())
                        .expect("a projected root index cannot contain more than u32::MAX rows"),
                ));
            }
            rows.push(row);
        }
        keys.push((
            Value::new_const(0),
            u32::try_from(rows.len())
                .expect("a projected root index cannot contain more than u32::MAX rows"),
        ));
        Self {
            keys: keys.into_boxed_slice(),
            rows: rows.into_boxed_slice(),
        }
    }

    pub(super) fn len(&self) -> usize {
        self.keys.len().saturating_sub(1)
    }

    pub(super) fn find(&self, value: Value) -> Option<usize> {
        let len = self.len();
        self.keys[..len]
            .binary_search_by_key(&value, |&(key, _)| key)
            .ok()
    }

    pub(super) fn value_at(&self, key_index: usize) -> Value {
        assert!(key_index < self.len(), "projected root key out of bounds");
        self.keys[key_index].0
    }

    pub(super) fn subset_at(&self, key_index: usize) -> SubsetRef<'_> {
        assert!(key_index < self.len(), "projected root key out of bounds");
        let start = self.keys[key_index].1 as usize;
        let end = self.keys[key_index + 1].1 as usize;
        let rows = &self.rows[start..end];
        debug_assert!(!rows.is_empty());
        let first = rows[0];
        let last = rows[rows.len() - 1];
        if last.index() - first.index() == rows.len() - 1 {
            SubsetRef::Dense(OffsetRange::new(first, last.inc()))
        } else {
            // SAFETY: construction consumes pairs sorted by `(Value, RowId)`,
            // so every equal-value range is RowId ordered.
            SubsetRef::Sparse(unsafe { SortedOffsetSlice::new_unchecked(rows) })
        }
    }
}

/// A shared root projection together with the continuation grid that
/// publishes the shared packed index below each of its keys.
#[derive(Default)]
pub(super) struct RootProjectionEntry {
    pub(super) projection: OnceLock<RootProjection>,
    pub(super) continuations: RootContinuationCache,
}

pub(super) type RootProjectionSlot = Arc<RootProjectionEntry>;
type RootProjectionMap = DashMap<RootProjectionKey, RootProjectionSlot>;
type CatalogContinuationMap = DashMap<ColumnIds, Arc<RootContinuationCache>>;

/// A cache of trie roots shared across all plans within a single
/// `run_rule_set` call. Two plans that constrain the same table with the same
/// fast constraints share the owning root subset and every index built below
/// it; see [`FamilyId`] for how descendants are identified.
///
/// Only roots that more than one plan actually uses are shared (`shared`), so
/// single-use roots stay per-plan and keep the pool-recycling behavior of the
/// unshared path — sharing a root that is never reused is pure overhead.
///
/// The `DashMap`s are setup caches rather than per-row probe structures. A plan
/// consults `roots` once while initializing each reused atom root; a single-use
/// root bypasses the cache completely. Likewise, the projection and
/// continuation maps are consulted only when a prepared access first acquires
/// its slot. That slot is retained in `PreparedIndexState`, and the hot
/// recursive probe path reads the resulting immutable arrays directly.
/// Contention is therefore limited to single-flight construction when plans
/// initialize the same root or index concurrently. Tables are frozen during a
/// run, so each key continues to denote the same subset after publication.
#[derive(Default)]
pub(super) struct TrieCache {
    pub(super) roots: DashMap<RootKey, Arc<TrieRoot>>,
    /// Interns canonical header-constraint sets to keep [`RootKey`] cheap.
    /// The table stays outside the id and remains the first part of `RootKey`.
    header_ids: DashMap<SmallVec<[Constraint; 2]>, HeaderConstraintId>,
    next_header_id: AtomicUsize,
    /// Root signatures used by more than one plan; only these are shared.
    pub(super) shared: HashSet<RootSignature>,
}

impl TrieCache {
    /// Return the interned id for a canonical set of fast header constraints.
    ///
    /// Id 0 is reserved for the common unconstrained case, so those atoms skip
    /// the interning map entirely. [`RootKey`] carries the table separately;
    /// identical constraint sets may therefore reuse an id across tables
    /// without making the roots alias.
    pub(super) fn header_id(&self, fast: &[Constraint]) -> HeaderConstraintId {
        if fast.is_empty() {
            return HeaderConstraintId::new_const(0);
        }
        let sig = canonical_constraints(fast);
        match self.header_ids.entry(sig) {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(v) => {
                let id = HeaderConstraintId::from_usize(
                    self.next_header_id.fetch_add(1, Ordering::Relaxed) + 1,
                );
                v.insert(id);
                id
            }
        }
    }

    /// The canonical root signature (table + sorted fast constraints) for `atom`
    /// given its headers.
    fn root_sig(plan: &Plan, atom: AtomId, table: TableId) -> RootSignature {
        let mut fast: SmallVec<[Constraint; 2]> = SmallVec::new();
        for h in plan.header().iter().filter(|h| h.atom == atom) {
            fast.extend(h.constraints.iter().cloned());
        }
        fast.sort_unstable();
        (table, fast)
    }

    /// Compute the set of root signatures used by more than one plan atom (across
    /// all plans); only these are worth sharing.
    pub(super) fn compute_shared<'a>(
        plans: impl Iterator<Item = &'a Plan>,
    ) -> HashSet<RootSignature> {
        let mut counts: HashMap<RootSignature, u32> = HashMap::default();
        for plan in plans {
            for (atom, info) in plan.atoms().iter() {
                *counts
                    .entry(Self::root_sig(plan, atom, info.table))
                    .or_default() += 1;
            }
        }
        counts
            .into_iter()
            .filter_map(|(sig, n)| (n > 1).then_some(sig))
            .collect()
    }

    /// Build a cache for the given shared root signatures. Only called when
    /// `shared` is non-empty, so the DashMap allocations always pay off.
    ///
    /// Shard the maps to the actual thread count rather than DashMap's default
    /// (`4 * num_cpus`): on a many-core host the default allocates hundreds of
    /// shards per `run_rule_set`, which dwarfs the sharing savings on smaller
    /// runs. Serial runs get a single shard.
    pub(super) fn with_shared(shared: HashSet<RootSignature>) -> TrieCache {
        TrieCache {
            roots: DashMap::with_hasher_and_shard_amount(Default::default(), dashmap_shards()),
            header_ids: DashMap::with_hasher_and_shard_amount(Default::default(), dashmap_shards()),
            next_header_id: AtomicUsize::new(0),
            shared,
        }
    }
}

/// DashMap requires at least 2 shards; that is plenty for serial runs and
/// still far below the default (4 * num_cpus).
fn dashmap_shards() -> usize {
    crate::parallel::current_num_threads()
        .next_power_of_two()
        .max(2)
}

/// Lazily created maps that publish shared indexes below a shared root.
#[derive(Default)]
struct SharedRootIndexes {
    projections: OnceLock<RootProjectionMap>,
    catalog_continuations: OnceLock<CatalogContinuationMap>,
}

/// Owning row subset for an atom. Lower trie levels are execution-scoped
/// packed nodes; below a shared root they are shared across plans.
///
/// A plan root holds the atom's header-filtered rows for a whole plan
/// execution. A residual root holds the rows left by one constrained probe
/// and belongs to a single frame.
pub(crate) struct TrieRoot {
    pub(super) subset: Subset,
    /// Present only for roots shared across plans.
    shared: Option<SharedRootIndexes>,
    plan_root: bool,
}

impl std::fmt::Debug for TrieRoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrieRoot")
            .field("subset", &self.subset)
            .finish()
    }
}

impl TrieRoot {
    pub(super) fn new(subset: Subset) -> Self {
        Self {
            subset,
            shared: None,
            plan_root: true,
        }
    }

    pub(super) fn new_shared(subset: Subset) -> Self {
        Self {
            subset,
            shared: Some(SharedRootIndexes::default()),
            plan_root: true,
        }
    }

    /// Whether plans starting from this root share the indexes below it.
    pub(super) fn is_shared(&self) -> bool {
        self.shared.is_some()
    }

    /// A frame-local residual that no plan-level slot may cache.
    pub(super) fn new_residual(subset: Subset) -> Self {
        Self {
            subset,
            shared: None,
            plan_root: false,
        }
    }

    /// Whether this root is the atom's rows for the whole plan execution, so
    /// per-plan state keyed by the atom may describe it.
    pub(super) fn is_plan_root(&self) -> bool {
        self.plan_root
    }

    /// Find the shared slot for projecting `column` after applying the scan's
    /// remaining slow `constraints`. The root subset already satisfies its
    /// fast (header) constraints, so callers must not include them here.
    /// Different slow constraints on the same root and column require
    /// separate projections.
    pub(super) fn projection_slot(
        &self,
        column: ColumnId,
        constraints: &[Constraint],
    ) -> Option<RootProjectionSlot> {
        let projections = self.shared.as_ref()?.projections.get_or_init(|| {
            DashMap::with_hasher_and_shard_amount(Default::default(), dashmap_shards())
        });
        let key = RootProjectionKey {
            column,
            constraints: canonical_constraints(constraints),
        };
        Some(match projections.entry(key) {
            Entry::Occupied(entry) => entry.get().clone(),
            Entry::Vacant(entry) => {
                let slot = Arc::new(RootProjectionEntry {
                    projection: OnceLock::new(),
                    continuations: RootContinuationCache::shared(),
                });
                entry.insert(slot.clone());
                slot
            }
        })
    }

    /// Find the shared continuation grid for the persistent catalog index over
    /// `columns`, whose key positions identify this root's rows for every plan.
    pub(super) fn catalog_continuations(
        &self,
        columns: &[ColumnId],
    ) -> Option<Arc<RootContinuationCache>> {
        let continuations = self.shared.as_ref()?.catalog_continuations.get_or_init(|| {
            DashMap::with_hasher_and_shard_amount(Default::default(), dashmap_shards())
        });
        Some(match continuations.entry(ColumnIds::from_slice(columns)) {
            Entry::Occupied(entry) => entry.get().clone(),
            Entry::Vacant(entry) => {
                let cache = Arc::new(RootContinuationCache::shared());
                entry.insert(cache.clone());
                cache
            }
        })
    }
}

#[cfg(test)]
#[path = "packed_cache_tests.rs"]
mod tests;
