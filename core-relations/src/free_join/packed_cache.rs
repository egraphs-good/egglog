//! Trie state shared across the plans of one rule-set execution.
//!
//! Plans that constrain the same table with the same fast constraints share
//! one [`TrieRoot`]. Everything built below a shared root is shared as well:
//! its packed root indexes, the continuation grids of its persistent catalog
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
    common::{DashMap, HashMap, HashSet},
    numeric_id::{NumericId, define_id},
    offsets::Subset,
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

/// Arena addresses published once per successor family. The rule-set run
/// keeps the arena alive until all plans and their shared cache are done.
type PackedRootSlots = Box<[OnceLock<usize>]>;
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
/// root bypasses the cache completely. The dense packed-root slots and catalog
/// continuation map are consulted only when a prepared access first acquires
/// its index. That index is retained in `PreparedIndexState`, and the hot
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
    packed_roots: OnceLock<PackedRootSlots>,
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

    /// Find the packed-root slot for an interned `(column, slow constraints)` family.
    /// As in the original column-index cache, dense slots avoid a separate
    /// hash table and canonicalization for each root. Families are fixed while
    /// a rule set runs; the root and its slots are discarded after that run.
    pub(super) fn packed_root_slot(
        &self,
        family: FamilyId,
        family_count: usize,
    ) -> Option<&OnceLock<usize>> {
        let roots = self.shared.as_ref()?.packed_roots.get_or_init(|| {
            std::iter::repeat_with(OnceLock::new)
                .take(family_count)
                .collect()
        });
        debug_assert_eq!(roots.len(), family_count);
        Some(&roots[family.index()])
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
