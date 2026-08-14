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
    HeaderConstraintId,
    u32,
    "an execution-local id for a canonical set of fast trie-root constraints"
);

/// Key for a shared trie root: the table plus an interned id for its fast
/// (header) constraints.
type RootKey = (TableId, HeaderConstraintId);

/// One round-local root projection can be reused by plans that share the same
/// root subset. Slow constraints are part of the key because they are applied
/// before projection; sorting them makes conjunction order irrelevant.
#[derive(Clone, Eq, Hash, PartialEq)]
struct RootProjectionKey {
    column: ColumnId,
    constraints: SmallVec<[Constraint; 2]>,
}

struct RootProjection {
    /// Final immutable scalar-index representation. Unlike the earlier pair
    /// cache, this is probed directly: queries do not copy it into their arenas.
    /// The trailing entry is an offset-only sentinel.
    keys: Box<[(Value, u32)]>,
    rows: Box<[RowId]>,
}

impl RootProjection {
    fn from_sorted_pairs(pairs: Vec<(Value, RowId)>) -> Self {
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

    fn len(&self) -> usize {
        self.keys.len().saturating_sub(1)
    }

    fn find(&self, value: Value) -> Option<usize> {
        let len = self.len();
        self.keys[..len]
            .binary_search_by_key(&value, |&(key, _)| key)
            .ok()
    }

    fn value_at(&self, key_index: usize) -> Value {
        assert!(key_index < self.len(), "projected root key out of bounds");
        self.keys[key_index].0
    }

    fn subset_at(&self, key_index: usize) -> SubsetRef<'_> {
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

type RootProjectionSlot = Arc<OnceLock<RootProjection>>;
type RootProjectionMap = DashMap<RootProjectionKey, RootProjectionSlot>;

/// A cache of trie roots shared across all plans within a single
/// `run_rule_set` call. Two plans that constrain the same table with the same
/// fast constraints share the owning root subset. Plan-execution packed
/// descendants remain separate.
///
/// Only roots that more than one plan actually uses are shared (`shared`), so
/// single-use roots stay per-plan and keep the pool-recycling behavior of the
/// unshared path — sharing a root that is never reused is pure overhead.
///
/// The `DashMap`s are setup caches rather than per-row probe structures. A plan
/// consults `roots` once while initializing each reused atom root; a single-use
/// root bypasses the cache completely. Likewise, the projection map is consulted
/// only when a prepared access first acquires a projection slot. That slot is
/// retained in `PreparedIndexState`, and the hot recursive probe path reads the
/// resulting immutable arrays directly. Contention is therefore limited to
/// single-flight construction when plans initialize the same root or projection
/// concurrently. Tables are frozen during a run, so each key continues to denote
/// the same subset after publication.
#[derive(Default)]
struct TrieCache {
    roots: DashMap<RootKey, Arc<TrieNode>>,
    /// Interns canonical header-constraint sets to keep [`RootKey`] cheap.
    /// The table stays outside the id and remains the first part of `RootKey`.
    header_ids: DashMap<SmallVec<[Constraint; 2]>, HeaderConstraintId>,
    next_header_id: AtomicUsize,
    /// Root signatures used by more than one plan; only these are shared.
    shared: HashSet<RootSignature>,
}

impl TrieCache {
    /// Return the interned id for a canonical set of fast header constraints.
    ///
    /// Id 0 is reserved for the common unconstrained case, so those atoms skip
    /// the interning map entirely. [`RootKey`] carries the table separately;
    /// identical constraint sets may therefore reuse an id across tables
    /// without making the roots alias.
    fn header_id(&self, fast: &[Constraint]) -> HeaderConstraintId {
        if fast.is_empty() {
            return HeaderConstraintId::new_const(0);
        }
        let mut sig: SmallVec<[Constraint; 2]> = SmallVec::from_iter(fast.iter().cloned());
        sig.sort_unstable();
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
    fn compute_shared<'a>(plans: impl Iterator<Item = &'a Plan>) -> HashSet<RootSignature> {
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
    fn with_shared(shared: HashSet<RootSignature>) -> TrieCache {
        // DashMap requires at least 2 shards; that is plenty for serial runs and
        // still far below the default (4 * num_cpus).
        let shards = crate::parallel::current_num_threads()
            .next_power_of_two()
            .max(2);
        TrieCache {
            roots: DashMap::with_hasher_and_shard_amount(Default::default(), shards),
            header_ids: DashMap::with_hasher_and_shard_amount(Default::default(), shards),
            next_header_id: AtomicUsize::new(0),
            shared,
        }
    }
}

/// Owning root subset for an atom. Lower trie levels are execution-scoped
/// packed nodes rather than persistent `TrieNode`s.
pub(crate) struct TrieNode {
    subset: Subset,
    /// Shared roots lazily cache sorted top-level projections across plans.
    /// Child publication remains query-local in the packed arena.
    root_projections: Option<OnceLock<RootProjectionMap>>,
}

impl std::fmt::Debug for TrieNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrieNode")
            .field("subset", &self.subset)
            .finish()
    }
}

impl TrieNode {
    fn new(subset: Subset) -> Self {
        Self {
            subset,
            root_projections: None,
        }
    }

    fn new_shared(subset: Subset) -> Self {
        Self {
            subset,
            root_projections: Some(OnceLock::new()),
        }
    }

    fn projection_slot(
        &self,
        column: ColumnId,
        constraints: &[Constraint],
    ) -> Option<RootProjectionSlot> {
        let projections = self.root_projections.as_ref()?.get_or_init(|| {
            let shards = crate::parallel::current_num_threads()
                .next_power_of_two()
                .max(2);
            DashMap::with_hasher_and_shard_amount(Default::default(), shards)
        });
        let mut canonical: SmallVec<[Constraint; 2]> = constraints.iter().cloned().collect();
        canonical.sort_unstable();
        let key = RootProjectionKey {
            column,
            constraints: canonical,
        };
        Some(match projections.entry(key) {
            Entry::Occupied(entry) => entry.get().clone(),
            Entry::Vacant(entry) => {
                let slot = Arc::new(OnceLock::new());
                entry.insert(slot.clone());
                slot
            }
        })
    }
}
