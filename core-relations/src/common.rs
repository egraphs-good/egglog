use std::{
    hash::{BuildHasherDefault, Hash, Hasher},
    mem,
    ops::Deref,
    sync::{Arc, Mutex},
};

use concurrency::ConcurrentVec;
use hashbrown::HashTable;
use numeric_id::{define_id, DenseIdMap, IdVec, NumericId};
use rustc_hash::FxHasher;

use crate::{pool::Clear, Subset, TableId, TableVersion, WrappedTable};

pub(crate) type HashMap<K, V> = hashbrown::HashMap<K, V, BuildHasherDefault<FxHasher>>;
pub(crate) type HashSet<T> = hashbrown::HashSet<T, BuildHasherDefault<FxHasher>>;
pub(crate) type IndexSet<T> = indexmap::IndexSet<T, BuildHasherDefault<FxHasher>>;
pub(crate) type IndexMap<K, V> = indexmap::IndexMap<K, V, BuildHasherDefault<FxHasher>>;
pub(crate) type DashMap<K, V> = dashmap::DashMap<K, V, BuildHasherDefault<FxHasher>>;

/// An intern table mapping a key to some numeric id type.
///
/// This is primarily used to manage the [`Value`]s associated with a a
/// base value.
#[derive(Clone)]
pub struct InternTable<K, V> {
    vals: Arc<ConcurrentVec<K>>,
    data: Vec<Arc<Mutex<HashTable<V>>>>,
    shards_log2: u32,
}

impl<K, V> Default for InternTable<K, V> {
    fn default() -> Self {
        Self::with_shards(4)
    }
}
impl<K, V> InternTable<K, V> {
    /// Create a new intern table with the given number of shards.
    ///
    /// The number of shards is passed as its base-2 log: we rely on the number
    /// of shards being a power of two.
    fn with_shards(shards_log2: u32) -> InternTable<K, V> {
        let mut data = Vec::new();
        data.resize_with(1 << shards_log2, Default::default);
        InternTable {
            vals: Arc::new(ConcurrentVec::with_capacity(512)),
            data,
            shards_log2,
        }
    }
}

impl<K: Eq + Hash + Clone, V: NumericId> InternTable<K, V> {
    pub fn intern(&self, k: &K) -> V {
        let hash = hash_value(k);
        // Use the top bits of the hash to pick the shard. Hashbrown uses the
        // bottom bits.
        let shard = ((hash >> (64 - self.shards_log2)) & ((1 << self.shards_log2) - 1)) as usize;
        let mut table = self.data[shard].lock().unwrap();
        let read_guard = self.vals.read();
        if let Some(v) = table.find(hash, |v| k == &read_guard[v.index()]) {
            *v
        } else {
            mem::drop(read_guard);
            let res = V::from_usize(self.vals.push(k.clone()));
            let read_guard = self.vals.read();
            *table
                .insert_unique(hash, res, |v| hash_value(&read_guard[v.index()]))
                .get()
        }
    }

    pub fn get(&self, v: V) -> impl Deref<Target = K> + '_ {
        MapDeref {
            base: self.vals.read(),
            index: v.index(),
        }
    }

    pub fn get_cloned(&self, v: V) -> K {
        self.vals.read()[v.index()].clone()
    }
}

fn hash_value(v: &impl Hash) -> u64 {
    let mut hasher = FxHasher::default();
    v.hash(&mut hasher);
    hasher.finish()
}

impl<K: NumericId, V> Clear for DenseIdMap<K, V> {
    fn reuse(&self) -> bool {
        self.capacity() > 0
    }
    fn clear(&mut self) {
        self.clear();
    }
    fn bytes(&self) -> usize {
        self.capacity() * mem::size_of::<Option<V>>()
    }
}

define_id!(pub Value, u32, "A generic identifier representing an egglog value");

impl Value {
    pub(crate) fn stale() -> Self {
        Value::new(u32::MAX)
    }
    /// Values have a special "Stale" value that is used to indicate that the
    /// value isn't intended to be read.
    pub(crate) fn set_stale(&mut self) {
        self.rep = u32::MAX;
    }

    /// Whether or not the given value is stale. See [`Value::set_stale`].
    pub(crate) fn is_stale(&self) -> bool {
        self.rep == u32::MAX
    }
}

struct MapDeref<T> {
    base: T,
    index: usize,
}

impl<S, T: Deref<Target = [S]>> Deref for MapDeref<T> {
    type Target = S;

    fn deref(&self) -> &S {
        &(&*self.base)[self.index]
    }
}

define_id!(pub(crate) ShardId, u32, "an identifier pointing to a shard in a sharded hash table");

/// Sharding metadata used for sharding hash tables.
///
/// This is a separate type in order to allow other data-structures to pre-shard
/// data bound for a particular table.
#[derive(Copy, Clone)]
pub(crate) struct ShardData {
    log2_shard_count: u32,
}

impl ShardData {
    pub(crate) fn new(n_shards: usize) -> Self {
        Self {
            log2_shard_count: n_shards.next_power_of_two().trailing_zeros(),
        }
    }
    pub(crate) fn n_shards(&self) -> usize {
        1 << self.log2_shard_count
    }
    pub(crate) fn shard_id(&self, hash: u64) -> ShardId {
        let high_bits = (hash.wrapping_shr(64 - (self.log2_shard_count + 7)))
            & ((1 << self.log2_shard_count) - 1);
        ShardId::from_usize(high_bits as usize)
    }
    pub(crate) fn get_shard<'a, K: ?Sized, V>(&self, val: &K, table: &'a IdVec<ShardId, V>) -> &'a V
    where
        for<'b> &'b K: Hash,
    {
        let hc = {
            let mut hasher = FxHasher::default();
            val.hash(&mut hasher);
            hasher.finish()
        };
        &table[self.shard_id(hc)]
    }

    pub(crate) fn get_shard_mut<'a, V>(
        &self,
        val: impl Hash,
        table: &'a mut IdVec<ShardId, V>,
    ) -> &'a mut V {
        let hc = {
            let mut hasher = FxHasher::default();
            val.hash(&mut hasher);
            hasher.finish()
        };
        &mut table[self.shard_id(hc)]
    }
}

/// A simple helper struct used when handling incremental rebuilds that tracks the subsets of set
/// of tables that have been passed to the tracker.
#[derive(Clone, Default)]
pub(crate) struct SubsetTracker {
    last_rebuilt_at: DenseIdMap<TableId, TableVersion>,
}

impl SubsetTracker {
    /// Hand back the subset of the table needed to be scanned in order to see all updates since
    /// the last call to this method.
    ///
    /// If the given table's major version has been incremented, this method will return the whole
    /// table. In other words, this method does not guarantee that the returned subset is disjoint
    /// from ones that have been returned in the past.
    pub(crate) fn recent_updates(&mut self, table_id: TableId, table: &WrappedTable) -> Subset {
        let current_version = table.version();
        let res = if let Some(last_version) = self.last_rebuilt_at.get(table_id) {
            if current_version.major == last_version.major {
                table.updates_since(last_version.minor)
            } else {
                table.all()
            }
        } else {
            table.all()
        };
        self.last_rebuilt_at.insert(table_id, current_version);
        res
    }
}

/// Iterate over the contents of a `DashMap`, using a lower-overhead method than the built-in
/// iterators available from `DashMap`.
pub(crate) fn iter_dashmap_bulk<K: Hash + Eq, V>(
    map: &mut DashMap<K, V>,
    mut f: impl FnMut(&K, &mut V),
) {
    let shards = map.shards_mut();
    for shard in shards {
        let mut_shard = shard.get_mut();
        // SAFETY: this iterator does not outlive `shard`: it is not returned from this function
        // and `f` cannot store it anywhere as it takes an arbitrary lifetime.
        for entry in unsafe { mut_shard.iter() } {
            // SAFETY: we have exclusive access to the whole table.
            let (k, v) = unsafe { entry.as_mut() };
            f(k, v.get_mut());
        }
    }
}
