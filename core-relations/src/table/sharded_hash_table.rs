//! Basic utilities around sharding a hashbrown `HashTable`.

use hashbrown::HashTable;
use crate::numeric_id::NumericId;

use crate::common::{ShardData, ShardId};

#[derive(Clone)]
pub(crate) struct ShardedHashTable<T> {
    shard_data: ShardData,
    shards: Vec<HashTable<T>>,
}

impl<T> Default for ShardedHashTable<T> {
    fn default() -> Self {
        let cur_threads = rayon::current_num_threads();
        if cur_threads == 1 {
            Self::with_shards(1)
        } else {
            Self::with_shards(cur_threads * 2)
        }
    }
}

impl<T> ShardedHashTable<T> {
    pub(crate) fn clear(&mut self) {
        self.shards.iter_mut().for_each(|s| s.clear());
    }
    pub(crate) fn with_shards(shards: usize) -> Self {
        let shard_data = ShardData::new(shards);
        let shards = (0..shard_data.n_shards())
            .map(|_| HashTable::new())
            .collect::<Vec<_>>();
        Self { shard_data, shards }
    }

    /// Extract a [`ShardData`] allowing users to compute shard information for
    /// this table.
    pub(crate) fn shard_data(&self) -> ShardData {
        self.shard_data
    }

    pub(crate) fn get_shard(&self, shard_id: ShardId) -> &HashTable<T> {
        &self.shards[shard_id.index()]
    }

    pub(crate) fn mut_shards(&mut self) -> &mut [HashTable<T>] {
        &mut self.shards
    }
}
