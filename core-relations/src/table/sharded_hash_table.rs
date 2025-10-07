//! Basic utilities around sharding a hashbrown `HashTable`.

use crate::numeric_id::NumericId;
use hashbrown::HashTable;
use serde::{Deserialize, Serialize, ser::SerializeStruct};

use crate::common::{ShardData, ShardId};

#[derive(Clone)]
pub(crate) struct ShardedHashTable<T> {
    shard_data: ShardData,
    shards: Vec<HashTable<T>>,
}

impl<T: Serialize + Clone> Serialize for ShardedHashTable<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Turn shards into Vec<Vec<T>> for serialization
        let serialized_shards: Vec<Vec<T>> = self
            .shards
            .iter()
            .map(|shard| shard.iter().cloned().collect())
            .collect();

        let mut state = serializer.serialize_struct("ShardedHashTable", 2)?;
        state.serialize_field("shard_data", &self.shard_data)?;
        state.serialize_field("shards", &serialized_shards)?;
        state.end()
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for ShardedHashTable<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Partial {
            shard_data: ShardData,
        }

        let helper: Partial = Partial::deserialize(deserializer)?;
        let shards = vec![]; // todo: this is a bogus default value. Need to reconstruct HashTable from Vec<T>
        Ok(ShardedHashTable {
            shard_data: helper.shard_data,
            shards,
        })
    }
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
