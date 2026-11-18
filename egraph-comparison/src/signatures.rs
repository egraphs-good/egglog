use crate::ids::BlockId;
use egglog_numeric_id::NumericId;
use hashbrown::HashTable;
use smallvec::SmallVec;
use std::{
    hash::{Hash, Hasher},
    ops::Range,
};

struct Entry {
    hash: u64,
    words: Range<usize>,
    block: BlockId,
}

/// Per-batch interning with one flat key arena. Table entries retain their hash
/// so growth never re-hashes long signatures. Collisions compare complete keys.
#[derive(Default)]
pub(crate) struct Signatures {
    table: HashTable<Entry>,
    words: Vec<usize>,
    scratch: Vec<usize>,
}

impl Signatures {
    /// Reuse allocations after processing a block; no signature survives a split.
    pub fn clear(&mut self) {
        // HashTable::clear touches its capacity. Release an oversized table so
        // a large split cannot make every later tiny batch cost O(large split).
        if self.table.capacity() > 4 * self.table.len().max(1) {
            self.table = HashTable::new();
        } else {
            self.table.clear();
        }
        self.words.clear();
    }

    pub fn intern(&mut self, previous: BlockId, nodes: &[SmallVec<[usize; 3]>]) -> BlockId {
        self.scratch.clear();
        self.scratch.push(previous.index());
        // These usize words are an encoding, not interchangeable IDs: each node
        // has a length prefix, followed by its SymbolId and ordered BlockIds.
        // Lengths distinguish different arities and adjacent-node boundaries.
        for node in nodes {
            self.scratch.push(node.len());
            self.scratch.extend_from_slice(node);
        }
        let mut hasher = rustc_hash::FxHasher::default();
        self.scratch.hash(&mut hasher);
        self.intern_hashed(hasher.finish())
    }

    fn intern_hashed(&mut self, hash: u64) -> BlockId {
        if let Some(entry) = self.table.find(hash, |entry| {
            self.words[entry.words.clone()] == self.scratch
        }) {
            return entry.block;
        }
        let block = BlockId::from_usize(self.table.len());
        let start = self.words.len();
        self.words.extend_from_slice(&self.scratch);
        self.table.insert_unique(
            hash,
            Entry {
                hash,
                words: start..self.words.len(),
                block,
            },
            |entry| entry.hash,
        );
        block
    }
}

#[cfg(test)]
#[path = "../tests/support/signatures.rs"]
mod tests;
