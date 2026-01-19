//! Incremental partition refinement utilities for block/hash tables.

use std::hash::Hash;

use crate::core_relations::{
    ColumnId, CounterId, Database, ExecutionState, MergeFn, Offset, RowId, SortedWritesTable,
    SortedWritesTableOptions, TableId, TableVersion, TaggedRowBuffer, Value, WrappedTable,
};
use crate::numeric_id::NumericId;
use hashbrown::HashMap;
use indexmap::IndexMap;

use super::unique_repeat_index::UniqueRepeatIndex;

const REFRESH_BATCH: usize = 1024;

/// Schema information for a block/hash table.
#[derive(Debug, Clone, Copy)]
pub struct BlockHashTable {
    pub table: TableId,
    pub key_col: ColumnId,
    pub hash_col: ColumnId,
    pub block_col: ColumnId,
    pub ts_col: ColumnId,
}

/// Add a block/hash table to the database.
///
/// The table has one key column (the e-class id), and three value columns:
/// hash, block id, and timestamp. The table is sorted by timestamp and uses
/// a "new row wins" merge function.
pub fn add_block_hash_table(db: &mut Database) -> BlockHashTable {
    let key_col = ColumnId::from_usize(0);
    let hash_col = ColumnId::from_usize(1);
    let block_col = ColumnId::from_usize(2);
    let ts_col = ColumnId::from_usize(3);
    let options = SortedWritesTableOptions {
        sort_by: Some(ts_col),
        ..Default::default()
    };
    let merge_fn: Box<MergeFn> = Box::new(|_, cur, new, out| {
        if cur == new {
            return false;
        }
        out.clear();
        out.extend_from_slice(new);
        true
    });
    let table = SortedWritesTable::new(1, 4, options, Vec::new(), merge_fn);
    let table_id = db.add_table(table, std::iter::empty(), std::iter::empty());
    BlockHashTable {
        table: table_id,
        key_col,
        hash_col,
        block_col,
        ts_col,
    }
}

/// Add an e-class fingerprint table to the database.
///
/// The table has one key column (the e-class id), and three value columns:
/// hash, block id, and timestamp. The table is sorted by timestamp and sums
/// incoming hashes (wrapping) while preserving the existing block id.
pub fn add_eclass_fingerprint_table(db: &mut Database) -> BlockHashTable {
    let key_col = ColumnId::new(0);
    let hash_col = ColumnId::new(1);
    let block_col = ColumnId::new(2);
    let ts_col = ColumnId::new(3);
    let options = SortedWritesTableOptions {
        sort_by: Some(ts_col),
        ..Default::default()
    };
    let hash_idx = hash_col.index();
    let ts_idx = ts_col.index();
    let merge_fn: Box<MergeFn> = Box::new(move |_, cur, new, out| {
        let combined = Value::new(cur[hash_idx].rep().wrapping_add(new[hash_idx].rep()));
        if combined == cur[hash_idx] {
            return false;
        }
        out.clear();
        out.extend_from_slice(cur);
        out[hash_idx] = combined;
        out[ts_idx] = new[ts_idx];
        true
    });
    let table = SortedWritesTable::new(1, 4, options, Vec::new(), merge_fn);
    let table_id = db.add_table(table, std::iter::empty(), std::iter::empty());
    BlockHashTable {
        table: table_id,
        key_col,
        hash_col,
        block_col,
        ts_col,
    }
}

#[derive(Clone)]
struct BlockHashIndex {
    updated_to: Option<TableVersion>,
    by_block: UniqueRepeatIndex<Value, HashEntry>,
    by_hash: UniqueRepeatIndex<Value, BlockEntry>,
    block_members: IndexMap<Value, Vec<Value>>,
}

impl BlockHashIndex {
    fn new() -> Self {
        Self {
            updated_to: None,
            by_block: UniqueRepeatIndex::new(),
            by_hash: UniqueRepeatIndex::new(),
            block_members: IndexMap::new(),
        }
    }

    fn refresh(
        &mut self,
        table: &WrappedTable,
        key_col: ColumnId,
        hash_col: ColumnId,
        block_col: ColumnId,
    ) {
        let cur_version = table.version();
        if self.updated_to.as_ref() == Some(&cur_version) {
            return;
        }
        let subset = if let Some(prev) = self.updated_to.as_ref() {
            if cur_version.major != prev.major {
                self.by_block.clear();
                self.by_hash.clear();
                self.block_members.clear();
                table.all()
            } else {
                table.updates_since(prev.minor)
            }
        } else {
            self.by_block.clear();
            self.by_hash.clear();
            self.block_members.clear();
            table.all()
        };
        if subset.size() == 0 {
            self.updated_to = Some(cur_version);
            return;
        }

        let cols = [key_col, hash_col, block_col];
        let mut buf = TaggedRowBuffer::new(cols.len());
        let mut start = Offset::new(0);
        loop {
            buf.clear();
            let next =
                table.scan_project(subset.as_ref(), &cols, start, REFRESH_BATCH, &[], &mut buf);
            for (row_id, row) in buf.non_stale() {
                let key = row[0];
                let hash = row[1];
                let block = row[2];
                self.by_block.add(block, HashEntry { hash, key }, row_id);
                self.by_hash.add(hash, BlockEntry { block, key }, row_id);
                self.block_members.entry(block).or_default().push(key);
            }
            match next {
                Some(next) => start = next,
                None => break,
            }
        }
        self.updated_to = Some(cur_version);
    }
}

/// Incremental partition refinement for block/hash tables.
#[derive(Clone)]
pub struct PartitionRefinement {
    table: TableId,
    key_col: ColumnId,
    hash_col: ColumnId,
    block_col: ColumnId,
    ts_col: ColumnId,
    block_counter: CounterId,
    timestamp_counter: CounterId,
    index: BlockHashIndex,
}

impl PartitionRefinement {
    /// Create a new `PartitionRefinement` helper for the given table.
    pub fn new(
        table: BlockHashTable,
        block_counter: CounterId,
        timestamp_counter: CounterId,
    ) -> Self {
        Self {
            table: table.table,
            key_col: table.key_col,
            hash_col: table.hash_col,
            block_col: table.block_col,
            ts_col: table.ts_col,
            block_counter,
            timestamp_counter,
            index: BlockHashIndex::new(),
        }
    }

    pub(crate) fn refresh_index(&mut self, table: &WrappedTable) {
        self.index
            .refresh(table, self.key_col, self.hash_col, self.block_col);
    }

    pub(crate) fn block_members(&self) -> &IndexMap<Value, Vec<Value>> {
        &self.index.block_members
    }

    /// Split blocks so each block id maps to at most one hash.
    ///
    /// The most popular hash keeps its block id. Ties are broken by the
    /// smaller hash value.
    pub fn split_blocks(&mut self, state: &mut ExecutionState) {
        self.split_blocks_impl(state, Self::stage_block_update_merge);
    }

    /// Split blocks, replacing existing rows when updating block ids.
    pub fn split_blocks_replacing(&mut self, state: &mut ExecutionState) {
        self.split_blocks_impl(state, Self::stage_block_update_replace);
    }

    /// Split blocks using the provided update strategy.
    ///
    /// The `stage_update` callback is invoked for each row that needs a new block:
    /// - `&Self`: access to table metadata (columns, counters).
    /// - `&mut ExecutionState`: staging area for updates.
    /// - `&mut Vec<Value>`: scratch buffer for row construction.
    /// - `&[Value]`: the current row values (hash and block will be adjusted).
    /// - `Value`: the new block id to assign.
    /// - `Value`: the timestamp to write.
    fn split_blocks_impl(
        &mut self,
        state: &mut ExecutionState,
        mut stage_update: impl FnMut(
            &Self,
            &mut ExecutionState,
            &mut Vec<Value>,
            &[Value],
            Value,
            Value,
        ),
    ) {
        let table = state.get_table(self.table);
        self.index
            .refresh(table, self.key_col, self.hash_col, self.block_col);
        let ts = Value::from_usize(state.read_counter(self.timestamp_counter));
        let mut groups: HashMap<Value, Vec<KeyedRow>> = HashMap::new();
        let mut scratch = Vec::new();
        for (&block, entries) in self.index.by_block.repeat_iter() {
            groups.clear();
            for &(entry, row_id) in entries {
                let Some(row) = table.get_row(&[entry.key]) else {
                    continue;
                };
                if row.id != row_id {
                    continue;
                }
                if row.vals[self.block_col.index()] != block
                    || row.vals[self.hash_col.index()] != entry.hash
                {
                    continue;
                }
                groups.entry(entry.hash).or_default().push(KeyedRow {
                    key: entry.key,
                    row_id,
                });
            }
            if groups.len() <= 1 {
                continue;
            }

            let mut winner_hash = None;
            let mut winner_count = 0usize;
            for (&hash, rows) in groups.iter() {
                let count = rows.len();
                if winner_hash.is_none()
                    || count > winner_count
                    || (count == winner_count && hash < winner_hash.expect("winner hash missing"))
                {
                    winner_hash = Some(hash);
                    winner_count = count;
                }
            }
            let winner_hash = winner_hash.expect("winner hash missing");
            for (&hash, rows) in groups.iter() {
                if hash == winner_hash {
                    continue;
                }
                let new_block = Value::from_usize(state.inc_counter(self.block_counter));
                for row in rows {
                    let Some(current) = table.get_row(&[row.key]) else {
                        continue;
                    };
                    if current.id != row.row_id {
                        continue;
                    }
                    stage_update(
                        self,
                        state,
                        &mut scratch,
                        current.vals.as_ref(),
                        new_block,
                        ts,
                    );
                }
            }
        }
    }

    /// Merge blocks so each hash maps to at most one block id.
    ///
    /// The most popular block keeps its id. Ties are broken by the smallest
    /// row id in the block.
    pub fn merge_blocks(&mut self, state: &mut ExecutionState) {
        self.merge_blocks_impl(state, Self::stage_block_update_merge);
    }

    /// Merge blocks, replacing existing rows when updating block ids.
    pub fn merge_blocks_replacing(&mut self, state: &mut ExecutionState) {
        self.merge_blocks_impl(state, Self::stage_block_update_replace);
    }

    fn merge_blocks_impl(
        &mut self,
        state: &mut ExecutionState,
        mut stage_update: impl FnMut(
            &Self,
            &mut ExecutionState,
            &mut Vec<Value>,
            &[Value],
            Value,
            Value,
        ),
    ) {
        let table = state.get_table(self.table);
        self.index
            .refresh(table, self.key_col, self.hash_col, self.block_col);
        let ts = Value::from_usize(state.read_counter(self.timestamp_counter));
        let mut groups: HashMap<Value, BlockGroup> = HashMap::new();
        let mut scratch = Vec::new();
        for (&hash, entries) in self.index.by_hash.repeat_iter() {
            groups.clear();
            for &(mapping, row_id) in entries {
                let Some(row) = table.get_row(&[mapping.key]) else {
                    continue;
                };
                if row.id != row_id {
                    continue;
                }
                if row.vals[self.hash_col.index()] != hash
                    || row.vals[self.block_col.index()] != mapping.block
                {
                    continue;
                }
                let entry = groups.entry(mapping.block).or_insert_with(|| BlockGroup {
                    min_row_id: row_id,
                    rows: Vec::new(),
                });
                entry.min_row_id = std::cmp::min(entry.min_row_id, row_id);
                entry.rows.push(KeyedRow {
                    key: mapping.key,
                    row_id,
                });
            }
            if groups.len() <= 1 {
                continue;
            }
            let mut winner_block = None;
            let mut winner_count = 0usize;
            let mut winner_row_id = RowId::from_usize(0);
            for (&block, group) in groups.iter() {
                let count = group.rows.len();
                if winner_block.is_none()
                    || count > winner_count
                    || (count == winner_count && group.min_row_id < winner_row_id)
                {
                    winner_block = Some(block);
                    winner_count = count;
                    winner_row_id = group.min_row_id;
                }
            }
            let winner_block = winner_block.expect("winner block missing");
            for (&block, group) in groups.iter() {
                if block == winner_block {
                    continue;
                }
                for row in &group.rows {
                    let Some(current) = table.get_row(&[row.key]) else {
                        continue;
                    };
                    if current.id != row.row_id {
                        continue;
                    }
                    stage_update(
                        self,
                        state,
                        &mut scratch,
                        current.vals.as_ref(),
                        winner_block,
                        ts,
                    );
                }
            }
        }
    }

    /// Stage a block update by inserting a new row for the key.
    ///
    /// This keeps the existing row and relies on the table's merge function to
    /// decide which value wins.
    fn stage_block_update_merge(
        &self,
        state: &mut ExecutionState,
        scratch: &mut Vec<Value>,
        row: &[Value],
        new_block: Value,
        ts: Value,
    ) {
        scratch.clear();
        scratch.extend_from_slice(row);
        scratch[self.block_col.index()] = new_block;
        scratch[self.ts_col.index()] = ts;
        state.stage_insert(self.table, scratch);
    }

    /// Stage a block update by removing the old row and reinserting with the new block.
    ///
    /// This avoids invoking the merge function, which is required for tables that
    /// use their merge logic to accumulate hashes.
    fn stage_block_update_replace(
        &self,
        state: &mut ExecutionState,
        scratch: &mut Vec<Value>,
        row: &[Value],
        new_block: Value,
        ts: Value,
    ) {
        let key = row[self.key_col.index()];
        state.stage_remove(self.table, &[key]);
        self.stage_block_update_merge(state, scratch, row, new_block, ts);
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
struct HashEntry {
    hash: Value,
    key: Value,
}

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
struct BlockEntry {
    block: Value,
    key: Value,
}

#[derive(Clone, Copy)]
struct KeyedRow {
    key: Value,
    row_id: RowId,
}

struct BlockGroup {
    min_row_id: RowId,
    rows: Vec<KeyedRow>,
}

#[cfg(test)]
mod tests {
    use super::{PartitionRefinement, add_block_hash_table};
    use crate::core_relations::{Database, Offset, TaggedRowBuffer, Value};
    use crate::numeric_id::NumericId;

    fn insert_row(
        db: &Database,
        table: super::BlockHashTable,
        key: Value,
        hash: Value,
        block: Value,
        ts_counter: crate::core_relations::CounterId,
    ) {
        db.with_execution_state(|state| {
            let ts = Value::from_usize(state.read_counter(ts_counter));
            state.stage_insert(table.table, &[key, hash, block, ts]);
        });
    }

    fn collect_rows(
        db: &Database,
        table: super::BlockHashTable,
    ) -> Vec<(Value, Value, Value, Value)> {
        let table_ref = db.get_table(table.table);
        let cols = [table.key_col, table.hash_col, table.block_col, table.ts_col];
        let subset = table_ref.all();
        let mut buf = TaggedRowBuffer::new(cols.len());
        let mut start = Offset::new(0);
        let mut out = Vec::new();
        loop {
            buf.clear();
            let next = table_ref.scan_project(subset.as_ref(), &cols, start, 1024, &[], &mut buf);
            for (_row_id, row) in buf.non_stale() {
                out.push((row[0], row[1], row[2], row[3]));
            }
            match next {
                Some(next) => start = next,
                None => break,
            }
        }
        out
    }

    #[test]
    fn split_blocks_keeps_popular_hash() {
        let mut db = Database::new();
        let block_counter = db.add_counter();
        let ts_counter = db.add_counter();
        let table = add_block_hash_table(&mut db);
        db.with_execution_state(|state| {
            state.inc_counter_by(block_counter, 100);
        });

        let block = Value::new_const(1);
        let hash_a = Value::new_const(10);
        let hash_b = Value::new_const(20);

        insert_row(&db, table, Value::new_const(1), hash_a, block, ts_counter);
        insert_row(&db, table, Value::new_const(2), hash_a, block, ts_counter);
        insert_row(&db, table, Value::new_const(3), hash_b, block, ts_counter);
        db.merge_all();

        db.inc_counter(ts_counter);
        let mut refinement = PartitionRefinement::new(table, block_counter, ts_counter);
        db.with_execution_state(|state| refinement.split_blocks(state));
        db.merge_all();

        let rows = collect_rows(&db, table);
        let new_block = Value::from_usize(100);
        assert!(rows.contains(&(Value::new_const(1), hash_a, block, Value::new_const(0))));
        assert!(rows.contains(&(Value::new_const(2), hash_a, block, Value::new_const(0))));
        assert!(rows.contains(&(Value::new_const(3), hash_b, new_block, Value::new_const(1))));
    }

    #[test]
    fn split_blocks_tiebreaks_by_smaller_hash() {
        let mut db = Database::new();
        let block_counter = db.add_counter();
        let ts_counter = db.add_counter();
        let table = add_block_hash_table(&mut db);
        db.with_execution_state(|state| {
            state.inc_counter_by(block_counter, 200);
        });

        let block = Value::new_const(1);
        let hash_small = Value::new_const(5);
        let hash_large = Value::new_const(10);

        insert_row(
            &db,
            table,
            Value::new_const(1),
            hash_small,
            block,
            ts_counter,
        );
        insert_row(
            &db,
            table,
            Value::new_const(2),
            hash_large,
            block,
            ts_counter,
        );
        db.merge_all();

        db.inc_counter(ts_counter);
        let mut refinement = PartitionRefinement::new(table, block_counter, ts_counter);
        db.with_execution_state(|state| refinement.split_blocks(state));
        db.merge_all();

        let rows = collect_rows(&db, table);
        let new_block = Value::from_usize(200);
        assert!(rows.contains(&(Value::new_const(1), hash_small, block, Value::new_const(0))));
        assert!(rows.contains(&(
            Value::new_const(2),
            hash_large,
            new_block,
            Value::new_const(1)
        )));
    }

    #[test]
    fn merge_blocks_tiebreaks_by_first_seen() {
        let mut db = Database::new();
        let block_counter = db.add_counter();
        let ts_counter = db.add_counter();
        let table = add_block_hash_table(&mut db);

        let hash = Value::new_const(7);
        let block_a = Value::new_const(1);
        let block_b = Value::new_const(2);

        insert_row(&db, table, Value::new_const(1), hash, block_a, ts_counter);
        insert_row(&db, table, Value::new_const(2), hash, block_b, ts_counter);
        db.merge_all();

        db.inc_counter(ts_counter);
        let mut refinement = PartitionRefinement::new(table, block_counter, ts_counter);
        db.with_execution_state(|state| refinement.merge_blocks(state));
        db.merge_all();

        let rows = collect_rows(&db, table);
        assert!(rows.contains(&(Value::new_const(1), hash, block_a, Value::new_const(0))));
        assert!(rows.contains(&(Value::new_const(2), hash, block_a, Value::new_const(1))));
    }

    #[test]
    fn split_blocks_refreshes_incrementally() {
        let mut db = Database::new();
        let block_counter = db.add_counter();
        let ts_counter = db.add_counter();
        let table = add_block_hash_table(&mut db);
        db.with_execution_state(|state| {
            state.inc_counter_by(block_counter, 300);
        });

        let block = Value::new_const(1);
        let hash_a = Value::new_const(10);
        let hash_b = Value::new_const(20);

        insert_row(&db, table, Value::new_const(1), hash_a, block, ts_counter);
        db.merge_all();

        let mut refinement = PartitionRefinement::new(table, block_counter, ts_counter);
        db.with_execution_state(|state| refinement.split_blocks(state));
        db.merge_all();

        db.inc_counter(ts_counter);
        insert_row(&db, table, Value::new_const(2), hash_b, block, ts_counter);
        db.merge_all();

        db.with_execution_state(|state| refinement.split_blocks(state));
        db.merge_all();

        let rows = collect_rows(&db, table);
        let new_block = Value::from_usize(300);
        assert!(rows.contains(&(Value::new_const(1), hash_a, block, Value::new_const(0))));
        assert!(rows.contains(&(Value::new_const(2), hash_b, new_block, Value::new_const(1))));
    }
}
