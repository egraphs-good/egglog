// WARNING: Partition refinement currently treats container ids as opaque values.
// Container elements are not visible to the fingerprint table, so hashes and
// blocks do not reflect container contents. This is a known limitation until
// container-aware fingerprints are implemented.

pub mod crc32_hash;
pub mod refinement;
pub mod signature_set;
pub mod unique_repeat_index;

use std::sync::Arc;

use crate::core_relations::{
    ColumnId, CounterId, Database, DeleteFn, DisplacedTable, ExecutionState, ExternalFunction,
    ExternalFunctionId, MergeFn, Offset, QueryEntry as CoreQueryEntry, SortedWritesTable,
    SortedWritesTableOptions, TableId, TaggedRowBuffer, Value,
};
#[cfg(test)]
use crate::core_relations::make_external_func;
use crate::numeric_id::NumericId;
use crate::partition_refinement::crc32_hash::{crc32_external_func, crc32_hash};
use crate::partition_refinement::refinement::{
    BlockHashTable, PartitionRefinement, add_eclass_fingerprint_table,
};
use crate::partition_refinement::signature_set::{EClassSignatureTable, EnodeSignatureTable};
use crate::{
    ColumnTy, EGraph, FunctionId, QueryEntry, RefinementInput, Result, RuleId, SchemaMath,
    UnionAction, run_rules_impl,
};
use hashbrown::{HashMap, HashSet};
use indexmap::{IndexMap, IndexSet};
use thiserror::Error;

/// A hasher used for partition refinement node hashing.
pub trait PartitionRefinementHasher {
    fn external_func() -> Box<dyn ExternalFunction + 'static>;
    fn hash(values: &[Value]) -> Value;
}

#[derive(Debug, Error)]
pub enum PartitionRefinementError {
    #[error("partition refinement is not enabled")]
    NotEnabled,
}

/// The default CRC32-based hasher.
pub struct Crc32PartitionHasher;

impl PartitionRefinementHasher for Crc32PartitionHasher {
    fn external_func() -> Box<dyn ExternalFunction + 'static> {
        crc32_external_func()
    }

    fn hash(values: &[Value]) -> Value {
        crc32_hash(values)
    }
}

#[cfg(test)]
pub(crate) struct ConstantPartitionHasher;

#[cfg(test)]
impl PartitionRefinementHasher for ConstantPartitionHasher {
    fn external_func() -> Box<dyn ExternalFunction + 'static> {
        Box::new(make_external_func(|_, _| Some(Value::new_const(0))))
    }

    fn hash(_values: &[Value]) -> Value {
        Value::new_const(0)
    }
}

/// Schema information for the node-hash table.
#[derive(Debug, Clone, Copy)]
pub struct NodeHashTable {
    pub table: TableId,
    pub row_id_col: ColumnId,
    pub hash_col: ColumnId,
    pub eclass_col: ColumnId,
    pub ts_col: ColumnId,
}

/// Schema information for the colliding-eclasses table.
#[derive(Debug, Clone, Copy)]
pub struct CollidingEClassesTable {
    pub table: TableId,
    pub eclass_col: ColumnId,
    pub ts_col: ColumnId,
}

/// Schema information for a collision signature table.
///
/// The key is `(table id, signature..., eclass)` so we can record multiple
/// e-classes per signature without conflicts.
#[derive(Debug, Clone, Copy)]
pub struct CollisionSignatureTable {
    pub table: TableId,
    pub signature_len: usize,
    pub key_len: usize,
    pub eclass_key_col: ColumnId,
}

/// Add a node-hash table to the database.
///
/// The table has one key column (stable row id), and three value columns:
/// hash, e-class id, and timestamp. It uses a last-writer-wins merge function
/// that also emits inverse hash deltas into the fingerprint table.
pub fn add_node_hash_table(
    db: &mut Database,
    fingerprint: BlockHashTable,
    block_counter: CounterId,
    ts_counter: CounterId,
) -> NodeHashTable {
    let row_id_col = ColumnId::from_usize(0);
    let hash_col = ColumnId::from_usize(1);
    let eclass_col = ColumnId::from_usize(2);
    let ts_col = ColumnId::from_usize(3);
    let hash_idx = hash_col.index();
    let eclass_idx = eclass_col.index();
    let delete_fingerprint = fingerprint;
    let delete_fn: Arc<DeleteFn> = Arc::new(move |state, row| {
        let hash = row[hash_idx];
        let eclass = row[eclass_idx];
        stage_fingerprint_delta(
            state,
            delete_fingerprint,
            eclass,
            invert_hash(hash),
            block_counter,
            ts_counter,
        );
    });
    let options = SortedWritesTableOptions {
        sort_by: Some(ts_col),
        on_delete: Some(delete_fn),
        ..Default::default()
    };
    let merge_fingerprint = fingerprint;
    let merge_fn: Box<MergeFn> = Box::new(move |state, cur, new, out| {
        let old_hash = cur[hash_idx];
        let old_eclass = cur[eclass_idx];
        let new_hash = new[hash_idx];
        let new_eclass = new[eclass_idx];
        if old_hash == new_hash && old_eclass == new_eclass {
            return false;
        }
        stage_fingerprint_delta(
            state,
            merge_fingerprint,
            old_eclass,
            invert_hash(old_hash),
            block_counter,
            ts_counter,
        );
        out.clear();
        out.extend_from_slice(new);
        true
    });
    let table = SortedWritesTable::new(1, 4, options, Vec::new(), merge_fn);
    let table_id = db.add_table(
        table,
        std::iter::empty(),
        std::iter::once(fingerprint.table),
    );
    NodeHashTable {
        table: table_id,
        row_id_col,
        hash_col,
        eclass_col,
        ts_col,
    }
}

/// Add a colliding-eclasses table to the database.
///
/// The table has one key column (the e-class id) and one value column
/// (timestamp). It is sorted by timestamp and keeps the newest row.
pub fn add_colliding_eclasses_table(db: &mut Database) -> CollidingEClassesTable {
    let eclass_col = ColumnId::new(0);
    let ts_col = ColumnId::new(1);
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
    let table = SortedWritesTable::new(1, 2, options, Vec::new(), merge_fn);
    let table_id = db.add_table(table, std::iter::empty(), std::iter::empty());
    CollidingEClassesTable {
        table: table_id,
        eclass_col,
        ts_col,
    }
}

/// Add a collision signature table to the database.
///
/// The table keys are `(table id, signature...)` and the value is the e-class.
/// Conflicting e-classes for the same key indicate a bug and will panic.
pub fn add_collision_signature_table(
    db: &mut Database,
    signature_len: usize,
) -> CollisionSignatureTable {
    let key_len = signature_len + 1;
    let eclass_key_col = ColumnId::new(
        signature_len
            .try_into()
            .expect("collision signature table key length overflow"),
    );
    let merge_fn: Box<MergeFn> = Box::new(|_, _cur, _new, _out| false);
    let table = SortedWritesTable::new(key_len, key_len, Default::default(), Vec::new(), merge_fn);
    let table_id = db.add_table(table, std::iter::empty(), std::iter::empty());
    CollisionSignatureTable {
        table: table_id,
        signature_len,
        key_len,
        eclass_key_col,
    }
}

fn stage_fingerprint_delta(
    state: &mut ExecutionState,
    fingerprint: BlockHashTable,
    eclass: Value,
    hash_delta: Value,
    block_counter: CounterId,
    ts_counter: CounterId,
) {
    let ts = Value::from_usize(state.read_counter(ts_counter));
    let seed_block = Value::from_usize(state.read_counter(block_counter));
    state.stage_insert(fingerprint.table, &[eclass, hash_delta, seed_block, ts]);
}

fn invert_hash(hash: Value) -> Value {
    Value::new(hash.rep().wrapping_neg())
}

fn next_block_id(state: &ExecutionState, block_counter: CounterId, current_block: Value) -> Value {
    loop {
        let candidate = Value::from_usize(state.inc_counter(block_counter));
        if candidate != current_block {
            return candidate;
        }
    }
}

const COLLISION_CLEAR_BATCH: usize = 256;

fn clear_signature_table(state: &mut ExecutionState, table: CollisionSignatureTable) {
    let table_ref = state.get_table(table.table);
    let subset = table_ref.all();
    let cols: Vec<ColumnId> = (0..table.key_len)
        .map(|idx| ColumnId::new(idx.try_into().expect("column id overflow")))
        .collect();
    let mut buf = TaggedRowBuffer::new(table.key_len);
    let mut start = Offset::new(0);
    loop {
        buf.clear();
        let next = table_ref.scan_project(
            subset.as_ref(),
            &cols,
            start,
            COLLISION_CLEAR_BATCH,
            &[],
            &mut buf,
        );
        for (_, row) in buf.non_stale() {
            state.stage_remove(table.table, row);
        }
        match next {
            Some(next) => start = next,
            None => break,
        }
    }
}

/// Internal state for hash-based partition refinement.
#[derive(Clone)]
pub(crate) struct HashPartitionRefinementState {
    pub(crate) fingerprint_table: BlockHashTable,
    pub(crate) node_hash_table: NodeHashTable,
    pub(crate) colliding_eclasses_table: CollidingEClassesTable,
    pub(crate) block_counter: CounterId,
    pub(crate) hash_func: ExternalFunctionId,
    pub(crate) seed_rules: Vec<RuleId>,
    pub(crate) node_hash_rules: Vec<RuleId>,
    pub(crate) collision_scan_rules: Vec<RuleId>,
    pub(crate) propagate_rule: RuleId,
    pub(crate) clear_displaced_rule: RuleId,
    pub(crate) refinement: Option<PartitionRefinement>,
    pub(crate) collision_signature_tables: IndexMap<usize, CollisionSignatureTable>,
}

impl HashPartitionRefinementState {
    fn signature_table_for_len(
        &mut self,
        db: &mut Database,
        signature_len: usize,
    ) -> CollisionSignatureTable {
        if let Some(table) = self.collision_signature_tables.get(&signature_len) {
            return *table;
        }
        let table = add_collision_signature_table(db, signature_len);
        self.collision_signature_tables.insert(signature_len, table);
        table
    }
}

impl EGraph {
    /// Initialize partition refinement state and tables with a custom hasher.
    pub(crate) fn init_partition_refinement_with_hasher<H: PartitionRefinementHasher>(&mut self) {
        let block_counter = self.db.add_counter();
        let fingerprint_table = add_eclass_fingerprint_table(&mut self.db);
        let node_hash_table = add_node_hash_table(
            &mut self.db,
            fingerprint_table,
            block_counter,
            self.timestamp_counter,
        );
        let colliding_eclasses_table = add_colliding_eclasses_table(&mut self.db);
        let hash_func = self.register_external_func(H::external_func());
        let propagate_rule = {
            let mut rb = self.new_rule("partition_refinement: propagate node hashes", true);
            let row_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
            let hash: QueryEntry = rb.new_var(ColumnTy::Id).into();
            let eclass: QueryEntry = rb.new_var(ColumnTy::Id).into();
            let ts: QueryEntry = rb.new_var(ColumnTy::Id).into();
            rb.query_raw_table(
                node_hash_table.table,
                &[row_id, hash.clone(), eclass.clone(), ts],
                node_hash_table.ts_col,
            );
            let fingerprint_table_id = fingerprint_table.table;
            rb.add_callback(move |inner, rb| {
                let seed_block_entry = CoreQueryEntry::Var(rb.read_counter(block_counter));
                rb.insert(
                    fingerprint_table_id,
                    &[
                        inner.convert(&eclass),
                        inner.convert(&hash),
                        seed_block_entry,
                        inner.next_ts(),
                    ],
                )?;
                Ok(())
            });
            rb.build_internal(None)
        };
        let clear_displaced_rule = {
            let uf_table = self.uf_table;
            let tracing = self.tracing;
            let mut rb = self.new_rule("partition_refinement: clear displaced fingerprints", true);
            let eclass: QueryEntry = rb.new_var(ColumnTy::Id).into();
            let uf_leader: QueryEntry = rb.new_var(ColumnTy::Id).into();
            let uf_ts: QueryEntry = rb.new_var(ColumnTy::Id).into();
            let mut uf_entries = vec![eclass.clone(), uf_leader, uf_ts];
            if tracing {
                let uf_proof: QueryEntry = rb.new_var(ColumnTy::Id).into();
                uf_entries.push(uf_proof);
            }
            let uf_atom = rb.query_raw_table(uf_table, &uf_entries, ColumnId::new(2));
            rb.set_focus(uf_atom.index());
            let hash: QueryEntry = rb.new_var(ColumnTy::Id).into();
            let block: QueryEntry = rb.new_var(ColumnTy::Id).into();
            let ts: QueryEntry = rb.new_var(ColumnTy::Id).into();
            rb.query_raw_table(
                fingerprint_table.table,
                &[eclass.clone(), hash, block, ts],
                fingerprint_table.ts_col,
            );
            let fingerprint_table_id = fingerprint_table.table;
            rb.add_callback(move |inner, rb| {
                rb.remove(fingerprint_table_id, &[inner.convert(&eclass)])?;
                Ok(())
            });
            rb.build_internal(None)
        };
        self.partition_refinement = Some(HashPartitionRefinementState {
            fingerprint_table,
            node_hash_table,
            colliding_eclasses_table,
            block_counter,
            hash_func,
            seed_rules: Vec::new(),
            node_hash_rules: Vec::new(),
            collision_scan_rules: Vec::new(),
            propagate_rule,
            clear_displaced_rule,
            refinement: Some(PartitionRefinement::new(
                fingerprint_table,
                block_counter,
                self.timestamp_counter,
            )),
            collision_signature_tables: IndexMap::new(),
        });
    }

    /// Initialize partition refinement state and tables.
    pub(crate) fn init_partition_refinement(&mut self) {
        self.init_partition_refinement_with_hasher::<Crc32PartitionHasher>();
    }

    /// Create a new EGraph with partition refinement enabled and a custom hasher.
    /// This is intended for benchmarking or experimentation; CRC32 is recommended by default.
    pub fn with_partition_refinement_with_hasher<H: PartitionRefinementHasher>() -> EGraph {
        let mut db = Database::new();
        let uf_table = db.add_table_named(
            DisplacedTable::default(),
            "$uf".into(),
            std::iter::empty(),
            std::iter::empty(),
        );
        let mut egraph = EGraph::create_internal(
            db,
            uf_table,
            crate::EGraphOptions {
                tracing: false,
                enable_partition_refinement: false,
            },
        );
        egraph.init_partition_refinement_with_hasher::<H>();
        egraph
    }

    /// Add hashing rules for a function that returns an e-class id.
    pub(crate) fn add_partition_refinement_rules_for_function(&mut self, func: FunctionId) {
        if self.partition_refinement.is_none() {
            return;
        }
        let info = &self.funcs[func];
        if info.ret_ty() != ColumnTy::Id {
            return;
        }
        let func_table = info.table;
        let func_name = info.name.to_string();
        let schema = info.schema.clone();
        let refinement_inputs = info.refinement_inputs.clone();
        let signature_len = schema.len();
        let (
            fingerprint_table,
            node_hash_table,
            colliding_eclasses_table,
            block_counter,
            hash_func,
            signature_table,
        ) = {
            let state = self
                .partition_refinement
                .as_mut()
                .expect("partition refinement missing");
            let signature_table = state.signature_table_for_len(&mut self.db, signature_len);
            (
                state.fingerprint_table,
                state.node_hash_table,
                state.colliding_eclasses_table,
                state.block_counter,
                state.hash_func,
                signature_table,
            )
        };

        let seed_rule = {
            let mut rb = self.new_rule(&format!("partition_refinement: seed {}", func_name), true);
            let mut entries = Vec::with_capacity(schema.len());
            for ty in &schema {
                entries.push(rb.new_var(*ty).into());
            }
            let _ = rb
                .query_table(func, &entries, None)
                .expect("seed rule arity mismatch");
            let mut id_entries = Vec::new();
            for (entry, input) in entries.iter().zip(refinement_inputs.iter()) {
                if matches!(input, RefinementInput::Block) {
                    id_entries.push(entry.clone());
                }
            }
            let zero = QueryEntry::Const {
                val: Value::new_const(0),
                ty: ColumnTy::Id,
            };
            let fingerprint_table_id = fingerprint_table.table;
            rb.add_callback(move |inner, rb| {
                let seed_block_entry = CoreQueryEntry::Var(rb.read_counter(block_counter));
                for entry in &id_entries {
                    rb.insert(
                        fingerprint_table_id,
                        &[
                            inner.convert(entry),
                            inner.convert(&zero),
                            seed_block_entry,
                            inner.next_ts(),
                        ],
                    )?;
                }
                Ok(())
            });
            rb.build_internal(None)
        };

        let node_hash_rule = {
            let mut rb = self.new_rule(
                &format!("partition_refinement: node hash {}", func_name),
                true,
            );
            let mut entries = Vec::with_capacity(schema.len());
            for ty in &schema {
                entries.push(rb.new_var(*ty).into());
            }
            let (_atom, row_id_entry) = rb
                .query_table_with_row_id(func, &entries)
                .expect("node hash rule missing row id");
            let eclass_entry = entries
                .last()
                .expect("function schema must include return value")
                .clone();
            let mut hash_inputs = Vec::with_capacity(schema.len());
            hash_inputs.push(QueryEntry::Const {
                val: Value::from_usize(func_table.index()),
                ty: ColumnTy::Id,
            });
            for (entry, input) in entries
                .iter()
                .take(schema.len() - 1)
                .zip(refinement_inputs.iter().take(schema.len() - 1))
            {
                match input {
                    RefinementInput::Block => {
                        let hash_var: QueryEntry = rb.new_var(ColumnTy::Id).into();
                        let block_var: QueryEntry = rb.new_var(ColumnTy::Id).into();
                        let ts_var: QueryEntry = rb.new_var(ColumnTy::Id).into();
                        rb.query_raw_table(
                            fingerprint_table.table,
                            &[entry.clone(), hash_var, block_var.clone(), ts_var],
                            fingerprint_table.ts_col,
                        );
                        hash_inputs.push(block_var);
                    }
                    RefinementInput::Raw => {
                        hash_inputs.push(entry.clone());
                    }
                }
            }
            let hash_var = rb.call_external_func(hash_func, &hash_inputs, ColumnTy::Id, {
                let func_name = func_name.clone();
                move || format!("crc32 hash failed for {}", func_name)
            });
            let hash_entry: QueryEntry = hash_var.into();
            let node_hash_table_id = node_hash_table.table;
            rb.add_callback(move |inner, rb| {
                rb.insert(
                    node_hash_table_id,
                    &[
                        inner.convert(&row_id_entry),
                        inner.convert(&hash_entry),
                        inner.convert(&eclass_entry),
                        inner.next_ts(),
                    ],
                )?;
                Ok(())
            });
            rb.build_internal(None)
        };

        let collision_scan_rule = {
            let mut rb = self.new_rule(
                &format!("partition_refinement: collide scan {}", func_name),
                true,
            );
            let mut entries = Vec::with_capacity(schema.len());
            for ty in &schema {
                entries.push(rb.new_var(*ty).into());
            }
            let _ = rb
                .query_table(func, &entries, None)
                .expect("collision scan arity mismatch");
            let eclass_entry = entries
                .last()
                .expect("function schema must include return value")
                .clone();
            let colliding_ts: QueryEntry = rb.new_var(ColumnTy::Id).into();
            let colliding_atom = rb.query_raw_table(
                colliding_eclasses_table.table,
                &[eclass_entry.clone(), colliding_ts],
                colliding_eclasses_table.ts_col,
            );
            rb.set_focus(colliding_atom.index());

            let fingerprint_table_id = fingerprint_table.table;
            let block_col = fingerprint_table.block_col;
            let signature_table_id = signature_table.table;
            let schema = schema.clone();
            let refinement_inputs = refinement_inputs.clone();
            let table_id_val = Value::from_usize(func_table.index());
            rb.add_callback(move |inner, rb| {
                let ret_idx = schema.len() - 1;
                let mut signature = Vec::with_capacity(signature_len + 1);
                signature.push(CoreQueryEntry::Const(table_id_val));
                for (entry, input) in entries[..ret_idx]
                    .iter()
                    .zip(&refinement_inputs[..ret_idx])
                {
                    match input {
                        RefinementInput::Block => {
                            let child = inner.convert(entry);
                            let block = rb.lookup(fingerprint_table_id, &[child], block_col)?;
                            signature.push(block.into());
                        }
                        RefinementInput::Raw => {
                            signature.push(inner.convert(entry));
                        }
                    }
                }
                let eclass_val = inner.convert(&eclass_entry);
                signature.push(eclass_val);
                rb.insert(signature_table_id, &signature)?;
                Ok(())
            });
            rb.build_internal(None)
        };

        if let Some(state) = self.partition_refinement.as_mut() {
            state.seed_rules.push(seed_rule);
            state.node_hash_rules.push(node_hash_rule);
            state.collision_scan_rules.push(collision_scan_rule);
        }
    }

    /// Run hash-based partition refinement until it stabilizes.
    pub fn run_hash_partition_refinement(&mut self) -> Result<bool> {
        self.run_hash_partition_refinement_impl(true)
    }

    /// Run hash-based partition refinement without collision resolution.
    #[cfg(test)]
    pub(crate) fn run_hash_partition_refinement_no_collisions(&mut self) -> Result<bool> {
        self.run_hash_partition_refinement_impl(false)
    }

    fn run_hash_partition_refinement_impl(&mut self, resolve_collisions: bool) -> Result<bool> {
        let (
            seed_rules,
            node_hash_rules,
            propagate_rule,
            clear_displaced_rule,
            fingerprint_table,
            block_counter,
            colliding_eclasses_table,
            collision_scan_rules,
            collision_signature_tables,
        ) = if let Some(state) = self.partition_refinement.as_ref() {
            (
                state.seed_rules.clone(),
                state.node_hash_rules.clone(),
                state.propagate_rule,
                state.clear_displaced_rule,
                state.fingerprint_table,
                state.block_counter,
                state.colliding_eclasses_table,
                state.collision_scan_rules.clone(),
                state.collision_signature_tables.clone(),
            )
        } else {
            return Err(PartitionRefinementError::NotEnabled.into());
        };
        let mut refinement = if let Some(state) = self.partition_refinement.as_mut() {
            state
                .refinement
                .take()
                .expect("partition refinement state missing refinement")
        } else {
            return Err(PartitionRefinementError::NotEnabled.into());
        };

        let result = (|| {
            let mut changed_any = false;
            loop {
                let mut changed = false;
                if !seed_rules.is_empty() {
                    // Use a fresh timestamp for seed writes.
                    self.inc_ts();
                    let ts = self.next_ts();
                    let report = run_rules_impl(
                        &mut self.db,
                        &mut self.rules,
                        &seed_rules,
                        ts,
                        self.report_level,
                    )?;
                    changed |= report.changed;
                }

                if !node_hash_rules.is_empty() {
                    // Use a fresh timestamp for node-hash writes.
                    self.inc_ts();
                    let ts = self.next_ts();
                    let report = run_rules_impl(
                        &mut self.db,
                        &mut self.rules,
                        &node_hash_rules,
                        ts,
                        self.report_level,
                    )?;
                    changed |= report.changed;
                    self.inc_ts();
                    let ts = self.next_ts();

                    let report = run_rules_impl(
                        &mut self.db,
                        &mut self.rules,
                        &[propagate_rule],
                        ts,
                        self.report_level,
                    )?;
                    changed |= report.changed;
                }

                // Use a later timestamp for split/merge updates so seminaive can see them.
                self.inc_ts();
                self.db.with_execution_state(|state| {
                    refinement.split_blocks_replacing(state);
                    refinement.merge_blocks_replacing(state);
                });
                let merged_blocks = self.db.merge_all();
                changed |= merged_blocks;
                changed_any |= changed;
                if !changed {
                    break;
                }
            }

            if resolve_collisions {
                let collision_changed = self.resolve_hash_collisions_with_refinement(
                    &mut refinement,
                    fingerprint_table,
                    colliding_eclasses_table,
                    &collision_scan_rules,
                    &collision_signature_tables,
                    block_counter,
                )?;
                let merged_eclasses =
                    self.merge_eclasses_by_block(&mut refinement, fingerprint_table)?;
                changed_any |= collision_changed || merged_eclasses;
                if merged_eclasses {
                    self.inc_ts();
                    let ts = self.next_ts();
                    let clear_report = run_rules_impl(
                        &mut self.db,
                        &mut self.rules,
                        &[clear_displaced_rule],
                        ts,
                        self.report_level,
                    )?;
                    changed_any |= clear_report.changed;
                    if !node_hash_rules.is_empty() {
                        self.inc_ts();
                        let ts = self.next_ts();
                        let hash_report = run_rules_impl(
                            &mut self.db,
                            &mut self.rules,
                            &node_hash_rules,
                            ts,
                            self.report_level,
                        )?;
                        self.inc_ts();
                        let ts = self.next_ts();
                        let propagate_report = run_rules_impl(
                            &mut self.db,
                            &mut self.rules,
                            &[propagate_rule],
                            ts,
                            self.report_level,
                        )?;
                        changed_any |= hash_report.changed || propagate_report.changed;
                    }
                }
            }
            Ok(changed_any)
        })();

        if let Some(state) = self.partition_refinement.as_mut() {
            state.refinement = Some(refinement);
        }
        result
    }

    /// Debug helper for partition refinement. Dumps fingerprint + node-hash rows
    /// and expected hashes when `RUST_LOG=info` is enabled.
    pub fn dump_partition_refinement_state(&self, label: &str, ids: &[Value]) {
        if !log::log_enabled!(log::Level::Info) {
            return;
        }
        let Some(state) = self.partition_refinement.as_ref() else {
            log::info!("partition refinement disabled; skip dump {label}");
            return;
        };
        log::info!("-- {label} --");
        let fingerprint = self.db.get_table(state.fingerprint_table.table);
        let fp_key_idx = state.fingerprint_table.key_col.index();
        let fp_hash_idx = state.fingerprint_table.hash_col.index();
        let fp_block_idx = state.fingerprint_table.block_col.index();
        let fp_ts_idx = state.fingerprint_table.ts_col.index();
        let node_hash = self.db.get_table(state.node_hash_table.table);
        let nh_row_id_idx = state.node_hash_table.row_id_col.index();
        let nh_hash_idx = state.node_hash_table.hash_col.index();
        let nh_eclass_idx = state.node_hash_table.eclass_col.index();

        let mut blocks = HashMap::new();
        let mut all_ids = HashSet::new();
        self.scan_table(fingerprint, |row| {
            let eclass = row[fp_key_idx];
            blocks.insert(eclass, row[fp_block_idx]);
            all_ids.insert(eclass);
        });
        self.scan_table(node_hash, |row| {
            all_ids.insert(row[nh_eclass_idx]);
        });

        let mut ids_to_dump = if ids.is_empty() {
            all_ids.into_iter().collect::<Vec<_>>()
        } else {
            ids.to_vec()
        };
        ids_to_dump.sort();
        ids_to_dump.dedup();

        let mut expected_by_row = HashMap::new();
        let hash_func = state.hash_func;
        self.db.with_execution_state(|exec| {
            for (_, info) in self.funcs.iter() {
                if info.ret_ty() != ColumnTy::Id {
                    continue;
                }
                let schema_math = SchemaMath {
                    tracing: self.tracing,
                    subsume: info.can_subsume,
                    row_id: info.row_id,
                    func_cols: info.schema.len(),
                };
                if !info.row_id {
                    log::info!("skipping {}: row ids not enabled", info.name);
                    continue;
                }
                let row_id_idx = schema_math.row_id_col();
                let ret_idx = info.schema.len() - 1;
                let refinement_inputs = &info.refinement_inputs;
                let table = self.db.get_table(info.table);
                let mut hash_inputs = Vec::with_capacity(ret_idx + 1);
                self.scan_table(table, |row| {
                    let row_id = row[row_id_idx];
                    let eclass = row[ret_idx];
                    hash_inputs.clear();
                    hash_inputs.push(Value::from_usize(info.table.index()));
                    for (col_idx, input) in refinement_inputs[..ret_idx].iter().enumerate() {
                        let val = row[col_idx];
                        match input {
                            RefinementInput::Block => {
                                let Some(block) = blocks.get(&val) else {
                                    log::info!(
                                        "missing fingerprint block for child {val:?} in {}",
                                        info.name
                                    );
                                    return;
                                };
                                hash_inputs.push(*block);
                            }
                            RefinementInput::Raw => hash_inputs.push(val),
                        }
                    }
                    let hash = exec
                        .call_external_func(hash_func, &hash_inputs)
                        .expect("hash function should return a value");
                    log::info!(
                        "row_id={row_id:?}, eclass={eclass:?}, hash={hash:?}, ts={:?}",
                        row[schema_math.ts_col()]
                    );
                    expected_by_row.insert(row_id, (hash, eclass));
                });
            }
        });

        for &id in &ids_to_dump {
            let canon = self.get_canon_repr(id, ColumnTy::Id);
            let row = fingerprint
                .get_row(&[canon])
                .expect("missing fingerprint row");
            log::info!(
                "eclass {id:?} canon={canon:?} hash={:?} block={:?} ts={:?}",
                row.vals[fp_hash_idx],
                row.vals[fp_block_idx],
                row.vals[fp_ts_idx]
            );
            let mut entries = Vec::new();
            let mut expected_sum = 0u32;
            self.scan_table(node_hash, |row| {
                if row[nh_eclass_idx] == canon {
                    let row_id = row[nh_row_id_idx];
                    let actual_hash = row[nh_hash_idx];
                    let expected_hash =
                        expected_by_row.get(&row_id).copied().map(|(hash, eclass)| {
                            if eclass != canon {
                                log::info!(
                                    "row_id {row_id:?} expected eclass {eclass:?} actual {canon:?}"
                                );
                            }
                            hash
                        });
                    if let Some(hash) = expected_hash {
                        expected_sum = expected_sum.wrapping_add(hash.rep());
                    }
                    entries.push((row_id, actual_hash, expected_hash));
                }
            });
            let sum = entries
                .iter()
                .fold(0u32, |acc, (_, hash, _)| acc.wrapping_add(hash.rep()));
            log::info!(
                "node_hash entries={entries:?} sum={:?} expected_sum={:?}",
                Value::new(sum),
                Value::new(expected_sum)
            );
        }
    }

    fn resolve_hash_collisions_with_refinement(
        &mut self,
        refinement: &mut PartitionRefinement,
        fingerprint_table: BlockHashTable,
        colliding_eclasses_table: CollidingEClassesTable,
        collision_scan_rules: &[RuleId],
        collision_signature_tables: &IndexMap<usize, CollisionSignatureTable>,
        block_counter: CounterId,
    ) -> Result<bool> {
        let mut changed_any = false;
        loop {
            let table = self.db.get_table(fingerprint_table.table);
            refinement.refresh_index(table);
            let mut colliding_blocks = Vec::new();
            for (block, members) in refinement.block_members().iter() {
                if members.len() > 1 {
                    colliding_blocks.push((*block, members.clone()));
                }
            }
            if colliding_blocks.is_empty() {
                return Ok(changed_any);
            }

            let mut candidates = IndexSet::new();
            for (_, members) in &colliding_blocks {
                for &eclass in members {
                    candidates.insert(eclass);
                }
            }

            self.inc_ts();
            let ts = self.next_ts();
            let ts_val = ts.to_value();
            self.db.with_execution_state(|state| {
                for table in collision_signature_tables.values() {
                    clear_signature_table(state, *table);
                }
                for &eclass in candidates.iter() {
                    state.stage_insert(colliding_eclasses_table.table, &[eclass, ts_val]);
                }
            });
            self.db.merge_all();

            if !collision_scan_rules.is_empty() {
                let _ = run_rules_impl(
                    &mut self.db,
                    &mut self.rules,
                    collision_scan_rules,
                    ts,
                    self.report_level,
                )?;
            }

            let mut enodes_by_eclass: IndexMap<Value, Vec<_>> = IndexMap::new();
            for &eclass in candidates.iter() {
                enodes_by_eclass.insert(eclass, Vec::new());
            }

            let mut enode_table = EnodeSignatureTable::new();
            for table in collision_signature_tables.values() {
                let signature_table = self.db.get_table(table.table);
                let signature_len = table.signature_len;
                let eclass_idx = table.eclass_key_col.index();
                self.scan_table(signature_table, |row| {
                    let eclass = row[eclass_idx];
                    if let Some(list) = enodes_by_eclass.get_mut(&eclass) {
                        let enode_sig = enode_table.intern(&row[..signature_len]);
                        list.push(enode_sig);
                    }
                });
            }

            let mut eclass_sig_table = EClassSignatureTable::new();
            let mut eclass_sigs = IndexMap::new();
            for (&eclass, enodes) in enodes_by_eclass.iter() {
                let sig = eclass_sig_table.intern(enodes.iter().copied());
                eclass_sigs.insert(eclass, sig);
            }

            let fingerprint_ref = self.db.get_table(fingerprint_table.table);
            let block_idx = fingerprint_table.block_col.index();
            let mut splits = Vec::new();
            for (block, members) in colliding_blocks {
                let mut groups: IndexMap<_, Vec<_>> = IndexMap::new();
                let mut seen = IndexSet::new();
                for eclass in members {
                    if !seen.insert(eclass) {
                        continue;
                    }
                    let Some(row) = fingerprint_ref.get_row(&[eclass]) else {
                        continue;
                    };
                    if row.vals[block_idx] != block {
                        continue;
                    }
                    let Some(sig) = eclass_sigs.get(&eclass) else {
                        continue;
                    };
                    groups.entry(*sig).or_default().push(eclass);
                }
                if groups.len() <= 1 {
                    continue;
                }

                let mut winner_sig = None;
                let mut winner_size = 0usize;
                let mut winner_min = None;
                for (sig, group) in groups.iter() {
                    let size = group.len();
                    let min_eclass = group
                        .iter()
                        .copied()
                        .min()
                        .expect("group should be non-empty");
                    let replace = winner_sig.is_none()
                        || size > winner_size
                        || (size == winner_size
                            && winner_min.is_some_and(|winner| min_eclass < winner));
                    if replace {
                        winner_sig = Some(*sig);
                        winner_size = size;
                        winner_min = Some(min_eclass);
                    }
                }
                let winner_sig = winner_sig.expect("winner missing");
                let mut to_split = Vec::new();
                for (sig, group) in groups {
                    if sig == winner_sig {
                        continue;
                    }
                    to_split.push(group);
                }
                if !to_split.is_empty() {
                    splits.push((block, to_split));
                }
            }

            if splits.is_empty() {
                return Ok(changed_any);
            }

            self.inc_ts();
            let ts_col = fingerprint_table.ts_col.index();
            self.db.with_execution_state(|state| {
                let ts = Value::from_usize(state.read_counter(self.timestamp_counter));
                let table = state.get_table(fingerprint_table.table);
                let mut scratch = Vec::new();
                for (block, groups) in splits {
                    for group in groups {
                        let new_block = next_block_id(state, block_counter, block);
                        for eclass in group {
                            let Some(row) = table.get_row(&[eclass]) else {
                                continue;
                            };
                            if row.vals[block_idx] != block {
                                continue;
                            }
                            scratch.clear();
                            scratch.extend_from_slice(row.vals.as_ref());
                            scratch[block_idx] = new_block;
                            scratch[ts_col] = ts;
                            state.stage_remove(fingerprint_table.table, &[eclass]);
                            state.stage_insert(fingerprint_table.table, &scratch);
                        }
                    }
                }
            });

            let merged = self.db.merge_all();
            changed_any |= merged;
            if !merged {
                return Ok(changed_any);
            }
        }
    }

    fn merge_eclasses_by_block(
        &mut self,
        refinement: &mut PartitionRefinement,
        fingerprint_table: BlockHashTable,
    ) -> Result<bool> {
        self.inc_ts();
        let table = self.db.get_table(fingerprint_table.table);
        refinement.refresh_index(table);
        let fingerprint_ref = self.db.get_table(fingerprint_table.table);
        let block_idx = fingerprint_table.block_col.index();
        let union_action = UnionAction::new(self);
        let mut changed = false;
        self.db.with_execution_state(|state| {
            let mut members = Vec::new();
            for (block, entries) in refinement.block_members().iter() {
                members.clear();
                for &eclass in entries {
                    let Some(row) = fingerprint_ref.get_row(&[eclass]) else {
                        continue;
                    };
                    if row.vals[block_idx] != *block {
                        continue;
                    }
                    members.push(eclass);
                }
                if members.len() <= 1 {
                    continue;
                }
                members.sort();
                members.dedup();
                if members.len() <= 1 {
                    continue;
                }
                let leader = members[0];
                for &eclass in &members[1..] {
                    union_action.union(state, leader, eclass);
                    changed = true;
                }
            }
        });
        if !changed {
            return Ok(false);
        }
        let merged = self.db.merge_all();
        if merged {
            self.rebuild()?;
        }
        Ok(merged)
    }
}
