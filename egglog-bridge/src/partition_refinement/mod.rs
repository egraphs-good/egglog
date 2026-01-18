// WARNING: Partition refinement currently treats container ids as opaque values.
// Container elements are not visible to the fingerprint table, so hashes and
// blocks do not reflect container contents. This is a known limitation until
// container-aware fingerprints are implemented.

pub mod crc32_hash;
pub mod refinement;
pub mod signature_set;
pub mod unique_repeat_index;

use std::sync::Arc;

use anyhow::anyhow;

use crate::core_relations::{
    ColumnId, CounterId, Database, DeleteFn, ExecutionState, ExternalFunctionId, MergeFn,
    SortedWritesTable, SortedWritesTableOptions, TableId, Value,
};
use crate::numeric_id::NumericId;
use crate::partition_refinement::crc32_hash::crc32_external_func;
use crate::partition_refinement::refinement::{
    BlockHashTable, PartitionRefinement, add_eclass_fingerprint_table,
};
use crate::{ColumnTy, EGraph, FunctionId, QueryEntry, Result, RuleId, run_rules_impl};

/// Schema information for the node-hash table.
#[derive(Debug, Clone, Copy)]
pub struct NodeHashTable {
    pub table: TableId,
    pub row_id_col: ColumnId,
    pub hash_col: ColumnId,
    pub eclass_col: ColumnId,
    pub ts_col: ColumnId,
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

fn stage_fingerprint_delta(
    state: &mut ExecutionState,
    fingerprint: BlockHashTable,
    eclass: Value,
    hash_delta: Value,
    block_counter: CounterId,
    ts_counter: CounterId,
) {
    let block = Value::from_usize(state.read_counter(block_counter));
    let ts = Value::from_usize(state.read_counter(ts_counter));
    state.stage_insert(fingerprint.table, &[eclass, hash_delta, block, ts]);
}

fn invert_hash(hash: Value) -> Value {
    Value::new(hash.rep().wrapping_neg())
}

/// Internal state for hash-based partition refinement.
#[derive(Clone)]
pub(crate) struct HashPartitionRefinementState {
    pub(crate) fingerprint_table: BlockHashTable,
    pub(crate) node_hash_table: NodeHashTable,
    pub(crate) block_counter: CounterId,
    pub(crate) crc32_func: ExternalFunctionId,
    pub(crate) seed_rules: Vec<RuleId>,
    pub(crate) node_hash_rules: Vec<RuleId>,
    pub(crate) propagate_rule: RuleId,
}

impl EGraph {
    /// Initialize partition refinement state and tables.
    pub(crate) fn init_partition_refinement(&mut self) {
        let block_counter = self.db.add_counter();
        let fingerprint_table = add_eclass_fingerprint_table(&mut self.db);
        let node_hash_table = add_node_hash_table(
            &mut self.db,
            fingerprint_table,
            block_counter,
            self.timestamp_counter,
        );
        let crc32_func = self.register_external_func(crc32_external_func());
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
                let block = rb.read_counter(block_counter);
                rb.insert(
                    fingerprint_table_id,
                    &[
                        inner.convert(&eclass),
                        inner.convert(&hash),
                        block.into(),
                        inner.next_ts(),
                    ],
                )?;
                Ok(())
            });
            rb.build_internal(None)
        };
        self.partition_refinement = Some(HashPartitionRefinementState {
            fingerprint_table,
            node_hash_table,
            block_counter,
            crc32_func,
            seed_rules: Vec::new(),
            node_hash_rules: Vec::new(),
            propagate_rule,
        });
    }

    /// Add hashing rules for a function that returns an e-class id.
    pub(crate) fn add_partition_refinement_rules_for_function(&mut self, func: FunctionId) {
        let Some(state) = self.partition_refinement.as_ref() else {
            return;
        };
        let info = &self.funcs[func];
        if info.ret_ty() != ColumnTy::Id {
            return;
        }
        let fingerprint_table = state.fingerprint_table;
        let node_hash_table = state.node_hash_table;
        let block_counter = state.block_counter;
        let crc32_func = state.crc32_func;
        let func_table = info.table;
        let func_name = info.name.to_string();
        let schema = info.schema.clone();

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
            for (entry, ty) in entries.iter().zip(schema.iter()) {
                if matches!(ty, ColumnTy::Id) {
                    id_entries.push(entry.clone());
                }
            }
            let zero = QueryEntry::Const {
                val: Value::new_const(0),
                ty: ColumnTy::Id,
            };
            let fingerprint_table_id = fingerprint_table.table;
            rb.add_callback(move |inner, rb| {
                let block = rb.read_counter(block_counter);
                for entry in &id_entries {
                    rb.insert(
                        fingerprint_table_id,
                        &[
                            inner.convert(entry),
                            inner.convert(&zero),
                            block.into(),
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
            for (entry, ty) in entries
                .iter()
                .take(schema.len() - 1)
                .zip(schema.iter().take(schema.len() - 1))
            {
                match ty {
                    ColumnTy::Id => {
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
                    ColumnTy::Base(_) => {
                        hash_inputs.push(entry.clone());
                    }
                }
            }
            let hash_var = rb.call_external_func(crc32_func, &hash_inputs, ColumnTy::Id, {
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

        if let Some(state) = self.partition_refinement.as_mut() {
            state.seed_rules.push(seed_rule);
            state.node_hash_rules.push(node_hash_rule);
        }
    }

    /// Run hash-based partition refinement until it stabilizes.
    pub fn run_hash_partition_refinement(&mut self) -> Result<bool> {
        let (seed_rules, node_hash_rules, propagate_rule, fingerprint_table, block_counter) =
            if let Some(state) = self.partition_refinement.as_ref() {
                (
                    state.seed_rules.clone(),
                    state.node_hash_rules.clone(),
                    state.propagate_rule,
                    state.fingerprint_table,
                    state.block_counter,
                )
            } else {
                return Err(anyhow!("partition refinement is not enabled"));
            };
        let mut refinement =
            PartitionRefinement::new(fingerprint_table, block_counter, self.timestamp_counter);
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
                // db.run_rule_set already merges, but ensure buffers are flushed.
                let merged = self.db.merge_all();
                changed |= report.changed || merged;
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
                let merged = self.db.merge_all();
                changed |= report.changed || merged;

                let report = run_rules_impl(
                    &mut self.db,
                    &mut self.rules,
                    &[propagate_rule],
                    ts,
                    self.report_level,
                )?;
                let merged = self.db.merge_all();
                changed |= report.changed || merged;
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
        Ok(changed_any)
    }
}
