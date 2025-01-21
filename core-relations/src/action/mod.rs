//! Instructions that are executed on the results of a query.
//!
//! This allows us to execute the "right-hand-side" of a rule. The
//! implementation here is optimized to execute on a batch of rows at a time.
use std::{mem, ops::Deref, sync::atomic::AtomicUsize};

use numeric_id::{DenseIdMap, NumericId};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use smallvec::SmallVec;

use crate::{
    common::{DashMap, Value},
    free_join::{CounterId, ExternalFunctionExt, TableId, TableInfo, Variable},
    pool::{with_pool_set, Clear, PoolSet, Pooled},
    primitives::PrimitiveFunctionId,
    table_spec::{ColumnId, MutationBuffer},
    Containers, ExternalFunctionId, Primitives, WrappedTable,
};

use self::mask::{Mask, MaskIter, ValueSource};

pub(crate) mod mask;

#[cfg(test)]
mod tests;

/// A representation of a value within a query or rule.
///
/// A QueryEntry is either a variable bound in a join, or an untyped constant.
#[derive(Copy, Clone, Debug)]
pub enum QueryEntry {
    Var(Variable),
    Const(Value),
}

impl From<Variable> for QueryEntry {
    fn from(var: Variable) -> Self {
        QueryEntry::Var(var)
    }
}

impl From<Value> for QueryEntry {
    fn from(val: Value) -> Self {
        QueryEntry::Const(val)
    }
}

/// A value that can be written to a table in an action.
#[derive(Debug, Clone, Copy)]
pub enum WriteVal {
    /// A variable or a constant.
    QueryEntry(QueryEntry),
    /// A fresh value from the given counter.
    IncCounter(CounterId),
    /// The value of the current row index.
    CurrentVal(usize),
}

impl<T> From<T> for WriteVal
where
    T: Into<QueryEntry>,
{
    fn from(val: T) -> Self {
        WriteVal::QueryEntry(val.into())
    }
}

impl From<CounterId> for WriteVal {
    fn from(ctr: CounterId) -> Self {
        WriteVal::IncCounter(ctr)
    }
}

/// A value that can be written to the database during a merge action.
#[derive(Debug, Copy, Clone)]
pub enum MergeVal {
    /// A fresh value from the given counter.
    Counter(CounterId),
    /// A standard constant value.
    Constant(Value),
}

impl From<CounterId> for MergeVal {
    fn from(ctr: CounterId) -> Self {
        MergeVal::Counter(ctr)
    }
}

impl From<Value> for MergeVal {
    fn from(val: Value) -> Self {
        MergeVal::Constant(val)
    }
}

pub(crate) type Bindings = DenseIdMap<Variable, Pooled<Vec<Value>>>;

#[derive(Default)]
pub(crate) struct PredictedVals {
    #[allow(clippy::type_complexity)]
    data: DashMap<(TableId, SmallVec<[Value; 3]>), Pooled<Vec<Value>>>,
}

impl Clear for PredictedVals {
    fn reuse(&self) -> bool {
        self.data.capacity() > 0
    }
    fn clear(&mut self) {
        if self.data.len() > 500 && rayon::current_num_threads() > 1 {
            self.data
                .shards_mut()
                .par_iter_mut()
                .for_each(|shard| shard.get_mut().clear());
        }
        self.data.clear()
    }
    fn bytes(&self) -> usize {
        self.data.capacity()
            * (std::mem::size_of::<(TableId, SmallVec<[Value; 3]>)>()
                + std::mem::size_of::<Pooled<Vec<Value>>>())
    }
}

impl PredictedVals {
    pub(crate) fn get_val(
        &self,
        table: TableId,
        key: &[Value],
        default: impl FnOnce() -> Pooled<Vec<Value>>,
    ) -> impl Deref<Target = Pooled<Vec<Value>>> + '_ {
        self.data
            .entry((table, SmallVec::from_slice(key)))
            .or_insert_with(default)
    }
}

#[derive(Copy, Clone)]
pub(crate) struct DbView<'a> {
    pub(crate) table_info: &'a DenseIdMap<TableId, TableInfo>,
    pub(crate) counters: &'a DenseIdMap<CounterId, AtomicUsize>,
    pub(crate) external_funcs: &'a DenseIdMap<ExternalFunctionId, Box<dyn ExternalFunctionExt>>,
    pub(crate) prims: &'a Primitives,
    pub(crate) containers: &'a Containers,
}

impl DbView<'_> {
    fn inc_counter(&self, ctr: CounterId) -> usize {
        self.counters[ctr].fetch_add(1, std::sync::atomic::Ordering::Release)
    }
    fn read_counter(&self, ctr: CounterId) -> usize {
        self.counters[ctr].load(std::sync::atomic::Ordering::Acquire)
    }
}

pub struct ExecutionState<'a> {
    pub(crate) predicted: &'a PredictedVals,
    pub(crate) db: DbView<'a>,
    pub(crate) buffers: DenseIdMap<TableId, Box<dyn MutationBuffer>>,
}

impl<'a> ExecutionState<'a> {
    pub fn new_handle(&self) -> ExecutionState<'a> {
        let mut res = ExecutionState {
            predicted: self.predicted,
            db: self.db,
            buffers: DenseIdMap::new(),
        };
        for (id, buf) in self.buffers.iter() {
            res.buffers.insert(id, buf.fresh_handle());
        }
        res
    }
    pub fn stage_insert(&mut self, table: TableId, vals: &[Value]) {
        self.buffers
            .get_or_insert(table, || self.db.table_info[table].table.new_buffer())
            .stage_insert(vals);
    }
    pub fn stage_remove(&mut self, table: TableId, vals: &[Value]) {
        self.buffers
            .get_or_insert(table, || self.db.table_info[table].table.new_buffer())
            .stage_remove(vals);
    }

    pub fn inc_counter(&self, ctr: CounterId) -> usize {
        self.db.inc_counter(ctr)
    }

    pub fn read_counter(&self, ctr: CounterId) -> usize {
        self.db.read_counter(ctr)
    }

    /// Get an immutable reference to the table with id `table`.
    pub fn get_table(&self, table: TableId) -> &WrappedTable {
        &self.db.table_info[table].table
    }

    pub fn prims(&self) -> &Primitives {
        self.db.prims
    }

    pub fn containers(&self) -> &Containers {
        self.db.containers
    }

    /// Get the _current_ value for a given key in `table`, or otherwise insert
    /// the unique _next_ value.
    ///
    /// Insertions into tables are not performed immediately, but rules and
    /// merge functions sometimes need to get the result of an insertion. For
    /// such cases, executions keep a cache of "predicted" values for a given
    /// mapping that manage the insertions, etc.
    pub fn predict_val(
        &mut self,
        table: TableId,
        key: &[Value],
        vals: impl ExactSizeIterator<Item = MergeVal>,
    ) -> Pooled<Vec<Value>> {
        with_pool_set(|ps| {
            if let Some(row) = self.db.table_info[table].table.get_row(key) {
                return row.vals;
            }
            Pooled::cloned(
                self.predicted
                    .get_val(table, key, || -> Pooled<Vec<Value>> {
                        let mut new = ps.get::<Vec<Value>>();
                        new.reserve(key.len() + vals.len());
                        new.extend_from_slice(key);
                        for val in vals {
                            new.push(match val {
                                MergeVal::Counter(ctr) => {
                                    Value::from_usize(self.db.inc_counter(ctr))
                                }
                                MergeVal::Constant(c) => c,
                            })
                        }
                        self.buffers
                            .get_or_insert(table, || self.db.table_info[table].table.new_buffer())
                            .stage_insert(&new);
                        new
                    })
                    .deref(),
            )
        })
    }
}

impl ExecutionState<'_> {
    pub(crate) fn run_instrs(&mut self, instrs: &[Instr], bindings: &mut Bindings) {
        let Some(batch_size) = bindings.iter().map(|(_, x)| x.len()).next() else {
            // Empty bindings; nothing to do.
            return;
        };
        with_pool_set(|ps| {
            let mut mask = Mask::new(0..batch_size, ps);
            for instr in instrs {
                if mask.is_empty() {
                    break;
                }
                self.run_instr(&mut mask, instr, bindings, ps);
            }
        })
    }
    fn run_instr(
        &mut self,
        mask: &mut Mask,
        inst: &Instr,
        bindings: &mut Bindings,
        pool_set: &PoolSet,
    ) {
        fn assert_impl(
            bindings: &mut Bindings,
            mask: &mut Mask,
            l: &QueryEntry,
            r: &QueryEntry,
            op: impl Fn(Value, Value) -> bool,
        ) {
            match (l, r) {
                (QueryEntry::Var(v1), QueryEntry::Var(v2)) => {
                    mask.iter(&bindings[*v1])
                        .zip(&bindings[*v2])
                        .retain(|(v1, v2)| op(*v1, *v2));
                }
                (QueryEntry::Var(v), QueryEntry::Const(c))
                | (QueryEntry::Const(c), QueryEntry::Var(v)) => {
                    mask.iter(&bindings[*v]).retain(|v| op(*v, *c));
                }
                (QueryEntry::Const(c1), QueryEntry::Const(c2)) => {
                    if !op(*c1, *c2) {
                        mask.clear();
                    }
                }
            }
        }

        // Helper macro for taking a slice of QueryEntries and creating a call
        // to `iter_dynamic` on `mask`.
        //
        // `iter_dynamic` takes a dynamically-determined number of "value
        // sources" (either a slice or a constant) and then does a masked
        // iteration on the "transpose" of these sources (row-wise).
        macro_rules! iter_entries {
            ($pool:expr, $entries:expr) => {
                iter_entries!(mask, $pool, $entries)
            };
            ($mask:expr, $pool:expr, $entries:expr) => {
                $mask.iter_dynamic(
                    $pool,
                    $entries.iter().map(|v| match v {
                        QueryEntry::Var(v) => {
                            debug_assert!(
                                bindings.get(*v).is_some(),
                                "variable {:?} not found in bindings {:?}",
                                v,
                                bindings
                            );
                            ValueSource::Slice(&bindings[*v])
                        }
                        QueryEntry::Const(c) => ValueSource::Const(*c),
                    }),
                )
            };
        }
        match inst {
            Instr::LookupOrInsertDefault {
                table: table_id,
                args,
                default,
                dst_col,
                dst_var,
            } => {
                let pool = pool_set.get_pool::<Vec<Value>>().clone();
                self.buffers.get_or_insert(*table_id, || {
                    self.db.table_info[*table_id].table.new_buffer()
                });
                let table = &self.db.table_info[*table_id].table;
                // Do two passes over the current vector. First, do a round of lookups. Then, for
                // any offsets where the lookup failed, insert the default value.
                let mut mask_copy = mask.clone();
                table.lookup_row_vectorized(&mut mask_copy, bindings, args, *dst_col, *dst_var);
                mask_copy.symmetric_difference(mask);
                if mask_copy.is_empty() {
                    return;
                }
                let mut out = mem::take(&mut bindings[*dst_var]);
                iter_entries!(mask_copy, pool, args).assign_vec(&mut out, |offset, key| {
                    // First, check if the entry is already in the table:
                    // if let Some(row) = table.get_row_column(&key, *dst_col) {
                    //     return row;
                    // }
                    // If not, insert the default value.
                    //
                    // We avoid doing this more than once by using the
                    // `predicted` map.
                    let prediction_key = (*table_id, SmallVec::<[Value; 3]>::from_slice(&key));
                    let buffers = &mut self.buffers;
                    // Bind some mutable references because the closure passed
                    // to or_insert_with is `move`.
                    let ctrs = &self.db.counters;
                    let bindings = &bindings;
                    let row = self
                        .predicted
                        .data
                        .entry(prediction_key)
                        .or_insert_with(move || {
                            let mut row = key;
                            // Extend the key with the default values.
                            row.reserve(default.len());
                            for val in default {
                                let val = match val {
                                    WriteVal::QueryEntry(QueryEntry::Const(c)) => *c,
                                    WriteVal::QueryEntry(QueryEntry::Var(v)) => {
                                        bindings[*v][offset]
                                    }
                                    WriteVal::IncCounter(ctr) => Value::from_usize(
                                        ctrs[*ctr]
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                                    ),
                                    WriteVal::CurrentVal(ix) => row[*ix],
                                };
                                row.push(val)
                            }
                            // Insert it into the table.
                            buffers.get_mut(*table_id).unwrap().stage_insert(&row);
                            row
                        });
                    row[dst_col.index()]
                });
                bindings.insert(*dst_var, out);
            }
            Instr::LookupWithDefault {
                table,
                args,
                dst_col,
                dst_var,
                default,
            } => {
                let table = &self.db.table_info[*table].table;
                table.lookup_with_default_vectorized(
                    mask, bindings, args, *dst_col, *default, *dst_var,
                );
            }
            Instr::Lookup {
                table,
                args,
                dst_col,
                dst_var,
            } => {
                let table = &self.db.table_info[*table].table;
                table.lookup_row_vectorized(mask, bindings, args, *dst_col, *dst_var);
            }
            Instr::Insert { table, vals } => {
                let pool = pool_set.get_pool::<Vec<Value>>().clone();
                iter_entries!(pool, vals).for_each(|vals| {
                    self.stage_insert(*table, &vals);
                })
            }
            Instr::InsertIfEq { table, l, r, vals } => {
                let pool = pool_set.get_pool::<Vec<Value>>().clone();
                match (l, r) {
                    (QueryEntry::Var(v1), QueryEntry::Var(v2)) => iter_entries!(pool, vals)
                        .zip(&bindings[*v1])
                        .zip(&bindings[*v2])
                        .for_each(|((vals, v1), v2)| {
                            if v1 == v2 {
                                self.stage_insert(*table, &vals);
                            }
                        }),
                    (QueryEntry::Var(v), QueryEntry::Const(c))
                    | (QueryEntry::Const(c), QueryEntry::Var(v)) => iter_entries!(pool, vals)
                        .zip(&bindings[*v])
                        .for_each(|(vals, cond)| {
                            if cond == c {
                                self.stage_insert(*table, &vals);
                            }
                        }),
                    (QueryEntry::Const(c1), QueryEntry::Const(c2)) => {
                        if c1 == c2 {
                            iter_entries!(pool, vals).for_each(|vals| {
                                self.stage_insert(*table, &vals);
                            })
                        }
                    }
                }
            }
            Instr::Remove { table, args } => {
                let pool = pool_set.get_pool::<Vec<Value>>().clone();
                iter_entries!(pool, args).for_each(|args| {
                    self.stage_remove(*table, &args);
                })
            }
            Instr::Prim { func, args, dst } => {
                let pool = pool_set.get_pool::<Vec<Value>>().clone();
                self.db
                    .prims
                    .apply_vectorized(*func, pool, mask, bindings, args, *dst);
            }
            Instr::External { func, args, dst } => {
                self.db.external_funcs[*func].invoke_batch(self, mask, bindings, args, *dst);
            }
            Instr::AssertAnyNe { ops, divider } => {
                let pool = pool_set.get_pool::<Vec<Value>>().clone();
                iter_entries!(pool, ops).retain(|vals| {
                    vals[0..*divider]
                        .iter()
                        .zip(&vals[*divider..])
                        .any(|(l, r)| l != r)
                })
            }
            Instr::AssertEq(l, r) => assert_impl(bindings, mask, l, r, |l, r| l == r),
            Instr::AssertNe(l, r) => assert_impl(bindings, mask, l, r, |l, r| l != r),
        }
    }
}

#[derive(Debug)]
pub(crate) enum Instr {
    /// Look up the value of the given table, inserting a new entry with a
    /// default value if it is not there.
    LookupOrInsertDefault {
        table: TableId,
        args: Vec<QueryEntry>,
        default: Vec<WriteVal>,
        dst_col: ColumnId,
        dst_var: Variable,
    },

    /// Look up the value of the given table; if the value is not there, use the
    /// given default.
    LookupWithDefault {
        table: TableId,
        args: Vec<QueryEntry>,
        dst_col: ColumnId,
        dst_var: Variable,
        default: QueryEntry,
    },

    /// Look up a value of the given table, halting execution if it is not
    /// there.
    Lookup {
        table: TableId,
        args: Vec<QueryEntry>,
        dst_col: ColumnId,
        dst_var: Variable,
    },

    /// Insert the given return value value with the provided arguments into the
    /// table.
    Insert {
        table: TableId,
        vals: Vec<QueryEntry>,
    },

    /// Insert `vals` into `table` if `l` and `r` are equal.
    InsertIfEq {
        table: TableId,
        l: QueryEntry,
        r: QueryEntry,
        vals: Vec<QueryEntry>,
    },

    /// Remove the entry corresponding to `args` in `func`.
    Remove {
        table: TableId,
        args: Vec<QueryEntry>,
    },

    /// Bind the result of a primitive function to a variable.
    Prim {
        func: PrimitiveFunctionId,
        args: Vec<QueryEntry>,
        dst: Variable,
    },

    /// Bind the result of the external function to a variable.
    External {
        func: ExternalFunctionId,
        args: Vec<QueryEntry>,
        dst: Variable,
    },

    /// Continue execution iff the two variables are equal.
    AssertEq(QueryEntry, QueryEntry),

    /// Continue execution iff the two variables are not equal.
    AssertNe(QueryEntry, QueryEntry),

    /// For the two slices: ops[0..divider] and ops[divider..], continue
    /// execution iff there is one pair of values at the same offset that are
    /// not equal.
    AssertAnyNe {
        ops: Vec<QueryEntry>,
        divider: usize,
    },
}
