//! Instructions that are executed on the results of a query.
//!
//! This allows us to execute the "right-hand-side" of a rule. The
//! implementation here is optimized to execute on a batch of rows at a time.
use std::ops::Deref;

use numeric_id::{DenseIdMap, NumericId};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use smallvec::SmallVec;

use crate::{
    common::{DashMap, Value},
    free_join::{CounterId, Counters, ExternalFunctions, TableId, TableInfo, Variable},
    pool::{with_pool_set, Clear, Pooled},
    table_spec::{ColumnId, MutationBuffer},
    BaseValues, ContainerValues, ExternalFunctionId, WrappedTable,
};

use self::mask::{Mask, MaskIter, ValueSource};

#[macro_use]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

/// Bindings store a sequence of values for a given set of variables.
///
/// The intent of bindings is to store a sequence of mappings from [`Variable`] to [`Value`], in a
/// struct-of-arrays style that is better laid out for processing bindings in batches.
#[derive(Debug)]
pub(crate) struct Bindings {
    matches: usize,
    /// The maximum number of calls to `push` that we can receive before we clear the
    /// [`Bindings`].
    // This is used to preallocate chunks of the flat `data` vector.
    max_batch_size: usize,
    data: Pooled<Vec<Value>>,
    /// Points into `data`. data[vars[var].. vars[var]+matches]` contains the values for `data`.
    var_offsets: DenseIdMap<Variable, usize>,
}

impl std::ops::Index<Variable> for Bindings {
    type Output = [Value];
    fn index(&self, var: Variable) -> &[Value] {
        self.get(var).unwrap()
    }
}

impl Bindings {
    pub(crate) fn new(max_batch_size: usize) -> Self {
        Bindings {
            matches: 0,
            max_batch_size,
            data: Default::default(),
            var_offsets: DenseIdMap::new(),
        }
    }
    fn assert_invariant(&self) {
        #[cfg(debug_assertions)]
        {
            assert!(self.matches <= self.max_batch_size);
            for (var, start) in self.var_offsets.iter() {
                assert!(
                    start + self.matches <= self.data.len(),
                    "Variable {:?} starts at {}, but data only has {} elements",
                    var,
                    start,
                    self.data.len()
                );
            }
        }
    }

    pub(crate) fn clear(&mut self) {
        self.matches = 0;
        self.var_offsets.clear();
        self.data.clear();
        self.assert_invariant();
    }

    fn get(&self, var: Variable) -> Option<&[Value]> {
        let start = self.var_offsets.get(var)?;
        Some(&self.data[*start..*start + self.matches])
    }

    fn add_mapping(&mut self, var: Variable, vals: &[Value]) {
        let start = self.data.len();
        self.data.extend_from_slice(vals);
        // We have a flat representation of the data, meaning that writing more than
        // `max_batch_size` values to `var` could overwrite values for a different variable, which
        // would produce some mysterious results that are hard to debug.
        debug_assert!(vals.len() <= self.max_batch_size);
        if vals.len() < self.max_batch_size {
            let target_len = self.data.len() + self.max_batch_size - vals.len();
            self.data.resize(target_len, Value::stale());
        }
        self.var_offsets.insert(var, start);
    }

    pub(crate) fn insert(&mut self, var: Variable, vals: &[Value]) {
        if self.var_offsets.n_ids() == 0 {
            self.matches = vals.len();
        } else {
            assert_eq!(self.matches, vals.len());
        }
        self.add_mapping(var, vals);
        self.assert_invariant();
    }

    /// Push a new set of bindings for the given variables.
    ///
    /// # Safety:
    /// This method assumes that all calls to `push`:
    /// * Have a mapping for every member of `used_vars`.
    /// * Are passed the same `used_vars`.
    ///
    /// It is unsafe to avoid bounds-checking. This method is called extremely frequently and the
    /// overhead of boundschecking is noticeable.
    pub(crate) unsafe fn push(
        &mut self,
        map: &DenseIdMap<Variable, Value>,
        used_vars: &[Variable],
    ) {
        if self.matches != 0 {
            assert!(self.matches < self.max_batch_size);
            #[cfg(debug_assertions)]
            {
                for var in used_vars {
                    assert!(
                        self.var_offsets.get(*var).is_some(),
                        "Variable {:?} not found in bindings {:?}",
                        var,
                        self.var_offsets
                    );
                }
            }
            for var in used_vars {
                let var = var.index();
                // Safe version: this degrades some benchmarks by ~6%
                // let start = self.var_offsets.raw()[var].unwrap();
                // self.data[start + self.matches] = map.raw()[var].unwrap();
                unsafe {
                    let start = self.var_offsets.raw().get_unchecked(var).unwrap_unchecked();
                    *self.data.get_unchecked_mut(start + self.matches) =
                        map.raw().get_unchecked(var).unwrap_unchecked();
                }
            }
        } else {
            for (var, val) in map.iter() {
                self.add_mapping(var, &[*val]);
            }
        }

        self.matches += 1;
        self.assert_invariant();
    }

    /// A method that removes the bindings for the given variable and allows for its values to be
    /// used independently from the [`Bindings`] struct. This is helpful when an operation needs to
    /// mutably borrow the values for one value while reading the values for another.
    ///
    /// To add the values back, use [`Bindings::replace`].
    pub(crate) fn take(&mut self, var: Variable) -> Option<ExtractedBinding> {
        let mut vals: Pooled<Vec<Value>> = with_pool_set(|ps| ps.get());
        vals.extend_from_slice(self.get(var)?);
        let start = self.var_offsets.take(var)?;
        Some(ExtractedBinding {
            var,
            offset: start,
            vals,
        })
    }

    /// Replace a binding extracted with [`Bindings::take`].
    ///
    /// # Panics
    /// This method will panic if the length of the values in `bdg` does not match the current
    /// number of matches in `Bindings`. It may panic if `bdg` was extracted from a different
    /// [`Bindings`] than the one it is being replaced in.
    pub(crate) fn replace(&mut self, bdg: ExtractedBinding) {
        // Replace the binding with the new values.
        let ExtractedBinding {
            var,
            offset,
            mut vals,
        } = bdg;
        assert_eq!(vals.len(), self.matches);
        self.data
            .splice(offset..offset + self.matches, vals.drain(..));
        self.var_offsets.insert(var, offset);
    }
}

/// A binding that has been extracted from a [`Bindings`] struct via the [`Bindings::take`] method.
///
/// This allows for a variable's contents to be read while the [`Bindings`] struct has been
/// borrowed mutably. The contents will not be readable until [`Bindings::replace`] is called.
pub(crate) struct ExtractedBinding {
    var: Variable,
    offset: usize,
    pub(crate) vals: Pooled<Vec<Value>>,
}

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
    pub(crate) counters: &'a Counters,
    pub(crate) external_funcs: &'a ExternalFunctions,
    pub(crate) bases: &'a BaseValues,
    pub(crate) containers: &'a ContainerValues,
}

/// A handle on a database that may be in the process of running a rule.
///
/// An ExecutionState grants immutable access to the (much of) the database, and also provides a
/// limited API to mutate database contents.
///
/// A few important notes:
///
/// ## Some tables may be missing
/// Callers external to this crate cannot construct an `ExecutionState` directly. Depending on the
/// context, some tables may not be available. In particular, when running [`crate::Table::merge`]
/// operations, only a table's read-side dependencies are available for reading (sim. for writing).
/// This allows tables that do not need access to one another to be merged in parallel.
///
/// When executing a rule, ExecutionState has access to all tables.
///
/// ## Limited Mutability
/// Callers can only stage insertsions and deletions to tables. These changes are not applied until
/// the next call to `merge` on the underlying table.
///
/// ## Predicted Values
/// ExecutionStates provide a means of synchronizing the results of a pending write across
/// different executions of a rule. This is particularly important in the case where the result of
/// an operation (such as "lookup or insert new id" operatiosn) is a fresh id. A common
/// ExecutionState ensures that future lookups will see the same id (even across calls to
/// [`ExecutionState::clone`]).
pub struct ExecutionState<'a> {
    pub(crate) predicted: &'a PredictedVals,
    pub(crate) db: DbView<'a>,
    pub(crate) buffers: DenseIdMap<TableId, Box<dyn MutationBuffer>>,
    /// Whether any mutations have been staged via this ExecutionState.
    pub(crate) changed: bool,
}

impl Clone for ExecutionState<'_> {
    fn clone(&self) -> Self {
        let mut res = ExecutionState {
            predicted: self.predicted,
            db: self.db,
            buffers: DenseIdMap::new(),
            changed: false,
        };
        for (id, buf) in self.buffers.iter() {
            res.buffers.insert(id, buf.fresh_handle());
        }
        res
    }
}

impl<'a> ExecutionState<'a> {
    pub(crate) fn new(
        predicted: &'a PredictedVals,
        db: DbView<'a>,
        buffers: DenseIdMap<TableId, Box<dyn MutationBuffer>>,
    ) -> Self {
        ExecutionState {
            predicted,
            db,
            buffers,
            changed: false,
        }
    }

    /// Stage an insertion of the given row into `table`.
    ///
    /// If you are using `egglog`, consider using `egglog_bridge::TableAction`.
    pub fn stage_insert(&mut self, table: TableId, row: &[Value]) {
        self.buffers
            .get_or_insert(table, || self.db.table_info[table].table.new_buffer())
            .stage_insert(row);
        self.changed = true;
    }

    /// Stage a removal of the given row from `table` if it is present.
    ///
    /// If you are using `egglog`, consider using `egglog_bridge::TableAction`.
    pub fn stage_remove(&mut self, table: TableId, key: &[Value]) {
        self.buffers
            .get_or_insert(table, || self.db.table_info[table].table.new_buffer())
            .stage_remove(key);
        self.changed = true;
    }

    /// Call an external function.
    pub fn call_external_func(
        &mut self,
        func: ExternalFunctionId,
        args: &[Value],
    ) -> Option<Value> {
        self.db.external_funcs[func].invoke(self, args)
    }

    pub fn inc_counter(&self, ctr: CounterId) -> usize {
        self.db.counters.inc(ctr)
    }

    pub fn read_counter(&self, ctr: CounterId) -> usize {
        self.db.counters.read(ctr)
    }

    /// Get an immutable reference to the table with id `table`.
    /// Dangerous: Reading from a table during action execution may break the semi-naive evaluation
    pub fn get_table(&self, table: TableId) -> &'a WrappedTable {
        &self.db.table_info[table].table
    }

    pub fn base_values(&self) -> &BaseValues {
        self.db.bases
    }

    pub fn container_values(&self) -> &'a ContainerValues {
        self.db.containers
    }

    /// Get the _current_ value for a given key in `table`, or otherwise insert
    /// the unique _next_ value.
    ///
    /// Insertions into tables are not performed immediately, but rules and
    /// merge functions sometimes need to get the result of an insertion. For
    /// such cases, executions keep a cache of "predicted" values for a given
    /// mapping that manage the insertions, etc.
    ///
    /// If you are using `egglog`, consider using `egglog_bridge::TableAction`.
    pub fn predict_val(
        &mut self,
        table: TableId,
        key: &[Value],
        vals: impl ExactSizeIterator<Item = MergeVal>,
    ) -> Pooled<Vec<Value>> {
        if let Some(row) = self.db.table_info[table].table.get_row(key) {
            return row.vals;
        }
        Pooled::cloned(
            self.predicted
                .get_val(table, key, || {
                    Self::construct_new_row(
                        &self.db,
                        &mut self.buffers,
                        &mut self.changed,
                        table,
                        key,
                        vals,
                    )
                })
                .deref(),
        )
    }

    fn construct_new_row(
        db: &DbView,
        buffers: &mut DenseIdMap<TableId, Box<dyn MutationBuffer>>,
        changed: &mut bool,
        table: TableId,
        key: &[Value],
        vals: impl ExactSizeIterator<Item = MergeVal>,
    ) -> Pooled<Vec<Value>> {
        with_pool_set(|ps| {
            let mut new = ps.get::<Vec<Value>>();
            new.reserve(key.len() + vals.len());
            new.extend_from_slice(key);
            for val in vals {
                new.push(match val {
                    MergeVal::Counter(ctr) => Value::from_usize(db.counters.inc(ctr)),
                    MergeVal::Constant(c) => c,
                })
            }
            buffers
                .get_or_insert(table, || db.table_info[table].table.new_buffer())
                .stage_insert(&new);
            *changed = true;
            new
        })
    }

    /// A variant of [`ExecutionState::predict_val`] that avoids materializing the full row, and
    /// instead only returns the value in the given column.
    pub fn predict_col(
        &mut self,
        table: TableId,
        key: &[Value],
        vals: impl ExactSizeIterator<Item = MergeVal>,
        col: ColumnId,
    ) -> Value {
        if let Some(val) = self.db.table_info[table].table.get_row_column(key, col) {
            return val;
        }
        self.predicted.get_val(table, key, || {
            Self::construct_new_row(
                &self.db,
                &mut self.buffers,
                &mut self.changed,
                table,
                key,
                vals,
            )
        })[col.index()]
    }
}

impl ExecutionState<'_> {
    pub(crate) fn run_instrs(&mut self, instrs: &[Instr], bindings: &mut Bindings) {
        if bindings.var_offsets.next_id().rep() == 0 {
            // If we have no variables, we want to run the rules once.
            bindings.matches = 1;
        }

        // Vectorized execution for larger batch sizes
        let mut mask = with_pool_set(|ps| Mask::new(0..bindings.matches, ps));
        for instr in instrs {
            if mask.is_empty() {
                break;
            }
            self.run_instr(&mut mask, instr, bindings);
        }
    }
    fn run_instr(&mut self, mask: &mut Mask, inst: &Instr, bindings: &mut Bindings) {
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

        match inst {
            Instr::LookupOrInsertDefault {
                table: table_id,
                args,
                default,
                dst_col,
                dst_var,
            } => {
                let pool = with_pool_set(|ps| ps.get_pool::<Vec<Value>>().clone());
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
                let mut out = bindings.take(*dst_var).unwrap();
                for_each_binding_with_mask!(mask_copy, args.as_slice(), bindings, |iter| {
                    iter.assign_vec(&mut out.vals, |offset, key| {
                        // First, check if the entry is already in the table:
                        // if let Some(row) = table.get_row_column(&key, *dst_col) {
                        //     return row;
                        // }
                        // If not, insert the default value.
                        //
                        // We avoid doing this more than once by using the
                        // `predicted` map.
                        let prediction_key = (
                            *table_id,
                            SmallVec::<[Value; 3]>::from_slice(key.as_slice()),
                        );
                        let buffers = &mut self.buffers;
                        // Bind some mutable references because the closure passed
                        // to or_insert_with is `move`.
                        let ctrs = &self.db.counters;
                        let bindings = &bindings;
                        let pool = pool.clone();
                        let row =
                            self.predicted
                                .data
                                .entry(prediction_key)
                                .or_insert_with(move || {
                                    let mut row = pool.get();
                                    row.extend_from_slice(key.as_slice());
                                    // Extend the key with the default values.
                                    row.reserve(default.len());
                                    for val in default {
                                        let val = match val {
                                            WriteVal::QueryEntry(QueryEntry::Const(c)) => *c,
                                            WriteVal::QueryEntry(QueryEntry::Var(v)) => {
                                                bindings[*v][offset]
                                            }
                                            WriteVal::IncCounter(ctr) => {
                                                Value::from_usize(ctrs.inc(*ctr))
                                            }
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
                });
                bindings.replace(out);
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

            Instr::LookupWithFallback {
                table: table_id,
                table_key,
                func,
                func_args,
                dst_col,
                dst_var,
            } => {
                let table = &self.db.table_info[*table_id].table;
                let mut lookup_result = mask.clone();
                table.lookup_row_vectorized(
                    &mut lookup_result,
                    bindings,
                    table_key,
                    *dst_col,
                    *dst_var,
                );
                let mut to_call_func = lookup_result.clone();
                to_call_func.symmetric_difference(mask);
                if to_call_func.is_empty() {
                    return;
                }

                // Call the given external function on all entries where the lookup failed.
                self.db.external_funcs[*func].invoke_batch_assign(
                    self,
                    &mut to_call_func,
                    bindings,
                    func_args,
                    *dst_var,
                );
                // The new mask should be the lanes where the lookup succeeded or where `func`
                // succeeded.
                lookup_result.union(&to_call_func);
                *mask = lookup_result;
            }
            Instr::Insert { table, vals } => {
                for_each_binding_with_mask!(mask, vals.as_slice(), bindings, |iter| {
                    iter.for_each(|vals| {
                        self.stage_insert(*table, vals.as_slice());
                    })
                });
            }
            Instr::InsertIfEq { table, l, r, vals } => match (l, r) {
                (QueryEntry::Var(v1), QueryEntry::Var(v2)) => {
                    for_each_binding_with_mask!(mask, vals.as_slice(), bindings, |iter| {
                        iter.zip(&bindings[*v1])
                            .zip(&bindings[*v2])
                            .for_each(|((vals, v1), v2)| {
                                if v1 == v2 {
                                    self.stage_insert(*table, &vals);
                                }
                            })
                    })
                }
                (QueryEntry::Var(v), QueryEntry::Const(c))
                | (QueryEntry::Const(c), QueryEntry::Var(v)) => {
                    for_each_binding_with_mask!(mask, vals.as_slice(), bindings, |iter| {
                        iter.zip(&bindings[*v]).for_each(|(vals, cond)| {
                            if cond == c {
                                self.stage_insert(*table, &vals);
                            }
                        })
                    })
                }
                (QueryEntry::Const(c1), QueryEntry::Const(c2)) => {
                    if c1 == c2 {
                        for_each_binding_with_mask!(mask, vals.as_slice(), bindings, |iter| iter
                            .for_each(|vals| {
                                self.stage_insert(*table, &vals);
                            }))
                    }
                }
            },
            Instr::Remove { table, args } => {
                for_each_binding_with_mask!(mask, args.as_slice(), bindings, |iter| {
                    iter.for_each(|args| {
                        self.stage_remove(*table, args.as_slice());
                    })
                });
            }
            Instr::External { func, args, dst } => {
                self.db.external_funcs[*func].invoke_batch(self, mask, bindings, args, *dst);
            }
            Instr::ExternalWithFallback {
                f1,
                args1,
                f2,
                args2,
                dst,
            } => {
                let mut f1_result = mask.clone();
                self.db.external_funcs[*f1].invoke_batch(
                    self,
                    &mut f1_result,
                    bindings,
                    args1,
                    *dst,
                );
                let mut to_call_f2 = f1_result.clone();
                to_call_f2.symmetric_difference(mask);
                if to_call_f2.is_empty() {
                    return;
                }
                // Call the given external function on all entries where the first call failed.
                self.db.external_funcs[*f2].invoke_batch_assign(
                    self,
                    &mut to_call_f2,
                    bindings,
                    args2,
                    *dst,
                );
                // The new mask should be the lanes where either `f1` or `f2` succeeded.
                f1_result.union(&to_call_f2);
                *mask = f1_result;
            }
            Instr::AssertAnyNe { ops, divider } => {
                for_each_binding_with_mask!(mask, ops.as_slice(), bindings, |iter| {
                    iter.retain(|vals| {
                        vals[0..*divider]
                            .iter()
                            .zip(&vals[*divider..])
                            .any(|(l, r)| l != r)
                    })
                })
            }
            Instr::AssertEq(l, r) => assert_impl(bindings, mask, l, r, |l, r| l == r),
            Instr::AssertNe(l, r) => assert_impl(bindings, mask, l, r, |l, r| l != r),
            Instr::ReadCounter { counter, dst } => {
                let mut vals = with_pool_set(|ps| ps.get::<Vec<Value>>());
                let ctr_val = Value::from_usize(self.read_counter(*counter));
                vals.resize(bindings.matches, ctr_val);
                bindings.insert(*dst, &vals);
            }
        }
    }
}

#[derive(Debug, Clone)]
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

    /// Look up the given key in the table: if the value is not present in the given table, then
    /// call the given external function with the given arguments. If the external function returns
    /// a value, that value is returned in the given `dst_var`. If the lookup fails and the
    /// external function does not return a value, then execution is halted.
    LookupWithFallback {
        table: TableId,
        table_key: Vec<QueryEntry>,
        func: ExternalFunctionId,
        func_args: Vec<QueryEntry>,
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

    /// Bind the result of the external function to a variable.
    External {
        func: ExternalFunctionId,
        args: Vec<QueryEntry>,
        dst: Variable,
    },

    /// Bind the result of the external function to a variable. If the first external function
    /// fails, then use the second external function. If both fail, execution is haulted, (as in a
    /// single failure of `External`).
    ExternalWithFallback {
        f1: ExternalFunctionId,
        args1: Vec<QueryEntry>,
        f2: ExternalFunctionId,
        args2: Vec<QueryEntry>,
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

    /// Read the value of a counter and write it to the given variable.
    ReadCounter {
        /// The counter to broadcast.
        counter: CounterId,
        /// The variable to write the value to.
        dst: Variable,
    },
}
