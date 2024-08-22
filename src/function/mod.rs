use std::mem;

use crate::*;
use index::*;
use smallvec::SmallVec;

mod binary_search;
pub mod index;
pub(crate) mod table;

pub type ValueVec = SmallVec<[Value; 3]>;

#[derive(Clone)]
pub struct Function {
    pub(crate) decl: ResolvedFunctionDecl,
    pub schema: ResolvedSchema,
    pub merge: MergeAction,
    pub(crate) nodes: table::Table,
    sorts: HashSet<Symbol>,
    pub(crate) indexes: Vec<Rc<ColumnIndex>>,
    pub(crate) rebuild_indexes: Vec<Option<CompositeColumnIndex>>,
    index_updated_through: usize,
    updates: usize,
    scratch: IndexSet<usize>,
}

#[derive(Clone)]
pub struct MergeAction {
    pub on_merge: Option<Rc<Program>>,
    pub merge_vals: MergeFn,
}

#[derive(Clone)]
pub enum MergeFn {
    AssertEq,
    Union,
    // the rc is make sure it's cheaply clonable, since calling the merge fn
    // requires a clone
    Expr(Rc<Program>),
}

/// All information we know determined by the input.
#[derive(Debug, Clone)]
pub struct TupleOutput {
    pub value: Value,
    pub timestamp: u32,
    pub subsumed: bool,
}

#[derive(Clone, Debug)]
pub struct ResolvedSchema {
    pub input: Vec<ArcSort>,
    pub output: ArcSort,
}

impl ResolvedSchema {
    pub fn get_by_pos(&self, index: usize) -> Option<&ArcSort> {
        if self.input.len() == index {
            Some(&self.output)
        } else {
            self.input.get(index)
        }
    }
}

impl Debug for Function {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Function")
            .field("decl", &self.decl)
            .field("schema", &self.schema)
            .field("nodes", &self.nodes)
            .field("indexes", &self.indexes)
            .field("rebuild_indexes", &self.rebuild_indexes)
            .field("index_updated_through", &self.index_updated_through)
            .field("updates", &self.updates)
            .field("scratch", &self.scratch)
            .finish()
    }
}

/// A non-Union merge discovered during rebuilding that has to be applied before
/// resuming execution.
pub(crate) type DeferredMerge = (ValueVec, Value, Value);

impl Function {
    pub(crate) fn new(egraph: &EGraph, decl: &ResolvedFunctionDecl) -> Result<Self, Error> {
        let mut input = Vec::with_capacity(decl.schema.input.len());
        for s in &decl.schema.input {
            input.push(match egraph.type_info().sorts.get(s) {
                Some(sort) => sort.clone(),
                None => {
                    return Err(Error::TypeError(TypeError::UndefinedSort(
                        *s,
                        decl.span.clone(),
                    )))
                }
            })
        }

        let output = match egraph.type_info().sorts.get(&decl.schema.output) {
            Some(sort) => sort.clone(),
            None => {
                return Err(Error::TypeError(TypeError::UndefinedSort(
                    decl.schema.output,
                    decl.span.clone(),
                )))
            }
        };

        let binding = IndexSet::from_iter([
            ResolvedVar {
                name: Symbol::from("old"),
                sort: output.clone(),
                is_global_ref: false,
            },
            ResolvedVar {
                name: Symbol::from("new"),
                sort: output.clone(),
                is_global_ref: false,
            },
        ]);

        // Invariant: the last element in the stack is the return value.
        let merge_vals = if let Some(merge_expr) = &decl.merge {
            let (actions, mapped_expr) = merge_expr.to_core_actions(
                egraph.type_info(),
                &mut binding.clone(),
                &mut ResolvedGen::new("$".to_string()),
            )?;
            let target = mapped_expr.get_corresponding_var_or_lit(egraph.type_info());
            let program = egraph
                .compile_expr(&binding, &actions, &target)
                .map_err(Error::TypeErrors)?;
            MergeFn::Expr(Rc::new(program))
        } else if output.is_eq_sort() {
            MergeFn::Union
        } else {
            MergeFn::AssertEq
        };

        let on_merge = if decl.merge_action.is_empty() {
            None
        } else {
            let (merge_action, _) = decl.merge_action.to_core_actions(
                egraph.type_info(),
                &mut binding.clone(),
                &mut ResolvedGen::new("$".to_string()),
            )?;
            let program = egraph
                .compile_actions(&binding, &merge_action)
                .map_err(Error::TypeErrors)?;
            Some(Rc::new(program))
        };

        let indexes = Vec::from_iter(
            input
                .iter()
                .chain(once(&output))
                .map(|x| Rc::new(ColumnIndex::new(x.name()))),
        );

        let rebuild_indexes = Vec::from_iter(input.iter().chain(once(&output)).map(|x| {
            if x.is_eq_container_sort() {
                Some(CompositeColumnIndex::new())
            } else {
                None
            }
        }));

        let sorts: HashSet<Symbol> = input
            .iter()
            .map(|x| x.name())
            .chain(once(output.name()))
            .collect();

        Ok(Function {
            decl: decl.clone(),
            schema: ResolvedSchema { input, output },
            nodes: Default::default(),
            scratch: Default::default(),
            sorts,
            // TODO: build indexes for primitive sorts lazily
            indexes,
            rebuild_indexes,
            index_updated_through: 0,
            updates: 0,
            merge: MergeAction {
                on_merge,
                merge_vals,
            },
        })
    }

    pub fn get(&self, inputs: &[Value]) -> Option<Value> {
        self.nodes.get(inputs).map(|output| output.value)
    }

    pub fn insert(&mut self, inputs: &[Value], value: Value, timestamp: u32) -> Option<Value> {
        self.insert_internal(inputs, value, timestamp, true)
    }
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.indexes
            .iter_mut()
            .for_each(|x| Rc::make_mut(x).clear());
        self.rebuild_indexes.iter_mut().for_each(|x| {
            if let Some(x) = x {
                x.clear()
            }
        });
        self.index_updated_through = 0;
    }
    pub fn insert_internal(
        &mut self,
        inputs: &[Value],
        value: Value,
        timestamp: u32,
        // Clean out all stale entries if they account for a sufficiently large
        // portion of the table after this entry is inserted.
        maybe_rehash: bool,
    ) -> Option<Value> {
        if cfg!(debug_assertions) {
            for (v, sort) in inputs
                .iter()
                .zip(self.schema.input.iter())
                .chain(once((&value, &self.schema.output)))
            {
                assert_eq!(sort.name(), v.tag);
            }
        }
        let res = self.nodes.insert(inputs, value, timestamp);
        if maybe_rehash {
            self.maybe_rehash();
        }
        res
    }

    /// Mark the given inputs as subsumed.
    pub fn subsume(&mut self, inputs: &[Value]) {
        self.nodes.get_mut(inputs).unwrap().subsumed = true;
    }

    /// Return a column index that contains (a superset of) the offsets for the
    /// given column. This method can return nothing if the indexes available
    /// contain too many irrelevant offsets.
    pub(crate) fn column_index(
        &self,
        col: usize,
        timestamps: &Range<u32>,
    ) -> Option<Rc<ColumnIndex>> {
        let range = self.nodes.transform_range(timestamps);
        if range.end > self.index_updated_through {
            return None;
        }
        let size = range.end.saturating_sub(range.start);
        // If this represents >12.5% overhead, don't use the index
        if (self.nodes.num_offsets() - size) > (size / 8) {
            return None;
        }
        let target = &self.indexes[col];
        Some(target.clone())
    }

    pub(crate) fn remove(&mut self, ks: &[Value], ts: u32) -> bool {
        let res = self.nodes.remove(ks, ts);
        self.maybe_rehash();
        res
    }

    pub(crate) fn clear_updates(&mut self) -> usize {
        mem::take(&mut self.updates)
    }

    fn build_indexes(&mut self, offsets: Range<usize>) {
        for (col, (index, rebuild_index)) in self
            .indexes
            .iter_mut()
            .zip(self.rebuild_indexes.iter_mut())
            .enumerate()
        {
            let as_mut = Rc::make_mut(index);
            if col == self.schema.input.len() {
                for (slot, _, out) in self.nodes.iter_range(offsets.clone(), true) {
                    as_mut.add(out.value, slot)
                }
            } else {
                for (slot, inp, _) in self.nodes.iter_range(offsets.clone(), true) {
                    as_mut.add(inp[col], slot)
                }
            }

            // rebuild_index
            if let Some(rebuild_index) = rebuild_index {
                if col == self.schema.input.len() {
                    for (slot, _, out) in self.nodes.iter_range(offsets.clone(), true) {
                        self.schema.output.foreach_tracked_values(
                            &out.value,
                            Box::new(|value| rebuild_index.add(value, slot)),
                        )
                    }
                } else {
                    for (slot, inp, _) in self.nodes.iter_range(offsets.clone(), true) {
                        self.schema.input[col].foreach_tracked_values(
                            &inp[col],
                            Box::new(|value| rebuild_index.add(value, slot)),
                        )
                    }
                }
            }
        }
    }

    fn update_indexes(&mut self, through: usize) {
        self.build_indexes(self.index_updated_through..through);
        self.index_updated_through = self.index_updated_through.max(through);
    }

    fn maybe_rehash(&mut self) {
        if !self.nodes.too_stale() {
            return;
        }

        for index in &mut self.indexes {
            // Everything works if we don't have a unique copy of the indexes,
            // but we ought to be able to avoid this copy.
            Rc::make_mut(index).clear();
        }
        for rebuild_index in self.rebuild_indexes.iter_mut().flatten() {
            rebuild_index.clear();
        }
        self.nodes.rehash();
        self.index_updated_through = 0;
        if self.nodes.is_empty() {
            return;
        }
        self.update_indexes(self.nodes.num_offsets());
    }

    pub(crate) fn iter_timestamp_range(
        &self,
        timestamps: &Range<u32>,
        include_subsumed: bool,
    ) -> impl Iterator<Item = (usize, &[Value], &TupleOutput)> {
        self.nodes
            .iter_timestamp_range(timestamps, include_subsumed)
    }

    pub fn rebuild(
        &mut self,
        uf: &mut UnionFind,
        timestamp: u32,
    ) -> Result<(usize, Vec<DeferredMerge>), Error> {
        // Make sure indexes are up to date.
        self.update_indexes(self.nodes.num_offsets());
        if self
            .schema
            .input
            .iter()
            .all(|s| !s.is_eq_sort() && !s.is_eq_container_sort())
            && !self.schema.output.is_eq_sort()
            && !self.schema.output.is_eq_container_sort()
        {
            return Ok((std::mem::take(&mut self.updates), Default::default()));
        }
        let mut deferred_merges = Vec::new();
        let mut scratch = ValueVec::new();
        let n_unions = uf.n_unions();

        if uf.new_ids(|sort| self.sorts.contains(&sort)) > (self.nodes.num_offsets() / 2) {
            // basic heuristic: if we displaced a large number of ids relative
            // to the size of the table, then just rebuild everything.
            for i in 0..self.nodes.num_offsets() {
                self.rebuild_at(i, timestamp, uf, &mut scratch, &mut deferred_merges)?;
            }
        } else {
            let mut to_canon = mem::take(&mut self.scratch);

            to_canon.clear();

            for (i, (ridx, idx)) in self
                .rebuild_indexes
                .iter()
                .zip(self.indexes.iter())
                .enumerate()
            {
                let sort = self.schema.get_by_pos(i).unwrap();
                if !sort.is_eq_container_sort() && !sort.is_eq_sort() {
                    // No need to canonicalize in this case
                    continue;
                }

                // attempt to use the rebuilding index if it exists
                if let Some(ridx) = ridx {
                    debug_assert!(sort.is_eq_container_sort());
                    to_canon.extend(ridx.iter().flat_map(|idx| idx.to_canonicalize(uf)))
                } else {
                    debug_assert!(sort.is_eq_sort());
                    to_canon.extend(idx.to_canonicalize(uf))
                }
            }

            for i in to_canon.iter().copied() {
                self.rebuild_at(i, timestamp, uf, &mut scratch, &mut deferred_merges)?;
            }
            self.scratch = to_canon;
        }
        self.maybe_rehash();
        Ok((
            uf.n_unions() - n_unions + std::mem::take(&mut self.updates),
            deferred_merges,
        ))
    }

    fn rebuild_at(
        &mut self,
        i: usize,
        timestamp: u32,
        uf: &mut UnionFind,
        scratch: &mut ValueVec,
        deferred_merges: &mut Vec<(ValueVec, Value, Value)>,
    ) -> Result<(), Error> {
        let mut result: Result<(), Error> = Ok(());
        let mut modified = false;
        let (args, out) = if let Some(x) = self.nodes.get_index(i, true) {
            x
        } else {
            // Entry is stale
            return result;
        };

        let mut out_val = out.value;
        scratch.clear();
        scratch.extend(args.iter().copied());

        for (val, ty) in scratch.iter_mut().zip(&self.schema.input) {
            modified |= ty.canonicalize(val, uf);
        }

        modified |= self.schema.output.canonicalize(&mut out_val, uf);

        if !modified {
            return result;
        }
        let out_ty = &self.schema.output;
        self.nodes
            .insert_and_merge(scratch, timestamp, out.subsumed, |prev| {
                if let Some(mut prev) = prev {
                    out_ty.canonicalize(&mut prev, uf);
                    let mut appended = false;
                    if self.merge.on_merge.is_some() && prev != out_val {
                        deferred_merges.push((scratch.clone(), prev, out_val));
                        appended = true;
                    }
                    match &self.merge.merge_vals {
                        MergeFn::Union => {
                            debug_assert!(self.schema.output.is_eq_sort());
                            uf.union_values(prev, out_val, self.schema.output.name())
                        }
                        MergeFn::AssertEq => {
                            if prev != out_val {
                                result = Err(Error::MergeError(self.decl.name, prev, out_val));
                            }
                            prev
                        }
                        MergeFn::Expr(_) => {
                            if !appended && prev != out_val {
                                deferred_merges.push((scratch.clone(), prev, out_val));
                            }
                            prev
                        }
                    }
                } else {
                    out_val
                }
            });
        if let Some((inputs, _)) = self.nodes.get_index(i, true) {
            if inputs != &scratch[..] {
                scratch.clear();
                scratch.extend_from_slice(inputs);
                self.nodes.remove(scratch, timestamp);
                scratch.clear();
            }
        }
        result
    }

    pub(crate) fn get_size(&self, range: &Range<u32>) -> usize {
        self.nodes.approximate_range_size(range)
    }

    pub fn is_extractable(&self) -> bool {
        !self.decl.unextractable
    }
}
