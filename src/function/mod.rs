use crate::*;
use index::*;
use smallvec::SmallVec;

mod binary_search;
pub mod index;
mod table;

pub type ValueVec = SmallVec<[Value; 3]>;

#[derive(Clone)]
pub struct Function {
    pub decl: FunctionDecl,
    pub schema: ResolvedSchema,
    pub(crate) is_variable: bool,
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

#[derive(Debug, Clone)]
pub struct TupleOutput {
    pub value: Value,
    pub timestamp: u32,
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

/// A non-Union merge discovered during rebuilding that has to be applied before
/// resuming execution.
pub(crate) type DeferredMerge = (ValueVec, Value, Value);

impl Function {
    pub fn new(egraph: &EGraph, decl: &FunctionDecl, is_variable: bool) -> Result<Self, Error> {
        let mut input = Vec::with_capacity(decl.schema.input.len());
        for s in &decl.schema.input {
            input.push(match egraph.proof_state.type_info.sorts.get(s) {
                Some(sort) => sort.clone(),
                None => return Err(Error::TypeError(TypeError::Unbound(*s))),
            })
        }

        let output = match egraph.proof_state.type_info.sorts.get(&decl.schema.output) {
            Some(sort) => sort.clone(),
            None => return Err(Error::TypeError(TypeError::Unbound(decl.schema.output))),
        };

        let merge_vals = if let Some(merge_expr) = &decl.merge {
            let mut types = IndexMap::<Symbol, ArcSort>::default();
            types.insert("old".into(), output.clone());
            types.insert("new".into(), output.clone());
            let (_, program) = egraph
                .compile_expr(&types, merge_expr, Some(output.clone()))
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
            let mut types = IndexMap::<Symbol, ArcSort>::default();
            types.insert("old".into(), output.clone());
            types.insert("new".into(), output.clone());
            let program = egraph
                .compile_actions(&types, &decl.merge_action)
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
            is_variable,
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
            // TODO figure out merge and default here
        })
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
        if (self.nodes.len() - size) > (size / 8) {
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
                for (slot, _, out) in self.nodes.iter_range(offsets.clone()) {
                    as_mut.add(out.value, slot)
                }
            } else {
                for (slot, inp, _) in self.nodes.iter_range(offsets.clone()) {
                    as_mut.add(inp[col], slot)
                }
            }

            // rebuild_index
            if let Some(rebuild_index) = rebuild_index {
                if col == self.schema.input.len() {
                    for (slot, _, out) in self.nodes.iter_range(offsets.clone()) {
                        self.schema.output.foreach_tracked_values(
                            &out.value,
                            Box::new(|value| rebuild_index.add(value, slot)),
                        )
                    }
                } else {
                    for (slot, inp, _) in self.nodes.iter_range(offsets.clone()) {
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
        self.update_indexes(self.nodes.len());
    }

    pub(crate) fn iter_timestamp_range(
        &self,
        timestamps: &Range<u32>,
    ) -> impl Iterator<Item = (usize, &[Value], &TupleOutput)> {
        self.nodes.iter_timestamp_range(timestamps)
    }

    pub fn rebuild(
        &mut self,
        uf: &mut UnionFind,
        timestamp: u32,
    ) -> Result<(usize, Vec<DeferredMerge>), Error> {
        // Make sure indexes are up to date.
        self.update_indexes(self.nodes.len());
        if self.schema.input.iter().all(|s| !s.is_eq_sort()) && !self.schema.output.is_eq_sort() {
            return Ok((std::mem::take(&mut self.updates), Default::default()));
        }
        let mut deferred_merges = Vec::new();
        let mut scratch = ValueVec::new();
        let n_unions = uf.n_unions();

        if uf.new_ids(|sort| self.sorts.contains(&sort)) > (self.nodes.len() / 2) {
            // basic heuristic: if we displaced a large number of ids relative
            // to the size of the table, then just rebuild everything.
            for i in 0..self.nodes.len() {
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
        let (args, out) = if let Some(x) = self.nodes.get_index(i) {
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
        self.nodes.insert_and_merge(scratch, timestamp, |prev| {
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
                        if !appended {
                            deferred_merges.push((scratch.clone(), prev, out_val));
                        }
                        prev
                    }
                }
            } else {
                out_val
            }
        });
        if let Some((inputs, _)) = self.nodes.get_index(i) {
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
}
