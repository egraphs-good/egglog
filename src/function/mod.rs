use crate::*;
use index::*;
use smallvec::SmallVec;

mod binary_search;
pub mod index;
pub(crate) mod table;

pub type ValueVec = SmallVec<[Value; 3]>;

#[derive(Clone)]
pub struct Function {
    pub decl: FunctionDecl,
    pub schema: ResolvedSchema,
    pub merge: MergeAction,
    pub(crate) nodes: table::Table,
    pub(crate) indexes: Vec<Rc<ColumnIndex>>,
    pub(crate) rebuild_indexes: Vec<Option<CompositeColumnIndex>>,
    index_updated_through: usize,
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

impl Function {
    pub fn new(egraph: &mut EGraph, decl: &FunctionDecl) -> Result<Self, Error> {
        assert!(!egraph.functions.contains_key(&decl.name));
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

        Ok(Function {
            decl: decl.clone(),
            schema: ResolvedSchema { input, output },
            nodes: Default::default(),
            // TODO: build indexes for primitive sorts lazily
            indexes,
            rebuild_indexes,
            index_updated_through: 0,
            merge: MergeAction {
                on_merge,
                merge_vals,
            },
            // TODO figure out merge and default here
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
        self.update_indexes(self.nodes.num_offsets());
    }

    pub(crate) fn iter_timestamp_range(
        &self,
        timestamps: &Range<u32>,
    ) -> impl Iterator<Item = (usize, &[Value], &TupleOutput)> {
        self.nodes.iter_timestamp_range(timestamps)
    }

    pub(crate) fn get_size(&self, range: &Range<u32>) -> usize {
        self.nodes.approximate_range_size(range)
    }
}
