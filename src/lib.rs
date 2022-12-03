pub mod ast;
mod binary_search;
mod extract;
mod gj;
mod index;
pub mod sort;
#[cfg(test)]
mod tests;
mod typecheck;
mod unionfind;
pub mod util;
mod value;

use hashbrown::hash_map::Entry;
use index::ColumnIndex;
use instant::{Duration, Instant};
use lazy_static::lazy_static;
use smallvec::SmallVec;
use sort::*;
use thiserror::Error;

use ast::*;

use std::borrow::Borrow;
use std::fmt::Write;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::iter::once;
use std::mem;
use std::ops::{Deref, Range};
use std::path::PathBuf;
use std::rc::Rc;
use std::{fmt::Debug, sync::Arc};
use typecheck::Program;

type ArcSort = Arc<dyn Sort>;

pub use value::*;

use gj::*;

use unionfind::*;
use util::*;

use crate::binary_search::transform_range;
use crate::typecheck::TypeError;

type ValueVec = SmallVec<[Value; 3]>;

#[derive(Debug, Clone, Eq)]
struct Input {
    data: ValueVec,
    /// The timestamp at which the given input became "stale"
    stale_at: u32,
    /// Counter used to ensure uniqueness
    counter: u32,
}

impl Hash for Input {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.as_slice().hash(state);
        self.stale_at.hash(state);
        self.counter.hash(state);
    }
}

impl PartialEq for Input {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.stale_at == other.stale_at && self.counter == other.counter
    }
}

impl Input {
    fn new(data: ValueVec) -> Input {
        Input {
            data,
            stale_at: u32::MAX,
            counter: 0,
        }
    }

    fn data(&self) -> &[Value] {
        self.data.as_slice()
    }

    fn live(&self) -> bool {
        self.stale_at == u32::MAX
    }
}

/// A custom type used to look elements up in the map. InputRefs can be created
/// from a slice of values without copying data, and they can serve as keys in
/// the map that only find live elements.
#[derive(Eq)]
#[repr(transparent)]
struct InputRef(pub [Value]);

impl InputRef {
    fn from_slice(vals: &[Value]) -> &InputRef {
        // SAFETY: InputRef is repr(transparent)
        unsafe { std::mem::transmute::<&[Value], &InputRef>(vals) }
    }
}

impl PartialEq for InputRef {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Hash for InputRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        // Only look for live values
        u32::MAX.hash(state);
        0.hash(state)
    }
}

impl Borrow<InputRef> for Input {
    fn borrow(&self) -> &InputRef {
        // Lookups via InputRef should never be able to "find" stale data.
        lazy_static! {
            pub static ref BOGUS: Vec<Value> = vec![Value::fake()];
        }
        // Warning: it's unclear if this is safe. We break the Borrow rules
        // by doing this! We are probably better off building our own variant of
        // IndexMap that allows us to insert "holes" or "tombstones." But we
        // should only do that once we understand the full story on rollback
        // support.
        if self.live() {
            InputRef::from_slice(self.data())
        } else if self.data().is_empty() {
            InputRef::from_slice(BOGUS.as_slice())
        } else {
            InputRef::from_slice(&[])
        }
    }
}

#[derive(Clone)]
pub struct Function {
    decl: FunctionDecl,
    schema: ResolvedSchema,
    merge: MergeFn,
    nodes: IndexMap<Input, TupleOutput>,
    indexes: Vec<Rc<ColumnIndex>>,
    index_updated_through: usize,
    updates: usize,
    counter: usize,
    n_stale: usize,
    scratch: IndexSet<usize>,
}

#[derive(Clone)]
enum MergeFn {
    AssertEq,
    Union,
    // the rc is make sure it's cheaply clonable, since calling the merge fn
    // requires a clone
    Expr(Rc<Program>),
}

#[derive(Debug, Clone)]
struct TupleOutput {
    value: Value,
    timestamp: u32,
}

#[derive(Clone, Debug)]
struct ResolvedSchema {
    input: Vec<ArcSort>,
    output: ArcSort,
}

impl Function {
    pub fn insert(&mut self, inputs: ValueVec, value: Value, timestamp: u32) -> Option<Value> {
        self.insert_internal(inputs, value, timestamp, true)
    }
    pub fn insert_internal(
        &mut self,
        inputs: ValueVec,
        value: Value,
        timestamp: u32,
        // Clean out all stale entries if they account for a sufficiently large
        // portion of the table after this entry is inserted.
        maybe_rehash: bool,
    ) -> Option<Value> {
        let (index, old) = if let Some((index, _, old)) = self
            .nodes
            .get_full_mut(InputRef::from_slice(inputs.as_slice()))
        {
            if old.value == value {
                return Some(value);
            } else {
                self.updates += 1;
                (index, old.value)
            }
        } else {
            self.nodes
                .insert(Input::new(inputs), TupleOutput { value, timestamp });
            self.updates += 1;
            return None;
        };
        self.set_stale(index, timestamp);
        self.nodes
            .insert(Input::new(inputs), TupleOutput { value, timestamp });
        if maybe_rehash {
            self.maybe_rehash();
        }
        Some(old)
    }

    /// Return a column index that contains (a superset of) the offsets for the
    /// given column. This method can return nothing if the indexes available
    /// contain too many irrelevant offsets.
    pub(crate) fn column_index(
        &self,
        col: usize,
        timestamps: &Range<u32>,
    ) -> Option<Rc<ColumnIndex>> {
        let range = transform_range(&self.nodes, |out| &out.timestamp, timestamps);
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

    fn set_stale(&mut self, i: usize, ts: u32) {
        debug_assert!(i < self.nodes.len());
        let (mut inp, out) = self.nodes.swap_remove_index(i).unwrap();
        debug_assert!(inp.live());
        inp.stale_at = ts;
        inp.counter = self.counter as u32;
        self.counter += 1;
        self.nodes.insert(inp, out);
        self.n_stale += 1;
        self.nodes.swap_indices(i, self.nodes.len() - 1);
    }

    fn build_indexes(&mut self, indexes: Range<usize>) {
        for (col, index) in self.indexes.iter_mut().enumerate() {
            let as_mut = Rc::make_mut(index);
            if col == self.schema.input.len() {
                for slot in indexes.clone() {
                    let (inp, out) = self.nodes.get_index(slot).unwrap();
                    if !inp.live() {
                        continue;
                    }
                    as_mut.add(out.value, slot)
                }
            } else {
                for slot in indexes.clone() {
                    let (inp, _) = self.nodes.get_index(slot).unwrap();
                    if !inp.live() {
                        continue;
                    }
                    as_mut.add(inp.data()[col], slot)
                }
            }
        }
    }

    fn update_indexes(&mut self, through: usize) {
        self.build_indexes(self.index_updated_through..through);
        self.index_updated_through = self.index_updated_through.max(through);
    }

    fn maybe_rehash(&mut self) {
        // Note for future: this is very much a necessary step. We see major
        // slowdowns in some tests without this code in place, but it also
        // removes old versions of tuples. The slowdown happens because certain
        // operations (rebuilding, index construction) still need to do O(n)
        // scans, where 'n' is the number of tuples, live or stale.
        if self.n_stale <= (self.nodes.len() / 2) {
            return;
        }
        for index in &mut self.indexes {
            // Everything works if we don't have a unique copy of the indexes,
            // but we ought to be able to avoid this copy.
            Rc::make_mut(index).clear();
        }
        self.nodes.retain(|k, _| k.live());
        self.n_stale = 0;
        self.counter = 0;
        self.index_updated_through = 0;
        self.update_indexes(self.nodes.len());
    }

    pub(crate) fn iter_timestamp_range(
        &self,
        timestamps: &Range<u32>,
    ) -> impl Iterator<Item = (usize, &Input, &TupleOutput)> {
        let indexes = transform_range(&self.nodes, |v| &v.timestamp, timestamps);
        indexes.filter_map(|i| {
            let (k, v) = self.nodes.get_index(i).unwrap();
            if k.live() {
                Some((i, k, v))
            } else {
                None
            }
        })
    }

    pub fn rebuild(&mut self, uf: &mut UnionFind, timestamp: u32) -> usize {
        // Make sure indexes are up to date.
        self.update_indexes(self.nodes.len());
        if self.schema.input.iter().all(|s| !s.is_eq_sort()) && !self.schema.output.is_eq_sort() {
            return std::mem::take(&mut self.updates);
        }
        let mut to_canon = mem::take(&mut self.scratch);
        to_canon.clear();
        to_canon.extend(self.indexes.iter().flat_map(|x| x.to_canonicalize(uf)));

        let n_unions = uf.n_unions();
        let mut scratch = ValueVec::new();
        for i in to_canon.iter().copied() {
            let mut modified = false;
            let (args, out) = self.nodes.get_index(i).unwrap();
            if !args.live() {
                continue;
            }
            let mut out_val = out.value;
            scratch.clear();
            scratch.extend(args.data.iter().copied());
            for (val, ty) in scratch
                .iter_mut()
                .zip(&self.schema.input)
                .chain(once((&mut out_val, &self.schema.output)))
            {
                if !ty.is_eq_sort() {
                    continue;
                }
                let new = uf.find_value(*val);
                if &new != val {
                    *val = new;
                    modified = true;
                }
            }
            if !modified {
                continue;
            }
            self.set_stale(i, timestamp);
            if let Some(prev) = self.insert_internal(scratch.clone(), out_val, timestamp, false) {
                // We need to merge these ids
                // TODO: call the merge fn
                if !self.schema.output.is_eq_sort() {
                    continue;
                }
                let next = uf.union_values(prev, out_val, self.schema.output.name());
                if next == out_val {
                    // No change and no need to update.
                    continue;
                }
                self.insert_internal(scratch.clone(), next, timestamp, false);
            }
        }
        self.maybe_rehash();
        self.scratch = to_canon;
        uf.n_unions() - n_unions + std::mem::take(&mut self.updates)
    }

    pub(crate) fn get_size(&self, range: &Range<u32>) -> usize {
        let indexes = transform_range(&self.nodes, |v| &v.timestamp, range);
        indexes.end - indexes.start
    }
}

pub type Subst = IndexMap<Symbol, Value>;

pub trait PrimitiveLike {
    fn name(&self) -> Symbol;
    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort>;
    fn apply(&self, values: &[Value]) -> Option<Value>;
}

#[derive(Clone)]
pub struct Primitive(Arc<dyn PrimitiveLike>);

impl Deref for Primitive {
    type Target = dyn PrimitiveLike;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl Hash for Primitive {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}

impl Eq for Primitive {}
impl PartialEq for Primitive {
    fn eq(&self, other: &Self) -> bool {
        // this is a bit of a hack, but clippy says we don't want to compare the
        // vtables, just the data pointers
        std::ptr::eq(
            Arc::as_ptr(&self.0) as *const u8,
            Arc::as_ptr(&other.0) as *const u8,
        )
    }
}

impl Debug for Primitive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Prim({})", self.0.name())
    }
}

impl<T: PrimitiveLike + 'static> From<T> for Primitive {
    fn from(p: T) -> Self {
        Self(Arc::new(p))
    }
}

pub struct SimplePrimitive {
    name: Symbol,
    input: Vec<ArcSort>,
    output: ArcSort,
    f: fn(&[Value]) -> Option<Value>,
}

impl PrimitiveLike for SimplePrimitive {
    fn name(&self) -> Symbol {
        self.name
    }
    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        if self.input.len() != types.len() {
            return None;
        }
        // TODO can we use a better notion of equality than just names?
        self.input
            .iter()
            .zip(types)
            .all(|(a, b)| a.name() == b.name())
            .then(|| self.output.clone())
    }
    fn apply(&self, values: &[Value]) -> Option<Value> {
        (self.f)(values)
    }
}

#[derive(Clone)]
pub struct EGraph {
    egraphs: Vec<Self>,
    unionfind: UnionFind,
    presorts: HashMap<Symbol, PreSort>,
    sorts: HashMap<Symbol, Arc<dyn Sort>>,
    primitives: HashMap<Symbol, Vec<Primitive>>,
    functions: HashMap<Symbol, Function>,
    rules: HashMap<Symbol, Rule>,
    saturated: bool,
    timestamp: u32,
    pub match_limit: usize,
    pub node_limit: usize,
    pub fact_directory: Option<PathBuf>,
    pub seminaive: bool,
}

#[derive(Clone, Debug)]
struct Rule {
    query: CompiledQuery,
    program: Program,
    matches: usize,
    times_banned: usize,
    banned_until: usize,
    todo_timestamp: u32,
    search_time: Duration,
    apply_time: Duration,
}

impl Default for EGraph {
    fn default() -> Self {
        let mut egraph = Self {
            egraphs: vec![],
            unionfind: Default::default(),
            sorts: Default::default(),
            functions: Default::default(),
            rules: Default::default(),
            primitives: Default::default(),
            presorts: Default::default(),
            match_limit: 10_000_000,
            node_limit: 100_000_000,
            timestamp: 0,
            saturated: false,
            fact_directory: None,
            seminaive: true,
        };
        egraph.add_sort(UnitSort::new("Unit".into()));
        egraph.add_sort(StringSort::new("String".into()));
        egraph.add_sort(I64Sort::new("i64".into()));
        egraph.add_sort(RationalSort::new("Rational".into()));
        egraph.presorts.insert("Map".into(), MapSort::make_sort);
        egraph
    }
}

#[derive(Debug, Error)]
#[error("Not found: {0}")]
pub struct NotFoundError(Expr);

impl EGraph {
    pub fn add_sort<S: Sort + 'static>(&mut self, sort: S) {
        self.add_arcsort(Arc::new(sort)).unwrap()
    }

    pub fn push(&mut self) {
        self.egraphs.push(self.clone());
    }

    pub fn pop(&mut self) -> Result<(), Error> {
        match self.egraphs.pop() {
            Some(e) => {
                *self = e;
                Ok(())
            }
            None => Err(Error::Pop),
        }
    }

    pub fn add_arcsort(&mut self, sort: ArcSort) -> Result<(), Error> {
        let name = sort.name();
        match self.sorts.entry(name) {
            Entry::Occupied(_) => Err(Error::SortAlreadyBound(name)),
            Entry::Vacant(e) => {
                e.insert(sort.clone());
                sort.register_primitives(self);
                Ok(())
            }
        }
    }

    pub fn get_sort<S: Sort + Send + Sync>(&self) -> Arc<S> {
        for sort in self.sorts.values() {
            let sort = sort.clone().as_arc_any();
            if let Ok(sort) = Arc::downcast(sort) {
                return sort;
            }
        }
        // TODO handle if multiple match?
        // could handle by type id??
        panic!("Failed to lookup sort: {}", std::any::type_name::<S>());
    }

    pub fn add_primitive(&mut self, prim: impl Into<Primitive>) {
        let prim = prim.into();
        self.primitives.entry(prim.name()).or_default().push(prim);
    }

    pub fn union(&mut self, id1: Id, id2: Id, sort: Symbol) -> Id {
        self.unionfind.union(id1, id2, sort)
    }

    #[track_caller]
    fn debug_assert_invariants(&self) {
        #[cfg(debug_assertions)]
        for (name, function) in self.functions.iter() {
            let timestamps = Vec::from_iter(function.nodes.iter().map(|(_, y)| y.timestamp));
            assert!(
                timestamps.windows(2).all(|x| x[0] <= x[1]),
                "functions must be sorted by timestamp"
            );
            for (i, (inputs, output)) in function.nodes.iter().enumerate() {
                if !inputs.live() {
                    continue;
                }
                for input in inputs.data() {
                    assert_eq!(
                        input,
                        &self.bad_find_value(*input),
                        "[{i}] {name}({inputs:?}) = {output:?}\n{:?}",
                        function.schema,
                    )
                }
                assert_eq!(
                    output.value,
                    self.bad_find_value(output.value),
                    "[{i}] {name}({inputs:?}) = {output:?}\n{:?}",
                    function.schema,
                )
            }
        }
    }

    pub fn check_fact(&mut self, fact: &Fact) -> Result<(), Error> {
        match fact {
            Fact::Eq(exprs) => {
                assert!(exprs.len() > 1);
                let values: Vec<(ArcSort, Value)> = exprs
                    .iter()
                    .map(|e| self.eval_expr(e, None, false))
                    .collect::<Result<_, _>>()?;

                let (_t0, v0) = &values[0];
                for (_t, v) in &values[1..] {
                    if v0 != v {
                        log::error!("Check failed");
                        // the check failed, so print out some useful info
                        self.rebuild();
                        for (_t, value) in &values {
                            if let Some((_tag, id)) = self.value_to_id(*value) {
                                let best = self.extract(*value).1;
                                log::error!("{}: {}", id, best);
                            }
                        }
                        return Err(Error::CheckError(values[0].1, *v));
                    }
                }
            }
            Fact::Fact(expr) => match expr {
                Expr::Lit(_) => panic!("can't check a literal"),
                Expr::Var(_) => panic!("can't check a var"),
                Expr::Call(_, _) => {
                    // println!("Checking fact: {}", expr);
                    let unit = self.get_sort::<UnitSort>();
                    self.eval_expr(expr, Some(unit), true)?;
                }
            },
        }
        Ok(())
    }

    pub fn find(&self, id: Id) -> Id {
        self.unionfind.find(id)
    }

    pub fn rebuild(&mut self) -> usize {
        self.unionfind.clear_recent_ids();
        let mut updates = 0;
        loop {
            let new = self.rebuild_one();
            log::debug!("{new} rebuilds?");
            self.unionfind.clear_recent_ids();
            updates += new;
            if new == 0 {
                break;
            }
        }
        self.debug_assert_invariants();
        updates
    }

    fn rebuild_one(&mut self) -> usize {
        let mut new_unions = 0;
        for function in self.functions.values_mut() {
            new_unions += function.rebuild(&mut self.unionfind, self.timestamp);
        }
        new_unions
    }

    pub fn declare_sort(
        &mut self,
        name: impl Into<Symbol>,
        presort_and_args: Option<(Symbol, &[Expr])>,
    ) -> Result<(), Error> {
        let name = name.into();
        let sort = match presort_and_args {
            Some((presort, args)) => {
                let mksort = self
                    .presorts
                    .get(&presort)
                    .ok_or(Error::PresortNotFound(presort))?;
                mksort(self, name, args)?
            }
            None => Arc::new(EqSort { name }),
        };
        self.add_arcsort(sort)
    }

    pub fn declare_function(&mut self, decl: &FunctionDecl) -> Result<(), Error> {
        let mut input = Vec::with_capacity(decl.schema.input.len());
        for s in &decl.schema.input {
            input.push(match self.sorts.get(s) {
                Some(sort) => sort.clone(),
                None => return Err(Error::TypeError(TypeError::Unbound(*s))),
            })
        }

        let output = match self.sorts.get(&decl.schema.output) {
            Some(sort) => sort.clone(),
            None => return Err(Error::TypeError(TypeError::Unbound(decl.schema.output))),
        };

        let merge = if let Some(merge_expr) = &decl.merge {
            let mut types = IndexMap::<Symbol, ArcSort>::default();
            types.insert("old".into(), output.clone());
            types.insert("new".into(), output.clone());
            let (_, program) = self
                .compile_expr(&types, merge_expr, Some(output.clone()))
                .map_err(Error::TypeErrors)?;
            MergeFn::Expr(Rc::new(program))
        } else if output.is_eq_sort() {
            MergeFn::Union
        } else {
            MergeFn::AssertEq
        };

        let indexes = Vec::from_iter(
            input
                .iter()
                .chain(once(&output))
                .map(|x| Rc::new(ColumnIndex::new(x.name()))),
        );

        let function = Function {
            decl: decl.clone(),
            schema: ResolvedSchema { input, output },
            nodes: Default::default(),
            scratch: Default::default(),
            // TODO: build indexes for primitive sorts lazily
            indexes,
            index_updated_through: 0,
            updates: 0,
            counter: 0,
            n_stale: 0,
            merge,
            // TODO figure out merge and default here
        };

        let old = self.functions.insert(decl.name, function);
        if old.is_some() {
            return Err(TypeError::FunctionAlreadyBound(decl.name).into());
        }

        Ok(())
    }

    pub fn declare_constructor(
        &mut self,
        variant: Variant,
        sort: impl Into<Symbol>,
    ) -> Result<(), Error> {
        let name = variant.name;
        let sort = sort.into();
        self.declare_function(&FunctionDecl {
            name,
            schema: Schema {
                input: variant.types,
                output: sort,
            },
            merge: None,
            default: None,
            cost: variant.cost,
        })?;
        // if let Some(ctors) = self.sorts.get_mut(&sort) {
        //     ctors.push(name);
        // }
        Ok(())
    }

    pub fn eval_lit(&self, lit: &Literal) -> Value {
        match lit {
            Literal::Int(i) => i.store(&*self.get_sort()).unwrap(),
            Literal::String(s) => s.store(&*self.get_sort()).unwrap(),
            Literal::Unit => ().store(&*self.get_sort()).unwrap(),
        }
    }

    pub fn print_function(&mut self, sym: Symbol, n: usize) -> Result<String, Error> {
        let f = self.functions.get(&sym).ok_or(TypeError::Unbound(sym))?;
        let schema = f.schema.clone();
        let nodes = f
            .nodes
            .iter()
            .take(n)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>();

        let out_is_unit = f.schema.output.name() == "Unit".into();

        let mut buf = String::new();
        let s = &mut buf;
        for (ins, out) in nodes {
            if !ins.live() {
                continue;
            }
            write!(s, "({}", sym).unwrap();
            for (a, t) in ins.data().iter().zip(&schema.input) {
                s.push(' ');
                let e = if t.is_eq_sort() {
                    self.extract(*a).1
                } else {
                    t.make_expr(*a)
                };
                write!(s, "{}", e).unwrap();
            }

            if out_is_unit {
                s.push(')');
            } else {
                let e = if schema.output.is_eq_sort() {
                    self.extract(out.value).1
                } else {
                    schema.output.make_expr(out.value)
                };
                write!(s, ") -> {}", e).unwrap();
            }
            s.push('\n');
            // write!(s, "{}(", self.decl.name)?;
            // for (i, arg) in args.iter().enumerate() {
            //     if i > 0 {
            //         write!(s, ", ")?;
            //     }
            //     write!(s, "{}", arg)?;
            // }
            // write!(s, ") = {}", value)?;
            // println!("{}", s);
        }

        Ok(buf)
    }

    pub fn print_size(&self, sym: Symbol) -> Result<String, Error> {
        let f = self.functions.get(&sym).ok_or(TypeError::Unbound(sym))?;
        Ok(format!("Function {} has size {}", sym, f.nodes.len()))
    }

    pub fn run_rules(&mut self, limit: usize) -> [Duration; 3] {
        let mut search_time = Duration::default();
        let mut apply_time = Duration::default();

        // we might have to do a rebuild before starting,
        // because the use can manually do stuff
        let initial_rebuild_start = Instant::now();
        self.rebuild();
        let mut rebuild_time = initial_rebuild_start.elapsed();

        for i in 0..limit {
            self.saturated = true;
            let [st, at] = self.step_rules(i);
            search_time += st;
            apply_time += at;

            let rebuild_start = Instant::now();
            let updates = self.rebuild();
            log::info!("database size: {}", self.num_tuples());
            log::info!("Made {updates} updates (iteration {i})");
            rebuild_time += rebuild_start.elapsed();
            self.timestamp += 1;
            if self.saturated {
                log::info!("Breaking early at iteration {}!", i);
                break;
            }
            if self.num_tuples() > self.node_limit {
                log::warn!(
                    "Node limit reached at iteration {}, {} nodes. Stopping!",
                    i,
                    self.num_tuples()
                );
                break;
            }
        }

        // Report the worst offenders
        log::debug!("Slowest rules:\n{}", {
            let mut msg = String::new();
            let mut vec = self.rules.iter().collect::<Vec<_>>();
            vec.sort_by_key(|(_, r)| r.search_time + r.apply_time);
            for (name, rule) in vec.iter().rev().take(5) {
                write!(
                    msg,
                    "{name}\n  Search: {}\n  Apply: {}\n",
                    rule.search_time.as_secs_f64(),
                    rule.apply_time.as_secs_f64()
                )
                .unwrap();
            }
            msg
        });

        // // TODO detect functions
        // for (name, r) in &self.functions {
        //     log::debug!("{name}:");
        //     for (args, val) in &r.nodes {
        //         log::debug!("  {args:?} = {val:?}");
        //     }
        // }
        [search_time, apply_time, rebuild_time]
    }

    fn step_rules(&mut self, iteration: usize) -> [Duration; 2] {
        // fn make_subst(rule: &Rule, values: &[Value]) -> Subst {
        //     let get_val = |t: &AtomTerm| match t {
        //         AtomTerm::Var(sym) => {
        //             let i = rule
        //                 .query
        //                 .vars
        //                 .get_index_of(sym)
        //                 .unwrap_or_else(|| panic!("Couldn't find variable '{sym}'"));
        //             values[i]
        //         }
        //         AtomTerm::Value(val) => *val,
        //     };

        //     todo!()
        //     // rule.bindings
        //     //     .iter()
        //     //     .map(|(k, t)| (*k, get_val(t)))
        //     //     .collect()
        // }

        let ban_length = 5;

        let mut rules = std::mem::take(&mut self.rules);
        let search_start = Instant::now();
        let mut searched = vec![];
        for (&name, rule) in rules.iter_mut() {
            let mut all_values = vec![];
            if rule.banned_until <= iteration {
                let mut fuel = self.match_limit << rule.times_banned;
                let rule_search_start = Instant::now();
                self.run_query(&rule.query, rule.todo_timestamp, |values| {
                    assert_eq!(values.len(), rule.query.vars.len());
                    all_values.extend_from_slice(values);
                    if fuel > 0 {
                        fuel -= 1;
                        Ok(())
                    } else {
                        Err(())
                    }
                });
                let rule_search_time = rule_search_start.elapsed();
                log::trace!(
                    "Searched for {name} in {} ({} results)",
                    rule_search_time.as_secs_f64(),
                    all_values.len()
                );
                rule.search_time += rule_search_time;
                searched.push((name, all_values));
            } else {
                self.saturated = false;
            }
        }
        let search_elapsed = search_start.elapsed();

        let apply_start = Instant::now();
        'outer: for (name, all_values) in searched {
            let rule = rules.get_mut(&name).unwrap();
            let n = rule.query.vars.len();

            // backoff logic
            let len = all_values.len() / n;
            let threshold = self.match_limit << rule.times_banned;
            if len > threshold {
                let ban_length = ban_length << rule.times_banned;
                rule.times_banned += 1;
                rule.banned_until = iteration + ban_length;
                log::info!("Banning rule {name} for {ban_length} iterations, matched {len} > {threshold} times");
                self.saturated = false;
                continue;
            }

            rule.todo_timestamp = self.timestamp;
            let rule_apply_start = Instant::now();

            let stack = &mut vec![];
            for values in all_values.chunks(n) {
                rule.matches += 1;
                if rule.matches > 10_000_000 {
                    log::warn!("Rule {} has matched {} times, bailing!", name, rule.matches);
                    break 'outer;
                }
                // we can ignore results here
                stack.clear();
                let _ = self.run_actions(stack, values, &rule.program, true);
            }

            rule.apply_time += rule_apply_start.elapsed();
        }
        self.rules = rules;
        let apply_elapsed = apply_start.elapsed();
        [search_elapsed, apply_elapsed]
    }

    fn add_rule_with_name(&mut self, name: String, rule: ast::Rule) -> Result<Symbol, Error> {
        let name = Symbol::from(name);
        let mut ctx = typecheck::Context::new(self);
        let query0 = ctx.typecheck_query(&rule.body).map_err(Error::TypeErrors)?;
        let query = self.compile_gj_query(query0, &ctx.types);
        let program = self
            .compile_actions(&ctx.types, &rule.head)
            .map_err(Error::TypeErrors)?;
        // println!(
        //     "Compiled rule {rule:?}\n{subst:?}to {program:#?}",
        //     subst = &ctx.types
        // );
        let compiled_rule = Rule {
            query,
            matches: 0,
            times_banned: 0,
            banned_until: 0,
            todo_timestamp: 0,
            program,
            search_time: Duration::default(),
            apply_time: Duration::default(),
        };
        match self.rules.entry(name) {
            Entry::Occupied(_) => panic!("Rule '{name}' was already present"),
            Entry::Vacant(e) => e.insert(compiled_rule),
        };
        Ok(name)
    }

    pub fn add_rule(&mut self, rule: ast::Rule) -> Result<Symbol, Error> {
        let name = format!("{}", rule);
        self.add_rule_with_name(name, rule)
    }

    pub fn clear_rules(&mut self) {
        self.rules = Default::default();
    }

    pub fn add_rewrite(&mut self, rewrite: ast::Rewrite) -> Result<Symbol, Error> {
        let mut name = format!("{} -> {}", rewrite.lhs, rewrite.rhs);
        if !rewrite.conditions.is_empty() {
            write!(name, " if {}", ListDisplay(&rewrite.conditions, ", ")).unwrap();
        }
        let var = Symbol::from("__rewrite_var");
        let rule = ast::Rule {
            body: [Fact::Eq(vec![Expr::Var(var), rewrite.lhs])]
                .into_iter()
                .chain(rewrite.conditions)
                .collect(),
            head: vec![Action::Union(Expr::Var(var), rewrite.rhs)],
        };
        self.add_rule_with_name(name, rule)
    }

    pub fn eval_actions(&mut self, actions: &[Action]) -> Result<(), Error> {
        let types = Default::default();
        let program = self
            .compile_actions(&types, actions)
            .map_err(Error::TypeErrors)?;
        let mut stack = vec![];
        self.run_actions(&mut stack, &[], &program, true)?;
        Ok(())
    }

    fn eval_expr(
        &mut self,
        expr: &Expr,
        expected_type: Option<ArcSort>,
        make_defaults: bool,
    ) -> Result<(ArcSort, Value), Error> {
        let types = Default::default();
        let (t, program) = self
            .compile_expr(&types, expr, expected_type)
            .map_err(Error::TypeErrors)?;
        let mut stack = vec![];
        self.run_actions(&mut stack, &[], &program, make_defaults)?;
        assert_eq!(stack.len(), 1);
        Ok((t, stack.pop().unwrap()))
    }

    fn run_command(&mut self, command: Command, should_run: bool) -> Result<String, Error> {
        Ok(match command {
            Command::Datatype { name, variants } => {
                self.declare_sort(name, None)?;
                for variant in variants {
                    self.declare_constructor(variant, name)?;
                }
                format!("Declared datatype {name}.")
            }
            Command::Sort(name, presort, args) => {
                self.declare_sort(name, Some((presort, &args)))?;
                format!(
                    "Declared sort {name} = ({presort} {})",
                    ListDisplay(&args, " ")
                )
            }
            Command::Function(fdecl) => {
                self.declare_function(&fdecl)?;
                format!("Declared function {}.", fdecl.name)
            }
            Command::Rule(rule) => {
                let name = self.add_rule(rule)?;
                format!("Declared rule {name}.")
            }
            Command::Rewrite(rewrite) => {
                let name = self.add_rewrite(rewrite)?;
                format!("Declared rw {name}.")
            }
            Command::BiRewrite(rewrite) => {
                let rw2 = rewrite.clone();
                let _name = self.add_rewrite(rewrite)?;
                let rewrite = Rewrite {
                    lhs: rw2.rhs,
                    rhs: rw2.lhs,
                    conditions: rw2.conditions,
                };
                let name = self.add_rewrite(rewrite)?;
                format!("Declared bi-rw {name}.")
            }
            Command::Run(limit) => {
                if should_run {
                    let [st, at, rt] = self.run_rules(limit);
                    let st = st.as_secs_f64();
                    let at = at.as_secs_f64();
                    let rt = rt.as_secs_f64();
                    let total = st + at + rt;
                    let size = self.num_tuples();
                    format!(
                        "Ran {limit} in {total:10.6}s.\n\
                        Search:  ({:.02}) {st:10.6}s\n\
                        Apply:   ({:.02}) {at:10.6}s\n\
                        Rebuild: ({:.02}) {rt:10.6}s\n\
                        Database size: {size}",
                        st / total,
                        at / total,
                        rt / total,
                    )
                } else {
                    log::info!("Skipping running!");
                    format!("Skipped run {limit}.")
                }
            }
            Command::Extract { e, variants } => {
                if should_run {
                    // TODO typecheck
                    let (cost, expr, exprs) = self.extract_expr(e, variants)?;
                    let mut msg = format!("Extracted with cost {cost}: {expr}");
                    if variants > 0 {
                        let line = "\n    ";
                        let v_exprs = ListDisplay(&exprs, line);
                        write!(msg, "\nVariants of {expr}:{line}{v_exprs}").unwrap();
                    }
                    msg
                } else {
                    "Skipping extraction.".into()
                }
            }
            Command::Check(fact) => {
                if should_run {
                    self.check_fact(&fact)?;
                    "Checked.".into()
                } else {
                    "Skipping check.".into()
                }
            }
            Command::Action(action) => {
                if should_run {
                    self.eval_actions(std::slice::from_ref(&action))?;
                    format!("Run {action}.")
                } else {
                    format!("Skipping running {action}.")
                }
            }
            Command::Define { name, expr, cost } => {
                if should_run {
                    let sort = self.define(name, expr, cost)?;
                    format!("Defined {name}: {sort:?}")
                } else {
                    format!("Skipping define {name}")
                }
            }
            Command::ClearRules => {
                self.clear_rules();
                "Clearing rules.".into()
            }
            Command::Query(_q) => {
                // let qsexp = sexp::Sexp::List(
                //     q.iter()
                //         .map(|fact| sexp::parse(&fact.to_string()).unwrap())
                //         .collect(),
                // );
                // let qcomp = self
                //     .compile_query(q)
                //     .unwrap_or_else(|_| panic!("Could not compile query"));
                // let mut res = vec![];
                // self.query(&qcomp, |v| {
                //     res.push(sexp::Sexp::List(
                //         v.iter()
                //             .map(|val| sexp::Sexp::Atom(sexp::Atom::S(format!("{}", val))))
                //             .collect(),
                //     ));
                // });
                // format!(
                //     "Query: {}\n  Bindings: {:?}\n  Results: {}",
                //     qsexp,
                //     qcomp,
                //     sexp::Sexp::List(res)
                // )
                todo!()
            }
            Command::Clear => {
                self.clear();
                "Cleared.".into()
            }
            Command::Push(n) => {
                (0..n).for_each(|_| self.push());
                format!("Pushed {n} levels.")
            }
            Command::Pop(n) => {
                for _ in 0..n {
                    self.pop()?;
                }
                format!("Popped {n} levels.")
            }
            Command::Print(f, n) => {
                let msg = self.print_function(f, n)?;
                println!("{}", msg);
                msg
            }
            Command::PrintSize(f) => {
                let msg = self.print_size(f)?;
                println!("{}", msg);
                msg
            }
            Command::Input { name, file } => {
                let func = self.functions.get_mut(&name).unwrap();
                let is_unit = func.schema.output.name().as_str() == "Unit";

                let mut filename = self.fact_directory.clone().unwrap_or_default();
                filename.push(file.as_str());

                // check that the function uses supported types
                for t in &func.schema.input {
                    match t.name().as_str() {
                        "i64" | "String" => {}
                        s => panic!("Unsupported type {} for input", s),
                    }
                }
                match func.schema.output.name().as_str() {
                    "i64" | "String" | "Unit" => {}
                    s => panic!("Unsupported type {} for input", s),
                }

                log::info!("Opening file '{:?}'...", filename);
                let mut f = File::open(filename).unwrap();
                let mut contents = String::new();
                f.read_to_string(&mut contents).unwrap();

                let mut actions: Vec<Action> = vec![];
                let mut str_buf: Vec<&str> = vec![];
                for line in contents.lines() {
                    str_buf.clear();
                    str_buf.extend(line.split('\t').map(|s| s.trim()));
                    if str_buf.is_empty() {
                        continue;
                    }

                    let parse = |s: &str| -> Expr {
                        if let Ok(i) = s.parse() {
                            Expr::Lit(Literal::Int(i))
                        } else {
                            Expr::Lit(Literal::String(s.into()))
                        }
                    };

                    let mut exprs: Vec<Expr> = str_buf.iter().map(|&s| parse(s)).collect();

                    actions.push(if is_unit {
                        Action::Expr(Expr::Call(name, exprs))
                    } else {
                        let out = exprs.pop().unwrap();
                        Action::Set(name, exprs, out)
                    });
                }
                self.eval_actions(&actions)?;
                format!("Read {} facts into {name} from '{file}'.", actions.len())
            }
        })
    }

    pub fn clear(&mut self) {
        for f in self.functions.values_mut() {
            f.nodes.clear();
        }
    }

    // Extract an expression from the current state, returning the cost, the extracted expression and some number
    // of other variants, if variants is not zero.
    pub fn extract_expr(
        &mut self,
        e: Expr,
        variants: usize,
    ) -> Result<(usize, Expr, Vec<Expr>), Error> {
        self.rebuild();
        let (_t, value) = self.eval_expr(&e, None, true)?;
        let (cost, expr) = self.extract(value);
        let exprs = if variants > 0 {
            self.extract_variants(value, variants)
        } else {
            vec![]
        };
        Ok((cost, expr, exprs))
    }

    pub fn define(
        &mut self,
        name: Symbol,
        expr: Expr,
        cost: Option<usize>,
    ) -> Result<ArcSort, Error> {
        let (sort, value) = self.eval_expr(&expr, None, true)?;
        self.declare_function(&FunctionDecl {
            name,
            schema: Schema {
                input: vec![],
                output: value.tag,
            },
            default: None,
            merge: None,
            cost,
        })?;
        let f = self.functions.get_mut(&name).unwrap();
        f.insert(ValueVec::default(), value, self.timestamp);
        Ok(sort)
    }

    fn run_program(&mut self, program: Vec<Command>) -> Result<Vec<String>, Error> {
        let mut msgs = vec![];
        let should_run = true;

        for command in program {
            let msg = self.run_command(command, should_run)?;
            log::info!("{}", msg);
            msgs.push(msg);
        }

        Ok(msgs)
    }

    // this is bad because we shouldn't inspect values like this, we should use type information
    #[cfg(debug_assertions)]
    fn bad_find_value(&self, value: Value) -> Value {
        if let Some((tag, id)) = self.value_to_id(value) {
            Value::from_id(tag, self.find(id))
        } else {
            value
        }
    }

    pub fn parse_and_run_program(&mut self, input: &str) -> Result<Vec<String>, Error> {
        let parser = ast::parse::ProgramParser::new();
        let program = parser
            .parse(input)
            .map_err(|e| e.map_token(|tok| tok.to_string()))?;
        self.run_program(program)
    }

    pub fn num_tuples(&self) -> usize {
        self.functions.values().map(|f| f.nodes.len()).sum()
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    ParseError(#[from] lalrpop_util::ParseError<usize, String, String>),
    #[error(transparent)]
    NotFoundError(#[from] NotFoundError),
    #[error(transparent)]
    TypeError(#[from] TypeError),
    #[error("Errors:\n{}", ListDisplay(.0, "\n"))]
    TypeErrors(Vec<TypeError>),
    #[error("Check failed: {0:?} != {1:?}")]
    CheckError(Value, Value),
    #[error("Sort {0} already declared.")]
    SortAlreadyBound(Symbol),
    #[error("Presort {0} not found.")]
    PresortNotFound(Symbol),
    #[error("Tried to pop too much")]
    Pop,
}
