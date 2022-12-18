use hashbrown::hash_map::Entry as HEntry;
use indexmap::map::Entry;
use smallvec::SmallVec;

use crate::{
    index::Offset,
    typecheck::{Atom, AtomTerm, Query},
    *,
};
use std::{
    cell::UnsafeCell,
    fmt::{self, Debug},
    ops::Range,
};

enum Instr<'a> {
    Intersect {
        value_idx: usize,
        trie_accesses: Vec<(usize, TrieAccess<'a>)>,
    },
    Call {
        prim: Primitive,
        args: Vec<AtomTerm>,
        check: bool, // check or assign to output variable
    },
}

type Result = std::result::Result<(), ()>;

struct Program<'a>(Vec<Instr<'a>>);

impl<'a> std::fmt::Display for Program<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for instr in &self.0 {
            match instr {
                Instr::Intersect {
                    value_idx,
                    trie_accesses,
                } => {
                    write!(f, " Intersect @ {} ", value_idx)?;
                    for (trie_idx, a) in trie_accesses {
                        write!(f, "  {}: {}", trie_idx, a)?;
                    }
                    writeln!(f)?
                }
                Instr::Call { prim, args, check } => {
                    writeln!(f, " Call {:?} {:?} {:?}", prim, args, check)?;
                }
            }
        }
        Ok(())
    }
}

struct Context<'b> {
    query: &'b CompiledQuery,
    tuple: Vec<Value>,
    matches: usize,
}

impl<'b> Context<'b> {
    fn new(
        egraph: &'b EGraph,
        cq: &'b CompiledQuery,
        timestamp_ranges: &[Range<u32>],
    ) -> Option<(Self, Program<'b>, Vec<Option<usize>>)> {
        let ctx = Context {
            query: cq,
            tuple: vec![Value::fake(); cq.vars.len()],
            matches: 0,
        };

        let (program, _vars, intersections) = egraph.compile_program(cq, timestamp_ranges)?;

        Some((ctx, program, intersections))
    }

    fn eval<F>(&mut self, tries: &mut [&LazyTrie], program: &[Instr], f: &mut F) -> Result
    where
        F: FnMut(&[Value]) -> Result,
    {
        let (instr, program) = match program.split_first() {
            None => {
                self.matches += 1;
                return f(&self.tuple);
            }
            Some(pair) => pair,
        };

        match instr {
            Instr::Intersect {
                value_idx,
                trie_accesses,
            } => {
                match trie_accesses.as_slice() {
                    [(j, access)] => tries[*j].for_each(access, |value, trie| {
                        let old_trie = std::mem::replace(&mut tries[*j], trie);
                        self.tuple[*value_idx] = value;
                        self.eval(tries, program, f)?;
                        tries[*j] = old_trie;
                        Ok(())
                    }),
                    [a, b] => {
                        let (a, b) = if tries[a.0].len() <= tries[b.0].len() {
                            (a, b)
                        } else {
                            (b, a)
                        };
                        tries[a.0].for_each(&a.1, |value, ta| {
                            if let Some(tb) = tries[b.0].get(&b.1, value) {
                                let old_ta = std::mem::replace(&mut tries[a.0], ta);
                                let old_tb = std::mem::replace(&mut tries[b.0], tb);
                                self.tuple[*value_idx] = value;
                                self.eval(tries, program, f)?;
                                tries[a.0] = old_ta;
                                tries[b.0] = old_tb;
                            }
                            Ok(())
                        })
                    }
                    _ => {
                        let (j_min, access_min) = trie_accesses
                            .iter()
                            .min_by_key(|(j, _a)| tries[*j].len())
                            .unwrap();

                        let mut new_tries = tries.to_vec();

                        tries[*j_min].for_each(access_min, |value, min_trie| {
                            new_tries[*j_min] = min_trie;
                            for (j, access) in trie_accesses {
                                if j != j_min {
                                    if let Some(t) = tries[*j].get(access, value) {
                                        new_tries[*j] = t;
                                    } else {
                                        return Ok(());
                                    }
                                }
                            }

                            // at this point, new_tries is ready to go
                            self.tuple[*value_idx] = value;
                            self.eval(&mut new_tries, program, f)
                        })
                    }
                }
            }
            Instr::Call { prim, args, check } => {
                let (out, args) = args.split_last().unwrap();
                let mut values: Vec<Value> = vec![];
                for arg in args {
                    values.push(match arg {
                        AtomTerm::Var(v) => {
                            let i = self.query.vars.get_index_of(v).unwrap();
                            self.tuple[i]
                        }
                        AtomTerm::Value(val) => *val,
                    })
                }

                if let Some(res) = prim.apply(&values) {
                    match out {
                        AtomTerm::Var(v) => {
                            let i = self.query.vars.get_index_of(v).unwrap();
                            if *check && self.tuple[i] != res {
                                return Ok(());
                            }
                            self.tuple[i] = res;
                        }
                        AtomTerm::Value(val) => {
                            assert!(check);
                            if val != &res {
                                return Ok(());
                            }
                        }
                    }
                    self.eval(tries, program, f)?;
                }

                Ok(())
            }
        }
    }
}

enum Constraint {
    Eq(usize, usize),
    Const(usize, Value),
}

impl Constraint {
    fn check(&self, tuple: &[Value], out: &TupleOutput) -> bool {
        let get = |i: usize| {
            if i < tuple.len() {
                &tuple[i]
            } else {
                debug_assert_eq!(i, tuple.len());
                &out.value
            }
        };
        match self {
            Constraint::Eq(i, j) => get(*i) == get(*j),
            Constraint::Const(i, t) => get(*i) == t,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct VarInfo {
    /// indexes into the `atoms` field of CompiledQuery
    occurences: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct CompiledQuery {
    query: Query,
    pub vars: IndexMap<Symbol, VarInfo>,
}

impl EGraph {
    pub(crate) fn compile_gj_query(
        &self,
        query: Query,
        types: &IndexMap<Symbol, ArcSort>,
    ) -> CompiledQuery {
        // NOTE: this vars order only used for ordering the tuple,
        // It is not the GJ variable order.
        let mut vars: IndexMap<Symbol, VarInfo> = Default::default();

        for var in types.keys() {
            vars.entry(*var).or_default();
        }

        for (i, atom) in query.atoms.iter().enumerate() {
            for v in atom.vars() {
                // only count grounded occurrences
                vars.entry(v).or_default().occurences.push(i)
            }
        }

        for prim in &query.filters {
            for v in prim.vars() {
                vars.entry(v).or_default();
            }
        }

        CompiledQuery { query, vars }
    }

    fn make_trie_access(
        &self,
        var: Symbol,
        atom: &Atom<Symbol>,
        timestamp_range: Range<u32>,
    ) -> TrieAccess {
        let column = atom
            .args
            .iter()
            .position(|arg| arg == &AtomTerm::Var(var))
            .unwrap();

        let function = &self.functions[&atom.head];

        let mut constraints = vec![];
        for (i, t) in atom.args.iter().enumerate() {
            match t {
                AtomTerm::Value(val) => constraints.push(Constraint::Const(i, *val)),
                AtomTerm::Var(_v) => {
                    if let Some(j) = atom.args[..i].iter().position(|t2| t == t2) {
                        constraints.push(Constraint::Eq(j, i));
                    }
                }
            }
        }

        TrieAccess {
            function,
            timestamp_range,
            column,
            constraints,
        }
    }

    fn compile_program(
        &self,
        query: &CompiledQuery,
        timestamp_ranges: &[Range<u32>],
    ) -> Option<(
        Program,
        Vec<Symbol>,        /* variable ordering */
        Vec<Option<usize>>, /* the first column accessed per-atom */
    )> {
        #[derive(Default)]
        struct VarInfo2 {
            occurences: Vec<usize>,
            intersected_on: usize,
            size_guess: usize,
        }

        let atoms = &query.query.atoms;
        let mut vars: IndexMap<Symbol, VarInfo2> = Default::default();
        for (i, atom) in atoms.iter().enumerate() {
            for v in atom.vars() {
                vars.entry(v).or_default().occurences.push(i)
            }
        }

        for info in vars.values_mut() {
            info.occurences.sort_unstable();
            info.occurences.dedup();
        }

        let relation_sizes: Vec<usize> = atoms
            .iter()
            .zip(timestamp_ranges)
            .map(|(atom, range)| self.functions[&atom.head].get_size(range))
            .collect();

        if relation_sizes.iter().any(|&s| s == 0) {
            return None;
        }

        for (_v, info) in &mut vars {
            assert!(!info.occurences.is_empty());
            info.size_guess = info
                .occurences
                .iter()
                .map(|&i| relation_sizes[i])
                .min()
                .unwrap();
            // info.size_guess >>= info.occurences.len() - 1;
        }

        let mut ordered_vars = IndexMap::default();
        while !vars.is_empty() {
            let (&var, _info) = vars
                .iter()
                .max_by_key(|(_v, info)| {
                    let size = info.size_guess as isize;
                    (info.occurences.len(), info.intersected_on, -size)
                })
                .unwrap();

            let info = vars.remove(&var).unwrap();
            for &i in &info.occurences {
                for v in atoms[i].vars() {
                    if let Some(info) = vars.get_mut(&v) {
                        info.intersected_on += 1;
                    }
                }
            }

            ordered_vars.insert(var, info);
        }
        vars = ordered_vars;

        let mut initial_columns = vec![None; atoms.len()];
        let mut program: Vec<Instr> = vars
            .iter()
            .map(|(&v, info)| {
                let idx = query.vars.get_index_of(&v).unwrap_or_else(|| {
                    panic!("variable {} not found in query", v);
                });
                Instr::Intersect {
                    value_idx: idx,
                    trie_accesses: info
                        .occurences
                        .iter()
                        .map(|&atom_idx| {
                            let atom = &atoms[atom_idx];
                            let range = timestamp_ranges[atom_idx].clone();
                            let access = self.make_trie_access(v, atom, range);
                            let initial_col = &mut initial_columns[atom_idx];
                            if initial_col.is_none() {
                                *initial_col = Some(access.column);
                            }
                            (atom_idx, access)
                        })
                        .collect(),
                }
            })
            .collect();

        // now we can try to add primitives
        // TODO this is very inefficient, since primitives all at the end
        let mut extra = query.query.filters.clone();
        while !extra.is_empty() {
            let next = extra.iter().position(|p| {
                assert!(!p.args.is_empty());
                p.args[..p.args.len() - 1].iter().all(|a| match a {
                    AtomTerm::Var(v) => vars.contains_key(v),
                    AtomTerm::Value(_) => true,
                })
            });

            if let Some(i) = next {
                let p = extra.remove(i);
                let check = match p.args.last().unwrap() {
                    AtomTerm::Var(v) => match vars.entry(*v) {
                        Entry::Occupied(_) => true,
                        Entry::Vacant(e) => {
                            e.insert(Default::default());
                            false
                        }
                    },
                    AtomTerm::Value(_) => true,
                };
                program.push(Instr::Call {
                    prim: p.head.clone(),
                    args: p.args.clone(),
                    check,
                });
            } else {
                panic!("cycle")
            }
        }

        Some((
            Program(program),
            vars.into_keys().collect(),
            initial_columns,
        ))
    }

    pub(crate) fn run_query<F>(&self, cq: &CompiledQuery, timestamp: u32, mut f: F)
    where
        F: FnMut(&[Value]) -> Result,
    {
        let has_atoms = !cq.query.atoms.is_empty();

        if has_atoms {
            let do_seminaive = self.seminaive;
            // for the later atoms, we consider everything
            let mut timestamp_ranges = vec![0..u32::MAX; cq.query.atoms.len()];
            for (atom_i, atom) in cq.query.atoms.iter().enumerate() {
                // this time, we only consider "new stuff" for this atom
                if do_seminaive {
                    timestamp_ranges[atom_i] = timestamp..u32::MAX;
                }

                // do the gj
                if let Some((mut ctx, program, cols)) = Context::new(self, cq, &timestamp_ranges) {
                    log::debug!(
                        "Query: {}\nNew atom: {}\nVars: {}\nProgram\n{}",
                        cq.query,
                        atom,
                        ListDisplay(cq.vars.keys(), " "),
                        program
                    );
                    let mut tries = Vec::with_capacity(cq.query.atoms.len());
                    for ((atom, ts), col) in cq
                        .query
                        .atoms
                        .iter()
                        .zip(timestamp_ranges.iter())
                        .zip(cols.iter())
                    {
                        // tries.push(LazyTrie::default());
                        if let Some(target) = col {
                            if let Some(col) = self.functions[&atom.head].column_index(*target, ts)
                            {
                                tries.push(LazyTrie::from_column_index(col))
                            } else {
                                tries.push(LazyTrie::default());
                            }
                        } else {
                            tries.push(LazyTrie::default());
                        }
                    }
                    let mut trie_refs = tries.iter().collect::<Vec<_>>();
                    ctx.eval(&mut trie_refs, &program.0, &mut f).unwrap_or(());
                    log::debug!("Matched {} times", ctx.matches);
                }

                if !do_seminaive {
                    break;
                }

                // now we can fix this atom to be "old stuff" only
                // range is half-open; timestamp is excluded
                timestamp_ranges[atom_i] = 0..timestamp;
            }
        } else if let Some((mut ctx, program, _)) = Context::new(self, cq, &[]) {
            let tries = LazyTrie::make_initial_vec(cq.query.atoms.len());
            let mut trie_refs = tries.iter().collect::<Vec<_>>();
            ctx.eval(&mut trie_refs, &program.0, &mut f).unwrap_or(());
        }
    }
}

struct LazyTrie(UnsafeCell<LazyTrieInner>);

impl Debug for LazyTrie {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(unsafe { &*self.0.get() }, f)
    }
}

type SparseMap = HashMap<Value, LazyTrie>;
type RowIdx = u32;

#[derive(Debug)]
enum LazyTrieInner {
    Borrowed {
        index: Rc<ColumnIndex>,
        map: SparseMap,
    },
    Delayed(SmallVec<[RowIdx; 4]>),
    Sparse(SparseMap),
}

impl Default for LazyTrie {
    fn default() -> Self {
        LazyTrie(UnsafeCell::new(LazyTrieInner::Delayed(Default::default())))
    }
}

impl LazyTrie {
    fn make_initial_vec(n: usize) -> Vec<Self> {
        (0..n).map(|_| LazyTrie::default()).collect()
    }

    fn len(&self) -> usize {
        match unsafe { &*self.0.get() } {
            LazyTrieInner::Delayed(v) => v.len(),
            LazyTrieInner::Sparse(m) => m.len(),
            LazyTrieInner::Borrowed { index, .. } => index.len(),
        }
    }
    fn from_column_index(index: Rc<ColumnIndex>) -> LazyTrie {
        LazyTrie(UnsafeCell::new(LazyTrieInner::Borrowed {
            index,
            map: Default::default(),
        }))
    }
    fn from_indexes(ixs: impl Iterator<Item = usize>) -> Option<LazyTrie> {
        let data = SmallVec::from_iter(ixs.map(|x| x as RowIdx));
        if data.is_empty() {
            return None;
        }

        Some(LazyTrie(UnsafeCell::new(LazyTrieInner::Delayed(data))))
    }

    unsafe fn force_mut(&self, access: &TrieAccess) -> *mut LazyTrieInner {
        let this = &mut *self.0.get();
        if let LazyTrieInner::Delayed(idxs) = this {
            *this = access.make_trie_inner(idxs);
        }
        self.0.get()
    }

    fn force_borrowed(&self, access: &TrieAccess) -> &LazyTrieInner {
        let this = unsafe { &mut *self.0.get() };
        match this {
            LazyTrieInner::Borrowed { index, .. } => {
                let mut map = SparseMap::with_capacity_and_hasher(index.len(), Default::default());
                map.extend(index.iter().filter_map(|(v, ixs)| {
                    LazyTrie::from_indexes(access.filter_live(ixs)).map(|trie| (v, trie))
                }));
                *this = LazyTrieInner::Sparse(map);
            }
            LazyTrieInner::Delayed(idxs) => {
                *this = access.make_trie_inner(idxs);
            }
            LazyTrieInner::Sparse(_) => {}
        }
        unsafe { &*self.0.get() }
    }

    fn for_each<'a>(
        &'a self,
        access: &TrieAccess,
        mut f: impl FnMut(Value, &'a LazyTrie) -> Result,
    ) -> Result {
        // There is probably something cleaner to do here compared with the
        // `force_borrowed` construct.
        match self.force_borrowed(access) {
            LazyTrieInner::Sparse(m) => {
                for (k, v) in m {
                    f(*k, v)?;
                }
                Ok(())
            }
            LazyTrieInner::Borrowed { .. } | LazyTrieInner::Delayed(_) => unreachable!(),
        }
    }

    fn get(&self, access: &TrieAccess, value: Value) -> Option<&LazyTrie> {
        match unsafe { &mut *self.force_mut(access) } {
            LazyTrieInner::Sparse(m) => m.get(&value),
            LazyTrieInner::Borrowed { index, map } => {
                let ixs = index.get(&value)?;
                match map.entry(value) {
                    HEntry::Occupied(o) => Some(o.into_mut()),
                    HEntry::Vacant(v) => {
                        Some(v.insert(LazyTrie::from_indexes(access.filter_live(ixs))?))
                    }
                }
            }
            LazyTrieInner::Delayed(_) => unreachable!(),
        }
    }
}

struct TrieAccess<'a> {
    function: &'a Function,
    timestamp_range: Range<u32>,
    column: usize,
    constraints: Vec<Constraint>,
}

impl<'a> std::fmt::Display for TrieAccess<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.function.decl.name, self.column)
    }
}

impl<'a> TrieAccess<'a> {
    fn filter_live<'b: 'a>(&'b self, ixs: &'b [Offset]) -> impl Iterator<Item = usize> + 'a {
        ixs.iter().copied().filter_map(|ix| {
            let ix = ix as usize;
            let (inp, out) = self.function.nodes.get_index(ix)?;
            if self.timestamp_range.contains(&out.timestamp)
                && self.constraints.iter().all(|c| c.check(inp, out))
            {
                Some(ix)
            } else {
                None
            }
        })
    }
    #[cold]
    fn make_trie_inner(&self, idxs: &[RowIdx]) -> LazyTrieInner {
        let arity = self.function.schema.input.len();
        let mut map = SparseMap::default();
        let mut insert = |i: usize, tup: &[Value], out: &TupleOutput, val: Value| {
            use hashbrown::hash_map::Entry;
            if self.timestamp_range.contains(&out.timestamp)
                && self.constraints.iter().all(|c| c.check(tup, out))
            {
                match map.entry(val) {
                    Entry::Occupied(mut e) => {
                        if let LazyTrieInner::Delayed(ref mut v) = e.get_mut().0.get_mut() {
                            v.push(i as RowIdx)
                        } else {
                            unreachable!()
                        }
                    }
                    Entry::Vacant(e) => {
                        e.insert(LazyTrie(UnsafeCell::new(LazyTrieInner::Delayed(
                            smallvec::smallvec![i as RowIdx,],
                        ))));
                    }
                }
            }
        };

        if idxs.is_empty() {
            if self.column < arity {
                for (i, tup, out) in self.function.iter_timestamp_range(&self.timestamp_range) {
                    insert(i, tup, out, tup[self.column])
                }
            } else {
                assert_eq!(self.column, arity);
                for (i, tup, out) in self.function.iter_timestamp_range(&self.timestamp_range) {
                    insert(i, tup, out, out.value);
                }
            };
        } else if self.column < arity {
            for idx in idxs {
                let i = *idx as usize;
                if let Some((tup, out)) = self.function.nodes.get_index(i) {
                    insert(i, tup, out, tup[self.column])
                }
            }
        } else {
            assert_eq!(self.column, arity);
            for idx in idxs {
                let i = *idx as usize;
                if let Some((tup, out)) = self.function.nodes.get_index(i) {
                    insert(i, tup, out, out.value)
                }
            }
        }

        // // Density test
        // if !map.is_empty() {
        //     let min = map.keys().map(|v| v.bits).min().unwrap();
        //     let max = map.keys().map(|v| v.bits).max().unwrap();
        //     let len = map.len();
        //     if max - min <= len as u64 * 2 {
        //         println!("Trie is dense with len {len}!");
        //     } else {
        //         println!("Trie is not dense with len {len}!");
        //     }
        // }

        LazyTrieInner::Sparse(map)
    }
}
