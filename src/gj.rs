use hashbrown::hash_map::Entry as HEntry;
use indexmap::map::Entry;
use log::log_enabled;
use smallvec::SmallVec;

use crate::{
    function::index::Offset,
    typecheck::{Atom, AtomTerm, ResolvedCall},
    *,
};
use std::{
    cell::UnsafeCell,
    fmt::{self, Debug},
    ops::Range,
};

type Query = crate::typecheck::Query<ResolvedCall>;

#[derive(Clone, Debug)]
enum Instr<'a> {
    Intersect {
        value_idx: usize,
        variable_name: Symbol,
        info: VarInfo2,
        trie_accesses: Vec<(usize, TrieAccess<'a>)>,
    },
    ConstrainConstant {
        index: usize,
        val: Value,
        trie_access: TrieAccess<'a>,
    },
    Call {
        prim: Primitive,
        args: Vec<AtomTerm>,
        check: bool, // check or assign to output variable
    },
}

// FIXME @mwillsey awful name, bad bad bad
#[derive(Default, Debug, Clone)]
struct VarInfo2 {
    occurences: Vec<usize>,
    intersected_on: usize,
    size_guess: usize,
}

struct InputSizes<'a> {
    cur_stage: usize,
    // a map from from stage to vector of costs for each stage,
    // where 'cost' is the largest relation being intersected
    stage_sizes: &'a mut HashMap<usize, Vec<usize>>,
}

impl<'a> InputSizes<'a> {
    fn add_measurement(&mut self, max_size: usize) {
        self.stage_sizes
            .entry(self.cur_stage)
            .or_default()
            .push(max_size);
    }

    fn next(&mut self) -> InputSizes {
        InputSizes {
            cur_stage: self.cur_stage + 1,
            stage_sizes: self.stage_sizes,
        }
    }
}

type Result = std::result::Result<(), ()>;

struct Program<'a>(Vec<Instr<'a>>);

impl<'a> std::fmt::Display for Instr<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instr::Intersect {
                value_idx,
                trie_accesses,
                variable_name,
                info,
            } => {
                write!(
                    f,
                    " Intersect @ {value_idx} sg={sg:6} {variable_name:15}",
                    sg = info.size_guess
                )?;
                for (trie_idx, a) in trie_accesses {
                    write!(f, "  {}: {}", trie_idx, a)?;
                }
                writeln!(f)?
            }
            Instr::ConstrainConstant {
                index,
                val,
                trie_access,
            } => {
                writeln!(f, " ConstrainConstant {index} {trie_access} = {val:?}")?;
            }
            Instr::Call { prim, args, check } => {
                writeln!(f, " Call {:?} {:?} {:?}", prim, args, check)?;
            }
        }
        Ok(())
    }
}

impl<'a> std::fmt::Display for Program<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, instr) in self.0.iter().enumerate() {
            write!(f, "{i:2}. {}", instr)?;
        }
        Ok(())
    }
}

struct Context<'b> {
    query: &'b CompiledQuery,
    join_var_ordering: Vec<Symbol>,
    tuple: Vec<Value>,
    matches: usize,
    egraph: &'b EGraph,
}

impl<'b> Context<'b> {
    fn new(
        egraph: &'b EGraph,
        cq: &'b CompiledQuery,
        timestamp_ranges: &[Range<u32>],
    ) -> Option<(Self, Program<'b>, Vec<Option<usize>>)> {
        let (program, join_var_ordering, intersections) =
            egraph.compile_program(cq, timestamp_ranges)?;

        let ctx = Context {
            query: cq,
            tuple: vec![Value::fake(); cq.vars.len()],
            join_var_ordering,
            matches: 0,
            egraph,
        };

        Some((ctx, program, intersections))
    }

    fn eval<F>(
        &mut self,
        tries: &mut [&LazyTrie],
        program: &[Instr],
        mut stage: InputSizes,
        f: &mut F,
    ) -> Result
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
            Instr::ConstrainConstant {
                index,
                val,
                trie_access,
            } => {
                if let Some(next) = tries[*index].get(trie_access, *val) {
                    let old = tries[*index];
                    tries[*index] = next;
                    self.eval(tries, program, stage.next(), f)?;
                    tries[*index] = old;
                }
                Ok(())
            }
            Instr::Intersect {
                value_idx,
                trie_accesses,
                ..
            } => {
                if let Some(x) = trie_accesses
                    .iter()
                    .map(|(atom, _)| tries[*atom].len())
                    .max()
                {
                    stage.add_measurement(x);
                }

                match trie_accesses.as_slice() {
                    [(j, access)] => tries[*j].for_each(access, |value, trie| {
                        let old_trie = std::mem::replace(&mut tries[*j], trie);
                        self.tuple[*value_idx] = value;
                        self.eval(tries, program, stage.next(), f)?;
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
                                self.eval(tries, program, stage.next(), f)?;
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
                            self.eval(&mut new_tries, program, stage.next(), f)
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
                        AtomTerm::Literal(lit) => self.egraph.eval_lit(lit),
                        AtomTerm::Global(g) => self.egraph.global_bindings.get(g).unwrap().1,
                    })
                }

                if let Some(res) = prim.apply(&values, self.egraph) {
                    match out {
                        AtomTerm::Var(v) => {
                            let i = self.query.vars.get_index_of(v).unwrap();

                            if *check {
                                assert_ne!(self.tuple[i], Value::fake());
                                if self.tuple[i] != res {
                                    return Ok(());
                                }
                            }

                            self.tuple[i] = res;
                        }
                        AtomTerm::Literal(lit) => {
                            assert!(check);
                            let val = &self.egraph.eval_lit(lit);
                            if val != &res {
                                return Ok(());
                            }
                        }
                        AtomTerm::Global(g) => {
                            assert!(check);
                            let (sort, val, _ts) = self.egraph.global_bindings.get(g).unwrap();
                            assert!(sort.name() == res.tag);
                            if val.bits != res.bits {
                                return Ok(());
                            }
                        }
                    }
                    self.eval(tries, program, stage.next(), f)?;
                }

                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug)]
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
    // Ordering is used for the tuple
    // The GJ variable ordering is stored in the context
    pub vars: IndexMap<Symbol, VarInfo>,
}

impl EGraph {
    pub(crate) fn compile_gj_query(
        &self,
        query: Query,
        types: &IndexMap<Symbol, ArcSort>,
    ) -> CompiledQuery {
        // NOTE: this vars order only used for ordering the tuple storing the resulting match
        // It is not the GJ variable order.
        let mut vars: IndexMap<Symbol, VarInfo> = Default::default();

        for var in types.keys() {
            vars.entry(*var).or_default();
        }

        for (i, atom) in query.funcs().enumerate() {
            for v in atom.vars() {
                // only count grounded occurrences
                vars.entry(v).or_default().occurences.push(i)
            }
        }

        // make sure everyone has an entry in the vars table
        for prim in query.filters() {
            for v in prim.vars() {
                vars.entry(v).or_default();
            }
        }

        CompiledQuery { query, vars }
    }

    fn make_trie_access_for_column(
        &self,
        atom: &Atom<Symbol>,
        column: usize,
        timestamp_range: Range<u32>,
    ) -> TrieAccess {
        let function = &self.functions[&atom.head];

        let mut constraints = vec![];
        for (i, t) in atom.args.iter().enumerate() {
            match t {
                AtomTerm::Literal(lit) => {
                    let val = self.eval_lit(lit);
                    constraints.push(Constraint::Const(i, val))
                }
                AtomTerm::Global(g) => {
                    constraints.push(Constraint::Const(i, self.global_bindings.get(g).unwrap().1))
                }
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
        self.make_trie_access_for_column(atom, column, timestamp_range)
    }

    // Returns `None` when no program is needed,
    // for example when there is nothing in one of the tables.
    fn compile_program(
        &self,
        query: &CompiledQuery,
        timestamp_ranges: &[Range<u32>],
    ) -> Option<(
        Program,
        Vec<Symbol>,        /* variable ordering */
        Vec<Option<usize>>, /* the first column accessed per-atom */
    )> {
        let atoms: &Vec<_> = &query.query.funcs().collect();
        let mut vars: IndexMap<Symbol, VarInfo2> = Default::default();
        let mut constants =
            IndexMap::<usize /* atom */, Vec<(usize /* column */, Value)>>::default();

        for (i, atom) in atoms.iter().enumerate() {
            for (col, arg) in atom.args.iter().enumerate() {
                match arg {
                    AtomTerm::Var(var) => vars.entry(*var).or_default().occurences.push(i),
                    AtomTerm::Literal(lit) => {
                        let val = self.eval_lit(lit);
                        constants.entry(i).or_default().push((col, val));
                    }
                    AtomTerm::Global(g) => {
                        let val = self.global_bindings.get(g).unwrap().1;
                        constants.entry(i).or_default().push((col, val));
                    }
                }
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

        // here we are picking the variable ordering
        let mut ordered_vars = IndexMap::default();
        while !vars.is_empty() {
            let mut var_cost = vars
                .iter()
                .map(|(v, info)| {
                    let size = info.size_guess as isize;
                    let cost = (info.occurences.len(), info.intersected_on, -size);
                    (cost, v)
                })
                .collect::<Vec<_>>();
            var_cost.sort();
            var_cost.reverse();

            log::debug!("Variable costs: {:?}", ListDebug(&var_cost, "\n"));

            let var = *var_cost[0].1;
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
        let const_instrs = constants.iter().flat_map(|(atom, consts)| {
            let initial_col = &mut initial_columns[*atom];
            if initial_col.is_none() {
                *initial_col = Some(consts[0].0);
            }
            consts.iter().map(|(col, val)| {
                let range = timestamp_ranges[*atom].clone();
                let trie_access = self.make_trie_access_for_column(&atoms[*atom], *col, range);

                Instr::ConstrainConstant {
                    index: *atom,
                    val: *val,
                    trie_access,
                }
            })
        });
        let mut program: Vec<Instr> = const_instrs.collect();

        let var_instrs = vars.iter().map(|(&v, info)| {
            let value_idx = query.vars.get_index_of(&v).unwrap_or_else(|| {
                panic!("variable {} not found in query", v);
            });
            Instr::Intersect {
                value_idx,
                variable_name: v,
                info: info.clone(),
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
        });
        program.extend(var_instrs);

        // now we can try to add primitives
        let mut extra: Vec<_> = query.query.filters().collect();
        while !extra.is_empty() {
            let next = extra.iter().position(|p| {
                assert!(!p.args.is_empty());
                p.args[..p.args.len() - 1].iter().all(|a| match a {
                    AtomTerm::Var(v) => vars.contains_key(v),
                    AtomTerm::Literal(_) => true,
                    AtomTerm::Global(_) => true,
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
                    AtomTerm::Literal(_) => true,
                    AtomTerm::Global(_) => true,
                };
                program.push(Instr::Call {
                    prim: p.head.clone(),
                    args: p.args.clone(),
                    check,
                });
            } else {
                panic!("cycle {:#?}", query)
            }
        }

        let resulting_program = Program(program);
        self.sanity_check_program(&resulting_program, query);

        Some((
            resulting_program,
            vars.into_keys().collect(),
            initial_columns,
        ))
    }

    fn sanity_check_program(&self, program: &Program, query: &CompiledQuery) {
        // sanity check the program
        let mut tuple_valid = vec![false; query.vars.len()];
        for instr in &program.0 {
            match instr {
                Instr::Intersect { value_idx, .. } => {
                    assert!(!tuple_valid[*value_idx]);
                    tuple_valid[*value_idx] = true;
                }
                Instr::ConstrainConstant { .. } => {}
                Instr::Call { check, args, .. } => {
                    let Some((last, args)) = args.split_last() else {
                        continue;
                    };

                    for a in args {
                        if let AtomTerm::Var(v) = a {
                            let i = query.vars.get_index_of(v).unwrap();
                            assert!(tuple_valid[i]);
                        }
                    }

                    match last {
                        AtomTerm::Var(v) => {
                            let i = query.vars.get_index_of(v).unwrap();
                            assert_eq!(*check, tuple_valid[i], "{instr}");
                            if !*check {
                                tuple_valid[i] = true;
                            }
                        }
                        AtomTerm::Literal(_) => {
                            assert!(*check);
                        }
                        AtomTerm::Global(_) => {
                            assert!(*check);
                        }
                    }
                }
            }
        }
    }

    fn gj_for_atom<F>(
        &self,
        // for debugging, the atom seminaive is focusing on
        atom_i: Option<usize>,
        timestamp_ranges: &[Range<u32>],
        cq: &CompiledQuery,
        include_subsumed: bool,
        mut f: F,
    ) where
        F: FnMut(&[Value]) -> Result,
    {
        // do the gj
        if let Some((mut ctx, program, cols)) = Context::new(self, cq, timestamp_ranges) {
            let start = Instant::now();
            let atom_info = if let Some(atom_i) = atom_i {
                let atom = &cq.query.funcs().collect::<Vec<_>>()[atom_i];
                format!("New atom: {atom}")
            } else {
                "Seminaive disabled".to_string()
            };
            log::debug!(
                "Query:\n{q}\n{atom_info}\nTuple: {tuple}\nJoin order: {order}\nProgram\n{program}",
                q = cq.query,
                order = ListDisplay(&ctx.join_var_ordering, " "),
                tuple = ListDisplay(cq.vars.keys(), " "),
            );
            let mut tries = Vec::with_capacity(cq.query.funcs().collect::<Vec<_>>().len());
            for ((atom, ts), col) in cq
                .query
                .funcs()
                .zip(timestamp_ranges.iter())
                .zip(cols.iter())
            {
                // tries.push(LazyTrie::default());
                if let Some(target) = col {
                    if let Some(col) = self.functions[&atom.head].column_index(*target, ts) {
                        tries.push(LazyTrie::from_column_index(col, include_subsumed))
                    } else {
                        tries.push(LazyTrie::new(include_subsumed));
                    }
                } else {
                    tries.push(LazyTrie::new(include_subsumed));
                }
            }
            let mut trie_refs = tries.iter().collect::<Vec<_>>();
            let mut meausrements = HashMap::<usize, Vec<usize>>::default();
            let stages = InputSizes {
                stage_sizes: &mut meausrements,
                cur_stage: 0,
            };
            ctx.eval(&mut trie_refs, &program.0, stages, &mut f)
                .unwrap_or(());
            let mut sums = Vec::from_iter(
                meausrements
                    .iter()
                    .map(|(x, y)| (*x, y.iter().copied().sum::<usize>())),
            );
            sums.sort_by_key(|(i, _sum)| *i);
            if log_enabled!(log::Level::Debug) {
                for (i, sum) in sums {
                    log::debug!("stage {i} total cost {sum}");
                }
            }
            let duration = start.elapsed();
            log::debug!("Matched {} times (took {:?})", ctx.matches, duration,);
            let iteration = self
                .ruleset_iteration
                .get::<Symbol>(&"".into())
                .unwrap_or(&0);
            if duration.as_millis() > 1000 {
                log::warn!(
                    "Query took a long time at iter {iteration} : {:?}",
                    duration
                );
            }
        }
    }

    pub(crate) fn run_query<F>(
        &self,
        cq: &CompiledQuery,
        timestamp: u32,
        include_subsumed: bool,
        mut f: F,
    ) where
        F: FnMut(&[Value]) -> Result,
    {
        let has_atoms = !cq.query.funcs().collect::<Vec<_>>().is_empty();

        if has_atoms {
            // check if any globals updated
            let mut global_updated = false;
            for atom in cq.query.funcs() {
                for arg in &atom.args {
                    if let AtomTerm::Global(g) = arg {
                        if self.global_bindings.get(g).unwrap().2 > timestamp {
                            global_updated = true;
                        }
                    }
                }
            }

            let do_seminaive = self.seminaive && !global_updated;
            // for the later atoms, we consider everything
            let mut timestamp_ranges =
                vec![0..u32::MAX; cq.query.funcs().collect::<Vec<_>>().len()];
            if do_seminaive {
                for (atom_i, _atom) in cq.query.funcs().enumerate() {
                    timestamp_ranges[atom_i] = timestamp..u32::MAX;

                    self.gj_for_atom(
                        Some(atom_i),
                        &timestamp_ranges,
                        cq,
                        include_subsumed,
                        &mut f,
                    );
                    // now we can fix this atom to be "old stuff" only
                    // range is half-open; timestamp is excluded
                    timestamp_ranges[atom_i] = 0..timestamp;
                }
            } else {
                self.gj_for_atom(None, &timestamp_ranges, cq, include_subsumed, &mut f);
            }
        } else if let Some((mut ctx, program, _)) = Context::new(self, cq, &[]) {
            let mut meausrements = HashMap::<usize, Vec<usize>>::default();
            let stages = InputSizes {
                stage_sizes: &mut meausrements,
                cur_stage: 0,
            };
            let tries = LazyTrie::make_initial_vec(
                cq.query.funcs().collect::<Vec<_>>().len(),
                include_subsumed,
            ); // TODO: bad use of collect here
            let mut trie_refs = tries.iter().collect::<Vec<_>>();
            ctx.eval(&mut trie_refs, &program.0, stages, &mut f)
                .unwrap_or(());
        }
    }
}

type IncludeSubsumed = bool;
struct LazyTrie(UnsafeCell<LazyTrieInner>, IncludeSubsumed);

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

impl LazyTrie {
    fn new(include_subsumed: bool) -> LazyTrie {
        LazyTrie(
            UnsafeCell::new(LazyTrieInner::Delayed(Default::default())),
            include_subsumed,
        )
    }
    fn make_initial_vec(n: usize, include_subsumed: bool) -> Vec<Self> {
        (0..n).map(|_| LazyTrie::new(include_subsumed)).collect()
    }

    fn len(&self) -> usize {
        match unsafe { &*self.0.get() } {
            LazyTrieInner::Delayed(v) => v.len(),
            LazyTrieInner::Sparse(m) => m.len(),
            LazyTrieInner::Borrowed { index, .. } => index.len(),
        }
    }
    fn from_column_index(index: Rc<ColumnIndex>, include_subsumed: bool) -> LazyTrie {
        LazyTrie(
            UnsafeCell::new(LazyTrieInner::Borrowed {
                index,
                map: Default::default(),
            }),
            include_subsumed,
        )
    }
    fn from_indexes(ixs: impl Iterator<Item = usize>, include_subsumed: bool) -> Option<LazyTrie> {
        let data = SmallVec::from_iter(ixs.map(|x| x as RowIdx));
        if data.is_empty() {
            return None;
        }

        Some(LazyTrie(
            UnsafeCell::new(LazyTrieInner::Delayed(data)),
            include_subsumed,
        ))
    }

    unsafe fn force_mut(&self, access: &TrieAccess) -> *mut LazyTrieInner {
        let this = &mut *self.0.get();
        if let LazyTrieInner::Delayed(idxs) = this {
            *this = access.make_trie_inner(idxs, self.1);
        }
        self.0.get()
    }

    fn force_borrowed(&self, access: &TrieAccess) -> &LazyTrieInner {
        let this = unsafe { &mut *self.0.get() };
        match this {
            LazyTrieInner::Borrowed { index, .. } => {
                let mut map = SparseMap::with_capacity_and_hasher(index.len(), Default::default());
                map.extend(index.iter().filter_map(|(v, ixs)| {
                    LazyTrie::from_indexes(access.filter_live(ixs, self.1), self.1)
                        .map(|trie| (v, trie))
                }));
                *this = LazyTrieInner::Sparse(map);
            }
            LazyTrieInner::Delayed(idxs) => {
                *this = access.make_trie_inner(idxs, self.1);
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
                    HEntry::Vacant(v) => Some(v.insert(LazyTrie::from_indexes(
                        access.filter_live(ixs, self.1),
                        self.1,
                    )?)),
                }
            }
            LazyTrieInner::Delayed(_) => unreachable!(),
        }
    }
}

#[derive(Clone, Debug)]
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
    fn filter_live<'b: 'a>(
        &'b self,
        ixs: &'b [Offset],
        include_subsumed: bool,
    ) -> impl Iterator<Item = usize> + 'a {
        ixs.iter().copied().filter_map(move |ix| {
            let ix = ix as usize;
            let (inp, out) = self.function.nodes.get_index(ix)?;
            if self.timestamp_range.contains(&out.timestamp)
                && self.constraints.iter().all(|c| c.check(inp, out))
                // If we are querying to run a rule, we should not include subsumed expressions
                // but if we are querying to run a check we can include them.
                // So we have a flag to control this behavior and pass it down to here.
                && (include_subsumed || !self.function.nodes.get_index_row(ix).unwrap().subsumed)
            {
                Some(ix)
            } else {
                None
            }
        })
    }

    #[cold]
    fn make_trie_inner(&self, idxs: &[RowIdx], include_subsumed: bool) -> LazyTrieInner {
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
                        e.insert(LazyTrie(
                            UnsafeCell::new(LazyTrieInner::Delayed(smallvec::smallvec![
                                i as RowIdx,
                            ])),
                            include_subsumed,
                        ));
                    }
                }
            }
        };

        if idxs.is_empty() {
            let rows = self.function.iter_timestamp_range(&self.timestamp_range);
            if self.column < arity {
                for (i, tup, out) in rows {
                    insert(i, tup, out, tup[self.column])
                }
            } else {
                assert_eq!(self.column, arity);
                for (i, tup, out) in rows {
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
