use hashbrown::hash_map::Entry as HEntry;
use indexmap::map::Entry;
use log::log_enabled;
use smallvec::SmallVec;

use crate::{
    function::index::Offset,
    typecheck::{Atom, AtomTerm, Query},
    *,
};
use std::{
    cell::UnsafeCell,
    cmp::min,
    fmt::{self, Debug},
    ops::Range,
};

#[derive(Clone)]
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
    is_nonparent_input: bool,
}

struct InputSizes<'a> {
    cur_stage: usize,
    // Each stage we're intersecting {set_k}_k
    // O(max_k(|set_k|))
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
                        AtomTerm::Value(val) => *val,
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
                        AtomTerm::Value(val) => {
                            assert!(check);
                            if val != &res {
                                return Ok(());
                            }
                        }
                        AtomTerm::Global(g) => {
                            assert!(check);
                            let (sort, val) = self.egraph.global_bindings.get(g).unwrap();
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

#[derive(Clone)]
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

        for (i, atom) in query.atoms.iter().enumerate() {
            for v in atom.vars() {
                // only count grounded occurrences
                vars.entry(v).or_default().occurences.push(i)
            }
        }

        // make sure everyone has an entry in the vars table
        for prim in &query.filters {
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
                AtomTerm::Value(val) => constraints.push(Constraint::Const(i, *val)),
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

    fn compile_program(
        &self,
        query: &CompiledQuery,
        timestamp_ranges: &[Range<u32>],
    ) -> Option<(
        Program,
        Vec<Symbol>,        /* variable ordering */
        Vec<Option<usize>>, /* the first column accessed per-atom */
    )> {
        let atoms = &query.query.atoms;
        let mut vars: IndexMap<Symbol, VarInfo2> = Default::default();
        let mut constants =
            IndexMap::<usize /* atom */, Vec<(usize /* column */, Value)>>::default();

        for (i, atom) in atoms.iter().enumerate() {
            for (col, arg) in atom.args.iter().enumerate() {
                match arg {
                    AtomTerm::Var(var) => vars.entry(*var).or_default().occurences.push(i),
                    AtomTerm::Value(val) => {
                        constants.entry(i).or_default().push((col, *val));
                    }
                    AtomTerm::Global(g) => {
                        let val = self.global_bindings.get(g).unwrap().1;
                        constants.entry(i).or_default().push((col, val));
                    }
                }
            }
        }

        for (_i, atom) in atoms.iter().enumerate() {
            if !atom.head.as_str().contains("_Parent_") {
                if let Some((_last, args)) = atom.args.split_last() {
                    for arg in args {
                        if let AtomTerm::Var(var) = arg {
                            vars.entry(*var).or_default().is_nonparent_input = true;
                        }
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
            .map(|(atom, range)| {
                if atom.head.as_str().contains("_Parent_") {
                    usize::MAX
                } else {
                    self.functions[&atom.head].get_size(range)
                }
            })
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

        let mut unionfind = UnionFind::default();
        let mut lookup = HashMap::<Symbol, Id>::default();
        for atom in atoms {
            for var in atom.vars() {
                if !lookup.contains_key(&var) {
                    let id = unionfind.make_set();
                    lookup.insert(var, id);
                }
            }
        }

        for atom in atoms {
            if atom.head.as_str().contains("_Parent_") {
                let first_var = atom.vars().next().unwrap();
                for var in atom.vars() {
                    unionfind.union_raw(lookup[&first_var], lookup[&var]);
                }
            }
        }

        let mut var_count_nonparent = HashMap::<Id, usize>::default();
        for atom in atoms {
            if !atom.head.as_str().contains("_Parent_") {
                let mut already_counted = HashSet::default();
                for var in atom.vars() {
                    let id = unionfind.find(lookup[&var]);
                    if already_counted.insert(id) {
                        *var_count_nonparent.entry(id).or_default() += 1;
                    }
                }
            }
        }

        let mut class_vars = HashMap::<Id, Vec<Symbol>>::default();
        for (var, id) in &lookup {
            class_vars
                .entry(unionfind.find(*id))
                .or_default()
                .push(*var);
        }
        let all_vars = vars.keys().cloned().collect::<Vec<Symbol>>();
        // the size guess for variables is the minimum across
        // all variables in the same class
        for v in all_vars {
            for var in &class_vars[&unionfind.find(lookup[&v])] {
                if vars.contains_key(var) && vars.contains_key(&v) {
                    let new_guess = min(vars[&v].size_guess, vars[var].size_guess);
                    vars[&v].size_guess = new_guess;
                }
            }
            // info.size_guess >>= info.occurences.len() - 1;
        }

        // here we are picking the variable ordering
        let mut ordered_vars = IndexMap::default();
        while !vars.is_empty() {
            let mut var_is_parent_lookup = HashMap::<Symbol, usize>::default();
            for atom in atoms {
                if atom.head.as_str().contains("_Parent_") {
                    let (todo, _others): (Vec<Symbol>, Vec<Symbol>) =
                        atom.vars().partition(|v| vars.contains_key(v));
                    if todo.len() == 1 {
                        let var = todo[0];
                        *var_is_parent_lookup.entry(var).or_default() += 1;
                    }
                }
            }

            let mut var_cost = vars
                .iter()
                .map(|(v, info)| {
                    let size = info.size_guess as isize;
                    let cost = (
                        var_is_parent_lookup.get(v).unwrap_or(&0),
                        var_count_nonparent
                            .get(&unionfind.find(lookup[v]))
                            .unwrap_or(&0),
                        info.intersected_on,
                        //occurences_nonparent.get(*v).unwrap_or(&0),
                        -size,
                    );
                    (cost, v)
                })
                .collect::<Vec<_>>();
            var_cost.sort();
            var_cost.reverse();

            log::debug!("Variable costs: {:?}", ListDebug(&var_cost, "\n"));

            let var = *var_cost[0].1;
            let info = vars.remove(&var).unwrap();
            for &i in &info.occurences {
                let (last, _rest) = atoms[i].args.split_last().unwrap();
                for v in atoms[i].vars() {
                    if last != &AtomTerm::Var(v) {
                        if let Some(info) = vars.get_mut(&v) {
                            info.intersected_on += 1;
                        }
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
        let mut extra = query.query.filters.clone();
        while !extra.is_empty() {
            let next = extra.iter().position(|p| {
                assert!(!p.args.is_empty());
                p.args[..p.args.len() - 1].iter().all(|a| match a {
                    AtomTerm::Var(v) => vars.contains_key(v),
                    AtomTerm::Value(_) => true,
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
                    AtomTerm::Value(_) => true,
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

        // sanity check the program
        let mut tuple_valid = vec![false; query.vars.len()];
        for instr in &program {
            match instr {
                Instr::Intersect { value_idx, .. } => {
                    assert!(!tuple_valid[*value_idx]);
                    tuple_valid[*value_idx] = true;
                }
                Instr::ConstrainConstant { .. } => {}
                Instr::Call { check, args, .. } => {
                    let Some((last, args)) = args.split_last() else {
                        continue
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
                        AtomTerm::Value(_) => {
                            assert!(*check);
                        }
                        AtomTerm::Global(_) => {
                            assert!(*check);
                        }
                    }
                }
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
                    let start = Instant::now();
                    log::debug!(
                        "Query:\n{q}\nNew atom: {atom}\nTuple: {tuple}\nJoin order: {order}\nProgram\n{program}",
                        q = cq.query,
                        order = ListDisplay(&ctx.join_var_ordering, " "),
                        tuple = ListDisplay(cq.vars.keys(), " "),
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

                if !do_seminaive {
                    break;
                }

                // now we can fix this atom to be "old stuff" only
                // range is half-open; timestamp is excluded
                timestamp_ranges[atom_i] = 0..timestamp;
            }
        } else if let Some((mut ctx, program, _)) = Context::new(self, cq, &[]) {
            let mut meausrements = HashMap::<usize, Vec<usize>>::default();
            let stages = InputSizes {
                stage_sizes: &mut meausrements,
                cur_stage: 0,
            };
            let tries = LazyTrie::make_initial_vec(cq.query.atoms.len());
            let mut trie_refs = tries.iter().collect::<Vec<_>>();
            ctx.eval(&mut trie_refs, &program.0, stages, &mut f)
                .unwrap_or(());
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

#[derive(Clone)]
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
