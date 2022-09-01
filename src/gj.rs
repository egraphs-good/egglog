use bumpalo::Bump;
use indexmap::map::Entry;

use crate::{
    typecheck::{Atom, AtomTerm},
    *,
};
use std::fmt::Debug;

#[derive(Debug, Clone)]
enum Instr {
    Intersect {
        idx: usize,
        trie_indices: Vec<usize>,
    },
    Call {
        prim: Primitive,
        args: Vec<AtomTerm>,
        check: bool, // check or assign to output variable
    },
}

struct TrieRequest {
    sym: Symbol,
    projection: Vec<usize>,
    constraints: Vec<Constraint>,
}

struct Context<'b> {
    bump: &'b Bump,
    query: &'b CompiledQuery,
    egraph: &'b EGraph,
    tries: Vec<&'b Trie<'b>>,
    tuple: Vec<Value>,
    empty: &'b Trie<'b>,
    val_pool: Vec<Vec<Value>>,
    trie_pool: Vec<Vec<&'b Trie<'b>>>,
}

impl<'b> Context<'b> {
    fn new(bump: &'b Bump, egraph: &'b EGraph, query: &'b CompiledQuery) -> Self {
        let default_trie = bump.alloc(Trie::default());
        let mut ctx = Context {
            egraph,
            query,
            bump,
            tuple: vec![Value::fake(); query.vars.len()],
            tries: vec![default_trie; query.atoms.len()],
            empty: bump.alloc(Trie::default()),
            val_pool: Default::default(),
            trie_pool: Default::default(),
        };

        for (atom_i, atom) in query.atoms.iter().enumerate() {
            // let mut to_project = vec![];
            let mut constraints = vec![];
            let (sym, args) = match atom {
                Atom::Func(sym, args) => (*sym, args),
                Atom::Prim(_, _) => continue,
            };

            for (i, t) in args.iter().enumerate() {
                match t {
                    AtomTerm::Value(val) => constraints.push(Constraint::Const(i, *val)),
                    AtomTerm::Var(_v) => {
                        if let Some(j) = args[..i].iter().position(|t2| t == t2) {
                            constraints.push(Constraint::Eq(j, i));
                        } else {
                            // to_project.push(v)
                        }
                    }
                }
            }

            let mut projection = vec![];
            for v in query.vars.keys() {
                if let Some(i) = args.iter().position(|t| t == &AtomTerm::Var(*v)) {
                    assert!(!projection.contains(&i));
                    projection.push(i);
                }
            }

            ctx.tries[atom_i] = ctx.build_trie(&TrieRequest {
                sym,
                projection,
                constraints,
            });
        }

        ctx
    }

    fn eval<F>(&mut self, program: &[Instr], f: &mut F)
    where
        F: FnMut(&[Value]),
    {
        let (instr, program) = match program.split_first() {
            None => return f(&self.tuple),
            Some(pair) => pair,
        };

        match instr {
            Instr::Intersect { idx, trie_indices } => {
                match trie_indices.len() {
                    1 => {
                        let j = trie_indices[0];
                        let r = self.tries[j];
                        for (val, trie) in r.0.iter() {
                            self.tuple[*idx] = *val;
                            self.tries[j] = trie;
                            self.eval(program, f);
                        }
                        self.tries[j] = r;
                    }
                    2 => {
                        let rs = [self.tries[trie_indices[0]], self.tries[trie_indices[1]]];
                        // smaller_idx
                        let si = rs[0].len() > rs[1].len();
                        let intersection = rs[si as usize]
                            .0
                            .keys()
                            .filter(|k| rs[(!si) as usize].0.contains_key(k));
                        for val in intersection {
                            self.tuple[*idx] = *val;
                            self.tries[trie_indices[0]] = rs[0].0.get(val).unwrap();
                            self.tries[trie_indices[1]] = rs[1].0.get(val).unwrap();

                            self.eval(program, f);
                        }
                        self.tries[trie_indices[0]] = rs[0];
                        self.tries[trie_indices[1]] = rs[1];
                    }
                    _ => {
                        // the index of the smallest trie
                        let j_min = trie_indices
                            .iter()
                            .copied()
                            .min_by_key(|j| self.tries[*j].len())
                            .unwrap();
                        let mut intersection = self.val_pool.pop().unwrap_or_default();
                        intersection.extend(self.tries[j_min].0.keys().cloned());

                        for &j in trie_indices {
                            if j != j_min {
                                let r = &self.tries[j].0;
                                intersection.retain(|t| r.contains_key(t));
                            }
                        }
                        let mut rs = self.trie_pool.pop().unwrap_or_default();
                        rs.extend(trie_indices.iter().map(|&j| self.tries[j]));

                        for val in intersection.drain(..) {
                            self.tuple[*idx] = val;

                            for (r, &j) in rs.iter().zip(trie_indices) {
                                self.tries[j] = match r.0.get(&val) {
                                    Some(t) => *t,
                                    None => self.empty,
                                }
                            }

                            self.eval(program, f);
                        }
                        self.val_pool.push(intersection);

                        // TODO is it necessary to reset the tries?
                        for (r, &j) in rs.iter().zip(trie_indices) {
                            self.tries[j] = r;
                        }
                        rs.clear();
                        self.trie_pool.push(rs);
                    }
                };
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
                                return;
                            }
                            self.tuple[i] = res;
                        }
                        AtomTerm::Value(val) => {
                            assert!(check);
                            if val != &res {
                                return;
                            }
                        }
                    }
                    self.eval(program, f);
                }
            }
        }
    }

    fn build_trie(&self, req: &TrieRequest) -> &'b Trie<'b> {
        let mut trie = Trie::default();
        if req.constraints.is_empty() {
            self.egraph.for_each_canonicalized(req.sym, |tuple| {
                trie.insert(self.bump, &req.projection, tuple);
            });
        } else {
            self.egraph.for_each_canonicalized(req.sym, |tuple| {
                for constraint in &req.constraints {
                    let ok = match constraint {
                        Constraint::Eq(i, j) => tuple[*i] == tuple[*j],
                        Constraint::Const(i, t) => &tuple[*i] == t,
                    };
                    if ok {
                        trie.insert(self.bump, &req.projection, tuple);
                    }
                }
            });
        }
        self.bump.alloc(trie)
    }
}

enum Constraint {
    Eq(usize, usize),
    Const(usize, Value),
}

#[derive(Debug, Default)]
struct Trie<'b>(HashMap<Value, &'b mut Self>);

impl Trie<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'b> Trie<'b> {
    fn insert(&mut self, bump: &'b Bump, shuffle: &[usize], tuple: &[Value]) {
        // debug_assert_eq!(shuffle.len(), tuple.len());
        debug_assert!(shuffle.len() <= tuple.len());
        let mut trie = self;
        for i in shuffle {
            trie = trie
                .0
                .entry(tuple[*i])
                .or_insert_with(|| bump.alloc(Trie::default()));
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
    atoms: Vec<Atom>,
    pub vars: IndexMap<Symbol, VarInfo>,
    // extra: Vec<PrimCall>,
    program: Vec<Instr>,
}

#[derive(Debug, Clone)]
struct PrimCall {
    prim: Primitive,
    args: Vec<AtomTerm>,
}

impl EGraph {
    pub(crate) fn compile_gj_query(
        &self,
        atoms: Vec<Atom>,
        _types: HashMap<Symbol, ArcSort>,
    ) -> CompiledQuery {
        let mut extra = vec![];
        let mut vars: IndexMap<Symbol, VarInfo> = Default::default();
        for (i, atom) in atoms.iter().enumerate() {
            match atom {
                Atom::Func(_, _) => {
                    for v in atom.vars() {
                        // only count grounded occurrences
                        vars.entry(v).or_default().occurences.push(i)
                    }
                }
                Atom::Prim(p, args) => {
                    extra.push(PrimCall {
                        prim: p.clone(),
                        args: args.clone(),
                    });
                }
            }
        }

        // right now, vars should only contain grounded variables
        for (v, info) in &mut vars {
            debug_assert!(info.occurences.windows(2).all(|w| w[0] <= w[1]));
            info.occurences.dedup();
            assert!(!info.occurences.is_empty(), "var {} has no occurences", v);
        }

        vars.sort_by(|_v1, i1, _v2, i2| i2.occurences.len().cmp(&i1.occurences.len()));

        let mut program: Vec<Instr> = vars
            .iter()
            .enumerate()
            .map(|(idx, (_v, info))| Instr::Intersect {
                idx,
                trie_indices: info.occurences.clone(),
            })
            .collect();

        // now we can try to add primitives
        // TODO this is very inefficient, since primitives all at the end
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
                    prim: p.prim.clone(),
                    args: p.args.clone(),
                    check,
                });
            } else {
                panic!("cycle")
            }
        }

        log::debug!("vars: [{}]", ListDisplay(vars.keys(), ", "));

        CompiledQuery {
            atoms,
            vars,
            program,
        }
    }

    pub(crate) fn run_query<F>(&self, query: &CompiledQuery, mut f: F)
    where
        F: FnMut(&[Value]),
    {
        let bump = Bump::new();
        let mut ctx = Context::new(&bump, self, query);
        ctx.eval(&query.program, &mut f)
    }
}
