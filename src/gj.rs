use bumpalo::Bump;

use crate::{
    typecheck::{Atom, AtomTerm},
    *,
};
use std::fmt::Debug;

enum Instr {
    // write
    Intersect {
        idx: usize,
        trie_indices: Vec<usize>,
    },
}

struct TrieRequest {
    sym: Symbol,
    projection: Vec<usize>,
    constraints: Vec<Constraint>,
}

struct Context<'b> {
    bump: &'b Bump,
    egraph: &'b EGraph,
    tries: Vec<&'b Trie<'b>>,
    tuple: Vec<Value>,
    empty: &'b Trie<'b>,
}

impl<'b> Context<'b> {
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
                // debug_assert!(js
                //     .iter()
                //     .all(|&j| query.atoms[j].1.contains(&AtomTerm::Var(x))));

                // the index of the smallest trie
                let j_min = trie_indices
                    .iter()
                    .copied()
                    .min_by_key(|j| self.tries[*j].len())
                    .unwrap();

                // TODO reuse this allocation
                let mut intersection: Vec<Value> = self.tries[j_min].0.keys().cloned().collect();

                for &j in trie_indices {
                    if j != j_min {
                        let r = &self.tries[j].0;
                        intersection.retain(|t| r.contains_key(t));
                    }
                }

                let rs: Vec<&'b Trie> = trie_indices.iter().map(|&j| self.tries[j]).collect();

                for val in intersection {
                    self.tuple[*idx] = val.clone();

                    for (r, &j) in rs.iter().zip(trie_indices) {
                        self.tries[j] = match r.0.get(&val) {
                            Some(t) => *t,
                            None => self.empty,
                        }
                    }

                    self.eval(program, f);
                }

                // TODO is it necessary to reset the tries?
                for (r, &j) in rs.iter().zip(trie_indices) {
                    self.tries[j] = r;
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
                .entry(tuple[*i].clone())
                .or_insert_with(|| bump.alloc(Trie::default()));
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct VarInfo {
    occurences: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct CompiledQuery {
    atoms: Vec<Atom>,
    pub vars: IndexMap<Symbol, VarInfo>,
}

impl EGraph {
    pub(crate) fn compile_gj_query(
        &self,
        atoms: Vec<Atom>,
        _types: HashMap<Symbol, Type>,
    ) -> CompiledQuery {
        let mut vars = IndexMap::<Symbol, VarInfo>::default();
        for (i, atom) in atoms.iter().enumerate() {
            for v in atom.vars() {
                vars.entry(v).or_default().occurences.push(i)
            }
        }

        for (v, info) in &mut vars {
            debug_assert!(info.occurences.windows(2).all(|w| w[0] <= w[1]));
            info.occurences.dedup();
            assert!(!info.occurences.is_empty(), "var {} has no occurences", v)
        }

        vars.sort_by(|_v1, i1, _v2, i2| i1.occurences.len().cmp(&i2.occurences.len()));

        CompiledQuery { atoms, vars }
    }

    pub(crate) fn run_query<F>(&self, query: &CompiledQuery, mut f: F)
    where
        F: FnMut(&[Value]),
    {
        let bump = Bump::new();
        let mut ctx = Context {
            egraph: self,
            bump: &bump,
            tuple: vec![Value::fake(); query.vars.len()],
            tries: vec![],
            empty: bump.alloc(Trie::default()),
        };

        ctx.tries = query
            .atoms
            .iter()
            .map(|atom| {
                // let mut to_project = vec![];
                let mut constraints = vec![];
                let (sym, args) = match atom {
                    Atom::Func(sym, args) => (*sym, args),
                    Atom::Prim(_, _) => todo!(),
                };

                for (i, t) in args.iter().enumerate() {
                    match t {
                        AtomTerm::Value(val) => constraints.push(Constraint::Const(i, val.clone())),
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

                ctx.build_trie(&TrieRequest {
                    sym,
                    projection,
                    constraints,
                })
            })
            .collect::<Vec<_>>();

        let mut program = vec![];
        // let mut atom_positions: Vec<usize> = (0..query.atoms.len()).collect();
        // let mut next_position = atom_positions.len();

        for (idx, info) in query.vars.values().enumerate() {
            program.push(Instr::Intersect {
                idx,
                trie_indices: info.occurences.clone(),
            })
            // let placed_occs: Vec<usize> = occs
            //     .iter()
            //     .map(|&i| {
            //         let placed = atom_positions[i];
            //         atom_positions[i] = occs.len();
            //         placed
            //     })
            //     .collect();
            // program.push(Instr::Intersect(placed_occs));
        }

        ctx.eval(&program, &mut f)
    }
}
