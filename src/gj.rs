use crate::*;
use std::fmt::Debug;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Atom(pub Symbol, pub Vec<AtomTerm>);

impl Atom {
    fn vars(&self) -> impl Iterator<Item = IndexVar> + '_ {
        self.1.iter().filter_map(|t| match t {
            AtomTerm::Var(v) => Some(*v),
            AtomTerm::Value(_) => None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AtomTerm {
    Var(IndexVar),
    Value(Value),
}

enum Constraint {
    Eq(usize, usize),
    Const(usize, Value),
}

pub type IndexVar = usize;

#[derive(Debug, Clone, Default)]
struct Trie(HashMap<Value, Self>);

impl Trie {
    fn insert(&mut self, shuffle: &[usize], tuple: &[Value]) {
        // debug_assert_eq!(shuffle.len(), tuple.len());
        debug_assert!(shuffle.len() <= tuple.len());
        let mut trie = self;
        for i in shuffle {
            trie = trie.0.entry(tuple[*i].clone()).or_default();
        }
    }
}

// for each var, says which atoms contain it
type VarOccurences = Vec<Vec<usize>>;

pub struct CompiledQuery {
    atoms: Vec<Atom>,
    var_order: Vec<IndexVar>,
    occurences: VarOccurences,
    buffer_size: usize,
}

impl EGraph {
    pub(crate) fn compile_gj_query(&self, atoms: &[Atom]) -> CompiledQuery {
        let n_vars = atoms
            .iter()
            .flat_map(|a| a.vars())
            .max()
            .map_or(0, |v| v + 1);
        let mut occurences = vec![vec![]; n_vars];
        for (i, atom) in atoms.iter().enumerate() {
            for v in atom.vars() {
                if occurences[v].last().copied() != Some(i) {
                    occurences[v].push(i);
                }
            }
        }

        // for (v, occs) in occurences.iter().enumerate() {
        //     assert!(!occs.is_empty(), "var {} has no occurences", v)
        // }

        let mut var_order: Vec<IndexVar> = (0..n_vars)
            .filter(|i| !occurences[*i].is_empty())
            .collect();

        // simple variable ordering for now
        var_order.sort_unstable_by_key(|&v| -(occurences[v].len() as i32));

        CompiledQuery {
            atoms: atoms.into(),
            var_order,
            occurences,
            buffer_size: n_vars,
        }
    }

    fn build_trie(
        &self,
        relation: Symbol,
        projection: &[usize],
        constraints: &[Constraint],
    ) -> Trie {
        let mut trie = Trie::default();
        if constraints.is_empty() {
            self.for_each_canonicalized(relation, |tuple| {
                trie.insert(projection, tuple);
            });
        } else {
            self.for_each_canonicalized(relation, |tuple| {
                for constraint in constraints {
                    let ok = match constraint {
                        Constraint::Eq(i, j) => tuple[*i] == tuple[*j],
                        Constraint::Const(i, t) => &tuple[*i] == t,
                    };
                    if ok {
                        trie.insert(projection, tuple);
                    }
                }
            });
        }
        trie
    }

    pub(crate) fn run_query<F>(&self, query: &CompiledQuery, mut f: F)
    where
        F: FnMut(&[Value]),
    {
        log::debug!("Eval {:?}", query.atoms);
        let tries = query
            .atoms
            .iter()
            .map(|atom| {
                let mut to_project = vec![];
                let mut constraints = vec![];
                for (i, t) in atom.1.iter().enumerate() {
                    match t {
                        AtomTerm::Value(val) => constraints.push(Constraint::Const(i, val.clone())),
                        AtomTerm::Var(v) => {
                            if let Some(j) = atom.1[..i].iter().position(|t2| t == t2) {
                                constraints.push(Constraint::Eq(j, i));
                            } else {
                                to_project.push(v)
                            }
                        }
                    }
                }

                let mut projection = vec![];
                for v in &query.var_order {
                    if let Some(i) = atom.1.iter().position(|t| t == &AtomTerm::Var(*v)) {
                        assert!(!projection.contains(&i));
                        projection.push(i);
                    }
                }

                self.build_trie(atom.0, &projection, &constraints)
            })
            .collect::<Vec<_>>();

        let tries: Vec<&Trie> = tries.iter().collect();

        let tuple = vec![Value::fake(); query.buffer_size];
        self.gj(0, query, &mut f, &tuple, &tries);
    }

    fn gj<F>(
        &self,
        depth: usize,
        query: &CompiledQuery,
        f: &mut F,
        tuple: &[Value],
        relations: &[&Trie],
    ) where
        F: FnMut(&[Value]),
    {
        // log::debug!("{:?}", tuple);
        if depth == query.var_order.len() {
            return f(tuple);
        }

        let x = query.var_order[depth];
        let js = &query.occurences[x];

        debug_assert!(js
            .iter()
            .all(|&j| query.atoms[j].1.contains(&AtomTerm::Var(x))));

        let j_min = js
            .iter()
            .copied()
            .min_by_key(|j| relations[*j].len())
            .unwrap();

        // for &j in js {
        //     log::debug!("{:?}", relations[j].0.keys());
        // }

        let mut intersection: Vec<Value> = relations[j_min].0.keys().cloned().collect();

        for &j in js {
            if j != j_min {
                let rj = &relations[j].0;
                intersection.retain(|t| rj.contains_key(t));
            }
        }

        // log::debug!("intersection of {:?}: {:?}", x, intersection);

        let empty = Trie::default();

        let mut tuple = tuple.to_vec();
        for val in intersection {
            let relations: Vec<_> = relations
                .iter()
                .zip(&query.atoms)
                .map(|(r, a)| {
                    if a.1.contains(&AtomTerm::Var(x)) {
                        r.0.get(&val).unwrap_or(&empty)
                    } else {
                        r
                    }
                })
                .collect();
            tuple[x] = val;
            self.gj(depth + 1, query, f, &tuple, &relations);
        }
    }
}
