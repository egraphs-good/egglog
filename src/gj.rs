use crate::*;
use std::fmt::Debug;

#[derive(Debug, Clone, Default)]
struct Trie(HashMap<Id, Self>);

impl Trie {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl Trie {
    fn insert(&mut self, shuffle: &[usize], tuple: &[Id]) {
        debug_assert_eq!(shuffle.len(), tuple.len());
        let mut trie = self;
        for i in shuffle {
            trie = trie.0.entry(tuple[*i]).or_default();
        }
    }
}

// for each var, says which atoms contain it
type VarOccurences = Vec<Vec<usize>>;

pub struct CompiledQuery<T> {
    atoms: Vec<Atom<T>>,
    var_order: Vec<IndexVar>,
    occurences: VarOccurences,
}

impl<T: Operator> EGraph<T> {
    pub(crate) fn compile_query(&self, atoms: &[Atom<T>]) -> CompiledQuery<T> {
        let n_vars = atoms
            .iter()
            .flat_map(|a| &a.vars)
            .max()
            .map_or(0, |v| v + 1);
        let mut occurences = vec![vec![]; n_vars];
        for (i, atom) in atoms.iter().enumerate() {
            for &v in &atom.vars {
                if occurences[v].last().copied() != Some(i) {
                    occurences[v].push(i);
                }
            }
        }

        for (v, occs) in occurences.iter().enumerate() {
            assert!(!occs.is_empty(), "var {} has no occurences", v)
        }

        let mut var_order: Vec<IndexVar> = (0..n_vars).collect();

        // simple variable ordering for now
        var_order.sort_unstable_by_key(|&v| -(occurences[v].len() as i32));

        CompiledQuery {
            atoms: atoms.into(),
            var_order,
            occurences,
        }
    }

    pub(crate) fn eval<F>(&self, query: &CompiledQuery<T>, mut f: F)
    where
        F: FnMut(&[Id]),
        T: Debug,
    {
        println!("Eval {:?}", query.atoms);
        let tries = query
            .atoms
            .iter()
            .map(|atom| {
                let mut eq_constraints = vec![];
                for (i, v) in atom.vars.iter().enumerate() {
                    if let Some(j) = atom.vars[..i].iter().position(|v2| v == v2) {
                        eq_constraints.push((j, i));
                    }
                }

                let mut shuffle = vec![];
                for v in &query.var_order {
                    if let Some(i) = atom.vars.iter().position(|v2| v == v2) {
                        if !shuffle.contains(&i) {
                            shuffle.push(i);
                        }
                    }
                }

                assert!(eq_constraints.iter().all(|(i, j)| i < j));

                let mut trie = Trie::default();
                self.for_each_canonicalized(&atom.op, |tuple| {
                    if eq_constraints.iter().all(|(i, j)| tuple[*i] == tuple[*j]) {
                        trie.insert(&shuffle, tuple);
                    }
                });

                trie
            })
            .collect::<Vec<_>>();

        let tries: Vec<&Trie> = tries.iter().collect();

        let tuple = vec![Id::fake(); query.var_order.len()];
        self.gj(0, query, &mut f, &tuple, &tries);
    }

    fn gj<F>(
        &self,
        depth: usize,
        query: &CompiledQuery<T>,
        f: &mut F,
        tuple: &[Id],
        relations: &[&Trie],
    ) where
        F: FnMut(&[Id]),
    {
        // println!("{:?}", tuple);
        if depth == query.var_order.len() {
            return f(tuple);
        }

        let x = query.var_order[depth];
        let js = &query.occurences[x];

        debug_assert!(js.iter().all(|&j| query.atoms[j].vars.contains(&x)));

        let j_min = js
            .iter()
            .copied()
            .min_by_key(|j| relations[*j].len())
            .unwrap();

        // for &j in js {
        //     println!("{:?}", relations[j].0.keys());
        // }

        let mut intersection: Vec<Id> = relations[j_min].0.keys().cloned().collect();

        for &j in js {
            if j != j_min {
                let rj = &relations[j].0;
                intersection.retain(|t| rj.contains_key(t));
            }
        }

        // println!("intersection of {:?}: {:?}", x, intersection);

        let empty = Trie::default();

        let mut tuple = tuple.to_vec();
        for val in intersection {
            let relations: Vec<_> = relations
                .iter()
                .zip(&query.atoms)
                .map(|(r, a)| {
                    if a.vars.contains(&x) {
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
