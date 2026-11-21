//! Recursive variant extraction for acyclic constructor e-classes.

use super::*;
use std::{cmp::Reverse, collections::BinaryHeap};

type Key = (String, Value);
type Answers = Vec<ExtractedTerm<DefaultCost>>;

struct Production<'a> {
    name: &'a str,
    cost: DefaultCost,
    children: Vec<Key>,
}

enum Visit<'a> {
    Enter(ArcSort, Value),
    Finish(Key, Vec<Production<'a>>),
}

impl EGraph {
    /// Returns up to `k` distinct syntax trees per root, in ascending default
    /// tree cost. For equal costs the tie order is constructor name, stored
    /// child values, then the lexicographic vector of child ranks (zero-based).
    /// It is stable for a fixed graph. Each child occurrence chooses independently.
    ///
    /// Supports ordinary constructors and base values. Reachable constructor
    /// cycles, containers and the term/proof encoding return an extraction error.
    /// Subsumed, hidden and unextractable constructors do not participate.
    pub fn extract_ranked_variants(
        &self,
        roots: Vec<(ArcSort, Value)>,
        k: usize,
    ) -> Result<ExtractedTermVariants<DefaultCost>, Error> {
        let mut termdag = TermDag::default();
        if k == 0 || roots.is_empty() {
            return Ok(ExtractedTermVariants {
                termdag,
                variants: vec![vec![]; roots.len()],
            });
        }
        if self.proof_state.original_typechecking.is_some() {
            return Err(Error::ExtractError(
                "ranked variants require an e-graph without term/proof encoding".into(),
            ));
        }

        let mut constructors: HashMap<&str, Vec<&Function>> = Default::default();
        for func in self.functions.values() {
            if func.func_type.subtype == FunctionSubtype::Constructor
                && !func.is_hidden()
                && !func.is_unextractable()
            {
                constructors
                    .entry(func.func_type.output.name())
                    .or_default()
                    .push(func);
            }
        }

        // An explicit DFS stack avoids recursion on the depth of the input.
        let mut answers: HashMap<Key, Answers> = Default::default();
        let mut active = HashSet::default();
        let mut stack: Vec<_> = roots
            .iter()
            .map(|(s, v)| Visit::Enter(s.clone(), *v))
            .collect();
        self.read(|state| -> Result<(), Error> {
            while let Some(visit) = stack.pop() {
                match visit {
                    Visit::Enter(sort, value) => {
                        let key = (sort.name().to_owned(), value);
                        if answers.contains_key(&key) {
                            continue;
                        }
                        if sort.is_container_sort() {
                            return Err(Error::ExtractError(format!(
                                "ranked variants do not support container sort {}",
                                sort.name()
                            )));
                        }
                        if !sort.is_eq_sort() {
                            let term = self.reconstruct_base_value(&sort, value, &mut termdag);
                            let cost = TreeCostModel::base_value_cost(
                                &DEFAULT_COST_MODEL,
                                self,
                                &sort,
                                value,
                            );
                            answers.insert(key, vec![ExtractedTerm { cost, term }]);
                            continue;
                        }
                        if !active.insert(key.clone()) {
                            return Err(Error::ExtractError(format!(
                                "ranked variants reached a cycle in sort {}",
                                sort.name()
                            )));
                        }
                        let mut next = Vec::new();
                        let mut rows = Vec::new();
                        for &func in constructors.get(sort.name()).into_iter().flatten() {
                            let name = func.extraction_term_name();
                            state.constructor_enodes_for_eclass(name, value, |enode| {
                                if !enode.subsumed {
                                    let children = func
                                        .func_type
                                        .input
                                        .iter()
                                        .zip(enode.children)
                                        .map(|(s, v)| {
                                            next.push(Visit::Enter(s.clone(), *v));
                                            (s.name().to_owned(), *v)
                                        })
                                        .collect();
                                    let cost = TreeCostModel::enode_cost(
                                        &DEFAULT_COST_MODEL,
                                        self,
                                        func,
                                        &enode,
                                    );
                                    rows.push(Production {
                                        name,
                                        cost,
                                        children,
                                    });
                                }
                            })?;
                        }
                        rows.sort_unstable_by(|a, b| {
                            a.name.cmp(b.name).then_with(|| a.children.cmp(&b.children))
                        });
                        stack.push(Visit::Finish(key, rows));
                        stack.extend(next);
                    }
                    Visit::Finish(key, rows) => {
                        let ranked = rank_class(&rows, &answers, k, &mut termdag);
                        active.remove(&key);
                        answers.insert(key, ranked);
                    }
                }
            }
            Ok(())
        })?;
        let variants = roots
            .into_iter()
            .map(|(s, v)| answers[&(s.name().to_owned(), v)].clone())
            .collect();
        Ok(ExtractedTermVariants { termdag, variants })
    }
}

// Merge the sorted Cartesian products of the constructor rows. A tuple's
// unique predecessor decrements its first nonzero coordinate. Each heap key
// increases along this edge even when additive cost saturates or ties.
fn rank_class(
    rows: &[Production<'_>],
    answers: &HashMap<Key, Answers>,
    k: usize,
    dag: &mut TermDag,
) -> Answers {
    let children: Vec<Vec<&Answers>> = rows
        .iter()
        .map(|r| r.children.iter().map(|key| &answers[key]).collect())
        .collect();
    let score = |row: usize, ranks: &[usize]| {
        children[row]
            .iter()
            .zip(ranks)
            .fold(rows[row].cost, |cost, (list, &rank)| {
                cost.saturating_add(list[rank].cost)
            })
    };
    let mut heap = BinaryHeap::new();
    for (i, lists) in children.iter().enumerate() {
        if lists.iter().all(|list| !list.is_empty()) {
            let ranks = vec![0; lists.len()];
            heap.push(Reverse((score(i, &ranks), i, ranks)));
        }
    }
    let mut seen = HashSet::default();
    let mut result = Vec::new();
    while let Some(Reverse((cost, i, ranks))) = heap.pop() {
        let lists = &children[i];
        let args = lists
            .iter()
            .zip(&ranks)
            .map(|(list, &r)| list[r].term)
            .collect();
        let term = dag.app(rows[i].name.to_owned(), args);
        if seen.insert(term) {
            result.push(ExtractedTerm { cost, term });
            if result.len() == k {
                break;
            }
        }
        // Each retained child list has at most k distinct terms. Any omitted
        // rank has k earlier replacements in this same argument position,
        // giving k distinct earlier parent trees even if other rows overlap.
        for (d, &r) in ranks.iter().enumerate() {
            if r + 1 < lists[d].len() {
                let mut next = ranks.clone();
                next[d] += 1;
                heap.push(Reverse((score(i, &next), i, next)));
            }
            if r != 0 {
                break;
            }
        }
    }
    result
}
