use crate::{Database, Error, Function, FunctionKind, HashMap};
use serde::Serialize;
use smallvec::SmallVec;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Comparison {
    /// Constructor e-classes have the same sets of bisimulation behaviors.
    pub terms_equal: bool,
    /// Additionally, declarations and the full database coalgebra agree.
    pub database_equal: bool,
    /// Rounds in constructor-only refinement.
    pub refinement_rounds: usize,
    /// Rounds in full-database refinement (reused when observations coincide).
    pub database_refinement_rounds: usize,
}

// Intern complete labels jointly across the inputs. Neither names nor schemas
// are copied per node or per refinement round; hash collisions still use Eq.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub(crate) enum Label<'a> {
    Literal(&'a str),
    Call(&'a str, &'a Function, bool),
}

fn intern<'a>(labels: &mut HashMap<Label<'a>, usize>, label: Label<'a>) -> usize {
    let next = labels.len();
    *labels.entry(label).or_insert(next)
}

#[derive(Clone)]
pub(crate) struct Node {
    pub symbol: usize,
    pub children: SmallVec<[usize; 2]>,
}

pub(crate) struct Graph<'a> {
    pub sorts: Vec<&'a str>,
    pub nodes: Vec<Vec<Node>>,
    /// Sorted, unique output IDs. Mark once per row instead of tree insertion.
    pub roots: Vec<usize>,
    pub index: HashMap<&'a str, usize>,
}

impl<'a> Graph<'a> {
    pub(crate) fn new(
        db: &'a Database,
        offset: usize,
        include_functions: bool,
        labels: &mut HashMap<Label<'a>, usize>,
    ) -> Self {
        let index: HashMap<_, _> = db
            .classes
            .keys()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i + offset))
            .collect();
        let sorts = db.classes.values().map(|c| c.sort.as_str()).collect();
        let mut nodes = vec![Vec::new(); db.classes.len()];
        for (id, class) in &db.classes {
            if let Some(literal) = &class.literal {
                nodes[index[id.as_str()] - offset].push(Node {
                    symbol: intern(labels, Label::Literal(literal)),
                    children: SmallVec::new(),
                });
            }
        }
        let operators: HashMap<_, _> = db
            .functions
            .iter()
            .filter(|(_, f)| include_functions || f.kind == FunctionKind::Constructor)
            .map(|(name, function)| {
                (
                    name.as_str(),
                    [
                        intern(labels, Label::Call(name, function, false)),
                        intern(labels, Label::Call(name, function, include_functions)),
                    ],
                )
            })
            .collect();
        let mut roots = vec![false; db.classes.len()];
        for row in &db.rows {
            if let Some(symbols) = operators.get(row.function.as_str()) {
                let output = index[row.output.as_str()] - offset;
                roots[output] = true;
                nodes[output].push(Node {
                    symbol: symbols[usize::from(row.subsumed)],
                    children: row.inputs.iter().map(|id| index[id.as_str()]).collect(),
                });
            }
        }
        Self {
            sorts,
            nodes,
            roots: roots
                .into_iter()
                .enumerate()
                .filter_map(|(i, root)| root.then_some(i + offset))
                .collect(),
            index,
        }
    }
}

pub(crate) struct Partition<'a> {
    pub left: Graph<'a>,
    pub right: Graph<'a>,
    pub blocks: Vec<usize>,
    pub rounds: usize,
}

impl<'a> Partition<'a> {
    pub fn new(left: &'a Database, right: &'a Database) -> Self {
        Self::with_functions(left, right, false)
    }

    pub fn database(left: &'a Database, right: &'a Database) -> Self {
        Self::with_functions(left, right, true)
    }

    fn with_functions(left: &'a Database, right: &'a Database, include_functions: bool) -> Self {
        let mut labels = HashMap::default();
        let left = Graph::new(left, 0, include_functions, &mut labels);
        let right = Graph::new(right, left.nodes.len(), include_functions, &mut labels);
        let mut sorts = HashMap::default();
        let blocks = left
            .sorts
            .iter()
            .chain(&right.sorts)
            .map(|&sort| {
                let next = sorts.len();
                *sorts.entry(sort).or_insert(next)
            })
            .collect();
        Self {
            left,
            right,
            blocks,
            rounds: 0,
        }
    }

    pub fn step(&mut self) -> bool {
        let mut signatures = HashMap::default();
        let mut next_blocks = Vec::with_capacity(self.blocks.len());
        // A node signature is its interned label followed by child blocks.
        // Unary/binary operators stay inline; larger arities spill safely.
        let mut key: (usize, Vec<SmallVec<[usize; 3]>>) = (0, Vec::new());
        for (i, nodes) in self.left.nodes.iter().chain(&self.right.nodes).enumerate() {
            key.0 = self.blocks[i]; // Retain old block: split, never merge.
            key.1.clear();
            key.1.extend(nodes.iter().map(|node| {
                let mut signature = SmallVec::new();
                signature.push(node.symbol);
                signature.extend(node.children.iter().map(|&child| self.blocks[child]));
                signature
            }));
            key.1.sort_unstable();
            key.1.dedup();
            if let Some(&block) = signatures.get(&key) {
                next_blocks.push(block);
            } else {
                let block = signatures.len();
                // Reuse scratch storage when the signature was already seen.
                // Store full signatures, not probabilistic fingerprints.
                signatures.insert((key.0, std::mem::take(&mut key.1)), block);
                next_blocks.push(block);
            }
        }
        self.rounds += 1;
        let changed = next_blocks != self.blocks;
        self.blocks = next_blocks;
        changed
    }

    pub fn finish(&mut self) {
        while self.step() {}
    }

    pub fn root_blocks(&self, graph: &Graph<'_>) -> BTreeSet<usize> {
        graph.roots.iter().map(|&id| self.blocks[id]).collect()
    }

    pub fn terms_equal(&self) -> bool {
        let mut coverage = vec![0u8; self.blocks.len()];
        for &root in &self.left.roots {
            coverage[self.blocks[root]] |= 1;
        }
        for &root in &self.right.roots {
            coverage[self.blocks[root]] |= 2;
        }
        coverage.iter().all(|&sides| sides == 0 || sides == 3)
    }
}

/// Exact refinement: hash collisions use full equality; there is no depth cutoff.
///
/// Each round scans all nodes/edges and sorts each class's signatures. There
/// are at most O(classes) splitting rounds, so worst-case time remains quadratic
/// (plus signature sorting). Names, schemas, and IDs are resolved only at setup.
/// Ordinary functions never participate in constructor-term syntax.
pub fn compare(left: &Database, right: &Database) -> Result<Comparison, Error> {
    left.validate()?;
    right.validate()?;
    let mut partition = Partition::new(left, right);
    partition.finish();
    let terms_equal = partition.terms_equal();
    let refinement_rounds = partition.rounds;
    let same_observations = [left, right].iter().all(|db| {
        db.functions
            .values()
            .all(|f| f.kind == FunctionKind::Constructor)
            && db.rows.iter().all(|row| !row.subsumed)
    });
    if same_observations {
        return Ok(Comparison {
            terms_equal,
            database_equal: terms_equal && left.functions == right.functions,
            refinement_rounds,
            database_refinement_rounds: refinement_rounds,
        });
    }
    // Release the first graph before constructing the full-database graph.
    drop(partition);
    let mut partition = Partition::database(left, right);
    partition.finish();
    // Every row is a member of an output class. Matching output blocks at the
    // fixed point already implies matching rows modulo the child/output blocks;
    // rebuilding a second set of string-keyed row signatures is redundant.
    let database_equal =
        terms_equal && left.functions == right.functions && partition.terms_equal();
    Ok(Comparison {
        terms_equal,
        database_equal,
        refinement_rounds,
        database_refinement_rounds: partition.rounds,
    })
}
