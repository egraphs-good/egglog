use crate::{Database, Error, Function, FunctionKind};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Comparison {
    /// Constructor e-classes have the same sets of bisimulation behaviors.
    pub terms_equal: bool,
    /// Additionally, declarations and all rows agree modulo those behaviors.
    pub database_equal: bool,
    pub refinement_rounds: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Node {
    Literal(String),
    Call(String, Function, Vec<usize>),
}

pub(crate) struct Graph {
    pub sorts: Vec<String>,
    pub nodes: Vec<Vec<Node>>,
    pub roots: BTreeSet<usize>,
    pub index: BTreeMap<String, usize>,
}

impl Graph {
    pub fn new(db: &Database, offset: usize) -> Self {
        let ids: Vec<_> = db.classes.keys().cloned().collect();
        let index: BTreeMap<_, _> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i + offset))
            .collect();
        let sorts = db.classes.values().map(|c| c.sort.clone()).collect();
        let mut nodes = vec![Vec::new(); ids.len()];
        for (id, class) in &db.classes {
            if let Some(literal) = &class.literal {
                nodes[index[id] - offset].push(Node::Literal(literal.clone()));
            }
        }
        let mut roots = BTreeSet::new();
        for row in &db.rows {
            let function = &db.functions[&row.function];
            if function.kind == FunctionKind::Constructor {
                roots.insert(index[&row.output]);
                nodes[index[&row.output] - offset].push(Node::Call(
                    row.function.clone(),
                    function.clone(),
                    row.inputs.iter().map(|id| index[id]).collect(),
                ));
            }
        }
        Self {
            sorts,
            nodes,
            roots,
            index,
        }
    }
}

pub(crate) struct Partition {
    pub left: Graph,
    pub right: Graph,
    pub blocks: Vec<usize>,
    pub rounds: usize,
}

impl Partition {
    pub fn new(left: &Database, right: &Database) -> Self {
        let left = Graph::new(left, 0);
        let right = Graph::new(right, left.nodes.len());
        // Sorts must never become equivalent, even for empty e-classes.
        let mut sorts = BTreeMap::new();
        let blocks = left
            .sorts
            .iter()
            .chain(&right.sorts)
            .map(|sort| {
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
        let mut signatures = BTreeMap::new();
        let mut next_blocks = Vec::with_capacity(self.blocks.len());
        for (i, nodes) in self.left.nodes.iter().chain(&self.right.nodes).enumerate() {
            let signature: BTreeSet<_> = nodes
                .iter()
                .map(|node| match node {
                    Node::Literal(value) => Node::Literal(value.clone()),
                    Node::Call(op, schema, children) => Node::Call(
                        op.clone(),
                        schema.clone(),
                        children.iter().map(|&child| self.blocks[child]).collect(),
                    ),
                })
                .collect();
            // Retain the old block: every round is a refinement, never a merge.
            let next = signatures.len();
            next_blocks.push(
                *signatures
                    .entry((self.blocks[i], signature))
                    .or_insert(next),
            );
        }
        self.rounds += 1;
        let changed = next_blocks != self.blocks;
        self.blocks = next_blocks;
        changed
    }

    pub fn finish(&mut self) {
        while self.step() {}
    }

    pub fn root_blocks(&self, graph: &Graph) -> BTreeSet<usize> {
        graph.roots.iter().map(|&id| self.blocks[id]).collect()
    }

    pub fn terms_equal(&self) -> bool {
        self.root_blocks(&self.left) == self.root_blocks(&self.right)
    }
}

/// Exact partition refinement; no hash collisions or iteration/depth cutoff.
///
/// This intentionally starts with the straightforward whole-graph algorithm.
/// A round scans all nodes and edges and sorts signatures; there are at most
/// O(classes) splitting rounds. The result ignores row order and duplicate rows
/// or bisimilar cyclic classes. Function rows do not participate in term syntax.
pub fn compare(left: &Database, right: &Database) -> Result<Comparison, Error> {
    left.validate()?;
    right.validate()?;
    let mut partition = Partition::new(left, right);
    partition.finish();
    let terms_equal = partition.terms_equal();
    let rows = |db: &Database, graph: &Graph| -> BTreeSet<_> {
        db.rows
            .iter()
            .map(|row| {
                (
                    row.function.clone(),
                    row.inputs
                        .iter()
                        .map(|id| partition.blocks[graph.index[id]])
                        .collect::<Vec<_>>(),
                    partition.blocks[graph.index[&row.output]],
                    row.subsumed,
                )
            })
            .collect()
    };
    let database_equal = terms_equal
        && left.functions == right.functions
        && rows(left, &partition.left) == rows(right, &partition.right);
    Ok(Comparison {
        terms_equal,
        database_equal,
        refinement_rounds: partition.rounds,
    })
}
