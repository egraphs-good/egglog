use crate::ids::{BlockId, ClassId, SymbolId};
use crate::{Database, Error, Function, FunctionKind, HashMap};
use egglog_numeric_id::NumericId;
use serde::Serialize;
use smallvec::SmallVec;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Comparison {
    /// Constructor e-classes have the same sets of bisimulation behaviors.
    pub terms_equal: bool,
    /// Additionally, declarations and the full database coalgebra agree.
    pub database_equal: bool,
    /// Worklist blocks processed during constructor-only refinement, not depth.
    pub refinement_steps: usize,
    /// Worklist blocks processed for database equality (possibly reused).
    pub database_refinement_steps: usize,
}

// Intern complete labels jointly across the inputs. Neither names nor schemas
// are copied per node or per refinement round; hash collisions still use Eq.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub(crate) enum Label<'a> {
    Literal(&'a str),
    Call(&'a str, &'a Function, bool),
}

fn intern<'a>(labels: &mut HashMap<Label<'a>, SymbolId>, label: Label<'a>) -> SymbolId {
    let next = SymbolId::from_usize(labels.len());
    *labels.entry(label).or_insert(next)
}

#[derive(Clone)]
pub(crate) struct Node {
    pub symbol: SymbolId,
    pub children: SmallVec<[ClassId; 2]>,
}

pub(crate) struct Graph<'a> {
    pub sorts: Vec<&'a str>,
    pub nodes: Vec<Vec<Node>>,
    /// Sorted, unique output IDs. Mark once per row instead of tree insertion.
    pub roots: Vec<ClassId>,
    pub index: HashMap<&'a str, ClassId>,
}

impl<'a> Graph<'a> {
    pub(crate) fn new(
        db: &'a Database,
        offset: usize,
        include_functions: bool,
        labels: &mut HashMap<Label<'a>, SymbolId>,
    ) -> Self {
        let index: HashMap<_, _> = db
            .classes
            .keys()
            .enumerate()
            .map(|(i, id)| (id.as_str(), ClassId::from_usize(i + offset)))
            .collect();
        let sorts = db.classes.values().map(|c| c.sort.as_str()).collect();
        let mut nodes = vec![Vec::new(); db.classes.len()];
        for (id, class) in &db.classes {
            if let Some(literal) = &class.literal {
                nodes[index[id.as_str()].index() - offset].push(Node {
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
                let output = index[row.output.as_str()].index() - offset;
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
                .filter_map(|(i, root)| root.then_some(ClassId::from_usize(i + offset)))
                .collect(),
            index,
        }
    }
}

pub(crate) struct Partition<'a> {
    pub left: Graph<'a>,
    pub right: Graph<'a>,
    pub blocks: Vec<BlockId>,
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
                let next = BlockId::from_usize(sorts.len());
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

    pub(crate) fn signature(
        &self,
        id: ClassId,
        nodes: &mut Vec<SmallVec<[usize; 3]>>,
        signatures: &mut crate::signatures::Signatures,
    ) -> BlockId {
        let class = if id.index() < self.left.nodes.len() {
            &self.left.nodes[id.index()]
        } else {
            &self.right.nodes[id.index() - self.left.nodes.len()]
        };
        nodes.clear();
        nodes.extend(class.iter().map(|node| {
            let mut signature = SmallVec::new();
            signature.push(node.symbol.index());
            signature.extend(
                node.children
                    .iter()
                    .map(|&child| self.blocks[child.index()].index()),
            );
            signature
        }));
        nodes.sort_unstable();
        nodes.dedup();
        signatures.intern(self.blocks[id.index()], nodes)
    }

    // Certificates use synchronous depth, independently of comparison worklist steps.
    pub fn step(&mut self) -> bool {
        let mut signatures = crate::signatures::Signatures::default();
        let mut nodes = Vec::new();
        let next_blocks = (0..self.blocks.len())
            .map(|i| self.signature(ClassId::from_usize(i), &mut nodes, &mut signatures))
            .collect::<Vec<_>>();
        self.rounds += 1;
        let changed = next_blocks != self.blocks;
        self.blocks = next_blocks;
        changed
    }

    pub fn finish(&mut self) {
        while self.step() {}
    }

    pub fn root_blocks(&self, graph: &Graph<'_>) -> BTreeSet<BlockId> {
        graph
            .roots
            .iter()
            .map(|&id| self.blocks[id.index()])
            .collect()
    }

    pub fn terms_equal(&self) -> bool {
        let mut coverage = vec![0u8; self.blocks.len()];
        for &root in &self.left.roots {
            coverage[self.blocks[root.index()].index()] |= 1;
        }
        for &root in &self.right.roots {
            coverage[self.blocks[root.index()].index()] |= 2;
        }
        coverage.iter().all(|&sides| sides == 0 || sides == 3)
    }
}

/// Exact partition refinement using a smaller-half worklist. Each class moves
/// to a newly numbered block at most O(log(classes)) times; only predecessors
/// of moved classes become dirty. Signatures still inspect and sort all nodes
/// of each dirty class, so their cost is additional to the worklist bound.
/// Hash collisions use full equality; there is no probabilistic/depth cutoff.
/// Ordinary functions never participate in constructor-term syntax.
pub fn compare(left: &Database, right: &Database) -> Result<Comparison, Error> {
    left.validate()?;
    right.validate()?;
    let mut partition = Partition::new(left, right);
    let refinement_steps = crate::hopcroft::refine(&mut partition);
    let terms_equal = partition.terms_equal();
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
            refinement_steps,
            database_refinement_steps: refinement_steps,
        });
    }
    // Release the first graph before constructing the full-database graph.
    drop(partition);
    let mut partition = Partition::database(left, right);
    let database_refinement_steps = crate::hopcroft::refine(&mut partition);
    // Every row is a member of an output class. Matching output blocks at the
    // fixed point already implies matching rows modulo the child/output blocks;
    // rebuilding a second set of string-keyed row signatures is redundant.
    let database_equal =
        terms_equal && left.functions == right.functions && partition.terms_equal();
    Ok(Comparison {
        terms_equal,
        database_equal,
        refinement_steps,
        database_refinement_steps,
    })
}
