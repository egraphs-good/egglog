use graphviz_rust::attributes::GraphAttributes;

use crate::{ast::Expr, util::HashMap};

type EClassID = String;

// Exposed graph structure which can be used to print/visualize the state of the e-graph.
#[derive(Debug)]
pub(crate) struct Graph {
    // All of the primitive values which are outputs of functions
    pub prim_outputs: Vec<PrimOutput>,
    // All of the e-classes which are have non primitive types
    pub eclasses: HashMap<EClassID, Vec<FnCall>>,
}

// A primitive value which is output from a function.
#[derive(Debug)]
pub(crate) struct PrimOutput(pub FnCall, pub PrimValue);

#[derive(Debug)]
pub(crate) struct FnCall(pub Fn, pub Vec<Arg>);

/// An argument is either a primitive value or a reference to a eclass
#[derive(Debug)]
pub(crate) enum Arg {
    Prim(PrimValue),
    Eq(EClassID),
}

#[derive(Debug)]
pub(crate) struct Fn {
    pub name: String,
    // TODO: Add cost
}

/// A primitive value (str, float, int, etc)
#[derive(Debug)]
pub(crate) struct PrimValue(String);

fn eclass_cluster_name(eclass_id: &EClassID) -> String {
    format!("cluster_{}", eclass_id)
}
// The Node ID for the eclass is the first node in the eclass cluster
fn eclass_node_id(eclass_id: &EClassID) -> graphviz_rust::dot_structures::NodeId {
    graphviz_rust::dot_structures::NodeId(
        graphviz_rust::dot_structures::Id::Plain(quote(format!("{}_0", eclass_id))),
        None,
    )
}

// An e-class is converted into a cluster with a node for each function call
fn eclass_to_graphviz(
    eclass_id: &EClassID,
    fn_calls: &[FnCall],
    id_gen: &mut NodeIDGenerator,
) -> Vec<graphviz_rust::dot_structures::Stmt> {
    let mut stmts: Vec<graphviz_rust::dot_structures::Stmt> = fn_calls
        .iter()
        .enumerate()
        .flat_map(|(index, fn_call)| {
            fn_call
                .1
                .iter()
                .flat_map(|arg| {
                    arg.to_graphviz(
                        id_gen,
                        graphviz_rust::dot_structures::NodeId(
                            graphviz_rust::dot_structures::Id::Plain(quote(format!(
                                "{}_{}",
                                eclass_id, index
                            ))),
                            None,
                        ),
                    )
                })
                .collect::<Vec<graphviz_rust::dot_structures::Stmt>>()
        })
        .collect();
    stmts.push(graphviz_rust::dot_structures::Stmt::Subgraph(
        graphviz_rust::dot_structures::Subgraph {
            id: graphviz_rust::dot_structures::Id::Plain(eclass_cluster_name(eclass_id)),
            stmts: fn_calls
                .iter()
                .enumerate()
                .map(|(index, fn_call)| {
                    graphviz_rust::dot_structures::Stmt::Node(
                        graphviz_rust::dot_structures::Node::new(
                            graphviz_rust::dot_structures::NodeId(
                                graphviz_rust::dot_structures::Id::Plain(quote(format!(
                                    "{}_{}",
                                    eclass_id, index
                                ))),
                                None,
                            ),
                            label_attributes(fn_call.0.name.clone()),
                        ),
                    )
                })
                .collect(),
        },
    ));
    stmts
}

impl PrimOutput {
    // A primitive output, should be a node with the value and function call
    fn to_graphviz(
        &self,
        id_gen: &mut NodeIDGenerator,
    ) -> Vec<graphviz_rust::dot_structures::Stmt> {
        let mut stmts = Vec::new();
        let label = format!("{}: {}", self.0 .0.name, self.1.to_string());
        let res_id = id_gen.next();
        stmts.push(graphviz_rust::dot_structures::Stmt::Node(
            graphviz_rust::dot_structures::Node::new(res_id.clone(), label_attributes(label)),
        ));
        stmts.extend(
            self.0
                 .1
                .iter()
                .flat_map(|arg| arg.to_graphviz(id_gen, res_id.clone())),
        );
        stmts
    }
}

impl Arg {
    // Returns an edge from the result to the argument
    // If it's an e-class, use the e-class-id as the target
    // Otherwise, create a node for the primitive value and use that as the target
    fn to_graphviz(
        &self,
        id_gen: &mut NodeIDGenerator,
        result_id: graphviz_rust::dot_structures::NodeId,
    ) -> Vec<graphviz_rust::dot_structures::Stmt> {
        match self {
            Arg::Prim(p) => {
                let arg_id = id_gen.next();
                vec![
                    graphviz_rust::dot_structures::Stmt::Node(
                        graphviz_rust::dot_structures::Node::new(
                            arg_id.clone(),
                            label_attributes(p.to_string()),
                        ),
                    ),
                    graphviz_rust::dot_structures::Stmt::Edge(
                        graphviz_rust::dot_structures::Edge {
                            ty: graphviz_rust::dot_structures::EdgeTy::Pair(
                                graphviz_rust::dot_structures::Vertex::N(result_id),
                                graphviz_rust::dot_structures::Vertex::N(arg_id),
                            ),
                            attributes: vec![],
                        },
                    ),
                ]
            }
            Arg::Eq(eclass_id) => {
                vec![graphviz_rust::dot_structures::Stmt::Edge(
                    graphviz_rust::dot_structures::Edge {
                        ty: graphviz_rust::dot_structures::EdgeTy::Pair(
                            graphviz_rust::dot_structures::Vertex::N(result_id),
                            graphviz_rust::dot_structures::Vertex::N(eclass_node_id(eclass_id)),
                        ),
                        attributes: vec![graphviz_rust::attributes::EdgeAttributes::lhead(
                            eclass_cluster_name(eclass_id),
                        )],
                    },
                )]
            }
        }
    }
}

pub(crate) fn from_expr(expr: &Expr) -> PrimValue {
    PrimValue(expr.to_string())
}

impl ToString for PrimValue {
    fn to_string(&self) -> String {
        self.0.clone()
    }
}

// Implement conversion of Graph to graphviz Graph
impl Graph {
    pub fn to_graphviz(&self) -> graphviz_rust::dot_structures::Graph {
        let id_generator = &mut NodeIDGenerator::new();
        // Set compound to true
        let mut statements = vec![graphviz_rust::dot_structures::Stmt::Attribute(
            GraphAttributes::compound(true),
        )];
        statements.extend(
            self.prim_outputs
                .iter()
                .flat_map(|po: &PrimOutput| po.to_graphviz(id_generator)),
        );
        statements.extend(
            self.eclasses.iter().flat_map(|(eclass_id, eclass)| {
                eclass_to_graphviz(eclass_id, eclass, id_generator)
            }),
        );
        graphviz_rust::dot_structures::Graph::DiGraph {
            id: graphviz_rust::dot_structures::Id::Plain("egg_smol".to_string()),
            strict: false,
            stmts: statements,
        }
    }
}

// Struct which generates an incrementing ID for each node
struct NodeIDGenerator {
    next_id: usize,
}

impl NodeIDGenerator {
    fn new() -> Self {
        Self { next_id: 0 }
    }

    fn next(&mut self) -> graphviz_rust::dot_structures::NodeId {
        let id = self.next_id;
        self.next_id += 1;
        graphviz_rust::dot_structures::NodeId(
            graphviz_rust::dot_structures::Id::Plain(id.to_string()),
            None,
        )
    }
}

fn label_attributes(label: String) -> Vec<graphviz_rust::dot_structures::Attribute> {
    vec![graphviz_rust::attributes::NodeAttributes::label(quote(
        label,
    ))]
}
fn quote(s: String) -> String {
    format!("{:?}", s)
}
