use super::*;
use graphviz_rust::dot_structures as g;

fn eclass_cluster_name(eclass_id: &EClassID) -> String {
    format!("cluster_{}", eclass_id)
}
// The Node ID for the eclass is the first node in the eclass cluster
fn eclass_node_id(eclass_id: &EClassID) -> g::NodeId {
    g::NodeId(g::Id::Plain(quote(format!("{}_0", eclass_id))), None)
}

// An e-class is converted into a cluster with a node for each function call
fn eclass_to_graphviz(
    eclass_id: &EClassID,
    fn_calls: &[FnCall],
    id_gen: &mut NodeIDGenerator,
) -> Vec<g::Stmt> {
    let mut stmts: Vec<g::Stmt> = fn_calls
        .iter()
        .enumerate()
        .flat_map(|(index, fn_call)| {
            fn_call
                .1
                .iter()
                .flat_map(|arg| {
                    arg.to_graphviz(
                        id_gen,
                        g::NodeId(
                            g::Id::Plain(quote(format!("{}_{}", eclass_id, index))),
                            None,
                        ),
                    )
                })
                .collect::<Vec<g::Stmt>>()
        })
        .collect();
    stmts.push(g::Stmt::Subgraph(g::Subgraph {
        id: g::Id::Plain(eclass_cluster_name(eclass_id)),
        stmts: fn_calls
            .iter()
            .enumerate()
            .map(|(index, fn_call)| {
                g::Stmt::Node(g::Node::new(
                    g::NodeId(
                        g::Id::Plain(quote(format!("{}_{}", eclass_id, index))),
                        None,
                    ),
                    label_attributes(fn_call.0.name.clone()),
                ))
            })
            .collect(),
    }));
    stmts
}

impl PrimOutput {
    // A primitive output, should be a node with the value and function call
    fn to_graphviz(&self, id_gen: &mut NodeIDGenerator) -> Vec<g::Stmt> {
        let mut stmts = Vec::new();
        let label = format!("{}: {}", self.0 .0.name, self.1.to_string());
        let res_id = id_gen.next();
        stmts.push(g::Stmt::Node(g::Node::new(
            res_id.clone(),
            label_attributes(label),
        )));
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
    fn to_graphviz(&self, id_gen: &mut NodeIDGenerator, result_id: g::NodeId) -> Vec<g::Stmt> {
        match self {
            Arg::Prim(p) => {
                let arg_id = id_gen.next();
                vec![
                    g::Stmt::Node(g::Node::new(
                        arg_id.clone(),
                        label_attributes(p.to_string()),
                    )),
                    g::Stmt::Edge(g::Edge {
                        ty: g::EdgeTy::Pair(g::Vertex::N(result_id), g::Vertex::N(arg_id)),
                        attributes: vec![],
                    }),
                ]
            }
            Arg::Eq(eclass_id) => {
                vec![g::Stmt::Edge(g::Edge {
                    ty: g::EdgeTy::Pair(
                        g::Vertex::N(result_id),
                        g::Vertex::N(eclass_node_id(eclass_id)),
                    ),
                    attributes: vec![graphviz_rust::attributes::EdgeAttributes::lhead(
                        eclass_cluster_name(eclass_id),
                    )],
                })]
            }
        }
    }
}

// Implement conversion of Graph to graphviz Graph
impl Graph {
    pub fn to_graphviz(&self) -> g::Graph {
        let id_generator = &mut NodeIDGenerator::new();
        // Set compound to true
        let mut statements = vec![g::Stmt::Attribute(GraphAttributes::compound(true))];
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
        g::Graph::DiGraph {
            id: g::Id::Plain("egg_smol".to_string()),
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

    fn next(&mut self) -> g::NodeId {
        let id = self.next_id;
        self.next_id += 1;
        g::NodeId(g::Id::Plain(id.to_string()), None)
    }
}

fn label_attributes(label: String) -> Vec<g::Attribute> {
    vec![graphviz_rust::attributes::NodeAttributes::label(quote(
        label,
    ))]
}
fn quote(s: String) -> String {
    format!("{:?}", s)
}
