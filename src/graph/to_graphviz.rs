use super::*;
use graphviz_rust::{attributes as a, dot_structures as d};

/// Implement conversion of Graph to graphviz Graph
impl ExportedGraph {
    pub fn to_graphviz(&self) -> d::Graph {
        let mut statements = vec![
            // Set to compound so we can have edge to clusters
            d::Stmt::Attribute(a::GraphAttributes::compound(true)),
            // Set default sub-graph rank to be same so that all nodes in e-class are on same level
            d::Stmt::Attribute(a::SubgraphAttributes::rank(a::rank::same)),
            d::Stmt::Attribute(a::GraphAttributes::fontname("helvetica".to_string())),
            d::Stmt::Attribute(a::GraphAttributes::style(quote(
                "rounded,dashed".to_string(),
            ))),
            d::Stmt::GAttribute(d::GraphAttributes::Edge(vec![
                a::EdgeAttributes::arrowsize(0.5),
            ])),
            d::Stmt::GAttribute(d::GraphAttributes::Node(vec![
                a::NodeAttributes::shape(a::shape::box_),
                a::NodeAttributes::style("rounded".to_string()),
                a::NodeAttributes::width(0.4),
                a::NodeAttributes::height(0.4),
            ])),
        ];
        statements.extend(
            self.prim_outputs
                .iter()
                .flat_map(|po| po.to_graphviz(&self.eclasses)),
        );
        statements.extend(
            self.eclasses.iter().flat_map(|(eclass_id, eclass)| {
                eclass_to_graphviz(eclass_id, eclass, &self.eclasses)
            }),
        );
        d::Graph::DiGraph {
            id: d::Id::Plain("egg_smol".to_string()),
            strict: false,
            stmts: statements,
        }
    }
}

/// An e-class is converted into a cluster with a node for each function call
fn eclass_to_graphviz(
    eclass_id: &EClassID,
    fn_calls: &[FnCall],
    eclasses: &EClasses,
) -> Vec<d::Stmt> {
    // Create node for all arguments of every function call
    let mut stmts: Vec<d::Stmt> = fn_calls
        .iter()
        .flat_map(|fn_call| {
            fn_call
                .1
                .iter()
                .flat_map(|arg| arg.to_graphviz(fn_call_id(fn_call), eclasses))
                .collect::<Vec<d::Stmt>>()
        })
        .collect();
    // Add a node for each function call in one e-class
    stmts.push(d::Stmt::Subgraph(d::Subgraph {
        id: d::Id::Plain(cluster_name(eclass_id)),
        // Nest in empty sub-graph so that we can use rank=same
        // https://stackoverflow.com/a/55562026/907060
        stmts: vec![d::Stmt::Subgraph(d::Subgraph {
            id: d::Id::Plain("".to_string()),
            stmts: fn_calls
                .iter()
                .map(|fn_call| {
                    d::Stmt::Node(d::Node::new(
                        d::NodeId(d::Id::Plain(quote(fn_call_id(fn_call))), None),
                        label_attributes(fn_call.0.name.clone()),
                    ))
                })
                .collect(),
        })],
    }));
    stmts
}

impl PrimOutput {
    /// A primitive output, should be a node with the value and function call
    fn to_graphviz(&self, eclasses: &EClasses) -> Vec<d::Stmt> {
        let mut stmts = Vec::new();
        let label = format!("{}: {}", self.0 .0.name, self.1.to_string());
        let res_id = fn_call_id(&self.0);
        stmts.push(d::Stmt::Node(d::Node::new(
            d::NodeId(d::Id::Plain(quote(res_id.clone())), None),
            label_attributes(label),
        )));
        stmts.extend(
            self.0
                 .1
                .iter()
                .flat_map(|arg| arg.to_graphviz(res_id.clone(), eclasses)),
        );
        stmts
    }
}

impl Arg {
    /// Returns an edge from the result to the argument
    /// If it's an e-class, use the e-class-id as the target
    /// Otherwise, create a node for the primitive value and use that as the target
    fn to_graphviz(&self, result_id: String, eclasses: &EClasses) -> Vec<d::Stmt> {
        let result_node = d::Vertex::N(d::NodeId(d::Id::Plain(quote(result_id.clone())), None));
        match self {
            Arg::Prim(p) => {
                let arg_id = d::NodeId(d::Id::Plain(quote(prim_value_id(result_id, p))), None);
                vec![
                    d::Stmt::Node(d::Node::new(
                        arg_id.clone(),
                        label_attributes(p.to_string()),
                    )),
                    d::Stmt::Edge(d::Edge {
                        ty: d::EdgeTy::Pair(result_node, d::Vertex::N(arg_id)),
                        attributes: vec![],
                    }),
                ]
            }
            Arg::Eq(id) => {
                vec![d::Stmt::Edge(d::Edge {
                    ty: d::EdgeTy::Pair(
                        result_node,
                        d::Vertex::N(d::NodeId(
                            d::Id::Plain(quote(enode_fn_id(id, eclasses))),
                            None,
                        )),
                    ),
                    attributes: vec![graphviz_rust::attributes::EdgeAttributes::lhead(
                        cluster_name(id),
                    )],
                })]
            }
        }
    }
}

fn label_attributes(label: String) -> Vec<d::Attribute> {
    vec![graphviz_rust::attributes::NodeAttributes::label(quote(
        label,
    ))]
}
fn quote(s: String) -> String {
    format!("{:?}", s)
}

fn cluster_name(canonical_id: &EClassID) -> String {
    format!("cluster_{}", canonical_id)
}

// Edges to enodes should point to the first function call in the e-class
fn enode_fn_id(eclass_id: &EClassID, eclasses: &EClasses) -> String {
    fn_call_id(&eclasses[eclass_id][0])
}

// Function calls are uniquely identified by the function name and the hash of the arguments
fn fn_call_id(fn_call: &FnCall) -> String {
    format!("{}_{}", fn_call.0.name, fn_call.2)
}

fn prim_value_id(parent_name: String, value: &PrimValue) -> String {
    format!("{}_{}", parent_name, value.to_string())
}
