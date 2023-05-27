use super::*;
use graphviz_rust::{attributes as a, dot_structures as d};

fn eclass_cluster_name(eclass_id: &EClassID) -> String {
    format!("cluster_{}", eclass_id)
}
/// The Node ID for the eclass is the first node in the eclass cluster
fn eclass_node_id(eclass_id: &EClassID) -> d::NodeId {
    d::NodeId(d::Id::Plain(quote(format!("e_{}_0", eclass_id))), None)
}

/// An e-class is converted into a cluster with a node for each function call
fn eclass_to_graphviz(eclass_id: &EClassID, fn_calls: &[FnCall]) -> Vec<d::Stmt> {
    let mut stmts: Vec<d::Stmt> = fn_calls
        .iter()
        .enumerate()
        .flat_map(|(index, fn_call)| {
            fn_call
                .1
                .iter()
                .flat_map(|arg| arg.to_graphviz(format!("e_{}_{}", eclass_id, index)))
                .collect::<Vec<d::Stmt>>()
        })
        .collect();
    stmts.push(d::Stmt::Subgraph(d::Subgraph {
        id: d::Id::Plain(eclass_cluster_name(eclass_id)),
        // Nest in empty sub-graph so that we can use rank=same
        // https://stackoverflow.com/a/55562026/907060
        stmts: vec![d::Stmt::Subgraph(d::Subgraph {
            id: d::Id::Plain("".to_string()),
            stmts: fn_calls
                .iter()
                .enumerate()
                .map(|(index, fn_call)| {
                    d::Stmt::Node(d::Node::new(
                        d::NodeId(
                            d::Id::Plain(quote(format!("e_{}_{}", eclass_id, index))),
                            None,
                        ),
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
    fn to_graphviz(&self) -> Vec<d::Stmt> {
        let mut stmts = Vec::new();
        let label = format!("{}: {}", self.0 .0.name, self.1.to_string());
        let res_id = format!("po_{}_{}", self.0 .0.name, self.2);
        stmts.push(d::Stmt::Node(d::Node::new(
            d::NodeId(
                d::Id::Plain(quote(res_id.clone())),
                None,
            ),
            label_attributes(label),
        )));
        stmts.extend(self.0 .1.iter().flat_map(|arg| arg.to_graphviz(res_id.clone())));
        stmts
    }
}

impl Arg {
    /// Returns an edge from the result to the argument
    /// If it's an e-class, use the e-class-id as the target
    /// Otherwise, create a node for the primitive value and use that as the target
    fn to_graphviz(&self, result_id:String) -> Vec<d::Stmt> {
        let result_node = d::Vertex::N(d::NodeId(d::Id::Plain(quote(result_id.clone())), None));
        match self {
            Arg::Prim(p) => {
                let arg_id = d::NodeId(
                    d::Id::Plain(quote(format!("p_{}_{}", result_id, p.to_string()))),
                    None,
                );
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
            Arg::Eq(eclass_id) => {
                vec![d::Stmt::Edge(d::Edge {
                    ty: d::EdgeTy::Pair(result_node, d::Vertex::N(eclass_node_id(eclass_id))),
                    attributes: vec![graphviz_rust::attributes::EdgeAttributes::lhead(
                        eclass_cluster_name(eclass_id),
                    )],
                })]
            }
        }
    }
}

/// Implement conversion of Graph to graphviz Graph
impl Graph {
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
        statements.extend(self.prim_outputs.iter().flat_map(|po| po.to_graphviz()));
        statements.extend(
            self.eclasses
                .iter()
                .flat_map(|(eclass_id, eclass)| eclass_to_graphviz(eclass_id, eclass)),
        );
        d::Graph::DiGraph {
            id: d::Id::Plain("egg_smol".to_string()),
            strict: false,
            stmts: statements,
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
