use super::*;
use graphviz_rust::{
    attributes::*,
    dot_generator::*,
    dot_structures::{
        Attribute, Edge, EdgeTy, Graph, GraphAttributes as GA, Id, Node, NodeId, Port, Stmt,
        Subgraph, Vertex,
    },
};

/// Implement conversion of Graph to graphviz Graph
impl ExportedGraph {
    pub fn to_graphviz(&self) -> Graph {
        let mut statements = vec![
            // Set to compound so we can have edge to clusters
            stmt!(GraphAttributes::compound(true)),
            // Set default sub-graph rank to be same so that all nodes in e-class are on same level
            stmt!(SubgraphAttributes::rank(rank::same)),
            stmt!(GraphAttributes::fontname("helvetica".to_string())),
            stmt!(GraphAttributes::style(quote("rounded,dashed".to_string()))),
            stmt!(GraphAttributes::margin(3.0)),
            stmt!(GraphAttributes::nodesep(0.0)),
            stmt!(GA::Edge(vec![EdgeAttributes::arrowsize(0.5)])),
            stmt!(GA::Node(vec![
                NodeAttributes::shape(shape::none),
                NodeAttributes::margin(0.0)
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
        graph!(strict di id!(), statements)
    }
}

/// An e-class is converted into a cluster with a node for each function call
fn eclass_to_graphviz(eclass_id: &EClassID, fn_calls: &[FnCall], eclasses: &EClasses) -> Vec<Stmt> {
    // Create node for all arguments of every function call
    let mut stmts: Vec<Stmt> = fn_calls
        .iter()
        .flat_map(|fn_call| {
            fn_call
                .1
                .iter()
                .enumerate()
                .flat_map(|(index, arg)| arg.to_graphviz(index, fn_call_id(fn_call), eclasses))
                .collect::<Vec<Stmt>>()
        })
        .collect();
    // Add a node for each function call in one e-class
    let subgraph_stmts = fn_calls
        .iter()
        .map(|fn_call| {
            let id = fn_call_id(fn_call);
            stmt!(node!(esc id; html_label(fn_call.0.name.clone(), fn_call.1.len())))
        })
        .collect::<Vec<Stmt>>();
    let cluster_id = cluster_name(eclass_id);
    // Nest in empty sub-graph so that we can use rank=same
    // https://stackoverflow.com/a/55562026/907060
    stmts.push(stmt!(subgraph!(cluster_id; subgraph!("", subgraph_stmts))));
    stmts
}

impl PrimOutput {
    /// A primitive output, should be a node with the value and function call
    fn to_graphviz(&self, eclasses: &EClasses) -> Vec<Stmt> {
        let label = format!("{}: {}", self.0 .0.name, self.1.to_string());
        let res_id = fn_call_id(&self.0);
        let node_id = quote(res_id.clone());
        let args = &self.0 .1;
        let mut stmts = vec![stmt!(node!(node_id; html_label(label, args.len())))];
        stmts.extend(
            args.iter()
                .enumerate()
                .flat_map(|(index, arg)| arg.to_graphviz(index, res_id.clone(), eclasses)),
        );
        stmts
    }
}

impl Arg {
    /// Returns an edge from the result to the argument
    /// If it's an e-class, use the e-class-id as the target
    /// Otherwise, create a node for the primitive value and use that as the target
    fn to_graphviz(&self, index: usize, result_id: String, eclasses: &EClasses) -> Vec<Stmt> {
        let source = node_id!(quote(result_id.clone()), port!(id!(format!("a{}", index))));
        match self {
            Arg::Prim(p) => {
                let arg_id = quote(prim_value_id(result_id, p));
                vec![
                    stmt!(node!(arg_id; html_label(p.to_string(), 0))),
                    stmt!(edge!(source => node_id!(arg_id))),
                ]
            }
            Arg::Eq(id) => {
                vec![stmt!(edge!(
                    source => node_id!(quote(enode_fn_id(id, eclasses)));
                    EdgeAttributes::lhead(cluster_name(id))
                ))]
            }
        }
    }
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

/// Returns an html label for the node with the function name and ports for each argumetn
fn html_label(label: String, n_args: usize) -> Attribute {
    NodeAttributes::label(format!(
        "<<TABLE CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"0\" style=\"rounded\">
        <tr><td CELLPADDING=\"4\" WIDTH=\"30\" HEIGHT=\"30\" colspan=\"{}\">{}</td></tr>
        {}
        </TABLE>>",
        n_args,
        label,
        (if n_args == 0 {
            "".to_string()
        } else {
            format!(
                "<TR>{}</TR>",
                (0..n_args)
                    .map(|i| format!("<TD PORT=\"a{}\"></TD>", i))
                    .collect::<Vec<String>>()
                    .join("")
            )
        })
    ))
}
