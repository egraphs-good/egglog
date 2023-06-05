use crate::util::{HashMap, HashSet};
use std::collections::VecDeque;
use std::fmt;

use super::*;
use graphviz_rust::{
    attributes::*,
    dot_generator::*,
    dot_structures::{
        Attribute, Edge, EdgeTy, Graph, GraphAttributes as GA, Id, Node, NodeId, Port, Stmt,
        Subgraph, Vertex,
    },
};

/// Mapping from e-class id to the ID of the node representing particular function call
type EClasses = HashMap<EClassID, VecDeque<String>>;
/// Mapping from subgraph id to the sort, whether it's an e-class, and the nodes in the subgraph
type Nodes = HashMap<String, (String, bool, Vec<Node>)>;
type Edges = Vec<Edge>;

/// Set of all sorts
type Sorts = HashSet<String>;
// Mapping of sort name to graphviz color
type SortColors = HashMap<String, usize>;

/// Creates a graphviz graph from an exported graph
///
/// Each function call is a node, and each edge is an argument to the function
/// E-classes are grouped in clusters
/// Functions which return primitive values are shown as a cluster with two items, the function node and the value node.
///
/// Each cluster is also labeled with the sort of the values it contains and colored by the sort
pub(crate) fn to_graphviz(g: &ExportedGraph) -> Graph {
    let eclasses = build_eclasses(g);
    let (nodes, edges, sorts) = build_nodes_edges(g, eclasses);
    let colors = build_colors(&sorts);
    build_graph(nodes, edges, &colors)
}

/// Splits the calls into those returning primitive values and those returning e-classes, grouping the e-classes by id
fn build_eclasses(g: &ExportedGraph) -> EClasses {
    let mut eclasses: EClasses = HashMap::default();
    for call in g {
        if let ExportedValue::EClass(id) = call.output.0 {
            eclasses
                .entry(id)
                .or_default()
                .push_front(call.function_node_id());
        }
    }
    eclasses
}

fn build_nodes_edges(calls: &ExportedGraph, eclasses: EClasses) -> (Nodes, Edges, Sorts) {
    let mut builder = SubgraphBuilder::new(eclasses);
    for call in calls {
        builder.add_call(call);
    }
    (builder.nodes, builder.edges, builder.sorts)
}

struct SubgraphBuilder {
    eclasses: EClasses,
    nodes: Nodes,
    edges: Edges,
    sorts: Sorts,
}

impl SubgraphBuilder {
    fn new(eclasses: EClasses) -> Self {
        Self {
            eclasses,
            nodes: Nodes::default(),
            sorts: Sorts::default(),
            edges: Edges::default(),
        }
    }

    fn add_call(&mut self, call: &ExportedCall) {
        let sort: &str = &call.output.1;
        self.sorts.insert(sort.to_string());

        let function_node_id = call.function_node_id();

        let mut nodes = vec![self.add_node(&function_node_id, &call.fn_name, &call.inputs)];
        match &call.output.0 {
            ExportedValue::EClass(eclass_id) => {
                let subgraph_id = format!("cluster_{}", eclass_id);
                let subgraph_entry = self.nodes.entry(subgraph_id).or_default();
                subgraph_entry.1 = true;
                subgraph_entry.0 = sort.to_string();
                subgraph_entry.2.extend(nodes);
            }
            ExportedValue::Prim(name, inner, _) => {
                let subgraph_id = format!("cluster_{}", function_node_id);
                let value_node_id = format!("{}_value", function_node_id);
                // Add the function return value, unless it's a unit sort
                if sort != "Unit" {
                    nodes.push(self.add_node(&value_node_id, name, inner));
                }
                self.add_value_subgraph(&subgraph_id, sort, nodes);
            }
        };
    }

    /// Adds a node with some children
    fn add_node(&mut self, node_id: &str, label: &str, children: &[ExportedValueWithSort]) -> Node {
        let html_label = html_label(label, children.len());
        let quoted_node_id = quote(node_id);
        for (i, value) in children.iter().enumerate() {
            let source = node_id!(quote(node_id), port!(id!(port_id(i))));
            self.sorts.insert(value.1.clone());
            let (child_node_id, child_subgraph_id) = match &value.0 {
                // Functions should point to one of the function nodes of the e-class.
                // We rotate between them in a round robin, to balance where the edgs point to
                ExportedValue::EClass(e_class_id) => {
                    let node_ids = self.eclasses.entry(*e_class_id).or_default();
                    // Lookup the first node id for this eclass id
                    let node_id = node_ids.front().unwrap().clone();
                    // Rotate the node ids so that we can point to different ones each time for better graph layout
                    node_ids.rotate_left(1);
                    (node_id, format!("cluster_{}", e_class_id))
                }
                // For primitives, we make a new node for each input, based on the function name and the index of the input
                ExportedValue::Prim(name, inner, _) => {
                    let child_node_id = format!("{}_{}", node_id, i);
                    let child_subgraph_id = format!("cluster_{}", child_node_id);
                    let subgraph_node = self.add_node(&child_node_id, name, inner);
                    self.add_value_subgraph(&child_subgraph_id, &value.1, vec![subgraph_node]);
                    (child_node_id, child_subgraph_id)
                }
            };
            let target = node_id!(quote(&child_node_id));
            self.edges
                .push(edge!(source => target; EdgeAttributes::lhead(quote(&child_subgraph_id))));
        }
        node!(quoted_node_id;NodeAttributes::label(html_label))
    }

    /// Adds a new subgraph, panics if it already exists. Used for values
    fn add_value_subgraph(&mut self, subgraph_id: &str, sort: &str, nodes: Vec<Node>) {
        let is_eclass = false;
        if self.nodes.contains_key(subgraph_id) {
            panic!("Subgraph already exists")
        }
        self.nodes.insert(
            subgraph_id.to_string(),
            (sort.to_string(), is_eclass, nodes),
        );
    }
}

fn build_colors(sorts: &Sorts) -> SortColors {
    sorts
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i % 12 + 1))
        .collect::<HashMap<_, _>>()
}

fn build_graph(nodes: Nodes, edges: Edges, sort_colors: &SortColors) -> Graph {
    let mut stmts = configuration_statements();
    for (subgraph_id, (sort, is_eclass, nodes)) in nodes {
        let subgraph_style = if is_eclass {
            "dashed,rounded,filled"
        } else {
            "dotted,rounded,filled"
        };
        let color = sort_colors[&sort];
        // Nest in empty sub-graph so that we can use rank=same
        // https://stackoverflow.com/a/55562026/907060
        let quoted_subgraph_id = quote(&subgraph_id);
        let subgraph_stmts = nodes.into_iter().map(|s| stmt!(s)).collect();
        let s = stmt!(subgraph!(quoted_subgraph_id;
            // Disable label for now, to reduce size
            // NodeAttributes::label(subgraph_html_label(&sort)),
            attr!("fillcolor", color),
            GA::Graph(vec![GraphAttributes::style(quote(subgraph_style))]),
            subgraph!("", subgraph_stmts)
        ));
        stmts.push(s);
    }
    stmts.extend(edges.into_iter().map(|s| stmt!(s)));
    graph!(di id!(), stmts)
}

fn configuration_statements() -> Vec<Stmt> {
    vec![
        // Set to compound so we can have edge to clusters
        stmt!(GraphAttributes::compound(true)),
        // Set default sub-graph rank to be same so that all nodes in e-class are on same level
        stmt!(SubgraphAttributes::rank(rank::same)),
        stmt!(GraphAttributes::fontname("helvetica".to_string())),
        stmt!(GraphAttributes::fontsize(9.0)),
        stmt!(GraphAttributes::margin(3.0)),
        stmt!(GraphAttributes::nodesep(0.0)),
        stmt!(GraphAttributes::colorscheme("set312".to_string())),
        stmt!(GA::Edge(vec![EdgeAttributes::arrowsize(0.5)])),
        stmt!(GA::Node(vec![
            NodeAttributes::shape(shape::none),
            NodeAttributes::margin(0.0)
        ])),
        // Draw edges first, so that they are behind nodes
        stmt!(GraphAttributes::outputorder(outputorder::edgesfirst)),
    ]
}

impl ExportedCall {
    /// The ID of the graphviz node for the function call
    fn function_node_id(&self) -> String {
        format!("{}_{}", self.fn_name, self.input_hash)
    }
}

/// Adds double quotes and escapes the quotes in the string
fn quote(s: &str) -> String {
    format!("{:?}", s)
}

/// Returns an html label for the node with the function name and ports for each argumetn
fn html_label(label: &str, n_args: usize) -> String {
    format!(
        "<<TABLE BGCOLOR=\"white\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"0\" style=\"rounded\"><tr><td CELLPADDING=\"4\" WIDTH=\"30\" HEIGHT=\"30\" colspan=\"{}\">{}</td></tr>{}</TABLE>>",
        n_args,
        Escape(label),
        (if n_args == 0 {
            "".to_string()
        } else {
            format!(
                "<TR>{}</TR>",
                (0..n_args)
                    .map(|i| format!("<TD PORT=\"{}\"></TD>", port_id(i)))
                    .collect::<Vec<String>>()
                    .join("")
            )
        })
    )
}
// fn subgraph_html_label(label: &str) -> String {
//     format!("<<TABLE CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"0\" border=\"0\"><tr><td><i>{}</i></td></tr></TABLE>>", Escape(label))
// }

fn port_id(i: usize) -> String {
    format!("i{}", i)
}

// Copied from https://doc.rust-lang.org/stable/nightly-rustc/src/rustdoc/html/escape.rs.html#10

/// Wrapper struct which will emit the HTML-escaped version of the contained
/// string when passed to a format string.
pub(crate) struct Escape<'a>(pub &'a str);

impl<'a> fmt::Display for Escape<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Because the internet is always right, turns out there's not that many
        // characters to escape: http://stackoverflow.com/questions/7381974
        let Escape(s) = *self;
        let pile_o_bits = s;
        let mut last = 0;
        for (i, ch) in s.char_indices() {
            let s = match ch {
                '>' => "&gt;",
                '<' => "&lt;",
                '&' => "&amp;",
                '\'' => "&#39;",
                '"' => "&quot;",
                _ => continue,
            };
            fmt.write_str(&pile_o_bits[last..i])?;
            fmt.write_str(s)?;
            // NOTE: we only expect single byte characters here - which is fine as long as we
            // only match single byte characters
            last = i + 1;
        }

        if last < s.len() {
            fmt.write_str(&pile_o_bits[last..])?;
        }
        Ok(())
    }
}
