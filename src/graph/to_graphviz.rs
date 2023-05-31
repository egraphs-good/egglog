use crate::util::{HashMap, HashSet};
use std::collections::VecDeque;

use super::*;
use graphviz_rust::{
    attributes::*,
    dot_generator::*,
    dot_structures::{
        Edge, EdgeTy, Graph, GraphAttributes as GA, Id, Node, NodeId, Port, Stmt, Subgraph, Vertex,
    },
};

/// Implement conversion of Graph to graphviz Graph
/// Places all equivalent nodes in the same cluster
/// For e-nodes these are are all in the same e-class
/// For
pub(crate) fn to_graphviz(g: ExportedGraph) -> Graph {
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

    // Use VecDeque so we can rotate efficiently
    let mut e_class_id_to_node_ids = g.iter().fold(
        HashMap::<usize, VecDeque<String>>::default(),
        |mut acc, call| {
            if let ExportedValue::EClass(id) = call.output {
                let node_id = format!("{}_{}", call.fn_name, call.input_hash);
                acc.entry(id).or_default().push_back(node_id);
            }
            acc
        },
    );
    // Get the node id for a given e-class id
    let mut get_enode_id = |e_class_id| -> String {
        let node_ids = e_class_id_to_node_ids.entry(e_class_id).or_default();
        // Lookup the first node id for this eclass id
        let node_id = node_ids.front().unwrap().clone();
        // Rotate the node ids so that we can point to different ones each time for better graph layout
        node_ids.rotate_left(1);
        node_id
    };

    // Map from subgraph id to map of node id to label
    let mut subgraph_to_node_to_label = HashMap::<String, HashMap<String, String>>::default();
    // Nodes to process and add
    let mut values_to_add = VecDeque::<ExportedValue>::default();
    // Nodes already added to the values_to_add queue
    let mut added_value: HashSet<ExportedValue> = HashSet::default();

    let mut add_value = |value: &ExportedValue| {
        if !added_value.contains(value) {
            values_to_add.push_back(value.clone());
            added_value.insert(value.clone());
        }
    };

    let mut add_node =
        |subgraph_value: &ExportedValue, node_id: String, name: String, n_args: usize| {
            subgraph_to_node_to_label
                .entry(subgraph_id(subgraph_value))
                .or_default()
                .insert(node_id, html_label(name, n_args));
        };

    let mut gen_value_id = |value: &ExportedValue| match value {
        ExportedValue::EClass(id) => get_enode_id(*id),
        ExportedValue::Prim(p) => prim_id(p),
        ExportedValue::Container {
            name,
            inner: _,
            inner_hash,
        } => container_id(name, *inner_hash),
    };

    let mut add_edge =
        |source_node_id: String, source_index: usize, target_value: &ExportedValue| {
            let source = node_id!(quote(source_node_id), port!(id!(port_id(source_index))));
            let target = node_id!(quote(gen_value_id(target_value)));
            let target_subgraph_id = subgraph_id(target_value);
            let edge = edge!(source => target; EdgeAttributes::lhead(target_subgraph_id));
            statements.push(stmt!(edge));
        };

    // Create subgraphs and nodes
    for ExportedCall {
        fn_name,
        inputs,
        output,
        input_hash,
    } in g
    {
        // Add function node
        let fn_id = format!("{}_{}", fn_name, input_hash);

        add_node(&output, fn_id.clone(), fn_name, inputs.len());
        add_value(&output);
        // Add edges from function node to input nodes
        for (i, input) in inputs.into_iter().enumerate() {
            add_edge(fn_id.clone(), i, &input);
            add_value(&input);
        }
    }

    // Add nodes for primitive values. Skip e-classes, since these appear as subgraphs
    while let Some(value) = values_to_add.pop_front() {
        match &value {
            ExportedValue::EClass(_) => {}
            ExportedValue::Prim(p) => add_node(&value, prim_id(p), p.into(), 0),
            ExportedValue::Container {
                name,
                inner,
                inner_hash,
            } => {
                let value_id = container_id(name, *inner_hash);
                add_node(&value, value_id.clone(), name.to_string(), inner.len());
                for (i, inner_value) in inner.iter().enumerate() {
                    add_value(&inner_value);
                    add_edge(value_id.clone(), i, inner_value);
                }
            }
        }
    }

    // Export each subgraph to nodes
    for (subgraph_id, node_id_to_label) in subgraph_to_node_to_label {
        let subgraph_stmts = node_id_to_label
            .into_iter()
            .map(|(node_id, label)| {
                let node_id = quote(node_id);
                stmt!(node!(node_id;NodeAttributes::label(label)))
            })
            .collect();
        // Nest in empty sub-graph so that we can use rank=same
        // https://stackoverflow.com/a/55562026/907060
        statements.push(stmt!(subgraph!(subgraph_id; subgraph!("", subgraph_stmts))));
    }
    graph!(di id!(), statements)
}

fn subgraph_id(exported_value: &ExportedValue) -> String {
    quote(match exported_value {
        ExportedValue::EClass(eclass_id) => format!("cluster_eclass_{}", eclass_id),
        ExportedValue::Prim(p) => format!("cluster_prim_{}", p),
        ExportedValue::Container {
            name,
            inner: _,
            inner_hash,
        } => format!("cluster_container_{}_{}", name, inner_hash),
    })
}

fn prim_id(prim: &str) -> String {
    format!("prim_{}", prim)
}

fn container_id(name: &str, inner_hash: Hash) -> String {
    format!("container_{}_{}", name, inner_hash)
}

/// Adds double quotes and
fn quote(s: String) -> String {
    format!("{:?}", s)
}

/// Returns an html label for the node with the function name and ports for each argumetn
fn html_label(label: String, n_args: usize) -> String {
    format!(
        "<<TABLE CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"0\" style=\"rounded\"><tr><td CELLPADDING=\"4\" WIDTH=\"30\" HEIGHT=\"30\" colspan=\"{}\">{}</td></tr>{}</TABLE>>",
        n_args,
        label,
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

fn port_id(i: usize) -> String {
    format!("i{}", i)
}
