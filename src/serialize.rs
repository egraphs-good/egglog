use ordered_float::NotNan;
use std::collections::VecDeque;

use crate::{
    ast::{Id, ResolvedFunctionDecl},
    function::{table::hash_values, ValueVec},
    util::HashMap,
    EGraph, Value,
};

pub struct SerializeConfig {
    // Maximumum number of functions to include in the serialized graph, any after this will be discarded
    pub max_functions: Option<usize>,
    // Maximum number of calls to include per function, any after this will be discarded
    pub max_calls_per_function: Option<usize>,
    // Whether to include temporary functions in the serialized graph
    pub include_temporary_functions: bool,
    // Whether to split primitive output values into their own e-classes with the function
    pub split_primitive_outputs: bool,
    // Root eclasses to include in the output
    pub root_eclasses: Vec<Value>,
}

/// Default is used for visualizations and limits number of functions and calls
impl Default for SerializeConfig {
    fn default() -> Self {
        SerializeConfig {
            max_functions: Some(40),
            max_calls_per_function: Some(40),
            include_temporary_functions: false,
            split_primitive_outputs: false,
            root_eclasses: vec![],
        }
    }
}

impl EGraph {
    /// Serialize the egraph into a format that can be read by the egraph-serialize crate.
    ///
    /// There are multiple different semantically valid ways to do this. This is how this implementation does it:
    ///
    /// For node costs:
    /// - Primitives: 1.0
    /// - Function without costs: 1.0
    /// - Function with costs: the cost
    ///
    /// For node IDs:
    /// - Functions: Function name + hash of input values
    /// - Args which are eq sorts: Choose one ID from the e-class, distribute roughly evenly.
    /// - Args and outputs values which are primitives: Sort name + hash of value
    ///   Notes: If `split_primitive_returns` is true, then each output value will be the function node id + `-output`
    ///
    /// For e-classes:
    /// - Eq sorts: Use the canonical ID of the e-class
    /// - Primitives: Use the node ID
    ///
    /// This is to achieve the following properties:
    /// - Equivalent primitive values will show up once in the e-graph.
    /// - Functions which return primitive values will be added to the e-class of that value.
    /// - Nodes will have consistant IDs throughout execution of e-graph (used for animating changes in the visualization)
    /// - Edges in the visualization will be well distributed (used for animating changes in the visualization)
    ///   (Note that this will be changed in `<https://github.com/egraphs-good/egglog/pull/158>` so that edges point to exact nodes instead of looking up the e-class)
    pub fn serialize(&self, config: SerializeConfig) -> egraph_serialize::EGraph {
        // First collect a list of all the calls we want to serialize as (function decl, inputs, the output, the node id)
        let all_calls: Vec<(
            &ResolvedFunctionDecl,
            &ValueVec,
            &Value,
            egraph_serialize::NodeId,
        )> = self
            .functions
            .values()
            .map(|function| {
                function
                    .nodes
                    .vals
                    .iter()
                    .filter(|(i, _)| i.live())
                    .take(config.max_calls_per_function.unwrap_or(usize::MAX))
                    .map(|(input, output)| {
                        (
                            &function.decl,
                            &input.data,
                            &output.value,
                            format!("{}-{}", function.decl.name, hash_values(&input.data)).into(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            // Filter out functions with no calls
            .filter(|f| !f.is_empty())
            .take(config.max_functions.unwrap_or(usize::MAX))
            .flatten()
            .collect();

        // Then create a mapping from each canonical e-class ID to the set of node IDs in that e-class
        // Note that this is only for e-classes, primitives have e-classes equal to their node ID
        // This is for when we need to find what node ID to use for an edge to an e-class, we can rotate them evenly
        // amoung all possible options.
        let mut node_ids: NodeIDs = all_calls
            .iter()
            .filter_map(|(_decl, _input, output, node_id)| {
                if self.get_sort_from_value(output).unwrap().is_eq_sort() {
                    let id = output.bits as usize;
                    let canonical: usize = self.unionfind.find(Id::from(id)).into();
                    let canonical_id: egraph_serialize::ClassId = canonical.to_string().into();
                    Some((canonical_id, node_id))
                } else {
                    None
                }
            })
            .fold(HashMap::default(), |mut acc, (canonical_id, node_id)| {
                acc.entry(canonical_id)
                    .or_insert_with(VecDeque::new)
                    .push_back(node_id.clone());
                acc
            });

        let mut egraph = egraph_serialize::EGraph::default();
        for (decl, input, output, node_id) in all_calls {
            let prim_node_id = if config.split_primitive_outputs {
                Some(format!("{}-value", node_id.clone()))
            } else {
                None
            };
            let eclass = self
                .serialize_value(&mut egraph, &mut node_ids, output, prim_node_id)
                .0;
            let children: Vec<_> = input
                .iter()
                // Filter out children which don't have an ID, meaning that we skipped emitting them due to size constraints
                .filter_map(|v| self.serialize_value(&mut egraph, &mut node_ids, v, None).1)
                .collect();
            egraph.nodes.insert(
                node_id,
                egraph_serialize::Node {
                    op: decl.name.to_string(),
                    eclass,
                    cost: NotNan::new(decl.cost.unwrap_or(1) as f64).unwrap(),
                    children,
                },
            );
        }

        let roots = config
            .root_eclasses
            .iter()
            .map(|v| self.serialize_value(&mut egraph, &mut node_ids, v, None).0)
            .collect();
        egraph.root_eclasses = roots;

        egraph
    }

    /// Serialize the value and return the eclass and node ID
    /// If this is a primitive value, we will add the node to the data, but if it is an eclass, we will not
    /// When this is called on the output of a node, we only use the e-class to know which e-class its a part of
    /// When this is called on an input of a node, we only use the node ID to know which node to point to.
    fn serialize_value(
        &self,
        egraph: &mut egraph_serialize::EGraph,
        node_ids: &mut NodeIDs,
        value: &Value,
        // The node ID to use for a primitve value, if this is None, use the hash of the value and the sort name
        // Set iff `split_primitive_outputs` is set and this is an output of a function.
        prim_node_id: Option<String>,
    ) -> (egraph_serialize::ClassId, Option<egraph_serialize::NodeId>) {
        let sort = self.get_sort_from_value(value).unwrap();
        let (class_id, node_id): (egraph_serialize::ClassId, Option<egraph_serialize::NodeId>) =
            if sort.is_eq_sort() {
                let id: usize = value.bits as usize;
                let canonical: usize = self.unionfind.find(Id::from(id)).into();
                let class_id: egraph_serialize::ClassId = canonical.to_string().into();
                (class_id.clone(), get_node_id(node_ids, class_id))
            } else {
                let (class_id, node_id): (egraph_serialize::ClassId, egraph_serialize::NodeId) =
                    if let Some(node_id) = prim_node_id {
                        (node_id.clone().into(), node_id.into())
                    } else {
                        let sort_name = sort.name().to_string();
                        let node_id_str =
                            format!("{}-{}", sort_name, hash_values(vec![*value].as_slice()));
                        (node_id_str.clone().into(), node_id_str.into())
                    };
                // Add node for value
                {
                    let children: Vec<egraph_serialize::NodeId> = sort
                        .inner_values(value)
                        .into_iter()
                        .filter_map(|(_, v)| self.serialize_value(egraph, node_ids, &v, None).1)
                        .collect();
                    // If this is a container sort, use the name, otherwise use the value
                    let op: String = if sort.is_container_sort() {
                        log::warn!("{} is a container sort", sort.name());
                        sort.serialized_name(value).to_string()
                    } else {
                        sort.make_expr(self, *value).1.to_string()
                    };
                    egraph.nodes.insert(
                        node_id.clone(),
                        egraph_serialize::Node {
                            op,
                            eclass: class_id.clone(),
                            cost: NotNan::new(0.0).unwrap(),
                            children,
                        },
                    );
                };
                (class_id, Some(node_id))
            };
        egraph.class_data.insert(
            class_id.clone(),
            egraph_serialize::ClassData {
                typ: Some(sort.name().to_string()),
            },
        );
        (class_id, node_id)
    }
}

type NodeIDs = HashMap<egraph_serialize::ClassId, VecDeque<egraph_serialize::NodeId>>;

/// Returns the node ID for the given class ID, rotating the queue
fn get_node_id(
    node_ids: &mut HashMap<egraph_serialize::ClassId, VecDeque<egraph_serialize::NodeId>>,
    class_id: egraph_serialize::ClassId,
) -> Option<egraph_serialize::NodeId> {
    let node_ids = node_ids.get_mut(&class_id)?;
    node_ids.rotate_left(1);
    Some(node_ids.front().unwrap().clone())
}
