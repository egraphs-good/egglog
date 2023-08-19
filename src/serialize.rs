use ordered_float::NotNan;
use std::collections::VecDeque;

use crate::{
    ast::{FunctionDecl, Id},
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
}

impl Default for SerializeConfig {
    fn default() -> Self {
        SerializeConfig {
            max_functions: Some(40),
            max_calls_per_function: Some(40),
            include_temporary_functions: false,
        }
    }
}

impl EGraph {
    /// Serialize the egraph into a format that can be read by the egraph-serialize crate.
    ///
    /// There are multiple different semantically valid ways to do this.
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
        // First collect a list of all the calls we want to serialize, into the function decl, the inputs, and the output, and if its an eq sort
        let all_calls: Vec<(&FunctionDecl, &ValueVec, &Value, egraph_serialize::NodeId)> = self
            .functions
            .values()
            .filter(|f| {
                config.include_temporary_functions || !self.is_temp_name(f.decl.name.to_string())
            })
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
        let mut node_ids: NodeIDs = all_calls
            .iter()
            .filter_map(|(_decl, _input, output, node_id)| {
                if self.get_sort(output).unwrap().is_eq_sort() {
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
            let eclass = self.serialize_value(&mut egraph, &mut node_ids, output).0;
            let children: Vec<_> = input
                .iter()
                // Filter out children which don't have an ID, meaning that we skipped emitting them due to size constraints
                .filter_map(|v| self.serialize_value(&mut egraph, &mut node_ids, v).1)
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
        egraph
    }

    /// Serialize the value and return the eclass and node ID
    /// If this is a primitive value, we will add the node to the data, but if it is an eclass, we will not
    fn serialize_value(
        &self,
        egraph: &mut egraph_serialize::EGraph,
        node_ids: &mut NodeIDs,
        value: &Value,
    ) -> (egraph_serialize::ClassId, Option<egraph_serialize::NodeId>) {
        let sort = self.get_sort(value).unwrap();
        let (class_id, node_id): (egraph_serialize::ClassId, Option<egraph_serialize::NodeId>) =
            if sort.is_eq_sort() {
                let id: usize = value.bits as usize;
                let canonical: usize = self.unionfind.find(Id::from(id)).into();
                let class_id: egraph_serialize::ClassId = canonical.to_string().into();
                (class_id.clone(), get_node_id(node_ids, class_id))
            } else {
                let sort_name = sort.name().to_string();
                let node_id_str = format!("{}-{}", sort_name, hash_values(vec![*value].as_slice()));
                let (eclass, node_id): (egraph_serialize::ClassId, egraph_serialize::NodeId) =
                    (node_id_str.clone().into(), node_id_str.into());
                // Add node for value
                {
                    let children: Vec<egraph_serialize::NodeId> = sort
                        .inner_values(value)
                        .into_iter()
                        .filter_map(|(_, v)| self.serialize_value(egraph, node_ids, &v).1)
                        .collect();
                    // If this is a container sort, use the name, otherwise use the value
                    let op: String = if sort.is_container_sort() {
                        log::warn!("{} is a container sort", sort.name());
                        sort.name().to_string()
                    } else {
                        sort.make_expr(self, *value).to_string()
                    };
                    egraph.nodes.insert(
                        node_id.clone(),
                        egraph_serialize::Node {
                            op,
                            eclass: eclass.clone(),
                            cost: NotNan::new(0.0).unwrap(),
                            children,
                        },
                    );
                };
                (eclass, Some(node_id))
            };
        egraph.class_data.insert(
            class_id.clone(),
            egraph_serialize::ClassData {
                typ: Some(sort.name().to_string()),
            },
        );
        (class_id, node_id)
    }

    /// Returns true if the name is in the form v{digits}__
    /// like v78___
    ///
    /// Checks for pattern created by Desugar.get_fresh
    fn is_temp_name(&self, name: String) -> bool {
        let number_underscores = self.proof_state.desugar.number_underscores;
        let res = name.starts_with('v')
            && name.ends_with("_".repeat(number_underscores).as_str())
            && name[1..name.len() - number_underscores]
                .parse::<u32>()
                .is_ok();
        res
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
