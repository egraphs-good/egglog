use ordered_float::NotNan;
use std::collections::VecDeque;

use crate::{
    ast::ResolvedFunctionDecl, function::table::hash_values, util::HashMap, EGraph, Value,
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

/// Default is used for exporting JSON and will output all nodes.
impl Default for SerializeConfig {
    fn default() -> Self {
        SerializeConfig {
            max_functions: None,
            max_calls_per_function: None,
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
    /// - Omitted nodes: infinite
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
            &[Value],
            &Value,
            egraph_serialize::NodeId,
        )> = self
            .functions
            .values()
            .filter(|function| !function.decl.ignore_viz)
            .map(|function| {
                function
                    .nodes
                    .iter(true)
                    .take(config.max_calls_per_function.unwrap_or(usize::MAX))
                    .map(|(input, output)| {
                        (
                            &function.decl,
                            input,
                            &output.value,
                            format!("{}-{}", function.decl.name, hash_values(input)).into(),
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
                    Some((self.value_to_class_id(output), node_id))
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
            // If we are splitting primitive outputs, then we will use the function node ID as the e-class for the output, so
            // that even if two functions have the same primitive output, they will be in different e-classes.
            let eclass = if config.split_primitive_outputs
                && !self.get_sort_from_value(output).unwrap().is_eq_sort()
            {
                format!("{}-value", node_id.clone()).into()
            } else {
                self.value_to_class_id(output)
            };
            self.serialize_value(&mut egraph, &mut node_ids, output, &eclass);
            let children: Vec<_> = input
                .iter()
                .map(|v| {
                    self.serialize_value(&mut egraph, &mut node_ids, v, &self.value_to_class_id(v))
                })
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

        egraph.root_eclasses = config
            .root_eclasses
            .iter()
            .map(|v| self.value_to_class_id(v))
            .collect();

        egraph
    }

    /**
     * Gets the serialized class ID for a value.
     */
    pub fn value_to_class_id(&self, value: &Value) -> egraph_serialize::ClassId {
        // Canonicalize the value first so that we always use the canonical e-class ID
        let sort = self.get_sort_from_value(value).unwrap();
        let mut value = *value;
        sort.canonicalize(&mut value, &self.unionfind);
        format!("{}-{}", value.tag, value.bits).into()
    }

    /**
     * Gets the value for a serialized class ID.
     */
    pub fn class_id_to_value(&self, eclass_id: &egraph_serialize::ClassId) -> Value {
        let s = eclass_id.to_string();
        let (tag, bits) = s.split_once('-').unwrap();
        Value {
            tag: tag.into(),
            bits: bits.parse().unwrap(),
        }
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
        class_id: &egraph_serialize::ClassId,
    ) -> egraph_serialize::NodeId {
        let sort = self.get_sort_from_value(value).unwrap();
        let node_id = if sort.is_eq_sort() {
            let node_ids = node_ids.entry(class_id.clone()).or_insert_with(|| {
                // If we don't find node IDs for this class, it means that all nodes for it were omitted due to size constraints
                // In this case, add a dummy node in this class to represent the missing nodes
                let node_id = egraph_serialize::NodeId::from(format!("{}-dummy", class_id));
                egraph.nodes.insert(
                    node_id.clone(),
                    egraph_serialize::Node {
                        op: "[...]".to_string(),
                        eclass: class_id.clone(),
                        cost: NotNan::new(f64::INFINITY).unwrap(),
                        children: vec![],
                    },
                );
                VecDeque::from(vec![node_id])
            });
            node_ids.rotate_left(1);
            node_ids.front().unwrap().clone()
        } else {
            let node_id: egraph_serialize::NodeId = class_id.to_string().into();
            // Add node for value
            {
                // Children will be empty unless this is a container sort
                let children: Vec<egraph_serialize::NodeId> = sort
                    .inner_values(value)
                    .into_iter()
                    .map(|(_, v)| {
                        self.serialize_value(egraph, node_ids, &v, &self.value_to_class_id(&v))
                    })
                    .collect();
                // If this is a container sort, use the name, otherwise use the value
                let op = if sort.is_container_sort() {
                    sort.serialized_name(value).to_string()
                } else {
                    sort.make_expr(self, *value).1.to_string()
                };
                egraph.nodes.insert(
                    node_id.clone(),
                    egraph_serialize::Node {
                        op,
                        eclass: class_id.clone(),
                        cost: NotNan::new(1.0).unwrap(),
                        children,
                    },
                );
            };
            node_id
        };
        egraph.class_data.insert(
            class_id.clone(),
            egraph_serialize::ClassData {
                typ: Some(sort.name().to_string()),
            },
        );
        node_id
    }
}

type NodeIDs = HashMap<egraph_serialize::ClassId, VecDeque<egraph_serialize::NodeId>>;
