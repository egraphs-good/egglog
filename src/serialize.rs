use crate::{extract::Extractor, util::HashMap, *};
use ordered_float::NotNan;
use std::collections::VecDeque;

pub struct SerializeConfig {
    // Maximumum number of functions to include in the serialized graph, any after this will be discarded
    pub max_functions: Option<usize>,
    // Maximum number of calls to include per function, any after this will be discarded
    pub max_calls_per_function: Option<usize>,
    // Whether to include temporary functions in the serialized graph
    pub include_temporary_functions: bool,
    // Root eclasses to include in the output
    pub root_eclasses: Vec<(ArcSort, Value)>,
}

struct Serializer<'a> {
    extractor: Extractor<'a>,
    termdag: TermDag,
    node_ids: NodeIDs,
    result: egraph_serialize::EGraph,
}

/// Default is used for exporting JSON and will output all nodes.
impl Default for SerializeConfig {
    fn default() -> Self {
        SerializeConfig {
            max_functions: None,
            max_calls_per_function: None,
            include_temporary_functions: false,
            root_eclasses: vec![],
        }
    }
}

/// A node in the serialized egraph.
#[derive(PartialEq, Debug, Clone)]
pub enum SerializedNode {
    /// A user defined function call.
    Function {
        /// The name of the function.
        name: Symbol,
        /// The offset of the index in the table.
        /// This can be resolved to the output and input values with table.get_index(offset, true).
        offset: usize,
    },
    /// A primitive value.
    Primitive(Value),
    /// A dummy node used to represent omitted nodes.
    Dummy(Value),
    /// A node that was split into multiple e-classes.
    Split(Box<SerializedNode>),
}

impl SerializedNode {
    /// Returns true if the node is a primitive value.
    pub fn is_primitive(&self) -> bool {
        match self {
            SerializedNode::Primitive(_) => true,
            SerializedNode::Split(node) => node.is_primitive(),
            _ => false,
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
    ///
    /// For e-classes IDs:
    /// - tag and value of canonicalized value
    ///
    /// This is to achieve the following properties:
    /// - Equivalent primitive values will show up once in the e-graph.
    /// - Functions which return primitive values will be added to the e-class of that value.
    /// - Nodes will have consistant IDs throughout execution of e-graph (used for animating changes in the visualization)
    /// - Edges in the visualization will be well distributed (used for animating changes in the visualization)
    ///   (Note that this will be changed in `<https://github.com/egraphs-good/egglog/pull/158>` so that edges point to exact nodes instead of looking up the e-class)
    pub fn serialize(&mut self, config: SerializeConfig) -> egraph_serialize::EGraph {
        // First collect a list of all the calls we want to serialize as (function decl, inputs, the output, the node id)
        let all_calls: Vec<(
            &Function,
            &[Value],
            &TupleOutput,
            egraph_serialize::ClassId,
            egraph_serialize::NodeId,
        )> = self
            .functions
            .iter()
            .filter(|(_, function)| !function.decl.ignore_viz)
            .map(|(name, function)| {
                function
                    .nodes
                    .iter_range(0..function.nodes.num_offsets(), true)
                    .take(config.max_calls_per_function.unwrap_or(usize::MAX))
                    .map(|(offset, input, output)| {
                        (
                            function,
                            input,
                            output,
                            self.value_to_class_id(&function.schema.output, &output.value),
                            self.to_node_id(
                                None,
                                SerializedNode::Function {
                                    name: *name,
                                    offset,
                                },
                            ),
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
        let node_ids: NodeIDs = all_calls.iter().fold(
            HashMap::default(),
            |mut acc, (func, _input, _output, class_id, node_id)| {
                if func.schema.output.is_eq_sort() {
                    acc.entry(class_id.clone())
                        .or_default()
                        .push_back(node_id.clone());
                }
                acc
            },
        );

        let mut termdag = std::mem::take(&mut self.termdag);

        let mut cost_map = None;
        if self.cost_cache.is_some() {
            let cost_map_ts = std::mem::take(&mut self.cost_cache).unwrap();
            if cost_map_ts.0 == self.timestamp {
                cost_map = Some(cost_map_ts.1);
            }
        }

        let mut serializer = Serializer {
            extractor: Extractor::new(self, &mut termdag, cost_map),
            node_ids,
            result: egraph_serialize::EGraph::default(),
            termdag,
        };

        for (func, input, output, class_id, node_id) in all_calls {
            self.serialize_value(
                &mut serializer,
                &func.schema.output,
                &output.value,
                &class_id,
            );

            assert_eq!(input.len(), func.schema.input.len());
            let children: Vec<_> = input
                .iter()
                .zip(&func.schema.input)
                .map(|(v, sort)| {
                    self.serialize_value(&mut serializer, sort, v, &self.value_to_class_id(sort, v))
                })
                .collect();
            serializer.result.nodes.insert(
                node_id,
                egraph_serialize::Node {
                    op: func.decl.name.to_string(),
                    eclass: class_id.clone(),
                    cost: NotNan::new(func.decl.cost.unwrap_or(1) as f64).unwrap(),
                    children,
                    subsumed: output.subsumed,
                },
            );
        }

        serializer.result.root_eclasses = config
            .root_eclasses
            .iter()
            .map(|(sort, v)| self.value_to_class_id(sort, v))
            .collect();

        let termdag = serializer.termdag;
        let result = serializer.result;
        self.cost_cache = Some((self.timestamp, serializer.extractor.cost_map()));
        self.termdag = termdag;
        result
    }

    /// Gets the serialized class ID for a value.
    pub fn value_to_class_id(&self, sort: &ArcSort, value: &Value) -> egraph_serialize::ClassId {
        // Canonicalize the value first so that we always use the canonical e-class ID
        let mut value = *value;
        sort.canonicalize(&mut value, &self.unionfind);
        assert!(
            !sort.name().to_string().contains('-'),
            "Tag cannot contain '-' when serializing"
        );
        format!("{}-{}", sort.name(), value.bits).into()
    }

    /// Gets the value for a serialized class ID.
    pub fn class_id_to_value(&self, eclass_id: &egraph_serialize::ClassId) -> Value {
        let s = eclass_id.to_string();
        let (tag, bits) = s.split_once('-').unwrap();
        #[cfg(not(debug_assertions))]
        let _ = tag;
        Value {
            #[cfg(debug_assertions)]
            tag: tag.into(),
            bits: bits.parse().unwrap(),
        }
    }

    /// Gets the serialized node ID for the primitive, omitted, or function value.
    pub fn to_node_id(
        &self,
        sort: Option<&ArcSort>,
        node: SerializedNode,
    ) -> egraph_serialize::NodeId {
        match node {
            SerializedNode::Function { name, offset } => {
                assert!(sort.is_none());
                format!("function-{}-{}", offset, name).into()
            }
            SerializedNode::Primitive(value) => format!(
                "primitive-{}",
                self.value_to_class_id(sort.unwrap(), &value)
            )
            .into(),
            SerializedNode::Dummy(value) => {
                format!("dummy-{}", self.value_to_class_id(sort.unwrap(), &value)).into()
            }
            SerializedNode::Split(node) => format!("split-{}", self.to_node_id(sort, *node)).into(),
        }
    }

    /// Gets the serialized node for the node ID.
    pub fn from_node_id(&self, node_id: &egraph_serialize::NodeId) -> SerializedNode {
        let node_id = node_id.to_string();
        let (tag, rest) = node_id.split_once('-').unwrap();
        match tag {
            "function" => {
                let (offset, name) = rest.split_once('-').unwrap();
                SerializedNode::Function {
                    name: name.into(),
                    offset: offset.parse().unwrap(),
                }
            }
            "primitive" => {
                let class_id: egraph_serialize::ClassId = rest.into();
                SerializedNode::Primitive(self.class_id_to_value(&class_id))
            }
            "dummy" => {
                let class_id: egraph_serialize::ClassId = rest.into();
                SerializedNode::Dummy(self.class_id_to_value(&class_id))
            }
            "split" => {
                let (_offset, rest) = rest.split_once('-').unwrap();
                let node_id: egraph_serialize::NodeId = rest.into();
                SerializedNode::Split(Box::new(self.from_node_id(&node_id)))
            }
            _ => std::panic::panic_any(format!("Unknown node ID: {}-{}", tag, rest)),
        }
    }

    /// Serialize the value and return the eclass and node ID
    /// If this is a primitive value, we will add the node to the data, but if it is an eclass, we will not
    /// When this is called on the output of a node, we only use the e-class to know which e-class its a part of
    /// When this is called on an input of a node, we only use the node ID to know which node to point to.
    fn serialize_value(
        &self,
        serializer: &mut Serializer,
        sort: &ArcSort,
        value: &Value,
        class_id: &egraph_serialize::ClassId,
    ) -> egraph_serialize::NodeId {
        let node_id = if sort.is_eq_sort() {
            let node_ids = serializer
                .node_ids
                .entry(class_id.clone())
                .or_insert_with(|| {
                    // If we don't find node IDs for this class, it means that all nodes for it were omitted due to size constraints
                    // In this case, add a dummy node in this class to represent the missing nodes
                    let node_id = self.to_node_id(Some(sort), SerializedNode::Dummy(*value));
                    serializer.result.nodes.insert(
                        node_id.clone(),
                        egraph_serialize::Node {
                            op: "[...]".to_string(),
                            eclass: class_id.clone(),
                            cost: NotNan::new(f64::INFINITY).unwrap(),
                            children: vec![],
                            subsumed: false,
                        },
                    );
                    VecDeque::from(vec![node_id])
                });
            node_ids.rotate_left(1);
            node_ids.front().unwrap().clone()
        } else {
            let node_id = self.to_node_id(Some(sort), SerializedNode::Primitive(*value));
            // Add node for value
            {
                // Children will be empty unless this is a container sort
                let children: Vec<egraph_serialize::NodeId> = sort
                    .inner_values(value)
                    .into_iter()
                    .map(|(s, v)| {
                        self.serialize_value(serializer, &s, &v, &self.value_to_class_id(&s, &v))
                    })
                    .collect();
                // If this is a container sort, use the name, otherwise use the value
                let op = if sort.is_container_sort() {
                    sort.serialized_name(value).to_string()
                } else {
                    let (_, term) = sort
                            .extract_term(self, *value, &serializer.extractor, &mut serializer.termdag)
                            .expect("Extraction should be successful since extractor has been fully initialized");

                    serializer
                        .termdag
                        .term_to_expr(&term, Span::Panic)
                        .to_string()
                };
                serializer.result.nodes.insert(
                    node_id.clone(),
                    egraph_serialize::Node {
                        op,
                        eclass: class_id.clone(),
                        cost: NotNan::new(1.0).unwrap(),
                        children,
                        subsumed: false,
                    },
                );
            };
            node_id
        };
        serializer.result.class_data.insert(
            class_id.clone(),
            egraph_serialize::ClassData {
                typ: Some(sort.name().to_string()),
            },
        );
        node_id
    }
}

type NodeIDs = HashMap<egraph_serialize::ClassId, VecDeque<egraph_serialize::NodeId>>;
