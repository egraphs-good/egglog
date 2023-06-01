pub(crate) mod from_egraph;
pub(crate) mod to_graphviz;

type Offset = usize;
type EClassID = Offset;
type Hash = u64;

/// Exposed graph structure which can be used to print/visualize the state of the e-graph.
pub(crate) type ExportedGraph = Vec<ExportedCall>;

#[derive(Debug)]
pub(crate) struct ExportedCall {
    fn_name: String,
    inputs: Vec<ExportedValueWithSort>,
    output: ExportedValueWithSort,
    /// Hash of arguments
    input_hash: Hash,
}

#[derive(Debug, Hash, Clone, PartialEq, Eq)]
pub(crate) struct ExportedValueWithSort(ExportedValue, String);

/// An argument is either a primitive value or a reference to a eclass
#[derive(Debug, Hash, Clone, PartialEq, Eq)]
pub(crate) enum ExportedValue {
    /// A primitive value, i.e. int, float, String
    Prim (String),
    /// A container sort, i.e. Vec, Map, Set
    Container {
        name: String,
        inner: Vec<ExportedValueWithSort>,
        inner_hash: Hash,
    },
    /// A reference to an eclass
    EClass(EClassID),
}
