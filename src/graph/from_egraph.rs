use super::*;
use crate::{ast::Id, function::table::hash_values, ArcSort, EGraph, Value};

// Limit the number of functions and calls to avoid blowing up the size of the graph
const MAX_FUNCTIONS: usize = 40;
const MAX_CALLS_PER_FUNCTION: usize = 40;

pub(crate) fn graph_from_egraph(egraph: &EGraph) -> ExportedGraph {
    egraph
        .functions
        .values()
        // Only include functions with a non temporary name
        .filter(|f| !is_temp_name(f.decl.name.to_string()))
        // Map each function to its calls
        .map(|function| {
            function
                .nodes
                .vals
                .iter()
                .filter(|(i, _)| i.live())
                .take(MAX_CALLS_PER_FUNCTION)
                .map(|(input, output)| {
                    let input_values = input.data();
                    ExportedCall {
                        fn_name: function.decl.name.to_string(),
                        inputs: input_values
                            .iter()
                            .map(|v| export_value_with_sort(egraph, *v))
                            .collect(),
                        output: export_value_with_sort(egraph, output.value),
                        input_hash: hash_values(input_values),
                    }
                })
                .collect::<Vec<_>>()
        })
        // Filter out functions with no calls
        .filter(|f| !f.is_empty())
        .take(MAX_FUNCTIONS)
        .flatten()
        .collect()

}

/// Returns true if the name is in the form v{digits}___
/// like v78___
fn is_temp_name(name: String) -> bool {
    name.starts_with('v') && name.ends_with("___") && name[1..name.len() - 3].parse::<u32>().is_ok()
}

fn export_value_with_sort(egraph: &EGraph, value: Value) -> ExportedValueWithSort {
    let sort = egraph.get_sort(&value).unwrap();
    ExportedValueWithSort(export_value(egraph, value, sort), sort.name().to_string())
}

fn export_value(egraph: &EGraph, value: Value, sort: &ArcSort) -> ExportedValue {
    if sort.is_eq_sort() {
        let id = value.bits as usize;
        let canonical: usize = egraph.unionfind.find(Id::from(id)).into();
        ExportedValue::EClass(canonical)
    } else {
        let inner_values: Vec<Value> = sort
            .inner_values(&value)
            .into_iter()
            .map(|(_, v)| v)
            .collect();

        // If this is a container sort, we just need to print the name
        // Otherwise, we need to print the value
        let str = if sort.is_container_sort() {
            sort.name().to_string()
        } else {
            sort.make_expr(egraph, value).to_string()
        };
        let inner_hash = hash_values(&inner_values);
        let inner_with_sorts = inner_values
            .into_iter()
            .map(|v| export_value_with_sort(egraph, v))
            .collect();
        ExportedValue::Prim(str, inner_with_sorts, inner_hash)
    }
}
