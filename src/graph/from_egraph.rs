use super::*;
use crate::{ast::Id, function::table::hash_values, ArcSort, EGraph, Value};

pub(crate) fn graph_from_egraph(egraph: &EGraph) -> ExportedGraph {
    let mut calls = ExportedGraph::default();
    for (_id, function) in egraph.functions.iter() {
        let name = function.decl.name.to_string();
        // Skip temporary names
        if is_temp_name(name.clone()) {
            continue;
        }
        for (input, output) in function.nodes.vals.iter() {
            if !input.live() {
                continue;
            }
            let input_values = input.data();
            let fn_call = ExportedCall {
                fn_name: name.clone(),
                inputs: input_values
                    .iter()
                    .map(|v| export_value_with_sort(egraph, *v))
                    .collect(),
                output: export_value_with_sort(egraph, output.value),
                input_hash: hash_values(input_values),
            };
            calls.push(fn_call);
        }
    }
    calls
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
