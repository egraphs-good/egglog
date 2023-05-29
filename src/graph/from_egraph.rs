use super::*;
use crate::{ast::Id, function::table::hash_values, EGraph, Value};

pub(crate) fn graph_from_egraph(egraph: &EGraph) -> ExportedGraph {
    let mut graph = ExportedGraph::default();
    for (_id, function) in egraph.functions.iter() {
        let name = function.decl.name.to_string();
        // Keep temporary functions if proofs are enabled, because the proofs reference them
        if is_temp_name(name.clone()) && !egraph.proofs_enabled {
            continue;
        }
        for (input, output) in function.nodes.vals.iter() {
            if !input.live() {
                continue;
            }
            let input_values = input.data();
            let output_value = output.value;
            let fn_call = FnCall(
                Fn { name: name.clone() },
                // Collect all inputs/args
                input_values
                    .iter()
                    .map(|v| arg_from_value(egraph, *v))
                    .collect(),
                hash_values(input_values),
            );
            // Add output
            match arg_from_value(egraph, output_value) {
                Arg::Eq(id) => graph.eclasses.entry(id).or_default().push(fn_call),
                Arg::Prim(prim_value) => graph.prim_outputs.push(PrimOutput(fn_call, prim_value)),
            }
        }
    }
    graph
}

/// Returns true if the name is in the form v{digits}___
/// like v78___
fn is_temp_name(name: String) -> bool {
    name.starts_with('v') && name.ends_with("___") && name[1..name.len() - 3].parse::<u32>().is_ok()
}

fn arg_from_value(egraph: &EGraph, value: Value) -> Arg {
    let sort = egraph.get_sort(&value).unwrap();
    if sort.is_eq_sort() {
        let id = value.bits as usize;
        let canonical: usize = egraph.unionfind.find(Id::from(id)).into();
        Arg::Eq(canonical)
    } else {
        let expr = sort.make_expr(egraph, value);
        let prim_value = from_expr(&expr);
        Arg::Prim(prim_value)
    }
}
