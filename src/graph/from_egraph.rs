use super::*;
use crate::{ast::Id, function::table::hash_values, EGraph, Value};

pub(crate) fn graph_from_egraph(egraph: &EGraph) -> Graph {
    let mut graph = Graph::default();
    for (_id, function) in egraph.functions.iter() {
        let name = function.decl.name.to_string();
        // Skip generated names
        if name.ends_with("___") {
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
