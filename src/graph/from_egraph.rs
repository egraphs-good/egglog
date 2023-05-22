use crate::{EGraph, Value, ast::Id};
use super::*;

pub (crate) fn graph_from_egraph(egraph: &EGraph) -> Graph {
    let mut prim_outputs = vec![];
    let mut eclasses = HashMap::<String, Vec<FnCall>>::default();
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
                input_values
                    .iter()
                    .map(|v| arg_from_value(egraph, *v))
                    .collect(),
            );

            match arg_from_value(egraph, output_value) {
                Arg::Eq(parent_id) => {
                    eclasses.entry(parent_id).or_default().push(fn_call);
                }
                Arg::Prim(prim_value) => {
                    prim_outputs.push(PrimOutput(fn_call, prim_value))
                }
            }
        }
    }
    Graph {
        prim_outputs,
        eclasses,
    }
}


fn arg_from_value(egraph: &EGraph, value: Value) -> Arg {
    let sort = egraph.get_sort(&value).unwrap();
    if sort.is_eq_sort() {
        let parent_id: usize = egraph.unionfind.find(Id::from(value.bits as usize)).into();
        let parent_id_string = format!("{}", parent_id);
        Arg::Eq(parent_id_string)
    } else {
        let expr = sort.make_expr(egraph, value);
        let prim_value = from_expr(&expr);
        Arg::Prim(prim_value)
    }
}
