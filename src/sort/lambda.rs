use std::{collections::BTreeMap, sync::Mutex};

use crate::typechecking::FuncType;

use super::*;

// #[derive(Debug, Hash, Eq, PartialEq, Copy, Clone)]
// struct ValueLambda {
//     var: Value,
//     body: Value,
// }

#[derive(Debug)]
pub struct LambdaSort {
    name: Symbol,
    input: ArcSort,
    output: ArcSort,
    // lambdas: Mutex<IndexSet<ValueLambda>>,
    // Map from vars to their ID
    // symbol_to_id: Mutex<BTreeMap<Symbol, Id>>,
    // Inverse map
    // id_to_symbol: Mutex<BTreeMap<Id, Symbol>>,
}

impl LambdaSort {
    pub fn make_sort(
        typeinfo: &mut TypeInfo,
        name: Symbol,
        args: &[Expr],
    ) -> Result<ArcSort, TypeError> {
        if let [Expr::Var(input), Expr::Var(output)] = args {
            // Ok(Arc::new(EqSort { name }))
            let input = typeinfo
                .sorts
                .get(input)
                .ok_or(TypeError::UndefinedSort(*input))?;
            let output = typeinfo
                .sorts
                .get(output)
                .ok_or(TypeError::UndefinedSort(*output))?;

            if !input.is_eq_sort() {
                return Err(TypeError::UndefinedSort(
                    "Lambdas must take EqSorts as input".into(),
                ));
            }
            if output.is_eq_container_sort() {
                return Err(TypeError::UndefinedSort(
                    "Lambdas returning other EqSort containers are not allowed".into(),
                ));
            }

            Ok(Arc::new(Self {
                name,
                input: input.clone(),
                output: output.clone(),
                // lambdas: Default::default(),
                // symbol_to_id: Default::default(),
                // id_to_symbol: Default::default(),
            }))
        } else {
            panic!()
        }
    }
}

impl Sort for LambdaSort {
    fn name(&self) -> Symbol {
        self.name
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    // Use this as an eqsort with more type info and a registration function
    fn is_eq_sort(&self) -> bool {
        true
    }

    fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
        EqSort { name: self.name }.canonicalize(value, unionfind)
    }

    fn register_primitives(self: Arc<Self>, typeinfo: &mut TypeInfo) {
        // typeinfo.add_primitive(Lambda {
        //     name: "lambda".into(),
        //     lambda: self.clone(),
        // });
        // typeinfo.func_types.insert(
        //     "lambda".into(),
        //     FuncType::new(
        //         vec![self.input.clone(), self.output.clone()],
        //         self.clone(),
        //         false,
        //     ),
        // );
        let string_sort: Arc<StringSort> = typeinfo.get_sort();
        typeinfo.func_types.insert(
            "var".into(),
            FuncType::new(vec![string_sort], self.input.clone(), false),
        );
        typeinfo.func_types.insert(
            "lambda".into(),
            FuncType::new(
                vec![self.input.clone(), self.output.clone()],
                self.clone(),
                false,
            ),
        );
        // typeinfo.add_primitive(Var {
        //     name: "var".into(),
        //     lambda: self.clone(),
        //     string: typeinfo.get_sort(),
        // });
        typeinfo.add_primitive(Apply {
            name: "apply".into(),
            lambda: self,
        });
    }

    fn make_expr(&self, egraph: &EGraph, value: Value) -> Expr {
        unimplemented!("No make_expr for EqSort {}", self.name)
    }

    fn register_egraph(self: Arc<Self>, egraph: &mut EGraph) {
        let string_sort: Arc<StringSort> = egraph.proof_state.type_info.get_sort();
        egraph
            .declare_function(
                &FunctionDecl {
                    name: "var".into(),
                    schema: Schema::new(vec![string_sort.name()], self.output.name()),
                    default: None,
                    merge: None,
                    merge_action: vec![],
                    cost: None,
                },
                false,
            )
            .expect("declaring var");
        egraph
            .declare_function(
                &FunctionDecl {
                    name: "lambda".into(),
                    schema: Schema::new(vec![self.input.name(), self.output.name()], self.name()),
                    default: None,
                    merge: None,
                    merge_action: vec![],
                    cost: None,
                },
                false,
            )
            .expect("declaring lambda");
    }
}

// impl IntoSort for ValueLambda {
//     type Sort = LambdaSort;
//     fn store(self, sort: &Self::Sort) -> Option<Value> {
//         let mut lambdas = sort.lambdas.lock().unwrap();
//         let (i, _) = lambdas.insert_full(self);
//         Some(Value {
//             tag: sort.name,
//             bits: i as u64,
//         })
//     }
// }

// impl FromSort for ValueLambda {
//     type Sort = LambdaSort;
//     fn load(sort: &Self::Sort, value: &Value) -> Self {
//         let lambdas = sort.lambdas.lock().unwrap();
//         *lambdas.get_index(value.bits as usize).unwrap()
//     }
// }

// struct Var {
//     name: Symbol,
//     lambda: Arc<LambdaSort>,
//     string: Arc<StringSort>,
// }

// impl PrimitiveLike for Var {
//     fn name(&self) -> Symbol {
//         self.name
//     }

//     fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
//         if let [sort] = types {
//             if sort.name() == self.string.name() {
//                 return Some(self.lambda.input.clone());
//             }
//         }
//         None
//     }

//     fn apply(&self, values: &[Value], egraph: &mut EGraph) -> Option<Value> {
//         let var = Symbol::load(&self.string, &values[0]);
//         let mut var_to_id = self.lambda.symbol_to_id.lock().unwrap();
//         let id = match var_to_id.entry(var) {
//             // If we have already saved a var for this ID used it
//             std::collections::btree_map::Entry::Occupied(o) => *o.get(),
//             // Otherwise, if we have the unionfind, make an ID and return it
//             // If we don't the unionfind, we are in type checking and return a dummy ID
//             std::collections::btree_map::Entry::Vacant(v) => egraph.map_or(Id::from(0), |e| {
//                 let id = e.unionfind.make_set();
//                 v.insert(id);
//                 self.lambda.id_to_symbol.lock().unwrap().insert(id, var);
//                 id
//             }),
//         };
//         Some(Value::from_id(self.lambda.input.name(), id))
//     }
// }

// struct Lambda {
//     name: Symbol,
//     lambda: Arc<LambdaSort>,
// }

// impl PrimitiveLike for Lambda {
//     fn name(&self) -> Symbol {
//         self.name
//     }

//     fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
//         if let [var_tp, body_tp] = types {
//             if var_tp.name() == self.lambda.input.name()
//                 && body_tp.name() == self.lambda.output.name()
//             {
//                 return Some(self.lambda.clone());
//             }
//         }
//         None
//     }

//     fn apply(&self, values: &[Value], _egraph: &mut EGraph) -> Option<Value> {
//         ValueLambda {
//             var: values[0],
//             body: values[1],
//         }
//         .store(&self.lambda)
//     }
// }

pub(crate) struct Apply {
    pub(crate) name: Symbol,
    pub(crate) lambda: Arc<LambdaSort>,
}

impl PrimitiveLike for Apply {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        // Types should be lambda and then input type and return outputs type
        if let [lambda_tp, input_tp] = types {
            if lambda_tp.name() == self.lambda.name() && input_tp.name() == self.lambda.input.name()
            {
                return Some(self.lambda.output.clone());
            }
        }
        None
    }

    fn apply(&self, values: &[Value], maybe_egraph: Option<&mut EGraph>) -> Option<Value> {
        let egraph = match maybe_egraph {
            Some(egraph) => egraph,
            None => {
                panic!("Cant use apply when creating a query")
            }
        };
        // let lambda = ValueLambda::load(&self.lambda, &values[0]);
        let lambda_value = values[0];
        let input_value = values[1];
        let lambda_name: Symbol = "lambda".into();
        let lambda_fn = egraph
            .functions
            .get_mut(&lambda_name)
            .expect("getting lambda fn");
        // Find lambda inputs which return this lambda value
        let (lambda_input, _) = lambda_fn
            .nodes
            .iter()
            .find(|(input, output)| output.value == lambda_value)
            .expect("finding lambda fn call");
        let var_value = lambda_input[0];
        let body_value = lambda_input[1];
        Some(
            substitute(
                egraph,
                &body_value,
                &var_value,
                &input_value,
                &mut HashMap::default(),
            )
            .unwrap_or(body_value),
        )
        // match egraph {
        //     None => Some(body),
        //     Some(e) => {
        //         // If we do have an e-graph, we need to substitute the var with the input
        //         // In body replace all instances of var_value with input_value

        //         Some(
        //             substitute(e, &body, &var_value, &input_value, &mut HashMap::default())
        //                 .unwrap_or(body),
        //         )
        //     }
        // }
    }
}

/// Substitutues instance of var_value in body with input_value
/// Returns a new value if any substitutions were made
fn substitute(
    egraph: &mut EGraph,
    body_value: &Value,
    var_value: &Value,
    input_value: &Value,
    // Mapping of values to their substituted values, so we don't end up in loops for cyclic graphs
    subtituted: &mut HashMap<Value, Value>,
) -> Option<Value> {
    println!("{:?}[{:?} -> {:?}]", body_value, var_value, input_value);
    if body_value == var_value {
        println!("Found var");
        return Some(*input_value);
    }
    if body_value == input_value {
        println!("Found input");
        return None;
    }
    if subtituted.contains_key(body_value) {
        println!("Found already substituted");
        return Some(*subtituted.get(body_value).unwrap());
    }
    let body_sort = egraph.get_sort(body_value).unwrap().clone();
    if body_sort.is_container_sort() {
        panic!("Container support not implemented")
    }
    // If the body is not an eq sort, we don't need to recurse, just return
    if !body_sort.is_eq_sort() {
        return None;
    }

    // If the body is an eq sort, we need to recurse

    let body_id = Id::from(body_value.bits as usize);
    let canonical_body_id = egraph.unionfind.find(body_id);

    let new_body_id = egraph.unionfind.make_set();
    let new_body_value = Value::from_id(body_sort.name(), new_body_id);
    subtituted.insert(*body_value, new_body_value);

    let mut made_changes = false;

    // Then, we want to iterate through all functions whose return sort is the body sort
    for name in egraph.functions.keys().cloned().collect::<Vec<_>>() {
        let function = egraph.functions.get(&name).unwrap().clone();
        if function.schema.output.name() != body_sort.name() {
            continue;
        }
        // For each function, we want to iterate through all of its e-nodes
        for (input, output) in function.nodes.iter() {
            // If the canonical ID of the output is not the same as the canonical ID of the body, skip it
            if egraph.unionfind.find(Id::from(output.value.bits as usize)) != canonical_body_id {
                continue;
            }
            // Now build up new inputs based on substituting the old inputs
            let mut any_new_inputs = false;
            let mut new_input = vec![];
            for i in input {
                let new_input_value = substitute(egraph, i, var_value, input_value, subtituted);
                new_input.push(match new_input_value {
                    Some(_) => {
                        any_new_inputs = true;
                        new_input_value.unwrap()
                    }
                    None => *i,
                })
            }
            // if !any_new_inputs {
            //     continue;
            // }
            made_changes = true;
            let res = egraph.functions.get_mut(&name).unwrap().nodes.insert(
                &new_input[..],
                new_body_value,
                egraph.timestamp,
            );
            match res {
                Some(new_value) => {
                    egraph.union(body_id, Id::from(new_value.bits as usize), body_sort.name());
                }
                None => {}
            }
            // if res.is_some() && res != Some(new_body_value) {
            //     panic!("Don't support when inserting returns different node currently for lambda subst {:?}", res)
            // }
        }
    }
    // if !made_changes {
    //     return None;
    // }
    Some(new_body_value)
}
