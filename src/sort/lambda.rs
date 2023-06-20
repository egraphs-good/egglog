use std::{collections::BTreeMap, sync::Mutex};

use super::*;

#[derive(Debug, Hash, Eq, PartialEq, Copy, Clone)]
struct ValueLambda {
    var_id: Id,
    body: Value,
}

#[derive(Debug)]
pub struct LambdaSort {
    name: Symbol,
    input: ArcSort,
    output: ArcSort,
    lambdas: Mutex<IndexSet<ValueLambda>>,
    // Map from vars to their ID
    symbol_to_id: Mutex<BTreeMap<Symbol, Id>>,
    // Inverse map
    id_to_symbol: Mutex<BTreeMap<Id, Symbol>>,
}

impl LambdaSort {
    pub fn make_sort(
        typeinfo: &mut TypeInfo,
        name: Symbol,
        args: &[Expr],
    ) -> Result<ArcSort, TypeError> {
        if let [Expr::Var(input), Expr::Var(output)] = args {
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
                    "Lambdasreturning other EqSort containers are not allowed".into(),
                ));
            }

            Ok(Arc::new(Self {
                name,
                input: input.clone(),
                output: output.clone(),
                lambdas: Default::default(),
                symbol_to_id: Default::default(),
                id_to_symbol: Default::default(),
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

    fn is_container_sort(&self) -> bool {
        true
    }

    fn is_eq_container_sort(&self) -> bool {
        self.output.is_eq_sort()
    }

    fn foreach_tracked_values<'a>(&'a self, value: &'a Value, mut f: Box<dyn FnMut(Value) + 'a>) {
        // TODO: Potential duplication of code
        let lambdas = self.lambdas.lock().unwrap();
        let lambda = lambdas.get_index(value.bits as usize).unwrap();
        f(lambda.body)
    }

    fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
        let lambdas = self.lambdas.lock().unwrap();
        let lambda = lambdas.get_index(value.bits as usize).unwrap();
        let mut body = lambda.body;
        let changed = self.output.canonicalize(&mut body, unionfind);
        let new_lambda = ValueLambda {
            var_id: lambda.var_id,
            body,
        };
        // drop(lambda);
        *value = new_lambda.store(self).unwrap();
        changed
    }

    fn register_primitives(self: Arc<Self>, typeinfo: &mut TypeInfo) {
        typeinfo.add_primitive(Lambda {
            name: "lambda".into(),
            lambda: self.clone(),
        });
        typeinfo.add_primitive(Var {
            name: "var".into(),
            lambda: self.clone(),
            string: typeinfo.get_sort(),
        });
        typeinfo.add_primitive(Apply {
            name: "apply".into(),
            lambda: self,
        });
    }

    fn make_expr(&self, egraph: &EGraph, value: Value) -> Expr {
        let lambda = ValueLambda::load(self, &value);
        let var_string = self.id_to_symbol.lock().unwrap()[&lambda.var_id];
        // Generate (lambda (var vv) body)
        Expr::call(
            "lambda",
            [
                Expr::call("var", [Expr::Lit(Literal::String(var_string))]),
                egraph.extract(lambda.body, &self.output).1,
            ],
        )
    }
}

impl IntoSort for ValueLambda {
    type Sort = LambdaSort;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        let mut lambdas = sort.lambdas.lock().unwrap();
        let (i, _) = lambdas.insert_full(self);
        Some(Value {
            tag: sort.name,
            bits: i as u64,
        })
    }
}

impl FromSort for ValueLambda {
    type Sort = LambdaSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let lambdas = sort.lambdas.lock().unwrap();
        *lambdas.get_index(value.bits as usize).unwrap()
    }
}

struct Var {
    name: Symbol,
    lambda: Arc<LambdaSort>,
    string: Arc<StringSort>,
}

impl PrimitiveLike for Var {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        if let [sort] = types {
            if sort.name() == self.string.name() {
                return Some(self.lambda.input.clone());
            }
        }
        None
    }

    fn apply(&self, values: &[Value], egraph: Option<&mut EGraph>) -> Option<Value> {
        let var = Symbol::load(&self.string, &values[0]);
        let mut var_to_id = self.lambda.symbol_to_id.lock().unwrap();
        let id = match var_to_id.entry(var) {
            // If we have already saved a var for this ID used it
            std::collections::btree_map::Entry::Occupied(o) => *o.get(),
            // Otherwise, if we have the unionfind, make an ID and return it
            // If we don't the unionfind, we are in type checking and return a dummy ID
            std::collections::btree_map::Entry::Vacant(v) => egraph.map_or(Id::from(0), |e| {
                let id = e.unionfind.make_set();
                v.insert(id);
                self.lambda.id_to_symbol.lock().unwrap().insert(id, var);
                id
            }),
        };
        Some(Value::from_id(self.lambda.input.name(), id))
    }
}

struct Lambda {
    name: Symbol,
    lambda: Arc<LambdaSort>,
}

impl PrimitiveLike for Lambda {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        if let [var_tp, body_tp] = types {
            if var_tp.name() == self.lambda.input.name()
                && body_tp.name() == self.lambda.output.name()
            {
                return Some(self.lambda.clone());
            }
        }
        None
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
        ValueLambda {
            var_id: Id::from(values[0].bits as usize),
            body: values[1],
        }
        .store(&self.lambda)
    }
}

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

    fn apply(&self, values: &[Value], egraph: Option<&mut EGraph>) -> Option<Value> {
        let lambda = ValueLambda::load(&self.lambda, &values[0]);
        let var_value = Value::from_id(self.lambda.input.name(), lambda.var_id);
        let body = lambda.body;
        let input_value = values[1];
        // If we don't have an e-graph, just return the body, we are just type checking
        match egraph {
            None => Some(body),
            Some(e) => {
                // If we do have an e-graph, we need to substitute the var with the input
                // In body replace all instances of var_value with input_value

                Some(
                    substitute(e, &body, &var_value, &input_value, &mut HashMap::default())
                        .unwrap_or(body),
                )
            }
        }
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
            if !any_new_inputs {
                continue;
            }
            made_changes = true;
            let res = egraph.functions.get_mut(&name).unwrap().nodes.insert(
                &new_input[..],
                new_body_value,
                egraph.timestamp,
            );
            if res.is_some() && res != Some(new_body_value) {
                panic!("Don't support when inserting returns different node currently for lambda subst {:?}", res)
            }
        }
    }
    if !made_changes {
        return None;
    }
    Some(new_body_value)
}
