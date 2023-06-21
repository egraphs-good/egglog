/// A sort for lambdas
///
/// They pretend to be an e-sort, but just with more type info and registration functions provided.
use crate::typechecking::FuncType;

use super::*;

#[derive(Debug)]
pub struct LambdaSort {
    name: Symbol,
    input: ArcSort,
    output: ArcSort,
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
        typeinfo.add_primitive(Apply {
            name: "apply".into(),
            lambda: self,
        });
    }

    fn make_expr(&self, _egraph: &EGraph, _value: Value) -> Expr {
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
            .find(|(_, output)| output.value == lambda_value)
            .expect("finding lambda fn call");
        let var_value = lambda_input[0];
        let body_value = lambda_input[1];
        Some(substitute(
            egraph,
            &body_value,
            &var_value,
            &input_value,
            &mut HashMap::default(),
            // "",
        ))
    }
}

/// Substitutues instance of var_value in body with input_value
/// Returns a new value if any substitutions were made
fn substitute(
    egraph: &mut EGraph,
    body_value: &Value,
    var_value: &Value,
    input_value: &Value,
    // Mapping of e-class IDs to their substituted values, so we don't end up in loops for cyclic graphs
    substituted: &mut HashMap<Id, Id>,
) -> Value {
    // If the body is not an eq sort, we don't need to recurse, just return the primitive
    let body_sort = egraph.get_sort(body_value).unwrap().clone();
    if !body_sort.is_eq_sort() {
        return *body_value;
    }
    // If the body is equal to the var, then we can just return the input
    if body_value == var_value {
        return *input_value;
    }
    // If the body is equal to the input, we can also just return the input
    if body_value == input_value {
        return *input_value;
    }

    if body_sort.is_container_sort() {
        panic!("Container support not implemented")
    }

    let body_id = Id::from(body_value.bits as usize);
    if let Some(replaced_id) = substituted.get(&body_id) {
        return Value::from_id(body_sort.name(), *replaced_id);
    }

    // Create a new e-class to store all the new bodies
    let mut new_body_id = egraph.unionfind.make_set();

    // Store that we are replacing the old body with the new body
    substituted.insert(body_id, new_body_id);
    // Build up a list of all the functions and their inputs which return this body
    // All of this will need to have their args substituted and unioned to make a new body
    let canonical_body_id = egraph.unionfind.find(body_id);
    let fn_calls = find_all_call(egraph, body_sort.name(), canonical_body_id);
    // For every function calls, create a new function call with the substituted args
    for (f_name, inputs) in fn_calls {
        // println!("{} {} {:?}", prefix, f_name, inputs);
        let new_inputs = inputs
            .iter()
            .map(|input| {
                substitute(
                    egraph,
                    input,
                    var_value,
                    input_value,
                    substituted,
                    // format!("{}  ", prefix).as_str(),
                )
            })
            .collect::<Vec<_>>();
        let new_res = egraph.functions.get_mut(&f_name).unwrap().insert(
            &new_inputs,
            Value::from_id(body_sort.name(), new_body_id),
            egraph.timestamp,
        );
        if let Some(new_res) = new_res {
            let res_id = new_res.bits as usize;
            new_body_id = egraph.union(new_body_id, Id::from(res_id), body_sort.name());
        }
    }
    Value::from_id(body_sort.name(), new_body_id)
}

/// Returns all the function calls in the egraph which return the e-class of the given sort
fn find_all_call(egraph: &EGraph, sort_name: Symbol, id: Id) -> Vec<(Symbol, Vec<Value>)> {
    egraph
        .functions
        .iter()
        // Filter for functions which return this sort and are not variables
        .filter(|(_, f)| f.schema.output.name() == sort_name && !f.is_variable)
        .flat_map(|(f_name, f)| {
            f.nodes
                .iter()
                // Filter for nodes where the canonical ID of the output is the same as the canonical ID of the body
                .filter(|(_, output)| {
                    egraph.unionfind.find(Id::from(output.value.bits as usize)) == id
                })
                .map(|(inputs, _)| (*f_name, inputs.to_vec()))
        })
        .collect::<Vec<_>>()
}
