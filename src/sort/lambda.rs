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

        if self.output.is_eq_sort() {
            f(lambda.body)
        }
    }

    fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
        println!("Canonicalizing lambda");
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

    fn apply(&self, values: &[Value], unionfind: Option<&mut UnionFind>) -> Option<Value> {
        let var = Symbol::load(&self.string, &values[0]);
        let mut var_to_id = self.lambda.symbol_to_id.lock().unwrap();
        let id = match var_to_id.entry(var) {
            // If we have already saved a var for this ID used it
            std::collections::btree_map::Entry::Occupied(o) => *o.get(),
            // Otherwise, if we have the unionfind, make an ID and return it
            // If we don't the unionfind, we are in type checking and return a dummy ID
            std::collections::btree_map::Entry::Vacant(v) => unionfind.map_or(Id::from(0), |u| {
                let id = u.make_set();
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

    fn apply(&self, values: &[Value], _unionfind: Option<&mut UnionFind>) -> Option<Value> {
        ValueLambda {
            var_id: Id::from(values[0].bits as usize),
            body: values[1],
        }
        .store(&self.lambda)
    }
}

struct Apply {
    name: Symbol,
    lambda: Arc<LambdaSort>,
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

    fn apply(&self, values: &[Value], _unionfind: Option<&mut UnionFind>) -> Option<Value> {
        let lambda = ValueLambda::load(&self.lambda, &values[0]);
        let var_value = Value::from_id(self.lambda.input.name(), lambda.var_id);
        let body = lambda.body;
        let input_value = values[1];
        // In body replace all instances of var_value with input_value
        Some(body)


    }
}
