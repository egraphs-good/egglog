//! Sort to represent functions as values.
//!
//! To declare the sort, you must specify the exact number of arguments and the sort of each, followed by the output sort:
//! `(sort IntToString (Function (i64) String))`
//!
//! To create a function value, use the `(function "name")` primitive and to apply it use the `(call function arg1 arg2 ...)` primitive.
//! The number of args must match the number of arguments in the function sort.
//!
//! To partially apply a function, use the `(partial function arg1)` primitive. This will return a new function value with the first argument applied.
//!
//! The value is stored similar to the `vec` sort, as an index into a set, where each item in
//! the set is a (Symbol, Vec<Value>) pair. The Symbol is the function name, and the Vec<Value> is
//! the list of partially applied arguments.
use std::sync::Mutex;

use crate::{ast::Literal, constraint::AllEqualTypeConstraint};

use super::*;

type ValueFunction = (Symbol, Vec<Value>);

#[derive(Debug)]
pub struct FunctionSort {
    name: Symbol,
    inputs: Vec<ArcSort>,
    output: ArcSort,
    functions: Mutex<IndexSet<ValueFunction>>,
}

impl FunctionSort {
    pub fn presort_names() -> Vec<Symbol> {
        vec!["function".into(), "apply".into(), "partial".into()]
    }
    pub fn make_sort(
        typeinfo: &mut TypeInfo,
        name: Symbol,
        args: &[Expr],
    ) -> Result<ArcSort, TypeError> {
        if let [Expr::Call((), first, rest_args), Expr::Var((), output)] = args {
            let output_sort = typeinfo
                .sorts
                .get(output)
                .ok_or(TypeError::UndefinedSort(*output))?;
            let input_sorts = rest_args
                .iter()
                .map(|arg| {
                    if let Expr::Var((), arg) = arg {
                        typeinfo
                            .sorts
                            .get(arg)
                            .ok_or(TypeError::UndefinedSort(*arg))
                            .map(|s| s.clone())
                    } else {
                        panic!("function sort must be called with list of input sorts");
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(Arc::new(Self {
                name,
                inputs: input_sorts,
                output: output_sort.clone(),
                functions: Default::default(),
            }));
        } else {
            panic!("function sort must be called with list of input args and output sort");
        }
    }

    fn get_value(&self, value: &Value) -> ValueFunction {
        let functions = self.functions.lock().unwrap();
        let (name, args) = functions.get_index(value.bits as usize).unwrap();
        (name.clone(), args.clone())
    }
}

impl Sort for FunctionSort {
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
        self.inputs.iter().any(|s| s.is_eq_sort())
    }

    fn inner_values(&self, value: &Value) -> Vec<(&ArcSort, Value)> {
        let inputs = self.get_value(value).1;
        inputs
            .into_iter()
            .zip(self.inputs)
            .map(|(v, s)| (&s, v))
            .collect()
    }

    fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
        let (name, inputs) = self.get_value(value);
        let mut changed = false;
        let new_outputs: Vec<Value> = inputs
            .into_iter()
            .zip(self.inputs)
            .map(|(mut v, s)| {
                changed |= s.canonicalize(&mut v, unionfind);
                v
            })
            .collect();
        *value = (name, new_outputs).store(self).unwrap();
        changed
    }

    fn register_primitives(self: Arc<Self>, typeinfo: &mut TypeInfo) {
        typeinfo.add_primitive(Ctor {
            name: "function".into(),
            function: self.clone(),
            string: typeinfo.get_sort_nofail(),
        });
        typeinfo.add_primitive(Partial {
            name: "partial".into(),
            function: self,
        });
        typeinfo.add_primitive(FunctionCall {
            name: "call".into(),
            function: self.clone(),
        });
    }

    fn make_expr(&self, egraph: &EGraph, value: Value) -> (Cost, Expr) {
        let mut termdag = TermDag::default();
        let extractor = Extractor::new(egraph, &mut termdag);
        self.extract_expr(egraph, value, &extractor, &mut termdag)
            .expect("Extraction should be successful since extractor has been fully initialized")
    }

    fn extract_expr(
        &self,
        _egraph: &EGraph,
        value: Value,
        extractor: &Extractor,
        termdag: &mut TermDag,
    ) -> Option<(Cost, Expr)> {
        let (name, inputs) = ValueFunction::load(self, &value);
        inputs.into_iter().zip(self.inputs).fold(
            Some((
                0,
                Expr::call("function", [Expr::Lit((), Literal::String(name))]),
            )),
            |acc, (value, sort)| {
                let (cost, expr) = acc?;
                let e = extractor.find_best(value, termdag, &sort)?;
                Some((cost.saturating_add(e.0), termdag.term_to_expr(&e.1)))
            },
        )
    }
}

impl IntoSort for ValueFunction {
    type Sort = FunctionSort;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        let mut functions = sort.functions.lock().unwrap();
        let (i, _) = functions.insert_full(self);
        Some(Value {
            tag: sort.name,
            bits: i as u64,
        })
    }
}

impl FromSort for ValueFunction {
    type Sort = FunctionSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        sort.get_value(value)
    }
}

struct Ctor {
    name: Symbol,
    function: Arc<FunctionSort>,
    string: Arc<StringSort>,
}

impl PrimitiveLike for Ctor {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.string.clone(), self.function.clone()],
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], egraph: &EGraph) -> Option<Value> {
        let name = Symbol::load(&self.string, &values[0]);
        (name, vec![]).store(&self.function)
    }
}

struct Partial {
    name: Symbol,
    function: Arc<FunctionSort>,
}

impl PrimitiveLike for Partial {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(self.name(), vec![self.function.clone()]).into_box()
    }

    fn apply(&self, values: &[Value], egraph: &EGraph) -> Option<Value> {
        let (name, mut args) = ValueFunction::load(&self.function, &values[0]);
        args.push(values[1]);
        (name, args).store(&self.function)
    }
}
