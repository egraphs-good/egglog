//! Sort to represent functions as values.
//!
//! To declare the sort, you must specify the exact number of arguments and the sort of each, followed by the output sort:
//! `(sort IntToString (UnstableFn (i64) String))`
//!
//! To create a function value, use the `(unstable-fn "name" [<partial args>])` primitive and to apply it use the `(unstable-app function arg1 arg2 ...)` primitive.
//! The number of args must match the number of arguments in the function sort.
//!
//!
//! The value is stored similar to the `vec` sort, as an index into a set, where each item in
//! the set is a `(Symbol, Vec<Value>)` pairs. The Symbol is the function name, and the `Vec<Value>` is
//! the list of partially applied arguments.
use std::sync::Mutex;

use crate::ast::Literal;

use super::*;

/// A function value is a name of a function, a list of partially applied arguments (values and sort)
/// Note that we must store the actual arcsorts so we can return them when returning inner values
/// and when canonicalizing
#[derive(Debug, Clone)]

struct ValueFunction(Symbol, Vec<(ArcSort, Value)>);

impl ValueFunction {
    /// Remove the arcsorts to make this hashable
    /// The arg values contain the sort name anyways
    fn hashable(&self) -> (Symbol, Vec<&Value>) {
        (self.0, self.1.iter().map(|(_, v)| v).collect())
    }
}

impl Hash for ValueFunction {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.hashable().hash(state);
    }
}

impl PartialEq for ValueFunction {
    fn eq(&self, other: &Self) -> bool {
        self.hashable() == other.hashable()
    }
}

impl Eq for ValueFunction {}

#[derive(Debug)]
pub struct FunctionSort {
    name: Symbol,
    inputs: Vec<ArcSort>,
    output: ArcSort,
    functions: Mutex<IndexSet<ValueFunction>>,
}

impl FunctionSort {
    pub fn presort_names() -> Vec<Symbol> {
        vec!["unstable-fn".into(), "unstable-app".into()]
    }
    pub fn make_sort(
        typeinfo: &mut TypeInfo,
        name: Symbol,
        args: &[Expr],
    ) -> Result<ArcSort, TypeError> {
        if let [inputs, Expr::Var(span, output)] = args {
            let output_sort = typeinfo
                .sorts
                .get(output)
                .ok_or(TypeError::UndefinedSort(*output, span.clone()))?;

            let input_sorts = match inputs {
                Expr::Call(_, first, rest_args) => {
                    let all_args = once(first).chain(rest_args.iter().map(|arg| {
                        if let Expr::Var(_, arg) = arg {
                            arg
                        } else {
                            panic!("function sort must be called with list of input sorts");
                        }
                    }));
                    all_args
                        .map(|arg| {
                            typeinfo
                                .sorts
                                .get(arg)
                                .ok_or(TypeError::UndefinedSort(*arg, span.clone()))
                                .map(|s| s.clone())
                        })
                        .collect::<Result<Vec<_>, _>>()?
                }
                // an empty list of inputs args is parsed as a unit literal
                Expr::Lit(_, Literal::Unit) => vec![],
                _ => panic!("function sort must be called with list of input sorts"),
            };

            Ok(Arc::new(Self {
                name,
                inputs: input_sorts,
                output: output_sort.clone(),
                functions: Default::default(),
            }))
        } else {
            panic!("function sort must be called with list of input args and output sort");
        }
    }

    fn get_value(&self, value: &Value) -> ValueFunction {
        let functions = self.functions.lock().unwrap();
        functions.get_index(value.bits as usize).unwrap().clone()
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

    fn serialized_name(&self, value: &Value) -> Symbol {
        self.get_value(value).0
    }

    fn inner_values(&self, value: &Value) -> Vec<(ArcSort, Value)> {
        let functions = self.functions.lock().unwrap();
        let input_values = functions.get_index(value.bits as usize).unwrap();
        input_values.1.clone()
    }

    fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
        let ValueFunction(name, inputs) = self.get_value(value);
        let mut changed = false;
        let mut new_outputs = vec![];
        for (s, mut v) in inputs.into_iter() {
            changed |= s.canonicalize(&mut v, unionfind);
            new_outputs.push((s, v));
        }
        *value = ValueFunction(name, new_outputs).store(self).unwrap();
        changed
    }

    fn register_primitives(self: Arc<Self>, typeinfo: &mut TypeInfo) {
        typeinfo.add_primitive(Ctor {
            name: "unstable-fn".into(),
            function: self.clone(),
            string: typeinfo.get_sort_nofail(),
        });
        typeinfo.add_primitive(Apply {
            name: "unstable-app".into(),
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
        let ValueFunction(name, inputs) = ValueFunction::load(self, &value);
        let (cost, args) = inputs.into_iter().try_fold(
            (
                1usize,
                vec![GenericExpr::Lit(DUMMY_SPAN.clone(), Literal::String(name))],
            ),
            |(cost, mut args), (sort, value)| {
                let (new_cost, term) = extractor.find_best(value, termdag, &sort)?;
                args.push(termdag.term_to_expr(&term));
                Some((cost.saturating_add(new_cost), args))
            },
        )?;

        Some((cost, Expr::call_no_span("unstable-fn", args)))
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

/// Takes a string and any number of partially applied args of any sort and returns a function
struct FunctionCTorTypeConstraint {
    name: Symbol,
    function: Arc<FunctionSort>,
    string: Arc<StringSort>,
    span: Span,
}

impl TypeConstraint for FunctionCTorTypeConstraint {
    fn get(
        &self,
        arguments: &[AtomTerm],
        typeinfo: &TypeInfo,
    ) -> Vec<Constraint<AtomTerm, ArcSort>> {
        // Must have at least one arg (plus the return value)
        if arguments.len() < 2 {
            return vec![Constraint::Impossible(
                constraint::ImpossibleConstraint::ArityMismatch {
                    atom: core::Atom {
                        span: self.span.clone(),
                        head: self.name,
                        args: arguments.to_vec(),
                    },
                    expected: 1,
                    actual: 0,
                },
            )];
        }
        let output_sort_constraint: constraint::Constraint<_, ArcSort> = Constraint::Assign(
            arguments[arguments.len() - 1].clone(),
            self.function.clone(),
        );
        // If first arg is a literal string and we know the name of the function and can use that to know what
        // types to expect
        if let AtomTerm::Literal(_, Literal::String(ref name)) = arguments[0] {
            if let Some(func_type) = typeinfo.func_types.get(name) {
                // The arguments contains the return sort as well as the function name
                let n_partial_args = arguments.len() - 2;
                // the number of partial args must match the number of inputs from the func type minus the number from
                // this function sort
                if self.function.inputs.len() + n_partial_args != func_type.input.len() {
                    return vec![Constraint::Impossible(
                        constraint::ImpossibleConstraint::ArityMismatch {
                            atom: core::Atom {
                                span: self.span.clone(),
                                head: self.name,
                                args: arguments.to_vec(),
                            },
                            expected: self.function.inputs.len() + func_type.input.len() + 1,
                            actual: arguments.len() - 1,
                        },
                    )];
                }
                // the output type and input types (starting after the partial args) must match between these functions
                let expected_output = self.function.output.clone();
                let expected_input = self.function.inputs.clone();
                let actual_output = func_type.output.clone();
                let actual_input: Vec<ArcSort> = func_type
                    .input
                    .iter()
                    .skip(n_partial_args)
                    .cloned()
                    .collect();
                if expected_output.name() != actual_output.name()
                    || expected_input
                        .iter()
                        .map(|s| s.name())
                        .ne(actual_input.iter().map(|s| s.name()))
                {
                    return vec![Constraint::Impossible(
                        constraint::ImpossibleConstraint::FunctionMismatch {
                            expected_output,
                            expected_input,
                            actual_output,
                            actual_input,
                        },
                    )];
                }
                // if they match, then just make sure the partial args match as well
                return func_type
                    .input
                    .iter()
                    .take(n_partial_args)
                    .zip(arguments.iter().skip(1))
                    .map(|(expected_sort, actual_term)| {
                        Constraint::Assign(actual_term.clone(), expected_sort.clone())
                    })
                    .chain(once(output_sort_constraint))
                    .collect();
            }
        }

        // Otherwise we just try assuming it's this function, we don't know if it is or not
        vec![
            Constraint::Assign(arguments[0].clone(), self.string.clone()),
            output_sort_constraint,
        ]
    }
}

// (unstable-fn "name" [<arg1>, <arg2>, ...])
struct Ctor {
    name: Symbol,
    function: Arc<FunctionSort>,
    string: Arc<StringSort>,
}

impl PrimitiveLike for Ctor {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        Box::new(FunctionCTorTypeConstraint {
            name: self.name,
            function: self.function.clone(),
            string: self.string.clone(),
            span: span.clone(),
        })
    }

    fn apply(&self, values: &[Value], egraph: Option<&mut EGraph>) -> Option<Value> {
        let egraph = egraph.expect("`unstable-fn` is not supported yet in facts.");
        let name = Symbol::load(&self.string, &values[0]);
        // self.function
        //     .sorts
        //     .insert(name.clone(), self.function.clone());
        let args = values[1..]
            .iter()
            .map(|arg| (egraph.get_sort_from_value(arg).unwrap().clone(), *arg))
            .collect();
        ValueFunction(name, args).store(&self.function)
    }
}

// (unstable-app <function> [<arg1>, <arg2>, ...])
struct Apply {
    name: Symbol,
    function: Arc<FunctionSort>,
}

impl PrimitiveLike for Apply {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        let mut sorts: Vec<ArcSort> = vec![self.function.clone()];
        sorts.extend(self.function.inputs.clone());
        sorts.push(self.function.output.clone());
        SimpleTypeConstraint::new(self.name(), sorts, span.clone()).into_box()
    }

    fn apply(&self, values: &[Value], egraph: Option<&mut EGraph>) -> Option<Value> {
        let egraph = egraph.expect("`unstable-app` is not supported yet in facts.");
        let ValueFunction(name, args) = ValueFunction::load(&self.function, &values[0]);
        let types: Vec<_> = args
            .iter()
            // get the sorts of partially applied args
            .map(|(sort, _)| sort.clone())
            // combine with the args for the function call and then the output
            .chain(self.function.inputs.clone())
            .chain(once(self.function.output.clone()))
            .collect();
        let values = args
            .iter()
            .map(|(_, v)| *v)
            .chain(values[1..].iter().copied())
            .collect();
        Some(call_fn(egraph, &name, types, values))
    }
}

/// Call function (either primitive or eqsort) <name> with value args <args> and return the value.
///
/// Does this in a similar way to how merge functions are resolved, using the stack and actions,
/// so that we can re-use the logic for primitive and regular functions.
fn call_fn(egraph: &mut EGraph, name: &Symbol, types: Vec<ArcSort>, args: Vec<Value>) -> Value {
    // Make a call with temp vars as each of the args
    let resolved_call = ResolvedCall::from_resolution(name, types.as_slice(), egraph.type_info());
    let arg_vars: Vec<_> = types
        .into_iter()
        // Skip last sort which is the output sort
        .take(args.len())
        .enumerate()
        .map(|(i, sort)| ResolvedVar {
            name: format!("__arg_{}", i).into(),
            sort,
            is_global_ref: false,
        })
        .collect();
    let binding = IndexSet::from_iter(arg_vars.clone());
    let resolved_args = arg_vars
        .into_iter()
        .map(|v| GenericExpr::Var(DUMMY_SPAN.clone(), v))
        .collect();
    let expr = GenericExpr::Call(DUMMY_SPAN.clone(), resolved_call, resolved_args);
    // Similar to how the merge function is created in `Function::new`
    let (actions, mapped_expr) = expr
        .to_core_actions(
            egraph.type_info(),
            &mut binding.clone(),
            &mut ResolvedGen::new("$".to_string()),
        )
        .unwrap();
    let target = mapped_expr.get_corresponding_var_or_lit(egraph.type_info());
    let program = egraph.compile_expr(&binding, &actions, &target).unwrap();
    // Similar to how the `MergeFn::Expr` case is handled in `Egraph::perform_set`
    // egraph.rebuild().unwrap();
    let mut stack = vec![];
    egraph
        .run_actions(&mut stack, &args, &program, true)
        .unwrap();
    stack.pop().unwrap()
}
