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
use crate::core::Atom;
use std::sync::Mutex;

use constraint::{get_atom_application_constraints, ImpossibleConstraint};

use crate::ast::Literal;

use super::*;

#[derive(Debug)]
pub struct ConstSort {
    literal: Literal,
}

impl Sort for ConstSort {
    fn name(&self) -> Symbol {
        format!("const-{}", self.literal).into()
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn make_expr(&self, _egraph: &EGraph, _value: Value) -> (Cost, Expr) {
        (
            0,
            Expr::call_no_span("#", vec![Expr::lit_no_span(self.literal.clone())]),
        )
    }
}

pub struct HashConst;
impl PrimitiveLike for HashConst {
    fn name(&self) -> Symbol {
        "#".into()
    }

    fn get_type_constraints(&self, _span: &Span) -> Box<dyn TypeConstraint> {
        Box::new(HashConstTypeConstraint)
    }

    fn apply(
        &self,
        _values: &[Value],
        _sorts: (&[ArcSort], &ArcSort),
        _egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
        Some(Value {
            tag: self.name(),
            bits: 0,
        })
    }
}

struct HashConstTypeConstraint;
impl TypeConstraint for HashConstTypeConstraint {
    fn get<'a>(
        &self,
        arguments: &[AtomTerm],
        _typeinfo: &'a TypeInfo,
    ) -> Vec<Constraint<'a, AtomTerm, ArcSort>> {
        if arguments.len() != 2 {
            return vec![Constraint::Impossible(
                ImpossibleConstraint::ArityMismatch {
                    atom: Atom {
                        span: DUMMY_SPAN.clone(),
                        head: "#".into(),
                        args: arguments.to_vec(),
                    },
                    expected: 1,
                    actual: arguments.len() - 1,
                },
            )];
        }

        if let AtomTerm::Literal(_span, literal) = &arguments[0] {
            let literal = literal.clone();
            vec![
                Constraint::Assign(arguments[0].clone(), Arc::new(StringSort)),
                Constraint::Assign(arguments[1].clone(), Arc::new(ConstSort { literal })),
            ]
        } else {
            vec![Constraint::Impossible(
                ImpossibleConstraint::CompileTimeConstantExpected {
                    span: arguments[0].span().clone(),
                    sort: Arc::new(StringSort),
                },
            )]
        }
    }
}

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
    fn get_value(&self, value: &Value) -> ValueFunction {
        let functions = self.functions.lock().unwrap();
        functions.get_index(value.bits as usize).unwrap().clone()
    }
}

impl Presort for FunctionSort {
    fn presort_name() -> Symbol {
        "UnstableFn".into()
    }

    fn reserved_primitives() -> Vec<Symbol> {
        vec!["unstable-fn".into(), "unstable-app".into()]
    }

    fn make_sort(
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
                                .cloned()
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
#[derive(Clone)]
struct FunctionCTorTypeConstraint {
    name: Symbol,
    function: Arc<FunctionSort>,
    span: Span,
}

impl TypeConstraint for FunctionCTorTypeConstraint {
    fn get<'a>(
        &self,
        arguments: &[AtomTerm],
        typeinfo: &'a TypeInfo,
    ) -> Vec<Constraint<'a, AtomTerm, ArcSort>> {
        // Must have at least one arg (plus the return value)
        if arguments.len() < 2 {
            return vec![Constraint::Impossible(
                ImpossibleConstraint::ArityMismatch {
                    atom: Atom {
                        span: self.span.clone(),
                        head: self.name,
                        args: arguments.to_vec(),
                    },
                    expected: 1,
                    actual: arguments.len() - 1,
                },
            )];
        }

        let this = self.clone();
        let arguments = arguments.to_vec();
        let argument = arguments[0].clone();
        vec![Constraint::LazyConstraint(
            arguments[0].clone(),
            Box::new(move |sort| {
                let sort = sort.clone().as_arc_any();
                let Ok(sort) = Arc::downcast::<ConstSort>(sort) else {
                    return Constraint::Impossible(
                        ImpossibleConstraint::CompileTimeConstantExpected {
                            span: argument.span().clone(),
                            sort: Arc::new(StringSort),
                        },
                    );
                };

                let Literal::String(head) = sort.literal else {
                    return Constraint::Impossible(
                        ImpossibleConstraint::CompileTimeConstantExpected {
                            span: argument.span().clone(),
                            sort: Arc::new(StringSort),
                        },
                    );
                };

                let mut all_constraints = vec![];
                let mut arguments = arguments[1..].to_vec();
                let output_sort = arguments.pop().unwrap();
                let output_sort_constraint =
                    Constraint::Assign(output_sort, this.function.clone() as ArcSort);
                all_constraints.push(output_sort_constraint);

                let mut dummy_args_constraint = vec![];
                for s in this
                    .function
                    .inputs
                    .iter()
                    .chain(once(&this.function.output))
                {
                    // This won't create ambiguity/conflicts since sort names are supposed to be unique
                    let dummy_atom_term =
                        AtomTerm::Var(DUMMY_SPAN.clone(), format!("$dummy_{}", s.name()).into());
                    dummy_args_constraint
                        .push(Constraint::Assign(dummy_atom_term.clone(), s.clone()));
                    arguments.push(dummy_atom_term);
                }
                all_constraints.extend(dummy_args_constraint);

                // Reuse the constraint generation for normal type checking
                let atom_constraints =
                    get_atom_application_constraints(&head, &arguments, &this.span, typeinfo);
                all_constraints.extend(atom_constraints);

                Constraint::And(all_constraints)
            }),
        )]
    }
}

// (unstable-fn "name" [<arg1>, <arg2>, ...])
struct Ctor {
    name: Symbol,
    function: Arc<FunctionSort>,
}

impl PrimitiveLike for Ctor {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        Box::new(FunctionCTorTypeConstraint {
            name: self.name,
            function: self.function.clone(),
            span: span.clone(),
        })
    }

    fn apply(
        &self,
        values: &[Value],
        sorts: (&[ArcSort], &ArcSort),
        egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
        egraph.expect("`unstable-fn` is not supported yet in facts.");
        let const_sort: Arc<ConstSort> = Arc::downcast(sorts.0[0].clone().as_arc_any()).unwrap();
        let Literal::String(name) = const_sort.literal else {
            panic!("`unstable-fn` must be called with a string literal as the first argument");
        };
        let args = sorts.0[1..]
            .iter()
            .cloned()
            .zip(values[1..].iter().cloned())
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

    fn apply(
        &self,
        values: &[Value],
        _sorts: (&[ArcSort], &ArcSort),
        egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
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
    let resolved_call = ResolvedCall::from_resolution(name, types.as_slice(), &egraph.type_info);
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
            &egraph.type_info,
            &mut binding.clone(),
            &mut egraph.symbol_gen,
        )
        .unwrap();
    let target = mapped_expr.get_corresponding_var_or_lit(&egraph.type_info);
    let program = egraph.compile_expr(&binding, &actions, &target).unwrap();
    // Similar to how the `MergeFn::Expr` case is handled in `Egraph::perform_set`
    // egraph.rebuild().unwrap();
    let mut stack = vec![];
    egraph.run_actions(&mut stack, &args, &program).unwrap();
    stack.pop().unwrap()
}
