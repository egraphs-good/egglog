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
//! the set is a `(Symbol, Vec<(Sort, Value)>)` pairs. The Symbol is the function name, and the `Vec<(Sort, Value)>` is
//! the list of partially applied arguments.
use std::sync::Mutex;

use super::*;

#[derive(Clone, Debug)]
pub struct FunctionContainer(ResolvedFunctionId, pub Vec<(ArcSort, Value)>, pub String);

// implement hash and equality based on values only not arcsorts, since
// arcsorts are not comparable and any two values that are equal must have the same sort

impl PartialEq for FunctionContainer {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
            && self.1.iter().map(|(_, v)| *v).collect::<Vec<_>>()
                == other.1.iter().map(|(_, v)| *v).collect::<Vec<_>>()
            && self.2 == other.2
    }
}

impl Eq for FunctionContainer {}

impl Hash for FunctionContainer {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        for (_, v) in &self.1 {
            v.hash(state);
        }
        self.2.hash(state);
    }
}

impl ContainerValue for FunctionContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn Rebuilder) -> bool {
        let mut changed = false;
        for (s, old) in &mut self.1 {
            if s.is_eq_sort() || s.is_eq_container_sort() {
                let new = rebuilder.rebuild_val(*old);
                changed |= *old != new;
                *old = new;
            }
        }
        changed
    }
    fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        self.1.iter().map(|(_, v)| v).copied()
    }
}

#[derive(Debug)]
pub struct FunctionSort {
    name: String,
    inputs: Vec<ArcSort>,
    output: ArcSort,
    // store all the arcsorts for functions that were added as partial args to this function sort
    // so that we can retrieve them during extraction
    partial_arcsorts: Arc<Mutex<Vec<ArcSort>>>,
}

impl FunctionSort {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn inputs(&self) -> &[ArcSort] {
        &self.inputs
    }

    pub fn output(&self) -> ArcSort {
        self.output.clone()
    }
}

impl Presort for FunctionSort {
    fn presort_name() -> &'static str {
        "UnstableFn"
    }

    fn reserved_primitives() -> Vec<&'static str> {
        vec!["unstable-fn", "unstable-app"]
    }

    fn make_sort(
        typeinfo: &mut TypeInfo,
        name: String,
        args: &[Expr],
    ) -> Result<ArcSort, TypeError> {
        if let [inputs, Expr::Var(span, output)] = args {
            let output_sort = typeinfo
                .get_sort_by_name(output)
                .ok_or(TypeError::UndefinedSort(output.clone(), span.clone()))?;

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
                                .get_sort_by_name(arg)
                                .ok_or(TypeError::UndefinedSort(arg.clone(), span.clone()))
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
                partial_arcsorts: Arc::new(Mutex::new(vec![])),
            }))
        } else {
            panic!("function sort must be called with list of input args and output sort");
        }
    }
}

impl Sort for FunctionSort {
    fn name(&self) -> &str {
        &self.name
    }

    fn column_ty(&self, _backend: &egglog_bridge::EGraph) -> ColumnTy {
        ColumnTy::Id
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend.register_container_ty::<FunctionContainer>();
        backend
            .base_values_mut()
            .register_type::<ResolvedFunction>();
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

    fn serialized_name(&self, container_values: &ContainerValues, value: Value) -> String {
        let val = container_values
            .get_val::<FunctionContainer>(value)
            .unwrap();
        val.2.clone()
    }

    fn inner_sorts(&self) -> Vec<ArcSort> {
        self.partial_arcsorts.lock().unwrap().clone()
    }

    fn inner_values(
        &self,
        container_values: &ContainerValues,
        value: Value,
    ) -> Vec<(ArcSort, Value)> {
        let val = container_values
            .get_val::<FunctionContainer>(value)
            .unwrap();
        val.1.clone()
    }

    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        eg.add_primitive(Ctor {
            name: "unstable-fn".into(),
            function: self.clone(),
        });
        eg.add_primitive(Apply {
            name: "unstable-app".into(),
            function: self.clone(),
        });
    }

    fn value_type(&self) -> Option<TypeId> {
        Some(TypeId::of::<FunctionContainer>())
    }

    fn reconstruct_termdag_container(
        &self,
        container_values: &ContainerValues,
        value: Value,
        termdag: &mut TermDag,
        mut element_terms: Vec<TermId>,
    ) -> TermId {
        let name = &container_values
            .get_val::<FunctionContainer>(value)
            .unwrap()
            .2;
        let head = termdag.lit(Literal::String(name.clone()));
        element_terms.insert(0, head);
        termdag.app("unstable-fn".to_owned(), element_terms)
    }
}

/// Takes a string and any number of partially applied args of any sort and returns a function
struct FunctionCTorTypeConstraint {
    name: String,
    function: Arc<FunctionSort>,
    span: Span,
}

impl TypeConstraint for FunctionCTorTypeConstraint {
    fn get(
        &self,
        arguments: &[AtomTerm],
        typeinfo: &TypeInfo,
    ) -> Vec<Box<dyn Constraint<AtomTerm, ArcSort>>> {
        // Must have at least one arg (plus the return value)
        if arguments.len() < 2 {
            return vec![constraint::impossible(
                constraint::ImpossibleConstraint::ArityMismatch {
                    atom: core::Atom {
                        span: self.span.clone(),
                        head: self.name.clone(),
                        args: arguments.to_vec(),
                    },
                    expected: 2,
                },
            )];
        }
        let output_sort_constraint: Box<dyn Constraint<_, ArcSort>> = constraint::assign(
            arguments[arguments.len() - 1].clone(),
            self.function.clone(),
        );
        // If first arg is a literal string and we know the name of the function and can use that to know what
        // types to expect
        if let AtomTerm::Literal(_, Literal::String(ref name)) = arguments[0] {
            if let Some(func_type) = typeinfo.get_func_type(name) {
                // The arguments contains the return sort as well as the function name
                let n_partial_args = arguments.len() - 2;
                // the number of partial args must match the number of inputs from the func type minus the number from
                // this function sort
                if self.function.inputs.len() + n_partial_args != func_type.input.len() {
                    return vec![constraint::impossible(
                        constraint::ImpossibleConstraint::ArityMismatch {
                            atom: core::Atom {
                                span: self.span.clone(),
                                head: self.name.clone(),
                                args: arguments.to_vec(),
                            },
                            expected: self.function.inputs.len() + func_type.input.len() + 1,
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
                    return vec![constraint::impossible(
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
                        constraint::assign(actual_term.clone(), expected_sort.clone())
                    })
                    .chain(once(output_sort_constraint))
                    .collect();
            }
        }

        // Otherwise we just try assuming it's this function, we don't know if it is or not
        vec![
            constraint::assign(arguments[0].clone(), StringSort.to_arcsort()),
            output_sort_constraint,
        ]
    }
}

// (unstable-fn "name" [<arg1>, <arg2>, ...])
#[derive(Clone)]
struct Ctor {
    name: String,
    function: Arc<FunctionSort>,
}

impl Primitive for Ctor {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        Box::new(FunctionCTorTypeConstraint {
            name: self.name.clone(),
            function: self.function.clone(),
            span: span.clone(),
        })
    }

    fn apply(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
        let (rf, args) = args.split_first().unwrap();
        let ResolvedFunction {
            id,
            partial_arcsorts,
            name,
        } = exec_state.base_values().unwrap(*rf);
        self.function
            .partial_arcsorts
            .lock()
            .unwrap()
            .extend(partial_arcsorts.iter().cloned());
        let args = partial_arcsorts
            .iter()
            .zip(args)
            .map(|(b, x)| (b.clone(), *x))
            .collect();
        let y = FunctionContainer(id, args, name);
        Some(
            exec_state
                .clone()
                .container_values()
                .register_val(y, exec_state),
        )
    }
}

#[derive(Clone, Debug)]
pub struct ResolvedFunction {
    pub id: ResolvedFunctionId,
    pub partial_arcsorts: Vec<ArcSort>,
    pub name: String,
}
// implement equality and hash based on id and  arcsort names, since arcsorts are not comparable

impl PartialEq for ResolvedFunction {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self
                .partial_arcsorts
                .iter()
                .map(|s| s.name())
                .collect::<Vec<_>>()
                == other
                    .partial_arcsorts
                    .iter()
                    .map(|s| s.name())
                    .collect::<Vec<_>>()
    }
}

impl Eq for ResolvedFunction {}

impl Hash for ResolvedFunction {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        for s in &self.partial_arcsorts {
            s.name().hash(state);
        }
    }
}

impl BaseValue for ResolvedFunction {}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ResolvedFunctionId {
    Lookup(egglog_bridge::TableAction),
    Prim(ExternalFunctionId),
}

// (unstable-app <function> [<arg1>, <arg2>, ...])
#[derive(Clone)]
struct Apply {
    name: String,
    function: Arc<FunctionSort>,
}

impl Primitive for Apply {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        let mut sorts: Vec<ArcSort> = vec![self.function.clone()];
        sorts.extend(self.function.inputs.clone());
        sorts.push(self.function.output.clone());
        SimpleTypeConstraint::new(self.name(), sorts, span.clone()).into_box()
    }

    fn apply(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
        let (fc, args) = args.split_first().unwrap();
        let fc = exec_state
            .container_values()
            .get_val::<FunctionContainer>(*fc)
            .unwrap()
            .clone();
        fc.apply(exec_state, args)
    }
}

impl FunctionContainer {
    /// Call function (primitive or table) `name` with value args `args` and return the value.
    ///
    /// Public so that other primitive sorts (external or internal) have access.
    pub fn apply(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
        let args: Vec<_> = self.1.iter().map(|(_, x)| x).chain(args).copied().collect();
        match &self.0 {
            ResolvedFunctionId::Lookup(action) => action.lookup(exec_state, &args),
            ResolvedFunctionId::Prim(prim) => exec_state.call_external_func(*prim, &args),
        }
    }
}
