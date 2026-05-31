//! Sort to represent functions as values.
//!
//! To declare the sort, you must specify the exact number of arguments and the sort of each, followed by the output sort:
//! `(sort IntToString (UnstableFn (i64) String))`
//!
//! To create a function value, use the `(unstable-fn "name" [<partial args>])` primitive and to apply it use the `(unstable-app function arg1 arg2 ...)` primitive.
//! The number of args must match the number of arguments in the function sort.
//!
//! The value is stored similar to the `vec` sort, as an index into a set, where each item in
//! the set is a `(Symbol, Vec<(Sort, Value)>)` pairs. The Symbol is the function name, and the `Vec<(Sort, Value)>` is
//! the list of partially applied arguments.
use std::any::TypeId;
use std::sync::Mutex;

use crate::exec_state::Internal;
use enum_map::EnumMap;

use super::*;

#[derive(Clone, Debug)]
pub struct FunctionContainer(
    pub ResolvedFunctionId,
    pub Vec<(ArcSort, Value)>,
    pub String,
    /// Pre-registered panic id used by `FunctionContainer::apply`
    /// on capability mismatch (see [`ResolvedFunction::panic_id`]).
    /// Excluded from equality/hash — two function values that differ
    /// only in their panic id are still the same function value.
    pub ExternalFunctionId,
);

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
        self.inputs
            .iter()
            .any(|s| s.is_eq_sort() || s.is_eq_container_sort())
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
        eg.add_pure_primitive(
            Ctor {
                name: "unstable-fn".into(),
                function: self.clone(),
            },
            None,
        );
        eg.add_pure_primitive(
            Apply {
                name: "unstable-app".into(),
                function: self.clone(),
            },
            None,
        );

        register_vec_primitives_for_function(eg, self.clone());
        register_multiset_primitives_for_function(eg, self.clone());
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
            // The arguments contains the return sort as well as the function name
            let n_partial_args = arguments.len() - 2;
            if let Some(func_type) = typeinfo.get_func_type(name) {
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

            if let Some(primitives) = typeinfo.get_prims(name) {
                // Primitive targets are checked by asking each overload whether
                // a full call would typecheck after stitching together:
                //
                //   explicit partial args from `(unstable-fn "name" ...)`
                //   + synthetic future args from the requested UnstableFn sort
                //   + one synthetic output term
                //
                // For example, `(unstable-fn "+" old)` as `UnstableFn (i64) i64`
                // checks each `+` overload as though it were called with
                // `(old, future_arg) -> future_output`. The i64 overload matches;
                // f64/string/etc. overloads become impossible constraints. If
                // `old` is omitted, the same sort only provides one future arg,
                // so no binary `+` overload has enough arguments to match.
                let mut primitive_constraints = Vec::with_capacity(primitives.len());
                for primitive in primitives {
                    let mut primitive_args = arguments[1..arguments.len() - 1].to_vec();
                    primitive_constraints.push(Vec::new());
                    let alternative_constraints = primitive_constraints.last_mut().unwrap();
                    for (index, sort) in self
                        .function
                        .inputs
                        .iter()
                        .chain(once(&self.function.output))
                        .enumerate()
                    {
                        let term = AtomTerm::Var(
                            self.span.clone(),
                            format!(
                                "__unstable_fn_target_{}_{}_arg_{index}",
                                name,
                                self.function.name()
                            ),
                        );
                        alternative_constraints
                            .push(constraint::assign(term.clone(), sort.clone()));
                        primitive_args.push(term);
                    }
                    alternative_constraints.extend(
                        primitive
                            .primitive
                            .get_type_constraints(&self.span)
                            .get(&primitive_args, typeinfo),
                    );
                }

                // No alternatives is defensive, one alternative is ordinary
                // non-overloaded primitive resolution, and multiple alternatives
                // are overloaded primitives such as `+`; the xor lets the type
                // solver pick exactly one viable overload.
                return match primitive_constraints.len() {
                    0 => vec![constraint::impossible(
                        constraint::ImpossibleConstraint::ArityMismatch {
                            atom: core::Atom {
                                span: self.span.clone(),
                                head: self.name.clone(),
                                args: arguments.to_vec(),
                            },
                            expected: n_partial_args + self.function.inputs.len() + 2,
                        },
                    )],
                    1 => once(output_sort_constraint)
                        .chain(primitive_constraints.pop().unwrap())
                        .collect(),
                    _ => vec![
                        output_sort_constraint,
                        constraint::xor(
                            primitive_constraints
                                .into_iter()
                                .map(constraint::and)
                                .collect(),
                        ),
                    ],
                };
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

// `Ctor` (`unstable-fn "name" [...]`) builds a `FunctionContainer` and
// interns it via `register_container`. Container interning is idempotent,
// so it's safe in every context; declaring `State = PureState`
// permits this primitive inside rule queries, actions, and global
// contexts alike.
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
}

impl PurePrim for Ctor {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let (rf, args) = args.split_first().unwrap();
        let ResolvedFunction {
            id,
            partial_arcsorts,
            name,
            panic_id,
        } = state.base_values().unwrap(*rf);
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
        let y = FunctionContainer(id, args, name, panic_id);
        Some(state.register_container(y))
    }
}

#[derive(Clone, Debug)]
pub struct ResolvedFunction {
    pub id: ResolvedFunctionId,
    pub partial_arcsorts: Vec<ArcSort>,
    pub name: String,
    /// Pre-registered runtime-panic id used by `FunctionContainer::apply`
    /// when an `unstable-fn` value is applied in a context where its
    /// wrapped function isn't valid (e.g. constructor minting in a
    /// rule body without `:naive`). Calling this id writes a
    /// descriptive message to the egraph's panic side channel and
    /// triggers early stop, so `run_rules` returns an `Err` rather
    /// than the calling thread unwinding.
    pub panic_id: ExternalFunctionId,
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
    /// Wraps a constructor-table lookup. Only admissible in
    /// write-capable contexts (`Write`/`Full`), where
    /// `FunctionContainer::apply` mints a fresh eclass via
    /// `lookup_or_insert`. In any read-only context (`Read`/`Pure`)
    /// it triggers the pre-registered runtime panic — a no-mint
    /// constructor would silently miss instead of producing the
    /// eclass the user asked for, so the call is rejected outright.
    Constructor(egglog_bridge::TableAction),
    /// Wraps a `(function …)` lookup — any non-constructor function,
    /// regardless of its `:merge` strategy. `FunctionContainer::apply`
    /// allows this only in DB-read-capable contexts (`Read`/`Full`);
    /// `Pure` and `Write` would be untracked seminaive reads.
    Function(egglog_bridge::TableAction),
    /// Wraps a primitive. Carries the unique exact-signature runtime
    /// id found for each context at build time. At dispatch time
    /// `FunctionContainer::apply` picks the id for the application
    /// context — so the runtime selection is independent of the
    /// build-site context, and an `unstable-fn` value may flow freely
    /// from one context to another.
    Primitive {
        context_ids: EnumMap<crate::Context, Option<ExternalFunctionId>>,
    },
}

// (unstable-app <function> [<arg1>, <arg2>, ...])
//
// Registered as a `PurePrim`; `FunctionContainer::apply` reads the
// runtime context to dispatch. Distinct `FunctionSort`s produce
// different signature keys, so `unstable-app` for `MathFn` stays a
// separate overload from `unstable-app` for `i64Fun`.

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
        SimpleTypeConstraint::new(&self.name, sorts, span.clone()).into_box()
    }
}

impl PurePrim for Apply {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let (fc_val, args) = args.split_first().unwrap();
        let fc = state
            .container_values()
            .get_val::<FunctionContainer>(*fc_val)
            .unwrap()
            .clone();
        state.apply_function(&fc, args)
    }
}

impl FunctionContainer {
    /// Apply the wrapped function. `state` is always a `PureState`
    /// (the type every primitive's `apply` receives). The surrounding
    /// context is stamped onto that state by the primitive wrapper, so
    /// callers do not pass a second copy of the same context.
    pub(crate) fn apply<'a, 'db>(
        &self,
        state: &mut crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value>
    where
        'db: 'a,
    {
        let ctx = state.ctx();
        let args: Vec<_> = self.1.iter().map(|(_, x)| x).chain(args).copied().collect();
        let can_mint = matches!(ctx, crate::Context::Write | crate::Context::Full);
        let can_read = matches!(ctx, crate::Context::Read | crate::Context::Full);
        let panic_id = self.3;
        // On capability mismatch, trigger the egglog runtime panic
        // pre-registered at the `unstable-fn` build site (see
        // `BackendRule::prim`). The panic writes to the egraph's
        // panic side channel and triggers early stop, so `run_rules`
        // surfaces the misuse as an `Err`.
        let mismatch = |state: &mut crate::PureState<'a, 'db>| -> Option<Value> {
            state.call_external_func(panic_id, &[])
        };
        match &self.0 {
            ResolvedFunctionId::Constructor(action) => {
                if can_mint {
                    action.lookup_or_insert(state.raw_exec_state(), &args)
                } else {
                    mismatch(state)
                }
            }
            ResolvedFunctionId::Function(action) => {
                if can_read {
                    action.lookup(state.raw_exec_state(), &args)
                } else {
                    mismatch(state)
                }
            }
            ResolvedFunctionId::Primitive { context_ids } => {
                // Pick the runtime id whose context matches the
                // application ctx.
                match context_ids[ctx] {
                    Some(id) => state.call_external_func(id, &args),
                    None => mismatch(state),
                }
            }
        }
    }
}
