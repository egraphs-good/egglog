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
use super::*;

/// A function value is a name of a function, a list of partially applied arguments (values and sort)
/// Note that we must store the actual arcsorts so we can return them when returning inner values
/// and when canonicalizing
#[derive(Clone, Debug)]
pub struct OldFunctionContainer(Symbol, Vec<(ArcSort, Value)>);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NewFunctionContainer(ResolvedFunctionId, Vec<(bool, core_relations::Value)>);

impl OldFunctionContainer {
    /// Remove the arcsorts to make this hashable
    /// The arg values contain the sort name anyways
    fn hashable(&self) -> (Symbol, Vec<&Value>) {
        (self.0, self.1.iter().map(|(_, v)| v).collect())
    }
}

impl Hash for OldFunctionContainer {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.hashable().hash(state);
    }
}

impl PartialEq for OldFunctionContainer {
    fn eq(&self, other: &Self) -> bool {
        self.hashable() == other.hashable()
    }
}

impl Eq for OldFunctionContainer {}

impl Container for NewFunctionContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn Rebuilder) -> bool {
        let mut changed = false;
        for (do_rebuild, old) in &mut self.1 {
            if *do_rebuild {
                let new = rebuilder.rebuild_val(*old);
                changed |= *old != new;
                *old = new;
            }
        }
        changed
    }
    fn iter(&self) -> impl Iterator<Item = core_relations::Value> + '_ {
        self.1.iter().map(|(_, v)| v).copied()
    }
}

#[derive(Debug)]
pub struct FunctionSort {
    name: Symbol,
    inputs: Vec<ArcSort>,
    output: ArcSort,
    functions: Mutex<IndexSet<OldFunctionContainer>>,
}

impl FunctionSort {
    pub fn name(&self) -> Symbol {
        self.name
    }

    pub fn inputs(&self) -> &[ArcSort] {
        &self.inputs
    }

    pub fn output(&self) -> ArcSort {
        self.output.clone()
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
                .get_sort_by_name(output)
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
                                .get_sort_by_name(arg)
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

    fn column_ty(&self, _backend: &egglog_bridge::EGraph) -> ColumnTy {
        ColumnTy::Id
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend.register_container_ty::<NewFunctionContainer>();
        backend.primitives_mut().register_type::<ResolvedFunction>();
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

    fn serialized_name(&self, value: &core_relations::Value) -> Symbol {
        // TODO(yz): I don't have a handle to the new backend, 
        // so I don't know what the function name actually is
        // 
        // OldFunctionContainer::load(self, value).0
        "unstable-fn".into()
    }

    fn inner_values(&self, value: &Value) -> Vec<(ArcSort, Value)> {
        let functions = self.functions.lock().unwrap();
        let input_values = functions.get_index(value.bits as usize).unwrap();
        input_values.1.clone()
    }

    fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
        let OldFunctionContainer(name, inputs) = OldFunctionContainer::load(self, value);
        let mut changed = false;
        let mut new_outputs = vec![];
        for (s, mut v) in inputs.into_iter() {
            changed |= s.canonicalize(&mut v, unionfind);
            new_outputs.push((s, v));
        }
        *value = OldFunctionContainer(name, new_outputs).store(self);
        changed
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

    fn extract_term(
        &self,
        _egraph: &EGraph,
        value: Value,
        extractor: &Extractor,
        termdag: &mut TermDag,
    ) -> Option<(Cost, Term)> {
        let OldFunctionContainer(name, inputs) = OldFunctionContainer::load(self, &value);
        let (cost, args) = inputs.into_iter().try_fold(
            (1usize, vec![termdag.lit(Literal::String(name))]),
            |(cost, mut args), (sort, value)| {
                let (new_cost, term) = extractor.find_best(value, termdag, &sort)?;
                args.push(term);
                Some((cost.saturating_add(new_cost), args))
            },
        )?;

        Some((cost, termdag.app("unstable-fn".into(), args)))
    }
}

impl IntoSort for OldFunctionContainer {
    type Sort = FunctionSort;
    fn store(self, sort: &Self::Sort) -> Value {
        let mut functions = sort.functions.lock().unwrap();
        let (i, _) = functions.insert_full(self);
        Value {
            #[cfg(debug_assertions)]
            tag: sort.name,
            bits: i as u64,
        }
    }
}

impl FromSort for OldFunctionContainer {
    type Sort = FunctionSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let functions = sort.functions.lock().unwrap();
        functions.get_index(value.bits as usize).unwrap().clone()
    }
}

/// Takes a string and any number of partially applied args of any sort and returns a function
struct FunctionCTorTypeConstraint {
    name: Symbol,
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
                        head: self.name,
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
                                head: self.name,
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
            constraint::assign(arguments[0].clone(), Arc::new(StringSort)),
            output_sort_constraint,
        ]
    }
}

// (unstable-fn "name" [<arg1>, <arg2>, ...])
#[derive(Clone)]
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
        _egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
        let name = Symbol::load(&StringSort, &values[0]);

        assert!(values.len() == sorts.0.len());
        let args: Vec<(ArcSort, Value)> = values[1..]
            .iter()
            .zip(&sorts.0[1..])
            .map(|(value, sort)| (sort.clone(), *value))
            .collect();

        Some(OldFunctionContainer(name, args).store(&self.function))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResolvedFunction {
    pub id: ResolvedFunctionId,
    pub do_rebuild: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ResolvedFunctionId {
    Lookup(egglog_bridge::Lookup),
    Prim(ExternalFunctionId),
}

impl ExternalFunction for Ctor {
    fn invoke(
        &self,
        exec_state: &mut ExecutionState,
        args: &[core_relations::Value],
    ) -> Option<core_relations::Value> {
        let (rf, args) = args.split_first().unwrap();
        let ResolvedFunction { id, do_rebuild } = exec_state.prims().unwrap(*rf);
        let args = do_rebuild.iter().zip(args).map(|(b, x)| (*b, *x)).collect();
        let y = NewFunctionContainer(id, args);
        Some(exec_state.clone().containers().register_val(y, exec_state))
    }
}

// (unstable-app <function> [<arg1>, <arg2>, ...])
#[derive(Clone)]
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
        Some(self.function.apply(&values[0], &values[1..], egraph))
    }
}

impl ExternalFunction for Apply {
    fn invoke(
        &self,
        exec_state: &mut ExecutionState,
        args: &[core_relations::Value],
    ) -> Option<core_relations::Value> {
        let (fc, args) = args.split_first().unwrap();
        let fc = exec_state
            .containers()
            .get_val::<NewFunctionContainer>(*fc)
            .unwrap()
            .clone();
        fc.apply(exec_state, args)
    }
}

impl FunctionSort {
    /// Call function (primitive or table) <name> with value args <args> and return the value.
    ///
    /// Public so that other primitive sorts (external or internal) have access.
    pub fn apply(&self, fn_value: &Value, arg_values: &[Value], egraph: &mut EGraph) -> Value {
        let OldFunctionContainer(name, args) = OldFunctionContainer::load(self, fn_value);
        let (mut types, mut args): (Vec<_>, Vec<_>) = args.into_iter().unzip();
        types.extend(self.inputs.clone());
        types.push(self.output.clone());
        args.extend(arg_values);

        // Make a call with temp vars as each of the args
        let resolved_call =
            ResolvedCall::from_resolution(&name, types.as_slice(), &egraph.type_info);
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
        let resolved_args = arg_vars.into_iter().map(|v| var!(v));
        let expr = call!(resolved_call, resolved_args);
        // Similar to how the merge function is created in `Function::new`
        let (actions, mapped_expr) = expr
            .to_core_actions(
                &egraph.type_info,
                &mut binding.clone(),
                &mut egraph.parser.symbol_gen,
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
}

impl NewFunctionContainer {
    /// Call function (primitive or table) <name> with value args <args> and return the value.
    ///
    /// Public so that other primitive sorts (external or internal) have access.
    pub fn apply(
        &self,
        exec_state: &mut ExecutionState,
        args: &[core_relations::Value],
    ) -> Option<core_relations::Value> {
        let args: Vec<_> = self.1.iter().map(|(_, x)| x).chain(args).copied().collect();
        match &self.0 {
            ResolvedFunctionId::Lookup(lookup) => lookup.run(exec_state, &args),
            ResolvedFunctionId::Prim(prim) => exec_state.call_external_func(*prim, &args),
        }
    }
}
