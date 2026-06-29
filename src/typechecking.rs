use std::hash::Hasher;

use crate::Context;
use crate::{
    core::{CoreActionContext, CoreRule, GenericActionsExt, ResolvedCall},
    *,
};
use ast::{
    MappedExprExt, ResolvedAction, ResolvedExpr, ResolvedFact, ResolvedRule, ResolvedVar, Rule,
    RuleEvalMode,
};
use core_relations::ExternalFunction;
use egglog_ast::generic_ast::GenericAction;
use egglog_bridge::ActionRegistry;
use enum_map::EnumMap;
use std::sync::{Arc, RwLock};

// `ExternalFunction` wrapper for `PurePrim`. Holds the primitive
// directly so the dispatch chain `external_funcs[id].invoke(...)` →
// `T::apply(...)` is just one vtable hop plus a direct call — no
// closure indirection that defeats inlining.
#[derive(Clone)]
struct PurePrimWrapper<T> {
    prim: T,
    /// The call-site [`Context`] this wrapper stamps onto the
    /// `PureState` before dispatching. `register_per_context` commits
    /// one wrapper per valid `Context` for the trait, so the
    /// typechecker's pick at each call site is encoded directly here.
    ctx: Context,
}

impl<T: PurePrim + Clone> ExternalFunction for PurePrimWrapper<T> {
    fn invoke(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
        self.prim.apply(PureState::wrap(exec_state, self.ctx), args)
    }
}

// `ExternalFunction` wrapper for primitives that need the
// `ActionRegistry` (`ReadPrim`, `WritePrim`, `FullPrim`). One generic
// over the `Wrap` strategy that knows how to construct the right
// state type and dispatch to the primitive's `apply`.
#[derive(Clone)]
struct RegistryPrimWrapper<T, S> {
    prim: T,
    registry: Arc<RwLock<ActionRegistry>>,
    /// Stamped onto the state wrapper.
    ctx: Context,
    _wrap: std::marker::PhantomData<fn() -> S>,
}

trait RegistryWrap<T>: Clone + Send + Sync {
    fn invoke(
        prim: &T,
        exec_state: &mut ExecutionState,
        ctx: Context,
        args: &[Value],
        registry: &ActionRegistry,
    ) -> Option<Value>;
}

#[derive(Clone)]
struct WrapRead;
impl<T: ReadPrim> RegistryWrap<T> for WrapRead {
    #[inline]
    fn invoke(
        prim: &T,
        exec_state: &mut ExecutionState,
        ctx: Context,
        args: &[Value],
        registry: &ActionRegistry,
    ) -> Option<Value> {
        prim.apply(ReadState::wrap(exec_state, registry, ctx), args)
    }
}
#[derive(Clone)]
struct WrapWrite;
impl<T: WritePrim> RegistryWrap<T> for WrapWrite {
    #[inline]
    fn invoke(
        prim: &T,
        exec_state: &mut ExecutionState,
        ctx: Context,
        args: &[Value],
        registry: &ActionRegistry,
    ) -> Option<Value> {
        prim.apply(WriteState::wrap(exec_state, registry, ctx), args)
    }
}
#[derive(Clone)]
struct WrapFull;
impl<T: FullPrim> RegistryWrap<T> for WrapFull {
    #[inline]
    fn invoke(
        prim: &T,
        exec_state: &mut ExecutionState,
        ctx: Context,
        args: &[Value],
        registry: &ActionRegistry,
    ) -> Option<Value> {
        prim.apply(FullState::wrap(exec_state, registry, ctx), args)
    }
}

impl<T: Clone + Send + Sync + 'static, S: RegistryWrap<T> + 'static> ExternalFunction
    for RegistryPrimWrapper<T, S>
{
    fn invoke(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
        let registry = self.registry.read().unwrap();
        S::invoke(&self.prim, exec_state, self.ctx, args, &registry)
    }
}

#[derive(Clone, Debug)]
pub struct FuncType {
    pub name: String,
    pub subtype: FunctionSubtype,
    pub input: Vec<ArcSort>,
    pub output: ArcSort,
}

impl PartialEq for FuncType {
    fn eq(&self, other: &Self) -> bool {
        if self.name == other.name
            && self.subtype == other.subtype
            && self.output.name() == other.output.name()
        {
            if self.input.len() != other.input.len() {
                return false;
            }
            for (a, b) in self.input.iter().zip(other.input.iter()) {
                if a.name() != b.name() {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }
}

impl Eq for FuncType {}

impl Hash for FuncType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.subtype.hash(state);
        self.output.name().hash(state);
        for inp in &self.input {
            inp.name().hash(state);
        }
    }
}
/// Validators take a termdag and arguments (as TermIds) and return
/// a newly computed TermId if the primitive application is valid,
/// or None if it is invalid.
pub type PrimitiveValidator = Arc<dyn Fn(&mut TermDag, &[TermId]) -> Option<TermId> + Send + Sync>;

#[derive(Clone)]
pub struct PrimitiveWithId {
    pub(crate) primitive: Arc<dyn Primitive>,
    pub(crate) validator: Option<PrimitiveValidator>,
    /// Runtime entrypoints for the contexts this primitive is valid in.
    /// The primitive definition is stored once, while each context keeps
    /// its own backend id so higher-order dispatch can still recover the
    /// application context at runtime.
    pub(crate) context_ids: EnumMap<Context, Option<ExternalFunctionId>>,
}

impl PrimitiveWithId {
    /// Takes the full signature of a primitive (both input and output types).
    /// Returns whether the primitive is compatible with this signature.
    pub fn accept(&self, tys: &[Arc<dyn Sort>], typeinfo: &TypeInfo) -> bool {
        let mut constraints = vec![];
        let lits: Vec<_> = (0..tys.len())
            .map(|i| AtomTerm::Literal(Span::Panic, Literal::Int(i as i64)))
            .collect();
        for (lit, ty) in lits.iter().zip(tys.iter()) {
            constraints.push(constraint::assign(lit.clone(), ty.clone()))
        }
        constraints.extend(
            self.primitive
                .get_type_constraints(&Span::Panic)
                .get(&lits, typeinfo),
        );
        let problem = Problem {
            constraints,
            range: HashSet::default(),
        };
        problem.solve(|sort| sort.name()).is_ok()
    }

    /// Returns whether this primitive has a runtime entrypoint for `context`.
    pub fn is_valid_in_context(&self, context: Context) -> bool {
        self.context_ids[context].is_some()
    }
}

impl Debug for PrimitiveWithId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Prim({})", self.primitive.name())
    }
}

/// Stores resolved typechecking information.
#[derive(Clone, Default)]
pub struct TypeInfo {
    mksorts: HashMap<String, MkSort>,
    // TODO(yz): I want to get rid of this as now we have user-defined primitives and constraint based type checking
    reserved_primitives: HashSet<&'static str>,
    pub(crate) sorts: HashMap<String, Arc<dyn Sort>>,
    primitives: HashMap<String, Vec<PrimitiveWithId>>,
    func_types: HashMap<String, FuncType>,
    pub(crate) global_sorts: HashMap<String, ArcSort>,
    /// Sorts that do not allow union (e.g., from `:no-union` sorts or relations).
    pub(crate) non_unionable_sorts: HashSet<String>,
}

// These methods need to be on the `EGraph` in order to
// register sorts and primitives with the backend.
impl EGraph {
    /// Add a user-defined sort to the e-graph.
    ///
    /// Also look at [`prelude::add_base_sort`] for a convenience method for adding user-defined sorts
    pub fn add_sort<S: Sort + 'static>(&mut self, sort: S, span: Span) -> Result<(), TypeError> {
        self.add_arcsort(Arc::new(sort), span)
    }

    /// Declare a sort. This corresponds to the `sort` keyword in egglog.
    /// It can either declares a new [`EqSort`] if `presort_and_args` is not provided,
    /// or an instantiation of a presort (e.g., containers like `Vec`).
    pub fn declare_sort(
        &mut self,
        name: impl Into<String>,
        presort_and_args: &Option<(String, Vec<Expr>)>,
        span: Span,
    ) -> Result<(), TypeError> {
        let name = name.into();
        if self.type_info.func_types.contains_key(&name) {
            return Err(TypeError::FunctionAlreadyBound(name, span));
        }

        let sort = match presort_and_args {
            None => Arc::new(EqSort { name }),
            Some((presort, args)) => {
                if let Some(mksort) = self.type_info.mksorts.get(presort) {
                    mksort(&mut self.type_info, name, args)?
                } else {
                    return Err(TypeError::PresortNotFound(presort.clone(), span));
                }
            }
        };

        self.add_arcsort(sort, span)
    }

    /// Add a user-defined sort to the e-graph.
    pub fn add_arcsort(&mut self, sort: ArcSort, span: Span) -> Result<(), TypeError> {
        sort.register_type(&mut self.backend);

        let name = sort.name();
        match self.type_info.sorts.entry(name.to_owned()) {
            HEntry::Occupied(_) => Err(TypeError::SortAlreadyBound(name.to_owned(), span)),
            HEntry::Vacant(e) => {
                e.insert(sort.clone());
                sort.register_primitives(self);
                Ok(())
            }
        }
    }

    /// Register a [`PurePrim`]. Pass `None` for the validator if not
    /// using the proof checker.
    ///
    /// Pick the trait whose state wrapper matches the body's needs:
    /// [`PurePrim`] for pure ops, [`WritePrim`] for writes,
    /// [`ReadPrim`] for table reads, [`FullPrim`] for both. The Rust
    /// type checker enforces the body only uses methods the chosen
    /// state allows.
    pub fn add_pure_primitive<T>(&mut self, x: T, validator: Option<PrimitiveValidator>)
    where
        T: PurePrim + Clone,
    {
        self.register_per_context(x, validator, PureState::valid_contexts(), |x, ctx| {
            Box::new(PurePrimWrapper { prim: x, ctx })
        });
    }

    /// Register a [`WritePrim`]. Pass `None` for the validator if not
    /// using the proof checker.
    pub fn add_write_primitive<T>(&mut self, x: T, validator: Option<PrimitiveValidator>)
    where
        T: WritePrim + Clone,
    {
        self.register_registry_primitive::<T, WrapWrite>(
            x,
            validator,
            WriteState::valid_contexts(),
        );
    }

    /// Register a [`ReadPrim`]. Pass `None` for the validator if not
    /// using the proof checker.
    pub fn add_read_primitive<T>(&mut self, x: T, validator: Option<PrimitiveValidator>)
    where
        T: ReadPrim + Clone,
    {
        self.register_registry_primitive::<T, WrapRead>(x, validator, ReadState::valid_contexts());
    }

    /// Register a [`FullPrim`]. Pass `None` for the validator if not
    /// using the proof checker.
    pub fn add_full_primitive<T>(&mut self, x: T, validator: Option<PrimitiveValidator>)
    where
        T: FullPrim + Clone,
    {
        self.register_registry_primitive::<T, WrapFull>(x, validator, FullState::valid_contexts());
    }

    fn register_registry_primitive<T, S>(
        &mut self,
        x: T,
        validator: Option<PrimitiveValidator>,
        valid_ctxs: &[Context],
    ) where
        T: Primitive + Clone,
        S: RegistryWrap<T> + 'static,
    {
        let registry = self.backend.action_registry().clone();
        self.register_per_context(x, validator, valid_ctxs, move |x, ctx| {
            Box::new(RegistryPrimWrapper::<T, S> {
                prim: x,
                registry: registry.clone(),
                ctx,
                _wrap: std::marker::PhantomData,
            })
        });
    }

    /// Shared registration engine. Stores one primitive definition, plus
    /// one runtime id per valid [`Context`]. Each wrapper carries its
    /// specific context stamped onto the state wrapper at invoke time.
    ///
    /// The typechecker filters by the context-id mask at each call site;
    /// an `unstable-fn` value built around the primitive bakes *all*
    /// signature-matching context ids, and `FunctionContainer::apply`
    /// picks the one whose context matches the application ctx — so
    /// values flow freely across contexts.
    fn register_per_context<T, F>(
        &mut self,
        x: T,
        validator: Option<PrimitiveValidator>,
        valid_ctxs: &[Context],
        mut build_wrapper: F,
    ) where
        T: Primitive + Clone,
        F: FnMut(T, Context) -> Box<dyn ExternalFunction>,
    {
        let primitive: Arc<dyn Primitive> = Arc::new(x.clone());
        let name = primitive.name().to_owned();
        let context_ids = EnumMap::from_fn(|ctx| {
            valid_ctxs.contains(&ctx).then(|| {
                self.backend
                    .register_external_func(build_wrapper(x.clone(), ctx))
            })
        });
        self.type_info
            .primitives
            .entry(name)
            .or_default()
            .push(PrimitiveWithId {
                primitive,
                validator,
                context_ids,
            });
    }
}

impl EGraph {
    pub(crate) fn typecheck_program(
        &mut self,
        program: &Vec<NCommand>,
    ) -> Result<Vec<ResolvedNCommand>, TypeError> {
        let mut result = vec![];
        for command in program {
            result.push(self.typecheck_command(command)?);
        }
        Ok(result)
    }

    fn typecheck_command(&mut self, command: &NCommand) -> Result<ResolvedNCommand, TypeError> {
        let symbol_gen = &mut self.parser.symbol_gen;

        let command: ResolvedNCommand = match command {
            NCommand::Function(fdecl) => {
                let resolved = self.type_info.typecheck_function(symbol_gen, fdecl)?;
                // If this is a let binding, add it to global_sorts
                // This preserves bahavior for lets after desugaring
                if resolved.internal_let {
                    let output_sort = self.type_info.sorts.get(&fdecl.schema.output).unwrap();
                    self.type_info
                        .global_sorts
                        .insert(fdecl.name.clone(), output_sort.clone());
                }
                ResolvedNCommand::Function(resolved)
            }
            NCommand::NormRule { rule } => ResolvedNCommand::NormRule {
                rule: self
                    .type_info
                    .typecheck_rule(symbol_gen, rule, self.seminaive)?,
            },
            NCommand::Sort {
                span,
                name,
                presort_and_args,
                uf,
                proof_func,
                unionable,
            } => {
                // Note this is bad since typechecking should be pure and idempotent
                // Otherwise typechecking the same program twice will fail
                self.declare_sort(name.clone(), presort_and_args, span.clone())?;
                // Mark as non-unionable if the sort declaration says so
                if !unionable {
                    self.type_info.non_unionable_sorts.insert(name.clone());
                }
                ResolvedNCommand::Sort {
                    span: span.clone(),
                    name: name.clone(),
                    presort_and_args: presort_and_args.clone(),
                    uf: uf.clone(),
                    proof_func: proof_func.clone(),
                    unionable: *unionable,
                }
            }
            NCommand::CoreAction(action @ Action::Let(span, var, _)) => {
                let action = self.type_info.typecheck_standalone_action(
                    symbol_gen,
                    action,
                    &Default::default(),
                    Context::Full,
                )?;
                self.ensure_global_name_prefix(span, var)?;
                let ResolvedAction::Let(_, resolved_var, _) = &action else {
                    unreachable!("typechecking an Action::Let should return ResolvedAction::Let")
                };
                self.type_info
                    .global_sorts
                    .insert(resolved_var.name.clone(), resolved_var.sort.clone());
                ResolvedNCommand::CoreAction(action)
            }
            NCommand::CoreAction(action) => {
                ResolvedNCommand::CoreAction(self.type_info.typecheck_standalone_action(
                    symbol_gen,
                    action,
                    &Default::default(),
                    Context::Full,
                )?)
            }
            NCommand::Extract(span, expr, variants) => {
                let res_expr = self.type_info.typecheck_standalone_expr(
                    symbol_gen,
                    expr,
                    &Default::default(),
                    Context::Full,
                )?;

                let res_variants = self.type_info.typecheck_standalone_expr(
                    symbol_gen,
                    variants,
                    &Default::default(),
                    Context::Full,
                )?;
                if res_variants.output_type().name() != I64Sort.name() {
                    return Err(TypeError::Mismatch {
                        expr: variants.clone(),
                        expected: I64Sort.to_arcsort(),
                        actual: res_variants.output_type(),
                    });
                }

                ResolvedNCommand::Extract(span.clone(), res_expr, res_variants)
            }
            NCommand::Check(span, facts) => ResolvedNCommand::Check(
                span.clone(),
                self.type_info.typecheck_facts(symbol_gen, facts)?,
            ),
            NCommand::Fail(span, cmd) => {
                ResolvedNCommand::Fail(span.clone(), Box::new(self.typecheck_command(cmd)?))
            }
            NCommand::RunSchedule(schedule) => ResolvedNCommand::RunSchedule(
                self.type_info.typecheck_schedule(symbol_gen, schedule)?,
            ),
            NCommand::Pop(span, n) => ResolvedNCommand::Pop(span.clone(), *n),
            NCommand::Push(n) => ResolvedNCommand::Push(*n),
            NCommand::AddRuleset(span, ruleset) => {
                ResolvedNCommand::AddRuleset(span.clone(), ruleset.clone())
            }
            NCommand::UnstableCombinedRuleset(span, name, sub_rulesets) => {
                ResolvedNCommand::UnstableCombinedRuleset(
                    span.clone(),
                    name.clone(),
                    sub_rulesets.clone(),
                )
            }
            NCommand::PrintOverallStatistics(span, file) => {
                ResolvedNCommand::PrintOverallStatistics(span.clone(), file.clone())
            }
            NCommand::PrintFunction(span, table, size, file, mode) => {
                ResolvedNCommand::PrintFunction(
                    span.clone(),
                    table.clone(),
                    *size,
                    file.clone(),
                    *mode,
                )
            }
            NCommand::PrintSize(span, n) => {
                // Should probably also resolve the function symbol here
                ResolvedNCommand::PrintSize(span.clone(), n.clone())
            }
            NCommand::ProveExists(span, constructor) => {
                let func_type = self
                    .type_info
                    .get_func_type(constructor)
                    .ok_or_else(|| TypeError::UnboundFunction(constructor.clone(), span.clone()))?;
                if func_type.subtype != FunctionSubtype::Constructor {
                    return Err(TypeError::ProveExistsRequiresConstructor(
                        constructor.clone(),
                        span.clone(),
                    ));
                }
                ResolvedNCommand::ProveExists(span.clone(), ResolvedCall::Func(func_type.clone()))
            }
            NCommand::Output { span, file, exprs } => {
                let exprs = exprs
                    .iter()
                    .map(|expr| {
                        self.type_info.typecheck_standalone_expr(
                            symbol_gen,
                            expr,
                            &Default::default(),
                            Context::Full,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                ResolvedNCommand::Output {
                    span: span.clone(),
                    file: file.clone(),
                    exprs,
                }
            }
            NCommand::Input { span, name, file } => ResolvedNCommand::Input {
                span: span.clone(),
                name: name.clone(),
                file: file.clone(),
            },
            NCommand::UserDefined(span, name, exprs) => {
                ResolvedNCommand::UserDefined(span.clone(), name.clone(), exprs.clone())
            }
        };
        if let ResolvedNCommand::NormRule { rule } = &command {
            self.warn_for_prefixed_non_globals_in_rule(rule)?;
        }
        Ok(command)
    }

    fn warn_for_prefixed_non_globals_in_var(
        &mut self,
        span: &Span,
        var: &ResolvedVar,
    ) -> Result<(), TypeError> {
        if var.is_global_ref {
            return Ok(());
        }
        if var.name.starts_with(crate::GLOBAL_NAME_PREFIX) {
            self.warn_prefixed_non_globals(span, &var.name)?;
        }
        Ok(())
    }

    fn warn_for_prefixed_non_globals_in_rule(
        &mut self,
        rule: &ResolvedRule,
    ) -> Result<(), TypeError> {
        let mut res: Result<(), TypeError> = Ok(());

        for fact in &rule.body {
            fact.visit_vars(&mut |span, var| {
                if res.is_ok() {
                    res = self.warn_for_prefixed_non_globals_in_var(span, var);
                }
            });
        }

        rule.head.visit_vars(&mut |span, var| {
            if res.is_ok() {
                res = self.warn_for_prefixed_non_globals_in_var(span, var);
            }
        });
        res
    }
}

impl TypeInfo {
    /// Adds a sort constructor to the typechecker's known set of types.
    pub fn add_presort<S: Presort>(&mut self, span: Span) -> Result<(), TypeError> {
        let name = S::presort_name();
        match self.mksorts.entry(name.to_owned()) {
            HEntry::Occupied(_) => Err(TypeError::SortAlreadyBound(name.to_owned(), span)),
            HEntry::Vacant(e) => {
                e.insert(S::make_sort);
                self.reserved_primitives.extend(S::reserved_primitives());
                Ok(())
            }
        }
    }

    /// Returns all sorts that satisfy the type and predicate.
    pub fn get_sorts_by<S: Sort>(&self, pred: impl Fn(&Arc<S>) -> bool) -> Vec<Arc<S>> {
        let mut results = Vec::new();
        for sort in self.sorts.values() {
            let sort = sort.clone().as_arc_any();
            if let Ok(sort) = Arc::downcast(sort)
                && pred(&sort)
            {
                results.push(sort);
            }
        }
        results
    }

    /// Returns all sorts based on the type.
    pub fn get_sorts<S: Sort>(&self) -> Vec<Arc<S>> {
        self.get_sorts_by(|_| true)
    }

    /// Returns a sort that satisfies the type and predicate.
    pub fn get_sort_by<S: Sort>(&self, pred: impl Fn(&Arc<S>) -> bool) -> Arc<S> {
        let results = self.get_sorts_by(pred);
        assert_eq!(
            results.len(),
            1,
            "Expected exactly one sort for type {}",
            std::any::type_name::<S>()
        );
        results.into_iter().next().unwrap()
    }

    /// Returns a sort based on the type.
    pub fn get_sort<S: Sort>(&self) -> Arc<S> {
        self.get_sort_by(|_| true)
    }

    /// Returns all sorts that satisfy the predicate.
    pub fn get_arcsorts_by(&self, f: impl Fn(&ArcSort) -> bool) -> Vec<ArcSort> {
        self.sorts.values().filter(|&x| f(x)).cloned().collect()
    }

    /// Returns a sort based on the predicate.
    pub fn get_arcsort_by(&self, f: impl Fn(&ArcSort) -> bool) -> ArcSort {
        let results = self.get_arcsorts_by(f);
        assert_eq!(
            results.len(),
            1,
            "Expected exactly one sort matching the given predicate"
        );
        results.into_iter().next().unwrap()
    }

    /// Returns the unique sort whose runtime values have Rust type `T`.
    pub fn get_arcsort_for_value_type<T: 'static>(&self) -> ArcSort {
        let results = self.get_arcsorts_by(|s| s.value_type() == Some(std::any::TypeId::of::<T>()));
        assert_eq!(
            results.len(),
            1,
            "Expected exactly one sort for type `{}`",
            std::any::type_name::<T>()
        );
        results.into_iter().next().unwrap()
    }

    /// Check if a sort allows union operations.
    /// A sort is unionable if it's an eq_sort and not marked as non-unionable
    /// (e.g., from `(sort Foo :no-union)` or relation desugaring).
    pub fn is_sort_unionable(&self, sort: &ArcSort) -> bool {
        sort.is_eq_sort() && !self.non_unionable_sorts.contains(sort.name())
    }

    fn function_to_functype(&self, func: &FunctionDecl) -> Result<FuncType, TypeError> {
        let input = func
            .schema
            .input
            .iter()
            .map(|name| {
                if let Some(sort) = self.sorts.get(name) {
                    Ok(sort.clone())
                } else {
                    Err(TypeError::UndefinedSort(name.clone(), func.span.clone()))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output = if let Some(sort) = self.sorts.get(&func.schema.output) {
            Ok(sort.clone())
        } else {
            Err(TypeError::UndefinedSort(
                func.schema.output.clone(),
                func.span.clone(),
            ))
        }?;

        Ok(FuncType {
            name: func.name.clone(),
            subtype: func.subtype,
            input,
            output: output.clone(),
        })
    }

    fn typecheck_function(
        &mut self,
        symbol_gen: &mut SymbolGen,
        fdecl: &FunctionDecl,
    ) -> Result<ResolvedFunctionDecl, TypeError> {
        if self.sorts.contains_key(&fdecl.name) {
            return Err(TypeError::SortAlreadyBound(
                fdecl.name.clone(),
                fdecl.span.clone(),
            ));
        }
        if self.is_primitive(&fdecl.name) {
            return Err(TypeError::PrimitiveAlreadyBound(
                fdecl.name.clone(),
                fdecl.span.clone(),
            ));
        }
        // View tables (with term_constructor) must have at least one input (the e-class)
        if fdecl.term_constructor.is_some() && fdecl.schema.input.is_empty() {
            return Err(TypeError::TermConstructorNoInputs(
                fdecl.name.clone(),
                fdecl.span.clone(),
            ));
        }
        let ftype = self.function_to_functype(fdecl)?;
        if self.func_types.insert(fdecl.name.clone(), ftype).is_some() {
            return Err(TypeError::FunctionAlreadyBound(
                fdecl.name.clone(),
                fdecl.span.clone(),
            ));
        }
        let mut bound_vars = IndexMap::default();
        let output_type = self.sorts.get(&fdecl.schema.output).unwrap();
        if fdecl.subtype == FunctionSubtype::Constructor && !output_type.is_eq_sort() {
            return Err(TypeError::ConstructorOutputNotSort(
                fdecl.name.clone(),
                fdecl.span.clone(),
            ));
        }
        bound_vars.insert("old", (fdecl.span.clone(), output_type.clone()));
        bound_vars.insert("new", (fdecl.span.clone(), output_type.clone()));

        Ok(ResolvedFunctionDecl {
            name: fdecl.name.clone(),
            subtype: fdecl.subtype,
            schema: fdecl.schema.clone(),
            resolved_schema: ResolvedCall::Func(self.func_types.get(&fdecl.name).unwrap().clone()),
            merge: match &fdecl.merge {
                // Merge expressions run as part of action-side table updates:
                // writes are allowed, but live DB reads would be untracked by
                // seminaive rule execution.
                Some(merge) => Some(self.typecheck_standalone_expr(
                    symbol_gen,
                    merge,
                    &bound_vars,
                    Context::Write,
                )?),
                None => None,
            },
            cost: fdecl.cost,
            unextractable: fdecl.unextractable,
            internal_hidden: fdecl.internal_hidden,
            internal_let: fdecl.internal_let,
            span: fdecl.span.clone(),
            term_constructor: fdecl.term_constructor.clone(),
        })
    }

    fn typecheck_schedule(
        &self,
        symbol_gen: &mut SymbolGen,
        schedule: &Schedule,
    ) -> Result<ResolvedSchedule, TypeError> {
        let schedule = match schedule {
            Schedule::Repeat(span, times, schedule) => ResolvedSchedule::Repeat(
                span.clone(),
                *times,
                Box::new(self.typecheck_schedule(symbol_gen, schedule)?),
            ),
            Schedule::Sequence(span, schedules) => {
                let schedules = schedules
                    .iter()
                    .map(|schedule| self.typecheck_schedule(symbol_gen, schedule))
                    .collect::<Result<Vec<_>, _>>()?;
                ResolvedSchedule::Sequence(span.clone(), schedules)
            }
            Schedule::Saturate(span, schedule) => ResolvedSchedule::Saturate(
                span.clone(),
                Box::new(self.typecheck_schedule(symbol_gen, schedule)?),
            ),
            Schedule::Run(span, RunConfig { ruleset, until }) => {
                let until = until
                    .as_ref()
                    .map(|facts| self.typecheck_facts(symbol_gen, facts))
                    .transpose()?;
                ResolvedSchedule::Run(
                    span.clone(),
                    ResolvedRunConfig {
                        ruleset: ruleset.clone(),
                        until,
                    },
                )
            }
        };

        Result::Ok(schedule)
    }

    fn typecheck_rule(
        &self,
        symbol_gen: &mut SymbolGen,
        rule: &Rule,
        global_seminaive: bool,
    ) -> Result<ResolvedRule, TypeError> {
        let Rule {
            span,
            head,
            body,
            name,
            ruleset,
            eval_mode,
            no_decomp,
            include_subsumed,
        } = rule;
        let mut constraints = vec![];

        // Compile with the permissive Read/Full primitive contexts (so the RHS
        // can read the database) when the whole EGraph is non-seminaive, or the
        // rule's own mode requires it (`:naive` / `:unsafe-seminaive`).
        let read_contexts = !global_seminaive
            || matches!(
                eval_mode,
                RuleEvalMode::Naive | RuleEvalMode::UnsafeSeminaive
            );
        let (query_ctx, action_ctx) = if read_contexts {
            (Context::Read, Context::Full)
        } else {
            (Context::Pure, Context::Write)
        };

        let (query, mapped_query) = Facts(body.clone()).to_query(self, symbol_gen);
        constraints.extend(query.get_constraints(self, query_ctx)?);

        let mut binding = query.get_vars();
        // We lower to core actions with `union_to_set_optimization`
        // later in the pipeline. For typechecking we do not need it.
        let mut ctx = CoreActionContext::new(self, &mut binding, symbol_gen, false);
        let (actions, mapped_action) = head.to_core_actions(&mut ctx)?;

        let mut problem = Problem::default();
        problem.add_rule(
            &CoreRule {
                span: span.clone(),
                body: query,
                head: actions,
            },
            self,
            symbol_gen,
            query_ctx,
            action_ctx,
        )?;

        let assignment = problem
            .solve(|sort: &ArcSort| sort.name())
            .map_err(|e| e.to_type_error())?;

        let body: Vec<ResolvedFact> = assignment.annotate_facts(&mapped_query, self, query_ctx);
        let actions: ResolvedActions =
            assignment.annotate_actions(&mapped_action, self, action_ctx)?;

        // Function lookups in actions need the `Full` action context; the
        // `Write` context (`!read_contexts`) can't express them.
        if !read_contexts {
            self.check_no_function_lookups_in_actions(&actions)?;
        }

        Ok(ResolvedRule {
            span: span.clone(),
            body,
            head: actions,
            name: name.clone(),
            ruleset: ruleset.clone(),
            eval_mode: *eval_mode,
            no_decomp: *no_decomp,
            include_subsumed: *include_subsumed,
        })
    }

    fn check_lookup_expr(&self, expr: &ResolvedExpr) -> Result<(), TypeError> {
        if let Some(span) = self.expr_has_function_lookup(expr) {
            return Err(TypeError::LookupInRuleDisallowed(
                "function".to_string(),
                span,
            ));
        }
        Ok(())
    }

    fn check_no_function_lookups_in_actions(
        &self,
        actions: &ResolvedActions,
    ) -> Result<(), TypeError> {
        for action in actions.iter() {
            match action {
                GenericAction::Let(_, _, rhs) => self.check_lookup_expr(rhs)?,
                GenericAction::Set(_, _, args, rhs) => {
                    for arg in args.iter() {
                        self.check_lookup_expr(arg)?;
                    }
                    self.check_lookup_expr(rhs)?;
                }
                GenericAction::Union(_, lhs, rhs) => {
                    self.check_lookup_expr(lhs)?;
                    self.check_lookup_expr(rhs)?;
                }
                GenericAction::Change(_, _, _, args) => {
                    for arg in args.iter() {
                        self.check_lookup_expr(arg)?;
                    }
                }
                GenericAction::Panic(..) => {}
                GenericAction::Expr(_, expr) => self.check_lookup_expr(expr)?,
            }
        }
        Ok(())
    }

    pub fn typecheck_facts(
        &self,
        symbol_gen: &mut SymbolGen,
        facts: &[Fact],
    ) -> Result<Vec<ResolvedFact>, TypeError> {
        let (query, mapped_facts) = Facts(facts.to_vec()).to_query(self, symbol_gen);
        let mut problem = Problem::default();
        // Top-level query-shaped commands (e.g. `check`) are read-only:
        // primitives may inspect the database but not write to it.
        problem.add_query(&query, self, Context::Read)?;
        let assignment = problem
            .solve(|sort: &ArcSort| sort.name())
            .map_err(|e| e.to_type_error())?;
        let annotated_facts = assignment.annotate_facts(&mapped_facts, self, Context::Read);
        Ok(annotated_facts)
    }

    // Standalone expressions/actions use action lowering. Top-level commands
    // pass `Full`; function `:merge` reuses this path with `Write` because
    // merge expressions run during table updates.
    fn typecheck_standalone_actions(
        &self,
        symbol_gen: &mut SymbolGen,
        actions: &Actions,
        binding: &IndexMap<&str, (Span, ArcSort)>,
        context: Context,
    ) -> Result<ResolvedActions, TypeError> {
        let mut binding_set: IndexSet<String> =
            binding.keys().copied().map(str::to_string).collect();
        // We lower to core actions with `union_to_set_optimization`
        // later in the pipeline. For typechecking we do not need it.
        let mut ctx = CoreActionContext::new(self, &mut binding_set, symbol_gen, false);
        let (actions, mapped_action) = actions.to_core_actions(&mut ctx)?;
        let mut problem = Problem::default();

        problem.add_actions(&actions, self, symbol_gen, context)?;

        // add bindings from the context
        for (var, (span, sort)) in binding {
            problem.assign_local_var_type(var, span.clone(), sort.clone())?;
        }

        let assignment = problem
            .solve(|sort: &ArcSort| sort.name())
            .map_err(|e| e.to_type_error())?;

        let annotated_actions = assignment.annotate_actions(&mapped_action, self, context)?;
        Ok(annotated_actions)
    }

    fn typecheck_standalone_expr(
        &self,
        symbol_gen: &mut SymbolGen,
        expr: &Expr,
        binding: &IndexMap<&str, (Span, ArcSort)>,
        context: Context,
    ) -> Result<ResolvedExpr, TypeError> {
        let action = Action::Expr(expr.span(), expr.clone());
        let typechecked_action =
            self.typecheck_standalone_action(symbol_gen, &action, binding, context)?;
        match typechecked_action {
            ResolvedAction::Expr(_, expr) => Ok(expr),
            _ => unreachable!(),
        }
    }

    pub(crate) fn typecheck_expr_with_output(
        &self,
        symbol_gen: &mut SymbolGen,
        expr: &Expr,
        binding: &IndexMap<&str, (Span, ArcSort)>,
        output_sort: ArcSort,
        context: Context,
    ) -> Result<ResolvedExpr, TypeError> {
        let action = Action::Expr(expr.span(), expr.clone());
        let mut binding_set: IndexSet<String> =
            binding.keys().copied().map(str::to_string).collect();
        let mut ctx = CoreActionContext::new(self, &mut binding_set, symbol_gen, false);
        let (actions, mapped_action) = Actions::singleton(action).to_core_actions(&mut ctx)?;
        let mut problem = Problem::default();

        problem.add_actions(&actions, self, symbol_gen, context)?;

        for (var, (span, sort)) in binding {
            problem.assign_local_var_type(var, span.clone(), sort.clone())?;
        }

        let [GenericAction::Expr(_, mapped_expr)] = mapped_action.0.as_slice() else {
            unreachable!("typechecking an expression should produce one expression action")
        };
        let output_atom = mapped_expr.get_corresponding_var_or_lit(self);
        problem.add_binding(output_atom, output_sort.clone());

        let assignment = problem
            .solve(|sort: &ArcSort| sort.name())
            .map_err(|e| e.to_type_error())?;

        let annotated_actions = assignment.annotate_actions(&mapped_action, self, context)?;
        match annotated_actions.0.into_iter().next().unwrap() {
            ResolvedAction::Expr(_, resolved_expr) => {
                let actual = resolved_expr.output_type();
                if actual.name() != output_sort.name() {
                    return Err(TypeError::Mismatch {
                        expr: expr.clone(),
                        expected: output_sort,
                        actual,
                    });
                }
                Ok(resolved_expr)
            }
            _ => unreachable!(),
        }
    }

    fn typecheck_standalone_action(
        &self,
        symbol_gen: &mut SymbolGen,
        action: &Action,
        binding: &IndexMap<&str, (Span, ArcSort)>,
        context: Context,
    ) -> Result<ResolvedAction, TypeError> {
        self.typecheck_standalone_actions(
            symbol_gen,
            &Actions::singleton(action.clone()),
            binding,
            context,
        )
        .map(|v| {
            assert_eq!(v.len(), 1);
            v.0.into_iter().next().unwrap()
        })
    }

    pub fn get_sort_by_name(&self, sym: &str) -> Option<&ArcSort> {
        self.sorts.get(sym)
    }

    pub fn get_prims(&self, sym: &str) -> Option<&[PrimitiveWithId]> {
        self.primitives.get(sym).map(Vec::as_slice)
    }

    pub fn is_primitive(&self, sym: &str) -> bool {
        self.primitives.contains_key(sym) || self.reserved_primitives.contains(sym)
    }

    pub fn primitive_has_validator(&self, id: ExternalFunctionId) -> bool {
        self.primitives
            .values()
            .flat_map(|v| v.iter())
            .any(|p| p.context_ids.iter().any(|(_, pid)| *pid == Some(id)) && p.validator.is_some())
    }

    pub fn get_func_type(&self, sym: &str) -> Option<&FuncType> {
        self.func_types.get(sym)
    }

    pub fn is_constructor(&self, sym: &str) -> bool {
        self.func_types
            .get(sym)
            .is_some_and(|f| f.subtype == FunctionSubtype::Constructor)
    }

    pub fn get_global_sort(&self, sym: &str) -> Option<&ArcSort> {
        self.global_sorts.get(sym)
    }

    pub fn is_global(&self, sym: &str) -> bool {
        self.global_sorts.contains_key(sym)
    }

    /// Check if an expression contains non-global function lookups (FunctionSubtype::Custom calls).
    /// Global function calls are allowed since they get desugared to constructors.
    /// Returns Some(span) if a lookup is found, None otherwise.
    pub fn expr_has_function_lookup(&self, expr: &ResolvedExpr) -> Option<Span> {
        use ast::GenericExpr;

        expr.find(&mut |e| {
            if let GenericExpr::Call(span, ResolvedCall::Func(func_type), _) = e
                && func_type.subtype == FunctionSubtype::Custom
                && !self.is_global(&func_type.name)
            {
                return Some(span.clone());
            }
            None
        })
    }
}

#[derive(Debug, Clone, Error)]
pub enum TypeError {
    #[error("{}\nArity mismatch, expected {expected} args: {expr}", .expr.span())]
    Arity { expr: Expr, expected: usize },
    #[error(
        "{}\n Expect expression {expr} to have type {}, but get type {}",
        .expr.span(), .expected.name(), .actual.name(),
    )]
    Mismatch {
        expr: Expr,
        expected: ArcSort,
        actual: ArcSort,
    },
    #[error("{1}\nUnbound symbol {0}")]
    Unbound(String, Span),
    #[error(
        "{1}\nVariable {0} is ungrounded. A variable is grounded when it appears as an argument to a constructor or function in the query, not just under primitives or equalities."
    )]
    Ungrounded(String, Span),
    #[error("{1}\nUndefined sort {0}")]
    UndefinedSort(String, Span),
    #[error("{1}\nUnbound function {0}")]
    UnboundFunction(String, Span),
    #[error("{1}\nprove-exists requires constructor function, but {0} is not a constructor")]
    ProveExistsRequiresConstructor(String, Span),
    #[error("{1}\nFunction already bound {0}")]
    FunctionAlreadyBound(String, Span),
    #[error("{1}\nSort {0} already declared.")]
    SortAlreadyBound(String, Span),
    #[error("{1}\nPrimitive {0} already declared.")]
    PrimitiveAlreadyBound(String, Span),
    #[error("Function type mismatch: expected {} => {}, actual {} => {}", .1.iter().map(|s| s.name().to_string()).collect::<Vec<_>>().join(", "), .0.name(), .3.iter().map(|s| s.name().to_string()).collect::<Vec<_>>().join(", "), .2.name())]
    FunctionTypeMismatch(ArcSort, Vec<ArcSort>, ArcSort, Vec<ArcSort>),
    #[error("{1}\nPresort {0} not found.")]
    PresortNotFound(String, Span),
    #[error("{}\nFailed to infer a type for: {}", .0.span(), .0)]
    InferenceFailure(Expr),
    #[error("{1}\nVariable {0} was already defined")]
    AlreadyDefined(String, Span),
    #[error("{1}\nThe output type of constructor function {0} must be sort")]
    ConstructorOutputNotSort(String, Span),
    #[error("{1}\nValue lookup of non-constructor function {0} in rule is disallowed.")]
    LookupInRuleDisallowed(String, Span),
    #[error("{1}\nCannot set constructor {0}. Use `union` instead or declare {0} as a function.")]
    SetConstructorDisallowed(String, Span),
    #[error("All alternative definitions considered failed\n{}", .0.iter().map(|e| format!("  {e}\n")).collect::<Vec<_>>().join(""))]
    AllAlternativeFailed(Vec<TypeError>),
    #[error("{}\nCannot union values of sort {}", .1, .0.name())]
    NonEqsortUnion(ArcSort, Span),
    #[error("{}\nCannot union values of sort {} because it is marked as non-unionable (e.g. from a relation)", .1, .0.name())]
    NonUnionableSort(ArcSort, Span),
    #[error(
        "{1}\nView table {0} with :internal-term-constructor must have at least one input (the e-class)."
    )]
    TermConstructorNoInputs(String, Span),
    #[error(
        "{span}\nNon-global variable `{name}` must not start with `{}`.",
        crate::GLOBAL_NAME_PREFIX
    )]
    NonGlobalPrefixed { name: String, span: Span },
    #[error(
        "{span}\nGlobal `{name}` must start with `{}`.",
        crate::GLOBAL_NAME_PREFIX
    )]
    GlobalMissingPrefix { name: String, span: Span },
}

#[cfg(test)]
mod test {
    use crate::{EGraph, Error, typechecking::TypeError};

    #[test]
    fn test_arity_mismatch() {
        let mut egraph = EGraph::default();

        let prog = "
            (relation f (i64 i64))
            (rule ((f a b c)) ())
       ";
        let res = egraph.parse_and_run_program(None, prog);
        match res {
            Err(Error::TypeError(TypeError::Arity {
                expected: 2,
                expr: e,
            })) => {
                assert_eq!(e.span().string(), "(f a b c)");
            }
            _ => panic!("Expected arity mismatch, got: {res:?}"),
        }
    }
}
