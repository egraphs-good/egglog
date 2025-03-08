use crate::{core::CoreRule, *};
use ast::Rule;

#[derive(Clone, Debug)]
pub struct FuncType {
    pub name: Symbol,
    pub subtype: FunctionSubtype,
    pub input: Vec<ArcSort>,
    pub output: ArcSort,
}

/// Stores resolved typechecking information.
/// TODO make these not public, use accessor methods
#[derive(Clone)]
pub struct TypeInfo {
    // get the sort from the sorts name()
    pub presorts: HashMap<Symbol, PreSort>,
    // TODO(yz): I want to get rid of this as now we have user-defined primitives and constraint based type checking
    pub reserved_primitives: HashSet<Symbol>,
    pub sorts: HashMap<Symbol, Arc<dyn Sort>>,
    pub primitives: HashMap<Symbol, Vec<Primitive>>,
    pub func_types: HashMap<Symbol, FuncType>,
    pub global_types: HashMap<Symbol, ArcSort>,
}

impl Default for TypeInfo {
    fn default() -> Self {
        let mut res = Self {
            presorts: Default::default(),
            reserved_primitives: Default::default(),
            sorts: Default::default(),
            primitives: Default::default(),
            func_types: Default::default(),
            global_types: Default::default(),
        };

        res.add_sort(UnitSort, span!()).unwrap();
        res.add_sort(StringSort, span!()).unwrap();
        res.add_sort(BoolSort, span!()).unwrap();
        res.add_sort(I64Sort, span!()).unwrap();
        res.add_sort(F64Sort, span!()).unwrap();
        res.add_sort(BigIntSort, span!()).unwrap();
        res.add_sort(BigRatSort, span!()).unwrap();

        // res.add_presort::<MapSort>(span!()).unwrap();
        // res.add_presort::<SetSort>(span!()).unwrap();
        // res.add_presort::<VecSort>(span!()).unwrap();
        // res.add_presort::<FunctionSort>(span!()).unwrap();
        // res.add_presort::<MultiSetSort>(span!()).unwrap();

        res
    }
}

impl TypeInfo {
    pub fn add_sort<S: Sort + 'static>(&mut self, sort: S, span: Span) -> Result<(), TypeError> {
        self.add_arcsort(Arc::new(sort), span)
    }

    /// Adds a sort constructor to the typechecker's known set of types.
    pub fn add_presort<S: Presort>(&mut self, span: Span) -> Result<(), TypeError> {
        let name = S::presort_name();
        match self.presorts.entry(name) {
            HEntry::Occupied(_) => Err(TypeError::SortAlreadyBound(name, span)),
            HEntry::Vacant(e) => {
                e.insert(S::make_sort);
                self.reserved_primitives.extend(S::reserved_primitives());
                Ok(())
            }
        }
    }

    pub fn add_arcsort(&mut self, sort: ArcSort, span: Span) -> Result<(), TypeError> {
        let name = sort.name();

        match self.sorts.entry(name) {
            HEntry::Occupied(_) => Err(TypeError::SortAlreadyBound(name, span)),
            HEntry::Vacant(e) => {
                e.insert(sort.clone());
                sort.register_primitives(self);
                Ok(())
            }
        }
    }

    pub fn get_sort_by<S: Sort + Send + Sync>(
        &self,
        pred: impl Fn(&Arc<S>) -> bool,
    ) -> Option<Arc<S>> {
        for sort in self.sorts.values() {
            let sort = sort.clone().as_arc_any();
            if let Ok(sort) = Arc::downcast(sort) {
                if pred(&sort) {
                    return Some(sort);
                }
            }
        }
        None
    }

    pub fn get_sort_nofail<S: Sort + Send + Sync>(&self) -> Arc<S> {
        match self.get_sort_by(|_| true) {
            Some(sort) => sort,
            None => panic!("Failed to lookup sort: {}", std::any::type_name::<S>()),
        }
    }

    pub fn add_primitive(&mut self, prim: impl Into<Primitive>) {
        let prim = prim.into();
        self.primitives.entry(prim.name()).or_default().push(prim);
    }

    pub(crate) fn typecheck_program(
        &mut self,
        symbol_gen: &mut SymbolGen,
        program: &Vec<NCommand>,
    ) -> Result<Vec<ResolvedNCommand>, TypeError> {
        let mut result = vec![];
        for command in program {
            result.push(self.typecheck_command(symbol_gen, command)?);
        }

        Ok(result)
    }

    pub(crate) fn function_to_functype(&self, func: &FunctionDecl) -> Result<FuncType, TypeError> {
        let input = func
            .schema
            .input
            .iter()
            .map(|name| {
                if let Some(sort) = self.sorts.get(name) {
                    Ok(sort.clone())
                } else {
                    Err(TypeError::UndefinedSort(*name, func.span.clone()))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output = if let Some(sort) = self.sorts.get(&func.schema.output) {
            Ok(sort.clone())
        } else {
            Err(TypeError::UndefinedSort(
                func.schema.output,
                func.span.clone(),
            ))
        }?;

        Ok(FuncType {
            name: func.name,
            subtype: func.subtype,
            input,
            output: output.clone(),
        })
    }

    fn typecheck_command(
        &mut self,
        symbol_gen: &mut SymbolGen,
        command: &NCommand,
    ) -> Result<ResolvedNCommand, TypeError> {
        let command: ResolvedNCommand = match command {
            NCommand::Function(fdecl) => {
                ResolvedNCommand::Function(self.typecheck_function(symbol_gen, fdecl)?)
            }
            NCommand::NormRule {
                rule,
                ruleset,
                name,
            } => ResolvedNCommand::NormRule {
                rule: self.typecheck_rule(symbol_gen, rule)?,
                ruleset: *ruleset,
                name: *name,
            },
            NCommand::Sort(span, sort, presort_and_args) => {
                // Note this is bad since typechecking should be pure and idempotent
                // Otherwise typechecking the same program twice will fail
                self.declare_sort(*sort, presort_and_args, span.clone())?;
                ResolvedNCommand::Sort(span.clone(), *sort, presort_and_args.clone())
            }
            NCommand::CoreAction(Action::Let(span, var, expr)) => {
                let expr = self.typecheck_expr(symbol_gen, expr, &Default::default())?;
                let output_type = expr.output_type();
                self.global_types.insert(*var, output_type.clone());
                let var = ResolvedVar {
                    name: *var,
                    sort: output_type,
                    // not a global reference, but a global binding
                    is_global_ref: false,
                };
                ResolvedNCommand::CoreAction(ResolvedAction::Let(span.clone(), var, expr))
            }
            NCommand::CoreAction(action) => ResolvedNCommand::CoreAction(self.typecheck_action(
                symbol_gen,
                action,
                &Default::default(),
            )?),
            NCommand::Check(span, facts) => {
                ResolvedNCommand::Check(span.clone(), self.typecheck_facts(symbol_gen, facts)?)
            }
            NCommand::Fail(span, cmd) => ResolvedNCommand::Fail(
                span.clone(),
                Box::new(self.typecheck_command(symbol_gen, cmd)?),
            ),
            NCommand::RunSchedule(schedule) => {
                ResolvedNCommand::RunSchedule(self.typecheck_schedule(symbol_gen, schedule)?)
            }
            NCommand::Pop(span, n) => ResolvedNCommand::Pop(span.clone(), *n),
            NCommand::Push(n) => ResolvedNCommand::Push(*n),
            NCommand::SetOption { name, value } => {
                let value = self.typecheck_expr(symbol_gen, value, &Default::default())?;
                ResolvedNCommand::SetOption { name: *name, value }
            }
            NCommand::AddRuleset(span, ruleset) => {
                ResolvedNCommand::AddRuleset(span.clone(), *ruleset)
            }
            NCommand::UnstableCombinedRuleset(span, name, sub_rulesets) => {
                ResolvedNCommand::UnstableCombinedRuleset(span.clone(), *name, sub_rulesets.clone())
            }
            NCommand::PrintOverallStatistics => ResolvedNCommand::PrintOverallStatistics,
            NCommand::PrintTable(span, table, size) => {
                ResolvedNCommand::PrintTable(span.clone(), *table, *size)
            }
            NCommand::PrintSize(span, n) => {
                // Should probably also resolve the function symbol here
                ResolvedNCommand::PrintSize(span.clone(), *n)
            }
            NCommand::Output { span, file, exprs } => {
                let exprs = exprs
                    .iter()
                    .map(|expr| self.typecheck_expr(symbol_gen, expr, &Default::default()))
                    .collect::<Result<Vec<_>, _>>()?;
                ResolvedNCommand::Output {
                    span: span.clone(),
                    file: file.clone(),
                    exprs,
                }
            }
            NCommand::Input { span, name, file } => ResolvedNCommand::Input {
                span: span.clone(),
                name: *name,
                file: file.clone(),
            },
        };
        Ok(command)
    }

    fn typecheck_function(
        &mut self,
        symbol_gen: &mut SymbolGen,
        fdecl: &FunctionDecl,
    ) -> Result<ResolvedFunctionDecl, TypeError> {
        if self.sorts.contains_key(&fdecl.name) {
            return Err(TypeError::SortAlreadyBound(fdecl.name, fdecl.span.clone()));
        }
        if self.is_primitive(fdecl.name) {
            return Err(TypeError::PrimitiveAlreadyBound(
                fdecl.name,
                fdecl.span.clone(),
            ));
        }
        let ftype = self.function_to_functype(fdecl)?;
        if self.func_types.insert(fdecl.name, ftype).is_some() {
            return Err(TypeError::FunctionAlreadyBound(
                fdecl.name,
                fdecl.span.clone(),
            ));
        }
        let mut bound_vars = IndexMap::default();
        let output_type = self.sorts.get(&fdecl.schema.output).unwrap();
        if fdecl.subtype == FunctionSubtype::Constructor && !output_type.is_eq_sort() {
            return Err(TypeError::ConstructorOutputNotSort(
                fdecl.name,
                fdecl.span.clone(),
            ));
        }
        bound_vars.insert("old".into(), (fdecl.span.clone(), output_type.clone()));
        bound_vars.insert("new".into(), (fdecl.span.clone(), output_type.clone()));

        Ok(ResolvedFunctionDecl {
            name: fdecl.name,
            subtype: fdecl.subtype,
            schema: fdecl.schema.clone(),
            merge: match &fdecl.merge {
                Some(merge) => Some(self.typecheck_expr(symbol_gen, merge, &bound_vars)?),
                None => None,
            },
            cost: fdecl.cost,
            unextractable: fdecl.unextractable,
            ignore_viz: fdecl.ignore_viz,
            span: fdecl.span.clone(),
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
                        ruleset: *ruleset,
                        until,
                    },
                )
            }
        };

        Result::Ok(schedule)
    }

    pub fn declare_sort(
        &mut self,
        name: impl Into<Symbol>,
        presort_and_args: &Option<(Symbol, Vec<Expr>)>,
        span: Span,
    ) -> Result<(), TypeError> {
        let name = name.into();
        if self.func_types.contains_key(&name) {
            return Err(TypeError::FunctionAlreadyBound(name, span));
        }

        let sort = match presort_and_args {
            Some((presort, args)) => {
                if let Some(mksort) = self.presorts.get(presort) {
                    mksort(self, name, args)?
                } else {
                    return Err(TypeError::PresortNotFound(*presort, span));
                }
            }
            None => Arc::new(EqSort { name }),
        };
        self.add_arcsort(sort, span)
    }

    fn typecheck_rule(
        &self,
        symbol_gen: &mut SymbolGen,
        rule: &Rule,
    ) -> Result<ResolvedRule, TypeError> {
        let Rule { span, head, body } = rule;
        let mut constraints = vec![];

        let (query, mapped_query) = Facts(body.clone()).to_query(self, symbol_gen);
        constraints.extend(query.get_constraints(self)?);

        let mut binding = query.get_vars();
        let (actions, mapped_action) = head.to_core_actions(self, &mut binding, symbol_gen)?;

        let mut problem = Problem::default();
        problem.add_rule(
            &CoreRule {
                span: span.clone(),
                body: query,
                head: actions,
            },
            self,
            symbol_gen,
        )?;

        let assignment = problem
            .solve(|sort: &ArcSort| sort.name())
            .map_err(|e| e.to_type_error())?;

        let body: Vec<ResolvedFact> = assignment.annotate_facts(&mapped_query, self);
        let actions: ResolvedActions = assignment.annotate_actions(&mapped_action, self)?;

        Self::check_lookup_actions(&actions)?;

        Ok(ResolvedRule {
            span: span.clone(),
            body,
            head: actions,
        })
    }

    fn check_lookup_expr(expr: &GenericExpr<ResolvedCall, ResolvedVar>) -> Result<(), TypeError> {
        match expr {
            GenericExpr::Call(span, head, args) => {
                match head {
                    ResolvedCall::Func(t) => {
                        // Only allowed to lookup constructor or relation
                        if t.subtype != FunctionSubtype::Constructor
                            && t.subtype != FunctionSubtype::Relation
                        {
                            Err(TypeError::LookupInRuleDisallowed(
                                head.to_symbol(),
                                span.clone(),
                            ))
                        } else {
                            Ok(())
                        }
                    }
                    ResolvedCall::Primitive(_) => Ok(()),
                }?;
                for arg in args.iter() {
                    Self::check_lookup_expr(arg)?
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn check_lookup_actions(actions: &ResolvedActions) -> Result<(), TypeError> {
        for action in actions.iter() {
            match action {
                GenericAction::Let(_, _, rhs) => Self::check_lookup_expr(rhs),
                GenericAction::Set(_, _, args, rhs) => {
                    for arg in args.iter() {
                        Self::check_lookup_expr(arg)?
                    }
                    Self::check_lookup_expr(rhs)
                }
                GenericAction::Union(_, lhs, rhs) => {
                    Self::check_lookup_expr(lhs)?;
                    Self::check_lookup_expr(rhs)
                }
                GenericAction::Change(_, _, _, args) => {
                    for arg in args.iter() {
                        Self::check_lookup_expr(arg)?
                    }
                    Ok(())
                }
                GenericAction::Extract(_, expr, variants) => {
                    Self::check_lookup_expr(expr)?;
                    Self::check_lookup_expr(variants)
                }
                GenericAction::Panic(..) => Ok(()),
                GenericAction::Expr(_, expr) => Self::check_lookup_expr(expr),
            }?
        }
        Ok(())
    }

    fn typecheck_facts(
        &self,
        symbol_gen: &mut SymbolGen,
        facts: &[Fact],
    ) -> Result<Vec<ResolvedFact>, TypeError> {
        let (query, mapped_facts) = Facts(facts.to_vec()).to_query(self, symbol_gen);
        let mut problem = Problem::default();
        problem.add_query(&query, self)?;
        let assignment = problem
            .solve(|sort: &ArcSort| sort.name())
            .map_err(|e| e.to_type_error())?;
        let annotated_facts = assignment.annotate_facts(&mapped_facts, self);
        Ok(annotated_facts)
    }

    fn typecheck_actions(
        &self,
        symbol_gen: &mut SymbolGen,
        actions: &Actions,
        binding: &IndexMap<Symbol, (Span, ArcSort)>,
    ) -> Result<ResolvedActions, TypeError> {
        let mut binding_set = binding.keys().cloned().collect::<IndexSet<_>>();
        let (actions, mapped_action) =
            actions.to_core_actions(self, &mut binding_set, symbol_gen)?;
        let mut problem = Problem::default();

        // add actions to problem
        problem.add_actions(&actions, self, symbol_gen)?;

        // add bindings from the context
        for (var, (span, sort)) in binding {
            problem.assign_local_var_type(*var, span.clone(), sort.clone())?;
        }

        let assignment = problem
            .solve(|sort: &ArcSort| sort.name())
            .map_err(|e| e.to_type_error())?;

        let annotated_actions = assignment.annotate_actions(&mapped_action, self)?;
        Ok(annotated_actions)
    }

    fn typecheck_expr(
        &self,
        symbol_gen: &mut SymbolGen,
        expr: &Expr,
        binding: &IndexMap<Symbol, (Span, ArcSort)>,
    ) -> Result<ResolvedExpr, TypeError> {
        let action = Action::Expr(expr.span(), expr.clone());
        let typechecked_action = self.typecheck_action(symbol_gen, &action, binding)?;
        match typechecked_action {
            ResolvedAction::Expr(_, expr) => Ok(expr),
            _ => unreachable!(),
        }
    }

    fn typecheck_action(
        &self,
        symbol_gen: &mut SymbolGen,
        action: &Action,
        binding: &IndexMap<Symbol, (Span, ArcSort)>,
    ) -> Result<ResolvedAction, TypeError> {
        self.typecheck_actions(symbol_gen, &Actions::singleton(action.clone()), binding)
            .map(|mut v| {
                assert_eq!(v.len(), 1);
                v.0.pop().unwrap()
            })
    }

    pub fn lookup_global(&self, sym: &Symbol) -> Option<ArcSort> {
        self.global_types.get(sym).cloned()
    }

    pub(crate) fn is_primitive(&self, sym: Symbol) -> bool {
        self.primitives.contains_key(&sym) || self.reserved_primitives.contains(&sym)
    }

    pub(crate) fn lookup_user_func(&self, sym: Symbol) -> Option<FuncType> {
        self.func_types.get(&sym).cloned()
    }

    pub(crate) fn is_global(&self, sym: Symbol) -> bool {
        self.global_types.contains_key(&sym)
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
    Unbound(Symbol, Span),
    #[error("{1}\nUndefined sort {0}")]
    UndefinedSort(Symbol, Span),
    #[error("{2}\nSort {0} definition is disallowed: {1}")]
    DisallowedSort(Symbol, String, Span),
    #[error("{1}\nUnbound function {0}")]
    UnboundFunction(Symbol, Span),
    #[error("{1}\nFunction already bound {0}")]
    FunctionAlreadyBound(Symbol, Span),
    #[error("{1}\nSort {0} already declared.")]
    SortAlreadyBound(Symbol, Span),
    #[error("{1}\nPrimitive {0} already declared.")]
    PrimitiveAlreadyBound(Symbol, Span),
    #[error("Function type mismatch: expected {} => {}, actual {} => {}", .1.iter().map(|s| s.name().to_string()).collect::<Vec<_>>().join(", "), .0.name(), .3.iter().map(|s| s.name().to_string()).collect::<Vec<_>>().join(", "), .2.name())]
    FunctionTypeMismatch(ArcSort, Vec<ArcSort>, ArcSort, Vec<ArcSort>),
    #[error("{1}\nPresort {0} not found.")]
    PresortNotFound(Symbol, Span),
    #[error("{}\nFailed to infer a type for: {0}", .0.span())]
    InferenceFailure(Expr),
    #[error("{1}\nVariable {0} was already defined")]
    AlreadyDefined(Symbol, Span),
    #[error("{1}\nThe output type of constructor function {0} must be sort")]
    ConstructorOutputNotSort(Symbol, Span),
    #[error("{1}\nValue lookup of non-constructor function {0} in rule is disallowed.")]
    LookupInRuleDisallowed(Symbol, Span),
    #[error("All alternative definitions considered failed\n{}", .0.iter().map(|e| format!("  {e}\n")).collect::<Vec<_>>().join(""))]
    AllAlternativeFailed(Vec<TypeError>),
    #[error("{1}\nCannot union values of sort {}", .0.name())]
    NonEqsortUnion(ArcSort, Span),
}

#[cfg(test)]
mod test {
    use crate::{typechecking::TypeError, EGraph, Error};

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
            _ => panic!("Expected arity mismatch, got: {:?}", res),
        }
    }
}
