use crate::{core::CoreRule, *};
use ast::Rule;

#[derive(Clone, Debug)]
pub struct FuncType {
    pub name: Symbol,
    pub input: Vec<ArcSort>,
    pub output: ArcSort,
    pub is_datatype: bool,
    pub has_default: bool,
}

/// Stores resolved typechecking information.
/// TODO make these not public, use accessor methods
#[derive(Clone)]
pub struct TypeInfo {
    // get the sort from the sorts name()
    pub presorts: HashMap<Symbol, PreSort>,
    // TODO(yz): I want to get rid of this as now we have user-defined primitives and constraint based type checking
    pub presort_names: HashSet<Symbol>,
    pub sorts: HashMap<Symbol, Arc<dyn Sort>>,
    pub primitives: HashMap<Symbol, Vec<Primitive>>,
    pub func_types: HashMap<Symbol, FuncType>,
    pub global_types: HashMap<Symbol, ArcSort>,
}

impl Default for TypeInfo {
    fn default() -> Self {
        let mut res = Self {
            presorts: Default::default(),
            presort_names: Default::default(),
            sorts: Default::default(),
            primitives: Default::default(),
            func_types: Default::default(),
            global_types: Default::default(),
        };

        res.add_sort(UnitSort::new(UNIT_SYM.into()));
        res.add_sort(StringSort::new("String".into()));
        res.add_sort(BoolSort::new("bool".into()));
        res.add_sort(I64Sort::new("i64".into()));
        res.add_sort(F64Sort::new("f64".into()));
        res.add_sort(RationalSort::new("Rational".into()));

        res.presort_names.extend(MapSort::presort_names());
        res.presort_names.extend(SetSort::presort_names());
        res.presort_names.extend(VecSort::presort_names());
        res.presort_names.extend(FunctionSort::presort_names());

        res.presorts.insert("Map".into(), MapSort::make_sort);
        res.presorts.insert("Set".into(), SetSort::make_sort);
        res.presorts.insert("Vec".into(), VecSort::make_sort);
        res.presorts
            .insert("UnstableFn".into(), FunctionSort::make_sort);

        res.add_primitive(ValueEq {
            unit: res.get_sort_nofail(),
        });

        res
    }
}

pub const UNIT_SYM: &str = "Unit";

impl TypeInfo {
    pub(crate) fn infer_literal(&self, lit: &Literal) -> ArcSort {
        match lit {
            Literal::Int(_) => self.sorts.get(&Symbol::from("i64")),
            Literal::F64(_) => self.sorts.get(&Symbol::from("f64")),
            Literal::String(_) => self.sorts.get(&Symbol::from("String")),
            Literal::Bool(_) => self.sorts.get(&Symbol::from("bool")),
            Literal::Unit => self.sorts.get(&Symbol::from("Unit")),
        }
        .unwrap()
        .clone()
    }

    pub fn add_sort<S: Sort + 'static>(&mut self, sort: S) {
        self.add_arcsort(Arc::new(sort)).unwrap()
    }

    pub fn add_arcsort(&mut self, sort: ArcSort) -> Result<(), TypeError> {
        let name = sort.name();

        match self.sorts.entry(name) {
            Entry::Occupied(_) => Err(TypeError::SortAlreadyBound(name)),
            Entry::Vacant(e) => {
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
        program: &Vec<NCommand>,
    ) -> Result<Vec<ResolvedNCommand>, TypeError> {
        let mut result = vec![];
        for command in program {
            result.push(self.typecheck_command(command)?);
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
                    Err(TypeError::Unbound(*name))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output = if let Some(sort) = self.sorts.get(&func.schema.output) {
            Ok(sort.clone())
        } else {
            Err(TypeError::Unbound(func.schema.output))
        }?;

        Ok(FuncType {
            name: func.name,
            input,
            output: output.clone(),
            is_datatype: output.is_eq_sort() && func.merge.is_none() && func.default.is_none(),
            has_default: func.default.is_some(),
        })
    }

    fn typecheck_command(&mut self, command: &NCommand) -> Result<ResolvedNCommand, TypeError> {
        let command: ResolvedNCommand = match command {
            NCommand::Function(fdecl) => {
                ResolvedNCommand::Function(self.typecheck_function(fdecl)?)
            }
            NCommand::NormRule {
                rule,
                ruleset,
                name,
            } => ResolvedNCommand::NormRule {
                rule: self.typecheck_rule(rule)?,
                ruleset: *ruleset,
                name: *name,
            },
            NCommand::Sort(sort, presort_and_args) => {
                // Note this is bad since typechecking should be pure and idempotent
                // Otherwise typechecking the same program twice will fail
                self.declare_sort(*sort, presort_and_args)?;
                ResolvedNCommand::Sort(*sort, presort_and_args.clone())
            }
            NCommand::CoreAction(Action::Let(ann, var, expr)) => {
                let expr = self.typecheck_expr(expr, &Default::default())?;
                let output_type = expr.output_type(self);
                self.global_types.insert(*var, output_type.clone());
                let var = ResolvedVar {
                    name: *var,
                    sort: output_type,
                    // not a global reference, but a global binding
                    is_global_ref: false,
                };
                ResolvedNCommand::CoreAction(ResolvedAction::Let(*ann, var, expr))
            }
            NCommand::CoreAction(action) => {
                ResolvedNCommand::CoreAction(self.typecheck_action(action, &Default::default())?)
            }
            NCommand::Check(span, facts) => {
                ResolvedNCommand::Check(*span, self.typecheck_facts(facts)?)
            }
            NCommand::Fail(cmd) => ResolvedNCommand::Fail(Box::new(self.typecheck_command(cmd)?)),
            NCommand::RunSchedule(schedule) => {
                ResolvedNCommand::RunSchedule(self.typecheck_schedule(schedule)?)
            }
            NCommand::Pop(n) => ResolvedNCommand::Pop(*n),
            NCommand::Push(n) => ResolvedNCommand::Push(*n),
            NCommand::SetOption { name, value } => {
                let value = self.typecheck_expr(value, &Default::default())?;
                ResolvedNCommand::SetOption { name: *name, value }
            }
            NCommand::AddRuleset(ruleset) => ResolvedNCommand::AddRuleset(*ruleset),
            NCommand::UnstableCombinedRuleset(name, sub_rulesets) => {
                ResolvedNCommand::UnstableCombinedRuleset(*name, sub_rulesets.clone())
            }
            NCommand::PrintOverallStatistics => ResolvedNCommand::PrintOverallStatistics,
            NCommand::CheckProof => ResolvedNCommand::CheckProof,
            NCommand::PrintTable(table, size) => ResolvedNCommand::PrintTable(*table, *size),
            NCommand::PrintSize(n) => {
                // Should probably also resolve the function symbol here
                ResolvedNCommand::PrintSize(*n)
            }
            NCommand::Output { file, exprs } => {
                let exprs = exprs
                    .iter()
                    .map(|expr| self.typecheck_expr(expr, &Default::default()))
                    .collect::<Result<Vec<_>, _>>()?;
                ResolvedNCommand::Output {
                    file: file.clone(),
                    exprs,
                }
            }
            NCommand::Input { name, file } => ResolvedNCommand::Input {
                name: *name,
                file: file.clone(),
            },
        };
        Ok(command)
    }

    fn typecheck_function(
        &mut self,
        fdecl: &FunctionDecl,
    ) -> Result<ResolvedFunctionDecl, TypeError> {
        if self.sorts.contains_key(&fdecl.name) {
            return Err(TypeError::SortAlreadyBound(fdecl.name));
        }
        if self.is_primitive(fdecl.name) {
            return Err(TypeError::PrimitiveAlreadyBound(fdecl.name));
        }
        let ftype = self.function_to_functype(fdecl)?;
        if self.func_types.insert(fdecl.name, ftype).is_some() {
            return Err(TypeError::FunctionAlreadyBound(fdecl.name));
        }
        let mut bound_vars = IndexMap::default();
        let output_type = self.sorts.get(&fdecl.schema.output).unwrap();
        bound_vars.insert("old".into(), (*DUMMY_SPAN, output_type.clone()));
        bound_vars.insert("new".into(), (*DUMMY_SPAN, output_type.clone()));

        Ok(ResolvedFunctionDecl {
            name: fdecl.name,
            schema: fdecl.schema.clone(),
            merge: match &fdecl.merge {
                Some(merge) => Some(self.typecheck_expr(merge, &bound_vars)?),
                None => None,
            },
            default: fdecl
                .default
                .as_ref()
                .map(|default| self.typecheck_expr(default, &Default::default()))
                .transpose()?,
            merge_action: self.typecheck_actions(&fdecl.merge_action, &bound_vars)?,
            cost: fdecl.cost,
            unextractable: fdecl.unextractable,
            ignore_viz: fdecl.ignore_viz,
        })
    }

    fn typecheck_schedule(&self, schedule: &Schedule) -> Result<ResolvedSchedule, TypeError> {
        let schedule = match schedule {
            Schedule::Repeat(span, times, schedule) => ResolvedSchedule::Repeat(
                *span,
                *times,
                Box::new(self.typecheck_schedule(schedule)?),
            ),
            Schedule::Sequence(span, schedules) => {
                let schedules = schedules
                    .iter()
                    .map(|schedule| self.typecheck_schedule(schedule))
                    .collect::<Result<Vec<_>, _>>()?;
                ResolvedSchedule::Sequence(*span, schedules)
            }
            Schedule::Saturate(span, schedule) => {
                ResolvedSchedule::Saturate(*span, Box::new(self.typecheck_schedule(schedule)?))
            }
            Schedule::Run(span, RunConfig { ruleset, until }) => {
                let until = until
                    .as_ref()
                    .map(|facts| self.typecheck_facts(facts))
                    .transpose()?;
                ResolvedSchedule::Run(
                    *span,
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
    ) -> Result<(), TypeError> {
        let name = name.into();
        if self.func_types.contains_key(&name) {
            return Err(TypeError::FunctionAlreadyBound(name));
        }

        let sort = match presort_and_args {
            Some((presort, args)) => {
                let mksort = self
                    .presorts
                    .get(presort)
                    .ok_or(TypeError::PresortNotFound(*presort))?;
                mksort(self, name, args)?
            }
            None => Arc::new(EqSort { name }),
        };
        self.add_arcsort(sort)
    }

    fn typecheck_rule(&self, rule: &Rule) -> Result<ResolvedRule, TypeError> {
        let Rule { ann, head, body } = rule;
        let mut constraints = vec![];

        let mut fresh_gen = SymbolGen::new("$".to_string());
        let (query, mapped_query) = Facts(body.clone()).to_query(self, &mut fresh_gen);
        constraints.extend(query.get_constraints(self)?);

        let mut binding = query.get_vars();
        let (actions, mapped_action) = head.to_core_actions(self, &mut binding, &mut fresh_gen)?;

        let mut problem = Problem::default();
        problem.add_rule(
            &CoreRule {
                ann: *ann,
                body: query,
                head: actions,
            },
            self,
        )?;

        let assignment = problem
            .solve(|sort: &ArcSort| sort.name())
            .map_err(|e| e.to_type_error())?;

        let body: Vec<ResolvedFact> = assignment.annotate_facts(&mapped_query, self);
        let actions: ResolvedActions = assignment.annotate_actions(&mapped_action, self)?;

        Ok(ResolvedRule {
            ann: *ann,
            body,
            head: actions,
        })
    }

    fn typecheck_facts(&self, facts: &[Fact]) -> Result<Vec<ResolvedFact>, TypeError> {
        let mut fresh_gen = SymbolGen::new("$".to_string());
        let (query, mapped_facts) = Facts(facts.to_vec()).to_query(self, &mut fresh_gen);
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
        actions: &Actions,
        binding: &IndexMap<Symbol, (Span, ArcSort)>,
    ) -> Result<ResolvedActions, TypeError> {
        let mut binding_set = binding.keys().cloned().collect::<IndexSet<_>>();
        let mut fresh_gen = SymbolGen::new("$".to_string());
        let (actions, mapped_action) =
            actions.to_core_actions(self, &mut binding_set, &mut fresh_gen)?;
        let mut problem = Problem::default();

        // add actions to problem
        problem.add_actions(&actions, self)?;

        // add bindings from the context
        for (var, (span, sort)) in binding {
            problem.assign_local_var_type(*var, *span, sort.clone())?;
        }

        let assignment = problem
            .solve(|sort: &ArcSort| sort.name())
            .map_err(|e| e.to_type_error())?;

        let annotated_actions = assignment.annotate_actions(&mapped_action, self)?;
        Ok(annotated_actions)
    }

    fn typecheck_expr(
        &self,
        expr: &Expr,
        binding: &IndexMap<Symbol, (Span, ArcSort)>,
    ) -> Result<ResolvedExpr, TypeError> {
        let action = Action::Expr(*DUMMY_SPAN, expr.clone());
        let typechecked_action = self.typecheck_action(&action, binding)?;
        match typechecked_action {
            ResolvedAction::Expr(_, expr) => Ok(expr),
            _ => unreachable!(),
        }
    }

    fn typecheck_action(
        &self,
        action: &Action,
        binding: &IndexMap<Symbol, (Span, ArcSort)>,
    ) -> Result<ResolvedAction, TypeError> {
        self.typecheck_actions(&Actions::singleton(action.clone()), binding)
            .map(|mut v| {
                assert_eq!(v.len(), 1);
                v.0.pop().unwrap()
            })
    }

    pub fn lookup_global(&self, sym: &Symbol) -> Option<ArcSort> {
        self.global_types.get(sym).cloned()
    }

    pub(crate) fn is_primitive(&self, sym: Symbol) -> bool {
        self.primitives.contains_key(&sym) || self.presort_names.contains(&sym)
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
    #[error("Arity mismatch, expected {expected} args: {expr}")]
    Arity { expr: Expr, expected: usize },
    #[error(
        "Type mismatch: expr = {expr}, expected = {}, actual = {}, reason: {reason}",
        .expected.name(), .actual.name(),
    )]
    Mismatch {
        expr: Expr,
        expected: ArcSort,
        actual: ArcSort,
        reason: String,
    },
    #[error("Tried to unify too many literals: {}", ListDisplay(.0, "\n"))]
    TooManyLiterals(Vec<Literal>),
    #[error("Unbound symbol {0}")]
    Unbound(Symbol),
    #[error("Undefined sort {0}")]
    UndefinedSort(Symbol),
    #[error("Unbound function {0}")]
    UnboundFunction(Symbol),
    #[error("Function already bound {0}")]
    FunctionAlreadyBound(Symbol),
    #[error("Function declarations are not allowed after a push.")]
    FunctionAfterPush(Symbol),
    #[error("Cannot set the datatype {} to a value. Did you mean to use union?", .0.name)]
    SetDatatype(FuncType),
    #[error("Sort declarations are not allowed after a push.")]
    SortAfterPush(Symbol),
    #[error("Global already bound {0}")]
    GlobalAlreadyBound(Symbol),
    #[error("Local already bound {0} with type {}. Got: {}", .1.name(), .2.name())]
    LocalAlreadyBound(Symbol, ArcSort, ArcSort),
    #[error("Sort {0} already declared.")]
    SortAlreadyBound(Symbol),
    #[error("Primitive {0} already declared.")]
    PrimitiveAlreadyBound(Symbol),
    #[error("Type mismatch: expected {}, actual {}", .0.name(), .1.name())]
    TypeMismatch(ArcSort, ArcSort),
    #[error("Presort {0} not found.")]
    PresortNotFound(Symbol),
    #[error("Cannot type a variable as unit: {0}")]
    UnitVar(Symbol),
    #[error("Failed to infer a type for: {0}")]
    InferenceFailure(Expr),
    #[error("No matching primitive for: ({op} {})", ListDisplay(.inputs, " "))]
    NoMatchingPrimitive { op: Symbol, inputs: Vec<Symbol> },
    #[error("Variable {0} was already defined")]
    AlreadyDefined(Symbol),
    #[error("All alternative definitions considered failed\n{}", .0.iter().map(|e| format!("  {e}\n")).collect::<Vec<_>>().join(""))]
    AllAlternativeFailed(Vec<TypeError>),
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
                let span = e.ann();
                assert_eq!(&prog[span.1..span.2], "(f a b c)");
            }
            _ => panic!("Expected arity mismatch, got: {:?}", res),
        }
    }
}
