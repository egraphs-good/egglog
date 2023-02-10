use crate::*;

#[derive(Clone, Debug)]
pub struct FuncType {
    pub input: Vec<ArcSort>,
    pub output: ArcSort,
}

impl FuncType {
    pub fn new(input: Vec<ArcSort>, output: ArcSort) -> Self {
        Self { input, output }
    }
}

#[derive(Default, Clone)]
pub struct TypeInfo {
    // get the sort from the sorts name()
    pub presorts: HashMap<Symbol, PreSort>,
    pub sorts: HashMap<Symbol, Arc<dyn Sort>>,
    pub primitives: HashMap<Symbol, Vec<Primitive>>,
    pub func_types: HashMap<Symbol, FuncType>,
    pub global_types: HashMap<Symbol, ArcSort>,
    pub local_types: HashMap<CommandId, HashMap<Symbol, ArcSort>>,
}


pub const UNIT_SYM: &str = "Unit";

impl TypeInfo {
    pub fn new() -> Self {
        let mut res = Self::default();
        res.add_sort(UnitSort::new(UNIT_SYM.into()));
        res.add_sort(StringSort::new("String".into()));
        res.add_sort(I64Sort::new("i64".into()));
        res.add_sort(F64Sort::new("f64".into()));
        res.add_sort(RationalSort::new("Rational".into()));
        res.presorts.insert("Map".into(), MapSort::make_sort);
        res
    }

    pub(crate) fn infer_literal(&self, lit: &Literal) -> ArcSort {
        match lit {
            Literal::Int(_) => self.sorts.get(&Symbol::from("i64")),
            Literal::F64(_) => self.sorts.get(&Symbol::from("f64")),
            Literal::String(_) => self.sorts.get(&Symbol::from("String")),
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

    pub fn get_sort<S: Sort + Send + Sync>(&self) -> Arc<S> {
        for sort in self.sorts.values() {
            let sort = sort.clone().as_arc_any();
            if let Ok(sort) = Arc::downcast(sort) {
                return sort;
            }
        }
        
        // TODO handle if multiple match?
        // could handle by type id??
        panic!("Failed to lookup sort: {}", std::any::type_name::<S>());
    }

    pub fn add_primitive(&mut self, prim: impl Into<Primitive>) {
        let prim = prim.into();
        self.primitives.entry(prim.name()).or_default().push(prim);
    }

    pub(crate) fn typecheck_program(
        program: &Vec<NormCommand>,
    ) -> Result<Self, TypeError> {
        let mut type_info = TypeInfo::default();
        for command in program {
            type_info.typecheck_command(command)?;
        }
        Ok(type_info)
    }

    pub(crate) fn schema_to_functype(&self, schema: &Schema) -> Result<FuncType, TypeError> {
        let input = schema
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
        let output = if let Some(sort) = self.sorts.get(&schema.output) {
            Ok(sort.clone())
        } else {
            Err(TypeError::Unbound(schema.output))
        }?;
        Ok(FuncType::new(input, output))
    }

    pub(crate) fn typecheck_command(
        &mut self,
        command: &NormCommand,
    ) -> Result<(), TypeError> {
        match &command.command {
            NCommand::Function(fdecl) => {
                if self.sorts.contains_key(&fdecl.name) {
                    return Err(TypeError::SortAlreadyBound(fdecl.name));
                }
                let ftype = self.schema_to_functype(&fdecl.schema)?;
                if self
                    .func_types
                    .insert(fdecl.name, ftype)
                    .is_some()
                {
                    return Err(TypeError::FunctionAlreadyBound(fdecl.name));
                }
            }
            NCommand::Declare(name, parent) => {
                if let Some(parent_type) = self.sorts.get(parent) {
                    if self
                        .global_types
                        .insert(*name, parent_type.clone())
                        .is_some()
                    {
                        return Err(TypeError::GlobalAlreadyBound(*name));
                    }
                } else {
                    return Err(TypeError::Unbound(*parent));
                }
            }
            NCommand::NormRule(_ruleset, rule) => {
                self.typecheck_rule(command.metadata.id, rule)?;
            }
            NCommand::Sort(sort, presort_and_args) => {
                self.declare_sort(*sort, presort_and_args)?;
            }
            _ => (),
        }
        Ok(())
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

    fn typecheck_rule(
        &mut self,
        ctx: CommandId,
        rule: &NormRule,
    ) -> Result<(), TypeError> {
        assert!(self.local_types.insert(ctx, Default::default()).is_none());
        self.typecheck_facts(ctx, &rule.body)?;
        self.typecheck_actions(ctx, &rule.head)?;
        Ok(())
    }

    fn typecheck_facts(
        &mut self,
        ctx: CommandId,
        facts: &Vec<NormFact>,
    ) -> Result<(), TypeError> {
        for fact in facts {
            self.typecheck_fact(ctx, fact)?;
        }
        Ok(())
    }

    fn typecheck_actions(
        &mut self,
        ctx: CommandId,
        actions: &Vec<NormAction>,
    ) -> Result<(), TypeError> {
        for action in actions {
            self.typecheck_action(ctx, action)?;
        }
        Ok(())
    }

    fn typecheck_action(
        &mut self,
        ctx: CommandId,
        action: &NormAction,
    ) -> Result<(), TypeError> {
        match action {
            NormAction::Let(var, expr) => {
                let expr_type = self.typecheck_expr(ctx, expr)?;
                if let Some(existing) = self
                    .local_types
                    .get_mut(&ctx)
                    .unwrap()
                    .insert(*var, expr_type)
                {
                    return Err(TypeError::LocalAlreadyBound(*var, existing));
                }
            }
            NormAction::LetLit(var, lit) => {
                let lit_type = self.infer_literal(lit);
                if let Some(existing) = self
                    .local_types
                    .get_mut(&ctx)
                    .unwrap()
                    .insert(*var, lit_type)
                {
                    return Err(TypeError::LocalAlreadyBound(*var, existing));
                }
            }
            NormAction::Delete(head, body) => {
                let child_types = body
                    .iter()
                    .map(|var| self.lookup(ctx, *var))
                    .collect::<Result<Vec<_>, _>>()?;
                let func_type = self.lookup_func(ctx, *head, child_types)?;
            }
            NormAction::Set(head, body, other) => {
                let child_types = body
                    .iter()
                    .map(|var| self.lookup(ctx, *var))
                    .collect::<Result<Vec<_>, _>>()?;
                let func_type = self.lookup_func(ctx, *head, child_types)?;
                let other_type = self.lookup(ctx, *other)?;
                if func_type.name() != other_type.name() {
                    return Err(TypeError::TypeMismatch(func_type, other_type));
                }
            }
            NormAction::Union(var1, var2) => {
                let var1_type = self.lookup(ctx, *var1)?;
                let var2_type = self.lookup(ctx, *var2)?;
                if var1_type.name() != var2_type.name() {
                    return Err(TypeError::TypeMismatch(var1_type, var2_type));
                }
            }
            NormAction::LetVar(var1, var2) => {
                let var2_type = self.lookup(ctx, *var2)?;
                if let Some(existing) = self
                    .local_types
                    .get_mut(&ctx)
                    .unwrap()
                    .insert(*var1, var2_type)
                {
                    return Err(TypeError::LocalAlreadyBound(*var1, existing));
                }
            }
            NormAction::Panic(..) => (),
        }
        Ok(())
    }

    fn typecheck_fact(
        &mut self,
        ctx: CommandId,
        fact: &NormFact,
    ) -> Result<(), TypeError> {
        match fact {
            NormFact::Assign(var, expr) | NormFact::Compute(var, expr) => {
                let expr_type = self.typecheck_expr(ctx, expr)?;
                if let Some(existing) = self
                    .local_types
                    .get_mut(&ctx)
                    .unwrap()
                    .insert(*var, expr_type)
                {
                    return Err(TypeError::LocalAlreadyBound(*var, existing));
                }
            }
            NormFact::AssignLit(var, lit) => {
                let lit_type = self.infer_literal(lit);
                if let Some(existing) = self
                    .local_types
                    .get_mut(&ctx)
                    .unwrap()
                    .insert(*var, lit_type)
                {
                    return Err(TypeError::LocalAlreadyBound(*var, existing));
                }
            }
            NormFact::ConstrainEq(var1, var2) => {
                let l1 = self.lookup(ctx, *var1);
                let l2 = self.lookup(ctx, *var2);
                if let Ok(v1type) = l1 {
                    if let Ok(v2type) = l2 {
                        if v1type.name() != v2type.name() {
                            return Err(TypeError::TypeMismatch(v1type, v2type));
                        }
                    } else {
                        self.local_types
                            .get_mut(&ctx)
                            .unwrap()
                            .insert(*var2, v1type);
                    }
                } else if let Ok(v2type) = l2 {
                    self.local_types
                        .get_mut(&ctx)
                        .unwrap()
                        .insert(*var1, v2type);
                } else {
                    return Err(TypeError::Unbound(*var1));
                }
            }
        }
        Ok(())
    }

    fn lookup(&self, ctx: CommandId, sym: Symbol) -> Result<ArcSort, TypeError> {
        self.global_types
            .get(&sym)
            .map(|x| Ok(x.clone()))
            .unwrap_or_else(|| {
                if let Some(found) = self.local_types.get(&ctx).unwrap().get(&sym) {
                    Ok(found.clone())
                } else {
                    Err(TypeError::Unbound(sym))
                }
            })
    }

    fn lookup_func(
        &self,
        ctx: CommandId,
        sym: Symbol,
        input_types: Vec<ArcSort>,
    ) -> Result<ArcSort, TypeError> {
        if let Some(found) = self.func_types.get(&sym) {
            Ok(found.output.clone())
        } else {
            for prim in self.primitives.get(&sym).unwrap() {
                if let Some(return_type) = prim.accept(&input_types) {
                    return Ok(return_type);
                }
            }
            Err(TypeError::NoMatchingPrimitive {
                op: sym,
                inputs: input_types.iter().map(|s| s.name()).collect(),
            })
        }
    }

    fn typecheck_expr(
        &self,
        ctx: CommandId,
        expr: &NormExpr,
    ) -> Result<ArcSort, TypeError> {
        match expr {
            NormExpr::Call(head, body) => {
                let child_types = body
                    .iter()
                    .map(|var| self.lookup(ctx, *var))
                    .collect::<Result<Vec<_>, _>>()?;
                self.lookup_func(ctx, *head, child_types)
            }
        }
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
    #[error("Function already bound {0}")]
    FunctionAlreadyBound(Symbol),
    #[error("Global already bound {0}")]
    GlobalAlreadyBound(Symbol),
    #[error("Local already bound {0} with type {}", .1.name())]
    LocalAlreadyBound(Symbol, ArcSort),
    #[error("Sort {0} already declared.")]
    SortAlreadyBound(Symbol),
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
}
