use crate::{proofs::RULE_PROOF_KEYWORD, *};

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
        &mut self,
        program: &Vec<NormCommand>,
    ) -> Result<(), TypeError> {
        for command in program {
            self.typecheck_command(command)?;
        }
        self.check_no_sorts_after_push(program)?;

        Ok(())
    }

    fn check_no_sorts_after_push(&self, program: &Vec<NormCommand>) -> Result<(), TypeError> {
        let mut found_push = false;
        for command in program {
            match &command.command {
                NCommand::Push(_) => {
                    found_push = true;
                }
                NCommand::Function(fdecl) => {
                    if found_push {
                        return Err(TypeError::FunctionAfterPush(fdecl.name));
                    }
                }
                NCommand::Sort(name, _) => {
                    if found_push {
                        return Err(TypeError::SortAfterPush(*name));
                    }
                }
                _ => {}
            }
        }
        Ok(())
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

    fn typecheck_ncommand(&mut self, command: &NCommand, id: CommandId) -> Result<(), TypeError> {
        match command {
            NCommand::Function(fdecl) => {
                if self.sorts.contains_key(&fdecl.name) {
                    return Err(TypeError::SortAlreadyBound(fdecl.name));
                }
                if self.primitives.contains_key(&fdecl.name) {
                    return Err(TypeError::PrimitiveAlreadyBound(fdecl.name));
                }
                let ftype = self.schema_to_functype(&fdecl.schema)?;
                if self.func_types.insert(fdecl.name, ftype).is_some() {
                    return Err(TypeError::FunctionAlreadyBound(fdecl.name));
                }
            }
            NCommand::Declare(name, parent, _cost) => {
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
                self.typecheck_rule(id, rule)?;
            }
            NCommand::Sort(sort, presort_and_args) => {
                self.declare_sort(*sort, presort_and_args)?;
            }
            NCommand::NormAction(action) => {
                self.typecheck_action(id, action, true)?;
            }
            NCommand::Check(facts) => {
                self.typecheck_facts(id, facts)?;
            }
            NCommand::Fail(cmd) => {
                self.typecheck_ncommand(cmd, id)?;
            }

            // TODO cover all cases in typechecking
            _ => (),
        }
        Ok(())
    }

    pub(crate) fn typecheck_command(&mut self, command: &NormCommand) -> Result<(), TypeError> {
        assert!(self
            .local_types
            .insert(command.metadata.id, Default::default())
            .is_none());
        self.typecheck_ncommand(&command.command, command.metadata.id)
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

    fn typecheck_rule(&mut self, ctx: CommandId, rule: &NormRule) -> Result<(), TypeError> {
        // also check the validity of the ssa
        let mut bindings = self.verify_normal_form_facts(&rule.body);
        self.verify_normal_form_actions(&rule.head, &mut bindings);

        self.typecheck_facts(ctx, &rule.body)?;
        self.typecheck_actions(ctx, &rule.head)?;
        Ok(())
    }

    fn typecheck_facts(&mut self, ctx: CommandId, facts: &Vec<NormFact>) -> Result<(), TypeError> {
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
            self.typecheck_action(ctx, action, false)?;
        }
        Ok(())
    }

    fn verify_normal_form_facts(&self, facts: &Vec<NormFact>) -> HashSet<Symbol> {
        let mut let_bound: HashSet<Symbol> = Default::default();
        let mut bound_in_constraint = vec![];

        for fact in facts {
            match fact {
                NormFact::Assign(var, NormExpr::Call(_head, body)) => {
                    assert!(let_bound.insert(*var));
                    body.iter().for_each(|bvar| {
                        if !self.global_types.contains_key(bvar) {
                            assert!(let_bound.insert(*bvar));
                        }
                    });
                }
                NormFact::AssignLit(var, _lit) => {
                    assert!(let_bound.insert(*var));
                }
                NormFact::ConstrainEq(var1, var2) => {
                    if !let_bound.contains(var1)
                        && !let_bound.contains(var2)
                        && !self.global_types.contains_key(var1)
                        && !self.global_types.contains_key(var2)
                    {
                        panic!("ConstrainEq on unbound variables");
                    }
                    bound_in_constraint.push(*var1);
                    bound_in_constraint.push(*var2);
                }
            }
        }
        let_bound.extend(bound_in_constraint);
        let_bound
    }

    fn verify_normal_form_actions(
        &self,
        actions: &Vec<NormAction>,
        let_bound: &mut HashSet<Symbol>,
    ) {
        let assert_bound = |var, let_bound: &HashSet<Symbol>| {
            assert!(
                let_bound.contains(var)
                    || self.global_types.contains_key(var)
                    || self.reserved_type(*var).is_some()
            )
        };

        for action in actions {
            match action {
                NormAction::Let(var, NormExpr::Call(_head, body)) => {
                    assert!(let_bound.insert(*var));
                    body.iter().for_each(|bvar| {
                        assert_bound(bvar, let_bound);
                    });
                }
                NormAction::LetVar(v1, v2) => {
                    assert_bound(v2, let_bound);
                    assert!(let_bound.insert(*v1));
                }
                NormAction::LetLit(v1, _lit) => {
                    assert!(let_bound.insert(*v1));
                }
                NormAction::Delete(NormExpr::Call(_head, body)) => {
                    body.iter().for_each(|bvar| {
                        assert_bound(bvar, let_bound);
                    });
                }
                NormAction::Set(NormExpr::Call(_head, body), var) => {
                    body.iter().for_each(|bvar| {
                        assert_bound(bvar, let_bound);
                    });
                    assert_bound(var, let_bound);
                }
                NormAction::Union(v1, v2) => {
                    assert_bound(v1, let_bound);
                    assert_bound(v2, let_bound);
                }
                NormAction::Panic(..) => (),
            }
        }
    }

    fn introduce_binding(
        &mut self,
        ctx: CommandId,
        var: Symbol,
        sort: Arc<dyn Sort>,
        is_global: bool,
    ) -> Result<(), TypeError> {
        if is_global {
            if let Some(_existing) = self.global_types.insert(var, sort) {
                return Err(TypeError::GlobalAlreadyBound(var));
            }
        } else if let Some(existing) = self
            .local_types
            .get_mut(&ctx)
            .unwrap()
            .insert(var, sort.clone())
        {
            return Err(TypeError::LocalAlreadyBound(var, existing, sort));
        }

        Ok(())
    }

    fn typecheck_action(
        &mut self,
        ctx: CommandId,
        action: &NormAction,
        is_global: bool,
    ) -> Result<(), TypeError> {
        match action {
            NormAction::Let(var, expr) => {
                let expr_type = self.typecheck_expr(ctx, expr)?.output;

                self.introduce_binding(ctx, *var, expr_type, is_global)?;
            }
            NormAction::LetLit(var, lit) => {
                let lit_type = self.infer_literal(lit);
                self.introduce_binding(ctx, *var, lit_type, is_global)?;
            }
            NormAction::Delete(expr) => {
                self.typecheck_expr(ctx, expr)?;
            }
            NormAction::Set(expr, other) => {
                let func_type = self.typecheck_expr(ctx, expr)?.output;
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
                self.introduce_binding(ctx, *var1, var2_type, is_global)?;
            }
            NormAction::Panic(..) => (),
        }
        Ok(())
    }

    fn typecheck_fact(&mut self, ctx: CommandId, fact: &NormFact) -> Result<(), TypeError> {
        match fact {
            NormFact::Assign(var, expr) => {
                let expr_type = self.typecheck_expr(ctx, expr)?;
                if let Some(existing) = self
                    .local_types
                    .get_mut(&ctx)
                    .unwrap()
                    .insert(*var, expr_type.output.clone())
                {
                    if expr_type.output.name() != existing.name() {
                        return Err(TypeError::TypeMismatch(expr_type.output, existing));
                    }
                }
            }
            NormFact::AssignLit(var, lit) => {
                let lit_type = self.infer_literal(lit);
                if let Some(existing) = self
                    .local_types
                    .get_mut(&ctx)
                    .unwrap()
                    .insert(*var, lit_type.clone())
                {
                    if lit_type.name() != existing.name() {
                        return Err(TypeError::TypeMismatch(lit_type, existing));
                    }
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

    pub fn reserved_type(&self, sym: Symbol) -> Option<ArcSort> {
        if sym == RULE_PROOF_KEYWORD.into() {
            Some(self.sorts.get::<Symbol>(&"Proof__".into()).unwrap().clone())
        } else {
            None
        }
    }

    pub fn lookup(&self, ctx: CommandId, sym: Symbol) -> Result<ArcSort, TypeError> {
        // special logic for reserved keywords
        if let Some(t) = self.reserved_type(sym) {
            return Ok(t);
        }

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

    fn set_local_type(
        &mut self,
        ctx: CommandId,
        sym: Symbol,
        sym_type: ArcSort,
    ) -> Result<(), TypeError> {
        if let Some(existing) = self
            .local_types
            .get_mut(&ctx)
            .unwrap()
            .insert(sym, sym_type.clone())
        {
            if existing.name() != sym_type.name() {
                return Err(TypeError::LocalAlreadyBound(sym, existing, sym_type));
            }
        }
        Ok(())
    }

    pub(crate) fn is_primitive(&self, sym: Symbol) -> bool {
        self.primitives.contains_key(&sym)
    }

    fn lookup_func(
        &self,
        _ctx: CommandId,
        sym: Symbol,
        input_types: Vec<ArcSort>,
    ) -> Result<ArcSort, TypeError> {
        if let Some(found) = self.func_types.get(&sym) {
            Ok(found.output.clone())
        } else {
            if let Some(prims) = self.primitives.get(&sym) {
                for prim in prims {
                    if let Some(return_type) = prim.accept(&input_types) {
                        return Ok(return_type);
                    }
                }
            }

            Err(TypeError::NoMatchingPrimitive {
                op: sym,
                inputs: input_types.iter().map(|s| s.name()).collect(),
            })
        }
    }

    pub(crate) fn typecheck_expr(
        &mut self,
        ctx: CommandId,
        expr: &NormExpr,
    ) -> Result<FuncType, TypeError> {
        match expr {
            NormExpr::Call(head, body) => {
                let child_types = if let Some(found) = self.func_types.get(head) {
                    found.input.clone()
                } else {
                    body.iter()
                        .map(|var| self.lookup(ctx, *var))
                        .collect::<Result<Vec<_>, _>>()?
                };
                for (child_type, var) in child_types.iter().zip(body.iter()) {
                    self.set_local_type(ctx, *var, child_type.clone())?;
                }

                Ok(FuncType::new(
                    child_types.clone(),
                    self.lookup_func(ctx, *head, child_types)?,
                ))
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
    #[error("Function declarations are not allowed after a push.")]
    FunctionAfterPush(Symbol),
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
}
