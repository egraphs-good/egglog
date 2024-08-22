use crate::{
    ast::{
        GenericAction, GenericActions, GenericExpr, GenericFact, MappedAction, ResolvedAction,
        ResolvedActions, ResolvedExpr, ResolvedFact, ResolvedVar,
    },
    core::{
        Atom, AtomTerm, CoreAction, CoreRule, GenericCoreActions, Query, ResolvedCall, SymbolOrEq,
    },
    sort::I64Sort,
    typechecking::TypeError,
    util::{FreshGen, HashSet, SymbolGen},
    ArcSort, CorrespondingVar, Span, Symbol, TypeInfo, DUMMY_SPAN,
};
use core::{hash::Hash, panic};
// Use immutable hashmap for performance
// cloning assignments is common and O(1) with immutable hashmap
use im_rc::HashMap;
use std::{fmt::Debug, iter::once, mem::swap};

#[derive(Clone, Debug)]
pub enum ImpossibleConstraint {
    ArityMismatch {
        atom: Atom<Symbol>,
        expected: usize,
        actual: usize,
    },
    FunctionMismatch {
        expected_output: ArcSort,
        expected_input: Vec<ArcSort>,
        actual_output: ArcSort,
        actual_input: Vec<ArcSort>,
    },
}

#[derive(Debug)]
pub enum Constraint<Var, Value> {
    Eq(Var, Var),
    Assign(Var, Value),
    And(Vec<Constraint<Var, Value>>),
    // Exactly one of the constraints holds
    // and all others are false
    Xor(Vec<Constraint<Var, Value>>),
    Impossible(ImpossibleConstraint),
}

pub enum ConstraintError<Var, Value> {
    InconsistentConstraint(Var, Value, Value),
    UnconstrainedVar(Var),
    NoConstraintSatisfied(Vec<ConstraintError<Var, Value>>),
    ImpossibleCaseIdentified(ImpossibleConstraint),
}

impl ConstraintError<AtomTerm, ArcSort> {
    pub fn to_type_error(&self) -> TypeError {
        match &self {
            ConstraintError::InconsistentConstraint(x, v1, v2) => TypeError::Mismatch {
                expr: x.to_expr(),
                expected: v1.clone(),
                actual: v2.clone(),
            },
            ConstraintError::UnconstrainedVar(v) => TypeError::InferenceFailure(v.to_expr()),
            ConstraintError::NoConstraintSatisfied(constraints) => TypeError::AllAlternativeFailed(
                constraints.iter().map(|c| c.to_type_error()).collect(),
            ),
            ConstraintError::ImpossibleCaseIdentified(ImpossibleConstraint::ArityMismatch {
                atom,
                expected,
                actual,
            }) => {
                assert_eq!(*actual, atom.args.len() - 1);
                TypeError::Arity {
                    expr: atom.to_expr(),
                    expected: *expected,
                }
            }
            ConstraintError::ImpossibleCaseIdentified(ImpossibleConstraint::FunctionMismatch {
                expected_output,
                expected_input,
                actual_output,
                actual_input,
            }) => TypeError::FunctionTypeMismatch(
                expected_output.clone(),
                expected_input.clone(),
                actual_output.clone(),
                actual_input.clone(),
            ),
        }
    }
}

impl<Var, Value> Constraint<Var, Value>
where
    Var: Eq + PartialEq + Hash + Clone + Debug,
    Value: Clone + Debug,
{
    /// Takes a partial assignment and update it based on the constraint.
    /// If there's a conflict, returns the conflicting variable, the assigned conflicting types.
    /// Otherwise, return whether the assignment is updated.
    fn update<K: Eq>(
        &self,
        assignment: &mut Assignment<Var, Value>,
        key: impl Fn(&Value) -> K + Copy,
    ) -> Result<bool, ConstraintError<Var, Value>> {
        match self {
            Constraint::Eq(x, y) => match (assignment.0.get(x), assignment.0.get(y)) {
                (Some(value), None) => {
                    assignment.insert(y.clone(), value.clone());
                    Ok(true)
                }
                (None, Some(value)) => {
                    assignment.insert(x.clone(), value.clone());
                    Ok(true)
                }
                (Some(v1), Some(v2)) => {
                    if key(v1) == key(v2) {
                        Ok(false)
                    } else {
                        Err(ConstraintError::InconsistentConstraint(
                            x.clone(),
                            v1.clone(),
                            v2.clone(),
                        ))
                    }
                }
                (None, None) => Ok(false),
            },
            Constraint::Assign(x, v) => match assignment.0.get(x) {
                None => {
                    assignment.insert(x.clone(), v.clone());
                    Ok(true)
                }
                Some(value) => {
                    if key(value) == key(v) {
                        Ok(false)
                    } else {
                        Err(ConstraintError::InconsistentConstraint(
                            x.clone(),
                            v.clone(),
                            value.clone(),
                        ))
                    }
                }
            },
            Constraint::Xor(cs) => {
                let mut success_count = 0;
                let orig_assignment = assignment.clone();
                let mut result_assignment = assignment.clone();
                let mut assignment_updated = false;
                let mut errors = vec![];
                for c in cs {
                    let result = c.update(assignment, key);
                    match result {
                        Ok(updated) => {
                            success_count += 1;
                            if success_count > 1 {
                                break;
                            }

                            if updated {
                                swap(&mut result_assignment, assignment);
                            }
                            assignment_updated = updated;
                        }
                        Err(error) => errors.push(error),
                    }
                }
                // If update is successful for only one sub constraint, then we have nailed down the only true constraint.
                // If update is successful for more than one constraint, then Xor succeeds with no updates.
                // If update fails for every constraint, then Xor fails
                match success_count.cmp(&1) {
                    std::cmp::Ordering::Equal => {
                        *assignment = result_assignment;
                        Ok(assignment_updated)
                    }
                    std::cmp::Ordering::Greater => {
                        *assignment = orig_assignment;
                        Ok(false)
                    }
                    std::cmp::Ordering::Less => Err(ConstraintError::NoConstraintSatisfied(errors)),
                }
            }
            Constraint::Impossible(constraint) => Err(ConstraintError::ImpossibleCaseIdentified(
                constraint.clone(),
            )),
            Constraint::And(cs) => {
                let orig_assignment = assignment.clone();
                let mut updated = false;
                for c in cs {
                    match c.update(assignment, key) {
                        Ok(upd) => updated |= upd,
                        Err(error) => {
                            // In the case of failure,
                            // we need to restore the assignment
                            *assignment = orig_assignment;
                            return Err(error);
                        }
                    }
                }
                Ok(updated)
            }
        }
    }
}

#[derive(Debug)]
pub struct Problem<Var, Value> {
    pub constraints: Vec<Constraint<Var, Value>>,
    pub range: HashSet<Var>,
}

impl Default for Problem<AtomTerm, ArcSort> {
    fn default() -> Self {
        Self {
            constraints: vec![],
            range: HashSet::default(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct Assignment<Var, Value>(pub HashMap<Var, Value>);

impl<Var, Value> Assignment<Var, Value>
where
    Var: Hash + Eq + PartialEq + Clone,
    Value: Clone,
{
    pub fn insert(&mut self, var: Var, value: Value) -> Option<Value> {
        self.0.insert(var, value)
    }

    pub fn get(&self, var: &Var) -> Option<&Value> {
        self.0.get(var)
    }
}

impl Assignment<AtomTerm, ArcSort> {
    pub(crate) fn annotate_expr(
        &self,
        expr: &GenericExpr<CorrespondingVar<Symbol, Symbol>, Symbol>,
        typeinfo: &TypeInfo,
    ) -> ResolvedExpr {
        match &expr {
            GenericExpr::Lit(span, literal) => ResolvedExpr::Lit(span.clone(), literal.clone()),
            GenericExpr::Var(span, var) => {
                let global_ty = typeinfo.lookup_global(var);
                let ty = global_ty
                    .clone()
                    // Span is ignored when looking up atom_terms
                    .or_else(|| self.get(&AtomTerm::Var(DUMMY_SPAN.clone(), *var)).cloned())
                    .expect("All variables should be assigned before annotation");
                ResolvedExpr::Var(
                    span.clone(),
                    ResolvedVar {
                        name: *var,
                        sort: ty.clone(),
                        is_global_ref: global_ty.is_some(),
                    },
                )
            }
            GenericExpr::Call(
                span,
                CorrespondingVar {
                    head,
                    to: corresponding_var,
                },
                args,
            ) => {
                // get the resolved call using resolve_rule
                let args: Vec<_> = args
                    .iter()
                    .map(|arg| self.annotate_expr(arg, typeinfo))
                    .collect();
                let types: Vec<_> = args
                    .iter()
                    .map(|arg| arg.output_type(typeinfo))
                    .chain(once(
                        self.get(&AtomTerm::Var(DUMMY_SPAN.clone(), *corresponding_var))
                            .unwrap()
                            .clone(),
                    ))
                    .collect();
                let resolved_call = ResolvedCall::from_resolution(head, &types, typeinfo);
                GenericExpr::Call(span.clone(), resolved_call, args)
            }
        }
    }

    pub(crate) fn annotate_fact(
        &self,
        facts: &GenericFact<CorrespondingVar<Symbol, Symbol>, Symbol>,
        typeinfo: &TypeInfo,
    ) -> ResolvedFact {
        match facts {
            GenericFact::Eq(span, facts) => ResolvedFact::Eq(
                span.clone(),
                facts
                    .iter()
                    .map(|expr| self.annotate_expr(expr, typeinfo))
                    .collect(),
            ),
            GenericFact::Fact(expr) => ResolvedFact::Fact(self.annotate_expr(expr, typeinfo)),
        }
    }

    pub(crate) fn annotate_facts(
        &self,
        mapped_facts: &[GenericFact<CorrespondingVar<Symbol, Symbol>, Symbol>],
        typeinfo: &TypeInfo,
    ) -> Vec<ResolvedFact> {
        mapped_facts
            .iter()
            .map(|fact| self.annotate_fact(fact, typeinfo))
            .collect()
    }

    pub(crate) fn annotate_action(
        &self,
        action: &MappedAction,
        typeinfo: &TypeInfo,
    ) -> Result<ResolvedAction, TypeError> {
        match action {
            GenericAction::Let(span, var, expr) => {
                let ty = self
                    .get(&AtomTerm::Var(span.clone(), *var))
                    .expect("All variables should be assigned before annotation");
                Ok(ResolvedAction::Let(
                    span.clone(),
                    ResolvedVar {
                        name: *var,
                        sort: ty.clone(),
                        is_global_ref: false,
                    },
                    self.annotate_expr(expr, typeinfo),
                ))
            }
            // Note mapped_var for set is a dummy variable that does not mean anything
            GenericAction::Set(
                span,
                CorrespondingVar {
                    head,
                    to: _mapped_var,
                },
                children,
                rhs,
            ) => {
                let children: Vec<_> = children
                    .iter()
                    .map(|child| self.annotate_expr(child, typeinfo))
                    .collect();
                let rhs = self.annotate_expr(rhs, typeinfo);
                let types: Vec<_> = children
                    .iter()
                    .map(|child| child.output_type(typeinfo))
                    .chain(once(rhs.output_type(typeinfo)))
                    .collect();
                let resolved_call = ResolvedCall::from_resolution(head, &types, typeinfo);
                if !matches!(resolved_call, ResolvedCall::Func(_)) {
                    return Err(TypeError::UnboundFunction(*head, span.clone()));
                }
                Ok(ResolvedAction::Set(
                    span.clone(),
                    resolved_call,
                    children,
                    rhs,
                ))
            }
            // Note mapped_var for delete is a dummy variable that does not mean anything
            GenericAction::Change(
                span,
                change,
                CorrespondingVar {
                    head,
                    to: _mapped_var,
                },
                children,
            ) => {
                let children: Vec<_> = children
                    .iter()
                    .map(|child| self.annotate_expr(child, typeinfo))
                    .collect();
                let types: Vec<_> = children
                    .iter()
                    .map(|child| child.output_type(typeinfo))
                    .collect();
                let resolved_call =
                    ResolvedCall::from_resolution_func_types(head, &types, typeinfo)
                        .ok_or_else(|| TypeError::UnboundFunction(*head, span.clone()))?;
                Ok(ResolvedAction::Change(
                    span.clone(),
                    *change,
                    resolved_call,
                    children.clone(),
                ))
            }
            GenericAction::Union(span, lhs, rhs) => Ok(ResolvedAction::Union(
                span.clone(),
                self.annotate_expr(lhs, typeinfo),
                self.annotate_expr(rhs, typeinfo),
            )),
            GenericAction::Extract(span, lhs, rhs) => Ok(ResolvedAction::Extract(
                span.clone(),
                self.annotate_expr(lhs, typeinfo),
                self.annotate_expr(rhs, typeinfo),
            )),
            GenericAction::Panic(span, msg) => Ok(ResolvedAction::Panic(span.clone(), msg.clone())),
            GenericAction::Expr(span, expr) => Ok(ResolvedAction::Expr(
                span.clone(),
                self.annotate_expr(expr, typeinfo),
            )),
        }
    }

    pub(crate) fn annotate_actions(
        &self,
        mapped_actions: &GenericActions<CorrespondingVar<Symbol, Symbol>, Symbol>,
        typeinfo: &TypeInfo,
    ) -> Result<ResolvedActions, TypeError> {
        let actions = mapped_actions
            .iter()
            .map(|action| self.annotate_action(action, typeinfo))
            .collect::<Result<_, _>>()?;

        Ok(ResolvedActions::new(actions))
    }
}

impl<Var, Value> Problem<Var, Value>
where
    Var: Eq + PartialEq + Hash + Clone + Debug,
    Value: Clone + Debug,
{
    pub(crate) fn solve<K: Eq + Debug>(
        &self,
        key: impl Fn(&Value) -> K + Copy,
    ) -> Result<Assignment<Var, Value>, ConstraintError<Var, Value>> {
        let mut assignment = Assignment(HashMap::default());
        let mut changed = true;
        while changed {
            changed = false;
            for constraint in self.constraints.iter() {
                changed |= constraint.update(&mut assignment, key)?;
            }
        }

        for v in self.range.iter() {
            if !assignment.0.contains_key(v) {
                return Err(ConstraintError::UnconstrainedVar(v.clone()));
            }
        }
        Ok(assignment)
    }

    pub(crate) fn add_binding(&mut self, var: Var, clone: Value) {
        self.constraints.push(Constraint::Assign(var, clone));
    }
}

impl Problem<AtomTerm, ArcSort> {
    pub(crate) fn add_query(
        &mut self,
        query: &Query<SymbolOrEq, Symbol>,
        typeinfo: &TypeInfo,
    ) -> Result<(), TypeError> {
        self.constraints.extend(query.get_constraints(typeinfo)?);
        self.range.extend(query.atom_terms());
        Ok(())
    }

    pub fn add_actions(
        &mut self,
        actions: &GenericCoreActions<Symbol, Symbol>,
        typeinfo: &TypeInfo,
    ) -> Result<(), TypeError> {
        let mut symbol_gen = SymbolGen::new("$".to_string());
        for action in actions.0.iter() {
            self.constraints
                .extend(action.get_constraints(typeinfo, &mut symbol_gen)?);

            // bound vars are added to range
            match action {
                CoreAction::Let(span, var, _, _) => {
                    self.range.insert(AtomTerm::Var(span.clone(), *var));
                }
                CoreAction::LetAtomTerm(span, v, _) => {
                    self.range.insert(AtomTerm::Var(span.clone(), *v));
                }
                _ => (),
            }
        }
        Ok(())
    }

    pub(crate) fn add_rule(
        &mut self,
        rule: &CoreRule,
        typeinfo: &TypeInfo,
    ) -> Result<(), TypeError> {
        let CoreRule {
            span: _,
            head,
            body,
        } = rule;
        self.add_query(body, typeinfo)?;
        self.add_actions(head, typeinfo)?;
        Ok(())
    }

    pub(crate) fn assign_local_var_type(
        &mut self,
        var: Symbol,
        span: Span,
        sort: ArcSort,
    ) -> Result<(), TypeError> {
        self.add_binding(AtomTerm::Var(span.clone(), var), sort);
        self.range.insert(AtomTerm::Var(span, var));
        Ok(())
    }
}

impl CoreAction {
    pub(crate) fn get_constraints(
        &self,
        typeinfo: &TypeInfo,
        symbol_gen: &mut SymbolGen,
    ) -> Result<Vec<Constraint<AtomTerm, ArcSort>>, TypeError> {
        match self {
            CoreAction::Let(span, symbol, f, args) => {
                let mut args = args.clone();
                args.push(AtomTerm::Var(span.clone(), *symbol));

                Ok(get_literal_and_global_constraints(&args, typeinfo)
                    .chain(get_atom_application_constraints(f, &args, span, typeinfo)?)
                    .collect())
            }
            CoreAction::Set(span, head, args, rhs) => {
                let mut args = args.clone();
                args.push(rhs.clone());

                Ok(get_literal_and_global_constraints(&args, typeinfo)
                    .chain(get_atom_application_constraints(
                        head, &args, span, typeinfo,
                    )?)
                    .collect())
            }
            CoreAction::Change(span, _change, head, args) => {
                let mut args = args.clone();
                // Add a dummy last output argument
                let var = symbol_gen.fresh(head);
                args.push(AtomTerm::Var(span.clone(), var));

                Ok(get_literal_and_global_constraints(&args, typeinfo)
                    .chain(get_atom_application_constraints(
                        head, &args, span, typeinfo,
                    )?)
                    .collect())
            }
            CoreAction::Union(_ann, lhs, rhs) => Ok(get_literal_and_global_constraints(
                &[lhs.clone(), rhs.clone()],
                typeinfo,
            )
            .chain(once(Constraint::Eq(lhs.clone(), rhs.clone())))
            .collect()),
            CoreAction::Extract(_ann, e, n) => {
                // e can be anything
                Ok(
                    get_literal_and_global_constraints(&[e.clone(), n.clone()], typeinfo)
                        .chain(once(Constraint::Assign(
                            n.clone(),
                            typeinfo.get_sort_nofail::<I64Sort>() as ArcSort,
                        )))
                        .collect(),
                )
            }
            CoreAction::Panic(_ann, _) => Ok(vec![]),
            CoreAction::LetAtomTerm(span, v, at) => {
                Ok(get_literal_and_global_constraints(&[at.clone()], typeinfo)
                    .chain(once(Constraint::Eq(
                        AtomTerm::Var(span.clone(), *v),
                        at.clone(),
                    )))
                    .collect())
            }
        }
    }
}

impl Atom<SymbolOrEq> {
    pub fn get_constraints(
        &self,
        type_info: &TypeInfo,
    ) -> Result<Vec<Constraint<AtomTerm, ArcSort>>, TypeError> {
        let literal_constraints = get_literal_and_global_constraints(&self.args, type_info);
        match &self.head {
            SymbolOrEq::Eq => {
                assert_eq!(self.args.len(), 2);
                let constraints = literal_constraints
                    .chain(once(Constraint::Eq(
                        self.args[0].clone(),
                        self.args[1].clone(),
                    )))
                    .collect();
                Ok(constraints)
            }
            SymbolOrEq::Symbol(head) => Ok(literal_constraints
                .chain(get_atom_application_constraints(
                    head, &self.args, &self.span, type_info,
                )?)
                .collect()),
        }
    }
}

fn get_atom_application_constraints(
    head: &Symbol,
    args: &[AtomTerm],
    span: &Span,
    type_info: &TypeInfo,
) -> Result<Vec<Constraint<AtomTerm, ArcSort>>, TypeError> {
    // An atom can have potentially different semantics due to polymorphism
    // e.g. (set-empty) can mean any empty set with some element type.
    // To handle this, we collect each possible instantiations of an atom
    // (where each instantiation is a vec of constraints, thus vec of vec)
    // into `xor_constraints`.
    // `Constraint::Xor` means one and only one of the instantiation can hold.
    let mut xor_constraints: Vec<Vec<Constraint<AtomTerm, ArcSort>>> = vec![];

    // function atom constraints
    if let Some(typ) = type_info.func_types.get(head) {
        let mut constraints = vec![];
        // arity mismatch
        if typ.input.len() + 1 != args.len() {
            constraints.push(Constraint::Impossible(
                ImpossibleConstraint::ArityMismatch {
                    atom: Atom {
                        span: span.clone(),
                        head: *head,
                        args: args.to_vec(),
                    },
                    expected: typ.input.len(),
                    actual: args.len() - 1,
                },
            ));
        } else {
            for (arg_typ, arg) in typ
                .input
                .iter()
                .cloned()
                .chain(once(typ.output.clone()))
                .zip(args.iter().cloned())
            {
                constraints.push(Constraint::Assign(arg, arg_typ));
            }
        }
        xor_constraints.push(constraints);
    }

    // primitive atom constraints
    if let Some(primitives) = type_info.primitives.get(head) {
        for p in primitives {
            let constraints = p.get_type_constraints(span).get(args, type_info);
            xor_constraints.push(constraints);
        }
    }

    // do literal and global variable constraints first
    // as they are the most "informative"
    match xor_constraints.len() {
        0 => Err(TypeError::UnboundFunction(*head, span.clone())),
        1 => Ok(xor_constraints.pop().unwrap()),
        _ => Ok(vec![Constraint::Xor(
            xor_constraints.into_iter().map(Constraint::And).collect(),
        )]),
    }
}

fn get_literal_and_global_constraints<'a>(
    args: &'a [AtomTerm],
    type_info: &'a TypeInfo,
) -> impl Iterator<Item = Constraint<AtomTerm, ArcSort>> + 'a {
    args.iter().filter_map(|arg| {
        match arg {
            AtomTerm::Var(_, _) => None,
            // Literal to type constraint
            AtomTerm::Literal(_, lit) => {
                let typ = type_info.infer_literal(lit);
                Some(Constraint::Assign(arg.clone(), typ.clone()))
            }
            AtomTerm::Global(_, v) => {
                if let Some(typ) = type_info.lookup_global(v) {
                    Some(Constraint::Assign(arg.clone(), typ.clone()))
                } else {
                    panic!("All global variables should be bound before type checking")
                }
            }
        }
    })
}

pub trait TypeConstraint {
    fn get(
        &self,
        arguments: &[AtomTerm],
        typeinfo: &TypeInfo,
    ) -> Vec<Constraint<AtomTerm, ArcSort>>;
}

/// Construct a set of `Assign` constraints that fully constrain the type of arguments
pub struct SimpleTypeConstraint {
    name: Symbol,
    sorts: Vec<ArcSort>,
    span: Span,
}

impl SimpleTypeConstraint {
    pub fn new(name: Symbol, sorts: Vec<ArcSort>, span: Span) -> SimpleTypeConstraint {
        SimpleTypeConstraint { name, sorts, span }
    }

    pub fn into_box(self) -> Box<dyn TypeConstraint> {
        Box::new(self)
    }
}

impl TypeConstraint for SimpleTypeConstraint {
    fn get(
        &self,
        arguments: &[AtomTerm],
        _typeinfo: &TypeInfo,
    ) -> Vec<Constraint<AtomTerm, ArcSort>> {
        if arguments.len() != self.sorts.len() {
            vec![Constraint::Impossible(
                ImpossibleConstraint::ArityMismatch {
                    atom: Atom {
                        span: self.span.clone(),
                        head: self.name,
                        args: arguments.to_vec(),
                    },
                    expected: self.sorts.len(),
                    actual: arguments.len(),
                },
            )]
        } else {
            arguments
                .iter()
                .cloned()
                .zip(self.sorts.iter().cloned())
                .map(|(arg, sort)| Constraint::Assign(arg, sort))
                .collect()
        }
    }
}

/// This constraint requires all types to be equivalent to each other
pub struct AllEqualTypeConstraint {
    name: Symbol,
    sort: Option<ArcSort>,
    exact_length: Option<usize>,
    output: Option<ArcSort>,
    span: Span,
}

impl AllEqualTypeConstraint {
    pub fn new(name: Symbol, span: Span) -> AllEqualTypeConstraint {
        AllEqualTypeConstraint {
            name,
            sort: None,
            exact_length: None,
            output: None,
            span,
        }
    }

    pub fn into_box(self) -> Box<dyn TypeConstraint> {
        Box::new(self)
    }

    /// Requires all arguments to have the given sort.
    /// If `with_output_sort` is not specified, this requirement
    /// also applies to the output argument.
    pub fn with_all_arguments_sort(mut self, sort: ArcSort) -> Self {
        self.sort = Some(sort);
        self
    }

    /// Requires the length of arguments to be `exact_length`.
    /// Note this includes both input arguments and output argument.
    pub fn with_exact_length(mut self, exact_length: usize) -> Self {
        self.exact_length = Some(exact_length);
        self
    }

    /// Requires the output argument to have the given sort.
    pub fn with_output_sort(mut self, output_sort: ArcSort) -> Self {
        self.output = Some(output_sort);
        self
    }
}

impl TypeConstraint for AllEqualTypeConstraint {
    fn get(
        &self,
        mut arguments: &[AtomTerm],
        _typeinfo: &TypeInfo,
    ) -> Vec<Constraint<AtomTerm, ArcSort>> {
        if arguments.is_empty() {
            panic!("all arguments should have length > 0")
        }

        match self.exact_length {
            Some(exact_length) if exact_length != arguments.len() => {
                return vec![Constraint::Impossible(
                    ImpossibleConstraint::ArityMismatch {
                        atom: Atom {
                            span: self.span.clone(),
                            head: self.name,
                            args: arguments.to_vec(),
                        },
                        expected: exact_length,
                        actual: arguments.len(),
                    },
                )]
            }
            _ => (),
        }

        let mut constraints = vec![];
        if let Some(output) = self.output.clone() {
            let (out, inputs) = arguments.split_last().unwrap();
            constraints.push(Constraint::Assign(out.clone(), output));
            arguments = inputs;
        }

        if let Some(sort) = self.sort.clone() {
            constraints.extend(
                arguments
                    .iter()
                    .cloned()
                    .map(|arg| Constraint::Assign(arg, sort.clone())),
            )
        } else if let Some((first, rest)) = arguments.split_first() {
            constraints.extend(
                rest.iter()
                    .cloned()
                    .map(|arg| Constraint::Eq(arg, first.clone())),
            );
        }
        constraints
    }
}
