use crate::{
    core::{
        Atom, CoreAction, CoreRule, GenericCoreActions, GenericCoreRule, HeadOrEq, Query,
        StringOrEq,
    },
    *,
};
use std::{cmp, rc::Rc};
// Use immutable hashmap for performance
// cloning assignments is common and O(1) with immutable hashmap
use im_rc::HashMap;
use std::{fmt::Debug, iter::once, mem::swap};

/// Represents constraints that are logically impossible to satisfy.
/// These are used to signal type errors during constraint solving.
#[derive(Clone, Debug)]
pub enum ImpossibleConstraint {
    ArityMismatch {
        atom: Atom<String>,
        // The expected arity for this atom
        expected: usize,
    },
    FunctionMismatch {
        expected_output: ArcSort,
        expected_input: Vec<ArcSort>,
        actual_output: ArcSort,
        actual_input: Vec<ArcSort>,
    },
}

/// A constraint that can be applied to variable assignments.
/// Constraints are used in type inference to represent relationships between variables and values.
pub trait Constraint<Var, Value>: dyn_clone::DynClone {
    /// Updates the assignment based on this constraint.
    /// Returns Ok(true) if the assignment was modified, Ok(false) if no changes were made,
    /// or Err if the constraint cannot be satisfied.
    ///
    /// `update` is allowed to modify the constraint itself, e.g. to convert a delayed constraint into an immediate one.
    /// The `key` function gets a string representation of the value for display.
    fn update(
        &mut self,
        assignment: &mut Assignment<Var, Value>,
        key: fn(&Value) -> &str,
    ) -> Result<bool, ConstraintError<Var, Value>>;

    /// Returns a human-readable string representation of this constraint.
    fn pretty(&self) -> String;
}

dyn_clone::clone_trait_object!(<Var, Value> Constraint<Var, Value>);

/// Creates an equality constraint between two variables.
/// If one of the variable has a known value, the constraint propagates value to the other variable.
/// If both variables have known but different values, the constraint fails.
pub fn eq<Var, Value>(x: Var, y: Var) -> Box<dyn Constraint<Var, Value>>
where
    Var: cmp::Eq + PartialEq + Hash + Clone + Debug + 'static,
    Value: Clone + Debug + 'static,
{
    Box::new(Eq(x, y))
}

/// Creates an assignment constraint that binds a variable to a specific value.
/// The constraint fails if the variable is already assigned to a different value.
pub fn assign<Var, Value>(x: Var, v: Value) -> Box<dyn Constraint<Var, Value>>
where
    Var: cmp::Eq + PartialEq + Hash + Clone + Debug + 'static,
    Value: Clone + Debug + 'static,
{
    Box::new(Assign(x, v))
}

/// Creates a conjunction constraint that requires all sub-constraints to be satisfied.
pub fn and<Var, Value>(cs: Vec<Box<dyn Constraint<Var, Value>>>) -> Box<dyn Constraint<Var, Value>>
where
    Var: cmp::Eq + PartialEq + Hash + Clone + Debug + 'static,
    Value: Clone + Debug + 'static,
{
    Box::new(And(cs))
}

/// Creates an exclusive-or constraint that requires exactly one sub-constraint to be satisfied.
/// The constraint proceeds if exactly one sub-constraint can be satisfied and all others lead to failure.
/// The constraint fails if zero sub-constraints can be satisfied.
pub fn xor<Var, Value>(cs: Vec<Box<dyn Constraint<Var, Value>>>) -> Box<dyn Constraint<Var, Value>>
where
    Var: cmp::Eq + PartialEq + Hash + Clone + Debug + 'static,
    Value: Clone + Debug + 'static,
{
    Box::new(Xor(cs))
}

/// Creates a constraint that always fails with the given impossible constraint.
/// This is used to signal type errors during constraint solving.
pub fn impossible<Var, Value>(constraint: ImpossibleConstraint) -> Box<dyn Constraint<Var, Value>>
where
    Var: cmp::Eq + PartialEq + Hash + Clone + Debug + 'static,
    Value: Clone + Debug + 'static,
{
    Box::new(Impossible { constraint })
}

/// Creates an implication constraint that activates when all watch variables are assigned.
/// The constraint function is called with the values of the watch variables to generate the actual constraint.
pub fn implies<Var, Value>(
    name: String,
    watch_vars: Vec<Var>,
    constraint: DelayedConstraintFn<Var, Value>,
) -> Box<dyn Constraint<Var, Value>>
where
    Var: cmp::Eq + PartialEq + Hash + Clone + Debug + 'static,
    Value: Clone + Debug + 'static,
{
    Box::new(Implies {
        name,
        watch_vars,
        constraint: DelayedConstraint::Delayed(constraint),
    })
}

pub type DelayedConstraintFn<Var, Value> = Rc<dyn Fn(&[&Value]) -> Box<dyn Constraint<Var, Value>>>;

#[derive(Clone)]
enum DelayedConstraint<Var, Value> {
    Delayed(DelayedConstraintFn<Var, Value>),
    Constraint(Box<dyn Constraint<Var, Value>>),
}

#[derive(Clone)]
struct Implies<Var, Value> {
    name: String,
    watch_vars: Vec<Var>,
    constraint: DelayedConstraint<Var, Value>,
}

impl<Var, Value> Constraint<Var, Value> for Implies<Var, Value>
where
    Var: cmp::Eq + PartialEq + Hash + Clone + Debug,
    Value: Clone + Debug,
{
    fn update(
        &mut self,
        assignment: &mut Assignment<Var, Value>,
        key: fn(&Value) -> &str,
    ) -> Result<bool, ConstraintError<Var, Value>> {
        let mut updated = false;
        // If the constraint is delayed, either make it immediate or return.
        if let DelayedConstraint::Delayed(delayed) = &self.constraint {
            let watch_vals: Option<Vec<&Value>> =
                self.watch_vars.iter().map(|v| assignment.get(v)).collect();
            let Some(watch_vals) = watch_vals else {
                return Ok(false);
            };
            let constraint = delayed(&watch_vals);
            self.constraint = DelayedConstraint::Constraint(constraint);
            updated = true;
        };

        // The constraint must be immediate now.
        let DelayedConstraint::Constraint(constraint) = &mut self.constraint else {
            unreachable!("update");
        };
        updated |= constraint.update(assignment, key)?;
        Ok(updated)
    }

    fn pretty(&self) -> String {
        let vars: String = self
            .watch_vars
            .iter()
            .map(|v| format!("{:?}", v))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{} => {}({})", vars, self.name, vars)
    }
}

#[derive(Clone)]
struct Eq<Var>(Var, Var);

impl<Var, Value> Constraint<Var, Value> for Eq<Var>
where
    Var: cmp::Eq + PartialEq + Hash + Clone + Debug,
    Value: Clone + Debug,
{
    fn update(
        &mut self,
        assignment: &mut Assignment<Var, Value>,
        key: fn(&Value) -> &str,
    ) -> Result<bool, ConstraintError<Var, Value>> {
        match (assignment.0.get(&self.0), assignment.0.get(&self.1)) {
            (Some(value), None) => {
                assignment.insert(self.1.clone(), value.clone());
                Ok(true)
            }
            (None, Some(value)) => {
                assignment.insert(self.0.clone(), value.clone());
                Ok(true)
            }
            (Some(v1), Some(v2)) => {
                if key(v1) == key(v2) {
                    Ok(false)
                } else {
                    Err(ConstraintError::InconsistentConstraint(
                        self.0.clone(),
                        v1.clone(),
                        v2.clone(),
                    ))
                }
            }
            (None, None) => Ok(false),
        }
    }

    fn pretty(&self) -> String {
        format!("{:?} = {:?}", self.0, self.1)
    }
}

#[derive(Clone)]
struct Assign<Var, Value>(Var, Value);

impl<Var, Value> Constraint<Var, Value> for Assign<Var, Value>
where
    Var: cmp::Eq + PartialEq + Hash + Clone + Debug,
    Value: Clone + Debug,
{
    fn update(
        &mut self,
        assignment: &mut Assignment<Var, Value>,
        key: fn(&Value) -> &str,
    ) -> Result<bool, ConstraintError<Var, Value>> {
        match assignment.0.get(&self.0) {
            None => {
                assignment.insert(self.0.clone(), self.1.clone());
                Ok(true)
            }
            Some(value) => {
                if key(value) == key(&self.1) {
                    Ok(false)
                } else {
                    Err(ConstraintError::InconsistentConstraint(
                        self.0.clone(),
                        self.1.clone(),
                        value.clone(),
                    ))
                }
            }
        }
    }

    fn pretty(&self) -> String {
        format!("{:?} = {:?}", self.0, self.1)
    }
}

#[derive(Clone)]
struct And<Var, Value>(Vec<Box<dyn Constraint<Var, Value>>>);

impl<Var, Value> Constraint<Var, Value> for And<Var, Value>
where
    Var: cmp::Eq + PartialEq + Hash + Clone + Debug,
    Value: Clone + Debug,
{
    fn update(
        &mut self,
        assignment: &mut Assignment<Var, Value>,
        key: fn(&Value) -> &str,
    ) -> Result<bool, ConstraintError<Var, Value>> {
        let orig_assignment = assignment.clone();
        let mut updated = false;
        for c in self.0.iter_mut() {
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

    fn pretty(&self) -> String {
        format!(
            "({})",
            self.0
                .iter()
                .map(|c| c.pretty())
                .collect::<Vec<_>>()
                .join(" /\\ ")
        )
    }
}

#[derive(Clone)]
struct Xor<Var, Value>(Vec<Box<dyn Constraint<Var, Value>>>);

impl<Var, Value> Constraint<Var, Value> for Xor<Var, Value>
where
    Var: cmp::Eq + PartialEq + Hash + Clone + Debug,
    Value: Clone + Debug,
{
    fn update(
        &mut self,
        assignment: &mut Assignment<Var, Value>,
        key: fn(&Value) -> &str,
    ) -> Result<bool, ConstraintError<Var, Value>> {
        let mut success_count = 0;
        let orig_assignment = assignment.clone();
        let orig_cs = self.0.clone();
        let mut result_assignment = assignment.clone();
        let mut assignment_updated = false;
        let mut errors = vec![];
        let mut result_constraint = None;

        let cs = std::mem::take(&mut self.0);
        for mut c in cs {
            let result = c.update(assignment, key);
            match result {
                Ok(updated) => {
                    success_count += 1;
                    if success_count > 1 {
                        break;
                    }

                    result_constraint = Some(c);
                    if updated {
                        swap(&mut result_assignment, assignment);
                    }
                    assignment_updated = updated;
                }
                Err(error) => errors.push(error),
            }
        }

        // Success roughly means "the constraint is compatible with the current assignment".
        //
        // If update is successful for only one sub constraint, then we have nailed down the only true constraint.
        // If update is successful for more than one constraint, then Xor succeeds with no updates.
        // If update fails for every constraint, then Xor fails
        match success_count.cmp(&1) {
            std::cmp::Ordering::Equal => {
                // Prune all other constraints. This is sound since the constraints are monotonic.
                self.0 = vec![result_constraint.unwrap()];
                *assignment = result_assignment;
                Ok(assignment_updated)
            }
            std::cmp::Ordering::Greater => {
                self.0 = orig_cs;
                *assignment = orig_assignment;
                Ok(false)
            }
            std::cmp::Ordering::Less => {
                self.0 = orig_cs;
                *assignment = orig_assignment;
                Err(ConstraintError::NoConstraintSatisfied(errors))
            }
        }
    }

    fn pretty(&self) -> String {
        format!(
            "({})",
            self.0
                .iter()
                .map(|c| c.pretty())
                .collect::<Vec<_>>()
                .join(" \\/ ")
        )
    }
}

#[derive(Clone)]
struct Impossible {
    constraint: ImpossibleConstraint,
}

impl<Var, Value> Constraint<Var, Value> for Impossible
where
    Var: cmp::Eq + PartialEq + Hash + Clone + Debug,
    Value: Clone + Debug,
{
    fn update(
        &mut self,
        _assignment: &mut Assignment<Var, Value>,
        _key: fn(&Value) -> &str,
    ) -> Result<bool, ConstraintError<Var, Value>> {
        Err(ConstraintError::ImpossibleCaseIdentified(
            self.constraint.clone(),
        ))
    }

    fn pretty(&self) -> String {
        format!("{:?}", self.constraint)
    }
}

/// Errors that can occur during constraint solving.
/// These represent various ways that constraint satisfaction can fail.
#[derive(Debug)]
pub enum ConstraintError<Var, Value> {
    /// A variable was assigned two different, incompatible values
    InconsistentConstraint(Var, Value, Value),
    /// A variable in the constraint range was not assigned any value
    UnconstrainedVar(Var),
    /// None of the alternative constraints in an XOR constraint could be satisfied
    NoConstraintSatisfied(Vec<ConstraintError<Var, Value>>),
    /// An impossible constraint was encountered during solving
    ImpossibleCaseIdentified(ImpossibleConstraint),
}

impl ConstraintError<AtomTerm, ArcSort> {
    /// Converts a [`ConstraintError`] produced by type checking into a type error.
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
            }) => TypeError::Arity {
                expr: atom.to_expr(),
                expected: *expected - 1,
            },
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

/// A constraint satisfaction problem consisting of constraints and a range of variables to solve for.
/// The problem is considered solved when *all* variables in the range are assigned.
pub struct Problem<Var, Value> {
    /// The list of constraints that must be satisfied
    pub constraints: Vec<Box<dyn Constraint<Var, Value>>>,
    /// The set of variables that must be assigned a value for the problem to be considered solved
    pub range: HashSet<Var>,
}

impl Debug for Problem<AtomTerm, ArcSort> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Problem")
            .field(
                "constraints",
                &self
                    .constraints
                    .iter()
                    .map(|c| c.pretty())
                    .collect::<Vec<_>>(),
            )
            .field("range", &self.range)
            .finish()
    }
}

impl<Var, Value> Default for Problem<Var, Value> {
    fn default() -> Self {
        Self {
            constraints: vec![],
            range: HashSet::default(),
        }
    }
}

/// A mapping from variables to their assigned values.
/// This is the result of constraint solving.
/// Uses an immutable HashMap for efficient cloning during constraint solving.
#[derive(Clone)]
pub struct Assignment<Var, Value>(pub HashMap<Var, Value>);

impl<Var, Value> Assignment<Var, Value>
where
    Var: Hash + cmp::Eq + PartialEq + Clone,
    Value: Clone,
{
    /// Insert into the assignment.
    pub fn insert(&mut self, var: Var, value: Value) -> Option<Value> {
        self.0.insert(var, value)
    }

    /// Get the value from the assignment.
    pub fn get(&self, var: &Var) -> Option<&Value> {
        self.0.get(var)
    }
}

impl Assignment<AtomTerm, ArcSort> {
    pub(crate) fn annotate_expr(
        &self,
        expr: &GenericExpr<CorrespondingVar<String, String>, String>,
        typeinfo: &TypeInfo,
    ) -> ResolvedExpr {
        match &expr {
            GenericExpr::Lit(span, literal) => ResolvedExpr::Lit(span.clone(), literal.clone()),
            GenericExpr::Var(span, var) => {
                let global_sort = typeinfo.get_global_sort(var);
                let ty = global_sort
                    // Span is ignored when looking up atom_terms
                    .or_else(|| self.get(&AtomTerm::Var(Span::Panic, var.clone())))
                    .expect("All variables should be assigned before annotation");
                ResolvedExpr::Var(
                    span.clone(),
                    ResolvedVar {
                        name: var.clone(),
                        sort: ty.clone(),
                        is_global_ref: global_sort.is_some(),
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
                    .map(|arg| arg.output_type())
                    .chain(once(
                        self.get(&AtomTerm::Var(span.clone(), corresponding_var.clone()))
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
        facts: &GenericFact<CorrespondingVar<String, String>, String>,
        typeinfo: &TypeInfo,
    ) -> ResolvedFact {
        match facts {
            GenericFact::Eq(span, e1, e2) => ResolvedFact::Eq(
                span.clone(),
                self.annotate_expr(e1, typeinfo),
                self.annotate_expr(e2, typeinfo),
            ),
            GenericFact::Fact(expr) => ResolvedFact::Fact(self.annotate_expr(expr, typeinfo)),
        }
    }

    pub(crate) fn annotate_facts(
        &self,
        mapped_facts: &[GenericFact<CorrespondingVar<String, String>, String>],
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
                    .get(&AtomTerm::Var(span.clone(), var.clone()))
                    .expect("All variables should be assigned before annotation");
                Ok(ResolvedAction::Let(
                    span.clone(),
                    ResolvedVar {
                        name: var.clone(),
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
                    .map(|child| child.output_type())
                    .chain(once(rhs.output_type()))
                    .collect();
                let resolved_call = ResolvedCall::from_resolution(head, &types, typeinfo);
                if !matches!(resolved_call, ResolvedCall::Func(_)) {
                    return Err(TypeError::UnboundFunction(head.clone(), span.clone()));
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
                let types: Vec<_> = children.iter().map(|child| child.output_type()).collect();
                let resolved_call =
                    ResolvedCall::from_resolution_func_types(head, &types, typeinfo)
                        .ok_or_else(|| TypeError::UnboundFunction(head.clone(), span.clone()))?;
                Ok(ResolvedAction::Change(
                    span.clone(),
                    *change,
                    resolved_call,
                    children.clone(),
                ))
            }
            GenericAction::Union(span, lhs, rhs) => {
                let lhs = self.annotate_expr(lhs, typeinfo);
                let rhs = self.annotate_expr(rhs, typeinfo);

                let sort = lhs.output_type();
                assert_eq!(sort.name(), rhs.output_type().name());
                if !sort.is_eq_sort() {
                    return Err(TypeError::NonEqsortUnion(sort, span.clone()));
                }

                Ok(ResolvedAction::Union(span.clone(), lhs, rhs))
            }
            GenericAction::Panic(span, msg) => Ok(ResolvedAction::Panic(span.clone(), msg.clone())),
            GenericAction::Expr(span, expr) => Ok(ResolvedAction::Expr(
                span.clone(),
                self.annotate_expr(expr, typeinfo),
            )),
        }
    }

    pub(crate) fn annotate_actions(
        &self,
        mapped_actions: &GenericActions<CorrespondingVar<String, String>, String>,
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
    Var: cmp::Eq + PartialEq + Hash + Clone + Debug + 'static,
    Value: Clone + Debug + 'static,
{
    pub(crate) fn solve(
        mut self,
        key: fn(&Value) -> &str,
    ) -> Result<Assignment<Var, Value>, ConstraintError<Var, Value>> {
        let mut assignment = Assignment(HashMap::default());
        let mut changed = true;
        while changed {
            changed = false;
            for constraint in self.constraints.iter_mut() {
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
        self.constraints.push(constraint::assign(var, clone));
    }
}

impl Problem<AtomTerm, ArcSort> {
    pub(crate) fn add_query(
        &mut self,
        query: &Query<StringOrEq, String>,
        typeinfo: &TypeInfo,
    ) -> Result<(), TypeError> {
        self.constraints.extend(query.get_constraints(typeinfo)?);
        self.range.extend(query.atom_terms());
        Ok(())
    }

    pub(crate) fn add_actions(
        &mut self,
        actions: &GenericCoreActions<String, String>,
        typeinfo: &TypeInfo,
        symbol_gen: &mut SymbolGen,
    ) -> Result<(), TypeError> {
        for action in actions.0.iter() {
            self.constraints
                .extend(action.get_constraints(typeinfo, symbol_gen)?);

            // bound vars are added to range
            match action {
                CoreAction::Let(span, var, _, _) => {
                    self.range.insert(AtomTerm::Var(span.clone(), var.clone()));
                }
                CoreAction::LetAtomTerm(span, v, _) => {
                    self.range.insert(AtomTerm::Var(span.clone(), v.clone()));
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
        symbol_gen: &mut SymbolGen,
    ) -> Result<(), TypeError> {
        let CoreRule {
            span: _,
            head,
            body,
        } = rule;
        self.add_query(body, typeinfo)?;
        self.add_actions(head, typeinfo, symbol_gen)?;
        Ok(())
    }

    pub(crate) fn assign_local_var_type(
        &mut self,
        var: &str,
        span: Span,
        sort: ArcSort,
    ) -> Result<(), TypeError> {
        self.add_binding(AtomTerm::Var(span.clone(), var.to_owned()), sort);
        self.range.insert(AtomTerm::Var(span, var.to_owned()));
        Ok(())
    }
}

impl CoreAction {
    pub(crate) fn get_constraints(
        &self,
        typeinfo: &TypeInfo,
        symbol_gen: &mut SymbolGen,
    ) -> Result<Vec<Box<dyn Constraint<AtomTerm, ArcSort>>>, TypeError> {
        match self {
            CoreAction::Let(span, symbol, f, args) => {
                let mut args = args.clone();
                args.push(AtomTerm::Var(span.clone(), symbol.clone()));

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
            .chain(once(constraint::eq(lhs.clone(), rhs.clone())))
            .collect()),
            CoreAction::Panic(_ann, _) => Ok(vec![]),
            CoreAction::LetAtomTerm(span, v, at) => {
                Ok(get_literal_and_global_constraints(&[at.clone()], typeinfo)
                    .chain(once(constraint::eq(
                        AtomTerm::Var(span.clone(), v.clone()),
                        at.clone(),
                    )))
                    .collect())
            }
        }
    }
}

impl Atom<StringOrEq> {
    pub(crate) fn get_constraints(
        &self,
        type_info: &TypeInfo,
    ) -> Result<Vec<Box<dyn Constraint<AtomTerm, ArcSort>>>, TypeError> {
        let literal_constraints = get_literal_and_global_constraints(&self.args, type_info);
        match &self.head {
            StringOrEq::Eq => {
                assert_eq!(self.args.len(), 2);
                let constraints = literal_constraints
                    .chain(once(constraint::eq(
                        self.args[0].clone(),
                        self.args[1].clone(),
                    )))
                    .collect();
                Ok(constraints)
            }
            StringOrEq::Head(head) => Ok(literal_constraints
                .chain(get_atom_application_constraints(
                    head, &self.args, &self.span, type_info,
                )?)
                .collect()),
        }
    }
}

fn get_atom_application_constraints(
    head: &str,
    args: &[AtomTerm],
    span: &Span,
    type_info: &TypeInfo,
) -> Result<Vec<Box<dyn Constraint<AtomTerm, ArcSort>>>, TypeError> {
    // An atom can have potentially different semantics due to polymorphism
    // e.g. (set-empty) can mean any empty set with some element type.
    // To handle this, we collect each possible instantiations of an atom
    // (where each instantiation is a vec of constraints, thus vec of vec)
    // into `xor_constraints`.
    // `constraint::xor` means one and only one of the instantiation can hold.
    let mut xor_constraints: Vec<Vec<Box<dyn Constraint<AtomTerm, ArcSort>>>> = vec![];

    // function atom constraints
    if let Some(typ) = type_info.get_func_type(head) {
        let mut constraints = vec![];
        // arity mismatch
        if typ.input.len() + 1 != args.len() {
            constraints.push(constraint::impossible(
                ImpossibleConstraint::ArityMismatch {
                    atom: Atom {
                        span: span.clone(),
                        head: head.to_owned(),
                        args: args.to_vec(),
                    },
                    expected: typ.input.len() + 1,
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
                constraints.push(constraint::assign(arg, arg_typ));
            }
        }
        xor_constraints.push(constraints);
    }

    // primitive atom constraints
    if let Some(primitives) = type_info.get_prims(head) {
        for p in primitives {
            let constraints = p.0.get_type_constraints(span).get(args, type_info);
            xor_constraints.push(constraints);
        }
    }

    // do literal and global variable constraints first
    // as they are the most "informative"
    match xor_constraints.len() {
        0 => Err(TypeError::UnboundFunction(head.to_owned(), span.clone())),
        1 => Ok(xor_constraints.pop().unwrap()),
        _ => Ok(vec![constraint::xor(
            xor_constraints.into_iter().map(constraint::and).collect(),
        )]),
    }
}

fn get_literal_and_global_constraints<'a>(
    args: &'a [AtomTerm],
    type_info: &'a TypeInfo,
) -> impl Iterator<Item = Box<dyn Constraint<AtomTerm, ArcSort>>> + 'a {
    args.iter().filter_map(|arg| {
        match arg {
            AtomTerm::Var(_, _) => None,
            // Literal to type constraint
            AtomTerm::Literal(_, lit) => {
                let typ = crate::sort::literal_sort(lit);
                Some(constraint::assign(arg.clone(), typ) as Box<dyn Constraint<AtomTerm, ArcSort>>)
            }
            AtomTerm::Global(_, v) => {
                if let Some(typ) = type_info.get_global_sort(v) {
                    Some(constraint::assign(arg.clone(), typ.clone()))
                } else {
                    panic!("All global variables should be bound before type checking")
                }
            }
        }
    })
}

/// A trait for generating type constraints from atom applications.
/// This is used to create constraints that ensure proper typing of function/primitive applications.
pub trait TypeConstraint {
    /// Generates constraints for the given arguments based on this type constraint.
    /// The constraints ensure that the arguments have compatible types.
    fn get(
        &self,
        arguments: &[AtomTerm],
        typeinfo: &TypeInfo,
    ) -> Vec<Box<dyn Constraint<AtomTerm, ArcSort>>>;
}

/// A type constraint that assigns specific sorts to each argument position.
/// Constructs a set of `Assign` constraints that fully constrain the type of arguments.
pub struct SimpleTypeConstraint {
    name: String,
    sorts: Vec<ArcSort>,
    span: Span,
}

impl SimpleTypeConstraint {
    /// Constructs a `SimpleTypeConstraint`
    pub fn new(name: &str, sorts: Vec<ArcSort>, span: Span) -> SimpleTypeConstraint {
        let name = name.to_owned();
        SimpleTypeConstraint { name, sorts, span }
    }

    /// Converts self to a boxed type constraint.
    pub fn into_box(self) -> Box<dyn TypeConstraint> {
        Box::new(self)
    }
}

impl TypeConstraint for SimpleTypeConstraint {
    fn get(
        &self,
        arguments: &[AtomTerm],
        _typeinfo: &TypeInfo,
    ) -> Vec<Box<dyn Constraint<AtomTerm, ArcSort>>> {
        if arguments.len() != self.sorts.len() {
            vec![constraint::impossible(
                ImpossibleConstraint::ArityMismatch {
                    atom: Atom {
                        span: self.span.clone(),
                        head: self.name.clone(),
                        args: arguments.to_vec(),
                    },
                    expected: self.sorts.len(),
                },
            )]
        } else {
            arguments
                .iter()
                .cloned()
                .zip(self.sorts.iter().cloned())
                .map(|(arg, sort)| constraint::assign(arg, sort))
                .collect()
        }
    }
}

/// A type constraint that requires all or some arguments to have the same type.
///
/// See the `with_all_arguments_sort`, `with_exact_length`, and `with_output_sort` methods
/// for configuring the constraint.
pub struct AllEqualTypeConstraint {
    name: String,
    sort: Option<ArcSort>,
    exact_length: Option<usize>,
    output: Option<ArcSort>,
    span: Span,
}

impl AllEqualTypeConstraint {
    /// Creates the `AllEqualTypeConstraint`.
    pub fn new(name: &str, span: Span) -> AllEqualTypeConstraint {
        AllEqualTypeConstraint {
            name: name.to_owned(),
            sort: None,
            exact_length: None,
            output: None,
            span,
        }
    }

    /// Converts self into a boxed type constraint.
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
    ) -> Vec<Box<dyn Constraint<AtomTerm, ArcSort>>> {
        if arguments.is_empty() {
            panic!("all arguments should have length > 0")
        }

        match self.exact_length {
            Some(exact_length) if exact_length != arguments.len() => {
                return vec![constraint::impossible(
                    ImpossibleConstraint::ArityMismatch {
                        atom: Atom {
                            span: self.span.clone(),
                            head: self.name.clone(),
                            args: arguments.to_vec(),
                        },
                        expected: exact_length,
                    },
                )];
            }
            _ => (),
        }

        let mut constraints = vec![];
        if let Some(output) = self.output.clone() {
            let (out, inputs) = arguments.split_last().unwrap();
            constraints.push(constraint::assign(out.clone(), output));
            arguments = inputs;
        }

        if let Some(sort) = self.sort.clone() {
            constraints.extend(
                arguments
                    .iter()
                    .cloned()
                    .map(|arg| constraint::assign(arg, sort.clone())),
            )
        } else if let Some((first, rest)) = arguments.split_first() {
            constraints.extend(
                rest.iter()
                    .cloned()
                    .map(|arg| constraint::eq(arg, first.clone())),
            );
        }
        constraints
    }
}

/// Checks that all variables in a rule's body are properly grounded.
/// A variable is grounded if it appears in a function call or is equal to a grounded variable.
/// This pass happens after type resolution and lowering to core rules, but before canonicalization.
pub(crate) fn grounded_check(
    rule: &GenericCoreRule<HeadOrEq<ResolvedCall>, ResolvedCall, ResolvedVar>,
) -> Result<(), TypeError> {
    use crate::core::ResolvedAtomTerm;
    let body = &rule.body;

    let range = rule
        .body
        .get_vars()
        .into_iter()
        .map(|v| ResolvedAtomTerm::Var(rule.span.clone(), v))
        .collect();
    let mut problem: Problem<ResolvedAtomTerm, ()> = Problem {
        constraints: vec![],
        range,
    };

    for atom in body.atoms.iter() {
        let mut add_global_and_literal = false;
        match &atom.head {
            HeadOrEq::Head(ResolvedCall::Func(_)) => {
                for arg in atom.args.iter() {
                    problem.constraints.push(assign(arg.clone(), ()));
                }
            }
            HeadOrEq::Head(ResolvedCall::Primitive(_)) => {
                let (out, inp) = atom.args.split_last().unwrap();
                let out = out.clone();
                problem.constraints.push(implies(
                    format!("grounded_{:?}", out),
                    inp.to_vec(),
                    Rc::new(move |_| assign(out.clone(), ())),
                ));
                add_global_and_literal = true;
            }
            HeadOrEq::Eq => {
                assert_eq!(atom.args.len(), 2);
                problem
                    .constraints
                    .push(eq(atom.args[0].clone(), atom.args[1].clone()));
                add_global_and_literal = true;
            }
        }
        if add_global_and_literal {
            for arg in atom.args.iter() {
                match arg {
                    ResolvedAtomTerm::Global(..) | ResolvedAtomTerm::Literal(..) => {
                        problem.constraints.push(assign(arg.clone(), ()));
                    }
                    ResolvedAtomTerm::Var(..) => {}
                }
            }
        }
    }

    let _assignment = problem.solve(|_| "grounded").map_err(|err| match err {
        ConstraintError::UnconstrainedVar(ResolvedAtomTerm::Var(span, v)) => {
            TypeError::Ungrounded(v.to_string(), span)
        }
        _ => panic!(
            "unexpected constraint error in groundedness check {:?}",
            err
        ),
    })?;

    Ok(())
}
