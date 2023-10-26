use crate::{
    typecheck::{Atom, AtomTerm, SymbolOrEq},
    typechecking::TypeError,
    util::HashMap,
    ArcSort, Symbol, TypeInfo,
};
use core::hash::Hash;
use std::{fmt::Debug, iter::once, mem::swap};

#[derive(Clone, Debug)]
pub enum ImpossibleConstraint {
    ArityMismatch {
        atom: Atom<Symbol>,
        expected: usize,
        actual: usize,
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
                reason: "mismatch".into(),
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
                assert_eq!(actual, &atom.args.len());
                TypeError::Arity {
                    expr: atom.to_expr(),
                    expected: *expected,
                }
            }
        }
    }
}

impl<Var, Value> Constraint<Var, Value>
where
    Var: Eq + PartialEq + Hash + Clone,
    Value: Clone,
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
                let mut updated = false;
                for c in cs {
                    updated |= c.update(assignment, key)?;
                }
                Ok(updated)
            }
        }
    }
}

#[derive(Default, Debug)]
pub struct Problem<Var, Value> {
    pub constraints: Vec<Constraint<Var, Value>>,
}

#[derive(Clone, Debug)]
pub(crate) struct Assignment<Var, Value>(pub HashMap<Var, Value>);

impl<Var, Value> Assignment<Var, Value>
where
    Var: Hash + Eq + PartialEq,
{
    pub fn insert(&mut self, var: Var, value: Value) -> Option<Value> {
        self.0.insert(var, value)
    }

    pub fn get(&self, var: &Var) -> Option<&Value> {
        self.0.get(var)
    }
}

impl<Var, Value> Problem<Var, Value>
where
    Var: Eq + PartialEq + Hash + Clone + Debug,
    Value: Clone,
{
    pub(crate) fn solve<'a, K: Eq + Debug>(
        &'a self,
        range: impl Iterator<Item = &'a Var>,
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

        for v in range {
            if !assignment.0.contains_key(v) {
                return Err(ConstraintError::UnconstrainedVar(v.clone()));
            }
        }
        Ok(assignment)
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
            SymbolOrEq::Symbol(head) => {
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
                    if typ.input.len() + 1 != self.args.len() {
                        constraints.push(Constraint::Impossible(
                            ImpossibleConstraint::ArityMismatch {
                                atom: Atom {
                                    head: *head,
                                    args: self.args.clone(),
                                },
                                expected: typ.input.len(),
                                actual: self.args.len() - 1,
                            },
                        ));
                    } else {
                        for (arg_typ, arg) in typ
                            .input
                            .iter()
                            .cloned()
                            .chain(once(typ.output.clone()))
                            .zip(self.args.iter().cloned())
                        {
                            constraints.push(Constraint::Assign(arg, arg_typ));
                        }
                    }
                    xor_constraints.push(constraints);
                }

                // primitive atom constraints
                if let Some(primitives) = type_info.primitives.get(head) {
                    for p in primitives {
                        let constraints = p.get_type_constraints().get(&self.args);
                        xor_constraints.push(constraints);
                    }
                }

                // do literal and global variable constraints first
                // as they are the most "informative"
                match xor_constraints.len() {
                    0 => Err(TypeError::UnboundFunction(*head)),
                    1 => Ok(literal_constraints
                        .chain(xor_constraints.pop().unwrap().into_iter())
                        .collect()),
                    _ => Ok(literal_constraints
                        .chain(once(Constraint::Xor(
                            xor_constraints.into_iter().map(Constraint::And).collect(),
                        )))
                        .collect()),
                }
            }
        }
    }
}

fn get_literal_and_global_constraints<'a>(
    args: &'a [AtomTerm],
    type_info: &'a TypeInfo,
) -> impl Iterator<Item = Constraint<AtomTerm, ArcSort>> + 'a {
    args.iter().filter_map(|arg| {
        match arg {
            AtomTerm::Var(_) => None,
            // Literal to type constraint
            AtomTerm::Literal(lit) => {
                let typ = type_info.infer_literal(lit);
                Some(Constraint::Assign(arg.clone(), typ.clone()))
            }
            AtomTerm::Global(v) => {
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
    fn get(&self, arguments: &[AtomTerm]) -> Vec<Constraint<AtomTerm, ArcSort>>;
}

/// Construct a set of `Assign` constraints that fully constrain the type of arguments
pub struct SimpleTypeConstraint {
    name: Symbol,
    sorts: Vec<ArcSort>,
}

impl SimpleTypeConstraint {
    pub fn new(name: Symbol, sorts: Vec<ArcSort>) -> SimpleTypeConstraint {
        SimpleTypeConstraint { name, sorts }
    }

    pub(crate) fn into_box(self) -> Box<dyn TypeConstraint> {
        Box::new(self)
    }
}

impl TypeConstraint for SimpleTypeConstraint {
    fn get(&self, arguments: &[AtomTerm]) -> Vec<Constraint<AtomTerm, ArcSort>> {
        if arguments.len() != self.sorts.len() {
            vec![Constraint::Impossible(
                ImpossibleConstraint::ArityMismatch {
                    atom: Atom {
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
}

impl AllEqualTypeConstraint {
    pub fn new(name: Symbol) -> AllEqualTypeConstraint {
        AllEqualTypeConstraint {
            name,
            sort: None,
            exact_length: None,
            output: None,
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
    fn get(&self, mut arguments: &[AtomTerm]) -> Vec<Constraint<AtomTerm, ArcSort>> {
        if arguments.is_empty() {
            panic!("all arguments should have length > 0")
        }

        match self.exact_length {
            Some(exact_length) if exact_length != arguments.len() => {
                return vec![Constraint::Impossible(
                    ImpossibleConstraint::ArityMismatch {
                        atom: Atom {
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
