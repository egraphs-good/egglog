use super::*;
use std::any::TypeId;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EitherData {
    Left(Value),
    Right(Value),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EitherContainer {
    do_rebuild_left: bool,
    do_rebuild_right: bool,
    pub data: EitherData,
}

impl ContainerValue for EitherContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn Rebuilder) -> bool {
        match &mut self.data {
            EitherData::Left(value) if self.do_rebuild_left => {
                let old = *value;
                let new = rebuilder.rebuild_val(old);
                *value = new;
                old != new
            }
            EitherData::Right(value) if self.do_rebuild_right => {
                let old = *value;
                let new = rebuilder.rebuild_val(old);
                *value = new;
                old != new
            }
            _ => false,
        }
    }

    fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        match &self.data {
            EitherData::Left(value) | EitherData::Right(value) => Some(*value).into_iter(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EitherSort {
    name: String,
    left: ArcSort,
    right: ArcSort,
}

impl EitherSort {
    pub fn left(&self) -> ArcSort {
        self.left.clone()
    }

    pub fn right(&self) -> ArcSort {
        self.right.clone()
    }
}

impl Presort for EitherSort {
    fn presort_name() -> &'static str {
        "Either"
    }

    fn reserved_primitives() -> Vec<&'static str> {
        vec![
            "either-left",
            "either-right",
            "either-unwrap-left",
            "either-unwrap-right",
            "either-match",
        ]
    }

    // Proof support is presort-wide: every Either instance has the same
    // constructor/projection metadata. Instance-specific canonicalization is
    // still gated by `is_eq_container_sort` and the active branch's rebuild
    // flag, so `(Either i64 String)` is admissible but contributes no rebuild
    // work.
    fn supports_proof_encoding() -> bool {
        true
    }

    fn make_sort(
        typeinfo: &mut TypeInfo,
        name: String,
        args: &[Expr],
    ) -> Result<ArcSort, TypeError> {
        if let [Expr::Var(left_span, left), Expr::Var(right_span, right)] = args {
            let left = typeinfo
                .get_sort_by_name(left)
                .ok_or(TypeError::UndefinedSort(left.clone(), left_span.clone()))?;
            let right = typeinfo
                .get_sort_by_name(right)
                .ok_or(TypeError::UndefinedSort(right.clone(), right_span.clone()))?;

            Ok(Self {
                name,
                left: left.clone(),
                right: right.clone(),
            }
            .to_arcsort())
        } else {
            Err(TypeError::InvalidSortArity {
                sort: Self::presort_name().to_string(),
                expected: 2,
                actual: args.len(),
            })
        }
    }
}

impl ContainerSort for EitherSort {
    type Container = EitherContainer;

    fn name(&self) -> &str {
        &self.name
    }

    fn inner_sorts(&self) -> Vec<ArcSort> {
        vec![self.left.clone(), self.right.clone()]
    }

    fn is_eq_container_sort(&self) -> bool {
        self.left.is_eq_sort()
            || self.right.is_eq_sort()
            || self.left.is_eq_container_sort()
            || self.right.is_eq_container_sort()
    }

    fn container_proof_spec(&self) -> Option<ContainerProofSpec> {
        Some(ContainerProofSpec {
            constructors: vec![
                ContainerProofConstructorSpec {
                    name: "either-left",
                    input_sorts: vec![self.left.clone()],
                    projections: vec![ContainerProofProjectionSpec {
                        primitive: "either-unwrap-left",
                        field: 0,
                    }],
                },
                ContainerProofConstructorSpec {
                    name: "either-right",
                    input_sorts: vec![self.right.clone()],
                    projections: vec![ContainerProofProjectionSpec {
                        primitive: "either-unwrap-right",
                        field: 0,
                    }],
                },
            ],
        })
    }

    fn inner_values(
        &self,
        container_values: &ContainerValues,
        value: Value,
    ) -> Vec<(ArcSort, Value)> {
        let either = container_values.get_val::<EitherContainer>(value).unwrap();
        match &either.data {
            EitherData::Left(value) => vec![(self.left.clone(), *value)],
            EitherData::Right(value) => vec![(self.right.clone(), *value)],
        }
    }

    fn register_primitives(&self, eg: &mut EGraph) {
        let arc = self.clone().to_arcsort();

        add_primitive_with_validator!(
            eg,
            "either-left" = {self.clone(): EitherSort} |left: # (self.left())| -> @EitherContainer (arc) {
                EitherContainer {
                    do_rebuild_left: self.ctx.left.is_eq_sort() || self.ctx.left.is_eq_container_sort(),
                    do_rebuild_right: self.ctx.right.is_eq_sort() || self.ctx.right.is_eq_container_sort(),
                    data: EitherData::Left(left),
                }
            },
            |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
                if args.len() == 1 {
                    Some(termdag.app("either-left".into(), args.to_vec()))
                } else {
                    None
                }
            }
        );

        add_primitive_with_validator!(
            eg,
            "either-right" = {self.clone(): EitherSort} |right: # (self.right())| -> @EitherContainer (arc) {
                EitherContainer {
                    do_rebuild_left: self.ctx.left.is_eq_sort() || self.ctx.left.is_eq_container_sort(),
                    do_rebuild_right: self.ctx.right.is_eq_sort() || self.ctx.right.is_eq_container_sort(),
                    data: EitherData::Right(right),
                }
            },
            |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
                if args.len() == 1 {
                    Some(termdag.app("either-right".into(), args.to_vec()))
                } else {
                    None
                }
            }
        );

        add_primitive_with_validator!(
            eg,
            "either-unwrap-left" = |xs: @EitherContainer (arc)| -?> # (self.left()) {
                match xs.data {
                    EitherData::Left(value) => Some(value),
                    EitherData::Right(_) => None,
                }
            },
            |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
                let [either] = args else {
                    return None;
                };
                match termdag.get(*either) {
                    Term::App(head, children) if head == "either-left" && children.len() == 1 => {
                        Some(children[0])
                    }
                    _ => None,
                }
            }
        );

        add_primitive_with_validator!(
            eg,
            "either-unwrap-right" = |xs: @EitherContainer (arc)| -?> # (self.right()) {
                match xs.data {
                    EitherData::Left(_) => None,
                    EitherData::Right(value) => Some(value),
                }
            },
            |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
                let [either] = args else {
                    return None;
                };
                match termdag.get(*either) {
                    Term::App(head, children) if head == "either-right" && children.len() == 1 => {
                        Some(children[0])
                    }
                    _ => None,
                }
            }
        );

        let either = eg.type_info.get_sort_by_name(self.name()).unwrap().clone();
        let fn_sorts = eg.type_info.get_sorts::<FunctionSort>();
        for left_fn in &fn_sorts {
            for right_fn in &fn_sorts {
                try_registering_either_match(eg, either.clone(), left_fn.clone(), right_fn.clone());
            }
        }
    }

    fn reconstruct_termdag(
        &self,
        container_values: &ContainerValues,
        value: Value,
        termdag: &mut TermDag,
        element_terms: Vec<TermId>,
    ) -> TermId {
        assert_eq!(element_terms.len(), 1);
        let either = container_values.get_val::<EitherContainer>(value).unwrap();
        let head = match either.data {
            EitherData::Left(_) => "either-left",
            EitherData::Right(_) => "either-right",
        };
        termdag.app(head.into(), element_terms)
    }

    fn serialized_name(&self, container_values: &ContainerValues, value: Value) -> String {
        let either = container_values.get_val::<EitherContainer>(value).unwrap();
        match either.data {
            EitherData::Left(_) => "either-left".to_owned(),
            EitherData::Right(_) => "either-right".to_owned(),
        }
    }
}

pub(crate) fn register_either_primitives_for_function(eg: &mut EGraph, fn_: Arc<FunctionSort>) {
    let eithers = eg
        .type_info
        .get_arcsorts_by(|sort| sort.value_type() == Some(TypeId::of::<EitherContainer>()));
    let fn_sorts = eg.type_info.get_sorts::<FunctionSort>();
    for either in eithers {
        for other_fn in &fn_sorts {
            try_registering_either_match(eg, either.clone(), fn_.clone(), other_fn.clone());
            if !Arc::ptr_eq(&fn_, other_fn) {
                try_registering_either_match(eg, either.clone(), other_fn.clone(), fn_.clone());
            }
        }
    }
}

fn try_registering_either_match(
    eg: &mut EGraph,
    either: ArcSort,
    left_fn: Arc<FunctionSort>,
    right_fn: Arc<FunctionSort>,
) {
    if either.value_type() != Some(TypeId::of::<EitherContainer>())
        || left_fn.inputs().len() != 1
        || right_fn.inputs().len() != 1
        || left_fn.inputs()[0].name() != either.inner_sorts()[0].name()
        || right_fn.inputs()[0].name() != either.inner_sorts()[1].name()
        || left_fn.output().name() != right_fn.output().name()
    {
        return;
    }

    eg.add_pure_primitive(
        EitherMatch {
            name: "either-match".into(),
            either,
            left_fn,
            right_fn,
        },
        None,
    );
}

#[derive(Clone)]
struct EitherMatch {
    name: String,
    either: ArcSort,
    left_fn: Arc<FunctionSort>,
    right_fn: Arc<FunctionSort>,
}

impl Primitive for EitherMatch {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            &self.name,
            vec![
                self.either.clone(),
                self.left_fn.clone(),
                self.right_fn.clone(),
                self.left_fn.output(),
            ],
            span.clone(),
        )
        .into_box()
    }
}

impl PurePrim for EitherMatch {
    fn apply<'a, 'db>(&self, mut state: PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let either = state
            .container_values()
            .get_val::<EitherContainer>(args[0])?
            .clone();
        let left_fn = state
            .container_values()
            .get_val::<FunctionContainer>(args[1])?
            .clone();
        let right_fn = state
            .container_values()
            .get_val::<FunctionContainer>(args[2])?
            .clone();

        match either.data {
            EitherData::Left(value) => state.apply_function(&left_fn, &[value]),
            EitherData::Right(value) => state.apply_function(&right_fn, &[value]),
        }
    }
}
