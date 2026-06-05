use super::*;
use std::any::TypeId;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MaybeContainer {
    pub do_rebuild: bool,
    pub data: Option<Value>,
}

impl ContainerValue for MaybeContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn Rebuilder) -> bool {
        if self.do_rebuild {
            if let Some(old) = self.data {
                let new = rebuilder.rebuild_val(old);
                self.data = Some(new);
                old != new
            } else {
                false
            }
        } else {
            false
        }
    }

    fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        self.data.iter().copied()
    }
}

#[derive(Clone, Debug)]
pub struct MaybeSort {
    name: String,
    element: ArcSort,
}

impl MaybeSort {
    pub fn element(&self) -> ArcSort {
        self.element.clone()
    }
}

impl Presort for MaybeSort {
    fn presort_name() -> &'static str {
        "Maybe"
    }

    fn reserved_primitives() -> Vec<&'static str> {
        vec![
            "maybe-none",
            "maybe-some",
            "maybe-unwrap",
            "maybe-unwrap-or",
            "maybe-f64-merge-with-tol",
            "unstable-maybe-match",
        ]
    }

    fn make_sort(
        typeinfo: &mut TypeInfo,
        name: String,
        args: &[Expr],
    ) -> Result<ArcSort, TypeError> {
        if let [Expr::Var(span, element)] = args {
            let element = typeinfo
                .get_sort_by_name(element)
                .ok_or(TypeError::UndefinedSort(element.clone(), span.clone()))?;

            Ok(Self {
                name,
                element: element.clone(),
            }
            .to_arcsort())
        } else {
            panic!("Maybe sort requires exactly one argument")
        }
    }
}

impl ContainerSort for MaybeSort {
    type Container = MaybeContainer;

    fn name(&self) -> &str {
        &self.name
    }

    fn inner_sorts(&self) -> Vec<ArcSort> {
        vec![self.element.clone()]
    }

    fn is_eq_container_sort(&self) -> bool {
        self.element.is_eq_sort() || self.element.is_eq_container_sort()
    }

    fn inner_values(
        &self,
        container_values: &ContainerValues,
        value: Value,
    ) -> Vec<(ArcSort, Value)> {
        let val = container_values
            .get_val::<MaybeContainer>(value)
            .unwrap()
            .clone();
        val.data
            .iter()
            .map(|v| (self.element.clone(), *v))
            .collect()
    }

    fn register_primitives(&self, eg: &mut EGraph) {
        let arc = self.clone().to_arcsort();

        add_primitive!(eg, "maybe-none" = {self.clone(): MaybeSort} || -> @MaybeContainer (arc) { MaybeContainer {
            do_rebuild: self.ctx.is_eq_container_sort(),
            data: None,
        } });

        add_primitive!(eg, "maybe-some" = {self.clone(): MaybeSort} |x: # (self.element())| -> @MaybeContainer (arc) { MaybeContainer {
            do_rebuild: self.ctx.is_eq_container_sort(),
            data: Some(x),
        } });

        add_primitive!(eg, "maybe-unwrap" = |xs: @MaybeContainer (arc)| -?> # (self.element()) { xs.data });
        add_primitive!(eg, "maybe-unwrap-or" = |xs: @MaybeContainer (arc), default: # (self.element())| -> # (self.element()) {
            xs.data.unwrap_or(default)
        });

        if self.element().name() == "f64" {
            add_primitive!(eg, "maybe-f64-merge-with-tol" = |old: @MaybeContainer (arc), new: @MaybeContainer (arc), tol: F| -?> @MaybeContainer (arc) {{
                match (old.data, new.data) {
                    (None, _) | (_, None) => Some(MaybeContainer { data: None, ..old }),
                    (Some(old_value), Some(new_value)) => {
                        let old_f = state.base_values().unwrap::<F>(old_value).0.0;
                        let new_f = state.base_values().unwrap::<F>(new_value).0.0;
                        let tolerance = tol.0.0.abs();
                        let merged =
                            old_f == new_f ||
                            (old_f == 0.0 && new_f == -0.0) ||
                            (old_f == -0.0 && new_f == 0.0) ||
                            (old_f - new_f).abs() <= tolerance;
                        merged.then_some(old)
                    }
                }
            }});
        }

        let maybe = eg.type_info.get_sort_by_name(self.name()).unwrap().clone();
        for fn_sort in eg.type_info.get_sorts::<FunctionSort>() {
            try_registering_maybe_match(eg, maybe.clone(), fn_sort);
        }
    }

    fn reconstruct_termdag(
        &self,
        _container_values: &ContainerValues,
        _value: Value,
        termdag: &mut TermDag,
        element_terms: Vec<TermId>,
    ) -> TermId {
        match element_terms.as_slice() {
            [] => termdag.app("maybe-none".into(), vec![]),
            [value] => termdag.app("maybe-some".into(), vec![*value]),
            _ => panic!("Maybe sort expected at most one element"),
        }
    }

    fn serialized_name(&self, container_values: &ContainerValues, value: Value) -> String {
        let maybe = container_values.get_val::<MaybeContainer>(value).unwrap();
        if maybe.data.is_some() {
            "maybe-some".to_owned()
        } else {
            "maybe-none".to_owned()
        }
    }
}

pub(crate) fn try_registering_maybe_match(eg: &mut EGraph, maybe: ArcSort, fn_: Arc<FunctionSort>) {
    if maybe.value_type() != Some(TypeId::of::<MaybeContainer>())
        || fn_.inputs().len() != 1
        || fn_.inputs()[0].name() != maybe.inner_sorts()[0].name()
    {
        return;
    }

    eg.add_pure_primitive(
        MaybeMatch {
            name: "unstable-maybe-match".into(),
            maybe,
            fn_,
        },
        None,
    );
}

pub(crate) fn register_maybe_primitives_for_function(eg: &mut EGraph, fn_: Arc<FunctionSort>) {
    for maybe in eg
        .type_info
        .get_arcsorts_by(|sort| sort.value_type() == Some(TypeId::of::<MaybeContainer>()))
    {
        try_registering_maybe_match(eg, maybe, fn_.clone());
    }
}

#[derive(Clone)]
struct MaybeMatch {
    name: String,
    maybe: ArcSort,
    fn_: Arc<FunctionSort>,
}

impl Primitive for MaybeMatch {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            &self.name,
            vec![
                self.maybe.clone(),
                self.fn_.clone(),
                self.fn_.output(),
                self.fn_.output(),
            ],
            span.clone(),
        )
        .into_box()
    }
}

impl PurePrim for MaybeMatch {
    fn apply<'a, 'db>(&self, mut state: PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let maybe = state
            .container_values()
            .get_val::<MaybeContainer>(args[0])?
            .clone();
        let fc = state
            .container_values()
            .get_val::<FunctionContainer>(args[1])?
            .clone();

        match maybe.data {
            Some(value) => state.apply_function(&fc, &[value]),
            None => Some(args[2]),
        }
    }
}
