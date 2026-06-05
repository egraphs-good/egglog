use super::*;

#[derive(Debug)]
pub struct UnitSort;

impl BaseSort for UnitSort {
    type Base = ();

    fn name(&self) -> &str {
        "Unit"
    }

    fn reconstruct_termdag(
        &self,
        _base_values: &BaseValues,
        _value: Value,
        termdag: &mut TermDag,
    ) -> TermId {
        termdag.lit(Literal::Unit)
    }
}

pub(crate) fn try_registering_if(eg: &mut EGraph, fn_: Arc<FunctionSort>, output: ArcSort) {
    if !fn_.inputs().is_empty() || fn_.output().name() != UnitSort.name() {
        return;
    }

    eg.add_pure_primitive(
        IfPrim {
            name: "unstable-if".into(),
            fn_,
            output,
        },
        None,
    );
}

pub(crate) fn register_if_primitives_for_function(eg: &mut EGraph, fn_: Arc<FunctionSort>) {
    for output in eg.type_info.get_arcsorts_by(|_| true) {
        try_registering_if(eg, fn_.clone(), output);
    }
}

pub(crate) fn register_if_primitives_for_output(eg: &mut EGraph, output: ArcSort) {
    for fn_ in eg.type_info.get_sorts::<FunctionSort>() {
        try_registering_if(eg, fn_, output.clone());
    }
}

#[derive(Clone)]
struct IfPrim {
    name: String,
    fn_: Arc<FunctionSort>,
    output: ArcSort,
}

impl Primitive for IfPrim {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            &self.name,
            vec![
                self.fn_.clone(),
                self.output.clone(),
                self.output.clone(),
                self.output.clone(),
            ],
            span.clone(),
        )
        .into_box()
    }
}

impl PurePrim for IfPrim {
    fn apply<'a, 'db>(&self, mut state: PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let fc = state
            .container_values()
            .get_val::<FunctionContainer>(args[0])?
            .clone();
        match state.apply_function(&fc, &[]) {
            Some(_) => Some(args[1]),
            None => Some(args[2]),
        }
    }
}
