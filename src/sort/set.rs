use super::*;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SetContainer {
    do_rebuild: bool,
    pub data: BTreeSet<Value>,
}

impl Container for SetContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn Rebuilder) -> bool {
        if self.do_rebuild {
            let mut xs: Vec<_> = self.data.iter().copied().collect();
            let changed = rebuilder.rebuild_slice(&mut xs);
            self.data = xs.into_iter().collect();
            changed
        } else {
            false
        }
    }
    fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        self.data.iter().copied()
    }
}

#[derive(Clone, Debug)]
pub struct SetSort {
    name: Symbol,
    element: ArcSort,
}

impl SetSort {
    pub fn element(&self) -> ArcSort {
        self.element.clone()
    }
}

impl Presort for SetSort {
    fn presort_name() -> Symbol {
        "Set".into()
    }

    fn reserved_primitives() -> Vec<Symbol> {
        vec![
            "set-of".into(),
            "set-empty".into(),
            "set-insert".into(),
            "set-not-contains".into(),
            "set-contains".into(),
            "set-remove".into(),
            "set-union".into(),
            "set-diff".into(),
            "set-intersect".into(),
            "set-get".into(),
            "set-length".into(),
        ]
    }

    fn make_sort(
        eg: &mut EGraph,
        name: Symbol,
        args: &[Expr],
        span: Span,
    ) -> Result<(), TypeError> {
        if let [Expr::Var(arg_span, e)] = args {
            let e = eg
                .get_sort_by_name(e)
                .ok_or(TypeError::UndefinedSort(*e, arg_span.clone()))?;

            if e.is_eq_container_sort() {
                return Err(TypeError::DisallowedSort(
                    name,
                    "Sets nested with other EqSort containers are not allowed".into(),
                    arg_span.clone(),
                ));
            }

            add_container_sort(
                eg,
                Self {
                    name,
                    element: e.clone(),
                },
                span,
            )
        } else {
            panic!()
        }
    }
}

impl ContainerSort for SetSort {
    type Container = SetContainer;

    fn name(&self) -> Symbol {
        self.name
    }

    fn inner_sorts(&self) -> Vec<ArcSort> {
        vec![self.element.clone()]
    }

    fn is_eq_container_sort(&self) -> bool {
        self.element.is_eq_sort()
    }

    fn inner_values(&self, containers: &Containers, value: Value) -> Vec<(ArcSort, Value)> {
        let val = containers.get_val::<SetContainer>(value).unwrap().clone();
        val.data
            .iter()
            .map(|e| (self.element.clone(), *e))
            .collect()
    }

    fn register_primitives(&self, eg: &mut EGraph) {
        let arc = self.clone().to_arcsort();

        add_primitive!(eg, "set-empty" = {self.clone(): SetSort} |                      | -> @SetContainer (arc) { SetContainer {
            do_rebuild: self.ctx.element.is_eq_sort(),
            data: BTreeSet::new()
        } });
        add_primitive!(eg, "set-of"    = {self.clone(): SetSort} [xs: # (self.element())] -> @SetContainer (arc) { SetContainer {
            do_rebuild: self.ctx.element.is_eq_sort(),
            data: xs.collect()
        } });

        add_primitive!(eg, "set-get" = |xs: @SetContainer (arc), i: i64| -?> # (self.element()) { xs.data.iter().nth(i as usize).copied() });
        add_primitive!(eg, "set-insert" = |mut xs: @SetContainer (arc), x: # (self.element())| -> @SetContainer (arc) {{ xs.data.insert( x); xs }});
        add_primitive!(eg, "set-remove" = |mut xs: @SetContainer (arc), x: # (self.element())| -> @SetContainer (arc) {{ xs.data.remove(&x); xs }});

        add_primitive!(eg, "set-length"       = |xs: @SetContainer (arc)| -> i64 { xs.data.len() as i64 });
        add_primitive!(eg, "set-contains"     = |xs: @SetContainer (arc), x: # (self.element())| -?> () { ( xs.data.contains(&x)).then_some(()) });
        add_primitive!(eg, "set-not-contains" = |xs: @SetContainer (arc), x: # (self.element())| -?> () { (!xs.data.contains(&x)).then_some(()) });

        add_primitive!(eg, "set-union"      = |mut xs: @SetContainer (arc), ys: @SetContainer (arc)| -> @SetContainer (arc) {{ xs.data.extend(ys.data);                  xs }});
        add_primitive!(eg, "set-diff"       = |mut xs: @SetContainer (arc), ys: @SetContainer (arc)| -> @SetContainer (arc) {{ xs.data.retain(|k| !ys.data.contains(k)); xs }});
        add_primitive!(eg, "set-intersect"  = |mut xs: @SetContainer (arc), ys: @SetContainer (arc)| -> @SetContainer (arc) {{ xs.data.retain(|k|  ys.data.contains(k)); xs }});
    }

    fn reconstruct_termdag(
        &self,
        _containers: &Containers,
        _value: Value,
        termdag: &mut TermDag,
        element_terms: Vec<Term>,
    ) -> Term {
        termdag.app("set-of".into(), element_terms)
    }

    fn serialized_name(&self, _value: Value) -> &str {
        "set-of"
    }
}
