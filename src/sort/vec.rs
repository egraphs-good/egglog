use super::{
    ArcSort, ContainerSort, ContainerValue, ContainerValues, Debug, EGraph, Expr, Hash, Presort,
    Rebuilder, Term, TermDag, TypeError, TypeInfo, Value, add_primitive, bool, i64,
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VecContainer {
    pub do_rebuild: bool,
    pub data: Vec<Value>,
}

impl ContainerValue for VecContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn Rebuilder) -> bool {
        if self.do_rebuild {
            rebuilder.rebuild_slice(&mut self.data)
        } else {
            false
        }
    }
    fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        self.data.iter().copied()
    }
}

#[derive(Clone, Debug)]
pub struct VecSort {
    name: String,
    element: ArcSort,
}

impl VecSort {
    #[must_use]
    pub fn element(&self) -> ArcSort {
        self.element.clone()
    }
}

impl Presort for VecSort {
    fn presort_name() -> &'static str {
        "Vec"
    }

    fn reserved_primitives() -> Vec<&'static str> {
        vec![
            "vec-of",
            "vec-append",
            "vec-empty",
            "vec-push",
            "vec-pop",
            "vec-not-contains",
            "vec-contains",
            "vec-length",
            "vec-get",
            "vec-set",
            "vec-remove",
        ]
    }

    fn make_sort(
        typeinfo: &mut TypeInfo,
        name: String,
        args: &[Expr],
    ) -> Result<ArcSort, TypeError> {
        if let [Expr::Var(span, e)] = args {
            let e = typeinfo
                .get_sort_by_name(e)
                .ok_or(TypeError::UndefinedSort(e.clone(), span.clone()))?;

            if e.is_eq_container_sort() {
                return Err(TypeError::DisallowedSort(
                    name,
                    "Vec nested with other EqSort containers are not allowed".into(),
                    span.clone(),
                ));
            }

            let out = Self {
                name,
                element: e.clone(),
            };
            Ok(out.to_arcsort())
        } else {
            panic!("Vec sort must have sort as argument. Got {args:?}")
        }
    }
}

impl ContainerSort for VecSort {
    type Container = VecContainer;

    fn name(&self) -> &str {
        &self.name
    }

    fn inner_sorts(&self) -> Vec<ArcSort> {
        vec![self.element.clone()]
    }

    fn is_eq_container_sort(&self) -> bool {
        self.element.is_eq_sort()
    }

    fn inner_values(
        &self,
        container_values: &ContainerValues,
        value: Value,
    ) -> Vec<(ArcSort, Value)> {
        let val = container_values
            .get_val::<VecContainer>(value)
            .unwrap()
            .clone();
        val.data
            .iter()
            .map(|e| (self.element.clone(), *e))
            .collect()
    }

    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    fn register_primitives(&self, eg: &mut EGraph) {
        let arc = self.clone().to_arcsort();

        add_primitive!(eg, "vec-empty"  = {self.clone(): VecSort} |                                | -> @VecContainer (arc) { VecContainer {
            do_rebuild: self.ctx.element.is_eq_sort(),
            data: Vec::new()
        } });
        add_primitive!(eg, "vec-of"     = {self.clone(): VecSort} [xs: # (self.element())          ] -> @VecContainer (arc) { VecContainer {
            do_rebuild: self.ctx.element.is_eq_sort(),
            data: xs                     .collect()
        } });
        add_primitive!(eg, "vec-append" = {self.clone(): VecSort} [xs: @VecContainer (arc)] -> @VecContainer (arc) { VecContainer {
            do_rebuild: self.ctx.element.is_eq_sort(),
            data: xs.flat_map(|x| x.data).collect()
        } });

        add_primitive!(eg, "vec-push" = |mut xs: @VecContainer (arc), x: # (self.element())| -> @VecContainer (arc) {{ xs.data.push(x); xs }});
        add_primitive!(eg, "vec-pop"  = |mut xs: @VecContainer (arc)                       | -> @VecContainer (arc) {{ xs.data.pop();   xs }});

        add_primitive!(eg, "vec-length"       = |xs: @VecContainer (arc)| -> i64 { xs.data.len() as i64 });
        add_primitive!(eg, "vec-contains"     = |xs: @VecContainer (arc), x: # (self.element())| -?> () { ( xs.data.contains(&x)).then_some(()) });
        add_primitive!(eg, "vec-not-contains" = |xs: @VecContainer (arc), x: # (self.element())| -?> () { (!xs.data.contains(&x)).then_some(()) });

        add_primitive!(eg, "vec-get"    = |    xs: @VecContainer (arc), i: i64                       | -?> # (self.element()) { xs.data.get(i as usize).copied() });
        add_primitive!(eg, "vec-set"    = |mut xs: @VecContainer (arc), i: i64, x: # (self.element())| -> @VecContainer (arc) {{ xs.data[i as usize] = x;    xs }});
        add_primitive!(eg, "vec-remove" = |mut xs: @VecContainer (arc), i: i64                       | -> @VecContainer (arc) {{ xs.data.remove(i as usize); xs }});
    }

    fn reconstruct_termdag(
        &self,
        _container_values: &ContainerValues,
        _value: Value,
        termdag: &mut TermDag,
        element_terms: Vec<Term>,
    ) -> Term {
        if element_terms.is_empty() {
            termdag.app("vec-empty".into(), &[])
        } else {
            termdag.app("vec-of".into(), &element_terms)
        }
    }

    fn serialized_name(&self, _container_values: &ContainerValues, _: Value) -> String {
        "vec-of".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_make_expr() {
        let mut egraph = EGraph::default();
        let outputs = egraph
            .parse_and_run_program(
                None,
                r"
            (sort IVec (Vec i64))
            (let v0 (vec-empty))
            (let v1 (vec-of 1 2 3 4))
            (extract v0)
            (extract v1)
            ",
            )
            .unwrap();

        // Check extracted expr is parsed as an original expr
        egraph
            .parse_and_run_program(
                None,
                &format!(
                    r"
                (check (= v0 {}))
                (check (= v1 {}))
                ",
                    outputs[0], outputs[1],
                ),
            )
            .unwrap();
    }
}
