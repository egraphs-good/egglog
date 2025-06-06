use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VecContainer {
    do_rebuild: bool,
    pub data: Vec<Value>,
}

impl Container for VecContainer {
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
    name: Symbol,
    element: ArcSort,
}

impl VecSort {
    pub fn element(&self) -> ArcSort {
        self.element.clone()
    }
}

impl Presort for VecSort {
    fn presort_name() -> Symbol {
        "Vec".into()
    }

    fn reserved_primitives() -> Vec<Symbol> {
        vec![
            "vec-of".into(),
            "vec-append".into(),
            "vec-empty".into(),
            "vec-push".into(),
            "vec-pop".into(),
            "vec-not-contains".into(),
            "vec-contains".into(),
            "vec-length".into(),
            "vec-get".into(),
            "vec-set".into(),
            "vec-remove".into(),
        ]
    }

    fn make_sort(
        typeinfo: &mut TypeInfo,
        name: Symbol,
        args: &[Expr],
    ) -> Result<ArcSort, TypeError> {
        if let [Expr::Var(span, e)] = args {
            let e = typeinfo
                .get_sort_by_name(e)
                .ok_or(TypeError::UndefinedSort(*e, span.clone()))?;

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
            panic!("Vec sort must have sort as argument. Got {:?}", args)
        }
    }
}

impl ContainerSort for VecSort {
    type Container = VecContainer;

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
        let val = containers.get_val::<VecContainer>(value).unwrap().clone();
        val.data
            .iter()
            .map(|e| (self.element.clone(), *e))
            .collect()
    }

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
        _containers: &Containers,
        _value: Value,
        termdag: &mut TermDag,
        element_terms: Vec<Term>,
    ) -> Term {
        if element_terms.is_empty() {
            termdag.app("vec-empty".into(), vec![])
        } else {
            termdag.app("vec-of".into(), element_terms)
        }
    }

    fn serialized_name(&self, _value: Value) -> &str {
        "vec-of"
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
                r#"
            (sort IVec (Vec i64))
            (let v0 (vec-empty))
            (let v1 (vec-of 1 2 3 4))
            (extract v0)
            (extract v1)
            "#,
            )
            .unwrap();

        // Check extracted expr is parsed as an original expr
        egraph
            .parse_and_run_program(
                None,
                &format!(
                    r#"
                (check (= v0 {}))
                (check (= v1 {}))
                "#,
                    outputs[0], outputs[1],
                ),
            )
            .unwrap();
    }
}
