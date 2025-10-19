use super::*;
use rpds::RedBlackTreeSetSync;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SetContainer {
    pub do_rebuild: bool,
    pub data: RedBlackTreeSetSync<Value>,
}

impl ContainerValue for SetContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn Rebuilder) -> bool {
        if self.do_rebuild {
            let mut changed = false;
            let mut to_remove = Vec::new();
            let mut to_add = Vec::new();
            for v in self.data.iter() {
                let rebuilt = rebuilder.rebuild_val(*v);
                if rebuilt != *v {
                    changed |= true;
                    to_remove.push(*v);
                    to_add.push(rebuilt);
                }
            }
            if changed {
                for v in to_remove {
                    self.data.remove_mut(&v);
                }
                for v in to_add {
                    self.data.insert_mut(v);
                }
            }
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
    name: String,
    element: ArcSort,
}

impl SetSort {
    pub fn element(&self) -> ArcSort {
        self.element.clone()
    }
}

impl Presort for SetSort {
    fn presort_name() -> &'static str {
        "Set"
    }

    fn reserved_primitives() -> Vec<&'static str> {
        vec![
            "set-of",
            "set-empty",
            "set-insert",
            "set-not-contains",
            "set-contains",
            "set-remove",
            "set-union",
            "set-diff",
            "set-intersect",
            "set-get",
            "set-length",
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
                    "Sets nested with other EqSort containers are not allowed".into(),
                    span.clone(),
                ));
            }

            let out = Self {
                name,
                element: e.clone(),
            };
            Ok(out.to_arcsort())
        } else {
            panic!()
        }
    }
}

impl ContainerSort for SetSort {
    type Container = SetContainer;

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
            .get_val::<SetContainer>(value)
            .unwrap()
            .clone();
        val.data
            .iter()
            .map(|e| (self.element.clone(), *e))
            .collect()
    }

    fn register_primitives(&self, eg: &mut EGraph) {
        let arc = self.clone().to_arcsort();

        add_primitive!(eg, "set-empty" = {self.clone(): SetSort} |                      | -> @SetContainer (arc) { SetContainer {
            do_rebuild: self.ctx.element.is_eq_sort(),
            data: RedBlackTreeSetSync::new_sync()
        } });
        add_primitive!(eg, "set-of"    = {self.clone(): SetSort} [xs: # (self.element())] -> @SetContainer (arc) { SetContainer {
            do_rebuild: self.ctx.element.is_eq_sort(),
            data: xs.collect::<RedBlackTreeSetSync<_>>()
        } });

        add_primitive!(eg, "set-get" = |xs: @SetContainer (arc), i: i64| -?> # (self.element()) { xs.data.iter().nth(i as usize).copied() });
        add_primitive!(eg, "set-insert" = |mut xs: @SetContainer (arc), x: # (self.element())| -> @SetContainer (arc) {{ xs.data.insert_mut(x); xs }});
        add_primitive!(eg, "set-remove" = |mut xs: @SetContainer (arc), x: # (self.element())| -> @SetContainer (arc) {{ xs.data.remove_mut(&x); xs }});

        add_primitive!(eg, "set-length"       = |xs: @SetContainer (arc)| -> i64 { xs.data.size() as i64 });
        add_primitive!(eg, "set-contains"     = |xs: @SetContainer (arc), x: # (self.element())| -?> () { ( xs.data.contains(&x)).then_some(()) });
        add_primitive!(eg, "set-not-contains" = |xs: @SetContainer (arc), x: # (self.element())| -?> () { (!xs.data.contains(&x)).then_some(()) });

        add_primitive!(eg, "set-union"     = |mut xs: @SetContainer (arc), ys: @SetContainer (arc)| -> @SetContainer (arc) {{
            for value in ys.data.iter().copied() {
                xs.data.insert_mut(value);
            }
            xs
        }});
        add_primitive!(eg, "set-diff"      = |mut xs: @SetContainer (arc), ys: @SetContainer (arc)| -> @SetContainer (arc) {{
            for value in ys.data.iter() {
                xs.data.remove_mut(value);
            }
            xs
        }});
        add_primitive!(eg, "set-intersect" = |mut xs: @SetContainer (arc), ys: @SetContainer (arc)| -> @SetContainer (arc) {{
            let mut to_remove = Vec::new();
            for value in xs.data.iter().copied() {
                if !ys.data.contains(&value) {
                    to_remove.push(value);
                }
            }
            for value in to_remove {
                xs.data.remove_mut(&value);
            }
            xs
        }});
    }

    fn reconstruct_termdag(
        &self,
        _container_values: &ContainerValues,
        _value: Value,
        termdag: &mut TermDag,
        element_terms: Vec<Term>,
    ) -> Term {
        termdag.app("set-of".into(), element_terms)
    }

    fn serialized_name(&self, _container_values: &ContainerValues, _: Value) -> String {
        "set-of".to_owned()
    }
}
