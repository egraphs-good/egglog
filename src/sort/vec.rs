use egglog_bridge::UnionAction;
use std::any::TypeId;
use std::iter::zip;

use super::*;

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
            "vec-union",
            "vec-range",
            "unstable-vec-map",
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
            .get_val::<VecContainer>(value)
            .unwrap()
            .clone();
        val.data
            .iter()
            .map(|e| (self.element.clone(), *e))
            .collect()
    }

    fn register_primitives(&self, eg: &mut EGraph) {
        let arc = self.clone().to_arcsort();

        add_primitive!(eg, "vec-empty"  = {self.clone(): VecSort} |                                | -> @VecContainer (arc) { VecContainer {
            do_rebuild: self.ctx.is_eq_container_sort(),
            data: Vec::new()
        } });
        add_primitive!(eg, "vec-of"     = {self.clone(): VecSort} [xs: # (self.element())          ] -> @VecContainer (arc) { VecContainer {
            do_rebuild: self.ctx.is_eq_container_sort(),
            data: xs                     .collect()
        } });
        add_primitive!(eg, "vec-append" = {self.clone(): VecSort} [xs: @VecContainer (arc)] -> @VecContainer (arc) { VecContainer {
            do_rebuild: self.ctx.is_eq_container_sort(),
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
        if self.element.is_eq_sort() {
            eg.add_primitive(Union {
                name: "vec-union".into(),
                vec: arc.clone(),
                action: eg.new_union_action(),
            });
        }
        // vec-range
        if self.element.name() == "i64" {
            add_primitive!(eg, "vec-range" = {self.clone(): VecSort} |end: i64| -> @VecContainer (arc) { VecContainer {
                do_rebuild: self.ctx.is_eq_container_sort(),
                data: {
                    let end: usize = end.try_into().unwrap_or(0);
                    (0..end)
                        .map(|i| exec_state.base_values().get::<i64>(i as i64))
                        .collect()
                }
            } });
        }
        // unstable-vec-map (fn Vec[A], (A -> B)) -> Vec[B]
        // For Map, support either defining Vec sort or Fn sort first
        let self_cloned = arc.clone();
        let element_name = self.element.name().to_string();

        // If we are registering a fn type A -> B and either and A == element or B == element,
        // Then find all Vec sorts with element type B or A respectively and register unstable-vec-map
        let all_vec_sorts = eg
            .type_info
            .get_arcsorts_by(|f| f.value_type() == Some(TypeId::of::<VecContainer>()));

        // Iterate through all function sorts and add any that match this element type
        // Then for each of those, register the unstable-vec-map primitive
        let register_map = Box::new(move |fn_: Arc<FunctionSort>, eg: &mut EGraph| {
            if fn_.inputs().len() != 1 {
                return;
            }
            let input_name = fn_.inputs()[0].name();
            let fn_output = fn_.output();
            let output_name = fn_output.name();

            //
            if input_name != element_name && output_name != element_name {
                return;
            }
            for some_vec_sort in &all_vec_sorts {
                let inner_sorts = some_vec_sort.inner_sorts();
                let some_vec_name = inner_sorts[0].name();
                let (input_vec, output_vec) =
                    if input_name == some_vec_name && output_name == element_name {
                        (some_vec_sort.clone(), self_cloned.clone())
                    } else if input_name == element_name && output_name == some_vec_name {
                        (self_cloned.clone(), some_vec_sort.clone())
                    } else {
                        continue;
                    };
                eg.add_primitive(VecMap {
                    name: "unstable-vec-map".into(),
                    vec: input_vec,
                    output_vec,
                    fn_: fn_.clone(),
                });
            }
        });

        for fn_sort in eg.type_info.get_sorts::<FunctionSort>() {
            register_map(fn_sort.clone(), eg);
        }
        let mut register = REGISTER_FN_PRIMITIVES.lock().unwrap();
        register.push(register_map);
    }

    fn reconstruct_termdag(
        &self,
        _container_values: &ContainerValues,
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

    fn serialized_name(&self, _container_values: &ContainerValues, _: Value) -> String {
        "vec-of".to_owned()
    }
}

// (unstable-vec-map (Vec[X], [X] -> Y) -> Vec[Y])
// will map the function over all elements in the vec and drop elements where it is undefined.
#[derive(Clone)]
struct VecMap {
    name: String,
    vec: ArcSort,
    output_vec: ArcSort,
    fn_: Arc<FunctionSort>,
}

impl Primitive for VecMap {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.fn_.clone(), self.vec.clone(), self.output_vec.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
        let fc = exec_state
            .container_values()
            .get_val::<FunctionContainer>(args[0])
            .unwrap()
            .clone();
        let vec = exec_state
            .container_values()
            .get_val::<VecContainer>(args[1])
            .unwrap()
            .clone();
        let mut new_data = Vec::with_capacity(vec.data.len());
        for v in vec.data {
            new_data.push(
                fc.apply(exec_state, &[v])
                    .expect("Vec map function returned None"),
            );
        }
        let vec = VecContainer {
            do_rebuild: self.output_vec.is_eq_container_sort(),
            data: new_data,
        };
        Some(
            exec_state
                .clone()
                .container_values()
                .register_val(vec, exec_state),
        )
    }
}

// (vec-union Vec[A] Vec[A]) -> Vec[A]
// where A: Eq
// Unions items from two vecs, asserting they are the same length.
#[derive(Clone)]
struct Union {
    name: String,
    vec: ArcSort,
    action: UnionAction,
}

impl Primitive for Union {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.vec.clone(), self.vec.clone(), self.vec.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
        let left = exec_state
            .container_values()
            .get_val::<VecContainer>(args[0])?
            .clone()
            .data;
        let right = exec_state
            .container_values()
            .get_val::<VecContainer>(args[1])?
            .clone()
            .data;
        if left.len() != right.len() {
            return None;
        }
        for (l, r) in zip(left, right) {
            self.action.union(exec_state, l, r);
        }
        Some(args[0])
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
