use super::*;

#[derive(Debug)]
pub struct VecSort {
    name: Symbol,
    element: ArcSort,
    vecs: Mutex<IndexSet<Vec<Value>>>,
}

impl VecSort {
    pub fn element(&self) -> ArcSort {
        self.element.clone()
    }

    pub fn element_name(&self) -> Symbol {
        self.element.name()
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
                .get_sort(e)
                .ok_or(TypeError::UndefinedSort(*e, span.clone()))?;

            if e.is_eq_container_sort() {
                return Err(TypeError::DisallowedSort(
                    name,
                    "Sets nested with other EqSort containers are not allowed".into(),
                    span.clone(),
                ));
            }

            Ok(Arc::new(Self {
                name,
                element: e.clone(),
                vecs: Default::default(),
            }))
        } else {
            panic!("Vec sort must have sort as argument. Got {:?}", args)
        }
    }
}

impl Sort for VecSort {
    fn name(&self) -> Symbol {
        self.name
    }

    fn column_ty(&self, prims: &Primitives) -> ColumnTy {
        ColumnTy::Primitive(prims.get_ty::<Vec<core_relations::Value>>())
    }

    fn register_type(&self, prims: &mut Primitives) {
        prims.register_type::<Vec<core_relations::Value>>();
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn is_container_sort(&self) -> bool {
        true
    }

    fn is_eq_container_sort(&self) -> bool {
        self.element.is_eq_sort()
    }

    fn inner_values(&self, value: &Value) -> Vec<(ArcSort, Value)> {
        // TODO: Potential duplication of code
        let vecs = self.vecs.lock().unwrap();
        let vec = vecs.get_index(value.bits as usize).unwrap();
        let mut result = Vec::new();
        for e in vec.iter() {
            result.push((self.element.clone(), *e));
        }
        result
    }

    fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
        let vecs = self.vecs.lock().unwrap();
        let vec = vecs.get_index(value.bits as usize).unwrap();
        let mut changed = false;
        let new_vec: Vec<_> = vec
            .iter()
            .map(|e| {
                let mut e = *e;
                changed |= self.element.canonicalize(&mut e, unionfind);
                e
            })
            .collect();
        drop(vecs);
        *value = new_vec.store(self);
        changed
    }

    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        eg.add_primitive(Primitive(
            Arc::new(VecRebuild {
                name: "rebuild".into(),
                vec: self.clone(),
            }),
            ExternalFunctionId::new_const(u32::MAX),
        ));

        add_primitive!(eg, "vec-empty"  = |                             | -> Vec<Value> (self.clone()) { Vec::new()             });
        add_primitive!(eg, "vec-of"     = [xs: # (self.element())       ] -> Vec<Value> (self.clone()) { xs.collect()           });
        add_primitive!(eg, "vec-append" = [xs: Vec<Value> (self.clone())] -> Vec<Value> (self.clone()) { xs.flatten().collect() });

        add_primitive!(eg, "vec-push" = |mut xs: Vec<Value> (self.clone()), x: # (self.element())| -> Vec<Value> (self.clone()) {{ xs.push(x); xs }});
        add_primitive!(eg, "vec-pop"  = |mut xs: Vec<Value> (self.clone())                       | -> Vec<Value> (self.clone()) {{ xs.pop();   xs }});

        add_primitive!(eg, "vec-length" = |xs: Vec<Value> (self.clone())| -> i64 { xs.len() as i64 });
        add_primitive!(eg, "vec-contains"     = |xs: Vec<Value> (self.clone()), x: # (self.element())| -?> () { ( xs.contains(&x)).then_some(()) });
        add_primitive!(eg, "vec-not-contains" = |xs: Vec<Value> (self.clone()), x: # (self.element())| -?> () { (!xs.contains(&x)).then_some(()) });

        add_primitive!(eg, "vec-get"    = |    xs: Vec<Value> (self.clone()), i: i64                       | -?> # (self.element()) { xs.get(i as usize).copied() });
        add_primitive!(eg, "vec-set"    = |mut xs: Vec<Value> (self.clone()), i: i64, x: # (self.element())| -> Vec<Value> (self.clone()) {{ xs[i as usize] = x;    xs }});
        add_primitive!(eg, "vec-remove" = |mut xs: Vec<Value> (self.clone()), i: i64                       | -> Vec<Value> (self.clone()) {{ xs.remove(i as usize); xs }});
    }

    fn extract_term(
        &self,
        _egraph: &EGraph,
        value: Value,
        extractor: &Extractor,
        termdag: &mut TermDag,
    ) -> Option<(Cost, Term)> {
        let vec = Vec::load(self, &value);
        let mut cost = 0usize;

        if vec.is_empty() {
            Some((cost, termdag.app("vec-empty".into(), vec![])))
        } else {
            let elems = vec
                .into_iter()
                .map(|e| {
                    let (extra_cost, term) = extractor.find_best(e, termdag, &self.element)?;
                    cost = cost.saturating_add(extra_cost);
                    Some(term)
                })
                .collect::<Option<Vec<_>>>()?;

            Some((cost, termdag.app("vec-of".into(), elems)))
        }
    }

    fn serialized_name(&self, _value: &Value) -> Symbol {
        "vec-of".into()
    }
}

impl IntoSort for Vec<Value> {
    type Sort = VecSort;
    fn store(self, sort: &Self::Sort) -> Value {
        let mut vecs = sort.vecs.lock().unwrap();
        let (i, _) = vecs.insert_full(self);
        Value {
            #[cfg(debug_assertions)]
            tag: sort.name,
            bits: i as u64,
        }
    }
}

impl FromSort for Vec<Value> {
    type Sort = VecSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let vecs = sort.vecs.lock().unwrap();
        vecs.get_index(value.bits as usize).unwrap().clone()
    }
}

// TODO: is this used anywhere?
struct VecRebuild {
    name: Symbol,
    vec: Arc<VecSort>,
}

impl PrimitiveLike for VecRebuild {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.vec.clone(), self.vec.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(
        &self,
        values: &[Value],
        _sorts: (&[ArcSort], &ArcSort),
        egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
        let egraph = egraph.unwrap();
        let vec = Vec::load(&self.vec, &values[0]);
        let new_vec: Vec<Value> = vec
            .iter()
            .map(|e| egraph.find(&self.vec.element, *e))
            .collect();
        drop(vec);
        Some(new_vec.store(&self.vec))
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
