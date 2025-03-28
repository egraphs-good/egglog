use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct VecContainer<V>(Vec<V>);

impl Container for VecContainer<core_relations::Value> {
    fn rebuild_contents(&mut self, rebuilder: &dyn Rebuilder) -> bool {
        rebuilder.rebuild_slice(&mut self.0)
    }
    fn iter(&self) -> impl Iterator<Item = core_relations::Value> + '_ {
        self.0.iter().copied()
    }
}

#[derive(Debug)]
pub struct VecSort {
    name: Symbol,
    element: ArcSort,
    vecs: Mutex<IndexSet<VecContainer<Value>>>,
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
        ColumnTy::Primitive(prims.get_ty::<VecContainer<core_relations::Value>>())
    }

    fn register_type(&self, prims: &mut Primitives) {
        prims.register_type::<VecContainer<core_relations::Value>>();
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
        vec.0.iter().map(|e| (self.element(), *e)).collect()
    }

    fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
        let vecs = self.vecs.lock().unwrap();
        let vec = vecs.get_index(value.bits as usize).unwrap();
        let mut changed = false;
        let new_vec = VecContainer(
            vec.0
                .iter()
                .map(|e| {
                    let mut e = *e;
                    changed |= self.element.canonicalize(&mut e, unionfind);
                    e
                })
                .collect(),
        );
        drop(vecs);
        *value = new_vec.store(self);
        changed
    }

    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        add_primitive!(eg, "vec-empty"  = |                                      | -> VecContainer<Value> (self.clone()) { VecContainer(Vec::new()                    ) });
        add_primitive!(eg, "vec-of"     = [xs: # (self.element())                ] -> VecContainer<Value> (self.clone()) { VecContainer(xs                  .collect()) });
        add_primitive!(eg, "vec-append" = [xs: VecContainer<Value> (self.clone())] -> VecContainer<Value> (self.clone()) { VecContainer(xs.flat_map(|x| x.0).collect()) });

        add_primitive!(eg, "vec-push" = |mut xs: VecContainer<Value> (self.clone()), x: # (self.element())| -> VecContainer<Value> (self.clone()) {{ xs.0.push(x); xs }});
        add_primitive!(eg, "vec-pop"  = |mut xs: VecContainer<Value> (self.clone())                       | -> VecContainer<Value> (self.clone()) {{ xs.0.pop();   xs }});

        add_primitive!(eg, "vec-length"       = |xs: VecContainer<Value> (self.clone())| -> i64 { xs.0.len() as i64 });
        add_primitive!(eg, "vec-contains"     = |xs: VecContainer<Value> (self.clone()), x: # (self.element())| -?> () { ( xs.0.contains(&x)).then_some(()) });
        add_primitive!(eg, "vec-not-contains" = |xs: VecContainer<Value> (self.clone()), x: # (self.element())| -?> () { (!xs.0.contains(&x)).then_some(()) });

        add_primitive!(eg, "vec-get"    = |    xs: VecContainer<Value> (self.clone()), i: i64                       | -?> # (self.element()) { xs.0.get(i as usize).copied() });
        add_primitive!(eg, "vec-set"    = |mut xs: VecContainer<Value> (self.clone()), i: i64, x: # (self.element())| -> VecContainer<Value> (self.clone()) {{ xs.0[i as usize] = x;    xs }});
        add_primitive!(eg, "vec-remove" = |mut xs: VecContainer<Value> (self.clone()), i: i64                       | -> VecContainer<Value> (self.clone()) {{ xs.0.remove(i as usize); xs }});
    }

    fn extract_term(
        &self,
        _egraph: &EGraph,
        value: Value,
        extractor: &Extractor,
        termdag: &mut TermDag,
    ) -> Option<(Cost, Term)> {
        let vec = VecContainer::load(self, &value);
        let mut cost = 0usize;

        if vec.0.is_empty() {
            Some((cost, termdag.app("vec-empty".into(), vec![])))
        } else {
            let elems = vec
                .0
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

impl IntoSort for VecContainer<Value> {
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

impl FromSort for VecContainer<Value> {
    type Sort = VecSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let vecs = sort.vecs.lock().unwrap();
        vecs.get_index(value.bits as usize).unwrap().clone()
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
