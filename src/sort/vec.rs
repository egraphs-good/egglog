use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VecContainer<V> {
    do_rebuild: bool,
    pub data: Vec<V>,
}

impl Container for VecContainer<core_relations::Value> {
    fn rebuild_contents(&mut self, rebuilder: &dyn Rebuilder) -> bool {
        if self.do_rebuild {
            rebuilder.rebuild_slice(&mut self.data)
        } else {
            false
        }
    }
    fn iter(&self) -> impl Iterator<Item = core_relations::Value> + '_ {
        self.data.iter().copied()
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

    fn column_ty(&self, _backend: &egglog_bridge::EGraph) -> ColumnTy {
        ColumnTy::Id
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend.register_container_ty::<VecContainer<core_relations::Value>>();
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

    fn old_inner_values(&self, value: &Value) -> Vec<(ArcSort, Value)> {
        // TODO: Potential duplication of code
        let vecs = self.vecs.lock().unwrap();
        let vec = vecs.get_index(value.bits as usize).unwrap();
        vec.data.iter().map(|e| (self.element(), *e)).collect()
    }

    fn inner_values(
        &self,
        egraph: &EGraph,
        value: &core_relations::Value,
    ) -> Vec<(ArcSort, core_relations::Value)> {
        let val = egraph
            .backend
            .containers()
            .get_val::<VecContainer<core_relations::Value>>(*value)
            .unwrap()
            .clone();
        val.data
            .iter()
            .map(|e| (self.element.clone(), *e))
            .collect()
    }

    fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
        let vecs = self.vecs.lock().unwrap();
        let vec = vecs.get_index(value.bits as usize).unwrap();
        let mut changed = false;
        let new_vec = VecContainer {
            do_rebuild: vec.do_rebuild,
            data: vec
                .data
                .iter()
                .map(|e| {
                    let mut e = *e;
                    changed |= self.element.canonicalize(&mut e, unionfind);
                    e
                })
                .collect(),
        };
        drop(vecs);
        *value = new_vec.store(self);
        changed
    }

    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        add_primitive!(eg, "vec-empty"  = |                                       | -> @VecContainer<Value> (self.clone()) { VecContainer { do_rebuild: self.__y.is_eq_container_sort(), data: Vec::new()                        } });
        add_primitive!(eg, "vec-of"     = [xs: # (self.element())                 ] -> @VecContainer<Value> (self.clone()) { VecContainer { do_rebuild: self.__y.is_eq_container_sort(), data: xs                     .collect() } });
        add_primitive!(eg, "vec-append" = [xs: @VecContainer<Value> (self.clone())] -> @VecContainer<Value> (self.clone()) { VecContainer { do_rebuild: self.__y.is_eq_container_sort(), data: xs.flat_map(|x| x.data).collect() } });

        add_primitive!(eg, "vec-push" = |mut xs: @VecContainer<Value> (self.clone()), x: # (self.element())| -> @VecContainer<Value> (self.clone()) {{ xs.data.push(x); xs }});
        add_primitive!(eg, "vec-pop"  = |mut xs: @VecContainer<Value> (self.clone())                       | -> @VecContainer<Value> (self.clone()) {{ xs.data.pop();   xs }});

        add_primitive!(eg, "vec-length"       = |xs: @VecContainer<Value> (self.clone())| -> i64 { xs.data.len() as i64 });
        add_primitive!(eg, "vec-contains"     = |xs: @VecContainer<Value> (self.clone()), x: # (self.element())| -?> () { ( xs.data.contains(&x)).then_some(()) });
        add_primitive!(eg, "vec-not-contains" = |xs: @VecContainer<Value> (self.clone()), x: # (self.element())| -?> () { (!xs.data.contains(&x)).then_some(()) });

        add_primitive!(eg, "vec-get"    = |    xs: @VecContainer<Value> (self.clone()), i: i64                       | -?> # (self.element()) { xs.data.get(i as usize).copied() });
        add_primitive!(eg, "vec-set"    = |mut xs: @VecContainer<Value> (self.clone()), i: i64, x: # (self.element())| -> @VecContainer<Value> (self.clone()) {{ xs.data[i as usize] = x;    xs }});
        add_primitive!(eg, "vec-remove" = |mut xs: @VecContainer<Value> (self.clone()), i: i64                       | -> @VecContainer<Value> (self.clone()) {{ xs.data.remove(i as usize); xs }});
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

        if vec.data.is_empty() {
            Some((cost, termdag.app("vec-empty".into(), vec![])))
        } else {
            let elems = vec
                .data
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

    fn serialized_name(&self, _value: &core_relations::Value) -> Symbol {
        "vec-of".into()
    }

    fn value_type(&self) -> Option<TypeId> {
        Some(TypeId::of::<VecContainer<core_relations::Value>>())
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
