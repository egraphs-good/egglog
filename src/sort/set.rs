use super::*;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SetContainer<V>(BTreeSet<V>);

impl Container for SetContainer<core_relations::Value> {
    fn rebuild_contents(&mut self, rebuilder: &dyn Rebuilder) -> bool {
        let mut xs: Vec<_> = self.0.iter().copied().collect();
        let changed = rebuilder.rebuild_slice(&mut xs);
        self.0 = xs.into_iter().collect();
        changed
    }
    fn iter(&self) -> impl Iterator<Item = core_relations::Value> + '_ {
        self.0.iter().copied()
    }
}

#[derive(Debug)]
pub struct SetSort {
    name: Symbol,
    element: ArcSort,
    sets: Mutex<IndexSet<SetContainer<Value>>>,
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
                sets: Default::default(),
            }))
        } else {
            panic!()
        }
    }
}

impl Sort for SetSort {
    fn name(&self) -> Symbol {
        self.name
    }

    fn column_ty(&self, _backend: &egglog_bridge::EGraph) -> ColumnTy {
        ColumnTy::Id
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend.register_container_ty::<SetContainer<core_relations::Value>>();
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
        let sets = self.sets.lock().unwrap();
        let set = sets.get_index(value.bits as usize).unwrap();
        let mut result = Vec::new();
        for e in set.0.iter() {
            result.push((self.element.clone(), *e));
        }
        result
    }

    fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
        let sets = self.sets.lock().unwrap();
        let set = sets.get_index(value.bits as usize).unwrap();
        let mut changed = false;
        let new_set = SetContainer(
            set.0
                .iter()
                .map(|e| {
                    let mut e = *e;
                    changed |= self.element.canonicalize(&mut e, unionfind);
                    e
                })
                .collect(),
        );
        drop(sets);
        *value = new_set.store(self);
        changed
    }

    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        add_primitive!(eg, "set-empty" = |                      | -> @SetContainer<Value> (self.clone()) { SetContainer(BTreeSet::new()) });
        add_primitive!(eg, "set-of"    = [xs: # (self.element())] -> @SetContainer<Value> (self.clone()) { SetContainer(xs.collect()   ) });

        add_primitive!(eg, "set-insert" = |mut xs: @SetContainer<Value> (self.clone()), x: # (self.element())| -> @SetContainer<Value> (self.clone()) {{ xs.0.insert( x); xs }});
        add_primitive!(eg, "set-remove" = |mut xs: @SetContainer<Value> (self.clone()), x: # (self.element())| -> @SetContainer<Value> (self.clone()) {{ xs.0.remove(&x); xs }});

        add_primitive!(eg, "set-length"       = |xs: @SetContainer<Value> (self.clone())| -> i64 { xs.0.len() as i64 });
        add_primitive!(eg, "set-contains"     = |xs: @SetContainer<Value> (self.clone()), x: # (self.element())| -?> () { ( xs.0.contains(&x)).then_some(()) });
        add_primitive!(eg, "set-not-contains" = |xs: @SetContainer<Value> (self.clone()), x: # (self.element())| -?> () { (!xs.0.contains(&x)).then_some(()) });

        add_primitive!(eg, "set-get" = |xs: @SetContainer<Value> (self.clone()), i: i64| -?> # (self.element()) { xs.0.iter().nth(i as usize).copied() });

        add_primitive!(eg, "set-union"      = |mut xs: @SetContainer<Value> (self.clone()), ys: @SetContainer<Value> (self.clone())| -> @SetContainer<Value> (self.clone()) {{ xs.0.extend(ys.0);                  xs }});
        add_primitive!(eg, "set-diff"       = |mut xs: @SetContainer<Value> (self.clone()), ys: @SetContainer<Value> (self.clone())| -> @SetContainer<Value> (self.clone()) {{ xs.0.retain(|k| !ys.0.contains(k)); xs }});
        add_primitive!(eg, "set-intersect"  = |mut xs: @SetContainer<Value> (self.clone()), ys: @SetContainer<Value> (self.clone())| -> @SetContainer<Value> (self.clone()) {{ xs.0.retain(|k|  ys.0.contains(k)); xs }});
    }

    fn extract_term(
        &self,
        _egraph: &EGraph,
        value: Value,
        extractor: &Extractor,
        termdag: &mut TermDag,
    ) -> Option<(Cost, Term)> {
        let set = SetContainer::load(self, &value);
        let mut children = vec![];
        let mut cost = 0usize;
        for e in set.0.iter() {
            let (child_cost, child_term) = extractor.find_best(*e, termdag, &self.element)?;
            cost = cost.saturating_add(child_cost);
            children.push(child_term);
        }
        Some((cost, termdag.app("set-of".into(), children)))
    }

    fn serialized_name(&self, _value: &Value) -> Symbol {
        "set-of".into()
    }
}

impl IntoSort for SetContainer<Value> {
    type Sort = SetSort;
    fn store(self, sort: &Self::Sort) -> Value {
        let mut sets = sort.sets.lock().unwrap();
        let (i, _) = sets.insert_full(self);
        Value {
            #[cfg(debug_assertions)]
            tag: sort.name,
            bits: i as u64,
        }
    }
}

impl FromSort for SetContainer<Value> {
    type Sort = SetSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let sets = sort.sets.lock().unwrap();
        sets.get_index(value.bits as usize).unwrap().clone()
    }
}
