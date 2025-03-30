use super::*;
use inner::MultiSet;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct MultiSetContainer<V: Clone + Hash + Ord>(MultiSet<V>);

impl Container for MultiSetContainer<core_relations::Value> {
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
pub struct MultiSetSort {
    name: Symbol,
    element: ArcSort,
    multisets: Mutex<IndexSet<MultiSetContainer<Value>>>,
}

impl MultiSetSort {
    pub fn element(&self) -> ArcSort {
        self.element.clone()
    }
}

impl Presort for MultiSetSort {
    fn presort_name() -> Symbol {
        "MultiSet".into()
    }

    fn reserved_primitives() -> Vec<Symbol> {
        vec![
            "multiset-of".into(),
            "multiset-insert".into(),
            "multiset-contains".into(),
            "multiset-not-contains".into(),
            "multiset-remove".into(),
            "multiset-length".into(),
            "multiset-sum".into(),
            "unstable-multiset-map".into(),
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
                    "Multisets nested with other EqSort containers are not allowed".into(),
                    span.clone(),
                ));
            }

            Ok(Arc::new(Self {
                name,
                element: e.clone(),
                multisets: Default::default(),
            }))
        } else {
            panic!()
        }
    }
}

impl Sort for MultiSetSort {
    fn name(&self) -> Symbol {
        self.name
    }

    fn column_ty(&self, _backend: &egglog_bridge::EGraph) -> ColumnTy {
        ColumnTy::Id
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend.register_container_ty::<MultiSetContainer<core_relations::Value>>();
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
        let multisets = self.multisets.lock().unwrap();
        let multiset = multisets.get_index(value.bits as usize).unwrap();
        multiset
            .0
            .iter()
            .map(|k| (self.element.clone(), *k))
            .collect()
    }

    fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
        let multisets = self.multisets.lock().unwrap();
        let multiset = multisets.get_index(value.bits as usize).unwrap().clone();
        let mut changed = false;
        let new_multiset = MultiSetContainer(multiset.0.map(|e| {
            let mut e = *e;
            changed |= self.element.canonicalize(&mut e, unionfind);
            e
        }));
        drop(multisets);
        *value = new_multiset.store(self);
        changed
    }

    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        add_primitive!(eg, "multiset-of" = [xs: # (self.element())] -> @MultiSetContainer<Value> (self.clone()) { MultiSetContainer(xs.collect()) });

        add_primitive!(eg, "multiset-pick" = |xs: @MultiSetContainer<Value> (self.clone())| -> # (self.element()) { *xs.0.pick().expect("Cannot pick from an empty multiset") });
        add_primitive!(eg, "multiset-insert" = |mut xs: @MultiSetContainer<Value> (self.clone()), x: # (self.element())| -> @MultiSetContainer<Value> (self.clone()) { MultiSetContainer(xs.0.insert( x) ) });
        add_primitive!(eg, "multiset-remove" = |mut xs: @MultiSetContainer<Value> (self.clone()), x: # (self.element())| -> @MultiSetContainer<Value> (self.clone()) { MultiSetContainer(xs.0.remove(&x)?) });

        add_primitive!(eg, "multiset-length"       = |xs: @MultiSetContainer<Value> (self.clone())| -> i64 { xs.0.len() as i64 });
        add_primitive!(eg, "multiset-contains"     = |xs: @MultiSetContainer<Value> (self.clone()), x: # (self.element())| -?> () { ( xs.0.contains(&x)).then_some(()) });
        add_primitive!(eg, "multiset-not-contains" = |xs: @MultiSetContainer<Value> (self.clone()), x: # (self.element())| -?> () { (!xs.0.contains(&x)).then_some(()) });

        add_primitive!(eg, "multiset-sum" = |xs: @MultiSetContainer<Value> (self.clone()), ys: @MultiSetContainer<Value> (self.clone())| -> @MultiSetContainer<Value> (self.clone()) { MultiSetContainer(xs.0.sum(ys.0)) });

        // let inner_name = self.element.name();
        // let fn_sort = eg.type_info.get_sort_by(|s: &Arc<FunctionSort>| {
        //     (s.output.name() == inner_name)
        //         && s.inputs.len() == 1
        //         && (s.inputs[0].name() == inner_name)
        // });
        // // Only include map function if we already declared a function sort with the correct signature
        // if let Some(fn_sort) = fn_sort {
        // 	// add_primitive!(eg, "unstable-multiset-map" = {});
        // }
    }

    fn extract_term(
        &self,
        _egraph: &EGraph,
        value: Value,
        extractor: &Extractor,
        termdag: &mut TermDag,
    ) -> Option<(Cost, Term)> {
        let multiset = MultiSetContainer::load(self, &value);
        let mut children = vec![];
        let mut cost = 0usize;
        for e in multiset.0.iter() {
            let (child_cost, child_term) = extractor.find_best(*e, termdag, &self.element)?;
            cost = cost.saturating_add(child_cost);
            children.push(child_term);
        }
        Some((cost, termdag.app("multiset-of".into(), children)))
    }

    fn serialized_name(&self, _value: &Value) -> Symbol {
        "multiset-of".into()
    }
}

impl IntoSort for MultiSetContainer<Value> {
    type Sort = MultiSetSort;
    fn store(self, sort: &Self::Sort) -> Value {
        let mut multisets = sort.multisets.lock().unwrap();
        let (i, _) = multisets.insert_full(self);
        Value {
            #[cfg(debug_assertions)]
            tag: sort.name,
            bits: i as u64,
        }
    }
}

impl FromSort for MultiSetContainer<Value> {
    type Sort = MultiSetSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let sets = sort.multisets.lock().unwrap();
        sets.get_index(value.bits as usize).unwrap().clone()
    }
}

// struct Map {
//     name: Symbol,
//     multiset: Arc<MultiSetSort>,
//     fn_: Arc<FunctionSort>,
// }

// impl PrimitiveLike for Map {
//     fn name(&self) -> Symbol {
//         self.name
//     }

//     fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
//         SimpleTypeConstraint::new(
//             self.name(),
//             vec![
//                 self.fn_.clone(),
//                 self.multiset.clone(),
//                 self.multiset.clone(),
//             ],
//             span.clone(),
//         )
//         .into_box()
//     }

//     fn apply(
//         &self,
//         values: &[Value],
//         _sorts: (&[ArcSort], &ArcSort),
//         egraph: Option<&mut EGraph>,
//     ) -> Option<Value> {
//         let egraph =
//             egraph.unwrap_or_else(|| panic!("`{}` is not supported yet in facts.", self.name));
//         let multiset = ValueMultiSet::load(&self.multiset, &values[1]);
//         let new_multiset = multiset.map(|e| self.fn_.apply(&values[0], &[*e], egraph));
//         Some(new_multiset.store(&self.multiset))
//     }
// }

// Place multiset in its own module to keep implementation details private from sort
mod inner {
    use im::OrdMap;
    use std::hash::Hash;
    /// Immutable multiset implementation, which is threadsafe and hash stable, regardless of insertion order.
    ///
    /// All methods that return a new multiset take ownership of the old multiset.
    #[derive(Debug, Hash, Eq, PartialEq, Clone)]
    pub(crate) struct MultiSet<T: Clone + Hash + Ord>(
        /// All values should be > 0
        OrdMap<T, usize>,
        /// cached length
        usize,
    );

    impl<T: Clone + Hash + Ord> MultiSet<T> {
        /// Create a new empty multiset.
        pub(crate) fn new() -> Self {
            MultiSet(OrdMap::new(), 0)
        }

        /// Check if the multiset contains a key.
        pub(crate) fn contains(&self, value: &T) -> bool {
            self.0.contains_key(value)
        }

        /// Return the total number of elements in the multiset.
        pub(crate) fn len(&self) -> usize {
            self.1
        }

        /// Return an iterator over all elements in the multiset.
        pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
            self.0
                .iter()
                .flat_map(|(k, v)| std::iter::repeat(k).take(*v))
        }

        /// Return an arbitrary element from the multiset.
        pub(crate) fn pick(&self) -> Option<&T> {
            self.0.keys().next()
        }

        /// Map a function over all elements in the multiset, taking ownership of it and returning a new multiset.
        pub(crate) fn map(self, mut f: impl FnMut(&T) -> T) -> MultiSet<T> {
            let mut new = MultiSet::new();
            for (k, v) in self.0.into_iter() {
                new.insert_multiple_mut(f(&k), v);
            }
            new
        }

        /// Insert a value into the multiset, taking ownership of it and returning a new multiset.
        pub(crate) fn insert(mut self, value: T) -> MultiSet<T> {
            self.insert_multiple_mut(value, 1);
            self
        }

        /// Remove a value from the multiset, taking ownership of it and returning a new multiset.
        pub(crate) fn remove(mut self, value: &T) -> Option<MultiSet<T>> {
            if let Some(v) = self.0.get(value) {
                self.1 -= 1;
                if *v == 1 {
                    self.0.remove(value);
                } else {
                    self.0.insert(value.clone(), v - 1);
                }
                Some(self)
            } else {
                None
            }
        }

        fn insert_multiple_mut(&mut self, value: T, n: usize) {
            self.1 += n;
            if let Some(v) = self.0.get(&value) {
                self.0.insert(value, v + n);
            } else {
                self.0.insert(value, n);
            }
        }

        /// Compute the sum of two multisets.
        pub fn sum(self, MultiSet(other_map, other_count): Self) -> Self {
            Self(
                self.0.union_with(other_map, std::ops::Add::add),
                self.1 + other_count,
            )
        }
    }

    impl<T: Clone + Hash + Ord> FromIterator<T> for MultiSet<T> {
        fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
            let mut multiset = MultiSet::new();
            for value in iter {
                multiset.insert_multiple_mut(value, 1);
            }
            multiset
        }
    }
}
