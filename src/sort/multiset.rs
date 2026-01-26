use super::*;
use inner::MultiSet;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MultiSetContainer {
    pub do_rebuild: bool,
    pub data: MultiSet<Value>,
}

impl ContainerValue for MultiSetContainer {
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
pub struct MultiSetSort {
    name: String,
    element: ArcSort,
}

impl MultiSetSort {
    pub fn element(&self) -> ArcSort {
        self.element.clone()
    }
}

impl Presort for MultiSetSort {
    fn presort_name() -> &'static str {
        "MultiSet"
    }

    fn reserved_primitives() -> Vec<&'static str> {
        vec![
            "multiset-of",
            "multiset-insert",
            "multiset-contains",
            "multiset-not-contains",
            "multiset-remove",
            "multiset-length",
            "multiset-sum",
            "unstable-multiset-map",
            "unstable-multiset-fill-index",
            "unstable-multiset-clear-index",
            "unstable-multiset-flat-map",
            "unstable-multiset-fold",
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
            panic!()
        }
    }
}

impl ContainerSort for MultiSetSort {
    type Container = MultiSetContainer;

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
            .get_val::<MultiSetContainer>(value)
            .unwrap()
            .clone();
        val.data
            .iter()
            .map(|k| (self.element.clone(), *k))
            .collect()
    }

    fn register_primitives(&self, eg: &mut EGraph) {
        let arc = self.clone().to_arcsort();

        add_primitive!(eg, "multiset-of" = {self.clone(): MultiSetSort} [xs: # (self.element())] -> @MultiSetContainer (arc) { MultiSetContainer {
            do_rebuild: self.ctx.element.is_eq_sort() || self.ctx.element.is_eq_container_sort(),
            data: xs.collect()
        } });

        add_primitive!(eg, "multiset-pick" = |xs: @MultiSetContainer (arc)| -> # (self.element()) { *xs.data.pick().expect("Cannot pick from an empty multiset") });
        add_primitive!(eg, "multiset-insert" = |mut xs: @MultiSetContainer (arc), x: # (self.element())| -> @MultiSetContainer (arc) { MultiSetContainer { data: xs.data.insert( x) , ..xs } });
        add_primitive!(eg, "multiset-remove" = |mut xs: @MultiSetContainer (arc), x: # (self.element())| -?> @MultiSetContainer (arc) { Some(MultiSetContainer { data: xs.data.remove(&x)?, ..xs } )});
        add_primitive!(eg, "multiset-remove-swapped" = |x: # (self.element()), mut xs: @MultiSetContainer (arc)| -?> @MultiSetContainer (arc) { Some(MultiSetContainer { data: xs.data.remove(&x)?, ..xs }) });
        add_primitive!(eg, "multiset-length"       = |xs: @MultiSetContainer (arc)| -> i64 { xs.data.len() as i64 });
        add_primitive!(eg, "multiset-contains"     = |xs: @MultiSetContainer (arc), x: # (self.element())| -?> () { ( xs.data.contains(&x)).then_some(()) });
        add_primitive!(eg, "multiset-not-contains" = |xs: @MultiSetContainer (arc), x: # (self.element())| -?> () { (!xs.data.contains(&x)).then_some(()) });
        add_primitive!(eg, "multiset-not-contains-swapped" = |x: # (self.element()), xs: @MultiSetContainer (arc)| -?> () { (!xs.data.contains(&x)).then_some(()) });
        add_primitive!(eg, "multiset-sum" = |xs: @MultiSetContainer (arc), ys: @MultiSetContainer (arc)| -> @MultiSetContainer (arc) { MultiSetContainer { data: xs.data.sum(ys.data), ..xs } });
        // Set counts to one
        add_primitive!(eg, "multiset-reset-counts" = |mut xs: @MultiSetContainer (arc)| -> @MultiSetContainer (arc) { {
            let mut new_data = MultiSet::<Value>::new();
            for (v, _) in xs.data.iter_counts() {
                new_data.insert_multiple_mut(v, 1);
            }
            MultiSetContainer { data: new_data, ..xs }
        }});
        add_primitive!(eg, "multiset-pick-max" = |xs: @MultiSetContainer (arc)| -?>  # (self.element()) {
            Some(xs.data.iter_counts().max_by_key(|(_, c)| *c)?.0)
        });
        add_primitive!(eg, "multiset-count" = |xs: @MultiSetContainer (arc), x: # (self.element())| -> i64 {
            xs.data.iter_counts().find(|(v, _)| *v == x).map(|(_, c)| c as i64).unwrap_or(0)
        });

        // Add multiset-sum-multisets if the inner arcsort is also a multiset
        for other_multiset_sort in eg.type_info.get_arcsorts_by(|f| {
            f.name() == self.element.name()
            // We can't query directly by arcsort type since it's wrapped in a ContainerSort which is not public
                && f.value_type() == Some(TypeId::of::<MultiSetContainer>())
        }) {
            eg.add_primitive(SumMultisets {
                name: "multiset-sum-multisets".into(),
                multiset: other_multiset_sort.clone(),
                multiset_of_multisets: arc.clone(),
                do_rebuild: other_multiset_sort.is_eq_container_sort(),
            });
        }

        // For Map, support either defining MultiSet sort or Fn sort first
        // Add map from MS[V] to MS[K]
        let self_cloned = arc.clone();
        let name = self.name().to_string();

        let register_map = Box::new(move |fn_: Arc<FunctionSort>, eg: &mut EGraph| {
            let all_other_multiset_sorts = eg.type_info.get_arcsorts_by(|f| {
                f.name() != name && f.value_type() == Some(TypeId::of::<MultiSetContainer>())
            });

            // All pairs of multisets to add a map for, all other multisets mapping to this one, this one mapping to itself,
            // and this one mapping to all other multisets
            let mut all_multiset_pairs_add = all_other_multiset_sorts
                .iter()
                .cloned()
                .map(|other_ms_sort| (other_ms_sort.clone(), self_cloned.clone()))
                .collect::<Vec<_>>();
            all_multiset_pairs_add.extend(
                all_other_multiset_sorts
                    .iter()
                    .cloned()
                    .map(|other_ms_sort| (self_cloned.clone(), other_ms_sort.clone())),
            );
            all_multiset_pairs_add.push((self_cloned.clone(), self_cloned.clone()));

            for (source_ms_sort, output_ms_sort) in &all_multiset_pairs_add {
                let source_element_sort = source_ms_sort.inner_sorts()[0].clone();
                let output_element_sort = output_ms_sort.inner_sorts()[0].clone();
                // If this function is from E -> K where E is inner_name and K is other_ms_sort's inner name
                if fn_.inputs().len() == 1
                    && fn_.inputs()[0].name() == source_element_sort.name()
                    && fn_.output().name() == output_element_sort.name()
                {
                    eg.add_primitive(Map {
                        name: "unstable-multiset-map".into(),
                        multiset: source_ms_sort.clone(),
                        output_multiset: output_ms_sort.clone(),
                        fn_: fn_.clone(),
                    });
                }
            }
        });
        let inner_name = self.element.name().to_string();
        let inner_name_cloned = inner_name.clone();
        let self_cloned = arc.clone();
        // For filter same as map
        let register_filter = Box::new(move |fn_: Arc<FunctionSort>, eg: &mut EGraph| {
            // Add filter if we have a function from E -> Unit
            if fn_.inputs().len() == 1
                && fn_.inputs()[0].name() == inner_name_cloned.clone()
                && fn_.output().name() == "Unit"
            {
                eg.add_primitive(Filter {
                    name: "unstable-multiset-filter".into(),
                    multiset: self_cloned.clone(),
                    fn_: fn_.clone(),
                });
            }
        });
        let element_clone = self.element.clone();
        let inner_name_cloned = inner_name.clone();
        let self_cloned = arc.clone();
        // fold is ((T, T) -> T, T, MultiSet[T]) -> T
        let register_fold = Box::new(move |fn_: Arc<FunctionSort>, eg: &mut EGraph| {
            if fn_.inputs().len() == 2
                && fn_.inputs()[0].name() == inner_name_cloned.clone()
                && fn_.inputs()[1].name() == inner_name_cloned.clone()
                && fn_.output().name() == inner_name_cloned.clone()
            {
                eg.add_primitive(Fold {
                    name: "unstable-multiset-fold".into(),
                    multiset: self_cloned.clone(),
                    fn_: fn_.clone(),
                    element: element_clone.clone(),
                });
            }
        });

        for fn_sort in eg.type_info.get_sorts::<FunctionSort>() {
            register_map(fn_sort.clone(), eg);
            register_filter(fn_sort.clone(), eg);
            register_fold(fn_sort.clone(), eg);
        }

        let mut register = REGISTER_FN_PRIMITIVES.lock().unwrap();
        register.push(register_map);
        register.push(register_filter);
        register.push(register_fold);
        // For FillIndex, you have to define the multiset sort first since the function sort depends on it
        register.push(Box::new(move |fn_: Arc<FunctionSort>, eg: &mut EGraph| {
            // add fill-index and clear-index if we have a function from (MultiSet[E], E) -> I64
            if fn_.inputs().len() == 2
                && fn_.inputs()[0].name() == arc.name()
                && fn_.inputs()[1].name() == inner_name
                && fn_.output().name() == "i64"
            {
                eg.add_primitive(FillIndex {
                    name: "unstable-multiset-fill-index".into(),
                    multiset: arc.clone(),
                    unit: eg.type_info.get_sort_by_name("Unit").unwrap().clone(),
                    fn_: fn_.clone(),
                });
                eg.add_primitive(ClearIndex {
                    name: "unstable-multiset-clear-index".into(),
                    multiset: arc.clone(),
                    unit: eg.type_info.get_sort_by_name("Unit").unwrap().clone(),
                    fn_: fn_.clone(),
                });
            }
            // Add flat-map if we have a function from E -> MultiSet[E]
            if fn_.inputs().len() == 1
                && fn_.inputs()[0].name() == inner_name
                && fn_.output().name() == arc.name()
            {
                eg.add_primitive(FlatMap {
                    name: "unstable-multiset-flat-map".into(),
                    multiset: arc.clone(),
                    fn_: fn_.clone(),
                });
            }
        }));
    }

    fn reconstruct_termdag(
        &self,
        _container_values: &ContainerValues,
        _value: Value,
        termdag: &mut TermDag,
        element_terms: Vec<Term>,
    ) -> Term {
        termdag.app("multiset-of".into(), element_terms)
    }

    fn serialized_name(&self, _container_values: &ContainerValues, _: Value) -> String {
        "multiset-of".to_owned()
    }
}

#[derive(Clone)]
struct Map {
    name: String,
    multiset: ArcSort,
    fn_: Arc<FunctionSort>,
    output_multiset: ArcSort,
}

impl Primitive for Map {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![
                self.fn_.clone(),
                self.multiset.clone(),
                self.output_multiset.clone(),
            ],
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
        let multiset = exec_state
            .container_values()
            .get_val::<MultiSetContainer>(args[1])
            .unwrap()
            .clone();
        let mut new_data = MultiSet::<Value>::new();
        // Filter out any elements which do not have the function defined for them
        for (v, c) in multiset.data.iter_counts() {
            let mapped = fc.apply(exec_state, &[v]);
            if let Some(mapped_v) = mapped {
                new_data.insert_multiple_mut(mapped_v, c);
            }
        }
        let multiset = MultiSetContainer {
            data: new_data,
            ..multiset
        };
        Some(
            exec_state
                .clone()
                .container_values()
                .register_val(multiset, exec_state),
        )
    }
}

// (unstable-multiset-fill-index ms: MultiSet[X] index_fn: [MultiSet[X], X] -> i64) -> Unit
// will set the index function for all elements in the multiset
#[derive(Clone)]
struct FillIndex {
    name: String,
    multiset: ArcSort,
    unit: ArcSort,
    fn_: Arc<FunctionSort>,
}

impl Primitive for FillIndex {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.multiset.clone(), self.fn_.clone(), self.unit.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
        let fc = exec_state
            .container_values()
            .get_val::<FunctionContainer>(args[1])
            .unwrap()
            .clone();
        let multiset = exec_state
            .container_values()
            .get_val::<MultiSetContainer>(args[0])
            .unwrap()
            .clone();
        let ResolvedFunctionId::Lookup(mut action) = fc.0 else {
            panic!(
                "Primitive functions cannot be used with unstable-multiset-fill-index, since they cannot be set"
            );
        };
        for (v, c) in multiset.data.iter_counts() {
            let mut row = vec![args[0].clone(), v];
            // If we have already filled this multiset once, skip since it should still be accurate with the right
            // merge function
            if action.lookup(exec_state, &row).is_some() {
                break;
            }
            row.push(exec_state.base_values().get::<i64>(c.try_into().unwrap()));
            action.insert(exec_state, row.into_iter());
        }
        Some(exec_state.base_values().get::<()>(()))
    }
}

// (unstable-multiset-clear-index ms: MultiSet[X] index_fn: [MultiSet[X], X] -> i64) -> Unit
// will clear the index function for all elements in the multiset
#[derive(Clone)]
struct ClearIndex {
    name: String,
    multiset: ArcSort,
    unit: ArcSort,
    fn_: Arc<FunctionSort>,
}

impl Primitive for ClearIndex {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.multiset.clone(), self.fn_.clone(), self.unit.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
        let fc = exec_state
            .container_values()
            .get_val::<FunctionContainer>(args[1])
            .unwrap()
            .clone();
        let multiset = exec_state
            .container_values()
            .get_val::<MultiSetContainer>(args[0])
            .unwrap()
            .clone();
        let ResolvedFunctionId::Lookup(action) = fc.0 else {
            panic!(
                "Primitive functions cannot be used with unstable-multiset-clear-index, since they cannot be deleted"
            );
        };
        for (v, _) in multiset.data.iter_counts() {
            action.remove(exec_state, &[args[0].clone(), v]);
        }
        Some(exec_state.base_values().get::<()>(()))
    }
}

// (unstable-multiset-flat-map (MultiSet[X], [X] -> MultiSet[X]) -> MultiSet[X])
// will map the function over all elements in the multiset and flatten the result. Any element in the multiset
// which does not have the function defined for it will be kept as-is.
#[derive(Clone)]
struct FlatMap {
    name: String,
    multiset: ArcSort,
    fn_: Arc<FunctionSort>,
}

impl Primitive for FlatMap {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![
                self.fn_.clone(),
                self.multiset.clone(),
                self.multiset.clone(),
            ],
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
        let multiset = exec_state
            .container_values()
            .get_val::<MultiSetContainer>(args[1])
            .unwrap()
            .clone();
        let mut new_data = MultiSet::<Value>::new();
        for (v, c) in multiset.data.iter_counts() {
            let mapped = fc.apply(exec_state, &[v]);
            if let Some(mapped_ms) = mapped {
                let mapped_ms = exec_state
                    .container_values()
                    .get_val::<MultiSetContainer>(mapped_ms)
                    .unwrap();
                for (mv, mc) in mapped_ms.data.iter_counts() {
                    new_data.insert_multiple_mut(mv, c * mc);
                }
            } else {
                new_data.insert_multiple_mut(v, c);
            }
        }
        let new_container = MultiSetContainer {
            data: new_data,
            ..multiset
        };
        Some(
            exec_state
                .clone()
                .container_values()
                .register_val(new_container, exec_state),
        )
    }
}

// (unstable-multiset-filter (MultiSet[X], [X] -> Unit) -> MultiSet[X])
// will filter the elements in the multiset based on whether the function is defined for them.
#[derive(Clone)]
struct Filter {
    name: String,
    multiset: ArcSort,
    fn_: Arc<FunctionSort>,
}

impl Primitive for Filter {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![
                self.fn_.clone(),
                self.multiset.clone(),
                self.multiset.clone(),
            ],
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
        let multiset = exec_state
            .container_values()
            .get_val::<MultiSetContainer>(args[1])
            .unwrap()
            .clone();
        let mut new_data = MultiSet::<Value>::new();
        // Filter out any elements which do not have the function defined for them
        for (v, c) in multiset.data.iter_counts() {
            let mapped = fc.apply(exec_state, &[v]);
            if mapped.is_some() {
                new_data.insert_multiple_mut(v, c);
            }
        }
        let multiset = MultiSetContainer {
            data: new_data,
            ..multiset
        };
        Some(
            exec_state
                .clone()
                .container_values()
                .register_val(multiset, exec_state),
        )
    }
}

// (multiset-sum-multisets (MultiSet[MultiSet[X]]) -> MultiSet[X])
// will sum all multisets in the outer multiset into a single multiset

#[derive(Clone)]
struct SumMultisets {
    name: String,
    multiset: ArcSort,
    multiset_of_multisets: ArcSort,
    do_rebuild: bool,
}

impl Primitive for SumMultisets {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.multiset_of_multisets.clone(), self.multiset.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
        let mut data = MultiSet::<Value>::new();
        let ms_of_ms = exec_state
            .container_values()
            .get_val::<MultiSetContainer>(args[0])
            .unwrap()
            .clone();
        for (ms_value, counts) in ms_of_ms.data.iter_counts() {
            let ms = exec_state
                .container_values()
                .get_val::<MultiSetContainer>(ms_value)
                .unwrap();
            for (v, c) in ms.data.iter_counts() {
                data.insert_multiple_mut(v, c * counts);
            }
        }
        let multiset = MultiSetContainer {
            data,
            do_rebuild: self.do_rebuild,
        };
        Some(
            exec_state
                .clone()
                .container_values()
                .register_val(multiset, exec_state),
        )
    }
}

// (unstable-multiset-fold ([X, X] -> X, X, MultiSet[X]) -> X
// will fold the multiset using the provided binary function and initial value
#[derive(Clone)]
struct Fold {
    name: String,
    multiset: ArcSort,
    fn_: Arc<FunctionSort>,
    element: ArcSort,
}

impl Primitive for Fold {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![
                self.fn_.clone(),
                self.element.clone(),
                self.multiset.clone(),
                self.element.clone(),
            ],
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
        let initial = args[1];
        let multiset = exec_state
            .container_values()
            .get_val::<MultiSetContainer>(args[2])
            .unwrap()
            .clone();
        let mut values = multiset.data.iter().cloned().collect::<Vec<_>>();
        let mut acc = if values.is_empty() {
            initial
        } else {
            let first = values[0];
            values.remove(0);
            first
        };
        for v in values {
            acc = fc.apply(exec_state, &[acc, v])?;
        }
        Some(acc)
    }
}

// Place multiset in its own module to keep implementation details private from sort
mod inner {
    use std::collections::BTreeMap;
    use std::hash::Hash;
    /// Immutable multiset implementation, which is threadsafe and hash stable, regardless of insertion order.
    ///
    /// All methods that return a new multiset take ownership of the old multiset.
    #[derive(Debug, Default, Hash, Eq, PartialEq, Clone)]
    pub struct MultiSet<T: Clone + Hash + Ord>(
        /// All values should be > 0
        BTreeMap<T, usize>,
        /// cached length
        usize,
    );

    impl<T: Clone + Hash + Ord> MultiSet<T> {
        /// Create a new empty multiset.
        pub fn new() -> Self {
            MultiSet(BTreeMap::new(), 0)
        }

        /// Check if the multiset contains a key.
        pub fn contains(&self, value: &T) -> bool {
            self.0.contains_key(value)
        }

        /// Return the total number of elements in the multiset.
        pub fn len(&self) -> usize {
            self.1
        }

        /// Return an iterator over all elements in the multiset.
        pub fn iter(&self) -> impl Iterator<Item = &T> {
            self.0.iter().flat_map(|(k, v)| std::iter::repeat_n(k, *v))
        }

        /// Return an iterator over values and counts
        pub fn iter_counts(&self) -> impl Iterator<Item = (T, usize)> {
            self.0.iter().map(|(k, v)| (k.clone(), *v))
        }

        /// Return an arbitrary element from the multiset.
        pub fn pick(&self) -> Option<&T> {
            self.0.keys().next()
        }

        /// Insert a value into the multiset, taking ownership of it and returning a new multiset.
        pub fn insert(mut self, value: T) -> MultiSet<T> {
            self.insert_multiple_mut(value, 1);
            self
        }

        /// Remove a value from the multiset, taking ownership of it and returning a new multiset.
        pub fn remove(mut self, value: &T) -> Option<MultiSet<T>> {
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

        pub fn insert_multiple_mut(&mut self, value: T, n: usize) {
            self.1 += n;
            if let Some(v) = self.0.get(&value) {
                self.0.insert(value, v + n);
            } else {
                self.0.insert(value, n);
            }
        }

        /// Compute the sum of two multisets.
        pub fn sum(mut self, MultiSet(other_map, other_count): Self) -> Self {
            let target_count = self.1 + other_count;
            for (k, v) in other_map {
                self.insert_multiple_mut(k, v);
            }
            assert_eq!(self.1, target_count);
            self
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
