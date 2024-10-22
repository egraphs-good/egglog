use std::sync::Mutex;

use inner::MultiSet;

use super::*;
use crate::constraint::{AllEqualTypeConstraint, SimpleTypeConstraint};

// Place multiset in its own module to keep implementation details private from sort
mod inner {
    use rpds::RedBlackTreeMapSync;
    use std::hash::Hash;
    /// Immutable multiset implementation, which is threadsafe and hash stable, regardless of insertion order.
    ///
    /// All methods that return a new multiset take ownership of the old multiset.
    #[derive(Debug, Hash, Eq, PartialEq, Clone)]
    pub(crate) struct MultiSet<T: Hash + Ord + Clone>(
        /// All values should be > 0
        RedBlackTreeMapSync<T, usize>,
    );

    impl<T: Hash + Ord + Clone> MultiSet<T> {
        /// Create a new empty multiset.
        pub(crate) fn new() -> Self {
            MultiSet(RedBlackTreeMapSync::new_sync())
        }

        /// Check if the multiset contains a key.
        pub(crate) fn contains(&self, value: &T) -> bool {
            self.0.contains_key(value)
        }

        /// Return the total number of elements in the multiset.
        pub(crate) fn len(&self) -> usize {
            self.0.iter().map(|(_, v)| *v).sum()
        }

        /// Return an iterator over all elements in the multiset.
        pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
            self.0
                .iter()
                .flat_map(|(k, v)| std::iter::repeat(k).take(*v))
        }

        /// Return an arbitrary element from the multiset.
        pub(crate) fn pick(&self) -> Option<&T> {
            self.0.first().map(|(k, _)| k)
        }

        /// Map a function over all elements in the multiset, taking ownership of it and returning a new multiset.
        pub(crate) fn map(self, mut f: impl FnMut(&T) -> T) -> MultiSet<T> {
            let mut new = MultiSet::new();
            for (k, v) in self.0.into_iter() {
                new.insert_multiple_mut(f(k), *v);
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
                if *v == 1 {
                    self.0.remove_mut(value);
                } else {
                    self.0.insert_mut(value.clone(), v - 1);
                }
                Some(self)
            } else {
                None
            }
        }

        fn insert_multiple_mut(&mut self, value: T, n: usize) {
            if let Some(v) = self.0.get(&value) {
                self.0.insert_mut(value, v + n);
            } else {
                self.0.insert_mut(value, n);
            }
        }

        /// Create a multiset from an iterator.
        pub(crate) fn from_iter(iter: impl IntoIterator<Item = T>) -> Self {
            let mut multiset = MultiSet::new();
            for value in iter {
                multiset.insert_multiple_mut(value, 1);
            }
            multiset
        }
    }
}

type ValueMultiSet = MultiSet<Value>;

#[derive(Debug)]
pub struct MultiSetSort {
    name: Symbol,
    element: ArcSort,
    multisets: Mutex<IndexSet<ValueMultiSet>>,
}

impl MultiSetSort {
    pub fn element(&self) -> ArcSort {
        self.element.clone()
    }

    pub fn element_name(&self) -> Symbol {
        self.element.name()
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
                .sorts
                .get(e)
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
            .iter()
            .map(|k| (self.element.clone(), *k))
            .collect()
    }

    fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
        let multisets = self.multisets.lock().unwrap();
        let multiset = multisets.get_index(value.bits as usize).unwrap().clone();
        let mut changed = false;
        let new_multiset = multiset.map(|e| {
            let mut e = *e;
            changed |= self.element.canonicalize(&mut e, unionfind);
            e
        });
        drop(multisets);
        *value = new_multiset.store(self).unwrap();
        changed
    }

    fn register_primitives(self: Arc<Self>, typeinfo: &mut TypeInfo) {
        typeinfo.add_primitive(MultiSetOf {
            name: "multiset-of".into(),
            multiset: self.clone(),
        });
        typeinfo.add_primitive(Insert {
            name: "multiset-insert".into(),
            multiset: self.clone(),
        });
        typeinfo.add_primitive(Contains {
            name: "multiset-contains".into(),
            multiset: self.clone(),
        });
        typeinfo.add_primitive(NotContains {
            name: "multiset-not-contains".into(),
            multiset: self.clone(),
        });
        typeinfo.add_primitive(Remove {
            name: "multiset-remove".into(),
            multiset: self.clone(),
        });
        typeinfo.add_primitive(Length {
            name: "multiset-length".into(),
            multiset: self.clone(),
        });
        typeinfo.add_primitive(Pick {
            name: "multiset-pick".into(),
            multiset: self.clone(),
        });
        let inner_name = self.element.name();
        let fn_sort = typeinfo.get_sort_by(|s: &Arc<FunctionSort>| {
            (s.output.name() == inner_name)
                && s.inputs.len() == 1
                && (s.inputs[0].name() == inner_name)
        });
        // Only include map function if we already declared a function sort with the correct signature
        if let Some(fn_sort) = fn_sort {
            typeinfo.add_primitive(Map {
                name: "unstable-multiset-map".into(),
                multiset: self.clone(),
                fn_: fn_sort,
            });
        }
    }

    fn make_expr(&self, egraph: &EGraph, value: Value) -> (Cost, Expr) {
        let mut termdag = TermDag::default();
        let extractor = Extractor::new(egraph, &mut termdag);
        self.extract_expr(egraph, value, &extractor, &mut termdag)
            .expect("Extraction should be successful since extractor has been fully initialized")
    }

    fn extract_expr(
        &self,
        _egraph: &EGraph,
        value: Value,
        extractor: &Extractor,
        termdag: &mut TermDag,
    ) -> Option<(Cost, Expr)> {
        let multiset = ValueMultiSet::load(self, &value);
        let mut children = vec![];
        let mut cost = 0usize;
        for e in multiset.iter() {
            let (child_cost, child_term) = extractor.find_best(*e, termdag, &self.element)?;
            cost = cost.saturating_add(child_cost);
            children.push(termdag.term_to_expr(&child_term));
        }
        let expr = Expr::call_no_span("multiset-of", children);
        Some((cost, expr))
    }

    fn serialized_name(&self, _value: &Value) -> Symbol {
        "multiset-of".into()
    }
}

impl IntoSort for ValueMultiSet {
    type Sort = MultiSetSort;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        let mut multisets = sort.multisets.lock().unwrap();
        let (i, _) = multisets.insert_full(self);
        Some(Value {
            tag: sort.name,
            bits: i as u64,
        })
    }
}

impl FromSort for ValueMultiSet {
    type Sort = MultiSetSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let sets = sort.multisets.lock().unwrap();
        sets.get_index(value.bits as usize).unwrap().clone()
    }
}

struct MultiSetOf {
    name: Symbol,
    multiset: Arc<MultiSetSort>,
}

impl PrimitiveLike for MultiSetOf {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        AllEqualTypeConstraint::new(self.name(), span.clone())
            .with_all_arguments_sort(self.multiset.element())
            .with_output_sort(self.multiset.clone())
            .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
        let multiset = MultiSet::from_iter(values.iter().copied());
        Some(multiset.store(&self.multiset).unwrap())
    }
}

struct Insert {
    name: Symbol,
    multiset: Arc<MultiSetSort>,
}

impl PrimitiveLike for Insert {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![
                self.multiset.clone(),
                self.multiset.element(),
                self.multiset.clone(),
            ],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
        let multiset = ValueMultiSet::load(&self.multiset, &values[0]);
        let multiset = multiset.insert(values[1]);
        multiset.store(&self.multiset)
    }
}

struct Contains {
    name: Symbol,
    multiset: Arc<MultiSetSort>,
}

impl PrimitiveLike for Contains {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![
                self.multiset.clone(),
                self.multiset.element(),
                Arc::new(UnitSort),
            ],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
        let multiset = ValueMultiSet::load(&self.multiset, &values[0]);
        if multiset.contains(&values[1]) {
            Some(Value::unit())
        } else {
            None
        }
    }
}

struct NotContains {
    name: Symbol,
    multiset: Arc<MultiSetSort>,
}

impl PrimitiveLike for NotContains {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![
                self.multiset.clone(),
                self.multiset.element(),
                Arc::new(UnitSort),
            ],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
        let multiset = ValueMultiSet::load(&self.multiset, &values[0]);
        if !multiset.contains(&values[1]) {
            Some(Value::unit())
        } else {
            None
        }
    }
}

struct Length {
    name: Symbol,
    multiset: Arc<MultiSetSort>,
}

impl PrimitiveLike for Length {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.multiset.clone(), Arc::new(I64Sort)],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
        let multiset = ValueMultiSet::load(&self.multiset, &values[0]);
        Some(Value::from(multiset.len() as i64))
    }
}

struct Remove {
    name: Symbol,
    multiset: Arc<MultiSetSort>,
}

impl PrimitiveLike for Remove {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![
                self.multiset.clone(),
                self.multiset.element(),
                self.multiset.clone(),
            ],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
        let multiset = ValueMultiSet::load(&self.multiset, &values[0]);
        let multiset = multiset.remove(&values[1]);
        multiset.store(&self.multiset)
    }
}

struct Pick {
    name: Symbol,
    multiset: Arc<MultiSetSort>,
}

impl PrimitiveLike for Pick {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.multiset.clone(), self.multiset.element()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
        let multiset = ValueMultiSet::load(&self.multiset, &values[0]);
        Some(*multiset.pick().expect("Cannot pick from an empty multiset"))
    }
}

struct Map {
    name: Symbol,
    multiset: Arc<MultiSetSort>,
    fn_: Arc<FunctionSort>,
}

impl PrimitiveLike for Map {
    fn name(&self) -> Symbol {
        self.name
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

    fn apply(&self, values: &[Value], egraph: Option<&mut EGraph>) -> Option<Value> {
        let egraph =
            egraph.unwrap_or_else(|| panic!("`{}` is not supported yet in facts.", self.name));
        let multiset = ValueMultiSet::load(&self.multiset, &values[1]);
        let new_multiset = multiset.map(|e| self.fn_.apply(&values[0], &[*e], egraph));
        new_multiset.store(&self.multiset)
    }
}
