use std::sync::Mutex;

use crate::constraint::AllEqualTypeConstraint;

use super::*;

type ValueVec = Vec<Value>;

#[derive(Debug)]
pub struct VecSort {
    name: Symbol,
    element: ArcSort,
    vecs: Mutex<IndexSet<ValueVec>>,
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
                .sorts
                .get(e)
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

    fn column_ty(&self, prims: &core_relations::Primitives) -> ColumnTy {
        ColumnTy::Primitive(prims.get_ty::<ValueVec>())
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
        let new_vec: ValueVec = vec
            .iter()
            .map(|e| {
                let mut e = *e;
                changed |= self.element.canonicalize(&mut e, unionfind);
                e
            })
            .collect();
        drop(vecs);
        *value = new_vec.store(self).unwrap();
        changed
    }

    fn register_primitives(self: Arc<Self>, typeinfo: &mut TypeInfo) {
        typeinfo.add_primitive(VecRebuild {
            name: "rebuild".into(),
            vec: self.clone(),
        });
        typeinfo.add_primitive(VecOf {
            name: "vec-of".into(),
            vec: self.clone(),
        });
        typeinfo.add_primitive(Append {
            name: "vec-append".into(),
            vec: self.clone(),
        });
        typeinfo.add_primitive(Ctor {
            name: "vec-empty".into(),
            vec: self.clone(),
        });
        typeinfo.add_primitive(Push {
            name: "vec-push".into(),
            vec: self.clone(),
        });
        typeinfo.add_primitive(Pop {
            name: "vec-pop".into(),
            vec: self.clone(),
        });
        typeinfo.add_primitive(NotContains {
            name: "vec-not-contains".into(),
            vec: self.clone(),
        });
        typeinfo.add_primitive(Contains {
            name: "vec-contains".into(),
            vec: self.clone(),
        });
        typeinfo.add_primitive(Length {
            name: "vec-length".into(),
            vec: self.clone(),
        });
        typeinfo.add_primitive(Get {
            name: "vec-get".into(),
            vec: self.clone(),
        });
        typeinfo.add_primitive(Set {
            name: "vec-set".into(),
            vec: self.clone(),
        });
        typeinfo.add_primitive(Remove {
            name: "vec-remove".into(),
            vec: self,
        })
    }

    fn extract_term(
        &self,
        _egraph: &EGraph,
        value: Value,
        extractor: &Extractor,
        termdag: &mut TermDag,
    ) -> Option<(Cost, Term)> {
        let vec: Vec<Value> = ValueVec::load(self, &value);
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

impl IntoSort for ValueVec {
    type Sort = VecSort;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        let mut vecs = sort.vecs.lock().unwrap();
        let (i, _) = vecs.insert_full(self);
        Some(Value {
            #[cfg(debug_assertions)]
            tag: sort.name,
            bits: i as u64,
        })
    }
}

impl FromSort for ValueVec {
    type Sort = VecSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let vecs = sort.vecs.lock().unwrap();
        vecs.get_index(value.bits as usize).unwrap().clone()
    }
}

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
        let vec = ValueVec::load(&self.vec, &values[0]);
        let new_vec: ValueVec = vec
            .iter()
            .map(|e| egraph.find(&self.vec.element, *e))
            .collect();
        drop(vec);
        Some(new_vec.store(&self.vec).unwrap())
    }
}
struct VecOf {
    name: Symbol,
    vec: Arc<VecSort>,
}

impl PrimitiveLike for VecOf {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        AllEqualTypeConstraint::new(self.name(), span.clone())
            .with_all_arguments_sort(self.vec.element())
            .with_output_sort(self.vec.clone())
            .into_box()
    }

    fn apply(
        &self,
        values: &[Value],
        _sorts: (&[ArcSort], &ArcSort),
        _egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
        let vec = ValueVec::from_iter(values.iter().copied());
        vec.store(&self.vec)
    }
}

struct Append {
    name: Symbol,
    vec: Arc<VecSort>,
}

impl PrimitiveLike for Append {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        AllEqualTypeConstraint::new(self.name(), span.clone())
            .with_all_arguments_sort(self.vec.clone())
            .into_box()
    }

    fn apply(
        &self,
        values: &[Value],
        _sorts: (&[ArcSort], &ArcSort),
        _egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
        let vec = ValueVec::from_iter(values.iter().flat_map(|v| ValueVec::load(&self.vec, v)));
        vec.store(&self.vec)
    }
}

struct Ctor {
    name: Symbol,
    vec: Arc<VecSort>,
}

impl PrimitiveLike for Ctor {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(self.name(), vec![self.vec.clone()], span.clone()).into_box()
    }

    fn apply(
        &self,
        values: &[Value],
        _sorts: (&[ArcSort], &ArcSort),
        _egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
        assert!(values.is_empty());
        ValueVec::default().store(&self.vec)
    }
}

struct Push {
    name: Symbol,
    vec: Arc<VecSort>,
}

impl PrimitiveLike for Push {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.vec.clone(), self.vec.element(), self.vec.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(
        &self,
        values: &[Value],
        _sorts: (&[ArcSort], &ArcSort),
        _egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
        let mut vec = ValueVec::load(&self.vec, &values[0]);
        vec.push(values[1]);
        vec.store(&self.vec)
    }
}

struct Pop {
    name: Symbol,
    vec: Arc<VecSort>,
}

impl PrimitiveLike for Pop {
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
        _egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
        let mut vec = ValueVec::load(&self.vec, &values[0]);
        vec.pop();
        vec.store(&self.vec)
    }
}

struct NotContains {
    name: Symbol,
    vec: Arc<VecSort>,
}

impl PrimitiveLike for NotContains {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.vec.clone(), self.vec.element(), Arc::new(UnitSort)],
            span.clone(),
        )
        .into_box()
    }

    fn apply(
        &self,
        values: &[Value],
        _sorts: (&[ArcSort], &ArcSort),
        _egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
        let vec = ValueVec::load(&self.vec, &values[0]);
        if vec.contains(&values[1]) {
            None
        } else {
            Some(Value::unit())
        }
    }
}

struct Contains {
    name: Symbol,
    vec: Arc<VecSort>,
}

impl PrimitiveLike for Contains {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.vec.clone(), self.vec.element(), Arc::new(UnitSort)],
            span.clone(),
        )
        .into_box()
    }

    fn apply(
        &self,
        values: &[Value],
        _sorts: (&[ArcSort], &ArcSort),
        _egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
        let vec = ValueVec::load(&self.vec, &values[0]);
        if vec.contains(&values[1]) {
            Some(Value::unit())
        } else {
            None
        }
    }
}

struct Length {
    name: Symbol,
    vec: Arc<VecSort>,
}

impl PrimitiveLike for Length {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.vec.clone(), Arc::new(I64Sort)],
            span.clone(),
        )
        .into_box()
    }

    fn apply(
        &self,
        values: &[Value],
        _sorts: (&[ArcSort], &ArcSort),
        _egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
        let vec = ValueVec::load(&self.vec, &values[0]);
        Some(Value::from(vec.len() as i64))
    }
}

struct Get {
    name: Symbol,
    vec: Arc<VecSort>,
}

impl PrimitiveLike for Get {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.vec.clone(), Arc::new(I64Sort), self.vec.element()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(
        &self,
        values: &[Value],
        _sorts: (&[ArcSort], &ArcSort),
        _egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
        let vec = ValueVec::load(&self.vec, &values[0]);
        let index = i64::load(&I64Sort, &values[1]);
        vec.get(index as usize).copied()
    }
}

struct Set {
    name: Symbol,
    vec: Arc<VecSort>,
}

impl PrimitiveLike for Set {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![
                self.vec.clone(),
                Arc::new(I64Sort),
                self.vec.element.clone(),
                self.vec.clone(),
            ],
            span.clone(),
        )
        .into_box()
    }

    fn apply(
        &self,
        values: &[Value],
        _sorts: (&[ArcSort], &ArcSort),
        _egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
        let mut vec = ValueVec::load(&self.vec, &values[0]);
        let index = i64::load(&I64Sort, &values[1]);
        vec[index as usize] = values[2];
        vec.store(&self.vec)
    }
}

struct Remove {
    name: Symbol,
    vec: Arc<VecSort>,
}

impl PrimitiveLike for Remove {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.vec.clone(), Arc::new(I64Sort), self.vec.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(
        &self,
        values: &[Value],
        _sorts: (&[ArcSort], &ArcSort),
        _egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
        let mut vec = ValueVec::load(&self.vec, &values[0]);
        let i = i64::load(&I64Sort, &values[1]);
        vec.remove(i.try_into().unwrap());
        vec.store(&self.vec)
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
