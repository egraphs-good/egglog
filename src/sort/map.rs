// use super::*;

// type ValueMap = BTreeMap<Value, Value>;

// /// A map from a key type to a value type supporting these primitives:
// /// - `map-empty`
// /// - `map-insert`
// /// - `map-get`
// /// - `map-contains`
// /// - `map-not-contains`
// /// - `map-remove`
// /// - `map-length`
// #[derive(Debug)]
// pub struct MapSort {
//     name: Symbol,
//     key: ArcSort,
//     value: ArcSort,
//     maps: Mutex<IndexSet<ValueMap>>,
// }

// impl MapSort {
//     fn key(&self) -> ArcSort {
//         self.key.clone()
//     }

//     fn value(&self) -> ArcSort {
//         self.value.clone()
//     }
// }

// impl Presort for MapSort {
//     fn presort_name() -> Symbol {
//         "Map".into()
//     }

//     fn reserved_primitives() -> Vec<Symbol> {
//         vec![
//             "map-empty".into(),
//             "map-insert".into(),
//             "map-get".into(),
//             "map-not-contains".into(),
//             "map-contains".into(),
//             "map-remove".into(),
//             "map-length".into(),
//         ]
//     }

//     fn make_sort(
//         typeinfo: &mut TypeInfo,
//         name: Symbol,
//         args: &[Expr],
//     ) -> Result<ArcSort, TypeError> {
//         if let [Expr::Var(k_span, k), Expr::Var(v_span, v)] = args {
//             let k = typeinfo
//                 .sorts
//                 .get(k)
//                 .ok_or(TypeError::UndefinedSort(*k, k_span.clone()))?;
//             let v = typeinfo
//                 .sorts
//                 .get(v)
//                 .ok_or(TypeError::UndefinedSort(*v, v_span.clone()))?;

//             // TODO: specialize the error message
//             if k.is_eq_container_sort() {
//                 return Err(TypeError::DisallowedSort(
//                     name,
//                     "Maps nested with other EqSort containers are not allowed".into(),
//                     k_span.clone(),
//                 ));
//             }

//             if v.is_container_sort() {
//                 return Err(TypeError::DisallowedSort(
//                     name,
//                     "Maps nested with other EqSort containers are not allowed".into(),
//                     v_span.clone(),
//                 ));
//             }

//             Ok(Arc::new(Self {
//                 name,
//                 key: k.clone(),
//                 value: v.clone(),
//                 maps: Default::default(),
//             }))
//         } else {
//             panic!()
//         }
//     }
// }

// impl Sort for MapSort {
//     fn name(&self) -> Symbol {
//         self.name
//     }

//     fn column_ty(&self, prims: &Primitives) -> ColumnTy {
//         ColumnTy::Primitive(prims.get_ty::<ValueMap>())
//     }

//     fn register_type(&self, prims: &mut Primitives) {
//         prims.register_type::<ValueMap>();
//     }

//     fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
//         self
//     }

//     fn is_container_sort(&self) -> bool {
//         true
//     }

//     fn is_eq_container_sort(&self) -> bool {
//         self.key.is_eq_sort() || self.value.is_eq_sort()
//     }

//     fn inner_values(&self, value: &Value) -> Vec<(ArcSort, Value)> {
//         let maps = self.maps.lock().unwrap();
//         let map = maps.get_index(value.bits as usize).unwrap();
//         let mut result = Vec::new();
//         for (k, v) in map.iter() {
//             result.push((self.key.clone(), *k));
//             result.push((self.value.clone(), *v));
//         }
//         result
//     }

//     fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
//         let maps = self.maps.lock().unwrap();
//         let map = maps.get_index(value.bits as usize).unwrap();
//         let mut changed = false;
//         let new_map: ValueMap = map
//             .iter()
//             .map(|(k, v)| {
//                 let (mut k, mut v) = (*k, *v);
//                 changed |= self.key.canonicalize(&mut k, unionfind);
//                 changed |= self.value.canonicalize(&mut v, unionfind);
//                 (k, v)
//             })
//             .collect();
//         drop(maps);
//         *value = new_map.store(self);
//         changed
//     }

//     fn register_primitives(self: Arc<Self>, typeinfo: &mut TypeInfo) {
//         typeinfo.add_primitive(Ctor {
//             name: "map-empty".into(),
//             map: self.clone(),
//         });
//         typeinfo.add_primitive(Insert {
//             name: "map-insert".into(),
//             map: self.clone(),
//         });
//         typeinfo.add_primitive(Get {
//             name: "map-get".into(),
//             map: self.clone(),
//         });
//         typeinfo.add_primitive(NotContains {
//             name: "map-not-contains".into(),
//             map: self.clone(),
//         });
//         typeinfo.add_primitive(Contains {
//             name: "map-contains".into(),
//             map: self.clone(),
//         });
//         typeinfo.add_primitive(Remove {
//             name: "map-remove".into(),
//             map: self.clone(),
//         });
//         typeinfo.add_primitive(Length {
//             name: "map-length".into(),
//             map: self,
//         });
//     }

//     fn extract_term(
//         &self,
//         _egraph: &EGraph,
//         value: Value,
//         extractor: &Extractor,
//         termdag: &mut TermDag,
//     ) -> Option<(Cost, Term)> {
//         let map = ValueMap::load(self, &value);
//         let mut term = termdag.app("map-empty".into(), vec![]);
//         let mut cost = 0usize;
//         for (k, v) in map.iter().rev() {
//             let k = extractor.find_best(*k, termdag, &self.key)?;
//             let v = extractor.find_best(*v, termdag, &self.value)?;
//             cost = cost.saturating_add(k.0).saturating_add(v.0);
//             term = termdag.app("map-insert".into(), vec![term, k.1, v.1]);
//         }
//         Some((cost, term))
//     }
// }

// impl IntoSort for ValueMap {
//     type Sort = MapSort;
//     fn store(self, sort: &Self::Sort) -> Value {
//         let mut maps = sort.maps.lock().unwrap();
//         let (i, _) = maps.insert_full(self);
//         Value {
//             #[cfg(debug_assertions)]
//             tag: sort.name,
//             bits: i as u64,
//         }
//     }
// }

// impl FromSort for ValueMap {
//     type Sort = MapSort;
//     fn load(sort: &Self::Sort, value: &Value) -> Self {
//         let maps = sort.maps.lock().unwrap();
//         maps.get_index(value.bits as usize).unwrap().clone()
//     }
// }

// struct Ctor {
//     name: Symbol,
//     map: Arc<MapSort>,
// }

// impl PrimitiveLike for Ctor {
//     fn name(&self) -> Symbol {
//         self.name
//     }

//     fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
//         SimpleTypeConstraint::new(self.name(), vec![self.map.clone()], span.clone()).into_box()
//     }

//     fn apply(
//         &self,
//         values: &[Value],
//         _sorts: (&[ArcSort], &ArcSort),
//         _egraph: Option<&mut EGraph>,
//     ) -> Option<Value> {
//         assert!(values.is_empty());
//         Some(ValueMap::default().store(&self.map))
//     }
// }

// struct Insert {
//     name: Symbol,
//     map: Arc<MapSort>,
// }

// impl PrimitiveLike for Insert {
//     fn name(&self) -> Symbol {
//         self.name
//     }

//     fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
//         SimpleTypeConstraint::new(
//             self.name(),
//             vec![
//                 self.map.clone(),
//                 self.map.key(),
//                 self.map.value(),
//                 self.map.clone(),
//             ],
//             span.clone(),
//         )
//         .into_box()
//     }

//     fn apply(
//         &self,
//         values: &[Value],
//         _sorts: (&[ArcSort], &ArcSort),
//         _egraph: Option<&mut EGraph>,
//     ) -> Option<Value> {
//         let mut map = ValueMap::load(&self.map, &values[0]);
//         map.insert(values[1], values[2]);
//         Some(map.store(&self.map))
//     }
// }

// struct Get {
//     name: Symbol,
//     map: Arc<MapSort>,
// }

// impl PrimitiveLike for Get {
//     fn name(&self) -> Symbol {
//         self.name
//     }

//     fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
//         SimpleTypeConstraint::new(
//             self.name(),
//             vec![self.map.clone(), self.map.key(), self.map.value()],
//             span.clone(),
//         )
//         .into_box()
//     }

//     fn apply(
//         &self,
//         values: &[Value],
//         _sorts: (&[ArcSort], &ArcSort),
//         _egraph: Option<&mut EGraph>,
//     ) -> Option<Value> {
//         let map = ValueMap::load(&self.map, &values[0]);
//         map.get(&values[1]).copied()
//     }
// }

// struct NotContains {
//     name: Symbol,
//     map: Arc<MapSort>,
// }

// impl PrimitiveLike for NotContains {
//     fn name(&self) -> Symbol {
//         self.name
//     }

//     fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
//         SimpleTypeConstraint::new(
//             self.name(),
//             vec![self.map.clone(), self.map.key(), Arc::new(UnitSort)],
//             span.clone(),
//         )
//         .into_box()
//     }

//     fn apply(
//         &self,
//         values: &[Value],
//         _sorts: (&[ArcSort], &ArcSort),
//         _egraph: Option<&mut EGraph>,
//     ) -> Option<Value> {
//         let map = ValueMap::load(&self.map, &values[0]);
//         if map.contains_key(&values[1]) {
//             None
//         } else {
//             Some(Value::unit())
//         }
//     }
// }

// struct Contains {
//     name: Symbol,
//     map: Arc<MapSort>,
// }

// impl PrimitiveLike for Contains {
//     fn name(&self) -> Symbol {
//         self.name
//     }

//     fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
//         SimpleTypeConstraint::new(
//             self.name(),
//             vec![self.map.clone(), self.map.key(), Arc::new(UnitSort)],
//             span.clone(),
//         )
//         .into_box()
//     }

//     fn apply(
//         &self,
//         values: &[Value],
//         _sorts: (&[ArcSort], &ArcSort),
//         _egraph: Option<&mut EGraph>,
//     ) -> Option<Value> {
//         let map = ValueMap::load(&self.map, &values[0]);
//         if map.contains_key(&values[1]) {
//             Some(Value::unit())
//         } else {
//             None
//         }
//     }
// }

// struct Remove {
//     name: Symbol,
//     map: Arc<MapSort>,
// }

// impl PrimitiveLike for Remove {
//     fn name(&self) -> Symbol {
//         self.name
//     }

//     fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
//         SimpleTypeConstraint::new(
//             self.name(),
//             vec![self.map.clone(), self.map.key(), self.map.clone()],
//             span.clone(),
//         )
//         .into_box()
//     }

//     fn apply(
//         &self,
//         values: &[Value],
//         _sorts: (&[ArcSort], &ArcSort),
//         _egraph: Option<&mut EGraph>,
//     ) -> Option<Value> {
//         let mut map = ValueMap::load(&self.map, &values[0]);
//         map.remove(&values[1]);
//         Some(map.store(&self.map))
//     }
// }

// struct Length {
//     name: Symbol,
//     map: Arc<MapSort>,
// }

// impl PrimitiveLike for Length {
//     fn name(&self) -> Symbol {
//         self.name
//     }

//     fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
//         SimpleTypeConstraint::new(
//             self.name(),
//             vec![self.map.clone(), Arc::new(I64Sort)],
//             span.clone(),
//         )
//         .into_box()
//     }

//     fn apply(
//         &self,
//         values: &[Value],
//         _sorts: (&[ArcSort], &ArcSort),
//         _egraph: Option<&mut EGraph>,
//     ) -> Option<Value> {
//         let map = ValueMap::load(&self.map, &values[0]);
//         Some(Value::from(map.len() as i64))
//     }
// }
