use super::*;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MapContainer {
    do_rebuild_keys: bool,
    do_rebuild_vals: bool,
    pub data: BTreeMap<Value, Value>,
}

impl ContainerValue for MapContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn ValueRebuilder) -> bool {
        let mut changed = false;
        if self.do_rebuild_keys {
            self.data = self
                .data
                .iter()
                .map(|(old, v)| {
                    let new = rebuilder.rebuild_val(*old);
                    changed |= *old != new;
                    (new, *v)
                })
                .collect();
        }
        if self.do_rebuild_vals {
            for old in self.data.values_mut() {
                let new = rebuilder.rebuild_val(*old);
                changed |= *old != new;
                *old = new;
            }
        }
        changed
    }
    fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        self.data.iter().flat_map(|(k, v)| [k, v]).copied()
    }
}

/// The entries of a flat `(map-of k0 v0 ...)` term as a Rust `BTreeMap` in
/// canonical key order, with `MapContainer`'s last-write-wins semantics on
/// duplicate keys; `None` for any other term.
fn map_term_to_btreemap<'a>(
    termdag: &'a TermDag,
    term_id: TermId,
) -> Option<BTreeMap<OrdTerm<'a>, TermId>> {
    match termdag.get(term_id) {
        Term::App(head, args) if head == "map-of" => map_of_args_to_btreemap(termdag, args),
        _ => None,
    }
}

/// Alternating `[k0, v0, ...]` `map-of` arguments as a `BTreeMap` (see
/// [`map_term_to_btreemap`]); `None` on odd arity.
fn map_of_args_to_btreemap<'a>(
    termdag: &'a TermDag,
    args: &[TermId],
) -> Option<BTreeMap<OrdTerm<'a>, TermId>> {
    if !args.len().is_multiple_of(2) {
        return None;
    }
    Some(
        args.chunks_exact(2)
            .map(|kv| (termdag.ord_term(kv[0]), kv[1]))
            .collect(),
    )
}

/// Flatten a map back to the `[k0, v0, k1, v1, ...]` argument list of its
/// canonical `(map-of ...)` term (sorted by key order, deduplicated).
fn map_term_args(map: BTreeMap<OrdTerm<'_>, TermId>) -> Vec<TermId> {
    map.into_iter().flat_map(|(k, v)| [k.id(), v]).collect()
}

/// Canonicalize alternating `[k0, v0, ...]` arguments to the flat
/// `(map-of ...)` term; `None` on odd arity.
fn normalize_map_term(termdag: &mut TermDag, args: &[TermId]) -> Option<TermId> {
    let flat = map_term_args(map_of_args_to_btreemap(termdag, args)?);
    Some(termdag.app("map-of".to_string(), flat))
}

/// A map from a key type to a value type supporting these primitives:
/// - `map-empty`
/// - `map-insert`
/// - `map-get`
/// - `map-contains`
/// - `map-not-contains`
/// - `map-remove`
/// - `map-length`
#[derive(Clone, Debug)]
pub struct MapSort {
    name: String,
    key: ArcSort,
    value: ArcSort,
}

impl MapSort {
    pub fn key(&self) -> ArcSort {
        self.key.clone()
    }

    pub fn value(&self) -> ArcSort {
        self.value.clone()
    }
}

impl Presort for MapSort {
    fn container_type() -> Option<TypeId> {
        Some(TypeId::of::<<Self as ContainerSort>::Container>())
    }

    fn builtin_definitions() -> Vec<proto::Program> {
        use proto as pb;
        [
            ("empty", "map-empty", vec![], 2),
            ("of", "map-from-pairs", vec![("entries", 4)], 2),
            ("get", "map-get", vec![("map", 2), ("key", 0)], 1),
            (
                "insert",
                "map-insert",
                vec![("map", 2), ("key", 0), ("value", 1)],
                2,
            ),
        ]
        .into_iter()
        .map(|(key, alias, inputs, output)| {
            let receiver = matches!(key, "get" | "insert");
            let parameters = inputs.iter().enumerate().skip(usize::from(receiver));
            pb::Program {
                ir_version: 1,
                sorts: vec![
                    pb::Sort {
                        kind: Some(pb::sort::Kind::Var(0)),
                        ..Default::default()
                    },
                    pb::Sort {
                        kind: Some(pb::sort::Kind::Var(1)),
                        ..Default::default()
                    },
                    pb::Sort {
                        kind: Some(pb::sort::Kind::Family(pb::HostSort {
                            name: "Map".into(),
                            args: vec![0, 1],
                        })),
                        ..Default::default()
                    },
                    pb::Sort {
                        kind: Some(pb::sort::Kind::Family(pb::HostSort {
                            name: "Pair".into(),
                            args: vec![0, 1],
                        })),
                        ..Default::default()
                    },
                    pb::Sort {
                        kind: Some(pb::sort::Kind::Family(pb::HostSort {
                            name: "Vec".into(),
                            args: vec![3],
                        })),
                        ..Default::default()
                    },
                ],
                declarations: vec![
                    pb::Declaration {
                        kind: Some(pb::declaration::Kind::HostSortFamily(pb::HostSortFamily {
                            name: "Map".into(),
                            arity: 2,
                            bindings: Some(pb::SortBindings {
                                python: Some(pb::TypeBinding {
                                    path: vec!["egglog".into(), "builtins".into(), "Map".into()],
                                    type_params: vec!["K".into(), "V".into()],
                                }),
                                rust: Some(pb::TypeBinding {
                                    path: vec![
                                        "egglog_experimental".into(),
                                        "typed".into(),
                                        "builtins".into(),
                                        "Map".into(),
                                    ],
                                    type_params: vec!["K".into(), "V".into()],
                                }),
                                ..Default::default()
                            }),
                        })),
                        ..Default::default()
                    },
                    pb::Declaration {
                        kind: Some(pb::declaration::Kind::HostSortFamily(pb::HostSortFamily {
                            name: "Pair".into(),
                            arity: 2,
                            ..Default::default()
                        })),
                        ..Default::default()
                    },
                    pb::Declaration {
                        kind: Some(pb::declaration::Kind::HostSortFamily(pb::HostSortFamily {
                            name: "Vec".into(),
                            arity: 1,
                            ..Default::default()
                        })),
                        ..Default::default()
                    },
                    pb::Declaration {
                        kind: Some(pb::declaration::Kind::HostPrimitive(pb::HostPrimitive {
                            name: format!("egglog.core.map.{key}"),
                            typing: Some(pb::host_primitive::Typing::Signature(
                                pb::GenericSignature {
                                    type_params: vec!["K".into(), "V".into()],
                                    inputs: inputs
                                        .iter()
                                        .map(|(name, sort)| pb::Arg {
                                            name: (*name).into(),
                                            sort: *sort,
                                        })
                                        .collect(),
                                    output: Some(output),
                                    varargs: vec![],
                                },
                            )),
                        })),
                        bindings: Some(pb::CallableBindings {
                            python: Some(pb::PythonBindings {
                                views: vec![pb::PythonCallable {
                                    kind: if receiver {
                                        pb::PythonCallKind::Method
                                    } else {
                                        pb::PythonCallKind::ClassMethod
                                    }
                                    .into(),
                                    path: vec![
                                        match key {
                                            "of" => "from_pairs",
                                            "get" => "__getitem__",
                                            _ => key,
                                        }
                                        .into(),
                                    ],
                                    owner: Some(pb::BindingOwner {
                                        kind: Some(pb::binding_owner::Kind::Sort(2)),
                                    }),
                                    receiver: receiver.then_some(0),
                                    params: parameters
                                        .clone()
                                        .map(|(i, (name, _))| pb::PythonParameter {
                                            core_input: Some(i as u32),
                                            name: (*name).into(),
                                            default_expr: None,
                                        })
                                        .collect(),
                                    mutates: None,
                                }],
                            }),
                            rust: Some(pb::RustBindings {
                                views: vec![pb::RustCallable {
                                    path: vec![if key == "of" { "from_pairs" } else { key }.into()],
                                    owner: Some(pb::BindingOwner {
                                        kind: Some(pb::binding_owner::Kind::Sort(2)),
                                    }),
                                    receiver: receiver.then_some(pb::RustReceiver {
                                        core_input: Some(0),
                                        borrowed: true,
                                    }),
                                    params: parameters
                                        .map(|(i, (name, _))| pb::RustParameter {
                                            core_input: Some(i as u32),
                                            name: (*name).into(),
                                            borrowed: false,
                                        })
                                        .collect(),
                                    ..Default::default()
                                }],
                            }),
                            egglog: Some(pb::EgglogBindings {
                                views: vec![pb::EgglogCallable {
                                    symbol: alias.into(),
                                    datatype_member: false,
                                }],
                            }),
                        }),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }
        })
        .collect()
    }

    fn presort_name() -> &'static str {
        "Map"
    }

    fn reserved_primitives() -> Vec<&'static str> {
        vec![
            "map-empty",
            "map-of",
            "map-from-pairs",
            "map-insert",
            "map-get",
            "map-not-contains",
            "map-contains",
            "map-remove",
            "map-length",
        ]
    }

    fn make_sort(
        typeinfo: &mut TypeInfo,
        name: String,
        args: &[Expr],
        span: Span,
    ) -> Result<ArcSort, TypeError> {
        if let [Expr::Var(k_span, k), Expr::Var(v_span, v)] = args {
            let k = typeinfo
                .get_sort_by_name(k)
                .ok_or(TypeError::UndefinedSort(k.clone(), k_span.clone()))?;
            let v = typeinfo
                .get_sort_by_name(v)
                .ok_or(TypeError::UndefinedSort(v.clone(), v_span.clone()))?;

            let out = Self {
                name,
                key: k.clone(),
                value: v.clone(),
            };
            Ok(out.to_arcsort())
        } else {
            Err(TypeError::BadPresortArguments(
                Self::presort_name().to_owned(),
                span,
            ))
        }
    }
}

impl ContainerSort for MapSort {
    type Container = MapContainer;

    fn name(&self) -> &str {
        &self.name
    }

    fn inner_sorts(&self) -> Vec<ArcSort> {
        vec![self.key.clone(), self.value.clone()]
    }

    fn is_eq_container_sort(&self) -> bool {
        self.key.is_eq_sort()
            || self.value.is_eq_sort()
            || self.key.is_eq_container_sort()
            || self.value.is_eq_container_sort()
    }

    fn inner_values(
        &self,
        container_values: &ContainerValues,
        value: Value,
    ) -> Vec<(ArcSort, Value)> {
        let val = container_values
            .get_val::<MapContainer>(value)
            .unwrap()
            .clone();
        val.data
            .iter()
            .flat_map(|(k, v)| [(self.key.clone(), *k), (self.value.clone(), *v)])
            .collect()
    }

    fn register_primitives(&self, eg: &mut EGraph) {
        let arc = self.clone().to_arcsort();
        eg.type_info
            .register_builtin_sort(
                "Map",
                arc.clone(),
                vec![self.key.clone(), self.value.clone()],
            )
            .unwrap();
        let empty = eg
            .type_info
            .instantiate_builtin("egglog.core.map.empty", &arc)
            .unwrap();
        let of = eg
            .type_info
            .instantiate_builtin("egglog.core.map.of", &arc)
            .unwrap();
        let get = eg
            .type_info
            .instantiate_builtin("egglog.core.map.get", &arc)
            .unwrap();
        let insert = eg
            .type_info
            .instantiate_builtin("egglog.core.map.insert", &arc)
            .unwrap();

        // The proof "term form" of a map is the flat `(map-of k0 v0 k1 v1 ...)`
        // in canonical key order (like `set-of`/`vec-of`), matching
        // `reconstruct_termdag`. Each validator round-trips through a Rust
        // `BTreeMap` (see `map_term_to_btreemap`), so it evaluates map terms
        // with `MapContainer`'s semantics; `None` for a malformed map term
        // fails the proof.
        let map_empty_validator = |termdag: &mut TermDag, _args: &[TermId]| -> Option<TermId> {
            Some(termdag.app("map-of".into(), vec![]))
        };
        let map_from_pairs_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let [entries] = args else { return None };
            let Term::App(head, entries) = termdag.get(*entries) else {
                return None;
            };
            if head != "vec-of" && !(head == "vec-empty" && entries.is_empty()) {
                return None;
            }
            let mut flat = Vec::with_capacity(entries.len() * 2);
            for entry in entries {
                let Term::App(head, children) = termdag.get(*entry) else {
                    return None;
                };
                if head != "pair" || children.len() != 2 {
                    return None;
                }
                flat.extend_from_slice(children);
            }
            normalize_map_term(termdag, &flat)
        };
        let map_insert_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let [map, key, value] = args else {
                return None;
            };
            let mut map = map_term_to_btreemap(termdag, *map)?;
            map.insert(termdag.ord_term(*key), *value);
            let flat = map_term_args(map);
            Some(termdag.app("map-of".into(), flat))
        };
        let map_get_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let [map, key] = args else { return None };
            map_term_to_btreemap(termdag, *map)?
                .get(&termdag.ord_term(*key))
                .copied()
        };
        let map_length_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let [map] = args else { return None };
            let len = map_term_to_btreemap(termdag, *map)?.len() as i64;
            Some(termdag.lit(Literal::Int(len)))
        };
        let map_contains_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let [map, key] = args else { return None };
            let contains =
                map_term_to_btreemap(termdag, *map)?.contains_key(&termdag.ord_term(*key));
            contains.then(|| termdag.lit(Literal::Unit))
        };
        let map_not_contains_validator =
            |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
                let [map, key] = args else { return None };
                let contains =
                    map_term_to_btreemap(termdag, *map)?.contains_key(&termdag.ord_term(*key));
                (!contains).then(|| termdag.lit(Literal::Unit))
            };

        add_primitive_with_validator!(eg, "map-empty" [instance = empty] = {self.clone(): MapSort} || -> @MapContainer { MapContainer {
            do_rebuild_keys: self.ctx.key.is_eq_sort() || self.ctx.key.is_eq_container_sort(),
            do_rebuild_vals: self.ctx.value.is_eq_sort() || self.ctx.value.is_eq_container_sort(),
            data: BTreeMap::new()
        } }, map_empty_validator);

        // Flat source/proof compatibility is derived from the fixed canonical
        // signature, but is not another wire callable or exported overload.
        eg.add_pure_primitive(
            MapOf {
                instance: of.clone(),
                key: self.key.clone(),
                value: self.value.clone(),
            },
            Some(std::sync::Arc::new(normalize_map_term)),
        );

        add_primitive_with_validator!(eg, "map-from-pairs" [instance = of] = {self.clone(): MapSort} |entries: @VecContainer| -> @MapContainer {
            MapContainer {
                do_rebuild_keys: self.ctx.key.is_eq_sort() || self.ctx.key.is_eq_container_sort(),
                do_rebuild_vals: self.ctx.value.is_eq_sort() || self.ctx.value.is_eq_container_sort(),
                data: entries.data.iter().map(|entry| {
                    let pair = state.value_to_container::<PairContainer>(*entry).expect("typed Pair entry");
                    (pair.first, pair.second)
                }).collect(),
            }
        }, map_from_pairs_validator);

        add_primitive_with_validator!(eg, "map-get" [instance = get] = |xs: @MapContainer, x: #| -?> # { xs.data.get(&x).copied() }, map_get_validator);
        add_primitive_with_validator!(eg, "map-insert" [instance = insert] = |mut xs: @MapContainer, x: #, y: #| -> @MapContainer {{ xs.data.insert(x, y); xs }}, map_insert_validator);
        add_primitive!(eg, "map-remove" = |mut xs: @MapContainer (arc), x: # (self.key())                     | -> @MapContainer (arc) {{ xs.data.remove(&x);   xs }});

        add_primitive_with_validator!(eg, "map-length"       = |xs: @MapContainer (arc)| -> i64 { xs.data.len() as i64 }, map_length_validator);
        add_primitive_with_validator!(eg, "map-contains"     = |xs: @MapContainer (arc), x: # (self.key())| -?> () { ( xs.data.contains_key(&x)).then_some(()) }, map_contains_validator);
        add_primitive_with_validator!(eg, "map-not-contains" = |xs: @MapContainer (arc), x: # (self.key())| -?> () { (!xs.data.contains_key(&x)).then_some(()) }, map_not_contains_validator);
    }

    fn reconstruct_termdag(
        &self,
        _container_values: &ContainerValues,
        _value: Value,
        termdag: &mut TermDag,
        element_terms: Vec<TermId>,
    ) -> TermId {
        // Flat `(map-of k0 v0 k1 v1 ...)` in canonical key order, so proof
        // checking can reproduce it from terms alone (and the rebuild proof's
        // Congr indices are flat, like `set-of`/`vec-of`).
        normalize_map_term(termdag, &element_terms).expect("map elements come in key/value pairs")
    }

    fn rebuild_container_normalizer(&self) -> Option<(String, PrimitiveValidator)> {
        Some(("map-of".to_owned(), Arc::new(normalize_map_term)))
    }

    fn serialized_name(&self, _container_values: &ContainerValues, _: Value) -> String {
        "map-of".to_owned()
    }
}

/// The flat `map-of` constructor: takes alternating key/value arguments and
/// builds a map. Used as the canonical term form for maps (analogous to
/// `set-of`/`vec-of`). Needs a custom type constraint because its arguments
/// alternate between the key and value sorts.
#[derive(Clone)]
struct MapOf {
    instance: builtin::BuiltinInstance,
    key: ArcSort,
    value: ArcSort,
}

impl Primitive for MapOf {
    fn name(&self) -> &str {
        "map-of"
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        self.instance.vec_pair_facade_constraints(self.name(), span)
    }

    fn lower_builtin_call(
        &self,
        types: &TypeInfo,
        args: &[u32],
        output: u32,
        span: Option<proto::Span>,
        destination: &mut proto::Program,
    ) -> Result<Option<proto::Call>, String> {
        use proto as pb;
        let definition = &self.instance.definition;
        // Import the exact registration first, so a stale descriptor cannot be
        // silently replaced by another definition sharing its canonical key.
        let key = builtin::import_definition(definition, destination)?;
        if key != "egglog.core.map.of" {
            return Err("expected canonical Map constructor".into());
        }
        let [k, v, pair, vector, _] = self.instance.vec_pair_facade_pattern()?;
        let mut patterns = vec![];
        let [k, v, pair, vector] = [k, v, pair, vector].map(|i| {
            builtin::import_sort(&definition.sorts, i, &mut patterns).expect("validated definition")
        });
        let Some(pb::sort::Kind::Family(mut generic_vec)) = patterns[vector as usize].kind.clone()
        else {
            unreachable!()
        };
        generic_vec.args = vec![k];
        let generic_vec_index = patterns.len() as u32;
        patterns.push(pb::Sort {
            kind: Some(pb::sort::Kind::Family(generic_vec)),
            ..Default::default()
        });

        // Check complete helper signatures against the recipe's patterns, not
        // merely their keys. Labels/presentation do not affect these endpoints.
        let vector_key = if args.is_empty() {
            "egglog.core.vec.empty"
        } else {
            "egglog.core.vec.of"
        };
        for (helper_key, parameters, fixed, tail, result) in [
            ("egglog.core.pair.make", 2, vec![k, v], vec![], pair),
            (
                vector_key,
                1,
                vec![],
                if args.is_empty() { vec![] } else { vec![k] },
                generic_vec_index,
            ),
        ] {
            let helper = types
                .builtin_definitions
                .get(helper_key)
                .ok_or("missing Map packing helper")?;
            let primitive = helper
                .declarations
                .iter()
                .find_map(|d| match &d.kind {
                    Some(pb::declaration::Kind::HostPrimitive(p)) => Some(p),
                    _ => None,
                })
                .ok_or("missing helper definition")?;
            let signature = builtin::normalized_signature(helper, primitive, &mut patterns)?;
            if signature.type_params.len() != parameters
                || signature.inputs.iter().map(|a| a.sort).collect::<Vec<_>>() != fixed
                || signature.varargs.iter().map(|a| a.sort).collect::<Vec<_>>() != tail
                || signature.output != Some(result)
            {
                return Err(format!("incompatible Map packing helper {helper_key}"));
            }
            types.export_builtin_definition(helper_key, destination)?;
        }
        types.export_builtin_definition(&key, destination)?;
        let Some(pb::sort::Kind::Family(map)) = destination
            .sorts
            .get(output as usize)
            .and_then(|s| s.kind.as_ref())
        else {
            return Err("expected closed Map result".into());
        };
        if map.name != "Map" || map.args.len() != 2 || !args.len().is_multiple_of(2) {
            return Err("invalid flat Map specialization".into());
        }
        let parameters = map.args.clone();
        let mut normalized = vec![];
        let expected = parameters
            .iter()
            .map(|i| builtin::import_sort(&destination.sorts, *i, &mut normalized))
            .collect::<Result<Vec<_>, _>>()?;
        for (i, arg) in args.iter().enumerate() {
            let node = destination
                .nodes
                .get(*arg as usize)
                .ok_or("Map argument out of bounds")?;
            if builtin::import_sort(&destination.sorts, node.sort_id, &mut normalized)?
                != expected[i % 2]
            {
                return Err("flat Map argument differs from canonical component".into());
            }
        }
        // Close the actual Pair/Vec recipe with the specialization's output
        // parameters. This needs no native alias, factory, or runtime container.
        let mut closed = destination.sorts.clone();
        let pair_sort = closed.len() as u32;
        let Some(pb::sort::Kind::Family(mut pair_pattern)) = patterns[pair as usize].kind.clone()
        else {
            unreachable!()
        };
        pair_pattern.args = parameters;
        closed.push(pb::Sort {
            kind: Some(pb::sort::Kind::Family(pair_pattern)),
            ..Default::default()
        });
        let vector_sort = closed.len() as u32;
        let Some(pb::sort::Kind::Family(mut vector_pattern)) =
            patterns[vector as usize].kind.clone()
        else {
            unreachable!()
        };
        vector_pattern.args = vec![pair_sort];
        closed.push(pb::Sort {
            kind: Some(pb::sort::Kind::Family(vector_pattern)),
            ..Default::default()
        });
        let pair_sort = builtin::import_sort(&closed, pair_sort, &mut destination.sorts)?;
        let vector_sort = builtin::import_sort(&closed, vector_sort, &mut destination.sorts)?;
        let mut pairs = vec![];
        for fields in args.chunks_exact(2) {
            pairs.push(u32::try_from(destination.nodes.len()).map_err(|_| "too many nodes")?);
            destination.nodes.push(pb::Node {
                sort_id: pair_sort,
                kind: Some(pb::node::Kind::Call(pb::Call {
                    func: "egglog.core.pair.make".into(),
                    args: fields.to_vec(),
                })),
                span,
            });
        }
        let entries = u32::try_from(destination.nodes.len()).map_err(|_| "too many nodes")?;
        destination.nodes.push(pb::Node {
            sort_id: vector_sort,
            kind: Some(pb::node::Kind::Call(pb::Call {
                func: vector_key.into(),
                args: pairs,
            })),
            span,
        });
        Ok(Some(pb::Call {
            func: key,
            args: vec![entries],
        }))
    }
}

impl PurePrim for MapOf {
    fn apply<'a, 'db>(&self, mut state: PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let mut data = BTreeMap::new();
        for chunk in args.chunks(2) {
            if let [k, v] = chunk {
                data.insert(*k, *v);
            }
        }
        let mc = MapContainer {
            do_rebuild_keys: self.key.is_eq_sort() || self.key.is_eq_container_sort(),
            do_rebuild_vals: self.value.is_eq_sort() || self.value.is_eq_container_sort(),
            data,
        };
        Some(state.register_container(mc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_packing_rejects_changed_actual_descriptors_and_helper_shapes() {
        for (source, key) in [
            ("map-of", "egglog.core.map.of"),
            ("map-of", "egglog.core.pair.make"),
            ("map-of", "egglog.core.vec.of"),
            ("map-of", "egglog.core.vec.empty"),
            ("pair", "egglog.core.pair.make"),
        ] {
            let mut graph = EGraph::default();
            graph
                .parse_and_run_program(None, "(sort M (Map i64 i64)) (sort P (Pair i64 i64))")
                .unwrap();
            let i = graph.get_sort_by_name("i64").unwrap().clone();
            let m = graph
                .get_sort_by_name(if source == "pair" { "P" } else { "M" })
                .unwrap()
                .clone();
            let empty = key.ends_with("empty");
            let mut native = if empty {
                vec![]
            } else {
                vec![i.clone(), i.clone()]
            };
            native.push(m);
            let core::ResolvedCall::Primitive(primitive) = core::ResolvedCall::from_resolution(
                source,
                &native,
                &graph.type_info,
                Context::Pure,
                &Span::Panic,
            )
            .unwrap() else {
                unreachable!()
            };
            let definition =
                Arc::make_mut(graph.type_info.builtin_definitions.get_mut(key).unwrap());
            let primitive_definition = definition
                .declarations
                .iter_mut()
                .find_map(|d| match &mut d.kind {
                    Some(proto::declaration::Kind::HostPrimitive(p)) => Some(p),
                    _ => None,
                })
                .unwrap();
            let Some(proto::host_primitive::Typing::Signature(signature)) =
                &mut primitive_definition.typing
            else {
                unreachable!()
            };
            // Same key and binder count, wrong endpoint. For Pair, input
            // mutation also discriminates when both concrete arguments are i64.
            if key.ends_with("make") {
                signature.inputs[1].sort = signature.inputs[0].sort;
            } else {
                signature.output = Some(0);
            }
            if source == "pair" {
                // This is a valid standalone descriptor. Only the resolved
                // registration's original record makes the resupply conflict.
                graph
                    .type_info
                    .export_builtin_definition(key, &mut proto::Program::default())
                    .unwrap();
            }
            let mut destination = proto::Program::default();
            let sort_id = graph.export_sort(&i, &mut destination.sorts).unwrap();
            destination.nodes.push(proto::Node {
                sort_id,
                kind: Some(proto::node::Kind::Var("x".into())),
                ..Default::default()
            });
            let original = destination.nodes.clone();
            assert!(
                primitive
                    .export_builtin_call(
                        &graph,
                        if empty { &[] } else { &[0, 0] },
                        None,
                        &mut destination
                    )
                    .is_err(),
                "{key}"
            );
            assert_eq!(destination.nodes, original, "no helpers emitted for {key}");
        }
    }
}
