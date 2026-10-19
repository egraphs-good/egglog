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

    fn family_description() -> Option<builtin::FamilyDescription> {
        Some(builtin::FamilyDescription {
            name: "Map".into(),
            arity: 2,
        })
    }

    fn builtin_definitions() -> Vec<builtin::BuiltinDescription> {
        use builtin::{
            Argument, BuiltinDescription, BuiltinTyping, Signature, SortPattern, TypeParameter,
        };
        let key = SortPattern::Param(0);
        let value = SortPattern::Param(1);
        let pair = SortPattern::Family {
            name: "Pair".into(),
            args: vec![key.clone(), value.clone()],
        };
        let patterns = [
            key.clone(),
            value.clone(),
            SortPattern::Family {
                name: "Map".into(),
                args: vec![key, value],
            },
            pair.clone(),
            SortPattern::Family {
                name: "Vec".into(),
                args: vec![pair],
            },
            SortPattern::Family {
                name: "i64".into(),
                args: vec![],
            },
            SortPattern::Family {
                name: "Unit".into(),
                args: vec![],
            },
        ];
        let mut definitions: Vec<_> = [
            ("empty", "map-empty", vec![], 2),
            ("of", "map-from-pairs", vec![("entries", 4)], 2),
            ("get", "map-get", vec![("map", 2), ("key", 0)], 1),
            ("remove", "map-remove", vec![("map", 2), ("key", 0)], 2),
            ("length", "map-length", vec![("map", 2)], 5),
            ("contains", "map-contains", vec![("map", 2), ("key", 0)], 6),
            (
                "not_contains",
                "map-not-contains",
                vec![("map", 2), ("key", 0)],
                6,
            ),
            (
                "insert",
                "map-insert",
                vec![("map", 2), ("key", 0), ("value", 1)],
                2,
            ),
        ]
        .into_iter()
        .map(|(key, alias, inputs, output)| BuiltinDescription {
            key: format!("egglog.core.map.{key}"),
            aliases: vec![alias.into()],
            typing: BuiltinTyping::Signature(Signature {
                params: vec![
                    TypeParameter {
                        name: "K".into(),
                        equality: false,
                    },
                    TypeParameter {
                        name: "V".into(),
                        equality: false,
                    },
                ],
                fixed: inputs
                    .into_iter()
                    .map(|(name, sort)| Argument {
                        name: name.into(),
                        sort: patterns[sort].clone(),
                    })
                    .collect(),
                tail: vec![],
                output: patterns[output].clone(),
            }),
        })
        .collect();
        let mut flat = definitions
            .iter()
            .find(|d| d.key == "egglog.core.map.of")
            .unwrap()
            .clone();
        flat.key = "egglog.core.map.flat".into();
        flat.aliases = vec!["map-of".into()];
        let BuiltinTyping::Signature(signature) = &mut flat.typing else {
            unreachable!()
        };
        let SortPattern::Family { args, .. } = &signature.output else {
            unreachable!()
        };
        signature.fixed.clear();
        signature.tail = ["key", "value"]
            .into_iter()
            .zip(args)
            .map(|(name, sort)| Argument {
                name: name.into(),
                sort: sort.clone(),
            })
            .collect();
        definitions.push(flat);
        definitions
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
        typeinfo: &TypeInfo,
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

    fn family_instance(&self) -> Option<builtin::NativeFamilyInstance> {
        Some(builtin::NativeFamilyInstance {
            family: "Map".into(),
            parameters: vec![self.key.clone(), self.value.clone()],
        })
    }

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
        let empty = eg
            .type_info
            .instantiate_builtin("egglog.core.map.empty", &arc)
            .unwrap();
        let of = eg
            .type_info
            .instantiate_builtin("egglog.core.map.of", &arc)
            .unwrap();
        let flat = eg
            .type_info
            .instantiate_builtin("egglog.core.map.flat", &arc)
            .unwrap();
        let get = eg
            .type_info
            .instantiate_builtin("egglog.core.map.get", &arc)
            .unwrap();
        let insert = eg
            .type_info
            .instantiate_builtin("egglog.core.map.insert", &arc)
            .unwrap();
        let [remove, length, contains, not_contains] =
            ["remove", "length", "contains", "not_contains"].map(|key| {
                eg.type_info
                    .instantiate_builtin(&format!("egglog.core.map.{key}"), &arc)
                    .unwrap()
            });

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

        // Flat source/proof compatibility shares the canonical Map parameters.
        eg.add_pure_primitive(
            MapOf {
                instance: flat,
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
        add_primitive!(eg, "map-remove" [instance = remove] = |mut xs: @MapContainer, x: #| -> @MapContainer {{ xs.data.remove(&x);   xs }});

        add_primitive_with_validator!(eg, "map-length" [instance = length] = |xs: @MapContainer| -> i64 { xs.data.len() as i64 }, map_length_validator);
        add_primitive_with_validator!(eg, "map-contains" [instance = contains] = |xs: @MapContainer, x: #| -?> () { ( xs.data.contains_key(&x)).then_some(()) }, map_contains_validator);
        add_primitive_with_validator!(eg, "map-not-contains" [instance = not_contains] = |xs: @MapContainer, x: #| -?> () { (!xs.data.contains_key(&x)).then_some(()) }, map_not_contains_validator);
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
/// `set-of`/`vec-of`). Its native signature repeats key/value argument groups.
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

    fn builtin_instance(&self) -> Option<&builtin::BuiltinInstance> {
        Some(&self.instance)
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        self.instance.type_constraints(span)
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
