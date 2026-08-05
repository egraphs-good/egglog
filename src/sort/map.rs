use super::*;
use crate::numeric_id::NumericId;
use std::collections::BTreeMap;

const REBUILD_KEYS: usize = 1;
const REBUILD_VALS: usize = 1 << 1;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MapContainer {
    do_rebuild_keys: bool,
    do_rebuild_vals: bool,
    pub data: BTreeMap<Value, Value>,
}

impl ContainerValue for MapContainer {
    fn encode_sequence(&self, _base_values: &BaseValues, out: &mut Vec<Value>) {
        out.push(map_rebuild_mask(self.do_rebuild_keys, self.do_rebuild_vals));
        out.extend(self.data.iter().flat_map(|(key, value)| [*key, *value]));
    }

    fn decode_sequence(sequence: &[Value], _base_values: &BaseValues) -> Self {
        let (&mask, data) = sequence
            .split_first()
            .expect("serialized MapContainer must include its rebuild mask");
        let mask = checked_map_mask(mask);
        assert!(
            data.len().is_multiple_of(2),
            "serialized MapContainer must contain key/value pairs"
        );
        debug_assert!(
            data.chunks_exact(2)
                .map(|pair| pair[0])
                .is_sorted_by(|left, right| left < right),
            "serialized MapContainer keys must be sorted and unique"
        );
        Self {
            do_rebuild_keys: mask & REBUILD_KEYS != 0,
            do_rebuild_vals: mask & REBUILD_VALS != 0,
            data: data
                .chunks_exact(2)
                .map(|pair| (pair[0], pair[1]))
                .collect(),
        }
    }

    fn sequence_values(sequence: &[Value]) -> &[Value] {
        sequence
            .get(1..)
            .expect("serialized MapContainer must include its rebuild mask")
    }

    fn visit_sequence_values(sequence: &[Value], visitor: &mut dyn FnMut(Value)) {
        let (&mask, data) = sequence
            .split_first()
            .expect("serialized MapContainer must include its rebuild mask");
        let mask = checked_map_mask(mask);
        assert!(
            data.len().is_multiple_of(2),
            "serialized MapContainer must contain key/value pairs"
        );
        for pair in data.chunks_exact(2) {
            if mask & REBUILD_KEYS != 0 {
                visitor(pair[0]);
            }
            if mask & REBUILD_VALS != 0 {
                visitor(pair[1]);
            }
        }
    }

    fn rebuild_sequence(
        sequence: &[Value],
        _base_values: &BaseValues,
        rebuilder: &dyn ValueRebuilder,
        out: &mut Vec<Value>,
    ) -> bool {
        let (&header, data) = sequence
            .split_first()
            .expect("serialized MapContainer must include its rebuild mask");
        let mask = checked_map_mask(header);
        assert!(
            data.len().is_multiple_of(2),
            "serialized MapContainer must contain key/value pairs"
        );
        if mask == 0 {
            return false;
        }

        // Rebuild keys before values. Iteration is in ascending old-key order,
        // so inserting rebuilt keys in that order preserves BTreeMap's existing
        // collision rule: when two old keys canonicalize together, the later
        // old key's value wins.
        let mut rebuilt = BTreeMap::new();
        for pair in data.chunks_exact(2) {
            let key = if mask & REBUILD_KEYS != 0 {
                rebuilder.rebuild_val(pair[0])
            } else {
                pair[0]
            };
            rebuilt.insert(key, pair[1]);
        }
        if mask & REBUILD_VALS != 0 {
            for value in rebuilt.values_mut() {
                *value = rebuilder.rebuild_val(*value);
            }
        }

        let rebuilt_data = rebuilt
            .iter()
            .flat_map(|(key, value)| [*key, *value])
            .collect::<Vec<_>>();
        if rebuilt_data == data {
            return false;
        }
        out.push(header);
        out.extend(rebuilt_data);
        true
    }
}

fn map_rebuild_mask(rebuild_keys: bool, rebuild_vals: bool) -> Value {
    Value::from_usize(
        usize::from(rebuild_keys) * REBUILD_KEYS + usize::from(rebuild_vals) * REBUILD_VALS,
    )
}

fn checked_map_mask(mask: Value) -> usize {
    let mask = mask.index();
    assert!(
        mask & !(REBUILD_KEYS | REBUILD_VALS) == 0,
        "serialized MapContainer has an invalid rebuild mask"
    );
    mask
}

/// Binary-search canonical alternating `[key, value, ...]` storage.
///
/// The returned index counts pairs rather than individual values.
fn find_map_key(data: &[Value], needle: Value) -> Result<usize, usize> {
    assert!(
        data.len().is_multiple_of(2),
        "serialized MapContainer must contain key/value pairs"
    );
    let (mut low, mut high) = (0, data.len() / 2);
    while low < high {
        let mid = low + (high - low) / 2;
        match data[mid * 2].cmp(&needle) {
            std::cmp::Ordering::Less => low = mid + 1,
            std::cmp::Ordering::Greater => high = mid,
            std::cmp::Ordering::Equal => return Ok(mid),
        }
    }
    Err(low)
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
    fn presort_name() -> &'static str {
        "Map"
    }

    fn reserved_primitives() -> Vec<&'static str> {
        vec![
            "map-empty",
            "map-of",
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

        // The proof "term form" of a map is the flat `(map-of k0 v0 k1 v1 ...)`
        // in canonical key order (like `set-of`/`vec-of`), matching
        // `reconstruct_termdag`. Each validator round-trips through a Rust
        // `BTreeMap` (see `map_term_to_btreemap`), so it evaluates map terms
        // with `MapContainer`'s semantics; `None` for a malformed map term
        // fails the proof.
        let map_empty_validator = |termdag: &mut TermDag, _args: &[TermId]| -> Option<TermId> {
            Some(termdag.app("map-of".into(), vec![]))
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

        eg.add_pure_primitive(
            MapEmpty {
                name: "map-empty".to_string(),
                map: arc.clone(),
                rebuild_mask: map_rebuild_mask(
                    self.key.is_eq_sort() || self.key.is_eq_container_sort(),
                    self.value.is_eq_sort() || self.value.is_eq_container_sort(),
                ),
            },
            Some(Arc::new(map_empty_validator)),
        );

        // `map-of` is the flat constructor used as the canonical term form. It
        // takes alternating key/value arguments, so it needs a custom type
        // constraint rather than the `add_primitive!` macro.
        eg.add_pure_primitive(
            MapOf {
                name: "map-of".to_string(),
                map: arc.clone(),
                key: self.key.clone(),
                value: self.value.clone(),
            },
            Some(std::sync::Arc::new(normalize_map_term)),
        );

        eg.add_pure_primitive(
            MapRead {
                name: "map-get".into(),
                map: arc.clone(),
                key: self.key(),
                value: self.value(),
                op: MapReadOp::Get,
            },
            Some(Arc::new(map_get_validator)),
        );
        eg.add_pure_primitive(
            MapEdit {
                name: "map-insert".into(),
                map: arc.clone(),
                key: self.key(),
                value: self.value(),
                rebuild_mask: map_rebuild_mask(
                    self.key.is_eq_sort() || self.key.is_eq_container_sort(),
                    self.value.is_eq_sort() || self.value.is_eq_container_sort(),
                ),
                op: MapEditOp::Insert,
            },
            Some(Arc::new(map_insert_validator)),
        );
        eg.add_pure_primitive(
            MapEdit {
                name: "map-remove".into(),
                map: arc.clone(),
                key: self.key(),
                value: self.value(),
                rebuild_mask: map_rebuild_mask(
                    self.key.is_eq_sort() || self.key.is_eq_container_sort(),
                    self.value.is_eq_sort() || self.value.is_eq_container_sort(),
                ),
                op: MapEditOp::Remove,
            },
            None,
        );
        for (name, op, validator) in [
            (
                "map-length",
                MapReadOp::Length,
                Arc::new(map_length_validator) as PrimitiveValidator,
            ),
            (
                "map-contains",
                MapReadOp::Contains,
                Arc::new(map_contains_validator) as PrimitiveValidator,
            ),
            (
                "map-not-contains",
                MapReadOp::NotContains,
                Arc::new(map_not_contains_validator) as PrimitiveValidator,
            ),
        ] {
            eg.add_pure_primitive(
                MapRead {
                    name: name.into(),
                    map: arc.clone(),
                    key: self.key(),
                    value: self.value(),
                    op,
                },
                Some(validator),
            );
        }
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

#[derive(Clone)]
struct MapEmpty {
    name: String,
    map: ArcSort,
    rebuild_mask: Value,
}

impl Primitive for MapEmpty {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(self.name(), vec![self.map.clone()], span.clone()).into_box()
    }
}

impl PurePrim for MapEmpty {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        if !args.is_empty() {
            return None;
        }
        Some(state.register_container_sequence::<MapContainer>(&[self.rebuild_mask]))
    }
}

#[derive(Clone, Copy)]
enum MapReadOp {
    Get,
    Length,
    Contains,
    NotContains,
}

/// Map reads use binary search over the borrowed canonical sequence, with an
/// explicit decode-based slow fallback.
#[derive(Clone)]
struct MapRead {
    name: String,
    map: ArcSort,
    key: ArcSort,
    value: ArcSort,
    op: MapReadOp,
}

impl Primitive for MapRead {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        let types = match self.op {
            MapReadOp::Get => vec![self.map.clone(), self.key.clone(), self.value.clone()],
            MapReadOp::Length => vec![self.map.clone(), I64Sort.to_arcsort()],
            MapReadOp::Contains | MapReadOp::NotContains => {
                vec![self.map.clone(), self.key.clone(), UnitSort.to_arcsort()]
            }
        };
        SimpleTypeConstraint::new(self.name(), types, span.clone()).into_box()
    }
}

impl PurePrim for MapRead {
    fn apply<'a, 'db>(&self, state: crate::PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let [map_id, rest @ ..] = args else {
            return None;
        };
        match self.op {
            MapReadOp::Get => {
                let [needle] = rest else { return None };
                state
                    .with_container_sequence::<MapContainer, _>(*map_id, |data| {
                        find_map_key(data, *needle)
                            .ok()
                            .map(|pair_index| data[pair_index * 2 + 1])
                    })
                    .or_else(|| {
                        state
                            .value_to_owned_container::<MapContainer>(*map_id)
                            .map(|map| map.data.get(needle).copied())
                    })?
            }
            MapReadOp::Length => {
                if !rest.is_empty() {
                    return None;
                }
                let len = state
                    .with_container_sequence::<MapContainer, _>(*map_id, |data| {
                        assert!(data.len().is_multiple_of(2));
                        data.len() / 2
                    })
                    .or_else(|| {
                        state
                            .value_to_owned_container::<MapContainer>(*map_id)
                            .map(|map| map.data.len())
                    })?;
                Some(state.base_values().get::<i64>(len as i64))
            }
            MapReadOp::Contains | MapReadOp::NotContains => {
                let [needle] = rest else { return None };
                let contains = state
                    .with_container_sequence::<MapContainer, _>(*map_id, |data| {
                        find_map_key(data, *needle).is_ok()
                    })
                    .or_else(|| {
                        state
                            .value_to_owned_container::<MapContainer>(*map_id)
                            .map(|map| map.data.contains_key(needle))
                    })?;
                let succeeds = match self.op {
                    MapReadOp::Contains => contains,
                    MapReadOp::NotContains => !contains,
                    _ => unreachable!(),
                };
                succeeds.then(|| state.base_values().get::<()>(()))
            }
        }
    }
}

#[derive(Clone, Copy)]
enum MapEditOp {
    Insert,
    Remove,
}

/// Map updates splice the borrowed alternating sequence directly, with an
/// explicit decode/edit/encode slow fallback.
#[derive(Clone)]
struct MapEdit {
    name: String,
    map: ArcSort,
    key: ArcSort,
    value: ArcSort,
    rebuild_mask: Value,
    op: MapEditOp,
}

impl Primitive for MapEdit {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        let types = match self.op {
            MapEditOp::Insert => vec![
                self.map.clone(),
                self.key.clone(),
                self.value.clone(),
                self.map.clone(),
            ],
            MapEditOp::Remove => vec![self.map.clone(), self.key.clone(), self.map.clone()],
        };
        SimpleTypeConstraint::new(self.name(), types, span.clone()).into_box()
    }
}

impl PurePrim for MapEdit {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let map_id = *args.first()?;
        let build_key = |data: &[Value]| -> Option<Vec<Value>> {
            let needle = *args.get(1)?;
            let found = find_map_key(data, needle);
            let mut key = Vec::with_capacity(data.len() + 3);
            key.push(self.rebuild_mask);
            match (self.op, found) {
                (MapEditOp::Insert, Ok(pair_index)) => {
                    let [_, _, value] = args else { return None };
                    let value_index = pair_index * 2 + 1;
                    key.extend_from_slice(&data[..value_index]);
                    key.push(*value);
                    key.extend_from_slice(&data[value_index + 1..]);
                }
                (MapEditOp::Insert, Err(pair_index)) => {
                    let [_, _, value] = args else { return None };
                    let value_index = pair_index * 2;
                    key.extend_from_slice(&data[..value_index]);
                    key.extend_from_slice(&[needle, *value]);
                    key.extend_from_slice(&data[value_index..]);
                }
                (MapEditOp::Remove, Ok(pair_index)) => {
                    let [_, _] = args else { return None };
                    let value_index = pair_index * 2;
                    key.extend_from_slice(&data[..value_index]);
                    key.extend_from_slice(&data[value_index + 2..]);
                }
                (MapEditOp::Remove, Err(_)) => {
                    let [_, _] = args else { return None };
                    key.extend_from_slice(data);
                }
            }
            Some(key)
        };

        if let Some(key) = state.with_container_sequence::<MapContainer, _>(map_id, build_key) {
            return Some(state.register_container_sequence::<MapContainer>(&key?));
        }

        // Slow compatibility path: reconstruct the Rust map, perform the
        // operation, and let normal container registration serialize it again.
        let mut map = state.value_to_owned_container::<MapContainer>(map_id)?;
        match self.op {
            MapEditOp::Insert => {
                let [_, key, value] = args else { return None };
                map.data.insert(*key, *value);
            }
            MapEditOp::Remove => {
                let [_, key] = args else { return None };
                map.data.remove(key);
            }
        }
        Some(state.register_container(map))
    }
}

/// The flat `map-of` constructor: takes alternating key/value arguments and
/// builds a map. Used as the canonical term form for maps (analogous to
/// `set-of`/`vec-of`). Needs a custom type constraint because its arguments
/// alternate between the key and value sorts.
#[derive(Clone)]
struct MapOf {
    name: String,
    map: ArcSort,
    key: ArcSort,
    value: ArcSort,
}

impl Primitive for MapOf {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        Box::new(MapOfTypeConstraint {
            name: self.name.clone(),
            key: self.key.clone(),
            value: self.value.clone(),
            map: self.map.clone(),
            span: span.clone(),
        })
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
        let mut key = Vec::with_capacity(data.len() * 2 + 1);
        key.push(map_rebuild_mask(
            self.key.is_eq_sort() || self.key.is_eq_container_sort(),
            self.value.is_eq_sort() || self.value.is_eq_container_sort(),
        ));
        key.extend(data.into_iter().flat_map(|(key, value)| [key, value]));
        Some(state.register_container_sequence::<MapContainer>(&key))
    }
}

/// Type constraint for [`MapOf`]: an even number of inputs alternating between
/// the key and value sorts, producing the map sort.
struct MapOfTypeConstraint {
    name: String,
    key: ArcSort,
    value: ArcSort,
    map: ArcSort,
    span: Span,
}

impl TypeConstraint for MapOfTypeConstraint {
    fn get(
        &self,
        arguments: &[AtomTerm],
        _typeinfo: &TypeInfo,
    ) -> Vec<Box<dyn Constraint<AtomTerm, ArcSort>>> {
        let arity_mismatch = |expected: usize| {
            vec![constraint::impossible(
                constraint::ImpossibleConstraint::ArityMismatch {
                    atom: Atom {
                        span: self.span.clone(),
                        head: self.name.clone(),
                        args: arguments.to_vec(),
                    },
                    expected,
                },
            )]
        };
        let Some((out, inputs)) = arguments.split_last() else {
            return arity_mismatch(1);
        };
        if inputs.len() % 2 != 0 {
            return arity_mismatch(inputs.len() + 2);
        }
        let mut cs: Vec<Box<dyn Constraint<AtomTerm, ArcSort>>> =
            vec![constraint::assign(out.clone(), self.map.clone())];
        for (i, arg) in inputs.iter().enumerate() {
            let sort = if i % 2 == 0 {
                self.key.clone()
            } else {
                self.value.clone()
            };
            cs.push(constraint::assign(arg.clone(), sort));
        }
        cs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Remap(Vec<(Value, Value)>);

    impl ValueRebuilder for Remap {
        fn rebuild_val(&self, value: Value) -> Value {
            self.0
                .iter()
                .find_map(|(from, to)| (*from == value).then_some(*to))
                .unwrap_or(value)
        }
    }

    fn value(index: usize) -> Value {
        Value::from_usize(index)
    }

    fn map(
        do_rebuild_keys: bool,
        do_rebuild_vals: bool,
        entries: &[(usize, usize)],
    ) -> MapContainer {
        MapContainer {
            do_rebuild_keys,
            do_rebuild_vals,
            data: entries
                .iter()
                .map(|(key, value)| (self::value(*key), self::value(*value)))
                .collect(),
        }
    }

    #[test]
    fn sequence_codec_round_trips_and_indexes_only_enabled_lanes() {
        for (rebuild_keys, rebuild_vals, expected_children) in [
            (false, false, vec![]),
            (true, false, vec![value(2), value(4)]),
            (false, true, vec![value(20), value(40)]),
            (true, true, vec![value(2), value(20), value(4), value(40)]),
        ] {
            let map = map(rebuild_keys, rebuild_vals, &[(2, 20), (4, 40)]);
            let mut encoded = Vec::new();
            map.encode_sequence(&BaseValues::default(), &mut encoded);
            assert_eq!(
                MapContainer::decode_sequence(&encoded, &BaseValues::default()),
                map
            );
            assert_eq!(
                MapContainer::sequence_values(&encoded),
                &[value(2), value(20), value(4), value(40)]
            );

            let mut children = Vec::new();
            MapContainer::visit_sequence_values(&encoded, &mut |child| children.push(child));
            assert_eq!(children, expected_children);
        }
    }

    #[test]
    fn sequence_rebuild_preserves_last_old_key_on_collision() {
        let original = map(true, true, &[(2, 20), (4, 40), (6, 60)]);
        let rebuilder = Remap(vec![
            (value(2), value(1)),
            (value(4), value(1)),
            (value(6), value(3)),
            (value(40), value(41)),
            (value(60), value(61)),
        ]);

        let mut encoded = Vec::new();
        original.encode_sequence(&BaseValues::default(), &mut encoded);
        let mut rebuilt = Vec::new();
        assert!(MapContainer::rebuild_sequence(
            &encoded,
            &BaseValues::default(),
            &rebuilder,
            &mut rebuilt
        ));
        assert_eq!(
            MapContainer::decode_sequence(&rebuilt, &BaseValues::default()),
            map(true, true, &[(1, 41), (3, 61)])
        );
    }

    #[test]
    fn sequence_rebuild_respects_mask_and_false_leaves_output_empty() {
        let original = map(false, true, &[(2, 20), (4, 40)]);
        let mut encoded = Vec::new();
        original.encode_sequence(&BaseValues::default(), &mut encoded);
        let mut rebuilt = Vec::new();
        assert!(MapContainer::rebuild_sequence(
            &encoded,
            &BaseValues::default(),
            &Remap(vec![(value(2), value(3)), (value(20), value(21))]),
            &mut rebuilt,
        ));
        assert_eq!(
            MapContainer::decode_sequence(&rebuilt, &BaseValues::default()),
            map(false, true, &[(2, 21), (4, 40)])
        );

        let mut unchanged = Vec::new();
        assert!(!MapContainer::rebuild_sequence(
            &rebuilt,
            &BaseValues::default(),
            &Remap(vec![(value(2), value(3))]),
            &mut unchanged,
        ));
        assert!(unchanged.is_empty());

        let mut non_rebuildable = Vec::new();
        map(false, false, &[(2, 20)]).encode_sequence(&BaseValues::default(), &mut non_rebuildable);
        assert!(!MapContainer::rebuild_sequence(
            &non_rebuildable,
            &BaseValues::default(),
            &Remap(vec![(value(2), value(3)), (value(20), value(21))]),
            &mut unchanged,
        ));
        assert!(unchanged.is_empty());
    }

    #[test]
    fn binary_search_reports_existing_and_insertion_pair_indices() {
        let data = [
            value(2),
            value(20),
            value(4),
            value(40),
            value(8),
            value(80),
        ];
        assert_eq!(find_map_key(&data, value(2)), Ok(0));
        assert_eq!(find_map_key(&data, value(4)), Ok(1));
        assert_eq!(find_map_key(&data, value(8)), Ok(2));
        assert_eq!(find_map_key(&data, value(1)), Err(0));
        assert_eq!(find_map_key(&data, value(3)), Err(1));
        assert_eq!(find_map_key(&data, value(7)), Err(2));
        assert_eq!(find_map_key(&data, value(9)), Err(3));
    }

    #[test]
    fn sequence_primitives_use_predictions_and_preserve_last_write_wins() {
        let mut egraph = EGraph::default();
        egraph
            .parse_and_run_program(
                None,
                r#"
                (sort IntMap (Map i64 i64))
                (let m
                    (map-insert
                        (map-insert
                            (map-insert (map-empty) 4 40)
                            2 20)
                        4 41))
                (check (= 2 (map-length m)))
                (check (= 20 (map-get m 2)))
                (check (= 41 (map-get m 4)))
                (check (map-contains m 2))
                (check (map-not-contains m 3))
                (check (= 1 (map-length (map-remove m 2))))
                (check (= m (map-remove m 99)))
                (check (= 41 (map-get (map-of 4 40 2 20 4 41) 4)))
                "#,
            )
            .unwrap();
    }
}
