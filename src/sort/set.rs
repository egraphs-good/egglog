use super::*;
use crate::numeric_id::NumericId;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SetContainer {
    pub do_rebuild: bool,
    pub data: BTreeSet<Value>,
}

impl ContainerValue for SetContainer {
    fn encode_sequence(&self, _base_values: &BaseValues, out: &mut Vec<Value>) {
        out.push(Value::from_usize(self.do_rebuild as usize));
        out.extend(self.data.iter().copied());
    }

    fn decode_sequence(sequence: &[Value], _base_values: &BaseValues) -> Self {
        let (&header, data) = sequence
            .split_first()
            .expect("serialized SetContainer must include its rebuild flag");
        assert!(
            header == Value::from_usize(0) || header == Value::from_usize(1),
            "serialized SetContainer has an invalid rebuild flag"
        );
        debug_assert!(data.is_sorted());
        debug_assert!(data.windows(2).all(|pair| pair[0] != pair[1]));
        Self {
            do_rebuild: header == Value::from_usize(1),
            data: data.iter().copied().collect(),
        }
    }

    fn sequence_values(sequence: &[Value]) -> &[Value] {
        sequence
            .get(1..)
            .expect("serialized SetContainer must include its rebuild flag")
    }

    fn visit_sequence_values(sequence: &[Value], visitor: &mut dyn FnMut(Value)) {
        let (&header, data) = sequence
            .split_first()
            .expect("serialized SetContainer must include its rebuild flag");
        match header.index() {
            0 => {}
            1 => data.iter().copied().for_each(visitor),
            _ => panic!("serialized SetContainer has an invalid rebuild flag"),
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
            .expect("serialized SetContainer must include its rebuild flag");
        if header == Value::from_usize(0) {
            return false;
        }
        assert_eq!(
            header,
            Value::from_usize(1),
            "serialized SetContainer has an invalid rebuild flag"
        );

        let mut rebuilt = data.to_vec();
        if !rebuilder.rebuild_slice(&mut rebuilt) {
            return false;
        }
        rebuilt.sort_unstable();
        rebuilt.dedup();
        if rebuilt == data {
            return false;
        }
        out.push(header);
        out.extend(rebuilt);
        true
    }
}

/// The elements of a `(set-of e0 ...)` term as a Rust `BTreeSet` in AST
/// order, matching `SetContainer`'s semantics; `None` for any other term.
fn set_term_to_btreeset<'a>(termdag: &'a TermDag, term: TermId) -> Option<BTreeSet<OrdTerm<'a>>> {
    match termdag.get(term) {
        Term::App(head, children) if head == "set-of" => {
            Some(children.iter().map(|c| termdag.ord_term(*c)).collect())
        }
        _ => None,
    }
}

/// Flatten a set back to the element list of its canonical `(set-of ...)`
/// term (sorted by AST order and deduplicated by construction).
fn set_term_args(set: BTreeSet<OrdTerm<'_>>) -> Vec<TermId> {
    set.into_iter().map(|e| e.id()).collect()
}

/// Canonicalize `elements` to the `(set-of e0 e1 ...)` term form: sorted by
/// [`TermDag::ast_cmp`] and deduplicated, so proof checking can reproduce it.
fn normalize_set_term(termdag: &mut TermDag, elements: &[TermId]) -> TermId {
    let set: BTreeSet<_> = elements.iter().map(|e| termdag.ord_term(*e)).collect();
    let elements = set_term_args(set);
    termdag.app("set-of".into(), elements)
}

#[derive(Clone, Debug)]
pub struct SetSort {
    name: String,
    element: ArcSort,
}

impl SetSort {
    pub fn element(&self) -> ArcSort {
        self.element.clone()
    }
}

impl Presort for SetSort {
    fn presort_name() -> &'static str {
        "Set"
    }

    fn reserved_primitives() -> Vec<&'static str> {
        vec![
            "set-of",
            "set-empty",
            "set-insert",
            "set-not-contains",
            "set-contains",
            "set-remove",
            "set-union",
            "set-diff",
            "set-intersect",
            "set-get",
            "set-length",
        ]
    }

    fn make_sort(
        typeinfo: &mut TypeInfo,
        name: String,
        args: &[Expr],
        span: Span,
    ) -> Result<ArcSort, TypeError> {
        if let [Expr::Var(arg_span, e)] = args {
            let e = typeinfo
                .get_sort_by_name(e)
                .ok_or(TypeError::UndefinedSort(e.clone(), arg_span.clone()))?;

            let out = Self {
                name,
                element: e.clone(),
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

impl ContainerSort for SetSort {
    type Container = SetContainer;

    fn name(&self) -> &str {
        &self.name
    }

    fn inner_sorts(&self) -> Vec<ArcSort> {
        vec![self.element.clone()]
    }

    fn is_eq_container_sort(&self) -> bool {
        self.element.is_eq_sort() || self.element.is_eq_container_sort()
    }

    fn inner_values(
        &self,
        container_values: &ContainerValues,
        value: Value,
    ) -> Vec<(ArcSort, Value)> {
        let val = container_values
            .get_val::<SetContainer>(value)
            .unwrap()
            .clone();
        val.data
            .iter()
            .map(|e| (self.element.clone(), *e))
            .collect()
    }

    fn register_primitives(&self, eg: &mut EGraph) {
        let arc = self.clone().to_arcsort();

        // Proof term form of a set: `(set-of e0 e1 ...)` sorted and
        // deduplicated, matching `reconstruct_termdag`. Each validator
        // round-trips through a Rust `BTreeSet` (see `set_term_to_btreeset`),
        // so it evaluates set terms with `SetContainer`'s semantics.
        let set_of_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            Some(normalize_set_term(termdag, args))
        };
        let set_empty_validator = |termdag: &mut TermDag, _args: &[TermId]| -> Option<TermId> {
            Some(termdag.app("set-of".into(), vec![]))
        };
        let set_insert_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let [set, value] = args else {
                return None;
            };
            let mut set = set_term_to_btreeset(termdag, *set)?;
            set.insert(termdag.ord_term(*value));
            let elements = set_term_args(set);
            Some(termdag.app("set-of".into(), elements))
        };
        let set_remove_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let [set, value] = args else {
                return None;
            };
            let mut set = set_term_to_btreeset(termdag, *set)?;
            set.remove(&termdag.ord_term(*value));
            let elements = set_term_args(set);
            Some(termdag.app("set-of".into(), elements))
        };
        let set_length_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let [set] = args else {
                return None;
            };
            let len = set_term_to_btreeset(termdag, *set)?.len() as i64;
            Some(termdag.lit(Literal::Int(len)))
        };
        let set_contains_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let [set, value] = args else {
                return None;
            };
            let contains = set_term_to_btreeset(termdag, *set)?.contains(&termdag.ord_term(*value));
            contains.then(|| termdag.lit(Literal::Unit))
        };
        let set_not_contains_validator = |termdag: &mut TermDag,
                                          args: &[TermId]|
         -> Option<TermId> {
            let [set, value] = args else {
                return None;
            };
            let contains = set_term_to_btreeset(termdag, *set)?.contains(&termdag.ord_term(*value));
            (!contains).then(|| termdag.lit(Literal::Unit))
        };
        let set_union_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let [left, right] = args else {
                return None;
            };
            let mut set = set_term_to_btreeset(termdag, *left)?;
            set.extend(set_term_to_btreeset(termdag, *right)?);
            let elements = set_term_args(set);
            Some(termdag.app("set-of".into(), elements))
        };
        let set_diff_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let [left, right] = args else {
                return None;
            };
            let mut set = set_term_to_btreeset(termdag, *left)?;
            let right = set_term_to_btreeset(termdag, *right)?;
            set.retain(|e| !right.contains(e));
            let elements = set_term_args(set);
            Some(termdag.app("set-of".into(), elements))
        };
        let set_intersect_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let [left, right] = args else {
                return None;
            };
            let mut set = set_term_to_btreeset(termdag, *left)?;
            let right = set_term_to_btreeset(termdag, *right)?;
            set.retain(|e| right.contains(e));
            let elements = set_term_args(set);
            Some(termdag.app("set-of".into(), elements))
        };

        add_primitive_with_validator!(eg, "set-empty" = {self.clone(): SetSort} |                      | -> @SetContainer (arc) { SetContainer {
            do_rebuild: self.ctx.is_eq_container_sort(),
            data: BTreeSet::new()
        } }, set_empty_validator);
        add_primitive_with_validator!(eg, "set-of"    = {self.clone(): SetSort} [xs: # (self.element())] -> @SetContainer (arc) { SetContainer {
            do_rebuild: self.ctx.is_eq_container_sort(),
            data: xs.collect()
        } }, set_of_validator);

        // No validator: `set-get` indexes the runtime `Value` order, which
        // terms cannot reproduce, so it is unsupported in proof mode.
        eg.add_pure_primitive(
            SetRead {
                name: "set-get".into(),
                set: arc.clone(),
                element: self.element(),
                op: SetReadOp::Get,
            },
            None,
        );
        eg.add_pure_primitive(
            SetEdit {
                name: "set-insert".into(),
                set: arc.clone(),
                element: self.element(),
                op: SetEditOp::Insert,
            },
            Some(Arc::new(set_insert_validator)),
        );
        eg.add_pure_primitive(
            SetEdit {
                name: "set-remove".into(),
                set: arc.clone(),
                element: self.element(),
                op: SetEditOp::Remove,
            },
            Some(Arc::new(set_remove_validator)),
        );
        for (name, op, validator) in [
            (
                "set-length",
                SetReadOp::Length,
                Arc::new(set_length_validator) as PrimitiveValidator,
            ),
            (
                "set-contains",
                SetReadOp::Contains,
                Arc::new(set_contains_validator) as PrimitiveValidator,
            ),
            (
                "set-not-contains",
                SetReadOp::NotContains,
                Arc::new(set_not_contains_validator) as PrimitiveValidator,
            ),
        ] {
            eg.add_pure_primitive(
                SetRead {
                    name: name.into(),
                    set: arc.clone(),
                    element: self.element(),
                    op,
                },
                Some(validator),
            );
        }
        for (name, op, validator) in [
            (
                "set-union",
                SetBinaryOp::Union,
                Arc::new(set_union_validator) as PrimitiveValidator,
            ),
            (
                "set-diff",
                SetBinaryOp::Diff,
                Arc::new(set_diff_validator) as PrimitiveValidator,
            ),
            (
                "set-intersect",
                SetBinaryOp::Intersect,
                Arc::new(set_intersect_validator) as PrimitiveValidator,
            ),
        ] {
            eg.add_pure_primitive(
                SetBinary {
                    name: name.into(),
                    set: arc.clone(),
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
        // Canonical form (sorted by deterministic AST order, deduped) so proof
        // checking can reproduce it from terms alone.
        normalize_set_term(termdag, &element_terms)
    }

    fn rebuild_container_normalizer(&self) -> Option<(String, PrimitiveValidator)> {
        Some((
            "set-of".to_owned(),
            Arc::new(|termdag: &mut TermDag, args: &[TermId]| {
                Some(normalize_set_term(termdag, args))
            }),
        ))
    }

    fn serialized_name(&self, _container_values: &ContainerValues, _: Value) -> String {
        "set-of".to_owned()
    }
}

#[derive(Clone, Copy)]
enum SetReadOp {
    Get,
    Length,
    Contains,
    NotContains,
}

#[derive(Clone)]
struct SetRead {
    name: String,
    set: ArcSort,
    element: ArcSort,
    op: SetReadOp,
}

impl Primitive for SetRead {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        let types = match self.op {
            SetReadOp::Get => vec![self.set.clone(), I64Sort.to_arcsort(), self.element.clone()],
            SetReadOp::Length => vec![self.set.clone(), I64Sort.to_arcsort()],
            SetReadOp::Contains | SetReadOp::NotContains => vec![
                self.set.clone(),
                self.element.clone(),
                UnitSort.to_arcsort(),
            ],
        };
        SimpleTypeConstraint::new(self.name(), types, span.clone()).into_box()
    }
}

impl PurePrim for SetRead {
    fn apply<'a, 'db>(&self, state: crate::PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let [set_id, rest @ ..] = args else {
            return None;
        };
        match self.op {
            SetReadOp::Get => {
                let [index] = rest else { return None };
                let index = usize::try_from(state.base_values().unwrap::<i64>(*index)).ok()?;
                state
                    .with_container_sequence::<SetContainer, _>(*set_id, |values| {
                        values.get(index).copied()
                    })
                    .or_else(|| {
                        state
                            .value_to_owned_container::<SetContainer>(*set_id)
                            .map(|set| set.data.iter().nth(index).copied())
                    })?
            }
            SetReadOp::Length => {
                if !rest.is_empty() {
                    return None;
                }
                let len = state
                    .with_container_sequence::<SetContainer, _>(*set_id, <[Value]>::len)
                    .or_else(|| {
                        state
                            .value_to_owned_container::<SetContainer>(*set_id)
                            .map(|set| set.data.len())
                    })?;
                Some(state.base_values().get::<i64>(len as i64))
            }
            SetReadOp::Contains | SetReadOp::NotContains => {
                let [needle] = rest else { return None };
                let contains = state
                    .with_container_sequence::<SetContainer, _>(*set_id, |values| {
                        values.binary_search(needle).is_ok()
                    })
                    .or_else(|| {
                        state
                            .value_to_owned_container::<SetContainer>(*set_id)
                            .map(|set| set.data.contains(needle))
                    })?;
                let succeeds = match self.op {
                    SetReadOp::Contains => contains,
                    SetReadOp::NotContains => !contains,
                    _ => unreachable!(),
                };
                succeeds.then(|| state.base_values().get::<()>(()))
            }
        }
    }
}

#[derive(Clone, Copy)]
enum SetEditOp {
    Insert,
    Remove,
}

#[derive(Clone)]
struct SetEdit {
    name: String,
    set: ArcSort,
    element: ArcSort,
    op: SetEditOp,
}

impl Primitive for SetEdit {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.set.clone(), self.element.clone(), self.set.clone()],
            span.clone(),
        )
        .into_box()
    }
}

impl PurePrim for SetEdit {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let [set_id, needle] = args else { return None };
        let build_key = |values: &[Value]| {
            let mut key = Vec::with_capacity(values.len() + 2);
            key.push(Value::from_usize(self.set.is_eq_container_sort() as usize));
            match (self.op, values.binary_search(needle)) {
                (SetEditOp::Insert, Ok(_)) | (SetEditOp::Remove, Err(_)) => {
                    key.extend_from_slice(values);
                }
                (SetEditOp::Insert, Err(index)) => {
                    key.extend_from_slice(&values[..index]);
                    key.push(*needle);
                    key.extend_from_slice(&values[index..]);
                }
                (SetEditOp::Remove, Ok(index)) => {
                    key.extend_from_slice(&values[..index]);
                    key.extend_from_slice(&values[index + 1..]);
                }
            }
            key
        };
        let key = state
            .with_container_sequence::<SetContainer, _>(*set_id, build_key)
            .or_else(|| {
                state
                    .value_to_owned_container::<SetContainer>(*set_id)
                    .map(|set| build_key(&set.data.into_iter().collect::<Vec<_>>()))
            })?;
        Some(state.register_container_sequence::<SetContainer>(&key))
    }
}

#[derive(Clone, Copy)]
enum SetBinaryOp {
    Union,
    Diff,
    Intersect,
}

#[derive(Clone)]
struct SetBinary {
    name: String,
    set: ArcSort,
    op: SetBinaryOp,
}

impl Primitive for SetBinary {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.set.clone(), self.set.clone(), self.set.clone()],
            span.clone(),
        )
        .into_box()
    }
}

impl PurePrim for SetBinary {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let [left, right] = args else { return None };
        let read = |id| {
            state
                .with_container_sequence::<SetContainer, _>(id, <[Value]>::to_vec)
                .or_else(|| {
                    state
                        .value_to_owned_container::<SetContainer>(id)
                        .map(|set| set.data.into_iter().collect())
                })
        };
        let left = read(*left)?;
        let right = read(*right)?;
        let mut key = Vec::with_capacity(left.len() + right.len() + 1);
        key.push(Value::from_usize(self.set.is_eq_container_sort() as usize));
        merge_sets(&left, &right, self.op, &mut key);
        Some(state.register_container_sequence::<SetContainer>(&key))
    }
}

fn merge_sets(left: &[Value], right: &[Value], op: SetBinaryOp, out: &mut Vec<Value>) {
    let (mut l, mut r) = (0, 0);
    while l < left.len() && r < right.len() {
        match left[l].cmp(&right[r]) {
            std::cmp::Ordering::Less => {
                if matches!(op, SetBinaryOp::Union | SetBinaryOp::Diff) {
                    out.push(left[l]);
                }
                l += 1;
            }
            std::cmp::Ordering::Greater => {
                if matches!(op, SetBinaryOp::Union) {
                    out.push(right[r]);
                }
                r += 1;
            }
            std::cmp::Ordering::Equal => {
                if matches!(op, SetBinaryOp::Union | SetBinaryOp::Intersect) {
                    out.push(left[l]);
                }
                l += 1;
                r += 1;
            }
        }
    }
    if matches!(op, SetBinaryOp::Union | SetBinaryOp::Diff) {
        out.extend_from_slice(&left[l..]);
    }
    if matches!(op, SetBinaryOp::Union) {
        out.extend_from_slice(&right[r..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Collapse {
        from: Value,
        to: Value,
    }

    impl ValueRebuilder for Collapse {
        fn rebuild_val(&self, value: Value) -> Value {
            if value == self.from { self.to } else { value }
        }
    }

    fn value(index: usize) -> Value {
        Value::from_usize(index)
    }

    #[test]
    fn sequence_codec_rebuilds_sorts_and_collapses() {
        let set = SetContainer {
            do_rebuild: true,
            data: [value(2), value(4), value(6)].into_iter().collect(),
        };
        let mut encoded = Vec::new();
        set.encode_sequence(&BaseValues::default(), &mut encoded);
        assert_eq!(
            SetContainer::decode_sequence(&encoded, &BaseValues::default()),
            set
        );

        let mut rebuilt = Vec::new();
        assert!(SetContainer::rebuild_sequence(
            &encoded,
            &BaseValues::default(),
            &Collapse {
                from: value(6),
                to: value(2),
            },
            &mut rebuilt,
        ));
        assert_eq!(rebuilt, vec![value(1), value(2), value(4)]);
        assert_eq!(
            SetContainer::sequence_values(&rebuilt),
            &[value(2), value(4)]
        );
    }

    #[test]
    fn sequence_codec_skips_non_rebuildable_sets() {
        let encoded = vec![value(0), value(2), value(4)];
        let mut rebuilt = Vec::new();
        assert!(!SetContainer::rebuild_sequence(
            &encoded,
            &BaseValues::default(),
            &Collapse {
                from: value(2),
                to: value(4),
            },
            &mut rebuilt,
        ));
        assert!(rebuilt.is_empty());
    }

    #[test]
    fn sequence_primitives_consume_local_predictions() {
        let mut egraph = EGraph::default();
        egraph
            .parse_and_run_program(
                None,
                r#"
                (sort IntSet (Set i64))
                (check (= 4
                    (set-length
                        (set-intersect
                            (set-insert (set-union (set-of 3 1) (set-of 2 4)) 4)
                            (set-of 4 3 2 1 1)))))
                (check (= (set-of 1 3)
                    (set-diff (set-remove (set-of 1 2 3) 9) (set-of 2))))
                (check (set-contains (set-insert (set-empty) 7) 7))
                (check (set-not-contains (set-remove (set-of 7) 7) 7))
                "#,
            )
            .unwrap();
    }

    #[test]
    fn sorted_slice_algebra_matches_btree_sets_exhaustively() {
        for left_mask in 0u8..16 {
            for right_mask in 0u8..16 {
                let left = (0..4)
                    .filter(|bit| left_mask & (1 << bit) != 0)
                    .map(|bit| value(bit + 10))
                    .collect::<Vec<_>>();
                let right = (0..4)
                    .filter(|bit| right_mask & (1 << bit) != 0)
                    .map(|bit| value(bit + 10))
                    .collect::<Vec<_>>();
                let left_set = left.iter().copied().collect::<BTreeSet<_>>();
                let right_set = right.iter().copied().collect::<BTreeSet<_>>();
                for op in [
                    SetBinaryOp::Union,
                    SetBinaryOp::Diff,
                    SetBinaryOp::Intersect,
                ] {
                    let mut actual = Vec::new();
                    merge_sets(&left, &right, op, &mut actual);
                    let expected: Vec<_> = match op {
                        SetBinaryOp::Union => left_set.union(&right_set).copied().collect(),
                        SetBinaryOp::Diff => left_set.difference(&right_set).copied().collect(),
                        SetBinaryOp::Intersect => {
                            left_set.intersection(&right_set).copied().collect()
                        }
                    };
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}
