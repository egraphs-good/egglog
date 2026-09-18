use super::*;
use crate::Write;
use crate::exec_state::Internal;
use crate::numeric_id::NumericId;
use inner::MultiSet;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MultiSetContainer {
    pub do_rebuild: bool,
    pub data: MultiSet<Value>,
}

const MULTISET_ENTRY_WIDTH: usize = 3;

fn encode_count(out: &mut Vec<Value>, count: usize) {
    let count = count as u64;
    out.push(Value::from_usize(count as u32 as usize));
    out.push(Value::from_usize((count >> 32) as u32 as usize));
}

fn decode_count(encoded: &[Value]) -> usize {
    let [low, high] = encoded else {
        panic!("encoded multiset count must have two lanes");
    };
    let count = low.index() as u64 | ((high.index() as u64) << 32);
    usize::try_from(count).expect("encoded multiset count exceeds usize")
}

fn encoded_multiset_entries(
    payload: &[Value],
) -> impl ExactSizeIterator<Item = (Value, usize)> + '_ {
    assert_eq!(
        payload.len() % MULTISET_ENTRY_WIDTH,
        0,
        "serialized MultiSetContainer has a partial entry"
    );
    payload.chunks_exact(MULTISET_ENTRY_WIDTH).map(|entry| {
        let count = decode_count(&entry[1..]);
        assert!(count > 0, "serialized multiset count must be positive");
        (entry[0], count)
    })
}

fn encoded_multiset_count(payload: &[Value], needle: Value) -> usize {
    assert_eq!(
        payload.len() % MULTISET_ENTRY_WIDTH,
        0,
        "serialized MultiSetContainer has a partial entry"
    );
    let (mut low, mut high) = (0, payload.len() / MULTISET_ENTRY_WIDTH);
    while low < high {
        let middle = low + (high - low) / 2;
        let offset = middle * MULTISET_ENTRY_WIDTH;
        match payload[offset].cmp(&needle) {
            std::cmp::Ordering::Less => low = middle + 1,
            std::cmp::Ordering::Greater => high = middle,
            std::cmp::Ordering::Equal => {
                return decode_count(&payload[offset + 1..offset + MULTISET_ENTRY_WIDTH]);
            }
        }
    }
    0
}

fn encoded_multiset_len(payload: &[Value]) -> Option<usize> {
    encoded_multiset_entries(payload).try_fold(0usize, |total, (_, count)| {
        let total = total.checked_add(count)?;
        i64::try_from(total).ok()?;
        Some(total)
    })
}

/// Canonical multiset term form `(multiset-of e0 e1 ...)`: elements sorted by
/// [`TermDag::ast_cmp`] with multiplicities kept as repeats, so proof checking
/// can reproduce it from terms.
fn normalize_multiset_term(termdag: &mut TermDag, mut children: Vec<TermId>) -> TermId {
    termdag.sort_terms_by_ast(&mut children);
    termdag.app("multiset-of".into(), children)
}

/// The element terms of a multiset's canonical term form `(multiset-of e0 …)`
/// (multiplicities kept as repeats); `None` for any other term.
fn multiset_term_children(termdag: &TermDag, term: TermId) -> Option<Vec<TermId>> {
    match termdag.get(term) {
        Term::App(head, children) if head == "multiset-of" => Some(children.clone()),
        _ => None,
    }
}

impl ContainerValue for MultiSetContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn ValueRebuilder) -> bool {
        // If the contents are an eq-sort then we want to rebuild
        if self.do_rebuild {
            let mut xs: Vec<_> = self.data.iter().copied().collect();
            let changed = rebuilder.rebuild_slice(&mut xs);
            self.data = xs.into_iter().collect();
            changed
        // if the contents are just a primitive then don't need to do anything.
        } else {
            false
        }
    }
    fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        self.data.iter().copied()
    }
}

impl SequenceContainerValue for MultiSetContainer {
    fn encode_sequence(&self, out: &mut Vec<Value>) {
        assert!(
            i64::try_from(self.data.len()).is_ok(),
            "multiset cardinality exceeds the i64 language representation"
        );
        out.push(Value::from_usize(self.do_rebuild as usize));
        for (value, count) in self.data.iter_counts() {
            out.push(value);
            encode_count(out, count);
        }
    }

    fn decode_sequence(sequence: &[Value]) -> Self {
        let (&header, data) = sequence
            .split_first()
            .expect("serialized MultiSetContainer must include its rebuild flag");
        assert!(
            header == Value::from_usize(0) || header == Value::from_usize(1),
            "serialized MultiSetContainer has an invalid rebuild flag"
        );
        assert!(
            encoded_multiset_len(data).is_some(),
            "serialized multiset cardinality exceeds the i64 language representation"
        );
        let mut decoded = MultiSet::new();
        let mut previous = None;
        for (value, count) in encoded_multiset_entries(data) {
            debug_assert!(previous.is_none_or(|previous| previous < value));
            previous = Some(value);
            decoded
                .checked_insert_multiple_mut(value, count)
                .expect("serialized multiset cardinality overflow");
        }
        Self {
            do_rebuild: header == Value::from_usize(1),
            data: decoded,
        }
    }

    fn sequence_values(sequence: &[Value]) -> &[Value] {
        sequence
            .get(1..)
            .expect("serialized MultiSetContainer must include its rebuild flag")
    }

    fn visit_sequence_values(sequence: &[Value], visitor: &mut dyn FnMut(Value)) {
        let (&header, data) = sequence
            .split_first()
            .expect("serialized MultiSetContainer must include its rebuild flag");
        match header.index() {
            0 => {}
            1 => {
                for (value, _) in encoded_multiset_entries(data) {
                    visitor(value);
                }
            }
            _ => panic!("serialized MultiSetContainer has an invalid rebuild flag"),
        }
    }

    fn rebuild_sequence(
        sequence: &[Value],
        rebuilder: &dyn ValueRebuilder,
        out: &mut Vec<Value>,
    ) -> bool {
        let (&header, data) = sequence
            .split_first()
            .expect("serialized MultiSetContainer must include its rebuild flag");
        if header == Value::from_usize(0) {
            return false;
        }
        assert_eq!(
            header,
            Value::from_usize(1),
            "serialized MultiSetContainer has an invalid rebuild flag"
        );

        let entries = encoded_multiset_entries(data).collect::<Vec<_>>();
        let mut rebuilt_values = entries.iter().map(|(value, _)| *value).collect::<Vec<_>>();
        if !rebuilder.rebuild_slice(&mut rebuilt_values) {
            return false;
        }
        let mut rebuilt = MultiSet::new();
        for (rebuilt_value, (_, count)) in rebuilt_values.into_iter().zip(entries) {
            rebuilt
                .checked_insert_multiple_mut(rebuilt_value, count)
                .expect("rebuilding a valid multiset cannot increase its cardinality");
        }
        Self {
            do_rebuild: true,
            data: rebuilt,
        }
        .encode_sequence(out);
        if out.as_slice() == sequence {
            out.clear();
            return false;
        }
        true
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
            "multiset-single",
            "multiset-insert",
            "multiset-remove",
            "multiset-remove-swapped",
            "multiset-subtract",
            "multiset-subtract-swapped",
            "multiset-length",
            "multiset-contains",
            "multiset-not-contains",
            "multiset-contains-swapped",
            "multiset-not-contains-swapped",
            "multiset-intersection",
            "multiset-sum",
            "multiset-reset-counts",
            "multiset-pick-max",
            "multiset-count",
            "multiset-sum-multisets",
            "unstable-multiset-map",
            "unstable-multiset-filter",
            "unstable-multiset-filter-not",
            "unstable-multiset-reduce",
            "unstable-multiset-fill-index",
            "unstable-multiset-clear-index",
            "unstable-multiset-flat-map",
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

impl ContainerSort for MultiSetSort {
    type Container = MultiSetContainer;

    fn name(&self) -> &str {
        &self.name
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend.register_sequence_container_ty::<MultiSetContainer>();
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

        // Proof term form of a multiset: `(multiset-of e0 e1 ...)`, matching
        // `reconstruct_termdag`. (Count merging for proof checking of
        // collapsing multisets is refined in the MultiSet proof stage.)
        let multiset_of_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            Some(normalize_multiset_term(termdag, args.to_vec()))
        };
        let multiset_length_validator =
            |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
                let [ms] = args else { return None };
                let len = multiset_term_children(termdag, *ms)?.len() as i64;
                Some(termdag.lit(Literal::Int(len)))
            };
        let multiset_contains_validator =
            |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
                let [ms, value] = args else { return None };
                multiset_term_children(termdag, *ms)?
                    .contains(value)
                    .then(|| termdag.lit(Literal::Unit))
            };
        let multiset_not_contains_validator =
            |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
                let [ms, value] = args else { return None };
                let contains = multiset_term_children(termdag, *ms)?.contains(value);
                (!contains).then(|| termdag.lit(Literal::Unit))
            };

        add_primitive_with_validator!(eg, "multiset-of" = {self.clone(): MultiSetSort} [xs: # (self.element())] -> @MultiSetContainer (arc) { MultiSetContainer {
            do_rebuild: self.ctx.is_eq_container_sort(),
            data: xs.collect()
        } }, multiset_of_validator);

        eg.add_pure_primitive(
            MultiSetSingle {
                name: "multiset-single".into(),
                multiset: arc.clone(),
                element: self.element(),
            },
            None,
        );
        for (name, op, validator) in [
            ("multiset-pick", MultiSetReadOp::Pick, None),
            (
                "multiset-length",
                MultiSetReadOp::Length,
                Some(Arc::new(multiset_length_validator) as PrimitiveValidator),
            ),
            (
                "multiset-contains",
                MultiSetReadOp::Contains,
                Some(Arc::new(multiset_contains_validator) as PrimitiveValidator),
            ),
            (
                "multiset-not-contains",
                MultiSetReadOp::NotContains,
                Some(Arc::new(multiset_not_contains_validator) as PrimitiveValidator),
            ),
            (
                "multiset-contains-swapped",
                MultiSetReadOp::ContainsSwapped,
                None,
            ),
            (
                "multiset-not-contains-swapped",
                MultiSetReadOp::NotContainsSwapped,
                None,
            ),
            ("multiset-pick-max", MultiSetReadOp::PickMax, None),
            ("multiset-count", MultiSetReadOp::Count, None),
        ] {
            eg.add_pure_primitive(
                MultiSetRead {
                    name: name.into(),
                    multiset: arc.clone(),
                    element: self.element(),
                    op,
                },
                validator,
            );
        }
        for (name, op) in [
            ("multiset-insert", MultiSetEditOp::Insert),
            ("multiset-remove", MultiSetEditOp::Remove),
            ("multiset-remove-swapped", MultiSetEditOp::RemoveSwapped),
            ("multiset-reset-counts", MultiSetEditOp::ResetCounts),
        ] {
            eg.add_pure_primitive(
                MultiSetEdit {
                    name: name.into(),
                    multiset: arc.clone(),
                    element: self.element(),
                    op,
                },
                None,
            );
        }
        for (name, op) in [
            ("multiset-subtract", MultiSetBinaryOp::Subtract),
            (
                "multiset-subtract-swapped",
                MultiSetBinaryOp::SubtractSwapped,
            ),
            ("multiset-intersection", MultiSetBinaryOp::Intersection),
            ("multiset-sum", MultiSetBinaryOp::Sum),
        ] {
            eg.add_pure_primitive(
                MultiSetBinary {
                    name: name.into(),
                    multiset: arc.clone(),
                    op,
                },
                None,
            );
        }

        // Add multiset-sum-multisets if the inner arcsort is also a multiset
        for other_multiset_sort in eg.type_info.get_arcsorts_by(|f| {
            f.name() == self.element.name()
            // We can't query directly by arcsort type since it's wrapped in a ContainerSort which is not public
                && f.value_type() == Some(TypeId::of::<MultiSetContainer>())
        }) {
            eg.add_pure_primitive(
                SumMultisets {
                    name: "multiset-sum-multisets".into(),
                    multiset: other_multiset_sort.clone(),
                    multiset_of_multisets: arc.clone(),
                },
                None,
            );
        }
        let all_ms_sorts = eg
            .type_info
            .get_arcsorts_by(|f| f.value_type() == Some(TypeId::of::<MultiSetContainer>()));
        for fn_sort in eg.type_info.get_sorts::<FunctionSort>() {
            for ms_sort in &all_ms_sorts {
                try_registering_multiset_map(eg, fn_sort.clone(), ms_sort.clone(), arc.clone());
                if ms_sort.name() != arc.name() {
                    try_registering_multiset_map(eg, fn_sort.clone(), arc.clone(), ms_sort.clone());
                }
            }
            try_registering_multiset_non_map_primitives(eg, fn_sort.clone(), arc.clone());
        }
        if self.element.is_eq_sort() {
            eg.add_write_primitive(
                UnionValues {
                    name: "multiset-union-values".into(),
                    multiset: arc.clone(),
                    element: self.element.clone(),
                },
                None,
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
        // Canonical form (sorted by deterministic AST order, multiplicities
        // preserved as repeats) so proof checking can reproduce it from terms.
        normalize_multiset_term(termdag, element_terms)
    }

    fn rebuild_container_normalizer(&self) -> Option<(String, PrimitiveValidator)> {
        Some((
            "multiset-of".to_owned(),
            Arc::new(|termdag: &mut TermDag, args: &[TermId]| {
                Some(normalize_multiset_term(termdag, args.to_vec()))
            }),
        ))
    }

    fn serialized_name(&self, _container_values: &ContainerValues, _: Value) -> String {
        "multiset-of".to_owned()
    }
}

fn multiset_entries<'a, 'db: 'a>(
    state: &impl Core<'a, 'db>,
    value: Value,
) -> Option<Vec<(Value, usize)>> {
    state
        .with_container_sequence::<MultiSetContainer, _>(value, |payload| {
            encoded_multiset_entries(payload).collect()
        })
        .or_else(|| {
            state
                .value_to_owned_container::<MultiSetContainer>(value)
                .map(|multiset| multiset.data.iter_counts().collect())
        })
}

fn multiset_key(
    rebuild: bool,
    entries: impl IntoIterator<Item = (Value, usize)>,
) -> Option<Vec<Value>> {
    let entries = entries.into_iter();
    let mut key = Vec::with_capacity(entries.size_hint().0 * MULTISET_ENTRY_WIDTH + 1);
    key.push(Value::from_usize(rebuild as usize));
    let mut previous = None;
    let mut total = 0usize;
    for (value, count) in entries {
        debug_assert!(previous.is_none_or(|previous| previous < value));
        debug_assert!(count > 0);
        total = total.checked_add(count)?;
        i64::try_from(total).ok()?;
        previous = Some(value);
        key.push(value);
        encode_count(&mut key, count);
    }
    Some(key)
}

#[derive(Clone)]
struct MultiSetSingle {
    name: String,
    multiset: ArcSort,
    element: ArcSort,
}

impl Primitive for MultiSetSingle {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![
                self.element.clone(),
                I64Sort.to_arcsort(),
                self.multiset.clone(),
            ],
            span.clone(),
        )
        .into_box()
    }
}

impl PurePrim for MultiSetSingle {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let [value, count] = args else { return None };
        let count = usize::try_from(state.base_values().unwrap::<i64>(*count)).ok()?;
        let entries = (count != 0).then_some((*value, count));
        let key = multiset_key(self.multiset.is_eq_container_sort(), entries)?;
        Some(state.register_container_sequence::<MultiSetContainer>(&key))
    }
}

#[derive(Clone, Copy)]
enum MultiSetReadOp {
    Pick,
    Length,
    Contains,
    NotContains,
    ContainsSwapped,
    NotContainsSwapped,
    PickMax,
    Count,
}

#[derive(Clone)]
struct MultiSetRead {
    name: String,
    multiset: ArcSort,
    element: ArcSort,
    op: MultiSetReadOp,
}

impl Primitive for MultiSetRead {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        let types = match self.op {
            MultiSetReadOp::Pick | MultiSetReadOp::PickMax => {
                vec![self.multiset.clone(), self.element.clone()]
            }
            MultiSetReadOp::Length => vec![self.multiset.clone(), I64Sort.to_arcsort()],
            MultiSetReadOp::Contains | MultiSetReadOp::NotContains => vec![
                self.multiset.clone(),
                self.element.clone(),
                UnitSort.to_arcsort(),
            ],
            MultiSetReadOp::ContainsSwapped | MultiSetReadOp::NotContainsSwapped => vec![
                self.element.clone(),
                self.multiset.clone(),
                UnitSort.to_arcsort(),
            ],
            MultiSetReadOp::Count => vec![
                self.multiset.clone(),
                self.element.clone(),
                I64Sort.to_arcsort(),
            ],
        };
        SimpleTypeConstraint::new(self.name(), types, span.clone()).into_box()
    }
}

impl PurePrim for MultiSetRead {
    fn apply<'a, 'db>(&self, state: crate::PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let (multiset, needle) = match self.op {
            MultiSetReadOp::ContainsSwapped | MultiSetReadOp::NotContainsSwapped => {
                let [needle, multiset] = args else {
                    return None;
                };
                (*multiset, Some(*needle))
            }
            MultiSetReadOp::Contains | MultiSetReadOp::NotContains | MultiSetReadOp::Count => {
                let [multiset, needle] = args else {
                    return None;
                };
                (*multiset, Some(*needle))
            }
            _ => {
                let [multiset] = args else { return None };
                (*multiset, None)
            }
        };
        if let Some(result) =
            state.with_container_sequence::<MultiSetContainer, _>(multiset, |payload| {
                match self.op {
                    MultiSetReadOp::Pick => encoded_multiset_entries(payload)
                        .next()
                        .map(|(value, _)| value),
                    MultiSetReadOp::Length => {
                        let len = i64::try_from(encoded_multiset_len(payload)?).ok()?;
                        Some(state.base_values().get::<i64>(len))
                    }
                    MultiSetReadOp::Contains | MultiSetReadOp::ContainsSwapped => {
                        (encoded_multiset_count(payload, needle.unwrap()) != 0)
                            .then(|| state.base_values().get::<()>(()))
                    }
                    MultiSetReadOp::NotContains | MultiSetReadOp::NotContainsSwapped => {
                        (encoded_multiset_count(payload, needle.unwrap()) == 0)
                            .then(|| state.base_values().get::<()>(()))
                    }
                    MultiSetReadOp::Count => {
                        let count =
                            i64::try_from(encoded_multiset_count(payload, needle.unwrap())).ok()?;
                        Some(state.base_values().get::<i64>(count))
                    }
                    MultiSetReadOp::PickMax => encoded_multiset_entries(payload)
                        .max_by_key(|(_, count)| *count)
                        .map(|(value, _)| value),
                }
            })
        {
            return result;
        }

        let multiset = state.value_to_owned_container::<MultiSetContainer>(multiset)?;
        match self.op {
            MultiSetReadOp::Pick => multiset.data.pick().copied(),
            MultiSetReadOp::Length => Some(
                state
                    .base_values()
                    .get::<i64>(i64::try_from(multiset.data.len()).ok()?),
            ),
            MultiSetReadOp::Contains | MultiSetReadOp::ContainsSwapped => multiset
                .data
                .contains(&needle.unwrap())
                .then(|| state.base_values().get::<()>(())),
            MultiSetReadOp::NotContains | MultiSetReadOp::NotContainsSwapped => {
                (!multiset.data.contains(&needle.unwrap()))
                    .then(|| state.base_values().get::<()>(()))
            }
            MultiSetReadOp::Count => {
                let count = multiset
                    .data
                    .iter_counts()
                    .find(|(value, _)| *value == needle.unwrap())
                    .map(|(_, count)| count)
                    .unwrap_or(0);
                Some(state.base_values().get::<i64>(i64::try_from(count).ok()?))
            }
            MultiSetReadOp::PickMax => multiset
                .data
                .iter_counts()
                .max_by_key(|(_, count)| *count)
                .map(|(value, _)| value),
        }
    }
}

#[derive(Clone, Copy)]
enum MultiSetEditOp {
    Insert,
    Remove,
    RemoveSwapped,
    ResetCounts,
}

#[derive(Clone)]
struct MultiSetEdit {
    name: String,
    multiset: ArcSort,
    element: ArcSort,
    op: MultiSetEditOp,
}

impl Primitive for MultiSetEdit {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        let types = match self.op {
            MultiSetEditOp::Insert | MultiSetEditOp::Remove => vec![
                self.multiset.clone(),
                self.element.clone(),
                self.multiset.clone(),
            ],
            MultiSetEditOp::RemoveSwapped => vec![
                self.element.clone(),
                self.multiset.clone(),
                self.multiset.clone(),
            ],
            MultiSetEditOp::ResetCounts => {
                vec![self.multiset.clone(), self.multiset.clone()]
            }
        };
        SimpleTypeConstraint::new(self.name(), types, span.clone()).into_box()
    }
}

impl PurePrim for MultiSetEdit {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let (multiset, needle) = match self.op {
            MultiSetEditOp::Insert | MultiSetEditOp::Remove => {
                let [multiset, needle] = args else {
                    return None;
                };
                (*multiset, Some(*needle))
            }
            MultiSetEditOp::RemoveSwapped => {
                let [needle, multiset] = args else {
                    return None;
                };
                (*multiset, Some(*needle))
            }
            MultiSetEditOp::ResetCounts => {
                let [multiset] = args else { return None };
                (*multiset, None)
            }
        };
        let mut entries = multiset_entries(&state, multiset)?;
        match self.op {
            MultiSetEditOp::Insert => {
                let needle = needle.unwrap();
                match entries.binary_search_by_key(&needle, |(value, _)| *value) {
                    Ok(index) => entries[index].1 = entries[index].1.checked_add(1)?,
                    Err(index) => entries.insert(index, (needle, 1)),
                }
            }
            MultiSetEditOp::Remove | MultiSetEditOp::RemoveSwapped => {
                let index = entries
                    .binary_search_by_key(&needle.unwrap(), |(value, _)| *value)
                    .ok()?;
                if entries[index].1 == 1 {
                    entries.remove(index);
                } else {
                    entries[index].1 -= 1;
                }
            }
            MultiSetEditOp::ResetCounts => {
                for (_, count) in &mut entries {
                    *count = 1;
                }
            }
        }
        let key = multiset_key(self.multiset.is_eq_container_sort(), entries)?;
        Some(state.register_container_sequence::<MultiSetContainer>(&key))
    }
}

#[derive(Clone, Copy)]
enum MultiSetBinaryOp {
    Subtract,
    SubtractSwapped,
    Intersection,
    Sum,
}

#[derive(Clone)]
struct MultiSetBinary {
    name: String,
    multiset: ArcSort,
    op: MultiSetBinaryOp,
}

impl Primitive for MultiSetBinary {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![
                self.multiset.clone(),
                self.multiset.clone(),
                self.multiset.clone(),
            ],
            span.clone(),
        )
        .into_box()
    }
}

impl PurePrim for MultiSetBinary {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let [left, right] = args else { return None };
        let mut left = multiset_entries(&state, *left)?;
        let mut right = multiset_entries(&state, *right)?;
        if matches!(self.op, MultiSetBinaryOp::SubtractSwapped) {
            std::mem::swap(&mut left, &mut right);
        }
        let entries = merge_multisets(&left, &right, self.op)?;
        let key = multiset_key(self.multiset.is_eq_container_sort(), entries)?;
        Some(state.register_container_sequence::<MultiSetContainer>(&key))
    }
}

fn merge_multisets(
    left: &[(Value, usize)],
    right: &[(Value, usize)],
    op: MultiSetBinaryOp,
) -> Option<Vec<(Value, usize)>> {
    let op = match op {
        MultiSetBinaryOp::SubtractSwapped => MultiSetBinaryOp::Subtract,
        other => other,
    };
    let mut out = Vec::with_capacity(left.len() + right.len());
    let (mut l, mut r) = (0, 0);
    while l < left.len() && r < right.len() {
        match left[l].0.cmp(&right[r].0) {
            std::cmp::Ordering::Less => {
                if matches!(op, MultiSetBinaryOp::Subtract | MultiSetBinaryOp::Sum) {
                    out.push(left[l]);
                }
                l += 1;
            }
            std::cmp::Ordering::Greater => {
                if matches!(op, MultiSetBinaryOp::Subtract) {
                    return None;
                }
                if matches!(op, MultiSetBinaryOp::Sum) {
                    out.push(right[r]);
                }
                r += 1;
            }
            std::cmp::Ordering::Equal => {
                let (value, left_count) = left[l];
                let right_count = right[r].1;
                let count = match op {
                    MultiSetBinaryOp::Subtract => left_count.checked_sub(right_count)?,
                    MultiSetBinaryOp::Intersection => left_count.min(right_count),
                    MultiSetBinaryOp::Sum => left_count.checked_add(right_count)?,
                    MultiSetBinaryOp::SubtractSwapped => unreachable!(),
                };
                if count != 0 {
                    out.push((value, count));
                }
                l += 1;
                r += 1;
            }
        }
    }
    match op {
        MultiSetBinaryOp::Subtract => {
            if r != right.len() {
                return None;
            }
            out.extend_from_slice(&left[l..]);
        }
        MultiSetBinaryOp::Sum => {
            out.extend_from_slice(&left[l..]);
            out.extend_from_slice(&right[r..]);
        }
        MultiSetBinaryOp::Intersection => {}
        MultiSetBinaryOp::SubtractSwapped => unreachable!(),
    }
    Some(out)
}

/**
 * Register a multiset map primitive if the function matches the input and output multiset.
 */
pub(crate) fn try_registering_multiset_map(
    eg: &mut EGraph,
    fn_: Arc<FunctionSort>,
    input_ms: ArcSort,
    output_ms: ArcSort,
) {
    if fn_.inputs().len() != 1
        || fn_.inputs()[0].name() != input_ms.inner_sorts()[0].name()
        || fn_.output().name() != output_ms.inner_sorts()[0].name()
    {
        return;
    }
    eg.add_pure_primitive(
        Map {
            name: "unstable-multiset-map".into(),
            multiset: input_ms,
            output_multiset: output_ms,
            fn_: fn_.clone(),
        },
        None,
    );
}

pub(crate) fn register_multiset_primitives_for_function(eg: &mut EGraph, fn_: Arc<FunctionSort>) {
    let all_ms_sorts = eg
        .type_info
        .get_arcsorts_by(|f| f.value_type() == Some(TypeId::of::<MultiSetContainer>()));
    for input_ms in &all_ms_sorts {
        for output_ms in &all_ms_sorts {
            try_registering_multiset_map(eg, fn_.clone(), input_ms.clone(), output_ms.clone());
        }
    }
    for ms_sort in &all_ms_sorts {
        try_registering_multiset_non_map_primitives(eg, fn_.clone(), ms_sort.clone());
    }
}

fn try_registering_multiset_non_map_primitives(
    eg: &mut EGraph,
    fn_: Arc<FunctionSort>,
    multiset: ArcSort,
) {
    let element = multiset.inner_sorts()[0].clone();
    let element_name = element.name();

    if fn_.inputs().len() == 1
        && fn_.inputs()[0].name() == element_name
        && fn_.output().name() == "Unit"
    {
        eg.add_pure_primitive(
            Filter {
                name: "unstable-multiset-filter".into(),
                multiset: multiset.clone(),
                fn_: fn_.clone(),
                skip_empty: true,
            },
            None,
        );
        eg.add_pure_primitive(
            Filter {
                name: "unstable-multiset-filter-not".into(),
                multiset: multiset.clone(),
                fn_: fn_.clone(),
                skip_empty: false,
            },
            None,
        );
    }

    if fn_.inputs().len() == 2
        && fn_.inputs()[0].name() == element_name
        && fn_.inputs()[1].name() == element_name
        && fn_.output().name() == element_name
    {
        eg.add_pure_primitive(
            Reduce {
                name: "unstable-multiset-reduce".into(),
                multiset: multiset.clone(),
                fn_: fn_.clone(),
                element: element.clone(),
            },
            None,
        );
    }

    if fn_.inputs().len() == 2
        && fn_.inputs()[0].name() == multiset.name()
        && fn_.inputs()[1].name() == element_name
        && fn_.output().name() == "i64"
    {
        let unit = eg.type_info.get_sort_by_name("Unit").unwrap().clone();
        eg.add_full_primitive(
            FillIndex {
                name: "unstable-multiset-fill-index".into(),
                multiset: multiset.clone(),
                unit: unit.clone(),
                fn_: fn_.clone(),
            },
            None,
        );
        eg.add_write_primitive(
            ClearIndex {
                name: "unstable-multiset-clear-index".into(),
                multiset: multiset.clone(),
                unit,
                fn_: fn_.clone(),
            },
            None,
        );
    }

    if fn_.inputs().len() == 1
        && fn_.inputs()[0].name() == element_name
        && fn_.output().name() == multiset.name()
    {
        eg.add_pure_primitive(
            FlatMap {
                name: "unstable-multiset-flat-map".into(),
                multiset,
                fn_: fn_.clone(),
            },
            None,
        );
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
            &self.name,
            vec![
                self.fn_.clone(),
                self.multiset.clone(),
                self.output_multiset.clone(),
            ],
            span.clone(),
        )
        .into_box()
    }
}

impl PurePrim for Map {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let fc = state.value_to_owned_container::<FunctionContainer>(args[0])?;
        // Copy before callbacks: they may intern containers and grow the
        // execution-local prediction storage backing the fast slice.
        let entries = multiset_entries(&state, args[1])?;
        let mut mapped_values = MultiSet::new();
        for (v, count) in entries {
            if let Some(mapped) = state.apply_function(&fc, &[v]) {
                mapped_values.checked_insert_multiple_mut(mapped, count)?;
            }
        }
        let key = multiset_key(
            self.output_multiset.is_eq_container_sort(),
            mapped_values.iter_counts(),
        )?;
        Some(state.register_container_sequence::<MultiSetContainer>(&key))
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

// `FillIndex` reads the target table to skip already-filled rows
// (so re-firing doesn't double-count under accumulator-style merges
// like `+ old new`). The read makes its effect depend on live DB
// state, so it's only valid in `Context::Full` — registered as a
// `FullPrim` and only callable from a `:naive` rule (or from a
// global action).
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
}

impl FullPrim for FillIndex {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::FullState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let fc = state.value_to_owned_container::<FunctionContainer>(args[1])?;
        let entries = multiset_entries(&state, args[0])?
            .into_iter()
            .map(|(value, count)| Some((value, i64::try_from(count).ok()?)))
            .collect::<Option<Vec<_>>>()?;
        let action = match fc.0 {
            ResolvedFunctionId::Constructor(a) | ResolvedFunctionId::Function(a) => a,
            // Primitive functions cannot be used with
            // unstable-multiset-fill-index, since they cannot be set.
            ResolvedFunctionId::Primitive { .. } => return None,
        };
        let unit_val = state.base_values().get::<()>(());
        let es = state.raw_exec_state();
        for (v, count) in entries {
            let mut row = vec![args[0], v];
            // Skip the whole fill if any index row already exists.
            // This relies on `unstable-multiset-fill-index` writing all
            // rows for a given multiset in one pass.
            if action.lookup(es, &row).is_some() {
                break;
            }
            row.push(es.base_values().get::<i64>(count));
            action.insert(es, row.into_iter());
        }
        Some(unit_val)
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

// `ClearIndex` removes table rows; action-only.
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
}

impl WritePrim for ClearIndex {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::WriteState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let fc = state.value_to_owned_container::<FunctionContainer>(args[1])?;
        let entries = multiset_entries(&state, args[0])?;
        let action = match fc.0 {
            ResolvedFunctionId::Constructor(a) | ResolvedFunctionId::Function(a) => a,
            // Primitive functions cannot be used with
            // unstable-multiset-clear-index, since they cannot be deleted.
            ResolvedFunctionId::Primitive { .. } => return None,
        };
        let unit_val = state.base_values().get::<()>(());
        let es = state.raw_exec_state();
        for (v, _) in entries {
            action.remove(es, &[args[0], v]);
        }
        Some(unit_val)
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
            &self.name,
            vec![
                self.fn_.clone(),
                self.multiset.clone(),
                self.multiset.clone(),
            ],
            span.clone(),
        )
        .into_box()
    }
}

impl PurePrim for FlatMap {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let fc = state.value_to_owned_container::<FunctionContainer>(args[0])?;
        let entries = multiset_entries(&state, args[1])?;
        let mut flattened = MultiSet::new();
        for (v, count) in entries {
            let mapped = state.apply_function(&fc, &[v]);
            if let Some(mapped_ms) = mapped {
                for (mapped, mapped_count) in multiset_entries(&state, mapped_ms)? {
                    flattened
                        .checked_insert_multiple_mut(mapped, count.checked_mul(mapped_count)?)?;
                }
            } else {
                flattened.checked_insert_multiple_mut(v, count)?;
            }
        }
        let key = multiset_key(
            self.multiset.is_eq_container_sort(),
            flattened.iter_counts(),
        )?;
        Some(state.register_container_sequence::<MultiSetContainer>(&key))
    }
}

// (unstable-multiset-filter (MultiSet[X], [X] -> Unit) -> MultiSet[X])
// will filter the elements in the multiset based on whether the function is defined for them.
// If skip_empty is true, it will keep elements where the function is defined, otherwise it will keep elements where the function is not defined.
#[derive(Clone)]
struct Filter {
    name: String,
    multiset: ArcSort,
    fn_: Arc<FunctionSort>,
    skip_empty: bool,
}

impl Primitive for Filter {
    fn name(&self) -> &str {
        &self.name
    }
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            &self.name,
            vec![
                self.fn_.clone(),
                self.multiset.clone(),
                self.multiset.clone(),
            ],
            span.clone(),
        )
        .into_box()
    }
}

impl PurePrim for Filter {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let fc = state.value_to_owned_container::<FunctionContainer>(args[0])?;
        let entries = multiset_entries(&state, args[1])?;
        let mut filtered = Vec::with_capacity(entries.len());
        for (v, count) in entries {
            let mapped = state.apply_function(&fc, &[v]);
            if mapped.is_some() == self.skip_empty {
                filtered.push((v, count));
            }
        }
        let key = multiset_key(self.multiset.is_eq_container_sort(), filtered)?;
        Some(state.register_container_sequence::<MultiSetContainer>(&key))
    }
}

// (multiset-sum-multisets (MultiSet[MultiSet[X]]) -> MultiSet[X])
// will sum all multisets in the outer multiset into a single multiset

#[derive(Clone)]
struct SumMultisets {
    name: String,
    multiset: ArcSort,
    multiset_of_multisets: ArcSort,
}

// `SumMultisets` flattens a multiset of multisets. Only reads container
// contents and registers the result — pure.
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
}

impl PurePrim for SumMultisets {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let outer = multiset_entries(&state, args[0])?;
        let mut flattened = MultiSet::new();
        for (multiset, outer_count) in outer {
            for (value, inner_count) in multiset_entries(&state, multiset)? {
                flattened
                    .checked_insert_multiple_mut(value, outer_count.checked_mul(inner_count)?)?;
            }
        }
        let key = multiset_key(
            self.multiset.is_eq_container_sort(),
            flattened.iter_counts(),
        )?;
        Some(state.register_container_sequence::<MultiSetContainer>(&key))
    }
}

// (unstable-multiset-reduce ([X, X] -> X, X, MultiSet[X]) -> X
// will reduce the multiset using the provided binary function and initial value
// Won't use the initial value if the multiset is non-empty.
#[derive(Clone)]
struct Reduce {
    name: String,
    multiset: ArcSort,
    fn_: Arc<FunctionSort>,
    element: ArcSort,
}

impl Primitive for Reduce {
    fn name(&self) -> &str {
        &self.name
    }
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            &self.name,
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
}

impl PurePrim for Reduce {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let fc = state.value_to_owned_container::<FunctionContainer>(args[0])?;
        let initial = args[1];
        let entries = multiset_entries(&state, args[2])?;
        let mut acc = initial;
        let mut has_value = false;
        for (value, count) in entries {
            for _ in 0..count {
                if has_value {
                    acc = state.apply_function(&fc, &[acc, value])?;
                } else {
                    acc = value;
                    has_value = true;
                }
            }
        }
        Some(acc)
    }
}

// (multiset-union-values MultiSet[A]) -> A
// where A: Eq
// Unions all values in the multiset together using the union action defined for the inner type.
#[derive(Clone)]
struct UnionValues {
    name: String,
    multiset: ArcSort,
    element: ArcSort,
}

// `UnionValues` writes to the union-find; action-only.
impl Primitive for UnionValues {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.multiset.clone(), self.element.clone()],
            span.clone(),
        )
        .into_box()
    }
}

impl WritePrim for UnionValues {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::WriteState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let entries = multiset_entries(&state, args[0])?;
        if entries.is_empty() {
            return None;
        }
        let first = entries[0].0;
        for (v, _) in entries.into_iter().skip(1) {
            state.union(first, v).ok()?;
        }
        Some(first)
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

        /// Subtract the counts of another multiset from this multiset, taking ownership of both and returning a new multiset.
        pub fn subtract(mut self, other: &MultiSet<T>) -> Option<MultiSet<T>> {
            for (k, v) in other.0.iter() {
                if let Some(self_v) = self.0.get_mut(k) {
                    if *self_v < *v {
                        return None;
                    }
                    *self_v -= *v;
                    self.1 -= *v;
                    if *self_v == 0 {
                        self.0.remove(k);
                    }
                } else {
                    return None;
                }
            }
            Some(self)
        }

        pub fn checked_insert_multiple_mut(&mut self, value: T, n: usize) -> Option<()> {
            if n == 0 {
                return Some(());
            }
            let total = self.1.checked_add(n)?;
            let count = self
                .0
                .get(&value)
                .copied()
                .unwrap_or_default()
                .checked_add(n)?;
            self.1 = total;
            self.0.insert(value, count);
            Some(())
        }

        pub fn insert_multiple_mut(&mut self, value: T, n: usize) {
            self.checked_insert_multiple_mut(value, n)
                .expect("multiset cardinality overflow");
        }

        /// Compute the sum of two multisets.
        pub fn sum(mut self, MultiSet(other_map, other_count): Self) -> Self {
            let target_count = self
                .1
                .checked_add(other_count)
                .expect("multiset cardinality overflow");
            for (k, v) in other_map {
                self.insert_multiple_mut(k, v);
            }
            assert_eq!(self.1, target_count);
            self
        }

        /// Compute the intersection of two multisets.
        /// The count of each element in the result is the minimum of its counts in the two multisets.
        pub fn intersection(self, MultiSet(other_map, _): Self) -> Self {
            let mut new_map = BTreeMap::new();
            for (k, v) in self.0.into_iter() {
                if let Some(other_v) = other_map.get(&k) {
                    let new_v = std::cmp::min(v, *other_v);
                    new_map.insert(k, new_v);
                }
            }
            let new_count = new_map.values().sum();
            MultiSet(new_map, new_count)
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
    fn compact_sequence_codec_round_trips_and_merges_counts() {
        let mut data = MultiSet::new();
        data.insert_multiple_mut(value(2), 3);
        data.insert_multiple_mut(value(4), 5);
        let multiset = MultiSetContainer {
            do_rebuild: true,
            data,
        };
        let mut encoded = Vec::new();
        multiset.encode_sequence(&mut encoded);
        assert_eq!(encoded.len(), 1 + 2 * MULTISET_ENTRY_WIDTH);
        assert_eq!(MultiSetContainer::decode_sequence(&encoded), multiset);

        let mut children = Vec::new();
        MultiSetContainer::visit_sequence_values(&encoded, &mut |value| children.push(value));
        assert_eq!(children, vec![value(2), value(4)]);

        let mut rebuilt = Vec::new();
        assert!(MultiSetContainer::rebuild_sequence(
            &encoded,
            &Collapse {
                from: value(4),
                to: value(2),
            },
            &mut rebuilt,
        ));
        let rebuilt = MultiSetContainer::decode_sequence(&rebuilt);
        assert_eq!(
            rebuilt.data.iter_counts().collect::<Vec<_>>(),
            vec![(value(2), 8)]
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn compact_sequence_codec_preserves_both_count_lanes() {
        let count = u32::MAX as usize + 17;
        let mut data = MultiSet::new();
        data.insert_multiple_mut(value(9), count);
        let multiset = MultiSetContainer {
            do_rebuild: false,
            data,
        };
        let mut encoded = Vec::new();
        multiset.encode_sequence(&mut encoded);
        assert_ne!(encoded[3], value(0));
        assert_eq!(MultiSetContainer::decode_sequence(&encoded), multiset);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    #[should_panic(expected = "cardinality exceeds the i64 language representation")]
    fn compact_sequence_codec_rejects_oversized_cardinality() {
        let mut encoded = vec![value(0), value(1)];
        encode_count(&mut encoded, i64::MAX as usize);
        encoded.push(value(2));
        encode_count(&mut encoded, 1);
        MultiSetContainer::decode_sequence(&encoded);
    }

    #[test]
    fn compact_encoding_handles_large_multiplicity() {
        let mut egraph = EGraph::default();
        egraph
            .parse_and_run_program(
                None,
                r#"
                (sort IntMultiSet (MultiSet i64))
                (let $large (multiset-single 7 1000000000))
                (check (= 1000000000 (multiset-length $large)))
                (check (= 1000000000 (multiset-count $large 7)))
                (check (= 7 (multiset-pick $large)))
                "#,
            )
            .unwrap();
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn multiset_primitives_reject_counts_beyond_i64() {
        let mut egraph = EGraph::default();
        egraph
            .parse_and_run_program(
                None,
                r#"
                (sort IntMultiSet (MultiSet i64))
                (let $max (multiset-single 7 9223372036854775807))
                (check (= 9223372036854775807 (multiset-length $max)))
                (check (= 9223372036854775807 (multiset-count $max 7)))
                (fail (multiset-insert $max 7))
                (fail (multiset-sum $max (multiset-single 8 1)))

                (let $two (multiset-single 8 2))
                (sort NestedIntMultiSet (MultiSet IntMultiSet))
                (fail (multiset-sum-multisets (multiset-of $max $max $two)))

                (function expand (i64) IntMultiSet :no-merge)
                (set (expand 1) $max)
                (set (expand 2) $two)
                (sort ExpandFn (UnstableFn (i64) IntMultiSet))
                (fail (unstable-multiset-flat-map
                    (unstable-fn "expand")
                    (multiset-of 1 1 2)))
                "#,
            )
            .unwrap();
    }

    #[test]
    fn sequence_primitives_consume_local_predictions() {
        let mut egraph = EGraph::default();
        egraph
            .parse_and_run_program(
                None,
                r#"
                (sort IntMultiSet (MultiSet i64))
                (sort NestedIntMultiSet (MultiSet IntMultiSet))
                (check (= 4
                    (multiset-count
                        (multiset-sum (multiset-single 2 3) (multiset-of 1 2))
                        2)))
                (check (= (multiset-of 1 2)
                    (multiset-subtract
                        (multiset-of 1 1 2 3)
                        (multiset-of 1 3))))
                (check (= (multiset-of 2 2)
                    (multiset-intersection
                        (multiset-of 1 2 2 2)
                        (multiset-of 2 2 3))))
                (check (= 1
                    (multiset-count
                        (multiset-reset-counts (multiset-single 9 1000000000))
                        9)))
                (check (= 2 (multiset-pick-max (multiset-of 1 1 2 2))))
                (check (= (multiset-of 1 1 2)
                    (multiset-sum-multisets
                        (multiset-of (multiset-of 1 2) (multiset-of 1)))))
                "#,
            )
            .unwrap();
    }

    #[test]
    fn counted_slice_algebra_matches_multiset_oracle_exhaustively() {
        fn sample(mut encoded: usize) -> MultiSet<Value> {
            let mut result = MultiSet::new();
            for index in 0..3 {
                let count = encoded % 3;
                encoded /= 3;
                if count != 0 {
                    result.insert_multiple_mut(value(index + 10), count);
                }
            }
            result
        }

        for left_code in 0..27 {
            for right_code in 0..27 {
                let left = sample(left_code);
                let right = sample(right_code);
                let left_entries = left.iter_counts().collect::<Vec<_>>();
                let right_entries = right.iter_counts().collect::<Vec<_>>();

                let sum = left.clone().sum(right.clone()).iter_counts().collect();
                assert_eq!(
                    merge_multisets(&left_entries, &right_entries, MultiSetBinaryOp::Sum),
                    Some(sum)
                );

                let intersection = left
                    .clone()
                    .intersection(right.clone())
                    .iter_counts()
                    .collect();
                assert_eq!(
                    merge_multisets(
                        &left_entries,
                        &right_entries,
                        MultiSetBinaryOp::Intersection,
                    ),
                    Some(intersection)
                );

                let subtraction = left
                    .clone()
                    .subtract(&right)
                    .map(|result| result.iter_counts().collect());
                assert_eq!(
                    merge_multisets(&left_entries, &right_entries, MultiSetBinaryOp::Subtract,),
                    subtraction
                );
            }
        }
    }
}
