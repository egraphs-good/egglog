use crate::numeric_id::NumericId;

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PairContainer {
    do_rebuild_first: bool,
    do_rebuild_second: bool,
    pub first: Value,
    pub second: Value,
}

impl ContainerValue for PairContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn ValueRebuilder) -> bool {
        let mut changed = false;
        if self.do_rebuild_first {
            let new = rebuilder.rebuild_val(self.first);
            changed |= self.first != new;
            self.first = new;
        }
        if self.do_rebuild_second {
            let new = rebuilder.rebuild_val(self.second);
            changed |= self.second != new;
            self.second = new;
        }
        changed
    }
    fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        [self.first, self.second].into_iter()
    }
}

impl SequenceContainerValue for PairContainer {
    fn encode_sequence(&self, _base_values: &BaseValues, out: &mut Vec<Value>) {
        let rebuild_mask =
            self.do_rebuild_first as usize | ((self.do_rebuild_second as usize) << 1);
        out.extend_from_slice(&[Value::from_usize(rebuild_mask), self.first, self.second]);
    }

    fn decode_sequence(sequence: &[Value], _base_values: &BaseValues) -> Self {
        let [rebuild_mask, first, second] = sequence else {
            panic!("serialized PairContainer must contain a rebuild mask and two values");
        };
        assert!(
            rebuild_mask.index() < 4,
            "serialized PairContainer has an invalid rebuild mask"
        );
        Self {
            do_rebuild_first: rebuild_mask.index() & 1 != 0,
            do_rebuild_second: rebuild_mask.index() & 2 != 0,
            first: *first,
            second: *second,
        }
    }

    fn sequence_values(sequence: &[Value]) -> &[Value] {
        let [_, values @ ..] = sequence else {
            unreachable!()
        };
        assert_eq!(
            values.len(),
            2,
            "serialized PairContainer must contain exactly two values"
        );
        values
    }

    fn visit_sequence_values(sequence: &[Value], visitor: &mut dyn FnMut(Value)) {
        let [rebuild_mask, first, second] = sequence else {
            panic!("serialized PairContainer must contain a rebuild mask and two values");
        };
        match rebuild_mask.index() {
            0 => {}
            1 => visitor(*first),
            2 => visitor(*second),
            3 => {
                visitor(*first);
                visitor(*second);
            }
            _ => panic!("serialized PairContainer has an invalid rebuild mask"),
        }
    }

    fn rebuild_sequence(
        sequence: &[Value],
        _base_values: &BaseValues,
        rebuilder: &dyn ValueRebuilder,
        out: &mut Vec<Value>,
    ) -> bool {
        let [rebuild_mask, first, second] = sequence else {
            panic!("serialized PairContainer must contain a rebuild mask and two values");
        };
        let (mut rebuilt_first, mut rebuilt_second) = (*first, *second);
        match rebuild_mask.index() {
            0 => return false,
            1 => rebuilt_first = rebuilder.rebuild_val(*first),
            2 => rebuilt_second = rebuilder.rebuild_val(*second),
            3 => {
                rebuilt_first = rebuilder.rebuild_val(*first);
                rebuilt_second = rebuilder.rebuild_val(*second);
            }
            _ => panic!("serialized PairContainer has an invalid rebuild mask"),
        }
        if rebuilt_first == *first && rebuilt_second == *second {
            return false;
        }
        out.extend_from_slice(&[*rebuild_mask, rebuilt_first, rebuilt_second]);
        true
    }
}

/// The `(first, second)` children of a `(pair a b)` term; `None` for any
/// other term.
fn pair_term_children(termdag: &TermDag, term: TermId) -> Option<(TermId, TermId)> {
    match termdag.get(term) {
        Term::App(head, children) if head == "pair" => match children.as_slice() {
            [first, second] => Some((*first, *second)),
            _ => None,
        },
        _ => None,
    }
}

/// Intern the `(pair a b)` term for `args`; `None` unless there are exactly
/// two. The inverse of [`pair_term_children`].
fn pair_term(termdag: &mut TermDag, args: &[TermId]) -> Option<TermId> {
    if args.len() != 2 {
        return None;
    }
    Some(termdag.app("pair".into(), args.to_vec()))
}

/// A pair of two values supporting these primitives:
/// - `pair`
/// - `pair-first`
/// - `pair-second`
#[derive(Clone, Debug)]
pub struct PairSort {
    name: String,
    first: ArcSort,
    second: ArcSort,
}

impl PairSort {
    pub fn first(&self) -> ArcSort {
        self.first.clone()
    }

    pub fn second(&self) -> ArcSort {
        self.second.clone()
    }
}

impl Presort for PairSort {
    fn presort_name() -> &'static str {
        "Pair"
    }

    fn reserved_primitives() -> Vec<&'static str> {
        vec!["pair", "pair-first", "pair-second"]
    }

    fn make_sort(
        typeinfo: &mut TypeInfo,
        name: String,
        args: &[Expr],
        span: Span,
    ) -> Result<ArcSort, TypeError> {
        if let [Expr::Var(a_span, a), Expr::Var(b_span, b)] = args {
            let a = typeinfo
                .get_sort_by_name(a)
                .ok_or(TypeError::UndefinedSort(a.clone(), a_span.clone()))?;
            let b = typeinfo
                .get_sort_by_name(b)
                .ok_or(TypeError::UndefinedSort(b.clone(), b_span.clone()))?;

            let out = Self {
                name,
                first: a.clone(),
                second: b.clone(),
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

impl ContainerSort for PairSort {
    type Container = PairContainer;

    fn name(&self) -> &str {
        &self.name
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend.register_sequence_container_ty::<PairContainer>();
    }

    fn inner_sorts(&self) -> Vec<ArcSort> {
        vec![self.first.clone(), self.second.clone()]
    }

    fn is_eq_container_sort(&self) -> bool {
        self.first.is_eq_sort()
            || self.second.is_eq_sort()
            || self.first.is_eq_container_sort()
            || self.second.is_eq_container_sort()
    }

    fn inner_values(
        &self,
        container_values: &ContainerValues,
        value: Value,
    ) -> Vec<(ArcSort, Value)> {
        let val = container_values
            .get_val::<PairContainer>(value)
            .unwrap()
            .clone();
        vec![
            (self.first.clone(), val.first),
            (self.second.clone(), val.second),
        ]
    }

    fn register_primitives(&self, eg: &mut EGraph) {
        let arc = self.clone().to_arcsort();

        // The proof "term form" of a pair: an s-expr `(pair a b)` headed by
        // the constructing primitive, matching `reconstruct_termdag`. The
        // validator lets the proof checker evaluate `pair` applications, and
        // `pair-first`/`pair-second` extract a child of a `(pair a b)` term.
        let pair_first_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let [pair] = args else {
                return None;
            };
            pair_term_children(termdag, *pair).map(|(first, _)| first)
        };
        let pair_second_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let [pair] = args else {
                return None;
            };
            pair_term_children(termdag, *pair).map(|(_, second)| second)
        };

        eg.add_pure_primitive(
            PairConstruct {
                name: "pair".into(),
                pair: arc.clone(),
                first: self.first(),
                second: self.second(),
                rebuild_mask: self.first.is_eq_sort() as usize
                    | ((self.second.is_eq_sort() as usize) << 1)
                    | (self.first.is_eq_container_sort() as usize)
                    | ((self.second.is_eq_container_sort() as usize) << 1),
            },
            Some(Arc::new(pair_term)),
        );
        eg.add_pure_primitive(
            PairProject {
                name: "pair-first".into(),
                pair: arc.clone(),
                output: self.first(),
                index: 0,
            },
            Some(Arc::new(pair_first_validator)),
        );
        eg.add_pure_primitive(
            PairProject {
                name: "pair-second".into(),
                pair: arc,
                output: self.second(),
                index: 1,
            },
            Some(Arc::new(pair_second_validator)),
        );
    }

    fn reconstruct_termdag(
        &self,
        _container_values: &ContainerValues,
        _value: Value,
        termdag: &mut TermDag,
        element_terms: Vec<TermId>,
    ) -> TermId {
        assert_eq!(element_terms.len(), 2);
        termdag.app("pair".into(), vec![element_terms[0], element_terms[1]])
    }

    fn rebuild_container_normalizer(&self) -> Option<(String, PrimitiveValidator)> {
        Some(("pair".to_owned(), Arc::new(pair_term)))
    }

    fn serialized_name(&self, _container_values: &ContainerValues, _: Value) -> String {
        self.name().to_owned()
    }
}

/// Construct a pair directly in its canonical flat representation.
#[derive(Clone)]
struct PairConstruct {
    name: String,
    pair: ArcSort,
    first: ArcSort,
    second: ArcSort,
    rebuild_mask: usize,
}

impl Primitive for PairConstruct {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.first.clone(), self.second.clone(), self.pair.clone()],
            span.clone(),
        )
        .into_box()
    }
}

impl PurePrim for PairConstruct {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let [first, second] = args else { return None };
        Some(state.register_container_sequence::<PairContainer>(&[
            Value::from_usize(self.rebuild_mask),
            *first,
            *second,
        ]))
    }
}

/// Project one pair lane through a borrowed sequence lookup, falling back to
/// the compatibility decoder for externally registered legacy storage.
#[derive(Clone)]
struct PairProject {
    name: String,
    pair: ArcSort,
    output: ArcSort,
    index: usize,
}

impl Primitive for PairProject {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.pair.clone(), self.output.clone()],
            span.clone(),
        )
        .into_box()
    }
}

impl PurePrim for PairProject {
    fn apply<'a, 'db>(&self, state: crate::PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let [pair] = args else { return None };
        state
            .with_container_sequence::<PairContainer, _>(*pair, |values| {
                values.get(self.index).copied()
            })
            .or_else(|| {
                state
                    .value_to_owned_container::<PairContainer>(*pair)
                    .map(|pair| {
                        Some(match self.index {
                            0 => pair.first,
                            1 => pair.second,
                            _ => unreachable!(),
                        })
                    })
            })?
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct SelectiveRebuilder {
        from: Value,
        to: Value,
        calls: AtomicUsize,
    }

    impl ValueRebuilder for SelectiveRebuilder {
        fn rebuild_val(&self, value: Value) -> Value {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if value == self.from { self.to } else { value }
        }
    }

    fn value(index: usize) -> Value {
        Value::from_usize(index)
    }

    #[test]
    fn sequence_codec_round_trips_and_selectively_rebuilds() {
        let pair = PairContainer {
            do_rebuild_first: false,
            do_rebuild_second: true,
            first: value(10),
            second: value(20),
        };
        let mut encoded = Vec::new();
        pair.encode_sequence(&BaseValues::default(), &mut encoded);
        assert_eq!(encoded, vec![value(2), value(10), value(20)]);
        assert_eq!(
            PairContainer::decode_sequence(&encoded, &BaseValues::default()),
            pair
        );
        assert_eq!(PairContainer::sequence_values(&encoded), &encoded[1..]);

        let mut visited = Vec::new();
        PairContainer::visit_sequence_values(&encoded, &mut |value| visited.push(value));
        assert_eq!(visited, vec![value(20)]);

        let rebuilder = SelectiveRebuilder {
            from: value(20),
            to: value(30),
            calls: AtomicUsize::new(0),
        };
        let mut rebuilt = Vec::new();
        assert!(PairContainer::rebuild_sequence(
            &encoded,
            &BaseValues::default(),
            &rebuilder,
            &mut rebuilt,
        ));
        assert_eq!(rebuilt, vec![value(2), value(10), value(30)]);
        assert_eq!(rebuilder.calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn sequence_codec_leaves_output_empty_when_unchanged() {
        let encoded = vec![value(1), value(10), value(20)];
        let rebuilder = SelectiveRebuilder {
            from: value(20),
            to: value(30),
            calls: AtomicUsize::new(0),
        };
        let mut rebuilt = Vec::new();
        assert!(!PairContainer::rebuild_sequence(
            &encoded,
            &BaseValues::default(),
            &rebuilder,
            &mut rebuilt,
        ));
        assert!(rebuilt.is_empty());
        assert_eq!(rebuilder.calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn sequence_primitives_consume_local_predictions() {
        let mut egraph = EGraph::default();
        egraph
            .parse_and_run_program(
                None,
                r#"
                (sort IntPair (Pair i64 i64))
                (check (= (pair-first (pair 10 20)) 10))
                (check (= (pair-second (pair 10 20)) 20))
                "#,
            )
            .unwrap();
    }
}
