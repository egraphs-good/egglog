use crate::Write;
use crate::constraint::AllEqualTypeConstraint;
use crate::numeric_id::NumericId;
use std::any::TypeId;
use std::iter::zip;

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VecContainer {
    pub do_rebuild: bool,
    pub data: Vec<Value>,
}

impl ContainerValue for VecContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn ValueRebuilder) -> bool {
        if self.do_rebuild {
            rebuilder.rebuild_slice(&mut self.data)
        } else {
            false
        }
    }
    fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        self.data.iter().copied()
    }
}

impl SequenceContainerValue for VecContainer {
    fn encode_sequence(&self, _base_values: &BaseValues, out: &mut Vec<Value>) {
        out.push(Value::from_usize(self.do_rebuild as usize));
        out.extend_from_slice(&self.data);
    }

    fn decode_sequence(sequence: &[Value], _base_values: &BaseValues) -> Self {
        let (&header, data) = sequence
            .split_first()
            .expect("serialized VecContainer must include its rebuild flag");
        assert!(
            header == Value::from_usize(0) || header == Value::from_usize(1),
            "serialized VecContainer has an invalid rebuild flag"
        );
        Self {
            do_rebuild: header == Value::from_usize(1),
            data: data.to_vec(),
        }
    }

    fn sequence_values(sequence: &[Value]) -> &[Value] {
        sequence
            .get(1..)
            .expect("serialized VecContainer must include its rebuild flag")
    }

    fn visit_sequence_values(sequence: &[Value], visitor: &mut dyn FnMut(Value)) {
        let (&header, data) = sequence
            .split_first()
            .expect("serialized VecContainer must include its rebuild flag");
        match header.index() {
            0 => {}
            1 => data.iter().copied().for_each(visitor),
            _ => panic!("serialized VecContainer has an invalid rebuild flag"),
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
            .expect("serialized VecContainer must include its rebuild flag");
        if header == Value::from_usize(0) {
            return false;
        }
        assert_eq!(
            header,
            Value::from_usize(1),
            "serialized VecContainer has an invalid rebuild flag"
        );
        out.push(header);
        out.extend_from_slice(data);
        if rebuilder.rebuild_slice(&mut out[1..]) {
            true
        } else {
            out.clear();
            false
        }
    }
}

#[derive(Clone, Debug)]
pub struct VecSort {
    name: String,
    element: ArcSort,
}

impl VecSort {
    pub fn element(&self) -> ArcSort {
        self.element.clone()
    }
}

/// The element terms of a vec's canonical term form (`(vec-of e0 …)`, or
/// `(vec-empty)` for the empty vec); `None` for any other term.
fn vec_term_children(termdag: &TermDag, term: TermId) -> Option<Vec<TermId>> {
    match termdag.get(term) {
        Term::App(head, children) if head == "vec-of" => Some(children.clone()),
        Term::App(head, _) if head == "vec-empty" => Some(vec![]),
        _ => None,
    }
}

/// Intern the canonical vec term for `children`: `(vec-of e0 ...)`, or
/// `(vec-empty)` when empty. The inverse of [`vec_term_children`].
fn vec_term(termdag: &mut TermDag, children: Vec<TermId>) -> TermId {
    if children.is_empty() {
        termdag.app("vec-empty".into(), vec![])
    } else {
        termdag.app("vec-of".into(), children)
    }
}

impl Presort for VecSort {
    fn presort_name() -> &'static str {
        "Vec"
    }

    fn reserved_primitives() -> Vec<&'static str> {
        vec![
            "vec-of",
            "vec-append",
            "vec-empty",
            "vec-push",
            "vec-pop",
            "vec-not-contains",
            "vec-contains",
            "vec-length",
            "vec-get",
            "vec-set",
            "vec-remove",
            "vec-union",
            "vec-range",
            "unstable-vec-map",
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

impl ContainerSort for VecSort {
    type Container = VecContainer;

    fn name(&self) -> &str {
        &self.name
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend.register_sequence_container_ty::<VecContainer>();
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
            .get_val::<VecContainer>(value)
            .unwrap()
            .clone();
        val.data
            .iter()
            .map(|e| (self.element.clone(), *e))
            .collect()
    }

    fn register_primitives(&self, eg: &mut EGraph) {
        let arc: Arc<dyn Sort> = self.clone().to_arcsort();

        // The proof "term form" of a vec: `(vec-of e0 e1 ...)`, or `(vec-empty)`
        // when empty, matching `reconstruct_termdag`. The validator lets the
        // proof checker evaluate `vec-of`/`vec-empty` applications.
        let vec_of_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            Some(vec_term(termdag, args.to_vec()))
        };
        let vec_empty_validator = |termdag: &mut TermDag, _args: &[TermId]| -> Option<TermId> {
            Some(termdag.app("vec-empty".into(), vec![]))
        };
        let vec_length_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let [vec] = args else { return None };
            let len = vec_term_children(termdag, *vec)?.len() as i64;
            Some(termdag.lit(Literal::Int(len)))
        };
        let vec_get_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let [vec, index] = args else { return None };
            let Term::Lit(Literal::Int(index)) = termdag.get(*index) else {
                return None;
            };
            let index = usize::try_from(*index).ok()?;
            vec_term_children(termdag, *vec)?.get(index).copied()
        };
        let vec_contains_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let [vec, value] = args else { return None };
            vec_term_children(termdag, *vec)?
                .contains(value)
                .then(|| termdag.lit(Literal::Unit))
        };
        let vec_not_contains_validator =
            |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
                let [vec, value] = args else { return None };
                let contains = vec_term_children(termdag, *vec)?.contains(value);
                (!contains).then(|| termdag.lit(Literal::Unit))
            };

        add_primitive_with_validator!(eg, "vec-empty"  = {self.clone(): VecSort} |                                | -> @VecContainer (arc) { VecContainer {
            do_rebuild: self.ctx.is_eq_container_sort(),
            data: Vec::new()
        } }, vec_empty_validator);
        add_primitive_with_validator!(eg, "vec-of"     = {self.clone(): VecSort} [xs: # (self.element())          ] -> @VecContainer (arc) { VecContainer {
            do_rebuild: self.ctx.is_eq_container_sort(),
            data: xs                     .collect()
        } }, vec_of_validator);
        eg.add_pure_primitive(
            VecAppend {
                name: "vec-append".into(),
                vec: arc.clone(),
            },
            None,
        );

        for (name, op) in [
            ("vec-push", VecEditOp::Push),
            ("vec-pop", VecEditOp::Pop),
            ("vec-set", VecEditOp::Set),
            ("vec-remove", VecEditOp::Remove),
        ] {
            eg.add_pure_primitive(
                VecEdit {
                    name: name.into(),
                    vec: arc.clone(),
                    element: self.element(),
                    op,
                },
                None,
            );
        }

        eg.add_pure_primitive(
            VecRead {
                name: "vec-length".into(),
                vec: arc.clone(),
                element: self.element(),
                op: VecReadOp::Length,
            },
            Some(Arc::new(vec_length_validator)),
        );
        eg.add_pure_primitive(
            VecRead {
                name: "vec-contains".into(),
                vec: arc.clone(),
                element: self.element(),
                op: VecReadOp::Contains,
            },
            Some(Arc::new(vec_contains_validator)),
        );
        eg.add_pure_primitive(
            VecRead {
                name: "vec-not-contains".into(),
                vec: arc.clone(),
                element: self.element(),
                op: VecReadOp::NotContains,
            },
            Some(Arc::new(vec_not_contains_validator)),
        );
        eg.add_pure_primitive(
            VecRead {
                name: "vec-get".into(),
                vec: arc.clone(),
                element: self.element(),
                op: VecReadOp::Get,
            },
            Some(Arc::new(vec_get_validator)),
        );
        if self.element.is_eq_sort() {
            eg.add_write_primitive(
                Union {
                    name: "vec-union".into(),
                    vec: arc.clone(),
                },
                None,
            );
        }
        // vec-range
        if self.element.name() == "i64" {
            add_primitive!(eg, "vec-range" = {self.clone(): VecSort} |end: i64| -> @VecContainer (arc) { VecContainer {
                do_rebuild: self.ctx.is_eq_container_sort(),
                data: {
                    let end: usize = end.try_into().unwrap_or(0);
                    (0..end)
                        .map(|i| state.base_values().get::<i64>(i as i64))
                        .collect()
                }
            } });
        }
        let all_vec_sorts = eg
            .type_info
            .get_arcsorts_by(|f| f.value_type() == Some(TypeId::of::<VecContainer>()));
        for fn_sort in eg.type_info.get_sorts::<FunctionSort>() {
            for vec_sort in &all_vec_sorts {
                try_registering_vec_map(eg, fn_sort.clone(), vec_sort.clone(), arc.clone());
                if vec_sort.name() != arc.name() {
                    try_registering_vec_map(eg, fn_sort.clone(), arc.clone(), vec_sort.clone());
                }
            }
        }
    }

    fn reconstruct_termdag(
        &self,
        _container_values: &ContainerValues,
        _value: Value,
        termdag: &mut TermDag,
        element_terms: Vec<TermId>,
    ) -> TermId {
        vec_term(termdag, element_terms)
    }

    fn rebuild_container_normalizer(&self) -> Option<(String, PrimitiveValidator)> {
        Some((
            "vec-of".to_owned(),
            Arc::new(|termdag: &mut TermDag, args: &[TermId]| {
                Some(vec_term(termdag, args.to_vec()))
            }),
        ))
    }

    fn serialized_name(&self, _container_values: &ContainerValues, _: Value) -> String {
        "vec-of".to_owned()
    }
}

#[derive(Clone, Copy)]
enum VecReadOp {
    Length,
    Contains,
    NotContains,
    Get,
}

/// Vec reads with a sequence-slice fast path and a reconstruction fallback.
#[derive(Clone)]
struct VecRead {
    name: String,
    vec: ArcSort,
    element: ArcSort,
    op: VecReadOp,
}

impl Primitive for VecRead {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        let types = match self.op {
            VecReadOp::Length => vec![self.vec.clone(), I64Sort.to_arcsort()],
            VecReadOp::Contains | VecReadOp::NotContains => {
                vec![
                    self.vec.clone(),
                    self.element.clone(),
                    UnitSort.to_arcsort(),
                ]
            }
            VecReadOp::Get => vec![self.vec.clone(), I64Sort.to_arcsort(), self.element.clone()],
        };
        SimpleTypeConstraint::new(self.name(), types, span.clone()).into_box()
    }
}

impl PurePrim for VecRead {
    fn apply<'a, 'db>(&self, state: crate::PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let [vec_id, rest @ ..] = args else {
            return None;
        };
        match self.op {
            VecReadOp::Length => {
                if !rest.is_empty() {
                    return None;
                }
                let len = state
                    .with_container_sequence::<VecContainer, _>(*vec_id, |values| values.len())
                    .or_else(|| {
                        state
                            .value_to_owned_container::<VecContainer>(*vec_id)
                            .map(|vec| vec.data.len())
                    })?;
                Some(state.base_values().get::<i64>(len as i64))
            }
            VecReadOp::Contains | VecReadOp::NotContains => {
                let [needle] = rest else { return None };
                let contains = state
                    .with_container_sequence::<VecContainer, _>(*vec_id, |values| {
                        values.contains(needle)
                    })
                    .or_else(|| {
                        state
                            .value_to_owned_container::<VecContainer>(*vec_id)
                            .map(|vec| vec.data.contains(needle))
                    })?;
                let succeeds = match self.op {
                    VecReadOp::Contains => contains,
                    VecReadOp::NotContains => !contains,
                    _ => unreachable!(),
                };
                succeeds.then(|| state.base_values().get::<()>(()))
            }
            VecReadOp::Get => {
                let [index] = rest else { return None };
                let index = usize::try_from(state.base_values().unwrap::<i64>(*index)).ok()?;
                state
                    .with_container_sequence::<VecContainer, _>(*vec_id, |values| {
                        values.get(index).copied()
                    })
                    .or_else(|| {
                        state
                            .value_to_owned_container::<VecContainer>(*vec_id)
                            .map(|vec| vec.data.get(index).copied())
                    })?
            }
        }
    }
}

#[derive(Clone, Copy)]
enum VecEditOp {
    Push,
    Pop,
    Set,
    Remove,
}

/// Vec updates with a serialized-slice fast path and a legacy slow fallback.
#[derive(Clone)]
struct VecEdit {
    name: String,
    vec: ArcSort,
    element: ArcSort,
    op: VecEditOp,
}

impl Primitive for VecEdit {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        let types = match self.op {
            VecEditOp::Push => vec![self.vec.clone(), self.element.clone(), self.vec.clone()],
            VecEditOp::Pop => vec![self.vec.clone(), self.vec.clone()],
            VecEditOp::Set => vec![
                self.vec.clone(),
                I64Sort.to_arcsort(),
                self.element.clone(),
                self.vec.clone(),
            ],
            VecEditOp::Remove => vec![self.vec.clone(), I64Sort.to_arcsort(), self.vec.clone()],
        };
        SimpleTypeConstraint::new(self.name(), types, span.clone()).into_box()
    }
}

impl PurePrim for VecEdit {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let vec_id = *args.first()?;
        let build_key = |data: &[Value]| -> Option<Vec<Value>> {
            let mut key = Vec::with_capacity(data.len() + 2);
            key.push(Value::from_usize(self.vec.is_eq_container_sort() as usize));
            match self.op {
                VecEditOp::Push => {
                    let [_, value] = args else { return None };
                    key.extend_from_slice(data);
                    key.push(*value);
                }
                VecEditOp::Pop => {
                    let [_] = args else { return None };
                    key.extend_from_slice(data.get(..data.len().saturating_sub(1))?);
                }
                VecEditOp::Set => {
                    let [_, index, value] = args else {
                        return None;
                    };
                    let index = usize::try_from(state.base_values().unwrap::<i64>(*index)).ok()?;
                    if index >= data.len() {
                        return None;
                    }
                    key.extend_from_slice(data);
                    key[index + 1] = *value;
                }
                VecEditOp::Remove => {
                    let [_, index] = args else { return None };
                    let index = usize::try_from(state.base_values().unwrap::<i64>(*index)).ok()?;
                    if index >= data.len() {
                        return None;
                    }
                    key.extend_from_slice(&data[..index]);
                    key.extend_from_slice(&data[index + 1..]);
                }
            }
            Some(key)
        };

        let key = state
            .with_container_sequence::<VecContainer, _>(vec_id, build_key)
            .or_else(|| {
                state
                    .value_to_owned_container::<VecContainer>(vec_id)
                    .map(|vec| build_key(&vec.data))
            })??;
        Some(state.register_container_sequence::<VecContainer>(&key))
    }
}

#[derive(Clone)]
struct VecAppend {
    name: String,
    vec: ArcSort,
}

impl Primitive for VecAppend {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        AllEqualTypeConstraint::new(self.name(), span.clone())
            .with_all_arguments_sort(self.vec.clone())
            .with_output_sort(self.vec.clone())
            .into_box()
    }
}

impl PurePrim for VecAppend {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let mut key = vec![Value::from_usize(self.vec.is_eq_container_sort() as usize)];
        for value in args {
            if state
                .with_container_sequence::<VecContainer, _>(*value, |values| {
                    key.extend_from_slice(values);
                })
                .is_none()
            {
                key.extend_from_slice(
                    &state.value_to_owned_container::<VecContainer>(*value)?.data,
                );
            }
        }
        Some(state.register_container_sequence::<VecContainer>(&key))
    }
}

/**
 * Register a vec map primitive if the function matches the input and output vec.
 */
pub(crate) fn try_registering_vec_map(
    eg: &mut EGraph,
    fn_: Arc<FunctionSort>,
    input_vec: ArcSort,
    output_vec: ArcSort,
) {
    if fn_.inputs().len() != 1
        || fn_.inputs()[0].name() != input_vec.inner_sorts()[0].name()
        || fn_.output().name() != output_vec.inner_sorts()[0].name()
    {
        return;
    }
    eg.add_pure_primitive(
        VecMap {
            name: "unstable-vec-map".into(),
            vec: input_vec,
            output_vec,
            fn_: fn_.clone(),
        },
        None,
    );
}

pub(crate) fn register_vec_primitives_for_function(eg: &mut EGraph, fn_: Arc<FunctionSort>) {
    let all_vec_sorts = eg
        .type_info
        .get_arcsorts_by(|f| f.value_type() == Some(TypeId::of::<VecContainer>()));
    for input_vec in &all_vec_sorts {
        for output_vec in &all_vec_sorts {
            try_registering_vec_map(eg, fn_.clone(), input_vec.clone(), output_vec.clone());
        }
    }
}

// (unstable-vec-map (Vec[X], [X] -> Y) -> Vec[Y])
// will map the function over all elements in the vec and drop elements where it is undefined.
#[derive(Clone)]
struct VecMap {
    name: String,
    vec: ArcSort,
    output_vec: ArcSort,
    fn_: Arc<FunctionSort>,
}

impl Primitive for VecMap {
    fn name(&self) -> &str {
        &self.name
    }
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            &self.name,
            vec![self.fn_.clone(), self.vec.clone(), self.output_vec.clone()],
            span.clone(),
        )
        .into_box()
    }
}

impl PurePrim for VecMap {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let function = state.prepare_function(args[0])?;
        // Copy before invoking the callback: it may intern another Vec and
        // grow this execution's local prediction storage.
        let input = state
            .with_container_sequence::<VecContainer, _>(args[1], <[Value]>::to_vec)
            .or_else(|| {
                state
                    .value_to_owned_container::<VecContainer>(args[1])
                    .map(|vec| vec.data)
            })?;
        let mut new_data = Vec::with_capacity(input.len());
        for v in input {
            if let Some(mapped) = state.apply_prepared_function(&function, &[v]) {
                new_data.push(mapped);
            }
        }
        let mut key = Vec::with_capacity(new_data.len() + 1);
        key.push(Value::from_usize(
            self.output_vec.is_eq_container_sort() as usize
        ));
        key.extend_from_slice(&new_data);
        Some(state.register_container_sequence::<VecContainer>(&key))
    }
}

// (vec-union Vec[A] Vec[A]) -> Vec[A]
// where A: Eq
// Unions items from two vecs, asserting they are the same length.
#[derive(Clone)]
struct Union {
    name: String,
    vec: ArcSort,
}

// `Union` unions the corresponding entries of two vecs of equal length.
// It writes to the union-find (via `Write::union`), so it implements
// `WritePrim` — valid in rule-action and global-action contexts,
// rejected at rule-build time if used in a rule query.
impl Primitive for Union {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.vec.clone(), self.vec.clone(), self.vec.clone()],
            span.clone(),
        )
        .into_box()
    }
}

impl WritePrim for Union {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::WriteState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        // The union calls below mutate state, so materialize the two borrowed
        // slices only after taking the allocation-free fast lookup path.
        let left = state
            .with_container_sequence::<VecContainer, _>(args[0], <[Value]>::to_vec)
            .or_else(|| {
                state
                    .value_to_owned_container::<VecContainer>(args[0])
                    .map(|vec| vec.data)
            })?;
        let right = state
            .with_container_sequence::<VecContainer, _>(args[1], <[Value]>::to_vec)
            .or_else(|| {
                state
                    .value_to_owned_container::<VecContainer>(args[1])
                    .map(|vec| vec.data)
            })?;
        if left.len() != right.len() {
            return None;
        }
        for (l, r) in zip(left, right) {
            state.union(l, r).ok()?;
        }
        Some(args[0])
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
