use crate::Write;
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
    fn container_type() -> Option<TypeId> {
        Some(TypeId::of::<<Self as ContainerSort>::Container>())
    }

    fn family_description() -> Option<builtin::FamilyDescription> {
        Some(builtin::FamilyDescription {
            name: "Vec".into(),
            arity: 1,
        })
    }

    fn builtin_definitions() -> Vec<builtin::BuiltinDescription> {
        use builtin::{
            Argument, BuiltinDescription, BuiltinTyping, Signature, SortPattern, TypeParameter,
        };
        let mut definitions: Vec<_> = [
            ("empty", "vec-empty", vec![], vec![], 1),
            ("of", "vec-of", vec![], vec![("values", 0)], 1),
            ("get", "vec-get", vec![("vec", 1), ("index", 2)], vec![], 0),
            ("range", "vec-range", vec![("end", 2)], vec![], 1),
            ("append", "vec-append", vec![], vec![("others", 1)], 1),
            (
                "union",
                "vec-union",
                vec![("vec", 1), ("other", 1)],
                vec![],
                1,
            ),
            (
                "push",
                "vec-push",
                vec![("vec", 1), ("value", 0)],
                vec![],
                1,
            ),
            ("pop", "vec-pop", vec![("vec", 1)], vec![], 1),
            ("length", "vec-length", vec![("vec", 1)], vec![], 2),
            (
                "contains",
                "vec-contains",
                vec![("vec", 1), ("value", 0)],
                vec![],
                3,
            ),
            (
                "not_contains",
                "vec-not-contains",
                vec![("vec", 1), ("value", 0)],
                vec![],
                3,
            ),
            (
                "set",
                "vec-set",
                vec![("vec", 1), ("index", 2), ("value", 0)],
                vec![],
                1,
            ),
            (
                "remove",
                "vec-remove",
                vec![("vec", 1), ("index", 2)],
                vec![],
                1,
            ),
        ]
        .into_iter()
        .map(|(key, alias, inputs, tail, output)| {
            let element = if key == "range" {
                SortPattern::Family {
                    name: "i64".into(),
                    args: vec![],
                }
            } else {
                SortPattern::Param(0)
            };
            let patterns = [
                element.clone(),
                SortPattern::Family {
                    name: "Vec".into(),
                    args: vec![element],
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
            BuiltinDescription {
                key: format!("egglog.core.vec.{key}"),
                aliases: vec![alias.into()],
                typing: BuiltinTyping::Signature(Signature {
                    params: if key == "range" {
                        vec![]
                    } else {
                        vec![TypeParameter {
                            name: "T".into(),
                            equality: key == "union",
                        }]
                    },
                    fixed: inputs
                        .into_iter()
                        .map(|(name, sort)| Argument {
                            name: name.into(),
                            sort: patterns[sort].clone(),
                        })
                        .collect(),
                    tail: tail
                        .into_iter()
                        .map(|(name, sort)| Argument {
                            name: name.into(),
                            sort: patterns[sort].clone(),
                        })
                        .collect(),
                    output: patterns[output].clone(),
                }),
            }
        })
        .collect();
        definitions.push(BuiltinDescription {
            key: "egglog.core.vec.map".into(),
            aliases: vec!["unstable-vec-map".into()],
            typing: BuiltinTyping::Signature(Signature {
                params: vec![
                    TypeParameter {
                        name: "T".into(),
                        equality: false,
                    },
                    TypeParameter {
                        name: "U".into(),
                        equality: false,
                    },
                ],
                fixed: vec![
                    Argument {
                        name: "f".into(),
                        sort: SortPattern::Function {
                            inputs: vec![SortPattern::Param(0)],
                            output: Box::new(SortPattern::Param(1)),
                        },
                    },
                    Argument {
                        name: "xs".into(),
                        sort: SortPattern::Family {
                            name: "Vec".into(),
                            args: vec![SortPattern::Param(0)],
                        },
                    },
                ],
                tail: vec![],
                output: SortPattern::Family {
                    name: "Vec".into(),
                    args: vec![SortPattern::Param(1)],
                },
            }),
        });
        definitions
    }

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
        eg.type_info
            .register_builtin_sort("Vec", arc.clone(), vec![self.element.clone()])
            .expect("invalid Vec family instance");
        let empty = eg
            .type_info
            .instantiate_builtin("egglog.core.vec.empty", &arc)
            .unwrap();
        let of = eg
            .type_info
            .instantiate_builtin("egglog.core.vec.of", &arc)
            .unwrap();
        let get = eg
            .type_info
            .instantiate_builtin("egglog.core.vec.get", &arc)
            .unwrap();
        let append = eg
            .type_info
            .instantiate_builtin("egglog.core.vec.append", &arc)
            .unwrap();
        let [push, pop, length, contains, not_contains, set, remove] = [
            "push",
            "pop",
            "length",
            "contains",
            "not_contains",
            "set",
            "remove",
        ]
        .map(|key| {
            eg.type_info
                .instantiate_builtin(&format!("egglog.core.vec.{key}"), &arc)
                .unwrap()
        });

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

        add_primitive_with_validator!(eg, "vec-empty" [instance = empty] = {self.clone(): VecSort} || -> @VecContainer { VecContainer {
            do_rebuild: self.ctx.is_eq_container_sort(),
            data: Vec::new()
        } }, vec_empty_validator);
        add_primitive_with_validator!(eg, "vec-of" [instance = of] = {self.clone(): VecSort} [xs: #] -> @VecContainer { VecContainer {
            do_rebuild: self.ctx.is_eq_container_sort(),
            data: xs                     .collect()
        } }, vec_of_validator);
        add_primitive!(eg, "vec-append" [instance = append] = {self.clone(): VecSort} [xs: @VecContainer] -> @VecContainer { VecContainer {
            do_rebuild: self.ctx.is_eq_container_sort(),
            data: xs.flat_map(|x| x.data).collect()
        } });

        add_primitive!(eg, "vec-push" [instance = push] = |mut xs: @VecContainer, x: #| -> @VecContainer {{ xs.data.push(x); xs }});
        add_primitive!(eg, "vec-pop" [instance = pop] = |mut xs: @VecContainer| -> @VecContainer {{ xs.data.pop();   xs }});

        add_primitive_with_validator!(eg, "vec-length" [instance = length] = |xs: @VecContainer| -> i64 { xs.data.len() as i64 }, vec_length_validator);
        add_primitive_with_validator!(eg, "vec-contains" [instance = contains] = |xs: @VecContainer, x: #| -?> () { ( xs.data.contains(&x)).then_some(()) }, vec_contains_validator);
        add_primitive_with_validator!(eg, "vec-not-contains" [instance = not_contains] = |xs: @VecContainer, x: #| -?> () { (!xs.data.contains(&x)).then_some(()) }, vec_not_contains_validator);

        add_primitive_with_validator!(eg, "vec-get" [instance = get] = |xs: @VecContainer, i: i64| -?> # { xs.data.get(i as usize).copied() }, vec_get_validator);
        add_primitive!(eg, "vec-set" [instance = set] = |mut xs: @VecContainer, i: i64, x: #| -?> @VecContainer {{ let idx = usize::try_from(i).ok()?; if idx >= xs.data.len() { None } else { xs.data[idx] = x; Some(xs) } }});
        add_primitive!(eg, "vec-remove" [instance = remove] = |mut xs: @VecContainer, i: i64| -?> @VecContainer {{ let idx = usize::try_from(i).ok()?; if idx >= xs.data.len() { None } else { xs.data.remove(idx); Some(xs) } }});
        if self.element.is_eq_sort() {
            eg.add_write_primitive(
                Union {
                    name: "vec-union".into(),
                    instance: eg
                        .type_info
                        .instantiate_builtin("egglog.core.vec.union", &arc)
                        .unwrap(),
                },
                None,
            );
        }
        // vec-range
        if self.element.name() == "i64" {
            let range = eg
                .type_info
                .instantiate_builtin("egglog.core.vec.range", &arc)
                .unwrap();
            add_primitive!(eg, "vec-range" [instance = range] = {self.clone(): VecSort} |end: i64| -> @VecContainer { VecContainer {
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
            instance: eg
                .type_info
                .instantiate_builtin_signature(
                    "egglog.core.vec.map",
                    &[fn_.clone(), input_vec.clone()],
                    &[],
                    &output_vec,
                )
                .unwrap(),
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
    instance: builtin::BuiltinInstance,
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

    fn builtin_instance(&self) -> Option<&builtin::BuiltinInstance> {
        Some(&self.instance)
    }
}

impl PurePrim for VecMap {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let fc = state
            .container_values()
            .get_val::<FunctionContainer>(args[0])
            .unwrap()
            .clone();
        let vec = state
            .container_values()
            .get_val::<VecContainer>(args[1])
            .unwrap()
            .clone();
        let mut new_data = Vec::with_capacity(vec.data.len());
        for v in vec.data {
            if let Some(mapped) = state.apply_function(&fc, &[v]) {
                new_data.push(mapped);
            }
        }
        let new_vec = VecContainer {
            do_rebuild: self.output_vec.is_eq_container_sort(),
            data: new_data,
        };
        Some(state.register_container(new_vec))
    }
}

// (vec-union Vec[A] Vec[A]) -> Vec[A]
// where A: Eq
// Unions items from two vecs, asserting they are the same length.
#[derive(Clone)]
struct Union {
    name: String,
    instance: builtin::BuiltinInstance,
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
        self.instance.type_constraints(span)
    }

    fn builtin_instance(&self) -> Option<&builtin::BuiltinInstance> {
        Some(&self.instance)
    }
}

impl WritePrim for Union {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::WriteState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let left = state
            .container_values()
            .get_val::<VecContainer>(args[0])?
            .clone()
            .data;
        let right = state
            .container_values()
            .get_val::<VecContainer>(args[1])?
            .clone()
            .data;
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
