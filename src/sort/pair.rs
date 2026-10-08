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
    fn builtin_definitions() -> Vec<proto::Program> {
        use proto as pb;
        [
            ("make", "pair", vec![("first", 0), ("second", 1)], 2),
            ("first", "pair-first", vec![("pair", 2)], 0),
            ("second", "pair-second", vec![("pair", 2)], 1),
        ]
        .into_iter()
        .map(|(key, alias, inputs, output)| pb::Program {
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
                        name: "Pair".into(),
                        args: vec![0, 1],
                    })),
                    ..Default::default()
                },
            ],
            declarations: vec![
                pb::Declaration {
                    kind: Some(pb::declaration::Kind::HostSortFamily(pb::HostSortFamily {
                        name: "Pair".into(),
                        arity: 2,
                        bindings: None,
                    })),
                    ..Default::default()
                },
                pb::Declaration {
                    kind: Some(pb::declaration::Kind::HostPrimitive(pb::HostPrimitive {
                        name: format!("egglog.core.pair.{key}"),
                        typing: Some(pb::host_primitive::Typing::Signature(
                            pb::GenericSignature {
                                type_params: vec!["A".into(), "B".into()],
                                inputs: inputs
                                    .into_iter()
                                    .map(|(name, sort)| pb::Arg {
                                        name: name.into(),
                                        sort,
                                    })
                                    .collect(),
                                output: Some(output),
                                varargs: vec![],
                            },
                        )),
                    })),
                    bindings: Some(pb::CallableBindings {
                        egglog: Some(pb::EgglogBindings {
                            views: vec![pb::EgglogCallable {
                                symbol: alias.into(),
                                datatype_member: false,
                            }],
                        }),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ],
            ..Default::default()
        })
        .collect()
    }

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
        eg.type_info
            .register_builtin_sort(
                "Pair",
                arc.clone(),
                vec![self.first.clone(), self.second.clone()],
            )
            .unwrap();
        let make = eg
            .type_info
            .instantiate_builtin("egglog.core.pair.make", &arc)
            .unwrap();
        let first = eg
            .type_info
            .instantiate_builtin("egglog.core.pair.first", &arc)
            .unwrap();
        let second = eg
            .type_info
            .instantiate_builtin("egglog.core.pair.second", &arc)
            .unwrap();

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

        add_primitive_with_validator!(eg, "pair" [instance = make] = {self.clone(): PairSort} |x: #, y: #| -> @PairContainer {
            PairContainer {
                do_rebuild_first: self.ctx.first.is_eq_sort() || self.ctx.first.is_eq_container_sort(),
                do_rebuild_second: self.ctx.second.is_eq_sort() || self.ctx.second.is_eq_container_sort(),
                first: x,
                second: y,
            }
        }, pair_term);

        add_primitive_with_validator!(eg, "pair-first" [instance = first] = |xs: @PairContainer| -> # { xs.first }, pair_first_validator);
        add_primitive_with_validator!(eg, "pair-second" [instance = second] = |xs: @PairContainer| -> # { xs.second }, pair_second_validator);
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
