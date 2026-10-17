use super::*;
use crate::Write;
use crate::exec_state::Internal;
use inner::MultiSet;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MultiSetContainer {
    pub do_rebuild: bool,
    pub data: MultiSet<Value>,
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
    fn container_type() -> Option<TypeId> {
        Some(TypeId::of::<<Self as ContainerSort>::Container>())
    }

    fn builtin_definitions() -> Vec<proto::Program> {
        use proto as pb;
        let mut definitions: Vec<_> = [
            ("of", vec![], vec![("args", 0)], 1),
            ("single", vec![("x", 0), ("i", 2)], vec![], 1),
            ("pick", vec![("multiset", 1)], vec![], 0),
            ("insert", vec![("multiset", 1), ("value", 0)], vec![], 1),
            ("remove", vec![("multiset", 1), ("value", 0)], vec![], 1),
            ("remove_swapped", vec![("x", 0), ("xs", 1)], vec![], 1),
            ("subtract", vec![("multiset", 1), ("other", 1)], vec![], 1),
            ("subtract_swapped", vec![("x", 1), ("xs", 1)], vec![], 1),
            ("length", vec![("multiset", 1)], vec![], 2),
            ("contains", vec![("multiset", 1), ("value", 0)], vec![], 3),
            (
                "not_contains",
                vec![("multiset", 1), ("value", 0)],
                vec![],
                3,
            ),
            ("contains_swapped", vec![("x", 0), ("xs", 1)], vec![], 3),
            ("not_contains_swapped", vec![("x", 0), ("xs", 1)], vec![], 3),
            (
                "intersection",
                vec![("multiset", 1), ("other", 1)],
                vec![],
                1,
            ),
            ("sum", vec![("multiset", 1), ("other", 1)], vec![], 1),
            ("reset_counts", vec![("multiset", 1)], vec![], 1),
            ("pick_max", vec![("multiset", 1)], vec![], 0),
            ("count", vec![("multiset", 1), ("value", 0)], vec![], 2),
            ("sum_multisets", vec![("xs", 4)], vec![], 1),
            ("union_values", vec![("multiset", 1)], vec![], 0),
        ]
        .into_iter()
        .map(|(key, inputs, varargs, output)| {
            let free = key.ends_with("_swapped");
            let receiver = !free && !matches!(key, "of" | "single" | "sum_multisets");
            let owner = (!free).then_some(pb::BindingOwner {
                kind: Some(pb::binding_owner::Kind::Sort(1)),
            });
            let parameters = inputs
                .iter()
                .chain(&varargs)
                .enumerate()
                .skip(usize::from(receiver));
            pb::Program {
                ir_version: 1,
                sorts: [
                    pb::Sort {
                        kind: Some(pb::sort::Kind::Var(0)),
                        ..Default::default()
                    },
                    pb::Sort {
                        kind: Some(pb::sort::Kind::Family(pb::HostSort {
                            name: "MultiSet".into(),
                            args: vec![0],
                        })),
                        ..Default::default()
                    },
                    pb::Sort {
                        kind: Some(pb::sort::Kind::Family(pb::HostSort {
                            name: "i64".into(),
                            args: vec![],
                        })),
                        ..Default::default()
                    },
                    pb::Sort {
                        kind: Some(pb::sort::Kind::Family(pb::HostSort {
                            name: "Unit".into(),
                            args: vec![],
                        })),
                        ..Default::default()
                    },
                    pb::Sort {
                        kind: Some(pb::sort::Kind::Family(pb::HostSort {
                            name: "MultiSet".into(),
                            args: vec![1],
                        })),
                        ..Default::default()
                    },
                ]
                .into_iter()
                .take(if key == "sum_multisets" { 5 } else { 4 })
                .collect(),
                declarations: vec![
                    pb::Declaration {
                        kind: Some(pb::declaration::Kind::HostSortFamily(pb::HostSortFamily {
                            name: "MultiSet".into(),
                            arity: 1,
                            bindings: Some(pb::SortBindings {
                                python: Some(pb::TypeBinding {
                                    path: vec![
                                        "egglog".into(),
                                        "builtins".into(),
                                        "MultiSet".into(),
                                    ],
                                    type_params: vec!["T".into()],
                                }),
                                rust: Some(pb::TypeBinding {
                                    path: vec![
                                        "egglog_experimental".into(),
                                        "typed".into(),
                                        "builtins".into(),
                                        "MultiSet".into(),
                                    ],
                                    type_params: vec!["T".into()],
                                }),
                                ..Default::default()
                            }),
                        })),
                        ..Default::default()
                    },
                    pb::Declaration {
                        kind: Some(pb::declaration::Kind::HostSortFamily(pb::HostSortFamily {
                            name: "i64".into(),
                            arity: 0,
                            ..Default::default()
                        })),
                        ..Default::default()
                    },
                    pb::Declaration {
                        kind: Some(pb::declaration::Kind::HostSortFamily(pb::HostSortFamily {
                            name: "Unit".into(),
                            arity: 0,
                            ..Default::default()
                        })),
                        ..Default::default()
                    },
                    pb::Declaration {
                        kind: Some(pb::declaration::Kind::HostPrimitive(pb::HostPrimitive {
                            name: format!("egglog.core.multiset.{key}"),
                            typing: Some(pb::host_primitive::Typing::Signature(
                                pb::GenericSignature {
                                    type_params: vec!["T".into()],
                                    inputs: inputs
                                        .iter()
                                        .map(|(name, sort)| pb::Arg {
                                            name: (*name).into(),
                                            sort: *sort,
                                        })
                                        .collect(),
                                    varargs: varargs
                                        .iter()
                                        .map(|(name, sort)| pb::Arg {
                                            name: (*name).into(),
                                            sort: *sort,
                                        })
                                        .collect(),
                                    output: Some(output),
                                    equality_type_params: if key == "union_values" {
                                        vec![0]
                                    } else {
                                        vec![]
                                    },
                                },
                            )),
                        })),
                        bindings: Some(pb::CallableBindings {
                            python: Some(pb::PythonBindings {
                                views: vec![pb::PythonCallable {
                                    kind: if free {
                                        pb::PythonCallKind::Function
                                    } else {
                                        match key {
                                            "of" => pb::PythonCallKind::Initializer,
                                            "single" | "sum_multisets" => {
                                                pb::PythonCallKind::ClassMethod
                                            }
                                            _ => pb::PythonCallKind::Method,
                                        }
                                    }
                                    .into(),
                                    path: if free {
                                        vec![
                                            "egglog".into(),
                                            "builtins".into(),
                                            format!("multiset_{key}"),
                                        ]
                                    } else {
                                        match key {
                                            "of" => vec![],
                                            "subtract" => vec!["__sub__".into()],
                                            "intersection" => vec!["__and__".into()],
                                            "sum" => vec!["__add__".into()],
                                            _ => vec![key.into()],
                                        }
                                    },
                                    owner,
                                    receiver: receiver.then_some(0),
                                    params: parameters
                                        .clone()
                                        .map(|(i, (name, _))| pb::PythonParameter {
                                            core_input: Some(i as u32),
                                            name: (*name).into(),
                                            default_expr: None,
                                        })
                                        .collect(),
                                    ..Default::default()
                                }],
                            }),
                            rust: Some(pb::RustBindings {
                                views: vec![pb::RustCallable {
                                    path: if free {
                                        vec![
                                            "egglog_experimental".into(),
                                            "typed".into(),
                                            "builtins".into(),
                                            format!("multiset_{key}"),
                                        ]
                                    } else {
                                        vec![if key == "of" { "new" } else { key }.into()]
                                    },
                                    owner,
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
                                    symbol: format!("multiset-{}", key.replace('_', "-")),
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
        .collect();

        // Reuse the same owned family declarations; only these seven callable
        // records add structural function patterns to the ordinary catalog.
        let families = definitions[0]
            .declarations
            .iter()
            .filter(|d| matches!(&d.kind, Some(pb::declaration::Kind::HostSortFamily(_))))
            .cloned()
            .collect::<Vec<_>>();
        definitions.extend(
            [
                ("map", vec![("f", 4), ("xs", 1)], 3, vec![0], 2, Some(1)),
                ("filter", vec![("f", 4), ("xs", 1)], 1, vec![0], 2, Some(1)),
                (
                    "filter_not",
                    vec![("f", 4), ("xs", 1)],
                    1,
                    vec![0],
                    2,
                    Some(1),
                ),
                (
                    "reduce",
                    vec![("f", 4), ("initial", 0), ("xs", 1)],
                    0,
                    vec![0, 0],
                    0,
                    None,
                ),
                ("flat_map", vec![("f", 4), ("xs", 1)], 1, vec![0], 1, None),
                (
                    "fill_index",
                    vec![("multiset", 1), ("f", 4)],
                    2,
                    vec![1, 0],
                    3,
                    Some(0),
                ),
                (
                    "clear_index",
                    vec![("multiset", 1), ("f", 4)],
                    2,
                    vec![1, 0],
                    3,
                    Some(0),
                ),
            ]
            .into_iter()
            .map(|(key, inputs, output, params, result, receiver)| {
                let map = key == "map";
                let free = receiver.is_none();
                let owner = (!free).then_some(pb::BindingOwner {
                    kind: Some(pb::binding_owner::Kind::Sort(1)),
                });
                let name = if key == "reduce" {
                    "multiset_fold".to_owned()
                } else if free {
                    format!("multiset_{key}")
                } else {
                    key.to_owned()
                };
                let parameters = inputs
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| receiver != Some(*i as u32));
                let mut declarations = families.clone();
                declarations.push(pb::Declaration {
                    kind: Some(pb::declaration::Kind::HostPrimitive(pb::HostPrimitive {
                        name: format!("egglog.core.multiset.{key}"),
                        typing: Some(pb::host_primitive::Typing::Signature(
                            pb::GenericSignature {
                                type_params: if map {
                                    vec!["T".into(), "V".into()]
                                } else {
                                    vec!["T".into()]
                                },
                                inputs: inputs
                                    .iter()
                                    .map(|(name, sort)| pb::Arg {
                                        name: (*name).into(),
                                        sort: *sort,
                                    })
                                    .collect(),
                                output: Some(output),
                                ..Default::default()
                            },
                        )),
                    })),
                    bindings: Some(pb::CallableBindings {
                        python: Some(pb::PythonBindings {
                            views: vec![pb::PythonCallable {
                                kind: if free {
                                    pb::PythonCallKind::Function
                                } else {
                                    pb::PythonCallKind::Method
                                }
                                .into(),
                                path: if free {
                                    vec!["egglog".into(), "builtins".into(), name.clone()]
                                } else {
                                    vec![name.clone()]
                                },
                                owner,
                                receiver,
                                params: parameters
                                    .clone()
                                    .map(|(i, (name, _))| pb::PythonParameter {
                                        core_input: Some(i as u32),
                                        name: (*name).into(),
                                        default_expr: None,
                                    })
                                    .collect(),
                                ..Default::default()
                            }],
                        }),
                        rust: Some(pb::RustBindings {
                            views: vec![pb::RustCallable {
                                path: if free {
                                    vec![
                                        "egglog_experimental".into(),
                                        "typed".into(),
                                        "builtins".into(),
                                        name,
                                    ]
                                } else {
                                    vec![name]
                                },
                                owner,
                                receiver: receiver.map(|i| pb::RustReceiver {
                                    core_input: Some(i),
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
                                symbol: format!("unstable-multiset-{}", key.replace('_', "-")),
                                datatype_member: false,
                            }],
                        }),
                    }),
                    ..Default::default()
                });
                pb::Program {
                    ir_version: 1,
                    sorts: vec![
                        pb::Sort {
                            kind: Some(pb::sort::Kind::Var(0)),
                            ..Default::default()
                        },
                        pb::Sort {
                            kind: Some(pb::sort::Kind::Family(pb::HostSort {
                                name: "MultiSet".into(),
                                args: vec![0],
                            })),
                            ..Default::default()
                        },
                        pb::Sort {
                            kind: Some(if map {
                                pb::sort::Kind::Var(1)
                            } else {
                                pb::sort::Kind::Family(pb::HostSort {
                                    name: "Unit".into(),
                                    args: vec![],
                                })
                            }),
                            ..Default::default()
                        },
                        pb::Sort {
                            kind: Some(pb::sort::Kind::Family(pb::HostSort {
                                name: if map { "MultiSet" } else { "i64" }.into(),
                                args: if map { vec![2] } else { vec![] },
                            })),
                            ..Default::default()
                        },
                        pb::Sort {
                            kind: Some(pb::sort::Kind::Func(pb::FuncSort { params, result })),
                            ..Default::default()
                        },
                    ],
                    declarations,
                    ..Default::default()
                }
            }),
        );
        definitions
    }

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
        eg.type_info
            .register_builtin_sort("MultiSet", arc.clone(), vec![self.element.clone()])
            .expect("invalid MultiSet family instance");
        let [
            of,
            single,
            pick,
            insert,
            remove,
            remove_swapped,
            subtract,
            subtract_swapped,
            length,
            contains,
            not_contains,
            contains_swapped,
            not_contains_swapped,
            intersection,
            sum,
            reset_counts,
            pick_max,
            count,
        ] = [
            "of",
            "single",
            "pick",
            "insert",
            "remove",
            "remove_swapped",
            "subtract",
            "subtract_swapped",
            "length",
            "contains",
            "not_contains",
            "contains_swapped",
            "not_contains_swapped",
            "intersection",
            "sum",
            "reset_counts",
            "pick_max",
            "count",
        ]
        .map(|key| {
            eg.type_info
                .instantiate_builtin(&format!("egglog.core.multiset.{key}"), &arc)
                .unwrap()
        });

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

        add_primitive_with_validator!(eg, "multiset-of" [instance = of] = {self.clone(): MultiSetSort} [xs: #] -> @MultiSetContainer { MultiSetContainer {
            do_rebuild: self.ctx.is_eq_container_sort(),
            data: xs.collect()
        } }, multiset_of_validator);

        add_primitive!(eg, "multiset-single" [instance = single] = {self.clone(): MultiSetSort} |x: #, i: i64| -?> @MultiSetContainer {
            i.try_into().ok().map(|i|
                MultiSetContainer {
                do_rebuild: self.ctx.is_eq_container_sort(),
                data: std::iter::repeat_n(x, i).collect()
            })
        });
        add_primitive!(eg, "multiset-pick" [instance = pick] = |xs: @MultiSetContainer| -?> # { xs.data.pick().copied() });
        add_primitive!(eg, "multiset-insert" [instance = insert] = |mut xs: @MultiSetContainer, x: #| -> @MultiSetContainer { MultiSetContainer { data: xs.data.insert( x) , ..xs } });
        add_primitive!(eg, "multiset-remove" [instance = remove] = |mut xs: @MultiSetContainer, x: #| -?> @MultiSetContainer { Some(MultiSetContainer { data: xs.data.remove(&x)?, ..xs } )});
        add_primitive!(eg, "multiset-remove-swapped" [instance = remove_swapped] = |x: #, mut xs: @MultiSetContainer| -?> @MultiSetContainer { Some(MultiSetContainer { data: xs.data.remove(&x)?, ..xs }) });
        add_primitive!(eg, "multiset-subtract" [instance = subtract] = |mut xs: @MultiSetContainer, other: @MultiSetContainer| -?> @MultiSetContainer { Some(MultiSetContainer { data: xs.data.subtract(&other.data)?, ..xs }) });
        add_primitive!(eg, "multiset-subtract-swapped" [instance = subtract_swapped] = |other: @MultiSetContainer, mut xs: @MultiSetContainer| -?> @MultiSetContainer { Some(MultiSetContainer { data: xs.data.subtract(&other.data)?, ..xs }) });
        add_primitive_with_validator!(eg, "multiset-length" [instance = length] = |xs: @MultiSetContainer| -> i64 { xs.data.len() as i64 }, multiset_length_validator);
        add_primitive_with_validator!(eg, "multiset-contains" [instance = contains] = |xs: @MultiSetContainer, x: #| -?> () { ( xs.data.contains(&x)).then_some(()) }, multiset_contains_validator);
        add_primitive_with_validator!(eg, "multiset-not-contains" [instance = not_contains] = |xs: @MultiSetContainer, x: #| -?> () { (!xs.data.contains(&x)).then_some(()) }, multiset_not_contains_validator);
        add_primitive!(eg, "multiset-contains-swapped" [instance = contains_swapped] = |x: #, xs: @MultiSetContainer| -?> () { (xs.data.contains(&x)).then_some(()) });
        add_primitive!(eg, "multiset-not-contains-swapped" [instance = not_contains_swapped] = |x: #, xs: @MultiSetContainer| -?> () { (!xs.data.contains(&x)).then_some(()) });
        add_primitive!(eg, "multiset-intersection" [instance = intersection] = |xs: @MultiSetContainer, ys: @MultiSetContainer| -> @MultiSetContainer { MultiSetContainer { data: xs.data.intersection(ys.data), ..xs } });
        add_primitive!(eg, "multiset-sum" [instance = sum] = |xs: @MultiSetContainer, ys: @MultiSetContainer| -> @MultiSetContainer { MultiSetContainer { data: xs.data.sum(ys.data), ..xs } });
        // Set counts to one
        add_primitive!(eg, "multiset-reset-counts" [instance = reset_counts] = |mut xs: @MultiSetContainer| -> @MultiSetContainer { {
            let mut new_data = MultiSet::<Value>::new();
            for (v, _) in xs.data.iter_counts() {
                new_data.insert_multiple_mut(v, 1);
            }
            MultiSetContainer { data: new_data, ..xs }
        }});
        add_primitive!(eg, "multiset-pick-max" [instance = pick_max] = |xs: @MultiSetContainer| -?> # {
            Some(xs.data.iter_counts().max_by_key(|(_, c)| *c)?.0)
        });
        add_primitive!(eg, "multiset-count" [instance = count] = |xs: @MultiSetContainer, x: #| -> i64 {
            xs.data.iter_counts().find(|(v, _)| *v == x).map(|(_, c)| c as i64).unwrap_or(0)
        });

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
                    instance: eg
                        .type_info
                        .instantiate_builtin_signature(
                            "egglog.core.multiset.sum_multisets",
                            std::slice::from_ref(&arc),
                            &[],
                            &other_multiset_sort,
                        )
                        .unwrap(),
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
                    instance: eg
                        .type_info
                        .instantiate_builtin_signature(
                            "egglog.core.multiset.union_values",
                            std::slice::from_ref(&arc),
                            &[],
                            &self.element,
                        )
                        .unwrap(),
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
            instance: eg
                .type_info
                .instantiate_builtin_signature(
                    "egglog.core.multiset.map",
                    &[fn_.clone(), input_ms.clone()],
                    &[],
                    &output_ms,
                )
                .unwrap(),
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
                instance: eg
                    .type_info
                    .instantiate_builtin_signature(
                        "egglog.core.multiset.filter",
                        &[fn_.clone(), multiset.clone()],
                        &[],
                        &multiset,
                    )
                    .unwrap(),
                skip_empty: true,
            },
            None,
        );
        eg.add_pure_primitive(
            Filter {
                name: "unstable-multiset-filter-not".into(),
                instance: eg
                    .type_info
                    .instantiate_builtin_signature(
                        "egglog.core.multiset.filter_not",
                        &[fn_.clone(), multiset.clone()],
                        &[],
                        &multiset,
                    )
                    .unwrap(),
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
                instance: eg
                    .type_info
                    .instantiate_builtin_signature(
                        "egglog.core.multiset.reduce",
                        &[fn_.clone(), element.clone(), multiset.clone()],
                        &[],
                        &element,
                    )
                    .unwrap(),
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
                instance: eg
                    .type_info
                    .instantiate_builtin_signature(
                        "egglog.core.multiset.fill_index",
                        &[multiset.clone(), fn_.clone()],
                        &[],
                        &unit,
                    )
                    .unwrap(),
            },
            None,
        );
        eg.add_write_primitive(
            ClearIndex {
                name: "unstable-multiset-clear-index".into(),
                instance: eg
                    .type_info
                    .instantiate_builtin_signature(
                        "egglog.core.multiset.clear_index",
                        &[multiset.clone(), fn_.clone()],
                        &[],
                        &unit,
                    )
                    .unwrap(),
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
                instance: eg
                    .type_info
                    .instantiate_builtin_signature(
                        "egglog.core.multiset.flat_map",
                        &[fn_.clone(), multiset.clone()],
                        &[],
                        &multiset,
                    )
                    .unwrap(),
            },
            None,
        );
    }
}

#[derive(Clone)]
struct Map {
    name: String,
    instance: builtin::BuiltinInstance,
}

impl Primitive for Map {
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

impl PurePrim for Map {
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
        let multiset = state
            .container_values()
            .get_val::<MultiSetContainer>(args[1])
            .unwrap()
            .clone();
        let mut new_data = MultiSet::<Value>::new();
        for (v, c) in multiset.data.iter_counts() {
            if let Some(mapped) = state.apply_function(&fc, &[v]) {
                new_data.insert_multiple_mut(mapped, c);
            }
        }
        let new_ms = MultiSetContainer {
            data: new_data,
            ..multiset
        };
        Some(state.register_container(new_ms))
    }
}

// (unstable-multiset-fill-index ms: MultiSet[X] index_fn: [MultiSet[X], X] -> i64) -> Unit
// will set the index function for all elements in the multiset
#[derive(Clone)]
struct FillIndex {
    name: String,
    instance: builtin::BuiltinInstance,
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
        self.instance.type_constraints(span)
    }

    fn builtin_instance(&self) -> Option<&builtin::BuiltinInstance> {
        Some(&self.instance)
    }
}

impl FullPrim for FillIndex {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::FullState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let fc = state
            .container_values()
            .get_val::<FunctionContainer>(args[1])
            .unwrap()
            .clone();
        let multiset = state
            .container_values()
            .get_val::<MultiSetContainer>(args[0])
            .unwrap()
            .clone();
        let action = match fc.0 {
            ResolvedFunctionId::Constructor(a) | ResolvedFunctionId::Function(a) => a,
            // Primitive functions cannot be used with
            // unstable-multiset-fill-index, since they cannot be set.
            ResolvedFunctionId::Primitive { .. } => return None,
        };
        let unit_val = state.base_values().get::<()>(());
        let es = state.raw_exec_state();
        for (v, c) in multiset.data.iter_counts() {
            let mut row: Vec<_> =
                fc.1.iter()
                    .map(|(_, value)| *value)
                    .chain([args[0], v])
                    .collect();
            // Skip the whole fill if any index row already exists.
            // This relies on `unstable-multiset-fill-index` writing all
            // rows for a given multiset in one pass.
            if action.lookup(es, &row).is_some() {
                break;
            }
            row.push(es.base_values().get::<i64>(c.try_into().ok()?));
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
    instance: builtin::BuiltinInstance,
}

// `ClearIndex` removes table rows; action-only.
impl Primitive for ClearIndex {
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

impl WritePrim for ClearIndex {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::WriteState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let fc = state
            .container_values()
            .get_val::<FunctionContainer>(args[1])
            .unwrap()
            .clone();
        let multiset = state
            .container_values()
            .get_val::<MultiSetContainer>(args[0])
            .unwrap()
            .clone();
        let action = match fc.0 {
            ResolvedFunctionId::Constructor(a) | ResolvedFunctionId::Function(a) => a,
            // Primitive functions cannot be used with
            // unstable-multiset-clear-index, since they cannot be deleted.
            ResolvedFunctionId::Primitive { .. } => return None,
        };
        let unit_val = state.base_values().get::<()>(());
        let es = state.raw_exec_state();
        for (v, _) in multiset.data.iter_counts() {
            let row: Vec<_> =
                fc.1.iter()
                    .map(|(_, value)| *value)
                    .chain([args[0], v])
                    .collect();
            action.remove(es, &row);
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
    instance: builtin::BuiltinInstance,
}

impl Primitive for FlatMap {
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

impl PurePrim for FlatMap {
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
        let multiset = state
            .container_values()
            .get_val::<MultiSetContainer>(args[1])
            .unwrap()
            .clone();
        let mut new_data = MultiSet::<Value>::new();
        for (v, c) in multiset.data.iter_counts() {
            let mapped = state.apply_function(&fc, &[v]);
            if let Some(mapped_ms) = mapped {
                let mapped_ms = state
                    .container_values()
                    .get_val::<MultiSetContainer>(mapped_ms)
                    .unwrap();
                for (mv, mc) in mapped_ms.data.iter_counts() {
                    new_data.insert_multiple_mut(mv, c.checked_mul(mc)?);
                }
            } else {
                new_data.insert_multiple_mut(v, c);
            }
        }
        let new_container = MultiSetContainer {
            data: new_data,
            ..multiset
        };
        Some(state.register_container(new_container))
    }
}

// (unstable-multiset-filter (MultiSet[X], [X] -> Unit) -> MultiSet[X])
// will filter the elements in the multiset based on whether the function is defined for them.
// If skip_empty is true, it will keep elements where the function is defined, otherwise it will keep elements where the function is not defined.
#[derive(Clone)]
struct Filter {
    name: String,
    instance: builtin::BuiltinInstance,
    skip_empty: bool,
}

impl Primitive for Filter {
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

impl PurePrim for Filter {
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
        let multiset = state
            .container_values()
            .get_val::<MultiSetContainer>(args[1])
            .unwrap()
            .clone();
        let mut new_data = MultiSet::<Value>::new();
        for (v, c) in multiset.data.iter_counts() {
            let mapped = state.apply_function(&fc, &[v]);
            if mapped.is_some() == self.skip_empty {
                new_data.insert_multiple_mut(v, c);
            }
        }
        let new_ms = MultiSetContainer {
            data: new_data,
            ..multiset
        };
        Some(state.register_container(new_ms))
    }
}

// (multiset-sum-multisets (MultiSet[MultiSet[X]]) -> MultiSet[X])
// will sum all multisets in the outer multiset into a single multiset

#[derive(Clone)]
struct SumMultisets {
    name: String,
    multiset: ArcSort,
    instance: builtin::BuiltinInstance,
}

// `SumMultisets` flattens a multiset of multisets. Only reads container
// contents and registers the result — pure.
impl Primitive for SumMultisets {
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

impl PurePrim for SumMultisets {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let mut data = MultiSet::<Value>::new();
        let ms_of_ms = state
            .container_values()
            .get_val::<MultiSetContainer>(args[0])
            .unwrap()
            .clone();
        for (ms_value, counts) in ms_of_ms.data.iter_counts() {
            let ms = state
                .container_values()
                .get_val::<MultiSetContainer>(ms_value)
                .unwrap();
            for (v, c) in ms.data.iter_counts() {
                data.insert_multiple_mut(v, c.checked_mul(counts)?);
            }
        }
        let multiset = MultiSetContainer {
            data,
            do_rebuild: self.multiset.is_eq_container_sort(),
        };
        Some(state.register_container(multiset))
    }
}

// (unstable-multiset-reduce ([X, X] -> X, X, MultiSet[X]) -> X
// will reduce the multiset using the provided binary function and initial value
// Won't use the initial value if the multiset is non-empty.
#[derive(Clone)]
struct Reduce {
    name: String,
    instance: builtin::BuiltinInstance,
}

impl Primitive for Reduce {
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

impl PurePrim for Reduce {
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
        let initial = args[1];
        let multiset = state
            .container_values()
            .get_val::<MultiSetContainer>(args[2])
            .unwrap()
            .clone();
        let mut values = multiset.data.iter().cloned().collect::<Vec<_>>();
        let mut acc = if values.is_empty() {
            initial
        } else {
            values.remove(0)
        };
        for v in values {
            acc = state.apply_function(&fc, &[acc, v])?;
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
    instance: builtin::BuiltinInstance,
}

// `UnionValues` writes to the union-find; action-only.
impl Primitive for UnionValues {
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

impl WritePrim for UnionValues {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::WriteState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let values = state
            .container_values()
            .get_val::<MultiSetContainer>(args[0])?
            .clone()
            .data;
        let values: Vec<_> = values.iter_counts().map(|(v, _c)| v).collect();
        if values.is_empty() {
            return None;
        }
        let first = values[0];
        for v in values.into_iter().skip(1) {
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

        pub fn insert_multiple_mut(&mut self, value: T, n: usize) {
            self.1 += n;
            if let Some(v) = self.0.get(&value) {
                self.0.insert(value, v + n);
            } else {
                self.0.insert(value, n);
            }
        }

        /// Compute the sum of two multisets.
        pub fn sum(mut self, MultiSet(other_map, other_count): Self) -> Self {
            let target_count = self.1 + other_count;
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
