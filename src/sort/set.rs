use super::*;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SetContainer {
    pub do_rebuild: bool,
    pub data: BTreeSet<Value>,
}

impl ContainerValue for SetContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn ValueRebuilder) -> bool {
        if self.do_rebuild {
            let mut xs: Vec<_> = self.data.iter().copied().collect();
            let changed = rebuilder.rebuild_slice(&mut xs);
            self.data = xs.into_iter().collect();
            changed
        } else {
            false
        }
    }
    fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        self.data.iter().copied()
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
    fn container_type() -> Option<TypeId> {
        Some(TypeId::of::<<Self as ContainerSort>::Container>())
    }

    fn family_description() -> Option<builtin::FamilyDescription> {
        Some(builtin::FamilyDescription {
            name: "Set".into(),
            arity: 1,
        })
    }

    fn builtin_definitions() -> Vec<builtin::BuiltinDescription> {
        use builtin::{
            Argument, BuiltinDescription, BuiltinTyping, Signature, SortPattern, TypeParameter,
        };
        let patterns = [
            SortPattern::Param(0),
            SortPattern::Family {
                name: "Set".into(),
                args: vec![SortPattern::Param(0)],
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
        [
            ("empty", "set-empty", vec![], 1),
            ("of", "set-of", vec![], 1),
            ("get", "set-get", vec![("set", 1), ("index", 2)], 0),
            ("insert", "set-insert", vec![("set", 1), ("value", 0)], 1),
            ("remove", "set-remove", vec![("set", 1), ("value", 0)], 1),
            ("length", "set-length", vec![("set", 1)], 2),
            (
                "contains",
                "set-contains",
                vec![("set", 1), ("value", 0)],
                3,
            ),
            (
                "not_contains",
                "set-not-contains",
                vec![("set", 1), ("value", 0)],
                3,
            ),
            ("union", "set-union", vec![("set", 1), ("other", 1)], 1),
            ("diff", "set-diff", vec![("set", 1), ("other", 1)], 1),
            (
                "intersect",
                "set-intersect",
                vec![("set", 1), ("other", 1)],
                1,
            ),
        ]
        .into_iter()
        .map(|(key, alias, inputs, output)| BuiltinDescription {
            key: format!("egglog.core.set.{key}"),
            aliases: vec![alias.into()],
            typing: BuiltinTyping::Signature(Signature {
                params: vec![TypeParameter {
                    name: "T".into(),
                    equality: false,
                }],
                fixed: inputs
                    .into_iter()
                    .map(|(name, sort)| Argument {
                        name: name.into(),
                        sort: patterns[sort].clone(),
                    })
                    .collect(),
                tail: if key == "of" {
                    vec![Argument {
                        name: "values".into(),
                        sort: patterns[0].clone(),
                    }]
                } else {
                    vec![]
                },
                output: patterns[output].clone(),
            }),
        })
        .collect()
    }

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
        eg.type_info
            .register_builtin_sort("Set", arc.clone(), vec![self.element.clone()])
            .expect("invalid Set family instance");
        let [
            empty,
            of,
            get,
            insert,
            remove,
            length,
            contains,
            not_contains,
            union,
            diff,
            intersect,
        ] = [
            "empty",
            "of",
            "get",
            "insert",
            "remove",
            "length",
            "contains",
            "not_contains",
            "union",
            "diff",
            "intersect",
        ]
        .map(|key| {
            eg.type_info
                .instantiate_builtin(&format!("egglog.core.set.{key}"), &arc)
                .unwrap()
        });

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

        add_primitive_with_validator!(eg, "set-empty" [instance = empty] = {self.clone(): SetSort} || -> @SetContainer { SetContainer {
            do_rebuild: self.ctx.is_eq_container_sort(),
            data: BTreeSet::new()
        } }, set_empty_validator);
        add_primitive_with_validator!(eg, "set-of" [instance = of] = {self.clone(): SetSort} [xs: #] -> @SetContainer { SetContainer {
            do_rebuild: self.ctx.is_eq_container_sort(),
            data: xs.collect()
        } }, set_of_validator);

        // No validator: `set-get` indexes the runtime `BTreeSet<Value>` order,
        // which terms cannot reproduce, so it is unsupported in proof mode.
        add_primitive!(eg, "set-get" [instance = get] = |xs: @SetContainer, i: i64| -?> # { xs.data.iter().nth(i as usize).copied() });
        add_primitive_with_validator!(eg, "set-insert" [instance = insert] = |mut xs: @SetContainer, x: #| -> @SetContainer {{ xs.data.insert( x); xs }}, set_insert_validator);
        add_primitive_with_validator!(eg, "set-remove" [instance = remove] = |mut xs: @SetContainer, x: #| -> @SetContainer {{ xs.data.remove(&x); xs }}, set_remove_validator);

        add_primitive_with_validator!(eg, "set-length" [instance = length] = |xs: @SetContainer| -> i64 { xs.data.len() as i64 }, set_length_validator);
        add_primitive_with_validator!(eg, "set-contains" [instance = contains] = |xs: @SetContainer, x: #| -?> () { ( xs.data.contains(&x)).then_some(()) }, set_contains_validator);
        add_primitive_with_validator!(eg, "set-not-contains" [instance = not_contains] = |xs: @SetContainer, x: #| -?> () { (!xs.data.contains(&x)).then_some(()) }, set_not_contains_validator);

        add_primitive_with_validator!(eg, "set-union" [instance = union] = |mut xs: @SetContainer, ys: @SetContainer| -> @SetContainer {{ xs.data.extend(ys.data);                  xs }}, set_union_validator);
        add_primitive_with_validator!(eg, "set-diff" [instance = diff] = |mut xs: @SetContainer, ys: @SetContainer| -> @SetContainer {{ xs.data.retain(|k| !ys.data.contains(k)); xs }}, set_diff_validator);
        add_primitive_with_validator!(eg, "set-intersect" [instance = intersect] = |mut xs: @SetContainer, ys: @SetContainer| -> @SetContainer {{ xs.data.retain(|k|  ys.data.contains(k)); xs }}, set_intersect_validator);
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
