use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PairContainer {
    do_rebuild_first: bool,
    do_rebuild_second: bool,
    pub first: Value,
    pub second: Value,
}

impl ContainerValue for PairContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn Rebuilder) -> bool {
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

    // Proof support is presort-wide: every Pair instance has the same
    // constructor/projection metadata. Instance-specific canonicalization is
    // still gated by `is_eq_container_sort` and the per-field rebuild flags, so
    // `(Pair i64 i64)` is admissible but contributes no rebuild work.
    fn supports_proof_encoding() -> bool {
        true
    }

    fn make_sort(
        typeinfo: &mut TypeInfo,
        name: String,
        args: &[Expr],
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
            panic!("Pair sort requires exactly two arguments")
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

    fn container_proof_spec(&self) -> Option<ContainerProofSpec> {
        Some(ContainerProofSpec {
            constructors: vec![ContainerProofConstructorSpec {
                name: "pair",
                input_sorts: vec![self.first.clone(), self.second.clone()],
                projections: vec![
                    ContainerProofProjectionSpec {
                        primitive: "pair-first",
                        field: 0,
                    },
                    ContainerProofProjectionSpec {
                        primitive: "pair-second",
                        field: 1,
                    },
                ],
            }],
        })
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

        add_primitive_with_validator!(
            eg,
            "pair" = {self.clone(): PairSort} |x: # (self.first()), y: # (self.second())| -> @PairContainer (arc) {
                PairContainer {
                    do_rebuild_first: self.ctx.first.is_eq_sort() || self.ctx.first.is_eq_container_sort(),
                    do_rebuild_second: self.ctx.second.is_eq_sort() || self.ctx.second.is_eq_container_sort(),
                    first: x,
                    second: y,
                }
            },
            |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
                if args.len() == 2 {
                    Some(termdag.app("pair".into(), args.to_vec()))
                } else {
                    None
                }
            }
        );

        add_primitive_with_validator!(
            eg,
            "pair-first"  = |xs: @PairContainer (arc)| -> # (self.first())  { xs.first  },
            |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
                let [pair] = args else {
                    return None;
                };
                match termdag.get(*pair) {
                    Term::App(head, children)
                        if head == "pair" && children.len() == 2 =>
                    {
                        Some(children[0])
                    }
                    _ => None,
                }
            }
        );
        add_primitive_with_validator!(
            eg,
            "pair-second" = |xs: @PairContainer (arc)| -> # (self.second()) { xs.second },
            |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
                let [pair] = args else {
                    return None;
                };
                match termdag.get(*pair) {
                    Term::App(head, children)
                        if head == "pair" && children.len() == 2 =>
                    {
                        Some(children[1])
                    }
                    _ => None,
                }
            }
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

    fn serialized_name(&self, _container_values: &ContainerValues, _: Value) -> String {
        self.name().to_owned()
    }
}
