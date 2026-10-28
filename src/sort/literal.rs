//! Owned native base literals, interned only into the receiving engine.

use super::*;
use crate::core_relations::BaseValue;
use egglog_ast::native_literal::{NativeLiteral, NativeLiteralValue};
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

pub(crate) trait LiteralPayload: Any + Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn compare(&self, other: &dyn LiteralPayload) -> Ordering;
    fn value_hash(&self) -> u64;
    fn intern(&self, values: &BaseValues) -> Value;
    fn standalone_values(&self) -> BaseValues;
}

impl<T: BaseValue + Ord> LiteralPayload for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn compare(&self, other: &dyn LiteralPayload) -> Ordering {
        self.cmp(
            other
                .as_any()
                .downcast_ref::<T>()
                .expect("matching native literal types"),
        )
    }
    fn value_hash(&self) -> u64 {
        let mut hash = DefaultHasher::new();
        self.hash(&mut hash);
        hash.finish()
    }
    fn intern(&self, values: &BaseValues) -> Value {
        values.get(self.clone())
    }
    fn standalone_values(&self) -> BaseValues {
        let mut values = BaseValues::default();
        values.register_type::<T>();
        values
    }
}

#[derive(Debug)]
pub(crate) struct BaseLiteral {
    pub(crate) sort: ArcSort,
    pub(crate) value: Arc<dyn LiteralPayload>,
}

impl NativeLiteral for BaseLiteral {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn compare(&self, other: &dyn NativeLiteral) -> Ordering {
        let other = other
            .as_any()
            .downcast_ref::<Self>()
            .expect("matching native literal implementations");
        self.sort
            .name()
            .cmp(other.sort.name())
            .then_with(|| {
                self.value
                    .as_any()
                    .type_id()
                    .cmp(&other.value.as_any().type_id())
            })
            .then_with(|| self.value.compare(other.value.as_ref()))
    }
    fn value_hash(&self) -> u64 {
        let mut hash = DefaultHasher::new();
        self.sort.name().hash(&mut hash);
        self.value.as_any().type_id().hash(&mut hash);
        self.value.value_hash().hash(&mut hash);
        hash.finish()
    }
}

impl Display for BaseLiteral {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Reconstruction belongs only to source presentation. Runtime and query
        // compilation intern the owned payload directly.
        let values = self.value.standalone_values();
        let value = self.value.intern(&values);
        self.sort.fmt_base_value(&values, value, f)
    }
}

/// Construct an inert typed native literal without interning in any e-graph.
/// The sort must represent exactly `T` and must be a base sort.
pub fn base_literal<T: BaseValue + Ord>(sort: ArcSort, value: T) -> Result<Literal, String> {
    if sort.is_container_sort() || sort.value_type() != Some(TypeId::of::<T>()) {
        return Err("native literal payload does not match its base sort".into());
    }
    Ok(Literal::Native(NativeLiteralValue(Arc::new(BaseLiteral {
        sort,
        value: Arc::new(value),
    }))))
}

/// Recover an owned native payload while checking its engine and sort type.
pub fn downcast_base_literal<T: BaseValue>(literal: &Literal) -> Result<T, String> {
    let Literal::Native(value) = literal else {
        return Err("expected a native base literal".into());
    };
    let literal = value
        .0
        .as_ref()
        .as_any()
        .downcast_ref::<BaseLiteral>()
        .ok_or("foreign native literal implementation")?;
    if literal.sort.value_type() != Some(TypeId::of::<T>()) {
        return Err("native literal has another value type".into());
    }
    literal
        .value
        .as_any()
        .downcast_ref::<T>()
        .cloned()
        .ok_or_else(|| "native literal payload disagrees with its sort".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Action, Command, Expr, Fact};
    use crate::prelude::BaseSort;

    #[test]
    fn native_base_literals_own_compare_hash_display_and_query() {
        let expected = BigInt::from(10).pow(1000) + BigInt::from(7);
        let literal = base_literal(BigIntSort.to_arcsort(), Z::new(expected.clone())).unwrap();
        let same = base_literal(BigIntSort.to_arcsort(), Z::new(expected.clone())).unwrap();
        let different = base_literal(BigIntSort.to_arcsort(), Z::new(BigInt::from(3))).unwrap();
        assert_eq!(literal, same);
        assert_ne!(literal, different);
        assert_ne!(literal.cmp(&different), Ordering::Equal);
        let hash = |value: &Literal| {
            let mut hash = DefaultHasher::new();
            value.hash(&mut hash);
            hash.finish()
        };
        assert_eq!(hash(&literal), hash(&same));
        assert!(base_literal(I64Sort.to_arcsort(), Z::new(expected.clone())).is_err());
        assert!(downcast_base_literal::<i64>(&literal).is_err());
        assert_eq!(downcast_base_literal::<Z>(&literal).unwrap().0, expected);
        assert_eq!(literal.to_string(), format!("(from-string \"{expected}\")"));
        let mut graph = EGraph::default();
        let value = Expr::Lit(span!(), literal.clone());
        graph
            .run_program(vec![Command::Action(Action::Let(
                span!(),
                "native_value".into(),
                value.clone(),
            ))])
            .unwrap();
        let (sort, native) = graph
            .eval_expr(&Expr::Var(span!(), "native_value".into()))
            .unwrap();
        assert_eq!(sort.name(), "BigInt");
        assert_eq!(graph.value_to_base::<Z>(native).0, expected);
        graph
            .run_program(vec![Command::Check(
                span!(),
                vec![Fact::Eq(span!(), value.clone(), Expr::Lit(span!(), same))],
            )])
            .unwrap();
        assert!(
            graph
                .run_program(vec![Command::Check(
                    span!(),
                    vec![Fact::Eq(span!(), value, Expr::Lit(span!(), different))]
                )])
                .is_err()
        );
    }
    #[test]
    fn native_literal_execution_and_proofs_never_use_source_presentation() {
        #[derive(Debug)]
        struct DataOnly;
        type Data = Boxed<(i64, i64)>;
        impl BaseSort for DataOnly {
            type Base = Data;
            fn name(&self) -> &str {
                "DataOnly"
            }
            fn reconstruct_termdag(
                &self,
                values: &BaseValues,
                value: Value,
                terms: &mut TermDag,
            ) -> TermId {
                terms
                    .lit(base_literal(DataOnly.to_arcsort(), values.unwrap::<Data>(value)).unwrap())
            }
            fn fmt_value(
                &self,
                _: &BaseValues,
                _: Value,
                _: &mut Formatter<'_>,
            ) -> std::fmt::Result {
                panic!("native literal was formatted for execution")
            }
        }
        for mode in 0..4 {
            let mut graph = EGraph::default();
            crate::add_base_sort(&mut graph, DataOnly, span!()).unwrap();
            let mut graph = match mode {
                0 => graph,
                1 => graph.with_term_encoding_enabled(),
                2 => graph.with_proofs_enabled(),
                _ => graph.with_proofs_enabled().with_proof_testing(),
            };
            graph
                .parse_and_run_program(
                    None,
                    "(datatype HeldData (HoldData DataOnly)) (sort DataVec (Vec DataOnly))",
                )
                .unwrap();
            let literal =
                base_literal(DataOnly.to_arcsort(), Data::new((i64::MIN, i64::MAX))).unwrap();
            let expr = Expr::Call(
                span!(),
                "HoldData".into(),
                vec![Expr::Lit(span!(), literal.clone())],
            );
            graph
                .run_program(vec![Command::Action(Action::Expr(span!(), expr.clone()))])
                .unwrap();
            graph
                .run_program(vec![Command::Check(
                    span!(),
                    vec![Fact::Eq(span!(), expr.clone(), expr)],
                )])
                .unwrap();
            let vector = Expr::Call(
                span!(),
                "vec-of".into(),
                vec![Expr::Lit(span!(), literal.clone())],
            );
            graph
                .run_program(vec![Command::Check(
                    span!(),
                    vec![Fact::Eq(span!(), vector.clone(), vector)],
                )])
                .unwrap();
            if mode >= 2 {
                let data = Expr::Lit(span!(), literal);
                graph
                    .run_program(vec![Command::Prove(
                        span!(),
                        vec![Fact::Eq(span!(), data.clone(), data)],
                    )])
                    .unwrap();
            }
        }
    }
}
