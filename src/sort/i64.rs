use crate::{ast::Literal, constraint::AllEqualTypeConstraint};

use super::*;

#[derive(Debug)]
pub struct I64Sort;

lazy_static! {
    static ref I64_SORT_NAME: String = "i64".into();
}

impl Sort for I64Sort {
    fn name(&self) -> String {
        I64_SORT_NAME.clone()
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    #[rustfmt::skip]
    // We need the closure for division and mod operations, as they can panic.
    // cf https://github.com/rust-lang/rust-clippy/issues/9422
    #[allow(clippy::unnecessary_lazy_evaluations)]
    fn register_primitives(self: Arc<Self>, typeinfo: &mut TypeInfo) {
        typeinfo.add_primitive(TermOrderingMin {
           });
        typeinfo.add_primitive(TermOrderingMax {
           });

        type Opt<T=()> = Option<T>;

        add_primitives!(typeinfo, "+" = |a: i64, b: i64| -> i64 { a + b });
        add_primitives!(typeinfo, "-" = |a: i64, b: i64| -> i64 { a - b });
        add_primitives!(typeinfo, "*" = |a: i64, b: i64| -> i64 { a * b });
        add_primitives!(typeinfo, "/" = |a: i64, b: i64| -> Opt<i64> { (b != 0).then(|| a / b) });
        add_primitives!(typeinfo, "%" = |a: i64, b: i64| -> Opt<i64> { (b != 0).then(|| a % b) });

        add_primitives!(typeinfo, "&" = |a: i64, b: i64| -> i64 { a & b });
        add_primitives!(typeinfo, "|" = |a: i64, b: i64| -> i64 { a | b });
        add_primitives!(typeinfo, "^" = |a: i64, b: i64| -> i64 { a ^ b });
        add_primitives!(typeinfo, "<<" = |a: i64, b: i64| -> Opt<i64> { b.try_into().ok().and_then(|b| a.checked_shl(b)) });
        add_primitives!(typeinfo, ">>" = |a: i64, b: i64| -> Opt<i64> { b.try_into().ok().and_then(|b| a.checked_shr(b)) });
        add_primitives!(typeinfo, "not-i64" = |a: i64| -> i64 { !a });

        add_primitives!(typeinfo, "log2" = |a: i64| -> i64 { (a as i64).ilog2() as i64 });

        add_primitives!(typeinfo, "<" = |a: i64, b: i64| -> Opt { (a < b).then_some(()) });
        add_primitives!(typeinfo, ">" = |a: i64, b: i64| -> Opt { (a > b).then_some(()) });
        add_primitives!(typeinfo, "<=" = |a: i64, b: i64| -> Opt { (a <= b).then_some(()) });
        add_primitives!(typeinfo, ">=" = |a: i64, b: i64| -> Opt { (a >= b).then_some(()) });

        add_primitives!(typeinfo, "bool-=" = |a: i64, b: i64| -> bool { a == b });
        add_primitives!(typeinfo, "bool-<" = |a: i64, b: i64| -> bool { a < b });
        add_primitives!(typeinfo, "bool->" = |a: i64, b: i64| -> bool { a > b });
        add_primitives!(typeinfo, "bool-<=" = |a: i64, b: i64| -> bool { a <= b });
        add_primitives!(typeinfo, "bool->=" = |a: i64, b: i64| -> bool { a >= b });

        add_primitives!(typeinfo, "min" = |a: i64, b: i64| -> i64 { a.min(b) });
        add_primitives!(typeinfo, "max" = |a: i64, b: i64| -> i64 { a.max(b) });

        add_primitives!(typeinfo, "to-string" = |a: i64| -> String { a.to_string() });

        // Must be in the i64 sort register function because the string sort is registered before the i64 sort.
        typeinfo.add_primitive(CountMatches {
            name: "count-matches".into(),
            string: typeinfo.get_sort_nofail(),
            int: self.clone(),
        });

    }

    fn make_expr(&self, _egraph: &EGraph, value: Value) -> (Cost, Expr) {
        #[cfg(debug_assertions)]
        debug_assert_eq!(value.tag, self.name());

        (
            1,
            GenericExpr::Lit(DUMMY_SPAN.clone(), Literal::Int(value.bits as _)),
        )
    }
}

impl IntoSort for i64 {
    type Sort = I64Sort;
    fn store(self, _sort: &Self::Sort) -> Option<Value> {
        Some(Value {
            #[cfg(debug_assertions)]
            tag: I64Sort.name(),
            bits: self as u64,
        })
    }
}

impl FromSort for i64 {
    type Sort = I64Sort;
    fn load(_sort: &Self::Sort, value: &Value) -> Self {
        value.bits as Self
    }
}

struct CountMatches {
    name: String,
    string: Arc<StringSort>,
    int: Arc<I64Sort>,
}

impl PrimitiveLike for CountMatches {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        AllEqualTypeConstraint::new(self.name(), span.clone())
            .with_all_arguments_sort(self.string.clone())
            .with_exact_length(3)
            .with_output_sort(self.int.clone())
            .into_box()
    }

    fn apply(
        &self,
        values: &[Value],
        _sorts: (&[ArcSort], &ArcSort),
        _egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
        let string1 = String::load(&self.string, &values[0]).to_string();
        let string2 = String::load(&self.string, &values[1]).to_string();
        Some(Value::from(string1.matches(&string2).count() as i64))
    }
}
