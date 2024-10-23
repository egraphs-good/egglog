use std::num::NonZeroU32;

use crate::{ast::Literal, constraint::AllEqualTypeConstraint};

use super::*;

#[derive(Debug)]
pub struct StringSort;

lazy_static! {
    static ref STRING_SORT_NAME: Symbol = "String".into();
}

impl Sort for StringSort {
    fn name(&self) -> Symbol {
        *STRING_SORT_NAME
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn make_expr(&self, _egraph: &EGraph, value: Value) -> (Cost, Expr) {
        assert!(value.tag == self.name());
        let sym = Symbol::from(NonZeroU32::new(value.bits as _).unwrap());
        (
            1,
            GenericExpr::Lit(DUMMY_SPAN.clone(), Literal::String(sym)),
        )
    }

    fn register_primitives(self: Arc<Self>, typeinfo: &mut TypeInfo) {
        typeinfo.add_primitive(Add {
            name: "+".into(),
            string: self.clone(),
        });
        typeinfo.add_primitive(Replace {
            name: "replace".into(),
            string: self,
        });
    }
}

// TODO could use a local symbol table

impl IntoSort for Symbol {
    type Sort = StringSort;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        Some(Value {
            tag: sort.name(),
            bits: NonZeroU32::from(self).get() as _,
        })
    }
}

impl FromSort for Symbol {
    type Sort = StringSort;
    fn load(_sort: &Self::Sort, value: &Value) -> Self {
        NonZeroU32::new(value.bits as u32).unwrap().into()
    }
}

struct Add {
    name: Symbol,
    string: Arc<StringSort>,
}

impl PrimitiveLike for Add {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        AllEqualTypeConstraint::new(self.name(), span.clone())
            .with_all_arguments_sort(self.string.clone())
            .into_box()
    }

    fn apply(
        &self,
        values: &[Value],
        _sorts: (&[ArcSort], &ArcSort),
        _egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
        let mut res_string: String = "".to_owned();
        for value in values {
            let sym = Symbol::load(&self.string, value);
            res_string.push_str(sym.as_str());
        }
        let res_symbol: Symbol = res_string.into();
        Some(Value::from(res_symbol))
    }
}

struct Replace {
    name: Symbol,
    string: Arc<StringSort>,
}

impl PrimitiveLike for Replace {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        AllEqualTypeConstraint::new(self.name(), span.clone())
            .with_all_arguments_sort(self.string.clone())
            .with_exact_length(4)
            .into_box()
    }

    fn apply(
        &self,
        values: &[Value],
        _sorts: (&[ArcSort], &ArcSort),
        _egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
        let string1 = Symbol::load(&self.string, &values[0]).to_string();
        let string2 = Symbol::load(&self.string, &values[1]).to_string();
        let string3 = Symbol::load(&self.string, &values[2]).to_string();
        let res: Symbol = string1.replace(&string2, &string3).into();
        Some(Value::from(res))
    }
}
