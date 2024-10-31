use std::sync::Mutex;

use crate::{ast::Literal, constraint::AllEqualTypeConstraint};

use super::*;

#[derive(Debug, Default)]
pub struct StringSort {
    strings: Mutex<IndexSet<String>>,
}

lazy_static! {
    static ref STRING_SORT_NAME: String = "String".into();
}

impl Sort for StringSort {
    fn name(&self) -> String {
        STRING_SORT_NAME.clone()
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn make_expr(&self, _egraph: &EGraph, value: Value) -> (Cost, Expr) {
        #[cfg(debug_assertions)]
        debug_assert_eq!(value.tag, self.name());

        let string = String::load(self, &value);

        (
            1,
            GenericExpr::Lit(DUMMY_SPAN.clone(), Literal::String(string)),
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

impl IntoSort for String {
    type Sort = StringSort;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        let mut strings = sort.strings.lock().unwrap();
        let (i, _) = strings.insert_full(self);
        Some(Value {
            #[cfg(debug_assertions)]
            tag: sort.name(),
            bits: i as u64,
        })
    }
}

impl FromSort for String {
    type Sort = StringSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let strings = sort.strings.lock().unwrap();
        strings.get_index(value.bits as usize).unwrap().clone()
    }
}

struct Add {
    name: String,
    string: Arc<StringSort>,
}

impl PrimitiveLike for Add {
    fn name(&self) -> String {
        self.name.clone()
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
            let sym = String::load(&self.string, value);
            res_string.push_str(sym.as_str());
        }
        res_string.store(&self.string)
    }
}

struct Replace {
    name: String,
    string: Arc<StringSort>,
}

impl PrimitiveLike for Replace {
    fn name(&self) -> String {
        self.name.clone()
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
        let string1 = String::load(&self.string, &values[0]).to_string();
        let string2 = String::load(&self.string, &values[1]).to_string();
        let string3 = String::load(&self.string, &values[2]).to_string();
        let res: String = string1.replace(&string2, &string3);
        res.store(&self.string)
    }
}
