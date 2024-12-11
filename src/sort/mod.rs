#[macro_use]
mod macros;
use lazy_static::lazy_static;
use std::fmt::Debug;
use std::{any::Any, sync::Arc};

mod bigint;
pub use bigint::*;
mod bigrat;
pub use bigrat::*;
mod bool;
pub use self::bool::*;
mod string;
pub use string::*;
mod unit;
pub use unit::*;
mod i64;
pub use self::i64::*;
mod f64;
pub use self::f64::*;
mod map;
pub use map::*;
mod set;
pub use set::*;
mod vec;
pub use vec::*;
mod r#fn;
pub use r#fn::*;
mod multiset;
pub use multiset::*;

use crate::constraint::AllEqualTypeConstraint;
use crate::extract::{Cost, Extractor};
use crate::*;

pub trait Sort: Any + Send + Sync + Debug {
    fn name(&self) -> Symbol;

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static>;

    fn is_eq_sort(&self) -> bool {
        false
    }

    // return true if it is a container sort.
    fn is_container_sort(&self) -> bool {
        false
    }

    // return true if it is a container sort that contains ids.
    // only eq_sort and eq_container_sort need to be canonicalized.
    fn is_eq_container_sort(&self) -> bool {
        false
    }

    // Only eq_container_sort need to implement this method,
    // which returns a list of ids to be tracked.
    fn foreach_tracked_values<'a>(
        &'a self,
        value: &'a Value,
        mut f: Box<dyn FnMut(ArcSort, Value) + 'a>,
    ) {
        for (sort, value) in self.inner_values(value) {
            if sort.is_eq_sort() {
                f(sort, value)
            }
        }
    }

    // Sort-wise canonicalization. Return true if value is modified.
    // Only EqSort or containers of EqSort should override.
    fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
        #[cfg(debug_assertions)]
        debug_assert_eq!(self.name(), value.tag);

        #[cfg(not(debug_assertions))]
        let _ = value;
        let _ = unionfind;
        false
    }

    /// Return the serialized name of the sort
    ///
    /// Only used for container sorts, which cannot be serialized with make_expr so need an explicit name
    fn serialized_name(&self, _value: &Value) -> Symbol {
        self.name()
    }

    /// Return the inner values and sorts.
    /// Only eq_container_sort need to implement this method,
    fn inner_values(&self, value: &Value) -> Vec<(ArcSort, Value)> {
        let _ = value;
        vec![]
    }

    fn register_primitives(self: Arc<Self>, info: &mut TypeInfo) {
        let _ = info;
    }

    /// Extracting an expression (with smallest cost) out of a primitive value
    fn make_expr(&self, egraph: &EGraph, value: Value) -> (Cost, Expr);

    /// For values like EqSort containers, to make/extract an expression from it
    /// requires an extractor. Moreover, the extraction may be unsuccessful if
    /// the extractor is not fully initialized.
    ///
    /// The default behavior is to call make_expr
    fn extract_expr(
        &self,
        egraph: &EGraph,
        value: Value,
        _extractor: &Extractor,
        _termdag: &mut TermDag,
    ) -> Option<(Cost, Expr)> {
        Some(self.make_expr(egraph, value))
    }
}

// Note: this trait is currently intended to be implemented on the
// same struct as `Sort`. If in the future we have dynamic presorts
// (for example, we want to add partial application) we should revisit
// this and make the methods take a `self` parameter.
pub trait Presort {
    fn presort_name() -> Symbol;
    fn reserved_primitives() -> Vec<Symbol>;
    fn make_sort(
        typeinfo: &mut TypeInfo,
        name: Symbol,
        args: &[Expr],
    ) -> Result<ArcSort, TypeError>;
}

#[derive(Debug)]
pub struct EqSort {
    pub name: Symbol,
}

impl Sort for EqSort {
    fn name(&self) -> Symbol {
        self.name
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn is_eq_sort(&self) -> bool {
        true
    }

    fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
        #[cfg(debug_assertions)]
        debug_assert_eq!(self.name(), value.tag);

        let bits = unionfind.find(value.bits);
        if bits != value.bits {
            value.bits = bits;
            true
        } else {
            false
        }
    }

    fn make_expr(&self, _egraph: &EGraph, _value: Value) -> (Cost, Expr) {
        unimplemented!("No make_expr for EqSort {}", self.name)
    }
}

pub trait FromSort: Sized {
    type Sort: Sort;
    fn load(sort: &Self::Sort, value: &Value) -> Self;
}

pub trait IntoSort: Sized {
    type Sort: Sort;
    fn store(self, sort: &Self::Sort) -> Option<Value>;
}

impl<T: IntoSort> IntoSort for Option<T> {
    type Sort = T::Sort;

    fn store(self, sort: &Self::Sort) -> Option<Value> {
        self?.store(sort)
    }
}

pub type PreSort =
    fn(typeinfo: &mut TypeInfo, name: Symbol, params: &[Expr]) -> Result<ArcSort, TypeError>;

pub(crate) struct ValueEq;

impl PrimitiveLike for ValueEq {
    fn name(&self) -> Symbol {
        "value-eq".into()
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        AllEqualTypeConstraint::new(self.name(), span.clone())
            .with_exact_length(3)
            .with_output_sort(Arc::new(UnitSort))
            .into_box()
    }

    fn apply(
        &self,
        values: &[Value],
        _sorts: (&[ArcSort], &ArcSort),
        _egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
        assert_eq!(values.len(), 2);
        if values[0] == values[1] {
            Some(Value::unit())
        } else {
            None
        }
    }
}

pub fn literal_sort(lit: &Literal) -> ArcSort {
    match lit {
        Literal::Int(_) => Arc::new(I64Sort) as ArcSort,
        Literal::Float(_) => Arc::new(F64Sort) as ArcSort,
        Literal::String(_) => Arc::new(StringSort) as ArcSort,
        Literal::Bool(_) => Arc::new(BoolSort) as ArcSort,
        Literal::Unit => Arc::new(UnitSort) as ArcSort,
    }
}
