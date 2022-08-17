#[macro_use]
mod macros;
use std::fmt::Debug;
use std::{any::Any, sync::Arc};

mod rational;
pub use rational::*;
mod string;
pub use string::*;
mod unit;
pub use unit::*;
mod i64;
pub use self::i64::*;

use crate::ast::Expr;
use crate::{ast::Symbol, EGraph, Value};

pub trait Sort: Any + Send + Sync + Debug {
    fn name(&self) -> Symbol;
    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static>;
    fn is_eq_sort(&self) -> bool {
        false
    }

    fn register_primitives(self: Arc<Self>, egraph: &mut EGraph) {
        let _ = egraph;
    }

    fn make_expr(&self, value: Value) -> Expr;
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

    fn make_expr(&self, _value: Value) -> Expr {
        unimplemented!("No make_expr for EqSort {}", self.name)
    }
}

pub trait FromSort: Sized {
    type Sort: Sort;
    fn load(sort: &Self::Sort, value: &Value) -> Self;
}

pub trait IntoSort: Sized {
    type Sort: Sort;
    // fn load(sort: &Self::Sort, value: &Value) -> Self;
    fn store(self, sort: &Self::Sort) -> Option<Value>;
    // fn into_option(self) -> Option<Self> {
    //     Some(self)
    // }
    // fn name() -> &'static str {
    //     Self::Sort::name()
    // }
    // fn get_type() -> Symbol {
    //     Self::Sort::get_type()
    // }
    // fn is_type(t: &Symbol) -> bool {
    //     Self::Sort::is_type(t)
    // }
}

impl<T: IntoSort> IntoSort for Option<T> {
    type Sort = T::Sort;

    fn store(self, sort: &Self::Sort) -> Option<Value> {
        self?.store(sort)
    }
}
