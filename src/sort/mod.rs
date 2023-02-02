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
mod f64;
pub use self::f64::*;
mod map;
pub use map::*;

use crate::*;

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
    fn store(self, sort: &Self::Sort) -> Option<Value>;
}

impl<T: IntoSort> IntoSort for Option<T> {
    type Sort = T::Sort;

    fn store(self, sort: &Self::Sort) -> Option<Value> {
        self?.store(sort)
    }
}

pub type PreSort = fn(egraph: &mut EGraph, name: Symbol, params: &[Expr]) -> Result<ArcSort, Error>;
