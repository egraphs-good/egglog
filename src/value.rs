use std::fmt::Display;
use num_rational::BigRational;

use crate::{
    ast::{Literal, Symbol},
    Id,
    InputType, NumType,
};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
// FIXME this shouldn't be pub
pub struct Value(pub ValueInner);

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub enum ValueInner {
    Bool(bool),
    Id(Id),
    I64(i64),
    Rational(BigRational),
    String(Symbol),
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            ValueInner::Bool(b) => b.fmt(f),
            ValueInner::Id(id) => id.fmt(f),
            ValueInner::I64(i) => i.fmt(f),
            ValueInner::String(s) => write!(f, "\"{s}\""),
            ValueInner::Rational(r) => r.fmt(f),
        }
    }
}

impl Value {
    pub fn fake() -> Self {
        Value(ValueInner::Id(Id::fake()))
    }

    pub(crate) fn to_literal(&self) -> Literal {
        match &self.0 {
            ValueInner::Bool(_) => todo!(),
            ValueInner::Id(_) => panic!("Id isn't a literal"),
            ValueInner::I64(i) => Literal::Int(*i),
            ValueInner::String(s) => Literal::String(*s),
            ValueInner::Rational(r) => Literal::Rational(r.clone()),
        }
    }

    pub fn get_type(&self) -> InputType {
        match &self.0 {
            ValueInner::Bool(_) => todo!(),
            ValueInner::Id(_) => panic!("Does't know the type of id without context"),
            ValueInner::I64(_) => InputType::NumType(NumType::I64),
            ValueInner::String(_) => InputType::String,
            ValueInner::Rational(_) => InputType::NumType(NumType::Rational),
        }
    }
}

macro_rules! impl_from {
    ($ctor:ident($t:ty)) => {
        impl From<Value> for $t {
            fn from(value: Value) -> Self {
                match value.0 {
                    ValueInner::$ctor(t) => t,
                    _ => panic!("Expected {}, got {value:?}", stringify!($ctor)),
                }
            }
        }

        impl From<$t> for Value {
            fn from(t: $t) -> Self {
                Value(ValueInner::$ctor(t))
            }
        }
    };
}

impl_from!(Id(Id));
impl_from!(I64(i64));
impl_from!(Bool(bool));
impl_from!(String(Symbol));
impl_from!(Rational(BigRational));
