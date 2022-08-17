
use std::{num::NonZeroU32};

use crate::{
    ast::{Symbol},
    Id,
};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
// FIXME this shouldn't be pub
pub struct Value {
    pub tag: Symbol,
    pub bits: u64,
}

impl Value {
    pub fn unit() -> Self {
        Value {
            tag: Symbol::new("Unit"),
            bits: 0,
        }
    }

    pub fn fake() -> Self {
        Value {
            tag: Symbol::new("__bogus__"),
            bits: 1234567890,
        }
    }

    pub fn from_id(tag: Symbol, id: Id) -> Self {
        Value {
            tag,
            bits: usize::from(id) as u64,
        }
    }
}

// impl Display for Value {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         match &self.0 {
//             ValueInner::Bool(b) => b.fmt(f),
//             ValueInner::Id(id) => id.fmt(f),
//             ValueInner::I64(i) => i.fmt(f),
//             ValueInner::String(s) => write!(f, "\"{s}\""),
//             ValueInner::Rational(r) => r.fmt(f),
//             ValueInner::Unit => write!(f, "()"),
//         }
//     }
// }

// impl Value {
//     pub fn fake() -> Self {
//         Value(ValueInner::Id(Id::fake()))
//     }

//     pub(crate) fn to_literal(&self) -> Literal {
//         match &self.0 {
//             ValueInner::Bool(_) => todo!(),
//             ValueInner::Id(_) => panic!("Id isn't a literal"),
//             ValueInner::I64(i) => Literal::Int(*i),
//             ValueInner::String(s) => Literal::String(*s),
//             ValueInner::Rational(r) => Literal::Rational(r.clone()),
//             ValueInner::Unit => Literal::Unit,
//         }
//     }

//     pub fn get_type(&self) -> Type {
//         match &self.0 {
//             ValueInner::Bool(_) => todo!(),
//             ValueInner::Id(_) => panic!("Does't know the type of id without context"),
//             ValueInner::I64(_) => Type::NumType(NumType::I64),
//             ValueInner::String(_) => Type::String,
//             ValueInner::Rational(_) => Type::NumType(NumType::Rational),
//             ValueInner::Unit => Type::Unit,
//         }
//     }
// }

// macro_rules! impl_from {
//     ($ctor:ident($t:ty)) => {
//         impl From<Value> for $t {
//             fn from(value: Value) -> Self {
//                 match value.0 {
//                     ValueInner::$ctor(t) => t,
//                     _ => panic!("Expected {}, got {value:?}", stringify!($ctor)),
//                 }
//             }
//         }

//         impl From<$t> for Value {
//             fn from(t: $t) -> Self {
//                 Value(ValueInner::$ctor(t))
//             }
//         }
//     };
// }

// impl From<()> for Value {
//     fn from(_: ()) -> Self {
//         Value(ValueInner::Unit)
//     }
// }

impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Self {
            tag: Symbol::from("i64"),
            bits: i as u64,
        }
    }
}

impl From<Symbol> for Value {
    fn from(s: Symbol) -> Self {
        Self {
            tag: Symbol::from("i64"),
            bits: NonZeroU32::from(s).get().into(),
        }
    }
}

// impl_from!(Id(Id));
// impl_from!(I64(i64));
// impl_from!(Bool(bool));
// impl_from!(String(Symbol));
// impl_from!(Rational(BigRational));
