use crate::Id;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub struct Value(ValueInner);

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub enum ValueInner {
    Id(Id),
    Int(i64),
}

impl Value {
    pub fn fake() -> Self {
        Value(ValueInner::Id(Id::fake()))
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
impl_from!(Int(i64));
