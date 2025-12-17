use crate::sort::{F, S};
use egglog_ast::generic_ast::Literal;
use ordered_float::OrderedFloat;

/// Trait for types that can be converted to/from egglog Literals
pub trait LiteralConvertible: Sized {
    /// Convert from a Literal to this type
    fn from_literal(lit: &Literal) -> Option<Self>;

    /// Convert this type to a Literal
    fn to_literal(self) -> Literal;
}

impl LiteralConvertible for i64 {
    fn from_literal(lit: &Literal) -> Option<Self> {
        match lit {
            Literal::Int(v) => Some(*v),
            _ => None,
        }
    }

    fn to_literal(self) -> Literal {
        Literal::Int(self)
    }
}

impl LiteralConvertible for bool {
    fn from_literal(lit: &Literal) -> Option<Self> {
        match lit {
            Literal::Bool(v) => Some(*v),
            _ => None,
        }
    }

    fn to_literal(self) -> Literal {
        Literal::Bool(self)
    }
}

impl LiteralConvertible for f64 {
    fn from_literal(lit: &Literal) -> Option<Self> {
        match lit {
            Literal::Float(v) => Some(**v),
            _ => None,
        }
    }

    fn to_literal(self) -> Literal {
        Literal::Float(OrderedFloat(self))
    }
}

impl LiteralConvertible for String {
    fn from_literal(lit: &Literal) -> Option<Self> {
        match lit {
            Literal::String(v) => Some(v.clone()),
            _ => None,
        }
    }

    fn to_literal(self) -> Literal {
        Literal::String(self)
    }
}

impl LiteralConvertible for () {
    fn from_literal(lit: &Literal) -> Option<Self> {
        match lit {
            Literal::Unit => Some(()),
            _ => None,
        }
    }

    fn to_literal(self) -> Literal {
        Literal::Unit
    }
}

// Implementation for F, the boxed float type used in egglog
impl LiteralConvertible for F {
    fn from_literal(lit: &Literal) -> Option<Self> {
        match lit {
            Literal::Float(v) => Some(F::new(*v)),
            _ => None,
        }
    }

    fn to_literal(self) -> Literal {
        Literal::Float(*self)
    }
}

// Implementation for S, the boxed string type used in egglog
impl LiteralConvertible for S {
    fn from_literal(lit: &Literal) -> Option<Self> {
        match lit {
            Literal::String(v) => Some(S::new(v.clone())),
            _ => None,
        }
    }

    fn to_literal(self) -> Literal {
        Literal::String((*self).clone())
    }
}
