use crate::*;

use std::fmt::Display;

use num_rational::BigRational;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub enum Literal {
    Int(i64),
    Rational(BigRational),
    String(Symbol),
}

macro_rules! impl_from {
    ($ctor:ident($t:ty)) => {
        impl From<Literal> for $t {
            fn from(literal: Literal) -> Self {
                match literal {
                    Literal::$ctor(t) => t,
                    #[allow(unreachable_patterns)]
                    _ => panic!("Expected {}, got {literal}", stringify!($ctor)),
                }
            }
        }

        impl From<$t> for Literal {
            fn from(t: $t) -> Self {
                Literal::$ctor(t)
            }
        }
    };
}

impl_from!(Int(i64));
impl_from!(String(Symbol));

impl Literal {
    pub fn to_value(&self) -> Value {
        match &self {
            Literal::Int(i) => Value::from(*i),
            Literal::String(s) => Value::from(*s),
            Literal::Rational(r) => Value::from(r.clone()),
        }
    }
}

impl Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Literal::Int(i) => Display::fmt(i, f),
            Literal::String(s) => write!(f, "{s}"),
            Literal::Rational(r) => write!(f, "{}//{}", r.numer(), r.denom()),
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub enum Expr {
    Lit(Literal),
    Var(Symbol),
    // TODO make this its own type
    Call(Symbol, Vec<Self>),
}

impl Expr {
    pub fn call(op: impl Into<Symbol>, children: impl IntoIterator<Item = Self>) -> Self {
        Self::Call(op.into(), children.into_iter().collect())
    }

    pub fn lit(lit: impl Into<Literal>) -> Self {
        Self::Lit(lit.into())
    }

    pub fn get_var(&self) -> Option<Symbol> {
        match self {
            Expr::Var(v) => Some(*v),
            _ => None,
        }
    }

    fn children(&self) -> &[Self] {
        match self {
            Expr::Var(_) | Expr::Lit(_) => &[],
            Expr::Call(_, children) => children,
        }
    }

    pub fn walk(&self, pre: &mut impl FnMut(&Self), post: &mut impl FnMut(&Self)) {
        pre(self);
        self.children()
            .iter()
            .for_each(|child| child.walk(pre, post));
        post(self);
    }

    pub fn fold<Out>(&self, f: &mut impl FnMut(&Self, Vec<Out>) -> Out) -> Out {
        let ts = self.children().iter().map(|child| child.fold(f)).collect();
        f(self, ts)
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Lit(lit) => Display::fmt(lit, f),
            Expr::Var(var) => Display::fmt(var, f),
            Expr::Call(op, args) => {
                write!(f, "({}", op)?;
                for arg in args {
                    write!(f, " {}", arg)?;
                }
                write!(f, ")")
            }
        }
    }
}
