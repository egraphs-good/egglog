use crate::*;

use std::fmt::Display;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub enum Literal {
    Int(i64),
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

impl Literal {
    pub fn to_value(&self) -> Value {
        match self {
            Literal::Int(i) => Value::from(*i),
        }
    }
}

impl Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Literal::Int(i) => Display::fmt(i, f),
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub enum Expr<T = Literal> {
    Leaf(T),
    Var(Symbol),
    // TODO make this its own type
    Node(Symbol, Vec<Self>),
}

impl<T> Expr<T> {
    pub fn new(op: impl Into<Symbol>, children: impl IntoIterator<Item = Self>) -> Self {
        Self::Node(op.into(), children.into_iter().collect())
    }

    pub fn leaf(op: impl Into<T>) -> Self {
        Self::Leaf(op.into())
    }

    pub fn get_var(&self) -> Option<Symbol> {
        match self {
            Expr::Var(v) => Some(*v),
            _ => None,
        }
    }

    fn children(&self) -> &[Self] {
        match self {
            Expr::Var(_) | Expr::Leaf(_) => &[],
            Expr::Node(_, children) => children,
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

impl<T: Display> Display for Expr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Leaf(val) => val.fmt(f),
            Expr::Var(var) => Display::fmt(var, f),
            Expr::Node(op, args) => {
                write!(f, "({}", op)?;
                for arg in args {
                    write!(f, " {}", arg)?;
                }
                write!(f, ")")
            }
        }
    }
}
