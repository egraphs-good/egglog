use crate::*;
use ordered_float::OrderedFloat;

use std::fmt::Display;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub enum Literal {
    Int(i64),
    F64(OrderedFloat<f64>),
    String(Symbol),
    Unit,
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
impl_from!(F64(OrderedFloat<f64>));
impl_from!(String(Symbol));

impl Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Literal::Int(i) => Display::fmt(i, f),
            Literal::F64(n) => {
                // need to display with decimal if there is none
                let str = n.to_string();
                if let Ok(_num) = str.parse::<i64>() {
                    write!(f, "{}.0", str)
                } else {
                    write!(f, "{}", str)
                }
            }
            Literal::String(s) => write!(f, "{s}"),
            Literal::Unit => write!(f, "()"),
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

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub enum SSAExpr {
    Call(Symbol, Vec<Symbol>),
    Primative(Symbol, Vec<Symbol>),
}

impl SSAExpr {
    pub fn to_expr(&self) -> Expr {
        match self {
            SSAExpr::Call(op, args) => {
                Expr::Call(*op, args.into_iter().map(|a| Expr::Var(*a)).collect())
            }
            SSAExpr::Primative(op, args) => {
                Expr::Call(*op, args.into_iter().map(|a| Expr::Var(*a)).collect())
            }
        }
    }
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

    pub(crate) fn to_sexp(&self) -> Sexp {
        let res = match self {
            Expr::Lit(lit) => Sexp::String(lit.to_string()),
            Expr::Var(v) => Sexp::String(v.to_string()),
            Expr::Call(op, children) => Sexp::List(
                vec![Sexp::String(op.to_string())]
                    .into_iter()
                    .chain(children.iter().map(|c| c.to_sexp()))
                    .collect(),
            ),
        };
        res
    }

    pub fn replace_canon(&self, canon: &HashMap<Symbol, Expr>) -> Self {
        match self {
            Expr::Lit(_lit) => self.clone(),
            Expr::Var(v) => canon.get(v).cloned().unwrap_or_else(|| self.clone()),
            Expr::Call(op, children) => {
                let children = children.iter().map(|c| c.replace_canon(canon)).collect();
                Expr::Call(*op, children)
            }
        }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp())
    }
}
