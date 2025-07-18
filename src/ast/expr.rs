use crate::{core::ResolvedCall, *};
use ordered_float::OrderedFloat;
use std::{fmt::Display, hash::Hasher};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub enum Literal {
    Int(i64),
    Float(OrderedFloat<f64>),
    String(String),
    Bool(bool),
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
impl_from!(Float(OrderedFloat<f64>));
impl_from!(String(String));

impl Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Literal::Int(i) => Display::fmt(i, f),
            Literal::Float(n) => {
                // need to display with decimal if there is none
                let str = n.to_string();
                if let Ok(_num) = str.parse::<i64>() {
                    write!(f, "{}.0", str)
                } else {
                    write!(f, "{}", str)
                }
            }
            Literal::Bool(b) => Display::fmt(b, f),
            Literal::String(s) => write!(f, "\"{}\"", s),
            Literal::Unit => write!(f, "()"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedVar {
    pub name: String,
    pub sort: ArcSort,
    /// Is this a reference to a global variable?
    /// After the `remove_globals` pass, this should be `false`.
    ///
    /// NB: we distinguish between a global reference and a global binding.
    /// The current implementation of `Eq` and `Hash` does not take this field
    /// into consideration.
    /// Overall, the definition of equality between two ResolvedVars is dicey.
    pub is_global_ref: bool,
}

impl PartialEq for ResolvedVar {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.sort.name() == other.sort.name()
    }
}

impl Eq for ResolvedVar {}

impl Hash for ResolvedVar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.sort.name().hash(state);
    }
}

impl Display for ResolvedVar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

pub type Expr = GenericExpr<String, String>;
/// A generated expression is an expression that is generated by the system
/// and does not have annotations.
pub(crate) type ResolvedExpr = GenericExpr<ResolvedCall, ResolvedVar>;
/// A [`MappedExpr`] arises naturally when you want a mapping between an expression
/// and its flattened form. It records this mapping by annotating each `Head`
/// with a `Leaf`, which it maps to in the flattened form.
/// A useful operation on `MappedExpr`s is [`MappedExpr::get_corresponding_var_or_lit``].
pub(crate) type MappedExpr<Head, Leaf> = GenericExpr<CorrespondingVar<Head, Leaf>, Leaf>;

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum GenericExpr<Head, Leaf> {
    Lit(Span, Literal),
    Var(Span, Leaf),
    Call(Span, Head, Vec<Self>),
}

impl ResolvedExpr {
    pub fn output_type(&self) -> ArcSort {
        match self {
            ResolvedExpr::Lit(_, lit) => sort::literal_sort(lit),
            ResolvedExpr::Var(_, resolved_var) => resolved_var.sort.clone(),
            ResolvedExpr::Call(_, resolved_call, _) => resolved_call.output().clone(),
        }
    }

    pub(crate) fn get_global_var(&self) -> Option<ResolvedVar> {
        match self {
            ResolvedExpr::Var(_, v) if v.is_global_ref => Some(v.clone()),
            _ => None,
        }
    }
}

#[macro_export]
macro_rules! call {
    ($func:expr, $args:expr) => {
        $crate::ast::GenericExpr::Call($crate::span!(), $func.into(), $args.into_iter().collect())
    };
}

#[macro_export]
macro_rules! lit {
    ($lit:expr) => {
        $crate::ast::GenericExpr::Lit($crate::span!(), $lit.into())
    };
}

#[macro_export]
macro_rules! var {
    ($var:expr) => {
        $crate::ast::GenericExpr::Var($crate::span!(), $var.into())
    };
}

// Rust macro annoyance; see stackoverflow.com/questions/26731243/how-do-i-use-a-macro-across-module-files
pub use {call, lit, var};

impl<Head: Clone + Display, Leaf: Hash + Clone + Display + Eq> GenericExpr<Head, Leaf> {
    pub fn span(&self) -> Span {
        match self {
            GenericExpr::Lit(span, _) => span.clone(),
            GenericExpr::Var(span, _) => span.clone(),
            GenericExpr::Call(span, _, _) => span.clone(),
        }
    }

    pub fn is_var(&self) -> bool {
        matches!(self, GenericExpr::Var(_, _))
    }

    pub fn get_var(&self) -> Option<Leaf> {
        match self {
            GenericExpr::Var(_ann, v) => Some(v.clone()),
            _ => None,
        }
    }

    fn children(&self) -> &[Self] {
        match self {
            GenericExpr::Var(_, _) | GenericExpr::Lit(_, _) => &[],
            GenericExpr::Call(_, _, children) => children,
        }
    }

    pub fn ast_size(&self) -> usize {
        let mut size = 0;
        self.walk(&mut |_e| size += 1, &mut |_| {});
        size
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

    /// Applys `f` to all sub-expressions (including `self`)
    /// bottom-up, collecting the results.
    pub fn visit_exprs(self, f: &mut impl FnMut(Self) -> Self) -> Self {
        match self {
            GenericExpr::Lit(..) => f(self),
            GenericExpr::Var(..) => f(self),
            GenericExpr::Call(span, op, children) => {
                let children = children.into_iter().map(|c| c.visit_exprs(f)).collect();
                f(GenericExpr::Call(span, op.clone(), children))
            }
        }
    }

    /// `subst` replaces occurrences of variables and head symbols in the expression.
    pub fn subst<Head2, Leaf2>(
        &self,
        subst_leaf: &mut impl FnMut(&Span, &Leaf) -> GenericExpr<Head2, Leaf2>,
        subst_head: &mut impl FnMut(&Head) -> Head2,
    ) -> GenericExpr<Head2, Leaf2> {
        match self {
            GenericExpr::Lit(span, lit) => GenericExpr::Lit(span.clone(), lit.clone()),
            GenericExpr::Var(span, v) => subst_leaf(span, v),
            GenericExpr::Call(span, op, children) => {
                let children = children
                    .iter()
                    .map(|c| c.subst(subst_leaf, subst_head))
                    .collect();
                GenericExpr::Call(span.clone(), subst_head(op), children)
            }
        }
    }

    pub fn subst_leaf<Leaf2>(
        &self,
        subst_leaf: &mut impl FnMut(&Span, &Leaf) -> GenericExpr<Head, Leaf2>,
    ) -> GenericExpr<Head, Leaf2> {
        self.subst(subst_leaf, &mut |x| x.clone())
    }

    pub fn vars(&self) -> impl Iterator<Item = Leaf> + '_ {
        let iterator: Box<dyn Iterator<Item = Leaf>> = match self {
            GenericExpr::Lit(_ann, _l) => Box::new(std::iter::empty()),
            GenericExpr::Var(_ann, v) => Box::new(std::iter::once(v.clone())),
            GenericExpr::Call(_ann, _head, exprs) => Box::new(exprs.iter().flat_map(|e| e.vars())),
        };
        iterator
    }
}

impl<Head, Leaf> Display for GenericExpr<Head, Leaf>
where
    Head: Display,
    Leaf: Display,
{
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            GenericExpr::Lit(_ann, lit) => write!(f, "{lit}"),
            GenericExpr::Var(_ann, var) => write!(f, "{var}"),
            GenericExpr::Call(_ann, op, children) => {
                write!(f, "({} {})", op, ListDisplay(children, " "))
            }
        }
    }
}
