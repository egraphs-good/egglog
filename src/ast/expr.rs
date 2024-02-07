use crate::{core::ResolvedCall, *};
use ordered_float::OrderedFloat;

use std::{fmt::Display, hash::Hasher};

use super::ToSexp;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub enum Literal {
    Int(i64),
    F64(OrderedFloat<f64>),
    String(Symbol),
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
            Literal::Bool(b) => Display::fmt(b, f),
            Literal::String(s) => write!(f, "\"{}\"", s),
            Literal::Unit => write!(f, "()"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedVar {
    pub name: Symbol,
    pub sort: ArcSort,
    /// Is this a reference to a global variable?
    /// After the `remove_globals` pass, this should be `false`.
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

impl SymbolLike for ResolvedVar {
    fn to_symbol(&self) -> Symbol {
        self.name
    }
}

impl Display for ResolvedVar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl ToSexp for ResolvedVar {
    fn to_sexp(&self) -> Sexp {
        Sexp::Symbol(self.name.to_string())
    }
}

pub type Expr = GenericExpr<Symbol, Symbol, ()>;
pub(crate) type ResolvedExpr = GenericExpr<ResolvedCall, ResolvedVar, ()>;
/// A [`MappedExpr`] arises naturally when you want a mapping between an expression
/// and its flattened form. It records this mapping by annotating each `Head`
/// with a `Leaf`, which it maps to in the flattened form.
/// A useful operation on `MappedExpr`s is [`MappedExpr::get_corresponding_var_or_lit``].
pub(crate) type MappedExpr<Head, Leaf, Ann> = GenericExpr<CorrespondingVar<Head, Leaf>, Leaf, Ann>;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub enum GenericExpr<Head, Leaf, Ann> {
    Lit(Ann, Literal),
    Var(Ann, Leaf),
    Call(Ann, Head, Vec<Self>),
}

impl ResolvedExpr {
    pub fn output_type(&self, type_info: &TypeInfo) -> ArcSort {
        match self {
            ResolvedExpr::Lit(_, lit) => type_info.infer_literal(lit),
            ResolvedExpr::Var(_, resolved_var) => resolved_var.sort.clone(),
            ResolvedExpr::Call(_, resolved_call, _) => resolved_call.output().clone(),
        }
    }
}

impl Expr {
    pub fn call(op: impl Into<Symbol>, children: impl IntoIterator<Item = Self>) -> Self {
        Self::Call((), op.into(), children.into_iter().collect())
    }

    pub fn lit(lit: impl Into<Literal>) -> Self {
        Self::Lit((), lit.into())
    }
}

impl<Head: Clone + Display, Leaf: Hash + Clone + Display + Eq, Ann: Clone>
    GenericExpr<Head, Leaf, Ann>
{
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

    pub fn map(self, f: &mut impl FnMut(Self) -> Self) -> Self {
        match self {
            GenericExpr::Lit(..) => f(self),
            GenericExpr::Var(..) => f(self),
            GenericExpr::Call(ann, op, children) => {
                let children = children.into_iter().map(|c| c.map(f)).collect();
                f(GenericExpr::Call(ann.clone(), op.clone(), children))
            }
        }
    }

    // TODO: Currently, subst_leaf takes a leaf but not an annotation over the leaf,
    // so it has to "make up" annotations for the returned GenericExpr. A better
    // approach is for subst_leaf to also take the annotation, which we should
    // implement after we use real non-() annotations
    pub fn subst<Head2, Leaf2>(
        self,
        subst_leaf: &mut impl FnMut(Leaf) -> GenericExpr<Head2, Leaf2, Ann>,
        subst_head: &mut impl FnMut(Head) -> Head2,
    ) -> GenericExpr<Head2, Leaf2, Ann> {
        match self {
            GenericExpr::Lit(ann, lit) => GenericExpr::Lit(ann.clone(), lit.clone()),
            GenericExpr::Var(_ann, v) => subst_leaf(v),
            GenericExpr::Call(ann, op, children) => {
                let children = children
                    .into_iter()
                    .map(|c| c.subst(subst_leaf, subst_head))
                    .collect();
                GenericExpr::Call(ann.clone(), subst_head(op), children)
            }
        }
    }

    pub fn subst_leaf<Leaf2>(
        self,
        subst: &mut impl FnMut(Leaf) -> GenericExpr<Head, Leaf2, Ann>,
    ) -> GenericExpr<Head, Leaf2, Ann> {
        self.subst(subst, &mut |op| op.clone())
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

impl<Head: Display, Leaf: Display, Ann> GenericExpr<Head, Leaf, Ann> {
    /// Converts this expression into a
    /// s-expression (symbolic expression).
    /// Example: `(Add (Add 2 3) 4)`
    pub fn to_sexp(&self) -> Sexp {
        let res = match self {
            GenericExpr::Lit(_ann, lit) => Sexp::Symbol(lit.to_string()),
            GenericExpr::Var(_ann, v) => Sexp::Symbol(v.to_string()),
            GenericExpr::Call(_ann, op, children) => Sexp::List(
                vec![Sexp::Symbol(op.to_string())]
                    .into_iter()
                    .chain(children.iter().map(|c| c.to_sexp()))
                    .collect(),
            ),
        };
        res
    }
}

impl<Head, Leaf, Ann> Display for GenericExpr<Head, Leaf, Ann>
where
    Head: Display,
    Leaf: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sexp())
    }
}

// currently only used for testing, but no reason it couldn't be used elsewhere later
#[cfg(test)]
pub(crate) fn parse_expr(s: &str) -> Result<Expr, lalrpop_util::ParseError<usize, String, String>> {
    let parser = ast::parse::ExprParser::new();
    parser
        .parse(s)
        .map_err(|e| e.map_token(|tok| tok.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_display_roundtrip() {
        let s = r#"(f (g a 3) 4.0 (H "hello"))"#;
        let e = parse_expr(s).unwrap();
        assert_eq!(format!("{}", e), s);
    }
}
