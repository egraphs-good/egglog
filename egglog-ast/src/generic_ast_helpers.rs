use std::fmt::{Display, Formatter};
use std::hash::Hash;

use ordered_float::OrderedFloat;

use super::util::ListDisplay;
use crate::generic_ast::*;
use crate::span::Span;

// Macro to implement From conversions for Literal types
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

impl<Head: Display, Leaf: Display> Display for GenericRule<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        let indent = " ".repeat(7);
        write!(f, "(rule (")?;
        for (i, fact) in self.body.iter().enumerate() {
            if i > 0 {
                write!(f, "{}", indent)?;
            }

            if i != self.body.len() - 1 {
                writeln!(f, "{}", fact)?;
            } else {
                write!(f, "{}", fact)?;
            }
        }
        write!(f, ")\n      (")?;
        for (i, action) in self.head.0.iter().enumerate() {
            if i > 0 {
                write!(f, "{}", indent)?;
            }
            if i != self.head.0.len() - 1 {
                writeln!(f, "{}", action)?;
            } else {
                write!(f, "{}", action)?;
            }
        }
        let ruleset = if !self.ruleset.is_empty() {
            format!(":ruleset {}", self.ruleset)
        } else {
            "".into()
        };
        let name = if !self.name.is_empty() {
            format!(":name \"{}\"", self.name)
        } else {
            "".into()
        };
        write!(f, ")\n{} {} {})", indent, ruleset, name)
    }
}

// Use the macro for Int, Float, and String conversions
impl_from!(Int(i64));
impl_from!(Float(OrderedFloat<f64>));
impl_from!(String(String));

impl<Head: Display, Leaf: Display> Display for GenericFact<Head, Leaf> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            GenericFact::Eq {
                field1: _,
                field2: e1,
                field3: e2,
            } => write!(f, "(= {e1} {e2})"),
            GenericFact::Fact { field1: expr } => write!(f, "{expr}"),
        }
    }
}

// Implement Display for GenericAction
impl<Head: Display, Leaf: Display> Display for GenericAction<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            GenericAction::Let {
                field1: _,
                field2: lhs,
                field3: rhs,
            } => write!(f, "(let {} {})", lhs, rhs),
            GenericAction::Set {
                field1: _,
                field2: lhs,
                field3: args,
                field4: rhs,
            } => write!(
                f,
                "(set ({} {}) {})",
                lhs,
                args.iter()
                    .map(|a| format!("{}", a))
                    .collect::<Vec<_>>()
                    .join(" "),
                rhs
            ),
            GenericAction::Union {
                field1: _,
                field2: lhs,
                field3: rhs,
            } => write!(f, "(union {} {})", lhs, rhs),
            GenericAction::Change {
                field1: _,
                field2: change,
                field3: lhs,
                field4: args,
            } => {
                let change_str = match change {
                    Change::Delete => "delete",
                    Change::Subsume => "subsume",
                };
                write!(
                    f,
                    "({} ({} {}))",
                    change_str,
                    lhs,
                    args.iter()
                        .map(|a| format!("{}", a))
                        .collect::<Vec<_>>()
                        .join(" ")
                )
            }
            GenericAction::Panic {
                field1: _,
                field2: msg,
            } => write!(f, "(panic \"{}\")", msg),
            GenericAction::Expr {
                field1: _,
                field2: e,
            } => write!(f, "{}", e),
        }
    }
}

impl<Head, Leaf> Display for GenericExpr<Head, Leaf>
where
    Head: Display,
    Leaf: Display,
{
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            GenericExpr::Lit {
                field1: _ann,
                field2: lit,
            } => write!(f, "{lit}"),
            GenericExpr::Var {
                span: _ann,
                name: var,
            } => write!(f, "{var}"),
            GenericExpr::Call {
                field1: _ann,
                field2: op,
                field3: children,
            } => {
                write!(f, "({} {})", op, ListDisplay(children, " "))
            }
        }
    }
}

impl<Head, Leaf> Default for GenericActions<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    fn default() -> Self {
        Self(vec![])
    }
}

impl<Head, Leaf> GenericRule<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub fn visit_exprs(
        self,
        f: &mut impl FnMut(GenericExpr<Head, Leaf>) -> GenericExpr<Head, Leaf>,
    ) -> Self {
        Self {
            span: self.span,
            head: self.head.visit_exprs(f),
            body: self
                .body
                .into_iter()
                .map(|bexpr| bexpr.visit_exprs(f))
                .collect(),
            name: self.name.clone(),
            ruleset: self.ruleset.clone(),
        }
    }
}

impl<Head, Leaf> GenericActions<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &GenericAction<Head, Leaf>> {
        self.0.iter()
    }

    pub fn visit_exprs(
        self,
        f: &mut impl FnMut(GenericExpr<Head, Leaf>) -> GenericExpr<Head, Leaf>,
    ) -> Self {
        Self(self.0.into_iter().map(|a| a.visit_exprs(f)).collect())
    }

    pub fn new(actions: Vec<GenericAction<Head, Leaf>>) -> Self {
        Self(actions)
    }

    pub fn singleton(action: GenericAction<Head, Leaf>) -> Self {
        Self(vec![action])
    }
}

impl<Head, Leaf> GenericAction<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + Eq + Display + Hash,
{
    // Applys `f` to all expressions in the action.
    pub fn map_exprs(
        &self,
        f: &mut impl FnMut(&GenericExpr<Head, Leaf>) -> GenericExpr<Head, Leaf>,
    ) -> Self {
        match self {
            GenericAction::Let {
                field1: span,
                field2: lhs,
                field3: rhs,
            } => GenericAction::Let {
                field1: span.clone(),
                field2: lhs.clone(),
                field3: f(rhs),
            },
            GenericAction::Set {
                field1: span,
                field2: lhs,
                field3: args,
                field4: rhs,
            } => {
                let right = f(rhs);
                GenericAction::Set {
                    field1: span.clone(),
                    field2: lhs.clone(),
                    field3: args.iter().map(f).collect(),
                    field4: right,
                }
            }
            GenericAction::Change {
                field1: span,
                field2: change,
                field3: lhs,
                field4: args,
            } => GenericAction::Change {
                field1: span.clone(),
                field2: *change,
                field3: lhs.clone(),
                field4: args.iter().map(f).collect(),
            },
            GenericAction::Union {
                field1: span,
                field2: lhs,
                field3: rhs,
            } => GenericAction::Union {
                field1: span.clone(),
                field2: f(lhs),
                field3: f(rhs),
            },
            GenericAction::Panic {
                field1: span,
                field2: msg,
            } => GenericAction::Panic {
                field1: span.clone(),
                field2: msg.clone(),
            },
            GenericAction::Expr {
                field1: span,
                field2: e,
            } => GenericAction::Expr {
                field1: span.clone(),
                field2: f(e),
            },
        }
    }

    /// Applys `f` to all sub-expressions (including `self`)
    /// bottom-up, collecting the results.
    pub fn visit_exprs(
        self,
        f: &mut impl FnMut(GenericExpr<Head, Leaf>) -> GenericExpr<Head, Leaf>,
    ) -> Self {
        match self {
            GenericAction::Let {
                field1: span,
                field2: lhs,
                field3: rhs,
            } => GenericAction::Let {
                field1: span,
                field2: lhs.clone(),
                field3: rhs.visit_exprs(f),
            },
            // TODO should we refactor `Set` so that we can map over Expr::Call(lhs, args)?
            // This seems more natural to oflatt
            // Currently, visit_exprs does not apply f to the first argument of Set.
            GenericAction::Set {
                field1: span,
                field2: lhs,
                field3: args,
                field4: rhs,
            } => {
                let args = args.into_iter().map(|e| e.visit_exprs(f)).collect();
                GenericAction::Set {
                    field1: span,
                    field2: lhs.clone(),
                    field3: args,
                    field4: rhs.visit_exprs(f),
                }
            }
            GenericAction::Change {
                field1: span,
                field2: change,
                field3: lhs,
                field4: args,
            } => {
                let args = args.into_iter().map(|e| e.visit_exprs(f)).collect();
                GenericAction::Change {
                    field1: span,
                    field2: change,
                    field3: lhs.clone(),
                    field4: args,
                }
            }
            GenericAction::Union {
                field1: span,
                field2: lhs,
                field3: rhs,
            } => GenericAction::Union {
                field1: span,
                field2: lhs.visit_exprs(f),
                field3: rhs.visit_exprs(f),
            },
            GenericAction::Panic {
                field1: span,
                field2: msg,
            } => GenericAction::Panic {
                field1: span,
                field2: msg.clone(),
            },
            GenericAction::Expr {
                field1: span,
                field2: e,
            } => GenericAction::Expr {
                field1: span,
                field2: e.visit_exprs(f),
            },
        }
    }

    pub fn subst(&self, subst: &mut impl FnMut(&Span, &Leaf) -> GenericExpr<Head, Leaf>) -> Self {
        self.map_exprs(&mut |e| e.subst_leaf(subst))
    }

    pub fn map_def_use(self, fvar: &mut impl FnMut(Leaf, bool) -> Leaf) -> Self {
        macro_rules! fvar_expr {
            () => {
                |span, s: _| GenericExpr::Var {
                    span: span.clone(),
                    name: fvar(s.clone(), false),
                }
            };
        }
        match self {
            GenericAction::Let {
                field1: span,
                field2: lhs,
                field3: rhs,
            } => {
                let lhs = fvar(lhs, true);
                let rhs = rhs.subst_leaf(&mut fvar_expr!());
                GenericAction::Let {
                    field1: span,
                    field2: lhs,
                    field3: rhs,
                }
            }
            GenericAction::Set {
                field1: span,
                field2: lhs,
                field3: args,
                field4: rhs,
            } => {
                let args = args
                    .into_iter()
                    .map(|e| e.subst_leaf(&mut fvar_expr!()))
                    .collect();
                let rhs = rhs.subst_leaf(&mut fvar_expr!());
                GenericAction::Set {
                    field1: span,
                    field2: lhs.clone(),
                    field3: args,
                    field4: rhs,
                }
            }
            GenericAction::Change {
                field1: span,
                field2: change,
                field3: lhs,
                field4: args,
            } => {
                let args = args
                    .into_iter()
                    .map(|e| e.subst_leaf(&mut fvar_expr!()))
                    .collect();
                GenericAction::Change {
                    field1: span,
                    field2: change,
                    field3: lhs.clone(),
                    field4: args,
                }
            }
            GenericAction::Union {
                field1: span,
                field2: lhs,
                field3: rhs,
            } => {
                let lhs = lhs.subst_leaf(&mut fvar_expr!());
                let rhs = rhs.subst_leaf(&mut fvar_expr!());
                GenericAction::Union {
                    field1: span,
                    field2: lhs,
                    field3: rhs,
                }
            }
            GenericAction::Panic {
                field1: span,
                field2: msg,
            } => GenericAction::Panic {
                field1: span,
                field2: msg.clone(),
            },
            GenericAction::Expr {
                field1: span,
                field2: e,
            } => GenericAction::Expr {
                field1: span,
                field2: e.subst_leaf(&mut fvar_expr!()),
            },
        }
    }
}

impl<Head, Leaf> GenericFact<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub fn visit_exprs(
        self,
        f: &mut impl FnMut(GenericExpr<Head, Leaf>) -> GenericExpr<Head, Leaf>,
    ) -> GenericFact<Head, Leaf> {
        match self {
            GenericFact::Eq {
                field1: span,
                field2: e1,
                field3: e2,
            } => GenericFact::Eq {
                field1: span,
                field2: e1.visit_exprs(f),
                field3: e2.visit_exprs(f),
            },
            GenericFact::Fact { field1: expr } => GenericFact::Fact {
                field1: expr.visit_exprs(f),
            },
        }
    }

    pub fn map_exprs<Head2, Leaf2>(
        &self,
        f: &mut impl FnMut(&GenericExpr<Head, Leaf>) -> GenericExpr<Head2, Leaf2>,
    ) -> GenericFact<Head2, Leaf2> {
        match self {
            GenericFact::Eq {
                field1: span,
                field2: e1,
                field3: e2,
            } => GenericFact::Eq {
                field1: span.clone(),
                field2: f(e1),
                field3: f(e2),
            },
            GenericFact::Fact { field1: expr } => GenericFact::Fact { field1: f(expr) },
        }
    }

    pub fn subst<Leaf2, Head2>(
        &self,
        subst_leaf: &mut impl FnMut(&Span, &Leaf) -> GenericExpr<Head2, Leaf2>,
        subst_head: &mut impl FnMut(&Head) -> Head2,
    ) -> GenericFact<Head2, Leaf2> {
        self.map_exprs(&mut |e| e.subst(subst_leaf, subst_head))
    }
}

impl<Head, Leaf> GenericFact<Head, Leaf>
where
    Leaf: Clone + PartialEq + Eq + Display + Hash,
    Head: Clone + Display,
{
    pub fn make_unresolved(self) -> GenericFact<String, String> {
        self.subst(
            &mut |span, v| GenericExpr::Var {
                span: span.clone(),
                name: v.to_string(),
            },
            &mut |h| h.to_string(),
        )
    }
}

impl<Head: Clone + Display, Leaf: Hash + Clone + Display + Eq> GenericExpr<Head, Leaf> {
    pub fn span(&self) -> Span {
        match self {
            GenericExpr::Lit {
                field1: span,
                field2: _,
            } => span.clone(),
            GenericExpr::Var { span, name: _ } => span.clone(),
            GenericExpr::Call {
                field1: span,
                field2: _,
                field3: _,
            } => span.clone(),
        }
    }

    pub fn is_var(&self) -> bool {
        matches!(self, GenericExpr::Var { span: _, name: _ })
    }

    pub fn get_var(&self) -> Option<Leaf> {
        match self {
            GenericExpr::Var {
                span: _ann,
                name: v,
            } => Some(v.clone()),
            _ => None,
        }
    }

    fn children(&self) -> &[Self] {
        match self {
            GenericExpr::Var { span: _, name: _ }
            | GenericExpr::Lit {
                field1: _,
                field2: _,
            } => &[],
            GenericExpr::Call {
                field1: _,
                field2: _,
                field3: children,
            } => children,
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
            GenericExpr::Lit { .. } => f(self),
            GenericExpr::Var { .. } => f(self),
            GenericExpr::Call {
                field1: span,
                field2: op,
                field3: children,
            } => {
                let children = children.into_iter().map(|c| c.visit_exprs(f)).collect();
                f(GenericExpr::Call {
                    field1: span,
                    field2: op.clone(),
                    field3: children,
                })
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
            GenericExpr::Lit {
                field1: span,
                field2: lit,
            } => GenericExpr::Lit {
                field1: span.clone(),
                field2: lit.clone(),
            },
            GenericExpr::Var { span, name: v } => subst_leaf(span, v),
            GenericExpr::Call {
                field1: span,
                field2: op,
                field3: children,
            } => {
                let children = children
                    .iter()
                    .map(|c| c.subst(subst_leaf, subst_head))
                    .collect();
                GenericExpr::Call {
                    field1: span.clone(),
                    field2: subst_head(op),
                    field3: children,
                }
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
            GenericExpr::Lit {
                field1: _ann,
                field2: _l,
            } => Box::new(std::iter::empty()),
            GenericExpr::Var {
                span: _ann,
                name: v,
            } => Box::new(std::iter::once(v.clone())),
            GenericExpr::Call {
                field1: _ann,
                field2: _head,
                field3: exprs,
            } => Box::new(exprs.iter().flat_map(|e| e.vars())),
        };
        iterator
    }
}

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
