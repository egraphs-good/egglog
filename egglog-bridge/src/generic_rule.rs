use std::fmt::{Display, Formatter};
use std::hash::Hash;

use crate::{GenericActions, GenericExpr, GenericFact, Span};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GenericRule<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub span: Span,
    pub head: GenericActions<Head, Leaf>,
    pub body: Vec<GenericFact<Head, Leaf>>,
}

impl<Head, Leaf> GenericRule<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub(crate) fn visit_exprs(
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
        }
    }

    pub(crate) fn fmt_with_ruleset(
        &self,
        f: &mut Formatter,
        ruleset: &str,
        name: &str,
    ) -> std::fmt::Result {
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
        let ruleset = if !ruleset.is_empty() {
            format!(":ruleset {}", ruleset)
        } else {
            "".into()
        };
        let name = if !name.is_empty() {
            format!(":name \"{}\"", name)
        } else {
            "".into()
        };
        write!(f, ")\n{} {} {})", indent, ruleset, name)
    }
}
