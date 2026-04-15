use egglog_ast::generic_ast::GenericExpr;

use crate::{
    EGraph, ResolvedExpr, ResolvedFact,
    ast::{Command, Expr, Fact, GenericNCommand, ResolvedNCommand},
    util::FreshGen,
};

/// Thin wrapper around an [`EGraph`] for the slotted encoding
pub(crate) struct SlottedInstrumentor<'a> {
    pub(crate) egraph: &'a mut EGraph,
}

impl<'a> SlottedInstrumentor<'a> {
    pub(crate) fn add_slotted_encoding(
        egraph: &'a mut EGraph,
        program: Vec<ResolvedNCommand>,
    ) -> Vec<Command> {
        Self { egraph }.add_slotted_encoding_helper(program)
    }

    fn add_slotted_encoding_helper(&mut self, program: Vec<ResolvedNCommand>) -> Vec<Command> {
        let mut res = vec![];

        for r in program {
            res.push(self.add_slotted_encoding_one(r));
        }

        res
    }

    fn add_intermediate_renamings_expr(&mut self, expr: &ResolvedExpr) -> Expr {
        let fresh_counter = self.egraph.parser.symbol_gen.fresh("m");
        let with_children = match expr {
            GenericExpr::Var(span, v) => Expr::Var(span.clone(), v.to_string()),
            GenericExpr::Call(span, head, children) => {
                let converted_children = children
                    .iter()
                    .map(|child| self.add_intermediate_renamings_expr(child))
                    .collect();
                Expr::Call(span.clone(), head.to_string(), converted_children)
            }
            GenericExpr::Lit(span, literal) => Expr::Lit(span.clone(), literal.clone()),
        };

        let span = expr.span();
        // Wrap the expression in a Rename: (Rename fresh_counter with_children)
        Expr::Call(
            span.clone(),
            "Rename".to_string(),
            vec![Expr::Var(span.clone(), fresh_counter), with_children],
        )
    }

    // Adds an intermediate Rename around every sub-expression
    fn add_intermediate_renamings_query(&mut self, generic_fact: &ResolvedFact) -> Fact {
        match generic_fact {
            ResolvedFact::Eq(s, lhs, rhs) => Fact::Eq(
                s.clone(),
                self.add_intermediate_renamings_expr(lhs),
                self.add_intermediate_renamings_expr(rhs),
            ),
            ResolvedFact::Fact(expr) => Fact::Fact(self.add_intermediate_renamings_expr(expr)),
        }
    }

    // Given a fact and a variable v, find all the top level renamings for that variable.
    // A renaming is an expression, mapping from the variable to the name at the top level.
    // For example (Rename m (Add (Rename m2 v) (Rename m3 v))) gives two renamings at the top level for v:
    // (compose m m2) and (compose m m3)
    fn find_renamings_around_var(
        &self,
        generic_fact: &Fact,
        v: &str,
        current_renaming: Expr,
    ) -> Vec<Expr> {
        match generic_fact {
            Fact::Eq(_, lhs, rhs) => {
                let mut result = self.find_renamings_in_expr(lhs, v, current_renaming.clone());
                result.extend(self.find_renamings_in_expr(rhs, v, current_renaming));
                result
            }
            Fact::Fact(expr) => self.find_renamings_in_expr(expr, v, current_renaming),
        }
    }

    /// Helper: walk an Expr tree (already containing Rename wrappers) looking for
    /// occurrences of variable `v`. Accumulates the composition of Rename maps
    /// along the path and returns the composed renaming for each occurrence.
    ///
    /// When we see `(Rename m inner)`, the effective renaming becomes
    /// `(compose current_renaming m)` and we recurse into `inner`.
    /// When we reach `Var(name)` where `name == v`, we return `current_renaming`.
    fn find_renamings_in_expr(
        &self,
        expr: &Expr,
        v: &str,
        current_renaming: Expr,
    ) -> Vec<Expr> {
        match expr {
            GenericExpr::Var(_, name) => {
                if name == v {
                    vec![current_renaming]
                } else {
                    vec![]
                }
            }
            GenericExpr::Call(span, head, children) if head == "Rename" && children.len() == 2 => {
                // (Rename m inner) — compose the current renaming with m
                let m = &children[0];
                let inner = &children[1];
                let composed = Expr::Call(
                    span.clone(),
                    "compose".to_string(),
                    vec![current_renaming, m.clone()],
                );
                self.find_renamings_in_expr(inner, v, composed)
            }
            GenericExpr::Call(_, _, children) => {
                let mut result = vec![];
                for child in children {
                    result.extend(self.find_renamings_in_expr(child, v, current_renaming.clone()));
                }
                result
            }
            GenericExpr::Lit(_, _) => vec![],
        }
    }

    fn add_slotted_encoding_one(&mut self, command: ResolvedNCommand) -> Command {
        match command {
            // TODO just doing "user rules" we don't have a header
            GenericNCommand::NormRule { rule } if rule.name.contains("user") => {
                todo!()
            }
            _ => command.to_command().make_unresolved(),
        }
    }
}
