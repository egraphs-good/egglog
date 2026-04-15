use egglog_ast::generic_ast::{GenericAction, GenericActions, GenericExpr, GenericRule};

use crate::{
    EGraph, ResolvedExpr, ResolvedFact,
    ast::{Action, Command, Expr, Fact, GenericNCommand, ResolvedExprExt, ResolvedNCommand},
    util::{FreshGen, HashMap, HashSet},
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

        // Only wrap in Rename if the expression's output type is an eq sort
        // (user-defined sorts like U). Primitive types (String, i64, etc.) are not wrapped.
        if expr.output_type().is_eq_sort() {
            let fresh_counter = self.egraph.parser.symbol_gen.fresh("m");
            let span = expr.span();
            Expr::Call(
                span.clone(),
                "Rename".to_string(),
                vec![Expr::Var(span.clone(), fresh_counter), with_children],
            )
        } else {
            with_children
        }
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
    fn find_renamings_around_var(&self, generic_fact: &Fact, v: &str) -> Vec<Expr> {
        match generic_fact {
            Fact::Eq(_, lhs, rhs) => {
                let mut result = self.find_renamings_in_expr(lhs, v, None);
                result.extend(self.find_renamings_in_expr(rhs, v, None));
                result
            }
            Fact::Fact(expr) => self.find_renamings_in_expr(expr, v, None),
        }
    }

    /// Helper: walk an Expr tree (already containing Rename wrappers) looking for
    /// occurrences of variable `v`. Accumulates the composition of Rename maps
    /// along the path and returns the composed renaming for each occurrence.
    ///
    /// Every variable is wrapped in at least one `Rename`, so `current_renaming`
    /// starts as `None` and becomes `Some(m)` at the first `Rename` node.
    /// At subsequent `Rename` nodes, the renaming is composed.
    /// When we reach `Var(name)` where `name == v`, we return `current_renaming`.
    fn find_renamings_in_expr(
        &self,
        expr: &Expr,
        v: &str,
        current_renaming: Option<Expr>,
    ) -> Vec<Expr> {
        match expr {
            GenericExpr::Var(_, name) => {
                if name == v {
                    // Every variable is inside a Rename, so current_renaming should be Some
                    vec![current_renaming.expect("variable should be wrapped in a Rename")]
                } else {
                    vec![]
                }
            }
            GenericExpr::Call(span, head, children) if head == "Rename" && children.len() == 2 => {
                let m = &children[0];
                let inner = &children[1];
                let composed = match current_renaming {
                    // First Rename encountered — use m directly
                    None => m.clone(),
                    // Subsequent Rename — compose with accumulated renaming
                    Some(r) => Expr::Call(span.clone(), "compose".to_string(), vec![r, m.clone()]),
                };
                self.find_renamings_in_expr(inner, v, Some(composed))
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

    /// Collect all unique variable names from a list of resolved facts.
    fn collect_vars_from_facts(facts: &[ResolvedFact]) -> HashSet<String> {
        let mut vars = HashSet::default();
        for fact in facts {
            fact.visit_vars(&mut |_span, leaf| {
                vars.insert(leaf.name.clone());
            });
        }
        vars
    }

    /// Given a variable-to-renaming map (built from the query), replace every
    /// occurrence of a variable `v` in the expression with `(Rename renaming v)`.
    fn apply_renamings_to_expr(expr: &Expr, var_to_renaming: &HashMap<String, Expr>) -> Expr {
        match expr {
            GenericExpr::Var(span, name) => {
                if let Some(renaming) = var_to_renaming.get(name) {
                    Expr::Call(
                        span.clone(),
                        "Rename".to_string(),
                        vec![renaming.clone(), Expr::Var(span.clone(), name.clone())],
                    )
                } else {
                    expr.clone()
                }
            }
            GenericExpr::Call(span, head, children) => {
                let new_children = children
                    .iter()
                    .map(|c| Self::apply_renamings_to_expr(c, var_to_renaming))
                    .collect();
                Expr::Call(span.clone(), head.clone(), new_children)
            }
            GenericExpr::Lit(_, _) => expr.clone(),
        }
    }

    /// Apply renamings to an action, wrapping variable references with their renamings.
    fn apply_renamings_to_action(
        action: &Action,
        var_to_renaming: &HashMap<String, Expr>,
    ) -> Action {
        match action {
            GenericAction::Union(span, lhs, rhs) => GenericAction::Union(
                span.clone(),
                Self::apply_renamings_to_expr(lhs, var_to_renaming),
                Self::apply_renamings_to_expr(rhs, var_to_renaming),
            ),
            GenericAction::Let(span, v, expr) => GenericAction::Let(
                span.clone(),
                v.clone(),
                Self::apply_renamings_to_expr(expr, var_to_renaming),
            ),
            GenericAction::Set(span, head, args, val) => GenericAction::Set(
                span.clone(),
                head.clone(),
                args.iter()
                    .map(|a| Self::apply_renamings_to_expr(a, var_to_renaming))
                    .collect(),
                Self::apply_renamings_to_expr(val, var_to_renaming),
            ),
            GenericAction::Change(span, change, head, args) => GenericAction::Change(
                span.clone(),
                *change,
                head.clone(),
                args.iter()
                    .map(|a| Self::apply_renamings_to_expr(a, var_to_renaming))
                    .collect(),
            ),
            GenericAction::Expr(span, expr) => GenericAction::Expr(
                span.clone(),
                Self::apply_renamings_to_expr(expr, var_to_renaming),
            ),
            GenericAction::Panic(span, msg) => GenericAction::Panic(span.clone(), msg.clone()),
        }
    }

    fn add_slotted_encoding_one(&mut self, command: ResolvedNCommand) -> Command {
        match command {
            GenericNCommand::NormRule { rule } if rule.name.contains("user") => {
                let span = rule.span.clone();

                // 1. Add intermediate renamings to the query (body) facts
                let renamed_facts: Vec<Fact> = rule
                    .body
                    .iter()
                    .map(|f| self.add_intermediate_renamings_query(f))
                    .collect();

                // 2. Collect all unique variables from the original query
                let all_vars = Self::collect_vars_from_facts(&rule.body);

                // 3. For each variable, gather all renamings across all renamed facts,
                //    then constrain them to be equal pairwise.
                let mut renaming_constraints: Vec<Fact> = vec![];
                let mut var_to_renaming: HashMap<String, Expr> = HashMap::default();

                for var in &all_vars {
                    let mut all_renamings: Vec<Expr> = vec![];
                    for renamed_fact in &renamed_facts {
                        let renamings = self.find_renamings_around_var(renamed_fact, var);
                        all_renamings.extend(renamings);
                    }

                    // Pick the first renaming as the canonical one for use in actions
                    if let Some(first) = all_renamings.first() {
                        var_to_renaming.insert(var.clone(), first.clone());
                    }

                    // Add equality constraints: all renamings for this var must be equal
                    for window in all_renamings.windows(2) {
                        renaming_constraints.push(Fact::Eq(
                            span.clone(),
                            window[0].clone(),
                            window[1].clone(),
                        ));
                    }
                }

                // 4. Build the new query: renamed facts + renaming equality constraints
                let mut new_body = renamed_facts;
                new_body.extend(renaming_constraints);

                // 5. Convert actions: make_unresolved, then apply renamings to variable references
                let unresolved_actions: Vec<Action> = rule.head.clone().make_unresolved().0;
                let new_actions: Vec<Action> = unresolved_actions
                    .iter()
                    .map(|a| Self::apply_renamings_to_action(a, &var_to_renaming))
                    .collect();

                // 6. Build the new rule command
                let res = Command::Rule {
                    rule: GenericRule {
                        span,
                        head: GenericActions(new_actions),
                        body: new_body,
                        name: rule.name.clone(),
                        ruleset: rule.ruleset.clone(),
                    },
                };

                eprintln!("{}", res);

                res
            }
            _ => command.to_command().make_unresolved(),
        }
    }
}
