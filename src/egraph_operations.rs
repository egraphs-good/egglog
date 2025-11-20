use crate::span;
use egglog_ast::span::{RustSpan, Span};
use egglog_core_relations::Value;

use crate::{
    EGraph, Error, ProofStore, TermProofId,
    ast::{
        Action, Command, Expr, Facts, GenericActions, GenericRule, RunConfig, Schedule, Schema,
        collect_query_vars,
    },
    util::{FreshGen, IndexMap},
};

/// Represents a single match of a query, containing values for all query variables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryMatch {
    /// Maps variable names to their values in this match
    bindings: IndexMap<String, Value>,
}

impl QueryMatch {
    /// Get the value bound to a variable name in this match.
    pub fn get(&self, var_name: &str) -> Option<Value> {
        self.bindings.get(var_name).copied()
    }

    /// Get all variable names in this match.
    pub fn vars(&self) -> impl Iterator<Item = &str> {
        self.bindings.keys().map(|s| s.as_str())
    }

    /// Get the number of variables in this match.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Check if this match has no variables.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

impl EGraph {
    /// Returns all matches for the given query as a vector of QueryMatch structs.
    ///
    /// Each QueryMatch contains bindings for all variables in the query, including
    /// variables from both sides of equality constraints (excluding globals).
    ///
    /// # Example
    /// ```
    /// # use egglog::prelude::*;
    /// # let mut egraph = EGraph::with_proofs();
    /// egraph.parse_and_run_program(None, "
    ///     (datatype Math
    ///         (Num i64)
    ///         (Add Math Math))
    ///     (Add (Num 1) (Num 2))
    /// ").unwrap();
    ///
    /// // Query for all Add expressions with an equality constraint.
    /// // Note that all variables appear in the match: lhs, x, and y.
    /// let matches = egraph.get_matches(facts![(= lhs (Add x y))]).unwrap();
    ///
    /// // We found 1 match with all variables bound: lhs, x, and y.
    /// // The variable 'lhs' comes from the left side of the equality,
    /// // while 'x' and 'y' come from the right side.
    /// assert_eq!(matches.len(), 1);
    /// assert!(matches[0].get("lhs").is_some());
    /// assert!(matches[0].get("x").is_some());
    /// assert!(matches[0].get("y").is_some());
    /// assert_eq!(matches[0].len(), 3);
    /// ```
    pub fn get_matches(&mut self, facts: Facts<String, String>) -> Result<Vec<QueryMatch>, Error> {
        let Facts(query_facts) = facts;

        let span = span!();

        let resolved_facts = self
            .type_info
            .typecheck_facts(&mut self.parser.symbol_gen, &query_facts)?;
        let query_vars = collect_query_vars(&resolved_facts);

        let constructor_name = self.parser.symbol_gen.fresh("get_matches_ctor");
        let relation_name = self.parser.symbol_gen.fresh("get_matches_rel");
        let ruleset_name = self.parser.symbol_gen.fresh("get_matches_ruleset");
        let rule_name = self.parser.symbol_gen.fresh("get_matches_rule");
        let match_sort_name = self.parser.symbol_gen.fresh("get_matches_sort");

        let constructor_schema = {
            let inputs = query_vars
                .iter()
                .map(|(_, sort)| sort.name().to_string())
                .collect::<Vec<_>>();
            Schema::new(inputs, match_sort_name.clone())
        };

        // TODO currently using Push and Pop to avoid touching the e-graph here.
        // we should either make push and pop fast or rewrite to clean up the new table and rule.
        let mut program = vec![
            Command::Push(1),
            Command::Sort(span.clone(), match_sort_name.clone(), None),
            Command::Constructor {
                span: span.clone(),
                name: constructor_name.clone(),
                schema: constructor_schema,
                cost: None,
                unextractable: true,
            },
            Command::Relation {
                span: span.clone(),
                name: relation_name.clone(),
                inputs: query_vars
                    .iter()
                    .map(|(_, sort)| sort.name().to_string())
                    .collect(),
            },
            Command::AddRuleset(span.clone(), ruleset_name.clone()),
        ];

        let body_facts = query_facts.clone();
        let action_expr = {
            let args = query_vars
                .iter()
                .map(|(var, _)| Expr::Var(span.clone(), var.name.clone()))
                .collect();
            Expr::Call(span.clone(), constructor_name.clone(), args)
        };

        let rule_actions = GenericActions(vec![Action::Expr(span.clone(), action_expr)]);

        program.push(Command::Rule {
            rule: GenericRule {
                span: span.clone(),
                head: rule_actions,
                body: body_facts,
                name: rule_name.clone(),
                ruleset: ruleset_name.clone(),
            },
        });

        program.push(Command::RunSchedule(Schedule::Run(
            span.clone(),
            RunConfig {
                ruleset: ruleset_name.clone(),
                until: None,
            },
        )));
        self.run_program(program)?;

        let constructor_function = self
            .functions
            .get(&constructor_name)
            .expect("constructor should exist");

        let mut results = Vec::new();
        self.backend
            .for_each(constructor_function.backend_id, |row| {
                let mut bindings = IndexMap::default();
                for ((var, _), value) in query_vars.iter().zip(row.vals.iter()) {
                    bindings.insert(var.name.clone(), *value);
                }
                results.push(QueryMatch { bindings });
            });

        self.run_program(vec![Command::Pop(span.clone(), 1)])?;

        Ok(results)
    }
    /// Runs the query and produces a proof for any bound variable; returns the first proof found.
    pub fn prove_query(
        &mut self,
        facts: Facts<String, String>,
        store: &mut ProofStore,
    ) -> Result<Option<TermProofId>, Error> {
        let Facts(query_facts) = facts;
        if !self.backend.proofs_enabled() {
            return Err(Error::BackendError(
                "get_proof requires proofs to be enabled. Create the EGraph with EGraph::with_proofs()."
                    .to_string(),
            ));
        }

        let span = span!();
        let resolved_facts = self
            .type_info
            .typecheck_facts(&mut self.parser.symbol_gen, &query_facts)?;
        let query_vars = collect_query_vars(&resolved_facts);

        for (target_var, target_sort) in query_vars {
            let constructor_name = self.parser.symbol_gen.fresh("get_proof_ctor");
            let ruleset_name = self.parser.symbol_gen.fresh("get_proof_ruleset");
            let rule_name = self.parser.symbol_gen.fresh("get_proof_rule");
            let proof_sort_name = self.parser.symbol_gen.fresh("get_proof_sort");

            let constructor_schema = Schema::new(
                vec![target_sort.name().to_string()],
                proof_sort_name.clone(),
            );

            let mut program = vec![
                Command::Push(1),
                Command::Sort(span.clone(), proof_sort_name.clone(), None),
                Command::Constructor {
                    span: span.clone(),
                    name: constructor_name.clone(),
                    schema: constructor_schema,
                    cost: None,
                    unextractable: true,
                },
                Command::AddRuleset(span.clone(), ruleset_name.clone()),
            ];

            let body_facts = query_facts.clone();
            let action_expr = Expr::Call(
                span.clone(),
                constructor_name.clone(),
                vec![Expr::Var(span.clone(), target_var.name.clone())],
            );
            let rule_actions = GenericActions(vec![Action::Expr(span.clone(), action_expr)]);

            program.push(Command::Rule {
                rule: GenericRule {
                    span: span.clone(),
                    head: rule_actions,
                    body: body_facts,
                    name: rule_name.clone(),
                    ruleset: ruleset_name.clone(),
                },
            });

            program.push(Command::RunSchedule(Schedule::Run(
                span.clone(),
                RunConfig {
                    ruleset: ruleset_name.clone(),
                    until: None,
                },
            )));
            self.run_program(program)?;

            let mut captured = None;
            if let Some(constructor_function) = self.functions.get(&constructor_name) {
                self.backend
                    .for_each(constructor_function.backend_id, |row| {
                        if captured.is_none() {
                            captured = row.vals.first().copied();
                        }
                    });
            }

            self.run_program(vec![Command::Pop(span.clone(), 1)])?;

            if let Some(value) = captured {
                let proof = self
                    .explain_term(value, store)
                    .map_err(|e| Error::BackendError(e.to_string()))?;
                return Ok(Some(proof));
            }
        }

        Ok(None)
    }
}
