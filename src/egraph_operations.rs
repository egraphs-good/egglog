use crate::span;
use egglog_ast::span::{RustSpan, Span};
use egglog_core_relations::Value;

use crate::{
    EGraph, Error,
    ast::{Facts, RunConfig, Schedule, collect_query_vars},
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
    /// Each QueryMatch contains bindings only for user-defined variables in the query.
    /// Internal variables generated during canonicalization (starting with $) are excluded.
    ///
    /// **Note**: This method requires proofs to be enabled. Create the EGraph with
    /// `EGraph::with_proofs()` to use this feature.
    ///
    /// TODO this implementation is in-progress.
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
    /// // Query for all Add expressions
    /// let matches = egraph.get_matches(facts![(= lhs (Add x y))]).unwrap();
    ///
    /// // We found 1 match with lhs, x, and y bound
    /// assert_eq!(matches.len(), 1);
    /// assert!(matches[0].get("x").is_some());
    /// assert!(matches[0].get("y").is_some());
    /// assert!(matches[0].get("lhs").is_some());
    /// assert_eq!(matches[0].len(), 3);
    /// ```
    pub fn get_matches(&mut self, facts: Facts<String, String>) -> Result<Vec<QueryMatch>, Error> {
        let Facts(query_facts) = facts;
        if !self.backend.proofs_enabled() {
            return Err(Error::BackendError(
                "get_matches requires proofs to be enabled. Create the EGraph with EGraph::with_proofs().".to_string(),
            ));
        }

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
            crate::ast::Schema::new(inputs, match_sort_name.clone())
        };

        let mut program = Vec::new();
        program.push(crate::ast::Command::Push(1));
        program.push(crate::ast::Command::Sort(
            span.clone(),
            match_sort_name.clone(),
            None,
        ));
        program.push(crate::ast::Command::Constructor {
            span: span.clone(),
            name: constructor_name.clone(),
            schema: constructor_schema,
            cost: None,
            unextractable: true,
        });

        program.push(crate::ast::Command::Relation {
            span: span.clone(),
            name: relation_name.clone(),
            inputs: query_vars
                .iter()
                .map(|(_, sort)| sort.name().to_string())
                .collect(),
        });

        program.push(crate::ast::Command::AddRuleset(
            span.clone(),
            ruleset_name.clone(),
        ));

        let body_facts = query_facts.clone();
        let action_expr = {
            let args = query_vars
                .iter()
                .map(|(var, _)| crate::ast::Expr::Var(span.clone(), var.name.clone()))
                .collect();
            crate::ast::Expr::Call(span.clone(), constructor_name.clone(), args)
        };

        let rule_actions =
            crate::ast::GenericActions(vec![crate::ast::Action::Expr(span.clone(), action_expr)]);

        program.push(crate::ast::Command::Rule {
            rule: crate::ast::GenericRule {
                span: span.clone(),
                head: rule_actions,
                body: body_facts,
                name: rule_name.clone(),
                ruleset: ruleset_name.clone(),
            },
        });

        program.push(crate::ast::Command::RunSchedule(Schedule::Run(
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

        self.run_program(vec![crate::ast::Command::Pop(span.clone(), 1)])?;

        Ok(results)
    }
}
