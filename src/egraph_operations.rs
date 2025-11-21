use egglog_ast::{
    generic_ast::{GenericExpr, GenericFact},
    span::Span,
};
use egglog_core_relations::{Value, make_external_func};

use crate::{
    BackendRule, EGraph, Error,
    ast::{
        Command, Fact, MappedExpr, ResolvedActions, ResolvedNCommand, ResolvedRule, ResolvedVar,
    },
    core::{ResolvedAtomTerm, ResolvedCall, ResolvedRuleExt},
    util::{FreshGen, IndexMap, IndexSet},
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
    ///     (datatype Math (Num i64) (Add Math Math))
    ///     (Add (Num 1) (Num 2))
    ///     (Add (Num 3) (Num 4))
    /// ").unwrap();
    ///
    /// // Query for all Add expressions
    /// let matches = egraph.get_matches(&[
    ///     Fact::Fact(expr!((Add x y)))
    /// ]).unwrap();
    ///
    /// // We found 2 matches, each with x and y bound
    /// assert_eq!(matches.len(), 2);
    /// assert!(matches[0].get("x").is_some());
    /// assert!(matches[0].get("y").is_some());
    /// assert_eq!(matches[0].len(), 2);
    /// ```
    pub fn get_matches(&mut self, facts: &[Fact]) -> Result<Vec<QueryMatch>, Error> {
        // get_matches requires proofs to be enabled because we need to access the original
        // query structure (including equality-bound variables) via mapped_facts
        if !self.backend.proofs_enabled() {
            return Err(Error::BackendError(
                "get_matches requires proofs to be enabled. Create the EGraph with EGraph::with_proofs().".to_string(),
            ));
        }

        let span = Span::Panic; // Using Panic as a marker span since we're building internal query
        let fresh_name = self.parser.symbol_gen.fresh("get_matches");
        let fresh_ruleset = self.parser.symbol_gen.fresh("get_matches_ruleset");

        // Parse and resolve the facts by creating a temporary check command
        let check_command = Command::Check(span.clone(), facts.to_vec());
        let resolved_commands = self.process_command(check_command)?;
        let resolved_facts = match &resolved_commands[0] {
            ResolvedNCommand::Check(_, facts) => facts.clone(),
            _ => unreachable!("Check command should resolve to Check"),
        };

        let rule = ResolvedRule {
            span: span.clone(),
            head: ResolvedActions::default(),
            body: resolved_facts.clone(),
            name: fresh_name.clone(),
            ruleset: fresh_ruleset.clone(),
        };

        let canonical_rule =
            rule.to_canonicalized_core_rule(&self.type_info, &mut self.parser.symbol_gen)?;
        dbg!(&canonical_rule.mapped_facts);
        let query = canonical_rule.rule.body.clone();

        // Create a side channel to collect matches
        let matches: egglog_bridge::SideChannel<Vec<QueryMatch>> = Default::default();
        let matches_ref = matches.clone();

        let mut ordered_vars = IndexSet::default();
        for fact in &canonical_rule.mapped_facts {
            match fact {
                GenericFact::Eq(_, lhs, rhs) => {
                    Self::collect_vars_from_mapped(lhs, &mut ordered_vars);
                    Self::collect_vars_from_mapped(rhs, &mut ordered_vars);
                }
                GenericFact::Fact(expr) => {
                    Self::collect_vars_from_mapped(expr, &mut ordered_vars);
                }
            }
        }
        let leaf_var_order: Vec<_> = ordered_vars.into_iter().collect();

        // TODO due to a bug, not all variables in the query may appear in the translator's
        // resolved_var_entries?
        let leaf_vars = {
            dbg!(&leaf_var_order);
            let mut translator = BackendRule::new(
                self.backend.new_rule("get_matches_temp", false),
                &self.functions,
                &self.type_info,
                canonical_rule.mapped_facts.clone(),
            );
            translator.query(&query, true);

            // Filter the resolved variables to only include those that exist in the canonicalized query
            let vars: Vec<_> = leaf_var_order
                .into_iter()
                .filter(|var| translator.resolved_var_entries.contains_key(var))
                .collect();
            dbg!(
                translator
                    .resolved_var_entries
                    .keys()
                    .map(|v| v.name.clone())
                    .collect::<Vec<_>>()
            );

            // Drop translator which releases the borrow on self.backend
            let temp_id = translator.build();
            self.backend.free_rule(temp_id);

            vars
        };

        let leaf_vars_for_closure = leaf_vars.clone();

        // Now that translator is dropped, we can register the external function
        let ext_id =
            self.backend
                .register_external_func(make_external_func(move |_es, vals: &[Value]| {
                    let mut bindings = IndexMap::default();
                    for (var, val) in leaf_vars_for_closure.iter().zip(vals.iter()) {
                        bindings.insert(var.name.clone(), *val);
                    }
                    matches_ref
                        .lock()
                        .unwrap()
                        .get_or_insert_with(Vec::new)
                        .push(QueryMatch { bindings });
                    Some(Value::new_const(0))
                }));

        // Second pass: create the actual translator for the real rule
        // Note: We pass empty mapped_facts because we don't need proof reconstruction
        // for the get_matches rule itself - we only needed proofs to extract variable names
        let mut translator = BackendRule::new(
            self.backend.new_rule("get_matches", false),
            &self.functions,
            &self.type_info,
            canonical_rule.mapped_facts.clone(),
        );
        translator.query(&query, true);
        dbg!(
            translator
                .resolved_var_entries
                .keys()
                .map(|v| v.name.clone())
                .collect::<Vec<_>>()
        );

        // Call the external function with all leaf variables as arguments.
        // Note: Some variables from the original query (like those in equality constraints)
        // may have been canonicalized away and won't be in the translator's entries.
        // We filter to only include variables that actually exist in the query.
        let var_entries: Vec<_> = leaf_vars
            .iter()
            .filter_map(|var| {
                let term = ResolvedAtomTerm::Var(span.clone(), var.clone());
                // Check if this variable actually exists in the translator's entries
                if translator.resolved_var_entries.contains_key(var) {
                    Some(translator.entry(&term))
                } else {
                    None
                }
            })
            .collect();

        translator
            .rb
            .call_external_func(ext_id, &var_entries, egglog_bridge::ColumnTy::Id, || {
                "get_matches should not panic".to_string()
            });
        let id = translator.build();
        let _ = self.backend.run_rules(&[id]).unwrap();
        self.backend.free_rule(id);
        self.backend.free_external_func(ext_id);

        Ok(matches.lock().unwrap().take().unwrap_or_default())
    }

    fn collect_vars_from_mapped(
        expr: &MappedExpr<ResolvedCall, ResolvedVar>,
        vars: &mut IndexSet<ResolvedVar>,
    ) {
        match expr {
            GenericExpr::Var(_, var) => {
                if !var.name.starts_with('$') {
                    vars.insert(var.clone());
                }
            }
            GenericExpr::Call(_, _, children) => {
                for child in children {
                    Self::collect_vars_from_mapped(child, vars);
                }
            }
            GenericExpr::Lit(_, _) => {}
        }
    }
}
