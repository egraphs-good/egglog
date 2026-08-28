use std::cell::OnceCell;
use std::sync::Arc;
use std::sync::Mutex;

use core_relations::{ExecutionState, ExternalFunction, Value};
use egglog_bridge::{
    ColumnTy, DefaultVal, FunctionConfig, FunctionId, MergeFn, RuleId, TableAction,
};
use egglog_reports::{IterationReport, RunReport};
use numeric_id::define_id;

use crate::{
    ast::{CompiledRule, ResolvedVar, Ruleset},
    core::GenericAtomTerm,
    util::IndexMap,
    *,
};

/// Read-only information about the e-graph handed to a scheduler callback.
pub struct SchedulerContext<'a> {
    egraph: &'a EGraph,
    num_nodes: OnceCell<usize>,
}

impl SchedulerContext<'_> {
    /// Return the current number of e-nodes. The count is computed only if the
    /// scheduler asks for it, then cached for the lifetime of this context.
    /// Actions chosen for an earlier rule have run before the next callback.
    /// Rebuilding is deferred, however, so merge functions may still increase
    /// or decrease this count before the iteration finishes. `can_stop`
    /// receives the final count after actions and rebuilding.
    pub fn num_nodes(&self) -> usize {
        *self.num_nodes.get_or_init(|| self.egraph.num_nodes())
    }
}

/// A scheduler decides which matches to be applied for a rule.
///
/// The matches that are not chosen in this iteration will be delayed
/// to the next iteration.
pub trait Scheduler: dyn_clone::DynClone + Send + Sync {
    /// Whether or not the rules can be considered as saturated once no database
    /// changes were made in the current iteration.
    ///
    /// This is only called when the runner is otherwise saturated.
    /// Default implementation just returns `true`.
    fn can_stop(&mut self, ctx: &SchedulerContext<'_>, rules: &[&str], ruleset: &str) -> bool {
        let _ = (ctx, rules, ruleset);
        true
    }

    /// Filter the matches for a rule.
    ///
    /// Chosen matches are applied (actions run, database flushed) before the
    /// next rule's `filter_matches` is called, so `ctx.num_nodes()` reflects
    /// earlier decisions. Queries are still collected together before any
    /// callback.
    ///
    /// Return `true` if the scheduler's next run of the rule should feed
    /// `filter_matches` with a new iteration of matches.
    fn filter_matches(
        &mut self,
        ctx: &SchedulerContext<'_>,
        rule: &str,
        ruleset: &str,
        matches: &mut Matches,
    ) -> bool;
}

dyn_clone::clone_trait_object!(Scheduler);

/// A collection of matches produced by a rule.
/// The user can choose which matches to be fired.
pub struct Matches {
    matches: Vec<Value>,
    chosen: Vec<usize>,
    vars: Vec<ResolvedVar>,
    /// Width of each stored tuple in `matches`. This is `vars.len()` for an
    /// ordinary rule. A rule whose head references no variables would otherwise
    /// have zero-width tuples, making its match count unrecoverable; for those we
    /// collect a single unit marker per match, so the width is 1 while `vars` is
    /// empty.
    tuple_width: usize,
    all_chosen: bool,
}

/// A match is a tuple of values corresponding to the variables in a rule.
/// It allows you to retrieve the value corresponding to a variable in the match.
pub struct Match<'a> {
    values: &'a [Value],
    vars: &'a [ResolvedVar],
}

impl Match<'_> {
    /// Get the value corresponding a variable in this match.
    pub fn get_value(&self, var: &str) -> Value {
        let idx = self.vars.iter().position(|v| v.name == var).unwrap();
        self.values[idx]
    }
}

impl Matches {
    fn new(matches: Vec<Value>, vars: Vec<ResolvedVar>) -> Self {
        // Variable-free rules collect one unit marker per match (see
        // `SchedulerRuleInfo::new`), so each stored tuple is one value wide even
        // though there are no variables.
        let tuple_width = vars.len().max(1);
        assert!(matches.len().is_multiple_of(tuple_width));
        Self {
            matches,
            vars,
            tuple_width,
            chosen: Vec::new(),
            all_chosen: false,
        }
    }

    /// The number of matches in total.
    pub fn match_size(&self) -> usize {
        self.matches.len() / self.tuple_width
    }

    /// The length of a tuple.
    pub fn tuple_len(&self) -> usize {
        self.vars.len()
    }

    /// Get `idx`-th match.
    pub fn get_match(&self, idx: usize) -> Match<'_> {
        Match {
            values: &self.matches[idx * self.tuple_len()..(idx + 1) * self.tuple_len()],
            vars: &self.vars,
        }
    }

    /// Pick the match at `idx` to be fired.
    pub fn choose(&mut self, idx: usize) {
        self.chosen.push(idx);
    }

    /// Pick all matches to be fired.
    ///
    /// This is more efficient than calling `choose` for each match.
    pub fn choose_all(&mut self) {
        self.all_chosen = true;
    }

    /// Apply the chosen matches and return the residual matches.
    fn instantiate(
        mut self,
        state: &mut ExecutionState<'_>,
        table_action: &TableAction,
    ) -> Vec<Value> {
        // Width of the stored tuples (1 for variable-free rules, see `new`) versus
        // the number of variable columns actually written into the `decided` table.
        // For a variable-free rule the stored unit marker is dropped and only the
        // trailing unit is inserted, producing the single `(unit)` row that the
        // action rule fires on.
        let tuple_width = self.tuple_width;
        let var_len = self.vars.len();
        let unit = state.base_values().get(());

        if self.all_chosen {
            for row in self.matches.chunks(tuple_width) {
                table_action.insert(
                    state,
                    row[..var_len].iter().cloned().chain(std::iter::once(unit)),
                );
            }
            vec![]
        } else {
            for idx in self.chosen.iter() {
                let row = &self.matches[idx * tuple_width..(idx + 1) * tuple_width];
                table_action.insert(
                    state,
                    row[..var_len].iter().cloned().chain(std::iter::once(unit)),
                );
            }

            // swap remove the chosen matches
            self.chosen.sort_unstable();
            self.chosen.dedup();
            let mut p = self.match_size();
            for c in self.chosen.into_iter().rev() {
                // It's important to decrement `p` first, because otherwise it might underflow when
                // matches are exhausted.
                p -= 1;
                if c != p {
                    let idx_c = c * tuple_width;
                    let idx_p = p * tuple_width;
                    for i in 0..tuple_width {
                        self.matches.swap(idx_c + i, idx_p + i);
                    }
                }
            }
            self.matches.truncate(p * tuple_width);

            self.matches
        }
    }
}

define_id!(
    pub SchedulerId, u32,
    "A unique identifier for a scheduler in the EGraph."
);

impl EGraph {
    /// Register a new scheduler and return its id.
    pub fn add_scheduler(&mut self, scheduler: Box<dyn Scheduler>) -> SchedulerId {
        self.schedulers.push(SchedulerRecord {
            scheduler,
            rule_info: Default::default(),
        })
    }

    /// Removes a scheduler
    pub fn remove_scheduler(&mut self, scheduler_id: SchedulerId) -> Option<Box<dyn Scheduler>> {
        self.schedulers.take(scheduler_id).map(|r| r.scheduler)
    }

    /// Runs a ruleset for one iteration using the given ruleset
    pub fn step_rules_with_scheduler(
        &mut self,
        scheduler_id: SchedulerId,
        ruleset: &str,
    ) -> Result<RunReport, Error> {
        fn collect_rules<'a>(
            ruleset: &str,
            rulesets: &'a IndexMap<String, Ruleset>,
            ids: &mut Vec<(String, &'a CompiledRule)>,
        ) -> Result<(), Error> {
            let Some(r) = rulesets.get(ruleset) else {
                return Err(Error::BackendError(format!("no such ruleset: {ruleset}")));
            };
            match r {
                Ruleset::Rules(rules) => {
                    for (rule_name, rule) in rules.iter() {
                        ids.push((rule_name.clone(), rule));
                    }
                }
                Ruleset::Combined(sub_rulesets) => {
                    for sub_ruleset in sub_rulesets {
                        collect_rules(sub_ruleset, rulesets, ids)?;
                    }
                }
            }
            Ok(())
        }

        let mut rules = Vec::new();
        let rulesets = std::mem::take(&mut self.rulesets);
        let collected = collect_rules(ruleset, &rulesets, &mut rules);
        // Restore `rulesets` before propagating any error so the EGraph is not
        // left with its rulesets taken out.
        if let Err(e) = collected {
            self.rulesets = rulesets;
            return Err(e);
        }
        let mut schedulers = std::mem::take(&mut self.schedulers);

        // `rulesets` and `schedulers` are now taken out of `self`. The body below
        // has several fallible steps (rule compilation, `run_rules`), so run it in
        // a closure and restore both fields afterward no matter how it exits.
        // Otherwise an early error would leave the EGraph with empty rulesets and
        // schedulers.
        let result = (|| -> Result<RunReport, Error> {
            // Step 1: build all the query/action rules and worklist if have not already
            let record = &mut schedulers[scheduler_id];
            for (id, rule) in rules.iter() {
                if !record.rule_info.contains_key(id) {
                    let info = SchedulerRuleInfo::new(self, rule, id)?;
                    record.rule_info.insert((*id).to_owned(), info);
                }
            }

            // Step 2: run all the queries for one iteration
            let query_rules = rules
                .iter()
                .filter_map(|(rule_id, _rule)| {
                    let rule_info = record.rule_info.get(rule_id).unwrap();

                    if rule_info.should_seek {
                        Some(rule_info.query_rule)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();

            let query_iter_report = self
                .backend
                .run_rules(&query_rules, Some(&self.type_info))
                .map_err(|e| Error::BackendError(e.to_string()))?;

            // Steps 3 and 4: choose and immediately apply each rule's matches.
            // Rebuilding remains deferred until every rule has run.
            let mut action_iteration_report = IterationReport::default();
            let mut any_applied = false;
            for (rule_id, _rule) in rules.iter() {
                let (pending, action_rule) = {
                    let rule_info = record.rule_info.get(rule_id).unwrap();
                    (
                        self.backend.table_size(rule_info.decided) > 0,
                        rule_info.action_rule,
                    )
                };
                let ctx = SchedulerContext {
                    egraph: self,
                    num_nodes: OnceCell::new(),
                };
                let mut inserted = false;
                self.backend
                    .with_execution_state(Some(&self.type_info), |state| {
                        let rule_info = record.rule_info.get_mut(rule_id).unwrap();
                        let matches: Vec<Value> =
                            std::mem::take(rule_info.matches.lock().unwrap().as_mut());
                        let mut matches = Matches::new(matches, rule_info.free_vars.clone());
                        rule_info.should_seek =
                            record
                                .scheduler
                                .filter_matches(&ctx, rule_id, ruleset, &mut matches);
                        inserted = (matches.all_chosen && matches.match_size() > 0)
                            || !matches.chosen.is_empty();
                        let table_action = TableAction::new(&self.backend, rule_info.decided);
                        let residual = matches.instantiate(state, &table_action);
                        // A naive query re-finds all residuals when the scheduler asks
                        // for a fresh query. Otherwise they remain available for the
                        // next callback; seminaive queries only add fresh matches.
                        *rule_info.matches.lock().unwrap() =
                            if rule_info.seminaive || !rule_info.should_seek {
                                residual
                            } else {
                                Vec::new()
                            };
                    });
                if inserted {
                    self.backend.flush_updates_no_rebuild();
                }
                if inserted || pending {
                    any_applied = true;
                    let rule_report = match self
                        .backend
                        .run_rules_no_rebuild(&[action_rule], Some(&self.type_info))
                    {
                        Ok(report) => report,
                        Err(action_error) => {
                            // This or an earlier action may already have made a
                            // union. Restore canonical tables before returning
                            // the error. The failed row remains pending and is
                            // retried on the next step.
                            if let Err(rebuild_error) = self.backend.rebuild_now() {
                                return Err(Error::BackendError(format!(
                                    "{action_error}; rebuilding after the failed action also failed: {rebuild_error}"
                                )));
                            }
                            return Err(Error::BackendError(action_error.to_string()));
                        }
                    };
                    let aggregate = &mut action_iteration_report.rule_set_report;
                    let current = rule_report.rule_set_report;
                    aggregate.changed |= current.changed;
                    aggregate.search_and_apply_time += current.search_and_apply_time;
                    aggregate.merge_time += current.merge_time;
                    for (rule, reports) in current.rule_reports {
                        aggregate
                            .rule_reports
                            .entry(rule)
                            .or_default()
                            .extend(reports);
                    }
                }
            }
            if any_applied {
                action_iteration_report.rebuild_time = self
                    .backend
                    .rebuild_now()
                    .map_err(|e| Error::BackendError(e.to_string()))?;
            }
            let mut action_report = RunReport::singleton(ruleset, action_iteration_report);

            // Step 5: combine the reports
            let mut query_report = RunReport::singleton(ruleset, query_iter_report);

            // query matches don't count
            query_report.updated = false;
            query_report.num_matches_per_rule.clear();
            // Scheduler state should not count as database progress. Instead it
            // determines whether a no-op iteration can be treated as fully stopped.
            action_report.can_stop = !action_report.updated && {
                let rule_ids = rules.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>();
                let ctx = SchedulerContext {
                    egraph: self,
                    num_nodes: OnceCell::new(),
                };
                record.scheduler.can_stop(&ctx, &rule_ids, ruleset)
            };

            query_report.union(action_report);

            Ok(query_report)
        })();

        self.rulesets = rulesets;
        self.schedulers = schedulers;

        result
    }
}

#[derive(Clone)]
pub(crate) struct SchedulerRecord {
    scheduler: Box<dyn Scheduler>,
    rule_info: HashMap<String, SchedulerRuleInfo>,
}

/// To enable scheduling without modifying the backend,
/// we split a rule (rule query action) into a worklist relation
/// two rules (rule query (worklist vars false)) and
/// (rule (worklist vars false) (action ... (delete (worklist vars false))))
#[derive(Clone)]
struct SchedulerRuleInfo {
    matches: Arc<Mutex<Vec<Value>>>,
    should_seek: bool,
    decided: FunctionId,
    query_rule: RuleId,
    action_rule: RuleId,
    free_vars: Vec<ResolvedVar>,
    /// Whether this rule's scheduler query uses delta evaluation.
    seminaive: bool,
}

struct CollectMatches {
    matches: Arc<Mutex<Vec<Value>>>,
}

impl Clone for CollectMatches {
    fn clone(&self) -> Self {
        Self {
            matches: Arc::new(Mutex::new(self.matches.lock().unwrap().clone())),
        }
    }
}

impl CollectMatches {
    fn new(matches: Arc<Mutex<Vec<Value>>>) -> Self {
        Self { matches }
    }
}

impl ExternalFunction for CollectMatches {
    fn invoke(&self, state: &mut core_relations::ExecutionState, args: &[Value]) -> Option<Value> {
        self.matches.lock().unwrap().extend(args.iter().copied());
        Some(state.base_values().get(()))
    }
}

impl SchedulerRuleInfo {
    fn new(
        egraph: &mut EGraph,
        rule: &CompiledRule,
        name: &str,
    ) -> Result<SchedulerRuleInfo, Error> {
        let free_vars = rule
            .core
            .head
            .get_free_vars()
            .into_iter()
            .collect::<Vec<_>>();
        let unit_type = egraph.backend.base_values().get_ty::<()>();
        let unit = egraph.backend.base_values().get(());
        let unit_entry = egraph.backend.base_value_constant(());

        let matches = Arc::new(Mutex::new(Vec::new()));
        let collect_matches = egraph
            .backend
            .register_external_func(Box::new(CollectMatches::new(matches.clone())));
        let schema = free_vars
            .iter()
            .map(|v| v.sort.column_ty(&egraph.backend))
            .chain(std::iter::once(ColumnTy::Base(unit_type)))
            .collect();
        // This table is registered for primitive access, so its name must not
        // shadow a user table in the name-indexed registry.
        let decided_name = egraph.parser.symbol_gen.fresh("scheduler_decided");
        let decided = egraph.backend.add_table(FunctionConfig {
            schema,
            default: DefaultVal::Const(unit),
            merge: MergeFn::AssertEq,
            name: decided_name,
            can_subsume: false,
        });

        // Step 1: rebuild the query with the same evaluation mode and planner
        // options as the ordinary compiled rule.
        let mut qrule = egraph.backend.new_rule(name, rule.seminaive);
        qrule.set_no_decomp(rule.no_decomp);
        let mut qrule_builder = BackendRule::new(
            qrule,
            &egraph.functions,
            &egraph.type_info,
            rule.requires_read_context,
        );
        qrule_builder.query(&rule.core.body, rule.include_subsumed)?;
        let mut entries = free_vars
            .iter()
            .map(|fv| qrule_builder.entry(&GenericAtomTerm::Var(span!(), fv.clone())))
            .collect::<Vec<_>>();
        // A rule whose head references no variables would otherwise collect empty
        // tuples, leaving the scheduler unable to tell whether the query matched and
        // so never applying its actions. Collect a single unit marker per match so
        // the match count is recoverable.
        if entries.is_empty() {
            entries.push(unit_entry.clone());
        }
        let _var = qrule_builder.rb.call_external_func(
            collect_matches,
            &entries,
            ColumnTy::Base(unit_type),
            || "collect_matches".to_string(),
        );
        let qrule_id = qrule_builder.build();

        // Step 2: build the action rule
        let mut arule_builder = BackendRule::new(
            egraph.backend.new_rule(name, false),
            &egraph.functions,
            &egraph.type_info,
            rule.requires_read_context,
        );
        let mut entries = free_vars
            .iter()
            .map(|fv| arule_builder.entry(&GenericAtomTerm::Var(span!(), fv.clone())))
            .collect::<Vec<_>>();
        entries.push(unit_entry);
        arule_builder
            .rb
            .query_table(decided, &entries, None)
            .unwrap();
        arule_builder.actions(&rule.core.head)?;
        // Remove the entry as it's now done
        entries.pop();
        arule_builder.rb.remove(decided, &entries);
        let arule_id = arule_builder.build();

        Ok(SchedulerRuleInfo {
            free_vars,
            query_rule: qrule_id,
            action_rule: arule_id,
            matches,
            decided,
            should_seek: true,
            seminaive: rule.seminaive,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Clone)]
    struct FirstNScheduler {
        n: usize,
    }

    impl Scheduler for FirstNScheduler {
        fn filter_matches(
            &mut self,
            _ctx: &SchedulerContext<'_>,
            _rule: &str,
            _ruleset: &str,
            matches: &mut Matches,
        ) -> bool {
            if matches.match_size() <= self.n {
                matches.choose_all();
            } else {
                for i in 0..self.n {
                    matches.choose(i);
                }
            }
            matches.match_size() < self.n * 2
        }
    }

    #[test]
    fn test_first_n_scheduler() {
        let mut egraph = EGraph::default();
        let scheduler = FirstNScheduler { n: 10 };
        let scheduler_id = egraph.add_scheduler(Box::new(scheduler));
        let input = r#"
        (relation R (i64))
        (R 0)
        (rule ((R x) (< x 100)) ((R (+ x 1))))
        (run-schedule (saturate (run)))

        (ruleset test)
        (relation S (i64))
        (rule ((R x)) ((S x)) :ruleset test :name "test-rule")
        "#;
        egraph.parse_and_run_program(None, input).unwrap();
        assert_eq!(egraph.get_size("R"), 101);
        let mut iter = 0;
        loop {
            let report = egraph
                .step_rules_with_scheduler(scheduler_id, "test")
                .unwrap();
            let table_size = egraph.get_size("S");
            iter += 1;
            assert_eq!(table_size, std::cmp::min(iter * 10, 101));

            // A rule whose matches were all consumed is not run (its action
            // rule is skipped), so it has no entry in the final iteration.
            let expected_matches = if iter <= 10 { 10 } else { 12 - iter };
            let rule_name: Arc<str> = "test-rule".into();
            let expected = if expected_matches == 0 {
                vec![]
            } else {
                vec![(&rule_name, &expected_matches)]
            };
            assert_eq!(
                report.num_matches_per_rule.iter().collect::<Vec<_>>(),
                expected
            );

            // Because of semi-naive, the exact rules that are run are more than just `test-rule`
            assert!(
                report
                    .search_and_apply_time_per_rule
                    .keys()
                    .all(|k| k.starts_with("test-rule"))
            );
            assert_eq!(
                report.merge_time_per_ruleset.keys().collect::<Vec<_>>(),
                [&"test".into()]
            );
            assert_eq!(
                report
                    .search_and_apply_time_per_ruleset
                    .keys()
                    .collect::<Vec<_>>(),
                [&"test".into()]
            );

            if report.can_stop {
                break;
            }
        }

        assert_eq!(iter, 12);
    }

    #[derive(Clone)]
    struct DrainThreeScheduler;

    impl Scheduler for DrainThreeScheduler {
        fn filter_matches(
            &mut self,
            _ctx: &SchedulerContext<'_>,
            _rule: &str,
            _ruleset: &str,
            matches: &mut Matches,
        ) -> bool {
            for i in 0..matches.match_size().min(3) {
                matches.choose(i);
            }
            false
        }
    }

    #[test]
    fn test_naive_scheduler_preserves_residual_matches_without_requerying() {
        let mut egraph = EGraph {
            seminaive: false,
            ..Default::default()
        };
        let scheduler_id = egraph.add_scheduler(Box::new(DrainThreeScheduler));
        let mut input = r#"
            (ruleset test)
            (relation R (i64))
            (relation S (i64))
            (rule ((R x)) ((S x)) :ruleset test :name "copy")
            "#
        .to_owned();
        for i in 0..20 {
            input.push_str(&format!("(R {i})\n"));
        }
        egraph.parse_and_run_program(None, &input).unwrap();

        for expected in [3, 6, 9, 12, 15, 18, 20] {
            egraph
                .step_rules_with_scheduler(scheduler_id, "test")
                .unwrap();
            assert_eq!(egraph.get_size("S"), expected);
        }
    }

    #[test]
    fn test_rule_modes_requery_and_read_earlier_actions() {
        for (case, mut egraph, annotation, requery) in [
            (
                "global naive",
                EGraph {
                    seminaive: false,
                    ..Default::default()
                },
                "",
                true,
            ),
            ("rule-local naive", EGraph::default(), ":naive", true),
            (
                "unsafe seminaive",
                EGraph::default(),
                ":unsafe-seminaive",
                false,
            ),
        ] {
            let scheduler_id = egraph.add_scheduler(Box::new(FirstNScheduler { n: 10 }));
            egraph
                .parse_and_run_program(
                    None,
                    &format!(
                        r#"
                        (ruleset test)
                        (relation trigger ())
                        (function f () i64 :merge new)
                        (function g () i64 :merge new)
                        (set (f) 0)
                        (trigger)
                        (rule ((trigger)) ((set (f) (+ (f) 1))) {annotation}
                              :ruleset test :name "a-write")
                        (rule ((trigger)) ((set (g) (f))) {annotation}
                              :ruleset test :name "b-read")
                        "#
                    ),
                )
                .unwrap();

            egraph
                .step_rules_with_scheduler(scheduler_id, "test")
                .unwrap();
            egraph
                .parse_and_run_program(None, "(check (= (f) 1)) (check (= (g) 1))")
                .unwrap_or_else(|error| panic!("{case}: {error}"));

            if requery {
                // A naive query must search the whole database again.
                egraph
                    .step_rules_with_scheduler(scheduler_id, "test")
                    .unwrap();
                egraph
                    .parse_and_run_program(None, "(check (= (f) 2)) (check (= (g) 2))")
                    .unwrap_or_else(|error| panic!("{case}: {error}"));
            }
        }
    }

    #[test]
    fn test_scheduler_preserves_include_subsumed_mode() {
        let mut egraph = EGraph::default();
        let scheduler_id = egraph.add_scheduler(Box::new(FirstNScheduler { n: 10 }));
        let input = r#"
        (ruleset analysis)
        (ruleset test)
        (datatype Math
          (Add Math Math)
          (Mul Math Math)
          (Num i64))
        (relation Hit (i64))
        (relation IncludedHit (i64))
        (let expr (Add (Mul (Num 0) (Num 1)) (Num 2)))
        (rewrite (Mul (Num 0) x) (Num 0) :subsume :ruleset analysis)
        (rewrite (Add (Num 0) x) x :subsume :ruleset analysis)
        (rule ((= e (Add (Mul (Num a) x) (Num b)))) ((Hit a))
              :ruleset test :name "visible-only")
        (rule ((= e (Add (Mul (Num a) x) (Num b)))) ((IncludedHit a))
              :ruleset test :name "including-subsumed"
              :internal-include-subsumed)
        (run-schedule (saturate (run analysis)))
        "#;
        egraph.parse_and_run_program(None, input).unwrap();

        egraph
            .step_rules_with_scheduler(scheduler_id, "test")
            .unwrap();

        assert_eq!(egraph.get_size("Hit"), 0);
        assert_eq!(egraph.get_size("IncludedHit"), 1);
    }

    #[derive(Clone, Default)]
    struct DelayStopScheduler {
        can_stop_calls: usize,
    }

    impl Scheduler for DelayStopScheduler {
        fn can_stop(
            &mut self,
            _ctx: &SchedulerContext<'_>,
            _rules: &[&str],
            _ruleset: &str,
        ) -> bool {
            self.can_stop_calls += 1;
            self.can_stop_calls > 1
        }

        fn filter_matches(
            &mut self,
            _ctx: &SchedulerContext<'_>,
            _rule: &str,
            _ruleset: &str,
            _matches: &mut Matches,
        ) -> bool {
            false
        }
    }

    #[test]
    fn test_scheduler_progress_is_separate_from_database_progress() {
        let mut egraph = EGraph::default();
        let scheduler_id = egraph.add_scheduler(Box::new(DelayStopScheduler::default()));
        let input = r#"
        (ruleset test)
        (relation R (i64))
        (rule ((R x)) ((R x)) :ruleset test :name "noop")
        (R 1)
        (R 2)
        (R 3)
        (R 4)
        "#;
        egraph.parse_and_run_program(None, input).unwrap();

        let before = egraph.get_size("R");
        let report = egraph
            .step_rules_with_scheduler(scheduler_id, "test")
            .unwrap();
        let after = egraph.get_size("R");

        assert_eq!(before, after);
        assert!(!report.updated);
        assert!(!report.can_stop);
    }

    #[test]
    fn test_step_rules_with_scheduler_unknown_ruleset() {
        let mut egraph = EGraph::default();
        let scheduler_id = egraph.add_scheduler(Box::new(DelayStopScheduler::default()));
        let err = egraph
            .step_rules_with_scheduler(scheduler_id, "does-not-exist")
            .unwrap_err();
        assert!(matches!(err, Error::BackendError(_)));
    }

    /// A scheduler that only inspects `match_size` and never chooses anything.
    #[derive(Clone)]
    struct InspectSizeScheduler;

    impl Scheduler for InspectSizeScheduler {
        fn filter_matches(
            &mut self,
            _ctx: &SchedulerContext<'_>,
            _rule: &str,
            _ruleset: &str,
            matches: &mut Matches,
        ) -> bool {
            // Calling `match_size` on a rule with no free variables used to panic
            // with a divide-by-zero. Just exercise it and stop.
            let _ = matches.match_size();
            false
        }
    }

    #[test]
    fn test_match_size_with_no_free_vars() {
        let mut egraph = EGraph::default();
        let scheduler_id = egraph.add_scheduler(Box::new(InspectSizeScheduler));
        // The action `(R 1)` references no variables, so the rule has no free vars.
        let input = r#"
        (ruleset test)
        (relation R (i64))
        (rule ((R x)) ((R 1)) :ruleset test :name "no-vars")
        (R 0)
        "#;
        egraph.parse_and_run_program(None, input).unwrap();
        egraph
            .step_rules_with_scheduler(scheduler_id, "test")
            .unwrap();
    }

    /// A scheduler that stops choosing matches once the e-graph holds
    /// `limit` e-nodes.
    #[derive(Clone)]
    struct NodeLimitScheduler {
        limit: usize,
    }

    impl Scheduler for NodeLimitScheduler {
        fn can_stop(
            &mut self,
            ctx: &SchedulerContext<'_>,
            _rules: &[&str],
            _ruleset: &str,
        ) -> bool {
            ctx.num_nodes() >= self.limit
        }

        fn filter_matches(
            &mut self,
            ctx: &SchedulerContext<'_>,
            _rule: &str,
            _ruleset: &str,
            matches: &mut Matches,
        ) -> bool {
            if ctx.num_nodes() < self.limit {
                matches.choose_all();
                true
            } else {
                false
            }
        }
    }

    #[test]
    fn test_scheduler_sees_egraph_size() {
        let mut egraph = EGraph::default();
        let scheduler_id = egraph.add_scheduler(Box::new(NodeLimitScheduler { limit: 10 }));
        // Each firing adds one `Num` e-node; `depth` rows (base-sort output)
        // and `seen` rows (a relation, i.e. a constructor over a fresh
        // non-unionable sort) are analysis data and must not count towards
        // `num_nodes`.
        let input = r#"
        (ruleset grow)
        (datatype Math (Num i64))
        (function depth (Math) i64 :no-merge)
        (relation seen (Math))
        (Num 0)
        (rule ((= e (Num i)) (< i 100))
              ((Num (+ i 1)) (set (depth e) i) (seen e))
              :ruleset grow :name "grow")
        "#;
        egraph.parse_and_run_program(None, input).unwrap();
        for _ in 0..20 {
            let report = egraph
                .step_rules_with_scheduler(scheduler_id, "grow")
                .unwrap();
            if report.can_stop {
                break;
            }
        }
        // One match per iteration, so the scheduler stops exactly at the limit.
        assert_eq!(egraph.get_size("Num"), 10);
        assert_eq!(egraph.num_nodes(), 10);
        // `depth` and `seen` rows exist but do not count as e-nodes.
        assert!(egraph.get_size("depth") > 0);
        assert!(egraph.get_size("seen") > 0);
    }

    #[test]
    fn test_num_nodes_uses_constructor_semantics_in_all_encodings() {
        for (mode, mut egraph) in [
            ("normal", EGraph::default()),
            ("term", EGraph::new_with_term_encoding()),
            ("proof", EGraph::new_with_proofs()),
        ] {
            egraph
                .parse_and_run_program(
                    None,
                    r#"
                    (datatype E (C))
                    (function analysis () E :merge old)
                    (relation R ())
                    (constructor Hidden () E :internal-hidden)
                    (C)
                    (C)
                    (set (analysis) (C))
                    (let alias (C))
                    (R)
                    (Hidden)
                    "#,
                )
                .unwrap();
            assert_eq!(egraph.get_size("C"), 1, "{mode}");
            assert_eq!(egraph.num_nodes(), 1, "{mode}");
        }
    }

    /// A scheduler that records the e-graph size it observes at each
    /// `filter_matches` call.
    #[derive(Clone)]
    struct SizeProbeScheduler {
        sizes_seen: Arc<Mutex<Vec<(String, usize)>>>,
    }

    impl Scheduler for SizeProbeScheduler {
        fn filter_matches(
            &mut self,
            ctx: &SchedulerContext<'_>,
            rule: &str,
            _ruleset: &str,
            matches: &mut Matches,
        ) -> bool {
            self.sizes_seen
                .lock()
                .unwrap()
                .push((rule.to_owned(), ctx.num_nodes()));
            matches.choose_all();
            true
        }
    }

    #[test]
    fn test_filter_matches_sees_fresh_sizes() {
        let mut egraph = EGraph::default();
        let sizes_seen = Arc::new(Mutex::new(Vec::new()));
        let scheduler_id = egraph.add_scheduler(Box::new(SizeProbeScheduler {
            sizes_seen: sizes_seen.clone(),
        }));
        // "a-grow" adds one Num per iteration; "b-watch" only matches. Since
        // each rule's matches are applied before the next rule is consulted,
        // "b-watch" must observe "a-grow"'s new node within the same iteration.
        let input = r#"
        (ruleset t)
        (datatype Math (Num i64))
        (Num 0)
        (rule ((= e (Num i)) (< i 10)) ((union e (Num (+ i 1)))) :ruleset t :name "a-grow")
        (rule ((Num i)) ((Num i)) :ruleset t :name "b-watch")
        "#;
        egraph.parse_and_run_program(None, input).unwrap();
        let report = egraph.step_rules_with_scheduler(scheduler_id, "t").unwrap();

        assert_eq!(
            *sizes_seen.lock().unwrap(),
            vec![("a-grow".to_owned(), 1), ("b-watch".to_owned(), 2)]
        );
        assert_eq!(egraph.get_size("Num"), 2);
        assert_eq!(
            report.iterations.len(),
            2,
            "one scheduler step has one query report and one action report"
        );
        assert_eq!(report.iterations[1].rule_reports().len(), 2);
    }

    /// A scheduler that fires every match.
    #[derive(Clone)]
    struct ChooseAllScheduler;

    impl Scheduler for ChooseAllScheduler {
        fn filter_matches(
            &mut self,
            _ctx: &SchedulerContext<'_>,
            _rule: &str,
            _ruleset: &str,
            matches: &mut Matches,
        ) -> bool {
            matches.choose_all();
            false
        }
    }

    #[test]
    fn test_failed_action_is_rebuilt_and_retried() {
        let mut egraph = EGraph::default();
        let scheduler_id = egraph.add_scheduler(Box::new(ChooseAllScheduler));
        egraph
            .parse_and_run_program(
                None,
                r#"
                (ruleset test)
                (datatype E (A) (B))
                (function P (E) i64 :merge old)
                (set (P (A)) 0)
                (set (P (B)) 0)
                (rule ((= a (A)) (= b (B)))
                      ((union a b) (panic "boom"))
                      :ruleset test :name "union-then-panic")
                "#,
            )
            .unwrap();

        assert!(
            egraph
                .step_rules_with_scheduler(scheduler_id, "test")
                .is_err()
        );
        assert_eq!(egraph.get_size("P"), 1, "the failed step must rebuild");
        assert!(
            egraph
                .step_rules_with_scheduler(scheduler_id, "test")
                .is_err(),
            "the pending failed action must not be silently dropped"
        );
    }

    #[test]
    fn test_scheduler_internal_table_does_not_shadow_a_constructor() {
        let mut egraph = EGraph::default();
        let scheduler_id = egraph.add_scheduler(Box::new(ChooseAllScheduler));
        egraph
            .parse_and_run_program(
                None,
                r#"
                (ruleset test)
                (datatype E (backend))
                (relation Seen (E))
                (let root (backend))
                (rule ((= x (backend))) ((Seen x))
                      :ruleset test :name "see-backend")
                "#,
            )
            .unwrap();

        egraph
            .step_rules_with_scheduler(scheduler_id, "test")
            .unwrap();
        assert_eq!(egraph.num_nodes(), 1);
    }

    #[test]
    fn test_no_free_vars_rule_applies_actions() {
        let mut egraph = EGraph::default();
        let scheduler_id = egraph.add_scheduler(Box::new(ChooseAllScheduler));
        // The action `(S)` references no variables, so the rule has no free vars.
        // The scheduler must still apply it when the query matches.
        let input = r#"
        (ruleset test)
        (relation R (i64))
        (relation S ())
        (rule ((R x)) ((S)) :ruleset test :name "no-vars")
        (R 0)
        "#;
        egraph.parse_and_run_program(None, input).unwrap();
        assert_eq!(egraph.get_size("S"), 0);
        egraph
            .step_rules_with_scheduler(scheduler_id, "test")
            .unwrap();
        assert_eq!(egraph.get_size("S"), 1);
    }
}
