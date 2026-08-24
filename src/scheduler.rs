use std::sync::Arc;
use std::sync::Mutex;

use core_relations::{ExecutionState, ExternalFunction, Value};
use egglog_bridge::{
    ColumnTy, DefaultVal, FunctionConfig, FunctionId, MergeFn, RuleId, TableAction,
};
use egglog_reports::RunReport;
use numeric_id::define_id;

use crate::{ast::ResolvedVar, core::GenericAtomTerm, core::ResolvedCoreRule, util::IndexMap, *};

/// Read-only information about the e-graph handed to a scheduler on each
/// callback.
pub struct SchedulerContext<'a> {
    /// The e-graph the rules run on, for size queries such as
    /// [`EGraph::get_size`], [`EGraph::total_size`], and [`EGraph::num_nodes`].
    ///
    /// Sizes only change between iterations: matches chosen in the current
    /// iteration are staged and not yet visible in the database.
    pub egraph: &'a EGraph,
    /// 0-based count of iterations this scheduler has completed on this
    /// ruleset. Schedulers can use a change in this value to detect an
    /// iteration boundary (e.g., to re-read sizes once per iteration).
    pub iteration: usize,
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
    fn can_stop(&mut self, ctx: &SchedulerContext, rules: &[&str], ruleset: &str) -> bool {
        let _ = (ctx, rules, ruleset);
        true
    }

    /// Filter the matches for a rule.
    ///
    /// Return `true` if the scheduler's next run of the rule should feed
    /// `filter_matches` with a new iteration of matches.
    fn filter_matches(
        &mut self,
        ctx: &SchedulerContext,
        rule: &str,
        ruleset: &str,
        matches: &mut Matches,
    ) -> bool;

    /// Whether each rule's chosen matches should be applied (actions run,
    /// database flushed) before the next rule's `filter_matches` is called,
    /// instead of batching all rules' chosen matches into a single apply at
    /// the end of the iteration.
    ///
    /// With immediate application, the e-graph sizes observed through the
    /// [`SchedulerContext`] are up to date within an iteration, at the cost
    /// of one flush and one action-rule run per rule that chose matches.
    /// Default is `false` (batched).
    fn apply_immediately(&self) -> bool {
        false
    }
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

    /// Whether at least one match has been chosen to be fired.
    pub fn any_chosen(&self) -> bool {
        (self.all_chosen && self.match_size() > 0) || !self.chosen.is_empty()
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
            iterations: 0,
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
            ids: &mut Vec<(String, &'a ResolvedCoreRule)>,
        ) -> Result<(), Error> {
            let Some(r) = rulesets.get(ruleset) else {
                return Err(Error::BackendError(format!("no such ruleset: {ruleset}")));
            };
            match r {
                Ruleset::Rules(rules) => {
                    for (rule_name, (core_rule, _)) in rules.iter() {
                        ids.push((rule_name.clone(), core_rule));
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
            let iteration = record.iterations;
            record.iterations += 1;
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

            // Steps 3 and 4: let the scheduler decide which matches need to be
            // kept, and run the action rules on the chosen ones.
            let mut action_report = if record.scheduler.apply_immediately() {
                // Immediate mode: apply each rule's chosen matches (flush +
                // action rule) before the next rule's `filter_matches`, so the
                // scheduler observes up-to-date e-graph sizes within the
                // iteration.
                let mut action_report = RunReport::default();
                for (rule_id, _rule) in rules.iter() {
                    let mut inserted = false;
                    let ctx = SchedulerContext {
                        egraph: self,
                        iteration,
                    };
                    self.backend
                        .with_execution_state(Some(&self.type_info), |state| {
                            let rule_info = record.rule_info.get_mut(rule_id).unwrap();

                            let matches: Vec<Value> =
                                std::mem::take(rule_info.matches.lock().unwrap().as_mut());
                            let mut matches = Matches::new(matches, rule_info.free_vars.clone());
                            rule_info.should_seek = record.scheduler.filter_matches(
                                &ctx,
                                rule_id,
                                ruleset,
                                &mut matches,
                            );
                            inserted = matches.any_chosen();
                            let table_action = TableAction::new(&self.backend, rule_info.decided);
                            *rule_info.matches.lock().unwrap() =
                                matches.instantiate(state, &table_action);
                        });
                    if inserted {
                        self.backend.flush_updates();
                        let action_rule = record.rule_info.get(rule_id).unwrap().action_rule;
                        let rule_report = self
                            .backend
                            .run_rules(&[action_rule], Some(&self.type_info))
                            .map_err(|e| Error::BackendError(e.to_string()))?;
                        action_report.union(RunReport::singleton(ruleset, rule_report));
                    }
                }
                action_report
            } else {
                // Batched mode: decide for every rule first, then apply all
                // chosen matches in a single flush and action-rule run.
                let ctx = SchedulerContext {
                    egraph: self,
                    iteration,
                };
                self.backend
                    .with_execution_state(Some(&self.type_info), |state| {
                        for (rule_id, _rule) in rules.iter() {
                            let rule_info = record.rule_info.get_mut(rule_id).unwrap();

                            let matches: Vec<Value> =
                                std::mem::take(rule_info.matches.lock().unwrap().as_mut());
                            let mut matches = Matches::new(matches, rule_info.free_vars.clone());
                            rule_info.should_seek = record.scheduler.filter_matches(
                                &ctx,
                                rule_id,
                                ruleset,
                                &mut matches,
                            );
                            let table_action = TableAction::new(&self.backend, rule_info.decided);
                            *rule_info.matches.lock().unwrap() =
                                matches.instantiate(state, &table_action);
                        }
                    });
                self.backend.flush_updates();

                let action_rules = rules
                    .iter()
                    .map(|(rule_id, _rule)| {
                        let rule_info = record.rule_info.get(rule_id).unwrap();
                        rule_info.action_rule
                    })
                    .collect::<Vec<_>>();
                let action_iter_report = self
                    .backend
                    .run_rules(&action_rules, Some(&self.type_info))
                    .map_err(|e| Error::BackendError(e.to_string()))?;
                RunReport::singleton(ruleset, action_iter_report)
            };

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
                    iteration,
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
    /// Number of iterations this scheduler has run, across all rulesets.
    iterations: usize,
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
        rule: &ResolvedCoreRule,
        name: &str,
    ) -> Result<SchedulerRuleInfo, Error> {
        let free_vars = rule.head.get_free_vars().into_iter().collect::<Vec<_>>();
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
        let decided = egraph.backend.add_table(FunctionConfig {
            schema,
            default: DefaultVal::Const(unit),
            merge: MergeFn::AssertEq,
            name: "backend".to_string(),
            can_subsume: false,
        });

        // Step 1: build the query rule
        let mut qrule_builder = BackendRule::new(
            egraph.backend.new_rule(name, true),
            &egraph.functions,
            &egraph.type_info,
            false, // seminaive query: Pure/Write contexts
        );
        qrule_builder.query(&rule.body, false)?;
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
            true, // action rule reads the DB: Read/Full contexts
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
        arule_builder.actions(&rule.head)?;
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
            _ctx: &SchedulerContext,
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

            let expected_matches = if iter <= 10 { 10 } else { 12 - iter };
            assert_eq!(
                report.num_matches_per_rule.iter().collect::<Vec<_>>(),
                [(&"test-rule".into(), &expected_matches)]
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

    #[test]
    fn test_scheduler_does_not_apply_fresh_subsumed_matches() {
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
        (let expr (Add (Mul (Num 0) (Num 1)) (Num 2)))
        (rewrite (Mul (Num 0) x) (Num 0) :subsume :ruleset analysis)
        (rewrite (Add (Num 0) x) x :subsume :ruleset analysis)
        (rule ((= e (Add (Mul (Num a) x) (Num b)))) ((Hit a)) :ruleset test :name "hit-subsumed-affine")
        (run-schedule (saturate (run analysis)))
        "#;
        egraph.parse_and_run_program(None, input).unwrap();

        let report = egraph
            .step_rules_with_scheduler(scheduler_id, "test")
            .unwrap();

        assert_eq!(egraph.get_size("Hit"), 0);
        assert!(
            !report.updated,
            "subsumed rows should not be collected as fresh scheduler matches"
        );
    }

    #[derive(Clone, Default)]
    struct DelayStopScheduler {
        can_stop_calls: usize,
    }

    impl Scheduler for DelayStopScheduler {
        fn can_stop(&mut self, _ctx: &SchedulerContext, _rules: &[&str], _ruleset: &str) -> bool {
            self.can_stop_calls += 1;
            self.can_stop_calls > 1
        }

        fn filter_matches(
            &mut self,
            _ctx: &SchedulerContext,
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
            _ctx: &SchedulerContext,
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
    /// `limit` e-nodes, and checks the iteration counter it is handed.
    #[derive(Clone)]
    struct NodeLimitScheduler {
        limit: usize,
        expected_iteration: usize,
    }

    impl Scheduler for NodeLimitScheduler {
        fn can_stop(&mut self, ctx: &SchedulerContext, _rules: &[&str], _ruleset: &str) -> bool {
            ctx.egraph.num_nodes() >= self.limit
        }

        fn filter_matches(
            &mut self,
            ctx: &SchedulerContext,
            _rule: &str,
            _ruleset: &str,
            matches: &mut Matches,
        ) -> bool {
            assert_eq!(ctx.iteration, self.expected_iteration);
            self.expected_iteration += 1;
            if ctx.egraph.num_nodes() < self.limit {
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
        let scheduler_id = egraph.add_scheduler(Box::new(NodeLimitScheduler {
            limit: 10,
            expected_iteration: 0,
        }));
        // Each firing adds one `Num` e-node; `depth` rows are analysis data
        // (base-sort output) and must not count towards `num_nodes`.
        let input = r#"
        (ruleset grow)
        (datatype Math (Num i64))
        (function depth (Math) i64 :no-merge)
        (Num 0)
        (rule ((= e (Num i)) (< i 100))
              ((Num (+ i 1)) (set (depth e) i))
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
        // `depth` rows exist but only count towards `total_size`.
        assert!(egraph.get_size("depth") > 0);
        assert_eq!(
            egraph.total_size(),
            egraph.num_nodes() + egraph.get_size("depth")
        );
    }

    /// A scheduler with `apply_immediately` that records the e-graph size it
    /// observes at each `filter_matches` call.
    #[derive(Clone)]
    struct EagerProbeScheduler {
        sizes_seen: Arc<Mutex<Vec<(String, usize)>>>,
    }

    impl Scheduler for EagerProbeScheduler {
        fn apply_immediately(&self) -> bool {
            true
        }

        fn filter_matches(
            &mut self,
            ctx: &SchedulerContext,
            rule: &str,
            _ruleset: &str,
            matches: &mut Matches,
        ) -> bool {
            self.sizes_seen
                .lock()
                .unwrap()
                .push((rule.to_owned(), ctx.egraph.num_nodes()));
            matches.choose_all();
            true
        }
    }

    #[test]
    fn test_apply_immediately_sees_fresh_sizes() {
        let mut egraph = EGraph::default();
        let sizes_seen = Arc::new(Mutex::new(Vec::new()));
        let scheduler_id = egraph.add_scheduler(Box::new(EagerProbeScheduler {
            sizes_seen: sizes_seen.clone(),
        }));
        // "a-grow" adds one Num per iteration; "b-watch" only matches. With
        // immediate application, "b-watch" must observe "a-grow"'s new node
        // within the same iteration.
        let input = r#"
        (ruleset t)
        (datatype Math (Num i64))
        (Num 0)
        (rule ((= e (Num i)) (< i 10)) ((Num (+ i 1))) :ruleset t :name "a-grow")
        (rule ((Num i)) ((Num i)) :ruleset t :name "b-watch")
        "#;
        egraph.parse_and_run_program(None, input).unwrap();
        egraph.step_rules_with_scheduler(scheduler_id, "t").unwrap();

        assert_eq!(
            *sizes_seen.lock().unwrap(),
            vec![("a-grow".to_owned(), 1), ("b-watch".to_owned(), 2)]
        );
        assert_eq!(egraph.get_size("Num"), 2);
    }

    /// A scheduler that fires every match.
    #[derive(Clone)]
    struct ChooseAllScheduler;

    impl Scheduler for ChooseAllScheduler {
        fn filter_matches(
            &mut self,
            _ctx: &SchedulerContext,
            _rule: &str,
            _ruleset: &str,
            matches: &mut Matches,
        ) -> bool {
            matches.choose_all();
            false
        }
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
