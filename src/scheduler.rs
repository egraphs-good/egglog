use std::sync::Arc;
use std::sync::Mutex;

use core_relations::{ExecutionState, ExternalFunction, Value};
use egglog_bridge::{
    ColumnTy, DefaultVal, FunctionConfig, FunctionId, MergeFn, RuleId, TableAction,
};
use egglog_reports::RunReport;
use numeric_id::define_id;

use crate::{ast::ResolvedVar, core::GenericAtomTerm, core::ResolvedCoreRule, util::IndexMap, *};

/// A scheduler decides which matches to be applied for a rule.
///
/// The matches that are not chosen in this iteration will be delayed
/// to the next iteration.
pub trait Scheduler: dyn_clone::DynClone + Send + Sync {
    /// Whether or not the rules can be considered as saturated (i.e.,
    /// `run_report.updated == false`).
    ///
    /// This is only called when the runner is otherwise saturated.
    /// Default implementation just returns `true`.
    fn can_stop(&mut self, rules: &[&str], ruleset: &str) -> bool {
        let _ = (rules, ruleset);
        true
    }

    /// Filter the matches for a rule.
    ///
    /// Return `true` if the scheduler's next run of the rule should feed
    /// `filter_matches` with a new iteration of matches.
    fn filter_matches(&mut self, rule: &str, ruleset: &str, matches: &mut Matches) -> bool;
}

dyn_clone::clone_trait_object!(Scheduler);

/// A collection of matches produced by a rule.
/// The user can choose which matches to be fired.
pub struct Matches {
    matches: Vec<Value>,
    chosen: Vec<usize>,
    vars: Vec<ResolvedVar>,
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
        let total_len = matches.len();
        let tuple_len = vars.len();
        assert!(total_len.is_multiple_of(tuple_len));
        Self {
            matches,
            vars,
            chosen: Vec::new(),
            all_chosen: false,
        }
    }

    /// The number of matches in total.
    pub fn match_size(&self) -> usize {
        self.matches.len() / self.vars.len()
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
        let tuple_len = self.tuple_len();
        let unit = state.base_values().get(());

        if self.all_chosen {
            for row in self.matches.chunks(tuple_len) {
                table_action.insert(state, row.iter().cloned().chain(std::iter::once(unit)));
            }
            vec![]
        } else {
            for idx in self.chosen.iter() {
                let row = &self.matches[idx * tuple_len..(idx + 1) * tuple_len];
                table_action.insert(state, row.iter().cloned().chain(std::iter::once(unit)));
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
                    let idx_c = c * tuple_len;
                    let idx_p = p * tuple_len;
                    for i in 0..tuple_len {
                        self.matches.swap(idx_c + i, idx_p + i);
                    }
                }
            }
            self.matches.truncate(p * tuple_len);

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
            ids: &mut Vec<(String, &'a ResolvedCoreRule)>,
        ) {
            match &rulesets[ruleset] {
                Ruleset::Rules(rules) => {
                    for (rule_name, (core_rule, _)) in rules.iter() {
                        ids.push((rule_name.clone(), core_rule));
                    }
                }
                Ruleset::Combined(sub_rulesets) => {
                    for sub_ruleset in sub_rulesets {
                        collect_rules(sub_ruleset, rulesets, ids);
                    }
                }
            }
        }

        let mut rules = Vec::new();
        let rulesets = std::mem::take(&mut self.rulesets);
        collect_rules(ruleset, &rulesets, &mut rules);
        let mut schedulers = std::mem::take(&mut self.schedulers);

        // Step 1: build all the query/action rules and worklist if have not already
        let record = &mut schedulers[scheduler_id];
        rules.iter().for_each(|(id, rule)| {
            record
                .rule_info
                .entry((*id).to_owned())
                .or_insert_with(|| SchedulerRuleInfo::new(self, rule, id));
        });

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
            .run_rules(&query_rules)
            .map_err(|e| Error::BackendError(e.to_string()))?;

        // Step 3: let the scheduler decide which matches need to be kept
        self.backend.with_execution_state(|state| {
            for (rule_id, _rule) in rules.iter() {
                let rule_info = record.rule_info.get_mut(rule_id).unwrap();

                let matches: Vec<Value> =
                    std::mem::take(rule_info.matches.lock().unwrap().as_mut());
                let mut matches = Matches::new(matches, rule_info.free_vars.clone());
                rule_info.should_seek =
                    record
                        .scheduler
                        .filter_matches(rule_id, ruleset, &mut matches);
                let table_action = TableAction::new(&self.backend, rule_info.decided);
                *rule_info.matches.lock().unwrap() = matches.instantiate(state, &table_action);
            }
        });
        self.backend.flush_updates();

        // Step 4: recheck the chosen keys and run the action rules
        let validation_rules = rules
            .iter()
            .map(|(rule_id, _rule)| {
                let rule_info = record.rule_info.get(rule_id).unwrap();
                rule_info.validation_rule
            })
            .collect::<Vec<_>>();
        let validation_iter_report = self
            .backend
            .run_rules(&validation_rules)
            .map_err(|e| Error::BackendError(e.to_string()))?;
        let action_rules = rules
            .iter()
            .map(|(rule_id, _rule)| {
                let rule_info = record.rule_info.get(rule_id).unwrap();
                rule_info.action_rule
            })
            .collect::<Vec<_>>();
        let action_iter_report = self
            .backend
            .run_rules(&action_rules)
            .map_err(|e| Error::BackendError(e.to_string()))?;
        let cleanup_rules = rules
            .iter()
            .map(|(rule_id, _rule)| {
                let rule_info = record.rule_info.get(rule_id).unwrap();
                rule_info.cleanup_rule
            })
            .collect::<Vec<_>>();
        let cleanup_iter_report = self
            .backend
            .run_rules(&cleanup_rules)
            .map_err(|e| Error::BackendError(e.to_string()))?;

        // Step 5: combine the reports
        let mut query_report = RunReport::singleton(ruleset, query_iter_report);
        let mut validation_report = RunReport::singleton(ruleset, validation_iter_report);
        let mut action_report = RunReport::singleton(ruleset, action_iter_report);
        let mut cleanup_report = RunReport::singleton(ruleset, cleanup_iter_report);

        // query matches don't count
        query_report.updated = false;
        query_report.num_matches_per_rule.clear();
        // validation and cleanup only touch scheduler-internal tables
        validation_report.updated = false;
        validation_report.num_matches_per_rule.clear();
        cleanup_report.updated = false;
        cleanup_report.num_matches_per_rule.clear();
        // if the scheduler says it shouldn't stop, then it's considered updated (unsaturated)
        action_report.updated = action_report.updated || {
            let rule_ids = rules.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>();
            !record.scheduler.can_stop(&rule_ids, ruleset)
        };

        query_report.union(validation_report);
        query_report.union(action_report);
        query_report.union(cleanup_report);

        self.rulesets = rulesets;
        self.schedulers = schedulers;

        Ok(query_report)
    }
}

#[derive(Clone)]
pub(crate) struct SchedulerRecord {
    scheduler: Box<dyn Scheduler>,
    rule_info: HashMap<String, SchedulerRuleInfo>,
}

/// To enable scheduling without modifying the backend, we split a rule into
/// scheduler-internal worklist and validation relations plus four rules:
/// (rule query (worklist vars false)),
/// (rule (worklist vars false) query (validated vars false)),
/// (rule (validated vars false) (action ...)), and
/// (rule (worklist vars false) (delete ...)).
///
/// Scheduler keys are the variables used by the rule actions. Body-only
/// variables are only witnesses that the key is currently valid. For example,
/// `(rule ((A x y)) ((Hit x)))` can have `A(1, 10)` and `A(1, 20)`, but the
/// scheduler key is only `x = 1`. Validation therefore writes one keyed
/// `validated(x)` row before the action rule runs, instead of letting every
/// body-only `y` witness run the same `Hit(1)` action.
///
/// Validation also protects held matches. If a scheduler delays a key and a
/// later rebuild, subsumption, or deletion makes the original body false, the
/// validation rule will not produce a `validated` row. The cleanup rule still
/// removes the stale worklist key so the scheduler does not keep carrying it.
#[derive(Clone)]
struct SchedulerRuleInfo {
    matches: Arc<Mutex<Vec<Value>>>,
    should_seek: bool,
    decided: FunctionId,
    query_rule: RuleId,
    validation_rule: RuleId,
    action_rule: RuleId,
    cleanup_rule: RuleId,
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
    fn new(egraph: &mut EGraph, rule: &ResolvedCoreRule, name: &str) -> SchedulerRuleInfo {
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
            .collect::<Vec<_>>();
        let decided = egraph.backend.add_table(FunctionConfig {
            schema: schema.clone(),
            default: DefaultVal::Const(unit),
            merge: MergeFn::AssertEq,
            name: "backend".to_string(),
            can_subsume: false,
        });
        let validated = egraph.backend.add_table(FunctionConfig {
            schema,
            default: DefaultVal::Const(unit),
            merge: MergeFn::AssertEq,
            name: "backend".to_string(),
            can_subsume: false,
        });

        // Step 1: build the query rule. Subsumed rows should not be offered as
        // fresh scheduler matches; they are no longer valid body witnesses.
        let mut qrule_builder = BackendRule::new(
            egraph.backend.new_rule(name, true),
            &egraph.functions,
            &egraph.type_info,
            true, // seminaive rule context
        );
        qrule_builder.query(&rule.body, false);
        let entries = free_vars
            .iter()
            .map(|fv| qrule_builder.entry(&GenericAtomTerm::Var(span!(), fv.clone())))
            .collect::<Vec<_>>();
        let _var = qrule_builder.rb.call_external_func(
            collect_matches,
            &entries,
            ColumnTy::Base(unit_type),
            || "collect_matches".to_string(),
        );
        let qrule_id = qrule_builder.build();

        // Step 2: build the validation rule. This rechecks that the original
        // body still has some witness for the chosen scheduler key, but writes
        // only one keyed row even if body-only variables have multiple matches.
        // For `A(1, 10)` and `A(1, 20)`, a chosen `x = 1` key validates once.
        let mut validation_rule_builder = BackendRule::new(
            egraph.backend.new_rule(name, false),
            &egraph.functions,
            &egraph.type_info,
            true, // seminaive rule context for rechecking the original body
        );
        let mut entries = free_vars
            .iter()
            .map(|fv| validation_rule_builder.entry(&GenericAtomTerm::Var(span!(), fv.clone())))
            .collect::<Vec<_>>();
        entries.push(unit_entry.clone());
        validation_rule_builder
            .rb
            .query_table(decided, &entries, None)
            .unwrap();
        validation_rule_builder.query(&rule.body, false);
        validation_rule_builder.rb.set(validated, &entries);
        let validation_rule = validation_rule_builder.build();

        // Step 3: build the action rule. It reads only validated scheduler keys,
        // so user actions run once per chosen key and never for stale keys.
        let mut arule_builder = BackendRule::new(
            egraph.backend.new_rule(name, false),
            &egraph.functions,
            &egraph.type_info,
            false, // seminaive off for scheduler action rule
        );
        let mut entries = free_vars
            .iter()
            .map(|fv| arule_builder.entry(&GenericAtomTerm::Var(span!(), fv.clone())))
            .collect::<Vec<_>>();
        entries.push(unit_entry.clone());
        arule_builder
            .rb
            .query_table(validated, &entries, None)
            .unwrap();
        arule_builder.actions(&rule.head).unwrap();
        let arule_id = arule_builder.build();

        // Cleanup is intentionally separate from validation and action rules so
        // stale scheduled matches are removed even when the original rule body
        // no longer matches after subsumption or deletion.
        let mut cleanup_rule_builder = BackendRule::new(
            egraph.backend.new_rule(name, false),
            &egraph.functions,
            &egraph.type_info,
            false, // scheduler maintenance rule with no user body/actions
        );
        let mut entries = free_vars
            .iter()
            .map(|fv| cleanup_rule_builder.entry(&GenericAtomTerm::Var(span!(), fv.clone())))
            .collect::<Vec<_>>();
        entries.push(unit_entry);
        cleanup_rule_builder
            .rb
            .query_table(decided, &entries, None)
            .unwrap();
        entries.pop();
        cleanup_rule_builder.rb.remove(decided, &entries);
        cleanup_rule_builder.rb.remove(validated, &entries);
        let cleanup_rule = cleanup_rule_builder.build();

        SchedulerRuleInfo {
            free_vars,
            query_rule: qrule_id,
            validation_rule,
            action_rule: arule_id,
            cleanup_rule,
            matches,
            decided,
            should_seek: true,
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::{Arc, Mutex};

    use super::*;

    #[derive(Clone)]
    struct FirstNScheduler {
        n: usize,
    }

    impl Scheduler for FirstNScheduler {
        fn filter_matches(&mut self, _rule: &str, _ruleset: &str, matches: &mut Matches) -> bool {
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

    #[derive(Clone, Default)]
    struct HoldFirstMatchScheduler {
        calls: usize,
        match_sizes: Arc<Mutex<Vec<usize>>>,
    }

    impl Scheduler for HoldFirstMatchScheduler {
        fn filter_matches(&mut self, _rule: &str, _ruleset: &str, matches: &mut Matches) -> bool {
            self.calls += 1;
            self.match_sizes.lock().unwrap().push(matches.match_size());
            if self.calls == 1 {
                false
            } else {
                matches.choose_all();
                true
            }
        }
    }

    #[derive(Clone, Default)]
    struct CountingScheduler {
        match_sizes: Arc<Mutex<Vec<usize>>>,
    }

    impl Scheduler for CountingScheduler {
        fn filter_matches(&mut self, _rule: &str, _ruleset: &str, matches: &mut Matches) -> bool {
            self.match_sizes.lock().unwrap().push(matches.match_size());
            matches.choose_all();
            true
        }
    }

    #[derive(Clone, Default)]
    struct ChooseFirstScheduler {
        match_sizes: Arc<Mutex<Vec<usize>>>,
    }

    impl Scheduler for ChooseFirstScheduler {
        fn filter_matches(&mut self, _rule: &str, _ruleset: &str, matches: &mut Matches) -> bool {
            self.match_sizes.lock().unwrap().push(matches.match_size());
            if matches.match_size() > 0 {
                matches.choose(0);
            }
            false
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

            if !report.updated {
                break;
            }
        }

        assert_eq!(iter, 12);
    }

    /// Subsumed rows should not be offered to the scheduler as fresh matches.
    ///
    /// The analysis rules reduce the original affine expression before the
    /// scheduled ruleset runs. The scheduled rule body would have matched the
    /// pre-subsumption shape, but the scheduler should see zero valid matches
    /// and should not report progress from internal scheduler bookkeeping.
    #[test]
    fn test_scheduler_does_not_match_subsumed_rows() {
        let mut egraph = EGraph::default();
        let match_sizes = Arc::new(Mutex::new(Vec::new()));
        let scheduler_id = egraph.add_scheduler(Box::new(CountingScheduler {
            match_sizes: match_sizes.clone(),
        }));
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

        assert_eq!(*match_sizes.lock().unwrap(), vec![0]);
        assert_eq!(egraph.get_size("Hit"), 0);
        assert!(
            !report.updated,
            "scheduled rules should not report progress from subsumed-row matches"
        );
    }

    /// Rechecking a chosen scheduler key should not duplicate user actions for
    /// multiple body-only witnesses.
    ///
    /// In `(rule ((A x y)) ((Hit x)))`, `y` proves that the body matches but
    /// does not appear in the action key. With `A(1, 10)` and `A(1, 20)`, a
    /// scheduler can choose one `x = 1` key. Validation may rediscover both
    /// `y` witnesses, but they must collapse into one validated key so `Hit(1)`
    /// runs once and the report counts one action match.
    #[test]
    fn test_scheduler_recheck_does_not_duplicate_body_only_witnesses() {
        let mut egraph = EGraph::default();
        let match_sizes = Arc::new(Mutex::new(Vec::new()));
        let scheduler_id = egraph.add_scheduler(Box::new(ChooseFirstScheduler {
            match_sizes: match_sizes.clone(),
        }));
        let input = r#"
        (ruleset test)
        (relation A (i64 i64))
        (relation Hit (i64))
        (A 1 10)
        (A 1 20)
        (rule ((A x y)) ((Hit x)) :ruleset test :name "hit")
        "#;
        egraph.parse_and_run_program(None, input).unwrap();

        let report = egraph
            .step_rules_with_scheduler(scheduler_id, "test")
            .unwrap();

        assert_eq!(*match_sizes.lock().unwrap(), vec![2]);
        assert_eq!(egraph.get_size("Hit"), 1);
        assert_eq!(report.num_matches_per_rule["hit"], 1);
        assert_eq!(report.iterations.len(), 4);
    }

    /// Held scheduler matches that become stale should be cleaned without
    /// running user actions.
    ///
    /// The scheduler first holds the match instead of choosing it. After the
    /// analysis rules subsume the matched expression, the held key is still in
    /// scheduler state but the original body is no longer true. The validation
    /// rule rejects that key, and the cleanup rule removes it so no stale
    /// worklist tuple remains.
    #[test]
    fn test_scheduler_drops_held_matches_that_become_subsumed() {
        let mut egraph = EGraph::default();
        let match_sizes = Arc::new(Mutex::new(Vec::new()));
        let scheduler_id = egraph.add_scheduler(Box::new(HoldFirstMatchScheduler {
            calls: 0,
            match_sizes: match_sizes.clone(),
        }));
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
        "#;
        egraph.parse_and_run_program(None, input).unwrap();

        let first = egraph
            .step_rules_with_scheduler(scheduler_id, "test")
            .unwrap();
        assert!(!first.updated);
        assert_eq!(egraph.get_size("Hit"), 0);

        egraph
            .parse_and_run_program(None, "(run-schedule (saturate (run analysis)))")
            .unwrap();
        let after_analysis_size = egraph.num_tuples();

        let second = egraph
            .step_rules_with_scheduler(scheduler_id, "test")
            .unwrap();

        assert_eq!(*match_sizes.lock().unwrap(), vec![1, 1]);
        assert_eq!(egraph.get_size("Hit"), 0);
        assert!(
            !second.updated,
            "stale held matches should be cleaned without running user actions"
        );
        assert_eq!(
            egraph.num_tuples(),
            after_analysis_size,
            "stale decided worklist rows should be cleaned"
        );
    }
}
