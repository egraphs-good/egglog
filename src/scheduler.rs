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

/// A scheduler that rematches the rebuilt e-graph every iteration.
///
/// For example, if `copy: R(x) -> S(x)` is skipped while `grow` adds a new
/// `R(1)` row, a backlog scheduler can replay only the skipped `copy(R(0))`
/// match while a fresh scheduler rematches and sees both `R(0)` and `R(1)`.
pub trait FreshScheduler: dyn_clone::DynClone + Send + Sync {
    /// Whether a rule should be queried against the current rebuilt e-graph in
    /// this iteration.
    fn should_search(&mut self, rule: &str, ruleset: &str) -> bool {
        let _ = (rule, ruleset);
        true
    }

    /// Whether or not the rules can be considered as saturated (i.e.,
    /// `run_report.updated == false`).
    fn can_stop(&mut self, rules: &[&str], ruleset: &str) -> bool {
        let _ = (rules, ruleset);
        true
    }

    /// Filter the current iteration's fresh matches for a rule.
    ///
    /// Unchosen matches are discarded after the iteration.
    fn filter_matches(&mut self, rule: &str, ruleset: &str, matches: &mut Matches);
}

dyn_clone::clone_trait_object!(Scheduler);
dyn_clone::clone_trait_object!(FreshScheduler);

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

#[derive(Clone)]
enum SchedulerKind {
    Backlog(Box<dyn Scheduler>),
    Fresh(Box<dyn FreshScheduler>),
}

impl EGraph {
    /// Register a new scheduler and return its id.
    pub fn add_scheduler(&mut self, scheduler: Box<dyn Scheduler>) -> SchedulerId {
        self.schedulers.push(SchedulerRecord {
            scheduler: SchedulerKind::Backlog(scheduler),
            rule_info: Default::default(),
        })
    }

    /// Register a new fresh-rematch scheduler and return its id.
    pub fn add_fresh_scheduler(&mut self, scheduler: Box<dyn FreshScheduler>) -> SchedulerId {
        self.schedulers.push(SchedulerRecord {
            scheduler: SchedulerKind::Fresh(scheduler),
            rule_info: Default::default(),
        })
    }

    /// Removes a backlog scheduler.
    ///
    /// Returns `None` for fresh scheduler ids; use
    /// [`EGraph::remove_fresh_scheduler`] to remove those.
    pub fn remove_scheduler(&mut self, scheduler_id: SchedulerId) -> Option<Box<dyn Scheduler>> {
        if matches!(
            self.schedulers.get(scheduler_id),
            Some(SchedulerRecord {
                scheduler: SchedulerKind::Backlog(_),
                ..
            })
        ) {
            self.schedulers
                .take(scheduler_id)
                .and_then(|r| match r.scheduler {
                    SchedulerKind::Backlog(scheduler) => Some(scheduler),
                    SchedulerKind::Fresh(_) => None,
                })
        } else {
            None
        }
    }

    /// Removes a fresh-rematch scheduler.
    ///
    /// Returns `None` for backlog scheduler ids; use [`EGraph::remove_scheduler`]
    /// to remove those.
    pub fn remove_fresh_scheduler(
        &mut self,
        scheduler_id: SchedulerId,
    ) -> Option<Box<dyn FreshScheduler>> {
        if matches!(
            self.schedulers.get(scheduler_id),
            Some(SchedulerRecord {
                scheduler: SchedulerKind::Fresh(_),
                ..
            })
        ) {
            self.schedulers
                .take(scheduler_id)
                .and_then(|r| match r.scheduler {
                    SchedulerKind::Fresh(scheduler) => Some(scheduler),
                    SchedulerKind::Backlog(_) => None,
                })
        } else {
            None
        }
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
        let fresh = matches!(record.scheduler, SchedulerKind::Fresh(_));
        rules.iter().for_each(|(id, rule)| {
            record
                .rule_info
                .entry((*id).to_owned())
                .or_insert_with(|| SchedulerRuleInfo::new(self, rule, id, fresh));
        });

        let scheduler = &mut record.scheduler;
        let rule_info = &mut record.rule_info;

        // Step 2: run all the queries for one iteration
        let query_rules = match scheduler {
            SchedulerKind::Backlog(_) => rules
                .iter()
                .filter_map(|(rule_id, _rule)| {
                    let rule_info = rule_info.get(rule_id).unwrap();
                    if rule_info.should_seek {
                        Some(rule_info.query_rule)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>(),
            SchedulerKind::Fresh(scheduler) => {
                for (rule_id, _rule) in rules.iter() {
                    rule_info
                        .get_mut(rule_id)
                        .unwrap()
                        .matches
                        .lock()
                        .unwrap()
                        .clear();
                }
                rules
                    .iter()
                    .filter_map(|(rule_id, _rule)| {
                        let rule_info = rule_info.get(rule_id).unwrap();
                        if scheduler.should_search(rule_id, ruleset) {
                            Some(rule_info.query_rule)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            }
        };

        let query_iter_report = self
            .backend
            .run_rules(&query_rules)
            .map_err(|e| Error::BackendError(e.to_string()))?;

        // Step 3: let the scheduler decide which matches need to be kept
        self.backend.with_execution_state(|state| {
            for (rule_id, _rule) in rules.iter() {
                let rule_info = rule_info.get_mut(rule_id).unwrap();

                let matches: Vec<Value> =
                    std::mem::take(rule_info.matches.lock().unwrap().as_mut());
                let mut matches = Matches::new(matches, rule_info.free_vars.clone());
                let table_action = TableAction::new(&self.backend, rule_info.decided);
                match scheduler {
                    SchedulerKind::Backlog(scheduler) => {
                        rule_info.should_seek =
                            scheduler.filter_matches(rule_id, ruleset, &mut matches);
                        *rule_info.matches.lock().unwrap() =
                            matches.instantiate(state, &table_action);
                    }
                    SchedulerKind::Fresh(scheduler) => {
                        scheduler.filter_matches(rule_id, ruleset, &mut matches);
                        let _ = matches.instantiate(state, &table_action);
                    }
                }
            }
        });
        self.backend.flush_updates();

        // Step 4: run the action rules
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

        // Step 5: combine the reports
        let mut query_report = RunReport::singleton(ruleset, query_iter_report);
        let mut action_report = RunReport::singleton(ruleset, action_iter_report);

        // query matches don't count
        query_report.updated = false;
        query_report.num_matches_per_rule.clear();
        // if the scheduler says it shouldn't stop, then it's considered updated (unsaturated)
        action_report.updated = action_report.updated || {
            let rule_ids = rules.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>();
            match scheduler {
                SchedulerKind::Backlog(scheduler) => !scheduler.can_stop(&rule_ids, ruleset),
                SchedulerKind::Fresh(scheduler) => !scheduler.can_stop(&rule_ids, ruleset),
            }
        };

        query_report.union(action_report);

        self.rulesets = rulesets;
        self.schedulers = schedulers;

        Ok(query_report)
    }
}

#[derive(Clone)]
pub(crate) struct SchedulerRecord {
    scheduler: SchedulerKind,
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
        fresh: bool,
    ) -> SchedulerRuleInfo {
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
        // Fresh schedulers rematch the rebuilt graph every step by querying non-seminaively.
        let mut qrule_builder = BackendRule::new(
            egraph.backend.new_rule(name, !fresh),
            &egraph.functions,
            &egraph.type_info,
            true, // seminaive rule context
        );
        qrule_builder.query(&rule.body, !fresh);
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

        // Step 2: build the action rule
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
        entries.push(unit_entry);
        arule_builder
            .rb
            .query_table(decided, &entries, None)
            .unwrap();
        arule_builder.actions(&rule.head).unwrap();
        // Remove the entry as it's now done
        entries.pop();
        arule_builder.rb.remove(decided, &entries);
        let arule_id = arule_builder.build();

        SchedulerRuleInfo {
            free_vars,
            query_rule: qrule_id,
            action_rule: arule_id,
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
    struct SkipCopyOnFirstIterationFreshScheduler {
        copy_calls: usize,
        copy_match_sizes: Arc<Mutex<Vec<usize>>>,
    }

    impl FreshScheduler for SkipCopyOnFirstIterationFreshScheduler {
        fn filter_matches(&mut self, rule: &str, _ruleset: &str, matches: &mut Matches) {
            if rule == "copy" {
                self.copy_calls += 1;
                self.copy_match_sizes
                    .lock()
                    .unwrap()
                    .push(matches.match_size());
                if self.copy_calls == 1 {
                    return;
                }
            }
            matches.choose_all();
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

    #[test]
    fn test_fresh_scheduler_rematches_rebuilt_graph() {
        let mut egraph = EGraph::default();
        let copy_match_sizes = Arc::new(Mutex::new(Vec::new()));
        let scheduler_id =
            egraph.add_fresh_scheduler(Box::new(SkipCopyOnFirstIterationFreshScheduler {
                copy_calls: 0,
                copy_match_sizes: copy_match_sizes.clone(),
            }));
        let input = r#"
        (ruleset test)
        (relation R (i64))
        (relation S (i64))
        (R 0)
        (rule ((R x) (< x 1)) ((R (+ x 1))) :ruleset test :name "grow")
        (rule ((R x)) ((S x)) :ruleset test :name "copy")
        "#;
        egraph.parse_and_run_program(None, input).unwrap();

        let first = egraph
            .step_rules_with_scheduler(scheduler_id, "test")
            .unwrap();
        assert!(first.updated);
        assert_eq!(egraph.get_size("R"), 2);
        assert_eq!(egraph.get_size("S"), 0);

        let second = egraph
            .step_rules_with_scheduler(scheduler_id, "test")
            .unwrap();
        assert!(second.updated);
        assert_eq!(*copy_match_sizes.lock().unwrap(), vec![1, 2]);
        assert_eq!(egraph.get_size("S"), 2);
        assert_eq!(second.num_matches_per_rule["copy"], 2);
    }

    #[test]
    fn test_fresh_scheduler_does_not_match_subsumed_rows() {
        let mut egraph = EGraph::default();
        let copy_match_sizes = Arc::new(Mutex::new(Vec::new()));
        let scheduler_id =
            egraph.add_fresh_scheduler(Box::new(SkipCopyOnFirstIterationFreshScheduler {
                copy_calls: 1,
                copy_match_sizes: copy_match_sizes.clone(),
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
        (rule ((= e (Add (Mul (Num a) x) (Num b)))) ((Hit a)) :ruleset test :name "copy")
        (run-schedule (saturate (run analysis)))
        "#;
        egraph.parse_and_run_program(None, input).unwrap();

        let report = egraph
            .step_rules_with_scheduler(scheduler_id, "test")
            .unwrap();

        assert_eq!(*copy_match_sizes.lock().unwrap(), vec![0]);
        assert_eq!(egraph.get_size("Hit"), 0);
        assert!(!report.updated);
    }

    #[test]
    fn test_remove_fresh_scheduler() {
        let mut egraph = EGraph::default();
        let backlog_id = egraph.add_scheduler(Box::new(FirstNScheduler { n: 1 }));
        let fresh_id =
            egraph.add_fresh_scheduler(Box::new(SkipCopyOnFirstIterationFreshScheduler::default()));

        assert!(egraph.remove_scheduler(fresh_id).is_none());
        assert!(egraph.remove_fresh_scheduler(backlog_id).is_none());
        assert!(egraph.remove_fresh_scheduler(fresh_id).is_some());
        assert!(egraph.remove_scheduler(backlog_id).is_some());
    }
}
