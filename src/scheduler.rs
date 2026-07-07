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
    /// Whether or not the rules can be considered as saturated once no database
    /// changes were made in the current iteration.
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
        // Scheduler state should not count as database progress. Instead it
        // determines whether a no-op iteration can be treated as fully stopped.
        action_report.can_stop = !action_report.updated && {
            let rule_ids = rules.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>();
            record.scheduler.can_stop(&rule_ids, ruleset)
        };

        query_report.union(action_report);

        self.rulesets = rulesets;
        self.schedulers = schedulers;

        // Under term encoding, run maintenance to fixpoint so the next
        // scheduled step queries a canonical e-graph. The maintenance rulesets
        // are run unscheduled; only the user's ruleset is scheduled.
        query_report.union(self.maybe_run_rebuild_schedule(ruleset)?);

        Ok(query_report)
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
        qrule_builder.query(&rule.body, true);
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

    #[derive(Clone)]
    struct ChooseAllScheduler;

    impl Scheduler for ChooseAllScheduler {
        fn filter_matches(&mut self, _rule: &str, _ruleset: &str, matches: &mut Matches) -> bool {
            matches.choose_all();
            // Re-seek next step so we keep firing on rows produced by rebuilding.
            true
        }
    }

    // A scheduled step under term encoding must run maintenance (rebuilding /
    // congruence / UF indexing) before returning, otherwise the next step and
    // any query would observe stale, un-canonicalized view tables.
    #[test]
    fn test_scheduler_runs_term_encoding_maintenance() {
        let mut egraph = EGraph::new_with_term_encoding();
        let scheduler_id = egraph.add_scheduler(Box::new(ChooseAllScheduler));
        egraph
            .parse_and_run_program(
                None,
                r#"
                (sort Math)
                (constructor Add (i64 i64) Math)
                (Add 1 2)
                (ruleset commutative)
                (rule ((Add a b)) ((union (Add a b) (Add b a)))
                      :ruleset commutative :name "commutativity")
                "#,
            )
            .unwrap();

        // One scheduled step fires the rule, creating `(Add 2 1)` and unioning
        // it with `(Add 1 2)`. The equality is only visible in the view tables
        // once maintenance has rebuilt them.
        egraph
            .step_rules_with_scheduler(scheduler_id, "commutative")
            .unwrap();

        // Passes only if maintenance ran during the scheduled step.
        egraph
            .parse_and_run_program(None, "(check (= (Add 1 2) (Add 2 1)))")
            .unwrap();
    }

    // Driving a ruleset through a scheduler to a fixpoint must reach the same
    // saturated e-graph as running it unscheduled, even under term encoding
    // (where maintenance runs between scheduled steps). Uses `choose_all` so
    // termination is unambiguous; the delaying-scheduler case is covered against
    // the real backoff scheduler in egglog-experimental.
    #[test]
    fn test_scheduler_saturation_matches_unscheduled_term_encoding() {
        let program = r#"
            (sort Math)
            (constructor Num (i64) Math)
            (constructor Add (Math Math) Math)
            (Add (Num 1) (Add (Num 2) (Num 3)))
            (ruleset rw)
            (rule ((Add x y)) ((union (Add x y) (Add y x)))
                  :name "comm" :ruleset rw)
            (rule ((Add (Add x y) z)) ((union (Add (Add x y) z) (Add x (Add y z))))
                  :name "assoc" :ruleset rw)
            (rule ((Add x (Add y z))) ((union (Add x (Add y z)) (Add (Add x y) z)))
                  :name "assoc2" :ruleset rw)
        "#;

        // (1) unscheduled saturation
        let mut base = EGraph::new_with_term_encoding();
        base.parse_and_run_program(None, program).unwrap();
        base.parse_and_run_program(None, "(run-schedule (saturate rw))")
            .unwrap();

        // (2) driven through a delaying scheduler to a fixpoint
        let mut sched = EGraph::new_with_term_encoding();
        let id = sched.add_scheduler(Box::new(ChooseAllScheduler));
        sched.parse_and_run_program(None, program).unwrap();
        let mut iters = 0;
        loop {
            let report = sched.step_rules_with_scheduler(id, "rw").unwrap();
            iters += 1;
            assert!(iters < 10_000, "scheduler did not converge");
            if report.can_stop {
                break;
            }
        }

        // Same number of e-nodes for every user constructor.
        for ctor in ["Num", "Add"] {
            assert_eq!(
                base.get_size(ctor),
                sched.get_size(ctor),
                "e-node count differs for {ctor}"
            );
        }

        // The same equalities and inequalities hold in both e-graphs.
        let checks = [
            "(check (= (Add (Num 1) (Add (Num 2) (Num 3))) (Add (Num 3) (Add (Num 2) (Num 1)))))",
            "(check (= (Add (Add (Num 1) (Num 2)) (Num 3)) (Add (Num 1) (Add (Num 2) (Num 3)))))",
            "(check (!= (Num 1) (Num 2)))",
            "(check (!= (Add (Num 1) (Num 2)) (Add (Num 1) (Num 3))))",
        ];
        for check in checks {
            base.parse_and_run_program(None, check)
                .unwrap_or_else(|e| panic!("unscheduled check failed: {check}: {e}"));
            sched
                .parse_and_run_program(None, check)
                .unwrap_or_else(|e| panic!("scheduled check failed: {check}: {e}"));
        }
    }

    // Fractional scheduler: fires one match at a time for a while (holding
    // residual matches across many rebuilds — the "residual staleness" path),
    // then chooses everything to converge. Used to check that residual matches
    // held across term-encoding maintenance don't change the final result.
    #[derive(Clone)]
    struct DelayThenAll {
        n: usize,
        calls: usize,
        budget: usize,
    }

    impl Scheduler for DelayThenAll {
        fn can_stop(&mut self, _rules: &[&str], _ruleset: &str) -> bool {
            // Only stoppable once past the delaying phase, so pending residual
            // matches during the delay never terminate the loop early.
            self.calls >= self.budget
        }

        fn filter_matches(&mut self, _rule: &str, _ruleset: &str, matches: &mut Matches) -> bool {
            self.calls += 1;
            if self.calls < self.budget {
                let size = matches.match_size();
                for i in 0..size.min(self.n) {
                    matches.choose(i);
                }
            } else {
                matches.choose_all();
            }
            true
        }
    }

    #[test]
    fn test_fractional_scheduler_matches_unscheduled_term_encoding() {
        let program = r#"
            (sort Math)
            (constructor Num (i64) Math)
            (constructor Add (Math Math) Math)
            (Add (Num 1) (Add (Num 2) (Num 3)))
            (ruleset rw)
            (rule ((Add x y)) ((union (Add x y) (Add y x)))
                  :name "comm" :ruleset rw)
            (rule ((Add (Add x y) z)) ((union (Add (Add x y) z) (Add x (Add y z))))
                  :name "assoc" :ruleset rw)
            (rule ((Add x (Add y z))) ((union (Add x (Add y z)) (Add (Add x y) z)))
                  :name "assoc2" :ruleset rw)
        "#;

        let mut base = EGraph::new_with_term_encoding();
        base.parse_and_run_program(None, program).unwrap();
        base.parse_and_run_program(None, "(run-schedule (saturate rw))")
            .unwrap();

        let mut sched = EGraph::new_with_term_encoding();
        let id = sched.add_scheduler(Box::new(DelayThenAll {
            n: 1,
            calls: 0,
            budget: 40,
        }));
        sched.parse_and_run_program(None, program).unwrap();
        let mut iters = 0;
        loop {
            let report = sched.step_rules_with_scheduler(id, "rw").unwrap();
            iters += 1;
            assert!(iters < 10_000, "scheduler did not converge");
            if report.can_stop {
                break;
            }
        }

        for ctor in ["Num", "Add"] {
            assert_eq!(
                base.get_size(ctor),
                sched.get_size(ctor),
                "e-node count differs for {ctor} (residual staleness affected the result)"
            );
        }
        for check in [
            "(check (= (Add (Num 1) (Add (Num 2) (Num 3))) (Add (Num 3) (Add (Num 2) (Num 1)))))",
            "(check (!= (Num 1) (Num 2)))",
        ] {
            base.parse_and_run_program(None, check).unwrap();
            sched.parse_and_run_program(None, check).unwrap();
        }
    }

    // Scheduling must keep proof tracking consistent: a term derived by a
    // scheduled rule firing must still have an extractable, checkable proof.
    #[test]
    fn test_scheduler_preserves_proofs() {
        // Proof-testing mode turns `check` into a proof extraction + check, so
        // the equality check below validates the proof produced for the term
        // derived by the scheduled rule firing.
        let mut egraph = EGraph::new_with_proofs().with_proof_testing();
        let id = egraph.add_scheduler(Box::new(ChooseAllScheduler));
        egraph
            .parse_and_run_program(
                None,
                r#"
                (sort Math)
                (constructor Add (i64 i64) Math)
                (Add 1 2)
                (ruleset commutative)
                (rule ((Add a b)) ((union (Add a b) (Add b a)))
                      :ruleset commutative :name "commutativity")
                "#,
            )
            .unwrap();

        // Drive the rule under the scheduler to a fixpoint.
        let mut iters = 0;
        loop {
            let report = egraph.step_rules_with_scheduler(id, "commutative").unwrap();
            iters += 1;
            assert!(iters < 1000, "scheduler did not converge");
            if report.can_stop {
                break;
            }
        }

        // Under proof-testing this extracts and checks a proof of the equality
        // derived by the scheduled rule.
        egraph
            .parse_and_run_program(None, "(check (= (Add 1 2) (Add 2 1)))")
            .unwrap();
    }

    // The backoff scheduler, ported verbatim (minus logging) from
    // egglog-experimental `src/scheduling.rs`, so we can check that a real
    // banning scheduler reaches the same fixpoint under term encoding. Backoff
    // is all-or-nothing (`choose_all` or ban), so it never leaves residual
    // matches.
    #[derive(Clone)]
    struct BackOffScheduler {
        default_match_limit: usize,
        default_ban_length: usize,
        stats: crate::util::HashMap<String, RuleStats>,
    }

    #[derive(Clone)]
    struct RuleStats {
        iteration: usize,
        times_applied: usize,
        banned_until: usize,
        times_banned: usize,
        match_limit: usize,
        ban_length: usize,
    }

    impl BackOffScheduler {
        fn new(match_limit: usize, ban_length: usize) -> Self {
            Self {
                default_match_limit: match_limit,
                default_ban_length: ban_length,
                stats: crate::util::HashMap::default(),
            }
        }

        fn get_stats(&mut self, rule: String) -> &mut RuleStats {
            self.stats.entry(rule).or_insert_with(|| RuleStats {
                iteration: 0,
                times_applied: 0,
                banned_until: 0,
                times_banned: 0,
                match_limit: self.default_match_limit,
                ban_length: self.default_ban_length,
            })
        }
    }

    impl Scheduler for BackOffScheduler {
        fn can_stop(&mut self, rules: &[&str], _ruleset: &str) -> bool {
            let stats = &mut self.stats;
            let mut banned: Vec<(&str, RuleStats)> = rules
                .iter()
                .filter_map(|rule| {
                    let s = stats.remove(*rule).unwrap();
                    if s.banned_until > s.iteration {
                        Some((*rule, s))
                    } else {
                        None
                    }
                })
                .collect();

            let result = if banned.is_empty() {
                true
            } else {
                let min_delta = banned
                    .iter()
                    .map(|(_, s)| s.banned_until - s.iteration)
                    .min()
                    .unwrap();
                for (_, s) in &mut banned {
                    s.banned_until -= min_delta;
                }
                false
            };

            for (rule, s) in banned {
                stats.insert(rule.to_owned(), s);
            }
            result
        }

        fn filter_matches(&mut self, rule: &str, _ruleset: &str, matches: &mut Matches) -> bool {
            let stats = self.get_stats(rule.to_owned());
            stats.iteration += 1;

            if stats.iteration < stats.banned_until {
                return false;
            }

            let threshold = stats
                .match_limit
                .checked_shl(stats.times_banned as u32)
                .unwrap();
            if matches.match_size() > threshold {
                let ban_length = stats.ban_length << stats.times_banned;
                stats.times_banned += 1;
                stats.banned_until = stats.iteration + ban_length;
                false
            } else {
                stats.times_applied += 1;
                matches.choose_all();
                true
            }
        }
    }

    // The real backoff scheduler must reach the same saturated e-graph as an
    // unscheduled run on a benchmark that triggers repeated banning, under term
    // encoding (maintenance runs between scheduled steps).
    #[test]
    fn test_backoff_scheduler_matches_unscheduled_term_encoding() {
        // An AC-rewrite closure over four numbers: enough matches that a small
        // match limit forces `comm`/`assoc` to be banned and later unbanned.
        let program = r#"
            (sort Math)
            (constructor Num (i64) Math)
            (constructor Add (Math Math) Math)
            (Add (Num 1) (Add (Num 2) (Add (Num 3) (Num 4))))
            (ruleset rw)
            (rule ((Add x y)) ((union (Add x y) (Add y x)))
                  :name "comm" :ruleset rw)
            (rule ((Add (Add x y) z)) ((union (Add (Add x y) z) (Add x (Add y z))))
                  :name "assoc" :ruleset rw)
            (rule ((Add x (Add y z))) ((union (Add x (Add y z)) (Add (Add x y) z)))
                  :name "assoc2" :ruleset rw)
        "#;

        let mut base = EGraph::new_with_term_encoding();
        base.parse_and_run_program(None, program).unwrap();
        base.parse_and_run_program(None, "(run-schedule (saturate rw))")
            .unwrap();

        let mut sched = EGraph::new_with_term_encoding();
        let id = sched.add_scheduler(Box::new(BackOffScheduler::new(4, 2)));
        sched.parse_and_run_program(None, program).unwrap();
        let mut banned_at_least_once = false;
        let mut iters = 0;
        loop {
            let report = sched.step_rules_with_scheduler(id, "rw").unwrap();
            iters += 1;
            assert!(iters < 10_000, "backoff scheduler did not converge");
            // A step that finds matches but makes no database change means a rule
            // was banned this iteration.
            if !report.updated && !report.can_stop {
                banned_at_least_once = true;
            }
            if report.can_stop {
                break;
            }
        }
        assert!(
            banned_at_least_once,
            "benchmark never triggered a ban; test would not exercise backoff"
        );

        for ctor in ["Num", "Add"] {
            assert_eq!(
                base.get_size(ctor),
                sched.get_size(ctor),
                "e-node count differs for {ctor}"
            );
        }

        let checks = [
            "(check (= (Add (Num 1) (Add (Num 2) (Add (Num 3) (Num 4)))) \
                       (Add (Num 4) (Add (Num 3) (Add (Num 2) (Num 1))))))",
            "(check (= (Add (Add (Num 1) (Num 2)) (Add (Num 3) (Num 4))) \
                       (Add (Num 1) (Add (Num 2) (Add (Num 3) (Num 4))))))",
            "(check (!= (Num 1) (Num 2)))",
        ];
        for check in checks {
            base.parse_and_run_program(None, check)
                .unwrap_or_else(|e| panic!("unscheduled check failed: {check}: {e}"));
            sched
                .parse_and_run_program(None, check)
                .unwrap_or_else(|e| panic!("backoff check failed: {check}: {e}"));
        }
    }

    #[derive(Clone, Default)]
    struct DelayStopScheduler {
        can_stop_calls: usize,
    }

    impl Scheduler for DelayStopScheduler {
        fn can_stop(&mut self, _rules: &[&str], _ruleset: &str) -> bool {
            self.can_stop_calls += 1;
            self.can_stop_calls > 1
        }

        fn filter_matches(&mut self, _rule: &str, _ruleset: &str, _matches: &mut Matches) -> bool {
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
}
