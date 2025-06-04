use std::sync::Arc;
use std::sync::Mutex;

use bit_vec::BitVec;
use core_relations::ExternalFunction;
use core_relations::Value;
use egglog_bridge::ColumnTy;
use egglog_bridge::DefaultVal;
use egglog_bridge::ExecutionState;
use egglog_bridge::FunctionConfig;
use egglog_bridge::FunctionId;
use egglog_bridge::MergeFn;
use egglog_bridge::RuleId;

use crate::ast::ResolvedVar;
use crate::core::GenericAtomTerm;
use crate::core::ResolvedCoreRule;
use crate::span;
use crate::util::IndexMap;
use crate::BackendRule;
use crate::HashMap;
use crate::Ruleset;
use crate::RunReport;
use crate::Symbol;

use crate::EGraph;

pub trait Scheduler: dyn_clone::DynClone + Send + Sync {
    fn filter_matches(&self, matches: &mut Matches);
}

dyn_clone::clone_trait_object!(Scheduler);

pub struct Matches {
    matches: Vec<Value>,
    chosen: BitVec,
    vars: Vec<ResolvedVar>,
}

pub struct Match<'a> {
    values: &'a [Value],
    vars: &'a [ResolvedVar],
}

impl Match<'_> {
    pub fn get_value(&self, var: Symbol) -> Value {
        let idx = self.vars.iter().position(|v| v.name == var).unwrap();
        self.values[idx]
    }
}

impl Matches {
    fn new(matches: Vec<Value>, vars: Vec<ResolvedVar>) -> Self {
        let total_len = matches.len();
        let tuple_len = vars.len();
        debug_assert!(total_len % tuple_len == 0);
        Self {
            matches,
            vars,
            chosen: BitVec::from_elem(total_len / tuple_len, false),
        }
    }

    pub fn match_size(&self) -> usize {
        self.chosen.len()
    }

    pub fn tuple_len(&self) -> usize {
        self.vars.len()
    }

    pub fn get_match(&self, idx: usize) -> Match {
        Match {
            values: &self.matches[idx * self.tuple_len()..(idx + 1) * self.tuple_len()],
            vars: &self.vars,
        }
    }

    pub fn choose(&mut self, idx: usize) {
        self.chosen.set(idx, true);
    }

    fn instantiate(self, state: &mut ExecutionState, function_id: FunctionId) -> Vec<Value> {
        let tuple_len = self.tuple_len();
        let idx_of = |i: usize| i / tuple_len;
        let mut chosen = Vec::new();
        let mut matches = self.matches;
        let unit = state.prims().get(());

        let mut i = 0;
        matches.retain(|item| {
            let index = idx_of(i);
            i += 1;
            if self.chosen[index] {
                chosen.push(*item);
                false
            } else {
                true
            }
        });
        chosen.chunks(tuple_len).for_each(|row| {
            state.stage_insert(function_id, row, Some(unit));
        });
        matches
    }
}

type SchedulerId = usize;

impl EGraph {
    pub fn register_scheduler<S: Scheduler + 'static>(&mut self, scheduler: S) -> SchedulerId {
        let id = self.schedulers.len();
        self.schedulers.push(SchedulerRecord {
            scheduler: Box::new(scheduler),
            rule_info: Default::default(),
        });
        id
    }

    pub fn step_rule_with_scheduler(
        &mut self,
        scheduler_id: SchedulerId,
        ruleset: Symbol,
    ) -> RunReport {
        let mut rules = Vec::new();
        let rulesets = std::mem::take(&mut self.rulesets);
        collect_rules(ruleset, &rulesets, &mut rules);
        let mut schedulers = std::mem::take(&mut self.schedulers);

        // Step 1: build all the query/action rules and worklist if have not already
        let record = &mut schedulers[scheduler_id];
        rules.iter().for_each(|(id, rule)| {
            record
                .rule_info
                .entry(*id)
                .or_insert_with(|| SchedulerRuleInfo::new(self, rule));
        });

        // Step 2: run all the queries for one iteration
        let query_rules = rules
            .iter()
            .map(|(rule_id, _rule)| {
                let rule_info = record.rule_info.get(rule_id).unwrap();
                rule_info.query_rule
            })
            .collect::<Vec<_>>();

        let query_iter_report = self.backend.run_rules(&query_rules).unwrap();

        // Step 3: let the scheduler decide which matches need to be kept
        self.backend.with_execution_state(|mut state| {
            for (rule_id, _rule) in rules.iter() {
                let rule_info = record.rule_info.get(rule_id).unwrap();

                let matches: Vec<Value> =
                    std::mem::take(rule_info.matches.lock().unwrap().as_mut());
                let mut matches = Matches::new(matches, rule_info.free_vars.clone());
                record.scheduler.filter_matches(&mut matches);
                *rule_info.matches.lock().unwrap() =
                    matches.instantiate(&mut state, rule_info.decided);
            }
        });

        // Step 4: run the action rules
        let action_rules = rules
            .iter()
            .map(|(rule_id, _rule)| {
                let rule_info = record.rule_info.get(rule_id).unwrap();
                rule_info.action_rule
            })
            .collect::<Vec<_>>();
        let action_iter_report = self.backend.run_rules(&action_rules).unwrap();

        self.rulesets = rulesets;
        self.schedulers = schedulers;

        // Step 5: combine the reports
        let per_ruleset = |x| [(ruleset, x)].into_iter().collect();
        let mut report = RunReport::default();
        report.updated = action_iter_report.changed;

        report.search_and_apply_time_per_ruleset = per_ruleset(
            query_iter_report.search_and_apply_time + action_iter_report.search_and_apply_time,
        );
        report.merge_time_per_ruleset =
            per_ruleset(query_iter_report.merge_time + action_iter_report.merge_time);
        report.rebuild_time_per_ruleset =
            per_ruleset(query_iter_report.rebuild_time + action_iter_report.rebuild_time);

        // TODO: correctly track the provenance of each rule
        // report.search_and_apply_time_per_rule = query_iter_report
        report.num_matches_per_rule = action_iter_report.rule_reports
            .iter()
            .map(|(rule, report)| (rule.as_str().into(), report.num_matches))
            .collect();
        report
    }
}

fn collect_rules<'a>(
    ruleset: Symbol,
    rulesets: &'a IndexMap<Symbol, Ruleset>,
    ids: &mut Vec<(Symbol, &'a ResolvedCoreRule)>,
) {
    match &rulesets[&ruleset] {
        Ruleset::Rules(rules) => {
            for (rule_name, (core_rule, _)) in rules.iter() {
                ids.push((*rule_name, core_rule));
            }
        }
        Ruleset::Combined(sub_rulesets) => {
            for sub_ruleset in sub_rulesets {
                collect_rules(*sub_ruleset, rulesets, ids);
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct SchedulerRecord {
    scheduler: Box<dyn Scheduler>,
    rule_info: HashMap<Symbol, SchedulerRuleInfo>,
}

/// To enable scheduling without modifying the backend,
/// we split a rule (rule query action) into a worklist relation
/// two rules (rule query (worklist vars false)) and
/// (rule (worklist vars false) (action ... (delete (worklist vars false))))
#[derive(Clone)]
struct SchedulerRuleInfo {
    matches: Arc<Mutex<Vec<Value>>>,
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
        self.matches
            .lock()
            .unwrap()
            .extend(args.iter().copied());
        Some(state.prims().get(()))
    }
}

impl SchedulerRuleInfo {
    pub fn new(egraph: &mut EGraph, rule: &ResolvedCoreRule) -> SchedulerRuleInfo {
        let free_vars = rule.head.get_free_vars().into_iter().collect::<Vec<_>>();
        let unit_type = egraph.backend.primitives().get_ty::<()>();
        let unit = egraph.backend.primitives().get(());
        let unit_entry = egraph.backend.primitive_constant(());

        let matches = Arc::new(Mutex::new(Vec::new()));
        let collect_matches = egraph
            .backend
            .register_external_func(CollectMatches::new(matches.clone()));
        let schema = free_vars
            .iter()
            .map(|v| v.sort.column_ty(&egraph.backend))
            .chain(std::iter::once(ColumnTy::Primitive(unit_type)))
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
            egraph.backend.new_rule("scheduler_query", true),
            &egraph.functions,
            &egraph.type_info,
        );
        qrule_builder.query(&rule.body, true);
        let entries = free_vars
            .iter()
            .map(|fv| qrule_builder.entry(&GenericAtomTerm::Var(span!(), fv.clone())))
            .collect::<Vec<_>>();
        let _var = qrule_builder.rb.call_external_func(
            collect_matches,
            &entries,
            ColumnTy::Primitive(unit_type),
            "collect_matches",
        );
        let qrule_id = qrule_builder.build();

        // Step 2: build the action rule
        let mut arule_builder = BackendRule::new(
            egraph.backend.new_rule("scheduler_action", false),
            &egraph.functions,
            &egraph.type_info,
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
        fn filter_matches(&self, matches: &mut Matches) {
            dbg!(&matches.match_size());
            for i in 0..std::cmp::min(self.n, matches.match_size()) {
                matches.choose(i);
            }
        }
    }

    #[test]
    fn test_first_n_scheduler() {
        let mut egraph = EGraph::default();
        let scheduler = FirstNScheduler { n: 10 };
        let scheduler_id = egraph.register_scheduler(scheduler);
        let input = r#"
        (relation R (i64))
        (R 0)
        (rule ((R x) (< x 100)) ((R (+ x 1))))
        (run-schedule (saturate (run)))

        (ruleset test)
        (relation S (i64))
        (rule ((R x)) ((S x)) :ruleset test)
        "#;
        egraph.parse_and_run_program(None, input).unwrap();
        let r_id = egraph.functions.get(&Symbol::from("R")).unwrap().backend_id;
        assert_eq!(egraph.get_size(r_id), 101);
        let s_id = egraph.functions.get(&Symbol::from("S")).unwrap().backend_id;
        let mut iter = 0;
        loop {
            let report = egraph.step_rule_with_scheduler(scheduler_id, "test".into());
            let table_size = egraph.get_size(s_id);
            dbg!(&table_size);
            iter += 1;
            assert_eq!(table_size, std::cmp::min(iter * 10, 101));
            if !report.updated {
                break;
            }
        }

        assert_eq!(iter, 12);
    }
}
