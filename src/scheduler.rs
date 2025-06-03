use std::sync::Arc;
use std::sync::Mutex;

use core_relations::ExecutionState;
use core_relations::ExternalFunction;
use core_relations::ExternalFunctionId;
use core_relations::PrimitiveId;
use core_relations::Value;
use egglog_bridge::ColumnTy;
use egglog_bridge::DefaultVal;
use egglog_bridge::FunctionConfig;
use egglog_bridge::FunctionId;
use egglog_bridge::FunctionRow;
use egglog_bridge::MergeFn;
use egglog_bridge::RuleId;

use crate::ast::ResolvedVar;
use crate::core::GenericAtomTerm;
use crate::core::ResolvedCoreRule;
use crate::literal_to_value;
use crate::span;
use crate::util::IndexMap;
use crate::BackendRule;
use crate::HashMap;
use crate::Ruleset;
use crate::RunReport;
use crate::Symbol;

use crate::EGraph;

pub trait Scheduler: dyn_clone::DynClone + Send + Sync {
    fn filter_matches(&self, matches: Matches);
}

dyn_clone::clone_trait_object!(Scheduler);

pub struct Matches {}
impl Matches {
    fn new(state: &mut ExecutionState, matches: &Arc<Mutex<Vec<Value>>>) -> Self {
        Self {  }
    }
}

type SchedulerId = usize;

impl EGraph {
    pub fn register_scheduler(&mut self, scheduler: Box<dyn Scheduler>) -> SchedulerId {
        let id = self.schedulers.len();
        self.schedulers.push(SchedulerRecord {
            scheduler,
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
        self.backend.run_rules(&query_rules).unwrap();

        // Step 3: let the scheduler decide which matches need to be kept
        self.backend.with_execution_state(|state| {
            for (rule_id, _rule) in rules.iter() {
                let rule_info = record.rule_info.get(rule_id).unwrap();

                let matches = Matches::new(state, &rule_info.matches);
                record.scheduler.filter_matches(matches);
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
        self.backend.run_rules(&action_rules).unwrap();

        self.rulesets = rulesets;
        self.schedulers = schedulers;
        RunReport {
            updated: true,
            num_matches_per_rule: Default::default(),
            rebuild_time_per_ruleset: Default::default(),
            search_and_apply_time_per_rule: Default::default(),
            search_and_apply_time_per_ruleset: Default::default(),
            merge_time_per_ruleset: Default::default(),
        }
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
    collect_matches: ExternalFunctionId,
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
            .extend(args.iter().map(|v| v.clone()));
        Some(state.prims().get(()))
    }
}

impl SchedulerRuleInfo {
    pub fn new(egraph: &mut EGraph, rule: &ResolvedCoreRule) -> SchedulerRuleInfo {
        let free_vars = rule.head.get_free_vars().into_iter().collect::<Vec<_>>();
        let unit_type = egraph.backend.primitives().get_ty::<()>();
        let unit = egraph.backend.primitives().get(());
        let unit_entry = egraph.backend.primitive_constant(unit);

        let matches = Arc::new(Mutex::new(Vec::new()));
        let collect_matches = egraph
            .backend
            .register_external_func(CollectMatches::new(matches.clone()));
        let schema = free_vars
            .iter()
            .map(|v| v.sort.column_ty(&egraph.backend))
            .collect();
        let decided = egraph.backend.add_table(FunctionConfig {
            schema,
            default: DefaultVal::Const(unit),
            merge: MergeFn::AssertEq,
            name: format!("backend"),
            can_subsume: false,
        });

        // Step 1: build the query rule
        let mut qrule_builder = BackendRule::new(
            egraph.backend.new_rule("scheduler_query", true),
            &egraph.functions,
            &egraph.type_info,
        );
        qrule_builder.query(&rule.body, false);
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
        // Find entries that are only true
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
            collect_matches,
        }
    }
}
