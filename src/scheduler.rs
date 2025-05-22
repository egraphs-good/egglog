use std::sync::Arc;

use dyn_clone::DynClone;
use egglog_bridge::{ColumnTy, DefaultVal, FunctionConfig, FunctionId, MergeFn, RuleId};

use crate::core::{
    CoreAction, CoreRule, GenericAtomTerm, GenericCoreAction, GenericCoreActions, ResolvedCoreRule,
};
use crate::util::IndexMap;
use crate::{actions, span, BackendRule, Ruleset, Symbol};

use crate::{EGraph, HashMap, RunReport};

type SchedulerId = usize;

impl EGraph {
    pub fn step_rules_with_scheduler(&mut self, ruleset: Symbol, id: SchedulerId) -> RunReport {
        let mut rules = Vec::new();
        let rulesets = std::mem::take(&mut self.rulesets);
        collect_rules(ruleset, &rulesets, &mut rules);
        let mut schedulers = std::mem::take(&mut self.schedulers);

        // Step 1: build all the query/action rules and worklist if have not already
        let record = &mut schedulers[id];
        rules
            .iter()
            .for_each(|(id, rule)| {
                record
                    .rule_info
                    .entry(*id)
                    .or_insert_with(|| SchedulerRuleInfo::new(self, rule));
            });
        
        // Step 2: run all the queries for one iteration
        let query_rules = rules.iter().map(|(rule_id, _rule)| {
            let rule_info = record.rule_info.get(rule_id).unwrap();
            rule_info.query_rule
        }).collect::<Vec<_>>();
        self.backend.run_rules(&query_rules).unwrap();

        // Step 3: let the scheduler decide which matches need to be kept
        for (rule_id, _rule) in rules.iter() {
            let rule_info = record.rule_info.get(rule_id).unwrap();
            let ms = Matches { function: rule_info.worklist_id };
            record.scheduler.schedule_matches(self, &ms);
        }

        let action_rules = rules.iter().map(|(rule_id, _rule)| {
            let rule_info = record.rule_info.get(rule_id).unwrap();
            rule_info.action_rule
        }).collect::<Vec<_>>();
        self.backend.run_rules(&action_rules).unwrap();

        self.rulesets = rulesets;
        self.schedulers = schedulers;
        todo!()
    }
}

fn collect_rules<'a>(
    ruleset: Symbol,
    rulesets: &'a IndexMap<Symbol, Ruleset>,
    ids: &mut Vec<(Symbol, &'a ResolvedCoreRule)>,
) {
    match &rulesets[&ruleset] {
        Ruleset::Rules(rule_name, rules) => {
            for (_, _, core_rule) in rules.values() {
                ids.push((*rule_name, core_rule));
            }
        }
        Ruleset::Combined(_, sub_rulesets) => {
            for sub_ruleset in sub_rulesets {
                collect_rules(*sub_ruleset, rulesets, ids);
            }
        }
    }
}

pub trait Scheduler: dyn_clone::DynClone + Send + Sync {
    fn can_stop(&mut self, iteration: usize) -> bool;

    fn schedule_matches(&mut self, egraph: &mut EGraph, matches: &Matches);
}

dyn_clone::clone_trait_object!(Scheduler);

#[derive(Clone)]
pub struct Matches {
    function: egglog_bridge::FunctionId,
}

struct MatchIterator<'a> {
    egraph: &'a EGraph,
}

impl Matches {
    fn iter(&self, egraph: &EGraph) -> MatchIterator<'_> {
        // egraph.backend.
        todo!()
    }
}

#[derive(Clone)]
pub(crate) struct SchedulerRecord {
    scheduler: Box<dyn Scheduler>,
    matches: Matches,
    rule_info: HashMap<Symbol, SchedulerRuleInfo>,
}

/// To enable scheduling without modifying the backend,
/// we split a rule (rule query action) into a worklist relation
/// two rules (rule query (worklist vars false)) and
/// (rule (worklist vars false) (action ... (delete (worklist vars false))))
///
/// Every time
#[derive(Clone)]
struct SchedulerRuleInfo {
    timestamp: u32,
    worklist_id: FunctionId,
    query_rule: RuleId,
    action_rule: RuleId,
}

impl SchedulerRuleInfo {
    pub fn new(egraph: &mut EGraph, rule: &ResolvedCoreRule) -> SchedulerRuleInfo {
        // step 1: create the worklist table
        let free_vars = rule.head.get_free_vars();
        let bool_ty = ColumnTy::Primitive(egraph.backend.primitives().get_ty::<bool>());
        let bool_true = egraph.backend.primitive_constant(true);
        let bool_false = egraph.backend.primitive_constant(false);
        let schema = free_vars
            .iter()
            .map(|v| v.sort.column_ty(&egraph.backend))
            .chain(Some(bool_ty))
            .collect();
        // TODO: This can be improved by checking if the `or` is actually the one we want
        // which can gets rid of the assertion below
        let bool_or = egraph.type_info.get_prims(&"or".into()).unwrap();
        assert!(bool_or.len() == 1);
        let bool_or = bool_or.into_iter().next().unwrap();

        // Step 1: build the work list relation
        let worklist_id = egraph.backend.add_table(FunctionConfig {
            schema,
            default: DefaultVal::Fail,
            merge: MergeFn::Primitive(bool_or.1, vec![MergeFn::Old, MergeFn::New]),
            name: format!("backend"),
            can_subsume: true,
        });

        // Step 2: build the query rule
        let mut qrule_builder = BackendRule::new(
            egraph.backend.new_rule("scheduler_query", true),
            &egraph.functions,
            &egraph.type_info,
        );
        qrule_builder.query(&rule.body, false);
        let mut entries = free_vars
            .iter()
            .map(|fv| qrule_builder.entry(&GenericAtomTerm::Var(span!(), fv.clone())))
            .collect::<Vec<_>>();
        // By default we don't match the entry, so it is set to false
        entries.push(bool_false);
        qrule_builder.rb.set(worklist_id, &entries);
        let qrule_id = qrule_builder.build();

        // Step 3: build the action rule
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
        entries.push(bool_true);
        arule_builder
            .rb
            .query_table(worklist_id, &entries, Some(false))
            .unwrap();
        arule_builder.actions(&rule.head).unwrap();
        // Remove the entry as it's now done
        entries.pop();
        arule_builder.rb.remove(worklist_id, &entries);
        let arule_id = arule_builder.build();

        SchedulerRuleInfo {
            timestamp: 0,
            worklist_id,
            query_rule: qrule_id,
            action_rule: arule_id,
        }
    }
}
