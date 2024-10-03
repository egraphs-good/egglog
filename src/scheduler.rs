use std::time::Instant;

use crate::{Ruleset, Symbol};

use crate::{EGraph, HashMap, ListDisplay, ResolvedRunConfig, ResolvedSchedule, RunReport, SearchResult, Span};

pub trait Scheduler {
    fn run_schedule(&mut self, egraph: &mut EGraph, sched: &ResolvedSchedule) -> RunReport {
        match sched {
            ResolvedSchedule::Run(span, config) => self.run_rules(egraph, span, config),
            ResolvedSchedule::Repeat(_span, limit, sched) => {
                let mut report = RunReport::default();
                for _i in 0..*limit {
                    let rec = self.run_schedule(egraph, sched);
                    report = report.union(&rec);
                    if !rec.updated {
                        break;
                    }
                }
                report
            }
            ResolvedSchedule::Saturate(_span, sched) => {
                let mut report = RunReport::default();
                loop {
                    let rec = self.run_schedule(egraph, sched);
                    report = report.union(&rec);
                    if !rec.updated {
                        break;
                    }
                }
                report
            }
            ResolvedSchedule::Sequence(_span, scheds) => {
                let mut report = RunReport::default();
                for sched in scheds {
                    report = report.union(&self.run_schedule(egraph, sched));
                }
                report
            }
        }
    }

    fn run_rules(
        &mut self,
        egraph: &mut EGraph,
        span: &Span,
        config: &ResolvedRunConfig,
    ) -> RunReport {
        let mut report: RunReport = Default::default();

        // first rebuild
        let rebuild_start = Instant::now();
        let updates = egraph.rebuild_nofail();
        log::debug!("database size: {}", egraph.num_tuples());
        log::debug!("Made {updates} updates");
        // add to the rebuild time for this ruleset
        report.add_ruleset_rebuild_time(config.ruleset, rebuild_start.elapsed());
        egraph.bump_timestamp();

        let ResolvedRunConfig { ruleset, until } = config;

        if let Some(facts) = until {
            if egraph.check_facts(span, facts).is_ok() {
                log::info!(
                    "Breaking early because of facts:\n {}!",
                    ListDisplay(facts, "\n")
                );
                return report;
            }
        }

        let subreport = self.step_rules(egraph, *ruleset);
        report = report.union(&subreport);

        log::debug!("database size: {}", egraph.num_tuples());
        egraph.bump_timestamp();

        report
    }

    fn step_rules(&mut self, egraph: &mut EGraph, ruleset: Symbol) -> RunReport {
        let n_unions_before = egraph.unionfind.n_unions();
        let mut run_report = Default::default();
        let mut search_results = HashMap::<Symbol, SearchResult>::default();
        self.search_rules(egraph, ruleset, &mut run_report, &mut search_results);
        self.apply_rules(egraph, ruleset, &mut run_report, &search_results);
        run_report.updated |=
            egraph.did_change_tables() || n_unions_before != egraph.unionfind.n_unions();

        run_report
    }

    /// Search all the rules in a ruleset.
    /// Add the search results for a rule to search_results, a map indexed by rule name.
    fn search_rules(
        &self,
        egraph: &EGraph,
        ruleset: Symbol,
        run_report: &mut RunReport,
        search_results: &mut HashMap<Symbol, SearchResult>,
    ) {
        let rules = egraph
            .rulesets
            .get(&ruleset)
            .unwrap_or_else(|| panic!("ruleset does not exist: {}", &ruleset));
        match rules {
            Ruleset::Rules(_ruleset_name, rule_names) => {
                let copy_rules = rule_names.clone();
                let search_start = Instant::now();

                for (rule_name, rule) in copy_rules.iter() {
                    let mut all_matches = vec![];
                    let rule_search_start = Instant::now();
                    let mut did_match = false;
                    let timestamp = egraph.rule_last_run_timestamp.get(rule_name).unwrap_or(&0);
                    egraph.run_query(&rule.query, *timestamp, false, |values| {
                        did_match = true;
                        assert_eq!(values.len(), rule.query.vars.len());
                        all_matches.extend_from_slice(values);
                        Ok(())
                    });
                    let rule_search_time = rule_search_start.elapsed();
                    log::trace!(
                        "Searched for {rule_name} in {:.3}s ({} results)",
                        rule_search_time.as_secs_f64(),
                        all_matches.len()
                    );
                    run_report.add_rule_search_time(*rule_name, rule_search_time);
                    search_results.insert(
                        *rule_name,
                        SearchResult {
                            all_matches,
                            did_match,
                        },
                    );
                }

                let search_time = search_start.elapsed();
                run_report.add_ruleset_search_time(ruleset, search_time);
            }
            Ruleset::Combined(_name, sub_rulesets) => {
                let start_time = Instant::now();
                for sub_ruleset in sub_rulesets {
                    self.search_rules(egraph, *sub_ruleset, run_report, search_results);
                }
                let search_time = start_time.elapsed();
                run_report.add_ruleset_search_time(ruleset, search_time);
            }
        }
    }

    fn apply_rules(
        &mut self,
        egraph: &mut EGraph,
        ruleset: Symbol,
        run_report: &mut RunReport,
        search_results: &HashMap<Symbol, SearchResult>,
    ) {
        // TODO this clone is not efficient
        let rules = egraph.rulesets.get(&ruleset).unwrap().clone();
        match rules {
            Ruleset::Rules(_name, compiled_rules) => {
                let apply_start = Instant::now();
                let rule_names = compiled_rules.keys().cloned().collect::<Vec<_>>();
                for rule_name in rule_names {
                    let SearchResult {
                        all_matches,
                        did_match,
                    } = search_results.get(&rule_name).unwrap();
                    let rule = compiled_rules.get(&rule_name).unwrap();
                    let num_vars = rule.query.vars.len();

                    // make sure the query requires matches
                    if num_vars != 0 {
                        run_report.add_rule_num_matches(rule_name, all_matches.len() / num_vars);
                    }

                    egraph.rule_last_run_timestamp
                        .insert(rule_name, egraph.get_timestamp());
                    let rule_apply_start = Instant::now();

                    // when there are no variables, a query can still fail to match
                    // here we handle that case
                    if num_vars == 0 {
                        if *did_match {
                            egraph.run_actions(&[], &rule.program, true)
                                .unwrap_or_else(|e| {
                                    panic!("error while running actions for {rule_name}: {e}")
                                });
                        }
                    } else {
                        for values in all_matches.chunks(num_vars) {
                            egraph.run_actions(values, &rule.program, true)
                                .unwrap_or_else(|e| {
                                    panic!("error while running actions for {rule_name}: {e}")
                                });
                        }
                    }

                    // add to the rule's apply time
                    run_report.add_rule_apply_time(rule_name, rule_apply_start.elapsed());
                }
                run_report.add_ruleset_apply_time(ruleset, apply_start.elapsed());
            }
            Ruleset::Combined(_name, sub_rulesets) => {
                let start_time = Instant::now();
                for sub_ruleset in sub_rulesets {
                    self.apply_rules(egraph, sub_ruleset, run_report, search_results);
                }
                let apply_time = start_time.elapsed();
                run_report.add_ruleset_apply_time(ruleset, apply_time);
            }
        }
    }
}

pub struct SimpleScheduler;

impl Scheduler for SimpleScheduler {}