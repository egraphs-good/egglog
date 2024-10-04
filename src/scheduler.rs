use std::fmt::{Display, Formatter};
use std::mem;
use std::time::{Duration, Instant};

use crate::{CompiledRule, Error, HashSet, Ruleset, Symbol, Value};

use crate::{EGraph, HashMap, ListDisplay, ResolvedRunConfig, ResolvedSchedule, Span};

/// For each rule, we produce a `SearchResult`
/// storing data about that rule's matches.
/// When a rule has no variables, it may still match- in this case
/// the `did_match` field is used.
pub struct SearchResult {
    all_matches: Vec<Value>,
    did_match: bool,
}

/// Running a schedule produces a report of the results.
/// This includes rough timing information and whether
/// the database was updated.
/// Calling `union` on two run reports adds the timing
/// information together.
#[derive(Debug, Clone, Default)]
pub struct RunReport {
    /// If any changes were made to the database, this is
    /// true.
    pub updated: bool,
    /// The time it took to run the query, for each rule.
    pub search_time_per_rule: HashMap<Symbol, Duration>,
    pub apply_time_per_rule: HashMap<Symbol, Duration>,
    pub search_time_per_ruleset: HashMap<Symbol, Duration>,
    pub num_matches_per_rule: HashMap<Symbol, usize>,
    pub apply_time_per_ruleset: HashMap<Symbol, Duration>,
    pub rebuild_time_per_ruleset: HashMap<Symbol, Duration>,
}

impl RunReport {
    /// add a ... and a maximum size to the name
    /// for printing, since they may be the rule itself
    fn truncate_rule_name(sym: Symbol) -> String {
        let mut s = sym.to_string();
        // replace newlines in s with a space
        s = s.replace('\n', " ");
        if s.len() > 80 {
            s.truncate(80);
            s.push_str("...");
        }
        s
    }

    pub fn add_rule_search_time(&mut self, rule: Symbol, time: Duration) {
        *self.search_time_per_rule.entry(rule).or_default() += time;
    }

    pub fn add_ruleset_search_time(&mut self, ruleset: Symbol, time: Duration) {
        *self.search_time_per_ruleset.entry(ruleset).or_default() += time;
    }

    pub fn add_rule_apply_time(&mut self, rule: Symbol, time: Duration) {
        *self.apply_time_per_rule.entry(rule).or_default() += time;
    }

    pub fn add_ruleset_apply_time(&mut self, ruleset: Symbol, time: Duration) {
        *self.apply_time_per_ruleset.entry(ruleset).or_default() += time;
    }

    pub fn add_ruleset_rebuild_time(&mut self, ruleset: Symbol, time: Duration) {
        *self.rebuild_time_per_ruleset.entry(ruleset).or_default() += time;
    }

    pub fn add_rule_num_matches(&mut self, rule: Symbol, num_matches: usize) {
        *self.num_matches_per_rule.entry(rule).or_default() += num_matches;
    }
}

impl Display for RunReport {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let all_rules = self
            .search_time_per_rule
            .keys()
            .chain(self.apply_time_per_rule.keys())
            .collect::<HashSet<_>>();
        let mut all_rules_vec = all_rules.iter().cloned().collect::<Vec<_>>();
        // sort rules by search and apply time
        all_rules_vec.sort_by_key(|rule| {
            let search_time = self
                .search_time_per_rule
                .get(*rule)
                .cloned()
                .unwrap_or(Duration::default())
                .as_millis();
            let apply_time = self
                .apply_time_per_rule
                .get(*rule)
                .cloned()
                .unwrap_or(Duration::default())
                .as_millis();
            search_time + apply_time
        });

        for rule in all_rules_vec {
            let truncated = Self::truncate_rule_name(*rule);
            // print out the search and apply time for rule
            let search_time = self
                .search_time_per_rule
                .get(rule)
                .cloned()
                .unwrap_or(Duration::default())
                .as_secs_f64();
            let apply_time = self
                .apply_time_per_rule
                .get(rule)
                .cloned()
                .unwrap_or(Duration::default())
                .as_secs_f64();
            let num_matches = self.num_matches_per_rule.get(rule).cloned().unwrap_or(0);
            writeln!(
                f,
                "Rule {truncated}: search {search_time:.3}s, apply {apply_time:.3}s, num matches {num_matches}",
            )?;
        }

        let rulesets = self
            .search_time_per_ruleset
            .keys()
            .chain(self.apply_time_per_ruleset.keys())
            .chain(self.rebuild_time_per_ruleset.keys())
            .collect::<HashSet<_>>();

        for ruleset in rulesets {
            // print out the search and apply time for rule
            let search_time = self
                .search_time_per_ruleset
                .get(ruleset)
                .cloned()
                .unwrap_or(Duration::default())
                .as_secs_f64();
            let apply_time = self
                .apply_time_per_ruleset
                .get(ruleset)
                .cloned()
                .unwrap_or(Duration::default())
                .as_secs_f64();
            let rebuild_time = self
                .rebuild_time_per_ruleset
                .get(ruleset)
                .cloned()
                .unwrap_or(Duration::default())
                .as_secs_f64();
            writeln!(
                f,
                "Ruleset {ruleset}: search {search_time:.3}s, apply {apply_time:.3}s, rebuild {rebuild_time:.3}s",
            )?;
        }

        Ok(())
    }
}

impl RunReport {
    fn union_times(
        times: &HashMap<Symbol, Duration>,
        other_times: &HashMap<Symbol, Duration>,
    ) -> HashMap<Symbol, Duration> {
        let mut new_times = times.clone();
        for (k, v) in other_times {
            let entry = new_times.entry(*k).or_default();
            *entry += *v;
        }
        new_times
    }

    fn union_counts(
        counts: &HashMap<Symbol, usize>,
        other_counts: &HashMap<Symbol, usize>,
    ) -> HashMap<Symbol, usize> {
        let mut new_counts = counts.clone();
        for (k, v) in other_counts {
            let entry = new_counts.entry(*k).or_default();
            *entry += *v;
        }
        new_counts
    }

    pub fn union(&self, other: &Self) -> Self {
        Self {
            updated: self.updated || other.updated,
            search_time_per_rule: Self::union_times(
                &self.search_time_per_rule,
                &other.search_time_per_rule,
            ),
            apply_time_per_rule: Self::union_times(
                &self.apply_time_per_rule,
                &other.apply_time_per_rule,
            ),
            num_matches_per_rule: Self::union_counts(
                &self.num_matches_per_rule,
                &other.num_matches_per_rule,
            ),
            search_time_per_ruleset: Self::union_times(
                &self.search_time_per_ruleset,
                &other.search_time_per_ruleset,
            ),
            apply_time_per_ruleset: Self::union_times(
                &self.apply_time_per_ruleset,
                &other.apply_time_per_ruleset,
            ),
            rebuild_time_per_ruleset: Self::union_times(
                &self.rebuild_time_per_ruleset,
                &other.rebuild_time_per_ruleset,
            ),
        }
    }
}

/// Interface for a scheduler.
///
/// At a high level, a scheduler is an interpreter over programs in the schedule DSL.
/// Users who want to customize the scheduler can implement this trait and override
/// methods as desired.
///
/// During the interpretation, the scheduler is responsible for keeping track of a [RunReport].
pub trait Scheduler {
    fn run_schedule(
        &mut self,
        egraph: &mut EGraph,
        sched: &ResolvedSchedule,
    ) -> Result<RunReport, Error> {
        match sched {
            ResolvedSchedule::Run(span, config) => self.run_rules(egraph, span, config),
            ResolvedSchedule::Repeat(_span, limit, sched) => {
                let mut report = RunReport::default();
                for _i in 0..*limit {
                    let rec = self.run_schedule(egraph, sched)?;
                    report = report.union(&rec);
                    if !rec.updated {
                        break;
                    }
                }
                Ok(report)
            }
            ResolvedSchedule::Saturate(_span, sched) => {
                let mut report = RunReport::default();
                loop {
                    let rec = self.run_schedule(egraph, sched)?;
                    report = report.union(&rec);
                    if !rec.updated {
                        break;
                    }
                }
                Ok(report)
            }
            ResolvedSchedule::Sequence(_span, scheds) => {
                let mut report = RunReport::default();
                for sched in scheds {
                    report = report.union(&self.run_schedule(egraph, sched)?);
                }
                Ok(report)
            }
            ResolvedSchedule::WithScheduler(span, scheduler, args, sched) => {
                let args = args
                    .iter()
                    .map(|arg| egraph.eval_resolved_expr(arg, true))
                    .collect::<Result<Vec<_>, _>>()?;
                let f = egraph
                    .scheduler_constructors
                    .get(scheduler)
                    .ok_or_else(|| Error::SchedulerNotFound(scheduler.to_string(), span.clone()))?;
                let mut scheduler = f(args);
                scheduler.run_schedule(egraph, sched)
            }
        }
    }

    /// Runs a ruleset
    fn run_rules(
        &mut self,
        egraph: &mut EGraph,
        span: &Span,
        config: &ResolvedRunConfig,
    ) -> Result<RunReport, Error> {
        let mut report: RunReport = Default::default();

        // Step 1: Rebuild
        let rebuild_start = Instant::now();
        let updates = egraph.rebuild_nofail();
        log::debug!("database size: {}", egraph.num_tuples());
        log::debug!("Made {updates} updates");
        report.add_ruleset_rebuild_time(config.ruleset, rebuild_start.elapsed());

        egraph.bump_timestamp();

        // Step 2: check if we should break early
        let ResolvedRunConfig { ruleset, until } = config;
        if let Some(facts) = until {
            if egraph.check_facts(span, facts).is_ok() {
                log::info!(
                    "Breaking early because of facts:\n {}!",
                    ListDisplay(facts, "\n")
                );
                return Ok(report);
            }
        }

        // Step 3: Run the rules
        let n_unions_before = egraph.unionfind.n_unions();
        let mut search_results = HashMap::<Symbol, SearchResult>::default();
        let rulesets = mem::take(&mut egraph.rulesets);
        self.search_rules(
            egraph,
            span,
            &rulesets,
            *ruleset,
            &mut report,
            &mut search_results,
        )?;
        self.apply_rules(
            egraph,
            span,
            &rulesets,
            *ruleset,
            &mut report,
            &search_results,
        )?;
        egraph.rulesets = rulesets;
        report.updated |=
            egraph.did_change_tables() || n_unions_before != egraph.unionfind.n_unions();
        log::debug!("database size: {}", egraph.num_tuples());

        egraph.bump_timestamp();

        Ok(report)
    }

    /// Search all the rules in a ruleset.
    /// Add the search results for a rule to search_results, a map indexed by rule name.
    ///
    /// The default behavior is to go through the (potentially combined) ruleset and call
    /// [Scheduler::search_rule] on each rule.
    fn search_rules(
        &mut self,
        egraph: &EGraph,
        span: &Span,
        rulesets: &HashMap<Symbol, Ruleset>,
        ruleset: Symbol,
        report: &mut RunReport,
        search_results: &mut HashMap<Symbol, SearchResult>,
    ) -> Result<(), Error> {
        log::warn!("searching ruleset: {}", ruleset);
        let rules = rulesets
            .get(&ruleset)
            .ok_or_else(|| Error::NoSuchRuleset(ruleset, span.clone()))?;
        match rules {
            Ruleset::Rules(_ruleset_name, rule_names) => {
                let search_start = Instant::now();

                for (rule_name, rule) in rule_names.iter() {
                    self.search_rule(egraph, span, *rule_name, rule, report, search_results);
                }

                let search_time = search_start.elapsed();
                report.add_ruleset_search_time(ruleset, search_time);
            }
            Ruleset::Combined(_name, sub_rulesets) => {
                let start_time = Instant::now();
                for sub_ruleset in sub_rulesets {
                    self.search_rules(
                        egraph,
                        span,
                        rulesets,
                        *sub_ruleset,
                        report,
                        search_results,
                    )?;
                }
                let search_time = start_time.elapsed();
                report.add_ruleset_search_time(ruleset, search_time);
            }
        }
        Ok(())
    }

    /// Searches a single rule.
    ///
    /// In the default case this is just a thin wrapper
    /// around [EGraph::run_query].
    fn search_rule(
        &mut self,
        egraph: &EGraph,
        _span: &Span,
        rule_name: Symbol,
        rule: &CompiledRule,
        report: &mut RunReport,
        search_results: &mut HashMap<Symbol, SearchResult>,
    ) {
        let rule_search_start = Instant::now();
        let mut all_matches = vec![];
        let mut did_match = false;

        let timestamp = egraph.get_rule_timestamp(rule_name);
        egraph.run_query(&rule.query, timestamp, false, |values| {
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
        report.add_rule_search_time(rule_name, rule_search_time);
        search_results.insert(
            rule_name,
            SearchResult {
                all_matches,
                did_match,
            },
        );
    }

    /// Applies rules in a ruleset
    ///
    /// The default behavior is to go through the (potentially combined) ruleset and call
    /// [Scheduler::apply_rule] on each rule.
    fn apply_rules(
        &mut self,
        egraph: &mut EGraph,
        span: &Span,
        rulesets: &HashMap<Symbol, Ruleset>,
        ruleset: Symbol,
        report: &mut RunReport,
        search_results: &HashMap<Symbol, SearchResult>,
    ) -> Result<(), Error> {
        let rules = rulesets
            .get(&ruleset)
            .unwrap_or_else(|| panic!("ruleset does not exist: {}", &ruleset));
        match rules {
            Ruleset::Rules(_name, compiled_rules) => {
                let apply_start = Instant::now();
                for (&rule_name, rule) in compiled_rules.iter() {
                    self.apply_rule(egraph, span, rule_name, rule, report, search_results);
                }
                report.add_ruleset_apply_time(ruleset, apply_start.elapsed());
            }
            Ruleset::Combined(_name, sub_rulesets) => {
                let start_time = Instant::now();
                for sub_ruleset in sub_rulesets {
                    self.apply_rules(egraph, span, rulesets, *sub_ruleset, report, search_results)?;
                }
                let apply_time = start_time.elapsed();
                report.add_ruleset_apply_time(ruleset, apply_time);
            }
        }
        Ok(())
    }

    /// Applies a single rule.
    ///
    /// In the default case this is just a thin wrapper
    /// around [EGraph::run_actions].
    fn apply_rule(
        &mut self,
        egraph: &mut EGraph,
        _span: &Span,
        rule_name: Symbol,
        rule: &CompiledRule,
        report: &mut RunReport,
        search_results: &HashMap<Symbol, SearchResult>,
    ) {
        let SearchResult {
            all_matches,
            did_match,
        } = search_results.get(&rule_name).unwrap();
        let num_vars = rule.query.vars.len();

        let num_matches = if num_vars != 0 {
            all_matches.len() / num_vars
        } else {
            *did_match as usize
        };
        report.add_rule_num_matches(rule_name, num_matches);

        // update the rule's timestamp since this rule is just fully applied
        egraph.update_rule_timestamp(rule_name);

        let rule_apply_start = Instant::now();
        // when there are no variables, a query can still fail to match
        // here we handle that case
        if num_vars == 0 {
            if *did_match {
                egraph
                    .run_actions(&[], &rule.program, true)
                    .unwrap_or_else(|e| panic!("error while running actions for {rule_name}: {e}"));
            }
        } else {
            for values in all_matches.chunks(num_vars) {
                egraph
                    .run_actions(values, &rule.program, true)
                    .unwrap_or_else(|e| panic!("error while running actions for {rule_name}: {e}"));
            }
        }

        // add to the rule's apply time
        report.add_rule_apply_time(rule_name, rule_apply_start.elapsed());
    }
}

pub struct SimpleScheduler;

impl Scheduler for SimpleScheduler {}

#[derive(Default)]
pub struct BackoffScheduler {
    stats: HashMap<Symbol, BackoffStats>,
}

impl BackoffScheduler {
    pub fn rule_stats(&mut self, rule_name: Symbol, rule: &CompiledRule) -> &mut BackoffStats {
        self.stats.entry(rule_name).or_insert_with(|| {
            let match_limit = *rule
                .props
                .get("match_limit")
                .map(|l| match l {
                    crate::Literal::Int(n) => n,
                    _ => panic!("match_limit must be an integer"),
                })
                .unwrap_or_else(|| {
                    log::warn!(
                        "rule {} has no match_limit; using a default limit of 1000",
                        rule_name
                    );
                    &1000
                }) as usize;

            let ban_length = *rule
                .props
                .get("ban_length")
                .map(|l| match l {
                    crate::Literal::Int(n) => n,
                    _ => panic!("ban_length must be an integer"),
                })
                .unwrap_or_else(|| {
                    log::warn!(
                        "rule {} has no ban_length; using a default length of 5",
                        rule_name
                    );
                    &5
                }) as usize;

            BackoffStats {
                iteration: 0,
                times_applied: 0,
                banned_until: 0,
                times_banned: 0,
                match_limit,
                ban_length,
            }
        })
    }
}

pub struct BackoffStats {
    iteration: usize,
    times_applied: usize,
    banned_until: usize,
    times_banned: usize,
    match_limit: usize,
    ban_length: usize,
}

impl Scheduler for BackoffScheduler {
    fn search_rule(
        &mut self,
        egraph: &EGraph,
        _span: &Span,
        rule_name: Symbol,
        rule: &CompiledRule,
        report: &mut RunReport,
        search_results: &mut HashMap<Symbol, SearchResult>,
    ) {
        let stats = self.rule_stats(rule_name, rule);
        stats.iteration += 1;

        // Step 1: Check if we should skip this rule
        if stats.iteration < stats.banned_until {
            log::info!(
                "Skipping {} ({}-{}) for {} iterations...",
                rule_name,
                stats.times_applied,
                stats.times_banned,
                stats.banned_until - stats.iteration,
            );
            return;
        }

        // Step 2: Search for the rule
        let rule_search_start = Instant::now();

        let threshold = stats
            .match_limit
            .checked_shl(stats.times_banned as u32)
            .unwrap();
        let mut fuel = threshold.saturating_add(1);
        let mut all_matches = vec![];

        let timestamp = egraph.get_rule_timestamp(rule_name);
        egraph.run_query(&rule.query, timestamp, false, |values| {
            assert_eq!(values.len(), rule.query.vars.len());
            all_matches.extend_from_slice(values);

            fuel -= 1;
            if fuel == 0 {
                Err(())
            } else {
                Ok(())
            }
        });

        let rule_search_time = rule_search_start.elapsed();
        report.add_rule_search_time(rule_name, rule_search_time);
        log::trace!(
            "Searched for {rule_name} in {:.3}s ({} results)",
            rule_search_time.as_secs_f64(),
            all_matches.len()
        );

        // Step 3: Decide if we should discard the matches
        let did_match = fuel != threshold;
        let search_result = if fuel == 0 {
            let ban_length = stats.ban_length << stats.times_banned;
            stats.times_banned += 1;
            stats.banned_until = stats.iteration + ban_length;
            log::info!(
                "Banning {} ({}-{}) for {} iters: (threshold: {})",
                rule_name,
                stats.times_applied,
                stats.times_banned,
                ban_length,
                threshold,
            );
            SearchResult {
                all_matches: vec![],
                did_match: true,
            }
        } else {
            stats.times_applied += 1;
            SearchResult {
                all_matches,
                did_match,
            }
        };
        search_results.insert(rule_name, search_result);
    }
}
