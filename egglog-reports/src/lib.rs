use clap::clap_derive::ValueEnum;
use rustc_hash::FxHasher;
use serde::Serialize;
use std::{
    fmt::{Display, Formatter},
    hash::BuildHasherDefault,
    sync::Arc,
};
use web_time::Duration;

pub(crate) type HashMap<K, V> = hashbrown::HashMap<K, V, BuildHasherDefault<FxHasher>>;
pub(crate) type IndexSet<T> = indexmap::IndexSet<T, BuildHasherDefault<FxHasher>>;

#[derive(ValueEnum, Default, Serialize, Debug, Clone, Copy)]
pub enum ReportLevel {
    /// Report only the time taken in each search/apply, merge, rebuild phase.
    #[default]
    TimeOnly,
    /// Report [`ReportLevel::TimeOnly`] and query plan for each rule
    WithPlan,
    /// Report [`ReportLevel::WithPlan`] and the detailed statistics at each stage of the query plan.
    StageInfo,
}

#[derive(Serialize, Clone, Debug)]
pub struct SingleScan(pub String, pub (String, i64));
#[derive(Serialize, Clone, Debug)]
pub struct Scan(pub String, pub Vec<(String, i64)>);

#[derive(Serialize, Clone, Debug)]
pub enum Stage {
    Intersect {
        scans: Vec<SingleScan>,
    },
    FusedIntersect {
        cover: Scan,             // build side
        to_intersect: Vec<Scan>, // probe sides
    },
}

#[derive(Serialize, Clone, Debug)]
pub struct StageStats {
    pub num_candidates: usize,
    pub num_succeeded: usize,
}

#[derive(Serialize, Clone, Debug, Default)]
pub struct Plan {
    pub stages: Vec<(
        Stage,
        Option<StageStats>,
        // indices of next stages
        Vec<usize>,
    )>,
}

#[derive(Debug, Serialize, Clone, Default)]
pub struct RuleReport {
    pub plan: Option<Plan>,
    pub search_and_apply_time: Duration,
    // TODO: succeeding matches
    pub num_matches: usize,
}

#[derive(Debug, Serialize, Clone, Default)]
pub struct RuleSetReport {
    pub changed: bool,
    pub rule_reports: HashMap<Arc<str>, Vec<RuleReport>>,
    pub search_and_apply_time: Duration,
    pub merge_time: Duration,
}

impl RuleSetReport {
    pub fn num_matches(&self, rule: &str) -> usize {
        self.rule_reports
            .get(rule)
            .map(|r| r.iter().map(|r| r.num_matches).sum())
            .unwrap_or(0)
    }

    pub fn rule_search_and_apply_time(&self, rule: &str) -> Duration {
        self.rule_reports
            .get(rule)
            .map(|r| r.iter().map(|r| r.search_and_apply_time).sum())
            .unwrap_or(Duration::ZERO)
    }
}

#[derive(Debug, Serialize, Clone, Default)]
pub struct IterationReport {
    pub rule_set_report: RuleSetReport,
    pub rebuild_time: Duration,
}

impl IterationReport {
    pub fn changed(&self) -> bool {
        self.rule_set_report.changed
    }

    pub fn search_and_apply_time(&self) -> Duration {
        self.rule_set_report.search_and_apply_time
    }

    pub fn rule_reports(&self) -> &HashMap<Arc<str>, Vec<RuleReport>> {
        &self.rule_set_report.rule_reports
    }

    pub fn rules(&self) -> impl Iterator<Item = &str> {
        self.rule_set_report.rule_reports.keys().map(|k| k.as_ref())
    }
}

/// Running a schedule produces a report of the results.
/// This includes rough timing information and whether
/// the database was updated.
/// Calling `union` on two run reports adds the timing
/// information together.
#[derive(Debug, Serialize, Clone, Default)]
pub struct RunReport {
    // Since `IterationReport`s are immutable, we can reference count them to avoid
    // expensive cloning when e-graphs are cloned.
    pub iterations: Vec<Arc<IterationReport>>,
    /// If any changes were made to the database.
    pub updated: bool,
    pub search_and_apply_time_per_rule: HashMap<Arc<str>, Duration>,
    pub num_matches_per_rule: HashMap<Arc<str>, usize>,
    pub search_and_apply_time_per_ruleset: HashMap<Arc<str>, Duration>,
    pub merge_time_per_ruleset: HashMap<Arc<str>, Duration>,
    pub rebuild_time_per_ruleset: HashMap<Arc<str>, Duration>,
}

impl Display for RunReport {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut rule_times_vec: Vec<_> = self.search_and_apply_time_per_rule.iter().collect();
        rule_times_vec.sort_by_key(|(_, time)| **time);

        for (rule, time) in rule_times_vec {
            let name = Self::truncate_rule_name(rule.to_string());
            let time = time.as_secs_f64();
            let num_matches = self.num_matches_per_rule.get(rule).copied().unwrap_or(0);
            writeln!(
                f,
                "Rule {name}: search and apply {time:.3}s, num matches {num_matches}",
            )?;
        }

        let rulesets = self
            .search_and_apply_time_per_ruleset
            .keys()
            .chain(self.merge_time_per_ruleset.keys())
            .chain(self.rebuild_time_per_ruleset.keys())
            .collect::<IndexSet<_>>();

        for ruleset in rulesets {
            // print out the search and apply time for rule
            let search_and_apply_time = self
                .search_and_apply_time_per_ruleset
                .get(ruleset)
                .cloned()
                .unwrap_or(Duration::ZERO)
                .as_secs_f64();
            let merge_time = self
                .merge_time_per_ruleset
                .get(ruleset)
                .cloned()
                .unwrap_or(Duration::ZERO)
                .as_secs_f64();
            let rebuild_time = self
                .rebuild_time_per_ruleset
                .get(ruleset)
                .cloned()
                .unwrap_or(Duration::ZERO)
                .as_secs_f64();
            writeln!(
                f,
                "Ruleset {ruleset}: search {search_and_apply_time:.3}s, merge {merge_time:.3}s, rebuild {rebuild_time:.3}s",
            )?;
        }

        Ok(())
    }
}

impl RunReport {
    /// add a ... and a maximum size to the name
    /// for printing, since they may be the rule itself
    fn truncate_rule_name(mut s: String) -> String {
        // replace newlines in s with a space
        s = s.replace('\n', " ");
        if s.len() > 80 {
            s.truncate(80);
            s.push_str("...");
        }
        s
    }

    fn union_times(
        times: &mut HashMap<Arc<str>, Duration>,
        other_times: HashMap<Arc<str>, Duration>,
    ) {
        for (k, v) in other_times {
            *times.entry(k).or_default() += v;
        }
    }

    fn union_counts(counts: &mut HashMap<Arc<str>, usize>, other_counts: HashMap<Arc<str>, usize>) {
        for (k, v) in other_counts {
            *counts.entry(k).or_default() += v;
        }
    }

    pub fn singleton(ruleset: &str, iteration: IterationReport) -> Self {
        let mut report = RunReport::default();

        for rule in iteration.rules() {
            *report
                .search_and_apply_time_per_rule
                .entry(rule.into())
                .or_default() += iteration.rule_set_report.rule_search_and_apply_time(rule);
            *report.num_matches_per_rule.entry(rule.into()).or_default() +=
                iteration.rule_set_report.num_matches(rule);
        }

        let ruleset: Arc<str> = ruleset.into();
        let per_ruleset = |x| [(ruleset.clone(), x)].into_iter().collect();

        report.search_and_apply_time_per_ruleset = per_ruleset(iteration.search_and_apply_time());
        report.merge_time_per_ruleset = per_ruleset(iteration.rule_set_report.merge_time);
        report.rebuild_time_per_ruleset = per_ruleset(iteration.rebuild_time);
        report.updated = iteration.changed();
        report.iterations.push(Arc::new(iteration));

        report
    }

    pub fn add_iteration(&mut self, ruleset: &str, iteration: IterationReport) {
        self.union(RunReport::singleton(ruleset, iteration));
    }

    /// Merge two reports.
    pub fn union(&mut self, other: Self) {
        self.iterations.extend(other.iterations);
        self.updated |= other.updated;
        RunReport::union_times(
            &mut self.search_and_apply_time_per_rule,
            other.search_and_apply_time_per_rule,
        );
        RunReport::union_counts(&mut self.num_matches_per_rule, other.num_matches_per_rule);
        RunReport::union_times(
            &mut self.search_and_apply_time_per_ruleset,
            other.search_and_apply_time_per_ruleset,
        );
        RunReport::union_times(
            &mut self.merge_time_per_ruleset,
            other.merge_time_per_ruleset,
        );
        RunReport::union_times(
            &mut self.rebuild_time_per_ruleset,
            other.rebuild_time_per_ruleset,
        );
    }
}
