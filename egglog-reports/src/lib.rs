use rustc_hash::FxHasher;
use std::{
    fmt::{Display, Formatter},
    hash::BuildHasherDefault,
};
use web_time::Duration;

pub(crate) type HashMap<K, V> = hashbrown::HashMap<K, V, BuildHasherDefault<FxHasher>>;
// pub(crate) type HashSet<T> = hashbrown::HashSet<T, BuildHasherDefault<FxHasher>>;
pub(crate) type IndexSet<T> = indexmap::IndexSet<T, BuildHasherDefault<FxHasher>>;
// pub(crate) type IndexMap<K, V> = indexmap::IndexMap<K, V, BuildHasherDefault<FxHasher>>;
pub(crate) type DashMap<K, V> = dashmap::DashMap<K, V, BuildHasherDefault<FxHasher>>;

#[derive(Debug, Default)]
pub struct Plan {
    // TODO
}

#[derive(Debug)]
pub struct RuleReport {
    pub plan: Plan,
    pub search_and_apply_time: Duration,
    // TODO: succeeding matches
    pub num_matches: usize,
}

#[derive(Debug, Default)]
pub struct RuleSetReport {
    pub changed: bool,
    pub rule_reports: DashMap<String, Vec<RuleReport>>,
    pub search_and_apply_time: Duration,
    pub merge_time: Duration,
}

#[derive(Debug, Default)]
pub struct IterationReport {
    pub changed: bool,
    pub rule_reports: HashMap<String, RuleReport>,
    pub search_and_apply_time: Duration,
    pub merge_time: Duration,
    pub rebuild_time: Duration,
}

impl RuleReport {
    pub fn union(&self, other: &RuleReport) -> RuleReport {
        RuleReport {
            plan: Default::default(),
            search_and_apply_time: self.search_and_apply_time + other.search_and_apply_time,
            num_matches: self.num_matches + other.num_matches,
        }
    }
}
/// Running a schedule produces a report of the results.
/// This includes rough timing information and whether
/// the database was updated.
/// Calling `union` on two run reports adds the timing
/// information together.
#[derive(Debug, Clone, Default)]
pub struct RunReport {
    /// If any changes were made to the database.
    pub updated: bool,
    pub search_and_apply_time_per_rule: HashMap<String, Duration>,
    pub num_matches_per_rule: HashMap<String, usize>,
    pub search_and_apply_time_per_ruleset: HashMap<String, Duration>,
    pub merge_time_per_ruleset: HashMap<String, Duration>,
    pub rebuild_time_per_ruleset: HashMap<String, Duration>,
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
}

impl Display for RunReport {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut rule_times_vec: Vec<_> = self.search_and_apply_time_per_rule.iter().collect();
        rule_times_vec.sort_by_key(|(_, time)| **time);

        for (rule, time) in rule_times_vec {
            let name = Self::truncate_rule_name(rule.clone());
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
    fn union_times(times: &mut HashMap<String, Duration>, other_times: HashMap<String, Duration>) {
        for (k, v) in other_times {
            *times.entry(k).or_default() += v;
        }
    }

    fn union_counts(counts: &mut HashMap<String, usize>, other_counts: HashMap<String, usize>) {
        for (k, v) in other_counts {
            *counts.entry(k).or_default() += v;
        }
    }

    /// Merge two reports.
    pub fn union(&mut self, other: Self) {
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
