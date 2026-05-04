use std::fmt::{self, Display, Formatter};
use std::time::{Duration, Instant};

use serde::{Serialize, Serializer};

type TimerHandle = usize;

#[derive(Default)]
pub struct Reporter {
    timers: Vec<Timer>,
    sizes: Vec<SizeMetric>,
}

pub struct Timer {
    name: String,
    tags: Vec<String>,
    started_at: Instant,
    breakdown: Vec<Duration>,
}

#[derive(Serialize)]
pub struct RunReport {
    timings: Vec<TimingStep>,
    sizes: Vec<SizeMetric>,
}

#[derive(Serialize)]
struct TimingStep {
    name: String,
    tags: Vec<String>,
    #[serde(with = "serde_millis")]
    total: Duration,
    #[serde(serialize_with = "serialize_duration_breakdown")]
    breakdown: Vec<Duration>,
}

#[derive(Clone, Serialize)]
pub struct SizeMetric {
    name: String,
    value: MetricValue,
}

#[derive(Clone, Serialize)]
pub enum MetricValue {
    Count(u64),
    Bytes(u64),
}

impl Reporter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_timer(&mut self, name: String, tags: Vec<String>) -> TimerHandle {
        let handle = self.timers.len();
        self.timers.push(Timer {
            name,
            tags,
            started_at: Instant::now(),
            breakdown: vec![],
        });
        handle
    }

    pub fn record_timer(&mut self, h: TimerHandle) {
        let old = self.timers[h].started_at;
        let cur = Instant::now();
        self.timers[h].breakdown.push(cur.duration_since(old));
        self.timers[h].started_at = cur;
    }

    pub fn record_size(&mut self, name: String, value: MetricValue) {
        self.sizes.push(SizeMetric { name, value });
    }

    pub fn build_report(&self) -> RunReport {
        let mut steps: Vec<_> = self
            .timers
            .iter()
            .map(|t| TimingStep {
                name: t.name.clone(),
                tags: t.tags.clone(),
                total: t.breakdown.iter().sum(),
                breakdown: t.breakdown.clone(),
            })
            .collect();
        steps.sort_by(|left, right| right.total.cmp(&left.total));

        RunReport {
            timings: steps,
            sizes: self.sizes.clone(),
        }
    }
}

impl Display for RunReport {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "timings:")?;
        for t in self.timings.iter() {
            writeln!(f, "{}", t)?;
        }
        write_metric_section(f, "sizes", &self.sizes)?;
        Ok(())
    }
}

impl Display for TimingStep {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "timing:")?;
        writeln!(f, "  name: {}", self.name)?;
        writeln!(f, "  tags: {:?}", self.tags)?;
        writeln!(f, "  total: {}", self.total.as_secs_f64())?;
        writeln!(
            f,
            "  breakdown: {:?}",
            self.breakdown
                .iter()
                .map(Duration::as_millis)
                .collect::<Vec<_>>()
        )?;
        Ok(())
    }
}

fn serialize_duration_breakdown<S>(breakdown: &[Duration], serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let breakdown_millis: Vec<_> = breakdown.iter().map(Duration::as_millis).collect();
    breakdown_millis.serialize(serializer)
}

fn write_metric_section<T>(f: &mut Formatter<'_>, title: &str, metrics: &[T]) -> fmt::Result
where
    T: Display,
{
    writeln!(f, "{title}:")?;
    if metrics.is_empty() {
        writeln!(f, "  none")?;
        return Ok(());
    }

    for metric in metrics {
        writeln!(f, "  {metric}")?;
    }
    Ok(())
}

impl Display for SizeMetric {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} = {}", self.name, self.value)
    }
}

impl Display for MetricValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            MetricValue::Count(value) => write!(f, "{value}"),
            MetricValue::Bytes(value) => write!(f, "{} bytes", value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_report_aggregates_timers_by_name_and_preserves_tags() {
        let mut reporter = Reporter::new();

        let step_timer_one = reporter.new_timer("step_one".to_string(), vec!["tag".to_string()]);
        reporter.record_timer(step_timer_one);

        let step_timer_two = reporter.new_timer("step_two".to_string(), vec!["tag".to_string()]);
        std::thread::sleep(Duration::from_millis(50));
        reporter.record_timer(step_timer_two);
        std::thread::sleep(Duration::from_millis(50));
        reporter.record_timer(step_timer_two);

        let report = reporter.build_report();

        assert_eq!(report.timings.len(), 2);
        assert_eq!(report.timings[0].name, "step_two");
        assert_eq!(report.timings[0].tags, vec!["tag".to_string()]);
        assert_eq!(report.timings[0].breakdown.len(), 2);
        assert!(report.timings[0].total >= Duration::from_millis(100));
    }
}
