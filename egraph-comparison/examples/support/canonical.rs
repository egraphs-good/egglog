use egraph_comparison::{CanonicalMode, Database, canonicalize, compare};
use std::{hint::black_box, time::Instant};

// Inputs are already parsed. Every public operation still validates its input.
pub fn measure(
    left: &Database,
    right: &Database,
    samples: usize,
    min_ms: u64,
    single_pass: bool,
    initial_compare_ms: f64,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let start = Instant::now();
    let baseline = canonicalize(left, CanonicalMode::Database)?;
    let baseline_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let actual = canonicalize(right, CanonicalMode::Database)?;
    if actual != baseline {
        return Err("equal benchmark inputs have different canonical bytes".into());
    }
    drop(actual);
    let cached_ms = start.elapsed().as_secs_f64() * 1000.0;
    let canonical_bytes = baseline.len();
    let other = baseline.clone();
    let bytes_time = super::timings(
        || {
            black_box(black_box(&baseline) == black_box(&other));
        },
        samples,
        min_ms,
    );
    drop(other);
    if single_pass {
        // Sum two contiguous operations, including freeing both outputs. Avoid
        // repeating very large canonicalizations solely to time their sum.
        let drop_start = Instant::now();
        drop(baseline);
        let pair_ms = baseline_ms + cached_ms + drop_start.elapsed().as_secs_f64() * 1000.0;
        return Ok(serde_json::json!({
            "compare_ms": initial_compare_ms, "canonical_pair_ms": pair_ms,
            "canonical_baseline_ms": baseline_ms, "canonical_cached_ms": cached_ms,
            "byte_compare": bytes_time, "canonical_bytes": canonical_bytes,
            "samples": 1, "pair_measurement": "sum_of_baseline_and_actual_passes",
        }));
    }
    let compare_time = super::timings(
        || {
            black_box(compare(black_box(left), black_box(right)).unwrap());
        },
        samples,
        min_ms,
    );
    let pair_time = super::timings(
        || {
            let a = canonicalize(black_box(left), CanonicalMode::Database).unwrap();
            let b = canonicalize(black_box(right), CanonicalMode::Database).unwrap();
            assert!(black_box(a == b));
        },
        samples,
        min_ms,
    );
    let cached_time = super::timings(
        || {
            let actual = canonicalize(black_box(right), CanonicalMode::Database).unwrap();
            assert!(black_box(actual == baseline));
        },
        samples,
        min_ms,
    );
    Ok(serde_json::json!({
        "compare": compare_time, "canonical_pair": pair_time,
        "canonical_cached": cached_time, "byte_compare": bytes_time,
        "canonical_bytes": canonical_bytes,
    }))
}
