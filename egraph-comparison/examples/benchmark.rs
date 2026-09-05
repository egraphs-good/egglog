//! Measure parsing, validation, and comparison separately, without process startup.
use clap::Parser;
use egraph_comparison::{Database, compare};
use std::{hint::black_box, path::PathBuf, time::Instant};

#[path = "support/canonical.rs"]
mod canonical;

#[derive(Parser)]
struct Args {
    left: PathBuf,
    right: PathBuf,
    #[arg(long, default_value_t = 7)]
    samples: usize,
    #[arg(long, default_value_t = 100)]
    min_sample_ms: u64,
    /// Repeated comparison for an external sampling profiler.
    #[arg(long)]
    profile_seconds: Option<u64>,
    /// Time one comparison for inputs too large for repeated sampling.
    #[arg(long)]
    single_pass: bool,
    /// Compare pairwise refinement with canonical serialization and byte equality.
    #[arg(long, conflicts_with = "profile_seconds")]
    canonical: bool,
}

fn timings(mut f: impl FnMut(), samples: usize, min_ms: u64) -> serde_json::Value {
    let start = Instant::now();
    f();
    let warmup_ns = start.elapsed().as_nanos().max(1);
    let iterations = (u128::from(min_ms) * 1_000_000 / warmup_ns).clamp(1, 100_000) as usize;
    let mut values = Vec::new();
    for _ in 0..samples {
        let start = Instant::now();
        for _ in 0..iterations {
            f();
        }
        values.push(start.elapsed().as_secs_f64() * 1000.0 / iterations as f64);
    }
    values.sort_by(f64::total_cmp);
    serde_json::json!({"median_ms":values[values.len()/2], "min_ms":values[0],
        "max_ms":values[values.len()-1], "samples":samples,"iterations_per_sample":iterations})
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.samples == 0 {
        return Err("samples must be positive".into());
    }
    let left_bytes = std::fs::read(&args.left)?;
    let right_bytes = std::fs::read(&args.right)?;
    let parse = || -> (Database, Database) {
        (
            serde_json::from_slice(&left_bytes).unwrap(),
            serde_json::from_slice(&right_bytes).unwrap(),
        )
    };
    let parse_started = Instant::now();
    let (left, right) = parse();
    let parse_ms = parse_started.elapsed().as_secs_f64() * 1000.0;
    let compare_started = Instant::now();
    let result = compare(&left, &right)?;
    let compare_ms = compare_started.elapsed().as_secs_f64() * 1000.0;
    // The benchmark corpus is equal with unrelated IDs and row order.
    if !result.database_equal {
        return Err("benchmark inputs must compare equal".into());
    }
    if args.canonical {
        let mut measurement = canonical::measure(
            &left,
            &right,
            args.samples,
            args.min_sample_ms,
            args.single_pass,
            compare_ms,
        )?;
        measurement["left_classes"] = left.classes.len().into();
        measurement["left_rows"] = left.rows.len().into();
        measurement["left_bytes"] = left_bytes.len().into();
        measurement["right_bytes"] = right_bytes.len().into();
        measurement["parse_ms"] = parse_ms.into();
        measurement["result"] = serde_json::to_value(result)?;
        serde_json::to_writer_pretty(std::io::stdout().lock(), &measurement)?;
        return Ok(());
    }
    if args.single_pass {
        serde_json::to_writer_pretty(
            std::io::stdout().lock(),
            &serde_json::json!({
                "left_classes":left.classes.len(),"left_rows":left.rows.len(),
                "left_bytes":left_bytes.len(),"right_bytes":right_bytes.len(),
                "result":result,"compare_ms":compare_ms,"parse_ms":parse_ms,"samples":1,
            }),
        )?;
        return Ok(());
    }
    if let Some(seconds) = args.profile_seconds {
        let start = Instant::now();
        while start.elapsed().as_secs() < seconds {
            black_box(compare(black_box(&left), black_box(&right)).unwrap());
        }
        return Ok(());
    }
    let compare_time = timings(
        || {
            black_box(compare(black_box(&left), black_box(&right)).unwrap());
        },
        args.samples,
        args.min_sample_ms,
    );
    let validation_time = timings(
        || {
            left.validate().unwrap();
            right.validate().unwrap();
        },
        args.samples,
        args.min_sample_ms,
    );
    let parse_time = timings(
        || {
            black_box(parse());
        },
        args.samples,
        args.min_sample_ms,
    );
    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &serde_json::json!({
            "left_classes":left.classes.len(),"right_classes":right.classes.len(),
            "left_rows":left.rows.len(),"right_rows":right.rows.len(),
            "left_bytes":left_bytes.len(),"right_bytes":right_bytes.len(),
            "result":result,"compare":compare_time,"validate":validation_time,"parse":parse_time,
        }),
    )?;
    Ok(())
}
