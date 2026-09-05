use clap::Parser;
use egraph_comparison::{
    CanonicalMode, Certificate, Database, canonicalize, certificate, compare, verify,
};
use std::{
    fs::File,
    io::{BufReader, Write},
    path::PathBuf,
    process::ExitCode,
};

#[derive(Parser)]
#[command(about = "Compare two serialized e-graphs modulo constructor bisimulation")]
struct Args {
    left: PathBuf,
    #[arg(required_unless_present = "canonical", conflicts_with = "canonical")]
    right: Option<PathBuf>,
    /// Write canonical JSON for one input; --terms-only selects the projection.
    #[arg(long, conflicts_with_all = ["certificate", "verify_certificate"])]
    canonical: bool,
    /// Use constructor-term equality for the exit status (still report both).
    #[arg(long)]
    terms_only: bool,
    /// Include a verifiable explanation of disequality in the JSON result.
    #[arg(long, conflicts_with = "verify_certificate")]
    certificate: bool,
    /// Verify a certificate JSON file (0 = valid, 1 = invalid, 2 = input error).
    #[arg(long)]
    verify_certificate: Option<PathBuf>,
}

fn run(args: &Args) -> Result<bool, Box<dyn std::error::Error>> {
    let read = |path: &PathBuf| -> Result<Database, Box<dyn std::error::Error>> {
        // Parsing a slice avoids the per-byte reader adapter. Drop each input
        // buffer before loading the next graph to bound the extra memory.
        let bytes = std::fs::read(path)?;
        Ok(serde_json::from_slice(&bytes)?)
    };
    let left = read(&args.left)?;
    if args.canonical {
        let mode = if args.terms_only {
            CanonicalMode::Terms
        } else {
            CanonicalMode::Database
        };
        std::io::stdout()
            .lock()
            .write_all(&canonicalize(&left, mode)?)?;
        return Ok(true);
    }
    let right = read(args.right.as_ref().ok_or("missing right input")?)?;
    if let Some(path) = &args.verify_certificate {
        let witness: Certificate = serde_json::from_reader(BufReader::new(File::open(path)?))?;
        let valid = verify(&witness, &left, &right)?;
        serde_json::to_writer_pretty(
            std::io::stdout().lock(),
            &serde_json::json!({"valid": valid}),
        )?;
        return Ok(valid);
    }
    let result = compare(&left, &right)?;
    let mut output = serde_json::to_value(&result)?;
    if args.certificate {
        output["certificate"] = serde_json::to_value(certificate(&left, &right)?)?;
    }
    serde_json::to_writer_pretty(std::io::stdout().lock(), &output)?;
    Ok(if args.terms_only {
        result.terms_equal
    } else {
        result.database_equal
    })
}

#[allow(clippy::disallowed_macros)]
fn main() -> ExitCode {
    match run(&Args::parse()) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("egraph-comparison: {error}");
            ExitCode::from(2)
        }
    }
}
