use clap::Parser;
use egraph_comparison::{Database, compare};
use std::{fs::File, io::BufReader, path::PathBuf, process::ExitCode};

#[derive(Parser)]
#[command(about = "Compare two serialized e-graphs modulo constructor bisimulation")]
struct Args {
    left: PathBuf,
    right: PathBuf,
    /// Use constructor-term equality for the exit status (still report both).
    #[arg(long)]
    terms_only: bool,
}

fn run(args: &Args) -> Result<bool, Box<dyn std::error::Error>> {
    let read = |path: &PathBuf| -> Result<Database, Box<dyn std::error::Error>> {
        Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
    };
    let result = compare(&read(&args.left)?, &read(&args.right)?)?;
    serde_json::to_writer_pretty(std::io::stdout().lock(), &result)?;
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
