use std::path::PathBuf;

use clap::Parser;
use egg_smol::EGraph;

#[derive(Debug, Parser)]
struct Args {
    #[clap(short = 'F', long)]
    fact_directory: Option<PathBuf>,
    inputs: Vec<PathBuf>,
}

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp(None)
        .format_target(false)
        .parse_default_env()
        .init();

    let args = Args::parse();

    if args.inputs.is_empty() {
        eprintln!("Pass in some files as arguments");
        std::process::exit(1)
    }

    for input in &args.inputs {
        let s = std::fs::read_to_string(input).unwrap_or_else(|_| {
            let arg = input.to_string_lossy();
            panic!("Failed to read file {arg}")
        });
        let mut egraph = EGraph::default();
        egraph.fact_directory = args.fact_directory.clone();
        match egraph.parse_and_run_program(&s) {
            Ok(msgs) => {
                for msg in msgs {
                    println!("{}", msg);
                }
            }
            Err(err) => {
                log::error!("{}", err);
                std::process::exit(1)
            }
        }
    }
}
