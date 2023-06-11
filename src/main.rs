use clap::Parser;
use egglog::EGraph;
use std::io::{self, BufRead, BufReader};
use std::path::PathBuf;

#[derive(Debug, Parser)]
struct Args {
    #[clap(short = 'F', long)]
    fact_directory: Option<PathBuf>,
    #[clap(long)]
    naive: bool,
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

    let mk_egraph = || {
        let mut egraph = EGraph::default();
        egraph.fact_directory = args.fact_directory.clone();
        egraph.seminaive = !args.naive;
        egraph
    };

    if args.inputs.is_empty() {
        let stdin = io::stdin();
        log::info!("Welcome to Egglog!");
        let mut egraph = mk_egraph();

        for line in BufReader::new(stdin).lines() {
            match line {
                Ok(line_str) => match egraph.parse_and_run_program(&line_str) {
                    Ok(_msgs) => {}
                    Err(err) => {
                        log::error!("{}", err);
                    }
                },
                Err(err) => {
                    log::error!("{}", err);
                    std::process::exit(1)
                }
            }
            log::logger().flush();
            if egraph.is_interactive_mode() {
                eprintln!("(done)");
            }
        }

        std::process::exit(1)
    }

    for (idx, input) in args.inputs.iter().enumerate() {
        let s = std::fs::read_to_string(input).unwrap_or_else(|_| {
            let arg = input.to_string_lossy();
            panic!("Failed to read file {arg}")
        });
        let mut egraph = mk_egraph();
        match egraph.parse_and_run_program(&s) {
            Ok(_msgs) => {}
            Err(err) => {
                log::error!("{}", err);
                std::process::exit(1)
            }
        }

        // no need to drop the egraph if we are going to exit
        if idx == args.inputs.len() - 1 {
            std::mem::forget(egraph)
        }
    }
}
