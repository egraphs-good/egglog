use clap::Parser;
use egg_smol::{CompilerPassStop, EGraph};
use std::io::{self, Read};
use std::path::PathBuf;

#[derive(Debug, Parser)]
struct Args {
    #[clap(short = 'F', long)]
    fact_directory: Option<PathBuf>,
    #[clap(long)]
    naive: bool,
    #[clap(long)]
    desugar: bool,
    #[clap(long)]
    resugar: bool,
    #[clap(long)]
    proofs: bool,
    #[clap(long, default_value_t = CompilerPassStop::All)]
    stop: CompilerPassStop,
    // TODO remove this evil hack
    #[clap(long, default_value_t = 3)]
    num_underscores: usize,
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
        egraph.set_underscores_for_desugaring(args.num_underscores);
        egraph.fact_directory = args.fact_directory.clone();
        egraph.seminaive = !args.naive;
        egraph
    };

    if args.inputs.is_empty() {
        let stdin = io::stdin();
        log::info!("Welcome to Egglog!");
        let mut egraph = mk_egraph();
        if args.proofs {
            egraph
                .parse_and_run_program("(set-option enable_proofs 1)")
                .unwrap();
        }
        let mut program = String::new();
        stdin
            .lock()
            .read_to_string(&mut program)
            .unwrap_or_else(|_| panic!("Failed to read program from stdin"));
        match egraph.parse_and_run_program(&program) {
            Ok(_msgs) => {}
            Err(err) => {
                log::error!("{}", err);
            }
        }

        std::process::exit(1)
    }

    for (idx, input) in args.inputs.iter().enumerate() {
        let program_read = std::fs::read_to_string(input).unwrap_or_else(|_| {
            let arg = input.to_string_lossy();
            panic!("Failed to read file {arg}")
        });
        let mut egraph = mk_egraph();
        let already_enables = program_read.starts_with("(set-option enable_proofs 1)");
        let program = if args.proofs && !already_enables {
            format!("(set-option enable_proofs 1)\n{}", program_read)
        } else {
            program_read
        };

        if args.desugar || args.resugar {
            let parsed = egraph.parse_program(&program).unwrap();
            let desugared_str = egraph
                .process_commands(parsed, args.stop)
                .unwrap()
                .into_iter()
                .map(|x| {
                    if args.resugar {
                        x.resugar().to_string()
                    } else {
                        x.to_string()
                    }
                })
                .collect::<Vec<String>>()
                .join("\n");
            println!("{}", desugared_str);
        } else {
            match egraph.parse_and_run_program(&program) {
                Ok(_msgs) => {}
                Err(err) => {
                    log::error!("{}", err);
                    std::process::exit(1)
                }
            }
        }

        // no need to drop the egraph if we are going to exit
        if idx == args.inputs.len() - 1 {
            std::mem::forget(egraph)
        }
    }
}
