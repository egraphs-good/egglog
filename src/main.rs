use clap::Parser;
use egglog::{CompilerPassStop, EGraph, SerializeConfig};
use std::io::{self, BufRead, BufReader};
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
    #[clap(long)]
    to_json: bool,
    #[clap(long)]
    to_dot: bool,
    #[clap(long)]
    to_svg: bool,
}

#[allow(clippy::disallowed_macros)]
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
        if args.proofs {
            egraph
                .parse_and_run_program("(set-option enable_proofs 1)")
                .unwrap();
        }
        egraph
    };

    if args.inputs.is_empty() {
        let stdin = io::stdin();
        log::info!("Welcome to Egglog!");
        let mut egraph = mk_egraph();

        for line in BufReader::new(stdin).lines() {
            match line {
                Ok(line_str) => match egraph.parse_and_run_program(&line_str) {
                    Ok(msgs) => {
                        for msg in msgs {
                            println!("{msg}");
                        }
                    }
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
                println!("(done)");
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
                Ok(msgs) => {
                    for msg in msgs {
                        println!("{msg}");
                    }
                }
                Err(err) => {
                    log::error!("{}", err);
                    std::process::exit(1)
                }
            }
        }

        if args.to_json {
            let json_path = input.with_extension("json");
            let serialized = egraph.serialize(SerializeConfig::default());
            serialized.to_json_file(json_path).unwrap();
        }

        if args.to_dot || args.to_svg {
            let serialized = egraph.serialize_for_graphviz();
            if args.to_dot {
                let dot_path = input.with_extension("dot");
                serialized.to_dot_file(dot_path).unwrap()
            }
            if args.to_svg {
                let svg_path = input.with_extension("svg");
                serialized.to_svg_file(svg_path).unwrap()
            }
        }
        // no need to drop the egraph if we are going to exit
        if idx == args.inputs.len() - 1 {
            std::mem::forget(egraph)
        }
    }
}
