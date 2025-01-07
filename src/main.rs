use clap::Parser;
use egglog::{EGraph, RunMode, SerializeConfig};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(version = env!("FULL_VERSION"), about = env!("CARGO_PKG_DESCRIPTION"))]
pub struct Args {
    #[clap(short = 'F', long)]
    fact_directory: Option<PathBuf>,
    #[clap(long)]
    naive: bool,
    #[clap(long)]
    desugar: bool,
    #[clap(long)]
    resugar: bool,
    #[clap(long, default_value_t = RunMode::Normal)]
    show: RunMode,
    // TODO remove this evil hack
    #[clap(long, default_value = "__")]
    reserved_symbol: String,
    inputs: Vec<PathBuf>,
    #[clap(long)]
    to_json: bool,
    #[clap(long)]
    to_dot: bool,
    #[clap(long)]
    to_svg: bool,
    #[clap(long)]
    serialize_split_primitive_outputs: bool,
    /// Maximum number of function nodes to render in dot/svg output
    #[clap(long, default_value = "40")]
    max_functions: usize,
    /// Maximum number of calls per function to render in dot/svg output
    #[clap(long, default_value = "40")]
    max_calls_per_function: usize,
    /// Number of times to inline leaves
    #[clap(long, default_value = "0")]
    serialize_n_inline_leaves: usize,
    #[clap(long)]
    no_messages: bool,
}

fn main() {
    cli(EGraph::default())
}

#[allow(clippy::disallowed_macros)]
pub fn cli(mut egraph: EGraph) {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp(None)
        .format_target(false)
        .parse_default_env()
        .init();

    let args = Args::parse();
    egraph.set_reserved_symbol(args.reserved_symbol.clone().into());
    egraph.fact_directory.clone_from(&args.fact_directory);
    egraph.seminaive = !args.naive;
    egraph.run_mode = args.show;
    if args.no_messages {
        egraph.disable_messages();
    }

    if args.inputs.is_empty() {
        log::info!("Welcome to Egglog REPL! (build: {})", env!("FULL_VERSION"));
        match egraph.repl() {
            Ok(()) => std::process::exit(0),
            Err(err) => {
                log::error!("{err}");
                std::process::exit(1)
            }
        }
    } else {
        for input in &args.inputs {
            let program = std::fs::read_to_string(input).unwrap_or_else(|_| {
                let arg = input.to_string_lossy();
                panic!("Failed to read file {arg}")
            });

            match egraph.parse_and_run_program(Some(input.to_str().unwrap().into()), &program) {
                Ok(msgs) => {
                    for msg in msgs {
                        println!("{msg}");
                    }
                }
                Err(err) => {
                    log::error!("{err}");
                    std::process::exit(1)
                }
            }

            if args.to_json || args.to_dot || args.to_svg {
                let mut serialized = egraph.serialize(SerializeConfig::default());
                if args.serialize_split_primitive_outputs {
                    serialized.split_classes(|id, _| egraph.from_node_id(id).is_primitive())
                }
                for _ in 0..args.serialize_n_inline_leaves {
                    serialized.inline_leaves();
                }

                // if we are splitting primitive outputs, add `-split` to the end of the file name
                let serialize_filename = if args.serialize_split_primitive_outputs {
                    input.with_file_name(format!(
                        "{}-split",
                        input.file_stem().unwrap().to_str().unwrap()
                    ))
                } else {
                    input.clone()
                };
                if args.to_dot {
                    let dot_path = serialize_filename.with_extension("dot");
                    serialized.to_dot_file(dot_path).unwrap()
                }
                if args.to_svg {
                    let svg_path = serialize_filename.with_extension("svg");
                    serialized.to_svg_file(svg_path).unwrap()
                }
                if args.to_json {
                    let json_path = serialize_filename.with_extension("json");
                    serialized.to_json_file(json_path).unwrap();
                }
            }
        }
    }

    // no need to drop the egraph if we are going to exit
    std::mem::forget(egraph)
}
