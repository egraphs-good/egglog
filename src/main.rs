use clap::Parser;
use egglog::{CompilerPassStop, EGraph, Error, SerializeConfig};
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
    /// Use the rust backend implimentation of eqsat,
    /// including a rust implementation of the union-find
    /// data structure and the rust implementation of
    /// the rebuilding algorithm (maintains congruence closure).
    #[clap(long)]
    terms_encoding: bool,
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
    #[clap(long)]
    serialize_split_primitive_outputs: bool,
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
        if args.terms_encoding {
            egraph.enable_terms_encoding();
        }
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
        let (program, program_offset) = if args.proofs && !already_enables {
            let expr = "(set-option enable_proofs 1)\n";
            (format!("{}{}", expr, program_read), expr.len())
        } else {
            (program_read, 0)
        };

        if args.desugar || args.resugar {
            let parsed = egraph.parse_program(&program).unwrap();
            let desugared_str: String = todo!("revisit this- currently process_commands is private because it returns something resolved");
            // let desugared_str = egraph
            //     .process_commands(parsed, args.stop)
            //     .unwrap()
            //     .into_iter()
            //     .map(|x| {
            //         if args.resugar {
            //             x.resugar().to_string()
            //         } else {
            //             x.to_string()
            //         }
            //     })
            //     .collect::<Vec<String>>()
            //     .join("\n");
            println!("{}", desugared_str);
        } else {
            match egraph.parse_and_run_program(&program) {
                Ok(msgs) => {
                    for msg in msgs {
                        println!("{msg}");
                    }
                }
                Err(err) => {
                    let err = match err {
                        Error::ParseError(err) => err
                            .map_location(|byte_offset| {
                                let byte_offset = byte_offset - program_offset;
                                let (line_num, sum_offset) = std::iter::once(0)
                                    .chain(program[program_offset..].split_inclusive('\n').scan(
                                        0,
                                        |sum_offset, l| {
                                            *sum_offset += l.len();

                                            if *sum_offset > byte_offset {
                                                None
                                            } else {
                                                Some(*sum_offset)
                                            }
                                        },
                                    ))
                                    .enumerate()
                                    .last()
                                    // No panic because of the initial 0
                                    .unwrap();
                                {
                                    format!(
                                        "{}:{}:{}",
                                        input.display(),
                                        line_num + 1,
                                        // TODO: Show utf8 aware character count
                                        byte_offset - sum_offset + 1
                                    )
                                }
                            })
                            .to_string(),
                        err => err.to_string(),
                    };
                    log::error!("{}", err);
                    std::process::exit(1)
                }
            }
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
        if args.to_json {
            let json_path = serialize_filename.with_extension("json");
            let config = SerializeConfig {
                split_primitive_outputs: args.serialize_split_primitive_outputs,
                ..SerializeConfig::default()
            };
            let serialized = egraph.serialize(config);
            serialized.to_json_file(json_path).unwrap();
        }

        if args.to_dot || args.to_svg {
            let serialized = egraph.serialize_for_graphviz(args.serialize_split_primitive_outputs);
            if args.to_dot {
                let dot_path = serialize_filename.with_extension("dot");
                serialized.to_dot_file(dot_path).unwrap()
            }
            if args.to_svg {
                let svg_path = serialize_filename.with_extension("svg");
                serialized.to_svg_file(svg_path).unwrap()
            }
        }
        // no need to drop the egraph if we are going to exit
        if idx == args.inputs.len() - 1 {
            std::mem::forget(egraph)
        }
    }
}
