use crate::*;
use std::io::{self, BufRead, BufReader, Read, Write};

#[cfg(feature = "bin")]
pub mod bin {
    use super::*;
    use clap::Parser;
    use std::path::PathBuf;

    #[derive(Debug, Parser)]
    #[command(version = env!("FULL_VERSION"), about = env!("CARGO_PKG_DESCRIPTION"))]
    struct Args {
        /// Directory for files when using `input` and `output` commands
        #[clap(short = 'F', long)]
        fact_directory: Option<PathBuf>,
        /// Turns off the seminaive optimization
        #[clap(long)]
        naive: bool,
        /// Prints extra information, which can be useful for debugging
        #[clap(long, default_value_t = RunMode::Normal)]
        show: RunMode,
        /// The file names for the egglog files to run
        inputs: Vec<PathBuf>,
        /// Serializes the egraph for each egglog file as JSON
        #[clap(long)]
        to_json: bool,
        /// Serializes the egraph for each egglog file as a dot file
        #[clap(long)]
        to_dot: bool,
        /// Serializes the egraph for each egglog file as an SVG
        #[clap(long)]
        to_svg: bool,
        /// Splits the serialized egraph into primitives and non-primitives
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
        /// Prevents egglog from printing messages
        #[clap(long)]
        no_messages: bool,

        #[clap(short = 'j', long, default_value = "1")]
        /// Number of threads to use for parallel execution. Passing `0` will use the maximum
        /// inferred parallelism available on the current system.
        threads: usize,
    }

    /// Start a command-line interface for the E-graph.
    ///
    /// This is what vanilla egglog uses, and custom egglog builds (i.e., "egglog batteries included")
    /// should also call this function.
    #[allow(clippy::disallowed_macros)]
    pub fn cli(mut egraph: EGraph) {
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Info)
            .format_timestamp(None)
            .format_target(false)
            .parse_default_env()
            .init();

        let args = Args::parse();
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .unwrap();
        log::debug!(
            "Initialized thread pool with {} threads",
            rayon::current_num_threads()
        );
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
                    let (mut serialized, trimmed_msg) = egraph.serialize(SerializeConfig {
                        max_functions: Some(args.max_functions),
                        max_calls_per_function: Some(args.max_calls_per_function),
                        ..SerializeConfig::default()
                    });
                    if let Some(msg) = trimmed_msg.as_ref() {
                        log::warn!("{msg}");
                    }
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
                        serialized
                            .to_dot_file(dot_path.clone())
                            .unwrap_or_else(|_| panic!("Failed to write dot file to {dot_path:?}"));
                    }
                    if args.to_svg {
                        let svg_path = serialize_filename.with_extension("svg");
                        serialized.to_svg_file(svg_path.clone()).unwrap_or_else( |_|
                            panic!("Failed to write svg file to {svg_path:?}. Make sure you have the `dot` executable installed")
                        );
                    }
                    if args.to_json {
                        let json_path = serialize_filename.with_extension("json");
                        serialized
                            .to_json_file(json_path.clone())
                            .unwrap_or_else(|_| {
                                panic!("Failed to write json file to {json_path:?}")
                            });
                    }
                }
            }
        }

        // no need to drop the egraph if we are going to exit
        std::mem::forget(egraph)
    }
}

impl EGraph {
    /// Start a Read-Eval-Print Loop with standard I/O.
    pub fn repl(&mut self) -> io::Result<()> {
        self.repl_with(io::stdin(), io::stdout())
    }

    /// Start a Read-Eval-Print Loop with the given input and output channel.
    pub fn repl_with<R, W>(&mut self, input: R, mut output: W) -> io::Result<()>
    where
        R: Read,
        W: Write,
    {
        let mut cmd_buffer = String::new();

        for line in BufReader::new(input).lines() {
            let line_str = line?;
            cmd_buffer.push_str(&line_str);
            cmd_buffer.push('\n');
            // handles multi-line commands
            if should_eval(&cmd_buffer) {
                run_command_in_scripting(self, &cmd_buffer, &mut output)?;
                cmd_buffer = String::new();
            }
        }

        if !cmd_buffer.is_empty() {
            run_command_in_scripting(self, &cmd_buffer, &mut output)?;
        }

        Ok(())
    }
}

fn should_eval(curr_cmd: &str) -> bool {
    all_sexps(SexpParser::new(None, curr_cmd)).is_ok()
}

fn run_command_in_scripting<W>(egraph: &mut EGraph, command: &str, mut output: W) -> io::Result<()>
where
    W: Write,
{
    match egraph.parse_and_run_program(None, command) {
        Ok(msgs) => {
            for msg in msgs {
                writeln!(output, "{msg}")?;
            }
        }
        Err(err) => log::error!("{err}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_eval() {
        #[rustfmt::skip]
        let test_cases = vec![
            vec![
                "(extract",
                "\"1",
                ")",
                "(",
                ")))",
                "\"",
                ";; )",
                ")"
            ],
            vec![
                "(extract 1) (extract",
                "2) (",
                "extract 3) (extract 4) ;;;; ("
            ],
            vec![
                "(extract \"\\\")\")"
            ]];
        for test in test_cases {
            let mut cmd_buffer = String::new();
            for (i, line) in test.iter().enumerate() {
                cmd_buffer.push_str(line);
                cmd_buffer.push('\n');
                assert_eq!(should_eval(&cmd_buffer), i == test.len() - 1);
            }
        }
    }

    #[test]
    fn test_repl() {
        let mut egraph = EGraph::default();

        let input = "(extract 1)";
        let mut output = Vec::new();
        egraph.repl_with(input.as_bytes(), &mut output).unwrap();
        assert_eq!(String::from_utf8(output).unwrap(), "1\n");

        let input = "\n\n\n";
        let mut output = Vec::new();
        egraph.repl_with(input.as_bytes(), &mut output).unwrap();
        assert_eq!(String::from_utf8(output).unwrap(), "");

        let input = "(set-option interactive_mode 1)";
        let mut output = Vec::new();
        egraph.repl_with(input.as_bytes(), &mut output).unwrap();
        assert_eq!(String::from_utf8(output).unwrap(), "(done)\n");

        let input = "(set-option interactive_mode 1)\n(extract 1)(extract 2)\n";
        let mut output = Vec::new();
        egraph.repl_with(input.as_bytes(), &mut output).unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            "(done)\n1\n(done)\n2\n(done)\n"
        );
    }
}
