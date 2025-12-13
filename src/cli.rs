use crate::*;
use std::io::{self, BufRead, BufReader, IsTerminal, Read, Write};

use clap::Parser;
use env_logger::Env;
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
    mode: RunMode,
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
    /// Enables proof generation and provenance tracking
    #[clap(long)]
    enable_proofs: bool,
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
    #[clap(short = 'j', long, default_value = "1")]
    /// Number of threads to use for parallel execution. Passing `0` will use the maximum
    /// inferred parallelism available on the current system.
    threads: usize,
    #[arg(value_enum)]
    #[clap(long, default_value_t = ReportLevel::TimeOnly)]
    report_level: ReportLevel,
    #[clap(long)]
    save_report: Option<PathBuf>,
    /// Treat missing `$` prefixes on globals as errors instead of warnings
    #[clap(long = "strict-mode")]
    strict_mode: bool,
}

/// Start a command-line interface for the E-graph.
///
/// This is what vanilla egglog uses, and custom egglog builds (i.e., "egglog batteries included")
/// should also call this function.
#[allow(clippy::disallowed_macros)]
pub fn cli(mut egraph: EGraph) {
    env_logger::Builder::from_env(Env::default().default_filter_or("warn"))
        .format_timestamp(None)
        .format_target(false)
        .parse_default_env()
        .init();

    let args = Args::parse();
    let threads = args.threads;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .ok();
    if args.enable_proofs && !egraph.proofs_enabled() {
        // NB: this clears any previous settings and state from the e-graph. It is not generally
        // safe to enable proofs mid-stream and given the way `cli` is invoked by main, this is not
        // a problem.
        egraph = EGraph::with_proofs();
    }
    log::debug!(
        "Initialized thread pool with {} threads",
        rayon::current_num_threads()
    );
    egraph.fact_directory.clone_from(&args.fact_directory);
    egraph.seminaive = !args.naive;
    egraph.set_report_level(args.report_level);
    if args.strict_mode {
        egraph.set_strict_mode(true);
    }
    if args.inputs.is_empty() {
        match egraph.repl(args.mode) {
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

            match run_commands(
                &mut egraph,
                Some(input.to_str().unwrap().into()),
                &program,
                io::stdout(),
                args.mode,
            ) {
                Ok(None) => {}
                _ => std::process::exit(1),
            }

            if args.to_json || args.to_dot || args.to_svg {
                let serialized_output = egraph.serialize(SerializeConfig {
                    max_functions: Some(args.max_functions),
                    max_calls_per_function: Some(args.max_calls_per_function),
                    ..SerializeConfig::default()
                });
                if !serialized_output.is_complete() {
                    log::warn!("{}", serialized_output.omitted_description());
                }
                let mut serialized = serialized_output.egraph;
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
                        .unwrap_or_else(|_| panic!("Failed to write json file to {json_path:?}"));
                }
            }
        }
    }

    if let Some(report_path) = args.save_report {
        let report = egraph.get_overall_run_report();
        serde_json::to_writer(
            std::fs::File::create(&report_path)
                .unwrap_or_else(|_| panic!("Failed to create report file at {report_path:?}")),
            &report,
        )
        .expect("Failed to serialize report");
        log::info!("Saved report to {report_path:?}");
    }

    // no need to drop the egraph if we are going to exit
    std::mem::forget(egraph)
}

impl EGraph {
    /// Start a Read-Eval-Print Loop with standard I/O.
    pub fn repl(&mut self, mode: RunMode) -> io::Result<()> {
        self.repl_with(io::stdin(), io::stdout(), mode, io::stdin().is_terminal())
    }

    /// Start a Read-Eval-Print Loop with the given input and output channel.
    pub fn repl_with<R, W>(
        &mut self,
        input: R,
        mut output: W,
        mode: RunMode,
        is_terminal: bool,
    ) -> io::Result<()>
    where
        R: Read,
        W: Write,
    {
        // https://doc.rust-lang.org/beta/std/io/trait.IsTerminal.html#examples
        if is_terminal {
            output.write_all(welcome_prompt().as_bytes())?;
            output.write_all(b"\n> ")?;
            output.flush()?;
        }
        let mut cmd_buffer = String::new();

        for line in BufReader::new(input).lines() {
            let line_str = line?;
            cmd_buffer.push_str(&line_str);
            cmd_buffer.push('\n');
            // handles multi-line commands
            if should_eval(&cmd_buffer) {
                run_commands(self, None, &cmd_buffer, &mut output, mode)?;
                cmd_buffer = String::new();
                if is_terminal {
                    output.write_all(b"> ")?;
                    output.flush()?;
                }
            }
        }

        if !cmd_buffer.is_empty() {
            run_commands(self, None, &cmd_buffer, &mut output, mode)?;
        }

        Ok(())
    }
}

fn welcome_prompt() -> String {
    format!("Welcome to Egglog REPL! (build: {})", env!("FULL_VERSION"))
}

fn should_eval(curr_cmd: &str) -> bool {
    all_sexps(SexpParser::new(None, curr_cmd)).is_ok()
}

fn run_commands<W>(
    egraph: &mut EGraph,
    filename: Option<String>,
    command: &str,
    mut output: W,
    mode: RunMode,
) -> io::Result<Option<Error>>
where
    W: Write,
{
    if mode == RunMode::ShowDesugaredEgglog {
        return Ok(match egraph.desugar_program(filename, command) {
            Ok(desugared) => {
                for line in desugared {
                    writeln!(output, "{line}")?;
                }
                None
            }
            Err(err) => {
                log::error!("{err}");
                Some(err)
            }
        });
    };

    Ok(match egraph.parse_and_run_program(filename, command) {
        Ok(msgs) => {
            if mode != RunMode::NoMessages {
                for msg in msgs {
                    write!(output, "{msg}")?;
                }
            }
            if mode == RunMode::Interactive {
                writeln!(output, "(done)")?;
            }
            None
        }
        Err(err) => {
            log::error!("{err}");
            if mode == RunMode::Interactive {
                writeln!(output, "(error)")?;
            }
            Some(err)
        }
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum RunMode {
    Normal,
    ShowDesugaredEgglog,
    Interactive,
    NoMessages,
}

impl Display for RunMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            RunMode::Normal => write!(f, "normal"),
            RunMode::ShowDesugaredEgglog => write!(f, "desugar"),
            RunMode::Interactive => write!(f, "interactive"),
            RunMode::NoMessages => write!(f, "no-messages"),
        }
    }
}

impl FromStr for RunMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "normal" => Ok(RunMode::Normal),
            "desugar" => Ok(RunMode::ShowDesugaredEgglog),
            "interactive" => Ok(RunMode::Interactive),
            "no-messages" => Ok(RunMode::NoMessages),
            _ => Err(format!("Unknown run mode: {s}")),
        }
    }
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
        egraph
            .repl_with(input.as_bytes(), &mut output, RunMode::Normal, false)
            .unwrap();
        assert_eq!(String::from_utf8(output).unwrap(), "1\n");

        let input = "\n\n\n";
        let mut output = Vec::new();
        egraph
            .repl_with(input.as_bytes(), &mut output, RunMode::Normal, false)
            .unwrap();
        assert_eq!(String::from_utf8(output).unwrap(), "");

        let input = "(extract 1)";
        let mut output = Vec::new();
        egraph
            .repl_with(input.as_bytes(), &mut output, RunMode::Interactive, false)
            .unwrap();
        assert_eq!(String::from_utf8(output).unwrap(), "1\n(done)\n");

        let input = "xyz";
        let mut output: Vec<u8> = Vec::new();
        egraph
            .repl_with(input.as_bytes(), &mut output, RunMode::Interactive, false)
            .unwrap();
        assert_eq!(String::from_utf8(output).unwrap(), "(error)\n");

        let input = "(extract 1)";
        let mut output = Vec::new();
        egraph
            .repl_with(
                input.as_bytes(),
                &mut output,
                RunMode::ShowDesugaredEgglog,
                false,
            )
            .unwrap();
        assert_eq!(String::from_utf8(output).unwrap(), "(extract 1 0)\n");

        let input = "(extract 1)";
        let mut output = Vec::new();
        egraph
            .repl_with(input.as_bytes(), &mut output, RunMode::NoMessages, false)
            .unwrap();
        assert_eq!(String::from_utf8(output).unwrap(), "");

        let input = "(extract 1)";
        let mut output = Vec::new();
        egraph
            .repl_with(input.as_bytes(), &mut output, RunMode::Normal, true)
            .unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            format!("{}\n> 1\n> ", welcome_prompt())
        );
    }
}
