//! Time one pre-parsed schedule on the single-threaded Rust baseline.
//! See lean/bench/compare.py for the checked Lean side and workload generation.
#[path = "lean_comparison/compiler.rs"]
mod compiler;
#[path = "lean_comparison/luminal.rs"]
mod luminal;
use egglog::EGraph;
use std::io::Write;
use std::{env, error::Error, fs, io, time::Instant};

#[cfg(feature = "bin")]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if let [mode, path] = args.as_slice()
        && mode == "luminal"
    {
        return luminal::run(path);
    }
    if let [mode, setup, schedule, input, expected] = args.as_slice()
        && mode == "compiler"
    {
        return compiler::run(setup, schedule, input, expected);
    }
    let [mode, setup, schedule, checks] = args.as_slice() else {
        return Err(io::Error::other("expected plain|proofs SETUP RUN CHECKS").into());
    };
    let mut graph = match mode.as_str() {
        "plain" | "signal" => EGraph::new(1),
        "proofs" => EGraph::new_with_proofs().with_num_threads(1),
        _ => return Err(io::Error::other("expected plain or proofs").into()),
    };
    graph.parse_and_run_program(Some(setup.clone()), &fs::read_to_string(setup)?)?;
    let run = graph.parse_program(Some(schedule.clone()), &fs::read_to_string(schedule)?)?;
    let mut stdout = io::stdout().lock();
    if mode == "signal" {
        writeln!(stdout, "ready")?;
        stdout.flush()?;
        let mut go = String::new();
        io::stdin().read_line(&mut go)?;
        if go != "go\n" {
            return Err(io::Error::other("expected benchmark start signal").into());
        }
    }
    let start = Instant::now();
    graph.run_program(run)?;
    let elapsed = start.elapsed().as_nanos();
    if mode == "signal" {
        writeln!(stdout, "done")?;
        stdout.flush()?;
    }
    for output in graph.parse_and_run_program(Some(checks.clone()), &fs::read_to_string(checks)?)? {
        write!(stdout, "{output}")?;
    }
    writeln!(stdout, "schedule_ns={elapsed}")?;
    Ok(())
}
