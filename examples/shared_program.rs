//! Minimal interchange utility: schema, encode, run-json, run-text, or text.
use egglog::{EGraph, program::Program};
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut stdout = std::io::stdout().lock();
    let mut args = std::env::args().skip(1);
    let mode = args
        .next()
        .ok_or("expected schema, encode, run-json, run-text, or text")?;
    if mode == "schema" {
        let schema = serde_json::to_string_pretty(&Program::schema())?;
        match args.next() {
            Some(path) => std::fs::write(path, format!("{schema}\n"))?,
            None => writeln!(stdout, "{schema}")?,
        }
        return Ok(());
    }
    let path = args.next().ok_or("expected an input file")?;
    let input = std::fs::read_to_string(&path)?;
    let program = match mode.as_str() {
        "encode" | "run-text" => Program::parse(Some(path), &input)?,
        "run-json" | "text" => Program::from_json(&input)?,
        _ => return Err("unknown mode".into()),
    };
    match mode.as_str() {
        "encode" => {
            let json = program.to_json()?;
            match args.next() {
                Some(path) => std::fs::write(path, format!("{json}\n"))?,
                None => writeln!(stdout, "{json}")?,
            }
        }
        "text" => write!(stdout, "{}", program.to_replayable_egglog()?)?,
        _ => {
            // Exercise the same serialized boundary for both input languages.
            let imported = Program::from_json(&program.to_json()?)?;
            for output in EGraph::default().run_shared_program(imported)? {
                write!(stdout, "{output}")?;
            }
        }
    }
    Ok(())
}
