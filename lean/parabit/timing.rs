// Link to the unmodified, pinned CAV release build. The only warm-up is the
// clock used by egg itself, which initializes once per compiler process.
use parabit::{Equivalence, EquivalenceString};
use std::{env, error::Error, fs, hint::black_box, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let mode = args.next().ok_or("expected track|explain")?;
    let file = args.next().ok_or("expected a .bwlang file")?;
    let input: EquivalenceString = serde_json::from_slice(&fs::read(file)?)?;
    let query = Equivalence::from(input);
    query.validate()?;

    black_box(quanta::Instant::now());
    let start = Instant::now();
    let solved = query.find_equivalence(&None);
    let found = solved.equiv.ok_or("missing solver outcome")?;
    let runner_ns = (solved.runner.report().total_time * 1e9) as u64;
    // Preserve time-to-equivalence if reconstruction reaches the external limit.
    if mode == "coverage" {
        eprintln!("search={} {}", found, start.elapsed().as_nanos());
    }
    match mode.as_str() {
        "track" => {}
        "explain" | "coverage" => {
            if found {
                // These are the same calls made by the upstream CLI. make_proof
                // checks the proof both before and after flattening/reduction.
                black_box(solved.make_proof().explanation_string());
            }
        }
        _ => return Err("expected track|explain".into()),
    }
    let solver_ns = start.elapsed().as_nanos();
    println!("{}", if found { "solved" } else { "unproved" });
    println!("solver_ns={solver_ns}");
    println!("runner_ns={runner_ns}");
    Ok(())
}
