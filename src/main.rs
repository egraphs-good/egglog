use egg_smol::EGraph;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() <= 1 {
        eprintln!("Pass in some files as arguments");
        std::process::exit(1)
    }

    for arg in &args[1..] {
        if arg.ends_with(".egg") {
            let s = std::fs::read_to_string(arg)
                .unwrap_or_else(|_| panic!("Failed to read file {arg}"));
            let mut egraph = EGraph::default();
            match egraph.run_program(&s) {
                Ok(msgs) => {
                    for msg in msgs {
                        println!("  {}", msg);
                    }
                }
                Err(err) => panic!("Top level error: {err}"),
            }
        }
    }
}
