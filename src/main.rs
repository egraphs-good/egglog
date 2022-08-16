use egg_smol::EGraph;

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp(None)
        .format_target(false)
        .parse_default_env()
        .init();

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
            match egraph.parse_and_run_program(&s) {
                Ok(msgs) => {
                    for msg in msgs {
                        println!("{}", msg);
                    }
                }
                Err(err) => {
                    log::error!("{}", err);
                    std::process::exit(1)
                }
            }
        }
    }
}
