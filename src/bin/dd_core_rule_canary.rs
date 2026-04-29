use std::env;
use std::path::PathBuf;

fn main() {
    let mut out = None;
    let mut repo_root = None;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => {
                out = args.next().map(PathBuf::from);
            }
            "--repo-root" => {
                repo_root = args.next().map(PathBuf::from);
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            _ if arg.starts_with("--out=") => {
                out = Some(PathBuf::from(arg.trim_start_matches("--out=")));
            }
            _ if arg.starts_with("--repo-root=") => {
                repo_root = Some(PathBuf::from(arg.trim_start_matches("--repo-root=")));
            }
            _ => {
                eprintln!("unknown argument: {arg}");
                print_help();
                std::process::exit(2);
            }
        }
    }

    let command = env::args().collect::<Vec<_>>().join(" ");
    if let Err(err) = egglog::dd_core_rule_canary::run(repo_root, out, command) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn print_help() {
    println!("usage: dd-core-rule-canary [--repo-root PATH] [--out PATH]");
}
