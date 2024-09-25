use std::{env, process::Command};

fn main() {
    lalrpop::process_root().unwrap();
    let output = Command::new("git")
        .args(&["rev-parse", "--short", "HEAD"])
        .output()
        .unwrap();
    let git_hash = String::from_utf8(output.stdout).unwrap();
    let build_date = chrono::Utc::now().format("%Y-%m-%d");
    let version = env::var("CARGO_PKG_VERSION").unwrap();
    let full_version = format!("{}_{}_{}", version, build_date, git_hash);
    println!("cargo:rustc-env=FULL_VERSION={}", full_version);
}
