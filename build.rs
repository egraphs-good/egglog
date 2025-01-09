#[allow(clippy::disallowed_macros)] // for println!
fn main() {
    use std::{env, process::Command};

    let git_hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .map(|output| {
            String::from_utf8(output.stdout)
                .map(|s| "_".to_owned() + &s)
                .unwrap_or_default()
        })
        .unwrap_or_default();
    let build_date = chrono::Utc::now().format("%Y-%m-%d");
    let version = env::var("CARGO_PKG_VERSION").unwrap();
    let full_version = format!("{}_{}{}", version, build_date, git_hash);
    println!("cargo:rustc-env=FULL_VERSION={}", full_version);
}
