use std::{fs, process::Command};

#[test]
fn exit_status_and_json_output() {
    let dir = std::env::temp_dir().join(format!("egraph-comparison-cli-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let empty = dir.join("empty.json");
    let different = dir.join("different.json");
    let invalid = dir.join("invalid.json");
    fs::write(
        &empty,
        r#"{"version":1,"classes":{},"functions":{},"rows":[]}"#,
    )
    .unwrap();
    fs::write(&different, r#"{"version":1,"classes":{},"functions":{"f":{"kind":"function","inputs":[],"output":"E"}},"rows":[]}"#).unwrap();
    fs::write(&invalid, "{").unwrap();
    let run = |a: &std::path::Path, b: &std::path::Path, terms: bool| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_egraph-comparison"));
        cmd.arg(a).arg(b);
        if terms {
            cmd.arg("--terms-only");
        }
        cmd.output().unwrap()
    };
    let same = run(&empty, &empty, false);
    assert!(same.status.success());
    let json: serde_json::Value = serde_json::from_slice(&same.stdout).unwrap();
    assert_eq!(json["database_equal"], true);
    assert_eq!(run(&empty, &different, false).status.code(), Some(1));
    assert!(run(&empty, &different, true).status.success());
    assert_eq!(run(&empty, &invalid, false).status.code(), Some(2));
    assert_eq!(
        run(&empty, &dir.join("missing"), false).status.code(),
        Some(2)
    );
    fs::remove_dir_all(dir).unwrap();
}
