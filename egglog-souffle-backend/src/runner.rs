//! Run an emitted Souffle program through the souffle binary, then parse
//! its stdout back into a normalized form that mirrors egglog's outputs
//! (printsize, etc.). Used by the tests/souffle_files harness to compare
//! souffle results against native egglog snapshots.

use std::path::Path;
use std::process::Command;

use crate::ir::Program;
use crate::emit;

/// Maps user-facing constructor names to the souffle relation names that
/// hold their canonical view tables. Constructed by the translator.
#[derive(Default, Debug, Clone)]
pub struct Manifest {
    /// `(user_constructor_name, souffle_view_relation_name)` pairs.
    pub view_relations: Vec<(String, String)>,
    /// `__check_K` query relations emitted from `(check ...)` commands.
    /// Each is empty iff the corresponding check failed; the runner
    /// turns a non-empty count into "passed" and 0 into "failed".
    pub check_relations: Vec<String>,
}

/// Output of [`run`] — printsize results in the same form egglog uses
/// internally, ready to be compared against captured native output.
#[derive(Default, Debug, Clone)]
pub struct RunOutput {
    /// `(user_constructor_name, size)` for each view we mapped from
    /// souffle's `.printsize` lines.
    pub view_sizes: Vec<(String, usize)>,
    /// Indexed by check id. `true` means the check passed (its query
    /// relation was non-empty); `false` means it failed.
    pub check_results: Vec<bool>,
    /// Raw stdout — kept around for debugging when something doesn't parse.
    pub raw_stdout: String,
}

#[derive(Debug)]
pub enum RunError {
    NoSouffleBinary,
    Spawn(String),
    Souffle { exit_code: Option<i32>, stderr: String, dl: String },
    Io(String),
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::NoSouffleBinary => write!(f, "souffle binary not found"),
            RunError::Spawn(s) => write!(f, "failed to spawn souffle: {s}"),
            RunError::Souffle { exit_code, stderr, dl } => write!(
                f,
                "souffle failed (exit {exit_code:?}):\nstderr:\n{stderr}\n--- emitted .dl ---\n{dl}"
            ),
            RunError::Io(s) => write!(f, "io error: {s}"),
        }
    }
}

impl std::error::Error for RunError {}

/// Find the souffle binary either via `SOUFFLE_BIN` or the dev-machine default.
pub fn find_souffle_binary() -> Option<String> {
    if let Ok(b) = std::env::var("SOUFFLE_BIN") {
        return Some(b);
    }
    let default = "/Users/oflatt/souffle/build/src/souffle";
    if Path::new(default).exists() {
        Some(default.to_string())
    } else {
        None
    }
}

/// Emit `program` to a temp .dl file, run souffle on it (with a 10s
/// timeout to guard against runaway loops while we iterate), then parse
/// stdout via `manifest` to produce a [`RunOutput`].
pub fn run(program: &Program, manifest: &Manifest) -> Result<RunOutput, RunError> {
    let bin = find_souffle_binary().ok_or(RunError::NoSouffleBinary)?;
    let dl = emit(program);

    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| RunError::Io(e.to_string()))?
        .as_nanos();
    let path = format!("/tmp/souffle-runner-{pid}-{nanos}.dl");
    std::fs::write(&path, &dl).map_err(|e| RunError::Io(e.to_string()))?;

    let output = Command::new("timeout")
        .arg("60")
        .arg(&bin)
        .arg(&path)
        .output()
        .map_err(|e| RunError::Spawn(e.to_string()))?;

    if !output.status.success() {
        return Err(RunError::Souffle {
            exit_code: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            dl,
        });
    }

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    Ok(parse_stdout(&stdout, manifest))
}

/// Parse souffle's printsize stdout (`<relation>\t<size>` per line) into
/// a `RunOutput`, translating relation names back to user constructor
/// names via `manifest`.
pub fn parse_stdout(stdout: &str, manifest: &Manifest) -> RunOutput {
    // The `outer-saturate` loop emits a fresh `.printsize` line per
    // outer iteration, so we keep the *last* size we see for each
    // relation — that's the value at fixpoint.
    let mut view_size_by_user: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut check_size_by_name: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for line in stdout.lines() {
        let Some((rel, n)) = line.split_once('\t') else {
            continue;
        };
        let Ok(size) = n.trim().parse::<usize>() else {
            continue;
        };
        if let Some((user, _)) = manifest
            .view_relations
            .iter()
            .find(|(_, sf_name)| sf_name == rel)
        {
            view_size_by_user.insert(user.clone(), size);
        }
        if manifest.check_relations.iter().any(|c| c == rel) {
            check_size_by_name.insert(rel.to_string(), size);
        }
    }
    let mut view_sizes: Vec<(String, usize)> = view_size_by_user.into_iter().collect();
    view_sizes.sort_by(|a, b| a.0.cmp(&b.0));
    let check_results: Vec<bool> = manifest
        .check_relations
        .iter()
        .map(|c| check_size_by_name.get(c).copied().unwrap_or(0) > 0)
        .collect();
    RunOutput {
        view_sizes,
        check_results,
        raw_stdout: stdout.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_stdout_strips_unknown_relations() {
        let manifest = Manifest {
            view_relations: vec![
                ("Add".into(), "Eg_AddView".into()),
                ("Mul".into(), "Eg_MulView".into()),
            ],
        };
        let stdout = "Eg_AddView\t3\nEg_MulView\t1\nEg_UF_Math\t4\n";
        let out = parse_stdout(stdout, &manifest);
        assert_eq!(
            out.view_sizes,
            vec![("Add".into(), 3), ("Mul".into(), 1)]
        );
    }
}
