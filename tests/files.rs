use hashbrown::HashSet;
use std::path::{Path, PathBuf};

use egglog::ast::{Command, Parser};
use egglog::*;
use libtest_mimic::Trial;

#[derive(Clone)]
struct Run {
    path: PathBuf,
    desugar: bool,
    proofs: bool,
}

impl Run {
    fn run(&self) {
        let _ = env_logger::builder().is_test(true).try_init();
        let program = std::fs::read_to_string(&self.path)
            .unwrap_or_else(|err| panic!("Couldn't read {:?}: {:?}", self.path, err));

        if !self.desugar {
            self.test_program(
                self.path.to_str().map(String::from),
                &program,
                "Top level error",
            );
        } else {
            let mut egraph = EGraph::default();
            let desugared_str = egraph
                .desugar_program(self.path.to_str().map(String::from), &program)
                .unwrap()
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("\n");

            self.test_program(
                None,
                &desugared_str,
                "ERROR after parse, to_string, and parse again.",
            );
        }
    }

    fn test_program(&self, filename: Option<String>, program: &str, message: &str) {
        let mut egraph = if self.proofs {
            EGraph::with_proofs()
        } else {
            EGraph::default()
        };
        match egraph.parse_and_run_program(filename, program) {
            Ok(msgs) => {
                if self.should_fail() {
                    panic!(
                        "Program should have failed! Instead, logged:\n {}",
                        msgs.iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>()
                            .join("\n")
                    );
                } else {
                    for msg in msgs {
                        log::info!("  {}", msg);
                    }
                    // Test graphviz dot generation
                    let mut serialized = egraph
                        .serialize(SerializeConfig {
                            max_functions: Some(40),
                            max_calls_per_function: Some(40),
                            ..Default::default()
                        })
                        .egraph;
                    serialized.to_dot();
                    // Also try splitting and inlining
                    serialized.split_classes(|id, _| egraph.from_node_id(id).is_primitive());
                    serialized.inline_leaves();
                    serialized.to_dot();
                }
            }
            Err(err) => {
                if !self.should_fail() {
                    panic!("{}: {err}", message)
                }
            }
        };
    }

    fn into_trial(self) -> Trial {
        let name = self.name().to_string();
        Trial::test(name, move || {
            self.run();
            Ok(())
        })
    }

    fn name(&self) -> impl std::fmt::Display + '_ {
        struct Wrapper<'a>(&'a Run);
        impl std::fmt::Display for Wrapper<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                if self.0.path.parent().unwrap().ends_with("fail-typecheck") {
                    write!(f, "fail-typecheck/")?;
                }
                let stem = self.0.path.file_stem().unwrap();
                let stem_str = stem.to_string_lossy().replace(['.', '-', ' '], "_");
                write!(f, "{stem_str}")?;
                if self.0.proofs {
                    write!(f, "_with_proofs")?;
                }
                if self.0.desugar {
                    write!(f, "_desugar")?;
                }
                Ok(())
            }
        }
        Wrapper(self)
    }

    fn should_fail(&self) -> bool {
        self.path.to_string_lossy().contains("fail-typecheck")
    }
}

fn generate_tests(glob: &str) -> Vec<Trial> {
    let mut trials = vec![];
    let mut push_trial = |run: Run| trials.push(run.into_trial());

    for entry in glob::glob(glob).unwrap() {
        let path = entry.unwrap().clone();
        let program = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("Couldn't read {:?}: {:?}", path, err));
        let proofs_ok = proofs_supported(&path, &program);

        let run = Run {
            path: path.clone(),
            desugar: false,
            proofs: false,
        };
        let should_fail = run.should_fail();

        push_trial(run.clone());
        if !should_fail && proofs_ok {
            push_trial(Run {
                proofs: true,
                ..run.clone()
            });
        }
        if !should_fail {
            push_trial(Run {
                desugar: true,
                proofs: false,
                ..run.clone()
            });
        }
    }

    trials
}

fn proofs_supported(path: &Path, program: &str) -> bool {
    let mut parser = Parser::default();
    let mut visited = HashSet::new();
    proofs_supported_inner(&mut parser, path, path, program, &mut visited)
}

fn proofs_supported_inner(
    parser: &mut Parser,
    root: &Path,
    path: &Path,
    program: &str,
    visited: &mut HashSet<PathBuf>,
) -> bool {
    let canonical = path.canonicalize().unwrap_or_else(|_| root.join(path));
    if !visited.insert(canonical.clone()) {
        return true;
    }

    let filename = path.to_string_lossy().into_owned();
    let commands = match parser.get_program_from_string(Some(filename), program) {
        Ok(cmds) => cmds,
        Err(_) => return false,
    };
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));

    commands
        .into_iter()
        .all(|command| command_allows_proofs(parser, root, base_dir, command, visited))
}

fn command_allows_proofs(
    parser: &mut Parser,
    root: &Path,
    base_dir: &Path,
    command: Command,
    visited: &mut HashSet<PathBuf>,
) -> bool {
    match command {
        Command::Function { merge: Some(_), .. } => false,
        Command::Fail(_, inner) => command_allows_proofs(parser, root, base_dir, *inner, visited),
        Command::Include(_, file) => {
            let include_path = {
                let candidate = Path::new(&file);
                if candidate.is_absolute() {
                    candidate.to_path_buf()
                } else {
                    base_dir.join(candidate)
                }
            };
            let Ok(contents) = std::fs::read_to_string(&include_path) else {
                return false;
            };
            proofs_supported_inner(parser, root, &include_path, &contents, visited)
        }
        Command::Rewrite(_, rewrite, _) | Command::BiRewrite(_, rewrite) => {
            expr_allows_proofs(parser, root, base_dir, &rewrite.lhs, visited)
                && expr_allows_proofs(parser, root, base_dir, &rewrite.rhs, visited)
        }
        Command::Rule { rule } => {
            rule.body
                .iter()
                .all(|fact| fact_allows_proofs(parser, root, base_dir, fact, visited))
                && rule.head.iter().all(|action| {
                    let mut ok = true;
                    action.clone().visit_exprs(&mut |expr| {
                        ok &= expr_allows_proofs(parser, root, base_dir, &expr, visited);
                        expr
                    });
                    ok
                })
        }
        _ => true,
    }
}

fn fact_allows_proofs(
    parser: &mut Parser,
    root: &Path,
    base_dir: &Path,
    fact: &egglog::ast::Fact,
    visited: &mut HashSet<PathBuf>,
) -> bool {
    match fact {
        egglog::ast::Fact::Fact(expr) => expr_allows_proofs(parser, root, base_dir, expr, visited),
        egglog::ast::Fact::Eq(_, lhs, rhs) => {
            expr_allows_proofs(parser, root, base_dir, lhs, visited)
                && expr_allows_proofs(parser, root, base_dir, rhs, visited)
        }
    }
}

fn expr_allows_proofs(
    _parser: &mut Parser,
    _root: &Path,
    _base_dir: &Path,
    expr: &egglog::ast::Expr,
    _visited: &mut HashSet<PathBuf>,
) -> bool {
    match expr {
        egglog::ast::Expr::Call(_, head, args) => {
            if head.starts_with("unstable-") || head == "+" {
                return false;
            }
            args.iter()
                .all(|arg| expr_allows_proofs(_parser, _root, _base_dir, arg, _visited))
        }
        _ => true,
    }
}

fn main() {
    let args = libtest_mimic::Arguments::from_args();
    let tests = generate_tests("tests/**/*.egg");
    libtest_mimic::run(&args, tests).exit();
}
