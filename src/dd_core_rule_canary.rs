use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::ast::{FunctionSubtype, Ruleset};
use crate::core::{GenericAtomTerm, GenericCoreAction};
use crate::{Change, EGraph, ResolvedCall, ResolvedVar};

const EXPERIMENT: &str = "resolved-core-rule-canary";

#[derive(Clone, Copy)]
enum Transform {
    None,
    StripPrintFunction,
    PrimitiveFilterNoPanic,
}

struct CorpusEntry {
    name: &'static str,
    path: &'static str,
    transform: Transform,
}

const CORPUS: &[CorpusEntry] = &[
    CorpusEntry {
        name: "path-no-print",
        path: "tests/web-demo/path.egg",
        transform: Transform::StripPrintFunction,
    },
    CorpusEntry {
        name: "relation-query-allowed",
        path: "tests/relation-query-allowed.egg",
        transform: Transform::None,
    },
    CorpusEntry {
        name: "bool",
        path: "tests/bool.egg",
        transform: Transform::None,
    },
    CorpusEntry {
        name: "i64",
        path: "tests/i64.egg",
        transform: Transform::None,
    },
    CorpusEntry {
        name: "primitives",
        path: "tests/primitives.egg",
        transform: Transform::None,
    },
    CorpusEntry {
        name: "primitive-filter-no-panic",
        path: "tests/repro-primitive-query.egg",
        transform: Transform::PrimitiveFilterNoPanic,
    },
    CorpusEntry {
        name: "repro-primitive-query",
        path: "tests/repro-primitive-query.egg",
        transform: Transform::None,
    },
];

pub fn run(
    repo_root: Option<PathBuf>,
    out: Option<PathBuf>,
    command: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let repo_root = match repo_root {
        Some(path) => path,
        None => default_repo_root()?,
    };
    let report = build_report(&repo_root, command);

    if let Some(path) = out {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(
            path,
            format!("{}\n", serde_json::to_string_pretty(&report)?),
        )?;
    } else {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }

    Ok(())
}

fn default_repo_root() -> Result<PathBuf, String> {
    let cwd = std::env::current_dir().map_err(|err| err.to_string())?;
    if cwd.join("repos/egglog/tests").is_dir() {
        return Ok(cwd.join("repos/egglog"));
    }
    if cwd.join("tests").is_dir() && cwd.join("src/lib.rs").is_file() {
        return Ok(cwd);
    }
    Err(format!(
        "could not infer egglog repo root from {}; pass --repo-root",
        cwd.display()
    ))
}

fn build_report(repo_root: &Path, command: String) -> Value {
    let mut totals = Counters::default();
    let mut corpus = Vec::new();
    let mut failed_entries = 0_u64;
    let mut entries_with_rules = 0_u64;

    for entry in CORPUS {
        let path = repo_root.join(entry.path);
        let mut entry_counters = Counters::default();
        let source_result = fs::read_to_string(&path);
        let mut rules = Vec::new();
        let parse_run = match source_result {
            Ok(source) => {
                let source = transform_source(source, entry.transform);
                let mut egraph = EGraph::default();
                match egraph.parse_and_run_program(Some(path.display().to_string()), &source) {
                    Ok(outputs) => {
                        let parse_run = json!({
                            "status": "ok",
                            "outputs": outputs.len(),
                        });
                        rules = summarize_rules(&egraph, &mut entry_counters);
                        parse_run
                    }
                    Err(err) => {
                        failed_entries += 1;
                        json!({
                            "status": "error",
                            "error": err.to_string(),
                        })
                    }
                }
            }
            Err(err) => {
                failed_entries += 1;
                json!({
                    "status": "error",
                    "error": err.to_string(),
                })
            }
        };

        if entry_counters.get("rules_lowered") > 0 {
            entries_with_rules += 1;
        }
        totals.merge(&entry_counters);

        corpus.push(json!({
            "name": entry.name,
            "path": entry.path,
            "transform": transform_name(entry.transform),
            "parse_run": parse_run,
            "rule_count": entry_counters.get("rules_lowered"),
            "unsupported_rule_count": entry_counters.get("unsupported_rules"),
            "conditional_rule_count": entry_counters.get("conditional_rules"),
            "metrics": entry_counters.into_json(),
            "rules": rules,
        }));
    }

    let status = if failed_entries > 0 {
        "fail"
    } else if totals.get("unsupported_rules") > 0 || totals.get("conditional_rules") > 0 {
        "pass-with-findings"
    } else {
        "pass"
    };

    json!({
        "experiment": EXPERIMENT,
        "status": status,
        "command": command,
        "corpus": corpus,
        "metrics": {
            "entries": CORPUS.len(),
            "entries_with_rules": entries_with_rules,
            "failed_entries": failed_entries,
            "totals": totals.into_json(),
        },
        "observations": [
            "The Gate 1 corpus can be parsed, run, and lowered through the existing egglog frontend into stored ResolvedCoreRule values.",
            "The actual rule corpus is smaller than the file list suggests: i64.egg and primitives.egg contain only top-level checks, not rules.",
            "The lowered rule shapes cover relation joins, constructor/function atoms, repeated-variable/literal arguments, pure primitive query atoms, relation/custom-function Set actions, a no-panic primitive-filter fixture, and one Panic action from the full negative test.",
            "repro-primitive-query.egg is useful as a negative semantic check, but it is not an execution-only DD scaffold fixture because its lowered action is Panic."
        ],
        "decision": "Use this canary as the first scaffold corpus. Keep path-no-print, relation-query-allowed, bool, and a reduced primitive-filter fixture in Gate 1; move full repro-primitive-query.egg behind explicit Panic support or replace it with a no-panic primitive-filter fixture.",
        "limitations": [
            "This canary classifies real lowered CoreRule shapes but does not compile or execute DD fragments.",
            "Primitive purity is not visible in the current public primitive API, so primitive query atoms are marked conditional on a purity/admission check.",
            "The corpus is Gate 1 only; it does not exercise equality/rebuild, delete/subsume, containers, scheduler admission, extraction, proof, push/pop, or serialization."
        ],
        "next_action": "Make the first DD scaffold acceptance require this canary plus one DD lowering pass for all supported Gate 1 rules, with repro-primitive-query either reduced or gated on Panic support."
    })
}

fn transform_source(source: String, transform: Transform) -> String {
    match transform {
        Transform::None => source,
        Transform::StripPrintFunction => {
            let mut transformed = source
                .lines()
                .filter(|line| !line.trim_start().starts_with("(print-function"))
                .collect::<Vec<_>>()
                .join("\n");
            transformed.push('\n');
            transformed
        }
        Transform::PrimitiveFilterNoPanic => {
            let transformed = source.replace(
                "((panic \"should not have matched\"))",
                "((primitive-filter-hit))",
            );
            format!("(relation primitive-filter-hit ())\n{transformed}")
        }
    }
}

fn transform_name(transform: Transform) -> &'static str {
    match transform {
        Transform::None => "none",
        Transform::StripPrintFunction => "strip-print-function",
        Transform::PrimitiveFilterNoPanic => "primitive-filter-no-panic",
    }
}

fn summarize_rules(egraph: &EGraph, counters: &mut Counters) -> Vec<Value> {
    let mut rules = Vec::new();

    for (ruleset_name, ruleset) in &egraph.rulesets {
        let Ruleset::Rules(rule_map) = ruleset else {
            continue;
        };

        for (rule_name, (core_rule, _rule_id)) in rule_map {
            counters.inc("rules_lowered");
            let mut rule_issues = Vec::new();
            let mut rule_conditionals = Vec::new();

            let body_atoms = core_rule
                .body
                .atoms
                .iter()
                .map(|atom| {
                    counters.inc("body_atoms");
                    let call = summarize_call(&atom.head);
                    match &atom.head {
                        ResolvedCall::Func(_) => counters.inc("body_function_atoms"),
                        ResolvedCall::Primitive(_) => {
                            counters.inc("body_primitive_atoms");
                            rule_conditionals.push(
                                "query primitive requires pure/admitted primitive metadata"
                                    .to_string(),
                            );
                        }
                    }
                    for arg in &atom.args {
                        if !term_supported(arg) {
                            counters.inc("unsupported_terms");
                            rule_issues.push("global atom term survived lowering".to_string());
                        }
                    }
                    json!({
                        "call": call,
                        "args": atom.args.iter().map(summarize_term).collect::<Vec<_>>(),
                    })
                })
                .collect::<Vec<_>>();

            let actions = core_rule
                .head
                .0
                .iter()
                .map(|action| summarize_action(action, counters, &mut rule_issues))
                .collect::<Vec<_>>();

            let scaffold_status = if !rule_issues.is_empty() {
                counters.inc("unsupported_rules");
                "unsupported"
            } else if !rule_conditionals.is_empty() {
                counters.inc("conditional_rules");
                "conditional"
            } else {
                "supported"
            };

            rules.push(json!({
                "ruleset": ruleset_name,
                "rule": rule_name,
                "span": core_rule.span.to_string(),
                "scaffold_status": scaffold_status,
                "conditional_requirements": rule_conditionals,
                "unsupported": rule_issues,
                "body_atoms": body_atoms,
                "actions": actions,
            }));
        }
    }

    rules
}

fn summarize_action(
    action: &GenericCoreAction<ResolvedCall, ResolvedVar>,
    counters: &mut Counters,
    rule_issues: &mut Vec<String>,
) -> Value {
    match action {
        GenericCoreAction::Let(_span, var, call, args) => {
            counters.inc("action_let");
            json!({
                "kind": "Let",
                "var": var.to_string(),
                "call": summarize_call(call),
                "args": args.iter().map(summarize_term).collect::<Vec<_>>(),
                "scaffold_status": "host-eval",
            })
        }
        GenericCoreAction::LetAtomTerm(_span, var, term) => {
            counters.inc("action_let_atom_term");
            json!({
                "kind": "LetAtomTerm",
                "var": var.to_string(),
                "term": summarize_term(term),
                "scaffold_status": if term_supported(term) { "supported" } else { "unsupported" },
            })
        }
        GenericCoreAction::Set(_span, call, args, value) => {
            counters.inc("action_set");
            json!({
                "kind": "Set",
                "call": summarize_call(call),
                "args": args.iter().map(summarize_term).collect::<Vec<_>>(),
                "value": summarize_term(value),
                "scaffold_status": "supported-host-action",
            })
        }
        GenericCoreAction::Change(_span, change, call, args) => {
            match change {
                Change::Delete => counters.inc("action_change_delete"),
                Change::Subsume => counters.inc("action_change_subsume"),
            }
            json!({
                "kind": "Change",
                "change": format!("{change:?}"),
                "call": summarize_call(call),
                "args": args.iter().map(summarize_term).collect::<Vec<_>>(),
                "scaffold_status": "abi-only",
            })
        }
        GenericCoreAction::Union(_span, lhs, rhs) => {
            counters.inc("action_union");
            rule_issues.push("Union action belongs to Gate 2 equality/direct rewrites".to_string());
            json!({
                "kind": "Union",
                "lhs": summarize_term(lhs),
                "rhs": summarize_term(rhs),
                "scaffold_status": "unsupported-gate-1",
            })
        }
        GenericCoreAction::Panic(_span, message) => {
            counters.inc("action_panic");
            rule_issues.push("Panic action is not in the first DD scaffold action set".to_string());
            json!({
                "kind": "Panic",
                "message": message,
                "scaffold_status": "unsupported-gate-1",
            })
        }
    }
}

fn summarize_call(call: &ResolvedCall) -> Value {
    match call {
        ResolvedCall::Func(func) => {
            let subtype = match func.subtype {
                FunctionSubtype::Constructor => "constructor",
                FunctionSubtype::Custom => "function",
            };
            json!({
                "kind": "function",
                "name": func.name,
                "subtype": subtype,
                "input_sorts": func.input.iter().map(|sort| sort.name()).collect::<Vec<_>>(),
                "output_sort": func.output.name(),
            })
        }
        ResolvedCall::Primitive(primitive) => json!({
            "kind": "primitive",
            "name": primitive.name(),
            "input_sorts": primitive.input().iter().map(|sort| sort.name()).collect::<Vec<_>>(),
            "output_sort": primitive.output().name(),
        }),
    }
}

fn summarize_term(term: &GenericAtomTerm<ResolvedVar>) -> Value {
    match term {
        GenericAtomTerm::Var(_span, var) => json!({
            "kind": "var",
            "name": var.name,
            "sort": var.sort.name(),
        }),
        GenericAtomTerm::Literal(_span, literal) => json!({
            "kind": "literal",
            "value": literal.to_string(),
        }),
        GenericAtomTerm::Global(_span, var) => json!({
            "kind": "global",
            "name": var.name,
            "sort": var.sort.name(),
        }),
    }
}

fn term_supported(term: &GenericAtomTerm<ResolvedVar>) -> bool {
    !matches!(term, GenericAtomTerm::Global(..))
}

#[derive(Default)]
struct Counters {
    values: BTreeMap<&'static str, u64>,
}

impl Counters {
    fn inc(&mut self, key: &'static str) {
        *self.values.entry(key).or_default() += 1;
    }

    fn get(&self, key: &'static str) -> u64 {
        self.values.get(key).copied().unwrap_or_default()
    }

    fn merge(&mut self, other: &Counters) {
        for (key, value) in &other.values {
            *self.values.entry(key).or_default() += value;
        }
    }

    fn into_json(self) -> Value {
        json!(self.values)
    }
}
