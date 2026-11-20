//! Compare the archived Luminal projections using a single egg graph per layout.
use egg::{
    CostFunction, EGraph, Extractor, FlatTerm, Id, Language, Pattern, RecExpr, Rewrite, Symbol,
    define_language, rewrite,
};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::hint::black_box;
use std::io::{self, Write};
use std::time::{Duration, Instant};
use symbolic_expressions::{Sexp, parser::parse_str};

define_language! {
    enum Tensor {
        "Times" = Times([Id; 2]),
        "Sum" = Sum(Id),
        "Dot" = Dot([Id; 2]),
        Reg(u32),
    }
}

struct PlanCost;

impl CostFunction<Tensor> for PlanCost {
    type Cost = usize;

    fn cost<C: FnMut(Id) -> usize>(&mut self, node: &Tensor, mut child: C) -> usize {
        node.fold(
            if matches!(node, Tensor::Sum(_)) { 4 } else { 1 },
            |a, c| a + child(c),
        )
    }
}

#[derive(Default)]
struct Layout {
    products: Vec<(u32, u32, u32)>,
    roots: Vec<Root>,
}

struct Root {
    source: u32,
    expected: (u32, u32),
}

fn atom(x: &Sexp) -> &str {
    if let Sexp::String(s) = x {
        s
    } else {
        panic!("expected atom: {x}")
    }
}

fn app(x: &Sexp) -> (&str, &[Sexp]) {
    let Sexp::List(list) = x else {
        panic!("expected application: {x}")
    };
    let (head, rest) = list.split_first().expect("empty application");
    (atom(head), rest)
}

fn operator(s: &str, name: &str) -> usize {
    s.strip_prefix(name)
        .expect("unexpected operator")
        .parse()
        .expect("layout suffix")
}

fn reg(x: &Sexp, layout: usize) -> u32 {
    let (op, [arg]) = app(x) else {
        panic!("invalid SSA register: {x}")
    };
    assert_eq!(operator(op, "Reg"), layout);
    atom(arg).parse().expect("register number")
}

// This reader deliberately accepts only the exporter’s projection vocabulary.
// Require the same fusion and costs as the native comparator, and a unique
// equality check and extraction request for every root.
fn read_workload(src: &str) -> Vec<Layout> {
    let mut layouts = BTreeMap::<usize, Layout>::new();
    let mut roots = BTreeMap::new();
    let mut expected = BTreeMap::new();
    let mut extracts = BTreeSet::new();
    let mut declared = BTreeSet::new();
    let mut ran = false;
    for line in src.lines().filter(|l| !l.trim().is_empty()) {
        let expr = parse_str(line).expect("projection syntax");
        match app(&expr) {
            ("sort", [s]) if !ran => {
                assert!(
                    layouts
                        .insert(operator(atom(s), "Tensor"), Layout::default())
                        .is_none()
                );
            }
            ("constructor", [name, ..]) if !ran => {
                let op = atom(name);
                let layout = op
                    .trim_start_matches(char::is_alphabetic)
                    .parse::<usize>()
                    .unwrap();
                assert!(layouts.contains_key(&layout));
                let sort = format!("Tensor{layout}");
                let code = op.trim_end_matches(char::is_numeric);
                let signature = match code {
                    "Reg" => format!("(constructor {op} (i64) {sort} :cost 0)"),
                    "Sum" => format!("(constructor {op} ({sort}) {sort} :cost 4)"),
                    "Times" | "Dot" => format!("(constructor {op} ({sort} {sort}) {sort})"),
                    _ => panic!("unexpected constructor {op}"),
                };
                assert_eq!(line.trim(), signature);
                assert!(declared.insert((layout, code.to_owned())));
            }
            ("rewrite", [lhs, _]) if !ran => {
                let g = operator(app(lhs).0, "Sum");
                assert_eq!(
                    line.trim(),
                    format!("(rewrite (Sum{g} (Times{g} a b)) (Dot{g} a b))")
                );
                assert!(declared.insert((g, "rule".to_owned())));
            }
            ("union", [out, value]) if !ran => {
                let (op, [a, b]) = app(value) else {
                    panic!("bad product")
                };
                let g = operator(op, "Times");
                layouts
                    .get_mut(&g)
                    .unwrap()
                    .products
                    .push((reg(out, g), reg(a, g), reg(b, g)));
            }
            ("let", [name, sum]) if !ran => {
                let (op, [arg]) = app(sum) else {
                    panic!("bad reduction")
                };
                let g = operator(op, "Sum");
                assert!(
                    roots
                        .insert(atom(name).to_owned(), (g, reg(arg, g)))
                        .is_none()
                );
            }
            ("run", [n]) if !ran && atom(n) == "1" => ran = true,
            ("check", [eq]) if ran => {
                let ("=", [name, rhs]) = app(eq) else {
                    panic!("bad equality")
                };
                let (op, [a, b]) = app(rhs) else {
                    panic!("bad target")
                };
                let g = operator(op, "Dot");
                let (group, root) = roots[atom(name)];
                assert_eq!(g, group);
                assert!(expected.insert(atom(name).to_owned(), ()).is_none());
                layouts.get_mut(&g).unwrap().roots.push(Root {
                    source: root,
                    expected: (reg(a, g), reg(b, g)),
                });
            }
            ("extract", [name]) if ran => {
                assert!(roots.contains_key(atom(name)) && extracts.insert(atom(name).to_owned()));
            }
            _ => panic!("unsupported projection command: {expr}"),
        }
    }
    assert!(ran && !roots.is_empty() && !layouts.is_empty());
    assert_eq!(expected.len(), roots.len());
    assert_eq!(extracts.len(), roots.len());
    assert_eq!(declared.len(), 5 * layouts.len());
    assert!(declared.iter().all(|(g, _)| layouts.contains_key(g)));
    layouts.into_values().collect()
}

fn reg_expr(n: u32) -> RecExpr<Tensor> {
    let mut e = RecExpr::default();
    e.add(Tensor::Reg(n));
    e
}

fn product_expr(a: u32, b: u32) -> RecExpr<Tensor> {
    let mut e = RecExpr::default();
    let a = e.add(Tensor::Reg(a));
    let b = e.add(Tensor::Reg(b));
    e.add(Tensor::Times([a, b]));
    e
}

fn sum_expr(n: u32) -> RecExpr<Tensor> {
    let mut e = RecExpr::default();
    let id = e.add(Tensor::Reg(n));
    e.add(Tensor::Sum(id));
    e
}

fn dot_inputs(e: &RecExpr<Tensor>) -> (u32, u32) {
    let Tensor::Dot([a, b]) = e[e.root()] else {
        panic!("extracted a non-fused root {e}")
    };
    let (Tensor::Reg(x), Tensor::Reg(y)) = (&e[a], &e[b]) else {
        panic!("unexpected operands in {e}")
    };
    (*x, *y)
}

type Rule = Rewrite<Tensor, ()>;

// egg's checker allows unknown trusted reasons. Require every annotated step
// to be a declared input equation or the single checked fusion rule.
fn known_reasons(term: &FlatTerm<Tensor>, rules: &BTreeSet<Symbol>) -> bool {
    term.forward_rule
        .iter()
        .chain(&term.backward_rule)
        .all(|r| rules.contains(r))
        && term
            .children
            .iter()
            .all(|child| known_reasons(child, rules))
}

#[derive(Default)]
struct Metrics {
    seed: Duration,
    search: Duration,
    extract: Duration,
    verify: Duration,
    explain: Duration,
    proofs: usize,
    proof_steps: usize,
}

fn run_layout(layout: &Layout, mode: &str, fusion: &Rule, t: &mut Metrics) {
    let now = Instant::now();
    let mut graph = EGraph::<Tensor, ()>::default();
    if mode != "plain" {
        graph = graph.with_explanations_enabled();
    }
    for &(result, a, b) in &layout.products {
        let value = graph.add_uncanonical(Tensor::Reg(result));
        let a = graph.add_uncanonical(Tensor::Reg(a));
        let b = graph.add_uncanonical(Tensor::Reg(b));
        let prod = graph.add_uncanonical(Tensor::Times([a, b]));
        if mode == "plain" {
            graph.union(value, prod);
        } else {
            graph.union_trusted(value, prod, format!("ssa-{result}"));
        }
    }
    let ids: Vec<_> = layout
        .roots
        .iter()
        .map(|r| {
            let reg = graph.add(Tensor::Reg(r.source));
            graph.add(Tensor::Sum(reg))
        })
        .collect();
    graph.rebuild();
    t.seed += now.elapsed();

    let now = Instant::now();
    let hits = fusion.search(&graph);
    fusion.apply(&mut graph, &hits);
    graph.rebuild();
    t.search += now.elapsed();

    let now = Instant::now();
    let extractor = Extractor::new(&graph, PlanCost);
    let plans: Vec<_> = ids.iter().map(|id| extractor.find_best(*id)).collect();
    black_box(&plans);
    drop(extractor);
    t.extract += now.elapsed();

    let now = Instant::now();
    for (r, (cost, plan)) in layout.roots.iter().zip(&plans) {
        assert_eq!(*cost, 3);
        assert_eq!(dot_inputs(plan), r.expected);
    }
    t.verify += now.elapsed();

    if mode == "explain" {
        let now = Instant::now();
        let mut rules = vec![fusion.clone()];
        for &(result, a, b) in &layout.products {
            rules.push(
                Rewrite::new(
                    format!("ssa-{result}"),
                    Pattern::from(reg_expr(result)),
                    Pattern::from(product_expr(a, b)),
                )
                .unwrap(),
            );
        }
        let names = rules.iter().map(|r| r.name).collect::<BTreeSet<_>>();
        assert_eq!(rules.len(), names.len());
        for (root, (_, best)) in layout.roots.iter().zip(&plans) {
            let mut proof = graph.explain_equivalence(&sum_expr(root.source), best);
            let steps = proof.make_flat_explanation();
            assert!(steps.len() >= 2 && steps.iter().all(|term| known_reasons(term, &names)));
            t.proof_steps += steps.len() - 1;
            proof.check_proof(&rules);
            black_box(&proof);
            t.proofs += 1;
        }
        t.explain += now.elapsed();
    }
}

fn main() {
    let args: Vec<_> = env::args().collect();
    let [_, mode, file] = args.as_slice() else {
        panic!("expected plain|track|explain PROJECTIONS.egg")
    };
    assert!(matches!(mode.as_str(), "plain" | "track" | "explain"));
    let input = read_workload(&fs::read_to_string(file).unwrap());
    let roots: usize = input.iter().map(|l| l.roots.len()).sum();
    let products: usize = input.iter().map(|l| l.products.len()).sum();
    // egg's first rebuild calibrates the process-wide quanta clock. It is
    // included in whole-process latency and excluded from the optimizer timer.
    let mut calibration = EGraph::<Tensor, ()>::default();
    calibration.rebuild();
    drop(black_box(calibration));
    let start = Instant::now();
    let fusion: Rule = rewrite!("ordered-fusion"; "(Sum (Times ?a ?b))" => "(Dot ?a ?b)");
    let mut t = Metrics::default();
    for layout in &input {
        run_layout(layout, mode, &fusion, &mut t);
    }
    let elapsed = start.elapsed();
    writeln!(io::stdout().lock(),
        "layouts={} roots={roots} products={products} proofs={} proof_steps={} seed_ns={} search_ns={} extract_ns={} check_ns={} explain_ns={} total_ns={}",
        input.len(),
        t.proofs,
        t.proof_steps,
        t.seed.as_nanos(),
        t.search.as_nanos(),
        t.extract.as_nanos(),
        t.verify.as_nanos(),
        t.explain.as_nanos(),
        elapsed.as_nanos()
    ).expect("writing benchmark metrics");
}
