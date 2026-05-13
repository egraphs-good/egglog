//! Per-iter parity probe: compare souffle and default-egglog tuple counts
//! after each `(run N)` of math-microbench, narrowing the iter where
//! divergence first appears.

use egglog::{souffle_translator, EGraph};
use egglog_souffle_backend::{emit::emit, runner};

const SOURCE_HEADER: &str = r#"
(datatype Math
    (Diff Math Math)
    (Integral Math Math)
    (Add Math Math)
    (Sub Math Math)
    (Mul Math Math)
    (Div Math Math)
    (Pow Math Math)
    (Ln Math)
    (Sqrt Math)
    (Sin Math)
    (Cos Math)
    (Const i64)
    (Var String))

(rewrite (Add a b) (Add b a))
(rewrite (Mul a b) (Mul b a))
(rewrite (Add a (Add b c)) (Add (Add a b) c))
(rewrite (Mul a (Mul b c)) (Mul (Mul a b) c))
(rewrite (Sub a b) (Add a (Mul (Const -1) b)))
(rewrite (Add a (Const 0)) a)
(rewrite (Mul a (Const 0)) (Const 0))
(rewrite (Mul a (Const 1)) a)
(rewrite (Sub a a) (Const 0))
(rewrite (Mul a (Add b c)) (Add (Mul a b) (Mul a c)))
(rewrite (Add (Mul a b) (Mul a c)) (Mul a (Add b c)))
(rewrite (Mul (Pow a b) (Pow a c)) (Pow a (Add b c)))
(rewrite (Pow x (Const 1)) x)
(rewrite (Pow x (Const 2)) (Mul x x))
(rewrite (Diff x (Add a b)) (Add (Diff x a) (Diff x b)))
(rewrite (Diff x (Mul a b)) (Add (Mul a (Diff x b)) (Mul b (Diff x a))))
(rewrite (Diff x (Sin x)) (Cos x))
(rewrite (Diff x (Cos x)) (Mul (Const -1) (Sin x)))
(rewrite (Integral (Const 1) x) x)
(rewrite (Integral (Cos x) x) (Sin x))
(rewrite (Integral (Sin x) x) (Mul (Const -1) (Cos x)))
(rewrite (Integral (Add f g) x) (Add (Integral f x) (Integral g x)))
(rewrite (Integral (Sub f g) x) (Sub (Integral f x) (Integral g x)))
(rewrite (Integral (Mul a b) x)
    (Sub (Mul a (Integral b x))
         (Integral (Mul (Diff x a) (Integral b x)) x)))

(Integral (Ln (Var "x")) (Var "x"))
(Integral (Add (Var "x") (Cos (Var "x"))) (Var "x"))
(Integral (Mul (Cos (Var "x")) (Var "x")) (Var "x"))
(Diff (Var "x") (Add (Const 1) (Mul (Const 2) (Var "x"))))
(Diff (Var "x") (Sub (Pow (Var "x") (Const 3)) (Mul (Const 7) (Pow (Var "x") (Const 2)))))
(Add (Mul (Var "y") (Add (Var "x") (Var "y"))) (Sub (Add (Var "x") (Const 2)) (Add (Var "x") (Var "x"))))
(Div (Const 1)
    (Sub (Div (Add (Const 1) (Sqrt (Var "five"))) (Const 2))
         (Div (Sub (Const 1) (Sqrt (Var "five"))) (Const 2))))
"#;

fn default_sizes_after_run(n: usize) -> (usize, usize) {
    let mut eg = EGraph::default();
    eg.ensure_no_reserved_symbols(false);
    let prog = format!("{SOURCE_HEADER}\n(run {n})\n(print-size Add)\n(print-size Mul)\n");
    let results = eg.parse_and_run_program(None, &prog).expect("default run");
    let mut add = 0usize;
    let mut mul = 0usize;
    let mut idx = 0;
    for r in results {
        if let egglog::CommandOutput::PrintFunctionSize(k) = r {
            if idx == 0 {
                add = k;
            } else if idx == 1 {
                mul = k;
            }
            idx += 1;
        }
    }
    (add, mul)
}

fn souffle_sizes_after_run(n: usize) -> (usize, usize, String) {
    let mut eg = EGraph::new_with_term_encoding().with_souffle_compat_strata();
    let prog = format!("{SOURCE_HEADER}\n(run {n})\n(print-size Add)\n(print-size Mul)\n");
    let commands = eg.resolve_program(None, &prog).expect("resolve");
    let out = souffle_translator::translate_with_manifest(&commands).expect("translate");
    let result = runner::run(&out.program, &out.manifest).expect("souffle run");
    let add = result
        .view_sizes
        .iter()
        .find(|(n, _)| n == "Add")
        .map(|(_, s)| *s)
        .unwrap_or(0);
    let mul = result
        .view_sizes
        .iter()
        .find(|(n, _)| n == "Mul")
        .map(|(_, s)| *s)
        .unwrap_or(0);
    (add, mul, result.raw_stdout)
}

/// Print IR relation names to confirm whether snap relations have @ prefix.
#[test]
fn print_ir_names() {
    let source = r#"
(datatype Math (Add Math Math) (Const i64))
(Add (Const 1) (Const 2))
"#;
    let mut eg = EGraph::new_with_term_encoding().with_souffle_compat_strata();
    let commands = eg.resolve_program(None, source).unwrap();
    let out = souffle_translator::translate_with_manifest(&commands).unwrap();
    for r in &out.program.relations {
        eprintln!("{}", r.name);
    }
}

/// Dump Sub_snap + Sub_canonical at each outer iter for minimal_intmul_gap.
/// Goal: confirm Sub_snap has the new iter-2 Sub at iter 3 start.
#[test]
fn dump_sub_snap_per_iter() {
    if runner::find_souffle_binary().is_none() {
        eprintln!("skipping: souffle binary not found");
        return;
    }
    let header = r#"
(datatype Math
    (Diff Math Math)
    (Integral Math Math)
    (Add Math Math)
    (Sub Math Math)
    (Mul Math Math)
    (Const i64)
    (Var String))
(rewrite (Sub a b) (Add a (Mul (Const -1) b)))
(rewrite (Integral (Mul a b) x)
    (Sub (Mul a (Integral b x))
         (Integral (Mul (Diff x a) (Integral b x)) x)))
(Integral (Mul (Var "y") (Var "x")) (Var "x"))
"#;
    let mut sf_eg = EGraph::new_with_term_encoding().with_souffle_compat_strata();
    let prog = format!("{header}\n(run 4)\n");
    let commands = sf_eg.resolve_program(None, &prog).expect("resolve");
    let out = souffle_translator::translate_with_manifest(&commands).expect("translate");
    let mut dl = emit(&out.program);
    // Print SubView_snap and SubView_canonical inside the loop body
    // (printsize is emitted at end of each iter). Also AddView.
    dl.push_str("\n.printsize Eg_SubView_snap\n");
    dl.push_str("\n.printsize Eg_SubView\n");
    dl.push_str("\n.printsize Eg_AddView\n");
    dl.push_str("\n.printsize Eg_AddView_buffer\n");
    dl.push_str("\n.printsize Eg_MulView_buffer\n");
    dl.push_str("\n.printsize IterCounter\n");
    dl.push_str("\n.output Eg_AddView(IO=stdout)\n");
    dl.push_str("\n.output Eg_AddView_canonical(IO=stdout)\n");
    dl.push_str("\n.output Eg_UF_Math(IO=stdout)\n");
    let path = format!("/tmp/souffle-snapdump-{}.dl", std::process::id());
    std::fs::write(&path, &dl).expect("write");
    let bin = runner::find_souffle_binary().unwrap();
    let result = std::process::Command::new("timeout")
        .arg("30")
        .arg(&bin)
        .arg(&path)
        .output()
        .expect("spawn");
    let stdout = String::from_utf8_lossy(&result.stdout);
    std::fs::write("/tmp/souffle-snapdump.txt", stdout.as_bytes()).expect("write");
    eprintln!("wrote /tmp/souffle-snapdump.txt ({} bytes)", stdout.len());
}

/// Try math-microbench with a small set of rules to reproduce
/// the +1 Mul gap.
#[test]
fn minimal_mul_gap_repro() {
    if runner::find_souffle_binary().is_none() {
        return;
    }
    let datatype = r#"(datatype Math
    (Diff Math Math)
    (Integral Math Math)
    (Add Math Math)
    (Sub Math Math)
    (Mul Math Math)
    (Cos Math)
    (Const i64)
    (Var String))"#;
    // Try various reduced rule sets
    let rule_sets: Vec<(&str, &str)> = vec![
        ("MulComm+IntMul", r#"
(rewrite (Mul a b) (Mul b a))
(rewrite (Integral (Mul a b) x)
    (Sub (Mul a (Integral b x))
         (Integral (Mul (Diff x a) (Integral b x)) x)))
"#),
        ("MulComm+SubToAdd+IntMul", r#"
(rewrite (Mul a b) (Mul b a))
(rewrite (Sub a b) (Add a (Mul (Const -1) b)))
(rewrite (Integral (Mul a b) x)
    (Sub (Mul a (Integral b x))
         (Integral (Mul (Diff x a) (Integral b x)) x)))
"#),
        ("All-IntAdd-IntCos-IntSin", r#"
(rewrite (Mul a b) (Mul b a))
(rewrite (Sub a b) (Add a (Mul (Const -1) b)))
(rewrite (Diff x (Mul a b)) (Add (Mul a (Diff x b)) (Mul b (Diff x a))))
(rewrite (Integral (Mul a b) x)
    (Sub (Mul a (Integral b x))
         (Integral (Mul (Diff x a) (Integral b x)) x)))
"#),
    ];
    let real_inits = r#"
(Integral (Mul (Cos (Var "x")) (Var "x")) (Var "x"))
"#;
    for (name, rules) in &rule_sets {
        for n in 0..=4 {
            let source = format!("{datatype}\n{rules}\n{real_inits}\n(run {n})\n(print-size Mul)\n");
            let mut def = EGraph::default();
            def.ensure_no_reserved_symbols(false);
            let dres = def.parse_and_run_program(None, &source).expect("default");
            let def_mul = dres
                .iter()
                .find_map(|r| match r {
                    egglog::CommandOutput::PrintFunctionSize(n) => Some(*n),
                    _ => None,
                })
                .unwrap_or(0);
            let mut sf = EGraph::new_with_term_encoding().with_souffle_compat_strata();
            let commands = sf.resolve_program(None, &source).expect("resolve");
            let out = souffle_translator::translate_with_manifest(&commands).expect("translate");
            let sf_result = runner::run(&out.program, &out.manifest).expect("souffle");
            let sf_mul = sf_result.view_sizes.iter().find(|(n, _)| n == "Mul").map(|(_, s)| *s).unwrap_or(0);
            let diff = (sf_mul as isize) - (def_mul as isize);
            eprintln!("[{name}] (run {n}): default={def_mul} souffle={sf_mul} diff={diff}");
        }
    }
}

/// Dump souffle's Mul_canonical table at (run 4) vs default's, find
/// the specific row that differs. Uses the MINIMAL repro from
/// bisect_two_rules: 7 essential rules + 1 init expression.
#[test]
fn diff_mul_at_run4_minimal() {
    if runner::find_souffle_binary().is_none() {
        return;
    }
    let minimal = r#"
(datatype Math
    (Diff Math Math)
    (Integral Math Math)
    (Add Math Math)
    (Sub Math Math)
    (Mul Math Math)
    (Cos Math)
    (Sin Math)
    (Const i64)
    (Var String))
(rewrite (Mul a b) (Mul b a))
(rewrite (Diff x (Mul a b)) (Add (Mul a (Diff x b)) (Mul b (Diff x a))))
(rewrite (Diff x (Sin x)) (Cos x))
(rewrite (Diff x (Cos x)) (Mul (Const -1) (Sin x)))
(rewrite (Integral (Cos x) x) (Sin x))
(rewrite (Integral (Sin x) x) (Mul (Const -1) (Cos x)))
(rewrite (Integral (Mul a b) x) (Sub (Mul a (Integral b x)) (Integral (Mul (Diff x a) (Integral b x)) x)))
(Integral (Mul (Cos (Var "x")) (Var "x")) (Var "x"))
"#;
    let mut eg = EGraph::new_with_term_encoding().with_souffle_compat_strata();
    let prog = format!("{minimal}\n(run 4)\n");
    let commands = eg.resolve_program(None, &prog).expect("resolve");
    let out = souffle_translator::translate_with_manifest(&commands).expect("translate");
    let mut dl = emit(&out.program);
    dl.push_str("\n.output Eg_MulView_canonical(IO=stdout)\n");
    dl.push_str("\n.output Eg_DiffView_canonical(IO=stdout)\n");
    dl.push_str("\n.output Eg_UF_Math(IO=stdout)\n");
    dl.push_str("\n.output Eg_SinView_canonical(IO=stdout)\n");
    dl.push_str("\n.output Eg_CosView_canonical(IO=stdout)\n");
    dl.push_str("\n.output Eg_IntegralView_canonical(IO=stdout)\n");
    dl.push_str("\n.output Eg_ConstView_canonical(IO=stdout)\n");
    dl.push_str("\n.output Eg_VarView_canonical(IO=stdout)\n");
    dl.push_str("\n.output Eg_NotLeader_Math(IO=stdout)\n");
    let path = format!("/tmp/souffle-mul-r4-{}.dl", std::process::id());
    std::fs::write(&path, &dl).expect("write");
    let bin = runner::find_souffle_binary().unwrap();
    let result = std::process::Command::new("timeout")
        .arg("60")
        .arg(&bin)
        .arg(&path)
        .output()
        .expect("spawn");
    let stdout = String::from_utf8_lossy(&result.stdout).into_owned();
    std::fs::write("/tmp/souffle-mul-r4.txt", &stdout).expect("write");
    eprintln!("wrote /tmp/souffle-mul-r4.txt ({} bytes)", stdout.len());

    // Also dump default's Mul table via (print-function Mul N).
    for n in &[3, 4] {
        let mut def_eg = EGraph::default();
        def_eg.ensure_no_reserved_symbols(false);
        let prog = format!("{minimal}\n(run {n})\n(print-function Mul 500)\n");
        let results = def_eg.parse_and_run_program(None, &prog).expect("default");
        let mut buf = String::new();
        for r in &results {
            buf.push_str(&format!("{r}\n"));
        }
        let path = format!("/tmp/default-mul-r{n}.txt");
        std::fs::write(&path, &buf).expect("write");
        eprintln!("wrote {} ({} bytes)", path, buf.len());
    }
}

/// Remove TWO rules at once from math-microbench to find a minimal
/// pair whose presence is needed to reproduce the +1 Mul gap.
#[test]
fn bisect_two_rules() {
    if runner::find_souffle_binary().is_none() {
        return;
    }
    // Rules that BY THEMSELVES (when removed) eliminate the gap:
    // 23=IntMul, 20=IntSin, 19=IntCos, 17=DiffCos, 16=DiffSin, 15=DiffMul, 1=MulComm.
    // So any of these being absent prevents the gap. They might all
    // CONTRIBUTE to the chain. The gap is interaction of all + others.
    //
    // Try: WITH everything, remove ONLY rules that didn't change the
    // gap (and see if removing 2 such rules also keeps gap or eliminates).
    //
    // First test: remove "innocent" rules to find the min set that still
    // reproduces the gap.
    let datatype = r#"(datatype Math
    (Diff Math Math)
    (Integral Math Math)
    (Add Math Math)
    (Sub Math Math)
    (Mul Math Math)
    (Div Math Math)
    (Pow Math Math)
    (Ln Math)
    (Sqrt Math)
    (Sin Math)
    (Cos Math)
    (Const i64)
    (Var String))"#;
    let all_inits = vec![
        r#"(Integral (Ln (Var "x")) (Var "x"))"#,
        r#"(Integral (Add (Var "x") (Cos (Var "x"))) (Var "x"))"#,
        r#"(Integral (Mul (Cos (Var "x")) (Var "x")) (Var "x"))"#,
        r#"(Diff (Var "x") (Add (Const 1) (Mul (Const 2) (Var "x"))))"#,
        r#"(Diff (Var "x") (Sub (Pow (Var "x") (Const 3)) (Mul (Const 7) (Pow (Var "x") (Const 2)))))"#,
        r#"(Add (Mul (Var "y") (Add (Var "x") (Var "y"))) (Sub (Add (Var "x") (Const 2)) (Add (Var "x") (Var "x"))))"#,
        r#"(Div (Const 1) (Sub (Div (Add (Const 1) (Sqrt (Var "five"))) (Const 2)) (Div (Sub (Const 1) (Sqrt (Var "five"))) (Const 2))))"#,
    ];
    // ESSENTIAL rules.
    let essential = vec![
        "(rewrite (Mul a b) (Mul b a))",
        "(rewrite (Diff x (Mul a b)) (Add (Mul a (Diff x b)) (Mul b (Diff x a))))",
        "(rewrite (Diff x (Sin x)) (Cos x))",
        "(rewrite (Diff x (Cos x)) (Mul (Const -1) (Sin x)))",
        "(rewrite (Integral (Cos x) x) (Sin x))",
        "(rewrite (Integral (Sin x) x) (Mul (Const -1) (Cos x)))",
        "(rewrite (Integral (Mul a b) x) (Sub (Mul a (Integral b x)) (Integral (Mul (Diff x a) (Integral b x)) x)))",
    ];
    let _all_rules_unused: String = essential.join("\n");
    let init2 = r#"(Integral (Mul (Cos (Var "x")) (Var "x")) (Var "x"))"#;
    // Minimal repro: 7 essential rules + 1 init at run 4 produces +1
    // Mul gap (default=46, souffle=47). Loop over runs to confirm
    // the gap appears at run 4 and persists at run 5.
    let rules_str: String = essential.join("\n");
    for n in 0..=5 {
        let inits_str = init2.to_string();
        let source = format!("{datatype}\n{rules_str}\n{inits_str}\n(run {n})\n(print-size Mul)\n");
        let mut def = EGraph::default();
        def.ensure_no_reserved_symbols(false);
        let dres = def.parse_and_run_program(None, &source).expect("default");
        let def_mul = dres.iter().find_map(|r| match r {
            egglog::CommandOutput::PrintFunctionSize(n) => Some(*n),
            _ => None,
        }).unwrap_or(0);
        let mut sf = EGraph::new_with_term_encoding().with_souffle_compat_strata();
        let commands = sf.resolve_program(None, &source).expect("resolve");
        let out = souffle_translator::translate_with_manifest(&commands).expect("translate");
        let sf_result = runner::run(&out.program, &out.manifest).expect("souffle");
        let sf_mul = sf_result.view_sizes.iter().find(|(n, _)| n == "Mul").map(|(_, s)| *s).unwrap_or(0);
        eprintln!("[run {}]: default={def_mul} souffle={sf_mul} diff={}",
            n,
            sf_mul as isize - def_mul as isize);
    }
}

/// Try math-microbench with just one extra rule at a time to find
/// which rule introduces the +1 Mul gap at (run 4).
#[test]
fn bisect_extra_mul_gap() {
    if runner::find_souffle_binary().is_none() {
        return;
    }
    // Start from the FULL math-microbench rule set and progressively
    // REMOVE rules to find the minimum set that reproduces +1 Mul.
    let datatype = r#"(datatype Math
    (Diff Math Math)
    (Integral Math Math)
    (Add Math Math)
    (Sub Math Math)
    (Mul Math Math)
    (Div Math Math)
    (Pow Math Math)
    (Ln Math)
    (Sqrt Math)
    (Sin Math)
    (Cos Math)
    (Const i64)
    (Var String))"#;
    let inits = r#"
(Integral (Ln (Var "x")) (Var "x"))
(Integral (Add (Var "x") (Cos (Var "x"))) (Var "x"))
(Integral (Mul (Cos (Var "x")) (Var "x")) (Var "x"))
(Diff (Var "x") (Add (Const 1) (Mul (Const 2) (Var "x"))))
(Diff (Var "x") (Sub (Pow (Var "x") (Const 3)) (Mul (Const 7) (Pow (Var "x") (Const 2)))))
(Add (Mul (Var "y") (Add (Var "x") (Var "y"))) (Sub (Add (Var "x") (Const 2)) (Add (Var "x") (Var "x"))))
(Div (Const 1)
    (Sub (Div (Add (Const 1) (Sqrt (Var "five"))) (Const 2))
         (Div (Sub (Const 1) (Sqrt (Var "five"))) (Const 2))))"#;
    let all_rules = vec![
        ("AddComm", "(rewrite (Add a b) (Add b a))"),
        ("MulComm", "(rewrite (Mul a b) (Mul b a))"),
        ("AddAssoc", "(rewrite (Add a (Add b c)) (Add (Add a b) c))"),
        ("MulAssoc", "(rewrite (Mul a (Mul b c)) (Mul (Mul a b) c))"),
        ("SubToAdd", "(rewrite (Sub a b) (Add a (Mul (Const -1) b)))"),
        ("AddZero", "(rewrite (Add a (Const 0)) a)"),
        ("MulZero", "(rewrite (Mul a (Const 0)) (Const 0))"),
        ("MulOne", "(rewrite (Mul a (Const 1)) a)"),
        ("SubSelf", "(rewrite (Sub a a) (Const 0))"),
        ("MulDistL", "(rewrite (Mul a (Add b c)) (Add (Mul a b) (Mul a c)))"),
        ("AddFactor", "(rewrite (Add (Mul a b) (Mul a c)) (Mul a (Add b c)))"),
        ("PowMul", "(rewrite (Mul (Pow a b) (Pow a c)) (Pow a (Add b c)))"),
        ("PowOne", "(rewrite (Pow x (Const 1)) x)"),
        ("PowTwo", "(rewrite (Pow x (Const 2)) (Mul x x))"),
        ("DiffAdd", "(rewrite (Diff x (Add a b)) (Add (Diff x a) (Diff x b)))"),
        ("DiffMul", "(rewrite (Diff x (Mul a b)) (Add (Mul a (Diff x b)) (Mul b (Diff x a))))"),
        ("DiffSin", "(rewrite (Diff x (Sin x)) (Cos x))"),
        ("DiffCos", "(rewrite (Diff x (Cos x)) (Mul (Const -1) (Sin x)))"),
        ("IntConst", "(rewrite (Integral (Const 1) x) x)"),
        ("IntCos", "(rewrite (Integral (Cos x) x) (Sin x))"),
        ("IntSin", "(rewrite (Integral (Sin x) x) (Mul (Const -1) (Cos x)))"),
        ("IntAdd", "(rewrite (Integral (Add f g) x) (Add (Integral f x) (Integral g x)))"),
        ("IntSub", "(rewrite (Integral (Sub f g) x) (Sub (Integral f x) (Integral g x)))"),
        ("IntMul", "(rewrite (Integral (Mul a b) x) (Sub (Mul a (Integral b x)) (Integral (Mul (Diff x a) (Integral b x)) x)))"),
    ];
    // Run with all rules first
    for skip_idx in (0..all_rules.len()).rev() {
        let mut rules_str = String::new();
        for (i, (_name, rule)) in all_rules.iter().enumerate() {
            if i == skip_idx {
                continue;
            }
            rules_str.push_str(rule);
            rules_str.push('\n');
        }
        let source = format!("{datatype}\n{rules_str}\n{inits}\n(run 4)\n(print-size Mul)\n");
        let mut def = EGraph::default();
        def.ensure_no_reserved_symbols(false);
        let dres = def.parse_and_run_program(None, &source).expect("default");
        let def_mul = dres
            .iter()
            .find_map(|r| match r {
                egglog::CommandOutput::PrintFunctionSize(n) => Some(*n),
                _ => None,
            })
            .unwrap_or(0);
        let mut sf = EGraph::new_with_term_encoding().with_souffle_compat_strata();
        let commands = sf.resolve_program(None, &source).expect("resolve");
        let out = souffle_translator::translate_with_manifest(&commands).expect("translate");
        let sf_result = runner::run(&out.program, &out.manifest).expect("souffle");
        let sf_mul = sf_result.view_sizes.iter().find(|(n, _)| n == "Mul").map(|(_, s)| *s).unwrap_or(0);
        let diff = (sf_mul as isize) - (def_mul as isize);
        eprintln!(
            "skip={} ({}): default Mul={} souffle Mul={} diff={}",
            skip_idx, all_rules[skip_idx].0, def_mul, sf_mul, diff
        );
    }
}

/// Even smaller: only Sub→Add + a Sub-creating rule. The Sub-creating
/// rule writes a NEW Sub at each iter using fresh values, so Sub→Add
/// should fire on the new Sub at the NEXT iter.
#[test]
fn minimal_sub_chain() {
    if runner::find_souffle_binary().is_none() {
        eprintln!("skipping: souffle binary not found");
        return;
    }
    let header = r#"
(datatype Math
    (Add Math Math)
    (Sub Math Math)
    (Mul Math Math)
    (Step Math)
    (Const i64)
    (Var String))

(rewrite (Sub a b) (Add a (Mul (Const -1) b)))
(rewrite (Step a) (Sub a a))

(Step (Var "x"))
"#;
    for n in 0..=4 {
        let mut def_eg = EGraph::default();
        def_eg.ensure_no_reserved_symbols(false);
        let prog = format!("{header}\n(run {n})\n(print-size Add)\n(print-size Sub)\n(print-size Step)\n(print-size Mul)\n");
        let dres = def_eg.parse_and_run_program(None, &prog).expect("default run");
        let mut sizes = vec![];
        for r in dres {
            if let egglog::CommandOutput::PrintFunctionSize(k) = r {
                sizes.push(k);
            }
        }
        let mut sf_eg = EGraph::new_with_term_encoding().with_souffle_compat_strata();
        let commands = sf_eg.resolve_program(None, &prog).expect("resolve");
        let out = souffle_translator::translate_with_manifest(&commands).expect("translate");
        let result = runner::run(&out.program, &out.manifest).expect("souffle run");
        let find = |name: &str| {
            result
                .view_sizes
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, s)| *s)
                .unwrap_or(0)
        };
        eprintln!(
            "(run {n}): default Add={} Sub={} Step={} Mul={} | souffle Add={} Sub={} Step={} Mul={}",
            sizes[0], sizes[1], sizes[2], sizes[3],
            find("Add"), find("Sub"), find("Step"), find("Mul"),
        );
    }
}

/// Minimal reproducer of the (run 2+) gap. Only IntMul + Sub desugar
/// + a single initial Integral(Mul ...) term. Goal: smallest possible
/// case that exhibits the off-by-one at run 2.
#[test]
fn minimal_intmul_gap() {
    if runner::find_souffle_binary().is_none() {
        eprintln!("skipping: souffle binary not found");
        return;
    }
    let header = r#"
(datatype Math
    (Diff Math Math)
    (Integral Math Math)
    (Add Math Math)
    (Sub Math Math)
    (Mul Math Math)
    (Const i64)
    (Var String))

(rewrite (Sub a b) (Add a (Mul (Const -1) b)))
(rewrite (Integral (Mul a b) x)
    (Sub (Mul a (Integral b x))
         (Integral (Mul (Diff x a) (Integral b x)) x)))

(Integral (Mul (Var "y") (Var "x")) (Var "x"))
"#;
    for n in 0..=4 {
        let mut def_eg = EGraph::default();
        def_eg.ensure_no_reserved_symbols(false);
        let prog = format!("{header}\n(run {n})\n(print-size Add)\n(print-size Mul)\n(print-size Integral)\n(print-size Sub)\n");
        let dres = def_eg
            .parse_and_run_program(None, &prog)
            .expect("default run");
        let mut sizes = vec![];
        for r in dres {
            if let egglog::CommandOutput::PrintFunctionSize(k) = r {
                sizes.push(k);
            }
        }
        let mut sf_eg = EGraph::new_with_term_encoding().with_souffle_compat_strata();
        let commands = sf_eg.resolve_program(None, &prog).expect("resolve");
        let out = souffle_translator::translate_with_manifest(&commands).expect("translate");
        let result = runner::run(&out.program, &out.manifest).expect("souffle run");
        let find = |name: &str| {
            result
                .view_sizes
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, s)| *s)
                .unwrap_or(0)
        };
        eprintln!(
            "(run {n}): default Add={} Mul={} Integral={} Sub={} | souffle Add={} Mul={} Integral={} Sub={}",
            sizes[0], sizes[1], sizes[2], sizes[3],
            find("Add"), find("Mul"), find("Integral"), find("Sub"),
        );
    }
}

#[test]
fn per_iter_parity_low_runs() {
    if runner::find_souffle_binary().is_none() {
        eprintln!("skipping: souffle binary not found");
        return;
    }
    for n in 0..=7 {
        let (def_add, def_mul) = default_sizes_after_run(n);
        let (sf_add, sf_mul, _stdout) = souffle_sizes_after_run(n);
        eprintln!(
            "(run {n}): default Add={def_add} Mul={def_mul} | souffle Add={sf_add} Mul={sf_mul} | diff Add={} Mul={}",
            (sf_add as isize) - (def_add as isize),
            (sf_mul as isize) - (def_mul as isize),
        );
    }
}

/// At (run 2), dump souffle's full Add and Mul tables, and have default
/// emit its Add/Mul tables, so we can find the SPECIFIC missing tuple.
#[test]
fn dump_run2_tables_for_diff() {
    if runner::find_souffle_binary().is_none() {
        eprintln!("skipping: souffle binary not found");
        return;
    }
    let mut eg = EGraph::new_with_term_encoding().with_souffle_compat_strata();
    let prog = format!("{SOURCE_HEADER}\n(run 2)\n");
    let commands = eg.resolve_program(None, &prog).expect("resolve");
    let out = souffle_translator::translate_with_manifest(&commands).expect("translate");
    let mut dl = emit(&out.program);
    // Add `.output` for every canonical relation so we can diff.
    let canonical_names: Vec<String> = out
        .program
        .relations
        .iter()
        .filter(|r| r.name.ends_with("_canonical"))
        .map(|r| {
            if let Some(rest) = r.name.strip_prefix('@') {
                format!("Eg_{rest}")
            } else {
                r.name.clone()
            }
        })
        .collect();
    for c in &canonical_names {
        dl.push_str(&format!("\n.output {c}(IO=stdout)\n"));
    }
    let path = format!("/tmp/souffle-dump-{}.dl", std::process::id());
    std::fs::write(&path, &dl).expect("write");
    let bin = runner::find_souffle_binary().unwrap();
    let result = std::process::Command::new("timeout")
        .arg("30")
        .arg(&bin)
        .arg(&path)
        .output()
        .expect("spawn");
    let stdout = String::from_utf8_lossy(&result.stdout).into_owned();
    std::fs::write("/tmp/souffle-run2-dump.txt", &stdout).expect("write");
    eprintln!("wrote /tmp/souffle-run2-dump.txt ({} bytes)", stdout.len());
}
