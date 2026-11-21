use egglog::{ArcSort, EGraph, Error, Value};

fn eval(g: &mut EGraph, text: &str) -> (ArcSort, Value) {
    let expr = g.parser.get_expr_from_string(None, text).unwrap();
    g.eval_expr(&expr).unwrap()
}

#[test]
fn ranks_independent_occurrences_and_merges_constructor_products() {
    let mut g = EGraph::default();
    g.parse_and_run_program(
        None,
        "
        (datatype E (A :cost 1) (B :cost 3) (C :cost 0) (Pair E E :cost 1))
        (union (A) (B)) (union (C) (A)) (subsume (C))
        (datatype Root (Box E :cost 6) (Two E E :cost 1))
        (union (Box (A)) (Two (A) (A)))",
    )
    .unwrap();
    let root = eval(&mut g, "(Box (A))");
    let p = eval(&mut g, "42");
    let expected = [
        (3, "(Two (A) (A))"),
        (5, "(Two (A) (B))"),
        (5, "(Two (B) (A))"),
        (7, "(Box (A))"),
        (7, "(Two (B) (B))"),
        (9, "(Box (B))"),
    ];
    for k in 0..=expected.len() + 1 {
        let got = g
            .extract_ranked_variants(vec![root.clone(), p.clone(), root.clone()], k)
            .unwrap();
        for n in [0, 2] {
            let strings: Vec<_> = got.variants[n]
                .iter()
                .map(|v| (v.cost, got.termdag.to_string(v.term)))
                .collect();
            let prefix: Vec<_> = expected
                .iter()
                .take(k)
                .map(|(c, s)| (*c, s.to_string()))
                .collect();
            assert_eq!(strings, prefix);
        }
        assert_eq!(got.variants[1].len(), usize::from(k > 0));
        if k > 0 {
            assert_eq!(got.termdag.to_string(got.variants[1][0].term), "42");
        }
    }
}

#[test]
fn refuses_only_reachable_cycles_and_containers() {
    let mut g = EGraph::default();
    g.parse_and_run_program(
        None,
        "
        (sort V (Vec i64))
        (datatype E (A) (B) (S E) (Lit i64) (Hold V) (Hidden E :unextractable))
        (union (A) (S (A))) (union (B) (Hidden (A)))",
    )
    .unwrap();
    for s in ["(B)", "(Lit 42)"] {
        let root = eval(&mut g, s);
        let got = g.extract_ranked_variants(vec![root], 3).unwrap();
        assert_eq!(got.variants[0].len(), 1);
        assert_eq!(got.termdag.to_string(got.variants[0][0].term), s);
    }
    for (s, reason) in [("(A)", "cycle"), ("(Hold (vec-of 1))", "container")] {
        let root = eval(&mut g, s);
        assert!(
            matches!(g.extract_ranked_variants(vec![root],1),Err(Error::ExtractError(s)) if s.contains(reason))
        );
    }
}

#[test]
fn bounded_prefix_of_an_exponential_family_including_saturated_costs() {
    let mut g = EGraph::default();
    g.parse_and_run_program(
        None,
        "(datatype E (A) (B) (Pair E E)) (union (A) (B)) (let $x0 (A))",
    )
    .unwrap();
    for d in 1..=68 {
        let prev = d - 1;
        g.parse_and_run_program(None, &format!("(let $x{d} (Pair $x{prev} $x{prev}))"))
            .unwrap();
    }
    let root = eval(&mut g, "$x68");
    let got = g.extract_ranked_variants(vec![root], 4).unwrap();
    assert_eq!(got.variants[0].len(), 4);
    assert!(got.variants[0].iter().all(|v| v.cost == u64::MAX));
    assert!(got.termdag.size() <= 2 + 4 * 68);
}

// Port of the nested reductions in issue #300 to current constructor syntax.
#[cfg(feature = "bin")]
#[test]
fn cli_recovers_all_six_array_reductions() {
    use std::{
        collections::BTreeSet,
        io::Write,
        process::{Command, Stdio},
    };
    let program = r#"(datatype Axis (AxisIndex i64))
(datatype Array (Input String) (Sum Array Axis))
(rewrite (Sum (Sum x (AxisIndex j)) (AxisIndex i))
         (Sum (Sum x (AxisIndex i)) (AxisIndex (- j 1)))
         :when ((< i j)))
(rewrite (Sum (Sum x (AxisIndex j)) (AxisIndex i))
         (Sum (Sum x (AxisIndex (+ i 1))) (AxisIndex j))
         :when ((> i j)))
(let $result (Sum (Sum (Sum (Input "y") (AxisIndex 2)) (AxisIndex 1)) (AxisIndex 0)))
(run-schedule (saturate (run)))
(extract $result 6)
"#;
    let run = |flags: &[&str]| {
        let mut child = Command::new(env!("CARGO_BIN_EXE_egglog"))
            .args(flags)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(program.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success());
        String::from_utf8(output.stdout).unwrap()
    };
    let mut expected = BTreeSet::new();
    for i in 0..3 {
        for j in 0..2 {
            expected.insert(format!(
                r#"(Sum (Sum (Sum (Input "y") (AxisIndex {i})) (AxisIndex {j})) (AxisIndex 0))"#
            ));
        }
    }
    let ranked = run(&["--ranked-variants"]);
    let actual: Vec<_> = ranked
        .lines()
        .map(str::trim)
        .filter(|s| s.starts_with("(Sum"))
        .map(str::to_owned)
        .collect();
    assert_eq!(actual.len(), 6);
    assert_eq!(actual.into_iter().collect::<BTreeSet<_>>(), expected);
    assert_eq!(
        run(&[])
            .lines()
            .filter(|s| s.trim().starts_with("(Sum"))
            .count(),
        3
    );
}

#[test]
fn term_encoding_has_an_explicit_boundary() {
    let g = EGraph::new_with_term_encoding();
    let root = (
        g.get_arcsort_for_value_type::<i64>(),
        g.base_to_value(42i64),
    );
    let err = g.extract_ranked_variants(vec![root], 2).unwrap_err();
    assert!(matches!(err, Error::ExtractError(s) if s.contains("term/proof")));
}
