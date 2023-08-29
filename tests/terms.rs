use egglog::*;

// This file tests the public API to terms.

#[test]
fn test_termdag_public() {
    let mut td = TermDag::default();
    let x = td.var("x".into());
    let seven = td.lit(7.into());
    let f = td.app("f".into(), vec![x, seven]);
    assert_eq!(td.to_string(&f), "(f x 7)");
}

#[test]
#[should_panic]
fn test_termdag_malicious_client() {
    // here is an example of how TermIds can be misused by passing
    // them into the wrong DAG.

    let mut td = TermDag::default();
    let x = td.var("x".into());
    // at this point, td = [0 |-> x]
    // snapshot the current td
    let td2 = td.clone();
    let y = td.var("y".into());
    // now td = [0 |-> x, 1 |-> y]
    let f = td.app("f".into(), vec![x.clone(), y.clone()]);
    // f is Term::App("f", [0, 1])
    assert_eq!(td.to_string(&f), "(f x y)");
    // recall that td2 = [0 |-> x]
    // notice that f refers to index 1, so this crashes:
    td2.to_string(&f);
}
