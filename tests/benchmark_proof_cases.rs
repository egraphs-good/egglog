// The benchmark module also contains execution helpers that this enumeration
// regression test intentionally does not call.
#[allow(dead_code)]
#[path = "../benches/common.rs"]
mod common;

#[test]
fn eggcc_2mm_has_only_the_normal_codspeed_case() {
    let names = common::bench_cases("tests/eggcc-2mm.egg")
        .into_iter()
        .map(|case| case.name)
        .collect::<Vec<_>>();

    // CodSpeed applies the matrix value as a substring filter. Keeping exactly
    // the normal case prevents the `eggcc-2mm` shard from also selecting the
    // proof-testing case, which is not tractable under its instrumentation.
    assert_eq!(names, ["eggcc-2mm"]);
}
