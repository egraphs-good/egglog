#[path = "../benches/math_microbenchmark_support.rs"]
mod math_microbenchmark_support;

#[test]
fn math_microbenchmark_smoke() {
    let mut input = math_microbenchmark_support::math_microbenchmark_setup();
    math_microbenchmark_support::run_math_microbenchmark_iters(&mut input, 1);
    assert!(input.egraph.get_size("MIntegral") > 0);
    assert!(input.egraph.get_size("MAdd") > 0);
    assert!(input.egraph.get_size("MMul") > 0);
}

#[test]
fn math_microbenchmark_setup_tolerates_preinitialized_rayon_pool() {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build_global();

    let mut input = math_microbenchmark_support::math_microbenchmark_setup();
    math_microbenchmark_support::run_math_microbenchmark_iters(&mut input, 1);
    assert!(input.egraph.get_size("MIntegral") > 0);
    assert!(input.egraph.get_size("MAdd") > 0);
    assert!(input.egraph.get_size("MMul") > 0);
}
