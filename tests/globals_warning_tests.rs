//! Tests for global variable naming warnings.
//! These tests use testing_logger which requires exclusive access to the global logger,
//! so they are in a separate test binary to avoid conflicts with env_logger in other tests.

use egglog::*;
use serial_test::serial;

#[test]
#[serial]
fn globals_missing_prefix_warns_by_default() {
    testing_logger::setup();

    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(None, "(let value 41)")
        .unwrap();

    testing_logger::validate(|logs| {
        let bodies: Vec<_> = logs.iter().map(|entry| entry.body.clone()).collect();
        assert!(
            bodies
                .iter()
                .any(|body| body.contains("Global `value` should start with `$`")),
            "expected warning about missing global prefix, got logs: {:?}",
            bodies
        );
    });
}

#[test]
#[serial]
fn globals_missing_prefix_warns_for_prefixed_pattern_variable_by_default() {
    testing_logger::setup();

    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(None, "(rule ((= $x 1)) ())")
        .unwrap();

    testing_logger::validate(|logs| {
        let bodies: Vec<_> = logs.iter().map(|entry| entry.body.clone()).collect();
        assert!(
            bodies
                .iter()
                .any(|body| body.contains("Non-global `$x` should not start with `$`")),
            "expected warning about missing global prefix, got logs: {:?}",
            bodies
        );
    });
}

#[test]
#[serial]
fn globals_missing_prefix_warns_for_prefixed_rule_let_by_default() {
    testing_logger::setup();

    let mut egraph = EGraph::default();
    egraph
        .parse_and_run_program(None, "(rule () ((let $y 1)))")
        .unwrap();

    testing_logger::validate(|logs| {
        let bodies: Vec<_> = logs.iter().map(|entry| entry.body.clone()).collect();
        assert!(
            bodies
                .iter()
                .any(|body| body.contains("Non-global `$y` should not start with `$`")),
            "expected warning about missing global prefix, got logs: {:?}",
            bodies
        );
    });
}
