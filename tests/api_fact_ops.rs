//! Tests for the fact-ops API on the [`Read`] / [`Write`]
//! traits, accessed outside a rule via [`EGraph::update`].
//!
//! `update` flushes pending writes only when the closure
//! returns, so a read in the same closure won't see a preceding
//! write — tests below split write and read into separate closures
//! to reflect that.

use egglog::prelude::*;
use egglog::{Error, RawValues};

fn make_eg_with_function() -> EGraph {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(function f (i64) i64 :no-merge)")
        .unwrap();
    eg
}

#[test]
fn test_set_then_lookup_function() -> Result<(), Error> {
    let mut eg = make_eg_with_function();
    eg.update(|mut fs| fs.set("f", (1_i64,), 42_i64))?;
    let got = eg
        .update(|fs| fs.lookup("f", 1_i64))?
        .map(|v| eg.value_to_base::<i64>(v));
    assert_eq!(got, Some(42));
    Ok(())
}

#[test]
fn test_lookup_missing_returns_none() -> Result<(), Error> {
    let mut eg = make_eg_with_function();
    let got = eg.update(|fs| fs.lookup("f", 999_i64))?;
    assert_eq!(got, None);
    Ok(())
}

#[test]
fn test_contains_function() -> Result<(), Error> {
    let mut eg = make_eg_with_function();
    eg.update(|mut fs| fs.set("f", (1_i64,), 42_i64))?;
    let (has1, has999) = eg.update(|fs| -> Result<_, Error> {
        Ok((fs.contains("f", 1_i64)?, fs.contains("f", 999_i64)?))
    })?;
    assert!(has1);
    assert!(!has999);
    Ok(())
}

#[test]
fn test_remove_function() -> Result<(), Error> {
    let mut eg = make_eg_with_function();
    eg.update(|mut fs| fs.set("f", (1_i64,), 42_i64))?;
    assert!(eg.update(|fs| fs.contains("f", 1_i64))?);

    eg.update(|mut fs| fs.remove("f", 1_i64))?;
    assert!(!eg.update(|fs| fs.contains("f", 1_i64))?);

    // Removing again is a no-op.
    eg.update(|mut fs| fs.remove("f", 1_i64))?;
    assert!(!eg.update(|fs| fs.contains("f", 1_i64))?);
    Ok(())
}

#[test]
fn test_relation_add_and_contains() -> Result<(), Error> {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(relation R (i64 i64))")?;
    eg.update(|mut fs| -> Result<_, Error> {
        fs.add("R", (1_i64, 2_i64))?;
        Ok(())
    })?;
    let (a, b, c) = eg.update(|fs| -> Result<_, Error> {
        Ok((
            fs.contains("R", (1_i64, 2_i64))?,
            fs.contains("R", (1_i64, 3_i64))?,
            fs.contains("R", (2_i64, 1_i64))?,
        ))
    })?;
    assert!(a);
    assert!(!b);
    assert!(!c);
    Ok(())
}

#[test]
fn test_relation_remove() -> Result<(), Error> {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(relation R (i64 i64))")?;
    eg.update(|mut fs| -> Result<_, Error> {
        fs.add("R", (1_i64, 2_i64))?;
        fs.add("R", (3_i64, 4_i64))?;
        Ok(())
    })?;
    let (a, b) = eg.update(|fs| -> Result<_, Error> {
        Ok((
            fs.contains("R", (1_i64, 2_i64))?,
            fs.contains("R", (3_i64, 4_i64))?,
        ))
    })?;
    assert!(a);
    assert!(b);

    eg.update(|mut fs| fs.remove("R", (1_i64, 2_i64)))?;
    let (a, b) = eg.update(|fs| -> Result<_, Error> {
        Ok((
            fs.contains("R", (1_i64, 2_i64))?,
            fs.contains("R", (3_i64, 4_i64))?,
        ))
    })?;
    assert!(!a);
    assert!(b);
    Ok(())
}

#[test]
fn test_constructor_add_returns_id() -> Result<(), Error> {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype List (Cons i64 List) (Nil))")?;

    // Zero-arg constructor uses RawValues(vec![]) — `()` would be a Unit column.
    // Calling add again with the same inputs returns the same id.
    let (cons, cons2, nil) = eg.update(|mut fs| -> Result<_, Error> {
        let nil = fs.add("Nil", RawValues(vec![]))?;
        let cons = fs.add("Cons", (1_i64, nil))?;
        let cons2 = fs.add("Cons", (1_i64, nil))?;
        Ok((cons, cons2, nil))
    })?;
    assert_eq!(cons, cons2);

    let (nil_present, cons_present) = eg.update(|fs| -> Result<_, Error> {
        Ok((
            fs.contains("Nil", RawValues(vec![]))?,
            fs.contains("Cons", (1_i64, nil))?,
        ))
    })?;
    assert!(nil_present);
    assert!(cons_present);
    Ok(())
}

#[test]
fn test_eclass_of_constructor() -> Result<(), Error> {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype List (Cons i64 List) (Nil))")?;
    let (cons, nil) = eg.update(|mut fs| -> Result<_, Error> {
        let nil = fs.add("Nil", RawValues(vec![]))?;
        let cons = fs.add("Cons", (1_i64, nil))?;
        Ok((cons, nil))
    })?;
    let (existing, absent) = eg.update(|fs| -> Result<_, Error> {
        Ok((
            fs.eclass_of("Cons", (1_i64, nil))?,
            fs.eclass_of("Cons", (99_i64, nil))?,
        ))
    })?;
    assert_eq!(existing, Some(cons));
    assert!(absent.is_none());
    Ok(())
}

#[test]
fn test_lookup_on_constructor_errors() {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype List (Cons i64 List) (Nil))")
        .unwrap();
    let result = eg.update(|fs| fs.lookup("Cons", (1_i64, 0_i64)));
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Cons") && err.contains("constructor"),
        "got: {err}"
    );
}

#[test]
fn test_eclass_of_on_function_errors() {
    let mut eg = make_eg_with_function();
    let result = eg.update(|fs| fs.eclass_of("f", 1_i64));
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("`f`") && err.contains("function"),
        "got: {err}"
    );
}

#[test]
fn test_set_constructor_errors() {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype List (Cons i64 List) (Nil))")
        .unwrap();
    let result = eg.update(|mut fs| fs.set("Nil", RawValues(vec![]), 0_i64));
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Nil") && err.contains("constructor"),
        "got: {err}"
    );
}

#[test]
fn test_add_function_errors() {
    let mut eg = make_eg_with_function();
    let result = eg.update(|mut fs| fs.add("f", 1_i64));
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("`f`") && err.contains("function"),
        "got: {err}"
    );
}

#[test]
fn test_wrong_arity_errors() {
    // Sending 2 args where the table expects 1.
    let mut eg = make_eg_with_function();
    let result = eg.update(|mut fs| fs.set("f", (1_i64, 2_i64), 42_i64));
    let err = result.unwrap_err().to_string();
    assert!(err.contains("expected 1 input"), "got: {err}");
}

#[test]
fn test_wrong_arity_on_add_errors() {
    // `add` should arity-check just like `set`.
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype List (Cons i64 List) (Nil))")
        .unwrap();
    let result = eg.update(|mut fs| {
        // Cons expects (i64, List), so 1 arg is wrong arity.
        fs.add("Cons", 1_i64)
    });
    let err = result.unwrap_err().to_string();
    assert!(err.contains("expected 2 input"), "got: {err}");
}

#[test]
fn test_union_same_value_is_noop() -> Result<(), Error> {
    // Unioning a value with itself should be a no-op, not an error.
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype List (Cons i64 List) (Nil))")?;
    let nil = eg.update(|mut fs| fs.add("Nil", RawValues(vec![])))?;
    eg.update(|mut fs| fs.union(nil, nil))?;
    Ok(())
}

#[test]
fn test_constructor_enodes_on_empty_table() -> Result<(), Error> {
    // Iterating an empty constructor table should never call the
    // callback, not error.
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype List (Cons i64 List) (Nil))")?;
    let mut count = 0;
    eg.update(|fs| fs.constructor_enodes("Cons", |_| count += 1))?;
    assert_eq!(count, 0);
    Ok(())
}

#[test]
fn test_set_replaces_function_value() -> Result<(), Error> {
    let mut eg = make_eg_with_function();
    eg.update(|mut fs| fs.set("f", (5_i64,), 50_i64))?;
    let got = eg
        .update(|fs| fs.lookup("f", 5_i64))?
        .map(|v| eg.value_to_base::<i64>(v));
    assert_eq!(got, Some(50));
    Ok(())
}

#[test]
fn test_set_unknown_table_errors() {
    let mut eg = EGraph::default();
    let result = eg.update(|mut fs| fs.set("nope", (1_i64,), 2_i64));
    assert!(result.is_err());
}

#[test]
fn test_lookup_unknown_table_errors() {
    let mut eg = EGraph::default();
    let result = eg.update(|fs| fs.lookup("nope", 1_i64));
    assert!(result.is_err());
}

#[test]
fn test_higher_arity_function() -> Result<(), Error> {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(function g (i64 i64 i64) i64 :no-merge)")?;
    eg.update(|mut fs| fs.set("g", (1_i64, 2_i64, 3_i64), 7_i64))?;
    let (v, has) = eg.update(|fs| -> Result<_, Error> {
        let v = fs
            .lookup("g", (1_i64, 2_i64, 3_i64))?
            .map(|val| fs.value_to_base::<i64>(val));
        let has = fs.contains("g", (1_i64, 2_i64, 3_i64))?;
        Ok((v, has))
    })?;
    assert_eq!(v, Some(7));
    assert!(has);
    Ok(())
}

#[test]
fn test_string_inputs() -> Result<(), Error> {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(function name-length (String) i64 :no-merge)")?;
    eg.update(|mut fs| fs.set("name-length", ("hello".to_string(),), 5_i64))?;
    let got = eg
        .update(|fs| fs.lookup("name-length", "hello".to_string()))?
        .map(|v| eg.value_to_base::<i64>(v));
    assert_eq!(got, Some(5));
    Ok(())
}
