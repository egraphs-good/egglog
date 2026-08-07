//! Tests for the e-graph introspection a primitive body can reach:
//! `Read::enodes_for_eclass`, `Read::constructor_schema` /
//! `function_schema` / `table_subtype`, and `Core::rebuild_container`.

use egglog::prelude::*;
use egglog::sort::{MapContainer, VecContainer};
use egglog::{ApiError, Core, Error, Read, Value};
use std::any::TypeId;

const MATH: &str = "
(datatype Math
  (Num i64)
  (Var String)
  (Add Math Math))
(function cost (Math) i64 :no-merge)
";

/// `enodes_for_eclass` returns exactly the rows a full scan would, filtered on
/// the eclass column — that equivalence is the invariant the indexed path has
/// to preserve.
#[test]
fn enodes_for_eclass_agrees_with_a_filtered_scan() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        &format!(
            "{MATH}
(let $a (Add (Num 1) (Num 2)))
(let $b (Add (Num 2) (Num 1)))
(union $a $b)
(let $c (Add (Num 9) (Num 9)))
"
        ),
    )?;

    // Every eclass `Add` produces, so the comparison covers a class with two
    // e-nodes and one with a single e-node.
    let mut eclasses: Vec<Value> = Vec::new();
    egraph.constructor_enodes("Add", |enode| eclasses.push(enode.eclass))?;
    eclasses.dedup();
    assert!(eclasses.len() >= 2, "expected several Add eclasses");

    let mut total_indexed = 0;
    for eclass in eclasses {
        let mut scanned: Vec<Vec<Value>> = Vec::new();
        egraph.constructor_enodes("Add", |enode| {
            if enode.eclass == eclass {
                scanned.push(enode.children.to_vec());
            }
        })?;

        let mut indexed: Vec<Vec<Value>> = Vec::new();
        egraph.read(|state| {
            state.enodes_for_eclass("Add", eclass, |enode| {
                indexed.push(enode.children.to_vec());
            })
        })?;

        scanned.sort();
        indexed.sort();
        assert_eq!(indexed, scanned, "mismatch for eclass {eclass:?}");
        total_indexed += indexed.len();
    }

    // Guards against both sides being vacuously empty: every row of the table
    // has to have been reached through the indexed lookup.
    let rows = egraph.update(|state| Ok(state.table_size("Add")))?.unwrap();
    assert_eq!(total_indexed, rows, "indexed lookup missed rows");
    Ok(())
}

#[test]
fn enodes_for_eclass_rejects_a_function_table() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(None, MATH)?;
    let err = egraph
        .read(|state| state.enodes_for_eclass("cost", Value::new_const(0), |_| {}))
        .unwrap_err();
    assert!(
        matches!(err, Error::ApiError(ApiError::WrongSubtype { .. })),
        "expected WrongSubtype, got {err}"
    );
    Ok(())
}

/// The schema accessors report a table's declared signature, and split by
/// subtype the way the rest of `Read` does.
#[test]
fn schema_accessors_report_the_declaration() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(None, MATH)?;

    egraph.read(|state| {
        let add = state.constructor_schema("Add").unwrap();
        let input: Vec<&str> = add.input.iter().map(|sort| sort.name()).collect();
        assert_eq!(input, ["Math", "Math"]);
        assert_eq!(add.output.name(), "Math");

        let cost = state.function_schema("cost").unwrap();
        assert_eq!(cost.output.name(), "i64");

        // Each rejects the other's subtype rather than answering.
        assert!(matches!(
            state.constructor_schema("cost"),
            Err(Error::ApiError(ApiError::WrongSubtype { .. }))
        ));
        assert!(matches!(
            state.function_schema("Add"),
            Err(Error::ApiError(ApiError::WrongSubtype { .. }))
        ));
        assert!(matches!(
            state.constructor_schema("nonesuch"),
            Err(Error::ApiError(ApiError::MissingTable { .. }))
        ));

        // `table_subtype` answers for either subtype without erroring.
        assert_eq!(
            state.table_subtype("Add"),
            Some(egglog::ast::FunctionSubtype::Constructor)
        );
        assert_eq!(
            state.table_subtype("cost"),
            Some(egglog::ast::FunctionSubtype::Custom)
        );
        assert_eq!(state.table_subtype("nonesuch"), None);
    });
    Ok(())
}

/// `rebuild_container` remaps a container's contents and interns the result,
/// leaving the original alone.
#[test]
fn rebuild_container_remaps_contents() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        &format!(
            "{MATH}
(sort MathVec (Vec Math))
(let $one (Num 1))
(let $two (Num 2))
(let $vec (vec-of $one $two))
(let $swapped (vec-of $two $one))
"
        ),
    )?;

    let one = egraph.eval_expr(&exprs::var("$one"))?.1;
    let two = egraph.eval_expr(&exprs::var("$two"))?.1;
    let vec = egraph.eval_expr(&exprs::var("$vec"))?.1;
    let swapped = egraph.eval_expr(&exprs::var("$swapped"))?.1;

    let rebuilt = egraph.update(|mut state| {
        Ok(
            state.rebuild_container(TypeId::of::<VecContainer>(), vec, &|value| {
                if value == one {
                    two
                } else if value == two {
                    one
                } else {
                    value
                }
            }),
        )
    })?;
    assert_eq!(rebuilt, swapped, "the swap should intern to the same vec");

    // A remap that changes nothing hands back the original value.
    let unchanged = egraph.update(|mut state| {
        Ok(state.rebuild_container(TypeId::of::<VecContainer>(), vec, &|value| value))
    })?;
    assert_eq!(unchanged, vec);

    // A value that is not a container of that type is returned untouched.
    let not_a_container = egraph.update(|mut state| {
        Ok(state.rebuild_container(TypeId::of::<MapContainer>(), vec, &|value| value))
    })?;
    assert_eq!(not_a_container, vec);
    Ok(())
}
