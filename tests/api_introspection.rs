//! Tests for the e-graph introspection a primitive body can reach:
//! `Read::constructor_enodes_for_eclass`, `Read::eclass_enodes`,
//! `Read::constructor_schema` /
//! `function_schema` / `table_subtype`, and `Core::map_container`.

use egglog::ast::Span;
use egglog::constraint::{SimpleTypeConstraint, TypeConstraint};
use egglog::prelude::*;
use egglog::sort::{I64Sort, MapContainer, S, StringSort, VecContainer};
use egglog::{ApiError, Core, Error, Primitive, Read, ReadPrim, ReadState, Value};
use std::any::TypeId;

const MATH: &str = "
(datatype Math
  (Num i64)
  (Var String)
  (Add Math Math))
(function cost (Math) i64 :no-merge)
";

/// `constructor_enodes_for_eclass` returns exactly the rows a full scan would,
/// filtered on the eclass column — that equivalence is the invariant the
/// indexed path has to preserve.
#[test]
fn constructor_enodes_for_eclass_agrees_with_a_filtered_scan() -> Result<(), Error> {
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
            state.constructor_enodes_for_eclass("Add", eclass, |enode| {
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
fn constructor_enodes_for_eclass_rejects_a_function_table() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(None, MATH)?;
    let err = egraph
        .read(|state| state.constructor_enodes_for_eclass("cost", Value::new_const(0), |_| {}))
        .unwrap_err();
    assert!(
        matches!(err, Error::ApiError(ApiError::WrongSubtype { .. })),
        "expected WrongSubtype, got {err}"
    );
    Ok(())
}

/// `eclass_enodes` finds an e-class's e-nodes across every constructor, and
/// says which one each came from — the caller does not have to know the sort,
/// or which tables to ask.
#[test]
fn eclass_enodes_spans_every_constructor() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        &format!(
            "{MATH}
(constructor Neg (Math) Math)
(datatype Other (Wrap Math))
(let $a (Add (Num 1) (Num 2)))
(union $a (Neg (Num 3)))
(let $unrelated (Wrap $a))
"
        ),
    )?;
    let eclass = egraph.eval_expr(&exprs::var("$a"))?.1;

    let mut found: Vec<String> = Vec::new();
    egraph.read(|state| {
        state.eclass_enodes(eclass, |enode| {
            assert_eq!(enode.eclass, eclass);
            found.push(enode.name.to_owned());
        })
    })?;
    found.sort();

    // Both of the class's constructors, and neither the `Other`-sorted
    // constructor that merely references it nor the `cost` function table.
    assert_eq!(found, ["Add", "Neg"]);
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

/// `map_container` remaps a container's contents and interns the result,
/// leaving the original alone, and reports a value that is not a container of
/// the given type rather than passing it through.
#[test]
fn map_container_remaps_contents() -> Result<(), Error> {
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
            state.map_container(TypeId::of::<VecContainer>(), vec, &|value| {
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
    let rebuilt = rebuilt.expect("a Vec container");
    assert_eq!(rebuilt, swapped, "the swap should intern to the same vec");

    // A remap that changes nothing hands back the original value.
    let unchanged = egraph.update(|mut state| {
        Ok(state.map_container(TypeId::of::<VecContainer>(), vec, &|value| value))
    })?;
    assert_eq!(unchanged, Some(vec));

    // A value that is not a container of that type is reported, not silently
    // handed back.
    let not_a_container = egraph.update(|mut state| {
        Ok(state.map_container(TypeId::of::<MapContainer>(), vec, &|value| value))
    })?;
    assert_eq!(not_a_container, None);
    Ok(())
}

/// A read primitive answering with a constructor's arity, so a program can
/// observe whether the declarations reached the primitive body.
#[derive(Clone)]
struct ConstructorArity;
impl Primitive for ConstructorArity {
    fn name(&self) -> &str {
        "constructor-arity"
    }
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![StringSort.to_arcsort(), I64Sort.to_arcsort()],
            span.clone(),
        )
        .into_box()
    }
}
impl ReadPrim for ConstructorArity {
    fn apply<'a, 'db>(&self, state: ReadState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let name = state.base_values().unwrap::<S>(args[0]).0;
        let arity = state.constructor_schema(&name).ok()?.input.len();
        Some(state.base_values().get::<i64>(i64::try_from(arity).ok()?))
    }
}

/// The declarations reach a primitive body during *rule execution*, which takes
/// a different route to its execution state than a top-level command does
/// (`run_rules` rather than `with_execution_state`).
#[test]
fn a_primitive_can_resolve_a_signature_from_inside_a_rule() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.add_read_primitive(ConstructorArity, None);
    egraph.parse_and_run_program(None, MATH)?;
    egraph.parse_and_run_program(
        None,
        r#"
(function arity-of (String) i64 :no-merge)
(rule () ((set (arity-of "Add") (constructor-arity "Add"))) :naive)
(run 1)
(check (= (arity-of "Add") 2))
"#,
    )?;
    Ok(())
}
