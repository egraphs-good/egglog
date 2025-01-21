use crate::lift_operation;

use super::{PrimitivePrinter, Primitives};

#[test]
fn basic_primitive() {
    let mut prims = Primitives::default();
    let add32 = lift_operation!([&mut prims] fn add32(x: i32, y: i32) -> i32 {
        x + y
    });
    let maybe_add = lift_operation!([&mut prims] fn maybe_add(t: bool, x: i64, y: i64) -> i64 {
        if t {
            x + y
        } else {
            x - y
        }
    });

    let a = prims.get(1i32);
    let b = prims.get(2i32);
    let c = prims.apply_op(add32, &[a, b]).unwrap();

    assert_eq!(prims.unwrap::<i32>(c), 3);

    let t = prims.get(true);
    let f = prims.get(false);
    let x = prims.get(1i64);
    let y = prims.get(5i64);
    let z = prims.apply_op(maybe_add, &[t, x, y]).unwrap();
    let w = prims.apply_op(maybe_add, &[f, x, y]).unwrap();

    assert_eq!(prims.unwrap::<i64>(z), 6);
    assert_eq!(prims.unwrap::<i64>(w), -4);
}

#[test]
#[should_panic]
fn arity_mismatch() {
    let mut prims = Primitives::default();
    let add32 = lift_operation!([&mut prims] fn add32(x: i32, y: i32) -> i32 {
        x + y
    });
    prims.apply_op(add32, &[]);
}

#[test]
#[should_panic]
fn operation_type_mismatch() {
    let mut prims = Primitives::default();
    let add32 = lift_operation!([&mut prims] fn add32(x: i32, y: i32) -> i32 {
        x + y
    });
    let x = prims.get(1i64);
    prims.apply_op(add32, &[x, x]);
}

#[test]
#[should_panic]
fn value_type_mismatch() {
    let mut prims = Primitives::default();
    prims.register_type::<i64>();
    prims.register_type::<i32>();
    let x = prims.get(1i64);
    prims.unwrap::<i32>(x);
}

#[test]
fn prim_printing() {
    let mut prims = Primitives::default();
    prims.register_type::<i64>();
    let ty = prims.get_ty::<i64>();
    let val = prims.get(24i64);
    assert_eq!(
        format!(
            "{:?}",
            PrimitivePrinter {
                prim: &prims,
                ty,
                val
            }
        ),
        "24"
    );
}
