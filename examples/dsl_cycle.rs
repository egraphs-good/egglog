use std::path::PathBuf;

use egglog::basic_tx_no_vt;
use egglog::egglog_ty;

#[egglog_ty]
enum A {
    ACon { b: B },
}

#[egglog_ty]
enum B {
    BCon { a: A },
    Empty {},
}

/// NB: this should panic because Cycle is not allowed.
///
/// Only DAG is supported.
fn main() {
    env_logger::init();
    let mut a = A::new_a_con(&B::<MyTx>::new_empty());
    let _b = B::<MyTx>::new_empty();
    a.set_b(&B::new_b_con(&a));
    MyTx::sgl().to_dot(PathBuf::from("egraph"));
}

basic_tx_no_vt!(MyTx);
