use std::{path::PathBuf, str::FromStr};

use egglog::basic_tx_vt;
use egglog::egglog_ty;

#[egglog_ty]
enum Eq {
    EqItem { v1: Var, v2: Var },
}
#[egglog_ty]
enum Var {
    VarItem { num: i64 },
    Expr { eq: Eq },
}

fn main() {
    env_logger::init();
    let mut v0 = Var::<MyTx>::new_var_item(1);
    let mut v1 = Var::new_var_item(1);
    let eq0 = Eq::new_eq_item(&v0, &v1);
    eq0.commit();
    MyTx::sgl().to_dot(PathBuf::from_str("egraph0").unwrap());

    v1.set_num(4).stage();
    eq0.commit();
    MyTx::sgl().to_dot(PathBuf::from_str("egraph1").unwrap());

    v0.set_num(4).stage();
    eq0.commit();
    MyTx::sgl().to_dot(PathBuf::from_str("egraph2").unwrap());
}

basic_tx_vt!(MyTx);
