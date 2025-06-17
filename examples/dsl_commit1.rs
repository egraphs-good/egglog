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
}

fn main() {
    env_logger::init();
    let _eq0 = Eq::<MyTx>::new_eq_item(&Var::new_var_item(3), &Var::new_var_item(4));
    let eq1 = Eq::<MyTx>::new_eq_item(&Var::new_var_item(4), &Var::new_var_item(5));
    let eq2 = Eq::<MyTx>::new_eq_item(&Var::new_var_item(3), &Var::new_var_item(5));
    eq2.commit();
    eq1.commit();

    MyTx::sgl().to_dot(PathBuf::from_str("egraph").unwrap());
}

basic_tx_vt!(MyTx);
