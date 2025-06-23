use egglog::{basic_tx_rx_vt, egglog_func, egglog_ty};
// #[egglog_ty]
// enum Expr{
//     Add { a:Box<Expr>, b:Box<Expr>},
//     Const { s:i64}
// }

#[egglog_ty]
enum Expr {
    Add { v: Box<Expr>, con: Box<Expr> },
    Const { s: i64 },
}

#[egglog_func(output=Expr)]
struct LeadTo {
    e: Expr,
}

fn main() {
    let a = Expr::<MyTx>::new_const(3);
    let b = Expr::new_const(5);
    let add = Expr::new_add(&a, &b);
    let c = Expr::<MyTx>::new_const(15);
    add.commit();
    add.commit();
    LeadTo::set((&add,), &c);
    MyTx::sgl().wag_to_dot("dsl_func.dot".into());
    MyTx::sgl().egraph_to_dot("dsl_func_egraph.dot".into());
}

basic_tx_rx_vt!(MyTx);
