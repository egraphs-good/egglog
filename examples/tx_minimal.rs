use egglog::basic_tx_minimal;
use egglog::{egglog_func, egglog_ty};

#[egglog_ty]
enum Cons {
    Value { v: i64, con: Box<Cons> },
    End {},
}

#[egglog_ty]
struct VecCon {
    v: Vec<Cons>,
}

#[egglog_ty]
enum Root {
    V { v: VecCon },
}
#[egglog_func(output= Root)]
struct F {}

fn main() {
    env_logger::init();
    let node1 = Cons::new_value(3, &Cons::<MyRx>::new_end());
    let node2 = Cons::new_value(2, &node1);
    let node3 = Cons::new_value(1, &node2);
    let _root = Root::new_v(&VecCon::new(vec![&node2]));
    let root = Root::new_v(&VecCon::new(vec![&node3]));
    F::set((), &root);
    MyRx::sgl().to_dot("egraph.dot".into());
}

basic_tx_minimal!(MyRx);
