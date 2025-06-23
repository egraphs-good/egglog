use egglog::basic_tx_rx_vt;
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

#[egglog_func(output=Root)]
struct Selected {}

fn main() {
    env_logger::init();
    let end = Cons::<MyTx>::new_end();
    let node1 = Cons::new_value(1, &end);
    let mut node2 = Cons::new_value(2, &node1);
    let _node3 = Cons::new_value(3, &node2);
    let mut root = Root::new_v(&VecCon::new(vec![&node2]));
    println!("node2's current version is {}", node2.cur_sym());
    // node2.set_v(4).stage();
    root.commit();

    println!("node2's current version is {}", node2.cur_sym());
    node2.set_v(6).stage();
    root.commit();
    root.pull();
    // let old_root = root.clone();
    Selected::<MyTx>::set((), &root);
    root.locate_latest();
    Selected::<MyTx>::set((), &root);
    // MyTx::on_union(&root, &old_root);
    // MyTx::on_union(&node2, &end);
    // MyTx::on_union(&node1, &end);
    MyTx::sgl().egraph_to_dot("egraph0.dot".into());
    MyTx::sgl().wag_to_dot("wag0.dot".into());
    let selected = Selected::<MyTx>::get(());
    println!("selected is {:?}", selected.cur_sym());
    MyTx::sgl().egraph_to_dot("egraph1.dot".into());
    MyTx::sgl().wag_to_dot("wag1.dot".into());
}

basic_tx_rx_vt!(MyTx);
