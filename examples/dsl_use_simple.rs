use egglog::basic_tx_vt;
use egglog::egglog_ty;
use std::time::Instant;

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

fn main() {
    env_logger::init();
    let a = Instant::now();

    let mut count = 0;
    let root = loop {
        let node1 = Cons::new_value(5, &Cons::<MyTx>::new_end());
        let node2 = Cons::new_value(5 + 1, &node1);
        let node3 = Cons::new_value(2, &node2);
        let root = Root::new_v(&VecCon::new(vec![&node1, &node2, &node3]));
        let _m = root.cur_sym();
        if count == 99 {
            break root;
        }
        count += 1;
    };
    root.commit();
    println!("push elpased: {:?}", a.elapsed());
    MyTx::sgl().to_dot("egraph.dot".into());
}

basic_tx_vt!(MyTx);
