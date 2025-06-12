use egglog_bridge::{
    add_expressions, define_rule, ColumnTy, DefaultVal, EGraph, FunctionConfig, MergeFn,
};
use mimalloc::MiMalloc;
use num_rational::Rational64;
use web_time::Instant;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[allow(clippy::disallowed_macros)]
fn main() {
    env_logger::init();
    #[cfg(feature = "serial_examples")]
    {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .unwrap();
    }
    #[cfg(not(feature = "serial_examples"))]
    {
        rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build_global()
            .unwrap();
    }
    for _ in 0..2 {
        let start = Instant::now();
        const N: usize = 12;

        let mut egraph = EGraph::default();
        let rational_ty = egraph.base_values_mut().register_type::<Rational64>();
        let string_ty = egraph.base_values_mut().register_type::<&'static str>();
        // tables
        let diff = egraph.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
            default: DefaultVal::FreshId,
            merge: MergeFn::UnionId,
            name: "diff".into(),
            can_subsume: false,
        });
        let integral = egraph.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
            default: DefaultVal::FreshId,
            merge: MergeFn::UnionId,
            name: "integral".into(),
            can_subsume: false,
        });

        let add = egraph.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
            default: DefaultVal::FreshId,
            merge: MergeFn::UnionId,
            name: "add".into(),
            can_subsume: false,
        });
        let sub = egraph.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
            default: DefaultVal::FreshId,
            merge: MergeFn::UnionId,
            name: "sub".into(),
            can_subsume: false,
        });

        let mul = egraph.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
            default: DefaultVal::FreshId,
            merge: MergeFn::UnionId,
            name: "mul".into(),
            can_subsume: false,
        });

        let div = egraph.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
            default: DefaultVal::FreshId,
            merge: MergeFn::UnionId,
            name: "div".into(),
            can_subsume: false,
        });

        let pow = egraph.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
            default: DefaultVal::FreshId,
            merge: MergeFn::UnionId,
            name: "pow".into(),
            can_subsume: false,
        });

        let ln = egraph.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id, ColumnTy::Id],
            default: DefaultVal::FreshId,
            merge: MergeFn::UnionId,
            name: "ln".into(),
            can_subsume: false,
        });

        let sqrt = egraph.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id, ColumnTy::Id],
            default: DefaultVal::FreshId,
            merge: MergeFn::UnionId,
            name: "sqrt".into(),
            can_subsume: false,
        });

        let sin = egraph.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id, ColumnTy::Id],
            default: DefaultVal::FreshId,
            merge: MergeFn::UnionId,
            name: "sin".into(),
            can_subsume: false,
        });

        let cos = egraph.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id, ColumnTy::Id],
            default: DefaultVal::FreshId,
            merge: MergeFn::UnionId,
            name: "cos".into(),
            can_subsume: false,
        });

        let rat = egraph.add_table(FunctionConfig {
            schema: vec![ColumnTy::Base(rational_ty), ColumnTy::Id],
            default: DefaultVal::FreshId,
            merge: MergeFn::UnionId,
            name: "rat".into(),
            can_subsume: false,
        });

        let var = egraph.add_table(FunctionConfig {
            schema: vec![ColumnTy::Base(string_ty), ColumnTy::Id],
            default: DefaultVal::FreshId,
            merge: MergeFn::UnionId,
            name: "var".into(),
            can_subsume: false,
        });

        let zero = egraph.base_value_constant(Rational64::new(0, 1));
        let one = egraph.base_value_constant(Rational64::new(1, 1));
        let neg1 = egraph.base_value_constant(Rational64::new(-1, 1));
        let two = egraph.base_value_constant(Rational64::new(2, 1));
        let rules = [
            define_rule! {
                [egraph] ((-> (add x y) id)) => ((set (add y x) id))
            },
            define_rule! {
                [egraph] ((-> (mul x y) id)) => ((set (mul y x) id))
            },
            define_rule! {
                [egraph] ((-> (add x (add y z)) id)) => ((set (add (add x y) z) id))
            },
            define_rule! {
                [egraph] ((-> (mul x (mul y z)) id)) => ((set (mul (mul x y) z) id))
            },
            define_rule! {
                [egraph] ((-> (sub x y) id)) => ((set (add x (mul (rat {neg1.clone()}) y)) id))
            },
            define_rule! {
                [egraph] ((-> (add a (rat {zero.clone()})) id)) => ((union a id))
            },
            define_rule! {
                [egraph] ((-> (rat {zero.clone()}) z_id) (-> (mul a z_id) id))
                        => ((union id z_id))
            },
            define_rule! {
                [egraph] ((-> (mul a (rat {one.clone()})) id)) => ((union a id))
            },
            define_rule! {
                [egraph] ((-> (sub x x) id)) => ((union id (rat {zero})))
            },
            define_rule! {
                [egraph] ((-> (mul x (add b c)) id)) => ((set (add (mul x b) (mul x c)) id))
            },
            define_rule! {
                [egraph] ((-> (add (mul x a) (mul x b)) id)) => ((set (mul x (add a b)) id))
            },
            define_rule! {
                [egraph] ((-> (mul (pow a b) (pow a c)) id)) => ((set (pow a (add b c)) id))
            },
            define_rule! {
                [egraph] ((-> (pow x (rat {one.clone()})) id)) => ((union x id))
            },
            define_rule! {
                [egraph] ((-> (pow x (rat {two})) id)) => ((set (mul x x) id))
            },
            define_rule! {
                [egraph] ((-> (diff x (add a b)) id)) => ((set (add (diff x a) (diff x b)) id))
            },
            define_rule! {
                [egraph] ((-> (diff x (mul a b)) id)) => ((set (add (mul a (diff x b)) (mul b (diff x a))) id))
            },
            define_rule! {
                [egraph] ((-> (diff x (sin x)) id)) => ((set (cos x) id))
            },
            define_rule! {
                [egraph] ((-> (diff x (cos x)) id)) => ((set (mul (rat {neg1.clone()}) (sin x)) id))
            },
            define_rule! {
                [egraph] ((-> (integral (rat {one}) x) id)) => ((union id x))
            },
            define_rule! {
                [egraph] ((-> (integral (cos x) x) id)) => ((set (sin x) id))
            },
            define_rule! {
                [egraph] ((-> (integral (sin x) x) id)) => ((set (mul (rat {neg1}) (cos x)) id))
            },
            define_rule! {
                [egraph] ((-> (integral (add f g) x) id)) => ((set (add (integral f x) (integral g x)) id))
            },
            define_rule! {
                [egraph] ((-> (integral (sub f g) x) id)) => ((set (sub (integral f x) (integral g x)) id))
            },
            define_rule! {
                [egraph] ((-> (integral (mul a b) x) id))
                => ((set (sub (mul a (integral b x))
                              (integral (mul (diff x a) (integral b x)) x)) id))
            },
        ];
        {
            let one = egraph.base_values().get(Rational64::new(1, 1));
            let two = egraph.base_values().get(Rational64::new(2, 1));
            let three = egraph.base_values().get(Rational64::new(3, 1));
            let seven = egraph.base_values().get(Rational64::new(7, 1));
            let x_str = egraph.base_values_mut().get::<&'static str>("x");
            let y_str = egraph.base_values_mut().get::<&'static str>("y");
            let five_str = egraph.base_values_mut().get::<&'static str>("five");
            add_expressions! {
                [egraph]

                (integral (ln (var x_str)) (var x_str))
                (integral (add (var x_str) (cos (var x_str))) (var x_str))
                (integral (mul (cos (var x_str)) (var x_str)) (var x_str))
                (diff (var x_str)
                    (add (rat one) (mul (rat two) (var x_str))))
                (diff (var x_str)
                    (sub (pow (var x_str) (rat three)) (mul (rat seven) (pow (var x_str) (rat two)))))
                (add
                    (mul (var y_str) (add (var x_str) (var y_str)))
                    (sub (add (var x_str) (rat two)) (add (var x_str) (var x_str))))
                (div (rat one)
                     (sub (div (add (rat one) (sqrt (var five_str))) (rat two))
                          (div (sub (rat one) (sqrt (var five_str))) (rat two))))
            }
        }

        for i in 0..N {
            let start = Instant::now();
            let changed = egraph.run_rules(&rules).unwrap().changed;
            println!(
                "Iteration {i} finished, duration={:?} (saturated={})",
                start.elapsed(),
                !changed
            );
            if !changed {
                break;
            }
        }

        println!("Function diff has size {}", egraph.table_size(diff));
        println!("Function integral has size {}", egraph.table_size(integral));
        println!("Function add has size {}", egraph.table_size(add));
        println!("Function sub has size {}", egraph.table_size(sub));
        println!("Function mul has size {}", egraph.table_size(mul));
        println!("Function div has size {}", egraph.table_size(div));
        println!("Function pow has size {}", egraph.table_size(pow));
        println!("Function ln has size {}", egraph.table_size(ln));
        println!("Function sqrt has size {}", egraph.table_size(sqrt));
        println!("Function sin has size {}", egraph.table_size(sin));
        println!("Function cos has size {}", egraph.table_size(cos));
        println!("Function rat has size {}", egraph.table_size(rat));
        println!("Function var has size {}", egraph.table_size(var));
        println!("Total time: {:?}", start.elapsed());
        // NB: We don't include dropping the egraph in the time. egglog main doesn't drop the
        // egraph either.
    }
}
