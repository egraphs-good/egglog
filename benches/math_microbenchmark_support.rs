use egglog::prelude::*;
use egglog::sort::S;
use egglog::{ArcSort, Value};
use std::sync::Once;

static CONFIGURE_RAYON: Once = Once::new();

pub struct MathMicroBenchInput {
    pub egraph: egglog::EGraph,
    pub ruleset: String,
}

fn configure_rayon_once() {
    CONFIGURE_RAYON.call_once(|| {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global();
    });
}

fn math_vars(math: &ArcSort, names: &[&'static str]) -> Vec<(&'static str, ArcSort)> {
    names.iter().map(|name| (*name, math.clone())).collect()
}

fn ctor1(ctx: &mut RustRuleContext<'_, '_>, name: &str, a: Value) -> Value {
    ctx.lookup(name, &[a]).unwrap()
}

fn ctor2(ctx: &mut RustRuleContext<'_, '_>, name: &str, a: Value, b: Value) -> Value {
    ctx.lookup(name, &[a, b]).unwrap()
}

fn m_diff(ctx: &mut RustRuleContext<'_, '_>, x: Value, f: Value) -> Value {
    ctor2(ctx, "MDiff", x, f)
}

fn m_integral(ctx: &mut RustRuleContext<'_, '_>, f: Value, x: Value) -> Value {
    ctor2(ctx, "MIntegral", f, x)
}

fn m_add(ctx: &mut RustRuleContext<'_, '_>, a: Value, b: Value) -> Value {
    ctor2(ctx, "MAdd", a, b)
}

fn m_sub(ctx: &mut RustRuleContext<'_, '_>, a: Value, b: Value) -> Value {
    ctor2(ctx, "MSub", a, b)
}

fn m_mul(ctx: &mut RustRuleContext<'_, '_>, a: Value, b: Value) -> Value {
    ctor2(ctx, "MMul", a, b)
}

fn m_div(ctx: &mut RustRuleContext<'_, '_>, a: Value, b: Value) -> Value {
    ctor2(ctx, "MDiv", a, b)
}

fn m_pow(ctx: &mut RustRuleContext<'_, '_>, a: Value, b: Value) -> Value {
    ctor2(ctx, "MPow", a, b)
}

fn m_ln(ctx: &mut RustRuleContext<'_, '_>, a: Value) -> Value {
    ctor1(ctx, "MLn", a)
}

fn m_sqrt(ctx: &mut RustRuleContext<'_, '_>, a: Value) -> Value {
    ctor1(ctx, "MSqrt", a)
}

fn m_sin(ctx: &mut RustRuleContext<'_, '_>, a: Value) -> Value {
    ctor1(ctx, "MSin", a)
}

fn m_cos(ctx: &mut RustRuleContext<'_, '_>, a: Value) -> Value {
    ctor1(ctx, "MCos", a)
}

fn m_const(ctx: &mut RustRuleContext<'_, '_>, n: i64) -> Value {
    ctor1(ctx, "MConst", ctx.base_to_value::<i64>(n))
}

fn m_var(ctx: &mut RustRuleContext<'_, '_>, name: &'static str) -> Value {
    ctor1(ctx, "MVar", ctx.base_to_value::<S>(name.to_owned().into()))
}

fn add_math_rule(
    egraph: &mut egglog::EGraph,
    ruleset: &str,
    rule_name: &str,
    vars: &[(&'static str, ArcSort)],
    facts: Facts<String, String>,
    action: impl Fn(&mut RustRuleContext<'_, '_>, &[Value]) -> Option<()>
    + Clone
    + Send
    + Sync
    + 'static,
) {
    let rule_name: &'static str = rule_name.to_owned().leak();
    rust_rule(egraph, rule_name, ruleset, vars, facts, move |a, b| {
        action(a, b)
    })
    .unwrap();
}

pub fn math_microbenchmark_setup() -> MathMicroBenchInput {
    configure_rayon_once();

    let mut egraph = egglog::EGraph::default();
    egraph
        .parse_and_run_program(
            None,
            r#"
(datatype Math
  (MDiff Math Math)
  (MIntegral Math Math)
  (MAdd Math Math)
  (MSub Math Math)
  (MMul Math Math)
  (MDiv Math Math)
  (MPow Math Math)
  (MLn Math)
  (MSqrt Math)
  (MSin Math)
  (MCos Math)
  (MConst i64)
  (MVar String))
"#,
        )
        .unwrap();

    let math = egraph.get_sort_by_name("Math").unwrap().clone();

    let seed_ruleset = "math_microbenchmark_seed";
    add_ruleset(&mut egraph, seed_ruleset).unwrap();
    add_math_rule(
        &mut egraph,
        seed_ruleset,
        "math_microbenchmark_seed",
        &[],
        Facts(Vec::new()),
        move |ctx, _| {
            let x = m_var(ctx, "x");
            let y = m_var(ctx, "y");
            let five = m_var(ctx, "five");

            let ln_x = m_ln(ctx, x);
            let _ = m_integral(ctx, ln_x, x);

            let cos_x = m_cos(ctx, x);
            let add_x_cos_x = m_add(ctx, x, cos_x);
            let _ = m_integral(ctx, add_x_cos_x, x);

            let cos_x = m_cos(ctx, x);
            let mul_cos_x_x = m_mul(ctx, cos_x, x);
            let _ = m_integral(ctx, mul_cos_x_x, x);

            let c1 = m_const(ctx, 1);
            let c2 = m_const(ctx, 2);
            let mul_2_x = m_mul(ctx, c2, x);
            let add_1_2x = m_add(ctx, c1, mul_2_x);
            let _ = m_diff(ctx, x, add_1_2x);

            let c3 = m_const(ctx, 3);
            let c7 = m_const(ctx, 7);
            let pow_x_3 = m_pow(ctx, x, c3);
            let c2_pow = m_const(ctx, 2);
            let pow_x_2 = m_pow(ctx, x, c2_pow);
            let mul_7_pow = m_mul(ctx, c7, pow_x_2);
            let sub_pow = m_sub(ctx, pow_x_3, mul_7_pow);
            let _ = m_diff(ctx, x, sub_pow);

            let add_x_y = m_add(ctx, x, y);
            let mul_y_add = m_mul(ctx, y, add_x_y);
            let c2_add = m_const(ctx, 2);
            let add_x_2 = m_add(ctx, x, c2_add);
            let add_x_x = m_add(ctx, x, x);
            let sub_add = m_sub(ctx, add_x_2, add_x_x);
            let _ = m_add(ctx, mul_y_add, sub_add);

            let c1 = m_const(ctx, 1);
            let c2 = m_const(ctx, 2);
            let sqrt_five = m_sqrt(ctx, five);
            let add_1_sqrt = m_add(ctx, c1, sqrt_five);
            let div_add = m_div(ctx, add_1_sqrt, c2);
            let c1_sub = m_const(ctx, 1);
            let sub_1_sqrt = m_sub(ctx, c1_sub, sqrt_five);
            let c2_sub = m_const(ctx, 2);
            let div_sub = m_div(ctx, sub_1_sqrt, c2_sub);
            let denom = m_sub(ctx, div_add, div_sub);
            let c1_div = m_const(ctx, 1);
            let _ = m_div(ctx, c1_div, denom);
            Some(())
        },
    );
    run_ruleset(&mut egraph, seed_ruleset).unwrap();

    let ruleset = "math_microbenchmark_rules";
    add_ruleset(&mut egraph, ruleset).unwrap();

    add_math_rule(
        &mut egraph,
        ruleset,
        "add_comm",
        &math_vars(&math, &["a", "b", "add"]),
        facts![(= add (MAdd a b))],
        |ctx, values| {
            let [a, b, add] = values else { unreachable!() };
            let rhs = m_add(ctx, *b, *a);
            ctx.union(*add, rhs);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "mul_comm",
        &math_vars(&math, &["a", "b", "mul"]),
        facts![(= mul (MMul a b))],
        |ctx, values| {
            let [a, b, mul] = values else { unreachable!() };
            let rhs = m_mul(ctx, *b, *a);
            ctx.union(*mul, rhs);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "add_assoc",
        &math_vars(&math, &["a", "b", "c", "add_outer"]),
        facts![(= add_outer (MAdd a (MAdd b c)))],
        |ctx, values| {
            let [a, b, c, add_outer] = values else {
                unreachable!()
            };
            let ab = m_add(ctx, *a, *b);
            let rhs = m_add(ctx, ab, *c);
            ctx.union(*add_outer, rhs);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "mul_assoc",
        &math_vars(&math, &["a", "b", "c", "mul_outer"]),
        facts![(= mul_outer (MMul a (MMul b c)))],
        |ctx, values| {
            let [a, b, c, mul_outer] = values else {
                unreachable!()
            };
            let ab = m_mul(ctx, *a, *b);
            let rhs = m_mul(ctx, ab, *c);
            ctx.union(*mul_outer, rhs);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "sub_to_add_neg",
        &math_vars(&math, &["a", "b", "sub"]),
        facts![(= sub (MSub a b))],
        |ctx, values| {
            let [a, b, sub] = values else { unreachable!() };
            let neg1 = m_const(ctx, -1);
            let neg_b = m_mul(ctx, neg1, *b);
            let rhs = m_add(ctx, *a, neg_b);
            ctx.union(*sub, rhs);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "add_zero",
        &math_vars(&math, &["a", "add"]),
        facts![(= add (MAdd a (MConst 0)))],
        |ctx, values| {
            let [a, add] = values else { unreachable!() };
            ctx.union(*add, *a);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "mul_zero",
        &math_vars(&math, &["a", "mul"]),
        facts![(= mul (MMul a (MConst 0)))],
        |ctx, values| {
            let [a, mul] = values else { unreachable!() };
            let _ = a;
            let z = m_const(ctx, 0);
            ctx.union(*mul, z);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "mul_one",
        &math_vars(&math, &["a", "mul"]),
        facts![(= mul (MMul a (MConst 1)))],
        |ctx, values| {
            let [a, mul] = values else { unreachable!() };
            ctx.union(*mul, *a);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "sub_self_zero",
        &math_vars(&math, &["a", "sub"]),
        facts![(= sub (MSub a a))],
        |ctx, values| {
            let [_, sub] = values else { unreachable!() };
            let z = m_const(ctx, 0);
            ctx.union(*sub, z);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "mul_distrib",
        &math_vars(&math, &["a", "b", "c", "mul"]),
        facts![(= mul (MMul a (MAdd b c)))],
        |ctx, values| {
            let [a, b, c, mul] = values else {
                unreachable!()
            };
            let ab = m_mul(ctx, *a, *b);
            let ac = m_mul(ctx, *a, *c);
            let rhs = m_add(ctx, ab, ac);
            ctx.union(*mul, rhs);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "add_factor",
        &math_vars(&math, &["a", "b", "c", "add"]),
        facts![(= add (MAdd (MMul a b) (MMul a c)))],
        |ctx, values| {
            let [a, b, c, add] = values else {
                unreachable!()
            };
            let bc = m_add(ctx, *b, *c);
            let rhs = m_mul(ctx, *a, bc);
            ctx.union(*add, rhs);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "mul_pow_combine",
        &math_vars(&math, &["a", "b", "c", "mul"]),
        facts![(= mul (MMul (MPow a b) (MPow a c)))],
        |ctx, values| {
            let [a, b, c, mul] = values else {
                unreachable!()
            };
            let bc = m_add(ctx, *b, *c);
            let rhs = m_pow(ctx, *a, bc);
            ctx.union(*mul, rhs);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "pow_one",
        &math_vars(&math, &["x", "pow"]),
        facts![(= pow (MPow x (MConst 1)))],
        |ctx, values| {
            let [x, pow] = values else { unreachable!() };
            ctx.union(*pow, *x);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "pow_two",
        &math_vars(&math, &["x", "pow"]),
        facts![(= pow (MPow x (MConst 2)))],
        |ctx, values| {
            let [x, pow] = values else { unreachable!() };
            let rhs = m_mul(ctx, *x, *x);
            ctx.union(*pow, rhs);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "diff_add",
        &math_vars(&math, &["x", "a", "b", "diff"]),
        facts![(= diff (MDiff x (MAdd a b)))],
        |ctx, values| {
            let [x, a, b, diff] = values else {
                unreachable!()
            };
            let da = m_diff(ctx, *x, *a);
            let db = m_diff(ctx, *x, *b);
            let rhs = m_add(ctx, da, db);
            ctx.union(*diff, rhs);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "diff_mul",
        &math_vars(&math, &["x", "a", "b", "diff"]),
        facts![(= diff (MDiff x (MMul a b)))],
        |ctx, values| {
            let [x, a, b, diff] = values else {
                unreachable!()
            };
            let db = m_diff(ctx, *x, *b);
            let da = m_diff(ctx, *x, *a);
            let a_db = m_mul(ctx, *a, db);
            let b_da = m_mul(ctx, *b, da);
            let rhs = m_add(ctx, a_db, b_da);
            ctx.union(*diff, rhs);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "diff_sin",
        &math_vars(&math, &["x", "diff"]),
        facts![(= diff (MDiff x (MSin x)))],
        |ctx, values| {
            let [x, diff] = values else { unreachable!() };
            let rhs = m_cos(ctx, *x);
            ctx.union(*diff, rhs);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "diff_cos",
        &math_vars(&math, &["x", "diff"]),
        facts![(= diff (MDiff x (MCos x)))],
        |ctx, values| {
            let [x, diff] = values else { unreachable!() };
            let neg1 = m_const(ctx, -1);
            let sin = m_sin(ctx, *x);
            let rhs = m_mul(ctx, neg1, sin);
            ctx.union(*diff, rhs);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "int_one",
        &math_vars(&math, &["x", "integ"]),
        facts![(= integ (MIntegral (MConst 1) x))],
        |ctx, values| {
            let [x, integ] = values else { unreachable!() };
            ctx.union(*integ, *x);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "int_cos",
        &math_vars(&math, &["x", "integ"]),
        facts![(= integ (MIntegral (MCos x) x))],
        |ctx, values| {
            let [x, integ] = values else { unreachable!() };
            let rhs = m_sin(ctx, *x);
            ctx.union(*integ, rhs);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "int_sin",
        &math_vars(&math, &["x", "integ"]),
        facts![(= integ (MIntegral (MSin x) x))],
        |ctx, values| {
            let [x, integ] = values else { unreachable!() };
            let neg1 = m_const(ctx, -1);
            let cos = m_cos(ctx, *x);
            let rhs = m_mul(ctx, neg1, cos);
            ctx.union(*integ, rhs);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "int_add",
        &math_vars(&math, &["f", "g", "x", "integ"]),
        facts![(= integ (MIntegral (MAdd f g) x))],
        |ctx, values| {
            let [f, g, x, integ] = values else {
                unreachable!()
            };
            let i_f = m_integral(ctx, *f, *x);
            let i_g = m_integral(ctx, *g, *x);
            let rhs = m_add(ctx, i_f, i_g);
            ctx.union(*integ, rhs);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "int_sub",
        &math_vars(&math, &["f", "g", "x", "integ"]),
        facts![(= integ (MIntegral (MSub f g) x))],
        |ctx, values| {
            let [f, g, x, integ] = values else {
                unreachable!()
            };
            let i_f = m_integral(ctx, *f, *x);
            let i_g = m_integral(ctx, *g, *x);
            let rhs = m_sub(ctx, i_f, i_g);
            ctx.union(*integ, rhs);
            Some(())
        },
    );
    add_math_rule(
        &mut egraph,
        ruleset,
        "int_mul",
        &math_vars(&math, &["a", "b", "x", "integ"]),
        facts![(= integ (MIntegral (MMul a b) x))],
        |ctx, values| {
            let [a, b, x, integ] = values else {
                unreachable!()
            };
            let i_b = m_integral(ctx, *b, *x);
            let a_i_b = m_mul(ctx, *a, i_b);
            let dxa = m_diff(ctx, *x, *a);
            let mul = m_mul(ctx, dxa, i_b);
            let i2 = m_integral(ctx, mul, *x);
            let rhs = m_sub(ctx, a_i_b, i2);
            ctx.union(*integ, rhs);
            Some(())
        },
    );

    MathMicroBenchInput {
        egraph,
        ruleset: ruleset.to_owned(),
    }
}

pub fn run_math_microbenchmark_iters(input: &mut MathMicroBenchInput, iters: usize) {
    for _ in 0..iters {
        run_ruleset(&mut input.egraph, &input.ruleset).unwrap();
    }
    input.egraph.serialize(egglog::SerializeConfig::default());
}

#[allow(dead_code)]
pub fn run_math_microbenchmark(input: &mut MathMicroBenchInput) {
    run_math_microbenchmark_iters(input, 11);
}
