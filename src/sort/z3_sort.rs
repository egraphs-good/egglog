use crate::ast::Literal;

use super::*;

use z3::{Config, Context};
use std::sync::Mutex;
use z3_sys;
use std::os::raw::c_uint;
use std::num::NonZeroU32;

#[derive(Debug)]
struct MyContext(z3_sys::Z3_context);

unsafe impl Send for MyContext {}
unsafe impl Sync for MyContext {}

#[derive(Debug)]
struct MyAst(z3_sys::Z3_ast);

unsafe impl Send for MyAst {}
unsafe impl Sync for MyAst {}

#[derive(Debug)]
pub struct Z3Bool {
    name: Symbol,
    ctx: Mutex<MyContext>,
    asts: Arc<Mutex<HashMap<c_uint, MyAst>>>,
 //   asts: Arc<Mutex<HashMap<c_uint,  Arc<Mutex<z3::ast::Bool<'ctx>>>>>>,
}

impl Z3Bool {
    pub fn new(name: Symbol) -> Self {
        let cfg = unsafe {z3_sys::Z3_mk_config()};
        Self { name,
            ctx : Mutex::new(MyContext(unsafe{z3_sys::Z3_mk_context(cfg)})),
            asts :  Default::default()
         }
    }
}

impl Sort for Z3Bool {
    fn name(&self) -> Symbol {
        self.name
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    #[rustfmt::skip]
    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        type Opt<T=()> = Option<T>;

      //  add_primitives!(eg, "true" = |a: ()| -> z3::ast::Bool<'ctx> { z3::ast::Bool::from_bool(self.ctx, true)});

     // add_primitives!(eg, "true" = |_a : ()| -> z3_sys::Z3_ast{ ||{ z3_sys::Z3_mk_true(self.lock().ctx)}});
    }

    fn make_expr(&self, value: Value) -> Expr {
        assert!(value.tag == self.name);
        let sym = Symbol::from(NonZeroU32::new(value.bits as _).unwrap());
        Expr::Lit(Literal::String(sym))
    }
}


impl IntoSort for z3_sys::Z3_ast {
    type Sort = Z3Bool;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        Some(Value {
            tag: sort.name,
            bits: unsafe {z3_sys::Z3_get_ast_id(&mut *sort.ctx.lock().unwrap().0, self)} as u64,
        })
    }
}

impl FromSort for z3_sys::Z3_ast {
    type Sort = Z3Bool;
    fn load(_sort: &Self::Sort, value: &Value) -> Self {
        value.bits as Self
    }
}

/*
impl<'a> IntoSort for z3::ast::Bool<'a> {
    type Sort = Z3Bool<'a>;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        Some(Value {
            tag: sort.name,
            bits: z3_sys::Z3_get_ast_id() as u64,
        })
    }
}

impl<'a> FromSort for z3::ast::Bool<'a> {
    type Sort = Z3Bool<'a>;
    fn load(_sort: &Self::Sort, value: &Value) -> Self {
        value.bits as Self
    }
}
*/

/*
impl<'a> IntoSort for z3::ast::Bool<'a> {
    type Sort = Z3Bool;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        Some(Value {
            tag: sort.name,
            bits: NonZeroU32::from(self.to_string()).get() as _,
        })
    }
}
*/
/*
impl<'a> FromSort for  z3::ast::Bool<'a>   {
    type Sort = Z3Bool;
    fn load(_sort: &Self::Sort, value: &Value) -> Self {
        NonZeroU32::new(value.bits as u32).unwrap().into()
    }
}
*/