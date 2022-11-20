use crate::ast::Literal;

use super::*;

use std::mem;
use std::sync::Mutex;
use z3;

/*

Z3Bool vs (Z3 bool) as sort. meh. Does it matter much?

I can specialize Z3Sort to Z3Bool, etc so long as I use the Arc<Context>
That is nice.

*/

// No. Z3Ast implements hash.
// So I can use a regular indexset.

/*
There are a couple choices:
1. Z3Sort<'ctx>. This was not working. There is something wrong about 'ctx which comes from the held Context anyhow.
2. Rebuild the high level z3 bindings from z3_sys to allow for hashmap storage of asts. This will largely be highly replaicetd from what the z3 crate already does
3. Use 'static to remove the lifetime parameter from z3::ast::Dynamic. mem::trasnmute will be needed at certain spots,
 because the context is not static
4. Actually have a static global Z3 context avaiable.
5. Don't use bindings at all. Construct ast/s-expressions and call smt solver via string interface


*/
#[derive(Debug)]
pub struct Z3Sort {
    name: Symbol,
    ctx: Arc<z3::Context>,
    stringsort: StringSort,
    asts: Mutex<IndexSet<z3::ast::Dynamic<'static>>>,
}

unsafe impl Send for Z3Sort {}
unsafe impl Sync for Z3Sort {}

impl Z3Sort {
    pub fn new(name: Symbol, stringsort: StringSort) -> Self {
        let cfg = z3::Config::new();
        Self {
            name,
            ctx: Arc::new(z3::Context::new(&cfg)),
            stringsort,
            asts: Default::default(),
        }
    }
}

struct Z3True {
    name: Symbol,
    sort: Arc<Z3Sort>,
}

impl PrimitiveLike for Z3True {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        //Some(self.sort.clone())
        match types {
            [] => Some(self.sort.clone()),
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        assert!(values.is_empty());
        //ValueMap::default().store(&self.map)
        let sort = self.sort.clone();
        let ctx: &z3::Context = &sort.ctx;
        let ctx: &'static z3::Context = unsafe { mem::transmute(ctx) };
        let d = z3::ast::Dynamic::from(z3::ast::Bool::from_bool(ctx, true));
        d.store(&sort)
    }
}

//example constant
struct Z3False {
    name: Symbol,
    sort: Arc<Z3Sort>,
}

impl PrimitiveLike for Z3False {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        //Some(self.sort.clone())
        match types {
            [] => Some(self.sort.clone()),
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        assert!(values.is_empty());
        //ValueMap::default().store(&self.map)
        let sort = self.sort.clone();
        let ctx: &z3::Context = &sort.ctx;
        let d = z3::ast::Dynamic::from(z3::ast::Bool::from_bool(
            unsafe { mem::transmute(ctx) },
            false,
        ));
        d.store(&sort)
    }
}

// exmaple unop
struct Z3Not {
    name: Symbol,
    sort: Arc<Z3Sort>,
}

impl PrimitiveLike for Z3Not {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        //Some(self.sort.clone())
        match types {
            [t] => {
                if t.name() == "Z3Sort".into() {
                    Some(self.sort.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        match values {
            [x] => {
                //ValueMap::default().store(&self.map)
                let sort = &self.sort;
                let d = z3::ast::Dynamic::load(sort, x);
                let d: z3::ast::Dynamic = d.as_bool().unwrap().not().into();
                d.store(sort)
            }
            _ => panic!("Z3-not called on wrong number of arguments"),
        }
    }
}

// example binop
struct Z3And {
    name: Symbol,
    sort: Arc<Z3Sort>,
}

impl PrimitiveLike for Z3And {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        //Some(self.sort.clone())
        match types {
            [t, t2] => {
                if t.name() == "Z3Sort".into() && t2.name() == "Z3Sort".into() {
                    Some(self.sort.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        match values {
            [x, y] => {
                //ValueMap::default().store(&self.map)
                let sort = &self.sort;
                let x = z3::ast::Dynamic::load(sort, x).as_bool().unwrap();
                let y = z3::ast::Dynamic::load(sort, y).as_bool().unwrap();
                let d: z3::ast::Dynamic = (x & y).into();
                d.store(sort)
            }
            _ => panic!("Z3-and called on wrong number of arguments"),
        }
    }
}

struct Z3Check {
    name: Symbol,
    sort: Arc<Z3Sort>,
}

impl PrimitiveLike for Z3Check {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        //Some(self.sort.clone())
        match types {
            [t] => {
                if t.name() == "Z3Sort".into() {
                    Some(self.sort.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        match values {
            [x] => {
                //ValueMap::default().store(&self.map)
                let sort = &self.sort;
                let x = z3::ast::Dynamic::load(sort, x).as_bool().unwrap();
                let s = z3::Solver::new(&self.sort.ctx);
                s.assert(&x);
                let res = s.check();
                //
                /* */
                let res: &str = match res {
                    z3::SatResult::Sat => "sat",
                    z3::SatResult::Unsat => "unsat",
                    z3::SatResult::Unknown => "unknown",
                };
                let res: Symbol = res.into();
                res.store(&self.sort.stringsort)
                //res.to_string().store();
                //d.store(sort)
            }
            _ => panic!("Z3-and called on wrong number of arguments"),
        }
    }
}

impl Sort for Z3Sort {
    fn name(&self) -> Symbol {
        self.name
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    #[rustfmt::skip]
    fn register_primitives(self: Arc<Self>, egraph: &mut EGraph) {
        egraph.add_primitive(Z3True {
            name: "z3true".into(),
            sort: self.clone(),
        }); 
        egraph.add_primitive(Z3False{
            name: "z3false".into(),
            sort: self.clone(),
        }); 
        egraph.add_primitive(Z3Not{
            name: "not".into(),
            sort: self.clone(),
        }); 
        egraph.add_primitive(Z3And{
            name: "and".into(),
            sort: self.clone(),
        }); 
        egraph.add_primitive(Z3Check{
            name: "check-sat".into(),
            sort: self,
        }); 
        
    }

    fn make_expr(&self, value: Value) -> Expr {
        assert!(value.tag == self.name);
        let ast = z3::ast::Dynamic::load(self, &value);
        Expr::Lit(Literal::String(ast.to_string().into()))
    }
}

impl IntoSort for z3::ast::Dynamic<'static> {
    type Sort = Z3Sort;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        let (i, _) = sort.asts.lock().unwrap().insert_full(self);
        Some(Value {
            tag: sort.name,
            bits: i as u64,
        })
    }
}

impl FromSort for z3::ast::Dynamic<'static> {
    type Sort = Z3Sort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let i = value.bits as usize;
        (*sort.asts.lock().unwrap().get_index(i).unwrap()).clone()
    }
}
