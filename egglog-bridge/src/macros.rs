use hashbrown::HashMap;

use crate::{rule::Variable, ColumnTy, FunctionId, QueryEntry, RuleBuilder};

#[macro_export]
#[doc(hidden)]
macro_rules! parse_rhs_atom_args {
    ($ebuilder:expr, $builder:expr, $table:ident, $v:expr, []) => { };
    ($ebuilder:expr, $builder:expr, $table:ident, $v:expr, [{ $e:expr } $($args:tt)*]) => {
        $v.push($e);
        $crate::parse_rhs_atom_args!($ebuilder, $builder, $table, $v, [$($args)*]);
    };
    ($ebuilder:expr, $builder:expr, $table:ident, $v:expr, [$var:ident $($args:tt)*]) => {
        let v = $ebuilder.lookup_var(stringify!($var)).unwrap_or_else(|| {
            panic!("use of unbound variable {} on the right-hand side of a rule", stringify!($var))
        });
        $v.push(v);
        $crate::parse_rhs_atom_args!($ebuilder, $builder, $table, $v, [$($args)*]);
    };
    ($ebuilder:expr, $builder:expr, $table:ident, $v:expr, [($func:tt $($fargs:tt)*) $($args:tt)*]) => {
        let ret = $crate::parse_rhs_atom!($ebuilder, $builder, ($func $($fargs)*));
        $v.push(ret.into());
        $crate::parse_rhs_atom_args!($ebuilder, $builder, $table, $v, [$($args)*]);
    }
}

#[macro_export]
#[doc(hidden)]
macro_rules! parse_rhs_atom {
    ($ebuilder:expr, $builder:expr, $var:ident) => {{
        $ebuilder.lookup_var(stringify!($var)).unwrap_or_else(|| {
            panic!("use of unbound variable {} on the right-hand side of a rule", stringify!($var))
        })
    }};
    ($ebuilder:expr, $builder:expr, ($func:tt $($args:tt)*)) => {{
        #[allow(clippy::vec_init_then_push)]
        {
            let mut vec = Vec::<$crate::QueryEntry>::new();
            $crate::parse_rhs_atom_args!($ebuilder, $builder, $func, vec, [$($args)*]);
            $builder.lookup($func.into(), &vec)
        }
    }};
}

#[macro_export]
#[doc(hidden)]
macro_rules! parse_rhs_command {
    ($ebuilder:expr, $builder:expr, []) => { };
    ($ebuilder:expr, $builder:expr, [(let $i:ident $($expr:tt)*) $($rest:tt)*]) => {
        let res = $crate::parse_rhs_atom!($ebuilder, $builder, $($expr)*);
        $ebuilder.bind_var(stringify!($i), res);
        $crate::parse_rhs_command!($ebuilder, $builder, [$($rest)*]);
    };
    ($ebuilder:expr, $builder:expr, [(set ($func:tt $($args:tt)*) $res:tt) $($rest:tt)*]) => {
        let mut vec = Vec::<$crate::QueryEntry>::new();
        $crate::parse_rhs_atom_args!($ebuilder, $builder, $func, vec, [$($args)*]);
        $crate::parse_rhs_atom_args!($ebuilder, $builder, $func, vec, [$res]);
        $builder.set($func.into(), &vec);
        $crate::parse_rhs_command!($ebuilder, $builder, [$($rest)*]);
    };
    ($ebuilder:expr, $builder:expr, [(union $l:tt $r:tt) $($rest:tt)*]) => {
        let lqe = $crate::parse_rhs_atom!($ebuilder, $builder, $l);
        let rqe = $crate::parse_rhs_atom!($ebuilder, $builder, $r);
        $builder.union(lqe.into(), rqe.into());
        $crate::parse_rhs_command!($ebuilder, $builder, [$($rest)*]);
    };
}

// left-hand side parsing

#[macro_export]
#[doc(hidden)]
macro_rules! parse_lhs_atom_args {
    ($ebuilder:expr, $builder:expr, $table:ident, $v:expr, []) => {};
    ($ebuilder:expr, $builder:expr, $table:ident, $v:expr, [{ $e:expr } $( $args:tt)*]) => {
        $v.push($e);
        $crate::parse_lhs_atom_args!($ebuilder, $builder, $table, $v, [$($args),*]);
    };
    ($ebuilder:expr, $builder:expr, $table:ident, $v:expr, [$var:ident $( $args:tt)*]) => {
        let ret = $ebuilder.get_var(stringify!($var), $table, $v.len(), &mut $builder);
        $v.push(ret.into());
        $crate::parse_lhs_atom_args!($ebuilder, $builder, $table, $v, [$($args),*]);
    };
    ($ebuilder:expr, $builder:expr, $table:ident, $v:expr, [($func:tt $($fargs:tt)*) $($args:tt)*]) => {
        let ret = $crate::parse_lhs_atom!($ebuilder, $builder, ($func $($fargs)*));
        $v.push(ret.into());
        $crate::parse_lhs_atom_args!($ebuilder, $builder, $table, $v, [$($args),*]);
    };
}
#[macro_export]
#[doc(hidden)]
macro_rules! parse_lhs_atom {
    ($ebuilder:expr, $builder:expr; $inferred_ty:expr, $var:ident) => {
        $ebuilder.bind_or_lookup_var(stringify!($var), $inferred_ty, &mut $builder)
    };
    ($ebuilder:expr, $builder:expr; $inferred_ty:expr, ($func:tt $($args:tt)*)) => {
        parse_lhs_atom!($ebuilder, $builder, ($func $($args)*))
    };
    ($ebuilder:expr, $builder:expr, ($func:tt $($args:tt)*)) => {{
        let mut vec = Vec::<$crate::QueryEntry>::new();
        $crate::parse_lhs_atom_args!($ebuilder, $builder, $func, vec, [$($args)*]);
        // Now return the last argument
        let ty = $ebuilder.infer_type($func.into(), vec.len(), &$builder);
        let res = $builder.new_var_named(ty, stringify!($func ($($args)*)));
        vec.push(res.clone());
        $builder.query_table($func.into(), &vec, Some(false)).unwrap();
        res
    }};
}

#[macro_export]
#[doc(hidden)]
macro_rules! parse_lhs_atom_with_ret {
    ($ebuilder:expr, $builder:expr, $ret:expr, ($func:tt $($args:tt)*)) => {{
        #[allow(clippy::vec_init_then_push)]
        {
            let mut vec = Vec::<$crate::QueryEntry>::new();
            $crate::parse_lhs_atom_args!($ebuilder, $builder, $func, vec, [$($args)*]);
            vec.push($ret.into());
            $builder.query_table($func.into(), &vec, Some(false)).unwrap();
        }
    }};
}

#[macro_export]
#[doc(hidden)]
macro_rules! parse_lhs {
    ($ebuilder:expr, $builder:expr, [$((-> ($func:tt $($args:tt)*) $ret:tt))*]) => {
        $(
            // First, parse the return value, getting a variable out:
            let ty = $ebuilder.infer_return_type($func.into(), &$builder);
            let ret_var = $crate::parse_lhs_atom!($ebuilder, $builder; ty, $ret);
            $crate::parse_lhs_atom_with_ret!($ebuilder, $builder, ret_var, ($func $($args)*));
        )*
    };
}

#[macro_export]
macro_rules! define_rule {
    ([$egraph:expr] ($($lhs:tt)*) => ($($rhs:tt)*))  => {{
        let mut ebuilder = $crate::macros::ExpressionBuilder::default();
        let mut builder = $egraph.new_rule(stringify!(($($lhs)* => $($rhs)*)), true);
        $crate::parse_lhs!(ebuilder, builder, [ $($lhs)* ]);
        $crate::parse_rhs_command!(ebuilder, builder, [ $($rhs)* ]);
        builder.build()
    }};
}

#[macro_export]
#[doc(hidden)]
macro_rules! add_expression_impl {
    ([ $egraph:expr ] $x:ident) => { $x };
    ([ $egraph:expr ] ($func:tt $($args:tt)*)) => {{
        let inner = [
            $($crate::add_expression_impl!([ $egraph ] $args),)*
        ];
        $egraph.add_term($func, &inner, stringify!($func ($($args)*)))
    }};
}

#[macro_export]
macro_rules! add_expressions {
    ([ $egraph:expr ] $($expr:tt)*) => {{
        $( $crate::add_expression_impl!([ $egraph ] $expr);)*
    }};
}

/// A struct used specifically to make macro invocations easier to parse. Prefer
/// using [`RuleBuilder`] for constructing rules directly.
///
/// [`RuleBuilder`]: crate::RuleBuilder
#[doc(hidden)]
#[derive(Default)]
pub struct ExpressionBuilder {
    vars: HashMap<&'static str, QueryEntry>,
}

impl ExpressionBuilder {
    pub fn bind_var(&mut self, name: &'static str, var: Variable) {
        if self.vars.contains_key(name) {
            return;
        }
        self.vars.insert(
            name,
            QueryEntry::Var {
                id: var,
                name: Some(name.into()),
            },
        );
    }
    pub fn bind_or_lookup_var(
        &mut self,
        name: &'static str,
        ty: ColumnTy,
        builder: &mut RuleBuilder,
    ) -> QueryEntry {
        if let Some(var) = self.vars.get(name) {
            return var.clone();
        }
        let var = builder.new_var_named(ty, name);
        self.vars.insert(name, var.clone());
        var
    }
    pub fn lookup_var(&mut self, name: &'static str) -> Option<QueryEntry> {
        self.vars.get(name).cloned()
    }

    pub fn get_var(
        &mut self,
        name: &'static str,
        func: FunctionId,
        col: usize,
        rb: &mut RuleBuilder,
    ) -> QueryEntry {
        if let Some(var) = self.vars.get(name) {
            return var.clone();
        }
        let ty = self.infer_type(func, col, rb);
        let var = rb.new_var_named(ty, name);
        self.vars.insert(name, var.clone());
        var
    }

    pub fn infer_return_type(&self, func: FunctionId, rb: &RuleBuilder) -> ColumnTy {
        rb.egraph().funcs[func].ret_ty()
    }

    pub fn infer_type(&self, func: FunctionId, col: usize, rb: &RuleBuilder) -> ColumnTy {
        rb.egraph().funcs[func].schema[col]
    }
}
