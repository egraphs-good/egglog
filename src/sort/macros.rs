/// This macro lets the user declare custom egglog primitives.
/// It supports a few special features:
///
/// - Failure: using `-?>` instead of `->` before the return type will
///   make the primitive fallible, requiring the body to return an
///   `Option<T>` instead of just `T`.
///
/// - Varargs: using `[xs: T]` as the argument list will cause the
///   primitive to take a variable number of arguments, which will be
///   passed to be body as an `Iterator<Item=T>`.
///
/// - Polymorphism: using `#` as a type will allow any type.
///   The value will be passed to the argument as an `egglog::Value`.
///   NOTE: currently, if one argument is polymorphic, all other
///   arguments must also be.
///
#[macro_export]
macro_rules! add_primitive {
    // This is a complicated macro. The first annoying thing is that
    // `macro_rules` macros have no `|` operator, so you have to break out
    // all of your choices into separate arms. To make the code shorter,
    // we break up the initial parsing into 4 sections: start, arguments,
    // arrow, and return type. To avoid exporting a million separate macros,
    // we use `@IDENT` to prefix internal helper branches. The point of parsing
    // separately is to put all of the possible inputs into a common format,
    // while tracking details like fixed vs. variable arity.

    // TODO: support all combinations of arguments (involves
    // adding/extending type constraints)

    // -------- START OF PARSING -------- //
    ($eg:expr, $name:literal = $($tail:tt)*) => {
        add_primitive!(@1 $eg, $name $($tail)*)
    };
    // -------- parse the arguments -------- //
    (@1 $eg:expr, $name:literal |$($x:ident : $t:ty),*| $($tail:tt)*) => {
        add_primitive!(@2 $eg, $name fixarg [$($x : $t,)*] $($tail)*)
    };
    (@1 $eg:expr, $name:literal |$($x:ident : #),*| $($tail:tt)*) => {
        add_primitive!(@2 $eg, $name fixarg [$($x : #,)*] $($tail)*)
    };
    (@1 $eg:expr, $name:literal [$x:ident : $t:ty] $($tail:tt)*) => {
        add_primitive!(@2 $eg, $name vararg [$x : $t,] $($tail)*)
    };
    (@1 $eg:expr, $name:literal [$x:ident : #] $($tail:tt)*) => {
        add_primitive!(@2 $eg, $name vararg [$x : #,] $($tail)*)
    };
    // -------- parse the arrow -------- //
    (@2 $eg:expr, $name:literal $v:ident [$($xs:tt)*] -> $($tail:tt)*) => {
        add_primitive!(@3 $eg, $name $v pure [$($xs)*] $($tail)*)
    };
    (@2 $eg:expr, $name:literal $v:ident [$($xs:tt)*] -?> $($tail:tt)*) => {
        add_primitive!(@3 $eg, $name $v fail [$($xs)*] $($tail)*)
    };
    // -------- parse the return type -------- //
    (@3 $eg:expr, $name:literal $v:ident $f:ident [$($xs:tt)*] $y:ty { $body:expr }) => {
        add_primitive!(@main $eg, $name $v $f [$($xs)*] [__y : $y,] $body)
    };
    (@3 $eg:expr, $name:literal $v:ident $f:ident [$($xs:tt)*] # { $body:expr }) => {
        add_primitive!(@main $eg, $name $v $f [$($xs)*] [__y : #,] $body)
    };
    // -------- END OF PARSING -------- //

    // This is the main body of this macro. It implements a primitive by
    // expanding into a struct definition, an implementation of `PrimitiveLike`
    // for that struct, and a statement to register the struct as a primitive.
    // We cache the egglog types that we need to generate the type constraints
    // inside the struct's fields.
    // Schema:
    // - $eg: mutable reference to the egraph
    // - $name: the name of this primitive as a string
    // - $v: arity of the primitive: either `fixarg` or `vararg`
    // - $f: fallibility of the primitive: either `pure` or `fail`
    // - $xs: the arguments to the primitive, as an array (+ trailing comma)
    //   - if $v is `vararg`, there will only be one element
    //   - if any argument is polymorphic, the type will be `#`
    //   - importantly, `#`s don't match the `t:ty` fragment specifier!
    // - $y: the return type of the primitive, in the same format as $xs
    //   - matching formats makes it easier to generate the struct fields
    //   - we make up `__y` as a struct field name that avoids collisions
    // - $body: the code to insert into the body of the primitive
    (@main $eg:expr, $name:literal $v:ident $f:ident [$($xs:tt)*] [$($y:tt)*] $body:expr) => {{
        #[allow(unused_imports)]
        use $crate::{*, constraint::*};
        #[allow(unused_imports)]
        use ::std::sync::Arc;
        use core_relations::{ExecutionState, ExternalFunction, Value as V};

        // Here we both assert the type of the reference and ensure
        // that $eg is only evaluated once. This requires binding a
        // new identifier, which changes the type of $eg in `@prim_use`.
        let eg: &mut EGraph = $eg;

        add_primitive!{@prim_def Prim [$($xs)* $($y)*] -> []}

        impl PrimitiveLike for Prim {
            fn name(&self) -> Symbol {
                $name.into()
            }

            fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
                add_primitive!(@types $v self span [$($xs)*] [$($y)*])
            }

            fn apply(&self, args: &[Value], _: (&[ArcSort], &ArcSort), _: Option<&mut EGraph>) -> Option<Value> {
                add_primitive!(@apply $v self args [$($xs)*] [$($y)*] $f $body)
            }
        }

        #[derive(Clone)]
        struct Ext;

        impl ExternalFunction for Ext {
            fn invoke(&self, exec_state: &mut ExecutionState, args: &[V]) -> Option<V> {
                let _prims = exec_state.prims();
                let _args = args;
                todo!()
            }
        }

        let prim = add_primitive!{@prim_use eg Prim [$($xs)* $($y)*] -> []};
        let ext = eg.backend.register_external_func(Ext);
        eg.add_primitive(Primitive(Arc::new(prim), ext))
    }};

    // -------- Body of get_type_constraints() -------- //
    (@types fixarg $self:ident $span:ident [$($x:ident : $xt:ty,)*] [$y:ident : $yt:ty,]) => {{
        let sorts = vec![$($self.$x.clone() as ArcSort,)* $self.$y.clone()];
        SimpleTypeConstraint::new($self.name(), sorts, $span.clone()).into_box()
    }};
    (@types fixarg $self:ident $span:ident [$($x:ident : #,)*] [$y:ident : $yt:ty,]) => {{
        AllEqualTypeConstraint::new($self.name(), $span.clone())
            .with_exact_length(add_primitive!(@len [$($x)*]) + 1)
            .with_output_sort($self.$y.clone())
            .into_box()
    }};
    (@types fixarg $self:ident $span:ident [$($x:ident : #,)*] [$y:ident : #,]) => {{
        AllEqualTypeConstraint::new($self.name(), $span.clone())
            .with_exact_length(add_primitive!(@len [$($x)*]) + 1)
            .into_box()
    }};
    (@types vararg $self:ident $span:ident [$x:ident : $xt:ty,] [$y:ident : $yt:ty,]) => {{
        AllEqualTypeConstraint::new($self.name(), $span.clone())
            .with_all_arguments_sort($self.$x.clone())
            .with_output_sort($self.$y.clone())
            .into_box()
    }};

    // -------- Body of apply() -------- //
    (@apply $v:ident $self:ident $args:ident [$($xs:tt)*] [$($y:tt)*] $f:ident $body:expr) => {{
        add_primitive!(@args $v $self $args [$($xs)*]);
        add_primitive!(@body $f $self [$($y)*] $body)
    }};

    // -------- Destruct apply() args -------- //
    (@args vararg $self:ident $args:ident [$x:ident : #,]) => {
        let $x = $args.iter();
    };
    (@args fixarg $self:ident $args:ident [$($x:ident : #,)*]) => {
        let [$($x,)*] = $args else { panic!("wrong number of arguments") };
    };
    (@args vararg $self:ident $args:ident [$x:ident : $t:ty,]) => {
        add_primitive!(@args vararg $self $args [$x : #,]);
        let $x = $x.map(|x| <$t as FromSort>::load(&$self.$x, x));
    };
    (@args fixarg $self:ident $args:ident [$($x:ident : $t:ty,)*]) => {
        add_primitive!(@args fixarg $self $args [$($x : #,)*]);
        $(let $x: $t = <$t as FromSort>::load(&$self.$x, $x);)*
    };

    // -------- Run apply() body and return -------- //
    (@body pure $self:ident [$y:ident : #,] $body:expr) => {{
        Some($body)
    }};
    (@body fail $self:ident [$y:ident : #,] $body:expr) => {{
        $body
    }};
    (@body pure $self:ident [$y:ident : $t:ty,] $body:expr) => {{
        let y: $t = $body;
        Some(y.store(&$self.$y))
    }};
    (@body fail $self:ident [$y:ident : $t:ty,] $body:expr) => {{
        let y: $t = $body?;
        Some(y.store(&$self.$y))
    }};

    // -------- Get the length of a list -------- //
    // no this is really the best way:
    // veykril.github.io/tlborm/decl-macros/building-blocks/counting.html
    (@len [$($x:ident)*]) => { <[()]>::len(&[$(add_primitive!(@sub $x)),*]) };
    (@sub $x:ident) => { () };

    // The below helper macros match on `(@helper ... [a*] -> [b*])`,
    // where `a` is the work to be performed, and `b` is the work to be
    // completed. This is necessary because macros can't output any
    // "partial Rust" (like just one struct field, for example).

    // -------- Generate struct definition -------- //
    (@prim_def $name:ident [$x:ident : #    , $($xs:tt)*] -> [$($f:tt)*]) => {
        add_primitive!(@prim_def $name [$($xs)*] -> [$($f)*])
    };
    (@prim_def $name:ident [$x:ident : $t:ty, $($xs:tt)*] -> [$($f:tt)*]) => {
        add_primitive!(@prim_def $name [$($xs)*] -> [$($f)*
            $x: Arc<<$t as IntoSort>::Sort>,])
    };
    (@prim_def $name:ident [] -> [$($fields:tt)*]) => {
        struct $name {
            $($fields)*
        }
    };

    // -------- Generate struct construction -------- //
    (@prim_use $eg:ident $name:ident [$x:ident : #    , $($xs:tt)*] -> [$($f:tt)*]) => {
        add_primitive!(@prim_use $eg $name [$($xs)*] -> [$($f)*])
    };
    (@prim_use $eg:ident $name:ident [$x:ident : $t:ty, $($xs:tt)*] -> [$($f:tt)*]) => {
        add_primitive!(@prim_use $eg $name [$($xs)*] -> [$($f)*
            $x: $eg.type_info
                   .get_sort_by::<<$t as IntoSort>::Sort>(|_| true)
                   .unwrap_or_else(|| panic!("Failed to lookup sort: {}", std::any::type_name::<$t>())),
        ])
    };
    (@prim_use $eg:ident $name:ident [] -> [$($fields:tt)*]) => {
        $name {
            $($fields)*
        }
    };
}
