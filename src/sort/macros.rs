#[macro_export]
macro_rules! to_tt {
    ($tt:tt, $callback:ident) => {
        $callback!($tt)
    };
}

#[macro_export]
macro_rules! unpack {
    (& $t:ident) => {
        $t
    };
    ($t:ident) => {
        $t
    };
}

#[macro_export]
macro_rules! add_primitives {
    // ($egraph:expr, $($rest:tt)*) => {
    //      add_primitives!(@doubled $egraph, $($rest)*, $($rest)*)
    // };
    ($type_info:expr,
        $name:literal = |$($param:ident : $param_t:ty),*| -> $ret:ty { $body:expr }
        // $name2:literal = |$($param2:ident : $(&)? $base_param_t:ident),*| -> $ret2:ty { $body2:expr }
    ) => {{
        let type_info: &mut _ = $type_info;
        #[allow(unused_imports, non_snake_case)]
        {
            use $crate::{*, sort::*};

            struct MyPrim {$(
                $param: Arc<<$param_t as FromSort>::Sort>,
            )*
                __out: Arc<<$ret as IntoSort>::Sort>,
            }

            impl $crate::PrimitiveLike for MyPrim {
                fn name(&self) -> $crate::Symbol {
                    $name.into()
                }

                fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
                    let mut types = types.iter();
                    $(
                        if self.$param.name() != types.next()?.name() {
                            return None;
                        }
                    )*
                    if types.next().is_some() {
                        None
                    } else {
                        Some(self.__out.clone())
                    }
                }

                fn apply(&self, values: &[Value]) -> Option<Value> {
                    if let [$($param),*] = values {
                        $(let $param: $param_t = <$param_t as FromSort>::load(&self.$param, $param);)*
                        //print!("{}( ", $name);
                        //$( print!("{}:{}={:?}, ", stringify!($param), self.$param.name(), $param); )*
                        use std::io::*;
                        //std::io::stdout().flush();
                        let result: $ret = $body;
                        //println!(") = {result:?}");
                        result.store(&self.__out)
                    } else {
                        panic!()
                    }
                }
            }
            type_info.add_primitive($crate::Primitive::from(MyPrim {
                $( $param: type_info.get_sort::<<$param_t as IntoSort>::Sort>(), )*
                __out: type_info.get_sort::<<$ret as IntoSort>::Sort>(),
            }))
        }
    }};
}
