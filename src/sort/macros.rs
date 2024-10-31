macro_rules! add_primitives {
    ($type_info:expr,
        $name:literal = |$($param:ident : $param_t:ty),*| -> $ret:ty { $body:expr }
    ) => {{
        let type_info: &mut _ = $type_info;
        #[allow(unused_imports, non_snake_case)]
        {
            use $crate::{*, sort::*, constraint::*};

            struct MyPrim {$(
                $param: Arc<<$param_t as FromSort>::Sort>,
            )*
                __out: Arc<<$ret as IntoSort>::Sort>,
            }

            impl $crate::PrimitiveLike for MyPrim {
                fn name(&self) -> String {
                    $name.into()
                }

                fn get_type_constraints(
                    &self,
                    span: &Span
                ) -> Box<dyn TypeConstraint> {
                    let sorts = vec![$(self.$param.clone(),)* self.__out.clone() as ArcSort];
                    SimpleTypeConstraint::new(self.name(), sorts, span.clone()).into_box()
                }

                fn apply(
                    &self,
                    values: &[Value],
                    _sorts: (&[ArcSort], &ArcSort),
                    _egraph: Option<&mut EGraph>,
                ) -> Option<Value> {
                    if let [$($param),*] = values {
                        $(let $param: $param_t = <$param_t as FromSort>::load(&self.$param, $param);)*
                        // print!("{}( ", $name);
                        // $( print!("{}={:?}, ", stringify!($param), $param); )*
                        let result: $ret = $body;
                        // println!(") = {result:?}");
                        result.store(&self.__out)
                    } else {
                        panic!("wrong number of arguments")
                    }
                }
            }
            type_info.add_primitive($crate::Primitive::from(MyPrim {
                $( $param: type_info.get_sort_nofail::<<$param_t as IntoSort>::Sort>(), )*
                __out: type_info.get_sort_nofail::<<$ret as IntoSort>::Sort>(),
            }))
        }
    }};
}
