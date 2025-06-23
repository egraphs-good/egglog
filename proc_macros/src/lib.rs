use core::panic;
use darling::{Error, FromMeta, ast::NestedMeta};

use heck::ToSnakeCase;
use helper::*;
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::{ToTokens, format_ident, quote};
use syn::{Data, DeriveInput, Type, parse_macro_input};
mod helper;

#[derive(Debug, FromMeta)]
struct SceneMeta {
    output: Ident,
}
#[proc_macro_attribute]
pub fn egglog_func(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let name = &input.ident;

    let attr_args = match NestedMeta::parse_meta_list(attr.into()) {
        Ok(v) => v,
        Err(e) => return TokenStream::from(Error::from(e).write_errors()),
    };
    let args = match SceneMeta::from_list(&attr_args) {
        Ok(v) => v,
        Err(e) => return TokenStream::from(e.write_errors()),
    };

    let output = args.output;
    let output_ty = format_ident!("{}Ty",output);
    let struct_def_expanded = match &input.data {
        Data::Struct(data_struct) => {
            let name_node = format_ident!("{}", name);
            // let derive_more_path  = derive_more_path();
            let egglog_path = egglog_path();
            let input_types = data_struct
                .fields
                .iter()
                .map(|field| &field.ty)
                .collect::<Vec<_>>();
            let _generic_decl = data_struct
                .fields
                .iter()
                .enumerate()
                .map(|(count, field)| {
                    let generic = format_ident!("T{}", count);
                    let ty = &field.ty;
                    quote!(#generic:AsRef<#ty>)
                })
                .collect::<Vec<_>>();

            let inventory_path = inventory_path();
            let _merge_option: proc_macro2::TokenStream = "no-merge".to_token_stream();
            let _merge_option: proc_macro2::TokenStream = "merge new".to_token_stream();
            quote! {
                #[allow(unused)]
                pub struct #name_node<T>{_p:std::marker::PhantomData<T>}
                const _:() = {
                    use #egglog_path::wrap::*;
                    impl<T:SingletonGetter> egglog::wrap::EgglogFunc for #name_node<T>{
                        type Output=#output<T,()>;
                        type Input=(#(#input_types<T,()>,)*);
                        type OutputTy=#output_ty;
                        const FUNC_NAME:&'static str = stringify!(#name_node);
                    }
                    impl<'a, T:TxSgl> #name_node<T> where T:TxSgl{
                        pub fn set(input: (#(&'a dyn AsRef<#input_types<T,()>>,)*), output: &dyn AsRef<#output<T,()>>){
                            T::on_func_set::<#name_node<T>>(input, output.as_ref());
                        }
                    }
                    impl<'a, R:RxSgl> #name_node<R> where R:RxSgl{
                        pub fn get(input: (#(&'a dyn AsRef<#input_types<R,()>>,)*)) -> #output<R,()>{
                            R::on_func_get::<#name_node<R>>(input).as_ref().clone()
                        }
                    }
                    #inventory_path::submit!{
                        Decl::EgglogFuncTy{
                            name: stringify!(#name_node),
                            input: &[ #(stringify!(#input_types)),*],
                            output: &(stringify!(#output))
                        }
                    }
                };
            }.into()
        }
        _ => {
            panic!()
        }
    };
    struct_def_expanded
}

/// generate `egglog` language from `rust native structure`   
///
/// # Example:  
///     
/// ```
/// #[allow(unused)]
/// #[derive(Debug, Clone, EgglogTy)]
/// enum Duration {
///     DurationBySecs {
///         seconds: f64,
///     },
///     DurationByMili {
///         milliseconds: f64,
///     },
/// }
/// ```
/// is transformed to
///
///
/// ```
/// #[derive(Debug, Clone, ::derive_more::Deref)]
/// pub struct DurationNode {
///     ty: _DurationNode,
///     #[deref]
///     sym: DurationSym,
/// }
///
/// fn to_egglog(&self) -> String {
///     match &self.ty {
///         _DurationNode::DurationBySecs { seconds } => {
///             format!("(let {} (DurationBySecs  {:.3}))", self.sym, seconds)
///         }
///         _DurationNode::DurationByMili { milliseconds } => {
///             format!("(let {} (DurationByMili  {:.3}))", self.sym, milliseconds)
///         }
///     }
/// }
/// impl crate::EgglogTy for Duration {
///     const SORT_DEF: crate::TySort =
///         crate::TySort(stringify!((Duration()(DurationByMili f64))));
/// }
/// ```
/// so that you can directly use to_egglog to generate let statement in eggglog
///
/// also there is a type def statement generated and specialized new function
///
///
#[proc_macro_attribute]
pub fn egglog_ty(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let name = &input.ident;

    let name_lowercase = format_ident!("{}", name.to_string().to_lowercase());
    let name_egglogty_impl = format_ident!("{}Ty", name);
    let egglog_path = egglog_path();
    let inventory_path = inventory_path();

    let type_def_expanded = match &input.data {
        Data::Enum(data_enum) => {
            let _variants_egglog_string = data_enum
                .variants
                .iter()
                .map(|variant| {
                    let tys = variant_to_tys(&variant);
                    let variant_name = &variant.ident;
                    quote! {  (#variant_name #(#tys )* )}
                })
                .collect::<Vec<_>>();
            let constructors = data_enum
                .variants
                .iter()
                .map(|variant| {
                    let tys = variant_to_tys(&variant);
                    let variant_name = &variant.ident;
                    let new_from_term_dyn_fn_name = format_ident!("new_{}_from_term_dyn",variant_name.to_string().to_snake_case());
                    quote! {  TyConstructor {  
                        cons_name: stringify!(#variant_name),
                        input:&[ #(stringify!(#tys)),* ] , 
                        output:stringify!(#name),
                        cost :None,
                        term_to_node: #name::<(),()>::#new_from_term_dyn_fn_name,
                        unextractable :false,
                    } }
                })
                .collect::<Vec<_>>();
            let expanded = quote! {
                use #egglog_path::wrap::*;
                impl EgglogTy for #name_egglogty_impl {
                    const TY_NAME:&'static str = stringify!(#name);
                    const TY_NAME_LOWER:&'static str = stringify!(#name_lowercase);
                }
                impl EgglogBaseTy for #name_egglogty_impl {
                    const CONSTRUCTORS : TyConstructors= TyConstructors(&[
                        #(#constructors),*
                    ]);
                }
                #inventory_path::submit!{
                    Decl::EgglogBaseTy { 
                        name: #name_egglogty_impl::TY_NAME,
                        cons: &#name_egglogty_impl::CONSTRUCTORS 
                    }
                }

            };
            expanded
        }
        Data::Struct(data_struct) => {
            // process (sort A (Vec M))  such things ..
            let f = data_struct
                .fields
                .iter()
                .nth(0)
                .expect("Struct should only have one Vec field");
            let first_generic = get_first_generic(&f.ty);
            let first_generic_ty = format_ident!("{}Ty",first_generic.to_token_stream().to_string());
            if is_vec_type(&f.ty) {
                let vec_expanded = quote! {
                    impl #egglog_path::wrap::EgglogTy for #name_egglogty_impl {
                        const TY_NAME:&'static str = stringify!(#name);
                        const TY_NAME_LOWER:&'static str = stringify!(#name_lowercase);
                    }
                    impl #egglog_path::wrap::EgglogContainerTy for #name_egglogty_impl {
                        type EleTy = #first_generic_ty;
                    } 
                    #inventory_path::submit!{
                        Decl::EgglogContainerTy { 
                            name: #name_egglogty_impl::TY_NAME,
                            ele_ty_name: <<#name_egglogty_impl as EgglogContainerTy>::EleTy as EgglogTy>::TY_NAME,
                            def_operator:"vec-of", 
                            term_to_node: #name::<(),()>::new_from_term_dyn
                        }
                    }
                };
                vec_expanded
            } else {
                panic!("only support Vec for struct")
            }
        }
        _ => panic!("only support enum"),
    };
    let struct_def_expanded = match &input.data {
        Data::Struct(data_struct) => {
            // process (sort A (Vec M))  such things ..
            let name_node_alias = format_ident!("{}NodeAlias", name);
            let name_node = format_ident!("{}", name);
            let name_inner = format_ident!("{}Inner", name);
            let name_counter = format_ident!("{}_COUNTER", name.to_string().to_uppercase());
            let f = data_struct
                .fields
                .iter()
                .nth(0)
                .expect("Struct should only have one Vec field");
            let field_name = &f.ident.as_ref().unwrap();
            let first_generic = get_first_generic(&f.ty);
            // let field_sym_ty = get_sym_type(first_generic);
            let (field_node, is_basic_ty) =
                match first_generic.to_token_stream().to_string().as_str() {
                    x if PANIC_TY_LIST.contains(&x) => {
                        panic!("{} not supported", x)
                    }
                    x if EGGLOG_BASIC_TY_LIST.contains(&x) => {
                        (first_generic.to_token_stream(), true)
                    }
                    _ => {
                        let first_generic_ident = match &first_generic {
                            syn::Type::Path(type_path) => {
                                type_path
                                    .path
                                    .segments
                                    .last()
                                    .expect("impossible")
                                    .clone()
                                    .ident
                            }
                            _ => panic!(
                                "{} type should be simple path",
                                first_generic.to_token_stream().to_string()
                            ),
                        };
                        let _first_generic = format_ident!("{}", first_generic_ident);
                        // postfix_type(&first_generic,"Node",Some("T")
                        (quote!(dyn AsRef<#_first_generic<T, ()>>), false)
                    }
                };
            let to_egglog_impl = if is_basic_ty {
                quote! {
                    impl<T:SingletonGetter, V:EgglogEnumVariantTy> ToEgglog for #name_node<T,V>
                    where #name_node<T,V>: EgglogNode{
                        fn to_egglog_string(&self) -> String{
                            format!("(let {} (vec-of {}))",self.node.sym,self.node.ty.v.iter_mut().fold("".to_owned(), |s,item| s+ item.as_str()+" " ))
                        }
                        fn to_egglog(&self) -> EgglogAction{
                            #egglog_path::ast::GenericAction::Let(span!(), self.cur_sym().to_string(), 
                                #egglog_path::ast::GenericExpr::Call(self.span.into(),"vec-of", self.node.ty.v.iter().map(|x| x.to_var()).collect()).to_owned_str()
                            )
                        }
                    }
                }
            } else {
                quote! {
                    impl<T:SingletonGetter, V:EgglogEnumVariantTy> ToEgglog for #name_node<T,V>
                    where #name_node<T,V>: EgglogNode{
                        fn to_egglog_string(&self) -> String{
                            format!("(let {} (vec-of {}))",self.cur_sym(),self.node.ty.v.iter().fold("".to_owned(), |s,item| s+ item.as_str()+" " ))
                        }
                        fn to_egglog(&self) -> EgglogAction{
                            #egglog_path::ast::GenericAction::Let(span!(), self.cur_sym().to_string(), 
                                #egglog_path::ast::GenericExpr::Call(self.span.into(), "vec-of", self.node.ty.v.iter().map(|x| x.to_var()).collect()).to_owned_str()
                            )
                        }
                    }
                    impl<T:TxSgl + VersionCtlSgl, V:EgglogEnumVariantTy> LocateVersion for #name_node<T,V>
                    where #name_node<T,V> : EgglogNode
                    {
                        fn locate_latest(&mut self){
                            T::set_latest(self.cur_sym_mut());
                            self.node.ty.v.iter_mut().for_each(|item| {T::set_latest(item.erase_mut())});
                        }
                        fn locate_next(&mut self){
                            T::set_next(self.cur_sym_mut());
                            self.node.ty.v.iter_mut().for_each(|item| {T::set_next(item.erase_mut())});
                        }
                        fn locate_prev(&mut self){
                            T::set_prev(self.cur_sym_mut());
                            self.node.ty.v.iter_mut().for_each(|item| {T::set_next(item.erase_mut())});
                        }
                    }
                }
            };
            let field_assignment  = if is_basic_ty {
                    quote! {
                        children.map(|x| match term_dag.get(x) {
                            Term::Lit(lit) => lit.clone().try_into().expect("literal type mismatch"),
                            Term::Var(v) => panic!(),
                            Term::App(app,v) => panic!(),
                        }).collect()
                    }
                } else {
                    quote! {
                        children.iter().map(|x| term2sym.get(x).unwrap().typed()).collect()
                    }
                } ;
            let field_ty = match first_generic.to_token_stream().to_string().as_str() {
                x if PANIC_TY_LIST.contains(&x) => {
                    panic!("{} not supported", x)
                }
                x if EGGLOG_BASIC_TY_LIST.contains(&x) => first_generic.to_token_stream(),
                _ => {
                    let first_generic = match &first_generic {
                        Type::Path(type_path) => {
                            type_path
                                .path
                                .segments
                                .last()
                                .expect("impossible")
                                .clone()
                                .ident
                        }
                        _ => panic!(
                            "{} keep the type simple!",
                            first_generic.to_token_stream().to_string()
                        ),
                    };
                    format_ident!("{}Ty", first_generic).to_token_stream()
                }
            };
            if is_vec_type(&f.ty) {
                let vec_expanded = quote! {
                    pub type #name_node_alias<T,V> = #egglog_path::wrap::Node<#name_egglogty_impl,T,#name_inner,V>;
                    #[allow(unused)]
                    #[derive(Clone,Debug)]
                    pub struct #name_egglogty_impl;
                    #[allow(unused)]
                    #[derive(::derive_more::DerefMut,::derive_more::Deref)]
                    pub struct #name_node<T: #egglog_path::wrap::SingletonGetter, V: #egglog_path::wrap::EgglogEnumVariantTy=()> {
                        node:#name_node_alias<T,V>
                    }
                    #[allow(unused)]
                    #[derive(Clone,Debug)]
                    pub struct #name_inner {
                        v:#egglog_path::wrap::Syms<#field_ty>
                    }
                    const _:() = {
                        use #egglog_path::wrap::*;
                        use #egglog_path::prelude::*;
                        use #egglog_path::*;
                        impl NodeInner<#name_egglogty_impl> for #name_inner{}
                        use std::marker::PhantomData;
                        static #name_counter: TyCounter<#name_egglogty_impl> = TyCounter::new();
                        impl<T:TxSgl> #name_node<T,()> {
                            #[track_caller]
                            pub fn new(#field_name:Vec<&#field_node>) -> #name_node<T,()>{
                                let #field_name = #field_name.into_iter().map(|r| r.as_ref().sym).collect();
                                use std::panic::Location;
                                let node = Node{ ty: #name_inner{v:#field_name}, span:Some(Location::caller()),sym: #name_counter.next_sym(),_p: PhantomData, _s: PhantomData};
                                let node = #name_node {node};
                                T::on_new(&node);
                                node
                            }
                            // /// with no side-effect (will not send command to EGraph or change node in WorkAreaGraph)
                            // #[track_caller]
                            // pub fn _new(#field_name:Vec<&#field_node>) -> #name_node<T,()>{
                            //     let ty = #name_inner{ v: #field_name.into() };
                            //     use std::panic::Location;
                            //     let node = Node { ty, sym: #name_counter.next_sym(),span:Some(Location::caller()), _p:PhantomData, _s:PhantomData::<()>};
                            //     let node = #name_node {node};
                            //     node
                            // }
                            #[track_caller]
                            pub fn new_from_term(term_id:TermId, term_dag: &TermDag, term2sym:&mut std::collections::HashMap<TermId, Sym>) -> #name_node<T,()>{
                                let children = match term_dag.get(term_id){
                                    Term::App(app,v) => v,
                                    _=> panic!()
                                };
                                let ty = #name_inner{v:#field_assignment };
                                use std::panic::Location;
                                let node = Node { ty, sym: #name_counter.next_sym(),span:Some(Location::caller()), _p:PhantomData, _s:PhantomData::<()>};
                                let node = #name_node {node};
                                term2sym.insert(term_id, node.cur_sym());
                                node
                            }
                            #[track_caller]
                            pub fn new_from_term_dyn(term_id:TermId, term_dag: &TermDag, term2sym:&mut std::collections::HashMap<TermId, Sym>) -> Box<dyn EgglogNode>{
                                Box::new(Self::new_from_term(term_id, term_dag, term2sym))
                            }
                        }
                        impl<T:SingletonGetter> EgglogNode for #name_node<T,()> {
                            fn succs_mut(&mut self) -> Vec<&mut Sym>{
                                self.node.ty.v.iter_mut().map(|s| s.erase_mut()).collect()
                            }
                            fn succs(&self) -> Vec<Sym>{
                                self.node.ty.v.iter().map(|s| s.erase()).collect()
                            }
                            fn roll_sym(&mut self) -> Sym{
                                let next_sym = #name_counter.next_sym();
                                self.node.sym = next_sym;
                                next_sym.erase()
                            }
                            fn cur_sym(&self) -> Sym{
                                self.node.sym.erase()
                            }
                            fn cur_sym_mut(&mut self) -> &mut Sym{
                                self.node.sym.erase_mut()
                            }
                            fn clone_dyn(&self) -> Box<dyn EgglogNode>{
                                Box::new(self.clone())
                            }
                        }
                        impl<T: SingletonGetter, V: EgglogEnumVariantTy> AsRef<#name_node<T, ()>> for #name_node<T, V> {
                            fn as_ref(&self) -> &#name_node<T, ()> {
                                unsafe {
                                    &*(self as *const #name_node<T,V> as *const #name_node<T,()>)
                                }
                            }
                        }
                        impl<T:SingletonGetter,V:EgglogEnumVariantTy > Clone for #name_node<T,V> {
                            fn clone(&self) -> Self {
                                Self { node: Node { ty: self.node.ty.clone(),span: self.span ,sym: self.node.sym.clone(), _p: PhantomData, _s: PhantomData }  }
                            }
                        }
                        #to_egglog_impl
                    };
                };
                vec_expanded
            } else {
                panic!("only support Vec for struct")
            }
        }
        Data::Enum(data_enum) => {
            let name_node_alias = format_ident!("{}NodeAlias", name);
            let name_node = format_ident!("{}", name);
            let _name_node = format_ident!("_{}", name);
            let name_inner = format_ident!("{}Inner", name);
            let name_counter = format_ident!("{}_COUNTER", name.to_string().to_uppercase());

            let variants_def_of_node_with_syms = data_enum
                .variants
                .iter()
                .map(|variant| {
                    let types_and_idents = variants_to_sym_typed_ident_list(variant);
                    let variant_name = &variant.ident;
                    quote! {#variant_name {#( #types_and_idents ),*  }}
                })
                .collect::<Vec<_>>();

            let to_egglog_string_match_arms = data_enum.variants.iter().map(|variant| {
                let variant_idents = variant_to_field_ident(variant);
                let variant_name = &variant.ident;
                let s = " {:.3}".repeat(variant_idents.len());
                let format_str = format!("(let {{}} ({} {}))", variant_name, s);
                quote! {#name_inner::#variant_name {#( #variant_idents ),*  } => {
                    format!(#format_str ,self.node.sym, #(#variant_idents),*)
                }}
            });
            let succs_match_arms = data_enum.variants.iter().map(|variant| {
                let variant_idents = variant_to_field_ident(variant);
                let variant_name = &variant.ident;
                let vec_needed_syms:Vec<_> =
                    variant_to_field_list_without_prefixed_ident_filter_out_basic_ty(variant);
                quote! {#name_inner::#variant_name {#( #variant_idents ),*  } => {
                    vec![#(#vec_needed_syms.erase()),*]
                }}
            });
            let succs_mut_match_arms = data_enum.variants.iter().map(|variant| {
                let variant_idents = variant_to_field_ident(variant);
                let variant_name = &variant.ident;
                let vec_needed_syms:Vec<_> =
                    variant_to_field_list_without_prefixed_ident_filter_out_basic_ty(variant);
                quote! {#name_inner::#variant_name {#( #variant_idents ),*  } => {
                    vec![#(#vec_needed_syms.erase_mut()),*]
                }}
            });
            let to_egglog_match_arms = data_enum.variants.iter().map(|variant| {
                let variant_fields = variant_to_field_ident(variant);
                let variant_name = &variant.ident;
                quote! {#name_inner::#variant_name {#( #variant_fields ),*  } => {
                    #egglog_path::ast::GenericAction::Let(span!(), self.cur_sym().to_string(), 
                        #egglog_path::ast::GenericExpr::Call(span!(),
                            stringify!(#variant_name),
                            vec![#(#variant_fields.to_var()),*]).to_owned_str()
                    )
                }}
            });
            let locate_latest_match_arms = data_enum.variants.iter().map(|variant| {
                let variant_idents = variant_to_field_ident(variant);
                let mapped_variant_idents = variant_to_mapped_ident_type_list(
                    variant,
                    |_,_| {
                        Some(quote! {})
                    },
                    |x,_| {
                        Some(quote! { T::set_latest(#x.erase_mut());})
                    },
                );
                let variant_name = &variant.ident;
                quote! {
                    #name_inner::#variant_name {#(#variant_idents),* } => {
                        T::set_latest(self.node.sym.erase_mut());
                        #(#mapped_variant_idents)*
                    }
                }
            });

            let locate_next_match_arms = data_enum.variants.iter().map(|variant| {
                let variant_idents = variant_to_field_ident(variant);
                let mapped_variant_idents = variant_to_mapped_ident_type_list(
                    variant,
                    |_,_| {
                        Some(quote! {})
                    },
                    |x,_| {
                        Some(quote! {T::set_next(#x.erase_mut());})
                    },
                );
                let variant_name = &variant.ident;
                quote! {
                    #name_inner::#variant_name {#(#variant_idents),* } => {
                        T::set_next(self.node.sym.erase_mut());
                        #(#mapped_variant_idents)*
                    }
                }
            });
            let locate_prev_match_arms = data_enum.variants.iter().map(|variant| {
                let variant_idents = variant_to_field_ident(variant);
                let mapped_variant_idents = variant_to_mapped_ident_type_list(
                    variant,
                    |_,_| {
                        Some(quote! {})
                    },
                    |x,_| {
                        Some(quote! {T::set_prev(#x.erase_mut());})
                    },
                );
                let variant_name = &variant.ident;
                quote! {
                    #name_inner::#variant_name {#(#variant_idents),* } => {
                        T::set_prev(self.node.sym.erase_mut());
                        #(#mapped_variant_idents)*
                    }
                }
            });
            let fns = data_enum.variants.iter().map(|variant|{
                let ref_node_list = variant_to_ref_node_list(&variant);
                let _new_fn_args= variant_to_sym_list(&variant);
                let field_idents_assign = variant_to_assign_node_field_list(&variant);
                let _new_fn_field_idents_assign = variant_to_typed_assign_node_field_list(&variant);
                let field_idents = variant_to_field_ident(&variant);
                let variant_name = &variant.ident;
                let new_fn_name = format_ident!("new_{}",variant_name.to_string().to_snake_case());
                let _new_fn_name = format_ident!("_new_{}",variant_name.to_string().to_snake_case());
                let new_from_term_fn_name = format_ident!("new_{}_from_term",variant_name.to_string().to_snake_case());
                let new_from_term_dyn_fn_name = format_ident!("new_{}_from_term_dyn",variant_name.to_string().to_snake_case());
                let field_ty = variant_to_tys(&variant);
                let field_assignments: Vec<_> = field_idents.into_iter().zip(field_ty.iter()).enumerate().map(|(i, (ident, ty))| {
                    if is_basic_ty(ty) {
                        quote! {
                            #ident: match term_dag.get(children[#i]) {
                                Term::Lit(lit) => lit.clone().try_into().expect("literal type mismatch"),
                                Term::Var(v) => panic!(),
                                Term::App(app,v) => panic!(),
                            }
                        }
                    } else {
                        quote! {
                            #ident: term2sym[&children[#i]].typed()
                        }
                    }
                }).collect();

                quote! {
                    #[track_caller]
                    pub fn #new_fn_name(#(#ref_node_list),*) -> #name_node<T,#variant_name>{
                        let ty = #name_inner::#variant_name {#(#field_idents_assign),*  };
                        use std::panic::Location;
                        let node = Node { ty, sym: #name_counter.next_sym(),span:Some(Location::caller()), _p:PhantomData, _s:PhantomData::<#variant_name>};
                        let node = #name_node {node};
                        T::on_new(&node);
                        node
                    }
                    /// with no side-effect (will not send command to EGraph or change node in WorkAreaGraph)
                    #[track_caller]
                    pub fn #_new_fn_name(#(#_new_fn_args),*) -> #name_node<T,#variant_name>{
                        let ty = #name_inner::#variant_name {#(#_new_fn_field_idents_assign),*  };
                        use std::panic::Location;
                        let node = Node { ty, sym: #name_counter.next_sym(),span:Some(Location::caller()), _p:PhantomData, _s:PhantomData::<#variant_name>};
                        let node = #name_node {node};
                        node
                    }
                    /// you should guarantee all the dep nodes has been init and store as (TermId, Sym) in term2sym
                    #[track_caller]
                    pub fn #new_from_term_fn_name(term_id:TermId, term_dag: &TermDag, term2sym:&mut std::collections::HashMap<TermId, Sym>) -> #name_node<T,#variant_name>{
                        let children = match term_dag.get(term_id){
                            Term::App(app,v) => v,
                            _=> panic!()
                        };
                        let ty = #name_inner::#variant_name {#(#field_assignments),*  };
                        use std::panic::Location;
                        let node = Node { ty, sym: #name_counter.next_sym(),span:Some(Location::caller()), _p:PhantomData, _s:PhantomData};
                        let node = #name_node {node};
                        term2sym.insert(term_id, node.cur_sym());
                        node
                    }
                    #[track_caller]
                    pub fn #new_from_term_dyn_fn_name(term_id:TermId, term_dag: &TermDag, term2sym:&mut std::collections::HashMap<TermId, Sym>) -> Box<dyn EgglogNode>{
                        Box::new(Self::#new_from_term_fn_name(term_id, term_dag, term2sym))
                    }
                }
            });
            let enum_variant_tys_def = data_enum.variants.iter().map(|variant| {
                let variant_name = &variant.ident;

                quote! {
                    #[derive(Clone)]
                    pub struct #variant_name;
                    impl EgglogEnumVariantTy for #variant_name {
                        const TY_NAME:&'static str = stringify!(#variant_name);
                    }
                }
            });

            let set_fns = data_enum.variants.iter().map(|variant|{
                let ref_node_list = variant_to_ref_node_list(&variant);
                let assign_node_field_list = variant_to_assign_node_field_list_without_prefixed_ident(&variant);
                let field_idents = variant_to_field_ident(variant);
                let variant_name = &variant.ident;

                let set_fns = assign_node_field_list.iter().zip(ref_node_list.iter().zip(field_idents.iter()
                    )).map(
                    |(assign_node_field,(ref_node,field_ident))|{
                        let set_fn_name = format_ident!("set_{}",field_ident.to_string());
                        quote! {
                            /// set fn of node, firstly update the sym version and specified field and then informs rx what happen on this node
                            /// rx's behavior depends on whether version control is enabled
                            #[track_caller]
                            pub fn #set_fn_name(&mut self,#ref_node) -> &mut Self{
                                let ___sym = #assign_node_field;
                                if let #name_inner::#variant_name{ #(#field_idents),*} = &mut self.node.ty{
                                    *#field_ident = ___sym
                                };
                                T::on_set(self);
                                self
                            }
                        }
                    }
                );
                let sym_list = variants_to_sym_type_list(variant);
                let get_sym_fns = sym_list.iter().zip(field_idents.iter()
                    ).map(
                    |(sym,field_ident)|{
                        let get_fn_name = format_ident!("{}_sym",field_ident.to_string());
                        quote! {
                            pub fn #get_fn_name(&self) -> #sym{
                                if let #name_inner::#variant_name{ #(#field_idents),*} = &self.node.ty{
                                    #field_ident.clone()
                                }else{
                                    panic!()
                                }
                            }
                        }
                    }
                );
                let get_mut_sym_fns = sym_list.iter().zip(field_idents.iter()
                    ).map(
                    |(sym,field_ident)|{
                        let get_fn_name = format_ident!("{}_sym_mut",field_ident.to_string());
                        quote! {
                            pub fn #get_fn_name(&mut self) -> &mut #sym{
                                if let #name_inner::#variant_name{ #(#field_idents),*} = &mut self.node.ty{
                                    #field_ident
                                }else{
                                    panic!()
                                }
                            }
                        }
                    }
                );

                let vec_needed_syms:Vec<_> =
                    variant_to_field_list_without_prefixed_ident_filter_out_basic_ty(variant);

                quote! {
                    #[allow(unused_variables)]
                    impl<T:TxSgl> #name_node<T,#variant_name>{
                        #(
                            #set_fns
                        )*
                        #(
                            #get_sym_fns
                        )*
                        #(
                            #get_mut_sym_fns
                        )*
                    }
                    impl<T:SingletonGetter> EgglogNode for #name_node<T,#variant_name>{
                        fn succs_mut(&mut self) -> Vec<&mut Sym>{
                            if let #name_inner::#variant_name{ #(#field_idents),*} = &mut self.node.ty{
                                vec![#(#vec_needed_syms.erase_mut()),*]
                            }else{
                                panic!()
                            }
                        }
                        fn succs(&self) -> Vec<Sym>{
                            if let #name_inner::#variant_name{ #(#field_idents),*} = &self.node.ty{
                                vec![#((#vec_needed_syms).erase()),*]
                            }else{
                                panic!()
                            }
                        }
                        fn roll_sym(&mut self) -> Sym{
                            let next_sym = #name_counter.next_sym();
                            self.node.sym = next_sym;
                            next_sym.erase()
                        }
                        fn cur_sym(&self) -> Sym{
                            self.node.sym.erase()
                        }
                        fn cur_sym_mut(&mut self) -> &mut Sym{
                            self.node.sym.erase_mut()
                        }
                        fn clone_dyn(&self) -> Box<dyn EgglogNode>{
                            Box::new(self.clone())
                        }
                    }

                }
            });

            let expanded = quote! {
                pub type #name_node_alias<T,V> = #egglog_path::wrap::Node<#name_egglogty_impl,T,#name_inner,V>;
                #[allow(unused)]
                #[derive(derive_more::Deref,)]
                pub struct #name_node<T: SingletonGetter,V:EgglogEnumVariantTy=()> {
                    node:#name_node_alias<T,V>
                }
                #[allow(unused)]
                #[derive(Debug,Clone)]
                pub struct #name_egglogty_impl;
                // impl NonUnitEgglogEnumVariantTy for #name_egglogty_impl { }
                #[allow(unused)]
                #[derive(Debug,Clone)]
                pub enum #name_inner {
                    #(#variants_def_of_node_with_syms),*
                }
                #[allow(unused_variables)]
                const _:() = {
                    use #egglog_path::*;
                    use std::marker::PhantomData;
                    use #egglog_path::wrap::*;
                    use #egglog_path::prelude::*;
                    use #egglog_path::ast::{GenericAction, GenericExpr};
                    #(#enum_variant_tys_def)*
                    impl<T:TxSgl> #name_node<T,()> {
                        #(#fns)*
                    }
                    impl<T:RxSgl, S: EgglogEnumVariantTy> #name_node<T,S> where Self:EgglogNode {
                        pub fn pull(&self){
                            T::on_pull::<#name_egglogty_impl>(self)
                        }
                    }
                    impl<T:SingletonGetter> EgglogNode for #name_node<T,()> {
                        fn succs_mut(&mut self) -> Vec<&mut Sym>{
                            match &mut self.node.ty{
                                #(#succs_mut_match_arms),*
                            }
                        }
                        fn succs(&self) -> Vec<Sym>{
                            match &self.node.ty{
                                #(#succs_match_arms),*
                            }
                        }
                        fn roll_sym(&mut self) -> Sym{
                            let next_sym = #name_counter.next_sym();
                            self.node.sym = next_sym;
                            next_sym.erase()
                        }
                        fn cur_sym(&self) -> Sym{
                            self.node.sym.erase()
                        }
                        fn cur_sym_mut(&mut self) -> &mut Sym{
                            self.node.sym.erase_mut()
                        }
                        fn clone_dyn(&self) -> Box<dyn EgglogNode>{
                            Box::new(self.clone())
                        }
                    }
                    impl<T:SingletonGetter, V:EgglogEnumVariantTy> ToEgglog for #name_node<T,V>
                    where #name_node<T,V>: EgglogNode{
                        fn to_egglog_string(&self) -> String{
                            match &self.node.ty{
                                #(#to_egglog_string_match_arms),*
                            }
                        }
                        fn to_egglog(&self) -> Action{
                            match &self.node.ty{
                                #(#to_egglog_match_arms),*
                            }
                        }
                    }
                    #[allow(unused_variables)]
                    impl<T:TxSgl + VersionCtlSgl, V:EgglogEnumVariantTy> LocateVersion for #name_node<T,V>
                    where #name_node<T,V> : EgglogNode {
                        fn locate_latest(&mut self) {
                            match &mut self.node.ty{
                                #(#locate_latest_match_arms),*
                            }
                        }
                        fn locate_next(&mut self) {
                            match &mut self.node.ty{
                                #(#locate_next_match_arms),*
                            }
                        }
                        fn locate_prev(&mut self) {
                            match &mut self.node.ty{
                                #(#locate_prev_match_arms),*
                            }
                        }
                    }
                    impl<T: SingletonGetter,  V: EgglogEnumVariantTy> AsRef<#name_node<T, ()>> for #name_node<T, V> {
                        fn as_ref(&self) -> &#name_node<T, ()> {
                            unsafe {
                                &*(self as *const #name_node<T,V> as *const #name_node<T,()>)
                            }
                        }
                    }

                    impl<T:SingletonGetter,V:EgglogEnumVariantTy > Clone for #name_node<T,V> {
                        fn clone(&self) -> Self {
                            Self { node: Node { ty: self.ty.clone(),span: self.span , sym: self.sym.clone(), _p: PhantomData, _s: PhantomData }  }
                        }
                    }

                    impl<T:TxSgl+ VersionCtlSgl + TxCommitSgl,S: EgglogEnumVariantTy> Commit for #name_node<T,S>
                    where
                        #name_node<T, S>: EgglogNode
                    {
                        fn commit(&self) {
                            T::on_commit(self);
                        }
                        fn stage(&self) {
                            T::on_stage(self);
                        }
                    }

                    impl NodeInner<#name_egglogty_impl> for #name_inner {}
                    static #name_counter: TyCounter<#name_egglogty_impl> = TyCounter::new();
                    #(#set_fns)*
                };
            };
            expanded
        }
        Data::Union(_) => todo!(),
    };

    TokenStream::from(quote! {
    #type_def_expanded
    #struct_def_expanded
    })
}
