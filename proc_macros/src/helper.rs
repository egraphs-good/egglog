use core::panic;

use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{ToTokens, format_ident, quote};
use syn::{Fields, GenericArgument, PathArguments, Type, Variant};

pub const PANIC_TY_LIST: [&'static str; 4] = ["i32", "u32", "u64", "f32"];
pub const EGGLOG_BASIC_TY_LIST: [&'static str; 3] = ["String", "i64", "f64"];

pub fn egglog_path() -> proc_macro2::TokenStream {
    match (
        crate_name("egglog"),
        std::env::var("CARGO_CRATE_NAME").as_deref(),
    ) {
        (Ok(FoundCrate::Itself), Ok(_)) => quote!(::egglog),
        (Ok(FoundCrate::Name(name)), _) => {
            let ident = proc_macro2::Ident::new(&name, Span::call_site());
            quote!(::#ident)
        }
        _ => quote!(::egglog),
    }
}
#[allow(unused)]
pub fn smallvec_wrapper_path() -> proc_macro2::TokenStream {
    match (
        crate_name("smallvec"),
        std::env::var("CARGO_CRATE_NAME").as_deref(),
    ) {
        (Ok(FoundCrate::Itself), Ok(_)) => quote!(crate),
        (Ok(FoundCrate::Name(name)), _) => {
            let ident = proc_macro2::Ident::new(&name, Span::call_site());
            quote!(::#ident)
        }
        _ => quote!(::smallvec),
    }
}
#[allow(unused)]
pub fn derive_more_path() -> proc_macro2::TokenStream {
    match (
        crate_name("derive_more"),
        std::env::var("CARGO_CRATE_NAME").as_deref(),
    ) {
        (Ok(FoundCrate::Itself), Ok(_)) => quote!(crate),
        (Ok(FoundCrate::Name(name)), _) => {
            let ident = proc_macro2::Ident::new(&name, Span::call_site());
            quote!(::#ident)
        }
        _ => quote!(::derive_more),
    }
}

// /// postfix a type with "Sym" except for basic types for egglog (String,i64...)
// /// for example:  Node ->  NodeSym
// pub fn postfix_type(ty: &Type, postfix: &str, generic:Option<&str>) -> proc_macro2::TokenStream {
//     match ty {
//         Type::Path(type_path) => {
//             let type_name = &type_path.path.segments.last().unwrap().ident;
//             let sym_name = format_ident!("{}{}", type_name, postfix).to_token_stream(); // 拼接 `Sym`
//             match generic{
//                 Some(g) => {
//                     let g = format_ident!("T");
//                     quote!(#sym_name<#g>) // concat generic
//                 },
//                 None => {
//                     sym_name
//                 },
//             }
//         }
//         _ => panic!("Unsupported type for `WithSymNode`"),
//     }
// }
#[allow(unused)]
pub fn get_ref_type(ty: &Type) -> proc_macro2::TokenStream {
    match ty {
        Type::Path(type_path) => {
            let type_name = &type_path.path.segments.last().unwrap().ident;
            let sym_name = format_ident!("&{}", type_name); // 拼接 `Sym`
            quote! { #sym_name }
        }
        _ => panic!("Unsupported type for `WithSymNode`"),
    }
}
pub fn inventory_path() -> proc_macro2::TokenStream {
    match (
        crate_name("inventory"),
        std::env::var("CARGO_CRATE_NAME").as_deref(),
    ) {
        (Ok(FoundCrate::Itself), Ok(_)) => quote!(crate),
        (Ok(FoundCrate::Name(name)), _) => {
            let ident = proc_macro2::Ident::new(&name, Span::call_site());
            quote!(::#ident)
        }
        _ => quote!(::inventory),
    }
}
pub fn is_vec_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            if segment.ident == "Vec" {
                return true;
            }
        }
    }
    false
}
pub fn is_box_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            if segment.ident == "Box" {
                return true;
            }
        }
    }
    false
}
pub fn get_first_generic(ty: &Type) -> &Type {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            if let PathArguments::AngleBracketed(args) = &segment.arguments {
                let arg = args
                    .args
                    .iter()
                    .nth(0)
                    .expect("type should at least have one generic");
                if let GenericArgument::Type(inner_ty) = arg {
                    // inner_ty is Vec<T>'s T
                    return inner_ty;
                }
            }
        }
    }
    panic!("first generic generic can only be Type")
}

/// given variant a{ x:X, y:Y}
/// return vec![ x:XSym, y:YSym ]
pub fn variants_to_sym_typed_ident_list(variant: &Variant) -> Vec<proc_macro2::TokenStream> {
    let types_and_idents = match &variant.fields {
        Fields::Named(fields_named) => fields_named.named.iter(),
        Fields::Unit => {
            panic!("add `{{}}` to the unit variant")
        }
        _ => panic!("only support named fields"),
    }
    .map(|f| {
        let f_ident = f
            .ident
            .as_ref()
            .expect("don't support unnamed field")
            .clone();
        // if it's a box type we should read the first generic
        if is_box_type(&f.ty) {
            (get_first_generic(&f.ty).clone(), f_ident)
        } else {
            (f.ty.clone(), f_ident)
        }
    })
    .map(|(f1, f2)| {
        let sym_ty = match f1.to_token_stream().to_string().as_str() {
            x if PANIC_TY_LIST.contains(&x) => {
                panic!("{} not supported", x)
            }
            x if EGGLOG_BASIC_TY_LIST.contains(&x) => f1.to_token_stream(),
            _ => {
                let f1_ident = match &f1 {
                    Type::Path(type_path) => {
                        type_path
                            .path
                            .segments
                            .last()
                            .expect("impossible")
                            .clone()
                            .ident
                    }
                    _ => panic!(),
                };
                let name_egglogty = format_ident!("{}Ty", f1_ident);
                quote!( Sym<#name_egglogty>)
            }
        };
        let ident = f2;
        quote! { #ident :#sym_ty}
    })
    .collect::<Vec<_>>();
    types_and_idents
}
pub fn variants_to_sym_type_list(variant: &Variant) -> Vec<proc_macro2::TokenStream> {
    let types = match &variant.fields {
        Fields::Named(fields_named) => fields_named.named.iter(),
        Fields::Unit => {
            panic!("add `{{}}` to the unit variant")
        }
        _ => panic!("only support named fields"),
    }
    .map(|f| {
        let f_ident = f
            .ident
            .as_ref()
            .expect("don't support unnamed field")
            .clone();
        // if it's a box type we should read the first generic
        if is_box_type(&f.ty) {
            (get_first_generic(&f.ty).clone(), f_ident)
        } else {
            (f.ty.clone(), f_ident)
        }
    })
    .map(|(f1, _)| {
        let sym_ty = match f1.to_token_stream().to_string().as_str() {
            x if PANIC_TY_LIST.contains(&x) => {
                panic!("{} not supported", x)
            }
            x if EGGLOG_BASIC_TY_LIST.contains(&x) => f1.to_token_stream(),
            _ => {
                let f1_ident = match &f1 {
                    Type::Path(type_path) => {
                        type_path
                            .path
                            .segments
                            .last()
                            .expect("impossible")
                            .clone()
                            .ident
                    }
                    _ => panic!(),
                };
                let name_egglogty = format_ident!("{}Ty", f1_ident);
                quote!( Sym<#name_egglogty>)
            }
        };
        quote! { #sym_ty}
    })
    .collect::<Vec<_>>();
    types
}
pub fn variant_to_ref_node_list(variant: &Variant, _: &Ident) -> Vec<proc_macro2::TokenStream> {
    let types_and_idents = match &variant.fields {
        Fields::Named(fields_named) => fields_named.named.iter(),
        Fields::Unit => {
            panic!("add `{{}}` to the unit variant")
        }
        _ => panic!("only support named fields"),
    }
    .map(|f| {
        let f_ident = f
            .ident
            .as_ref()
            .expect("don't support unnamed field")
            .clone();
        // if it's a box type we should read the first generic
        if is_box_type(&f.ty) {
            (get_first_generic(&f.ty).clone(), f_ident)
        } else {
            (f.ty.clone(), f_ident)
        }
    })
    .map(|(f1, f2)| {
        let node_ty = match f1.to_token_stream().to_string().as_str() {
            x if PANIC_TY_LIST.contains(&x) => {
                panic!("{} not supported", x)
            }
            x if EGGLOG_BASIC_TY_LIST.contains(&x) => f1.to_token_stream(),
            _ => {
                let f1_ident = match &f1 {
                    Type::Path(type_path) => {
                        type_path
                            .path
                            .segments
                            .last()
                            .expect("impossible")
                            .clone()
                            .ident
                    }
                    _ => panic!(),
                };
                let name_node = format_ident!("{}", f1_ident);
                quote! { &#name_node<T, impl EgglogEnumVariantTy> }
            }
        };
        let ident = f2;
        quote! { #ident : #node_ty}
    })
    .collect::<Vec<_>>();
    types_and_idents
}
pub fn variant_to_mapped_ident_list(
    variant: &Variant,
    map_basic_ty: impl Fn(&Ident) -> TokenStream,
    map_complex_ty: impl Fn(&Ident) -> TokenStream,
) -> Vec<proc_macro2::TokenStream> {
    let types_and_idents = match &variant.fields {
        Fields::Named(fields_named) => fields_named.named.iter(),
        Fields::Unit => {
            panic!("add `{{}}` to the unit variant")
        }
        _ => panic!("only support named fields"),
    }
    .map(|f| {
        let f_ident = f
            .ident
            .as_ref()
            .expect("don't support unnamed field")
            .clone();
        // if it's a box type we should read the first generic
        if is_box_type(&f.ty) {
            (get_first_generic(&f.ty).clone(), f_ident)
        } else {
            (f.ty.clone(), f_ident)
        }
    })
    .map(|(f1, f2)| {
        let mapped_ident = match f1.to_token_stream().to_string().as_str() {
            x if PANIC_TY_LIST.contains(&x) => {
                panic!("{} not supported", x)
            }
            x if EGGLOG_BASIC_TY_LIST.contains(&x) => map_basic_ty(&f2),
            _ => map_complex_ty(&f2),
        };
        quote! {  #mapped_ident}
    })
    .collect::<Vec<_>>();
    types_and_idents
}
pub fn variants_to_assign_node_field_list(variant: &Variant) -> Vec<proc_macro2::TokenStream> {
    let types_and_idents = match &variant.fields {
        Fields::Named(fields_named) => fields_named.named.iter(),
        Fields::Unit => {
            panic!("add `{{}}` to the unit variant")
        }
        _ => panic!("only support named fields"),
    }
    .map(|f| {
        let f_ident = f
            .ident
            .as_ref()
            .expect("don't support unnamed field")
            .clone();
        // if it's a box type we should read the first generic
        if is_box_type(&f.ty) {
            (get_first_generic(&f.ty).clone(), f_ident)
        } else {
            (f.ty.clone(), f_ident)
        }
    })
    .zip(variant_to_field_ident(variant))
    .map(|((f1, f2), ident)| {
        let node_ty = match f1.to_token_stream().to_string().as_str() {
            x if PANIC_TY_LIST.contains(&x) => {
                panic!("{} not supported", x)
            }
            x if EGGLOG_BASIC_TY_LIST.contains(&x) => ident.to_token_stream(),
            _ => {
                quote!(#ident.sym)
            }
        };
        let ident = f2;
        quote! { #ident : #node_ty}
    })
    .collect::<Vec<_>>();
    types_and_idents
}
pub fn variants_to_assign_node_field_list_without_prefixed_ident(
    variant: &Variant,
) -> Vec<proc_macro2::TokenStream> {
    let types_and_idents = match &variant.fields {
        Fields::Named(fields_named) => fields_named.named.iter(),
        Fields::Unit => {
            panic!("add `{{}}` to the unit variant")
        }
        _ => panic!("only support named fields"),
    }
    .map(|f| {
        let f_ident = f
            .ident
            .as_ref()
            .expect("don't support unnamed field")
            .clone();
        // if it's a box type we should read the first generic
        if is_box_type(&f.ty) {
            (get_first_generic(&f.ty).clone(), f_ident)
        } else {
            (f.ty.clone(), f_ident)
        }
    })
    .zip(variant_to_field_ident(variant))
    .map(|((f1, _), ident)| {
        let node_ty = match f1.to_token_stream().to_string().as_str() {
            x if PANIC_TY_LIST.contains(&x) => {
                panic!("{} not supported", x)
            }
            x if EGGLOG_BASIC_TY_LIST.contains(&x) => ident.to_token_stream(),
            _ => {
                quote!(#ident.sym)
            }
        };
        quote! { #node_ty}
    })
    .collect::<Vec<_>>();
    types_and_idents
}
pub fn variant_to_field_list_without_prefixed_ident_filter_out_basic_ty(
    variant: &Variant,
) -> Vec<proc_macro2::TokenStream> {
    let ty = match &variant.fields {
        Fields::Named(fields_named) => fields_named.named.iter(),
        Fields::Unit => {
            panic!("add `{{}}` to the unit variant")
        }
        _ => panic!("only support named fields"),
    }
    .map(|f| {
        let f_ident = f
            .ident
            .as_ref()
            .expect("don't support unnamed field")
            .clone();
        // if it's a box type we should read the first generic
        if is_box_type(&f.ty) {
            (get_first_generic(&f.ty).clone(), f_ident)
        } else {
            (f.ty.clone(), f_ident)
        }
    })
    .zip(variant_to_field_ident(variant))
    .filter_map(|((f1, _f2), ident)| {
        let node_ty = match f1.to_token_stream().to_string().as_str() {
            x if PANIC_TY_LIST.contains(&x) => {
                panic!("{} not supported", x)
            }
            x if EGGLOG_BASIC_TY_LIST.contains(&x) => return None,
            _ => {
                quote!(#ident)
            }
        };
        Some(quote! { #node_ty})
    })
    .collect::<Vec<_>>();
    ty
}

/// given variant a{ x:X, y:Y}
/// return vec![ X, Y ]
pub fn variant_to_tys(variant: &Variant) -> Vec<Type> {
    let tys = match &variant.fields {
        Fields::Named(fields_named) => fields_named.named.iter(),
        Fields::Unit => {
            panic!("add `{{}}` to the unit variant")
        }
        _ => panic!("only support named fields"),
    }
    .map(|f| {
        if is_box_type(&f.ty) {
            get_first_generic(&f.ty).clone()
        } else {
            f.ty.clone()
        }
    })
    .collect::<Vec<_>>();
    tys
}

/// given variant a{ x:X, y:Y}
/// return iterator [ x, y ].iter()
pub fn variant_to_field_ident(variant: &Variant) -> impl Iterator<Item = &proc_macro2::Ident> {
    match &variant.fields {
        Fields::Named(fields_named) => fields_named.named.iter(),
        Fields::Unit => {
            panic!("add `{{}}` to the unit variant")
        }
        _ => panic!("only support named fields"),
    }
    .map(|f| f.ident.as_ref().unwrap())
}
