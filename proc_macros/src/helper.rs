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

pub fn variant_to_mapped_ident_type_list(
    variant: &Variant,
    mut map_basic_ident: impl FnMut(&Ident, &Ident) -> Option<TokenStream>,
    mut map_complex_ident: impl FnMut(&Ident, &Ident) -> Option<TokenStream>,
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
        let f1 = &format_ident!("{}",&f1.to_token_stream().to_string());
        match f1.to_string().as_str() {
            x if PANIC_TY_LIST.contains(&x) => {
                panic!("{} not supported", x)
            }
            x if EGGLOG_BASIC_TY_LIST.contains(&x) => map_basic_ident(&f2, &f1),
            _ => map_complex_ident(&f2, &f1),
        }.map(|x|quote! {  #x})
    })
    .flatten().collect();
    types_and_idents
}
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
pub fn is_basic_ty(ty: &proc_macro2::TokenStream) -> bool {
    if EGGLOG_BASIC_TY_LIST.contains(&ty.to_string().as_str()) {
        return true;
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
    variant_to_mapped_ident_type_list(variant, 
        |basic,basic_ty| Some(quote!{#basic:#basic_ty}), 
        |complex, complex_ty|{
                let name_egglogty = format_ident!("{}Ty", complex_ty);
        Some(quote!{#complex:Sym<#name_egglogty>})})
}
pub fn variants_to_sym_type_list(variant: &Variant) -> Vec<proc_macro2::TokenStream> {
    variant_to_mapped_ident_type_list(variant, 
        |_,basic_ty| Some(quote!{#basic_ty}), 
        |_, complex_ty|{
                let name_egglogty = format_ident!("{}Ty", complex_ty);
        Some(quote!{Sym<#name_egglogty>})})
}
pub fn variant_to_ref_node_list(variant: &Variant) -> Vec<proc_macro2::TokenStream> {
    variant_to_mapped_ident_type_list(variant, 
        |basic,basic_ty| Some(quote!{#basic:#basic_ty}), 
        |complex, complex_ty|Some(quote!{#complex: &#complex_ty<T, impl EgglogEnumVariantTy>}))
}
pub fn variant_to_sym_list(variant: &Variant) -> Vec<proc_macro2::TokenStream> {
    variant_to_mapped_ident_type_list(variant, 
        |basic,basic_ty| Some(quote!(#basic:#basic_ty)), 
        |complex,_| Some(quote!(#complex:Sym)))

}
pub fn variant_to_assign_node_field_list(variant: &Variant) -> Vec<proc_macro2::TokenStream> {
    variant_to_mapped_ident_type_list(variant, 
        |basic,_| Some(quote! {#basic}), 
        |complex, _|Some(quote!(#complex:#complex.sym)))
}
pub fn variant_to_typed_assign_node_field_list(variant: &Variant) -> Vec<proc_macro2::TokenStream> {
    variant_to_mapped_ident_type_list(variant, 
        |basic,_| Some(quote! {#basic}), 
        |complex, _|Some(quote!(#complex:#complex.typed())))
}
pub fn variant_to_assign_node_field_list_without_prefixed_ident(
    variant: &Variant,
) -> Vec<proc_macro2::TokenStream> {
    variant_to_mapped_ident_type_list(variant, 
        |basic,_|Some(quote! {#basic}), 
        |complex,_|Some(quote! {#complex.sym}))
}
pub fn variant_to_field_list_without_prefixed_ident_filter_out_basic_ty(
    variant: &Variant,
) -> Vec<proc_macro2::TokenStream> {
    variant_to_mapped_ident_type_list(variant, |_,_|None, |ident,_|Some(quote!{#ident}))
}

/// given variant a{ x:X, y:Y}
/// return vec![ X, Y ]
pub fn variant_to_tys(variant: &Variant) -> Vec<proc_macro2::TokenStream> {
    variant_to_mapped_ident_type_list(variant, 
        |_,ty|Some(quote!{#ty}), 
        |_,ty|Some(quote!{#ty}))
}

/// given variant a{ x:X, y:Y}
/// return iterator [ x, y ].iter()
pub fn variant_to_field_ident(variant: &Variant) -> Vec<proc_macro2::TokenStream> {
    variant_to_mapped_ident_type_list(variant, 
        |ident,_|Some(quote!{#ident}), 
        |ident,_|Some(quote!{#ident}))
}

// /// given variant a{ x:Box<X>}
// /// return  dyn AsRef<#_first_generic<T, ()>> 
// /// given variant a{ x:i32}
// /// return  i32 
// pub fn variant_to_as_ref_type(variant: &Variant) -> (proc_macro2::TokenStream, bool){
//     let mut is_basic = false;
//     (variant_to_mapped_ident_type_list(variant, 
//         |_,basic_ty|{is_basic = true;Some( quote!{#basic_ty})}, 
//         |_,complex|{Some(quote!{dyn AsRef<#complex<T,()>>})}).first().unwrap().clone()
//     ,is_basic)
// }