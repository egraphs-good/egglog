use proc_macro::TokenStream;

#[proc_macro]
pub fn add_primitive(input: TokenStream) -> TokenStream {
    let _ = input;

    unimplemented!()
}
