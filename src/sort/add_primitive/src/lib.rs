use proc_macro::{Span, TokenStream};
use quote::quote;
use syn::parse::{Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::{braced, bracketed, parse_macro_input, Expr, Ident, LitStr, Token};
///
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
#[proc_macro]
pub fn add_primitive(input: TokenStream) -> TokenStream {
    // If you're trying to read this code, you should read the big
    // `quote!` block at the bottom of this function first. Trying
    // to parse all the intermediate gobbledygook is going to be
    // hard without that context.

    // Parse the input by implementing `syn::Parse` for our data types.
    let AddPrimitive(eg, name, args, arrow, ret, body) = parse_macro_input!(input);

    // Create a new field name for the `Prim` struct to hold the return sort.
    // (We reuse the closure argument names for the argument sorts.)
    let y = Ident::new("__y", Span::mixed_site().into());

    // List out the sorts that we need to store.
    let fields = args
        .vec
        .iter()
        .map(|arg| (&arg.x, &arg.t))
        .chain([(&y, &ret)])
        .filter_map(|(x, t)| match t {
            Type::Value => None,
            Type::Type(t) => Some((x, t)),
        });
    // Create a struct field definition and instantiation for each sort.
    let field_defs = fields
        .clone()
        .map(|(x, t)| quote!(#x: Arc<<#t as IntoSort>::Sort>));
    let field_uses = fields.map(|(x, t)| {
        quote! {
            #x: eg.type_info
               .get_sort_by::<<#t as IntoSort>::Sort>(|_| true)
               .unwrap_or_else(|| panic!("Failed to lookup sort: {}", std::any::type_name::<#t>()))
        }
    });
    // Bundle up the defs and uses into structs.
    let prim_def = quote!(struct Prim { #(#field_defs,)* });
    let prim_use = quote!(       Prim { #(#field_uses,)* });

    // Create the type constraint for the `egglog` typechecker.
    // TODO: add a new type constraint that supports all possible combinations
    // of features that this macro supports. Right now `|a: #, b: i64|` is
    // impossible to express with the current type constraints.
    let type_constraint = {
        let arg_is_value = args.vec.iter().any(|arg| matches!(arg.t, Type::Value));
        let ret_is_value = matches!(ret, Type::Value);

        // If we're in the simple case, we can use the `SimpleTypeConstraint`.
        // However, use of either `#` or varargs makes this impossible.
        if !args.is_varargs && !arg_is_value && !ret_is_value {
            let x = args.vec.iter().map(|arg| &arg.x);
            quote! {
                let sorts = vec![#(self.#x.clone() as ArcSort,)* self.#y.clone()];
                SimpleTypeConstraint::new(self.name(), sorts, span.clone()).into_box()
            }
        } else {
            // As a fallback, try to use `AllEqualTypeConstraint` for everything else.
            let Arg { x, t } = &args.vec[0];
            for arg in &args.vec {
                // NOTE: this is a conservative (incomplete!) check, as `syn::Type`
                // is not `PartialEq`. See the TODO on `type_constraint`.
                if let (Type::Value, Type::Type(_)) | (Type::Type(_), Type::Value) = (t, &arg.t) {
                    todo!("AllEqualTypeConstraint doesn't support multiple argument types")
                }
            }

            let new = quote!(AllEqualTypeConstraint::new(self.name(), span.clone()));

            let len = args.vec.len() + 1;
            let len = quote!(.with_exact_length(#len));
            let len = if args.is_varargs { quote!() } else { len };

            let args = quote!(.with_all_arguments_sort(self.#x.clone()));
            let args = if arg_is_value { quote!() } else { args };

            let ret = quote!(.with_output_sort(self.#y.clone()));
            let ret = if ret_is_value { quote!() } else { ret };

            quote!(#new #len #args #ret .into_box())
        }
    };

    // Create the function body for either `apply` or `invoke`.
    let body = |target| {
        // Bind the argument names that were passed in to this macro.
        let bind = if args.is_varargs {
            let x = &args.vec[0].x;
            quote!(let #x = args.iter();)
        } else {
            let x = args.vec.iter().map(|arg| &arg.x);
            quote!(let [#(#x,)*] = args else { panic!("wrong number of arguments") };)
        };

        // Cast the arguments to the desired type.
        let cast1 = |x, t: &syn::Type| match target {
            Target::OldBackend => quote!(<#t as FromSort>::load(&self.#x, #x)),
            Target::NewBackend => quote!(prims.unwrap::<#t>(*#x)),
        };
        let cast = if args.is_varargs {
            let Arg { x, t } = &args.vec[0];
            match t {
                Type::Value => quote!(),
                Type::Type(t) => {
                    let cast = cast1(x, t);
                    quote!(let #x = #x.map(|#x| #cast);)
                }
            }
        } else {
            args.vec
                .iter()
                .map(|Arg { x, t }| match t {
                    Type::Value => quote!(),
                    Type::Type(t) => {
                        let cast = cast1(x, t);
                        quote!(let #x: #t = #cast;)
                    }
                })
                .collect()
        };

        // Do typechecking on the result of the body.
        let yt = ret.specialize(&target);
        // If the primitive is fallible, put a `?` after the body.
        let fail = if arrow.fallible { quote!(?) } else { quote!() };
        // Cast the result back to an interned value.
        let ret = match &ret {
            Type::Value => quote!(#y),
            Type::Type(t) => match target {
                Target::OldBackend => quote!(#y.store(&self.#y)),
                Target::NewBackend => quote!(prims.get::<#t>(#y)),
            },
        };

        quote! {
            #bind
            #cast
            let #y: #yt = (#body)#fail;
            Some(#ret)
        }
    };
    let apply = body(Target::OldBackend);
    let invoke = body(Target::NewBackend);

    // This is the big `quote!` block that ties everything together. We
    // create two structs: `Prim` for the frontend, and `Ext` for the
    // backend. `Prim` has to support `get_type_constraint`, so it stores
    // sorts in its fields. `Ext` only has to support `invoke`, so it does not.
    quote!{{
        #[allow(unused_imports)] use crate::{*, constraint::*};
        #[allow(unused_imports)] use ::std::sync::Arc;
        use core_relations::{ExecutionState, ExternalFunction, Value as V};

        let eg: &mut EGraph = #eg;

        #prim_def

        impl PrimitiveLike for Prim {
            fn name(&self) -> Symbol {
                #name.into()
            }

            fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
                #type_constraint
            }

            fn apply(&self, args: &[Value], _: (&[ArcSort], &ArcSort), _: Option<&mut EGraph>) -> Option<Value> {
                #apply
            }
        }

        #[derive(Clone)]
        struct Ext;

        impl ExternalFunction for Ext {
            fn invoke(&self, exec_state: &mut ExecutionState, args: &[V]) -> Option<V> {
                #[allow(unused_variables)] let prims = exec_state.prims();
                #invoke
            }
        }

        let prim = #prim_use;
        let ext = eg.backend.register_external_func(Ext);
        eg.add_primitive(Primitive(Arc::new(prim), ext))
    }}.into()
}

struct AddPrimitive(Expr, LitStr, Args, Arrow, Type, Expr);

impl Parse for AddPrimitive {
    fn parse(input: ParseStream) -> Result<Self> {
        let eg = input.parse()?;
        input.parse::<Token![,]>()?;
        let name = input.parse()?;
        input.parse::<Token![=]>()?;
        let args = input.parse()?;
        let arrow = input.parse()?;
        let ret = input.parse()?;

        let body;
        braced!(body in input);
        let body = body.parse()?;

        Ok(AddPrimitive(eg, name, args, arrow, ret, body))
    }
}

struct Args {
    is_varargs: bool,
    vec: Vec<Arg>,
}

impl Parse for Args {
    fn parse(input: ParseStream) -> Result<Self> {
        fn var(input: ParseStream) -> Result<Args> {
            let arg;
            bracketed!(arg in input);
            Ok(Args {
                is_varargs: true,
                vec: vec![arg.parse()?],
            })
        }

        fn fix(input: ParseStream) -> Result<Args> {
            let mut args = Punctuated::<Arg, Token![,]>::new();

            input.parse::<Token![|]>()?;
            loop {
                if input.peek(Token![|]) {
                    break;
                }
                args.push_value(input.parse()?);
                if input.peek(Token![|]) {
                    break;
                }
                args.push_punct(input.parse()?);
            }
            input.parse::<Token![|]>()?;

            Ok(Args {
                is_varargs: false,
                vec: args.into_iter().collect(),
            })
        }

        var(input).or_else(|_| fix(input))
    }
}

struct Arg {
    x: Ident,
    t: Type,
}

impl Parse for Arg {
    fn parse(input: ParseStream) -> Result<Self> {
        let x = input.parse()?;
        input.parse::<Token![:]>()?;
        let t = input.parse()?;
        Ok(Arg { x, t })
    }
}

// We separate `Value` from other types because we
// need to be able to specialize it to both backends.
enum Type {
    Value,
    Type(syn::Type),
}

impl Type {
    fn specialize(&self, target: &Target) -> syn::Type {
        match self {
            Type::Type(t) => t.clone(),
            Type::Value => syn::Type::Verbatim(match target {
                Target::NewBackend => quote!(V),
                Target::OldBackend => quote!(Value),
            }),
        }
    }
}

impl Parse for Type {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.parse::<Token![#]>().is_ok() {
            Ok(Type::Value)
        } else {
            Ok(Type::Type(input.parse()?))
        }
    }
}

struct Arrow {
    fallible: bool,
}

impl Parse for Arrow {
    fn parse(input: ParseStream) -> Result<Self> {
        syn::custom_punctuation!(FailArrow, -?>);

        if input.parse::<Token![->]>().is_ok() {
            Ok(Arrow { fallible: false })
        } else if input.parse::<FailArrow>().is_ok() {
            Ok(Arrow { fallible: true })
        } else {
            Err(input.error("expected -> or -?>"))
        }
    }
}

enum Target {
    NewBackend,
    OldBackend,
}
