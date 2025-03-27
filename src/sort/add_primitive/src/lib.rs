use proc_macro::{Span, TokenStream};
use quote::quote;
use syn::parse::{Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::{braced, bracketed, parenthesized, parse_macro_input, Expr, Ident, LitStr, Token};
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
/// - Polymorphism: using `#` as a type will allow any type during
///   typechecking. These will be given type `Value` in the body.
///
/// - Specialized Constraints: using `T (E)` as a type will use
///   `E:expr` in the type constraint but `T:ty` in the body.
///
#[proc_macro]
pub fn add_primitive(input: TokenStream) -> TokenStream {
    // If you're trying to read this code, you should read the big
    // `quote!` block at the bottom of this function first. Trying
    // to parse all the intermediate gobbledygook is going to be
    // hard without that context.

    // Parse the input by implementing `syn::Parse` for our data types.
    let AddPrimitive {
        eg,
        name,
        is_varargs,
        args,
        is_fallible,
        ret,
        body,
    } = parse_macro_input!(input);

    // Create a new field name for the `Prim` struct to hold the return sort.
    // (We reuse the closure argument names for the argument sorts.)
    let y = Ident::new("__y", Span::mixed_site().into());
    // Create a type that refers to whichever `Value` import is closer.
    let value_type = syn::Type::Verbatim(quote!(Value));

    // List out the sorts that we need to store.
    let (field_defs, field_uses): (Vec<_>, Vec<_>) = args
        .iter()
        .map(|arg| (&arg.x, &arg.t))
        .chain([(&y, &ret)])
        .filter_map(|(x, t)| {
            t.field
                .as_ref()
                .map(|(d, u)| (quote!(#x: #d), quote!(#x: #u)))
        })
        .unzip();
    // Bundle up the defs and uses into structs.
    let prim_def = quote!(struct Prim { #(#field_defs,)* });
    let prim_use = quote!(       Prim { #(#field_uses,)* });

    // Create the type constraint for the `egglog` typechecker.
    // TODO: add a new type constraint that supports all possible combinations
    // of features that this macro supports. Right now `|a: #, b: i64|` is
    // impossible to express with the current type constraints.
    let type_constraint = {
        let has_type = |t: &Type| t.field.is_some();
        let args_have_types = args.iter().all(|arg| has_type(&arg.t));
        let ret_has_type = has_type(&ret);

        // If we're in the simple case, we can use the `SimpleTypeConstraint`.
        // However, use of either `#` or varargs makes this impossible.
        if !is_varargs && args_have_types && ret_has_type {
            let x = args.iter().map(|arg| &arg.x);
            quote! {
                let sorts: Vec<ArcSort> = vec![#(self.#x.clone() as ArcSort,)* self.#y.clone()];
                SimpleTypeConstraint::new(self.name(), sorts, span.clone()).into_box()
            }
        } else {
            // As a fallback, try to use `AllEqualTypeConstraint` for everything else.
            let Arg { x, t } = &args[0];
            for arg in &args {
                // NOTE: this is a conservative (incomplete!) check, as `syn::Type`
                // is not `PartialEq`. See the TODO on `type_constraint`.
                assert_eq!(
                    has_type(t),
                    has_type(&arg.t),
                    "AllEqualTypeConstraint doesn't support multiple argument types"
                )
            }

            let new = quote!(AllEqualTypeConstraint::new(self.name(), span.clone()));

            let len = args.len() + 1;
            let len = quote!(.with_exact_length(#len));
            let len = if is_varargs { quote!() } else { len };

            let args = quote!(.with_all_arguments_sort(self.#x.clone()));
            let args = if args_have_types { args } else { quote!() };

            let ret = quote!(.with_output_sort(self.#y.clone()));
            let ret = if ret_has_type { ret } else { quote!() };

            quote!(#new #len #args #ret .into_box())
        }
    };

    // Create the function body for either `apply` or `invoke`.
    let body = |target| {
        // Bind the argument names that were passed in to this macro.
        let bind = if is_varargs {
            let x = &args[0].x;
            quote!(let #x = args.iter();)
        } else {
            let x = args.iter().map(|arg| &arg.x);
            quote!(let [#(#x),*] = args else { panic!("wrong number of arguments") };)
        };

        // Cast the arguments to the desired type.
        let cast1 = |x, t: &syn::Type| match target {
            Target::OldBackend => quote!(<#t as FromSort>::load(&self.#x, #x)),
            Target::NewBackend => quote!(prims.unwrap::<#t>(*#x)),
        };
        let cast = if is_varargs {
            let Arg { x, t } = &args[0];
            match &t.cast {
                None => quote!(let #x = #x.copied();),
                Some(t) => {
                    let cast = cast1(x, t);
                    quote!(let #x = #x.map(|#x| #cast);)
                }
            }
        } else {
            args.iter()
                .map(|Arg { x, t }| match &t.cast {
                    None => quote!(let #x = *#x;),
                    Some(t) => {
                        let cast = cast1(x, t);
                        quote!(let #x: #t = #cast;)
                    }
                })
                .collect()
        };

        // Do typechecking on the result of the body.
        let yt = match &ret.cast {
            None => &value_type,
            Some(t) => t,
        };
        // If the primitive is fallible, put a `?` after the body.
        let fail = if is_fallible { quote!(?) } else { quote!() };
        // Cast the result back to an interned value.
        let ret = match &ret.cast {
            None => quote!(#y),
            Some(t) => match target {
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
        #[allow(unused_imports)] use ::egglog::{*, constraint::*};
        #[allow(unused_imports)] use ::std::sync::Arc;

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

        {
            use core_relations::{ExecutionState, ExternalFunction, Value};
            impl ExternalFunction for Ext {
                fn invoke(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
                    #[allow(unused_variables)] let prims = exec_state.prims();
                    #invoke
                }
            }
        }

        let prim = #prim_use;
        let ext = eg.backend.register_external_func(Ext);
        eg.add_primitive(Primitive(Arc::new(prim), ext))
    }}.into()
}

struct AddPrimitive {
    eg: Expr,
    name: LitStr,
    args: Vec<Arg>,
    ret: Type,
    body: Expr,
    is_varargs: bool,
    is_fallible: bool,
}

impl Parse for AddPrimitive {
    fn parse(input: ParseStream) -> Result<Self> {
        let eg = input.parse()?;
        input.parse::<Token![,]>()?;
        let name = input.parse()?;
        input.parse::<Token![=]>()?;
        let Args { is_varargs, args } = input.parse()?;
        let Arrow { is_fallible } = input.parse()?;
        let ret = input.parse()?;

        let body;
        braced!(body in input);
        let body = body.parse()?;

        Ok(AddPrimitive {
            eg,
            name,
            args,
            ret,
            body,
            is_varargs,
            is_fallible,
        })
    }
}

struct Args {
    is_varargs: bool,
    args: Vec<Arg>,
}

impl Parse for Args {
    fn parse(input: ParseStream) -> Result<Self> {
        fn var(input: ParseStream) -> Result<Args> {
            let arg;
            bracketed!(arg in input);
            Ok(Args {
                is_varargs: true,
                args: vec![arg.parse()?],
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
                args: args.into_iter().collect(),
            })
        }

        var(input).or_else(|a| {
            fix(input).map_err(|b| {
                let mut e = a;
                e.combine(b);
                e
            })
        })
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

struct Type {
    cast: Option<syn::Type>,
    field: Option<(syn::Type, Expr)>,
}

impl Parse for Type {
    fn parse(input: ParseStream) -> Result<Self> {
        let cast = match input.parse::<Token![#]>() {
            Ok(_) => None,
            Err(_) => Some(input.parse()?),
        };

        let field_def = syn::Type::Verbatim(match &cast {
            None => quote!(ArcSort),
            Some(t) => quote!(Arc<<#t as IntoSort>::Sort>),
        });

        let field = if input.peek(syn::token::Paren) {
            let inner;
            parenthesized!(inner in input);
            let field_use = inner.parse()?;

            Some((field_def, field_use))
        } else if let Some(t) = &cast {
            let sort = quote!(<#t as IntoSort>::Sort);

            let field_use = Expr::Verbatim(quote! {
               eg.type_info
                   .get_sort_by::<#sort>(|_| true)
                   .unwrap_or_else(|| panic!(
                       "Failed to lookup sort: {}",
                       ::std::any::type_name::<#t>()
                   ))
            });

            Some((field_def, field_use))
        } else {
            None
        };

        Ok(Type { cast, field })
    }
}

struct Arrow {
    is_fallible: bool,
}

impl Parse for Arrow {
    fn parse(input: ParseStream) -> Result<Self> {
        syn::custom_punctuation!(FailArrow, -?>);

        if input.parse::<Token![->]>().is_ok() {
            Ok(Arrow { is_fallible: false })
        } else if input.parse::<FailArrow>().is_ok() {
            Ok(Arrow { is_fallible: true })
        } else {
            Err(input.error("expected -> or -?>"))
        }
    }
}

enum Target {
    NewBackend,
    OldBackend,
}
