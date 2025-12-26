use proc_macro::{Span, TokenStream};
use quote::quote;
use syn::parse::{Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::{Expr, Ident, LitStr, Token, braced, bracketed, parenthesized, parse_macro_input};

/// This macro lets the user declare custom egglog primitives with explicit validators.
/// It combines primitive registration with a custom validator function.
///
/// # Example
/// ```rust,ignore
/// add_primitive_with_validator!(
///     eg,
///     "custom-op" = |a: Value| -> Value { a },
///     |termdag, lhs| {
///         // Custom validation logic
///         None
///     }
/// );
/// ```
#[proc_macro]
pub fn add_primitive_with_validator(input: TokenStream) -> TokenStream {
    // Convert to string to do manual parsing
    let input_str = input.to_string();

    // Find the end of the primitive body (last })
    let mut brace_depth = 0;
    let mut in_string = false;
    let mut primitive_end = 0;

    for (i, ch) in input_str.chars().enumerate() {
        if ch == '"' && (i == 0 || input_str.chars().nth(i.saturating_sub(1)) != Some('\\')) {
            in_string = !in_string;
        }
        if !in_string {
            if ch == '{' {
                brace_depth += 1;
            } else if ch == '}' {
                brace_depth -= 1;
                if brace_depth == 0 {
                    primitive_end = i + 1;
                }
            }
        }
    }

    // Find the comma after the primitive body
    let rest = &input_str[primitive_end..];
    let comma_pos = rest.find(',').expect("Expected comma after primitive body");
    let split_point = primitive_end + comma_pos;

    let primitive_part = &input_str[..split_point];
    let validator_part = &input_str[split_point + 1..].trim();

    // Parse the primitive part
    let prim_tokens: TokenStream = primitive_part.parse().unwrap();
    let parsed = parse_macro_input!(prim_tokens as AddPrimitive);

    // Parse the validator expression
    let validator_expr: Expr =
        syn::parse_str(validator_part).expect("Failed to parse validator expression");

    // Build the primitive construction with validator
    build_add_primitive_impl(parsed, Some(validator_expr))
}
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
/// - Containers: you must put an `@` in front of container types.
///
/// - Specialized Constraints: using `T (E)` as a type will use
///   `(E:expr).clone()` in the type constraint but `T:ty` in the body.
///   This is necessary because the relationship between Rust types
///   and egglog sorts is not 1-to-1.
///
/// - Context: sometimes you want your primitive to reference
///   information from the scope of the `add_primitive!` call.
///   Putting `{x: T}` between the `=` and the argument list
///   will let you access the expression `x` of type `T` from inside
///   the body as `self.ctx`. `T` must be the real Rust type of `x`.
///   `T` must be `Clone` and `'static`.
#[proc_macro]
pub fn add_primitive(input: TokenStream) -> TokenStream {
    build_add_primitive_impl(parse_macro_input!(input), None)
}

fn build_add_primitive_impl(parsed: AddPrimitive, validator: Option<Expr>) -> TokenStream {
    // If you're trying to read this code, you should read the big
    // `quote!` block at the bottom of this function first. Trying
    // to parse all the intermediate gobbledygook is going to be
    // hard without that context.

    // Parse the input by implementing `syn::Parse` for our data types.
    let AddPrimitive {
        eg,
        name,
        context,
        is_varargs,
        args,
        is_fallible,
        ret,
        body,
    } = parsed;

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
                .map(|(d, u)| (quote!(#x: #d), quote!(#x: #u.clone())))
        })
        .chain(match context.0 {
            Some((e, t)) => vec![(quote!(ctx: #t), quote!(ctx: #e))],
            None => vec![],
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
            let Arg {
                x,
                t,
                is_mutable: _,
            } = &args[0];
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

    // Create the function body for `apply`.
    let apply = {
        // Bind the argument names that were passed in to this macro.
        let bind = if is_varargs {
            let x = &args[0].x;
            quote!(let #x = args.iter();)
        } else {
            let x = args.iter().map(|arg| &arg.x);
            quote!(let [#(#x),*] = args else { panic!("wrong number of arguments") };)
        };

        // Cast the arguments to the desired type.
        let cast1 = |x, t: &syn::Type, is_container| match is_container {
            false => quote!(exec_state.base_values().unwrap::<#t>(*#x)),
            true => quote!(exec_state.container_values().get_val::<#t>(*#x).unwrap().clone()),
        };
        let cast = if is_varargs {
            let Arg { x, t, is_mutable } = &args[0];
            let mutable = if *is_mutable { quote!(mut) } else { quote!() };
            match &t.cast {
                None => quote!(let #mutable #x = #x.copied();),
                Some((t, is_container)) => {
                    let cast = cast1(x, t, *is_container);
                    quote!(let #mutable #x = #x.map(|#x| #cast);)
                }
            }
        } else {
            args.iter()
                .map(|Arg { x, t, is_mutable }| {
                    let mutable = if *is_mutable { quote!(mut) } else { quote!() };
                    match &t.cast {
                        None => quote!(let #mutable #x: Value = *#x;),
                        Some((t, is_container)) => {
                            let cast = cast1(x, t, *is_container);
                            quote!(let #mutable #x: #t = #cast;)
                        }
                    }
                })
                .collect()
        };

        // If the primitive is fallible, put a `?` after the body.
        let fail = if is_fallible { quote!(?) } else { quote!() };
        // Cast the result back to an interned value.
        let (yt, ret) = match &ret.cast {
            None => (&value_type, quote!(#y)),
            Some((t, is_container)) => (
                t,
                match is_container {
                    false => quote!(exec_state.base_values().get::<#t>(#y)),
                    true => quote!(
                        exec_state.container_values().register_val::<#t>(#y, exec_state)
                    ),
                },
            ),
        };

        quote! {
            #bind
            #cast
            let #y: #yt = (#body)#fail;
            Some(#ret)
        }
    };

    // This is the big `quote!` block that ties everything together.
    let add_call = match validator {
        None => quote!(eg.add_primitive(#prim_use);),
        Some(validator_expr) => quote!(eg.add_primitive_with_validator(
            #prim_use,
            Some(::std::sync::Arc::new(#validator_expr))
        );),
    };

    quote! {{
        #[allow(unused_imports)] use ::egglog::{*, ast::*, constraint::*};
        #[allow(unused_imports)] use ::std::{any::TypeId, sync::Arc};

        #[derive(Clone)]
        #prim_def

        impl Primitive for Prim {
            fn name(&self) -> &str {
                #name
            }

            fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
                #type_constraint
            }

            fn apply(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
                #apply
            }
        }

        let eg: &mut EGraph = #eg;
        #add_call
    }}
    .into()
}

struct AddPrimitive {
    eg: Expr,
    name: LitStr,
    context: Context,
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
        let context = input.parse()?;
        let Args { is_varargs, args } = input.parse()?;
        let Arrow { is_fallible } = input.parse()?;
        let ret = input.parse()?;

        let body;
        braced!(body in input);
        let body = body.parse()?;

        Ok(AddPrimitive {
            eg,
            name,
            context,
            args,
            ret,
            body,
            is_varargs,
            is_fallible,
        })
    }
}

struct Context(Option<(Expr, syn::Type)>);

impl Parse for Context {
    fn parse(input: ParseStream) -> Result<Self> {
        if let Ok(context) = (|| {
            let context;
            braced!(context in input);
            Ok(context)
        })() {
            let e = context.parse()?;
            context.parse::<Token![:]>()?;
            let t = context.parse()?;
            Ok(Context(Some((e, t))))
        } else {
            Ok(Context(None))
        }
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
    is_mutable: bool,
}

impl Parse for Arg {
    fn parse(input: ParseStream) -> Result<Self> {
        let is_mutable = input.parse::<Token![mut]>().is_ok();
        let x = input.parse()?;
        input.parse::<Token![:]>()?;
        let t = input.parse()?;
        Ok(Arg { x, t, is_mutable })
    }
}

struct Type {
    cast: Option<(syn::Type, bool)>,
    field: Option<(syn::Type, Expr)>,
}

impl Parse for Type {
    fn parse(input: ParseStream) -> Result<Self> {
        let is_container = input.parse::<Token![@]>().is_ok();
        let cast = match input.parse::<Token![#]>() {
            Ok(_) => None,
            Err(_) => Some((input.parse()?, is_container)),
        };

        let field_def = syn::Type::Verbatim(quote!(ArcSort));

        let field = if input.peek(syn::token::Paren) {
            let inner;
            parenthesized!(inner in input);
            let field_use = inner.parse()?;

            Some((field_def, field_use))
        } else if let Some((t, _)) = &cast {
            let field_use = Expr::Verbatim(quote! {
                eg.get_arcsort_by(|s| s.value_type() == Some(TypeId::of::<#t>()))
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

/// This macro lets the user declare literal primitives with automatic validator generation.
/// It automatically generates validators by converting between Rust values and Literal types.
///
/// # Example
/// ```rust,ignore
/// add_literal_prim!(eg, "not" = |a: bool| -> bool { !a });
/// add_literal_prim!(eg, "+" = |a: i64, b: i64| -> i64 { a + b });
/// add_literal_prim!(eg, "<" = |a: i64, b: i64| -> bool { a < b });
/// ```
#[proc_macro]
pub fn add_literal_prim(input: TokenStream) -> TokenStream {
    // Parse the input using the same structure as add_primitive
    let parsed = parse_macro_input!(input as AddPrimitive);

    // Varargs not supported for literal primitives
    if parsed.is_varargs {
        return syn::Error::new_spanned(
            &parsed.name,
            "varargs not supported for literal primitives",
        )
        .to_compile_error()
        .into();
    }

    // Context not supported for literal primitives
    if parsed.context.0.is_some() {
        return syn::Error::new_spanned(
            &parsed.name,
            "context not supported for literal primitives",
        )
        .to_compile_error()
        .into();
    }

    // Check for polymorphic types
    for arg in &parsed.args {
        if arg.t.cast.is_none() {
            return syn::Error::new_spanned(
                &arg.x,
                "polymorphic types not supported for literal primitives",
            )
            .to_compile_error()
            .into();
        }
    }

    if parsed.ret.cast.is_none() {
        return syn::Error::new_spanned(
            &parsed.name,
            "polymorphic return type not supported for literal primitives",
        )
        .to_compile_error()
        .into();
    }

    // Generate the validator body
    let validator_body =
        generate_literal_validator(&parsed.args, &parsed.ret, &parsed.body, parsed.is_fallible);

    // Create the validator expression
    let validator_expr = syn::parse2::<Expr>(quote! {
        |termdag: &mut ::egglog::TermDag, args: &[::egglog::TermId]| -> Option<::egglog::TermId> {
            use egglog::termdag::Term;
            use egglog_ast::generic_ast::Literal;
            Some({
                #validator_body
            })
        }
    })
    .unwrap();

    // Use the shared implementation with the validator
    build_add_primitive_impl(parsed, Some(validator_expr))
}

// Helper function to generate literal validator that computes results
fn generate_literal_validator(
    args: &[Arg],
    ret: &Type,
    body: &Expr,
    is_fallible: bool,
) -> quote::__private::TokenStream {
    // Generate extraction code for arguments
    let arg_extracts = args.iter().enumerate().map(|(i, arg)| {
        let x = &arg.x;
        if let Some((ty, _)) = &arg.t.cast {
            quote! {
                let #x = if let Term::Lit(lit) = termdag.get(args[#i]) {
                    match <#ty as egglog::prelude::LiteralConvertible>::from_literal(&lit) {
                        Some(val) => val,
                        None => panic!("Failed to extract literal for argument {}", #i),
                    }
                } else {
                    panic!("Argument {} is not a literal", #i);
                };
            }
        } else {
            quote!(panic!("Polymorphic arguments not supported for literal primitives");)
        }
    });

    // Generate converter for return value
    let ret_conv = if let Some((ty, _)) = &ret.cast {
        quote!(<#ty as egglog::prelude::LiteralConvertible>::to_literal)
    } else {
        quote!(|_| panic!("Polymorphic types not supported for literal primitives"))
    };

    // Generate the body execution and result creation
    let body_exec = if is_fallible {
        quote! {
            let result: Option<_> = #body;
            match result {
                Some(result) => {
                    let result_lit = #ret_conv(result);
                    termdag.lit(result_lit)
                }
                None => panic!("Primitive operation failed"),
            }
        }
    } else {
        quote! {
            let result = #body;
            let result_lit = #ret_conv(result);
            termdag.lit(result_lit)
        }
    };

    // Put it all together
    quote! {
        #(#arg_extracts)*
        #body_exec
    }
}
