//! Proc macros for embedding egglog programs in Rust with compile-time
//! validation.
//!
//! See `egglog-block-macro-proposal.md` at the repo root for the full
//! design. **This crate is currently a step-2 draft** — it implements
//! validation only. Steps 3-5 (typed bindings for `(rust-rule ...)`,
//! eclass-typed bindings, AOT-skipping the runtime typecheck) are TODO.
//!
//! # What works today
//!
//! ```rust,ignore
//! use egglog::EGraph;
//! use egglog_macros::egglog;
//!
//! let mut eg = EGraph::default();
//! egglog!(eg, "
//!     (datatype Math (Num i64) (Add Math Math))
//!     (function fib (i64) i64 :no-merge)
//!     (rule ((= f0 (fib x))) ((set (fib (+ x 1)) f0)))
//! ");
//! ```
//!
//! At compile time the macro:
//! 1. Pulls the source string out of the input tokens.
//! 2. Spins up a fresh in-process `EGraph` and runs `resolve_program`
//!    (parser + typechecker) over the source.
//! 3. On success, expands to `eg.parse_and_run_program(None, "<source>")?`
//!    — same as if the user wrote the call themselves, but now with a
//!    compile-time guarantee that the egglog code is well-formed.
//! 4. On failure, emits a Rust compile error pointing at the macro
//!    invocation with the egglog typechecker's message.
//!
//! # Limitations of this draft
//!
//! - **Source has to be a string literal.** A real proc macro would
//!   accept an unquoted s-expression block:
//!   `egglog!(eg, (datatype Math (Num i64)) ...)`. That requires
//!   walking the TokenStream and re-emitting tokens-as-text, which is
//!   step 2.5 of the proposal.
//! - **No typed bindings yet.** A `(rust-rule ...)` form inside the
//!   block isn't recognized — users still write `add_rust_rule` calls
//!   outside the macro. Step 3.
//! - **The block sees a fresh, empty EGraph.** If you reference a sort
//!   or function declared elsewhere in the program (e.g. via a prior
//!   `parse_and_run_program` call), the typecheck will fail because
//!   the embedded EGraph doesn't know about it. Step 1's
//!   schema-snapshot mechanism would address this.
//! - **Pulls in the full `egglog` crate at proc-macro build time.**
//!   This is heavy. Step 1 (split `egglog-frontend`) trims it down to
//!   parser + typechecker only.

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;

/// Validate an egglog program at compile time.
///
/// Syntax: `egglog!(<egraph_expr>, <source_string_literal>)`.
///
/// The source string is parsed and typechecked at compile time using a
/// fresh `EGraph`. On success, the macro expands to a runtime call to
/// `<egraph_expr>.parse_and_run_program(None, <source>)`. On failure,
/// the egglog error becomes a Rust compile error.
///
/// # Compile-time validation in action
///
/// A typo like an unbound function name fails the typechecker at
/// compile time:
///
/// ```compile_fail
/// use egglog::EGraph;
/// use egglog_macros::egglog;
/// let mut eg = EGraph::default();
/// // `fbi` doesn't exist — should be `fib`
/// egglog!(eg, "(function fib (i64) i64 :no-merge)
///              (set (fbi 0) 0)").unwrap();
/// ```
///
/// And a type mismatch:
///
/// ```compile_fail
/// use egglog::EGraph;
/// use egglog_macros::egglog;
/// let mut eg = EGraph::default();
/// // `fib` expects an `i64`, not a `String`
/// egglog!(eg, r#"(function fib (i64) i64 :no-merge)
///                (set (fib "hello") 0)"#).unwrap();
/// ```
#[proc_macro]
pub fn egglog(input: TokenStream) -> TokenStream {
    let input2: proc_macro2::TokenStream = input.into();
    match expand(input2) {
        Ok(ts) => ts.into(),
        Err(err) => err.into_compile_error().into(),
    }
}

fn expand(input: proc_macro2::TokenStream) -> Result<proc_macro2::TokenStream, syn::Error> {
    let mut iter = input.clone().into_iter();
    let egraph_tokens = take_until_comma(&mut iter)?;
    let source_lit = take_string_literal(&mut iter)?;
    expect_eof(&mut iter)?;

    // Run the egglog parser + typechecker at compile time using a
    // fresh EGraph. We don't actually run the program — `resolve_program`
    // stops after typechecking.
    let mut eg = egglog::EGraph::default();
    if let Err(e) = eg.resolve_program(None, &source_lit) {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("egglog!: program failed to typecheck:\n{e}"),
        ));
    }

    Ok(quote! {
        (#egraph_tokens).parse_and_run_program(None, #source_lit)
    })
}

fn take_until_comma(
    iter: &mut proc_macro2::token_stream::IntoIter,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    let mut tokens = proc_macro2::TokenStream::new();
    let mut found_comma = false;
    for tok in iter.by_ref() {
        if let proc_macro2::TokenTree::Punct(p) = &tok
            && p.as_char() == ','
        {
            found_comma = true;
            break;
        }
        tokens.extend(std::iter::once(tok));
    }
    if !found_comma {
        return Err(syn::Error::new(
            Span::call_site(),
            "egglog!: expected `,` between egraph expression and source string",
        ));
    }
    Ok(tokens)
}

fn take_string_literal(
    iter: &mut proc_macro2::token_stream::IntoIter,
) -> Result<String, syn::Error> {
    let tok = iter.next().ok_or_else(|| {
        syn::Error::new(
            Span::call_site(),
            "egglog!: expected a source string literal after the comma",
        )
    })?;
    let lit: syn::LitStr = syn::parse2(quote! { #tok }).map_err(|_| {
        syn::Error::new_spanned(
            &tok,
            "egglog!: source must be a string literal in this draft (later \
             versions will accept an unquoted s-expression block — see \
             egglog-block-macro-proposal.md)",
        )
    })?;
    Ok(lit.value())
}

fn expect_eof(iter: &mut proc_macro2::token_stream::IntoIter) -> Result<(), syn::Error> {
    if let Some(tok) = iter.next() {
        return Err(syn::Error::new_spanned(
            &tok,
            "egglog!: unexpected tokens after source string",
        ));
    }
    Ok(())
}
