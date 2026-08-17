//! Compile-time-checked egglog programs.
//!
//! These macros write an egglog program directly in Rust and **check it while
//! your crate compiles** — a parse or type error in the program becomes a build
//! error at the macro call site, so you never ship a program that fails to
//! typecheck. They differ in what they hand back:
//!
//! - [`egglog_checked!`] → `Result<Vec<Command>, egglog::Error>` — the checked
//!   program as commands, for you to run into an e-graph of your choosing (or
//!   several, or to inspect).
//! - [`run_egglog_checked!`] → `Result<EGraph, egglog::Error>` — a fresh
//!   [`EGraph`](egglog::EGraph) with the program already run into it.
//!
//! For egglog split across crates, [`egglog_header!`] declares a reusable schema
//! (checked once, where it's declared) as an exported handle; then list the
//! schema names in [`egglog_checked!`] — `egglog_checked!(math, seen; <program>)`
//! — to check a fragment against them.
//!
//! ```
//! use egglog::EGraph;
//! use egglog_checked::run_egglog_checked;
//!
//! let egraph: EGraph = run_egglog_checked!(
//!     (datatype Math (Num i64) (Add Math Math))
//!     (rule ((= e (Add (Num a) (Num b)))) ((union e (Num (+ a b)))))
//!     (let start (Add (Num 1) (Num 2)))
//!     (run 1)
//!     (check (= start (Num 3)))
//! )
//! .unwrap();
//! # let _ = egraph;
//! ```
//!
//! The program is fully known at compile time, so there are no `#` splices of
//! Rust values. For programs built from runtime values, or checked against an
//! existing e-graph, use the quasiquote macros in the `egglog-quote` crate
//! (`egglog!`, `run_egglog!`, `expr!`, …).
//!
//! Checking does not run rules, so a nonterminating `(run …)` costs nothing at
//! build time. The commands handed back are unresolved and re-typecheck against
//! whatever e-graph you run them in.
//!
//! The expansions reference `::egglog::…` (and `::egglog_checked::…`) paths, so
//! the calling crate must depend on both under their default names (a renamed
//! `package = "egglog"` dependency won't resolve) — the same contract as
//! egglog's other proc macros.

use egglog_ast::tokens::atom_run;
use proc_macro::TokenStream;
use proc_macro2::{Delimiter, Ident, Span, TokenStream as TokenStream2, TokenTree};
use quote::{quote, quote_spanned};

/// Write an egglog program checked at compile time; expand to the program's
/// commands.
///
/// Returns `Result<Vec<Command>, egglog::Error>`
/// ([`Command`](egglog::ast::Command)). Expansion fails the build on parse or
/// type errors, so at run time the commands are just handed back — run them
/// into any e-graph with [`EGraph::run_program`](egglog::EGraph::run_program),
/// or inspect them. See the [crate docs](crate) for details.
///
/// ```
/// use egglog::{ast::Command, EGraph};
/// use egglog_checked::egglog_checked;
///
/// let program: Vec<Command> = egglog_checked!(
///     (datatype Math (Num i64) (Add Math Math))
///     (rule ((= e (Add (Num a) (Num b)))) ((union e (Num (+ a b)))))
/// )
/// .unwrap();
///
/// // Run the checked program into an e-graph you control.
/// let mut egraph = EGraph::default();
/// egraph.run_program(program).unwrap();
/// ```
///
/// # Against schemas from elsewhere
///
/// Prefix the program with a list of [`egglog_header!`] schema names and a `;`
/// to check it with those declarations in scope (they're in scope but not
/// returned — only the program's commands come back). This is how egglog split
/// across crates typechecks: each schema is declared once with `egglog_header!`,
/// and every fragment lists the schemas it needs.
///
/// ```
/// # use egglog::ast::Command;
/// # use egglog_checked::{egglog_header, egglog_checked};
/// egglog_header!(math (datatype Math (Num i64) (Add Math Math)));
/// egglog_header!(seen (relation seen (i64)));
///
/// // The fragment uses `Add`/`Num` (from `math`) and `seen` (from `seen`);
/// // list the schemas in dependency order.
/// let fragment: Vec<Command> = egglog_checked!(math, seen;
///     (rule ((= e (Add (Num a) (Num b)))) ((seen a)))
/// )
/// .unwrap();
/// # let _ = fragment;
/// ```
#[proc_macro]
pub fn egglog_checked(input: TokenStream) -> TokenStream {
    let input = TokenStream2::from(input);

    // Internal form emitted by the schema machinery (`egglog_header!` +
    // `egglog_checked!(a, b; …)`): `@egglog_checked [ <declarations> ] <program>`
    // checks the program with the pooled declarations in scope but returns only
    // the program's commands. Not written by hand — if you have declarations,
    // just make them part of the program.
    if let Some(rest) = strip_checked_marker(&input) {
        let (declarations, program) = match split_declarations(rest) {
            Ok(split) => split,
            Err((span, msg)) => return compile_error_at(span, &msg),
        };
        return match check_with(declarations, program) {
            Ok(program_src) => emit_commands(&program_src),
            Err((span, msg)) => compile_error_at(span, &msg),
        };
    }

    let (headers, program) = match split_header_prefix(input) {
        Ok(split) => split,
        Err((span, msg)) => return compile_error_at(span, &msg),
    };

    // `<h1>, <h2>, … ; <program>`: hand off to the first header in compose mode;
    // the header macros pool their declarations and end back here (inline form).
    if let Some((first, rest)) = headers.split_first() {
        return quote! {
            #first!( @egglog_compose {} [ #(#rest)* ] { #program } )
        }
        .into();
    }

    // `<program>`: check the whole program on its own.
    match check(program) {
        Ok(src) => emit_commands(&src),
        Err((span, msg)) => compile_error_at(span, &msg),
    }
}

/// Write an egglog program checked at compile time; expand to a fresh
/// [`EGraph`](egglog::EGraph) with the program run into it.
///
/// Returns `Result<EGraph, egglog::Error>`: expansion fails the build on parse
/// or type errors, and the `Result` reports errors that only running can raise
/// (a failed `check`, `panic`, …). See the [crate docs](crate) for details.
#[proc_macro]
pub fn run_egglog_checked(input: TokenStream) -> TokenStream {
    let src = match check(input.into()) {
        Ok(src) => src,
        Err((span, msg)) => return compile_error_at(span, &msg),
    };
    // Runtime: build a fresh e-graph and run the (already-checked) program,
    // handing back the populated e-graph.
    quote! {{
        let mut __egraph = ::egglog::EGraph::default();
        ::egglog::EGraph::parse_and_run_program(
            &mut __egraph,
            ::core::option::Option::None,
            #src,
        )
        .map(move |_| __egraph)
    }}
    .into()
}

/// Define a reusable, compile-time-checked egglog *header* (schema).
///
/// `egglog_header!(<name> <declarations>)` typechecks `<declarations>` (sorts,
/// constructors, functions, …) **here** — so a broken schema is a build error at
/// this call, not at each use — then generates a schema *handle* macro `<name>`.
/// Pass that name to [`egglog_checked!`] to check a program against the schema:
/// `egglog_checked!(<name>; <program>)`, or list several with
/// `egglog_checked!(a, b; <program>)`. You don't call `<name>!` directly.
///
/// The handle is `#[macro_export]`ed, so a schema can be declared in one crate
/// and used from others (`use my_schema_crate::math;`). Crates that use it need
/// `egglog-checked` and `egglog` as dependencies, under those default names.
///
/// ```
/// use egglog::ast::Command;
/// use egglog_checked::{egglog_header, egglog_checked};
///
/// // Declare the schema once (checked right here).
/// egglog_header!(math
///     (datatype Math (Num i64) (Add Math Math))
///     (function lower (Math) i64 :no-merge)
/// );
///
/// // Check fragments against it via `egglog_checked!`.
/// let fragment: Vec<Command> = egglog_checked!(math;
///     (rule ((= e (Add (Num a) (Num b)))) ((union e (Num (+ a b)))))
/// )
/// .unwrap();
/// # let _ = fragment;
/// ```
#[proc_macro]
pub fn egglog_header(input: TokenStream) -> TokenStream {
    let mut it = TokenStream2::from(input).into_iter();
    let name = match it.next() {
        Some(TokenTree::Ident(id)) => id,
        other => {
            let span = other.map(|t| t.span()).unwrap_or_else(Span::call_site);
            return compile_error_at(
                span,
                "egglog_header! expects a macro name first, e.g. \
                 `egglog_header!(math (datatype …))`",
            );
        }
    };
    let declarations: TokenStream2 = it.collect();

    // Typecheck the declarations here, so a broken schema errors at the header.
    let mut src = String::new();
    let mut segments = Vec::new();
    if let Err((span, msg)) = render(declarations.clone(), &mut src, &mut segments) {
        return compile_error_at(span, &msg);
    }
    if let Err((span, msg)) = typecheck(&src, &segments) {
        return compile_error_at(span, &msg);
    }

    // The handle is driven by `egglog_checked!(a, b, …; …)`: its `@egglog_compose`
    // arms pool each schema's declarations into an accumulator, then hand off to
    // the next schema, finally landing in `egglog_checked!([ pooled ] program)`.
    // Calling the handle directly is a mistake, so the last arm says so. (`$…`
    // below is emitted verbatim into the generated `macro_rules!`.)
    let doc = format!(
        "egglog schema handle generated by `egglog_header!`. Pass it to \
         `egglog_checked!({name}; <program>)` to check a program against this schema."
    );
    let direct_use = format!(
        "`{name}!` is an egglog schema handle — use `egglog_checked!({name}; <program>)` \
         to check a program against it"
    );
    // Alongside the checking macro, emit `<name>_schema()` returning the schema's
    // own commands, so the schema can be *run* into an e-graph from a single
    // source of truth (the header) rather than being restated.
    let schema_fn = Ident::new(&format!("{name}_schema"), name.span());
    let schema_doc = format!(
        "The declarations of the `{name}` egglog schema, as commands to run into \
         an e-graph. Pair with `egglog_checked!({name}; …)`-checked fragments."
    );
    quote! {
        #[doc = #doc]
        #[macro_export]
        macro_rules! #name {
            (@egglog_compose { $($__acc:tt)* } [] { $($__prog:tt)* }) => {
                ::egglog_checked::egglog_checked!( @egglog_checked [ $($__acc)* #declarations ] $($__prog)* )
            };
            (@egglog_compose { $($__acc:tt)* } [ $__next:ident $($__rest:ident)* ] { $($__prog:tt)* }) => {
                $__next!( @egglog_compose { $($__acc)* #declarations } [ $($__rest)* ] { $($__prog)* } )
            };
            ($($__other:tt)*) => {
                ::core::compile_error!(#direct_use)
            };
        }

        #[doc = #schema_doc]
        #[allow(dead_code)]
        pub fn #schema_fn()
        -> ::core::result::Result<::std::vec::Vec<::egglog::ast::Command>, ::egglog::Error> {
            let mut __parser = ::egglog::ast::Parser::default();
            ::egglog::ast::Parser::get_program_from_string(
                &mut __parser,
                ::core::option::Option::None,
                #src,
            )
            .map_err(::egglog::Error::from)
        }
    }
    .into()
}

/// A rendered atom or list: the byte range it occupies in the source string,
/// and the `proc_macro2` span of the token(s) it came from. Used to point a
/// diagnostic at the offending token when egglog reports an error at a byte
/// offset.
struct Segment {
    start: usize,
    end: usize,
    span: Span,
}

/// A diagnostic: a span to point at and a message. Converted to a
/// `compile_error!` only at the proc-macro boundary, so the checking logic
/// stays unit-testable.
type Fail = (Span, String);

/// Render the macro input to egglog source and typecheck it at expansion time,
/// returning the source to embed in the expansion.
fn check(input: TokenStream2) -> Result<String, Fail> {
    let mut src = String::new();
    let mut segments = Vec::new();
    render(input, &mut src, &mut segments)?;
    typecheck(&src, &segments)?;
    Ok(src)
}

/// Like [`check`], but the program is typechecked with `declarations` (a schema
/// — sorts, constructors, functions, …) in scope. Only the program's source is
/// returned; the declarations aren't, so a fragment in one crate can be checked
/// against types declared in another without re-emitting them.
fn check_with(declarations: TokenStream2, program: TokenStream2) -> Result<String, Fail> {
    let mut src = String::new();
    let mut segments = Vec::new();
    render(declarations, &mut src, &mut segments)?;
    src.push('\n');
    let program_start = src.len();
    render(program, &mut src, &mut segments)?;
    typecheck(&src, &segments)?;
    Ok(src[program_start..].to_string())
}

/// Parse + typecheck `src` in a throwaway e-graph (registering declarations but
/// running nothing). On failure, points at the offending token, falling back to
/// the call site if the error has no usable location.
fn typecheck(src: &str, segments: &[Segment]) -> Result<(), Fail> {
    let mut egraph = egglog::EGraph::default();
    if let Err(e) = egraph.resolve_program(None, src) {
        let span = error_span(&e, segments).unwrap_or_else(Span::call_site);
        return Err((span, format!("egglog_checked!: {e}")));
    }
    Ok(())
}

/// Split an optional leading `h1, h2, … ;` header-name list from the program
/// that follows. With no top-level `;`, there are no headers and the whole
/// input is the program.
fn split_header_prefix(input: TokenStream2) -> Result<(Vec<Ident>, TokenStream2), Fail> {
    let has_semi = input
        .clone()
        .into_iter()
        .any(|tt| matches!(&tt, TokenTree::Punct(p) if p.as_char() == ';'));
    if !has_semi {
        return Ok((Vec::new(), input));
    }
    let mut headers = Vec::new();
    let mut it = input.into_iter();
    for tt in it.by_ref() {
        match tt {
            TokenTree::Punct(ref p) if p.as_char() == ';' => break,
            TokenTree::Punct(ref p) if p.as_char() == ',' => {}
            TokenTree::Ident(id) => headers.push(id),
            other => {
                return Err((
                    other.span(),
                    "expected header names (identifiers) before `;`, e.g. \
                     `egglog_checked!(math, seen; <program>)`"
                        .to_string(),
                ));
            }
        }
    }
    Ok((headers, it.collect()))
}

/// If the input begins with the internal `@egglog_checked` marker, return the
/// tokens after it; otherwise `None`.
fn strip_checked_marker(input: &TokenStream2) -> Option<TokenStream2> {
    let mut it = input.clone().into_iter();
    match (it.next(), it.next()) {
        (Some(TokenTree::Punct(p)), Some(TokenTree::Ident(id)))
            if p.as_char() == '@' && id == "egglog_checked" =>
        {
            Some(it.collect())
        }
        _ => None,
    }
}

/// Expand to `Result<Vec<Command>, egglog::Error>` that re-parses `src` (already
/// checked at expansion time) into unresolved commands.
fn emit_commands(src: &str) -> TokenStream {
    quote! {{
        let mut __parser = ::egglog::ast::Parser::default();
        ::egglog::ast::Parser::get_program_from_string(
            &mut __parser,
            ::core::option::Option::None,
            #src,
        )
        .map_err(::egglog::Error::from)
    }}
    .into()
}

/// Split a leading `[ … ]` declarations group from the program that follows
/// (the internal `@egglog_checked` form).
fn split_declarations(input: TokenStream2) -> Result<(TokenStream2, TokenStream2), Fail> {
    let mut it = input.into_iter();
    match it.next() {
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Bracket => {
            Ok((g.stream(), it.collect()))
        }
        other => {
            let span = other.map(|t| t.span()).unwrap_or_else(Span::call_site);
            Err((span, "expected a `[ … ]` declarations group".to_string()))
        }
    }
}

/// Map an egglog error back to the `proc_macro2` span of the token it points
/// at, via the byte offsets it carries into the rendered source.
fn error_span(err: &egglog::Error, segments: &[Segment]) -> Option<Span> {
    match err.span()? {
        egglog::ast::Span::Egglog(s) => map_offset(segments, s.i, s.j),
        _ => None,
    }
}

/// Find the token span for the source byte range `[i, j)`: the smallest segment
/// containing the start offset, else the smallest one overlapping the range.
fn map_offset(segments: &[Segment], i: usize, j: usize) -> Option<Span> {
    segments
        .iter()
        .filter(|s| s.start <= i && i < s.end)
        .min_by_key(|s| s.end - s.start)
        .or_else(|| {
            segments
                .iter()
                .filter(|s| s.start < j && i < s.end)
                .min_by_key(|s| s.end - s.start)
        })
        .map(|s| s.span)
}

/// Emit a `compile_error!` pointing at `span`. Brace-delimited so it's valid in
/// both expression position (the `egglog_checked!` family) and item position
/// (`egglog_header!`, which expands to a `macro_rules!` item).
fn compile_error_at(span: Span, msg: &str) -> TokenStream {
    quote_spanned! { span => compile_error! { #msg } }.into()
}

/// Render a token stream as egglog source text. Lists become parenthesized;
/// every other run of directly-adjacent tokens becomes one atom (mirroring
/// egglog's tokenizer), so `my-ruleset`, `:no-merge`, and `-1.0` survive as
/// single atoms while space-separated tokens stay apart. A `#` (splice) is
/// rejected — a compile-time program has no runtime values to splice.
fn render(ts: TokenStream2, out: &mut String, segments: &mut Vec<Segment>) -> Result<(), Fail> {
    let mut it = ts.into_iter().peekable();
    let mut first = true;
    while let Some(tt) = it.next() {
        if !first {
            out.push(' ');
        }
        first = false;
        match tt {
            TokenTree::Group(g) => match g.delimiter() {
                // A transparent (macro-metavariable) group: render inline.
                Delimiter::None => render(g.stream(), out, segments)?,
                _ => {
                    let start = out.len();
                    out.push('(');
                    render(g.stream(), out, segments)?;
                    out.push(')');
                    segments.push(Segment {
                        start,
                        end: out.len(),
                        span: g.span(),
                    });
                }
            },
            TokenTree::Punct(ref p) if p.as_char() == '#' => {
                return Err((
                    p.span(),
                    "egglog_checked! does not support `#` splices — the program \
                     must be fully known at compile time"
                        .to_string(),
                ));
            }
            // A double-quoted string is its own atom.
            TokenTree::Literal(ref lit) if lit.to_string().starts_with('"') => {
                let start = out.len();
                out.push_str(&lit.to_string());
                segments.push(Segment {
                    start,
                    end: out.len(),
                    span: lit.span(),
                });
            }
            first_tt => {
                let start = out.len();
                let (atom, span) = atom_run(first_tt, &mut it);
                out.push_str(&atom);
                segments.push(Segment {
                    start,
                    end: out.len(),
                    span,
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(s: &str) -> TokenStream2 {
        s.parse().unwrap()
    }

    fn render_str(input: &str) -> Result<String, Fail> {
        let mut src = String::new();
        let mut segments = Vec::new();
        render(ts(input), &mut src, &mut segments)?;
        Ok(src)
    }

    #[test]
    fn render_glues_adjacent_tokens_into_egglog_atoms() {
        // `?x`, `:no-merge`, `my-ruleset`, and `-1` must survive as single atoms.
        assert_eq!(render_str("(f ?x)").unwrap(), "(f ?x)");
        assert_eq!(
            render_str("(function f (i64) i64 :no-merge)").unwrap(),
            "(function f (i64) i64 :no-merge)"
        );
        assert_eq!(
            render_str("(ruleset my-ruleset)").unwrap(),
            "(ruleset my-ruleset)"
        );
        assert_eq!(render_str("(set (f 0) -1)").unwrap(), "(set (f 0) -1)");
    }

    #[test]
    fn render_rejects_splices() {
        let (_span, msg) = render_str("(Num #x)").unwrap_err();
        assert!(msg.contains("splice"), "{msg}");
    }

    #[test]
    fn check_accepts_valid_and_reports_type_errors() {
        assert!(check(ts("(datatype Math (Num i64))")).is_ok());

        let (_span, msg) = check(ts("(relation r (Nonexistent))")).unwrap_err();
        assert!(msg.contains("Undefined sort Nonexistent"), "{msg}");
    }

    #[test]
    fn check_with_uses_declarations_in_scope() {
        let decls = ts("(datatype Math (Num i64) (Add Math Math))");

        let ok = ts("(rule ((= e (Add (Num a) (Num b)))) ((union e (Num (+ a b)))))");
        assert!(check_with(decls.clone(), ok).is_ok());

        // `Mul` is in neither the declarations nor the fragment -> error.
        let bad = ts("(rule ((= e (Add (Num a) (Num b)))) ((union e (Mul (Num 1) (Num 2)))))");
        let (_span, msg) = check_with(decls, bad).unwrap_err();
        assert!(msg.contains("Mul"), "{msg}");
    }

    #[test]
    fn header_prefix_splits_on_semicolon() {
        let (headers, program) = split_header_prefix(ts("a, b, c; (run 1)")).unwrap();
        let names: Vec<_> = headers.iter().map(|i| i.to_string()).collect();
        assert_eq!(names, ["a", "b", "c"]);
        assert!(!program.is_empty());

        // No top-level `;` -> no headers, whole input is the program.
        let (none, _) = split_header_prefix(ts("(datatype Math (Num i64))")).unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn checked_marker_is_recognized() {
        assert!(strip_checked_marker(&ts("@egglog_checked [x] (y)")).is_some());
        assert!(strip_checked_marker(&ts("(datatype Math (Num i64))")).is_none());
    }

    #[test]
    fn error_offset_maps_to_the_innermost_token() {
        // Render (no reformatting here), then map a source byte offset back to a
        // token span: offset of `y` should select the `y` atom, not the list.
        let mut src = String::new();
        let mut segments = Vec::new();
        render(ts("(Add x y)"), &mut src, &mut segments).unwrap();
        assert_eq!(src, "(Add x y)");

        let y = src.find('y').unwrap();
        let span = map_offset(&segments, y, y + 1).expect("offset should map to a token");
        // `src` equals the input here, so the byte offset is the token's column.
        assert_eq!(span.start().column, y);
    }
}
