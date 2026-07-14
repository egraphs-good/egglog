//! Compile-time-checked egglog programs.
//!
//! Both macros write an egglog program directly in Rust and **check it while
//! your crate compiles** — a parse or type error in the program becomes a build
//! error at the macro call site, so you never ship a program that fails to
//! typecheck. They differ in what they hand back:
//!
//! - [`egglog_static!`] → `Result<Vec<Command>, egglog::Error>` — the checked
//!   program as commands, for you to run into an e-graph of your choosing (or
//!   several, or to inspect).
//! - [`run_egglog_static!`] → `Result<EGraph, egglog::Error>` — a fresh
//!   [`EGraph`](egglog::EGraph) with the program already run into it.
//!
//! ```
//! use egglog::EGraph;
//! use egglog_static::run_egglog_static;
//!
//! let egraph: EGraph = run_egglog_static!(
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
//! Checking runs egglog's own parser and typechecker at macro-expansion time:
//! declarations are registered and every form is resolved, but rules are **not**
//! run, so a nonterminating `(run …)` costs nothing at build time. Because a
//! resolved program holds sort handles bound to the e-graph it was checked in,
//! the check is a build-time gate only — the commands handed back are
//! *unresolved* and re-typecheck against whatever e-graph you run them in.

use egglog_ast::tokens::atom_run;
use proc_macro::TokenStream;
use proc_macro2::{Delimiter, Span, TokenStream as TokenStream2, TokenTree};
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
/// use egglog_static::egglog_static;
///
/// let program: Vec<Command> = egglog_static!(
///     (datatype Math (Num i64) (Add Math Math))
///     (rule ((= e (Add (Num a) (Num b)))) ((union e (Num (+ a b)))))
/// )
/// .unwrap();
///
/// // Run the checked program into an e-graph you control.
/// let mut egraph = EGraph::default();
/// egraph.run_program(program).unwrap();
/// ```
#[proc_macro]
pub fn egglog_static(input: TokenStream) -> TokenStream {
    let src = match check(input) {
        Ok(src) => src,
        Err(err) => return err,
    };
    // Runtime: re-parse the (already-checked) source into unresolved commands.
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

/// Write an egglog program checked at compile time; expand to a fresh
/// [`EGraph`](egglog::EGraph) with the program run into it.
///
/// Returns `Result<EGraph, egglog::Error>`: expansion fails the build on parse
/// or type errors, and the `Result` reports errors that only running can raise
/// (a failed `check`, `panic`, …). See the [crate docs](crate) for details.
#[proc_macro]
pub fn run_egglog_static(input: TokenStream) -> TokenStream {
    let src = match check(input) {
        Ok(src) => src,
        Err(err) => return err,
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

/// A rendered atom or list: the byte range it occupies in the source string,
/// and the `proc_macro2` span of the token(s) it came from. Used to point a
/// diagnostic at the offending token when egglog reports an error at a byte
/// offset.
struct Segment {
    start: usize,
    end: usize,
    span: Span,
}

/// Render the macro input to egglog source and typecheck it at expansion time.
/// On success returns the source (to embed in the expansion); on failure
/// returns a `compile_error!` token stream to emit in place.
fn check(input: TokenStream) -> Result<String, TokenStream> {
    let mut src = String::new();
    let mut segments = Vec::new();
    if let Err((span, msg)) = render(input.into(), &mut src, &mut segments) {
        return Err(compile_error_at(span, &msg));
    }
    // Parse + typecheck in a throwaway e-graph, registering declarations but
    // running nothing.
    let mut egraph = egglog::EGraph::default();
    if let Err(e) = egraph.resolve_program(None, &src) {
        // Point the diagnostic at the offending token, falling back to the
        // whole call site if the error carries no usable location.
        let span = error_span(&e, &segments).unwrap_or_else(Span::call_site);
        return Err(compile_error_at(span, &format!("egglog_static!: {e}")));
    }
    Ok(src)
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

/// Emit a `compile_error!` pointing at `span`.
fn compile_error_at(span: Span, msg: &str) -> TokenStream {
    quote_spanned! { span => compile_error!(#msg) }.into()
}

/// Render a token stream as egglog source text. Lists become parenthesized;
/// every other run of directly-adjacent tokens becomes one atom (mirroring
/// egglog's tokenizer), so `my-ruleset`, `:no-merge`, and `-1.0` survive as
/// single atoms while space-separated tokens stay apart. A `#` (splice) is
/// rejected — a compile-time program has no runtime values to splice.
fn render(
    ts: TokenStream2,
    out: &mut String,
    segments: &mut Vec<Segment>,
) -> Result<(), (Span, String)> {
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
                    "egglog_static! does not support `#` splices — the program \
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
