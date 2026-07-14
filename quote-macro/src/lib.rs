//! Procedural macros backing egglog's quasiquotes — `expr!`, `query!`,
//! `command!`, `egglog!`, `rule!`, `action!`/`actions!`, `sexp!`/`sexps!`, and
//! their `resolve_*!` / `run_*!` variants.
//!
//! These are re-exported from the `egglog` crate; the user-facing guide — what
//! each macro produces, the `#` / `#..` / `:#` splice forms, and examples —
//! lives in `egglog::prelude`. The doc on each macro below notes its signature.

use proc_macro::TokenStream;
use proc_macro2::{Delimiter, TokenStream as TokenStream2, TokenTree};
use quote::quote;
use std::iter::Peekable;

/// Parse one egglog expression: `expr!([parser,] <egglog>)` → `Result<Expr, ParseError>`.
///
/// ```ignore
/// let x = expr!((+ ?a 1))?;                 // default parser
/// let y = expr!(my_parser, (Mul :shape ?s ...))?;
/// ```
/// See `egglog::prelude` for the parser argument, the `#` / `#..` / `:#` splices, and the `resolve_*!` / `run_*!` variants.
#[proc_macro]
pub fn expr(input: TokenStream) -> TokenStream {
    build(input, Method::Expr)
}

/// Resolve one expression against an e-graph: `resolve_expr!(egraph, <expr>)` →
/// `Result<ResolvedExpr, Error>` (typecheck, free names = globals, no eval).
#[proc_macro]
pub fn resolve_expr(input: TokenStream) -> TokenStream {
    build(input, Method::ResolveExpr)
}

/// Evaluate one expression against an e-graph: `run_expr!(egraph, <expr>)` →
/// `Result<(ArcSort, Value), Error>` (via `eval_expr`).
#[proc_macro]
pub fn run_expr(input: TokenStream) -> TokenStream {
    build(input, Method::RunExpr)
}

/// Parse a query — a sequence of facts (query atoms):
/// `query!([parser,] <fact>*)` → `Result<Facts, ParseError>`.
#[proc_macro]
pub fn query(input: TokenStream) -> TokenStream {
    build(input, Method::Facts)
}

/// Resolve a query against an e-graph: `resolve_query!(egraph, <fact>*)` →
/// `Result<Vec<ResolvedFact>, Error>` (typecheck the query body, no run).
#[proc_macro]
pub fn resolve_query(input: TokenStream) -> TokenStream {
    build(input, Method::ResolveQuery)
}

/// Run a query against an e-graph: `run_query!(egraph, <fact>*)` →
/// `Result<Vec<HashMap<String, Value>>, Error>` — one map (var name → value)
/// per match. Query variables (and their sorts) are derived from the facts, so
/// no explicit `vars` list is needed.
#[proc_macro]
pub fn run_query(input: TokenStream) -> TokenStream {
    build(input, Method::RunQuery)
}

/// Parse one egglog action: `action!([parser,] <egglog action>)` → `Result<Vec<Action>, ParseError>`.
#[proc_macro]
pub fn action(input: TokenStream) -> TokenStream {
    build(input, Method::Action)
}

/// Parse a sequence of actions: `actions!([parser,] <egglog action>*)` → `Result<Actions, ParseError>`.
#[proc_macro]
pub fn actions(input: TokenStream) -> TokenStream {
    build(input, Method::Actions)
}

/// Resolve one action against an e-graph, as a top-level action command:
/// `resolve_action!(egraph, <action>)` → `Result<Vec<ResolvedCommand>, Error>`.
#[proc_macro]
pub fn resolve_action(input: TokenStream) -> TokenStream {
    build(input, Method::ResolveCommand)
}

/// Run one action against an e-graph, as a top-level action command:
/// `run_action!(egraph, <action>)` → `Result<Vec<CommandOutput>, Error>`.
#[proc_macro]
pub fn run_action(input: TokenStream) -> TokenStream {
    build(input, Method::RunCommand)
}

/// Resolve a sequence of actions against an e-graph, as top-level action
/// commands: `resolve_actions!(egraph, <action>*)` → `Result<Vec<ResolvedCommand>, Error>`.
#[proc_macro]
pub fn resolve_actions(input: TokenStream) -> TokenStream {
    build(input, Method::ResolveProgram)
}

/// Run a sequence of actions against an e-graph, as top-level action commands:
/// `run_actions!(egraph, <action>*)` → `Result<Vec<CommandOutput>, Error>`.
#[proc_macro]
pub fn run_actions(input: TokenStream) -> TokenStream {
    build(input, Method::RunProgram)
}

/// Parse one egglog command: `command!([parser,] <egglog command>)` → `Result<Vec<Command>, ParseError>`.
///
/// Returns a `Vec` because a single surface command may desugar into several.
#[proc_macro]
pub fn command(input: TokenStream) -> TokenStream {
    build(input, Method::Command)
}

/// Resolve one command against an e-graph: `resolve_command!(egraph, <command>)`
/// → `Result<Vec<ResolvedCommand>, Error>` (typecheck, no execution).
#[proc_macro]
pub fn resolve_command(input: TokenStream) -> TokenStream {
    build(input, Method::ResolveCommand)
}

/// Run one command against an e-graph: `run_command!(egraph, <command>)` →
/// `Result<Vec<CommandOutput>, Error>`.
#[proc_macro]
pub fn run_command(input: TokenStream) -> TokenStream {
    build(input, Method::RunCommand)
}

/// Parse a whole program: `egglog!([parser,] <egglog command>*)` → `Result<Vec<Command>, ParseError>`.
///
/// The flagship macro — a sequence of commands, ready to run with
/// `egraph.run_program(egglog!(..)?)`.
///
/// ```ignore
/// egraph.run_program(egglog!(
///     (datatype Math (Num i64) (Add Math Math))
///     (rule ((= e (Add (Num a) (Num b)))) ((union e (Num (+ a b)))))
/// )?)?;
/// ```
/// See `egglog::prelude` for the parser argument, the `#` / `#..` / `:#` splices, and the `resolve_*!` / `run_*!` variants.
#[proc_macro]
pub fn egglog(input: TokenStream) -> TokenStream {
    build(input, Method::Program)
}

/// Resolve a program against an e-graph: `resolve_egglog!(egraph, <commands>)`
/// → `Result<Vec<ResolvedCommand>, Error>`. Parses with the e-graph's parser
/// and typechecks (so you get type errors here), but **does not run** anything.
///
/// ```ignore
/// let resolved = resolve_egglog!(egraph, (rule ((= x (Foo)))((delete x))))?;
/// ```
#[proc_macro]
pub fn resolve_egglog(input: TokenStream) -> TokenStream {
    build(input, Method::ResolveProgram)
}

/// Run a program against an e-graph: `run_egglog!(egraph, <commands>)` →
/// `Result<Vec<CommandOutput>, Error>`. Parses with the e-graph's parser,
/// resolves, and executes — i.e. `egraph.run_program(egglog!(..)?)` with the
/// parse using the e-graph's own parser.
///
/// ```ignore
/// run_egglog!(egraph, (datatype Math (Num i64) (Add Math Math)))?;
/// ```
#[proc_macro]
pub fn run_egglog(input: TokenStream) -> TokenStream {
    build(input, Method::RunProgram)
}

/// Build one rule command: `rule!([parser,] (<facts>) (<actions>))` → `Result<Vec<Command>, ParseError>`.
///
/// Sugar for the `(rule (<facts>) (<actions>))` command — the two groups are
/// the body (facts) and head (actions).
#[proc_macro]
pub fn rule(input: TokenStream) -> TokenStream {
    build(input, Method::Rule)
}

/// Resolve a rule against an e-graph: `resolve_rule!(egraph, (<facts>) (<actions>))`
/// → `Result<Vec<ResolvedCommand>, Error>` (typecheck, no execution).
#[proc_macro]
pub fn resolve_rule(input: TokenStream) -> TokenStream {
    build(input, Method::ResolveRule)
}

/// Run a rule against an e-graph: `run_rule!(egraph, (<facts>) (<actions>))` →
/// `Result<Vec<CommandOutput>, Error>`.
#[proc_macro]
pub fn run_rule(input: TokenStream) -> TokenStream {
    build(input, Method::RunRule)
}

/// Build an *un-parsed* s-expression: `sexp!(<egglog>)` → `Sexp` (no parser
/// argument). Supports `#` / `#..` / `:#` splices, and is meant to be
/// `#`-spliced into another quasiquote later (where a `Sexp` is used as-is).
///
/// ```ignore
/// let kind = "MyOp";
/// let fields = vec!["?a", "?b"];
/// let frag = sexp!((#kind #..fields));   // Sexp `(MyOp ?a ?b)`
/// let e = expr!((wrap #frag 7))?;        // Expr `(wrap (MyOp ?a ?b) 7)`
/// ```
#[proc_macro]
pub fn sexp(input: TokenStream) -> TokenStream {
    build(input, Method::Sexp)
}

/// Build a sequence of un-parsed s-expressions: `sexps!(<egglog>*)` → `Vec<Sexp>`.
///
/// Like [`sexp`] but for many forms at once — handy as the operand of a `#..`
/// spread.
#[proc_macro]
pub fn sexps(input: TokenStream) -> TokenStream {
    build(input, Method::Sexps)
}

#[derive(Clone, Copy)]
enum Method {
    Expr,
    Command,
    Action,
    Facts,
    Actions,
    Program,
    Rule,
    /// Build a `Sexp` (or `Vec<Sexp>`) without parsing — for splicing.
    Sexp,
    Sexps,
    /// Egraph-context command-family variants: parse the body with the
    /// e-graph's own parser into a `Vec<Command>`, then resolve (→
    /// `Vec<ResolvedCommand>`) or run (→ `Vec<CommandOutput>`) against it.
    ResolveProgram,
    RunProgram,
    ResolveCommand,
    RunCommand,
    ResolveRule,
    RunRule,
    /// Egraph-context expression variants: parse one expression, then resolve
    /// (→ `ResolvedExpr`, no eval) or run (→ `(ArcSort, Value)` via `eval_expr`).
    ResolveExpr,
    RunExpr,
    /// Egraph-context query variants: parse the body as facts, then resolve (→
    /// `Vec<ResolvedFact>`) or run (→ query matches) against the e-graph.
    ResolveQuery,
    RunQuery,
}

fn build(input: TokenStream, method: Method) -> TokenStream {
    // `sexp!`/`sexps!` build a `Sexp` without parsing, so they take no parser
    // and just return the assembled value(s).
    if let Method::Sexp | Method::Sexps = method {
        let items = build_items(&sexp_seq(input.into()));
        let out = match method {
            Method::Sexp => quote! {{
                let __span = ::egglog::span!();
                let __sexps: ::std::vec::Vec<::egglog::ast::Sexp> = #items;
                assert_eq!(__sexps.len(), 1, "sexp! expects exactly one form (use sexps! for many)");
                __sexps.into_iter().next().unwrap()
            }},
            _ => quote! {{
                let __span = ::egglog::span!();
                let __sexps: ::std::vec::Vec<::egglog::ast::Sexp> = #items;
                __sexps
            }},
        };
        return out.into();
    }

    // Command-family egraph-context variants (`resolve_egglog!`/`run_egglog!`,
    // `resolve_command!`/`run_command!`, `resolve_rule!`/`run_rule!`): a
    // REQUIRED e-graph, then the egglog body. Parse the body with the e-graph's
    // own parser (so its registered sorts/macros are in scope) into a
    // `Vec<Command>`, then resolve (→ `Vec<ResolvedCommand>`, no execution) or
    // run (→ `Vec<CommandOutput>`) against it.
    if let Method::ResolveProgram
    | Method::RunProgram
    | Method::ResolveCommand
    | Method::RunCommand
    | Method::ResolveRule
    | Method::RunRule = method
    {
        let (ctx, body) = split_parser(input.into());
        let Some(ctx) = ctx else {
            return quote!(compile_error!(
                "this macro needs an e-graph first, e.g. `run_egglog!(egraph, ..)` / `resolve_egglog!(egraph, ..)`"
            ))
            .into();
        };
        let items = build_items(&sexp_seq(body));
        // How the parsed forms become a `Vec<Command>` (mirrors the parse-only
        // variants): a whole program, a single command, or a `(rule …)` wrap.
        let cmds_build = match method {
            Method::ResolveCommand | Method::RunCommand => quote! {
                assert_eq!(__sexps.len(), 1, "this macro expects exactly one command");
                __cmds.extend(__eg.parser.parse_command(&__sexps[0])?);
            },
            Method::ResolveRule | Method::RunRule => quote! {
                assert_eq!(__sexps.len(), 2, "rule macros expect `(<facts>) (<actions>)`");
                let __r = ::egglog::ast::Sexp::List(
                    ::std::vec![
                        ::egglog::ast::Sexp::Atom("rule".to_string(), __span.clone()),
                        __sexps[0].clone(),
                        __sexps[1].clone(),
                    ],
                    __span.clone(),
                );
                __cmds.extend(__eg.parser.parse_command(&__r)?);
            },
            // Program (`egglog!`): every form is a command.
            _ => quote! {
                for __s in &__sexps {
                    __cmds.extend(__eg.parser.parse_command(__s)?);
                }
            },
        };
        let finish = match method {
            Method::ResolveProgram | Method::ResolveCommand | Method::ResolveRule => {
                quote!(__eg.resolve_commands(__cmds))
            }
            _ => quote!(__eg.run_program(__cmds)),
        };
        return quote! {{
            let __eg = &mut (#ctx);
            let __span = ::egglog::span!();
            let __sexps: ::std::vec::Vec<::egglog::ast::Sexp> = #items;
            (|| -> ::std::result::Result<_, ::egglog::Error> {
                let mut __cmds: ::std::vec::Vec<::egglog::ast::Command> = ::std::vec::Vec::new();
                #cmds_build
                #finish
            })()
        }}
        .into();
    }

    // Egraph-context expression variants (`resolve_expr!` / `run_expr!`): parse
    // one expression with the e-graph's parser, then resolve it to a
    // `ResolvedExpr` (no eval) or evaluate it to `(ArcSort, Value)`.
    if let Method::ResolveExpr | Method::RunExpr = method {
        let (ctx, body) = split_parser(input.into());
        let Some(ctx) = ctx else {
            return quote!(compile_error!(
                "this macro needs an e-graph first, e.g. `run_expr!(egraph, ..)` / `resolve_expr!(egraph, ..)`"
            ))
            .into();
        };
        let items = build_items(&sexp_seq(body));
        let op = match method {
            Method::ResolveExpr => quote!(__eg.resolve_expr(&__e)),
            _ => quote!(__eg.eval_expr(&__e)),
        };
        return quote! {{
            let __eg = &mut (#ctx);
            let __span = ::egglog::span!();
            let __sexps: ::std::vec::Vec<::egglog::ast::Sexp> = #items;
            (|| -> ::std::result::Result<_, ::egglog::Error> {
                assert_eq!(__sexps.len(), 1, "expr macros expect exactly one expression");
                let __e = __eg.parser.parse_expr(&__sexps[0])?;
                #op
            })()
        }}
        .into();
    }

    // Egraph-context query variants (`resolve_query!` / `run_query!`): parse the
    // body as facts with the e-graph's parser, then resolve them to
    // `Vec<ResolvedFact>` (no run) or run the query and return the matches.
    if let Method::ResolveQuery | Method::RunQuery = method {
        let (ctx, body) = split_parser(input.into());
        let Some(ctx) = ctx else {
            return quote!(compile_error!(
                "this macro needs an e-graph first, e.g. `run_query!(egraph, ..)` / `resolve_query!(egraph, ..)`"
            ))
            .into();
        };
        let items = build_items(&sexp_seq(body));
        let op = match method {
            Method::ResolveQuery => quote!(__eg.resolve_facts(&__facts)),
            _ => quote!(__eg.query_all(::egglog::ast::Facts(__facts))),
        };
        return quote! {{
            let __eg = &mut (#ctx);
            let __span = ::egglog::span!();
            let __sexps: ::std::vec::Vec<::egglog::ast::Sexp> = #items;
            (|| -> ::std::result::Result<_, ::egglog::Error> {
                let mut __facts: ::std::vec::Vec<::egglog::ast::Fact> = ::std::vec::Vec::new();
                for __s in &__sexps {
                    __facts.push(__eg.parser.parse_fact(__s)?);
                }
                #op
            })()
        }}
        .into();
    }

    let (parser_opt, body) = split_parser(input.into());
    // With an explicit parser (which may have registered macros — named args,
    // `for`, …) use it; otherwise a fresh default parser handles built-in syntax.
    let parser_decl = match parser_opt {
        Some(p) => quote!(let __parser = &mut (#p);),
        None => quote!(
            #[allow(unused_mut)]
            let mut __parser = ::egglog::ast::Parser::default();
        ),
    };
    let items = build_items(&sexp_seq(body));

    // Single-form quasiquotes parse `__sexps[0]`; the plural ones map over all.
    let single = matches!(method, Method::Expr | Method::Command | Method::Action);
    let result = match method {
        Method::Expr => quote!(__parser.parse_expr(&__sexps[0])),
        Method::Command => quote!(__parser.parse_command(&__sexps[0])),
        Method::Action => quote!(__parser.parse_action(&__sexps[0])),
        Method::Program => quote! {
            __sexps
                .iter()
                .map(|__s| __parser.parse_command(__s))
                .collect::<::std::result::Result<::std::vec::Vec<::std::vec::Vec<_>>, _>>()
                .map(|__vs| __vs.into_iter().flatten().collect::<::std::vec::Vec<_>>())
        },
        Method::Rule => quote! {{
            assert_eq!(__sexps.len(), 2, "rule! expects `(<facts>) (<actions>)`");
            let __r = ::egglog::ast::Sexp::List(
                ::std::vec![
                    ::egglog::ast::Sexp::Atom("rule".to_string(), __span.clone()),
                    __sexps[0].clone(),
                    __sexps[1].clone(),
                ],
                __span.clone(),
            );
            __parser.parse_command(&__r)
        }},
        Method::Facts => quote! {
            __sexps
                .iter()
                .map(|__s| __parser.parse_fact(__s))
                .collect::<::std::result::Result<::std::vec::Vec<_>, _>>()
                .map(::egglog::ast::Facts)
        },
        Method::Actions => quote! {
            __sexps
                .iter()
                .map(|__s| __parser.parse_action(__s))
                .collect::<::std::result::Result<::std::vec::Vec<::std::vec::Vec<_>>, _>>()
                .map(|__vs| ::egglog::ast::GenericActions(__vs.into_iter().flatten().collect()))
        },
        Method::Sexp
        | Method::Sexps
        | Method::ResolveProgram
        | Method::RunProgram
        | Method::ResolveCommand
        | Method::RunCommand
        | Method::ResolveRule
        | Method::RunRule
        | Method::ResolveExpr
        | Method::RunExpr
        | Method::ResolveQuery
        | Method::RunQuery => {
            unreachable!("handled before parsing")
        }
    };
    let check = if single {
        quote!(assert_eq!(__sexps.len(), 1, "this egglog quasiquote expects exactly one form");)
    } else {
        quote!()
    };
    quote! {{
        let __span = ::egglog::span!();
        let __sexps: ::std::vec::Vec<::egglog::ast::Sexp> = #items;
        #check
        #parser_decl
        #result
    }}
    .into()
}

/// Split off an optional `<parser>,` prefix. egglog has no top-level commas, so
/// a depth-0 comma unambiguously separates a caller-supplied parser from the
/// egglog body; with no such comma the whole input is the body (default parser).
fn split_parser(ts: TokenStream2) -> (Option<TokenStream2>, TokenStream2) {
    let has_comma = ts
        .clone()
        .into_iter()
        .any(|tt| matches!(&tt, TokenTree::Punct(p) if p.as_char() == ','));
    if !has_comma {
        return (None, ts);
    }
    let mut parser = TokenStream2::new();
    let mut body = TokenStream2::new();
    let mut seen = false;
    for tt in ts {
        if !seen {
            if let TokenTree::Punct(ref p) = tt
                && p.as_char() == ','
            {
                seen = true;
                continue;
            }
            parser.extend(std::iter::once(tt));
        } else {
            body.extend(std::iter::once(tt));
        }
    }
    (Some(parser), body)
}

/// A child of a list being built: either one `Sexp` expression, or a spread
/// (`#..xs`) that extends the enclosing list with every element of `xs`.
enum Child {
    Single(TokenStream2),
    Spread(TokenStream2),
}

/// Emit an expression that builds a `Vec<Sexp>` from a child sequence: `push`
/// for single elements, `extend` for `#..` spreads.
fn build_items(children: &[Child]) -> TokenStream2 {
    let stmts = children.iter().map(|c| match c {
        Child::Single(e) => quote!(__items.push(#e);),
        Child::Spread(e) => quote! {
            __items.extend(
                ::std::iter::IntoIterator::into_iter(#e)
                    .map(|__e| ::egglog::ast::ToSexp::to_sexp(__e, __span.clone())),
            );
        },
    });
    quote! {{
        let mut __items: ::std::vec::Vec<::egglog::ast::Sexp> = ::std::vec::Vec::new();
        #(#stmts)*
        __items
    }}
}

/// Unwrap a parenthesized splice target — `#(expr)`, `#..(expr)`, `:#(expr)` —
/// to its inner tokens, so the generated `to_sexp(expr, ..)` carries no
/// redundant parentheses (which would trip the `unused_parens` lint under
/// `-D warnings`). A bare `#ident` target passes through unchanged.
fn splice_target(tt: TokenTree) -> TokenStream2 {
    if let TokenTree::Group(g) = &tt
        && g.delimiter() == Delimiter::Parenthesis
    {
        return g.stream();
    }
    TokenStream2::from(tt)
}

/// Turn a token stream into a sequence of list children (each builds one `Sexp`,
/// or splices many via `#..`).
fn sexp_seq(ts: TokenStream2) -> Vec<Child> {
    let mut out = Vec::new();
    let mut it = ts.into_iter().peekable();
    while let Some(tt) = it.next() {
        match tt {
            // `:#field` / `:#(rust expr)` — splice a runtime keyword. A `:`
            // directly followed by `#value` yields the atom `:<value>` (e.g. a
            // constructor field name), so named-arg patterns can carry runtime
            // field names: `(#kind :#field ?v ...)`. Must precede the atom_run
            // fallback, which would otherwise glue `:` `#` into one dead atom.
            TokenTree::Punct(ref colon)
                if colon.as_char() == ':'
                    && matches!(
                        it.peek(),
                        Some(TokenTree::Punct(h))
                            if h.as_char() == '#' && h.span().start() == colon.span().end()
                    ) =>
            {
                it.next(); // consume the `#`
                match it.next() {
                    Some(target) => out.push(Child::Single({
                        let target = splice_target(target);
                        quote! {
                            ::egglog::ast::keyword_to_sexp(#target, __span.clone())
                        }
                    })),
                    None => out.push(Child::Single(quote!(compile_error!(
                        "`:#` must be followed by a keyword value"
                    )))),
                }
            }
            // `#..xs` — unquote-splicing: extend the enclosing list with each
            // element of `xs` (an `IntoIterator<Item: ToSexp>`), like Racket's
            // `,@`. `#x` — splice a single value. The token(s) after are a Rust
            // expression, not egglog.
            TokenTree::Punct(ref p) if p.as_char() == '#' => {
                // `..` is two adjacent `.` puncts; look ahead without consuming.
                let is_spread = {
                    let mut ahead = it.clone();
                    matches!(ahead.next(), Some(TokenTree::Punct(a)) if a.as_char() == '.')
                        && matches!(ahead.next(), Some(TokenTree::Punct(b)) if b.as_char() == '.')
                };
                if is_spread {
                    it.next();
                    it.next(); // consume the two dots of `..`
                    match it.next() {
                        Some(target) => out.push(Child::Spread({
                            let target = splice_target(target);
                            quote!(#target)
                        })),
                        None => out.push(Child::Single(quote!(compile_error!(
                            "`#..` must be followed by a value to splice"
                        )))),
                    }
                } else {
                    match it.next() {
                        Some(target) => out.push(Child::Single({
                            let target = splice_target(target);
                            quote! {
                                ::egglog::ast::ToSexp::to_sexp(#target, __span.clone())
                            }
                        })),
                        None => out.push(Child::Single(quote!(compile_error!(
                            "`#` must be followed by a value to splice"
                        )))),
                    }
                }
            }
            TokenTree::Group(g) => match g.delimiter() {
                Delimiter::None => out.extend(sexp_seq(g.stream())),
                _ => {
                    let items = build_items(&sexp_seq(g.stream()));
                    out.push(Child::Single(quote! {
                        ::egglog::ast::Sexp::List(#items, __span.clone())
                    }));
                }
            },
            // A double-quoted string is its own literal.
            TokenTree::Literal(ref lit) if lit.to_string().starts_with('"') => {
                out.push(Child::Single(quote! {
                    ::egglog::ast::Sexp::Literal(
                        ::egglog::ast::Literal::String((#lit).to_string()),
                        __span.clone(),
                    )
                }));
            }
            // Anything else (ident, punct, number) begins an egglog atom: greedily
            // absorb following tokens that are directly adjacent (no whitespace),
            // so `?x`, `:no-merge`, `-1.0`, `>=` become single atoms while
            // space-separated tokens like `(+ 6 87)` stay separate.
            first => {
                let atom = atom_run(first.clone(), &mut it);
                out.push(Child::Single(
                    quote!(::egglog::ast::atom_to_sexp(#atom, __span.clone())),
                ));
            }
        }
    }
    out
}

/// The source text of a single token (for atoms): a punct's char, an
/// identifier's name, or a literal's text.
fn tt_str(tt: &TokenTree) -> String {
    match tt {
        TokenTree::Punct(p) => p.as_char().to_string(),
        TokenTree::Ident(i) => i.to_string(),
        TokenTree::Literal(l) => l.to_string(),
        TokenTree::Group(g) => g.to_string(),
    }
}

/// Reassemble one egglog atom starting at `first` by absorbing every following
/// token that begins exactly where the previous one ended — i.e. with no
/// whitespace between them. This mirrors egglog's tokenizer (an atom is a
/// maximal run of non-space, non-paren characters): `?x`, `:no-merge`, `-1.0`,
/// `>=`, `my-ruleset` all become single atoms, while `(+ 6 87)` stays three
/// tokens because the spaces break the run.
fn atom_run(first: TokenTree, it: &mut Peekable<impl Iterator<Item = TokenTree>>) -> String {
    let mut s = tt_str(&first);
    let mut end = first.span().end();
    while let Some(next) = it.peek() {
        if matches!(next, TokenTree::Group(_)) {
            break;
        }
        if let TokenTree::Literal(l) = next
            && l.to_string().starts_with('"')
        {
            break; // a string literal is its own token, never glued
        }
        if next.span().start() != end {
            break; // whitespace between tokens ends the atom
        }
        s.push_str(&tt_str(next));
        end = next.span().end();
        it.next();
    }
    s
}
