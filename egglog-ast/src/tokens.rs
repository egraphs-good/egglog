//! Turning Rust proc-macro tokens into egglog atoms.
//!
//! Shared by the quasiquote proc-macro crates (`egglog-quote`, `egglog-checked`)
//! so they reconstruct egglog atoms from Rust tokens identically. Gated behind
//! the `tokens` feature so egglog-ast's normal build doesn't pull `proc-macro2`.

use proc_macro2::{Span, TokenTree};
use std::iter::Peekable;

/// The source text of a single token: a punct's char, an identifier's name, or
/// a literal's / group's text.
pub fn tt_str(tt: &TokenTree) -> String {
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
///
/// Returns the atom text and the span of its first token, which anchors any
/// diagnostic reported inside the atom.
pub fn atom_run(
    first: TokenTree,
    it: &mut Peekable<impl Iterator<Item = TokenTree>>,
) -> (String, Span) {
    let span = first.span();
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
    (s, span)
}
