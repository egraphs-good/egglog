use crate::util::{FreshGen, HashMap, HashSet, SymbolGen};
use crate::*;
use std::fmt::Write;

pub type TermId = usize;

#[allow(rustdoc::private_intra_doc_links)]
/// Like [`Expr`]s but with sharing and deduplication.
///
/// Terms refer to their children indirectly via opaque [TermId]s (internally
/// these are just `usize`s) that map into an ambient [`TermDag`].
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Term {
    Lit(Literal),
    Var(String),
    App(String, Vec<TermId>),
}

/// A hashconsing arena for [`Term`]s.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct TermDag {
    /// A bidirectional map between deduplicated `Term`s and indices.
    nodes: IndexSet<Term>,
}

const MAX_PRETTY_LINE_WIDTH: usize = 80;
const PRETTY_INDENT_STEP: usize = 2;
const MIN_SHARED_TERM_SIZE: usize = 4;

#[macro_export]
macro_rules! match_term_app {
    ($e:expr; $body:tt) => {
        match $e {
            Term::App(head, args) => {
                match (head.as_str(), args.as_slice())
                    $body
            }
            _ => panic!("not an app")
        }
    }
}

#[derive(Clone)]
struct RenderedTerm {
    inline: String,
    pretty: String,
}

impl RenderedTerm {
    fn from_symbol(symbol: String) -> Self {
        Self {
            inline: symbol.clone(),
            pretty: symbol,
        }
    }

    fn is_multiline(&self) -> bool {
        self.pretty.contains('\n')
    }
}

/// Context used during term rendering with let-binding support.
struct TermRenderContext<'a> {
    /// Generator for fresh variable names used in let bindings.
    fresh: &'a mut SymbolGen,
    /// Maps each term ID to the number of times it is referenced in the DAG.
    /// Terms referenced multiple times are candidates for let-binding.
    ref_counts: &'a HashMap<TermId, usize>,
    /// Maps each term ID to its size (number of nodes in the subtree).
    /// Used to decide whether a term is large enough to warrant let-binding.
    sizes: &'a HashMap<TermId, usize>,
    /// Maps term IDs to the variable names they've been bound to.
    /// Once a term is let-bound, subsequent references use this name.
    bindings: HashMap<TermId, String>,
    /// Buffer where let bindings are accumulated as they are created.
    buf: &'a mut String,
    /// Function that takes a constructor name and returns the name hint to use
    name_hint_fn: Box<dyn Fn(&str) -> String + 'a>,
}

impl<'a> TermRenderContext<'a> {
    fn new<F>(
        fresh: &'a mut SymbolGen,
        ref_counts: &'a HashMap<TermId, usize>,
        sizes: &'a HashMap<TermId, usize>,
        buf: &'a mut String,
        name_hint_fn: F,
    ) -> Self
    where
        F: Fn(&str) -> String + 'a,
    {
        Self {
            fresh,
            ref_counts,
            sizes,
            bindings: HashMap::default(),
            buf,
            name_hint_fn: Box::new(name_hint_fn),
        }
    }

    fn get_name_hint(&self, constructor_name: &str) -> String {
        (self.name_hint_fn)(constructor_name)
    }
}

impl TermDag {
    /// Returns the number of nodes in this DAG.
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Convert the given term to its id.
    ///
    /// Panics if the term does not already exist in this [TermDag].
    pub fn lookup(&self, node: &Term) -> TermId {
        self.nodes.get_index_of(node).unwrap()
    }

    /// Convert the given id to the corresponding term.
    ///
    /// Panics if the id is not valid.
    pub fn get(&self, id: TermId) -> &Term {
        self.nodes.get_index(id).unwrap()
    }

    /// Make and return a [`Term::App`] with the given head symbol and children,
    /// and insert into the DAG if it is not already present.
    ///
    /// Panics if any of the children are not already in the DAG.
    pub fn app(&mut self, sym: String, children: Vec<TermId>) -> TermId {
        let node = Term::App(sym, children);

        self.add_node(&node)
    }

    /// Make a [`Term::Lit`] with the given literal and return its [`TermId`],
    /// inserting it into the DAG if it is not already present.
    pub fn lit(&mut self, lit: Literal) -> TermId {
        let node = Term::Lit(lit);

        self.add_node(&node)
    }

    /// Make and return a [`Term::Var`] with the given symbol, and insert into
    /// the DAG if it is not already present.
    pub fn var(&mut self, sym: String) -> TermId {
        let node = Term::Var(sym);

        self.add_node(&node)
    }

    fn add_node(&mut self, node: &Term) -> TermId {
        self.nodes.get_index_of(node).unwrap_or_else(|| {
            let id = self.nodes.len();
            self.nodes.insert(node.clone());
            id
        })
    }

    /// Recursively converts the given expression to a term.
    ///
    /// This involves inserting every subexpression into this DAG. Because
    /// TermDags are hashconsed, the resulting term is guaranteed to maximally
    /// share subterms.
    pub fn expr_to_term(&mut self, expr: &GenericExpr<String, String>) -> TermId {
        let res = match expr {
            GenericExpr::Lit(_, lit) => Term::Lit(lit.clone()),
            GenericExpr::Var(_, v) => Term::Var(v.to_owned()),
            GenericExpr::Call(_, op, args) => {
                let args = args.iter().map(|a| self.expr_to_term(a)).collect();
                Term::App(op.clone(), args)
            }
        };
        self.add_node(&res)
    }

    /// Recursively converts the given term to an expression.
    ///
    /// Panics if the term contains subterms that are not in the DAG.
    pub fn term_to_expr(&self, term: &TermId, span: Span) -> Expr {
        let term = self.get(*term);
        match term {
            Term::Lit(lit) => Expr::Lit(span, lit.clone()),
            Term::Var(v) => Expr::Var(span, v.clone()),
            Term::App(op, args) => {
                let args: Vec<_> = args
                    .iter()
                    .map(|a| self.term_to_expr(a, span.clone()))
                    .collect();
                Expr::Call(span, op.clone(), args)
            }
        }
    }

    /// Prints a term to a string, putting let bindings for shared subterms.
    pub fn to_string_with_let(&self, fresh: &mut SymbolGen, term_id: TermId) -> String {
        self.to_string_with_let_and_hint(fresh, term_id, "t")
    }

    /// Prints a term to a string, putting let bindings for shared subterms.
    /// Uses the given name hint for generated variable names.
    pub fn to_string_with_let_and_hint(
        &self,
        fresh: &mut SymbolGen,
        term_id: TermId,
        name_hint: &str,
    ) -> String {
        let mut buf = String::new();
        let hint = name_hint.to_string();
        let final_str =
            self.to_string_with_let_internal(fresh, term_id, &mut buf, move |_| hint.clone());
        format!("{buf}\n{final_str}")
    }

    /// Prints a term to a string, putting let bindings for shared subterms in `buf`.
    /// Returns the final string representation of the term.
    /// The `name_hint_fn` takes a constructor name and returns the hint to use for that term.
    pub(crate) fn to_string_with_let_internal<'a, F>(
        &self,
        fresh: &'a mut SymbolGen,
        term_id: TermId,
        buf: &'a mut String,
        name_hint_fn: F,
    ) -> String
    where
        F: Fn(&str) -> String + 'a,
    {
        let (ref_counts, sizes) = self.collect_term_stats(term_id);
        let mut ctx = TermRenderContext::new(fresh, &ref_counts, &sizes, buf, name_hint_fn);
        let rendered = self.render_term(term_id, &mut ctx, false, 0);
        rendered.pretty
    }

    fn render_term(
        &self,
        term_id: TermId,
        ctx: &mut TermRenderContext,
        allow_binding: bool,
        indent: usize,
    ) -> RenderedTerm {
        if let Some(existing) = ctx.bindings.get(&term_id) {
            return RenderedTerm::from_symbol(existing.clone());
        }

        // Get the constructor name for the hint function (if it's an App)
        let constructor_name = match self.get(term_id) {
            Term::App(name, _) => Some(name.clone()),
            _ => None,
        };

        let rendered = match self.get(term_id) {
            Term::App(name, children) => {
                let mut child_renderings = Vec::with_capacity(children.len());
                for child_id in children {
                    let rendered_child =
                        self.render_term(*child_id, ctx, true, indent + PRETTY_INDENT_STEP);
                    child_renderings.push(rendered_child);
                }

                let mut inline = format!("({}", name);
                for child in &child_renderings {
                    inline.push(' ');
                    inline.push_str(&child.inline);
                }
                inline.push(')');

                let inline_len = inline.chars().count();
                let exceeds_width = indent + inline_len > MAX_PRETTY_LINE_WIDTH;
                let child_multiline = child_renderings.iter().any(|c| c.is_multiline());

                let pretty = if exceeds_width || child_multiline {
                    if child_renderings.is_empty() {
                        format!("({})", name)
                    } else {
                        let mut s = format!("({}", name);
                        for (idx, child) in child_renderings.iter().enumerate() {
                            s.push('\n');
                            s.push_str(&" ".repeat(indent + PRETTY_INDENT_STEP));
                            s.push_str(&child.pretty);
                            if idx + 1 == child_renderings.len() {
                                s.push(')');
                            }
                        }
                        s
                    }
                } else {
                    inline.clone()
                };

                RenderedTerm { inline, pretty }
            }
            Term::Lit(lit) => {
                let repr = format!("{lit}");
                RenderedTerm {
                    inline: repr.clone(),
                    pretty: repr,
                }
            }
            Term::Var(v) => RenderedTerm {
                inline: v.clone(),
                pretty: v.clone(),
            },
        };

        let term_size = *ctx.sizes.get(&term_id).unwrap_or(&1);
        let repeat_count = ctx.ref_counts.get(&term_id).copied().unwrap_or(1);
        let should_bind = allow_binding && repeat_count > 1 && term_size >= MIN_SHARED_TERM_SIZE;

        if should_bind {
            let hint = ctx.get_name_hint(constructor_name.as_deref().unwrap_or("t"));
            let let_name = ctx.fresh.fresh(&hint);
            self.push_binding(ctx.buf, &let_name, &rendered.pretty);
            ctx.bindings.insert(term_id, let_name.clone());
            RenderedTerm::from_symbol(let_name)
        } else {
            rendered
        }
    }

    fn push_binding(&self, buf: &mut String, name: &str, body: &str) {
        let trimmed = body.trim_end();
        if trimmed.is_empty() {
            buf.push_str("(let ");
            buf.push_str(name);
            buf.push_str(")\n");
            return;
        }

        if trimmed.contains('\n') {
            buf.push_str("(let ");
            buf.push_str(name);
            buf.push('\n');
            let lines: Vec<&str> = trimmed.lines().collect();
            for (idx, line) in lines.iter().enumerate() {
                buf.push_str(&" ".repeat(PRETTY_INDENT_STEP));
                buf.push_str(line);
                if idx + 1 < lines.len() {
                    buf.push('\n');
                } else {
                    buf.push(')');
                    buf.push('\n');
                }
            }
        } else {
            buf.push_str("(let ");
            buf.push_str(name);
            buf.push(' ');
            buf.push_str(trimmed);
            buf.push_str(")\n");
        }
    }

    fn collect_term_stats(
        &self,
        term_id: TermId,
    ) -> (HashMap<TermId, usize>, HashMap<TermId, usize>) {
        let mut counts = HashMap::default();
        let mut visited = HashSet::default();
        self.collect_term_ref_counts_inner(term_id, &mut counts, &mut visited);

        let mut sizes = HashMap::default();
        self.compute_term_size(term_id, &mut sizes);

        (counts, sizes)
    }

    fn compute_term_size(&self, term_id: TermId, sizes: &mut HashMap<TermId, usize>) -> usize {
        if let Some(size) = sizes.get(&term_id) {
            return *size;
        }

        let size = match self.get(term_id) {
            Term::App(_, children) => {
                1 + children
                    .iter()
                    .map(|child| self.compute_term_size(*child, sizes))
                    .sum::<usize>()
            }
            Term::Lit(_) | Term::Var(_) => 1,
        };

        sizes.insert(term_id, size);
        size
    }

    fn collect_term_ref_counts_inner(
        &self,
        term_id: TermId,
        counts: &mut HashMap<TermId, usize>,
        visited: &mut HashSet<TermId>,
    ) {
        *counts.entry(term_id).or_insert(0) += 1;
        if !visited.insert(term_id) {
            return;
        }

        if let Term::App(_, children) = self.get(term_id) {
            for child in children {
                self.collect_term_ref_counts_inner(*child, counts, visited);
            }
        }
    }

    /// Converts the given term to a string.
    ///
    /// Panics if the term or any of its subterms are not in the DAG.
    pub fn to_string(&self, term: TermId) -> String {
        let mut result = String::new();
        // subranges of the `result` string containing already stringified subterms
        let mut ranges = HashMap::<TermId, (usize, usize)>::default();
        // use a stack to avoid stack overflow

        let mut stack = vec![(term, false, None)];
        while let Some((id, space_before, mut start_index)) = stack.pop() {
            if space_before {
                result.push(' ');
            }

            if let Some((start, end)) = ranges.get(&id) {
                result.extend_from_within(*start..*end);
                continue;
            }

            match self.nodes[id].clone() {
                Term::App(name, children) => {
                    if start_index.is_some() {
                        result.push(')');
                    } else {
                        stack.push((id, false, Some(result.len())));
                        write!(&mut result, "({}", name).unwrap();
                        for c in children.iter().rev() {
                            stack.push((*c, true, None));
                        }
                    }
                }
                Term::Lit(lit) => {
                    start_index = Some(result.len());
                    write!(&mut result, "{lit}").unwrap();
                }
                Term::Var(v) => {
                    start_index = Some(result.len());
                    write!(&mut result, "{v}").unwrap();
                }
            }

            if let Some(start_index) = start_index {
                ranges.insert(id, (start_index, result.len()));
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ast::*, span, util::SymbolGen};

    fn parse_term(s: &str) -> (TermDag, TermId) {
        let e = Parser::default().get_expr_from_string(None, s).unwrap();
        let mut td = TermDag::default();
        let t = td.expr_to_term(&e);
        (td, t)
    }

    #[test]
    fn test_to_from_expr() {
        let s = r#"(f (g x y) x y (g x y))"#;
        let e = Parser::default().get_expr_from_string(None, s).unwrap();
        let mut td = TermDag::default();
        assert_eq!(td.size(), 0);
        let t = td.expr_to_term(&e);
        assert_eq!(td.size(), 4);
        // the expression above has 4 distinct subterms.
        // in left-to-right, depth-first order, they are:
        //     x, y, (g x y), and the root call to f
        // so we can compute expected answer by hand:
        assert_eq!(
            td.nodes.as_slice().iter().cloned().collect::<Vec<_>>(),
            vec![
                Term::Var("x".into()),
                Term::Var("y".into()),
                Term::App("g".into(), vec![0, 1]),
                Term::App("f".into(), vec![2, 0, 1, 2]),
            ]
        );
        // This is tested using string equality because e1 and e2 have different
        let e2 = td.term_to_expr(&t, span!());
        // annotations. A better way to test this would be to implement a map_ann
        // function for GenericExpr.
        assert_eq!(format!("{e}"), format!("{e2}")); // roundtrip
    }

    #[test]
    fn test_match_term_app() {
        let s = r#"(f (g x y) x y (g x y))"#;
        let (td, t) = parse_term(s);
        let term = td.get(t);
        match_term_app!(term; {
            ("f", [_, x, _, _]) => {
                let span = span!();
                assert_eq!(
                    td.term_to_expr(x, span.clone()),
                    crate::ast::GenericExpr::Var(span, "x".to_owned())
                )
            }
            (head, _) => panic!("unexpected head {}, in {}:{}:{}", head, file!(), line!(), column!())
        })
    }

    #[test]
    fn test_to_string() {
        let s = r#"(f (g x y) x y (g x y))"#;
        let (td, t) = parse_term(s);
        assert_eq!(td.to_string(t), s);
    }

    #[test]
    fn test_lookup() {
        let s = r#"(f (g x y) x y (g x y))"#;
        let (td, t) = parse_term(s);
        assert_eq!(t, td.size() - 1);
    }

    #[test]
    fn test_app_var_lit() {
        let s = r#"(f (g x y) x 7 (g x y))"#;
        let (mut td, t) = parse_term(s);
        let x = td.var("x".into());
        let y = td.var("y".into());
        let seven = td.lit(7.into());
        let g = td.app("g".into(), vec![x, y]);
        let t2 = td.app("f".into(), vec![g, x, seven, g]);
        assert_eq!(t, t2);
    }

    #[test]
    fn test_to_string_with_let_inlines_small_terms() {
        let s = r#"(f (g x) (g x) (g x))"#;
        let (td, t) = parse_term(s);
        let mut sym = SymbolGen::new(String::new());
        let result = td.to_string_with_let(&mut sym, t);
        // No let bindings means result is just newline + repr
        assert_eq!(result.trim(), s);
    }

    #[test]
    fn test_to_string_with_let_shares_large_terms() {
        let g_segment = ["(g a b)"; 8].join(" ");
        let s = format!("(f (h {0}) (h {0}))", g_segment);
        let (td, t) = parse_term(&s);
        let mut buf = String::new();
        let mut sym = SymbolGen::new(String::new());
        let repr = td.to_string_with_let_internal(&mut sym, t, &mut buf, |_| "t".to_string());
        let first_line = buf.lines().next().expect("expected let binding");
        assert!(first_line.starts_with("(let t"));
        assert!(buf.contains("(h"));
        let has_lonely_paren = buf.lines().any(|line| line.trim() == ")");
        assert!(
            !has_lonely_paren,
            "unexpected standalone closing paren in\n{buf}"
        );
        assert!(buf.trim_end().ends_with(')'));
        assert_eq!(repr, "(f t t)");
    }

    #[test]
    fn test_to_string_with_let_wraps_long_lines() {
        let s = r#"(verylongfunctionnamewithmanysegments alpha_argument beta_argument gamma_argument delta_argument epsilon_argument zeta_argument)"#;
        let (td, t) = parse_term(s);
        let mut sym = SymbolGen::new(String::new());
        let result = td.to_string_with_let(&mut sym, t);
        // No let bindings, so result is just newline + repr
        let repr = result.trim();
        assert!(repr.contains('\n'));
        assert!(repr.contains("\n  "));
        assert!(repr.starts_with("(verylongfunctionnamewithmanysegments"));
    }

    #[test]
    fn test_multiline_parentheses_share_final_line() {
        let expr = "(Trans (Add 3 2) (start) (Rule (Add 3 2) (Add 2 3) (name rw1) (premises t1) (substitution (a 2) (b 3))) t)";
        let (td, t) = parse_term(expr);
        let mut buf = String::new();
        let mut sym = SymbolGen::new(String::new());
        let repr = td.to_string_with_let_internal(&mut sym, t, &mut buf, |_| "t".to_string());
        assert!(repr.contains('\n'), "expected multiline output, got {repr}");
        let has_lonely_paren = repr.lines().any(|line| line.trim() == ")");
        assert!(
            !has_lonely_paren,
            "found standalone closing paren line in {repr}"
        );
        if let Some(last_line) = repr.lines().last() {
            assert!(
                last_line.ends_with(')'),
                "last line should end with closing paren: {last_line}"
            );
        }
        let buf_has_lonely = buf.lines().any(|line| line.trim() == ")");
        assert!(
            !buf_has_lonely,
            "bindings contain standalone closing paren in\n{buf}"
        );
    }
}
