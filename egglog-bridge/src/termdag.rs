use egglog_ast::{
    generic_ast::{Expr, GenericExpr, Literal},
    span::Span,
};

use crate::*;
use hashbrown::HashMap;
use indexmap::IndexSet;
use std::{fmt::Write, io};

pub type TermId = usize;

#[allow(rustdoc::private_intra_doc_links)]
/// Like [`Expr`]s but with sharing and deduplication.
///
/// Terms refer to their children indirectly via opaque [TermId]s (internally
/// these are just `usize`s) that map into an ambient [`TermDag`].
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Term {
    Lit(Literal),
    /// This is a placeholder, used to represent terms that are backed by a base type that we
    /// cannot model in the source / AST language.
    UnknownLit,
    Var(String),
    App(String, Vec<TermId>),
}

/// A hashconsing arena for [`Term`]s.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TermDag {
    /// A bidirectional map between deduplicated `Term`s and indices.
    nodes: IndexSet<Term>,
}

impl Default for TermDag {
    fn default() -> Self {
        Self::new()
    }
}

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

#[derive(Clone, Debug)]
pub struct PrettyPrintConfig {
    pub line_width: usize,
    pub indent_size: usize,
}

impl Default for PrettyPrintConfig {
    fn default() -> Self {
        Self {
            line_width: 512,
            indent_size: 4,
        }
    }
}

pub(crate) struct PrettyPrinter<'w, W: io::Write> {
    writer: &'w mut W,
    config: &'w PrettyPrintConfig,
    current_indent: usize,
    current_line_pos: usize,
}

impl<'w, W: io::Write> PrettyPrinter<'w, W> {
    pub(crate) fn new(writer: &'w mut W, config: &'w PrettyPrintConfig) -> Self {
        Self {
            writer,
            config,
            current_indent: 0,
            current_line_pos: 0,
        }
    }

    pub(crate) fn write_str(&mut self, s: &str) -> io::Result<()> {
        write!(self.writer, "{s}")?;
        self.current_line_pos += s.len();
        Ok(())
    }

    pub(crate) fn newline(&mut self) -> io::Result<()> {
        writeln!(self.writer)?;
        self.current_line_pos = 0;
        self.write_indent()?;
        Ok(())
    }

    pub(crate) fn write_indent(&mut self) -> io::Result<()> {
        for _ in 0..self.current_indent {
            write!(self.writer, " ")?;
        }
        self.current_line_pos = self.current_indent;
        Ok(())
    }

    pub(crate) fn increase_indent(&mut self) {
        self.current_indent += self.config.indent_size;
    }

    pub(crate) fn decrease_indent(&mut self) {
        self.current_indent = self.current_indent.saturating_sub(self.config.indent_size);
    }

    pub(crate) fn should_break(&self, additional_chars: usize) -> bool {
        self.current_line_pos + additional_chars > self.config.line_width
    }

    pub(crate) fn write_with_break(&mut self, s: &str) -> io::Result<()> {
        if self.should_break(s.len()) && self.current_line_pos > self.current_indent {
            self.newline()?;
            self.write_indent()?;
        }
        self.write_str(s)
    }
}

impl TermDag {
    /// Create a new empty TermDag without type tracking
    pub fn new() -> Self {
        Self {
            nodes: IndexSet::new(),
        }
    }

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
    pub fn app(&mut self, sym: String, children: Vec<Term>) -> Term {
        let node = Term::App(sym, children.iter().map(|c| self.lookup(c)).collect());

        self.add_node(&node);

        node
    }

    /// Like [`TermDag::app`], but expects and returns [`TermId`]s instead of
    /// [`Term`]s.
    pub fn app_id(&mut self, sym: String, children: Vec<TermId>) -> TermId {
        let node = Term::App(sym, children);

        self.add_node(&node);

        self.lookup(&node)
    }

    /// Make and return a [`Term::Lit`] with the given literal, and insert into
    /// the DAG if it is not already present.
    pub fn lit(&mut self, lit: Literal) -> Term {
        let node = Term::Lit(lit);

        self.add_node(&node);

        node
    }

    pub fn unknown_lit(&mut self) -> Term {
        let node = Term::UnknownLit;
        self.add_node(&node);
        node
    }

    /// Like [`TermDag::lit`], but returns a [`TermId`] instead of a [`Term`].
    pub fn lit_id(&mut self, lit: Literal) -> TermId {
        let node = Term::Lit(lit);
        self.add_node(&node);
        self.lookup(&node)
    }

    /// Make and return a [`Term::Var`] with the given symbol, and insert into
    /// the DAG if it is not already present.
    pub fn var(&mut self, sym: String) -> Term {
        let node = Term::Var(sym);

        self.add_node(&node);

        node
    }

    pub fn var_id(&mut self, sym: String) -> TermId {
        let node = Term::Var(sym);
        self.add_node(&node);
        self.lookup(&node)
    }

    /// Get the children of a term by its ID
    pub fn get_children(&self, id: TermId) -> Vec<TermId> {
        match self.get(id) {
            Term::App(_, children) => children.clone(),
            Term::Lit(_) | Term::Var(_) | Term::UnknownLit => Vec::new(),
        }
    }

    fn add_node(&mut self, node: &Term) {
        if self.nodes.get(node).is_none() {
            self.nodes.insert(node.clone());
        }
    }

    /// Recursively converts the given expression to a term.
    ///
    /// This involves inserting every subexpression into this DAG. Because
    /// TermDags are hashconsed, the resulting term is guaranteed to maximally
    /// share subterms.
    pub fn expr_to_term(&mut self, expr: &GenericExpr<String, String>) -> Term {
        let res = match expr {
            GenericExpr::Lit(_, lit) => Term::Lit(lit.clone()),
            GenericExpr::Var(_, v) => Term::Var(v.to_owned()),
            GenericExpr::Call(_, op, args) => {
                let args = args
                    .iter()
                    .map(|a| {
                        let term = self.expr_to_term(a);
                        self.lookup(&term)
                    })
                    .collect();
                Term::App(op.clone(), args)
            }
        };
        self.add_node(&res);
        res
    }

    /// Recursively converts the given term to an expression.
    ///
    /// Panics if the term contains subterms that are not in the DAG or cannot be represented by
    /// the current syntax.
    pub fn term_to_expr(&self, term: &Term, span: Span) -> Expr {
        match term {
            Term::UnknownLit => panic!("unknown base value"),
            Term::Lit(lit) => Expr::Lit(span, lit.clone()),
            Term::Var(v) => Expr::Var(span, v.clone()),
            Term::App(op, args) => {
                let args: Vec<_> = args
                    .iter()
                    .map(|a| self.term_to_expr(self.get(*a), span.clone()))
                    .collect();
                Expr::Call(span, op.clone(), args)
            }
        }
    }

    /// Converts the given term to a string.
    ///
    /// Panics if the term or any of its subterms are not in the DAG.
    pub fn to_string(&self, term: &Term) -> String {
        let mut result = String::new();
        // subranges of the `result` string containing already stringified subterms
        let mut ranges = HashMap::<TermId, (usize, usize)>::default();
        let id = self.lookup(term);
        // use a stack to avoid stack overflow

        let mut stack = vec![(id, false, None)];
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
                Term::UnknownLit => {
                    start_index = Some(result.len());
                    write!(&mut result, "<unknown-base-val>").unwrap();
                }
            }

            if let Some(start_index) = start_index {
                ranges.insert(id, (start_index, result.len()));
            }
        }

        result
    }

    /// Pretty-print the given term to a string.
    pub fn to_string_pretty(&self, term: &Term) -> String {
        let mut buf = Vec::new();
        self.print_term_pretty(term, &PrettyPrintConfig::default(), &mut buf)
            .expect("pretty printing term failed");
        String::from_utf8(buf).expect("pretty printer emitted invalid UTF-8")
    }

    /// Pretty-print the given term to a string by term id.
    pub fn to_string_pretty_id(&self, term: TermId) -> String {
        self.to_string_pretty(self.get(term))
    }

    /// Print the term with pretty-printing configuration.
    pub fn print_term_pretty(
        &self,
        term: &Term,
        config: &PrettyPrintConfig,
        writer: &mut impl io::Write,
    ) -> io::Result<()> {
        let mut printer = PrettyPrinter::new(writer, config);
        self.print_term_with_printer(term, &mut printer)
    }

    pub(crate) fn print_term_with_printer<W: io::Write>(
        &self,
        term: &Term,
        printer: &mut PrettyPrinter<W>,
    ) -> io::Result<()> {
        match term {
            Term::Lit(lit) => {
                printer.write_str(&format!("{lit}"))?;
            }
            Term::UnknownLit => {
                printer.write_str("(unsupported-base-val)")?;
            }
            Term::Var(v) => {
                printer.write_str(v)?;
            }
            Term::App(head, args) => {
                printer.write_str(&format!("({head}"))?;
                if !args.is_empty() {
                    printer.increase_indent();
                    for arg in args.iter() {
                        printer.write_with_break(" ")?;
                        self.print_term_with_printer(self.get(*arg), printer)?;
                    }
                    printer.decrease_indent();
                }
                printer.write_str(")")?;
            }
        }
        Ok(())
    }

    /// Project a particular argument of a term by index.
    /// Returns None if the term is not an application or the index is out of bounds.
    pub fn proj(&self, term: &Term, arg_idx: usize) -> Option<TermId> {
        match term {
<<<<<<< HEAD
            Term::App(_hd, args) => args.get(arg_idx).copied(),
=======
            Term::App(_hd, args) => {
                args.get(arg_idx).copied()
            }
>>>>>>> ced37a12 (some progress)
            _ => None,
        }
    }

    /// Project a particular argument of a term by index, given the term's id.
    pub fn proj_id(&self, term: TermId, arg_idx: usize) -> Option<TermId> {
        self.proj(self.get(term), arg_idx)
    }
}
