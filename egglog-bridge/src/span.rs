use ordered_float::OrderedFloat;
use std::collections::HashMap;
use std::fmt::{self, Debug, Display};
use std::sync::Arc;

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Span {
    Panic,
    Egglog(Arc<EgglogSpan>),
    Rust(Arc<RustSpan>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EgglogSpan {
    pub file: Arc<SrcFile>,
    pub i: usize,
    pub j: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RustSpan {
    pub file: &'static str,
    pub line: u32,
    pub column: u32,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct SrcFile {
    pub name: Option<String>,
    pub contents: String,
}

impl SrcFile {
    pub fn get_location(&self, offset: usize) -> (usize, usize) {
        let mut line = 1;
        let mut col = 1;
        for (i, c) in self.contents.char_indices() {
            if i == offset {
                break;
            }
            if c == '\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        (line, col)
    }
}

impl Span {
    pub fn string(&self) -> &str {
        match self {
            Span::Panic => panic!("Span::Panic in Span::string"),
            Span::Rust(_) => panic!("Span::Rust cannot track end position"),
            Span::Egglog(span) => &span.file.contents[span.i..span.j],
        }
    }
}

struct Location {
    line: usize,
    col: usize,
}

impl Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Span::Panic => panic!("Span::Panic in impl Display"),
            Span::Rust(span) => write!(f, "At {}:{} of {}", span.line, span.column, span.file),
            Span::Egglog(span) => {
                let (start_line, start_col) = span.file.get_location(span.i);
                let (end_line, end_col) = span
                    .file
                    .get_location((span.j.saturating_sub(1)).max(span.i));
                let quote = self.string();
                match (&span.file.name, start_line == end_line) {
                    (Some(filename), true) => write!(
                        f,
                        "In {}:{}-{} of {filename}: {quote}",
                        start_line, start_col, end_col
                    ),
                    (Some(filename), false) => write!(
                        f,
                        "In {}:{}-{}:{} of {filename}: {quote}",
                        start_line, start_col, end_line, end_col
                    ),
                    (None, false) => write!(
                        f,
                        "In {}:{}-{}:{}: {quote}",
                        start_line, start_col, end_line, end_col
                    ),
                    (None, true) => {
                        write!(f, "In {}:{}-{}: {quote}", start_line, start_col, end_col)
                    }
                }
            }
        }
    }
}
