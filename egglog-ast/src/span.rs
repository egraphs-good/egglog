use std::fmt::{self, Debug, Display};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

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

#[derive(Debug)]
pub struct SrcFile {
    pub name: Option<String>,
    pub contents: String,
    line_starts: OnceLock<Vec<usize>>,
}

impl SrcFile {
    pub fn new(name: Option<String>, contents: String) -> Self {
        Self {
            name,
            contents,
            line_starts: OnceLock::new(),
        }
    }

    fn line_starts(&self) -> &[usize] {
        self.line_starts.get_or_init(|| {
            let mut starts = vec![0];
            for (i, c) in self.contents.char_indices() {
                if c == '\n' {
                    starts.push(i + 1);
                }
            }
            starts
        })
    }

    pub fn get_location(&self, offset: usize) -> (usize, usize) {
        let offset = offset.min(self.contents.len());
        let starts = self.line_starts();
        let line_idx = starts
            .partition_point(|start| *start <= offset)
            .saturating_sub(1);
        let line_start = starts[line_idx];
        let col = self.contents[line_start..offset].chars().count() + 1;
        (line_idx + 1, col)
    }
}

impl Clone for SrcFile {
    fn clone(&self) -> Self {
        let line_starts = OnceLock::new();
        if let Some(starts) = self.line_starts.get() {
            let _ = line_starts.set(starts.clone());
        }
        Self {
            name: self.name.clone(),
            contents: self.contents.clone(),
            line_starts,
        }
    }
}

impl PartialEq for SrcFile {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.contents == other.contents
    }
}

impl Eq for SrcFile {}

impl Hash for SrcFile {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.contents.hash(state);
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

impl Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
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
                // Use just the file name, not the full path, for cross-platform consistency in snapshots
                let display_name = span.file.name.as_ref().map(|path| {
                    std::path::Path::new(path)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(path)
                });
                match (&display_name, start_line == end_line) {
                    (Some(filename), true) => write!(
                        f,
                        "In {start_line}:{start_col}-{end_col} of {filename}: {quote}"
                    ),
                    (Some(filename), false) => write!(
                        f,
                        "In {start_line}:{start_col}-{end_line}:{end_col} of {filename}: {quote}"
                    ),
                    (None, false) => write!(
                        f,
                        "In {start_line}:{start_col}-{end_line}:{end_col}: {quote}"
                    ),
                    (None, true) => {
                        write!(f, "In {start_line}:{start_col}-{end_col}: {quote}")
                    }
                }
            }
        }
    }
}
