use std::any::Any;
use std::fmt::{self, Debug, Display};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Span {
    Panic,
    Egglog(Arc<EgglogSpan>),
    Rust(Arc<RustSpan>),
    /// A caller-owned source location, retained without interpreting its payload.
    External(Arc<ExternalSpan>),
}

/// Opaque source metadata shared by cloned programs and runtime diagnostics.
///
/// Equality and hashing use allocation identity; two separately allocated
/// locations remain distinct even when their payloads contain equal values.
#[derive(Clone)]
pub struct ExternalSpan(pub Arc<dyn Any + Send + Sync>);

impl PartialEq for ExternalSpan {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ExternalSpan {}

impl Hash for ExternalSpan {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).cast::<()>().hash(state);
    }
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
            Span::External(_) => panic!("Span::External has no source text"),
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
            Span::External(_) => f.write_str("At external source"),
            Span::Egglog(span) => {
                let (start_line, start_col) = span.file.get_location(span.i);
                let (end_line, end_col) = span
                    .file
                    .get_location((span.j.saturating_sub(1)).max(span.i));
                let quote = self.string();
                match (&span.file.name, start_line == end_line) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_spans_retain_payload_and_allocation_identity() {
        assert_eq!(
            std::mem::size_of::<Span>(),
            2 * std::mem::size_of::<usize>()
        );
        let origin = Arc::new(String::from("authored site"));
        let span = Span::External(Arc::new(ExternalSpan(origin.clone())));
        let cloned = span.clone();
        let distinct = Span::External(Arc::new(ExternalSpan(Arc::new(String::from(
            "authored site",
        )))));
        assert_eq!(span, cloned);
        assert_ne!(span, distinct);
        let mut original_hash = std::collections::hash_map::DefaultHasher::new();
        span.hash(&mut original_hash);
        let mut cloned_hash = std::collections::hash_map::DefaultHasher::new();
        cloned.hash(&mut cloned_hash);
        assert_eq!(original_hash.finish(), cloned_hash.finish());
        let Span::External(external) = cloned else {
            unreachable!()
        };
        assert_eq!(external.0.downcast_ref::<String>(), Some(origin.as_ref()));
        assert_eq!(
            format!("{:?}", Span::External(external)),
            "At external source"
        );
    }

    #[test]
    fn egglog_span_display_preserves_full_path() {
        let path = "/tmp/egglog/full/path/example.egg".to_string();
        let span = Span::Egglog(Arc::new(EgglogSpan {
            file: Arc::new(SrcFile {
                name: Some(path.clone()),
                contents: "(bad)".to_string(),
            }),
            i: 0,
            j: 5,
        }));

        assert_eq!(span.to_string(), format!("In 1:1-5 of {path}: (bad)"));
    }
}
