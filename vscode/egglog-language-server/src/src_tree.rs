use tower_lsp::lsp_types::*;
use tree_sitter::{Node, Parser, Point, Query, QueryCursor, Tree};
use tree_sitter_traversal::traverse;

#[derive(Debug)]
pub struct SrcTree {
    src: String,
    // Must be always matched to `src`
    tree: Tree,
}

impl SrcTree {
    pub fn new(src: String) -> Self {
        let language = tree_sitter_egglog::language();
        let mut parser = Parser::new();
        parser
            .set_language(language)
            .expect("Error loading egglog language");

        let tree = parser
            .parse(&src, None)
            // Not possible
            .expect("Error parsing egglog source");

        Self { src, tree }
    }

    pub fn src(&self) -> &str {
        &self.src
    }

    pub fn tree(&self) -> &Tree {
        &self.tree
    }

    pub fn global_types(&self) -> Vec<String> {
        let queries = &[
            Query::new(
                tree_sitter_egglog::language(),
                r#"(command "datatype" (ident) @name)"#,
            )
            .unwrap(),
            Query::new(
                tree_sitter_egglog::language(),
                r#"(command "sort" (ident) @name)"#,
            )
            .unwrap(),
        ];

        queries
            .iter()
            .flat_map(|q| {
                let mut cursor = QueryCursor::new();
                cursor
                    .matches(q, self.tree.root_node(), self.src.as_bytes())
                    .map(|m| {
                        m.captures[0]
                            .node
                            .utf8_text(self.src.as_bytes())
                            .unwrap()
                            .to_string()
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    // names for other than types
    pub fn global_idents(&self) -> Vec<String> {
        let queries = &[
            Query::new(
                tree_sitter_egglog::language(),
                r#"(command "datatype" (variant (ident) @name))"#,
            )
            .unwrap(),
            Query::new(
                tree_sitter_egglog::language(),
                r#"(command "relation" (ident) @name)"#,
            )
            .unwrap(),
            Query::new(
                tree_sitter_egglog::language(),
                r#"(command "function" (ident) @name)"#,
            )
            .unwrap(),
            Query::new(
                tree_sitter_egglog::language(),
                r#"(command "let" (ident) @name)"#,
            )
            .unwrap(),
            Query::new(
                tree_sitter_egglog::language(),
                r#"(command "declare" (ident) @name)"#,
            )
            .unwrap(),
        ];

        queries
            .iter()
            .flat_map(|q| {
                let mut cursor = QueryCursor::new();
                cursor
                    .matches(q, self.tree.root_node(), self.src.as_bytes())
                    .map(|m| {
                        m.captures[0]
                            .node
                            .utf8_text(self.src.as_bytes())
                            .unwrap()
                            .to_string()
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    pub fn globals_all(&self) -> Vec<String> {
        self.global_types()
            .into_iter()
            .chain(self.global_idents())
            .collect()
    }

    pub fn formatted(&self, tab_width: usize) -> String {
        let src = &self.src;
        let tree = &self.tree;
        let root_node = tree.root_node();

        let mut buf = String::new();
        let cursor = root_node.walk();

        {
            // FIXME: Many ad-hocs, I don't like this codes.
            use std::fmt::Write;

            let mut emptyline = true;
            let mut last_kind = "";
            let mut paren_level = 0;
            for n in traverse(cursor, tree_sitter_traversal::Order::Pre)
                .filter(|n| n.child_count() == 0)
                .skip_while(|n| n.kind() == "ws")
            {
                let text = n.utf8_text(src.as_bytes()).unwrap();

                match text {
                    text if n.kind() == "lparen" => {
                        if last_kind == "lparen" {
                            write!(buf, "{}", text).unwrap();
                        } else if emptyline {
                            for _ in 0..tab_width * paren_level {
                                write!(buf, " ").unwrap();
                            }
                            write!(buf, "{}", text).unwrap();
                        } else if last_kind == "rparen" && paren_level == 0 {
                            writeln!(buf).unwrap();
                            write!(buf, "{}", text).unwrap();
                        } else {
                            write!(buf, " {}", text).unwrap();
                        }
                        emptyline = false;
                        paren_level += 1;
                    }
                    text if n.kind() == "rparen" => {
                        paren_level = paren_level.saturating_sub(1);
                        if emptyline {
                            for _ in 0..tab_width * paren_level {
                                write!(buf, " ").unwrap();
                            }
                        }
                        write!(buf, "{}", text).unwrap();
                        emptyline = false;
                    }
                    text if n.kind() == "comment" => {
                        if emptyline {
                            for _ in 0..tab_width * paren_level {
                                write!(buf, " ").unwrap();
                            }
                        } else {
                            write!(buf, " ").unwrap();
                        }
                        writeln!(buf, "{}", text).unwrap();
                        emptyline = true;
                    }
                    text if n.kind() == "ws" => {
                        let newlines = text.chars().filter(|&c| c == '\n').count();
                        let n = if emptyline { 1 } else { 0 };

                        for _ in n..newlines {
                            writeln!(buf).unwrap();
                            emptyline = true;
                        }

                        if !emptyline {
                            continue;
                        }
                    }
                    text => {
                        if last_kind == "lparen" {
                            write!(buf, "{}", text).unwrap();
                        } else {
                            if emptyline {
                                for _ in 0..tab_width * paren_level {
                                    write!(buf, " ").unwrap();
                                }
                            } else {
                                write!(buf, " ").unwrap();
                            }
                            write!(buf, "{}", text).unwrap();
                        }
                        emptyline = false;
                    }
                }

                last_kind = n.kind();
            }
        }

        while buf.ends_with('\n') {
            buf.pop();
        }
        buf.push('\n');

        buf
    }

    pub fn diagnstics(&self) -> Vec<Diagnostic> {
        let src = &self.src;
        let tree = &self.tree;
        let root_node = tree.root_node();

        // Fixme: Better traverse
        traverse(root_node.walk(), tree_sitter_traversal::Order::Post)
            .filter(|n| n.is_error() || n.is_missing())
            .map(|node| {
                let start = node.start_position();
                let end = node.end_position();
                let message = if node.has_error() && node.is_missing() {
                    node.to_sexp()
                        .trim_start_matches('(')
                        .trim_end_matches(')')
                        .to_string()
                } else {
                    let mut cursor = node.walk();
                    format!(
                        "Unexpected token(s) {}",
                        node.children(&mut cursor)
                            .filter_map(|n| n.utf8_text(src.as_bytes()).ok())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                };
                Diagnostic {
                    range: Range {
                        start: Position {
                            line: start.row as u32,
                            character: start.column as u32,
                        },
                        end: Position {
                            line: end.row as u32,
                            character: end.column as u32,
                        },
                    },
                    severity: Some(DiagnosticSeverity::ERROR),
                    message,
                    ..Default::default()
                }
            })
            .collect()
    }

    pub fn definition(&self, ident: &str) -> Option<Node> {
        if ident.contains('"') || ident.contains('\\') {
            return None;
        }

        let queries = &[
            Query::new(
                tree_sitter_egglog::language(),
                &format!(
                    r#"(command "datatype" (ident) @name (#eq? @name "{}")) @command"#,
                    ident
                ),
            )
            .unwrap(),
            Query::new(
                tree_sitter_egglog::language(),
                &format!(
                    r#"(command "datatype" (variant (ident) @name) (#eq? @name "{}")) @command"#,
                    ident
                ),
            )
            .unwrap(),
            Query::new(
                tree_sitter_egglog::language(),
                &format!(
                    r#"(command "relation" (ident) @name (#eq? @name "{}")) @command"#,
                    ident
                ),
            )
            .unwrap(),
            Query::new(
                tree_sitter_egglog::language(),
                &format!(
                    r#"(command "function" (ident) @name (#eq? @name "{}")) @command"#,
                    ident
                ),
            )
            .unwrap(),
            Query::new(
                tree_sitter_egglog::language(),
                &format!(
                    r#"(command "let" (ident) @name (#eq? @name "{}")) @command"#,
                    ident
                ),
            )
            .unwrap(),
            Query::new(
                tree_sitter_egglog::language(),
                &format!(
                    r#"(command "sort" (ident) @name (#eq? @name "{}")) @command"#,
                    ident
                ),
            )
            .unwrap(),
            Query::new(
                tree_sitter_egglog::language(),
                &format!(
                    r#"(command "declare" (ident) @name (#eq? @name "{}")) @command"#,
                    ident
                ),
            )
            .unwrap(),
        ];

        for query in queries {
            let mut cursor = QueryCursor::new();
            let Some((capture, _)) = cursor
                .captures(query, self.tree.root_node(), self.src.as_bytes())
                .next()
            else {
                continue;
            };

            let m = capture.captures[0].node;

            return Some(m);
        }
        None
    }

    pub fn completion(&self, pos: Point) -> Vec<CompletionItem> {
        fn is_root_command(mut node: Node) -> bool {
            while node.kind() == "lparen" || node.kind() == "rparen" || node.kind() == "ws" {
                if let Some(n) = node.prev_sibling() {
                    node = n;
                } else {
                    break;
                }
            }

            node.prev_sibling().is_none() && {
                if let Some(p) = node.parent() {
                    p.is_error()
                        && (p.parent().map(|p| p.kind()) == Some("source_file") || {
                            if let Some(mut p) = p.next_sibling() {
                                if p.kind() == "ws" {
                                    if let Some(q) = p.next_sibling() {
                                        p = q;
                                    } else {
                                        return false;
                                    }
                                }

                                p.is_error()
                            } else {
                                false
                            }
                        })
                } else {
                    false
                }
            }
        }

        const BUILTIN_TYPES: &[&str] = &[
            // types
            "Unit", "bool", "i64", "f64", "Map", "Rational", "String",
        ];
        const BUILTIN: &[&str] = &[
            // functions
            "map",
            "rational",
            // i64
            "+",
            "-",
            "*",
            "/",
            "%",
            "&",
            "|",
            "<<",
            ">>",
            "not-i64",
            "<",
            ">",
            "<=",
            ">=",
            "min",
            "max",
            "log2",
            "to-f64",
            "to-string",
            // f64
            "neg",
            "to-i64",
            // map
            "empty",
            "insert",
            "get",
            "not-contains",
            "contains",
            "set-uniton",
            "set-diff",
            "set-intersect",
            "map-remove",
            // rational
            "abs",
            "pow",
            "log",
            "sqrt",
        ];

        let root = self.tree.root_node();

        let mut node = root.named_descendant_for_point_range(pos, pos).unwrap();

        if node.kind() == "rparen" {
            if let Some(p) = node.prev_sibling() {
                node = p;
            }
        }

        if node
            .utf8_text(self.src.as_bytes())
            .unwrap()
            .starts_with(':')
        {
            // Attributes
            if let Some(command) = node.parent() {
                if let Some(node) = command.child(1) {
                    let command_name = node.utf8_text(self.src.as_bytes()).unwrap();

                    let attrs: &[&str] = match command_name {
                        "function" => &["cost", "unextractable", "on_merge", "merge", "default"],
                        "rule" => &["ruleset", "name"],
                        "rewrite" | "birewrite" => &["when", "ruleset"],
                        "run" => &["until"],
                        "query-extract" => &["variants"],
                        _ => return Vec::new(),
                    };

                    return attrs
                        .iter()
                        .map(|a| CompletionItem {
                            label: a.to_string(),
                            kind: Some(CompletionItemKind::FIELD),
                            ..Default::default()
                        })
                        .collect();
                }
            }
        } else if node.prev_sibling().is_some() {
            // Triggerd by space
            // Completion global variables

            if node.parent().map(|p| p.kind()) == Some("variant") {
                // complete types
                let global_types = self.global_types();

                return global_types
                    .iter()
                    .map(|s| s.as_str())
                    .chain(BUILTIN_TYPES.iter().copied())
                    .map(|s| CompletionItem {
                        label: s.to_string(),
                        kind: Some(CompletionItemKind::CLASS),
                        ..Default::default()
                    })
                    .collect();
            } else {
                let globals = self.globals_all();
                return globals
                    .iter()
                    .map(|s| s.as_str())
                    .map(|s| CompletionItem {
                        label: s.to_string(),
                        kind: Some(CompletionItemKind::FUNCTION),
                        ..Default::default()
                    })
                    .collect();
            }
        } else if is_root_command(node) {
            // Completion keywords
            const KEYWORDS: &[&str] = &[
                "set-option",
                "datatype",
                "sort",
                "function",
                "declare",
                "relation",
                "ruleset",
                "rule",
                "rewrite",
                "birewrite",
                "let",
                "run",
                "simplify",
                "calc",
                "query-extract",
                "check",
                "check-proof",
                "run-schedule",
                "print-stats",
                "push",
                "pop",
                "print-function",
                "print-size",
                "input",
                "output",
                "fail",
                "include",
                "set",
                "delete",
                "union",
                "panic",
                "extract",
            ];

            let globals = self.globals_all();

            return KEYWORDS
                .iter()
                .map(|k| CompletionItem {
                    label: k.to_string(),
                    kind: Some(CompletionItemKind::KEYWORD),
                    ..Default::default()
                })
                .chain(globals.iter().map(|s| s.as_str()).map(|s| CompletionItem {
                    label: s.to_string(),
                    kind: Some(CompletionItemKind::FUNCTION),
                    ..Default::default()
                }))
                .collect();
        } else {
            let globals = self.globals_all();
            return globals
                .iter()
                .map(|s| s.as_str())
                .chain(BUILTIN.iter().copied())
                .map(|s| CompletionItem {
                    label: s.to_string(),
                    kind: Some(CompletionItemKind::FUNCTION),
                    ..Default::default()
                })
                .collect();
        }

        Vec::new()
    }

    pub fn includes(&self) -> Vec<String> {
        let query = Query::new(
            tree_sitter_egglog::language(),
            r#"(command "include" (string) @path)"#,
        )
        .unwrap();

        let mut cursor = QueryCursor::new();
        cursor
            .captures(&query, self.tree.root_node(), self.src.as_bytes())
            .map(|(capture, _)| {
                capture.captures[0]
                    .node
                    .utf8_text(self.src.as_bytes())
                    .unwrap()
                    .trim_start_matches('"')
                    .trim_end_matches('"')
                    .to_string()
            })
            .collect()
    }
}

#[test]
fn test_diagnostic() {
    fn has_error(src: String) -> bool {
        let src_tree = SrcTree::new(src);

        !src_tree.diagnstics().is_empty()
    }

    assert!(has_error("(sort)".to_string()));
    assert!(!has_error("(sort X)".to_string()));

    assert!(has_error("(declare)".to_string()));
    assert!(has_error("(declare x)".to_string()));
    assert!(!has_error("(declare x T)".to_string()));
}
