use std::collections::HashSet;
use std::process::Stdio;
use std::sync::RwLock;

use dashmap::mapref::one::Ref;
use dashmap::DashMap;
use tower_lsp::jsonrpc::{Error, Result};
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};
use tree_sitter::Node;
use tree_sitter_highlight::{Highlight, HighlightConfiguration, HighlightEvent, Highlighter};

mod src_tree;

use src_tree::SrcTree;

// Token types for language server side
pub const LSP_LEGEND_TYPE: &[SemanticTokenType] = &[
    SemanticTokenType::KEYWORD,
    SemanticTokenType::FUNCTION,
    SemanticTokenType::TYPE,
    SemanticTokenType::DECORATOR,
    SemanticTokenType::VARIABLE,
    SemanticTokenType::NUMBER,
    SemanticTokenType::STRING,
    SemanticTokenType::COMMENT,
    SemanticTokenType::OPERATOR,
    SemanticTokenType::PARAMETER,
];
// Token types for Tree-sitter side
// Both must match indices
const TS_HIGHLIGHT_NAMES: &[&str] = &[
    "keyword",
    "function",
    "type",
    "attribute",
    "variable",
    "number",
    "string",
    "comment",
];

#[derive(Debug)]
struct Backend {
    client: Client,
    workspace: RwLock<Option<Url>>,
    document_map: DashMap<Url, SrcTree>,
}

struct TextDocumentItem {
    uri: Url,
    text: String,
    version: i32,
}

struct Definition {
    src: String,
    url: Url,
    range: Range,
}

fn desugar(src: &str) -> anyhow::Result<String> {
    use std::io::Write;

    let mut child = std::process::Command::new("egglog")
        .args(["--desugar", "/dev/stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    write!(child.stdin.take().unwrap(), "{}", src)?;

    let output = child.wait_with_output()?;

    if !output.status.success() {
        anyhow::bail!("child proccess fail");
    }
    Ok(String::from_utf8(output.stdout)?)
}

impl Backend {
    async fn on_change(&self, params: TextDocumentItem) {
        // TODO: Support incremental update
        let src_tree = SrcTree::new(params.text);
        let diagnostics = src_tree.diagnstics();

        self.document_map.insert(params.uri.clone(), src_tree);
        self.client
            .publish_diagnostics(params.uri, diagnostics, Some(params.version))
            .await;
    }

    fn url(&self, path: &str) -> Option<Url> {
        self.workspace
            .read()
            .unwrap()
            .as_ref()
            .and_then(|root| root.join(path).ok())
    }

    fn load(&self, url: &Url) -> Option<Ref<Url, SrcTree>> {
        self.document_map
            .entry(url.clone())
            .or_try_insert_with(|| {
                let src = std::fs::read_to_string(url.path())?;
                let src_tree = SrcTree::new(src);
                std::io::Result::Ok(src_tree)
            })
            .map(|rm| rm.downgrade())
            .ok()
    }

    fn definition(&self, url: Url, ident: &str) -> Option<Definition> {
        let mut visited = HashSet::new();

        let mut stack = vec![url];

        while let Some(url) = stack.pop() {
            if visited.contains(&url) {
                continue;
            }
            if let Some(src_tree) = self.load(&url) {
                if let Some(node) = src_tree.definition(ident) {
                    let src = src_tree.src();
                    let src = node.utf8_text(src.as_bytes()).unwrap().to_string();
                    let start = node.start_position();
                    let end = node.end_position();
                    let range = Range {
                        start: Position {
                            line: start.row as _,
                            character: start.column as _,
                        },
                        end: Position {
                            line: end.row as _,
                            character: end.column as _,
                        },
                    };

                    return Some(Definition { src, url, range });
                }

                for path in src_tree.includes() {
                    if let Some(url) = self.url(&path) {
                        stack.push(url);
                    }
                }
            }
            visited.insert(url);
        }

        None
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, params: InitializeParams) -> Result<InitializeResult> {
        *self.workspace.write().unwrap() = params
            .workspace_folders
            .and_then(|v| v.get(0).map(|w| w.uri.clone()))
            .map(|url| {
                let mut s = url.to_string();
                if !s.ends_with('/') {
                    s.push('/');
                }
                Url::parse(s.as_str()).unwrap()
            });

        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                // diagnostic_provider: Some(DiagnosticServerCapabilities::Simple(true)),
                semantic_tokens_provider: Some(
                    SemanticTokensServerCapabilities::SemanticTokensRegistrationOptions(
                        SemanticTokensRegistrationOptions {
                            text_document_registration_options: {
                                TextDocumentRegistrationOptions {
                                    document_selector: Some(vec![DocumentFilter {
                                        language: Some("egglog".to_string()),
                                        scheme: Some("file".to_string()),
                                        pattern: None,
                                    }]),
                                }
                            },
                            semantic_tokens_options: SemanticTokensOptions {
                                work_done_progress_options: WorkDoneProgressOptions::default(),
                                legend: SemanticTokensLegend {
                                    token_types: LSP_LEGEND_TYPE.into(),
                                    token_modifiers: vec![],
                                },
                                range: Some(false),
                                full: Some(SemanticTokensFullOptions::Bool(true)),
                            },
                            static_registration_options: StaticRegistrationOptions::default(),
                        },
                    ),
                ),
                document_formatting_provider: Some(OneOf::Left(true)),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                completion_provider: Some(CompletionOptions {
                    resolve_provider: Some(false),
                    trigger_characters: Some(vec![
                        "(".to_string(),
                        " ".to_string(),
                        ":".to_string(),
                    ]),
                    work_done_progress_options: Default::default(),
                    all_commit_characters: None,
                    completion_item: None,
                }),
                definition_provider: Some(OneOf::Left(true)),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "server initialized!")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, "file opened!")
            .await;
        let _ = self
            .on_change(TextDocumentItem {
                uri: params.text_document.uri,
                text: params.text_document.text,
                version: params.text_document.version,
            })
            .await;
    }

    async fn did_change(&self, mut params: DidChangeTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, "did_change")
            .await;
        let _ = self
            .on_change(TextDocumentItem {
                uri: params.text_document.uri,
                text: std::mem::take(&mut params.content_changes[0].text),
                version: params.text_document.version,
            })
            .await;
    }

    async fn did_save(&self, _: DidSaveTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, "file saved!")
            .await;
    }
    async fn did_close(&self, param: DidCloseTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, "file closed!")
            .await;

        self.document_map.remove(&param.text_document.uri);
    }

    async fn semantic_tokens_full(
        &self,
        params: SemanticTokensParams,
    ) -> Result<Option<SemanticTokensResult>> {
        self.client
            .log_message(MessageType::INFO, "semantic_tokens_full")
            .await;

        let mut highlighter = Highlighter::new();
        let language = tree_sitter_egglog::language();
        let mut language_config =
            HighlightConfiguration::new(language, tree_sitter_egglog::HIGHLIGHTS_QUERY, "", "")
                .map_err(|_| Error::invalid_params("Failed to load highlighter"))?;

        language_config.configure(TS_HIGHLIGHT_NAMES);

        let src_tree = self
            .document_map
            .get(&params.text_document.uri)
            .ok_or_else(Error::internal_error)?;

        let tree = src_tree.tree();
        let root_node = tree.root_node();

        let highlights = highlighter
            .highlight(&language_config, src_tree.src().as_bytes(), None, |_| None)
            .map_err(|_| Error::internal_error())?;

        let mut current_hightlight: Option<Highlight> = None;
        let mut pre_line = 0;
        let mut pre_start = 0;
        let semantic_tokens = highlights
            .filter_map(|h| h.ok())
            .filter_map(|h| match h {
                HighlightEvent::Source { start, end } => {
                    let node = root_node.descendant_for_byte_range(start, end).unwrap();

                    let s = start;
                    let e = end;
                    let line = node.start_position().row as u32;
                    let start = node.start_position().column as u32;
                    let delta_line = line - pre_line;
                    let delta_start = if delta_line == 0 {
                        start - pre_start
                    } else {
                        start
                    };
                    if let Some(t) = current_hightlight {
                        let ret = Some(SemanticToken {
                            delta_line,
                            delta_start,
                            length: (e - s) as u32,
                            token_type: t.0 as u32,
                            token_modifiers_bitset: 0,
                        });
                        pre_line = line;
                        pre_start = start;
                        ret
                    } else {
                        None
                    }
                }
                HighlightEvent::HighlightStart(highlight) => {
                    current_hightlight = Some(highlight);
                    None
                }
                HighlightEvent::HighlightEnd => {
                    current_hightlight = None;
                    None
                }
            })
            .collect::<Vec<_>>();

        if semantic_tokens.is_empty() {
            Ok(None)
        } else {
            Ok(Some(SemanticTokensResult::Tokens(SemanticTokens {
                result_id: None,
                data: semantic_tokens,
            })))
        }
    }

    async fn formatting(&self, params: DocumentFormattingParams) -> Result<Option<Vec<TextEdit>>> {
        let src_tree = self
            .document_map
            .get(&params.text_document.uri)
            .ok_or_else(Error::internal_error)?;
        let fmt = src_tree.formatted(params.options.tab_size as usize);

        let lines = src_tree.tree().root_node().end_position().row;

        Ok(Some(vec![TextEdit {
            range: Range {
                start: Position {
                    line: 0,
                    character: 0,
                },
                end: Position {
                    line: lines as u32 + 1,
                    character: 0,
                },
            },
            new_text: fmt,
        }]))
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        fn desugar_node(mut node: Node, src: &str) -> Option<String> {
            while node.kind() != "command" {
                let Some(p) = node.parent() else {
                    return None;
                };

                node = p;
            }

            let src = format!(
                "{}\n(let __marker 0)\n{}",
                &src[..node.start_byte()],
                &src[node.start_byte()..node.end_byte()]
            );

            desugar(&src).ok().map(|s| {
                let mut lines = s.lines();
                for line in lines.by_ref() {
                    if line.contains("__marker") {
                        break;
                    }
                }
                lines.collect::<Vec<_>>().join("\n")
            })
        }

        let pos = params.text_document_position_params;

        let src_tree = self
            .document_map
            .get(&pos.text_document.uri)
            .ok_or_else(|| Error::invalid_params("unknown uri"))?;

        let root = src_tree.tree().root_node();

        let posisiton = tree_sitter::Point {
            row: pos.position.line as _,
            column: pos.position.character as _,
        };

        let node = root
            .descendant_for_point_range(posisiton, posisiton)
            .ok_or_else(|| Error::invalid_params("Postion out of range"))?;

        let mut markdown = String::new();

        let desugar_result = desugar_node(node, src_tree.src());

        let ident = if node.kind() == "ident" || node.kind() == "type" {
            Some(
                node.utf8_text(src_tree.src().as_bytes())
                    .unwrap()
                    .to_string(),
            )
        } else {
            None
        };
        drop(src_tree);

        if let Some(definition) =
            ident.and_then(|ident| self.definition(pos.text_document.uri.clone(), &ident))
        {
            markdown.push_str(&format!(
                "#### Definition\n\n```egglog\n{}\n```\n",
                definition.src
            ));
        }

        if let Some(desugared) = desugar_result {
            markdown.push_str(&format!("#### Desugar\n\n```egglog\n{}\n```", desugared));
        }

        if markdown.is_empty() {
            Ok(None)
        } else {
            Ok(Some(Hover {
                contents: HoverContents::Scalar(MarkedString::String(markdown)),
                range: None,
            }))
        }
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        self.client
            .log_message(MessageType::INFO, "completion")
            .await;

        let uri = &params.text_document_position.text_document.uri;

        let src_tree = self
            .document_map
            .get(uri)
            .ok_or_else(Error::internal_error)?;

        let pos = tree_sitter::Point {
            row: params.text_document_position.position.line as _,
            column: params.text_document_position.position.character as _,
        };

        let items = src_tree.completion(pos);

        if items.is_empty() {
            Ok(None)
        } else {
            Ok(Some(CompletionResponse::Array(items)))
        }
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        self.client
            .log_message(MessageType::INFO, "goto_definition")
            .await;

        let src_tree = self
            .document_map
            .get(&params.text_document_position_params.text_document.uri)
            .ok_or_else(|| Error::invalid_params("unknown uri"))?;

        let root = src_tree.tree().root_node();
        let p = tree_sitter::Point {
            row: params.text_document_position_params.position.line as _,
            column: params.text_document_position_params.position.character as _,
        };
        let Some(node) = root.descendant_for_point_range(p, p) else {
            return Ok(None);
        };

        if node.kind() != "ident" && node.kind() != "type" {
            return Ok(None);
        }

        let ident = node
            .utf8_text(src_tree.src().as_bytes())
            .unwrap()
            .to_string();

        drop(src_tree);

        if let Some(def) = self.definition(
            params.text_document_position_params.text_document.uri,
            &ident,
        ) {
            Ok(Some(GotoDefinitionResponse::Scalar(Location {
                uri: def.url,
                range: def.range,
            })))
        } else {
            Ok(None)
        }
    }
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| Backend {
        client,
        workspace: RwLock::new(None),
        document_map: DashMap::new(),
    });
    Server::new(stdin, stdout, socket).serve(service).await;
}
