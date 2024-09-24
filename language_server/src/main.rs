use async_lsp::{
    client_monitor::ClientProcessMonitorLayer, concurrency::ConcurrencyLayer, lsp_types::*,
    panic::CatchUnwindLayer, router::Router, server::LifecycleLayer, tracing::TracingLayer,
    ClientSocket, LanguageClient, LanguageServer, ResponseError,
};
use futures::future::BoxFuture;
use zeque_lexer::Token;
use std::{
    collections::BTreeMap,
    ops::ControlFlow::{self, Break, Continue},
    time::Duration,
};
use tower::ServiceBuilder;
use tracing::Level;

// mod lexeme;
mod lexeme2;
// pub use lexeme2::LexemeTree;
// mod buffered_rope;

// mod fast_replace_string;
mod gap_buffer;
mod text_buffer;
mod unit;

struct TextDocument {
    content: text_buffer::Lines,
    tokens: Vec<Token>,
    token_lengths: Vec<u32>,
    // lexemes: LexemeTree,
    version: i32,
    language_id: String,
}

impl TextDocument {
    fn new(content: &str, version: i32, language_id: String) -> Self {
        // eprintln!("TextDocument::new, content=```\n{}\n```", content);
        TextDocument {
            content: content.into(),
            tokens: vec![],
            token_lengths: vec![],
            version,
            language_id,
        }
    }

    fn apply_change(&mut self, change_event: TextDocumentContentChangeEvent) {
        // let mut tokens = Vec::with_capacity(64);
        // let start = std::time::Instant::now();
        let Some(range) = change_event.range else {
            self.content.replace_entire_document(&change_event.text);
            return;
        };

        // eprintln!(
        //     "TextDocument::apply_change, replacement=```\n{}\n```",
        //     change_event.text
        // );
        let edit = self.content.edit(
            unit::Unit::new(range.start.line as usize),
            unit::Unit::new(range.end.line as usize),
            unit::Offset::new(range.start.character as usize),
            unit::Offset::new(range.end.character as usize),
            &change_event.text,
        );

        eprintln!("{edit:#?}");

        // tokens.extend(Tokens::new(edit.edited_lines));
        //
        // self.content.edit(
        //     range.start.line as usize,
        //     range.start.character as usize,
        //     range.end.line as usize,
        //     range.end.character as usize,
        //     &change_event.text,
        //     |s| {
        //         tokens.extend(Tokens::new(s));
        //     },
        // );
        // let _end = start.elapsed();
        // eprintln!("{end:?}: {tokens:?}");
    }
}

struct ServerState {
    client: ClientSocket,
    documents: BTreeMap<Url, TextDocument>,
}

impl ServerState {
    fn new_router(client: ClientSocket) -> Router<Self> {
        Router::from_language_server(Self {
            client,
            documents: BTreeMap::new(),
        })
    }
}

impl LanguageServer for ServerState {
    type Error = ResponseError;
    type NotifyResult = ControlFlow<async_lsp::Result<()>>;

    fn initialize(
        &mut self,
        _params: InitializeParams,
    ) -> BoxFuture<'static, Result<InitializeResult, Self::Error>> {
        Box::pin(async move {
            Ok(InitializeResult {
                capabilities: ServerCapabilities {
                    hover_provider: Some(HoverProviderCapability::Simple(true)),
                    definition_provider: Some(OneOf::Left(true)),
                    text_document_sync: Some(TextDocumentSyncCapability::Kind(
                        TextDocumentSyncKind::INCREMENTAL,
                    )),
                    completion_provider: Some(CompletionOptions {
                        resolve_provider: Some(true),
                        ..Default::default()
                    }),
                    diagnostic_provider: Some(DiagnosticServerCapabilities::Options(
                        DiagnosticOptions {
                            inter_file_dependencies: false,
                            workspace_diagnostics: false,
                            ..Default::default()
                        },
                    )),
                    ..Default::default()
                },
                server_info: None,
            })
        })
    }

    fn hover(&mut self, _: HoverParams) -> BoxFuture<'static, Result<Option<Hover>, Self::Error>> {
        let mut client = self.client.clone();
        Box::pin(async move {
            tokio::time::sleep(Duration::from_secs(1)).await;
            client
                .show_message(ShowMessageParams {
                    typ: MessageType::INFO,
                    message: "Hello LSP".into(),
                })
                .unwrap();
            Ok(Some(Hover {
                contents: HoverContents::Scalar(MarkedString::String(format!(
                    "I am a hover text!"
                ))),
                range: None,
            }))
        })
    }

    fn definition(
        &mut self,
        _: GotoDefinitionParams,
    ) -> BoxFuture<'static, Result<Option<GotoDefinitionResponse>, ResponseError>> {
        unimplemented!("Not yet implemented!");
    }

    fn did_change_configuration(
        &mut self,
        _: DidChangeConfigurationParams,
    ) -> ControlFlow<async_lsp::Result<()>> {
        Continue(())
    }

    fn did_open(&mut self, params: DidOpenTextDocumentParams) -> Self::NotifyResult {
        let text_document = TextDocument::new(
            params.text_document.text.as_str(),
            params.text_document.version,
            params.text_document.language_id,
        );
        self.documents
            .insert(params.text_document.uri, text_document);
        Continue(())
    }

    fn did_change(&mut self, params: DidChangeTextDocumentParams) -> Self::NotifyResult {
        let url = &params.text_document.uri;
        let Some(document) = self.documents.get_mut(url) else {
            return Break(Err(async_lsp::Error::Protocol(format!(
                "Never opened `{url}`"
            ))));
        };

        document.version = params.text_document.version;
        for change_event in params.content_changes {
            document.apply_change(change_event);
        }
        Continue(())
    }

    fn document_diagnostic(
        &mut self,
        _: DocumentDiagnosticParams,
    ) -> BoxFuture<'static, Result<DocumentDiagnosticReportResult, Self::Error>> {
        Box::pin(async move {
            Ok(DocumentDiagnosticReportResult::Report(
                DocumentDiagnosticReport::Full(RelatedFullDocumentDiagnosticReport {
                    related_documents: None,
                    full_document_diagnostic_report: FullDocumentDiagnosticReport {
                        result_id: None,
                        items: vec![],
                    },
                }),
            ))
        })
    }

    fn did_save(&mut self, _: DidSaveTextDocumentParams) -> Self::NotifyResult {
        Continue(())
    }

    fn did_close(&mut self, params: DidCloseTextDocumentParams) -> Self::NotifyResult {
        if self.documents.remove(&params.text_document.uri).is_none() {
            return Break(Err(async_lsp::Error::Protocol(
                "closing document that wasn't opened".to_string(),
            )));
        }
        Continue(())
    }

    fn completion(
        &mut self,
        _: CompletionParams,
    ) -> BoxFuture<'static, Result<Option<CompletionResponse>, Self::Error>> {
        Box::pin(async move {
            let item = CompletionItem {
                label: "pookie".to_string(),
                ..Default::default()
            };
            Ok(Some(CompletionResponse::Array(vec![item])))
        })
    }

    fn completion_item_resolve(
        &mut self,
        params: CompletionItem,
    ) -> BoxFuture<'static, Result<CompletionItem, Self::Error>> {
        Box::pin(async move { Ok(params) })
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let (server, _) = async_lsp::MainLoop::new_server(|client| {
        ServiceBuilder::new()
            .layer(TracingLayer::default())
            .layer(LifecycleLayer::default())
            .layer(CatchUnwindLayer::default())
            .layer(ConcurrencyLayer::default())
            .layer(ClientProcessMonitorLayer::new(client.clone()))
            .service(ServerState::new_router(client))
    });

    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .init();

    // Prefer truly asynchronous piped stdin/stdout without blocking tasks.
    #[cfg(unix)]
    let (stdin, stdout) = (
        async_lsp::stdio::PipeStdin::lock_tokio().unwrap(),
        async_lsp::stdio::PipeStdout::lock_tokio().unwrap(),
    );
    // Fallback to spawn blocking read/write otherwise.
    #[cfg(not(unix))]
    let (stdin, stdout) = (
        tokio_util::compat::TokioAsyncReadCompatExt::compat(tokio::io::stdin()),
        tokio_util::compat::TokioAsyncWriteCompatExt::compat_write(tokio::io::stdout()),
    );

    server.run_buffered(stdin, stdout).await.unwrap();
}
