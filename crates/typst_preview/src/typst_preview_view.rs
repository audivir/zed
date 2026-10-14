use crate::typst_world::TypstSystemWorld;
use crate::{
    ExportPdf, OpenFixedFollowingPreview, OpenFixedPreview, OpenFixedPreviewToTheSide,
    OpenFollowingPreview, OpenPreview, OpenPreviewToTheSide,
};
use anyhow::{Context as _, Result};
use editor::scroll::Autoscroll;
use editor::{Editor, EditorEvent, MultiBufferOffset, SelectionEffects};
use gpui::{
    AnyEntity, App, Bounds, Context, Entity, EntityId, EventEmitter, FocusHandle, Focusable, Font,
    IntoElement, Point, Render, Size, Subscription, Task, WeakEntity, Window, div,
};
use language::{
    Diagnostic, DiagnosticEntry, DiagnosticMessage, DiagnosticSet, DiagnosticSeverity,
    DiagnosticSourceKind, HighlightedText, LanguageServerId, PointUtf16,
};
use parking_lot::Mutex;
use pdf_viewer::pdf::Pdf;
use pdf_viewer::pdf_renderer::{PageMetadata, PdfTarget};
use pdf_viewer::{PdfView, ScrollAnchor, ScrollMode};
use project::lsp_store::DocumentDiagnosticsUpdate;
use std::any::TypeId;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use text::Point as BufferPoint;
use typst::World;
use typst::layout::{Abs, Frame, FrameItem, Point as TypstPoint};
use typst::syntax::package::PackageSpec;
use typst::syntax::{LinkedNode, RootedPath, Side, Source, Span, SyntaxKind, VirtualRoot};
use typst_layout::PagedDocument;
use ui::prelude::*;
use util::ResultExt as _;
use workspace::item::{Item, ItemBufferKind, ItemHandle};
use workspace::searchable::{SearchEvent, SearchableItemHandle};
use workspace::{Pane, SplitDirection, ToolbarItemLocation, Workspace};

const REPARSE_DEBOUNCE: Duration = Duration::from_millis(1000);

pub struct TypstPreviewView {
    workspace: WeakEntity<Workspace>,
    active_editor: Option<EditorState>,
    observed_editors: HashMap<EntityId, Subscription>,
    focus_handle: FocusHandle,
    pdf_view: Entity<PdfView>,
    world: Option<Arc<Mutex<TypstSystemWorld>>>,
    document: Option<Arc<PagedDocument>>,
    compile_state: TypstCompileState,
    /// Warnings from the last successful compile, which do not fail compilation.
    compile_warnings: Vec<String>,
    pending_update_task: Option<Task<Result<()>>>,
    mode: TypstPreviewMode,
    project_resolution: bool,
    /// Packages already downloaded or attempted, so a failing download is not retried on every
    /// compile.
    requested_packages: HashSet<PackageSpec>,
    package_download_task: Option<Task<()>>,
    /// Files that currently show diagnostics from this preview, cleared on the next compile.
    published_diagnostic_paths: HashSet<PathBuf>,
    /// Id of the `Pdf` shown in the preview and the hashes of the Typst pages it was made from.
    rendered_page_hashes: Option<(u64, Vec<u128>)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TypstPreviewMode {
    /// Always shows the contents of the provided editor.
    Default,
    /// Follows the active editor.
    Follow,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TypstCompileState {
    Uninitialized,
    Compiling,
    Rendering,
    Finished,
    Failed(String),
    DownloadingPackages(Vec<PackageSpec>),
}

/// Language server ID for diagnostics published by the Typst preview itself.
///
/// Kept distinct from any real Typst language server so neither clobbers the diagnostics of the
/// other on the same buffer.
const TYPST_PREVIEW_DIAGNOSTICS_SERVER_ID: LanguageServerId = LanguageServerId(usize::MAX - 100);

/// Container for a Typst compile diagnostic with its resolved file path and byte range.
///
/// `file_path` and `range` are `None` for spanless diagnostics, which show without a squiggle.
struct TypstDiagnostic {
    severity: DiagnosticSeverity,
    message: String,
    file_path: Option<PathBuf>,
    range: Option<Range<usize>>,
    /// `range` as UTF-16 points, for files without an open buffer to convert offsets.
    point_range: Option<Range<PointUtf16>>,
}

/// Hashes each page so renders of pages that did not change between compiles can be reused.
fn page_hashes(document: &PagedDocument) -> Vec<u128> {
    document.pages().iter().map(typst::utils::hash128).collect()
}

/// Resolves the span of each diagnostic to a file path and byte range.
///
/// Must run right after `typst::compile`, since spans are meaningless once `world` changes.
fn resolve_typst_diagnostics(
    world: &TypstSystemWorld,
    diagnostics: &[typst::diag::SourceDiagnostic],
) -> Vec<TypstDiagnostic> {
    diagnostics
        .iter()
        .map(|diagnostic| {
            let severity = match diagnostic.severity {
                typst::diag::Severity::Error => DiagnosticSeverity::ERROR,
                typst::diag::Severity::Warning => DiagnosticSeverity::WARNING,
            };
            let location = diagnostic.span.id().and_then(|file_id| {
                let path = world.resolve_path(file_id).ok()?;
                let range = typst::WorldExt::range(world, diagnostic.span)?;
                let point_range = world.source(file_id).ok().and_then(|source| {
                    Some(
                        byte_to_point_utf16(&source, range.start)?
                            ..byte_to_point_utf16(&source, range.end)?,
                    )
                });
                Some((path, range, point_range))
            });
            let (file_path, range, point_range) = match location {
                Some((path, range, point_range)) => (Some(path), Some(range), point_range),
                None => (None, None, None),
            };
            TypstDiagnostic {
                severity,
                message: diagnostic.message.to_string(),
                file_path,
                range,
                point_range,
            }
        })
        .collect()
}

fn byte_to_point_utf16(source: &Source, byte: usize) -> Option<PointUtf16> {
    let lines = source.lines();
    let line = lines.byte_to_line(byte)?;
    let line_start = lines.line_to_byte(line)?;
    let column = lines.byte_to_utf16(byte)? - lines.byte_to_utf16(line_start)?;
    Some(PointUtf16::new(line as u32, column as u32))
}

/// Publishes `diagnostics` to the project, so they show in open buffers, the status bar, and the
/// diagnostics panel, including those in imported files.
///
/// Clears the diagnostics of files in `published_paths` that no longer have any and records the
/// newly published files there. Remote projects fall back to the buffer of `editor`.
fn publish_project_diagnostics(
    workspace: &WeakEntity<Workspace>,
    editor: Option<&Entity<Editor>>,
    diagnostics: &[TypstDiagnostic],
    published_paths: &mut HashSet<PathBuf>,
    cx: &mut App,
) {
    let Some(project) = workspace
        .upgrade()
        .map(|workspace| workspace.read(cx).project().clone())
    else {
        return;
    };
    if !project.read(cx).is_local() {
        publish_editor_diagnostics(editor, diagnostics, cx);
        return;
    }

    let mut diagnostics_by_path: HashMap<PathBuf, Vec<lsp::Diagnostic>> = HashMap::default();
    for diagnostic in diagnostics {
        let (Some(path), Some(point_range)) = (&diagnostic.file_path, &diagnostic.point_range)
        else {
            continue;
        };
        let position = |point: PointUtf16| lsp::Position::new(point.row, point.column);
        let severity = if diagnostic.severity == DiagnosticSeverity::ERROR {
            lsp::DiagnosticSeverity::ERROR
        } else {
            lsp::DiagnosticSeverity::WARNING
        };
        diagnostics_by_path
            .entry(path.clone())
            .or_default()
            .push(lsp::Diagnostic {
                range: lsp::Range::new(position(point_range.start), position(point_range.end)),
                severity: Some(severity),
                source: Some("Typst".to_string()),
                message: lsp::DiagnosticMessage::String(diagnostic.message.clone()),
                ..Default::default()
            });
    }
    for path in published_paths.drain() {
        diagnostics_by_path.entry(path).or_default();
    }

    let mut updates = Vec::new();
    for (path, diagnostics) in diagnostics_by_path {
        let Ok(uri) = lsp::Uri::from_file_path(&path) else {
            continue;
        };
        if !diagnostics.is_empty() {
            published_paths.insert(path);
        }
        updates.push(DocumentDiagnosticsUpdate {
            diagnostics: lsp::PublishDiagnosticsParams {
                uri,
                diagnostics,
                version: None,
            },
            result_id: None,
            registration_id: None,
            server_id: TYPST_PREVIEW_DIAGNOSTICS_SERVER_ID,
            disk_based_sources: Cow::Borrowed(&[]),
        });
    }
    project.read(cx).lsp_store().update(cx, |lsp_store, cx| {
        lsp_store
            .merge_lsp_diagnostics(DiagnosticSourceKind::Pushed, updates, |_, _, _| false, cx)
            .context("publishing Typst diagnostics")
            .log_err();
    });
}

/// Publishes `diagnostics` as squiggles on the buffer of `editor`, replacing the previous set.
///
/// Only diagnostics for the file of the editor apply, since remote projects accept no diagnostics
/// from the client.
fn publish_editor_diagnostics(
    editor: Option<&Entity<Editor>>,
    diagnostics: &[TypstDiagnostic],
    cx: &mut App,
) {
    let Some(editor) = editor else { return };
    let Some(buffer) = editor.read(cx).buffer().read(cx).as_singleton() else {
        return;
    };
    let Some(active_path) = buffer
        .read(cx)
        .file()
        .and_then(|file| file.as_local())
        .map(|local| local.abs_path(cx))
    else {
        return;
    };

    let snapshot = buffer.read(cx).snapshot();
    let entries: Vec<DiagnosticEntry<language::PointUtf16>> = diagnostics
        .iter()
        .filter(|d| d.file_path.as_deref() == Some(active_path.as_path()))
        .filter_map(|d| {
            let range = d.range.clone()?;
            Some(DiagnosticEntry::new(
                snapshot.offset_to_point_utf16(range.start)
                    ..snapshot.offset_to_point_utf16(range.end),
                Diagnostic {
                    source: Some("Typst".to_string()),
                    severity: d.severity,
                    message: DiagnosticMessage::plain(d.message.clone()),
                    is_primary: true,
                    ..Default::default()
                },
            ))
        })
        .enumerate()
        .map(|(index, mut entry)| {
            entry.diagnostic.group_id = index;
            entry
        })
        .collect();

    let diagnostic_set = DiagnosticSet::new(entries, &snapshot);
    buffer.update(cx, |buffer, cx| {
        buffer.update_diagnostics(TYPST_PREVIEW_DIAGNOSTICS_SERVER_ID, diagnostic_set, cx);
    });
}

/// Whether a sibling of `kind` within a paragraph still belongs to the same paragraph.
///
/// Returns `false` for paragraph breaks, headings, list items, and math blocks.
fn continues_same_paragraph(kind: SyntaxKind) -> bool {
    !matches!(
        kind,
        SyntaxKind::Parbreak
            | SyntaxKind::Heading
            | SyntaxKind::ListItem
            | SyntaxKind::EnumItem
            | SyntaxKind::TermItem
            | SyntaxKind::Equation
    )
}

/// Returns the node holding the paragraph flow of `leaf`, skipping inline formatting.
///
/// `*Strong*` and `_Emph_` nest their text one level deeper than the surrounding text, which
/// would otherwise split one visual paragraph into several lookups.
fn paragraph_container<'a>(leaf: &LinkedNode<'a>) -> Option<LinkedNode<'a>> {
    let mut container = leaf.parent()?.clone();
    while let Some(parent) = container.parent()
        && matches!(parent.kind(), SyntaxKind::Strong | SyntaxKind::Emph)
    {
        container = parent.parent()?.clone();
    }
    Some(container)
}

/// Returns the spans of all leaves in the paragraph containing `cursor`.
///
/// Includes nested `Strong` and `Emph` content, so the highlight does not depend on which side
/// of a formatted word was clicked. Returns `None` if `cursor` is not inside text.
fn paragraph_leaf_spans(source: &Source, cursor: usize) -> Option<Vec<Span>> {
    fn is_text(node: &LinkedNode) -> bool {
        matches!(node.kind(), SyntaxKind::Text | SyntaxKind::MathText)
    }
    fn collect_leaf_spans(node: &LinkedNode, out: &mut Vec<Span>) {
        if is_text(node) {
            out.push(node.span());
            return;
        }
        for child in node.children() {
            collect_leaf_spans(&child, out);
        }
    }
    /// Returns the direct child of `container` that `leaf` descends from.
    fn direct_child_containing<'a>(
        container: &LinkedNode<'a>,
        leaf: &LinkedNode<'a>,
    ) -> Option<LinkedNode<'a>> {
        let mut node = leaf.clone();
        loop {
            let parent = node.parent()?;
            if parent.span() == container.span() {
                return Some(node);
            }
            node = parent.clone();
        }
    }

    let root = LinkedNode::new(source.root());
    let leaf = root
        .leaf_at(cursor, Side::Before)
        .filter(is_text)
        .or_else(|| root.leaf_at(cursor, Side::After).filter(is_text))?;

    let Some(container) = paragraph_container(&leaf) else {
        return Some(vec![leaf.span()]);
    };
    let Some(anchor) = direct_child_containing(&container, &leaf) else {
        return Some(vec![leaf.span()]);
    };

    let children: Vec<LinkedNode> = container.children().collect();
    let Some(anchor_index) = children.iter().position(|c| c.span() == anchor.span()) else {
        return Some(vec![leaf.span()]);
    };

    let mut start = anchor_index;
    while start > 0 && continues_same_paragraph(children[start - 1].kind()) {
        start -= 1;
    }
    let mut end = anchor_index;
    while end + 1 < children.len() && continues_same_paragraph(children[end + 1].kind()) {
        end += 1;
    }

    let mut spans = Vec::new();
    for child in &children[start..=end] {
        collect_leaf_spans(child, &mut spans);
    }
    Some(spans)
}

/// Collects every point in `frame` where glyphs of a text run carry `span`.
///
/// Unlike `typst_ide::jump_from_cursor`, keeps scanning after the first match, so a span that
/// wraps across several lines highlights all of them.
fn find_all_in_frame(frame: &Frame, span: Span, out: &mut Vec<(TypstPoint, Abs)>) {
    for &(pos, ref item) in frame.items() {
        match item {
            FrameItem::Group(group) => {
                let mut nested = Vec::new();
                find_all_in_frame(&group.frame, span, &mut nested);
                out.extend(
                    nested
                        .into_iter()
                        .map(|(p, size)| (pos + p.transform(group.transform), size)),
                );
            }
            FrameItem::Text(text) => {
                if text.glyphs.iter().any(|glyph| glyph.span.0 == span) {
                    out.push((pos, text.size));
                }
            }
            _ => {}
        }
    }
}

/// Returns an approximate one-line highlight height for text at `font_size`.
fn line_height_for_font_size(font_size: Abs) -> f32 {
    font_size.to_pt() as f32 * 1.4
}

struct EditorState {
    editor: Entity<Editor>,
    _subscription: Subscription,
}

impl TypstPreviewView {
    /// Returns the pane in `direction`, or the active pane.
    fn get_pane(
        workspace: &mut Workspace,
        direction: Option<SplitDirection>,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) -> Entity<Pane> {
        match direction {
            Some(direction) => workspace
                .find_pane_in_direction(direction, cx)
                .unwrap_or_else(|| {
                    workspace.split_pane(workspace.active_pane().clone(), direction, window, cx)
                }),
            None => workspace.active_pane().clone(),
        }
    }

    /// Returns the pane to open a source file of the preview in: the previewed editor's pane,
    /// else any pane other than the preview's, else a new pane left of the preview.
    fn source_pane(
        workspace: &mut Workspace,
        preview_pane: Option<&Entity<Pane>>,
        source_editor: Option<&Entity<Editor>>,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) -> Entity<Pane> {
        if let Some(pane) = source_editor.and_then(|editor| workspace.pane_for(editor))
            && Some(&pane) != preview_pane
        {
            return pane;
        }
        let Some(preview_pane) = preview_pane else {
            return workspace.active_pane().clone();
        };
        if let Some(pane) = workspace.panes().iter().find(|pane| *pane != preview_pane) {
            return pane.clone();
        }
        workspace.split_pane(preview_pane.clone(), SplitDirection::Left, window, cx)
    }

    pub fn register(workspace: &mut Workspace, _window: &mut Window, _cx: &mut Context<Workspace>) {
        workspace.register_action(move |workspace, _: &OpenPreview, window, cx| {
            if let Some(editor) = Self::resolve_active_item_as_typst_editor(workspace, cx) {
                let view = Self::create_typst_view(workspace, editor.clone(), true, window, cx);
                let pane = Self::get_pane(workspace, None, window, cx);
                pane.update(cx, |pane, cx| {
                    if let Some(existing_view_idx) =
                        Self::find_existing_independent_preview_item_idx(pane, &editor, true, cx)
                    {
                        pane.activate_item(existing_view_idx, true, true, window, cx);
                    } else {
                        pane.add_item(Box::new(view.clone()), true, true, None, window, cx)
                    }
                });
                cx.notify();
            }
        });

        workspace.register_action(move |workspace, _: &OpenFixedPreview, window, cx| {
            if let Some(editor) = Self::resolve_active_item_as_typst_editor(workspace, cx) {
                let view = Self::create_typst_view(workspace, editor.clone(), false, window, cx);
                let pane = Self::get_pane(workspace, None, window, cx);
                pane.update(cx, |pane, cx| {
                    if let Some(existing_view_idx) =
                        Self::find_existing_independent_preview_item_idx(pane, &editor, false, cx)
                    {
                        pane.activate_item(existing_view_idx, true, true, window, cx);
                    } else {
                        pane.add_item(Box::new(view.clone()), true, true, None, window, cx)
                    }
                });
                cx.notify();
            }
        });

        workspace.register_action(move |workspace, _: &OpenPreviewToTheSide, window, cx| {
            if let Some(editor) = Self::resolve_active_item_as_typst_editor(workspace, cx) {
                let view = Self::create_typst_view(workspace, editor.clone(), true, window, cx);
                let pane = Self::get_pane(workspace, Some(SplitDirection::Right), window, cx);
                pane.update(cx, |pane, cx| {
                    if let Some(existing_view_idx) =
                        Self::find_existing_independent_preview_item_idx(pane, &editor, true, cx)
                    {
                        pane.activate_item(existing_view_idx, true, true, window, cx);
                    } else {
                        pane.add_item(Box::new(view.clone()), false, false, None, window, cx)
                    }
                });
                editor.focus_handle(cx).focus(window, cx);
                cx.notify();
            }
        });

        workspace.register_action(
            move |workspace, _: &OpenFixedPreviewToTheSide, window, cx| {
                if let Some(editor) = Self::resolve_active_item_as_typst_editor(workspace, cx) {
                    let view =
                        Self::create_typst_view(workspace, editor.clone(), false, window, cx);
                    let pane = Self::get_pane(workspace, Some(SplitDirection::Right), window, cx);
                    pane.update(cx, |pane, cx| {
                        if let Some(existing_view_idx) =
                            Self::find_existing_independent_preview_item_idx(
                                pane, &editor, false, cx,
                            )
                        {
                            pane.activate_item(existing_view_idx, true, true, window, cx);
                        } else {
                            pane.add_item(Box::new(view.clone()), false, false, None, window, cx)
                        }
                    });
                    editor.focus_handle(cx).focus(window, cx);
                    cx.notify();
                }
            },
        );

        workspace.register_action(move |workspace, _: &OpenFollowingPreview, window, cx| {
            if let Some(editor) = Self::resolve_active_item_as_typst_editor(workspace, cx) {
                let existing_follow_view_idx = {
                    let active_pane = workspace.active_pane().read(cx);
                    active_pane
                        .items_of_type::<TypstPreviewView>()
                        .find(|view| {
                            view.read(cx).mode == TypstPreviewMode::Follow
                                && view.read(cx).project_resolution
                        })
                        .and_then(|view| active_pane.index_for_item(&view))
                };

                if let Some(existing_follow_view_idx) = existing_follow_view_idx {
                    workspace.active_pane().update(cx, |pane, cx| {
                        pane.activate_item(existing_follow_view_idx, true, true, window, cx);
                    });
                } else {
                    let view =
                        Self::create_following_typst_view(workspace, editor, true, window, cx);
                    workspace.active_pane().update(cx, |pane, cx| {
                        pane.add_item(Box::new(view.clone()), true, true, None, window, cx)
                    });
                }
                cx.notify();
            }
        });

        workspace.register_action(
            move |workspace, _: &OpenFixedFollowingPreview, window, cx| {
                if let Some(editor) = Self::resolve_active_item_as_typst_editor(workspace, cx) {
                    let existing_follow_view_idx = {
                        let active_pane = workspace.active_pane().read(cx);
                        active_pane
                            .items_of_type::<TypstPreviewView>()
                            .find(|view| {
                                view.read(cx).mode == TypstPreviewMode::Follow
                                    && !view.read(cx).project_resolution
                            })
                            .and_then(|view| active_pane.index_for_item(&view))
                    };

                    if let Some(existing_follow_view_idx) = existing_follow_view_idx {
                        workspace.active_pane().update(cx, |pane, cx| {
                            pane.activate_item(existing_follow_view_idx, true, true, window, cx);
                        });
                    } else {
                        let view =
                            Self::create_following_typst_view(workspace, editor, false, window, cx);
                        workspace.active_pane().update(cx, |pane, cx| {
                            pane.add_item(Box::new(view.clone()), true, true, None, window, cx)
                        });
                    }
                    cx.notify();
                }
            },
        );
    }

    fn find_existing_independent_preview_item_idx(
        pane: &Pane,
        editor: &Entity<Editor>,
        project_resolution: bool,
        cx: &App,
    ) -> Option<usize> {
        pane.items_of_type::<TypstPreviewView>()
            .find(|view| {
                let view_read = view.read(cx);
                view_read.mode == TypstPreviewMode::Default
                    && view_read.project_resolution == project_resolution
                    && view_read
                        .active_editor
                        .as_ref()
                        .is_some_and(|active_editor| active_editor.editor == *editor)
            })
            .and_then(|view| pane.index_for_item(&view))
    }

    pub fn resolve_active_item_as_typst_editor(
        workspace: &Workspace,
        cx: &mut Context<Workspace>,
    ) -> Option<Entity<Editor>> {
        if let Some(editor) = workspace
            .active_item(cx)
            .and_then(|item| item.act_as::<Editor>(cx))
            && Self::is_typst_file(&editor, cx)
        {
            return Some(editor);
        }
        None
    }

    fn create_typst_view(
        workspace: &mut Workspace,
        editor: Entity<Editor>,
        project_resolution: bool,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) -> Entity<TypstPreviewView> {
        let workspace_handle = workspace.weak_handle();
        TypstPreviewView::new(
            TypstPreviewMode::Default,
            project_resolution,
            editor,
            workspace,
            workspace_handle,
            window,
            cx,
        )
    }

    fn create_following_typst_view(
        workspace: &mut Workspace,
        editor: Entity<Editor>,
        project_resolution: bool,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) -> Entity<TypstPreviewView> {
        let workspace_handle = workspace.weak_handle();
        TypstPreviewView::new(
            TypstPreviewMode::Follow,
            project_resolution,
            editor,
            workspace,
            workspace_handle,
            window,
            cx,
        )
    }

    pub fn new(
        mode: TypstPreviewMode,
        project_resolution: bool,
        active_editor: Entity<Editor>,
        workspace: &Workspace,
        workspace_handle: WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) -> Entity<Self> {
        cx.new(|cx| {
            let empty_pdf = Arc::new(Pdf::empty());
            let pdf_view = cx.new(|cx| {
                let mut pdf_view = PdfView::from_pdf(empty_pdf, window, cx);
                pdf_view
                    .extra_context_menu_actions
                    .push(("Export PDF".into(), Box::new(ExportPdf)));
                pdf_view
            });

            let mut this = Self {
                active_editor: None,
                observed_editors: Default::default(),
                focus_handle: cx.focus_handle(),
                workspace: workspace_handle.clone(),
                pdf_view: pdf_view,
                world: None,
                document: None,
                compile_state: TypstCompileState::Uninitialized,
                compile_warnings: Vec::new(),
                pending_update_task: None,
                mode,
                project_resolution,
                requested_packages: HashSet::default(),
                package_download_task: None,
                published_diagnostic_paths: HashSet::default(),
                rendered_page_hashes: None,
            };

            this.set_editor(active_editor, window, cx);

            // diagnostics come from this preview, so they go away with it. Deferred, since the
            // view is often released while its workspace is being updated.
            cx.on_release(|this, cx| {
                let workspace = this.workspace.clone();
                let mut published_paths = std::mem::take(&mut this.published_diagnostic_paths);
                cx.defer(move |cx| {
                    publish_project_diagnostics(&workspace, None, &[], &mut published_paths, cx);
                });
            })
            .detach();

            // observe the workspace to react to typing in included files.
            if let Some(workspace_ent) = &workspace_handle.upgrade() {
                let mut editors_to_observe = Vec::new();
                for pane in workspace.panes() {
                    for item in pane.read(cx).items() {
                        if let Some(editor) = item.act_as::<Editor>(cx)
                            && Self::is_typst_file(&editor, cx)
                        {
                            editors_to_observe.push(editor);
                        }
                    }
                }
                for editor in editors_to_observe {
                    this.observe_additional_editor(editor, window, cx);
                }

                cx.observe_in(workspace_ent, window, |this, workspace, window, cx| {
                    let item = workspace.read(cx).active_item(cx);
                    this.workspace_updated(item, window, cx);
                })
                .detach();

                // recompile when files the document reads change on disk, like images edited in
                // another application.
                cx.subscribe_in(
                    workspace.project(),
                    window,
                    |this, project, event, window, cx| {
                        if let project::Event::WorktreeUpdatedEntries(worktree_id, changes) = event
                            && this.disk_changes_affect_document(project, *worktree_id, changes, cx)
                        {
                            this.update_typst_from_active_editor(true, false, window, cx);
                        }
                    },
                )
                .detach();
            } else {
                log::error!("Failed to listen to workspace updates");
            }

            this
        })
    }

    fn disk_changes_affect_document(
        &self,
        project: &Entity<project::Project>,
        worktree_id: project::WorktreeId,
        changes: &project::UpdatedEntriesSet,
        cx: &App,
    ) -> bool {
        let Some(world) = &self.world else {
            return false;
        };
        let Some(worktree) = project.read(cx).worktree_for_id(worktree_id, cx) else {
            return false;
        };
        let worktree = worktree.read(cx);
        // a running compile holds the world and may have read a file before it changed.
        let Some(world) = world.try_lock() else {
            return true;
        };
        let compile_failed = matches!(self.compile_state, TypstCompileState::Failed(_));
        changes_affect_world(
            &world,
            compile_failed,
            changes
                .iter()
                .map(|(path, _, change)| (worktree.absolutize(path), *change)),
        )
    }

    fn workspace_updated(
        &mut self,
        active_item: Option<Box<dyn ItemHandle>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(item) = active_item
            && item.item_id() != cx.entity_id()
            && let Some(editor) = item.act_as::<Editor>(cx)
            && Self::is_typst_file(&editor, cx)
        {
            if self.mode == TypstPreviewMode::Follow {
                self.set_editor(editor, window, cx);
            } else {
                self.observe_additional_editor(editor, window, cx);
            }
        }
    }

    fn observe_additional_editor(
        &mut self,
        editor: Entity<Editor>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let id = editor.entity_id();
        if self.observed_editors.contains_key(&id) {
            return;
        }
        if let Some(active) = &self.active_editor {
            if active.editor.entity_id() == id {
                return;
            }
        }
        let sub = cx.subscribe_in(
            &editor,
            window,
            |this, _editor, event: &EditorEvent, window, cx| match event {
                EditorEvent::Edited { .. }
                | EditorEvent::BufferEdited { .. }
                | EditorEvent::DirtyChanged
                | EditorEvent::BuffersEdited { .. } => {
                    this.update_typst_from_active_editor(true, false, window, cx);
                }
                EditorEvent::SelectionsChanged { .. } => {
                    let (selection_start, file_path, editor_is_focused) =
                        _editor.update(cx, |editor, cx| {
                            let index = Self::selected_source_index(editor, cx);
                            let file_path = editor
                                .file_at(MultiBufferOffset(0), cx)
                                .and_then(|f| f.as_local().map(|f| f.abs_path(cx)));
                            let focused = editor.focus_handle(cx).is_focused(window);
                            (index, file_path, focused)
                        });
                    this.sync_preview_to_source_index(
                        selection_start,
                        file_path,
                        editor_is_focused,
                        cx,
                    );
                    cx.notify();
                }
                _ => {}
            },
        );
        self.observed_editors.insert(id, sub);
    }

    pub fn is_typst_file<V>(editor: &Entity<Editor>, cx: &Context<V>) -> bool {
        let buffer = editor.read(cx).buffer().read(cx);
        if let Some(buffer) = buffer.as_singleton()
            && let Some(language) = buffer.read(cx).language()
        {
            return language.name() == "Typst";
        }
        false
    }

    fn set_editor(&mut self, editor: Entity<Editor>, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(active) = &self.active_editor
            && active.editor == editor
        {
            return;
        }

        let subscription = cx.subscribe_in(
            &editor,
            window,
            |this, editor, event: &EditorEvent, window, cx| {
                match event {
                    EditorEvent::Edited { .. }
                    | EditorEvent::BufferEdited { .. }
                    | EditorEvent::DirtyChanged
                    | EditorEvent::BuffersEdited { .. } => {
                        this.update_typst_from_active_editor(true, false, window, cx);
                    }
                    EditorEvent::SelectionsChanged { .. } => {
                        let (selection_start, file_path, editor_is_focused) =
                            editor.update(cx, |editor, cx| {
                                let index = Self::selected_source_index(editor, cx);
                                let file_path = editor
                                    .file_at(MultiBufferOffset(0), cx)
                                    .and_then(|f| f.as_local().map(|f| f.abs_path(cx)));
                                let focused = editor.focus_handle(cx).is_focused(window);
                                (index, file_path, focused)
                            });
                        this.sync_preview_to_source_index(
                            selection_start,
                            file_path,
                            editor_is_focused,
                            cx,
                        );
                        cx.notify();
                    }
                    _ => {}
                };
            },
        );

        self.active_editor = Some(EditorState {
            editor,
            _subscription: subscription,
        });

        self.update_typst_from_active_editor(false, true, window, cx);
    }

    fn update_typst_from_active_editor(
        &mut self,
        wait_for_debounce: bool,
        should_reveal: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(state) = &self.active_editor {
            // dropping the old task cancels it and restarts the debounce while the user types.
            if wait_for_debounce {
                self.pending_update_task = None;
            }
            self.pending_update_task = Some(self.schedule_typst_update(
                wait_for_debounce,
                should_reveal,
                state.editor.clone(),
                window,
                cx,
            ));
        }
    }

    /// Resolves the project root and main file from `active_path`.
    ///
    /// Walks up looking for a `typst.toml` entrypoint or a `main.typ`. Performs filesystem I/O,
    /// so callers must run it on a background thread.
    fn resolve_project(project_resolution: bool, active_path: &Path) -> (PathBuf, PathBuf) {
        let unresolved = (
            active_path
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| active_path.to_path_buf()),
            active_path.to_path_buf(),
        );
        if !project_resolution {
            return unresolved;
        }
        let mut current = active_path.parent();
        let mut found = None;
        while let Some(dir) = current {
            let toml_path = dir.join("typst.toml");
            if toml_path.exists()
                && let Ok(toml_content) = std::fs::read_to_string(&toml_path)
                && let Ok(manifest) = toml_content.parse::<toml::Table>()
                && let Some(entrypoint) = manifest
                    .get("package")
                    .and_then(|p| p.get("entrypoint"))
                    .and_then(|e| e.as_str())
            {
                let main_file = dir.join(entrypoint);
                if main_file.exists() {
                    found = Some((dir.to_path_buf(), main_file));
                    break;
                }
            }
            let potential_main = dir.join("main.typ");
            if potential_main.exists() {
                found = Some((dir.to_path_buf(), potential_main));
                break;
            }
            current = dir.parent();
        }
        found.unwrap_or(unresolved)
    }

    fn process_compiled_typst(
        view: &mut Self,
        bytes: Vec<u8>,
        document: Arc<PagedDocument>,
        new_world_arc: Arc<Mutex<TypstSystemWorld>>,
        pages: Vec<PageMetadata>,
        page_hashes: Vec<u128>,
        max_width: f32,
        selection_start: usize,
        should_reveal: bool,
        active_path: PathBuf,
        workspace_weak: WeakEntity<Workspace>,
        cx: &mut Context<Self>,
    ) {
        view.world = Some(new_world_arc.clone());
        view.document = Some(document.clone());

        let http_client = cx.http_client();
        cx.spawn(async move |view, cx| {
            pdf_viewer::pdf_renderer::ensure_pdfium_installed(http_client)
                .await
                .log_err();
            let pdf = Arc::new(Pdf::from_bytes(bytes, Some((pages, max_width)), None).await);
            view.update(cx, |view, cx| {
                let previous_pdf = view.pdf_view.read(cx).pdf(cx);
                if let Some((previous_id, previous_hashes)) = &view.rendered_page_hashes
                    && *previous_id == previous_pdf.id
                {
                    pdf.reuse_rendered_pages(&previous_pdf, |page_idx| {
                        previous_hashes.get(page_idx) == page_hashes.get(page_idx)
                    });
                }
                view.rendered_page_hashes = Some((pdf.id, page_hashes));
                let preview = cx.weak_entity();
                view.pdf_view.update(cx, |pv, cx| {
                    pv.update_pdf(pdf, cx);
                    let document_clone = document.clone();
                    let world_arc = new_world_arc.clone();
                    let workspace_weak = workspace_weak.clone();

                    pv.on_page_click = Some(Arc::new(move |page_index, point, window, cx| {
                        let typst_point = typst::layout::Point::new(
                            typst::layout::Abs::pt(point.x as f64),
                            typst::layout::Abs::pt(point.y as f64),
                        );

                        if let Some(page_number) = std::num::NonZeroUsize::new(page_index + 1) {
                            let world = world_arc.lock();
                            let position = typst::introspection::PagedPosition {
                                page: page_number,
                                point: typst_point,
                            };
                            if let Some(typst_ide::Jump::File(file_id, offset)) =
                                typst_ide::jump_from_click(&*world, &*document_clone, &position)
                                && let Some(path_buf) = world.resolve_path(file_id).ok()
                            {
                                let normalized_path = util::normalize_path(&path_buf);
                                if let Some(workspace) = workspace_weak.upgrade() {
                                    // move the cursor without focusing, so clicking in the preview
                                    // does not steal focus, like the Markdown preview.
                                    let target_editor = workspace.update(cx, |ws, cx| {
                                        for pane in ws.panes() {
                                            for item in pane.read(cx).items() {
                                                if let Some(editor) = item.act_as::<Editor>(cx)
                                                    && let Some(file) = editor
                                                        .read(cx)
                                                        .file_at(MultiBufferOffset(0), cx)
                                                    && let Some(local) = file.as_local()
                                                {
                                                    let item_path = util::normalize_path(
                                                        &local.abs_path(cx).to_path_buf(),
                                                    );
                                                    if item_path == normalized_path {
                                                        return Some(editor.clone());
                                                    }
                                                }
                                            }
                                        }
                                        None
                                    });

                                    if let Some(editor) = target_editor {
                                        Self::move_cursor_to_source_index(
                                            &editor, offset, window, cx,
                                        );
                                        return;
                                    }

                                    // open without focusing it, next to the previewed editor rather than
                                    // as a tab beside the preview.
                                    let open_task = workspace.update(cx, |ws, cx| {
                                        let preview_pane = preview
                                            .upgrade()
                                            .and_then(|preview| ws.pane_for(&preview));
                                        let source_editor = preview.upgrade().and_then(|preview| {
                                            preview
                                                .read(cx)
                                                .active_editor
                                                .as_ref()
                                                .map(|state| state.editor.clone())
                                        });
                                        let pane = Self::source_pane(
                                            ws,
                                            preview_pane.as_ref(),
                                            source_editor.as_ref(),
                                            window,
                                            cx,
                                        );
                                        match ws
                                            .project()
                                            .read(cx)
                                            .find_project_path(&normalized_path, cx)
                                        {
                                            Some(project_path) => ws.open_path(
                                                project_path,
                                                Some(pane.downgrade()),
                                                false,
                                                window,
                                                cx,
                                            ),
                                            None => ws.open_abs_path(
                                                normalized_path,
                                                workspace::OpenOptions {
                                                    focus: Some(false),
                                                    ..Default::default()
                                                },
                                                window,
                                                cx,
                                            ),
                                        }
                                    });
                                    window
                                        .spawn(cx, async move |cx| {
                                            if let Ok(item) = open_task.await {
                                                cx.update(|window, cx| {
                                                    if let Some(new_editor) =
                                                        item.act_as::<Editor>(cx)
                                                    {
                                                        Self::move_cursor_to_source_index(
                                                            &new_editor,
                                                            offset,
                                                            window,
                                                            cx,
                                                        );
                                                    }
                                                })
                                                .ok();
                                            }
                                        })
                                        .detach();
                                }
                            }
                        }
                    }));
                });

                view.sync_preview_to_source_index(
                    selection_start,
                    Some(active_path),
                    should_reveal,
                    cx,
                );
                cx.emit(SearchEvent::MatchesInvalidated);
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn schedule_typst_update(
        &mut self,
        wait_for_debounce: bool,
        should_reveal_selection: bool,
        editor: Entity<Editor>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        cx.spawn_in(window, async move |view, cx| {
            if wait_for_debounce {
                cx.background_executor().timer(REPARSE_DEBOUNCE).await;
            }

            let editor_clone = editor.clone();
            let update = view.update(cx, |view, cx| {
                let is_active_editor = view
                    .active_editor
                    .as_ref()
                    .is_some_and(|active_editor| active_editor.editor == editor_clone);
                if !is_active_editor {
                    return None;
                }

                let (active_snapshot, selection_start, file_path) =
                    editor_clone.update(cx, |editor, cx| {
                        let active_snapshot = editor.buffer().read(cx).snapshot(cx);
                        let selection_start = Self::selected_source_index(editor, cx);
                        let file_path = editor
                            .file_at(MultiBufferOffset(0), cx)
                            .and_then(|f| f.as_local().map(|f| f.abs_path(cx)));
                        (active_snapshot, selection_start, file_path)
                    });

                let Some(active_path) = file_path else {
                    view.compile_state =
                        TypstCompileState::Failed("Save this file to preview it".to_string());
                    cx.notify();
                    return None;
                };

                let overrides = if let Some(workspace) = view.workspace.upgrade() {
                    let mut map = HashMap::new();
                    for pane in workspace.read(cx).panes() {
                        for item in pane.read(cx).items() {
                            if item.item_id() == cx.entity_id() {
                                continue;
                            }
                            if let Some(editor) = item.act_as::<Editor>(cx) {
                                let editor_read = editor.read(cx);
                                let buffer = editor_read.buffer().read(cx);
                                // saved buffers match the disk, which the world reads and caches
                                // itself.
                                if buffer.is_dirty(cx)
                                    && let Some(file) =
                                        editor_read.file_at(MultiBufferOffset(0), cx)
                                    && let Some(local) = file.as_local()
                                {
                                    let snapshot = buffer.snapshot(cx);
                                    map.insert(local.abs_path(cx).to_path_buf(), snapshot);
                                }
                            }
                        }
                    }
                    map
                } else {
                    HashMap::new()
                };

                // the active editor state takes precedence.
                let mut final_overrides = overrides;
                final_overrides.insert(active_path.clone(), active_snapshot);

                view.compile_state = TypstCompileState::Compiling;
                view.compile_warnings.clear();
                cx.notify();

                Some((
                    selection_start,
                    active_path,
                    view.project_resolution,
                    final_overrides,
                    view.world.clone(),
                ))
            })?;

            if let Some((
                selection_start,
                active_path,
                project_resolution,
                final_overrides,
                current_world,
            )) = update
            {
                let active_path_sync = active_path.clone();
                let workspace_weak = view.update(cx, |v, _| v.workspace.clone())?;
                let active_editor = view.update(cx, |v, _| {
                    v.active_editor.as_ref().map(|s| s.editor.clone())
                })?;

                struct CompileOutcome {
                    diagnostics: Vec<TypstDiagnostic>,
                    missing_packages: Vec<PackageSpec>,
                    result: Result<(
                        Vec<u8>,
                        Arc<PagedDocument>,
                        Arc<Mutex<TypstSystemWorld>>,
                        Vec<PageMetadata>,
                        Vec<u128>,
                        f32,
                    )>,
                }

                let compile_outcome: CompileOutcome = cx
                    .background_executor()
                    .spawn(async move {
                        let start = Instant::now();

                        // resolving the project performs filesystem I/O.
                        let (new_root, main_file_path) =
                            Self::resolve_project(project_resolution, &active_path);

                        let world_arc = match current_world.filter(|w| w.lock().root == new_root) {
                            Some(world) => world,
                            None => Arc::new(Mutex::new(TypstSystemWorld::new(new_root))),
                        };

                        let main_contents = match final_overrides.get(&main_file_path) {
                            Some(snap) => snap.text(),
                            None => match std::fs::read_to_string(&main_file_path) {
                                Ok(text) => text,
                                Err(e) => {
                                    return CompileOutcome {
                                        diagnostics: Vec::new(),
                                        missing_packages: Vec::new(),
                                        result: Err(anyhow::anyhow!(
                                            "Failed to read {}: {}",
                                            main_file_path.display(),
                                            e
                                        )),
                                    };
                                }
                            },
                        };

                        let mut world = world_arc.lock();
                        world.update_main(main_contents.into(), &main_file_path);
                        world.file_overrides = final_overrides;

                        log::debug!("Prepared compilation in {:?}", start.elapsed());
                        let start = Instant::now();

                        let compiled = typst::compile(&*world);
                        let mut diagnostics = resolve_typst_diagnostics(&world, &compiled.warnings);
                        for diagnostic in &diagnostics {
                            log::debug!("Typst compile warning: {}", diagnostic.message);
                        }

                        log::debug!("Compiled in {:?}", start.elapsed());
                        let start = Instant::now();

                        let document = match compiled.output {
                            Ok(document) => document,
                            Err(errors) => {
                                diagnostics.extend(resolve_typst_diagnostics(&world, &errors));
                                return CompileOutcome {
                                    diagnostics,
                                    missing_packages: world.take_missing_packages(),
                                    result: Err(anyhow::anyhow!(
                                        "Compilation failed:\n{:?}",
                                        errors
                                    )),
                                };
                            }
                        };

                        log::debug!("Generated output in {:?}", start.elapsed());
                        let start = Instant::now();

                        let bytes =
                            match typst_pdf::pdf(&document, &typst_pdf::PdfOptions::default()) {
                                Ok(bytes) => bytes,
                                Err(e) => {
                                    return CompileOutcome {
                                        diagnostics,
                                        missing_packages: world.take_missing_packages(),
                                        result: Err(anyhow::anyhow!("PDF export failed:\n{:?}", e)),
                                    };
                                }
                            };

                        log::debug!("Generated PDF in {:?}", start.elapsed());
                        let start = Instant::now();

                        let mut max_width = 0.0f32;
                        let mut pages = Vec::with_capacity(document.pages().len());
                        let page_hashes = page_hashes(&document);

                        for (i, page) in document.pages().iter().enumerate() {
                            let w = page.frame.width().to_pt() as f32;
                            let h = page.frame.height().to_pt() as f32;
                            max_width = max_width.max(w);

                            pages.push(PageMetadata {
                                page_number: i,
                                size: Size {
                                    width: w,
                                    height: h,
                                },
                                bounds: Bounds {
                                    origin: Point { x: 0.0, y: h }, // Matches PDFium's bottom-left origin fallback
                                    size: Size {
                                        width: w,
                                        height: h,
                                    },
                                },
                            });
                        }

                        log::debug!(
                            "Page sizes extracted for {} pages in {:?}",
                            document.pages().len(),
                            start.elapsed()
                        );

                        let world_arc_ret = world_arc.clone();
                        CompileOutcome {
                            diagnostics,
                            missing_packages: world.take_missing_packages(),
                            result: Ok((
                                bytes,
                                Arc::new(document),
                                world_arc_ret,
                                pages,
                                page_hashes,
                                max_width,
                            )),
                        }
                    })
                    .await;

                view.update_in(cx, move |view, window, cx| {
                    view.compile_state = TypstCompileState::Rendering;
                    publish_project_diagnostics(
                        &view.workspace,
                        active_editor.as_ref(),
                        &compile_outcome.diagnostics,
                        &mut view.published_diagnostic_paths,
                        cx,
                    );
                    match compile_outcome.result {
                        Ok((bytes, document, new_world_arc, pages, page_hashes, max_width)) => {
                            let warnings = compile_outcome
                                .diagnostics
                                .iter()
                                .filter(|d| d.severity == DiagnosticSeverity::WARNING)
                                .map(|d| d.message.clone())
                                .collect();
                            Self::process_compiled_typst(
                                view,
                                bytes,
                                document,
                                new_world_arc,
                                pages,
                                page_hashes,
                                max_width,
                                selection_start,
                                should_reveal_selection,
                                active_path_sync,
                                workspace_weak,
                                cx,
                            );
                            view.compile_warnings = warnings;
                            view.compile_state = TypstCompileState::Finished;
                        }
                        Err(e) => {
                            view.compile_state = TypstCompileState::Failed(e.to_string());
                            cx.notify();
                        }
                    }
                    view.download_missing_packages(compile_outcome.missing_packages, window, cx);
                    view.pending_update_task = None;
                })
                .ok();
            }

            Ok(())
        })
    }

    /// Downloads `@preview` packages that imports need into the Typst package cache, then
    /// compiles again.
    fn download_missing_packages(
        &mut self,
        missing_packages: Vec<PackageSpec>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let packages = missing_packages
            .into_iter()
            .filter(|package| self.requested_packages.insert(package.clone()))
            .collect::<Vec<_>>();
        if packages.is_empty() {
            return;
        }

        self.compile_state = TypstCompileState::DownloadingPackages(packages.clone());
        cx.notify();
        let http_client = cx.http_client();
        self.package_download_task = Some(cx.spawn_in(window, async move |view, cx| {
            let mut failures = Vec::new();
            for package in &packages {
                let result = match crate::typst_world::package_download_dir(package) {
                    Some(destination) => {
                        download_package(http_client.as_ref(), package, &destination).await
                    }
                    None => Err(anyhow::anyhow!("no Typst package cache directory")),
                };
                if let Err(error) = result {
                    log::error!("failed to download Typst package {package}: {error:#}");
                    failures.push(format!("{package}: {error:#}"));
                }
            }
            view.update_in(cx, |view, window, cx| {
                view.package_download_task = None;
                if failures.is_empty() {
                    view.update_typst_from_active_editor(false, false, window, cx);
                } else {
                    view.compile_state = TypstCompileState::Failed(format!(
                        "Failed to download Typst packages:\n{}",
                        failures.join("\n")
                    ));
                    cx.notify();
                }
            })
            .ok();
        }));
    }

    fn export_path(&self) -> Option<PathBuf> {
        export_path(&self.world.as_ref()?.lock())
    }

    fn export_pdf(&mut self, _: &ExportPdf, _window: &mut Window, cx: &mut Context<Self>) {
        let document = self.document.clone();
        let export_path = self.export_path();
        let workspace = self.workspace.clone();
        cx.spawn(async move |_, cx| {
            let result = async {
                let document = document.context("the Typst document has not compiled yet")?;
                let path = export_path.context("the Typst document has no main file")?;
                let write_path = path.clone();
                cx.background_spawn(async move {
                    let bytes = export_bytes(&document, chrono::Local::now())?;
                    std::fs::write(&write_path, bytes)
                        .with_context(|| format!("writing {}", write_path.display()))
                })
                .await?;
                anyhow::Ok(path)
            }
            .await;

            let message = match result {
                Ok(path) => format!("Exported {}", path.display()),
                Err(error) => format!("Failed to export PDF: {error:#}"),
            };
            workspace
                .update(cx, |workspace, cx| {
                    struct TypstExportToast;
                    workspace.show_toast(
                        workspace::Toast::new(
                            workspace::notifications::NotificationId::unique::<TypstExportToast>(),
                            message,
                        )
                        .autohide(),
                        cx,
                    );
                })
                .log_err();
        })
        .detach();
    }

    fn selected_source_index(editor: &Editor, cx: &mut App) -> usize {
        editor
            .selections
            .last::<MultiBufferOffset>(&editor.display_snapshot(cx))
            .range()
            .start
            .0
    }

    fn sync_preview_to_source_index(
        &mut self,
        source_index: usize,
        file_path: Option<PathBuf>,
        reveal: bool,
        cx: &mut Context<Self>,
    ) {
        if reveal {
            if let (Some(document), Some(world)) = (&self.document, &self.world) {
                let world_lock = world.lock();
                let source = if let Some(path) = file_path {
                    let mut found_id = None;
                    if world_lock.resolve_path(world_lock.main()).ok().as_ref() == Some(&path) {
                        found_id = Some(world_lock.main());
                    } else {
                        for id in world_lock.sources.read().keys() {
                            if world_lock.resolve_path(*id).ok().as_ref() == Some(&path) {
                                found_id = Some(*id);
                                break;
                            }
                        }
                    }

                    if let Some(id) = found_id {
                        world_lock.source(id).ok()
                    } else {
                        let rel_path = path.strip_prefix(&world_lock.root).unwrap_or(&path);
                        let mut path_str = rel_path.to_string_lossy().replace("\\", "/");
                        if !path_str.starts_with('/') {
                            path_str.insert(0, '/');
                        }
                        let file_id = typst::syntax::VirtualPath::new(path_str).ok().map(|vpath| {
                            typst::syntax::FileId::new(RootedPath::new(VirtualRoot::Project, vpath))
                        });
                        file_id.and_then(|file_id| world_lock.source(file_id).ok())
                    }
                } else {
                    Some(world_lock.main_source().clone())
                };

                if let Some(source) = source {
                    let safe_index = source_index.min(source.text().len());
                    let positions = typst_ide::jump_from_cursor(&**document, &source, safe_index);
                    if let Some(position) = positions.last() {
                        let page_index = position.page.get() - 1;

                        // `jump_from_cursor` only resolves the single leaf under the cursor, so
                        // the anchor and bottom are computed from every leaf in the paragraph.
                        let paragraph_hits: Vec<(TypstPoint, Abs)> =
                            paragraph_leaf_spans(&source, safe_index)
                                .and_then(|spans| {
                                    let page = document.pages().get(page_index)?;
                                    let mut hits = Vec::new();
                                    for span in spans {
                                        find_all_in_frame(&page.frame, span, &mut hits);
                                    }
                                    Some(hits)
                                })
                                .unwrap_or_default();

                        let point = paragraph_hits
                            .iter()
                            .min_by(|(a, _), (b, _)| a.y.to_pt().total_cmp(&b.y.to_pt()))
                            .map(|(p, _)| Point {
                                x: p.x.to_pt() as f32,
                                y: p.y.to_pt() as f32,
                            })
                            .unwrap_or(Point {
                                x: position.point.x.to_pt() as f32,
                                y: position.point.y.to_pt() as f32,
                            });
                        let block_bottom_y = paragraph_hits
                            .iter()
                            .map(|(p, _)| p.y.to_pt() as f32)
                            .reduce(f32::max);
                        // derive the line height from the actual font size of the paragraph.
                        let line_height = paragraph_hits
                            .first()
                            .map(|(_, size)| line_height_for_font_size(*size));

                        self.pdf_view.update(cx, |pv, cx| {
                            pv.scroll_to_target(
                                PdfTarget {
                                    page_index,
                                    point,
                                    block_bottom_y,
                                    line_height,
                                },
                                ScrollMode::Highlight,
                                ScrollAnchor::Baseline,
                                cx,
                            );
                        });
                    }
                    if positions.len() > 1 {
                        log::debug!("ignoring {} other preview positions", positions.len() - 1);
                    }
                }
            }
        }
    }

    /// Moves the cursor of `editor` to the start of the line containing `source_index`.
    ///
    /// Does not focus `editor` or change the active pane. Snapping to the line matches the
    /// Markdown preview and stays meaningful after the paragraph reflows in the PDF.
    fn move_cursor_to_source_index(
        editor: &Entity<Editor>,
        source_index: usize,
        window: &mut Window,
        cx: &mut App,
    ) {
        editor.update(cx, |editor, cx| {
            let snapshot = editor.buffer().read(cx).snapshot(cx);
            let safe_index = source_index.min(snapshot.len().0);
            let row = snapshot.offset_to_point(MultiBufferOffset(safe_index)).row;
            let line_start = snapshot.point_to_offset(BufferPoint::new(row, 0));
            let selection = line_start..line_start;
            editor.change_selections(
                SelectionEffects::scroll(Autoscroll::fit()),
                window,
                cx,
                |selections| selections.select_ranges(vec![selection]),
            );
        });
    }
}

impl Focusable for TypstPreviewView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl EventEmitter<()> for TypstPreviewView {}
impl EventEmitter<SearchEvent> for TypstPreviewView {}

impl Item for TypstPreviewView {
    type Event = ();

    fn act_as_type<'a>(
        &'a self,
        type_id: TypeId,
        self_handle: &'a Entity<Self>,
        _: &'a App,
    ) -> Option<AnyEntity> {
        if type_id == TypeId::of::<Self>() {
            Some(self_handle.clone().into())
        } else if type_id == TypeId::of::<Editor>() {
            self.active_editor
                .as_ref()
                .map(|state| state.editor.clone().into())
        } else if type_id == TypeId::of::<PdfView>() {
            Some(self.pdf_view.clone().into())
        } else {
            None
        }
    }

    fn tab_icon(&self, window: &Window, cx: &App) -> Option<ui::Icon> {
        self.pdf_view.tab_icon(window, cx)
    }

    fn breadcrumb_location(&self, cx: &App) -> ToolbarItemLocation {
        self.pdf_view.breadcrumb_location(cx)
    }

    fn breadcrumbs(&self, cx: &App) -> Option<(Vec<HighlightedText>, Option<Font>)> {
        self.pdf_view.breadcrumbs(cx)
    }

    fn tab_content_text(&self, _detail: usize, cx: &App) -> SharedString {
        self.active_editor
            .as_ref()
            .map(|editor_state| {
                let buffer = editor_state.editor.read(cx).buffer().read(cx);
                let title = buffer.title(cx);
                format!("Preview {}", title).into()
            })
            .unwrap_or_else(|| SharedString::from("Typst Preview"))
    }

    fn buffer_kind(&self, cx: &App) -> ItemBufferKind {
        self.pdf_view.buffer_kind(cx)
    }

    fn as_searchable(
        &self,
        _handle: &Entity<Self>,
        _cx: &App,
    ) -> Option<Box<dyn SearchableItemHandle>> {
        Some(Box::new(self.pdf_view.clone()))
    }
}

impl Render for TypstPreviewView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        use TypstCompileState::*;
        let page_count = self.pdf_view.read(cx).pdf(cx).metadata.page_count;
        let bg_color = cx.theme().colors().editor_background;

        div()
            .id("TypstPreview")
            .key_context("TypstPreview")
            .track_focus(&self.focus_handle(cx))
            .on_action(cx.listener(Self::export_pdf))
            .size_full()
            .bg(bg_color)
            .child(
                if !matches!(self.compile_state, Failed(_)) && page_count != 0 {
                    let warning_count = self.compile_warnings.len();
                    div()
                        .relative()
                        .size_full()
                        .child(self.pdf_view.clone())
                        .when(warning_count > 0, |this| {
                            let message =
                                self.compile_warnings.first().cloned().unwrap_or_default();
                            let label = if warning_count > 1 {
                                format!("{} ({} more warning(s))", message, warning_count - 1)
                            } else {
                                message
                            };
                            this.child(
                                div()
                                    .absolute()
                                    .top_0()
                                    .left_0()
                                    .right_0()
                                    .p_1()
                                    .bg(cx.theme().status().warning_background)
                                    .text_color(cx.theme().status().warning)
                                    .child(label),
                            )
                        })
                        .into_any_element()
                } else {
                    let (status_message, text_color) = match self.compile_state.clone() {
                        Failed(err) => (err, cx.theme().status().error),
                        Compiling | Rendering => (
                            "Compiling Typst...".to_string(),
                            cx.theme().colors().text_muted,
                        ),
                        DownloadingPackages(packages) => (
                            format!(
                                "Downloading {}...",
                                packages
                                    .iter()
                                    .map(ToString::to_string)
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ),
                            cx.theme().colors().text_muted,
                        ),
                        Uninitialized => (
                            "Waiting for input...".to_string(),
                            cx.theme().colors().text_muted,
                        ),
                        Finished => ("No view found".to_string(), cx.theme().status().error),
                    };

                    div()
                        .size_full()
                        .flex()
                        .items_center()
                        .justify_center()
                        .p_4()
                        .text_color(text_color)
                        .child(status_message)
                        .into_any_element()
                },
            )
    }
}

/// Returns where the exported PDF goes, next to the main file, as the Typst CLI does.
fn export_path(world: &TypstSystemWorld) -> Option<PathBuf> {
    let main_path = world.resolve_path(world.main()).ok()?;
    Some(main_path.with_extension("pdf"))
}

/// Returns whether any of the changed paths on disk were read by the last compile of `world`.
///
/// After a failed compile, new files count as well, since the failure may be a missing include
/// or image.
fn changes_affect_world(
    world: &TypstSystemWorld,
    compile_failed: bool,
    mut changes: impl Iterator<Item = (PathBuf, project::PathChange)>,
) -> bool {
    changes.any(|(path, change)| match change {
        project::PathChange::Loaded => false,
        project::PathChange::Added | project::PathChange::AddedOrUpdated
            if compile_failed && path.starts_with(&world.root) =>
        {
            true
        }
        _ => world.has_read(&path),
    })
}

/// Exports `document` as a PDF created at `now`, as the Typst CLI does, unless the document sets
/// its own date.
///
/// The preview leaves the date out, so its bytes stay the same between compiles.
fn export_bytes(document: &PagedDocument, now: chrono::DateTime<chrono::Local>) -> Result<Vec<u8>> {
    use chrono::{Datelike as _, Timelike as _};

    let timestamp = typst::foundations::Datetime::from_ymd_hms(
        now.year(),
        now.month().try_into()?,
        now.day().try_into()?,
        now.hour().try_into()?,
        now.minute().try_into()?,
        now.second().try_into()?,
    )
    .and_then(|datetime| {
        typst_pdf::Timestamp::new_local(datetime, now.offset().local_minus_utc() / 60)
    });
    typst_pdf::pdf(
        document,
        &typst_pdf::PdfOptions {
            timestamp,
            ..Default::default()
        },
    )
    .map_err(|errors| anyhow::anyhow!("PDF export failed:\n{errors:?}"))
}

/// Downloads and unpacks an `@preview` package from the Typst package registry into
/// `destination`.
async fn download_package(
    http_client: &dyn http_client::HttpClient,
    package: &PackageSpec,
    destination: &Path,
) -> Result<()> {
    let parent = destination
        .parent()
        .context("package directory has no parent")?;
    std::fs::create_dir_all(parent).with_context(|| format!("creating {parent:?}"))?;
    let url = crate::typst_world::package_download_url(package);
    http_client::github_download::download_server_binary(
        http_client,
        &url,
        None,
        destination,
        http_client::github::AssetKind::TarGz,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scratch directory tree that removes itself on drop.
    struct ScratchDir(PathBuf);

    impl ScratchDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "typst_preview_view_test_{name}_{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn join(&self, rel: &str) -> PathBuf {
            self.0.join(rel)
        }
    }

    impl Drop for ScratchDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    #[gpui::test]
    async fn diagnostics_are_published_to_the_project_and_cleared(cx: &mut gpui::TestAppContext) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            util::path!("/root"),
            serde_json::json!({
                "main.typ": "#include \"chapter.typ\"",
                "chapter.typ": "#undefined",
            }),
        )
        .await;
        let project = project::Project::test(fs, [util::path!("/root").as_ref()], cx).await;
        let window = cx.add_window(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = window
            .read_with(cx, |multi_workspace, _| {
                multi_workspace.workspace().downgrade()
            })
            .unwrap();

        let chapter = PathBuf::from(util::path!("/root/chapter.typ"));
        let diagnostics = vec![TypstDiagnostic {
            severity: DiagnosticSeverity::ERROR,
            message: "unknown variable: undefined".to_string(),
            file_path: Some(chapter.clone()),
            range: Some(1..10),
            point_range: Some(PointUtf16::new(0, 1)..PointUtf16::new(0, 10)),
        }];
        let mut published_paths = HashSet::default();
        cx.update(|cx| {
            publish_project_diagnostics(&workspace, None, &diagnostics, &mut published_paths, cx)
        });
        cx.run_until_parked();
        assert_eq!(published_paths, HashSet::from_iter([chapter]));
        let summary = project.read_with(cx, |project, cx| project.diagnostic_summary(false, cx));
        assert_eq!((summary.error_count, summary.warning_count), (1, 0));

        // a clean compile clears the previously published file.
        cx.update(|cx| {
            publish_project_diagnostics(&workspace, None, &[], &mut published_paths, cx)
        });
        cx.run_until_parked();
        assert!(published_paths.is_empty());
        let summary = project.read_with(cx, |project, cx| project.diagnostic_summary(false, cx));
        assert_eq!((summary.error_count, summary.warning_count), (0, 0));
    }

    #[gpui::test]
    async fn closing_the_preview_clears_its_diagnostics(cx: &mut gpui::TestAppContext) {
        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            util::path!("/root"),
            serde_json::json!({ "main.typ": "Hello #undefined_variable" }),
        )
        .await;
        let project = project::Project::test(fs, [util::path!("/root").as_ref()], cx).await;
        let window = cx.add_window(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = window
            .read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone())
            .unwrap();
        let cx = &mut gpui::VisualTestContext::from_window(window.into(), cx);

        let editor = workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_abs_path(
                    PathBuf::from(util::path!("/root/main.typ")),
                    workspace::OpenOptions::default(),
                    window,
                    cx,
                )
            })
            .await
            .unwrap()
            .downcast::<Editor>()
            .unwrap();
        let preview = workspace.update_in(cx, |workspace, window, cx| {
            TypstPreviewView::create_typst_view(workspace, editor, false, window, cx)
        });
        let error_count = |cx: &mut gpui::VisualTestContext| {
            project.read_with(cx, |project, cx| {
                project.diagnostic_summary(false, cx).error_count
            })
        };
        for _ in 0..500 {
            cx.run_until_parked();
            if error_count(cx) > 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(error_count(cx), 1, "the compile error should be published");

        drop(preview);
        // release observers run when the app next flushes its effects.
        cx.update(|_, _| {});
        cx.run_until_parked();
        assert_eq!(
            error_count(cx),
            0,
            "closing the preview should clear its diagnostics"
        );
    }

    #[test]
    fn export_path_sits_next_to_the_main_file() {
        let root = ScratchDir::new("export_path");
        let mut world = TypstSystemWorld::new(root.0.clone());
        world.update_main("= Hello".into(), &root.join("chapters/intro.typ"));
        assert_eq!(export_path(&world), Some(root.join("chapters/intro.pdf")));
    }

    #[gpui::test]
    async fn download_package_unpacks_the_registry_archive(cx: &mut gpui::TestAppContext) {
        use async_compression::futures::bufread::GzipEncoder;
        use futures::{AsyncReadExt as _, io::BufReader};

        let mut tar_bytes = Vec::new();
        let mut archive = async_tar::Builder::new(&mut tar_bytes);
        for (name, contents) in [
            ("typst.toml", "[package]\nname = \"demo\"\n"),
            ("lib.typ", "#let greet = [hi]"),
        ] {
            let mut header = async_tar::Header::new_gnu();
            header.set_size(contents.len() as u64);
            header.set_mode(0o644);
            archive
                .append_data(&mut header, name, contents.as_bytes())
                .await
                .unwrap();
        }
        archive.into_inner().await.unwrap();
        let mut gzipped = Vec::new();
        GzipEncoder::new(BufReader::new(tar_bytes.as_slice()))
            .read_to_end(&mut gzipped)
            .await
            .unwrap();

        let http_client = http_client::FakeHttpClient::create(move |request| {
            let gzipped = gzipped.clone();
            async move {
                assert_eq!(
                    request.uri().to_string(),
                    "https://packages.typst.org/preview/demo-1.0.0.tar.gz"
                );
                Ok(http_client::Response::new(gzipped.into()))
            }
        });
        let package: PackageSpec = "@preview/demo:1.0.0".parse().unwrap();
        let root = ScratchDir::new("download_package");
        let destination = root.join("preview/demo/1.0.0");

        cx.background_executor.allow_parking();
        download_package(http_client.as_ref(), &package, &destination)
            .await
            .unwrap();

        assert_eq!(
            std::fs::read_to_string(destination.join("lib.typ")).unwrap(),
            "#let greet = [hi]"
        );
        assert!(destination.join("typst.toml").exists());
    }

    #[test]
    fn resolve_project_finds_main_typ_in_an_ancestor_directory() {
        let root = ScratchDir::new("finds_main_typ");
        std::fs::create_dir_all(root.join("src/chapters")).unwrap();
        std::fs::write(root.join("main.typ"), "").unwrap();
        let active_path = root.join("src/chapters/intro.typ");
        std::fs::write(&active_path, "").unwrap();

        let (resolved_root, main_file) = TypstPreviewView::resolve_project(true, &active_path);

        assert_eq!(resolved_root, root.0);
        assert_eq!(main_file, root.join("main.typ"));
    }

    #[test]
    fn resolve_project_prefers_typst_toml_entrypoint_over_main_typ() {
        let root = ScratchDir::new("prefers_manifest_entrypoint");
        std::fs::write(
            root.join("typst.toml"),
            "[package]\nentrypoint = \"src/entry.typ\"\n",
        )
        .unwrap();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/entry.typ"), "").unwrap();
        // the manifest entrypoint wins over `main.typ`.
        std::fs::write(root.join("main.typ"), "").unwrap();
        let active_path = root.join("src/entry.typ");

        let (resolved_root, main_file) = TypstPreviewView::resolve_project(true, &active_path);

        assert_eq!(resolved_root, root.0);
        assert_eq!(main_file, root.join("src/entry.typ"));
    }

    #[test]
    fn resolve_project_falls_back_to_the_active_files_own_directory() {
        // without `typst.toml` or `main.typ` above, the directory of the file is the root.
        let root = ScratchDir::new("falls_back_when_nothing_found");
        let active_path = root.join("standalone.typ");
        std::fs::write(&active_path, "").unwrap();

        let (resolved_root, main_file) = TypstPreviewView::resolve_project(true, &active_path);

        assert_eq!(resolved_root, root.0);
        assert_eq!(main_file, active_path);
    }

    #[test]
    fn resolve_project_skips_the_filesystem_walk_when_disabled() {
        let root = ScratchDir::new("skips_walk_when_disabled");
        std::fs::write(root.join("main.typ"), "").unwrap();
        let active_path = root.join("standalone.typ");
        std::fs::write(&active_path, "").unwrap();

        // without project resolution, an adjacent `main.typ` is ignored.
        let (resolved_root, main_file) = TypstPreviewView::resolve_project(false, &active_path);

        assert_eq!(resolved_root, root.0);
        assert_eq!(main_file, active_path);
    }

    #[gpui::test]
    async fn exported_pdfs_carry_their_creation_date(cx: &mut gpui::TestAppContext) {
        cx.executor().allow_parking();
        let root = ScratchDir::new("export_date");
        let main_path = root.join("main.typ");
        let mut world = TypstSystemWorld::new(root.0.clone());
        world.update_main("Hello".into(), &main_path);
        let document = typst::compile::<PagedDocument>(&world)
            .output
            .expect("document should compile");

        let now = chrono::DateTime::parse_from_rfc3339("2024-03-05T06:07:08+02:00")
            .unwrap()
            .with_timezone(&chrono::Local);
        let bytes = export_bytes(&document, now).unwrap();
        let pdf = Pdf::from_bytes(bytes, None, None).await;
        let created = pdf.metadata.properties.created.clone();
        let expected = now.format("%Y-%m-%d %H:%M:%S %:z").to_string();
        assert_eq!(created.as_deref(), Some(expected.as_str()));
    }

    #[test]
    fn only_changes_to_files_the_document_read_trigger_a_recompile() {
        let root = ScratchDir::new("disk_changes");
        let main_path = root.join("main.typ");
        std::fs::write(root.join("chapter.typ"), "Chapter").unwrap();
        let mut world = TypstSystemWorld::new(root.0.clone());
        world.update_main("#include \"chapter.typ\"".into(), &main_path);
        typst::compile::<PagedDocument>(&world)
            .output
            .expect("document should compile");

        let affects = |path: &str, change, compile_failed| {
            changes_affect_world(
                &world,
                compile_failed,
                std::iter::once((root.join(path), change)),
            )
        };
        assert!(affects("chapter.typ", project::PathChange::Updated, false));
        assert!(!affects(
            "unrelated.typ",
            project::PathChange::Updated,
            false
        ));
        assert!(!affects("chapter.typ", project::PathChange::Loaded, false));
        assert!(!affects("image.png", project::PathChange::Added, false));
        assert!(affects("image.png", project::PathChange::Added, true));
    }

    #[test]
    fn editing_one_page_only_changes_its_hash() {
        let root = ScratchDir::new("page_hashes");
        let main_path = root.join("main.typ");
        let mut world = TypstSystemWorld::new(root.0.clone());

        let mut compile = |source: &str| {
            world.update_main(source.to_string().into(), &main_path);
            let document = typst::compile::<PagedDocument>(&world)
                .output
                .expect("document should compile");
            page_hashes(&document)
        };
        let before = compile("First page\n#pagebreak()\nSecond page\n#pagebreak()\nThird page\n");
        let after =
            compile("First page\n#pagebreak()\nSecond page, edited\n#pagebreak()\nThird page\n");

        assert_eq!(before.len(), 3);
        assert_eq!(before[0], after[0]);
        assert_ne!(before[1], after[1]);
        assert_eq!(before[2], after[2]);
    }

    #[test]
    fn resolve_typst_diagnostics_locates_the_error_in_the_main_file() {
        let root = ScratchDir::new("diagnostics_main_file_error");
        let main_path = root.join("main.typ");
        let source = "Hello #undefined_var\n";

        let mut world = TypstSystemWorld::new(root.0.clone());
        world.update_main(source.into(), &main_path);

        let compiled = typst::compile::<PagedDocument>(&world);
        let errors = compiled
            .output
            .expect_err("referencing an undefined variable should fail to compile");

        let diagnostics = resolve_typst_diagnostics(&world, &errors);
        assert!(!diagnostics.is_empty());

        let diagnostic = &diagnostics[0];
        assert_eq!(diagnostic.severity, DiagnosticSeverity::ERROR);
        assert_eq!(diagnostic.file_path.as_deref(), Some(main_path.as_path()));

        let range = diagnostic
            .range
            .clone()
            .expect("expected the error's span to resolve to a byte range");
        assert_eq!(&source[range], "undefined_var");
    }

    #[test]
    fn resolve_typst_diagnostics_locates_errors_in_included_files() {
        let root = ScratchDir::new("diagnostics_included_file_error");
        let main_path = root.join("main.typ");
        let other_path = root.join("other.typ");
        std::fs::write(&other_path, "Hello ä\n#undefined_var\n").unwrap();

        let mut world = TypstSystemWorld::new(root.0.clone());
        world.update_main("#include \"other.typ\"\n".into(), &main_path);

        let compiled = typst::compile::<PagedDocument>(&world);
        let errors = compiled
            .output
            .expect_err("the included file's undefined variable should fail to compile");

        let diagnostics = resolve_typst_diagnostics(&world, &errors);
        let in_other_file = diagnostics
            .iter()
            .find(|d| d.file_path.as_deref() == Some(other_path.as_path()));
        assert!(
            in_other_file.is_some(),
            "expected an error attributed to other.typ (not just main.typ), got: {:?}",
            diagnostics
                .iter()
                .map(|d| (&d.file_path, &d.message))
                .collect::<Vec<_>>()
        );
        // the second line, after the `#`, measured in UTF-16 units.
        assert_eq!(
            in_other_file.and_then(|d| d.point_range.clone()),
            Some(PointUtf16::new(1, 1)..PointUtf16::new(1, 14))
        );
    }

    #[test]
    fn text_leaf_span_covers_a_full_wrapped_run_not_just_its_first_line() {
        let root = ScratchDir::new("wrapped_leaf_spans_multiple_lines");
        let main_path = root.join("main.typ");
        // a narrow page wraps one unbroken text run, a single span, across several lines.
        let preamble = "#set page(width: 80pt, height: 400pt, margin: 4pt)\n";
        let sentence = "This is a long sentence that will definitely wrap across several lines on such a narrow page.";
        let source = format!("{preamble}{sentence}\n");

        let mut world = TypstSystemWorld::new(root.0.clone());
        world.update_main(source.clone().into(), &main_path);

        let compiled = typst::compile::<PagedDocument>(&world);
        let document = compiled.output.expect("fixture should compile");
        let typst_source = world.main_source().clone();

        // a cursor well past the first rendered line.
        let cursor = source
            .find("narrow page")
            .expect("fixture source contains this substring");

        let spans = paragraph_leaf_spans(&typst_source, cursor)
            .expect("cursor sits inside the sentence's plain-text leaf");
        assert_eq!(
            spans.len(),
            1,
            "one plain sentence with no *bold*/_italic_ splits is one leaf"
        );

        let page = document.pages().first().expect("fixture has one page");
        let mut points = Vec::new();
        find_all_in_frame(&page.frame, spans[0], &mut points);

        assert!(
            points.len() > 1,
            "expected the long sentence to word-wrap across multiple lines on an 80pt-wide \
             page, got {} line(s); the fixture may need a narrower page or longer sentence",
            points.len()
        );

        // `jump_from_cursor` resolves to the first line; the last line found must lie below it.
        let positions = typst_ide::jump_from_cursor(&document, &typst_source, cursor);
        let first_line_y = positions
            .last()
            .expect("should resolve a jump position")
            .point
            .y
            .to_pt();
        let last_line_y = points
            .iter()
            .map(|(p, _)| p.y.to_pt())
            .fold(f64::MIN, f64::max);
        assert!(
            last_line_y > first_line_y,
            "expected the sentence's last line to sit below its first, got first={first_line_y} last={last_line_y}"
        );
    }

    #[test]
    fn paragraph_highlight_is_consistent_on_either_side_of_a_bold_word() {
        let root = ScratchDir::new("paragraph_spans_bold_split");
        let main_path = root.join("main.typ");
        // the text before `*Typst*` fits one line, but the whole paragraph wraps, so leaf and
        // paragraph highlights differ.
        let preamble = "#set page(width: 320pt, height: 400pt, margin: 4pt)\n";
        let sentence = "This is a small test file to demonstrate basic *Typst* formatting. \
             It compiles incredibly fast and has a very clean syntax.";
        let source = format!("{preamble}{sentence}\n");

        let mut world = TypstSystemWorld::new(root.0.clone());
        world.update_main(source.clone().into(), &main_path);

        let compiled = typst::compile::<PagedDocument>(&world);
        let document = compiled.output.expect("fixture should compile");
        let typst_source = world.main_source().clone();
        let page = document.pages().first().expect("fixture has one page");

        // cursors on either side of the bold word hit different leaves of the same paragraph.
        let before_bold = source.find("small").expect("fixture contains this word");
        let after_bold = source.find("compiles").expect("fixture contains this word");

        let before_spans = paragraph_leaf_spans(&typst_source, before_bold)
            .expect("cursor sits inside a plain-text leaf");
        let after_spans = paragraph_leaf_spans(&typst_source, after_bold)
            .expect("cursor sits inside a plain-text leaf");

        // the paragraph must consist of several leaves for this test to be meaningful.
        assert_eq!(
            before_spans.len(),
            3,
            "expected text/Typst/text as three merged leaves"
        );

        let before_set: std::collections::HashSet<_> = before_spans.iter().copied().collect();
        let after_set: std::collections::HashSet<_> = after_spans.iter().copied().collect();
        assert_eq!(
            before_set, after_set,
            "clicking either side of *Typst* should resolve to the same whole-paragraph span set"
        );

        let bounds = |spans: &[Span]| {
            let mut points = Vec::new();
            for &span in spans {
                find_all_in_frame(&page.frame, span, &mut points);
            }
            let min_y = points
                .iter()
                .map(|(p, _)| p.y.to_pt())
                .fold(f64::MAX, f64::min);
            let max_y = points
                .iter()
                .map(|(p, _)| p.y.to_pt())
                .fold(f64::MIN, f64::max);
            (min_y, max_y)
        };
        let before_bounds = bounds(&before_spans);
        let after_bounds = bounds(&after_spans);
        assert_eq!(
            before_bounds, after_bounds,
            "the rendered block extent must be identical regardless of which leaf was clicked"
        );

        // the paragraph must wrap for this test to be meaningful.
        let (min_y, max_y) = before_bounds;
        assert!(
            max_y > min_y,
            "expected the paragraph to wrap across multiple lines, got a single-line bounding \
             box (min={min_y} max={max_y}); the fixture may need a narrower page"
        );
    }

    #[test]
    fn line_height_for_font_size_scales_with_the_font_size() {
        assert_eq!(line_height_for_font_size(Abs::pt(10.0)), 14.0);
        assert_eq!(line_height_for_font_size(Abs::pt(20.0)), 28.0);
    }

    #[test]
    fn highlight_line_height_tracks_the_documents_actual_font_size_not_a_fixed_guess() {
        let root = ScratchDir::new("line_height_tracks_font_size");

        // documents differing only in font size must yield different line heights.
        let render_line_height = |name: &str, font_size_pt: u32| -> f32 {
            let main_path = root.join(&format!("{name}.typ"));
            let source = format!(
                "#set page(width: 200pt, height: 200pt, margin: 4pt)\n\
                 #set text(size: {font_size_pt}pt)\n\
                 Some plain paragraph text.\n"
            );
            let mut world = TypstSystemWorld::new(root.0.clone());
            world.update_main(source.clone().into(), &main_path);
            let compiled = typst::compile::<PagedDocument>(&world);
            let document = compiled.output.expect("fixture should compile");
            let typst_source = world.main_source().clone();
            let page = document.pages().first().expect("fixture has one page");

            let cursor = source
                .find("plain")
                .expect("fixture source contains this word");
            let spans = paragraph_leaf_spans(&typst_source, cursor)
                .expect("cursor sits inside the paragraph's plain-text leaf");
            let mut hits = Vec::new();
            for span in spans {
                find_all_in_frame(&page.frame, span, &mut hits);
            }
            let (_, size) = hits
                .first()
                .expect("expected at least one rendered glyph run");
            line_height_for_font_size(*size)
        };

        let small = render_line_height("small", 10);
        let large = render_line_height("large", 20);

        assert!(
            large > small * 1.5,
            "expected doubling the font size to noticeably grow the derived line height, got \
             small={small} large={large}"
        );
    }
}
