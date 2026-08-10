use crate::typst_world::TypstSystemWorld;
use crate::{
    OpenFixedFollowingPreview, OpenFixedPreview, OpenFixedPreviewToTheSide, OpenFollowingPreview,
    OpenPreview, OpenPreviewToTheSide,
};
use anyhow::Result;
use editor::scroll::Autoscroll;
use editor::{Editor, EditorEvent, MultiBufferOffset, SelectionEffects};
use text::Point as BufferPoint;
use gpui::{
    AnyEntity, App, Bounds, Context, Entity, EntityId, EventEmitter, FocusHandle, Focusable, Font,
    IntoElement, Point, Render, Size, Subscription, Task, WeakEntity, Window, div,
};
use language::{
    Diagnostic, DiagnosticEntry, DiagnosticSet, DiagnosticSeverity, HighlightedText,
    LanguageServerId,
};
use parking_lot::Mutex;
use pdf_viewer::pdf::Pdf;
use pdf_viewer::pdf_renderer::{PageMetadata, PdfTarget};
use pdf_viewer::{PdfView, ScrollAnchor, ScrollMode};
use std::any::TypeId;
use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use typst::layout::{Abs, Frame, FrameItem, Point as TypstPoint};
use typst::syntax::{LinkedNode, Side, Source, Span, SyntaxKind};
use typst::{World, layout::PagedDocument};
use ui::prelude::*;
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
    active_source_index: Option<usize>,
    compile_state: TypstCompileState,
    /// Warnings from the most recent successful compile. Typst can produce
    /// these even when compilation otherwise succeeds; they don't fail the
    /// compile so they don't flow through TypstCompileState::Failed.
    compile_warnings: Vec<String>,
    pending_update_task: Option<Task<Result<()>>>,
    mode: TypstPreviewMode,
    project_resolution: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TypstPreviewMode {
    /// The preview will always show the contents of the provided editor.
    Default,
    /// The preview will "follow" the currently active editor.
    Follow,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TypstCompileState {
    Uninitialized,
    Compiling,
    Rendering,
    Finished,
    Failed(String),
}

/// A fake language server id reserved for diagnostics `typst_preview` publishes itself
/// (typst compile errors/warnings), distinct from any real language server that might also
/// be attached to the same `.typ` buffer (e.g. a user-installed Typst LSP extension) so
/// neither source clobbers the other's diagnostics on the same buffer.
const TYPST_PREVIEW_DIAGNOSTICS_SERVER_ID: LanguageServerId = LanguageServerId(usize::MAX - 100);

/// A Typst compile diagnostic (error or warning) with its source location resolved to a real
/// file path and byte range, rather than just a flattened message string. `file_path`/`range`
/// are `None` when the diagnostic's `Span` doesn't resolve to a location (Typst does emit
/// spanless diagnostics in some cases). The message is still shown, just without a squiggle.
struct TypstDiagnostic {
    severity: DiagnosticSeverity,
    message: String,
    file_path: Option<PathBuf>,
    range: Option<Range<usize>>,
}

/// Resolves each diagnostic's `Span` to a `(file_path, byte_range)` via the same `World` the
/// compile just ran against, while `world`/its sources are still in scope. Must run
/// synchronously right after `typst::compile`. `Span`s aren't meaningful once `world` is
/// dropped or mutated again (e.g. by the next debounced recompile).
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
                let source = world.source(file_id).ok()?;
                let range = source.range(diagnostic.span)?;
                Some((path, range))
            });
            TypstDiagnostic {
                severity,
                message: diagnostic.message.to_string(),
                file_path: location.as_ref().map(|(path, _)| path.clone()),
                range: location.map(|(_, range)| range),
            }
        })
        .collect()
}

/// Publishes `diagnostics` as inline squiggles on `editor`'s buffer, replacing whatever
/// `typst_preview` published there last time (an empty `diagnostics` list correctly clears
/// stale ones once an error is fixed, since `Buffer::update_diagnostics` replaces the whole
/// set for a given server id rather than appending to it). Only diagnostics whose resolved
/// file matches the editor's own file are applied. Diagnostics for other files (e.g. an
/// `#import`ed file that isn't open) are dropped here; they're still visible in the
/// compile-warnings banner in the preview pane itself.
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
            Some(DiagnosticEntry {
                range: snapshot.offset_to_point_utf16(range.start)
                    ..snapshot.offset_to_point_utf16(range.end),
                diagnostic: Diagnostic {
                    source: Some("Typst".to_string()),
                    severity: d.severity,
                    message: d.message.clone(),
                    is_primary: true,
                    ..Default::default()
                },
            })
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

/// Whether a syntax node of `kind`, as a sibling within a paragraph's flow, still belongs to
/// the *same* paragraph as its neighbors. `false` for whatever actually starts a new block
/// (a blank-line paragraph break, a heading, a list/enum/term item, a math block/equation).
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

/// Walks up from `leaf` through inline-formatting wrappers (`*Strong*`, `_Emph_`) to the node
/// whose direct children are the actual paragraph-level flow `leaf` sits in: plain text
/// interleaved with inline formatting, but not yet past whatever block-level container (a
/// heading's own text, a list item's own text, the document root, ...) that paragraph itself
/// lives inside. Needed because `*Typst*` inside a sentence puts "Typst" one level deeper in
/// the tree (inside `Strong`'s own nested content) than the plain text on either side of it,
/// which otherwise splits one visual paragraph into several unrelated per-leaf lookups.
fn paragraph_container<'a>(leaf: &LinkedNode<'a>) -> Option<LinkedNode<'a>> {
    let mut container = leaf.parent()?.clone();
    while let Some(parent) = container.parent()
        && matches!(parent.kind(), SyntaxKind::Strong | SyntaxKind::Emph)
    {
        container = parent.parent()?.clone();
    }
    Some(container)
}

/// Every leaf's span (recursively, so nested `Strong`/`Emph` content is included) that
/// belongs to the same paragraph as the plain-text leaf containing `cursor`, not just that
/// one leaf's own span. A "leaf" is one run of plain text between markup elements (e.g. the
/// text on either side of a `*bold*` word), the granularity Typst's own span tracking
/// supports; a paragraph is usually several of these siblings back to back, so highlighting
/// only the single leaf under the cursor made the block-highlight's extent depend on which
/// side of a `*bold*`/`_italic_` word you clicked, inconsistent for what's visually one
/// paragraph. `None` if `cursor` isn't inside any text at all.
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
    /// The direct child of `container` that `leaf` descends from (`leaf` itself, if it's
    /// already a direct child).
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

/// Every point in `frame` (recursively through nested groups) where a text run's glyphs
/// carry `span`. Mirrors `typst_ide::jump_from_cursor`'s private `find_in_frame`, except it
/// keeps scanning every frame item instead of returning as soon as it finds the first
/// match. A leaf's span is shared across however many lines it word-wraps to, so this is
/// what lets the preview highlight the whole block instead of only ever landing on its
/// first line the way a single `find_in_frame` call does.
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

/// A rough one-line highlight-box height for text set at `font_size`, not real typographic
/// ascent/descent metrics (`Frame`/`TextItem` doesn't expose those here), just a multiple of
/// the nominal size big enough to cover a line's glyphs plus a little breathing room. Tracks
/// the paragraph's *actual* font size, unlike a single fixed guess that doesn't.
fn line_height_for_font_size(font_size: Abs) -> f32 {
    font_size.to_pt() as f32 * 1.4
}

struct EditorState {
    editor: Entity<Editor>,
    _subscription: Subscription,
}

impl TypstPreviewView {
    /// Get either a pane in the given direction or the active pane
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
            let pdf_view = cx.new(|cx| PdfView::from_pdf(empty_pdf, window, cx));

            let mut this = Self {
                active_editor: None,
                observed_editors: Default::default(),
                focus_handle: cx.focus_handle(),
                workspace: workspace_handle.clone(),
                pdf_view: pdf_view,
                world: None,
                document: None,
                active_source_index: None,
                compile_state: TypstCompileState::Uninitialized,
                compile_warnings: Vec::new(),
                pending_update_task: None,
                mode,
                project_resolution,
            };

            this.set_editor(active_editor, window, cx);

            // Always observe the workspace so we can hook into typing on included files
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
            } else {
                log::error!("Failed to listen to workspace updates");
            }

            this
        })
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
            // Drop the old task to cancel it and reset the debounce timer
            // so we don't start rendering while the user is still actively typing.
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

    /// Walks up from `active_path` looking for a `typst.toml` (using its manifest
    /// `entrypoint`) or a `main.typ` to resolve the project root and main file. Does real
    /// filesystem I/O (`exists`/`read_to_string` per directory up to the root). Callers
    /// must run this on a background thread, not synchronously on the UI thread.
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
                && let Ok(manifest) = toml_content.parse::<toml::Value>()
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
        max_width: f32,
        selection_start: usize,
        should_reveal: bool,
        active_path: PathBuf,
        workspace_weak: WeakEntity<Workspace>,
        cx: &mut Context<Self>,
    ) {
        view.world = Some(new_world_arc.clone());
        view.document = Some(document.clone());

        cx.spawn(async move |view, cx| {
            let pdf = Arc::new(Pdf::from_bytes(bytes, Some((pages, max_width)), None).await);
            view.update(cx, |view, cx| {
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

                        if let Some(page) = document_clone.pages.get(page_index) {
                            let world = world_arc.lock();
                            if let Some(typst_ide::Jump::File(file_id, offset)) =
                                typst_ide::jump_from_click(
                                    &*world,
                                    &document_clone,
                                    &page.frame,
                                    typst_point,
                                )
                                && let Some(path_buf) = world.resolve_path(file_id).ok()
                            {
                                let normalized_path = util::normalize_path(&path_buf);
                                if let Some(workspace) = workspace_weak.upgrade() {
                                    // Is the file already open in any pane? Just update its
                                    // cursor, don't focus it or change the active pane/tab,
                                    // so clicking around in the PDF preview doesn't keep
                                    // yanking focus away from it (matches how the Markdown
                                    // preview's own click-to-source navigation behaves).
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

                                    // Not open anywhere yet, open it without focusing it,
                                    // same reasoning as above.
                                    window
                                        .spawn(cx, async move |cx| {
                                            cx.background_executor()
                                                .timer(std::time::Duration::from_millis(10))
                                                .await;

                                            let open_task_res = cx.update(|window, cx| {
                                                workspace.update(cx, |ws, cx| {
                                                    ws.open_abs_path(
                                                        normalized_path,
                                                        workspace::OpenOptions {
                                                            focus: Some(false),
                                                            ..Default::default()
                                                        },
                                                        window,
                                                        cx,
                                                    )
                                                })
                                            });

                                            if let Ok(open_task) = open_task_res
                                                && let Ok(item) = open_task.await
                                            {
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
                // Wait for the user to stop typing
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
                    view.compile_state = TypstCompileState::Failed(
                        "Save this file to preview it".to_string(),
                    );
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
                                if let Some(file) = editor_read.file_at(MultiBufferOffset(0), cx)
                                    && let Some(local) = file.as_local()
                                {
                                    let snapshot = editor_read.buffer().read(cx).snapshot(cx);
                                    map.insert(local.abs_path(cx).to_path_buf(), snapshot);
                                }
                            }
                        }
                    }
                    map
                } else {
                    HashMap::new()
                };

                // Prioritize the exact state of the currently active editor
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

            if let Some((selection_start, active_path, project_resolution, final_overrides, current_world)) =
                update
            {
                let active_path_sync = active_path.clone();
                let workspace_weak = view.update(cx, |v, _| v.workspace.clone())?;
                let active_editor = view.update(cx, |v, _| {
                    v.active_editor.as_ref().map(|s| s.editor.clone())
                })?;

                struct CompileOutcome {
                    diagnostics: Vec<TypstDiagnostic>,
                    result: Result<(
                        Vec<u8>,
                        Arc<PagedDocument>,
                        Arc<Mutex<TypstSystemWorld>>,
                        Vec<PageMetadata>,
                        f32,
                    )>,
                }

                let compile_outcome: CompileOutcome = cx
                    .background_executor()
                    .spawn(async move {
                        let start = Instant::now();

                        // Off the UI thread: walks parent directories looking for
                        // typst.toml/main.typ (see resolve_project's doc comment).
                        let (new_root, main_file_path) =
                            Self::resolve_project(project_resolution, &active_path);

                        let world_arc = match current_world.filter(|w| w.lock().root == new_root) {
                            Some(world) => world,
                            None => Arc::new(Mutex::new(TypstSystemWorld::new(new_root))),
                        };

                        // Extract main text fully off the UI thread
                        let main_contents = match final_overrides.get(&main_file_path) {
                            Some(snap) => snap.text(),
                            None => match std::fs::read_to_string(&main_file_path) {
                                Ok(text) => text,
                                Err(e) => {
                                    return CompileOutcome {
                                        diagnostics: Vec::new(),
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

                        log::info!("Prepared compilation in {:?}", start.elapsed());
                        let start = Instant::now();

                        // Compile the document using the comemo-cached world
                        let compiled = typst::compile(&*world);
                        let mut diagnostics = resolve_typst_diagnostics(&world, &compiled.warnings);
                        for diagnostic in &diagnostics {
                            log::warn!("Typst compile warning: {}", diagnostic.message);
                        }

                        log::info!("Compiled in {:?}", start.elapsed());
                        let start = Instant::now();

                        let document = match compiled.output {
                            Ok(document) => document,
                            Err(errors) => {
                                diagnostics.extend(resolve_typst_diagnostics(&world, &errors));
                                return CompileOutcome {
                                    diagnostics,
                                    result: Err(anyhow::anyhow!(
                                        "Compilation failed:\n{:?}",
                                        errors
                                    )),
                                };
                            }
                        };

                        log::info!("Generated output in {:?}", start.elapsed());
                        let start = Instant::now();

                        // Export to PDF bytes
                        let bytes = match typst_pdf::pdf(&document, &typst_pdf::PdfOptions::default())
                        {
                            Ok(bytes) => bytes,
                            Err(e) => {
                                return CompileOutcome {
                                    diagnostics,
                                    result: Err(anyhow::anyhow!("PDF export failed:\n{:?}", e)),
                                };
                            }
                        };

                        log::info!("Generated PDF in {:?}", start.elapsed());
                        let start = Instant::now();

                        let mut max_width = 0.0f32;
                        let mut pages = Vec::with_capacity(document.pages.len());

                        for (i, page) in document.pages.iter().enumerate() {
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

                        log::info!(
                            "Page sizes extracted for {} pages in {:?}",
                            document.pages.len(),
                            start.elapsed()
                        );

                        let world_arc_ret = world_arc.clone();
                        CompileOutcome {
                            diagnostics,
                            result: Ok((bytes, Arc::new(document), world_arc_ret, pages, max_width)),
                        }
                    })
                    .await;

                view.update(cx, move |view, cx| {
                    view.compile_state = TypstCompileState::Rendering;
                    publish_editor_diagnostics(active_editor.as_ref(), &compile_outcome.diagnostics, cx);
                    match compile_outcome.result {
                        Ok((bytes, document, new_world_arc, pages, max_width)) => {
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
                    view.pending_update_task = None;
                })
                .ok();
            }

            Ok(())
        })
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
        self.active_source_index = Some(source_index);
        self.sync_active_root_block(cx);

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
                        let vpath = typst::syntax::VirtualPath::new(path_str);
                        let file_id = typst::syntax::FileId::new(None, vpath);
                        world_lock.source(file_id).ok()
                    }
                } else {
                    Some(world_lock.main_source().clone())
                };

                if let Some(source) = source {
                    let safe_index = source_index.min(source.text().len());
                    let positions = typst_ide::jump_from_cursor(document, &source, safe_index);
                    if let Some(position) = positions.last() {
                        let page_index = position.page.get() - 1;

                        // jump_from_cursor only ever resolves to the position of the single
                        // syntax leaf (a run of plain text between markup elements) the
                        // cursor happens to be in. A paragraph is usually several such
                        // leaves back to back (e.g. split by a *bold* word), so anchoring the
                        // highlight to just that one leaf's own extent made it depend on
                        // which side of the bold word was clicked, inconsistent for what's
                        // visually one paragraph. Recompute both the anchor point and the
                        // block's bottom from every leaf in the whole paragraph instead.
                        let paragraph_hits: Vec<(TypstPoint, Abs)> = paragraph_leaf_spans(&source, safe_index)
                            .and_then(|spans| {
                                let page = document.pages.get(page_index)?;
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
                        // A rough one-line box height (covering a line's ascent/descent, not
                        // just its baseline) derived from the paragraph's own font size,
                        // rather than a fixed guess that doesn't track the document's actual
                        // text size and made the highlight over/undershoot real lines.
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
                        log::warn!("Other positions ignored:");
                        for position in positions.iter() {
                            log::warn!("pos: {:?}", position);
                        }
                        return;
                    }
                }
            }
        }
    }

    fn sync_active_root_block(&mut self, _cx: &mut Context<Self>) {
        // self.pdf_view.update(cx, |pdf_view, cx| {
        //     pdf_view.set_active_root_for_source_index(self.active_source_index, cx);
        // });
        // if let Some(pdf_view) = &self.pdf_view {
        // TODO: Implement Forward Search (scroll to element matching source index)
        // pdf_view.update(cx, |pv, cx| pv.scroll_to_source_index(self.active_source_index, cx));
        // }
    }

    /// Moves `editor`'s cursor to the start of the line containing `source_index`, without
    /// focusing `editor` or otherwise changing which pane/tab is active. Clicking around in
    /// the PDF preview shouldn't keep yanking focus away from it. Snaps to the line rather
    /// than the exact clicked character: matches how the Markdown preview's own
    /// click-to-source navigation behaves (see `change_selection_to_source_index` there), and
    /// is more robust than an exact offset, which doesn't mean much once a paragraph has
    /// reflowed across the PDF's own line breaks anyway.
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
                            let message = self
                                .compile_warnings
                                .first()
                                .cloned()
                                .unwrap_or_default();
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch directory tree that removes itself on drop.
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

    #[test]
    fn resolve_project_finds_main_typ_in_an_ancestor_directory() {
        let root = ScratchDir::new("finds_main_typ");
        std::fs::create_dir_all(root.join("src/chapters")).unwrap();
        std::fs::write(root.join("main.typ"), "").unwrap();
        let active_path = root.join("src/chapters/intro.typ");
        std::fs::write(&active_path, "").unwrap();

        let (resolved_root, main_file) =
            TypstPreviewView::resolve_project(true, &active_path);

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
        // A main.typ also exists, but the manifest's entrypoint should win.
        std::fs::write(root.join("main.typ"), "").unwrap();
        let active_path = root.join("src/entry.typ");

        let (resolved_root, main_file) =
            TypstPreviewView::resolve_project(true, &active_path);

        assert_eq!(resolved_root, root.0);
        assert_eq!(main_file, root.join("src/entry.typ"));
    }

    #[test]
    fn resolve_project_falls_back_to_the_active_files_own_directory() {
        // No typst.toml/main.typ anywhere above this file, so resolution should fall back
        // to treating the active file's own directory as the root.
        let root = ScratchDir::new("falls_back_when_nothing_found");
        let active_path = root.join("standalone.typ");
        std::fs::write(&active_path, "").unwrap();

        let (resolved_root, main_file) =
            TypstPreviewView::resolve_project(true, &active_path);

        assert_eq!(resolved_root, root.0);
        assert_eq!(main_file, active_path);
    }

    #[test]
    fn resolve_project_skips_the_filesystem_walk_when_disabled() {
        let root = ScratchDir::new("skips_walk_when_disabled");
        std::fs::write(root.join("main.typ"), "").unwrap();
        let active_path = root.join("standalone.typ");
        std::fs::write(&active_path, "").unwrap();

        // Even though a main.typ exists right there, project_resolution: false must not
        // walk the filesystem at all and should just treat the file as standalone.
        let (resolved_root, main_file) =
            TypstPreviewView::resolve_project(false, &active_path);

        assert_eq!(resolved_root, root.0);
        assert_eq!(main_file, active_path);
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
        std::fs::write(&other_path, "Hello #undefined_var\n").unwrap();

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
    }

    #[test]
    fn text_leaf_span_covers_a_full_wrapped_run_not_just_its_first_line() {
        let root = ScratchDir::new("wrapped_leaf_spans_multiple_lines");
        let main_path = root.join("main.typ");
        // A narrow page plus one long run of plain text (no markup breaking it into more
        // than one syntax leaf) forces word-wrap across several rendered lines while it all
        // stays a single leaf/span as far as Typst's own source tracking is concerned.
        let preamble = "#set page(width: 80pt, height: 400pt, margin: 4pt)\n";
        let sentence = "This is a long sentence that will definitely wrap across several lines on such a narrow page.";
        let source = format!("{preamble}{sentence}\n");

        let mut world = TypstSystemWorld::new(root.0.clone());
        world.update_main(source.clone().into(), &main_path);

        let compiled = typst::compile::<PagedDocument>(&world);
        let document = compiled.output.expect("fixture should compile");
        let typst_source = world.main_source().clone();

        // A cursor near the end of the sentence, well past whatever its first rendered line
        // covers.
        let cursor = source
            .find("narrow page")
            .expect("fixture source contains this substring");

        let spans = paragraph_leaf_spans(&typst_source, cursor)
            .expect("cursor sits inside the sentence's plain-text leaf");
        assert_eq!(spans.len(), 1, "one plain sentence with no *bold*/_italic_ splits is one leaf");

        let page = document.pages.first().expect("fixture has one page");
        let mut points = Vec::new();
        find_all_in_frame(&page.frame, spans[0], &mut points);

        assert!(
            points.len() > 1,
            "expected the long sentence to word-wrap across multiple lines on an 80pt-wide \
             page, got {} line(s); the fixture may need a narrower page or longer sentence",
            points.len()
        );

        // jump_from_cursor (what the highlight relied on exclusively before this fix)
        // always resolves to the *first* line's position. Confirm the last line found by
        // find_all_in_frame sits strictly below it, which is exactly the extra vertical
        // extent the old fixed-height highlight bar couldn't cover.
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
        // Wide enough that the plain-text run *before* "*Typst*" fits on one line by itself,
        // but the whole paragraph (both text runs plus the bold word) still wraps across more
        // than one line, so this test can actually tell "just the clicked leaf's own lines"
        // (the old, inconsistent behavior) apart from "the whole paragraph's lines".
        let preamble = "#set page(width: 320pt, height: 400pt, margin: 4pt)\n";
        let sentence = "This is a small test file to demonstrate basic *Typst* formatting. \
             It compiles incredibly fast and has a very clean syntax.";
        let source = format!("{preamble}{sentence}\n");

        let mut world = TypstSystemWorld::new(root.0.clone());
        world.update_main(source.clone().into(), &main_path);

        let compiled = typst::compile::<PagedDocument>(&world);
        let document = compiled.output.expect("fixture should compile");
        let typst_source = world.main_source().clone();
        let page = document.pages.first().expect("fixture has one page");

        // Two cursors either side of the bold word: different syntax leaves, but visually
        // the same one paragraph.
        let before_bold = source.find("small").expect("fixture contains this word");
        let after_bold = source.find("compiles").expect("fixture contains this word");

        let before_spans = paragraph_leaf_spans(&typst_source, before_bold)
            .expect("cursor sits inside a plain-text leaf");
        let after_spans = paragraph_leaf_spans(&typst_source, after_bold)
            .expect("cursor sits inside a plain-text leaf");

        // Sanity: the paragraph is genuinely made of more than one syntax leaf (the plain
        // text before "*Typst*", the bold word itself, and the plain text after it): three
        // leaves merged into one paragraph's worth of spans, not a single trivial leaf that
        // would make this fixture unable to exercise the bug being fixed at all.
        assert_eq!(before_spans.len(), 3, "expected text/Typst/text as three merged leaves");

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
            let min_y = points.iter().map(|(p, _)| p.y.to_pt()).fold(f64::MAX, f64::min);
            let max_y = points.iter().map(|(p, _)| p.y.to_pt()).fold(f64::MIN, f64::max);
            (min_y, max_y)
        };
        let before_bounds = bounds(&before_spans);
        let after_bounds = bounds(&after_spans);
        assert_eq!(
            before_bounds, after_bounds,
            "the rendered block extent must be identical regardless of which leaf was clicked"
        );

        // Sanity: the paragraph actually wraps across multiple lines, otherwise this test
        // couldn't distinguish "just the clicked leaf's own lines" from "the whole
        // paragraph's lines" in the first place.
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

        // Two otherwise-identical documents, differing only in body text size. The derived
        // line height must scale accordingly rather than landing on the same fixed value.
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
            let page = document.pages.first().expect("fixture has one page");

            let cursor = source
                .find("plain")
                .expect("fixture source contains this word");
            let spans = paragraph_leaf_spans(&typst_source, cursor)
                .expect("cursor sits inside the paragraph's plain-text leaf");
            let mut hits = Vec::new();
            for span in spans {
                find_all_in_frame(&page.frame, span, &mut hits);
            }
            let (_, size) = hits.first().expect("expected at least one rendered glyph run");
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
