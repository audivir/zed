use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;
use editor::Editor;
use file_icons::FileIcons;
use gpui::{
    AnyElement, App, Bounds, Context, DispatchPhase, Element, ElementId, Entity, EventEmitter,
    FocusHandle, Focusable, GlobalElementId, InspectorElementId, IntoElement, LayoutId,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement, ParsedSvg,
    PinchEvent, Pixels, Point, Render, RenderImage, ScrollDelta, ScrollEvent, Size, Style, Styled,
    Subscription, SvgSize, Task, WeakEntity, Window, checkerboard, div, img, point, px, size,
};
use image_viewer::{FitToView, ResetZoom, ZoomIn, ZoomOut, ZoomToActualSize};
use language::{Buffer, BufferEvent};
use multi_buffer::MultiBuffer;
use project::{Project, ProjectEntryId, ProjectPath};
use ui::{Tooltip, prelude::*};
use workspace::invalid_item_view::InvalidItemView;
use workspace::item::{Item, ItemHandle, ProjectItem, SerializableItem};
use workspace::notifications::NotifyResultExt;
use workspace::{
    ItemId, Pane, ToolbarItemEvent, ToolbarItemLocation, ToolbarItemView, Workspace, WorkspaceId,
    delete_unloaded_items,
};

use crate::{OpenAsText, OpenFollowingPreview, OpenPreview, OpenPreviewToTheSide};
use persistence::SvgPreviewDb;

const MIN_ZOOM: f32 = 0.1;
const MAX_ZOOM: f32 = 20.0;
const ZOOM_STEP: f32 = 1.1;
const SCROLL_LINE_MULTIPLIER: f32 = 20.0;
const BASE_SQUARE_SIZE: f32 = 32.0;
/// Largest side of a rasterized SVG in device pixels, matching the limit of the SVG renderer.
const MAX_RASTER_SIZE: f32 = 8192.0;
/// Delay before re-rasterizing after a zoom change, so continuous zooming rasterizes once.
const RASTERIZE_DEBOUNCE: Duration = Duration::from_millis(80);

/// SVG file opened from the project, backed by its text buffer so edits show up live.
pub struct SvgFile {
    buffer: Entity<Buffer>,
}

impl project::ProjectItem for SvgFile {
    fn try_open(
        project: &Entity<Project>,
        path: &ProjectPath,
        cx: &mut App,
    ) -> Option<Task<anyhow::Result<Entity<Self>>>> {
        // For single-file worktrees the relative path is empty, so fall back
        // to the absolute path to detect SVGs opened directly.
        let is_svg = SvgPreviewView::is_svg_path(path.path.as_std_path())
            || project
                .read(cx)
                .absolute_path(path, cx)
                .is_some_and(|abs_path| SvgPreviewView::is_svg_path(&abs_path));
        if !is_svg {
            return None;
        }

        let open_buffer = project.update(cx, |project, cx| project.open_buffer(path.clone(), cx));
        Some(cx.spawn(async move |cx| {
            let buffer = open_buffer.await?;
            Ok(cx.new(|_| SvgFile { buffer }))
        }))
    }

    fn entry_id(&self, cx: &App) -> Option<ProjectEntryId> {
        project::ProjectItem::entry_id(self.buffer.read(cx), cx)
    }

    fn project_path(&self, cx: &App) -> Option<ProjectPath> {
        project::ProjectItem::project_path(self.buffer.read(cx), cx)
    }

    fn is_dirty(&self) -> bool {
        // Unsaved changes belong to the text editors of the buffer, which prompt for them.
        false
    }
}

/// SVG rasterized for display at a specific width.
struct RasterizedSvg {
    image: Arc<RenderImage>,
    width: u32,
    /// The parse this was rasterized from, so changed content is rasterized again even when its
    /// width stays the same.
    source: Arc<ParsedSvg>,
}

pub struct SvgPreviewView {
    focus_handle: FocusHandle,
    buffer: Option<Entity<Buffer>>,
    /// Set when the view shows an SVG file opened from the project, rather than previewing an
    /// editor.
    svg_file: Option<Entity<SvgFile>>,
    parsed_svg: Option<Result<Arc<ParsedSvg>, SharedString>>,
    rasterized_svg: Option<RasterizedSvg>,
    /// Width of the rasterization in flight, so prepaint does not request it again every frame.
    pending_raster_width: Option<u32>,
    zoom_level: f32,
    pan_offset: Point<Pixels>,
    last_mouse_position: Option<Point<Pixels>>,
    container_bounds: Option<Bounds<Pixels>>,
    /// Whether the next layout should fit the SVG to the view, once its size is known.
    fit_on_layout: bool,
    _parse: Task<()>,
    _rasterize: Task<()>,
    _buffer_subscription: Option<Subscription>,
    _workspace_subscription: Option<Subscription>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SvgPreviewMode {
    /// The preview will always show the contents of the provided editor.
    Default,
    /// The preview will "follow" the last active editor of an SVG file.
    Follow,
}

impl SvgPreviewView {
    pub fn new(
        mode: SvgPreviewMode,
        active_buffer: Entity<MultiBuffer>,
        workspace_handle: WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) -> Entity<Self> {
        cx.new(|cx| {
            let workspace_subscription = if mode == SvgPreviewMode::Follow
                && let Some(workspace) = workspace_handle.upgrade()
            {
                Some(Self::subscribe_to_workspace(workspace, window, cx))
            } else {
                None
            };

            let buffer = active_buffer.read_with(cx, |buffer, _cx| buffer.as_singleton());
            let mut this = Self::with_buffer(buffer, None, window, cx);
            this._workspace_subscription = workspace_subscription;
            this
        })
    }

    fn with_buffer(
        buffer: Option<Entity<Buffer>>,
        svg_file: Option<Entity<SvgFile>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let subscription = buffer
            .as_ref()
            .map(|buffer| Self::create_buffer_subscription(buffer, window, cx));

        let mut this = Self {
            focus_handle: cx.focus_handle(),
            buffer,
            svg_file,
            parsed_svg: None,
            rasterized_svg: None,
            pending_raster_width: None,
            zoom_level: 1.0,
            pan_offset: Point::default(),
            last_mouse_position: None,
            container_bounds: None,
            fit_on_layout: true,
            _parse: Task::ready(()),
            _rasterize: Task::ready(()),
            _buffer_subscription: subscription,
            _workspace_subscription: None,
        };
        this.parse_svg(window, cx);
        this
    }

    fn subscribe_to_workspace(
        workspace: Entity<Workspace>,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> Subscription {
        cx.subscribe_in(
            &workspace,
            window,
            move |this: &mut SvgPreviewView, workspace, event: &workspace::Event, window, cx| {
                if let workspace::Event::ActiveItemChanged = event {
                    let workspace = workspace.read(cx);
                    if let Some(active_item) = workspace.active_item(cx)
                        && let Some(buffer) = active_item.downcast::<MultiBuffer>()
                        && Self::is_svg_file(&buffer, cx)
                    {
                        let Some(buffer) = buffer.read(cx).as_singleton() else {
                            return;
                        };
                        if this.buffer.as_ref() != Some(&buffer) {
                            this._buffer_subscription =
                                Some(Self::create_buffer_subscription(&buffer, window, cx));
                            this.buffer = Some(buffer);
                            this.fit_on_layout = true;
                            this.parse_svg(window, cx);
                            cx.notify();
                        }
                    } else {
                        this.set_parsed_svg(None, window, cx);
                    }
                }
            },
        )
    }

    fn parse_svg(&mut self, window: &Window, cx: &mut Context<Self>) {
        let Some(buffer) = self.buffer.as_ref() else {
            return;
        };

        let renderer = cx.svg_renderer();
        let content = buffer.read(cx).snapshot();
        let background_task =
            cx.background_spawn(async move { renderer.parse_svg(content.text().as_bytes()) });

        self._parse = cx.spawn_in(window, async move |this, cx| {
            let result = background_task.await;

            this.update_in(cx, |view, window, cx| {
                let parsed_svg = result
                    .map(Arc::new)
                    .map_err(|error| error.to_string().into());
                view.set_parsed_svg(Some(parsed_svg), window, cx);
            })
            .ok();
        });
    }

    fn set_parsed_svg(
        &mut self,
        parsed_svg: Option<Result<Arc<ParsedSvg>, SharedString>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.parsed_svg = parsed_svg;
        // The previous rasterization stays visible until the new content is rasterized, unless
        // there is nothing left to show.
        self.pending_raster_width = None;
        self._rasterize = Task::ready(());
        if !matches!(self.parsed_svg, Some(Ok(_))) {
            self.set_rasterized_svg(None, window);
        }
        cx.notify();
    }

    fn set_rasterized_svg(&mut self, rasterized_svg: Option<RasterizedSvg>, window: &mut Window) {
        if let Some(previous) = std::mem::replace(&mut self.rasterized_svg, rasterized_svg) {
            window.drop_image(previous.image).ok();
        }
    }

    /// Rasterizes the parsed SVG at `width` device pixels, unless it already is or will be.
    fn request_rasterization(&mut self, width: u32, window: &mut Window, cx: &mut Context<Self>) {
        let Some(Ok(parsed_svg)) = self.parsed_svg.clone() else {
            return;
        };
        let shows_current_content = self
            .rasterized_svg
            .as_ref()
            .is_some_and(|rasterized| Arc::ptr_eq(&rasterized.source, &parsed_svg));
        if self.pending_raster_width == Some(width)
            || (self.pending_raster_width.is_none()
                && shows_current_content
                && self
                    .rasterized_svg
                    .as_ref()
                    .is_some_and(|rasterized| rasterized.width == width))
        {
            return;
        }

        // The first rasterization of new content shows up right away; zoom changes wait until
        // zooming pauses.
        let debounce = shows_current_content;
        let source = parsed_svg.clone();
        let renderer = cx.svg_renderer();
        self.pending_raster_width = Some(width);
        self._rasterize = cx.spawn_in(window, async move |this, cx| {
            if debounce {
                cx.background_executor().timer(RASTERIZE_DEBOUNCE).await;
            }
            let result = cx
                .background_spawn(async move {
                    renderer.render_parsed(
                        &parsed_svg,
                        SvgSize::Size(size(gpui::DevicePixels(width as i32), Default::default())),
                    )
                })
                .await;

            this.update_in(cx, |view, window, cx| {
                view.pending_raster_width = None;
                match result {
                    Ok(image) => view.set_rasterized_svg(
                        Some(RasterizedSvg {
                            image,
                            width,
                            source,
                        }),
                        window,
                    ),
                    Err(error) => {
                        log::error!("failed to rasterize SVG: {error}");
                        view.parsed_svg = Some(Err(error.to_string().into()));
                        view.set_rasterized_svg(None, window);
                    }
                }
                cx.notify();
            })
            .ok();
        });
    }

    fn intrinsic_size(&self) -> Option<Size<f32>> {
        match &self.parsed_svg {
            Some(Ok(parsed_svg)) => {
                Some(parsed_svg.size()).filter(|size| size.width > 0.0 && size.height > 0.0)
            }
            _ => None,
        }
    }

    pub fn open_for_project_path(
        project_path: ProjectPath,
        workspace: &mut Workspace,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        let open_buffer = workspace
            .project()
            .update(cx, |project, cx| project.open_buffer(project_path, cx));

        cx.spawn_in(window, async move |workspace, cx| {
            let Some(buffer) = open_buffer
                .await
                .notify_workspace_async_err(workspace.clone(), cx)
            else {
                return;
            };

            workspace
                .update_in(cx, |workspace, window, cx| {
                    let buffer = cx.new(|cx| MultiBuffer::singleton(buffer, cx));
                    let pane = workspace.active_pane().clone();
                    Self::open_preview_in_pane(workspace, buffer, pane, window, cx);
                })
                .ok();
        })
        .detach();
    }

    fn find_existing_preview_item_idx(
        pane: &Pane,
        buffer: &Entity<MultiBuffer>,
        cx: &App,
    ) -> Option<usize> {
        let buffer_id = buffer.read(cx).as_singleton()?.entity_id();
        pane.items_of_type::<SvgPreviewView>()
            .find(|view| {
                view.read(cx)
                    .buffer
                    .as_ref()
                    .is_some_and(|buffer| buffer.entity_id() == buffer_id)
            })
            .and_then(|view| pane.index_for_item(&view))
    }

    pub fn resolve_active_item_as_svg_buffer(
        workspace: &Workspace,
        cx: &mut Context<Workspace>,
    ) -> Option<Entity<MultiBuffer>> {
        workspace
            .active_item(cx)?
            .act_as::<MultiBuffer>(cx)
            .filter(|buffer| Self::is_svg_file(&buffer, cx))
    }

    fn create_svg_view(
        mode: SvgPreviewMode,
        workspace: &mut Workspace,
        buffer: Entity<MultiBuffer>,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) -> Entity<SvgPreviewView> {
        let workspace_handle = workspace.weak_handle();
        SvgPreviewView::new(mode, buffer, workspace_handle, window, cx)
    }

    fn create_buffer_subscription(
        buffer: &Entity<Buffer>,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> Subscription {
        cx.subscribe_in(
            buffer,
            window,
            move |this, _buffer, event: &BufferEvent, window, cx| match event {
                BufferEvent::Edited { .. } | BufferEvent::Saved | BufferEvent::Reloaded => {
                    this.parse_svg(window, cx);
                }
                _ => {}
            },
        )
    }

    pub fn is_svg_path(path: &Path) -> bool {
        path.extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("svg"))
    }

    pub fn is_svg_file(buffer: &Entity<MultiBuffer>, cx: &App) -> bool {
        buffer
            .read(cx)
            .as_singleton()
            .and_then(|buffer| buffer.read(cx).file())
            .is_some_and(|file| Self::is_svg_path(Path::new(file.file_name(cx))))
    }

    pub fn open_preview_in_pane(
        workspace: &mut Workspace,
        buffer: Entity<MultiBuffer>,
        pane: Entity<Pane>,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        Self::activate_or_add_preview(workspace, buffer, pane, true, window, cx);
    }

    pub fn open_preview_to_the_side_of_pane(
        workspace: &mut Workspace,
        buffer: Entity<MultiBuffer>,
        origin_pane: Entity<Pane>,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        let target_pane = workspace.adjacent_pane_of(&origin_pane, window, cx);
        Self::activate_or_add_preview(workspace, buffer, target_pane, false, window, cx);
    }

    fn activate_or_add_preview(
        workspace: &mut Workspace,
        buffer: Entity<MultiBuffer>,
        pane: Entity<Pane>,
        focus: bool,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        let existing_view_idx = Self::find_existing_preview_item_idx(pane.read(cx), &buffer, cx);
        if let Some(existing_view_idx) = existing_view_idx {
            pane.update(cx, |pane, cx| {
                pane.activate_item(existing_view_idx, focus, focus, window, cx);
            });
        } else {
            let view =
                Self::create_svg_view(SvgPreviewMode::Default, workspace, buffer, window, cx);
            pane.update(cx, |pane, cx| {
                pane.add_item(Box::new(view), focus, focus, None, window, cx)
            });
        }
        cx.notify();
    }

    /// Opens the SVG at `project_path` in a text editor in the active pane.
    pub fn open_as_text_for_project_path(
        project_path: ProjectPath,
        workspace: &mut Workspace,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        let open_buffer = workspace
            .project()
            .update(cx, |project, cx| project.open_buffer(project_path, cx));

        cx.spawn_in(window, async move |workspace, cx| {
            let Some(buffer) = open_buffer
                .await
                .notify_workspace_async_err(workspace.clone(), cx)
            else {
                return;
            };

            workspace
                .update_in(cx, |workspace, window, cx| {
                    let pane = workspace.active_pane().clone();
                    Self::open_buffer_as_text(workspace, buffer, pane, window, cx);
                })
                .ok();
        })
        .detach();
    }

    /// Opens the source of the SVG shown in `view` in a text editor, replacing `view`.
    fn open_as_text(
        workspace: &mut Workspace,
        view: Entity<SvgPreviewView>,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        let Some(buffer) = view.read(cx).buffer.clone() else {
            return;
        };
        let pane = workspace
            .pane_for(&view)
            .unwrap_or_else(|| workspace.active_pane().clone());
        Self::open_buffer_as_text(workspace, buffer, pane, window, cx);
    }

    /// Opens `buffer` in a text editor in `pane`.
    ///
    /// Activates an existing editor of the buffer instead, if there is one. A file preview of the
    /// buffer in `pane` is replaced, as panes keep one item per file.
    fn open_buffer_as_text(
        workspace: &mut Workspace,
        buffer: Entity<Buffer>,
        pane: Entity<Pane>,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        for pane in workspace.panes().to_vec() {
            let existing_editor = pane.read(cx).items_of_type::<Editor>().find(|editor| {
                editor.read(cx).buffer().read(cx).as_singleton().as_ref() == Some(&buffer)
            });
            if let Some(editor) = existing_editor {
                pane.update(cx, |pane, cx| {
                    if let Some(index) = pane.index_for_item(&editor) {
                        pane.activate_item(index, true, true, window, cx);
                    }
                });
                return;
            }
        }

        let project = workspace.project().clone();
        let editor = cx.new(|cx| Editor::for_buffer(buffer.clone(), Some(project), window, cx));
        pane.update(cx, |pane, cx| {
            let file_preview = pane.items_of_type::<SvgPreviewView>().find(|view| {
                let view = view.read(cx);
                view.svg_file.is_some() && view.buffer.as_ref() == Some(&buffer)
            });
            let index = file_preview
                .as_ref()
                .and_then(|view| pane.index_for_item(view));
            if let Some(file_preview) = file_preview {
                pane.remove_item(file_preview.entity_id(), false, false, window, cx);
            }
            pane.add_item(Box::new(editor), true, true, index, window, cx);
        });
    }

    pub fn register(workspace: &mut Workspace, _window: &mut Window, _cx: &mut Context<Workspace>) {
        workspace.register_action(move |workspace, _: &OpenPreview, window, cx| {
            if let Some(item) = workspace.item_for_action(window, cx)
                && let Some(buffer) = item.act_as::<MultiBuffer>(cx)
                && Self::is_svg_file(&buffer, cx)
                && let Some(pane) = workspace.pane_for_item_id(item.item_id())
            {
                Self::open_preview_in_pane(workspace, buffer, pane, window, cx);
            }
        });

        workspace.register_action(move |workspace, _: &OpenPreviewToTheSide, window, cx| {
            if let Some(buffer) = Self::resolve_active_item_as_svg_buffer(workspace, cx) {
                let pane = workspace.active_pane().clone();
                Self::open_preview_to_the_side_of_pane(workspace, buffer, pane, window, cx);
            }
        });

        workspace.register_action(move |workspace, _: &OpenFollowingPreview, window, cx| {
            if let Some(editor) = Self::resolve_active_item_as_svg_buffer(workspace, cx)
                && Self::is_svg_file(&editor, cx)
            {
                let view =
                    Self::create_svg_view(SvgPreviewMode::Follow, workspace, editor, window, cx);
                workspace.active_pane().update(cx, |pane, cx| {
                    pane.add_item(Box::new(view), true, true, None, window, cx)
                });
                cx.notify();
            }
        });

        workspace.register_action(move |workspace, _: &OpenAsText, window, cx| {
            if let Some(view) = workspace
                .active_item(cx)
                .and_then(|item| item.downcast::<SvgPreviewView>())
            {
                Self::open_as_text(workspace, view, window, cx);
            }
        });
    }

    fn zoom_in(&mut self, _: &ZoomIn, _window: &mut Window, cx: &mut Context<Self>) {
        self.set_zoom(self.zoom_level * ZOOM_STEP, None, cx);
    }

    fn zoom_out(&mut self, _: &ZoomOut, _window: &mut Window, cx: &mut Context<Self>) {
        self.set_zoom(self.zoom_level / ZOOM_STEP, None, cx);
    }

    fn reset_zoom(&mut self, _: &ResetZoom, _window: &mut Window, cx: &mut Context<Self>) {
        self.zoom_level = 1.0;
        self.pan_offset = Point::default();
        cx.notify();
    }

    fn zoom_to_actual_size(
        &mut self,
        _: &ZoomToActualSize,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.zoom_level = 1.0;
        self.pan_offset = Point::default();
        cx.notify();
    }

    fn fit_to_view(&mut self, _: &FitToView, _window: &mut Window, cx: &mut Context<Self>) {
        if let Some((bounds, svg_size)) = self.container_bounds.zip(self.intrinsic_size()) {
            self.zoom_level = compute_fit_to_view_zoom(bounds, svg_size);
            self.pan_offset = Point::default();
            cx.notify();
        }
    }

    fn set_zoom(
        &mut self,
        new_zoom: f32,
        zoom_center: Option<Point<Pixels>>,
        cx: &mut Context<Self>,
    ) {
        let old_zoom = self.zoom_level;
        self.zoom_level = new_zoom.clamp(MIN_ZOOM, MAX_ZOOM);

        if let Some((center, bounds)) = zoom_center.zip(self.container_bounds) {
            let relative_center = point(
                center.x - bounds.origin.x - bounds.size.width / 2.0,
                center.y - bounds.origin.y - bounds.size.height / 2.0,
            );
            let mouse_offset_from_svg = relative_center - self.pan_offset;
            let zoom_ratio = self.zoom_level / old_zoom;
            self.pan_offset += mouse_offset_from_svg * (1.0 - zoom_ratio);
        }

        cx.notify();
    }

    fn is_dragging(&self) -> bool {
        self.last_mouse_position.is_some()
    }

    fn handle_scroll_wheel(
        &mut self,
        event: &ScrollEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if event.modifiers.control || event.modifiers.platform {
            let delta: f32 = match event.delta {
                ScrollDelta::Pixels(pixels) => pixels.y.into(),
                ScrollDelta::Lines(lines) => lines.y * SCROLL_LINE_MULTIPLIER,
            };
            let zoom_factor = if delta > 0.0 {
                1.0 + delta.abs() * 0.01
            } else {
                1.0 / (1.0 + delta.abs() * 0.01)
            };
            self.set_zoom(self.zoom_level * zoom_factor, Some(event.position), cx);
        } else {
            let delta = match event.delta {
                ScrollDelta::Pixels(pixels) => pixels,
                ScrollDelta::Lines(lines) => lines.map(|d| px(d * SCROLL_LINE_MULTIPLIER)),
            };
            self.pan_offset += delta;
            cx.notify();
        }
    }

    fn handle_mouse_down(
        &mut self,
        event: &MouseDownEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if event.button == MouseButton::Left || event.button == MouseButton::Middle {
            self.last_mouse_position = Some(event.position);
            cx.notify();
        }
    }

    fn handle_mouse_up(
        &mut self,
        _event: &MouseUpEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.last_mouse_position = None;
        cx.notify();
    }

    fn handle_mouse_move(
        &mut self,
        event: &MouseMoveEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(last_position) = self.last_mouse_position {
            self.pan_offset += event.position - last_position;
            self.last_mouse_position = Some(event.position);
            cx.notify();
        }
    }

    fn handle_pinch(&mut self, event: &PinchEvent, _window: &mut Window, cx: &mut Context<Self>) {
        let zoom_factor = 1.0 + event.delta;
        self.set_zoom(self.zoom_level * zoom_factor, Some(event.position), cx);
    }
}

fn compute_fit_to_view_zoom(container_bounds: Bounds<Pixels>, svg_size: Size<f32>) -> f32 {
    let container_width: f32 = container_bounds.size.width.into();
    let container_height: f32 = container_bounds.size.height.into();
    let scale_x = container_width / svg_size.width;
    let scale_y = container_height / svg_size.height;
    scale_x.min(scale_y).clamp(MIN_ZOOM, 1.0)
}

/// Returns the width in device pixels to rasterize an SVG of `svg_size` at, for display at
/// `scale` device pixels per SVG pixel.
///
/// Rasterizing at the displayed resolution keeps the SVG sharp at any zoom, but the larger side
/// is capped at [`MAX_RASTER_SIZE`], beyond which the SVG is scaled up instead.
fn raster_width(svg_size: Size<f32>, scale: f32) -> u32 {
    let scale = scale
        .min(MAX_RASTER_SIZE / svg_size.width)
        .min(MAX_RASTER_SIZE / svg_size.height);
    (svg_size.width * scale).round().max(1.0) as u32
}

struct SvgContentElement {
    view: Entity<SvgPreviewView>,
}

impl IntoElement for SvgContentElement {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for SvgContentElement {
    type RequestLayoutState = ();
    type PrepaintState = Option<(AnyElement, bool)>;

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        (
            window.request_layout(
                Style {
                    size: size(relative(1.).into(), relative(1.).into()),
                    ..Default::default()
                },
                [],
                cx,
            ),
            (),
        )
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        let svg_size = self.view.read(cx).intrinsic_size()?;

        let (zoom_level, pan_offset, is_dragging, image) = self.view.update(cx, |view, cx| {
            view.container_bounds = Some(bounds);
            if std::mem::take(&mut view.fit_on_layout) {
                view.zoom_level = compute_fit_to_view_zoom(bounds, svg_size);
                view.pan_offset = Point::default();
            }
            let width = raster_width(svg_size, view.zoom_level * window.scale_factor());
            view.request_rasterization(width, window, cx);
            (
                view.zoom_level,
                view.pan_offset,
                view.is_dragging(),
                view.rasterized_svg
                    .as_ref()
                    .map(|rasterized| rasterized.image.clone()),
            )
        });

        let scaled_width = px(svg_size.width * zoom_level);
        let scaled_height = px(svg_size.height * zoom_level);
        let left = bounds.size.width / 2.0 - scaled_width / 2.0 + pan_offset.x;
        let top = bounds.size.height / 2.0 - scaled_height / 2.0 + pan_offset.y;

        let mut content = div()
            .relative()
            .size_full()
            .child(
                div()
                    .absolute()
                    .left(left)
                    .top(top)
                    .w(scaled_width)
                    .h(scaled_height)
                    .child(
                        div()
                            .size_full()
                            .absolute()
                            .top_0()
                            .left_0()
                            .child(div().size_full().bg(checkerboard(
                                cx.theme().colors().panel_background,
                                BASE_SQUARE_SIZE * zoom_level,
                            )))
                            .border_1()
                            .border_color(cx.theme().colors().border),
                    )
                    .when_some(image, |this, image| {
                        this.child(
                            img(image)
                                .id(("svg-preview-image", self.view.entity_id()))
                                .size_full(),
                        )
                    }),
            )
            .into_any_element();

        content.prepaint_as_root(bounds.origin, bounds.size.into(), window, cx);
        Some((content, is_dragging))
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let Some((mut element, is_dragging)) = prepaint.take() else {
            return;
        };

        // A drag can end outside of the view, which then never receives the mouse up.
        if is_dragging {
            let view = self.view.downgrade();
            window.on_mouse_event(move |_event: &MouseUpEvent, phase, _window, cx| {
                if phase == DispatchPhase::Bubble
                    && let Some(view) = view.upgrade()
                {
                    view.update(cx, |this, cx| {
                        this.last_mouse_position = None;
                        cx.notify();
                    });
                }
            });
        }

        element.paint(window, cx);
    }
}

impl Render for SvgPreviewView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .id("SvgPreview")
            .key_context("SvgPreview")
            .track_focus(&self.focus_handle(cx))
            .on_action(cx.listener(Self::zoom_in))
            .on_action(cx.listener(Self::zoom_out))
            .on_action(cx.listener(Self::reset_zoom))
            .on_action(cx.listener(Self::fit_to_view))
            .on_action(cx.listener(Self::zoom_to_actual_size))
            .size_full()
            .bg(cx.theme().colors().editor_background)
            .map(|this| match &self.parsed_svg {
                Some(Ok(_)) => this.child(
                    div()
                        .id("svg-container")
                        .size_full()
                        .overflow_hidden()
                        .cursor(if self.is_dragging() {
                            gpui::CursorStyle::ClosedHand
                        } else {
                            gpui::CursorStyle::OpenHand
                        })
                        .on_scroll(cx.listener(Self::handle_scroll_wheel))
                        .on_pinch(cx.listener(Self::handle_pinch))
                        .on_mouse_down(MouseButton::Left, cx.listener(Self::handle_mouse_down))
                        .on_mouse_down(MouseButton::Middle, cx.listener(Self::handle_mouse_down))
                        .on_mouse_up(MouseButton::Left, cx.listener(Self::handle_mouse_up))
                        .on_mouse_up(MouseButton::Middle, cx.listener(Self::handle_mouse_up))
                        .on_mouse_move(cx.listener(Self::handle_mouse_move))
                        .child(SvgContentElement { view: cx.entity() }),
                ),
                Some(Err(error)) => this
                    .justify_center()
                    .items_center()
                    .child(div().p_4().child(error.clone())),
                None if self.buffer.is_none() => this
                    .justify_center()
                    .items_center()
                    .child(div().p_4().child("No SVG file selected")),
                None => this,
            })
    }
}

impl Focusable for SvgPreviewView {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl EventEmitter<()> for SvgPreviewView {}

impl Item for SvgPreviewView {
    type Event = ();

    fn for_each_project_item(
        &self,
        cx: &App,
        f: &mut dyn FnMut(gpui::EntityId, &dyn project::ProjectItem),
    ) {
        if let Some(svg_file) = &self.svg_file {
            f(svg_file.entity_id(), svg_file.read(cx))
        }
    }

    fn tab_icon(&self, _window: &Window, cx: &App) -> Option<Icon> {
        self.buffer
            .as_ref()
            .and_then(|buffer| buffer.read(cx).file())
            .and_then(|file| FileIcons::get_icon(file.path().as_std_path(), cx))
            .map(Icon::from_path)
            .or_else(|| Some(Icon::new(IconName::Image)))
    }

    fn tab_content_text(&self, _detail: usize, cx: &App) -> SharedString {
        let file_name = self
            .buffer
            .as_ref()
            .and_then(|buffer| buffer.read(cx).file())
            .map(|file| file.file_name(cx).to_string());
        match file_name {
            Some(file_name) if self.svg_file.is_some() => file_name.into(),
            Some(file_name) => format!("Preview {file_name}").into(),
            None => "SVG Preview".into(),
        }
    }

    fn tab_tooltip_text(&self, cx: &App) -> Option<SharedString> {
        self.svg_file.as_ref()?;
        let file = self.buffer.as_ref()?.read(cx).file()?;
        Some(file.full_path(cx).to_string_lossy().into_owned().into())
    }

    fn telemetry_event_text(&self) -> Option<&'static str> {
        Some("svg preview: open")
    }

    fn to_item_events(_event: &Self::Event, _f: &mut dyn FnMut(workspace::item::ItemEvent)) {}

    fn buffer_kind(&self, _: &App) -> workspace::item::ItemBufferKind {
        if self.svg_file.is_some() {
            workspace::item::ItemBufferKind::Singleton
        } else {
            workspace::item::ItemBufferKind::None
        }
    }
}

impl ProjectItem for SvgPreviewView {
    type Item = SvgFile;

    fn for_project_item(
        _project: Entity<Project>,
        _pane: Option<&Pane>,
        item: Entity<Self::Item>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let buffer = item.read(cx).buffer.clone();
        Self::with_buffer(Some(buffer), Some(item), window, cx)
    }

    fn for_broken_project_item(
        abs_path: &Path,
        is_local: bool,
        e: &anyhow::Error,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<InvalidItemView> {
        Some(InvalidItemView::new(abs_path, is_local, e, window, cx))
    }
}

impl SerializableItem for SvgPreviewView {
    fn serialized_item_kind() -> &'static str {
        "SvgPreviewView"
    }

    fn deserialize(
        project: Entity<Project>,
        _workspace: WeakEntity<Workspace>,
        workspace_id: WorkspaceId,
        item_id: ItemId,
        window: &mut Window,
        cx: &mut App,
    ) -> Task<anyhow::Result<Entity<Self>>> {
        let db = SvgPreviewDb::global(cx);
        window.spawn(cx, async move |cx| {
            let svg_path = db
                .get_svg_path(item_id, workspace_id)?
                .context("No SVG path found")?;

            let (worktree, relative_path) = project
                .update(cx, |project, cx| {
                    project.find_or_create_worktree(svg_path, false, cx)
                })
                .await
                .context("Path not found")?;
            let worktree_id = worktree.update(cx, |worktree, _cx| worktree.id());
            let project_path = ProjectPath {
                worktree_id,
                path: relative_path,
            };

            let buffer = project
                .update(cx, |project, cx| project.open_buffer(project_path, cx))
                .await?;

            cx.update(|window, cx| {
                let svg_file = cx.new(|_| SvgFile {
                    buffer: buffer.clone(),
                });
                cx.new(|cx| Self::with_buffer(Some(buffer), Some(svg_file), window, cx))
            })
        })
    }

    fn cleanup(
        workspace_id: WorkspaceId,
        alive_items: Vec<ItemId>,
        _window: &mut Window,
        cx: &mut App,
    ) -> Task<anyhow::Result<()>> {
        let db = SvgPreviewDb::global(cx);
        delete_unloaded_items(alive_items, workspace_id, "svg_previews", &db, cx)
    }

    fn serialize(
        &mut self,
        workspace: &mut Workspace,
        item_id: ItemId,
        _closing: bool,
        cx: &mut Context<Self>,
    ) -> Option<Task<anyhow::Result<()>>> {
        // Previews of an editor follow that editor, so only previews of files are restored.
        self.svg_file.as_ref()?;
        let workspace_id = workspace.database_id()?;
        let svg_path = self
            .buffer
            .as_ref()?
            .read(cx)
            .file()?
            .as_local()?
            .abs_path(cx);

        let db = SvgPreviewDb::global(cx);
        Some(cx.background_spawn(
            async move { db.save_svg_path(item_id, workspace_id, svg_path).await },
        ))
    }

    fn should_serialize(&self, _event: &Self::Event) -> bool {
        false
    }
}

/// Zoom controls and the text editor toggle for the active SVG preview.
pub struct SvgPreviewToolbarControls {
    view: Option<WeakEntity<SvgPreviewView>>,
    _subscription: Option<Subscription>,
}

impl SvgPreviewToolbarControls {
    pub fn new() -> Self {
        Self {
            view: None,
            _subscription: None,
        }
    }
}

impl Render for SvgPreviewToolbarControls {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let Some(view) = self.view.as_ref().and_then(|view| view.upgrade()) else {
            return div().into_any_element();
        };
        let zoom_percentage = format!("{}%", (view.read(cx).zoom_level * 100.0).round() as i32);
        let focus_handle = view.focus_handle(cx);

        h_flex()
            .gap_1()
            .child(
                IconButton::new("zoom-out", IconName::Dash)
                    .icon_size(IconSize::Small)
                    .tooltip(|_window, cx| Tooltip::for_action("Zoom Out", &ZoomOut, cx))
                    .on_click({
                        let focus_handle = focus_handle.clone();
                        move |_, window, cx| {
                            focus_handle.dispatch_action(&ZoomOut, window, cx);
                        }
                    }),
            )
            .child(
                h_flex()
                    .id("zoom-label")
                    .px_1()
                    .cursor_pointer()
                    .child(Label::new(zoom_percentage).size(LabelSize::Small))
                    .tooltip(|_window, cx| Tooltip::for_action("Reset Zoom", &ResetZoom, cx))
                    .on_click({
                        let focus_handle = focus_handle.clone();
                        move |_, window, cx| {
                            focus_handle.dispatch_action(&ResetZoom, window, cx);
                        }
                    }),
            )
            .child(
                IconButton::new("zoom-in", IconName::Plus)
                    .icon_size(IconSize::Small)
                    .tooltip(|_window, cx| Tooltip::for_action("Zoom In", &ZoomIn, cx))
                    .on_click({
                        let focus_handle = focus_handle.clone();
                        move |_, window, cx| {
                            focus_handle.dispatch_action(&ZoomIn, window, cx);
                        }
                    }),
            )
            .child(
                IconButton::new("fit-to-view", IconName::Maximize)
                    .icon_size(IconSize::Small)
                    .tooltip(|_window, cx| Tooltip::for_action("Fit to View", &FitToView, cx))
                    .on_click({
                        let focus_handle = focus_handle.clone();
                        move |_, window, cx| {
                            focus_handle.dispatch_action(&FitToView, window, cx);
                        }
                    }),
            )
            .child(
                IconButton::new("open-as-text", IconName::FileCode)
                    .icon_size(IconSize::Small)
                    .tooltip(|_window, cx| Tooltip::for_action("Open as Text", &OpenAsText, cx))
                    .on_click(move |_, window, cx| {
                        focus_handle.dispatch_action(&OpenAsText, window, cx);
                    }),
            )
            .into_any_element()
    }
}

impl EventEmitter<ToolbarItemEvent> for SvgPreviewToolbarControls {}

impl ToolbarItemView for SvgPreviewToolbarControls {
    fn set_active_pane_item(
        &mut self,
        active_pane_item: Option<&dyn ItemHandle>,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> ToolbarItemLocation {
        self.view = None;
        self._subscription = None;

        if let Some(view) = active_pane_item.and_then(|item| item.downcast::<SvgPreviewView>()) {
            self._subscription = Some(cx.observe(&view, |_, _, cx| cx.notify()));
            self.view = Some(view.downgrade());
            cx.notify();
            return ToolbarItemLocation::PrimaryRight;
        }

        ToolbarItemLocation::Hidden
    }
}

mod persistence {
    use std::path::PathBuf;

    use db::{
        query,
        sqlez::{domain::Domain, thread_safe_connection::ThreadSafeConnection},
        sqlez_macros::sql,
    };
    use workspace::{ItemId, WorkspaceDb, WorkspaceId};

    pub struct SvgPreviewDb(ThreadSafeConnection);

    impl Domain for SvgPreviewDb {
        const NAME: &str = stringify!(SvgPreviewDb);

        const MIGRATIONS: &[&str] = &[sql!(
                CREATE TABLE svg_previews (
                    workspace_id INTEGER,
                    item_id INTEGER UNIQUE,

                    svg_path BLOB,

                    PRIMARY KEY(workspace_id, item_id),
                    FOREIGN KEY(workspace_id) REFERENCES workspaces(workspace_id)
                    ON DELETE CASCADE
                ) STRICT;
        )];
    }

    db::static_connection!(SvgPreviewDb, [WorkspaceDb]);

    impl SvgPreviewDb {
        query! {
            pub async fn save_svg_path(
                item_id: ItemId,
                workspace_id: WorkspaceId,
                svg_path: PathBuf
            ) -> Result<()> {
                INSERT OR REPLACE INTO svg_previews(item_id, workspace_id, svg_path)
                VALUES (?, ?, ?)
            }
        }

        query! {
            pub fn get_svg_path(item_id: ItemId, workspace_id: WorkspaceId) -> Result<Option<PathBuf>> {
                SELECT svg_path
                FROM svg_previews
                WHERE item_id = ? AND workspace_id = ?
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{TestAppContext, VisualTestContext};
    use project::{FakeFs, Fs as _};
    use util::path;
    use workspace::MultiWorkspace;

    const SQUARE_SVG: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20"><rect width="40" height="20" fill="red"/></svg>"#;

    #[test]
    fn raster_width_caps_the_larger_side() {
        let tall = Size::new(100.0, 1000.0);
        let width = raster_width(tall, 20.0);
        // The height would be 20000 device pixels uncapped, so the width shrinks with it.
        assert_eq!(width, (MAX_RASTER_SIZE / 10.0).round() as u32);
        assert_eq!(raster_width(tall, 2.0), 200);
    }

    fn init_test(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::init(cx);
        });
    }

    #[gpui::test]
    async fn svg_files_open_in_the_preview_and_open_as_text_shows_their_source(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/root"),
            serde_json::json!({ "image.svg": SQUARE_SVG }),
        )
        .await;
        let project = Project::test(fs, [path!("/root").as_ref()], cx).await;
        let window =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = window
            .read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone())
            .expect("workspace");
        let cx = &mut VisualTestContext::from_window(window.into(), cx);

        let item = workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_abs_path(
                    path!("/root/image.svg").into(),
                    workspace::OpenOptions::default(),
                    window,
                    cx,
                )
            })
            .await
            .expect("open svg");
        let view = item
            .downcast::<SvgPreviewView>()
            .expect("svg files open in the preview");
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert_eq!(view.intrinsic_size(), Some(Size::new(40.0, 20.0)));
        });

        workspace.update_in(cx, |workspace, window, cx| {
            assert_eq!(editor_count(workspace, cx), 0);
            SvgPreviewView::open_as_text(workspace, view.clone(), window, cx);
            assert_eq!(editor_count(workspace, cx), 1);
            let preview_count = workspace
                .panes()
                .iter()
                .map(|pane| pane.read(cx).items_of_type::<SvgPreviewView>().count())
                .sum::<usize>();
            assert_eq!(preview_count, 0, "the editor replaces the preview");
            // A second request activates the editor of the first one.
            SvgPreviewView::open_as_text(workspace, view.clone(), window, cx);
            assert_eq!(editor_count(workspace, cx), 1);
            let editor = workspace
                .active_item(cx)
                .and_then(|item| item.downcast::<Editor>())
                .expect("editor is active");
            assert_eq!(editor.read(cx).text(cx), SQUARE_SVG);
        });
    }

    #[gpui::test]
    async fn zooming_rasterizes_the_svg_at_the_displayed_resolution(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/root"),
            serde_json::json!({ "image.svg": SQUARE_SVG }),
        )
        .await;
        let project = Project::test(fs, [path!("/root").as_ref()], cx).await;
        let window =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = window
            .read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone())
            .expect("workspace");
        let cx = &mut VisualTestContext::from_window(window.into(), cx);
        let view = workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_abs_path(
                    path!("/root/image.svg").into(),
                    workspace::OpenOptions::default(),
                    window,
                    cx,
                )
            })
            .await
            .expect("open svg")
            .downcast::<SvgPreviewView>()
            .expect("svg files open in the preview");
        cx.run_until_parked();

        let raster_width_after_drawing = |cx: &mut VisualTestContext| {
            cx.update(|window, cx| {
                window.refresh();
                window.draw(cx).clear(cx);
            });
            cx.executor().advance_clock(RASTERIZE_DEBOUNCE);
            cx.run_until_parked();
            let scale_factor = cx.update(|window, _| window.scale_factor());
            view.read_with(cx, |view, _| {
                let rasterized_width = view.rasterized_svg.as_ref().map(|svg| svg.width);
                let expected_width =
                    raster_width(Size::new(40.0, 20.0), view.zoom_level * scale_factor);
                (rasterized_width, expected_width)
            })
        };

        let (rasterized_width, expected_width) = raster_width_after_drawing(cx);
        assert_eq!(rasterized_width, Some(expected_width));

        view.update_in(cx, |view, window, cx| {
            view.set_zoom(8.0, None, cx);
            view.zoom_in(&ZoomIn, window, cx);
        });
        let (rasterized_width, expected_width) = raster_width_after_drawing(cx);
        assert!(expected_width > 300, "zoomed far beyond the intrinsic 40px");
        assert_eq!(rasterized_width, Some(expected_width));
    }

    #[gpui::test]
    async fn content_rewritten_on_disk_at_the_same_size_is_rasterized_again(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/root"),
            serde_json::json!({ "image.svg": SQUARE_SVG }),
        )
        .await;
        let project = Project::test(fs.clone(), [path!("/root").as_ref()], cx).await;
        let window =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = window
            .read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone())
            .expect("workspace");
        let cx = &mut VisualTestContext::from_window(window.into(), cx);
        let view = workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_abs_path(
                    path!("/root/image.svg").into(),
                    workspace::OpenOptions::default(),
                    window,
                    cx,
                )
            })
            .await
            .expect("open svg")
            .downcast::<SvgPreviewView>()
            .expect("svg files open in the preview");
        cx.run_until_parked();

        let rasterized_image_after_drawing = |cx: &mut VisualTestContext| {
            cx.update(|window, cx| {
                window.refresh();
                window.draw(cx).clear(cx);
            });
            cx.executor().advance_clock(RASTERIZE_DEBOUNCE);
            cx.run_until_parked();
            view.read_with(cx, |view, _| {
                view.rasterized_svg
                    .as_ref()
                    .map(|rasterized| rasterized.image.clone())
                    .expect("the svg is rasterized")
            })
        };
        let original_image = rasterized_image_after_drawing(cx);

        // Same size and zoom, so only the content tells the old rasterization apart.
        fs.save(
            path!("/root/image.svg").as_ref(),
            &SQUARE_SVG.replace("red", "blue").into(),
            language::LineEnding::Unix,
        )
        .await
        .expect("rewrite svg");
        cx.run_until_parked();
        let rewritten_image = rasterized_image_after_drawing(cx);

        assert!(!Arc::ptr_eq(&original_image, &rewritten_image));
        assert_ne!(original_image.as_bytes(0), rewritten_image.as_bytes(0));
    }

    #[gpui::test]
    async fn saved_svg_tabs_are_restored_as_file_previews(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/root"),
            serde_json::json!({ "image.svg": SQUARE_SVG }),
        )
        .await;
        let project = Project::test(fs, [path!("/root").as_ref()], cx).await;
        let window =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = window
            .read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone())
            .expect("workspace");
        let cx = &mut VisualTestContext::from_window(window.into(), cx);

        let workspace_id = cx
            .update(|_, cx| workspace::WorkspaceDb::global(cx))
            .next_id()
            .await
            .expect("workspace id");
        let item_id: ItemId = 1;
        cx.update(|_, cx| SvgPreviewDb::global(cx))
            .save_svg_path(item_id, workspace_id, path!("/root/image.svg").into())
            .await
            .expect("save");

        let restored = cx
            .update(|window, cx| {
                SvgPreviewView::deserialize(
                    project.clone(),
                    workspace.downgrade(),
                    workspace_id,
                    item_id,
                    window,
                    cx,
                )
            })
            .await
            .expect("restore");
        cx.run_until_parked();
        restored.read_with(cx, |restored, cx| {
            assert!(
                restored.svg_file.is_some(),
                "restored as the file, not a preview"
            );
            assert_eq!(restored.tab_content_text(0, cx), "image.svg");
            assert_eq!(restored.intrinsic_size(), Some(Size::new(40.0, 20.0)));
        });
    }

    fn editor_count(workspace: &Workspace, cx: &App) -> usize {
        workspace
            .panes()
            .iter()
            .map(|pane| pane.read(cx).items_of_type::<Editor>().count())
            .sum()
    }
}
