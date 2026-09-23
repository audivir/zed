use crate::{
    document_properties::DocumentPropertiesModal,
    pdf::Pdf,
    pdf_renderer::{
        PageData, PageRotation, PdfChapter, PdfLinkAction, PdfSearchResult, PdfTarget, PdfTextSpan,
        RenderStyle,
    },
    pdf_store::{PdfItem, PdfItemEvent, open_pdf},
};
use anyhow::{Context as _, Error, Result};
use collections::{HashMap, HashSet};
use editor::{Editor, EditorSettings};
use gpui::{
    AnyElement, App, Bounds, Context, DispatchPhase, Element, ElementId, Entity, EntityId,
    EventEmitter, FocusHandle, Focusable, Font, GlobalElementId, InspectorElementId,
    InteractiveElement, IntoElement, KeyDownEvent, LayoutId, MouseButton, ParentElement,
    PinchEvent, Pixels, Point, Render, ScrollHandle, ScrollWheelEvent, Size, Style, Styled,
    Subscription, Task, WeakEntity, Window, actions, div, img, px, size, uniform_list,
};
use language::File as _;
use persistence::PdfViewerDb;
use project::{Project, ProjectPath, search::SearchQuery};
use settings::{SeedQuerySetting, Settings};
use std::{
    ops::RangeInclusive,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
use theme_settings::ThemeSettings;
use ui::{Banner, FluentBuilder as _, ScrollAxes, Severity, Tooltip, WithScrollbar, prelude::*};
use util::ResultExt as _;
use util::paths::PathExt;
use workspace::{
    ItemId, Pane, ToolbarItemEvent, ToolbarItemLocation, ToolbarItemView, Workspace, WorkspaceId,
    delete_unloaded_items,
    invalid_item_view::InvalidItemView,
    item::{
        HighlightedText, Item, ItemBufferKind, ItemHandle, ProjectItem, SerializableItem,
        TabContentParams,
    },
    searchable::{
        Direction, SearchEvent, SearchOptions, SearchToken, SearchableItem, SearchableItemHandle,
    },
};
mod document_properties;
pub mod pdf;
pub mod pdf_renderer;
pub mod pdf_store;

actions!(
    pdf_viewer,
    [
        /// Zooms in the PDF.
        ZoomIn,
        /// Zooms out the PDF.
        ZoomOut,
        /// Resets the zoom to 100%.
        ResetZoom,
        /// Fits the PDF to the view.
        FitToView,
        /// Fits the PDF to the view width.
        FitToWidth,
        /// Zooms to the actual size (100%).
        ZoomToActualSize,
        /// Goes to the next page.
        NextPage,
        /// Goes to the previous page.
        PreviousPage,
        /// Goes to the first page.
        FirstPage,
        /// Goes to the last page.
        LastPage,
        /// Copies the selected text.
        Copy,
        /// Selects all text in the document.
        SelectAll,
        /// Toggles the outline sidebar.
        ToggleOutline,
        /// Toggles the thumbnail sidebar.
        ToggleThumbnails,
        /// Toggles between continuous scrolling and single-page view.
        ToggleSinglePageView,
        /// Rotates the view 90 degrees clockwise without modifying the file.
        RotatePageClockwise,
        /// Inverts the lightness of the pages, for reading in dark themes.
        ToggleInvertedColors,
        /// Opens the PDF in the default application of the system, for example to print it.
        OpenInDefaultApp,
        /// Shows the title, author, and other properties of the document.
        ShowDocumentProperties
    ]
);

const MIN_ZOOM: f32 = 0.1;
const MAX_ZOOM: f32 = 20.0;
const ZOOM_STEP: f32 = 1.1;
const PAGE_GAP: f32 = 16.0;
const DEFAULT_OFFSET: f32 = 40.0;
const PRERENDER_PAGES: usize = 3;
const THUMBNAIL_SIDEBAR_WIDTH: f32 = 140.0;
const THUMBNAIL_ROW_HEIGHT: f32 = 150.0;
/// Thumbnail rows kept rendered on each side of the visible ones, so short scrolls in the sidebar
/// do not re-render thumbnails.
const THUMBNAIL_CACHE_MARGIN: usize = 32;
const SEARCH_DEBOUNCE: Duration = Duration::from_millis(300);
/// Distance to scroll past the edge of the page in single-page mode to flip to the next one.
const PAGE_FLIP_OVERSCROLL: f32 = 80.0;
/// Pause in scroll events that ends a gesture, so trackpad momentum flips at most one page.
const PAGE_FLIP_GESTURE_GAP: Duration = Duration::from_millis(200);

#[derive(Clone)]
pub enum PdfSource {
    Item(Entity<PdfItem>),
    Memory(Arc<Pdf>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScrollMode {
    Highlight,
    Center,
    Top,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScrollAnchor {
    Top,
    Baseline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewMode {
    /// All pages laid out top to bottom with free scrolling.
    Continuous,
    /// Only `current_page` is shown, and navigation goes page by page.
    ///
    /// Scrolling stays within the current page, and scrolling past its edge flips the page.
    SinglePage,
}

/// Position in the document text as a page index and a character index within its spans.
///
/// Characters follow the PDFium extraction order, which is reading order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct TextPosition {
    page_index: usize,
    char_index: usize,
}

#[derive(Clone, Debug)]
struct TextSelection {
    anchor: TextPosition,
    head: TextPosition,
}

impl TextSelection {
    /// Returns `(start, end)` in document order, regardless of drag direction.
    fn ordered(&self) -> (TextPosition, TextPosition) {
        if self.anchor <= self.head {
            (self.anchor, self.head)
        } else {
            (self.head, self.anchor)
        }
    }
}

/// Returns the index of the character in `text_spans` closest to `point` in page space.
///
/// Returns `None` only when the page has no text.
fn nearest_char_index(text_spans: &[PdfTextSpan], point: Point<f32>) -> Option<usize> {
    text_spans
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            distance_to_bounds_center(a.bounds, point)
                .total_cmp(&distance_to_bounds_center(b.bounds, point))
        })
        .map(|(index, _)| index)
}

fn distance_to_bounds_center(bounds: Bounds<f32>, point: Point<f32>) -> f32 {
    let center = Point {
        x: bounds.origin.x + bounds.size.width / 2.0,
        y: bounds.origin.y + bounds.size.height / 2.0,
    };
    let dx = center.x - point.x;
    let dy = center.y - point.y;
    (dx * dx + dy * dy).sqrt()
}

/// Returns the half-open range of selected span indices on `page_index`.
///
/// `span_count` is the number of text spans on that page.
fn selected_char_range_for_page(
    selection: &TextSelection,
    page_index: usize,
    span_count: usize,
) -> Option<std::ops::Range<usize>> {
    let (start, end) = selection.ordered();
    if span_count == 0 || page_index < start.page_index || page_index > end.page_index {
        return None;
    }
    let range_start = if page_index == start.page_index {
        start.char_index.min(span_count)
    } else {
        0
    };
    let range_end = if page_index == end.page_index {
        (end.char_index + 1).min(span_count)
    } else {
        span_count
    };
    (range_start < range_end).then_some(range_start..range_end)
}

/// Groups consecutive characters in `range` into one rectangle per line.
///
/// Two spans share a line when their vertical bounds overlap by at least half of the shorter
/// height.
fn selection_highlight_rects(
    text_spans: &[PdfTextSpan],
    range: std::ops::Range<usize>,
) -> Vec<Bounds<f32>> {
    let mut rects: Vec<Bounds<f32>> = Vec::new();
    for span in &text_spans[range] {
        let bounds = span.bounds;
        if let Some(last) = rects.last_mut() {
            let overlap = (last.origin.y + last.size.height)
                .min(bounds.origin.y + bounds.size.height)
                - last.origin.y.max(bounds.origin.y);
            let shorter_height = last.size.height.min(bounds.size.height);
            if shorter_height > 0.0 && overlap >= shorter_height * 0.5 {
                let left = last.origin.x.min(bounds.origin.x);
                let right =
                    (last.origin.x + last.size.width).max(bounds.origin.x + bounds.size.width);
                let top = last.origin.y.min(bounds.origin.y);
                let bottom =
                    (last.origin.y + last.size.height).max(bounds.origin.y + bounds.size.height);
                last.origin.x = left;
                last.origin.y = top;
                last.size.width = right - left;
                last.size.height = bottom - top;
                continue;
            }
        }
        rects.push(bounds);
    }
    rects
}

/// Returns the selected text across the available pages in `page_text_spans`.
fn selection_to_text(
    selection: &TextSelection,
    page_text_spans: &[(usize, Arc<Vec<PdfTextSpan>>)],
) -> String {
    let mut result = String::new();
    for (page_index, spans) in page_text_spans {
        let Some(range) = selected_char_range_for_page(selection, *page_index, spans.len()) else {
            continue;
        };
        if !result.is_empty() {
            result.push('\n');
        }
        for span in &spans[range] {
            result.push_str(&span.text);
        }
    }
    result
}

/// Screen bounds and text spans of a page, captured in `prepaint` for drag hit-testing.
struct SelectionHitRegion {
    page_index: usize,
    bounds: Bounds<Pixels>,
    text_spans: Arc<Vec<PdfTextSpan>>,
    /// Unrotated page size in points, which maps rotated screen points back to page points.
    page_size: Size<f32>,
}

/// Converts a window-space drag point into a `TextPosition`.
///
/// Prefers the page region containing the point and otherwise falls back to the vertically
/// nearest page, so dragging past the document edges still extends the selection.
fn hit_test_selection_point(
    regions: &[SelectionHitRegion],
    point: Point<Pixels>,
    zoom: f32,
    rotation: PageRotation,
) -> Option<TextPosition> {
    let region = regions
        .iter()
        .find(|region| region.bounds.contains(&point))
        .or_else(|| {
            regions.iter().min_by(|a, b| {
                vertical_distance(a.bounds, point).total_cmp(&vertical_distance(b.bounds, point))
            })
        })?;
    let local = page_point_at(point, region.bounds, zoom, rotation, region.page_size);
    let char_index = nearest_char_index(&region.text_spans, local)?;
    Some(TextPosition {
        page_index: region.page_index,
        char_index,
    })
}

/// Converts a window-space point into unrotated page coordinates in points.
fn page_point_at(
    point: Point<Pixels>,
    page_bounds: Bounds<Pixels>,
    zoom: f32,
    rotation: PageRotation,
    page_size: Size<f32>,
) -> Point<f32> {
    let display = Point {
        x: f32::from(point.x - page_bounds.origin.x) / zoom,
        y: f32::from(point.y - page_bounds.origin.y) / zoom,
    };
    rotation.to_page_point(display, page_size)
}

fn vertical_distance(bounds: Bounds<Pixels>, point: Point<Pixels>) -> f32 {
    if point.y < bounds.origin.y {
        f32::from(bounds.origin.y - point.y)
    } else if point.y > bounds.origin.y + bounds.size.height {
        f32::from(point.y - (bounds.origin.y + bounds.size.height))
    } else {
        0.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CharClass {
    Word,
    Punctuation,
    Whitespace,
}

fn char_class(c: char) -> CharClass {
    if c.is_whitespace() {
        CharClass::Whitespace
    } else if c.is_alphanumeric() || c == '_' {
        CharClass::Word
    } else {
        CharClass::Punctuation
    }
}

/// Returns the character class of a single-character span; empty text counts as whitespace.
fn span_char_class(span: &PdfTextSpan) -> CharClass {
    span.text
        .chars()
        .next()
        .map(char_class)
        .unwrap_or(CharClass::Whitespace)
}

/// Returns the maximal run of same-class characters containing `char_index`.
///
/// Used for double-click word selection, where punctuation runs count as words.
fn word_range_at(text_spans: &[PdfTextSpan], char_index: usize) -> std::ops::Range<usize> {
    let Some(span) = text_spans.get(char_index) else {
        return char_index..char_index;
    };
    let class = span_char_class(span);
    let mut start = char_index;
    while start > 0 && span_char_class(&text_spans[start - 1]) == class {
        start -= 1;
    }
    let mut end = char_index;
    while end + 1 < text_spans.len() && span_char_class(&text_spans[end + 1]) == class {
        end += 1;
    }
    start..end + 1
}

/// Returns the maximal run of characters on the visual line of `char_index`.
///
/// Every candidate is compared against the clicked character directly, so the scan cannot
/// drift onto an adjacent line.
fn line_range_at(text_spans: &[PdfTextSpan], char_index: usize) -> std::ops::Range<usize> {
    let Some(anchor_bounds) = text_spans.get(char_index).map(|s| s.bounds) else {
        return char_index..char_index;
    };
    let same_line = |bounds: Bounds<f32>| {
        let overlap = (anchor_bounds.origin.y + anchor_bounds.size.height)
            .min(bounds.origin.y + bounds.size.height)
            - anchor_bounds.origin.y.max(bounds.origin.y);
        let shorter_height = anchor_bounds.size.height.min(bounds.size.height);
        shorter_height > 0.0 && overlap >= shorter_height * 0.5
    };
    let mut start = char_index;
    while start > 0 && same_line(text_spans[start - 1].bounds) {
        start -= 1;
    }
    let mut end = char_index;
    while end + 1 < text_spans.len() && same_line(text_spans[end + 1].bounds) {
        end += 1;
    }
    start..end + 1
}

/// Returns the selection from the first to the last page in `pages_with_text`.
///
/// Each entry is a `(page_index, char_count)` pair for a page with text. Returns `None` if no
/// page has text.
fn select_all_range(pages_with_text: &[(usize, usize)]) -> Option<TextSelection> {
    let mut sorted = pages_with_text.to_vec();
    sorted.sort_unstable_by_key(|(page_index, _)| *page_index);
    let (&(first_page, _), &(last_page, last_span_count)) = sorted.first().zip(sorted.last())?;
    Some(TextSelection {
        anchor: TextPosition {
            page_index: first_page,
            char_index: 0,
        },
        head: TextPosition {
            page_index: last_page,
            char_index: last_span_count.saturating_sub(1),
        },
    })
}

/// Container for explicitly expanded or collapsed outline nodes, keyed by child index path.
///
/// Nodes without an entry default to expanded at the top level and collapsed below.
#[derive(Default)]
struct OutlineExpansionState(HashSet<Vec<usize>>);

impl OutlineExpansionState {
    fn is_expanded(&self, path: &[usize]) -> bool {
        if self.0.contains(path) {
            true
        } else if self.0.contains(&Self::collapsed_marker(path)) {
            false
        } else {
            path.len() <= 1
        }
    }

    fn toggle(&mut self, path: Vec<usize>) {
        if self.is_expanded(&path) {
            self.0.remove(&path);
            self.0.insert(Self::collapsed_marker(&path));
        } else {
            self.0.remove(&Self::collapsed_marker(&path));
            self.0.insert(path);
        }
    }

    /// Returns the path with a trailing `usize::MAX`, which records an explicit collapse.
    fn collapsed_marker(path: &[usize]) -> Vec<usize> {
        let mut marker = path.to_vec();
        marker.push(usize::MAX);
        marker
    }
}

pub struct PdfView {
    source: PdfSource,
    stale_pdf: Option<Arc<Pdf>>,
    stale_page_cache: HashMap<usize, PageData>,
    last_search_query: Option<(PdfSearchKey, Vec<PdfSearchResult>)>,
    /// Error of the last failed search, shown until the next search succeeds or is cleared.
    search_error: Option<SharedString>,
    project: Option<Entity<Project>>,
    focus_handle: FocusHandle,
    scroll_handle: ScrollHandle,
    zoom_level: f32,
    container_bounds: Option<Bounds<Pixels>>,
    current_page: usize,
    current_tasks: HashMap<usize, Task<()>>,
    new_state: bool,
    search_matches: Vec<PdfSearchResult>,
    active_match_index: Option<usize>,
    search_token: Option<SearchToken>,
    pub active_jump: Option<(ScrollAnchor, PdfTarget)>,
    clear_jump_task: Option<Task<()>>,
    #[allow(clippy::type_complexity)]
    pub on_page_click: Option<Arc<dyn Fn(usize, Point<f32>, &mut Window, &mut Context<Self>)>>,
    password_input: Entity<Editor>,
    password_task: Option<Task<()>>,
    /// In-flight `select_all` extraction, stored so tests can await it.
    select_all_task: Option<Task<()>>,
    /// Current text selection.
    ///
    /// Drag selection only reaches rendered pages, while `select_all` extracts text for every
    /// page.
    text_selection: Option<TextSelection>,
    /// Anchor of a single-click mouse-down, held until the drag reaches another character.
    ///
    /// Keeps a one-character selection from flashing on a plain click.
    pending_click_anchor: Option<TextPosition>,
    is_selecting_text: bool,
    show_outline: bool,
    outline_expanded: OutlineExpansionState,
    show_thumbnails: bool,
    /// Keeps the first `prepaint` from overwriting a restored zoom with fit-to-view.
    suppress_initial_fit_to_view: bool,
    view_mode: ViewMode,
    /// View-only page rotation, not persisted.
    rotation: PageRotation,
    /// Whether pages render with inverted lightness, not persisted.
    inverted_colors: bool,
    page_flip: PageFlip,
}

/// Scrolling past the edge of the page in single-page mode.
#[derive(Default)]
struct PageFlip {
    overscroll: f32,
    last_scroll_at: Option<Instant>,
    /// Set after a flip until the gesture ends.
    flipped: bool,
}

impl PdfView {
    pub fn new(
        pdf_item: Entity<PdfItem>,
        project: Entity<Project>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        cx.subscribe(&pdf_item, Self::on_pdf_event).detach();

        Self {
            source: PdfSource::Item(pdf_item),
            stale_pdf: None,
            stale_page_cache: Default::default(),
            last_search_query: None,
            search_error: None,
            project: Some(project),
            focus_handle: cx.focus_handle(),
            scroll_handle: ScrollHandle::default(),
            zoom_level: 1.0,
            container_bounds: None,
            current_page: 0,
            current_tasks: Default::default(),
            new_state: false,
            search_matches: Vec::new(),
            active_match_index: None,
            search_token: None,
            active_jump: None,
            clear_jump_task: None,
            on_page_click: None,
            password_input: Self::new_password_input(window, cx),
            password_task: None,
            select_all_task: None,
            text_selection: None,
            pending_click_anchor: None,
            is_selecting_text: false,
            show_outline: false,
            outline_expanded: Default::default(),
            show_thumbnails: false,
            suppress_initial_fit_to_view: false,
            view_mode: ViewMode::Continuous,
            rotation: PageRotation::None,
            inverted_colors: false,
            page_flip: PageFlip::default(),
        }
    }

    fn new_password_input(window: &mut Window, cx: &mut Context<Self>) -> Entity<Editor> {
        cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_masked(true, cx);
            editor
        })
    }

    /// Returns the current PDF regardless of its source.
    /// Returns the on-screen size of a page in points, after the view rotation.
    fn display_page_size(&self, pdf: &Pdf, page_index: usize) -> Size<f32> {
        self.rotation
            .display_size(pdf.get_page_size(page_index).unwrap_or_default())
    }

    /// Returns the widest on-screen page width in points, after the view rotation.
    fn display_max_width(&self, pdf: &Pdf) -> f32 {
        if self.rotation.swaps_dimensions() {
            pdf.metadata
                .pages
                .iter()
                .map(|page| page.size.height)
                .fold(0.0, f32::max)
        } else {
            pdf.metadata.max_width
        }
    }

    /// Returns the on-screen offset of a target within its page, in points.
    fn display_target_point(&self, pdf: &Pdf, target: &PdfTarget) -> Point<f32> {
        let page_size = pdf.get_page_size(target.page_index).unwrap_or_default();
        self.rotation.to_display_point(target.point, page_size)
    }

    pub fn pdf(&self, cx: &App) -> Arc<Pdf> {
        match &self.source {
            PdfSource::Item(item) => item.read(cx).pdf.clone(),
            PdfSource::Memory(pdf) => pdf.clone(),
        }
    }

    /// Creates a view for an already loaded PDF.
    pub fn from_pdf(pdf: Arc<Pdf>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        Self {
            source: PdfSource::Memory(pdf),
            stale_pdf: None,
            stale_page_cache: Default::default(),
            last_search_query: None,
            search_error: None,
            project: None,
            focus_handle: cx.focus_handle(),
            scroll_handle: ScrollHandle::default(),
            zoom_level: 1.0,
            container_bounds: None,
            current_page: 0,
            current_tasks: Default::default(),
            new_state: false,
            search_matches: Vec::new(),
            active_match_index: None,
            search_token: None,
            active_jump: None,
            clear_jump_task: None,
            on_page_click: None,
            password_input: Self::new_password_input(window, cx),
            password_task: None,
            select_all_task: None,
            text_selection: None,
            pending_click_anchor: None,
            is_selecting_text: false,
            show_outline: false,
            outline_expanded: Default::default(),
            show_thumbnails: false,
            suppress_initial_fit_to_view: false,
            view_mode: ViewMode::Continuous,
            rotation: PageRotation::None,
            inverted_colors: false,
            page_flip: PageFlip::default(),
        }
    }

    /// Replaces the underlying PDF.
    pub fn update_pdf(&mut self, pdf: Arc<Pdf>, cx: &mut Context<Self>) {
        if let PdfSource::Memory(_) = &self.source {
            self.stale_pdf = Some(self.pdf(cx));
            self.stale_page_cache.clear();
            self.source = PdfSource::Memory(pdf);
            self.current_tasks.clear();

            let had_matches = !self.search_matches.is_empty();
            self.active_match_index = None;
            self.last_search_query = None;
            if had_matches {
                self.search_matches.clear();
                cx.emit(SearchEvent::MatchesInvalidated);
            }

            cx.notify();
        }
    }

    pub fn stale_pdf(&self, cx: &App) -> Option<Arc<Pdf>> {
        match &self.source {
            PdfSource::Item(item) => item.read(cx).stale_pdf.clone(),
            PdfSource::Memory(_) => self.stale_pdf.clone(),
        }
    }

    pub fn clear_stale_pdf(&mut self, cx: &mut Context<Self>) {
        match &self.source {
            PdfSource::Item(item) => {
                item.update(cx, |this, _| this.stale_pdf = None);
            }
            PdfSource::Memory(_) => {
                self.stale_pdf = None;
            }
        }
    }

    fn on_pdf_event(&mut self, _: Entity<PdfItem>, event: &PdfItemEvent, cx: &mut Context<Self>) {
        match event {
            PdfItemEvent::MetadataUpdated
            | PdfItemEvent::FileHandleChanged
            | PdfItemEvent::Reloaded => {
                cx.emit(PdfViewEvent::TitleChanged);
                cx.notify();
            }
            PdfItemEvent::ReloadNeeded => {}
        }
    }

    fn submit_password(&mut self, _: &menu::Confirm, window: &mut Window, cx: &mut Context<Self>) {
        let password = self.password_input.update(cx, |editor, cx| {
            let text = editor.text(cx);
            editor.clear(window, cx);
            text
        });
        if password.is_empty() {
            return;
        }

        match &self.source {
            PdfSource::Item(item) => {
                item.update(cx, |item, cx| item.submit_password(password, cx));
            }
            PdfSource::Memory(pdf) => {
                let bytes = (*pdf.bytes).clone();
                self.password_task = Some(cx.spawn(async move |this, cx| {
                    let pdf = Arc::new(Pdf::from_bytes(bytes, None, Some(password)).await);
                    this.update(cx, |this, cx| this.update_pdf(pdf, cx))
                        .log_err();
                }));
            }
        }
    }

    fn copy(&mut self, _: &Copy, _window: &mut Window, cx: &mut Context<Self>) {
        let Some(selection) = self.text_selection.clone() else {
            return;
        };
        let (start, end) = selection.ordered();
        let pdf = self.pdf(cx);
        let mut page_text_spans = Vec::new();
        for page_index in start.page_index..=end.page_index {
            let spans = pdf.text_spans_for_page(page_index).or_else(|| {
                self.stale_page_cache
                    .get(&page_index)
                    .map(|data| data.text_spans.clone())
            });
            if let Some(spans) = spans {
                page_text_spans.push((page_index, spans));
            }
        }
        let text = selection_to_text(&selection, &page_text_spans);
        if !text.is_empty() {
            cx.write_to_clipboard(gpui::ClipboardItem::new_string(text));
        }
    }

    /// Selects all text in the document.
    ///
    /// Extracts the text of every page without rasterizing, so unrendered pages are included.
    fn select_all(&mut self, _: &SelectAll, _window: &mut Window, cx: &mut Context<Self>) {
        let pdf = self.pdf(cx);
        let page_count = pdf.metadata.page_count;
        if page_count == 0 {
            return;
        }

        self.select_all_task = Some(cx.spawn(async move |this, cx| {
            // extract one page at a time, since concurrent requests would queue ahead of the
            // visible page render on the single PDFium mutex.
            for page_index in 0..page_count {
                let task = cx.update(|cx| pdf.clone().request_text_spans(cx, page_index));
                task.await;
            }

            this.update(cx, |this, cx| {
                let pdf = this.pdf(cx);
                let pages_with_text: Vec<(usize, usize)> = (0..page_count)
                    .filter_map(|page_index| {
                        let spans = pdf.text_spans_for_page(page_index).or_else(|| {
                            this.stale_page_cache
                                .get(&page_index)
                                .map(|data| data.text_spans.clone())
                        })?;
                        (!spans.is_empty()).then_some((page_index, spans.len()))
                    })
                    .collect();

                let Some(selection) = select_all_range(&pages_with_text) else {
                    return;
                };
                this.text_selection = Some(selection);
                this.is_selecting_text = false;
                cx.notify();
            })
            .ok();
        }));
    }

    fn toggle_outline(&mut self, _: &ToggleOutline, _window: &mut Window, cx: &mut Context<Self>) {
        self.show_outline = !self.show_outline;
        cx.notify();
    }

    fn toggle_thumbnails(
        &mut self,
        _: &ToggleThumbnails,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.show_thumbnails = !self.show_thumbnails;
        cx.notify();
    }

    fn toggle_single_page_view(
        &mut self,
        _: &ToggleSinglePageView,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.view_mode = match self.view_mode {
            ViewMode::Continuous => ViewMode::SinglePage,
            ViewMode::SinglePage => ViewMode::Continuous,
        };
        // re-snap to the page top so the view is not left mid-scroll in either mode.
        self.jump_to_page(self.current_page, cx);
    }

    fn toggle_inverted_colors(
        &mut self,
        _: &ToggleInvertedColors,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.inverted_colors = !self.inverted_colors;
        cx.notify();
    }

    fn render_style(&self) -> RenderStyle {
        RenderStyle {
            rotation: self.rotation,
            inverted: self.inverted_colors,
        }
    }

    fn rotate_page_clockwise(
        &mut self,
        _: &RotatePageClockwise,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.rotation = self.rotation.rotated_clockwise();
        // a quarter turn swaps the page dimensions, so re-snap the scroll position.
        self.jump_to_page(self.current_page, cx);
        cx.notify();
    }

    fn open_in_default_app(
        &mut self,
        _: &OpenInDefaultApp,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let path = self.file_path_for_system(cx);
        cx.spawn(async move |_, cx| {
            let path = path.await?;
            cx.update(|cx| cx.open_with_system(&path));
            anyhow::Ok(())
        })
        .detach_and_log_err(cx);
    }

    fn show_document_properties(
        &mut self,
        _: &ShowDocumentProperties,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(workspace) = Workspace::for_window(window, cx) else {
            return;
        };
        let pdf = self.pdf(cx);
        let page_index = self.current_page;
        let directory = match &self.source {
            PdfSource::Item(item) => item
                .read(cx)
                .abs_path(cx)
                .and_then(|path| path.parent().map(Path::to_path_buf)),
            PdfSource::Memory(_) => None,
        }
        .unwrap_or_default();
        workspace.update(cx, |workspace, cx| {
            workspace.toggle_modal(window, cx, |_window, cx| {
                DocumentPropertiesModal::new(pdf, page_index, directory, cx)
            });
        });
    }

    /// Returns a file holding the document, writing generated documents such as Typst previews,
    /// which only exist in memory, to a temporary file.
    fn file_path_for_system(&self, cx: &mut Context<Self>) -> Task<Result<std::path::PathBuf>> {
        if let PdfSource::Item(item) = &self.source
            && let Some(abs_path) = item.read(cx).abs_path(cx)
        {
            return Task::ready(Ok(abs_path));
        }

        let pdf = self.pdf(cx);
        let path = std::env::temp_dir().join(format!("zed-pdf-{:016x}.pdf", pdf.id));
        cx.background_spawn(async move {
            std::fs::write(&path, pdf.bytes.as_slice())
                .with_context(|| format!("writing {path:?}"))?;
            Ok(path)
        })
    }

    fn render_thumbnail_sidebar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let page_count = self.pdf(cx).metadata.page_count;

        div()
            .id("pdf-thumbnail-sidebar")
            .w(px(THUMBNAIL_SIDEBAR_WIDTH))
            .h_full()
            .flex_shrink_0()
            .bg(cx.theme().colors().panel_background)
            .border_r_1()
            .border_color(cx.theme().colors().border)
            .child(
                uniform_list(
                    "pdf-thumbnails-list",
                    page_count,
                    cx.processor(move |view, range: std::ops::Range<usize>, window, cx| {
                        view.pdf(cx).evict_thumbnails_outside(
                            range.start.saturating_sub(THUMBNAIL_CACHE_MARGIN)
                                ..range.end + THUMBNAIL_CACHE_MARGIN,
                        );
                        range
                            .map(|page_idx| view.render_thumbnail_row(page_idx, window, cx))
                            .collect::<Vec<_>>()
                    }),
                )
                .size_full(),
            )
    }

    fn render_thumbnail_row(
        &mut self,
        page_idx: usize,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let pdf = self.pdf(cx);
        let is_current = page_idx == self.current_page;

        // start or reuse a thumbnail render; `on_rendered` re-renders this row.
        let view_weak = cx.entity().downgrade();
        let inverted = self.inverted_colors;
        if let Some(task) = pdf
            .clone()
            .request_thumbnail(cx, page_idx, inverted, move |cx| {
                if let Some(view) = view_weak.upgrade() {
                    view.update(cx, |_, cx| cx.notify());
                }
            })
        {
            task.detach();
        }

        let thumbnail = pdf.thumbnail_cache.read().get(&page_idx).cloned();
        let max_width = THUMBNAIL_SIDEBAR_WIDTH - 16.0;
        let max_height = THUMBNAIL_ROW_HEIGHT - 28.0;
        let (thumb_width, thumb_height) = pdf
            .get_page_size(page_idx)
            .ok()
            .map(|size| {
                let aspect = (size.width / size.height).max(0.01);
                if max_width / aspect <= max_height {
                    (max_width, max_width / aspect)
                } else {
                    (max_height * aspect, max_height)
                }
            })
            .unwrap_or((max_width, max_height));

        v_flex()
            .id(("pdf-thumbnail", page_idx))
            .h(px(THUMBNAIL_ROW_HEIGHT))
            .w_full()
            .items_center()
            .justify_center()
            .gap_1()
            .cursor_pointer()
            .when(is_current, |this| {
                this.bg(cx.theme().colors().element_selected)
            })
            .hover(|style| style.bg(cx.theme().colors().element_hover))
            .on_click(cx.listener(move |this, _, _window, cx| {
                this.jump_to_page(page_idx, cx);
            }))
            .child(
                div()
                    .w(px(thumb_width))
                    .h(px(thumb_height))
                    .flex_none()
                    .border_1()
                    .border_color(if is_current {
                        cx.theme().colors().border_focused
                    } else {
                        cx.theme().colors().border
                    })
                    .bg(if inverted {
                        gpui::black()
                    } else {
                        gpui::white()
                    })
                    .when_some(thumbnail, |this, data| {
                        this.child(img(data.image.clone()).size_full())
                    }),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(cx.theme().colors().text_muted)
                    .child((page_idx + 1).to_string()),
            )
            .into_any_element()
    }

    fn is_outline_node_expanded(&self, path: &[usize]) -> bool {
        self.outline_expanded.is_expanded(path)
    }

    fn toggle_outline_node(&mut self, path: Vec<usize>, cx: &mut Context<Self>) {
        self.outline_expanded.toggle(path);
        cx.notify();
    }

    fn render_outline_sidebar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let pdf = self.pdf(cx);
        v_flex()
            .id("pdf-outline-sidebar")
            .w(px(240.0))
            .h_full()
            .flex_shrink_0()
            .overflow_y_scroll()
            .bg(cx.theme().colors().panel_background)
            .border_r_1()
            .border_color(cx.theme().colors().border)
            .py_1()
            .when(pdf.metadata.chapters.is_empty(), |this| {
                this.child(
                    div()
                        .p_2()
                        .text_color(cx.theme().colors().text_muted)
                        .child("No bookmarks in this document"),
                )
            })
            .children(
                pdf.metadata
                    .chapters
                    .iter()
                    .enumerate()
                    .map(|(index, chapter)| self.render_outline_node(chapter, vec![index], 0, cx)),
            )
    }

    fn render_outline_node(
        &self,
        chapter: &PdfChapter,
        path: Vec<usize>,
        depth: usize,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let has_children = !chapter.children.is_empty();
        let expanded = has_children && self.is_outline_node_expanded(&path);
        let target = chapter.target.clone();
        let title = chapter
            .title
            .clone()
            .unwrap_or_else(|| "Untitled".to_string());

        let toggle_path = path.clone();
        let path_hash = gpui::hash(&path);
        let row = h_flex()
            .id(("pdf-outline-node", path_hash))
            .w_full()
            .gap_1()
            .pl(px(8.0 + depth as f32 * 14.0))
            .pr_1()
            .py_0p5()
            .cursor_pointer()
            .hover(|style| style.bg(cx.theme().colors().element_hover))
            .child(div().w(px(14.0)).flex_none().when(has_children, |this| {
                this.child(
                    div()
                        .id(("pdf-outline-toggle", path_hash))
                        .cursor_pointer()
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |this, _, _window, cx| {
                                this.toggle_outline_node(toggle_path.clone(), cx);
                            }),
                        )
                        .child(
                            Icon::new(if expanded {
                                IconName::ChevronDown
                            } else {
                                IconName::ChevronRight
                            })
                            .size(IconSize::XSmall),
                        ),
                )
            }))
            .child(
                div()
                    .flex_1()
                    .overflow_hidden()
                    .text_ellipsis()
                    .child(title),
            )
            .on_click(cx.listener(move |this, _, _window, cx| {
                this.scroll_to_target(target.clone(), ScrollMode::Center, ScrollAnchor::Top, cx);
            }));

        if expanded {
            v_flex()
                .child(row)
                .children(chapter.children.iter().enumerate().map(|(i, child)| {
                    let mut child_path = path.clone();
                    child_path.push(i);
                    self.render_outline_node(child, child_path, depth + 1, cx)
                }))
                .into_any_element()
        } else {
            row.into_any_element()
        }
    }

    fn zoom_in(&mut self, _: &ZoomIn, _window: &mut Window, cx: &mut Context<Self>) {
        log::debug!("Clicked zoom in");
        self.set_zoom(self.zoom_level * ZOOM_STEP, None, cx);
    }

    fn zoom_out(&mut self, _: &ZoomOut, _window: &mut Window, cx: &mut Context<Self>) {
        self.set_zoom(self.zoom_level / ZOOM_STEP, None, cx);
    }

    fn reset_zoom(&mut self, _: &ResetZoom, _window: &mut Window, cx: &mut Context<Self>) {
        self.set_zoom(1.0, None, cx);
    }

    fn fit_to_view(&mut self, _: &FitToView, _window: &mut Window, cx: &mut Context<Self>) {
        let pdf = self.pdf(cx);
        let page_size = (self.current_page < pdf.metadata.page_count)
            .then(|| self.display_page_size(&pdf, self.current_page));
        if let Some((bounds, size)) = self.container_bounds.zip(page_size) {
            let new_zoom = PdfView::compute_fit_to_view_zoom(bounds, size);
            self.set_zoom(new_zoom, None, cx);
        }
    }

    fn fit_to_width(&mut self, _: &FitToWidth, _window: &mut Window, cx: &mut Context<Self>) {
        let pdf = self.pdf(cx);
        let page_size = (self.current_page < pdf.metadata.page_count)
            .then(|| self.display_page_size(&pdf, self.current_page));
        if let Some((bounds, size)) = self.container_bounds.zip(page_size) {
            let container_width: f32 = bounds.size.width.into();
            let scale_x = container_width / size.width;
            self.set_zoom(scale_x * 0.98, None, cx); // Slight margin to accommodate potential scrollbars
        }
    }

    fn next_page(&mut self, _: &NextPage, _window: &mut Window, cx: &mut Context<Self>) {
        let pdf = self.pdf(cx);
        let target = self.current_page.saturating_add(1);
        if target < pdf.metadata.page_count {
            self.jump_to_page(target, cx);
        }
    }

    fn previous_page(&mut self, _: &PreviousPage, _window: &mut Window, cx: &mut Context<Self>) {
        let target = self.current_page.saturating_sub(1);
        self.jump_to_page(target, cx);
    }

    fn first_page(&mut self, _: &FirstPage, _window: &mut Window, cx: &mut Context<Self>) {
        self.jump_to_page(0, cx);
    }

    fn last_page(&mut self, _: &LastPage, _window: &mut Window, cx: &mut Context<Self>) {
        let page_count = self.pdf(cx).metadata.page_count;
        if let Some(last_page) = page_count.checked_sub(1) {
            self.jump_to_page(last_page, cx);
        }
    }

    fn compute_fit_to_view_zoom(container_bounds: Bounds<Pixels>, page_size: Size<f32>) -> f32 {
        let container_width: f32 = container_bounds.size.width.into();
        let container_height: f32 = container_bounds.size.height.into();
        let scale_x = container_width / page_size.width;
        let scale_y = container_height / page_size.height;
        let zoom = scale_x.min(scale_y).min(1.0);
        // unsized panes and zero-sized pages produce 0, NaN, or infinity here.
        if zoom.is_finite() && zoom > 0.0 {
            zoom.clamp(MIN_ZOOM, MAX_ZOOM)
        } else {
            1.0
        }
    }

    fn zoom_to_actual_size(
        &mut self,
        _: &ZoomToActualSize,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.zoom_level = 1.0;
        cx.notify();
    }

    fn set_zoom(
        &mut self,
        new_zoom: f32,
        zoom_center: Option<Point<Pixels>>,
        cx: &mut Context<Self>,
    ) {
        self.new_state = true;

        let old_zoom = self.zoom_level;

        if (new_zoom - old_zoom).abs() > 0.001 {
            // `read` ensures the cache update is not skipped while a background task inserts.
            let pdf = self.pdf(cx);
            let cache = pdf.page_cache.read();
            for (k, v) in cache.iter() {
                let should_insert = self
                    .stale_page_cache
                    .get(k)
                    .is_none_or(|stale| v.scale >= stale.scale);
                if should_insert {
                    self.stale_page_cache.insert(*k, v.clone());
                }
            }
            log::debug!(
                "Stale page cache stored (size {:?})",
                self.stale_page_cache.len()
            );
        }

        self.zoom_level = new_zoom.clamp(MIN_ZOOM, MAX_ZOOM);
        let zoom_ratio = self.zoom_level / old_zoom;

        let mut offset = self.scroll_handle.offset();
        let pdf = self.pdf(cx);

        let center_x;
        let center_y;
        if let Some((center, bounds)) = zoom_center.zip(self.container_bounds) {
            center_x = center.x - bounds.origin.x;
            center_y = center.y - bounds.origin.y;
        } else if let Some(bounds) = self.container_bounds {
            center_x = bounds.size.width / 2.0;
            center_y = bounds.size.height / 2.0;
        } else {
            offset.x *= zoom_ratio;
            offset.y *= zoom_ratio;
            self.scroll_handle.set_offset(offset);
            cx.notify();
            return;
        }

        let page_count = pdf.metadata.page_count;
        if page_count == 0 {
            offset.x *= zoom_ratio;
            offset.y *= zoom_ratio;
            self.scroll_handle.set_offset(offset);
            cx.notify();
            return;
        }

        // the base offset changes with margin centering.
        let viewport_width = self
            .container_bounds
            .map(|b| b.size.width)
            .unwrap_or(px(0.0));
        let max_width = self.display_max_width(&pdf);
        let old_max_width = px(max_width * old_zoom);
        let new_max_width = px(max_width * self.zoom_level);

        let old_base_offset_x = (viewport_width - old_max_width).max(px(0.0)) / 2.0;
        let new_base_offset_x = (viewport_width - new_max_width).max(px(0.0)) / 2.0;

        // horizontal document position relative to the document origin.
        let doc_x = -offset.x + center_x - old_base_offset_x;
        offset.x = -((doc_x * zoom_ratio) + new_base_offset_x - center_x);

        // vertical document position including unscaled gaps.
        let doc_y = -offset.y + center_y;

        let gap = px(PAGE_GAP);
        let laid_out_pages = self.laid_out_pages(page_count);
        let mut current_top = px(0.0);
        let mut target_page = laid_out_pages.end.saturating_sub(1);
        let mut local_y = px(0.0);

        for i in laid_out_pages.clone() {
            let page_height = px(self.display_page_size(&pdf, i).height * old_zoom);
            if doc_y < current_top + page_height + gap {
                target_page = i;
                local_y = doc_y - current_top;
                break;
            }
            current_top += page_height + gap;
        }

        let mut new_current_top = px(0.0);
        for i in laid_out_pages.start..target_page {
            new_current_top += px(self.display_page_size(&pdf, i).height * self.zoom_level) + gap;
        }

        let new_doc_y = new_current_top + local_y * zoom_ratio;
        offset.y = -(new_doc_y - center_y);

        self.scroll_handle.set_offset(offset);
        cx.notify();
    }

    /// Applies a saved zoom level and page, such as after deserialization.
    ///
    /// Must be called before the first `prepaint` to suppress the initial fit-to-view.
    pub fn restore_view_state(
        &mut self,
        zoom_level: f32,
        page_index: usize,
        cx: &mut Context<Self>,
    ) {
        self.zoom_level = zoom_level.clamp(MIN_ZOOM, MAX_ZOOM);
        self.suppress_initial_fit_to_view = true;
        self.jump_to_page(page_index, cx);
    }

    /// Tracks scrolling in single-page mode and flips the page once the scrolling goes far
    /// enough past the edge of the current one.
    ///
    /// `delta_y` is positive when scrolling up, like `ScrollWheelEvent` deltas. Returns whether
    /// the page flipped.
    fn scroll_single_page(
        &mut self,
        delta_y: f32,
        touch_phase: gpui::TouchPhase,
        now: Instant,
        cx: &mut Context<Self>,
    ) -> bool {
        let pdf = self.pdf(cx);
        let Some(viewport) = self.container_bounds else {
            return false;
        };
        let Some(range) = single_page_scroll_range(
            &pdf,
            self.current_page,
            self.zoom_level,
            self.rotation,
            viewport.size.height.as_f32(),
        ) else {
            return false;
        };

        let gesture_started = touch_phase == gpui::TouchPhase::Started
            || self
                .page_flip
                .last_scroll_at
                .is_none_or(|last| now.saturating_duration_since(last) > PAGE_FLIP_GESTURE_GAP);
        if gesture_started {
            self.page_flip = PageFlip::default();
        }
        self.page_flip.last_scroll_at = Some(now);

        let scroll_y = -self.scroll_handle.offset().y.as_f32();
        let at_edge = (delta_y < 0.0 && scroll_y >= *range.end() - 0.5)
            || (delta_y > 0.0 && scroll_y <= *range.start() + 0.5);
        let overscroll = if at_edge { -delta_y } else { 0.0 };
        if overscroll == 0.0 {
            self.page_flip.overscroll = 0.0;
            return false;
        }
        if self.page_flip.flipped {
            return false;
        }
        if overscroll.signum() != self.page_flip.overscroll.signum() {
            self.page_flip.overscroll = 0.0;
        }
        self.page_flip.overscroll += overscroll;
        if self.page_flip.overscroll.abs() < PAGE_FLIP_OVERSCROLL {
            return false;
        }

        self.page_flip.overscroll = 0.0;
        if overscroll > 0.0 {
            if self.current_page + 1 < pdf.metadata.page_count {
                self.page_flip.flipped = true;
                self.jump_to_page(self.current_page + 1, cx);
            }
        } else if let Some(previous_page) = self.current_page.checked_sub(1) {
            self.page_flip.flipped = true;
            // going back lands on the end of the previous page, where reading left off.
            if let Some(previous_range) = single_page_scroll_range(
                &pdf,
                previous_page,
                self.zoom_level,
                self.rotation,
                viewport.size.height.as_f32(),
            ) {
                let mut offset = self.scroll_handle.offset();
                offset.y = px(-previous_range.end());
                self.scroll_handle.set_offset(offset);
            }
            self.current_page = previous_page;
            cx.notify();
        }
        self.page_flip.flipped
    }

    /// Returns the pages laid out in the scrolled content, which is only the shown page in
    /// single-page mode.
    fn laid_out_pages(&self, page_count: usize) -> std::ops::Range<usize> {
        match self.view_mode {
            ViewMode::Continuous => 0..page_count,
            ViewMode::SinglePage => self.current_page..(self.current_page + 1).min(page_count),
        }
    }

    /// Returns the on-screen top of a page in the scrolled document, in pixels.
    ///
    /// In single-page mode a page is laid out alone once shown, so its top is always 0.
    fn page_top(&self, pdf: &Pdf, page_index: usize) -> f32 {
        if self.view_mode == ViewMode::SinglePage {
            return 0.0;
        }
        (0..page_index)
            .map(|index| self.display_page_size(pdf, index).height * self.zoom_level + PAGE_GAP)
            .sum()
    }

    /// Returns the viewport scroll position relative to the top-left corner of the current
    /// page, in points, so it survives zoom changes.
    pub fn scroll_position_in_page(&self, cx: &App) -> Point<f32> {
        let pdf = self.pdf(cx);
        let offset = self.scroll_handle.offset();
        Point {
            x: -offset.x.as_f32() / self.zoom_level,
            y: (-offset.y.as_f32() - self.page_top(&pdf, self.current_page)) / self.zoom_level,
        }
    }

    /// Scrolls so the viewport starts at a position relative to the top-left corner of a page,
    /// as returned by [`Self::scroll_position_in_page`].
    pub fn scroll_to_position_in_page(
        &mut self,
        page_index: usize,
        position: Point<f32>,
        cx: &mut Context<Self>,
    ) {
        let pdf = self.pdf(cx);
        if page_index >= pdf.metadata.page_count {
            return;
        }
        let y = (self.page_top(&pdf, page_index) + position.y * self.zoom_level).max(0.0);
        let x = (position.x * self.zoom_level).max(0.0);
        self.scroll_handle.set_offset(gpui::point(px(-x), px(-y)));
        self.current_page = page_index;
        cx.notify();
    }

    /// Scrolls to the top of a page.
    pub fn jump_to_page(&mut self, page_index: usize, cx: &mut Context<Self>) {
        // the page point that lands in the top-left corner of the rotated page.
        let page_size = self.pdf(cx).get_page_size(page_index).unwrap_or_default();
        let point = self.rotation.to_page_point(Point::default(), page_size);
        self.jump_to_target(
            PdfTarget {
                page_index,
                point,
                block_bottom_y: None,
                line_height: None,
            },
            ScrollMode::Top,
            cx,
        );
    }

    /// Scrolls to a point on a page, positioned according to `mode`.
    pub fn jump_to_target(&mut self, target: PdfTarget, mode: ScrollMode, cx: &mut Context<Self>) {
        use ScrollMode::*;

        let pdf = self.pdf(cx);

        if target.page_index >= pdf.metadata.page_count {
            return; // out of bounds
        }

        let zoom = self.zoom_level;
        let mut total_y_offset = self.page_top(&pdf, target.page_index);

        // add the offset within the target page.
        let point = self.display_target_point(&pdf, &target);
        total_y_offset += point.y * zoom;
        let x_offset = point.x * zoom;

        // scroll offsets are negative.
        let mut scroll_y = total_y_offset;
        let mut scroll_x = x_offset;

        match mode {
            Highlight | Center => {
                if let Some(bounds) = self.container_bounds {
                    scroll_y = (scroll_y - bounds.size.height.as_f32() / 2.0).max(0.0);
                    scroll_x = (scroll_x - bounds.size.width.as_f32() / 2.0).max(0.0);
                } else {
                    scroll_y = (scroll_y - DEFAULT_OFFSET).max(0.0);
                    scroll_x = (scroll_x - DEFAULT_OFFSET).max(0.0);
                }
            }
            Top => {
                scroll_y = (scroll_y - DEFAULT_OFFSET).max(0.0);
                scroll_x = (scroll_x - DEFAULT_OFFSET).max(0.0);
            }
        }

        self.scroll_handle
            .set_offset(gpui::point(px(-scroll_x), px(-scroll_y)));

        self.current_page = target.page_index;
        cx.notify();
    }

    pub fn scroll_to_target(
        &mut self,
        target: PdfTarget,
        mode: ScrollMode,
        anchor: ScrollAnchor,
        cx: &mut Context<Self>,
    ) {
        if mode == ScrollMode::Highlight {
            self.active_jump = Some((anchor, target.clone()));
            self.clear_jump_task = Some(cx.spawn(async move |this, cx| {
                cx.background_executor()
                    .timer(Duration::from_millis(1500))
                    .await;
                this.update(cx, |this, cx| {
                    this.active_jump = None;
                    cx.notify();
                })
                .ok();
            }));
        }

        let pdf = self.pdf(cx);
        if target.page_index >= pdf.metadata.page_count {
            return;
        }

        // single-page mode must switch pages even when the target sits where the viewport is.
        if self.view_mode == ViewMode::SinglePage && target.page_index != self.current_page {
            self.jump_to_target(target, mode, cx);
            return;
        }

        let zoom = self.zoom_level;
        let mut target_y = self.page_top(&pdf, target.page_index);
        let point = self.display_target_point(&pdf, &target);
        target_y += point.y * zoom;
        let target_x = point.x * zoom;

        if let Some(bounds) = self.container_bounds {
            let scroll = self.scroll_handle.offset();
            let scroll_x = scroll.x.as_f32();
            let scroll_y = scroll.y.as_f32();
            let viewport = bounds.size;

            let in_view_y = target_y >= -scroll_y + DEFAULT_OFFSET
                && target_y <= -scroll_y + viewport.height.as_f32() - DEFAULT_OFFSET;
            let in_view_x = target_x >= -scroll_x
                && target_x <= -scroll_x + viewport.width.as_f32() - DEFAULT_OFFSET;

            if in_view_y && in_view_x {
                return;
            }
        }

        self.jump_to_target(target, mode, cx);
    }
}

struct PdfContentElement {
    pdf_view: Entity<PdfView>,
}

impl PdfContentElement {
    fn new(pdf_view: Entity<PdfView>) -> Self {
        Self { pdf_view }
    }
}

impl IntoElement for PdfContentElement {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for PdfContentElement {
    type RequestLayoutState = ();
    type PrepaintState = Option<AnyElement>;

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
        let (
            do_log,
            pdf,
            stale_pdf,
            first_layout,
            suppress_initial_fit_to_view,
            current_zoom_level,
            current_page,
            scroll_handle,
            search_matches,
            active_match,
            active_jump,
            view_mode,
            rotation,
            render_style,
        ) = {
            let pdf_view = self.pdf_view.read(cx);
            (
                pdf_view.new_state,
                pdf_view.pdf(cx),
                pdf_view.stale_pdf(cx),
                pdf_view.container_bounds.is_none(),
                pdf_view.suppress_initial_fit_to_view,
                pdf_view.zoom_level,
                pdf_view.current_page,
                pdf_view.scroll_handle.clone(),
                pdf_view.search_matches.clone(),
                pdf_view.active_match_index,
                pdf_view.active_jump.clone(),
                pdf_view.view_mode,
                pdf_view.rotation,
                pdf_view.render_style(),
            )
        };

        let page_count = pdf.metadata.page_count;
        let first_page = pdf.metadata.pages.first()?;

        // a quarter turn swaps the on-screen page box so the rotated image fills it.
        let display_page_size = |size: Size<f32>| rotation.display_size(size);

        // a restored zoom takes priority over the initial fit-to-view.
        let initial_zoom_level = (first_layout && !suppress_initial_fit_to_view)
            .then(|| PdfView::compute_fit_to_view_zoom(bounds, display_page_size(first_page.size)));

        let zoom_level = initial_zoom_level.unwrap_or(current_zoom_level);

        // rasterize at physical pixel density to stay sharp on HiDPI displays. Every comparison
        // against a cached `PageData::scale` must use this value, or pages never match.
        let render_scale = zoom_level * window.scale_factor();

        let gap = px(PAGE_GAP);

        // the scroll container clamps the offset too, but only after the hit regions below were
        // placed with the unclamped one.
        if view_mode == ViewMode::SinglePage
            && let Some(range) = single_page_scroll_range(
                &pdf,
                current_page,
                zoom_level,
                rotation,
                bounds.size.height.as_f32(),
            )
        {
            let mut offset = scroll_handle.offset();
            let scroll_y = (-offset.y.as_f32()).clamp(*range.start(), *range.end());
            if scroll_y != -offset.y.as_f32() {
                offset.y = px(-scroll_y);
                scroll_handle.set_offset(offset);
            }
        }

        let scroll_x = scroll_handle.offset().x.abs();
        let scroll_y = scroll_handle.offset().y.abs();
        let viewport_top = scroll_y;
        let viewport_bottom = scroll_y + bounds.size.height;

        let mut max_visible_height = px(0.0);
        let mut best_page = current_page;
        let mut current_top = px(0.0);

        let mut first_visible: Option<usize> = None;
        let mut last_visible: Option<usize> = None;

        let mut page_dimensions = Vec::with_capacity(page_count);
        let mut page_tops = Vec::with_capacity(page_count);

        for (page_idx, page) in pdf.metadata.pages.iter().enumerate() {
            let page_size = display_page_size(page.size);
            let width = px(page_size.width * zoom_level);
            let height = px(page_size.height * zoom_level);

            page_dimensions.push((width, height));
            page_tops.push(current_top);

            let page_bottom = current_top + height;
            let visible_start = viewport_top.max(current_top);
            let visible_end = viewport_bottom.min(page_bottom);
            let visible_height = (visible_end - visible_start).max(px(0.0));

            // any visible pixel counts.
            if visible_height > px(0.0) {
                if first_visible.is_none() {
                    first_visible = Some(page_idx);
                }
                last_visible = Some(page_idx);
            }

            // track the most visible page for the sidebar.
            if visible_height > max_visible_height {
                max_visible_height = visible_height;
                best_page = page_idx;
            }

            current_top += height + gap;
        }

        let first_visible = first_visible.unwrap_or(best_page);
        let last_visible = last_visible.unwrap_or(best_page);

        // single-page mode pins the visible range to `current_page`, which `jump_to_page` keeps
        // aligned with the scroll offset.
        let (first_visible, last_visible) = if view_mode == ViewMode::SinglePage {
            (current_page, current_page)
        } else {
            (first_visible, last_visible)
        };

        let mut all_visible_rendered = true;
        for p in first_visible..=last_visible {
            if !pdf.page_cache.read().contains_key(&p) {
                all_visible_rendered = false;
                break;
            }
        }

        if all_visible_rendered && stale_pdf.is_some() {
            self.pdf_view
                .update(cx, |this, cx| this.clear_stale_pdf(cx));
        }

        if current_page != best_page && view_mode == ViewMode::Continuous {
            let view = self.pdf_view.clone();
            view.update(cx, |this, _| {
                this.current_page = best_page;
            });
        }

        let start_prerender = first_visible.saturating_sub(PRERENDER_PAGES);
        let end_prerender = (last_visible + PRERENDER_PAGES).min(page_count.saturating_sub(1));

        self.pdf_view.update(cx, |this, _| {
            this.new_state = false;
            this.current_tasks
                .retain(|&page_idx, _| page_idx >= start_prerender && page_idx <= end_prerender);
        });

        // evict bitmaps of pages far outside the prerender window.
        pdf.page_cache
            .write()
            .retain(|&page_idx, _| page_idx >= start_prerender && page_idx <= end_prerender);
        pdf.failed_pages
            .lock()
            .retain(|&page_idx, _| page_idx >= start_prerender && page_idx <= end_prerender);

        // visible pages render at high priority and prerender pages at low priority, so newly
        // visible pages are not stuck behind prefetch work.
        let mut render_queue = Vec::new();

        for p in first_visible..=last_visible {
            render_queue.push((p, gpui::Priority::High));
        }

        for offset in 1..=PRERENDER_PAGES {
            let forward = last_visible + offset;
            if forward < page_count {
                render_queue.push((forward, gpui::Priority::Low));
            }
            if let Some(backward) = first_visible.checked_sub(offset) {
                render_queue.push((backward, gpui::Priority::Low));
            }
        }

        for (page_idx, priority) in render_queue {
            let pdf = pdf.clone();
            let pdf_view_weak = self.pdf_view.downgrade();

            // the viewport center clamped into this page selects the strip for tall pages, so
            // off-screen pages render the edge closest to the viewport.
            let page_top = page_tops[page_idx];
            let page_height = page_dimensions[page_idx].1;
            let clamped_center = ((viewport_top + viewport_bottom) / 2.0)
                .max(page_top)
                .min(page_top + page_height);
            let visible_center_y: f32 = f32::from(clamped_center - page_top) / zoom_level;

            self.pdf_view.update(cx, |this, cx| {
                if let Some(task) = pdf.clone().request_page(
                    cx,
                    page_idx,
                    render_scale,
                    visible_center_y,
                    render_style,
                    priority,
                    move |cx| {
                        if let Some(view) = pdf_view_weak.upgrade() {
                            view.update(cx, |this, cx| {
                                this.current_tasks.remove(&page_idx);
                                cx.notify();
                            });
                        }
                    },
                ) {
                    this.current_tasks.insert(page_idx, task);
                }
            });
        }

        self.pdf_view.update(cx, |this, _| {
            this.container_bounds = Some(bounds);
            if let Some(initial_zoom_level) = initial_zoom_level {
                this.zoom_level = initial_zoom_level;
            }
        });

        // center the pages without clipping the scroll axis.
        let max_width = page_dimensions
            .iter()
            .map(|(w, _)| w.into())
            .fold(0.0f32, f32::max);
        let viewport_width = bounds.size.width.into();
        let container_width = max_width.max(viewport_width);

        let mut children: Vec<Div> = Vec::new();
        let tasks_empty = self.pdf_view.read(cx).current_tasks.is_empty();

        let mut need_cache = false;
        let mut current_render_top = px(0.0);

        let text_selection = self.pdf_view.read(cx).text_selection.clone();
        // page bounds and text spans for the drag handlers of the outer container.
        let mut selection_hit_regions: Vec<SelectionHitRegion> = Vec::new();

        for (page_idx, &(width, height)) in page_dimensions.iter().enumerate() {
            if view_mode == ViewMode::SinglePage && page_idx != current_page {
                continue;
            }

            let page_margin_left = (container_width - f32::from(width)) / 2.0;
            let global_page_bounds = Bounds {
                origin: Point {
                    x: bounds.origin.x + px(page_margin_left) - scroll_x,
                    y: bounds.origin.y + current_render_top - scroll_y,
                },
                size: Size { width, height },
            };
            // exclude renders at an older resolution.
            let page_data = pdf
                .page_cache
                .read()
                .get(&page_idx)
                .cloned()
                .filter(|data| (data.scale - render_scale).abs() < 0.01)
                .or_else(|| {
                    stale_pdf
                        .as_ref()
                        .and_then(|s| s.page_cache.read().get(&page_idx).cloned())
                });

            let stale_data = self
                .pdf_view
                .read(cx)
                .stale_page_cache
                .get(&page_idx)
                .cloned();

            let page_click_handler = self.pdf_view.read(cx).on_page_click.clone();
            let view_weak = self.pdf_view.downgrade();
            let zoom = zoom_level;

            let active_data = page_data.as_ref().or(stale_data.as_ref());
            // text, links, and matches use unrotated page coordinates in points.
            let page_size = pdf
                .metadata
                .pages
                .get(page_idx)
                .map(|page| page.size)
                .unwrap_or_default();
            let to_display_bounds = |bounds: Bounds<f32>| {
                let bounds = rotation.to_display_bounds(bounds, page_size);
                Bounds {
                    origin: Point {
                        x: px(bounds.origin.x * zoom_level),
                        y: px(bounds.origin.y * zoom_level),
                    },
                    size: Size {
                        width: px(bounds.size.width * zoom_level),
                        height: px(bounds.size.height * zoom_level),
                    },
                }
            };
            if let Some(data) = active_data {
                selection_hit_regions.push(SelectionHitRegion {
                    page_index: page_idx,
                    bounds: global_page_bounds,
                    text_spans: data.text_spans.clone(),
                    page_size,
                });
            }
            let text_spans_for_selection = active_data.map(|data| data.text_spans.clone());
            let view_weak_for_selection = view_weak.clone();

            let mut child_element = div()
                .ml(px(page_margin_left))
                .w(width)
                .h(height)
                .bg(cx.theme().colors().panel_background)
                .border_1()
                .border_color(cx.theme().colors().border)
                .shadow_sm()
                .relative()
                .debug_selector(|| format!("pdf-page-{page_idx}"))
                .on_mouse_down(MouseButton::Left, move |e, window, cx| {
                    if let Some(handler) = &page_click_handler {
                        let point = page_point_at(
                            e.position,
                            global_page_bounds,
                            zoom,
                            rotation,
                            page_size,
                        );
                        if let Some(view) = view_weak.upgrade() {
                            view.update(cx, |_, cx| {
                                handler(page_idx, point, window, cx);
                            });
                        }
                    }
                })
                .on_mouse_down(MouseButton::Left, move |e, _window, cx| {
                    let Some(text_spans) = &text_spans_for_selection else {
                        return;
                    };
                    let local =
                        page_point_at(e.position, global_page_bounds, zoom, rotation, page_size);
                    let Some(char_index) = nearest_char_index(text_spans, local) else {
                        return;
                    };
                    // double-click selects a word and triple-click or more a line. A single click
                    // only records the anchor of a potential drag.
                    if let Some(view) = view_weak_for_selection.upgrade() {
                        view.update(cx, |this, cx| {
                            if e.click_count == 1 {
                                this.text_selection = None;
                                this.pending_click_anchor = Some(TextPosition {
                                    page_index: page_idx,
                                    char_index,
                                });
                            } else {
                                let char_range = if e.click_count == 2 {
                                    word_range_at(text_spans, char_index)
                                } else {
                                    line_range_at(text_spans, char_index)
                                };
                                this.text_selection = Some(TextSelection {
                                    anchor: TextPosition {
                                        page_index: page_idx,
                                        char_index: char_range.start,
                                    },
                                    head: TextPosition {
                                        page_index: page_idx,
                                        char_index: char_range.end.saturating_sub(1),
                                    },
                                });
                                this.pending_click_anchor = None;
                            }
                            this.is_selecting_text = true;
                            cx.notify();
                        });
                    }
                });

            // stack stale and new images.
            let mut image_container = div().size_full().relative();
            let mut need_cache_page = false;

            // stale data goes on the bottom layer to avoid a blank frame. Strip renders are
            // positioned by `page_offset_pt` and `covered_height_pt`.
            if let Some(old_data) = &stale_data {
                if do_log {
                    log::debug!("Drawn stale for page {:?}", page_idx);
                }
                let top = px(old_data.page_offset_pt * zoom_level);
                let strip_height = px(old_data.covered_height_pt() * zoom_level);
                image_container = image_container.child(
                    div()
                        .absolute()
                        .top(top)
                        .left_0()
                        .w_full()
                        .h(strip_height)
                        .child(img(old_data.image.clone()).size_full()),
                );
                need_cache_page = true;
            }

            // new data goes on top when it differs from the stale data.
            if let Some(new_data) = &page_data {
                let is_same = stale_data
                    .as_ref()
                    .is_some_and(|s| Arc::ptr_eq(&s.image, &new_data.image));
                if !is_same {
                    if do_log {
                        log::debug!("Drawn new for page {:?}", page_idx);
                    }
                    let top = px(new_data.page_offset_pt * zoom_level);
                    let strip_height = px(new_data.covered_height_pt() * zoom_level);
                    image_container = image_container.child(
                        div()
                            .absolute()
                            .top(top)
                            .left_0()
                            .w_full()
                            .h(strip_height)
                            .child(
                                img(new_data.image.clone())
                                    .id(("pdf-viewer-doc", page_idx))
                                    .size_full(),
                            ),
                    );
                }
                need_cache_page = false;
            }

            // show a loading state while neither is ready.
            if page_data.is_none() && stale_data.is_none() {
                image_container = image_container.child(
                    div()
                        .absolute()
                        .top_0()
                        .left_0()
                        .size_full()
                        .flex()
                        .items_center()
                        .justify_center()
                        .child("Loading..."),
                );
            }

            child_element = child_element.child(image_container);
            need_cache = need_cache || need_cache_page;

            if let Some(selection) = &text_selection
                && let Some(data) = active_data
                && let Some(range) =
                    selected_char_range_for_page(selection, page_idx, data.text_spans.len())
            {
                let mut selection_layer = div().absolute().top_0().left_0().size_full();
                for rect in selection_highlight_rects(&data.text_spans, range) {
                    let rect = to_display_bounds(rect);
                    selection_layer = selection_layer.child(
                        div()
                            .absolute()
                            .top(rect.origin.y)
                            .left(rect.origin.x)
                            .w(rect.size.width)
                            .h(rect.size.height)
                            .bg(cx.theme().colors().element_selection_background),
                    );
                }
                child_element = child_element.child(selection_layer);
            }

            if let Some(data) = active_data {
                let mut link_layer = div().absolute().top_0().left_0().size_full();

                for (link_idx, link) in data.links.iter().enumerate() {
                    let action = link.action.clone();
                    let view_weak = self.pdf_view.downgrade();

                    let bounds = to_display_bounds(link.bounds);
                    let tooltip: SharedString = match &link.action {
                        PdfLinkAction::External(url) => url.clone().into(),
                        PdfLinkAction::Internal(target) => {
                            format!("Go to Page {}", target.page_index + 1).into()
                        }
                    };

                    link_layer = link_layer.child(
                        div()
                            .id(ElementId::named_usize(
                                format!("pdf-link-{page_idx}"),
                                link_idx,
                            ))
                            .absolute()
                            .top(bounds.origin.y)
                            .left(bounds.origin.x)
                            .w(bounds.size.width)
                            .h(bounds.size.height)
                            .rounded_xs()
                            .cursor_pointer()
                            // translucent, since theme hover colors may hide the link text.
                            .hover(|style| style.bg(gpui::rgba(0x0060df22)))
                            .tooltip(Tooltip::text(tooltip))
                            .on_mouse_down(MouseButton::Left, move |_, _window, cx| {
                                log::debug!("Clicked {:?}", action);
                                match action.clone() {
                                    PdfLinkAction::External(url) => {
                                        cx.open_url(&url);
                                    }
                                    PdfLinkAction::Internal(target) => {
                                        if let Some(view) = view_weak.upgrade() {
                                            view.update(cx, |this, cx| {
                                                this.scroll_to_target(
                                                    target,
                                                    ScrollMode::Highlight,
                                                    ScrollAnchor::Top,
                                                    cx,
                                                );
                                            });
                                        }
                                    }
                                }
                            }),
                    );
                }

                child_element = child_element.child(link_layer);
            }

            if !search_matches.is_empty() {
                let mut highlight_layer = div().absolute().top_0().left_0().size_full();

                for (match_idx, m) in search_matches.iter().enumerate() {
                    if m.page_index == page_idx {
                        let is_active = Some(match_idx) == active_match;

                        let bg_color = if is_active {
                            gpui::rgba(0xff990066) // Active Match (Orange-ish)
                        } else {
                            gpui::rgba(0xffff0044) // Inactive Match (Yellow-ish)
                        };
                        let border_color = if is_active {
                            gpui::rgba(0xff9900ff)
                        } else {
                            gpui::rgba(0xffff0088)
                        };

                        for bound in &m.bounds {
                            let bound = to_display_bounds(*bound);

                            highlight_layer = highlight_layer.child(
                                div()
                                    .absolute()
                                    .top(bound.origin.y)
                                    .left(bound.origin.x)
                                    .w(bound.size.width)
                                    .h(bound.size.height)
                                    .bg(bg_color)
                                    .border_1()
                                    .border_color(border_color)
                                    .rounded_sm(),
                            );
                        }
                    }
                }
                child_element = child_element.child(highlight_layer);
            }

            // the bar marks source lines, which only run horizontally on unrotated pages.
            if let Some((anchor, jump)) = &active_jump
                && jump.page_index == page_idx
                && rotation == PageRotation::None
            {
                use ScrollAnchor::*;

                // fall back to 20pt for jump sources without font size information.
                let line_height = px(jump.line_height.unwrap_or(20.0) * zoom_level);
                let y = px(jump.point.y * zoom_level);

                // Typst jumps target the baseline, so shift the bar up by one line.
                let top = match anchor {
                    Top => y,
                    Baseline => y - line_height + px(4.0 * zoom_level),
                };
                // extend the bar to `block_bottom_y`, the baseline of the last line of the
                // source block. Sources without a block fall back to a single line.
                let block_extent =
                    px((jump.block_bottom_y.unwrap_or(jump.point.y) - jump.point.y) * zoom_level);
                let h = line_height + block_extent;

                // a subtle border-colored bar matching the position marker of the Markdown
                // preview.
                child_element = child_element.child(
                    div()
                        .absolute()
                        .top(top)
                        .left(px(8.0))
                        .w(px(4.0))
                        .h(h)
                        .bg(cx.theme().colors().border)
                        .rounded_xs(),
                );
            }

            children.push(child_element);
            current_render_top += height + gap;
        }

        if !children.is_empty() {
            let selection_hit_regions = Arc::new(selection_hit_regions);
            let pdf_view_weak = self.pdf_view.downgrade();

            let mut pdf_content = div()
                .id("pdf-container")
                .size_full()
                .flex()
                .flex_col()
                .gap(gap)
                .overflow_scroll()
                .track_scroll(&scroll_handle)
                .on_mouse_move({
                    let selection_hit_regions = selection_hit_regions.clone();
                    let pdf_view_weak = pdf_view_weak.clone();
                    move |event: &gpui::MouseMoveEvent, _window, cx| {
                        let Some(view) = pdf_view_weak.upgrade() else {
                            return;
                        };
                        view.update(cx, |this, cx| {
                            if !this.is_selecting_text {
                                return;
                            }
                            let Some(position) = hit_test_selection_point(
                                &selection_hit_regions,
                                event.position,
                                zoom_level,
                                rotation,
                            ) else {
                                return;
                            };
                            if let Some(selection) = &mut this.text_selection {
                                if selection.head != position {
                                    selection.head = position;
                                    cx.notify();
                                }
                            } else if let Some(anchor) = this.pending_click_anchor
                                && position != anchor
                            {
                                // the drag left the starting character, so show the selection.
                                this.text_selection = Some(TextSelection {
                                    anchor,
                                    head: position,
                                });
                                cx.notify();
                            }
                        });
                    }
                })
                .on_mouse_up(MouseButton::Left, move |_event, _window, cx| {
                    if let Some(view) = pdf_view_weak.upgrade() {
                        view.update(cx, |this, _cx| {
                            this.is_selecting_text = false;
                            this.pending_click_anchor = None;
                        });
                    }
                });

            for child in children {
                pdf_content = pdf_content.child(child);
            }

            let mut element = pdf_content.into_any_element();

            element.prepaint_as_root(bounds.origin, bounds.size.into(), window, cx);

            self.pdf_view.update(cx, |this, _| {
                if tasks_empty && !need_cache && !this.stale_page_cache.is_empty() {
                    this.stale_page_cache.clear();
                    log::debug!("Cleared stale page cache");
                }
            });

            Some(element)
        } else {
            None
        }
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let Some(mut element) = prepaint.take() else {
            return;
        };

        // captured so that zooming takes precedence over the scroll container below.
        window.on_mouse_event({
            let pdf_view = self.pdf_view.downgrade();
            move |event: &ScrollWheelEvent, phase, window, cx| {
                if phase != DispatchPhase::Capture || !bounds.contains(&event.position) {
                    return;
                }
                let delta = f32::from(event.delta.pixel_delta(window.line_height()).y);
                if !(event.modifiers.control || event.modifiers.platform) {
                    let now = cx.background_executor().now();
                    let flipped = pdf_view
                        .update(cx, |this, cx| {
                            this.view_mode == ViewMode::SinglePage
                                && this.scroll_single_page(delta, event.touch_phase, now, cx)
                        })
                        .log_err()
                        .unwrap_or(false);
                    // the flip already moved the viewport, so the new page does not scroll too.
                    if flipped {
                        cx.stop_propagation();
                    }
                    return;
                }
                let zoom_factor = if delta > 0.0 {
                    1.0 + delta * 0.01
                } else {
                    1.0 / (1.0 - delta * 0.01)
                };
                pdf_view
                    .update(cx, |this, cx| {
                        this.set_zoom(this.zoom_level * zoom_factor, Some(event.position), cx)
                    })
                    .log_err();
                cx.stop_propagation();
            }
        });
        window.on_mouse_event({
            let pdf_view = self.pdf_view.downgrade();
            move |event: &PinchEvent, phase, _window, cx| {
                if phase != DispatchPhase::Bubble || !bounds.contains(&event.position) {
                    return;
                }
                pdf_view
                    .update(cx, |this, cx| {
                        this.set_zoom(
                            this.zoom_level * (1.0 + event.delta),
                            Some(event.position),
                            cx,
                        )
                    })
                    .log_err();
            }
        });

        element.paint(window, cx);
    }
}

/// Returns the vertical scroll offsets that keep `page_index` in view in single-page mode, where
/// it is the only page laid out.
fn single_page_scroll_range(
    pdf: &Pdf,
    page_index: usize,
    zoom: f32,
    rotation: PageRotation,
    viewport_height: f32,
) -> Option<RangeInclusive<f32>> {
    let page = pdf.metadata.pages.get(page_index)?;
    let height = rotation.display_size(page.size).height * zoom;
    Some(0.0..=(height - viewport_height).max(0.0))
}

/// Query and options of a document search, to reuse its results when repeated.
#[derive(Clone, PartialEq)]
struct PdfSearchKey {
    query: SharedString,
    match_case: bool,
    whole_word: bool,
    regex: bool,
}

/// Finds the matches of `query` in the text layer of a page, with one rectangle per line of each
/// match.
fn search_text_spans(
    query: &SearchQuery,
    page_index: usize,
    spans: &[PdfTextSpan],
) -> Vec<PdfSearchResult> {
    let mut text = String::new();
    let mut span_starts = Vec::with_capacity(spans.len());
    for span in spans {
        span_starts.push(text.len());
        text.push_str(&span.text);
    }

    query
        .search_str(&text)
        .into_iter()
        .filter(|range| !range.is_empty())
        .filter_map(|range| {
            let first = span_starts
                .partition_point(|&start| start <= range.start)
                .checked_sub(1)?;
            let end = span_starts.partition_point(|&start| start < range.end);
            let mut line_bounds: Vec<Bounds<f32>> = Vec::new();
            for span in spans.get(first..end)? {
                let bounds = span.bounds;
                // generated spaces and line breaks have no area.
                if bounds.size.width <= 0.0 || bounds.size.height <= 0.0 {
                    continue;
                }
                match line_bounds.last_mut() {
                    Some(line)
                        if (bounds.origin.y - line.origin.y).abs() < line.size.height / 2.0 =>
                    {
                        *line = line.union(&bounds);
                    }
                    _ => line_bounds.push(bounds),
                }
            }
            (!line_bounds.is_empty()).then_some(PdfSearchResult {
                page_index,
                bounds: line_bounds,
            })
        })
        .collect()
}

pub enum PdfViewEvent {
    TitleChanged,
    MatchesInvalidated,
}

impl EventEmitter<PdfViewEvent> for PdfView {}

impl Item for PdfView {
    type Event = PdfViewEvent;

    fn to_item_events(event: &Self::Event, f: &mut dyn FnMut(workspace::item::ItemEvent)) {
        match event {
            PdfViewEvent::TitleChanged => {
                f(workspace::item::ItemEvent::UpdateTab);
                f(workspace::item::ItemEvent::UpdateBreadcrumbs);
            }
            PdfViewEvent::MatchesInvalidated => {}
        }
    }

    fn for_each_project_item(
        &self,
        cx: &App,
        f: &mut dyn FnMut(EntityId, &dyn project::ProjectItem),
    ) {
        if let PdfSource::Item(pdf_item) = &self.source {
            f(pdf_item.entity_id(), pdf_item.read(cx))
        }
    }

    fn tab_tooltip_text(&self, cx: &App) -> Option<SharedString> {
        if let PdfSource::Item(pdf_item) = &self.source {
            let abs_path = pdf_item.read(cx).abs_path(cx)?;
            Some(abs_path.compact().to_string_lossy().into_owned().into())
        } else {
            None
        }
    }

    fn tab_content(&self, params: TabContentParams, _window: &Window, cx: &App) -> AnyElement {
        let label_color = params.text_color();

        Label::new(self.tab_content_text(params.detail.unwrap_or_default(), cx))
            .single_line()
            .color(label_color)
            .when(params.preview, |this| this.italic())
            .into_any_element()
    }

    fn tab_content_text(&self, _: usize, cx: &App) -> SharedString {
        match &self.source {
            PdfSource::Item(pdf_item) => pdf_item.read(cx).file.file_name(cx).to_string().into(),
            PdfSource::Memory(_) => ":memory:".into(),
        }
    }

    fn tab_icon(&self, _: &Window, _cx: &App) -> Option<Icon> {
        Some(Icon::new(IconName::FileDoc))
    }

    fn breadcrumb_location(&self, cx: &App) -> ToolbarItemLocation {
        let show_breadcrumb = EditorSettings::get_global(cx).toolbar.breadcrumbs;
        if show_breadcrumb {
            ToolbarItemLocation::PrimaryLeft
        } else {
            ToolbarItemLocation::Hidden
        }
    }

    fn breadcrumbs(&self, cx: &App) -> Option<(Vec<HighlightedText>, Option<Font>)> {
        let font = ThemeSettings::get_global(cx).buffer_font.clone();
        let text: SharedString =
            if let (PdfSource::Item(pdf_item), Some(project)) = (&self.source, &self.project) {
                breadcrumbs_text_for_pdf(project.read(cx), pdf_item.read(cx), cx).into()
            } else {
                "".into()
            };

        Some((
            vec![HighlightedText {
                text,
                highlights: vec![],
            }],
            Some(font),
        ))
    }

    fn can_split(&self) -> bool {
        true
    }

    fn clone_on_split(
        &self,
        _workspace_id: Option<WorkspaceId>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Option<Entity<Self>>>
    where
        Self: Sized,
    {
        Task::ready(Some(cx.new(|cx| Self {
            source: self.source.clone(),
            stale_pdf: None,
            stale_page_cache: Default::default(),
            last_search_query: None,
            search_error: None,
            project: self.project.clone(),
            focus_handle: cx.focus_handle(),
            scroll_handle: self.scroll_handle.clone(),
            zoom_level: self.zoom_level,
            container_bounds: None,
            current_page: self.current_page,
            current_tasks: Default::default(),
            new_state: false,
            search_matches: Vec::new(),
            active_match_index: None,
            search_token: None,
            active_jump: None,
            clear_jump_task: None,
            on_page_click: self.on_page_click.clone(),
            password_input: Self::new_password_input(window, cx),
            password_task: None,
            select_all_task: None,
            text_selection: None,
            pending_click_anchor: None,
            is_selecting_text: false,
            show_outline: false,
            outline_expanded: Default::default(),
            show_thumbnails: false,
            suppress_initial_fit_to_view: false,
            view_mode: ViewMode::Continuous,
            rotation: PageRotation::None,
            inverted_colors: self.inverted_colors,
            page_flip: PageFlip::default(),
        })))
    }

    fn has_deleted_file(&self, cx: &App) -> bool {
        if let PdfSource::Item(item) = &self.source {
            item.read(cx).file.disk_state().is_deleted()
        } else {
            false
        }
    }

    fn buffer_kind(&self, _: &App) -> ItemBufferKind {
        ItemBufferKind::Singleton
    }

    fn as_searchable(
        &self,
        handle: &Entity<Self>,
        _: &App,
    ) -> Option<Box<dyn SearchableItemHandle>> {
        Some(Box::new(handle.clone()))
    }
}

fn breadcrumbs_text_for_pdf(project: &Project, pdf: &PdfItem, cx: &App) -> String {
    let mut path = pdf.file.path().clone();
    if project.visible_worktrees(cx).count() > 1
        && let Some(worktree) = project.worktree_for_id(pdf.project_path(cx).worktree_id, cx)
    {
        path = worktree.read(cx).root_name().join(&path).into();
    }

    path.display(project.path_style(cx)).to_string()
}

impl SerializableItem for PdfView {
    fn serialized_item_kind() -> &'static str {
        "PdfView"
    }

    fn deserialize(
        project: Entity<Project>,
        _workspace: WeakEntity<Workspace>,
        workspace_id: WorkspaceId,
        item_id: ItemId,
        window: &mut Window,
        cx: &mut App,
    ) -> Task<Result<Entity<Self>>> {
        let db = PdfViewerDb::global(cx);
        window.spawn(cx, async move |cx| {
            let pdf_path = db
                .get_pdf_path(item_id, workspace_id)?
                .context("No pdf path found")?;
            let view_state = db.get_pdf_view_state(item_id, workspace_id)?;

            let (worktree, relative_path) = project
                .update(cx, |project, cx| {
                    project.find_or_create_worktree(pdf_path.clone(), false, cx)
                })
                .await
                .context("Path not found")?;
            let worktree_id = worktree.update(cx, |worktree, _cx| worktree.id());

            let project_path = ProjectPath {
                worktree_id,
                path: relative_path,
            };

            let pdf_item = cx
                .update(|_, cx| open_pdf(project.clone(), project_path, cx))?
                .await?;

            cx.update(|window, cx| {
                Ok(cx.new(|cx| {
                    let mut view = PdfView::new(pdf_item, project, window, cx);
                    if let Some((zoom_level, current_page, page_offset, inverted_colors)) =
                        view_state
                    {
                        view.inverted_colors = inverted_colors.unwrap_or(false);
                        let current_page = current_page.max(0) as usize;
                        view.restore_view_state(zoom_level as f32, current_page, cx);
                        if let Some((x, y)) = page_offset {
                            view.scroll_to_position_in_page(
                                current_page,
                                Point {
                                    x: x as f32,
                                    y: y as f32,
                                },
                                cx,
                            );
                        }
                    }
                    view
                }))
            })?
        })
    }

    fn cleanup(
        workspace_id: WorkspaceId,
        alive_items: Vec<ItemId>,
        _window: &mut Window,
        cx: &mut App,
    ) -> Task<Result<()>> {
        let db = PdfViewerDb::global(cx);
        delete_unloaded_items(alive_items, workspace_id, "pdf_viewers", &db, cx)
    }

    fn serialize(
        &mut self,
        workspace: &mut Workspace,
        item_id: ItemId,
        _closing: bool,
        cx: &mut Context<Self>,
    ) -> Option<Task<Result<()>>> {
        let PdfSource::Item(pdf_item) = &self.source else {
            return None;
        };

        let workspace_id = workspace.database_id()?;
        let pdf_path = pdf_item.read(cx).abs_path(cx)?;
        let zoom_level = self.zoom_level as f64;
        let current_page = self.current_page as i64;
        // rotation is not persisted, so positions on a rotated page would not match on restore.
        let position =
            (self.rotation == PageRotation::None).then(|| self.scroll_position_in_page(cx));
        let page_offset = position.map(|position| (position.x as f64, position.y as f64));
        let inverted_colors = self.inverted_colors;

        let db = PdfViewerDb::global(cx);
        Some(cx.background_spawn({
            async move {
                log::debug!("Saving pdf at path {pdf_path:?}");
                db.save_pdf_view_state(
                    item_id,
                    workspace_id,
                    pdf_path,
                    (zoom_level, current_page),
                    page_offset,
                    inverted_colors,
                )
                .await
            }
        }))
    }

    fn should_serialize(&self, _event: &Self::Event) -> bool {
        false
    }
}

impl EventEmitter<()> for PdfView {}
impl Focusable for PdfView {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for PdfView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let pdf = self.pdf(cx);
        let needs_password = pdf.metadata.needs_password;
        let load_error = (pdf.metadata.page_count == 0 && !needs_password)
            .then(|| pdf.metadata.error.clone())
            .flatten();
        let has_load_error = load_error.is_some();

        div()
            .id("pdf-viewer")
            .track_focus(&self.focus_handle(cx))
            .key_context("PdfViewer")
            .on_action(cx.listener(Self::zoom_in))
            .on_action(cx.listener(Self::zoom_out))
            .on_action(cx.listener(Self::reset_zoom))
            .on_action(cx.listener(Self::fit_to_view))
            .on_action(cx.listener(Self::fit_to_width))
            .on_action(cx.listener(Self::zoom_to_actual_size))
            .on_action(cx.listener(Self::next_page))
            .on_action(cx.listener(Self::previous_page))
            .on_action(cx.listener(Self::first_page))
            .on_action(cx.listener(Self::last_page))
            .on_action(cx.listener(Self::copy))
            .on_action(cx.listener(Self::select_all))
            .size_full()
            .relative()
            .bg(cx.theme().colors().editor_background)
            .when(needs_password, |this| {
                let hint = pdf.metadata.error.clone();
                this.child(
                    div()
                        .key_context("PdfPassword")
                        .on_action(cx.listener(Self::submit_password))
                        .size_full()
                        .flex()
                        .items_center()
                        .justify_center()
                        .child(
                            v_flex()
                                .gap_2()
                                .items_center()
                                .w(px(280.0))
                                .child(Icon::new(IconName::Lock).size(IconSize::Medium))
                                .child(Headline::new("Password Required").size(HeadlineSize::Small))
                                .when_some(hint, |this, hint| {
                                    this.child(
                                        div()
                                            .text_center()
                                            .text_color(cx.theme().colors().text_muted)
                                            .child(hint),
                                    )
                                })
                                .child(
                                    div()
                                        .w_full()
                                        .px_2()
                                        .py_1()
                                        .rounded_md()
                                        .border_1()
                                        .border_color(cx.theme().colors().border)
                                        .bg(cx.theme().colors().editor_background)
                                        .child(self.password_input.clone()),
                                )
                                .child(Button::new("unlock-pdf", "Unlock").on_click(cx.listener(
                                    |this, _, window, cx| {
                                        this.submit_password(&menu::Confirm, window, cx);
                                    },
                                ))),
                        ),
                )
            })
            .when_some(load_error, |this, error| {
                this.child(
                    div()
                        .size_full()
                        .flex()
                        .items_center()
                        .justify_center()
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .items_center()
                                .gap_1()
                                .child("Failed to load PDF")
                                .child(
                                    div()
                                        .text_color(cx.theme().colors().text_muted)
                                        .child(error),
                                ),
                        ),
                )
            })
            .on_action(cx.listener(Self::toggle_outline))
            .on_action(cx.listener(Self::toggle_thumbnails))
            .on_action(cx.listener(Self::toggle_single_page_view))
            .on_action(cx.listener(Self::rotate_page_clockwise))
            .on_action(cx.listener(Self::open_in_default_app))
            .on_action(cx.listener(Self::toggle_inverted_colors))
            .on_action(cx.listener(Self::show_document_properties))
            .when(!has_load_error && !needs_password, |this| {
                let show_outline = self.show_outline;
                let show_thumbnails = self.show_thumbnails;
                let search_error = self.search_error.clone();
                let content = div()
                    .id("pdf-content-scroll-area")
                    .relative()
                    .flex_1()
                    .h_full()
                    .child(PdfContentElement::new(cx.entity()))
                    .when_some(search_error, |content, error| {
                        content.child(
                            div().absolute().top_2().left_2().right_2().child(
                                Banner::new()
                                    .severity(Severity::Error)
                                    .child(Label::new(error)),
                            ),
                        )
                    })
                    .custom_scrollbars(
                        ui::Scrollbars::new(ScrollAxes::Both)
                            .tracked_scroll_handle(&self.scroll_handle)
                            .with_track_along(
                                ScrollAxes::Both,
                                cx.theme().colors().panel_background,
                            ),
                        window,
                        cx,
                    );

                this.child(
                    h_flex()
                        .size_full()
                        .when(show_thumbnails, |row| {
                            row.child(self.render_thumbnail_sidebar(cx))
                        })
                        .when(show_outline, |row| {
                            row.child(self.render_outline_sidebar(cx))
                        })
                        .child(content),
                )
            })
    }
}

impl ProjectItem for PdfView {
    type Item = PdfItem;

    fn for_project_item(
        project: Entity<Project>,
        _: Option<&Pane>,
        item: Entity<Self::Item>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self
    where
        Self: Sized,
    {
        Self::new(item, project, window, cx)
    }

    fn for_broken_project_item(
        abs_path: &Path,
        is_local: bool,
        e: &Error,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<InvalidItemView>
    where
        Self: Sized,
    {
        Some(InvalidItemView::new(abs_path, is_local, e, window, cx))
    }
}

impl SearchableItem for PdfView {
    type Match = PdfSearchResult;

    fn supported_options(&self) -> SearchOptions {
        SearchOptions {
            case: true,
            word: true,
            regex: true,
            replacement: false,
            selection: false,
            select_all: false,
            find_in_results: false,
        }
    }

    fn get_matches(&self, _window: &mut Window, _cx: &mut App) -> (Vec<Self::Match>, SearchToken) {
        (
            self.search_matches.clone(),
            self.search_token.unwrap_or_default(),
        )
    }

    fn clear_matches(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        let had_matches = !self.search_matches.is_empty();
        self.search_matches.clear();
        self.active_match_index = None;
        if self.search_error.take().is_some() {
            cx.notify();
        }
        if had_matches {
            cx.emit(SearchEvent::MatchesInvalidated);
            cx.notify();
        }
    }

    fn update_matches(
        &mut self,
        matches: &[Self::Match],
        active_match_index: Option<usize>,
        token: SearchToken,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.search_matches = matches.to_vec();
        self.active_match_index = active_match_index;
        self.search_token = Some(token);
        cx.emit(SearchEvent::MatchesInvalidated);
        cx.notify();
    }

    fn query_suggestion(
        &mut self,
        _ignore_settings: Option<SeedQuerySetting>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> String {
        String::new()
    }

    fn activate_match(
        &mut self,
        index: usize,
        matches: &[Self::Match],
        _token: SearchToken,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.active_match_index = Some(index);
        if let Some(m) = matches.get(index)
            && let Some(first_bound) = m.bounds.first()
        {
            let target = PdfTarget {
                page_index: m.page_index,
                point: first_bound.origin, // jump_to_page handles the scaling
                block_bottom_y: None,
                line_height: None,
            };
            self.scroll_to_target(target, ScrollMode::Center, ScrollAnchor::Baseline, cx);
        }

        let next_match = matches.get(index + 1);
        let prev_match = matches.get(index.saturating_sub(1));

        for m in [next_match, prev_match].into_iter().flatten() {
            let pdf = self.pdf(cx);
            let page_idx = m.page_index;

            // `request_page` skips cached and in-flight pages. Center the strip on the match.
            let visible_center_y = m.bounds.first().map_or(0.0, |b| b.origin.y);
            // same HiDPI scale as `prepaint`, so prefetched pages are not blurrier.
            let render_scale = self.zoom_level * window.scale_factor();
            if let Some(task) = pdf.request_page(
                cx,
                page_idx,
                render_scale,
                visible_center_y,
                self.render_style(),
                gpui::Priority::Low,
                |_| {},
            ) {
                task.detach()
            }
        }

        cx.notify();
    }

    fn select_matches(
        &mut self,
        _matches: &[Self::Match],
        _token: SearchToken,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) {
    }
    fn replace(
        &mut self,
        _: &Self::Match,
        _: &SearchQuery,
        _token: SearchToken,
        _window: &mut Window,
        _: &mut Context<Self>,
    ) {
    }

    fn find_matches(
        &mut self,
        query: Arc<SearchQuery>,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Vec<Self::Match>> {
        let pdf = self.pdf(cx);
        let key = PdfSearchKey {
            query: query.as_str().to_string().into(),
            match_case: query.case_sensitive(),
            whole_word: query.whole_word(),
            regex: query.is_regex(),
        };

        if let Some((last_key, last_matches)) = &self.last_search_query
            && last_key == &key
        {
            return Task::ready(last_matches.clone());
        }

        cx.spawn(async move |this, cx| {
            // debounce the expensive search; typing again drops this task.
            cx.background_executor().timer(SEARCH_DEBOUNCE).await;

            let result = if key.regex {
                // PDFium only searches literal text, so regexes run over the text layer.
                for page_index in 0..pdf.metadata.page_count {
                    let task = cx.update(|cx| pdf.clone().request_text_spans(cx, page_index));
                    task.await;
                }
                let pages = (0..pdf.metadata.page_count)
                    .filter_map(|page_index| {
                        Some((page_index, pdf.text_spans_for_page(page_index)?))
                    })
                    .collect::<Vec<_>>();
                Ok(cx
                    .background_spawn(async move {
                        pages
                            .iter()
                            .flat_map(|(page_index, spans)| {
                                search_text_spans(&query, *page_index, spans)
                            })
                            .collect::<Vec<_>>()
                    })
                    .await)
            } else {
                cx.background_executor()
                    .spawn({
                        let key = key.clone();
                        async move {
                            crate::pdf_renderer::PdfRenderer::new(
                                pdf.id,
                                pdf.bytes.clone(),
                                pdf.password.clone(),
                            )
                            .search_document(
                                &key.query,
                                key.match_case,
                                key.whole_word,
                                pdf.metadata.clone(),
                            )
                            .await
                        }
                    })
                    .await
            };

            this.update(cx, |view, cx| match result {
                Ok(matches) => {
                    view.search_error = None;
                    view.last_search_query = Some((key, matches.clone()));
                    cx.notify();
                    matches
                }
                Err(error) => {
                    log::error!("PDF search failed: {error:?}");
                    // failures are not cached, so repeating the query retries the search.
                    view.search_error = Some(format!("Search failed: {error}").into());
                    cx.notify();
                    Vec::new()
                }
            })
            .unwrap_or_default()
        })
    }

    fn active_match_index(
        &mut self,
        direction: Direction,
        matches: &[Self::Match],
        _token: SearchToken,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<usize> {
        if matches.is_empty() {
            return None;
        }

        if let Some(active) = self.active_match_index {
            return Some(active);
        }

        // without an active match, pick the one closest to the viewport.
        match direction {
            Direction::Next => matches
                .iter()
                .position(|m| m.page_index >= self.current_page)
                .or(Some(0)),
            Direction::Prev => matches
                .iter()
                .rposition(|m| m.page_index <= self.current_page)
                .or(Some(matches.len().saturating_sub(1))),
        }
    }
}

pub struct PdfViewToolbarControls {
    pdf_view: Option<WeakEntity<PdfView>>,
    _subscription: Option<Subscription>,
    page_input: Option<Entity<Editor>>,
}

impl PdfViewToolbarControls {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            pdf_view: None,
            _subscription: None,
            page_input: None,
        }
    }
}

impl Render for PdfViewToolbarControls {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let Some(pdf_view) = self.pdf_view.as_ref().and_then(|v| v.upgrade()) else {
            return div().into_any_element();
        };
        let Some(page_input) = self.page_input.clone() else {
            return div().into_any_element();
        };

        let (zoom_level, current_page, page_count) = {
            let view = pdf_view.read(cx);
            (
                view.zoom_level,
                view.current_page,
                view.pdf(cx).metadata.page_count,
            )
        };

        if page_count == 0 {
            return div().into_any_element();
        }

        let max_digits = page_count.to_string().len().max(1);
        let input_width = px(10.0 + (max_digits as f32 * 8.0));

        let zoom_percentage: SharedString =
            format!("{}%", (zoom_level * 100.0).round() as i32).into();
        let page_text: SharedString = format!(" / {}", page_count).into();

        if !page_input.focus_handle(cx).is_focused(window) {
            let current_page_str = format!("{}", current_page + 1);
            page_input.update(cx, |editor, cx| {
                if editor.text(cx) != current_page_str {
                    editor.set_text(current_page_str, window, cx);
                }
            });
        }

        let show_outline = pdf_view.read(cx).show_outline;
        let show_thumbnails = pdf_view.read(cx).show_thumbnails;
        let is_single_page = pdf_view.read(cx).view_mode == ViewMode::SinglePage;
        let is_rotated = pdf_view.read(cx).rotation != PageRotation::None;
        let inverted_colors = pdf_view.read(cx).inverted_colors;

        h_flex()
            .w_full()
            .justify_end()
            .gap_1()
            .child(
                IconButton::new("document-properties", IconName::Info)
                    .icon_size(IconSize::Small)
                    .tooltip(|_window, cx| {
                        Tooltip::for_action("Document Properties", &ShowDocumentProperties, cx)
                    })
                    .on_click({
                        let pdf_view = pdf_view.downgrade();
                        move |_, window, cx| {
                            if let Some(view) = pdf_view.upgrade() {
                                view.update(cx, |this, cx| {
                                    this.show_document_properties(
                                        &ShowDocumentProperties,
                                        window,
                                        cx,
                                    )
                                });
                            }
                        }
                    }),
            )
            .child(
                IconButton::new("open-in-default-app", IconName::ArrowUpRight)
                    .icon_size(IconSize::Small)
                    .tooltip(|_window, cx| {
                        Tooltip::for_action("Open in Default App", &OpenInDefaultApp, cx)
                    })
                    .on_click({
                        let pdf_view = pdf_view.downgrade();
                        move |_, window, cx| {
                            if let Some(view) = pdf_view.upgrade() {
                                view.update(cx, |this, cx| {
                                    this.open_in_default_app(&OpenInDefaultApp, window, cx)
                                });
                            }
                        }
                    }),
            )
            .child(
                IconButton::new("toggle-inverted-colors", IconName::Reader)
                    .icon_size(IconSize::Small)
                    .toggle_state(inverted_colors)
                    .tooltip(|_window, cx| {
                        Tooltip::for_action("Invert Colors", &ToggleInvertedColors, cx)
                    })
                    .on_click({
                        let pdf_view = pdf_view.downgrade();
                        move |_, window, cx| {
                            if let Some(view) = pdf_view.upgrade() {
                                view.update(cx, |this, cx| {
                                    this.toggle_inverted_colors(&ToggleInvertedColors, window, cx)
                                });
                            }
                        }
                    }),
            )
            .child(
                IconButton::new("rotate-page-clockwise", IconName::RotateCw)
                    .icon_size(IconSize::Small)
                    .toggle_state(is_rotated)
                    .tooltip(|_window, cx| {
                        Tooltip::for_action("Rotate Clockwise", &RotatePageClockwise, cx)
                    })
                    .on_click({
                        let pdf_view = pdf_view.downgrade();
                        move |_, window, cx| {
                            if let Some(view) = pdf_view.upgrade() {
                                view.update(cx, |this, cx| {
                                    this.rotate_page_clockwise(&RotatePageClockwise, window, cx)
                                });
                            }
                        }
                    }),
            )
            .child(
                IconButton::new("toggle-single-page-view", IconName::FileDoc)
                    .icon_size(IconSize::Small)
                    .toggle_state(is_single_page)
                    .tooltip(|_window, cx| {
                        Tooltip::for_action("Toggle Single-Page View", &ToggleSinglePageView, cx)
                    })
                    .on_click({
                        let pdf_view = pdf_view.downgrade();
                        move |_, window, cx| {
                            if let Some(view) = pdf_view.upgrade() {
                                view.update(cx, |this, cx| {
                                    this.toggle_single_page_view(&ToggleSinglePageView, window, cx)
                                });
                            }
                        }
                    }),
            )
            .child(
                IconButton::new("toggle-thumbnails", IconName::Image)
                    .icon_size(IconSize::Small)
                    .toggle_state(show_thumbnails)
                    .tooltip(|_window, cx| {
                        Tooltip::for_action("Toggle Thumbnails", &ToggleThumbnails, cx)
                    })
                    .on_click({
                        let pdf_view = pdf_view.downgrade();
                        move |_, window, cx| {
                            if let Some(view) = pdf_view.upgrade() {
                                view.update(cx, |this, cx| {
                                    this.toggle_thumbnails(&ToggleThumbnails, window, cx)
                                });
                            }
                        }
                    }),
            )
            .child(
                IconButton::new("toggle-outline", IconName::ListTree)
                    .icon_size(IconSize::Small)
                    .toggle_state(show_outline)
                    .tooltip(|_window, cx| {
                        Tooltip::for_action("Toggle Outline", &ToggleOutline, cx)
                    })
                    .on_click({
                        let pdf_view = pdf_view.downgrade();
                        move |_, window, cx| {
                            if let Some(view) = pdf_view.upgrade() {
                                view.update(cx, |this, cx| {
                                    this.toggle_outline(&ToggleOutline, window, cx)
                                });
                            }
                        }
                    }),
            )
            .child(
                IconButton::new("prev-page", IconName::ChevronLeft)
                    .icon_size(IconSize::Small)
                    .disabled(current_page == 0)
                    .tooltip(|_window, cx| Tooltip::for_action("Previous Page", &PreviousPage, cx))
                    .on_click({
                        let pdf_view = pdf_view.downgrade();
                        move |_, window, cx| {
                            if let Some(view) = pdf_view.upgrade() {
                                view.update(cx, |this, cx| {
                                    this.previous_page(&PreviousPage, window, cx)
                                });
                            }
                        }
                    }),
            )
            .child(
                h_flex()
                    .gap_1()
                    .items_center()
                    .child(
                        div()
                            .w(input_width)
                            .flex()
                            .justify_end()
                            .text_size(px(12.0))
                            .track_focus(&page_input.focus_handle(cx))
                            .bg(cx.theme().colors().editor_background)
                            .border_1()
                            .border_color(cx.theme().colors().border)
                            .rounded_sm()
                            .px_1()
                            .on_key_down(cx.listener({
                                let pdf_view = pdf_view.downgrade();
                                move |this, event: &KeyDownEvent, window, cx| {
                                    if event.keystroke.key == "enter"
                                        && let Some(view) = pdf_view.upgrade()
                                        && let Some(input) = &this.page_input
                                        && let Ok(page) = input.read(cx).text(cx).parse::<usize>()
                                    {
                                        let target_page = page.saturating_sub(1);
                                        view.update(cx, |v, cx| {
                                            v.jump_to_page(target_page, cx);
                                        });
                                        window.focus(&view.focus_handle(cx), cx);
                                    }
                                }
                            }))
                            .child(page_input),
                    )
                    .child(
                        Label::new(page_text), //.size(LabelSize::Small)
                    ),
            )
            .child(
                IconButton::new("next-page", IconName::ChevronRight)
                    .icon_size(IconSize::Small)
                    .disabled(current_page + 1 >= page_count)
                    .tooltip(|_window, cx| Tooltip::for_action("Next Page", &NextPage, cx))
                    .on_click({
                        let pdf_view = pdf_view.downgrade();
                        move |_, window, cx| {
                            if let Some(view) = pdf_view.upgrade() {
                                view.update(cx, |this, cx| this.next_page(&NextPage, window, cx));
                            }
                        }
                    }),
            )
            .gap_1()
            .child(
                IconButton::new("zoom-out", IconName::Dash)
                    .icon_size(IconSize::Small)
                    .tooltip(|_window, cx| Tooltip::for_action("Zoom Out", &ZoomOut, cx))
                    .on_click({
                        let pdf_view = pdf_view.downgrade();
                        move |_, window, cx| {
                            if let Some(view) = pdf_view.upgrade() {
                                view.update(cx, |this, cx| {
                                    this.zoom_out(&ZoomOut, window, cx);
                                });
                            }
                        }
                    }),
            )
            .child(
                Button::new("zoom-level", zoom_percentage)
                    .label_size(LabelSize::Small)
                    .tooltip(|_window, cx| Tooltip::for_action("Reset Zoom", &ResetZoom, cx))
                    .on_click({
                        let pdf_view = pdf_view.downgrade();
                        move |_, window, cx| {
                            if let Some(view) = pdf_view.upgrade() {
                                view.update(cx, |this, cx| {
                                    this.reset_zoom(&ResetZoom, window, cx);
                                });
                            }
                        }
                    }),
            )
            .child(
                IconButton::new("zoom-in", IconName::Plus)
                    .icon_size(IconSize::Small)
                    .tooltip(|_window, cx| Tooltip::for_action("Zoom In", &ZoomIn, cx))
                    .on_click({
                        let pdf_view = pdf_view.downgrade();
                        move |_, window, cx| {
                            if let Some(view) = pdf_view.upgrade() {
                                view.update(cx, |this, cx| {
                                    this.zoom_in(&ZoomIn, window, cx);
                                });
                            }
                        }
                    }),
            )
            .child(
                IconButton::new("fit-to-width", IconName::ArrowRightLeft)
                    .icon_size(IconSize::Small)
                    .tooltip(|_window, cx| Tooltip::for_action("Fit to Width", &FitToWidth, cx))
                    .on_click({
                        let pdf_view = pdf_view.downgrade();
                        move |_, window, cx| {
                            if let Some(view) = pdf_view.upgrade() {
                                view.update(cx, |this, cx| {
                                    this.fit_to_width(&FitToWidth, window, cx);
                                });
                            }
                        }
                    }),
            )
            .child(
                IconButton::new("fit-to-view", IconName::Maximize)
                    .icon_size(IconSize::Small)
                    .tooltip(|_window, cx| Tooltip::for_action("Fit to View", &FitToView, cx))
                    .on_click({
                        let pdf_view = pdf_view.downgrade();
                        move |_, window, cx| {
                            if let Some(view) = pdf_view.upgrade() {
                                view.update(cx, |this, cx| {
                                    this.fit_to_view(&FitToView, window, cx);
                                });
                            }
                        }
                    }),
            )
            .into_any_element()
    }
}

impl EventEmitter<ToolbarItemEvent> for PdfViewToolbarControls {}
impl EventEmitter<SearchEvent> for PdfView {}

impl ToolbarItemView for PdfViewToolbarControls {
    fn set_active_pane_item(
        &mut self,
        active_pane_item: Option<&dyn ItemHandle>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> ToolbarItemLocation {
        self.pdf_view = None;
        self._subscription = None;

        if let Some(item) = active_pane_item.and_then(|i| i.act_as::<PdfView>(cx)) {
            if self.page_input.is_none() {
                self.page_input = Some(cx.new(|cx| Editor::single_line(window, cx)));
            }

            self._subscription = Some(cx.observe(&item, |_, _, cx| {
                cx.notify();
            }));
            self.pdf_view = Some(item.downgrade());
            cx.notify();
            return ToolbarItemLocation::PrimaryRight;
        }

        ToolbarItemLocation::Hidden
    }
}

pub fn init(cx: &mut App) {
    workspace::register_project_item::<PdfView>(cx);
    workspace::register_serializable_item::<PdfView>(cx);
}

mod persistence {
    use std::path::PathBuf;

    use db::{
        query,
        sqlez::{domain::Domain, thread_safe_connection::ThreadSafeConnection},
        sqlez_macros::sql,
    };
    use workspace::{ItemId, WorkspaceDb, WorkspaceId};

    pub struct PdfViewerDb(ThreadSafeConnection);

    impl Domain for PdfViewerDb {
        const NAME: &str = stringify!(PdfViewerDb);

        const MIGRATIONS: &[&str] = &[
            sql!(
                CREATE TABLE pdf_viewers (
                    workspace_id INTEGER,
                    item_id INTEGER UNIQUE,

                    pdf_path BLOB,

                    PRIMARY KEY(workspace_id, item_id),
                    FOREIGN KEY(workspace_id) REFERENCES workspaces(workspace_id)
                    ON DELETE CASCADE
                ) STRICT;
            ),
            sql!(
                ALTER TABLE pdf_viewers ADD COLUMN zoom_level REAL;
                ALTER TABLE pdf_viewers ADD COLUMN current_page INTEGER;
            ),
            sql!(
                ALTER TABLE pdf_viewers ADD COLUMN page_offset_x REAL;
                ALTER TABLE pdf_viewers ADD COLUMN page_offset_y REAL;
            ),
            sql!(
                ALTER TABLE pdf_viewers ADD COLUMN inverted_colors INTEGER;
            ),
        ];
    }

    db::static_connection!(PdfViewerDb, [WorkspaceDb]);

    impl PdfViewerDb {
        // store the path and view state in one statement, since `INSERT OR REPLACE` would wipe
        // columns written by a separate query.
        query! {
            pub async fn save_pdf_view_state(
                item_id: ItemId,
                workspace_id: WorkspaceId,
                pdf_path: PathBuf,
                zoom_level_and_page: (f64, i64),
                page_offset: Option<(f64, f64)>,
                inverted_colors: bool
            ) -> Result<()> {
                INSERT OR REPLACE INTO pdf_viewers
                    (item_id, workspace_id, pdf_path, zoom_level, current_page, page_offset_x, page_offset_y, inverted_colors)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            }
        }

        query! {
            pub fn get_pdf_path(item_id: ItemId, workspace_id: WorkspaceId) -> Result<Option<PathBuf>> {
                SELECT pdf_path
                FROM pdf_viewers
                WHERE item_id = ? AND workspace_id = ?
            }
        }

        // returns `(zoom_level, current_page, (page_offset_x, page_offset_y), inverted_colors)` if
        // saved.
        query! {
            pub fn get_pdf_view_state(item_id: ItemId, workspace_id: WorkspaceId) -> Result<Option<(f64, i64, Option<(f64, f64)>, Option<bool>)>> {
                SELECT zoom_level, current_page, page_offset_x, page_offset_y, inverted_colors
                FROM pdf_viewers
                WHERE item_id = ? AND workspace_id = ?
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(text: &str, x: f32, y: f32, w: f32, h: f32) -> PdfTextSpan {
        PdfTextSpan {
            text: text.to_string(),
            bounds: Bounds {
                origin: Point { x, y },
                size: Size {
                    width: w,
                    height: h,
                },
            },
        }
    }

    // two lines, `Hello` and `World`, of 10x10 characters at y=0 and y=10.
    fn two_line_fixture() -> Vec<PdfTextSpan> {
        "Hello"
            .chars()
            .enumerate()
            .map(|(i, c)| span(&c.to_string(), i as f32 * 10.0, 0.0, 10.0, 10.0))
            .chain(
                "World"
                    .chars()
                    .enumerate()
                    .map(|(i, c)| span(&c.to_string(), i as f32 * 10.0, 10.0, 10.0, 10.0)),
            )
            .collect()
    }

    #[test]
    fn nearest_char_index_finds_the_closest_span() {
        let spans = two_line_fixture();

        // center of `e` in `Hello`.
        assert_eq!(
            nearest_char_index(&spans, Point { x: 15.0, y: 5.0 }),
            Some(1)
        );
        // center of the last `o` in `World`.
        assert_eq!(
            nearest_char_index(&spans, Point { x: 45.0, y: 15.0 }),
            Some(9)
        );
        // points far outside the page still resolve.
        assert_eq!(
            nearest_char_index(
                &spans,
                Point {
                    x: -1000.0,
                    y: -1000.0
                }
            ),
            Some(0)
        );

        assert_eq!(nearest_char_index(&[], Point { x: 0.0, y: 0.0 }), None);
    }

    // one line, `Hi, Bob`: a word, punctuation, whitespace, and another word.
    fn word_boundary_fixture() -> Vec<PdfTextSpan> {
        "Hi, Bob"
            .chars()
            .enumerate()
            .map(|(i, c)| span(&c.to_string(), i as f32 * 10.0, 0.0, 10.0, 10.0))
            .collect()
    }

    #[test]
    fn word_range_at_stops_at_class_boundaries() {
        let spans = word_boundary_fixture();

        // any character of `Hi` selects the word.
        assert_eq!(word_range_at(&spans, 0), 0..2);
        assert_eq!(word_range_at(&spans, 1), 0..2);
        // the comma forms its own run.
        assert_eq!(word_range_at(&spans, 2), 2..3);
        // the space forms its own run.
        assert_eq!(word_range_at(&spans, 3), 3..4);
        // any character of `Bob` selects the word.
        assert_eq!(word_range_at(&spans, 4), 4..7);
        assert_eq!(word_range_at(&spans, 6), 4..7);

        assert_eq!(word_range_at(&[], 0), 0..0);
    }

    #[test]
    fn line_range_at_stops_at_line_boundaries() {
        let spans = two_line_fixture();

        // the first line.
        assert_eq!(line_range_at(&spans, 0), 0..5);
        assert_eq!(line_range_at(&spans, 4), 0..5);
        // the second line.
        assert_eq!(line_range_at(&spans, 5), 5..10);
        assert_eq!(line_range_at(&spans, 9), 5..10);

        assert_eq!(line_range_at(&[], 0), 0..0);
    }

    #[test]
    fn select_all_range_spans_first_to_last_page_regardless_of_input_order() {
        // unordered pages with a gap still bound on the lowest and highest page with text.
        let pages_with_text = vec![(4, 3), (0, 10)];

        let selection = select_all_range(&pages_with_text).expect("pages have text");
        assert_eq!(
            selection.anchor,
            TextPosition {
                page_index: 0,
                char_index: 0,
            }
        );
        // the last character index is inclusive.
        assert_eq!(
            selection.head,
            TextPosition {
                page_index: 4,
                char_index: 2,
            }
        );
    }

    #[test]
    fn select_all_range_is_none_with_no_pages() {
        assert!(select_all_range(&[]).is_none());
    }

    #[test]
    fn selected_char_range_for_page_covers_the_right_pages_and_chars() {
        let selection = TextSelection {
            anchor: TextPosition {
                page_index: 1,
                char_index: 3,
            },
            head: TextPosition {
                page_index: 2,
                char_index: 1,
            },
        };

        // before the first page.
        assert_eq!(selected_char_range_for_page(&selection, 0, 10), None);
        // first page, from the anchor to the end.
        assert_eq!(selected_char_range_for_page(&selection, 1, 10), Some(3..10));
        // last page, up to and including the head.
        assert_eq!(selected_char_range_for_page(&selection, 2, 10), Some(0..2));
        // after the last page.
        assert_eq!(selected_char_range_for_page(&selection, 3, 10), None);
        // page without text.
        assert_eq!(selected_char_range_for_page(&selection, 1, 0), None);
    }

    #[test]
    fn selected_char_range_for_page_is_direction_independent() {
        // a backward drag selects the same range as a forward drag.
        let forward = TextSelection {
            anchor: TextPosition {
                page_index: 0,
                char_index: 1,
            },
            head: TextPosition {
                page_index: 0,
                char_index: 4,
            },
        };
        let backward = TextSelection {
            anchor: TextPosition {
                page_index: 0,
                char_index: 4,
            },
            head: TextPosition {
                page_index: 0,
                char_index: 1,
            },
        };
        assert_eq!(
            selected_char_range_for_page(&forward, 0, 10),
            selected_char_range_for_page(&backward, 0, 10)
        );
    }

    #[test]
    fn single_line_selection_stays_one_rect() {
        let spans = two_line_fixture();
        // one line collapses to one rectangle.
        let rects = selection_highlight_rects(&spans, 0..5);
        assert_eq!(rects.len(), 1);
        assert_eq!(rects[0].origin.x, 0.0);
        assert_eq!(rects[0].size.width, 50.0);
    }

    #[test]
    fn multi_line_selection_produces_one_rect_per_line() {
        let spans = two_line_fixture();
        // two lines yield two rectangles.
        let rects = selection_highlight_rects(&spans, 2..8);
        assert_eq!(rects.len(), 2);
    }

    #[test]
    fn regex_search_maps_matches_to_one_rect_per_line() {
        // "foo bar" on one line and "baz" on the next, with a generated line break between.
        let spans = [
            span("f", 0.0, 0.0, 10.0, 10.0),
            span("o", 10.0, 0.0, 10.0, 10.0),
            span("o", 20.0, 0.0, 10.0, 10.0),
            span(" ", 30.0, 0.0, 10.0, 10.0),
            span("b", 40.0, 0.0, 10.0, 10.0),
            span("a", 50.0, 0.0, 10.0, 10.0),
            span("r", 60.0, 0.0, 10.0, 10.0),
            span("\n", 0.0, 0.0, 0.0, 0.0),
            span("b", 0.0, 20.0, 10.0, 10.0),
            span("a", 10.0, 20.0, 10.0, 10.0),
            span("z", 20.0, 20.0, 10.0, 10.0),
        ];
        let query = |pattern: &str| {
            SearchQuery::regex(
                pattern,
                false,
                false,
                false,
                false,
                Default::default(),
                Default::default(),
                false,
                None,
            )
            .unwrap()
        };

        let matches = search_text_spans(&query("ba[rz]"), 3, &spans);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].page_index, 3);
        assert_eq!(
            matches[0].bounds,
            vec![span("", 40.0, 0.0, 30.0, 10.0).bounds]
        );
        assert_eq!(
            matches[1].bounds,
            vec![span("", 0.0, 20.0, 30.0, 10.0).bounds]
        );

        let across_lines = search_text_spans(&query("bar\\sbaz"), 0, &spans);
        assert_eq!(
            across_lines[0].bounds,
            vec![
                span("", 40.0, 0.0, 30.0, 10.0).bounds,
                span("", 0.0, 20.0, 30.0, 10.0).bounds,
            ]
        );
    }

    #[test]
    fn selection_to_text_joins_pages_with_newlines_in_document_order() {
        let page0 = Arc::new(two_line_fixture()); // "HelloWorld" as one span list
        let page1 = Arc::new(vec![span("!", 0.0, 0.0, 10.0, 10.0)]);

        let selection = TextSelection {
            anchor: TextPosition {
                page_index: 0,
                char_index: 0,
            },
            head: TextPosition {
                page_index: 1,
                char_index: 0,
            },
        };

        let text = selection_to_text(&selection, &[(0, page0), (1, page1)]);
        assert_eq!(text, "HelloWorld\n!");
    }

    #[test]
    fn hit_test_selection_point_falls_back_to_nearest_region_outside_all_of_them() {
        let page0 = Arc::new(two_line_fixture());
        let regions = vec![SelectionHitRegion {
            page_index: 0,
            bounds: Bounds {
                origin: Point {
                    x: px(0.0),
                    y: px(0.0),
                },
                size: Size {
                    width: px(100.0),
                    height: px(100.0),
                },
            },
            text_spans: page0,
            page_size: Size {
                width: 100.0,
                height: 100.0,
            },
        }];

        // inside the page bounds.
        let inside = hit_test_selection_point(
            &regions,
            Point {
                x: px(15.0),
                y: px(5.0),
            },
            1.0,
            PageRotation::None,
        );
        assert_eq!(
            inside,
            Some(TextPosition {
                page_index: 0,
                char_index: 1
            })
        );

        // below the last page, via the nearest-region fallback.
        let below = hit_test_selection_point(
            &regions,
            Point {
                x: px(15.0),
                y: px(10_000.0),
            },
            1.0,
            PageRotation::None,
        );
        assert!(below.is_some());

        assert_eq!(
            hit_test_selection_point(
                &[],
                Point {
                    x: px(0.0),
                    y: px(0.0)
                },
                1.0,
                PageRotation::None,
            ),
            None
        );
    }

    #[test]
    fn outline_nodes_default_expanded_at_the_top_level_only() {
        let state = OutlineExpansionState::default();

        // top-level nodes default to expanded and deeper nodes to collapsed.
        assert!(state.is_expanded(&[0]));
        assert!(state.is_expanded(&[3]));
        assert!(!state.is_expanded(&[0, 0]));
        assert!(!state.is_expanded(&[0, 1, 2]));
    }

    #[test]
    fn toggling_an_outline_node_flips_its_expanded_state_independent_of_its_default() {
        let mut state = OutlineExpansionState::default();

        // collapse a top-level node.
        assert!(state.is_expanded(&[0]));
        state.toggle(vec![0]);
        assert!(!state.is_expanded(&[0]));
        state.toggle(vec![0]);
        assert!(state.is_expanded(&[0]));

        // expand a nested node without affecting its sibling.
        assert!(!state.is_expanded(&[0, 0]));
        assert!(!state.is_expanded(&[0, 1]));
        state.toggle(vec![0, 0]);
        assert!(state.is_expanded(&[0, 0]));
        assert!(!state.is_expanded(&[0, 1]));
    }

    fn init_test(cx: &mut gpui::TestAppContext) {
        cx.update(|cx| {
            let store = settings::SettingsStore::test(cx);
            cx.set_global(store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn inverting_colors_re_renders_pages_in_the_new_style(cx: &mut gpui::TestAppContext) {
        init_test(cx);
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/solid_red.pdf");
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);
        let (view, cx) =
            cx.add_window_view(|window, cx| PdfView::from_pdf(pdf.clone(), window, cx));
        view.update_in(cx, |view, _window, cx| view.restore_view_state(1.0, 0, cx));
        cx.run_until_parked();
        let rendered_inverted = || {
            pdf.page_cache
                .read()
                .get(&0)
                .map(|data| data.style.inverted)
        };
        assert_eq!(rendered_inverted(), Some(false));

        view.update_in(cx, |view, window, cx| {
            view.toggle_inverted_colors(&ToggleInvertedColors, window, cx)
        });
        cx.run_until_parked();
        assert_eq!(rendered_inverted(), Some(true));
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn regex_search_runs_over_the_text_layer(cx: &mut gpui::TestAppContext) {
        init_test(cx);
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/whole_words.pdf");
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);
        let (view, cx) =
            cx.add_window_view(|window, cx| PdfView::from_pdf(pdf.clone(), window, cx));

        let search = |pattern: &str, whole_word: bool, cx: &mut gpui::VisualTestContext| {
            let query = SearchQuery::regex(
                pattern,
                whole_word,
                false,
                false,
                false,
                Default::default(),
                Default::default(),
                false,
                None,
            )
            .unwrap();
            let task = view.update_in(cx, |view, window, cx| {
                view.find_matches(Arc::new(query), window, cx)
            });
            cx.executor().advance_clock(SEARCH_DEBOUNCE);
            task
        };

        let matches = search("c\\w+", false, cx).await;
        assert_eq!(matches.len(), 3);
        assert!(matches.iter().all(|found| found.page_index == 0));
        let matches = search("ca.", true, cx).await;
        assert_eq!(
            matches.len(),
            2,
            "whole words skip the one inside concatenate"
        );
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn first_and_last_page_actions_jump_to_the_document_ends(cx: &mut gpui::TestAppContext) {
        init_test(cx);
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_fixtures/two_pages_with_text.pdf"
        );
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);
        let (view, cx) =
            cx.add_window_view(|window, cx| PdfView::from_pdf(pdf.clone(), window, cx));
        view.update_in(cx, |view, _window, cx| view.restore_view_state(1.0, 0, cx));
        cx.run_until_parked();

        view.update_in(cx, |view, window, cx| view.last_page(&LastPage, window, cx));
        cx.run_until_parked();
        assert_eq!(view.read_with(cx, |view, _| view.current_page), 1);

        view.update_in(cx, |view, window, cx| {
            view.first_page(&FirstPage, window, cx)
        });
        cx.run_until_parked();
        assert_eq!(view.read_with(cx, |view, _| view.current_page), 0);
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn document_properties_open_in_a_modal(cx: &mut gpui::TestAppContext) {
        init_test(cx);
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/solid_red.pdf");
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);
        assert_eq!(
            crate::document_properties::property_rows(&pdf, 0),
            vec![
                ("Title", "untitled".to_string()),
                ("Author", "anonymous".to_string()),
                ("Subject", "unspecified".to_string()),
                ("Created", "2026-08-09 17:51:06 +02:00".to_string()),
                ("Application", "anonymous".to_string()),
                (
                    "PDF Producer",
                    "ReportLab PDF Library - (opensource)".to_string()
                ),
                ("PDF Version", "1.4".to_string()),
                ("Pages", "1".to_string()),
                ("Page Size", "35 × 35 mm".to_string()),
                (
                    "File Size",
                    util::size::format_file_size(pdf.bytes.len() as u64, true)
                ),
            ]
        );

        let fs = project::FakeFs::new(cx.executor());
        let project = Project::test(fs, [], cx).await;
        let window = cx.add_window(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let cx = &mut gpui::VisualTestContext::from_window(*window, cx);
        let workspace = window
            .read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone())
            .unwrap();
        let view = cx.new_window_entity(|window, cx| PdfView::from_pdf(pdf.clone(), window, cx));

        view.update_in(cx, |view, window, cx| {
            view.show_document_properties(&ShowDocumentProperties, window, cx)
        });
        cx.run_until_parked();
        workspace.read_with(cx, |workspace, cx| {
            assert!(
                workspace
                    .active_modal::<DocumentPropertiesModal>(cx)
                    .is_some()
            );
        });
    }

    // only test bodies take `PDFIUM_TEST_LOCK`, so holding it across an await cannot deadlock.
    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn restore_view_state_sets_zoom_and_page_and_suppresses_auto_fit(
        cx: &mut gpui::TestAppContext,
    ) {
        init_test(cx);
        // real PDFium work on a background task needs to block a thread.
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/solid_red.pdf");
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);

        let (view, cx) = cx.add_window_view(|window, cx| PdfView::from_pdf(pdf, window, cx));

        view.update(cx, |view, cx| {
            assert!(!view.suppress_initial_fit_to_view);
            // restored zoom levels are clamped like any other.
            view.restore_view_state(MAX_ZOOM * 5.0, 0, cx);

            assert_eq!(view.zoom_level, MAX_ZOOM);
            assert_eq!(view.current_page, 0);
            assert!(view.suppress_initial_fit_to_view);
        });
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn toggling_single_page_view_flips_the_mode_and_back(cx: &mut gpui::TestAppContext) {
        init_test(cx);
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/solid_red.pdf");
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);

        let (view, cx) = cx.add_window_view(|window, cx| PdfView::from_pdf(pdf, window, cx));

        view.update_in(cx, |view, window, cx| {
            assert_eq!(view.view_mode, ViewMode::Continuous);

            view.toggle_single_page_view(&ToggleSinglePageView, window, cx);
            assert_eq!(view.view_mode, ViewMode::SinglePage);

            view.toggle_single_page_view(&ToggleSinglePageView, window, cx);
            assert_eq!(view.view_mode, ViewMode::Continuous);
        });
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn rotating_clockwise_four_times_returns_to_unrotated_and_keeps_selection(
        cx: &mut gpui::TestAppContext,
    ) {
        init_test(cx);
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/solid_red.pdf");
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);

        let (view, cx) = cx.add_window_view(|window, cx| PdfView::from_pdf(pdf, window, cx));

        view.update_in(cx, |view, window, cx| {
            assert_eq!(view.rotation, PageRotation::None);

            // the selection is in page coordinates, so it survives rotation.
            view.text_selection = Some(TextSelection {
                anchor: TextPosition {
                    page_index: 0,
                    char_index: 0,
                },
                head: TextPosition {
                    page_index: 0,
                    char_index: 0,
                },
            });
            view.is_selecting_text = true;

            view.rotate_page_clockwise(&RotatePageClockwise, window, cx);
            assert_eq!(view.rotation, PageRotation::Clockwise90);
            assert!(view.text_selection.is_some());

            view.rotate_page_clockwise(&RotatePageClockwise, window, cx);
            assert_eq!(view.rotation, PageRotation::Clockwise180);

            view.rotate_page_clockwise(&RotatePageClockwise, window, cx);
            assert_eq!(view.rotation, PageRotation::Clockwise270);

            view.rotate_page_clockwise(&RotatePageClockwise, window, cx);
            assert_eq!(view.rotation, PageRotation::None);
        });
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn select_all_extracts_text_for_every_page_not_just_rendered_ones(
        cx: &mut gpui::TestAppContext,
    ) {
        init_test(cx);
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_fixtures/two_pages_with_text.pdf"
        );
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);
        assert_eq!(pdf.metadata.page_count, 2);

        let (view, cx) =
            cx.add_window_view(|window, cx| PdfView::from_pdf(pdf.clone(), window, cx));

        let task = view.update_in(cx, |view, window, cx| {
            view.select_all(&SelectAll, window, cx);
            view.select_all_task.take()
        });
        task.expect("select_all should have spawned a task").await;

        for page_index in 0..pdf.metadata.page_count {
            assert!(
                pdf.text_spans_for_page(page_index)
                    .is_some_and(|spans| !spans.is_empty()),
                "expected select_all to have extracted real text for page {page_index}"
            );
        }

        view.update(cx, |view, _| {
            let selection = view
                .text_selection
                .as_ref()
                .expect("select_all should have produced a selection");
            let (start, end) = selection.ordered();
            assert_eq!(start.page_index, 0);
            assert_eq!(
                end.page_index, 1,
                "expected the selection to cover the whole two-page document, not just \
                 whatever page(s) happened to already be rendered on initial layout"
            );
        });
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn in_memory_documents_are_written_to_a_file_for_the_default_app(
        cx: &mut gpui::TestAppContext,
    ) {
        init_test(cx);
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/solid_red.pdf");
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes.clone(), None, None).await);
        let expected_path = std::env::temp_dir().join(format!("zed-pdf-{:016x}.pdf", pdf.id));
        std::fs::remove_file(&expected_path).ok();

        let (view, cx) = cx.add_window_view(|window, cx| PdfView::from_pdf(pdf, window, cx));
        let written_path = view
            .update(cx, |view, cx| view.file_path_for_system(cx))
            .await
            .unwrap();

        assert_eq!(written_path, expected_path);
        assert_eq!(std::fs::read(&expected_path).unwrap(), bytes);
        std::fs::remove_file(&expected_path).ok();
    }

    #[gpui::test]
    async fn view_state_round_trips_through_the_database(cx: &mut gpui::TestAppContext) {
        init_test(cx);
        let workspace_db = cx.update(|cx| workspace::WorkspaceDb::global(cx));
        let workspace_id = workspace_db.next_id().await.unwrap();
        let db = cx.update(|cx| PdfViewerDb::global(cx));
        let path = std::path::PathBuf::from("/tmp/document.pdf");

        db.save_pdf_view_state(
            1,
            workspace_id,
            path.clone(),
            (1.5, 3),
            Some((12.0, -4.0)),
            true,
        )
        .await
        .unwrap();
        assert_eq!(
            db.get_pdf_path(1, workspace_id).unwrap(),
            Some(path.clone())
        );
        assert_eq!(
            db.get_pdf_view_state(1, workspace_id).unwrap(),
            Some((1.5, 3, Some((12.0, -4.0)), Some(true)))
        );

        // views saved while rotated store no offset.
        db.save_pdf_view_state(1, workspace_id, path, (1.5, 3), None, false)
            .await
            .unwrap();
        assert_eq!(
            db.get_pdf_view_state(1, workspace_id).unwrap(),
            Some((1.5, 3, None, Some(false)))
        );
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn scroll_position_in_page_survives_a_zoom_change(cx: &mut gpui::TestAppContext) {
        init_test(cx);
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_fixtures/two_pages_with_text.pdf"
        );
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);
        let (view, cx) =
            cx.add_window_view(|window, cx| PdfView::from_pdf(pdf.clone(), window, cx));
        cx.run_until_parked();

        let position = Point { x: 12.0, y: 150.0 };
        view.update_in(cx, |view, _window, cx| {
            view.restore_view_state(2.0, 1, cx);
            view.scroll_to_position_in_page(1, position, cx);
            assert_eq!(view.current_page, 1);
            let saved = view.scroll_position_in_page(cx);
            assert!((saved.x - position.x).abs() < 0.01 && (saved.y - position.y).abs() < 0.01);

            // a restore at another zoom level lands on the same spot of the page.
            view.restore_view_state(3.0, 1, cx);
            view.scroll_to_position_in_page(1, saved, cx);
            let restored = view.scroll_position_in_page(cx);
            assert!(
                (restored.x - position.x).abs() < 0.01 && (restored.y - position.y).abs() < 0.01,
                "{restored:?}"
            );
            let page_size = view.display_page_size(&pdf, 0);
            let expected_scroll_y = page_size.height * 3.0 + PAGE_GAP + position.y * 3.0;
            assert!((-view.scroll_handle.offset().y.as_f32() - expected_scroll_y).abs() < 0.01);
        });
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn double_click_selects_the_same_word_on_rotated_pages(cx: &mut gpui::TestAppContext) {
        init_test(cx);
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_fixtures/two_pages_with_text.pdf"
        );
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);
        let page_size = pdf.get_page_size(0).unwrap();

        let (view, cx) =
            cx.add_window_view(|window, cx| PdfView::from_pdf(pdf.clone(), window, cx));
        view.update_in(cx, |view, _window, cx| view.restore_view_state(1.0, 0, cx));
        cx.run_until_parked();
        cx.run_until_parked();

        let text_spans = pdf
            .text_spans_for_page(0)
            .expect("page 0 should have rendered with real text by now");
        let (target_index, expected_range) = (0..text_spans.len())
            .map(|i| (i, word_range_at(&text_spans, i)))
            .find(|(i, range)| {
                range.len() > 1 && span_char_class(&text_spans[*i]) == CharClass::Word
            })
            .expect("fixture should contain at least one real word");
        let target_center = text_spans[target_index].bounds.center();

        for rotations in 1..=3 {
            view.update_in(cx, |view, window, cx| {
                view.text_selection = None;
                view.rotate_page_clockwise(&RotatePageClockwise, window, cx);
            });
            cx.run_until_parked();
            cx.run_until_parked();

            let rotation = view.read_with(cx, |view, _| view.rotation);
            let page_bounds = cx
                .debug_bounds("pdf-page-0")
                .unwrap_or_else(|| panic!("page 0 should be laid out after {rotations} turns"));
            let display_size = rotation.display_size(page_size);
            assert!(
                (f32::from(page_bounds.size.width) - display_size.width).abs() < 2.0,
                "the page box should match the rotated page size"
            );
            let display_center = rotation.to_display_point(target_center, page_size);
            let click_point = Point {
                x: page_bounds.origin.x + px(display_center.x),
                y: page_bounds.origin.y + px(display_center.y),
            };
            cx.simulate_event(gpui::MouseDownEvent {
                position: click_point,
                modifiers: gpui::Modifiers::none(),
                button: MouseButton::Left,
                click_count: 2,
                first_mouse: false,
            });
            cx.simulate_event(gpui::MouseUpEvent {
                position: click_point,
                modifiers: gpui::Modifiers::none(),
                button: MouseButton::Left,
                click_count: 2,
            });

            view.update(cx, |view, _cx| {
                let selection = view.text_selection.as_ref().unwrap_or_else(|| {
                    panic!("double-click after {rotations} turns should select a word")
                });
                let (start, end) = selection.ordered();
                assert_eq!(
                    (start.page_index, start.char_index, end.char_index),
                    (0, expected_range.start, expected_range.end - 1),
                    "double-click after {rotations} turns selected the wrong word"
                );
            });
        }
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn double_click_selects_the_same_word_at_different_zoom_levels(
        cx: &mut gpui::TestAppContext,
    ) {
        init_test(cx);
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_fixtures/two_pages_with_text.pdf"
        );
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);

        let (view, cx) =
            cx.add_window_view(|window, cx| PdfView::from_pdf(pdf.clone(), window, cx));
        // let the background render finish and trigger a second paint with the cached page.
        cx.run_until_parked();
        cx.run_until_parked();

        let text_spans = pdf
            .text_spans_for_page(0)
            .expect("page 0 should have rendered with real text by now");
        // a multi-character word, so selecting the wrong word is detectable.
        let (target_index, expected_range) = (0..text_spans.len())
            .map(|i| (i, word_range_at(&text_spans, i)))
            .find(|(i, range)| {
                range.len() > 1 && span_char_class(&text_spans[*i]) == CharClass::Word
            })
            .expect("fixture should contain at least one real word");
        let target_center = {
            let bounds = text_spans[target_index].bounds;
            Point {
                x: bounds.origin.x + bounds.size.width / 2.0,
                y: bounds.origin.y + bounds.size.height / 2.0,
            }
        };

        for zoom in [1.0_f32, 5.0_f32, 15.0_f32] {
            view.update_in(cx, |view, _window, cx| {
                view.restore_view_state(zoom, 0, cx);
            });
            cx.run_until_parked();
            cx.run_until_parked();

            let page_bounds = cx
                .debug_bounds("pdf-page-0")
                .unwrap_or_else(|| panic!("page 0 should be laid out at zoom {zoom}"));
            let click_point = Point {
                x: page_bounds.origin.x + px(target_center.x * zoom),
                y: page_bounds.origin.y + px(target_center.y * zoom),
            };

            cx.simulate_event(gpui::MouseDownEvent {
                position: click_point,
                modifiers: gpui::Modifiers::none(),
                button: MouseButton::Left,
                click_count: 2,
                first_mouse: false,
            });
            cx.simulate_event(gpui::MouseUpEvent {
                position: click_point,
                modifiers: gpui::Modifiers::none(),
                button: MouseButton::Left,
                click_count: 2,
            });

            view.update(cx, |view, _cx| {
                let selection = view
                    .text_selection
                    .as_ref()
                    .unwrap_or_else(|| panic!("double-click at zoom {zoom} should select a word"));
                let (start, end) = selection.ordered();
                assert_eq!(
                    (start.page_index, start.char_index, end.char_index),
                    (0, expected_range.start, expected_range.end - 1),
                    "double-click at zoom {zoom} selected the wrong word"
                );
            });
        }
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn double_click_selects_the_right_word_even_when_the_page_is_strip_rendered(
        cx: &mut gpui::TestAppContext,
    ) {
        init_test(cx);
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_fixtures/tall_page_with_text.pdf"
        );
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);

        let (view, cx) =
            cx.add_window_view(|window, cx| PdfView::from_pdf(pdf.clone(), window, cx));
        cx.run_until_parked();
        cx.run_until_parked();

        let text_spans = pdf
            .text_spans_for_page(0)
            .expect("page 0 should have rendered with real text by now");

        // words spread from top to bottom, so some lie far from the initial strip center.
        let mut words = Vec::new();
        let mut i = 0;
        while i < text_spans.len() {
            if span_char_class(&text_spans[i]) == CharClass::Word {
                let range = word_range_at(&text_spans, i);
                i = range.end;
                if range.len() > 1 {
                    words.push(range);
                }
            } else {
                i += 1;
            }
        }
        assert!(
            words.len() >= 3,
            "expected at least 3 words in the tall-page fixture, found {}",
            words.len()
        );

        // high enough zoom that only a strip of the page is rasterized.
        let zoom = 6.0_f32;
        view.update_in(cx, |view, _window, cx| {
            view.restore_view_state(zoom, 0, cx);
        });
        cx.run_until_parked();
        cx.run_until_parked();

        let full_height_pt = pdf.metadata.pages[0].size.height;
        let covered_height_pt = pdf
            .page_cache
            .read()
            .get(&0)
            .expect("page 0 should be cached by now")
            .covered_height_pt();
        assert!(
            covered_height_pt < full_height_pt,
            "expected zoom {zoom} to force a strip render (covered {covered_height_pt}pt of \
             {full_height_pt}pt) - bump the zoom or the fixture's page height if this no \
             longer holds"
        );

        for expected_range in &words {
            let bounds = text_spans[expected_range.start].bounds;
            let target_center = Point {
                x: bounds.origin.x + bounds.size.width / 2.0,
                y: bounds.origin.y + bounds.size.height / 2.0,
            };

            // scroll the word into view before clicking, which also requests a new strip.
            let target_y_in_page = target_center.y * zoom;
            view.update_in(cx, |view, _window, cx| {
                view.scroll_handle.set_offset(Point {
                    x: px(0.0),
                    y: -px((target_y_in_page - 100.0).max(0.0)),
                });
                cx.notify();
            });
            cx.run_until_parked();
            cx.run_until_parked();

            let page_bounds = cx
                .debug_bounds("pdf-page-0")
                .unwrap_or_else(|| panic!("page 0 should be laid out at zoom {zoom}"));
            let click_point = Point {
                x: page_bounds.origin.x + px(target_center.x * zoom),
                y: page_bounds.origin.y + px(target_center.y * zoom),
            };

            cx.simulate_event(gpui::MouseDownEvent {
                position: click_point,
                modifiers: gpui::Modifiers::none(),
                button: MouseButton::Left,
                click_count: 2,
                first_mouse: false,
            });
            cx.simulate_event(gpui::MouseUpEvent {
                position: click_point,
                modifiers: gpui::Modifiers::none(),
                button: MouseButton::Left,
                click_count: 2,
            });

            view.update(cx, |view, _cx| {
                let selection = view.text_selection.as_ref().unwrap_or_else(|| {
                    panic!("double-click on {expected_range:?} should select a word")
                });
                let (start, end) = selection.ordered();
                let selected_text: String = text_spans[start.char_index..=end.char_index]
                    .iter()
                    .map(|s| s.text.as_str())
                    .collect();
                let expected_text: String = text_spans[expected_range.clone()]
                    .iter()
                    .map(|s| s.text.as_str())
                    .collect();
                assert_eq!(
                    (start.page_index, start.char_index, end.char_index),
                    (0, expected_range.start, expected_range.end - 1),
                    "double-clicking {expected_text:?} while strip-rendered selected \
                     {selected_text:?} instead"
                );
            });
        }
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn double_click_selects_the_right_word_in_dense_text_at_extreme_zoom(
        cx: &mut gpui::TestAppContext,
    ) {
        init_test(cx);
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_fixtures/dense_paragraph.pdf"
        );
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);

        let (view, cx) =
            cx.add_window_view(|window, cx| PdfView::from_pdf(pdf.clone(), window, cx));
        cx.run_until_parked();
        cx.run_until_parked();

        let text_spans = pdf
            .text_spans_for_page(0)
            .expect("page 0 should have rendered with real text by now");

        let mut words = Vec::new();
        let mut i = 0;
        while i < text_spans.len() {
            if span_char_class(&text_spans[i]) == CharClass::Word {
                let range = word_range_at(&text_spans, i);
                i = range.end;
                if range.len() > 1 {
                    words.push(range);
                }
            } else {
                i += 1;
            }
        }
        assert!(
            words.len() > 20,
            "expected plenty of real words in the dense-paragraph fixture, found {}",
            words.len()
        );

        // high enough zoom to hit both `MAX_RENDER_WIDTH` and `MAX_RENDER_HEIGHT`.
        let zoom = 8.0_f32;
        view.update_in(cx, |view, _window, cx| {
            view.restore_view_state(zoom, 0, cx);
        });
        cx.run_until_parked();
        cx.run_until_parked();

        let cached = pdf
            .page_cache
            .read()
            .get(&0)
            .expect("page 0 should be cached by now")
            .clone();
        let full_height_pt = pdf.metadata.pages[0].size.height;
        assert!(
            cached.image_scale < zoom * 2.0,
            "expected MAX_RENDER_WIDTH to cap this page's resolution below the nominal \
             scale; image_scale={} nominal={}",
            cached.image_scale,
            zoom * 2.0
        );
        assert!(
            cached.covered_height_pt() < full_height_pt,
            "expected MAX_RENDER_HEIGHT to force a strip render too"
        );

        // every word on the page, across all ten lines.
        for expected_range in &words {
            let bounds = text_spans[expected_range.start].bounds;
            let target_center = Point {
                x: bounds.origin.x + bounds.size.width / 2.0,
                y: bounds.origin.y + bounds.size.height / 2.0,
            };

            // pan both axes, since the page is wider and taller than the window.
            let target_x_in_page = target_center.x * zoom;
            let target_y_in_page = target_center.y * zoom;
            view.update_in(cx, |view, _window, cx| {
                view.scroll_handle.set_offset(Point {
                    x: -px((target_x_in_page - 100.0).max(0.0)),
                    y: -px((target_y_in_page - 100.0).max(0.0)),
                });
                cx.notify();
            });
            cx.run_until_parked();
            cx.run_until_parked();

            let page_bounds = cx
                .debug_bounds("pdf-page-0")
                .unwrap_or_else(|| panic!("page 0 should be laid out at zoom {zoom}"));
            let click_point = Point {
                x: page_bounds.origin.x + px(target_center.x * zoom),
                y: page_bounds.origin.y + px(target_center.y * zoom),
            };

            cx.simulate_event(gpui::MouseDownEvent {
                position: click_point,
                modifiers: gpui::Modifiers::none(),
                button: MouseButton::Left,
                click_count: 2,
                first_mouse: false,
            });
            cx.simulate_event(gpui::MouseUpEvent {
                position: click_point,
                modifiers: gpui::Modifiers::none(),
                button: MouseButton::Left,
                click_count: 2,
            });

            view.update(cx, |view, _cx| {
                let selection = view.text_selection.as_ref().unwrap_or_else(|| {
                    panic!("double-click on {expected_range:?} should select a word")
                });
                let (start, end) = selection.ordered();
                let selected_text: String = text_spans[start.char_index..=end.char_index]
                    .iter()
                    .map(|s| s.text.as_str())
                    .collect();
                let expected_text: String = text_spans[expected_range.clone()]
                    .iter()
                    .map(|s| s.text.as_str())
                    .collect();
                assert_eq!(
                    (start.page_index, start.char_index, end.char_index),
                    (0, expected_range.start, expected_range.end - 1),
                    "double-clicking {expected_text:?} at extreme zoom selected \
                     {selected_text:?} instead"
                );
            });
        }
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn modified_scroll_and_pinch_zoom_but_plain_scroll_does_not(
        cx: &mut gpui::TestAppContext,
    ) {
        init_test(cx);
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_fixtures/dense_paragraph.pdf"
        );
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);
        let (view, cx) =
            cx.add_window_view(|window, cx| PdfView::from_pdf(pdf.clone(), window, cx));
        view.update_in(cx, |view, _window, cx| view.restore_view_state(1.0, 0, cx));
        cx.run_until_parked();

        let position = cx
            .debug_bounds("pdf-page-0")
            .unwrap_or_else(|| panic!("page 0 should be laid out"))
            .center();
        let zoom_level =
            |cx: &mut gpui::VisualTestContext| view.read_with(cx, |view, _| view.zoom_level);
        let scroll = |cx: &mut gpui::VisualTestContext, modifiers: gpui::Modifiers| {
            cx.simulate_event(ScrollWheelEvent {
                position,
                delta: gpui::ScrollDelta::Pixels(gpui::point(px(0.), px(50.))),
                modifiers,
                ..Default::default()
            });
            cx.run_until_parked();
        };

        scroll(cx, gpui::Modifiers::none());
        assert_eq!(zoom_level(cx), 1.0);

        scroll(cx, gpui::Modifiers::control());
        assert!((zoom_level(cx) - 1.5).abs() < 0.001, "{}", zoom_level(cx));

        cx.simulate_event(PinchEvent {
            position,
            delta: -0.5,
            ..Default::default()
        });
        cx.run_until_parked();
        assert!((zoom_level(cx) - 0.75).abs() < 0.001, "{}", zoom_level(cx));
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn single_page_view_stays_on_the_page_and_flips_once_per_gesture(
        cx: &mut gpui::TestAppContext,
    ) {
        init_test(cx);
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_fixtures/two_pages_with_text.pdf"
        );
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);
        let (view, cx) =
            cx.add_window_view(|window, cx| PdfView::from_pdf(pdf.clone(), window, cx));
        view.update_in(cx, |view, window, cx| {
            view.restore_view_state(6.0, 0, cx);
            view.toggle_single_page_view(&ToggleSinglePageView, window, cx);
        });
        cx.run_until_parked();

        let position = cx
            .debug_bounds("pdf-page-0")
            .unwrap_or_else(|| panic!("page 0 should be laid out"))
            .center();
        let scroll = |cx: &mut gpui::VisualTestContext, delta_y: f32, phase: gpui::TouchPhase| {
            cx.simulate_event(ScrollWheelEvent {
                position,
                delta: gpui::ScrollDelta::Pixels(gpui::point(px(0.), px(delta_y))),
                touch_phase: phase,
                ..Default::default()
            });
            cx.run_until_parked();
        };
        let state = |cx: &mut gpui::VisualTestContext| {
            view.read_with(cx, |view, cx| {
                let viewport = view.container_bounds.expect("view should be laid out");
                let range = single_page_scroll_range(
                    &view.pdf(cx),
                    view.current_page,
                    view.zoom_level,
                    view.rotation,
                    viewport.size.height.as_f32(),
                )
                .expect("current page should exist");
                (
                    view.current_page,
                    -view.scroll_handle.offset().y.as_f32(),
                    range,
                )
            })
        };

        // one long gesture, as a trackpad fling with momentum sends.
        scroll(cx, -40.0, gpui::TouchPhase::Started);
        for _ in 0..200 {
            scroll(cx, -40.0, gpui::TouchPhase::Moved);
            let (page, scroll_y, range) = state(cx);
            assert!(
                range.contains(&scroll_y),
                "page {page} scrolled to {scroll_y}, outside {range:?}"
            );
        }
        let (page, _, range) = state(cx);
        assert_eq!(
            page, 1,
            "scrolling past the end should flip to the next page"
        );
        assert!(
            *range.end() > 0.0,
            "the page should be taller than the viewport to exercise scrolling within it"
        );

        // the same gesture turning around must not flip back.
        for _ in 0..200 {
            scroll(cx, 40.0, gpui::TouchPhase::Moved);
        }
        let (page, scroll_y, range) = state(cx);
        assert_eq!(page, 1, "a gesture flips at most one page");
        assert_eq!(scroll_y, *range.start());

        cx.executor().advance_clock(PAGE_FLIP_GESTURE_GAP * 2);
        for _ in 0..2 {
            scroll(cx, 40.0, gpui::TouchPhase::Moved);
        }
        let (page, scroll_y, range) = state(cx);
        assert_eq!(page, 0, "a new gesture should flip back");
        assert_eq!(
            scroll_y,
            *range.end(),
            "flipping back should land on the end of the previous page"
        );

        // offsets are relative to the shown page, since it is the only one laid out.
        let target = PdfTarget {
            page_index: 1,
            point: Point { x: 0.0, y: 15.0 },
            block_bottom_y: None,
            line_height: None,
        };
        let target_y = view.update(cx, |view, cx| {
            view.jump_to_target(target.clone(), ScrollMode::Top, cx);
            view.display_target_point(&view.pdf(cx), &target).y
        });
        cx.run_until_parked();
        let (page, scroll_y, _) = state(cx);
        assert_eq!(page, 1);
        assert_eq!(scroll_y, target_y * 6.0 - DEFAULT_OFFSET);
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn a_plain_click_never_shows_a_selection_but_a_real_drag_does(
        cx: &mut gpui::TestAppContext,
    ) {
        init_test(cx);
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_fixtures/dense_paragraph.pdf"
        );
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);

        let (view, cx) =
            cx.add_window_view(|window, cx| PdfView::from_pdf(pdf.clone(), window, cx));
        cx.run_until_parked();
        cx.run_until_parked();

        let text_spans = pdf
            .text_spans_for_page(0)
            .expect("page 0 should have rendered with real text by now");
        let mut words = Vec::new();
        let mut i = 0;
        while i < text_spans.len() {
            if span_char_class(&text_spans[i]) == CharClass::Word {
                let range = word_range_at(&text_spans, i);
                i = range.end;
                if range.len() > 1 {
                    words.push(range);
                }
            } else {
                i += 1;
            }
        }
        assert!(words.len() >= 2, "need at least two words for a drag");

        let zoom = 1.0_f32;
        view.update_in(cx, |view, _window, cx| {
            view.restore_view_state(zoom, 0, cx);
        });
        cx.run_until_parked();
        cx.run_until_parked();

        let page_bounds = cx
            .debug_bounds("pdf-page-0")
            .unwrap_or_else(|| panic!("page 0 should be laid out"));
        let click_point_for = |range: &std::ops::Range<usize>| {
            let bounds = text_spans[range.start].bounds;
            let center = Point {
                x: bounds.origin.x + bounds.size.width / 2.0,
                y: bounds.origin.y + bounds.size.height / 2.0,
            };
            Point {
                x: page_bounds.origin.x + px(center.x * zoom),
                y: page_bounds.origin.y + px(center.y * zoom),
            }
        };

        // a plain click must never populate `text_selection`, not even transiently.
        let plain_click_point = click_point_for(&words[0]);
        cx.simulate_event(gpui::MouseDownEvent {
            position: plain_click_point,
            modifiers: gpui::Modifiers::none(),
            button: MouseButton::Left,
            click_count: 1,
            first_mouse: false,
        });
        view.update(cx, |view, _cx| {
            assert!(
                view.text_selection.is_none(),
                "a plain click's mouse-down should not show a selection, even before mouse-up"
            );
        });
        cx.simulate_event(gpui::MouseUpEvent {
            position: plain_click_point,
            modifiers: gpui::Modifiers::none(),
            button: MouseButton::Left,
            click_count: 1,
        });
        view.update(cx, |view, _cx| {
            assert!(
                view.text_selection.is_none(),
                "a plain click should still show no selection after mouse-up"
            );
        });

        // a drag between words produces a visible selection.
        let drag_start = click_point_for(&words[0]);
        let drag_end = click_point_for(&words[1]);
        cx.simulate_event(gpui::MouseDownEvent {
            position: drag_start,
            modifiers: gpui::Modifiers::none(),
            button: MouseButton::Left,
            click_count: 1,
            first_mouse: false,
        });
        cx.simulate_event(gpui::MouseMoveEvent {
            position: drag_end,
            modifiers: gpui::Modifiers::none(),
            pressed_button: Some(MouseButton::Left),
        });
        view.update(cx, |view, _cx| {
            let selection = view
                .text_selection
                .as_ref()
                .expect("a real drag should produce a visible selection");
            let (start, end) = selection.ordered();
            assert_eq!(start.char_index, words[0].start);
            assert_eq!(end.char_index, words[1].start);
        });
        cx.simulate_event(gpui::MouseUpEvent {
            position: drag_end,
            modifiers: gpui::Modifiers::none(),
            button: MouseButton::Left,
            click_count: 1,
        });
        view.update(cx, |view, _cx| {
            assert!(
                view.text_selection.is_some(),
                "the drag's selection should survive mouse-up"
            );
        });
    }

    #[test]
    fn compute_fit_to_view_zoom_stays_finite_and_clamped_for_degenerate_input() {
        let page_size = Size {
            width: 612.0,
            height: 792.0,
        };
        let normal_bounds = Bounds {
            origin: Point::default(),
            size: Size {
                width: px(1000.0),
                height: px(800.0),
            },
        };
        let zoom = PdfView::compute_fit_to_view_zoom(normal_bounds, page_size);
        assert!(zoom.is_finite() && zoom > 0.0);
        assert!((MIN_ZOOM..=MAX_ZOOM).contains(&zoom));

        // zero container bounds fall back to a zoom of 1.0.
        let zero_bounds = Bounds {
            origin: Point::default(),
            size: Size {
                width: px(0.0),
                height: px(0.0),
            },
        };
        let zoom = PdfView::compute_fit_to_view_zoom(zero_bounds, page_size);
        assert_eq!(zoom, 1.0);

        // zero page bounds would yield an infinite or NaN scale.
        let zero_page = Size {
            width: 0.0,
            height: 0.0,
        };
        let zoom = PdfView::compute_fit_to_view_zoom(normal_bounds, zero_page);
        assert_eq!(zoom, 1.0);
    }
}
