use crate::{
    pdf::Pdf,
    pdf_renderer::{
        PageData, PageRotation, PdfChapter, PdfLinkAction, PdfSearchResult, PdfTarget,
        PdfTextSpan,
    },
    pdf_store::{PdfItem, PdfItemEvent, open_pdf},
};
use anyhow::{Context as _, Error, Result};
use collections::{HashMap, HashSet};
use editor::{Editor, EditorSettings};
use gpui::{
    AnyElement, App, Bounds, Context, Element, ElementId, Entity, EntityId, EventEmitter,
    FocusHandle, Focusable, Font, GlobalElementId, InspectorElementId, InteractiveElement,
    IntoElement, KeyDownEvent, LayoutId, MouseButton, ParentElement, Pixels, Point, Render,
    ScrollHandle, Size, Style, Styled, Subscription, Task, WeakEntity, Window, actions, div, img,
    px, size, uniform_list,
};
use language::File as _;
use persistence::PdfViewerDb;
use project::{Project, ProjectPath, search::SearchQuery};
use settings::{SeedQuerySetting, Settings};
use std::{path::Path, sync::Arc, time::Duration};
use theme_settings::ThemeSettings;
use ui::{FluentBuilder as _, ScrollAxes, Tooltip, WithScrollbar, prelude::*};
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
pub mod pdf;
pub mod pdf_renderer;
pub mod pdf_store;

actions!(
    pdf_viewer,
    [
        /// Zoom in the pdf.
        ZoomIn,
        /// Zoom out the pdf.
        ZoomOut,
        /// Reset zoom to 100%.
        ResetZoom,
        /// Fit the pdf to view.
        FitToView,
        /// Fit the pdf to width.
        FitToWidth,
        /// Zoom to actual size (100%).
        ZoomToActualSize,
        /// Go to the next page.
        NextPage,
        /// Go to the previous page.
        PreviousPage,
        /// Copy the selected text.
        Copy,
        /// Select all text on currently-rendered pages.
        SelectAll,
        /// Toggle the bookmark/outline sidebar.
        ToggleOutline,
        /// Toggle the thumbnail sidebar.
        ToggleThumbnails,
        /// Toggle between continuous-scroll and single-page view.
        ToggleSinglePageView,
        /// Rotate the view 90 degrees clockwise (view-only, not saved to the file).
        RotatePageClockwise
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
const SEARCH_DEBOUNCE: Duration = Duration::from_millis(300);

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
    /// All pages laid out top-to-bottom, scrolling freely between them (the default).
    Continuous,
    /// Only `current_page` is rendered/shown; navigation is page-by-page (`NextPage`/
    /// `PreviousPage`/outline/thumbnail clicks, all of which already route through
    /// `jump_to_page`). Note this doesn't clamp manual scroll-wheel input to stay within the
    /// current page, see `ToggleSinglePageView`'s doc comment.
    SinglePage,
}

/// A position within a document's text, identified the same way `PageData.text_spans`
/// already orders characters: by page, then by index within that page's per-character spans
/// (PDFium's own extraction order, which is reading order).
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
    /// Returns `(start, end)` in document order, regardless of which direction the drag ran.
    fn ordered(&self) -> (TextPosition, TextPosition) {
        if self.anchor <= self.head {
            (self.anchor, self.head)
        } else {
            (self.head, self.anchor)
        }
    }
}

/// Finds the character in `text_spans` whose bounds are closest to `point` (both in
/// page-point space). Used to turn a mouse click into a `TextPosition`. Returns `None` only
/// when the page has no extracted text at all.
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

/// The half-open range of `text_spans` indices selected on `page_index`, or `None` if this
/// page isn't part of the selection at all. `span_count` is that page's `text_spans.len()`.
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

/// Groups consecutive same-line characters in `range` into per-line rectangles, so rendering
/// a selection spanning many characters doesn't need one element per character, roughly one
/// per line instead. Two spans are treated as the same line when their vertical bounds
/// overlap by at least half of the shorter one's height.
fn selection_highlight_rects(text_spans: &[PdfTextSpan], range: std::ops::Range<usize>) -> Vec<Bounds<f32>> {
    let mut rects: Vec<Bounds<f32>> = Vec::new();
    for span in &text_spans[range] {
        let bounds = span.bounds;
        if let Some(last) = rects.last_mut() {
            let overlap = (last.origin.y + last.size.height).min(bounds.origin.y + bounds.size.height)
                - last.origin.y.max(bounds.origin.y);
            let shorter_height = last.size.height.min(bounds.size.height);
            if shorter_height > 0.0 && overlap >= shorter_height * 0.5 {
                let left = last.origin.x.min(bounds.origin.x);
                let right = (last.origin.x + last.size.width).max(bounds.origin.x + bounds.size.width);
                let top = last.origin.y.min(bounds.origin.y);
                let bottom = (last.origin.y + last.size.height).max(bounds.origin.y + bounds.size.height);
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

/// Extracts the selected text across however many of `page_text_spans` (one entry per
/// selected page, in order, paired with that page's `text_spans`) are actually available.
/// Pages outside the currently-rendered/cached set simply aren't included, see the note on
/// `PdfView::text_selection` for why that's an intentional scope cut, not a bug.
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

/// A page's screen bounds and text spans, captured once per `prepaint` pass for the outer
/// container's drag-selection mouse handlers to hit-test against.
type SelectionHitRegion = (usize, Bounds<Pixels>, Arc<Vec<PdfTextSpan>>);

/// Turns a window-space drag point into a `TextPosition`, for extending a selection as the
/// mouse moves. Prefers whichever page region actually contains the point; if the drag has
/// gone above/below the whole document, falls back to the vertically-nearest page instead of
/// leaving the selection stuck, matching how dragging past a text editor's edges still
/// extends the selection there.
fn hit_test_selection_point(
    regions: &[SelectionHitRegion],
    point: Point<Pixels>,
    zoom: f32,
) -> Option<TextPosition> {
    let region = regions
        .iter()
        .find(|(_, bounds, _)| bounds.contains(&point))
        .or_else(|| {
            regions.iter().min_by(|(_, a, _), (_, b, _)| {
                vertical_distance(*a, point).total_cmp(&vertical_distance(*b, point))
            })
        })?;
    let (page_index, bounds, text_spans) = region;
    let local = Point {
        x: f32::from(point.x - bounds.origin.x) / zoom,
        y: f32::from(point.y - bounds.origin.y) / zoom,
    };
    let char_index = nearest_char_index(text_spans, local)?;
    Some(TextPosition {
        page_index: *page_index,
        char_index,
    })
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

/// Each `PdfTextSpan` is one character (see `PdfRenderer::process_page`'s `text_page.chars()`
/// extraction), so classifying a span is just classifying its first char. An empty `text`
/// (shouldn't happen, but PDFium's `unicode_string()` isn't guaranteed non-empty) is treated
/// as whitespace rather than panicking on an empty-string index.
fn span_char_class(span: &PdfTextSpan) -> CharClass {
    span.text.chars().next().map(char_class).unwrap_or(CharClass::Whitespace)
}

/// The maximal run of same-class (word / punctuation / whitespace) characters containing
/// `char_index`, for double-click word selection. Mirrors how most text editors treat a
/// contiguous run of punctuation as its own "word" too, rather than only ever selecting
/// alphanumeric runs.
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

/// The maximal run of characters on the same visual line as `char_index`, for triple-click
/// line selection. Uses the same "vertical bounds overlap by at least half the shorter span's
/// height" test as `selection_highlight_rects` groups lines by, but compares every candidate
/// directly against the clicked char's own bounds (rather than chaining neighbor-to-neighbor)
/// so the scan can't drift onto an adjacent line one char at a time.
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

/// Builds the `TextSelection` spanning every page in `pages_with_text` (each a `(page_index,
/// char_count)` pair for a page that has at least one character), from the first page's start
/// to the last page's end. `None` if no page has any text yet. Pulled out of `PdfView::select_all`
/// so the "which pages bound the selection" decision can be tested without a real render pipeline.
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

/// Tracks which outline/bookmark tree nodes the user has explicitly expanded or collapsed,
/// identified by their path of child indices from the root (e.g. `[1, 0]` is the first child
/// of the second top-level chapter). Nodes with no entry here fall back to a default:
/// top-level chapters expanded, anything deeper collapsed, so a freshly opened document's
/// outline isn't a single collapsed list, but also isn't fully expanded for a deeply nested
/// one.
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

    /// A path is never a real child-index sequence once it ends in `usize::MAX`, so this
    /// doubles as an explicit "collapsed" record without needing a second set alongside `0`.
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
    last_search_query: Option<(SharedString, bool, Vec<PdfSearchResult>)>,
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
    /// The in-flight `select_all` extraction task, if any. Stored rather than `.detach()`ed
    /// so tests can retrieve and await it directly, matching `password_task`/
    /// `clear_jump_task`'s pattern for other spawned tasks that mutate `self` on completion.
    select_all_task: Option<Task<()>>,
    /// The current text selection, if any. Click-drag selection is limited to pages that
    /// have already been rendered (`text_spans` is only populated as a side effect of
    /// `process_page`'s full render, and drag hit-testing needs a rendered page's on-screen
    /// bounds anyway), dragging into a not-yet-rendered page just doesn't extend there.
    /// `select_all` isn't subject to that limit: it requests every page's text via the
    /// separate, non-rasterizing `Pdf::request_text_spans` path instead, see its doc
    /// comment.
    text_selection: Option<TextSelection>,
    /// A plain (single-click, not double/triple) mouse-down's anchor, held here instead of
    /// in `text_selection` until the drag actually moves to a different character. Chrome/
    /// Edge only ever show a selection after a real drag or a double/triple-click, so eagerly
    /// putting the 1-character click position into `text_selection` (then clearing it on
    /// mouse-up if nothing moved) made an unwanted selection flash on screen for one frame
    /// before disappearing, rather than never appearing at all.
    pending_click_anchor: Option<TextPosition>,
    is_selecting_text: bool,
    show_outline: bool,
    outline_expanded: OutlineExpansionState,
    show_thumbnails: bool,
    /// Set by `restore_view_state` when reopening a previously-serialized PdfView, to stop
    /// the first `prepaint` pass from overwriting the restored zoom with its usual
    /// fit-to-view-on-first-layout default.
    suppress_initial_fit_to_view: bool,
    view_mode: ViewMode,
    /// View-only page rotation (see `PageRotation`'s doc comment). A per-session display
    /// setting, not persisted and never written back to the file.
    rotation: PageRotation,
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
        }
    }

    fn new_password_input(window: &mut Window, cx: &mut Context<Self>) -> Entity<Editor> {
        cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_masked(true, cx);
            editor
        })
    }

    /// Helper to grab the current PDF regardless of its source
    pub fn pdf(&self, cx: &App) -> Arc<Pdf> {
        match &self.source {
            PdfSource::Item(item) => item.read(cx).pdf.clone(),
            PdfSource::Memory(pdf) => pdf.clone(),
        }
    }

    /// Create a PdfView from an already loaded Pdf
    pub fn from_pdf(pdf: Arc<Pdf>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        Self {
            source: PdfSource::Memory(pdf),
            stale_pdf: None,
            stale_page_cache: Default::default(),
            last_search_query: None,
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
        }
    }

    /// Update the underlying pdf
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
                    this.update(cx, |this, cx| this.update_pdf(pdf, cx)).log_err();
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

    /// Selects all text on currently-rendered pages. Selection (see `text_selection`'s doc
    /// comment) is already scoped to pages that have actually been rasterized/text-extracted,
    /// not the whole document sight-unseen. "Select all" inherits that same scope rather than
    /// forcing every page in a large, mostly-unrendered document to render just to be selected.
    /// Chrome/Edge's PDF viewer can select-all across an entire document instantly because
    /// PDFium text extraction is decoupled from rasterizing a bitmap: extraction is cheap
    /// (just walking already-parsed page objects), rendering is the expensive part. Request
    /// every page's text the same way (`Pdf::request_text_spans` /
    /// `PdfRenderer::extract_text_only`, no bitmap involved) rather than limiting Select All
    /// to whatever already happens to be on screen.
    fn select_all(&mut self, _: &SelectAll, _window: &mut Window, cx: &mut Context<Self>) {
        let pdf = self.pdf(cx);
        let page_count = pdf.metadata.page_count;
        if page_count == 0 {
            return;
        }

        self.select_all_task = Some(cx.spawn(async move |this, cx| {
            // One page at a time, not all of them at once: every extraction (like every
            // render) goes through the same single PDFium instance behind one async mutex
            // (see `get_pdfium`), and firing page_count requests concurrently would queue that
            // many waiters on it in one burst, which can starve the currently-visible page's
            // own (higher-priority) render behind a long line of this low-priority work on a
            // large document, making the whole viewer look stuck on "Loading...". Awaiting
            // sequentially keeps at most one extra waiter ahead of any render request that
            // comes in while this runs.
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
        // Re-snap to the top of the current page either way: entering single-page mode
        // shouldn't leave the view mid-scroll through a page that's no longer rendered, and
        // leaving it re-aligns continuous scroll now that neighboring pages are back.
        self.jump_to_page(self.current_page, cx);
    }

    fn rotate_page_clockwise(
        &mut self,
        _: &RotatePageClockwise,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.rotation = self.rotation.rotated_clockwise();
        // Text selection, search-highlight boxes, and link click-through hit-testing are
        // computed in the page's own (unrotated) coordinate space and aren't rotation-aware
        // yet (see PdfContentElement::paint), so clear any in-progress selection rather than
        // leave one anchored to coordinates that no longer line up with what's on screen.
        self.text_selection = None;
        self.is_selecting_text = false;
        // A quarter-turn swaps this page's on-screen width/height, so re-snap the scroll
        // position the same way toggling single-page view does, rather than leaving the
        // viewport wherever the old (now wrong-shaped) layout happened to have it.
        self.jump_to_page(self.current_page, cx);
        cx.notify();
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

        // Kick off (or reuse an in-flight/already-cached) thumbnail render; once it lands,
        // on_rendered just asks for a re-render so this row picks up the freshly cached image.
        let view_weak = cx.entity().downgrade();
        if let Some(task) = pdf.clone().request_thumbnail(cx, page_idx, move |cx| {
            if let Some(view) = view_weak.upgrade() {
                view.update(cx, |_, cx| cx.notify());
            }
        }) {
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
                    .bg(gpui::white())
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
            .child(
                div().w(px(14.0)).flex_none().when(has_children, |this| {
                    this.child(
                        div()
                            .id(("pdf-outline-toggle", path_hash))
                            .cursor_pointer()
                            .on_mouse_down(MouseButton::Left, cx.listener(move |this, _, _window, cx| {
                                this.toggle_outline_node(toggle_path.clone(), cx);
                            }))
                            .child(Icon::new(if expanded {
                                IconName::ChevronDown
                            } else {
                                IconName::ChevronRight
                            })
                            .size(IconSize::XSmall)),
                    )
                }),
            )
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
        log::info!("Clicked zoom in");
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
        let page_size = pdf.get_page_size(self.current_page).ok();
        if let Some((bounds, size)) = self.container_bounds.zip(page_size) {
            let new_zoom = PdfView::compute_fit_to_view_zoom(bounds, size);
            self.set_zoom(new_zoom, None, cx);
        }
    }

    fn fit_to_width(&mut self, _: &FitToWidth, _window: &mut Window, cx: &mut Context<Self>) {
        let pdf = self.pdf(cx);
        let page_size = pdf.get_page_size(self.current_page).ok();
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

    fn compute_fit_to_view_zoom(container_bounds: Bounds<Pixels>, page_size: Size<f32>) -> f32 {
        let container_width: f32 = container_bounds.size.width.into();
        let container_height: f32 = container_bounds.size.height.into();
        let scale_x = container_width / page_size.width;
        let scale_y = container_height / page_size.height;
        let zoom = scale_x.min(scale_y).min(1.0);
        // Degenerate container/page bounds (e.g. pane not yet sized, or a
        // corrupt page with zero width/height) can produce 0/NaN/inf here;
        // fall back to a sane default rather than propagating that further.
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
            // Use read() to guarantee we don't skip the cache update if a background task is inserting
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
            log::info!(
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

        // Calculate dynamic base offset due to margin centering
        let viewport_width = self
            .container_bounds
            .map(|b| b.size.width)
            .unwrap_or(px(0.0));
        let old_max_width = px(pdf.metadata.max_width * old_zoom);
        let new_max_width = px(pdf.metadata.max_width * self.zoom_level);

        let old_base_offset_x = (viewport_width - old_max_width).max(px(0.0)) / 2.0;
        let new_base_offset_x = (viewport_width - new_max_width).max(px(0.0)) / 2.0;

        // Calculate exact horizontal document position relative to the document's true origin
        let doc_x = -offset.x + center_x - old_base_offset_x;
        offset.x = -((doc_x * zoom_ratio) + new_base_offset_x - center_x);

        // Calculate exact vertical document position considering unscaled gaps
        let doc_y = -offset.y + center_y;

        let gap = px(PAGE_GAP);
        let mut current_top = px(0.0);
        let mut target_page = page_count.saturating_sub(1);
        let mut local_y = px(0.0);

        for i in 0..page_count {
            let page_height = px(pdf.get_page_size(i).unwrap_or_default().height * old_zoom);
            if doc_y < current_top + page_height + gap {
                target_page = i;
                local_y = doc_y - current_top;
                break;
            }
            current_top += page_height + gap;
        }

        let mut new_current_top = px(0.0);
        for i in 0..target_page {
            new_current_top +=
                px(pdf.get_page_size(i).unwrap_or_default().height * self.zoom_level) + gap;
        }

        let new_doc_y = new_current_top + local_y * zoom_ratio;
        offset.y = -(new_doc_y - center_y);

        self.scroll_handle.set_offset(offset);
        cx.notify();
    }

    /// Applies a previously-saved zoom level and page, e.g. right after reopening a PdfView
    /// that was serialized on a prior session (see `SerializableItem::deserialize`). Must be
    /// called before the view's first `prepaint` for `suppress_initial_fit_to_view` to have
    /// any effect. After that pass, the fit-to-view default it suppresses never runs again
    /// regardless.
    pub fn restore_view_state(&mut self, zoom_level: f32, page_index: usize, cx: &mut Context<Self>) {
        self.zoom_level = zoom_level.clamp(MIN_ZOOM, MAX_ZOOM);
        self.suppress_initial_fit_to_view = true;
        self.jump_to_page(page_index, cx);
    }

    /// Jump to the beginning of a page
    pub fn jump_to_page(&mut self, page_index: usize, cx: &mut Context<Self>) {
        self.jump_to_target(
            PdfTarget {
                page_index,
                point: Point::default(),
                block_bottom_y: None,
                line_height: None,
            },
            ScrollMode::Top,
            cx,
        );
    }

    /// Jump to a point of a page, center if if center=true
    pub fn jump_to_target(&mut self, target: PdfTarget, mode: ScrollMode, cx: &mut Context<Self>) {
        use ScrollMode::*;

        let pdf = self.pdf(cx);

        if target.page_index >= pdf.metadata.page_count {
            return; // out of bounds
        }

        let mut total_y_offset = 0.0;
        let zoom = self.zoom_level;

        // sum the heights and gaps of all pages BEFORE the target page
        for i in 0..target.page_index {
            let size = pdf.get_page_size(i).unwrap();
            total_y_offset += size.height * zoom + PAGE_GAP;
        }

        // add the final page offset
        total_y_offset += target.point.y * zoom;
        let x_offset = target.point.x * zoom;

        // apply the scroll offset, must be negative
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

        let zoom = self.zoom_level;
        let mut target_y = 0.0;
        for i in 0..target.page_index {
            target_y += pdf.get_page_size(i).unwrap().height * zoom + PAGE_GAP;
        }
        target_y += target.point.y * zoom;
        let target_x = target.point.x * zoom;

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
            )
        };

        let page_count = pdf.metadata.page_count;
        if page_count == 0 {
            return None;
        }

        // A page's own on-screen box swaps width/height under a quarter-turn rotation, so
        // the rendered (already-rotated, see PdfRenderer::process_page) image still fills it
        // correctly instead of looking squashed or letterboxed.
        let display_page_size = |size: Size<f32>| {
            if rotation.swaps_dimensions() {
                Size {
                    width: size.height,
                    height: size.width,
                }
            } else {
                size
            }
        };

        // A restored zoom (see PdfView::restore_view_state) takes priority over the usual
        // fit-to-view-on-first-layout default.
        let initial_zoom_level = (first_layout && !suppress_initial_fit_to_view).then(|| {
            PdfView::compute_fit_to_view_zoom(
                bounds,
                display_page_size(pdf.get_page_size(0).unwrap()),
            )
        });

        let zoom_level = initial_zoom_level.unwrap_or(current_zoom_level);

        // GPUI's Pixels are logical/device-independent. Rasterizing at exactly `zoom_level`
        // px-per-point leaves the bitmap under-resolved on a HiDPI display, where the
        // compositor scales that same logical-pixel box up to more physical pixels than the
        // source image actually has, producing a blurry upscale. `PageData::scale` records
        // whatever was actually requested (see `request_page`'s call below), so every other
        // comparison against a page's cached scale in this function must use this same value,
        // not the bare `zoom_level`, otherwise a page rendered at `render_scale` never
        // matches an `== zoom_level` check and looks permanently uncached.
        let render_scale = zoom_level * window.scale_factor();

        let gap = px(PAGE_GAP);

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

        for page_idx in 0..page_count {
            let page_size = display_page_size(pdf.get_page_size(page_idx).unwrap());
            let width = px(page_size.width * zoom_level);
            let height = px(page_size.height * zoom_level);

            page_dimensions.push((width, height));
            page_tops.push(current_top);

            let page_bottom = current_top + height;
            let visible_start = viewport_top.max(current_top);
            let visible_end = viewport_bottom.min(page_bottom);
            let visible_height = (visible_end - visible_start).max(px(0.0));

            // If even 1 pixel is visible, add it to our visible bounds
            if visible_height > px(0.0) {
                if first_visible.is_none() {
                    first_visible = Some(page_idx);
                }
                last_visible = Some(page_idx);
            }

            // Keep tracking best_page for the UI sidebar/breadcrumbs
            if visible_height > max_visible_height {
                max_visible_height = visible_height;
                best_page = page_idx;
            }

            current_top += height + gap;
        }

        let first_visible = first_visible.unwrap_or(best_page);
        let last_visible = last_visible.unwrap_or(best_page);

        // Single-page mode ignores the scroll-derived visible range above and pins it to
        // current_page. Navigation (NextPage/PreviousPage, outline/thumbnail clicks) already
        // routes through jump_to_page, which keeps the scroll offset aligned with that page,
        // so this is consistent with (not fighting) the geometry those computed from.
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

        if current_page != best_page {
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

        // Evict rendered bitmaps for pages that have scrolled far outside the
        // prerender window so the cache doesn't grow unbounded over a long scroll.
        pdf.page_cache
            .write()
            .retain(|&page_idx, _| page_idx >= start_prerender && page_idx <= end_prerender);
        pdf.failed_pages
            .lock()
            .retain(|&page_idx, _| page_idx >= start_prerender && page_idx <= end_prerender);

        // Visible pages render at `High` priority, prerender-window pages at `Low`, so a
        // fast scroll's newly-visible pages don't sit queued behind still-in-flight
        // prefetch work for pages the user hasn't scrolled to yet, mirroring how Chromium
        // prioritizes visible-viewport tiles over prefetch.
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

            // Page-point-space Y this page cares about most: the overall viewport's
            // vertical center, clamped into this page's own bounds. Only affects pages too
            // tall to render whole at this scale (see MAX_RENDER_HEIGHT); clamping means
            // even an off-screen prerender page picks the edge closest to the viewport
            // rather than always defaulting to its top.
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
                    rotation,
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

        // Center the pages safely without clipping the scroll axis
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
        // Screen bounds + text spans per page, captured for the outer container's
        // mouse-move/mouse-up handlers below to turn a window-space drag point into a
        // (page_index, char_index) TextPosition without redoing the whole layout pass.
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
            // 1. Fetch page data, BUT filter it so we don't accidentally accept the old resolution as "new"!
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
            // Text selection, link hit-testing, and search highlighting below are all
            // computed in the page's own (unrotated) coordinate space. Under a rotation
            // they'd land in the wrong place relative to what's actually on screen, so
            // they're disabled entirely while rotated rather than shown misaligned. See
            // FEATURE_ROADMAP.md's rotate-page entry.
            let interactive_layers_supported = rotation == PageRotation::None;
            if interactive_layers_supported
                && let Some(data) = active_data
            {
                selection_hit_regions.push((page_idx, global_page_bounds, data.text_spans.clone()));
            }
            let text_spans_for_selection = interactive_layers_supported
                .then(|| active_data.map(|data| data.text_spans.clone()))
                .flatten();
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
                        let local_x = e.position.x - global_page_bounds.origin.x;
                        let local_y = e.position.y - global_page_bounds.origin.y;

                        let pdf_x = local_x.as_f32() / zoom;
                        let pdf_y = local_y.as_f32() / zoom;

                        if let Some(view) = view_weak.upgrade() {
                            view.update(cx, |_, cx| {
                                handler(page_idx, Point { x: pdf_x, y: pdf_y }, window, cx);
                            });
                        }
                    }
                })
                .on_mouse_down(MouseButton::Left, move |e, _window, cx| {
                    let Some(text_spans) = &text_spans_for_selection else {
                        return;
                    };
                    let local = Point {
                        x: (e.position.x - global_page_bounds.origin.x).as_f32() / zoom,
                        y: (e.position.y - global_page_bounds.origin.y).as_f32() / zoom,
                    };
                    let Some(char_index) = nearest_char_index(text_spans, local) else {
                        return;
                    };
                    // Double-click selects the word under the cursor, triple-click the whole
                    // line; anything beyond that (quadruple-click, etc.) just repeats the
                    // triple-click behavior rather than needing a paragraph/document notion.
                    // A plain (click_count 1) click doesn't select anything yet, see
                    // `pending_click_anchor`'s doc comment; it only records where a
                    // subsequent drag, if any, should start from.
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

            // 2. Stack the images instead of mutually exclusive if/else!
            let mut image_container = div().size_full().relative();
            let mut need_cache_page = false;

            // Layer 1: Always put the stale data on the bottom layer to prevent a blank frame
            //
            // An oversized page's PageData only covers a vertical strip of it (see
            // MAX_RENDER_HEIGHT in pdf_renderer.rs), positioned by page_offset_pt /
            // covered_height_pt rather than filling the whole page box. For an ordinary
            // page these just work out to 0 and the full page height, matching size_full().
            if let Some(old_data) = &stale_data {
                if do_log {
                    log::info!("Drawn stale for page {:?}", page_idx);
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

            // Layer 2: Put the new data on top, but only if it's actually different from the stale data
            if let Some(new_data) = &page_data {
                let is_same = stale_data
                    .as_ref()
                    .is_some_and(|s| Arc::ptr_eq(&s.image, &new_data.image));
                if !is_same {
                    if do_log {
                        log::info!("Drawn new for page {:?}", page_idx);
                    }
                    let top = px(new_data.page_offset_pt * zoom_level);
                    let strip_height = px(new_data.covered_height_pt() * zoom_level);
                    image_container = image_container.child(
                        div().absolute().top(top).left_0().w_full().h(strip_height).child(
                            img(new_data.image.clone())
                                .id(("pdf-viewer-doc", page_idx))
                                .size_full(),
                        ),
                    );
                }
                need_cache_page = false;
            }

            // Fallback: If neither are ready, show loading
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

            // Text selection highlight
            if let Some(selection) = &text_selection
                && let Some(data) = active_data
                && let Some(range) =
                    selected_char_range_for_page(selection, page_idx, data.text_spans.len())
            {
                let mut selection_layer = div().absolute().top_0().left_0().size_full();
                for rect in selection_highlight_rects(&data.text_spans, range) {
                    let top = px(rect.origin.y * zoom_level);
                    let left = px(rect.origin.x * zoom_level);
                    let w = px(rect.size.width * zoom_level);
                    let h = px(rect.size.height * zoom_level);
                    selection_layer = selection_layer.child(
                        div()
                            .absolute()
                            .top(top)
                            .left(left)
                            .w(w)
                            .h(h)
                            .bg(cx.theme().colors().element_selection_background),
                    );
                }
                child_element = child_element.child(selection_layer);
            }

            // Interactive Links
            if interactive_layers_supported
                && let Some(data) = active_data
            {
                let mut link_layer = div().absolute().top_0().left_0().size_full();

                for link in data.links.iter() {
                    let action = link.action.clone();
                    let view_weak = self.pdf_view.downgrade();

                    let top = px(link.bounds.origin.y * zoom_level);
                    let left = px(link.bounds.origin.x * zoom_level);
                    let w = px(link.bounds.size.width * zoom_level);
                    let h = px(link.bounds.size.height * zoom_level);

                    link_layer = link_layer.child(
                        div()
                            .absolute()
                            .top(top)
                            .left(left)
                            .w(w)
                            .h(h)
                            .bg(gpui::rgba(0xff000044))
                            .border_1()
                            .border_color(gpui::rgba(0xff0000ff))
                            .cursor_pointer()
                            .hover(|style| style.bg(gpui::rgba(0x0060df22)))
                            .on_mouse_down(MouseButton::Left, move |_, _window, cx| {
                                log::info!("Clicked {:?}", action);
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

            // Render Search Highlights
            if interactive_layers_supported && !search_matches.is_empty() {
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
                            let top = px(bound.origin.y * zoom_level);
                            let left = px(bound.origin.x * zoom_level);
                            let w = px(bound.size.width * zoom_level);
                            let h = px(bound.size.height * zoom_level);

                            highlight_layer = highlight_layer.child(
                                div()
                                    .absolute()
                                    .top(top)
                                    .left(left)
                                    .w(w)
                                    .h(h)
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

            // Render Active Jump Highlight
            if let Some((anchor, jump)) = &active_jump
                && jump.page_index == page_idx
            {
                use ScrollAnchor::*;

                // A fixed 20pt guess when the jump source has no real font-size info to
                // derive this from (bookmarks, links, search matches), see PdfTarget's doc
                // comment. Typst preview jumps carry the paragraph's actual font size instead,
                // so the bar's height tracks the document's real text size rather than always
                // over/undershooting it by some fixed amount.
                let line_height = px(jump.line_height.unwrap_or(20.0) * zoom_level);
                let y = px(jump.point.y * zoom_level);

                // Typst jump.point.y targets the text baseline. Shift up by `line_height` so
                // the bar sits above the baseline.
                let top = match anchor {
                    Top => y,
                    Baseline => y - line_height + px(4.0 * zoom_level),
                };
                // block_bottom_y (see PdfTarget's doc comment) is the baseline of the last
                // line covered by the same source block/span jump.point's first line came
                // from. Extend the bar down to cover all of it, not just the first line's
                // height, so it doesn't look stuck on the block's first line regardless of
                // where in the block the cursor actually is. Falls back to jump.point.y (a
                // zero-height extension, i.e. just line_height) for jump sources with no
                // block notion: bookmarks, links, search matches.
                let block_extent = px((jump.block_bottom_y.unwrap_or(jump.point.y) - jump.point.y) * zoom_level);
                let h = line_height + block_extent;

                // Matches the Markdown preview's own current-position gutter marker (see
                // `push_root_block`/`pop_root_block` in the markdown crate): a subtle
                // `border`-colored bar with small rounded corners, not an accent color with a
                // drop shadow: a quieter indicator that doesn't compete with the page itself.
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
                // .items_center()
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
                            let Some(position) =
                                hit_test_selection_point(&selection_hit_regions, event.position, zoom_level)
                            else {
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
                                // The drag has actually moved off the click's starting
                                // character; only now does a selection become visible.
                                this.text_selection = Some(TextSelection { anchor, head: position });
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
                    log::info!("Cleared stale page cache");
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
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let Some(mut element) = prepaint.take() else {
            return;
        };

        element.paint(window, cx);
    }
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
                    if let Some((zoom_level, current_page)) = view_state {
                        view.restore_view_state(zoom_level as f32, current_page.max(0) as usize, cx);
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
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<Task<Result<()>>> {
        let PdfSource::Item(pdf_item) = &self.source else {
            return None;
        };

        let workspace_id = workspace.database_id()?;
        let pdf_path = pdf_item.read(cx).abs_path(cx)?;
        let zoom_level = self.zoom_level as f64;
        let current_page = self.current_page as i64;

        let db = PdfViewerDb::global(cx);
        Some(cx.background_spawn({
            async move {
                log::debug!("Saving pdf at path {pdf_path:?}");
                db.save_pdf_view_state(item_id, workspace_id, pdf_path, zoom_level, current_page)
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
            .when(!has_load_error && !needs_password, |this| {
                let show_outline = self.show_outline;
                let show_thumbnails = self.show_thumbnails;
                let content = div()
                    .id("pdf-content-scroll-area")
                    .flex_1()
                    .h_full()
                    .child(PdfContentElement::new(cx.entity()))
                    .custom_scrollbars(
                        ui::Scrollbars::new(ScrollAxes::Both)
                            .tracked_scroll_handle(&self.scroll_handle)
                            .with_track_along(ScrollAxes::Both, cx.theme().colors().panel_background),
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
            word: false,
            regex: false,
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

            // Fire and forget: request_page's own logic already prevents
            // duplicate rendering if it's already cached or in-flight! Low priority: these
            // are adjacent search matches, not necessarily the currently visible page. We do
            // know roughly where on the page the match itself is, though, so use that as the
            // strip center rather than defaulting to the top.
            let visible_center_y = m.bounds.first().map_or(0.0, |b| b.origin.y);
            // Same HiDPI scaling as the main prerender loop in `prepaint`, otherwise a
            // prefetched adjacent-match page would cache at a blurrier resolution than the
            // one the main loop would request for it once it actually scrolls into view.
            let render_scale = self.zoom_level * window.scale_factor();
            if let Some(task) = pdf.request_page(
                cx,
                page_idx,
                render_scale,
                visible_center_y,
                self.rotation,
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
        let search_string: SharedString = query.as_str().to_string().into();
        let match_case = query.case_sensitive();

        if let Some((last_query, last_case, last_matches)) = &self.last_search_query
            && last_query == &search_string
            && *last_case == match_case
        {
            return Task::ready(last_matches.clone());
        }

        cx.spawn(async move |this, cx| {
            // Debounce: Wait 300ms before starting the heavy FFI search task.
            // If the user types another letter, Zed drops this Task, and the timer cancels.
            cx.background_executor().timer(SEARCH_DEBOUNCE).await;

            let search_string_bg = search_string.to_string();
            let matches = cx
                .background_executor()
                .spawn(async move {
                    crate::pdf_renderer::PdfRenderer::new(
                        pdf.id,
                        pdf.bytes.clone(),
                        pdf.password.clone(),
                    )
                    .search_document(&search_string_bg, match_case, pdf.metadata.clone())
                        .await
                        .unwrap_or_else(|e| {
                            log::error!("PDF search failed: {:?}", e);
                            Vec::new()
                        })
                })
                .await;

            _ = this.update(cx, |view, _| {
                view.last_search_query = Some((search_string, match_case, matches.clone()));
            });

            matches
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

        // If no match is currently active, find the closest one to the current viewport
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

        if let Some(input) = &self.page_input
            && !input.focus_handle(cx).is_focused(window)
        {
            let current_page_str = format!("{}", current_page + 1);
            input.update(cx, |editor, cx| {
                if editor.text(cx) != current_page_str {
                    editor.set_text(current_page_str, window, cx);
                }
            });
        }

        let show_outline = pdf_view.read(cx).show_outline;
        let show_thumbnails = pdf_view.read(cx).show_thumbnails;
        let is_single_page = pdf_view.read(cx).view_mode == ViewMode::SinglePage;
        let is_rotated = pdf_view.read(cx).rotation != PageRotation::None;

        h_flex()
            .w_full()
            .justify_end()
            .gap_1()
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
                    .tooltip(|_window, cx| Tooltip::for_action("Toggle Outline", &ToggleOutline, cx))
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
                            .track_focus(&self.page_input.as_ref().unwrap().focus_handle(cx))
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
                            .child(self.page_input.as_ref().unwrap().clone()),
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
        ];
    }

    db::static_connection!(PdfViewerDb, [WorkspaceDb]);

    impl PdfViewerDb {
        // Saves the path together with the last-known view state (zoom/page) in one
        // INSERT OR REPLACE, rather than as two separate queries. REPLACE deletes and
        // reinserts the row, so a second query that only updated some columns would have its
        // values wiped out by a later call to this one.
        query! {
            pub async fn save_pdf_view_state(
                item_id: ItemId,
                workspace_id: WorkspaceId,
                pdf_path: PathBuf,
                zoom_level: f64,
                current_page: i64
            ) -> Result<()> {
                INSERT OR REPLACE INTO pdf_viewers
                    (item_id, workspace_id, pdf_path, zoom_level, current_page)
                VALUES (?, ?, ?, ?, ?)
            }
        }

        query! {
            pub fn get_pdf_path(item_id: ItemId, workspace_id: WorkspaceId) -> Result<Option<PathBuf>> {
                SELECT pdf_path
                FROM pdf_viewers
                WHERE item_id = ? AND workspace_id = ?
            }
        }

        // Returns (zoom_level, current_page), if they were ever saved for this item.
        query! {
            pub fn get_pdf_view_state(item_id: ItemId, workspace_id: WorkspaceId) -> Result<Option<(f64, i64)>> {
                SELECT zoom_level, current_page
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

    /// Two lines of "Hello"/"World", five 10x10 chars each, left-to-right, top line at y=0,
    /// second line at y=10, a small but representative fixture for the hit-testing/range/
    /// highlight-grouping tests below.
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

        // Dead center of the 'e' in "Hello" (index 1, bounds x:[10,20) y:[0,10)).
        assert_eq!(nearest_char_index(&spans, Point { x: 15.0, y: 5.0 }), Some(1));
        // Dead center of the 'o' in "World" (index 9, second line, last char).
        assert_eq!(nearest_char_index(&spans, Point { x: 45.0, y: 15.0 }), Some(9));
        // Far outside the page entirely, still resolves to *something* close, not None.
        assert_eq!(nearest_char_index(&spans, Point { x: -1000.0, y: -1000.0 }), Some(0));

        assert_eq!(nearest_char_index(&[], Point { x: 0.0, y: 0.0 }), None);
    }

    /// One line, "Hi, Bob" (7 chars incl. the comma and both spaces), for word-boundary tests:
    /// indices 0-1 "Hi" (word), 2 "," (punctuation), 3 " " (whitespace), 4-6 "Bob" (word).
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

        // Anywhere inside "Hi" selects the whole word, not just the clicked char.
        assert_eq!(word_range_at(&spans, 0), 0..2);
        assert_eq!(word_range_at(&spans, 1), 0..2);
        // The comma is its own single-char "word", a distinct class from either neighbor.
        assert_eq!(word_range_at(&spans, 2), 2..3);
        // The space between "," and "Bob" is its own run too.
        assert_eq!(word_range_at(&spans, 3), 3..4);
        // Anywhere inside "Bob" selects the whole word, including up to the last char.
        assert_eq!(word_range_at(&spans, 4), 4..7);
        assert_eq!(word_range_at(&spans, 6), 4..7);

        assert_eq!(word_range_at(&[], 0), 0..0);
    }

    #[test]
    fn line_range_at_stops_at_line_boundaries() {
        let spans = two_line_fixture();

        // Anywhere inside "Hello" (first line, indices 0..5) selects the whole line.
        assert_eq!(line_range_at(&spans, 0), 0..5);
        assert_eq!(line_range_at(&spans, 4), 0..5);
        // Anywhere inside "World" (second line, indices 5..10) selects that line instead.
        assert_eq!(line_range_at(&spans, 5), 5..10);
        assert_eq!(line_range_at(&spans, 9), 5..10);

        assert_eq!(line_range_at(&[], 0), 0..0);
    }

    #[test]
    fn select_all_range_spans_first_to_last_page_regardless_of_input_order() {
        // Deliberately out of order and with a gap (page 1 has no text yet); select-all
        // should still bound on the lowest/highest page that actually has text.
        let pages_with_text = vec![(4, 3), (0, 10)];

        let selection = select_all_range(&pages_with_text).expect("pages have text");
        assert_eq!(
            selection.anchor,
            TextPosition {
                page_index: 0,
                char_index: 0,
            }
        );
        // Last char index is span_count - 1 (inclusive), not span_count itself.
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

        // Before the selection's first page: nothing selected.
        assert_eq!(selected_char_range_for_page(&selection, 0, 10), None);
        // First page: from char_index 3 to the end of that page's spans.
        assert_eq!(selected_char_range_for_page(&selection, 1, 10), Some(3..10));
        // Middle page (fully within the selection, if there were one): everything.
        // Last page: from the start up to and including char_index 1.
        assert_eq!(selected_char_range_for_page(&selection, 2, 10), Some(0..2));
        // Past the selection's last page: nothing selected.
        assert_eq!(selected_char_range_for_page(&selection, 3, 10), None);
        // A page with no text at all: nothing selected, regardless of range.
        assert_eq!(selected_char_range_for_page(&selection, 1, 0), None);
    }

    #[test]
    fn selected_char_range_for_page_is_direction_independent() {
        // Dragging bottom-to-top (head before anchor in document order) must select the same
        // range as the equivalent top-to-bottom drag.
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
        // All of "Hello" (indices 0..5), one line, should collapse to one rect spanning it.
        let rects = selection_highlight_rects(&spans, 0..5);
        assert_eq!(rects.len(), 1);
        assert_eq!(rects[0].origin.x, 0.0);
        assert_eq!(rects[0].size.width, 50.0);
    }

    #[test]
    fn multi_line_selection_produces_one_rect_per_line() {
        let spans = two_line_fixture();
        // "llo" (end of line 1) through "Wor" (start of line 2): two lines, two rects.
        let rects = selection_highlight_rects(&spans, 2..8);
        assert_eq!(rects.len(), 2);
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
        let regions: Vec<SelectionHitRegion> = vec![(
            0,
            Bounds {
                origin: Point {
                    x: px(0.0),
                    y: px(0.0),
                },
                size: Size {
                    width: px(100.0),
                    height: px(100.0),
                },
            },
            page0,
        )];

        // Inside the page's bounds: hits normally.
        let inside = hit_test_selection_point(
            &regions,
            Point {
                x: px(15.0),
                y: px(5.0),
            },
            1.0,
        );
        assert_eq!(
            inside,
            Some(TextPosition {
                page_index: 0,
                char_index: 1
            })
        );

        // Below the page entirely (e.g. dragged past the last page): still resolves via the
        // nearest-region fallback instead of leaving the selection stuck.
        let below = hit_test_selection_point(
            &regions,
            Point {
                x: px(15.0),
                y: px(10_000.0),
            },
            1.0,
        );
        assert!(below.is_some());

        assert_eq!(
            hit_test_selection_point(&[], Point { x: px(0.0), y: px(0.0) }, 1.0),
            None
        );
    }

    #[test]
    fn outline_nodes_default_expanded_at_the_top_level_only() {
        let state = OutlineExpansionState::default();

        // Depth 0 (top-level chapters, one index in the path) default to expanded; anything
        // deeper defaults to collapsed until the user opens it.
        assert!(state.is_expanded(&[0]));
        assert!(state.is_expanded(&[3]));
        assert!(!state.is_expanded(&[0, 0]));
        assert!(!state.is_expanded(&[0, 1, 2]));
    }

    #[test]
    fn toggling_an_outline_node_flips_its_expanded_state_independent_of_its_default() {
        let mut state = OutlineExpansionState::default();

        // Collapse a node that starts expanded by default (top-level).
        assert!(state.is_expanded(&[0]));
        state.toggle(vec![0]);
        assert!(!state.is_expanded(&[0]));
        state.toggle(vec![0]);
        assert!(state.is_expanded(&[0]));

        // Expand a node that starts collapsed by default (depth > 0), and confirm a sibling
        // at the same depth is unaffected.
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

    // PDFIUM_TEST_LOCK is only ever taken by test bodies themselves, never by production
    // code or another task within the same test, so holding it across an await here can't
    // deadlock; it's exactly the kind of case this lint can't distinguish from a real one.
    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn restore_view_state_sets_zoom_and_page_and_suppresses_auto_fit(
        cx: &mut gpui::TestAppContext,
    ) {
        init_test(cx);
        // Pdf::from_bytes does real PDFium FFI work on a background task, not something
        // gpui's deterministic test scheduler can drive itself.
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/solid_red.pdf");
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);

        let (view, cx) =
            cx.add_window_view(|window, cx| PdfView::from_pdf(pdf, window, cx));

        view.update(cx, |view, cx| {
            assert!(!view.suppress_initial_fit_to_view);
            // Restoring a zoom well past MAX_ZOOM should still come out clamped, the same
            // way any other zoom_level assignment does.
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
    async fn rotating_clockwise_four_times_returns_to_unrotated_and_clears_selection(
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

            // Text selection, search highlighting, and link hit-testing aren't
            // rotation-aware yet (see prepaint's interactive_layers_supported gate), so an
            // in-progress selection must be dropped rather than left anchored to
            // coordinates that no longer match what's on screen post-rotation.
            view.text_selection = Some(TextSelection {
                anchor: TextPosition { page_index: 0, char_index: 0 },
                head: TextPosition { page_index: 0, char_index: 0 },
            });
            view.is_selecting_text = true;

            view.rotate_page_clockwise(&RotatePageClockwise, window, cx);
            assert_eq!(view.rotation, PageRotation::Clockwise90);
            assert!(view.text_selection.is_none());
            assert!(!view.is_selecting_text);

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

        let (view, cx) = cx.add_window_view(|window, cx| PdfView::from_pdf(pdf.clone(), window, cx));

        let task = view.update_in(cx, |view, window, cx| {
            view.select_all(&SelectAll, window, cx);
            view.select_all_task.take()
        });
        task.expect("select_all should have spawned a task").await;

        for page_index in 0..pdf.metadata.page_count {
            assert!(
                pdf.text_spans_for_page(page_index).is_some_and(|spans| !spans.is_empty()),
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

        let (view, cx) = cx.add_window_view(|window, cx| PdfView::from_pdf(pdf.clone(), window, cx));
        // Give the fire-and-forget render task kicked off by the first prepaint time to
        // actually rasterize page 0 (real PDFium FFI work on a background thread), then let
        // its on_rendered callback's cx.notify() drive a second paint that picks the now-
        // cached PageData up into `active_data`.
        cx.run_until_parked();
        cx.run_until_parked();

        let text_spans = pdf
            .text_spans_for_page(0)
            .expect("page 0 should have rendered with real text by now");
        // A real multi-char word, not punctuation/whitespace, so word_range_at actually
        // selects more than one character and a wrong-word regression has room to show up.
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

        let (view, cx) = cx.add_window_view(|window, cx| PdfView::from_pdf(pdf.clone(), window, cx));
        cx.run_until_parked();
        cx.run_until_parked();

        let text_spans = pdf
            .text_spans_for_page(0)
            .expect("page 0 should have rendered with real text by now");

        // Collect every real word on the page ("TopWord", "MiddleWord", "BottomWord", see
        // the fixture's generator), spanning from near the top of the page to near the
        // bottom, so at least one of them is far from wherever the initial strip's center
        // happens to land.
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

        // A zoom level high enough (combined with this test window's HiDPI scale factor)
        // that the page's rendered bitmap exceeds MAX_RENDER_HEIGHT and only a strip of it
        // actually gets rasterized, the scenario under test.
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

            // Scroll so the target word is actually within the viewport before clicking it -
            // a real user can't click something scrolled off-screen, and doing so here would
            // just silently no-op (leaving whatever the previous iteration selected) rather
            // than exercising the hit-test at all. This also forces `prepaint` to request a
            // *different* strip of the page (centered on this word), which is the case that
            // actually matters for this test.
            let target_y_in_page = target_center.y * zoom;
            view.update_in(cx, |view, _window, cx| {
                view.scroll_handle
                    .set_offset(Point { x: px(0.0), y: -px((target_y_in_page - 100.0).max(0.0)) });
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
                let selection = view
                    .text_selection
                    .as_ref()
                    .unwrap_or_else(|| panic!("double-click on {expected_range:?} should select a word"));
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

        let (view, cx) = cx.add_window_view(|window, cx| PdfView::from_pdf(pdf.clone(), window, cx));
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

        // A zoom level high enough (with this test window's HiDPI scale factor) that a
        // standard-width letter page hits *both* MAX_RENDER_WIDTH and MAX_RENDER_HEIGHT,
        // forcing a capped-resolution, strip-rendered page, the most extreme real-world
        // case, matching a user zooming a normal document in past 500%.
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

        // Every real word on the page, scattered across all ten lines, not just whatever
        // happens to be near the initial strip's center.
        for expected_range in &words {
            let bounds = text_spans[expected_range.start].bounds;
            let target_center = Point {
                x: bounds.origin.x + bounds.size.width / 2.0,
                y: bounds.origin.y + bounds.size.height / 2.0,
            };

            // Scroll both axes so the target word is actually within the viewport before
            // clicking it. At extreme zoom the page is wider than the window too, not just
            // taller, so a word past the right edge needs a horizontal pan just as much as a
            // word further down needs a vertical scroll.
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
                let selection = view
                    .text_selection
                    .as_ref()
                    .unwrap_or_else(|| panic!("double-click on {expected_range:?} should select a word"));
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

        let (view, cx) = cx.add_window_view(|window, cx| PdfView::from_pdf(pdf.clone(), window, cx));
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

        // A plain click (mouse down, then straight back up at the same point, click_count 1)
        // must never populate `text_selection`, not even transiently, since that's what
        // painted the one-frame "flicker" this test guards against.
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

        // A real drag from one word to another must still produce a visible selection.
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

        // A pane that hasn't been sized yet (zero container bounds) would otherwise divide
        // by zero and produce a zoom of 0; falls back to 1.0 instead.
        let zero_bounds = Bounds {
            origin: Point::default(),
            size: Size {
                width: px(0.0),
                height: px(0.0),
            },
        };
        let zoom = PdfView::compute_fit_to_view_zoom(zero_bounds, page_size);
        assert_eq!(zoom, 1.0);

        // A corrupt page with zero-sized bounds would otherwise divide by zero the other way
        // (an infinite or NaN scale factor).
        let zero_page = Size {
            width: 0.0,
            height: 0.0,
        };
        let zoom = PdfView::compute_fit_to_view_zoom(normal_bounds, zero_page);
        assert_eq!(zoom, 1.0);
    }
}
