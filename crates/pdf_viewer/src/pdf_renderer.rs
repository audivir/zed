use anyhow::{Result, anyhow};
use futures::lock::Mutex;
use gpui::{Bounds, Point, RenderImage, Size};
use image::{Frame, RgbaImage};
use once_cell::sync::OnceCell;
use pdfium_render::prelude::{
    PdfBitmap, PdfBitmapFormat, PdfBookmark, PdfColor, PdfDestination, PdfDocument,
    PdfLink as PdfiumLink, PdfPage, PdfPageAnnotation, PdfPageAnnotationCommon,
    PdfPageRenderRotation, PdfRect, PdfRenderConfig, PdfSearchDirection, PdfSearchOptions,
    Pdfium, PdfiumError, PdfiumInternalError, PdfiumLibraryBindings,
};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use std::time::Instant;

/// Hard ceiling on a single rendered page's pixel width, regardless of requested zoom.
/// See the comment at its use site in `process_page` for why this exists.
const MAX_RENDER_WIDTH: f32 = 8192.0;

/// Hard ceiling on a single rendered *bitmap's* pixel height. Unlike `MAX_RENDER_WIDTH`,
/// this doesn't cap the page's effective resolution. When a page's full height at the
/// requested scale would exceed this, `process_page` renders a vertical strip of that
/// height around `visible_center_y` instead of downscaling the whole page.
const MAX_RENDER_HEIGHT: f32 = 8192.0;

/// A page's extracted interactive text layer and link annotations, as returned by
/// `PdfRenderer::extract_text_and_links`.
type TextAndLinks = (Arc<Vec<PdfTextSpan>>, Arc<Vec<PdfLink>>);

fn rect_to_bounds(rect: &PdfRect) -> Bounds<f32> {
    Bounds {
        origin: Point {
            x: rect.left().value,
            y: rect.top().value,
        },
        size: Size {
            width: rect.width().value,
            height: rect.height().value,
        },
    }
}

/// Represents a localized piece of text on the page for the interactive text layer.
#[derive(Debug, Clone)]
pub struct PdfTextSpan {
    pub text: String,
    pub bounds: Bounds<f32>,
}

#[derive(Debug, Clone)]
pub struct PdfTarget {
    pub page_index: usize,
    pub point: Point<f32>,
    /// The page-point-space Y coordinate of the bottom of the "block" `point` belongs to, if
    /// the caller knows one, e.g. the Typst preview's source-to-PDF sync uses this so the
    /// active-jump highlight can cover a whole (possibly multi-line) source span instead of
    /// just its first line. `None` for jump sources with no such notion (bookmarks, links,
    /// search matches), where the highlight falls back to a single line's height around
    /// `point` as before.
    pub block_bottom_y: Option<f32>,
    /// The approximate height, in page points, of a single line of text at `point`. Used to
    /// pad the active-jump highlight above/below the raw baseline-to-baseline range
    /// (`point.y`..`block_bottom_y`) so it actually covers each line's glyphs instead of just
    /// their baselines. `None` falls back to a fixed guess, for jump sources with no real
    /// font-size information to derive this from.
    pub line_height: Option<f32>,
}

#[derive(Debug, Clone)]
pub enum PdfLinkAction {
    External(String),    // A URL
    Internal(PdfTarget), // Specific point on a page
}

#[derive(Debug, Clone)]
pub struct PdfLink {
    pub bounds: Bounds<f32>,
    pub action: PdfLinkAction,
}

/// A view-only, per-session rotation applied at render time. Never written back to the
/// PDF's own `/Rotate` page attribute. Whole-document rather than per-page, matching
/// Chrome/Edge's single rotate-clockwise toolbar action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PageRotation {
    #[default]
    None,
    Clockwise90,
    Clockwise180,
    Clockwise270,
}

impl PageRotation {
    /// The next rotation state after clicking "rotate clockwise" once more.
    pub fn rotated_clockwise(self) -> Self {
        match self {
            PageRotation::None => PageRotation::Clockwise90,
            PageRotation::Clockwise90 => PageRotation::Clockwise180,
            PageRotation::Clockwise180 => PageRotation::Clockwise270,
            PageRotation::Clockwise270 => PageRotation::None,
        }
    }

    /// Whether this rotation swaps a page's effective width/height for layout and rendering
    /// (a quarter-turn), as opposed to a half-turn or no rotation.
    pub fn swaps_dimensions(self) -> bool {
        matches!(self, PageRotation::Clockwise90 | PageRotation::Clockwise270)
    }

    fn to_pdfium(self) -> PdfPageRenderRotation {
        match self {
            PageRotation::None => PdfPageRenderRotation::None,
            PageRotation::Clockwise90 => PdfPageRenderRotation::Degrees90,
            PageRotation::Clockwise180 => PdfPageRenderRotation::Degrees180,
            PageRotation::Clockwise270 => PdfPageRenderRotation::Degrees270,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PageMetadata {
    pub page_number: usize,
    pub size: Size<f32>,
    pub bounds: Bounds<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PdfSearchResult {
    pub page_index: usize,
    pub bounds: Vec<Bounds<f32>>,
}

#[derive(Debug, Clone)]
pub struct PdfChapter {
    pub title: Option<String>,
    pub target: PdfTarget,
    pub children: Vec<PdfChapter>,
}

#[derive(Debug, Clone)]
pub struct DocumentMetadata {
    pub pages: Vec<PageMetadata>,
    pub chapters: Vec<PdfChapter>,
    pub page_count: usize,
    pub max_width: f32,
    pub error: Option<String>,
    /// Set when the document could not be opened because it is password-protected,
    /// either because no password was supplied or the supplied one was incorrect.
    pub needs_password: bool,
}

impl DocumentMetadata {
    pub fn from_error(error: Option<String>) -> Self {
        Self {
            pages: vec![],
            chapters: vec![],
            page_count: 0,
            max_width: 0.0,
            error,
            needs_password: false,
        }
    }

    pub fn password_required(incorrect_attempt: bool) -> Self {
        let error = if incorrect_attempt {
            "Incorrect password."
        } else {
            "This document is password protected."
        };
        Self {
            pages: vec![],
            chapters: vec![],
            page_count: 0,
            max_width: 0.0,
            error: Some(error.to_string()),
            needs_password: true,
        }
    }
}

/// Extracted assets of a PDF page
#[derive(Clone)]
pub struct PageData {
    pub image: Arc<RenderImage>,
    /// The scale this render was requested *for*. Compared against a future request's
    /// scale to decide whether this cached render is still good enough. Can be higher than
    /// `image_scale` when the page hit `MAX_RENDER_WIDTH`'s cap: rendering "again" at the
    /// same nominal `scale` wouldn't produce a sharper image (the cap is what's limiting
    /// it, not how many times it's been rendered), so treating the capped result as
    /// satisfying that nominal scale avoids re-rendering it every frame for nothing.
    pub scale: f32,
    /// The scale actually used to rasterize `image`, in pixels per page-point. Equal to
    /// `scale` unless the page hit `MAX_RENDER_WIDTH`'s cap. Use this (not `scale`) to
    /// convert `image`'s pixel dimensions back to page-point space.
    pub image_scale: f32,
    /// Page-point-space Y offset where `image` starts. Zero for an ordinary whole-page
    /// render; nonzero when the page was too tall to render whole at this scale (see
    /// `MAX_RENDER_HEIGHT`) and `image` only covers a vertical strip of it, the strip
    /// closest to whatever `visible_center_y` was requested at render time.
    pub page_offset_pt: f32,
    /// The rotation `image` was rasterized at. Compared against a future request's rotation
    /// the same way `scale` is: a cached render at the wrong rotation isn't reusable, no
    /// matter how good its resolution is.
    pub rotation: PageRotation,
    pub text_spans: Arc<Vec<PdfTextSpan>>,
    pub links: Arc<Vec<PdfLink>>,
}

impl PageData {
    /// The page-point-space height `image` covers, derived from its pixel height and the
    /// scale it was actually rendered at.
    pub fn covered_height_pt(&self) -> f32 {
        self.image.size(0).height.0 as f32 / self.image_scale
    }

    /// Whether `image` already covers page-point-space Y coordinate `y_pt`, i.e. whether a
    /// caller interested in that point can reuse this render as-is rather than requesting a
    /// new strip centered elsewhere. Always true for an unstripped (ordinary, whole-page)
    /// render, since its covered range is the entire page.
    pub fn covers_page_y(&self, y_pt: f32) -> bool {
        y_pt >= self.page_offset_pt && y_pt <= self.page_offset_pt + self.covered_height_pt()
    }
}

impl std::fmt::Debug for PageData {
    // `RenderImage` doesn't implement `Debug`, so this is written by hand rather than derived.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PageData")
            .field("scale", &self.scale)
            .field("image_scale", &self.image_scale)
            .field("page_offset_pt", &self.page_offset_pt)
            .field("text_spans", &self.text_spans.len())
            .field("links", &self.links.len())
            .finish_non_exhaustive()
    }
}

/// Holds the process-lifetime PDFium binding together with a cache of already-opened
/// documents, so that repeated page renders and searches against the same document don't
/// each have to re-parse it from scratch. All access is serialized behind a single mutex
/// because PDFium itself is not safe to call from multiple threads concurrently, even
/// across unrelated documents.
struct PdfiumState {
    pdfium: &'static Pdfium,
    /// Keyed by `Pdf::id` (a content hash), so opening the same document twice, even in
    /// separate `Pdf` instances, shares a single parsed `PdfDocument`.
    documents: HashMap<u64, PdfDocument<'static>>,
}

/// Global PDFium instance to share across all GPUI background tasks
static PDFIUM: OnceCell<Mutex<PdfiumState>> = OnceCell::new();

/// Document IDs that `evict_document` couldn't remove immediately because the `PDFIUM` mutex
/// was held elsewhere. `Drop` can't await the async mutex, so that eviction is a best-effort
/// `try_lock`; without this queue, losing that race means the entry, and the `PdfDocument` it
/// holds, would stay cached for the rest of the process. Drained the next time anyone actually
/// acquires the mutex (see `lock_pdfium_state`), so eviction is always eventually applied.
static PENDING_EVICTIONS: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());

/// Serializes every test in this crate that performs real PDFium FFI work (across `pdf.rs`,
/// `pdf_renderer.rs`, and `pdf_viewer.rs`). Rust's test harness runs `#[test]`/`#[gpui::test]`
/// functions concurrently on separate threads by default; without this, they all queue up on
/// the single `PDFIUM` async mutex above at once, and a test can end up waiting long enough to
/// trip gpui's 15-second "parking" timeout even though nothing is actually deadlocked. This
/// lock has nothing to do with correctness of the production code. It's only about keeping the test
/// suite's real (non-simulated) FFI work from contending with itself.
#[cfg(test)]
pub(crate) static PDFIUM_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Automatically binds to system libpdfium.dylib / .so / .dll
fn get_bindings() -> Result<Box<dyn PdfiumLibraryBindings>> {
    let bindings = Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path("./"))
        .or_else(|_| {
            Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path("./pdfium/"))
        })
        .or_else(|_| {
            Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(
                "/opt/homebrew/lib",
            ))
        })
        .or_else(|_| {
            Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(
                "/usr/local/lib",
            ))
        })
        .or_else(|_| Pdfium::bind_to_system_library())
        .map_err(|e| anyhow!("Failed to bind to PDFium: {:?}", e))?;

    Ok(bindings)
}

fn get_pdfium() -> Result<&'static Mutex<PdfiumState>> {
    PDFIUM.get_or_try_init(|| {
        let bindings = get_bindings()?;

        // Leaked deliberately: PDFium has no shutdown story we hook into, and leaking here
        // is what lets cached `PdfDocument`s (which borrow this reference) live for as long
        // as they're needed instead of being tied to the lifetime of a `MutexGuard`.
        let pdfium: &'static Pdfium = Box::leak(Box::new(Pdfium::new(bindings)));

        Ok(Mutex::new(PdfiumState {
            pdfium,
            documents: HashMap::new(),
        }))
    })
}

/// Returns the cached, already-parsed document for `id`, opening and inserting it on first
/// access. Uses `load_pdf_from_byte_vec` (rather than `load_pdf_from_byte_slice`) so the
/// returned `PdfDocument` only borrows the process-lifetime `Pdfium` binding, not `bytes` -
/// letting it safely outlive the call that opened it and be reused by later renders/searches.
fn open_document<'s>(
    state: &'s mut PdfiumState,
    id: u64,
    bytes: &Arc<Vec<u8>>,
    password: Option<&str>,
) -> Result<&'s PdfDocument<'static>, PdfiumError> {
    match state.documents.entry(id) {
        Entry::Occupied(entry) => Ok(entry.into_mut()),
        Entry::Vacant(entry) => {
            let document = state
                .pdfium
                .load_pdf_from_byte_vec((**bytes).clone(), password)?;
            Ok(entry.insert(document))
        }
    }
}

/// Evicts a document from the shared cache, e.g. once the last `Arc<Pdf>` referencing it is
/// dropped. Uses `try_lock` rather than blocking, since `Drop` can't await the async mutex; if
/// the mutex is contended, the id is queued in `PENDING_EVICTIONS` instead so it's still
/// guaranteed to be evicted (see `lock_pdfium_state`) rather than left cached indefinitely.
pub fn evict_document(id: u64) {
    if let Some(state_lock) = PDFIUM.get()
        && let Some(mut state) = state_lock.try_lock()
    {
        state.documents.remove(&id);
        return;
    }
    PENDING_EVICTIONS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .push(id);
}

/// Locks the shared `PdfiumState`, applying any evictions that `evict_document` queued while it
/// couldn't acquire the lock itself. Every call site that needs the mutex should go through this
/// rather than locking `PDFIUM` directly, so a document dropped mid-render doesn't leak forever.
async fn lock_pdfium_state() -> Result<futures::lock::MutexGuard<'static, PdfiumState>> {
    let mut state = get_pdfium()?.lock().await;
    let pending = std::mem::take(
        &mut *PENDING_EVICTIONS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
    );
    for id in pending {
        state.documents.remove(&id);
    }
    Ok(state)
}

pub struct PdfRenderer {
    id: u64,
    pdf_bytes: Arc<Vec<u8>>,
    password: Option<String>,
}

impl PdfRenderer {
    pub fn new(id: u64, pdf_bytes: Arc<Vec<u8>>, password: Option<String>) -> Self {
        Self {
            id,
            pdf_bytes,
            password,
        }
    }

    /// Extracts the dimensions of every page without rendering them.
    pub async fn get_metadata(
        &self,
        pages_max_width: Option<(Vec<PageMetadata>, f32)>,
    ) -> Result<DocumentMetadata> {
        let start = Instant::now();

        let mut pdfium_state = lock_pdfium_state().await?;

        let document = match open_document(
            &mut pdfium_state,
            self.id,
            &self.pdf_bytes,
            self.password.as_deref(),
        ) {
            Ok(document) => document,
            Err(PdfiumError::PdfiumLibraryInternalError(PdfiumInternalError::PasswordError)) => {
                return Ok(DocumentMetadata::password_required(
                    self.password.is_some(),
                ));
            }
            Err(e) => return Err(anyhow!("Failed to read PDF bytes: {:?}", e)),
        };

        log::info!("Document parsed in {:?}", start.elapsed());

        let (pages, max_width) = pages_max_width.unwrap_or_else(|| {
            let start = Instant::now();
            let mut max_width = 0.0f32;
            let pages: Vec<PageMetadata> = document
                .pages()
                .iter()
                .enumerate()
                .map(|(i, page)| {
                    // PDFium returns sizes in points (1/72 of an inch)
                    let page_bounds = page
                        .boundaries()
                        .crop()
                        .or_else(|_| page.boundaries().media());
                    let width = page.width().value;
                    max_width = max_width.max(width);
                    let height = page.height().value;
                    PageMetadata {
                        page_number: i,
                        size: Size { width, height },
                        bounds: page_bounds
                            .map(|b| rect_to_bounds(&b.bounds))
                            .unwrap_or(Bounds {
                                origin: Point { x: 0.0, y: height },
                                size: Size { width, height },
                            }),
                    }
                })
                .collect();
            log::info!(
                "Page sizes extracted for {} pages in {:?}",
                document.pages().len(),
                start.elapsed()
            );
            (pages, max_width)
        });

        let start = Instant::now();

        let mut chapters = Vec::new();
        if let Some(root_bookmark) = document.bookmarks().root() {
            chapters.push(Self::parse_bookmark(&root_bookmark, document, &pages));
            for sibling in root_bookmark.iter_siblings() {
                let chapter = Self::parse_bookmark(&sibling, document, &pages);
                chapters.push(chapter);
            }
        }

        log::info!(
            "Chapters extracted for {} pages in {:?}",
            document.pages().len(),
            start.elapsed()
        );

        let page_count = pages.len();
        let metadata = DocumentMetadata {
            pages,
            chapters,
            page_count,
            max_width,
            error: None,
            needs_password: false,
        };

        Ok(metadata)
    }

    /// Recursive helper to parse a bookmark and all its children
    #[allow(clippy::only_used_in_recursion)]
    fn parse_bookmark(
        bookmark: &PdfBookmark,
        document: &PdfDocument,
        pages: &[PageMetadata],
    ) -> PdfChapter {
        let mut page_index = 0;
        let mut point = Point { x: 0.0, y: 0.0 };

        // The target may be attached to the bookmark directly. For documents (often
        // Acrobat-authored) that route every outline entry through an explicit action,
        // it's wrapped in a `GoToDestinationInSameDocument` action instead. Same target,
        // one extra level of indirection. Other action types (opening a different
        // document, launching a file, etc.) aren't internal navigation and are left
        // unresolved.
        let action = bookmark.action();
        let destination = bookmark.destination().or_else(|| {
            action
                .as_ref()
                .and_then(|action| action.as_local_destination_action())
                .and_then(|local| local.destination().ok())
        });

        if let Some(destination) = destination
            && let Some((idx, target_point)) = Self::resolve_destination_target(&destination, pages)
        {
            // A corrupt or adversarial document can point at a page index that doesn't
            // exist; fall back to the default (page 0, origin) rather than panicking.
            page_index = idx;
            point = target_point;
        }

        // Recursively parse children
        let mut children = Vec::new();
        for child in bookmark.iter_direct_children() {
            let child_chapter = Self::parse_bookmark(&child, document, pages);
            children.push(child_chapter);
        }

        PdfChapter {
            title: bookmark.title(),
            target: PdfTarget {
                page_index,
                point,
                block_bottom_y: None,
                line_height: None,
            },
            children,
        }
    }

    fn get_bounds(rect: &PdfRect, page_bounds: &Bounds<f32>) -> Bounds<f32> {
        Bounds {
            origin: Point {
                x: rect.left().value - page_bounds.origin.x,
                y: page_bounds.origin.y - rect.top().value,
            },
            size: Size {
                width: rect.width().value,
                height: rect.height().value,
            },
        }
    }

    // Get point of a destination
    fn get_point(destination: &PdfDestination, page_bounds: &Bounds<f32>) -> Point<f32> {
        let bounds_left = page_bounds.origin.x;
        let bounds_top = page_bounds.origin.y;

        let mut target_left = 0.0;
        let mut target_top = 0.0;

        if let Ok(view_settings) = destination.view_settings() {
            // Import the enum variants to keep the match arms clean
            use pdfium_render::prelude::PdfDestinationViewSettings::*;

            match view_settings {
                // Tuple variant: (Left, Top, Zoom)
                SpecificCoordinatesAndZoom(left, top, _zoom) => {
                    if let Some(l) = left {
                        target_left = l.value - bounds_left;
                    }
                    if let Some(t) = top {
                        target_top = bounds_top - t.value;
                    }
                }
                // Tuple variant: (Top)
                FitPageHorizontallyToWindow(top) | FitBoundsHorizontallyToWindow(top) => {
                    if let Some(t) = top {
                        target_top = bounds_top - t.value;
                    }
                }
                // Tuple variant: (Left)
                FitPageVerticallyToWindow(left) | FitBoundsVerticallyToWindow(left) => {
                    if let Some(l) = left {
                        target_left = l.value - bounds_left;
                    }
                }
                // Tuple variant: (PdfRect)
                FitPageToRectangle(rect) => {
                    target_left = rect.left().value - bounds_left;
                    target_top = bounds_top - rect.top().value;
                }
                // For anything else, retain our top-left defaults
                FitPageToWindow | FitBoundsToWindow | Unknown => {}
            }
        }

        Point {
            x: target_left,
            y: target_top,
        }
    }

    /// Resolves a `PdfDestination`'s target page index and its point within that page's
    /// own coordinate space. Returns `None` if the destination points at a page that
    /// doesn't actually exist in this document (a corrupt/adversarial document, or a
    /// bookmark/link whose target page was later deleted), rather than panicking on an
    /// out-of-bounds index.
    fn resolve_destination_target(
        destination: &PdfDestination,
        pages: &[PageMetadata],
    ) -> Option<(usize, Point<f32>)> {
        let page_index = destination.page_index().ok()? as usize;
        let page_metadata = pages.get(page_index)?;
        Some((page_index, Self::get_point(destination, &page_metadata.bounds)))
    }

    fn get_link(
        annot: &PdfPageAnnotation,
        link: &PdfiumLink,
        page_bounds: &Bounds<f32>,
        pages: &[PageMetadata],
    ) -> Option<PdfLink> {
        if let PdfPageAnnotation::Link(link_annot) = &annot
            && let Ok(bounds) = link_annot.bounds()
        {
            let action = link.action();

            if let Some(uri) = action
                .as_ref()
                .and_then(|action| action.as_uri_action())
                .and_then(|uri_action| uri_action.uri().ok())
            {
                return Some(PdfLink {
                    bounds: Self::get_bounds(&bounds, page_bounds),
                    action: PdfLinkAction::External(uri),
                });
            }

            // The destination may be attached to the link directly. For documents
            // (often Acrobat-authored) that route every link through an explicit action,
            // it's wrapped in a `GoToDestinationInSameDocument` action instead.
            let destination = link.destination().or_else(|| {
                action
                    .as_ref()
                    .and_then(|action| action.as_local_destination_action())
                    .and_then(|local| local.destination().ok())
            });

            if let Some(destination) = destination
                && let Some((target_page_index, point)) =
                    Self::resolve_destination_target(&destination, pages)
            {
                return Some(PdfLink {
                    bounds: Self::get_bounds(&bounds, page_bounds),
                    action: PdfLinkAction::Internal(PdfTarget {
                        page_index: target_page_index,
                        point,
                        block_bottom_y: None,
                        line_height: None,
                    }),
                });
            }
        }

        None
    }

    // Processes a specific page using PDFium
    pub async fn process_page(
        &self,
        page_index: usize,
        scale: f32,
        // Page-point-space Y coordinate the caller most wants rendered, e.g. the vertical
        // center of what's currently visible. Only matters when the page needs to be
        // rendered as a strip (see MAX_RENDER_HEIGHT below); ignored otherwise.
        visible_center_y: f32,
        // View-only rotation, applied at render time only. Never touches the page's own
        // `/Rotate` attribute. text_spans/links extracted below are unaffected by it (they
        // stay in the page's natural coordinate space), which is why `prev_data` is reused
        // for them regardless of what rotation it was rendered at.
        rotation: PageRotation,
        prev_data: Option<PageData>,
        metadata: Arc<DocumentMetadata>,
    ) -> Result<PageData> {
        let start = Instant::now();

        let mut pdfium_state = lock_pdfium_state().await?;

        let document = open_document(
            &mut pdfium_state,
            self.id,
            &self.pdf_bytes,
            self.password.as_deref(),
        )
        .map_err(|e| anyhow!("Failed to read PDF bytes: {:?}", e))?;

        let page = document
            .pages()
            .get(page_index as i32)
            .map_err(|e| anyhow!("Page out of bounds: {:?}", e))?;

        // A quarter-turn swaps which of the page's own dimensions ends up as the rendered
        // output's width vs. height; a half-turn or no rotation doesn't.
        let (source_width_pt, source_height_pt) = if rotation.swaps_dimensions() {
            (page.height().value, page.width().value)
        } else {
            (page.width().value, page.height().value)
        };

        // Cap the rendered bitmap's width instead of ever asking pdfium for an arbitrarily
        // large one at high zoom: a several-thousand-point-wide page at max zoom could
        // otherwise ask for a multi-gigabyte allocation. Height following proportionally
        // from this would still be uncapped for a narrow-but-very-tall page, so height is
        // handled separately below.
        let target_width = ((source_width_pt * scale).round()).min(MAX_RENDER_WIDTH) as i32;
        let effective_scale = target_width as f32 / source_width_pt;
        let full_height_px = source_height_pt * effective_scale;

        // Chromium's PDF viewer tiles oversized pages so only the tiles inside the current
        // viewport are ever rendered/held. We render a single vertical strip instead of a
        // full 2D tile grid: when the page is too tall to render whole at this scale, render
        // just a MAX_RENDER_HEIGHT-tall strip around `visible_center_y`, positioned within
        // the page by `page_offset_pt` (see PageData) rather than the whole page downscaled.
        let (target_height, page_offset_pt) = if full_height_px > MAX_RENDER_HEIGHT {
            let strip_height_pt = MAX_RENDER_HEIGHT / effective_scale;
            let max_offset_pt = (source_height_pt - strip_height_pt).max(0.0);
            let offset_pt =
                (visible_center_y - strip_height_pt / 2.0).clamp(0.0, max_offset_pt);
            (MAX_RENDER_HEIGHT.round() as i32, offset_pt)
        } else {
            (full_height_px.round() as i32, 0.0)
        };

        let mut bitmap = PdfBitmap::empty(target_width, target_height, PdfBitmapFormat::default())
            .map_err(|e| anyhow!("Failed to allocate render target: {:?}", e))?;

        let render_config = PdfRenderConfig::new()
            // Both width and height are set explicitly (rather than relying on
            // set_target_width alone to infer an aspect-preserving height) so the output
            // size reflects source_width_pt/source_height_pt, the *post-rotation* logical
            // dimensions, rather than pdfium-render's own aspect-ratio inference, which
            // only accounts for rotation via a separate maximum-width/height constraint path
            // this crate doesn't use.
            .set_target_width(target_width)
            .set_target_height((source_height_pt * effective_scale).round() as i32)
            .set_origin(0, -((page_offset_pt * effective_scale).round() as i32))
            .set_clear_color(PdfColor::WHITE)
            .rotate(rotation.to_pdfium(), false);

        page.render_into_bitmap_with_config(&mut bitmap, &render_config)
            .map_err(|e| anyhow!("Render failed: {:?}", e))?;

        // gpui's `RenderImage` expects BGRA bytes. `as_rgba_bytes()` normalizes whatever
        // pixel format pdfium actually rendered (Gray/BGR/BGRA/BGRx all differ) into true
        // RGBA; swapping R and B gets us the rest of the way to BGRA. This is still one
        // lightweight in-memory pass instead of the BMP encode+decode round trip this used
        // to go through (encode via `image::write_to`, then gpui's asset pipeline decoding
        // the very bytes we just wrote right back into pixels).
        let width = bitmap.width() as u32;
        let height = bitmap.height() as u32;
        let mut bgra_bytes = bitmap.as_rgba_bytes();
        for pixel in bgra_bytes.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }

        let buffer = RgbaImage::from_raw(width, height, bgra_bytes)
            .ok_or_else(|| anyhow!("Rendered bitmap buffer size didn't match its dimensions"))?;
        let gpui_image = Arc::new(RenderImage::new(vec![Frame::new(buffer)]));

        let render_text = prev_data.is_none();
        let (text_spans, links) = if let Some(prev_data) = prev_data {
            (prev_data.text_spans, prev_data.links)
        } else {
            Self::extract_text_and_links(&page, page_index, &metadata)?
        };

        log::info!(
            "Rendered page {} (scale {}, with text? {:?}) in {:?}",
            page_index + 1,
            scale,
            render_text,
            start.elapsed()
        );

        Ok(PageData {
            image: gpui_image,
            scale,
            image_scale: effective_scale,
            page_offset_pt,
            rotation,
            text_spans,
            links,
        })
    }

    /// Extracts a page's interactive text layer and link annotations, the "just walk what
    /// PDFium already parsed" half of `process_page`, without any of its bitmap
    /// rasterization. Shared by `process_page` (which needs both, once per page, alongside
    /// the render) and `extract_text_only` (which needs just this, cheaply, for pages that
    /// don't otherwise need to be rasterized at all, see its doc comment).
    fn extract_text_and_links(
        page: &PdfPage,
        page_index: usize,
        metadata: &DocumentMetadata,
    ) -> Result<TextAndLinks> {
        let page_metadata = metadata
            .pages
            .get(page_index)
            .ok_or_else(|| anyhow!("Page metadata missing for page {}", page_index))?;
        let page_bounds = page_metadata.bounds;

        // TODO: dont propagate error, but add error logging

        let mut text_spans_vec = Vec::new();
        if let Ok(text_page) = page.text() {
            for text_char in text_page.chars().iter() {
                if let Ok(bounds) = text_char.loose_bounds()
                    && let Some(text) = text_char.unicode_string()
                {
                    text_spans_vec.push(PdfTextSpan {
                        text,
                        bounds: PdfRenderer::get_bounds(&bounds, &page_bounds),
                    });
                }
            }
        }
        // TODO: add error logging

        let link_annotations = page
            .annotations()
            .iter()
            .filter(|annot| matches!(annot, PdfPageAnnotation::Link(_)));

        let page_links = page.links().iter();

        let mut links_vec = Vec::new();

        for (annot, link_action) in link_annotations.zip(page_links) {
            if let Some(link) = PdfRenderer::get_link(&annot, &link_action, &page_bounds, &metadata.pages) {
                links_vec.push(link)
            }
        }

        Ok((Arc::new(text_spans_vec), Arc::new(links_vec)))
    }

    /// Extracts `page_index`'s text layer without rasterizing a bitmap at all. Much cheaper
    /// than `process_page`, which this crate's full-document operations (namely Select All)
    /// need: reading every page's text shouldn't require rendering every page's image first,
    /// the same way Chrome/Edge's own PDFium-based viewer decouples (cheap) text extraction
    /// from (expensive) rasterization.
    pub async fn extract_text_only(
        &self,
        page_index: usize,
        metadata: Arc<DocumentMetadata>,
    ) -> Result<Arc<Vec<PdfTextSpan>>> {
        let mut pdfium_state = lock_pdfium_state().await?;

        let document = open_document(
            &mut pdfium_state,
            self.id,
            &self.pdf_bytes,
            self.password.as_deref(),
        )
        .map_err(|e| anyhow!("Failed to read PDF bytes: {:?}", e))?;

        let page = document
            .pages()
            .get(page_index as i32)
            .map_err(|e| anyhow!("Page out of bounds: {:?}", e))?;

        let (text_spans, _links) = Self::extract_text_and_links(&page, page_index, &metadata)?;
        Ok(text_spans)
    }

    /// Search the document for a specific text string using PDFium
    pub async fn search_document(
        &self,
        query: &str,
        match_case: bool,
        metadata: Arc<DocumentMetadata>,
    ) -> Result<Vec<PdfSearchResult>> {
        let start = Instant::now();

        let mut pdfium_state = lock_pdfium_state().await?;

        let document = open_document(
            &mut pdfium_state,
            self.id,
            &self.pdf_bytes,
            self.password.as_deref(),
        )
        .map_err(|e| anyhow!("Failed to read PDF bytes: {:?}", e))?;

        let mut results = Vec::new();
        let search_options = PdfSearchOptions::new().match_case(match_case);

        for (page_index, page) in document.pages().iter().enumerate() {
            let Some(page_metadata) = metadata.pages.get(page_index) else {
                continue;
            };
            let page_bounds = page_metadata.bounds;
            if let Ok(text_page) = page.text()
                && let Ok(search) = text_page.search(query, &search_options)
            {
                for match_segments in search.iter(PdfSearchDirection::SearchForward) {
                    let mut bounds_vec = Vec::new();

                    for segment in match_segments.iter() {
                        // let rect = segment.bounds();
                        // bounds_vec.push(Self::get_bounds(&rect, &page_bounds));

                        let mut min_left = f32::MAX;
                        let mut max_right = f32::MIN;
                        let mut min_bottom = f32::MAX;
                        let mut max_top = f32::MIN;
                        let mut has_bounds = false;

                        let text = segment.text();
                        let Ok(segment_chars) = segment.chars() else {
                            continue;
                        };
                        for text_char in segment_chars.iter().take(text.len()) {
                            if let Ok(rect) = text_char.loose_bounds() {
                                min_left = min_left.min(rect.left().value);
                                max_right = max_right.max(rect.right().value);
                                min_bottom = min_bottom.min(rect.bottom().value);
                                max_top = max_top.max(rect.top().value);
                                has_bounds = true;
                            }
                        }

                        if has_bounds {
                            bounds_vec.push(Bounds {
                                origin: Point {
                                    x: min_left - page_bounds.origin.x,
                                    y: page_bounds.origin.y - max_top,
                                },
                                size: Size {
                                    width: max_right - min_left,
                                    height: max_top - min_bottom,
                                },
                            });
                        }
                    }

                    if !bounds_vec.is_empty() {
                        results.push(PdfSearchResult {
                            page_index,
                            bounds: bounds_vec,
                        });
                    }
                }
            }
        }

        log::info!("Searched in {:?}", start.elapsed());

        Ok(results)
    }
}

impl PdfChapter {
    /// Recursively formats the chapter and its children into a readable string buffer.
    pub fn format_tree(&self, depth: usize, out: &mut String) {
        // Create an indentation string (e.g., 0 spaces, 2 spaces, 4 spaces...)
        let indent = "  ".repeat(depth);

        let title = self.title.as_deref().unwrap_or("<Untitled>");
        let page = self.target.page_index;
        let x = self.target.point.x;
        let y = self.target.point.y;

        // Append the current chapter's info
        out.push_str(&format!(
            "{}- {} (Page: {}, x: {:.2}, y: {:.2})\n",
            indent, title, page, x, y
        ));

        // Recursively format all children, increasing the depth by 1
        for child in &self.children {
            child.format_tree(depth + 1, out);
        }
    }
}

impl DocumentMetadata {
    /// Helper to print the entire bookmark tree to the console/logs
    pub fn debug_log_bookmarks(&self) {
        if self.chapters.is_empty() {
            log::info!("No bookmarks found in this PDF.");
            return;
        }
        let mut output = String::from("\n--- PDF Bookmark Tree ---\n");

        for chapter in self.chapters.iter() {
            // Start formatting from depth 0
            chapter.format_tree(0, &mut output);
        }

        // Log the final built string
        log::info!("{}", output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // All tests in this module load the same fixture, so they'd otherwise share one entry
    // in the process-global `DOCUMENT_CACHE` (correct behavior in the app, where a document
    // unlocked once stays unlocked, but it makes the tests interfere with each other when
    // Rust runs them concurrently). Serialize them and evict before/after each so every test
    // starts from, and leaves, a clean cache. This also happens to be the same crate-wide
    // lock other pdfium-touching tests use to avoid contending on the global `PDFIUM` mutex.
    use super::PDFIUM_TEST_LOCK as TEST_LOCK;

    fn fixture(name: &str) -> (u64, Arc<Vec<u8>>) {
        let path = format!("{}/test_fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let id = gpui::hash(&bytes);
        (id, Arc::new(bytes))
    }

    #[test]
    fn missing_password_reports_needs_password_instead_of_erroring() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (id, bytes) = fixture("password_protected.pdf");
        evict_document(id);
        let renderer = PdfRenderer::new(id, bytes, None);

        let metadata = futures::executor::block_on(renderer.get_metadata(None)).unwrap();

        assert!(metadata.needs_password);
        assert_eq!(metadata.page_count, 0);
        evict_document(id);
    }

    #[test]
    fn wrong_password_is_reported_as_an_incorrect_attempt() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (id, bytes) = fixture("password_protected.pdf");
        evict_document(id);
        let renderer = PdfRenderer::new(id, bytes, Some("not-the-password".to_string()));

        let metadata = futures::executor::block_on(renderer.get_metadata(None)).unwrap();

        assert!(metadata.needs_password);
        assert_eq!(metadata.error.as_deref(), Some("Incorrect password."));
        evict_document(id);
    }

    #[test]
    fn correct_password_unlocks_and_caches_the_document() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (id, bytes) = fixture("password_protected.pdf");
        evict_document(id);
        let renderer = PdfRenderer::new(id, bytes.clone(), Some("hunter2".to_string()));

        let metadata = futures::executor::block_on(renderer.get_metadata(None)).unwrap();
        assert!(!metadata.needs_password);
        assert_eq!(metadata.page_count, 1);

        // A second renderer for the same content, *without* the password, should still be
        // able to render a page: the already-unlocked document is shared via the cache
        // rather than each renderer re-opening (and re-authenticating) the file itself.
        let cached_renderer = PdfRenderer::new(id, bytes, None);
        let page = futures::executor::block_on(cached_renderer.process_page(
            0,
            1.0,
            0.0,
            PageRotation::None,
            None,
            Arc::new(metadata),
        ));
        assert!(page.is_ok(), "expected cached render to succeed: {page:?}");

        evict_document(id);
    }

    #[test]
    fn evict_document_queues_for_later_when_the_mutex_is_contended() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (id, bytes) = fixture("solid_red.pdf");
        evict_document(id);
        let renderer = PdfRenderer::new(id, bytes, None);

        futures::executor::block_on(renderer.get_metadata(None)).unwrap();
        assert!(
            get_pdfium()
                .unwrap()
                .try_lock()
                .unwrap()
                .documents
                .contains_key(&id),
            "expected the document to be cached after get_metadata"
        );

        // Simulate `Pdf::drop` racing a render: hold the mutex ourselves so
        // `evict_document`'s `try_lock` loses the race, and confirm it queues the id
        // instead of silently leaving it cached forever.
        let held = futures::executor::block_on(get_pdfium().unwrap().lock());
        evict_document(id);
        assert!(
            held.documents.contains_key(&id),
            "the document must still be cached while the mutex is held"
        );
        drop(held);

        // The next real lock acquisition must apply the deferred eviction.
        futures::executor::block_on(lock_pdfium_state()).unwrap();
        assert!(
            !get_pdfium()
                .unwrap()
                .try_lock()
                .unwrap()
                .documents
                .contains_key(&id),
            "deferred eviction should have been applied once the mutex was next acquired"
        );
    }

    #[test]
    fn rendered_page_uses_gpuis_native_bgra_byte_order() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (id, bytes) = fixture("solid_red.pdf");
        evict_document(id);
        let renderer = PdfRenderer::new(id, bytes, None);

        let metadata = futures::executor::block_on(renderer.get_metadata(None)).unwrap();
        assert_eq!(metadata.page_count, 1);

        let page = futures::executor::block_on(
            renderer.process_page(0, 1.0, 0.0, PageRotation::None, None, Arc::new(metadata)),
        )
        .unwrap();

        let size = page.image.size(0);
        let (width, height) = (size.width.0 as usize, size.height.0 as usize);
        assert!(width > 0 && height > 0);

        let buffer = page.image.as_bytes(0).unwrap();
        // Sample the center pixel to stay clear of any antialiased edges.
        let offset = ((height / 2) * width + width / 2) * 4;
        let pixel = &buffer[offset..offset + 4];
        let [b, g, r, a] = [pixel[0], pixel[1], pixel[2], pixel[3]];

        // process_page builds the RenderImage directly from pdfium's normalized RGBA bytes
        // plus a manual R/B swap, rather than routing through a BMP encode/decode round
        // trip. A solid-red PDF fill is exactly the case that would come out wrong, e.g.
        // blue instead of red, if that swap were missing, backwards, or applied twice.
        assert!(r > 200, "expected a strong red channel, got {pixel:?}");
        assert!(b < 50, "expected a near-zero blue channel, got {pixel:?}");
        assert!(g < 50, "expected a near-zero green channel, got {pixel:?}");
        assert!(a > 200, "expected an opaque alpha channel, got {pixel:?}");

        evict_document(id);
    }

    #[test]
    fn render_target_width_is_capped_at_extreme_zoom() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (id, bytes) = fixture("solid_red.pdf");
        evict_document(id);
        let renderer = PdfRenderer::new(id, bytes, None);

        let metadata = futures::executor::block_on(renderer.get_metadata(None)).unwrap();

        // The fixture page is 100pt wide; at scale 1000 an uncapped render would ask
        // pdfium for a 100,000px-wide bitmap. It should come back capped instead of
        // hanging or failing.
        let page = futures::executor::block_on(renderer.process_page(
            0,
            1000.0,
            0.0,
            PageRotation::None,
            None,
            Arc::new(metadata),
        ))
        .unwrap();

        let width = page.image.size(0).width.0;
        assert!(
            width as f32 <= MAX_RENDER_WIDTH,
            "expected width capped at {MAX_RENDER_WIDTH}, got {width}"
        );

        evict_document(id);
    }

    #[test]
    fn ordinary_pages_are_not_stripped() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (id, bytes) = fixture("solid_red.pdf");
        evict_document(id);
        let renderer = PdfRenderer::new(id, bytes, None);

        let metadata = futures::executor::block_on(renderer.get_metadata(None)).unwrap();
        let page_height_pt = metadata.pages[0].size.height;

        let page = futures::executor::block_on(renderer.process_page(
            0,
            1.0,
            0.0,
            PageRotation::None,
            None,
            Arc::new(metadata),
        ))
        .unwrap();

        assert_eq!(page.page_offset_pt, 0.0);
        assert!(page.covers_page_y(0.0));
        assert!(page.covers_page_y(page_height_pt));

        evict_document(id);
    }

    #[test]
    fn tall_pages_render_a_strip_around_the_requested_center() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (id, bytes) = fixture("tall_page.pdf");
        evict_document(id);
        let renderer = PdfRenderer::new(id, bytes, None);

        let metadata = futures::executor::block_on(renderer.get_metadata(None)).unwrap();
        let page_height_pt = metadata.pages[0].size.height;
        assert!(
            page_height_pt > MAX_RENDER_HEIGHT,
            "fixture should be tall enough to exceed the render-height cap at scale 1.0"
        );

        let top_strip = futures::executor::block_on(renderer.process_page(
            0,
            1.0,
            0.0,
            PageRotation::None,
            None,
            Arc::new(metadata.clone()),
        ))
        .unwrap();
        let bottom_strip = futures::executor::block_on(renderer.process_page(
            0,
            1.0,
            page_height_pt,
            PageRotation::None,
            None,
            Arc::new(metadata),
        ))
        .unwrap();

        // Both strips are capped rather than covering the (uncapped) full page height.
        let strip_height = top_strip.image.size(0).height.0 as f32;
        assert!(strip_height <= MAX_RENDER_HEIGHT);
        assert!(strip_height < page_height_pt);
        assert_eq!(strip_height, bottom_strip.image.size(0).height.0 as f32);

        // Requesting near the top vs. near the bottom should land at different offsets,
        // each covering the point that was actually asked for and not the other one.
        assert!(top_strip.page_offset_pt < bottom_strip.page_offset_pt);
        assert!(top_strip.covers_page_y(0.0));
        assert!(bottom_strip.covers_page_y(page_height_pt));
        assert!(!top_strip.covers_page_y(page_height_pt));
        assert!(!bottom_strip.covers_page_y(0.0));

        // And the actual rendered pixels should differ, proving the two requests produced
        // genuinely different content, not the same (clamped) strip both times.
        assert_ne!(
            top_strip.image.as_bytes(0),
            bottom_strip.image.as_bytes(0),
            "expected different page regions to render different content"
        );

        evict_document(id);
    }

    #[test]
    fn page_annotations_are_rendered_by_default() {
        // PdfRenderConfig::new() defaults do_set_flag_render_annotations to true and
        // process_page never calls .render_annotations(false), so a PDF's own annotations
        // (highlights, stamps, ink, ...) should already show up in the rendered bitmap with
        // no extra work. This fixture has a solid blue Rectangle *annotation* (not page
        // content) covering the whole page, so an unannotated render would come back white.
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (id, bytes) = fixture("annotated_page.pdf");
        evict_document(id);
        let renderer = PdfRenderer::new(id, bytes, None);

        let metadata = futures::executor::block_on(renderer.get_metadata(None)).unwrap();
        let page = futures::executor::block_on(renderer.process_page(
            0,
            1.0,
            0.0,
            PageRotation::None,
            None,
            Arc::new(metadata),
        ))
        .unwrap();

        let size = page.image.size(0);
        let (width, height) = (size.width.0 as usize, size.height.0 as usize);
        let buffer = page.image.as_bytes(0).unwrap();
        let offset = ((height / 2) * width + width / 2) * 4;
        let pixel = &buffer[offset..offset + 4];
        let [b, g, r, _a] = [pixel[0], pixel[1], pixel[2], pixel[3]];

        assert!(
            b > 200 && r < 50 && g < 50,
            "expected the annotation's solid blue fill at the page center, got {pixel:?}; \
             annotation rendering may be disabled"
        );

        evict_document(id);
    }

    #[test]
    fn clockwise_rotation_swaps_output_dimensions_and_actually_rotates_the_pixels() {
        // A 200x100pt page with its left half (x in [0,100], full height) filled solid blue,
        // the right half left white. A real 90-degree-clockwise rotation, not just a
        // dimension swap, moves that left-half content to the top half of the (now
        // 100x200) output: rotating a photo 90 degrees clockwise turns its left edge into
        // its new top edge. If PageRotation::Clockwise90 were wired up wrong (e.g. rotating
        // the wrong direction, or resizing the bitmap without actually telling pdfium to
        // rotate), this would come back either the wrong shape or blue in the wrong half.
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (id, bytes) = fixture("left_half_blue.pdf");
        evict_document(id);
        let renderer = PdfRenderer::new(id, bytes, None);

        let metadata = futures::executor::block_on(renderer.get_metadata(None)).unwrap();
        assert_eq!(metadata.pages[0].size, Size { width: 200.0, height: 100.0 });

        let page = futures::executor::block_on(renderer.process_page(
            0,
            1.0,
            0.0,
            PageRotation::Clockwise90,
            None,
            Arc::new(metadata),
        ))
        .unwrap();

        let size = page.image.size(0);
        let (width, height) = (size.width.0 as usize, size.height.0 as usize);
        assert_eq!(
            (width, height),
            (100, 200),
            "a quarter-turn should swap the rendered bitmap's width and height"
        );

        let buffer = page.image.as_bytes(0).unwrap();
        let sample = |x: usize, y: usize| -> [u8; 4] {
            let offset = (y * width + x) * 4;
            let pixel = &buffer[offset..offset + 4];
            [pixel[0], pixel[1], pixel[2], pixel[3]]
        };

        let [top_b, top_g, top_r, _] = sample(width / 2, height / 4);
        assert!(
            top_b > 200 && top_r < 50 && top_g < 50,
            "expected the top half (former left half) to be blue after a clockwise rotation, \
             got {:?}",
            [top_b, top_g, top_r]
        );

        let [bottom_b, bottom_g, bottom_r, _] = sample(width / 2, height * 3 / 4);
        assert!(
            bottom_b > 200 && bottom_r > 200 && bottom_g > 200,
            "expected the bottom half (former right half) to stay white after a clockwise \
             rotation, got {:?}",
            [bottom_b, bottom_g, bottom_r]
        );

        evict_document(id);
    }

    #[test]
    fn get_metadata_does_not_panic_on_a_dangling_bookmark_reference() {
        // A bookmark that pointed at a real page, which was then removed from the document -
        // its destination now references a page object that's no longer part of /Pages,
        // approximating the "bookmark points somewhere page_index() can't resolve validly"
        // case parse_bookmark's out-of-bounds guard protects against.
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (id, bytes) = fixture("dangling_bookmark.pdf");
        evict_document(id);
        let renderer = PdfRenderer::new(id, bytes, None);

        let metadata = futures::executor::block_on(renderer.get_metadata(None));

        // The key assertion is simply that this doesn't panic. Whatever chapter data comes
        // back (possibly none, if pdfium can't resolve the destination at all) is secondary.
        assert!(
            metadata.is_ok(),
            "expected a dangling bookmark to be handled gracefully, got {metadata:?}"
        );

        evict_document(id);
    }

    #[test]
    fn bookmark_targets_wrapped_in_a_goto_action_are_resolved_not_defaulted_to_page_zero() {
        // pypdf (like Acrobat) writes every outline entry as an /A GoTo-action dict rather
        // than a direct /Dest entry. A bookmark whose target is only reachable by unwrapping
        // that action must resolve to the real target, not silently fall back to (page 0,
        // origin) as if it were genuinely dangling. This fixture's target (page index 2, a
        // 400x600 page, deliberately a different size than pages 0/1) exercises both the
        // action-unwrapping and the target-page-bounds lookup in the same pass.
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (id, bytes) = fixture("action_based_bookmark.pdf");
        evict_document(id);
        let renderer = PdfRenderer::new(id, bytes, None);

        let metadata = futures::executor::block_on(renderer.get_metadata(None)).unwrap();

        assert_eq!(metadata.page_count, 3);
        let chapter = metadata
            .chapters
            .first()
            .expect("expected the one outline entry to be parsed");
        assert_eq!(chapter.target.page_index, 2);
        // Fit.xyz(left=50, top=500) in PDF's bottom-up space: get_point's y is measured
        // down from the *target* page's own top edge, so on this 600pt-tall page that's
        // 600 - 500 = 100. Using page 0/1's height (200) for that subtraction instead, the
        // bug this test guards against, would give a nonsensical negative 200 - 500 = -300.
        assert!(
            (chapter.target.point.x - 50.0).abs() < 1.0,
            "expected target x near 50, got {}",
            chapter.target.point.x
        );
        assert!(
            (chapter.target.point.y - 100.0).abs() < 1.0,
            "expected target y near 100, got {}",
            chapter.target.point.y
        );

        evict_document(id);
    }
}
