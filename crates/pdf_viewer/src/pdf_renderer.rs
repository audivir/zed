use anyhow::{Result, anyhow};
use futures::lock::Mutex;
use gpui::{Bounds, Point, RenderImage, Size};
use image::{Frame, RgbaImage};
use once_cell::sync::OnceCell;
use pdfium_render::prelude::{
    PdfBitmap, PdfBitmapFormat, PdfBookmark, PdfColor, PdfDestination, PdfDocument,
    PdfLink as PdfiumLink, PdfPage, PdfPageAnnotation, PdfPageAnnotationCommon,
    PdfPageRenderRotation, PdfRect, PdfRenderConfig, PdfSearchDirection, PdfSearchOptions, Pdfium,
    PdfiumError, PdfiumInternalError, PdfiumLibraryBindings,
};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use std::time::Instant;

/// Maximum pixel width of a rendered page, regardless of zoom.
const MAX_RENDER_WIDTH: f32 = 8192.0;

/// Maximum pixel height of a rendered bitmap.
///
/// Taller pages render a strip of this height around `visible_center_y` instead of downscaling.
const MAX_RENDER_HEIGHT: f32 = 8192.0;

/// Text layer and link annotations of a page.
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

/// Container for a positioned piece of text in the interactive text layer.
#[derive(Debug, Clone)]
pub struct PdfTextSpan {
    pub text: String,
    pub bounds: Bounds<f32>,
}

#[derive(Debug, Clone)]
pub struct PdfTarget {
    pub page_index: usize,
    pub point: Point<f32>,
    /// Page-space Y coordinate of the bottom of the block containing `point`.
    ///
    /// Lets the jump highlight cover a multi-line source span. `None` highlights a single line.
    pub block_bottom_y: Option<f32>,
    /// Approximate height of one text line at `point`, in page points.
    ///
    /// Pads the jump highlight to cover glyphs rather than baselines. `None` uses a fixed guess.
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

/// View-only rotation for the whole document, applied at render time.
///
/// Never written back to the `/Rotate` attribute of the PDF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PageRotation {
    #[default]
    None,
    Clockwise90,
    Clockwise180,
    Clockwise270,
}

impl PageRotation {
    /// Returns the rotation after one more clockwise quarter turn.
    pub fn rotated_clockwise(self) -> Self {
        match self {
            PageRotation::None => PageRotation::Clockwise90,
            PageRotation::Clockwise90 => PageRotation::Clockwise180,
            PageRotation::Clockwise180 => PageRotation::Clockwise270,
            PageRotation::Clockwise270 => PageRotation::None,
        }
    }

    /// Whether this rotation is a quarter turn that swaps page width and height.
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
    /// Whether opening failed because a password is missing or wrong.
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

/// Container for the rendered image, text, and links of a PDF page.
#[derive(Clone)]
pub struct PageData {
    pub image: Arc<RenderImage>,
    /// Scale this render was requested for.
    ///
    /// Exceeds `image_scale` when capped by `MAX_RENDER_WIDTH`. Treating the capped render as
    /// satisfying the requested scale avoids re-rendering it every frame.
    pub scale: f32,
    /// Scale used to rasterize `image`, in pixels per page point.
    ///
    /// Use this rather than `scale` to convert pixel sizes back to page space.
    pub image_scale: f32,
    /// Page-space Y offset where `image` starts.
    ///
    /// Nonzero only for strips of pages taller than `MAX_RENDER_HEIGHT`.
    pub page_offset_pt: f32,
    /// Rotation `image` was rasterized at; renders at another rotation are not reusable.
    pub rotation: PageRotation,
    pub text_spans: Arc<Vec<PdfTextSpan>>,
    pub links: Arc<Vec<PdfLink>>,
}

impl PageData {
    /// Page-space height covered by `image`.
    pub fn covered_height_pt(&self) -> f32 {
        self.image.size(0).height.0 as f32 / self.image_scale
    }

    /// Whether `image` covers the page-space Y coordinate `y_pt`.
    pub fn covers_page_y(&self, y_pt: f32) -> bool {
        y_pt >= self.page_offset_pt && y_pt <= self.page_offset_pt + self.covered_height_pt()
    }
}

impl std::fmt::Debug for PageData {
    // `RenderImage` does not implement `Debug`.
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

/// Container for the PDFium binding and a cache of opened documents.
///
/// A single mutex serializes all access, since PDFium is not thread-safe even across documents.
struct PdfiumState {
    pdfium: &'static Pdfium,
    /// Documents keyed by `Pdf::id`, a content hash, so identical content shares one parse.
    documents: HashMap<u64, PdfDocument<'static>>,
}

/// Global PDFium instance shared by all background tasks.
static PDFIUM: OnceCell<Mutex<PdfiumState>> = OnceCell::new();

/// Document IDs whose eviction lost the `try_lock` race against the `PDFIUM` mutex.
///
/// Drained by `lock_pdfium_state`, so every eviction eventually applies.
static PENDING_EVICTIONS: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());

/// Serializes tests in this crate that call into PDFium.
///
/// Concurrent tests otherwise queue on the `PDFIUM` mutex long enough to trip the 15-second
/// parking timeout of gpui.
#[cfg(test)]
pub(crate) static PDFIUM_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Binds to the system PDFium library.
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

        // leaked so cached `PdfDocument`s, which borrow the binding, can outlive any guard.
        let pdfium: &'static Pdfium = Box::leak(Box::new(Pdfium::new(bindings)));

        Ok(Mutex::new(PdfiumState {
            pdfium,
            documents: HashMap::new(),
        }))
    })
}

/// Returns the cached document for `id`, opening it on first access.
///
/// `load_pdf_from_byte_vec` makes the document borrow only the process-lifetime binding, so it
/// can be reused by later renders and searches.
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

/// Evicts a document from the shared cache.
///
/// Uses `try_lock` since `Drop` cannot await; on contention, the ID is queued in
/// `PENDING_EVICTIONS`.
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

/// Locks the shared `PdfiumState` and applies queued evictions.
///
/// All access to `PDFIUM` must go through this, so documents dropped mid-render do not leak.
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

    /// Extracts the dimensions of every page without rendering.
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
                return Ok(DocumentMetadata::password_required(self.password.is_some()));
            }
            Err(e) => return Err(anyhow!("Failed to read PDF bytes: {:?}", e)),
        };

        log::debug!("Document parsed in {:?}", start.elapsed());

        let (pages, max_width) = pages_max_width.unwrap_or_else(|| {
            let start = Instant::now();
            let mut max_width = 0.0f32;
            let pages: Vec<PageMetadata> = document
                .pages()
                .iter()
                .enumerate()
                .map(|(i, page)| {
                    // PDFium sizes are in points (1/72 inch).
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
            log::debug!(
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

        log::debug!(
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

    /// Parses a bookmark and its children recursively.
    #[allow(clippy::only_used_in_recursion)]
    fn parse_bookmark(
        bookmark: &PdfBookmark,
        document: &PdfDocument,
        pages: &[PageMetadata],
    ) -> PdfChapter {
        let mut page_index = 0;
        let mut point = Point { x: 0.0, y: 0.0 };

        // the target is attached directly or wrapped in a `GoToDestinationInSameDocument`
        // action, as Acrobat writes it. Other actions are not internal navigation.
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
            // corrupt documents may reference missing pages; keep the page 0 default.
            page_index = idx;
            point = target_point;
        }

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

    fn get_point(destination: &PdfDestination, page_bounds: &Bounds<f32>) -> Point<f32> {
        let bounds_left = page_bounds.origin.x;
        let bounds_top = page_bounds.origin.y;

        let mut target_left = 0.0;
        let mut target_top = 0.0;

        if let Ok(view_settings) = destination.view_settings() {
            use pdfium_render::prelude::PdfDestinationViewSettings::*;

            match view_settings {
                SpecificCoordinatesAndZoom(left, top, _zoom) => {
                    if let Some(l) = left {
                        target_left = l.value - bounds_left;
                    }
                    if let Some(t) = top {
                        target_top = bounds_top - t.value;
                    }
                }
                FitPageHorizontallyToWindow(top) | FitBoundsHorizontallyToWindow(top) => {
                    if let Some(t) = top {
                        target_top = bounds_top - t.value;
                    }
                }
                FitPageVerticallyToWindow(left) | FitBoundsVerticallyToWindow(left) => {
                    if let Some(l) = left {
                        target_left = l.value - bounds_left;
                    }
                }
                FitPageToRectangle(rect) => {
                    target_left = rect.left().value - bounds_left;
                    target_top = bounds_top - rect.top().value;
                }
                // keep the top-left default.
                FitPageToWindow | FitBoundsToWindow | Unknown => {}
            }
        }

        Point {
            x: target_left,
            y: target_top,
        }
    }

    /// Resolves the target page index of a destination and its point within that page.
    ///
    /// Returns `None` if the target page does not exist in this document.
    fn resolve_destination_target(
        destination: &PdfDestination,
        pages: &[PageMetadata],
    ) -> Option<(usize, Point<f32>)> {
        let page_index = destination.page_index().ok()? as usize;
        let page_metadata = pages.get(page_index)?;
        Some((
            page_index,
            Self::get_point(destination, &page_metadata.bounds),
        ))
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

            // the destination is attached directly or wrapped in a
            // `GoToDestinationInSameDocument` action, as Acrobat writes it.
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

    /// Renders a page and extracts its text and links.
    pub async fn process_page(
        &self,
        page_index: usize,
        scale: f32,
        // page-space Y coordinate to center a strip on; ignored unless the page is rendered as a
        // strip.
        visible_center_y: f32,
        // view-only rotation. Text and links stay in page space, so `prev_data` is reused for them
        // at any rotation.
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

        // a quarter turn swaps output width and height.
        let (source_width_pt, source_height_pt) = if rotation.swaps_dimensions() {
            (page.height().value, page.width().value)
        } else {
            (page.width().value, page.height().value)
        };

        // cap the width so extreme zoom cannot request a multi-gigabyte bitmap. Height is
        // capped separately below.
        let target_width = ((source_width_pt * scale).round()).min(MAX_RENDER_WIDTH) as i32;
        let effective_scale = target_width as f32 / source_width_pt;
        let full_height_px = source_height_pt * effective_scale;

        // render only a strip around `visible_center_y` for pages too tall to render whole, a
        // one-dimensional version of the tiling in the Chromium PDF viewer.
        let (target_height, page_offset_pt) = if full_height_px > MAX_RENDER_HEIGHT {
            let strip_height_pt = MAX_RENDER_HEIGHT / effective_scale;
            let max_offset_pt = (source_height_pt - strip_height_pt).max(0.0);
            let offset_pt = (visible_center_y - strip_height_pt / 2.0).clamp(0.0, max_offset_pt);
            (MAX_RENDER_HEIGHT.round() as i32, offset_pt)
        } else {
            (full_height_px.round() as i32, 0.0)
        };

        let mut bitmap = PdfBitmap::empty(target_width, target_height, PdfBitmapFormat::default())
            .map_err(|e| anyhow!("Failed to allocate render target: {:?}", e))?;

        let render_config = PdfRenderConfig::new()
            // set both dimensions explicitly so the output follows the rotated logical size rather
            // than the aspect inference of pdfium-render.
            .set_target_width(target_width)
            .set_target_height((source_height_pt * effective_scale).round() as i32)
            .set_origin(0, -((page_offset_pt * effective_scale).round() as i32))
            .set_clear_color(PdfColor::WHITE)
            .rotate(rotation.to_pdfium(), false);

        page.render_into_bitmap_with_config(&mut bitmap, &render_config)
            .map_err(|e| anyhow!("Render failed: {:?}", e))?;

        // `as_rgba_bytes` normalizes every PDFium pixel format to RGBA; swapping red and blue
        // yields the BGRA that `RenderImage` expects.
        let width = bitmap.width() as u32;
        let height = bitmap.height() as u32;
        let mut bgra_bytes = bitmap.as_rgba_bytes();
        for pixel in bgra_bytes.as_chunks_mut::<4>().0 {
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

        log::debug!(
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

    /// Extracts the text layer and link annotations of a page without rasterizing.
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

        // TODO: log errors instead of propagating them.

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
        // TODO: log errors.

        let link_annotations = page
            .annotations()
            .iter()
            .filter(|annot| matches!(annot, PdfPageAnnotation::Link(_)));

        let page_links = page.links().iter();

        let mut links_vec = Vec::new();

        for (annot, link_action) in link_annotations.zip(page_links) {
            if let Some(link) =
                PdfRenderer::get_link(&annot, &link_action, &page_bounds, &metadata.pages)
            {
                links_vec.push(link)
            }
        }

        Ok((Arc::new(text_spans_vec), Arc::new(links_vec)))
    }

    /// Extracts the text layer of `page_index` without rasterizing.
    ///
    /// Lets Select All read every page without rendering it.
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

    /// Searches the document for `query`.
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

        log::debug!("Searched in {:?}", start.elapsed());

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // tests load the same fixture and would share one cached document, so they run serialized
    // and evict before and after each test.
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

        // a second renderer without the password reuses the unlocked cached document.
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

        // hold the mutex so the `try_lock` in `evict_document` fails and the ID is queued.
        let held = futures::executor::block_on(get_pdfium().unwrap().lock());
        evict_document(id);
        assert!(
            held.documents.contains_key(&id),
            "the document must still be cached while the mutex is held"
        );
        drop(held);

        // the next lock applies the deferred eviction.
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
        assert!(width > 0 && height > 0);

        let buffer = page.image.as_bytes(0).unwrap();
        // sample the center pixel to avoid antialiased edges.
        let offset = ((height / 2) * width + width / 2) * 4;
        let pixel = &buffer[offset..offset + 4];
        let [b, g, r, a] = [pixel[0], pixel[1], pixel[2], pixel[3]];

        // a solid red fill comes out blue if the red and blue swap is missing or doubled.
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

        // an uncapped render at scale 1000 would request a 100,000px wide bitmap.
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

        // both strips are capped below the full page height.
        let strip_height = top_strip.image.size(0).height.0 as f32;
        assert!(strip_height <= MAX_RENDER_HEIGHT);
        assert!(strip_height < page_height_pt);
        assert_eq!(strip_height, bottom_strip.image.size(0).height.0 as f32);

        // requests near the top and bottom land at different offsets covering their points.
        assert!(top_strip.page_offset_pt < bottom_strip.page_offset_pt);
        assert!(top_strip.covers_page_y(0.0));
        assert!(bottom_strip.covers_page_y(page_height_pt));
        assert!(!top_strip.covers_page_y(page_height_pt));
        assert!(!bottom_strip.covers_page_y(0.0));

        // the strips render different content.
        assert_ne!(
            top_strip.image.as_bytes(0),
            bottom_strip.image.as_bytes(0),
            "expected different page regions to render different content"
        );

        evict_document(id);
    }

    #[test]
    fn page_annotations_are_rendered_by_default() {
        // the fixture has a solid blue rectangle annotation covering the page, so the render is
        // blue only if annotations are rendered, which is the PDFium default.
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
        // a 200x100pt page with a blue left half. A clockwise quarter turn moves that half to
        // the top of the 100x200 output.
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (id, bytes) = fixture("left_half_blue.pdf");
        evict_document(id);
        let renderer = PdfRenderer::new(id, bytes, None);

        let metadata = futures::executor::block_on(renderer.get_metadata(None)).unwrap();
        assert_eq!(
            metadata.pages[0].size,
            Size {
                width: 200.0,
                height: 100.0
            }
        );

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
        // a bookmark whose target page was removed from the document.
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (id, bytes) = fixture("dangling_bookmark.pdf");
        evict_document(id);
        let renderer = PdfRenderer::new(id, bytes, None);

        let metadata = futures::executor::block_on(renderer.get_metadata(None));

        // the key assertion is that parsing does not panic.
        assert!(
            metadata.is_ok(),
            "expected a dangling bookmark to be handled gracefully, got {metadata:?}"
        );

        evict_document(id);
    }

    #[test]
    fn bookmark_targets_wrapped_in_a_goto_action_are_resolved_not_defaulted_to_page_zero() {
        // pypdf, like Acrobat, writes outline entries as GoTo actions. The target is page 2, sized
        // differently from pages 0 and 1 to exercise the target page bounds lookup.
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
        // `top=500` on the 600pt target page is 100pt from its top edge. Using the height of
        // page 0 instead would give -300.
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
