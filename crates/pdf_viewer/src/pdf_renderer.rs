use anyhow::{Result, anyhow};
use futures::lock::Mutex;
use gpui::{Bounds, Point, RenderImage, Size};
use http_client::HttpClient;
use http_client::github::AssetKind;
use http_client::github_download::download_server_binary;
use image::{Frame, RgbaImage};
use once_cell::sync::OnceCell;
use pdfium_render::prelude::{
    PdfBitmap, PdfBitmapFormat, PdfBookmark, PdfColor, PdfDestination, PdfDocument,
    PdfDocumentMetadataTagType, PdfDocumentVersion, PdfLink as PdfiumLink, PdfPage,
    PdfPageAnnotation, PdfPageAnnotationCommon, PdfPageRenderRotation, PdfRect, PdfRenderConfig,
    PdfSearchDirection, PdfSearchOptions, Pdfium, PdfiumError, PdfiumInternalError,
    PdfiumLibraryBindings,
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

/// View-only settings a page render depends on besides its scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RenderStyle {
    pub rotation: PageRotation,
    /// Inverts the lightness of the page, for reading in dark themes.
    pub inverted: bool,
}

impl From<PageRotation> for RenderStyle {
    fn from(rotation: PageRotation) -> Self {
        Self {
            rotation,
            inverted: false,
        }
    }
}

/// Inverts the lightness of a pixel while keeping its hue, so white paper turns black while
/// colored highlights and figures keep their colors.
fn invert_lightness(pixel: &mut [u8; 4]) {
    let [first, second, third, _alpha] = *pixel;
    let max = first.max(second).max(third) as u16;
    let min = first.min(second).min(third) as u16;
    for channel in &mut pixel[..3] {
        // stays within 0..=255, since every channel lies between `min` and `max`.
        *channel = (*channel as u16 + 255 - max - min) as u8;
    }
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

    /// Returns the on-screen size of a page with the given unrotated size.
    pub fn display_size(self, page_size: Size<f32>) -> Size<f32> {
        if self.swaps_dimensions() {
            Size {
                width: page_size.height,
                height: page_size.width,
            }
        } else {
            page_size
        }
    }

    /// Maps a point in unrotated page coordinates to the rotated, on-screen page.
    pub fn to_display_point(self, point: Point<f32>, page_size: Size<f32>) -> Point<f32> {
        let Size { width, height } = page_size;
        match self {
            PageRotation::None => point,
            PageRotation::Clockwise90 => Point {
                x: height - point.y,
                y: point.x,
            },
            PageRotation::Clockwise180 => Point {
                x: width - point.x,
                y: height - point.y,
            },
            PageRotation::Clockwise270 => Point {
                x: point.y,
                y: width - point.x,
            },
        }
    }

    /// Maps a point on the rotated, on-screen page back to unrotated page coordinates.
    pub fn to_page_point(self, point: Point<f32>, page_size: Size<f32>) -> Point<f32> {
        let Size { width, height } = page_size;
        match self {
            PageRotation::None => point,
            PageRotation::Clockwise90 => Point {
                x: point.y,
                y: height - point.x,
            },
            PageRotation::Clockwise180 => Point {
                x: width - point.x,
                y: height - point.y,
            },
            PageRotation::Clockwise270 => Point {
                x: width - point.y,
                y: point.x,
            },
        }
    }

    /// Maps a rectangle in unrotated page coordinates to the rotated, on-screen page.
    pub fn to_display_bounds(self, bounds: Bounds<f32>, page_size: Size<f32>) -> Bounds<f32> {
        let first = self.to_display_point(bounds.origin, page_size);
        let second = self.to_display_point(bounds.bottom_right(), page_size);
        Bounds::from_corners(
            Point {
                x: first.x.min(second.x),
                y: first.y.min(second.y),
            },
            Point {
                x: first.x.max(second.x),
                y: first.y.max(second.y),
            },
        )
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
    pub properties: DocumentProperties,
    /// Files embedded in the document, in the order `PdfRenderer::attachment_bytes` indexes them.
    pub attachments: Vec<PdfAttachmentInfo>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PdfAttachmentInfo {
    pub name: String,
    /// Size of the embedded file in bytes.
    pub size: usize,
}

/// Descriptive fields of the document information dictionary, with empty ones left out.
///
/// Leaves out the modification date, since pdfium-render 0.9 looks it up under
/// `ModificationDate` rather than the `ModDate` key documents use.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DocumentProperties {
    pub title: Option<String>,
    pub author: Option<String>,
    pub subject: Option<String>,
    pub keywords: Option<String>,
    pub creator: Option<String>,
    pub producer: Option<String>,
    pub created: Option<String>,
    pub version: Option<String>,
}

impl DocumentProperties {
    fn from_document(document: &PdfDocument) -> Self {
        let metadata = document.metadata();
        let tag = |tag_type| {
            metadata
                .get(tag_type)
                .map(|tag| tag.value().trim().to_string())
                .filter(|value| !value.is_empty())
        };
        let version = match document.version() {
            PdfDocumentVersion::Unset => None,
            PdfDocumentVersion::Pdf1_0 => Some("1.0".to_string()),
            PdfDocumentVersion::Pdf1_1 => Some("1.1".to_string()),
            PdfDocumentVersion::Pdf1_2 => Some("1.2".to_string()),
            PdfDocumentVersion::Pdf1_3 => Some("1.3".to_string()),
            PdfDocumentVersion::Pdf1_4 => Some("1.4".to_string()),
            PdfDocumentVersion::Pdf1_5 => Some("1.5".to_string()),
            PdfDocumentVersion::Pdf1_6 => Some("1.6".to_string()),
            PdfDocumentVersion::Pdf1_7 => Some("1.7".to_string()),
            PdfDocumentVersion::Pdf2_0 => Some("2.0".to_string()),
            PdfDocumentVersion::Other(version) => {
                Some(format!("{}.{}", version / 10, version % 10))
            }
        };
        Self {
            title: tag(PdfDocumentMetadataTagType::Title),
            author: tag(PdfDocumentMetadataTagType::Author),
            subject: tag(PdfDocumentMetadataTagType::Subject),
            keywords: tag(PdfDocumentMetadataTagType::Keywords),
            creator: tag(PdfDocumentMetadataTagType::Creator),
            producer: tag(PdfDocumentMetadataTagType::Producer),
            created: tag(PdfDocumentMetadataTagType::CreationDate)
                .map(|date| format_pdf_date(&date)),
            version,
        }
    }
}

/// Formats a PDF date such as `D:20240131235959+01'00'` as `2024-01-31 23:59:59 +01:00`.
///
/// Dates that do not follow the format are returned unchanged.
pub fn format_pdf_date(raw: &str) -> String {
    let date = raw.strip_prefix("D:").unwrap_or(raw);
    let digit_count = date.bytes().take_while(u8::is_ascii_digit).count();
    if digit_count < 4 {
        return raw.to_string();
    }
    let (numbers, zone) = date.split_at(digit_count);

    let mut formatted = numbers[..4].to_string();
    for (range, separator) in [
        (4..6, '-'),
        (6..8, '-'),
        (8..10, ' '),
        (10..12, ':'),
        (12..14, ':'),
    ] {
        let Some(part) = numbers.get(range) else {
            break;
        };
        formatted.push(separator);
        formatted.push_str(part);
    }

    let mut zone_chars = zone.chars();
    match zone_chars.next() {
        Some('Z') => formatted.push_str(" UTC"),
        Some(sign @ ('+' | '-')) => {
            let offset = zone_chars.filter(char::is_ascii_digit).collect::<String>();
            if let Some(hours) = offset.get(..2) {
                let minutes = offset.get(2..4).unwrap_or("00");
                formatted.push_str(&format!(" {sign}{hours}:{minutes}"));
            }
        }
        _ => {}
    }
    formatted
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
            properties: DocumentProperties::default(),
            attachments: Vec::new(),
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
            properties: DocumentProperties::default(),
            attachments: Vec::new(),
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
    /// Style `image` was rasterized with; renders in another style are not reusable.
    pub style: RenderStyle,
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

/// `pdfium-binaries` release installed when no PDFium library is found.
const PDFIUM_RELEASE: &str = "8100";

/// Returns the `pdfium-binaries` archive name and its SHA-256 for this platform.
fn pdfium_asset() -> Option<(&'static str, &'static str)> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => Some((
            "pdfium-mac-arm64",
            "7ea647c7a7a9c1169be3f947000bc071ece2fade2cd6bd7e401e3675c6560cce",
        )),
        ("macos", "x86_64") => Some((
            "pdfium-mac-x64",
            "d513a0530a0f0c1ca9929ded06660e27edb9a108a7d18d9df55a5fb060b091c8",
        )),
        ("linux", "aarch64") => Some((
            "pdfium-linux-arm64",
            "58c87a95a40a24c914a9c755fae798843f46f06adf9e5e9f2c893302d4bdb040",
        )),
        ("linux", "x86_64") => Some((
            "pdfium-linux-x64",
            "1c1894a12d4d96ee01400c3874444126254180c5c7d2ab628576b67f8120320a",
        )),
        _ => None,
    }
}

/// Directory of the PDFium release installed by [`ensure_pdfium_installed`].
fn installed_pdfium_dir() -> std::path::PathBuf {
    paths::data_dir()
        .join("pdfium")
        .join(format!("chromium-{PDFIUM_RELEASE}"))
}

/// Serializes installs, so PDFs opened at the same time download PDFium once.
static PDFIUM_INSTALL_LOCK: Mutex<()> = Mutex::new(());

/// Downloads the pinned PDFium release into the data directory, unless a PDFium library can
/// already be loaded.
///
/// # Errors
///
/// Returns `Err` if no prebuilt release exists for this platform, or the download, checksum, or
/// extraction fails.
pub async fn ensure_pdfium_installed(http_client: Arc<dyn HttpClient>) -> Result<()> {
    if PDFIUM.get().is_some() {
        return Ok(());
    }
    let _install_guard = PDFIUM_INSTALL_LOCK.lock().await;
    if get_pdfium().is_ok() {
        return Ok(());
    }

    let (asset, sha256) = pdfium_asset().ok_or_else(|| {
        anyhow!(
            "no prebuilt PDFium is available for {}-{}",
            std::env::consts::OS,
            std::env::consts::ARCH
        )
    })?;
    let destination = installed_pdfium_dir();
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let url = format!(
        "https://github.com/bblanchon/pdfium-binaries/releases/download/chromium/{PDFIUM_RELEASE}/{asset}.tgz"
    );
    log::info!("PDFium not found, installing {url} into {destination:?}");
    download_server_binary(
        http_client.as_ref(),
        &url,
        Some(sha256),
        &destination,
        AssetKind::TarGz,
    )
    .await?;

    get_pdfium().map(|_| ())
}

/// Binds to PDFium next to the working directory, in the data directory, or on the system.
fn get_bindings() -> Result<Box<dyn PdfiumLibraryBindings>> {
    let bindings = Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path("./"))
        .or_else(|_| {
            Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path("./pdfium/"))
        })
        .or_else(|_| {
            Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(
                &installed_pdfium_dir().join("lib"),
            ))
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
    let mutex = get_pdfium()?;
    // GPUI test executors can leak a pending lock future when a test ends, and `futures` then
    // hands every later wake-up to that leaked waiter, hanging the next test that waits for the
    // lock. Polling with `try_lock` never registers a waiter.
    #[cfg(test)]
    let mut state = futures::future::poll_fn(|cx| match mutex.try_lock() {
        Some(state) => std::task::Poll::Ready(state),
        None => {
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        }
    })
    .await;
    #[cfg(not(test))]
    let mut state = mutex.lock().await;
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
            properties: DocumentProperties::from_document(document),
            attachments: document
                .attachments()
                .iter()
                .map(|attachment| PdfAttachmentInfo {
                    name: attachment.name(),
                    size: attachment.len(),
                })
                .collect(),
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
        // view-only rotation and colors. Text and links stay in page space, so `prev_data` is
        // reused for them in any style.
        style: RenderStyle,
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

        let rotation = style.rotation;
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
            if style.inverted {
                invert_lightness(pixel);
            }
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
            style,
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

    /// Returns the contents of the embedded file at `index` in `DocumentMetadata::attachments`.
    pub async fn attachment_bytes(&self, index: usize) -> Result<Vec<u8>> {
        let mut pdfium_state = lock_pdfium_state().await?;
        let document = open_document(
            &mut pdfium_state,
            self.id,
            &self.pdf_bytes,
            self.password.as_deref(),
        )
        .map_err(|e| anyhow!("Failed to read PDF bytes: {:?}", e))?;
        let index = u16::try_from(index)?;
        document
            .attachments()
            .get(index)
            .and_then(|attachment| attachment.save_to_bytes())
            .map_err(|e| anyhow!("Failed to read attachment {index}: {:?}", e))
    }

    /// Searches the document for `query`.
    pub async fn search_document(
        &self,
        query: &str,
        match_case: bool,
        whole_word: bool,
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
        let search_options = PdfSearchOptions::new()
            .match_case(match_case)
            .match_whole_word(whole_word);

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
                        // the segment rect spans exactly the matched characters, including
                        // spaces in the query, but hugs the ink vertically. `chars()` returns
                        // every character touching the rect, neighbors included, so only
                        // characters centered inside it contribute their font-height boxes.
                        let rect = segment.bounds();
                        let left = rect.left().value;
                        let right = rect.right().value;
                        let mut bottom = rect.bottom().value;
                        let mut top = rect.top().value;
                        if let Ok(segment_chars) = segment.chars() {
                            for text_char in segment_chars.iter() {
                                let Ok(char_rect) = text_char.loose_bounds() else {
                                    continue;
                                };
                                let center =
                                    (char_rect.left().value + char_rect.right().value) / 2.0;
                                if (left..=right).contains(&center) {
                                    bottom = bottom.min(char_rect.bottom().value);
                                    top = top.max(char_rect.top().value);
                                }
                            }
                        }

                        if right > left && top > bottom {
                            bounds_vec.push(Bounds {
                                origin: Point {
                                    x: left - page_bounds.origin.x,
                                    y: page_bounds.origin.y - top,
                                },
                                size: Size {
                                    width: right - left,
                                    height: top - bottom,
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

    #[test]
    fn rotation_maps_page_points_to_the_display_and_back() {
        let page_size = Size {
            width: 100.0,
            height: 200.0,
        };
        let point = Point { x: 10.0, y: 30.0 };
        let cases = [
            (PageRotation::None, Point { x: 10.0, y: 30.0 }),
            (PageRotation::Clockwise90, Point { x: 170.0, y: 10.0 }),
            (PageRotation::Clockwise180, Point { x: 90.0, y: 170.0 }),
            (PageRotation::Clockwise270, Point { x: 30.0, y: 90.0 }),
        ];
        for (rotation, expected) in cases {
            let display = rotation.to_display_point(point, page_size);
            assert_eq!(display, expected, "{rotation:?}");
            let display_size = rotation.display_size(page_size);
            assert!(display.x <= display_size.width && display.y <= display_size.height);
            assert_eq!(
                rotation.to_page_point(display, page_size),
                point,
                "{rotation:?}"
            );
        }

        let bounds = Bounds {
            origin: point,
            size: Size {
                width: 20.0,
                height: 5.0,
            },
        };
        let rotated = PageRotation::Clockwise90.to_display_bounds(bounds, page_size);
        assert_eq!(rotated.origin, Point { x: 165.0, y: 10.0 });
        assert_eq!(
            rotated.size,
            Size {
                width: 5.0,
                height: 20.0
            }
        );
    }

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
    fn pdf_dates_are_formatted_for_display() {
        assert_eq!(
            format_pdf_date("D:20240131235959+01'00'"),
            "2024-01-31 23:59:59 +01:00"
        );
        assert_eq!(
            format_pdf_date("D:20240131235959-0530"),
            "2024-01-31 23:59:59 -05:30"
        );
        assert_eq!(
            format_pdf_date("D:20240131120000Z"),
            "2024-01-31 12:00:00 UTC"
        );
        assert_eq!(format_pdf_date("D:2024"), "2024");
        assert_eq!(format_pdf_date("D:202401"), "2024-01");
        assert_eq!(format_pdf_date("yesterday"), "yesterday");
    }

    #[test]
    fn whole_word_search_skips_matches_inside_longer_words() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (id, bytes) = fixture("whole_words.pdf");
        evict_document(id);
        let renderer = PdfRenderer::new(id, bytes, None);
        let metadata = Arc::new(
            futures::executor::block_on(renderer.get_metadata(None)).expect("fixture should open"),
        );

        let search = |whole_word| {
            futures::executor::block_on(renderer.search_document(
                "cat",
                false,
                whole_word,
                metadata.clone(),
            ))
            .expect("search should succeed")
            .len()
        };
        assert_eq!(search(false), 3);
        assert_eq!(search(true), 2);

        evict_document(id);
    }

    #[test]
    fn search_highlights_cover_exactly_the_matched_characters() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (id, bytes) = fixture("whole_words.pdf");
        evict_document(id);
        let renderer = PdfRenderer::new(id, bytes, None);
        let metadata = Arc::new(
            futures::executor::block_on(renderer.get_metadata(None)).expect("fixture should open"),
        );
        let spans = futures::executor::block_on(renderer.extract_text_only(0, metadata.clone()))
            .expect("fixture should have a text layer");
        let text: String = spans.iter().map(|span| span.text.as_str()).collect();

        // an outer space of a query also matches a line start, so it has no fixed match text.
        for query in ["cat", "cat con"] {
            let results = futures::executor::block_on(renderer.search_document(
                query,
                false,
                false,
                metadata.clone(),
            ))
            .expect("search should succeed");
            assert!(!results.is_empty(), "{query:?} should match");

            let mut expected: Vec<std::ops::Range<usize>> = text
                .match_indices(query)
                .map(|(start, matched)| start..start + matched.len())
                .collect();
            for result in &results {
                let [bounds] = result.bounds.as_slice() else {
                    panic!("{query:?} should match on one line");
                };
                let covered: Vec<usize> = spans
                    .iter()
                    .enumerate()
                    .filter(|(_, span)| {
                        let center = span.bounds.center();
                        bounds.contains(&center)
                    })
                    .map(|(index, _)| index)
                    .collect();
                let range = covered.first().copied().unwrap_or_default()
                    ..covered.last().map_or(0, |last| last + 1);
                assert_eq!(covered.len(), range.len(), "{query:?} covers a gap");
                let position = expected.iter().position(|matched| *matched == range);
                assert!(
                    position.is_some(),
                    "{query:?} highlight covers {:?} instead of a match",
                    &text[range]
                );
                expected.remove(position.unwrap_or_default());

                let tallest_span = spans[range]
                    .iter()
                    .map(|span| span.bounds.size.height)
                    .fold(0.0, f32::max);
                assert!(bounds.size.height >= tallest_span - 0.01);
            }
        }

        evict_document(id);
    }

    #[test]
    fn embedded_files_are_listed_and_readable() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (id, bytes) = fixture("embedded_attachment.pdf");
        evict_document(id);
        let renderer = PdfRenderer::new(id, bytes, None);

        let metadata =
            futures::executor::block_on(renderer.get_metadata(None)).expect("fixture should open");
        let contents = b"hello from an attachment\n";
        assert_eq!(
            metadata.attachments,
            vec![PdfAttachmentInfo {
                name: "notes.txt".to_string(),
                size: contents.len(),
            }]
        );
        let attachment = futures::executor::block_on(renderer.attachment_bytes(0))
            .expect("attachment should be readable");
        assert_eq!(attachment, contents);
        assert!(futures::executor::block_on(renderer.attachment_bytes(1)).is_err());

        evict_document(id);
    }

    #[test]
    fn get_metadata_reads_the_document_information() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (id, bytes) = fixture("solid_red.pdf");
        evict_document(id);
        let renderer = PdfRenderer::new(id, bytes, None);

        let metadata =
            futures::executor::block_on(renderer.get_metadata(None)).expect("fixture should open");
        assert_eq!(
            metadata.properties,
            DocumentProperties {
                title: Some("untitled".to_string()),
                author: Some("anonymous".to_string()),
                subject: Some("unspecified".to_string()),
                keywords: None,
                creator: Some("anonymous".to_string()),
                producer: Some("ReportLab PDF Library - (opensource)".to_string()),
                created: Some("2026-08-09 17:51:06 +02:00".to_string()),
                version: Some("1.4".to_string()),
            }
        );

        evict_document(id);
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
            PageRotation::None.into(),
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
            PageRotation::None.into(),
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
            PageRotation::None.into(),
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
            PageRotation::None.into(),
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
            PageRotation::None.into(),
            None,
            Arc::new(metadata.clone()),
        ))
        .unwrap();
        let bottom_strip = futures::executor::block_on(renderer.process_page(
            0,
            1.0,
            page_height_pt,
            PageRotation::None.into(),
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
            PageRotation::None.into(),
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
    fn inverted_renders_turn_paper_dark_but_keep_colors() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (id, bytes) = fixture("left_half_blue.pdf");
        evict_document(id);
        let renderer = PdfRenderer::new(id, bytes, None);
        let metadata = futures::executor::block_on(renderer.get_metadata(None)).unwrap();

        let style = RenderStyle {
            rotation: PageRotation::None,
            inverted: true,
        };
        let page = futures::executor::block_on(renderer.process_page(
            0,
            1.0,
            0.0,
            style,
            None,
            Arc::new(metadata),
        ))
        .unwrap();
        assert_eq!(page.style, style);

        let width = page.image.size(0).width.0 as usize;
        let height = page.image.size(0).height.0 as usize;
        let buffer = page.image.as_bytes(0).unwrap();
        let sample = |x: usize, y: usize| {
            let offset = (y * width + x) * 4;
            [buffer[offset], buffer[offset + 1], buffer[offset + 2]]
        };
        let [blue, green, red] = sample(width / 4, height / 2);
        assert!(
            blue > 200 && green < 50 && red < 50,
            "blue should stay blue, got {:?}",
            [blue, green, red]
        );
        assert_eq!(
            sample(width * 3 / 4, height / 2),
            [0, 0, 0],
            "white turns black"
        );

        evict_document(id);
    }

    #[test]
    fn inverting_lightness_keeps_hue() {
        let invert = |mut pixel: [u8; 4]| {
            invert_lightness(&mut pixel);
            pixel
        };
        assert_eq!(invert([255, 255, 255, 255]), [0, 0, 0, 255]);
        assert_eq!(invert([0, 0, 0, 255]), [255, 255, 255, 255]);
        assert_eq!(invert([100, 100, 100, 128]), [155, 155, 155, 128]);
        assert_eq!(invert([255, 0, 0, 255]), [255, 0, 0, 255]);
        assert_eq!(invert([200, 250, 200, 255]), [5, 55, 5, 255]);
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
            PageRotation::Clockwise90.into(),
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
