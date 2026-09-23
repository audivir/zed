use crate::pdf_renderer::{
    DocumentMetadata, PageData, PageMetadata, PdfLinkAction, PdfRenderer, PdfTextSpan, RenderStyle,
};
use anyhow::{Result, bail};
use collections::{HashMap, HashSet};
use gpui::{App, Size, Task, hash};
use parking_lot::{Mutex, RwLock};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Render scale for thumbnail sidebar images, independent of the zoom level.
const THUMBNAIL_SCALE: f32 = 0.2;

/// Removes the page from `rendering_pages` even if GPUI cancels the task.
#[derive(Debug)]
struct Guard {
    pdf: Arc<Pdf>,
    page: usize,
}

impl Drop for Guard {
    fn drop(&mut self) {
        self.pdf.rendering_pages.lock().remove(&self.page);
    }
}

/// Removes the page from `rendering_thumbnails` even if GPUI cancels the task.
struct ThumbnailGuard {
    pdf: Arc<Pdf>,
    page: usize,
}

impl Drop for ThumbnailGuard {
    fn drop(&mut self) {
        self.pdf.rendering_thumbnails.lock().remove(&self.page);
    }
}

/// PDF document with its raw bytes and rendered page caches.
#[derive(Debug)]
pub struct Pdf {
    pub id: u64,
    pub bytes: Arc<Vec<u8>>,
    pub metadata: Arc<DocumentMetadata>,
    /// Rendered pages keyed by page index.
    pub page_cache: RwLock<HashMap<usize, PageData>>,
    /// Text extracted without rendering, used for pages missing from `page_cache`.
    pub text_only_cache: RwLock<HashMap<usize, Arc<Vec<PdfTextSpan>>>>,
    /// Pages currently rendering, with their scale and style.
    pub rendering_pages: Mutex<HashMap<usize, (f32, RenderStyle)>>,
    /// Pages whose last render failed, with the failing scale and style.
    ///
    /// Prevents retrying the same failing render every frame.
    pub failed_pages: Mutex<HashMap<usize, (f32, RenderStyle)>>,
    /// Rendered thumbnail images.
    ///
    /// Kept apart from `page_cache` since thumbnails and the main view need the same page at very
    /// different scales and would otherwise evict each other.
    pub thumbnail_cache: RwLock<HashMap<usize, PageData>>,
    rendering_thumbnails: Mutex<HashSet<usize>>,
    failed_thumbnails: Mutex<HashSet<usize>>,
    /// Password used to open the document.
    ///
    /// Retrying with another password creates a new `Pdf`.
    pub password: Option<String>,
}

impl Drop for Pdf {
    fn drop(&mut self) {
        // releases the parsed document from the renderer cache so closed PDFs do not accumulate.
        crate::pdf_renderer::evict_document(self.id);
    }
}

impl PartialEq for Pdf {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Pdf {}

impl Hash for Pdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.id);
    }
}

impl Pdf {
    /// Returns an empty PDF without data.
    pub fn empty() -> Self {
        Self {
            id: 0,
            bytes: Arc::new(vec![]),
            metadata: Arc::new(DocumentMetadata::from_error(Some("Empty input".into()))),
            page_cache: RwLock::new(Default::default()),
            text_only_cache: RwLock::new(Default::default()),
            rendering_pages: Mutex::new(Default::default()),
            failed_pages: Mutex::new(Default::default()),
            thumbnail_cache: RwLock::new(Default::default()),
            rendering_thumbnails: Mutex::new(Default::default()),
            failed_thumbnails: Mutex::new(Default::default()),
            password: None,
        }
    }

    /// Creates a PDF from bytes, unlocking it with `password` if encrypted.
    ///
    /// A missing or wrong password sets `metadata.needs_password` instead of failing.
    pub async fn from_bytes(
        bytes: Vec<u8>,
        pages_max_width: Option<(Vec<PageMetadata>, f32)>,
        password: Option<String>,
    ) -> Self {
        let id = hash(&bytes);
        let arc = Arc::new(bytes);
        let renderer = PdfRenderer::new(id, arc.clone(), password.clone());
        let metadata = renderer
            .get_metadata(pages_max_width)
            .await
            .unwrap_or_else(|e| DocumentMetadata::from_error(Some(format!("{:?}", e))));
        Self {
            id,
            bytes: arc,
            metadata: Arc::new(metadata),
            page_cache: RwLock::new(Default::default()),
            text_only_cache: RwLock::new(Default::default()),
            rendering_pages: Mutex::new(Default::default()),
            failed_pages: Mutex::new(Default::default()),
            thumbnail_cache: RwLock::new(Default::default()),
            rendering_thumbnails: Mutex::new(Default::default()),
            failed_thumbnails: Mutex::new(Default::default()),
            password,
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    /// Returns the size of the page at `page_idx`.
    pub fn get_page_size(&self, page_idx: usize) -> Result<Size<f32>> {
        if let Some(page_metadata) = self.metadata.pages.get(page_idx) {
            return Ok(page_metadata.size);
        }
        bail!("Out of bounds")
    }

    /// Requests rendering of a page at the given scale.
    ///
    /// `visible_center_y` is the page-space Y coordinate of interest. For pages too tall to render
    /// whole, it selects the rendered strip; pass `0.0` without viewport context. `priority`
    /// orders this render against other pending renders.
    #[allow(clippy::too_many_arguments)]
    pub fn request_page(
        self: Arc<Self>,
        cx: &mut App,
        page_idx: usize,
        requested_scale: f32,
        visible_center_y: f32,
        style: RenderStyle,
        priority: gpui::Priority,
        on_rendered: impl FnOnce(&mut App) + Send + 'static,
    ) -> Option<Task<()>> {
        // reuse a cached render in this style and at least this scale that covers the point.
        let prev_page = {
            let cache = self.page_cache.read();

            if let Some(rendered) = cache.get(&page_idx) {
                // tolerance for float precision.
                if rendered.scale >= requested_scale * 0.99
                    && rendered.style == style
                    && rendered.covers_page_y(visible_center_y)
                {
                    return None; // We already have a good enough render, abort the task
                }
                // a render in another style has the wrong shape or colors.
                (rendered.style == style).then(|| rendered.clone())
            } else {
                None
            }
        };

        {
            let failed = self.failed_pages.lock();
            if let Some(&(failed_scale, failed_style)) = failed.get(&page_idx)
                && failed_scale >= requested_scale * 0.99
                && failed_style == style
            {
                return None; // Already know this render fails at this resolution; don't retry every frame
            }
        }

        {
            let mut rendering = self.rendering_pages.lock();
            if let Some(&(rendering_scale, rendering_style)) = rendering.get(&page_idx)
                && rendering_scale >= requested_scale * 0.99
                && rendering_style == style
            {
                return None; // Already queuing a render of equal or better resolution
            }
            rendering.insert(page_idx, (requested_scale, style));
        }

        Some(cx.spawn(async move |cx| {
            let pdf = self.clone();
            let bytes = self.bytes.clone();
            let password = self.password.clone();

            cx.background_executor()
                .spawn_with_priority(priority, async move {
                    let _guard = Guard {
                        pdf: pdf.clone(),
                        page: page_idx,
                    };

                    match PdfRenderer::new(pdf.id, bytes, password)
                        .process_page(
                            page_idx,
                            requested_scale,
                            visible_center_y,
                            style,
                            prev_page,
                            pdf.metadata.clone(),
                        )
                        .await
                    {
                        Ok(new_page) => {
                            pdf.failed_pages.lock().remove(&page_idx);

                            let mut cache = pdf.page_cache.write();

                            // overwrite only a worse or stale-style render that landed first.
                            let should_insert = cache
                                .get(&page_idx)
                                .is_none_or(|c| c.scale <= requested_scale || c.style != style);
                            if should_insert {
                                cache.insert(page_idx, new_page);
                                log::debug!("Written to cache");
                            }
                        }
                        Err(e) => {
                            log::error!("Failed to render page {}: {:?}", page_idx, e);
                            pdf.failed_pages
                                .lock()
                                .insert(page_idx, (requested_scale, style));
                        }
                    }
                })
                .await;

            cx.update(|cx| on_rendered(cx));
        }))
    }

    /// Requests a thumbnail render of `page_idx` at `THUMBNAIL_SCALE`, with inverted lightness
    /// if `inverted`.
    pub fn request_thumbnail(
        self: Arc<Self>,
        cx: &mut App,
        page_idx: usize,
        inverted: bool,
        on_rendered: impl FnOnce(&mut App) + Send + 'static,
    ) -> Option<Task<()>> {
        let style = RenderStyle {
            inverted,
            ..RenderStyle::default()
        };
        if self
            .thumbnail_cache
            .read()
            .get(&page_idx)
            .is_some_and(|thumbnail| thumbnail.style == style)
        {
            return None;
        }
        if self.failed_thumbnails.lock().contains(&page_idx) {
            return None;
        }
        if !self.rendering_thumbnails.lock().insert(page_idx) {
            return None; // Already queued
        }

        Some(cx.spawn(async move |cx| {
            let pdf = self.clone();
            let bytes = self.bytes.clone();
            let password = self.password.clone();

            cx.background_executor()
                .spawn_with_priority(gpui::Priority::Low, async move {
                    let _guard = ThumbnailGuard {
                        pdf: pdf.clone(),
                        page: page_idx,
                    };

                    match PdfRenderer::new(pdf.id, bytes, password)
                        // thumbnails always render unrotated.
                        .process_page(
                            page_idx,
                            THUMBNAIL_SCALE,
                            0.0,
                            style,
                            None,
                            pdf.metadata.clone(),
                        )
                        .await
                    {
                        Ok(new_page) => {
                            pdf.thumbnail_cache.write().insert(page_idx, new_page);
                        }
                        Err(e) => {
                            log::error!(
                                "Failed to render thumbnail for page {}: {:?}",
                                page_idx,
                                e
                            );
                            pdf.failed_thumbnails.lock().insert(page_idx);
                        }
                    }
                })
                .await;

            cx.update(|cx| on_rendered(cx));
        }))
    }

    /// Copies the renders and text of `previous` for the pages where `unchanged` holds.
    ///
    /// Lets a recompiled document show its untouched pages without rasterizing them again. Pages
    /// with internal links are skipped, since the pages their links point to may have moved.
    pub fn reuse_rendered_pages(&self, previous: &Pdf, unchanged: impl Fn(usize) -> bool) {
        let reusable = |page_idx: usize| {
            let same_size = self
                .metadata
                .pages
                .get(page_idx)
                .zip(previous.metadata.pages.get(page_idx))
                .is_some_and(|(page, previous_page)| page.size == previous_page.size);
            same_size && unchanged(page_idx)
        };
        let has_internal_links = |data: &PageData| {
            data.links
                .iter()
                .any(|link| matches!(link.action, PdfLinkAction::Internal(_)))
        };

        for (source, destination) in [
            (&previous.page_cache, &self.page_cache),
            (&previous.thumbnail_cache, &self.thumbnail_cache),
        ] {
            let reused = source
                .read()
                .iter()
                .filter(|(page_idx, data)| reusable(**page_idx) && !has_internal_links(data))
                .map(|(page_idx, data)| (*page_idx, data.clone()))
                .collect::<Vec<_>>();
            destination.write().extend(reused);
        }

        let reused_text = previous
            .text_only_cache
            .read()
            .iter()
            .filter(|(page_idx, _)| reusable(**page_idx))
            .map(|(page_idx, text_spans)| (*page_idx, text_spans.clone()))
            .collect::<Vec<_>>();
        self.text_only_cache.write().extend(reused_text);
    }

    /// Drops thumbnails of pages outside `pages`, so scrolling through the thumbnails of a long
    /// document does not keep every one of them in memory.
    pub fn evict_thumbnails_outside(&self, pages: std::ops::Range<usize>) {
        self.thumbnail_cache
            .write()
            .retain(|page_idx, _| pages.contains(page_idx));
    }

    /// Returns the text spans of `page_idx` from `page_cache` or `text_only_cache`.
    pub fn text_spans_for_page(&self, page_idx: usize) -> Option<Arc<Vec<PdfTextSpan>>> {
        if let Some(data) = self.page_cache.read().get(&page_idx) {
            return Some(data.text_spans.clone());
        }
        self.text_only_cache.read().get(&page_idx).cloned()
    }

    /// Extracts and caches the text layer of `page_idx` without rasterizing.
    ///
    /// Lets full-document operations like Select All read text without rendering every page.
    /// Does nothing if the text is already cached. Concurrent requests are not deduplicated,
    /// since the extraction is cheap and rarely requested.
    pub fn request_text_spans(self: Arc<Self>, cx: &mut App, page_idx: usize) -> Task<()> {
        if self.text_spans_for_page(page_idx).is_some() {
            return Task::ready(());
        }

        cx.spawn(async move |cx| {
            let pdf = self.clone();
            let bytes = self.bytes.clone();
            let password = self.password.clone();
            let metadata = self.metadata.clone();

            cx.background_executor()
                .spawn_with_priority(gpui::Priority::Low, async move {
                    match PdfRenderer::new(pdf.id, bytes, password)
                        .extract_text_only(page_idx, metadata)
                        .await
                    {
                        Ok(text_spans) => {
                            pdf.text_only_cache.write().insert(page_idx, text_spans);
                        }
                        Err(e) => {
                            log::error!("Failed to extract text for page {}: {:?}", page_idx, e);
                        }
                    }
                })
                .await;
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pdf_renderer::PageRotation;
    use gpui::TestAppContext;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn from_bytes_reports_a_graceful_error_for_a_corrupted_file_instead_of_panicking() {
        // a real PDF header and object stream truncated before the xref table and trailer, as left
        // by an interrupted download or save.
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/corrupted.pdf");
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));

        let pdf = futures::executor::block_on(Pdf::from_bytes(bytes, None, None));

        assert_eq!(pdf.metadata.page_count, 0);
        assert!(
            pdf.metadata.error.is_some(),
            "expected a load error to be surfaced, not silently swallowed"
        );
        assert!(!pdf.metadata.needs_password);
    }

    // only test bodies take `PDFIUM_TEST_LOCK`, so holding it across an await cannot deadlock.
    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn request_text_spans_extracts_without_ever_rasterizing_a_bitmap(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/solid_red.pdf");
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);
        assert_eq!(pdf.metadata.page_count, 1);

        assert!(
            pdf.text_spans_for_page(0).is_none(),
            "shouldn't have any text data before requesting it"
        );

        cx.update(|cx| pdf.clone().request_text_spans(cx, 0)).await;

        assert!(
            pdf.text_spans_for_page(0).is_some(),
            "expected text extraction to populate this page's spans - even an empty list \
             counts, since this fixture has no text of its own"
        );
        assert!(
            pdf.text_only_cache.read().contains_key(&0),
            "expected the extracted text to land in text_only_cache"
        );
        assert!(
            !pdf.page_cache.read().contains_key(&0),
            "text-only extraction must not rasterize a bitmap into page_cache - that's the \
             whole point of decoupling it from process_page (this is what lets Select All \
             cover the whole document without forcing every page to render, matching Chrome)"
        );
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn text_spans_for_page_prefers_a_full_render_over_text_only_extraction(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/solid_red.pdf");
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);

        let task = cx
            .update(|cx| {
                pdf.clone().request_page(
                    cx,
                    0,
                    1.0,
                    0.0,
                    PageRotation::None.into(),
                    gpui::Priority::High,
                    |_| {},
                )
            })
            .expect("first request for an uncached page should return a task");
        task.await;
        assert!(pdf.page_cache.read().contains_key(&0));

        assert!(
            pdf.text_spans_for_page(0).is_some(),
            "a full render already extracted this page's text as a side effect - \
             text_spans_for_page should find it there without needing text_only_cache at all"
        );
        assert!(
            pdf.text_only_cache.read().is_empty(),
            "no need to separately text-only-extract a page that's already been fully rendered"
        );
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn request_page_renders_once_then_short_circuits_from_cache(cx: &mut TestAppContext) {
        // real PDFium calls on a background task need to block a thread.
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/solid_red.pdf");
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);
        assert_eq!(pdf.metadata.page_count, 1);

        let rendered = Arc::new(AtomicBool::new(false));
        let rendered_in_callback = rendered.clone();
        let task = cx.update(|cx| {
            pdf.clone().request_page(
                cx,
                0,
                1.0,
                0.0,
                PageRotation::None.into(),
                gpui::Priority::High,
                move |_| rendered_in_callback.store(true, Ordering::SeqCst),
            )
        });
        let task = task.expect("first request for an uncached page should return a task");
        task.await;

        assert!(
            rendered.load(Ordering::SeqCst),
            "on_rendered callback should fire once the page finishes"
        );
        assert!(pdf.page_cache.read().contains_key(&0));

        // a cached page at the same or lower resolution short-circuits, so speculative prefetch is
        // cheap.
        let second = cx.update(|cx| {
            pdf.clone().request_page(
                cx,
                0,
                1.0,
                0.0,
                PageRotation::None.into(),
                gpui::Priority::Low,
                |_| {},
            )
        });
        assert!(
            second.is_none(),
            "expected a cached page at the same resolution to short-circuit"
        );
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn reuse_rendered_pages_copies_unchanged_pages_without_internal_links(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_fixtures/two_pages_with_text.pdf"
        );
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let previous = Arc::new(Pdf::from_bytes(bytes.clone(), None, None).await);
        assert_eq!(previous.metadata.page_count, 2);
        for page_idx in 0..2 {
            let task = cx.update(|cx| {
                previous.clone().request_page(
                    cx,
                    page_idx,
                    1.0,
                    0.0,
                    PageRotation::None.into(),
                    gpui::Priority::High,
                    |_| {},
                )
            });
            task.expect("uncached page should render").await;
        }

        let pdf = Pdf::from_bytes(bytes, None, None).await;
        pdf.reuse_rendered_pages(&previous, |page_idx| page_idx == 1);
        assert_eq!(
            pdf.page_cache.read().keys().copied().collect::<Vec<_>>(),
            vec![1]
        );

        if let Some(data) = previous.page_cache.write().get_mut(&1) {
            data.links = Arc::new(vec![crate::pdf_renderer::PdfLink {
                bounds: gpui::Bounds::default(),
                action: PdfLinkAction::Internal(crate::pdf_renderer::PdfTarget {
                    page_index: 0,
                    point: gpui::Point::default(),
                    block_bottom_y: None,
                    line_height: None,
                }),
            }]);
        }
        let pdf = Pdf::from_bytes(Vec::new(), None, None).await;
        pdf.reuse_rendered_pages(&previous, |_| true);
        assert!(
            pdf.page_cache.read().is_empty(),
            "pages of another size must not be reused"
        );
        let pdf = Pdf::from_bytes(previous.bytes.to_vec(), None, None).await;
        pdf.reuse_rendered_pages(&previous, |_| true);
        assert_eq!(
            pdf.page_cache.read().keys().copied().collect::<Vec<_>>(),
            vec![0],
            "a page linking into the document must be rendered again"
        );
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn a_different_visible_center_re_renders_an_oversized_page(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/tall_page.pdf");
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);
        let page_height_pt = pdf.metadata.pages[0].size.height;

        let top_task = cx
            .update(|cx| {
                pdf.clone().request_page(
                    cx,
                    0,
                    1.0,
                    0.0,
                    PageRotation::None.into(),
                    gpui::Priority::High,
                    |_| {},
                )
            })
            .expect("first request for an uncached page should return a task");
        top_task.await;

        let top_offset = pdf
            .page_cache
            .read()
            .get(&0)
            .expect("page should be cached after rendering")
            .page_offset_pt;

        // the cached strip does not cover the far end of the page, so this must render again.
        let bottom_task = cx
            .update(|cx| {
                pdf.clone().request_page(
                    cx,
                    0,
                    1.0,
                    page_height_pt,
                    PageRotation::None.into(),
                    gpui::Priority::High,
                    |_| {},
                )
            })
            .expect("a visible_center_y outside the cached strip should trigger a re-render");
        bottom_task.await;

        let bottom_offset = pdf
            .page_cache
            .read()
            .get(&0)
            .expect("page should still be cached after re-rendering")
            .page_offset_pt;

        assert!(
            bottom_offset > top_offset,
            "expected the re-render to move the strip toward the bottom of the page"
        );
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn request_thumbnail_renders_once_then_short_circuits_from_its_own_cache(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/solid_red.pdf");
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);

        let rendered = Arc::new(AtomicBool::new(false));
        let rendered_in_callback = rendered.clone();
        let task = cx.update(|cx| {
            pdf.clone().request_thumbnail(cx, 0, false, move |_| {
                rendered_in_callback.store(true, Ordering::SeqCst)
            })
        });
        let task = task.expect("first thumbnail request for an uncached page should return a task");
        task.await;

        assert!(rendered.load(Ordering::SeqCst));
        assert!(pdf.thumbnail_cache.read().contains_key(&0));
        // thumbnails do not touch the main page cache.
        assert!(!pdf.page_cache.read().contains_key(&0));

        let second = cx.update(|cx| pdf.clone().request_thumbnail(cx, 0, false, |_| {}));
        assert!(
            second.is_none(),
            "expected a cached thumbnail to short-circuit rather than re-render"
        );

        let inverted = cx.update(|cx| pdf.clone().request_thumbnail(cx, 0, true, |_| {}));
        inverted
            .expect("a thumbnail in another style should render again")
            .await;
        assert!(pdf.thumbnail_cache.read()[&0].style.inverted);
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn thumbnails_outside_the_kept_range_are_evicted(cx: &mut TestAppContext) {
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
        for page_idx in 0..2 {
            let task = cx.update(|cx| pdf.clone().request_thumbnail(cx, page_idx, false, |_| {}));
            task.expect("uncached thumbnail should render").await;
        }

        pdf.evict_thumbnails_outside(1..2);
        assert_eq!(
            pdf.thumbnail_cache
                .read()
                .keys()
                .copied()
                .collect::<Vec<_>>(),
            vec![1]
        );
    }

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn render_failure_is_recorded_and_not_retried_every_frame(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/solid_red.pdf");
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);
        assert_eq!(pdf.metadata.page_count, 1);

        // page 5 does not exist, so the real render failure path runs.
        let task = cx
            .update(|cx| {
                pdf.clone().request_page(
                    cx,
                    5,
                    1.0,
                    0.0,
                    PageRotation::None.into(),
                    gpui::Priority::High,
                    |_| {},
                )
            })
            .expect("a page never rendered before should still return a task, even a doomed one");
        task.await;

        assert!(
            pdf.failed_pages.lock().contains_key(&5),
            "expected the render failure to be recorded in failed_pages"
        );
        assert!(!pdf.page_cache.read().contains_key(&5));

        // a repeated request for a failed page must not retry the render.
        let second = cx.update(|cx| {
            pdf.clone().request_page(
                cx,
                5,
                1.0,
                0.0,
                PageRotation::None.into(),
                gpui::Priority::High,
                |_| {},
            )
        });
        assert!(
            second.is_none(),
            "expected a page already known to fail at this scale to be skipped, not retried"
        );
    }
}
