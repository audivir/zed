use crate::pdf_renderer::{
    DocumentMetadata, PageData, PageMetadata, PageRotation, PdfRenderer, PdfTextSpan,
};
use anyhow::{Result, bail};
use collections::{HashMap, HashSet};
use gpui::{App, Size, Task, hash};
use parking_lot::{Mutex, RwLock};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// The render scale used for thumbnail-sidebar images. Fixed rather than zoom-dependent,
/// since thumbnails never need to sharpen with the main view's zoom level, so unlike the
/// main page cache, a thumbnail is either cached at this one scale or not cached at all.
const THUMBNAIL_SCALE: f32 = 0.2;

/// Assure pages are removed even if GPUI cancels a task
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

/// Same purpose as `Guard`, but for the separate thumbnail-rendering bookkeeping.
struct ThumbnailGuard {
    pdf: Arc<Pdf>,
    page: usize,
}

impl Drop for ThumbnailGuard {
    fn drop(&mut self) {
        self.pdf.rendering_thumbnails.lock().remove(&self.page);
    }
}

/// A PDF document, containing its raw bytes and a cache of rendered frames
#[derive(Debug)]
pub struct Pdf {
    /// The unique ID for the PDF
    pub id: u64,
    /// The raw PDF bytes
    pub bytes: Arc<Vec<u8>>,
    /// Document metadata about pages and chapters
    pub metadata: Arc<DocumentMetadata>,
    /// Cache of rendered pages: Key is page_index
    pub page_cache: RwLock<HashMap<usize, PageData>>,
    /// Text extracted without ever rendering a bitmap, see `request_text_spans`. Consulted
    /// as a fallback by `text_spans_for_page` for pages `page_cache` doesn't have yet (a
    /// full render already extracts text as a side effect, so this only ever fills in for
    /// pages that haven't been rasterized).
    pub text_only_cache: RwLock<HashMap<usize, Arc<Vec<PdfTextSpan>>>>,
    /// Tracking currently rendering pages, and the scale/rotation they're rendering at
    pub rendering_pages: Mutex<HashMap<usize, (f32, PageRotation)>>,
    /// Pages whose most recent render attempt failed, and the scale/rotation that failed,
    /// so we don't silently re-attempt (and re-fail) the same render every frame.
    pub failed_pages: Mutex<HashMap<usize, (f32, PageRotation)>>,
    /// Cache of rendered thumbnail-sidebar images, kept separate from `page_cache` since
    /// thumbnails and the main view want the same page at very different scales at the same
    /// time, and sharing one cache would mean one view constantly evicting the other's render.
    pub thumbnail_cache: RwLock<HashMap<usize, PageData>>,
    rendering_thumbnails: Mutex<HashSet<usize>>,
    failed_thumbnails: Mutex<HashSet<usize>>,
    /// The password used to open this document, if any. `Pdf` is otherwise immutable after
    /// construction, so retrying with a different password creates a new `Pdf` rather than
    /// mutating this one.
    pub password: Option<String>,
}

impl Drop for Pdf {
    fn drop(&mut self) {
        // Best-effort: releases the cached, parsed PdfDocument for this content hash once
        // nothing references it anymore, so closed PDFs don't accumulate in the renderer's
        // document cache for the lifetime of the process.
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
    /// An empty PDF containing no data
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

    /// Create a PDF from bytes, optionally unlocking it with `password` if it's encrypted.
    /// If the document turns out to be password-protected, check
    /// `metadata.needs_password` on the result rather than treating this as a load failure.
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

    /// Get this PDF's ID
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get the raw bytes of the PDF
    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    /// Get the size of a specific page
    pub fn get_page_size(&self, page_idx: usize) -> Result<Size<f32>> {
        if let Some(page_metadata) = self.metadata.pages.get(page_idx) {
            return Ok(page_metadata.size);
        }
        bail!("Out of bounds")
    }

    /// Request the parsing and rendering of a page at a specific scale. `visible_center_y`
    /// is the page-point-space Y coordinate the caller is most interested in (e.g. the
    /// vertical center of what's currently visible), only relevant for pages too tall to
    /// render whole at this scale, where it decides which vertical strip gets rendered (see
    /// `MAX_RENDER_HEIGHT` in `pdf_renderer.rs`); pass `0.0` if the caller has no viewport
    /// context (e.g. search's adjacent-match prefetch). `priority` controls how eagerly this
    /// render is scheduled relative to other pending page renders (e.g. a currently-visible
    /// page should use `Priority::High`, a prerender/prefetch page `Priority::Low`), see the
    /// comment at this function's call sites in `pdf_viewer.rs`.
    #[allow(clippy::too_many_arguments)]
    pub fn request_page(
        self: Arc<Self>,
        cx: &mut App,
        page_idx: usize,
        requested_scale: f32,
        visible_center_y: f32,
        rotation: PageRotation,
        priority: gpui::Priority,
        on_rendered: impl FnOnce(&mut App) + Send + 'static,
    ) -> Option<Task<()>> {
        // Fast path: check if we already have an image at this resolution or higher, at the
        // right rotation, that also covers the page-point we're interested in (relevant for
        // strip-rendered pages; always true for an ordinary whole-page render).
        let prev_page = {
            // Acquire the read lock
            let cache = self.page_cache.read();

            if let Some(rendered) = cache.get(&page_idx) {
                // Buffer of 0.01 handles tiny float precision mismatches
                if rendered.scale >= requested_scale * 0.99
                    && rendered.rotation == rotation
                    && rendered.covers_page_y(visible_center_y)
                {
                    return None; // We already have a good enough render, abort the task
                }
                // A cached render at the wrong rotation isn't reusable as a scale/coverage
                // fallback either: its pixels are the wrong shape for the new rotation.
                (rendered.rotation == rotation).then(|| rendered.clone())
            } else {
                None
            }
        };

        {
            let failed = self.failed_pages.lock();
            if let Some(&(failed_scale, failed_rotation)) = failed.get(&page_idx)
                && failed_scale >= requested_scale * 0.99
                && failed_rotation == rotation
            {
                return None; // Already know this render fails at this resolution; don't retry every frame
            }
        }

        {
            let mut rendering = self.rendering_pages.lock();
            if let Some(&(rendering_scale, rendering_rotation)) = rendering.get(&page_idx)
                && rendering_scale >= requested_scale * 0.99
                && rendering_rotation == rotation
            {
                return None; // Already queuing a render of equal or better resolution
            }
            rendering.insert(page_idx, (requested_scale, rotation));
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
                            rotation,
                            prev_page,
                            pdf.metadata.clone(),
                        )
                        .await
                    {
                        Ok(new_page) => {
                            pdf.failed_pages.lock().remove(&page_idx);

                            let mut cache = pdf.page_cache.write();

                            // Overwrite only if the new render is better than what snuck into
                            // the cache, or that cached render is now stale because it's at a
                            // different rotation.
                            let should_insert = cache
                                .get(&page_idx)
                                .is_none_or(|c| c.scale <= requested_scale || c.rotation != rotation);
                            if should_insert {
                                cache.insert(page_idx, new_page);
                                log::info!("Written to cache");
                            }
                        }
                        Err(e) => {
                            log::error!("Failed to render page {}: {:?}", page_idx, e);
                            pdf.failed_pages
                                .lock()
                                .insert(page_idx, (requested_scale, rotation));
                        }
                    }
                })
                .await;

            // Notify the UI safely
            cx.update(|cx| on_rendered(cx));
        }))
    }

    /// Requests a low-resolution thumbnail render of `page_idx` for the thumbnail sidebar,
    /// at a fixed `THUMBNAIL_SCALE` rather than the caller-chosen scale `request_page` takes -
    /// see `thumbnail_cache`'s doc comment for why thumbnails use a separate cache/request
    /// path instead of sharing `request_page`'s.
    pub fn request_thumbnail(
        self: Arc<Self>,
        cx: &mut App,
        page_idx: usize,
        on_rendered: impl FnOnce(&mut App) + Send + 'static,
    ) -> Option<Task<()>> {
        if self.thumbnail_cache.read().contains_key(&page_idx) {
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
                        // Thumbnails always render unrotated. They're a small navigation
                        // aid, not expected to visually match the (possibly rotated) main
                        // view pixel-for-pixel.
                        .process_page(
                            page_idx,
                            THUMBNAIL_SCALE,
                            0.0,
                            PageRotation::None,
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

    /// `page_idx`'s text spans, from whichever cache already has them: `page_cache` (a full
    /// render already extracts text as a side effect) if present, else `text_only_cache` (see
    /// `request_text_spans`). `None` if neither has extracted this page's text yet.
    pub fn text_spans_for_page(&self, page_idx: usize) -> Option<Arc<Vec<PdfTextSpan>>> {
        if let Some(data) = self.page_cache.read().get(&page_idx) {
            return Some(data.text_spans.clone());
        }
        self.text_only_cache.read().get(&page_idx).cloned()
    }

    /// Extracts and caches `page_idx`'s text layer without rasterizing a bitmap at all (see
    /// `PdfRenderer::extract_text_only`). Lets full-document operations like Select All read
    /// every page's text without forcing every page to render first, the same way Chrome/
    /// Edge's own PDFium-based viewer decouples text extraction from rendering. A no-op if
    /// the text is already available via either cache. Unlike `request_page`, this doesn't
    /// dedup against a concurrent in-flight request for the same page: a plain cache miss
    /// here just means redoing some (cheap, non-rasterizing) PDFium work, and callers of this
    /// (currently just Select All, invoked at most a handful of times a session) don't queue
    /// up densely enough for that to matter the way it does for prerendering's per-frame
    /// `request_page` calls.
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
    use gpui::TestAppContext;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn from_bytes_reports_a_graceful_error_for_a_corrupted_file_instead_of_panicking() {
        // Not a garbage/empty buffer, but a real PDF header and object stream, truncated
        // partway through (missing its xref table and trailer entirely). This is what
        // actually shows up in practice (an interrupted download/save, a disk full mid-write)
        // more than a fully empty or non-PDF file.
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

    // PDFIUM_TEST_LOCK is only ever taken by test bodies themselves, never by production
    // code or another task within the same test, so holding it across an await here can't
    // deadlock; it's exactly the kind of case this lint can't distinguish from a real one.
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
                pdf.clone()
                    .request_page(cx, 0, 1.0, 0.0, PageRotation::None, gpui::Priority::High, |_| {})
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
        // This exercises real PDFium FFI calls and a real (non-gpui) async mutex on a
        // background task, not simulated work gpui's deterministic test scheduler can drive
        // itself; it needs to actually block a thread.
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
                PageRotation::None,
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

        // Same page, same-or-lower resolution, already cached: should short-circuit rather
        // than spawn another render (this is what makes it safe for callers like search's
        // adjacent-match prefetch to call request_page speculatively on every match).
        let second = cx.update(|cx| {
            pdf.clone()
                .request_page(cx, 0, 1.0, 0.0, PageRotation::None, gpui::Priority::Low, |_| {})
        });
        assert!(
            second.is_none(),
            "expected a cached page at the same resolution to short-circuit"
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
                    PageRotation::None,
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

        // Same page, same resolution, but the far end of the page: the cached strip from
        // the first request doesn't cover this point, so this must NOT short-circuit like
        // the ordinary (same-point) cache-hit case does.
        let bottom_task = cx
            .update(|cx| {
                pdf.clone().request_page(
                    cx,
                    0,
                    1.0,
                    page_height_pt,
                    PageRotation::None,
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
            pdf.clone()
                .request_thumbnail(cx, 0, move |_| rendered_in_callback.store(true, Ordering::SeqCst))
        });
        let task = task.expect("first thumbnail request for an uncached page should return a task");
        task.await;

        assert!(rendered.load(Ordering::SeqCst));
        assert!(pdf.thumbnail_cache.read().contains_key(&0));
        // The main page cache is untouched: thumbnails use their own cache/scale entirely.
        assert!(!pdf.page_cache.read().contains_key(&0));

        let second = cx.update(|cx| pdf.clone().request_thumbnail(cx, 0, |_| {}));
        assert!(
            second.is_none(),
            "expected a cached thumbnail to short-circuit rather than re-render"
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

        // Page 5 doesn't exist on this 1-page document, so process_page will fail with
        // "Page out of bounds", exercising the real render-failure path end to end, rather
        // than just asserting on manually-populated state.
        let task = cx
            .update(|cx| {
                pdf.clone()
                    .request_page(cx, 5, 1.0, 0.0, PageRotation::None, gpui::Priority::High, |_| {})
            })
            .expect("a page never rendered before should still return a task, even a doomed one");
        task.await;

        assert!(
            pdf.failed_pages.lock().contains_key(&5),
            "expected the render failure to be recorded in failed_pages"
        );
        assert!(!pdf.page_cache.read().contains_key(&5));

        // A second request for the same page at the same scale must short-circuit instead of
        // re-attempting (and re-failing) the same render on every subsequent frame.
        let second = cx.update(|cx| {
            pdf.clone()
                .request_page(cx, 5, 1.0, 0.0, PageRotation::None, gpui::Priority::High, |_| {})
        });
        assert!(
            second.is_none(),
            "expected a page already known to fail at this scale to be skipped, not retried"
        );
    }
}
