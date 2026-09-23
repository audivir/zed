use crate::{pdf::Pdf, pdf_renderer::PdfRenderer};
use anyhow::Context as _;
use gpui::{DismissEvent, EventEmitter, FocusHandle, Focusable, Task};
use std::{path::PathBuf, sync::Arc};
use ui::{Modal, ModalHeader, Section, SectionHeader, prelude::*};
use workspace::{ModalView, notifications::DetachAndPromptErr as _};

const POINTS_PER_MILLIMETER: f32 = 72.0 / 25.4;

/// Shows the document information of a PDF, like the properties dialogs of other viewers.
pub struct DocumentPropertiesModal {
    pdf: Arc<Pdf>,
    rows: Vec<(&'static str, String)>,
    /// Directory the save dialog for attachments starts in.
    directory: PathBuf,
    focus_handle: FocusHandle,
}

impl DocumentPropertiesModal {
    pub fn new(
        pdf: Arc<Pdf>,
        page_index: usize,
        directory: PathBuf,
        cx: &mut Context<Self>,
    ) -> Self {
        Self {
            rows: property_rows(&pdf, page_index),
            pdf,
            directory,
            focus_handle: cx.focus_handle(),
        }
    }

    /// Asks where to save the embedded file at `index` and writes it there.
    fn save_attachment(
        &mut self,
        index: usize,
        cx: &mut Context<Self>,
    ) -> Task<anyhow::Result<()>> {
        let Some(attachment) = self.pdf.metadata.attachments.get(index) else {
            return Task::ready(Ok(()));
        };
        let save_dialog = cx.prompt_for_new_path(&self.directory, Some(&attachment.name));
        let pdf = self.pdf.clone();
        cx.spawn(async move |_, cx| {
            let Some(path) = save_dialog.await?? else {
                return Ok(());
            };
            cx.background_spawn(async move {
                let bytes = PdfRenderer::new(pdf.id, pdf.bytes.clone(), pdf.password.clone())
                    .attachment_bytes(index)
                    .await?;
                std::fs::write(&path, bytes).with_context(|| format!("writing {path:?}"))
            })
            .await
        })
    }

    fn cancel(&mut self, _: &menu::Cancel, _window: &mut Window, cx: &mut Context<Self>) {
        cx.emit(DismissEvent);
    }
}

/// Returns the labeled properties of `pdf`, with the page size of `page_index`.
pub(crate) fn property_rows(pdf: &Pdf, page_index: usize) -> Vec<(&'static str, String)> {
    let properties = &pdf.metadata.properties;
    let mut rows = [
        ("Title", properties.title.clone()),
        ("Author", properties.author.clone()),
        ("Subject", properties.subject.clone()),
        ("Keywords", properties.keywords.clone()),
        ("Created", properties.created.clone()),
        ("Application", properties.creator.clone()),
        ("PDF Producer", properties.producer.clone()),
        ("PDF Version", properties.version.clone()),
    ]
    .into_iter()
    .filter_map(|(label, value)| Some((label, value?)))
    .collect::<Vec<_>>();

    rows.push(("Pages", pdf.metadata.page_count.to_string()));
    if let Ok(size) = pdf.get_page_size(page_index) {
        rows.push((
            "Page Size",
            format!(
                "{:.0} × {:.0} mm",
                size.width / POINTS_PER_MILLIMETER,
                size.height / POINTS_PER_MILLIMETER
            ),
        ));
    }
    rows.push((
        "File Size",
        util::size::format_file_size(pdf.bytes.len() as u64, true),
    ));
    rows
}

impl EventEmitter<DismissEvent> for DocumentPropertiesModal {}

impl ModalView for DocumentPropertiesModal {}

impl Focusable for DocumentPropertiesModal {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for DocumentPropertiesModal {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let rows = self.rows.iter().map(|(label, value)| {
            h_flex()
                .gap_4()
                .items_start()
                .child(
                    div()
                        .w(rems(7.))
                        .flex_none()
                        .child(Label::new(*label).color(Color::Muted)),
                )
                .child(div().flex_1().min_w_0().child(Label::new(value.clone())))
        });
        let attachments =
            self.pdf
                .metadata
                .attachments
                .iter()
                .enumerate()
                .map(|(index, attachment)| {
                    h_flex()
                        .gap_2()
                        .justify_between()
                        .child(
                            h_flex()
                                .gap_2()
                                .min_w_0()
                                .child(Label::new(attachment.name.clone()))
                                .child(
                                    Label::new(util::size::format_file_size(
                                        attachment.size as u64,
                                        true,
                                    ))
                                    .color(Color::Muted),
                                ),
                        )
                        .child(Button::new(("save-attachment", index), "Save…").on_click(
                            cx.listener(move |this, _, window, cx| {
                                this.save_attachment(index, cx).detach_and_prompt_err(
                                    "Failed to save the attachment",
                                    window,
                                    cx,
                                    |_, _, _| None,
                                );
                            }),
                        ))
                })
                .collect::<Vec<_>>();

        div()
            .track_focus(&self.focus_handle)
            .key_context("DocumentPropertiesModal")
            .on_action(cx.listener(Self::cancel))
            .elevation_3(cx)
            .w(rems(32.))
            .child(
                Modal::new("pdf-document-properties", None)
                    .header(
                        ModalHeader::new()
                            .show_dismiss_button(true)
                            .child(Headline::new("Document Properties").size(HeadlineSize::Small)),
                    )
                    .section(Section::new().child(v_flex().gap_1().pb_2().children(rows)))
                    .when(!attachments.is_empty(), |modal| {
                        modal.section(
                            Section::new()
                                .header(SectionHeader::new("Attachments"))
                                .child(v_flex().gap_1().pb_2().children(attachments)),
                        )
                    }),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::await_holding_lock)]
    #[gpui::test]
    async fn attachments_are_saved_where_the_user_picks(cx: &mut gpui::TestAppContext) {
        cx.update(|cx| {
            let store = settings::SettingsStore::test(cx);
            cx.set_global(store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
        cx.executor().allow_parking();
        let _guard = crate::pdf_renderer::PDFIUM_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_fixtures/embedded_attachment.pdf"
        );
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
        let pdf = Arc::new(Pdf::from_bytes(bytes, None, None).await);
        let destination = std::env::temp_dir().join(format!(
            "pdf_viewer_attachment_test_{}.txt",
            std::process::id()
        ));

        let (modal, cx) = cx.add_window_view(|_window, cx| {
            DocumentPropertiesModal::new(pdf.clone(), 0, PathBuf::default(), cx)
        });
        let save = modal.update(cx, |modal, cx| modal.save_attachment(0, cx));
        cx.simulate_new_path_selection({
            let destination = destination.clone();
            |_| Some(destination)
        });
        save.await.expect("saving the attachment should succeed");

        let saved = std::fs::read(&destination).expect("attachment should be written");
        std::fs::remove_file(&destination).ok();
        assert_eq!(saved, b"hello from an attachment\n");
    }
}
