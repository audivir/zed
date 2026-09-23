use crate::pdf::Pdf;
use gpui::{DismissEvent, EventEmitter, FocusHandle, Focusable};
use ui::{Modal, ModalHeader, Section, prelude::*};
use workspace::ModalView;

const POINTS_PER_MILLIMETER: f32 = 72.0 / 25.4;

/// Shows the document information of a PDF, like the properties dialogs of other viewers.
pub struct DocumentPropertiesModal {
    rows: Vec<(&'static str, String)>,
    focus_handle: FocusHandle,
}

impl DocumentPropertiesModal {
    pub fn new(pdf: &Pdf, page_index: usize, cx: &mut Context<Self>) -> Self {
        Self {
            rows: property_rows(pdf, page_index),
            focus_handle: cx.focus_handle(),
        }
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
                    .section(Section::new().child(v_flex().gap_1().pb_2().children(rows))),
            )
    }
}
