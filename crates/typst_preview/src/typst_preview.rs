use gpui::{App, actions};
use workspace::Workspace;
pub mod typst_preview_view;
pub mod typst_world;

// these actions live in `zed_actions` because the editor context menu needs them and cannot
// depend on this crate.
pub use zed_actions::preview::typst::{OpenPreview, OpenPreviewToTheSide};

actions!(
    typst_preview,
    [
        /// Opens a preview that follows the active Typst editor.
        OpenFollowingPreview,
        /// Opens a preview for the active Typst document, ignoring the project root.
        OpenFixedPreview,
        /// Opens a preview for the active Typst document in a split pane.
        OpenFixedPreviewToTheSide,
        /// Opens a following preview that always compiles the active file directly.
        OpenFixedFollowingPreview,
        /// Writes the compiled PDF of the previewed document next to its main file.
        ExportPdf
    ]
);

pub fn init(cx: &mut App) {
    cx.observe_new(|workspace: &mut Workspace, window, cx| {
        let Some(window) = window else {
            return;
        };
        typst_preview_view::TypstPreviewView::register(workspace, window, cx);
    })
    .detach();
}
