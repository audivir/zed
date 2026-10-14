use gpui::{App, actions};
use workspace::Workspace;

pub mod svg_preview_view;

pub use svg_preview_view::{SvgPreviewToolbarControls, SvgPreviewView};
pub use zed_actions::preview::svg::{OpenPreview, OpenPreviewToTheSide};

actions!(
    svg,
    [
        /// Opens a following SVG preview that syncs with the editor.
        OpenFollowingPreview,
        /// Opens the source of the previewed SVG in a text editor.
        OpenAsText
    ]
);

pub fn init(cx: &mut App) {
    workspace::register_project_item::<SvgPreviewView>(cx);
    cx.observe_new(|workspace: &mut Workspace, window, cx| {
        let Some(window) = window else {
            return;
        };
        crate::svg_preview_view::SvgPreviewView::register(workspace, window, cx);
    })
    .detach();
}
