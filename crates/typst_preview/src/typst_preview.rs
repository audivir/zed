use gpui::{App, actions};
use workspace::Workspace;
pub mod typst_preview_view;
pub mod typst_world;

// Shared with the editor crate's mouse-context-menu/keymap (see zed_actions::preview). Editor
// can't depend on this crate, so these two live in a dependency-free shared spot the same way
// markdown/svg preview's OpenPreview/OpenPreviewToTheSide do) rather than this crate's own
// namespace.
pub use zed_actions::preview::typst::{OpenPreview, OpenPreviewToTheSide};

actions!(
    typst_preview,
    [
        /// Opens a preview that follows the active Typst editor.
        OpenFollowingPreview,
        /// Opens a preview strictly for the active Typst document, ignoring project root.
        OpenFixedPreview,
        /// Opens a preview strictly for the active Typst document in a split pane.
        OpenFixedPreviewToTheSide,
        /// Opens a preview that follows the active editor, always compiling the active file directly.
        OpenFixedFollowingPreview
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
