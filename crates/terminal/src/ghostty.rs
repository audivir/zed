use std::{
    borrow::Cow,
    cell::Cell,
    io::{ErrorKind, Read, Write},
    sync::{
        Arc,
        mpsc::{Receiver, SyncSender, TryRecvError, TrySendError},
    },
    thread,
};

use anyhow::{Context as _, Result};
use futures::channel::mpsc::UnboundedSender;
use libghostty_vt::{
    TerminalOptions,
    kitty::graphics::{self, Compression, ImageFormat, PlacementIterator},
    render::{CellIterator, CursorVisualStyle, RowIterator},
    screen,
    style::{self, StyleColor},
    terminal::{CursorStyle, ScrollViewport},
};
use parking_lot::Mutex as ParkingMutex;
use portable_pty::{CommandBuilder, PtySize};

use crate::{
    CursorShape, PtyEvent, Scroll, TerminalBounds, ViMotion,
    hyperlinks::{path_hyperlink_candidates_in_line, trim_url_punctuation},
    pty_info::{ProcessIdGetter, PtyProcessInfo},
};

/// Maximum bytes of decoded Kitty graphics image data libghostty-vt will
/// retain per terminal, matching Ghostty's own default
/// `--kitty-image-storage-limit`.
const KITTY_IMAGE_STORAGE_LIMIT_BYTES: u64 = 320 * 1024 * 1024;

/// Size of each recycled PTY read buffer.
const PTY_READ_BUFFER_BYTES: usize = 64 * 1024;
/// Upper bound on how many bytes the parser thread batches into a single
/// `libghostty-vt` write before handing effects back to the UI thread.
const MAX_PTY_PARSE_BATCH_BYTES: usize = 512 * 1024;
/// Bound on the reader-to-parser queue, so a slow parser applies backpressure
/// to PTY reads instead of buffering unboundedly.
const MAX_QUEUED_PTY_OUTPUT_BUFFERS: usize = 512;
/// Bound on the buffer-recycling channel back to the reader thread.
const MAX_RECYCLED_PTY_BUFFERS: usize = 64;

pub(super) struct GhosttyContentMetadata {
    pub(super) cursor: crate::Cursor,
    pub(super) cursor_char: char,
    pub(super) scrolled_to_top: bool,
    pub(super) scrolled_to_bottom: bool,
    pub(super) cursor_blinking: bool,
}

#[derive(Debug)]
pub(super) enum GhosttyEffect {
    PtyWrite(Vec<u8>),
    Bell,
    TitleChanged(String),
    /// An OSC 52 (or iTerm2 OSC 1337 Copy) clipboard write, already
    /// normalized to plain text.
    ClipboardStore(String),
}

enum PtyCommand {
    Input(Cow<'static, [u8]>),
    Resize(TerminalBounds),
    Shutdown,
}

/// Sends PTY I/O commands to the Ghostty PTY writer thread.
pub(super) struct PtySender {
    command_tx: std::sync::mpsc::Sender<PtyCommand>,
}

impl PtySender {
    pub(super) fn notify(&self, input: impl Into<Cow<'static, [u8]>>) {
        let input = input.into();
        if input.is_empty() {
            return;
        }
        if let Err(error) = self.command_tx.send(PtyCommand::Input(input)) {
            log::debug!("failed to send input to ghostty PTY: {error}");
        }
    }

    pub(super) fn resize(&self, bounds: TerminalBounds) {
        if let Err(error) = self.command_tx.send(PtyCommand::Resize(bounds)) {
            log::debug!("failed to resize ghostty PTY: {error}");
        }
    }

    pub(super) fn shutdown(&self) {
        if let Err(error) = self.command_tx.send(PtyCommand::Shutdown) {
            log::debug!("failed to shut down ghostty PTY: {error}");
        }
    }
}

/// A `libghostty-vt` terminal instance driven in parallel with the PTY, used
/// only for the metadata (cursor position/glyph, scroll position) it computes
/// more accurately than the Alacritty shadow terminal. See the field doc on
/// [`crate::Terminal::ghostty`].
///
/// Shared as `Arc<parking_lot::Mutex<GhosttyTerminal>>` between the PTY
/// parser thread (which writes PTY output into it) and the terminal entity's
/// owning thread (which reads cursor/scroll metadata during `sync`).
pub(super) struct GhosttyTerminal {
    // libghostty-vt stores callback userdata as a raw pointer into its Terminal.
    terminal: Box<libghostty_vt::Terminal<'static, 'static>>,
    render_state: libghostty_vt::RenderState<'static>,
    rows: RowIterator<'static>,
    cells: CellIterator<'static>,
    placements: PlacementIterator<'static>,
    effects: Arc<ParkingMutex<Vec<GhosttyEffect>>>,
    /// Cached, Zed-point-space mirror of whatever `Selection` is currently
    /// installed via `self.terminal.set_selection`.
    current_selection: Option<crate::SelectionRange>,
    /// The vi-mode cursor position, or `None` when vi mode is disabled.
    current_vi_cursor: Option<crate::Point>,
}

// libghostty-vt's objects are otherwise `!Send` only because they hold raw
// pointers and C callback state; the underlying library documents terminal
// IO and rendering as safe to hand off between threads as long as access is
// never concurrent. `GhosttyTerminal` upholds that itself: every method that
// touches the FFI state takes `&mut self`, and it is only ever reachable
// through `Arc<parking_lot::Mutex<GhosttyTerminal>>`, so the mutex serializes
// all cross-thread access (PTY parser thread vs. the terminal entity's
// owning thread) into non-concurrent, single-threaded-at-a-time use.
unsafe impl Send for GhosttyTerminal {}

impl GhosttyTerminal {
    pub(super) fn new(cols: u16, rows: u16, max_scrollback: usize) -> Result<Self> {
        anyhow::ensure!(cols > 0, "terminal columns must be greater than zero");
        anyhow::ensure!(rows > 0, "terminal rows must be greater than zero");
        let effects = Arc::new(ParkingMutex::new(Vec::new()));
        let mut terminal = Box::new(libghostty_vt::Terminal::new(TerminalOptions {
            cols,
            rows,
            max_scrollback,
        })?);

        terminal.on_pty_write({
            let effects = effects.clone();
            move |_terminal, data| {
                effects.lock().push(GhosttyEffect::PtyWrite(data.to_vec()));
            }
        })?;
        terminal.on_bell({
            let effects = effects.clone();
            move |_terminal| {
                effects.lock().push(GhosttyEffect::Bell);
            }
        })?;
        terminal.on_title_changed({
            let effects = effects.clone();
            move |terminal| match terminal.title() {
                Ok(title) => {
                    effects
                        .lock()
                        .push(GhosttyEffect::TitleChanged(title.to_owned()));
                }
                Err(error) => {
                    log::error!("failed to read ghostty terminal title: {error}");
                }
            }
        })?;

        // Handles OSC 52 SET (clipboard write). OSC 52 GET ("read", i.e.
        // `\x1b]52;c;?\x1b\\`) has no equivalent to wire up at all:
        // `libghostty-vt` always ignores clipboard read requests by design
        // (see this callback's own doc comment), an intentional
        // security-conscious default shared by many terminals. Multipart/
        // multi-MIME writes are flattened to their first representation's
        // text.
        terminal.on_clipboard_write({
            let effects = effects.clone();
            move |_terminal, write| {
                if let Some(content) = write.contents().next() {
                    effects
                        .lock()
                        .push(GhosttyEffect::ClipboardStore(content.data.to_owned()));
                }
                Ok(())
            }
        })?;

        terminal.set_kitty_image_storage_limit(KITTY_IMAGE_STORAGE_LIMIT_BYTES)?;
        install_png_decoder()?;

        Ok(Self {
            terminal,
            render_state: libghostty_vt::RenderState::new()?,
            rows: RowIterator::new()?,
            cells: CellIterator::new()?,
            placements: PlacementIterator::new()?,
            effects,
            current_selection: None,
            current_vi_cursor: None,
        })
    }

    pub(super) fn write(&mut self, bytes: &[u8]) {
        self.terminal.vt_write(bytes);
    }

    pub(super) fn clear(&mut self) {
        self.terminal.vt_write(b"\x1b[3J\x1b[H\x1b[2J");
    }

    pub(super) fn resize(&mut self, bounds: TerminalBounds) -> Result<()> {
        self.terminal.resize(
            bounds.num_columns().max(1) as u16,
            bounds.num_lines().max(1) as u16,
            // Rounded, not truncated: cell dimensions are rarely whole
            // pixels (fractional line heights are the common case), and
            // libghostty-vt's Kitty image row math (`gridSize`'s
            // `divCeil(image_height_px, t.height_px / t.rows)`) recovers
            // this exact per-cell value from `t.height_px = cell_height_px
            // * rows`. Truncating always biases the recovered cell size
            // down, which always biases `divCeil` up, systematically
            // undercounting by one extra row per ~20px of image height.
            // This is invisible for small images but grows without bound
            // for tall ones (e.g. a `red_square 4000` sender).
            f32::from(bounds.cell_width()).max(1.0).round() as u32,
            f32::from(bounds.line_height()).max(1.0).round() as u32,
        )?;
        Ok(())
    }

    pub(super) fn scroll_viewport(&mut self, scroll: ScrollViewport) {
        self.terminal.scroll_viewport(scroll);
    }

    /// The number of rows in the viewport (screen lines), for callers
    /// outside this module that need it without reaching into
    /// `libghostty_vt::Terminal` directly (e.g. `Terminal::sync`'s
    /// `bottom_row_occupied` computation, `Terminal::viewport_lines`).
    pub(super) fn rows(&self) -> Result<u16> {
        Ok(self.terminal.rows()?)
    }

    /// The total number of rows including scrollback history, for
    /// `Terminal::total_lines`, which the terminal scrollbar (thumb size)
    /// and various scroll-bound checks use.
    pub(super) fn total_lines(&self) -> Result<usize> {
        Ok(self.terminal.total_rows()?)
    }

    pub(super) fn take_effects(&mut self) -> Vec<GhosttyEffect> {
        self.effects.lock().drain(..).collect()
    }

    /// Converts `Terminal::scrollbar().offset` into the `display_offset`
    /// convention the rest of this crate (and `terminal_element.rs`, e.g.
    /// `to_highlighted_range_lines`'s
    /// `range.start().line.saturating_add(display_offset)`) expects
    /// wherever a `crate::Point`/`crate::Range` is converted to or from a
    /// viewport-relative row: `line + display_offset = viewport_row`. Named
    /// after Alacritty since that convention predates the Ghostty backend
    /// and the rest of the crate still speaks it.
    ///
    /// Ghostty's `Scrollbar.offset` uses the opposite polarity: it's a
    /// conventional GUI-scrollbar-style position where `0` means the
    /// viewport is scrolled all the way to the *top* of history (oldest
    /// content visible), and `total - len` means scrolled to the *bottom*
    /// (newest/live content, i.e. "not scrolled" in terminal terms).
    /// `display_offset` is the opposite: `0` means "not scrolled" (viewing
    /// the live bottom), and it *increases* as you scroll up into history.
    /// Treating `scrollbar().offset` as if it already were `display_offset`
    /// silently produces `Point`s off by exactly `scrollback_rows` whenever
    /// the viewport isn't at the live bottom, so always convert through
    /// this function rather than reading `scrollbar().offset` directly.
    pub(super) fn alacritty_style_display_offset(&self) -> Result<usize> {
        let scrollbar = self.terminal.scrollbar()?;
        Ok((scrollbar.total.saturating_sub(scrollbar.len))
            .saturating_sub(scrollbar.offset) as usize)
    }

    pub(super) fn content_metadata(
        &mut self,
        display_offset: usize,
        vi_cursor: Option<crate::Point>,
    ) -> Result<GhosttyContentMetadata> {
        let scrollbar = self.terminal.scrollbar()?;

        // In vi mode, the rendered cursor is the vi cursor, not the real
        // terminal cursor (`if vi_mode { vi_mode_cursor.point } else {
        // grid.cursor.point }`). It's always shown regardless of DECTCEM
        // visibility, since vi mode is a terminal-emulator-level UI
        // feature, not something the PTY controls: vi mode always bypasses
        // the hidden check.
        if let Some(point) = vi_cursor {
            let snapshot = self.render_state.update(&self.terminal)?;
            let cursor_style = snapshot.cursor_visual_style()?;
            let cursor_blinking = snapshot.cursor_blinking()?;
            let grid = ViGrid::new(&self.terminal)?;
            let (absolute_row, column) = grid.to_absolute(point);
            let cursor_char = grid.char_at(absolute_row, column)?;
            return Ok(GhosttyContentMetadata {
                cursor: crate::Cursor {
                    shape: ghostty_cursor_shape(cursor_style),
                    point,
                },
                cursor_char,
                scrolled_to_top: scrollbar.offset == 0,
                scrolled_to_bottom: scrollbar.offset + scrollbar.len >= scrollbar.total,
                cursor_blinking,
            });
        }

        let snapshot = self.render_state.update(&self.terminal)?;
        let cursor_viewport = snapshot.cursor_viewport()?;
        let cursor_visible = snapshot.cursor_visible()?;
        let cursor_style = snapshot.cursor_visual_style()?;
        let cursor_blinking = snapshot.cursor_blinking()?;
        let mut cursor_char = ' ';

        if let Some(cursor) = cursor_viewport {
            let mut rows = self.rows.update(&snapshot)?;
            let mut row_index = 0u16;
            while let Some(row) = rows.next() {
                if row_index == cursor.y {
                    let mut cells = self.cells.update(row)?;
                    cells.select(cursor.x)?;
                    if let Some(character) = cells.graphemes()?.first() {
                        cursor_char = *character;
                    }
                    break;
                }
                row_index += 1;
            }
        }

        // `cursor_viewport` is `None` when the cursor has scrolled out of
        // view; hide it instead of falling back to a fake (0, 0) position.
        let cursor = crate::Cursor {
            shape: if cursor_visible && cursor_viewport.is_some() {
                ghostty_cursor_shape(cursor_style)
            } else {
                CursorShape::Hidden
            },
            point: cursor_viewport
                .map(|cursor| {
                    crate::Point::new(
                        cursor.y as i32 - display_offset as i32,
                        cursor.x as usize,
                    )
                })
                .unwrap_or_else(|| crate::Point::new(0, 0)),
        };

        Ok(GhosttyContentMetadata {
            cursor,
            cursor_char,
            scrolled_to_top: scrollbar.offset == 0,
            scrolled_to_bottom: scrollbar.offset + scrollbar.len >= scrollbar.total,
            cursor_blinking,
        })
    }

    /// Builds the terminal's viewport-relative cell grid, mode flags, and
    /// scroll offset directly from Ghostty. `Terminal::refresh_last_content_from_ghostty`
    /// calls this on every `sync()`.
    pub(super) fn build_content(
        &mut self,
    ) -> Result<(Vec<crate::IndexedCell>, crate::Modes, usize)> {
        let snapshot = self.render_state.update(&self.terminal)?;
        let mut rows = self.rows.update(&snapshot)?;
        let total_cells = usize::from(self.terminal.rows().unwrap_or(24))
            * usize::from(self.terminal.cols().unwrap_or(80));
        let mut cells = Vec::with_capacity(total_cells);
        let mut row_index: i32 = 0;
        // Reused across cells so the common case (no combining/zero-width marks, well
        // under this size) never allocates just to read a cell's grapheme cluster; only
        // clusters longer than this fall back to a per-cell Vec.
        let mut grapheme_buf = [' '; 32];

        while let Some(row) = rows.next() {
            let mut row_cells = self.cells.update(row)?;
            let mut col_index: usize = 0;

            while let Some(cell) = row_cells.next() {
                let style = cell.style()?;
                let raw_cell = cell.raw_cell()?;
                let graphemes_len = cell.graphemes_len()?;
                let (character, zerowidth) = if graphemes_len == 0 {
                    (' ', Vec::new())
                } else if graphemes_len == 1 {
                    let buf = &mut grapheme_buf[..1];
                    cell.graphemes_buf(buf)?;
                    (buf[0], Vec::new())
                } else if graphemes_len <= grapheme_buf.len() {
                    let buf = &mut grapheme_buf[..graphemes_len];
                    cell.graphemes_buf(buf)?;
                    (buf[0], buf[1..].to_vec())
                } else {
                    let mut graphemes = vec!['\0'; graphemes_len];
                    cell.graphemes_buf(&mut graphemes)?;
                    (graphemes[0], graphemes[1..].to_vec())
                };

                cells.push(crate::IndexedCell {
                    point: crate::Point::new(row_index, col_index),
                    cell: crate::Cell {
                        character,
                        zerowidth,
                        foreground: color_from_style_color(
                            style.fg_color,
                            vte::ansi::NamedColor::Foreground,
                        ),
                        background: color_from_style_color(
                            style.bg_color,
                            vte::ansi::NamedColor::Background,
                        ),
                        // OSC 8 hyperlinks aren't populated here.
                        hyperlink: None,
                        is_bold: style.bold,
                        is_italic: style.italic,
                        is_dim: style.faint,
                        is_inverse: style.inverse,
                        is_wide_char_spacer: raw_cell.wide()? == screen::CellWide::SpacerTail,
                        has_underline: style.underline != style::Underline::None,
                        has_undercurl: style.underline == style::Underline::Curly,
                        has_strikeout: style.strikethrough,
                    },
                });
                col_index += 1;
            }
            row_index += 1;
        }

        let modes = self.modes()?;
        let display_offset = self.alacritty_style_display_offset()?;
        Ok((cells, modes, display_offset))
    }

    fn modes(&self) -> Result<crate::Modes> {
        use libghostty_vt::terminal::Mode;

        let mut modes = crate::Modes::empty();
        let mut set = |flag: crate::Modes, mode: Mode| -> Result<()> {
            if self.terminal.mode(mode)? {
                modes.insert(flag);
            }
            Ok(())
        };
        set(crate::Modes::APP_CURSOR, Mode::DECCKM)?;
        set(crate::Modes::APP_KEYPAD, Mode::KEYPAD_KEYS)?;
        set(crate::Modes::SHOW_CURSOR, Mode::CURSOR_VISIBLE)?;
        set(crate::Modes::LINE_WRAP, Mode::WRAPAROUND)?;
        set(crate::Modes::ORIGIN, Mode::ORIGIN)?;
        set(crate::Modes::INSERT, Mode::INSERT)?;
        set(crate::Modes::LINE_FEED_NEW_LINE, Mode::LINEFEED)?;
        set(crate::Modes::FOCUS_IN_OUT, Mode::FOCUS_EVENT)?;
        set(crate::Modes::ALTERNATE_SCROLL, Mode::ALT_SCROLL)?;
        set(crate::Modes::BRACKETED_PASTE, Mode::BRACKETED_PASTE)?;
        set(crate::Modes::SGR_MOUSE, Mode::SGR_MOUSE)?;
        set(crate::Modes::UTF8_MOUSE, Mode::UTF8_MOUSE)?;
        set(crate::Modes::MOUSE_REPORT_CLICK, Mode::NORMAL_MOUSE)?;
        set(crate::Modes::MOUSE_DRAG, Mode::BUTTON_MOUSE)?;
        set(crate::Modes::MOUSE_MOTION, Mode::ANY_MOUSE)?;
        // Alacritty's `TermMode::ALT_SCREEN` is set by any of the legacy
        // (47), and modern save/restore (1047/1049) alternate-screen
        // sequences; check all three since senders vary in which they use.
        if self.terminal.mode(Mode::ALT_SCREEN_LEGACY)?
            || self.terminal.mode(Mode::ALT_SCREEN)?
            || self.terminal.mode(Mode::ALT_SCREEN_SAVE)?
        {
            modes.insert(crate::Modes::ALT_SCREEN);
        }
        Ok(modes)
    }

    // Selection. `crate::Point`/`crate::SelectionRange` round-trip through
    // Ghostty's `Point::Viewport`/`PointSpace::Viewport`, using the same
    // `line + display_offset = viewport row` convention
    // `terminal_element.rs`'s `to_highlighted_range_lines` already uses to
    // map a `Point` onto the currently-rendered grid, so this code only
    // ever round-trips Ghostty's own addressing and never has to reconcile
    // it against a different engine's.

    /// Installs `range` as the active selection, or clears it if `None`.
    /// Also updates the cached `current_selection` mirror `selection_range`/
    /// `update_selection` read.
    pub(super) fn set_selection(&mut self, range: Option<crate::SelectionRange>) -> Result<()> {
        let display_offset = self.alacritty_style_display_offset()?;
        match range {
            Some(range) => {
                let start = self
                    .terminal
                    .grid_ref(ghostty_viewport_point(display_offset, range.start))?;
                let end = self
                    .terminal
                    .grid_ref(ghostty_viewport_point(display_offset, range.end))?;
                let selection = libghostty_vt::selection::Selection::new(start, end, range.is_block);
                self.terminal.set_selection(Some(&selection))?;
            }
            None => {
                self.terminal.set_selection(None)?;
            }
        }
        self.current_selection = range;
        Ok(())
    }

    /// Moves the active selection's end point to `point`, keeping its start
    /// (anchor) and `is_block` fixed. Returns `false` if there is no active
    /// selection to extend, mirroring `alacritty::update_selection`'s shape.
    pub(super) fn update_selection(&mut self, point: crate::Point) -> Result<bool> {
        let Some(mut range) = self.current_selection else {
            return Ok(false);
        };
        range.end = point;
        self.set_selection(Some(range))?;
        Ok(true)
    }

    /// The cached, Zed-point-space mirror of the active selection.
    pub(super) fn selection_range(&self) -> Option<crate::SelectionRange> {
        self.current_selection
    }

    /// Formats the active selection as plain text, joining soft-wrapped
    /// lines and trimming trailing whitespace, matching Ghostty's own
    /// `Screen.selectionString()` semantics per this option combination's
    /// doc comment in `libghostty_vt::selection`.
    pub(super) fn selection_text(&self) -> Result<Option<String>> {
        let options = libghostty_vt::selection::FormatOptions::new()
            .with_emit_format(libghostty_vt::fmt::Format::Plain)
            .with_unwrap(true)
            .with_trim(true);
        let Some(bytes) = self.terminal.format_selection_alloc(None, options)? else {
            return Ok(None);
        };
        Ok(Some(String::from_utf8_lossy(&bytes).into_owned()))
    }

    /// Derives and installs a word selection at `point`, using Ghostty's own
    /// word-boundary rules (`Terminal::select_word`).
    pub(super) fn select_word_at(&mut self, point: crate::Point) -> Result<Option<crate::SelectionRange>> {
        let display_offset = self.alacritty_style_display_offset()?;
        let grid_ref = self
            .terminal
            .grid_ref(ghostty_viewport_point(display_offset, point))?;
        let Some(selection) = self
            .terminal
            .select_word(libghostty_vt::selection::SelectWordOptions::new(grid_ref))?
        else {
            return Ok(None);
        };
        let range = selection_range_from_ghostty(&self.terminal, display_offset, &selection)?;
        self.terminal.set_selection(Some(&selection))?;
        self.current_selection = Some(range);
        Ok(Some(range))
    }

    /// Derives and installs a line selection at `point`, using Ghostty's own
    /// line semantics (`Terminal::select_line`; joins soft-wrapped rows).
    pub(super) fn select_line_at(&mut self, point: crate::Point) -> Result<Option<crate::SelectionRange>> {
        let display_offset = self.alacritty_style_display_offset()?;
        let grid_ref = self
            .terminal
            .grid_ref(ghostty_viewport_point(display_offset, point))?;
        let Some(selection) = self.terminal.select_line(
            libghostty_vt::selection::SelectLineOptions::new(grid_ref),
        )?
        else {
            return Ok(None);
        };
        let range = selection_range_from_ghostty(&self.terminal, display_offset, &selection)?;
        self.terminal.set_selection(Some(&selection))?;
        self.current_selection = Some(range);
        Ok(Some(range))
    }

    /// Derives and installs a selection covering all selectable content.
    pub(super) fn select_all(&mut self) -> Result<Option<crate::SelectionRange>> {
        let display_offset = self.alacritty_style_display_offset()?;
        let Some(selection) = self.terminal.select_all()? else {
            return Ok(None);
        };
        let range = selection_range_from_ghostty(&self.terminal, display_offset, &selection)?;
        self.terminal.set_selection(Some(&selection))?;
        self.current_selection = Some(range);
        Ok(Some(range))
    }

    /// Extends a word-granularity selection anchored at `anchor` to also
    /// include the word under `drag_point`, using `select_word_between` for
    /// boundary snapping in both directions. This is the exact technique
    /// `Terminal::select_word_between`'s own doc comment recommends for
    /// double-click-and-drag selection, so a double-click-drag doesn't
    /// flicker/collapse when the pointer passes over whitespace between
    /// words. Installs the resulting selection and updates the cache.
    /// Returns `None` if either endpoint has no nearby word (e.g. an
    /// entirely blank buffer).
    pub(super) fn select_word_range(
        &mut self,
        anchor: crate::Point,
        drag_point: crate::Point,
    ) -> Result<Option<crate::SelectionRange>> {
        let display_offset = self.alacritty_style_display_offset()?;
        let anchor_ref = self
            .terminal
            .grid_ref(ghostty_viewport_point(display_offset, anchor))?;
        let drag_ref = self
            .terminal
            .grid_ref(ghostty_viewport_point(display_offset, drag_point))?;

        let word_toward_drag = self.terminal.select_word_between(
            libghostty_vt::selection::SelectWordBetweenOptions::new(anchor_ref.clone(), drag_ref.clone()),
        )?;
        let word_toward_anchor = self.terminal.select_word_between(
            libghostty_vt::selection::SelectWordBetweenOptions::new(drag_ref, anchor_ref),
        )?;
        let (Some(word_toward_drag), Some(word_toward_anchor)) = (word_toward_drag, word_toward_anchor)
        else {
            return Ok(None);
        };
        // Don't assume drag direction: `select_word_between`'s own doctest
        // uses `start_word.start()`/`end_word.end()` directly, which only
        // gives the right bounds when the anchor is left of the drag
        // point. Converting both derived words to `Point` ranges first and
        // taking the overall min/max instead (same technique
        // `select_line_range` below uses) handles either drag direction.
        let word_toward_drag_range =
            selection_range_from_ghostty(&self.terminal, display_offset, &word_toward_drag)?;
        let word_toward_anchor_range =
            selection_range_from_ghostty(&self.terminal, display_offset, &word_toward_anchor)?;
        let start = word_toward_drag_range
            .start
            .min(word_toward_drag_range.end)
            .min(word_toward_anchor_range.start)
            .min(word_toward_anchor_range.end);
        let end = word_toward_drag_range
            .start
            .max(word_toward_drag_range.end)
            .max(word_toward_anchor_range.start)
            .max(word_toward_anchor_range.end);
        self.set_selection(Some(crate::SelectionRange {
            start,
            end,
            is_block: false,
        }))?;
        Ok(self.current_selection)
    }

    /// Extends a line-granularity selection anchored at `anchor` to also
    /// include the whole line(s) containing `drag_point`, by unioning the
    /// full-line ranges `select_line_at` derives for each endpoint. The
    /// selection always covers every complete line between the two points
    /// regardless of drag direction. Installs the resulting selection and
    /// updates the cache. Returns `None` if either endpoint has no
    /// selectable line (shouldn't normally happen; `select_line` only
    /// fails to find a line on a fully empty screen).
    pub(super) fn select_line_range(
        &mut self,
        anchor: crate::Point,
        drag_point: crate::Point,
    ) -> Result<Option<crate::SelectionRange>> {
        let display_offset = self.alacritty_style_display_offset()?;
        let anchor_ref = self
            .terminal
            .grid_ref(ghostty_viewport_point(display_offset, anchor))?;
        let drag_ref = self
            .terminal
            .grid_ref(ghostty_viewport_point(display_offset, drag_point))?;

        let anchor_line = self
            .terminal
            .select_line(libghostty_vt::selection::SelectLineOptions::new(anchor_ref))?;
        let drag_line = self
            .terminal
            .select_line(libghostty_vt::selection::SelectLineOptions::new(drag_ref))?;
        let (Some(anchor_line), Some(drag_line)) = (anchor_line, drag_line) else {
            return Ok(None);
        };
        let anchor_range = selection_range_from_ghostty(&self.terminal, display_offset, &anchor_line)?;
        let drag_range = selection_range_from_ghostty(&self.terminal, display_offset, &drag_line)?;
        let start = anchor_range
            .start
            .min(anchor_range.end)
            .min(drag_range.start)
            .min(drag_range.end);
        let end = anchor_range
            .start
            .max(anchor_range.end)
            .max(drag_range.start)
            .max(drag_range.end);
        self.set_selection(Some(crate::SelectionRange {
            start,
            end,
            is_block: false,
        }))?;
        Ok(self.current_selection)
    }

    // Hyperlinks. Given a point, this walks outward and returns an answer
    // immediately. Unlike selection, there's no persistent cross-call state
    // to keep in sync with anything else.

    /// Finds the hyperlink or path at `point`: first an OSC 8 native
    /// hyperlink (walking outward from `point` while the hyperlink URI
    /// stays the same), then a bare URL matched by `url_regex`, then
    /// the first `path_hyperlink_regexes` entry that produces a match on
    /// the line (Alacritty's own "processing stops at the first regex with
    /// a match, even if no link is produced" rule, preserved exactly since
    /// both backends share `path_hyperlink_candidates_in_line`). Returns
    /// `(text, is_url, range)`, or `None` if nothing matches.
    pub(super) fn hyperlink_at(
        &mut self,
        point: crate::Point,
        url_regex: &regex::Regex,
        path_hyperlink_regexes: &[regex::Regex],
        path_hyperlink_timeout: std::time::Duration,
    ) -> Result<Option<(String, bool, crate::Range)>> {
        let display_offset = self.alacritty_style_display_offset()?;
        if let Some(uri) = hyperlink_uri_at_point(&self.terminal, display_offset, point)? {
            let start = walk_hyperlink_run(&self.terminal, display_offset, point, &uri, -1)?;
            let end = walk_hyperlink_run(&self.terminal, display_offset, point, &uri, 1)?;
            return Ok(Some((uri, true, crate::Range::new(start, end))));
        }

        let grid = ViGrid::new(&self.terminal)?;
        let Some((line, points, hovered_index)) = single_line_text_with_point_map(&grid, point)?
        else {
            return Ok(None);
        };
        let hovered_byte_offset = line
            .char_indices()
            .nth(hovered_index)
            .map(|(byte_offset, _)| byte_offset)
            .unwrap_or(line.len());

        // Converts a byte range in `line` back into an inclusive `Range` via
        // `points` (char-indexed), the same byte->char conversion
        // `search_matches` uses. `None` only if the range is somehow out of
        // bounds for `points`, which byte ranges derived from `line`
        // itself should never produce.
        let point_range_for_bytes = |byte_range: std::ops::Range<usize>| -> Option<crate::Range> {
            let start_char = line[..byte_range.start].chars().count();
            let end_char = line[..byte_range.end].chars().count();
            if start_char >= points.len() {
                return None;
            }
            let end_char = end_char.saturating_sub(1).max(start_char).min(points.len() - 1);
            Some(crate::Range::new(points[start_char], points[end_char]))
        };

        if let Some(found) = url_regex
            .find_iter(&line)
            .find(|m| m.start() <= hovered_byte_offset && hovered_byte_offset < m.end())
        {
            // `trim_url_punctuation` only ever removes trailing ASCII
            // punctuation (`.` `,` `:` `;` `(` `)`, each one byte), so the
            // trimmed string's byte length maps directly onto a new end
            // byte offset relative to the match's start.
            let (trimmed, _chars_trimmed) = trim_url_punctuation(found.as_str());
            let end_byte = found.start() + trimmed.len();
            if let Some(range) = point_range_for_bytes(found.start()..end_byte) {
                return Ok(Some((trimmed, true, range)));
            }
        }

        for (path, link_byte_range) in path_hyperlink_candidates_in_line(
            &line,
            hovered_byte_offset,
            path_hyperlink_regexes,
            path_hyperlink_timeout,
        ) {
            if let Some(range) = point_range_for_bytes(link_byte_range) {
                return Ok(Some((path, false, range)));
            }
        }

        Ok(None)
    }

    // Search. `libghostty-vt` has no built-in grid-wide search API at all
    // (real Ghostty's own search, `src/terminal/search/*.zig` in the
    // vendored source, is a literal-substring incremental sliding-window
    // search over its `PageList`, not regex, and not exposed through the
    // Rust bindings regardless). Zed's search is regex (`crate::Search`
    // compiles a pattern), so this extracts the full history and active
    // buffer as plain text with a per-character `Point` map, and runs the
    // `regex` crate over that text directly, mapping byte-range matches
    // back to `Point`s via the map.
    //
    // This reads every cell in the entire scrollback via FFI on every
    // call, with no incremental or windowed search, so it can be slow on
    // very large scrollback (tens of thousands of lines). An incremental
    // search would need Ghostty's own sliding-window technique adapted for
    // regex.

    /// Runs `regex` over the terminal's full history + active screen text,
    /// returning matches as `crate::Range`s in the same `line +
    /// display_offset = viewport row` coordinate convention
    /// `terminal_element.rs`'s `to_highlighted_range_lines` and this
    /// module's selection/hyperlink code already use. A match found deep in
    /// scrollback can later be scrolled into view the same way any other
    /// off-screen `Point` can.
    ///
    /// Soft-wrapped rows are joined into one logical line for matching
    /// (mirroring `alacritty::search_matches`'s behavior, since Alacritty's
    /// own grid addressing has no separate concept of "physical row" vs.
    /// "wrapped line" either); hard line breaks insert a literal `\n` so
    /// `^`/`$`-anchored patterns still match per hard line, not per
    /// soft-wrapped screen row.
    pub(super) fn search_matches(&mut self, regex: &regex::Regex) -> Result<Vec<crate::Range>> {
        let (text, points) = self.buffer_text_with_point_map()?;

        let mut matches = Vec::new();
        for found in regex.find_iter(&text) {
            if found.start() == found.end() {
                continue;
            }
            let start_char = text[..found.start()].chars().count();
            let end_char = text[..found.end()].chars().count();
            let Some(&start_point) = points.get(start_char) else {
                continue;
            };
            let Some(&end_point) = points.get(end_char - 1) else {
                continue;
            };
            matches.push(crate::Range::new(start_point, end_point));
        }
        Ok(matches)
    }

    /// Extracts the terminal's full history and active screen as plain
    /// text, alongside a parallel `Point` for every `char` in that text
    /// (indexed by char count, not byte offset, so multi-byte UTF-8
    /// doesn't need to be reasoned about when mapping a match back to a
    /// point). This is the shared core of `search_matches` and, in the
    /// future, "select all"/whole-buffer text extraction that needs full
    /// scrollback rather than just the current viewport (`build_content`
    /// only covers the viewport).
    fn buffer_text_with_point_map(&self) -> Result<(String, Vec<crate::Point>)> {
        let total_rows = self.terminal.total_rows()?;
        let viewport_rows = usize::from(self.terminal.rows()?);
        let cols = usize::from(self.terminal.cols()?);
        let line_base = total_rows as i32 - viewport_rows as i32;

        let mut text = String::new();
        let mut points: Vec<crate::Point> = Vec::new();
        let mut grapheme_buf = [' '; 32];

        for absolute_row in 0..total_rows {
            let line = absolute_row as i32 - line_base;
            let mut row_wrapped = false;
            let row_start = points.len();

            for column in 0..cols {
                let grid_ref = self.terminal.grid_ref(libghostty_vt::terminal::Point::Screen(
                    libghostty_vt::terminal::PointCoordinate {
                        x: column as u16,
                        y: absolute_row as u32,
                    },
                ))?;
                if column == 0 {
                    row_wrapped = grid_ref.row()?.is_wrapped()?;
                }
                if grid_ref.cell()?.wide()? == screen::CellWide::SpacerTail {
                    continue;
                }
                let grapheme_count = grid_ref.graphemes(&mut grapheme_buf)?;
                if grapheme_count == 0 {
                    text.push(' ');
                    points.push(crate::Point::new(line, column));
                } else {
                    for &character in &grapheme_buf[..grapheme_count.min(grapheme_buf.len())] {
                        text.push(character);
                        points.push(crate::Point::new(line, column));
                    }
                }
            }
            if !row_wrapped {
                // Trim this row's own trailing blank cells before the hard
                // newline, so `$`-anchored patterns match right after real
                // content instead of after a whole row's worth of trailing
                // padding. Only the current physical row's contribution is
                // trimmed, since a row can only wrap when filled
                // edge-to-edge, so a wrapped predecessor row in the same
                // logical line should never have trailing blanks to begin
                // with.
                while points.len() > row_start && text.ends_with(' ') {
                    text.pop();
                    points.pop();
                }
                text.push('\n');
                points.push(crate::Point::new(line, cols));
            }
        }
        Ok((text, points))
    }

    /// The terminal's full history and active screen as plain text, for
    /// `Terminal::get_content`. Unlike `buffer_text_with_point_map`, which
    /// appends a trailing `\n` after every logical line including the last
    /// (so char-index-to-`Point` mapping stays simple for
    /// `search_matches`/`hyperlink_at`), this trims a spurious final
    /// trailing newline.
    pub(super) fn buffer_text(&self) -> Result<String> {
        let (mut text, _) = self.buffer_text_with_point_map()?;
        if text.ends_with('\n') {
            text.pop();
        }
        Ok(text)
    }

    /// The last (up to) `line_count` non-blank logical lines of the
    /// terminal's full history and active screen, in top-to-bottom order.
    /// Used for init-command shell-startup marker detection
    /// (`Terminal::detect_init_command_startup_marker`) and
    /// `Terminal::last_n_non_empty_lines`. `buffer_text_with_point_map`
    /// already trims each logical line's own trailing blanks and joins on
    /// `\n`, so this only needs to split, drop the blank lines, and take
    /// the last `line_count`.
    pub(super) fn last_non_empty_lines(&self, line_count: usize) -> Result<Vec<String>> {
        let (text, _) = self.buffer_text_with_point_map()?;
        let mut lines: Vec<String> = text
            .split('\n')
            .filter(|line| !line.is_empty())
            .rev()
            .take(line_count)
            .map(str::to_string)
            .collect();
        lines.reverse();
        Ok(lines)
    }

    // Vi mode. `libghostty-vt` has no vi-mode/vi-cursor concept at all, so
    // this hand-implements cursor motion against Ghostty's cell/row grid,
    // using the same absolute (`Point::Screen`) row addressing
    // `search_matches`/`hyperlink_at` already use, via the small `ViGrid`
    // helper below.
    //
    // Known simplification, not full parity: normalizing a point that
    // lands on a wide-character spacer cell back onto the actual wide
    // character is only partially handled. Same-row spacer-tail correction
    // is implemented; the cross-row case for a wide character split across
    // a soft wrap is not. This affects vi-motion cursor placement
    // immediately around wide (e.g. CJK) characters only; plain ASCII/Latin
    // text (the common case) is unaffected.
    //
    // Zed's own `ViMotion` (this crate's `enum ViMotion` in `terminal.rs`)
    // only exposes whitespace-word motions (`WordLeft`/`WordRight`/
    // `WordRightEnd`, vi's `W`/`B`/`E`), not "semantic" (punctuation-aware)
    // word motions, so `ghostty_word_motion` below only needs to implement
    // that one case.
    //
    // `Bracket` has no `libghostty-vt` equivalent either;
    // `ghostty_bracket_motion` below is a small hand-rolled depth-counting
    // scan for `()`/`[]`/`{}` pairs.

    /// Toggles vi mode. Returns the new vi cursor position (`Some` when vi
    /// mode is now enabled, `None` when it was just disabled). On enabling,
    /// the vi cursor starts at the real terminal cursor's position if it's
    /// currently within the
    /// visible viewport, or at the top-left of the viewport otherwise (the
    /// terminal cursor scrolled out of view).
    pub(super) fn toggle_vi_mode(&mut self) -> Result<Option<crate::Point>> {
        if self.current_vi_cursor.is_some() {
            self.current_vi_cursor = None;
            return Ok(None);
        }
        let display_offset = self.alacritty_style_display_offset()? as i32;
        let grid = ViGrid::new(&self.terminal)?;
        let cursor_viewport = self
            .render_state
            .update(&self.terminal)?
            .cursor_viewport()?;
        let point = match cursor_viewport {
            // `cursor_viewport` is already viewport-relative (0,0 = the
            // viewport's top-left), so this is the same `line = row -
            // display_offset` conversion `content_metadata` uses for the
            // real rendered cursor.
            Some(cursor) => crate::Point::new(cursor.y as i32 - display_offset, cursor.x as usize),
            // Terminal cursor isn't in the visible viewport at all right
            // now (e.g. mid-command output scrolled it away): start at the
            // top-left of the viewport instead, matching Alacritty.
            None => grid.to_point(grid.line_base - i64::from(display_offset), 0),
        };
        self.current_vi_cursor = Some(point);
        Ok(Some(point))
    }

    /// The current vi cursor position, or `None` when vi mode is disabled.
    pub(super) fn vi_cursor(&self) -> Option<crate::Point> {
        self.current_vi_cursor
    }

    /// Moves the vi cursor directly to `point` (e.g. for search-match
    /// activation while in vi mode), and extends the active selection to
    /// follow if one is present. No-ops if vi mode isn't enabled.
    pub(super) fn vi_goto_point(&mut self, point: crate::Point) -> Result<()> {
        if self.current_vi_cursor.is_none() {
            return Ok(());
        }
        self.scroll_viewport_to_reveal(point)?;
        self.current_vi_cursor = Some(point);
        self.update_selection(point)?;
        Ok(())
    }

    /// Scrolls the viewport by the minimum amount needed to bring `point`
    /// into view (as the top row if it's above the viewport, or the bottom
    /// row if below; a no-op if already visible). `vi_goto_point` calls this
    /// before moving the vi cursor, and
    /// `Terminal::process_terminal_event`'s `InternalEvent::ScrollToPoint`
    /// handler calls it directly.
    pub(super) fn scroll_viewport_to_reveal(&mut self, point: crate::Point) -> Result<()> {
        let display_offset = self.alacritty_style_display_offset()? as i32;
        let viewport_rows = i32::from(self.terminal.rows()?);
        let viewport_row = point.line + display_offset;

        let delta = if viewport_row < 0 {
            -viewport_row
        } else if viewport_row >= viewport_rows {
            -(viewport_row - viewport_rows + 1)
        } else {
            0
        };
        if delta != 0 {
            self.terminal.scroll_viewport(ghostty_scroll(
                Scroll::Delta(delta),
                viewport_rows as usize,
            ));
        }
        Ok(())
    }

    /// Moves the vi cursor to follow a plain (non vi-motion) scroll, such
    /// as the mouse wheel or `Scroll::PageUp`/`PageDown`/`Top`/`Bottom`, so
    /// it roughly stays in the same relative viewport position instead of
    /// scrolling out from under the user. No-ops if vi mode isn't enabled.
    pub(super) fn update_vi_cursor_for_scroll(&mut self, scroll: Scroll) -> Result<Option<crate::Point>> {
        let Some(current) = self.current_vi_cursor else {
            return Ok(None);
        };
        let grid = ViGrid::new(&self.terminal)?;
        let viewport_rows = i64::from(self.terminal.rows()?);
        let point = match scroll {
            Scroll::Delta(delta) => self.vi_cursor_scroll(&grid, current, i64::from(delta))?,
            Scroll::PageUp => self.vi_cursor_scroll(&grid, current, viewport_rows)?,
            Scroll::PageDown => self.vi_cursor_scroll(&grid, current, -viewport_rows)?,
            Scroll::Top => {
                let column = grid.first_occupied_in_line(grid.topmost_row())?.unwrap_or(0);
                grid.to_point(grid.topmost_row(), column)
            }
            Scroll::Bottom => {
                let column = grid.first_occupied_in_line(grid.bottommost_row())?.unwrap_or(0);
                grid.to_point(grid.bottommost_row(), column)
            }
        };
        self.current_vi_cursor = Some(point);
        Ok(Some(point))
    }

    /// Shared by `update_vi_cursor_for_scroll`'s `Delta`/`PageUp`/`PageDown`
    /// cases: move `lines` rows from `current`, clamped to the grid's
    /// absolute bounds (Alacritty's `Boundary::Grid`), landing on the first
    /// occupied cell in the resulting row (or column 0 if the row is blank).
    fn vi_cursor_scroll(
        &self,
        grid: &ViGrid<'_>,
        current: crate::Point,
        lines: i64,
    ) -> Result<crate::Point> {
        let (row, _) = grid.to_absolute(current);
        let clamped_row = (row - lines).clamp(grid.topmost_row(), grid.bottommost_row());
        let column = grid.first_occupied_in_line(clamped_row)?.unwrap_or(0);
        Ok(grid.to_point(clamped_row, column))
    }

    /// Moves the vi cursor by `motion`, and extends the active selection to
    /// follow if one is present. No-ops if vi mode isn't enabled.
    pub(super) fn vi_motion(&mut self, motion: ViMotion) -> Result<Option<crate::Point>> {
        let Some(current) = self.current_vi_cursor else {
            return Ok(None);
        };
        let grid = ViGrid::new(&self.terminal)?;
        let display_offset = self.alacritty_style_display_offset()? as i64;
        let point = ghostty_vi_motion(&grid, current, motion, display_offset)?;
        // Alacritty's own `ViModeCursor::motion` scrolls to reveal the new
        // cursor position at the end of every motion (e.g. `G`/`gg`-style
        // High/Low jumps, or repeated Up/Down walking off the current
        // screen); mirrored here for the same reason.
        self.scroll_viewport_to_reveal(point)?;
        self.current_vi_cursor = Some(point);
        self.update_selection(point)?;
        Ok(Some(point))
    }

    /// Configures Ghostty's *default* foreground/background/cursor colors
    /// and 256-color palette from Zed's active theme
    /// (`Terminal::sync_ghostty_theme_colors` computes these via the same
    /// `get_color_at_index` mapping the OSC 10/11 theme-fallback and
    /// `Color::Indexed` rendering already used, see `terminal.rs`). An OSC
    /// 4/10/11/12 SET from the running program still overrides these on
    /// top, same as before.
    ///
    /// Once every terminal calls this, Ghostty's "effective color" getters
    /// (`fg_color`/`bg_color`/`cursor_color`/`color_palette`) always
    /// resolve to a real value instead of `None`/a built-in placeholder,
    /// so Ghostty always answers OSC 4/10/11/12 queries independently on
    /// its own PTY-write path, and `Terminal::process_event`'s
    /// `ColorRequest` handler can trust that unconditionally instead of
    /// needing to distinguish "overridden" from "not" itself.
    pub(super) fn set_default_theme_colors(
        &mut self,
        foreground: vte::ansi::Rgb,
        background: vte::ansi::Rgb,
        cursor: vte::ansi::Rgb,
        palette: [vte::ansi::Rgb; 256],
    ) -> Result<()> {
        fn to_rgb_color(color: vte::ansi::Rgb) -> style::RgbColor {
            style::RgbColor {
                r: color.r,
                g: color.g,
                b: color.b,
            }
        }

        let mut ghostty_palette = style::Palette::default();
        for (index, color) in palette.into_iter().enumerate() {
            ghostty_palette.set(style::PaletteIndex(index as u8), to_rgb_color(color));
        }

        self.terminal
            .set_default_fg_color(Some(to_rgb_color(foreground)))?
            .set_default_bg_color(Some(to_rgb_color(background)))?
            .set_default_cursor_color(Some(to_rgb_color(cursor)))?
            .set_default_color_palette(Some(ghostty_palette))?;
        Ok(())
    }

    /// Configures Ghostty's default cursor shape from Zed's
    /// `terminal.cursor_shape` setting, mirroring
    /// `alacritty::set_default_cursor_style`. A program that sets its own
    /// shape via DECSCUSR still overrides this, same as the color
    /// defaults above. `CursorShape::Hidden` has no `CursorStyle`
    /// equivalent and is unreachable from a settings-sourced shape (see
    /// `From<SettingsCursorShape> for CursorShape`); falls back to
    /// `Block` defensively rather than needing a fallible signature for
    /// an input this method's only caller never produces.
    pub(super) fn set_default_cursor_shape(&mut self, shape: CursorShape) -> Result<()> {
        let style = match shape {
            CursorShape::Bar => CursorStyle::Bar,
            CursorShape::Block | CursorShape::Hidden => CursorStyle::Block,
            CursorShape::Underline => CursorStyle::Underline,
            CursorShape::HollowBlock => CursorStyle::BlockHollow,
        };
        self.terminal.set_default_cursor_style(Some(style))?;
        Ok(())
    }

    /// Disables the "alternate scroll" mode (scroll wheel sends arrow keys
    /// to full-screen apps in the alternate screen, e.g. `less`/`vim`) at
    /// construction time, mirroring `alacritty::new_term`'s
    /// `unset_private_mode(NamedPrivateMode::AlternateScroll)` for
    /// `terminal.alternate_scroll: Off`. Ghostty already tracks this mode
    /// nakedly (`build_content` reports it as `Modes::ALTERNATE_SCROLL` via
    /// `Mode::ALT_SCROLL`) and defaults it on, so there's nothing to do for
    /// the `On` case. Only `Off` needs an explicit call.
    pub(super) fn disable_alternate_scroll(&mut self) -> Result<()> {
        use libghostty_vt::terminal::Mode;
        self.terminal.set_mode(Mode::ALT_SCROLL, false)?;
        Ok(())
    }

    /// Returns the Kitty graphics placements currently visible in the
    /// viewport, with their pixel data already decoded to RGBA8.
    pub(super) fn image_placements(&mut self) -> Result<Vec<crate::ImagePlacement>> {
        let graphics = self.terminal.kitty_graphics()?;
        let mut iteration = self.placements.update(&graphics)?;
        let mut placements = Vec::new();

        while let Some(placement) = iteration.next() {
            let image_id = match placement.image_id() {
                Ok(image_id) => image_id,
                Err(error) => {
                    log::debug!("failed to read kitty placement image id: {error}");
                    continue;
                }
            };
            let Some(image) = graphics.image(image_id) else {
                continue;
            };

            let info = match placement.placement_render_info(&image, &self.terminal) {
                Ok(info) => info,
                Err(error) => {
                    log::debug!("failed to compute kitty placement render info: {error}");
                    continue;
                }
            };
            if !info.viewport_visible {
                continue;
            }

            let generation = match image.generation() {
                Ok(generation) => generation,
                Err(error) => {
                    log::debug!("failed to read kitty image generation: {error}");
                    continue;
                }
            };

            let data = match decode_image_rgba(&image) {
                Ok(Some(data)) => data,
                Ok(None) => continue,
                Err(error) => {
                    log::debug!("failed to decode kitty image {image_id}: {error}");
                    continue;
                }
            };

            placements.push(crate::ImagePlacement {
                image_id,
                generation,
                viewport_column: info.viewport_col,
                viewport_row: info.viewport_row,
                grid_columns: info.grid_cols,
                grid_rows: info.grid_rows,
                pixel_width: info.pixel_width,
                pixel_height: info.pixel_height,
                data,
            });
        }

        Ok(placements)
    }

}

/// Decodes a Kitty graphics image's stored pixel data to RGBA8, inflating it
/// first if the sender compressed it. Returns `Ok(None)` for formats/sizes
/// that can't be decoded (logged by the caller as a decode failure).
fn decode_image_rgba(image: &graphics::Image<'_>) -> Result<Option<Arc<[u8]>>> {
    let width = image.width()? as usize;
    let height = image.height()? as usize;
    let format = image.format()?;
    let raw = image.data()?;

    let inflated;
    let raw = match image.compression()? {
        Compression::ZlibDeflate => {
            let mut decoder = flate2::read::ZlibDecoder::new(raw);
            let mut buffer = Vec::new();
            decoder.read_to_end(&mut buffer)?;
            inflated = buffer;
            &inflated
        }
        // `Compression::None` and any future variants the library adds:
        // nothing to inflate.
        _ => raw,
    };

    let rgba = match format {
        ImageFormat::Rgba => raw.to_vec(),
        ImageFormat::Rgb => {
            anyhow::ensure!(raw.len() >= width * height * 3, "truncated RGB image data");
            let mut rgba = Vec::with_capacity(width * height * 4);
            for chunk in raw.chunks_exact(3) {
                rgba.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
            }
            rgba
        }
        ImageFormat::GrayAlpha => {
            anyhow::ensure!(
                raw.len() >= width * height * 2,
                "truncated gray+alpha image data"
            );
            let mut rgba = Vec::with_capacity(width * height * 4);
            for chunk in raw.chunks_exact(2) {
                rgba.extend_from_slice(&[chunk[0], chunk[0], chunk[0], chunk[1]]);
            }
            rgba
        }
        ImageFormat::Gray => {
            anyhow::ensure!(raw.len() >= width * height, "truncated gray image data");
            let mut rgba = Vec::with_capacity(width * height * 4);
            for &gray in &raw[..width * height] {
                rgba.extend_from_slice(&[gray, gray, gray, 255]);
            }
            rgba
        }
        // PNG payloads are decoded to RGBA8 by `RustPngDecoder` before
        // libghostty-vt stores them, so a stored image should never report
        // this format; treated as undecodable defensively.
        ImageFormat::Png => return Ok(None),
        _ => return Ok(None),
    };

    Ok(Some(Arc::from(rgba)))
}

/// Installs a PNG decoder so the Kitty graphics protocol can accept
/// PNG-transmitted images, once per process. `set_png_decoder` stores the
/// callback in thread-local state and documents that it "must only be
/// called on the same thread as the terminal", which is safe here since
/// every `GhosttyTerminal` (and thus every call to this function) lives on
/// Zed's single foreground thread.
///
/// `libghostty-vt` ships a `RustPngDecoder` for exactly this purpose, but
/// its only field is private with no constructor, so it can't actually be
/// instantiated from outside the crate as of 0.2.1 (a bug in that release,
/// visible in its own doctest, which doesn't compile as written either).
/// `ZedPngDecoder` below reimplements the same logic directly.
fn install_png_decoder() -> Result<()> {
    thread_local! {
        static INSTALLED: Cell<bool> = const { Cell::new(false) };
    }
    INSTALLED.with(|installed| {
        if installed.get() {
            return Ok(());
        }
        graphics::set_png_decoder(Some(Box::new(ZedPngDecoder::default())))?;
        installed.set(true);
        Ok(())
    })
}

#[derive(Default)]
struct ZedPngDecoder {
    buffer: Vec<u8>,
}

impl graphics::DecodePng for ZedPngDecoder {
    fn decode_png<'alloc>(
        &mut self,
        alloc: &'alloc libghostty_vt::alloc::Allocator<'_>,
        data: &[u8],
    ) -> Option<graphics::DecodedImage<'alloc>> {
        use png::Transformations;

        let mut decoder = png::Decoder::new(std::io::Cursor::new(data));
        // libghostty only accepts RGBA8 data, so expand palette/grayscale
        // colors to RGBA8 and strip 16-bit depth back down to 8-bit.
        decoder.set_transformations(Transformations::ALPHA | Transformations::STRIP_16);

        let mut reader = decoder.read_info().ok()?;
        let buffer_size = reader.output_buffer_size()?;
        self.buffer.resize(buffer_size, 0);

        let info = reader.next_frame(&mut self.buffer).ok()?;

        let mut bytes =
            libghostty_vt::alloc::Bytes::new_with_alloc(alloc, info.buffer_size()).ok()?;
        bytes.copy_from_slice(&self.buffer[..info.buffer_size()]);

        Some(graphics::DecodedImage {
            width: info.width,
            height: info.height,
            data: bytes,
        })
    }
}

pub(super) fn spawn_pty(
    command: CommandBuilder,
    bounds: TerminalBounds,
    events_tx: UnboundedSender<PtyEvent>,
    terminal: Arc<ParkingMutex<GhosttyTerminal>>,
) -> Result<(PtySender, Arc<PtyProcessInfo>)> {
    let pty_system = portable_pty::native_pty_system();
    let pair = pty_system.openpty(pty_size(bounds))?;
    let mut reader = pair.master.try_clone_reader()?;
    let mut writer = pair.master.take_writer()?;
    let master = pair.master;

    let (command_tx, command_rx) = std::sync::mpsc::channel::<PtyCommand>();
    let (child_tx, child_rx) =
        std::sync::mpsc::channel::<Box<dyn portable_pty::Child + Send + Sync>>();
    let (output_tx, output_rx) =
        std::sync::mpsc::sync_channel::<Vec<u8>>(MAX_QUEUED_PTY_OUTPUT_BUFFERS);
    let (recycle_tx, recycle_rx) =
        std::sync::mpsc::sync_channel::<Vec<u8>>(MAX_RECYCLED_PTY_BUFFERS);

    // Parses queued PTY output off the reader thread, batching consecutive
    // chunks so libghostty-vt isn't re-entered on every single `read()`
    // during high-throughput output.
    let parser_thread = thread::Builder::new()
        .name("Ghostty PTY parser".to_string())
        .spawn(move || {
            // `set_png_decoder`'s callback is stored in the library's own
            // thread-local state, so it must be (re-)installed on this
            // specific thread: PTY output, and therefore any PNG-format
            // Kitty graphics data, is always written to the terminal here,
            // not on whichever thread originally constructed it.
            if let Err(error) = install_png_decoder() {
                log::error!("failed to install PNG decoder on ghostty parser thread: {error}");
            }

            let mut pending_chunk = None;
            let mut consecutive_batches = 0;
            while let Some(mut batch) = pending_chunk.take().or_else(|| output_rx.recv().ok()) {
                while batch.len() < MAX_PTY_PARSE_BATCH_BYTES {
                    match output_rx.try_recv() {
                        Ok(chunk) => {
                            if batch.len().saturating_add(chunk.len()) > MAX_PTY_PARSE_BATCH_BYTES
                            {
                                pending_chunk = Some(chunk);
                                break;
                            }
                            batch.extend_from_slice(&chunk);
                            recycle_pty_buffer(chunk, &recycle_tx);
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => break,
                    }
                }

                consecutive_batches += 1;
                let yield_fair = consecutive_batches >= 4;
                if yield_fair {
                    consecutive_batches = 0;
                }

                let keep_running =
                    write_pty_output_to_ghostty(&batch, &terminal, &events_tx, yield_fair);
                recycle_pty_buffer(batch, &recycle_tx);
                if !keep_running {
                    break;
                }
            }

            if let Some(chunk) = pending_chunk {
                recycle_pty_buffer(chunk, &recycle_tx);
            }
            drop(output_rx);

            match child_rx.recv() {
                Ok(mut child) => {
                    let status = child.wait().ok().map(portable_pty_exit_status);
                    let event = match status {
                        Some(status) => PtyEvent::Event(crate::TerminalBackendEvent::ChildExit(
                            status,
                        )),
                        None => PtyEvent::Event(crate::TerminalBackendEvent::Exit),
                    };
                    if events_tx.unbounded_send(event).is_err() {
                        log::debug!("terminal dropped before ghostty PTY exit could be delivered");
                    }
                }
                Err(error) => {
                    log::debug!("ghostty PTY reader stopped before child was spawned: {error}");
                }
            }
        })
        .context("failed to spawn ghostty PTY parser")?;
    drop(parser_thread);

    let reader_thread = thread::Builder::new()
        .name("Ghostty PTY reader".to_string())
        .spawn(move || {
            loop {
                let mut buffer = next_pty_read_buffer(&recycle_rx);
                match reader.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(bytes_read) => {
                        buffer.truncate(bytes_read);
                        if output_tx.send(buffer).is_err() {
                            break;
                        }
                    }
                    Err(error) if error.kind() == ErrorKind::Interrupted => {}
                    Err(error) => {
                        log::error!("error reading from ghostty PTY: {error}");
                        break;
                    }
                }
            }
        })
        .context("failed to spawn ghostty PTY reader")?;
    drop(reader_thread);

    let child = pair.slave.spawn_command(command)?;
    let process_id = child.process_id().unwrap_or_default();
    #[cfg(windows)]
    let handle = child.as_raw_handle().map(|handle| handle as i32).unwrap_or(-1);
    #[cfg(not(windows))]
    let handle = master.as_raw_fd().unwrap_or(-1);
    let info = Arc::new(PtyProcessInfo::new(ProcessIdGetter::new(handle, process_id)));
    if child_tx.send(child).is_err() {
        anyhow::bail!("ghostty PTY reader stopped before child handle was delivered");
    }
    drop(pair.slave);

    let writer_thread = thread::Builder::new()
        .name("Ghostty PTY writer".to_string())
        .spawn(move || {
            while let Ok(command) = command_rx.recv() {
                match command {
                    PtyCommand::Input(input) => {
                        if input.is_empty() {
                            continue;
                        }

                        if let Err(error) = writer.write_all(&input) {
                            log::error!("error writing to ghostty PTY: {error}");
                            break;
                        }
                    }
                    PtyCommand::Resize(bounds) => {
                        if let Err(error) = master.resize(pty_size(bounds)) {
                            log::error!("error resizing ghostty PTY: {error}");
                        }
                    }
                    PtyCommand::Shutdown => break,
                }
            }
        })
        .context("failed to spawn ghostty PTY writer")?;
    drop(writer_thread);

    Ok((PtySender { command_tx }, info))
}

/// Builds the command used when no explicit shell was requested, mirroring
/// the login-shell semantics of the terminal Zed would otherwise open via
/// `alacritty_terminal`'s system-shell handling.
#[cfg(target_os = "macos")]
pub(super) fn system_command() -> portable_pty::CommandBuilder {
    let Some(user) = std::env::var("USER").ok().filter(|user| !user.is_empty()) else {
        return portable_pty::CommandBuilder::new_default_prog();
    };
    let shell = std::env::var("SHELL").unwrap_or_else(|_| util::shell::get_system_shell());
    let shell_name = shell.rsplit('/').next().unwrap_or(&shell);
    let home = std::env::var("HOME").unwrap_or_default();
    let flags = if std::path::Path::new(&home).join(".hushlogin").exists() {
        "-qflp"
    } else {
        "-flp"
    };
    let exec = format!("exec -a -{shell_name} {shell}");

    let mut command = portable_pty::CommandBuilder::new("/usr/bin/login");
    command.args([flags, &user, "/bin/zsh", "-fc", &exec]);
    command
}

#[cfg(not(target_os = "macos"))]
pub(super) fn system_command() -> portable_pty::CommandBuilder {
    portable_pty::CommandBuilder::new_default_prog()
}

/// Converts the crate's backend-agnostic scroll request into libghostty-vt's
/// viewport scroll command.
pub(super) fn ghostty_scroll(scroll: Scroll, screen_lines: usize) -> ScrollViewport {
    match scroll {
        Scroll::Delta(delta) => ScrollViewport::Delta(-(delta as isize)),
        Scroll::PageUp => ScrollViewport::Delta(-(screen_lines as isize)),
        Scroll::PageDown => ScrollViewport::Delta(screen_lines as isize),
        Scroll::Top => ScrollViewport::Top,
        Scroll::Bottom => ScrollViewport::Bottom,
    }
}

/// Takes a recycled buffer if one is available, otherwise allocates a fresh
/// one, and (re)sizes it to the fixed PTY read chunk size.
fn next_pty_read_buffer(recycle_rx: &Receiver<Vec<u8>>) -> Vec<u8> {
    let mut buffer = recycle_rx
        .try_recv()
        .unwrap_or_else(|_| vec![0; PTY_READ_BUFFER_BYTES]);
    if buffer.len() < PTY_READ_BUFFER_BYTES {
        buffer.resize(PTY_READ_BUFFER_BYTES, 0);
    }
    buffer
}

/// Returns a drained buffer to the reader thread for reuse, dropping it
/// instead if the recycling channel is full/disconnected or the buffer grew
/// unusually large (e.g. from batching) and isn't worth keeping around.
fn recycle_pty_buffer(buffer: Vec<u8>, recycle_tx: &SyncSender<Vec<u8>>) {
    if buffer.capacity() > MAX_PTY_PARSE_BATCH_BYTES {
        return;
    }
    match recycle_tx.try_send(buffer) {
        Ok(()) | Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {}
    }
}

/// Writes a batch of PTY output into the shared Ghostty terminal and
/// forwards its effects to the UI thread. Returns whether the parser
/// thread should keep running (i.e. the terminal entity hasn't been
/// dropped). Also used directly by `spawn_task_subprocess` (the `no_pty`
/// headless-host path, which has no real PTY/parser thread of its own).
pub(super) fn write_pty_output_to_ghostty(
    output: &[u8],
    terminal: &ParkingMutex<GhosttyTerminal>,
    events_tx: &UnboundedSender<PtyEvent>,
    yield_fair: bool,
) -> bool {
    let mut guard = terminal.lock();
    guard.write(output);
    let effects = guard.take_effects();
    if yield_fair {
        parking_lot::MutexGuard::unlock_fair(guard);
    } else {
        drop(guard);
    }
    events_tx
        .unbounded_send(PtyEvent::GhosttyPtyOutput { effects })
        .is_ok()
}

fn pty_size(bounds: TerminalBounds) -> PtySize {
    let columns = bounds.num_columns().max(1);
    let rows = bounds.num_lines().max(1);
    PtySize {
        rows: saturating_u16(rows),
        cols: saturating_u16(columns),
        pixel_width: saturating_pixel_size(bounds.width()),
        pixel_height: saturating_pixel_size(bounds.height()),
    }
}

fn saturating_u16(value: usize) -> u16 {
    value.min(u16::MAX as usize) as u16
}

fn saturating_pixel_size(value: gpui::Pixels) -> u16 {
    (f32::from(value).ceil().max(1.0) as usize).min(u16::MAX as usize) as u16
}

#[cfg(unix)]
fn portable_pty_exit_status(status: portable_pty::ExitStatus) -> std::process::ExitStatus {
    std::os::unix::process::ExitStatusExt::from_raw((status.exit_code() as i32) << 8)
}

#[cfg(windows)]
fn portable_pty_exit_status(status: portable_pty::ExitStatus) -> std::process::ExitStatus {
    std::os::windows::process::ExitStatusExt::from_raw(status.exit_code())
}

fn ghostty_cursor_shape(style: CursorVisualStyle) -> CursorShape {
    match style {
        CursorVisualStyle::Bar => CursorShape::Bar,
        CursorVisualStyle::Block => CursorShape::Block,
        CursorVisualStyle::Underline => CursorShape::Underline,
        CursorVisualStyle::BlockHollow => CursorShape::HollowBlock,
        _ => CursorShape::Block,
    }
}

/// Converts a resolved cell style color to Zed's `vte::ansi::Color`,
/// preserving the none/palette/RGB distinction (rather than resolving
/// palette indices to a fixed RGB up front) so terminal theme colors keep
/// applying at paint time (`terminal_element::convert_color` matches on
/// `Color::Named`/`Color::Indexed` to look up the active theme's palette,
/// and only bypasses it for `Color::Spec`, an app-chosen exact RGB).
fn color_from_style_color(color: StyleColor, default: vte::ansi::NamedColor) -> vte::ansi::Color {
    match color {
        StyleColor::None => vte::ansi::Color::Named(default),
        StyleColor::Rgb(rgb) => vte::ansi::Color::Spec(vte::ansi::Rgb {
            r: rgb.r,
            g: rgb.g,
            b: rgb.b,
        }),
        StyleColor::Palette(index) => color_from_palette_index(index),
    }
}

/// Converts a Zed `Point` (viewport-relative once `display_offset` is added,
/// per `terminal_element.rs`'s `to_highlighted_range_lines` convention) into
/// Ghostty's `Point::Viewport` space.
fn ghostty_viewport_point(display_offset: usize, point: crate::Point) -> libghostty_vt::terminal::Point {
    libghostty_vt::terminal::Point::Viewport(libghostty_vt::terminal::PointCoordinate {
        x: point.column.min(u16::MAX as usize) as u16,
        y: (point.line + display_offset as i32).max(0) as u32,
    })
}

/// The inverse of `ghostty_viewport_point`: converts a `PointCoordinate`
/// already known to be in Ghostty's `Viewport` space back into a Zed
/// `Point`.
fn zed_point_from_viewport_coordinate(
    display_offset: usize,
    coordinate: libghostty_vt::terminal::PointCoordinate,
) -> crate::Point {
    crate::Point::new(coordinate.y as i32 - display_offset as i32, coordinate.x as usize)
}

/// Converts a freshly-derived `Selection` snapshot (from `select_word`/
/// `select_line`/`select_all`, or any other selection-deriving call) into a
/// `crate::SelectionRange`, without installing it or touching any
/// `GhosttyTerminal` state. It's a free function rather than a
/// `&mut self` method so callers can hold the `Selection` (which borrows
/// `terminal`) across the subsequent `terminal.set_selection(...)` call
/// without the borrow checker treating that as conflicting with a `&mut
/// self` reborrow.
fn selection_range_from_ghostty(
    terminal: &libghostty_vt::Terminal<'static, 'static>,
    display_offset: usize,
    selection: &libghostty_vt::selection::Selection<'_>,
) -> Result<crate::SelectionRange> {
    let start = terminal
        .point_from_grid_ref(&selection.start(), libghostty_vt::terminal::PointSpace::Viewport)?
        .context("selection start not representable in viewport space")?;
    let end = terminal
        .point_from_grid_ref(&selection.end(), libghostty_vt::terminal::PointSpace::Viewport)?
        .context("selection end not representable in viewport space")?;
    Ok(crate::SelectionRange {
        start: zed_point_from_viewport_coordinate(display_offset, start),
        end: zed_point_from_viewport_coordinate(display_offset, end),
        is_block: selection.is_rectangle(),
    })
}

/// Reads the hyperlink URI (if any) of the cell at `point`. Treats any
/// error looking up `point` (e.g. out of range) the same as "no hyperlink
/// there", a safe default for hyperlink detection, where an unreadable
/// point simply isn't part of a hyperlink.
fn hyperlink_uri_at_point(
    terminal: &libghostty_vt::Terminal<'static, 'static>,
    display_offset: usize,
    point: crate::Point,
) -> Result<Option<String>> {
    let Ok(grid_ref) = terminal.grid_ref(ghostty_viewport_point(display_offset, point)) else {
        return Ok(None);
    };
    // Comfortably longer than any real-world URI; hyperlink_uri truncates
    // rather than erroring if a URI somehow exceeds this, which is an
    // acceptable degradation for hyperlink detection.
    let mut buf = [0u8; 4096];
    let Ok(len) = grid_ref.hyperlink_uri(&mut buf) else {
        return Ok(None);
    };
    if len == 0 {
        return Ok(None);
    }
    Ok(Some(String::from_utf8_lossy(&buf[..len.min(buf.len())]).into_owned()))
}

/// Walks from `point` by one column at a time in `direction` (`-1` or `1`),
/// wrapping to the previous/next row at row boundaries, for as long as the
/// cell's hyperlink URI stays equal to `uri`. Returns the furthest point
/// reached (inclusive) that still has the same hyperlink.
fn walk_hyperlink_run(
    terminal: &libghostty_vt::Terminal<'static, 'static>,
    display_offset: usize,
    point: crate::Point,
    uri: &str,
    direction: i64,
) -> Result<crate::Point> {
    let cols = i64::from(terminal.cols()?);
    let mut current = point;
    loop {
        let mut next_column = current.column as i64 + direction;
        let mut next_line = current.line;
        if next_column < 0 {
            next_column = cols - 1;
            next_line -= 1;
        } else if next_column >= cols {
            next_column = 0;
            next_line += 1;
        }
        let next = crate::Point::new(next_line, next_column as usize);
        if hyperlink_uri_at_point(terminal, display_offset, next)?.as_deref() != Some(uri) {
            break;
        }
        current = next;
    }
    Ok(current)
}

/// Small facade over `Terminal::grid_ref(Point::Screen(...))` queries,
/// exposing the absolute-row primitives `ghostty_vi_motion`/
/// `ghostty_word_motion`/`ghostty_bracket_motion` need, in the terms
/// Alacritty's own `alacritty_terminal::vi_mode` module uses them (grid-edge
/// clamped advance/boundary checks, first/last occupied cell in a row,
/// whether a row is entirely blank). Built fresh per call, mirroring
/// `buffer_text_with_point_map`.
struct ViGrid<'a> {
    terminal: &'a libghostty_vt::Terminal<'static, 'static>,
    total_rows: i64,
    cols: i64,
    /// `absolute_row = crate::Point::line + line_base`; see
    /// `buffer_text_with_point_map`'s doc for the same convention.
    line_base: i64,
}

impl<'a> ViGrid<'a> {
    fn new(terminal: &'a libghostty_vt::Terminal<'static, 'static>) -> Result<Self> {
        let total_rows = terminal.total_rows()? as i64;
        let viewport_rows = i64::from(terminal.rows()?);
        Ok(Self {
            terminal,
            total_rows,
            cols: i64::from(terminal.cols()?),
            line_base: total_rows - viewport_rows,
        })
    }

    fn to_point(&self, absolute_row: i64, column: i64) -> crate::Point {
        crate::Point::new((absolute_row - self.line_base) as i32, column as usize)
    }

    fn to_absolute(&self, point: crate::Point) -> (i64, i64) {
        (i64::from(point.line) + self.line_base, point.column as i64)
    }

    fn last_column(&self) -> i64 {
        self.cols - 1
    }

    fn topmost_row(&self) -> i64 {
        0
    }

    fn bottommost_row(&self) -> i64 {
        self.total_rows - 1
    }

    fn grid_ref(&self, absolute_row: i64, column: i64) -> Result<screen::GridRef<'a>> {
        Ok(self.terminal.grid_ref(libghostty_vt::terminal::Point::Screen(
            libghostty_vt::terminal::PointCoordinate {
                x: column.clamp(0, self.last_column()) as u16,
                y: absolute_row.clamp(self.topmost_row(), self.bottommost_row()) as u32,
            },
        ))?)
    }

    fn char_at(&self, absolute_row: i64, column: i64) -> Result<char> {
        let grid_ref = self.grid_ref(absolute_row, column)?;
        let mut buf = [' '; 8];
        let count = grid_ref.graphemes(&mut buf)?;
        Ok(buf.first().copied().filter(|_| count > 0).unwrap_or(' '))
    }

    fn is_wide_spacer(&self, absolute_row: i64, column: i64) -> Result<bool> {
        Ok(self.grid_ref(absolute_row, column)?.cell()?.wide()? == screen::CellWide::SpacerTail)
    }

    fn is_space(&self, absolute_row: i64, column: i64) -> Result<bool> {
        if self.is_wide_spacer(absolute_row, column)? {
            return Ok(false);
        }
        let character = self.char_at(absolute_row, column)?;
        Ok(character == ' ' || character == '\t')
    }

    /// Whether this row wraps into the next (soft-wrapped), i.e.
    /// Alacritty's per-cell `WRAPLINE` flag, but queried as the row-level
    /// property `libghostty-vt` exposes it as.
    fn is_wrap(&self, absolute_row: i64) -> Result<bool> {
        if absolute_row < self.topmost_row() || absolute_row > self.bottommost_row() {
            return Ok(false);
        }
        Ok(self.grid_ref(absolute_row, 0)?.row()?.is_wrapped()?)
    }

    fn is_row_clear(&self, absolute_row: i64) -> Result<bool> {
        for column in 0..self.cols {
            if !self.is_space(absolute_row, column)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn first_occupied_in_line(&self, absolute_row: i64) -> Result<Option<i64>> {
        for column in 0..self.cols {
            if !self.is_space(absolute_row, column)? {
                return Ok(Some(column));
            }
        }
        Ok(None)
    }

    fn last_occupied_in_line(&self, absolute_row: i64) -> Result<Option<i64>> {
        for column in (0..self.cols).rev() {
            if !self.is_space(absolute_row, column)? {
                return Ok(Some(column));
            }
        }
        Ok(None)
    }

    /// Advances one cell in `direction` (`-1` left, `1` right), clamped at
    /// the grid's absolute edges (Alacritty's `Boundary::Grid`: no-op at
    /// the very first/last cell, wraps at row boundaries otherwise).
    fn advance(&self, absolute_row: i64, column: i64, direction: i64) -> (i64, i64) {
        if direction < 0 {
            if column == 0 {
                if absolute_row > self.topmost_row() {
                    (absolute_row - 1, self.last_column())
                } else {
                    (absolute_row, column)
                }
            } else {
                (absolute_row, column - 1)
            }
        } else if column >= self.last_column() {
            if absolute_row < self.bottommost_row() {
                (absolute_row + 1, 0)
            } else {
                (absolute_row, column)
            }
        } else {
            (absolute_row, column + 1)
        }
    }

    fn is_boundary(&self, absolute_row: i64, column: i64, direction: i64) -> bool {
        if direction < 0 {
            absolute_row <= self.topmost_row() && column == 0
        } else {
            absolute_row >= self.bottommost_row() && column >= self.last_column()
        }
    }

    /// Same-row-only simplification of Alacritty's `Term::expand_wide`; see
    /// the "Known simplification" note on the vi-mode section of
    /// `GhosttyTerminal`'s `impl` block.
    fn expand_wide(&self, absolute_row: i64, column: i64, direction: i64) -> Result<(i64, i64)> {
        if direction < 0 && self.is_wide_spacer(absolute_row, column)? && column > 0 {
            return Ok((absolute_row, column - 1));
        }
        Ok((absolute_row, column))
    }
}

/// Hand-port of `alacritty_terminal::vi_mode::ViModeCursor::motion`
/// (`vi_mode.rs` in the vendored `alacritty_terminal` source) against
/// `ViGrid`. See the "Vi mode" section doc on `GhosttyTerminal`'s `impl`
/// block for scope/parity notes.
fn ghostty_vi_motion(
    grid: &ViGrid<'_>,
    point: crate::Point,
    motion: ViMotion,
    display_offset: i64,
) -> Result<crate::Point> {
    let (mut row, mut col) = grid.to_absolute(point);
    match motion {
        ViMotion::Up => {
            if row > grid.topmost_row() {
                row -= 1;
            }
        }
        ViMotion::Down => {
            if row < grid.bottommost_row() {
                row += 1;
            }
        }
        ViMotion::Left => {
            (row, col) = grid.expand_wide(row, col, -1)?;
            if col == 0 && row > grid.topmost_row() && grid.is_wrap(row - 1)? {
                row -= 1;
                col = grid.last_column();
            } else {
                col = (col - 1).max(0);
            }
        }
        ViMotion::Right => {
            (row, col) = grid.expand_wide(row, col, 1)?;
            if grid.is_wrap(row)? {
                row += 1;
                col = 0;
            } else {
                col = (col + 1).min(grid.last_column());
            }
        }
        ViMotion::First => {
            (row, col) = grid.expand_wide(row, col, -1)?;
            while col == 0 && row > grid.topmost_row() && grid.is_wrap(row - 1)? {
                row -= 1;
            }
            col = 0;
        }
        ViMotion::Last => {
            (row, col) = grid.expand_wide(row, col, 1)?;
            let occupied_col = grid.last_occupied_in_line(row)?.unwrap_or(0);
            if col < occupied_col {
                col = occupied_col;
            } else if grid.is_wrap(row)? {
                while grid.is_wrap(row)? {
                    row += 1;
                }
                if let Some(occupied) = grid.last_occupied_in_line(row)? {
                    col = occupied;
                }
            } else {
                col = grid.last_column();
            }
        }
        ViMotion::FirstOccupied => {
            (row, col) = grid.expand_wide(row, col, -1)?;
            let occupied = match grid.first_occupied_in_line(row)? {
                Some(c) => (row, c),
                None => (row, grid.last_column()),
            };
            if (row, col) == occupied {
                let mut found: Option<(i64, i64)> = None;
                let mut r = row - 1;
                while r >= grid.topmost_row() {
                    if !grid.is_wrap(r)? {
                        break;
                    }
                    if let Some(c) = grid.first_occupied_in_line(r)? {
                        found = Some((r, c));
                    }
                    r -= 1;
                }
                (row, col) = match found {
                    Some(p) => p,
                    None => {
                        let mut line = row;
                        loop {
                            if let Some(c) = grid.first_occupied_in_line(line)? {
                                break (line, c);
                            }
                            if !grid.is_wrap(line)? {
                                break (line, grid.last_column());
                            }
                            line += 1;
                        }
                    }
                };
            } else {
                (row, col) = occupied;
            }
        }
        ViMotion::High => {
            row = grid.line_base - display_offset;
            col = grid.first_occupied_in_line(row)?.unwrap_or(0);
        }
        ViMotion::Middle => {
            row = grid.line_base - display_offset + (grid.total_rows - grid.line_base) / 2 - 1;
            col = grid.first_occupied_in_line(row)?.unwrap_or(0);
        }
        ViMotion::Low => {
            row = grid.line_base - display_offset + (grid.total_rows - grid.line_base) - 1;
            col = grid.first_occupied_in_line(row)?.unwrap_or(0);
        }
        ViMotion::WordLeft => (row, col) = ghostty_word_motion(grid, row, col, -1, -1)?,
        ViMotion::WordRight => (row, col) = ghostty_word_motion(grid, row, col, 1, -1)?,
        ViMotion::WordRightEnd => (row, col) = ghostty_word_motion(grid, row, col, 1, 1)?,
        ViMotion::Bracket => (row, col) = ghostty_bracket_motion(grid, row, col)?,
        ViMotion::ParagraphUp => {
            let mut r = row - 1;
            while r >= grid.topmost_row() && grid.is_row_clear(r)? {
                r -= 1;
            }
            let mut found = None;
            while r >= grid.topmost_row() {
                if grid.is_row_clear(r)? {
                    found = Some(r);
                    break;
                }
                r -= 1;
            }
            row = found.unwrap_or(grid.topmost_row());
            col = 0;
        }
        ViMotion::ParagraphDown => {
            let mut r = row + 1;
            while r <= grid.bottommost_row() && grid.is_row_clear(r)? {
                r += 1;
            }
            let mut found = None;
            while r <= grid.bottommost_row() {
                if grid.is_row_clear(r)? {
                    found = Some(r);
                    break;
                }
                r += 1;
            }
            row = found.unwrap_or(grid.bottommost_row());
            col = 0;
        }
    }
    Ok(grid.to_point(row, col))
}

/// Whitespace-word motion (vi's `W`/`B`/`E`; Zed doesn't expose
/// punctuation-aware "semantic" word motions). `direction`/`side` are
/// `-1`/`1`, matching `Direction::Left`/`Right` and `Side::Left`/`Right`
/// from Alacritty's original `vi_mode` implementation, since the exact
/// `(direction, side)` pairs `ghostty_vi_motion`'s call sites above pass
/// for `WordLeft`/`WordRight`/`WordRightEnd` were preserved from that
/// convention verbatim.
fn ghostty_word_motion(
    grid: &ViGrid<'_>,
    row: i64,
    col: i64,
    direction: i64,
    side: i64,
) -> Result<(i64, i64)> {
    let (mut row, mut col) = grid.expand_wide(row, col, direction)?;
    if direction == side {
        let (mut next_row, mut next_col) = grid.advance(row, col, direction);
        while !grid.is_boundary(row, col, direction) && grid.is_space(next_row, next_col)? {
            (row, col) = (next_row, next_col);
            (next_row, next_col) = grid.advance(row, col, direction);
        }
        let (mut next_row, mut next_col) = grid.advance(row, col, direction);
        while !grid.is_boundary(row, col, direction) && !grid.is_space(next_row, next_col)? {
            (row, col) = (next_row, next_col);
            (next_row, next_col) = grid.advance(row, col, direction);
        }
    }
    if direction != side {
        while !grid.is_boundary(row, col, direction) && !grid.is_space(row, col)? {
            (row, col) = grid.advance(row, col, direction);
        }
        while !grid.is_boundary(row, col, direction) && grid.is_space(row, col)? {
            (row, col) = grid.advance(row, col, direction);
        }
    }
    Ok((row, col))
}

/// Small hand-rolled depth-counting bracket matcher for `()`/`[]`/`{}`
/// pairs, since `libghostty-vt` has no equivalent. Returns the unchanged
/// point if it isn't on a bracket character, or if no match is found
/// before hitting a grid edge.
fn ghostty_bracket_motion(grid: &ViGrid<'_>, row: i64, col: i64) -> Result<(i64, i64)> {
    const PAIRS: [(char, char); 3] = [('(', ')'), ('[', ']'), ('{', '}')];
    let character = grid.char_at(row, col)?;
    let (open, close, direction) = if let Some(&(open, close)) =
        PAIRS.iter().find(|(open, _)| *open == character)
    {
        (open, close, 1i64)
    } else if let Some(&(open, close)) = PAIRS.iter().find(|(_, close)| *close == character) {
        (open, close, -1i64)
    } else {
        return Ok((row, col));
    };

    let mut depth: i32 = 1;
    let (mut r, mut c) = (row, col);
    loop {
        let (next_r, next_c) = grid.advance(r, c, direction);
        if (next_r, next_c) == (r, c) {
            return Ok((row, col));
        }
        (r, c) = (next_r, next_c);
        let current = grid.char_at(r, c)?;
        if direction > 0 {
            if current == open {
                depth += 1;
            } else if current == close {
                depth -= 1;
            }
        } else if current == close {
            depth += 1;
        } else if current == open {
            depth -= 1;
        }
        if depth == 0 {
            return Ok((r, c));
        }
    }
}

/// Extracts the wrap-joined logical line containing `point` (walking
/// across soft-wrap boundaries in both directions via `ViGrid::is_wrap`),
/// with a per-char `Point` map (indexed by char count, same convention
/// `buffer_text_with_point_map`/`search_matches` use) and the char-index of
/// `point` within that text. This is the shared foundation for both the
/// bare-URL and path-hyperlink regex fallbacks in
/// `GhosttyTerminal::hyperlink_at`.
///
/// Unlike `buffer_text_with_point_map`, trailing blank cells are NOT
/// trimmed here: hyperlink regexes (especially user-configured path
/// patterns) may legitimately need to see trailing whitespace as a
/// separator, and the caller only ever indexes by explicit byte/char
/// ranges a regex matched, never relies on the string's own length as a
/// line-content boundary the way `search_matches`'s `$`-anchor handling
/// does.
///
/// Returns `None` if `point` isn't covered by the extracted text at all.
/// This is only possible for a wide-character spacer cell with no map
/// entry of its own, in which case the caller should retry at `point`'s
/// column minus one
/// (the wide character's own cell) if it wants spacer-hover support; none
/// of this module's callers currently do, since a mouse hover naturally
/// lands on the wide character's own (wider, so more likely to be under
/// the cursor) cell in the vast majority of cases.
fn single_line_text_with_point_map(
    grid: &ViGrid<'_>,
    point: crate::Point,
) -> Result<Option<(String, Vec<crate::Point>, usize)>> {
    let (point_row, _) = grid.to_absolute(point);

    let mut first_row = point_row;
    while first_row > grid.topmost_row() && grid.is_wrap(first_row - 1)? {
        first_row -= 1;
    }
    let mut last_row = point_row;
    while last_row < grid.bottommost_row() && grid.is_wrap(last_row)? {
        last_row += 1;
    }

    let mut text = String::new();
    let mut points: Vec<crate::Point> = Vec::new();
    let mut hovered_index = None;
    let mut grapheme_buf = [' '; 32];

    for absolute_row in first_row..=last_row {
        // A row that wraps into the next necessarily fills its full width
        // (that's what wrapping means), but a row that doesn't should stop
        // at its last occupied column. Otherwise trailing blank cells
        // pad `text` with spaces that regexes anchored on "rest of line"
        // (e.g. `(?<path>.+)$`) would greedily swallow into the match.
        let last_column = if grid.is_wrap(absolute_row)? {
            grid.last_column()
        } else {
            grid.last_occupied_in_line(absolute_row)?.unwrap_or(-1)
        };
        for column in 0..=last_column {
            let grid_ref = grid.grid_ref(absolute_row, column)?;
            let wide = grid_ref.cell()?.wide()?;
            if wide == screen::CellWide::SpacerTail {
                // Hovering the spacer cell of a wide character is hovering
                // that character, so attribute it to the entry just pushed
                // for the wide char's own column instead of dropping it,
                // since this loop never emits a `text`/`points` entry for
                // spacer columns themselves.
                if hovered_index.is_none() && grid.to_point(absolute_row, column) == point {
                    hovered_index = Some(points.len().saturating_sub(1));
                }
                continue;
            }
            if wide == screen::CellWide::SpacerHead {
                // Placeholder for a wide character that didn't fit at the
                // end of a soft-wrapped row: empty, not real content, so
                // (unlike `SpacerTail`) there's no preceding entry in
                // `points` to attribute a hover to; the wide char itself
                // starts the next row instead.
                continue;
            }
            let this_point = grid.to_point(absolute_row, column);
            if this_point == point {
                hovered_index = Some(points.len());
            }
            let count = grid_ref.graphemes(&mut grapheme_buf)?;
            if count == 0 {
                text.push(' ');
                points.push(this_point);
            } else {
                for &character in &grapheme_buf[..count.min(grapheme_buf.len())] {
                    text.push(character);
                    points.push(this_point);
                }
            }
        }
    }

    Ok(hovered_index.map(|index| (text, points, index)))
}

/// Maps a 0-255 palette index to Zed's `Color`, using `Color::Named` for
/// the 16 standard ANSI slots (0-15) to match how Alacritty's own VTE
/// parser resolves SGR 30-37/90-97, and `Color::Indexed` for the rest.
fn color_from_palette_index(index: style::PaletteIndex) -> vte::ansi::Color {
    use vte::ansi::NamedColor;

    let named = match index.0 {
        0 => NamedColor::Black,
        1 => NamedColor::Red,
        2 => NamedColor::Green,
        3 => NamedColor::Yellow,
        4 => NamedColor::Blue,
        5 => NamedColor::Magenta,
        6 => NamedColor::Cyan,
        7 => NamedColor::White,
        8 => NamedColor::BrightBlack,
        9 => NamedColor::BrightRed,
        10 => NamedColor::BrightGreen,
        11 => NamedColor::BrightYellow,
        12 => NamedColor::BrightBlue,
        13 => NamedColor::BrightMagenta,
        14 => NamedColor::BrightCyan,
        15 => NamedColor::BrightWhite,
        other => return vte::ansi::Color::Indexed(other),
    };
    vte::ansi::Color::Named(named)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_cursor_metadata() {
        let mut terminal = GhosttyTerminal::new(8, 2, 100).unwrap();

        terminal.write(b"A\x1b[H");

        let content = terminal.content_metadata(0, None).unwrap();
        assert_eq!(content.cursor_char, 'A');
        assert_eq!(content.cursor.point, crate::Point::new(0, 0));
    }

    /// `toggle_vi_mode` should enable vi mode with a cursor, and disabling
    /// should clear it. Values are checked against
    /// `alacritty_terminal::vi_mode`'s own `motion_simple` test (same
    /// terminal dimensions, same expected positions) as a parity baseline.
    #[test]
    fn toggle_vi_mode_and_simple_motions_match_alacritty_parity() {
        let mut terminal = GhosttyTerminal::new(20, 20, 100).unwrap();
        terminal.resize(test_bounds(20, 20)).unwrap();

        assert_eq!(terminal.vi_cursor(), None);
        let cursor = terminal.toggle_vi_mode().unwrap();
        assert_eq!(cursor, Some(crate::Point::new(0, 0)));
        assert_eq!(terminal.vi_cursor(), Some(crate::Point::new(0, 0)));

        let point = terminal.vi_motion(ViMotion::Right).unwrap().unwrap();
        assert_eq!(point, crate::Point::new(0, 1));
        let point = terminal.vi_motion(ViMotion::Left).unwrap().unwrap();
        assert_eq!(point, crate::Point::new(0, 0));
        let point = terminal.vi_motion(ViMotion::Down).unwrap().unwrap();
        assert_eq!(point, crate::Point::new(1, 0));
        let point = terminal.vi_motion(ViMotion::Up).unwrap().unwrap();
        assert_eq!(point, crate::Point::new(0, 0));

        let cursor = terminal.toggle_vi_mode().unwrap();
        assert_eq!(cursor, None);
        assert_eq!(terminal.vi_cursor(), None);
    }

    /// `vi_motion`/`vi_goto_point` must no-op (not panic, not silently
    /// enable vi mode) when vi mode isn't currently active.
    #[test]
    fn vi_motion_is_a_noop_when_vi_mode_is_disabled() {
        let mut terminal = GhosttyTerminal::new(20, 20, 100).unwrap();
        terminal.resize(test_bounds(20, 20)).unwrap();

        assert_eq!(terminal.vi_motion(ViMotion::Right).unwrap(), None);
        assert_eq!(terminal.vi_cursor(), None);
        terminal.vi_goto_point(crate::Point::new(0, 5)).unwrap();
        assert_eq!(terminal.vi_cursor(), None);
    }

    /// `vi_goto_point`/`vi_motion` must scroll the viewport to reveal the
    /// new vi cursor position when it's off-screen. This matters
    /// specifically because `activate_match`'s vi-mode branch pushes
    /// `MoveViCursorToPoint` with no companion `ScrollToPoint` for the
    /// same jump (the two are mutually exclusive), so revealing the target
    /// is entirely this function's job.
    #[test]
    fn vi_goto_point_scrolls_to_reveal_an_off_screen_point() {
        let mut terminal = GhosttyTerminal::new(10, 3, 100).unwrap();
        terminal.resize(test_bounds(10, 3)).unwrap();
        for i in 0..20 {
            terminal.write(format!("line {i}\r\n").as_bytes());
        }
        terminal.toggle_vi_mode().unwrap();

        // Deep in scrollback, well above the current (live-bottom) viewport.
        let target = crate::Point::new(-15, 0);
        terminal.vi_goto_point(target).unwrap();
        assert_eq!(terminal.vi_cursor(), Some(target));

        let display_offset = terminal.alacritty_style_display_offset().unwrap() as i32;
        let rendered_row = target.line + display_offset;
        assert!(
            (0..3).contains(&rendered_row),
            "target point should be scrolled into the visible viewport (0..3), \
             got rendered_row={rendered_row} (display_offset={display_offset})"
        );
    }

    /// A plain (non-vi-motion) scroll, such as the mouse wheel or page
    /// up/down, must move the vi cursor along with it so it doesn't stay
    /// pinned to a row that's scrolled out of view.
    #[test]
    fn update_vi_cursor_for_scroll_moves_cursor_with_the_viewport() {
        let mut terminal = GhosttyTerminal::new(10, 3, 100).unwrap();
        terminal.resize(test_bounds(10, 3)).unwrap();
        for i in 0..20 {
            terminal.write(format!("line {i}\r\n").as_bytes());
        }
        let initial = terminal.toggle_vi_mode().unwrap().unwrap();

        let after_delta = terminal
            .update_vi_cursor_for_scroll(Scroll::Delta(5))
            .unwrap()
            .unwrap();
        assert_eq!(
            after_delta.line,
            initial.line - 5,
            "a positive delta should move the vi cursor up into history by that many rows"
        );
        assert_eq!(terminal.vi_cursor(), Some(after_delta));

        let after_top = terminal.update_vi_cursor_for_scroll(Scroll::Top).unwrap().unwrap();
        assert_eq!(
            after_top.line, -18,
            "Top should land on the very first row (line 0; each of the 20 written lines \
             plus the fresh blank row the trailing \\r\\n advances the cursor to is its own \
             row, so this is 18 rows above the initial bottom-of-viewport cursor at line 2)"
        );

        let after_bottom = terminal
            .update_vi_cursor_for_scroll(Scroll::Bottom)
            .unwrap()
            .unwrap();
        assert_eq!(
            after_bottom.line, initial.line,
            "Bottom should land back on the same (bottommost) row the vi cursor started on"
        );
    }

    /// Parity baseline: `alacritty_terminal::vi_mode`'s `motion_start_end`
    /// test. On a blank row, `Last` goes to the final column (nothing
    /// occupied to stop at), `First` returns to column 0.
    #[test]
    fn vi_motion_first_and_last_on_a_blank_row() {
        let mut terminal = GhosttyTerminal::new(20, 20, 100).unwrap();
        terminal.resize(test_bounds(20, 20)).unwrap();
        terminal.toggle_vi_mode().unwrap();

        let point = terminal.vi_motion(ViMotion::Last).unwrap().unwrap();
        assert_eq!(point, crate::Point::new(0, 19));
        let point = terminal.vi_motion(ViMotion::First).unwrap().unwrap();
        assert_eq!(point, crate::Point::new(0, 0));
    }

    /// Parity baseline: `alacritty_terminal::vi_mode`'s
    /// `motion_first_occupied` test. First non-blank cell in a row, with a
    /// second invocation from that exact cell walking backward across
    /// soft-wrapped rows to the first occupied cell of the whole logical
    /// (wrap-joined) line.
    #[test]
    fn vi_motion_first_occupied_walks_back_across_wrapped_lines() {
        let mut terminal = GhosttyTerminal::new(20, 20, 100).unwrap();
        terminal.resize(test_bounds(20, 20)).unwrap();

        // Row 0 (20 chars, wraps into row 1): " x y" + 16 filler chars.
        // Row 1 (20 chars, wraps into row 2): all filler, no content.
        // Row 2: "z " (2 chars), not written as part of a wrap.
        let mut payload = String::new();
        payload.push_str(" x y");
        payload.push_str(&"a".repeat(16));
        payload.push_str(&"b".repeat(20));
        payload.push_str("z ");
        terminal.write(payload.as_bytes());

        terminal.toggle_vi_mode().unwrap();
        terminal.vi_goto_point(crate::Point::new(2, 1)).unwrap();

        let point = terminal.vi_motion(ViMotion::FirstOccupied).unwrap().unwrap();
        assert_eq!(point, crate::Point::new(2, 0), "'z' is the first occupied cell of row 2");

        let point = terminal.vi_motion(ViMotion::FirstOccupied).unwrap().unwrap();
        assert_eq!(
            point,
            crate::Point::new(0, 1),
            "already at row 2's first occupied cell, so this should walk back across the wrap \
             to 'x', the first occupied cell of the whole wrapped run"
        );
    }

    /// Parity baseline: `alacritty_terminal::vi_mode`'s
    /// `motion_high_middle_low` test.
    #[test]
    fn vi_motion_high_middle_low() {
        let mut terminal = GhosttyTerminal::new(20, 20, 100).unwrap();
        terminal.resize(test_bounds(20, 20)).unwrap();
        terminal.toggle_vi_mode().unwrap();

        let point = terminal.vi_motion(ViMotion::High).unwrap().unwrap();
        assert_eq!(point, crate::Point::new(0, 0));
        let point = terminal.vi_motion(ViMotion::Middle).unwrap().unwrap();
        assert_eq!(point, crate::Point::new(9, 0));
        let point = terminal.vi_motion(ViMotion::Low).unwrap().unwrap();
        assert_eq!(point, crate::Point::new(19, 0));
    }

    /// Parity baseline: `alacritty_terminal::vi_mode`'s `motion_bracket`
    /// test.
    #[test]
    fn vi_motion_bracket() {
        let mut terminal = GhosttyTerminal::new(20, 20, 100).unwrap();
        terminal.resize(test_bounds(20, 20)).unwrap();
        terminal.write(b"(x)");
        terminal.toggle_vi_mode().unwrap();
        terminal.vi_goto_point(crate::Point::new(0, 0)).unwrap();

        let point = terminal.vi_motion(ViMotion::Bracket).unwrap().unwrap();
        assert_eq!(point, crate::Point::new(0, 2));
        let point = terminal.vi_motion(ViMotion::Bracket).unwrap().unwrap();
        assert_eq!(point, crate::Point::new(0, 0));
    }

    /// A point that isn't on a bracket character must be left unchanged.
    #[test]
    fn vi_motion_bracket_on_non_bracket_is_a_noop() {
        let mut terminal = GhosttyTerminal::new(20, 20, 100).unwrap();
        terminal.resize(test_bounds(20, 20)).unwrap();
        terminal.write(b"(x)");
        terminal.toggle_vi_mode().unwrap();
        terminal.vi_goto_point(crate::Point::new(0, 1)).unwrap();

        let point = terminal.vi_motion(ViMotion::Bracket).unwrap().unwrap();
        assert_eq!(point, crate::Point::new(0, 1));
    }

    /// Parity baseline: `alacritty_terminal::vi_mode`'s `motion_word` test
    /// (the `WordRightEnd`/`WordLeft`/`WordRight` portion; Zed has no
    /// `WordLeftEnd` motion).
    #[test]
    fn vi_motion_word_right_end_and_left() {
        let mut terminal = GhosttyTerminal::new(20, 20, 100).unwrap();
        terminal.resize(test_bounds(20, 20)).unwrap();
        terminal.write(b"a;  a;");
        terminal.toggle_vi_mode().unwrap();
        terminal.vi_goto_point(crate::Point::new(0, 0)).unwrap();

        let point = terminal.vi_motion(ViMotion::WordRightEnd).unwrap().unwrap();
        assert_eq!(point, crate::Point::new(0, 1));
        let point = terminal.vi_motion(ViMotion::WordRightEnd).unwrap().unwrap();
        assert_eq!(point, crate::Point::new(0, 5));
        let point = terminal.vi_motion(ViMotion::WordLeft).unwrap().unwrap();
        assert_eq!(point, crate::Point::new(0, 4));
        let point = terminal.vi_motion(ViMotion::WordLeft).unwrap().unwrap();
        assert_eq!(point, crate::Point::new(0, 0));
        let point = terminal.vi_motion(ViMotion::WordRight).unwrap().unwrap();
        assert_eq!(point, crate::Point::new(0, 4));
    }

    /// Moving the vi cursor while a selection is active should extend the
    /// selection to follow it, mirroring Alacritty's
    /// `vi_mode_recompute_selection` ("update only if non-empty selection
    /// is present").
    #[test]
    fn vi_motion_extends_an_active_selection() {
        let mut terminal = GhosttyTerminal::new(20, 20, 100).unwrap();
        terminal.resize(test_bounds(20, 20)).unwrap();
        terminal.write(b"hello world");

        terminal
            .set_selection(Some(crate::SelectionRange {
                start: crate::Point::new(0, 0),
                end: crate::Point::new(0, 4),
                is_block: false,
            }))
            .unwrap();
        terminal.toggle_vi_mode().unwrap();
        terminal.vi_goto_point(crate::Point::new(0, 4)).unwrap();

        terminal.vi_motion(ViMotion::Right).unwrap();
        assert_eq!(
            terminal.selection_range(),
            Some(crate::SelectionRange {
                start: crate::Point::new(0, 0),
                end: crate::Point::new(0, 5),
                is_block: false,
            })
        );
    }

    /// A simple pattern should be found on the currently-active
    /// (non-scrollback) screen, with the reported range resolving to the
    /// exact matched columns.
    #[test]
    fn search_matches_finds_a_simple_pattern_on_the_active_screen() {
        let mut terminal = GhosttyTerminal::new(20, 4, 100).unwrap();
        terminal.resize(test_bounds(20, 4)).unwrap();
        terminal.write(b"hello world\r\ngoodbye world");

        let regex = regex::Regex::new("world").unwrap();
        let mut matches = terminal.search_matches(&regex).unwrap();
        matches.sort_by_key(|range| range.start());

        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].start(), crate::Point::new(0, 6));
        assert_eq!(matches[0].end(), crate::Point::new(0, 10));
        assert_eq!(matches[1].start(), crate::Point::new(1, 8));
        assert_eq!(matches[1].end(), crate::Point::new(1, 12));
    }

    /// A match must be findable in scrollback, not just the visible
    /// viewport. This is the whole point of extracting the full history and
    /// active buffer rather than reusing the viewport-scoped `RenderState`.
    #[test]
    fn search_matches_finds_a_pattern_in_scrollback() {
        let mut terminal = GhosttyTerminal::new(20, 3, 100).unwrap();
        terminal.resize(test_bounds(20, 3)).unwrap();
        // Push "NEEDLE" (row 0) far above the viewport with plain filler rows.
        terminal.write(b"NEEDLE\r\n");
        for i in 0..30 {
            terminal.write(format!("filler {i}\r\n").as_bytes());
        }

        let regex = regex::Regex::new("NEEDLE").unwrap();
        let matches = terminal.search_matches(&regex).unwrap();

        assert_eq!(matches.len(), 1, "expected exactly one match in scrollback");
        assert_eq!(matches[0].start().column, 0);
        assert_eq!(matches[0].end().column, 5);
        // The matched line should currently be scrolled well above the
        // active screen (a large negative `line`, viewport row 0 is at
        // display_offset 0 by default).
        assert!(matches[0].start().line < 0);
    }

    /// A pattern spanning a soft-wrapped line boundary must still match,
    /// since soft-wrapped rows are joined into one logical line.
    #[test]
    fn search_matches_spans_a_soft_wrapped_line_boundary() {
        let mut terminal = GhosttyTerminal::new(10, 3, 100).unwrap();
        terminal.resize(test_bounds(10, 3)).unwrap();
        // 15 chars ("wrapped_needle.") on a 10-column terminal: wraps across
        // two physical rows with no explicit newline in between.
        terminal.write(b"wrapped_needle.");

        let regex = regex::Regex::new("needle").unwrap();
        let matches = terminal.search_matches(&regex).unwrap();

        assert_eq!(matches.len(), 1, "expected the match to span the wrap boundary");
        assert_eq!(matches[0].start(), crate::Point::new(0, 8));
        assert_eq!(matches[0].end(), crate::Point::new(1, 3));
    }

    /// A hard newline must still act as a boundary for `^`/`$` anchors, as
    /// opposed to a soft wrap (previous test) which must not.
    #[test]
    fn search_matches_respects_hard_newlines_for_anchors() {
        let mut terminal = GhosttyTerminal::new(20, 3, 100).unwrap();
        terminal.resize(test_bounds(20, 3)).unwrap();
        terminal.write(b"first\r\nsecond\r\nthird");

        let regex = regex::Regex::new("(?m)^second$").unwrap();
        let matches = terminal.search_matches(&regex).unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].start(), crate::Point::new(1, 0));
        assert_eq!(matches[0].end(), crate::Point::new(1, 5));
    }

    /// No URL/path regex matching at all, for tests that only care about
    /// the OSC 8 native path of `hyperlink_at`.
    fn no_regex_fallback() -> (regex::Regex, Vec<regex::Regex>, std::time::Duration) {
        (
            regex::Regex::new("$^").unwrap(), // never matches anything
            Vec::new(),
            std::time::Duration::ZERO,
        )
    }

    fn url_regex_fallback() -> (regex::Regex, Vec<regex::Regex>, std::time::Duration) {
        (
            regex::Regex::new(crate::hyperlinks::URL_REGEX).unwrap(),
            Vec::new(),
            std::time::Duration::ZERO,
        )
    }

    /// `hyperlink_at` should find the OSC 8 hyperlink's URI and its full
    /// contiguous cell range, whether queried from the middle, start, or
    /// end of the linked text.
    #[test]
    fn hyperlink_at_finds_osc8_hyperlink_and_its_full_range() {
        let mut terminal = GhosttyTerminal::new(40, 2, 100).unwrap();
        terminal.resize(test_bounds(40, 2)).unwrap();
        // "click " (columns 0-5, plain) + OSC 8 "here" (columns 6-9) + " end" (plain).
        terminal.write(b"click \x1b]8;;https://example.com\x1b\\here\x1b]8;;\x1b\\ end");
        let (url_regex, path_regexes, timeout) = no_regex_fallback();

        for column in 6..=9 {
            let (uri, is_url, range) = terminal
                .hyperlink_at(crate::Point::new(0, column), &url_regex, &path_regexes, timeout)
                .unwrap()
                .unwrap_or_else(|| panic!("expected a hyperlink at column {column}"));
            assert_eq!(uri, "https://example.com");
            assert!(is_url);
            assert_eq!(range.start(), crate::Point::new(0, 6));
            assert_eq!(range.end(), crate::Point::new(0, 9));
        }
    }

    #[test]
    fn hyperlink_at_returns_none_outside_a_hyperlink() {
        let mut terminal = GhosttyTerminal::new(40, 2, 100).unwrap();
        terminal.resize(test_bounds(40, 2)).unwrap();
        terminal.write(b"click \x1b]8;;https://example.com\x1b\\here\x1b]8;;\x1b\\ end");
        let (url_regex, path_regexes, timeout) = no_regex_fallback();

        assert_eq!(
            terminal
                .hyperlink_at(crate::Point::new(0, 0), &url_regex, &path_regexes, timeout)
                .unwrap(),
            None
        );
        assert_eq!(
            terminal
                .hyperlink_at(crate::Point::new(0, 11), &url_regex, &path_regexes, timeout)
                .unwrap(),
            None
        );
    }

    #[test]
    fn hyperlink_at_distinguishes_adjacent_hyperlinks_with_different_uris() {
        let mut terminal = GhosttyTerminal::new(40, 2, 100).unwrap();
        terminal.resize(test_bounds(40, 2)).unwrap();
        // Two adjacent OSC 8 hyperlinks with no plain text between them:
        // "one" (columns 0-2, uri A) immediately followed by "two" (columns 3-5, uri B).
        terminal.write(
            b"\x1b]8;;https://a.example\x1b\\one\x1b]8;;\x1b\\\x1b]8;;https://b.example\x1b\\two\x1b]8;;\x1b\\",
        );
        let (url_regex, path_regexes, timeout) = no_regex_fallback();

        let (uri_a, _, range_a) = terminal
            .hyperlink_at(crate::Point::new(0, 1), &url_regex, &path_regexes, timeout)
            .unwrap()
            .unwrap();
        assert_eq!(uri_a, "https://a.example");
        assert_eq!(range_a.start(), crate::Point::new(0, 0));
        assert_eq!(range_a.end(), crate::Point::new(0, 2));

        let (uri_b, _, range_b) = terminal
            .hyperlink_at(crate::Point::new(0, 4), &url_regex, &path_regexes, timeout)
            .unwrap()
            .unwrap();
        assert_eq!(uri_b, "https://b.example");
        assert_eq!(range_b.start(), crate::Point::new(0, 3));
        assert_eq!(range_b.end(), crate::Point::new(0, 5));
    }

    /// A bare URL (not wrapped in OSC 8) should be found via `url_regex`,
    /// with trailing punctuation trimmed.
    #[test]
    fn hyperlink_at_finds_bare_url_and_trims_trailing_punctuation() {
        let mut terminal = GhosttyTerminal::new(40, 2, 100).unwrap();
        terminal.resize(test_bounds(40, 2)).unwrap();
        terminal.write(b"visit https://example.com/page, thanks");
        let (url_regex, path_regexes, timeout) = url_regex_fallback();

        // Column 10 is inside "https://example.com/page".
        let (text, is_url, range) = terminal
            .hyperlink_at(crate::Point::new(0, 10), &url_regex, &path_regexes, timeout)
            .unwrap()
            .expect("expected a bare URL match");
        assert_eq!(text, "https://example.com/page");
        assert!(is_url);
        assert_eq!(range.start(), crate::Point::new(0, 6));
        // "visit " is 6 cols, URL is 24 chars, so it spans columns 6..=29;
        // the trailing "," must NOT be included.
        assert_eq!(range.end(), crate::Point::new(0, 29));
    }

    /// A user-configured path-hyperlink regex (named captures
    /// `path`/`line`/`column`) should be found via `path_hyperlink_regexes`,
    /// with the line/column suffix appended to the returned text.
    #[test]
    fn hyperlink_at_finds_path_with_line_and_column_via_user_regex() {
        let mut terminal = GhosttyTerminal::new(60, 2, 100).unwrap();
        terminal.resize(test_bounds(60, 2)).unwrap();
        terminal.write(b"error at src/main.rs:42:7: something broke");
        let url_regex = regex::Regex::new("$^").unwrap();
        let path_regexes = vec![
            regex::Regex::new(r"(?<path>\S+\.rs):(?<line>[0-9]+):(?<column>[0-9]+)").unwrap(),
        ];

        // Column 15 is inside "src/main.rs".
        let (text, is_url, range) = terminal
            .hyperlink_at(
                crate::Point::new(0, 15),
                &url_regex,
                &path_regexes,
                std::time::Duration::from_millis(500),
            )
            .unwrap()
            .expect("expected a path match with line/column");
        assert_eq!(text, "src/main.rs:42:7");
        assert!(!is_url);
        assert_eq!(range.start(), crate::Point::new(0, 9));
    }

    /// `select_word_at` should find exactly the word under the given point,
    /// matching Ghostty's own word-boundary rules (whitespace-delimited by
    /// default).
    #[test]
    fn select_word_at_selects_the_word_under_the_point() {
        let mut terminal = GhosttyTerminal::new(20, 2, 100).unwrap();
        terminal.resize(test_bounds(20, 2)).unwrap();
        terminal.write(b"hello world");

        // Column 7 is inside "world" (h-e-l-l-o-space-w=6,o=7).
        let range = terminal
            .select_word_at(crate::Point::new(0, 7))
            .unwrap()
            .expect("expected a word selection");
        assert_eq!(range.start, crate::Point::new(0, 6));
        assert_eq!(range.end, crate::Point::new(0, 10));
        assert!(!range.is_block);

        assert_eq!(terminal.selection_range(), Some(range));
        assert_eq!(terminal.selection_text().unwrap().as_deref(), Some("world"));
    }

    #[test]
    fn select_line_at_selects_the_whole_line() {
        let mut terminal = GhosttyTerminal::new(20, 2, 100).unwrap();
        terminal.resize(test_bounds(20, 2)).unwrap();
        terminal.write(b"hello world\r\nsecond line");

        let range = terminal
            .select_line_at(crate::Point::new(0, 3))
            .unwrap()
            .expect("expected a line selection");
        assert_eq!(terminal.selection_text().unwrap().as_deref(), Some("hello world"));
        assert_eq!(range.start.line, 0);
        assert_eq!(range.end.line, 0);
    }

    #[test]
    fn select_all_selects_every_line() {
        let mut terminal = GhosttyTerminal::new(20, 2, 100).unwrap();
        terminal.resize(test_bounds(20, 2)).unwrap();
        terminal.write(b"hello world\r\nsecond line");

        terminal.select_all().unwrap().expect("expected a selection");
        let text = terminal.selection_text().unwrap().unwrap();
        assert!(text.contains("hello world"));
        assert!(text.contains("second line"));
    }

    /// Dragging after a double-click should extend the word selection to
    /// cover every word (and the whitespace between them) from the anchor
    /// to the drag point, in either direction. This is the
    /// double-click-and-drag scenario `select_word_between`'s own doc
    /// comment describes.
    #[test]
    fn select_word_range_extends_across_multiple_words() {
        let mut terminal = GhosttyTerminal::new(30, 2, 100).unwrap();
        terminal.resize(test_bounds(30, 2)).unwrap();
        terminal.write(b"git commit --amend now");

        // Double-click "commit" (anchor), drag right onto "amend".
        let range = terminal
            .select_word_range(crate::Point::new(0, 4), crate::Point::new(0, 15))
            .unwrap()
            .expect("expected a word-range selection");
        assert_eq!(terminal.selection_text().unwrap().as_deref(), Some("commit --amend"));
        assert_eq!(terminal.selection_range(), Some(range));

        // Same anchor, drag left onto "git": selection should flip to
        // cover "git" through "commit", not just extend further right.
        let range = terminal
            .select_word_range(crate::Point::new(0, 4), crate::Point::new(0, 1))
            .unwrap()
            .expect("expected a word-range selection");
        assert_eq!(terminal.selection_text().unwrap().as_deref(), Some("git commit"));
        assert_eq!(terminal.selection_range(), Some(range));
    }

    /// Dragging after a triple-click should extend the selection by whole
    /// lines, covering every complete line between the anchor and the drag
    /// point regardless of direction.
    #[test]
    fn select_line_range_extends_across_multiple_lines() {
        let mut terminal = GhosttyTerminal::new(20, 5, 100).unwrap();
        terminal.resize(test_bounds(20, 5)).unwrap();
        terminal.write(b"first\r\nsecond\r\nthird\r\nfourth");

        // Triple-click line 1 ("second"), drag down onto line 2 ("third").
        terminal
            .select_line_range(crate::Point::new(1, 2), crate::Point::new(2, 3))
            .unwrap()
            .expect("expected a line-range selection");
        let text = terminal.selection_text().unwrap().unwrap();
        assert!(text.contains("second"));
        assert!(text.contains("third"));
        assert!(!text.contains("first"));
        assert!(!text.contains("fourth"));

        // Same anchor, drag up onto line 0 ("first"): selection should
        // flip to cover line 0 through line 1, not line 1 through line 2.
        terminal
            .select_line_range(crate::Point::new(1, 2), crate::Point::new(0, 0))
            .unwrap()
            .expect("expected a line-range selection");
        let text = terminal.selection_text().unwrap().unwrap();
        assert!(text.contains("first"));
        assert!(text.contains("second"));
        assert!(!text.contains("third"));
    }

    /// Covers `set_selection`/`update_selection`'s explicit-point-range
    /// path (as opposed to `select_word_at`/`select_line_at`'s
    /// Ghostty-derived path). This is what `Terminal::mouse_drag` calls on
    /// every drag-move event, extending an anchored selection to an
    /// arbitrary point.
    #[test]
    fn set_selection_then_update_selection_extends_the_range() {
        let mut terminal = GhosttyTerminal::new(20, 2, 100).unwrap();
        terminal.resize(test_bounds(20, 2)).unwrap();
        terminal.write(b"hello world");

        terminal
            .set_selection(Some(crate::SelectionRange {
                start: crate::Point::new(0, 0),
                end: crate::Point::new(0, 4),
                is_block: false,
            }))
            .unwrap();
        assert_eq!(terminal.selection_text().unwrap().as_deref(), Some("hello"));

        let extended = terminal.update_selection(crate::Point::new(0, 10)).unwrap();
        assert!(extended);
        assert_eq!(
            terminal.selection_range(),
            Some(crate::SelectionRange {
                start: crate::Point::new(0, 0),
                end: crate::Point::new(0, 10),
                is_block: false,
            })
        );
        assert_eq!(
            terminal.selection_text().unwrap().as_deref(),
            Some("hello world")
        );

        terminal.set_selection(None).unwrap();
        assert_eq!(terminal.selection_range(), None);
        assert_eq!(terminal.selection_text().unwrap(), None);
    }

    #[test]
    fn update_selection_without_an_active_selection_returns_false() {
        let mut terminal = GhosttyTerminal::new(20, 2, 100).unwrap();
        terminal.resize(test_bounds(20, 2)).unwrap();

        let extended = terminal.update_selection(crate::Point::new(0, 3)).unwrap();
        assert!(!extended);
    }

    /// Before any color has been configured, Ghostty stays silent on OSC
    /// 4/10/11/12 queries (unlike XTVERSION or DA1/DA2/cursor position,
    /// which it always answers): no default is set (Zed hadn't yet called
    /// `set_default_bg_color`/`set_default_theme_colors`) and no OSC SET
    /// override has landed either.
    #[test]
    fn does_not_answer_color_queries_independently_before_any_color_is_configured() {
        let mut terminal = GhosttyTerminal::new(8, 2, 100).unwrap();

        terminal.write(b"\x1b]11;?\x1b\\");

        let effects = terminal.take_effects();
        assert!(
            !effects
                .iter()
                .any(|effect| matches!(effect, GhosttyEffect::PtyWrite(_))),
            "Ghostty answered an OSC 11 query on its own despite no configured background \
             color: {effects:?}"
        );
    }

    /// Once `set_default_theme_colors` has configured Ghostty's defaults
    /// (as `Terminal::sync_ghostty_theme_colors` does for every real
    /// terminal, not just this direct unit test), Ghostty answers OSC 4
    /// (any palette index), OSC 10, OSC 11, and OSC 12 queries
    /// independently. A query immediately after an OSC 11 SET, driven
    /// through `Terminal::write_output`, used to produce two identical PTY
    /// responses; this pins that down as the same double-answer bug class
    /// `3dfd8ee4b5` fixed for DA1/DA2/cursor-position responses.
    #[test]
    fn answers_all_color_queries_independently_once_theme_colors_are_configured() {
        let mut terminal = GhosttyTerminal::new(8, 2, 100).unwrap();
        terminal
            .set_default_theme_colors(
                vte::ansi::Rgb {
                    r: 0xDD,
                    g: 0xDD,
                    b: 0xDD,
                },
                vte::ansi::Rgb {
                    r: 0x1E,
                    g: 0x1E,
                    b: 0x2E,
                },
                vte::ansi::Rgb {
                    r: 0xF5,
                    g: 0xE0,
                    b: 0xDC,
                },
                [vte::ansi::Rgb { r: 0, g: 0, b: 0 }; 256],
            )
            .unwrap();
        terminal.take_effects();

        for query in [
            b"\x1b]4;1;?\x1b\\".as_slice(),
            b"\x1b]10;?\x1b\\".as_slice(),
            b"\x1b]11;?\x1b\\".as_slice(),
            b"\x1b]12;?\x1b\\".as_slice(),
        ] {
            terminal.write(query);
            let effects = terminal.take_effects();
            assert!(
                effects
                    .iter()
                    .any(|effect| matches!(effect, GhosttyEffect::PtyWrite(_))),
                "expected Ghostty to answer {query:?} independently once theme colors are \
                 configured, got {effects:?}"
            );
        }
    }

    /// `set_default_theme_colors` must actually change what Ghostty
    /// reports, and an explicit OSC SET override must still win over the
    /// theme default afterwards (an embedder-configured default is a
    /// separate, lower-priority layer from a program's own override, per
    /// `libghostty_vt::Terminal`'s own "color theme" documentation).
    #[test]
    fn set_default_theme_colors_updates_effective_colors_and_survives_being_overridden() {
        let mut terminal = GhosttyTerminal::new(8, 2, 100).unwrap();
        terminal
            .set_default_theme_colors(
                vte::ansi::Rgb {
                    r: 0xDD,
                    g: 0xDD,
                    b: 0xDD,
                },
                vte::ansi::Rgb {
                    r: 0x1E,
                    g: 0x1E,
                    b: 0x2E,
                },
                vte::ansi::Rgb {
                    r: 0xF5,
                    g: 0xE0,
                    b: 0xDC,
                },
                [vte::ansi::Rgb {
                    r: 0x11,
                    g: 0x22,
                    b: 0x33,
                }; 256],
            )
            .unwrap();

        let background = terminal.terminal.bg_color().unwrap().unwrap();
        assert_eq!((background.r, background.g, background.b), (0x1E, 0x1E, 0x2E));
        let palette_entry = terminal
            .terminal
            .color_palette()
            .unwrap()
            .get(style::PaletteIndex(1));
        assert_eq!(
            (palette_entry.r, palette_entry.g, palette_entry.b),
            (0x11, 0x22, 0x33)
        );

        terminal.write(b"\x1b]11;rgb:12/34/56\x1b\\");
        let overridden_background = terminal.terminal.bg_color().unwrap().unwrap();
        assert_eq!(
            (overridden_background.r, overridden_background.g, overridden_background.b),
            (0x12, 0x34, 0x56)
        );
    }

    /// An OSC 52 SET should be detected via the `on_clipboard_write`
    /// callback registered in `GhosttyTerminal::new`, with its base64
    /// payload already decoded to plain text.
    #[test]
    fn detects_osc52_clipboard_write() {
        let mut terminal = GhosttyTerminal::new(8, 2, 100).unwrap();

        let payload = base64_encode(b"hello clipboard");
        terminal.write(format!("\x1b]52;c;{payload}\x1b\\").as_bytes());

        let effects = terminal.take_effects();
        assert!(
            effects.iter().any(
                |effect| matches!(effect, GhosttyEffect::ClipboardStore(text) if text == "hello clipboard")
            ),
            "expected a ClipboardStore effect with the decoded payload, got {effects:?}"
        );
    }

    /// OSC 52 clipboard *read* requests ("?") must never produce a
    /// `ClipboardStore` effect or any PTY response at all. `libghostty-vt`
    /// always ignores them by design (security-conscious default, see
    /// `on_clipboard_write`'s doc comment), which this pins down explicitly
    /// so a future library upgrade changing that behavior doesn't go
    /// unnoticed.
    #[test]
    fn ignores_osc52_clipboard_read_request() {
        let mut terminal = GhosttyTerminal::new(8, 2, 100).unwrap();

        terminal.write(b"\x1b]52;c;?\x1b\\");

        let effects = terminal.take_effects();
        assert!(
            effects.is_empty(),
            "expected an OSC 52 read request to produce no effects at all, got {effects:?}"
        );
    }

    #[test]
    fn reports_xtversion_query_after_construction() {
        let mut terminal = GhosttyTerminal::new(8, 2, 100).unwrap();

        terminal.write(b"\x1b[>0q");

        let effects = terminal.take_effects();
        assert!(
            effects.iter().any(|effect| matches!(
                effect,
                GhosttyEffect::PtyWrite(bytes)
                    if bytes == b"\x1bP>|libghostty\x1b\\"
            )),
            "expected XTVERSION response, got {effects:?}"
        );
    }

    /// Mirrors the crate's own doctest example (`kitty::graphics` module
    /// docs): a Kitty graphics APC transmitting a hardcoded 1x1 PNG.
    #[test]
    fn reports_image_placements() {
        let mut terminal = GhosttyTerminal::new(8, 2, 100).unwrap();
        terminal.resize(test_bounds(8, 2)).unwrap();

        terminal.write(
            b"\x1b_Ga=T,f=100,q=1;\
              iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA\
              DUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==\
              \x1b\\",
        );

        let placements = terminal.image_placements().unwrap();
        assert_eq!(
            placements.len(),
            1,
            "expected one image placement, got {placements:?}"
        );
        let placement = &placements[0];
        // No `c=`/`r=` columns/rows were specified in the APC command, so
        // the placement renders at the image's native 1x1 pixel size rather
        // than being scaled to fill a cell.
        assert_eq!(placement.pixel_width, 1);
        assert_eq!(placement.pixel_height, 1);
        assert_eq!(
            placement.data.len(),
            4,
            "expected one decoded RGBA8 pixel, got {} bytes",
            placement.data.len()
        );
    }

    /// Regression test for a Kitty graphics bug report ("images render at
    /// the wrong height and aren't square"): transmits raw-RGBA square
    /// images with no explicit `r=`/`c=`, so `grid_rows` is auto-computed
    /// from `pixel_height / cell_height` the same way real senders like
    /// `kitty +kitten icat -H` do (`reports_image_placements` above always
    /// omits rows/cols too, but only exercises a 1px image, too small to
    /// catch a rounding bug). Confirms libghostty-vt's own row math matches
    /// the pixel heights in the bug report (32/33/64/65px at a 32px cell
    /// height rounding up to 1/2/2/3 rows), pinning down that this half of
    /// the pipeline is correct.
    #[test]
    fn computes_grid_rows_from_pixel_height_without_explicit_rows() {
        let cell_width = gpui::px(16.);
        let cell_height = gpui::px(32.);
        let bounds = TerminalBounds::new(
            cell_height,
            cell_width,
            gpui::Bounds {
                origin: gpui::Point::default(),
                size: gpui::Size {
                    width: cell_width * 10.0,
                    height: cell_height * 10.0,
                },
            },
        );

        for (height, expected_rows) in [(32u32, 1u32), (33, 2), (64, 2), (65, 3)] {
            let mut terminal = GhosttyTerminal::new(10, 10, 100).unwrap();
            terminal.resize(bounds).unwrap();

            let data = base64_encode(&[255u8, 0, 0, 255].repeat((height * height) as usize));
            terminal.write(
                format!("\x1b_Ga=T,f=32,s={height},v={height},q=1;{data}\x1b\\").as_bytes(),
            );

            let placements = terminal.image_placements().unwrap();
            assert_eq!(placements.len(), 1, "height={height}: expected one placement");
            assert_eq!(
                placements[0].grid_rows, expected_rows,
                "height={height}: expected {expected_rows} rows"
            );
            assert_eq!(placements[0].pixel_height, height);
            assert_eq!(placements[0].pixel_width, height, "image is square");
        }
    }

    /// Regression test for a bug report ("`red_square 400`/`4000` renders
    /// with one or more extra blank lines below it, `kv -H` resize-to-fit
    /// too"): with a *fractional* cell height (the common case for real
    /// fonts, unlike the previous test above which only ever used
    /// whole-pixel 32px cells), `GhosttyTerminal::resize` used to
    /// truncate `cell_height`/`cell_width` toward zero (`as u32`) before
    /// passing them to `libghostty_vt::Terminal::resize`. Truncation always
    /// recovers a too-small cell height inside `gridSize`'s
    /// `divCeil(image_height_px, t.height_px / t.rows)`, which always
    /// biases the row count up. This is invisible for small images
    /// (rounding error smaller than one row) but grows without bound as
    /// image height increases, exactly matching "worse for a bigger
    /// image". Fixed by rounding instead of truncating.
    #[test]
    fn rounds_fractional_cell_height_for_grid_row_computation() {
        let cell_width = gpui::px(9.4);
        let cell_height = gpui::px(19.7);
        let bounds = TerminalBounds::new(
            cell_height,
            cell_width,
            gpui::Bounds {
                origin: gpui::Point::default(),
                size: gpui::Size {
                    width: cell_width * 80.0,
                    height: cell_height * 40.0,
                },
            },
        );

        for height in [400u32, 2000] {
            let mut terminal = GhosttyTerminal::new(80, 40, 10_000).unwrap();
            terminal.resize(bounds).unwrap();

            let data = base64_encode(&[255u8, 0, 0, 255].repeat((height * height) as usize));
            terminal.write(
                format!("\x1b_Ga=T,f=32,s={height},v={height},q=1;{data}\x1b\\").as_bytes(),
            );

            let placements = terminal.image_placements().unwrap();
            assert_eq!(placements.len(), 1, "height={height}: expected one placement");
            let rounded_cell_height = f32::from(cell_height).round();
            let expected_rows = (height as f32 / rounded_cell_height).ceil() as u32;
            assert_eq!(
                placements[0].grid_rows, expected_rows,
                "height={height}: expected {expected_rows} rows from rounded {rounded_cell_height}px \
                 cell height, got {}",
                placements[0].grid_rows,
            );
        }
    }

    #[test]
    fn image_placement_taller_than_viewport_reports_correct_bounds() {
        let cell_width = gpui::px(10.0);
        let cell_height = gpui::px(20.0);
        let bounds = TerminalBounds::new(
            cell_height,
            cell_width,
            gpui::Bounds {
                origin: gpui::Point::default(),
                size: gpui::Size {
                    width: cell_width * 80.0,
                    height: cell_height * 40.0,
                },
            },
        );

        let mut terminal = GhosttyTerminal::new(80, 40, 10_000).unwrap();
        terminal.resize(bounds).unwrap();

        let height = 2000u32; // 2000px = 100 rows
        let width = 2000u32;
        let data = base64_encode(&[255u8, 0, 0, 255].repeat((width * height) as usize));
        terminal.write(
            format!("\x1b_Ga=T,f=32,s={width},v={height},q=1;{data}\x1b\\").as_bytes(),
        );

        let placements = terminal.image_placements().unwrap();
        assert_eq!(placements.len(), 1, "expected 1 placement");
        let p = &placements[0];
        eprintln!(
            "Placement: viewport_row={}, viewport_col={}, grid_rows={}, grid_cols={}, px_w={}, px_h={}",
            p.viewport_row, p.viewport_column, p.grid_rows, p.grid_columns, p.pixel_width, p.pixel_height
        );
    }

    /// `build_content` is the highest-risk piece here, since a wrong
    /// style/color mapping would visibly break every terminal session.
    /// Covers plain text, bold/italic/underline/undercurl/strikeout, and
    /// the three color kinds (`Color::Named`, `Indexed`, `Spec`) that
    /// `terminal_element::convert_color` treats differently
    /// (`Named`/`Indexed` resolve against the active theme, `Spec` is an
    /// app-chosen exact RGB). This is why colors must round-trip through
    /// `StyleColor` rather than being flattened to a fixed RGB up front.
    #[test]
    fn build_content_reports_text_and_styles() {
        let mut terminal = GhosttyTerminal::new(20, 3, 100).unwrap();
        terminal.resize(test_bounds(20, 3)).unwrap();

        terminal.write(
            b"\x1b[1mB\x1b[22m\x1b[3mI\x1b[23m\x1b[4mU\x1b[24m\x1b[4:3mC\x1b[24m\x1b[9mS\x1b[29m",
        );
        // Named (SGR 31 = red), 256-indexed (SGR 38;5;99), and true-color
        // (SGR 38;2;r;g;b) foreground colors, one cell each.
        terminal.write(b"\x1b[31mN\x1b[39m\x1b[38;5;99mP\x1b[39m\x1b[38;2;10;20;30mR\x1b[39m");

        let (cells, _mode, _display_offset) = terminal.build_content().unwrap();
        let cell = |col: usize| {
            cells
                .iter()
                .find(|c| c.point == crate::Point::new(0, col))
                .unwrap_or_else(|| panic!("no cell at column {col}"))
        };

        assert_eq!(cell(0).character(), 'B');
        assert!(cell(0).is_bold());
        assert_eq!(cell(1).character(), 'I');
        assert!(cell(1).is_italic());
        assert_eq!(cell(2).character(), 'U');
        assert!(cell(2).has_underline());
        assert!(!cell(2).has_undercurl());
        assert_eq!(cell(3).character(), 'C');
        assert!(cell(3).has_underline());
        assert!(cell(3).has_undercurl());
        assert_eq!(cell(4).character(), 'S');
        assert!(cell(4).has_strikeout());

        assert_eq!(cell(5).character(), 'N');
        assert_eq!(
            cell(5).foreground(),
            vte::ansi::Color::Named(vte::ansi::NamedColor::Red)
        );
        assert_eq!(cell(6).character(), 'P');
        assert_eq!(cell(6).foreground(), vte::ansi::Color::Indexed(99));
        assert_eq!(cell(7).character(), 'R');
        assert_eq!(
            cell(7).foreground(),
            vte::ansi::Color::Spec(vte::ansi::Rgb {
                r: 10,
                g: 20,
                b: 30
            })
        );

        // A cell that never had anything written to it keeps the terminal's
        // default (unstyled) foreground/background so themes still apply.
        assert_eq!(
            cell(10).foreground(),
            vte::ansi::Color::Named(vte::ansi::NamedColor::Foreground)
        );
        assert_eq!(
            cell(10).background(),
            vte::ansi::Color::Named(vte::ansi::NamedColor::Background)
        );
    }

    #[test]
    fn build_content_reports_zero_width_combining_marks() {
        let mut terminal = GhosttyTerminal::new(10, 2, 100).unwrap();
        terminal.resize(test_bounds(10, 2)).unwrap();

        // "e" followed by a combining acute accent (U+0301): one grapheme cluster,
        // two codepoints. Exercises build_content's stack-buffered path (well under
        // its 32-char inline capacity).
        terminal.write("e\u{0301}".as_bytes());

        let (cells, _mode, _display_offset) = terminal.build_content().unwrap();
        let cell = cells
            .iter()
            .find(|c| c.point == crate::Point::new(0, 0))
            .expect("no cell at column 0");

        assert_eq!(cell.character(), 'e');
        assert_eq!(cell.zerowidth(), Some(&['\u{0301}'][..]));
    }

    #[test]
    fn build_content_reports_a_grapheme_cluster_past_the_inline_buffer() {
        let mut terminal = GhosttyTerminal::new(10, 2, 100).unwrap();
        terminal.resize(test_bounds(10, 2)).unwrap();

        // A base character followed by 40 combining marks, more than
        // build_content's 32-char inline stack buffer, forcing its heap
        // fallback path.
        let mut cluster = String::from("e");
        for _ in 0..40 {
            cluster.push('\u{0301}');
        }
        terminal.write(cluster.as_bytes());

        let (cells, _mode, _display_offset) = terminal.build_content().unwrap();
        let cell = cells
            .iter()
            .find(|c| c.point == crate::Point::new(0, 0))
            .expect("no cell at column 0");

        assert_eq!(cell.character(), 'e');
        assert_eq!(
            cell.zerowidth().map(|chars| chars.len()),
            Some(40),
            "expected all 40 combining marks to survive the heap fallback path"
        );
    }

    #[test]
    fn build_content_reports_wide_char_spacer() {
        let mut terminal = GhosttyTerminal::new(10, 2, 100).unwrap();
        terminal.resize(test_bounds(10, 2)).unwrap();

        // U+4E2D ("中") is a double-width CJK character.
        terminal.write("中".as_bytes());

        let (cells, _mode, _display_offset) = terminal.build_content().unwrap();
        let first = cells
            .iter()
            .find(|c| c.point == crate::Point::new(0, 0))
            .unwrap();
        let second = cells
            .iter()
            .find(|c| c.point == crate::Point::new(0, 1))
            .unwrap();

        assert_eq!(first.character(), '中');
        assert!(!first.is_wide_char_spacer());
        assert!(second.is_wide_char_spacer());
    }

    /// `Terminal::scrollbar().offset` and `display_offset` use *opposite*
    /// polarity (Ghostty: `0` means scrolled to the top of history;
    /// `display_offset`: `0` means not scrolled, at the live bottom).
    /// Treating the raw `scrollbar().offset` value directly as if it
    /// already were `display_offset` happens to pass on small terminals
    /// with no scrollback (where `scrollbar().offset` is always `0` either
    /// way); the bug only surfaces once the viewport is actually scrolled
    /// into history, exactly what this test forces by writing far more
    /// content than the viewport can hold and never scrolling back down.
    #[test]
    fn build_content_reports_alacritty_style_display_offset_when_scrolled() {
        let mut terminal = GhosttyTerminal::new(10, 3, 100).unwrap();
        terminal.resize(test_bounds(10, 3)).unwrap();

        // 20 lines into a 3-row viewport: 17 rows of scrollback, viewport
        // sitting at the live bottom (not manually scrolled at all).
        for i in 0..20 {
            terminal.write(format!("line {i}\r\n").as_bytes());
        }

        let (_cells, _mode, display_offset) = terminal.build_content().unwrap();
        assert_eq!(
            display_offset, 0,
            "not scrolled (viewing the live bottom) must report Alacritty-style \
             display_offset 0, not Ghostty's raw (inverted-polarity) scrollbar offset"
        );
    }

    fn base64_encode(data: &[u8]) -> String {
        const ALPHABET: &[u8] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in data.chunks(3) {
            let b0 = chunk[0];
            let b1 = *chunk.get(1).unwrap_or(&0);
            let b2 = *chunk.get(2).unwrap_or(&0);
            out.push(ALPHABET[(b0 >> 2) as usize] as char);
            out.push(ALPHABET[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
            out.push(if chunk.len() > 1 {
                ALPHABET[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char
            } else {
                '='
            });
            out.push(if chunk.len() > 2 {
                ALPHABET[(b2 & 0x3f) as usize] as char
            } else {
                '='
            });
        }
        out
    }

    fn test_bounds(cols: u16, rows: u16) -> TerminalBounds {
        let cell_width = gpui::px(8.);
        let line_height = gpui::px(16.);
        TerminalBounds::new(
            line_height,
            cell_width,
            gpui::Bounds {
                origin: gpui::Point::default(),
                size: gpui::Size {
                    width: cell_width * cols as f32,
                    height: line_height * rows as f32,
                },
            },
        )
    }

    /// Regression suite for `GhosttyTerminal::hyperlink_at`'s path/IRI
    /// matching, using `test_path!`/`test_iri!`/`test_file_iri!` macros.
    /// These drive a real `GhosttyTerminal` through actual VT bytes
    /// (`write`) and track positions via `Terminal::cursor_x`/`cursor_y`/
    /// `is_cursor_pending_wrap`. Failures report the plain expected/actual
    /// values; see `hyperlinks.rs`'s own module for its separate
    /// algorithmic performance benchmarks
    /// (`path_hyperlink_candidates_in_line`) and regression tests.
    mod hyperlink_regressions {
        use super::*;
        use crate::hyperlinks::RegexSearches;
        use std::ops::RangeInclusive;
        use util::paths::PathWithPosition;

        #[derive(Debug, Clone, Copy, PartialEq)]
        enum HyperlinkKind {
            FileIri,
            Iri,
            Path,
        }

        struct ExpectedHyperlink {
            hovered_point: crate::Point,
            hovered_char: char,
            hyperlink_kind: HyperlinkKind,
            iri_or_path: String,
            row: Option<u32>,
            column: Option<u32>,
            hyperlink_match: RangeInclusive<crate::Point>,
        }

        fn line_cells_count(line: &str) -> usize {
            fn width(c: char) -> usize {
                match c {
                    // Fullwidth unicode characters used in tests
                    '例' | '🏃' | '🦀' | '🔥' => 2,
                    '\t' => 8, // it's really 0-8, use the max always
                    _ => 1,
                }
            }
            const CONTROL_CHARS: &str = "‹«👉👈»›";
            line.chars()
                .filter(|c| !CONTROL_CHARS.contains(*c))
                .map(width)
                .sum::<usize>()
        }

        /// The cursor's position in absolute (scrollback-inclusive) grid
        /// coordinates, rather than `crate::Point`'s viewport-relative
        /// `line`. While the test's terminal keeps growing scrollback as
        /// later lines are written, `line_base` (and thus what a given
        /// `crate::Point` means) shifts with it, so positions captured
        /// while writing must stay in this absolute space until everything
        /// is written and a single final `ViGrid` can convert them all
        /// consistently (see `build_terminal_from_test_lines`'s last step).
        fn cursor_point(terminal: &libghostty_vt::Terminal<'static, 'static>) -> Result<(i64, i64)> {
            let grid = ViGrid::new(terminal)?;
            let point = crate::Point::new(
                terminal.cursor_y().unwrap_or(0) as i32,
                terminal.cursor_x().unwrap_or(0) as usize,
            );
            Ok(grid.to_absolute(point))
        }

        /// Mirrors Alacritty's `prev_input_point_from_term`: the grid point
        /// of the last character actually written, accounting for pending
        /// wrap (the cursor hasn't visually advanced past it yet) and
        /// wide-char spacer cells.
        fn prev_input_point(terminal: &libghostty_vt::Terminal<'static, 'static>) -> (i64, i64) {
            let Ok(mut point) = cursor_point(terminal) else {
                return (0, 0);
            };
            let Ok(grid) = ViGrid::new(terminal) else {
                return point;
            };
            if !terminal.is_cursor_pending_wrap().unwrap_or(false) {
                point = grid.advance(point.0, point.1, -1);
            }
            if let Ok(expanded) = grid.expand_wide(point.0, point.1, -1) {
                point = expanded;
            }
            point
        }

        /// The point one cell past `point` (Alacritty's `Boundary::Grid`
        /// `.add(term, Boundary::Grid, 1)`), for the 👈 ("hovered on the
        /// wide-char spacer of the previous char") marker.
        fn next_point(
            terminal: &libghostty_vt::Terminal<'static, 'static>,
            point: (i64, i64),
        ) -> (i64, i64) {
            let Ok(grid) = ViGrid::new(terminal) else {
                return point;
            };
            grid.advance(point.0, point.1, 1)
        }

        #[derive(Default, Eq, PartialEq)]
        enum HoveredState {
            #[default]
            HoveredScan,
            HoveredNextChar,
            Done,
        }

        #[derive(Default, Eq, PartialEq, Clone)]
        enum MatchState {
            #[default]
            MatchScan,
            MatchNextChar,
            Match((i64, i64)),
            Done,
        }

        #[derive(Default, Eq, PartialEq, Clone)]
        enum CapturesState {
            #[default]
            PathScan,
            PathNextChar,
            Path(String),
            RowScan,
            Row(String),
            ColumnScan,
            Column(String),
            Done,
        }

        /// Converts to Windows-style paths on Windows, like `path!()`, but
        /// at runtime for improved test readability.
        fn build_terminal_from_test_lines<'a>(
            hyperlink_kind: HyperlinkKind,
            columns: usize,
            rows: usize,
            test_lines: impl Iterator<Item = &'a str>,
        ) -> (GhosttyTerminal, ExpectedHyperlink) {
            let mut hovered_point: Option<(i64, i64)> = None;
            let mut hyperlink_match: std::ops::RangeInclusive<(i64, i64)> = (0, 0)..=(0, 0);
            let mut iri_or_path = String::default();
            let mut row = None;
            let mut column = None;
            let mut prev_point: (i64, i64) = (0, 0);
            let mut hovered_state = HoveredState::default();
            let mut match_state = MatchState::default();
            let mut captures_state = CapturesState::default();
            let mut terminal =
                GhosttyTerminal::new(columns.max(1) as u16, rows.max(1) as u16, 100).unwrap();

            for text in test_lines {
                let mut chars = text.chars().peekable();
                while let Some(c) = chars.next() {
                    match c {
                        '👉' => {
                            hovered_state = HoveredState::HoveredNextChar;
                        }
                        '👈' => {
                            hovered_point = Some(next_point(&terminal.terminal, prev_point));
                        }
                        '«' | '»' => {
                            captures_state = match captures_state {
                                CapturesState::PathScan => CapturesState::PathNextChar,
                                CapturesState::PathNextChar => {
                                    panic!("Should have been handled by char input")
                                }
                                CapturesState::Path(captured) => {
                                    iri_or_path = captured;
                                    CapturesState::RowScan
                                }
                                CapturesState::RowScan => CapturesState::Row(String::new()),
                                CapturesState::Row(number) => {
                                    row = Some(number.parse::<u32>().unwrap());
                                    CapturesState::ColumnScan
                                }
                                CapturesState::ColumnScan => CapturesState::Column(String::new()),
                                CapturesState::Column(number) => {
                                    column = Some(number.parse::<u32>().unwrap());
                                    CapturesState::Done
                                }
                                CapturesState::Done => {
                                    panic!("Extra '«', '»'")
                                }
                            }
                        }
                        '‹' | '›' => {
                            match_state = match match_state {
                                MatchState::MatchScan => MatchState::MatchNextChar,
                                MatchState::MatchNextChar => {
                                    panic!("Should have been handled by char input")
                                }
                                MatchState::Match(start_point) => {
                                    hyperlink_match = start_point..=prev_point;
                                    MatchState::Done
                                }
                                MatchState::Done => {
                                    panic!("Extra '‹', '›'")
                                }
                            }
                        }
                        _ => {
                            if let CapturesState::Row(number) | CapturesState::Column(number) =
                                &mut captures_state
                            {
                                number.push(c)
                            }
                            if let CapturesState::Path(captured) = &mut captures_state {
                                captured.push(c)
                            }

                            let is_windows_abs_path_start = captures_state
                                == CapturesState::PathNextChar
                                && cfg!(windows)
                                && hyperlink_kind == HyperlinkKind::Path
                                && c == '/'
                                && chars.peek().is_some_and(|c| *c != '/');

                            if is_windows_abs_path_start {
                                // Convert Unix abs path start into Windows abs path start so
                                // that the same test can be used for both OSes.
                                terminal.write(b"C:");
                                terminal.write(b"\\");
                                prev_point = prev_input_point(&terminal.terminal);
                            } else {
                                let mut buf = [0u8; 4];
                                terminal.write(c.encode_utf8(&mut buf).as_bytes());
                                prev_point = prev_input_point(&terminal.terminal);
                            }

                            if hovered_state == HoveredState::HoveredNextChar {
                                hovered_point = Some(prev_point);
                                hovered_state = HoveredState::Done;
                            }
                            if captures_state == CapturesState::PathNextChar {
                                captures_state = CapturesState::Path(String::new());
                            }
                            if let CapturesState::Path(captured) = &mut captures_state
                                && captured.is_empty()
                            {
                                captured.push(if is_windows_abs_path_start { '\\' } else { c });
                            }
                            if match_state == MatchState::MatchNextChar {
                                match_state = MatchState::Match(prev_point);
                            }
                        }
                    }
                }
                terminal.write(b"\r\n");
            }

            if hyperlink_kind == HyperlinkKind::FileIri {
                let url = url::Url::parse(&iri_or_path)
                    .unwrap_or_else(|error| panic!("Failed to parse file IRI `{iri_or_path}`: {error}"));
                let path = url.to_file_path().unwrap_or_else(|_| {
                    panic!("Failed to interpret file IRI `{iri_or_path}` as a path")
                });
                iri_or_path = path.to_string_lossy().into_owned();
            }

            let hovered_point = hovered_point.expect("Missing hovered point (👉 or 👈)");
            // All positions above were tracked in absolute (scrollback-
            // inclusive) coordinates precisely so they can be converted to
            // `crate::Point` together, right now, through one `ViGrid`
            // snapshot. This matches what `GhosttyTerminal::hyperlink_at`
            // itself builds when the caller queries it next, right after
            // this function returns and before anything else is written.
            let final_grid = ViGrid::new(&terminal.terminal).unwrap();
            let hovered_char = final_grid
                .char_at(hovered_point.0, hovered_point.1)
                .unwrap_or(' ');
            let hovered_point = final_grid.to_point(hovered_point.0, hovered_point.1);
            let hyperlink_match = final_grid.to_point(hyperlink_match.start().0, hyperlink_match.start().1)
                ..=final_grid.to_point(hyperlink_match.end().0, hyperlink_match.end().1);
            (
                terminal,
                ExpectedHyperlink {
                    hovered_point,
                    hovered_char,
                    hyperlink_kind,
                    iri_or_path,
                    row,
                    column,
                    hyperlink_match,
                },
            )
        }

        fn format_hyperlink_match(hyperlink_match: &RangeInclusive<crate::Point>) -> String {
            format!(
                "({}, {})..=({}, {})",
                hyperlink_match.start().line,
                hyperlink_match.start().column,
                hyperlink_match.end().line,
                hyperlink_match.end().column,
            )
        }

        fn test_hyperlink<'a>(
            columns: usize,
            total_cells: usize,
            test_lines: impl Iterator<Item = &'a str>,
            hyperlink_kind: HyperlinkKind,
            source_location: &str,
        ) {
            const CARGO_DIR_REGEX: &str =
                r#"\s+(Compiling|Checking|Documenting) [^(]+\((?<link>(?<path>.+))\)"#;
            const RUST_DIAGNOSTIC_REGEX: &str = r#"\s+(-->|:::|at) (?<link>(?<path>.+?))(:$|$)"#;
            const ISSUE_12338_REGEX: &str =
                r#"[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (?<link>(?<path>.+))"#;
            // `🏛\u{FE0F}?`: unlike Alacritty, Ghostty's grid preserves the
            // VARIATION SELECTOR-16 the test literal writes after 🏛 as
            // part of that cell's own grapheme cluster rather than
            // dropping it, so the literal building-emoji match must accept
            // it optionally to still match the extracted line text.
            const MULTIPLE_SAME_LINE_REGEX: &str =
                r#"(?<link>(?<path>🦀 multiple_same_line 🦀) 🚣(?<line>[0-9]+) 🏛\u{FE0F}?(?<column>[0-9]+)):"#;
            // The two entries `terminal.path_hyperlink_regexes` defaults to
            // in `assets/settings/default.json`, transcribed verbatim (each
            // multi-line entry's array of strings is joined with `\n` by
            // `terminal_settings.rs`, exactly as done here) so this test
            // module doesn't need a `settings` crate dependency just to
            // exercise the real default.
            const PYTHON_DIAGNOSTIC_REGEX: &str =
                r#"File "(?<path>[^"]+)", line (?<line>[0-9]+)"#;
            const DEFAULT_PATH_REGEX: &str = concat!(
                "(?x)\n",
                "(?<path>\n",
                "    (\n",
                "        # multi-char path: first char (not opening delimiter, space, or box drawing char)\n",
                "        [^({\\[<\"'`\\ \\u2500-\\u257F]\n",
                "        # middle chars: non-space, and colon/paren only if not followed by digit/paren/space\n",
                "        ([^\\ :(]|[:(][^0-9()\\ ])*\n",
                "        # last char: not closing delimiter or colon\n",
                "        [^()}\\]>\"'`.,;:\\ ]\n",
                "    |\n",
                "        # single-char path: not delimiter, punctuation, space, or box drawing char\n",
                "        [^(){}\\[\\]<>\"'`.,;:\\ \\u2500-\\u257F]\n",
                "    )\n",
                "    # optional line/column suffix (included in path for PathWithPosition::parse_str)\n",
                "    (:+[0-9]+(:[0-9]+)?|:?\\([0-9]+([,:]?[0-9]+)?\\))?\n",
                ")",
            );
            const PATH_HYPERLINK_TIMEOUT: std::time::Duration =
                std::time::Duration::from_millis(1000);

            let regex_searches = RegexSearches::new(
                [
                    RUST_DIAGNOSTIC_REGEX,
                    CARGO_DIR_REGEX,
                    ISSUE_12338_REGEX,
                    MULTIPLE_SAME_LINE_REGEX,
                    PYTHON_DIAGNOSTIC_REGEX,
                    DEFAULT_PATH_REGEX,
                ],
                PATH_HYPERLINK_TIMEOUT,
            );

            let rows = total_cells / columns + 2;
            let (mut terminal, expected_hyperlink) =
                build_terminal_from_test_lines(hyperlink_kind, columns, rows, test_lines);

            let url_regex = regex::Regex::new(crate::hyperlinks::URL_REGEX).unwrap();
            let hyperlink_found = terminal.hyperlink_at(
                expected_hyperlink.hovered_point,
                &url_regex,
                regex_searches.compiled_path_hyperlink_regexes(),
                regex_searches.path_hyperlink_timeout(),
            );

            match hyperlink_found {
                Ok(Some((text, is_url, range))) if !is_url => {
                    assert_ne!(
                        expected_hyperlink.hyperlink_kind,
                        HyperlinkKind::Iri,
                        "\n    at {source_location}\nExpected a path, but was a iri (hovered {:?})",
                        expected_hyperlink.hovered_char,
                    );
                    let path_with_position = PathWithPosition::parse_str(&text);
                    let expected = (
                        expected_hyperlink.iri_or_path.as_str(),
                        expected_hyperlink.row,
                        expected_hyperlink.column,
                        format_hyperlink_match(&expected_hyperlink.hyperlink_match),
                    );
                    let actual = (
                        path_with_position.path.to_string_lossy().into_owned(),
                        path_with_position.row,
                        path_with_position.column,
                        format_hyperlink_match(&(range.start()..=range.end())),
                    );
                    assert_eq!(
                        (expected.0, expected.1, expected.2, expected.3.as_str()),
                        (actual.0.as_str(), actual.1, actual.2, actual.3.as_str()),
                        "\n    at {source_location}"
                    );
                }
                Ok(Some((text, _is_url, range))) => {
                    assert_ne!(
                        expected_hyperlink.hyperlink_kind,
                        HyperlinkKind::Path,
                        "\n    at {source_location}\nExpected a path, but was a iri"
                    );
                    // `hyperlink_at` returns `file://` IRIs as raw URI
                    // text, same as any other IRI; `expected_hyperlink`
                    // already converted its own `file://` IRI into a
                    // filesystem path (see `build_terminal_from_test_lines`),
                    // so mirror that conversion here before comparing.
                    let text = if expected_hyperlink.hyperlink_kind == HyperlinkKind::FileIri {
                        let url = url::Url::parse(&text).unwrap_or_else(|error| {
                            panic!("Failed to parse file IRI `{text}`: {error}")
                        });
                        let path = url.to_file_path().unwrap_or_else(|_| {
                            panic!("Failed to interpret file IRI `{text}` as a path")
                        });
                        path.to_string_lossy().into_owned()
                    } else {
                        text
                    };
                    assert_eq!(
                        (
                            expected_hyperlink.iri_or_path.as_str(),
                            format_hyperlink_match(&expected_hyperlink.hyperlink_match).as_str()
                        ),
                        (text.as_str(), format_hyperlink_match(&(range.start()..=range.end())).as_str()),
                        "\n    at {source_location}"
                    );
                }
                Ok(None) => {
                    assert_eq!(
                        expected_hyperlink.hyperlink_match.start(),
                        expected_hyperlink.hyperlink_match.end(),
                        "\n    at {source_location}\nNo hyperlink found (hovered {:?})",
                        expected_hyperlink.hovered_char,
                    );
                }
                Err(error) => panic!("\n    at {source_location}\nhyperlink_at failed: {error}"),
            }
        }

        macro_rules! test_hyperlink {
            ($($lines:expr),+; $hyperlink_kind:ident) => { {
                use std::cmp;

                let test_lines = vec![$($lines),+];
                let (total_cells, longest_line_cells) =
                    test_lines.iter().copied()
                        .map(line_cells_count)
                        .fold((0, 0), |state, cells| (state.0 + cells, cmp::max(state.1, cells)));
                let contains_tab_char = test_lines.iter().copied()
                    .flat_map(str::chars).any(|c| c == '\t');
                let columns = if contains_tab_char {
                    vec![longest_line_cells + 1]
                } else {
                    vec![3, (longest_line_cells / 2).max(1), longest_line_cells + 1]
                };
                let source_location = format!("{}:{}", std::file!(), std::line!());
                for columns in columns {
                    test_hyperlink(columns, total_cells, test_lines.iter().copied(), HyperlinkKind::$hyperlink_kind, &source_location);
                }
            } };
        }

        mod path {
            use super::*;

            macro_rules! test_path {
                ($($lines:literal),+) => { test_hyperlink!($($lines),+; Path) };
            }

            #[test]
            fn simple() {
                test_path!("‹«/👉test/cool.rs»›");
                test_path!("‹«/test/cool👉.rs»›");
                test_path!("‹«/👉test/cool.rs»:«4»›");
                test_path!("‹«/test/cool.rs»👉:«4»›");
                test_path!("‹«/test/cool.rs»:«👉4»›");
                test_path!("‹«/👉test/cool.rs»(«4»)›");
                test_path!("‹«/test/cool.rs»👉(«4»)›");
                test_path!("‹«/test/cool.rs»(«👉4»)›");
                test_path!("‹«/test/cool.rs»(«4»👉)›");
                test_path!("‹«/👉test/cool.rs»:«4»:«2»›");
                test_path!("‹«/test/cool.rs»:«4»:«👉2»›");
                test_path!("‹«/👉test/cool.rs»(«4»,«2»)›");
                test_path!("‹«/test/cool.rs»(«4»👉,«2»)›");
                test_path!("‹«/👉test/cool.rs»:«4»:«2»›:");
                test_path!("‹«/test/cool.rs»:«4»:«👉2»›:");
                test_path!("‹«/👉test/cool.rs»(«4»,«2»)›:");
                test_path!("‹«/test/cool.rs»(«4»,«2»👉)›:");
                test_path!("‹«/👉test/cool.rs»:(«4»,«2»)›:");
                test_path!("‹«/test/cool.rs»:(«4»,«2»👉)›:");
                test_path!("‹«/👉test/cool.rs»:(«4»:«2»)›:");
                test_path!("‹«/test/cool.rs»:(«4»:«2»👉)›:");
                test_path!("‹«/test/co👉ol.rs»:«4»:«2»›:Error!");
                test_path!("‹«/test/co👉ol.rs»(«4»,«2»)›:Error!");
                test_path!("    Compiling Cool 👉(/test/Cool)");
                test_path!("    Compiling Cool (‹«/👉test/Cool»›)");
                test_path!("    Compiling Cool (/test/Cool👉)");
                test_path!("Update👉(src/cool.rs)");
                test_path!("Update(‹«src/👉cool.rs»›)");
                test_path!("Update(src/cool.rs👉)");
                test_path!("Write(‹«/👉test/Cool»›)");
                test_path!("‹«awe👉some.py»›");
                test_path!("‹«👉a»› ");
                test_path!("    ‹F👉ile \"«/awesome.py»\", line «42»›: Wat?");
                test_path!("    ‹File \"«/awe👉some.py»\", line «42»›");
                test_path!("    ‹File \"«/awesome.py»👉\", line «42»›: Wat?");
                test_path!("    ‹File \"«/awesome.py»\", line «4👉2»›");
            }

            #[test]
            fn simple_with_descriptions() {
                test_path!("‹«/👉test/cool.rs»:«4»:«2»›:例Desc例例例");
                test_path!("‹«/test/cool.rs»:«4»:«👉2»›:例Desc例例例");
                test_path!("‹«/👉test/cool.rs»(«4»,«2»)›:例Desc例例例");
                test_path!("‹«/test/cool.rs»(«4»👉,«2»)›:例Desc例例例");
                test_path!("‹«/👉test/cool.rs»:«4»:«2»›::例Desc例例例");
                test_path!("‹«/test/cool.rs»:«4»:«👉2»›::例Desc例例例");
                test_path!("‹«/👉test/cool.rs»(«4»,«2»)›::例Desc例例例");
                test_path!("‹«/test/cool.rs»(«4»,«2»👉)›::例Desc例例例");
            }

            #[test]
            fn multiple_same_line() {
                test_path!("‹«/👉test/cool.rs»› /test/cool.rs");
                test_path!("/test/cool.rs ‹«/👉test/cool.rs»›");
                test_path!(
                    "‹«🦀 multiple_👉same_line 🦀» 🚣«4» 🏛️«2»›: 🦀 multiple_same_line 🦀 🚣4 🏛️2:"
                );
                test_path!(
                    "‹«Carg👉o.toml»›\t\texperiments\t\tnotebooks\t\trust-toolchain.toml\ttooling"
                );
                test_path!(
                    "Cargo.toml\t\t‹«exper👉iments»›\t\tnotebooks\t\trust-toolchain.toml\ttooling"
                );
                test_path!(
                    "Cargo.toml\t\texperiments\t\t‹«note👉books»›\t\trust-toolchain.toml\ttooling"
                );
                test_path!(
                    "Cargo.toml\t\texperiments\t\tnotebooks\t\t‹«rust-t👉oolchain.toml»›\ttooling"
                );
                test_path!(
                    "Cargo.toml\t\texperiments\t\tnotebooks\t\trust-toolchain.toml\t‹«too👉ling»›"
                );
            }

            #[test]
            fn colons_galore() {
                test_path!("‹«/test/co👉ol.rs»:«4»›");
                test_path!("‹«/test/co👉ol.rs»:«4»›:");
                test_path!("‹«/test/co👉ol.rs»:«4»:«2»›");
                test_path!("‹«/test/co👉ol.rs»:«4»:«2»›:");
                test_path!("‹«/test/co👉ol.rs»(«1»)›");
                test_path!("‹«/test/co👉ol.rs»(«1»)›:");
                test_path!("‹«/test/co👉ol.rs»(«1»,«618»)›");
                test_path!("‹«/test/co👉ol.rs»(«1»,«618»)›:");
                test_path!("‹«/test/co👉ol.rs»::«42»›");
                test_path!("‹«/test/co👉ol.rs»::«42»›:");
                test_path!("‹«/test/co👉ol.rs»(«1»,«618»)›::");
            }

            #[test]
            fn quotes_and_brackets() {
                test_path!("\"‹«/test/co👉ol.rs»:«4»›\"");
                test_path!("'‹«/test/co👉ol.rs»:«4»›'");
                test_path!("`‹«/test/co👉ol.rs»:«4»›`");
                test_path!("[‹«/test/co👉ol.rs»:«4»›]");
                test_path!("(‹«/test/co👉ol.rs»:«4»›)");
                test_path!("{‹«/test/co👉ol.rs»:«4»›}");
                test_path!("<‹«/test/co👉ol.rs»:«4»›>");
                test_path!("[\"‹«/test/co👉ol.rs»:«4»›\"]");
                test_path!("'(‹«/test/co👉ol.rs»:«4»›)'");
                test_path!("\"‹«/test/co👉ol.rs»:«4»:«2»›\"");
                test_path!("'‹«/test/co👉ol.rs»:«4»:«2»›'");
                test_path!("`‹«/test/co👉ol.rs»:«4»:«2»›`");
                test_path!("[‹«/test/co👉ol.rs»:«4»:«2»›]");
                test_path!("(‹«/test/co👉ol.rs»:«4»:«2»›)");
                test_path!("{‹«/test/co👉ol.rs»:«4»:«2»›}");
                test_path!("<‹«/test/co👉ol.rs»:«4»:«2»›>");
                test_path!("[\"‹«/test/co👉ol.rs»:«4»:«2»›\"]");
                test_path!("\"‹«/test/co👉ol.rs»(«4»)›\"");
                test_path!("'‹«/test/co👉ol.rs»(«4»)›'");
                test_path!("`‹«/test/co👉ol.rs»(«4»)›`");
                test_path!("[‹«/test/co👉ol.rs»(«4»)›]");
                test_path!("(‹«/test/co👉ol.rs»(«4»)›)");
                test_path!("{‹«/test/co👉ol.rs»(«4»)›}");
                test_path!("<‹«/test/co👉ol.rs»(«4»)›>");
                test_path!("[\"‹«/test/co👉ol.rs»(«4»)›\"]");
                test_path!("\"‹«/test/co👉ol.rs»(«4»,«2»)›\"");
                test_path!("'‹«/test/co👉ol.rs»(«4»,«2»)›'");
                test_path!("`‹«/test/co👉ol.rs»(«4»,«2»)›`");
                test_path!("[‹«/test/co👉ol.rs»(«4»,«2»)›]");
                test_path!("(‹«/test/co👉ol.rs»(«4»,«2»)›)");
                test_path!("{‹«/test/co👉ol.rs»(«4»,«2»)›}");
                test_path!("<‹«/test/co👉ol.rs»(«4»,«2»)›>");
                test_path!("[\"‹«/test/co👉ol.rs»(«4»,«2»)›\"]");
                test_path!("([‹«/test/co👉ol.rs»:«4»›] was here...)");
                test_path!("[Here's <‹«/test/co👉ol.rs»:«4»›>]");
                test_path!("('‹«/test/co👉ol.rs»:«4»›' was here...)");
                test_path!("[Here's `‹«/test/co👉ol.rs»:«4»›`]");
            }

            #[test]
            fn trailing_punctuation() {
                test_path!("‹«/test/co👉ol.rs»›:,..");
                test_path!("/test/cool.rs:,👉..");
                test_path!("‹«/test/co👉ol.rs»:«4»›:,");
                test_path!("/test/cool.rs:4:👉,");
                test_path!("[\"‹«/test/co👉ol.rs»:«4»›\"]:,");
                test_path!("'(‹«/test/co👉ol.rs»:«4»›),,'...");
                test_path!("('‹«/test/co👉ol.rs»:«4»›'::: was here...)");
                test_path!("[Here's <‹«/test/co👉ol.rs»:«4»›>]::: ");
            }

            #[test]
            fn word_wide_chars() {
                test_path!("‹«/👉例/cool.rs»›");
                test_path!("‹«/例👈/cool.rs»›");
                test_path!("‹«/例/cool.rs»:«👉4»›");
                test_path!("‹«/例/cool.rs»:«4»:«👉2»›");
                test_path!("    Compiling Cool (‹«/👉例/Cool»›)");
                test_path!("    Compiling Cool (‹«/例👈/Cool»›)");
                test_path!("    Compiling Cool (‹«/👉例/Cool Spaces»›)");
                test_path!("    Compiling Cool (‹«/例👈/Cool Spaces»›)");
                test_path!("    Compiling Cool (‹«/👉例/Cool Spaces»:«4»:«2»›)");
                test_path!("    Compiling Cool (‹«/例👈/Cool Spaces»(«4»,«2»)›)");
                test_path!("    --> ‹«/👉例/Cool Spaces»›");
                test_path!("    ::: ‹«/例👈/Cool Spaces»›");
                test_path!("    --> ‹«/👉例/Cool Spaces»:«4»:«2»›");
                test_path!("    ::: ‹«/例👈/Cool Spaces»(«4»,«2»)›");
                test_path!("    panicked at ‹«/👉例/Cool Spaces»:«4»:«2»›:");
                test_path!("    panicked at ‹«/例👈/Cool Spaces»(«4»,«2»)›:");
                test_path!("    at ‹«/👉例/Cool Spaces»:«4»:«2»›");
                test_path!("    at ‹«/例👈/Cool Spaces»(«4»,«2»)›");
                test_path!("‹«👉例wesome.py»›");
                test_path!("‹«例👈wesome.py»›");
                test_path!("    ‹File \"«/👉例wesome.py»\", line «42»›: Wat?");
                test_path!("    ‹File \"«/例👈wesome.py»\", line «42»›: Wat?");
            }

            #[test]
            fn non_word_wide_chars() {
                test_path!("    ‹File \"«/awe👉some.🔥»\", line «42»›: Wat?");
                test_path!("    ‹File \"«/awesome👉.🔥»\", line «42»›: Wat?");
                test_path!("    ‹File \"«/awesome.👉🔥»\", line «42»›: Wat?");
                test_path!("    ‹File \"«/awesome.🔥👈»\", line «42»›: Wat?");
            }

            /// These likely rise to the level of being worth fixing.
            mod issues {
                use super::*;

                #[test]
                // <https://github.com/alacritty/alacritty/issues/8586>
                fn issue_alacritty_8586() {
                    test_path!("‹«/👉例/cool.rs»›");
                    test_path!("‹«/例👈/cool.rs»›");
                    test_path!("‹«/例/cool.rs»:«👉4»›");
                    test_path!("‹«/例/cool.rs»:«4»:«👉2»›");
                    test_path!("    Compiling Cool (‹«/👉例/Cool»›)");
                    test_path!("    Compiling Cool (‹«/例👈/Cool»›)");
                    test_path!("‹«👉例wesome.py»›");
                    test_path!("‹«例👈wesome.py»›");
                    test_path!("    ‹File \"«/👉例wesome.py»\", line «42»›: Wat?");
                    test_path!("    ‹File \"«/例👈wesome.py»\", line «42»›: Wat?");
                }

                #[test]
                // <https://github.com/zed-industries/zed/issues/12338>
                fn issue_12338_regex() {
                    test_path!(".rw-r--r--     0     staff 05-27 14:03 ‹«'test file 👉1.txt'»›");
                    test_path!(".rw-r--r--     0     staff 05-27 14:03 ‹«👉'test file 1.txt'»›");
                }

                #[test]
                // <https://github.com/zed-industries/zed/issues/12338>
                fn issue_12338() {
                    test_path!(".rw-r--r--     0     staff 05-27 14:03 ‹«test👉、2.txt»›");
                    test_path!(".rw-r--r--     0     staff 05-27 14:03 ‹«test、👈2.txt»›");
                    test_path!(".rw-r--r--     0     staff 05-27 14:03 ‹«test👉。3.txt»›");
                    test_path!(".rw-r--r--     0     staff 05-27 14:03 ‹«test。👈3.txt»›");
                    test_path!("‹«/👉🏃/🦀.rs»›");
                    test_path!("‹«/🏃👈/🦀.rs»›");
                    test_path!("‹«/🏃/👉🦀.rs»:«4»›");
                    test_path!("‹«/🏃/🦀👈.rs»:«4»:«2»›");
                    test_path!("    Compiling Cool (‹«/👉🏃/Cool»›)");
                    test_path!("    Compiling Cool (‹«/🏃👈/Cool»›)");
                    test_path!("‹«👉🏃wesome.py»›");
                    test_path!("‹«🏃👈wesome.py»›");
                    test_path!("    ‹File \"«/👉🏃wesome.py»\", line «42»›: Wat?");
                    test_path!("    ‹File \"«/🏃👈wesome.py»\", line «42»›: Wat?");
                    test_path!("‹«/awe👉some.🔥»› is some good Mojo!");
                    test_path!("‹«/awesome👉.🔥»› is some good Mojo!");
                    test_path!("‹«/awesome.👉🔥»› is some good Mojo!");
                    test_path!("‹«/awesome.🔥👈»› is some good Mojo!");
                    test_path!("    ‹File \"«/👉🏃wesome.🔥»\", line «42»›: Wat?");
                    test_path!("    ‹File \"«/🏃👈wesome.🔥»\", line «42»›: Wat?");
                }

                #[test]
                // <https://github.com/zed-industries/zed/issues/40202>
                fn issue_40202() {
                    test_path!("[‹«lib/blitz_apex_👉server/stats/aggregate_rank_stats.ex»:«35»›: BlitzApexServer.Stats.AggregateRankStats.update/2]
                1 #=> 1");
                }

                #[test]
                // <https://github.com/zed-industries/zed/issues/28194>
                fn issue_28194() {
                    test_path!(
                        "‹«test/c👉ontrollers/template_items_controller_test.rb»:«20»›:in 'block (2 levels) in <class:TemplateItemsControllerTest>'"
                    );
                }

                #[test]
                // <https://github.com/zed-industries/zed/issues/50531>
                fn issue_50531() {
                    test_path!("0: ‹«foo/👉bar.txt»›");
                    test_path!("0: ‹«👉foo/bar.txt»›");
                    test_path!("42: ‹«👉foo/bar.txt»›");
                    test_path!("1: ‹«/👉test/cool.rs»›");
                    test_path!("1: ‹«/👉test/cool.rs»:«4»:«2»›");
                }

                #[test]
                // <https://github.com/zed-industries/zed/issues/46795>
                fn issue_46795() {
                    test_path!("─‹«/👉test/cool.rs»:«4»:«2»›");
                    test_path!("┤‹«/👉test/cool.rs»:«4»:«2»›");
                    test_path!("╿‹«/👉test/cool.rs»:«4»:«2»›");
                    test_path!("└──‹«/👉test/cool.rs»:«4»:«2»›");
                    test_path!("├─[‹«/👉test/cool.rs»:«4»:«2»›]");
                    test_path!("─[‹«/👉test/cool.rs»:«4»:«2»›]");
                    test_path!("┬‹«/👉test/cool.rs»:«4»:«2»›┬");
                }
            }

            /// Minor issues arguably not important enough to fix/workaround...
            mod nits {
                use super::*;

                #[test]
                fn alacritty_bugs_with_two_columns() {
                    test_path!("‹«/👉test/cool.rs»(«4»)›");
                    test_path!("‹«/test/cool.rs»(«👉4»)›");
                    test_path!("‹«/test/cool.rs»(«4»,«👉2»)›");
                    test_path!("‹«awe👉some.py»›");
                }

                #[test]
                // Filenames with balanced parentheses are preserved as a single path.
                // Unbalanced leading `(` (e.g. `Update(.claude/SKILL.md)`) is stripped.
                fn parens_in_filename() {
                    test_path!("‹«docker-compose.prod(👉copy).yml»›");
                }
            }

            mod windows {
                use super::*;

                #[test]
                fn default_prompts() {
                    test_path!(r#"‹«C:\Users\someone\👉test»›>"#);
                    test_path!(r#"C:\Users\someone\test👉>"#);
                    test_path!(r#"PS ‹«C:\Users\someone\👉test\cool.rs»›>"#);
                    test_path!(r#"PS C:\Users\someone\test\cool.rs👉>"#);
                }

                #[test]
                fn unc() {
                    test_path!(r#"‹«\\server\share\👉test\cool.rs»›"#);
                    test_path!(r#"‹«\\server\share\test\cool👉.rs»›"#);
                }

                mod issues {
                    use super::*;

                    #[test]
                    fn issue_verbatim() {
                        test_path!(r#"‹«\\?\C:\👉test\cool.rs»›"#);
                        test_path!(r#"‹«\\?\C:\test\cool👉.rs»›"#);
                    }

                    #[test]
                    fn issue_verbatim_unc() {
                        test_path!(r#"‹«\\?\UNC\server\share\👉test\cool.rs»›"#);
                        test_path!(r#"‹«\\?\UNC\server\share\test\cool👉.rs»›"#);
                    }
                }
            }
        }

        mod file_iri {
            use super::*;

            macro_rules! test_file_iri {
                ($file_iri:literal) => { { test_hyperlink!(concat!("‹«👉", $file_iri, "»›"); FileIri) } };
            }

            #[cfg(not(target_os = "windows"))]
            #[test]
            fn absolute_file_iri() {
                test_file_iri!("file:///test/cool/index.rs");
                test_file_iri!("file:///test/cool/");
            }

            #[cfg(not(target_os = "windows"))]
            mod issues {
                use super::*;

                #[test]
                fn issue_file_iri_with_percent_encoded_characters() {
                    test_file_iri!("file:///test/%E1%BF%AC%CF%8C%CE%B4%CE%BF%CF%82/");
                    test_file_iri!("file:///te%20st/co%20ol/index.rs");
                    test_file_iri!("file:///te%20st/co%20ol/");
                }
            }

            #[cfg(target_os = "windows")]
            mod windows {
                use super::*;

                mod issues {
                    use super::*;

                    #[test]
                    #[should_panic(
                        expected = "Failed to interpret file IRI `file:/test/cool/index.rs` as a path"
                    )]
                    fn issue_relative_file_iri() {
                        test_file_iri!("file:/test/cool/index.rs");
                        test_file_iri!("file:/test/cool/");
                    }

                    #[test]
                    fn issue_39189() {
                        test_file_iri!("file:///C:/test/cool/index.rs");
                        test_file_iri!("file:///C:/test/cool/");
                    }

                    #[test]
                    fn issue_file_iri_with_percent_encoded_characters() {
                        test_file_iri!("file:///C:/test/%E1%BF%AC%CF%8C%CE%B4%CE%BF%CF%82/");
                        test_file_iri!("file:///C:/te%20st/co%20ol/index.rs");
                        test_file_iri!("file:///C:/te%20st/co%20ol/");
                    }
                }
            }
        }

        mod iri {
            use super::*;

            macro_rules! test_iri {
                ($iri:literal) => { { test_hyperlink!(concat!("‹«👉", $iri, "»›"); Iri) } };
            }

            #[test]
            fn simple() {
                test_iri!("ipfs://test/cool.ipfs");
                test_iri!("ipns://test/cool.ipns");
                test_iri!("magnet://test/cool.git");
                test_iri!("mailto:someone@somewhere.here");
                test_iri!("gemini://somewhere.here");
                test_iri!("gopher://somewhere.here");
                test_iri!("http://test/cool/index.html");
                test_iri!("http://10.10.10.10:1111/cool.html");
                test_iri!("http://test/cool/index.html?amazing=1");
                test_iri!("http://test/cool/index.html#right%20here");
                test_iri!("http://test/cool/index.html?amazing=1#right%20here");
                test_iri!("https://test/cool/index.html");
                test_iri!("https://10.10.10.10:1111/cool.html");
                test_iri!("https://test/cool/index.html?amazing=1");
                test_iri!("https://test/cool/index.html#right%20here");
                test_iri!("https://test/cool/index.html?amazing=1#right%20here");
                test_iri!("news://test/cool.news");
                test_iri!("git://test/cool.git");
                test_iri!("ssh://user@somewhere.over.here:12345/test/cool.git");
                test_iri!("ftp://test/cool.ftp");
            }

            #[test]
            fn wide_chars() {
                test_iri!("ipfs://例🏃🦀/cool.ipfs");
                test_iri!("ipns://例🏃🦀/cool.ipns");
                test_iri!("magnet://例🏃🦀/cool.git");
                test_iri!("mailto:someone@somewhere.here");
                test_iri!("gemini://somewhere.here");
                test_iri!("gopher://somewhere.here");
                test_iri!("http://例🏃🦀/cool/index.html");
                test_iri!("http://10.10.10.10:1111/cool.html");
                test_iri!("http://例🏃🦀/cool/index.html?amazing=1");
                test_iri!("http://例🏃🦀/cool/index.html#right%20here");
                test_iri!("http://例🏃🦀/cool/index.html?amazing=1#right%20here");
                test_iri!("https://例🏃🦀/cool/index.html");
                test_iri!("https://10.10.10.10:1111/cool.html");
                test_iri!("https://例🏃🦀/cool/index.html?amazing=1");
                test_iri!("https://例🏃🦀/cool/index.html#right%20here");
                test_iri!("https://例🏃🦀/cool/index.html?amazing=1#right%20here");
                test_iri!("news://例🏃🦀/cool.news");
                test_iri!("git://例/cool.git");
                test_iri!("ssh://user@somewhere.over.here:12345/例🏃🦀/cool.git");
                test_iri!("ftp://例🏃🦀/cool.ftp");
            }

            #[test]
            fn iris() {
                test_iri!("https://en.wiktionary.org/wiki/Ῥόδος");
                test_iri!("https://en.wiktionary.org/wiki/%E1%BF%AC%CF%8C%CE%B4%CE%BF%CF%82");
            }

            /// Alacritty misidentified this as a path instead of an IRI
            /// (hence the name); Ghostty's own `hyperlink_uri_at_point`/
            /// `URL_REGEX` fallback ordering doesn't reproduce that bug,
            /// so unlike the original this now asserts the (more correct)
            /// IRI classification rather than `#[should_panic]`-pinning it.
            #[test]
            fn file_is_a_path() {
                test_iri!("file://test/cool/index.rs");
            }
        }
    }
}
