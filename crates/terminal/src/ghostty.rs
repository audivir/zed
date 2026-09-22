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

/// Maximum bytes of decoded Kitty image data retained per terminal, the Ghostty default.
const KITTY_IMAGE_STORAGE_LIMIT_BYTES: u64 = 320 * 1024 * 1024;

/// Size of each recycled PTY read buffer.
const PTY_READ_BUFFER_BYTES: usize = 64 * 1024;
/// Maximum bytes the parser thread batches into one write before returning effects.
const MAX_PTY_PARSE_BATCH_BYTES: usize = 512 * 1024;
/// Capacity of the reader-to-parser queue, so a slow parser applies backpressure to reads.
const MAX_QUEUED_PTY_OUTPUT_BUFFERS: usize = 512;
/// Capacity of the channel recycling buffers back to the reader thread.
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
    /// OSC 52 or iTerm2 OSC 1337 clipboard write, normalized to plain text.
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

/// Wrapper around a `libghostty-vt` terminal that parses PTY output and drives rendering.
///
/// Shared as `Arc<parking_lot::Mutex<GhosttyTerminal>>` between the PTY parser thread and the
/// thread owning the terminal entity.
pub(super) struct GhosttyTerminal {
    // boxed because libghostty-vt stores callback userdata as a raw pointer into it.
    terminal: Box<libghostty_vt::Terminal<'static, 'static>>,
    render_state: libghostty_vt::RenderState<'static>,
    rows: RowIterator<'static>,
    cells: CellIterator<'static>,
    placements: PlacementIterator<'static>,
    effects: Arc<ParkingMutex<Vec<GhosttyEffect>>>,
    /// Active selection in Zed point space, mirroring the one installed in Ghostty.
    current_selection: Option<crate::SelectionRange>,
    /// Vi-mode cursor position, or `None` when vi mode is disabled.
    current_vi_cursor: Option<crate::Point>,
    /// Buffer text and point map for search, dropped whenever the grid changes.
    search_text: Option<(String, Vec<crate::Point>)>,
    /// Formatted selection text, dropped whenever the grid or the selection changes.
    selection_text: Option<Option<String>>,
    /// Decoded RGBA pixels keyed by image ID, with the image generation they were decoded from.
    decoded_images: collections::HashMap<u32, (u64, Arc<[u8]>)>,
}

// SAFETY: libghostty-vt objects are `!Send` only because they hold raw pointers and callback
// state, and the library allows moving them between threads as long as access is never
// concurrent. Every FFI method takes `&mut self`, and the value is only reachable through a
// mutex, which serializes access from the parser thread and the entity thread.
unsafe impl Send for GhosttyTerminal {}

/// Scrollback bytes budgeted per line of `terminal.max_scroll_history_lines`.
///
/// Ghostty limits scrollback in bytes and uses about 9 bytes per cell, so this keeps at least the
/// configured number of lines for terminals up to 256 columns wide.
const SCROLLBACK_BYTES_PER_LINE: usize = 256 * 9;

impl GhosttyTerminal {
    pub(super) fn new(cols: u16, rows: u16, max_scrollback_lines: usize) -> Result<Self> {
        anyhow::ensure!(cols > 0, "terminal columns must be greater than zero");
        anyhow::ensure!(rows > 0, "terminal rows must be greater than zero");
        let effects = Arc::new(ParkingMutex::new(Vec::new()));
        let mut terminal = Box::new(libghostty_vt::Terminal::new(TerminalOptions {
            cols,
            rows,
            max_scrollback: max_scrollback_lines.saturating_mul(SCROLLBACK_BYTES_PER_LINE),
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

        // handles OSC 52 writes. libghostty-vt always ignores clipboard reads by design.
        // Multi-MIME writes are flattened to the text of their first representation.
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
            search_text: None,
            selection_text: None,
            decoded_images: collections::HashMap::default(),
        })
    }

    pub(super) fn write(&mut self, bytes: &[u8]) {
        self.search_text = None;
        self.selection_text = None;
        self.terminal.vt_write(bytes);
    }

    pub(super) fn clear(&mut self) {
        self.search_text = None;
        self.selection_text = None;
        self.terminal.vt_write(b"\x1b[3J\x1b[H\x1b[2J");
    }

    pub(super) fn resize(&mut self, bounds: TerminalBounds) -> Result<()> {
        self.search_text = None;
        self.selection_text = None;
        self.terminal.resize(
            bounds.num_columns().max(1) as u16,
            bounds.num_lines().max(1) as u16,
            // round instead of truncating: libghostty-vt recovers the cell size from the total
            // height, and truncation undercounts Kitty image rows by one per ~20px of height.
            f32::from(bounds.cell_width()).max(1.0).round() as u32,
            f32::from(bounds.line_height()).max(1.0).round() as u32,
        )?;
        Ok(())
    }

    pub(super) fn scroll_viewport(&mut self, scroll: ScrollViewport) {
        self.terminal.scroll_viewport(scroll);
    }

    /// Returns the number of viewport rows.
    pub(super) fn rows(&self) -> Result<u16> {
        Ok(self.terminal.rows()?)
    }

    /// Returns the number of rows used by scrollback plus the screen down to the cursor or the
    /// last non-blank row below it.
    ///
    /// Reads the live grid, since callers size embedded terminals between syncs.
    pub(super) fn used_lines(&self) -> Result<usize> {
        let total_rows = self.terminal.total_rows()?;
        if self.modes()?.contains(crate::Modes::ALT_SCREEN) {
            return Ok(total_rows);
        }
        let grid = ViGrid::new(&self.terminal)?;
        let screen_lines = i64::from(self.terminal.rows()?);
        let cursor_line = i64::from(self.terminal.cursor_y()?);
        let mut last_used_line = cursor_line;
        for line in (cursor_line + 1..screen_lines).rev() {
            if !grid.is_row_clear(grid.line_base + line)? {
                last_used_line = line;
                break;
            }
        }
        let history_rows = total_rows.saturating_sub(screen_lines as usize);
        Ok(history_rows + last_used_line as usize + 1)
    }

    /// Returns the total number of rows including scrollback.
    pub(super) fn total_lines(&self) -> Result<usize> {
        Ok(self.terminal.total_rows()?)
    }

    pub(super) fn take_effects(&mut self) -> Vec<GhosttyEffect> {
        self.effects.lock().drain(..).collect()
    }

    /// Returns the scroll position as a `display_offset`, where `line + display_offset` is the
    /// viewport row.
    ///
    /// Ghostty uses the opposite polarity: `Scrollbar.offset` is `0` at the top of history,
    /// while `display_offset` is `0` at the live bottom and grows when scrolling up. Always
    /// convert through this function.
    pub(super) fn alacritty_style_display_offset(&self) -> Result<usize> {
        let scrollbar = self.terminal.scrollbar()?;
        Ok(
            (scrollbar.total.saturating_sub(scrollbar.len)).saturating_sub(scrollbar.offset)
                as usize,
        )
    }

    pub(super) fn content_metadata(
        &mut self,
        display_offset: usize,
        vi_cursor: Option<crate::Point>,
    ) -> Result<GhosttyContentMetadata> {
        let scrollbar = self.terminal.scrollbar()?;

        // the vi cursor replaces the terminal cursor and is shown regardless of DECTCEM.
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

        // hide a cursor scrolled out of view instead of faking a (0, 0) position.
        let cursor = crate::Cursor {
            shape: if cursor_visible && cursor_viewport.is_some() {
                ghostty_cursor_shape(cursor_style)
            } else {
                CursorShape::Hidden
            },
            point: cursor_viewport
                .map(|cursor| {
                    crate::Point::new(cursor.y as i32 - display_offset as i32, cursor.x as usize)
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

    /// Builds the viewport cell grid, mode flags, and display offset.
    ///
    /// Cell points follow the crate convention where `line + display_offset` is the viewport row.
    pub(super) fn build_content(
        &mut self,
    ) -> Result<(Vec<crate::IndexedCell>, crate::Modes, usize)> {
        let display_offset = self.alacritty_style_display_offset()?;
        let first_line = -(display_offset as i32);
        let snapshot = self.render_state.update(&self.terminal)?;
        let mut rows = self.rows.update(&snapshot)?;
        let total_cells = usize::from(self.terminal.rows().unwrap_or(24))
            * usize::from(self.terminal.cols().unwrap_or(80));
        let mut cells = Vec::with_capacity(total_cells);
        let mut line = first_line;
        // consecutive cells of one link share an allocation.
        let mut previous_hyperlink: Option<crate::Hyperlink> = None;
        // reused across cells so typical grapheme clusters never allocate.
        let mut grapheme_buf = [' '; 32];

        while let Some(row) = rows.next() {
            // the row flag may report false positives, but skips per-cell checks on plain rows.
            let row_has_hyperlink = row.raw_row()?.has_hyperlink()?;
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

                let point = crate::Point::new(line, col_index);
                let hyperlink = if row_has_hyperlink && raw_cell.has_hyperlink()? {
                    hyperlink_uri_at_point(&self.terminal, display_offset, point)?.map(|uri| {
                        match &previous_hyperlink {
                            Some(previous) if previous.uri() == uri => previous.clone(),
                            _ => crate::Hyperlink::new(None, uri.into()),
                        }
                    })
                } else {
                    None
                };
                previous_hyperlink.clone_from(&hyperlink);

                cells.push(crate::IndexedCell {
                    point,
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
                        hyperlink,
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
            line += 1;
        }

        let modes = self.modes()?;
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
        // senders use legacy (47) or save/restore (1047/1049) alternate screen modes.
        if self.terminal.mode(Mode::ALT_SCREEN_LEGACY)?
            || self.terminal.mode(Mode::ALT_SCREEN)?
            || self.terminal.mode(Mode::ALT_SCREEN_SAVE)?
        {
            modes.insert(crate::Modes::ALT_SCREEN);
        }
        Ok(modes)
    }

    // selections use the screen space of Ghostty rather than its viewport space, since a
    // selection extended while scrolling has endpoints outside the current viewport.

    /// Returns the screen row of `crate::Point` line zero, the top of the active screen.
    fn screen_line_base(&self) -> Result<i64> {
        Ok(self.terminal.total_rows()? as i64 - i64::from(self.terminal.rows()?))
    }

    /// Installs `range` as the active selection, or clears it if `None`.
    pub(super) fn set_selection(&mut self, range: Option<crate::SelectionRange>) -> Result<()> {
        let line_base = self.screen_line_base()?;
        match range {
            Some(range) => {
                let start = self
                    .terminal
                    .grid_ref(ghostty_screen_point(line_base, range.start))?;
                let end = self
                    .terminal
                    .grid_ref(ghostty_screen_point(line_base, range.end))?;
                let selection =
                    libghostty_vt::selection::Selection::new(start, end, range.is_block);
                self.terminal.set_selection(Some(&selection))?;
            }
            None => {
                self.terminal.set_selection(None)?;
            }
        }
        self.current_selection = range;
        self.selection_text = None;
        Ok(())
    }

    /// Moves the end of the active selection to `point`.
    ///
    /// Returns `false` if there is no active selection.
    pub(super) fn update_selection(&mut self, point: crate::Point) -> Result<bool> {
        let Some(mut range) = self.current_selection else {
            return Ok(false);
        };
        range.end = point;
        self.set_selection(Some(range))?;
        Ok(true)
    }

    /// Returns the active selection in Zed point space.
    pub(super) fn selection_range(&self) -> Option<crate::SelectionRange> {
        self.current_selection
    }

    /// Formats the active selection as plain text, joining soft wraps and trimming trailing
    /// whitespace.
    ///
    /// The result is cached until the grid or the selection changes, since callers read it on
    /// every sync.
    pub(super) fn selection_text(&mut self) -> Result<Option<String>> {
        if let Some(text) = &self.selection_text {
            return Ok(text.clone());
        }
        let options = libghostty_vt::selection::FormatOptions::new()
            .with_emit_format(libghostty_vt::fmt::Format::Plain)
            .with_unwrap(true)
            .with_trim(true);
        let text = self
            .terminal
            .format_selection_alloc(None, options)?
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned());
        self.selection_text = Some(text.clone());
        Ok(text)
    }

    /// Selects the word at `point` using the word boundaries of Ghostty.
    pub(super) fn select_word_at(
        &mut self,
        point: crate::Point,
    ) -> Result<Option<crate::SelectionRange>> {
        let line_base = self.screen_line_base()?;
        let grid_ref = self
            .terminal
            .grid_ref(ghostty_screen_point(line_base, point))?;
        let Some(selection) = self
            .terminal
            .select_word(libghostty_vt::selection::SelectWordOptions::new(grid_ref))?
        else {
            return Ok(None);
        };
        let range = selection_range_from_ghostty(&self.terminal, line_base, &selection)?;
        self.terminal.set_selection(Some(&selection))?;
        self.current_selection = Some(range);
        self.selection_text = None;
        Ok(Some(range))
    }

    /// Selects the line at `point`, joining soft-wrapped rows.
    pub(super) fn select_line_at(
        &mut self,
        point: crate::Point,
    ) -> Result<Option<crate::SelectionRange>> {
        let line_base = self.screen_line_base()?;
        let grid_ref = self
            .terminal
            .grid_ref(ghostty_screen_point(line_base, point))?;
        let Some(selection) = self
            .terminal
            .select_line(libghostty_vt::selection::SelectLineOptions::new(grid_ref))?
        else {
            return Ok(None);
        };
        let range = selection_range_from_ghostty(&self.terminal, line_base, &selection)?;
        self.terminal.set_selection(Some(&selection))?;
        self.current_selection = Some(range);
        self.selection_text = None;
        Ok(Some(range))
    }

    /// Selects all selectable content.
    pub(super) fn select_all(&mut self) -> Result<Option<crate::SelectionRange>> {
        let line_base = self.screen_line_base()?;
        let Some(selection) = self.terminal.select_all()? else {
            return Ok(None);
        };
        let range = selection_range_from_ghostty(&self.terminal, line_base, &selection)?;
        self.terminal.set_selection(Some(&selection))?;
        self.current_selection = Some(range);
        self.selection_text = None;
        Ok(Some(range))
    }

    /// Extends a word selection from `anchor` to the word under `drag_point`.
    ///
    /// Snaps in both directions with `select_word_between`, so dragging across whitespace does
    /// not collapse the selection. Returns `None` if either endpoint has no nearby word.
    pub(super) fn select_word_range(
        &mut self,
        anchor: crate::Point,
        drag_point: crate::Point,
    ) -> Result<Option<crate::SelectionRange>> {
        let line_base = self.screen_line_base()?;
        let anchor_ref = self
            .terminal
            .grid_ref(ghostty_screen_point(line_base, anchor))?;
        let drag_ref = self
            .terminal
            .grid_ref(ghostty_screen_point(line_base, drag_point))?;

        let word_toward_drag = self.terminal.select_word_between(
            libghostty_vt::selection::SelectWordBetweenOptions::new(
                anchor_ref.clone(),
                drag_ref.clone(),
            ),
        )?;
        let word_toward_anchor = self.terminal.select_word_between(
            libghostty_vt::selection::SelectWordBetweenOptions::new(drag_ref, anchor_ref),
        )?;
        let (Some(word_toward_drag), Some(word_toward_anchor)) =
            (word_toward_drag, word_toward_anchor)
        else {
            return Ok(None);
        };
        // take the overall bounds of both words, since the drag can run in either direction.
        let word_toward_drag_range =
            selection_range_from_ghostty(&self.terminal, line_base, &word_toward_drag)?;
        let word_toward_anchor_range =
            selection_range_from_ghostty(&self.terminal, line_base, &word_toward_anchor)?;
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

    /// Extends a line selection from `anchor` to the lines containing `drag_point`.
    ///
    /// Returns `None` if either endpoint has no selectable line.
    pub(super) fn select_line_range(
        &mut self,
        anchor: crate::Point,
        drag_point: crate::Point,
    ) -> Result<Option<crate::SelectionRange>> {
        let line_base = self.screen_line_base()?;
        let anchor_ref = self
            .terminal
            .grid_ref(ghostty_screen_point(line_base, anchor))?;
        let drag_ref = self
            .terminal
            .grid_ref(ghostty_screen_point(line_base, drag_point))?;

        let anchor_line = self
            .terminal
            .select_line(libghostty_vt::selection::SelectLineOptions::new(anchor_ref))?;
        let drag_line = self
            .terminal
            .select_line(libghostty_vt::selection::SelectLineOptions::new(drag_ref))?;
        let (Some(anchor_line), Some(drag_line)) = (anchor_line, drag_line) else {
            return Ok(None);
        };
        let anchor_range = selection_range_from_ghostty(&self.terminal, line_base, &anchor_line)?;
        let drag_range = selection_range_from_ghostty(&self.terminal, line_base, &drag_line)?;
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

    /// Finds the hyperlink or path at `point`.
    ///
    /// Checks OSC 8 hyperlinks first, then bare URLs matching `url_regex`, then
    /// `path_hyperlink_regexes`, where the first regex matching the line ends the search. Returns
    /// `(text, is_url, range)`.
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

        // converts a byte range in `line` into an inclusive `Range` via the char-indexed `points`.
        let point_range_for_bytes = |byte_range: std::ops::Range<usize>| -> Option<crate::Range> {
            let start_char = line[..byte_range.start].chars().count();
            let end_char = line[..byte_range.end].chars().count();
            if start_char >= points.len() {
                return None;
            }
            let end_char = end_char
                .saturating_sub(1)
                .max(start_char)
                .min(points.len() - 1);
            Some(crate::Range::new(points[start_char], points[end_char]))
        };

        if let Some(found) = url_regex
            .find_iter(&line)
            .find(|m| m.start() <= hovered_byte_offset && hovered_byte_offset < m.end())
        {
            // trimming removes single-byte ASCII punctuation, so byte lengths map directly.
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

    // libghostty-vt has no grid-wide regex search, so search extracts the whole buffer as text with
    // a point per character. This reads all scrollback on every call and can be slow for very
    // large histories.

    /// Returns all matches of `regex` in the scrollback and active screen.
    ///
    /// Soft-wrapped rows form one logical line, and hard line breaks become `\n`, so anchored
    /// patterns match per hard line.
    pub(super) fn search_matches(&mut self, regex: &regex::Regex) -> Result<Vec<crate::Range>> {
        if self.search_text.is_none() {
            self.search_text = Some(self.buffer_text_with_point_map()?);
        }
        let Some((text, points)) = self.search_text.as_ref() else {
            return Ok(Vec::new());
        };

        let mut matches = Vec::new();
        // matches arrive in order, so char offsets are counted incrementally.
        let mut counted_bytes = 0;
        let mut counted_chars = 0;
        let mut char_offset = |byte_offset: usize| {
            counted_chars += text[counted_bytes..byte_offset].chars().count();
            counted_bytes = byte_offset;
            counted_chars
        };
        for found in regex.find_iter(text) {
            if found.start() == found.end() {
                continue;
            }
            let start_char = char_offset(found.start());
            let end_char = char_offset(found.end());
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

    /// Extracts the scrollback and active screen as text with a `Point` for every `char`.
    fn buffer_text_with_point_map(&self) -> Result<(String, Vec<crate::Point>)> {
        let total_rows = self.terminal.total_rows()?;
        let viewport_rows = usize::from(self.terminal.rows()?);
        let cols = usize::from(self.terminal.cols()?);
        let line_base = total_rows as i32 - viewport_rows as i32;
        // one formatter call replaces a lookup per cell, which costs seconds on large scrollback.
        let formatted = self.format_screen_rows(total_rows, cols)?;

        let mut text = String::new();
        let mut points: Vec<crate::Point> = Vec::new();
        let mut grapheme_buf = [' '; 32];

        for absolute_row in 0..total_rows {
            let line = absolute_row as i32 - line_base;
            let row_ref = self.screen_grid_ref(0, absolute_row)?;
            let row_wrapped = row_ref.row()?.is_wrapped()?;
            let row_start = points.len();

            match formatted.get(absolute_row) {
                // printable ASCII cells are single-width without clusters, so characters map to
                // consecutive columns.
                Some(row_text)
                    if row_text.len() <= cols
                        && row_text.bytes().all(|byte| (b' '..=b'~').contains(&byte)) =>
                {
                    for (column, character) in row_text.chars().enumerate() {
                        text.push(character);
                        points.push(crate::Point::new(line, column));
                    }
                    if row_wrapped {
                        for column in row_text.len()..cols {
                            text.push(' ');
                            points.push(crate::Point::new(line, column));
                        }
                    }
                }
                _ => self.append_row_cells(
                    absolute_row,
                    line,
                    cols,
                    &mut grapheme_buf,
                    &mut text,
                    &mut points,
                )?,
            }

            if !row_wrapped {
                // trim trailing blanks of this row, so `$` matches right after real content.
                // Wrapped rows are full, so only the current row can have trailing blanks.
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

    fn screen_grid_ref(&self, column: usize, absolute_row: usize) -> Result<screen::GridRef<'_>> {
        Ok(self
            .terminal
            .grid_ref(libghostty_vt::terminal::Point::Screen(
                libghostty_vt::terminal::PointCoordinate {
                    x: column as u16,
                    y: absolute_row as u32,
                },
            ))?)
    }

    /// Returns the plain text of every screen row, one entry per row.
    ///
    /// Trailing blank rows may be missing.
    fn format_screen_rows(&self, total_rows: usize, cols: usize) -> Result<Vec<String>> {
        if total_rows == 0 || cols == 0 {
            return Ok(Vec::new());
        }
        let first = self.screen_grid_ref(0, 0)?;
        let last = self.screen_grid_ref(cols - 1, total_rows - 1)?;
        let selection = libghostty_vt::selection::Selection::new(first, last, false);
        let options = libghostty_vt::selection::FormatOptions::new()
            .with_emit_format(libghostty_vt::fmt::Format::Plain)
            .with_unwrap(false)
            .with_trim(false)
            .with_selection(&selection);
        let Some(bytes) = self.terminal.format_selection_alloc(None, options)? else {
            return Ok(Vec::new());
        };
        let text = String::from_utf8_lossy(&bytes);
        Ok(text.split('\n').map(str::to_string).collect())
    }

    /// Appends the cells of one row with exact column positions, one lookup per cell.
    fn append_row_cells(
        &self,
        absolute_row: usize,
        line: i32,
        cols: usize,
        grapheme_buf: &mut [char; 32],
        text: &mut String,
        points: &mut Vec<crate::Point>,
    ) -> Result<()> {
        for column in 0..cols {
            let grid_ref = self.screen_grid_ref(column, absolute_row)?;
            if grid_ref.cell()?.wide()? == screen::CellWide::SpacerTail {
                continue;
            }
            let grapheme_count = grid_ref.graphemes(grapheme_buf)?;
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
        Ok(())
    }

    /// Returns the scrollback and active screen as text, without a trailing newline.
    pub(super) fn buffer_text(&self) -> Result<String> {
        let (mut text, _) = self.buffer_text_with_point_map()?;
        if text.ends_with('\n') {
            text.pop();
        }
        Ok(text)
    }

    /// Returns up to `line_count` of the last non-blank logical lines, top to bottom.
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

    // libghostty-vt has no vi mode, so cursor motion is implemented against the screen grid via
    // `ViGrid`. Wide characters split across a soft wrap are not normalized. Only whitespace word
    // motions and a depth-counting bracket motion are needed.

    /// Toggles vi mode and returns the new vi cursor, or `None` when disabled.
    ///
    /// The vi cursor starts at the terminal cursor if visible, otherwise at the top-left of the
    /// viewport.
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
            Some(cursor) => crate::Point::new(cursor.y as i32 - display_offset, cursor.x as usize),
            // start at the top-left of the viewport when the terminal cursor is scrolled away.
            None => grid.to_point(grid.line_base - i64::from(display_offset), 0),
        };
        self.current_vi_cursor = Some(point);
        Ok(Some(point))
    }

    /// Returns the vi cursor, or `None` when vi mode is disabled.
    pub(super) fn vi_cursor(&self) -> Option<crate::Point> {
        self.current_vi_cursor
    }

    /// Moves the vi cursor to `point` and extends an active selection to follow.
    ///
    /// Does nothing unless vi mode is enabled.
    pub(super) fn vi_goto_point(&mut self, point: crate::Point) -> Result<()> {
        if self.current_vi_cursor.is_none() {
            return Ok(());
        }
        self.scroll_viewport_to_reveal(point)?;
        self.current_vi_cursor = Some(point);
        self.update_selection(point)?;
        Ok(())
    }

    /// Scrolls the viewport by the minimum amount that brings `point` into view.
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
            self.terminal
                .scroll_viewport(ghostty_scroll(Scroll::Delta(delta), viewport_rows as usize));
        }
        Ok(())
    }

    /// Moves the vi cursor along with a plain scroll so it keeps its relative viewport position.
    ///
    /// Does nothing unless vi mode is enabled.
    pub(super) fn update_vi_cursor_for_scroll(
        &mut self,
        scroll: Scroll,
    ) -> Result<Option<crate::Point>> {
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
                let column = grid
                    .first_occupied_in_line(grid.topmost_row())?
                    .unwrap_or(0);
                grid.to_point(grid.topmost_row(), column)
            }
            Scroll::Bottom => {
                let column = grid
                    .first_occupied_in_line(grid.bottommost_row())?
                    .unwrap_or(0);
                grid.to_point(grid.bottommost_row(), column)
            }
        };
        self.current_vi_cursor = Some(point);
        Ok(Some(point))
    }

    /// Moves `lines` rows from `current`, clamped to the grid, onto the first occupied cell.
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

    /// Moves the vi cursor by `motion` and extends an active selection to follow.
    ///
    /// Does nothing unless vi mode is enabled.
    pub(super) fn vi_motion(&mut self, motion: ViMotion) -> Result<Option<crate::Point>> {
        let Some(current) = self.current_vi_cursor else {
            return Ok(None);
        };
        let grid = ViGrid::new(&self.terminal)?;
        let display_offset = self.alacritty_style_display_offset()? as i64;
        let point = ghostty_vi_motion(&grid, current, motion, display_offset)?;
        // reveal the new cursor after every motion, such as jumps or walking off screen.
        self.scroll_viewport_to_reveal(point)?;
        self.current_vi_cursor = Some(point);
        self.update_selection(point)?;
        Ok(Some(point))
    }

    /// Sets the default foreground, background, cursor, and palette colors from the theme.
    ///
    /// OSC 4/10/11/12 overrides from programs still apply on top. With defaults configured,
    /// Ghostty answers color queries itself, so Zed never needs to.
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

    /// Sets the default cursor shape from the `terminal.cursor_shape` setting.
    ///
    /// DECSCUSR still overrides it. `CursorShape::Hidden` never comes from settings and falls
    /// back to `Block`.
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

    /// Disables alternate scroll mode for `terminal.alternate_scroll: Off`.
    ///
    /// Ghostty enables the mode by default, so `On` needs no call.
    pub(super) fn disable_alternate_scroll(&mut self) -> Result<()> {
        use libghostty_vt::terminal::Mode;
        self.terminal.set_mode(Mode::ALT_SCROLL, false)?;
        Ok(())
    }

    /// Returns the IDs of images with at least one placement, visible or not.
    pub(super) fn live_image_ids(&mut self) -> Result<collections::HashSet<u32>> {
        let graphics = self.terminal.kitty_graphics()?;
        let mut iteration = self.placements.update(&graphics)?;
        let mut ids = collections::HashSet::default();
        while let Some(placement) = iteration.next() {
            match placement.image_id() {
                Ok(image_id) => {
                    ids.insert(image_id);
                }
                Err(error) => log::debug!("failed to read kitty placement image id: {error}"),
            }
        }
        Ok(ids)
    }

    /// Returns the Kitty graphics placements visible in the viewport, decoded to RGBA8.
    pub(super) fn image_placements(&mut self) -> Result<Vec<crate::ImagePlacement>> {
        let graphics = self.terminal.kitty_graphics()?;
        let mut iteration = self.placements.update(&graphics)?;
        let mut placements = Vec::new();
        let mut placed_image_ids = collections::HashSet::default();

        while let Some(placement) = iteration.next() {
            let image_id = match placement.image_id() {
                Ok(image_id) => image_id,
                Err(error) => {
                    log::debug!("failed to read kitty placement image id: {error}");
                    continue;
                }
            };
            placed_image_ids.insert(image_id);
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

            // decoding copies or inflates the whole image, so it is only redone on new pixels.
            let data = match self.decoded_images.get(&image_id) {
                Some((decoded_generation, data)) if *decoded_generation == generation => {
                    data.clone()
                }
                _ => match decode_image_rgba(&image) {
                    Ok(Some(data)) => {
                        self.decoded_images
                            .insert(image_id, (generation, data.clone()));
                        data
                    }
                    Ok(None) => continue,
                    Err(error) => {
                        log::debug!("failed to decode kitty image {image_id}: {error}");
                        continue;
                    }
                },
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

        self.decoded_images
            .retain(|image_id, _| placed_image_ids.contains(image_id));
        Ok(placements)
    }
}

/// Decodes the pixel data of a Kitty graphics image to RGBA8, inflating it if compressed.
///
/// Returns `Ok(None)` for undecodable formats or sizes.
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
        // uncompressed, or a future variant with nothing to inflate.
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
        // PNG payloads are decoded before storage, so this format should never be stored.
        ImageFormat::Png => return Ok(None),
        _ => return Ok(None),
    };

    Ok(Some(Arc::from(rgba)))
}

/// Installs a PNG decoder for Kitty graphics on the current thread.
///
/// The decoder is stored in thread-local state of the library. `RustPngDecoder` from
/// `libghostty-vt` 0.2.1 cannot be constructed outside its crate, so `ZedPngDecoder`
/// reimplements it.
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
        // libghostty-vt only accepts RGBA8, so expand palette and grayscale and strip 16-bit depth.
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

    // parses queued PTY output, batching chunks so libghostty-vt is not re-entered per read.
    let parser_thread = thread::Builder::new()
        .name("Ghostty PTY parser".to_string())
        .spawn(move || {
            // the decoder is thread-local, and PNG data is written on this thread.
            if let Err(error) = install_png_decoder() {
                log::error!("failed to install PNG decoder on ghostty parser thread: {error}");
            }

            let mut pending_chunk = None;
            let mut consecutive_batches = 0;
            while let Some(mut batch) = pending_chunk.take().or_else(|| output_rx.recv().ok()) {
                while batch.len() < MAX_PTY_PARSE_BATCH_BYTES {
                    match output_rx.try_recv() {
                        Ok(chunk) => {
                            if batch.len().saturating_add(chunk.len()) > MAX_PTY_PARSE_BATCH_BYTES {
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
                        Some(status) => {
                            PtyEvent::Event(crate::TerminalBackendEvent::ChildExit(status))
                        }
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
    let handle = child
        .as_raw_handle()
        .map(|handle| handle as i32)
        .unwrap_or(-1);
    #[cfg(not(windows))]
    let handle = master.as_raw_fd().unwrap_or(-1);
    let info = Arc::new(PtyProcessInfo::new(ProcessIdGetter::new(
        handle, process_id,
    )));
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

/// Builds the login shell command used when no explicit shell was requested.
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

/// Converts a scroll request into a libghostty-vt viewport scroll command.
pub(super) fn ghostty_scroll(scroll: Scroll, screen_lines: usize) -> ScrollViewport {
    match scroll {
        Scroll::Delta(delta) => ScrollViewport::Delta(-(delta as isize)),
        Scroll::PageUp => ScrollViewport::Delta(-(screen_lines as isize)),
        Scroll::PageDown => ScrollViewport::Delta(screen_lines as isize),
        Scroll::Top => ScrollViewport::Top,
        Scroll::Bottom => ScrollViewport::Bottom,
    }
}

/// Returns a recycled or newly allocated buffer sized to the PTY read chunk size.
fn next_pty_read_buffer(recycle_rx: &Receiver<Vec<u8>>) -> Vec<u8> {
    let mut buffer = recycle_rx
        .try_recv()
        .unwrap_or_else(|_| vec![0; PTY_READ_BUFFER_BYTES]);
    if buffer.len() < PTY_READ_BUFFER_BYTES {
        buffer.resize(PTY_READ_BUFFER_BYTES, 0);
    }
    buffer
}

/// Returns a drained buffer to the reader thread for reuse.
///
/// Drops it instead when the channel is full or disconnected, or the buffer grew unusually large.
fn recycle_pty_buffer(buffer: Vec<u8>, recycle_tx: &SyncSender<Vec<u8>>) {
    if buffer.capacity() > MAX_PTY_PARSE_BATCH_BYTES {
        return;
    }
    match recycle_tx.try_send(buffer) {
        Ok(()) | Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {}
    }
}

/// Writes a batch of PTY output into the Ghostty terminal and forwards its effects.
///
/// Returns whether the parser thread should keep running. Also used by the headless subprocess
/// path, which has no PTY.
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

/// Converts a resolved cell style color to `vte::ansi::Color`.
///
/// Keeps palette indices unresolved so theme colors apply at paint time.
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

/// Converts a Zed `Point` into the viewport space of Ghostty.
fn ghostty_viewport_point(
    display_offset: usize,
    point: crate::Point,
) -> libghostty_vt::terminal::Point {
    libghostty_vt::terminal::Point::Viewport(libghostty_vt::terminal::PointCoordinate {
        x: point.column.min(u16::MAX as usize) as u16,
        y: (point.line + display_offset as i32).max(0) as u32,
    })
}

fn ghostty_screen_point(line_base: i64, point: crate::Point) -> libghostty_vt::terminal::Point {
    libghostty_vt::terminal::Point::Screen(libghostty_vt::terminal::PointCoordinate {
        x: point.column.min(u16::MAX as usize) as u16,
        y: (i64::from(point.line) + line_base).clamp(0, i64::from(u32::MAX)) as u32,
    })
}

fn zed_point_from_screen_coordinate(
    line_base: i64,
    coordinate: libghostty_vt::terminal::PointCoordinate,
) -> crate::Point {
    crate::Point::new(
        (i64::from(coordinate.y) - line_base) as i32,
        coordinate.x as usize,
    )
}

/// Converts a derived `Selection` into a `crate::SelectionRange` without installing it.
///
/// A free function so callers can hold the `Selection`, which borrows `terminal`, across a
/// subsequent `terminal.set_selection` call.
fn selection_range_from_ghostty(
    terminal: &libghostty_vt::Terminal<'static, 'static>,
    line_base: i64,
    selection: &libghostty_vt::selection::Selection<'_>,
) -> Result<crate::SelectionRange> {
    let start = terminal
        .point_from_grid_ref(
            &selection.start(),
            libghostty_vt::terminal::PointSpace::Screen,
        )?
        .context("selection start not representable in screen space")?;
    let end = terminal
        .point_from_grid_ref(
            &selection.end(),
            libghostty_vt::terminal::PointSpace::Screen,
        )?
        .context("selection end not representable in screen space")?;
    Ok(crate::SelectionRange {
        start: zed_point_from_screen_coordinate(line_base, start),
        end: zed_point_from_screen_coordinate(line_base, end),
        is_block: selection.is_rectangle(),
    })
}

/// Returns the hyperlink URI of the cell at `point`, treating lookup errors as no hyperlink.
fn hyperlink_uri_at_point(
    terminal: &libghostty_vt::Terminal<'static, 'static>,
    display_offset: usize,
    point: crate::Point,
) -> Result<Option<String>> {
    let Ok(grid_ref) = terminal.grid_ref(ghostty_viewport_point(display_offset, point)) else {
        return Ok(None);
    };
    // longer than any real-world URI; longer URIs are truncated.
    let mut buf = [0u8; 4096];
    let Ok(len) = grid_ref.hyperlink_uri(&mut buf) else {
        return Ok(None);
    };
    if len == 0 {
        return Ok(None);
    }
    Ok(Some(
        String::from_utf8_lossy(&buf[..len.min(buf.len())]).into_owned(),
    ))
}

/// Walks from `point` in `direction` while the cell hyperlink stays `uri`.
///
/// Wraps across rows and returns the furthest point with the same hyperlink.
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

/// Facade over screen-space grid queries needed by the vi motions.
struct ViGrid<'a> {
    terminal: &'a libghostty_vt::Terminal<'static, 'static>,
    total_rows: i64,
    cols: i64,
    /// Offset such that `absolute_row = crate::Point::line + line_base`.
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
        Ok(self
            .terminal
            .grid_ref(libghostty_vt::terminal::Point::Screen(
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

    /// Whether the row soft-wraps into the next.
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

    /// Advances one cell in `direction`, wrapping at row ends and stopping at the grid edges.
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

    /// Moves off a wide character spacer within the same row.
    fn expand_wide(&self, absolute_row: i64, column: i64, direction: i64) -> Result<(i64, i64)> {
        if direction < 0 && self.is_wide_spacer(absolute_row, column)? && column > 0 {
            return Ok((absolute_row, column - 1));
        }
        Ok((absolute_row, column))
    }
}

/// Applies a vi motion on `ViGrid`, ported from the vi mode of Alacritty.
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

/// Applies a whitespace word motion (`W`, `B`, or `E`).
///
/// `direction` and `side` are `-1` for left and `1` for right.
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

/// Returns the bracket matching the one at the given point for `()`, `[]`, and `{}`.
///
/// Returns the unchanged point if it is not on a bracket or no match exists.
fn ghostty_bracket_motion(grid: &ViGrid<'_>, row: i64, col: i64) -> Result<(i64, i64)> {
    const PAIRS: [(char, char); 3] = [('(', ')'), ('[', ']'), ('{', '}')];
    let character = grid.char_at(row, col)?;
    let (open, close, direction) =
        if let Some(&(open, close)) = PAIRS.iter().find(|(open, _)| *open == character) {
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

/// Extracts the wrap-joined logical line containing `point` with a `Point` per `char`.
///
/// Also returns the char index of `point`. Trailing blanks are kept, since path regexes may rely
/// on whitespace separators. Returns `None` if `point` is not covered by the text.
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
        // only wrapped rows span the full width; others stop at their last occupied column, so
        // rest-of-line regexes do not swallow padding.
        let last_column = if grid.is_wrap(absolute_row)? {
            grid.last_column()
        } else {
            grid.last_occupied_in_line(absolute_row)?.unwrap_or(-1)
        };
        for column in 0..=last_column {
            let grid_ref = grid.grid_ref(absolute_row, column)?;
            let wide = grid_ref.cell()?.wide()?;
            if wide == screen::CellWide::SpacerTail {
                // hovering the spacer of a wide character hovers the character itself.
                if hovered_index.is_none() && grid.to_point(absolute_row, column) == point {
                    hovered_index = Some(points.len().saturating_sub(1));
                }
                continue;
            }
            if wide == screen::CellWide::SpacerHead {
                // placeholder for a wide character moved to the next row, not content.
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

/// Maps a palette index to `Color::Named` for the 16 ANSI slots and `Color::Indexed` otherwise.
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

    #[test]
    fn vi_motion_is_a_noop_when_vi_mode_is_disabled() {
        let mut terminal = GhosttyTerminal::new(20, 20, 100).unwrap();
        terminal.resize(test_bounds(20, 20)).unwrap();

        assert_eq!(terminal.vi_motion(ViMotion::Right).unwrap(), None);
        assert_eq!(terminal.vi_cursor(), None);
        terminal.vi_goto_point(crate::Point::new(0, 5)).unwrap();
        assert_eq!(terminal.vi_cursor(), None);
    }

    // search activation in vi mode relies on `vi_goto_point` alone to reveal the target.
    #[test]
    fn vi_goto_point_scrolls_to_reveal_an_off_screen_point() {
        let mut terminal = GhosttyTerminal::new(10, 3, 100).unwrap();
        terminal.resize(test_bounds(10, 3)).unwrap();
        for i in 0..20 {
            terminal.write(format!("line {i}\r\n").as_bytes());
        }
        terminal.toggle_vi_mode().unwrap();

        // deep in scrollback, above the live viewport.
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

        let after_top = terminal
            .update_vi_cursor_for_scroll(Scroll::Top)
            .unwrap()
            .unwrap();
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

    #[test]
    fn vi_motion_first_occupied_walks_back_across_wrapped_lines() {
        let mut terminal = GhosttyTerminal::new(20, 20, 100).unwrap();
        terminal.resize(test_bounds(20, 20)).unwrap();

        // row 0 wraps with ` x y` plus filler, row 1 is filler, and row 2 is `z `.
        let mut payload = String::new();
        payload.push_str(" x y");
        payload.push_str(&"a".repeat(16));
        payload.push_str(&"b".repeat(20));
        payload.push_str("z ");
        terminal.write(payload.as_bytes());

        terminal.toggle_vi_mode().unwrap();
        terminal.vi_goto_point(crate::Point::new(2, 1)).unwrap();

        let point = terminal
            .vi_motion(ViMotion::FirstOccupied)
            .unwrap()
            .unwrap();
        assert_eq!(
            point,
            crate::Point::new(2, 0),
            "'z' is the first occupied cell of row 2"
        );

        let point = terminal
            .vi_motion(ViMotion::FirstOccupied)
            .unwrap()
            .unwrap();
        assert_eq!(
            point,
            crate::Point::new(0, 1),
            "already at row 2's first occupied cell, so this should walk back across the wrap \
             to 'x', the first occupied cell of the whole wrapped run"
        );
    }

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

    #[test]
    fn search_matches_sees_new_output_and_maps_multibyte_offsets() {
        let mut terminal = GhosttyTerminal::new(20, 4, 100).unwrap();
        terminal.resize(test_bounds(20, 4)).unwrap();
        terminal.write("é ab é ab".as_bytes());
        let regex = regex::Regex::new("ab").unwrap();

        let matches = terminal.search_matches(&regex).unwrap();
        let columns: Vec<_> = matches.iter().map(|range| range.start().column).collect();
        assert_eq!(columns, [2, 7]);

        terminal.write(b"\r\nab");
        let matches = terminal.search_matches(&regex).unwrap();
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[2].start(), crate::Point::new(1, 0));
    }

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

    #[test]
    fn search_matches_finds_a_pattern_in_scrollback() {
        let mut terminal = GhosttyTerminal::new(20, 3, 100).unwrap();
        terminal.resize(test_bounds(20, 3)).unwrap();
        // push `NEEDLE` far above the viewport with filler rows.
        terminal.write(b"NEEDLE\r\n");
        for i in 0..30 {
            terminal.write(format!("filler {i}\r\n").as_bytes());
        }

        let regex = regex::Regex::new("NEEDLE").unwrap();
        let matches = terminal.search_matches(&regex).unwrap();

        assert_eq!(matches.len(), 1, "expected exactly one match in scrollback");
        assert_eq!(matches[0].start().column, 0);
        assert_eq!(matches[0].end().column, 5);
        // the match lies above the active screen.
        assert!(matches[0].start().line < 0);
    }

    #[test]
    fn search_matches_spans_a_soft_wrapped_line_boundary() {
        let mut terminal = GhosttyTerminal::new(10, 3, 100).unwrap();
        terminal.resize(test_bounds(10, 3)).unwrap();
        // 15 characters wrap across two rows of a 10-column terminal.
        terminal.write(b"wrapped_needle.");

        let regex = regex::Regex::new("needle").unwrap();
        let matches = terminal.search_matches(&regex).unwrap();

        assert_eq!(
            matches.len(),
            1,
            "expected the match to span the wrap boundary"
        );
        assert_eq!(matches[0].start(), crate::Point::new(0, 8));
        assert_eq!(matches[0].end(), crate::Point::new(1, 3));
    }

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

    // disables URL and path matching for tests of the OSC 8 path.
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

    #[test]
    fn hyperlink_at_finds_osc8_hyperlink_and_its_full_range() {
        let mut terminal = GhosttyTerminal::new(40, 2, 100).unwrap();
        terminal.resize(test_bounds(40, 2)).unwrap();
        // `click ` is plain, `here` in columns 6 to 9 is linked, and ` end` is plain.
        terminal.write(b"click \x1b]8;;https://example.com\x1b\\here\x1b]8;;\x1b\\ end");
        let (url_regex, path_regexes, timeout) = no_regex_fallback();

        for column in 6..=9 {
            let (uri, is_url, range) = terminal
                .hyperlink_at(
                    crate::Point::new(0, column),
                    &url_regex,
                    &path_regexes,
                    timeout,
                )
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
        // `one` (URI A) directly followed by `two` (URI B).
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

    #[test]
    fn hyperlink_at_finds_bare_url_and_trims_trailing_punctuation() {
        let mut terminal = GhosttyTerminal::new(40, 2, 100).unwrap();
        terminal.resize(test_bounds(40, 2)).unwrap();
        terminal.write(b"visit https://example.com/page, thanks");
        let (url_regex, path_regexes, timeout) = url_regex_fallback();

        // column 10 lies inside the URL.
        let (text, is_url, range) = terminal
            .hyperlink_at(crate::Point::new(0, 10), &url_regex, &path_regexes, timeout)
            .unwrap()
            .expect("expected a bare URL match");
        assert_eq!(text, "https://example.com/page");
        assert!(is_url);
        assert_eq!(range.start(), crate::Point::new(0, 6));
        // the URL spans columns 6 to 29, excluding the trailing comma.
        assert_eq!(range.end(), crate::Point::new(0, 29));
    }

    #[test]
    fn hyperlink_at_finds_path_with_line_and_column_via_user_regex() {
        let mut terminal = GhosttyTerminal::new(60, 2, 100).unwrap();
        terminal.resize(test_bounds(60, 2)).unwrap();
        terminal.write(b"error at src/main.rs:42:7: something broke");
        let url_regex = regex::Regex::new("$^").unwrap();
        let path_regexes =
            vec![regex::Regex::new(r"(?<path>\S+\.rs):(?<line>[0-9]+):(?<column>[0-9]+)").unwrap()];

        // column 15 lies inside `src/main.rs`.
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

    #[test]
    fn select_word_at_selects_the_word_under_the_point() {
        let mut terminal = GhosttyTerminal::new(20, 2, 100).unwrap();
        terminal.resize(test_bounds(20, 2)).unwrap();
        terminal.write(b"hello world");

        // column 7 lies inside `world`.
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
        assert_eq!(
            terminal.selection_text().unwrap().as_deref(),
            Some("hello world")
        );
        assert_eq!(range.start.line, 0);
        assert_eq!(range.end.line, 0);
    }

    #[test]
    fn select_all_selects_every_line() {
        let mut terminal = GhosttyTerminal::new(20, 2, 100).unwrap();
        terminal.resize(test_bounds(20, 2)).unwrap();
        terminal.write(b"hello world\r\nsecond line");

        terminal
            .select_all()
            .unwrap()
            .expect("expected a selection");
        let text = terminal.selection_text().unwrap().unwrap();
        assert!(text.contains("hello world"));
        assert!(text.contains("second line"));
    }

    #[test]
    fn select_word_range_extends_across_multiple_words() {
        let mut terminal = GhosttyTerminal::new(30, 2, 100).unwrap();
        terminal.resize(test_bounds(30, 2)).unwrap();
        terminal.write(b"git commit --amend now");

        // anchor on `commit` and drag right onto `amend`.
        let range = terminal
            .select_word_range(crate::Point::new(0, 4), crate::Point::new(0, 15))
            .unwrap()
            .expect("expected a word-range selection");
        assert_eq!(
            terminal.selection_text().unwrap().as_deref(),
            Some("commit --amend")
        );
        assert_eq!(terminal.selection_range(), Some(range));

        // dragging left onto `git` flips the selection instead of extending it.
        let range = terminal
            .select_word_range(crate::Point::new(0, 4), crate::Point::new(0, 1))
            .unwrap()
            .expect("expected a word-range selection");
        assert_eq!(
            terminal.selection_text().unwrap().as_deref(),
            Some("git commit")
        );
        assert_eq!(terminal.selection_range(), Some(range));
    }

    #[test]
    fn select_line_range_extends_across_multiple_lines() {
        let mut terminal = GhosttyTerminal::new(20, 5, 100).unwrap();
        terminal.resize(test_bounds(20, 5)).unwrap();
        terminal.write(b"first\r\nsecond\r\nthird\r\nfourth");

        // anchor on line 1 and drag down onto line 2.
        terminal
            .select_line_range(crate::Point::new(1, 2), crate::Point::new(2, 3))
            .unwrap()
            .expect("expected a line-range selection");
        let text = terminal.selection_text().unwrap().unwrap();
        assert!(text.contains("second"));
        assert!(text.contains("third"));
        assert!(!text.contains("first"));
        assert!(!text.contains("fourth"));

        // dragging up onto line 0 flips the selection to lines 0 and 1.
        terminal
            .select_line_range(crate::Point::new(1, 2), crate::Point::new(0, 0))
            .unwrap()
            .expect("expected a line-range selection");
        let text = terminal.selection_text().unwrap().unwrap();
        assert!(text.contains("first"));
        assert!(text.contains("second"));
        assert!(!text.contains("third"));
    }

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
    fn selection_extended_after_scrolling_keeps_its_offscreen_anchor() {
        let mut terminal = GhosttyTerminal::new(20, 3, 100).unwrap();
        terminal.resize(test_bounds(20, 3)).unwrap();
        for line in 0..10 {
            terminal.write(format!("line {line}\r\n").as_bytes());
        }

        terminal.scroll_viewport(ScrollViewport::Top);
        let top_display_offset = terminal.alacritty_style_display_offset().unwrap() as i32;
        terminal
            .set_selection(Some(crate::SelectionRange {
                start: crate::Point::new(-top_display_offset, 0),
                end: crate::Point::new(-top_display_offset, 5),
                is_block: false,
            }))
            .unwrap();

        terminal.scroll_viewport(ScrollViewport::Bottom);
        terminal.update_selection(crate::Point::new(1, 5)).unwrap();

        let text = terminal.selection_text().unwrap().unwrap_or_default();
        assert!(
            text.starts_with("line 0"),
            "lost offscreen anchor: {text:?}"
        );
        assert!(text.ends_with("line 9"), "missing drag end: {text:?}");
    }

    #[test]
    fn update_selection_without_an_active_selection_returns_false() {
        let mut terminal = GhosttyTerminal::new(20, 2, 100).unwrap();
        terminal.resize(test_bounds(20, 2)).unwrap();

        let extended = terminal.update_selection(crate::Point::new(0, 3)).unwrap();
        assert!(!extended);
    }

    // unlike XTVERSION or DA1/DA2, Ghostty ignores color queries until a color is configured.
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

    // a query right after an OSC 11 SET used to be answered twice.
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

    // program overrides take precedence over configured defaults.
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
        assert_eq!(
            (background.r, background.g, background.b),
            (0x1E, 0x1E, 0x2E)
        );
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
            (
                overridden_background.r,
                overridden_background.g,
                overridden_background.b
            ),
            (0x12, 0x34, 0x56)
        );
    }

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

    // guards against a library upgrade starting to answer clipboard reads.
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

    // transmits a 1x1 PNG, as in the `kitty::graphics` doctest.
    #[test]
    fn live_image_ids_drop_deleted_images() {
        let mut terminal = GhosttyTerminal::new(8, 4, 100).unwrap();
        terminal.resize(test_bounds(8, 4)).unwrap();
        terminal.write(
            b"\x1b_Ga=T,f=100,i=7,q=1;\
              iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA\
              DUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==\
              \x1b\\",
        );
        assert!(terminal.live_image_ids().unwrap().contains(&7));

        terminal.write(b"\x1b_Ga=d,d=A,q=1\x1b\\");
        assert!(terminal.live_image_ids().unwrap().is_empty());
    }

    // reference extraction that reads every cell individually.
    fn buffer_text_per_cell(terminal: &GhosttyTerminal) -> (String, Vec<crate::Point>) {
        let total_rows = terminal.terminal.total_rows().unwrap();
        let viewport_rows = usize::from(terminal.terminal.rows().unwrap());
        let cols = usize::from(terminal.terminal.cols().unwrap());
        let line_base = total_rows as i32 - viewport_rows as i32;
        let mut text = String::new();
        let mut points = Vec::new();
        let mut grapheme_buf = [' '; 32];
        for absolute_row in 0..total_rows {
            let line = absolute_row as i32 - line_base;
            let row_wrapped = terminal
                .screen_grid_ref(0, absolute_row)
                .unwrap()
                .row()
                .unwrap()
                .is_wrapped()
                .unwrap();
            let row_start = points.len();
            terminal
                .append_row_cells(
                    absolute_row,
                    line,
                    cols,
                    &mut grapheme_buf,
                    &mut text,
                    &mut points,
                )
                .unwrap();
            if !row_wrapped {
                while points.len() > row_start && text.ends_with(' ') {
                    text.pop();
                    points.pop();
                }
                text.push('\n');
                points.push(crate::Point::new(line, cols));
            }
        }
        (text, points)
    }

    #[test]
    fn buffer_text_matches_per_cell_extraction() {
        let mut terminal = GhosttyTerminal::new(12, 4, 1000).unwrap();
        terminal.resize(test_bounds(12, 4)).unwrap();
        terminal.write(
            "plain ascii\r\n\r\n  indented  \r\na\tb\ttab\r\n中文 wide\r\ne\u{0301} accent\r\n\
             \u{1F980} crab\r\n\x1b[31mred\x1b[0m text\r\nthis line is long enough to wrap twice\r\n\
             trailing   \r\nlast"
                .as_bytes(),
        );
        assert_eq!(
            terminal.buffer_text_with_point_map().unwrap(),
            buffer_text_per_cell(&terminal)
        );
    }

    #[test]
    fn scrollback_keeps_the_configured_number_of_lines() {
        let mut terminal = GhosttyTerminal::new(200, 40, 1_000).unwrap();
        terminal.resize(test_bounds(200, 40)).unwrap();
        let mut output = String::new();
        for line in 0..2_000 {
            output.push_str(&format!("line {line}\r\n"));
        }
        terminal.write(output.as_bytes());
        assert!(terminal.total_lines().unwrap() >= 1_000);
    }

    #[test]
    fn selection_text_follows_selection_and_output_changes() {
        let mut terminal = GhosttyTerminal::new(20, 3, 100).unwrap();
        terminal.resize(test_bounds(20, 3)).unwrap();
        terminal.write(b"hello world");
        let select = |terminal: &mut GhosttyTerminal, end: usize| {
            terminal
                .set_selection(Some(crate::SelectionRange {
                    start: crate::Point::new(0, 0),
                    end: crate::Point::new(0, end),
                    is_block: false,
                }))
                .unwrap();
        };

        select(&mut terminal, 4);
        assert_eq!(terminal.selection_text().unwrap().as_deref(), Some("hello"));
        select(&mut terminal, 10);
        assert_eq!(
            terminal.selection_text().unwrap().as_deref(),
            Some("hello world")
        );

        terminal.write(b"\rHELLO");
        assert_eq!(
            terminal.selection_text().unwrap().as_deref(),
            Some("HELLO world")
        );
    }

    #[test]
    fn image_placements_reuse_decoded_pixels_until_the_image_changes() {
        let mut terminal = GhosttyTerminal::new(8, 4, 100).unwrap();
        terminal.resize(test_bounds(8, 4)).unwrap();
        let transmit = |terminal: &mut GhosttyTerminal, pixel: [u8; 4]| {
            let payload = base64_encode(&pixel);
            terminal.write(format!("\x1b_Ga=T,f=32,s=1,v=1,i=9,q=1;{payload}\x1b\\").as_bytes());
        };

        transmit(&mut terminal, [255, 0, 0, 255]);
        let first = terminal.image_placements().unwrap();
        let second = terminal.image_placements().unwrap();
        assert!(Arc::ptr_eq(&first[0].data, &second[0].data));

        transmit(&mut terminal, [0, 0, 255, 255]);
        let replaced = terminal.image_placements().unwrap();
        let replaced = replaced
            .iter()
            .find(|placement| placement.image_id == 9)
            .unwrap();
        assert_eq!(&replaced.data[..4], &[0, 0, 255, 255]);

        terminal.write(b"\x1b_Ga=d,d=A,q=1\x1b\\");
        assert!(terminal.image_placements().unwrap().is_empty());
        assert!(terminal.decoded_images.is_empty());
    }

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
        // without `c=` or `r=`, the placement keeps its native pixel size.
        assert_eq!(placement.pixel_width, 1);
        assert_eq!(placement.pixel_height, 1);
        assert_eq!(
            placement.data.len(),
            4,
            "expected one decoded RGBA8 pixel, got {} bytes",
            placement.data.len()
        );
    }

    // square RGBA images without explicit rows, like `kitty +kitten icat -H`, must round 32, 33,
    // 64, and 65px at a 32px cell height up to 1, 2, 2, and 3 rows.
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
            terminal
                .write(format!("\x1b_Ga=T,f=32,s={height},v={height},q=1;{data}\x1b\\").as_bytes());

            let placements = terminal.image_placements().unwrap();
            assert_eq!(
                placements.len(),
                1,
                "height={height}: expected one placement"
            );
            assert_eq!(
                placements[0].grid_rows, expected_rows,
                "height={height}: expected {expected_rows} rows"
            );
            assert_eq!(placements[0].pixel_height, height);
            assert_eq!(placements[0].pixel_width, height, "image is square");
        }
    }

    // truncating a fractional cell height biased the recovered row count up, adding blank lines
    // below tall images.
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
            terminal
                .write(format!("\x1b_Ga=T,f=32,s={height},v={height},q=1;{data}\x1b\\").as_bytes());

            let placements = terminal.image_placements().unwrap();
            assert_eq!(
                placements.len(),
                1,
                "height={height}: expected one placement"
            );
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
        terminal.write(format!("\x1b_Ga=T,f=32,s={width},v={height},q=1;{data}\x1b\\").as_bytes());

        let placements = terminal.image_placements().unwrap();
        assert_eq!(placements.len(), 1, "expected 1 placement");
        let p = &placements[0];
        eprintln!(
            "Placement: viewport_row={}, viewport_col={}, grid_rows={}, grid_cols={}, px_w={}, px_h={}",
            p.viewport_row,
            p.viewport_column,
            p.grid_rows,
            p.grid_columns,
            p.pixel_width,
            p.pixel_height
        );
    }

    // colors must stay `Named`, `Indexed`, or `Spec`, since only the first two follow the theme.
    #[test]
    fn build_content_reports_text_and_styles() {
        let mut terminal = GhosttyTerminal::new(20, 3, 100).unwrap();
        terminal.resize(test_bounds(20, 3)).unwrap();

        terminal.write(
            b"\x1b[1mB\x1b[22m\x1b[3mI\x1b[23m\x1b[4mU\x1b[24m\x1b[4:3mC\x1b[24m\x1b[9mS\x1b[29m",
        );
        // named, 256-indexed, and true-color foregrounds, one cell each.
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

        // untouched cells keep the default colors so themes still apply.
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

        // `e` plus a combining accent fits the inline grapheme buffer.
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

        // 40 combining marks exceed the inline buffer and use the heap fallback.
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

        // `中` is a double-width character.
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

    // the Ghostty scrollbar offset has the opposite polarity of `display_offset`, which only
    // shows once scrollback exists.
    #[test]
    fn build_content_reports_alacritty_style_display_offset_when_scrolled() {
        let mut terminal = GhosttyTerminal::new(10, 3, 100).unwrap();
        terminal.resize(test_bounds(10, 3)).unwrap();

        // 20 lines in a 3-row viewport leave 17 rows of scrollback.
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

    #[test]
    fn build_content_uses_grid_lines_when_scrolled() {
        let mut terminal = GhosttyTerminal::new(10, 3, 100).unwrap();
        terminal.resize(test_bounds(10, 3)).unwrap();
        for i in 0..20 {
            terminal.write(format!("line {i}\r\n").as_bytes());
        }
        terminal.scroll_viewport(ScrollViewport::Top);

        let (cells, _mode, display_offset) = terminal.build_content().unwrap();
        assert!(display_offset > 0);
        let first = cells.first().unwrap();
        assert_eq!(first.point.line, -(display_offset as i32));
        let first_row: String = cells
            .iter()
            .filter(|cell| cell.point.line == first.point.line)
            .map(|cell| cell.character())
            .collect();
        assert_eq!(first_row.trim(), "line 0");
    }

    #[test]
    fn build_content_reports_osc8_hyperlinks() {
        let mut terminal = GhosttyTerminal::new(20, 2, 100).unwrap();
        terminal.resize(test_bounds(20, 2)).unwrap();
        terminal.write(b"a \x1b]8;;https://zed.dev\x1b\\link\x1b]8;;\x1b\\ b");

        let (cells, _mode, _display_offset) = terminal.build_content().unwrap();
        let uri_at = |column: usize| {
            cells
                .iter()
                .find(|cell| cell.point == crate::Point::new(0, column))
                .and_then(|cell| cell.hyperlink())
                .map(|hyperlink| hyperlink.uri().to_string())
        };
        assert_eq!(uri_at(0), None);
        for column in 2..=5 {
            assert_eq!(uri_at(column).as_deref(), Some("https://zed.dev"));
        }
        assert_eq!(uri_at(6), None);
    }

    fn base64_encode(data: &[u8]) -> String {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
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

    // drives a real `GhosttyTerminal` through VT bytes to test path and IRI matching of
    // `hyperlink_at`.
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
                    // full-width characters used in tests
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

        /// Returns the cursor position in screen space.
        ///
        /// `line_base` shifts while scrollback grows, so positions stay absolute until all
        /// lines are written.
        fn cursor_point(
            terminal: &libghostty_vt::Terminal<'static, 'static>,
        ) -> Result<(i64, i64)> {
            let grid = ViGrid::new(terminal)?;
            let point = crate::Point::new(
                terminal.cursor_y().unwrap_or(0) as i32,
                terminal.cursor_x().unwrap_or(0) as usize,
            );
            Ok(grid.to_absolute(point))
        }

        /// Returns the grid point of the last written character, accounting for pending wrap and
        /// wide character spacers.
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

        /// Returns the point one cell past `point`, for the wide character spacer marker.
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

        /// Builds a terminal from annotated test lines and returns the expected hyperlink.
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
                                // turn a Unix absolute path into a Windows one.
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
                let url = url::Url::parse(&iri_or_path).unwrap_or_else(|error| {
                    panic!("Failed to parse file IRI `{iri_or_path}`: {error}")
                });
                let path = url.to_file_path().unwrap_or_else(|_| {
                    panic!("Failed to interpret file IRI `{iri_or_path}` as a path")
                });
                iri_or_path = path.to_string_lossy().into_owned();
            }

            let hovered_point = hovered_point.expect("Missing hovered point (👉 or 👈)");
            // convert all absolute positions through one final `ViGrid`, matching what
            // `hyperlink_at` builds next.
            let final_grid = ViGrid::new(&terminal.terminal).unwrap();
            let hovered_char = final_grid
                .char_at(hovered_point.0, hovered_point.1)
                .unwrap_or(' ');
            let hovered_point = final_grid.to_point(hovered_point.0, hovered_point.1);
            let hyperlink_match = final_grid
                .to_point(hyperlink_match.start().0, hyperlink_match.start().1)
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
            // Ghostty keeps VARIATION SELECTOR-16 in the grapheme cluster of 🏛.
            const MULTIPLE_SAME_LINE_REGEX: &str = r#"(?<link>(?<path>🦀 multiple_same_line 🦀) 🚣(?<line>[0-9]+) 🏛\u{FE0F}?(?<column>[0-9]+)):"#;
            // the default `terminal.path_hyperlink_regexes` from `default.json`, joined with `\n`.
            const PYTHON_DIAGNOSTIC_REGEX: &str = r#"File "(?<path>[^"]+)", line (?<line>[0-9]+)"#;
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
                    // `hyperlink_at` returns raw `file://` IRIs, so convert them like
                    // the expected value.
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
                        (
                            text.as_str(),
                            format_hyperlink_match(&(range.start()..=range.end())).as_str()
                        ),
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

            // likely worth fixing.
            mod issues {
                use super::*;

                #[test]
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
                fn issue_12338_regex() {
                    test_path!(".rw-r--r--     0     staff 05-27 14:03 ‹«'test file 👉1.txt'»›");
                    test_path!(".rw-r--r--     0     staff 05-27 14:03 ‹«👉'test file 1.txt'»›");
                }

                #[test]
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
                fn issue_40202() {
                    test_path!("[‹«lib/blitz_apex_👉server/stats/aggregate_rank_stats.ex»:«35»›: BlitzApexServer.Stats.AggregateRankStats.update/2]
                1 #=> 1");
                }

                #[test]
                fn issue_28194() {
                    test_path!(
                        "‹«test/c👉ontrollers/template_items_controller_test.rb»:«20»›:in 'block (2 levels) in <class:TemplateItemsControllerTest>'"
                    );
                }

                #[test]
                fn issue_50531() {
                    test_path!("0: ‹«foo/👉bar.txt»›");
                    test_path!("0: ‹«👉foo/bar.txt»›");
                    test_path!("42: ‹«👉foo/bar.txt»›");
                    test_path!("1: ‹«/👉test/cool.rs»›");
                    test_path!("1: ‹«/👉test/cool.rs»:«4»:«2»›");
                }

                #[test]
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

            // minor issues not worth fixing.
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
                // balanced parentheses stay in the path; an unbalanced leading `(` is stripped.
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

            // Ghostty classifies this as an IRI, unlike the path misclassification in Alacritty.
            #[test]
            fn file_is_a_path() {
                test_iri!("file://test/cool/index.rs");
            }
        }
    }
}
