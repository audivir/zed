use log::{info, warn};
use regex::Regex;
use std::{
    ops::Range as StdRange,
    time::{Duration, Instant},
};
use url::Url;
use util::paths::{PathStyle, UrlExt};

use crate::Range;

pub(crate) const URL_REGEX: &str = r#"(ipfs:|ipns:|magnet:|mailto:|gemini://|gopher://|https://|http://|news:|file://|git://|ssh:|ftp://|zed://)[^\u{0000}-\u{001F}\u{007F}-\u{009F}<>"\s{-}\^⟨⟩`']+"#;

pub(crate) struct RegexSearches {
    path_hyperlink_regexes: Vec<Regex>,
    path_hyperlink_timeout: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HyperlinkMatch {
    pub(crate) text: String,
    pub(crate) is_url: bool,
    pub(crate) range: Range,
}

impl Default for RegexSearches {
    fn default() -> Self {
        Self::new(Vec::<String>::new(), Duration::ZERO)
    }
}
impl RegexSearches {
    pub(crate) fn new(
        path_hyperlink_regexes: impl IntoIterator<Item: AsRef<str>>,
        path_hyperlink_timeout: Duration,
    ) -> Self {
        Self {
            path_hyperlink_regexes: Self::path_hyperlink_regexes(path_hyperlink_regexes),
            path_hyperlink_timeout,
        }
    }

    /// Compiled `terminal.path_hyperlink_regexes` patterns.
    pub(crate) fn compiled_path_hyperlink_regexes(&self) -> &[Regex] {
        &self.path_hyperlink_regexes
    }

    /// Value of the `terminal.path_hyperlink_timeout_ms` setting.
    pub(crate) fn path_hyperlink_timeout(&self) -> Duration {
        self.path_hyperlink_timeout
    }

    fn path_hyperlink_regexes(
        path_hyperlink_regexes: impl IntoIterator<Item: AsRef<str>>,
    ) -> Vec<Regex> {
        path_hyperlink_regexes
            .into_iter()
            .filter_map(|regex| {
                Regex::new(regex.as_ref())
                    .inspect_err(|error| {
                        warn!(
                            concat!(
                                "Ignoring path hyperlink regex specified in ",
                                "`terminal.path_hyperlink_regexes`:\n\n\t{}\n\nError: {}",
                            ),
                            regex.as_ref(),
                            error
                        );
                    })
                    .ok()
            })
            .collect()
    }
}

/// Normalizes a raw URL or path match into a `HyperlinkMatch`.
///
/// `file://` URIs resolve to plain paths so trailing line numbers are handled correctly.
pub(crate) fn normalize_hyperlink_match(
    maybe_url_or_path: String,
    is_url: bool,
    range: Range,
    path_style: PathStyle,
) -> HyperlinkMatch {
    if is_url {
        // treat `file://` IRIs as paths so trailing line numbers work. `Url::to_file_path`
        // handles Windows drive letters.
        if maybe_url_or_path.starts_with("file://") {
            if let Ok(url) = Url::parse(&maybe_url_or_path) {
                if let Ok(path) = url.to_file_path_ext(path_style) {
                    return HyperlinkMatch {
                        text: path.to_string_lossy().into_owned(),
                        is_url: false,
                        range,
                    };
                } else if let Some(path) = try_osc8_url_to_path(url)
                    && path_style.is_posix()
                {
                    return HyperlinkMatch {
                        text: path,
                        is_url: false,
                        range,
                    };
                }
            }
            // fallback when URL parsing fails.
            let path = maybe_url_or_path
                .strip_prefix("file://")
                .unwrap_or(&maybe_url_or_path);
            HyperlinkMatch {
                text: path.to_string(),
                is_url: false,
                range,
            }
        } else {
            HyperlinkMatch {
                text: maybe_url_or_path,
                is_url: true,
                range,
            }
        }
    } else {
        HyperlinkMatch {
            text: maybe_url_or_path,
            is_url: false,
            range,
        }
    }
}

// OSC 8 encodes file URIs as `file://{host}{path}`, so the host part is skipped.
fn try_osc8_url_to_path(url: url::Url) -> Option<String> {
    use percent_encoding::percent_decode;
    if url.scheme() != "file" {
        return None;
    }

    let bytes = url
        .path_segments()?
        .skip(1)
        .flat_map(|segment| percent_decode(segment.as_bytes()))
        .collect::<Vec<u8>>();
    bytes.try_into().ok()
}

/// Trims trailing delimiter punctuation from a URL.
///
/// Returns the trimmed string and the number of characters removed from the end.
pub(crate) fn trim_url_punctuation(url: &str) -> (String, usize) {
    let mut sanitized_url = url.to_string();
    let mut chars_trimmed = 0;

    let (open_parens, mut close_parens) =
        sanitized_url
            .chars()
            .fold((0, 0), |(opens, closes), c| match c {
                '(' => (opens + 1, closes),
                ')' => (opens, closes + 1),
                _ => (opens, closes),
            });

    while let Some(last_char) = sanitized_url.chars().last() {
        let should_remove = match last_char {
            // valid inside a URL, but usually a text delimiter at its end.
            '.' | ',' | ':' | ';' => true,
            '(' => true,
            ')' if close_parens > open_parens => {
                close_parens -= 1;

                true
            }
            _ => false,
        };

        if should_remove {
            sanitized_url.pop();
            chars_trimmed += 1;
        } else {
            break;
        }
    }

    (sanitized_url, chars_trimmed)
}

/// Returns the byte offset past the first unbalanced `(` in `s`.
///
/// Strips prefixes like `Update(` from path matches while keeping balanced parentheses in
/// file names like `file(copy).txt`.
pub(crate) fn first_unbalanced_open_paren(s: &str) -> Option<usize> {
    let mut balance: i32 = 0;
    let mut first_unmatched = None;
    for (i, c) in s.char_indices() {
        match c {
            '(' => {
                if balance == 0 {
                    first_unmatched = Some(i + c.len_utf8());
                }
                balance += 1;
            }
            ')' => {
                balance -= 1;
                if balance <= 0 {
                    balance = 0;
                    first_unmatched = None;
                }
            }
            _ => {}
        }
    }
    first_unmatched.filter(|_| balance > 0)
}

/// Returns path hyperlink candidates in `line` whose link range covers `hovered_byte_offset`.
///
/// Tries each configured regex in order, using the named captures `path`, `line`, `column`, and
/// `link` described on the `terminal.path_hyperlink_regexes` setting. Each candidate is a path
/// with `:line[:column]` appended when captured, plus its clickable byte range in `line`.
/// Multiple candidates are returned because the caller must verify each range in its own
/// coordinates.
///
/// Scanning stops at the first regex that matches anywhere on the line, even without covering
/// the hover point, or once the configured timeout is exceeded.
pub(crate) fn path_hyperlink_candidates_in_line(
    line: &str,
    hovered_byte_offset: usize,
    path_hyperlink_regexes: &[Regex],
    path_hyperlink_timeout: Duration,
) -> Vec<(String, StdRange<usize>)> {
    if path_hyperlink_regexes.is_empty() || path_hyperlink_timeout.as_millis() == 0 {
        return Vec::new();
    }
    let search_start_time = Instant::now();

    let timed_out = || {
        let elapsed_time = Instant::now().saturating_duration_since(search_start_time);
        (elapsed_time > path_hyperlink_timeout)
            .then_some((elapsed_time.as_millis(), path_hyperlink_timeout.as_millis()))
    };

    for regex in path_hyperlink_regexes {
        let mut path_found = false;
        let mut candidates = Vec::new();

        for captures in regex.captures_iter(line) {
            path_found = true;
            let match_range = captures.get(0).unwrap().range();
            let (mut path_range, line_column) = if let Some(path) = captures.name("path") {
                let parse = |name: &str| {
                    captures
                        .name(name)
                        .and_then(|capture| capture.as_str().parse().ok())
                };

                (
                    path.range(),
                    parse("line").map(|line: u32| (line, parse("column"))),
                )
            } else {
                (match_range.clone(), None)
            };
            let mut link_range = captures
                .name("link")
                .map_or_else(|| match_range.clone(), |link| link.range());

            // strip delimiter parentheses like `Update(.claude/SKILL.md)`, the counterpart of
            // trimming unbalanced trailing `)` from URLs.
            if let Some(trim) = first_unbalanced_open_paren(&line[path_range.clone()]) {
                path_range.start += trim;
                link_range.start = link_range.start.max(path_range.start);
            }

            if !link_range.contains(&hovered_byte_offset) {
                continue;
            }

            let mut path = line[path_range].to_string();
            if let Some((line_no, column)) = line_column {
                path += &format!(":{line_no}");
                if let Some(column) = column {
                    path += &format!(":{column}");
                }
            }
            candidates.push((path, link_range));
        }

        if !candidates.is_empty() {
            return candidates;
        }

        if path_found {
            return Vec::new();
        }

        if let Some((timed_out_ms, timeout_ms)) = timed_out() {
            warn!("Timed out processing path hyperlink regexes after {timed_out_ms}ms");
            info!("{timeout_ms}ms time out specified in `terminal.path_hyperlink_timeout_ms`");
            return Vec::new();
        }
    }

    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terminal_settings::TerminalSettings;
    use settings::Settings;

    fn re_test(re: &str, hay: &str, expected: Vec<&str>) {
        let results: Vec<_> = Regex::new(re)
            .unwrap()
            .find_iter(hay)
            .map(|m| m.as_str())
            .collect();
        assert_eq!(results, expected);
    }

    #[test]
    fn test_default_path_hyperlink_regexes() -> anyhow::Result<()> {
        let content = settings::parse_json_with_comments(&settings::default_settings())?;
        let settings = TerminalSettings::from_settings(&content);
        let searches = RegexSearches::new(
            &settings.path_hyperlink_regexes,
            Duration::from_millis(settings.path_hyperlink_timeout_ms),
        );
        let examples = [
            (r#"File "src/main.py", line 42"#, "src/main.py", Some("42")),
            ("src/main.rs:42", "src/main.rs:42", None),
        ];
        assert_eq!(searches.path_hyperlink_regexes.len(), examples.len());
        assert_eq!(
            searches.path_hyperlink_regexes.len(),
            settings.path_hyperlink_regexes.len()
        );
        for (regex, (input, path, line)) in searches.path_hyperlink_regexes.iter().zip(examples) {
            let captures = regex
                .captures(input)
                .ok_or_else(|| anyhow::anyhow!("Default regex does not match {input}"))?;
            assert_eq!(
                captures.name("path").map(|capture| capture.as_str()),
                Some(path)
            );
            assert_eq!(captures.name("line").map(|capture| capture.as_str()), line);
        }
        Ok(())
    }

    #[test]
    fn test_url_regex() {
        re_test(
            URL_REGEX,
            "test http://example.com test 'https://website1.com' test mailto:bob@example.com train",
            vec![
                "http://example.com",
                "https://website1.com",
                "mailto:bob@example.com",
            ],
        );
        re_test(
            URL_REGEX,
            "open zed://channel/the-channel and zed://settings/theme now",
            vec!["zed://channel/the-channel", "zed://settings/theme"],
        );
    }

    #[test]
    fn test_url_parentheses_sanitization() {
        let test_cases = vec![
            // unbalanced parentheses
            ("https://www.google.com/)", "https://www.google.com/"),
            ("https://example.com/path)", "https://example.com/path"),
            ("https://test.com/))", "https://test.com/"),
            ("https://test.com/(((", "https://test.com/"),
            ("https://test.com/(test)(", "https://test.com/(test)"),
            // balanced parentheses
            (
                "https://en.wikipedia.org/wiki/Example_(disambiguation)",
                "https://en.wikipedia.org/wiki/Example_(disambiguation)",
            ),
            ("https://test.com/(hello)", "https://test.com/(hello)"),
            (
                "https://example.com/path(1)(2)",
                "https://example.com/path(1)(2)",
            ),
            // edge cases
            ("https://test.com/", "https://test.com/"),
            ("https://example.com", "https://example.com"),
        ];

        for (input, expected) in test_cases {
            let (result, _) = trim_url_punctuation(input);
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
    }

    #[test]
    fn test_url_punctuation_sanitization() {
        let test_cases = vec![
            ("https://example.com.", "https://example.com"),
            (
                "https://github.com/zed-industries/zed.",
                "https://github.com/zed-industries/zed",
            ),
            (
                "https://example.com/path/file.html.",
                "https://example.com/path/file.html",
            ),
            (
                "https://example.com/file.pdf.",
                "https://example.com/file.pdf",
            ),
            ("https://example.com:8080.", "https://example.com:8080"),
            ("https://example.com..", "https://example.com"),
            (
                "https://en.wikipedia.org/wiki/C.E.O.",
                "https://en.wikipedia.org/wiki/C.E.O",
            ),
            ("https://example.com,", "https://example.com"),
            ("https://example.com/path,", "https://example.com/path"),
            ("https://example.com,,", "https://example.com"),
            ("https://example.com:", "https://example.com"),
            ("https://example.com/path:", "https://example.com/path"),
            ("https://example.com::", "https://example.com"),
            ("https://example.com;", "https://example.com"),
            ("https://example.com/path;", "https://example.com/path"),
            ("https://example.com;;", "https://example.com"),
            ("https://example.com.,", "https://example.com"),
            ("https://example.com.:;", "https://example.com"),
            ("https://example.com!.", "https://example.com!"),
            ("https://example.com/).", "https://example.com/"),
            ("https://example.com/);", "https://example.com/"),
            ("https://example.com/;)", "https://example.com/"),
            (
                "https://example.com/v1.0/api",
                "https://example.com/v1.0/api",
            ),
            ("https://192.168.1.1", "https://192.168.1.1"),
            ("https://sub.domain.com", "https://sub.domain.com"),
            (
                "https://example.com?query=value",
                "https://example.com?query=value",
            ),
            ("https://example.com?a=1&b=2", "https://example.com?a=1&b=2"),
            (
                "https://example.com/path:8080",
                "https://example.com/path:8080",
            ),
        ];

        for (input, expected) in test_cases {
            let (result, _) = trim_url_punctuation(input);
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
    }

    /// Asserts that hovering `hover_byte_offset` in `line` yields `expected_path` over
    /// `expected_link_range`.
    fn assert_path_match(
        regex: &str,
        line: &str,
        hover_byte_offset: usize,
        expected_path: &str,
        expected_link_range: StdRange<usize>,
    ) {
        let regexes = vec![Regex::new(regex).unwrap()];
        let candidates = path_hyperlink_candidates_in_line(
            line,
            hover_byte_offset,
            &regexes,
            Duration::from_millis(500),
        );
        assert_eq!(
            candidates,
            vec![(expected_path.to_string(), expected_link_range)],
            "line={line:?} hover_byte_offset={hover_byte_offset}"
        );
    }

    const PATH_LINE_COLUMN_REGEX: &str =
        r"(?<link>(?<path>\S+\.rs)(:(?<line>[0-9]+))?(:(?<column>[0-9]+))?)";

    #[test]
    fn path_only() {
        // hovering the `t` of `test`.
        assert_path_match(
            PATH_LINE_COLUMN_REGEX,
            "/test/cool.rs",
            1,
            "/test/cool.rs",
            0..13,
        );
    }

    #[test]
    fn path_and_line() {
        let line = "/test/cool.rs:4";
        assert_path_match(PATH_LINE_COLUMN_REGEX, line, 1, "/test/cool.rs:4", 0..15);
        // hovering the line number.
        assert_path_match(PATH_LINE_COLUMN_REGEX, line, 14, "/test/cool.rs:4", 0..15);
    }

    #[test]
    fn path_line_and_column() {
        // the match spans 17 bytes.
        let line = "/test/cool.rs:4:2";
        assert_path_match(PATH_LINE_COLUMN_REGEX, line, 1, "/test/cool.rs:4:2", 0..17);
        assert_path_match(PATH_LINE_COLUMN_REGEX, line, 16, "/test/cool.rs:4:2", 0..17);
    }

    #[test]
    fn colons_galore() {
        assert_path_match(
            PATH_LINE_COLUMN_REGEX,
            "/test/cool.rs:4:2:",
            1,
            "/test/cool.rs:4:2",
            0..17,
        );
    }

    #[test]
    fn multiple_same_line() {
        let line = "/test/cool.rs /test/cool.rs";
        assert_path_match(PATH_LINE_COLUMN_REGEX, line, 1, "/test/cool.rs", 0..13);
        assert_path_match(PATH_LINE_COLUMN_REGEX, line, 15, "/test/cool.rs", 14..27);
    }

    #[test]
    fn no_match_when_regex_finds_nothing_on_line() {
        let regexes = vec![Regex::new(PATH_LINE_COLUMN_REGEX).unwrap()];
        let candidates = path_hyperlink_candidates_in_line(
            "no path here",
            3,
            &regexes,
            Duration::from_millis(500),
        );
        assert!(candidates.is_empty());
    }

    #[test]
    fn no_candidates_when_hover_outside_any_match() {
        // hovering past the end of the only match.
        let line = "/test/cool.rs   ";
        let regexes = vec![Regex::new(PATH_LINE_COLUMN_REGEX).unwrap()];
        let candidates = path_hyperlink_candidates_in_line(
            line,
            line.len() - 1,
            &regexes,
            Duration::from_millis(500),
        );
        assert!(candidates.is_empty());
    }
}
