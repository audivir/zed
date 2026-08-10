use chrono::{Datelike, Local};
use editor::MultiBufferSnapshot;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;
use typst::diag::{FileError, FileResult};
use typst::foundations::Datetime;
use typst::syntax::{FileId, Source, VirtualPath};
use typst::text::{Font, FontBook};
use typst::utils::LazyHash;
use typst::{Library, LibraryExt, World};
use ui::SharedString;

/// A lazy-loader for fonts to prevent loading gigabytes of font data into RAM at startup.
#[derive(Clone)]
pub enum FontSlot {
    Memory(Font),
    Disk {
        path: PathBuf,
        index: u32,
        font: OnceLock<Option<Font>>,
    },
}

impl FontSlot {
    pub fn get(&self) -> Option<Font> {
        match self {
            FontSlot::Memory(f) => Some(f.clone()),
            FontSlot::Disk { path, index, font } => font
                .get_or_init(|| {
                    let data = std::fs::read(path).ok()?;
                    Font::new(typst::foundations::Bytes::new(data), *index)
                })
                .clone(),
        }
    }
}

/// Reads `FontInfo` out of every face `db` knows about and records a lazy [`FontSlot`] for
/// each, appending both into `book`/`fonts`.
fn collect_fontdb_faces(db: &fontdb::Database, book: &mut FontBook, fonts: &mut Vec<FontSlot>) {
    for face in db.faces() {
        let path = match &face.source {
            fontdb::Source::File(path) | fontdb::Source::SharedFile(path, _) => path.clone(),
            _ => continue,
        };

        // Read the file once to extract Typst's FontInfo, then immediately drop the bytes.
        // The FontSlot will reload the bytes lazily if Typst actually tries to render it.
        if let Ok(data) = std::fs::read(&path) {
            let bytes = typst::foundations::Bytes::new(data);
            if let Some(font) = typst::text::Font::new(bytes, face.index) {
                book.push(font.info().clone());
                fonts.push(FontSlot::Disk {
                    path,
                    index: face.index,
                    font: OnceLock::new(),
                });
            }
        }
    }
}

/// Typst's bundled asset fonts plus every font installed on the system. Neither depends on
/// the active project root, so scanning them (parsing every bundled font's metadata, walking
/// every system font directory) is done once per process instead of on every project-root
/// switch. `FontBook` and `FontSlot` are both cheap to clone (metadata + `Arc`-backed fonts),
/// so callers just clone this and layer project-local fonts on top.
fn static_fonts() -> &'static (FontBook, Vec<FontSlot>) {
    static STATIC_FONTS: OnceLock<(FontBook, Vec<FontSlot>)> = OnceLock::new();

    STATIC_FONTS.get_or_init(|| {
        let start = Instant::now();

        let mut book = FontBook::new();
        let mut fonts = Vec::new();

        for data in typst_assets::fonts() {
            let bytes = typst::foundations::Bytes::new(data.to_vec());
            for font in typst::text::Font::iter(bytes) {
                book.push(font.info().clone());
                fonts.push(FontSlot::Memory(font));
            }
        }

        let mut db = fontdb::Database::new();
        db.load_system_fonts();
        collect_fontdb_faces(&db, &mut book, &mut fonts);

        log::info!(
            "Scanned {} bundled + system fonts in {:?}",
            fonts.len(),
            start.elapsed()
        );

        (book, fonts)
    })
}

/// A Typst World implementation that keeps fonts and the main buffer in memory,
/// but dynamically resolves includes and images from the physical workspace directory.
pub struct TypstSystemWorld {
    library: LazyHash<Library>,
    book: LazyHash<FontBook>,
    fonts: Vec<FontSlot>,
    main_id: FileId,
    main_source: Source,
    pub root: PathBuf,
    pub file_overrides: HashMap<PathBuf, MultiBufferSnapshot>,
    pub sources: RwLock<HashMap<FileId, Source>>,
}

impl TypstSystemWorld {
    pub fn new(root: PathBuf) -> Self {
        let start = Instant::now();

        // Bundled + system fonts are cached process-wide (see `static_fonts`); only the
        // project-local font directory actually depends on `root` and needs rescanning here.
        let (static_book, static_font_slots) = static_fonts();
        let mut book = static_book.clone();
        let mut fonts = static_font_slots.clone();

        let mut project_db = fontdb::Database::new();
        project_db.load_fonts_dir(&root);
        collect_fontdb_faces(&project_db, &mut book, &mut fonts);

        let main_id = FileId::new(None, VirtualPath::new("/main.typ"));

        let world = Self {
            library: LazyHash::new(Library::default()),
            book: LazyHash::new(book),
            fonts,
            main_id,
            main_source: Source::new(main_id, String::new()),
            root,
            file_overrides: Default::default(),
            sources: RwLock::new(Default::default()),
        };

        log::info!("Initialized TypstWorld in {:?}", start.elapsed());

        world
    }

    /// Update the main source file dynamically
    pub fn update_main(&mut self, text: SharedString, file_path: &Path) {
        let rel_path = file_path
            .strip_prefix(&self.root)
            .unwrap_or(Path::new("main.typ"));
        let mut path_str = rel_path.to_string_lossy().replace("\\", "/");
        if !path_str.starts_with('/') {
            path_str.insert(0, '/');
        }
        let vpath = typst::syntax::VirtualPath::new(path_str);
        let new_id = typst::syntax::FileId::new(None, vpath);

        if self.main_id == new_id {
            self.main_source.replace(text.as_ref());
        } else {
            self.main_id = new_id;
            self.main_source = Source::new(new_id, text.to_string());
        }
    }

    /// Provides access to the main source file
    pub fn main_source(&self) -> &Source {
        &self.main_source
    }

    pub fn resolve_path(&self, id: FileId) -> FileResult<PathBuf> {
        if let Some(package) = id.package() {
            let package_dir = package_dir(package).ok_or_else(|| {
                FileError::Package(typst::diag::PackageError::NotFound(package.clone()))
            })?;
            id.vpath()
                .resolve(&package_dir)
                .ok_or_else(|| FileError::NotFound(id.vpath().as_rootless_path().into()))
        } else {
            id.vpath()
                .resolve(&self.root)
                .ok_or_else(|| FileError::NotFound(id.vpath().as_rootless_path().into()))
        }
    }
}

impl World for TypstSystemWorld {
    fn library(&self) -> &LazyHash<Library> {
        &self.library
    }

    fn book(&self) -> &LazyHash<FontBook> {
        &self.book
    }

    fn main(&self) -> FileId {
        self.main_id
    }

    fn source(&self, id: FileId) -> FileResult<Source> {
        // 1. Serve the active unsaved editor buffer
        if id == self.main_id {
            return Ok(self.main_source.clone());
        }

        // 2. Resolve `#import` paths to the real disk (including packages)
        let path = self.resolve_path(id)?;

        let text = if let Some(override_text) = self.file_overrides.get(&path) {
            override_text.text()
        } else {
            std::fs::read_to_string(&path).map_err(|_| FileError::NotFound(path.clone()))?
        };

        let mut sources = self.sources.write();
        if let Some(source) = sources.get_mut(&id) {
            if source.text() != text {
                log::info!("Updating source: {:?}", path);
                source.replace(&text);
            }
            Ok(source.clone())
        } else {
            log::info!("Adding source: {:?}", path);
            let source = Source::new(id, text);
            sources.insert(id, source.clone());
            Ok(source)
        }
    }

    fn file(&self, id: FileId) -> FileResult<typst::foundations::Bytes> {
        let path = self.resolve_path(id)?;

        let bytes = std::fs::read(&path).map_err(|_| FileError::NotFound(path.clone()))?;

        Ok(typst::foundations::Bytes::new(bytes))
    }

    fn font(&self, index: usize) -> Option<Font> {
        self.fonts.get(index).and_then(|slot| slot.get())
    }

    fn today(&self, offset: Option<i64>) -> Option<Datetime> {
        let now = Local::now();
        let naive = match offset {
            None => now.naive_local(),
            Some(o) => {
                // `o` comes from user-controlled Typst source
                // (`datetime.today(offset: ...)`); use checked arithmetic so an
                // out-of-range offset falls through to `None` instead of
                // overflowing.
                let seconds = i32::try_from(o.checked_mul(3600)?).ok()?;
                let tz = chrono::FixedOffset::east_opt(seconds)?;
                now.with_timezone(&tz).naive_local()
            }
        };

        Datetime::from_ymd(
            naive.year(),
            naive.month().try_into().ok()?,
            naive.day().try_into().ok()?,
        )
    }
}

impl typst_ide::IdeWorld for TypstSystemWorld {
    fn upcast(&self) -> &dyn World {
        self
    }
}

fn package_dir(spec: &typst::syntax::package::PackageSpec) -> Option<PathBuf> {
    let subdir = format!("{}/{}/{}", spec.namespace, spec.name, spec.version);

    if let Ok(cache_dir) = std::env::var("TYPST_PACKAGE_CACHE_DIR") {
        let p = PathBuf::from(cache_dir).join(&subdir);
        if p.exists() {
            return Some(p);
        }
    }

    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()?;

    let mut paths = Vec::new();
    if cfg!(target_os = "macos") {
        paths.push(format!("{home}/Library/Caches/typst/packages"));
        paths.push(format!("{home}/Library/Application Support/typst/packages"));
    } else if cfg!(target_os = "windows") {
        paths.push(
            std::env::var("LOCALAPPDATA")
                .ok()
                .unwrap_or_else(|| format!("{home}/AppData/Local"))
                + "/typst/packages",
        );
        paths.push(
            std::env::var("APPDATA")
                .ok()
                .unwrap_or_else(|| format!("{home}/AppData/Roaming"))
                + "/typst/packages",
        );
    } else {
        paths.push(
            std::env::var("XDG_CACHE_HOME").unwrap_or_else(|_| format!("{home}/.cache"))
                + "/typst/packages",
        );
        paths.push(
            std::env::var("XDG_DATA_HOME").unwrap_or_else(|_| format!("{home}/.local/share"))
                + "/typst/packages",
        );
    }

    for dir in paths {
        let p = PathBuf::from(dir).join(&subdir);
        if p.exists() {
            return Some(p);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch directory that removes itself on drop, so a failed assertion doesn't leave
    /// stray fixture files behind in the OS temp dir.
    struct ScratchDir(PathBuf);

    impl ScratchDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!("{name}_{}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for ScratchDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    #[test]
    fn project_local_fonts_are_added_on_top_of_the_cached_static_set() {
        let (_, static_font_slots) = static_fonts();
        let baseline = static_font_slots.len();

        let dir = ScratchDir::new("typst_world_local_font_test");
        let font_bytes = typst_assets::fonts()
            .next()
            .expect("typst_assets ships at least one bundled font");
        std::fs::write(dir.0.join("local.ttf"), font_bytes).unwrap();

        let world = TypstSystemWorld::new(dir.0.clone());

        assert!(
            world.fonts.len() > baseline,
            "expected the project-local font ({}) to be added on top of the {baseline} \
             cached static fonts, got {}",
            dir.0.join("local.ttf").display(),
            world.fonts.len(),
        );
    }

    #[test]
    fn font_scan_is_shared_across_different_project_roots() {
        let dir_a = ScratchDir::new("typst_world_shared_fonts_test_a");
        let dir_b = ScratchDir::new("typst_world_shared_fonts_test_b");

        let world_a = TypstSystemWorld::new(dir_a.0.clone());
        let world_b = TypstSystemWorld::new(dir_b.0.clone());

        let (_, static_font_slots) = static_fonts();
        // Neither scratch dir has its own fonts, so both worlds should end up with exactly
        // the cached static font set, confirming `new()` reuses `static_fonts()` rather
        // than re-scanning the system from scratch for every project root.
        assert_eq!(world_a.fonts.len(), static_font_slots.len());
        assert_eq!(world_b.fonts.len(), static_font_slots.len());
    }
}
