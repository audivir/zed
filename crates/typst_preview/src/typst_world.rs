use chrono::{Datelike, Local};
use editor::MultiBufferSnapshot;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Instant, SystemTime};
use typst::diag::{FileError, FileResult};
use typst::foundations::{Datetime, Duration};
use typst::syntax::package::PackageSpec;
use typst::syntax::{FileId, RootedPath, Source, VirtualPath, VirtualRoot};
use typst::text::{Font, FontBook};
use typst::utils::LazyHash;
use typst::{Library, LibraryExt, World};
use ui::SharedString;

/// Lazily loaded font, avoiding reading all font data at startup.
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

/// Records font info and a lazy [`FontSlot`] for every face in `db`.
fn collect_fontdb_faces(db: &fontdb::Database, book: &mut FontBook, fonts: &mut Vec<FontSlot>) {
    for face in db.faces() {
        let path = match &face.source {
            fontdb::Source::File(path) | fontdb::Source::SharedFile(path, _) => path.clone(),
            _ => continue,
        };

        // read once for the font info and drop the bytes; `FontSlot` reloads them on demand.
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

/// Returns bundled Typst fonts plus installed system fonts, scanned once per process.
///
/// Both are independent of the project root and cheap to clone, so callers layer
/// project-local fonts on top of a clone.
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

        log::debug!(
            "Scanned {} bundled + system fonts in {:?}",
            fonts.len(),
            start.elapsed()
        );

        (book, fonts)
    })
}

/// Typst world that keeps fonts and the main buffer in memory.
///
/// Includes and images resolve from the workspace directory on disk.
pub struct TypstSystemWorld {
    library: LazyHash<Library>,
    book: LazyHash<FontBook>,
    fonts: Vec<FontSlot>,
    main_id: FileId,
    main_source: Source,
    pub root: PathBuf,
    pub file_overrides: HashMap<PathBuf, MultiBufferSnapshot>,
    pub sources: RwLock<HashMap<FileId, Source>>,
    disk_files: RwLock<HashMap<PathBuf, DiskFile>>,
    /// `@preview` packages that imports asked for but that are not installed locally.
    missing_packages: parking_lot::Mutex<HashSet<PackageSpec>>,
}

/// Container for file bytes read from disk, keyed by their modification time and size.
struct DiskFile {
    modified: Option<SystemTime>,
    len: u64,
    bytes: typst::foundations::Bytes,
}

impl TypstSystemWorld {
    pub fn new(root: PathBuf) -> Self {
        let start = Instant::now();

        // bundled and system fonts are cached process-wide; only project fonts depend on `root`.
        let (static_book, static_font_slots) = static_fonts();
        let mut book = static_book.clone();
        let mut fonts = static_font_slots.clone();

        let mut project_db = fontdb::Database::new();
        project_db.load_fonts_dir(&root);
        collect_fontdb_faces(&project_db, &mut book, &mut fonts);

        let main_id = FileId::new(RootedPath::new(
            VirtualRoot::Project,
            VirtualPath::new("/main.typ").expect("hardcoded path is always valid"),
        ));

        let world = Self {
            library: LazyHash::new(Library::default()),
            book: LazyHash::new(book),
            fonts,
            main_id,
            main_source: Source::new(main_id, String::new()),
            root,
            file_overrides: Default::default(),
            sources: RwLock::new(Default::default()),
            disk_files: RwLock::new(Default::default()),
            missing_packages: Default::default(),
        };

        log::debug!("Initialized TypstWorld in {:?}", start.elapsed());

        world
    }

    /// Returns the bytes of `path`, re-reading only when its modification time or size changed.
    ///
    /// The flag reports whether the file was re-read.
    fn read_disk_file(&self, path: &Path) -> FileResult<(typst::foundations::Bytes, bool)> {
        let not_found = || FileError::NotFound(path.to_path_buf());
        let metadata = std::fs::metadata(path).map_err(|_| not_found())?;
        let modified = metadata.modified().ok();
        let len = metadata.len();
        if let Some(file) = self.disk_files.read().get(path)
            && file.modified.is_some()
            && file.modified == modified
            && file.len == len
        {
            return Ok((file.bytes.clone(), false));
        }

        let bytes = typst::foundations::Bytes::new(std::fs::read(path).map_err(|_| not_found())?);
        self.disk_files.write().insert(
            path.to_path_buf(),
            DiskFile {
                modified,
                len,
                bytes: bytes.clone(),
            },
        );
        Ok((bytes, true))
    }

    /// Returns whether a compile read `path` from disk.
    pub fn has_read(&self, path: &Path) -> bool {
        self.disk_files.read().contains_key(path)
    }

    /// Replaces the main source with the given editor text.
    pub fn update_main(&mut self, text: SharedString, file_path: &Path) {
        let rel_path = file_path
            .strip_prefix(&self.root)
            .unwrap_or(Path::new("main.typ"));
        let mut path_str = rel_path.to_string_lossy().replace("\\", "/");
        if !path_str.starts_with('/') {
            path_str.insert(0, '/');
        }
        let new_id = match typst::syntax::VirtualPath::new(&path_str) {
            Ok(vpath) => typst::syntax::FileId::new(RootedPath::new(VirtualRoot::Project, vpath)),
            Err(error) => {
                log::warn!(
                    "Invalid Typst virtual path {path_str:?}: {error}; keeping previous main file"
                );
                self.main_id
            }
        };

        if self.main_id == new_id {
            self.main_source.replace(text.as_ref());
        } else {
            self.main_id = new_id;
            self.main_source = Source::new(new_id, text.to_string());
        }
    }

    pub fn main_source(&self) -> &Source {
        &self.main_source
    }

    /// Returns the `@preview` packages that the last compilations could not find locally.
    pub fn take_missing_packages(&self) -> Vec<PackageSpec> {
        self.missing_packages.lock().drain().collect()
    }

    pub fn resolve_path(&self, id: FileId) -> FileResult<PathBuf> {
        let root = match id.root() {
            VirtualRoot::Package(package) => package_dir(package).ok_or_else(|| {
                if package.namespace == "preview" {
                    self.missing_packages.lock().insert(package.clone());
                }
                FileError::Package(typst::diag::PackageError::NotFound(package.clone()))
            })?,
            VirtualRoot::Project => self.root.clone(),
        };
        id.vpath()
            .realize(&root)
            .map_err(|_| FileError::NotFound(Path::new(id.vpath().get_without_slash()).into()))
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
        // serve the unsaved editor buffer.
        if id == self.main_id {
            return Ok(self.main_source.clone());
        }

        // resolve imports, including packages, from disk.
        let path = self.resolve_path(id)?;

        let text = if let Some(override_text) = self.file_overrides.get(&path) {
            // forget the disk state so the file is re-read once the override disappears.
            self.disk_files.write().remove(&path);
            override_text.text()
        } else {
            let (bytes, changed) = self.read_disk_file(&path)?;
            if !changed && let Some(source) = self.sources.read().get(&id) {
                return Ok(source.clone());
            }
            std::str::from_utf8(&bytes)
                .map_err(|_| FileError::InvalidUtf8)?
                .to_string()
        };

        let mut sources = self.sources.write();
        if let Some(source) = sources.get_mut(&id) {
            if source.text() != text {
                log::debug!("Updating source: {:?}", path);
                source.replace(&text);
            }
            Ok(source.clone())
        } else {
            log::debug!("Adding source: {:?}", path);
            let source = Source::new(id, text);
            sources.insert(id, source.clone());
            Ok(source)
        }
    }

    fn file(&self, id: FileId) -> FileResult<typst::foundations::Bytes> {
        let path = self.resolve_path(id)?;
        let (bytes, _changed) = self.read_disk_file(&path)?;
        Ok(bytes)
    }

    fn font(&self, index: usize) -> Option<Font> {
        self.fonts.get(index).and_then(|slot| slot.get())
    }

    fn today(&self, offset: Option<Duration>) -> Option<Datetime> {
        let now = Local::now();
        let naive = match offset {
            None => now.naive_local(),
            Some(offset) => {
                // `offset` comes from user-controlled Typst source, so overflow yields `None`.
                let seconds = offset.seconds();
                if !seconds.is_finite() {
                    return None;
                }
                let seconds = i32::try_from(seconds as i64).ok()?;
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

fn package_dir(spec: &PackageSpec) -> Option<PathBuf> {
    let subdir = package_subdir(spec);
    package_roots()
        .into_iter()
        .map(|root| root.join(&subdir))
        .find(|path| path.exists())
}

/// Returns where a downloaded package is stored, following the Typst CLI, so both share their
/// package cache.
pub fn package_download_dir(spec: &PackageSpec) -> Option<PathBuf> {
    Some(
        package_roots()
            .into_iter()
            .next()?
            .join(package_subdir(spec)),
    )
}

/// Returns the URL of the archive of an `@preview` package on the Typst package registry.
pub fn package_download_url(spec: &PackageSpec) -> String {
    format!(
        "https://packages.typst.org/{}/{}-{}.tar.gz",
        spec.namespace, spec.name, spec.version
    )
}

fn package_subdir(spec: &PackageSpec) -> String {
    format!("{}/{}/{}", spec.namespace, spec.name, spec.version)
}

/// Returns the package directories searched in order, with the download cache first.
fn package_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(cache_dir) = std::env::var("TYPST_PACKAGE_CACHE_DIR") {
        roots.push(PathBuf::from(cache_dir));
    }

    let Some(home) = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
    else {
        return roots;
    };

    let paths = if cfg!(target_os = "macos") {
        vec![
            format!("{home}/Library/Caches/typst/packages"),
            format!("{home}/Library/Application Support/typst/packages"),
        ]
    } else if cfg!(target_os = "windows") {
        vec![
            std::env::var("LOCALAPPDATA")
                .ok()
                .unwrap_or_else(|| format!("{home}/AppData/Local"))
                + "/typst/packages",
            std::env::var("APPDATA")
                .ok()
                .unwrap_or_else(|| format!("{home}/AppData/Roaming"))
                + "/typst/packages",
        ]
    } else {
        vec![
            std::env::var("XDG_CACHE_HOME").unwrap_or_else(|_| format!("{home}/.cache"))
                + "/typst/packages",
            std::env::var("XDG_DATA_HOME").unwrap_or_else(|_| format!("{home}/.local/share"))
                + "/typst/packages",
        ]
    };
    roots.extend(paths.into_iter().map(PathBuf::from));
    roots
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scratch directory that removes itself on drop.
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
    fn disk_sources_are_reread_after_changes() {
        let dir = ScratchDir::new("typst_world_disk_cache_test");
        let path = dir.0.join("chapter.typ");
        std::fs::write(&path, "first").unwrap();
        let world = TypstSystemWorld::new(dir.0.clone());
        let id = FileId::new(RootedPath::new(
            VirtualRoot::Project,
            VirtualPath::new("/chapter.typ").unwrap(),
        ));

        assert_eq!(world.source(id).unwrap().text(), "first");
        assert_eq!(world.source(id).unwrap().text(), "first");

        std::fs::write(&path, "second version").unwrap();
        assert_eq!(world.source(id).unwrap().text(), "second version");
        assert_eq!(
            world.file(id).unwrap().as_slice(),
            b"second version".as_slice()
        );
    }

    #[test]
    fn missing_preview_packages_are_recorded_for_download() {
        let dir = ScratchDir::new("typst_world_missing_package_test");
        let mut world = TypstSystemWorld::new(dir.0.clone());
        world.update_main(
            concat!(
                "#import \"@preview/zed-surely-missing-package:0.1.2\": *\n",
                "#import \"@local/zed-surely-missing-package:0.1.2\": *\n",
            )
            .into(),
            &dir.0.join("main.typ"),
        );

        let compiled = typst::compile::<typst_layout::PagedDocument>(&world);
        assert!(compiled.output.is_err());
        let missing = world.take_missing_packages();
        assert_eq!(
            missing.iter().map(ToString::to_string).collect::<Vec<_>>(),
            ["@preview/zed-surely-missing-package:0.1.2"],
            "only the registry namespace can be downloaded"
        );
        assert!(world.take_missing_packages().is_empty());
        assert_eq!(
            package_download_url(&missing[0]),
            "https://packages.typst.org/preview/zed-surely-missing-package-0.1.2.tar.gz"
        );
        assert!(
            package_download_dir(&missing[0])
                .unwrap()
                .ends_with("preview/zed-surely-missing-package/0.1.2")
        );
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
        // without project fonts, both worlds reuse the cached static font set.
        assert_eq!(world_a.fonts.len(), static_font_slots.len());
        assert_eq!(world_b.fonts.len(), static_font_slots.len());
    }
}
