use chrono::{Datelike, Local};
use editor::MultiBufferSnapshot;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{Instant, SystemTime};
use typst::diag::{FileError, FileResult};
use typst::foundations::{Datetime, Duration};
use typst::syntax::package::PackageSpec;
use typst::syntax::{FileId, RootedPath, Source, VirtualPath, VirtualRoot};
use typst::text::{Font, FontBook};
use typst::utils::LazyHash;
use typst::{Library, LibraryExt, World};
use ui::SharedString;
use util::ResultExt as _;
use util::rel_path::RelPath;
use worktree::{RemoteFileLoader, Snapshot, TypstPackage};

/// Where the project files of a Typst document are read from.
///
/// Packages of remote projects are read from the remote host when it has them installed, and
/// otherwise from the local package directories.
#[derive(Clone)]
pub enum ProjectFiles {
    Local,
    Remote(RemoteProjectFiles),
}

/// Files of a remote worktree, addressed by their worktree-relative paths.
pub trait RemoteFileSource: Send + Sync {
    fn read_file(&self, path: &RelPath) -> anyhow::Result<Vec<u8>>;

    /// Returns the modification time and size of a file without downloading it.
    fn file_metadata(&self, path: &RelPath) -> anyhow::Result<(Option<SystemTime>, u64)>;

    /// Reads `path` from `package` installed on the remote host.
    fn read_package_file(&self, package: &PackageSpec, path: &RelPath) -> anyhow::Result<Vec<u8>>;

    /// Returns the modification time and size of `path` in `package`, failing if the remote host
    /// does not have the package installed.
    fn package_file_metadata(
        &self,
        package: &PackageSpec,
        path: &RelPath,
    ) -> anyhow::Result<(Option<SystemTime>, u64)>;
}

// compiles run on a background thread, which can wait for the network.
impl RemoteFileSource for RemoteFileLoader {
    fn read_file(&self, path: &RelPath) -> anyhow::Result<Vec<u8>> {
        futures::executor::block_on(self.load(path))
    }

    fn file_metadata(&self, path: &RelPath) -> anyhow::Result<(Option<SystemTime>, u64)> {
        let (mtime, size) = futures::executor::block_on(self.metadata(path))?;
        Ok((mtime.map(|mtime| mtime.timestamp_for_user()), size))
    }

    fn read_package_file(&self, package: &PackageSpec, path: &RelPath) -> anyhow::Result<Vec<u8>> {
        let version = package.version.to_string();
        let package = TypstPackage {
            namespace: &package.namespace,
            name: &package.name,
            version: &version,
        };
        futures::executor::block_on(self.load_typst_package_file(&package, path))
    }

    fn package_file_metadata(
        &self,
        package: &PackageSpec,
        path: &RelPath,
    ) -> anyhow::Result<(Option<SystemTime>, u64)> {
        let version = package.version.to_string();
        let package = TypstPackage {
            namespace: &package.namespace,
            name: &package.name,
            version: &version,
        };
        let (mtime, size) =
            futures::executor::block_on(self.typst_package_file_metadata(&package, path))?;
        Ok((mtime.map(|mtime| mtime.timestamp_for_user()), size))
    }
}

/// Files of a remote project, downloaded when a compile needs them.
#[derive(Clone)]
pub struct RemoteProjectFiles {
    pub source: Arc<dyn RemoteFileSource>,
    /// absolute path of the worktree root on the remote host.
    pub worktree_root: PathBuf,
    /// worktree entries, whose modification times decide when a file is downloaded again.
    pub snapshot: Snapshot,
    /// whether the remote host has a package installed, asked once per compile since the files
    /// are replaced before every compile.
    remote_packages: Arc<parking_lot::Mutex<HashMap<PackageSpec, bool>>>,
}

impl RemoteProjectFiles {
    pub fn new(loader: RemoteFileLoader, worktree_root: PathBuf, snapshot: Snapshot) -> Self {
        Self::with_source(Arc::new(loader), worktree_root, snapshot)
    }

    pub fn with_source(
        source: Arc<dyn RemoteFileSource>,
        worktree_root: PathBuf,
        snapshot: Snapshot,
    ) -> Self {
        Self {
            source,
            worktree_root,
            snapshot,
            remote_packages: Default::default(),
        }
    }

    /// Returns whether the remote host has `package` installed, like private `@local` packages
    /// that only exist next to the project.
    fn has_package(&self, package: &PackageSpec) -> bool {
        *self
            .remote_packages
            .lock()
            .entry(package.clone())
            .or_insert_with(|| {
                RelPath::from_unix_str("typst.toml")
                    .and_then(|manifest| self.source.package_file_metadata(package, manifest))
                    .map_err(|error| {
                        log::debug!("package {package} is not on the remote: {error:#}")
                    })
                    .is_ok()
            })
    }

    /// Returns the modification time and size of `rel_path`, if it is a file.
    ///
    /// Files in directories the worktree has not scanned, like ignored build directories with
    /// generated images, are looked up on the remote. Anything else missing from the snapshot
    /// does not exist, which avoids a request for every file the project resolution probes.
    fn metadata(&self, rel_path: &RelPath) -> Option<(Option<SystemTime>, u64)> {
        if let Some(entry) = self.snapshot.entry_for_path(rel_path) {
            return entry.is_file().then(|| {
                (
                    entry.mtime.map(|mtime| mtime.timestamp_for_user()),
                    entry.size,
                )
            });
        }
        let in_unscanned_dir = rel_path
            .ancestors()
            .skip(1)
            .find_map(|ancestor| self.snapshot.entry_for_path(ancestor))
            .is_some_and(|ancestor| ancestor.kind.is_unloaded());
        if !in_unscanned_dir {
            return None;
        }
        self.source
            .file_metadata(rel_path)
            .map_err(|error| log::debug!("no remote file {rel_path:?}: {error:#}"))
            .ok()
    }

    fn rel_path(&self, path: &Path) -> Option<Arc<RelPath>> {
        let relative = path.strip_prefix(&self.worktree_root).ok()?;
        Some(
            RelPath::new(relative, self.snapshot.path_style())
                .ok()?
                .into_arc(),
        )
    }

    fn read_rel(&self, rel_path: &RelPath) -> FileResult<Vec<u8>> {
        self.source
            .read_file(rel_path)
            .map_err(|error| FileError::Other(Some(format!("{error:#}").into())))
    }
}

impl ProjectFiles {
    /// Returns the modification time and size of the file at `path`, if it exists.
    fn metadata(&self, path: &Path) -> Option<(Option<SystemTime>, u64)> {
        match self {
            ProjectFiles::Local => {
                let metadata = std::fs::metadata(path).ok()?;
                Some((metadata.modified().ok(), metadata.len()))
            }
            ProjectFiles::Remote(remote) => remote.metadata(&remote.rel_path(path)?),
        }
    }

    fn read(&self, path: &Path) -> FileResult<Vec<u8>> {
        match self {
            ProjectFiles::Local => {
                std::fs::read(path).map_err(|_| FileError::NotFound(path.to_path_buf()))
            }
            ProjectFiles::Remote(remote) => {
                let rel_path = remote
                    .rel_path(path)
                    .ok_or_else(|| FileError::NotFound(path.to_path_buf()))?;
                remote.read_rel(&rel_path)
            }
        }
    }

    pub fn exists(&self, path: &Path) -> bool {
        self.metadata(path).is_some()
    }

    pub fn read_to_string(&self, path: &Path) -> FileResult<String> {
        String::from_utf8(self.read(path)?).map_err(|_| FileError::InvalidUtf8)
    }

    pub fn is_remote(&self) -> bool {
        matches!(self, ProjectFiles::Remote(_))
    }
}

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
    /// replaced before every compile, so a remote snapshot reflects the latest changes.
    pub files: ProjectFiles,
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

/// Directories below a project root whose fonts are loaded, including their subdirectories.
const PROJECT_FONT_DIRS: &[&str] = &["fonts", "assets/fonts"];

fn is_font_file(path: &Path) -> bool {
    path.extension().is_some_and(|extension| {
        ["ttf", "otf", "ttc", "otc"]
            .iter()
            .any(|font_extension| extension.eq_ignore_ascii_case(font_extension))
    })
}

/// Loads the fonts of a project: font files directly in `root` and everything in
/// [`PROJECT_FONT_DIRS`].
///
/// The root itself is not scanned recursively, since it falls back to the directory of the
/// previewed file, which can be the home directory with cloud drives and whole libraries below it.
fn load_project_fonts(db: &mut fontdb::Database, root: &Path) {
    match std::fs::read_dir(root) {
        Ok(entries) => {
            for entry in entries.flatten() {
                let path = entry.path();
                if is_font_file(&path) && path.is_file() {
                    db.load_font_file(&path).log_err();
                }
            }
        }
        Err(error) => log::debug!("failed to list project fonts in {root:?}: {error}"),
    }
    for font_dir in PROJECT_FONT_DIRS {
        let font_dir = root.join(font_dir);
        if font_dir.is_dir() {
            db.load_fonts_dir(font_dir);
        }
    }
}

/// Downloads the fonts of a remote project from the same places [`load_project_fonts`] looks.
fn load_remote_project_fonts(
    remote: &RemoteProjectFiles,
    root: &Path,
    book: &mut FontBook,
    fonts: &mut Vec<FontSlot>,
) {
    let Some(root) = remote.rel_path(root) else {
        return;
    };
    let mut font_paths = remote
        .snapshot
        .child_entries(&root)
        .filter(|entry| entry.is_file() && is_font_file(entry.path.as_std_path()))
        .map(|entry| entry.path.clone())
        .collect::<Vec<_>>();
    for font_dir in PROJECT_FONT_DIRS {
        let Ok(font_dir) = RelPath::from_unix_str(font_dir) else {
            continue;
        };
        let font_dir = root.join(font_dir);
        font_paths.extend(
            remote
                .snapshot
                .traverse_from_path(true, false, true, &font_dir)
                .take_while(|entry| entry.path.starts_with(&font_dir))
                .filter(|entry| is_font_file(entry.path.as_std_path()))
                .map(|entry| entry.path.clone()),
        );
    }

    for font_path in font_paths {
        match remote.read_rel(&font_path) {
            Ok(data) => {
                for font in Font::iter(typst::foundations::Bytes::new(data)) {
                    book.push(font.info().clone());
                    fonts.push(FontSlot::Memory(font));
                }
            }
            Err(error) => log::debug!("failed to download project font {font_path:?}: {error:?}"),
        }
    }
}

impl TypstSystemWorld {
    pub fn new(root: PathBuf, files: ProjectFiles) -> Self {
        let start = Instant::now();

        // bundled and system fonts are cached process-wide; only project fonts depend on `root`.
        let (static_book, static_font_slots) = static_fonts();
        let mut book = static_book.clone();
        let mut fonts = static_font_slots.clone();

        match &files {
            ProjectFiles::Local => {
                let mut project_db = fontdb::Database::new();
                load_project_fonts(&mut project_db, &root);
                collect_fontdb_faces(&project_db, &mut book, &mut fonts);
            }
            ProjectFiles::Remote(remote) => {
                load_remote_project_fonts(remote, &root, &mut book, &mut fonts);
            }
        }

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
            files,
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
    fn read_disk_file(
        &self,
        id: FileId,
        path: &Path,
    ) -> FileResult<(typst::foundations::Bytes, bool)> {
        let remote_package = match (id.root(), &self.files) {
            (VirtualRoot::Package(package), ProjectFiles::Remote(remote))
                if remote.has_package(package) =>
            {
                let package_path = RelPath::from_unix_str(id.vpath().get_without_slash())
                    .map_err(|_| FileError::NotFound(path.to_path_buf()))?;
                Some((remote, package, package_path))
            }
            _ => None,
        };
        let local_files = ProjectFiles::Local;
        let files = match id.root() {
            VirtualRoot::Package(_) => &local_files,
            VirtualRoot::Project => &self.files,
        };
        let metadata = match &remote_package {
            Some((remote, package, package_path)) => remote
                .source
                .package_file_metadata(package, package_path)
                .ok(),
            None => files.metadata(path),
        };
        let (modified, len) = metadata.ok_or_else(|| FileError::NotFound(path.to_path_buf()))?;
        if let Some(file) = self.disk_files.read().get(path)
            && file.modified.is_some()
            && file.modified == modified
            && file.len == len
        {
            return Ok((file.bytes.clone(), false));
        }

        let bytes = match &remote_package {
            Some((remote, package, package_path)) => remote
                .source
                .read_package_file(package, package_path)
                .map_err(|error| FileError::Other(Some(format!("{error:#}").into())))?,
            None => files.read(path)?,
        };
        let bytes = typst::foundations::Bytes::new(bytes);
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
            VirtualRoot::Package(package) => match &self.files {
                // stands in for the package directory on the remote host, named like the Typst
                // CLI names package files in diagnostics.
                ProjectFiles::Remote(remote) if remote.has_package(package) => {
                    PathBuf::from(package.to_string())
                }
                _ => package_dir(package).ok_or_else(|| {
                    if package.namespace == "preview" {
                        self.missing_packages.lock().insert(package.clone());
                    }
                    FileError::Package(typst::diag::PackageError::NotFound(package.clone()))
                })?,
            },
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
            let (bytes, changed) = self.read_disk_file(id, &path)?;
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
        let (bytes, _changed) = self.read_disk_file(id, &path)?;
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
    paths::typst_package_dirs()
        .into_iter()
        .map(|root| root.join(&subdir))
        .find(|path| path.exists())
}

/// Returns where a downloaded package is stored, following the Typst CLI, so both share their
/// package cache.
pub fn package_download_dir(spec: &PackageSpec) -> Option<PathBuf> {
    Some(paths::typst_package_cache_dir()?.join(package_subdir(spec)))
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
        let world = TypstSystemWorld::new(dir.0.clone(), ProjectFiles::Local);
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
        let mut world = TypstSystemWorld::new(dir.0.clone(), ProjectFiles::Local);
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

        let world = TypstSystemWorld::new(dir.0.clone(), ProjectFiles::Local);

        assert!(
            world.fonts.len() > baseline,
            "expected the project-local font ({}) to be added on top of the {baseline} \
             cached static fonts, got {}",
            dir.0.join("local.ttf").display(),
            world.fonts.len(),
        );
    }

    #[test]
    fn project_fonts_come_from_font_directories_but_not_every_subdirectory() {
        let (_, static_font_slots) = static_fonts();
        let baseline = static_font_slots.len();
        let font_bytes = typst_assets::fonts()
            .next()
            .expect("typst_assets ships at least one bundled font");

        let dir = ScratchDir::new("typst_world_font_dir_test");
        std::fs::create_dir_all(dir.0.join("fonts/display")).unwrap();
        std::fs::write(dir.0.join("fonts/display/heading.ttf"), font_bytes).unwrap();
        assert!(
            TypstSystemWorld::new(dir.0.clone(), ProjectFiles::Local)
                .fonts
                .len()
                > baseline
        );

        // a root that falls back to a large directory, such as the home directory, must not
        // load fonts from unrelated subdirectories.
        let dir = ScratchDir::new("typst_world_unrelated_font_test");
        std::fs::create_dir_all(dir.0.join("Books/novel/fonts")).unwrap();
        std::fs::write(dir.0.join("Books/novel/fonts/body.otf"), font_bytes).unwrap();
        assert_eq!(
            TypstSystemWorld::new(dir.0.clone(), ProjectFiles::Local)
                .fonts
                .len(),
            baseline
        );
    }

    #[test]
    fn font_scan_is_shared_across_different_project_roots() {
        let dir_a = ScratchDir::new("typst_world_shared_fonts_test_a");
        let dir_b = ScratchDir::new("typst_world_shared_fonts_test_b");

        let world_a = TypstSystemWorld::new(dir_a.0.clone(), ProjectFiles::Local);
        let world_b = TypstSystemWorld::new(dir_b.0.clone(), ProjectFiles::Local);

        let (_, static_font_slots) = static_fonts();
        // without project fonts, both worlds reuse the cached static font set.
        assert_eq!(world_a.fonts.len(), static_font_slots.len());
        assert_eq!(world_b.fonts.len(), static_font_slots.len());
    }
}
