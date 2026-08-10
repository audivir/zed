use crate::pdf::Pdf;
use anyhow::{Context as _, Error, Result, anyhow};
use collections::{HashMap, HashSet, hash_map};
use futures::{StreamExt, channel::oneshot};
use gpui::{
    App, Context, Entity, EntityId, EventEmitter, Global, Subscription, Task, WeakEntity,
    prelude::*,
};
use language::{DiskState, File};
use project::{
    Project, ProjectEntryId, ProjectItem, ProjectPath,
    image_store::ImageId as PdfId,
    worktree_store::{WorktreeStore, WorktreeStoreEvent},
};
use rpc::{ErrorCode, ErrorExt as _};
use std::path::PathBuf;
use std::sync::Arc;
use util::{ResultExt, rel_path::RelPath};
use worktree::{LoadedBinaryFile, PathChange, Worktree};

#[derive(Debug)]
pub enum PdfItemEvent {
    ReloadNeeded,
    Reloaded,
    FileHandleChanged,
    MetadataUpdated,
}

impl EventEmitter<PdfItemEvent> for PdfItem {}

pub enum PdfStoreEvent {
    PdfAdded(Entity<PdfItem>),
}

impl EventEmitter<PdfStoreEvent> for PdfStore {}

pub struct PdfItem {
    pub id: PdfId,
    pub file: Arc<worktree::File>,
    pub pdf: Arc<Pdf>,
    pub stale_pdf: Option<Arc<Pdf>>,
    reload_task: Option<Task<()>>,
}

impl PdfItem {
    pub fn project_path(&self, cx: &App) -> ProjectPath {
        ProjectPath {
            worktree_id: self.file.worktree_id(cx),
            path: self.file.path().clone(),
        }
    }

    pub fn abs_path(&self, cx: &App) -> Option<PathBuf> {
        Some(self.file.as_local()?.abs_path(cx))
    }

    fn file_updated(&mut self, new_file: Arc<worktree::File>, cx: &mut Context<Self>) {
        let mut file_changed = false;

        let old_file = &self.file;
        if new_file.path() != old_file.path() {
            file_changed = true;
        }

        let old_state = old_file.disk_state();
        let new_state = new_file.disk_state();
        if old_state != new_state {
            file_changed = true;
            if matches!(new_state, DiskState::Present { .. }) {
                cx.emit(PdfItemEvent::ReloadNeeded)
            }
        }

        self.file = new_file;
        if file_changed {
            cx.emit(PdfItemEvent::FileHandleChanged);
            cx.notify();
        }
    }

    fn reload(&mut self, cx: &mut Context<Self>) -> Option<oneshot::Receiver<()>> {
        log::info!("Creating reload task");
        let local_file = self.file.as_local()?;
        let (tx, rx) = futures::channel::oneshot::channel();

        let content = local_file.load_bytes(cx);
        // Preserve the password across an on-disk reload so an externally-edited,
        // still-encrypted document doesn't reprompt the user for it.
        let password = self.pdf.password.clone();
        self.reload_task = Some(cx.spawn(async move |this, cx| {
            if let Some(bytes) = content
                .await
                .context("Failed to load pdf content")
                .log_err()
                && let Some(pdf) = create_gpui_pdf(bytes, password).await.log_err()
            {
                this.update(cx, |this, cx| {
                    this.pdf = pdf;
                    this.stale_pdf = Some(this.pdf.clone());
                    cx.emit(PdfItemEvent::Reloaded);
                })
                .log_err();
            }

            _ = tx.send(());
            log::info!("Reload task completed");
        }));
        Some(rx)
    }

    /// Retries opening this document with a user-supplied password, e.g. after
    /// `metadata.needs_password` was set because none was supplied or a prior guess was
    /// wrong. Reuses the already-loaded raw bytes instead of re-reading the file from disk.
    pub fn submit_password(&mut self, password: String, cx: &mut Context<Self>) {
        let bytes = (*self.pdf.bytes).clone();
        self.reload_task = Some(cx.spawn(async move |this, cx| {
            let pdf = Arc::new(Pdf::from_bytes(bytes, None, Some(password)).await);
            this.update(cx, |this, cx| {
                this.pdf = pdf;
                this.stale_pdf = Some(this.pdf.clone());
                cx.emit(PdfItemEvent::Reloaded);
                cx.notify();
            })
            .log_err();
        }));
    }
}

pub fn is_pdf_file(project: &Entity<Project>, path: &ProjectPath, cx: &App) -> bool {
    let ext = util::maybe!({
        let worktree_abs_path = project
            .read(cx)
            .worktree_for_id(path.worktree_id, cx)?
            .read(cx)
            .abs_path();
        path.path
            .extension()
            .or_else(|| worktree_abs_path.extension()?.to_str())
            .map(str::to_lowercase)
    });

    ext.as_deref() == Some("pdf")
}

impl ProjectItem for PdfItem {
    fn try_open(
        project: &Entity<Project>,
        path: &ProjectPath,
        cx: &mut App,
    ) -> Option<Task<Result<Entity<Self>>>> {
        if is_pdf_file(project, path, cx) {
            Some(cx.spawn({
                let path = path.clone();
                let project = project.clone();
                async move |cx| cx.update(|cx| open_pdf(project, path, cx)).await
            }))
        } else {
            None
        }
    }

    fn entry_id(&self, _: &App) -> Option<ProjectEntryId> {
        self.file.entry_id
    }

    fn project_path(&self, cx: &App) -> Option<ProjectPath> {
        Some(self.project_path(cx))
    }

    fn is_dirty(&self) -> bool {
        false
    }
}

trait PdfStoreImpl {
    fn open_pdf(
        &self,
        path: Arc<RelPath>,
        worktree: Entity<Worktree>,
        cx: &mut Context<PdfStore>,
    ) -> Task<Result<Entity<PdfItem>>>;

    fn reload_pdfs(
        &self,
        images: HashSet<Entity<PdfItem>>,
        cx: &mut Context<PdfStore>,
    ) -> Task<Result<()>>;

    fn as_local(&self) -> Option<Entity<LocalPdfStore>>;
}

struct LocalPdfStore {
    local_pdf_ids_by_path: HashMap<ProjectPath, PdfId>,
    local_pdf_ids_by_entry_id: HashMap<ProjectEntryId, PdfId>,
    pdf_store: WeakEntity<PdfStore>,
    _subscription: Subscription,
}

pub struct PdfStore {
    state: Box<dyn PdfStoreImpl>,
    opened_pdfs: HashMap<PdfId, WeakEntity<PdfItem>>,
    worktree_store: Entity<WorktreeStore>,
    #[allow(clippy::type_complexity)]
    loading_pdfs_by_path:
        HashMap<ProjectPath, postage::watch::Receiver<Option<Result<Entity<PdfItem>, Arc<Error>>>>>,
}

impl PdfStore {
    pub fn local(worktree_store: Entity<WorktreeStore>, cx: &mut Context<Self>) -> Self {
        log::info!("Local PDF store created");
        let this = cx.weak_entity();
        Self {
            state: Box::new(cx.new(|cx| {
                let subscription =
                    cx.subscribe(&worktree_store, |this: &mut LocalPdfStore, _, event, cx| {
                        log::info!("trying to subscribe");
                        if let WorktreeStoreEvent::WorktreeAdded(worktree) = event {
                            log::info!("subscribed to worktree");
                            this.subscribe_to_worktree(worktree, cx);
                        }
                    });

                let mut local_store = LocalPdfStore {
                    local_pdf_ids_by_path: Default::default(),
                    local_pdf_ids_by_entry_id: Default::default(),
                    pdf_store: this,
                    _subscription: subscription,
                };

                // Catch up on existing worktrees that were already open
                let worktrees = worktree_store.read(cx).worktrees().collect::<Vec<_>>();
                for worktree in worktrees {
                    log::info!("Subscribing to existing worktree during init");
                    local_store.subscribe_to_worktree(&worktree, cx);
                }

                local_store
            })),
            opened_pdfs: Default::default(),
            loading_pdfs_by_path: Default::default(),
            worktree_store,
        }
    }

    pub fn pdfs(&self) -> impl '_ + Iterator<Item = Entity<PdfItem>> {
        self.opened_pdfs.values().filter_map(|pdf| pdf.upgrade())
    }

    pub fn get(&self, pdf_id: PdfId) -> Option<Entity<PdfItem>> {
        self.opened_pdfs.get(&pdf_id).and_then(|pdf| pdf.upgrade())
    }

    pub fn get_by_path(&self, path: &ProjectPath, cx: &App) -> Option<Entity<PdfItem>> {
        self.pdfs()
            .find(|pdf| &pdf.read(cx).project_path(cx) == path)
    }

    pub fn open_pdf(
        &mut self,
        project_path: ProjectPath,
        cx: &mut Context<Self>,
    ) -> Task<Result<Entity<PdfItem>>> {
        let existing_pdf = self.get_by_path(&project_path, cx);
        if let Some(existing_pdf) = existing_pdf {
            return Task::ready(Ok(existing_pdf));
        }

        let Some(worktree) = self
            .worktree_store
            .read(cx)
            .worktree_for_id(project_path.worktree_id, cx)
        else {
            return Task::ready(Err(anyhow!("no such worktree")));
        };

        let loading_watch = match self.loading_pdfs_by_path.entry(project_path.clone()) {
            // If the given path is already being loaded, then wait for that existing
            // task to complete and return the same pdf.
            hash_map::Entry::Occupied(e) => e.get().clone(),

            // Otherwise, record the fact that this path is now being loaded.
            hash_map::Entry::Vacant(entry) => {
                let (mut tx, rx) = postage::watch::channel();
                entry.insert(rx.clone());

                let load_pdf = self.state.open_pdf(project_path.path.clone(), worktree, cx);

                cx.spawn(async move |this, cx| {
                    let load_result = load_pdf.await;
                    *tx.borrow_mut() = Some(this.update(cx, |this, _cx| {
                        // Record the fact that the pdf is no longer loading.
                        this.loading_pdfs_by_path.remove(&project_path);
                        let pdf = load_result.map_err(Arc::new)?;
                        Ok(pdf)
                    })?);
                    anyhow::Ok(())
                })
                .detach();
                rx
            }
        };

        cx.background_spawn(async move {
            Self::wait_for_loading_pdf(loading_watch)
                .await
                .map_err(|e| e.cloned())
        })
    }

    pub async fn wait_for_loading_pdf(
        mut receiver: postage::watch::Receiver<Option<Result<Entity<PdfItem>, Arc<Error>>>>,
    ) -> Result<Entity<PdfItem>, Arc<Error>> {
        loop {
            if let Some(result) = receiver.borrow().as_ref() {
                match result {
                    Ok(pdf) => return Ok(pdf.to_owned()),
                    Err(e) => return Err(e.to_owned()),
                }
            }
            receiver.next().await;
        }
    }

    pub fn reload_pdfs(
        &self,
        pdfs: HashSet<Entity<PdfItem>>,
        cx: &mut Context<PdfStore>,
    ) -> Task<Result<()>> {
        let location = std::panic::Location::caller();
        log::info!(
            "Called PdfStore::reload_pdfs from {}:{}",
            location.file(),
            location.line()
        );
        if pdfs.is_empty() {
            return Task::ready(Ok(()));
        }

        self.state.reload_pdfs(pdfs, cx)
    }

    fn add_pdf(&mut self, pdf: Entity<PdfItem>, cx: &mut Context<PdfStore>) -> Result<()> {
        let pdf_id = pdf.read(cx).id;
        self.opened_pdfs.insert(pdf_id, pdf.downgrade());
        cx.subscribe(&pdf, Self::on_pdf_event).detach();
        cx.emit(PdfStoreEvent::PdfAdded(pdf));
        Ok(())
    }

    fn on_pdf_event(&mut self, pdf: Entity<PdfItem>, event: &PdfItemEvent, cx: &mut Context<Self>) {
        log::info!("Got PDF event: {:?}", event);
        if let PdfItemEvent::FileHandleChanged = event
            && let Some(local) = self.state.as_local()
        {
            local.update(cx, |local, cx| {
                local.pdf_changed_file(pdf, cx);
            })
        }
    }

}

impl PdfStoreImpl for Entity<LocalPdfStore> {
    fn open_pdf(
        &self,
        path: Arc<RelPath>,
        worktree: Entity<Worktree>,
        cx: &mut Context<PdfStore>,
    ) -> Task<Result<Entity<PdfItem>>> {
        let this = self.clone();

        let load_file = worktree.update(cx, |worktree, cx| {
            worktree.load_binary_file(path.as_ref(), cx)
        });
        cx.spawn(async move |pdf_store, cx| {
            let LoadedBinaryFile { file, content } = load_file.await?;
            let pdf = create_gpui_pdf(content, None).await?;

            let entity = cx.new(|cx| PdfItem {
                id: cx.entity_id().as_non_zero_u64().into(),
                file: file.clone(),
                pdf,
                stale_pdf: None,
                reload_task: None,
            });

            let pdf_id = cx.read_entity(&entity, |model, _| model.id);

            this.update(cx, |this, cx| {
                pdf_store.update(cx, |pdf_store, cx| pdf_store.add_pdf(entity.clone(), cx))??;
                this.local_pdf_ids_by_path.insert(
                    ProjectPath {
                        worktree_id: file.worktree_id(cx),
                        path: file.path.clone(),
                    },
                    pdf_id,
                );

                if let Some(entry_id) = file.entry_id {
                    this.local_pdf_ids_by_entry_id.insert(entry_id, pdf_id);
                }

                anyhow::Ok(())
            })?;

            Ok(entity)
        })
    }

    fn reload_pdfs(
        &self,
        pdfs: HashSet<Entity<PdfItem>>,
        cx: &mut Context<PdfStore>,
    ) -> Task<Result<()>> {
        let location = std::panic::Location::caller();
        log::info!(
            "Called LocalPdfStore::reload_pdfs from {}:{}",
            location.file(),
            location.line()
        );
        cx.spawn(async move |_, cx| {
            for pdf in pdfs {
                if let Some(rec) = pdf.update(cx, |pdf, cx| pdf.reload(cx)) {
                    rec.await?
                }
            }
            Ok(())
        })
    }

    fn as_local(&self) -> Option<Entity<LocalPdfStore>> {
        Some(self.clone())
    }
}

impl LocalPdfStore {
    fn subscribe_to_worktree(&mut self, worktree: &Entity<Worktree>, cx: &mut Context<Self>) {
        cx.subscribe(worktree, |this, worktree, event, cx| {
            if worktree.read(cx).is_local()
                && let worktree::Event::UpdatedEntries(changes) = event
            {
                this.local_worktree_entries_changed(&worktree, changes, cx);
            }
        })
        .detach();
    }

    fn local_worktree_entries_changed(
        &mut self,
        worktree_handle: &Entity<Worktree>,
        changes: &[(Arc<RelPath>, ProjectEntryId, PathChange)],
        cx: &mut Context<Self>,
    ) {
        let snapshot = worktree_handle.read(cx).snapshot();
        for (path, entry_id, _) in changes {
            self.local_worktree_entry_changed(*entry_id, path, worktree_handle, &snapshot, cx);
        }
    }

    fn local_worktree_entry_changed(
        &mut self,
        entry_id: ProjectEntryId,
        path: &Arc<RelPath>,
        worktree: &Entity<worktree::Worktree>,
        snapshot: &worktree::Snapshot,
        cx: &mut Context<Self>,
    ) -> Option<()> {
        let project_path = ProjectPath {
            worktree_id: snapshot.id(),
            path: path.clone(),
        };
        let pdf_id = match self.local_pdf_ids_by_entry_id.get(&entry_id) {
            Some(&pdf_id) => pdf_id,
            None => self.local_pdf_ids_by_path.get(&project_path).copied()?,
        };

        let pdf = self
            .pdf_store
            .update(cx, |pdf_store, _| {
                if let Some(pdf) = pdf_store.get(pdf_id) {
                    Some(pdf)
                } else {
                    pdf_store.opened_pdfs.remove(&pdf_id);
                    None
                }
            })
            .ok()
            .flatten();
        let pdf = if let Some(pdf) = pdf {
            pdf
        } else {
            self.local_pdf_ids_by_path.remove(&project_path);
            self.local_pdf_ids_by_entry_id.remove(&entry_id);
            return None;
        };

        pdf.update(cx, |pdf, cx| {
            let old_file = &pdf.file;
            if old_file.worktree != *worktree {
                return;
            }

            let snapshot_entry = old_file
                .entry_id
                .and_then(|entry_id| snapshot.entry_for_id(entry_id))
                .or_else(|| snapshot.entry_for_path(old_file.path.as_ref()));

            let new_file = if let Some(entry) = snapshot_entry {
                worktree::File {
                    disk_state: match entry.mtime {
                        Some(mtime) => DiskState::Present {
                            mtime,
                            size: entry.size,
                        },
                        None => old_file.disk_state,
                    },
                    is_local: true,
                    entry_id: Some(entry.id),
                    path: entry.path.clone(),
                    worktree: worktree.clone(),
                    is_private: entry.is_private,
                }
            } else {
                worktree::File {
                    disk_state: DiskState::Deleted,
                    is_local: true,
                    entry_id: old_file.entry_id,
                    path: old_file.path.clone(),
                    worktree: worktree.clone(),
                    is_private: old_file.is_private,
                }
            };

            if new_file == **old_file {
                return;
            }

            if new_file.path != old_file.path {
                self.local_pdf_ids_by_path.remove(&ProjectPath {
                    path: old_file.path.clone(),
                    worktree_id: old_file.worktree_id(cx),
                });
                self.local_pdf_ids_by_path.insert(
                    ProjectPath {
                        worktree_id: new_file.worktree_id(cx),
                        path: new_file.path.clone(),
                    },
                    pdf_id,
                );
            }

            if new_file.entry_id != old_file.entry_id {
                if let Some(entry_id) = old_file.entry_id {
                    self.local_pdf_ids_by_entry_id.remove(&entry_id);
                }
                if let Some(entry_id) = new_file.entry_id {
                    self.local_pdf_ids_by_entry_id.insert(entry_id, pdf_id);
                }
            }

            pdf.file_updated(Arc::new(new_file), cx);
        });
        None
    }

    fn pdf_changed_file(&mut self, pdf: Entity<PdfItem>, cx: &mut App) -> Option<()> {
        let pdf = pdf.read(cx);
        let file = &pdf.file;

        let pdf_id = pdf.id;
        if let Some(entry_id) = file.entry_id {
            match self.local_pdf_ids_by_entry_id.get(&entry_id) {
                Some(_) => {
                    return None;
                }
                None => {
                    self.local_pdf_ids_by_entry_id.insert(entry_id, pdf_id);
                }
            }
        };
        self.local_pdf_ids_by_path.insert(
            ProjectPath {
                worktree_id: file.worktree_id(cx),
                path: file.path.clone(),
            },
            pdf_id,
        );

        Some(())
    }
}

async fn create_gpui_pdf(content: Vec<u8>, password: Option<String>) -> Result<Arc<Pdf>> {
    Ok(Arc::new(Pdf::from_bytes(content, None, password).await))
}

pub fn open_pdf(
    project: Entity<Project>,
    path: impl Into<ProjectPath>,
    cx: &mut App,
) -> Task<Result<Entity<PdfItem>>> {
    if project.read(cx).is_disconnected(cx) {
        return Task::ready(Err(anyhow!(ErrorCode::Disconnected)));
    }

    let (pdf_store, _pdf_project) = GlobalPdfManager::global(&project, cx);

    let open_pdf_task = pdf_store.update(cx, |pdf_store, cx| pdf_store.open_pdf(path.into(), cx));

    cx.spawn(async move |_cx| {
        let pdf_item = open_pdf_task.await?;

        Ok(pdf_item)
    })
}

pub struct PdfProject {
    pdf_store: Entity<PdfStore>,
}

impl PdfProject {
    pub fn new(pdf_store: Entity<PdfStore>, cx: &mut App) -> Entity<Self> {
        let project = cx.new(|_| PdfProject {
            pdf_store: pdf_store.clone(),
        });

        let project_clone = project.clone();
        cx.subscribe(&pdf_store.clone(), move |_, event, cx| {
            project_clone.update(cx, |this, cx| {
                this.on_pdf_store_event(pdf_store.clone(), event, cx);
            });
        })
        .detach();
        project
    }

    /// Whether this project is from collab (not counting remote servers).
    #[inline]
    pub fn is_via_collab(&self) -> bool {
        false
        // match &self.client_state {
        //     ProjectClientState::Local | ProjectClientState::Shared { .. } => false,
        //     ProjectClientState::Collab { .. } => true,
        // }
    }

    fn on_pdf_store_event(
        &mut self,
        _: Entity<PdfStore>,
        event: &PdfStoreEvent,
        cx: &mut Context<Self>,
    ) {
        match event {
            PdfStoreEvent::PdfAdded(pdf) => {
                cx.subscribe(pdf, |this, pdf, event, cx| {
                    this.on_pdf_event(pdf, event, cx);
                })
                .detach();
            }
        }
    }

    fn on_pdf_event(
        &mut self,
        pdf: Entity<PdfItem>,
        event: &PdfItemEvent,
        cx: &mut Context<Self>,
    ) -> Option<()> {
        // TODO: handle image events from remote
        if let PdfItemEvent::ReloadNeeded = event
            && !self.is_via_collab()
        {
            self.reload_pdfs([pdf].into_iter().collect(), cx)
                .detach_and_log_err(cx);
        }

        None
    }

    pub fn reload_pdfs(
        &self,
        pdfs: HashSet<Entity<PdfItem>>,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        self.pdf_store
            .update(cx, |pdf_store, cx| pdf_store.reload_pdfs(pdfs, cx))
    }
}

struct GlobalPdfManager(HashMap<EntityId, (Entity<PdfStore>, Entity<PdfProject>)>);
impl Global for GlobalPdfManager {}

impl GlobalPdfManager {
    pub fn global(
        project: &Entity<Project>,
        cx: &mut App,
    ) -> (Entity<PdfStore>, Entity<PdfProject>) {
        if !cx.has_global::<GlobalPdfManager>() {
            cx.set_global(GlobalPdfManager(Default::default()));
        }

        let project_id = project.entity_id();

        if let Some(managed) = cx.global::<GlobalPdfManager>().0.get(&project_id) {
            return managed.clone();
        }

        let worktree_store = project.read(cx).worktree_store().clone();
        let pdf_store = cx.new(|cx| PdfStore::local(worktree_store, cx));
        let pdf_project = PdfProject::new(pdf_store.clone(), cx);

        cx.global_mut::<GlobalPdfManager>()
            .0
            .insert(project_id, (pdf_store.clone(), pdf_project.clone()));

        (pdf_store, pdf_project)
    }
}
