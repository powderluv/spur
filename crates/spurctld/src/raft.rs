// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raft-based consensus for spurctld.
//!
//! Raft is always-on: even single-node deployments run a 1-member Raft
//! cluster that self-elects instantly.  The Raft log is the sole durable
//! store — entries are `WalOperation` values proposed via
//! `ClusterManager::propose()` and applied through `StateMachineApply`.

use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openraft::storage::{LogState, RaftLogReader, Snapshot, SnapshotMeta};
use openraft::{
    BasicNode, Config, Entry, EntryPayload, LogId, Raft, RaftSnapshotBuilder, StorageError,
    StoredMembership, Vote,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use spur_core::wal::WalOperation;

pub type NodeId = u64;

openraft::declare_raft_types!(
    pub SpurTypeConfig:
        D = WalOperation,
        R = ClientResponse,
        Node = BasicNode,
);

pub type SpurRaft = Raft<SpurTypeConfig>;

/// Maximum size of a single Raft gRPC message (request or response), in bytes.
///
/// tonic defaults both encode and decode limits to 4 MiB. Snapshots ship in
/// chunks of `RAFT_SNAPSHOT_MAX_CHUNK_SIZE` raw bytes, and `serde_json` inflates
/// the chunk's `Vec<u8>` payload roughly 3.4x (each byte becomes a decimal
/// integer plus a comma), so the largest message we ever put on the wire is
/// about `3.4 * RAFT_SNAPSHOT_MAX_CHUNK_SIZE`. This limit keeps large headroom
/// over that bound. Invariant: `4 * RAFT_SNAPSHOT_MAX_CHUNK_SIZE < RAFT_MAX_MESSAGE_SIZE`.
pub const RAFT_MAX_MESSAGE_SIZE: usize = 32 * 1024 * 1024;

/// Raw bytes per `install_snapshot` chunk. Bounded so the JSON-encoded message
/// stays well under `RAFT_MAX_MESSAGE_SIZE` (see the invariant above). A larger
/// snapshot simply ships in more chunks.
pub const RAFT_SNAPSHOT_MAX_CHUNK_SIZE: u64 = 1024 * 1024;

/// Build a Raft gRPC client with message-size limits raised to
/// `RAFT_MAX_MESSAGE_SIZE`. Used by both the live network path and tests so the
/// exact transport configuration is covered.
pub(crate) fn raft_client(
    channel: tonic::transport::Channel,
) -> spur_proto::raft_proto::raft_internal_client::RaftInternalClient<tonic::transport::Channel> {
    spur_proto::raft_proto::raft_internal_client::RaftInternalClient::new(channel)
        .max_decoding_message_size(RAFT_MAX_MESSAGE_SIZE)
        .max_encoding_message_size(RAFT_MAX_MESSAGE_SIZE)
}

/// Set when a committed WAL entry transitions a job to a terminal state.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct JobFinalized {
    pub job_id: u32,
    pub state: spur_core::job::JobState,
    pub exit_code: i32,
}

/// Response returned after a Raft write is committed.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClientResponse {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub jobs_finalized: Vec<JobFinalized>,
    #[serde(default)]
    pub reservation_created: bool,
    #[serde(default)]
    pub partition_created: bool,
}

/// Trait for applying committed Raft entries to the cluster state.
/// Implemented by ClusterManager to avoid a circular dependency with SpurStore.
pub trait StateMachineApply: Send + Sync {
    fn apply_operation(&self, op: &WalOperation) -> ClientResponse;
    fn snapshot_state(&self) -> Result<Vec<u8>, anyhow::Error>;
    fn restore_from_snapshot(&self, data: &[u8]) -> Result<(), anyhow::Error>;
}

/// Disk-backed Raft storage.
/// Layout: `{state_dir}/raft/{vote.json, log/*.json, snapshot.json}`
pub struct SpurStore {
    inner: RwLock<StoreInner>,
    raft_dir: PathBuf,
    applier: Arc<dyn StateMachineApply>,
}

impl std::fmt::Debug for SpurStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpurStore")
            .field("raft_dir", &self.raft_dir)
            .finish()
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct StoreInner {
    vote: Option<Vote<NodeId>>,
    committed: Option<LogId<NodeId>>,
    last_purged: Option<LogId<NodeId>>,
    log: BTreeMap<u64, Entry<SpurTypeConfig>>,
    last_applied: Option<LogId<NodeId>>,
    last_membership: StoredMembership<NodeId, BasicNode>,
    applied_count: u64,
    /// Highest log index present on disk at startup. Everything up to it is
    /// replayed, not new, thus its side effects must not be reported as
    /// events that happen now. See [`WAL_REPLAY_SPAN`].
    #[serde(skip)]
    replay_upto: Option<u64>,
}

/// Name of the span that wraps every replayed WAL operation. `spurctld` filters
/// INFO and below inside it, because a restart re-applies the whole log: without
/// this, one node removal is reported once per controller process, and a reader
/// who greps the journal chases a fault that did not happen. WARN and ERROR stay,
/// because a bad transition during a replay is a real defect.
pub const WAL_REPLAY_SPAN: &str = "wal_replay";

#[derive(Debug, Serialize, Deserialize)]
struct PersistedSnapshot {
    meta: SnapshotMeta<NodeId, BasicNode>,
    data: Vec<u8>,
}

impl SpurStore {
    /// Recover storage with `strict = true`: any unreadable/undeserializable vote,
    /// log entry, or snapshot is a hard error. This is the safe default — see
    /// `new_with_recovery_mode` for why silently skipping such records is
    /// dangerous and should only ever be an explicit, deliberate choice.
    pub fn new(state_dir: &Path, applier: Arc<dyn StateMachineApply>) -> anyhow::Result<Self> {
        Self::new_with_recovery_mode(state_dir, applier, true)
    }

    /// Recover storage from `{state_dir}/raft/`.
    ///
    /// `strict` controls what happens when a vote, log entry, or snapshot record
    /// exists but cannot be deserialized (disk corruption, or a restart with a
    /// `spurctld` build whose `WalOperation`/state schema has drifted from
    /// whatever wrote the record). These are records of already-committed
    /// cluster state — nodes, jobs, reservations — so losing one is a
    /// correctness violation, not a cosmetic issue.
    ///
    /// `strict = true` (default, via `new`) refuses to start: the affected
    /// record is named in the error so an operator can roll back to a
    /// compatible binary. `strict = false` preserves the old behavior of
    /// logging and skipping the record, for deliberate forensic recovery
    /// when the loss has already been assessed as acceptable.
    pub fn new_with_recovery_mode(
        state_dir: &Path,
        applier: Arc<dyn StateMachineApply>,
        strict: bool,
    ) -> anyhow::Result<Self> {
        let raft_dir = state_dir.join("raft");
        let log_dir = raft_dir.join("log");
        std::fs::create_dir_all(&log_dir)?;

        let mut inner = StoreInner::default();
        let mut skipped_records = 0u64;

        let vote_path = raft_dir.join("vote.json");
        if vote_path.exists() {
            match std::fs::read_to_string(&vote_path) {
                Ok(data) => match serde_json::from_str(&data) {
                    Ok(v) => inner.vote = Some(v),
                    Err(e) => {
                        if strict {
                            anyhow::bail!("failed to parse vote.json: {e}");
                        }
                        warn!("failed to parse vote.json: {e}");
                        skipped_records += 1;
                    }
                },
                Err(e) => {
                    if strict {
                        anyhow::bail!("failed to read vote.json: {e}");
                    }
                    warn!("failed to read vote.json: {e}");
                    skipped_records += 1;
                }
            }
        }

        if let Ok(entries) = std::fs::read_dir(&log_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "json") {
                    match std::fs::read_to_string(&path) {
                        Ok(data) => match serde_json::from_str::<Entry<SpurTypeConfig>>(&data) {
                            Ok(e) => {
                                inner.log.insert(e.log_id.index, e);
                            }
                            Err(e) => {
                                if strict {
                                    anyhow::bail!("failed to parse log entry {path:?}: {e}");
                                }
                                warn!("failed to parse log entry {:?}: {e}", path);
                                skipped_records += 1;
                            }
                        },
                        Err(e) => {
                            if strict {
                                anyhow::bail!("failed to read log entry {path:?}: {e}");
                            }
                            warn!("failed to read log entry {:?}: {e}", path);
                            skipped_records += 1;
                        }
                    }
                }
            }
        }

        let snap_path = raft_dir.join("snapshot.json");
        if snap_path.exists() {
            match std::fs::read_to_string(&snap_path) {
                Ok(data) => match serde_json::from_str::<PersistedSnapshot>(&data) {
                    Ok(ps) => match applier.restore_from_snapshot(&ps.data) {
                        Ok(()) => {
                            inner.last_applied = ps.meta.last_log_id;
                            inner.last_membership = ps.meta.last_membership.clone();
                        }
                        Err(e) => {
                            if strict {
                                anyhow::bail!("failed to restore state from snapshot.json: {e}");
                            }
                            warn!("failed to restore state from snapshot.json: {e}");
                            skipped_records += 1;
                        }
                    },
                    Err(e) => {
                        if strict {
                            anyhow::bail!("failed to parse snapshot.json: {e}");
                        }
                        warn!("failed to parse snapshot.json: {e}");
                        skipped_records += 1;
                    }
                },
                Err(e) => {
                    if strict {
                        anyhow::bail!("failed to read snapshot.json: {e}");
                    }
                    warn!("failed to read snapshot.json: {e}");
                    skipped_records += 1;
                }
            }
        }

        let purged_path = raft_dir.join("purged.json");
        if purged_path.exists() {
            // Always hard-fail: a corrupt purged.json cannot be recovered safely
            // even in lenient mode, since it would silently un-purge history.
            let data = std::fs::read_to_string(&purged_path)?;
            inner.last_purged = Some(
                serde_json::from_str::<LogId<NodeId>>(&data)
                    .map_err(|e| anyhow::anyhow!("failed to parse purged.json: {e}"))?,
            );
        }

        inner.replay_upto = inner.log.keys().next_back().copied();

        info!(
            log_entries = inner.log.len(),
            vote = ?inner.vote,
            replay_upto = ?inner.replay_upto,
            "raft store recovered from disk"
        );
        if skipped_records > 0 {
            tracing::error!(
                skipped_records,
                "raft store recovered with GAPS: {skipped_records} record(s) could not be \
                 deserialized and were dropped (lenient recovery mode) — cluster state may be \
                 missing nodes, jobs, or other committed changes"
            );
        }

        Ok(Self {
            inner: RwLock::new(inner),
            raft_dir,
            applier,
        })
    }

    fn persist_last_purged(&self, log_id: &LogId<NodeId>) -> Result<(), std::io::Error> {
        let path = self.raft_dir.join("purged.json");
        let tmp = self.raft_dir.join("purged.json.tmp");
        let data = serde_json::to_vec(log_id)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        // Write-then-rename: a crash mid-write must never corrupt purged.json.
        std::fs::write(&tmp, &data)
            .map_err(|e| std::io::Error::new(e.kind(), format!("{tmp:?}: {e}")))?;
        std::fs::rename(&tmp, &path)
            .map_err(|e| std::io::Error::new(e.kind(), format!("{path:?}: {e}")))
    }

    fn persist_vote(&self, vote: &Vote<NodeId>) -> Result<(), std::io::Error> {
        let path = self.raft_dir.join("vote.json");
        let data = serde_json::to_vec(vote)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(&path, &data)
            .map_err(|e| std::io::Error::new(e.kind(), format!("{path:?}: {e}")))
    }

    fn persist_log_entry(&self, entry: &Entry<SpurTypeConfig>) -> Result<(), std::io::Error> {
        let path = self
            .raft_dir
            .join("log")
            .join(format!("{:020}.json", entry.log_id.index));
        let data = serde_json::to_vec(entry)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(&path, &data)
            .map_err(|e| std::io::Error::new(e.kind(), format!("{path:?}: {e}")))
    }

    fn remove_log_entry(&self, index: u64) {
        let path = self
            .raft_dir
            .join("log")
            .join(format!("{:020}.json", index));
        let _ = std::fs::remove_file(&path);
    }

    fn persist_snapshot(
        &self,
        meta: &SnapshotMeta<NodeId, BasicNode>,
        data: &[u8],
    ) -> Result<(), std::io::Error> {
        let ps = PersistedSnapshot {
            meta: meta.clone(),
            data: data.to_vec(),
        };
        let path = self.raft_dir.join("snapshot.json");
        let json = serde_json::to_vec(&ps)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(&path, &json)
            .map_err(|e| std::io::Error::new(e.kind(), format!("{path:?}: {e}")))
    }

    fn load_persisted_snapshot(&self) -> Option<PersistedSnapshot> {
        let path = self.raft_dir.join("snapshot.json");
        if !path.exists() {
            return None;
        }
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|data| serde_json::from_str(&data).ok())
    }
}

impl RaftLogReader<SpurTypeConfig> for Arc<SpurStore> {
    async fn try_get_log_entries<RB: std::ops::RangeBounds<u64> + Clone + std::fmt::Debug>(
        &mut self,
        range: RB,
    ) -> Result<Vec<Entry<SpurTypeConfig>>, StorageError<NodeId>> {
        let inner = self.inner.read();
        Ok(inner.log.range(range).map(|(_, e)| e.clone()).collect())
    }
}

impl RaftSnapshotBuilder<SpurTypeConfig> for Arc<SpurStore> {
    async fn build_snapshot(&mut self) -> Result<Snapshot<SpurTypeConfig>, StorageError<NodeId>> {
        let inner = self.inner.read();

        let snapshot_data = self.applier.snapshot_state().map_err(|e| {
            StorageError::from_io_error(
                openraft::ErrorSubject::Store,
                openraft::ErrorVerb::Read,
                std::io::Error::other(e),
            )
        })?;

        let snap_id = format!(
            "{}-{}",
            inner.last_applied.map(|l| l.index).unwrap_or(0),
            inner.applied_count
        );

        let meta = SnapshotMeta {
            last_log_id: inner.last_applied,
            last_membership: inner.last_membership.clone(),
            snapshot_id: snap_id,
        };

        self.persist_snapshot(&meta, &snapshot_data).map_err(|e| {
            StorageError::from_io_error(
                openraft::ErrorSubject::Store,
                openraft::ErrorVerb::Write,
                e,
            )
        })?;

        Ok(Snapshot {
            meta,
            snapshot: Box::new(Cursor::new(snapshot_data)),
        })
    }
}

impl openraft::RaftStorage<SpurTypeConfig> for Arc<SpurStore> {
    type LogReader = Self;
    type SnapshotBuilder = Self;

    async fn get_log_state(&mut self) -> Result<LogState<SpurTypeConfig>, StorageError<NodeId>> {
        let inner = self.inner.read();
        let last = inner.log.iter().next_back().map(|(_, e)| e.log_id);
        Ok(LogState {
            last_purged_log_id: inner.last_purged,
            last_log_id: last,
        })
    }

    async fn get_log_reader(&mut self) -> Self::LogReader {
        self.clone()
    }

    async fn save_vote(&mut self, vote: &Vote<NodeId>) -> Result<(), StorageError<NodeId>> {
        self.persist_vote(vote).map_err(|e| {
            StorageError::from_io_error(openraft::ErrorSubject::Vote, openraft::ErrorVerb::Write, e)
        })?;
        self.inner.write().vote = Some(*vote);
        Ok(())
    }

    async fn read_vote(&mut self) -> Result<Option<Vote<NodeId>>, StorageError<NodeId>> {
        Ok(self.inner.read().vote)
    }

    async fn append_to_log<I: IntoIterator<Item = Entry<SpurTypeConfig>> + Send>(
        &mut self,
        entries: I,
    ) -> Result<(), StorageError<NodeId>> {
        let mut inner = self.inner.write();
        for entry in entries {
            self.persist_log_entry(&entry).map_err(|e| {
                StorageError::from_io_error(
                    openraft::ErrorSubject::Logs,
                    openraft::ErrorVerb::Write,
                    e,
                )
            })?;
            inner.log.insert(entry.log_id.index, entry);
        }
        Ok(())
    }

    async fn delete_conflict_logs_since(
        &mut self,
        log_id: LogId<NodeId>,
    ) -> Result<(), StorageError<NodeId>> {
        let mut inner = self.inner.write();
        let keys: Vec<_> = inner.log.range(log_id.index..).map(|(k, _)| *k).collect();
        for k in keys {
            inner.log.remove(&k);
            self.remove_log_entry(k);
        }
        Ok(())
    }

    async fn purge_logs_upto(&mut self, log_id: LogId<NodeId>) -> Result<(), StorageError<NodeId>> {
        self.persist_last_purged(&log_id).map_err(|e| {
            StorageError::from_io_error(openraft::ErrorSubject::Logs, openraft::ErrorVerb::Write, e)
        })?;
        let mut inner = self.inner.write();
        let keys: Vec<_> = inner.log.range(..=log_id.index).map(|(k, _)| *k).collect();
        for k in keys {
            inner.log.remove(&k);
            self.remove_log_entry(k);
        }
        inner.last_purged = Some(log_id);
        Ok(())
    }

    async fn last_applied_state(
        &mut self,
    ) -> Result<(Option<LogId<NodeId>>, StoredMembership<NodeId, BasicNode>), StorageError<NodeId>>
    {
        let inner = self.inner.read();
        Ok((inner.last_applied, inner.last_membership.clone()))
    }

    async fn apply_to_state_machine(
        &mut self,
        entries: &[Entry<SpurTypeConfig>],
    ) -> Result<Vec<ClientResponse>, StorageError<NodeId>> {
        let mut inner = self.inner.write();
        let mut results = Vec::new();

        for entry in entries {
            inner.last_applied = Some(entry.log_id);
            match &entry.payload {
                EntryPayload::Normal(op) => {
                    debug!(index = entry.log_id.index, "raft: applying WalOperation");
                    inner.applied_count += 1;
                    // An entry that was already on disk when this process started is
                    // a replay of history, not something happening now.
                    let replaying = inner
                        .replay_upto
                        .is_some_and(|upto| entry.log_id.index <= upto);
                    if replaying {
                        let span = tracing::info_span!(WAL_REPLAY_SPAN, index = entry.log_id.index);
                        let _enter = span.enter();
                        results.push(self.applier.apply_operation(op));
                    } else {
                        results.push(self.applier.apply_operation(op));
                    }
                }
                EntryPayload::Membership(mem) => {
                    inner.last_membership = StoredMembership::new(Some(entry.log_id), mem.clone());
                    results.push(ClientResponse::default());
                }
                EntryPayload::Blank => {
                    results.push(ClientResponse::default());
                }
            }
        }
        Ok(results)
    }

    async fn get_snapshot_builder(&mut self) -> Self::SnapshotBuilder {
        self.clone()
    }

    async fn begin_receiving_snapshot(
        &mut self,
    ) -> Result<Box<Cursor<Vec<u8>>>, StorageError<NodeId>> {
        Ok(Box::new(Cursor::new(Vec::new())))
    }

    async fn install_snapshot(
        &mut self,
        meta: &SnapshotMeta<NodeId, BasicNode>,
        snapshot: Box<Cursor<Vec<u8>>>,
    ) -> Result<(), StorageError<NodeId>> {
        let data = snapshot.into_inner();

        // Unlike startup recovery (SpurStore::new), there is no lenient option
        // here: this is a live snapshot pushed by the current Raft leader, not
        // a one-time operator recovery decision. A follower that can't apply it
        // must reject it rather than silently mark itself caught up while
        // actually missing whatever the leader just sent. Validate before
        // persisting: writing an unparseable snapshot to disk first would make
        // the next (strict-by-default) startup hard-fail permanently on it.
        self.applier.restore_from_snapshot(&data).map_err(|e| {
            StorageError::from_io_error(
                openraft::ErrorSubject::Store,
                openraft::ErrorVerb::Write,
                std::io::Error::other(e),
            )
        })?;

        self.persist_snapshot(meta, &data).map_err(|e| {
            StorageError::from_io_error(
                openraft::ErrorSubject::Store,
                openraft::ErrorVerb::Write,
                e,
            )
        })?;

        let mut inner = self.inner.write();
        inner.last_applied = meta.last_log_id;
        inner.last_membership = meta.last_membership.clone();
        info!(last_applied = ?meta.last_log_id, "installed snapshot from leader");
        Ok(())
    }

    async fn get_current_snapshot(
        &mut self,
    ) -> Result<Option<Snapshot<SpurTypeConfig>>, StorageError<NodeId>> {
        if let Some(ps) = self.load_persisted_snapshot() {
            Ok(Some(Snapshot {
                meta: ps.meta,
                snapshot: Box::new(Cursor::new(ps.data)),
            }))
        } else {
            Ok(None)
        }
    }
}

/// Raft network factory — connects to peers on the dedicated Raft port.
pub struct SpurNetwork {
    peers: BTreeMap<NodeId, String>,
}

impl openraft::RaftNetworkFactory<SpurTypeConfig> for Arc<SpurNetwork> {
    type Network = SpurNetworkConnection;

    async fn new_client(&mut self, target: NodeId, _node: &BasicNode) -> Self::Network {
        let addr = self
            .peers
            .get(&target)
            .cloned()
            .unwrap_or_else(|| format!("unknown-{}", target));
        SpurNetworkConnection {
            target,
            addr,
            client: None,
        }
    }
}

/// Connection to a single Raft peer.
pub struct SpurNetworkConnection {
    #[allow(dead_code)]
    target: NodeId,
    addr: String,
    client: Option<
        spur_proto::raft_proto::raft_internal_client::RaftInternalClient<tonic::transport::Channel>,
    >,
}

impl SpurNetworkConnection {
    async fn get_client(
        &mut self,
    ) -> Result<
        &mut spur_proto::raft_proto::raft_internal_client::RaftInternalClient<
            tonic::transport::Channel,
        >,
        openraft::error::RPCError<NodeId, BasicNode, openraft::error::RaftError<NodeId>>,
    > {
        if self.client.is_none() {
            let url = if self.addr.starts_with("http") {
                self.addr.clone()
            } else {
                format!("http://{}", self.addr)
            };
            let channel = tonic::transport::Channel::from_shared(url)
                .map_err(|e| {
                    openraft::error::RPCError::Unreachable(openraft::error::Unreachable::new(
                        &std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()),
                    ))
                })?
                .connect_timeout(std::time::Duration::from_secs(2))
                .timeout(std::time::Duration::from_secs(5))
                .connect()
                .await
                .map_err(|e| {
                    openraft::error::RPCError::Unreachable(openraft::error::Unreachable::new(
                        &std::io::Error::new(std::io::ErrorKind::ConnectionRefused, e.to_string()),
                    ))
                })?;
            self.client = Some(raft_client(channel));
        }
        Ok(self.client.as_mut().unwrap())
    }
}

impl openraft::RaftNetwork<SpurTypeConfig> for SpurNetworkConnection {
    async fn append_entries(
        &mut self,
        rpc: openraft::raft::AppendEntriesRequest<SpurTypeConfig>,
        _option: openraft::network::RPCOption,
    ) -> Result<
        openraft::raft::AppendEntriesResponse<NodeId>,
        openraft::error::RPCError<NodeId, BasicNode, openraft::error::RaftError<NodeId>>,
    > {
        let payload = serde_json::to_vec(&rpc).map_err(|e| {
            openraft::error::RPCError::Unreachable(openraft::error::Unreachable::new(
                &std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
            ))
        })?;

        if payload.len() > RAFT_MAX_MESSAGE_SIZE {
            let hint = (rpc.entries.len() as u64 / 2).max(1);
            return Err(openraft::error::RPCError::PayloadTooLarge(
                openraft::error::PayloadTooLarge::new_entries_hint(hint),
            ));
        }

        let client = self.get_client().await?;
        let resp = client
            .append_entries(spur_proto::raft_proto::RaftRequest { payload })
            .await
            .map_err(|e| {
                self.client = None;
                openraft::error::RPCError::Unreachable(openraft::error::Unreachable::new(
                    &std::io::Error::new(std::io::ErrorKind::ConnectionRefused, e.to_string()),
                ))
            })?;

        serde_json::from_slice(&resp.into_inner().payload).map_err(|e| {
            openraft::error::RPCError::Unreachable(openraft::error::Unreachable::new(
                &std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
            ))
        })
    }

    async fn install_snapshot(
        &mut self,
        rpc: openraft::raft::InstallSnapshotRequest<SpurTypeConfig>,
        _option: openraft::network::RPCOption,
    ) -> Result<
        openraft::raft::InstallSnapshotResponse<NodeId>,
        openraft::error::RPCError<
            NodeId,
            BasicNode,
            openraft::error::RaftError<NodeId, openraft::error::InstallSnapshotError>,
        >,
    > {
        let payload = serde_json::to_vec(&rpc).map_err(|e| {
            openraft::error::RPCError::Unreachable(openraft::error::Unreachable::new(
                &std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
            ))
        })?;

        let client = self.get_client().await.map_err(|e| match e {
            openraft::error::RPCError::Unreachable(u) => openraft::error::RPCError::Unreachable(u),
            _ => openraft::error::RPCError::Unreachable(openraft::error::Unreachable::new(
                &std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "connection failed"),
            )),
        })?;

        let resp = client
            .install_snapshot(spur_proto::raft_proto::RaftRequest { payload })
            .await
            .map_err(|e| {
                self.client = None;
                openraft::error::RPCError::Unreachable(openraft::error::Unreachable::new(
                    &std::io::Error::new(std::io::ErrorKind::ConnectionRefused, e.to_string()),
                ))
            })?;

        serde_json::from_slice(&resp.into_inner().payload).map_err(|e| {
            openraft::error::RPCError::Unreachable(openraft::error::Unreachable::new(
                &std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
            ))
        })
    }

    async fn vote(
        &mut self,
        rpc: openraft::raft::VoteRequest<NodeId>,
        _option: openraft::network::RPCOption,
    ) -> Result<
        openraft::raft::VoteResponse<NodeId>,
        openraft::error::RPCError<NodeId, BasicNode, openraft::error::RaftError<NodeId>>,
    > {
        let payload = serde_json::to_vec(&rpc).map_err(|e| {
            openraft::error::RPCError::Unreachable(openraft::error::Unreachable::new(
                &std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
            ))
        })?;

        let client = self.get_client().await?;
        let resp = client
            .vote(spur_proto::raft_proto::RaftRequest { payload })
            .await
            .map_err(|e| {
                self.client = None;
                openraft::error::RPCError::Unreachable(openraft::error::Unreachable::new(
                    &std::io::Error::new(std::io::ErrorKind::ConnectionRefused, e.to_string()),
                ))
            })?;

        serde_json::from_slice(&resp.into_inner().payload).map_err(|e| {
            openraft::error::RPCError::Unreachable(openraft::error::Unreachable::new(
                &std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
            ))
        })
    }
}

/// Handle to the running Raft node — exposes leadership queries.
pub struct RaftHandle {
    pub raft: SpurRaft,
    pub node_id: NodeId,
    pub peers: BTreeMap<NodeId, String>,
}

impl RaftHandle {
    /// Cheap, locally-cached leadership check. Can be stale for the length of
    /// an election timeout after a partition — callers that need a
    /// consensus-backed answer (e.g. before writes that bypass Raft, like
    /// accounting reconciliation) must use `ensure_leader` instead.
    pub fn is_leader(&self) -> bool {
        let metrics = self.raft.metrics().borrow().clone();
        metrics.current_leader == Some(self.node_id)
    }

    /// Consensus-backed leadership check: confirms leadership by exchanging
    /// heartbeats with a quorum of followers, so a partitioned former leader
    /// gets `false` rather than a stale cached `true`. Costs a network
    /// round-trip; use for periodic/batch operations, not on every request.
    pub async fn ensure_leader(&self) -> bool {
        self.raft.ensure_linearizable().await.is_ok()
    }

    pub fn current_leader(&self) -> Option<NodeId> {
        self.raft.metrics().borrow().current_leader
    }
}

/// The system hostname as a UTF-8 string.
pub fn system_hostname() -> anyhow::Result<String> {
    let name = hostname::get().map_err(|e| anyhow::anyhow!("failed to read hostname: {e}"))?;
    Ok(name.to_string_lossy().into_owned())
}

/// How a node's Raft id was determined, for startup diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeIdSource {
    Explicit,
    PeersPosition,
    HostnameOrdinal,
}

impl std::fmt::Display for NodeIdSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            NodeIdSource::Explicit => "explicit controller.node_id",
            NodeIdSource::PeersPosition => "position in controller.peers",
            NodeIdSource::HostnameOrdinal => "hostname ordinal",
        };
        f.write_str(s)
    }
}

/// Resolve this node's Raft id (multi-node; `peers` non-empty). Precedence:
/// explicit -> position in `peers` -> hostname ordinal. Validated to be in
/// `1..=peers.len()`, else fails fast (an out-of-range id splits brain).
pub fn resolve_node_id(
    explicit: Option<u64>,
    hostname: &str,
    peers: &[String],
) -> anyhow::Result<(u64, NodeIdSource)> {
    let n = peers.len() as u64;

    if let Some(id) = explicit {
        if !(1..=n).contains(&id) {
            anyhow::bail!(
                "controller.node_id = {id} is out of range for {n} configured \
                 controller.peers (must be 1..={n})"
            );
        }
        return Ok((id, NodeIdSource::Explicit));
    }

    if let Some(id) = node_id_from_peers(hostname, peers)? {
        return Ok((id, NodeIdSource::PeersPosition));
    }

    // Ordinal fallback only fits IP-only peers (no hostname to match). With
    // hostname peers a no-match is a misconfig: fail fast, don't guess an id.
    if !all_peers_ip_only(peers) {
        anyhow::bail!(
            "this host ({hostname:?}) matched no entry in controller.peers; fix the \
             hostname to match its peers entry or set controller.node_id in spur.conf"
        );
    }

    // Accept the ordinal only if in range; out-of-range is a misconfig.
    if let Some(id) = node_id_from_hostname(hostname) {
        if (1..=n).contains(&id) {
            return Ok((id, NodeIdSource::HostnameOrdinal));
        }
        anyhow::bail!(
            "hostname {hostname:?} yields Raft node_id {id} via ordinal fallback, \
             out of range for {n} configured controller.peers; fix the hostname \
             to match its controller.peers entry or set controller.node_id"
        );
    }

    anyhow::bail!(
        "controller.peers are IP-only and hostname {hostname:?} has no ordinal to \
         derive a node_id from; set controller.node_id in spur.conf"
    )
}

/// True when every peer host is an IP literal (no hostname to match against).
fn all_peers_ip_only(peers: &[String]) -> bool {
    peers
        .iter()
        .all(|entry| peer_host(entry).parse::<std::net::IpAddr>().is_ok())
}

/// Parse a node_id from a hostname string. The ordinal after the last '-'
/// is treated as a 0-based index and converted to a 1-based node_id.
pub fn node_id_from_hostname(hostname: &str) -> Option<u64> {
    let ordinal: u64 = hostname.rsplit('-').next()?.parse().ok()?;
    Some(ordinal + 1)
}

/// Node id (index + 1) of the `peers` entry matching `hostname`, on the full
/// name or first DNS label (so `spurctld-0` matches `spurctld-0.ns.svc...`).
/// `Ok(None)` if none match; errors if several do (guessing risks split-brain).
pub fn node_id_from_peers(hostname: &str, peers: &[String]) -> anyhow::Result<Option<u64>> {
    let matches: Vec<u64> = peers
        .iter()
        .enumerate()
        .filter(|(_, entry)| host_matches(hostname, peer_host(entry)))
        .map(|(i, _)| i as u64 + 1)
        .collect();
    match matches.as_slice() {
        [] => Ok(None),
        [id] => Ok(Some(*id)),
        ids => anyhow::bail!(
            "hostname {hostname:?} matches multiple controller.peers entries \
             (node ids {ids:?}); set controller.node_id explicitly"
        ),
    }
}

/// Extract the host part from a `host:port` peer entry, handling bracketed IPv6
/// (`[::1]:6821`) and bare hosts.
fn peer_host(entry: &str) -> &str {
    let entry = entry.trim();
    if let Some(rest) = entry.strip_prefix('[') {
        return rest.split(']').next().unwrap_or(rest);
    }
    // One ':' is "host:port"; more colons are an unbracketed IPv6 literal (no port).
    match entry.rsplit_once(':') {
        Some((host, _)) if !host.contains(':') => host,
        _ => entry,
    }
}

/// True if `hostname` matches `peer_host` in full or on the first DNS label.
fn host_matches(hostname: &str, peer_host: &str) -> bool {
    if hostname.eq_ignore_ascii_case(peer_host) {
        return true;
    }
    first_label(hostname).eq_ignore_ascii_case(first_label(peer_host))
}

/// The segment of a hostname before the first '.' (its first DNS label).
fn first_label(host: &str) -> &str {
    host.split('.').next().unwrap_or(host)
}

/// Guard against a shifted node identity across restarts: the id is re-derived
/// each boot but persisted Raft state belongs to the old id, so a reordered or
/// renamed peer would orphan committed entries. Persist first, then verify.
fn persist_or_verify_node_id(state_dir: &Path, node_id: NodeId) -> anyhow::Result<()> {
    let raft_dir = state_dir.join("raft");
    std::fs::create_dir_all(&raft_dir)?;
    let path = raft_dir.join("node_id");

    match std::fs::read_to_string(&path) {
        Ok(contents) => {
            let prev: NodeId = contents.trim().parse().map_err(|_| {
                anyhow::anyhow!(
                    "failed to parse persisted node_id {:?} at {}",
                    contents.trim(),
                    path.display()
                )
            })?;
            if prev != node_id {
                anyhow::bail!(
                    "resolved Raft node_id {node_id} differs from persisted id {prev} at {}; \
                     the controller.peers ordering or this host's name likely changed. \
                     Running under a new id would orphan committed Raft state. Restore the \
                     previous identity, or wipe the raft state dir to rejoin as a fresh member.",
                    path.display()
                );
            }
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let tmp = raft_dir.join("node_id.tmp");
            std::fs::write(&tmp, node_id.to_string())?;
            std::fs::rename(&tmp, &path)?;
            Ok(())
        }
        Err(e) => Err(anyhow::anyhow!("failed to read {}: {e}", path.display())),
    }
}

/// Build a 1-indexed peer map from the config peers list.
pub fn build_peer_map(peers: &[String]) -> BTreeMap<NodeId, String> {
    peers
        .iter()
        .enumerate()
        .map(|(i, addr)| (i as u64 + 1, addr.clone()))
        .collect()
}

/// Test-only convenience wrapper around `start_raft_with_recovery_mode` with
/// strict WAL/snapshot recovery (see `SpurStore::new`). Production startup
/// (main.rs) calls `start_raft_with_recovery_mode` directly since it needs to
/// thread through the `--allow-partial-wal-recovery` CLI flag.
#[cfg(test)]
pub async fn start_raft(
    node_id: NodeId,
    peers: &[String],
    state_dir: &Path,
    applier: Arc<dyn StateMachineApply>,
) -> anyhow::Result<RaftHandle> {
    start_raft_with_recovery_mode(node_id, peers, state_dir, applier, true).await
}

pub async fn start_raft_with_recovery_mode(
    node_id: NodeId,
    peers: &[String],
    state_dir: &Path,
    applier: Arc<dyn StateMachineApply>,
    strict_wal_recovery: bool,
) -> anyhow::Result<RaftHandle> {
    let config = Config {
        heartbeat_interval: 500,
        election_timeout_min: 1500,
        election_timeout_max: 3000,
        snapshot_max_chunk_size: RAFT_SNAPSHOT_MAX_CHUNK_SIZE,
        ..Default::default()
    };
    let config = Arc::new(config.validate().map_err(|e| anyhow::anyhow!("{e}"))?);

    persist_or_verify_node_id(state_dir, node_id)?;

    let store = Arc::new(SpurStore::new_with_recovery_mode(
        state_dir,
        applier,
        strict_wal_recovery,
    )?);
    let peer_map = build_peer_map(peers);
    let network = Arc::new(SpurNetwork {
        peers: peer_map.clone(),
    });

    let (log_store, state_machine) = openraft::storage::Adaptor::new(store.clone());
    let raft = Raft::new(node_id, config, network, log_store, state_machine).await?;

    // Symmetric bootstrap only applies on first-ever startup; skip it once we
    // have prior state, else openraft logs an ERROR rejecting it every restart.
    let already_initialized = {
        let inner = store.inner.read();
        inner.vote.is_some() || !inner.log.is_empty() || inner.last_applied.is_some()
    };
    if already_initialized {
        debug!("raft store has prior state; skipping redundant initialize()");
    } else {
        let members: BTreeMap<NodeId, BasicNode> = peer_map
            .iter()
            .map(|(id, addr)| (*id, BasicNode::new(addr.clone())))
            .collect();
        if let Err(e) = raft.initialize(members).await {
            debug!("raft initialize: {e} (already initialized)");
        }
    }

    info!(node_id, peers = ?peer_map, "raft node started");
    Ok(RaftHandle {
        raft,
        node_id,
        peers: peer_map,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    struct NoopApplier;
    impl StateMachineApply for NoopApplier {
        fn apply_operation(&self, _op: &WalOperation) -> ClientResponse {
            ClientResponse::default()
        }
        fn snapshot_state(&self) -> Result<Vec<u8>, anyhow::Error> {
            Ok(Vec::new())
        }
        fn restore_from_snapshot(&self, _data: &[u8]) -> Result<(), anyhow::Error> {
            Ok(())
        }
    }

    fn noop_applier() -> Arc<dyn StateMachineApply> {
        Arc::new(NoopApplier)
    }

    #[test]
    fn peer_map_empty() {
        let m = build_peer_map(&[]);
        assert!(m.is_empty());
    }

    #[test]
    fn peer_map_single() {
        let m = build_peer_map(&["host1:6821".into()]);
        assert_eq!(m.len(), 1);
        assert_eq!(m[&1], "host1:6821");
    }

    #[test]
    fn peer_map_three_nodes() {
        let m = build_peer_map(&["a:6821".into(), "b:6821".into(), "c:6821".into()]);
        assert_eq!(m.len(), 3);
        assert_eq!(m[&1], "a:6821");
        assert_eq!(m[&2], "b:6821");
        assert_eq!(m[&3], "c:6821");
    }

    #[test]
    fn hostname_spurctld_0() {
        assert_eq!(node_id_from_hostname("spurctld-0"), Some(1));
    }

    #[test]
    fn hostname_spurctld_2() {
        assert_eq!(node_id_from_hostname("spurctld-2"), Some(3));
    }

    #[test]
    fn hostname_custom_prefix() {
        assert_eq!(node_id_from_hostname("my-cluster-ctrl-5"), Some(6));
    }

    #[test]
    fn hostname_no_dash() {
        assert_eq!(node_id_from_hostname("localhost"), None);
    }

    #[test]
    fn hostname_non_numeric_suffix() {
        assert_eq!(node_id_from_hostname("ctrl-abc"), None);
    }

    #[test]
    fn peers_position_mapping() {
        let peers = vec![
            "hostA:6821".to_string(),
            "hostB:6821".to_string(),
            "hostC:6821".to_string(),
        ];
        assert_eq!(node_id_from_peers("hostA", &peers).unwrap(), Some(1));
        assert_eq!(node_id_from_peers("hostB", &peers).unwrap(), Some(2));
        assert_eq!(node_id_from_peers("hostC", &peers).unwrap(), Some(3));
    }

    #[test]
    fn peers_short_hostname_matches_fqdn_entry() {
        let peers = vec![
            "spurctld-0.ns.svc.cluster.local:6821".to_string(),
            "spurctld-1.ns.svc.cluster.local:6821".to_string(),
        ];
        assert_eq!(node_id_from_peers("spurctld-0", &peers).unwrap(), Some(1));
        assert_eq!(node_id_from_peers("spurctld-1", &peers).unwrap(), Some(2));
    }

    #[test]
    fn peers_fqdn_hostname_matches_short_entry() {
        let peers = vec!["node1:6821".to_string(), "node2:6821".to_string()];
        assert_eq!(
            node_id_from_peers("node2.example.com", &peers).unwrap(),
            Some(2)
        );
    }

    #[test]
    fn peers_no_match_is_none() {
        let peers = vec!["hostA:6821".to_string(), "hostB:6821".to_string()];
        assert_eq!(node_id_from_peers("hostZ", &peers).unwrap(), None);
    }

    #[test]
    fn peers_ambiguous_match_errors() {
        // Same first DNS label in two different domains: both reduce to "n1".
        let peers = vec!["n1.dc-a:6821".to_string(), "n1.dc-b:6821".to_string()];
        assert!(node_id_from_peers("n1", &peers).is_err());
    }

    #[test]
    fn peers_matches_bracketed_ipv6_entry() {
        let peers = vec!["[::1]:6821".to_string(), "hostB:6821".to_string()];
        assert_eq!(peer_host("[::1]:6821"), "::1");
        assert_eq!(node_id_from_peers("hostB", &peers).unwrap(), Some(2));
    }

    #[test]
    fn peer_host_parsing() {
        assert_eq!(peer_host("host:6821"), "host");
        assert_eq!(peer_host("host"), "host");
        assert_eq!(peer_host("host.example.com:6821"), "host.example.com");
        assert_eq!(peer_host("[2001:db8::1]:6821"), "2001:db8::1");
    }

    fn three_peers() -> Vec<String> {
        vec![
            "hostA:6821".to_string(),
            "hostB:6821".to_string(),
            "hostC:6821".to_string(),
        ]
    }

    #[test]
    fn resolve_explicit_wins() {
        let (id, source) = resolve_node_id(Some(2), "hostA", &three_peers()).unwrap();
        assert_eq!(id, 2);
        assert_eq!(source, NodeIdSource::Explicit);
    }

    #[test]
    fn resolve_explicit_out_of_range_errors() {
        assert!(resolve_node_id(Some(9), "hostA", &three_peers()).is_err());
        assert!(resolve_node_id(Some(0), "hostA", &three_peers()).is_err());
    }

    #[test]
    fn resolve_peers_position() {
        let (id, source) = resolve_node_id(None, "hostC", &three_peers()).unwrap();
        assert_eq!(id, 3);
        assert_eq!(source, NodeIdSource::PeersPosition);
    }

    #[test]
    fn resolve_hostname_ordinal_fallback_for_ip_peers() {
        // IP-only peers can't match a hostname; the ordinal fallback applies.
        let peers = vec![
            "10.0.0.1:6821".to_string(),
            "10.0.0.2:6821".to_string(),
            "10.0.0.3:6821".to_string(),
        ];
        let (id, source) = resolve_node_id(None, "spurctld-1", &peers).unwrap();
        assert_eq!(id, 2);
        assert_eq!(source, NodeIdSource::HostnameOrdinal);
    }

    #[test]
    fn resolve_ordinal_out_of_range_errors() {
        // spurctld-5 -> ordinal id 6, but only 3 IP-only peers exist.
        let peers = vec![
            "10.0.0.1:6821".to_string(),
            "10.0.0.2:6821".to_string(),
            "10.0.0.3:6821".to_string(),
        ];
        assert!(resolve_node_id(None, "spurctld-5", &peers).is_err());
    }

    #[test]
    fn resolve_no_match_errors() {
        assert!(resolve_node_id(None, "unknownhost", &three_peers()).is_err());
    }

    #[test]
    fn resolve_hostname_peers_no_match_does_not_fall_back_to_ordinal() {
        // spurctld-1 has a valid in-range ordinal (2), but hostname peers with
        // no match must fail fast, not fall back to it.
        let err = resolve_node_id(None, "spurctld-1", &three_peers()).unwrap_err();
        assert!(
            err.to_string().contains("matched no entry"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn resolve_ip_only_peers_without_ordinal_errors() {
        let peers = vec!["10.0.0.1:6821".to_string(), "10.0.0.2:6821".to_string()];
        assert!(resolve_node_id(None, "plainhost", &peers).is_err());
    }

    #[test]
    fn node_id_persists_on_first_boot() {
        let dir = TempDir::new().unwrap();
        persist_or_verify_node_id(dir.path(), 2).unwrap();
        let persisted = std::fs::read_to_string(dir.path().join("raft").join("node_id")).unwrap();
        assert_eq!(persisted.trim(), "2");
    }

    #[test]
    fn node_id_same_across_restart_is_ok() {
        let dir = TempDir::new().unwrap();
        persist_or_verify_node_id(dir.path(), 3).unwrap();
        persist_or_verify_node_id(dir.path(), 3).unwrap();
    }

    #[test]
    fn node_id_shift_across_restart_fails_fast() {
        let dir = TempDir::new().unwrap();
        persist_or_verify_node_id(dir.path(), 1).unwrap();
        let err = persist_or_verify_node_id(dir.path(), 2).unwrap_err();
        assert!(
            err.to_string().contains("differs from persisted id"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn resolve_ambiguous_match_errors() {
        let peers = vec!["n1.dc-a:6821".to_string(), "n1.dc-b:6821".to_string()];
        assert!(resolve_node_id(None, "n1", &peers).is_err());
    }

    #[test]
    fn store_persists_vote() {
        let dir = TempDir::new().unwrap();
        let store = SpurStore::new(dir.path(), noop_applier()).unwrap();

        let vote = Vote {
            leader_id: openraft::LeaderId {
                term: 5,
                node_id: 2,
            },
            committed: true,
        };
        store.persist_vote(&vote).unwrap();

        let store2 = SpurStore::new(dir.path(), noop_applier()).unwrap();
        let inner = store2.inner.read();
        let recovered = inner.vote.as_ref().unwrap();
        assert_eq!(recovered.leader_id.term, 5);
        assert_eq!(recovered.leader_id.node_id, 2);
        assert!(recovered.committed);
    }

    #[test]
    fn store_persists_log_entries() {
        let dir = TempDir::new().unwrap();
        let store = SpurStore::new(dir.path(), noop_applier()).unwrap();

        let entry = Entry {
            log_id: LogId {
                leader_id: openraft::LeaderId {
                    term: 1,
                    node_id: 1,
                },
                index: 42,
            },
            payload: EntryPayload::Blank,
        };
        store.persist_log_entry(&entry).unwrap();

        let store2 = SpurStore::new(dir.path(), noop_applier()).unwrap();
        let inner = store2.inner.read();
        assert!(inner.log.contains_key(&42));
        assert_eq!(inner.log[&42].log_id.index, 42);
    }

    #[test]
    fn store_removes_log_entries() {
        let dir = TempDir::new().unwrap();
        let store = SpurStore::new(dir.path(), noop_applier()).unwrap();

        for idx in 0..5 {
            let entry = Entry {
                log_id: LogId {
                    leader_id: openraft::LeaderId {
                        term: 1,
                        node_id: 1,
                    },
                    index: idx,
                },
                payload: EntryPayload::Blank,
            };
            store.persist_log_entry(&entry).unwrap();
            store.inner.write().log.insert(idx, entry);
        }

        store.remove_log_entry(2);
        store.remove_log_entry(3);

        let store2 = SpurStore::new(dir.path(), noop_applier()).unwrap();
        let inner = store2.inner.read();
        assert_eq!(inner.log.len(), 3); // 0, 1, 4
        assert!(inner.log.contains_key(&0));
        assert!(inner.log.contains_key(&1));
        assert!(!inner.log.contains_key(&2));
        assert!(!inner.log.contains_key(&3));
        assert!(inner.log.contains_key(&4));
    }

    #[test]
    fn store_persists_snapshot() {
        let dir = TempDir::new().unwrap();
        let store = SpurStore::new(dir.path(), noop_applier()).unwrap();

        let meta = SnapshotMeta {
            last_log_id: Some(LogId {
                leader_id: openraft::LeaderId {
                    term: 3,
                    node_id: 1,
                },
                index: 100,
            }),
            last_membership: StoredMembership::default(),
            snapshot_id: "snap-100".into(),
        };
        let data = b"test snapshot data";
        store.persist_snapshot(&meta, data).unwrap();

        let loaded = store.load_persisted_snapshot().unwrap();
        assert_eq!(loaded.meta.snapshot_id, "snap-100");
        assert_eq!(loaded.data, data);
    }

    #[test]
    fn store_empty_dir_recovery() {
        let dir = TempDir::new().unwrap();
        let store = SpurStore::new(dir.path(), noop_applier()).unwrap();

        let inner = store.inner.read();
        assert!(inner.vote.is_none());
        assert!(inner.log.is_empty());
        assert!(inner.last_applied.is_none());
    }

    #[tokio::test]
    async fn store_persists_last_purged_across_restart() {
        use openraft::RaftStorage;
        let dir = TempDir::new().unwrap();
        let log_id = LogId {
            leader_id: openraft::LeaderId {
                term: 7,
                node_id: 1,
            },
            index: 9999,
        };

        // Exercise the real public purge path, not the internal helper.
        {
            let mut store = Arc::new(SpurStore::new(dir.path(), noop_applier()).unwrap());
            store.purge_logs_upto(log_id).await.unwrap();
        }

        // Simulate restart: reconstruct from the same dir and verify via get_log_state.
        let mut store2 = Arc::new(SpurStore::new(dir.path(), noop_applier()).unwrap());
        let state = store2.get_log_state().await.unwrap();
        assert_eq!(state.last_purged_log_id, Some(log_id));
    }

    /// Everything on disk at startup is history, thus its log events must be
    /// marked as a replay. A fresh store replays nothing.
    #[test]
    fn replay_upto_is_the_highest_index_on_disk() {
        let dir = TempDir::new().unwrap();
        {
            let store = SpurStore::new(dir.path(), noop_applier()).unwrap();
            assert_eq!(store.inner.read().replay_upto, None, "fresh store");
            for index in [3u64, 17, 9] {
                store
                    .persist_log_entry(&Entry {
                        log_id: LogId {
                            leader_id: openraft::LeaderId {
                                term: 1,
                                node_id: 1,
                            },
                            index,
                        },
                        payload: EntryPayload::Blank,
                    })
                    .unwrap();
            }
        }

        let restarted = SpurStore::new(dir.path(), noop_applier()).unwrap();
        assert_eq!(restarted.inner.read().replay_upto, Some(17));
    }

    #[test]
    fn store_no_purged_file_recovers_as_none() {
        let dir = TempDir::new().unwrap();
        let store = SpurStore::new(dir.path(), noop_applier()).unwrap();
        let inner = store.inner.read();
        assert!(inner.last_purged.is_none());
    }

    #[test]
    fn store_corrupt_purged_file_hard_fails() {
        let dir = TempDir::new().unwrap();
        // Create the store once so the raft/ subdir exists, then corrupt purged.json.
        SpurStore::new(dir.path(), noop_applier()).unwrap();
        std::fs::write(
            dir.path().join("raft").join("purged.json"),
            "not valid json",
        )
        .unwrap();

        assert!(SpurStore::new(dir.path(), noop_applier()).is_err());
    }

    /// Mirrors a real restart with a schema-incompatible binary: a WAL entry
    /// whose payload variant this build's `WalOperation` enum doesn't know
    /// about (e.g. written by a newer/older `spurctld`).
    fn write_raw_log_entry(dir: &std::path::Path, index: u64, json: &str) {
        std::fs::write(
            dir.join("raft")
                .join("log")
                .join(format!("{index:020}.json")),
            json,
        )
        .unwrap();
    }

    const UNKNOWN_VARIANT_ENTRY: &str = r#"{"log_id":{"leader_id":{"term":1,"node_id":1},"index":0},"payload":{"Normal":{"SomeFutureVariant":{}}}}"#;

    #[test]
    fn store_unknown_wal_variant_hard_fails_by_default() {
        let dir = TempDir::new().unwrap();
        SpurStore::new(dir.path(), noop_applier()).unwrap();
        write_raw_log_entry(dir.path(), 0, UNKNOWN_VARIANT_ENTRY);

        assert!(SpurStore::new(dir.path(), noop_applier()).is_err());
    }

    #[test]
    fn store_unknown_wal_variant_lenient_mode_skips_and_continues() {
        let dir = TempDir::new().unwrap();
        SpurStore::new(dir.path(), noop_applier()).unwrap();
        write_raw_log_entry(dir.path(), 0, UNKNOWN_VARIANT_ENTRY);

        let store = SpurStore::new_with_recovery_mode(dir.path(), noop_applier(), false).unwrap();
        assert!(store.inner.read().log.is_empty());
    }

    #[test]
    fn store_corrupt_snapshot_hard_fails_by_default() {
        let dir = TempDir::new().unwrap();
        SpurStore::new(dir.path(), noop_applier()).unwrap();
        std::fs::write(
            dir.path().join("raft").join("snapshot.json"),
            "not valid json",
        )
        .unwrap();

        assert!(SpurStore::new(dir.path(), noop_applier()).is_err());
    }

    #[test]
    fn store_corrupt_snapshot_lenient_mode_skips_and_continues() {
        let dir = TempDir::new().unwrap();
        SpurStore::new(dir.path(), noop_applier()).unwrap();
        std::fs::write(
            dir.path().join("raft").join("snapshot.json"),
            "not valid json",
        )
        .unwrap();

        assert!(SpurStore::new_with_recovery_mode(dir.path(), noop_applier(), false).is_ok());
    }

    #[test]
    fn store_corrupt_vote_file_hard_fails_by_default() {
        let dir = TempDir::new().unwrap();
        SpurStore::new(dir.path(), noop_applier()).unwrap();
        std::fs::write(dir.path().join("raft").join("vote.json"), "not valid json").unwrap();

        assert!(SpurStore::new(dir.path(), noop_applier()).is_err());
    }

    #[test]
    fn store_corrupt_vote_file_lenient_mode_skips_and_continues() {
        let dir = TempDir::new().unwrap();
        SpurStore::new(dir.path(), noop_applier()).unwrap();
        std::fs::write(dir.path().join("raft").join("vote.json"), "not valid json").unwrap();

        let store = SpurStore::new_with_recovery_mode(dir.path(), noop_applier(), false).unwrap();
        assert!(store.inner.read().vote.is_none());
    }

    struct FailingRestoreApplier;
    impl StateMachineApply for FailingRestoreApplier {
        fn apply_operation(&self, _op: &WalOperation) -> ClientResponse {
            ClientResponse::default()
        }
        fn snapshot_state(&self) -> Result<Vec<u8>, anyhow::Error> {
            Ok(Vec::new())
        }
        fn restore_from_snapshot(&self, _data: &[u8]) -> Result<(), anyhow::Error> {
            Err(anyhow::anyhow!("applier rejected snapshot payload"))
        }
    }

    #[test]
    fn store_snapshot_restore_failure_hard_fails_by_default() {
        let dir = TempDir::new().unwrap();
        let store = SpurStore::new(dir.path(), noop_applier()).unwrap();
        let meta = SnapshotMeta {
            last_log_id: Some(LogId {
                leader_id: openraft::LeaderId {
                    term: 1,
                    node_id: 1,
                },
                index: 5,
            }),
            last_membership: StoredMembership::default(),
            snapshot_id: "snap-5".into(),
        };
        store.persist_snapshot(&meta, b"opaque state").unwrap();

        assert!(SpurStore::new(dir.path(), Arc::new(FailingRestoreApplier)).is_err());
    }

    #[test]
    fn store_snapshot_restore_failure_lenient_mode_skips_and_continues() {
        let dir = TempDir::new().unwrap();
        let store = SpurStore::new(dir.path(), noop_applier()).unwrap();
        let meta = SnapshotMeta {
            last_log_id: Some(LogId {
                leader_id: openraft::LeaderId {
                    term: 1,
                    node_id: 1,
                },
                index: 5,
            }),
            last_membership: StoredMembership::default(),
            snapshot_id: "snap-5".into(),
        };
        store.persist_snapshot(&meta, b"opaque state").unwrap();

        let store =
            SpurStore::new_with_recovery_mode(dir.path(), Arc::new(FailingRestoreApplier), false)
                .unwrap();
        // A rejected restore must not advance last_applied/last_membership —
        // otherwise Raft believes the state machine is caught up to the
        // snapshot when it never actually restored anything.
        let inner = store.inner.read();
        assert!(inner.last_applied.is_none());
        assert_eq!(inner.last_membership, StoredMembership::default());
    }

    #[tokio::test]
    async fn start_raft_skips_redundant_initialize_on_restart() {
        use std::time::Duration;

        let dir = TempDir::new().unwrap();

        let handle = start_raft(1, &["[::1]:0".into()], dir.path(), noop_applier())
            .await
            .unwrap();
        handle
            .raft
            .wait(Some(Duration::from_secs(5)))
            .metrics(|m| m.current_leader == Some(1), "leader elected")
            .await
            .expect("single-node raft did not self-elect within 5s");
        handle.raft.shutdown().await.unwrap();

        // Restart against the same state_dir: prior vote/log on disk means
        // already_initialized is true, so initialize() must be skipped, yet
        // the node must still become leader and be usable.
        let handle = start_raft(1, &["[::1]:0".into()], dir.path(), noop_applier())
            .await
            .unwrap();
        handle
            .raft
            .wait(Some(Duration::from_secs(5)))
            .metrics(|m| m.current_leader == Some(1), "leader elected")
            .await
            .expect("restarted single-node raft did not resume leadership within 5s");
    }

    /// The HA follower-catch-up path: a leader pushes a snapshot via the Raft
    /// `install_snapshot` RPC. If this node's applier can't apply it (e.g. a
    /// schema-incompatible build in a multi-controller cluster), the RPC must
    /// fail rather than mark the follower caught up on state it doesn't have.
    #[tokio::test]
    async fn install_snapshot_rejects_when_applier_cannot_restore() {
        use openraft::RaftStorage;

        let dir = TempDir::new().unwrap();
        let mut store =
            Arc::new(SpurStore::new(dir.path(), Arc::new(FailingRestoreApplier)).unwrap());

        let meta = SnapshotMeta {
            last_log_id: Some(LogId {
                leader_id: openraft::LeaderId {
                    term: 1,
                    node_id: 2,
                },
                index: 10,
            }),
            last_membership: StoredMembership::default(),
            snapshot_id: "snap-10".into(),
        };

        let result = store
            .install_snapshot(&meta, Box::new(Cursor::new(b"opaque state".to_vec())))
            .await;

        assert!(result.is_err());
        // A rejected snapshot must never reach disk: persisting it first would
        // make the next (strict-by-default) startup hard-fail on it forever.
        assert!(!dir.path().join("raft").join("snapshot.json").exists());
    }

    #[tokio::test]
    async fn append_entries_rejects_oversized_batch_with_payload_too_large() {
        use openraft::network::RPCOption;
        use openraft::RaftNetwork;
        use spur_core::job::JobSpec;

        let mut conn = SpurNetworkConnection {
            target: 99,
            addr: "http://127.0.0.1:1".into(),
            client: None,
        };

        let big_spec = JobSpec {
            script: Some("x".repeat(2 * 1024 * 1024)),
            ..Default::default()
        };
        let entry = Entry::<SpurTypeConfig> {
            log_id: LogId {
                leader_id: openraft::LeaderId {
                    term: 1,
                    node_id: 1,
                },
                index: 1,
            },
            payload: EntryPayload::Normal(WalOperation::JobSubmit {
                job_id: 1,
                spec: Box::new(big_spec),
            }),
        };

        let rpc = openraft::raft::AppendEntriesRequest::<SpurTypeConfig> {
            vote: Vote::new(1, 1),
            prev_log_id: None,
            entries: vec![entry; 20],
            leader_commit: None,
        };

        let option = RPCOption::new(std::time::Duration::from_secs(5));
        let result = conn.append_entries(rpc, option).await;

        let err = result.expect_err("oversized batch should be rejected");
        match err {
            openraft::error::RPCError::PayloadTooLarge(too_large) => {
                assert_eq!(too_large.entries_hint(), 10);
            }
            other => panic!("expected PayloadTooLarge, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn append_entries_accepts_small_batch() {
        use openraft::network::RPCOption;
        use openraft::RaftNetwork;

        let mut conn = SpurNetworkConnection {
            target: 99,
            addr: "http://127.0.0.1:1".into(),
            client: None,
        };

        let entry = Entry::<SpurTypeConfig> {
            log_id: LogId {
                leader_id: openraft::LeaderId {
                    term: 1,
                    node_id: 1,
                },
                index: 1,
            },
            payload: EntryPayload::Blank,
        };

        let rpc = openraft::raft::AppendEntriesRequest::<SpurTypeConfig> {
            vote: Vote::new(1, 1),
            prev_log_id: None,
            entries: vec![entry],
            leader_commit: None,
        };

        let option = RPCOption::new(std::time::Duration::from_secs(5));
        let result = conn.append_entries(rpc, option).await;

        // Small batch passes the size guard but fails to connect (bogus address).
        // The error should be Unreachable (connection failure), NOT PayloadTooLarge.
        let err = result.expect_err("should fail to connect to bogus address");
        assert!(
            matches!(err, openraft::error::RPCError::Unreachable(_)),
            "expected Unreachable, got: {err:?}"
        );
    }
}
